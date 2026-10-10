//! Initial, deliberately simple row layout for the multi-stark backend.
//!
//! Each arithmetic gate, public binding, and table query occupies one row.
//! Advice width is max(3, largest table arity). Coefficients, cell labels,
//! wiring permutation, public indices and table IDs are preprocessed columns.
//!
//! For each logical value, its cell occurrences form a fixed cycle. Every
//! cell pushes `(namespace, COPY, own_label, value)` and pulls
//! `(namespace, COPY, next_label, value)`. Each label occurs exactly once
//! on each side, forcing the value to agree along the cycle. No prover-chosen
//! wiring labels or copy multiplicities are permitted.
//!
//! Public bindings pull `(namespace, PUBLIC, index, value)`. Index zero is an
//! unconditional zero-valued anchor: its required external claim prevents
//! deactivating the entire computation even when there are no user publics.
//! Fixed tables have their own traces of witness multiplicities; padding
//! repeats a real table row rather than admitting additional tuples.
//! Partitioned computation traces retain global copy labels and additionally
//! pull one `(namespace, ACTIVE=3, partition)` claim each, selected by a fixed
//! indicator column. The verifier's claims require every partition to be active.

use std::fmt;

use crate::traits::Field;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;

use crate::expr::Expr;
use crate::lookup::Lookup;
use crate::system::CircuitInputs;

use super::{Assignment, Circuit, Value, Witness, WitnessError};

mod multiplicities;
#[cfg(feature = "kzg")]
mod profile;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringError {
    SizeOverflow,
    /// The computation trace cap must be a power of two, at least two.
    InvalidTraceHeight,
    /// Fixed tables are not partitioned by this lowering.
    TableExceedsTraceHeight,
    CustomTraceExceedsHeight,
    /// Integer labels/counts must embed injectively in the prime subfield.
    FieldTooSmall,
    QuotientBudgetTooSmall,
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for LoweringError {}

fn computation_relations<F: Field>(
    namespace: F,
    width: usize,
    has_custom_traces: bool,
) -> (Expr<F>, Vec<Lookup<Expr<F>>>) {
    let main = |i| Expr::main(u32::try_from(i).expect("checked column"));
    let prep = |i| Expr::preprocessed(u32::try_from(i).expect("checked column"));
    let constant = Expr::constant;
    let a = main(0);
    let b = main(1);
    let c = main(2);
    let relation =
        prep(0) * a.clone() * b.clone() + prep(1) * a.clone() + prep(2) * b + prep(3) * c + prep(4);
    let mut lookups = Vec::new();
    for col in 0..width {
        lookups.push(Lookup::push(
            constant(F::ONE),
            vec![
                constant(namespace),
                constant(F::ZERO),
                prep(5 + col),
                main(col),
            ],
        ));
        lookups.push(Lookup::pull(
            constant(F::ONE),
            vec![
                constant(namespace),
                constant(F::ZERO),
                prep(5 + width + col),
                main(col),
            ],
        ));
    }
    let public_enable = 5 + 2 * width;
    lookups.push(Lookup::pull(
        prep(public_enable),
        vec![
            constant(namespace),
            constant(F::ONE),
            prep(public_enable + 1),
            a,
        ],
    ));
    let mut table_args = vec![
        constant(namespace),
        constant(F::TWO),
        prep(public_enable + 3),
    ];
    table_args.extend((0..width).map(main));
    lookups.push(Lookup::pull(prep(public_enable + 2), table_args));
    if has_custom_traces {
        lookups.push(Lookup::push(
            prep(public_enable + 4),
            vec![
                constant(namespace),
                constant(F::from_u8(4)),
                constant(F::from_u8(4)),
                prep(public_enable + 3),
                prep(public_enable + 1),
                main(0),
            ],
        ));
    }
    (relation, lookups)
}

fn merged_table_lookup<F: Field>(namespace: F, width: usize) -> Lookup<Expr<F>> {
    let mut args = vec![Expr::constant(namespace), Expr::constant(F::TWO)];
    args.extend((0..width).map(|i| Expr::preprocessed(u32::try_from(i).expect("checked column"))));
    Lookup::push(Expr::main(0), args)
}

/// A fixed circuit and its multi-stark row layout.
///
/// Use [`Self::circuit_inputs`] for setup, [`Self::traces`] for proving, and
/// [`Self::claims`] with independently known expected public values for
/// verification. Always include the claims, even for a circuit with no public
/// values. All of this lowering's circuits must be included in the same system.
pub struct MultiStarkCircuit<F: Field> {
    circuit: Circuit<F>,
    namespace: F,
    width: usize,
    wires: Vec<Value>,
    fixed: RowMajorMatrix<F>,
    main_heights: Vec<usize>,
    successors: Option<Vec<usize>>,
    hash_row_ends: Vec<usize>,
    hash_definitions: Vec<CircuitInputs<F>>,
    merged_table_height: Option<usize>,
}

/// Validated witness source: one computation partition per shard, followed by
/// one shard for shared tables and custom gates. Does not retain generated traces.
pub struct TraceShards<'a, F: Field> {
    compiled: &'a MultiStarkCircuit<F>,
    assignment: &'a Assignment<F>,
}

impl<F: Field> TraceShards<'_, F> {
    pub fn compiled(&self) -> &MultiStarkCircuit<F> {
        self.compiled
    }

    pub fn len(&self) -> usize {
        self.compiled.main_heights.len()
            + usize::from(self.compiled.auxiliary_widths().next().is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Generate a single canonical circuit, including an individual hash trace.
    pub fn trace(&self, index: usize) -> Result<RowMajorMatrix<F>, WitnessError> {
        let c = self.compiled;
        if index >= c.num_circuits() {
            return Err(WitnessError::ShardIndex {
                index,
                count: c.num_circuits(),
            });
        }
        if index < c.main_heights.len() {
            return Ok(c.main_trace(self.assignment, index));
        }
        let aux = index - c.main_heights.len();
        let tables = if c.merged_table_height.is_some() {
            1
        } else {
            c.circuit.tables.len()
        };
        if aux < tables {
            return Ok(c.table_traces(self.assignment).remove(aux));
        }
        Ok(c.circuit.hashes.as_ref().unwrap().trace(
            self.assignment.prepared_hash(&c.circuit)?,
            &c.hash_definitions,
            aux - tables,
        ))
    }

    pub fn traces(&self, shard: usize) -> Result<Vec<RowMajorMatrix<F>>, WitnessError> {
        if shard >= self.len() {
            return Err(WitnessError::ShardIndex {
                index: shard,
                count: self.len(),
            });
        }
        let c = self.compiled;
        let mut traces: Vec<_> = c
            .main_heights
            .iter()
            .enumerate()
            .map(|(index, _)| {
                if index == shard {
                    c.main_trace(self.assignment, index)
                } else {
                    RowMajorMatrix::new(vec![], c.width)
                }
            })
            .collect();
        if shard == c.main_heights.len() {
            traces.extend(c.auxiliary_traces(self.assignment)?);
        } else {
            traces.extend(
                c.auxiliary_widths()
                    .map(|width| RowMajorMatrix::new(vec![], width)),
            );
        }
        Ok(traces)
    }
}

/// Exact dimensions of this backend's initial row layout, without allocating
/// traces or wiring. `trace_field_bytes` includes main/table advice and fixed
/// field arrays once; it is NOT an estimate of total prover memory or RSS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiStarkLayout {
    pub used_rows: usize,
    /// Total padded computation rows, summed across all partitions.
    pub main_height: usize,
    /// Heights of individual computation traces. `main_height` is their sum,
    /// not the FFT domain size when the layout is partitioned.
    pub main_heights: Vec<usize>,
    pub main_width: usize,
    pub preprocessed_width: usize,
    pub table_heights: Vec<usize>,
    /// (height, advice width, fixed width) of additional custom-gate traces.
    pub custom_traces: Vec<(usize, usize, usize)>,
    pub advice_cells: usize,
    pub preprocessed_cells: usize,
    pub trace_field_bytes: usize,
}

impl<F: Field> Circuit<F> {
    /// Inspect and validate layout size without allocating the physical trace.
    /// Lowering uses this same calculation, including field/overflow checks.
    pub fn multi_stark_layout(&self) -> Result<MultiStarkLayout, LoweringError> {
        self.layout(None)
    }

    /// Partition computation rows into bounded, power-of-two traces. Fixed
    /// tables must already fit the cap. Copy labels remain globally unique.
    pub fn multi_stark_layout_with_max_height(
        &self,
        max_height: usize,
    ) -> Result<MultiStarkLayout, LoweringError> {
        self.layout(Some(max_height))
    }

    fn layout(&self, max_height: Option<usize>) -> Result<MultiStarkLayout, LoweringError> {
        let hash_rows = self
            .hashes
            .as_ref()
            .map_or(Some(0usize), |h| {
                h.calls.iter().try_fold(0usize, |sum, call| {
                    sum.checked_add(call.input.len())?.checked_add(32)
                })
            })
            .ok_or(LoweringError::SizeOverflow)?;
        let width = self
            .tables
            .iter()
            .map(super::TableDefinition::width)
            .max()
            .unwrap_or(0)
            .max(3);
        let used_rows = self
            .gates
            .len()
            .checked_add(self.publics.len())
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_add(self.lookups.len()))
            .and_then(|n| n.checked_add(hash_rows))
            .ok_or(LoweringError::SizeOverflow)?;
        let main_heights = partition_heights(used_rows, max_height)?;
        let height = main_heights
            .iter()
            .try_fold(0usize, |sum, &n| sum.checked_add(n))
            .ok_or(LoweringError::SizeOverflow)?;
        let cells = height
            .checked_mul(width)
            .ok_or(LoweringError::SizeOverflow)?;
        let fixed_width = width
            .checked_mul(2)
            .and_then(|n| n.checked_add(9))
            .and_then(|n| n.checked_add(usize::from(hash_rows > 0)))
            .and_then(|n| n.checked_add(usize::from(main_heights.len() > 1)))
            .ok_or(LoweringError::SizeOverflow)?;
        let fixed_len = height
            .checked_mul(fixed_width)
            .ok_or(LoweringError::SizeOverflow)?;
        if u32::try_from(width).is_err() || u32::try_from(fixed_width).is_err() {
            return Err(LoweringError::SizeOverflow);
        }
        let largest_count = cells
            .max(self.tables.len())
            .max(self.publics.len() + 1)
            .max(4);
        if !F::prime_order_exceeds(largest_count) {
            return Err(LoweringError::FieldTooSmall);
        }
        let mut table_heights = Vec::with_capacity(self.tables.len());
        let mut table_rows = 0usize;
        for table in &self.tables {
            let height = table
                .rows
                .len()
                .max(2)
                .checked_next_power_of_two()
                .ok_or(LoweringError::SizeOverflow)?;
            table_rows = table_rows
                .checked_add(height)
                .ok_or(LoweringError::SizeOverflow)?;
            if max_height.is_some_and(|cap| height > cap) {
                return Err(LoweringError::TableExceedsTraceHeight);
            }
            table_heights.push(height);
        }
        let mut advice_cells = cells
            .checked_add(table_rows)
            .ok_or(LoweringError::SizeOverflow)?;
        let mut preprocessed_cells = table_rows
            .checked_mul(width)
            .and_then(|n| n.checked_add(fixed_len))
            .ok_or(LoweringError::SizeOverflow)?;
        let custom_traces = self.hashes.as_ref().map_or_else(Vec::new, |h| h.layouts());
        for &(height, advice, fixed) in &custom_traces {
            if max_height.is_some_and(|cap| height > cap) {
                return Err(LoweringError::CustomTraceExceedsHeight);
            }
            advice_cells = advice_cells
                .checked_add(
                    height
                        .checked_mul(advice)
                        .ok_or(LoweringError::SizeOverflow)?,
                )
                .ok_or(LoweringError::SizeOverflow)?;
            preprocessed_cells = preprocessed_cells
                .checked_add(
                    height
                        .checked_mul(fixed)
                        .ok_or(LoweringError::SizeOverflow)?,
                )
                .ok_or(LoweringError::SizeOverflow)?;
        }
        let trace_field_bytes = advice_cells
            .checked_add(preprocessed_cells)
            .and_then(|n| n.checked_mul(size_of::<F>()))
            .ok_or(LoweringError::SizeOverflow)?;
        Ok(MultiStarkLayout {
            used_rows,
            main_height: height,
            main_heights,
            main_width: width,
            preprocessed_width: fixed_width,
            table_heights,
            custom_traces,
            advice_cells,
            preprocessed_cells,
            trace_field_bytes,
        })
    }

    /// Lowers to one arithmetic/wiring trace followed by one trace per fixed
    /// table. The native field remains `F` throughout.
    ///
    /// `namespace` identifies this circuit's private lookup vocabulary. When
    /// composing several lowerings or hand-authored circuits in one system,
    /// the application must give them disjoint lookup namespaces.
    ///
    /// Requires quotient degree 2 (e.g. FRI `log_blowup >= 1`) because the
    /// arithmetic selector times a multiplication has degree three.
    pub fn lower_to_multi_stark(self, namespace: F) -> Result<MultiStarkCircuit<F>, LoweringError> {
        let layout = self.multi_stark_layout()?;
        Ok(self.lower(namespace, layout))
    }

    /// Split computation rows while keeping one global copy permutation and
    /// shared fixed tables. Each computation trace gets an independent public
    /// activation claim, so a prover cannot omit disconnected trace pieces.
    /// This bounds individual domains, not total proving memory.
    pub fn lower_to_multi_stark_with_max_height(
        self,
        namespace: F,
        max_height: usize,
    ) -> Result<MultiStarkCircuit<F>, LoweringError> {
        let layout = self.multi_stark_layout_with_max_height(max_height)?;
        Ok(self.lower(namespace, layout))
    }

    /// Keep integer copy links instead of dense main fixed matrices and wires.
    /// Generate each partition's setup and witness with `circuit_input` and
    /// `trace_shards`. The logical circuit and custom fixed traces stay resident.
    pub fn lower_to_multi_stark_sharded(
        self,
        namespace: F,
        max_height: usize,
    ) -> Result<MultiStarkCircuit<F>, LoweringError> {
        let layout = self.multi_stark_layout_with_max_height(max_height)?;
        let hash_definitions = self
            .hashes
            .as_ref()
            .filter(|h| !h.calls.is_empty())
            .map_or_else(Vec::new, |h| h.definitions(namespace));
        let mut end = 0;
        let hash_row_ends = self.hashes.as_ref().map_or_else(Vec::new, |h| {
            h.calls
                .iter()
                .map(|call| {
                    end += call.input.len() + call.output.len();
                    end
                })
                .collect()
        });
        let mut compiled = MultiStarkCircuit {
            circuit: self,
            namespace,
            width: layout.main_width,
            wires: vec![],
            fixed: RowMajorMatrix::new(vec![], layout.preprocessed_width),
            main_heights: layout.main_heights,
            successors: None,
            hash_row_ends,
            hash_definitions,
            merged_table_height: None,
        };
        let mut successors = vec![0; layout.main_height * compiled.width];
        let mut endpoints = vec![(usize::MAX, usize::MAX); compiled.circuit.num_values()];
        let mut wires = vec![compiled.circuit.zero; compiled.width];
        for row in 0..layout.main_height {
            compiled.layout_wires(row, &mut wires);
            for (col, value) in wires.iter().enumerate() {
                let cell = row * compiled.width + col;
                let (first, last) = &mut endpoints[value.index];
                if *first == usize::MAX {
                    *first = cell;
                } else {
                    successors[*last] = cell;
                }
                *last = cell;
            }
        }
        for (first, last) in endpoints {
            if first != usize::MAX {
                successors[last] = first;
            }
        }
        compiled.successors = Some(successors);
        Ok(compiled)
    }

    fn lower(self, namespace: F, layout: MultiStarkLayout) -> MultiStarkCircuit<F> {
        let width = layout.main_width;
        let fixed_width = layout.preprocessed_width;
        let cells = layout.main_height * width;
        let fixed_len = layout.main_height * fixed_width;

        let mut wires = vec![self.zero; cells];
        let mut fixed = F::zero_vec(fixed_len);
        if layout.main_heights.len() > 1 {
            // A fixed indicator, not a witness-controlled activation flag.
            let mut offset = 0;
            for &height in &layout.main_heights {
                fixed[offset * fixed_width + fixed_width - 1] = F::ONE;
                offset += height;
            }
        }
        for (row, gate) in self.gates.iter().enumerate() {
            wires[row * width..row * width + 3].copy_from_slice(&gate.wires);
            fixed[row * fixed_width..row * fixed_width + 5].copy_from_slice(&gate.coefficients);
        }
        let public_enable = 5 + 2 * width;
        let public_index = public_enable + 1;
        let table_enable = public_enable + 2;
        let table_index = public_enable + 3;
        // The anchor is public index zero. User public indices start at one.
        for (index, value) in std::iter::once(self.zero)
            .chain(self.publics.iter().copied())
            .enumerate()
        {
            let row = self.gates.len() + index;
            wires[row * width] = value;
            fixed[row * fixed_width + public_enable] = F::ONE;
            fixed[row * fixed_width + public_index] = F::from_usize(index);
        }
        for (index, lookup) in self.lookups.iter().enumerate() {
            let row = self.gates.len() + self.publics.len() + 1 + index;
            wires[row * width..row * width + lookup.values.len()].copy_from_slice(&lookup.values);
            fixed[row * fixed_width + table_enable] = F::ONE;
            fixed[row * fixed_width + table_index] = F::from_usize(lookup.table.index);
        }
        if let Some(hashes) = &self.hashes {
            let mut row = self.gates.len() + self.publics.len() + 1 + self.lookups.len();
            for (call_index, call) in hashes.calls.iter().enumerate() {
                for (offset, &value) in call.input.iter().chain(&call.output).enumerate() {
                    wires[row * width] = value;
                    fixed[row * fixed_width + table_index] = F::from_usize(call_index);
                    fixed[row * fixed_width + public_index] = F::from_usize(offset);
                    fixed[row * fixed_width + table_index + 1] = if offset < call.input.len() {
                        F::ONE
                    } else {
                        F::NEG_ONE
                    };
                    row += 1;
                }
            }
        }

        // Build each value's cycle in physical order using just its first and
        // last occurrence. No per-value heap allocations or stored cell lists.
        let mut endpoints = vec![(usize::MAX, usize::MAX); self.num_values()];
        for (cell, value) in wires.iter().enumerate() {
            let row = cell / width;
            let col = cell % width;
            fixed[row * fixed_width + 5 + col] = F::from_usize(cell);
            let (first, last) = &mut endpoints[value.index];
            if *first == usize::MAX {
                *first = cell;
            } else {
                fixed[(*last / width) * fixed_width + 5 + width + *last % width] =
                    F::from_usize(cell);
            }
            *last = cell;
        }
        for (first, last) in endpoints {
            if first != usize::MAX {
                fixed[(last / width) * fixed_width + 5 + width + last % width] =
                    F::from_usize(first);
            }
        }
        let hash_definitions = self
            .hashes
            .as_ref()
            .filter(|h| !h.calls.is_empty())
            .map_or_else(Vec::new, |h| h.definitions(namespace));
        MultiStarkCircuit {
            circuit: self,
            namespace,
            width,
            wires,
            fixed: RowMajorMatrix::new(fixed, fixed_width),
            main_heights: layout.main_heights,
            successors: None,
            hash_row_ends: vec![],
            hash_definitions,
            merged_table_height: None,
        }
    }
}

impl<F: Field> MultiStarkCircuit<F> {
    /// Combine fixed tables in one trace, retaining a fixed table-ID column.
    /// This saves commitments/openings at the cost of a taller table domain.
    pub fn merge_table_traces(mut self, max_height: usize) -> Result<Self, LoweringError> {
        if max_height < 2 || !max_height.is_power_of_two() {
            return Err(LoweringError::InvalidTraceHeight);
        }
        if !self.circuit.tables.is_empty() {
            let rows = self
                .circuit
                .tables
                .iter()
                .try_fold(0usize, |n, table| n.checked_add(table.rows.len()))
                .ok_or(LoweringError::SizeOverflow)?;
            let height = rows
                .max(2)
                .checked_next_power_of_two()
                .ok_or(LoweringError::SizeOverflow)?;
            if height > max_height {
                return Err(LoweringError::TableExceedsTraceHeight);
            }
            height
                .checked_mul(self.width + 1)
                .ok_or(LoweringError::SizeOverflow)?;
            self.merged_table_height = Some(height);
        }
        Ok(self)
    }

    pub fn circuit(&self) -> &Circuit<F> {
        &self.circuit
    }

    pub fn main_width(&self) -> usize {
        self.width
    }

    /// Total padded computation rows, not an individual FFT domain size.
    pub fn main_height(&self) -> usize {
        self.main_heights.iter().sum()
    }

    /// Individual computation domains, before FRI blowup.
    pub fn main_heights(&self) -> &[usize] {
        &self.main_heights
    }

    pub fn num_circuits(&self) -> usize {
        self.main_heights.len() + self.auxiliary_widths().count()
    }

    pub fn witness(&self) -> Witness<'_, F> {
        self.circuit.witness()
    }

    /// Setup data in canonical order: computation partitions, then fixed tables.
    pub fn circuit_inputs(&self) -> Vec<CircuitInputs<F>> {
        self.selected_inputs(None)
    }

    /// Canonical (height, width), without materializing preprocessing or witness data.
    pub fn trace_dimensions(&self, index: usize) -> Option<(usize, usize)> {
        if let Some(&height) = self.main_heights.get(index) {
            return Some((height, self.width));
        }
        let aux = index.checked_sub(self.main_heights.len())?;
        let tables = if let Some(height) = self.merged_table_height {
            if aux == 0 {
                return Some((height, 1));
            }
            1
        } else {
            if let Some(table) = self.circuit.tables.get(aux) {
                return Some((table.rows.len().max(2).next_power_of_two(), 1));
            }
            self.circuit.tables.len()
        };
        let definition = self.hash_definitions.get(aux.checked_sub(tables)?)?;
        Some((
            definition.preprocessed.as_ref()?.height(),
            definition.main_width,
        ))
    }

    /// Generate setup data for one circuit, without cloning the other fixed matrices.
    pub fn circuit_input(&self, index: usize) -> Option<CircuitInputs<F>> {
        self.selected_inputs(Some(index)).pop()
    }

    fn selected_inputs(&self, selected: Option<usize>) -> Vec<CircuitInputs<F>> {
        // Widths were checked against u32 during lowering.
        let main = |i| Expr::main(u32::try_from(i).expect("checked column"));
        let prep = |i| Expr::preprocessed(u32::try_from(i).expect("checked column"));
        let constant = Expr::constant;
        let (relation, lookups) = computation_relations(
            self.namespace,
            self.width,
            !self.hash_definitions.is_empty(),
        );
        let mut inputs = Vec::new();
        for index in 0..self.main_heights.len() {
            if selected.is_some_and(|i| i != index) {
                continue;
            }
            let mut local_lookups = lookups.clone();
            if self.main_heights.len() > 1 {
                // Separate channel from COPY=0, PUBLIC=1 and TABLE=2.
                // Exactly one pull per trace, even if it has no user publics.
                local_lookups.push(Lookup::pull(
                    prep(self.fixed.width() - 1),
                    vec![
                        constant(self.namespace),
                        constant(F::from_u8(3)),
                        constant(F::from_usize(index)),
                    ],
                ));
            }
            inputs.push(CircuitInputs {
                main_width: self.width,
                preprocessed: Some(self.fixed_partition(index)),
                constraints: vec![relation.clone()],
                lookups: local_lookups,
                ..Default::default()
            });
        }
        if let Some(height) = self.merged_table_height {
            if selected.is_none_or(|i| i == self.main_heights.len()) {
                let width = self.width + 1;
                let mut rows = Vec::with_capacity(height * width);
                for (index, table) in self.circuit.tables.iter().enumerate() {
                    for row in &table.rows {
                        rows.push(F::from_usize(index));
                        rows.extend_from_slice(row);
                        rows.resize(rows.len() + self.width - row.len(), F::ZERO);
                    }
                }
                // Padding repeats a genuine tagged row, never a new table entry.
                while rows.len() < height * width {
                    rows.extend_from_within(..width);
                }
                inputs.push(CircuitInputs {
                    main_width: 1,
                    preprocessed: Some(RowMajorMatrix::new(rows, width)),
                    lookups: vec![merged_table_lookup(self.namespace, width)],
                    ..Default::default()
                });
            }
        } else {
            for (index, table) in self.circuit.tables.iter().enumerate() {
                if selected.is_some_and(|i| i != self.main_heights.len() + index) {
                    continue;
                }
                let height = table.rows.len().max(2).next_power_of_two();
                let mut rows = Vec::with_capacity(height * self.width);
                for i in 0..height {
                    let row = table.rows.get(i).unwrap_or(&table.rows[0]);
                    rows.extend_from_slice(row);
                    rows.resize(rows.len() + self.width - row.len(), F::ZERO);
                }
                let mut args = vec![
                    constant(self.namespace),
                    constant(F::TWO),
                    constant(F::from_usize(index)),
                ];
                args.extend((0..self.width).map(prep));
                inputs.push(CircuitInputs {
                    main_width: 1,
                    preprocessed: Some(RowMajorMatrix::new(rows, self.width)),
                    lookups: vec![Lookup::push(main(0), args)],
                    ..Default::default()
                });
            }
        }
        let hash_start = self.main_heights.len()
            + if self.merged_table_height.is_some() {
                1
            } else {
                self.circuit.tables.len()
            };
        inputs.extend(
            self.hash_definitions
                .iter()
                .enumerate()
                .filter(|(index, _)| selected.is_none_or(|i| i == hash_start + index))
                .map(|(_, d)| d.clone()),
        );
        inputs
    }

    /// Generates traces and table multiplicities from a logical assignment.
    pub fn traces(
        &self,
        assignment: &Assignment<F>,
    ) -> Result<Vec<RowMajorMatrix<F>>, WitnessError> {
        self.trace_shards(assignment)?;
        let mut traces: Vec<_> = (0..self.main_heights.len())
            .map(|index| self.main_trace(assignment, index))
            .collect();
        traces.extend(self.auxiliary_traces(assignment)?);
        Ok(traces)
    }

    /// Bind an immutable, validated assignment to this circuit, then generate
    /// shards on demand for batch proving with `Retention::Regenerate`. The
    /// circuit, fixed data, assignment and compact hash preparation remain
    /// resident; this only bounds expanded witness traces.
    pub fn trace_shards<'a>(
        &'a self,
        assignment: &'a Assignment<F>,
    ) -> Result<TraceShards<'a, F>, WitnessError> {
        if assignment.owner() != self.circuit.owner {
            return Err(WitnessError::ForeignAssignment);
        }
        Ok(TraceShards {
            compiled: self,
            assignment,
        })
    }

    fn layout_wires(&self, row: usize, wires: &mut [Value]) {
        self.layout_row::<false>(row, wires, &mut []);
    }

    fn layout_fixed(&self, row: usize, fixed: &mut [F]) {
        self.layout_row::<true>(row, &mut [], fixed);
    }

    fn layout_row<const FIXED: bool>(&self, row: usize, wires: &mut [Value], fixed: &mut [F]) {
        if FIXED {
            fixed.fill(F::ZERO);
        } else {
            wires.fill(self.circuit.zero);
        }
        let c = &self.circuit;
        let public_enable = 5 + 2 * self.width;
        let public_index = public_enable + 1;
        let table_enable = public_enable + 2;
        let table_index = public_enable + 3;
        if let Some(gate) = c.gates.get(row) {
            if FIXED {
                fixed[..5].copy_from_slice(&gate.coefficients);
            } else {
                wires[..3].copy_from_slice(&gate.wires);
            }
        } else if row < c.gates.len() + 1 + c.publics.len() {
            let index = row - c.gates.len();
            if FIXED {
                fixed[public_enable] = F::ONE;
                fixed[public_index] = F::from_usize(index);
            } else {
                wires[0] = if index == 0 {
                    c.zero
                } else {
                    c.publics[index - 1]
                };
            }
        } else {
            let index = row - c.gates.len() - 1 - c.publics.len();
            if let Some(lookup) = c.lookups.get(index) {
                if FIXED {
                    fixed[table_enable] = F::ONE;
                    fixed[table_index] = F::from_usize(lookup.table.index);
                } else {
                    wires[..lookup.values.len()].copy_from_slice(&lookup.values);
                }
            } else {
                let index = index - c.lookups.len();
                let call_index = self.hash_row_ends.partition_point(|&end| end <= index);
                if let Some(call) = c.hashes.as_ref().and_then(|h| h.calls.get(call_index)) {
                    let offset = index
                        - call_index
                            .checked_sub(1)
                            .map_or(0, |i| self.hash_row_ends[i]);
                    if FIXED {
                        fixed[table_index] = F::from_usize(call_index);
                        fixed[public_index] = F::from_usize(offset);
                        fixed[table_index + 1] = if offset < call.input.len() {
                            F::ONE
                        } else {
                            F::NEG_ONE
                        };
                    } else {
                        wires[0] = if offset < call.input.len() {
                            call.input[offset]
                        } else {
                            call.output[offset - call.input.len()]
                        };
                    }
                }
            }
        }
    }

    fn fixed_partition(&self, index: usize) -> RowMajorMatrix<F> {
        let start = self.main_heights[..index].iter().sum::<usize>();
        let height = self.main_heights[index];
        let width = self.fixed.width();
        let Some(successors) = &self.successors else {
            return RowMajorMatrix::new(
                self.fixed.values[start * width..(start + height) * width].to_vec(),
                width,
            );
        };
        let mut values = F::zero_vec(height * width);
        values
            .par_chunks_mut(width * (1 << 12))
            .enumerate()
            .for_each(|(tile, output)| {
                for (offset, fixed) in output.chunks_exact_mut(width).enumerate() {
                    let row = start + tile * (1 << 12) + offset;
                    self.layout_fixed(row, fixed);
                    for col in 0..self.width {
                        let cell = row * self.width + col;
                        fixed[5 + col] = F::from_usize(cell);
                        fixed[5 + self.width + col] = F::from_usize(successors[cell]);
                    }
                }
            });
        if self.main_heights.len() > 1 {
            values[width - 1] = F::ONE;
        }
        RowMajorMatrix::new(values, width)
    }

    fn main_trace(&self, assignment: &Assignment<F>, index: usize) -> RowMajorMatrix<F> {
        let start = self.main_heights[..index].iter().sum::<usize>() * self.width;
        let end = start + self.main_heights[index] * self.width;
        if self.successors.is_some() {
            let mut values = F::zero_vec(end - start);
            values
                .par_chunks_mut(self.width * (1 << 12))
                .enumerate()
                .for_each(|(tile, output)| {
                    let mut wires = vec![self.circuit.zero; self.width];
                    for (offset, row_values) in output.chunks_exact_mut(self.width).enumerate() {
                        let row = start / self.width + tile * (1 << 12) + offset;
                        self.layout_wires(row, &mut wires);
                        for (output, wire) in row_values.iter_mut().zip(&wires) {
                            *output = assignment.values()[wire.index];
                        }
                    }
                });
            return RowMajorMatrix::new(values, self.width);
        }
        RowMajorMatrix::new(
            self.wires[start..end]
                .par_iter()
                .map(|v| assignment.values()[v.index])
                .collect(),
            self.width,
        )
    }

    fn auxiliary_widths(&self) -> impl Iterator<Item = usize> + '_ {
        let tables = if self.merged_table_height.is_some() {
            1
        } else {
            self.circuit.tables.len()
        };
        std::iter::repeat_n(1, tables).chain(self.hash_definitions.iter().map(|d| d.main_width))
    }

    fn auxiliary_traces(
        &self,
        assignment: &Assignment<F>,
    ) -> Result<Vec<RowMajorMatrix<F>>, WitnessError> {
        let mut traces = self.table_traces(assignment);
        if !self.hash_definitions.is_empty() {
            traces.extend(self.circuit.hashes.as_ref().unwrap().traces(
                assignment.prepared_hash(&self.circuit)?,
                &self.hash_definitions,
            ));
        }
        Ok(traces)
    }

    fn table_traces(&self, assignment: &Assignment<F>) -> Vec<RowMajorMatrix<F>> {
        let mut traces = Vec::new();
        let multiplicities = multiplicities::table_counts(&self.circuit, assignment.values());
        if let Some(height) = self.merged_table_height {
            let mut merged = Vec::with_capacity(height);
            for (counts, table) in multiplicities.into_iter().zip(&self.circuit.tables) {
                merged.extend_from_slice(&counts[..table.rows.len()]);
            }
            merged.resize(height, F::ZERO);
            traces.push(RowMajorMatrix::new_col(merged));
        } else {
            traces.extend(multiplicities.into_iter().map(RowMajorMatrix::new_col));
        }
        traces
    }

    /// Encodes the expected public statement, including the mandatory anchor.
    /// Pass the returned claims to both `prove_multiple_claims` and
    /// `verify_multiple_claims`. The verifier supplies its own expected values;
    /// it must not trust values taken from the prover's witness.
    pub fn claims(&self, public_values: &[F]) -> Result<Vec<Vec<F>>, WitnessError> {
        if public_values.len() != self.circuit.publics.len() {
            return Err(WitnessError::PublicCount {
                expected: self.circuit.publics.len(),
                actual: public_values.len(),
            });
        }
        let mut claims: Vec<_> = std::iter::once(F::ZERO)
            .chain(public_values.iter().copied())
            .enumerate()
            .map(|(index, value)| vec![self.namespace, F::ONE, F::from_usize(index), value])
            .collect();
        if self.main_heights.len() > 1 {
            claims.extend(
                (0..self.main_heights.len())
                    .map(|index| vec![self.namespace, F::from_u8(3), F::from_usize(index)]),
            );
        }
        if let Some(hashes) = &self.circuit.hashes {
            claims.extend(hashes.claims(self.namespace));
        }
        Ok(claims)
    }
}

fn partition_heights(used_rows: usize, cap: Option<usize>) -> Result<Vec<usize>, LoweringError> {
    if let Some(cap) = cap {
        if cap < 2 || !cap.is_power_of_two() {
            return Err(LoweringError::InvalidTraceHeight);
        }
        let mut heights = vec![cap; used_rows / cap];
        let tail = used_rows % cap;
        if tail != 0 || heights.is_empty() {
            heights.push(
                tail.max(2)
                    .checked_next_power_of_two()
                    .ok_or(LoweringError::SizeOverflow)?,
            );
        }
        Ok(heights)
    } else {
        Ok(vec![
            used_rows
                .max(2)
                .checked_next_power_of_two()
                .ok_or(LoweringError::SizeOverflow)?,
        ])
    }
}

#[cfg(test)]
mod row_selection_tests;

#[cfg(test)]
mod partition_tests {
    use super::*;
    use crate::traits::Algebra;
    use crate::{plonkish::CircuitBuilder, types::Val};

    #[test]
    fn merged_tables_keep_ids_and_unpadded_multiplicities() {
        let mut b = CircuitBuilder::<Val>::new();
        let a = b.fixed_table(
            "a",
            vec![vec![Val::ONE], vec![Val::TWO], vec![Val::from_u8(3)]],
        );
        let c = b.fixed_table("c", vec![vec![Val::ONE, Val::ZERO]]);
        let one = b.constant(Val::ONE);
        let zero = b.constant(Val::ZERO);
        b.lookup(a, &[one]);
        b.lookup(a, &[one]);
        b.lookup(c, &[one, zero]);
        let compiled = b
            .finish()
            .lower_to_multi_stark(Val::from_u8(93))
            .unwrap()
            .merge_table_traces(4)
            .unwrap();
        let assignment = compiled.witness().generate().unwrap();
        let definitions = compiled.circuit_inputs();
        let traces = compiled.traces(&assignment).unwrap();
        assert_eq!(definitions.len(), 2);
        let fixed = definitions[1].preprocessed.as_ref().unwrap();
        assert_eq!(fixed.width(), 4);
        assert_eq!(fixed.height(), 4);
        assert_eq!(
            &fixed.values[..4],
            &[Val::ZERO, Val::ONE, Val::ZERO, Val::ZERO]
        );
        assert_eq!(
            &fixed.values[12..],
            &[Val::ONE, Val::ONE, Val::ZERO, Val::ZERO]
        );
        assert_eq!(traces[1].values, [Val::TWO, Val::ZERO, Val::ZERO, Val::ONE]);
        assert!(matches!(
            compiled.merge_table_traces(2),
            Err(LoweringError::TableExceedsTraceHeight)
        ));
    }

    #[test]
    fn generated_fixed_partitions_match_dense_layout() {
        use crate::plonkish::gadgets::{ByteGadgets, blake3};
        let build = || {
            let mut b = CircuitBuilder::<Val>::new();
            b.enable_compact_blake3();
            let one = b.constant(Val::ONE);
            let two = b.add(one, one);
            b.expose_public(two);
            let table = b.fixed_table("wide", vec![vec![Val::ONE; 5]]);
            b.lookup(table, &[one; 5]);
            let gadgets = ByteGadgets::new(&mut b);
            let bytes: Vec<_> = (0..3).map(|i| gadgets.constant(&mut b, i)).collect();
            blake3(&mut b, &gadgets, &bytes);
            b.finish()
        };
        // Custom hash tables need 2^16 rows. Multiple main partitions and a
        // partial final partition are exercised separately below.
        let dense = build()
            .lower_to_multi_stark_with_max_height(Val::ONE, 1 << 16)
            .unwrap();
        let lazy = build()
            .lower_to_multi_stark_sharded(Val::ONE, 1 << 16)
            .unwrap();
        compare_layouts(&dense, &lazy);
        let build = || {
            let mut b = CircuitBuilder::<Val>::new();
            let one = b.constant(Val::ONE);
            let table = b.fixed_table("wide", vec![vec![Val::ONE; 5]]);
            for _ in 0..9 {
                b.lookup(table, &[one; 5]);
            }
            b.expose_public(one);
            b.finish()
        };
        compare_layouts(
            &build()
                .lower_to_multi_stark_with_max_height(Val::ONE, 8)
                .unwrap()
                .merge_table_traces(8)
                .unwrap(),
            &build()
                .lower_to_multi_stark_sharded(Val::ONE, 8)
                .unwrap()
                .merge_table_traces(8)
                .unwrap(),
        );
        let build = || {
            let mut b = CircuitBuilder::<Val>::new();
            let one = b.constant(Val::ONE);
            let table = b.fixed_table("tile-boundary", vec![vec![Val::ONE; 5]]);
            for _ in 0..5003 {
                b.lookup(table, &[one; 5]);
            }
            b.expose_public(one);
            b.finish()
        };
        compare_layouts(
            &build()
                .lower_to_multi_stark_with_max_height(Val::ONE, 8192)
                .unwrap(),
            &build()
                .lower_to_multi_stark_sharded(Val::ONE, 8192)
                .unwrap(),
        );
    }

    fn compare_layouts(dense: &MultiStarkCircuit<Val>, lazy: &MultiStarkCircuit<Val>) {
        assert!(lazy.fixed.values.is_empty());
        assert!(lazy.wires.is_empty());
        assert_eq!(dense.main_heights(), lazy.main_heights());
        assert_eq!(dense.main_height(), lazy.main_height());
        assert_eq!(dense.num_circuits(), lazy.num_circuits());
        for i in 0..dense.num_circuits() {
            let a = dense.circuit_input(i).unwrap();
            let b = lazy.circuit_input(i).unwrap();
            let dimensions = Some((a.preprocessed.as_ref().unwrap().height(), a.main_width));
            assert_eq!(dense.trace_dimensions(i), dimensions);
            assert_eq!(lazy.trace_dimensions(i), dimensions);
            assert_eq!(a.preprocessed, b.preprocessed);
            assert_eq!(
                format!("{:?}", a.constraints),
                format!("{:?}", b.constraints)
            );
            assert_eq!(format!("{:?}", a.lookups), format!("{:?}", b.lookups));
        }
        assert!(lazy.circuit_input(lazy.num_circuits()).is_none());
        assert!(lazy.trace_dimensions(lazy.num_circuits()).is_none());
        let a = dense.witness().generate().unwrap();
        let b = lazy.witness().generate().unwrap();
        let expected = dense.traces(&a).unwrap();
        assert_eq!(expected, lazy.traces(&b).unwrap());
        let shards = lazy.trace_shards(&b).unwrap();
        for (index, trace) in expected.iter().enumerate() {
            assert_eq!(*trace, shards.trace(index).unwrap());
        }
        assert!(shards.trace(expected.len()).is_err());
        assert_eq!(
            dense.claims(a.public_values()).unwrap(),
            lazy.claims(b.public_values()).unwrap()
        );
    }

    #[test]
    fn root_dimensions_fit_goldilocks_domains() {
        let heights = partition_heights(1_723_866_138, Some(1 << 24)).unwrap();
        assert_eq!(heights.len(), 103);
        assert!(heights.iter().all(|&n| n.is_power_of_two() && n <= 1 << 24));
        assert!(heights.iter().all(|&n| n.ilog2() + 2 <= 32));
        assert_eq!(partition_heights(8, Some(4)).unwrap(), [4, 4]);
        assert_eq!(partition_heights(9, Some(4)).unwrap(), [4, 4, 2]);
        for cap in [0, 1, 3, 6] {
            assert_eq!(
                partition_heights(10, Some(cap)),
                Err(LoweringError::InvalidTraceHeight)
            );
        }
    }
}
