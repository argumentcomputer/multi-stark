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

use crate::expr::Expr;
use crate::lookup::Lookup;
use crate::system::CircuitInputs;

use super::{Assignment, Circuit, Value, Witness, WitnessError};

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
    hash_definitions: Vec<CircuitInputs<F>>,
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

    fn lower(self, namespace: F, layout: MultiStarkLayout) -> MultiStarkCircuit<F> {
        let width = layout.main_width;
        let fixed_width = layout.preprocessed_width;
        let cells = layout.main_height * width;
        let fixed_len = layout.main_height * fixed_width;

        let mut wires = vec![self.zero; cells];
        let mut fixed = vec![F::ZERO; fixed_len];
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
            hash_definitions,
        }
    }
}

impl<F: Field> MultiStarkCircuit<F> {
    pub fn circuit(&self) -> &Circuit<F> {
        &self.circuit
    }

    pub fn main_width(&self) -> usize {
        self.width
    }

    /// Total padded computation rows, not an individual FFT domain size.
    pub fn main_height(&self) -> usize {
        self.fixed.height()
    }

    /// Individual computation domains, before FRI blowup.
    pub fn main_heights(&self) -> &[usize] {
        &self.main_heights
    }

    pub fn witness(&self) -> Witness<'_, F> {
        self.circuit.witness()
    }

    /// Setup data in canonical order: computation partitions, then fixed tables.
    pub fn circuit_inputs(&self) -> Vec<CircuitInputs<F>> {
        // Widths were checked against u32 during lowering.
        let main = |i| Expr::main(u32::try_from(i).expect("checked column"));
        let prep = |i| Expr::preprocessed(u32::try_from(i).expect("checked column"));
        let constant = Expr::constant;
        let a = main(0);
        let b = main(1);
        let c = main(2);
        let relation = prep(0) * a.clone() * b.clone()
            + prep(1) * a.clone()
            + prep(2) * b
            + prep(3) * c
            + prep(4);
        let mut lookups = Vec::new();
        for col in 0..self.width {
            lookups.push(Lookup::push(
                constant(F::ONE),
                vec![
                    constant(self.namespace),
                    constant(F::ZERO),
                    prep(5 + col),
                    main(col),
                ],
            ));
            lookups.push(Lookup::pull(
                constant(F::ONE),
                vec![
                    constant(self.namespace),
                    constant(F::ZERO),
                    prep(5 + self.width + col),
                    main(col),
                ],
            ));
        }
        let public_enable = 5 + 2 * self.width;
        lookups.push(Lookup::pull(
            prep(public_enable),
            vec![
                constant(self.namespace),
                constant(F::ONE),
                prep(public_enable + 1),
                a,
            ],
        ));
        let mut table_args = vec![
            constant(self.namespace),
            constant(F::TWO),
            prep(public_enable + 3),
        ];
        table_args.extend((0..self.width).map(main));
        lookups.push(Lookup::pull(prep(public_enable + 2), table_args));
        if !self.hash_definitions.is_empty() {
            lookups.push(Lookup::push(
                prep(public_enable + 4),
                vec![
                    constant(self.namespace),
                    constant(F::from_u8(4)),
                    constant(F::from_u8(4)),
                    prep(public_enable + 3),
                    prep(public_enable + 1),
                    main(0),
                ],
            ));
        }
        let mut inputs = Vec::new();
        let mut offset = 0;
        for (index, &height) in self.main_heights.iter().enumerate() {
            let end = offset + height * self.fixed.width();
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
                preprocessed: Some(RowMajorMatrix::new(
                    self.fixed.values[offset..end].to_vec(),
                    self.fixed.width(),
                )),
                constraints: vec![relation.clone()],
                lookups: local_lookups,
                ..Default::default()
            });
            offset = end;
        }
        for (index, table) in self.circuit.tables.iter().enumerate() {
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
        inputs.extend(self.hash_definitions.clone());
        inputs
    }

    /// Generates traces and table multiplicities from a logical assignment.
    pub fn traces(
        &self,
        assignment: &Assignment<F>,
    ) -> Result<Vec<RowMajorMatrix<F>>, WitnessError> {
        if assignment.owner != self.circuit.owner {
            return Err(WitnessError::ForeignAssignment);
        }
        self.circuit.check_values(&assignment.values)?;
        let mut traces = Vec::new();
        let mut offset = 0;
        for &height in &self.main_heights {
            let end = offset + height * self.width;
            traces.push(RowMajorMatrix::new(
                self.wires[offset..end]
                    .iter()
                    .map(|v| assignment.values[v.index])
                    .collect(),
                self.width,
            ));
            offset = end;
        }
        let mut multiplicities: Vec<Vec<F>> = self
            .circuit
            .tables
            .iter()
            .map(|table| vec![F::ZERO; table.rows.len().max(2).next_power_of_two()])
            .collect();
        let mut args = Vec::new();
        for lookup in &self.circuit.lookups {
            let table = &self.circuit.tables[lookup.table.index];
            args.clear();
            args.extend(lookup.values.iter().map(|v| assignment.values[v.index]));
            let row = table.indices[&args];
            multiplicities[lookup.table.index][row] += F::ONE;
        }
        traces.extend(multiplicities.into_iter().map(RowMajorMatrix::new_col));
        if !self.hash_definitions.is_empty() {
            traces.extend(
                self.circuit
                    .hashes
                    .as_ref()
                    .unwrap()
                    .traces(&assignment.values, &self.hash_definitions)?,
            );
        }
        Ok(traces)
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
mod partition_tests {
    use super::*;
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
