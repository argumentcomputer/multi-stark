//! Lookups: the [`Lookup`] type, the logUp stage-2 constraint synthesis, and
//! the concrete lookup witness ([`LookupValues`]) plus its stage-2 trace
//! construction.
//!
//! Synthesis is a frontend-to-frontend function: given the lookups, it
//! produces the extension-field constraints for the running-accumulator
//! argument as ordinary [`ExtExpr`] data. [`crate::system::System::new`]
//! appends these to a circuit's `ext_constraints` before compilation, so they
//! are interned, coordinate-expanded and canonicalized like any other
//! constraint.
//!
//! # Layout conventions (owned here)
//!
//! - **Publics** (each an extension value, stored as `d` base coordinates):
//!   slot 0 = β (lookup challenge), 1 = γ (fingerprint challenge),
//!   2 = current accumulator, 3 = next accumulator. So `num_publics = 4·d`.
//! - **Stage-2 columns** (extension values, flattened to base): slot 0 = the
//!   running accumulator, slots `1..=L` = the message inverse of each lookup,
//!   in lookup order. So `stage2_width = (1 + L)·d` base columns.

use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, batch_multiplicative_inverse};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::expr::{Expr, ExtExpr, RowOffset};

/// A lookup: a multiplicity and a vector of arguments. `E` is a frontend
/// expression in a [`crate::expr::CircuitSpec`] and a node id once compiled.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Lookup<E> {
    pub multiplicity: E,
    pub args: Vec<E>,
}

/// Number of extension-valued public inputs the lookup argument uses:
/// β, γ, current accumulator, next accumulator.
pub const LOOKUP_PUBLIC_SIZE: usize = 4;

/// Public-input width (base coordinates) for a circuit whose extension
/// degree is `d`.
pub fn num_publics(d: usize) -> usize {
    LOOKUP_PUBLIC_SIZE * d
}

/// Stage-2 trace width (flattened base columns) for `num_lookups` lookups
/// at extension degree `d`: one accumulator column plus one inverse column
/// per lookup.
pub fn stage2_width(num_lookups: usize, d: usize) -> usize {
    (1 + num_lookups) * d
}

/// Builds the logUp stage-2 constraints for `lookups` at extension degree
/// `d`. Order: the per-lookup message/inverse constraints in lookup order,
/// then the first-row, transition, and last-row accumulator constraints.
pub fn synthesize_lookups<F: Field>(lookups: &[Lookup<Expr<F>>], d: usize) -> Vec<ExtExpr<F>> {
    let d = u32::try_from(d).expect("extension degree exceeds u32");
    let beta = ExtExpr::public(0, d);
    let gamma = ExtExpr::public(1, d);
    let acc = ExtExpr::public(2, d);
    let next_acc = ExtExpr::public(3, d);
    let acc_col = ExtExpr::stage2(0, d, RowOffset::Current);
    let next_acc_col = ExtExpr::stage2(0, d, RowOffset::Next);

    let mut constraints = Vec::with_capacity(lookups.len() + 3);
    // acc_expr = acc_col + Σ_j multiplicity_j · inv_j, built once and reused
    // by the transition and last-row constraints (the interner shares it).
    let mut acc_expr = acc_col.clone();
    for (j, lookup) in lookups.iter().enumerate() {
        let slot = 1 + u32::try_from(j).expect("lookup count exceeds u32");
        let inv = ExtExpr::stage2(slot, d, RowOffset::Current);

        // fingerprint = Σ_i args[i] · γ^i, via Horner over the reversed args.
        let mut coeffs = lookup.args.iter().rev();
        let mut fingerprint = match coeffs.next() {
            Some(arg) => ExtExpr::from(arg.clone()),
            None => ExtExpr::from(Expr::constant(F::ZERO)),
        };
        for arg in coeffs {
            fingerprint = fingerprint * gamma.clone() + arg.clone();
        }

        // message = β + fingerprint; constraint: message · inv − 1 = 0.
        let message = beta.clone() + fingerprint;
        constraints.push(message * inv.clone() - Expr::constant(F::ONE));

        acc_expr = acc_expr + lookup.multiplicity.clone() * inv;
    }

    // First row: acc_col = acc.
    constraints.push(Expr::IsFirstRow * (acc_col - acc));
    // Transition: acc_expr = next row's accumulator column.
    constraints.push(Expr::IsTransition * (acc_expr.clone() - next_acc_col));
    // Last row: acc_expr = next_acc.
    constraints.push(Expr::IsLastRow * (acc_expr - next_acc));

    constraints
}

/// Computes a fingerprint of the coefficients using Horner's method.
#[inline]
pub(crate) fn fingerprint<F, I, Iter>(r: &F, coeffs: Iter) -> F
where
    F: PrimeCharacteristicRing,
    I: Into<F>,
    Iter: DoubleEndedIterator<Item = I>,
{
    coeffs
        .rev()
        .fold(F::ZERO, |acc, coeff| acc * r.clone() + coeff.into())
}

/// Concrete lookup values of one circuit, stored flat.
///
/// Every row of a circuit has the same lookups (each with a fixed number of
/// arguments), so all multiplicities and argument values are kept in two
/// row-major vectors instead of nested per-row, per-lookup allocations.
#[derive(Clone)]
pub struct LookupValues<F> {
    /// Number of trace rows.
    height: usize,
    /// Number of lookups per row.
    num_lookups: usize,
    /// Multiplicity of each (row, lookup); `height * num_lookups` values.
    multiplicities: Vec<F>,
    /// Offset of each lookup's arguments within a row's argument block;
    /// `num_lookups + 1` entries, the last being the block width.
    arg_offsets: Vec<usize>,
    /// Concatenated argument values, row-major; `height * arg_offsets.last()`
    /// values.
    args: Vec<F>,
}

impl<F: Field> LookupValues<F> {
    /// Builds flat lookup values from per-row, per-lookup concrete lookups.
    ///
    /// Every row must have the same number of lookups. Argument counts may
    /// vary across rows within a lookup slot; shorter argument lists are
    /// zero-padded to the slot's maximum width. Padding is transparent to the
    /// protocol: the message fingerprint is a Horner evaluation, so trailing
    /// zero coefficients do not change its value.
    ///
    /// # Panics
    /// Panics if rows have differing numbers of lookups.
    pub fn from_rows(rows: Vec<Vec<Lookup<F>>>) -> Self {
        let height = rows.len();
        let num_lookups = rows.first().map_or(0, |row| row.len());
        let mut slot_widths = vec![0usize; num_lookups];
        for row in &rows {
            assert_eq!(
                row.len(),
                num_lookups,
                "every row must have the same number of lookups"
            );
            for (width, lookup) in slot_widths.iter_mut().zip(row) {
                *width = (*width).max(lookup.args.len());
            }
        }
        let mut arg_offsets = Vec::with_capacity(num_lookups + 1);
        arg_offsets.push(0);
        for width in &slot_widths {
            arg_offsets.push(arg_offsets.last().unwrap() + width);
        }
        let args_width = *arg_offsets.last().unwrap();
        let mut multiplicities = Vec::with_capacity(height * num_lookups);
        let mut args = Vec::with_capacity(height * args_width);
        for row in rows {
            for (lookup, &width) in row.into_iter().zip(&slot_widths) {
                multiplicities.push(lookup.multiplicity);
                let padding = width - lookup.args.len();
                args.extend(lookup.args);
                args.extend(std::iter::repeat_n(F::ZERO, padding));
            }
        }
        Self {
            height,
            num_lookups,
            multiplicities,
            arg_offsets,
            args,
        }
    }

    /// Returns an allocation-free builder; see [`LookupValuesBuilder`].
    pub fn builder(height: usize, slot_arg_widths: &[usize]) -> LookupValuesBuilder<F> {
        LookupValuesBuilder::new(height, slot_arg_widths)
    }

    /// The arguments of the given lookup on the given row.
    #[inline]
    fn args_at(&self, row: usize, lookup: usize) -> &[F] {
        let start = row * self.arg_offsets[self.num_lookups];
        &self.args[start + self.arg_offsets[lookup]..start + self.arg_offsets[lookup + 1]]
    }

    /// Computes the stage 2 traces and the intermediate accumulators for each
    /// circuit given a lookup challenge, a fingerprint challenge and the current
    /// accumulator value (computed from the initial claims).
    pub fn stage_2_traces<EF: ExtensionField<F>>(
        circuits: &[Self],
        lookup_challenge: EF,
        fingerprint_challenge: &EF,
        mut accumulator: EF,
    ) -> (Vec<RowMajorMatrix<EF>>, Vec<EF>) {
        // Compute the message for each lookup, in flat circuit-major order.
        let _g = tracing::info_span!("stark/lookup_messages").entered();
        let num_messages = circuits
            .iter()
            .map(|circuit| circuit.height * circuit.num_lookups)
            .sum();
        let mut messages: Vec<EF> = Vec::with_capacity(num_messages);
        for circuit in circuits {
            messages.extend(
                (0..circuit.height * circuit.num_lookups)
                    .into_par_iter()
                    .map(|idx| {
                        let (row, lookup) = (idx / circuit.num_lookups, idx % circuit.num_lookups);
                        let args = circuit.args_at(row, lookup);
                        lookup_challenge + fingerprint(fingerprint_challenge, args.iter().cloned())
                    })
                    .collect::<Vec<_>>(),
            );
        }
        drop(_g);

        // Compute the inverses of all messages in batch.
        let messages_inverses = tracing::info_span!("stark/batch_inverse")
            .in_scope(|| batch_multiplicative_inverse(&messages));
        // Only the inverses are consumed below.
        drop(messages);

        // Compute and collect intermediate accumulators and traces.
        let _g = tracing::info_span!("stark/lookup_traces").entered();
        let mut intermediate_accumulators = Vec::with_capacity(circuits.len());
        let mut traces = Vec::with_capacity(circuits.len());
        let mut offset = 0;
        for circuit in circuits {
            // Get the slice containing the messages inverses for the current circuit.
            let num_circuit_messages = circuit.height * circuit.num_lookups;
            let circuit_messages_inverses =
                &messages_inverses[offset..offset + num_circuit_messages];
            offset += num_circuit_messages;

            let vec = if circuit.num_lookups == 0 {
                // No row lookup. Just repeat the accumulator for each row.
                vec![accumulator; circuit.height]
            } else {
                // Flatten each row accumulator followed by the inverse of the message
                // associated with each row lookup.
                let mut vec = Vec::with_capacity(circuit.height * (1 + circuit.num_lookups));
                for (row_multiplicities, row_messages_inverses) in circuit
                    .multiplicities
                    .chunks_exact(circuit.num_lookups)
                    .zip(circuit_messages_inverses.chunks_exact(circuit.num_lookups))
                {
                    vec.push(accumulator);
                    for (&multiplicity, &message_inverse) in
                        row_multiplicities.iter().zip(row_messages_inverses)
                    {
                        accumulator += EF::from(multiplicity) * message_inverse;
                        vec.push(message_inverse);
                    }
                }
                vec
            };
            let width = 1 + circuit.num_lookups;
            debug_assert_eq!(vec.len() % width, 0);
            let trace = RowMajorMatrix::new(vec, width);
            intermediate_accumulators.push(accumulator);
            traces.push(trace);
        }
        drop(_g);
        (traces, intermediate_accumulators)
    }
}

/// Incremental, allocation-free constructor for [`LookupValues`].
///
/// Rows start zeroed — multiplicity zero and zero arguments in every slot —
/// and are filled in place through the parallel row writers of
/// [`Self::rows_mut`]. Values are written directly into the final flat
/// storage, so no per-lookup allocation ever happens.
pub struct LookupValuesBuilder<F> {
    height: usize,
    num_lookups: usize,
    arg_offsets: Vec<usize>,
    /// Row stride of `args`: the total argument width, with a minimum of 1
    /// so row chunking stays well-defined when every slot is argument-less.
    row_stride: usize,
    multiplicities: Vec<F>,
    args: Vec<F>,
}

impl<F: Field> LookupValuesBuilder<F> {
    /// `slot_arg_widths[j]` is the maximum number of arguments of lookup
    /// slot `j`. Rows writing fewer arguments leave zero padding, which does
    /// not change the message fingerprint (a Horner evaluation).
    pub fn new(height: usize, slot_arg_widths: &[usize]) -> Self {
        let num_lookups = slot_arg_widths.len();
        let mut arg_offsets = Vec::with_capacity(num_lookups + 1);
        arg_offsets.push(0);
        for width in slot_arg_widths {
            arg_offsets.push(arg_offsets.last().unwrap() + width);
        }
        let row_stride = (*arg_offsets.last().unwrap()).max(1);
        Self {
            height,
            num_lookups,
            arg_offsets,
            row_stride,
            multiplicities: F::zero_vec(height * num_lookups),
            args: F::zero_vec(height * row_stride),
        }
    }

    /// Row writers, one per row, in row order. Returned as a `Vec` so the
    /// caller can parallelize with its own runtime (e.g. rayon's
    /// `into_par_iter`, zipped with a parallel iterator over trace rows)
    /// regardless of how this crate's `parallel` feature is set.
    ///
    /// # Panics
    /// Panics if the builder has no lookup slots (nothing to write; call
    /// [`Self::finish`] directly).
    pub fn rows_mut(&mut self) -> Vec<LookupRowMut<'_, F>> {
        assert!(self.num_lookups > 0, "builder has no lookup slots");
        let Self {
            num_lookups,
            arg_offsets,
            row_stride,
            multiplicities,
            args,
            ..
        } = self;
        let arg_offsets = arg_offsets.as_slice();
        multiplicities
            .chunks_exact_mut(*num_lookups)
            .zip(args.chunks_exact_mut(*row_stride))
            .map(|(multiplicities, args)| LookupRowMut {
                multiplicities,
                args,
                arg_offsets,
            })
            .collect()
    }

    /// Finalizes into [`LookupValues`].
    pub fn finish(mut self) -> LookupValues<F> {
        if self.arg_offsets[self.num_lookups] == 0 {
            // Every slot is argument-less; drop the dummy stride storage.
            self.args = vec![];
        }
        LookupValues {
            height: self.height,
            num_lookups: self.num_lookups,
            multiplicities: self.multiplicities,
            arg_offsets: self.arg_offsets,
            args: self.args,
        }
    }
}

/// Mutable view of one row of a [`LookupValuesBuilder`].
pub struct LookupRowMut<'a, F> {
    multiplicities: &'a mut [F],
    args: &'a mut [F],
    arg_offsets: &'a [usize],
}

impl<F: Field> LookupRowMut<'_, F> {
    /// Writes lookup slot `slot` with "push" semantics (adds a claim).
    /// Fewer arguments than the slot's width leave zero padding.
    ///
    /// # Panics
    /// Panics if `args` exceeds the slot's width.
    #[inline]
    pub fn push(&mut self, slot: usize, multiplicity: F, args: &[F]) {
        let start = self.arg_offsets[slot];
        let end = self.arg_offsets[slot + 1];
        assert!(
            args.len() <= end - start,
            "too many arguments for lookup slot {slot}"
        );
        self.multiplicities[slot] = multiplicity;
        self.args[start..start + args.len()].copy_from_slice(args);
        self.args[start + args.len()..end].fill(F::ZERO);
    }

    /// Writes lookup slot `slot` with "pull" semantics (removes a claim):
    /// same as [`Self::push`] with negated multiplicity.
    #[inline]
    pub fn pull(&mut self, slot: usize, multiplicity: F, args: &[F]) {
        self.push(slot, -multiplicity, args);
    }
}
