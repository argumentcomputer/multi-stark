//! Lookups: the [`Lookup`] type, the direct logUp constraint evaluation, and
//! the concrete lookup witness ([`LookupValues`]) plus its stage-2 trace
//! construction.
//!
//! The logUp constraints are protocol machinery, not user data, so they are
//! NOT compiled into the circuit graph: [`logup_constraint_values`] evaluates
//! them directly — in `PackedVal` on the prover's quotient domain and in the
//! challenge field at ζ for the verifier — and their values are folded after
//! the user-constraint roots in the canonical protocol order.
//! [`synthesize_lookups`] remains as the executable specification the direct
//! evaluation is pinned against in tests.
//!
//! # Layout conventions (owned here)
//!
//! - **Publics** (each an extension value, stored as `d` base coordinates):
//!   slot 0 = β (lookup challenge), 1 = γ (fingerprint challenge),
//!   2 = the accumulator entering the circuit (`acc_initial`), 3 = the
//!   accumulator leaving it (`acc_final`). So `num_publics = 4·d`.
//! - **Stage-2 columns** (extension values, flattened to base): slot `j` =
//!   the partial accumulator entering lookup `j`'s chained step (a single
//!   pass-through accumulator when the circuit has no lookups). So
//!   `stage2_width = max(L, 1)·d` base columns. No message inverses are
//!   committed — see `synthesize_lookups` for how the chained-accumulator
//!   argument works. The committed accumulator is gauge-free (only
//!   differences are constrained); the prover starts it at zero.

use p3_field::{
    Algebra, ExtensionField, Field, PrimeCharacteristicRing, batch_multiplicative_inverse,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::expr::{Expr, ExtExpr, RowOffset};

/// A lookup: a multiplicity and a vector of arguments. `E` is a frontend
/// expression in a [`crate::system::CircuitInputs`] and a node id once
/// compiled.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Lookup<E> {
    pub multiplicity: E,
    pub args: Vec<E>,
}

impl<E> Lookup<E> {
    /// Returns a [`Lookup`] with multiplicity zero and no arguments.
    #[inline]
    pub fn empty() -> Self
    where
        E: PrimeCharacteristicRing,
    {
        Self {
            multiplicity: E::ZERO,
            args: vec![],
        }
    }

    /// "Pushing" has the semantics of adding a claim to the claim set.
    #[inline]
    pub fn push(multiplicity: E, args: Vec<E>) -> Self {
        Self { multiplicity, args }
    }

    /// "Pulling" has the semantics of removing a claim from the claim set.
    #[inline]
    pub fn pull(multiplicity: E, args: Vec<E>) -> Self
    where
        E: std::ops::Neg<Output = E>,
    {
        Self {
            multiplicity: -multiplicity,
            args,
        }
    }
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
/// at extension degree `d`: one partial-accumulator column per lookup
/// (`acc_j` entering lookup `j`'s step), or a single pass-through
/// accumulator column when the circuit has no lookups.
pub fn stage2_width(num_lookups: usize, d: usize) -> usize {
    num_lookups.max(1) * d
}

/// Number of base-field constraint values the logUp argument contributes:
/// one chained-accumulator step per lookup (or the single pass-through
/// constraint when there are none), each expanded into `d` coordinates.
pub fn logup_constraint_count(num_lookups: usize, d: usize) -> usize {
    num_lookups.max(1) * d
}

/// Coordinate product in the binomial extension `X^d = w`, schoolbook:
/// `out[k] = Σ_{i+j=k} a_i·b_j + w · Σ_{i+j=k+d} a_i·b_j`. Generic-degree
/// fallback; the hot D=2 path uses [`mul2`].
fn coord_mul<F: Field, A: Algebra<F> + Copy>(a: &[A], b: &[A], w: F) -> Vec<A> {
    let d = a.len();
    let mut out = vec![A::ZERO; d];
    for i in 0..d {
        for j in 0..d {
            let prod = a[i] * b[j];
            if i + j < d {
                out[i + j] += prod;
            } else {
                out[i + j - d] += prod * w;
            }
        }
    }
    out
}

/// Degree-2 coordinate product `X² = w`, Karatsuba (3 multiplications,
/// matching the compiled graph's expansion): allocation-free.
#[inline]
fn mul2<F: Field, A: Algebra<F> + Copy>(a: (A, A), b: (A, A), w: F) -> (A, A) {
    let v0 = a.0 * b.0;
    let v1 = a.1 * b.1;
    let cross = (a.0 + a.1) * (b.0 + b.1) - v0 - v1;
    (v0 + v1 * w, cross)
}

/// Directly evaluates the logUp constraint values at one evaluation context,
/// in the canonical protocol order: for each lookup (in order) the `d`
/// coordinates of its chained-accumulator step — interior steps
/// `m_j·(acc_{j+1} − acc_j) − mult_j`, the wrap step carrying the
/// `is_last_row·Δ` boundary injection (see [`synthesize_lookups`] for the full
/// derivation). Semantically identical to compiling the synthesized
/// constraints and evaluating their roots (see the pin test), without
/// materializing them.
///
/// Everything is base-field coordinate arithmetic, so the working type `A`
/// is generic: `PackedVal` on the prover's quotient domain, the challenge
/// field at ζ for the verifier — both sides share this one implementation.
///
/// Layout contracts (see the module docs): `publics` holds the 4·d base
/// coordinates of (β, γ, acc_initial, acc_final); `stage2` / `stage2_next`
/// are the flattened stage-2 base columns, slot `j` the partial accumulator
/// entering lookup `j`'s step. `is_last_row` is the NORMALIZED last-row
/// selector value. `lookups` carries node ids into `node_vals` (multiplicity
/// and args embed in coordinate 0).
#[allow(clippy::too_many_arguments)]
pub fn logup_constraint_values<F: Field, A: Algebra<F> + Copy>(
    lookups: &[Lookup<crate::graph::NodeId>],
    node_vals: &[A],
    stage2: &[A],
    stage2_next: &[A],
    publics: &[A],
    is_last_row: A,
    w: F,
    d: usize,
    out: &mut Vec<A>,
) {
    // Allocation-free Karatsuba fast path for the reference degree-2
    // extension; this runs per point-packet on the prover's quotient
    // domain, so it must not touch the heap.
    if d == 2 {
        let nv = |id: crate::graph::NodeId| node_vals[id.index()];
        let beta = (publics[0], publics[1]);
        let gamma = (publics[2], publics[3]);
        // The boundary injection: is_last_row·(acc_final − acc_initial).
        let inj = (
            is_last_row * (publics[6] - publics[4]),
            is_last_row * (publics[7] - publics[5]),
        );

        if lookups.is_empty() {
            // Pass-through accumulator: acc′ − acc + is_last_row·Δ = 0.
            out.push(stage2_next[0] - stage2[0] + inj.0);
            out.push(stage2_next[1] - stage2[1] + inj.1);
            return;
        }

        let last = lookups.len() - 1;
        for (j, lookup) in lookups.iter().enumerate() {
            let source = (stage2[2 * j], stage2[2 * j + 1]);
            let target = if j < last {
                (stage2[2 * j + 2], stage2[2 * j + 3])
            } else {
                (stage2_next[0] + inj.0, stage2_next[1] + inj.1)
            };
            // fingerprint = Σ_i args[i] · γ^i, Horner over the reversed args.
            let mut f = (A::ZERO, A::ZERO);
            for &arg in lookup.args.iter().rev() {
                f = mul2(f, gamma, w);
                f.0 += nv(arg);
            }
            // step: (β + fingerprint)·(target − source) − mult = 0.
            let c = mul2(
                (f.0 + beta.0, f.1 + beta.1),
                (target.0 - source.0, target.1 - source.1),
                w,
            );
            out.push(c.0 - nv(lookup.multiplicity));
            out.push(c.1);
        }
        return;
    }

    let beta = &publics[..d];
    let gamma = &publics[d..2 * d];
    let acc_initial = &publics[2 * d..3 * d];
    let acc_final = &publics[3 * d..4 * d];
    // The boundary injection: is_last_row·(acc_final − acc_initial).
    let inj: Vec<A> = acc_final
        .iter()
        .zip(acc_initial)
        .map(|(&f, &i)| is_last_row * (f - i))
        .collect();

    if lookups.is_empty() {
        for k in 0..d {
            out.push(stage2_next[k] - stage2[k] + inj[k]);
        }
        return;
    }

    let last = lookups.len() - 1;
    for (j, lookup) in lookups.iter().enumerate() {
        let source = &stage2[j * d..(j + 1) * d];
        let mut target: Vec<A> = if j < last {
            stage2[(j + 1) * d..(j + 2) * d].to_vec()
        } else {
            stage2_next[..d]
                .iter()
                .zip(&inj)
                .map(|(&t, &i)| t + i)
                .collect()
        };
        for (t, &s) in target.iter_mut().zip(source) {
            *t -= s;
        }

        // fingerprint = Σ_i args[i] · γ^i, via Horner over the reversed args
        // (base values embed in coordinate 0).
        let mut fingerprint = vec![A::ZERO; d];
        for &arg in lookup.args.iter().rev() {
            fingerprint = coord_mul(&fingerprint, gamma, w);
            fingerprint[0] += node_vals[arg.index()];
        }

        // step: (β + fingerprint)·(target − source) − mult = 0.
        let mut message = fingerprint;
        for (m, &b) in message.iter_mut().zip(beta) {
            *m += b;
        }
        let mut constraint = coord_mul(&message, &target, w);
        constraint[0] -= node_vals[lookup.multiplicity.index()];
        out.extend_from_slice(&constraint);
    }
}

/// The maximum degree multiple over the logUp constraints, computed
/// analytically from the compiled lookup-expression degrees (mirroring the
/// graph compiler's rules: columns degree 1, publics/IsTransition 0,
/// IsFirstRow/IsLastRow 1, add = max, mul = sum).
pub fn logup_max_degree<F: Field>(graph: &crate::graph::ConstraintGraph<F>) -> u32 {
    let node_degree = |id: crate::graph::NodeId| graph.degrees[id.index()];
    // Every step is m_j·(accumulator difference) − mult_j, and the
    // difference — including the wrap step's is_last_row·Δ injection
    // (selector·publics) — has degree multiple 1. So per lookup:
    // max(max arg degree + 1, mult degree); a lookup-free circuit's
    // pass-through constraint is degree 1.
    graph
        .lookups
        .iter()
        .map(|l| {
            let message_degree = l.args.iter().map(|&a| node_degree(a)).max().unwrap_or(0);
            (message_degree + 1).max(node_degree(l.multiplicity))
        })
        .max()
        .unwrap_or(1)
}

/// Builds the chained-accumulator logUp constraints for `lookups` at
/// extension degree `d` (the executable specification that
/// [`logup_constraint_values`] is pinned against in tests; it is not
/// compiled into the circuit graph).
///
/// # How the chained-accumulator argument works
///
/// Stage 2 commits one partial accumulator per lookup: `acc_j` (slot `j`)
/// is the running sum entering lookup `j`'s step on that row. Each step
/// asserts `acc_{j+1} − acc_j = mult_j / m_j`, multiplied through by the
/// message `m_j = β + fingerprint(γ, args_j)` to stay polynomial:
///
/// ```text
/// interior (j < L−1):  m_j · (acc_{j+1} − acc_j) − mult_j = 0
/// wrap     (j = L−1):  m_j · (acc_0′ − acc_j + is_last_row·Δ) − mult_j = 0
/// ```
///
/// where `acc_0′` is the NEXT row's first slot, `Δ = acc_final −
/// acc_initial` (publics slots 3 and 2), and `is_last_row` is the NORMALIZED
/// last-row Lagrange selector (value exactly 1 on the last row — see the
/// selector-normalization pin test).
///
/// The wrap step needs no transition selector: on the trace subgroup
/// "next row" is rotation by the generator, so the last row's `acc_0′` IS
/// the first row's `acc_0` and the chain is a cycle. Telescoping the step
/// constraints around that cycle cancels every accumulator term, leaving
/// `Σ_rows Σ_j mult_j/m_j = Δ` — the injected `is_last_row·Δ` perturbs
/// exactly one link, converting "the multiset sum is zero" into "the
/// multiset sum is the public accumulator difference". No first-row,
/// transition, or last-row constraints remain: the committed accumulator
/// is gauge-free (only differences are constrained; the prover starts it
/// at zero by convention).
///
/// A circuit with no lookups gets a single pass-through column with
/// `acc′ − acc + is_last_row·Δ = 0`, forcing `Δ = 0`.
///
/// Zero-multiplicity rows (padding) force `acc_{j+1} = acc_j` through a
/// (whp) nonzero message — the accumulators simply ride through.
///
/// Degrees are uniform: every constraint is `max(deg(args)+1, deg(mult))`
/// (the injected term is selector·publics, degree multiple 1, absorbed by
/// `add = max`).
pub fn synthesize_lookups<F: Field>(lookups: &[Lookup<Expr<F>>], d: usize) -> Vec<ExtExpr<F>> {
    let d = u32::try_from(d).expect("extension degree exceeds u32");
    let beta = ExtExpr::public(0, d);
    let gamma = ExtExpr::public(1, d);
    let acc_initial = ExtExpr::public(2, d);
    let acc_final = ExtExpr::public(3, d);
    // The boundary injection: is_last_row·Δ. `Expr::IsLastRow` is the
    // normalized last-row selector protocol-wide.
    let injection = ExtExpr::from(Expr::IsLastRow) * (acc_final - acc_initial);

    if lookups.is_empty() {
        // Pass-through accumulator: acc′ − acc + is_last_row·Δ = 0.
        return vec![
            ExtExpr::stage2(0, d, RowOffset::Next) - ExtExpr::stage2(0, d, RowOffset::Current)
                + injection,
        ];
    }

    let last = lookups.len() - 1;
    let mut constraints = Vec::with_capacity(lookups.len());
    for (j, lookup) in lookups.iter().enumerate() {
        let slot = u32::try_from(j).expect("lookup count exceeds u32");
        let source = ExtExpr::stage2(slot, d, RowOffset::Current);
        let target = if j < last {
            ExtExpr::stage2(slot + 1, d, RowOffset::Current)
        } else {
            ExtExpr::stage2(0, d, RowOffset::Next) + injection.clone()
        };

        // fingerprint = Σ_i args[i] · γ^i, via Horner over the reversed args.
        let mut coeffs = lookup.args.iter().rev();
        let mut fingerprint = match coeffs.next() {
            Some(arg) => ExtExpr::from(arg.clone()),
            None => ExtExpr::from(Expr::constant(F::ZERO)),
        };
        for arg in coeffs {
            fingerprint = fingerprint * gamma.clone() + arg.clone();
        }

        // message = β + fingerprint; step: m·(target − source) − mult = 0.
        let message = beta.clone() + fingerprint;
        constraints.push(message * (target - source) - lookup.multiplicity.clone());
    }

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
                // Pass-through accumulator column; the committed values are
                // gauge-free (only differences are constrained), zero by
                // convention.
                vec![EF::ZERO; circuit.height]
            } else {
                // One partial accumulator per lookup slot: `acc_j` is the
                // running sum ENTERING lookup j's step; the last step's
                // result carries into the next row's slot 0. Single pass,
                // one multiply-add per (row, lookup). The column is
                // gauge-free, so it starts at zero regardless of the global
                // accumulator; the circuit's total contribution is added to
                // the global chain at the end.
                let mut vec = Vec::with_capacity(circuit.height * circuit.num_lookups);
                let mut local = EF::ZERO;
                for (row_multiplicities, row_messages_inverses) in circuit
                    .multiplicities
                    .chunks_exact(circuit.num_lookups)
                    .zip(circuit_messages_inverses.chunks_exact(circuit.num_lookups))
                {
                    for (&multiplicity, &message_inverse) in
                        row_multiplicities.iter().zip(row_messages_inverses)
                    {
                        vec.push(local);
                        local += EF::from(multiplicity) * message_inverse;
                    }
                }
                accumulator += local;
                vec
            };
            let width = circuit.num_lookups.max(1);
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

#[cfg(test)]
mod tests {
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_field::Field;

    use crate::{
        p3_adapter::{LookupAir, SymbolicExpression, var},
        system::{ProverKey, System, SystemWitness},
        types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val},
    };

    use super::*;

    /// The selector `p3` provides is the UNNORMALIZED Lagrange numerator
    /// `is_last_row(x) = (x^n − 1)/(x − g^{−1})`; its value at the last row
    /// is `n·g` (the zerofier derivative there). The chained logUp wrap
    /// constraint uses the selector additively, so it must be normalized to
    /// value exactly 1 at the last row: `is_last_row/(n·g)` has value 1 there.
    /// This pins that constant (and `1/n` for the first row) against the
    /// textbook Lagrange basis product, at arbitrary points, several sizes.
    #[test]
    fn selector_normalization_constants() {
        use crate::config::StarkGenericConfig;
        use crate::types::ExtVal;
        use p3_commit::PolynomialSpace;
        use p3_field::{PrimeCharacteristicRing, TwoAdicField};

        let config = GoldilocksBlake3Config::new(
            CommitmentParameters {
                log_blowup: 1,
                cap_height: 0,
            },
            FriParameters {
                log_final_poly_len: 0,
                max_log_arity: 1,
                num_queries: 1,
                commit_proof_of_work_bits: 0,
                query_proof_of_work_bits: 0,
            },
        );
        for log_n in [2usize, 3, 5, 8] {
            let n = 1usize << log_n;
            let g = Val::two_adic_generator(log_n);
            let domain: crate::types::Domain = <crate::types::Pcs as p3_commit::Pcs<
                ExtVal,
                crate::types::Challenger,
            >>::natural_domain_for_degree(
                config.pcs(), n
            );
            for seed in [3u64, 12345, 0xdead_beef] {
                let zeta: ExtVal =
                    ExtVal::from_u64(seed).exp_u64(7) + ExtVal::from_u64(seed * 31 + 1);
                let sels = domain.selectors_at_point(zeta);
                // Textbook Lagrange basis of the last row:
                // Π_{i≠n−1} (ζ − g^i)/(g^{n−1} − g^i).
                let last = g.exp_u64((n - 1) as u64);
                let mut ref_last = ExtVal::ONE;
                for i in 0..n - 1 {
                    let gi = g.exp_u64(i as u64);
                    ref_last *= (zeta - ExtVal::from(gi)) * ExtVal::from(last - gi).inverse();
                }
                let norm = (Val::from_usize(n) * g).inverse();
                assert_eq!(
                    sels.is_last_row * ExtVal::from(norm),
                    ref_last,
                    "last-row normalization, log_n={log_n}"
                );
                let mut ref_first = ExtVal::ONE;
                for i in 1..n {
                    let gi = g.exp_u64(i as u64);
                    ref_first *= (zeta - ExtVal::from(gi)) * ExtVal::from(Val::ONE - gi).inverse();
                }
                assert_eq!(
                    sels.is_first_row * ExtVal::from(Val::from_usize(n).inverse()),
                    ref_first,
                    "first-row normalization, log_n={log_n}"
                );
            }
        }
    }

    /// Pin: `logup_constraint_values` (the direct evaluation the prover and
    /// verifier run) computes exactly the values of the constraints
    /// `synthesize_lookups` specifies, coordinate for coordinate, in order —
    /// checked against the reference `ExtExpr` evaluator at pseudo-random
    /// points. This is the executable mirror contract between the two.
    #[test]
    fn direct_logup_matches_synthesized_reference() {
        use crate::eval::{VarValues, eval_expr, eval_ext_expr};
        use crate::graph::ExtensionParams;
        use p3_field::PrimeCharacteristicRing;

        let params = crate::system::extension_params::<GoldilocksBlake3Config>();
        let (w, d) = (params.w, params.degree);

        // Lookups with assorted shapes: multi-arg with a product, single
        // arg, and the degenerate empty-args case.
        let lookups = vec![
            Lookup::push(
                Expr::main(0),
                vec![
                    Expr::constant(Val::from_u32(7)),
                    Expr::main(1),
                    Expr::main(2) * Expr::main(3),
                ],
            ),
            Lookup::pull(Expr::main(4), vec![Expr::main(5)]),
            Lookup {
                multiplicity: Expr::constant(Val::ONE),
                args: vec![],
            },
        ];
        let synthesized = synthesize_lookups(&lookups, d);

        // Deterministic pseudo-random base-field values (an equality of
        // polynomials checked at random points).
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            Val::from_u64(seed >> 8)
        };
        let main_cur: Vec<Val> = (0..6).map(|_| next()).collect();
        let main_next: Vec<Val> = (0..6).map(|_| next()).collect();
        let s2_width = stage2_width(lookups.len(), d);
        let s2_cur: Vec<Val> = (0..s2_width).map(|_| next()).collect();
        let s2_next: Vec<Val> = (0..s2_width).map(|_| next()).collect();
        let publics: Vec<Val> = (0..num_publics(d)).map(|_| next()).collect();
        let (isf, isl, ist) = (next(), next(), next());

        let empty: [Val; 0] = [];
        let view = VarValues {
            preprocessed: [&empty, &empty],
            main: [&main_cur, &main_next],
            stage2: [&s2_cur, &s2_next],
            publics: &publics,
            is_first_row: isf,
            is_last_row: isl,
            is_transition: ist,
        };

        let ref_params = ExtensionParams {
            degree: d,
            w,
            karatsuba: false,
        };
        let mut reference: Vec<Val> = vec![];
        for constraint in &synthesized {
            reference.extend(eval_ext_expr(constraint, &view, &ref_params));
        }

        // The direct evaluator reads lookup expressions out of a node-value
        // buffer by id; build a synthetic buffer with consecutive ids.
        let mut node_vals: Vec<Val> = vec![];
        let mut fresh = |v: Val| {
            node_vals.push(v);
            crate::graph::NodeId(u32::try_from(node_vals.len() - 1).unwrap())
        };
        let lookup_ids: Vec<Lookup<crate::graph::NodeId>> = lookups
            .iter()
            .map(|l| Lookup {
                multiplicity: fresh(eval_expr(&l.multiplicity, &view)),
                args: l.args.iter().map(|a| fresh(eval_expr(a, &view))).collect(),
            })
            .collect();
        let mut direct: Vec<Val> = vec![];
        logup_constraint_values(
            &lookup_ids,
            &node_vals,
            &s2_cur,
            &s2_next,
            &publics,
            // the (normalized) last-row selector value; the
            // reference evaluation reads the same value through the view's
            // `is_last_row`, so any value pins the identity.
            isl,
            w,
            d,
            &mut direct,
        );

        assert_eq!(direct.len(), logup_constraint_count(lookups.len(), d));
        assert_eq!(direct, reference);
    }

    enum CS {
        Even,
        Odd,
        /// A circuit on its own channel that nothing references; used by the
        /// sparse-activation tests as the deactivated table.
        Dead,
    }
    impl<F> BaseAir<F> for CS {
        fn width(&self) -> usize {
            6
        }
    }
    impl CS {
        fn lookups(&self) -> Vec<Lookup<SymbolicExpression<Val>>> {
            // provide removes multiplicity
            let multiplicity = var(0);
            let input = var(1);
            let input_is_zero = var(3);
            let input_not_zero = var(4);
            let recursion_output = var(5);
            let even_index = Val::ZERO.into();
            let odd_index = Val::ONE.into();
            let one: SymbolicExpression<_> = Val::ONE.into();
            match self {
                Self::Even => vec![
                    Lookup::pull(
                        multiplicity,
                        vec![
                            even_index,
                            input.clone(),
                            input_not_zero.clone() * recursion_output.clone() + input_is_zero,
                        ],
                    ),
                    Lookup::push(
                        input_not_zero,
                        vec![odd_index, input - one, recursion_output],
                    ),
                ],
                Self::Odd => vec![
                    Lookup::pull(
                        multiplicity,
                        vec![
                            odd_index,
                            input.clone(),
                            input_not_zero.clone() * recursion_output.clone(),
                        ],
                    ),
                    Lookup::push(
                        input_not_zero,
                        vec![even_index, input - one, recursion_output],
                    ),
                ],
                Self::Dead => vec![Lookup::pull(
                    multiplicity,
                    vec![Val::from_u32(2).into(), input],
                )],
            }
        }
    }
    impl<AB> Air<AB> for CS
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            // both even and odd have the same constraints, they only differ on the lookups
            let main = builder.main();
            let local = main.current_slice();
            let multiplicity = local[0];
            let input = local[1];
            let input_inverse = local[2];
            let input_is_zero = local[3];
            let input_not_zero = local[4];
            builder.assert_bools([input_is_zero, input_not_zero]);
            builder
                .when(multiplicity)
                .assert_one(input_is_zero + input_not_zero);
            builder.when(input_is_zero).assert_zero(input);
            builder
                .when(input_not_zero)
                .assert_one(input * input_inverse);
        }
    }
    const COMMITMENT_PARAMETERS: CommitmentParameters = CommitmentParameters {
        log_blowup: 1,
        cap_height: 0,
    };

    const FRI_PARAMETERS: FriParameters = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 64,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };

    fn system() -> (
        System<GoldilocksBlake3Config>,
        ProverKey<GoldilocksBlake3Config>,
    ) {
        let config = GoldilocksBlake3Config::new(COMMITMENT_PARAMETERS, FRI_PARAMETERS);
        let even = LookupAir::new(CS::Even, CS::Even.lookups());
        let odd = LookupAir::new(CS::Odd, CS::Odd.lookups());
        System::new(config, [even, odd])
    }

    fn witness_traces() -> Vec<RowMajorMatrix<Val>> {
        let f = Val::from_u32;
        #[rustfmt::skip]
        let traces = vec![
            RowMajorMatrix::new(
                vec![
                    // row 1
                    f(1), f(4), f(4).inverse(), f(0), f(1), f(1),
                    // row 2
                    f(1), f(2), f(2).inverse(), f(0), f(1), f(1),
                    // row 3
                    f(1), f(0), f(0), f(1), f(0), f(0),
                    // row 4
                    f(0), f(0), f(0), f(0), f(0), f(0),
                ],
                6,
            ),
            RowMajorMatrix::new(
                vec![
                    // row 1
                    f(1), f(3), f(3).inverse(), f(0), f(1), f(1),
                    // row 2
                    f(1), f(1), f(1).inverse(), f(0), f(1), f(1),
                    // row 3
                    f(0), f(0), f(0), f(0), f(0), f(0),
                    // row 4
                    f(0), f(0), f(0), f(0), f(0), f(0),
                ],
                6,
            ),
        ];
        traces
    }

    fn witness(system: &System<GoldilocksBlake3Config>) -> SystemWitness<Val> {
        SystemWitness::from_stage_1(witness_traces(), system)
    }

    /// The builder must produce the exact same flat storage as the nested
    /// conversion, including zero padding for short and empty slots.
    #[test]
    fn builder_matches_from_rows() {
        let f = Val::from_u32;
        let rows = vec![
            vec![
                Lookup::push(f(1), vec![f(2), f(3)]),
                Lookup::pull(f(4), vec![f(5), f(6), f(7)]),
            ],
            vec![Lookup::empty(), Lookup::push(f(8), vec![f(9)])],
        ];
        let a = LookupValues::from_rows(rows);
        let mut builder = LookupValues::builder(2, &[2, 3]);
        for (i, row) in builder.rows_mut().iter_mut().enumerate() {
            if i == 0 {
                row.push(0, f(1), &[f(2), f(3)]);
                row.pull(1, f(4), &[f(5), f(6), f(7)]);
            } else {
                row.push(1, f(8), &[f(9)]);
            }
        }
        let b = builder.finish();
        assert_eq!(a.height, b.height);
        assert_eq!(a.num_lookups, b.num_lookups);
        assert_eq!(a.arg_offsets, b.arg_offsets);
        assert_eq!(a.multiplicities, b.multiplicities);
        assert_eq!(a.args, b.args);
    }

    #[test]
    fn lookup_test() {
        let (system, key) = system();
        let witness = witness(&system);
        let f = Val::from_u32;
        let claim = &[f(0), f(4), f(1)];
        let proof = system.prove(&key, claim, witness);
        system.verify(claim, &proof).unwrap();
    }

    /// A three-circuit system where the third circuit (`Dead`, its own
    /// channel, referenced by nothing) gets an empty trace: the prover must
    /// deactivate it, the proof must carry per-circuit data for the two
    /// active circuits only, and verification must accept.
    #[test]
    fn sparse_inactive_circuit_accepted() {
        let config = GoldilocksBlake3Config::new(COMMITMENT_PARAMETERS, FRI_PARAMETERS);
        let even = LookupAir::new(CS::Even, CS::Even.lookups());
        let odd = LookupAir::new(CS::Odd, CS::Odd.lookups());
        let dead = LookupAir::new(CS::Dead, CS::Dead.lookups());
        let (system, key) = System::new(config, [even, odd, dead]);
        let mut witness = witness_traces();
        witness.push(RowMajorMatrix::new(vec![], 6));
        let witness = SystemWitness::from_stage_1(witness, &system);
        let f = Val::from_u32;
        let claim = &[f(0), f(4), f(1)];
        let proof = system.prove(&key, claim, witness);
        assert_eq!(proof.active, vec![true, true, false]);
        assert_eq!(proof.log_degrees.len(), 2);
        assert_eq!(proof.intermediate_accumulators.len(), 2);
        assert_eq!(proof.stage_1_opened_values.len(), 2);
        system.verify(claim, &proof).unwrap();
    }

    /// Tampering with the activation bitmap must be rejected: activating a
    /// circuit the proof carries no data for, or deactivating one it does.
    #[test]
    fn sparse_bitmap_tamper_rejected() {
        let config = GoldilocksBlake3Config::new(COMMITMENT_PARAMETERS, FRI_PARAMETERS);
        let even = LookupAir::new(CS::Even, CS::Even.lookups());
        let odd = LookupAir::new(CS::Odd, CS::Odd.lookups());
        let dead = LookupAir::new(CS::Dead, CS::Dead.lookups());
        let (system, key) = System::new(config, [even, odd, dead]);
        let mut witness = witness_traces();
        witness.push(RowMajorMatrix::new(vec![], 6));
        let witness = SystemWitness::from_stage_1(witness, &system);
        let f = Val::from_u32;
        let claim = &[f(0), f(4), f(1)];
        let mut proof = system.prove(&key, claim, witness);
        proof.active[2] = true;
        assert!(system.verify(claim, &proof).is_err());
        proof.active[2] = false;
        proof.active[1] = false;
        assert!(system.verify(claim, &proof).is_err());
    }

    /// Deactivating a circuit the claim's execution NEEDS (here: emptying the
    /// Odd table while Even still pushes into the odd channel) must leave the
    /// lookup accumulator unbalanced and be rejected.
    #[test]
    fn sparse_needed_circuit_rejected() {
        let (system, key) = system();
        let mut traces = witness_traces();
        // Empty the Odd table: Even's pushes into the odd channel and the
        // claim's pull can no longer be matched.
        traces[1] = RowMajorMatrix::new(vec![], 6);
        let witness = SystemWitness::from_stage_1(traces, &system);
        let f = Val::from_u32;
        let claim = &[f(0), f(4), f(1)];
        let proof = system.prove(&key, claim, witness);
        assert_eq!(proof.active, vec![true, false]);
        assert!(system.verify(claim, &proof).is_err());
    }

    #[test]
    fn test_claim_split_rejected() {
        let (system, key) = system();
        let witness = witness(&system);
        let f = Val::from_u32;
        // Prove a single claim, then attempt to verify with the same values
        // regrouped into two claims. The transcript is length-prefixed, so the
        // regrouped claims must yield different challenges and fail.
        let claim = &[f(0), f(4), f(1)];
        let proof = system.prove(&key, claim, witness);
        let split_claims: &[&[Val]] = &[&[f(0), f(4)], &[f(1)]];
        let result = system.verify_multiple_claims(split_claims, &proof);
        assert!(result.is_err());
    }
}
