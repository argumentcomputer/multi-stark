use p3_air::{Air, BaseAir, ExtensionBuilder, WindowAccess};
use p3_field::{batch_multiplicative_inverse, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;

use crate::{
    builder::{symbolic::SymbolicExpression, TwoStagedBuilder},
    types::{ExtVal, Val},
};

/// Per-thread chunk size for the fused denom-build / batch-invert / row-sum
/// step in [`stage_2_traces`]. Sized to keep the per-chunk working set
/// (`CHUNK × K × (sizeof(ExtVal) + sizeof(Val))`) comfortably in L2 across
/// typical circuit widths, and matches the internal block size used by
/// `batch_multiplicative_inverse`.
const STAGE_2_INVERT_CHUNK_ROWS: usize = 1024;

/// Each circuit is required to have 4 arguments for the second stage. Namely,
/// the lookup challenge, fingerprint challenge, current accumulator and next
/// accumulator.
pub const LOOKUP_PUBLIC_SIZE: usize = 4;

#[derive(Clone)]
pub struct Lookup<Expr> {
    pub multiplicity: Expr,
    pub args: Vec<Expr>,
}

impl<Expr> Lookup<Expr> {
    /// Returns a [`Lookup`] with multiplicity zero and no arguments.
    #[inline]
    pub fn empty() -> Self
    where
        Expr: PrimeCharacteristicRing,
    {
        Self {
            multiplicity: Expr::ZERO,
            args: vec![],
        }
    }

    /// "Pushing" has the semantics of adding a claim to the claim set.
    #[inline]
    pub fn push(multiplicity: Expr, args: Vec<Expr>) -> Self {
        Self { multiplicity, args }
    }

    /// "Pulling" has the semantics of removing a claim from the claim set.
    #[inline]
    pub fn pull(multiplicity: Expr, args: Vec<Expr>) -> Self
    where
        Expr: std::ops::Neg<Output = Expr>,
    {
        Self {
            multiplicity: -multiplicity,
            args,
        }
    }
}

pub struct LookupAir<A> {
    pub inner_air: A,
    pub lookups: Vec<Lookup<SymbolicExpression<Val>>>,
    pub preprocessed: Option<RowMajorMatrix<Val>>,
}

impl<A: BaseAir<Val>> LookupAir<A> {
    pub fn new(inner_air: A, lookups: Vec<Lookup<SymbolicExpression<Val>>>) -> Self {
        let preprocessed = inner_air.preprocessed_trace();
        Self {
            inner_air,
            lookups,
            preprocessed,
        }
    }

    /// One column for the accumulator and one column for the inverse of the
    /// message associated with each lookup.
    pub fn stage_2_width(&self) -> usize {
        1 + self.lookups.len()
    }
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

/// References needed to build one circuit's stage 2 trace without
/// materializing per-row `Lookup<Val>` values: the main trace, the
/// preprocessed trace (if any), and the AIR-side symbolic lookups.
pub struct CircuitWitness<'a> {
    pub main: &'a RowMajorMatrix<Val>,
    pub preprocessed: Option<&'a RowMajorMatrix<Val>>,
    pub lookups: &'a [Lookup<SymbolicExpression<Val>>],
}

impl Lookup<SymbolicExpression<Val>> {
    /// Build all circuits' stage 2 traces in a single pass, resolving
    /// concrete lookup values directly from each row of the main trace.
    ///
    /// For each circuit:
    /// 1. Chunked fused loop over rows. Per chunk, denominators
    ///    `lookup_challenge + fingerprint(γ, args)` and multiplicities are
    ///    built from the symbolic lookups, batch-inverted in place, and the
    ///    inverses are written directly into stage 2 columns 1..=K. A
    ///    per-row partial sum is accumulated into `row_sums`.
    /// 2. Three-phase parallel exclusive prefix scan of `row_sums` seeded
    ///    with the incoming accumulator. The exclusive prefix is written
    ///    into stage 2 column 0.
    ///
    /// Peak working set per thread: `STAGE_2_INVERT_CHUNK_ROWS × K ×
    /// (sizeof(ExtVal) + sizeof(Val))`. The previous implementation's
    /// flat `messages` / `messages_inverses` arrays (each
    /// `total_lookups × sizeof(ExtVal)`) are gone; only the stage 2 output
    /// buffer itself stays alive.
    ///
    /// Returns: per-circuit stage 2 traces and intermediate accumulators
    /// (the second is the running-sum value at the end of each circuit).
    pub fn stage_2_traces(
        circuits: &[CircuitWitness<'_>],
        lookup_challenge: ExtVal,
        fingerprint_challenge: &ExtVal,
        initial_accumulator: ExtVal,
    ) -> (Vec<RowMajorMatrix<ExtVal>>, Vec<ExtVal>) {
        // Process circuits sequentially so the running accumulator can chain
        // directly across them. Within each circuit, Phase 1 (chunked-fused
        // denom build / batch invert / inverse-write / row-sum) and Phase 2a
        // (local inclusive prefix scan) still run in parallel via rayon.
        let mut accumulator = initial_accumulator;
        let mut intermediate_accumulators = Vec::with_capacity(circuits.len());
        let mut traces = Vec::with_capacity(circuits.len());

        for circuit in circuits {
            let num_rows = circuit.main.height();
            let k = circuit.lookups.len();
            let width = 1 + k;

            // Stage 2 output buffer. Column 0 = accumulator (filled in
            // phase 2). Columns 1..=K = message inverses (filled in
            // phase 1).
            let mut stage_2 = ExtVal::zero_vec(num_rows * width);

            // Fast path: no lookups → accumulator column is constant.
            if k == 0 {
                stage_2
                    .chunks_exact_mut(width)
                    .for_each(|row| row[0] = accumulator);
                intermediate_accumulators.push(accumulator);
                traces.push(RowMajorMatrix::new(stage_2, width));
                continue;
            }

            // Per-row partial sums: row_sums[r] = Σ_s mult[r,s] · inv[r,s]
            let mut row_sums = ExtVal::zero_vec(num_rows);

            // ── Phase 1: chunked fused denom build / batch invert ──
            row_sums
                .par_chunks_mut(STAGE_2_INVERT_CHUNK_ROWS)
                .zip(stage_2.par_chunks_mut(STAGE_2_INVERT_CHUNK_ROWS * width))
                .enumerate()
                .for_each(|(chunk_idx, (row_sums_chunk, stage_2_chunk))| {
                    let chunk_rows = row_sums_chunk.len();
                    let start_row = chunk_idx * STAGE_2_INVERT_CHUNK_ROWS;

                    // Thread-local denominator and multiplicity buffers for
                    // this chunk. Dropped before the row-sum pass to
                    // release memory back to the allocator promptly.
                    let mut denoms: Vec<ExtVal> = ExtVal::zero_vec(chunk_rows * k);
                    let mut mults: Vec<Val> = Val::zero_vec(chunk_rows * k);

                    // 1a. Resolve symbolic args + multiplicity against each
                    // row of the main (and preprocessed) trace.
                    //
                    // Trailing-zero invariant: rows whose effective args are
                    // shorter than the AIR-side max width (e.g., the
                    // unselected side of a match branch where selectors zero
                    // out trailing contributions) still produce the correct
                    // message — `fingerprint` is reverse-Horner from zero,
                    // so trailing zeros fold out.
                    for local_r in 0..chunk_rows {
                        let r = start_row + local_r;
                        let main_row = circuit.main.row_slice(r).unwrap();
                        let prep_row = circuit.preprocessed.map(|p| p.row_slice(r).unwrap());
                        let prep_ref = prep_row.as_deref();

                        for (s, lookup) in circuit.lookups.iter().enumerate() {
                            // Interpret args in the base field; fingerprint
                            // then lifts each via `Into<ExtVal>`. This keeps
                            // the per-arg work in the cheap base-field path
                            // and only the Horner fold runs in the extension.
                            let fp: ExtVal = fingerprint(
                                fingerprint_challenge,
                                lookup
                                    .args
                                    .iter()
                                    .map(|arg| arg.interpret::<Val, _>(&main_row, prep_ref)),
                            );
                            denoms[local_r * k + s] = lookup_challenge + fp;
                            mults[local_r * k + s] =
                                lookup.multiplicity.interpret(&main_row, prep_ref);
                        }
                    }

                    // 1b. Batch invert this chunk's denominators.
                    let inverses = batch_multiplicative_inverse(&denoms);
                    drop(denoms);

                    // 1c. Write inverses into stage 2 cols 1..=K and
                    // accumulate per-row partial sums.
                    for local_r in 0..chunk_rows {
                        let mut row_sum = ExtVal::ZERO;
                        let row_off = local_r * width;
                        for s in 0..k {
                            let inv = inverses[local_r * k + s];
                            let m = mults[local_r * k + s];
                            stage_2_chunk[row_off + 1 + s] = inv;
                            row_sum += ExtVal::from(m) * inv;
                        }
                        row_sums_chunk[local_r] = row_sum;
                    }
                });

            // ── Phase 2: parallel exclusive prefix scan of row_sums ──
            //
            // After this, stage_2[r, 0] = accumulator + Σ_{i<r} row_sums[i].
            let num_threads = current_num_threads().max(1);
            let scan_chunk = num_rows.div_ceil(num_threads);

            // 2a. Local inclusive prefix per chunk.
            row_sums.par_chunks_mut(scan_chunk).for_each(|c| {
                for i in 1..c.len() {
                    let prev = c[i - 1];
                    c[i] += prev;
                }
            });

            // 2b. Sequential combine of chunk totals → per-chunk start
            // offsets in the global exclusive prefix.
            let num_chunks = num_rows.div_ceil(scan_chunk);
            let mut chunk_offsets = ExtVal::zero_vec(num_chunks);
            chunk_offsets[0] = accumulator;
            for c in 1..num_chunks {
                let prev_chunk_end = (c * scan_chunk).min(num_rows);
                chunk_offsets[c] = chunk_offsets[c - 1] + row_sums[prev_chunk_end - 1];
            }
            // Running-sum value at the end of this circuit.
            accumulator = chunk_offsets[num_chunks - 1] + row_sums[num_rows - 1];

            // 2c. Write the exclusive prefix into stage 2 column 0 in
            // parallel.
            stage_2
                .par_chunks_mut(scan_chunk * width)
                .enumerate()
                .for_each(|(c, chunk)| {
                    let chunk_start = c * scan_chunk;
                    let off = chunk_offsets[c];
                    for (p, row) in chunk.chunks_exact_mut(width).enumerate() {
                        let excl = if p == 0 {
                            ExtVal::ZERO
                        } else {
                            row_sums[chunk_start + p - 1]
                        };
                        row[0] = off + excl;
                    }
                });

            intermediate_accumulators.push(accumulator);
            traces.push(RowMajorMatrix::new(stage_2, width));
        }

        (traces, intermediate_accumulators)
    }
}

impl<A> BaseAir<Val> for LookupAir<A>
where
    A: BaseAir<Val>,
{
    fn width(&self) -> usize {
        self.inner_air.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val>> {
        self.preprocessed.clone()
    }
}

impl<A, AB> Air<AB> for LookupAir<A>
where
    A: Air<AB>,
    AB: TwoStagedBuilder<F = Val, EF = ExtVal>,
{
    fn eval(&self, builder: &mut AB) {
        if self.preprocessed.is_some() {
            let preprocessed = builder.preprocessed().clone();
            let preprocessed_row = preprocessed.current_slice();
            self.eval_with_preprocessed_row(builder, Some(preprocessed_row))
        } else {
            self.eval_with_preprocessed_row(builder, None)
        }
    }
}

impl<A> LookupAir<A> {
    fn eval_with_preprocessed_row<AB>(&self, builder: &mut AB, preprocessed_row: Option<&[AB::Var]>)
    where
        A: Air<AB>,
        AB: TwoStagedBuilder<F = Val, EF = ExtVal>,
    {
        // Call `eval` for regular stage 1 constraints.
        self.inner_air.eval(builder);

        // Extract challenges and accumulators from stage 2 public values.
        let stage_2_public_values = builder.stage_2_public_values();
        debug_assert_eq!(stage_2_public_values.len(), LOOKUP_PUBLIC_SIZE);
        let lookup_challenge = stage_2_public_values[0].into();
        let fingerprint_challenge = stage_2_public_values[1].into();
        let acc = stage_2_public_values[2];
        let next_acc = stage_2_public_values[3];

        // Bind relevant variables to construct the stage 2 constraints.
        let stage_2 = builder.stage_2();
        let stage_2_row = stage_2.row_slice(0).unwrap();
        let stage_2_next_row = stage_2.row_slice(1).unwrap();
        let acc_col = stage_2_row[0];
        let next_acc_col = stage_2_next_row[0];
        let messages_inverses = &stage_2_row[1..];
        let lookups = &self.lookups;
        debug_assert_eq!(messages_inverses.len(), lookups.len());

        // Compute the final accumulator for the current row with the inverses
        // of the messages from the stage 2 trace while asserting that these
        // inverses are indeed the inverses of the messages computed on the main
        // trace.
        let main = builder.main();
        let row = main.current_slice();
        let mut acc_expr = acc_col.into();
        for (lookup, &message_inverse) in lookups.iter().zip(messages_inverses) {
            let multiplicity: AB::ExprEF =
                lookup.multiplicity.interpret(row, preprocessed_row).into();
            let args = lookup
                .args
                .iter()
                .map(|arg| arg.interpret(row, preprocessed_row));
            let fingerprint = fingerprint(&fingerprint_challenge, args);
            let message: AB::ExprEF = lookup_challenge.clone() + fingerprint;
            let message_inverse = message_inverse.into();
            builder.assert_one_ext(message * message_inverse.clone());
            acc_expr += multiplicity * message_inverse;
        }

        // The initial accumulator value must be set correctly.
        builder.when_first_row().assert_eq_ext(acc_col, acc);

        // The accumulator computed on the main trace for the current row must
        // equal the accumulator of the next row from the stage 2 trace.
        builder
            .when_transition()
            .assert_eq_ext(acc_expr.clone(), next_acc_col);

        // The final accumulator must match the expected value.
        builder.when_last_row().assert_eq_ext(acc_expr, next_acc);
    }
}

#[cfg(test)]
mod tests {
    use p3_air::{AirBuilder, WindowAccess};
    use p3_field::Field;

    use crate::{
        builder::symbolic::var,
        system::{ProverKey, System, SystemWitness},
        types::{CommitmentParameters, FriParameters},
    };

    use super::*;

    enum CS {
        Even,
        Odd,
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
    fn system(commitment_parameters: CommitmentParameters) -> (System<CS>, ProverKey) {
        let even = LookupAir::new(CS::Even, CS::Even.lookups());
        let odd = LookupAir::new(CS::Odd, CS::Odd.lookups());
        System::new(commitment_parameters, [even, odd])
    }

    #[test]
    fn lookup_test() {
        let commitment_parameters = CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        };
        let (system, key) = system(commitment_parameters);
        let f = Val::from_u32;
        #[rustfmt::skip]
        let witness = SystemWitness::from_stage_1(
            vec![
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
            ],
            &system,
        );
        let claim = &[f(0), f(4), f(1)];
        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 64,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        };
        let proof = system.prove(fri_parameters, &key, claim, witness);
        system.verify(fri_parameters, claim, &proof).unwrap();
    }
}
