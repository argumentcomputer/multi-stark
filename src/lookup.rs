use p3_air::{Air, BaseAir, ExtensionBuilder, WindowAccess};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, batch_multiplicative_inverse};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;

use crate::builder::{TwoStagedBuilder, symbolic::SymbolicExpression};

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

pub struct LookupAir<A, F: Field> {
    pub inner_air: A,
    pub lookups: Vec<Lookup<SymbolicExpression<F>>>,
    pub preprocessed: Option<RowMajorMatrix<F>>,
}

impl<A: BaseAir<F>, F: Field> LookupAir<A, F> {
    pub fn new(inner_air: A, lookups: Vec<Lookup<SymbolicExpression<F>>>) -> Self {
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

impl<F: Field> Lookup<SymbolicExpression<F>> {
    /// Computes the concrete lookup attributes for its respective expressions
    /// given a trace row and a preprocessed trace row.
    pub fn compute_expr(&self, row: &[F], preprocessed: Option<&[F]>) -> Lookup<F> {
        let multiplicity = self.multiplicity.interpret(row, preprocessed);
        let args = self
            .args
            .iter()
            .map(|arg| arg.interpret(row, preprocessed))
            .collect();
        Lookup { multiplicity, args }
    }
}

impl<F: Field> Lookup<F> {
    /// Computes the stage 2 traces and the intermediate accumulators for each
    /// circuit given a lookup challenge, a fingerprint challenge and the current
    /// accumulator value (computed from the initial claims).
    ///
    /// Note: the lookups are expected to be fully padded. That is, for each
    /// circuit, every row must have the exact same number of lookups.
    pub fn stage_2_traces<EF: ExtensionField<F>>(
        lookups: &[Vec<Vec<Self>>],
        lookup_challenge: EF,
        fingerprint_challenge: &EF,
        mut accumulator: EF,
    ) -> (Vec<RowMajorMatrix<EF>>, Vec<EF>) {
        // Number of lookups per circuit. Every row in a circuit is assumed to
        // have the same number of lookups (the lookups are expected to be fully
        // padded), so this is taken from the first row.
        let num_lookups_per_circuit: Vec<usize> = lookups
            .iter()
            .map(|circuit_lookups| circuit_lookups.len() * circuit_lookups[0].len())
            .collect();

        // Compute the message for each lookup, in flat circuit-major order.
        // Flatten the references serially first so the parallel map operates
        // on an indexed slice and `collect` can write straight into the
        // output Vec without tree-reducing worker buffers.
        let _g = tracing::info_span!("stark/lookup_messages").entered();
        let flat: Vec<&Self> = lookups.iter().flatten().flatten().collect();
        let messages: Vec<EF> = flat
            .par_iter()
            .map(|lookup| lookup.compute_message(lookup_challenge, fingerprint_challenge))
            .collect();
        drop(_g);

        // Compute the inverses of all messages in batch.
        let messages_inverses = tracing::info_span!("stark/batch_inverse")
            .in_scope(|| batch_multiplicative_inverse(&messages));

        // Compute and collect intermediate accumulators and traces.
        let _g = tracing::info_span!("stark/lookup_traces").entered();
        let mut intermediate_accumulators = Vec::with_capacity(lookups.len());
        let mut traces = Vec::with_capacity(lookups.len());
        let mut offset = 0;
        for (circuit_lookups, num_circuit_messages) in lookups.iter().zip(num_lookups_per_circuit) {
            // Get the slice containing the messages inverses for the current circuit.
            let circuit_messages_inverses =
                &messages_inverses[offset..offset + num_circuit_messages];
            offset += num_circuit_messages;

            let num_row_lookups = circuit_lookups[0].len();
            let vec = if num_row_lookups == 0 {
                // No row lookup. Just repeat the accumulator for each row.
                vec![accumulator; circuit_lookups.len()]
            } else {
                // Flatten each row accumulator followed by the inverse of the message
                // associated with each row lookup.
                circuit_lookups
                    .iter()
                    .zip(circuit_messages_inverses.chunks_exact(num_row_lookups))
                    .flat_map(|(row_lookups, row_messages_inverses)| {
                        let mut row = Vec::with_capacity(1 + row_lookups.len());
                        row.push(accumulator);
                        row.extend(row_lookups.iter().zip(row_messages_inverses).map(
                            |(lookup, &message_inverse)| {
                                accumulator += EF::from(lookup.multiplicity) * message_inverse;
                                message_inverse
                            },
                        ));
                        row
                    })
                    .collect()
            };
            let width = 1 + num_row_lookups;
            debug_assert_eq!(vec.len() % width, 0);
            let trace = RowMajorMatrix::new(vec, width);
            intermediate_accumulators.push(accumulator);
            traces.push(trace);
        }
        drop(_g);
        (traces, intermediate_accumulators)
    }

    fn compute_message<EF: ExtensionField<F>>(
        &self,
        lookup_challenge: EF,
        fingerprint_challenge: &EF,
    ) -> EF {
        let fingerprint = fingerprint(fingerprint_challenge, self.args.iter().cloned());
        lookup_challenge + fingerprint
    }
}

impl<A, F> BaseAir<F> for LookupAir<A, F>
where
    A: BaseAir<F>,
    F: Field,
{
    fn width(&self) -> usize {
        self.inner_air.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.preprocessed.clone()
    }
}

impl<A, F, AB> Air<AB> for LookupAir<A, F>
where
    A: Air<AB>,
    F: Field,
    AB: TwoStagedBuilder<F = F>,
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

impl<A, F: Field> LookupAir<A, F> {
    fn eval_with_preprocessed_row<AB>(&self, builder: &mut AB, preprocessed_row: Option<&[AB::Var]>)
    where
        A: Air<AB>,
        AB: TwoStagedBuilder<F = F>,
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
        types::{CommitmentParameters, FriParameters, GoldilocksKeccakConfig, Val},
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

    fn system() -> (System<CS>, ProverKey) {
        let config = GoldilocksKeccakConfig::new(COMMITMENT_PARAMETERS, FRI_PARAMETERS);
        let even = LookupAir::new(CS::Even, CS::Even.lookups());
        let odd = LookupAir::new(CS::Odd, CS::Odd.lookups());
        System::new(config, [even, odd])
    }

    fn witness(system: &System<CS>) -> SystemWitness {
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
            system,
        );
        witness
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
