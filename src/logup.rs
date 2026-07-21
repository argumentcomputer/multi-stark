//! The logUp lookup argument scheme.
//!
//! LogUp represents each claim as a term `multiplicity / (β + fingerprint(γ,
//! args))` of a running sum, where β is the lookup challenge and γ the
//! fingerprint challenge. Multiplicities are native, so every
//! [`Interaction`] lowers to a single signed term: pushes and requires keep
//! their guard/multiplicity as a positive numerator, pulls and provides
//! negate it. In particular the permutation and multiplicity channels share
//! one accumulator — cross-channel matching happens to balance under logUp,
//! which portable circuits must not rely on (see [`Interaction`]).
//!
//! The claim multiset balances if and only if the running sum — seeded with
//! `Σ 1/(β + fingerprint(γ, claim))` over the public claims — ends at zero.
//!
//! Each circuit's stage 2 trace has one column for the running accumulator
//! plus one column per term holding the inverse of that term's message,
//! whose correctness is enforced by a `message · inverse = 1` constraint.

use p3_air::{ExtensionBuilder, WindowAccess};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, batch_multiplicative_inverse};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;

use crate::{
    builder::{TwoStagedBuilder, symbolic::SymbolicExpression},
    lookup::{
        CircuitWitnessInput, Interaction, LOOKUP_PUBLIC_SIZE, LookupAir, LookupArgument,
        fingerprint,
    },
};

/// A [`LookupAir`] under the logUp scheme.
pub type LogUpAir<A, F> = LookupAir<A, F, LogUp<F>>;

/// The concrete lookup data of one circuit under logUp: one term per row per
/// symbolic term.
pub type LogUpWitness<F> = Vec<Vec<LogUpTerm<F>>>;

/// The logUp claim set of one circuit: one signed term per interaction.
pub struct LogUp<F: Field> {
    pub terms: Vec<LogUpTerm<SymbolicExpression<F>>>,
}

/// A single logUp term: a signed multiplicity and the claim arguments.
/// Positive terms add to the claim set, negative terms remove from it.
#[derive(Clone)]
pub struct LogUpTerm<Expr> {
    pub multiplicity: Expr,
    pub args: Vec<Expr>,
}

impl<Expr> LogUpTerm<Expr> {
    /// Returns a [`LogUpTerm`] with multiplicity zero and no arguments.
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

    #[inline]
    pub fn new(multiplicity: Expr, args: Vec<Expr>) -> Self {
        Self { multiplicity, args }
    }
}

impl<F: Field> LogUpTerm<SymbolicExpression<F>> {
    /// Computes the concrete term attributes for its respective expressions
    /// given a trace row and a preprocessed trace row.
    fn compute_expr(&self, row: &[F], preprocessed: Option<&[F]>) -> LogUpTerm<F> {
        let multiplicity = self.multiplicity.interpret(row, preprocessed);
        let args = self
            .args
            .iter()
            .map(|arg| arg.interpret(row, preprocessed))
            .collect();
        LogUpTerm { multiplicity, args }
    }
}

impl<F: Field> LogUpTerm<F> {
    fn compute_message<EF: ExtensionField<F>>(
        &self,
        lookup_challenge: EF,
        fingerprint_challenge: &EF,
    ) -> EF {
        let fingerprint = fingerprint(fingerprint_challenge, self.args.iter().cloned());
        lookup_challenge + fingerprint
    }
}

impl<F: Field> LookupArgument<F> for LogUp<F> {
    type Witness = LogUpWitness<F>;

    /// Every interaction lowers to one signed term: the guard/multiplicity,
    /// negated for the removing directions (pull, provide).
    fn new(interactions: Vec<Interaction<F>>) -> Self {
        let terms = interactions
            .into_iter()
            .map(|interaction| match interaction {
                Interaction::Push { guard, args } => LogUpTerm {
                    multiplicity: guard,
                    args,
                },
                Interaction::Pull { guard, args } => LogUpTerm {
                    multiplicity: -guard,
                    args,
                },
                Interaction::Require { multiplicity, args } => LogUpTerm { multiplicity, args },
                Interaction::Provide { multiplicity, args } => LogUpTerm {
                    multiplicity: -multiplicity,
                    args,
                },
            })
            .collect();
        Self { terms }
    }

    /// One column for the accumulator and one column for the inverse of the
    /// message associated with each term.
    fn stage_2_width(&self) -> usize {
        1 + self.terms.len()
    }

    fn compute_witness(circuits: &[CircuitWitnessInput<'_, F, Self>]) -> Vec<Self::Witness> {
        circuits
            .iter()
            .map(|(lookups, main, preprocessed)| match preprocessed {
                Some(preprocessed) => main
                    .row_slices()
                    .zip(preprocessed.row_slices())
                    .map(|(row, preprocessed_row)| {
                        lookups
                            .terms
                            .iter()
                            .map(|term| term.compute_expr(row, Some(preprocessed_row)))
                            .collect()
                    })
                    .collect(),
                None => main
                    .row_slices()
                    .map(|row| {
                        lookups
                            .terms
                            .iter()
                            .map(|term| term.compute_expr(row, None))
                            .collect()
                    })
                    .collect(),
            })
            .collect()
    }

    /// Computes the stage 2 traces (running accumulator and message inverses
    /// per row) and the intermediate accumulators for each circuit.
    fn stage_2_traces<EF: ExtensionField<F>>(
        witnesses: &[Self::Witness],
        lookup_challenge: EF,
        fingerprint_challenge: &EF,
        claims_accumulator: EF,
    ) -> (Vec<RowMajorMatrix<EF>>, Vec<EF>) {
        let mut accumulator = claims_accumulator;

        // Number of terms per circuit. Every row in a circuit has the same
        // number of terms, so this is taken from the first row.
        let num_terms_per_circuit: Vec<usize> = witnesses
            .iter()
            .map(|circuit_terms| circuit_terms.len() * circuit_terms[0].len())
            .collect();

        // Compute the message for each term, in flat circuit-major order.
        // Flatten the references serially first so the parallel map operates
        // on an indexed slice and `collect` can write straight into the
        // output Vec without tree-reducing worker buffers.
        let _g = tracing::info_span!("stark/lookup_messages").entered();
        let flat: Vec<&LogUpTerm<F>> = witnesses.iter().flatten().flatten().collect();
        let messages: Vec<EF> = flat
            .par_iter()
            .map(|term| term.compute_message(lookup_challenge, fingerprint_challenge))
            .collect();
        drop(_g);

        // Compute the inverses of all messages in batch.
        let messages_inverses = tracing::info_span!("stark/batch_inverse")
            .in_scope(|| batch_multiplicative_inverse(&messages));

        // Compute and collect intermediate accumulators and traces.
        let _g = tracing::info_span!("stark/lookup_traces").entered();
        let mut intermediate_accumulators = Vec::with_capacity(witnesses.len());
        let mut traces = Vec::with_capacity(witnesses.len());
        let mut offset = 0;
        for (circuit_terms, num_circuit_messages) in witnesses.iter().zip(num_terms_per_circuit) {
            // Get the slice containing the messages inverses for the current circuit.
            let circuit_messages_inverses =
                &messages_inverses[offset..offset + num_circuit_messages];
            offset += num_circuit_messages;

            let num_row_terms = circuit_terms[0].len();
            let vec = if num_row_terms == 0 {
                // No row terms. Just repeat the accumulator for each row.
                vec![accumulator; circuit_terms.len()]
            } else {
                // Flatten each row accumulator followed by the inverse of the
                // message associated with each row term.
                circuit_terms
                    .iter()
                    .zip(circuit_messages_inverses.chunks_exact(num_row_terms))
                    .flat_map(|(row_terms, row_messages_inverses)| {
                        let mut row = Vec::with_capacity(1 + row_terms.len());
                        row.push(accumulator);
                        row.extend(row_terms.iter().zip(row_messages_inverses).map(
                            |(term, &message_inverse)| {
                                accumulator += EF::from(term.multiplicity) * message_inverse;
                                message_inverse
                            },
                        ));
                        row
                    })
                    .collect()
            };
            let width = 1 + num_row_terms;
            debug_assert_eq!(vec.len() % width, 0);
            let trace = RowMajorMatrix::new(vec, width);
            intermediate_accumulators.push(accumulator);
            traces.push(trace);
        }
        drop(_g);
        (traces, intermediate_accumulators)
    }

    fn fold_claims<EF: ExtensionField<F>>(
        lookup_challenge: EF,
        fingerprint_challenge: &EF,
        claims: &[&[F]],
    ) -> EF {
        let mut acc = EF::ZERO;
        for claim in claims {
            let message =
                lookup_challenge + fingerprint(fingerprint_challenge, claim.iter().cloned());
            acc += message.inverse();
        }
        acc
    }

    fn balance_target<EF: ExtensionField<F>>() -> EF {
        EF::ZERO
    }

    fn eval<AB>(&self, builder: &mut AB, preprocessed_row: Option<&[AB::Var]>)
    where
        AB: TwoStagedBuilder<F = F>,
    {
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
        debug_assert_eq!(messages_inverses.len(), self.terms.len());

        // Compute the final accumulator for the current row with the inverses
        // of the messages from the stage 2 trace while asserting that these
        // inverses are indeed the inverses of the messages computed on the main
        // trace.
        let main = builder.main();
        let row = main.current_slice();
        let mut acc_expr = acc_col.into();
        for (term, &message_inverse) in self.terms.iter().zip(messages_inverses) {
            let multiplicity: AB::ExprEF =
                term.multiplicity.interpret(row, preprocessed_row).into();
            let args = term
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
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_field::Field;

    use crate::{
        builder::symbolic::var,
        system::{ProverKey, System, SystemWitness},
        types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val},
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
        fn lookups(&self) -> Vec<Interaction<Val>> {
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
                    Interaction::provide(
                        multiplicity,
                        vec![
                            even_index,
                            input.clone(),
                            input_not_zero.clone() * recursion_output.clone() + input_is_zero,
                        ],
                    ),
                    Interaction::require(
                        input_not_zero,
                        vec![odd_index, input - one, recursion_output],
                    ),
                ],
                Self::Odd => vec![
                    Interaction::provide(
                        multiplicity,
                        vec![
                            odd_index,
                            input.clone(),
                            input_not_zero.clone() * recursion_output.clone(),
                        ],
                    ),
                    Interaction::require(
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

    fn system() -> (
        System<GoldilocksBlake3Config, CS>,
        ProverKey<GoldilocksBlake3Config>,
    ) {
        let config = GoldilocksBlake3Config::new(COMMITMENT_PARAMETERS, FRI_PARAMETERS);
        let even = LookupAir::new(CS::Even, CS::Even.lookups());
        let odd = LookupAir::new(CS::Odd, CS::Odd.lookups());
        System::new(config, [even, odd])
    }

    fn witness(system: &System<GoldilocksBlake3Config, CS>) -> SystemWitness<Val> {
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
