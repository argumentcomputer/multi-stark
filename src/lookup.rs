use p3_air::{Air, BaseAir, BaseAirWithPublicValues, ExtensionBuilder};
use p3_field::{PrimeCharacteristicRing, batch_multiplicative_inverse};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

use crate::{
    builder::{PreprocessedBuilder, TwoStagedBuilder, symbolic::SymbolicExpression},
    types::{ExtVal, Val},
};

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

impl Lookup<SymbolicExpression<Val>> {
    /// Computes the concrete lookup attributes for its respective expressions
    /// given a trace row and a preprocessed trace row.
    pub fn compute_expr(&self, row: &[Val], preprocessed: Option<&[Val]>) -> Lookup<Val> {
        let multiplicity = self.multiplicity.interpret(row, preprocessed);
        let args = self
            .args
            .iter()
            .map(|arg| arg.interpret(row, preprocessed))
            .collect();
        Lookup { multiplicity, args }
    }
}

impl Lookup<Val> {
    /// Computes the stage 2 traces and the intermediate accumulators for each
    /// circuit given a lookup challenge, a fingerprint challenge and the current
    /// accumulator value (computed from the initial claims).
    ///
    /// Note: the lookups are expected to be fully padded. That is, for each
    /// circuit, every row must have the exact same number of lookups.
    pub fn stage_2_traces(
        lookups: &[Vec<Vec<Self>>],
        lookup_challenge: ExtVal,
        fingerprint_challenge: &ExtVal,
        mut accumulator: ExtVal,
    ) -> (Vec<RowMajorMatrix<ExtVal>>, Vec<ExtVal>) {
        // Collect the number of lookups per circuit while accumulating the total
        // number of lookups.
        let mut num_lookups_per_circuit = Vec::with_capacity(lookups.len());
        let mut total_num_lookups = 0;
        for circuit_lookups in lookups {
            let num_rows = circuit_lookups.len();
            // Every row is assumed to have the same number of lookups, which is
            // the number of lookups of the first row.
            let num_row_lookups = circuit_lookups[0].len();
            let num_circuit_lookups = num_rows * num_row_lookups;
            num_lookups_per_circuit.push(num_circuit_lookups);
            total_num_lookups += num_circuit_lookups;
        }

        // Compute and collect all messages. There's one message per lookup.
        let mut messages = Vec::with_capacity(total_num_lookups);
        for circuit_lookups in lookups {
            let circuit_messages = circuit_lookups
                .iter()
                .flatten()
                .map(|lookup| lookup.compute_message(lookup_challenge, fingerprint_challenge));
            messages.extend(circuit_messages);
        }

        // Compute the inverses of all messages in batch.
        let messages_inverses = batch_multiplicative_inverse(&messages);

        // Compute and collect intermediate accumulators and traces.
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
                                accumulator += ExtVal::from(lookup.multiplicity) * message_inverse;
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
        (traces, intermediate_accumulators)
    }

    fn compute_message(&self, lookup_challenge: ExtVal, fingerprint_challenge: &ExtVal) -> ExtVal {
        let fingerprint = fingerprint(fingerprint_challenge, self.args.iter().cloned());
        lookup_challenge + fingerprint
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

impl<A> BaseAirWithPublicValues<Val> for LookupAir<A>
where
    A: BaseAirWithPublicValues<Val>,
{
    fn num_public_values(&self) -> usize {
        self.inner_air.num_public_values()
    }
}

impl<A, AB> Air<AB> for LookupAir<A>
where
    A: Air<AB>,
    AB: PreprocessedBuilder + TwoStagedBuilder<F = Val, EF = ExtVal>,
{
    fn eval(&self, builder: &mut AB) {
        if let Some(preprocessed) = builder.preprocessed() {
            let preprocessed_row = preprocessed.row_slice(0);
            debug_assert!(preprocessed_row.is_some());
            self.eval_with_preprocessed_row(builder, preprocessed_row.as_deref())
        } else {
            self.eval_with_preprocessed_row(builder, None)
        }
    }
}

impl<A> LookupAir<A> {
    fn eval_with_preprocessed_row<AB>(&self, builder: &mut AB, preprocessed_row: Option<&[AB::Var]>)
    where
        A: Air<AB>,
        AB: PreprocessedBuilder + TwoStagedBuilder<F = Val, EF = ExtVal>,
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
        let row = main.row_slice(0).unwrap();
        let mut acc_expr = acc_col.into();
        for (lookup, &message_inverse) in lookups.iter().zip(messages_inverses) {
            let multiplicity: AB::ExprEF =
                lookup.multiplicity.interpret(&row, preprocessed_row).into();
            let args = lookup
                .args
                .iter()
                .map(|arg| arg.interpret(&row, preprocessed_row));
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
    use p3_air::AirBuilder;
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
            let local = main.row_slice(0).unwrap();
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
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
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
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let proof = system.prove(fri_parameters, &key, claim, witness);
        system.verify(fri_parameters, claim, &proof).unwrap();
    }
}
