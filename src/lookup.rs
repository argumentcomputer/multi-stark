use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithPublicValues};
use p3_field::{Algebra, Field, PrimeCharacteristicRing, batch_multiplicative_inverse};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

use crate::{
    builder::{PreprocessedBuilder, TwoStagedBuilder, symbolic::SymbolicExpression},
    system::MIN_IO_SIZE,
    types::Val,
};

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

    pub fn stage_2_width(&self) -> usize {
        self.lookups.len() + 1
    }
}

impl Lookup<SymbolicExpression<Val>> {
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
    pub fn compute_message(&self, lookup_challenge: Val, fingerprint_challenge: &Val) -> Val {
        lookup_challenge + fingerprint(fingerprint_challenge, self.args.iter().copied())
    }

    pub fn stage_2_traces(
        lookups: &[Vec<Vec<Self>>],
        values: &[Val],
    ) -> (Vec<RowMajorMatrix<Val>>, Vec<Val>) {
        let lookup_challenge = values[0];
        let fingerprint_challenge = &values[1];
        let mut accumulator = values[2];

        // Collect the number of lookups per circuit.
        let mut num_lookups = Vec::with_capacity(lookups.len());
        for lookups_per_circuit in lookups {
            num_lookups.push(lookups_per_circuit.len() * lookups_per_circuit[0].len());
        }

        // Compute and collect all messages.
        let mut messages = Vec::with_capacity(num_lookups.iter().sum());
        for lookups_per_circuit in lookups {
            let circuit_messages = lookups_per_circuit
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
        for (circuit_lookups, num_circuit_messages) in lookups.iter().zip(num_lookups) {
            // Get the slice containing the messages inverses for the current circuit.
            let circuit_messages_inverses =
                &messages_inverses[offset..offset + num_circuit_messages];
            offset += num_circuit_messages;

            let num_row_lookups = circuit_lookups[0].len();
            let vec = if num_row_lookups == 0 {
                // No row lookup. Just repeat the accumulator.
                vec![accumulator; circuit_lookups.len()]
            } else {
                circuit_lookups
                    .iter()
                    .zip(circuit_messages_inverses.chunks_exact(num_row_lookups))
                    .flat_map(|(row_lookups, row_messages_inverses)| {
                        let mut row = Vec::with_capacity(row_lookups.len() + 1);
                        row.push(accumulator);
                        row.extend(row_lookups.iter().zip(row_messages_inverses).map(
                            |(lookup, &message_inverse)| {
                                accumulator += lookup.multiplicity * message_inverse;
                                message_inverse
                            },
                        ));
                        row
                    })
                    .collect()
            };
            debug_assert_eq!(vec.len() % (num_row_lookups + 1), 0);
            let trace = RowMajorMatrix::new(vec, num_row_lookups + 1);
            intermediate_accumulators.push(accumulator);
            traces.push(trace);
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
        self.inner_air.preprocessed_trace()
    }
}

impl<A> BaseAirWithPublicValues<Val> for LookupAir<A>
where
    A: BaseAir<Val>,
{
    fn num_public_values(&self) -> usize {
        MIN_IO_SIZE
    }
}

impl<A, AB> Air<AB> for LookupAir<A>
where
    A: Air<AB>,
    AB: AirBuilderWithPublicValues<F = Val> + TwoStagedBuilder + PreprocessedBuilder,
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
        AB: AirBuilderWithPublicValues<F = Val> + TwoStagedBuilder + PreprocessedBuilder,
    {
        let lookups = &self.lookups;
        self.inner_air.eval(builder);
        let public_values = builder.public_values();
        debug_assert_eq!(public_values.len(), MIN_IO_SIZE);
        let lookup_challenge = public_values[0];
        let fingerprint_challenge = public_values[1];
        let acc = public_values[2];
        let next_acc = public_values[3];

        let main = builder.main();
        let stage_2 = builder.stage_2();
        let stage_2_row = stage_2.row_slice(0).unwrap();
        let stage_2_next_row = stage_2.row_slice(1).unwrap();
        let next_acc_col = &stage_2_next_row[0];
        let inverse_of_messages = &stage_2_row[1..];
        debug_assert_eq!(inverse_of_messages.len(), lookups.len());

        let row = main.row_slice(0).unwrap();
        let acc_col = &stage_2_row[0];
        let mut acc_expr: AB::Expr = acc_col.clone().into();
        for (lookup, inverse_of_message) in lookups.iter().zip(inverse_of_messages) {
            let multiplicity = lookup.multiplicity.interpret(&row, preprocessed_row);
            let fingerprint = fingerprint::<Val, _>(
                &fingerprint_challenge.into(),
                lookup
                    .args
                    .iter()
                    .map(|arg| arg.interpret(&row, preprocessed_row)),
            );
            let message = lookup_challenge.into() + fingerprint;
            builder.assert_one(message.clone() * inverse_of_message.clone());
            acc_expr += multiplicity * inverse_of_message.clone();
        }
        builder
            .when_transition()
            .assert_eq(acc_expr.clone(), next_acc_col.clone());
        builder.when_first_row().assert_eq(acc_col.clone(), acc);
        builder.when_last_row().assert_eq(acc_expr, next_acc);
    }
}

/// Computes a fingerprint of the coefficients using Horner's method.
fn fingerprint<F: Field, Expr: Algebra<F>>(
    r: &Expr,
    coeffs: impl DoubleEndedIterator<Item = Expr>,
) -> Expr {
    coeffs
        .rev()
        .fold(F::ZERO.into(), |acc, coeff| acc * r.clone() + coeff)
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;

    use crate::{
        builder::symbolic::{Entry, SymbolicVariable},
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
            let var = |index| {
                SymbolicExpression::<Val>::from(SymbolicVariable::new(
                    Entry::Main { offset: 0 },
                    index,
                ))
            };
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
        AB: AirBuilderWithPublicValues,
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
