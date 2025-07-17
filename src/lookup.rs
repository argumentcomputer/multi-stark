use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithPublicValues};
use p3_field::{Algebra, Field};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

use crate::{
    builder::{
        TwoStagedAir, TwoStagedBuilder,
        symbolic::{Entry, SymbolicExpression},
    },
    system::{CircuitWitness, MIN_IO_SIZE, SystemWitness},
    types::Val,
};

#[derive(Clone)]
pub struct Lookup<Expr> {
    pub multiplicity: Expr,
    pub args: Vec<Expr>,
}

pub struct LookupAir<A, F> {
    pub inner_air: A,
    pub lookups: Vec<Lookup<SymbolicExpression<F>>>,
}

impl<A, F> LookupAir<A, F> {
    pub fn new(inner_air: A, lookups: Vec<Lookup<SymbolicExpression<F>>>) -> Self {
        Self { inner_air, lookups }
    }
}

impl Lookup<SymbolicExpression<Val>> {
    pub fn compute_expr(&self, row: &[Val]) -> Lookup<Val> {
        let multiplicity = self.multiplicity.interpret(row);
        let args = self.args.iter().map(|arg| arg.interpret(row)).collect();
        Lookup { multiplicity, args }
    }
}

impl Lookup<Val> {
    pub fn compute_message(&self, lookup_challenge: Val, fingerprint_challenge: Val) -> Val {
        let args = fingerprint_reverse::<Val, Val, _>(
            &fingerprint_challenge,
            self.args.iter().rev().copied(),
        );
        lookup_challenge + args
    }
}

impl SystemWitness<Val> {
    #[allow(clippy::type_complexity)]
    pub fn stage_2_from_symbolic_lookups(
        &self,
        circuit_lookups: &[Vec<Lookup<SymbolicExpression<Val>>>],
    ) -> Box<dyn Fn(&[Val], &mut Vec<Val>) -> Self> {
        let lookups = self
            .circuits
            .iter()
            .zip(circuit_lookups.iter())
            .map(|(circuit, lookups)| {
                circuit
                    .trace
                    .row_slices()
                    .map(|row| {
                        lookups
                            .iter()
                            .map(|lookup| lookup.compute_expr(row))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.stage_2_from_lookups(lookups)
    }

    #[allow(clippy::type_complexity)]
    pub fn stage_2_from_lookups(
        &self,
        lookups: Vec<Vec<Vec<Lookup<Val>>>>,
    ) -> Box<dyn Fn(&[Val], &mut Vec<Val>) -> Self> {
        Box::new(move |values, intermediate_accumulators| {
            let lookup_challenge = values[0];
            let fingenprint_challenge = values[1];
            let mut accumulator = values[2];
            let circuits = lookups
                .iter()
                .map(|lookups_per_circuit| {
                    let num_lookups = lookups_per_circuit[0].len();
                    let vec = lookups_per_circuit
                        .iter()
                        .flat_map(|lookups_per_row| {
                            let mut row = Vec::with_capacity(lookups_per_row.len() + 1);
                            row.push(accumulator);
                            row.extend(lookups_per_row.iter().map(|lookup| {
                                let inverse_of_message = lookup
                                    .compute_message(lookup_challenge, fingenprint_challenge)
                                    .inverse();
                                accumulator += inverse_of_message * lookup.multiplicity;
                                inverse_of_message
                            }));
                            row
                        })
                        .collect::<Vec<_>>();
                    debug_assert_eq!(vec.len() % (num_lookups + 1), 0);
                    let trace = RowMajorMatrix::new(vec, num_lookups + 1);
                    intermediate_accumulators.push(accumulator);
                    CircuitWitness { trace }
                })
                .collect();
            Self { circuits }
        })
    }
}

impl<A> BaseAir<Val> for LookupAir<A, Val>
where
    A: BaseAir<Val>,
{
    fn width(&self) -> usize {
        self.inner_air.width()
    }
}

impl<A> BaseAirWithPublicValues<Val> for LookupAir<A, Val>
where
    A: BaseAir<Val>,
{
    fn num_public_values(&self) -> usize {
        MIN_IO_SIZE
    }
}

impl<A> TwoStagedAir<Val> for LookupAir<A, Val>
where
    A: BaseAir<Val>,
{
    fn stage_2_width(&self) -> usize {
        self.lookups.len() + 1
    }
}

impl<A, AB> Air<AB> for LookupAir<A, Val>
where
    A: Air<AB>,
    AB: AirBuilderWithPublicValues<F = Val> + TwoStagedBuilder,
{
    fn eval(&self, builder: &mut AB) {
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
            let multiplicity = lookup.multiplicity.interpret::<AB::Expr, AB::Var>(&row);
            let args = fingerprint_reverse::<Val, AB::Expr, _>(
                &fingerprint_challenge.into(),
                lookup
                    .args
                    .iter()
                    .rev()
                    .map(|arg| arg.interpret::<AB::Expr, AB::Var>(&row)),
            );
            let message = lookup_challenge.into() + args;
            builder.assert_one(message.clone() * inverse_of_message.clone());
            acc_expr += multiplicity * inverse_of_message.clone();
        }
        builder
            .when_transition()
            .assert_eq(acc_expr, next_acc_col.clone());
        builder.when_first_row().assert_eq(acc_col.clone(), acc);
        builder.when_last_row().assert_eq(acc_col.clone(), next_acc);
    }
}

fn fingerprint_reverse<F: Field, Expr: Algebra<F>, Iter: Iterator<Item = Expr>>(
    r: &Expr,
    coeffs: Iter,
) -> Expr {
    coeffs.fold(F::ZERO.into(), |acc, coeff| acc * r.clone() + coeff)
}

impl<F: Field> SymbolicExpression<F> {
    pub fn interpret<Expr: Algebra<F>, Var: Into<Expr> + Clone>(&self, row: &[Var]) -> Expr {
        match self {
            Self::Variable(var) => match var.entry {
                Entry::Main { offset: 0 } => row[var.index].clone().into(),
                _ => unimplemented!(),
            },
            Self::Constant(c) => (*c).into(),
            Self::Add { x, y, .. } => x.interpret::<Expr, Var>(row) + y.interpret::<Expr, Var>(row),
            Self::Sub { x, y, .. } => x.interpret::<Expr, Var>(row) - y.interpret::<Expr, Var>(row),
            Self::Neg { x, .. } => -x.interpret::<Expr, Var>(row),
            Self::Mul { x, y, .. } => x.interpret::<Expr, Var>(row) * y.interpret::<Expr, Var>(row),
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;

    use crate::{
        builder::symbolic::SymbolicVariable,
        prover::Claim,
        system::{Circuit, System},
        types::{FriParameters, new_stark_config},
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
            let multiplicity = -var(0);
            let input = var(1);
            let input_is_zero = var(3);
            let input_not_zero = var(4);
            let recursion_output = var(5);
            let even_index = Val::from_u32(0).into();
            let odd_index = Val::from_u32(1).into();
            let one: SymbolicExpression<_> = Val::from_u32(1).into();
            match self {
                Self::Even => vec![
                    Lookup {
                        multiplicity,
                        args: vec![
                            even_index,
                            input.clone(),
                            input_not_zero.clone() * recursion_output.clone() + input_is_zero,
                        ],
                    },
                    Lookup {
                        multiplicity: input_not_zero,
                        args: vec![odd_index, input - one, recursion_output],
                    },
                ],
                Self::Odd => vec![
                    Lookup {
                        multiplicity,
                        args: vec![
                            odd_index,
                            input.clone(),
                            input_not_zero.clone() * recursion_output.clone(),
                        ],
                    },
                    Lookup {
                        multiplicity: input_not_zero,
                        args: vec![even_index, input - one, recursion_output],
                    },
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
    fn system() -> System<LookupAir<CS, Val>> {
        let even = Circuit::from_air(LookupAir {
            inner_air: CS::Even,
            lookups: CS::Even.lookups(),
        })
        .unwrap();
        let odd = Circuit::from_air(LookupAir {
            inner_air: CS::Odd,
            lookups: CS::Odd.lookups(),
        })
        .unwrap();
        System::new([("even", even), ("odd", odd)].into_iter())
    }

    #[test]
    fn lookup_test() {
        let system = system();
        let f = Val::from_u32;
        #[rustfmt::skip]
        let witness = SystemWitness {
            circuits: vec![
                CircuitWitness {
                    trace: RowMajorMatrix::new(
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
                },
                CircuitWitness {
                    trace: RowMajorMatrix::new(
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
                },
            ],
        };
        let claim = Claim {
            circuit_name: "even".into(),
            args: vec![f(4), f(1)],
        };
        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let config = new_stark_config(&fri_parameters);
        let circuit_lookups = [CS::Even.lookups(), CS::Odd.lookups()];
        let stage_2 = witness.stage_2_from_symbolic_lookups(&circuit_lookups);
        let proof = system.prove(&config, &claim, witness, stage_2);
        system.verify(&config, &claim, &proof).unwrap();
    }
}
