use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Algebra, Field};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

use crate::{
    builder::{
        TwoStagedBuilder,
        symbolic::{Entry, SymbolicExpression},
    },
    system::{CircuitWitness, MIN_IO_SIZE, SystemWitness},
    types::Val,
};

pub struct Lookup<Expr> {
    pub multiplicity: Expr,
    pub args: Vec<Expr>,
}

pub struct LookupAir<A> {
    pub inner_air: A,
    pub lookups: Vec<Lookup<SymbolicExpression<Val>>>,
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
            fingerprint_challenge,
            self.args.iter().rev().copied(),
        );
        lookup_challenge + args
    }
}

impl SystemWitness<Val> {
    pub fn stage_2_from_lookups(
        &self,
        lookups: &[Lookup<SymbolicExpression<Val>>],
    ) -> Box<dyn Fn(&[Val], &mut Vec<Val>) -> SystemWitness<Val>> {
        let lookups = self
            .circuits
            .iter()
            .map(|circuit| {
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
        Box::new(move |values, intermediate_accumulators| {
            let lookup_challenge = values[0];
            let fingenprint_challenge = values[1];
            let mut accumulator = values[2];
            let circuits = lookups
                .iter()
                .map(|lookups_per_circuit| {
                    let num_lookups = lookups_per_circuit.len();
                    let vec = lookups_per_circuit
                        .iter()
                        .flat_map(|lookups_per_row| {
                            let mut row = Vec::with_capacity(lookups_per_row.len() + 1);
                            row.push(accumulator);
                            row.extend(lookups_per_row.iter().map(|lookup| {
                                let message =
                                    lookup.compute_message(lookup_challenge, fingenprint_challenge);
                                accumulator += message;
                                message.inverse()
                            }));
                            row
                        })
                        .collect();
                    let trace = RowMajorMatrix::new(vec, num_lookups + 1);
                    intermediate_accumulators.push(accumulator);
                    CircuitWitness { trace }
                })
                .collect();
            SystemWitness { circuits }
        })
    }
}

impl<A> BaseAir<Val> for LookupAir<A>
where
    A: BaseAir<Val>,
{
    fn width(&self) -> usize {
        self.inner_air.width()
    }
}

impl<A, AB> Air<AB> for LookupAir<A>
where
    A: Air<AB>,
    AB: AirBuilderWithPublicValues<F = Val> + TwoStagedBuilder,
{
    fn eval(&self, builder: &mut AB) {
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
        debug_assert_eq!(inverse_of_messages.len(), self.lookups.len());

        let row = main.row_slice(0).unwrap();
        let acc_col = &stage_2_row[1];
        let mut acc_expr: AB::Expr = acc_col.clone().into();
        for (lookup, inverse_of_message) in self.lookups.iter().zip(inverse_of_messages) {
            let multiplicity = lookup.multiplicity.interpret::<AB::Expr, AB::Var>(&row);
            let args = fingerprint_reverse::<Val, AB::Expr, _>(
                fingerprint_challenge.into(),
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
        builder
            .when_last_row()
            .assert_eq(next_acc_col.clone(), next_acc);
    }
}

fn fingerprint_reverse<F: Field, Expr: Algebra<F>, Iter: Iterator<Item = Expr>>(
    r: Expr,
    coeffs: Iter,
) -> Expr {
    coeffs.fold(F::ZERO.into(), |acc, coeff| acc * r.clone() + coeff)
}

impl<F: Field> SymbolicExpression<F> {
    pub fn interpret<Expr: Algebra<F>, Var: Into<Expr> + Clone>(&self, row: &[Var]) -> Expr {
        match self {
            SymbolicExpression::Variable(var) => match var.entry {
                Entry::Main { offset } => row[offset].clone().into(),
                _ => unimplemented!(),
            },
            SymbolicExpression::Constant(c) => (*c).into(),
            SymbolicExpression::Add { x, y, .. } => {
                x.interpret::<Expr, Var>(row) + y.interpret::<Expr, Var>(row)
            }
            SymbolicExpression::Sub { x, y, .. } => {
                x.interpret::<Expr, Var>(row) - y.interpret::<Expr, Var>(row)
            }
            SymbolicExpression::Neg { x, .. } => -x.interpret::<Expr, Var>(row),
            SymbolicExpression::Mul { x, y, .. } => {
                x.interpret::<Expr, Var>(row) * y.interpret::<Expr, Var>(row)
            }
            _ => unimplemented!(),
        }
    }
}
