use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use crate::{
    builder::{
        TwoStagedBuilder,
        symbolic::{Entry, SymbolicExpression},
    },
    system::MIN_IO_SIZE,
    types::Val,
};

pub struct Lookup {
    pub multiplicity: SymbolicExpression<Val>,
    pub args: Vec<SymbolicExpression<Val>>,
}

pub struct LookupAir<A> {
    pub inner_air: A,
    pub lookups: Vec<Lookup>,
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
            let multiplicity = lookup.multiplicity.interpret::<AB>(&row);
            let args = fingerprint_reverse::<AB, _>(
                fingerprint_challenge.into(),
                lookup
                    .args
                    .iter()
                    .rev()
                    .map(|arg| arg.interpret::<AB>(&row)),
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

fn fingerprint_reverse<AB: AirBuilder, Iter: Iterator<Item = AB::Expr>>(
    r: AB::Expr,
    coeffs: Iter,
) -> AB::Expr {
    coeffs.fold(AB::F::ZERO.into(), |acc, coeff| acc * r.clone() + coeff)
}

impl<F: Field> SymbolicExpression<F> {
    fn interpret<AB: AirBuilder<F = F>>(&self, row: &[AB::Var]) -> AB::Expr {
        match self {
            SymbolicExpression::Variable(var) => match var.entry {
                Entry::Main { offset } => row[offset].clone().into(),
                _ => unimplemented!(),
            },
            SymbolicExpression::Constant(c) => (*c).into(),
            SymbolicExpression::Add { x, y, .. } => x.interpret::<AB>(row) + y.interpret::<AB>(row),
            SymbolicExpression::Sub { x, y, .. } => x.interpret::<AB>(row) - y.interpret::<AB>(row),
            SymbolicExpression::Neg { x, .. } => -x.interpret::<AB>(row),
            SymbolicExpression::Mul { x, y, .. } => x.interpret::<AB>(row) * y.interpret::<AB>(row),
            _ => unimplemented!(),
        }
    }
}
