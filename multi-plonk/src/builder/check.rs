//! Debug builder that checks every constraint on every row, used in tests to
//! locate constraint violations before invoking the full prover.

use ark_ff::Zero;

use crate::air::{Air, AirBuilder};
use crate::matrix::Matrix;
use crate::types::Val;

pub fn check_constraints<A>(
    air: &A,
    preprocessed: Option<&Matrix<Val>>,
    stage_1: &Matrix<Val>,
    stage_2: &Matrix<Val>,
    stage_2_public_values: &[Val],
) where
    A: for<'a> Air<DebugConstraintBuilder<'a>>,
{
    let height = stage_1.height();
    for i in 0..height {
        let i_next = (i + 1) % height;
        let mut builder = DebugConstraintBuilder {
            row_index: i,
            preprocessed_local: &[],
            preprocessed_next: &[],
            stage_1_local: stage_1.row(i),
            stage_1_next: stage_1.row(i_next),
            stage_2_local: stage_2.row(i),
            stage_2_next: stage_2.row(i_next),
            stage_2_public_values,
            is_first_row: Val::from(u64::from(i == 0)),
            is_last_row: Val::from(u64::from(i == height - 1)),
            is_transition: Val::from(u64::from(i != height - 1)),
        };
        if let Some(preprocessed) = preprocessed {
            builder.preprocessed_local = preprocessed.row(i);
            builder.preprocessed_next = preprocessed.row(i_next);
            air.eval(&mut builder);
        } else {
            air.eval(&mut builder);
        }
    }
}

pub struct DebugConstraintBuilder<'a> {
    pub row_index: usize,
    pub preprocessed_local: &'a [Val],
    pub preprocessed_next: &'a [Val],
    pub stage_1_local: &'a [Val],
    pub stage_1_next: &'a [Val],
    pub stage_2_local: &'a [Val],
    pub stage_2_next: &'a [Val],
    pub stage_2_public_values: &'a [Val],
    pub is_first_row: Val,
    pub is_last_row: Val,
    pub is_transition: Val,
}

impl<'a> AirBuilder for DebugConstraintBuilder<'a> {
    type Var = Val;
    type Expr = Val;

    fn main_local(&self) -> &[Val] {
        self.stage_1_local
    }
    fn main_next(&self) -> &[Val] {
        self.stage_1_next
    }
    fn preprocessed_local(&self) -> &[Val] {
        self.preprocessed_local
    }
    fn preprocessed_next(&self) -> &[Val] {
        self.preprocessed_next
    }
    fn stage_2_local(&self) -> &[Val] {
        self.stage_2_local
    }
    fn stage_2_next(&self) -> &[Val] {
        self.stage_2_next
    }
    fn stage_2_public_values(&self) -> &[Val] {
        self.stage_2_public_values
    }

    fn is_first_row(&self) -> Val {
        self.is_first_row
    }
    fn is_last_row(&self) -> Val {
        self.is_last_row
    }
    fn is_transition(&self) -> Val {
        self.is_transition
    }

    fn assert_zero<I: Into<Val>>(&mut self, x: I) {
        let x: Val = x.into();
        assert!(
            x.is_zero(),
            "constraint had nonzero value on row {}: {:?}",
            self.row_index,
            x
        );
    }
}
