//! AIR trait + minimal `AirBuilder` hierarchy.
//!
//! Mirrors the `p3_air::Air` / `p3_air::AirBuilder` shape, simplified for a
//! single field (no extension). Each builder exposes:
//!   * a 2-row window over the main, preprocessed, and stage-2 traces
//!   * row selectors (`is_first_row`, `is_last_row`, `is_transition`)
//!   * stage-2 public values (lookup challenges + accumulators)
//!   * an `assert_zero` sink that constraint folders accumulate into

use std::ops::{Add, Mul, Neg, Sub};

use crate::matrix::Matrix;
use crate::types::Val;

/// Static circuit metadata: trace width and optional preprocessed trace.
pub trait BaseAir {
    fn width(&self) -> usize;
    fn preprocessed_trace(&self) -> Option<Matrix<Val>> {
        None
    }
}

/// Constraint-evaluation interface implemented by symbolic, prover, verifier
/// and debug builders.
pub trait AirBuilder: Sized {
    type Var: Copy + Into<Self::Expr>;
    type Expr: Sized
        + Clone
        + From<Val>
        + From<Self::Var>
        + Add<Self::Expr, Output = Self::Expr>
        + Sub<Self::Expr, Output = Self::Expr>
        + Mul<Self::Expr, Output = Self::Expr>
        + Neg<Output = Self::Expr>;

    fn main_local(&self) -> &[Self::Var];
    fn main_next(&self) -> &[Self::Var];
    fn preprocessed_local(&self) -> &[Self::Var];
    fn preprocessed_next(&self) -> &[Self::Var];
    fn stage_2_local(&self) -> &[Self::Var];
    fn stage_2_next(&self) -> &[Self::Var];
    fn stage_2_public_values(&self) -> &[Self::Var];

    fn is_first_row(&self) -> Self::Expr;
    fn is_last_row(&self) -> Self::Expr;
    fn is_transition(&self) -> Self::Expr;

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I);

    fn assert_eq<I, J>(&mut self, x: I, y: J)
    where
        I: Into<Self::Expr>,
        J: Into<Self::Expr>,
    {
        self.assert_zero(x.into() - y.into());
    }

    fn assert_one<I: Into<Self::Expr>>(&mut self, x: I) {
        let one: Self::Expr = Val::from(1u64).into();
        self.assert_zero(x.into() - one);
    }

    fn assert_bool<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        let one: Self::Expr = Val::from(1u64).into();
        self.assert_zero(x.clone() * (x - one));
    }

    fn when<I: Into<Self::Expr>>(&mut self, condition: I) -> FilteredBuilder<'_, Self> {
        FilteredBuilder {
            inner: self,
            condition: condition.into(),
        }
    }

    fn when_first_row(&mut self) -> FilteredBuilder<'_, Self> {
        let cond = self.is_first_row();
        self.when(cond)
    }

    fn when_last_row(&mut self) -> FilteredBuilder<'_, Self> {
        let cond = self.is_last_row();
        self.when(cond)
    }

    fn when_transition(&mut self) -> FilteredBuilder<'_, Self> {
        let cond = self.is_transition();
        self.when(cond)
    }
}

/// Sub-builder that scopes assertions under a boolean `condition` expression
/// (multiplies the assertion by the condition before forwarding it).
pub struct FilteredBuilder<'a, AB: AirBuilder> {
    inner: &'a mut AB,
    condition: AB::Expr,
}

impl<'a, AB: AirBuilder> FilteredBuilder<'a, AB> {
    pub fn assert_zero<I: Into<AB::Expr>>(&mut self, x: I) {
        let x: AB::Expr = x.into();
        self.inner.assert_zero(self.condition.clone() * x);
    }

    pub fn assert_eq<I, J>(&mut self, x: I, y: J)
    where
        I: Into<AB::Expr>,
        J: Into<AB::Expr>,
    {
        self.assert_zero(x.into() - y.into());
    }

    pub fn assert_one<I: Into<AB::Expr>>(&mut self, x: I) {
        let one: AB::Expr = Val::from(1u64).into();
        self.assert_eq(x, one);
    }
}

/// AIR generic over the builder type, so the same constraints can be evaluated
/// symbolically (degree analysis), concretely on a coset (prover), or at a
/// single point (verifier).
pub trait Air<AB: AirBuilder>: BaseAir {
    fn eval(&self, builder: &mut AB);
}
