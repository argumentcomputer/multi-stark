//! Concrete constraint folders.
//!
//! Both prover and verifier flatten all `assert_zero(x)` calls to a single
//! field element via random-LC by powers of `alpha`.
//!
//! The prover folder is invoked once per row of the **coset evaluation
//! domain** (size `quotient_factor * trace_size`). The verifier folder is
//! invoked once at the out-of-domain challenge point `zeta`.

use crate::air::AirBuilder;
use crate::types::Val;
use ark_ff::Zero;

/// Evaluates constraints at a single coset point during quotient computation.
/// All windows are 1-row slices of the coset domain (current + next-by-step).
pub struct ProverConstraintFolder<'a> {
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
    pub alpha_powers: &'a [Val],
    pub accumulator: Val,
    pub constraint_index: usize,
}

impl<'a> AirBuilder for ProverConstraintFolder<'a> {
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
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += alpha_power * x;
        self.constraint_index += 1;
    }
}

/// Evaluates constraints at the out-of-domain point `zeta` from the verifier's
/// opened values.
pub struct VerifierConstraintFolder<'a> {
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
    pub alpha: Val,
    pub accumulator: Val,
}

impl<'a> AirBuilder for VerifierConstraintFolder<'a> {
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
        self.accumulator = self.accumulator * self.alpha + x;
    }
}

impl Default for ProverConstraintFolder<'_> {
    fn default() -> Self {
        Self {
            preprocessed_local: &[],
            preprocessed_next: &[],
            stage_1_local: &[],
            stage_1_next: &[],
            stage_2_local: &[],
            stage_2_next: &[],
            stage_2_public_values: &[],
            is_first_row: Val::zero(),
            is_last_row: Val::zero(),
            is_transition: Val::zero(),
            alpha_powers: &[],
            accumulator: Val::zero(),
            constraint_index: 0,
        }
    }
}
