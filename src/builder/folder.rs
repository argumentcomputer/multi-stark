/// Constraint folders for the prover and verifier, adapted from Plonky3.
use p3_air::{AirBuilder, ExtensionBuilder, RowWindow};
use p3_field::{BasedVectorSpace, PackedField};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;

use crate::types::{ExtVal, PackedExtVal, PackedVal, Val};

use super::TwoStagedBuilder;

#[derive(Debug)]
pub struct ProverConstraintFolder<'a> {
    pub preprocessed: RowWindow<'a, PackedVal>,
    pub stage_1: RowMajorMatrixView<'a, PackedVal>,
    pub stage_2: RowMajorMatrixView<'a, PackedExtVal>,
    pub stage_1_public_values: &'a [Val],
    pub stage_2_public_values: &'a [ExtVal],
    pub is_first_row: PackedVal,
    pub is_last_row: PackedVal,
    pub is_transition: PackedVal,
    pub alpha_powers: &'a [ExtVal],
    pub decomposed_alpha_powers: &'a [Vec<Val>],
    pub accumulator: PackedExtVal,
    pub constraint_index: usize,
}

type ViewPair<'a, T> = VerticalPair<RowMajorMatrixView<'a, T>, RowMajorMatrixView<'a, T>>;

#[derive(Debug)]
pub struct VerifierConstraintFolder<'a> {
    pub preprocessed: RowWindow<'a, ExtVal>,
    pub stage_1: ViewPair<'a, ExtVal>,
    pub stage_2: ViewPair<'a, ExtVal>,
    pub stage_1_public_values: &'a [Val],
    pub stage_2_public_values: &'a [ExtVal],
    pub is_first_row: ExtVal,
    pub is_last_row: ExtVal,
    pub is_transition: ExtVal,
    pub alpha: ExtVal,
    pub accumulator: ExtVal,
}

impl<'a> AirBuilder for ProverConstraintFolder<'a> {
    type F = Val;
    type Expr = PackedVal;
    type Var = PackedVal;
    type PreprocessedWindow = RowWindow<'a, PackedVal>;
    type MainWindow = RowWindow<'a, PackedVal>;
    type PublicVar = Val;

    #[inline]
    fn main(&self) -> Self::MainWindow {
        RowWindow::from_view(&self.stage_1)
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("multi-stark only supports a window size of 2")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: PackedVal = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += Into::<PackedExtVal>::into(alpha_power) * x;
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let expr_array: [Self::Expr; N] = array.map(Into::into);
        self.accumulator += PackedExtVal::from_basis_coefficients_fn(|i| {
            let alpha_powers = &self.decomposed_alpha_powers[i]
                [self.constraint_index..(self.constraint_index + N)];
            PackedVal::packed_linear_combination::<N>(alpha_powers, &expr_array)
        });
        self.constraint_index += N;
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.stage_1_public_values
    }
}

impl<'a> ExtensionBuilder for ProverConstraintFolder<'a> {
    type EF = ExtVal;
    type ExprEF = PackedExtVal;
    type VarEF = PackedExtVal;

    /// Assert that an extension field expression is zero.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: PackedExtVal = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += Into::<PackedExtVal>::into(alpha_power) * x;
        self.constraint_index += 1;
    }
}

impl<'a> TwoStagedBuilder for ProverConstraintFolder<'a> {
    type MP = RowMajorMatrixView<'a, PackedExtVal>;

    type Stage2PublicVar = Self::EF;

    fn stage_2(&self) -> Self::MP {
        self.stage_2
    }

    fn stage_2_public_values(&self) -> &[Self::Stage2PublicVar] {
        self.stage_2_public_values
    }
}

impl<'a> AirBuilder for VerifierConstraintFolder<'a> {
    type F = Val;
    type Expr = ExtVal;
    type Var = ExtVal;
    type PreprocessedWindow = RowWindow<'a, ExtVal>;
    type MainWindow = RowWindow<'a, ExtVal>;
    type PublicVar = Val;

    fn main(&self) -> Self::MainWindow {
        RowWindow::from_two_rows(self.stage_1.top.values, self.stage_1.bottom.values)
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("multi-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: ExtVal = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.stage_1_public_values
    }
}

impl<'a> ExtensionBuilder for VerifierConstraintFolder<'a> {
    type EF = ExtVal;
    type ExprEF = ExtVal;
    type VarEF = ExtVal;

    /// Assert that an extension field expression is zero.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: ExtVal = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}

impl<'a> TwoStagedBuilder for VerifierConstraintFolder<'a> {
    type MP = ViewPair<'a, ExtVal>;

    type Stage2PublicVar = Self::EF;

    fn stage_2(&self) -> Self::MP {
        self.stage_2
    }

    fn stage_2_public_values(&self) -> &[Self::Stage2PublicVar] {
        self.stage_2_public_values
    }
}
