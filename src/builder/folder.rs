/// Constraint folders for the prover and verifier, adapted from Plonky3.
use p3_air::{AirBuilder, ExtensionBuilder, RowWindow};
use p3_field::{Algebra, BasedVectorSpace, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;

use super::TwoStagedBuilder;

/// Packed base-field values of `F`.
type PackedVal<F> = <F as Field>::Packing;
/// Packed extension-field values of `EF` over `F`.
type PackedExt<F, EF> = <EF as ExtensionField<F>>::ExtensionPacking;

pub struct ProverConstraintFolder<'a, F: Field, EF: ExtensionField<F>> {
    pub preprocessed: RowWindow<'a, PackedVal<F>>,
    pub stage_1: RowMajorMatrixView<'a, PackedVal<F>>,
    pub stage_2: RowMajorMatrixView<'a, PackedExt<F, EF>>,
    pub stage_1_public_values: &'a [F],
    pub stage_2_public_values: &'a [EF],
    pub is_first_row: PackedVal<F>,
    pub is_last_row: PackedVal<F>,
    pub is_transition: PackedVal<F>,
    pub alpha_powers: &'a [EF],
    pub decomposed_alpha_powers: &'a [Vec<F>],
    pub accumulator: PackedExt<F, EF>,
    pub constraint_index: usize,
}

type ViewPair<'a, T> = VerticalPair<RowMajorMatrixView<'a, T>, RowMajorMatrixView<'a, T>>;

pub struct VerifierConstraintFolder<'a, F: Field, EF: ExtensionField<F>> {
    pub preprocessed: RowWindow<'a, EF>,
    pub stage_1: ViewPair<'a, EF>,
    pub stage_2: ViewPair<'a, EF>,
    pub stage_1_public_values: &'a [F],
    pub stage_2_public_values: &'a [EF],
    pub is_first_row: EF,
    pub is_last_row: EF,
    pub is_transition: EF,
    pub alpha: EF,
    pub accumulator: EF,
}

impl<'a, F: Field, EF: ExtensionField<F>> AirBuilder for ProverConstraintFolder<'a, F, EF> {
    type F = F;
    type Expr = PackedVal<F>;
    type Var = PackedVal<F>;
    type PreprocessedWindow = RowWindow<'a, PackedVal<F>>;
    type MainWindow = RowWindow<'a, PackedVal<F>>;
    type PublicVar = F;

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
        let x: PackedVal<F> = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += Into::<PackedExt<F, EF>>::into(alpha_power) * x;
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let expr_array: [Self::Expr; N] = array.map(Into::into);
        self.accumulator += PackedExt::<F, EF>::from_basis_coefficients_fn(|i| {
            let alpha_powers = &self.decomposed_alpha_powers[i]
                [self.constraint_index..(self.constraint_index + N)];
            PackedVal::<F>::batched_linear_combination(&expr_array, alpha_powers)
        });
        self.constraint_index += N;
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.stage_1_public_values
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for ProverConstraintFolder<'_, F, EF> {
    type EF = EF;
    type ExprEF = PackedExt<F, EF>;
    type VarEF = PackedExt<F, EF>;

    /// Assert that an extension field expression is zero.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: PackedExt<F, EF> = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += Into::<PackedExt<F, EF>>::into(alpha_power) * x;
        self.constraint_index += 1;
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> TwoStagedBuilder for ProverConstraintFolder<'a, F, EF> {
    type MP = RowMajorMatrixView<'a, PackedExt<F, EF>>;

    type Stage2PublicVar = Self::EF;

    fn stage_2(&self) -> Self::MP {
        self.stage_2
    }

    fn stage_2_public_values(&self) -> &[Self::Stage2PublicVar] {
        self.stage_2_public_values
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> AirBuilder for VerifierConstraintFolder<'a, F, EF> {
    type F = F;
    type Expr = EF;
    type Var = EF;
    type PreprocessedWindow = RowWindow<'a, EF>;
    type MainWindow = RowWindow<'a, EF>;
    type PublicVar = F;

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
        let x: EF = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.stage_1_public_values
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for VerifierConstraintFolder<'_, F, EF> {
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;

    /// Assert that an extension field expression is zero.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: EF = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> TwoStagedBuilder for VerifierConstraintFolder<'a, F, EF> {
    type MP = ViewPair<'a, EF>;

    type Stage2PublicVar = Self::EF;

    fn stage_2(&self) -> Self::MP {
        self.stage_2
    }

    fn stage_2_public_values(&self) -> &[Self::Stage2PublicVar] {
        self.stage_2_public_values
    }
}
