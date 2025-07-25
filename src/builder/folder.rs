/// Adapted from Plonky3's `https://github.com/Plonky3/Plonky3/blob/main/uni-stark/src/folder.rs`
use p3_air::{AirBuilder, AirBuilderWithPublicValues, PairBuilder};
use p3_field::{BasedVectorSpace, PackedField};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;

use crate::types::{ExtVal, PackedExtVal, PackedVal, Val};

use super::TwoStagedBuilder;

#[derive(Debug)]
pub struct ProverConstraintFolder<'a> {
    pub preprocessed: Option<RowMajorMatrixView<'a, PackedVal>>,
    pub stage_1: RowMajorMatrixView<'a, PackedVal>,
    pub stage_2: RowMajorMatrixView<'a, PackedVal>,
    pub public_values: &'a [Val],
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
    pub preprocessed: Option<ViewPair<'a, ExtVal>>,
    pub stage_1: ViewPair<'a, ExtVal>,
    pub stage_2: ViewPair<'a, ExtVal>,
    pub public_values: &'a [Val],
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
    type M = RowMajorMatrixView<'a, PackedVal>;

    #[inline]
    fn main(&self) -> Self::M {
        self.stage_1
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
}

impl AirBuilderWithPublicValues for ProverConstraintFolder<'_> {
    type PublicVar = Self::F;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<'a> PairBuilder for ProverConstraintFolder<'a> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.unwrap()
    }
}

impl<'a> TwoStagedBuilder for ProverConstraintFolder<'a> {
    fn stage_2(&self) -> Self::M {
        self.stage_2
    }
}

impl<'a> AirBuilder for VerifierConstraintFolder<'a> {
    type F = Val;
    type Expr = ExtVal;
    type Var = ExtVal;
    type M = ViewPair<'a, ExtVal>;

    fn main(&self) -> Self::M {
        self.stage_1
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
}

impl AirBuilderWithPublicValues for VerifierConstraintFolder<'_> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<'a> PairBuilder for VerifierConstraintFolder<'a> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.unwrap()
    }
}

impl<'a> TwoStagedBuilder for VerifierConstraintFolder<'a> {
    fn stage_2(&self) -> Self::M {
        self.stage_2
    }
}
