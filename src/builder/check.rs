use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;

use crate::types::{ExtVal, Val};

use super::{PreprocessedBuilder, TwoStagedBuilder};

pub fn check_constraints<A>(
    air: &A,
    preprocessed: Option<&RowMajorMatrix<Val>>,
    stage_1: &RowMajorMatrix<Val>,
    stage_2: &RowMajorMatrix<ExtVal>,
    public_values: &[Val],
    stage_2_public_values: &[ExtVal],
) where
    A: for<'a> Air<DebugConstraintBuilder<'a>>,
{
    let height = stage_1.height();

    (0..height).for_each(|i| {
        let i_next = (i + 1) % height;

        let stage_1_local = stage_1.row_slice(i).unwrap(); // i < height so unwrap should never fail.
        let stage_1_next = stage_1.row_slice(i_next).unwrap(); // i_next < height so unwrap should never fail.
        let stage_1 = VerticalPair::new(
            RowMajorMatrixView::new_row(&*stage_1_local),
            RowMajorMatrixView::new_row(&*stage_1_next),
        );

        let stage_2_local = stage_2.row_slice(i).unwrap(); // i < height so unwrap should never fail.
        let stage_2_next = stage_2.row_slice(i_next).unwrap(); // i_next < height so unwrap should never fail.
        let stage_2 = VerticalPair::new(
            RowMajorMatrixView::new_row(&*stage_2_local),
            RowMajorMatrixView::new_row(&*stage_2_next),
        );
        let mut builder = DebugConstraintBuilder {
            row_index: i,
            preprocessed: None,
            stage_1,
            stage_2,
            public_values,
            stage_2_public_values,
            is_first_row: Val::from_bool(i == 0),
            is_last_row: Val::from_bool(i == height - 1),
            is_transition: Val::from_bool(i != height - 1),
        };
        // We must call `eval` on the same block as the `preprocessed` matrix view, otherwise the borrow checker will complain.
        // Mutation is used to remove code duplication.
        if let Some(preprocessed) = preprocessed {
            let preprocessed_local = preprocessed.row_slice(i).unwrap(); // i < height so unwrap should never fail.
            let preprocessed_next = preprocessed.row_slice(i_next).unwrap(); // i_next < height so unwrap should never fail.
            let preprocessed = Some(VerticalPair::new(
                RowMajorMatrixView::new_row(&*preprocessed_local),
                RowMajorMatrixView::new_row(&*preprocessed_next),
            ));
            builder.preprocessed = preprocessed;
            air.eval(&mut builder);
        } else {
            air.eval(&mut builder);
        }
    });
}

#[derive(Debug)]
pub struct DebugConstraintBuilder<'a> {
    /// The index of the row currently being evaluated.
    row_index: usize,
    /// A view of the current and next row as a vertical pair.
    preprocessed: Option<VerticalPair<RowMajorMatrixView<'a, Val>, RowMajorMatrixView<'a, Val>>>,
    stage_1: VerticalPair<RowMajorMatrixView<'a, Val>, RowMajorMatrixView<'a, Val>>,
    stage_2: VerticalPair<RowMajorMatrixView<'a, ExtVal>, RowMajorMatrixView<'a, ExtVal>>,
    /// The public values provided for constraint validation (e.g. inputs or outputs).
    public_values: &'a [Val],
    stage_2_public_values: &'a [ExtVal],
    /// A flag indicating whether this is the first row.
    is_first_row: Val,
    /// A flag indicating whether this is the last row.
    is_last_row: Val,
    /// A flag indicating whether this is a transition row (not the last row).
    is_transition: Val,
}

impl<'a> AirBuilder for DebugConstraintBuilder<'a> {
    type F = Val;
    type Expr = Val;
    type Var = Val;
    type M = VerticalPair<RowMajorMatrixView<'a, Val>, RowMajorMatrixView<'a, Val>>;

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
            panic!("only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        assert_eq!(
            x.into(),
            Val::ZERO,
            "constraints had nonzero value on row {}",
            self.row_index
        );
    }

    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        let x = x.into();
        let y = y.into();
        assert_eq!(
            x, y,
            "values didn't match on row {}: {} != {}",
            self.row_index, x, y
        );
    }
}

impl AirBuilderWithPublicValues for DebugConstraintBuilder<'_> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl PreprocessedBuilder for DebugConstraintBuilder<'_> {
    fn preprocessed(&self) -> Option<Self::M> {
        self.preprocessed
    }
}

impl<'a> ExtensionBuilder for DebugConstraintBuilder<'a> {
    type EF = ExtVal;
    type ExprEF = ExtVal;
    type VarEF = ExtVal;

    /// Assert that an extension field expression is zero.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x = x.into();
        assert_eq!(
            x,
            ExtVal::ZERO,
            "constraints had nonzero value on row {}",
            self.row_index
        );
    }

    fn assert_eq_ext<I1, I2>(&mut self, x: I1, y: I2)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
    {
        let x = x.into();
        let y = y.into();
        assert_eq!(
            x, y,
            "values didn't match on row {}: {} != {}",
            self.row_index, x, y
        );
    }
}

impl<'a> TwoStagedBuilder for DebugConstraintBuilder<'a> {
    type MP = VerticalPair<RowMajorMatrixView<'a, ExtVal>, RowMajorMatrixView<'a, ExtVal>>;

    type Stage2PublicVar = Self::EF;

    fn stage_2(&self) -> Self::MP {
        self.stage_2
    }

    fn stage_2_public_values(&self) -> &[Self::Stage2PublicVar] {
        self.stage_2_public_values
    }
}
