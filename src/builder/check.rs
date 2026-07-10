use p3_air::{Air, AirBuilder, ExtensionBuilder, RowWindow};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;

use p3_field::{ExtensionField, Field};

use super::TwoStagedBuilder;

pub fn check_constraints<F, EF, A>(
    air: &A,
    preprocessed: Option<&RowMajorMatrix<F>>,
    stage_1: &RowMajorMatrix<F>,
    stage_2: &RowMajorMatrix<EF>,
    public_values: &[F],
    stage_2_public_values: &[EF],
) where
    F: Field,
    EF: ExtensionField<F>,
    A: for<'a> Air<DebugConstraintBuilder<'a, F, EF>>,
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
            preprocessed: RowWindow::from_two_rows(&[], &[]),
            stage_1,
            stage_2,
            public_values,
            stage_2_public_values,
            is_first_row: F::from_bool(i == 0),
            is_last_row: F::from_bool(i == height - 1),
            is_transition: F::from_bool(i != height - 1),
        };
        // We must call `eval` on the same block as the `preprocessed` matrix view, otherwise the borrow checker will complain.
        // Mutation is used to remove code duplication.
        if let Some(preprocessed) = preprocessed {
            let preprocessed_local = preprocessed.row_slice(i).unwrap(); // i < height so unwrap should never fail.
            let preprocessed_next = preprocessed.row_slice(i_next).unwrap(); // i_next < height so unwrap should never fail.
            builder.preprocessed =
                RowWindow::from_two_rows(&preprocessed_local, &preprocessed_next);
            air.eval(&mut builder);
        } else {
            air.eval(&mut builder);
        }
    });
}

#[derive(Debug)]
pub struct DebugConstraintBuilder<'a, F: Field, EF: ExtensionField<F>> {
    /// The index of the row currently being evaluated.
    row_index: usize,
    /// A two-row window over the preprocessed trace (current and next row).
    preprocessed: RowWindow<'a, F>,
    stage_1: VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>,
    stage_2: VerticalPair<RowMajorMatrixView<'a, EF>, RowMajorMatrixView<'a, EF>>,
    /// The public values provided for constraint validation (e.g. inputs or outputs).
    public_values: &'a [F],
    stage_2_public_values: &'a [EF],
    /// A flag indicating whether this is the first row.
    is_first_row: F,
    /// A flag indicating whether this is the last row.
    is_last_row: F,
    /// A flag indicating whether this is a transition row (not the last row).
    is_transition: F,
}

impl<'a, F: Field, EF: ExtensionField<F>> AirBuilder for DebugConstraintBuilder<'a, F, EF> {
    type F = F;
    type Expr = F;
    type Var = F;
    type PreprocessedWindow = RowWindow<'a, F>;
    type MainWindow = RowWindow<'a, F>;
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
            panic!("only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        assert_eq!(
            x.into(),
            F::ZERO,
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

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for DebugConstraintBuilder<'_, F, EF> {
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;

    /// Assert that an extension field expression is zero.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x = x.into();
        assert_eq!(
            x,
            EF::ZERO,
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

impl<'a, F: Field, EF: ExtensionField<F>> TwoStagedBuilder for DebugConstraintBuilder<'a, F, EF> {
    type MP = VerticalPair<RowMajorMatrixView<'a, EF>, RowMajorMatrixView<'a, EF>>;

    type Stage2PublicVar = Self::EF;

    fn stage_2(&self) -> Self::MP {
        self.stage_2
    }

    fn stage_2_public_values(&self) -> &[Self::Stage2PublicVar] {
        self.stage_2_public_values
    }
}
