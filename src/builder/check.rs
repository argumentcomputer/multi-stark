use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;

use super::{PreprocessedBuilder, TwoStagedBuilder};

/// Runs constraint checks using a given AIR definition and trace matrix.
///
/// Iterates over every row in `main`, providing both the current and next row
/// (with wraparound) to the AIR logic. Also injects public values into the builder
/// for first/last row assertions.
///
/// # Arguments
/// - `air`: The AIR logic to run
/// - `main`: The trace matrix (rows of witness values)
/// - `public_values`: Public values provided to the builder
pub fn check_constraints<F, A>(
    air: &A,
    preprocessed: Option<&RowMajorMatrix<F>>,
    stage_1: &RowMajorMatrix<F>,
    stage_2: &RowMajorMatrix<F>,
    public_values: &[F],
) where
    F: Field,
    A: for<'a> Air<DebugConstraintBuilder<'a, F>>,
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
            is_first_row: F::from_bool(i == 0),
            is_last_row: F::from_bool(i == height - 1),
            is_transition: F::from_bool(i != height - 1),
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

/// A builder that runs constraint assertions during testing.
///
/// Used in conjunction with [`check_constraints`] to simulate
/// an execution trace and verify that the AIR logic enforces all constraints.
#[derive(Debug)]
pub struct DebugConstraintBuilder<'a, F: Field> {
    /// The index of the row currently being evaluated.
    row_index: usize,
    /// A view of the current and next row as a vertical pair.
    preprocessed: Option<VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>>,
    stage_1: VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>,
    stage_2: VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>,
    /// The public values provided for constraint validation (e.g. inputs or outputs).
    public_values: &'a [F],
    /// A flag indicating whether this is the first row.
    is_first_row: F,
    /// A flag indicating whether this is the last row.
    is_last_row: F,
    /// A flag indicating whether this is a transition row (not the last row).
    is_transition: F,
}

impl<'a, F> AirBuilder for DebugConstraintBuilder<'a, F>
where
    F: Field,
{
    type F = F;
    type Expr = F;
    type Var = F;
    type M = VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>;

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
}

impl<F: Field> AirBuilderWithPublicValues for DebugConstraintBuilder<'_, F> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<F: Field> TwoStagedBuilder for DebugConstraintBuilder<'_, F> {
    fn stage_2(&self) -> Self::M {
        self.stage_2
    }
}

impl<F: Field> PreprocessedBuilder for DebugConstraintBuilder<'_, F> {
    fn preprocessed(&self) -> Option<Self::M> {
        self.preprocessed
    }
}
