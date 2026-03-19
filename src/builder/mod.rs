use p3_air::ExtensionBuilder;
use p3_matrix::Matrix;

pub mod check;
pub mod folder;
pub mod symbolic;

/// Extension of [`ExtensionBuilder`] that provides access to a second-stage
/// trace and its associated public values (used by the lookup argument).
pub trait TwoStagedBuilder: ExtensionBuilder {
    /// Matrix type for the stage 2 trace window.
    type MP: Matrix<Self::VarEF>;

    /// Variable type for stage 2 public values.
    type Stage2PublicVar: Into<Self::ExprEF> + Copy;

    /// Returns the stage 2 trace window.
    fn stage_2(&self) -> Self::MP;

    /// Returns the stage 2 public values (lookup and fingerprint challenges,
    /// current accumulator, next accumulator).
    fn stage_2_public_values(&self) -> &[Self::Stage2PublicVar];
}
