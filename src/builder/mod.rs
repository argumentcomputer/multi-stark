use p3_air::{AirBuilder, ExtensionBuilder};
use p3_matrix::Matrix;

pub mod check;
pub mod folder;
pub mod symbolic;

pub trait PreprocessedBuilder: AirBuilder {
    fn preprocessed(&self) -> Option<Self::M>;
}

pub trait TwoStagedBuilder: ExtensionBuilder {
    type MP: Matrix<Self::VarEF>;

    type Stage2PublicVar: Into<Self::ExprEF> + Copy;

    fn stage_2(&self) -> Self::MP;

    fn stage_2_public_values(&self) -> &[Self::Stage2PublicVar];
}
