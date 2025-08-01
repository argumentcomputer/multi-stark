use p3_air::AirBuilder;

pub mod check;
pub mod folder;
pub mod symbolic;

pub trait TwoStagedBuilder: AirBuilder {
    fn stage_2(&self) -> Self::M;
}

pub trait PreprocessedBuilder: AirBuilder {
    fn preprocessed(&self) -> Option<Self::M>;
}
