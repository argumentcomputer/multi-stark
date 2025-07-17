use p3_air::{AirBuilder, BaseAir};

pub mod check;
pub mod folder;
pub mod symbolic;

pub trait TwoStagedAir<F>: BaseAir<F> {
    fn stage_2_width(&self) -> usize;
}

pub trait TwoStagedBuilder: AirBuilder {
    fn stage_2(&self) -> Self::M;
}
