use crate::{
    builder::{
        TwoStagedAir,
        symbolic::{SymbolicAirBuilder, get_max_constraint_degree, get_symbolic_constraints},
    },
    ensure_eq,
    types::Val,
};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

/// Each circuit is required to have at least 4 arguments. Namely, the lookup challenge,
/// fingerprint challenge, current accumulator and next accumulator
pub const MIN_IO_SIZE: usize = 4;

pub struct System<A> {
    pub circuits: Vec<Circuit<A>>,
}

impl<A> System<A> {
    #[inline]
    pub fn new(circuits: impl IntoIterator<Item = Circuit<A>>) -> Self {
        Self {
            circuits: circuits.into_iter().collect(),
        }
    }
}

pub struct Circuit<A> {
    pub air: A,
    pub constraint_count: usize,
    pub max_constraint_degree: usize,
    pub preprocessed_width: usize,
    pub stage_1_width: usize,
    pub stage_2_width: usize,
}

#[derive(Clone)]
pub struct CircuitWitness<Val> {
    pub trace: RowMajorMatrix<Val>,
}

#[derive(Clone)]
pub struct SystemWitness<Val> {
    pub circuits: Vec<CircuitWitness<Val>>,
}

impl<A: BaseAirWithPublicValues<Val> + TwoStagedAir<Val> + Air<SymbolicAirBuilder<Val>>>
    Circuit<A>
{
    pub fn from_air(air: A) -> Result<Self, String> {
        let io_size = air.num_public_values();
        ensure_eq!(io_size, MIN_IO_SIZE, "Incompatible IO size");
        let stage_1_width = air.width();
        let stage_2_width = air.stage_2_width();
        let preprocessed_width = air.preprocessed_trace().map_or(0, |mat| mat.width());
        let constraint_count = get_symbolic_constraints(
            &air,
            preprocessed_width,
            stage_1_width,
            stage_2_width,
            io_size,
        )
        .len();
        let max_constraint_degree = get_max_constraint_degree(
            &air,
            preprocessed_width,
            stage_1_width,
            stage_2_width,
            io_size,
        );
        Ok(Self {
            air,
            max_constraint_degree,
            preprocessed_width,
            constraint_count,
            stage_1_width,
            stage_2_width,
        })
    }
}
