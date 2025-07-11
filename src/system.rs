use crate::{
    builder::symbolic::{SymbolicAirBuilder, get_max_constraint_degree, get_symbolic_constraints},
    ensure_eq,
    types::Val,
};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use std::collections::BTreeMap as Map;

pub type Name = String;

/// Each circuit is required to have at least 4 arguments. Namely, the lookup challenge,
/// fingerprint challenge, current accumulator and next accumulator
pub const MIN_IO_SIZE: usize = 4;

pub struct System<A> {
    pub circuits: Vec<Circuit<A>>,
    pub circuit_names: Map<Name, usize>,
}

impl<A> System<A> {
    pub fn new<Str: ToString, Iter: Iterator<Item = (Str, Circuit<A>)>>(iter: Iter) -> Self {
        let mut circuits = vec![];
        let mut circuit_names = Map::new();
        iter.for_each(|(name, circuit)| {
            let idx = circuits.len();
            if let Some(prev_idx) = circuit_names.insert(name.to_string(), idx) {
                eprintln!(
                    "Warning: circuit of name `{}` was redefined",
                    name.to_string()
                );
                circuits[prev_idx] = circuit;
            } else {
                circuits.push(circuit);
            }
        });
        Self {
            circuits,
            circuit_names,
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

impl<A: BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>> Circuit<A> {
    pub fn from_air(air: A, stage_2_width: usize) -> Result<Self, String> {
        let io_size = air.num_public_values();
        ensure_eq!(io_size, MIN_IO_SIZE, "Incompatible IO size");
        let stage_1_width = air.width();
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
