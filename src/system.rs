use crate::{
    builder::symbolic::{SymbolicAirBuilder, get_max_constraint_degree, get_symbolic_constraints},
    ensure_eq,
    types::Val,
};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use std::collections::BTreeMap as Map;

pub type Name = String;

/// Each circuit is required to have at least 3 arguments. Namely,
/// the accumulator, the lookup challenge and the fingerprint challenge.
pub const MIN_IO_SIZE: usize = 3;

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
    pub stage1_width: usize,
    pub stage2_width: usize,
}

impl<A> Circuit<A> {
    pub fn width(&self) -> usize {
        self.stage1_width + self.stage2_width + self.preprocessed_width
    }
}

pub struct CircuitWitness {
    pub trace: RowMajorMatrix<Val>,
}

pub struct SystemWitness {
    pub circuits: Vec<CircuitWitness>,
}

impl<A: BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>> Circuit<A> {
    pub fn from_air_single_stage(air: A) -> Result<Self, String> {
        let io_size = air.num_public_values();
        ensure_eq!(io_size, MIN_IO_SIZE, "Incompatible IO size");
        let preprocessed_width = air.preprocessed_trace().map_or(0, |mat| mat.width());
        let constraint_count = get_symbolic_constraints(&air, preprocessed_width, io_size).len();
        let max_constraint_degree = get_max_constraint_degree(&air, preprocessed_width, io_size);
        let stage1_width = air.width() - preprocessed_width;
        let stage2_width = 0;
        Ok(Self {
            air,
            max_constraint_degree,
            preprocessed_width,
            constraint_count,
            stage1_width,
            stage2_width,
        })
    }

    pub fn is_well_formed(&self) -> Result<(), String> {
        let io_size = self.air.num_public_values();
        let preprocessed_width = self.air.preprocessed_trace().map_or(0, |mat| mat.width());
        let constraint_count =
            get_symbolic_constraints(&self.air, preprocessed_width, io_size).len();
        let max_constraint_count =
            get_max_constraint_degree(&self.air, preprocessed_width, io_size);
        let width = self.air.width();
        // As of now, only the minimum IO size is supported.
        ensure_eq!(io_size, MIN_IO_SIZE, "Incompatible IO size");
        ensure_eq!(
            self.constraint_count,
            constraint_count,
            "Incompatible constraint count"
        );
        ensure_eq!(
            self.max_constraint_degree,
            max_constraint_count,
            "Incompatible constraint degree"
        );
        ensure_eq!(
            self.preprocessed_width,
            preprocessed_width,
            "Incompatible widths"
        );
        ensure_eq!(self.width(), width, "Incompatible widths");
        Ok(())
    }
}

impl<A: BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>> System<A> {
    pub fn is_well_formed(&self) -> Result<(), String> {
        ensure_eq!(
            self.circuits.len(),
            self.circuit_names.len(),
            "Map of names is not well-formed"
        );
        let mut idxs = self.circuit_names.values().copied().collect::<Vec<_>>();
        idxs.sort();
        ensure_eq!(
            idxs,
            (0..self.circuits.len()).collect::<Vec<_>>(),
            "Map of names is not well-formed"
        );
        self.circuits.iter().try_for_each(|c| c.is_well_formed())?;
        Ok(())
    }
}
