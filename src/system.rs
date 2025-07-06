use crate::{
    ensure,
    types::{ExtVal, StarkConfig, Val},
};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{ProverConstraintFolder, SymbolicAirBuilder};
use std::collections::BTreeMap as Map;

pub type Name = &'static str;

/// Each circuit is required to have at least 3 arguments. Namely,
/// the accumulator, the lookup challenge and the fingerprint challenge.
pub const MIN_IO_SIZE: usize = 3;

pub struct System<A> {
    pub circuits: Vec<Circuit<A>>,
    pub circuit_names: Map<Name, usize>,
}

pub struct Circuit<A> {
    pub air: A,
    pub stage1_width: usize,
    pub stage2_width: usize,
}

pub struct CircuitWitness {
    pub stage1: RowMajorMatrix<Val>,
    pub stage2: RowMajorMatrix<ExtVal>,
}

pub struct SystemWitness {
    pub circuits: Vec<CircuitWitness>,
}

pub struct Claim {
    pub circuit_name: Name,
    pub args: Vec<Val>,
}

pub struct Proof {
    pub claim: Claim,
}

impl<A: BaseAirWithPublicValues<Val>> Circuit<A> {
    pub fn is_well_formed(&self) -> Result<(), String> {
        let width = self.air.width();
        // As of now, only the minimum IO size is supported.
        ensure!(
            self.air.num_public_values() == MIN_IO_SIZE,
            "Incompatible IO size"
        );
        ensure!(
            self.stage1_width + self.stage2_width == width,
            "Incompatible widths"
        );
        Ok(())
    }
}

impl<A: BaseAirWithPublicValues<Val>> System<A> {
    pub fn is_well_formed(&self) -> Result<(), String> {
        ensure!(
            self.circuits.len() == self.circuit_names.len(),
            "Map of names is not well-formed"
        );
        let mut idxs = self.circuit_names.values().copied().collect::<Vec<_>>();
        idxs.sort();
        ensure!(
            idxs == (0..self.circuits.len()).collect::<Vec<_>>(),
            "Map of names is not well-formed"
        );
        self.circuits.iter().try_for_each(|c| c.is_well_formed())?;
        Ok(())
    }
}

impl<A: Air<SymbolicAirBuilder<Val>> + for<'a> Air<ProverConstraintFolder<'a, StarkConfig>>>
    System<A>
{
    #[allow(unused_variables)]
    pub fn prove(&self, claim: Claim, witness: SystemWitness) -> Proof {
        todo!()
    }

    #[allow(unused_variables)]
    pub fn verify(&self, proof: Proof) {}
}
