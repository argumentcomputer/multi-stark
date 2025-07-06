use crate::types::{ExtVal, StarkConfig, Val};
use p3_air::Air;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{ProverConstraintFolder, SymbolicAirBuilder};
use std::collections::BTreeMap as Map;

pub type Name = &'static str;

pub struct System<A> {
    pub circuits: Vec<Circuit<A>>,
    pub circuit_names: Map<Name, usize>,
}

pub struct Circuit<A> {
    pub air: A,
    pub stage1_width: usize,
    pub stage2_width: usize,
    pub stage2_challenges: usize,
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
