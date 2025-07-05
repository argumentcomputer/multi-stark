use crate::types::{Challenger, ExtVal, Pcs};
use p3_commit::Pcs as PcsTrait;

pub struct Proof {
    pub commitments: Commitments,
    pub opened_values: OpenedValues,
    pub opening_proof: PcsProof,
    pub degree_bits: usize,
}

type Commitment = <Pcs as PcsTrait<ExtVal, Challenger>>::Commitment;
type PcsProof = <Pcs as PcsTrait<ExtVal, Challenger>>::Proof;

pub struct Commitments {
    pub trace: Commitment,
    pub quotient_chunks: Commitment,
    pub random: Option<Commitment>,
}

pub struct OpenedValues {
    pub trace_local: Vec<ExtVal>,
    pub trace_next: Vec<ExtVal>,
    pub quotient_chunks: Vec<Vec<ExtVal>>,
    pub random: Option<Vec<ExtVal>>,
}
