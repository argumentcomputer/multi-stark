use p3_challenger::DuplexChallenger;
use p3_commit::{ExtensionMmcs, Pcs as PcsTrait};
#[cfg(not(feature = "cuda"))]
use p3_dft::Radix2DitParallel;
use p3_field::{ExtensionField, Field, extension::BinomialExtensionField};
use p3_fri::{FriParameters as InnerFriParameters, TwoAdicFriPcs};
use p3_goldilocks::{
    Goldilocks, Poseidon2Goldilocks, default_goldilocks_poseidon2_8,
    default_goldilocks_poseidon2_16,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

#[cfg(feature = "cuda")]
use crate::cuda::CudaRadix2Dft;

pub type Val = Goldilocks;
pub type PackedVal = <Val as Field>::Packing;
pub type ExtVal = BinomialExtensionField<Val, 2>;
pub type PackedExtVal = <ExtVal as ExtensionField<Val>>::ExtensionPacking;

/// Width-16 Poseidon2 permutation: used for leaf hashing and the duplex challenger,
/// where high rate amortizes the larger permutation cost across many absorbed elements.
pub type PermSponge = Poseidon2Goldilocks<16>;
/// Width-8 Poseidon2 permutation: used for the 2-to-1 Merkle compression function,
/// where each call hashes a fixed full-state input and the smaller permutation is faster.
pub type PermCompress = Poseidon2Goldilocks<8>;
/// Sponge hash for Merkle leaves: rate 12, capacity 4, digest 4 elements (128-bit
/// indifferentiability bound = |F|^(cap/2) = 2^128 over Goldilocks).
pub type Hash = PaddingFreeSponge<PermSponge, 16, 12, 4>;
/// Pairwise inner-node compression: 2 children × 4 digest elements absorbed
/// into a width-8 permutation; first 4 elements squeezed as the new digest.
pub type Compress = TruncatedPermutation<PermCompress, 2, 4, 8>;
pub type Challenger = DuplexChallenger<Val, PermSponge, 16, 12>;
pub type Mmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, Hash, Compress, 2, 4>;
pub type ExtMmcs = ExtensionMmcs<Val, ExtVal, Mmcs>;
pub type Pcs = TwoAdicFriPcs<Val, Dft, Mmcs, ExtMmcs>;

/// Configuration for the STARK prover and verifier, bundling the PCS and an
/// initial challenger state.
pub struct StarkConfig {
    /// The PCS used to commit polynomials and prove opening proofs.
    pcs: Pcs,
    /// An initialised instance of the challenger.
    challenger: Challenger,
}

pub type Commitment = <Pcs as PcsTrait<ExtVal, Challenger>>::Commitment;
pub type Domain = <Pcs as PcsTrait<ExtVal, Challenger>>::Domain;
pub type ProverData = <Pcs as PcsTrait<ExtVal, Challenger>>::ProverData;
pub type EvaluationsOnDomain<'a> = <Pcs as PcsTrait<ExtVal, Challenger>>::EvaluationsOnDomain<'a>;
pub type PcsError = <Pcs as PcsTrait<ExtVal, Challenger>>::Error;
pub type PcsProof = <Pcs as PcsTrait<ExtVal, Challenger>>::Proof;

impl StarkConfig {
    pub fn pcs(&self) -> &Pcs {
        &self.pcs
    }
    pub fn initialise_challenger(&self) -> Challenger {
        self.challenger.clone()
    }

    pub fn new(commitment_parameters: CommitmentParameters, fri_parameters: FriParameters) -> Self {
        let pcs = new_pcs(commitment_parameters, fri_parameters);
        let challenger = Challenger::new(default_goldilocks_poseidon2_16());
        Self { pcs, challenger }
    }
}

/// The committer is able to commit to polynomials but not open them.
/// This is used for preprocessed traces.
pub struct Committer {
    pcs: Pcs,
}

impl Committer {
    pub fn new(commitment_parameters: CommitmentParameters) -> Self {
        let dummy_parameters = FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 0,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        };
        let pcs = new_pcs(commitment_parameters, dummy_parameters);
        Self { pcs }
    }

    pub fn natural_domain_for_degree(&self, degree: usize) -> Domain {
        <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(&self.pcs, degree)
    }

    pub fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Domain, RowMajorMatrix<Val>)>,
    ) -> (Commitment, ProverData) {
        <Pcs as PcsTrait<ExtVal, Challenger>>::commit(&self.pcs, evaluations)
    }
}

#[derive(Clone, Copy)]
pub struct CommitmentParameters {
    pub log_blowup: usize,
    /// Height of the Merkle cap (number of top layers included in the commitment).
    /// A cap height of 0 means only the root is committed.
    pub cap_height: usize,
}

/// Parameters controlling the FRI protocol.
///
/// These parameters determine the concrete security level. The FRI soundness
/// error is approximately `ρ^num_queries` where `ρ = 2^(-log_blowup)` (set in
/// [`CommitmentParameters`]). For example, `log_blowup = 1` with
/// `num_queries = 100` gives ~2^(-100) soundness error from FRI queries alone.
/// The PoW bits add grinding cost on top of this bound.
#[derive(Clone, Copy)]
pub struct FriParameters {
    /// Log2 of the degree of the final polynomial (0 means a constant).
    pub log_final_poly_len: usize,
    /// Maximum folding arity per FRI round (log2). A value of 1 means binary folding.
    pub max_log_arity: usize,
    /// Number of query repetitions for soundness amplification.
    pub num_queries: usize,
    /// Number of bits for the PoW phase before sampling _each_ batching challenge.
    pub commit_proof_of_work_bits: usize,
    /// Number of bits for the PoW phase before sampling the queries.
    pub query_proof_of_work_bits: usize,
}

#[cfg(not(feature = "cuda"))]
type Dft = Radix2DitParallel<Val>;
#[cfg(feature = "cuda")]
type Dft = CudaRadix2Dft<Val>;

fn new_mmcs(cap_height: usize) -> Mmcs {
    let hash = Hash::new(default_goldilocks_poseidon2_16());
    let compress = Compress::new(default_goldilocks_poseidon2_8());
    Mmcs::new(hash, compress, cap_height)
}

fn new_pcs(commitment_parameters: CommitmentParameters, fri_parameters: FriParameters) -> Pcs {
    let val_mmcs = new_mmcs(commitment_parameters.cap_height);
    let mmcs = ExtensionMmcs::new(val_mmcs.clone());
    let inner_parameters = InnerFriParameters {
        log_blowup: commitment_parameters.log_blowup,
        log_final_poly_len: fri_parameters.log_final_poly_len,
        max_log_arity: fri_parameters.max_log_arity,
        num_queries: fri_parameters.num_queries,
        commit_proof_of_work_bits: fri_parameters.commit_proof_of_work_bits,
        query_proof_of_work_bits: fri_parameters.query_proof_of_work_bits,
        mmcs,
    };
    let dft = Dft::default();
    Pcs::new(dft, val_mmcs, inner_parameters)
}
