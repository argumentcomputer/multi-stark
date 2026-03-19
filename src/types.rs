use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_commit::{ExtensionMmcs, Pcs as PcsTrait};
use p3_dft::Radix2DitParallel;
use p3_field::{ExtensionField, Field, extension::BinomialExtensionField};
use p3_fri::{FriParameters as InnerFriParameters, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};

pub type Val = Goldilocks;
pub type PackedVal = <Val as Field>::Packing;
pub type ExtVal = BinomialExtensionField<Val, 2>;
pub type PackedExtVal = <ExtVal as ExtensionField<Val>>::ExtensionPacking;
pub type Challenger = SerializingChallenger64<Val, HashChallenger<u8, Keccak256Hash, 32>>;
pub type Mmcs = MerkleTreeMmcs<
    [Val; p3_keccak::VECTOR_LEN],
    [u64; p3_keccak::VECTOR_LEN],
    SerializingHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>>,
    KeccakCompressionFunction,
    2,
    4,
>;
pub type ExtMmcs = ExtensionMmcs<Val, ExtVal, Mmcs>;
pub type Pcs = TwoAdicFriPcs<Val, Dft, Mmcs, ExtMmcs>;

/// Configuration for the STARK prover and verifier, bundling the PCS and an
/// initial challenger state.
#[derive(Debug)]
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
        let challenger = Challenger::from_hasher(vec![], Keccak256Hash {});
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

type KeccakCompressionFunction =
    CompressionFunctionFromHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>, 2, 4>;
type Dft = Radix2DitParallel<Val>;

fn new_mmcs(cap_height: usize) -> Mmcs {
    let u64_hash = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF {});
    let field_hash = SerializingHasher::new(u64_hash);
    let compress = KeccakCompressionFunction::new(u64_hash);
    Mmcs::new(field_hash, compress, cap_height)
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
