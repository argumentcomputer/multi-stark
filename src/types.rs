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
    4,
>;
pub type ExtMmcs = ExtensionMmcs<Val, ExtVal, Mmcs>;
pub type Pcs = TwoAdicFriPcs<Val, Dft, Mmcs, ExtMmcs>;

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

    pub fn new(
        commitment_parameters: &CommitmentParameters,
        fri_parameters: &FriParameters,
    ) -> Self {
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
    pub fn new(commitment_parameters: &CommitmentParameters) -> Self {
        let dummy_parameters = &FriParameters {
            log_final_poly_len: 0,
            num_queries: 0,
            proof_of_work_bits: 0,
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

pub struct CommitmentParameters {
    pub log_blowup: usize,
}

pub struct FriParameters {
    pub log_final_poly_len: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
}

type KeccakCompressionFunction =
    CompressionFunctionFromHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>, 2, 4>;
type Dft = Radix2DitParallel<Val>;

fn new_mmcs() -> Mmcs {
    let u64_hash = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF {});
    let field_hash = SerializingHasher::new(u64_hash);
    let compress = KeccakCompressionFunction::new(u64_hash);
    Mmcs::new(field_hash, compress)
}

fn new_pcs(commitment_parameters: &CommitmentParameters, fri_parameters: &FriParameters) -> Pcs {
    let val_mmcs = new_mmcs();
    let mmcs = ExtensionMmcs::new(val_mmcs.clone());
    let inner_parameters = InnerFriParameters {
        log_blowup: commitment_parameters.log_blowup,
        log_final_poly_len: fri_parameters.log_final_poly_len,
        num_queries: fri_parameters.num_queries,
        proof_of_work_bits: fri_parameters.proof_of_work_bits,
        mmcs,
    };
    let dft = Dft::default();
    Pcs::new(dft, val_mmcs, inner_parameters)
}
