//! An alternative STARK configuration: KoalaBear field with a degree-4
//! binomial extension, Poseidon2 hashing, and a FRI-based PCS.
//!
//! Unlike the byte-oriented reference configuration
//! ([`crate::types::GoldilocksBlake3Config`]), hashing here is field-native:
//! the Merkle tree and the Fiat-Shamir challenger both run on a Poseidon2
//! permutation over KoalaBear, which makes this configuration friendly to
//! recursive verification.

use crate::config::StarkGenericConfig;
use crate::types::{CommitmentParameters, FriParameters};
use p3_challenger::{CanObserve, DuplexChallenger};
use p3_commit::{ExtensionMmcs, Pcs as PcsTrait};
use p3_dft::Radix2DitParallel;
use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField, extension::BinomialExtensionField,
};
use p3_fri::{FriParameters as InnerFriParameters, TwoAdicFriPcs};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear, default_koalabear_poseidon2_16};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

pub type Val = KoalaBear;
pub type PackedVal = <Val as Field>::Packing;
pub type ExtVal = BinomialExtensionField<Val, 4>;
pub type PackedExtVal = <ExtVal as ExtensionField<Val>>::ExtensionPacking;
pub type Perm = Poseidon2KoalaBear<16>;
pub type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
pub type Mmcs = MerkleTreeMmcs<PackedVal, PackedVal, Poseidon2Sponge, Poseidon2Compression, 2, 8>;
pub type ExtMmcs = ExtensionMmcs<Val, ExtVal, Mmcs>;
pub type Pcs = TwoAdicFriPcs<Val, Dft, Mmcs, ExtMmcs>;

pub type Commitment = <Pcs as PcsTrait<ExtVal, Challenger>>::Commitment;
pub type Domain = <Pcs as PcsTrait<ExtVal, Challenger>>::Domain;
pub type ProverData = <Pcs as PcsTrait<ExtVal, Challenger>>::ProverData;
pub type EvaluationsOnDomain<'a> = <Pcs as PcsTrait<ExtVal, Challenger>>::EvaluationsOnDomain<'a>;
pub type PcsError = <Pcs as PcsTrait<ExtVal, Challenger>>::Error;
pub type PcsProof = <Pcs as PcsTrait<ExtVal, Challenger>>::Proof;

/// A KoalaBear/Poseidon2 [`StarkGenericConfig`] implementation.
pub struct KoalaBearPoseidon2Config {
    /// The PCS used to commit polynomials and prove opening proofs.
    pcs: Pcs,
    /// The Poseidon2 permutation driving the challenger.
    perm: Perm,
    /// Field elements observed into every fresh challenger: a
    /// domain-separation tag followed by a digest of all protocol
    /// parameters.
    challenger_seed: Vec<Val>,
    /// Largest log2 degree the PCS can commit to and open.
    max_log_degree: usize,
    /// Largest quotient degree the PCS can serve trace evaluations for
    /// (the FRI blowup factor).
    max_quotient_degree: usize,
}

impl KoalaBearPoseidon2Config {
    pub fn new(commitment_parameters: CommitmentParameters, fri_parameters: FriParameters) -> Self {
        let perm = default_koalabear_poseidon2_16();
        let pcs = new_pcs(&perm, commitment_parameters, fri_parameters);
        // Seed the challenger with a protocol tag for domain separation,
        // followed by every protocol parameter. Binding the parameters into
        // the seed means transcripts produced under different parameters
        // never collide (see the transcript contract on
        // [`StarkGenericConfig::initialise_challenger`]).
        let mut challenger_seed: Vec<Val> = b"multi-stark/v0"
            .iter()
            .map(|&byte| Val::from_u8(byte))
            .collect();
        challenger_seed.extend(
            [
                commitment_parameters.log_blowup,
                commitment_parameters.cap_height,
                fri_parameters.log_final_poly_len,
                fri_parameters.max_log_arity,
                fri_parameters.num_queries,
                fri_parameters.commit_proof_of_work_bits,
                fri_parameters.query_proof_of_work_bits,
            ]
            .map(Val::from_usize),
        );
        let max_log_degree = Val::TWO_ADICITY - commitment_parameters.log_blowup;
        let max_quotient_degree = 1 << commitment_parameters.log_blowup;
        Self {
            pcs,
            perm,
            challenger_seed,
            max_log_degree,
            max_quotient_degree,
        }
    }
}

impl StarkGenericConfig for KoalaBearPoseidon2Config {
    type Pcs = Pcs;
    type Challenge = ExtVal;
    type Challenger = Challenger;

    fn pcs(&self) -> &Pcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Challenger {
        let mut challenger = Challenger::new(self.perm.clone());
        for value in &self.challenger_seed {
            challenger.observe(*value);
        }
        challenger
    }

    fn max_log_degree(&self) -> usize {
        self.max_log_degree
    }

    fn max_quotient_degree(&self) -> usize {
        self.max_quotient_degree
    }
}

type Poseidon2Sponge = PaddingFreeSponge<Perm, 16, 8, 8>;
type Poseidon2Compression = TruncatedPermutation<Perm, 2, 8, 16>;
type Dft = Radix2DitParallel<Val>;

fn new_mmcs(perm: &Perm, cap_height: usize) -> Mmcs {
    let hash = Poseidon2Sponge::new(perm.clone());
    let compress = Poseidon2Compression::new(perm.clone());
    Mmcs::new(hash, compress, cap_height)
}

fn new_pcs(
    perm: &Perm,
    commitment_parameters: CommitmentParameters,
    fri_parameters: FriParameters,
) -> Pcs {
    let val_mmcs = new_mmcs(perm, commitment_parameters.cap_height);
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
