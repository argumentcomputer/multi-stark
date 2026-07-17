//! An alternative STARK configuration: BabyBear field with a degree-4
//! binomial extension, Blake3 hashing, and a FRI-based PCS.
//!
//! This is the 32-bit-field analogue of the reference configuration
//! ([`crate::types::GoldilocksBlake3Config`]): field elements are serialized
//! to bytes and hashed with Blake3, both in the Merkle tree and in the
//! Fiat-Shamir challenger.

use crate::config::StarkGenericConfig;
use crate::types::{CommitmentParameters, FriParameters};
use p3_baby_bear::BabyBear;
use p3_blake3::Blake3;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::{ExtensionMmcs, Pcs as PcsTrait};
use p3_dft::Radix2DitParallel;
use p3_field::{ExtensionField, Field, TwoAdicField, extension::BinomialExtensionField};
use p3_fri::{FriParameters as InnerFriParameters, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

pub type Val = BabyBear;
pub type PackedVal = <Val as Field>::Packing;
pub type ExtVal = BinomialExtensionField<Val, 4>;
pub type PackedExtVal = <ExtVal as ExtensionField<Val>>::ExtensionPacking;
pub type Challenger = SerializingChallenger32<Val, HashChallenger<u8, Blake3, 32>>;
pub type Mmcs =
    MerkleTreeMmcs<Val, u8, SerializingHasher<Blake3>, Blake3CompressionFunction, 2, 32>;
pub type ExtMmcs = ExtensionMmcs<Val, ExtVal, Mmcs>;
pub type Pcs = TwoAdicFriPcs<Val, Dft, Mmcs, ExtMmcs>;

pub type Commitment = <Pcs as PcsTrait<ExtVal, Challenger>>::Commitment;
pub type Domain = <Pcs as PcsTrait<ExtVal, Challenger>>::Domain;
pub type ProverData = <Pcs as PcsTrait<ExtVal, Challenger>>::ProverData;
pub type EvaluationsOnDomain<'a> = <Pcs as PcsTrait<ExtVal, Challenger>>::EvaluationsOnDomain<'a>;
pub type PcsError = <Pcs as PcsTrait<ExtVal, Challenger>>::Error;
pub type PcsProof = <Pcs as PcsTrait<ExtVal, Challenger>>::Proof;

/// A BabyBear/Blake3 [`StarkGenericConfig`] implementation.
pub struct BabyBearBlake3Config {
    /// The PCS used to commit polynomials and prove opening proofs.
    pcs: Pcs,
    /// Seed for fresh challengers: a domain-separation tag followed by a
    /// digest of all protocol parameters.
    challenger_seed: Vec<u8>,
    /// Largest log2 degree the PCS can commit to and open.
    max_log_degree: usize,
    /// Largest quotient degree the PCS can serve trace evaluations for
    /// (the FRI blowup factor).
    max_quotient_degree: usize,
}

impl BabyBearBlake3Config {
    pub fn new(commitment_parameters: CommitmentParameters, fri_parameters: FriParameters) -> Self {
        let pcs = new_pcs(commitment_parameters, fri_parameters);
        // Seed the challenger with a protocol tag for domain separation,
        // followed by every protocol parameter. Binding the parameters into
        // the seed means transcripts produced under different parameters
        // never collide (see the transcript contract on
        // [`StarkGenericConfig::initialise_challenger`]).
        let mut challenger_seed = b"multi-stark/v0".to_vec();
        for parameter in [
            commitment_parameters.log_blowup,
            commitment_parameters.cap_height,
            fri_parameters.log_final_poly_len,
            fri_parameters.max_log_arity,
            fri_parameters.num_queries,
            fri_parameters.commit_proof_of_work_bits,
            fri_parameters.query_proof_of_work_bits,
        ] {
            let parameter = u64::try_from(parameter).expect("parameter exceeds u64");
            challenger_seed.extend_from_slice(&parameter.to_le_bytes());
        }
        let max_log_degree = Val::TWO_ADICITY - commitment_parameters.log_blowup;
        let max_quotient_degree = 1 << commitment_parameters.log_blowup;
        Self {
            pcs,
            challenger_seed,
            max_log_degree,
            max_quotient_degree,
        }
    }
}

impl StarkGenericConfig for BabyBearBlake3Config {
    type Pcs = Pcs;
    type Challenge = ExtVal;
    type Challenger = Challenger;

    fn pcs(&self) -> &Pcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Challenger {
        Challenger::from_hasher(self.challenger_seed.clone(), Blake3)
    }

    fn max_log_degree(&self) -> usize {
        self.max_log_degree
    }

    fn max_quotient_degree(&self) -> usize {
        self.max_quotient_degree
    }
}

type Blake3CompressionFunction = CompressionFunctionFromHasher<Blake3, 2, 32>;
type Dft = Radix2DitParallel<Val>;

fn new_mmcs(cap_height: usize) -> Mmcs {
    let byte_hash = Blake3;
    let field_hash = SerializingHasher::new(byte_hash);
    let compress = Blake3CompressionFunction::new(byte_hash);
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
