//! Core type aliases and configuration for the KZG-based prover/verifier.
//!
//! The PCS used is [`SonicKZG10`] over BLS12-381, which gives:
//!   * Single-element (1×G1) batched openings at one point for any number of
//!     polynomials, via random-LC at the prover side.
//!   * Verifier cost = a constant number of pairings (independent of how many
//!     polynomials are batched).
//!
//! Polynomials are stored in **monomial basis** (`DensePolynomial<Fr>`).

use ark_bls12_381::{Bls12_381, Fr};
use ark_crypto_primitives::sponge::{
    CryptographicSponge,
    poseidon::{PoseidonConfig, PoseidonSponge},
};
use ark_ff::UniformRand;
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::{
    PolynomialCommitment, kzg10,
    sonic_pc::{self, CommitterKey, SonicKZG10, UniversalParams, VerifierKey},
};
use ark_std::rand::RngCore;

pub type Val = Fr;
pub type UniPoly = DensePolynomial<Fr>;
pub type PC = SonicKZG10<Bls12_381, UniPoly>;
pub type Sponge = PoseidonSponge<Fr>;

pub type SrsParams = UniversalParams<Bls12_381>;
pub type CkParams = CommitterKey<Bls12_381>;
pub type VkParams = VerifierKey<Bls12_381>;
pub type Commitment = sonic_pc::Commitment<Bls12_381>;
pub type CommitmentState = kzg10::Randomness<Fr, UniPoly>;
pub type OpeningProof = kzg10::Proof<Bls12_381>;

/// Configuration shared by prover and verifier.
#[derive(Clone)]
pub struct PlonkConfig {
    pub committer_key: CkParams,
    pub verifier_key: VkParams,
    pub sponge_config: PoseidonConfig<Fr>,
}

impl PlonkConfig {
    /// One-time KZG ceremony + Poseidon parameter generation.
    ///
    /// `max_degree` should be at least `(max_constraint_degree - 1) * largest_trace_size`
    /// so the quotient polynomial fits.
    pub fn setup<R: RngCore>(max_degree: usize, rng: &mut R) -> Self {
        let pp = PC::setup(max_degree, None, rng).expect("KZG setup failed");
        let (committer_key, verifier_key) =
            PC::trim(&pp, max_degree, 0, None).expect("KZG trim failed");
        let sponge_config = test_poseidon_config(rng);
        Self {
            committer_key,
            verifier_key,
            sponge_config,
        }
    }

    pub fn make_sponge(&self) -> Sponge {
        PoseidonSponge::new(&self.sponge_config)
    }
}

/// WARNING: insecure parameters intended for examples and tests only. Do not
/// use these in production. Mirrors the parameter set from `kzg_sonic.rs`.
fn test_poseidon_config<R: RngCore>(rng: &mut R) -> PoseidonConfig<Fr> {
    let full_rounds = 8;
    let partial_rounds = 31;
    let alpha = 17;
    let mds = vec![
        vec![Fr::from(1u64), Fr::from(0u64), Fr::from(1u64)],
        vec![Fr::from(1u64), Fr::from(1u64), Fr::from(0u64)],
        vec![Fr::from(0u64), Fr::from(1u64), Fr::from(1u64)],
    ];
    let ark: Vec<Vec<Fr>> = (0..full_rounds + partial_rounds)
        .map(|_| (0..3).map(|_| Fr::rand(rng)).collect())
        .collect();
    PoseidonConfig::new(full_rounds, partial_rounds, alpha, mds, ark, 2, 1)
}
