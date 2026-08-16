//! The KZG structured reference string: powers of a secret τ in G1,
//! plus the two G2 points verification pairs against.
//!
//! Production use must load a ceremony transcript (e.g. the Ethereum
//! KZG ceremony); [`Srs::unsafe_dev_setup`] generates one from a seed
//! for tests and development — the "secret" is derived from the seed in
//! the clear, so proofs under it carry no security.

use ark_bls12_381::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{CurveGroup, PrimeGroup};
use ark_ff::{Field, PrimeField};

/// `[G, τG, τ²G, …]` in G1 and `[H, τH]` in G2.
pub struct Srs {
    pub g1: Vec<G1Affine>,
    pub g2: G2Affine,
    pub tau_g2: G2Affine,
}

impl Srs {
    /// The largest polynomial length (degree + 1) this SRS can commit.
    #[inline]
    pub fn max_len(&self) -> usize {
        self.g1.len()
    }

    /// A deterministic SRS with τ derived from `seed`. TESTS AND
    /// DEVELOPMENT ONLY: τ is recoverable, so commitments under this
    /// SRS are not binding against anyone who knows the seed.
    pub fn unsafe_dev_setup(max_len: usize, seed: &[u8]) -> Self {
        let mut wide = [0u8; 64];
        blake3::Hasher::new()
            .update(b"multi-stark/kzg/dev-srs")
            .update(seed)
            .finalize_xof()
            .fill(&mut wide);
        let tau = Fr::from_le_bytes_mod_order(&wide);

        let g1_gen = G1Projective::generator();
        let mut acc = Fr::ONE;
        let powers: Vec<G1Projective> = (0..max_len)
            .map(|_| {
                let point = g1_gen * acc;
                acc *= tau;
                point
            })
            .collect();
        let g2_gen = G2Projective::generator();
        Self {
            g1: G1Projective::normalize_batch(&powers),
            g2: g2_gen.into_affine(),
            tau_g2: (g2_gen * tau).into_affine(),
        }
    }
}
