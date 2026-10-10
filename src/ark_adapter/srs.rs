//! KZG parameters: G1 powers, G2 anchors and G2 powers for degree checks.
//! Imported parameters need validated point decoding, [`Srs::validate`] and
//! trusted provenance. Consistency does not establish an unknown trapdoor.
//! Public-setup metadata records the full available G1 degree range even
//! when only an honest prover's prefix is loaded. Truncation is not a degree proof.
//! [`Srs::unsafe_dev_setup`] reveals its trapdoor and is only for tests.

use ark_bls12_381::{Bls12_381, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{
    AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM, pairing::Pairing,
    scalar_mul::BatchMulPreprocessing,
};
use ark_ff::{Field, PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use p3_maybe_rayon::prelude::*;

mod cache;
pub mod filecoin;

/// Authenticated public parameters and their complete polynomial degree allowance.
/// The identity names the ceremony, independently of the locally loaded prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicSetup {
    pub max_degree: usize,
    pub id: [u8; 32],
}

fn dev_tau(seed: &[u8]) -> Fr {
    let mut wide = [0u8; 64];
    blake3::Hasher::new()
        .update(b"multi-stark/kzg/dev-srs")
        .update(seed)
        .finalize_xof()
        .fill(&mut wide);
    Fr::from_le_bytes_mod_order(&wide)
}

/// `[G, τG, τ²G, …]` in G1 and `[H, τH]` in G2.
pub struct Srs {
    pub g1: Vec<G1Affine>,
    pub g2: G2Affine,
    pub tau_g2: G2Affine,
    /// Entry k is τ^(max_len - 2^k) H, for degree-bound checks.
    pub degree_keys: Vec<G2Affine>,
    pub(crate) public_setup: Option<PublicSetup>,
}

impl Srs {
    /// Number of locally loaded G1 powers, not the adversary's degree allowance.
    #[inline]
    pub fn max_len(&self) -> usize {
        self.g1.len()
    }

    pub fn public_setup(&self) -> Option<PublicSetup> {
        self.public_setup
    }

    /// Whether a trace needs the legacy shifted degree commitment.
    pub fn requires_shifted_commitment(&self, height: usize) -> bool {
        self.public_setup.is_none() && height < self.max_len()
    }

    /// Construct parameters for globally bounded polynomial identities.
    ///
    /// The caller authenticates the ceremony identity and full public degree
    /// range. Point and progression checks establish consistency, not provenance
    /// or secrecy of the trapdoor. A two-point prefix suffices for verification.
    pub fn from_public_powers(
        g1: Vec<G1Affine>,
        g2: G2Affine,
        tau_g2: G2Affine,
        setup: PublicSetup,
    ) -> Result<Self, &'static str> {
        let srs = Self {
            g1,
            g2,
            tau_g2,
            degree_keys: vec![],
            public_setup: Some(setup),
        };
        if srs
            .g1
            .par_iter()
            .any(|p| !p.is_on_curve() || !p.is_in_correct_subgroup_assuming_on_curve())
            || [&srs.g2, &srs.tau_g2]
                .into_iter()
                .any(|p| !p.is_on_curve() || !p.is_in_correct_subgroup_assuming_on_curve())
        {
            return Err("invalid public parameter point");
        }
        srs.validate()?;
        Ok(srs)
    }

    /// Consistency check for user-supplied parameters: the G1 powers
    /// must form one geometric progression in the secret the G2 pair
    /// encodes — `e(g1[i+1], H) = e(g1[i], τH)` for every `i` — and the
    /// anchors must not be the identity. Batched into two MSMs and one
    /// 2-pairing product with a random combiner derived from the SRS
    /// bytes themselves (whoever fixed the SRS could not predict it).
    ///
    /// Curve and subgroup membership are not checked here; callers must
    /// establish both before checking the progression.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.g1.len() < 2 || !self.g1.len().is_power_of_two() {
            return Err("SRS length must be a power of two >= 2");
        }
        if self.g1[0].is_zero()
            || self.g1[1].is_zero()
            || self.g2.is_zero()
            || self.tau_g2.is_zero()
        {
            return Err("SRS anchor is the identity");
        }
        if let Some(setup) = self.public_setup {
            if setup.max_degree < self.max_len() - 1 || !self.degree_keys.is_empty() {
                return Err("inconsistent public degree policy");
            }
        } else {
            let logs = p3_util::log2_strict_usize(self.max_len());
            if self.degree_keys.len() != logs + 1 {
                return Err("missing degree keys");
            }
        }
        for (k, key) in self.degree_keys.iter().enumerate() {
            let shift = self.max_len() - (1 << k);
            if !Bls12_381::multi_pairing(
                [
                    self.g1[shift],
                    (-G1Projective::from(self.g1[0])).into_affine(),
                ],
                [self.g2, *key],
            )
            .is_zero()
            {
                return Err("inconsistent degree key");
            }
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"multi-stark/kzg/srs-validate/v2");
        hasher.update(&(self.g1.len() as u64).to_le_bytes());
        let mut bytes = Vec::with_capacity(192);
        for point in &self.g1 {
            bytes.clear();
            point.serialize_compressed(&mut bytes).expect("Vec write");
            hasher.update(&bytes);
        }
        bytes.clear();
        self.g2.serialize_compressed(&mut bytes).expect("Vec write");
        self.tau_g2
            .serialize_compressed(&mut bytes)
            .expect("Vec write");
        hasher.update(&bytes);
        let mut wide = [0u8; 64];
        hasher.finalize_xof().fill(&mut wide);
        let r = Fr::from_le_bytes_mod_order(&wide);
        if r.is_zero() {
            return Err("zero parameter-validation challenge");
        }
        const BLOCK_POINTS: usize = 1 << 16;
        let mut r_powers = Vec::with_capacity(BLOCK_POINTS);
        let mut acc = Fr::ONE;
        let mut low = G1Projective::zero();
        let mut high = G1Projective::zero();
        for offset in (0..self.g1.len() - 1).step_by(BLOCK_POINTS) {
            let count = BLOCK_POINTS.min(self.g1.len() - 1 - offset);
            r_powers.clear();
            for _ in 0..count {
                r_powers.push(acc);
                acc *= r;
            }
            low += G1Projective::msm(&self.g1[offset..offset + count], &r_powers)
                .expect("equal lengths");
            high += G1Projective::msm(&self.g1[offset + 1..offset + count + 1], &r_powers)
                .expect("equal lengths");
        }
        // e(high, H) = e(low, τH)  ⇔  e(high, H)·e(−low, τH) = 1.
        let check = Bls12_381::multi_pairing(
            [high.into_affine(), (-low).into_affine()],
            [self.g2, self.tau_g2],
        );
        if check.is_zero() {
            Ok(())
        } else {
            Err("G1 powers are not one τ-progression against the G2 pair")
        }
    }

    /// A deterministic SRS with τ derived from `seed`. TESTS AND
    /// DEVELOPMENT ONLY: τ is recoverable, so commitments under this
    /// SRS are not binding against anyone who knows the seed.
    pub fn unsafe_dev_setup(max_len: usize, seed: &[u8]) -> Self {
        assert!(max_len >= 2 && max_len.is_power_of_two());
        let tau = dev_tau(seed);

        let g1_gen = G1Projective::generator();
        // τ-powers in independent chunks (each chunk seeds itself with
        // τ^start), so the scalar multiplications parallelize under the
        // `parallel` feature; serial otherwise.
        let chunk = 1usize << 14;
        // A bounded fixed-base table amortizes generator multiplication across
        // powers without a table whose size grows with the SRS degree.
        let table = BatchMulPreprocessing::new(g1_gen, max_len.min(1 << 12));
        let mut g1 = vec![G1Affine::identity(); max_len];
        g1.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(index, output)| {
                let mut acc = tau.pow([(index * chunk) as u64]);
                let scalars: Vec<_> = (0..output.len())
                    .map(|_| {
                        let value = acc;
                        acc *= tau;
                        value
                    })
                    .collect();
                output.copy_from_slice(&table.batch_mul(&scalars));
            });
        let g2_gen = G2Projective::generator();
        Self {
            g1,
            g2: g2_gen.into_affine(),
            tau_g2: (g2_gen * tau).into_affine(),
            degree_keys: (0..=p3_util::log2_strict_usize(max_len))
                .map(|k| (g2_gen * tau.pow([(max_len - (1 << k)) as u64])).into_affine())
                .collect(),
            public_setup: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_base_setup_matches_individual_multiplication_across_tiles() {
        let seed = b"fixed-base-tile-boundaries";
        let srs = Srs::unsafe_dev_setup(1 << 15, seed);
        let tau = dev_tau(seed);
        for i in [0usize, 1, 31, (1 << 14) - 1, 1 << 14, (1 << 15) - 1] {
            assert_eq!(
                srs.g1[i],
                (G1Projective::generator() * tau.pow([i as u64])).into_affine()
            );
        }
    }

    #[test]
    fn dev_setup_validates() {
        Srs::unsafe_dev_setup(1 << 5, b"validate-test")
            .validate()
            .unwrap();
    }

    #[test]
    fn corrupted_power_rejected() {
        let mut srs = Srs::unsafe_dev_setup(1 << 5, b"validate-test");
        srs.g1[7] = (G1Projective::from(srs.g1[7]) + G1Projective::generator()).into_affine();
        assert!(srs.validate().is_err());
    }

    #[test]
    fn mismatched_tau_g2_rejected() {
        let mut srs = Srs::unsafe_dev_setup(1 << 5, b"validate-test");
        srs.tau_g2 = (G2Projective::from(srs.tau_g2) + G2Projective::generator()).into_affine();
        assert!(srs.validate().is_err());
    }
}
