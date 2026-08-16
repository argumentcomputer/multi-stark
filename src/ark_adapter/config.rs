//! [`StarkGenericConfig`] instantiation for the KZG backend: BLS12-381
//! scalar field (its own challenge field, `D = 1`), Blake3 transcript,
//! monomial KZG commitments.

use std::sync::Arc;

use ark_serialize::CanonicalSerialize;

use crate::config::StarkGenericConfig;
use crate::traits::Pcs;

use super::field::Scalar;
use super::pcs::KzgPcs;
use super::srs::Srs;
use super::transcript::Blake3Transcript;

pub struct KzgConfig {
    pcs: KzgPcs,
    /// Bytes observed into every fresh challenger: a domain tag plus a
    /// digest of the protocol parameters INCLUDING the SRS (see the
    /// transcript contract on [`StarkGenericConfig::initialise_challenger`]).
    transcript_seed: Vec<u8>,
    max_log_degree: usize,
}

impl KzgConfig {
    /// Public parameters are caller-supplied and taken on trust here:
    /// call [`Srs::validate`] first on parameters you did not generate.
    ///
    /// # Panics
    /// Panics if the SRS length is not a power of two (trace domains
    /// are, and `max_log_degree` is read off the SRS).
    pub fn new(srs: Arc<Srs>, max_quotient_degree: usize) -> Self {
        assert!(
            srs.max_len().is_power_of_two(),
            "SRS length must be a power of two"
        );
        let max_log_degree = p3_util::log2_strict_usize(srs.max_len());
        let mut transcript_seed = b"multi-stark/kzg/v0".to_vec();
        for parameter in [max_log_degree, max_quotient_degree] {
            transcript_seed.extend(u64::try_from(parameter).unwrap().to_le_bytes());
        }
        // Bind the SRS: τ·G1 and τ·G2 determine it entirely.
        srs.g1[1]
            .serialize_compressed(&mut transcript_seed)
            .expect("serialization into a Vec cannot fail");
        srs.tau_g2
            .serialize_compressed(&mut transcript_seed)
            .expect("serialization into a Vec cannot fail");
        Self {
            pcs: KzgPcs::new(srs, max_quotient_degree),
            transcript_seed,
            max_log_degree,
        }
    }
}

impl StarkGenericConfig for KzgConfig {
    type Pcs = KzgPcs;
    type Challenge = Scalar;
    type Challenger = Blake3Transcript;

    fn pcs(&self) -> &KzgPcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Blake3Transcript {
        let mut challenger = Blake3Transcript::new();
        challenger.observe_bytes(&self.transcript_seed);
        challenger
    }

    fn max_log_degree(&self) -> usize {
        self.max_log_degree
    }

    fn max_quotient_degree(&self) -> usize {
        self.pcs.max_quotient_degree()
    }

    fn log_blowup(&self) -> usize {
        // KZG commits polynomials, not evaluation blowups.
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;
    use crate::lookup::Lookup;
    use crate::prover::Proof;
    use crate::system::{CircuitInputs, System, SystemWitness};
    use crate::traits::{Algebra, Field};
    use p3_matrix::dense::RowMajorMatrix;

    /// `a·b = c` per row, with a self-canceling push/pull lookup pair to
    /// exercise the stage-2 machinery — the KZG twin of the BabyBear
    /// config's smoke test.
    fn mul_circuit() -> CircuitInputs<Scalar> {
        let m = Expr::main;
        CircuitInputs {
            main_width: 3,
            constraints: vec![m(0) * m(1) - m(2)],
            lookups: vec![
                Lookup::push(Expr::constant(Scalar::from_u32(1)), vec![m(0), m(2)]),
                Lookup::pull(Expr::constant(Scalar::from_u32(1)), vec![m(0), m(2)]),
            ],
            ..Default::default()
        }
    }

    fn kzg_system() -> (System<KzgConfig>, crate::system::ProverKey<KzgConfig>) {
        let srs = Arc::new(Srs::unsafe_dev_setup(1 << 8, b"test"));
        let config = KzgConfig::new(srs, 8);
        System::new(config, [mul_circuit()])
    }

    fn witness(system: &System<KzgConfig>) -> SystemWitness<Scalar> {
        let f = Scalar::from_u32;
        let trace = RowMajorMatrix::new([2, 3, 6, 4, 5, 20, 7, 8, 56, 1, 1, 1].map(f).to_vec(), 3);
        SystemWitness::from_stage_1(vec![trace], system)
    }

    #[test]
    fn kzg_prove_verify() {
        let (system, key) = kzg_system();
        let no_claims: &[&[Scalar]] = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness(&system));
        system
            .verify_multiple_claims(no_claims, &proof)
            .expect("KZG proof failed to verify");
    }

    #[test]
    fn kzg_tampering_rejected() {
        let (system, key) = kzg_system();
        let no_claims: &[&[Scalar]] = &[];
        let prove = || system.prove_multiple_claims(&key, no_claims, witness(&system));

        let mut tampered = prove();
        tampered.intermediate_accumulators[0] += <Scalar as Algebra<Scalar>>::ONE;
        assert!(system.verify_multiple_claims(no_claims, &tampered).is_err());

        let mut tampered = prove();
        tampered.stage_1_opened_values[0][0][0] += <Scalar as Algebra<Scalar>>::ONE;
        assert!(system.verify_multiple_claims(no_claims, &tampered).is_err());

        let mut tampered = prove();
        tampered.quotient_opened_values[0][0][0] += <Scalar as Algebra<Scalar>>::ONE;
        assert!(system.verify_multiple_claims(no_claims, &tampered).is_err());
    }

    #[test]
    fn kzg_wrong_claim_rejected() {
        let (system, key) = kzg_system();
        let no_claims: &[&[Scalar]] = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness(&system));
        let claim = [Scalar::from_u32(42)];
        assert!(system.verify(&claim, &proof).is_err());
    }

    #[test]
    fn kzg_serialization_round_trip() {
        let (system, key) = kzg_system();
        let no_claims: &[&[Scalar]] = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness(&system));
        let bytes = proof.to_bytes().expect("serialize");
        let proof2 = Proof::<KzgConfig>::from_bytes(&bytes).expect("deserialize");
        system.verify_multiple_claims(no_claims, &proof2).unwrap();
    }

    /// Two circuits at different trace heights: ζ·g differs per height,
    /// so the opening carries three distinct points and the per-point
    /// witness batching (and cross-point pairing batch) is exercised.
    #[test]
    fn kzg_two_circuits_two_heights() {
        let srs = Arc::new(Srs::unsafe_dev_setup(1 << 8, b"test"));
        let config = KzgConfig::new(srs, 8);
        let (system, key) = System::new(config, [mul_circuit(), mul_circuit()]);
        let f = Scalar::from_u32;
        let small = RowMajorMatrix::new([2, 3, 6, 4, 5, 20, 7, 8, 56, 1, 1, 1].map(f).to_vec(), 3);
        let mut long = small.values.clone();
        for _ in 0..2 {
            long.extend(long.clone());
        }
        let witness =
            SystemWitness::from_stage_1(vec![small, RowMajorMatrix::new(long, 3)], &system);
        let no_claims: &[&[Scalar]] = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness);
        let bytes = proof.to_bytes().expect("serialize");
        println!(
            "KZG proof: {} bytes (two circuits, heights 4 and 16)",
            bytes.len()
        );
        system.verify_multiple_claims(no_claims, &proof).unwrap();
    }
}
