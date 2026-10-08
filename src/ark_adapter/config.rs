//! [`ProofConfig`] instantiation for the KZG backend: BLS12-381
//! scalar field (its own challenge field, `D = 1`), Blake3 transcript,
//! monomial KZG commitments.

use std::sync::Arc;

use ark_serialize::CanonicalSerialize;

use crate::config::ProofConfig;
use crate::traits::Pcs;

use super::field::Scalar;
use super::pcs::KzgPcs;
use super::srs::Srs;
use super::transcript::Blake3Transcript;

#[derive(Clone)]
pub struct KzgConfig {
    pcs: KzgPcs,
    /// Bytes observed into every fresh challenger: a domain tag plus a
    /// digest of the protocol parameters INCLUDING the SRS (see the
    /// transcript contract on [`ProofConfig::initialise_challenger`]).
    transcript_seed: Vec<u8>,
    max_log_degree: usize,
    stream_lookups: bool,
    stream_quotient: bool,
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
            srs.max_len() >= 2 && srs.max_len().is_power_of_two(),
            "SRS length must be a power of two"
        );
        let max_log_degree = p3_util::log2_strict_usize(srs.max_len());
        let mut transcript_seed = b"multi-stark/kzg/v3".to_vec();
        for parameter in [max_log_degree, max_quotient_degree] {
            transcript_seed.extend(u64::try_from(parameter).unwrap().to_le_bytes());
        }
        // Bind the anchors and their τ multiples; validated powers follow.
        srs.g1[0]
            .serialize_compressed(&mut transcript_seed)
            .expect("serialization into a Vec cannot fail");
        srs.g2
            .serialize_compressed(&mut transcript_seed)
            .expect("serialization into a Vec cannot fail");
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
            stream_lookups: false,
            stream_quotient: false,
        }
    }

    /// Reconstruct and commit one lookup trace at a time from committed columns.
    /// This changes memory use, not the proof or its transcript.
    pub fn with_streaming_lookups(mut self) -> Self {
        self.stream_lookups = true;
        self
    }

    /// Evaluate each quotient on trace-sized cosets, bounding temporary matrices.
    pub fn with_streaming_quotient(mut self) -> Self {
        self.stream_quotient = true;
        self
    }
}

impl ProofConfig for KzgConfig {
    fn accelerated_lookup_commit(
        &self,
        inputs: &[crate::config::LookupCommitInput<'_, Self>],
        beta: Scalar,
        gamma: Scalar,
        mut acc: Scalar,
    ) -> Option<crate::config::AcceleratedLookupCommitment<Self>> {
        if !self.stream_lookups {
            return None;
        }
        let mut parts = Vec::new();
        let mut accumulators = Vec::new();
        for (i, input) in inputs.iter().enumerate() {
            let start = std::time::Instant::now();
            let domain = input.stage_1.0.matrices[input.stage_1.1].domain;
            let fixed = input
                .preprocessed
                .map(|(data, slot)| self.pcs.get_evaluations_on_domain(data, slot, domain));
            let main = self
                .pcs
                .get_evaluations_on_domain(input.stage_1.0, input.stage_1.1, domain);
            use crate::traits::{Algebra, EvaluationDomain};
            let mut trace = Vec::with_capacity(domain.size() * input.circuit.stage_2_width);
            let mut local = Scalar::ZERO;
            for start in (0..domain.size()).step_by(1 << 16) {
                let end = (start + (1 << 16)).min(domain.size());
                let values = crate::system::compute_lookup_values_range(
                    input.circuit,
                    &main,
                    fixed.as_ref(),
                    start..end,
                );
                let (mut chunks, next) = crate::lookup::LookupValues::stage_2_traces(
                    &[values],
                    &[input.circuit.lookup_group_size],
                    beta,
                    &gamma,
                    Scalar::ZERO,
                );
                trace.extend(chunks.remove(0).values.into_iter().map(|v| v + local));
                local += next[0];
            }
            drop(main);
            drop(fixed);
            acc += local;
            accumulators.push(acc);
            let trace = p3_matrix::dense::RowMajorMatrix::new(trace, input.circuit.stage_2_width);
            let (_, data) = self.pcs.commit(vec![(domain, trace)]);
            parts.push(data);
            tracing::info!(
                circuit = i,
                seconds = start.elapsed().as_secs_f64(),
                "KZG lookup committed"
            );
        }
        let (commitment, data) = super::pcs::KzgProverData::concatenate(parts);
        Some((commitment, data, accumulators))
    }

    fn accelerated_quotient_commit(
        &self,
        inputs: &[crate::config::QuotientCommitInput<'_, Self>],
        alpha: Scalar,
    ) -> Option<(super::pcs::KzgCommitment, super::pcs::KzgProverData)> {
        if !self.stream_quotient {
            return None;
        }
        use crate::traits::{Algebra, EvaluationDomain, TwoAdicField};
        let mut parts = Vec::with_capacity(inputs.len());
        for (i, input) in inputs.iter().enumerate() {
            let start = std::time::Instant::now();
            let n = input.trace_domain.size();
            let ratio = input.quotient_domain.size() / n;
            let generator = Scalar::two_adic_generator(input.quotient_domain.log_size);
            let mut shift = input.quotient_domain.shift;
            let mut values = vec![Scalar::ZERO; input.quotient_domain.size()];
            for coset in 0..ratio {
                let domain = super::domain::Radix2Coset {
                    log_size: input.trace_domain.log_size,
                    shift,
                };
                let fixed = input
                    .preprocessed
                    .map(|(data, slot)| self.pcs.get_evaluations_on_domain(data, slot, domain));
                let main =
                    self.pcs
                        .get_evaluations_on_domain(input.stage_1.0, input.stage_1.1, domain);
                let stage2 =
                    self.pcs
                        .get_evaluations_on_domain(input.stage_2.0, input.stage_2.1, domain);
                let chunk = crate::prover::quotient_values::<Self>(
                    input.circuit,
                    &input.lookup_publics,
                    input.trace_domain,
                    domain,
                    &fixed,
                    &main,
                    &stage2,
                    alpha,
                    input.constraint_count,
                );
                for (row, value) in chunk.into_iter().enumerate() {
                    values[row * ratio + coset] = value;
                }
                shift *= generator;
                tracing::info!(
                    circuit = i,
                    coset,
                    seconds = start.elapsed().as_secs_f64(),
                    "KZG quotient coset evaluated"
                );
            }
            let (_, data) = self.pcs.commit_quotient(vec![(
                input.quotient_domain,
                p3_matrix::dense::RowMajorMatrix::new_col(values),
                ratio,
            )]);
            parts.push(data);
            tracing::info!(
                circuit = i,
                seconds = start.elapsed().as_secs_f64(),
                "KZG quotient committed"
            );
        }
        Some(super::pcs::KzgProverData::concatenate(parts))
    }

    fn omit_inactive_preprocessed_openings(&self) -> bool {
        true
    }
    fn omit_unused_next_row_openings(&self) -> bool {
        true
    }

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

    fn max_log_quotient_domain(&self) -> usize {
        <Scalar as crate::traits::TwoAdicField>::TWO_ADICITY
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
    fn opens_next_rows_only_when_the_graph_reads_them() {
        let f = Scalar::from_u8;
        let (system, key) = System::new(
            KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(8, b"next-rows")), 4),
            [
                CircuitInputs {
                    main_width: 1,
                    preprocessed: Some(RowMajorMatrix::new_col(vec![f(7); 4])),
                    constraints: vec![Expr::main_next(0) - Expr::preprocessed_next(0)],
                    ..Default::default()
                },
                CircuitInputs {
                    main_width: 1,
                    preprocessed: Some(RowMajorMatrix::new_col(vec![f(9); 2])),
                    constraints: vec![Expr::main(0) - Expr::preprocessed(0)],
                    ..Default::default()
                },
            ],
        );
        let proof = system.prove_multiple_claims(
            &key,
            &[],
            SystemWitness::from_stage_1(
                vec![
                    RowMajorMatrix::new_col(vec![f(7); 4]),
                    RowMajorMatrix::new_col(vec![f(9); 2]),
                ],
                &system,
            ),
        );
        system.verify_multiple_claims(&[], &proof).unwrap();
        assert_eq!(
            proof
                .stage_1_opened_values
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            [2, 1]
        );
        assert_eq!(
            proof
                .preprocessed_opened_values
                .as_ref()
                .unwrap()
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            [2, 1]
        );
        let codec =
            super::super::compact::FixedProofCodec::new(&system, &proof.log_degrees).unwrap();
        let decoded = codec.decode(&codec.encode(&proof).unwrap()).unwrap();
        system.verify_multiple_claims(&[], &decoded).unwrap();
        let mut bad = proof.clone();
        bad.stage_1_opened_values[0].pop();
        assert!(system.verify_multiple_claims(&[], &bad).is_err());
        let mut bad = proof.clone();
        bad.preprocessed_opened_values.as_mut().unwrap()[0][1][0] += Scalar::ONE;
        assert!(system.verify_multiple_claims(&[], &bad).is_err());
        let mut bad = proof.clone();
        bad.stage_1_opened_values[1].push(vec![Scalar::ZERO]);
        assert!(system.verify_multiple_claims(&[], &bad).is_err());
        // KZG commits columns separately, so inactive fixed matrices need no openings.
        let partial = system.prove_multiple_claims(
            &key,
            &[],
            SystemWitness::from_stage_1(
                vec![
                    RowMajorMatrix::new(vec![], 1),
                    RowMajorMatrix::new_col(vec![f(9); 2]),
                ],
                &system,
            ),
        );
        system.verify_multiple_claims(&[], &partial).unwrap();
        assert_eq!(partial.active, [false, true]);
        assert_eq!(
            partial.preprocessed_opened_values.as_ref().unwrap()[0].len(),
            0
        );
        let mut bad = partial.clone();
        bad.preprocessed_opened_values.as_mut().unwrap()[1].clear();
        assert!(system.verify_multiple_claims(&[], &bad).is_err());
        let mut bad = partial;
        bad.preprocessed_opened_values.as_mut().unwrap()[0].push(vec![f(7)]);
        assert!(system.verify_multiple_claims(&[], &bad).is_err());
    }

    #[test]
    fn quotient_slices_fit_an_srs_exactly_as_large_as_the_trace() {
        let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(4, b"exact-srs")), 4);
        let (system, key) = System::new(config, [mul_circuit()]);
        let proof = system.prove_multiple_claims(&key, &[], witness(&system));
        system.verify_multiple_claims(&[], &proof).unwrap();
    }

    #[test]
    fn batch_messages_work_with_scalar_challenges() {
        use crate::batch::{BatchMessage, ShardInput};
        let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(2, b"batch")), 4);
        let (system, key) = System::new(
            config,
            [CircuitInputs {
                main_width: 1,
                preprocessed: Some(RowMajorMatrix::new_col(vec![Scalar::ONE, Scalar::ZERO])),
                lookups: vec![Lookup::pull(Expr::preprocessed(0), vec![Expr::main(0)])],
                ..Default::default()
            }],
        );
        let value = Scalar::from_u8(9);
        let witness =
            SystemWitness::from_stage_1(vec![RowMajorMatrix::new_col(vec![value; 2])], &system);
        let mut proof = system.prove_batch(
            &key,
            vec![ShardInput {
                claims: vec![],
                witness,
            }],
            vec![BatchMessage {
                args: vec![value],
                multiplicity: Scalar::ONE,
            }],
        );
        system.verify_batch(&proof).unwrap();
        proof.preamble.messages[0].args[0] += Scalar::ONE;
        assert!(system.verify_batch(&proof).is_err());
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
