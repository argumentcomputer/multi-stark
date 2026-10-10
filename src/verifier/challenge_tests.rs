use std::{collections::VecDeque, sync::Arc};

use p3_matrix::dense::RowMajorMatrix;

use super::*;
use crate::{
    ark_adapter::{KzgConfig, KzgPcs, PublicSetup, Radix2Coset, Scalar, Srs},
    batch::{BatchMessage, BatchPreamble, BatchProof, ShardHeader},
    expr::Expr,
    prover::Commitments,
    system::CircuitInputs,
    traits::{OpenedValues, OpeningRounds, VerifyRounds},
};

#[derive(Clone)]
struct ScriptedTranscript(VecDeque<Scalar>);

impl Transcript for ScriptedTranscript {
    type F = Scalar;
    type Challenge = Scalar;
    type Commitment = ();

    fn observe_field(&mut self, _: Scalar) {}
    fn observe_field_slice(&mut self, _: &[Scalar]) {}
    fn observe_challenge(&mut self, _: Scalar) {}
    fn observe_commitment(&mut self, _: ()) {}

    fn sample_challenge(&mut self) -> Scalar {
        self.0.pop_front().expect("scripted challenge")
    }
}

struct ScriptedPcs;

impl Pcs for ScriptedPcs {
    type F = Scalar;
    type Challenge = Scalar;
    type Domain = Radix2Coset;
    type Challenger = ScriptedTranscript;
    type Commitment = ();
    type ProverData = ();
    type Proof = ();
    type Error = ();
    type Evaluations<'a> = RowMajorMatrix<Scalar>;

    fn natural_domain_for_degree(&self, degree: usize) -> Radix2Coset {
        Radix2Coset {
            log_size: degree.ilog2() as usize,
            shift: Scalar::ONE,
        }
    }

    fn max_quotient_degree(&self) -> usize {
        1
    }

    fn commit(&self, _: Vec<(Radix2Coset, RowMajorMatrix<Scalar>)>) -> ((), ()) {
        unreachable!("verifier-only fixture")
    }

    fn commit_quotient(&self, _: Vec<(Radix2Coset, RowMajorMatrix<Scalar>, usize)>) -> ((), ()) {
        unreachable!("verifier-only fixture")
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        _: &'a (),
        _: usize,
        _: Radix2Coset,
    ) -> RowMajorMatrix<Scalar> {
        unreachable!("verifier-only fixture")
    }

    fn open(
        &self,
        _: OpeningRounds<'_, (), Scalar>,
        _: &mut ScriptedTranscript,
    ) -> (OpenedValues<Scalar>, ()) {
        unreachable!("verifier-only fixture")
    }

    fn verify(
        &self,
        _: VerifyRounds<(), Radix2Coset, Scalar>,
        _: &(),
        _: &mut ScriptedTranscript,
    ) -> Result<(), ()> {
        Ok(())
    }
}

struct ScriptedConfig {
    pcs: ScriptedPcs,
    challenges: [Scalar; 4],
}

impl ProofConfig for ScriptedConfig {
    type Pcs = ScriptedPcs;
    type Challenge = Scalar;
    type Challenger = ScriptedTranscript;

    fn pcs(&self) -> &ScriptedPcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> ScriptedTranscript {
        ScriptedTranscript(self.challenges.into())
    }

    fn max_log_degree(&self) -> usize {
        4
    }

    fn max_quotient_degree(&self) -> usize {
        1
    }

    fn log_blowup(&self) -> usize {
        0
    }

    fn omit_unused_next_row_openings(&self) -> bool {
        true
    }
}

fn scripted_proof(beta: Scalar, zeta: Scalar) -> (System<ScriptedConfig>, Proof<ScriptedConfig>) {
    let (system, _) = System::new(
        ScriptedConfig {
            pcs: ScriptedPcs,
            challenges: [beta, Scalar::from_u8(3), Scalar::from_u8(11), zeta],
        },
        [CircuitInputs {
            main_width: 1,
            ..Default::default()
        }],
    );
    let proof = Proof {
        active: vec![true],
        commitments: Commitments {
            stage_1_trace: (),
            stage_2_trace: (),
            quotient_chunks: (),
        },
        intermediate_accumulators: vec![Scalar::ZERO],
        log_degrees: vec![2],
        opening_proof: (),
        quotient_opened_values: vec![vec![vec![Scalar::ZERO]]],
        preprocessed_opened_values: None,
        stage_1_opened_values: vec![vec![vec![Scalar::ZERO]]],
        stage_2_opened_values: vec![vec![vec![Scalar::ZERO]; 2]],
    };
    (system, proof)
}

#[test]
fn zero_zeta_is_rejected_before_opening_verification() {
    let (system, proof) = scripted_proof(Scalar::from_u8(7), Scalar::ZERO);
    assert!(matches!(
        system.verify_multiple_claims(&[], &proof),
        Err(VerificationError::InvalidChallenge)
    ));
}

#[test]
fn every_active_domain_point_is_rejected() {
    for zeta in Scalar::two_adic_generator(2).powers().take(4) {
        let (system, proof) = scripted_proof(Scalar::from_u8(7), zeta);
        assert!(matches!(
            system.verify_multiple_claims(&[], &proof),
            Err(VerificationError::InvalidChallenge)
        ));
    }
    let (system, proof) = scripted_proof(Scalar::from_u8(7), Scalar::TWO);
    system.verify_multiple_claims(&[], &proof).unwrap();
}

#[test]
fn domain_exclusion_checks_every_active_height() {
    let (mut system, mut proof) = scripted_proof(Scalar::from_u8(7), Scalar::two_adic_generator(3));
    let (other, _) = scripted_proof(Scalar::from_u8(7), Scalar::TWO);
    system.circuits.extend(other.circuits);
    system.preprocessed_indices.push(None);
    proof.active.push(false);
    system.verify_multiple_claims(&[], &proof).unwrap();
    proof.active[1] = true;
    proof.log_degrees.push(3);
    proof.intermediate_accumulators.push(Scalar::ZERO);
    proof
        .stage_1_opened_values
        .push(proof.stage_1_opened_values[0].clone());
    proof
        .stage_2_opened_values
        .push(proof.stage_2_opened_values[0].clone());
    proof
        .quotient_opened_values
        .push(proof.quotient_opened_values[0].clone());
    assert!(matches!(
        system.verify_multiple_claims(&[], &proof),
        Err(VerificationError::InvalidChallenge)
    ));
}

#[test]
fn zero_public_claim_denominator_is_rejected() {
    let beta = Scalar::from_u8(7);
    let (system, proof) = scripted_proof(beta, Scalar::TWO);
    assert!(matches!(
        system.verify_multiple_claims(&[&[-beta]], &proof),
        Err(VerificationError::InvalidChallenge)
    ));
}

#[test]
fn zero_batch_claim_and_message_denominators_are_rejected() {
    let beta = Scalar::from_u8(7);
    let (system, proof) = scripted_proof(beta, Scalar::TWO);
    let mut batch = BatchProof {
        preamble: BatchPreamble {
            headers: vec![ShardHeader {
                active: proof.active.clone(),
                stage_1_trace: (),
                log_degrees: proof.log_degrees.clone(),
                claims: vec![vec![-beta]],
            }],
            messages: vec![],
        },
        proofs: vec![proof],
    };
    assert!(matches!(
        system.verify_batch_shard(&batch.preamble, 0, &batch.proofs[0]),
        Err(VerificationError::InvalidChallenge)
    ));
    batch.preamble.headers[0].claims.clear();
    batch.preamble.messages.push(BatchMessage {
        args: vec![-beta],
        multiplicity: Scalar::ONE,
    });
    assert!(matches!(
        system.verify_batch(&batch),
        Err(VerificationError::InvalidChallenge)
    ));
    batch.preamble.messages.clear();
    system.verify_batch(&batch).unwrap();
}

fn high_degree_air_proof(constant: Scalar) -> (System<KzgConfig>, Proof<KzgConfig>) {
    let powers = Srs::unsafe_dev_setup(8, b"high-degree-air");
    let srs = Arc::new(
        Srs::from_public_powers(
            powers.g1,
            powers.g2,
            powers.tau_g2,
            PublicSetup {
                max_degree: 7,
                id: [5; 32],
            },
        )
        .unwrap(),
    );
    let pcs = KzgPcs::new(srs.clone(), 1);
    let (system, _) = System::new(
        KzgConfig::with_max_trace_len(srs, 2, 1),
        [CircuitInputs {
            main_width: 1,
            constraints: vec![Expr::main(0)],
            ..Default::default()
        }],
    );
    assert_eq!(system.circuits[0].constraint_count(), 2);
    let trace_domain = pcs.natural_domain_for_degree(2);
    let polynomial_domain = pcs.natural_domain_for_degree(4);
    let (main, mut main_data) = pcs.commit(vec![(
        polynomial_domain,
        RowMajorMatrix::new_col(
            polynomial_domain
                .points()
                .into_iter()
                .map(|x| x * x + constant)
                .collect(),
        ),
    )]);
    // A malicious committer can use public powers beyond the trace height.
    main_data.matrices[0].domain = trace_domain;
    let (stage2, stage2_data) = pcs.commit(vec![(
        trace_domain,
        RowMajorMatrix::new_col(vec![Scalar::ZERO; 2]),
    )]);

    let mut transcript = system.config.initialise_challenger();
    system.observe_shape(&mut transcript);
    transcript.observe_field(Scalar::ONE);
    transcript.observe_commitment(main.clone());
    transcript.observe_field(Scalar::ONE);
    observe_claims::<KzgConfig>(&mut transcript, &[]);
    sample_lookup_challenges::<KzgConfig>(&mut transcript);
    transcript.observe_commitment(stage2.clone());
    transcript.observe_challenge(Scalar::ZERO);
    let alpha = transcript.sample_challenge();
    let (quotient, quotient_data) = pcs.commit(vec![(
        trace_domain,
        RowMajorMatrix::new_col(vec![alpha; 2]),
    )]);
    transcript.observe_commitment(quotient.clone());
    let zeta = transcript.sample_challenge();
    let (opened, opening_proof) = pcs.open(
        vec![
            (&main_data, vec![vec![zeta]]),
            (
                &stage2_data,
                vec![vec![zeta, trace_domain.next_point(zeta)]],
            ),
            (&quotient_data, vec![vec![zeta]]),
        ],
        &mut transcript,
    );
    let mut opened = opened.into_iter();
    let proof = Proof {
        active: vec![true],
        commitments: Commitments {
            stage_1_trace: main,
            stage_2_trace: stage2,
            quotient_chunks: quotient,
        },
        intermediate_accumulators: vec![Scalar::ZERO],
        log_degrees: vec![1],
        opening_proof,
        stage_1_opened_values: opened.next().unwrap(),
        stage_2_opened_values: opened.next().unwrap(),
        quotient_opened_values: opened.next().unwrap(),
        preprocessed_opened_values: None,
    };
    (system, proof)
}

#[test]
fn public_degree_air_accepts_a_high_degree_domain_vanishing_polynomial() {
    let (system, proof) = high_degree_air_proof(Scalar::NEG_ONE);
    system.verify_multiple_claims(&[], &proof).unwrap();
}

#[test]
fn public_degree_air_rejects_invalid_rows_with_valid_high_degree_openings() {
    let (system, proof) = high_degree_air_proof(Scalar::ZERO);
    assert!(matches!(
        system.verify_multiple_claims(&[], &proof),
        Err(VerificationError::OodEvaluationMismatch)
    ));
}
