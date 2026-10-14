use crate::types::{ExtVal, Val};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use std::collections::VecDeque;
type Domain = TwoAdicMultiplicativeCoset<Val>;

use p3_matrix::dense::RowMajorMatrix;

use super::*;
use crate::{
    batch::{BatchMessage, BatchPreamble, BatchProof, ShardHeader},
    prover::Commitments,
    system::CircuitInputs,
    traits::{OpenedValues, OpeningRounds, VerifyRounds},
};

#[derive(Clone)]
struct ScriptedTranscript(VecDeque<ExtVal>);

impl Transcript for ScriptedTranscript {
    type F = Val;
    type Challenge = ExtVal;
    type Commitment = ();

    fn observe_field(&mut self, _: Val) {}
    fn observe_field_slice(&mut self, _: &[Val]) {}
    fn observe_challenge(&mut self, _: ExtVal) {}
    fn observe_commitment(&mut self, _: ()) {}

    fn sample_challenge(&mut self) -> ExtVal {
        self.0.pop_front().expect("scripted challenge")
    }
}

struct ScriptedPcs;

impl Pcs for ScriptedPcs {
    type F = Val;
    type Challenge = ExtVal;
    type Domain = Domain;
    type Challenger = ScriptedTranscript;
    type Commitment = ();
    type ProverData = ();
    type Proof = ();
    type Error = ();
    type Evaluations<'a> = RowMajorMatrix<Val>;

    fn natural_domain_for_degree(&self, degree: usize) -> Domain {
        Domain::new(Val::ONE, degree.ilog2() as usize).unwrap()
    }

    fn max_quotient_degree(&self) -> usize {
        1
    }

    fn commit(&self, _: Vec<(Domain, RowMajorMatrix<Val>)>) -> ((), ()) {
        unreachable!("verifier-only fixture")
    }

    fn commit_quotient(&self, _: Vec<(Domain, RowMajorMatrix<Val>, usize)>) -> ((), ()) {
        unreachable!("verifier-only fixture")
    }

    fn get_evaluations_on_domain(&self, _: &(), _: usize, _: Domain) -> RowMajorMatrix<Val> {
        unreachable!("verifier-only fixture")
    }

    fn open(
        &self,
        _: OpeningRounds<'_, (), ExtVal>,
        _: &mut ScriptedTranscript,
    ) -> (OpenedValues<ExtVal>, ()) {
        unreachable!("verifier-only fixture")
    }

    fn verify(
        &self,
        _: VerifyRounds<(), Domain, ExtVal>,
        _: &(),
        _: &mut ScriptedTranscript,
    ) -> Result<(), ()> {
        Ok(())
    }
}

struct ScriptedConfig {
    pcs: ScriptedPcs,
    challenges: [ExtVal; 4],
}

impl ProofConfig for ScriptedConfig {
    type Pcs = ScriptedPcs;
    type Challenge = ExtVal;
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

fn scripted_proof(beta: Val, zeta: Val) -> (System<ScriptedConfig>, Proof<ScriptedConfig>) {
    let (system, _) = System::new(
        ScriptedConfig {
            pcs: ScriptedPcs,
            challenges: [beta, Val::from_u8(3), Val::from_u8(11), zeta].map(ExtVal::from),
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
        intermediate_accumulators: vec![<ExtVal as Algebra<ExtVal>>::ZERO],
        log_degrees: vec![2],
        opening_proof: (),
        quotient_opened_values: vec![vec![vec![<ExtVal as Algebra<ExtVal>>::ZERO; 2]]],
        preprocessed_opened_values: None,
        stage_1_opened_values: vec![vec![vec![<ExtVal as Algebra<ExtVal>>::ZERO]]],
        stage_2_opened_values: vec![vec![vec![<ExtVal as Algebra<ExtVal>>::ZERO; 2]; 2]],
    };
    (system, proof)
}

#[test]
fn zero_zeta_is_rejected_before_opening_verification() {
    let (system, proof) = scripted_proof(Val::from_u8(7), Val::ZERO);
    assert!(matches!(
        system.verify_multiple_claims(&[], &proof),
        Err(VerificationError::InvalidChallenge)
    ));
}

#[test]
fn every_active_domain_point_is_rejected() {
    for zeta in Val::two_adic_generator(2).powers().take(4) {
        let (system, proof) = scripted_proof(Val::from_u8(7), zeta);
        assert!(matches!(
            system.verify_multiple_claims(&[], &proof),
            Err(VerificationError::InvalidChallenge)
        ));
    }
    let (system, proof) = scripted_proof(Val::from_u8(7), Val::TWO);
    system.verify_multiple_claims(&[], &proof).unwrap();
}

#[test]
fn domain_exclusion_checks_every_active_height() {
    let (mut system, mut proof) = scripted_proof(Val::from_u8(7), Val::two_adic_generator(3));
    let (other, _) = scripted_proof(Val::from_u8(7), Val::TWO);
    system.circuits.extend(other.circuits);
    system.preprocessed_indices.push(None);
    proof.active.push(false);
    system.verify_multiple_claims(&[], &proof).unwrap();
    proof.active[1] = true;
    proof.log_degrees.push(3);
    proof
        .intermediate_accumulators
        .push(<ExtVal as Algebra<ExtVal>>::ZERO);
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
    let beta = Val::from_u8(7);
    let (system, proof) = scripted_proof(beta, Val::TWO);
    assert!(matches!(
        system.verify_multiple_claims(&[&[-beta]], &proof),
        Err(VerificationError::InvalidChallenge)
    ));
}

#[test]
fn zero_batch_claim_and_message_denominators_are_rejected() {
    let beta = Val::from_u8(7);
    let (system, proof) = scripted_proof(beta, Val::TWO);
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
        multiplicity: Val::ONE,
    });
    assert!(matches!(
        system.verify_batch(&batch),
        Err(VerificationError::InvalidChallenge)
    ));
    batch.preamble.messages.clear();
    system.verify_batch(&batch).unwrap();
}
