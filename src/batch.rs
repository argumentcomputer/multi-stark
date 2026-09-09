//! Batch proving: several proofs of one system that share the lookup
//! challenges, so that lookup messages may be pushed in one proof and pulled
//! in another.
//!
//! A single [`Proof`](crate::prover::Proof) establishes a lookup identity over
//! its own rows: the pushed and pulled multisets cancel, so the final
//! accumulator is zero. Splitting the rows of one execution across K proofs —
//! trace shards — breaks that per-proof identity: a message pushed in shard 0
//! may be pulled in shard 3. The identity that must hold is the one over all
//! shards together, and it holds only if every shard evaluates its messages
//! under the same challenges β, γ.
//!
//! # Protocol
//!
//! 1. **Round one.** Every shard commits its stage-1 traces
//!    ([`System::prove_stage_1`]) and publishes a [`ShardHeader`]: activation
//!    bitmap, stage-1 commitment, trace heights and claims.
//! 2. **Preamble.** The headers, in shard order, together with the batch's
//!    public [`BatchMessage`]s form the [`BatchPreamble`]. Everything that
//!    contributes a lookup message anywhere in the batch is in it: the
//!    committed traces (through the commitments), the claims, and the
//!    verifier-side messages.
//! 3. **Batch transcript.** One challenger, identical for every shard, is
//!    seeded with the parameters, observes the system shape and the whole
//!    preamble, and samples β, γ ([`System::batch_challenger`]). Sampling
//!    after the preamble is what makes the challenges independent of every
//!    message, exactly as a single proof samples them after its own
//!    commitment and claims.
//! 4. **Round two.** Each shard forks the batch transcript, observes its
//!    shard index, and continues as a single proof does from the stage-2
//!    commitment on ([`System::prove_after_challenges`]). Its final
//!    accumulator is not zero: it is the shard's *residual*, the part of the
//!    global sum contributed by its rows and claims.
//!
//! The batch verifier ([`System::verify_batch`]) checks each header against
//! its proof, verifies every shard under the shared challenges and requires
//!
//! ```text
//! Σ_k residual_k  +  Σ_m multiplicity_m · (β + fingerprint(γ, args_m))⁻¹  =  0
//! ```
//!
//! where the second sum ranges over the preamble's messages. A message with
//! multiplicity one is a claim that no shard had to carry; a message with
//! multiplicity minus one is a pull the verifier performs on the batch's
//! behalf. Together they let an application close identities that no single
//! shard can — e.g. the boundary terms of a table split across shards.
//!
//! A batch with one shard and no messages proves what a single proof of the
//! same witness proves, but under a different transcript; the two are
//! deliberately distinct protocols, and neither verifier accepts the other's
//! proofs.
//!
//! # What is and is not checked here
//!
//! The batch verifier guarantees the shape agreement between preamble and
//! proofs, the shared-challenge transcript, each shard's constraints and
//! openings, and the residual sum. It attaches no meaning to the messages:
//! which messages a batch is allowed to carry, and how many, is the
//! application's policy, and an application that lets a prover choose messages
//! freely gives up soundness (with two free base-field values a prover can
//! solve the two coordinates of any target imbalance). Applications must fix
//! the admissible message set from data the verifier trusts or checks.

use crate::config::{Com, PcsError, StarkGenericConfig, Val};
use crate::lookup::fingerprint;
use crate::prover::{Proof, Stage1, claims_accumulator, observe_claims, sample_lookup_challenges};
use crate::system::{ProverKey, System, SystemWitness};
use crate::verifier::VerificationError;
use crate::{ensure, ensure_eq};

use bincode::error::{DecodeError, EncodeError};
use bincode::serde::{decode_from_slice, encode_to_vec};
use p3_challenger::CanObserve;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use serde::{Deserialize, Serialize};

/// What a shard publishes after round one: everything about it that the
/// lookup challenges must depend on.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ShardHeader<SC: StarkGenericConfig> {
    /// Activation bitmap over the canonical circuit set.
    pub active: Vec<bool>,
    /// The shard's stage-1 commitment.
    pub stage_1_trace: Com<SC>,
    /// Log2 trace height of each active circuit.
    pub log_degrees: Vec<u8>,
    /// The shard's public claims, each pushed with multiplicity one.
    pub claims: Vec<Vec<Val<SC>>>,
}

impl<SC: StarkGenericConfig> Clone for ShardHeader<SC> {
    fn clone(&self) -> Self {
        Self {
            active: self.active.clone(),
            stage_1_trace: self.stage_1_trace.clone(),
            log_degrees: self.log_degrees.clone(),
            claims: self.claims.clone(),
        }
    }
}

impl<SC: StarkGenericConfig> PartialEq for ShardHeader<SC>
where
    Com<SC>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.active == other.active
            && self.stage_1_trace == other.stage_1_trace
            && self.log_degrees == other.log_degrees
            && self.claims == other.claims
    }
}

/// What a batch prover keeps of each shard between round one, which commits
/// stage 1 and publishes the header, and round two, which proves from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retention {
    /// Hold every shard's committed stage 1 (traces, lookup witness, LDE and
    /// Merkle tree) across the barrier. Nothing is recomputed; the peak is
    /// the stage-1 state of the whole batch at once.
    Retain,
    /// Keep only the headers. Round two rebuilds each shard's witness and
    /// recommits its stage 1, which must reproduce the header the batch
    /// challenges were derived from. The peak is one shard's stage-1 state
    /// plus whatever the witness source holds, at the price of one extra
    /// witness build and stage-1 commitment per shard.
    Regenerate,
}

/// A public lookup message the verifier contributes to the batch's balance,
/// with a signed multiplicity (as a field element).
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BatchMessage<SC: StarkGenericConfig> {
    pub args: Vec<Val<SC>>,
    pub multiplicity: Val<SC>,
}

impl<SC: StarkGenericConfig> Clone for BatchMessage<SC> {
    fn clone(&self) -> Self {
        Self {
            args: self.args.clone(),
            multiplicity: self.multiplicity,
        }
    }
}

impl<SC: StarkGenericConfig> BatchMessage<SC> {
    /// A message pushed once: the batch-level analogue of a claim.
    pub fn push(args: Vec<Val<SC>>) -> Self {
        Self {
            args,
            multiplicity: Val::<SC>::ONE,
        }
    }

    /// A message pulled once.
    pub fn pull(args: Vec<Val<SC>>) -> Self {
        Self {
            args,
            multiplicity: Val::<SC>::NEG_ONE,
        }
    }
}

/// Everything the shared lookup challenges are derived from.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BatchPreamble<SC: StarkGenericConfig> {
    /// One header per shard, in shard order.
    pub headers: Vec<ShardHeader<SC>>,
    /// The verifier-side messages entering the residual sum.
    pub messages: Vec<BatchMessage<SC>>,
}

impl<SC: StarkGenericConfig> Clone for BatchPreamble<SC> {
    fn clone(&self) -> Self {
        Self {
            headers: self.headers.clone(),
            messages: self.messages.clone(),
        }
    }
}

/// One shard's input to [`System::prove_batch`]: its claims and its stage-1
/// witness.
pub struct ShardInput<SC: StarkGenericConfig> {
    pub claims: Vec<Vec<Val<SC>>>,
    pub witness: SystemWitness<Val<SC>>,
}

/// A vector of shard proofs under one preamble.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BatchProof<SC: StarkGenericConfig> {
    pub preamble: BatchPreamble<SC>,
    /// One proof per header, in shard order.
    pub proofs: Vec<Proof<SC>>,
}

impl<SC: StarkGenericConfig> Clone for BatchProof<SC> {
    fn clone(&self) -> Self {
        Self {
            preamble: self.preamble.clone(),
            proofs: self.proofs.clone(),
        }
    }
}

impl<SC: StarkGenericConfig> BatchProof<SC> {
    /// Serializes the batch with the same encoding as [`Proof::to_bytes`],
    /// canonicalizing every shard proof first.
    pub fn to_bytes(&self) -> Result<Vec<u8>, EncodeError> {
        let mut batch = self.clone();
        for proof in &mut batch.proofs {
            SC::canonicalize_proof(proof);
        }
        encode_to_vec(&batch, Proof::<SC>::serde_config())
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let (batch, _num_bytes) = decode_from_slice(bytes, Proof::<SC>::serde_config())?;
        Ok(batch)
    }
}

impl<SC: StarkGenericConfig> Stage1<SC> {
    /// The header this shard contributes to a batch preamble, with the claims
    /// it will carry.
    pub fn header(&self, claims: &[&[Val<SC>]]) -> ShardHeader<SC> {
        ShardHeader {
            active: self.active.clone(),
            stage_1_trace: self.stage_1_trace_commit.clone(),
            log_degrees: self
                .log_degrees
                .iter()
                .map(|&n| u8::try_from(n).expect("log degree exceeds u8"))
                .collect(),
            claims: claims.iter().map(|claim| claim.to_vec()).collect(),
        }
    }
}

fn claim_slices<SC: StarkGenericConfig>(claims: &[Vec<Val<SC>>]) -> Vec<&[Val<SC>]> {
    claims.iter().map(Vec::as_slice).collect()
}

impl<SC: StarkGenericConfig> System<SC> {
    /// The batch transcript up to and including the lookup challenges: the
    /// parameter-seeded challenger observes the system shape, the
    /// preprocessed commitment, every header in order (activation bitmap,
    /// stage-1 commitment, heights, length-prefixed claims), the messages,
    /// and then samples β and γ. Every shard's transcript forks from the
    /// returned state.
    pub fn batch_challenger(
        &self,
        preamble: &BatchPreamble<SC>,
    ) -> (SC::Challenger, SC::Challenge, SC::Challenge) {
        let mut challenger = self.config.initialise_challenger();
        self.observe_shape(&mut challenger);
        if let Some(commit) = &self.preprocessed_commit {
            challenger.observe(commit.clone());
        }
        challenger.observe(Val::<SC>::from_usize(preamble.headers.len()));
        for header in &preamble.headers {
            for &is_active in &header.active {
                challenger.observe(Val::<SC>::from_bool(is_active));
            }
            challenger.observe(header.stage_1_trace.clone());
            challenger.observe(Val::<SC>::from_usize(header.log_degrees.len()));
            for &log_degree in &header.log_degrees {
                challenger.observe(Val::<SC>::from_u8(log_degree));
            }
            observe_claims::<SC>(&mut challenger, &claim_slices::<SC>(&header.claims));
        }
        challenger.observe(Val::<SC>::from_usize(preamble.messages.len()));
        for message in &preamble.messages {
            challenger.observe(Val::<SC>::from_usize(message.args.len()));
            challenger.observe_slice(&message.args);
            challenger.observe(message.multiplicity);
        }
        let (lookup_argument_challenge, fingerprint_challenge) =
            sample_lookup_challenges::<SC>(&mut challenger);
        (challenger, lookup_argument_challenge, fingerprint_challenge)
    }

    /// The accumulator contributed by the batch's messages:
    /// `Σ multiplicity · (β + fingerprint(γ, args))⁻¹`.
    pub fn messages_accumulator(
        lookup_argument_challenge: SC::Challenge,
        fingerprint_challenge: &SC::Challenge,
        messages: &[BatchMessage<SC>],
    ) -> SC::Challenge {
        let mut acc = SC::Challenge::ZERO;
        for message in messages {
            let m = lookup_argument_challenge
                + fingerprint(fingerprint_challenge, message.args.iter().cloned());
            acc += m.inverse() * SC::Challenge::from(message.multiplicity);
        }
        acc
    }

    /// Verifies shard `shard` of a batch and returns its residual (final
    /// accumulator). The shard's header must describe `proof` exactly.
    pub fn verify_batch_shard(
        &self,
        preamble: &BatchPreamble<SC>,
        shard: usize,
        proof: &Proof<SC>,
    ) -> Result<SC::Challenge, VerificationError<PcsError<SC>>>
    where
        Val<SC>: TwoAdicField,
        Com<SC>: PartialEq,
    {
        let (challenger, lookup_argument_challenge, fingerprint_challenge) =
            self.batch_challenger(preamble);
        self.verify_batch_shard_from(
            preamble,
            shard,
            proof,
            challenger,
            lookup_argument_challenge,
            fingerprint_challenge,
        )
    }

    fn verify_batch_shard_from(
        &self,
        preamble: &BatchPreamble<SC>,
        shard: usize,
        proof: &Proof<SC>,
        mut challenger: SC::Challenger,
        lookup_argument_challenge: SC::Challenge,
        fingerprint_challenge: SC::Challenge,
    ) -> Result<SC::Challenge, VerificationError<PcsError<SC>>>
    where
        Val<SC>: TwoAdicField,
        Com<SC>: PartialEq,
    {
        let header = preamble
            .headers
            .get(shard)
            .ok_or(VerificationError::BatchShapeMismatch)?;
        let quotient_degrees = self.verify_shape(proof)?;
        // The challenges were derived from the header, so the proof must be
        // the one the header describes.
        ensure!(
            header.active == proof.active
                && header.stage_1_trace == proof.commitments.stage_1_trace
                && header.log_degrees == proof.log_degrees,
            VerificationError::BatchShapeMismatch
        );
        challenger.observe(Val::<SC>::from_usize(shard));
        let acc = claims_accumulator::<SC>(
            lookup_argument_challenge,
            &fingerprint_challenge,
            &claim_slices::<SC>(&header.claims),
        );
        self.verify_after_challenges(
            proof,
            &quotient_degrees,
            challenger,
            lookup_argument_challenge,
            fingerprint_challenge,
            acc,
        )?;
        proof
            .intermediate_accumulators
            .last()
            .copied()
            .ok_or(VerificationError::InvalidProofShape)
    }

    /// Verifies a batch: header/proof agreement, every shard under the shared
    /// challenges, and the residual sum against the preamble's messages.
    pub fn verify_batch(
        &self,
        batch: &BatchProof<SC>,
    ) -> Result<(), VerificationError<PcsError<SC>>>
    where
        Val<SC>: TwoAdicField,
        Com<SC>: PartialEq,
        SC::Challenger: Clone,
    {
        let BatchProof { preamble, proofs } = batch;
        ensure!(
            !preamble.headers.is_empty() && preamble.headers.len() == proofs.len(),
            VerificationError::BatchShapeMismatch
        );
        let (challenger, lookup_argument_challenge, fingerprint_challenge) =
            self.batch_challenger(preamble);
        let mut total = Self::messages_accumulator(
            lookup_argument_challenge,
            &fingerprint_challenge,
            &preamble.messages,
        );
        for (shard, proof) in proofs.iter().enumerate() {
            total += self.verify_batch_shard_from(
                preamble,
                shard,
                proof,
                challenger.clone(),
                lookup_argument_challenge,
                fingerprint_challenge,
            )?;
        }
        ensure_eq!(
            total,
            SC::Challenge::ZERO,
            VerificationError::UnbalancedBatch
        );
        Ok(())
    }
}

impl<SC> System<SC>
where
    SC: StarkGenericConfig,
    Val<SC>: TwoAdicField,
    SC::Dft: TwoAdicSubgroupDft<Val<SC>>,
{
    /// Round two for one shard: forks the batch transcript at the shard
    /// index and proves from the committed stage 1. `claims` must be the
    /// claims recorded in the shard's header.
    #[tracing::instrument(level = "info", skip_all, name = "stark/prove_batch_shard")]
    pub fn prove_batch_shard(
        &self,
        key: &ProverKey<SC>,
        stage_1: Stage1<SC>,
        claims: &[&[Val<SC>]],
        preamble: &BatchPreamble<SC>,
        shard: usize,
    ) -> Proof<SC> {
        let (mut challenger, lookup_argument_challenge, fingerprint_challenge) =
            self.batch_challenger(preamble);
        challenger.observe(Val::<SC>::from_usize(shard));
        let acc =
            claims_accumulator::<SC>(lookup_argument_challenge, &fingerprint_challenge, claims);
        self.prove_after_challenges(
            key,
            stage_1,
            challenger,
            lookup_argument_challenge,
            fingerprint_challenge,
            acc,
        )
    }

    /// Proves a batch in one process from shards built on demand: shard `k`
    /// carries `claims[k]` and the witness `witness(k)` builds. Round one for
    /// every shard, the preamble, then round two for every shard, keeping
    /// what `retention` says between the two. Distributed provers run the
    /// same three steps with the preamble exchanged between them.
    ///
    /// Under [`Retention::Regenerate`], `witness` is called twice per shard
    /// and must return the same witness both times: the shard's stage-1
    /// commitment is what the batch challenges were derived from.
    ///
    /// # Panics
    /// Panics if `claims` is empty, if any shard's traces are all empty, or
    /// if a regenerated shard does not reproduce its round-one header.
    #[tracing::instrument(level = "info", skip_all, name = "stark/prove_batch")]
    pub fn prove_batch_with<W>(
        &self,
        key: &ProverKey<SC>,
        claims: &[Vec<Vec<Val<SC>>>],
        messages: Vec<BatchMessage<SC>>,
        retention: Retention,
        mut witness: W,
    ) -> BatchProof<SC>
    where
        Com<SC>: PartialEq,
        W: FnMut(usize) -> SystemWitness<Val<SC>>,
    {
        assert!(!claims.is_empty(), "cannot prove an empty batch");
        let mut retained: Vec<Option<Stage1<SC>>> = Vec::with_capacity(claims.len());
        let headers = claims
            .iter()
            .enumerate()
            .map(|(shard, claims)| {
                let _g = tracing::info_span!("stark/batch_round_1", shard).entered();
                let stage_1 = self.prove_stage_1(witness(shard));
                let header = stage_1.header(&claim_slices::<SC>(claims));
                retained.push(match retention {
                    Retention::Retain => Some(stage_1),
                    Retention::Regenerate => None,
                });
                header
            })
            .collect();
        let preamble = BatchPreamble { headers, messages };
        let proofs = retained
            .into_iter()
            .zip(claims)
            .enumerate()
            .map(|(shard, (stage_1, claims))| {
                let _g = tracing::info_span!("stark/batch_round_2", shard).entered();
                let claims = claim_slices::<SC>(claims);
                let stage_1 = stage_1.unwrap_or_else(|| {
                    let stage_1 = self.prove_stage_1(witness(shard));
                    assert!(
                        stage_1.header(&claims) == preamble.headers[shard],
                        "shard {shard} did not reproduce its round-one header"
                    );
                    stage_1
                });
                self.prove_batch_shard(key, stage_1, &claims, &preamble, shard)
            })
            .collect();
        BatchProof { preamble, proofs }
    }

    /// Proves a batch from shards built up front, retaining every shard's
    /// stage 1 across the barrier ([`Retention::Retain`]).
    ///
    /// # Panics
    /// Panics if `shards` is empty or any shard's traces are all empty.
    pub fn prove_batch(
        &self,
        key: &ProverKey<SC>,
        shards: Vec<ShardInput<SC>>,
        messages: Vec<BatchMessage<SC>>,
    ) -> BatchProof<SC>
    where
        Com<SC>: PartialEq,
    {
        let (claims, mut witnesses): (Vec<_>, Vec<_>) = shards
            .into_iter()
            .map(|shard| (shard.claims, Some(shard.witness)))
            .unzip();
        self.prove_batch_with(key, &claims, messages, Retention::Retain, |shard| {
            witnesses[shard].take().expect("each shard is built once")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_circuits::u32_add::tests::{AddCalls, byte_system};
    use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;

    const ADD_WIDTH: usize = 14;

    fn config() -> GoldilocksBlake3Config {
        config_with_pow(0)
    }

    fn config_with_pow(bits: usize) -> GoldilocksBlake3Config {
        GoldilocksBlake3Config::new(
            CommitmentParameters {
                log_blowup: 1,
                cap_height: 0,
            },
            FriParameters {
                log_final_poly_len: 0,
                max_log_arity: 1,
                num_queries: 64,
                commit_proof_of_work_bits: bits,
                query_proof_of_work_bits: bits,
            },
        )
    }

    fn calls() -> AddCalls {
        AddCalls {
            calls: vec![(10, 5), (30, 20), (100, 100), (8000, 10000)],
        }
    }

    fn claims() -> Vec<Vec<Val>> {
        let f = Val::from_u32;
        vec![
            vec![f(1), f(10), f(5), f(15)],
            vec![f(1), f(30), f(20), f(50)],
            vec![f(1), f(100), f(100), f(200)],
            vec![f(1), f(8000), f(10000), f(18000)],
        ]
    }

    fn rows(trace: &RowMajorMatrix<Val>, range: std::ops::Range<usize>) -> RowMajorMatrix<Val> {
        RowMajorMatrix::new(
            trace.values[range.start * ADD_WIDTH..range.end * ADD_WIDTH].to_vec(),
            ADD_WIDTH,
        )
    }

    /// The u32-add rows split in two; the byte table — pulled by rows of
    /// both shards — lives whole in shard 0 and is inactive in shard 1.
    fn two_shards(
        system: &System<GoldilocksBlake3Config>,
    ) -> Vec<ShardInput<GoldilocksBlake3Config>> {
        let full = calls().witness(system);
        let byte_table = full.traces[0].clone();
        let add = &full.traces[1];
        assert_eq!(add.height(), 4);
        let claims = claims();
        let shard_0 = SystemWitness::from_stage_1(vec![byte_table, rows(add, 0..2)], system);
        let shard_1 = SystemWitness::from_stage_1(
            vec![RowMajorMatrix::new(vec![], 1), rows(add, 2..4)],
            system,
        );
        vec![
            ShardInput {
                claims: claims[0..2].to_vec(),
                witness: shard_0,
            },
            ShardInput {
                claims: claims[2..4].to_vec(),
                witness: shard_1,
            },
        ]
    }

    #[test]
    fn two_shards_verify_and_round_trip() {
        let (system, key) = byte_system(config());
        let batch = system.prove_batch(&key, two_shards(&system), vec![]);
        assert_eq!(batch.proofs.len(), 2);
        assert!(batch.proofs[0].active[0], "byte table active in shard 0");
        assert!(!batch.proofs[1].active[0], "byte table inactive in shard 1");
        system.verify_batch(&batch).unwrap();

        let bytes = batch.to_bytes().unwrap();
        let decoded = BatchProof::<GoldilocksBlake3Config>::from_bytes(&bytes).unwrap();
        system.verify_batch(&decoded).unwrap();
    }

    #[test]
    fn regenerated_shards_prove_the_same_batch() {
        let (system, key) = byte_system(config());
        let retained = system.prove_batch(&key, two_shards(&system), vec![]);
        let claims: Vec<_> = two_shards(&system).into_iter().map(|s| s.claims).collect();
        let mut builds = 0;
        let regenerated =
            system.prove_batch_with(&key, &claims, vec![], Retention::Regenerate, |shard| {
                builds += 1;
                two_shards(&system).swap_remove(shard).witness
            });
        assert_eq!(builds, 4, "each shard is built once per round");
        assert_eq!(
            regenerated.to_bytes().unwrap(),
            retained.to_bytes().unwrap()
        );
        system.verify_batch(&regenerated).unwrap();
    }

    #[test]
    fn positive_proof_of_work_batches_are_reproducible() {
        // With a grind in every shard's opening, the proof bytes must still
        // not depend on which policy built the batch or on the run.
        let (system, key) = byte_system(config_with_pow(6));
        let claims: Vec<_> = two_shards(&system).into_iter().map(|s| s.claims).collect();
        let prove = |retention| {
            system
                .prove_batch_with(&key, &claims, vec![], retention, |shard| {
                    two_shards(&system).swap_remove(shard).witness
                })
                .to_bytes()
                .unwrap()
        };
        let first = prove(Retention::Regenerate);
        assert_eq!(first, prove(Retention::Regenerate));
        assert_eq!(first, prove(Retention::Retain));
        let batch = BatchProof::<GoldilocksBlake3Config>::from_bytes(&first).unwrap();
        system.verify_batch(&batch).unwrap();
    }

    #[test]
    #[should_panic(expected = "did not reproduce its round-one header")]
    fn regenerated_shard_must_match_its_header() {
        let (system, key) = byte_system(config());
        let claims: Vec<_> = two_shards(&system).into_iter().map(|s| s.claims).collect();
        let mut round = [0usize; 2];
        system.prove_batch_with(&key, &claims, vec![], Retention::Regenerate, |shard| {
            round[shard] += 1;
            // Shard 1 comes back with different rows in round two.
            let source = if shard == 1 && round[shard] == 2 {
                0
            } else {
                shard
            };
            two_shards(&system).swap_remove(source).witness
        });
    }

    #[test]
    fn shard_residuals_are_nonzero_and_cancel() {
        let (system, key) = byte_system(config());
        let batch = system.prove_batch(&key, two_shards(&system), vec![]);
        let r0 = system
            .verify_batch_shard(&batch.preamble, 0, &batch.proofs[0])
            .unwrap();
        let r1 = system
            .verify_batch_shard(&batch.preamble, 1, &batch.proofs[1])
            .unwrap();
        assert_ne!(r0, ExtVal::ZERO);
        assert_ne!(r1, ExtVal::ZERO);
        assert_eq!(r0 + r1, ExtVal::ZERO);
    }

    #[test]
    fn a_shard_is_not_a_standalone_proof() {
        let (system, key) = byte_system(config());
        let shards = two_shards(&system);
        let batch = system.prove_batch(&key, shards, vec![]);
        let claims = claims();
        let shard_0_claims: Vec<&[Val]> = claims[0..2].iter().map(Vec::as_slice).collect();
        assert!(matches!(
            system.verify_multiple_claims(&shard_0_claims, &batch.proofs[0]),
            Err(VerificationError::UnbalancedChannel)
        ));
    }

    #[test]
    fn batch_of_one_is_a_distinct_protocol() {
        let (system, key) = byte_system(config());
        let witness = calls().witness(&system);
        let claims = claims();
        let batch = system.prove_batch(
            &key,
            vec![ShardInput {
                claims: claims.clone(),
                witness,
            }],
            vec![],
        );
        system.verify_batch(&batch).unwrap();
        let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();
        // Balanced on its own, but proven under the batch transcript: the
        // single-proof verifier derives different challenges and rejects.
        assert!(
            system
                .verify_multiple_claims(&claim_refs, &batch.proofs[0])
                .is_err()
        );
    }

    #[test]
    fn messages_stand_in_for_claims() {
        let (system, key) = byte_system(config());
        let mut shards = two_shards(&system);
        let moved: Vec<Vec<Val>> = std::mem::take(&mut shards[1].claims);
        let messages = moved.into_iter().map(BatchMessage::push).collect();
        let batch = system.prove_batch(&key, shards, messages);
        assert!(batch.preamble.headers[1].claims.is_empty());
        system.verify_batch(&batch).unwrap();
    }

    #[test]
    fn opposite_messages_cancel() {
        let (system, key) = byte_system(config());
        let f = Val::from_u32;
        let args = vec![f(7), f(8), f(9)];
        let messages = vec![BatchMessage::push(args.clone()), BatchMessage::pull(args)];
        let batch = system.prove_batch(&key, two_shards(&system), messages);
        system.verify_batch(&batch).unwrap();
    }

    #[test]
    fn rejects_unbalanced_message() {
        let (system, key) = byte_system(config());
        let f = Val::from_u32;
        let messages = vec![BatchMessage::push(vec![f(7), f(8), f(9)])];
        let batch = system.prove_batch(&key, two_shards(&system), messages);
        assert!(matches!(
            system.verify_batch(&batch),
            Err(VerificationError::UnbalancedBatch)
        ));
    }

    #[test]
    fn rejects_message_added_after_proving() {
        let (system, key) = byte_system(config());
        let mut batch = system.prove_batch(&key, two_shards(&system), vec![]);
        let f = Val::from_u32;
        let args = vec![f(7), f(8), f(9)];
        // The pair is balanced, but the challenges were sampled without it.
        batch
            .preamble
            .messages
            .push(BatchMessage::push(args.clone()));
        batch.preamble.messages.push(BatchMessage::pull(args));
        assert!(system.verify_batch(&batch).is_err());
    }

    #[test]
    fn rejects_reordered_and_missing_shards() {
        let (system, key) = byte_system(config());
        let batch = system.prove_batch(&key, two_shards(&system), vec![]);

        let mut swapped = batch.clone();
        swapped.proofs.swap(0, 1);
        assert!(matches!(
            system.verify_batch(&swapped),
            Err(VerificationError::BatchShapeMismatch)
        ));

        let mut truncated = batch.clone();
        truncated.proofs.pop();
        assert!(matches!(
            system.verify_batch(&truncated),
            Err(VerificationError::BatchShapeMismatch)
        ));

        let mut headerless = batch;
        headerless.preamble.headers.pop();
        assert!(matches!(
            system.verify_batch(&headerless),
            Err(VerificationError::BatchShapeMismatch)
        ));
    }

    #[test]
    fn rejects_tampered_header_claim() {
        let (system, key) = byte_system(config());
        let mut batch = system.prove_batch(&key, two_shards(&system), vec![]);
        batch.preamble.headers[1].claims[0][3] += Val::ONE;
        assert!(system.verify_batch(&batch).is_err());
    }

    #[test]
    fn rejects_tampered_residual() {
        let (system, key) = byte_system(config());
        let mut batch = system.prove_batch(&key, two_shards(&system), vec![]);
        let last = batch.proofs[0].intermediate_accumulators.len() - 1;
        batch.proofs[0].intermediate_accumulators[last] += ExtVal::ONE;
        assert!(system.verify_batch(&batch).is_err());
    }

    type ExtVal = <GoldilocksBlake3Config as StarkGenericConfig>::Challenge;
}
