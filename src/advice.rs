//! Re-encoding of a proof's FRI opening transport for per-query verifiers.
//!
//! Plonky3 v0.6.0 ships FRI query openings as *pruned Merkle multiproofs*:
//! per commitment, one flat list of boundary sibling digests shared by all
//! queries, verified by an amortized bottom-up walk
//! (`MerkleTreeMmcs::verify_batch_pruned`). The in-circuit recursive
//! verifier instead consumes one full authentication path per query — the
//! legacy per-query layout — because its per-query control flow is a far
//! smaller circuit than the amortized walk's sort/merge bookkeeping.
//!
//! [`proof_to_advice_bytes`] converts a verified [`Proof`] into that
//! per-query **advice encoding**: the outer proof fields unchanged, the FRI
//! opening transport expanded from pruned multiproofs to per-query paths.
//!
//! # Soundness: encoding freedom
//!
//! The advice bytes are untrusted prover input to the recursive verifier —
//! never digest-bound, never observed into the transcript. What the
//! transcript binds are the *commitments* (read from the advice and
//! observed), and every expanded sibling digest is authenticated against
//! them by the per-query Merkle checks. Pruning is transport compression:
//! a pruned proof and its expansion authenticate identical opened values
//! against identical commitments, so a valid advice encoding exists iff a
//! valid pruned proof exists for the same statement. Per-query
//! verification is at least as strong as the amortized walk — duplicate
//! queries that disagree on opened values would need a hash collision to
//! both authenticate.
//!
//! # How the expansion recovers interior digests
//!
//! No tree data is available (the input is a proof, not prover data), and
//! the walk that recomputes interior digests lives inside p3. Rather than
//! reimplement it, the expansion *runs* it: `verify_fri` is invoked with a
//! [`MerkleTreeMmcs`] whose compression function records every
//! `(inputs → output)` call. The recorded map is functional (a collision
//! would break Blake3), so each query's full path is read back by walking
//! the tree top-down from its cap entry: at every level the map yields the
//! two children — one continues the path, the other is the sibling the
//! legacy wire format carries. Matrix-injection levels (a shorter matrix's
//! row hash compressed into the running digest) consume no wire sibling,
//! mirroring the per-path verifier's schedule. Query indices come from the
//! same instrumented run, via a challenger wrapper that records
//! `sample_bits` results.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use bincode::serde::encode_to_vec;
use p3_blake3::Blake3;
use p3_challenger::{CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, OpenedValuesForRound};
use p3_field::{BasedVectorSpace, Field};
use p3_fri::verifier::verify_fri;
use p3_fri::{BatchMultiOpening, CommitPhaseMultiStep, FriProof};
use p3_fri::{FriParameters as InnerFriParameters, TwoAdicFriFolding};
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicHasher, PseudoCompressionFunction,
    SerializingHasher,
};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

use crate::prover::{Commitments, Proof};
use crate::system::System;
use crate::types::{
    Challenger, Commitment, CommitmentParameters, ExtVal, FriParameters, GoldilocksBlake3Config,
    Val,
};

type Digest = [u8; 32];
type Blake3Compress = CompressionFunctionFromHasher<Blake3, 2, 32>;
type RecMmcs = MerkleTreeMmcs<Val, u8, SerializingHasher<Blake3>, RecordingCompress, 2, 32>;
type RecExtMmcs = ExtensionMmcs<Val, ExtVal, RecMmcs>;

/// Why an advice re-encoding could not be produced. Every variant except
/// [`AdviceError::Encode`] indicates an invalid proof or a bug: expansion
/// only runs after native verification succeeds.
#[derive(Debug)]
pub enum AdviceError {
    /// Native verification of the proof failed; nothing was expanded.
    Verification(String),
    /// The instrumented PCS run failed (unreachable after native
    /// verification passed, absent a bug).
    Recording(String),
    /// A digest needed for path read-back was never computed by the
    /// instrumented run.
    MissingDigest,
    /// A cap index fell outside the commitment.
    CapIndexOutOfBounds,
    /// The recorded `sample_bits` log did not contain the query indices.
    MissingQueryIndices,
    /// Serialization of the legacy layout failed.
    Encode(bincode::error::EncodeError),
}

// ---------------------------------------------------------------------------
// Instrumentation
// ---------------------------------------------------------------------------

/// A compression function that forwards to Blake3 and records every call.
/// The map is keyed by output: Blake3 collisions aside, each digest has a
/// unique preimage pair, so read-back is unambiguous.
#[derive(Clone)]
struct RecordingCompress {
    inner: Blake3Compress,
    log: Arc<Mutex<HashMap<Digest, [Digest; 2]>>>,
}

impl PseudoCompressionFunction<Digest, 2> for RecordingCompress {
    fn compress(&self, input: [Digest; 2]) -> Digest {
        let output = self.inner.compress(input);
        self.log.lock().unwrap().insert(output, input);
        output
    }
}

/// A challenger that forwards everything and records `sample_bits` calls.
/// FRI query indices are the last `num_queries` recorded samples: they are
/// drawn after the query proof-of-work check, and nothing samples bits
/// after them.
#[derive(Clone)]
struct RecordingChallenger<C> {
    inner: C,
    samples: Arc<Mutex<Vec<usize>>>,
}

impl<T, C: CanObserve<T>> CanObserve<T> for RecordingChallenger<C> {
    fn observe(&mut self, value: T) {
        self.inner.observe(value);
    }
}

impl<T, C: CanSample<T>> CanSample<T> for RecordingChallenger<C> {
    fn sample(&mut self) -> T {
        self.inner.sample()
    }
}

impl<C: CanSampleBits<usize>> CanSampleBits<usize> for RecordingChallenger<C> {
    fn sample_bits(&mut self, bits: usize) -> usize {
        let value = self.inner.sample_bits(bits);
        self.samples.lock().unwrap().push(value);
        value
    }
}

impl<F: Field, C: FieldChallenger<F>> FieldChallenger<F> for RecordingChallenger<C> {}

impl<C: GrindingChallenger> GrindingChallenger for RecordingChallenger<C> {
    type Witness = C::Witness;

    fn grind(&mut self, bits: usize) -> Self::Witness {
        self.inner.grind(bits)
    }

    // Delegate so the inner challenger's own `sample_bits` runs unrecorded:
    // proof-of-work samples never pollute the query-index log.
    fn check_witness(&mut self, bits: usize, witness: Self::Witness) -> bool {
        self.inner.check_witness(bits, witness)
    }
}

// ---------------------------------------------------------------------------
// Legacy wire layout (the advice encoding)
// ---------------------------------------------------------------------------

/// One round's input opening for one query: the opened rows and a full
/// Merkle authentication path.
#[derive(Serialize, Deserialize)]
pub struct AdviceBatchOpening {
    pub opened_values: Vec<Vec<Val>>,
    pub opening_proof: Vec<Digest>,
}

/// One FRI folding step for one query: the folding arity, the `arity - 1`
/// sibling evaluations, and a full path into that round's commitment.
#[derive(Serialize, Deserialize)]
pub struct AdviceCommitPhaseStep {
    pub log_arity: u8,
    pub sibling_values: Vec<ExtVal>,
    pub opening_proof: Vec<Digest>,
}

/// One query's complete opening data: input openings per round, then one
/// folding step per commit phase.
#[derive(Serialize, Deserialize)]
pub struct AdviceQueryProof {
    pub input_proof: Vec<AdviceBatchOpening>,
    pub commit_phase_openings: Vec<AdviceCommitPhaseStep>,
}

/// The FRI proof in per-query transport.
#[derive(Serialize, Deserialize)]
pub struct AdviceFriProof {
    pub commit_phase_commits: Vec<Commitment>,
    pub commit_pow_witnesses: Vec<Val>,
    pub query_proofs: Vec<AdviceQueryProof>,
    pub final_poly: Vec<ExtVal>,
    pub query_pow_witness: Val,
}

/// The full advice proof: every field of [`Proof`] unchanged except the
/// opening transport.
#[derive(Serialize, Deserialize)]
pub struct AdviceProof {
    pub active: Vec<bool>,
    pub commitments: Commitments<Commitment>,
    pub intermediate_accumulators: Vec<ExtVal>,
    pub log_degrees: Vec<u8>,
    pub opening_proof: AdviceFriProof,
    pub quotient_opened_values: OpenedValuesForRound<ExtVal>,
    pub preprocessed_opened_values: Option<OpenedValuesForRound<ExtVal>>,
    pub stage_1_opened_values: OpenedValuesForRound<ExtVal>,
    pub stage_2_opened_values: OpenedValuesForRound<ExtVal>,
}

// ---------------------------------------------------------------------------
// Path read-back
// ---------------------------------------------------------------------------

/// One step of the per-path verifier's ascent, derived from the matrix
/// dimensions exactly as `verify_batch` derives it.
enum LevelOp {
    /// An arity-`step` compression; the path consumes `step - 1` wire
    /// siblings here.
    Fold { step: usize },
    /// A shorter matrix's row hash is compressed into the running digest;
    /// nothing on the wire.
    Inject,
}

/// The ascent schedule (leaf to cap) for a tree of the given dimensions,
/// mirroring `verify_batch`'s traversal: one `Fold` per arity-schedule
/// step, an `Inject` after any step whose folded height picks up shorter
/// matrices.
fn ascent_schedule(
    schedule: &[usize],
    dimensions: &[Dimensions],
    max_height: usize,
) -> Vec<LevelOp> {
    let mut heights: Vec<usize> = dimensions.iter().map(|d| d.height).collect();
    heights.sort_unstable_by(|a, b| b.cmp(a));
    let leaf_npt = max_height.next_power_of_two();
    // Heights hashed into the leaf layer are consumed up front.
    let mut next = heights
        .iter()
        .position(|h| h.next_power_of_two() != leaf_npt)
        .unwrap_or(heights.len());

    let mut ops = Vec::new();
    let mut curr_height_padded = max_height.next_multiple_of(2);
    for &step in schedule {
        ops.push(LevelOp::Fold { step });
        let logical_next = curr_height_padded / step;
        curr_height_padded = logical_next.next_multiple_of(2);
        let logical_next_npt = logical_next.next_power_of_two();
        if next < heights.len() && heights[next].next_power_of_two() == logical_next_npt {
            ops.push(LevelOp::Inject);
            while next < heights.len() && heights[next].next_power_of_two() == logical_next_npt {
                next += 1;
            }
        }
    }
    ops
}

/// Reads one query's full authentication path out of the recorded
/// compression map by walking its tree top-down from the cap entry,
/// emitting siblings in the bottom-up order the legacy wire carries.
///
/// `leaf_check`, when supplied, is the expected leaf digest (the hash of
/// the query's opened rows); the walk must land exactly there.
fn expand_path(
    map: &HashMap<Digest, [Digest; 2]>,
    commitment: &Commitment,
    schedule: &[usize],
    dimensions: &[Dimensions],
    max_height: usize,
    index: usize,
    leaf_check: Option<Digest>,
) -> Result<Vec<Digest>, AdviceError> {
    let ops = ascent_schedule(schedule, dimensions, max_height);

    // Per-fold node index at each level, bottom-up.
    let mut level_indices = Vec::new();
    let mut idx = index;
    for op in &ops {
        if let LevelOp::Fold { step } = op {
            level_indices.push(idx);
            idx /= step;
        }
    }
    let cap_index = idx;
    if cap_index >= commitment.num_roots() {
        return Err(AdviceError::CapIndexOutOfBounds);
    }

    let mut digest: Digest = commitment[cap_index];
    let mut fold_level = level_indices.len();
    // Wire order is bottom-up; the walk is top-down, so collect reversed.
    let mut siblings_rev: Vec<Digest> = Vec::new();
    for op in ops.iter().rev() {
        let inputs = map.get(&digest).ok_or(AdviceError::MissingDigest)?;
        match op {
            LevelOp::Inject => {
                // inject_inputs = [running digest, row hash]; descend left.
                digest = inputs[0];
            }
            LevelOp::Fold { step } => {
                fold_level -= 1;
                let pos_in_group = level_indices[fold_level] % step;
                // The recording mmcs is binary (N = 2), so `step` is 2.
                for k in (0..*step).rev() {
                    if k != pos_in_group {
                        siblings_rev.push(inputs[k]);
                    }
                }
                digest = inputs[pos_in_group];
            }
        }
    }
    if let Some(expected) = leaf_check
        && digest != expected
    {
        return Err(AdviceError::MissingDigest);
    }
    siblings_rev.reverse();
    Ok(siblings_rev)
}

// ---------------------------------------------------------------------------
// Re-encoding
// ---------------------------------------------------------------------------

/// Converts a natively-verified proof into the per-query advice encoding.
///
/// `commitment_parameters` and `fri_parameters` must be the ones the
/// system's config was built from (the config does not expose them back).
/// Verifies the proof natively first and refuses to expand on failure.
pub fn proof_to_advice_bytes(
    system: &System<GoldilocksBlake3Config>,
    commitment_parameters: CommitmentParameters,
    fri_parameters: FriParameters,
    claims: &[&[Val]],
    proof: &Proof<GoldilocksBlake3Config>,
) -> Result<Vec<u8>, AdviceError> {
    system
        .verify_multiple_claims(claims, proof)
        .map_err(|e| AdviceError::Verification(format!("{e:?}")))?;
    let ctx = system
        .pcs_verification_context(claims, proof)
        .map_err(|e| AdviceError::Verification(format!("{e:?}")))?;

    // Instrumented components sharing one compression log.
    let log: Arc<Mutex<HashMap<Digest, [Digest; 2]>>> = Arc::new(Mutex::new(HashMap::new()));
    let rec_compress = RecordingCompress {
        inner: Blake3Compress::new(Blake3),
        log: Arc::clone(&log),
    };
    let rec_mmcs = RecMmcs::new(
        SerializingHasher::new(Blake3),
        rec_compress,
        commitment_parameters.cap_height,
    );
    let rec_params = InnerFriParameters {
        log_blowup: commitment_parameters.log_blowup,
        log_final_poly_len: fri_parameters.log_final_poly_len,
        max_log_arity: fri_parameters.max_log_arity,
        num_queries: fri_parameters.num_queries,
        commit_proof_of_work_bits: fri_parameters.commit_proof_of_work_bits,
        query_proof_of_work_bits: fri_parameters.query_proof_of_work_bits,
        mmcs: ExtensionMmcs::<Val, ExtVal, _>::new(rec_mmcs.clone()),
    };

    // The stored FRI proof, re-typed over the instrumented mmcs. The
    // commitment, digest and pruned-path types are identical; only the
    // phantom mmcs parameter changes.
    let fri = &proof.opening_proof;
    let rec_proof: FriProof<ExtVal, RecExtMmcs, Val, Vec<BatchMultiOpening<Val, RecMmcs>>> =
        FriProof {
            commit_phase_commits: fri.commit_phase_commits.clone(),
            commit_pow_witnesses: fri.commit_pow_witnesses.clone(),
            input_openings: fri
                .input_openings
                .iter()
                .map(|o| BatchMultiOpening {
                    opened_values: o.opened_values.clone(),
                    opening_proof: o.opening_proof.clone(),
                })
                .collect(),
            commit_phase_openings: fri
                .commit_phase_openings
                .iter()
                .map(|s| CommitPhaseMultiStep {
                    log_arity: s.log_arity,
                    sibling_values: s.sibling_values.clone(),
                    opening_proof: s.opening_proof.clone(),
                })
                .collect(),
            final_poly: fri.final_poly.clone(),
            query_pow_witness: fri.query_pow_witness,
        };

    // Run the PCS phase exactly as `TwoAdicFriPcs::verify` would — the
    // evaluation observations, then `verify_fri` — against the context's
    // rounds and transcript state, with the instrumented components.
    let mut challenger = RecordingChallenger::<Challenger> {
        inner: ctx.challenger,
        samples: Arc::new(Mutex::new(Vec::new())),
    };
    for (_, round) in &ctx.rounds {
        for (_, mat) in round {
            for (_, point) in mat {
                challenger.observe_algebra_slice(point);
            }
        }
    }
    let folding = TwoAdicFriFolding::<
        Vec<BatchMultiOpening<Val, RecMmcs>>,
        <RecMmcs as Mmcs<Val>>::Error,
    >(PhantomData);
    verify_fri(
        &folding,
        &rec_params,
        &rec_proof,
        &mut challenger,
        &ctx.rounds,
        &rec_mmcs,
    )
    .map_err(|e| AdviceError::Recording(format!("{e:?}")))?;

    // The query indices are the last `num_queries` recorded samples.
    let samples = challenger.samples.lock().unwrap();
    let query_indices: Vec<usize> = samples
        .len()
        .checked_sub(fri_parameters.num_queries)
        .map(|start| samples[start..].to_vec())
        .ok_or(AdviceError::MissingQueryIndices)?;
    drop(samples);
    let map = log.lock().unwrap();

    let log_arities: Vec<usize> = fri
        .commit_phase_openings
        .iter()
        .map(|s| usize::from(s.log_arity))
        .collect();
    let total_log_reduction: usize = log_arities.iter().sum();
    let log_global_max_height =
        total_log_reduction + commitment_parameters.log_blowup + fri_parameters.log_final_poly_len;

    let hasher = SerializingHasher::new(Blake3);

    // Input rounds: per round, per query, opened rows + expanded path.
    // Dimensions and reduced indices mirror p3's `open_inputs`.
    let mut input_openings_per_query: Vec<Vec<AdviceBatchOpening>> = (0..fri_parameters
        .num_queries)
        .map(|_| Vec::new())
        .collect();
    for ((commit, mats), batch) in ctx.rounds.iter().zip(&fri.input_openings) {
        let heights: Vec<usize> = mats
            .iter()
            .map(|(domain, _)| domain.size() << commitment_parameters.log_blowup)
            .collect();
        let dims: Vec<Dimensions> = heights
            .iter()
            .zip(mats)
            .map(|(&height, (_, points))| Dimensions {
                width: points.first().map_or(0, |(_, values)| values.len()),
                height,
            })
            .collect();
        let max_height = heights.iter().copied().max().unwrap_or(1);
        let bits_reduced = log_global_max_height - log2_strict_usize(max_height);
        let schedule = rec_mmcs
            .proof_arity_schedule(&dims)
            .map_err(|e| AdviceError::Recording(format!("{e:?}")))?;

        // Matrices whose padded height matches the tallest are hashed into
        // the leaf digest, in tallest-first order — mirroring `verify_batch`.
        let leaf_npt = max_height.next_power_of_two();
        let mut order: Vec<usize> = (0..dims.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(dims[i].height));
        let leaf_matrices: Vec<usize> = order
            .iter()
            .copied()
            .take_while(|&i| dims[i].height.next_power_of_two() == leaf_npt)
            .collect();

        for (q, per_query) in input_openings_per_query.iter_mut().enumerate() {
            let reduced_index = query_indices[q] >> bits_reduced;
            let opened_values = batch.opened_values[q].clone();
            let leaf = hasher
                .hash_iter_slices(leaf_matrices.iter().map(|&mi| opened_values[mi].as_slice()));
            let opening_proof = expand_path(
                &map,
                commit,
                &schedule,
                &dims,
                max_height,
                reduced_index,
                Some(leaf),
            )?;
            per_query.push(AdviceBatchOpening {
                opened_values,
                opening_proof,
            });
        }
    }

    // Commit-phase rounds: single-matrix trees of the folded codewords,
    // flattened to base columns by the extension mmcs. The wire sibling
    // values come straight off the multiproof; only the paths are read
    // back. No leaf check: the folded row is not materialized here — the
    // walk's landing digest is pinned by the functional map and the cap.
    let ext_d = <ExtVal as BasedVectorSpace<Val>>::DIMENSION;
    let mut steps_per_query: Vec<Vec<AdviceCommitPhaseStep>> = (0..fri_parameters.num_queries)
        .map(|_| Vec::new())
        .collect();
    let mut log_current_height = log_global_max_height;
    for (round, step) in fri.commit_phase_openings.iter().enumerate() {
        let log_arity = usize::from(step.log_arity);
        let arity = 1 << log_arity;
        let log_folded_height = log_current_height - log_arity;
        let dims = [Dimensions {
            width: arity * ext_d,
            height: 1 << log_folded_height,
        }];
        let schedule = rec_mmcs
            .proof_arity_schedule(&dims)
            .map_err(|e| AdviceError::Recording(format!("{e:?}")))?;
        let commit = &fri.commit_phase_commits[round];
        let bits_consumed: usize = log_arities[..=round].iter().sum();
        for (q, per_query) in steps_per_query.iter_mut().enumerate() {
            let group_index = query_indices[q] >> bits_consumed;
            let opening_proof = expand_path(
                &map,
                commit,
                &schedule,
                &dims,
                1 << log_folded_height,
                group_index,
                None,
            )?;
            per_query.push(AdviceCommitPhaseStep {
                log_arity: step.log_arity,
                sibling_values: step.sibling_values[q].clone(),
                opening_proof,
            });
        }
        log_current_height = log_folded_height;
    }

    let query_proofs: Vec<AdviceQueryProof> = input_openings_per_query
        .into_iter()
        .zip(steps_per_query)
        .map(|(input_proof, commit_phase_openings)| AdviceQueryProof {
            input_proof,
            commit_phase_openings,
        })
        .collect();

    let advice = AdviceProof {
        active: proof.active.clone(),
        commitments: Commitments {
            stage_1_trace: proof.commitments.stage_1_trace.clone(),
            stage_2_trace: proof.commitments.stage_2_trace.clone(),
            quotient_chunks: proof.commitments.quotient_chunks.clone(),
        },
        intermediate_accumulators: proof.intermediate_accumulators.clone(),
        log_degrees: proof.log_degrees.clone(),
        opening_proof: AdviceFriProof {
            commit_phase_commits: fri.commit_phase_commits.clone(),
            commit_pow_witnesses: fri.commit_pow_witnesses.clone(),
            query_proofs,
            final_poly: fri.final_poly.clone(),
            query_pow_witness: fri.query_pow_witness,
        },
        quotient_opened_values: proof.quotient_opened_values.clone(),
        preprocessed_opened_values: proof.preprocessed_opened_values.clone(),
        stage_1_opened_values: proof.stage_1_opened_values.clone(),
        stage_2_opened_values: proof.stage_2_opened_values.clone(),
    };
    encode_to_vec(&advice, Proof::<GoldilocksBlake3Config>::serde_config())
        .map_err(AdviceError::Encode)
}

#[cfg(test)]
mod tests {
    use bincode::serde::decode_from_slice;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;
    use crate::system::ProverKey;
    use crate::system::{CircuitInputs, SystemWitness};

    fn parameters(cap_height: usize) -> (CommitmentParameters, FriParameters) {
        (
            CommitmentParameters {
                log_blowup: 1,
                cap_height,
            },
            FriParameters {
                log_final_poly_len: 0,
                max_log_arity: 1,
                num_queries: 8,
                commit_proof_of_work_bits: 0,
                query_proof_of_work_bits: 0,
            },
        )
    }

    /// Two constraint-free circuits at different trace heights (so the
    /// stage-1 tree exercises matrix injection), one with a preprocessed
    /// matrix (so the preprocessed round exists).
    fn test_system(
        cap_height: usize,
    ) -> (
        System<GoldilocksBlake3Config>,
        ProverKey<GoldilocksBlake3Config>,
    ) {
        let (cp, fp) = parameters(cap_height);
        let config = GoldilocksBlake3Config::new(cp, fp);
        let preprocessed = RowMajorMatrix::new((0..8u32).map(Val::from_u32).collect::<Vec<_>>(), 1);
        let inputs = [
            CircuitInputs {
                main_width: 2,
                preprocessed: Some(preprocessed),
                ..Default::default()
            },
            CircuitInputs {
                main_width: 3,
                ..Default::default()
            },
        ];
        System::new(config, inputs)
    }

    fn test_proof(
        system: &System<GoldilocksBlake3Config>,
        key: &ProverKey<GoldilocksBlake3Config>,
    ) -> Proof<GoldilocksBlake3Config> {
        let trace_1 = RowMajorMatrix::new((0..16u32).map(Val::from_u32).collect::<Vec<_>>(), 2);
        let trace_2 = RowMajorMatrix::new(
            (0..12u32)
                .map(|i| Val::from_u32(7 * i + 3))
                .collect::<Vec<_>>(),
            3,
        );
        let witness = SystemWitness::from_stage_1(vec![trace_1, trace_2], system);
        system.prove_multiple_claims(key, &[], witness)
    }

    #[test]
    fn advice_expands_and_paths_verify_per_query() {
        for cap_height in [0, 1] {
            let (cp, fp) = parameters(cap_height);
            let (system, key) = test_system(cap_height);
            let proof = test_proof(&system, &key);
            let bytes = proof_to_advice_bytes(&system, cp, fp, &[], &proof)
                .unwrap_or_else(|e| panic!("advice expansion failed: {e:?}"));

            let (advice, consumed): (AdviceProof, usize) =
                decode_from_slice(&bytes, Proof::<GoldilocksBlake3Config>::serde_config())
                    .expect("advice bytes decode under the same serde config");
            assert_eq!(consumed, bytes.len(), "no trailing bytes");
            assert_eq!(advice.opening_proof.query_proofs.len(), fp.num_queries);

            // Recover the query indices the same way the expansion did.
            let ctx = system.pcs_verification_context(&[], &proof).unwrap();
            let plain_mmcs = MerkleTreeMmcs::<Val, u8, _, _, 2, 32>::new(
                SerializingHasher::new(Blake3),
                Blake3Compress::new(Blake3),
                cp.cap_height,
            );
            let plain_params = InnerFriParameters {
                log_blowup: cp.log_blowup,
                log_final_poly_len: fp.log_final_poly_len,
                max_log_arity: fp.max_log_arity,
                num_queries: fp.num_queries,
                commit_proof_of_work_bits: fp.commit_proof_of_work_bits,
                query_proof_of_work_bits: fp.query_proof_of_work_bits,
                mmcs: ExtensionMmcs::<Val, ExtVal, _>::new(plain_mmcs.clone()),
            };
            let mut challenger = RecordingChallenger::<Challenger> {
                inner: ctx.challenger,
                samples: Arc::new(Mutex::new(Vec::new())),
            };
            for (_, round) in &ctx.rounds {
                for (_, mat) in round {
                    for (_, point) in mat {
                        challenger.observe_algebra_slice(point);
                    }
                }
            }
            let folding = TwoAdicFriFolding::<
                Vec<BatchMultiOpening<Val, _>>,
                <MerkleTreeMmcs<Val, u8, SerializingHasher<Blake3>, Blake3Compress, 2, 32> as Mmcs<
                    Val,
                >>::Error,
            >(PhantomData);
            verify_fri(
                &folding,
                &plain_params,
                &proof.opening_proof,
                &mut challenger,
                &ctx.rounds,
                &plain_mmcs,
            )
            .unwrap();
            let samples = challenger.samples.lock().unwrap();
            let indices = &samples[samples.len() - fp.num_queries..];

            let log_arities: Vec<usize> = proof
                .opening_proof
                .commit_phase_openings
                .iter()
                .map(|s| usize::from(s.log_arity))
                .collect();
            let log_global_max_height =
                log_arities.iter().sum::<usize>() + cp.log_blowup + fp.log_final_poly_len;

            // Oracle: every expanded input-round path must satisfy p3's own
            // per-path verifier.
            for (round, (commit, mats)) in ctx.rounds.iter().enumerate() {
                let heights: Vec<usize> = mats
                    .iter()
                    .map(|(domain, _)| domain.size() << cp.log_blowup)
                    .collect();
                let dims: Vec<Dimensions> = heights
                    .iter()
                    .zip(mats)
                    .map(|(&height, (_, points))| Dimensions {
                        width: points.first().map_or(0, |(_, values)| values.len()),
                        height,
                    })
                    .collect();
                let max_height = heights.iter().copied().max().unwrap();
                let bits_reduced = log_global_max_height - log2_strict_usize(max_height);
                for (q, &index) in indices.iter().enumerate() {
                    let opening = &advice.opening_proof.query_proofs[q].input_proof[round];
                    plain_mmcs
                        .verify_batch(
                            commit,
                            &dims,
                            index >> bits_reduced,
                            p3_commit::BatchOpeningRef::new(
                                &opening.opened_values,
                                &opening.opening_proof,
                            ),
                        )
                        .unwrap_or_else(|e| {
                            panic!("round {round} query {q}: expanded path rejected: {e:?}")
                        });
                }
            }

            // Commit-phase paths: length and sibling-value parity with the
            // multiproof (the Lean verifier is the end-to-end oracle here).
            let mut log_current = log_global_max_height;
            for (round, step) in proof.opening_proof.commit_phase_openings.iter().enumerate() {
                let log_arity = usize::from(step.log_arity);
                let log_folded = log_current - log_arity;
                let expected_levels = log_folded.saturating_sub(cp.cap_height);
                for (q, qp) in advice.opening_proof.query_proofs.iter().enumerate() {
                    let s = &qp.commit_phase_openings[round];
                    assert_eq!(usize::from(s.log_arity), log_arity);
                    assert_eq!(s.sibling_values, step.sibling_values[q]);
                    assert_eq!(
                        s.opening_proof.len(),
                        expected_levels,
                        "round {round} query {q}: phase path length"
                    );
                }
                log_current = log_folded;
            }
        }
    }

    #[test]
    fn advice_refuses_invalid_proof() {
        let cap_height = 0;
        let (cp, fp) = parameters(cap_height);
        let (system, key) = test_system(cap_height);
        let mut proof = test_proof(&system, &key);
        proof.log_degrees[0] ^= 1;
        assert!(matches!(
            proof_to_advice_bytes(&system, cp, fp, &[], &proof),
            Err(AdviceError::Verification(_))
        ));
    }
}
