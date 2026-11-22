//! Exact `MerkleCap` root-count validation helpers shared by CPU and CUDA
//! MMCS/PCS verifiers.
//!
//! Plonky3's `MerkleCap` derives `Deserialize` directly on `Vec<Digest>`,
//! `SerializingChallenger` observes `cap.roots()` raw without a length prefix,
//! and upstream `MerkleTreeMmcs::{verify_batch, verify_multi_batch}` only check
//! `cap_idx < commit.num_roots()`. Without enforcing the exact expected root
//! count (`1 << cap_height.min(log2_ceil_usize(max_height))`) both in MMCS
//! verification and before Fiat-Shamir observation, a prover can inject extra
//! cap roots to grind challenges for free or collide adjacent transcript fields.

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField, coset::TwoAdicMultiplicativeCoset};
use p3_fri::verifier::FriError;
use p3_fri::{CommitmentWithOpeningPoints, FriParameters, FriProof};
use p3_matrix::Dimensions;
use p3_merkle_tree::{MerkleTreeError, PrunedMerklePaths};
use p3_symmetric::MerkleCap;
use p3_util::{log2_ceil_usize, log2_strict_usize};

/// Exact Merkle cap root count for a binary Merkle tree whose tallest committed
/// matrix has `log2_ceil(max_height) == log_max_height`.
#[inline]
pub(crate) fn expected_merkle_cap_roots(cap_height: usize, log_max_height: usize) -> Option<usize> {
    let effective_cap_height = cap_height.min(log_max_height);
    let shift = u32::try_from(effective_cap_height).ok()?;
    1usize.checked_shl(shift)
}

/// Validates that `commit` carries the exact expected root count for a binary
/// Merkle tree with `log2_ceil(max_height) == log_max_height`.
#[inline]
pub(crate) fn verify_merkle_cap_len_for_log_height<F, W>(
    cap_height: usize,
    commit: &MerkleCap<F, W>,
    log_max_height: usize,
) -> Result<(), MerkleTreeError> {
    let expected_roots = expected_merkle_cap_roots(cap_height, log_max_height)
        .ok_or(MerkleTreeError::CapMismatch)?;
    if commit.num_roots() != expected_roots {
        return Err(MerkleTreeError::CapMismatch);
    }
    Ok(())
}

/// Validates that `commit` carries the exact expected root count for the
/// tallest matrix in `dimensions`.
#[inline]
pub(crate) fn verify_merkle_cap_len<F, W>(
    cap_height: usize,
    commit: &MerkleCap<F, W>,
    dimensions: &[Dimensions],
) -> Result<(), MerkleTreeError> {
    let max_height = dimensions
        .iter()
        .map(|dims| dims.height)
        .max()
        .ok_or(MerkleTreeError::EmptyBatch)?;
    if max_height == 0 {
        return Err(MerkleTreeError::EmptyBatch);
    }
    verify_merkle_cap_len_for_log_height(cap_height, commit, log2_ceil_usize(max_height))
}

/// MMCS that can validate a commitment's exact Merkle cap root count against a
/// verifier-known tree height before observing the commitment into a transcript.
pub trait CapValidatingMmcs<T: Send + Sync + Clone>: Mmcs<T> {
    fn validate_commitment_cap_for_log_height(
        &self,
        commit: &Self::Commitment,
        log_max_height: usize,
    ) -> Result<(), Self::Error>;
}

impl<F, EF, InnerMmcs> CapValidatingMmcs<EF> for ExtensionMmcs<F, EF, InnerMmcs>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMmcs:
        CapValidatingMmcs<F, MultiProof = PrunedMerklePaths<u8, 32>, Error = MerkleTreeError>,
{
    fn validate_commitment_cap_for_log_height(
        &self,
        commit: &Self::Commitment,
        log_max_height: usize,
    ) -> Result<(), Self::Error> {
        // Delegate through `verify_multi_batch` with an empty query set so
        // `InnerMmcs` enforces its own configured `cap_height` without needing
        // access to `ExtensionMmcs`'s private `inner` field.
        let height = u32::try_from(log_max_height)
            .ok()
            .and_then(|shift| 1usize.checked_shl(shift))
            .ok_or(MerkleTreeError::CapMismatch)?;
        let empty_rows: &[Vec<&[EF]>] = &[];
        let empty_proof = PrunedMerklePaths {
            sibling_hashes: Vec::new(),
        };
        self.verify_multi_batch(
            commit,
            &[Dimensions { width: 1, height }],
            &[],
            empty_rows,
            &empty_proof,
        )
    }
}

/// Validates the exact Merkle cap root count of every input commitment and
/// every FRI commit-phase commitment before any commitment is observed into the
/// Fiat-Shamir transcript.
pub(crate) fn validate_fri_commitment_caps<
    Val,
    Challenge,
    InputMmcs,
    FriMmcs,
    Witness,
    InputProof,
>(
    params: &FriParameters<FriMmcs>,
    proof: &FriProof<Challenge, FriMmcs, Witness, InputProof>,
    commitments_with_opening_points: &[CommitmentWithOpeningPoints<
        Challenge,
        InputMmcs::Commitment,
        TwoAdicMultiplicativeCoset<Val>,
    >],
    input_mmcs: &InputMmcs,
) -> Result<(), FriError<FriMmcs::Error, InputMmcs::Error>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: CapValidatingMmcs<Val>,
    FriMmcs: CapValidatingMmcs<Challenge>,
{
    if params.num_queries == 0 {
        return Err(FriError::ZeroQueries);
    }

    for (comm, mats) in commitments_with_opening_points {
        if let Some(max_log_height) = mats
            .iter()
            .map(|(domain, _)| log2_strict_usize(domain.size()) + params.log_blowup)
            .max()
        {
            input_mmcs
                .validate_commitment_cap_for_log_height(comm, max_log_height)
                .map_err(FriError::InputError)?;
        }
    }

    let expected_rounds = proof.commit_phase_commits.len();
    if proof.commit_phase_openings.len() != expected_rounds {
        return Err(FriError::CommitPhaseOpeningsCountMismatch {
            expected: expected_rounds,
            got: proof.commit_phase_openings.len(),
        });
    }

    let mut total_log_reduction = 0usize;
    let mut log_arities = Vec::with_capacity(expected_rounds);
    for (round, opening) in proof.commit_phase_openings.iter().enumerate() {
        let log_arity = usize::from(opening.log_arity);
        if !(1..=params.max_log_arity).contains(&log_arity) {
            return Err(FriError::InvalidLogArity {
                round,
                log_arity,
                max: params.max_log_arity,
            });
        }
        total_log_reduction = total_log_reduction.checked_add(log_arity).ok_or(
            FriError::GlobalMaxHeightTooLarge {
                log_global_max_height: usize::MAX,
                two_adicity: Val::TWO_ADICITY,
            },
        )?;
        log_arities.push(log_arity);
    }

    let log_global_max_height = total_log_reduction
        .checked_add(params.log_blowup)
        .and_then(|v| v.checked_add(params.log_final_poly_len))
        .ok_or(FriError::GlobalMaxHeightTooLarge {
            log_global_max_height: usize::MAX,
            two_adicity: Val::TWO_ADICITY,
        })?;
    if log_global_max_height > Val::TWO_ADICITY {
        return Err(FriError::GlobalMaxHeightTooLarge {
            log_global_max_height,
            two_adicity: Val::TWO_ADICITY,
        });
    }

    let expected_log_global_max_height = commitments_with_opening_points
        .iter()
        .flat_map(|(_, mats)| {
            mats.iter()
                .map(|(domain, _)| log2_strict_usize(domain.size()) + params.log_blowup)
        })
        .max();
    if let Some(expected) = expected_log_global_max_height
        && log_global_max_height != expected
    {
        return Err(FriError::GlobalMaxHeightMismatch {
            expected,
            got: log_global_max_height,
        });
    }

    let mut log_current_height = log_global_max_height;
    for (comm, &log_arity) in proof.commit_phase_commits.iter().zip(&log_arities) {
        let log_folded_height = log_current_height - log_arity;
        params
            .mmcs
            .validate_commitment_cap_for_log_height(comm, log_folded_height)
            .map_err(FriError::CommitPhaseMmcsError)?;
        log_current_height = log_folded_height;
    }

    Ok(())
}
