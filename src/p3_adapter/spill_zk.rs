//! CPU hiding spillable MMCS and 2× interleaved odd-coset FRI PCS for
//! Protocol `v6-zk`.
//!
//! - Leaf salting uses zero salt-matrix storage: each commitment draws a single
//!   256-bit OS CSPRNG `salt_seed` (or the deterministic public seed for
//!   preprocessed setup commitments) and derives 128-bit leaf salts on the fly
//!   via BLAKE3-CTR (`derive_leaf_salt(salt_seed, height, row_index)`) during
//!   both leaf hashing and query opening.
//! - Witness commits interleave uniform CSPRNG rows on the odd coset (`2×`
//!   domain extension), while preprocessed commits evaluate degree-`< N` public
//!   polynomials onto the `2N · B` LDE coset without adding random rows.
//! - Supports both in-memory multi-buffer SIMD BLAKE3 hashing and disk-spilled
//!   streaming commitments/openings with `FriBatching::{Tensor, Horner}` and
//!   exact `C2` `MerkleCap` root-count enforcement.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs, OpenedValues, Pcs, PeriodicLdeTable};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PrimeCharacteristicRing, TwoAdicField, dot_product,
};
use p3_fri::verifier::FriError;
use p3_fri::{
    BatchMultiOpening, CommitmentWithOpeningPoints, FriParameters, FriProof,
    ProverDataWithOpeningPoints, TwoAdicFriFolding, TwoAdicFriFoldingForMmcs,
    build_periodic_lde_table_two_adic, prover,
};
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow};
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_merkle_tree::{MerkleTreeError, PrunedMerklePaths};
use p3_symmetric::MerkleCap;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use tracing::debug_span;

use super::blake3_simd;
use super::cap::{
    CapValidatingMmcs, validate_fri_commitment_caps, verify_merkle_cap_len_for_log_height,
};
use super::domain_blake3::Blake3Domain;
use super::fri_batching::{
    FriBatching, FriBatchingSchedule, pack_ext_weights, verify_fri_with_batching,
};
use super::spill::{
    MerkleDigests, STREAM_BLOCK_BYTES, SpillPackWriter, SpilledMatrix, block_rows_for_group,
    build_bitrev_coset, compress_and_inject_prehashed, compress_layer_binary,
    compute_inverse_denominators_from_dims, interpolate_spilled_coset, padded_len_binary,
    validate_reachable_heights_iter,
};
use super::zk::{
    HidingBatchProof, HidingMultiProof, interleave_random_rows, open_hiding_batch_salts,
    open_hiding_multi_salts, preprocessed_salt_seed, random_seed, verify_hiding_batch,
    verify_hiding_multi_batch,
};
use crate::traits::ExtensionOf;
use crate::types::Val;

type HidingCommitResult = (
    <HidingSpillableMmcs as Mmcs<Val>>::Commitment,
    HidingSpillableMmcsData<RowMajorMatrix<Val>>,
);

/// In-memory salted binary BLAKE3 Merkle tree storing only a 32-byte
/// `salt_seed` (zero salt-matrix memory overhead).
struct FastInMemoryHidingMerkleTree<M> {
    leaves: Vec<M>,
    digests: MerkleDigests,
    salt_seed: [u8; 32],
}

impl<M: Matrix<Val>> FastInMemoryHidingMerkleTree<M> {
    fn build(
        leaves: Vec<M>,
        leaf_domain: Blake3Domain,
        compress_domain: Blake3Domain,
        salt_seed: [u8; 32],
    ) -> Self {
        assert!(!leaves.is_empty(), "No matrices given?");
        let max_height = validate_reachable_heights_iter(leaves.iter().map(Matrix::height));

        let mut order: Vec<usize> = (0..leaves.len()).collect();
        order.sort_by_key(|&idx| Reverse(leaves[idx].height()));

        let tallest_end = order
            .iter()
            .take_while(|&&idx| leaves[idx].height() == max_height)
            .count();
        let tallest = &order[..tallest_end];
        let mut cursor = tallest_end;

        let tallest_refs: Vec<&M> = tallest.iter().map(|&idx| &leaves[idx]).collect();
        // SAFETY: `hash_matrix_rows_salted_uninit` initializes every element of `dst` (`0..max_height`).
        let first_layer = unsafe {
            blake3_simd::alloc_and_fill_digests(padded_len_binary(max_height), max_height, |dst| {
                blake3_simd::hash_matrix_rows_salted_uninit(
                    &tallest_refs,
                    max_height,
                    0,
                    &salt_seed,
                    dst,
                    leaf_domain,
                );
            })
        };

        let mut digest_layers = vec![first_layer];
        let mut arity_schedule = Vec::new();

        loop {
            let prev_layer = digest_layers.last().unwrap().as_slice();
            if prev_layer.len() <= 1 {
                break;
            }
            let step = 2usize;
            let raw_next = prev_layer.len() / step;
            let next_layer_len = raw_next.next_power_of_two();

            let inject_start = cursor;
            while cursor < order.len()
                && leaves[order[cursor]].height().next_power_of_two() == next_layer_len
            {
                cursor += 1;
            }
            let inject_group = &order[inject_start..cursor];

            let next_digests = if inject_group.is_empty() {
                compress_layer_binary(prev_layer, compress_domain)
            } else {
                let inject_refs: Vec<&M> = inject_group.iter().map(|&idx| &leaves[idx]).collect();
                compress_and_inject_in_memory_salted(
                    prev_layer,
                    &inject_refs,
                    &salt_seed,
                    leaf_domain,
                    compress_domain,
                )
            };
            arity_schedule.push(step);
            digest_layers.push(next_digests);
        }

        Self {
            leaves,
            digests: MerkleDigests {
                digest_layers,
                arity_schedule,
            },
            salt_seed,
        }
    }

    fn cap(&self, cap_height: usize) -> MerkleCap<Val, [u8; 32]> {
        self.digests.cap(cap_height)
    }

    fn max_height(&self) -> usize {
        self.leaves
            .iter()
            .map(Matrix::height)
            .max()
            .expect("No committed matrices?")
    }

    fn matrix_heights(&self) -> Vec<usize> {
        self.leaves.iter().map(Matrix::height).collect()
    }

    fn dimensions(&self) -> Vec<Dimensions> {
        self.leaves.iter().map(Matrix::dimensions).collect()
    }

    fn open_batch(&self, index: usize, cap_height: usize) -> (Vec<Vec<Val>>, HidingBatchProof) {
        let heights = self.matrix_heights();
        let max_height = self.max_height();
        assert!(
            index < max_height,
            "index {index} out of bounds for height {max_height}"
        );
        let log_max_height = log2_ceil_usize(max_height);
        let openings: Vec<Vec<Val>> = self
            .leaves
            .iter()
            .map(|matrix| {
                let log2_height = log2_ceil_usize(matrix.height());
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                matrix.row(reduced_index).unwrap().into_iter().collect()
            })
            .collect();
        let siblings = self.digests.open_single_proof(index, cap_height);
        let salts = open_hiding_batch_salts(&self.salt_seed, &heights, index, cap_height);
        (openings, HidingBatchProof { salts, siblings })
    }

    fn open_multi_batch(
        &self,
        indices: &[usize],
        cap_height: usize,
    ) -> (Vec<Vec<Vec<Val>>>, HidingMultiProof) {
        let heights = self.matrix_heights();
        let max_height = self.max_height();
        let log_max_height = log2_ceil_usize(max_height);
        let opened_values: Vec<Vec<Vec<Val>>> = indices
            .iter()
            .map(|&index| {
                assert!(
                    index < max_height,
                    "index {index} out of bounds for height {max_height}"
                );
                self.leaves
                    .iter()
                    .map(|matrix| {
                        let log2_height = log2_ceil_usize(matrix.height());
                        let bits_reduced = log_max_height - log2_height;
                        let reduced_index = index >> bits_reduced;
                        matrix.row(reduced_index).unwrap().into_iter().collect()
                    })
                    .collect()
            })
            .collect();
        let pruned = self.digests.open_multi_proof(indices, cap_height);
        let salts = open_hiding_multi_salts(&self.salt_seed, &heights, indices, cap_height);
        (opened_values, HidingMultiProof { salts, pruned })
    }
}

/// Disk-spilled salted binary BLAKE3 Merkle tree storing only a 32-byte
/// `salt_seed` (zero salt-matrix memory or disk overhead).
struct SpilledHidingMerkleTree {
    matrices: Vec<SpilledMatrix>,
    digests: MerkleDigests,
    salt_seed: [u8; 32],
    original_traces: Mutex<Vec<Option<SpilledMatrix>>>,
}

impl SpilledHidingMerkleTree {
    fn build_with_prehashed(
        matrices: Vec<SpilledMatrix>,
        original_traces: Vec<Option<SpilledMatrix>>,
        mut prehashed: Vec<Option<Vec<[u8; 32]>>>,
        leaf_domain: Blake3Domain,
        compress_domain: Blake3Domain,
        salt_seed: [u8; 32],
    ) -> Self {
        assert!(!matrices.is_empty(), "No matrices given?");
        let max_height =
            validate_reachable_heights_iter(matrices.iter().map(SpilledMatrix::height));

        let mut order: Vec<usize> = (0..matrices.len()).collect();
        order.sort_by_key(|&idx| Reverse(matrices[idx].height()));

        let tallest_end = order
            .iter()
            .take_while(|&&idx| matrices[idx].height() == max_height)
            .count();
        let tallest = &order[..tallest_end];
        let mut cursor = tallest_end;

        let first_layer = if let &[single_idx] = tallest
            && let Some(mut digests) = prehashed.get_mut(single_idx).and_then(Option::take)
        {
            debug_assert_eq!(digests.len(), max_height);
            digests.resize(padded_len_binary(max_height), [0u8; 32]);
            digests
        } else {
            hash_leaf_group_streaming_salted(
                &matrices,
                tallest,
                max_height,
                &salt_seed,
                leaf_domain,
            )
        };
        let mut digest_layers = vec![first_layer];
        let mut arity_schedule = Vec::new();

        loop {
            let prev_layer = digest_layers.last().unwrap().as_slice();
            if prev_layer.len() <= 1 {
                break;
            }
            let step = 2usize;
            let raw_next = prev_layer.len() / step;
            let next_layer_len = raw_next.next_power_of_two();

            let inject_start = cursor;
            while cursor < order.len()
                && matrices[order[cursor]].height().next_power_of_two() == next_layer_len
            {
                cursor += 1;
            }
            let inject_group = &order[inject_start..cursor];

            let next_digests = if inject_group.is_empty() {
                compress_layer_binary(prev_layer, compress_domain)
            } else if let &[single_idx] = inject_group
                && let Some(rows_digests) = prehashed.get_mut(single_idx).and_then(Option::take)
            {
                compress_and_inject_prehashed(prev_layer, &rows_digests, compress_domain)
            } else {
                compress_and_inject_streaming_salted(
                    prev_layer,
                    &matrices,
                    inject_group,
                    &salt_seed,
                    leaf_domain,
                    compress_domain,
                )
            };
            arity_schedule.push(step);
            digest_layers.push(next_digests);
        }

        Self {
            matrices,
            digests: MerkleDigests {
                digest_layers,
                arity_schedule,
            },
            salt_seed,
            original_traces: Mutex::new(original_traces),
        }
    }

    fn cap(&self, cap_height: usize) -> MerkleCap<Val, [u8; 32]> {
        self.digests.cap(cap_height)
    }

    fn max_height(&self) -> usize {
        self.matrices
            .iter()
            .map(SpilledMatrix::height)
            .max()
            .expect("No committed matrices?")
    }

    fn matrix_heights(&self) -> Vec<usize> {
        self.matrices.iter().map(SpilledMatrix::height).collect()
    }

    fn dimensions(&self) -> Vec<Dimensions> {
        self.matrices
            .iter()
            .map(SpilledMatrix::dimensions)
            .collect()
    }

    fn open_batch(&self, index: usize, cap_height: usize) -> (Vec<Vec<Val>>, HidingBatchProof) {
        let heights = self.matrix_heights();
        let max_height = self.max_height();
        assert!(
            index < max_height,
            "index {index} out of bounds for height {max_height}"
        );
        let log_max_height = log2_ceil_usize(max_height);
        let openings: Vec<Vec<Val>> = self
            .matrices
            .iter()
            .map(|matrix| {
                let log2_height = log2_ceil_usize(matrix.height());
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                matrix.read_row(reduced_index)
            })
            .collect();
        let siblings = self.digests.open_single_proof(index, cap_height);
        let salts = open_hiding_batch_salts(&self.salt_seed, &heights, index, cap_height);
        (openings, HidingBatchProof { salts, siblings })
    }

    fn open_multi_batch(
        &self,
        indices: &[usize],
        cap_height: usize,
    ) -> (Vec<Vec<Vec<Val>>>, HidingMultiProof) {
        let heights = self.matrix_heights();
        let max_height = self.max_height();
        let log_max_height = log2_ceil_usize(max_height);
        let opened_values: Vec<Vec<Vec<Val>>> = indices
            .iter()
            .map(|&index| {
                assert!(
                    index < max_height,
                    "index {index} out of bounds for height {max_height}"
                );
                self.matrices
                    .iter()
                    .map(|matrix| {
                        let log2_height = log2_ceil_usize(matrix.height());
                        let bits_reduced = log_max_height - log2_height;
                        let reduced_index = index >> bits_reduced;
                        matrix.read_row(reduced_index)
                    })
                    .collect()
            })
            .collect();
        let pruned = self.digests.open_multi_proof(indices, cap_height);
        let salts = open_hiding_multi_salts(&self.salt_seed, &heights, indices, cap_height);
        (opened_values, HidingMultiProof { salts, pruned })
    }

    fn spilled_original_trace(&self, idx: usize) -> Option<SpilledMatrix> {
        let guard = self
            .original_traces
            .lock()
            .expect("original_traces lock poisoned");
        guard.get(idx).and_then(Clone::clone)
    }

    fn release_original_traces(&self) {
        let mut guard = self
            .original_traces
            .lock()
            .expect("original_traces lock poisoned");
        guard.clear();
    }
}

fn hash_leaf_group_streaming_salted(
    matrices: &[SpilledMatrix],
    group: &[usize],
    max_height: usize,
    salt_seed: &[u8; 32],
    leaf_domain: Blake3Domain,
) -> Vec<[u8; 32]> {
    let padded = padded_len_binary(max_height);
    let block_rows = block_rows_for_group(matrices, group, max_height);
    // SAFETY: the loop covers `0..max_height` in contiguous chunks and initializes
    // each via `hash_raw_multi_rows_salted_uninit`.
    unsafe {
        blake3_simd::alloc_and_fill_digests(padded, max_height, |dst| {
            for start in (0..max_height).step_by(block_rows) {
                let n_rows = (max_height - start).min(block_rows);
                let blocks: Vec<(Vec<u8>, usize)> = group
                    .iter()
                    .map(|&idx| {
                        let m = &matrices[idx];
                        (m.read_raw_rows(start, n_rows), m.row_bytes())
                    })
                    .collect();
                blake3_simd::hash_raw_multi_rows_salted_uninit(
                    &blocks,
                    max_height,
                    start,
                    salt_seed,
                    &mut dst[start..start + n_rows],
                    leaf_domain,
                );
            }
        })
    }
}

fn compress_and_inject_in_memory_salted<M: Matrix<Val>>(
    prev_layer: &[[u8; 32]],
    inject_group: &[&M],
    salt_seed: &[u8; 32],
    leaf_domain: Blake3Domain,
    compress_domain: Blake3Domain,
) -> Vec<[u8; 32]> {
    let next_len = inject_group[0].height();
    let raw_next = prev_layer.len() / 2;
    let next_len_padded = padded_len_binary(raw_next);
    const BLOCK_ROWS: usize = 1 << 14;
    let mut row_digests = vec![[0u8; 32]; next_len.min(BLOCK_ROWS)];

    // SAFETY: initializes `dst[..next_len]` in contiguous chunks and (if
    // `raw_next > next_len`) `dst[next_len..raw_next]`, covering all of `0..raw_next`.
    unsafe {
        blake3_simd::alloc_and_fill_digests(next_len_padded, raw_next, |dst| {
            for start in (0..next_len).step_by(BLOCK_ROWS) {
                let n_rows = (next_len - start).min(BLOCK_ROWS);
                blake3_simd::hash_matrix_rows_salted(
                    inject_group,
                    next_len,
                    start,
                    salt_seed,
                    &mut row_digests[..n_rows],
                    leaf_domain,
                );
                blake3_simd::compress_and_inject_from_digests_uninit(
                    &prev_layer[2 * start..],
                    &row_digests[..n_rows],
                    &mut dst[start..start + n_rows],
                    compress_domain,
                );
            }

            if raw_next > next_len {
                let zeros = vec![[0u8; 32]; raw_next - next_len];
                blake3_simd::compress_and_inject_from_digests_uninit(
                    &prev_layer[2 * next_len..],
                    &zeros,
                    &mut dst[next_len..raw_next],
                    compress_domain,
                );
            }
        })
    }
}

fn compress_and_inject_streaming_salted(
    prev_layer: &[[u8; 32]],
    matrices: &[SpilledMatrix],
    inject_group: &[usize],
    salt_seed: &[u8; 32],
    leaf_domain: Blake3Domain,
    compress_domain: Blake3Domain,
) -> Vec<[u8; 32]> {
    let next_len = matrices[inject_group[0]].height();
    let raw_next = prev_layer.len() / 2;
    let next_len_padded = padded_len_binary(raw_next);
    let block_rows = block_rows_for_group(matrices, inject_group, next_len);
    let mut row_digests = vec![[0u8; 32]; block_rows];

    // SAFETY: initializes `dst[..next_len]` in contiguous chunks and (if
    // `raw_next > next_len`) `dst[next_len..raw_next]`, covering all of `0..raw_next`.
    unsafe {
        blake3_simd::alloc_and_fill_digests(next_len_padded, raw_next, |dst| {
            for start in (0..next_len).step_by(block_rows) {
                let n_rows = (next_len - start).min(block_rows);
                let blocks: Vec<(Vec<u8>, usize)> = inject_group
                    .iter()
                    .map(|&idx| {
                        let m = &matrices[idx];
                        (m.read_raw_rows(start, n_rows), m.row_bytes())
                    })
                    .collect();
                blake3_simd::hash_raw_multi_rows_salted(
                    &blocks,
                    next_len,
                    start,
                    salt_seed,
                    &mut row_digests[..n_rows],
                    leaf_domain,
                );
                blake3_simd::compress_and_inject_from_digests_uninit(
                    &prev_layer[2 * start..],
                    &row_digests[..n_rows],
                    &mut dst[start..start + n_rows],
                    compress_domain,
                );
            }

            if raw_next > next_len {
                let zeros = vec![[0u8; 32]; raw_next - next_len];
                blake3_simd::compress_and_inject_from_digests_uninit(
                    &prev_layer[2 * next_len..],
                    &zeros,
                    &mut dst[next_len..raw_next],
                    compress_domain,
                );
            }
        })
    }
}

/// Prover data for [`HidingSpillableMmcs`].
pub struct HidingSpillableMmcsData<M>(HidingSpillableMmcsDataInner<M>);

enum HidingSpillableMmcsDataInner<M> {
    InMemory {
        tree: FastInMemoryHidingMerkleTree<M>,
        retained_traces: Mutex<Vec<Option<RowMajorMatrix<Val>>>>,
    },
    Spilled {
        tree: SpilledHidingMerkleTree,
        materialized: Vec<OnceLock<M>>,
        materialize: fn(&SpilledMatrix) -> M,
    },
}

impl<M> HidingSpillableMmcsData<M> {
    pub const fn is_spilled(&self) -> bool {
        matches!(self.0, HidingSpillableMmcsDataInner::Spilled { .. })
    }

    pub(crate) fn spilled_original_trace(&self, idx: usize) -> Option<SpilledMatrix> {
        match &self.0 {
            HidingSpillableMmcsDataInner::InMemory { .. } => None,
            HidingSpillableMmcsDataInner::Spilled { tree, .. } => tree.spilled_original_trace(idx),
        }
    }

    pub(crate) fn take_in_memory_original_trace(&self, idx: usize) -> Option<RowMajorMatrix<Val>> {
        match &self.0 {
            HidingSpillableMmcsDataInner::InMemory {
                retained_traces, ..
            } => {
                let mut guard = retained_traces
                    .lock()
                    .expect("retained_traces lock poisoned");
                guard.get_mut(idx).and_then(Option::take)
            }
            HidingSpillableMmcsDataInner::Spilled { .. } => None,
        }
    }

    pub(crate) fn release_original_traces(&self) {
        match &self.0 {
            HidingSpillableMmcsDataInner::InMemory {
                retained_traces, ..
            } => {
                let mut guard = retained_traces
                    .lock()
                    .expect("retained_traces lock poisoned");
                guard.clear();
            }
            HidingSpillableMmcsDataInner::Spilled { tree, .. } => tree.release_original_traces(),
        }
    }
}

/// CPU hiding MMCS for Protocol `v6-zk` using domain-separated BLAKE3
/// (`ZkMerkleLeaf`, `ZkMerkleCompress`) and zero-storage on-the-fly BLAKE3-CTR
/// 128-bit leaf salts.
#[derive(Clone, Debug)]
pub struct HidingSpillableMmcs {
    cap_height: usize,
    leaf_domain: Blake3Domain,
    compress_domain: Blake3Domain,
}

impl HidingSpillableMmcs {
    /// Creates a Protocol `v6-zk` hiding MMCS.
    pub const fn new_v6_zk(cap_height: usize) -> Self {
        Self {
            cap_height,
            leaf_domain: Blake3Domain::ZkMerkleLeaf,
            compress_domain: Blake3Domain::ZkMerkleCompress,
        }
    }

    pub const fn cap_height(&self) -> usize {
        self.cap_height
    }

    pub const fn leaf_domain(&self) -> Blake3Domain {
        self.leaf_domain
    }

    pub const fn compress_domain(&self) -> Blake3Domain {
        self.compress_domain
    }

    pub(crate) fn commit_with_seed<M: Matrix<Val>>(
        &self,
        inputs: Vec<M>,
        salt_seed: [u8; 32],
    ) -> (<Self as Mmcs<Val>>::Commitment, HidingSpillableMmcsData<M>) {
        let tree = FastInMemoryHidingMerkleTree::build(
            inputs,
            self.leaf_domain,
            self.compress_domain,
            salt_seed,
        );
        let commitment = tree.cap(self.cap_height);
        (
            commitment,
            HidingSpillableMmcsData(HidingSpillableMmcsDataInner::InMemory {
                tree,
                retained_traces: Mutex::new(Vec::new()),
            }),
        )
    }

    pub(crate) fn commit_in_memory_with_originals(
        &self,
        ldes: Vec<RowMajorMatrix<Val>>,
        original_traces: Vec<Option<RowMajorMatrix<Val>>>,
        salt_seed: [u8; 32],
    ) -> HidingCommitResult {
        let tree = FastInMemoryHidingMerkleTree::build(
            ldes,
            self.leaf_domain,
            self.compress_domain,
            salt_seed,
        );
        let commitment = tree.cap(self.cap_height);
        (
            commitment,
            HidingSpillableMmcsData(HidingSpillableMmcsDataInner::InMemory {
                tree,
                retained_traces: Mutex::new(original_traces),
            }),
        )
    }

    pub(crate) fn commit_spilled_prehashed(
        &self,
        matrices: Vec<SpilledMatrix>,
        original_traces: Vec<Option<SpilledMatrix>>,
        prehashed: Vec<Option<Vec<[u8; 32]>>>,
        salt_seed: [u8; 32],
    ) -> HidingCommitResult {
        let count = matrices.len();
        let tree = SpilledHidingMerkleTree::build_with_prehashed(
            matrices,
            original_traces,
            prehashed,
            self.leaf_domain,
            self.compress_domain,
            salt_seed,
        );
        let cap = tree.cap(self.cap_height);
        let materialized = (0..count).map(|_| OnceLock::new()).collect();
        (
            cap,
            HidingSpillableMmcsData(HidingSpillableMmcsDataInner::Spilled {
                tree,
                materialized,
                materialize: SpilledMatrix::read_all,
            }),
        )
    }

    pub(crate) fn matrix_dimensions<M: Matrix<Val>>(
        &self,
        prover_data: &HidingSpillableMmcsData<M>,
    ) -> Vec<Dimensions> {
        match &prover_data.0 {
            HidingSpillableMmcsDataInner::InMemory { tree, .. } => tree.dimensions(),
            HidingSpillableMmcsDataInner::Spilled { tree, .. } => tree.dimensions(),
        }
    }
}

impl Mmcs<Val> for HidingSpillableMmcs {
    type ProverData<M> = HidingSpillableMmcsData<M>;
    type Commitment = MerkleCap<Val, [u8; 32]>;
    type Proof = HidingBatchProof;
    type MultiProof = HidingMultiProof;
    type Error = MerkleTreeError;

    fn commit<M: Matrix<Val>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
        self.commit_with_seed(inputs, random_seed())
    }

    fn open_batch<M: Matrix<Val>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<Val, Self> {
        match &prover_data.0 {
            HidingSpillableMmcsDataInner::InMemory { tree, .. } => {
                let (opened_values, opening_proof) = tree.open_batch(index, self.cap_height);
                BatchOpening::new(opened_values, opening_proof)
            }
            HidingSpillableMmcsDataInner::Spilled { tree, .. } => {
                let (opened_values, opening_proof) = tree.open_batch(index, self.cap_height);
                BatchOpening::new(opened_values, opening_proof)
            }
        }
    }

    fn get_matrices<'a, M: Matrix<Val>>(&self, prover_data: &'a Self::ProverData<M>) -> Vec<&'a M> {
        match &prover_data.0 {
            HidingSpillableMmcsDataInner::InMemory { tree, .. } => tree.leaves.iter().collect(),
            HidingSpillableMmcsDataInner::Spilled {
                tree,
                materialized,
                materialize,
            } => tree
                .matrices
                .iter()
                .zip(materialized)
                .map(|(spilled, slot)| slot.get_or_init(|| materialize(spilled)))
                .collect(),
        }
    }

    fn get_matrix_heights<M: Matrix<Val>>(&self, prover_data: &Self::ProverData<M>) -> Vec<usize> {
        match &prover_data.0 {
            HidingSpillableMmcsDataInner::InMemory { tree, .. } => tree.matrix_heights(),
            HidingSpillableMmcsDataInner::Spilled { tree, .. } => tree.matrix_heights(),
        }
    }

    fn get_max_height<M: Matrix<Val>>(&self, prover_data: &Self::ProverData<M>) -> usize {
        match &prover_data.0 {
            HidingSpillableMmcsDataInner::InMemory { tree, .. } => tree.max_height(),
            HidingSpillableMmcsDataInner::Spilled { tree, .. } => tree.max_height(),
        }
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, Val, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, opening_proof) = batch_opening.unpack();
        verify_hiding_batch(
            self.cap_height,
            self.leaf_domain,
            self.compress_domain,
            commit,
            dimensions,
            index,
            opened_values,
            opening_proof,
        )
    }

    fn open_multi_batch<M: Matrix<Val>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> (Vec<Vec<Vec<Val>>>, Self::MultiProof) {
        match &prover_data.0 {
            HidingSpillableMmcsDataInner::InMemory { tree, .. } => {
                tree.open_multi_batch(indices, self.cap_height)
            }
            HidingSpillableMmcsDataInner::Spilled { tree, .. } => {
                tree.open_multi_batch(indices, self.cap_height)
            }
        }
    }

    fn verify_multi_batch<R: AsRef<[Val]> + PartialEq>(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<R>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error> {
        verify_hiding_multi_batch(
            self.cap_height,
            self.leaf_domain,
            self.compress_domain,
            commit,
            dimensions,
            indices,
            opened_values,
            proof,
        )
    }
}

impl CapValidatingMmcs<Val> for HidingSpillableMmcs {
    fn validate_commitment_cap_for_log_height(
        &self,
        commit: &Self::Commitment,
        log_max_height: usize,
    ) -> Result<(), Self::Error> {
        verify_merkle_cap_len_for_log_height(self.cap_height, commit, log_max_height)
    }

    fn empty_multi_proof() -> Self::MultiProof {
        HidingMultiProof {
            salts: Vec::new(),
            pruned: PrunedMerklePaths {
                sibling_hashes: Vec::new(),
            },
        }
    }
}

/// CPU 2× interleaved odd-coset hiding two-adic FRI PCS for Protocol `v6-zk`,
/// supporting both in-memory and disk-spilled execution with `FriBatching`.
#[derive(Clone, Debug)]
pub struct HidingSpillableTwoAdicFriPcs<Dft, FriMmcs> {
    dft: Dft,
    mmcs: HidingSpillableMmcs,
    fri: FriParameters<FriMmcs>,
    spill_dir: Option<PathBuf>,
    batching: FriBatching,
}

impl<Dft, FriMmcs> HidingSpillableTwoAdicFriPcs<Dft, FriMmcs> {
    pub const fn new(dft: Dft, mmcs: HidingSpillableMmcs, fri: FriParameters<FriMmcs>) -> Self {
        Self {
            dft,
            mmcs,
            fri,
            spill_dir: None,
            batching: FriBatching::Tensor,
        }
    }

    pub const fn with_batching(mut self, batching: FriBatching) -> Self {
        self.batching = batching;
        self
    }

    pub const fn batching(&self) -> FriBatching {
        self.batching
    }

    pub const fn leaf_domain(&self) -> Blake3Domain {
        self.mmcs.leaf_domain()
    }

    pub fn set_spill_dir(&mut self, spill_dir: Option<PathBuf>) {
        self.spill_dir = spill_dir;
    }

    pub fn spill_dir(&self) -> Option<&Path> {
        self.spill_dir.as_deref()
    }

    /// Computes the blinded `2N · B` LDE of a height-`N` witness matrix by
    /// interleaving odd-coset CSPRNG rows (height `2N`) and extending by
    /// `log_blowup`.
    pub(crate) fn compute_blinded_lde(
        &self,
        domain: TwoAdicMultiplicativeCoset<Val>,
        evals: &RowMajorMatrix<Val>,
    ) -> RowMajorMatrix<Val>
    where
        Dft: TwoAdicSubgroupDft<Val>,
    {
        assert_eq!(domain.size(), evals.height());
        let interleaved = interleave_random_rows(evals);
        let shift = Val::GENERATOR / domain.shift();
        self.dft
            .coset_lde_batch(interleaved, self.fri.log_blowup, shift)
            .bit_reverse_rows()
            .to_row_major_matrix()
    }

    /// Computes the public `2N · B` LDE of a height-`N` preprocessed matrix
    /// (degree `< N < 2N`, NO random rows) for uniform ZK verification.
    pub(crate) fn compute_preprocessed_lde(
        &self,
        domain: TwoAdicMultiplicativeCoset<Val>,
        evals: RowMajorMatrix<Val>,
    ) -> RowMajorMatrix<Val>
    where
        Dft: TwoAdicSubgroupDft<Val>,
    {
        assert_eq!(domain.size(), evals.height());
        let shift = Val::GENERATOR / domain.shift();
        self.dft
            .coset_lde_batch(evals, self.fri.log_blowup + 1, shift)
            .bit_reverse_rows()
            .to_row_major_matrix()
    }

    pub(crate) fn commit_spilled_ldes_prehashed(
        &self,
        spilled_ldes: Vec<SpilledMatrix>,
        original_traces: Option<Vec<Option<SpilledMatrix>>>,
        prehashed: Vec<Option<Vec<[u8; 32]>>>,
        salt_seed: [u8; 32],
    ) -> HidingCommitResult {
        let min_height = 1usize << (self.fri.log_blowup + 1);
        for lde in &spilled_ldes {
            assert!(
                lde.height() >= min_height,
                "committed ZK LDE height {} is smaller than 2 * blowup ({min_height})",
                lde.height()
            );
        }
        let originals = original_traces.unwrap_or_default();
        self.mmcs
            .commit_spilled_prehashed(spilled_ldes, originals, prehashed, salt_seed)
    }

    pub(crate) fn commit_in_memory_ldes(
        &self,
        ldes: Vec<RowMajorMatrix<Val>>,
    ) -> HidingCommitResult {
        let min_height = 1usize << (self.fri.log_blowup + 1);
        for lde in &ldes {
            assert!(
                lde.height() >= min_height,
                "committed ZK LDE height {} is smaller than 2 * blowup ({min_height})",
                lde.height()
            );
        }
        self.mmcs.commit(ldes)
    }

    /// Commits a streaming iterator of witness `(domain, matrix, retain_1x)`
    /// items with odd-coset random row interleaving and fresh CSPRNG leaf
    /// salting.
    pub(crate) fn commit_streaming_with_originals(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (TwoAdicMultiplicativeCoset<Val>, RowMajorMatrix<Val>, bool),
        >,
    ) -> Option<HidingCommitResult>
    where
        Dft: TwoAdicSubgroupDft<Val>,
    {
        self.commit_streaming_inner(evaluations, true, random_seed())
    }

    /// Commits a streaming iterator of public preprocessed `(domain, matrix, retain_1x)`
    /// items WITHOUT random rows and with the deterministic public preprocessed
    /// salt seed so independent prover and verifier setups produce identical
    /// `preprocessed_commit` roots.
    pub(crate) fn commit_preprocessed_streaming(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (TwoAdicMultiplicativeCoset<Val>, RowMajorMatrix<Val>, bool),
        >,
    ) -> Option<HidingCommitResult>
    where
        Dft: TwoAdicSubgroupDft<Val>,
    {
        self.commit_streaming_inner(evaluations, false, preprocessed_salt_seed())
    }

    fn commit_streaming_inner(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (TwoAdicMultiplicativeCoset<Val>, RowMajorMatrix<Val>, bool),
        >,
        blind_rows: bool,
        salt_seed: [u8; 32],
    ) -> Option<HidingCommitResult>
    where
        Dft: TwoAdicSubgroupDft<Val>,
    {
        if let Some(dir) = &self.spill_dir {
            let mut lde_pack: Option<SpillPackWriter> = None;
            let mut orig_pack: Option<SpillPackWriter> = None;
            let mut spilled_ldes = Vec::new();
            let mut spilled_originals = Vec::new();
            let mut prehashed: Vec<Option<Vec<[u8; 32]>>> = Vec::new();
            let mut first_idx_for_height: HashMap<usize, Option<usize>> = HashMap::new();

            for (domain, evals, retain_1x) in evaluations {
                if retain_1x {
                    let writer = orig_pack.get_or_insert_with(|| SpillPackWriter::create(dir));
                    spilled_originals.push(Some(writer.append_matrix(&evals)));
                } else {
                    spilled_originals.push(None);
                }
                let lde = if blind_rows {
                    self.compute_blinded_lde(domain, &evals)
                } else {
                    self.compute_preprocessed_lde(domain, evals)
                };
                let lde_height = lde.height();
                let idx = spilled_ldes.len();
                let writer = lde_pack.get_or_insert_with(|| SpillPackWriter::create(dir));
                match first_idx_for_height.get_mut(&lde_height) {
                    None => {
                        let (spilled, digests) = writer.append_matrix_and_hash_salted(
                            &lde,
                            self.mmcs.leaf_domain(),
                            &salt_seed,
                        );
                        spilled_ldes.push(spilled);
                        prehashed.push(Some(digests));
                        first_idx_for_height.insert(lde_height, Some(idx));
                    }
                    Some(prev) => {
                        if let Some(prev_idx) = prev.take() {
                            prehashed[prev_idx] = None;
                        }
                        spilled_ldes.push(writer.append_matrix(&lde));
                        prehashed.push(None);
                    }
                }
            }

            if spilled_ldes.is_empty() {
                return None;
            }
            if let Some(writer) = &mut lde_pack {
                writer.flush();
            }
            if let Some(writer) = &mut orig_pack {
                writer.flush();
            }
            let originals = if orig_pack.is_some() {
                Some(spilled_originals)
            } else {
                None
            };
            Some(self.commit_spilled_ldes_prehashed(spilled_ldes, originals, prehashed, salt_seed))
        } else {
            let mut ldes = Vec::new();
            let mut originals = Vec::new();
            let mut any_retained = false;
            for (domain, evals, retain_1x) in evaluations {
                if retain_1x {
                    any_retained = true;
                    originals.push(Some(evals.clone()));
                } else {
                    originals.push(None);
                }
                ldes.push(if blind_rows {
                    self.compute_blinded_lde(domain, &evals)
                } else {
                    self.compute_preprocessed_lde(domain, evals)
                });
            }
            if ldes.is_empty() {
                return None;
            }
            if any_retained {
                Some(
                    self.mmcs
                        .commit_in_memory_with_originals(ldes, originals, salt_seed),
                )
            } else {
                Some(self.mmcs.commit_with_seed(ldes, salt_seed))
            }
        }
    }
}

impl<EF, Dft, FriMmcs, Challenger> Pcs<EF, Challenger>
    for HidingSpillableTwoAdicFriPcs<Dft, FriMmcs>
where
    EF: ExtensionField<Val> + ExtensionOf<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    FriMmcs: CapValidatingMmcs<EF>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = <HidingSpillableMmcs as Mmcs<Val>>::Commitment;
    type ProverData = <HidingSpillableMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixCow<'a, Val>>;
    type Proof = FriProof<EF, FriMmcs, Val, Vec<BatchMultiOpening<Val, HidingSpillableMmcs>>>;
    type Error = FriError<FriMmcs::Error, <HidingSpillableMmcs as Mmcs<Val>>::Error>;
    const ZK: bool = true;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        TwoAdicMultiplicativeCoset::new(Val::ONE, log2_strict_usize(degree)).unwrap()
    }

    fn log_max_lde_height(&self) -> usize {
        Val::TWO_ADICITY
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let mut iter = evaluations.into_iter().map(|(d, m)| (d, m, false));
        self.commit_streaming_with_originals(&mut iter)
            .expect("commit requires at least one matrix")
    }

    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        _num_chunks: usize,
    ) -> Vec<RowMajorMatrix<Val>> {
        evaluations
            .into_iter()
            .map(|(domain, evals)| self.compute_blinded_lde(domain, &evals))
            .collect()
    }

    fn commit_ldes(&self, ldes: Vec<RowMajorMatrix<Val>>) -> (Self::Commitment, Self::ProverData) {
        let min_height = 1usize << (self.fri.log_blowup + 1);
        for lde in &ldes {
            assert!(
                lde.height() >= min_height,
                "committed ZK LDE height {} is smaller than 2 * blowup ({min_height})",
                lde.height()
            );
        }
        let salt_seed = random_seed();
        if let Some(dir) = &self.spill_dir {
            let mut height_counts: HashMap<usize, usize> = HashMap::new();
            for lde in &ldes {
                *height_counts.entry(lde.height()).or_insert(0) += 1;
            }
            let mut pack = SpillPackWriter::create(dir);
            let mut spilled = Vec::with_capacity(ldes.len());
            let mut prehashed = Vec::with_capacity(ldes.len());
            for lde in ldes {
                if height_counts[&lde.height()] == 1 {
                    let (sp, digests) = pack.append_matrix_and_hash_salted(
                        &lde,
                        self.mmcs.leaf_domain(),
                        &salt_seed,
                    );
                    spilled.push(sp);
                    prehashed.push(Some(digests));
                } else {
                    spilled.push(pack.append_matrix(&lde));
                    prehashed.push(None);
                }
            }
            pack.flush();
            self.mmcs
                .commit_spilled_prehashed(spilled, Vec::new(), prehashed, salt_seed)
        } else {
            self.mmcs.commit_with_seed(ldes, salt_seed)
        }
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        match &prover_data.0 {
            HidingSpillableMmcsDataInner::InMemory { tree, .. } => {
                let lde = &tree.leaves[idx];
                if domain.shift() == Val::GENERATOR && lde.height() >= domain.size() {
                    return lde.split_rows(domain.size()).0.as_cow().bit_reverse_rows();
                }
                let poly_height = lde.height() >> self.fri.log_blowup;
                let lde_mat = lde.as_view().bit_reverse_rows().to_row_major_matrix();
                self.re_evaluate_from_lde(lde_mat, poly_height, domain)
            }
            HidingSpillableMmcsDataInner::Spilled { tree, .. } => {
                let spilled = &tree.matrices[idx];
                if domain.shift() == Val::GENERATOR && spilled.height() >= domain.size() {
                    let prefix = spilled.read_rows(0, domain.size());
                    let width = prefix.width();
                    return RowMajorMatrixCow::new(Cow::Owned(prefix.values), width)
                        .bit_reverse_rows();
                }
                if domain.shift() == Val::ONE
                    && let Some(orig) = tree.spilled_original_trace(idx)
                    && orig.height() == domain.size()
                {
                    let natural = orig.read_all();
                    let bitrev = natural.as_view().bit_reverse_rows().to_row_major_matrix();
                    let width = bitrev.width();
                    return RowMajorMatrixCow::new(Cow::Owned(bitrev.values), width)
                        .bit_reverse_rows();
                }
                let poly_height = spilled.height() >> self.fri.log_blowup;
                let lde_mat = spilled
                    .read_all()
                    .as_view()
                    .bit_reverse_rows()
                    .to_row_major_matrix();
                self.re_evaluate_from_lde(lde_mat, poly_height, domain)
            }
        }
    }

    fn open(
        &self,
        commitment_data_with_opening_points: Vec<
            ProverDataWithOpeningPoints<'_, EF, Self::ProverData>,
        >,
        challenger: &mut Challenger,
    ) -> (OpenedValues<EF>, Self::Proof) {
        let dims_and_points: Vec<(Vec<Dimensions>, &Vec<Vec<EF>>)> =
            commitment_data_with_opening_points
                .iter()
                .map(|(data, points)| {
                    let dims = self.mmcs.matrix_dimensions(data);
                    debug_assert_eq!(
                        dims.len(),
                        points.len(),
                        "each matrix should have a corresponding set of evaluation points"
                    );
                    (dims, points)
                })
                .collect();

        let global_max_height = dims_and_points
            .iter()
            .flat_map(|(dims, _)| dims.iter().map(|d| d.height))
            .max()
            .expect("No Matrices Supplied?");
        let mut opened_max_width = 0usize;
        let mut openings_per_height = [0usize; 33];
        for (dims, points) in &dims_and_points {
            for (dim, pts) in dims.iter().zip(*points) {
                if !pts.is_empty() {
                    opened_max_width = opened_max_width.max(dim.width);
                    let lh = log2_strict_usize(dim.height);
                    openings_per_height[lh] += pts.len();
                }
            }
        }
        let max_openings_per_height = openings_per_height.into_iter().max().unwrap_or(0);
        let log_global_max_height = log2_strict_usize(global_max_height);

        let coset = build_bitrev_coset(log_global_max_height);
        let inv_denoms = compute_inverse_denominators_from_dims(&dims_and_points, &coset);
        drop(coset);

        let adjusted_weights: LinearMap<EF, Vec<EF>> = inv_denoms
            .iter()
            .map(|(point, denoms)| (*point, compute_adjusted_weights(*point, denoms)))
            .collect();

        // Pass 1: Barycentric interpolation over the low coset of size `h = 2N`
        // (degree `< 2N` blinded polynomial).
        let all_opened_values: OpenedValues<EF> = commitment_data_with_opening_points
            .iter()
            .zip(&dims_and_points)
            .map(|((data, points), (dims, _))| {
                points
                    .iter()
                    .enumerate()
                    .map(|(mat_idx, points_for_mat)| {
                        let dim = dims[mat_idx];
                        let h = dim.height >> self.fri.log_blowup;
                        if points_for_mat.is_empty() {
                            return Vec::new();
                        }
                        match &data.0 {
                            HidingSpillableMmcsDataInner::InMemory { tree, .. } => {
                                let mat = &tree.leaves[mat_idx];
                                let mat_view = mat.as_view();
                                let (low_coset, _) = mat_view.split_rows(h);
                                points_for_mat
                                    .iter()
                                    .map(|&point| {
                                        let _guard = debug_span!(
                                            "evaluate matrix",
                                            dims = %mat.dimensions()
                                        )
                                        .entered();
                                        let adj = &adjusted_weights.get(&point).unwrap()[..h];
                                        let ys = low_coset.interpolate_coset_with_precomputation(
                                            Val::GENERATOR,
                                            point,
                                            adj,
                                        );
                                        challenger.observe_algebra_slice(&ys);
                                        ys
                                    })
                                    .collect()
                            }
                            HidingSpillableMmcsDataInner::Spilled { tree, .. } => {
                                let spilled = &tree.matrices[mat_idx];
                                let _guard = debug_span!(
                                    "evaluate spilled matrix",
                                    dims = %dim
                                )
                                .entered();
                                let evals_per_point = interpolate_spilled_coset(
                                    spilled,
                                    h,
                                    points_for_mat,
                                    &adjusted_weights,
                                );
                                for ys in &evals_per_point {
                                    challenger.observe_algebra_slice(ys);
                                }
                                evals_per_point
                            }
                        }
                    })
                    .collect()
            })
            .collect();

        drop(adjusted_weights);

        let schedule = FriBatchingSchedule::<EF>::sample(
            self.batching,
            opened_max_width,
            max_openings_per_height,
            challenger,
        );
        let packed_col_weights: Vec<EF::ExtensionPacking> = match &schedule {
            FriBatchingSchedule::Horner { alpha } => {
                EF::ExtensionPacking::packed_ext_powers_capped(*alpha, opened_max_width).collect()
            }
            FriBatchingSchedule::Tensor { col_weights, .. } => {
                pack_ext_weights::<Val, EF>(col_weights)
            }
        };

        // Pass 2: Build reduced-opening codewords.
        let mut num_reduced = [0usize; 33];
        let mut opening_idx_by_height = [0usize; 33];
        let mut reduced_openings: [Option<Vec<EF>>; 33] = core::array::from_fn(|_| None);

        for (((data, points), (dims, _)), openings_for_round) in commitment_data_with_opening_points
            .iter()
            .zip(&dims_and_points)
            .zip(&all_opened_values)
        {
            for (mat_idx, (points_for_mat, openings_for_mat)) in
                points.iter().zip(openings_for_round).enumerate()
            {
                if points_for_mat.is_empty() || openings_for_mat.is_empty() {
                    continue;
                }
                let dim = dims[mat_idx];
                let log_height = log2_strict_usize(dim.height);
                let _guard = debug_span!("reduce matrix quotient", dims = %dim).entered();

                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| <EF as PrimeCharacteristicRing>::zero_vec(dim.height));
                debug_assert_eq!(reduced_opening_for_log_height.len(), dim.height);

                let mut point_params = Vec::with_capacity(points_for_mat.len());
                for (&point, openings) in points_for_mat.iter().zip(openings_for_mat) {
                    let (opening_weight, reduced_val) = match &schedule {
                        FriBatchingSchedule::Horner { alpha } => {
                            let w_open = PrimeCharacteristicRing::exp_u64(
                                alpha,
                                u64::try_from(num_reduced[log_height])
                                    .expect("reduced column count exceeds u64"),
                            );
                            let rv: EF = dot_product(
                                PrimeCharacteristicRing::powers(alpha),
                                openings.iter().copied(),
                            );
                            num_reduced[log_height] += dim.width;
                            (w_open, rv)
                        }
                        FriBatchingSchedule::Tensor {
                            col_weights,
                            open_weights,
                        } => {
                            let w_open = open_weights[opening_idx_by_height[log_height]];
                            opening_idx_by_height[log_height] += 1;
                            let rv: EF =
                                dot_product(col_weights.iter().copied(), openings.iter().copied());
                            (w_open, rv)
                        }
                    };
                    let inv_denom = inv_denoms.get(&point).unwrap().as_slice();
                    point_params.push((opening_weight, reduced_val, inv_denom));
                }

                match &data.0 {
                    HidingSpillableMmcsDataInner::InMemory { tree, .. } => {
                        let mat = &tree.leaves[mat_idx];
                        let mat_compressed: Vec<EF> = mat
                            .as_view()
                            .rowwise_packed_dot_product::<EF>(&packed_col_weights)
                            .collect();
                        for &(opening_weight, reduced_val, inv_denom) in &point_params {
                            mat_compressed
                                .par_iter()
                                .zip(reduced_opening_for_log_height.par_iter_mut())
                                .zip(inv_denom.par_iter())
                                .for_each(|((&reduced_row, ro), &inv_d)| {
                                    *ro += opening_weight * (reduced_val - reduced_row) * inv_d;
                                });
                        }
                    }
                    HidingSpillableMmcsDataInner::Spilled { tree, .. } => {
                        let spilled = &tree.matrices[mat_idx];
                        let row_bytes = spilled.row_bytes().max(size_of::<u64>());
                        let chunk_rows = (STREAM_BLOCK_BYTES / row_bytes)
                            .clamp(1, 1 << 14)
                            .min(dim.height);
                        for row_start in (0..dim.height).step_by(chunk_rows) {
                            let n_rows = (dim.height - row_start).min(chunk_rows);
                            let row_end = row_start + n_rows;
                            let chunk = spilled.read_rows(row_start, n_rows);
                            let mat_compressed: Vec<EF> = chunk
                                .as_view()
                                .rowwise_packed_dot_product::<EF>(&packed_col_weights)
                                .collect();
                            drop(chunk);
                            for &(opening_weight, reduced_val, inv_denom) in &point_params {
                                mat_compressed
                                    .par_iter()
                                    .zip(
                                        reduced_opening_for_log_height[row_start..row_end]
                                            .par_iter_mut(),
                                    )
                                    .zip(inv_denom[row_start..row_end].par_iter())
                                    .for_each(|((&reduced_row, ro), &inv_d)| {
                                        *ro += opening_weight * (reduced_val - reduced_row) * inv_d;
                                    });
                            }
                        }
                    }
                }
            }
        }

        drop(inv_denoms);
        drop(packed_col_weights);

        let fri_input: Vec<Vec<EF>> = reduced_openings.into_iter().rev().flatten().collect();
        let folding: TwoAdicFriFoldingForMmcs<Val, HidingSpillableMmcs> =
            TwoAdicFriFolding(PhantomData);
        let fri_proof = prover::prove_fri(
            &folding,
            &self.fri,
            fri_input,
            challenger,
            log_global_max_height,
            &commitment_data_with_opening_points,
            &self.mmcs,
        );

        (all_opened_values, fri_proof)
    }

    fn verify(
        &self,
        commitments_with_opening_points: Vec<
            CommitmentWithOpeningPoints<EF, Self::Commitment, Self::Domain>,
        >,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // Every committed polynomial in `v6-zk` has degree `< 2 * domain.size()`
        // and LDE height `2 * domain.size() * 2^log_blowup`. Double each domain
        // before cap validation and FRI verification.
        let mut doubled_commitments = Vec::with_capacity(commitments_with_opening_points.len());
        for (comm, mats) in commitments_with_opening_points {
            let mut doubled_mats = Vec::with_capacity(mats.len());
            for (domain, pts) in mats {
                let ext_log_size = domain.log_size() + 1;
                let ext_domain = TwoAdicMultiplicativeCoset::new(domain.shift(), ext_log_size)
                    .ok_or(FriError::GlobalMaxHeightTooLarge {
                        log_global_max_height: ext_log_size + self.fri.log_blowup,
                        two_adicity: Val::TWO_ADICITY,
                    })?;
                doubled_mats.push((ext_domain, pts));
            }
            doubled_commitments.push((comm, doubled_mats));
        }

        validate_fri_commitment_caps(&self.fri, proof, &doubled_commitments, &self.mmcs)?;
        for (_, round) in &doubled_commitments {
            for (_, mat) in round {
                for (_, point) in mat {
                    challenger.observe_algebra_slice(point);
                }
            }
        }
        let folding: TwoAdicFriFoldingForMmcs<Val, HidingSpillableMmcs> =
            TwoAdicFriFolding(PhantomData);
        verify_fri_with_batching(
            &folding,
            &self.fri,
            proof,
            challenger,
            &doubled_commitments,
            &self.mmcs,
            self.batching,
        )?;
        Ok(())
    }

    fn build_periodic_lde_table(
        &self,
        periodic_cols: &[Vec<Val>],
        trace_domain: Self::Domain,
        quotient_domain: Self::Domain,
    ) -> PeriodicLdeTable<Val> {
        build_periodic_lde_table_two_adic(&self.dft, periodic_cols, &trace_domain, &quotient_domain)
    }
}

impl<Dft: TwoAdicSubgroupDft<Val>, FriMmcs> HidingSpillableTwoAdicFriPcs<Dft, FriMmcs> {
    fn re_evaluate_from_lde<'a>(
        &self,
        lde_mat: RowMajorMatrix<Val>,
        poly_height: usize,
        domain: TwoAdicMultiplicativeCoset<Val>,
    ) -> BitReversedMatrixView<RowMajorMatrixCow<'a, Val>> {
        let mut coeffs = self.dft.coset_idft_batch(lde_mat, Val::GENERATOR);
        let width = coeffs.width();
        coeffs.values.truncate(poly_height * width);
        let target_size = domain.size();
        if target_size >= poly_height {
            coeffs.values.resize(target_size * width, Val::ZERO);
        } else {
            // Fold polynomial modulo `(X^m - shift^m)` so evaluating a degree
            // `< poly_height` polynomial on a smaller coset `shift * H_m` (e.g.
            // evaluating a degree-`< 2N` blinded trace on `H_N`) is exact.
            debug_assert!(poly_height.is_multiple_of(target_size));
            let blocks = poly_height / target_size;
            let shift_pow_m = domain
                .shift()
                .exp_u64(u64::try_from(target_size).expect("domain size exceeds u64"));
            let mut factor = shift_pow_m;
            for b in 1..blocks {
                let src_off = b * target_size * width;
                for i in 0..(target_size * width) {
                    let v = coeffs.values[src_off + i];
                    coeffs.values[i] += factor * v;
                }
                factor *= shift_pow_m;
            }
            coeffs.values.truncate(target_size * width);
        }
        let result = self
            .dft
            .coset_dft_batch(coeffs, domain.shift())
            .bit_reverse_rows()
            .to_row_major_matrix();
        let result_width = result.width();
        RowMajorMatrixCow::new(Cow::Owned(result.values), result_width).bit_reverse_rows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p3_adapter::zk::csprng_field_vec;
    use std::fs;

    fn temp_spill_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "multi-stark-zk-spill-test-{tag}-{}-{nanos}",
            std::process::id()
        ))
    }

    #[test]
    fn hiding_mmcs_in_memory_and_spilled_match_and_enforce_soundness() {
        let mats = vec![
            RowMajorMatrix::new(csprng_field_vec(64 * 5), 5),
            RowMajorMatrix::new(csprng_field_vec(16 * 3), 3),
            RowMajorMatrix::new(csprng_field_vec(64 * 2), 2),
            RowMajorMatrix::new(csprng_field_vec(4 * 7), 7),
        ];
        let dims: Vec<Dimensions> = mats.iter().map(Matrix::dimensions).collect();
        let queries = [63usize, 12, 0, 12, 41, 7, 13, 62];
        let fixed_seed = [0x5Au8; 32];

        for cap_height in [0usize, 1, 2] {
            let mmcs = HidingSpillableMmcs::new_v6_zk(cap_height);
            let (mem_com, mem_data) = mmcs.commit_with_seed(mats.clone(), fixed_seed);
            let (rand_com, _) = mmcs.commit(mats.clone());
            assert_ne!(
                mem_com.roots(),
                rand_com.roots(),
                "distinct CSPRNG salt seeds must yield distinct Merkle caps"
            );

            // Spilled path with same seed must produce bit-identical cap and proofs.
            let dir = temp_spill_dir(&format!("cap{cap_height}"));
            let mut pack = SpillPackWriter::create(&dir);
            let spilled_mats: Vec<_> = mats.iter().map(|m| pack.append_matrix(m)).collect();
            pack.flush();
            let (sp_com, sp_data) =
                mmcs.commit_spilled_prehashed(spilled_mats, Vec::new(), Vec::new(), fixed_seed);
            assert_eq!(mem_com.roots(), sp_com.roots());
            assert!(sp_data.is_spilled());

            for &idx in &queries {
                let b_mem = mmcs.open_batch(idx, &mem_data);
                let b_sp = mmcs.open_batch(idx, &sp_data);
                assert_eq!(b_mem.opened_values, b_sp.opened_values);
                assert_eq!(b_mem.opening_proof, b_sp.opening_proof);
                mmcs.verify_batch(
                    &mem_com,
                    &dims,
                    idx,
                    BatchOpeningRef::new(&b_mem.opened_values, &b_mem.opening_proof),
                )
                .unwrap();

                // Tampered salt must fail.
                let mut bad_proof = b_mem.opening_proof.clone();
                bad_proof.salts[0][0] ^= 1;
                assert!(matches!(
                    mmcs.verify_batch(
                        &mem_com,
                        &dims,
                        idx,
                        BatchOpeningRef::new(&b_mem.opened_values, &bad_proof),
                    ),
                    Err(MerkleTreeError::CapMismatch)
                ));
            }

            let (vals_mem, multi_mem) = mmcs.open_multi_batch(&queries, &mem_data);
            let (vals_sp, multi_sp) = mmcs.open_multi_batch(&queries, &sp_data);
            assert_eq!(vals_mem, vals_sp);
            assert_eq!(multi_mem, multi_sp);
            mmcs.verify_multi_batch(&mem_com, &dims, &queries, &vals_mem, &multi_mem)
                .unwrap();

            // Tampered multi-batch salt must fail.
            let mut bad_multi = multi_mem.clone();
            bad_multi.salts[0][0][0] ^= 0x80;
            assert!(matches!(
                mmcs.verify_multi_batch(&mem_com, &dims, &queries, &vals_mem, &bad_multi),
                Err(MerkleTreeError::CapMismatch)
            ));

            // Tampered leaf value must fail.
            let mut bad_vals = vals_mem.clone();
            bad_vals[0][0][0] += Val::ONE;
            assert!(
                mmcs.verify_multi_batch(&mem_com, &dims, &queries, &bad_vals, &multi_mem)
                    .is_err()
            );

            // C2: inflated MerkleCap root count must fail.
            let mut inflated_roots = mem_com.roots().to_vec();
            inflated_roots.extend_from_slice(mem_com.roots());
            let inflated = MerkleCap::new(inflated_roots);
            assert!(matches!(
                mmcs.verify_multi_batch(&inflated, &dims, &queries, &vals_mem, &multi_mem),
                Err(MerkleTreeError::CapMismatch)
            ));

            drop(sp_data);
            drop(pack);
            let _ = fs::remove_dir_all(&dir);
        }
    }
}
