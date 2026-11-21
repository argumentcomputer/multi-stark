//! Disk-backed spilling and streaming commitment/opening support for the CPU
//! two-adic FRI PCS.
//!
//! When disk spilling is enabled on [`crate::types::GoldilocksBlake3Config`],
//! committed 4× LDE leaf matrices are streamed to unlinked temporary files
//! instead of remaining resident in RAM across commitment, quotient evaluation,
//! and FRI opening. Compact Merkle `digest_layers` stay in memory; leaf hashing,
//! barycentric interpolation, reduced-opening compression, and Merkle query
//! openings stream bounded row blocks directly from disk.
//!
//! All operations produce commitments and proofs byte-identical to Plonky3's
//! in-memory [`MerkleTreeMmcs`] and [`p3_fri::TwoAdicFriPcs`].

use std::borrow::Cow;
use std::cmp::Reverse;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use p3_blake3::Blake3;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs, OpenedValues, Pcs, PeriodicLdeTable};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PrimeCharacteristicRing, PrimeField64,
    TwoAdicField, batch_multiplicative_inverse, dot_product, scale_slice_in_place_single_core,
};
use p3_fri::verifier::{self, FriError};
use p3_fri::{
    BatchMultiOpening, CommitmentWithOpeningPoints, FriParameters, FriProof,
    ProverDataWithOpeningPoints, TwoAdicFriFolding, TwoAdicFriFoldingForMmcs,
    build_periodic_lde_table_two_adic, prover,
};
use p3_goldilocks::Goldilocks;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow};
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_merkle_tree::{MerkleTreeMmcs, PrunedMerklePaths};
use p3_symmetric::{CryptographicHasher, MerkleCap, PseudoCompressionFunction, SerializingHasher};
use p3_util::linear_map::LinearMap;
use p3_util::{log2_ceil_usize, log2_strict_usize, reverse_slice_index_bits};
use tracing::debug_span;

use crate::types::{Blake3CompressionFunction, ExtVal, Val};

/// CPU reference MMCS used for in-memory commitments and verification.
pub type CpuMmcs =
    MerkleTreeMmcs<Val, u8, SerializingHasher<Blake3>, Blake3CompressionFunction, 2, 32>;
type CpuData<M> = <CpuMmcs as Mmcs<Val>>::ProverData<M>;
type SpillCommitResult = (
    <SpillableMmcs as Mmcs<Val>>::Commitment,
    SpillableMmcsData<RowMajorMatrix<Val>>,
);

/// Target memory budget per streamed row block during Merkle hashing and FRI
/// opening passes (64 MiB).
const STREAM_BLOCK_BYTES: usize = 64 << 20;

/// Write buffer size for sequential spill-file packing (1 MiB of field elements).
const SPILL_WRITE_CHUNK_ELEMS: usize = 1 << 17;

static SPILL_FILE_SEQ: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SPILL_FILES: AtomicUsize = AtomicUsize::new(0);
static TOTAL_SPILLED_BYTES: AtomicU64 = AtomicU64::new(0);

/// Number of currently open temporary spill files across this process.
pub fn active_spill_files() -> usize {
    ACTIVE_SPILL_FILES.load(Ordering::Relaxed)
}

/// Cumulative bytes written to spill files across this process.
pub fn total_spilled_bytes() -> u64 {
    TOTAL_SPILLED_BYTES.load(Ordering::Relaxed)
}

/// An owned temporary spill file. On Unix the directory entry is unlinked
/// immediately after creation so disk space is automatically reclaimed by the
/// kernel on drop, panic, or abnormal termination.
pub(crate) struct SpillFile {
    file: File,
    #[cfg(not(unix))]
    remove_on_drop: Option<PathBuf>,
    #[cfg(not(unix))]
    io_lock: Mutex<()>,
}

impl SpillFile {
    pub(crate) fn create(dir: &Path) -> Arc<Self> {
        fs::create_dir_all(dir)
            .unwrap_or_else(|err| panic!("failed to create spill directory {dir:?}: {err}"));
        let pid = std::process::id();
        loop {
            let seq = SPILL_FILE_SEQ.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!("multi-stark-spill-{pid}-{seq}.bin"));
            match OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&path)
            {
                Ok(file) => {
                    #[cfg(unix)]
                    {
                        // Unlink immediately so no stale files remain even on
                        // panic or SIGKILL.
                        let _ = fs::remove_file(&path);
                    }
                    ACTIVE_SPILL_FILES.fetch_add(1, Ordering::Relaxed);
                    return Arc::new(Self {
                        file,
                        #[cfg(not(unix))]
                        remove_on_drop: Some(path),
                        #[cfg(not(unix))]
                        io_lock: Mutex::new(()),
                    });
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(err) => {
                    panic!("failed to create spill file {path:?}: {err}");
                }
            }
        }
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) {
        if buf.is_empty() {
            return;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(buf, offset).unwrap_or_else(|err| {
                panic!("failed to read spill file at offset {offset}: {err}")
            });
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let _guard = self.io_lock.lock().expect("spill file lock poisoned");
            let mut handle = &self.file;
            handle
                .seek(SeekFrom::Start(offset))
                .unwrap_or_else(|err| panic!("failed to seek spill file to {offset}: {err}"));
            handle.read_exact(buf).unwrap_or_else(|err| {
                panic!("failed to read spill file at offset {offset}: {err}")
            });
        }
    }
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        ACTIVE_SPILL_FILES.fetch_sub(1, Ordering::Relaxed);
        #[cfg(not(unix))]
        if let Some(path) = self.remove_on_drop.take() {
            let _ = fs::remove_file(path);
        }
    }
}

/// Packs multiple spilled matrices sequentially into a single [`SpillFile`] so
/// even hundreds of circuits consume only one file descriptor per commitment
/// round.
pub(crate) struct SpillPackWriter {
    spill: Arc<SpillFile>,
    writer: BufWriter<File>,
    next_offset: u64,
    scratch: Vec<u8>,
}

impl SpillPackWriter {
    pub(crate) fn create(dir: &Path) -> Self {
        let spill = SpillFile::create(dir);
        let write_handle = spill
            .file
            .try_clone()
            .expect("failed to clone spill file handle for writing");
        Self {
            spill,
            writer: BufWriter::with_capacity(1 << 20, write_handle),
            next_offset: 0,
            scratch: Vec::with_capacity(SPILL_WRITE_CHUNK_ELEMS * size_of::<u64>()),
        }
    }

    /// Appends `mat` in canonical little-endian `u64` row-major order and
    /// returns a lightweight handle to its byte range in the pack file.
    pub(crate) fn append_matrix(&mut self, mat: &RowMajorMatrix<Val>) -> SpilledMatrix {
        let height = mat.height();
        let width = mat.width();
        let offset = self.next_offset;
        let total_elems = mat.values.len();
        for chunk in mat.values.chunks(SPILL_WRITE_CHUNK_ELEMS) {
            self.scratch.clear();
            for elem in chunk {
                self.scratch
                    .extend_from_slice(&elem.as_canonical_u64().to_le_bytes());
            }
            self.writer
                .write_all(&self.scratch)
                .expect("failed to write matrix chunk to spill file");
        }
        let bytes = u64::try_from(total_elems)
            .expect("matrix element count exceeds u64")
            .checked_mul(u64::try_from(size_of::<u64>()).unwrap())
            .expect("matrix byte length overflow");
        self.next_offset = self
            .next_offset
            .checked_add(bytes)
            .expect("spill pack offset overflow");
        TOTAL_SPILLED_BYTES.fetch_add(bytes, Ordering::Relaxed);
        SpilledMatrix {
            file: Arc::clone(&self.spill),
            offset,
            height,
            width,
        }
    }

    /// Flushes all buffered writes to the OS file before positional reads.
    pub(crate) fn flush(&mut self) {
        self.writer.flush().expect("failed to flush spill file");
    }
}

/// Handle to a row-major [`Goldilocks`] matrix stored in canonical little-endian
/// `u64` bytes inside a [`SpillFile`].
#[derive(Clone)]
pub(crate) struct SpilledMatrix {
    file: Arc<SpillFile>,
    offset: u64,
    height: usize,
    width: usize,
}

impl SpilledMatrix {
    #[inline]
    pub(crate) const fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub(crate) const fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub(crate) const fn dimensions(&self) -> Dimensions {
        Dimensions {
            width: self.width,
            height: self.height,
        }
    }

    #[inline]
    const fn row_bytes(&self) -> usize {
        self.width * size_of::<u64>()
    }

    /// Reads `num_rows` contiguous rows starting at `start_row` as raw
    /// canonical little-endian bytes (identical to `Goldilocks::into_byte_stream`).
    pub(crate) fn read_raw_rows(&self, start_row: usize, num_rows: usize) -> Vec<u8> {
        assert!(
            start_row.saturating_add(num_rows) <= self.height,
            "spilled read out of bounds: {start_row}..{} > {}",
            start_row.saturating_add(num_rows),
            self.height
        );
        let row_bytes = self.row_bytes();
        let byte_len = num_rows
            .checked_mul(row_bytes)
            .expect("spilled read byte length overflow");
        let byte_offset = self
            .offset
            .checked_add(
                u64::try_from(start_row)
                    .expect("start_row exceeds u64")
                    .checked_mul(u64::try_from(row_bytes).expect("row_bytes exceeds u64"))
                    .expect("spilled byte offset overflow"),
            )
            .expect("spilled offset overflow");
        let mut buf = vec![0u8; byte_len];
        self.file.read_exact_at(byte_offset, &mut buf);
        buf
    }

    /// Reads `num_rows` contiguous rows starting at `start_row` into a
    /// [`RowMajorMatrix`].
    pub(crate) fn read_rows(&self, start_row: usize, num_rows: usize) -> RowMajorMatrix<Val> {
        if self.width == 0 {
            return RowMajorMatrix::new(Vec::new(), 0);
        }
        let raw = self.read_raw_rows(start_row, num_rows);
        RowMajorMatrix::new(bytes_to_goldilocks(&raw), self.width)
    }

    /// Reads a single row (for FRI query openings).
    pub(crate) fn read_row(&self, row: usize) -> Vec<Val> {
        if self.width == 0 {
            return Vec::new();
        }
        let raw = self.read_raw_rows(row, 1);
        bytes_to_goldilocks(&raw)
    }

    /// Reads `num_rows` rows starting at `start_row` PLUS one cyclic halo row
    /// `(start_row + num_rows) % self.height`, returning a `(num_rows + 1)`-row
    /// matrix for chunked lookup evaluation.
    pub(crate) fn read_window_with_wrap(
        &self,
        start_row: usize,
        num_rows: usize,
    ) -> RowMajorMatrix<Val> {
        assert!(num_rows > 0 && start_row + num_rows <= self.height);
        if self.width == 0 {
            return RowMajorMatrix::new(Vec::new(), 0);
        }
        if start_row + num_rows < self.height {
            self.read_rows(start_row, num_rows + 1)
        } else {
            let mut mat = self.read_rows(start_row, num_rows);
            mat.values.extend_from_slice(&self.read_row(0));
            mat
        }
    }

    /// Reads the entire matrix back into RAM.
    pub(crate) fn read_all(&self) -> RowMajorMatrix<Val> {
        self.read_rows(0, self.height)
    }
}

#[inline]
fn bytes_to_goldilocks(raw: &[u8]) -> Vec<Val> {
    let (chunks, remainder) = raw.as_chunks::<8>();
    debug_assert!(remainder.is_empty());
    chunks
        .iter()
        .map(|&chunk| Goldilocks::new(u64::from_le_bytes(chunk)))
        .collect()
}

/// Binary Merkle-tree padded layer length (matching `p3_merkle_tree::padded_len(raw_len, 2)`).
#[inline]
const fn padded_len_binary(raw_len: usize) -> usize {
    if raw_len <= 1 {
        raw_len
    } else {
        raw_len.div_ceil(2) * 2
    }
}

/// Validates that every matrix height sits on the `ceil(max_height / 2^k)`
/// ladder anchored at the tallest matrix (matching `p3_merkle_tree`).
fn validate_reachable_heights(matrices: &[SpilledMatrix]) -> usize {
    let max_height = matrices
        .iter()
        .map(SpilledMatrix::height)
        .max()
        .filter(|&h| h > 0)
        .expect("all matrices have height 0");
    let log_max_height = log2_ceil_usize(max_height);
    for m in matrices {
        let h = m.height();
        let bits_reduced = log_max_height - log2_ceil_usize(h);
        let expected_height = ((max_height - 1) >> bits_reduced) + 1;
        assert_eq!(
            h, expected_height,
            "matrix height {h} incompatible with tallest height {max_height} (expected {expected_height})"
        );
    }
    max_height
}

/// A binary Blake3 Merkle tree whose leaf matrices reside on disk in
/// [`SpilledMatrix`] handles, while only the compact internal `digest_layers`
/// remain in RAM.
pub(crate) struct SpilledMerkleTree {
    matrices: Vec<SpilledMatrix>,
    digest_layers: Vec<Vec<[u8; 32]>>,
    arity_schedule: Vec<usize>,
    /// Optional spilled 1× original traces retained for zero-FFT on-demand
    /// lookup evaluation; released once stage-2 completes for stage-1 data,
    /// or retained for the lifetime of `ProverKey` for preprocessed data.
    original_traces: Mutex<Vec<Option<SpilledMatrix>>>,
}

impl SpilledMerkleTree {
    pub(crate) fn build(
        matrices: Vec<SpilledMatrix>,
        original_traces: Vec<Option<SpilledMatrix>>,
        compress: &Blake3CompressionFunction,
    ) -> Self {
        assert!(!matrices.is_empty(), "No matrices given?");
        let max_height = validate_reachable_heights(&matrices);

        // Stable sort tallest-first to match `MerkleTree::new` insertion order
        // within each height group.
        let mut order: Vec<usize> = (0..matrices.len()).collect();
        order.sort_by_key(|&idx| Reverse(matrices[idx].height()));

        let tallest_end = order
            .iter()
            .take_while(|&&idx| matrices[idx].height() == max_height)
            .count();
        let tallest = &order[..tallest_end];
        let mut cursor = tallest_end;

        let first_layer = hash_leaf_group_streaming(&matrices, tallest, max_height);
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
                compress_layer_binary(prev_layer, compress)
            } else {
                compress_and_inject_streaming(prev_layer, &matrices, inject_group, compress)
            };
            arity_schedule.push(step);
            digest_layers.push(next_digests);
        }

        Self {
            matrices,
            digest_layers,
            arity_schedule,
            original_traces: Mutex::new(original_traces),
        }
    }

    pub(crate) fn cap(&self, cap_height: usize) -> MerkleCap<Val, [u8; 32]> {
        let num_layers = self.digest_layers.len();
        let effective_cap_height = cap_height.min(num_layers.saturating_sub(1));
        let layer_idx = num_layers - 1 - effective_cap_height;
        let layer = &self.digest_layers[layer_idx];
        let cap_len: usize = self.arity_schedule[layer_idx..].iter().product();
        let cap_len = cap_len.min(layer.len());
        MerkleCap::new(layer[..cap_len].to_vec())
    }

    pub(crate) fn max_height(&self) -> usize {
        self.matrices
            .iter()
            .map(SpilledMatrix::height)
            .max()
            .expect("No committed matrices?")
    }

    pub(crate) fn matrix_heights(&self) -> Vec<usize> {
        self.matrices.iter().map(SpilledMatrix::height).collect()
    }

    pub(crate) fn dimensions(&self) -> Vec<Dimensions> {
        self.matrices
            .iter()
            .map(SpilledMatrix::dimensions)
            .collect()
    }

    pub(crate) fn open_batch(
        &self,
        index: usize,
        cap_height: usize,
    ) -> (Vec<Vec<Val>>, Vec<[u8; 32]>) {
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

        let num_layers = self.digest_layers.len();
        let effective_cap_height = cap_height.min(num_layers.saturating_sub(1));
        let proof_levels = num_layers
            .saturating_sub(1)
            .saturating_sub(effective_cap_height);

        let mut proof = Vec::with_capacity(proof_levels);
        let mut idx = index;
        for layer_idx in 0..proof_levels {
            let step = self.arity_schedule[layer_idx];
            let group_start = (idx / step) * step;
            let pos_in_group = idx % step;
            for k in 0..step {
                if k != pos_in_group {
                    proof.push(self.digest_layers[layer_idx][group_start + k]);
                }
            }
            idx /= step;
        }
        (openings, proof)
    }

    pub(crate) fn open_multi_batch(
        &self,
        indices: &[usize],
        cap_height: usize,
    ) -> (Vec<Vec<Vec<Val>>>, PrunedMerklePaths<u8, 32>) {
        let max_height = self.max_height();
        let log_max_height = log2_ceil_usize(max_height);
        let num_layers = self.digest_layers.len();
        let effective_cap_height = cap_height.min(num_layers.saturating_sub(1));
        let proof_levels = num_layers
            .saturating_sub(1)
            .saturating_sub(effective_cap_height);

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

        // Walk the pruned frontier bottom-up in the exact wire order defined by
        // `p3_merkle_tree::pruning::walk_frontier`:
        // - sorted unique leaf indices
        // - level 0 .. proof_levels
        // - ascending parent groups, emitting missing child positions ascending.
        let mut nodes: Vec<usize> = indices.to_vec();
        nodes.sort_unstable();
        nodes.dedup();

        let mut sibling_hashes = Vec::new();
        if !nodes.is_empty() {
            let mut parents: Vec<usize> = Vec::with_capacity(nodes.len());
            for (level, &arity) in self.arity_schedule[..proof_levels].iter().enumerate() {
                parents.clear();
                let mut i = 0;
                while i < nodes.len() {
                    let group = nodes[i] / arity;
                    let group_start = group * arity;
                    let mut member = i;
                    for k in 0..arity {
                        if member < nodes.len() && nodes[member] == group_start + k {
                            member += 1;
                        } else {
                            sibling_hashes.push(self.digest_layers[level][group_start + k]);
                        }
                    }
                    parents.push(group);
                    i = member;
                }
                core::mem::swap(&mut nodes, &mut parents);
            }
        }

        (opened_values, PrunedMerklePaths { sibling_hashes })
    }

    pub(crate) fn spilled_original_trace(&self, idx: usize) -> Option<SpilledMatrix> {
        let guard = self
            .original_traces
            .lock()
            .expect("original_traces lock poisoned");
        guard.get(idx).and_then(Clone::clone)
    }

    pub(crate) fn release_original_traces(&self) {
        let mut guard = self
            .original_traces
            .lock()
            .expect("original_traces lock poisoned");
        guard.clear();
    }
}

fn block_rows_for_group(matrices: &[SpilledMatrix], group: &[usize], height: usize) -> usize {
    let total_row_bytes: usize = group
        .iter()
        .map(|&idx| matrices[idx].row_bytes())
        .sum::<usize>()
        .max(size_of::<u64>());
    (STREAM_BLOCK_BYTES / total_row_bytes)
        .clamp(1, 1 << 14)
        .min(height.max(1))
}

fn hash_leaf_group_streaming(
    matrices: &[SpilledMatrix],
    group: &[usize],
    max_height: usize,
) -> Vec<[u8; 32]> {
    let padded = padded_len_binary(max_height);
    let mut digests = vec![[0u8; 32]; padded];
    let block_rows = block_rows_for_group(matrices, group, max_height);
    let hasher = Blake3;

    for start in (0..max_height).step_by(block_rows) {
        let n_rows = (max_height - start).min(block_rows);
        let blocks: Vec<(Vec<u8>, usize)> = group
            .iter()
            .map(|&idx| {
                let m = &matrices[idx];
                (m.read_raw_rows(start, n_rows), m.row_bytes())
            })
            .collect();
        digests[start..start + n_rows]
            .par_iter_mut()
            .enumerate()
            .for_each(|(local_row, digest)| {
                *digest = if let [(buf, row_bytes)] = blocks.as_slice() {
                    hasher.hash_slice(&buf[local_row * *row_bytes..(local_row + 1) * *row_bytes])
                } else {
                    hasher.hash_iter_slices(blocks.iter().map(|(buf, row_bytes)| {
                        &buf[local_row * *row_bytes..(local_row + 1) * *row_bytes]
                    }))
                };
            });
    }
    digests
}

fn compress_layer_binary(
    prev_layer: &[[u8; 32]],
    compress: &Blake3CompressionFunction,
) -> Vec<[u8; 32]> {
    let raw_next = prev_layer.len() / 2;
    let next_len_padded = padded_len_binary(raw_next);
    let mut next_digests = vec![[0u8; 32]; next_len_padded];
    next_digests[..raw_next]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, digest)| {
            *digest = compress.compress([prev_layer[2 * i], prev_layer[2 * i + 1]]);
        });
    next_digests
}

fn compress_and_inject_streaming(
    prev_layer: &[[u8; 32]],
    matrices: &[SpilledMatrix],
    inject_group: &[usize],
    compress: &Blake3CompressionFunction,
) -> Vec<[u8; 32]> {
    let next_len = matrices[inject_group[0]].height();
    let raw_next = prev_layer.len() / 2;
    let next_len_padded = padded_len_binary(raw_next);
    let mut next_digests = vec![[0u8; 32]; next_len_padded];
    let block_rows = block_rows_for_group(matrices, inject_group, next_len);
    let hasher = Blake3;

    for start in (0..next_len).step_by(block_rows) {
        let n_rows = (next_len - start).min(block_rows);
        let blocks: Vec<(Vec<u8>, usize)> = inject_group
            .iter()
            .map(|&idx| {
                let m = &matrices[idx];
                (m.read_raw_rows(start, n_rows), m.row_bytes())
            })
            .collect();
        next_digests[start..start + n_rows]
            .par_iter_mut()
            .enumerate()
            .for_each(|(local_row, out)| {
                let i = start + local_row;
                let digest = compress.compress([prev_layer[2 * i], prev_layer[2 * i + 1]]);
                let rows_digest = if let [(buf, row_bytes)] = blocks.as_slice() {
                    hasher.hash_slice(&buf[local_row * *row_bytes..(local_row + 1) * *row_bytes])
                } else {
                    hasher.hash_iter_slices(blocks.iter().map(|(buf, row_bytes)| {
                        &buf[local_row * *row_bytes..(local_row + 1) * *row_bytes]
                    }))
                };
                *out = compress.compress([digest, rows_digest]);
            });
    }

    for (i, out) in next_digests[next_len..raw_next].iter_mut().enumerate() {
        let idx = next_len + i;
        let digest = compress.compress([prev_layer[2 * idx], prev_layer[2 * idx + 1]]);
        *out = compress.compress([digest, [0u8; 32]]);
    }

    next_digests
}

/// Prover data for [`SpillableMmcs`]: either Plonky3's standard in-memory
/// [`p3_merkle_tree::MerkleTree`] or a disk-spilled [`SpilledMerkleTree`].
pub struct SpillableMmcsData<M>(SpillableMmcsDataInner<M>);

enum SpillableMmcsDataInner<M> {
    InMemory {
        tree: CpuData<M>,
        /// Optional 1× original traces retained for zero-FFT lazy lookup
        /// evaluation in in-memory mode; released once stage-2 completes.
        retained_traces: Mutex<Vec<Option<RowMajorMatrix<Val>>>>,
    },
    Spilled {
        tree: SpilledMerkleTree,
        /// Lazy fallback cache used ONLY if external code calls
        /// [`Mmcs::get_matrices`] directly on spilled prover data.
        materialized: Vec<OnceLock<M>>,
        materialize: fn(&SpilledMatrix) -> M,
    },
}

impl<M> SpillableMmcsData<M> {
    /// Returns `true` if the LDE leaf matrices are backed by temporary disk
    /// spill files rather than resident in memory.
    pub const fn is_spilled(&self) -> bool {
        matches!(self.0, SpillableMmcsDataInner::Spilled { .. })
    }

    /// Number of spilled matrices that were materialized back into memory via
    /// [`Mmcs::get_matrices`]. On the streaming prover path this stays 0.
    pub fn materialized_spilled_count(&self) -> usize {
        match &self.0 {
            SpillableMmcsDataInner::InMemory { .. } => 0,
            SpillableMmcsDataInner::Spilled { materialized, .. } => materialized
                .iter()
                .filter(|slot| slot.get().is_some())
                .count(),
        }
    }

    pub(crate) fn spilled_original_trace(&self, idx: usize) -> Option<SpilledMatrix> {
        match &self.0 {
            SpillableMmcsDataInner::InMemory { .. } => None,
            SpillableMmcsDataInner::Spilled { tree, .. } => tree.spilled_original_trace(idx),
        }
    }

    pub(crate) fn take_in_memory_original_trace(&self, idx: usize) -> Option<RowMajorMatrix<Val>> {
        match &self.0 {
            SpillableMmcsDataInner::InMemory {
                retained_traces, ..
            } => {
                let mut guard = retained_traces
                    .lock()
                    .expect("retained_traces lock poisoned");
                guard.get_mut(idx).and_then(Option::take)
            }
            SpillableMmcsDataInner::Spilled { .. } => None,
        }
    }

    pub(crate) fn release_original_traces(&self) {
        match &self.0 {
            SpillableMmcsDataInner::InMemory {
                retained_traces, ..
            } => {
                let mut guard = retained_traces
                    .lock()
                    .expect("retained_traces lock poisoned");
                guard.clear();
            }
            SpillableMmcsDataInner::Spilled { tree, .. } => tree.release_original_traces(),
        }
    }
}

/// CPU MMCS supporting both in-memory commitments (default, zero overhead) and
/// disk-spilled commitments with identical commitments and opening proofs.
#[derive(Clone, Debug)]
pub struct SpillableMmcs {
    cpu: CpuMmcs,
    compress: Blake3CompressionFunction,
}

impl SpillableMmcs {
    pub fn new(
        hash: SerializingHasher<Blake3>,
        compress: Blake3CompressionFunction,
        cap_height: usize,
    ) -> Self {
        Self {
            cpu: CpuMmcs::new(hash, compress.clone(), cap_height),
            compress,
        }
    }

    pub const fn cap_height(&self) -> usize {
        self.cpu.cap_height()
    }

    pub(crate) fn commit_in_memory_with_originals(
        &self,
        ldes: Vec<RowMajorMatrix<Val>>,
        original_traces: Vec<Option<RowMajorMatrix<Val>>>,
    ) -> SpillCommitResult {
        let (commitment, tree) = self.cpu.commit(ldes);
        (
            commitment,
            SpillableMmcsData(SpillableMmcsDataInner::InMemory {
                tree,
                retained_traces: Mutex::new(original_traces),
            }),
        )
    }

    pub(crate) fn commit_spilled(
        &self,
        matrices: Vec<SpilledMatrix>,
        original_traces: Vec<Option<SpilledMatrix>>,
    ) -> SpillCommitResult {
        let count = matrices.len();
        let tree = SpilledMerkleTree::build(matrices, original_traces, &self.compress);
        let cap = tree.cap(self.cap_height());
        let materialized = (0..count).map(|_| OnceLock::new()).collect();
        (
            cap,
            SpillableMmcsData(SpillableMmcsDataInner::Spilled {
                tree,
                materialized,
                materialize: SpilledMatrix::read_all,
            }),
        )
    }

    pub(crate) fn matrix_dimensions<M: Matrix<Val>>(
        &self,
        prover_data: &SpillableMmcsData<M>,
    ) -> Vec<Dimensions> {
        match &prover_data.0 {
            SpillableMmcsDataInner::InMemory { tree, .. } => self
                .cpu
                .get_matrices(tree)
                .into_iter()
                .map(Matrix::dimensions)
                .collect(),
            SpillableMmcsDataInner::Spilled { tree, .. } => tree.dimensions(),
        }
    }
}

impl Mmcs<Val> for SpillableMmcs {
    type ProverData<M> = SpillableMmcsData<M>;
    type Commitment = <CpuMmcs as Mmcs<Val>>::Commitment;
    type Proof = <CpuMmcs as Mmcs<Val>>::Proof;
    type MultiProof = <CpuMmcs as Mmcs<Val>>::MultiProof;
    type Error = <CpuMmcs as Mmcs<Val>>::Error;

    fn commit<M: Matrix<Val>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
        let (commitment, tree) = self.cpu.commit(inputs);
        (
            commitment,
            SpillableMmcsData(SpillableMmcsDataInner::InMemory {
                tree,
                retained_traces: Mutex::new(Vec::new()),
            }),
        )
    }

    fn open_batch<M: Matrix<Val>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<Val, Self> {
        match &prover_data.0 {
            SpillableMmcsDataInner::InMemory { tree, .. } => {
                let (opened_values, opening_proof) = self.cpu.open_batch(index, tree).unpack();
                BatchOpening::new(opened_values, opening_proof)
            }
            SpillableMmcsDataInner::Spilled { tree, .. } => {
                let (opened_values, opening_proof) = tree.open_batch(index, self.cap_height());
                BatchOpening::new(opened_values, opening_proof)
            }
        }
    }

    fn get_matrices<'a, M: Matrix<Val>>(&self, prover_data: &'a Self::ProverData<M>) -> Vec<&'a M> {
        match &prover_data.0 {
            SpillableMmcsDataInner::InMemory { tree, .. } => self.cpu.get_matrices(tree),
            SpillableMmcsDataInner::Spilled {
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
            SpillableMmcsDataInner::InMemory { tree, .. } => self.cpu.get_matrix_heights(tree),
            SpillableMmcsDataInner::Spilled { tree, .. } => tree.matrix_heights(),
        }
    }

    fn get_max_height<M: Matrix<Val>>(&self, prover_data: &Self::ProverData<M>) -> usize {
        match &prover_data.0 {
            SpillableMmcsDataInner::InMemory { tree, .. } => self.cpu.get_max_height(tree),
            SpillableMmcsDataInner::Spilled { tree, .. } => tree.max_height(),
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
        self.cpu.verify_batch(
            commit,
            dimensions,
            index,
            BatchOpeningRef::new(opened_values, opening_proof),
        )
    }

    fn open_multi_batch<M: Matrix<Val>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> (Vec<Vec<Vec<Val>>>, Self::MultiProof) {
        match &prover_data.0 {
            SpillableMmcsDataInner::InMemory { tree, .. } => {
                self.cpu.open_multi_batch(indices, tree)
            }
            SpillableMmcsDataInner::Spilled { tree, .. } => {
                tree.open_multi_batch(indices, self.cap_height())
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
        self.cpu
            .verify_multi_batch(commit, dimensions, indices, opened_values, proof)
    }
}

/// CPU two-adic FRI PCS supporting optional disk-backed LDE spilling with
/// streaming commitment, quotient-domain slicing, and FRI opening.
///
/// When `spill_dir` is `None`, all operations run in memory identically to
/// Plonky3's [`p3_fri::TwoAdicFriPcs`]. When `spill_dir` is `Some(dir)`, LDE
/// matrices are spilled one by one into `dir`, keeping resident memory bounded
/// to compact Merkle digest layers and a single streaming work chunk.
#[derive(Clone, Debug)]
pub struct SpillableTwoAdicFriPcs<Dft, FriMmcs> {
    dft: Dft,
    mmcs: SpillableMmcs,
    fri: FriParameters<FriMmcs>,
    spill_dir: Option<PathBuf>,
}

impl<Dft, FriMmcs> SpillableTwoAdicFriPcs<Dft, FriMmcs> {
    pub const fn new(dft: Dft, mmcs: SpillableMmcs, fri: FriParameters<FriMmcs>) -> Self {
        Self {
            dft,
            mmcs,
            fri,
            spill_dir: None,
        }
    }

    pub fn set_spill_dir(&mut self, spill_dir: Option<PathBuf>) {
        self.spill_dir = spill_dir;
    }

    pub fn spill_dir(&self) -> Option<&Path> {
        self.spill_dir.as_deref()
    }

    pub(crate) fn compute_lde(
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
            .coset_lde_batch(evals, self.fri.log_blowup, shift)
            .bit_reverse_rows()
            .to_row_major_matrix()
    }

    pub(crate) fn commit_spilled_ldes(
        &self,
        spilled_ldes: Vec<SpilledMatrix>,
        original_traces: Option<Vec<Option<SpilledMatrix>>>,
    ) -> SpillCommitResult {
        let min_height = 1usize << self.fri.log_blowup;
        for lde in &spilled_ldes {
            assert!(
                lde.height() >= min_height,
                "committed LDE height {} is smaller than the blowup factor {min_height}",
                lde.height()
            );
        }
        let originals = original_traces.unwrap_or_default();
        self.mmcs.commit_spilled(spilled_ldes, originals)
    }

    pub(crate) fn commit_in_memory_ldes(
        &self,
        ldes: Vec<RowMajorMatrix<Val>>,
    ) -> SpillCommitResult {
        let min_height = 1usize << self.fri.log_blowup;
        for lde in &ldes {
            assert!(
                lde.height() >= min_height,
                "committed LDE height {} is smaller than the blowup factor {min_height}",
                lde.height()
            );
        }
        self.mmcs.commit(ldes)
    }

    /// Commits a streaming iterator of `(domain, matrix, retain_1x)` items.
    ///
    /// Returns `None` if the iterator yields no items. When disk spilling is
    /// enabled, processes one matrix at a time so neither all 1× traces nor all
    /// 4× LDEs coexist in memory simultaneously.
    pub(crate) fn commit_streaming_with_originals(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (TwoAdicMultiplicativeCoset<Val>, RowMajorMatrix<Val>, bool),
        >,
    ) -> Option<SpillCommitResult>
    where
        Dft: TwoAdicSubgroupDft<Val>,
    {
        if let Some(dir) = &self.spill_dir {
            let mut lde_pack: Option<SpillPackWriter> = None;
            let mut orig_pack: Option<SpillPackWriter> = None;
            let mut spilled_ldes = Vec::new();
            let mut spilled_originals = Vec::new();

            for (domain, evals, retain_1x) in evaluations {
                if retain_1x {
                    let writer = orig_pack.get_or_insert_with(|| SpillPackWriter::create(dir));
                    spilled_originals.push(Some(writer.append_matrix(&evals)));
                } else {
                    spilled_originals.push(None);
                }
                let lde = self.compute_lde(domain, evals);
                let writer = lde_pack.get_or_insert_with(|| SpillPackWriter::create(dir));
                spilled_ldes.push(writer.append_matrix(&lde));
                // `lde` drops here before pulling the next matrix.
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
            Some(self.commit_spilled_ldes(spilled_ldes, originals))
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
                ldes.push(self.compute_lde(domain, evals));
            }
            if ldes.is_empty() {
                return None;
            }
            if any_retained {
                Some(self.mmcs.commit_in_memory_with_originals(ldes, originals))
            } else {
                Some(self.mmcs.commit(ldes))
            }
        }
    }
}

impl<Dft, FriMmcs, Challenger> Pcs<ExtVal, Challenger> for SpillableTwoAdicFriPcs<Dft, FriMmcs>
where
    Dft: TwoAdicSubgroupDft<Val>,
    FriMmcs: Mmcs<ExtVal>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = <SpillableMmcs as Mmcs<Val>>::Commitment;
    type ProverData = <SpillableMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixCow<'a, Val>>;
    type Proof = FriProof<ExtVal, FriMmcs, Val, Vec<BatchMultiOpening<Val, SpillableMmcs>>>;
    type Error = FriError<FriMmcs::Error, <SpillableMmcs as Mmcs<Val>>::Error>;
    const ZK: bool = false;

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
        if let Some(dir) = &self.spill_dir {
            let mut pack = SpillPackWriter::create(dir);
            let mut spilled_ldes = Vec::new();
            for (domain, evals) in evaluations {
                let lde = self.compute_lde(domain, evals);
                spilled_ldes.push(pack.append_matrix(&lde));
            }
            pack.flush();
            self.commit_spilled_ldes(spilled_ldes, None)
        } else {
            let ldes: Vec<_> = evaluations
                .into_iter()
                .map(|(domain, evals)| self.compute_lde(domain, evals))
                .collect();
            self.mmcs.commit(ldes)
        }
    }

    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        _num_chunks: usize,
    ) -> Vec<RowMajorMatrix<Val>> {
        evaluations
            .into_iter()
            .map(|(domain, evals)| self.compute_lde(domain, evals))
            .collect()
    }

    fn commit_ldes(&self, ldes: Vec<RowMajorMatrix<Val>>) -> (Self::Commitment, Self::ProverData) {
        let min_height = 1usize << self.fri.log_blowup;
        for lde in &ldes {
            assert!(
                lde.height() >= min_height,
                "committed LDE height {} is smaller than the blowup factor {min_height}",
                lde.height()
            );
        }
        if let Some(dir) = &self.spill_dir {
            let mut pack = SpillPackWriter::create(dir);
            let mut spilled = Vec::with_capacity(ldes.len());
            for lde in ldes {
                spilled.push(pack.append_matrix(&lde));
            }
            pack.flush();
            self.mmcs.commit_spilled(spilled, Vec::new())
        } else {
            self.mmcs.commit(ldes)
        }
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        match &prover_data.0 {
            SpillableMmcsDataInner::InMemory { tree, .. } => {
                let lde = self.mmcs.cpu.get_matrices(tree)[idx];
                if domain.shift() == Val::GENERATOR && lde.height() >= domain.size() {
                    return lde.split_rows(domain.size()).0.as_cow().bit_reverse_rows();
                }
                let poly_height = lde.height() >> self.fri.log_blowup;
                let lde_mat = lde.as_view().bit_reverse_rows().to_row_major_matrix();
                self.re_evaluate_from_lde(lde_mat, poly_height, domain)
            }
            SpillableMmcsDataInner::Spilled { tree, .. } => {
                let spilled = &tree.matrices[idx];
                if domain.shift() == Val::GENERATOR && spilled.height() >= domain.size() {
                    // Read ONLY the `domain.size()` prefix of the bit-reversed
                    // LDE (e.g. half the LDE when quotient_degree = 2, blowup = 4).
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
            ProverDataWithOpeningPoints<'_, ExtVal, Self::ProverData>,
        >,
        challenger: &mut Challenger,
    ) -> (OpenedValues<ExtVal>, Self::Proof) {
        let dims_and_points: Vec<(Vec<Dimensions>, &Vec<Vec<ExtVal>>)> =
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

        let (global_max_height, global_max_width) = dims_and_points
            .iter()
            .flat_map(|(dims, _)| dims.iter().map(|d| (d.height, d.width)))
            .reduce(|(hmax, wmax), (h, w)| (hmax.max(h), wmax.max(w)))
            .expect("No Matrices Supplied?");
        let log_global_max_height = log2_strict_usize(global_max_height);

        let coset = {
            let coset =
                TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_global_max_height).unwrap();
            let mut coset_points: Vec<Val> = coset.iter().collect();
            reverse_slice_index_bits(&mut coset_points);
            coset_points
        };

        let inv_denoms = compute_inverse_denominators_from_dims(&dims_and_points, &coset);
        drop(coset);

        let adjusted_weights: LinearMap<ExtVal, Vec<ExtVal>> = inv_denoms
            .iter()
            .map(|(point, denoms)| (*point, compute_adjusted_weights(*point, denoms)))
            .collect();

        // Pass 1: Barycentric interpolation at each opening point, streaming
        // only the first `h = height >> log_blowup` rows of spilled matrices in
        // bounded chunks (and reading each matrix once across all its opening
        // points).
        let all_opened_values: OpenedValues<ExtVal> = commitment_data_with_opening_points
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
                            SpillableMmcsDataInner::InMemory { tree, .. } => {
                                let mat = self.mmcs.cpu.get_matrices(tree)[mat_idx];
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
                            SpillableMmcsDataInner::Spilled { tree, .. } => {
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

        // Free adjusted_weights before allocating reduced openings.
        drop(adjusted_weights);

        let alpha: ExtVal = challenger.sample_algebra_element();
        let packed_alpha_powers: Vec<_> =
            <ExtVal as ExtensionField<Val>>::ExtensionPacking::packed_ext_powers_capped(
                alpha,
                global_max_width,
            )
            .collect();

        // Pass 2: Build reduced-opening codewords, streaming spilled matrices in
        // bounded row blocks.
        let mut num_reduced = [0usize; 33];
        let mut reduced_openings: [Option<Vec<ExtVal>>; 33] = core::array::from_fn(|_| None);

        for (((data, points), (dims, _)), openings_for_round) in commitment_data_with_opening_points
            .iter()
            .zip(&dims_and_points)
            .zip(&all_opened_values)
        {
            for (mat_idx, (points_for_mat, openings_for_mat)) in
                points.iter().zip(openings_for_round).enumerate()
            {
                let dim = dims[mat_idx];
                let log_height = log2_strict_usize(dim.height);
                let _guard = debug_span!("reduce matrix quotient", dims = %dim).entered();

                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| ExtVal::zero_vec(dim.height));
                debug_assert_eq!(reduced_opening_for_log_height.len(), dim.height);

                let mut point_params = Vec::with_capacity(points_for_mat.len());
                for (&point, openings) in points_for_mat.iter().zip(openings_for_mat) {
                    let alpha_pow_offset = alpha.exp_u64(
                        u64::try_from(num_reduced[log_height])
                            .expect("reduced column count exceeds u64"),
                    );
                    let reduced_val: ExtVal = dot_product(alpha.powers(), openings.iter().copied());
                    let inv_denom = inv_denoms.get(&point).unwrap().as_slice();
                    point_params.push((alpha_pow_offset, reduced_val, inv_denom));
                    num_reduced[log_height] += dim.width;
                }

                if point_params.is_empty() {
                    continue;
                }

                match &data.0 {
                    SpillableMmcsDataInner::InMemory { tree, .. } => {
                        let mat = self.mmcs.cpu.get_matrices(tree)[mat_idx];
                        let mat_compressed: Vec<ExtVal> = mat
                            .as_view()
                            .rowwise_packed_dot_product::<ExtVal>(&packed_alpha_powers)
                            .collect();
                        for &(alpha_pow_offset, reduced_val, inv_denom) in &point_params {
                            mat_compressed
                                .par_iter()
                                .zip(reduced_opening_for_log_height.par_iter_mut())
                                .zip(inv_denom.par_iter())
                                .for_each(|((&reduced_row, ro), &inv_d)| {
                                    *ro += alpha_pow_offset * (reduced_val - reduced_row) * inv_d;
                                });
                        }
                    }
                    SpillableMmcsDataInner::Spilled { tree, .. } => {
                        let spilled = &tree.matrices[mat_idx];
                        let row_bytes = spilled.row_bytes().max(size_of::<u64>());
                        let chunk_rows = (STREAM_BLOCK_BYTES / row_bytes)
                            .clamp(1, 1 << 14)
                            .min(dim.height);
                        for row_start in (0..dim.height).step_by(chunk_rows) {
                            let n_rows = (dim.height - row_start).min(chunk_rows);
                            let row_end = row_start + n_rows;
                            let chunk = spilled.read_rows(row_start, n_rows);
                            let mat_compressed: Vec<ExtVal> = chunk
                                .as_view()
                                .rowwise_packed_dot_product::<ExtVal>(&packed_alpha_powers)
                                .collect();
                            drop(chunk);
                            for &(alpha_pow_offset, reduced_val, inv_denom) in &point_params {
                                mat_compressed
                                    .par_iter()
                                    .zip(
                                        reduced_opening_for_log_height[row_start..row_end]
                                            .par_iter_mut(),
                                    )
                                    .zip(inv_denom[row_start..row_end].par_iter())
                                    .for_each(|((&reduced_row, ro), &inv_d)| {
                                        *ro +=
                                            alpha_pow_offset * (reduced_val - reduced_row) * inv_d;
                                    });
                            }
                        }
                    }
                }
            }
        }

        // Release opening scratch buffers before FRI commit phase starts.
        drop(inv_denoms);
        drop(packed_alpha_powers);

        let fri_input: Vec<Vec<ExtVal>> = reduced_openings.into_iter().rev().flatten().collect();
        let folding: TwoAdicFriFoldingForMmcs<Val, SpillableMmcs> = TwoAdicFriFolding(PhantomData);
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
            CommitmentWithOpeningPoints<ExtVal, Self::Commitment, Self::Domain>,
        >,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        for (_, round) in &commitments_with_opening_points {
            for (_, mat) in round {
                for (_, point) in mat {
                    challenger.observe_algebra_slice(point);
                }
            }
        }
        let folding: TwoAdicFriFoldingForMmcs<Val, SpillableMmcs> = TwoAdicFriFolding(PhantomData);
        verifier::verify_fri(
            &folding,
            &self.fri,
            proof,
            challenger,
            &commitments_with_opening_points,
            &self.mmcs,
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

impl<Dft: TwoAdicSubgroupDft<Val>, FriMmcs> SpillableTwoAdicFriPcs<Dft, FriMmcs> {
    fn re_evaluate_from_lde<'a>(
        &self,
        lde_mat: RowMajorMatrix<Val>,
        poly_height: usize,
        domain: TwoAdicMultiplicativeCoset<Val>,
    ) -> BitReversedMatrixView<RowMajorMatrixCow<'a, Val>> {
        let mut coeffs = self.dft.coset_idft_batch(lde_mat, Val::GENERATOR);
        let width = coeffs.width();
        coeffs.values.truncate(poly_height * width);
        coeffs.values.resize(domain.size() * width, Val::ZERO);
        let result = self
            .dft
            .coset_dft_batch(coeffs, domain.shift())
            .bit_reverse_rows()
            .to_row_major_matrix();
        let result_width = result.width();
        RowMajorMatrixCow::new(Cow::Owned(result.values), result_width).bit_reverse_rows()
    }
}

fn compute_inverse_denominators_from_dims(
    dims_and_points: &[(Vec<Dimensions>, &Vec<Vec<ExtVal>>)],
    coset: &[Val],
) -> LinearMap<ExtVal, Vec<ExtVal>> {
    let mut max_log_height_for_point: LinearMap<ExtVal, usize> = LinearMap::new();
    for (dims, points) in dims_and_points {
        for (dim, points_for_mat) in dims.iter().zip(*points) {
            let log_height = log2_strict_usize(dim.height);
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = (*lh).max(log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    max_log_height_for_point
        .into_iter()
        .map(|(z, log_height)| {
            let diffs: Vec<ExtVal> = coset[..(1 << log_height)].iter().map(|&x| z - x).collect();
            (z, batch_multiplicative_inverse(&diffs))
        })
        .collect()
}

/// Streams rows `0..h` of `spilled` once in bounded chunks and computes the
/// exact barycentric interpolation at every point in `points`, matching
/// `Interpolate::interpolate_coset_with_precomputation`.
fn interpolate_spilled_coset(
    spilled: &SpilledMatrix,
    h: usize,
    points: &[ExtVal],
    adjusted_weights: &LinearMap<ExtVal, Vec<ExtVal>>,
) -> Vec<Vec<ExtVal>> {
    let width = spilled.width();
    let mut accumulators: Vec<Vec<ExtVal>> =
        (0..points.len()).map(|_| ExtVal::zero_vec(width)).collect();
    let row_bytes = spilled.row_bytes().max(size_of::<u64>());
    let chunk_rows = (STREAM_BLOCK_BYTES / row_bytes).clamp(1, 1 << 14).min(h);

    for row_start in (0..h).step_by(chunk_rows) {
        let n_rows = (h - row_start).min(chunk_rows);
        let chunk = spilled.read_rows(row_start, n_rows);
        for (acc, &point) in accumulators.iter_mut().zip(points) {
            let adj_slice = &adjusted_weights.get(&point).unwrap()[row_start..row_start + n_rows];
            let partial = chunk.as_view().columnwise_dot_product(adj_slice);
            for (dst, src) in acc.iter_mut().zip(partial) {
                *dst += src;
            }
        }
    }

    let log_h = log2_strict_usize(h);
    let g_pow_n = Val::GENERATOR.exp_power_of_2(log_h);
    let denom_inv = g_pow_n
        .mul_2exp_u64(u64::try_from(log_h).expect("log_h exceeds u64"))
        .inverse();

    for (evals, &point) in accumulators.iter_mut().zip(points) {
        let z_pow_n = point.exp_power_of_2(log_h);
        let scaling_factor = point * (z_pow_n - g_pow_n) * denom_inv;
        scale_slice_in_place_single_core(evals, scaling_factor);
    }

    accumulators
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    fn make_mmcs(cap_height: usize) -> (CpuMmcs, SpillableMmcs) {
        let byte_hash = Blake3;
        let field_hash = SerializingHasher::new(byte_hash);
        let compress = Blake3CompressionFunction::new(byte_hash);
        (
            CpuMmcs::new(field_hash, compress.clone(), cap_height),
            SpillableMmcs::new(field_hash, compress, cap_height),
        )
    }

    fn temp_spill_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "multi-stark-spill-test-{tag}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn assert_spilled_matches_cpu(
        mats: &[RowMajorMatrix<Val>],
        cap_height: usize,
        query_indices: &[usize],
        tag: &str,
    ) {
        let dir = temp_spill_dir(tag);
        let (cpu, spillable) = make_mmcs(cap_height);
        let (cpu_com, cpu_data) = cpu.commit(mats.to_vec());

        let mut pack = SpillPackWriter::create(&dir);
        let spilled_mats: Vec<_> = mats.iter().map(|m| pack.append_matrix(m)).collect();
        pack.flush();
        let (spilled_com, spilled_data) = spillable.commit_spilled(spilled_mats, Vec::new());

        assert_eq!(
            spilled_com.roots(),
            cpu_com.roots(),
            "cap mismatch for {tag} (cap_height={cap_height})"
        );
        assert!(spilled_data.is_spilled());
        assert_eq!(
            spillable.get_matrix_heights(&spilled_data),
            cpu.get_matrix_heights(&cpu_data)
        );
        assert_eq!(
            spillable.get_max_height(&spilled_data),
            cpu.get_max_height(&cpu_data)
        );

        let dims: Vec<Dimensions> = mats.iter().map(Matrix::dimensions).collect();
        for &idx in query_indices {
            let cpu_open = cpu.open_batch(idx, &cpu_data);
            let spilled_open = spillable.open_batch(idx, &spilled_data);
            assert_eq!(spilled_open.opened_values, cpu_open.opened_values);
            assert_eq!(spilled_open.opening_proof, cpu_open.opening_proof);
            cpu.verify_batch(
                &spilled_com,
                &dims,
                idx,
                BatchOpeningRef::new(&spilled_open.opened_values, &spilled_open.opening_proof),
            )
            .expect("CpuMmcs must accept spilled single opening");
        }

        let (cpu_vals, cpu_multi) = cpu.open_multi_batch(query_indices, &cpu_data);
        let (spilled_vals, spilled_multi) =
            spillable.open_multi_batch(query_indices, &spilled_data);
        assert_eq!(spilled_vals, cpu_vals);
        assert_eq!(spilled_multi.sibling_hashes, cpu_multi.sibling_hashes);
        cpu.verify_multi_batch(
            &spilled_com,
            &dims,
            query_indices,
            &spilled_vals,
            &spilled_multi,
        )
        .expect("CpuMmcs must accept spilled multi-batch opening");

        // Ensure nothing materialized the spilled matrices back into RAM.
        assert_eq!(spilled_data.materialized_spilled_count(), 0);
        drop(spilled_data);
        drop(pack);
        let remaining = fs::read_dir(&dir).unwrap().count();
        assert_eq!(
            remaining, 0,
            "spill directory must have zero leftover files"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn spilled_merkle_matches_cpu_across_shapes_and_caps() {
        let mut rng = SmallRng::seed_from_u64(42);
        let rand_mat = |rng: &mut SmallRng, h: usize, w: usize| {
            RowMajorMatrix::new(
                (0..h * w).map(|_| Goldilocks::new(rng.random())).collect(),
                w,
            )
        };

        // Single matrix, power-of-two height, multiple cap heights + unsorted/duplicate queries.
        let single = vec![rand_mat(&mut rng, 32, 7)];
        for cap_height in [0, 1, 2, 5] {
            assert_spilled_matches_cpu(
                &single,
                cap_height,
                &[17, 0, 31, 17, 4, 5, 0],
                &format!("single-cap{cap_height}"),
            );
        }

        // Height = 1 edge case.
        let unit = vec![rand_mat(&mut rng, 1, 5), rand_mat(&mut rng, 1, 3)];
        for cap_height in [0, 1, 2] {
            assert_spilled_matches_cpu(
                &unit,
                cap_height,
                &[0, 0],
                &format!("unit-cap{cap_height}"),
            );
        }

        // Mixed heights with gaps and multiple matrices at identical heights (insertion order matters!).
        let mixed = vec![
            rand_mat(&mut rng, 16, 4),
            rand_mat(&mut rng, 64, 3),
            rand_mat(&mut rng, 4, 9),
            rand_mat(&mut rng, 64, 5),
            rand_mat(&mut rng, 16, 2),
            rand_mat(&mut rng, 1, 6),
        ];
        for cap_height in [0, 1, 2] {
            assert_spilled_matches_cpu(
                &mixed,
                cap_height,
                &[63, 12, 0, 12, 41, 7, 13, 62],
                &format!("mixed-cap{cap_height}"),
            );
        }

        // Non-power-of-two heights on the ceil(max_height / 2^k) ladder.
        let non_pow2 = vec![
            rand_mat(&mut rng, 11, 3),
            rand_mat(&mut rng, 6, 4),
            rand_mat(&mut rng, 3, 2),
            rand_mat(&mut rng, 2, 5),
            rand_mat(&mut rng, 1, 1),
        ];
        for cap_height in [0, 1, 2] {
            assert_spilled_matches_cpu(
                &non_pow2,
                cap_height,
                &[10, 0, 5, 10, 3],
                &format!("nonpow2-cap{cap_height}"),
            );
        }
    }

    #[test]
    fn spilled_window_wrap_and_interpolation_match_in_memory() {
        let mut rng = SmallRng::seed_from_u64(99);
        let dir = temp_spill_dir("window-interp");
        let mut pack = SpillPackWriter::create(&dir);

        // Height = 1 wraparound edge case.
        let mat1 = RowMajorMatrix::new(vec![Goldilocks::new(11), Goldilocks::new(22)], 2);
        let sp1 = pack.append_matrix(&mat1);

        // Height = 16 matrix for interior + final wraparound chunks and coset interpolation.
        let h = 16usize;
        let w = 7usize;
        let mat16 = RowMajorMatrix::new(
            (0..h * w).map(|_| Goldilocks::new(rng.random())).collect(),
            w,
        );
        let sp16 = pack.append_matrix(&mat16);
        pack.flush();

        // Height = 1 window: row 0 followed by cyclic row 0.
        let win1 = sp1.read_window_with_wrap(0, 1);
        assert_eq!(win1.height(), 2);
        assert_eq!(
            win1.row_slice(0).unwrap().as_ref(),
            &[Goldilocks::new(11), Goldilocks::new(22)]
        );
        assert_eq!(
            win1.row_slice(1).unwrap().as_ref(),
            &[Goldilocks::new(11), Goldilocks::new(22)]
        );

        // Interior chunk [0..5) -> rows 0..6 (halo is row 5).
        let win_mid = sp16.read_window_with_wrap(0, 5);
        assert_eq!(win_mid.height(), 6);
        for r in 0..6 {
            assert_eq!(
                win_mid.row_slice(r).unwrap().as_ref(),
                mat16.row_slice(r).unwrap().as_ref()
            );
        }

        // Final chunk [12..16) -> rows 12, 13, 14, 15, 0 (halo wraps to row 0).
        let win_end = sp16.read_window_with_wrap(12, 4);
        assert_eq!(win_end.height(), 5);
        for local in 0..4 {
            assert_eq!(
                win_end.row_slice(local).unwrap().as_ref(),
                mat16.row_slice(12 + local).unwrap().as_ref()
            );
        }
        assert_eq!(
            win_end.row_slice(4).unwrap().as_ref(),
            mat16.row_slice(0).unwrap().as_ref()
        );

        // Compare `interpolate_spilled_coset` against `Interpolate::interpolate_coset_with_precomputation`.
        let log_h = log2_strict_usize(h);
        let coset: Vec<Val> = {
            let mut pts: Vec<Val> = TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_h)
                .unwrap()
                .iter()
                .collect();
            reverse_slice_index_bits(&mut pts);
            pts
        };
        let p1 = ExtVal::new([Goldilocks::new(12345), Goldilocks::new(67890)]);
        let p2 = ExtVal::new([Goldilocks::new(54321), Goldilocks::new(9876)]);
        let mut adjusted = LinearMap::new();
        for p in [p1, p2] {
            let diffs: Vec<ExtVal> = coset.iter().map(|&x| p - x).collect();
            let invs = batch_multiplicative_inverse(&diffs);
            adjusted.insert(p, compute_adjusted_weights(p, &invs));
        }

        let got = interpolate_spilled_coset(&sp16, h, &[p1, p2], &adjusted);
        for (idx, p) in [p1, p2].into_iter().enumerate() {
            let expected = mat16.interpolate_coset_with_precomputation(
                Val::GENERATOR,
                p,
                adjusted.get(&p).unwrap(),
            );
            assert_eq!(got[idx], expected);
        }

        drop(sp1);
        drop(sp16);
        drop(pack);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }
}
