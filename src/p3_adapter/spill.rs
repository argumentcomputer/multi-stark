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
use std::collections::HashMap;
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
    BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing, PrimeField64, TwoAdicField, dot_product,
    scale_slice_in_place_single_core,
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
use p3_symmetric::{CryptographicHasher, MerkleCap, SerializingHasher};
use p3_util::linear_map::LinearMap;
use p3_util::{log2_ceil_usize, log2_strict_usize, reverse_slice_index_bits};
use tracing::debug_span;

use super::blake3_simd;
use crate::types::{Blake3CompressionFunction, ExtVal, PackedVal, Val};

/// CPU reference MMCS used for verification and reference tests.
pub type CpuMmcs =
    MerkleTreeMmcs<Val, u8, SerializingHasher<Blake3>, Blake3CompressionFunction, 2, 32>;
type SpillCommitResult = (
    <SpillableMmcs as Mmcs<Val>>::Commitment,
    SpillableMmcsData<RowMajorMatrix<Val>>,
);

/// Target memory budget per streamed row block during Merkle hashing and FRI
/// opening passes (64 MiB).
const STREAM_BLOCK_BYTES: usize = 64 << 20;

/// Write buffer size for sequential spill-file packing (8 MiB = 1M field elements).
const SPILL_WRITE_CHUNK_ELEMS: usize = 1 << 20;

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
            writer: BufWriter::with_capacity(4 << 20, write_handle),
            next_offset: 0,
            scratch: Vec::new(),
        }
    }

    #[inline]
    fn pack_chunk_bytes(&mut self, chunk: &[Val]) -> usize {
        let byte_len = chunk.len() * size_of::<u64>();
        if self.scratch.len() < byte_len {
            self.scratch.resize(byte_len, 0);
        }
        let dst = &mut self.scratch[..byte_len];
        if chunk.len() >= 4096 {
            dst.par_chunks_exact_mut(8)
                .zip(chunk.par_iter())
                .for_each(|(out, elem)| {
                    out.copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
                });
        } else {
            let (out_chunks, _) = dst.as_chunks_mut::<8>();
            for (out, elem) in out_chunks.iter_mut().zip(chunk) {
                *out = elem.as_canonical_u64().to_le_bytes();
            }
        }
        byte_len
    }

    /// Appends `mat` in canonical little-endian `u64` row-major order and
    /// returns a lightweight handle to its byte range in the pack file.
    pub(crate) fn append_matrix(&mut self, mat: &RowMajorMatrix<Val>) -> SpilledMatrix {
        let height = mat.height();
        let width = mat.width();
        let offset = self.next_offset;
        let total_elems = mat.values.len();
        for chunk in mat.values.chunks(SPILL_WRITE_CHUNK_ELEMS) {
            let byte_len = self.pack_chunk_bytes(chunk);
            self.writer
                .write_all(&self.scratch[..byte_len])
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

    /// Appends `mat` to the spill file AND computes each row's BLAKE3 leaf
    /// digest from the same cache-hot serialized chunks, eliminating a full
    /// disk read pass for single-matrix height groups.
    pub(crate) fn append_matrix_and_hash(
        &mut self,
        mat: &RowMajorMatrix<Val>,
    ) -> (SpilledMatrix, Vec<[u8; 32]>) {
        let height = mat.height();
        let width = mat.width();
        let offset = self.next_offset;
        if width == 0 {
            let empty_digest = Blake3.hash_slice(&[]);
            return (
                SpilledMatrix {
                    file: Arc::clone(&self.spill),
                    offset,
                    height,
                    width,
                },
                vec![empty_digest; height],
            );
        }
        let row_bytes = width * size_of::<u64>();
        let chunk_rows =
            ((SPILL_WRITE_CHUNK_ELEMS * size_of::<u64>()) / row_bytes).clamp(1, height.max(1));
        // SAFETY: the loop partitions `0..height` into disjoint contiguous chunks
        // `row_start..row_start + n_rows` and initializes each via `hash_raw_rows_uninit`.
        let digests = unsafe {
            blake3_simd::alloc_and_fill_digests(height, height, |dst| {
                for row_start in (0..height).step_by(chunk_rows) {
                    let n_rows = (height - row_start).min(chunk_rows);
                    let elem_start = row_start * width;
                    let elem_end = (row_start + n_rows) * width;
                    let byte_len = self.pack_chunk_bytes(&mat.values[elem_start..elem_end]);
                    blake3_simd::hash_raw_rows_uninit(
                        &self.scratch[..byte_len],
                        row_bytes,
                        &mut dst[row_start..row_start + n_rows],
                    );
                    self.writer
                        .write_all(&self.scratch[..byte_len])
                        .expect("failed to write matrix chunk to spill file");
                }
            })
        };
        let total_elems = mat.values.len();
        let bytes = u64::try_from(total_elems)
            .expect("matrix element count exceeds u64")
            .checked_mul(u64::try_from(size_of::<u64>()).unwrap())
            .expect("matrix byte length overflow");
        self.next_offset = self
            .next_offset
            .checked_add(bytes)
            .expect("spill pack offset overflow");
        TOTAL_SPILLED_BYTES.fetch_add(bytes, Ordering::Relaxed);
        (
            SpilledMatrix {
                file: Arc::clone(&self.spill),
                offset,
                height,
                width,
            },
            digests,
        )
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
    if chunks.len() >= 4096 {
        chunks
            .par_iter()
            .map(|&chunk| Goldilocks::new(u64::from_le_bytes(chunk)))
            .collect()
    } else {
        chunks
            .iter()
            .map(|&chunk| Goldilocks::new(u64::from_le_bytes(chunk)))
            .collect()
    }
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
fn validate_reachable_heights_iter(heights: impl Iterator<Item = usize> + Clone) -> usize {
    let max_height = heights
        .clone()
        .max()
        .filter(|&h| h > 0)
        .expect("all matrices have height 0");
    let log_max_height = log2_ceil_usize(max_height);
    for h in heights {
        assert!(
            h > 0,
            "matrix height 0 incompatible with tallest height {max_height}"
        );
        let bits_reduced = log_max_height - log2_ceil_usize(h);
        let expected_height = ((max_height - 1) >> bits_reduced) + 1;
        assert_eq!(
            h, expected_height,
            "matrix height {h} incompatible with tallest height {max_height} (expected {expected_height})"
        );
    }
    max_height
}

/// Shared Merkle digest layers and opening-proof traversal for both in-memory
/// and disk-spilled binary BLAKE3 Merkle trees.
struct MerkleDigests {
    digest_layers: Vec<Vec<[u8; 32]>>,
    arity_schedule: Vec<usize>,
}

impl MerkleDigests {
    fn cap(&self, cap_height: usize) -> MerkleCap<Val, [u8; 32]> {
        let num_layers = self.digest_layers.len();
        let effective_cap_height = cap_height.min(num_layers.saturating_sub(1));
        let layer_idx = num_layers - 1 - effective_cap_height;
        let layer = &self.digest_layers[layer_idx];
        let cap_len: usize = self.arity_schedule[layer_idx..].iter().product();
        let cap_len = cap_len.min(layer.len());
        MerkleCap::new(layer[..cap_len].to_vec())
    }

    fn open_single_proof(&self, index: usize, cap_height: usize) -> Vec<[u8; 32]> {
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
        proof
    }

    fn open_multi_proof(&self, indices: &[usize], cap_height: usize) -> PrunedMerklePaths<u8, 32> {
        let num_layers = self.digest_layers.len();
        let effective_cap_height = cap_height.min(num_layers.saturating_sub(1));
        let proof_levels = num_layers
            .saturating_sub(1)
            .saturating_sub(effective_cap_height);

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

        PrunedMerklePaths { sibling_hashes }
    }
}

/// Fast in-memory binary BLAKE3 Merkle tree accelerated with multi-buffer SIMD
/// hashing for both leaf rows and internal 2-to-1 compression layers.
struct FastInMemoryMerkleTree<M> {
    leaves: Vec<M>,
    digests: MerkleDigests,
}

impl<M: Matrix<Val>> FastInMemoryMerkleTree<M> {
    fn build(leaves: Vec<M>) -> Self {
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
        // SAFETY: `hash_matrix_rows_uninit` initializes every element of `dst` (`0..max_height`).
        let first_layer = unsafe {
            blake3_simd::alloc_and_fill_digests(padded_len_binary(max_height), max_height, |dst| {
                blake3_simd::hash_matrix_rows_uninit(&tallest_refs, 0, dst);
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
                compress_layer_binary(prev_layer)
            } else {
                let inject_refs: Vec<&M> = inject_group.iter().map(|&idx| &leaves[idx]).collect();
                compress_and_inject_in_memory(prev_layer, &inject_refs)
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

    fn open_batch(&self, index: usize, cap_height: usize) -> (Vec<Vec<Val>>, Vec<[u8; 32]>) {
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
        let proof = self.digests.open_single_proof(index, cap_height);
        (openings, proof)
    }

    fn open_multi_batch(
        &self,
        indices: &[usize],
        cap_height: usize,
    ) -> (Vec<Vec<Vec<Val>>>, PrunedMerklePaths<u8, 32>) {
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
        let proof = self.digests.open_multi_proof(indices, cap_height);
        (opened_values, proof)
    }
}

/// A binary Blake3 Merkle tree whose leaf matrices reside on disk in
/// [`SpilledMatrix`] handles, while only the compact internal `digest_layers`
/// remain in RAM.
pub(crate) struct SpilledMerkleTree {
    matrices: Vec<SpilledMatrix>,
    digests: MerkleDigests,
    /// Optional spilled 1× original traces retained for zero-FFT on-demand
    /// lookup evaluation; released once stage-2 completes for stage-1 data,
    /// or retained for the lifetime of `ProverKey` for preprocessed data.
    original_traces: Mutex<Vec<Option<SpilledMatrix>>>,
}

impl SpilledMerkleTree {
    pub(crate) fn build_with_prehashed(
        matrices: Vec<SpilledMatrix>,
        original_traces: Vec<Option<SpilledMatrix>>,
        mut prehashed: Vec<Option<Vec<[u8; 32]>>>,
    ) -> Self {
        assert!(!matrices.is_empty(), "No matrices given?");
        let max_height =
            validate_reachable_heights_iter(matrices.iter().map(SpilledMatrix::height));

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

        let first_layer = if let &[single_idx] = tallest
            && let Some(mut digests) = prehashed.get_mut(single_idx).and_then(Option::take)
        {
            debug_assert_eq!(digests.len(), max_height);
            digests.resize(padded_len_binary(max_height), [0u8; 32]);
            digests
        } else {
            hash_leaf_group_streaming(&matrices, tallest, max_height)
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
                compress_layer_binary(prev_layer)
            } else if let &[single_idx] = inject_group
                && let Some(rows_digests) = prehashed.get_mut(single_idx).and_then(Option::take)
            {
                compress_and_inject_prehashed(prev_layer, &rows_digests)
            } else {
                compress_and_inject_streaming(prev_layer, &matrices, inject_group)
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
            original_traces: Mutex::new(original_traces),
        }
    }

    pub(crate) fn cap(&self, cap_height: usize) -> MerkleCap<Val, [u8; 32]> {
        self.digests.cap(cap_height)
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

        let proof = self.digests.open_single_proof(index, cap_height);
        (openings, proof)
    }

    pub(crate) fn open_multi_batch(
        &self,
        indices: &[usize],
        cap_height: usize,
    ) -> (Vec<Vec<Vec<Val>>>, PrunedMerklePaths<u8, 32>) {
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

        let proof = self.digests.open_multi_proof(indices, cap_height);
        (opened_values, proof)
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
    let block_rows = block_rows_for_group(matrices, group, max_height);
    // SAFETY: the loop covers `0..max_height` in contiguous chunks and initializes
    // each via `hash_raw_multi_rows_uninit`.
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
                blake3_simd::hash_raw_multi_rows_uninit(&blocks, &mut dst[start..start + n_rows]);
            }
        })
    }
}

fn compress_layer_binary(prev_layer: &[[u8; 32]]) -> Vec<[u8; 32]> {
    let raw_next = prev_layer.len() / 2;
    let next_len_padded = padded_len_binary(raw_next);
    // SAFETY: `compress_pairs_uninit` initializes every element of `dst` (`0..raw_next`).
    unsafe {
        blake3_simd::alloc_and_fill_digests(next_len_padded, raw_next, |dst| {
            blake3_simd::compress_pairs_uninit(prev_layer, dst);
        })
    }
}

fn compress_and_inject_prehashed(
    prev_layer: &[[u8; 32]],
    rows_digests: &[[u8; 32]],
) -> Vec<[u8; 32]> {
    let next_len = rows_digests.len();
    let raw_next = prev_layer.len() / 2;
    let next_len_padded = padded_len_binary(raw_next);
    // SAFETY: initializes `dst[..next_len]` and (if `raw_next > next_len`)
    // `dst[next_len..raw_next]`, covering all of `0..raw_next`.
    unsafe {
        blake3_simd::alloc_and_fill_digests(next_len_padded, raw_next, |dst| {
            blake3_simd::compress_and_inject_from_digests_uninit(
                prev_layer,
                rows_digests,
                &mut dst[..next_len],
            );
            if raw_next > next_len {
                let zeros = vec![[0u8; 32]; raw_next - next_len];
                blake3_simd::compress_and_inject_from_digests_uninit(
                    &prev_layer[2 * next_len..],
                    &zeros,
                    &mut dst[next_len..raw_next],
                );
            }
        })
    }
}

fn compress_and_inject_in_memory<M: Matrix<Val>>(
    prev_layer: &[[u8; 32]],
    inject_group: &[&M],
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
                blake3_simd::hash_matrix_rows(inject_group, start, &mut row_digests[..n_rows]);
                blake3_simd::compress_and_inject_from_digests_uninit(
                    &prev_layer[2 * start..],
                    &row_digests[..n_rows],
                    &mut dst[start..start + n_rows],
                );
            }

            if raw_next > next_len {
                let zeros = vec![[0u8; 32]; raw_next - next_len];
                blake3_simd::compress_and_inject_from_digests_uninit(
                    &prev_layer[2 * next_len..],
                    &zeros,
                    &mut dst[next_len..raw_next],
                );
            }
        })
    }
}

fn compress_and_inject_streaming(
    prev_layer: &[[u8; 32]],
    matrices: &[SpilledMatrix],
    inject_group: &[usize],
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
                blake3_simd::hash_raw_multi_rows(&blocks, &mut row_digests[..n_rows]);
                blake3_simd::compress_and_inject_from_digests_uninit(
                    &prev_layer[2 * start..],
                    &row_digests[..n_rows],
                    &mut dst[start..start + n_rows],
                );
            }

            if raw_next > next_len {
                let zeros = vec![[0u8; 32]; raw_next - next_len];
                blake3_simd::compress_and_inject_from_digests_uninit(
                    &prev_layer[2 * next_len..],
                    &zeros,
                    &mut dst[next_len..raw_next],
                );
            }
        })
    }
}

/// Prover data for [`SpillableMmcs`]: either an in-memory multi-buffer SIMD
/// BLAKE3 Merkle tree or a disk-spilled [`SpilledMerkleTree`].
pub struct SpillableMmcsData<M>(SpillableMmcsDataInner<M>);

enum SpillableMmcsDataInner<M> {
    InMemory {
        tree: FastInMemoryMerkleTree<M>,
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

/// CPU MMCS supporting both in-memory commitments (default, accelerated via
/// multi-buffer SIMD BLAKE3) and disk-spilled commitments with identical
/// commitments and opening proofs.
#[derive(Clone, Debug)]
pub struct SpillableMmcs {
    cpu: CpuMmcs,
}

impl SpillableMmcs {
    pub fn new(
        hash: SerializingHasher<Blake3>,
        compress: Blake3CompressionFunction,
        cap_height: usize,
    ) -> Self {
        Self {
            cpu: CpuMmcs::new(hash, compress, cap_height),
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
        let tree = FastInMemoryMerkleTree::build(ldes);
        let commitment = tree.cap(self.cap_height());
        (
            commitment,
            SpillableMmcsData(SpillableMmcsDataInner::InMemory {
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
    ) -> SpillCommitResult {
        let count = matrices.len();
        let tree = SpilledMerkleTree::build_with_prehashed(matrices, original_traces, prehashed);
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

    #[cfg(test)]
    pub(crate) fn commit_spilled(
        &self,
        matrices: Vec<SpilledMatrix>,
        original_traces: Vec<Option<SpilledMatrix>>,
    ) -> SpillCommitResult {
        self.commit_spilled_prehashed(matrices, original_traces, Vec::new())
    }

    pub(crate) fn matrix_dimensions<M: Matrix<Val>>(
        &self,
        prover_data: &SpillableMmcsData<M>,
    ) -> Vec<Dimensions> {
        match &prover_data.0 {
            SpillableMmcsDataInner::InMemory { tree, .. } => tree.dimensions(),
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
        let tree = FastInMemoryMerkleTree::build(inputs);
        let commitment = tree.cap(self.cap_height());
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
                let (opened_values, opening_proof) = tree.open_batch(index, self.cap_height());
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
            SpillableMmcsDataInner::InMemory { tree, .. } => tree.leaves.iter().collect(),
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
            SpillableMmcsDataInner::InMemory { tree, .. } => tree.matrix_heights(),
            SpillableMmcsDataInner::Spilled { tree, .. } => tree.matrix_heights(),
        }
    }

    fn get_max_height<M: Matrix<Val>>(&self, prover_data: &Self::ProverData<M>) -> usize {
        match &prover_data.0 {
            SpillableMmcsDataInner::InMemory { tree, .. } => tree.max_height(),
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
                tree.open_multi_batch(indices, self.cap_height())
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

    pub(crate) fn commit_spilled_ldes_prehashed(
        &self,
        spilled_ldes: Vec<SpilledMatrix>,
        original_traces: Option<Vec<Option<SpilledMatrix>>>,
        prehashed: Vec<Option<Vec<[u8; 32]>>>,
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
        self.mmcs
            .commit_spilled_prehashed(spilled_ldes, originals, prehashed)
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
    /// 4× LDEs coexist in memory simultaneously, hashing unique-height matrices
    /// in RAM before dropping them to avoid a second disk pass.
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
            let mut prehashed: Vec<Option<Vec<[u8; 32]>>> = Vec::new();
            let mut first_idx_for_height: HashMap<usize, Option<usize>> = HashMap::new();

            for (domain, evals, retain_1x) in evaluations {
                if retain_1x {
                    let writer = orig_pack.get_or_insert_with(|| SpillPackWriter::create(dir));
                    spilled_originals.push(Some(writer.append_matrix(&evals)));
                } else {
                    spilled_originals.push(None);
                }
                let lde = self.compute_lde(domain, evals);
                let lde_height = lde.height();
                let idx = spilled_ldes.len();
                let writer = lde_pack.get_or_insert_with(|| SpillPackWriter::create(dir));
                match first_idx_for_height.get_mut(&lde_height) {
                    None => {
                        let (spilled, digests) = writer.append_matrix_and_hash(&lde);
                        spilled_ldes.push(spilled);
                        prehashed.push(Some(digests));
                        first_idx_for_height.insert(lde_height, Some(idx));
                    }
                    Some(prev) => {
                        // A second matrix shares this height; drop the single-matrix
                        // prehash so this height group streams concatenated rows.
                        if let Some(prev_idx) = prev.take() {
                            prehashed[prev_idx] = None;
                        }
                        spilled_ldes.push(writer.append_matrix(&lde));
                        prehashed.push(None);
                    }
                }
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
            Some(self.commit_spilled_ldes_prehashed(spilled_ldes, originals, prehashed))
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
            let mut prehashed: Vec<Option<Vec<[u8; 32]>>> = Vec::new();
            let mut first_idx_for_height: HashMap<usize, Option<usize>> = HashMap::new();
            for (domain, evals) in evaluations {
                let lde = self.compute_lde(domain, evals);
                let h = lde.height();
                let idx = spilled_ldes.len();
                match first_idx_for_height.get_mut(&h) {
                    None => {
                        let (spilled, digests) = pack.append_matrix_and_hash(&lde);
                        spilled_ldes.push(spilled);
                        prehashed.push(Some(digests));
                        first_idx_for_height.insert(h, Some(idx));
                    }
                    Some(prev) => {
                        if let Some(prev_idx) = prev.take() {
                            prehashed[prev_idx] = None;
                        }
                        spilled_ldes.push(pack.append_matrix(&lde));
                        prehashed.push(None);
                    }
                }
            }
            pack.flush();
            self.commit_spilled_ldes_prehashed(spilled_ldes, None, prehashed)
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
            let mut height_counts: HashMap<usize, usize> = HashMap::new();
            for lde in &ldes {
                *height_counts.entry(lde.height()).or_insert(0) += 1;
            }
            let mut pack = SpillPackWriter::create(dir);
            let mut spilled = Vec::with_capacity(ldes.len());
            let mut prehashed = Vec::with_capacity(ldes.len());
            for lde in ldes {
                if height_counts[&lde.height()] == 1 {
                    let (sp, digests) = pack.append_matrix_and_hash(&lde);
                    spilled.push(sp);
                    prehashed.push(Some(digests));
                } else {
                    spilled.push(pack.append_matrix(&lde));
                    prehashed.push(None);
                }
            }
            pack.flush();
            self.mmcs
                .commit_spilled_prehashed(spilled, Vec::new(), prehashed)
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
                let lde = &tree.leaves[idx];
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

        let coset = build_bitrev_coset(log_global_max_height);

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
                        let mat = &tree.leaves[mat_idx];
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

/// Builds the bit-reversed evaluation coset `GENERATOR * <g>` of size
/// `1 << log_size` in parallel chunks.
fn build_bitrev_coset(log_size: usize) -> Vec<Val> {
    let size = 1usize << log_size;
    let g = Val::two_adic_generator(log_size);
    let mut coset = Val::zero_vec(size);
    const CHUNK: usize = 4096;
    if size <= CHUNK {
        let mut cur = Val::GENERATOR;
        for slot in &mut coset {
            *slot = cur;
            cur *= g;
        }
    } else {
        coset
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut cur = Val::GENERATOR
                    * g.exp_u64(u64::try_from(chunk_idx * CHUNK).expect("coset index exceeds u64"));
                for slot in chunk {
                    *slot = cur;
                    cur *= g;
                }
            });
    }
    reverse_slice_index_bits(&mut coset);
    coset
}

#[inline]
fn invert_lanes_in_place(lanes: &[Val], invs: &mut [Val]) {
    let n = lanes.len();
    if n == 0 {
        return;
    }
    let mut acc = Val::ONE;
    for (slot, &v) in invs.iter_mut().zip(lanes) {
        *slot = acc;
        acc *= v;
    }
    let mut inv = acc.inverse();
    for i in (0..n).rev() {
        invs[i] *= inv;
        inv *= lanes[i];
    }
}

struct NormScratch {
    reals: Vec<PackedVal>,
    norms: Vec<PackedVal>,
    prefixes: Vec<PackedVal>,
    lane_invs: Vec<Val>,
}

/// Computes `[(z - x)^{-1} for x in coset]` in `Goldilocks[X]/(X^2 - W)` using
/// the base-field norm trick:
///
/// ```text
/// (z_0 - x + z_1·X)^{-1} = ((z_0 - x) - z_1·X) / ((z_0 - x)^2 - W·z_1^2)
/// ```
///
/// Every chunk runs Montgomery batch inversion in the base field `PackedVal`
/// inside L1 cache (3× fewer base multiplications than extension-field batch
/// inversion, SIMD-vectorized across `PackedVal::WIDTH` lanes, parallelized
/// across all cores, and with zero intermediate `diffs` allocation).
fn goldilocks_quadratic_inverse_denominators(z: ExtVal, coset: &[Val]) -> Vec<ExtVal> {
    const CHUNK: usize = 1024;
    let coords = <ExtVal as BasedVectorSpace<Val>>::as_basis_coefficients_slice(&z);
    let z0 = coords[0];
    let z1 = coords[1];
    let neg_z1 = -z1;
    let crate::traits::ExtensionModulus::Binomial { w } =
        <ExtVal as crate::traits::ExtensionOf<Val>>::MODULUS
    else {
        unreachable!("ExtVal is a binomial quadratic extension")
    };
    let w_z1_sq = w * z1 * z1;

    let pw = PackedVal::WIDTH;
    let packs_per_chunk = CHUNK / pw;
    let mut out = ExtVal::zero_vec(coset.len());

    let process_chunk = |scratch: &mut NormScratch, x_chunk: &[Val], out_chunk: &mut [ExtVal]| {
        let (x_packed, x_tail) = PackedVal::pack_slice_with_suffix(x_chunk);
        let n_packs = x_packed.len();
        if n_packs > 0 {
            let z0_p = PackedVal::from(z0);
            let neg_z1_p = PackedVal::from(neg_z1);
            let w_z1_sq_p = PackedVal::from(w_z1_sq);
            let mut acc = PackedVal::ONE;
            for (i, &xp) in x_packed.iter().enumerate() {
                let real = z0_p - xp;
                let norm = real * real - w_z1_sq_p;
                scratch.reals[i] = real;
                scratch.norms[i] = norm;
                scratch.prefixes[i] = acc;
                acc *= norm;
            }
            invert_lanes_in_place(acc.as_slice(), &mut scratch.lane_invs);
            let mut inv = *PackedVal::from_slice(&scratch.lane_invs);
            for i in (0..n_packs).rev() {
                let inv_norm = scratch.prefixes[i] * inv;
                inv *= scratch.norms[i];
                let c0 = scratch.reals[i] * inv_norm;
                let c1 = neg_z1_p * inv_norm;
                let c0_s = c0.as_slice();
                let c1_s = c1.as_slice();
                let dst = &mut out_chunk[i * pw..(i + 1) * pw];
                for lane in 0..pw {
                    dst[lane] = ExtVal::new([c0_s[lane], c1_s[lane]]);
                }
            }
        }
        if !x_tail.is_empty() {
            let tail_out = &mut out_chunk[n_packs * pw..];
            for (&x, dst) in x_tail.iter().zip(tail_out.iter_mut()) {
                let real = z0 - x;
                let inv_norm = (real * real - w_z1_sq).inverse();
                *dst = ExtVal::new([real * inv_norm, neg_z1 * inv_norm]);
            }
        }
    };

    if coset.len() <= CHUNK {
        let mut scratch = NormScratch {
            reals: vec![PackedVal::ZERO; packs_per_chunk.max(1)],
            norms: vec![PackedVal::ZERO; packs_per_chunk.max(1)],
            prefixes: vec![PackedVal::ZERO; packs_per_chunk.max(1)],
            lane_invs: vec![Val::ZERO; pw],
        };
        process_chunk(&mut scratch, coset, &mut out);
    } else {
        out.par_chunks_mut(CHUNK)
            .zip(coset.par_chunks(CHUNK))
            .for_each_init(
                || NormScratch {
                    reals: vec![PackedVal::ZERO; packs_per_chunk.max(1)],
                    norms: vec![PackedVal::ZERO; packs_per_chunk.max(1)],
                    prefixes: vec![PackedVal::ZERO; packs_per_chunk.max(1)],
                    lane_invs: vec![Val::ZERO; pw],
                },
                |scratch, (out_chunk, x_chunk)| process_chunk(scratch, x_chunk, out_chunk),
            );
    }

    out
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
            (
                z,
                goldilocks_quadratic_inverse_denominators(z, &coset[..(1 << log_height)]),
            )
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
    use p3_field::batch_multiplicative_inverse;
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

        // Also check our SIMD in-memory MMCS matches `CpuMmcs` bit-for-bit.
        let (mem_com, mem_data) = spillable.commit(mats.to_vec());
        assert_eq!(
            mem_com.roots(),
            cpu_com.roots(),
            "in-memory SIMD cap mismatch for {tag} (cap_height={cap_height})"
        );
        let (cpu_vals, cpu_multi) = cpu.open_multi_batch(query_indices, &cpu_data);
        let (mem_vals, mem_multi) = spillable.open_multi_batch(query_indices, &mem_data);
        assert_eq!(mem_vals, cpu_vals);
        assert_eq!(mem_multi.sibling_hashes, cpu_multi.sibling_hashes);

        // Check spilled (streaming disk hash) path.
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

        // Also check hash-before-spill (`append_matrix_and_hash` + `commit_spilled_prehashed`).
        let mut height_counts: HashMap<usize, usize> = HashMap::new();
        for m in mats {
            *height_counts.entry(m.height()).or_insert(0) += 1;
        }
        let mut pack_pre = SpillPackWriter::create(&dir);
        let mut sp_pre = Vec::with_capacity(mats.len());
        let mut prehashed = Vec::with_capacity(mats.len());
        for m in mats {
            if height_counts[&m.height()] == 1 {
                let (sp, digests) = pack_pre.append_matrix_and_hash(m);
                sp_pre.push(sp);
                prehashed.push(Some(digests));
            } else {
                sp_pre.push(pack_pre.append_matrix(m));
                prehashed.push(None);
            }
        }
        pack_pre.flush();
        let (pre_com, pre_data) = spillable.commit_spilled_prehashed(sp_pre, Vec::new(), prehashed);
        assert_eq!(pre_com.roots(), cpu_com.roots());
        let (pre_vals, pre_multi) = spillable.open_multi_batch(query_indices, &pre_data);
        assert_eq!(pre_vals, cpu_vals);
        assert_eq!(pre_multi.sibling_hashes, cpu_multi.sibling_hashes);
        assert_eq!(pre_data.materialized_spilled_count(), 0);
        drop(pre_data);
        drop(pack_pre);

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

    #[test]
    fn parallel_bitrev_coset_and_norm_trick_match_reference() {
        for log_size in [0usize, 1, 2, 3, 5, 10, 13] {
            let expected_coset: Vec<Val> = {
                let mut pts: Vec<Val> = TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_size)
                    .unwrap()
                    .iter()
                    .collect();
                reverse_slice_index_bits(&mut pts);
                pts
            };
            let got_coset = build_bitrev_coset(log_size);
            assert_eq!(got_coset, expected_coset, "coset mismatch at 2^{log_size}");

            for point in [
                ExtVal::new([Goldilocks::new(12345), Goldilocks::new(67890)]),
                ExtVal::new([Goldilocks::new(987_654_321), Goldilocks::ZERO]),
                ExtVal::new([Goldilocks::ZERO, Goldilocks::new(42)]),
            ] {
                let diffs: Vec<ExtVal> = expected_coset.iter().map(|&x| point - x).collect();
                let expected_invs = batch_multiplicative_inverse(&diffs);
                let got_invs = goldilocks_quadratic_inverse_denominators(point, &got_coset);
                assert_eq!(
                    got_invs, expected_invs,
                    "norm-trick inverse mismatch at 2^{log_size}"
                );
            }
        }
    }
}
