//! Multi-buffer SIMD BLAKE3 hashing for CPU Merkle trees.
//!
//! Plonky3's `MerkleTreeMmcs` with `SerializingHasher<Blake3>` and
//! `CompressionFunctionFromHasher<Blake3, 2, 32>` hashes one row and compresses
//! one 64-byte internal node pair at a time. Because BLAKE3's built-in SIMD
//! only parallelizes *across 1 KiB chunks of a single message*, single-block
//! internal compressions (64 bytes) and small-to-medium leaf rows (≤ 16 KiB)
//! run completely scalar.
//!
//! This module implements 16-way AVX-512 (`avx512f`) and 8-way AVX2 (`avx2`)
//! multi-buffer BLAKE3 kernels with runtime CPU feature detection and a clean
//! scalar fallback, producing digests 100% bit-for-bit identical to
//! `p3_blake3::Blake3`.

use core::mem::MaybeUninit;

use p3_blake3::Blake3;
use p3_field::PrimeField64;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::CryptographicHasher;

use crate::types::Val;

const BLOCK_LEN: usize = 64;
const CHUNK_LEN: usize = 1024;
/// Maximum row byte length handled by the multi-buffer SIMD path (16 chunks = 16 KiB).
/// Beyond 16 KiB, `blake3::Hasher` already saturates 16-chunk AVX-512 within each single row.
const MAX_MULTIBUF_ROW_BYTES: usize = 16 * CHUNK_LEN;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;

const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimdBackend {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    Scalar,
}

#[inline]
fn detect_backend() -> SimdBackend {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return SimdBackend::Avx512;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return SimdBackend::Avx2;
        }
    }
    SimdBackend::Scalar
}

#[inline]
fn compress_pair_scalar(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut block = [0u8; 64];
    block[..32].copy_from_slice(left);
    block[32..].copy_from_slice(right);
    compress_64_root_scalar(&block)
}

#[inline]
fn compress_64_root_scalar(block: &[u8; 64]) -> [u8; 32] {
    let mut msg = [0u32; 16];
    let (chunks, _) = block.as_chunks::<4>();
    for (w, &bytes) in msg.iter_mut().zip(chunks) {
        *w = u32::from_le_bytes(bytes);
    }
    let mut state = [
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        IV[4],
        IV[5],
        IV[6],
        IV[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        0,
        0,
        64,
        CHUNK_START | CHUNK_END | ROOT,
    ];
    for schedule in &MSG_SCHEDULE {
        g_scalar(&mut state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
        g_scalar(&mut state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
        g_scalar(&mut state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
        g_scalar(&mut state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);
        g_scalar(&mut state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
        g_scalar(
            &mut state,
            1,
            6,
            11,
            12,
            msg[schedule[10]],
            msg[schedule[11]],
        );
        g_scalar(
            &mut state,
            2,
            7,
            8,
            13,
            msg[schedule[12]],
            msg[schedule[13]],
        );
        g_scalar(
            &mut state,
            3,
            4,
            9,
            14,
            msg[schedule[14]],
            msg[schedule[15]],
        );
    }
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&(state[i] ^ state[i + 8]).to_le_bytes());
    }
    out
}

#[inline(always)]
fn g_scalar(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(x);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(y);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

/// Allocates a `Vec<[u8; 32]>` of length `padded`, initializes `0..active` via
/// `fill` (without any upfront single-threaded zero-fill), zero-fills the
/// padding tail `active..padded`, and returns the initialized `Vec`.
///
/// # Safety
/// `fill` MUST initialize every element of the `&mut [MaybeUninit<[u8; 32]>]`
/// slice it receives before returning.
pub(crate) unsafe fn alloc_and_fill_digests(
    padded: usize,
    active: usize,
    fill: impl FnOnce(&mut [MaybeUninit<[u8; 32]>]),
) -> Vec<[u8; 32]> {
    debug_assert!(active <= padded);
    let mut vec: Vec<[u8; 32]> = Vec::with_capacity(padded);
    let spare = &mut vec.spare_capacity_mut()[..padded];
    fill(&mut spare[..active]);
    for slot in &mut spare[active..padded] {
        slot.write([0u8; 32]);
    }
    // SAFETY: Caller guarantees `fill` initialized every element of
    // `spare[..active]`, and the loop above initialized `spare[active..padded]`.
    unsafe {
        vec.set_len(padded);
    }
    vec
}

#[inline]
fn compress_pairs_slice(
    prev_layer: &[[u8; 32]],
    out: &mut [MaybeUninit<[u8; 32]>],
    backend: SimdBackend,
) {
    debug_assert!(prev_layer.len() >= 2 * out.len());
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if backend == SimdBackend::Avx512 {
            while i + 16 <= out.len() {
                // SAFETY: `avx512f` detected at runtime; `out[i..i+16]` has 16 writable slots.
                unsafe {
                    avx512::compress_16_pairs(
                        &prev_layer[2 * i..2 * i + 32],
                        out[i..].as_mut_ptr().cast(),
                    );
                }
                i += 16;
            }
            if i + 8 <= out.len() {
                // SAFETY: `avx512f` implies `avx2`; `out[i..i+8]` has 8 writable slots.
                unsafe {
                    avx2::compress_8_pairs(
                        &prev_layer[2 * i..2 * i + 16],
                        out[i..].as_mut_ptr().cast(),
                    );
                }
                i += 8;
            }
        } else if backend == SimdBackend::Avx2 {
            while i + 8 <= out.len() {
                // SAFETY: `avx2` detected at runtime; `out[i..i+8]` has 8 writable slots.
                unsafe {
                    avx2::compress_8_pairs(
                        &prev_layer[2 * i..2 * i + 16],
                        out[i..].as_mut_ptr().cast(),
                    );
                }
                i += 8;
            }
        }
    }
    let _ = backend;
    while i < out.len() {
        out[i].write(compress_pair_scalar(
            &prev_layer[2 * i],
            &prev_layer[2 * i + 1],
        ));
        i += 1;
    }
}

/// Compresses `out.len()` adjacent 32-byte digest pairs from `prev_layer`
/// (`out[i] = BLAKE3(prev_layer[2*i] || prev_layer[2*i + 1])`) in parallel
/// using multi-buffer SIMD BLAKE3.
pub(crate) fn compress_pairs_uninit(prev_layer: &[[u8; 32]], out: &mut [MaybeUninit<[u8; 32]>]) {
    assert!(prev_layer.len() >= 2 * out.len());
    if out.is_empty() {
        return;
    }
    let backend = detect_backend();
    const PAR_THRESHOLD: usize = 128;
    const CHUNK_PAIRS: usize = 128;
    if out.len() < PAR_THRESHOLD {
        compress_pairs_slice(prev_layer, out, backend);
    } else {
        out.par_chunks_mut(CHUNK_PAIRS)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let start = chunk_idx * CHUNK_PAIRS;
                compress_pairs_slice(&prev_layer[2 * start..], dst, backend);
            });
    }
}

#[inline]
fn compress_and_inject_slice(
    prev_layer: &[[u8; 32]],
    rows_digests: &[[u8; 32]],
    out: &mut [MaybeUninit<[u8; 32]>],
    backend: SimdBackend,
) {
    debug_assert_eq!(rows_digests.len(), out.len());
    debug_assert!(prev_layer.len() >= 2 * out.len());
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if backend == SimdBackend::Avx512 {
            while i + 16 <= out.len() {
                // SAFETY: `avx512f` detected at runtime; `out[i..i+16]` has 16 writable slots.
                unsafe {
                    avx512::compress_16_pairs_and_inject(
                        &prev_layer[2 * i..2 * i + 32],
                        (&rows_digests[i..i + 16]).try_into().unwrap(),
                        out[i..].as_mut_ptr().cast(),
                    );
                }
                i += 16;
            }
            if i + 8 <= out.len() {
                // SAFETY: `avx512f` implies `avx2`; `out[i..i+8]` has 8 writable slots.
                unsafe {
                    avx2::compress_8_pairs_and_inject(
                        &prev_layer[2 * i..2 * i + 16],
                        (&rows_digests[i..i + 8]).try_into().unwrap(),
                        out[i..].as_mut_ptr().cast(),
                    );
                }
                i += 8;
            }
        } else if backend == SimdBackend::Avx2 {
            while i + 8 <= out.len() {
                // SAFETY: `avx2` detected at runtime; `out[i..i+8]` has 8 writable slots.
                unsafe {
                    avx2::compress_8_pairs_and_inject(
                        &prev_layer[2 * i..2 * i + 16],
                        (&rows_digests[i..i + 8]).try_into().unwrap(),
                        out[i..].as_mut_ptr().cast(),
                    );
                }
                i += 8;
            }
        }
    }
    let _ = backend;
    while i < out.len() {
        let parent = compress_pair_scalar(&prev_layer[2 * i], &prev_layer[2 * i + 1]);
        out[i].write(compress_pair_scalar(&parent, &rows_digests[i]));
        i += 1;
    }
}

/// Fused binary compress + leaf-digest injection into uninitialized output:
/// `out[i] = BLAKE3(BLAKE3(prev_layer[2*i] || prev_layer[2*i + 1]) || rows_digests[i])`.
pub(crate) fn compress_and_inject_from_digests_uninit(
    prev_layer: &[[u8; 32]],
    rows_digests: &[[u8; 32]],
    out: &mut [MaybeUninit<[u8; 32]>],
) {
    assert_eq!(rows_digests.len(), out.len());
    assert!(prev_layer.len() >= 2 * out.len());
    if out.is_empty() {
        return;
    }
    let backend = detect_backend();
    const PAR_THRESHOLD: usize = 128;
    const CHUNK_PAIRS: usize = 256;
    if out.len() < PAR_THRESHOLD {
        compress_and_inject_slice(prev_layer, rows_digests, out, backend);
    } else {
        out.par_chunks_mut(CHUNK_PAIRS)
            .zip(rows_digests.par_chunks(CHUNK_PAIRS))
            .enumerate()
            .for_each(|(chunk_idx, (dst, row_chunk))| {
                let start = chunk_idx * CHUNK_PAIRS;
                compress_and_inject_slice(&prev_layer[2 * start..], row_chunk, dst, backend);
            });
    }
}

#[inline]
fn hash_raw_rows_slice(
    raw: &[u8],
    row_bytes: usize,
    out: &mut [MaybeUninit<[u8; 32]>],
    backend: SimdBackend,
) {
    debug_assert_eq!(raw.len(), out.len() * row_bytes);
    if out.is_empty() {
        return;
    }
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if row_bytes <= MAX_MULTIBUF_ROW_BYTES {
            if backend == SimdBackend::Avx512 {
                while i + 16 <= out.len() {
                    let ptrs: [*const u8; 16] =
                        core::array::from_fn(|lane| raw[(i + lane) * row_bytes..].as_ptr());
                    // SAFETY: `avx512f` detected at runtime; each pointer has `row_bytes` readable bytes
                    // and `out[i..i+16]` has 16 writable slots.
                    unsafe {
                        avx512::hash_16_fixed_len(&ptrs, row_bytes, out[i..].as_mut_ptr().cast());
                    }
                    i += 16;
                }
                if i + 8 <= out.len() {
                    let ptrs: [*const u8; 8] =
                        core::array::from_fn(|lane| raw[(i + lane) * row_bytes..].as_ptr());
                    // SAFETY: `avx512f` implies `avx2`; each pointer has `row_bytes` readable bytes.
                    unsafe {
                        avx2::hash_8_fixed_len(&ptrs, row_bytes, out[i..].as_mut_ptr().cast());
                    }
                    i += 8;
                }
            } else if backend == SimdBackend::Avx2 {
                while i + 8 <= out.len() {
                    let ptrs: [*const u8; 8] =
                        core::array::from_fn(|lane| raw[(i + lane) * row_bytes..].as_ptr());
                    // SAFETY: `avx2` detected at runtime; each pointer has `row_bytes` readable bytes.
                    unsafe {
                        avx2::hash_8_fixed_len(&ptrs, row_bytes, out[i..].as_mut_ptr().cast());
                    }
                    i += 8;
                }
            }
        }
    }
    let _ = backend;
    let hasher = Blake3;
    while i < out.len() {
        out[i].write(hasher.hash_slice(&raw[i * row_bytes..(i + 1) * row_bytes]));
        i += 1;
    }
}

/// Hashes `out.len()` contiguous fixed-length rows of `row_bytes` bytes each
/// directly into uninitialized digest slots.
pub(crate) fn hash_raw_rows_uninit(
    raw: &[u8],
    row_bytes: usize,
    out: &mut [MaybeUninit<[u8; 32]>],
) {
    assert_eq!(raw.len(), out.len() * row_bytes);
    if out.is_empty() {
        return;
    }
    let backend = detect_backend();
    if row_bytes > MAX_MULTIBUF_ROW_BYTES {
        let hasher = Blake3;
        out.par_iter_mut().enumerate().for_each(|(row, digest)| {
            digest.write(hasher.hash_slice(&raw[row * row_bytes..(row + 1) * row_bytes]));
        });
        return;
    }
    let chunk_rows = (65_536 / row_bytes.max(64))
        .clamp(16, 256)
        .next_multiple_of(16);
    if out.len() <= chunk_rows {
        hash_raw_rows_slice(raw, row_bytes, out, backend);
    } else {
        out.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let start = chunk_idx * chunk_rows;
                let byte_start = start * row_bytes;
                let byte_end = (start + dst.len()) * row_bytes;
                hash_raw_rows_slice(&raw[byte_start..byte_end], row_bytes, dst, backend);
            });
    }
}

/// Hashes `out.len()` rows formed by concatenating the corresponding row slices
/// across `blocks` (each `(raw_bytes, row_bytes)`).
pub(crate) fn hash_raw_multi_rows_uninit(
    blocks: &[(Vec<u8>, usize)],
    out: &mut [MaybeUninit<[u8; 32]>],
) {
    if out.is_empty() {
        return;
    }
    if let [(buf, row_bytes)] = blocks {
        hash_raw_rows_uninit(buf, *row_bytes, out);
        return;
    }
    let total_row_bytes: usize = blocks.iter().map(|(_, rb)| *rb).sum();
    if total_row_bytes > MAX_MULTIBUF_ROW_BYTES {
        let hasher = Blake3;
        out.par_iter_mut()
            .enumerate()
            .for_each(|(local_row, digest)| {
                digest.write(
                    hasher.hash_iter_slices(
                        blocks
                            .iter()
                            .map(|(buf, rb)| &buf[local_row * *rb..(local_row + 1) * *rb]),
                    ),
                );
            });
        return;
    }
    let backend = detect_backend();
    let batch = match backend {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx512 => 16usize,
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => 8usize,
        SimdBackend::Scalar => 1usize,
    };
    if batch == 1 {
        let hasher = Blake3;
        out.par_iter_mut()
            .enumerate()
            .for_each(|(local_row, digest)| {
                digest.write(
                    hasher.hash_iter_slices(
                        blocks
                            .iter()
                            .map(|(buf, rb)| &buf[local_row * *rb..(local_row + 1) * *rb]),
                    ),
                );
            });
        return;
    }
    let chunk_rows = (65_536 / total_row_bytes.max(64))
        .clamp(batch, 256)
        .next_multiple_of(batch);
    out.par_chunks_mut(chunk_rows).enumerate().for_each_init(
        || vec![0u8; batch * total_row_bytes],
        |scratch, (chunk_idx, dst)| {
            let chunk_start = chunk_idx * chunk_rows;
            let mut offset = 0;
            while offset < dst.len() {
                let take = (dst.len() - offset).min(batch);
                for lane in 0..take {
                    let row = chunk_start + offset + lane;
                    let mut pos = lane * total_row_bytes;
                    for (buf, rb) in blocks {
                        if *rb > 0 {
                            scratch[pos..pos + *rb]
                                .copy_from_slice(&buf[row * *rb..(row + 1) * *rb]);
                            pos += *rb;
                        }
                    }
                }
                hash_raw_rows_slice(
                    &scratch[..take * total_row_bytes],
                    total_row_bytes,
                    &mut dst[offset..offset + take],
                    backend,
                );
                offset += take;
            }
        },
    );
}

/// Hashes `out.len()` rows starting at `start_row` across one or more in-memory
/// Goldilocks matrices (`Matrix<Val>`) directly into uninitialized digest slots.
pub(crate) fn hash_matrix_rows_uninit<M: Matrix<Val>>(
    matrices: &[&M],
    start_row: usize,
    out: &mut [MaybeUninit<[u8; 32]>],
) {
    if out.is_empty() {
        return;
    }
    let total_width: usize = matrices.iter().map(|m| m.width()).sum();
    let total_row_bytes = total_width * size_of::<u64>();
    let backend = detect_backend();

    if total_row_bytes == 0 {
        let empty_digest = Blake3.hash_slice(&[]);
        for slot in out {
            slot.write(empty_digest);
        }
        return;
    }

    if total_row_bytes > MAX_MULTIBUF_ROW_BYTES {
        let hasher = Blake3;
        out.par_iter_mut().enumerate().for_each_init(
            || vec![0u8; total_row_bytes],
            |scratch, (local_row, digest)| {
                let r = start_row + local_row;
                let mut pos = 0;
                for m in matrices {
                    // SAFETY: `r < m.height()`.
                    for elem in unsafe { m.row_unchecked(r) } {
                        scratch[pos..pos + 8]
                            .copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
                        pos += 8;
                    }
                }
                digest.write(hasher.hash_slice(scratch));
            },
        );
        return;
    }

    let batch = match backend {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx512 => 16usize,
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => 8usize,
        SimdBackend::Scalar => 1usize,
    };

    // Fast path for rows fitting in a single 64-byte BLAKE3 block (<= 8
    // Goldilocks columns: covers all FRI layers and quotient LDEs): serialize
    // directly into a 1 KiB zero-padded stack block array with zero heap scratch.
    #[cfg(target_arch = "x86_64")]
    if total_row_bytes <= BLOCK_LEN && batch > 1 {
        let block_len_u32 = u32::try_from(total_row_bytes).unwrap();
        let chunk_rows = 256usize;
        let process_small = |chunk_start: usize, dst: &mut [MaybeUninit<[u8; 32]>]| {
            let mut offset = 0;
            while offset + batch <= dst.len() {
                let mut blocks = [[0u8; BLOCK_LEN]; 16];
                for (lane, block) in blocks[..batch].iter_mut().enumerate() {
                    let r = chunk_start + offset + lane;
                    let mut pos = 0;
                    for m in matrices {
                        // SAFETY: `r < m.height()`.
                        for elem in unsafe { m.row_unchecked(r) } {
                            block[pos..pos + 8]
                                .copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
                            pos += 8;
                        }
                    }
                }
                if backend == SimdBackend::Avx512 {
                    // SAFETY: `avx512f` detected at runtime; `dst[offset..offset+16]` has 16 slots.
                    unsafe {
                        avx512::hash_16_single_block(
                            &blocks,
                            block_len_u32,
                            dst[offset..].as_mut_ptr().cast(),
                        );
                    }
                } else {
                    // SAFETY: `avx2` detected at runtime; `dst[offset..offset+8]` has 8 slots.
                    unsafe {
                        avx2::hash_8_single_block(
                            (&blocks[..8]).try_into().unwrap(),
                            block_len_u32,
                            dst[offset..].as_mut_ptr().cast(),
                        );
                    }
                }
                offset += batch;
            }
            if offset < dst.len() {
                let hasher = Blake3;
                let mut block = [0u8; BLOCK_LEN];
                while offset < dst.len() {
                    let r = chunk_start + offset;
                    let mut pos = 0;
                    for m in matrices {
                        // SAFETY: `r < m.height()`.
                        for elem in unsafe { m.row_unchecked(r) } {
                            block[pos..pos + 8]
                                .copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
                            pos += 8;
                        }
                    }
                    dst[offset].write(hasher.hash_slice(&block[..total_row_bytes]));
                    offset += 1;
                }
            }
        };
        if out.len() <= chunk_rows {
            process_small(start_row, out);
        } else {
            out.par_chunks_mut(chunk_rows)
                .enumerate()
                .for_each(|(chunk_idx, dst)| {
                    process_small(start_row + chunk_idx * chunk_rows, dst);
                });
        }
        return;
    }

    let chunk_rows = (65_536 / total_row_bytes.max(64))
        .clamp(batch, 256)
        .next_multiple_of(batch);

    let process_chunk =
        |scratch: &mut Vec<u8>, chunk_start: usize, dst: &mut [MaybeUninit<[u8; 32]>]| {
            let mut offset = 0;
            while offset < dst.len() {
                let take = (dst.len() - offset).min(batch);
                for lane in 0..take {
                    let r = chunk_start + offset + lane;
                    let mut pos = lane * total_row_bytes;
                    for m in matrices {
                        // SAFETY: `r < m.height()`.
                        for elem in unsafe { m.row_unchecked(r) } {
                            scratch[pos..pos + 8]
                                .copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
                            pos += 8;
                        }
                    }
                }
                hash_raw_rows_slice(
                    &scratch[..take * total_row_bytes],
                    total_row_bytes,
                    &mut dst[offset..offset + take],
                    backend,
                );
                offset += take;
            }
        };

    if out.len() <= chunk_rows {
        let mut scratch = vec![0u8; batch * total_row_bytes];
        process_chunk(&mut scratch, start_row, out);
    } else {
        out.par_chunks_mut(chunk_rows).enumerate().for_each_init(
            || vec![0u8; batch * total_row_bytes],
            |scratch, (chunk_idx, dst)| {
                process_chunk(scratch, start_row + chunk_idx * chunk_rows, dst);
            },
        );
    }
}

/// Reinterprets an initialized digest slice as `&mut [MaybeUninit<[u8; 32]>]`.
///
/// # Safety
/// The caller must never write uninitialized data into the returned slice.
#[inline]
unsafe fn as_uninit_slice(slice: &mut [[u8; 32]]) -> &mut [MaybeUninit<[u8; 32]>] {
    // SAFETY: `MaybeUninit<[u8; 32]>` has identical layout to `[u8; 32]`, and
    // callers guarantee they only write initialized `[u8; 32]` values.
    unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
}

/// Safe wrapper over [`hash_raw_multi_rows_uninit`] writing into an existing
/// initialized digest buffer.
pub(crate) fn hash_raw_multi_rows(blocks: &[(Vec<u8>, usize)], out: &mut [[u8; 32]]) {
    // SAFETY: `hash_raw_multi_rows_uninit` only writes initialized `[u8; 32]` values.
    let uninit = unsafe { as_uninit_slice(out) };
    hash_raw_multi_rows_uninit(blocks, uninit);
}

/// Safe wrapper over [`hash_matrix_rows_uninit`] writing into an existing
/// initialized digest buffer.
pub(crate) fn hash_matrix_rows<M: Matrix<Val>>(
    matrices: &[&M],
    start_row: usize,
    out: &mut [[u8; 32]],
) {
    // SAFETY: `hash_matrix_rows_uninit` only writes initialized `[u8; 32]` values.
    let uninit = unsafe { as_uninit_slice(out) };
    hash_matrix_rows_uninit(matrices, start_row, uninit);
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use core::arch::x86_64::*;

    use super::{BLOCK_LEN, CHUNK_END, CHUNK_LEN, CHUNK_START, IV, MSG_SCHEDULE, PARENT, ROOT};

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn splat_u32(v: u32) -> __m512i {
        _mm512_set1_epi32(v.cast_signed())
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn iv_vecs() -> [__m512i; 8] {
        [
            splat_u32(IV[0]),
            splat_u32(IV[1]),
            splat_u32(IV[2]),
            splat_u32(IV[3]),
            splat_u32(IV[4]),
            splat_u32(IV[5]),
            splat_u32(IV[6]),
            splat_u32(IV[7]),
        ]
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn g_16(
        v: &mut [__m512i; 16],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        mx: __m512i,
        my: __m512i,
    ) {
        v[a] = _mm512_add_epi32(_mm512_add_epi32(v[a], v[b]), mx);
        v[d] = _mm512_ror_epi32::<16>(_mm512_xor_si512(v[d], v[a]));
        v[c] = _mm512_add_epi32(v[c], v[d]);
        v[b] = _mm512_ror_epi32::<12>(_mm512_xor_si512(v[b], v[c]));
        v[a] = _mm512_add_epi32(_mm512_add_epi32(v[a], v[b]), my);
        v[d] = _mm512_ror_epi32::<8>(_mm512_xor_si512(v[d], v[a]));
        v[c] = _mm512_add_epi32(v[c], v[d]);
        v[b] = _mm512_ror_epi32::<7>(_mm512_xor_si512(v[b], v[c]));
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn compress_block_16(
        cv: &mut [__m512i; 8],
        msg: &[__m512i; 16],
        block_len: u32,
        counter: u64,
        flags: u32,
    ) {
        let counter_lo = u32::try_from(counter & 0xFFFF_FFFF).unwrap();
        let counter_hi = u32::try_from(counter >> 32).unwrap();
        let mut v = [
            cv[0],
            cv[1],
            cv[2],
            cv[3],
            cv[4],
            cv[5],
            cv[6],
            cv[7],
            splat_u32(IV[0]),
            splat_u32(IV[1]),
            splat_u32(IV[2]),
            splat_u32(IV[3]),
            splat_u32(counter_lo),
            splat_u32(counter_hi),
            splat_u32(block_len),
            splat_u32(flags),
        ];
        for s in &MSG_SCHEDULE {
            g_16(&mut v, 0, 4, 8, 12, msg[s[0]], msg[s[1]]);
            g_16(&mut v, 1, 5, 9, 13, msg[s[2]], msg[s[3]]);
            g_16(&mut v, 2, 6, 10, 14, msg[s[4]], msg[s[5]]);
            g_16(&mut v, 3, 7, 11, 15, msg[s[6]], msg[s[7]]);
            g_16(&mut v, 0, 5, 10, 15, msg[s[8]], msg[s[9]]);
            g_16(&mut v, 1, 6, 11, 12, msg[s[10]], msg[s[11]]);
            g_16(&mut v, 2, 7, 8, 13, msg[s[12]], msg[s[13]]);
            g_16(&mut v, 3, 4, 9, 14, msg[s[14]], msg[s[15]]);
        }
        for i in 0..8 {
            cv[i] = _mm512_xor_si512(v[i], v[i + 8]);
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn transpose_16x16(rows: &[__m512i; 16]) -> [__m512i; 16] {
        let mut t0 = [_mm512_setzero_si512(); 16];
        for k in 0..8 {
            t0[2 * k] = _mm512_unpacklo_epi32(rows[2 * k], rows[2 * k + 1]);
            t0[2 * k + 1] = _mm512_unpackhi_epi32(rows[2 * k], rows[2 * k + 1]);
        }
        let mut t1 = [_mm512_setzero_si512(); 16];
        for j in 0..4 {
            let g = 4 * j;
            t1[g] = _mm512_unpacklo_epi64(t0[g], t0[g + 2]);
            t1[g + 1] = _mm512_unpackhi_epi64(t0[g], t0[g + 2]);
            t1[g + 2] = _mm512_unpacklo_epi64(t0[g + 1], t0[g + 3]);
            t1[g + 3] = _mm512_unpackhi_epi64(t0[g + 1], t0[g + 3]);
        }
        let mut out = [_mm512_setzero_si512(); 16];
        for m in 0..4 {
            let lo0 = _mm512_shuffle_i32x4::<0x44>(t1[m], t1[4 + m]);
            let hi0 = _mm512_shuffle_i32x4::<0xEE>(t1[m], t1[4 + m]);
            let lo1 = _mm512_shuffle_i32x4::<0x44>(t1[8 + m], t1[12 + m]);
            let hi1 = _mm512_shuffle_i32x4::<0xEE>(t1[8 + m], t1[12 + m]);
            out[m] = _mm512_shuffle_i32x4::<0x88>(lo0, lo1);
            out[4 + m] = _mm512_shuffle_i32x4::<0xDD>(lo0, lo1);
            out[8 + m] = _mm512_shuffle_i32x4::<0x88>(hi0, hi1);
            out[12 + m] = _mm512_shuffle_i32x4::<0xDD>(hi0, hi1);
        }
        out
    }

    /// Transposes 16 digests of 8 `u32` words each into 8 AVX-512 word vectors.
    #[target_feature(enable = "avx512f")]
    #[inline]
    fn load_16_digests_transposed(digests: &[[u8; 32]; 16]) -> [__m512i; 8] {
        let mut lo = [_mm256_setzero_si256(); 8];
        let mut hi = [_mm256_setzero_si256(); 8];
        // SAFETY: `digests` holds 16 valid 32-byte arrays.
        unsafe {
            for k in 0..8 {
                lo[k] = _mm256_loadu_si256(digests[k].as_ptr().cast());
                hi[k] = _mm256_loadu_si256(digests[8 + k].as_ptr().cast());
            }
        }
        let lo_t = super::avx2::transpose_8x8(&lo);
        let hi_t = super::avx2::transpose_8x8(&hi);
        core::array::from_fn(|w| _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo_t[w]), hi_t[w]))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn unpack_cv_16(cv: &[__m512i; 8], out: *mut [u8; 32]) {
        let mut a = [_mm512_setzero_si512(); 8];
        for k in 0..4 {
            a[2 * k] = _mm512_unpacklo_epi32(cv[2 * k], cv[2 * k + 1]);
            a[2 * k + 1] = _mm512_unpackhi_epi32(cv[2 * k], cv[2 * k + 1]);
        }
        let b = [
            _mm512_unpacklo_epi64(a[0], a[2]),
            _mm512_unpackhi_epi64(a[0], a[2]),
            _mm512_unpacklo_epi64(a[1], a[3]),
            _mm512_unpackhi_epi64(a[1], a[3]),
            _mm512_unpacklo_epi64(a[4], a[6]),
            _mm512_unpackhi_epi64(a[4], a[6]),
            _mm512_unpacklo_epi64(a[5], a[7]),
            _mm512_unpackhi_epi64(a[5], a[7]),
        ];
        for r in 0..4 {
            let lo_tmp = _mm512_shuffle_i32x4::<0x44>(b[r], b[4 + r]);
            let lo_pair = _mm512_shuffle_i32x4::<0xD8>(lo_tmp, lo_tmp);
            let hi_tmp = _mm512_shuffle_i32x4::<0xEE>(b[r], b[4 + r]);
            let hi_pair = _mm512_shuffle_i32x4::<0xD8>(hi_tmp, hi_tmp);
            // SAFETY: caller guarantees `out` points to 16 writable 32-byte slots.
            unsafe {
                _mm256_storeu_si256(out.add(r).cast(), _mm512_castsi512_si256(lo_pair));
                _mm256_storeu_si256(
                    out.add(4 + r).cast(),
                    _mm512_extracti64x4_epi64::<1>(lo_pair),
                );
                _mm256_storeu_si256(out.add(8 + r).cast(), _mm512_castsi512_si256(hi_pair));
                _mm256_storeu_si256(
                    out.add(12 + r).cast(),
                    _mm512_extracti64x4_epi64::<1>(hi_pair),
                );
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn hash_16_single_block(
        blocks: &[[u8; BLOCK_LEN]; 16],
        block_len: u32,
        out: *mut [u8; 32],
    ) {
        let mut rows = [_mm512_setzero_si512(); 16];
        // SAFETY: `blocks` holds 16 contiguous 64-byte arrays.
        unsafe {
            for (lane, row) in rows.iter_mut().enumerate() {
                *row = _mm512_loadu_si512(blocks[lane].as_ptr().cast());
            }
        }
        let msg = transpose_16x16(&rows);
        let mut cv = iv_vecs();
        compress_block_16(&mut cv, &msg, block_len, 0, CHUNK_START | CHUNK_END | ROOT);
        // SAFETY: caller guarantees `out` points to 16 writable slots.
        unsafe { unpack_cv_16(&cv, out) };
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn compress_16_pairs(prev_32: &[[u8; 32]], out: *mut [u8; 32]) {
        debug_assert!(prev_32.len() >= 32);
        let mut rows = [_mm512_setzero_si512(); 16];
        let base = prev_32.as_ptr().cast::<u8>();
        // SAFETY: `prev_32` has 32 contiguous 32-byte elements = 16 contiguous 64-byte blocks.
        unsafe {
            for (lane, row) in rows.iter_mut().enumerate() {
                *row = _mm512_loadu_si512(base.add(lane * 64).cast());
            }
        }
        let msg = transpose_16x16(&rows);
        let mut cv = iv_vecs();
        compress_block_16(&mut cv, &msg, 64, 0, CHUNK_START | CHUNK_END | ROOT);
        // SAFETY: caller guarantees `out` points to 16 writable slots.
        unsafe { unpack_cv_16(&cv, out) };
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn compress_16_pairs_and_inject(
        prev_32: &[[u8; 32]],
        rows_digests: &[[u8; 32]; 16],
        out: *mut [u8; 32],
    ) {
        debug_assert!(prev_32.len() >= 32);
        let mut rows = [_mm512_setzero_si512(); 16];
        let base = prev_32.as_ptr().cast::<u8>();
        // SAFETY: `prev_32` has 32 contiguous 32-byte elements = 16 contiguous 64-byte blocks.
        unsafe {
            for (lane, row) in rows.iter_mut().enumerate() {
                *row = _mm512_loadu_si512(base.add(lane * 64).cast());
            }
        }
        let msg1 = transpose_16x16(&rows);
        let mut parent_cv = iv_vecs();
        compress_block_16(&mut parent_cv, &msg1, 64, 0, CHUNK_START | CHUNK_END | ROOT);

        let right_cv = load_16_digests_transposed(rows_digests);
        let msg2 = [
            parent_cv[0],
            parent_cv[1],
            parent_cv[2],
            parent_cv[3],
            parent_cv[4],
            parent_cv[5],
            parent_cv[6],
            parent_cv[7],
            right_cv[0],
            right_cv[1],
            right_cv[2],
            right_cv[3],
            right_cv[4],
            right_cv[5],
            right_cv[6],
            right_cv[7],
        ];
        let mut final_cv = iv_vecs();
        compress_block_16(&mut final_cv, &msg2, 64, 0, CHUNK_START | CHUNK_END | ROOT);
        // SAFETY: caller guarantees `out` points to 16 writable slots.
        unsafe { unpack_cv_16(&final_cv, out) };
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn hash_chunk_16(
        ptrs: &[*const u8; 16],
        chunk_offset: usize,
        chunk_len: usize,
        counter: u64,
        is_root: bool,
    ) -> [__m512i; 8] {
        let mut cv = iv_vecs();
        let num_blocks = chunk_len.div_ceil(BLOCK_LEN).max(1);
        for b in 0..num_blocks {
            let block_offset = chunk_offset + b * BLOCK_LEN;
            let rem = chunk_len.saturating_sub(b * BLOCK_LEN);
            let block_len = rem.min(BLOCK_LEN);
            let mut rows = [_mm512_setzero_si512(); 16];
            // SAFETY: caller guarantees each `ptrs[lane]` has `chunk_offset + chunk_len` readable bytes.
            unsafe {
                if block_len == BLOCK_LEN {
                    for lane in 0..16 {
                        rows[lane] = _mm512_loadu_si512(ptrs[lane].add(block_offset).cast());
                    }
                } else {
                    for lane in 0..16 {
                        let mut tmp = [0u8; BLOCK_LEN];
                        if block_len > 0 {
                            core::ptr::copy_nonoverlapping(
                                ptrs[lane].add(block_offset),
                                tmp.as_mut_ptr(),
                                block_len,
                            );
                        }
                        rows[lane] = _mm512_loadu_si512(tmp.as_ptr().cast());
                    }
                }
            }
            let msg = transpose_16x16(&rows);
            let mut flags = 0u32;
            if b == 0 {
                flags |= CHUNK_START;
            }
            if b + 1 == num_blocks {
                flags |= CHUNK_END;
                if is_root {
                    flags |= ROOT;
                }
            }
            compress_block_16(
                &mut cv,
                &msg,
                u32::try_from(block_len).unwrap(),
                counter,
                flags,
            );
        }
        cv
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn compress_parent_16(
        left: &[__m512i; 8],
        right: &[__m512i; 8],
        is_root: bool,
    ) -> [__m512i; 8] {
        let msg = [
            left[0], left[1], left[2], left[3], left[4], left[5], left[6], left[7], right[0],
            right[1], right[2], right[3], right[4], right[5], right[6], right[7],
        ];
        let mut cv = iv_vecs();
        let flags = if is_root { PARENT | ROOT } else { PARENT };
        compress_block_16(&mut cv, &msg, 64, 0, flags);
        cv
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn hash_16_fixed_len(
        ptrs: &[*const u8; 16],
        row_bytes: usize,
        out: *mut [u8; 32],
    ) {
        debug_assert!(row_bytes <= super::MAX_MULTIBUF_ROW_BYTES);
        if row_bytes <= CHUNK_LEN {
            // SAFETY: caller guarantees `ptrs` validity for `row_bytes` and `out` has 16 slots.
            unsafe {
                let cv = hash_chunk_16(ptrs, 0, row_bytes, 0, true);
                unpack_cv_16(&cv, out);
            }
            return;
        }
        let num_chunks = row_bytes.div_ceil(CHUNK_LEN);
        // Max `num_chunks` is 16 -> max CV stack depth is 5.
        let mut cv_stack = [[_mm512_setzero_si512(); 8]; 6];
        let mut stack_len = 0usize;
        for c in 0..num_chunks {
            let chunk_offset = c * CHUNK_LEN;
            let chunk_len = (row_bytes - chunk_offset).min(CHUNK_LEN);
            // SAFETY: caller guarantees `ptrs` validity for `row_bytes`.
            let mut new_cv = unsafe {
                hash_chunk_16(
                    ptrs,
                    chunk_offset,
                    chunk_len,
                    u64::try_from(c).unwrap(),
                    false,
                )
            };
            if c + 1 < num_chunks {
                let mut total_chunks = c + 1;
                while total_chunks & 1 == 0 {
                    stack_len -= 1;
                    new_cv = compress_parent_16(&cv_stack[stack_len], &new_cv, false);
                    total_chunks >>= 1;
                }
                cv_stack[stack_len] = new_cv;
                stack_len += 1;
            } else {
                while stack_len > 1 {
                    stack_len -= 1;
                    new_cv = compress_parent_16(&cv_stack[stack_len], &new_cv, false);
                }
                stack_len -= 1;
                let root_cv = compress_parent_16(&cv_stack[stack_len], &new_cv, true);
                // SAFETY: caller guarantees `out` has 16 writable slots.
                unsafe { unpack_cv_16(&root_cv, out) };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::*;

    use super::{BLOCK_LEN, CHUNK_END, CHUNK_LEN, CHUNK_START, IV, MSG_SCHEDULE, PARENT, ROOT};

    #[target_feature(enable = "avx2")]
    #[inline]
    fn splat_u32(v: u32) -> __m256i {
        _mm256_set1_epi32(v.cast_signed())
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn iv_vecs() -> [__m256i; 8] {
        [
            splat_u32(IV[0]),
            splat_u32(IV[1]),
            splat_u32(IV[2]),
            splat_u32(IV[3]),
            splat_u32(IV[4]),
            splat_u32(IV[5]),
            splat_u32(IV[6]),
            splat_u32(IV[7]),
        ]
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn rot16(x: __m256i) -> __m256i {
        let mask = _mm256_setr_epi8(
            2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11,
            8, 9, 14, 15, 12, 13,
        );
        _mm256_shuffle_epi8(x, mask)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn rot8(x: __m256i) -> __m256i {
        let mask = _mm256_setr_epi8(
            1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10,
            11, 8, 13, 14, 15, 12,
        );
        _mm256_shuffle_epi8(x, mask)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn rot12(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_srli_epi32::<12>(x), _mm256_slli_epi32::<20>(x))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn rot7(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_srli_epi32::<7>(x), _mm256_slli_epi32::<25>(x))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn g_8(
        v: &mut [__m256i; 16],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        mx: __m256i,
        my: __m256i,
    ) {
        v[a] = _mm256_add_epi32(_mm256_add_epi32(v[a], v[b]), mx);
        v[d] = rot16(_mm256_xor_si256(v[d], v[a]));
        v[c] = _mm256_add_epi32(v[c], v[d]);
        v[b] = rot12(_mm256_xor_si256(v[b], v[c]));
        v[a] = _mm256_add_epi32(_mm256_add_epi32(v[a], v[b]), my);
        v[d] = rot8(_mm256_xor_si256(v[d], v[a]));
        v[c] = _mm256_add_epi32(v[c], v[d]);
        v[b] = rot7(_mm256_xor_si256(v[b], v[c]));
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn compress_block_8(
        cv: &mut [__m256i; 8],
        msg: &[__m256i; 16],
        block_len: u32,
        counter: u64,
        flags: u32,
    ) {
        let counter_lo = u32::try_from(counter & 0xFFFF_FFFF).unwrap();
        let counter_hi = u32::try_from(counter >> 32).unwrap();
        let mut v = [
            cv[0],
            cv[1],
            cv[2],
            cv[3],
            cv[4],
            cv[5],
            cv[6],
            cv[7],
            splat_u32(IV[0]),
            splat_u32(IV[1]),
            splat_u32(IV[2]),
            splat_u32(IV[3]),
            splat_u32(counter_lo),
            splat_u32(counter_hi),
            splat_u32(block_len),
            splat_u32(flags),
        ];
        for s in &MSG_SCHEDULE {
            g_8(&mut v, 0, 4, 8, 12, msg[s[0]], msg[s[1]]);
            g_8(&mut v, 1, 5, 9, 13, msg[s[2]], msg[s[3]]);
            g_8(&mut v, 2, 6, 10, 14, msg[s[4]], msg[s[5]]);
            g_8(&mut v, 3, 7, 11, 15, msg[s[6]], msg[s[7]]);
            g_8(&mut v, 0, 5, 10, 15, msg[s[8]], msg[s[9]]);
            g_8(&mut v, 1, 6, 11, 12, msg[s[10]], msg[s[11]]);
            g_8(&mut v, 2, 7, 8, 13, msg[s[12]], msg[s[13]]);
            g_8(&mut v, 3, 4, 9, 14, msg[s[14]], msg[s[15]]);
        }
        for i in 0..8 {
            cv[i] = _mm256_xor_si256(v[i], v[i + 8]);
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) fn transpose_8x8(rows: &[__m256i; 8]) -> [__m256i; 8] {
        let t0 = [
            _mm256_unpacklo_epi32(rows[0], rows[1]),
            _mm256_unpackhi_epi32(rows[0], rows[1]),
            _mm256_unpacklo_epi32(rows[2], rows[3]),
            _mm256_unpackhi_epi32(rows[2], rows[3]),
            _mm256_unpacklo_epi32(rows[4], rows[5]),
            _mm256_unpackhi_epi32(rows[4], rows[5]),
            _mm256_unpacklo_epi32(rows[6], rows[7]),
            _mm256_unpackhi_epi32(rows[6], rows[7]),
        ];
        let t1 = [
            _mm256_unpacklo_epi64(t0[0], t0[2]),
            _mm256_unpackhi_epi64(t0[0], t0[2]),
            _mm256_unpacklo_epi64(t0[1], t0[3]),
            _mm256_unpackhi_epi64(t0[1], t0[3]),
            _mm256_unpacklo_epi64(t0[4], t0[6]),
            _mm256_unpackhi_epi64(t0[4], t0[6]),
            _mm256_unpacklo_epi64(t0[5], t0[7]),
            _mm256_unpackhi_epi64(t0[5], t0[7]),
        ];
        [
            _mm256_permute2x128_si256::<0x20>(t1[0], t1[4]),
            _mm256_permute2x128_si256::<0x20>(t1[1], t1[5]),
            _mm256_permute2x128_si256::<0x20>(t1[2], t1[6]),
            _mm256_permute2x128_si256::<0x20>(t1[3], t1[7]),
            _mm256_permute2x128_si256::<0x31>(t1[0], t1[4]),
            _mm256_permute2x128_si256::<0x31>(t1[1], t1[5]),
            _mm256_permute2x128_si256::<0x31>(t1[2], t1[6]),
            _mm256_permute2x128_si256::<0x31>(t1[3], t1[7]),
        ]
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn unpack_cv_8(cv: &[__m256i; 8], out: *mut [u8; 32]) {
        let rows = transpose_8x8(cv);
        // SAFETY: caller guarantees `out` points to 8 writable 32-byte slots.
        unsafe {
            for (lane, row) in rows.iter().enumerate() {
                _mm256_storeu_si256(out.add(lane).cast(), *row);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn hash_8_single_block(
        blocks: &[[u8; BLOCK_LEN]; 8],
        block_len: u32,
        out: *mut [u8; 32],
    ) {
        let mut left = [_mm256_setzero_si256(); 8];
        let mut right = [_mm256_setzero_si256(); 8];
        // SAFETY: `blocks` holds 8 contiguous 64-byte arrays.
        unsafe {
            for lane in 0..8 {
                let p = blocks[lane].as_ptr();
                left[lane] = _mm256_loadu_si256(p.cast());
                right[lane] = _mm256_loadu_si256(p.add(32).cast());
            }
        }
        let lo = transpose_8x8(&left);
        let hi = transpose_8x8(&right);
        let msg = [
            lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2], hi[3],
            hi[4], hi[5], hi[6], hi[7],
        ];
        let mut cv = iv_vecs();
        compress_block_8(&mut cv, &msg, block_len, 0, CHUNK_START | CHUNK_END | ROOT);
        // SAFETY: caller guarantees `out` points to 8 writable slots.
        unsafe { unpack_cv_8(&cv, out) };
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn compress_8_pairs(prev_16: &[[u8; 32]], out: *mut [u8; 32]) {
        debug_assert!(prev_16.len() >= 16);
        let mut left = [_mm256_setzero_si256(); 8];
        let mut right = [_mm256_setzero_si256(); 8];
        // SAFETY: `prev_16` has 16 contiguous 32-byte digests.
        unsafe {
            for lane in 0..8 {
                left[lane] = _mm256_loadu_si256(prev_16[2 * lane].as_ptr().cast());
                right[lane] = _mm256_loadu_si256(prev_16[2 * lane + 1].as_ptr().cast());
            }
        }
        let lo = transpose_8x8(&left);
        let hi = transpose_8x8(&right);
        let msg = [
            lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2], hi[3],
            hi[4], hi[5], hi[6], hi[7],
        ];
        let mut cv = iv_vecs();
        compress_block_8(&mut cv, &msg, 64, 0, CHUNK_START | CHUNK_END | ROOT);
        // SAFETY: caller guarantees `out` points to 8 writable slots.
        unsafe { unpack_cv_8(&cv, out) };
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn compress_8_pairs_and_inject(
        prev_16: &[[u8; 32]],
        rows_digests: &[[u8; 32]; 8],
        out: *mut [u8; 32],
    ) {
        debug_assert!(prev_16.len() >= 16);
        let mut left = [_mm256_setzero_si256(); 8];
        let mut right = [_mm256_setzero_si256(); 8];
        let mut row_vecs = [_mm256_setzero_si256(); 8];
        // SAFETY: `prev_16` has 16 digests and `rows_digests` has 8 digests.
        unsafe {
            for lane in 0..8 {
                left[lane] = _mm256_loadu_si256(prev_16[2 * lane].as_ptr().cast());
                right[lane] = _mm256_loadu_si256(prev_16[2 * lane + 1].as_ptr().cast());
                row_vecs[lane] = _mm256_loadu_si256(rows_digests[lane].as_ptr().cast());
            }
        }
        let lo = transpose_8x8(&left);
        let hi = transpose_8x8(&right);
        let msg1 = [
            lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2], hi[3],
            hi[4], hi[5], hi[6], hi[7],
        ];
        let mut parent_cv = iv_vecs();
        compress_block_8(&mut parent_cv, &msg1, 64, 0, CHUNK_START | CHUNK_END | ROOT);

        let right_cv = transpose_8x8(&row_vecs);
        let msg2 = [
            parent_cv[0],
            parent_cv[1],
            parent_cv[2],
            parent_cv[3],
            parent_cv[4],
            parent_cv[5],
            parent_cv[6],
            parent_cv[7],
            right_cv[0],
            right_cv[1],
            right_cv[2],
            right_cv[3],
            right_cv[4],
            right_cv[5],
            right_cv[6],
            right_cv[7],
        ];
        let mut final_cv = iv_vecs();
        compress_block_8(&mut final_cv, &msg2, 64, 0, CHUNK_START | CHUNK_END | ROOT);
        // SAFETY: caller guarantees `out` points to 8 writable slots.
        unsafe { unpack_cv_8(&final_cv, out) };
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hash_chunk_8(
        ptrs: &[*const u8; 8],
        chunk_offset: usize,
        chunk_len: usize,
        counter: u64,
        is_root: bool,
    ) -> [__m256i; 8] {
        let mut cv = iv_vecs();
        let num_blocks = chunk_len.div_ceil(BLOCK_LEN).max(1);
        for b in 0..num_blocks {
            let block_offset = chunk_offset + b * BLOCK_LEN;
            let rem = chunk_len.saturating_sub(b * BLOCK_LEN);
            let block_len = rem.min(BLOCK_LEN);
            let mut left = [_mm256_setzero_si256(); 8];
            let mut right = [_mm256_setzero_si256(); 8];
            // SAFETY: caller guarantees each `ptrs[lane]` has `chunk_offset + chunk_len` readable bytes.
            unsafe {
                if block_len == BLOCK_LEN {
                    for lane in 0..8 {
                        let p = ptrs[lane].add(block_offset);
                        left[lane] = _mm256_loadu_si256(p.cast());
                        right[lane] = _mm256_loadu_si256(p.add(32).cast());
                    }
                } else {
                    for lane in 0..8 {
                        let mut tmp = [0u8; BLOCK_LEN];
                        if block_len > 0 {
                            core::ptr::copy_nonoverlapping(
                                ptrs[lane].add(block_offset),
                                tmp.as_mut_ptr(),
                                block_len,
                            );
                        }
                        left[lane] = _mm256_loadu_si256(tmp.as_ptr().cast());
                        right[lane] = _mm256_loadu_si256(tmp.as_ptr().add(32).cast());
                    }
                }
            }
            let lo = transpose_8x8(&left);
            let hi = transpose_8x8(&right);
            let msg = [
                lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2], hi[3],
                hi[4], hi[5], hi[6], hi[7],
            ];
            let mut flags = 0u32;
            if b == 0 {
                flags |= CHUNK_START;
            }
            if b + 1 == num_blocks {
                flags |= CHUNK_END;
                if is_root {
                    flags |= ROOT;
                }
            }
            compress_block_8(
                &mut cv,
                &msg,
                u32::try_from(block_len).unwrap(),
                counter,
                flags,
            );
        }
        cv
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn compress_parent_8(left: &[__m256i; 8], right: &[__m256i; 8], is_root: bool) -> [__m256i; 8] {
        let msg = [
            left[0], left[1], left[2], left[3], left[4], left[5], left[6], left[7], right[0],
            right[1], right[2], right[3], right[4], right[5], right[6], right[7],
        ];
        let mut cv = iv_vecs();
        let flags = if is_root { PARENT | ROOT } else { PARENT };
        compress_block_8(&mut cv, &msg, 64, 0, flags);
        cv
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn hash_8_fixed_len(
        ptrs: &[*const u8; 8],
        row_bytes: usize,
        out: *mut [u8; 32],
    ) {
        debug_assert!(row_bytes <= super::MAX_MULTIBUF_ROW_BYTES);
        if row_bytes <= CHUNK_LEN {
            // SAFETY: caller guarantees `ptrs` validity for `row_bytes` and `out` has 8 slots.
            unsafe {
                let cv = hash_chunk_8(ptrs, 0, row_bytes, 0, true);
                unpack_cv_8(&cv, out);
            }
            return;
        }
        let num_chunks = row_bytes.div_ceil(CHUNK_LEN);
        let mut cv_stack = [[_mm256_setzero_si256(); 8]; 6];
        let mut stack_len = 0usize;
        for c in 0..num_chunks {
            let chunk_offset = c * CHUNK_LEN;
            let chunk_len = (row_bytes - chunk_offset).min(CHUNK_LEN);
            // SAFETY: caller guarantees `ptrs` validity for `row_bytes`.
            let mut new_cv = unsafe {
                hash_chunk_8(
                    ptrs,
                    chunk_offset,
                    chunk_len,
                    u64::try_from(c).unwrap(),
                    false,
                )
            };
            if c + 1 < num_chunks {
                let mut total_chunks = c + 1;
                while total_chunks & 1 == 0 {
                    stack_len -= 1;
                    new_cv = compress_parent_8(&cv_stack[stack_len], &new_cv, false);
                    total_chunks >>= 1;
                }
                cv_stack[stack_len] = new_cv;
                stack_len += 1;
            } else {
                while stack_len > 1 {
                    stack_len -= 1;
                    new_cv = compress_parent_8(&cv_stack[stack_len], &new_cv, false);
                }
                stack_len -= 1;
                let root_cv = compress_parent_8(&cv_stack[stack_len], &new_cv, true);
                // SAFETY: caller guarantees `out` has 8 writable slots.
                unsafe { unpack_cv_8(&root_cv, out) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Blake3CompressionFunction;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PseudoCompressionFunction, SerializingHasher};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    fn available_backends() -> Vec<SimdBackend> {
        let mut backends = vec![SimdBackend::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                backends.push(SimdBackend::Avx2);
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                backends.push(SimdBackend::Avx512);
            }
        }
        backends
    }

    #[test]
    fn compress_pairs_matches_p3_blake3_all_backends() {
        let mut rng = SmallRng::seed_from_u64(0xB1A3_0001);
        let ref_compress = Blake3CompressionFunction::new(Blake3);
        for count in [
            0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256, 257, 600,
        ] {
            let prev: Vec<[u8; 32]> = (0..2 * count).map(|_| rng.random()).collect();
            let expected: Vec<[u8; 32]> = (0..count)
                .map(|i| ref_compress.compress([prev[2 * i], prev[2 * i + 1]]))
                .collect();

            for backend in available_backends() {
                // SAFETY: `compress_pairs_slice` initializes all `count` slots.
                let got = unsafe {
                    alloc_and_fill_digests(count, count, |dst| {
                        compress_pairs_slice(&prev, dst, backend);
                    })
                };
                assert_eq!(
                    got, expected,
                    "compress_pairs_slice mismatch for {backend:?} count={count}"
                );
            }

            // SAFETY: `compress_pairs_uninit` initializes all `count` slots.
            let got_par = unsafe {
                alloc_and_fill_digests(count, count, |dst| {
                    compress_pairs_uninit(&prev, dst);
                })
            };
            assert_eq!(got_par, expected, "compress_pairs mismatch count={count}");
        }
    }

    #[test]
    fn compress_and_inject_matches_p3_blake3_all_backends() {
        let mut rng = SmallRng::seed_from_u64(0xB1A3_0002);
        let ref_compress = Blake3CompressionFunction::new(Blake3);
        for count in [0usize, 1, 7, 8, 9, 15, 16, 17, 128, 300] {
            let prev: Vec<[u8; 32]> = (0..2 * count).map(|_| rng.random()).collect();
            let rows: Vec<[u8; 32]> = (0..count).map(|_| rng.random()).collect();
            let expected: Vec<[u8; 32]> = (0..count)
                .map(|i| {
                    let p = ref_compress.compress([prev[2 * i], prev[2 * i + 1]]);
                    ref_compress.compress([p, rows[i]])
                })
                .collect();

            for backend in available_backends() {
                // SAFETY: `compress_and_inject_slice` initializes all `count` slots.
                let got = unsafe {
                    alloc_and_fill_digests(count, count, |dst| {
                        compress_and_inject_slice(&prev, &rows, dst, backend);
                    })
                };
                assert_eq!(
                    got, expected,
                    "compress_and_inject_slice mismatch for {backend:?} count={count}"
                );
            }

            // SAFETY: `compress_and_inject_from_digests_uninit` initializes all `count` slots.
            let got_par = unsafe {
                alloc_and_fill_digests(count, count, |dst| {
                    compress_and_inject_from_digests_uninit(&prev, &rows, dst);
                })
            };
            assert_eq!(got_par, expected);
        }
    }

    #[test]
    fn hash_raw_rows_matches_p3_blake3_across_chunk_boundaries() {
        let mut rng = SmallRng::seed_from_u64(0xB1A3_0003);
        let ref_hasher = Blake3;
        let lengths = [
            0usize, 1, 7, 8, 32, 56, 63, 64, 65, 72, 128, 512, 960, 1016, 1024, 1025, 1032, 2048,
            2056, 3072, 4096, 5120, 6144, 7168, 8192, 16384, 16392,
        ];
        for &row_bytes in &lengths {
            for &num_rows in &[0usize, 1, 8, 16, 25] {
                let mut raw = vec![0u8; num_rows * row_bytes];
                rng.fill(&mut raw[..]);
                let expected: Vec<[u8; 32]> = (0..num_rows)
                    .map(|r| ref_hasher.hash_slice(&raw[r * row_bytes..(r + 1) * row_bytes]))
                    .collect();

                for backend in available_backends() {
                    // SAFETY: `hash_raw_rows_slice` initializes all `num_rows` slots.
                    let got = unsafe {
                        alloc_and_fill_digests(num_rows, num_rows, |dst| {
                            hash_raw_rows_slice(&raw, row_bytes, dst, backend);
                        })
                    };
                    assert_eq!(
                        got, expected,
                        "hash_raw_rows_slice mismatch for {backend:?} row_bytes={row_bytes} num_rows={num_rows}"
                    );
                }

                // SAFETY: `hash_raw_rows_uninit` initializes all `num_rows` slots.
                let got_par = unsafe {
                    alloc_and_fill_digests(num_rows, num_rows, |dst| {
                        hash_raw_rows_uninit(&raw, row_bytes, dst);
                    })
                };
                assert_eq!(got_par, expected);
            }
        }
    }

    #[test]
    fn hash_matrix_rows_matches_serializing_hasher_including_non_canonical_values() {
        let mut rng = SmallRng::seed_from_u64(0xB1A3_0004);
        let ref_hasher = SerializingHasher::new(Blake3);
        let height = 35usize;
        // Include non-canonical Goldilocks representations (>= ORDER_U64) to verify canonicalization.
        let m1 = RowMajorMatrix::new(
            (0..height * 5)
                .map(|i| {
                    if i % 7 == 0 {
                        Goldilocks::new(u64::MAX - u64::try_from(i).unwrap())
                    } else {
                        Goldilocks::new(rng.random())
                    }
                })
                .collect(),
            5,
        );
        let m2 = RowMajorMatrix::new(
            (0..height * 130)
                .map(|_| Goldilocks::new(rng.random()))
                .collect(),
            130,
        );
        let expected_single: Vec<[u8; 32]> = (0..height)
            .map(|r| ref_hasher.hash_iter(m1.row_slice(r).unwrap().iter().copied()))
            .collect();
        let mut got_single = vec![[0u8; 32]; height];
        hash_matrix_rows(&[&m1], 0, &mut got_single);
        assert_eq!(got_single, expected_single);

        let expected_multi: Vec<[u8; 32]> = (0..height)
            .map(|r| {
                ref_hasher.hash_iter(
                    m1.row_slice(r)
                        .unwrap()
                        .iter()
                        .chain(m2.row_slice(r).unwrap().iter())
                        .copied(),
                )
            })
            .collect();
        let mut got_multi = vec![[0u8; 32]; height];
        hash_matrix_rows(&[&m1, &m2], 0, &mut got_multi);
        assert_eq!(got_multi, expected_multi);
    }
}
