//! Raw FFI declarations for the kernels in `cuda/gpu_api.cu`.
//!
//! Both functions return [`sppark::Error`]; on success `code == 0`.

use core::ffi::c_void;

use sppark::{Error, NTTDirection, NTTInputOutputOrder, NTTType};

unsafe extern "C" {
    /// Compute a batched forward or inverse NTT in place over a flat
    /// `domain_size * num_cols` buffer. Layout matches Plonky3's
    /// row-major `RowMajorMatrix<F>` with `width = num_cols`.
    pub(super) fn compute_ntt_batch(
        gpu_id: i32,
        inout: *mut c_void,
        lg_domain_size: u32,
        num_cols: u32,
        order: NTTInputOutputOrder,
        direction: NTTDirection,
        ntt_type: NTTType,
    ) -> Error;

    /// Compute a batched coset low-degree extension with a caller-provided
    /// shift. Pipeline (all on device): iDFT → multiply coefficients by
    /// `shift^row` → zero-pad to extended size → forward DFT (NR order, so
    /// output is bit-reversed). `shift_canonical` is the Goldilocks element
    /// in canonical [0, p) form — i.e. `Goldilocks::value` directly.
    ///
    /// Output layout: row-major, `(domain_size << lg_blowup) × num_cols`.
    /// Rows are in bit-reversed order; caller wraps in `BitReversedMatrixView`.
    pub(super) fn compute_coset_lde_with_shift(
        gpu_id: i32,
        out: *mut c_void,
        input: *const c_void,
        lg_domain_size: u32,
        lg_blowup: u32,
        num_cols: u32,
        shift_canonical: u64,
    ) -> Error;

    /// Apply the width-8 Goldilocks-Poseidon2 permutation to a single
    /// state in place. State is 8 canonical-form `u64`s. Used for kernel
    /// validation against Plonky3's CPU `permute_mut`.
    pub(super) fn compute_poseidon2_permute_8(
        gpu_id: i32,
        state: *mut u64,
    ) -> Error;

    /// Apply the width-16 Goldilocks-Poseidon2 permutation to a single
    /// state in place. State is 16 canonical-form `u64`s. Used for kernel
    /// validation against Plonky3's CPU `permute_mut`.
    pub(super) fn compute_poseidon2_permute_16(
        gpu_id: i32,
        state: *mut u64,
    ) -> Error;

    /// Hash each row of a row-major `num_rows × num_cols` Goldilocks
    /// matrix to a 4-element digest using the width-16 Poseidon2 sponge
    /// (rate=12, cap=4, out=4). Output buffer is `num_rows × 4` u64.
    pub(super) fn compute_poseidon2_leaf_hash_rows(
        gpu_id: i32,
        matrix: *const u64,
        num_cols: u32,
        num_rows: u64,
        digests_out: *mut u64,
    ) -> Error;

    /// Pair-compress `2n` consecutive digests into `n` digests.
    /// Each output is `permute_8(left ‖ right)[0..4]`.
    pub(super) fn compute_poseidon2_compress_pairs(
        gpu_id: i32,
        digests: *const u64,
        n: u64,
        out: *mut u64,
    ) -> Error;

    /// Mix-compress two parallel arrays of `n` digests element-wise.
    /// `out[i] = compress(left[i], right[i])`. Used at multi-matrix tree
    /// levels where a smaller matrix's leaf hashes are injected into the
    /// accumulating tree.
    pub(super) fn compute_poseidon2_compress_paired(
        gpu_id: i32,
        left: *const u64,
        right: *const u64,
        n: u64,
        out: *mut u64,
    ) -> Error;
}
