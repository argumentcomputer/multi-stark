// Goldilocks NTT/LDE FFI surface, backed by sppark's batched kernels.
//
// NTT entry point (`compute_ntt_batch`) is adapted from Supranational LLC's
// plonky3-accelerate `cuda/gpu/gpu_api.cu` (Apache-2.0).
//
// Coset-LDE entry point (`compute_coset_lde_with_shift`) is fresh, using an
// explicit `shift: fr_t` parameter rather than sppark's hardcoded
// `partial_group_gen_powers`. This lets Plonky3 pass `Val::GENERATOR = 7`
// (the multiplicative-group generator) instead of being constrained to
// sppark's 2-adic generator. Pipeline: iDFT → coset_shift_columns → DFT,
// all on device.
//
// Poseidon Merkle hashing lives in `merkle_api.cu` (Phase 3).
//
// Hardcoded to FEATURE_GOLDILOCKS — multi-stark only targets Goldilocks.

#include <cuda.h>
#include <ff/goldilocks.hpp>
#include <ntt/ntt.cuh>

__launch_bounds__(1024) __global__
void transpose(fr_t* out, const fr_t* in, size_t rows, uint32_t cols)
{
    size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
    if (tid >= rows * cols)
        return;
    uint32_t col_idx = tid / cols;
    size_t   row_idx = tid % cols;
    out[tid] = in[row_idx * rows + col_idx];
}

extern "C"
RustError::by_value compute_ntt_batch(int gpu_id, fr_t* inout,
                                      uint32_t lg_domain_size, uint32_t num_cols,
                                      NTT::InputOutputOrder ntt_order,
                                      NTT::Direction ntt_direction,
                                      NTT::Type ntt_type)
{
    auto& gpu = select_gpu(gpu_id);

    size_t domain_size = (size_t)1 << lg_domain_size;
    size_t total_size = domain_size * num_cols;
    uint32_t nblocks = (total_size + 1024 - 1) / 1024;
    size_t transposed_off = num_cols > 1 ? total_size : 0;

    try {
        dev_ptr_t<fr_t> d_mem(total_size + transposed_off);
        fr_t* d_inout = &d_mem[0];
        fr_t* d_transposed = &d_inout[transposed_off];

        gpu.HtoD(&d_inout[0], &inout[0], total_size);

        if (num_cols > 1) {
            transpose<<<nblocks, 1024, 0, gpu>>>(&d_transposed[0], &d_inout[0],
                                                 num_cols, domain_size);
            CUDA_OK(cudaGetLastError());
        }

        for (size_t i = 0; i < num_cols; i++)
            NTT::Base_dev_ptr(gpu, &d_transposed[i * domain_size], lg_domain_size,
                              ntt_order, ntt_direction, ntt_type);

        if (num_cols > 1) {
            transpose<<<nblocks, 1024, 0, gpu>>>(&d_inout[0], &d_transposed[0],
                                                 domain_size, num_cols);
            CUDA_OK(cudaGetLastError());
        }

        gpu.DtoH(&inout[0], &d_inout[0], total_size);
        gpu.sync();
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

// Per-element exponentiation: state[i] *= shift^i (left-to-right binary).
//
// The matrix is laid out column-major at this point (after the transpose in
// compute_coset_lde_with_shift). Each column has `domain_size` rows. We
// dispatch one thread per (column, row) and have it compute its own
// `shift^row` via O(log row) field multiplications. Per-thread cost is
// negligible compared to the surrounding NTTs.
__launch_bounds__(1024) __global__
void coset_shift_columns_kernel(fr_t* d_data, fr_t shift,
                                size_t domain_size, uint32_t num_cols)
{
    size_t tid = threadIdx.x + blockDim.x * (size_t)blockIdx.x;
    size_t total = domain_size * num_cols;
    if (tid >= total) return;

    size_t col = tid / domain_size;
    size_t row = tid % domain_size;

    // shift^row by binary exponentiation. row fits in u64; loop is at most 64.
    fr_t s = fr_t((uint64_t)1);
    fr_t base = shift;
    uint64_t exp = (uint64_t)row;
    while (exp != 0) {
        if (exp & 1u) s = s * base;
        base = base * base;
        exp >>= 1;
    }

    d_data[col * domain_size + row] = d_data[col * domain_size + row] * s;
}

// Coset LDE with caller-provided shift. Mirrors Plonky3's
// `coset_lde_batch(mat, added_bits, shift)`: iDFT → shift coefficients →
// zero-pad → DFT (forward, NR order so output is bit-reversed). The caller
// wraps the result in `BitReversedMatrixView`.
extern "C"
RustError::by_value compute_coset_lde_with_shift(int gpu_id,
                                                  fr_t* out, const fr_t* in,
                                                  uint32_t lg_domain_size,
                                                  uint32_t lg_blowup,
                                                  uint32_t num_cols,
                                                  uint64_t shift_canonical)
{
    auto& gpu = select_gpu(gpu_id);

    size_t domain_size = (size_t)1 << lg_domain_size;
    size_t ext_domain_size = domain_size << lg_blowup;
    size_t total_size = domain_size * num_cols;
    size_t ext_total_size = ext_domain_size * num_cols;
    uint32_t nblocks_full = (ext_total_size + 1024 - 1) / 1024;
    uint32_t nblocks_shift = (total_size + 1024 - 1) / 1024;
    size_t transposed_off = num_cols > 1 ? ext_total_size : 0;

    // gl64_t(uint64_t) calls to() which canonicalizes; safe to pass any u64.
    fr_t shift = fr_t(shift_canonical);

    try {
        dev_ptr_t<fr_t> d_mem(ext_total_size + transposed_off);
        fr_t* d_inout = &d_mem[0];
        fr_t* d_transposed = &d_inout[transposed_off];

        // 1. Upload the original (pre-LDE-size) matrix at the *end* of the
        //    extended buffer; zero the front. After the iDFT below we'll
        //    have coefficients in the trailing `domain_size` slots, ready
        //    for shift + zero-pad-by-leading-zeros + forward DFT to fill
        //    the full `ext_domain_size`.
        gpu.HtoD(&d_inout[ext_total_size - total_size], &in[0], total_size);
        CUDA_OK(cudaMemsetAsync(&d_inout[0], 0,
                                (ext_total_size - total_size) * sizeof(fr_t),
                                gpu));

        // 2. Transpose to column-major if multi-column (so each column is
        //    contiguous and we can run an NTT per column).
        if (num_cols > 1) {
            transpose<<<nblocks_full, 1024, 0, gpu>>>(
                &d_transposed[0], &d_inout[0], num_cols, ext_domain_size);
            CUDA_OK(cudaGetLastError());
        }

        // 3. iDFT each column's trailing `domain_size` slot.
        //    NN order: input natural, output natural — i.e. the
        //    coefficients that will get coset-shifted and zero-padded.
        for (size_t i = 0; i < num_cols; i++) {
            NTT::Base_dev_ptr(gpu, &d_transposed[i * ext_domain_size +
                                                 ext_domain_size - domain_size],
                              lg_domain_size,
                              NTT::InputOutputOrder::NN,
                              NTT::Direction::inverse,
                              NTT::Type::standard);
        }

        // 4. Pack coefficients to the front of each column (so leading
        //    `domain_size` slots = coefficients, trailing slots = zero
        //    padding, matching the forward-NTT layout of the next step).
        //    For single-column we can do this in-place; for multi-column
        //    each column is contiguous in d_transposed so a strided memcpy
        //    works. We use a small kernel for both.
        for (size_t i = 0; i < num_cols; i++) {
            CUDA_OK(cudaMemcpyAsync(&d_transposed[i * ext_domain_size],
                                    &d_transposed[i * ext_domain_size +
                                                  ext_domain_size - domain_size],
                                    domain_size * sizeof(fr_t),
                                    cudaMemcpyDeviceToDevice, gpu));
            CUDA_OK(cudaMemsetAsync(&d_transposed[i * ext_domain_size + domain_size],
                                    0,
                                    (ext_domain_size - domain_size) * sizeof(fr_t),
                                    gpu));
        }

        // 5. Coset shift: multiply coefficients by shift^row.
        coset_shift_columns_kernel<<<nblocks_shift, 1024, 0, gpu>>>(
            &d_transposed[0], shift, domain_size, num_cols);
        CUDA_OK(cudaGetLastError());

        // 6. Forward DFT each column over the extended domain.
        //    NR order: natural input, bit-reversed output. The caller
        //    wraps the result in BitReversedMatrixView so downstream
        //    code treats it as natural-order evaluations.
        for (size_t i = 0; i < num_cols; i++) {
            NTT::Base_dev_ptr(gpu, &d_transposed[i * ext_domain_size],
                              lg_domain_size + lg_blowup,
                              NTT::InputOutputOrder::NR,
                              NTT::Direction::forward,
                              NTT::Type::standard);
        }

        // 7. Transpose back to row-major.
        if (num_cols > 1) {
            transpose<<<nblocks_full, 1024, 0, gpu>>>(
                &d_inout[0], &d_transposed[0], ext_domain_size, num_cols);
            CUDA_OK(cudaGetLastError());
        }

        gpu.DtoH(&out[0], &d_inout[0], ext_total_size);
        gpu.sync();
    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}
