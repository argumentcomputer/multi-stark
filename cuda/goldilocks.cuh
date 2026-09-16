// SPDX-License-Identifier: MIT OR Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace multi_stark_cuda {

constexpr uint64_t GOLDILOCKS_P = 0xffffffff00000001ULL;
constexpr uint64_t GOLDILOCKS_EPSILON = 0x00000000ffffffffULL;

__device__ __forceinline__ uint64_t canonicalize(uint64_t value) {
    return value >= GOLDILOCKS_P ? value - GOLDILOCKS_P : value;
}

__device__ __forceinline__ uint64_t goldilocks_add(uint64_t left, uint64_t right) {
    left = canonicalize(left);
    right = canonicalize(right);
    // Written this way to avoid relying on overflow behavior in the source
    // language. `GOLDILOCKS_P - right` is always representable.
    const uint64_t gap = GOLDILOCKS_P - right;
    return left >= gap ? left - gap : left + right;
}

__device__ __forceinline__ uint64_t goldilocks_sub(uint64_t left, uint64_t right) {
    left = canonicalize(left);
    right = canonicalize(right);
    return left >= right ? left - right : GOLDILOCKS_P - (right - left);
}

__device__ __forceinline__ uint64_t goldilocks_mul(uint64_t left, uint64_t right) {
    left = canonicalize(left);
    right = canonicalize(right);

    const uint64_t low = left * right;
    const uint64_t high = __umul64hi(left, right);

    // Reduce low + high * 2^64 using 2^64 = 2^32 - 1 (mod p).
    const uint64_t high_high = high >> 32;
    const uint64_t high_low = high & GOLDILOCKS_EPSILON;
    uint64_t reduced_low = low - high_high;
    if (low < high_high) {
        // The wrapped subtraction added 2^64; replace that with +p.
        reduced_low -= GOLDILOCKS_EPSILON;
    }
    const uint64_t reduced_high = high_low * GOLDILOCKS_EPSILON;
    return goldilocks_add(reduced_low, reduced_high);
}

__device__ __forceinline__ uint64_t goldilocks_pow(uint64_t base, uint64_t exponent) {
    uint64_t result = 1;
    while (exponent != 0) {
        if ((exponent & 1U) != 0) {
            result = goldilocks_mul(result, base);
        }
        base = goldilocks_mul(base, base);
        exponent >>= 1;
    }
    return result;
}

// The resident PCS canonicalizes every committed LDE, and interpolation
// tables are serialized with `as_canonical_u64`. Restrict the faster
// canonical-input arithmetic to this boundary instead of weakening the
// representation guarantees of lookup/quotient code, whose host inputs may
// legitimately use lazy representatives.
__device__ __forceinline__ uint64_t canonical_add(uint64_t a,uint64_t b){
    const uint64_t sum=a+b;
    if(sum<a)return sum+GOLDILOCKS_EPSILON;
    return sum>=GOLDILOCKS_P?sum-GOLDILOCKS_P:sum;
}
__device__ __forceinline__ uint64_t canonical_mul(uint64_t a,uint64_t b){
    const uint64_t low=a*b,high=__umul64hi(a,b),high_high=high>>32;
    uint64_t reduced_low=low-high_high;
    if(low<high_high)reduced_low-=GOLDILOCKS_EPSILON;
    const uint64_t reduced_high=(high&GOLDILOCKS_EPSILON)*GOLDILOCKS_EPSILON;
    return canonical_add(canonicalize(reduced_low),canonicalize(reduced_high));
}

} // namespace multi_stark_cuda
