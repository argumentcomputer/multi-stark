// SPDX-License-Identifier: MIT OR Apache-2.0
//
// First-party CUDA kernels for Goldilocks arithmetic and batched radix-2
// transforms. The ABI accepts host buffers; device residency is intentionally
// deferred to the later PCS/FRI backend.

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <new>

namespace {

constexpr uint64_t GOLDILOCKS_P = 0xffffffff00000001ULL;
constexpr uint64_t GOLDILOCKS_EPSILON = 0x00000000ffffffffULL;
constexpr unsigned int THREADS = 256;
constexpr unsigned int MAX_BLOCKS = 65535;
constexpr uint32_t BLAKE3_CHUNK_START = 1U << 0;
constexpr uint32_t BLAKE3_CHUNK_END = 1U << 1;
constexpr uint32_t BLAKE3_PARENT = 1U << 2;
constexpr uint32_t BLAKE3_ROOT = 1U << 3;
// All work currently uses CUDA's per-thread default stream. Stream-ordered
// allocation preserves the same dependency order while avoiding cudaFree's
// device-wide synchronization on thousands of short-lived PCS buffers.
inline cudaError_t persistent_malloc(void** pointer,size_t bytes){return cudaMalloc(pointer,bytes);}
inline cudaError_t persistent_free(void* pointer){return pointer?cudaFree(pointer):cudaSuccess;}
inline cudaError_t stream_malloc(void** pointer,size_t bytes){return cudaMallocAsync(pointer,bytes,cudaStreamPerThread);}
inline cudaError_t stream_free(void* pointer){return pointer?cudaFreeAsync(pointer,cudaStreamPerThread):cudaSuccess;}
#define cudaMalloc stream_malloc
#define cudaFree stream_free

struct ConstantCacheEntry{int device=-1;size_t count=0;uint64_t kind=0,key0=0,key1=0;uint64_t* values=nullptr;ConstantCacheEntry* next=nullptr;};
ConstantCacheEntry* constant_cache=nullptr;volatile int constant_cache_lock=0;
cudaError_t cached_device_constants(int device,const uint64_t* host,size_t count,
    uint64_t kind,uint64_t key0,uint64_t key1,const uint64_t** output){
    while(__sync_lock_test_and_set(&constant_cache_lock,1)){}
    for(auto* entry=constant_cache;entry;entry=entry->next)if(entry->device==device&&
        entry->count==count&&entry->kind==kind&&entry->key0==key0&&entry->key1==key1){
        *output=entry->values;__sync_lock_release(&constant_cache_lock);return cudaSuccess;}
    auto* entry=new(std::nothrow) ConstantCacheEntry;
    if(!entry){__sync_lock_release(&constant_cache_lock);return cudaErrorMemoryAllocation;}
    entry->device=device;entry->count=count;entry->kind=kind;entry->key0=key0;entry->key1=key1;
    cudaError_t status=persistent_malloc(reinterpret_cast<void**>(&entry->values),count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMemcpy(entry->values,host,count*sizeof(uint64_t),cudaMemcpyHostToDevice);
    if(status==cudaSuccess){*output=entry->values;entry->next=constant_cache;constant_cache=entry;}
    else{persistent_free(entry->values);delete entry;}
    __sync_lock_release(&constant_cache_lock);return status;
}
__device__ __constant__ uint32_t BLAKE3_IV[8] = {
    0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
    0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U,
};
__device__ __constant__ unsigned int BLAKE3_PERMUTATION[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8,
};

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

__device__ __forceinline__ uint32_t rotate_right(uint32_t value,
                                                  unsigned int count) {
    return __funnelshift_r(value, value, count);
}

__device__ __forceinline__ void blake3_g(uint32_t state[16], unsigned int a,
                                         unsigned int b, unsigned int c,
                                         unsigned int d, uint32_t x,
                                         uint32_t y) {
    state[a] = state[a] + state[b] + x;
    state[d] = rotate_right(state[d] ^ state[a], 16);
    state[c] += state[d];
    state[b] = rotate_right(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + y;
    state[d] = rotate_right(state[d] ^ state[a], 8);
    state[c] += state[d];
    state[b] = rotate_right(state[b] ^ state[c], 7);
}

__device__ __forceinline__ void blake3_compress(
    const uint32_t chaining_value[8], const uint32_t block[16],
    uint64_t counter, uint32_t block_length, uint32_t flags,
    uint32_t output[16]) {
    uint32_t state[16];
    uint32_t message[16];
#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        state[i] = chaining_value[i];
        state[i + 8] = BLAKE3_IV[i];
    }
    state[12] = static_cast<uint32_t>(counter);
    state[13] = static_cast<uint32_t>(counter >> 32);
    state[14] = block_length;
    state[15] = flags;
#pragma unroll
    for (unsigned int i = 0; i < 16; ++i) {
        message[i] = block[i];
    }

#pragma unroll
    for (unsigned int round = 0; round < 7; ++round) {
        blake3_g(state, 0, 4, 8, 12, message[0], message[1]);
        blake3_g(state, 1, 5, 9, 13, message[2], message[3]);
        blake3_g(state, 2, 6, 10, 14, message[4], message[5]);
        blake3_g(state, 3, 7, 11, 15, message[6], message[7]);
        blake3_g(state, 0, 5, 10, 15, message[8], message[9]);
        blake3_g(state, 1, 6, 11, 12, message[10], message[11]);
        blake3_g(state, 2, 7, 8, 13, message[12], message[13]);
        blake3_g(state, 3, 4, 9, 14, message[14], message[15]);
        if (round != 6) {
            uint32_t permuted[16];
#pragma unroll
            for (unsigned int i = 0; i < 16; ++i) {
                permuted[i] = message[BLAKE3_PERMUTATION[i]];
            }
#pragma unroll
            for (unsigned int i = 0; i < 16; ++i) {
                message[i] = permuted[i];
            }
        }
    }

#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        output[i] = state[i] ^ state[i + 8];
        output[i + 8] = state[i + 8] ^ chaining_value[i];
    }
}

__device__ __forceinline__ uint32_t load_u32_le(const uint8_t* bytes,
                                                 size_t available,
                                                 size_t offset) {
    uint32_t result = 0;
#pragma unroll
    for (unsigned int byte = 0; byte < 4; ++byte) {
        if (offset + byte < available) {
            result |= static_cast<uint32_t>(bytes[offset + byte]) << (8 * byte);
        }
    }
    return result;
}

__device__ __forceinline__ void blake3_chunk(
    const uint8_t* bytes, size_t length, uint64_t chunk_counter, bool root,
    uint32_t output[8]) {
    uint32_t chaining_value[8];
#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        chaining_value[i] = BLAKE3_IV[i];
    }
    const size_t blocks = length == 0 ? 1 : (length + 63) / 64;
    for (size_t block_index = 0; block_index < blocks; ++block_index) {
        const size_t block_offset = block_index * 64;
        const size_t block_length =
            block_offset < length
                ? ((length - block_offset) < 64 ? length - block_offset : 64)
                : 0;
        uint32_t block[16];
#pragma unroll
        for (unsigned int word = 0; word < 16; ++word) {
            block[word] = load_u32_le(bytes + block_offset, block_length,
                                      static_cast<size_t>(word) * 4);
        }
        uint32_t flags = block_index == 0 ? BLAKE3_CHUNK_START : 0;
        const bool last = block_index + 1 == blocks;
        if (last) {
            flags |= BLAKE3_CHUNK_END;
            if (root) {
                flags |= BLAKE3_ROOT;
            }
        }
        uint32_t compressed[16];
        blake3_compress(chaining_value, block, chunk_counter,
                        static_cast<uint32_t>(block_length), flags, compressed);
        if (last) {
#pragma unroll
            for (unsigned int i = 0; i < 8; ++i) {
                output[i] = compressed[i];
            }
        } else {
#pragma unroll
            for (unsigned int i = 0; i < 8; ++i) {
                chaining_value[i] = compressed[i];
            }
        }
    }
}

__device__ __forceinline__ void blake3_parent(const uint32_t left[8],
                                               const uint32_t right[8],
                                               bool root, uint32_t output[8]) {
    uint32_t block[16];
#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        block[i] = left[i];
        block[i + 8] = right[i];
    }
    uint32_t compressed[16];
    blake3_compress(BLAKE3_IV, block, 0, 64,
                    BLAKE3_PARENT | (root ? BLAKE3_ROOT : 0), compressed);
#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        output[i] = compressed[i];
    }
}

__device__ __forceinline__ void blake3_hash_digest_pair(
    const uint8_t* left, const uint8_t* right, uint8_t* digest) {
    uint32_t block[16];
#pragma unroll
    for (unsigned int word = 0; word < 8; ++word) {
        block[word] = load_u32_le(left, 32, static_cast<size_t>(word) * 4);
        block[word + 8] =
            load_u32_le(right, 32, static_cast<size_t>(word) * 4);
    }
    uint32_t compressed[16];
    blake3_compress(BLAKE3_IV, block, 0, 64,
                    BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT,
                    compressed);
#pragma unroll
    for (unsigned int word = 0; word < 8; ++word) {
        const uint32_t value = compressed[word];
        digest[word * 4] = static_cast<uint8_t>(value);
        digest[word * 4 + 1] = static_cast<uint8_t>(value >> 8);
        digest[word * 4 + 2] = static_cast<uint8_t>(value >> 16);
        digest[word * 4 + 3] = static_cast<uint8_t>(value >> 24);
    }
}

__device__ __forceinline__ size_t reverse_index_bits(size_t index, unsigned int bits) {
    if (bits == 0) {
        return 0;
    }
    return static_cast<size_t>(__brevll(static_cast<unsigned long long>(index)) >> (64 - bits));
}

unsigned int blocks_for(size_t work_items) {
    if (work_items == 0) {
        return 0;
    }
    const size_t required = (work_items + THREADS - 1) / THREADS;
    return static_cast<unsigned int>(required < MAX_BLOCKS ? required : MAX_BLOCKS);
}

bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool product_fits(size_t left, size_t right) {
    return left == 0 || right <= SIZE_MAX / left;
}

unsigned int strict_log2(size_t value) {
    unsigned int result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    cudaError_t allocate(size_t elements) {
        if (!product_fits(elements, sizeof(uint64_t))) {
            return cudaErrorInvalidValue;
        }
        return cudaMalloc(reinterpret_cast<void**>(&pointer_), elements * sizeof(uint64_t));
    }

    uint64_t* get() { return pointer_; }
    const uint64_t* get() const { return pointer_; }

  private:
    uint64_t* pointer_ = nullptr;
};

// cudaMemcpy from ordinary Vec storage uses an internal pageable-memory
// staging buffer.  Large LDEs are dominated by that staging copy, so pin the
// caller's matrix for the duration of the synchronous operation.  Registration
// is deliberately best-effort: constrained hosts can reject pinning without
// making the CUDA backend unusable, in which case cudaMemcpy retains its
// normal pageable fallback.
class HostRegistration {
  public:
    HostRegistration(void* pointer, size_t bytes) : pointer_(pointer) {
        if (pointer != nullptr && bytes != 0) {
            registered_ = cudaHostRegister(pointer, bytes, cudaHostRegisterDefault) ==
                          cudaSuccess;
            if (!registered_) {
                // Do not let an optional registration failure poison the
                // thread-local CUDA error observed by the real operation.
                cudaGetLastError();
            }
        }
    }

    HostRegistration(const HostRegistration&) = delete;
    HostRegistration& operator=(const HostRegistration&) = delete;

    ~HostRegistration() {
        if (registered_) {
            cudaHostUnregister(pointer_);
        }
    }

  private:
    void* pointer_ = nullptr;
    bool registered_ = false;
};

struct ResidentMerkleTree {
    uint8_t* rows = nullptr;
    uint8_t* digests = nullptr;
    size_t row_bytes = 0;
    size_t row_count = 0;

    ~ResidentMerkleTree() {
        if (digests != nullptr) {
            cudaFree(digests);
        }
        if (rows != nullptr) {
            cudaFree(rows);
        }
    }
};

struct ResidentMixedMerkleTree {
    uint8_t* digests = nullptr;
    size_t row_count = 0;

    ~ResidentMixedMerkleTree() {
        if (digests != nullptr) {
            cudaFree(digests);
        }
    }
};

struct ResidentLde {
    uint64_t* values = nullptr;
    bool values_managed = false;
    // Original row-major evaluations, retained for prover stages (notably
    // lookup construction) which consume the witness after its LDE has been
    // committed. Null for LDEs synthesized directly on the device.
    uint64_t* trace_values = nullptr;
    const uint64_t* host_trace_values = nullptr;
    bool host_trace_registered = false;
    size_t trace_height = 0;
    size_t height = 0;
    size_t width = 0;
    uint8_t* interpolation_scratch = nullptr;
    size_t interpolation_scratch_bytes = 0;

    ~ResidentLde() {
        if (values != nullptr) {
            if (values_managed) persistent_free(values);
            else cudaFree(values);
        }
        if (trace_values != nullptr) cudaFree(trace_values);
        if (host_trace_registered) cudaHostUnregister(
            const_cast<uint64_t*>(host_trace_values));
        if (interpolation_scratch != nullptr) cudaFree(interpolation_scratch);
    }
};

// Kernels read ResidentLde metadata directly. Keep that small control block in
// CUDA managed memory so device access does not depend on Linux HMM support for
// arbitrary pageable host allocations.
cudaError_t create_resident_lde(ResidentLde** output) {
    if (output == nullptr) return cudaErrorInvalidValue;
    *output = nullptr;
    void* storage = nullptr;
    const cudaError_t status = cudaMallocManaged(&storage, sizeof(ResidentLde));
    if (status != cudaSuccess) return status;
    *output = new (storage) ResidentLde;
    return cudaSuccess;
}

cudaError_t destroy_resident_lde(ResidentLde* lde) {
    if (lde == nullptr) return cudaSuccess;
    lde->~ResidentLde();
    return cudaFree(lde);
}

// Stable C ABI instruction used by Rust's compiled constraint DAG. All
// expressions are base-field-only by this stage.
struct ConstraintNode {
    uint64_t value;
    uint32_t a;
    uint32_t b;
    uint32_t op;
    uint32_t aux;
    uint32_t out;
};

struct ConstraintLookup {
    uint32_t multiplicity;
    uint32_t arg_start;
    uint32_t arg_count;
    uint32_t emit_after;
    uint32_t output;
};

struct Ext2 { uint64_t c0; uint64_t c1; };

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
__device__ __forceinline__ Ext2 canonical_ext2_add(Ext2 a,Ext2 b){
    return {canonical_add(a.c0,b.c0),canonical_add(a.c1,b.c1)};
}
__device__ __forceinline__ Ext2 canonical_ext2_mul(Ext2 a,Ext2 b,uint64_t w){
    const uint64_t p0=canonical_mul(a.c0,b.c0),p1=canonical_mul(a.c1,b.c1);
    return {canonical_add(p0,canonical_mul(w,p1)),
            canonical_add(canonical_mul(a.c0,b.c1),canonical_mul(a.c1,b.c0))};
}

__device__ __forceinline__ Ext2 ext2_add(Ext2 a, Ext2 b) {
    return {goldilocks_add(a.c0, b.c0), goldilocks_add(a.c1, b.c1)};
}
__device__ __forceinline__ Ext2 ext2_sub(Ext2 a, Ext2 b) {
    return {goldilocks_sub(a.c0, b.c0), goldilocks_sub(a.c1, b.c1)};
}
__device__ __forceinline__ Ext2 ext2_mul(Ext2 a, Ext2 b, uint64_t w) {
    const uint64_t p0 = goldilocks_mul(a.c0, b.c0);
    const uint64_t p1 = goldilocks_mul(a.c1, b.c1);
    return {goldilocks_add(p0, goldilocks_mul(w, p1)),
            goldilocks_add(goldilocks_mul(a.c0, b.c1),
                           goldilocks_mul(a.c1, b.c0))};
}

struct ResidentReducedOpening {
    Ext2* values=nullptr; uint8_t* scratch=nullptr; size_t scratch_bytes=0; size_t height=0;
    ~ResidentReducedOpening(){if(values)cudaFree(values);if(scratch)cudaFree(scratch);}
};

struct ResidentFriWorkspace {
    Ext2* inv_denoms=nullptr; uint64_t* coset=nullptr; Ext2* output=nullptr;
    Ext2* alpha=nullptr; Ext2* interpolation_partials=nullptr;
    size_t inv_count=0; size_t coset_count=0;
    size_t output_capacity=0; size_t alpha_capacity=0,partial_capacity=0;
    ~ResidentFriWorkspace(){if(inv_denoms)cudaFree(inv_denoms);if(coset)cudaFree(coset);if(output)cudaFree(output);if(alpha)cudaFree(alpha);if(interpolation_partials)cudaFree(interpolation_partials);}
};

struct InterpolationTask {
    const void* lde; size_t height; size_t inv_offset; size_t output_offset;
    uint64_t scale0; uint64_t scale1;
};

struct ReductionTask {
    void* reduced; const void* lde; size_t height; size_t inv_offset;
    uint64_t y0; uint64_t y1; uint64_t offset0; uint64_t offset1;
};

__global__ void interpolate_lde_columns(Ext2* output, const ResidentLde* lde,
    const Ext2* inv_denoms, const uint64_t* coset, size_t height, Ext2 scale,
    uint64_t ext_w) {
    extern __shared__ Ext2 sums[]; const size_t column=blockIdx.x; Ext2 sum{0,0};
    for(size_t row=threadIdx.x;row<height;row+=blockDim.x){
        Ext2 weight=ext2_mul(inv_denoms[row],{coset[row],0},ext_w);
        Ext2 term={goldilocks_mul(weight.c0,lde->values[row*lde->width+column]),goldilocks_mul(weight.c1,lde->values[row*lde->width+column])};
        sum=ext2_add(sum,term);
    } sums[threadIdx.x]=sum;__syncthreads();
    for(unsigned int stride=blockDim.x/2;stride;stride>>=1){if(threadIdx.x<stride)sums[threadIdx.x]=ext2_add(sums[threadIdx.x],sums[threadIdx.x+stride]);__syncthreads();}
    if(threadIdx.x==0)output[column]=ext2_mul(sums[0],scale,ext_w);
}

constexpr unsigned int INTERPOLATION_COLUMNS=32;
constexpr unsigned int INTERPOLATION_LANES=8;
constexpr size_t INTERPOLATION_ROWS=512;
__global__ void interpolate_lde_partials(Ext2* partials,const ResidentLde* lde,
    const Ext2* inv_denoms,const uint64_t* coset,size_t height,uint64_t ext_w){
    __shared__ Ext2 sums[INTERPOLATION_LANES][INTERPOLATION_COLUMNS];
    __shared__ Ext2 weights[INTERPOLATION_ROWS];
    const size_t column=static_cast<size_t>(blockIdx.x)*INTERPOLATION_COLUMNS+threadIdx.x;
    const size_t row_begin=static_cast<size_t>(blockIdx.y)*INTERPOLATION_ROWS;
    const size_t row_end=row_begin+INTERPOLATION_ROWS<height?row_begin+INTERPOLATION_ROWS:height;
    const size_t linear_thread=threadIdx.y*INTERPOLATION_COLUMNS+threadIdx.x;
    for(size_t offset=linear_thread;row_begin+offset<row_end;offset+=INTERPOLATION_COLUMNS*INTERPOLATION_LANES)
        weights[offset]=canonical_ext2_mul(inv_denoms[row_begin+offset],{coset[row_begin+offset],0},ext_w);
    __syncthreads();
    Ext2 sum{0,0};
    if(column<lde->width){
        for(size_t row=row_begin+threadIdx.y;row<row_end;row+=INTERPOLATION_LANES){
            const Ext2 weight=weights[row-row_begin];
            const uint64_t value=lde->values[row*lde->width+column];
            sum.c0=canonical_add(sum.c0,canonical_mul(weight.c0,value));
            sum.c1=canonical_add(sum.c1,canonical_mul(weight.c1,value));
        }
    }
    sums[threadIdx.y][threadIdx.x]=sum;__syncthreads();
    for(unsigned int stride=INTERPOLATION_LANES/2;stride;stride>>=1){
        if(threadIdx.y<stride)sums[threadIdx.y][threadIdx.x]=canonical_ext2_add(
            sums[threadIdx.y][threadIdx.x],sums[threadIdx.y+stride][threadIdx.x]);
        __syncthreads();
    }
    if(threadIdx.y==0&&column<lde->width)
        partials[static_cast<size_t>(blockIdx.y)*lde->width+column]=sums[0][threadIdx.x];
}

__global__ void interpolate_lde_partials2(Ext2* partials0,Ext2* partials1,
    const ResidentLde* lde,const Ext2* inv0,const Ext2* inv1,
    const uint64_t* coset,size_t height,uint64_t ext_w){
    __shared__ Ext2 sums0[INTERPOLATION_LANES][INTERPOLATION_COLUMNS];
    __shared__ Ext2 sums1[INTERPOLATION_LANES][INTERPOLATION_COLUMNS];
    __shared__ Ext2 weights0[INTERPOLATION_ROWS];
    __shared__ Ext2 weights1[INTERPOLATION_ROWS];
    const size_t column=static_cast<size_t>(blockIdx.x)*INTERPOLATION_COLUMNS+threadIdx.x;
    const size_t row_begin=static_cast<size_t>(blockIdx.y)*INTERPOLATION_ROWS;
    const size_t row_end=row_begin+INTERPOLATION_ROWS<height?row_begin+INTERPOLATION_ROWS:height;
    const size_t linear_thread=threadIdx.y*INTERPOLATION_COLUMNS+threadIdx.x;
    for(size_t offset=linear_thread;row_begin+offset<row_end;offset+=INTERPOLATION_COLUMNS*INTERPOLATION_LANES){
        const size_t row=row_begin+offset;const Ext2 x{coset[row],0};
        weights0[offset]=canonical_ext2_mul(inv0[row],x,ext_w);
        weights1[offset]=canonical_ext2_mul(inv1[row],x,ext_w);
    }
    __syncthreads();Ext2 sum0{0,0},sum1{0,0};
    if(column<lde->width){
        for(size_t row=row_begin+threadIdx.y;row<row_end;row+=INTERPOLATION_LANES){
            const uint64_t value=lde->values[row*lde->width+column];
            const Ext2 w0=weights0[row-row_begin],w1=weights1[row-row_begin];
            sum0.c0=canonical_add(sum0.c0,canonical_mul(w0.c0,value));
            sum0.c1=canonical_add(sum0.c1,canonical_mul(w0.c1,value));
            sum1.c0=canonical_add(sum1.c0,canonical_mul(w1.c0,value));
            sum1.c1=canonical_add(sum1.c1,canonical_mul(w1.c1,value));
        }
    }
    sums0[threadIdx.y][threadIdx.x]=sum0;sums1[threadIdx.y][threadIdx.x]=sum1;
    __syncthreads();
    for(unsigned int stride=INTERPOLATION_LANES/2;stride;stride>>=1){
        if(threadIdx.y<stride){
            sums0[threadIdx.y][threadIdx.x]=canonical_ext2_add(sums0[threadIdx.y][threadIdx.x],sums0[threadIdx.y+stride][threadIdx.x]);
            sums1[threadIdx.y][threadIdx.x]=canonical_ext2_add(sums1[threadIdx.y][threadIdx.x],sums1[threadIdx.y+stride][threadIdx.x]);
        }
        __syncthreads();
    }
    if(threadIdx.y==0&&column<lde->width){const size_t at=static_cast<size_t>(blockIdx.y)*lde->width+column;
        partials0[at]=sums0[0][threadIdx.x];partials1[at]=sums1[0][threadIdx.x];}
}

__global__ void finish_lde_interpolation(Ext2* output,const Ext2* partials,
    size_t partial_rows,size_t width,Ext2 scale,uint64_t ext_w){
    extern __shared__ Ext2 sums[];const size_t column=blockIdx.x;Ext2 sum{0,0};
    for(size_t row=threadIdx.x;row<partial_rows;row+=blockDim.x)
        sum=canonical_ext2_add(sum,partials[row*width+column]);
    sums[threadIdx.x]=sum;__syncthreads();
    for(unsigned int stride=blockDim.x/2;stride;stride>>=1){
        if(threadIdx.x<stride)sums[threadIdx.x]=canonical_ext2_add(sums[threadIdx.x],sums[threadIdx.x+stride]);
        __syncthreads();
    }
    if(threadIdx.x==0)output[column]=canonical_ext2_mul(sums[0],scale,ext_w);
}

struct InterpolationBlockDesc {
    const uint64_t* values;
    const uint64_t* coset;
    const Ext2* inv0;
    const Ext2* inv1;
    Ext2* partial0;
    Ext2* partial1;
    size_t width;
    size_t column_begin;
    size_t row_begin;
    size_t row_end;
};

struct InterpolationFinishDesc {
    Ext2* output;
    const Ext2* partials;
    size_t partial_rows;
    size_t width;
    size_t column;
    Ext2 scale;
};

__global__ void interpolate_lde_blocks(const InterpolationBlockDesc* blocks,
    uint64_t ext_w) {
    const InterpolationBlockDesc d=blocks[blockIdx.x];
    __shared__ Ext2 sums0[INTERPOLATION_LANES][INTERPOLATION_COLUMNS];
    __shared__ Ext2 sums1[INTERPOLATION_LANES][INTERPOLATION_COLUMNS];
    __shared__ Ext2 weights0[INTERPOLATION_ROWS];
    __shared__ Ext2 weights1[INTERPOLATION_ROWS];
    const size_t column=d.column_begin+threadIdx.x;
    const size_t linear_thread=threadIdx.y*INTERPOLATION_COLUMNS+threadIdx.x;
    for(size_t offset=linear_thread;d.row_begin+offset<d.row_end;
        offset+=INTERPOLATION_COLUMNS*INTERPOLATION_LANES){
        const size_t row=d.row_begin+offset;const Ext2 x{d.coset[row],0};
        weights0[offset]=canonical_ext2_mul(d.inv0[row],x,ext_w);
        if(d.inv1)weights1[offset]=canonical_ext2_mul(d.inv1[row],x,ext_w);
    }
    __syncthreads();Ext2 sum0{0,0},sum1{0,0};
    if(column<d.width){
        for(size_t row=d.row_begin+threadIdx.y;row<d.row_end;row+=INTERPOLATION_LANES){
            const uint64_t value=d.values[row*d.width+column];
            const Ext2 w0=weights0[row-d.row_begin];
            sum0.c0=canonical_add(sum0.c0,canonical_mul(w0.c0,value));
            sum0.c1=canonical_add(sum0.c1,canonical_mul(w0.c1,value));
            if(d.inv1){const Ext2 w1=weights1[row-d.row_begin];
                sum1.c0=canonical_add(sum1.c0,canonical_mul(w1.c0,value));
                sum1.c1=canonical_add(sum1.c1,canonical_mul(w1.c1,value));}
        }
    }
    sums0[threadIdx.y][threadIdx.x]=sum0;sums1[threadIdx.y][threadIdx.x]=sum1;
    __syncthreads();
    for(unsigned int stride=INTERPOLATION_LANES/2;stride;stride>>=1){
        if(threadIdx.y<stride){
            sums0[threadIdx.y][threadIdx.x]=canonical_ext2_add(sums0[threadIdx.y][threadIdx.x],sums0[threadIdx.y+stride][threadIdx.x]);
            if(d.inv1)sums1[threadIdx.y][threadIdx.x]=canonical_ext2_add(sums1[threadIdx.y][threadIdx.x],sums1[threadIdx.y+stride][threadIdx.x]);
        }
        __syncthreads();
    }
    if(threadIdx.y==0&&column<d.width){const size_t at=(d.row_begin/INTERPOLATION_ROWS)*d.width+column;
        d.partial0[at]=sums0[0][threadIdx.x];
        if(d.inv1)d.partial1[at]=sums1[0][threadIdx.x];}
}

__global__ void finish_lde_interpolation_blocks(const InterpolationFinishDesc* finishes,
    uint64_t ext_w) {
    const InterpolationFinishDesc d=finishes[blockIdx.x];
    extern __shared__ Ext2 sums[];Ext2 sum{0,0};
    for(size_t row=threadIdx.x;row<d.partial_rows;row+=blockDim.x)
        sum=canonical_ext2_add(sum,d.partials[row*d.width+d.column]);
    sums[threadIdx.x]=sum;__syncthreads();
    for(unsigned int stride=blockDim.x/2;stride;stride>>=1){
        if(threadIdx.x<stride)sums[threadIdx.x]=canonical_ext2_add(sums[threadIdx.x],sums[threadIdx.x+stride]);
        __syncthreads();
    }
    if(threadIdx.x==0)*d.output=canonical_ext2_mul(sums[0],d.scale,ext_w);
}

__global__ void accumulate_reduced_opening(Ext2* output,const ResidentLde* lde,
    const Ext2* inv_denoms,const Ext2* alpha_powers,size_t height,Ext2 reduced_y,
    Ext2 alpha_offset,uint64_t ext_w){
    for(size_t row=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;row<height;row+=static_cast<size_t>(gridDim.x)*blockDim.x){
        Ext2 compressed{0,0};for(size_t column=0;column<lde->width;++column){const uint64_t v=lde->values[row*lde->width+column];compressed.c0=goldilocks_add(compressed.c0,goldilocks_mul(alpha_powers[column].c0,v));compressed.c1=goldilocks_add(compressed.c1,goldilocks_mul(alpha_powers[column].c1,v));}
        const Ext2 term=ext2_mul(alpha_offset,ext2_mul(ext2_sub(reduced_y,compressed),inv_denoms[row],ext_w),ext_w);output[row]=ext2_add(output[row],term);
    }
}

__global__ void fold_fri_ext2(Ext2* output,const Ext2* input,uint64_t* powers,
    size_t height,Ext2 beta,uint64_t ext_w){
    for(size_t row=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;row<height;row+=static_cast<size_t>(gridDim.x)*blockDim.x){
        const Ext2 lo=input[2*row],hi=input[2*row+1];
        const Ext2 even={goldilocks_mul(goldilocks_add(lo.c0,hi.c0),0x7fffffff80000001ULL),goldilocks_mul(goldilocks_add(lo.c1,hi.c1),0x7fffffff80000001ULL)};
        Ext2 odd=ext2_mul(ext2_sub(lo,hi),beta,ext_w);odd.c0=goldilocks_mul(odd.c0,powers[row]);odd.c1=goldilocks_mul(odd.c1,powers[row]);output[row]=ext2_add(even,odd);
    }
}
__global__ void init_fri_powers(uint64_t* powers,size_t height,uint64_t g_inv){
    const unsigned int bits=static_cast<unsigned int>(__ffsll(height)-1);
    for(size_t row=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;row<height;row+=static_cast<size_t>(gridDim.x)*blockDim.x){
        const uint64_t exponent=bits==0?0:__brevll(static_cast<unsigned long long>(row))>>(64U-bits);
        powers[row]=goldilocks_mul(0x7fffffff80000001ULL,goldilocks_pow(g_inv,exponent));
    }
}
__global__ void add_scaled_ext2(Ext2* output,const Ext2* input,size_t count,Ext2 scale,uint64_t ext_w){
    for(size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=static_cast<size_t>(gridDim.x)*blockDim.x)output[i]=ext2_add(output[i],ext2_mul(scale,input[i],ext_w));
}

__device__ __forceinline__ Ext2 ext2_inverse(Ext2 value,uint64_t ext_w){
    const uint64_t norm=goldilocks_sub(goldilocks_mul(value.c0,value.c0),goldilocks_mul(ext_w,goldilocks_mul(value.c1,value.c1)));
    const uint64_t inv=goldilocks_pow(norm,GOLDILOCKS_P-2);
    return {goldilocks_mul(value.c0,inv),goldilocks_sub(0,goldilocks_mul(value.c1,inv))};
}
__global__ void lookup_messages(Ext2* conjugates,uint64_t* norms,const uint64_t* args,
    const size_t* arg_offsets,size_t height,size_t num_lookups,size_t args_width,
    Ext2 beta,Ext2 gamma,uint64_t ext_w){
    const size_t count=height*num_lookups;
    for(size_t index=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;index<count;index+=static_cast<size_t>(gridDim.x)*blockDim.x){
        const size_t row=index/num_lookups,lookup=index%num_lookups;Ext2 fingerprint{0,0};
        for(size_t j=arg_offsets[lookup+1];j>arg_offsets[lookup];--j)fingerprint=ext2_add(ext2_mul(fingerprint,gamma,ext_w),{args[row*args_width+j-1],0});
        const Ext2 message=ext2_add(beta,fingerprint);conjugates[index]={message.c0,goldilocks_sub(0,message.c1)};
        norms[index]=goldilocks_sub(goldilocks_mul(message.c0,message.c0),goldilocks_mul(ext_w,goldilocks_mul(message.c1,message.c1)));
    }
}
// Evaluate the lookup-only prefix of a circuit DAG against the original
// row-major witness retained by the stage-1 LDE. This avoids materializing
// and uploading the much wider concrete LookupValues buffer.
__global__ void lookup_messages_graph(
    Ext2* conjugates,uint64_t* norms,uint64_t* multiplicities,
    const ConstraintNode* nodes,size_t node_count,size_t slot_count,
    const ConstraintLookup* lookups,size_t lookup_count,const uint32_t* lookup_args,
    const ResidentLde* preprocessed,const uint64_t* main_trace,size_t main_width,
    bool main_trace_is_chunk,
    Ext2 beta,Ext2 gamma,uint64_t ext_w,size_t height,size_t row_start,
    size_t row_count,uint64_t* global_scratch){
    extern __shared__ uint64_t shared_values[];
    const size_t lane=threadIdx.x,tile=blockDim.x;
    uint64_t* values=global_scratch?global_scratch+static_cast<size_t>(blockIdx.x)*slot_count*tile:shared_values;
    for(size_t local_row=static_cast<size_t>(blockIdx.x)*tile+lane;local_row<row_count;local_row+=static_cast<size_t>(gridDim.x)*tile){
        const size_t row=row_start+local_row;
        const size_t next=(row+1)&(height-1);
        size_t next_lookup=0;
        for(size_t i=0;i<node_count;++i){const ConstraintNode n=nodes[i];
            const uint64_t a=n.op>=6?values[n.a*tile+lane]:0;
            const uint64_t b=n.op>=6&&n.op!=9?values[n.b*tile+lane]:0;uint64_t v=0;
            switch(n.op){
                case 0:v=n.value;break;
                case 1:{const bool prep_column=n.aux<2;
                    const size_t r=prep_column?((n.aux&1U)?next:row):
                        (main_trace_is_chunk?((n.aux&1U)?local_row+1:local_row):((n.aux&1U)?next:row));
                    v=n.aux<2?preprocessed->trace_values[r*preprocessed->width+n.a]:main_trace[r*main_width+n.a];break;}
                case 2:v=0;break;
                case 3:v=static_cast<uint64_t>(row==0);break;
                case 4:v=static_cast<uint64_t>(row+1==height);break;
                case 5:v=static_cast<uint64_t>(row+1!=height);break;
                case 6:v=goldilocks_add(a,b);break;
                case 7:v=goldilocks_sub(a,b);break;
                case 8:v=goldilocks_mul(a,b);break;
                case 9:v=a==0?0:GOLDILOCKS_P-canonicalize(a);break;
            }values[n.out*tile+lane]=v;
            while(next_lookup<lookup_count&&lookups[next_lookup].emit_after==i){
                const ConstraintLookup l=lookups[next_lookup++];Ext2 fingerprint{0,0};
                for(size_t k=l.arg_count;k>0;--k)fingerprint=ext2_add(ext2_mul(fingerprint,gamma,ext_w),{values[lookup_args[l.arg_start+k-1]*tile+lane],0});
                const Ext2 message=ext2_add(beta,fingerprint);const size_t index=local_row*lookup_count+l.output;
                conjugates[index]={message.c0,goldilocks_sub(0,message.c1)};
                norms[index]=goldilocks_sub(goldilocks_mul(message.c0,message.c0),goldilocks_mul(ext_w,goldilocks_mul(message.c1,message.c1)));
                multiplicities[index]=values[l.multiplicity*tile+lane];
            }
        }
    }
}
__global__ void lookup_group_deltas_batched(Ext2* output,const uint64_t* multiplicities,
    const Ext2* conjugates,const uint64_t* norm_inverses,
    size_t height,size_t num_lookups,size_t group_size,uint64_t ext_w){
    const size_t slots=(num_lookups+group_size-1)/group_size,total=height*slots;
    for(size_t index=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;index<total;index+=static_cast<size_t>(gridDim.x)*blockDim.x){
        const size_t row=index/slots,slot=index%slots,begin=slot*group_size,end=(begin+group_size<num_lookups?begin+group_size:num_lookups);Ext2 delta{0,0};
        for(size_t lookup=begin;lookup<end;++lookup){const size_t message_index=row*num_lookups+lookup;
            const uint64_t norm_inverse=norm_inverses[message_index];
            const Ext2 inverse={goldilocks_mul(conjugates[message_index].c0,norm_inverse),goldilocks_mul(conjugates[message_index].c1,norm_inverse)};const uint64_t multiplicity=multiplicities[message_index];
            delta.c0=goldilocks_add(delta.c0,goldilocks_mul(multiplicity,inverse.c0));delta.c1=goldilocks_add(delta.c1,goldilocks_mul(multiplicity,inverse.c1));
        }output[index]=delta;
    }
}
__global__ void norm_block_products(uint64_t* output,const uint64_t* input,size_t count){
    extern __shared__ uint64_t products[];const size_t index=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    products[threadIdx.x]=index<count?input[index]:1;__syncthreads();
    for(unsigned int stride=blockDim.x/2;stride;stride>>=1){if(threadIdx.x<stride)products[threadIdx.x]=goldilocks_mul(products[threadIdx.x],products[threadIdx.x+stride]);__syncthreads();}
    if(threadIdx.x==0)output[blockIdx.x]=products[0];
}
__global__ void batch_inverse_norm_blocks(uint64_t* output,const uint64_t* input,
    const uint64_t* block_inverses,size_t count){
    extern __shared__ uint64_t products[];const size_t index=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;const uint64_t own=index<count?input[index]:1;
    products[threadIdx.x]=own;__syncthreads();for(unsigned int stride=1;stride<blockDim.x;stride<<=1){uint64_t prior=1;if(threadIdx.x>=stride)prior=products[threadIdx.x-stride];__syncthreads();if(threadIdx.x>=stride)products[threadIdx.x]=goldilocks_mul(products[threadIdx.x],prior);__syncthreads();}
    const uint64_t prefix=threadIdx.x?products[threadIdx.x-1]:1;
    const uint64_t block_product=products[blockDim.x-1];__syncthreads();
    const uint64_t total_inverse=block_inverses?block_inverses[blockIdx.x]:goldilocks_pow(block_product,GOLDILOCKS_P-2);
    products[threadIdx.x]=own;__syncthreads();for(unsigned int stride=1;stride<blockDim.x;stride<<=1){uint64_t prior=1;if(threadIdx.x+stride<blockDim.x)prior=products[threadIdx.x+stride];__syncthreads();if(threadIdx.x+stride<blockDim.x)products[threadIdx.x]=goldilocks_mul(products[threadIdx.x],prior);__syncthreads();}
    const uint64_t suffix=threadIdx.x+1<blockDim.x?products[threadIdx.x+1]:1;if(index<count)output[index]=goldilocks_mul(total_inverse,goldilocks_mul(prefix,suffix));
}
cudaError_t batch_inverse_norms(uint64_t* output,const uint64_t* input,size_t count){
    const size_t block_count=(count+THREADS-1)/THREADS;uint64_t *products=nullptr,*inverses=nullptr;cudaError_t status=cudaSuccess;
    if(block_count<=1){batch_inverse_norm_blocks<<<1,THREADS,THREADS*sizeof(uint64_t)>>>(output,input,nullptr,count);return cudaGetLastError();}
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&products),block_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&inverses),block_count*sizeof(uint64_t));
    if(status==cudaSuccess){norm_block_products<<<static_cast<unsigned int>(block_count),THREADS,THREADS*sizeof(uint64_t)>>>(products,input,count);status=cudaGetLastError();}
    if(status==cudaSuccess){batch_inverse_norm_blocks<<<static_cast<unsigned int>((block_count+THREADS-1)/THREADS),THREADS,THREADS*sizeof(uint64_t)>>>(inverses,products,nullptr,block_count);status=cudaGetLastError();}
    if(status==cudaSuccess){batch_inverse_norm_blocks<<<static_cast<unsigned int>(block_count),THREADS,THREADS*sizeof(uint64_t)>>>(output,input,inverses,count);status=cudaGetLastError();}
    cudaFree(inverses);cudaFree(products);return status;
}

__global__ void initialize_coset_selector_denominators(
    uint64_t* denominators, size_t quotient_size, size_t next_step,
    uint64_t coset_shift, uint64_t coset_generator, uint64_t trace_last,
    uint64_t vanishing_start, uint64_t vanishing_step) {
    constexpr size_t ROWS_PER_WORKER = 16;
    const size_t worker_count =
        (quotient_size + ROWS_PER_WORKER - 1) / ROWS_PER_WORKER;
    for (size_t worker = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         worker < worker_count;
         worker += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t first_row = worker * ROWS_PER_WORKER;
        uint64_t x = goldilocks_mul(
            coset_shift, goldilocks_pow(coset_generator, first_row));
        uint64_t vanishing_power = goldilocks_pow(
            vanishing_step, first_row & (next_step - 1));
        const size_t last_row =
            first_row + ROWS_PER_WORKER < quotient_size
                ? first_row + ROWS_PER_WORKER
                : quotient_size;
        for (size_t row = first_row; row < last_row; ++row) {
            const uint64_t vanishing = goldilocks_sub(
                goldilocks_mul(vanishing_start, vanishing_power), 1);
            denominators[row] = goldilocks_sub(x, 1);
            denominators[quotient_size + row] =
                goldilocks_sub(x, trace_last);
            denominators[2 * quotient_size + row] =
                goldilocks_sub(x, trace_last);
            denominators[3 * quotient_size + row] = vanishing;
            x = goldilocks_mul(x, coset_generator);
            vanishing_power = goldilocks_mul(vanishing_power, vanishing_step);
        }
    }
}

__global__ void finish_coset_selectors(
    uint64_t* selectors, size_t quotient_size) {
    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < quotient_size;
         row += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint64_t vanishing = selectors[3 * quotient_size + row];
        selectors[row] = goldilocks_mul(vanishing, selectors[row]);
        selectors[quotient_size + row] = goldilocks_mul(
            vanishing, selectors[quotient_size + row]);
    }
}

cudaError_t generate_coset_selectors(
    uint64_t* selectors, size_t quotient_size, size_t next_step,
    uint64_t coset_shift, uint64_t coset_generator,
    uint64_t trace_last, uint64_t vanishing_start, uint64_t vanishing_step) {
    constexpr size_t ROWS_PER_WORKER = 16;
    const size_t worker_count =
        (quotient_size + ROWS_PER_WORKER - 1) / ROWS_PER_WORKER;
    initialize_coset_selector_denominators<<<blocks_for(worker_count), THREADS>>>(
        selectors, quotient_size, next_step, coset_shift, coset_generator,
        trace_last, vanishing_start, vanishing_step);
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = batch_inverse_norms(selectors, selectors, 2 * quotient_size);
    }
    if (status == cudaSuccess) {
        finish_coset_selectors<<<blocks_for(quotient_size), THREADS>>>(
            selectors, quotient_size);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = batch_inverse_norms(
            selectors + 3 * quotient_size,
            selectors + 3 * quotient_size,
            quotient_size);
    }
    return status;
}

extern "C" int multi_stark_cuda_generate_coset_selectors(
    int device_id, uint64_t* output, size_t quotient_size, size_t next_step,
    uint64_t coset_shift, uint64_t coset_generator, uint64_t trace_last,
    uint64_t vanishing_start, uint64_t vanishing_step) {
    if (output == nullptr || !is_power_of_two(quotient_size) ||
        !is_power_of_two(next_step) || next_step > quotient_size) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    uint64_t* selectors = nullptr;
    if (status == cudaSuccess)
        status = cudaMalloc(reinterpret_cast<void**>(&selectors),
                            4 * quotient_size * sizeof(uint64_t));
    if (status == cudaSuccess)
        status = generate_coset_selectors(
            selectors, quotient_size, next_step, coset_shift,
            coset_generator, trace_last, vanishing_start, vanishing_step);
    if (status == cudaSuccess)
        status = cudaMemcpy(output, selectors,
                            4 * quotient_size * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost);
    cudaFree(selectors);
    return static_cast<int>(status);
}
__global__ void denominator_norms(uint64_t* norms,const uint64_t* coset,
    size_t count,Ext2 point,uint64_t ext_w){
    for(size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
        i+=static_cast<size_t>(gridDim.x)*blockDim.x){
        const uint64_t real=goldilocks_sub(point.c0,coset[i]);
        norms[i]=goldilocks_sub(goldilocks_mul(real,real),
            goldilocks_mul(ext_w,goldilocks_mul(point.c1,point.c1)));
    }
}
__global__ void finish_inverse_denominators(Ext2* output,const uint64_t* inverses,
    const uint64_t* coset,size_t count,Ext2 point){
    for(size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
        i+=static_cast<size_t>(gridDim.x)*blockDim.x){
        const uint64_t inverse=inverses[i];
        output[i]={goldilocks_mul(goldilocks_sub(point.c0,coset[i]),inverse),
            goldilocks_mul(goldilocks_sub(0,point.c1),inverse)};
    }
}
__global__ void lookup_group_deltas(Ext2* output,const uint64_t* multiplicities,
    const uint64_t* args,const size_t* arg_offsets,size_t height,size_t num_lookups,
    size_t args_width,size_t group_size,Ext2 beta,Ext2 gamma,uint64_t ext_w){
    const size_t slots=(num_lookups+group_size-1)/group_size,total=height*slots;
    for(size_t index=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;index<total;index+=static_cast<size_t>(gridDim.x)*blockDim.x){
        const size_t row=index/slots,slot=index%slots,begin=slot*group_size,end=(begin+group_size<num_lookups?begin+group_size:num_lookups);Ext2 delta{0,0};
        for(size_t lookup=begin;lookup<end;++lookup){Ext2 fingerprint{0,0};
            for(size_t j=arg_offsets[lookup+1];j>arg_offsets[lookup];--j)fingerprint=ext2_add(ext2_mul(fingerprint,gamma,ext_w),{args[row*args_width+j-1],0});
            const Ext2 inverse=ext2_inverse(ext2_add(beta,fingerprint),ext_w);const uint64_t multiplicity=multiplicities[row*num_lookups+lookup];
            delta.c0=goldilocks_add(delta.c0,goldilocks_mul(multiplicity,inverse.c0));delta.c1=goldilocks_add(delta.c1,goldilocks_mul(multiplicity,inverse.c1));
        }output[index]=delta;
    }
}
__global__ void reverse_u64(uint64_t* output,const uint64_t* input,size_t count){const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)output[i]=input[count-1-i];}
__global__ void finish_product_inverse(uint64_t* output,const uint64_t* prefix,const uint64_t* input,size_t count){if(blockIdx.x==0&&threadIdx.x==0)output[0]=goldilocks_pow(goldilocks_mul(prefix[count-1],input[count-1]),GOLDILOCKS_P-2);}
__global__ void scan_ext2_blocks(Ext2* output,Ext2* sums,const Ext2* input,size_t count){
    extern __shared__ Ext2 scan_values[];const size_t index=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const Ext2 own=index<count?input[index]:Ext2{0,0};scan_values[threadIdx.x]=own;__syncthreads();
    for(unsigned int stride=1;stride<blockDim.x;stride<<=1){Ext2 prior{0,0};if(threadIdx.x>=stride)prior=scan_values[threadIdx.x-stride];__syncthreads();if(threadIdx.x>=stride)scan_values[threadIdx.x]=ext2_add(scan_values[threadIdx.x],prior);__syncthreads();}
    if(index<count)output[index]=ext2_sub(scan_values[threadIdx.x],own);if(threadIdx.x==blockDim.x-1)sums[blockIdx.x]=scan_values[threadIdx.x];
}
__global__ void add_scan_block_offsets(Ext2* values,const Ext2* offsets,size_t count){
    const size_t index=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(index<count)values[index]=ext2_add(values[index],offsets[blockIdx.x]);
}

cudaError_t exclusive_scan_ext2(Ext2* output,const Ext2* input,size_t count){
    const size_t blocks=(count+THREADS-1)/THREADS;Ext2* sums=nullptr;Ext2* offsets=nullptr;cudaError_t status=cudaMalloc(reinterpret_cast<void**>(&sums),blocks*sizeof(Ext2));
    if(status==cudaSuccess){scan_ext2_blocks<<<static_cast<unsigned int>(blocks),THREADS,THREADS*sizeof(Ext2)>>>(output,sums,input,count);status=cudaGetLastError();}
    if(status==cudaSuccess&&blocks>1)status=cudaMalloc(reinterpret_cast<void**>(&offsets),blocks*sizeof(Ext2));
    if(status==cudaSuccess&&blocks>1)status=exclusive_scan_ext2(offsets,sums,blocks);
    if(status==cudaSuccess&&blocks>1){add_scan_block_offsets<<<static_cast<unsigned int>(blocks),THREADS>>>(output,offsets,count);status=cudaGetLastError();}
    cudaFree(offsets);cudaFree(sums);return status;
}
__global__ void scan_mul_blocks(uint64_t* output,uint64_t* sums,const uint64_t* input,size_t count){
    extern __shared__ uint64_t scan_products[];const size_t index=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint64_t own=index<count?input[index]:1;scan_products[threadIdx.x]=own;__syncthreads();
    for(unsigned int stride=1;stride<blockDim.x;stride<<=1){uint64_t prior=1;if(threadIdx.x>=stride)prior=scan_products[threadIdx.x-stride];__syncthreads();if(threadIdx.x>=stride)scan_products[threadIdx.x]=goldilocks_mul(scan_products[threadIdx.x],prior);__syncthreads();}
    if(index<count)output[index]=threadIdx.x?scan_products[threadIdx.x-1]:1;if(threadIdx.x==blockDim.x-1)sums[blockIdx.x]=scan_products[threadIdx.x];
}
__global__ void mul_scan_block_offsets(uint64_t* values,const uint64_t* offsets,size_t count){const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)values[i]=goldilocks_mul(values[i],offsets[blockIdx.x]);}
cudaError_t exclusive_scan_mul(uint64_t* output,const uint64_t* input,size_t count){
    const size_t blocks=(count+THREADS-1)/THREADS;uint64_t *sums=nullptr,*offsets=nullptr;cudaError_t status=cudaMalloc(reinterpret_cast<void**>(&sums),blocks*sizeof(uint64_t));
    if(status==cudaSuccess){scan_mul_blocks<<<static_cast<unsigned int>(blocks),THREADS,THREADS*sizeof(uint64_t)>>>(output,sums,input,count);status=cudaGetLastError();}
    if(status==cudaSuccess&&blocks>1)status=cudaMalloc(reinterpret_cast<void**>(&offsets),blocks*sizeof(uint64_t));if(status==cudaSuccess&&blocks>1)status=exclusive_scan_mul(offsets,sums,blocks);
    if(status==cudaSuccess&&blocks>1){mul_scan_block_offsets<<<static_cast<unsigned int>(blocks),THREADS>>>(output,offsets,count);status=cudaGetLastError();}cudaFree(offsets);cudaFree(sums);return status;
}

__device__ __forceinline__ size_t reverse_low_bits(size_t value,
                                                   unsigned int bits) {
    return static_cast<size_t>(__brevll(static_cast<unsigned long long>(value)) >>
                               (64U - bits));
}

__global__ void evaluate_constraint_graph(
    uint64_t* output, const ConstraintNode* nodes, size_t node_count,
    const uint32_t* roots, size_t root_count, const ResidentLde* preprocessed,
    const ResidentLde* main, const ResidentLde* stage2,
    const uint64_t* publics, const uint64_t* selectors, size_t quotient_size,
    size_t next_step) {
    extern __shared__ uint64_t values[];
    const size_t lane = threadIdx.x;
    const size_t tile = blockDim.x;
    const unsigned int log_size = static_cast<unsigned int>(__ffsll(quotient_size) - 1);
    for (size_t row = static_cast<size_t>(blockIdx.x) * tile + lane;
         row < quotient_size; row += static_cast<size_t>(gridDim.x) * tile) {
        const size_t storage_row = reverse_low_bits(row, log_size);
        const size_t next_storage_row =
            reverse_low_bits((row + next_step) & (quotient_size - 1), log_size);
        for (size_t i = 0; i < node_count; ++i) {
            const ConstraintNode node = nodes[i];
            uint64_t result = 0;
            const uint64_t left = node.a < i ? values[node.a * tile + lane] : 0;
            const uint64_t right = node.b < i ? values[node.b * tile + lane] : 0;
            switch (node.op) {
                case 0: result = node.value; break;
                case 1: {
                    const ResidentLde* matrix = node.aux < 2 ? preprocessed
                                               : node.aux < 4 ? main : stage2;
                    const bool next = (node.aux & 1U) != 0;
                    const size_t r = next ? next_storage_row : storage_row;
                    result = matrix->values[r * matrix->width + node.a];
                    break;
                }
                case 2: result = publics[node.a]; break;
                case 3: result = selectors[row]; break;
                case 4: result = selectors[quotient_size + row]; break;
                case 5: result = selectors[2 * quotient_size + row]; break;
                case 6: result = goldilocks_add(left, right); break;
                case 7: result = goldilocks_sub(left, right); break;
                case 8: result = goldilocks_mul(left, right); break;
                case 9: result = left == 0 ? 0 : GOLDILOCKS_P - canonicalize(left); break;
            }
            values[node.out * tile + lane] = result;
        }
        for (size_t root = 0; root < root_count; ++root) {
            output[row * root_count + root] = values[roots[root] * tile + lane];
        }
    }
}

__global__ void evaluate_quotient(
    uint64_t* output, const ConstraintNode* nodes, size_t node_count, size_t slot_count,
    const uint32_t* roots, size_t root_count, const ConstraintLookup* lookups,
    size_t lookup_count, const uint32_t* lookup_args, size_t group_size,
    const ResidentLde* preprocessed, const ResidentLde* main,
    const ResidentLde* stage2, const uint64_t* publics,
    const uint64_t* selectors, const uint64_t* alpha, const uint64_t* delta,
    uint64_t ext_w, size_t quotient_size, size_t next_step,
    uint64_t* global_scratch) {
    extern __shared__ uint64_t shared_values[];
    const size_t lane = threadIdx.x, tile = blockDim.x;
    uint64_t* values=global_scratch ? global_scratch+static_cast<size_t>(blockIdx.x)*slot_count*tile : shared_values;
    const unsigned int log_size = static_cast<unsigned int>(__ffsll(quotient_size) - 1);
    for (size_t row = static_cast<size_t>(blockIdx.x) * tile + lane;
         row < quotient_size; row += static_cast<size_t>(gridDim.x) * tile) {
        const size_t sr = reverse_low_bits(row, log_size);
        const size_t nr = reverse_low_bits((row + next_step) & (quotient_size - 1), log_size);
        for (size_t i = 0; i < node_count; ++i) {
            const ConstraintNode n = nodes[i];
            const uint64_t a = n.op>=6 ? values[n.a * tile + lane] : 0;
            const uint64_t b = n.op>=6&&n.op!=9 ? values[n.b * tile + lane] : 0;
            uint64_t v = 0;
            switch (n.op) {
                case 0: v = n.value; break;
                case 1: { const ResidentLde* m = n.aux < 2 ? preprocessed : n.aux < 4 ? main : stage2;
                          const size_t r = (n.aux & 1U) ? nr : sr; v = m->values[r * m->width + n.a]; break; }
                case 2: v = publics[n.a]; break;
                case 3: v = selectors[row]; break;
                case 4: v = selectors[quotient_size + row]; break;
                case 5: v = selectors[2 * quotient_size + row]; break;
                case 6: v = goldilocks_add(a,b); break;
                case 7: v = goldilocks_sub(a,b); break;
                case 8: v = goldilocks_mul(a,b); break;
                case 9: v = a == 0 ? 0 : GOLDILOCKS_P - canonicalize(a); break;
            }
            values[n.out * tile + lane] = v;
        }
        const size_t total_constraints = root_count +
            (lookup_count == 0 ? 2 : ((lookup_count + group_size - 1) / group_size) * 2);
        Ext2 acc{0,0}; size_t ci = 0;
        for (; ci < root_count; ++ci) {
            const uint64_t c = values[roots[ci] * tile + lane];
            acc.c0 = goldilocks_add(acc.c0, goldilocks_mul(c, alpha[ci]));
            acc.c1 = goldilocks_add(acc.c1, goldilocks_mul(c, alpha[total_constraints + ci]));
        }
        const Ext2 beta{publics[0], publics[1]}, gamma{publics[2], publics[3]};
        const Ext2 injection{goldilocks_mul(selectors[quotient_size + row], delta[0]),
                             goldilocks_mul(selectors[quotient_size + row], delta[1])};
        const size_t groups = lookup_count == 0 ? 1 : (lookup_count + group_size - 1) / group_size;
        for (size_t g = 0; g < groups; ++g) {
            Ext2 constraints[8]; size_t count = 0;
            if (lookup_count == 0) {
                constraints[count++] = ext2_add(ext2_sub(
                    {stage2->values[nr * stage2->width], stage2->values[nr * stage2->width + 1]},
                    {stage2->values[sr * stage2->width], stage2->values[sr * stage2->width + 1]}), injection);
            } else {
                const size_t begin = g * group_size;
                const size_t end = begin + group_size < lookup_count ? begin + group_size : lookup_count;
                Ext2 product{1,0}, messages[8];
                for (size_t j=begin;j<end;++j) { Ext2 f{0,0}; const ConstraintLookup l=lookups[j];
                    for(size_t k=l.arg_count;k>0;--k) { f=ext2_mul(f,gamma,ext_w); f.c0=goldilocks_add(f.c0,values[lookup_args[l.arg_start+k-1]*tile+lane]); }
                    messages[j-begin]=ext2_add(f,beta); product=ext2_mul(product,messages[j-begin],ext_w); }
                const Ext2 source{stage2->values[sr*stage2->width+2*g],stage2->values[sr*stage2->width+2*g+1]};
                Ext2 target = g+1<groups ? Ext2{stage2->values[sr*stage2->width+2*g+2],stage2->values[sr*stage2->width+2*g+3]}
                                         : ext2_add({stage2->values[nr*stage2->width],stage2->values[nr*stage2->width+1]},injection);
                Ext2 rhs{0,0};
                for(size_t j=begin;j<end;++j) { Ext2 others{1,0}; for(size_t k=begin;k<end;++k) if(k!=j) others=ext2_mul(others,messages[k-begin],ext_w);
                    const uint64_t mult=values[lookups[j].multiplicity*tile+lane]; rhs.c0=goldilocks_add(rhs.c0,goldilocks_mul(others.c0,mult)); rhs.c1=goldilocks_add(rhs.c1,goldilocks_mul(others.c1,mult)); }
                constraints[count++]=ext2_sub(ext2_mul(product,ext2_sub(target,source),ext_w),rhs);
            }
            for(size_t k=0;k<count;++k) { const uint64_t c0=constraints[k].c0,c1=constraints[k].c1;
                acc.c0=goldilocks_add(acc.c0,goldilocks_add(goldilocks_mul(c0,alpha[ci]),goldilocks_mul(c1,alpha[ci+1])));
                acc.c1=goldilocks_add(acc.c1,goldilocks_add(goldilocks_mul(c0,alpha[total_constraints+ci]),goldilocks_mul(c1,alpha[total_constraints+ci+1]))); ci+=2; }
        }
        const uint64_t inv=selectors[3*quotient_size+row];
        output[2*row]=goldilocks_mul(acc.c0,inv); output[2*row+1]=goldilocks_mul(acc.c1,inv);
    }
}

cudaError_t quotient_shared_memory_budget(int device_id, size_t* budget) {
    if (budget == nullptr) return cudaErrorInvalidValue;
    int maximum = 0;
    const cudaError_t status = cudaDeviceGetAttribute(
        &maximum, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    if (status != cudaSuccess) return status;
    *budget = static_cast<size_t>(maximum) < 96 * 1024
        ? static_cast<size_t>(maximum) : 96 * 1024;
    return cudaSuccess;
}

cudaError_t configure_quotient_shared_memory(int device_id, size_t bytes) {
    static volatile int lock = 0;
    static size_t configured[64] = {};
    if (device_id < 0 || device_id >= 64) return cudaErrorInvalidDevice;
    while (__sync_lock_test_and_set(&lock, 1)) {}
    cudaError_t status = cudaSuccess;
    if (configured[device_id] < bytes) {
        status = cudaFuncSetAttribute(
            evaluate_quotient, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(bytes));
        if (status == cudaSuccess) configured[device_id] = bytes;
    }
    __sync_lock_release(&lock);
    return status;
}

__global__ void canonicalize_goldilocks(uint64_t* values, size_t count) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        values[index] = canonicalize(values[index]);
    }
}

__global__ void gather_lde_rows(uint64_t* output, const uint64_t* values,
                                const uint64_t* rows, size_t row_count,
                                size_t width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = row_count * width;
    if (index < elements) {
        const size_t output_row = index / width;
        const size_t column = index - output_row * width;
        output[index] = values[rows[output_row] * width + column];
    }
}

// Convert the raw bit-reversed storage of DFT(quotient evaluations) into
// shifted coefficient slices.  This is the device form of
// shifted_quotient_slices: output rows are natural coefficient indices and
// columns are [slice][extension coordinate].  The tail of the blown-up
// allocation is zeroed by the caller before this kernel runs.
__global__ void gather_shifted_quotient_slices(
    uint64_t* output, const uint64_t* transformed, const uint64_t* weights,
    size_t quotient_height, size_t trace_height, size_t quotient_degree,
    size_t extension_degree) {
    const size_t width = quotient_degree * extension_degree;
    const size_t elements = trace_height * width;
    const unsigned int log_height = static_cast<unsigned int>(__ffsll(quotient_height) - 1);
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += grid_stride) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        const size_t chunk = column / extension_degree;
        const size_t coordinate = column - chunk * extension_degree;
        const size_t coefficient = chunk * trace_height + row;
        const size_t negated = (quotient_height - coefficient) & (quotient_height - 1);
        const size_t source = reverse_low_bits(negated, log_height);
        output[index] = goldilocks_mul(
            transformed[source * extension_degree + coordinate], weights[chunk]);
    }
}

__global__ void gather_mixed_lde_row(uint64_t* output,
                                     const uint64_t* const* values,
                                     const size_t* widths,
                                     const size_t* rows,
                                     const size_t* offsets,
                                     size_t matrix_count) {
    const size_t matrix = blockIdx.x;
    if (matrix >= matrix_count) return;
    for (size_t column = threadIdx.x; column < widths[matrix]; column += blockDim.x) {
        output[offsets[matrix] + column] =
            values[matrix][rows[matrix] * widths[matrix] + column];
    }
}

__global__ void gather_mixed_lde_rows(uint64_t* output,
                                      const uint64_t* const* values,
                                      const size_t* widths,
                                      const size_t* row_shifts,
                                      const size_t* offsets,
                                      const uint64_t* indices,
                                      size_t matrix_count, size_t query_count,
                                      size_t max_height, size_t row_width) {
    const size_t task = blockIdx.x;
    const size_t matrix = task % matrix_count;
    const size_t query = task / matrix_count;
    if (query >= query_count) return;
    const size_t row = static_cast<size_t>(indices[query]) >> row_shifts[matrix];
    for (size_t column = threadIdx.x; column < widths[matrix]; column += blockDim.x) {
        output[query * row_width + offsets[matrix] + column] =
            values[matrix][row * widths[matrix] + column];
    }
}

__global__ void radix2_dif_stage(uint64_t* values, size_t height, size_t width,
                                 size_t half, const uint64_t* twiddles) {
    const size_t total = (height >> 1) * width;
    const size_t stride = height / (2 * half);
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += grid_stride) {
        const size_t butterfly = index / width;
        const size_t column = index - butterfly * width;
        const size_t offset = butterfly % half;
        const size_t group = butterfly / half;
        const size_t row_0 = group * (2 * half) + offset;
        const size_t row_1 = row_0 + half;
        const size_t index_0 = row_0 * width + column;
        const size_t index_1 = row_1 * width + column;
        const uint64_t left = values[index_0];
        const uint64_t right = values[index_1];
        values[index_0] = goldilocks_add(left, right);
        values[index_1] = goldilocks_mul(
            goldilocks_sub(left, right), twiddles[offset * stride]);
    }
}

// Fuse two consecutive radix-2 DIF stages. Each thread owns one column of
// one four-row butterfly, so row-major accesses remain coalesced while the
// intermediate values never return to global memory.
__global__ void radix4_dif_stage(uint64_t* values,size_t height,size_t width,
    size_t half,const uint64_t* twiddles){
    const size_t quarter=half>>1,total=(height>>2)*width,stride=height/(2*half);
    const size_t grid_stride=static_cast<size_t>(blockDim.x)*gridDim.x;
    for(size_t index=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;index<total;index+=grid_stride){
        const size_t butterfly=index/width,column=index-butterfly*width;
        const size_t offset=butterfly%quarter,group=butterfly/quarter,base=group*(4*quarter)+offset;
        const size_t i0=base*width+column,i1=(base+quarter)*width+column;
        const size_t i2=(base+2*quarter)*width+column,i3=(base+3*quarter)*width+column;
        const uint64_t a=values[i0],b=values[i1],c=values[i2],d=values[i3];
        const uint64_t ac0=goldilocks_add(a,c),ac1=goldilocks_mul(goldilocks_sub(a,c),twiddles[offset*stride]);
        const uint64_t bd0=goldilocks_add(b,d),bd1=goldilocks_mul(goldilocks_sub(b,d),twiddles[(offset+quarter)*stride]);
        const uint64_t second_twiddle=twiddles[offset*(2*stride)];
        values[i0]=goldilocks_add(ac0,bd0);
        values[i1]=goldilocks_mul(goldilocks_sub(ac0,bd0),second_twiddle);
        values[i2]=goldilocks_add(ac1,bd1);
        values[i3]=goldilocks_mul(goldilocks_sub(ac1,bd1),second_twiddle);
    }
}

// Fuse three consecutive DIF stages. A thread owns one column of an
// eight-row butterfly, cutting the global-memory passes for large row-major
// transforms from three to one while using the identical radix-2 twiddles.
__global__ void radix8_dif_stage(uint64_t* values,size_t height,size_t width,
    size_t half,const uint64_t* twiddles){
    const size_t eighth=half>>2,total=(height>>3)*width,stride=height/(2*half);
    const size_t grid_stride=static_cast<size_t>(blockDim.x)*gridDim.x;
    for(size_t index=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;index<total;index+=grid_stride){
        const size_t butterfly=index/width,column=index-butterfly*width;
        const size_t offset=butterfly%eighth,group=butterfly/eighth;
        const size_t base=group*(8*eighth)+offset;
        const size_t i0=(base+0*eighth)*width+column,i1=(base+1*eighth)*width+column;
        const size_t i2=(base+2*eighth)*width+column,i3=(base+3*eighth)*width+column;
        const size_t i4=(base+4*eighth)*width+column,i5=(base+5*eighth)*width+column;
        const size_t i6=(base+6*eighth)*width+column,i7=(base+7*eighth)*width+column;
        const uint64_t a0=values[i0],a1=values[i1],a2=values[i2],a3=values[i3];
        const uint64_t a4=values[i4],a5=values[i5],a6=values[i6],a7=values[i7];
        const uint64_t x0=goldilocks_add(a0,a4),x1=goldilocks_add(a1,a5);
        const uint64_t x2=goldilocks_add(a2,a6),x3=goldilocks_add(a3,a7);
        const uint64_t x4=goldilocks_mul(goldilocks_sub(a0,a4),twiddles[(offset+0*eighth)*stride]);
        const uint64_t x5=goldilocks_mul(goldilocks_sub(a1,a5),twiddles[(offset+1*eighth)*stride]);
        const uint64_t x6=goldilocks_mul(goldilocks_sub(a2,a6),twiddles[(offset+2*eighth)*stride]);
        const uint64_t x7=goldilocks_mul(goldilocks_sub(a3,a7),twiddles[(offset+3*eighth)*stride]);
        const uint64_t t20=twiddles[offset*(2*stride)];
        const uint64_t t21=twiddles[(offset+eighth)*(2*stride)];
        const uint64_t y0=goldilocks_add(x0,x2),y1=goldilocks_add(x1,x3);
        const uint64_t y2=goldilocks_mul(goldilocks_sub(x0,x2),t20);
        const uint64_t y3=goldilocks_mul(goldilocks_sub(x1,x3),t21);
        const uint64_t y4=goldilocks_add(x4,x6),y5=goldilocks_add(x5,x7);
        const uint64_t y6=goldilocks_mul(goldilocks_sub(x4,x6),t20);
        const uint64_t y7=goldilocks_mul(goldilocks_sub(x5,x7),t21);
        const uint64_t t3=twiddles[offset*(4*stride)];
        values[i0]=goldilocks_add(y0,y1);values[i1]=goldilocks_mul(goldilocks_sub(y0,y1),t3);
        values[i2]=goldilocks_add(y2,y3);values[i3]=goldilocks_mul(goldilocks_sub(y2,y3),t3);
        values[i4]=goldilocks_add(y4,y5);values[i5]=goldilocks_mul(goldilocks_sub(y4,y5),t3);
        values[i6]=goldilocks_add(y6,y7);values[i7]=goldilocks_mul(goldilocks_sub(y6,y7),t3);
    }
}

// Fuse the small-half tail of a DIF transform in shared memory. Once a
// 2*start_half row group fits in a block, every remaining butterfly is local
// to that group; launching and round-tripping through global memory for each
// of those stages only adds latency and bandwidth traffic.
__global__ void radix2_dif_tail(uint64_t* values, size_t height, size_t width,
                                size_t start_half,
                                const uint64_t* twiddles) {
    extern __shared__ uint64_t local_values[];
    const size_t rows_per_group = 2 * start_half;
    const size_t elements_per_group = rows_per_group * width;
    const size_t groups = height / rows_per_group;

    for (size_t group = blockIdx.x; group < groups; group += gridDim.x) {
        const size_t global_base = group * elements_per_group;
        for (size_t index = threadIdx.x; index < elements_per_group;
             index += blockDim.x) {
            local_values[index] = values[global_base + index];
        }
        __syncthreads();

        for (size_t half = start_half;; half >>= 1) {
            const size_t butterflies = start_half * width;
            const size_t stride = height / (2 * half);
            for (size_t index = threadIdx.x; index < butterflies;
                 index += blockDim.x) {
                const size_t butterfly = index / width;
                const size_t column = index - butterfly * width;
                const size_t offset = butterfly % half;
                const size_t subgroup = butterfly / half;
                const size_t row_0 = subgroup * (2 * half) + offset;
                const size_t row_1 = row_0 + half;
                const size_t index_0 = row_0 * width + column;
                const size_t index_1 = row_1 * width + column;
                const uint64_t left = local_values[index_0];
                const uint64_t right = local_values[index_1];
                local_values[index_0] = goldilocks_add(left, right);
                local_values[index_1] = goldilocks_mul(
                    goldilocks_sub(left, right), twiddles[offset * stride]);
            }
            __syncthreads();
            if (half == 1) {
                break;
            }
        }

        for (size_t index = threadIdx.x; index < elements_per_group;
             index += blockDim.x) {
            values[global_base + index] = local_values[index];
        }
        __syncthreads();
    }
}

// Wide row-major matrices cannot fit all columns of a row group in shared
// memory. Tile the columns as well, preserving coalesced row-major loads while
// fusing the final DIF stages independently for each column tile.
__global__ void radix2_dif_tail_tiled(uint64_t* values,size_t height,size_t width,
    size_t start_half,const uint64_t* twiddles,size_t columns_per_tile){
    extern __shared__ uint64_t local_values[];const size_t rows=2*start_half;
    const size_t groups=height/rows,column_tiles=(width+columns_per_tile-1)/columns_per_tile;
    const size_t tile_count=groups*column_tiles;
    for(size_t tile_index=blockIdx.x;tile_index<tile_count;tile_index+=gridDim.x){
        const size_t group=tile_index/column_tiles,column_tile=tile_index%column_tiles;
        const size_t column_base=column_tile*columns_per_tile;
        const size_t columns=(column_base+columns_per_tile<width)?columns_per_tile:width-column_base;
        const size_t elements=rows*columns;
        for(size_t index=threadIdx.x;index<elements;index+=blockDim.x){const size_t row=index/columns,column=index-row*columns;
            local_values[index]=values[(group*rows+row)*width+column_base+column];}
        __syncthreads();
        for(size_t half=start_half;;half>>=1){const size_t butterflies=start_half*columns,stride=height/(2*half);
            for(size_t index=threadIdx.x;index<butterflies;index+=blockDim.x){const size_t butterfly=index/columns,column=index-butterfly*columns;
                const size_t offset=butterfly%half,subgroup=butterfly/half,row0=subgroup*(2*half)+offset,row1=row0+half;
                const size_t i0=row0*columns+column,i1=row1*columns+column;const uint64_t left=local_values[i0],right=local_values[i1];
                local_values[i0]=goldilocks_add(left,right);local_values[i1]=goldilocks_mul(goldilocks_sub(left,right),twiddles[offset*stride]);}
            __syncthreads();if(half==1)break;
        }
        for(size_t index=threadIdx.x;index<elements;index+=blockDim.x){const size_t row=index/columns,column=index-row*columns;
            values[(group*rows+row)*width+column_base+column]=local_values[index];}
        __syncthreads();
    }
}

cudaError_t launch_dif(uint64_t* values, size_t height, size_t width,
                       const uint64_t* twiddles) {
    if (height <= 1 || width == 0) {
        return cudaSuccess;
    }
    const size_t total = (height >> 1) * width;
    const unsigned int blocks = blocks_for(total);
    // Tall, wide row-major batches amortize the heavier fused kernel. Width-2
    // FRI codewords retain the shared-memory tail specialized below.
    if (width >= 8 && height >= (size_t(1) << 18)) {
        size_t half = height >> 1;
        while (half >= 4) {
            radix8_dif_stage<<<blocks_for((height >> 3) * width), THREADS>>>(
                values, height, width, half, twiddles);
            const cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) return status;
            half >>= 3;
        }
        if (half == 2) {
            radix4_dif_stage<<<blocks_for((height >> 2) * width), THREADS>>>(
                values, height, width, half, twiddles);
            return cudaGetLastError();
        }
        if (half == 1) {
            radix2_dif_stage<<<blocks, THREADS>>>(values, height, width, 1,
                                                  twiddles);
        }
        return cudaGetLastError();
    }
    if (width > 2) {
        for (size_t half = height >> 1;; half >>= 1) {
            radix2_dif_stage<<<blocks, THREADS>>>(values, height, width, half,
                                                  twiddles);
            const cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) return status;
            if (half == 1) return cudaSuccess;
        }
    }
    constexpr size_t TAIL_HALF = 128;
    for (size_t half = height >> 1; half > TAIL_HALF; half >>= 1) {
        radix2_dif_stage<<<blocks, THREADS>>>(values, height, width, half,
                                              twiddles);
        const cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            return status;
        }
    }

    const size_t start_half =
        height < 2 * TAIL_HALF ? height >> 1 : TAIL_HALF;
    const size_t groups = height / (2 * start_half);
    const unsigned int tail_blocks = static_cast<unsigned int>(
        groups < MAX_BLOCKS ? groups : MAX_BLOCKS);
    const size_t shared_bytes = 2 * start_half * width * sizeof(uint64_t);
    radix2_dif_tail<<<tail_blocks, THREADS, shared_bytes>>>(
        values, height, width, start_half, twiddles);
    return cudaGetLastError();
}

__global__ void bit_reverse_scale_and_shift(uint64_t* values, size_t height,
                                            size_t width, unsigned int log_height,
                                            uint64_t height_inverse,
                                            const uint64_t* shift_powers) {
    const size_t total = height * width;
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += grid_stride) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        const size_t reverse_row = reverse_index_bits(row, log_height);
        if (row > reverse_row) {
            continue;
        }

        const size_t reverse_index = reverse_row * width + column;
        const uint64_t left = values[index];
        if (row == reverse_row) {
            values[index] = goldilocks_mul(
                left, goldilocks_mul(height_inverse, shift_powers[row]));
            continue;
        }

        const uint64_t right = values[reverse_index];
        values[index] = goldilocks_mul(
            right, goldilocks_mul(height_inverse, shift_powers[row]));
        values[reverse_index] = goldilocks_mul(
            left, goldilocks_mul(height_inverse, shift_powers[reverse_row]));
    }
}

__global__ void goldilocks_ops_kernel(uint64_t* sums, uint64_t* differences,
                                      uint64_t* products, uint64_t* inverses,
                                      const uint64_t* left, const uint64_t* right,
                                      size_t len) {
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < len; index += grid_stride) {
        const uint64_t a = left[index];
        const uint64_t b = right[index];
        sums[index] = goldilocks_add(a, b);
        differences[index] = goldilocks_sub(a, b);
        products[index] = goldilocks_mul(a, b);
        const uint64_t canonical_a = canonicalize(a);
        inverses[index] = canonical_a == 0
                              ? 0
                              : goldilocks_pow(canonical_a, GOLDILOCKS_P - 2);
    }
}

// One warp owns one message. Chunk compression is distributed across lanes;
// lane zero then reduces the (at most 32) chunk chaining values into the root.
// This matches the BLAKE3 tree shape while exposing parallelism within wide
// Merkle leaves instead of assigning a long row to one scalar CUDA thread.
__global__ void blake3_hash_rows_kernel(uint8_t* digests,
                                        const uint8_t* messages,
                                        size_t message_bytes,
                                        size_t message_count) {
    __shared__ uint32_t chunk_values[THREADS / 32][32][8];
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp_in_block = threadIdx.x >> 5;
    const size_t warp_index =
        static_cast<size_t>(blockIdx.x) * (blockDim.x / 32) + warp_in_block;
    const size_t warp_stride =
        static_cast<size_t>(gridDim.x) * (blockDim.x / 32);

    for (size_t message_index = warp_index; message_index < message_count;
         message_index += warp_stride) {
        const uint8_t* message = messages + message_index * message_bytes;
        const size_t chunks = message_bytes == 0 ? 1 : (message_bytes + 1023) / 1024;
        if (lane < chunks) {
            const size_t chunk_offset = static_cast<size_t>(lane) * 1024;
            const size_t chunk_length =
                chunk_offset < message_bytes
                    ? ((message_bytes - chunk_offset) < 1024
                           ? message_bytes - chunk_offset
                           : 1024)
                    : 0;
            uint32_t chunk_output[8];
            blake3_chunk(message + chunk_offset, chunk_length, lane,
                         chunks == 1, chunk_output);
#pragma unroll
            for (unsigned int word = 0; word < 8; ++word) {
                chunk_values[warp_in_block][lane][word] = chunk_output[word];
            }
        }
        __syncwarp();

        if (lane == 0) {
            size_t count = chunks;
            while (count > 1) {
                const bool root_level = count == 2;
                size_t next_count = 0;
                for (size_t index = 0; index + 1 < count; index += 2) {
                    uint32_t parent[8];
                    blake3_parent(chunk_values[warp_in_block][index],
                                  chunk_values[warp_in_block][index + 1],
                                  root_level, parent);
#pragma unroll
                    for (unsigned int word = 0; word < 8; ++word) {
                        chunk_values[warp_in_block][next_count][word] = parent[word];
                    }
                    ++next_count;
                }
                if ((count & 1U) != 0) {
#pragma unroll
                    for (unsigned int word = 0; word < 8; ++word) {
                        chunk_values[warp_in_block][next_count][word] =
                            chunk_values[warp_in_block][count - 1][word];
                    }
                    ++next_count;
                }
                count = next_count;
            }

            uint8_t* digest = digests + message_index * 32;
#pragma unroll
            for (unsigned int word = 0; word < 8; ++word) {
                const uint32_t value = chunk_values[warp_in_block][0][word];
                digest[word * 4] = static_cast<uint8_t>(value);
                digest[word * 4 + 1] = static_cast<uint8_t>(value >> 8);
                digest[word * 4 + 2] = static_cast<uint8_t>(value >> 16);
                digest[word * 4 + 3] = static_cast<uint8_t>(value >> 24);
            }
        }
        __syncwarp();
    }
}

__global__ void blake3_hash_digest_pairs_kernel(
    uint8_t* digests, const uint8_t* left, const uint8_t* right,
    size_t count, bool interleaved) {
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
         index < count; index += grid_stride) {
        const uint8_t* left_digest =
            left + (interleaved ? 2 * index : index) * 32;
        const uint8_t* right_digest =
            right + (interleaved ? 2 * index + 1 : index) * 32;
        blake3_hash_digest_pair(left_digest, right_digest,
                                digests + index * 32);
    }
}

cudaError_t launch_blake3_rows(uint8_t* digests, const uint8_t* messages,
                               size_t message_bytes, size_t message_count) {
    constexpr unsigned int WARPS_PER_BLOCK = THREADS / 32;
    const size_t required =
        (message_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned int blocks = static_cast<unsigned int>(
        required < MAX_BLOCKS ? required : MAX_BLOCKS);
    blake3_hash_rows_kernel<<<blocks, THREADS, 0, cudaStreamPerThread>>>(
        digests, messages, message_bytes, message_count);
    return cudaGetLastError();
}

cudaError_t launch_blake3_digest_pairs(uint8_t* digests,
                                       const uint8_t* left,
                                       const uint8_t* right, size_t count,
                                       bool interleaved) {
    blake3_hash_digest_pairs_kernel<<<blocks_for(count), THREADS, 0,
                                      cudaStreamPerThread>>>(
        digests, left, right, count, interleaved);
    return cudaGetLastError();
}

__global__ void gather_resident_lde_group(
    uint64_t* output, const uint64_t* const* columns, const size_t* strides,
    size_t row_start, size_t rows, size_t width) {
    const size_t count = rows * width;
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += grid_stride) {
        const size_t column = index % width;
        const size_t row = row_start + index / width;
        output[index] = columns[column][row * strides[column]];
    }
}

cudaError_t hash_resident_lde_group(uint8_t* digests,
                                    const void* const* handles,
                                    size_t handle_count, size_t height) {
    size_t total_width = 0;
    for (size_t index = 0; index < handle_count; ++index) {
        const ResidentLde* lde = static_cast<const ResidentLde*>(handles[index]);
        if (lde != nullptr && lde->height == height) {
            if (lde->width > SIZE_MAX - total_width) {
                return cudaErrorInvalidValue;
            }
            total_width += lde->width;
        }
    }
    if (total_width == 0 || total_width > (32 * 1024) / sizeof(uint64_t) ||
        !product_fits(height, total_width)) {
        return cudaErrorInvalidValue;
    }

    constexpr size_t ROW_STAGING_BYTES = size_t(32) << 20;
    const size_t row_bytes = total_width * sizeof(uint64_t);
    const size_t rows_per_chunk =
        (ROW_STAGING_BYTES / row_bytes) > 0 ? (ROW_STAGING_BYTES / row_bytes) : 1;
    const uint64_t** host_columns =
        new (std::nothrow) const uint64_t*[total_width];
    size_t* host_strides = new (std::nothrow) size_t[total_width];
    if (host_columns == nullptr || host_strides == nullptr) {
        delete[] host_columns;
        delete[] host_strides;
        return cudaErrorMemoryAllocation;
    }
    size_t column_offset = 0;
    for (size_t index = 0; index < handle_count; ++index) {
        const ResidentLde* lde = static_cast<const ResidentLde*>(handles[index]);
        if (lde == nullptr || lde->height != height) {
            continue;
        }
        for (size_t column = 0; column < lde->width; ++column) {
            host_columns[column_offset] = lde->values + column;
            host_strides[column_offset++] = lde->width;
        }
    }
    DeviceBuffer device_columns;
    DeviceBuffer device_strides;
    cudaError_t status = device_columns.allocate(total_width);
    if (status == cudaSuccess) {
        status = device_strides.allocate(total_width);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_columns.get(), host_columns,
                            total_width * sizeof(uint64_t*),
                            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_strides.get(), host_strides,
                            total_width * sizeof(size_t),
                            cudaMemcpyHostToDevice);
    }
    delete[] host_columns;
    delete[] host_strides;
    DeviceBuffer combined_rows;
    if (status == cudaSuccess) {
        status = combined_rows.allocate(
            (height < rows_per_chunk ? height : rows_per_chunk) * total_width);
    }
    for (size_t row_start = 0; status == cudaSuccess && row_start < height;
         row_start += rows_per_chunk) {
        const size_t rows =
            (height - row_start < rows_per_chunk) ? height - row_start
                                                   : rows_per_chunk;
        const size_t count = rows * total_width;
        gather_resident_lde_group<<<blocks_for(count), THREADS, 0,
                                    cudaStreamPerThread>>>(
            combined_rows.get(),
            reinterpret_cast<const uint64_t* const*>(device_columns.get()),
            reinterpret_cast<const size_t*>(device_strides.get()), row_start,
            rows, total_width);
        status = cudaGetLastError();
        if (status == cudaSuccess) {
            status = launch_blake3_rows(
                digests + row_start * 32,
                reinterpret_cast<const uint8_t*>(combined_rows.get()), row_bytes,
                rows);
        }
    }
    return status;
}

cudaError_t copy_to_device(DeviceBuffer& destination, const uint64_t* source,
                           size_t elements) {
    cudaError_t status = destination.allocate(elements);
    if (status != cudaSuccess) {
        return status;
    }
    return cudaMemcpy(destination.get(), source, elements * sizeof(uint64_t),
                      cudaMemcpyHostToDevice);
}

cudaError_t copy_to_host(uint64_t* destination, const DeviceBuffer& source,
                         size_t elements) {
    return cudaMemcpy(destination, source.get(), elements * sizeof(uint64_t),
                      cudaMemcpyDeviceToHost);
}

}  // namespace

extern "C" int multi_stark_cuda_dft_batch(int device_id, uint64_t* values,
                                           size_t height, size_t width,
                                           const uint64_t* twiddles) {
    if (values == nullptr || twiddles == nullptr || !is_power_of_two(height) ||
        width == 0 || !product_fits(height, width)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    const size_t elements = height * width;
    HostRegistration registered_values(values, elements * sizeof(uint64_t));
    DeviceBuffer device_values;
    DeviceBuffer device_twiddles;
    status = copy_to_device(device_values, values, elements);
    if (status == cudaSuccess) {
        status = copy_to_device(device_twiddles, twiddles, height / 2);
    }
    if (status == cudaSuccess) {
        status = launch_dif(device_values.get(), height, width, device_twiddles.get());
    }
    if (status == cudaSuccess) {
        status = copy_to_host(values, device_values, elements);
    }
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_coset_lde_batch(
    int device_id, uint64_t* output, const uint64_t* input, size_t height,
    size_t width, size_t added_bits, const uint64_t* inverse_twiddles,
    const uint64_t* shift_powers, const uint64_t* forward_twiddles,
    uint64_t height_inverse) {
    if (output == nullptr || input == nullptr || inverse_twiddles == nullptr ||
        shift_powers == nullptr || forward_twiddles == nullptr ||
        !is_power_of_two(height) || width == 0 ||
        added_bits >= sizeof(size_t) * 8 || height > (SIZE_MAX >> added_bits)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t extended_height = height << added_bits;
    if (!product_fits(extended_height, width)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    const size_t input_elements = height * width;
    const size_t output_elements = extended_height * width;
    HostRegistration registered_input(
        const_cast<uint64_t*>(input), input_elements * sizeof(uint64_t));
    HostRegistration registered_output(output,
                                       output_elements * sizeof(uint64_t));
    DeviceBuffer device_values;
    DeviceBuffer device_inverse_twiddles;
    DeviceBuffer device_shift_powers;
    DeviceBuffer device_forward_twiddles;
    status = device_values.allocate(output_elements);
    if (status == cudaSuccess) {
        status = cudaMemset(device_values.get(), 0, output_elements * sizeof(uint64_t));
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_values.get(), input,
                            input_elements * sizeof(uint64_t),
                            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = copy_to_device(device_inverse_twiddles, inverse_twiddles, height / 2);
    }
    if (status == cudaSuccess) {
        status = copy_to_device(device_shift_powers, shift_powers, height);
    }
    if (status == cudaSuccess) {
        status = copy_to_device(device_forward_twiddles, forward_twiddles,
                                extended_height / 2);
    }
    if (status == cudaSuccess) {
        status = launch_dif(device_values.get(), height, width,
                            device_inverse_twiddles.get());
    }
    if (status == cudaSuccess) {
        const size_t total = input_elements;
        bit_reverse_scale_and_shift<<<blocks_for(total), THREADS>>>(
            device_values.get(), height, width, strict_log2(height),
            height_inverse, device_shift_powers.get());
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = launch_dif(device_values.get(), extended_height, width,
                            device_forward_twiddles.get());
    }
    if (status == cudaSuccess) {
        status = copy_to_host(output, device_values, output_elements);
    }
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_coset_lde_create(
    int device_id, void** handle, const uint64_t* input, size_t height,
    size_t width, size_t added_bits, const uint64_t* inverse_twiddles,
    const uint64_t* shift_powers, const uint64_t* forward_twiddles,
    uint64_t height_inverse) {
    if (handle == nullptr || input == nullptr || (height > 1 && inverse_twiddles == nullptr) ||
        shift_powers == nullptr || forward_twiddles == nullptr ||
        !is_power_of_two(height) || width == 0 ||
        added_bits >= sizeof(size_t) * 8 || height > (SIZE_MAX >> added_bits)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *handle = nullptr;
    const size_t extended_height = height << added_bits;
    if (!product_fits(extended_height, width)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    ResidentLde* lde = nullptr;
    status = create_resident_lde(&lde);
    if (status != cudaSuccess) return static_cast<int>(status);
    lde->height = extended_height;
    lde->width = width;
    const size_t input_elements = height * width;
    const size_t output_elements = extended_height * width;
    const size_t input_bytes=input_elements*sizeof(uint64_t);
    bool input_registered=false;
    const char* managed_lde = getenv("MULTI_STARK_CUDA_MANAGED_LDE");
    const bool use_managed_lde = managed_lde != nullptr &&
                                 managed_lde[0] != '\0' &&
                                 managed_lde[0] != '0';
    if (use_managed_lde) {
        status = cudaMallocManaged(reinterpret_cast<void**>(&lde->values),
                                   output_elements * sizeof(uint64_t),
                                   cudaMemAttachGlobal);
        if (status == cudaSuccess) lde->values_managed = true;
    } else {
        status = cudaMalloc(reinterpret_cast<void**>(&lde->values),
                            output_elements * sizeof(uint64_t));
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&lde->trace_values),
                            input_elements * sizeof(uint64_t));
    }
    if (status == cudaSuccess) lde->trace_height = height;

    // Large pageable uploads otherwise serialize through the driver's hidden
    // staging pool. Registration is best-effort: constrained hosts retain the
    // fully independent pageable path, while recursive proofs can DMA their
    // largest trace matrices directly.
    if(status==cudaSuccess&&input_bytes>=(size_t(8)<<20)){
        const cudaError_t registration=cudaHostRegister(
            const_cast<uint64_t*>(input),input_bytes,cudaHostRegisterDefault);
        if(registration==cudaSuccess)input_registered=true;else cudaGetLastError();
    }

    const uint64_t *device_inverse_twiddles=nullptr,*device_shift_powers=nullptr,*device_forward_twiddles=nullptr;
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(lde->values, 0,
                                 output_elements * sizeof(uint64_t),
                                 cudaStreamPerThread);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(lde->trace_values, input,
                                 input_bytes,
                                 cudaMemcpyHostToDevice,
                                 cudaStreamPerThread);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(lde->values, lde->trace_values,
                                 input_elements * sizeof(uint64_t),
                                 cudaMemcpyDeviceToDevice,
                                 cudaStreamPerThread);
    }
    if (status == cudaSuccess && height > 1) {
        status = cached_device_constants(device_id,inverse_twiddles,height/2,1,0,0,&device_inverse_twiddles);
    }
    if (status == cudaSuccess) {
        status = cached_device_constants(device_id,shift_powers,height,3,shift_powers[0],height>1?shift_powers[1]:0,&device_shift_powers);
    }
    if (status == cudaSuccess) {
        status = cached_device_constants(device_id,forward_twiddles,extended_height/2,2,0,0,&device_forward_twiddles);
    }
    if (status == cudaSuccess) {
        status = launch_dif(lde->values, height, width,device_inverse_twiddles);
    }
    if (status == cudaSuccess) {
        bit_reverse_scale_and_shift<<<blocks_for(input_elements), THREADS>>>(
            lde->values, height, width, strict_log2(height), height_inverse,
            device_shift_powers);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = launch_dif(lde->values, extended_height, width,device_forward_twiddles);
    }
    if (status == cudaSuccess) {
        canonicalize_goldilocks<<<blocks_for(output_elements), THREADS>>>(
            lde->values, output_elements);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaStreamSynchronize(cudaStreamPerThread);
    if(input_registered){const cudaError_t unregister_status=cudaHostUnregister(const_cast<uint64_t*>(input));
        if(status==cudaSuccess)status=unregister_status;}
    if (status != cudaSuccess) {
        destroy_resident_lde(lde);
        return static_cast<int>(status);
    }
    *handle = lde;
    return static_cast<int>(cudaSuccess);
}

extern "C" int multi_stark_cuda_prepare_lde_constants(
    int device_id,const uint64_t* inverse_twiddles,size_t inverse_count,
    const uint64_t* shift_powers,size_t height,const uint64_t* forward_twiddles,
    size_t forward_count) {
    if((inverse_count&&!inverse_twiddles)||!shift_powers||!height||
       !forward_twiddles||!forward_count)return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status=cudaSetDevice(device_id);const uint64_t* ignored=nullptr;
    if(status==cudaSuccess&&inverse_count)
        status=cached_device_constants(device_id,inverse_twiddles,inverse_count,1,0,0,&ignored);
    if(status==cudaSuccess)
        status=cached_device_constants(device_id,shift_powers,height,3,shift_powers[0],height>1?shift_powers[1]:0,&ignored);
    if(status==cudaSuccess)
        status=cached_device_constants(device_id,forward_twiddles,forward_count,2,0,0,&ignored);
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_lde_create_from_host(
    int device_id, void** handle, const uint64_t* input, size_t height,
    size_t width) {
    if (handle == nullptr || input == nullptr || !is_power_of_two(height) ||
        width == 0 || !product_fits(height, width)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *handle = nullptr;
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    ResidentLde* lde = nullptr;
    status = create_resident_lde(&lde);
    if (status != cudaSuccess) return static_cast<int>(status);
    lde->height = height;
    lde->width = width;
    status = cudaMalloc(reinterpret_cast<void**>(&lde->values),
                        height * width * sizeof(uint64_t));
    if (status == cudaSuccess) {
        status = cudaMemcpy(lde->values, input,
                            height * width * sizeof(uint64_t),
                            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        const size_t count = height * width;
        canonicalize_goldilocks<<<blocks_for(count), THREADS>>>(lde->values, count);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(cudaStreamPerThread);
    }
    if (status != cudaSuccess) {
        destroy_resident_lde(lde);
        return static_cast<int>(status);
    }
    *handle = lde;
    return static_cast<int>(cudaSuccess);
}

extern "C" int multi_stark_cuda_zero_lde_create(
    int device_id, void** handle, size_t height, size_t width) {
    if (!handle || !is_power_of_two(height) || width == 0 || !product_fits(height,width))
        return static_cast<int>(cudaErrorInvalidValue);
    *handle=nullptr;cudaError_t status=cudaSetDevice(device_id);
    ResidentLde* lde=nullptr;if(status==cudaSuccess)status=create_resident_lde(&lde);
    lde->height=height;lde->width=width;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&lde->values),height*width*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMemset(lde->values,0,height*width*sizeof(uint64_t));
    if(status==cudaSuccess)*handle=lde;else destroy_resident_lde(lde);return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_lde_copy_to_host(int device_id,
                                                  const void* handle,
                                                  uint64_t* output) {
    if (handle == nullptr || output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    const ResidentLde* lde = static_cast<const ResidentLde*>(handle);
    HostRegistration registered_output(
        output, lde->height * lde->width * sizeof(uint64_t));
    status = cudaMemcpy(output, lde->values,
                        lde->height * lde->width * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost);
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_lde_copy_row(int device_id,
                                               const void* handle,
                                               size_t row,
                                               uint64_t* output) {
    if (handle == nullptr || output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    const ResidentLde* lde = static_cast<const ResidentLde*>(handle);
    if (row >= lde->height) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    status = cudaMemcpy(output, lde->values + row * lde->width,
                        lde->width * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost);
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_lde_copy_rows(
    int device_id, const void* handle, const uint64_t* rows,
    size_t row_count, uint64_t* output) {
    if (handle == nullptr || rows == nullptr || row_count == 0 ||
        output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    const ResidentLde* lde = static_cast<const ResidentLde*>(handle);
    if (!product_fits(row_count, lde->width)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    for (size_t index = 0; index < row_count; ++index) {
        if (rows[index] >= lde->height) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }

    DeviceBuffer device_rows;
    DeviceBuffer device_output;
    status = copy_to_device(device_rows, rows, row_count);
    if (status == cudaSuccess) {
        status = device_output.allocate(row_count * lde->width);
    }
    if (status == cudaSuccess) {
        gather_lde_rows<<<blocks_for(row_count * lde->width), THREADS>>>(
            device_output.get(), lde->values, device_rows.get(), row_count,
            lde->width);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(output, device_output.get(),
                            row_count * lde->width * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost);
    }
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_mixed_lde_open_row(
    int device_id, uint64_t* output, const void* const* handles,
    size_t handle_count, size_t index) {
    if (output == nullptr || handles == nullptr || handle_count == 0) return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status=cudaSetDevice(device_id); size_t max_height=0,total=0;
    for(size_t i=0;i<handle_count;++i){const ResidentLde* l=static_cast<const ResidentLde*>(handles[i]);
        if(l==nullptr || l->height==0 || (l->height&(l->height-1))!=0 || l->width>SIZE_MAX-total) return static_cast<int>(cudaErrorInvalidValue);
        if(l->height>max_height)max_height=l->height; total+=l->width;}
    const uint64_t** hv=new(std::nothrow) const uint64_t*[handle_count]; size_t* hw=new(std::nothrow) size_t[3*handle_count];
    if(hv==nullptr||hw==nullptr){delete[] hv;delete[] hw;return static_cast<int>(cudaErrorMemoryAllocation);} size_t* hr=hw+handle_count;size_t* ho=hr+handle_count;
    size_t off=0;for(size_t i=0;i<handle_count;++i){const ResidentLde* l=static_cast<const ResidentLde*>(handles[i]);hv[i]=l->values;hw[i]=l->width;
        hr[i]=index>>(strict_log2(max_height)-strict_log2(l->height));ho[i]=off;off+=l->width;}
    const uint64_t** dv=nullptr;size_t* dm=nullptr;uint64_t* dout=nullptr;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&dv),handle_count*sizeof(*dv));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&dm),3*handle_count*sizeof(size_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&dout),total*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMemcpy(dv,hv,handle_count*sizeof(*dv),cudaMemcpyHostToDevice);
    if(status==cudaSuccess)status=cudaMemcpy(dm,hw,3*handle_count*sizeof(size_t),cudaMemcpyHostToDevice);
    if(status==cudaSuccess){gather_mixed_lde_row<<<static_cast<unsigned int>(handle_count),THREADS>>>(dout,dv,dm,dm+handle_count,dm+2*handle_count,handle_count);status=cudaGetLastError();}
    if(status==cudaSuccess)status=cudaMemcpy(output,dout,total*sizeof(uint64_t),cudaMemcpyDeviceToHost);
    cudaFree(dout);cudaFree(dm);cudaFree(dv);delete[] hw;delete[] hv;return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_mixed_lde_open_rows(
    int device_id, uint64_t* output, const void* const* handles,
    size_t handle_count, const uint64_t* indices, size_t query_count) {
    if (output == nullptr || handles == nullptr || handle_count == 0 ||
        indices == nullptr || query_count == 0) return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status=cudaSetDevice(device_id);size_t max_height=0,total=0;
    const uint64_t** hv=new(std::nothrow) const uint64_t*[handle_count];
    size_t* hm=new(std::nothrow) size_t[3*handle_count];
    if(hv==nullptr||hm==nullptr){delete[] hv;delete[] hm;return static_cast<int>(cudaErrorMemoryAllocation);}
    size_t* hw=hm,*hh=hm+handle_count,*ho=hh+handle_count;size_t off=0;
    for(size_t i=0;i<handle_count;++i){const ResidentLde* l=static_cast<const ResidentLde*>(handles[i]);
        if(l==nullptr||l->height==0||(l->height&(l->height-1))!=0||l->width>SIZE_MAX-total){delete[] hm;delete[] hv;return static_cast<int>(cudaErrorInvalidValue);}
        hv[i]=l->values;hw[i]=l->width;hh[i]=l->height;ho[i]=off;off+=l->width;total+=l->width;if(l->height>max_height)max_height=l->height;}
    for(size_t i=0;i<handle_count;++i)hh[i]=strict_log2(max_height)-strict_log2(hh[i]);
    for(size_t q=0;q<query_count;++q)if(indices[q]>=max_height){delete[] hm;delete[] hv;return static_cast<int>(cudaErrorInvalidValue);}
    const uint64_t** dv=nullptr;size_t* dm=nullptr;uint64_t* di=nullptr;uint64_t* dout=nullptr;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&dv),handle_count*sizeof(*dv));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&dm),3*handle_count*sizeof(size_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&di),query_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&dout),query_count*total*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMemcpy(dv,hv,handle_count*sizeof(*dv),cudaMemcpyHostToDevice);
    if(status==cudaSuccess)status=cudaMemcpy(dm,hm,3*handle_count*sizeof(size_t),cudaMemcpyHostToDevice);
    if(status==cudaSuccess)status=cudaMemcpy(di,indices,query_count*sizeof(uint64_t),cudaMemcpyHostToDevice);
    if(status==cudaSuccess){gather_mixed_lde_rows<<<static_cast<unsigned int>(handle_count*query_count),THREADS>>>(dout,dv,dm,dm+handle_count,dm+2*handle_count,di,handle_count,query_count,max_height,total);status=cudaGetLastError();}
    if(status==cudaSuccess)status=cudaMemcpy(output,dout,query_count*total*sizeof(uint64_t),cudaMemcpyDeviceToHost);
    cudaFree(dout);cudaFree(di);cudaFree(dm);cudaFree(dv);delete[] hm;delete[] hv;return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_constraint_graph(
    int device_id, uint64_t* output, const void* nodes, size_t node_count,
    const uint32_t* roots, size_t root_count, const void* preprocessed_handle,
    const void* main_handle, const void* stage2_handle,
    const uint64_t* publics, size_t public_count, const uint64_t* selectors,
    size_t quotient_size, size_t next_step) {
    if (output == nullptr || nodes == nullptr || node_count == 0 ||
        roots == nullptr || root_count == 0 || main_handle == nullptr ||
        stage2_handle == nullptr || publics == nullptr || selectors == nullptr ||
        !is_power_of_two(quotient_size) || !is_power_of_two(next_step) ||
        next_step > quotient_size) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    ConstraintNode* device_nodes = nullptr;
    uint32_t* device_roots = nullptr;
    uint64_t* device_publics = nullptr;
    uint64_t* device_selectors = nullptr;
    uint64_t* device_output = nullptr;
    auto allocate_copy = [&](void** destination, const void* source, size_t bytes) {
        if (status != cudaSuccess) return;
        status = cudaMalloc(destination, bytes);
        if (status == cudaSuccess) {
            status = cudaMemcpy(*destination, source, bytes, cudaMemcpyHostToDevice);
        }
    };
    allocate_copy(reinterpret_cast<void**>(&device_nodes), nodes,
                  node_count * sizeof(ConstraintNode));
    allocate_copy(reinterpret_cast<void**>(&device_roots), roots,
                  root_count * sizeof(uint32_t));
    allocate_copy(reinterpret_cast<void**>(&device_publics), publics,
                  public_count * sizeof(uint64_t));
    allocate_copy(reinterpret_cast<void**>(&device_selectors), selectors,
                  3 * quotient_size * sizeof(uint64_t));
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&device_output),
                            quotient_size * root_count * sizeof(uint64_t));
    }
    const size_t shared_budget = 48 * 1024;
    size_t tile = shared_budget / (node_count * sizeof(uint64_t));
    if (tile > 32) tile = 32;
    if (tile == 0) status = cudaErrorInvalidValue;
    if (status == cudaSuccess) {
        const size_t block_count = (quotient_size + tile - 1) / tile;
        evaluate_constraint_graph<<<static_cast<unsigned int>(
                                        block_count < MAX_BLOCKS ? block_count : MAX_BLOCKS),
                                    static_cast<unsigned int>(tile),
                                    node_count * tile * sizeof(uint64_t)>>>(
            device_output, device_nodes, node_count, device_roots, root_count,
            static_cast<const ResidentLde*>(preprocessed_handle),
            static_cast<const ResidentLde*>(main_handle),
            static_cast<const ResidentLde*>(stage2_handle), device_publics,
            device_selectors, quotient_size, next_step);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(output, device_output,
                            quotient_size * root_count * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost);
    }
    cudaFree(device_output);
    cudaFree(device_selectors);
    cudaFree(device_publics);
    cudaFree(device_roots);
    cudaFree(device_nodes);
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_quotient_values(
    int device_id, uint64_t* output, const void* nodes, size_t node_count,
    size_t slot_count,
    const uint32_t* roots, size_t root_count, const void* lookups,
    size_t lookup_count, const uint32_t* lookup_args, size_t lookup_arg_count,
    size_t group_size, const void* preprocessed_handle, const void* main_handle,
    const void* stage2_handle, const uint64_t* publics, size_t public_count,
    uint64_t coset_shift, uint64_t coset_generator, uint64_t trace_last,
    uint64_t vanishing_start, uint64_t vanishing_step,
    const uint64_t* alpha, size_t constraint_count,
    const uint64_t* delta, uint64_t ext_w, size_t quotient_size,
    size_t next_step) {
    if (output == nullptr || nodes == nullptr || roots == nullptr ||
        main_handle == nullptr || stage2_handle == nullptr || publics == nullptr ||
        alpha == nullptr || delta == nullptr ||
        node_count == 0 || slot_count == 0 || group_size == 0 ||
        !is_power_of_two(quotient_size) || !is_power_of_two(next_step) ||
        next_step > quotient_size ||
        (lookup_count != 0 && (lookups == nullptr || lookup_args == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    auto align8=[](size_t n){return (n+7)&~size_t(7);}; size_t bytes=0;
    auto reserve=[&](size_t n){size_t at=bytes;bytes+=align8(n);return at;};
    const size_t on=reserve(node_count*sizeof(ConstraintNode)), oroot=reserve(root_count*sizeof(uint32_t));
    const size_t ol=reserve(lookup_count*sizeof(ConstraintLookup)), oa=reserve(lookup_arg_count*sizeof(uint32_t));
    const size_t op=reserve(public_count*sizeof(uint64_t)), os=reserve(4*quotient_size*sizeof(uint64_t));
    const size_t oalpha=reserve(2*constraint_count*sizeof(uint64_t)), od=reserve(2*sizeof(uint64_t));
    const size_t oo=reserve(2*quotient_size*sizeof(uint64_t)); uint8_t* allocation=nullptr;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&allocation),bytes);
    auto cp=[&](size_t at,const void* src,size_t n){if(status==cudaSuccess&&n!=0)status=cudaMemcpy(allocation+at,src,n,cudaMemcpyHostToDevice);};
    cp(on,nodes,node_count*sizeof(ConstraintNode));cp(oroot,roots,root_count*sizeof(uint32_t));
    cp(ol,lookups,lookup_count*sizeof(ConstraintLookup));cp(oa,lookup_args,lookup_arg_count*sizeof(uint32_t));
    cp(op,publics,public_count*sizeof(uint64_t));
    cp(oalpha,alpha,2*constraint_count*sizeof(uint64_t));cp(od,delta,2*sizeof(uint64_t));
    auto* dn=reinterpret_cast<ConstraintNode*>(allocation+on);auto* dr=reinterpret_cast<uint32_t*>(allocation+oroot);
    auto* dl=reinterpret_cast<ConstraintLookup*>(allocation+ol);auto* da=reinterpret_cast<uint32_t*>(allocation+oa);
    auto* dp=reinterpret_cast<uint64_t*>(allocation+op);auto* ds=reinterpret_cast<uint64_t*>(allocation+os);
    auto* dal=reinterpret_cast<uint64_t*>(allocation+oalpha);auto* dd=reinterpret_cast<uint64_t*>(allocation+od);
    auto* dout=reinterpret_cast<uint64_t*>(allocation+oo);
    if(status==cudaSuccess)status=generate_coset_selectors(ds,quotient_size,next_step,
        coset_shift,coset_generator,trace_last,vanishing_start,vanishing_step);
    size_t budget=0;if(status==cudaSuccess)status=quotient_shared_memory_budget(device_id,&budget);
    size_t tile=budget/(slot_count*sizeof(uint64_t));bool global=tile<28;
    if(global)tile=128;else if(tile>32)tile=32;size_t blocks=(quotient_size+tile-1)/tile;const size_t block_cap=global?256:1024;if(blocks>block_cap)blocks=block_cap;
    uint64_t* scratch=nullptr;if(status==cudaSuccess&&global)status=cudaMalloc(reinterpret_cast<void**>(&scratch),blocks*slot_count*tile*sizeof(uint64_t));
    const size_t dynamic_shared=global?0:slot_count*tile*sizeof(uint64_t);
    if(status==cudaSuccess&&dynamic_shared>48*1024)
        status=configure_quotient_shared_memory(device_id,dynamic_shared);
    if(status==cudaSuccess) {
        evaluate_quotient<<<static_cast<unsigned int>(blocks),static_cast<unsigned int>(tile),dynamic_shared>>>(
            dout,dn,node_count,slot_count,dr,root_count,dl,lookup_count,da,group_size,
            static_cast<const ResidentLde*>(preprocessed_handle),static_cast<const ResidentLde*>(main_handle),
            static_cast<const ResidentLde*>(stage2_handle),dp,ds,dal,dd,ext_w,quotient_size,next_step,scratch);
        status=cudaGetLastError();
    }
    if(status==cudaSuccess) status=cudaMemcpy(output,dout,2*quotient_size*sizeof(uint64_t),cudaMemcpyDeviceToHost);
    cudaFree(scratch);
    cudaFree(allocation);
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_quotient_lde(
    int device_id, void** output_handle, const void* nodes, size_t node_count,
    size_t slot_count, const uint32_t* roots, size_t root_count,
    const void* lookups, size_t lookup_count, const uint32_t* lookup_args,
    size_t lookup_arg_count, size_t group_size, const void* preprocessed_handle,
    const void* main_handle, const void* stage2_handle, const uint64_t* publics,
    size_t public_count, uint64_t coset_shift, uint64_t coset_generator,
    uint64_t trace_last, uint64_t vanishing_start, uint64_t vanishing_step,
    const uint64_t* alpha,
    size_t constraint_count, const uint64_t* delta, uint64_t ext_w,
    size_t quotient_size, size_t next_step, size_t quotient_degree,
    size_t log_blowup, const uint64_t* quotient_twiddles,
    const uint64_t* lde_twiddles, const uint64_t* slice_weights) {
    if (output_handle == nullptr || nodes == nullptr || roots == nullptr ||
        main_handle == nullptr || stage2_handle == nullptr || publics == nullptr ||
        alpha == nullptr || delta == nullptr ||
        quotient_twiddles == nullptr || lde_twiddles == nullptr ||
        slice_weights == nullptr || node_count == 0 || slot_count == 0 ||
        group_size == 0 || !is_power_of_two(quotient_size) ||
        !is_power_of_two(quotient_degree) || quotient_degree > quotient_size ||
        quotient_size % quotient_degree != 0 || !is_power_of_two(next_step) ||
        next_step > quotient_size ||
        log_blowup >= sizeof(size_t) * 8 ||
        (lookup_count != 0 && (lookups == nullptr || lookup_args == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *output_handle = nullptr;
    const size_t trace_height = quotient_size / quotient_degree;
    if (trace_height > (SIZE_MAX >> log_blowup)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const size_t lde_height = trace_height << log_blowup;
    const size_t width = 2 * quotient_degree;
    if (!product_fits(lde_height, width)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t status = cudaSetDevice(device_id);
    auto align8=[](size_t n){return (n+7)&~size_t(7);}; size_t bytes=0;
    auto reserve=[&](size_t n){size_t at=bytes;bytes+=align8(n);return at;};
    const size_t on=reserve(node_count*sizeof(ConstraintNode)), oroot=reserve(root_count*sizeof(uint32_t));
    const size_t ol=reserve(lookup_count*sizeof(ConstraintLookup)), oa=reserve(lookup_arg_count*sizeof(uint32_t));
    const size_t op=reserve(public_count*sizeof(uint64_t)), os=reserve(4*quotient_size*sizeof(uint64_t));
    const size_t oalpha=reserve(2*constraint_count*sizeof(uint64_t)), od=reserve(2*sizeof(uint64_t));
    const size_t oo=reserve(2*quotient_size*sizeof(uint64_t)); uint8_t* allocation=nullptr;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&allocation),bytes);
    auto cp=[&](size_t at,const void* src,size_t n){if(status==cudaSuccess&&n!=0)status=cudaMemcpy(allocation+at,src,n,cudaMemcpyHostToDevice);};
    cp(on,nodes,node_count*sizeof(ConstraintNode));cp(oroot,roots,root_count*sizeof(uint32_t));
    cp(ol,lookups,lookup_count*sizeof(ConstraintLookup));cp(oa,lookup_args,lookup_arg_count*sizeof(uint32_t));
    cp(op,publics,public_count*sizeof(uint64_t));
    cp(oalpha,alpha,2*constraint_count*sizeof(uint64_t));cp(od,delta,2*sizeof(uint64_t));
    auto* dn=reinterpret_cast<ConstraintNode*>(allocation+on);auto* dr=reinterpret_cast<uint32_t*>(allocation+oroot);
    auto* dl=reinterpret_cast<ConstraintLookup*>(allocation+ol);auto* da=reinterpret_cast<uint32_t*>(allocation+oa);
    auto* dp=reinterpret_cast<uint64_t*>(allocation+op);auto* ds=reinterpret_cast<uint64_t*>(allocation+os);
    auto* dal=reinterpret_cast<uint64_t*>(allocation+oalpha);auto* dd=reinterpret_cast<uint64_t*>(allocation+od);
    auto* quotient=reinterpret_cast<uint64_t*>(allocation+oo);

    if(status==cudaSuccess)status=generate_coset_selectors(ds,quotient_size,next_step,
        coset_shift,coset_generator,trace_last,vanishing_start,vanishing_step);

    const uint64_t *device_quotient_twiddles=nullptr,*device_lde_twiddles=nullptr,*device_weights=nullptr;
    if(status==cudaSuccess)status=cached_device_constants(device_id,quotient_twiddles,quotient_size/2,2,0,0,&device_quotient_twiddles);
    if(status==cudaSuccess)status=cached_device_constants(device_id,lde_twiddles,lde_height/2,2,0,0,&device_lde_twiddles);
    if(status==cudaSuccess)status=cached_device_constants(device_id,slice_weights,quotient_degree,4,slice_weights[0],quotient_degree>1?slice_weights[1]:0,&device_weights);

    size_t budget=0;if(status==cudaSuccess)status=quotient_shared_memory_budget(device_id,&budget);
    size_t tile=budget/(slot_count*sizeof(uint64_t));bool global=tile<28;
    if(global)tile=128;else if(tile>32)tile=32;size_t blocks=(quotient_size+tile-1)/tile;const size_t block_cap=global?256:1024;if(blocks>block_cap)blocks=block_cap;
    uint64_t* scratch=nullptr;if(status==cudaSuccess&&global)status=cudaMalloc(reinterpret_cast<void**>(&scratch),blocks*slot_count*tile*sizeof(uint64_t));
    const size_t dynamic_shared=global?0:slot_count*tile*sizeof(uint64_t);
    if(status==cudaSuccess&&dynamic_shared>48*1024)
        status=configure_quotient_shared_memory(device_id,dynamic_shared);
    if(status==cudaSuccess) {
        evaluate_quotient<<<static_cast<unsigned int>(blocks),static_cast<unsigned int>(tile),dynamic_shared>>>(
            quotient,dn,node_count,slot_count,dr,root_count,dl,lookup_count,da,group_size,
            static_cast<const ResidentLde*>(preprocessed_handle),static_cast<const ResidentLde*>(main_handle),
            static_cast<const ResidentLde*>(stage2_handle),dp,ds,dal,dd,ext_w,quotient_size,next_step,scratch);
        status=cudaGetLastError();
    }
    if(status==cudaSuccess)status=launch_dif(quotient,quotient_size,2,device_quotient_twiddles);

    ResidentLde* lde = nullptr;
    if(status==cudaSuccess) {
        status=create_resident_lde(&lde);
    }
    if(status==cudaSuccess) {
        lde->height=lde_height;lde->width=width;
        status=cudaMalloc(reinterpret_cast<void**>(&lde->values),lde_height*width*sizeof(uint64_t));
    }
    if(status==cudaSuccess)status=cudaMemset(lde->values,0,lde_height*width*sizeof(uint64_t));
    if(status==cudaSuccess) {
        gather_shifted_quotient_slices<<<blocks_for(trace_height*width),THREADS>>>(
            lde->values,quotient,device_weights,quotient_size,trace_height,
            quotient_degree,2);
        status=cudaGetLastError();
    }
    if(status==cudaSuccess)status=launch_dif(lde->values,lde_height,width,device_lde_twiddles);
    if(status==cudaSuccess)status=cudaStreamSynchronize(0);
    if(status==cudaSuccess) {
        *output_handle=lde;
    } else if(lde) {
        destroy_resident_lde(lde);
    }
    cudaFree(scratch);cudaFree(allocation);
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_lde_interpolate(int device_id,uint64_t* output,const void* handle,
    size_t height,const uint64_t* inv_denoms,const uint64_t* coset,const uint64_t* scale,uint64_t ext_w){
    if(!output||!handle||!inv_denoms||!coset||!scale)return static_cast<int>(cudaErrorInvalidValue);
    ResidentLde* l=const_cast<ResidentLde*>(static_cast<const ResidentLde*>(handle));if(height==0||height>l->height)return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status=cudaSetDevice(device_id);const size_t ib=height*sizeof(Ext2),cb=height*sizeof(uint64_t),ob=l->width*sizeof(Ext2),needed=ib+cb+ob;
    if(status==cudaSuccess&&l->interpolation_scratch_bytes<needed){if(l->interpolation_scratch)status=cudaFree(l->interpolation_scratch);l->interpolation_scratch=nullptr;if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&l->interpolation_scratch),needed);if(status==cudaSuccess)l->interpolation_scratch_bytes=needed;}
    uint8_t* mem=l->interpolation_scratch;
    if(status==cudaSuccess)status=cudaMemcpy(mem,inv_denoms,ib,cudaMemcpyHostToDevice);
    if(status==cudaSuccess)status=cudaMemcpy(mem+ib,coset,cb,cudaMemcpyHostToDevice);
    if(status==cudaSuccess){interpolate_lde_columns<<<static_cast<unsigned int>(l->width),THREADS,THREADS*sizeof(Ext2)>>>(reinterpret_cast<Ext2*>(mem+ib+cb),l,reinterpret_cast<Ext2*>(mem),reinterpret_cast<uint64_t*>(mem+ib),height,{scale[0],scale[1]},ext_w);status=cudaGetLastError();}
    if(status==cudaSuccess)status=cudaMemcpy(output,mem+ib+cb,ob,cudaMemcpyDeviceToHost);return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_fri_workspace_create(int device_id,void** handle,
    const uint64_t* points,const size_t* counts,size_t point_count,
    const uint64_t* coset,size_t coset_count,uint64_t ext_w){
    if(!handle||!points||!counts||!point_count||!coset||!coset_count)
        return static_cast<int>(cudaErrorInvalidValue);*handle=nullptr;
    size_t inv_count=0,max_count=0;
    for(size_t i=0;i<point_count;++i){
        if(!counts[i]||counts[i]>coset_count||inv_count>SIZE_MAX-counts[i])
            return static_cast<int>(cudaErrorInvalidValue);
        inv_count+=counts[i];if(counts[i]>max_count)max_count=counts[i];
    }
    cudaError_t status=cudaSetDevice(device_id);auto* w=new(std::nothrow) ResidentFriWorkspace;if(!w)return static_cast<int>(cudaErrorMemoryAllocation);
    w->inv_count=inv_count;w->coset_count=coset_count;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&w->inv_denoms),inv_count*sizeof(Ext2));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&w->coset),coset_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMemcpy(w->coset,coset,coset_count*sizeof(uint64_t),cudaMemcpyHostToDevice);
    uint64_t *norms=nullptr,*inverses=nullptr;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&norms),max_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&inverses),max_count*sizeof(uint64_t));
    size_t offset=0;
    for(size_t i=0;status==cudaSuccess&&i<point_count;++i){const Ext2 point{points[2*i],points[2*i+1]};
        denominator_norms<<<blocks_for(counts[i]),THREADS>>>(norms,w->coset,counts[i],point,ext_w);status=cudaGetLastError();
        if(status==cudaSuccess)status=batch_inverse_norms(inverses,norms,counts[i]);
        if(status==cudaSuccess){finish_inverse_denominators<<<blocks_for(counts[i]),THREADS>>>(w->inv_denoms+offset,inverses,w->coset,counts[i],point);status=cudaGetLastError();}
        offset+=counts[i];
    }
    if(inverses)cudaFree(inverses);if(norms)cudaFree(norms);
    if(status!=cudaSuccess){delete w;return static_cast<int>(status);}*handle=w;return static_cast<int>(cudaSuccess);
}

extern "C" int multi_stark_cuda_fri_interpolate_batch(int device_id,void* handle,uint64_t* output,
    size_t output_count,const InterpolationTask* tasks,size_t task_count,uint64_t ext_w){
    if(!handle||!output||!output_count||!tasks||!task_count)return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status=cudaSetDevice(device_id);auto* w=static_cast<ResidentFriWorkspace*>(handle);
    if(status==cudaSuccess&&w->output_capacity<output_count){if(w->output)status=cudaFree(w->output);w->output=nullptr;if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&w->output),output_count*sizeof(Ext2));if(status==cudaSuccess)w->output_capacity=output_count;}
    size_t partial_count=0,block_count=0,finish_count=0;
    for(size_t i=0;status==cudaSuccess&&i<task_count;){const auto& t=tasks[i];auto* l=static_cast<const ResidentLde*>(t.lde);
        if(!l||!t.height||t.height>l->height||t.inv_offset>w->inv_count-t.height||t.output_offset>output_count-l->width){status=cudaErrorInvalidValue;break;}
        const size_t partial_rows=(t.height+INTERPOLATION_ROWS-1)/INTERPOLATION_ROWS;
        const bool pair=i+1<task_count&&tasks[i+1].lde==t.lde&&tasks[i+1].height==t.height&&
            tasks[i+1].inv_offset<=w->inv_count-t.height&&tasks[i+1].output_offset<=output_count-l->width;
        const size_t columns=(l->width+INTERPOLATION_COLUMNS-1)/INTERPOLATION_COLUMNS;
        if(partial_rows>SIZE_MAX/l->width||partial_count>SIZE_MAX-partial_rows*l->width*(pair?2:1)||
           columns&&partial_rows>SIZE_MAX/columns||block_count>SIZE_MAX-partial_rows*columns||
           finish_count>SIZE_MAX-l->width*(pair?2:1)){status=cudaErrorInvalidValue;break;}
        partial_count+=partial_rows*l->width*(pair?2:1);block_count+=partial_rows*columns;
        finish_count+=l->width*(pair?2:1);i+=pair?2:1;}
    if(status==cudaSuccess&&w->partial_capacity<partial_count){if(w->interpolation_partials)status=cudaFree(w->interpolation_partials);w->interpolation_partials=nullptr;
        if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&w->interpolation_partials),partial_count*sizeof(Ext2));
        if(status==cudaSuccess)w->partial_capacity=partial_count;}
    InterpolationBlockDesc* host_blocks=status==cudaSuccess?new(std::nothrow) InterpolationBlockDesc[block_count]:nullptr;
    InterpolationFinishDesc* host_finishes=status==cudaSuccess?new(std::nothrow) InterpolationFinishDesc[finish_count]:nullptr;
    if(status==cudaSuccess&&(!host_blocks||!host_finishes))status=cudaErrorMemoryAllocation;
    size_t partial_at=0,block_at=0,finish_at=0;
    for(size_t i=0;status==cudaSuccess&&i<task_count;){const auto& t=tasks[i];auto* l=static_cast<const ResidentLde*>(t.lde);
        const size_t partial_rows=(t.height+INTERPOLATION_ROWS-1)/INTERPOLATION_ROWS;
        const size_t one_count=partial_rows*l->width;
        const bool pair=i+1<task_count&&tasks[i+1].lde==t.lde&&tasks[i+1].height==t.height&&
            tasks[i+1].inv_offset<=w->inv_count-t.height&&tasks[i+1].output_offset<=output_count-l->width;
        Ext2* p0=w->interpolation_partials+partial_at;Ext2* p1=pair?p0+one_count:nullptr;
        for(size_t row=0;row<t.height;row+=INTERPOLATION_ROWS)
            for(size_t column=0;column<l->width;column+=INTERPOLATION_COLUMNS)
                host_blocks[block_at++]={l->values,w->coset,w->inv_denoms+t.inv_offset,
                    pair?w->inv_denoms+tasks[i+1].inv_offset:nullptr,p0,p1,l->width,column,row,
                    row+INTERPOLATION_ROWS<t.height?row+INTERPOLATION_ROWS:t.height};
        for(size_t column=0;column<l->width;++column)
            host_finishes[finish_at++]={w->output+t.output_offset+column,p0,partial_rows,l->width,column,{t.scale0,t.scale1}};
        if(pair){const auto& u=tasks[i+1];for(size_t column=0;column<l->width;++column)
            host_finishes[finish_at++]={w->output+u.output_offset+column,p1,partial_rows,l->width,column,{u.scale0,u.scale1}};}
        partial_at+=one_count*(pair?2:1);i+=pair?2:1;}
    InterpolationBlockDesc* device_blocks=nullptr;InterpolationFinishDesc* device_finishes=nullptr;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&device_blocks),block_count*sizeof(InterpolationBlockDesc));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&device_finishes),finish_count*sizeof(InterpolationFinishDesc));
    if(status==cudaSuccess)status=cudaMemcpyAsync(device_blocks,host_blocks,block_count*sizeof(InterpolationBlockDesc),cudaMemcpyHostToDevice,cudaStreamPerThread);
    if(status==cudaSuccess)status=cudaMemcpyAsync(device_finishes,host_finishes,finish_count*sizeof(InterpolationFinishDesc),cudaMemcpyHostToDevice,cudaStreamPerThread);
    if(status==cudaSuccess)status=cudaStreamSynchronize(cudaStreamPerThread);
    delete[] host_finishes;host_finishes=nullptr;delete[] host_blocks;host_blocks=nullptr;
    if(status==cudaSuccess){const dim3 block(INTERPOLATION_COLUMNS,INTERPOLATION_LANES);
        interpolate_lde_blocks<<<static_cast<unsigned int>(block_count),block>>>(device_blocks,ext_w);status=cudaGetLastError();}
    if(status==cudaSuccess){finish_lde_interpolation_blocks<<<static_cast<unsigned int>(finish_count),THREADS,THREADS*sizeof(Ext2)>>>(device_finishes,ext_w);status=cudaGetLastError();}
    if(device_finishes)cudaFree(device_finishes);if(device_blocks)cudaFree(device_blocks);
    delete[] host_finishes;delete[] host_blocks;
    if(status==cudaSuccess)status=cudaMemcpy(output,w->output,output_count*sizeof(Ext2),cudaMemcpyDeviceToHost);return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_fri_reduce_batch(int device_id,void* handle,
    const ReductionTask* tasks,size_t task_count,const uint64_t* alpha,size_t alpha_count,uint64_t ext_w){
    if(!handle||!tasks||!task_count||!alpha||!alpha_count)return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status=cudaSetDevice(device_id);auto* w=static_cast<ResidentFriWorkspace*>(handle);
    if(status==cudaSuccess&&w->alpha_capacity<alpha_count){if(w->alpha)status=cudaFree(w->alpha);w->alpha=nullptr;if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&w->alpha),alpha_count*sizeof(Ext2));if(status==cudaSuccess)w->alpha_capacity=alpha_count;}
    if(status==cudaSuccess)status=cudaMemcpy(w->alpha,alpha,alpha_count*sizeof(Ext2),cudaMemcpyHostToDevice);
    for(size_t i=0;status==cudaSuccess&&i<task_count;++i){const auto& t=tasks[i];auto* r=static_cast<ResidentReducedOpening*>(t.reduced);auto* l=static_cast<const ResidentLde*>(t.lde);
        if(!r||!l||t.height!=r->height||t.height>l->height||t.inv_offset+t.height>w->inv_count||l->width>alpha_count){status=cudaErrorInvalidValue;break;}
        accumulate_reduced_opening<<<blocks_for(t.height),THREADS>>>(r->values,l,w->inv_denoms+t.inv_offset,w->alpha,t.height,{t.y0,t.y1},{t.offset0,t.offset1},ext_w);status=cudaGetLastError();}
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_fri_workspace_destroy(int device_id,void* handle){
    cudaError_t status=cudaSetDevice(device_id);if(status==cudaSuccess)delete static_cast<ResidentFriWorkspace*>(handle);return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_reduced_into_lde(int device_id,void** output,void* reduced){
    if(!output||!reduced)return static_cast<int>(cudaErrorInvalidValue);*output=nullptr;
    auto* r=static_cast<ResidentReducedOpening*>(reduced);cudaError_t status=cudaSetDevice(device_id);
    ResidentLde* l=nullptr;if(status==cudaSuccess)status=create_resident_lde(&l);if(status==cudaSuccess){l->height=r->height;l->width=2;}
    if(status!=cudaSuccess){destroy_resident_lde(l);return static_cast<int>(status);}
    l->values=reinterpret_cast<uint64_t*>(r->values);r->values=nullptr;*output=l;
    return static_cast<int>(cudaSuccess);
}

extern "C" int multi_stark_cuda_fri_fold_resident(int device_id,void** output,const void* input,
    const void* next_reduced,const uint64_t* beta,size_t log_arity,uint64_t beta_power0,
    uint64_t beta_power1,uint64_t g_inv,uint64_t ext_w){
    if(!output||!input||!beta||log_arity!=1)return static_cast<int>(cudaErrorInvalidValue);*output=nullptr;
    auto* in=static_cast<const ResidentLde*>(input);if(in->width!=2||in->height%2)return static_cast<int>(cudaErrorInvalidValue);
    const size_t final_height=in->height>>log_arity;auto* next=static_cast<const ResidentReducedOpening*>(next_reduced);
    if(next&&next->height!=final_height)return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status=cudaSetDevice(device_id);Ext2 *a=reinterpret_cast<Ext2*>(in->values),*first=nullptr;uint64_t* powers=nullptr;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&first),(in->height/2)*sizeof(Ext2));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&powers),(in->height/2)*sizeof(uint64_t));
    if(status==cudaSuccess){init_fri_powers<<<blocks_for(in->height/2),THREADS>>>(powers,in->height/2,g_inv);status=cudaGetLastError();}
    if(status==cudaSuccess){fold_fri_ext2<<<blocks_for(final_height),THREADS>>>(first,a,powers,final_height,{beta[0],beta[1]},ext_w);status=cudaGetLastError();}
    a=first;
    if(status==cudaSuccess&&next){add_scaled_ext2<<<blocks_for(final_height),THREADS>>>(a,next->values,final_height,{beta_power0,beta_power1},ext_w);status=cudaGetLastError();}
    ResidentLde* result=nullptr;if(status==cudaSuccess)status=create_resident_lde(&result);
    if(status==cudaSuccess){result->values=reinterpret_cast<uint64_t*>(a);result->height=final_height;result->width=2;first=nullptr;*output=result;}else destroy_resident_lde(result);
    cudaFree(powers);cudaFree(first);return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_reduced_create(int device_id,void** handle,size_t height){
    if(!handle||!is_power_of_two(height))return static_cast<int>(cudaErrorInvalidValue);*handle=nullptr;cudaError_t status=cudaSetDevice(device_id);
    auto* r=new(std::nothrow) ResidentReducedOpening;if(!r)return static_cast<int>(cudaErrorMemoryAllocation);r->height=height;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&r->values),height*sizeof(Ext2));if(status==cudaSuccess)status=cudaMemset(r->values,0,height*sizeof(Ext2));
    if(status!=cudaSuccess){delete r;return static_cast<int>(status);}*handle=r;return static_cast<int>(cudaSuccess);
}
extern "C" int multi_stark_cuda_lookup_trace(int device_id,uint64_t* output,uint64_t* total,
    const uint64_t* multiplicities,const uint64_t* args,const size_t* arg_offsets,
    size_t height,size_t num_lookups,size_t args_width,size_t group_size,
    const uint64_t* beta,const uint64_t* gamma,uint64_t ext_w){
    if(!output||!total||!multiplicities||!arg_offsets||!height||!num_lookups||!group_size||!beta||!gamma||(args_width&& !args))return static_cast<int>(cudaErrorInvalidValue);
    const size_t slots=(num_lookups+group_size-1)/group_size,count=height*slots;cudaError_t status=cudaSetDevice(device_id);
    const size_t message_count=height*num_lookups;uint64_t *dm=nullptr,*da=nullptr,*norms=nullptr,*norm_inverses=nullptr;size_t* offsets=nullptr;Ext2 *conjugates=nullptr,*deltas=nullptr,*scan=nullptr;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&dm),height*num_lookups*sizeof(uint64_t));
    if(status==cudaSuccess&&args_width)status=cudaMalloc(reinterpret_cast<void**>(&da),height*args_width*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&offsets),(num_lookups+1)*sizeof(size_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&deltas),count*sizeof(Ext2));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&scan),count*sizeof(Ext2));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&conjugates),message_count*sizeof(Ext2));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&norms),message_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&norm_inverses),message_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMemcpyAsync(dm,multiplicities,height*num_lookups*sizeof(uint64_t),cudaMemcpyHostToDevice,0);
    if(status==cudaSuccess&&args_width)status=cudaMemcpyAsync(da,args,height*args_width*sizeof(uint64_t),cudaMemcpyHostToDevice,0);
    if(status==cudaSuccess)status=cudaMemcpyAsync(offsets,arg_offsets,(num_lookups+1)*sizeof(size_t),cudaMemcpyHostToDevice,0);
    if(status==cudaSuccess){lookup_messages<<<blocks_for(message_count),THREADS>>>(conjugates,norms,da,offsets,height,num_lookups,args_width,{beta[0],beta[1]},{gamma[0],gamma[1]},ext_w);status=cudaGetLastError();}
    if(status==cudaSuccess)status=batch_inverse_norms(norm_inverses,norms,message_count);
    if(status==cudaSuccess){lookup_group_deltas_batched<<<blocks_for(count),THREADS>>>(deltas,dm,conjugates,norm_inverses,height,num_lookups,group_size,ext_w);status=cudaGetLastError();}
    if(status==cudaSuccess)status=exclusive_scan_ext2(scan,deltas,count);
    if(status==cudaSuccess)status=cudaMemcpy(output,scan,count*sizeof(Ext2),cudaMemcpyDeviceToHost);
    if(status==cudaSuccess)status=cudaMemcpy(total,scan+count-1,sizeof(Ext2),cudaMemcpyDeviceToHost);if(status==cudaSuccess)status=cudaMemcpy(total+2,deltas+count-1,sizeof(Ext2),cudaMemcpyDeviceToHost);
    cudaFree(norm_inverses);cudaFree(norms);cudaFree(conjugates);cudaFree(scan);cudaFree(deltas);cudaFree(offsets);cudaFree(da);cudaFree(dm);return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_lookup_graph_lde(int device_id,void** output_handle,uint64_t* total,
    const void* nodes,size_t node_count,size_t slot_count,const void* lookups,size_t lookup_count,
    const uint32_t* lookup_args,size_t lookup_arg_count,const void* preprocessed_handle,
    const void* main_handle,size_t group_size,const uint64_t* beta,const uint64_t* gamma,
    uint64_t ext_w,size_t added_bits,const uint64_t* inverse_twiddles,
    const uint64_t* shift_powers,const uint64_t* forward_twiddles,uint64_t height_inverse){
    auto* main=const_cast<ResidentLde*>(static_cast<const ResidentLde*>(main_handle));
    auto* prep=static_cast<const ResidentLde*>(preprocessed_handle);
    if(!output_handle||!total||!nodes||!node_count||!slot_count||!lookups||!lookup_count||
       (lookup_arg_count&&!lookup_args)||!main||
       (!main->trace_values&&!main->host_trace_values)||!main->trace_height||
       !group_size||!beta||!gamma||!inverse_twiddles||!shift_powers||!forward_twiddles||
       !is_power_of_two(main->trace_height)||added_bits>=sizeof(size_t)*8||
       main->trace_height>(SIZE_MAX>>added_bits)||(prep&&!prep->trace_values)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *output_handle=nullptr;const size_t height=main->trace_height;
    const size_t groups=(lookup_count+group_size-1)/group_size,width=2*groups;
    const size_t count=height*groups,extended_height=height<<added_bits;
    constexpr size_t LOOKUP_ROWS_PER_CHUNK=size_t(1)<<16;
    const size_t chunk_rows=height<LOOKUP_ROWS_PER_CHUNK?height:LOOKUP_ROWS_PER_CHUNK;
    const size_t message_count=chunk_rows*lookup_count;
    if(!product_fits(extended_height,width)||!product_fits(height,lookup_count))return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status=cudaSetDevice(device_id);uint8_t* metadata=nullptr;
    auto align8=[](size_t n){return (n+7)&~size_t(7);};size_t bytes=0;
    auto reserve=[&](size_t n){const size_t at=bytes;bytes+=align8(n);return at;};
    const size_t on=reserve(node_count*sizeof(ConstraintNode));
    const size_t ol=reserve(lookup_count*sizeof(ConstraintLookup));
    const size_t oa=reserve(lookup_arg_count*sizeof(uint32_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&metadata),bytes);
    auto cp=[&](size_t at,const void* src,size_t n){if(status==cudaSuccess)status=cudaMemcpy(metadata+at,src,n,cudaMemcpyHostToDevice);};
    cp(on,nodes,node_count*sizeof(ConstraintNode));cp(ol,lookups,lookup_count*sizeof(ConstraintLookup));cp(oa,lookup_args,lookup_arg_count*sizeof(uint32_t));
    auto* dn=reinterpret_cast<ConstraintNode*>(metadata+on);auto* dl=reinterpret_cast<ConstraintLookup*>(metadata+ol);auto* da=reinterpret_cast<uint32_t*>(metadata+oa);
    Ext2 *conjugates=nullptr,*deltas=nullptr;uint64_t *norms=nullptr,*norm_inverses=nullptr,*multiplicities=nullptr,*scratch=nullptr,*trace_chunk=nullptr;ResidentLde* lde=nullptr;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&conjugates),message_count*sizeof(Ext2));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&norms),message_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&norm_inverses),message_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&multiplicities),message_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&deltas),count*sizeof(Ext2));
    if(status==cudaSuccess)status=create_resident_lde(&lde);
    if(status==cudaSuccess){lde->height=extended_height;lde->width=width;status=cudaMalloc(reinterpret_cast<void**>(&lde->values),extended_height*width*sizeof(uint64_t));}
    if(status==cudaSuccess)status=cudaMemset(lde->values,0,extended_height*width*sizeof(uint64_t));
    const size_t budget=48*1024;size_t tile=budget/(slot_count*sizeof(uint64_t));const bool global=tile<32;
    if(global)tile=128;else if(tile>32)tile=32;size_t blocks=(chunk_rows+tile-1)/tile;const size_t cap=global?256:1024;if(blocks>cap)blocks=cap;
    if(status==cudaSuccess&&global)status=cudaMalloc(reinterpret_cast<void**>(&scratch),blocks*slot_count*tile*sizeof(uint64_t));
    if(status==cudaSuccess&&main->host_trace_values)status=cudaMalloc(reinterpret_cast<void**>(&trace_chunk),(chunk_rows+1)*main->width*sizeof(uint64_t));
    for(size_t row_start=0;status==cudaSuccess&&row_start<height;row_start+=LOOKUP_ROWS_PER_CHUNK){
        const size_t rows=height-row_start<LOOKUP_ROWS_PER_CHUNK?height-row_start:LOOKUP_ROWS_PER_CHUNK;
        const size_t messages=rows*lookup_count;
        const size_t chunk_blocks_raw=(rows+tile-1)/tile;
        const size_t chunk_blocks=chunk_blocks_raw<cap?chunk_blocks_raw:cap;
        const uint64_t* active_trace=main->trace_values;
        if(main->host_trace_values){
            status=cudaMemcpy(trace_chunk,main->host_trace_values+row_start*main->width,rows*main->width*sizeof(uint64_t),cudaMemcpyHostToDevice);
            const size_t next_row=(row_start+rows)&(height-1);
            if(status==cudaSuccess)status=cudaMemcpy(trace_chunk+rows*main->width,main->host_trace_values+next_row*main->width,main->width*sizeof(uint64_t),cudaMemcpyHostToDevice);
            active_trace=trace_chunk;
        }
        if(status==cudaSuccess){lookup_messages_graph<<<static_cast<unsigned int>(chunk_blocks),static_cast<unsigned int>(tile),global?0:slot_count*tile*sizeof(uint64_t)>>>(
            conjugates,norms,multiplicities,dn,node_count,slot_count,dl,lookup_count,da,prep,active_trace,main->width,
            main->host_trace_values!=nullptr,{beta[0],beta[1]},{gamma[0],gamma[1]},ext_w,height,row_start,rows,scratch);status=cudaGetLastError();}
        if(status==cudaSuccess)status=batch_inverse_norms(norm_inverses,norms,messages);
        if(status==cudaSuccess){lookup_group_deltas_batched<<<blocks_for(rows*groups),THREADS>>>(
            deltas+row_start*groups,multiplicities,conjugates,norm_inverses,rows,lookup_count,group_size,ext_w);status=cudaGetLastError();}
    }
    if(status==cudaSuccess)status=exclusive_scan_ext2(reinterpret_cast<Ext2*>(lde->values),deltas,count);
    if(status==cudaSuccess)status=cudaMemcpy(total,lde->values+2*(count-1),sizeof(Ext2),cudaMemcpyDeviceToHost);
    if(status==cudaSuccess)status=cudaMemcpy(total+2,deltas+count-1,sizeof(Ext2),cudaMemcpyDeviceToHost);
    const uint64_t *dit=nullptr,*dshift=nullptr,*dft=nullptr;
    if(status==cudaSuccess)status=cached_device_constants(device_id,inverse_twiddles,height/2,1,0,0,&dit);
    if(status==cudaSuccess)status=cached_device_constants(device_id,shift_powers,height,3,shift_powers[0],height>1?shift_powers[1]:0,&dshift);
    if(status==cudaSuccess)status=cached_device_constants(device_id,forward_twiddles,extended_height/2,2,0,0,&dft);
    if(status==cudaSuccess)status=launch_dif(lde->values,height,width,dit);
    if(status==cudaSuccess){bit_reverse_scale_and_shift<<<blocks_for(height*width),THREADS>>>(lde->values,height,width,strict_log2(height),height_inverse,dshift);status=cudaGetLastError();}
    if(status==cudaSuccess)status=launch_dif(lde->values,extended_height,width,dft);
    if(status==cudaSuccess){canonicalize_goldilocks<<<blocks_for(extended_height*width),THREADS>>>(lde->values,extended_height*width);status=cudaGetLastError();}
    if(status==cudaSuccess)status=cudaStreamSynchronize(0);
    if(status==cudaSuccess)*output_handle=lde;else destroy_resident_lde(lde);
    cudaFree(trace_chunk);cudaFree(scratch);cudaFree(deltas);cudaFree(multiplicities);cudaFree(norm_inverses);cudaFree(norms);cudaFree(conjugates);cudaFree(metadata);
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_lookup_lde(int device_id,void** output_handle,uint64_t* total,
    const uint64_t* multiplicities,const uint64_t* args,const size_t* arg_offsets,
    size_t height,size_t num_lookups,size_t args_width,size_t group_size,
    const uint64_t* beta,const uint64_t* gamma,uint64_t ext_w,size_t added_bits,
    const uint64_t* inverse_twiddles,const uint64_t* shift_powers,
    const uint64_t* forward_twiddles,uint64_t height_inverse){
    if(!output_handle||!total||!multiplicities||!arg_offsets||!height||!num_lookups||
       !group_size||!beta||!gamma||!inverse_twiddles||!shift_powers||
       !forward_twiddles||(args_width&&!args)||!is_power_of_two(height)||
       added_bits>=sizeof(size_t)*8||height>(SIZE_MAX>>added_bits))
        return static_cast<int>(cudaErrorInvalidValue);
    *output_handle=nullptr;const size_t slots=(num_lookups+group_size-1)/group_size;
    const size_t width=2*slots,count=height*slots,extended_height=height<<added_bits;
    if(!product_fits(extended_height,width))return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status=cudaSetDevice(device_id);const size_t message_count=height*num_lookups;
    uint64_t *dm=nullptr,*da=nullptr,*norms=nullptr,*norm_inverses=nullptr;size_t* offsets=nullptr;
    Ext2 *conjugates=nullptr,*deltas=nullptr;ResidentLde* lde=nullptr;
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&dm),height*num_lookups*sizeof(uint64_t));
    if(status==cudaSuccess&&args_width)status=cudaMalloc(reinterpret_cast<void**>(&da),height*args_width*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&offsets),(num_lookups+1)*sizeof(size_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&deltas),count*sizeof(Ext2));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&conjugates),message_count*sizeof(Ext2));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&norms),message_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&norm_inverses),message_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=create_resident_lde(&lde);
    if(status==cudaSuccess){lde->height=extended_height;lde->width=width;status=cudaMalloc(reinterpret_cast<void**>(&lde->values),extended_height*width*sizeof(uint64_t));}
    if(status==cudaSuccess)status=cudaMemset(lde->values,0,extended_height*width*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMemcpy(dm,multiplicities,height*num_lookups*sizeof(uint64_t),cudaMemcpyHostToDevice);
    if(status==cudaSuccess&&args_width)status=cudaMemcpy(da,args,height*args_width*sizeof(uint64_t),cudaMemcpyHostToDevice);
    if(status==cudaSuccess)status=cudaMemcpy(offsets,arg_offsets,(num_lookups+1)*sizeof(size_t),cudaMemcpyHostToDevice);
    if(status==cudaSuccess){lookup_messages<<<blocks_for(message_count),THREADS>>>(conjugates,norms,da,offsets,height,num_lookups,args_width,{beta[0],beta[1]},{gamma[0],gamma[1]},ext_w);status=cudaGetLastError();}
    if(status==cudaSuccess)status=batch_inverse_norms(norm_inverses,norms,message_count);
    if(status==cudaSuccess){lookup_group_deltas_batched<<<blocks_for(count),THREADS>>>(deltas,dm,conjugates,norm_inverses,height,num_lookups,group_size,ext_w);status=cudaGetLastError();}
    if(status==cudaSuccess)status=exclusive_scan_ext2(reinterpret_cast<Ext2*>(lde->values),deltas,count);
    if(status==cudaSuccess)status=cudaMemcpy(total,lde->values+2*(count-1),sizeof(Ext2),cudaMemcpyDeviceToHost);
    if(status==cudaSuccess)status=cudaMemcpy(total+2,deltas+count-1,sizeof(Ext2),cudaMemcpyDeviceToHost);
    const uint64_t *dit=nullptr,*dshift=nullptr,*dft=nullptr;
    if(status==cudaSuccess)status=cached_device_constants(device_id,inverse_twiddles,height/2,1,0,0,&dit);
    if(status==cudaSuccess)status=cached_device_constants(device_id,shift_powers,height,3,shift_powers[0],height>1?shift_powers[1]:0,&dshift);
    if(status==cudaSuccess)status=cached_device_constants(device_id,forward_twiddles,extended_height/2,2,0,0,&dft);
    if(status==cudaSuccess)status=launch_dif(lde->values,height,width,dit);
    if(status==cudaSuccess){bit_reverse_scale_and_shift<<<blocks_for(height*width),THREADS>>>(lde->values,height,width,strict_log2(height),height_inverse,dshift);status=cudaGetLastError();}
    if(status==cudaSuccess)status=launch_dif(lde->values,extended_height,width,dft);
    if(status==cudaSuccess){canonicalize_goldilocks<<<blocks_for(extended_height*width),THREADS>>>(lde->values,extended_height*width);status=cudaGetLastError();}
    if(status==cudaSuccess)status=cudaStreamSynchronize(0);
    if(status==cudaSuccess)*output_handle=lde;else destroy_resident_lde(lde);
    cudaFree(norm_inverses);cudaFree(norms);cudaFree(conjugates);cudaFree(deltas);
    cudaFree(offsets);cudaFree(da);cudaFree(dm);return static_cast<int>(status);
}
extern "C" int multi_stark_cuda_reduced_add(int device_id,void* reduced,const void* lde_handle,size_t height,
    const uint64_t* inv_denoms,const uint64_t* alpha_powers,const uint64_t* reduced_y,const uint64_t* alpha_offset,uint64_t ext_w){
    if(!reduced||!lde_handle||!inv_denoms||!alpha_powers||!reduced_y||!alpha_offset)return static_cast<int>(cudaErrorInvalidValue);
    auto* r=static_cast<ResidentReducedOpening*>(reduced);auto* l=static_cast<const ResidentLde*>(lde_handle);if(height!=r->height||height>l->height)return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status=cudaSetDevice(device_id);size_t ib=height*sizeof(Ext2),ab=l->width*sizeof(Ext2),needed=ib+ab;
    if(status==cudaSuccess&&r->scratch_bytes<needed){if(r->scratch)status=cudaFree(r->scratch);r->scratch=nullptr;if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&r->scratch),needed);if(status==cudaSuccess)r->scratch_bytes=needed;}
    uint8_t* mem=r->scratch;if(status==cudaSuccess)status=cudaMemcpy(mem,inv_denoms,ib,cudaMemcpyHostToDevice);if(status==cudaSuccess)status=cudaMemcpy(mem+ib,alpha_powers,ab,cudaMemcpyHostToDevice);
    if(status==cudaSuccess){accumulate_reduced_opening<<<blocks_for(height),THREADS>>>(r->values,l,reinterpret_cast<Ext2*>(mem),reinterpret_cast<Ext2*>(mem+ib),height,{reduced_y[0],reduced_y[1]},{alpha_offset[0],alpha_offset[1]},ext_w);status=cudaGetLastError();}return static_cast<int>(status);
}
extern "C" int multi_stark_cuda_reduced_copy(int device_id,const void* handle,uint64_t* output){if(!handle||!output)return static_cast<int>(cudaErrorInvalidValue);auto* r=static_cast<const ResidentReducedOpening*>(handle);cudaError_t status=cudaSetDevice(device_id);if(status==cudaSuccess)status=cudaMemcpy(output,r->values,r->height*sizeof(Ext2),cudaMemcpyDeviceToHost);return static_cast<int>(status);}
extern "C" int multi_stark_cuda_reduced_destroy(int device_id,void* handle){cudaError_t status=cudaSetDevice(device_id);if(status==cudaSuccess)delete static_cast<ResidentReducedOpening*>(handle);return static_cast<int>(status);}

extern "C" int multi_stark_cuda_lde_destroy(int device_id, void* handle) {
    if (handle == nullptr) {
        return static_cast<int>(cudaSuccess);
    }
    const cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    return static_cast<int>(destroy_resident_lde(static_cast<ResidentLde*>(handle)));
}

extern "C" int multi_stark_cuda_lde_release_trace(int device_id,void* handle){
    if(!handle)return static_cast<int>(cudaSuccess);
    cudaError_t status=cudaSetDevice(device_id);
    auto* lde=static_cast<ResidentLde*>(handle);
    if(status==cudaSuccess&&lde->trace_values){
        status=cudaFree(lde->trace_values);
        if(status==cudaSuccess){
            lde->trace_values=nullptr;
            // Later waves may allocate on other per-thread streams. Ensure
            // this trace has returned to the pool before admitting them.
            status=cudaStreamSynchronize(cudaStreamPerThread);
        }
    }
    if(status==cudaSuccess&&lde->host_trace_registered){status=cudaHostUnregister(const_cast<uint64_t*>(lde->host_trace_values));lde->host_trace_registered=false;}
    if(status==cudaSuccess){lde->host_trace_values=nullptr;lde->trace_height=0;}
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_lde_attach_trace(
    int device_id, void* handle, const uint64_t* trace, size_t height,
    size_t width) {
    if (!handle || !trace || height == 0 || width == 0 ||
        !product_fits(height, width)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    auto* lde = static_cast<ResidentLde*>(handle);
    if (status != cudaSuccess || lde->width != width) {
        return status == cudaSuccess ? static_cast<int>(cudaErrorInvalidValue)
                                     : static_cast<int>(status);
    }
    if (lde->trace_values != nullptr) return static_cast<int>(cudaSuccess);
    if (status == cudaSuccess) {
        lde->host_trace_values = trace;
        lde->trace_height = height;
        const size_t bytes=height*width*sizeof(uint64_t);
        if(bytes>=(size_t(8)<<20)){
            const cudaError_t registration=cudaHostRegister(
                const_cast<uint64_t*>(trace),bytes,cudaHostRegisterDefault);
            if(registration==cudaSuccess)lde->host_trace_registered=true;
            else cudaGetLastError();
        }
    }
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_goldilocks_ops(
    int device_id, uint64_t* sums, uint64_t* differences, uint64_t* products,
    uint64_t* inverses, const uint64_t* left, const uint64_t* right,
    size_t len) {
    if (sums == nullptr || differences == nullptr || products == nullptr ||
        inverses == nullptr || left == nullptr || right == nullptr || len == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    DeviceBuffer device_left;
    DeviceBuffer device_right;
    DeviceBuffer device_sums;
    DeviceBuffer device_differences;
    DeviceBuffer device_products;
    DeviceBuffer device_inverses;
    status = copy_to_device(device_left, left, len);
    if (status == cudaSuccess) {
        status = copy_to_device(device_right, right, len);
    }
    if (status == cudaSuccess) {
        status = device_sums.allocate(len);
    }
    if (status == cudaSuccess) {
        status = device_differences.allocate(len);
    }
    if (status == cudaSuccess) {
        status = device_products.allocate(len);
    }
    if (status == cudaSuccess) {
        status = device_inverses.allocate(len);
    }
    if (status == cudaSuccess) {
        goldilocks_ops_kernel<<<blocks_for(len), THREADS>>>(
            device_sums.get(), device_differences.get(), device_products.get(),
            device_inverses.get(), device_left.get(), device_right.get(), len);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = copy_to_host(sums, device_sums, len);
    }
    if (status == cudaSuccess) {
        status = copy_to_host(differences, device_differences, len);
    }
    if (status == cudaSuccess) {
        status = copy_to_host(products, device_products, len);
    }
    if (status == cudaSuccess) {
        status = copy_to_host(inverses, device_inverses, len);
    }
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_blake3_hash_rows(
    int device_id, uint8_t* digests, const uint8_t* messages,
    size_t message_bytes, size_t message_count) {
    if (digests == nullptr || messages == nullptr || message_count == 0 ||
        message_bytes == 0 || message_bytes > 32 * 1024 ||
        !product_fits(message_bytes, message_count) ||
        !product_fits(static_cast<size_t>(32), message_count)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    const size_t input_bytes = message_bytes * message_count;
    const size_t output_bytes = 32 * message_count;
    DeviceBuffer device_messages;
    DeviceBuffer device_digests;
    status = device_messages.allocate((input_bytes + 7) / 8);
    if (status == cudaSuccess) {
        status = device_digests.allocate((output_bytes + 7) / 8);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_messages.get(), messages, input_bytes,
                            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = launch_blake3_rows(
            reinterpret_cast<uint8_t*>(device_digests.get()),
            reinterpret_cast<const uint8_t*>(device_messages.get()),
            message_bytes, message_count);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(digests, device_digests.get(), output_bytes,
                            cudaMemcpyDeviceToHost);
    }
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_blake3_merkle_root(
    int device_id, uint8_t* root, const uint8_t* rows, size_t row_bytes,
    size_t row_count) {
    if (root == nullptr || rows == nullptr || row_bytes == 0 ||
        row_bytes > 32 * 1024 || !is_power_of_two(row_count) ||
        !product_fits(row_bytes, row_count)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    const size_t input_bytes = row_bytes * row_count;
    DeviceBuffer device_rows;
    DeviceBuffer layer_a;
    DeviceBuffer layer_b;
    status = device_rows.allocate((input_bytes + 7) / 8);
    if (status == cudaSuccess) {
        status = layer_a.allocate(4 * row_count);
    }
    if (status == cudaSuccess && row_count > 1) {
        status = layer_b.allocate(4 * (row_count / 2));
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_rows.get(), rows, input_bytes,
                            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = launch_blake3_rows(
            reinterpret_cast<uint8_t*>(layer_a.get()),
            reinterpret_cast<const uint8_t*>(device_rows.get()), row_bytes,
            row_count);
    }

    size_t count = row_count;
    uint8_t* current = reinterpret_cast<uint8_t*>(layer_a.get());
    uint8_t* next = reinterpret_cast<uint8_t*>(layer_b.get());
    while (status == cudaSuccess && count > 1) {
        status = launch_blake3_digest_pairs(next, current, current, count / 2,
                                            true);
        count /= 2;
        uint8_t* swap = current;
        current = next;
        next = swap;
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(root, current, 32, cudaMemcpyDeviceToHost);
    }
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_merkle_create(
    int device_id, void** handle, uint8_t* root, const uint8_t* rows,
    size_t row_bytes, size_t row_count) {
    if (handle == nullptr || root == nullptr || rows == nullptr ||
        row_bytes == 0 || row_bytes > 32 * 1024 ||
        !is_power_of_two(row_count) || !product_fits(row_bytes, row_count) ||
        row_count > SIZE_MAX / 64) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *handle = nullptr;
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    ResidentMerkleTree* tree = new (std::nothrow) ResidentMerkleTree;
    if (tree == nullptr) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    }
    tree->row_bytes = row_bytes;
    tree->row_count = row_count;
    const size_t rows_bytes = row_bytes * row_count;
    status = cudaMalloc(reinterpret_cast<void**>(&tree->rows), rows_bytes);
    if (status == cudaSuccess) {
        // A full binary tree has fewer than 2*n digests. Keeping all layers in
        // one allocation makes the prover-data handle compact and openings
        // independent of host-side allocation lifetimes.
        status = cudaMalloc(reinterpret_cast<void**>(&tree->digests),
                            64 * row_count);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(tree->rows, rows, rows_bytes,
                            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = launch_blake3_rows(tree->digests, tree->rows, row_bytes,
                                    row_count);
    }

    size_t count = row_count;
    size_t offset = 0;
    while (status == cudaSuccess && count > 1) {
        uint8_t* current = tree->digests + offset;
        offset += count * 32;
        status = launch_blake3_digest_pairs(tree->digests + offset, current,
                                            current, count / 2, true);
        count /= 2;
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(root, tree->digests + offset, 32,
                            cudaMemcpyDeviceToHost);
    }
    if (status != cudaSuccess) {
        delete tree;
        return static_cast<int>(status);
    }
    *handle = tree;
    return static_cast<int>(cudaSuccess);
}

__global__ void gather_merkle_siblings(uint8_t* output,const uint8_t* digests,
    size_t row_count,size_t index,size_t levels){
    const size_t byte=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(byte>=levels*32)return;const size_t level=byte/32,within=byte%32;
    const size_t count=row_count>>level;
    const size_t offset=(2*row_count-2*count)*32;
    const size_t sibling=(index>>level)^1U;
    output[byte]=digests[offset+sibling*32+within];
}

__global__ void gather_merkle_siblings_batch(uint8_t* output,const uint8_t* digests,
    size_t row_count,const uint64_t* indices,size_t query_count,size_t levels){
    const size_t byte=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t path_bytes=levels*32;if(byte>=query_count*path_bytes)return;
    const size_t query=byte/path_bytes,path_byte=byte-query*path_bytes;
    const size_t level=path_byte/32,within=path_byte%32,count=row_count>>level;
    const size_t offset=(2*row_count-2*count)*32;
    const size_t sibling=(static_cast<size_t>(indices[query])>>level)^1U;
    output[byte]=digests[offset+sibling*32+within];
}

cudaError_t copy_merkle_siblings(uint8_t* output,const uint8_t* digests,
    size_t row_count,size_t index){
    const size_t levels=strict_log2(row_count),bytes=levels*32;
    if(levels==0)return cudaSuccess;
    uint8_t* gathered=nullptr;cudaError_t status=cudaMalloc(
        reinterpret_cast<void**>(&gathered),bytes);
    if(status==cudaSuccess){gather_merkle_siblings<<<blocks_for(bytes),THREADS>>>(
        gathered,digests,row_count,index,levels);status=cudaGetLastError();}
    if(status==cudaSuccess)status=cudaMemcpy(output,gathered,bytes,cudaMemcpyDeviceToHost);
    if(gathered)cudaFree(gathered);return status;
}

cudaError_t copy_merkle_siblings_batch(uint8_t* output,const uint8_t* digests,
    size_t row_count,const uint64_t* indices,size_t query_count){
    const size_t levels=strict_log2(row_count),bytes=query_count*levels*32;
    uint64_t* device_indices=nullptr;uint8_t* gathered=nullptr;
    cudaError_t status=cudaMalloc(reinterpret_cast<void**>(&device_indices),query_count*sizeof(uint64_t));
    if(status==cudaSuccess)status=cudaMalloc(reinterpret_cast<void**>(&gathered),bytes);
    if(status==cudaSuccess)status=cudaMemcpy(device_indices,indices,query_count*sizeof(uint64_t),cudaMemcpyHostToDevice);
    if(status==cudaSuccess){gather_merkle_siblings_batch<<<blocks_for(bytes),THREADS>>>(gathered,digests,row_count,device_indices,query_count,levels);status=cudaGetLastError();}
    if(status==cudaSuccess)status=cudaMemcpy(output,gathered,bytes,cudaMemcpyDeviceToHost);
    cudaFree(gathered);cudaFree(device_indices);return status;
}

extern "C" int multi_stark_cuda_merkle_open(
    int device_id, const void* handle, size_t index, uint8_t* row,
    uint8_t* siblings) {
    if (handle == nullptr || row == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    const ResidentMerkleTree* tree =
        static_cast<const ResidentMerkleTree*>(handle);
    if (index >= tree->row_count || (tree->row_count > 1 && siblings == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    status = cudaMemcpy(row, tree->rows + index * tree->row_bytes,
                        tree->row_bytes, cudaMemcpyDeviceToHost);
    if(status==cudaSuccess)status=copy_merkle_siblings(
        siblings,tree->digests,tree->row_count,index);
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_merkle_destroy(int device_id, void* handle) {
    if (handle == nullptr) {
        return static_cast<int>(cudaSuccess);
    }
    const cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    delete static_cast<ResidentMerkleTree*>(handle);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int multi_stark_cuda_mixed_merkle_create(
    int device_id, void** handle, uint8_t* root,
    const uint8_t* const* level_rows, const size_t* level_row_bytes,
    const size_t* level_heights, size_t level_count) {
    if (handle == nullptr || root == nullptr || level_rows == nullptr ||
        level_row_bytes == nullptr || level_heights == nullptr ||
        level_count == 0 || !is_power_of_two(level_heights[0])) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    for (size_t level = 0; level < level_count; ++level) {
        if (level_rows[level] == nullptr || level_row_bytes[level] == 0 ||
            level_row_bytes[level] > 32 * 1024 ||
            !is_power_of_two(level_heights[level]) ||
            (level > 0 && level_heights[level] >= level_heights[level - 1]) ||
            !product_fits(level_row_bytes[level], level_heights[level])) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    *handle = nullptr;
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    ResidentMixedMerkleTree* tree =
        new (std::nothrow) ResidentMixedMerkleTree;
    if (tree == nullptr) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    }
    tree->row_count = level_heights[0];
    status = cudaMalloc(reinterpret_cast<void**>(&tree->digests),
                        64 * tree->row_count);

    DeviceBuffer device_rows;
    DeviceBuffer injected_digests;
    const size_t first_bytes = level_row_bytes[0] * level_heights[0];
    if (status == cudaSuccess) {
        status = device_rows.allocate((first_bytes + 7) / 8);
    }
    if (status == cudaSuccess) {
        status = injected_digests.allocate(4 * tree->row_count);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_rows.get(), level_rows[0], first_bytes,
                            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = launch_blake3_rows(
            tree->digests,
            reinterpret_cast<const uint8_t*>(device_rows.get()),
            level_row_bytes[0], level_heights[0]);
    }

    size_t group = 1;
    size_t count = tree->row_count;
    size_t offset = 0;
    while (status == cudaSuccess && count > 1) {
        uint8_t* current = tree->digests + offset;
        offset += count * 32;
        count /= 2;
        uint8_t* next = tree->digests + offset;
        status = launch_blake3_digest_pairs(next, current, current, count,
                                            true);

        if (status == cudaSuccess && group < level_count &&
            level_heights[group] == count) {
            const size_t bytes = level_row_bytes[group] * count;
            DeviceBuffer injected_rows;
            status = injected_rows.allocate((bytes + 7) / 8);
            if (status == cudaSuccess) {
                status = cudaMemcpy(injected_rows.get(), level_rows[group],
                                    bytes, cudaMemcpyHostToDevice);
            }
            if (status == cudaSuccess) {
                status = launch_blake3_rows(
                    reinterpret_cast<uint8_t*>(injected_digests.get()),
                    reinterpret_cast<const uint8_t*>(injected_rows.get()),
                    level_row_bytes[group], count);
            }
            if (status == cudaSuccess) {
                status = launch_blake3_digest_pairs(
                    next, next,
                    reinterpret_cast<const uint8_t*>(injected_digests.get()),
                    count, false);
            }
            ++group;
        }
    }
    if (status == cudaSuccess && group != level_count) {
        status = cudaErrorInvalidValue;
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(root, tree->digests + offset, 32,
                            cudaMemcpyDeviceToHost);
    }
    if (status != cudaSuccess) {
        delete tree;
        return static_cast<int>(status);
    }
    *handle = tree;
    return static_cast<int>(cudaSuccess);
}

extern "C" int multi_stark_cuda_mixed_merkle_open(
    int device_id, const void* handle, size_t index, uint8_t* siblings) {
    if (handle == nullptr || siblings == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    const ResidentMixedMerkleTree* tree =
        static_cast<const ResidentMixedMerkleTree*>(handle);
    if (index >= tree->row_count) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    status=copy_merkle_siblings(siblings,tree->digests,tree->row_count,index);
    return static_cast<int>(status);
}

extern "C" int multi_stark_cuda_mixed_merkle_open_batch(
    int device_id, const void* handle, const uint64_t* indices,
    size_t query_count, uint8_t* siblings) {
    if(handle==nullptr||indices==nullptr||query_count==0||siblings==nullptr)return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status=cudaSetDevice(device_id);if(status!=cudaSuccess)return static_cast<int>(status);
    const ResidentMixedMerkleTree* tree=static_cast<const ResidentMixedMerkleTree*>(handle);
    for(size_t q=0;q<query_count;++q)if(indices[q]>=tree->row_count)return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(copy_merkle_siblings_batch(siblings,tree->digests,tree->row_count,indices,query_count));
}

extern "C" int multi_stark_cuda_mixed_merkle_destroy(int device_id,
                                                       void* handle) {
    if (handle == nullptr) {
        return static_cast<int>(cudaSuccess);
    }
    const cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    delete static_cast<ResidentMixedMerkleTree*>(handle);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int multi_stark_cuda_mixed_merkle_create_from_ldes(
    int device_id, void** handle, uint8_t* root,
    const void* const* lde_handles, size_t lde_count) {
    if (handle == nullptr || root == nullptr || lde_handles == nullptr ||
        lde_count == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    size_t max_height = 0;
    for (size_t index = 0; index < lde_count; ++index) {
        const ResidentLde* lde =
            static_cast<const ResidentLde*>(lde_handles[index]);
        if (lde == nullptr || !is_power_of_two(lde->height) || lde->width == 0) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        if (lde->height > max_height) {
            max_height = lde->height;
        }
    }
    if (max_height > SIZE_MAX / 64) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *handle = nullptr;
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    ResidentMixedMerkleTree* tree =
        new (std::nothrow) ResidentMixedMerkleTree;
    if (tree == nullptr) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    }
    tree->row_count = max_height;
    status = cudaMalloc(reinterpret_cast<void**>(&tree->digests),
                        64 * max_height);
    size_t max_injected_height = 0;
    for (size_t index = 0; index < lde_count; ++index) {
        const ResidentLde* lde =
            static_cast<const ResidentLde*>(lde_handles[index]);
        if (lde->height < max_height && lde->height > max_injected_height) {
            max_injected_height = lde->height;
        }
    }
    DeviceBuffer injected_digests;
    if (status == cudaSuccess && max_injected_height != 0) {
        status = injected_digests.allocate(4 * max_injected_height);
    }
    if (status == cudaSuccess) {
        status = hash_resident_lde_group(tree->digests, lde_handles,
                                         lde_count, max_height);
    }

    size_t count = max_height;
    size_t offset = 0;
    while (status == cudaSuccess && count > 1) {
        uint8_t* current = tree->digests + offset;
        offset += count * 32;
        count /= 2;
        uint8_t* next = tree->digests + offset;
        status = launch_blake3_digest_pairs(next, current, current, count,
                                            true);

        bool inject = false;
        for (size_t index = 0; index < lde_count; ++index) {
            const ResidentLde* lde =
                static_cast<const ResidentLde*>(lde_handles[index]);
            inject = inject || lde->height == count;
        }
        if (status == cudaSuccess && inject) {
            status = hash_resident_lde_group(
                reinterpret_cast<uint8_t*>(injected_digests.get()),
                lde_handles, lde_count, count);
        }
        if (status == cudaSuccess && inject) {
            status = launch_blake3_digest_pairs(
                next, next,
                reinterpret_cast<const uint8_t*>(injected_digests.get()),
                count, false);
        }
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(root, tree->digests + offset, 32,
                            cudaMemcpyDeviceToHost);
    }
    if (status != cudaSuccess) {
        delete tree;
        return static_cast<int>(status);
    }
    *handle = tree;
    return static_cast<int>(cudaSuccess);
}

extern "C" int multi_stark_cuda_fri_merkle_create(
    int device_id,void** handle,uint8_t* root,const void* codeword_handle,size_t arity){
    if(!codeword_handle||!is_power_of_two(arity))return static_cast<int>(cudaErrorInvalidValue);
    const auto* codeword=static_cast<const ResidentLde*>(codeword_handle);
    if(codeword->width!=2||codeword->height%arity)return static_cast<int>(cudaErrorInvalidValue);
    ResidentLde view;view.values=codeword->values;view.height=codeword->height/arity;view.width=2*arity;
    const void* views[1]={&view};
    const int status=multi_stark_cuda_mixed_merkle_create_from_ldes(device_id,handle,root,views,1);
    view.values=nullptr;
    return status;
}

extern "C" int multi_stark_cuda_mixed_merkle_create_from_host_matrices(
    int device_id, void** handle, uint8_t* root,
    const uint64_t* const* matrix_values, const size_t* heights,
    const size_t* widths, size_t matrix_count) {
    if (handle == nullptr || root == nullptr || matrix_values == nullptr ||
        heights == nullptr || widths == nullptr || matrix_count == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    void** lde_handles = new (std::nothrow) void*[matrix_count];
    if (lde_handles == nullptr) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    }
    for (size_t index = 0; index < matrix_count; ++index) {
        lde_handles[index] = nullptr;
    }
    size_t created = 0;
    for (; status == cudaSuccess && created < matrix_count; ++created) {
        if (matrix_values[created] == nullptr ||
            !is_power_of_two(heights[created]) || widths[created] == 0 ||
            !product_fits(heights[created], widths[created])) {
            status = cudaErrorInvalidValue;
            break;
        }
        ResidentLde* matrix = nullptr;
        status = create_resident_lde(&matrix);
        if (status != cudaSuccess) break;
        matrix->height = heights[created];
        matrix->width = widths[created];
        status = cudaMalloc(reinterpret_cast<void**>(&matrix->values),
                            matrix->height * matrix->width * sizeof(uint64_t));
        if (status == cudaSuccess) {
            status = cudaMemcpy(matrix->values, matrix_values[created],
                                matrix->height * matrix->width * sizeof(uint64_t),
                                cudaMemcpyHostToDevice);
        }
        if (status != cudaSuccess) {
            destroy_resident_lde(matrix);
            break;
        }
        lde_handles[created] = matrix;
    }
    if (status == cudaSuccess) {
        status = static_cast<cudaError_t>(
            multi_stark_cuda_mixed_merkle_create_from_ldes(
                device_id, handle, root,
                const_cast<const void* const*>(lde_handles), matrix_count));
    }
    for (size_t index = 0; index < created; ++index) {
        destroy_resident_lde(static_cast<ResidentLde*>(lde_handles[index]));
    }
    delete[] lde_handles;
    return static_cast<int>(status);
}

extern "C" const char* multi_stark_cuda_error_string(int status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}

extern "C" int multi_stark_cuda_memory_info(int device_id, size_t* free_bytes,
                                              size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaSetDevice(device_id);
    if (status == cudaSuccess) status = cudaMemGetInfo(free_bytes, total_bytes);
    return static_cast<int>(status);
}
