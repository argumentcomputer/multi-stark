// Goldilocks transforms on the caller's stream, with immutable panel plans.
#include <ff/goldilocks.hpp>
#include <ntt/ntt.cuh>
#include "ntt.cuh"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

bool valid_arguments(const void* d_inout, uint32_t lg, int order, int direction, int coset) {
    return d_inout && lg > 0 && lg <= MAX_LG_DOMAIN_SIZE && order >= 0 && order <= 3 &&
           direction >= 0 && direction <= 1 && coset >= 0 && coset <= 1;
}

}  // namespace

extern "C" int multi_stark_sppark_max_lg_domain() { return MAX_LG_DOMAIN_SIZE; }

// `batch` in-place transforms of 2^lg field elements each, `stride`
// elements apart from `d_inout` on `device`, in one launch sequence.
// `order` is NTT::InputOutputOrder (NN, NR, RN, RR), `direction` 0 forward
// or 1 inverse, `coset` 1 for the multiplicative coset by the field
// generator. Inverse transforms are normalized by 1/2^lg upstream.
static int ntt_batch_on_stream(int device, uint64_t* d_inout, uint32_t lg,
                                                    int order, int direction, int coset,
                                                    uint32_t batch, size_t stride, cudaStream_t caller) {
    // Upstream indexes with index_t, 32 bits at this domain limit.
    if (!valid_arguments(d_inout, lg, order, direction, coset) || batch == 0 || batch > 65535 ||
        (batch > 1 && stride < (size_t(1) << lg)) || stride > size_t(std::numeric_limits<index_t>::max()))
        return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) return static_cast<int>(status);
    // Upstream keeps only the devices it supports, so its logical index can
    // differ from the CUDA ordinal; find the entry by ordinal.
    const gpu_t* found = nullptr;
    for (const gpu_t* candidate : all_gpus())
        if (candidate->cid() == device) found = candidate;
    if (!found) return static_cast<int>(cudaErrorInvalidDevice);
    const gpu_t& gpu = select_gpu(found->id());
    stream_t stream(gpu.id(), caller);
    NTT::Base_dev_ptr_batch(stream, reinterpret_cast<fr_t*>(d_inout), lg,
                            static_cast<NTT::InputOutputOrder>(order),
                            static_cast<NTT::Direction>(direction),
                            static_cast<NTT::Type>(coset), batch, stride);
    return static_cast<int>(cudaSuccess);
}

extern "C" int multi_stark_sppark_ntt_batch_device(int device, uint64_t* data, uint32_t lg,
                                                    int order, int direction, int coset,
                                                    uint32_t batch, size_t stride) {
    return ntt_batch_on_stream(device, data, lg, order, direction, coset, batch, stride, cudaStreamPerThread);
}

// A fresh caller-owned stream carrying producer, transforms and consumer.
// Reusing it after each borrowed wrapper is dropped checks its ownership.
extern "C" int multi_stark_sppark_borrowed_round_trip(int device, uint64_t* values, uint32_t lg) {
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) return static_cast<int>(status);
    cudaStream_t caller = nullptr;
    status = cudaStreamCreateWithFlags(&caller, cudaStreamNonBlocking);
    if (status != cudaSuccess) return static_cast<int>(status);
    uint64_t* data = nullptr;
    const size_t count = size_t(1) << lg, bytes = count * sizeof(uint64_t);
    status = cudaMallocAsync(reinterpret_cast<void**>(&data), bytes, caller);
    if (status == cudaSuccess) status = cudaMemcpyAsync(data, values, bytes, cudaMemcpyHostToDevice, caller);
    int result = static_cast<int>(status);
    if (result == 0) result = ntt_batch_on_stream(device, data, lg, 1, 0, 0, 1, count, caller);
    if (result == 0) result = ntt_batch_on_stream(device, data, lg, 2, 1, 0, 1, count, caller);
    if (result == 0) result = static_cast<int>(cudaMemcpyAsync(values, data, bytes, cudaMemcpyDeviceToHost, caller));
    if (data) cudaFreeAsync(data, caller);
    const auto synced = cudaStreamSynchronize(caller);
    if (result == 0) result = static_cast<int>(synced);
    cudaStreamDestroy(caller);
    return result;
}

// One transform: the batch entry with a single vector.
extern "C" int multi_stark_sppark_ntt_device(int device, uint64_t* d_inout, uint32_t lg,
                                              int order, int direction, int coset) {
    return multi_stark_sppark_ntt_batch_device(device, d_inout, lg, order, direction, coset, 1,
                                               size_t(1) << lg);
}

// The batched transform on host memory: `batch * stride` words uploaded,
// transformed, downloaded and synchronized. For contract checks and small
// inputs, not the prover.
extern "C" int multi_stark_sppark_ntt_batch_host(int device, uint64_t* inout, uint32_t lg, int order,
                                                  int direction, int coset, uint32_t batch,
                                                  size_t stride) {
    if (!valid_arguments(inout, lg, order, direction, coset) || batch == 0 || stride < (size_t(1) << lg))
        return static_cast<int>(cudaErrorInvalidValue);
    const size_t bytes = batch * stride * sizeof(uint64_t);
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) return static_cast<int>(status);
    uint64_t* d_inout = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&d_inout), bytes);
    if (status != cudaSuccess) return static_cast<int>(status);
    status = cudaMemcpyAsync(d_inout, inout, bytes, cudaMemcpyHostToDevice, cudaStreamPerThread);
    int result = static_cast<int>(status);
    if (result == 0)
        result = multi_stark_sppark_ntt_batch_device(device, d_inout, lg, order, direction, coset, batch, stride);
    if (result == 0)
        result = static_cast<int>(
            cudaMemcpyAsync(inout, d_inout, bytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    const cudaError_t synced = cudaStreamSynchronize(cudaStreamPerThread);
    if (result == 0) result = static_cast<int>(synced);
    cudaFree(d_inout);
    return result;
}

// One transform on host memory: the batch entry with a single vector.
extern "C" int multi_stark_sppark_ntt_host(int device, uint64_t* inout, uint32_t lg, int order,
                                            int direction, int coset) {
    return multi_stark_sppark_ntt_batch_host(device, inout, lg, order, direction, coset, 1, size_t(1) << lg);
}

// Row-major matrices are gathered into contiguous column panels for the
// inverse NTT, coefficient-order restoration with coset shift and expansion,
// and forward NR transform. Scatter preserves the bit-reversed row storage
// consumed by commitments. Gather canonicalizes lazy field representatives.
#include "goldilocks.cuh"

namespace {

constexpr unsigned PANEL_THREADS = 256;
constexpr size_t MAX_BLOCKS = 65535;
constexpr unsigned TILE = 32;
constexpr unsigned TILE_ROWS = 8;
// Matrices narrower than this take the row-per-thread gather and scatter,
// whose per-thread row segments are then a few contiguous words; from here
// up the tiled transposes win.
constexpr size_t NARROW_MATRIX = 8;

unsigned blocks_for_total(size_t total) {
    const size_t blocks = (total + PANEL_THREADS - 1) / PANEL_THREADS;
    return static_cast<unsigned>(blocks < MAX_BLOCKS ? blocks : MAX_BLOCKS);
}

// Tiles over rows on the grid's first dimension, which is wide enough for
// 2^26 rows, and over columns on the second.
dim3 tile_grid(size_t rows, size_t columns) {
    return dim3(static_cast<unsigned>((rows + TILE - 1) / TILE), static_cast<unsigned>((columns + TILE - 1) / TILE));
}

__device__ __forceinline__ size_t reverse_bits(size_t index, unsigned log) {
    return log == 0 ? 0 : static_cast<size_t>(__brev(static_cast<unsigned>(index)) >> (32 - log));
}

// Row-major rows [0, height) of columns [first, first + count) into the
// column-major panel (columns `column_stride` apart), canonical, through
// a shared-memory tile so both sides are coalesced.
__global__ void gather_tiles(const uint64_t* __restrict__ source, size_t height, size_t width, size_t first,
                             size_t count, size_t column_stride, uint64_t* __restrict__ panel) {
    __shared__ uint64_t tile[TILE][TILE + 1];
    const size_t row0 = static_cast<size_t>(blockIdx.x) * TILE;
    const size_t col0 = static_cast<size_t>(blockIdx.y) * TILE;
    for (unsigned r = threadIdx.y; r < TILE; r += TILE_ROWS) {
        const size_t row = row0 + r, col = col0 + threadIdx.x;
        if (row < height && col < count)
            tile[r][threadIdx.x] = multi_stark_cuda::canonicalize(source[row * width + first + col]);
    }
    __syncthreads();
    for (unsigned c = threadIdx.y; c < TILE; c += TILE_ROWS) {
        const size_t col = col0 + c, row = row0 + threadIdx.x;
        if (row < height && col < count) panel[col * column_stride + row] = tile[threadIdx.x][c];
    }
}

// The column panel back into row-major storage without a row permutation.
__global__ void scatter_tiles(const uint64_t* __restrict__ panel, size_t rows, size_t column_stride, size_t width,
                              size_t first, size_t count,
                              uint64_t* __restrict__ values) {
    __shared__ uint64_t tile[TILE][TILE + 1];
    const size_t row0 = static_cast<size_t>(blockIdx.x) * TILE;
    const size_t col0 = static_cast<size_t>(blockIdx.y) * TILE;
    for (unsigned c = threadIdx.y; c < TILE; c += TILE_ROWS) {
        const size_t col = col0 + c, row = row0 + threadIdx.x;
        if (row < rows && col < count) tile[c][threadIdx.x] = panel[col * column_stride + row];
    }
    __syncthreads();
    for (unsigned r = threadIdx.y; r < TILE; r += TILE_ROWS) {
        const size_t row = row0 + r, col = col0 + threadIdx.x;
        if (row < rows && col < count) {
            values[row * width + first + col] = tile[threadIdx.x][r];
        }
    }
}

// The expansion with the coefficient order restored: B[c][i] =
// A[c][rev(i)] * shift^i for i < N, zero beyond, so the forward transform
// runs in NR order and the scatter stays natural. The permuted reads are
// cheap while the compact panel fits the L2 cache. One column per grid row.
__global__ void shift_columns(const uint64_t* __restrict__ coefficients, const uint64_t* __restrict__ shift_powers,
                              size_t height, unsigned log_height, size_t extended_height,
                              uint64_t* __restrict__ panel) {
    const size_t column = blockIdx.y;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < extended_height; i += stride) {
        uint64_t value = 0;
        if (i < height)
            value = multi_stark_cuda::goldilocks_mul(coefficients[column * height + reverse_bits(i, log_height)],
                                                     shift_powers[i]);
        panel[column * extended_height + i] = value;
    }
}

// The gather and scatter for a narrow panel: one thread per row moves its
// few words, the panel side coalesced across the block; the tiled
// transpose would leave most of its block idle on such panels.
__global__ void gather_rows(const uint64_t* __restrict__ source, size_t height, size_t width, size_t first,
                            unsigned count, size_t column_stride, uint64_t* __restrict__ panel) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < height; row += stride) {
        const uint64_t* in = source + row * width + first;
        for (unsigned c = 0; c < count; ++c) panel[c * column_stride + row] = multi_stark_cuda::canonicalize(in[c]);
    }
}

__global__ void scatter_rows(const uint64_t* __restrict__ panel, size_t rows, size_t column_stride, size_t width,
                             size_t first, unsigned count,
                             uint64_t* __restrict__ values) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < rows; row += stride) {
        uint64_t* out = values + row * width + first;
        for (unsigned c = 0; c < count; ++c) out[c] = panel[c * column_stride + row];
    }
}

// The panel's gather and scatter by the matrix width.
cudaError_t gather_panel(const uint64_t* source, size_t height, size_t width, size_t first, size_t count,
                         size_t column_stride, uint64_t* panel) {
    if (width < NARROW_MATRIX)
        gather_rows<<<blocks_for_total(height), PANEL_THREADS, 0, cudaStreamPerThread>>>(
            source, height, width, first, static_cast<unsigned>(count), column_stride, panel);
    else
        gather_tiles<<<tile_grid(height, count), dim3(TILE, TILE_ROWS), 0, cudaStreamPerThread>>>(
            source, height, width, first, count, column_stride, panel);
    return cudaGetLastError();
}

cudaError_t scatter_panel(const uint64_t* panel, size_t rows, size_t column_stride, size_t width, size_t first,
                          size_t count,  uint64_t* values) {
    if (width < NARROW_MATRIX)
        scatter_rows<<<blocks_for_total(rows), PANEL_THREADS, 0, cudaStreamPerThread>>>(
            panel, rows, column_stride, width, first, static_cast<unsigned>(count), values);
    else
        scatter_tiles<<<tile_grid(rows, count), dim3(TILE, TILE_ROWS), 0, cudaStreamPerThread>>>(
            panel, rows, column_stride, width, first, count, values);
    return cudaGetLastError();
}

unsigned log2_exact(size_t value) {
    unsigned log = 0;
    while ((size_t(1) << log) < value) ++log;
    return log;
}

// A decimal setting, or `fallback` when unset or not a plain number. This
// unit is compiled by nvcc's host compiler without the C standard pin the
// crate's C units get, where glibc redirects strtoul to a C23 symbol the
// Lean toolchain's libc does not carry.
unsigned long long decimal_setting(const char* name, unsigned long long fallback) {
    const char* configured = getenv(name);
    if (!configured || !*configured) return fallback;
    unsigned long long value = 0;
    for (const char* c = configured; *c; ++c) {
        if (*c < '0' || *c > '9' || value > (~0ull - 9) / 10) return fallback;
        value = value * 10 + unsigned(*c - '0');
    }
    return value;
}

// MULTI_STARK_SPPARK_STAGE_TIMING=1 prints the stage times of every coset
// LDE to stderr: events on the caller's stream around each stage, which
// the transforms are fenced to, and one synchronization per panel.
struct StageTimer {
    static constexpr int STAGES = 5;
    bool enabled = decimal_setting("MULTI_STARK_SPPARK_STAGE_TIMING", 0) != 0;
    cudaEvent_t marks[STAGES + 1] = {};
    StageTimer() {
        if (!enabled) return;
        for (auto& mark : marks)
            if (cudaEventCreate(&mark) != cudaSuccess) enabled = false;
    }
    ~StageTimer() {
        for (auto mark : marks)
            if (mark) cudaEventDestroy(mark);
    }
    void mark(int stage) {
        if (enabled) cudaEventRecord(marks[stage], cudaStreamPerThread);
    }
    void report(size_t height, size_t width, size_t added_bits) {
        if (!enabled || cudaEventSynchronize(marks[STAGES]) != cudaSuccess) return;
        static const char* const names[STAGES] = {"gather", "inverse", "restore", "forward", "scatter"};
        fprintf(stderr, "sppark lde height=%zu width=%zu added_bits=%zu", height, width, added_bits);
        for (int stage = 0; stage < STAGES; ++stage) {
            float ms = 0;
            cudaEventElapsedTime(&ms, marks[stage], marks[stage + 1]);
            fprintf(stderr, " %s=%.3f", names[stage], ms);
        }
        fprintf(stderr, "\n");
    }
};

// The `count` columns of a panel, `stride` elements apart, through batched
// launch sequences of as many columns as the group budget holds.
int transform_columns(int device, uint64_t* panel, uint32_t lg, int order, int direction, size_t count,
                      size_t stride, size_t group) {
    if (lg == 0) return 0;
    int result = 0;
    for (size_t first = 0; result == 0 && first < count; first += group) {
        const size_t batch = count - first < group ? count - first : group;
        result = multi_stark_sppark_ntt_batch_device(device, panel + first * stride, lg, order, direction, 0,
                                                     static_cast<uint32_t>(batch), stride);
    }
    return result;
}

}  // namespace

extern "C" int multi_stark_sppark_l2_bytes(int device, size_t* bytes) {
    int l2 = 0;
    const cudaError_t status = cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device);
    if (status == cudaSuccess) *bytes = l2 > 0 ? size_t(l2) : size_t(64) << 20;
    return static_cast<int>(status);
}

static std::atomic<uint64_t> transforms_run{0};

extern "C" uint64_t multi_stark_sppark_transforms_run() {
    return transforms_run.load(std::memory_order_relaxed);
}

extern "C" int multi_stark_sppark_coset_lde(int device, const uint64_t* trace, uint64_t* values,
                                            const MultiStarkNttPlan* plan,
                                            const uint64_t* shift_powers) {
    transforms_run.fetch_add(1, std::memory_order_relaxed);
    const size_t height = plan->height, width = plan->width;
    const size_t extended_height = plan->extended_height, columns = plan->columns;
    const unsigned log_height = log2_exact(height), log_extended = log2_exact(extended_height);
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) return static_cast<int>(status);
    // Pool allocation and freeing follow all panel kernels on the same stream.
    // CUPTI 2026.2.1 faults when tracing an async free of non-pool memory.
    uint64_t* scratch = nullptr;
    status = cudaMallocAsync(reinterpret_cast<void**>(&scratch), plan->scratch_bytes, cudaStreamPerThread);
    if (status != cudaSuccess) return static_cast<int>(status);
    uint64_t* a = scratch;
    uint64_t* b = scratch + columns * height;
    int result = 0;
    StageTimer timer;
    for (size_t first = 0; result == 0 && first < width; first += columns) {
        const size_t count = width - first < columns ? width - first : columns;
        timer.mark(0);
        result = static_cast<int>(gather_panel(trace, height, width, first, count, height, a));
        timer.mark(1);
        if (result == 0) result = transform_columns(device, a, log_height, 1, 1, count, height, plan->inverse_group);
        timer.mark(2);
        if (result == 0) {
            const dim3 column_grid(blocks_for_total(extended_height), static_cast<unsigned>(count));
            shift_columns<<<column_grid, PANEL_THREADS, 0, cudaStreamPerThread>>>(
                a, shift_powers, height, log_height, extended_height, b);
            result = static_cast<int>(cudaGetLastError());
        }
        timer.mark(3);
        if (result == 0) result = transform_columns(device, b, log_extended, 1, 0, count, extended_height, plan->forward_group);
        timer.mark(4);
        if (result == 0)
            result = static_cast<int>(scatter_panel(b, extended_height, extended_height, width, first, count, values));
        timer.mark(5);
        if (result == 0) timer.report(height, count, log_extended - log_height);
    }
    const cudaError_t freed = cudaFreeAsync(scratch, cudaStreamPerThread);
    if (result == 0) result = static_cast<int>(freed);
    return result;
}

extern "C" int multi_stark_sppark_forward(int device, uint64_t* values, const MultiStarkNttPlan* plan) {
    transforms_run.fetch_add(1, std::memory_order_relaxed);
    const size_t height = plan->height, width = plan->width, columns = plan->columns;
    const unsigned log_height = log2_exact(height);
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) return static_cast<int>(status);
    uint64_t* panel = nullptr;
    status = cudaMallocAsync(reinterpret_cast<void**>(&panel), plan->scratch_bytes, cudaStreamPerThread);
    if (status != cudaSuccess) return static_cast<int>(status);
    int result = 0;
    for (size_t first = 0; result == 0 && first < width; first += columns) {
        const size_t count = width - first < columns ? width - first : columns;
        result = static_cast<int>(gather_panel(values, height, width, first, count, height, panel));
        if (result == 0) result = transform_columns(device, panel, log_height, 1, 0, count, height, plan->forward_group);
        if (result == 0)
            result = static_cast<int>(scatter_panel(panel, height, height, width, first, count, values));
    }
    const cudaError_t freed = cudaFreeAsync(panel, cudaStreamPerThread);
    if (result == 0) result = static_cast<int>(freed);
    return result;
}
