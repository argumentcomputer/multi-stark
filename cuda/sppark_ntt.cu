// sppark's Goldilocks NTT behind a C interface for the comparison backend.
//
// Every transform runs on an upstream stream leased for the call and is
// fenced against the caller's per-thread stream with events, so a caller
// sees the same completion contract as the first-party kernels: work
// enqueued after the call on its own stream follows the transform, and the
// host is not blocked. Upstream's field type is a plain 64-bit word with
// canonical values, the same storage the prover's matrices use. The fork
// is built with SPPARK_NO_CXX_RUNTIME: a CUDA failure inside upstream ends
// the process with a message rather than throwing, which matches the
// status checks on the Rust side.
#include <ff/goldilocks.hpp>
#include <ntt/ntt.cuh>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

struct EventPair {
    cudaEvent_t before = nullptr;
    cudaEvent_t after = nullptr;
    ~EventPair() {
        if (before) cudaEventDestroy(before);
        if (after) cudaEventDestroy(after);
    }
    cudaError_t create() {
        cudaError_t status = cudaEventCreateWithFlags(&before, cudaEventDisableTiming);
        if (status == cudaSuccess)
            status = cudaEventCreateWithFlags(&after, cudaEventDisableTiming);
        return status;
    }
};

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
extern "C" int multi_stark_sppark_ntt_batch_device(int device, uint64_t* d_inout, uint32_t lg,
                                                    int order, int direction, int coset,
                                                    uint32_t batch, size_t stride) {
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
    EventPair events;
    status = events.create();
    if (status != cudaSuccess) return static_cast<int>(status);
    status = cudaEventRecord(events.before, cudaStreamPerThread);
    if (status != cudaSuccess) return static_cast<int>(status);
    // Upstream's CUDA failures end the process (the fork's runtime mode), so
    // everything past the launch either completes or never returns.
    stream_t stream(gpu.id());
    stream.wait(events.before);
    NTT::Base_dev_ptr_batch(stream, reinterpret_cast<fr_t*>(d_inout), lg,
                            static_cast<NTT::InputOutputOrder>(order),
                            static_cast<NTT::Direction>(direction),
                            static_cast<NTT::Type>(coset), batch, stride);
    stream.record(events.after);
    status = cudaStreamWaitEvent(cudaStreamPerThread, events.after, 0);
    // Only once the caller's stream waits on the transform may the private
    // stream go; if that wait could not be installed, drain the stream here
    // so nothing still runs against the caller's buffer on return.
    if (status != cudaSuccess) stream.sync();
    return static_cast<int>(status);
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

// --- Resident coset LDE through sppark ---------------------------------
//
// The prover's matrices are row-major with the transform along the column,
// while upstream transforms contiguous vectors. A panel of columns is
// gathered into column-major scratch, transformed, and scattered back in
// the bit-reversed row order the commitment expects:
//
//   A[c][r]  = canonical(trace[r][f + c])                     r < N
//   inverse NR on A[c], normalized upstream                   (bit-reversed coeffs)
//   B[c][i << added_bits] = A[c][i] * shift^rev(i)            i < N, zero elsewhere
//   forward RN on B[c] over M = N << added_bits               (natural-order evals)
//   values[rev(r)][f + c] = B[c][r]                           r < M
//
// That feeds the forward transform in bit-reversed order, so no pass
// restores the coefficient order and the row permutation folds into the
// scatter. The default restores the order instead (B[c][i] = A[c][rev(i)]
// * shift^i), transforms in NR order and scatters naturally;
// `fused_expansion` decides and records why. Neither folds the expansion
// into the transform's first pass; that remains upstream's kernels' job. The gather reduces representatives at or
// above the modulus, which upstream does not accept. The compact panel A
// and the extended panel B cost 8 * (N + M) * C bytes for C columns; C is
// sized from MULTI_STARK_SPPARK_PANEL_BYTES (default 4 GiB) and the width.
// The columns of a panel go through batched launch sequences (the fork's
// grid dimension) in groups sized to the L2 cache;
// MULTI_STARK_SPPARK_BATCH_BYTES sets the group, 0 launches every column
// on its own.

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

// The panel back into row-major rows of columns [first, first + count);
// with `reverse_rows`, panel row r lands in output row rev(r) over
// 2^log_rows rows, so a natural-order transform result is stored in the
// bit-reversed row order the commitment expects.
__global__ void scatter_tiles(const uint64_t* __restrict__ panel, size_t rows, size_t column_stride, size_t width,
                              size_t first, size_t count, unsigned log_rows, bool reverse_rows,
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
            const size_t out = reverse_rows ? reverse_bits(row, log_rows) : row;
            values[out * width + first + col] = tile[threadIdx.x][r];
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
                             size_t first, unsigned count, unsigned log_rows, bool reverse_rows,
                             uint64_t* __restrict__ values) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < rows; row += stride) {
        const size_t out_row = reverse_rows ? reverse_bits(row, log_rows) : row;
        uint64_t* out = values + out_row * width + first;
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
                          size_t count, unsigned log_rows, bool reverse_rows, uint64_t* values) {
    if (width < NARROW_MATRIX)
        scatter_rows<<<blocks_for_total(rows), PANEL_THREADS, 0, cudaStreamPerThread>>>(
            panel, rows, column_stride, width, first, static_cast<unsigned>(count), log_rows, reverse_rows, values);
    else
        scatter_tiles<<<tile_grid(rows, count), dim3(TILE, TILE_ROWS), 0, cudaStreamPerThread>>>(
            panel, rows, column_stride, width, first, count, log_rows, reverse_rows, values);
    return cudaGetLastError();
}

// powers[rev(i)] for i < 2^log: the coset powers in the order the
// bit-reversed coefficients meet them.
__global__ void reverse_powers(const uint64_t* __restrict__ powers, size_t count, unsigned log,
                               uint64_t* __restrict__ reversed) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
        reversed[i] = powers[reverse_bits(i, log)];
}

// The expansion: bit-reversed coefficients A[c][i] (height per column)
// become the bit-reversed input of the forward transform over the extended
// height, B[c][i << added_bits] = A[c][i] * shift^rev(i), zero elsewhere,
// so a forward RN transform yields natural-order evaluations of the coset.
// One column per grid row, rows coalesced along the block.
__global__ void spread_columns(const uint64_t* __restrict__ coefficients, const uint64_t* __restrict__ reversed_powers,
                               size_t height, size_t extended_height, unsigned added_bits,
                               uint64_t* __restrict__ panel) {
    const size_t column = blockIdx.y;
    const size_t mask = (size_t(1) << added_bits) - 1;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; j < extended_height; j += stride) {
        uint64_t value = 0;
        if ((j & mask) == 0) {
            const size_t i = j >> added_bits;
            value = multi_stark_cuda::goldilocks_mul(coefficients[column * height + i], reversed_powers[i]);
        }
        panel[column * extended_height + j] = value;
    }
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

// Read per construction: one getenv against a transform of gigabytes, and
// tests vary it within a process.
size_t panel_budget_bytes() {
    const unsigned long long budget = decimal_setting("MULTI_STARK_SPPARK_PANEL_BYTES", 0);
    return budget ? size_t(budget) : size_t(4) << 30;
}

// -1 unread, 0 first-party, 1 sppark above the height threshold, 2 sppark
// for every height (tests compare the paths on small shapes). Read and
// written from concurrent constructions.
std::atomic<int> backend_flag{-1};

unsigned min_log_height() {
    return unsigned(decimal_setting("MULTI_STARK_SPPARK_MIN_LOG_HEIGHT", 18));
}

// The columns one panel holds at `column_bytes` each: as many as the
// budget admits, at most the width and the batch a launch grid can carry.
// Zero when one column does not fit, which the dispatch rules decline
// before any allocation.
size_t panel_columns(size_t width, size_t column_bytes, size_t budget) {
    size_t columns = budget / column_bytes;
    if (columns > 65535) columns = 65535;
    return columns < width ? columns : width;
}

// Whether a transform of 2^log rows is within upstream's compiled domain.
bool within_domain(size_t height, size_t added_bits) {
    if (height == 0 || (height & (height - 1))) return false;
    const unsigned log = log2_exact(height);
    return added_bits <= MAX_LG_DOMAIN_SIZE && log + added_bits <= MAX_LG_DOMAIN_SIZE;
}

// The bytes one batched launch sequence keeps in flight: the columns of a
// group go through every stage together, so a group that fits the L2 cache
// keeps the reuse between stages that one column at a time had, while
// short columns still share launches. MULTI_STARK_SPPARK_BATCH_BYTES
// overrides the device's L2 size; 0 launches every column on its own.
size_t l2_bytes(int device) {
    int l2 = 0;
    if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device) != cudaSuccess || l2 <= 0)
        l2 = 64 << 20;
    return static_cast<size_t>(l2);
}

size_t batch_group_bytes(int device) {
    const unsigned long long setting = decimal_setting("MULTI_STARK_SPPARK_BATCH_BYTES", ~0ull);
    return setting != ~0ull ? static_cast<size_t>(setting) : l2_bytes(device);
}

// Whether a panel's expansion feeds the forward transform in bit-reversed
// order with the row permutation in the scatter (MULTI_STARK_SPPARK_FUSED=1),
// or restores the coefficient order first and scatters naturally. Measured
// on the RTX PRO 6000, the bit-reversed feed saves the restoring pass but
// its reversing scatter costs as much on wide panels and more on tall
// narrow ones, so restoring is the default.
bool fused_expansion() { return decimal_setting("MULTI_STARK_SPPARK_FUSED", 0) == 1; }

// The reversed coset powers the bit-reversed feed reads: one word per input
// row, nothing on the restoring path.
size_t expansion_extra_bytes(size_t height) {
    return fused_expansion() ? height * sizeof(uint64_t) : 0;
}

// MULTI_STARK_SPPARK_STAGE_TIMING=1 prints the stage times of every coset
// LDE to stderr: events on the caller's stream around each stage, which
// the transforms are fenced to, and one synchronization per LDE.
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
        static const char* const names[STAGES] = {"gather", "inverse", "spread", "forward", "scatter"};
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
                      size_t stride) {
    if (lg == 0) return 0;
    const size_t column_bytes = (size_t(1) << lg) * sizeof(uint64_t);
    size_t group = batch_group_bytes(device) / column_bytes;
    if (group == 0) group = 1;
    int result = 0;
    for (size_t first = 0; result == 0 && first < count; first += group) {
        const size_t batch = count - first < group ? count - first : group;
        result = multi_stark_sppark_ntt_batch_device(device, panel + first * stride, lg, order, direction, 0,
                                                     static_cast<uint32_t>(batch), stride);
    }
    return result;
}

}  // namespace

// Whether the prover's transforms take the sppark path: MULTI_STARK_CUDA_NTT=sppark,
// or a runtime selection, which tests use to compare both paths in one process.
extern "C" int multi_stark_sppark_backend_selected() {
    int flag = backend_flag.load(std::memory_order_acquire);
    if (flag < 0) {
        const char* configured = getenv("MULTI_STARK_CUDA_NTT");
        int expected = -1;
        const int read = configured && strcmp(configured, "sppark") == 0;
        // The first reader publishes; a concurrent selection wins over it.
        flag = backend_flag.compare_exchange_strong(expected, read, std::memory_order_acq_rel) ? read : expected;
    }
    return flag;
}

// 0 first-party, 1 sppark above the height threshold, 2 sppark always.
extern "C" void multi_stark_sppark_select_backend(int selected) {
    backend_flag.store(selected, std::memory_order_release);
}

// Whether a transform of `height` input rows is tall enough for the sppark
// path. Below MULTI_STARK_SPPARK_MIN_LOG_HEIGHT (18) the panel's gather and
// scatter passes cost more than the first-party kernels save, and the
// first-party kernels stay.
extern "C" int multi_stark_sppark_takes(size_t height) {
    const int flag = multi_stark_sppark_backend_selected();
    if (flag == 2) return 1;
    if (flag != 1) return 0;
    return height >= (size_t(1) << min_log_height());
}

// Whether a resident coset LDE of the shape takes the sppark path: tall
// enough, the extended height within upstream's compiled domain, and one
// column's scratch, the compact and the extended panel, within the budget.
extern "C" int multi_stark_sppark_takes_lde(size_t height, size_t width, size_t added_bits) {
    if (width == 0 || !multi_stark_sppark_takes(height) || !within_domain(height, added_bits)) return 0;
    const size_t extended_height = height << added_bits;
    return (height + extended_height) * sizeof(uint64_t) + expansion_extra_bytes(height) <= panel_budget_bytes();
}

// Whether a forward transform of the shape takes the sppark path.
extern "C" int multi_stark_sppark_takes_forward(size_t height, size_t width) {
    if (width == 0 || !multi_stark_sppark_takes(height) || !within_domain(height, 0)) return 0;
    return height * sizeof(uint64_t) <= panel_budget_bytes();
}

// The scratch the sppark path allocates for one LDE, within the panel
// budget: the compact and the extended panel of the columns the budget
// admits, plus the bit-reversed feed's reversed coset powers when that
// expansion is selected. Zero when the shape does not take the path.
extern "C" size_t multi_stark_sppark_panel_bytes(size_t height, size_t width, size_t added_bits) {
    if (!multi_stark_sppark_takes_lde(height, width, added_bits)) return 0;
    const size_t column_bytes = (height + (height << added_bits)) * sizeof(uint64_t);
    const size_t extra = expansion_extra_bytes(height);
    return panel_columns(width, column_bytes, panel_budget_bytes() - extra) * column_bytes + extra;
}

// The scratch the sppark path allocates for one forward transform.
extern "C" size_t multi_stark_sppark_forward_panel_bytes(size_t height, size_t width) {
    if (!multi_stark_sppark_takes_forward(height, width)) return 0;
    const size_t column_bytes = height * sizeof(uint64_t);
    return panel_columns(width, column_bytes, panel_budget_bytes()) * column_bytes;
}

// The coset LDE of `trace` (height x width, natural row order, device memory)
// into `values` (extended_height x width, bit-reversed rows), with the coset
// shift powers `shift_powers[i] = shift^i` for i < height. Scratch is
// allocated per call within the panel budget.
// Transforms the adapter has run so far, whatever the metrics setting, so a
// test can tell a proof that went through sppark from one that did not.
static std::atomic<uint64_t> transforms_run{0};

extern "C" uint64_t multi_stark_sppark_transforms_run() {
    return transforms_run.load(std::memory_order_relaxed);
}

extern "C" int multi_stark_sppark_coset_lde(int device, const uint64_t* trace, uint64_t* values,
                                            size_t height, size_t width, size_t added_bits,
                                            const uint64_t* shift_powers) {
    transforms_run.fetch_add(1, std::memory_order_relaxed);
    if (!trace || !values || !shift_powers || height == 0 || width == 0 || (height & (height - 1)))
        return static_cast<int>(cudaErrorInvalidValue);
    const size_t extended_height = height << added_bits;
    const unsigned log_height = log2_exact(height);
    const unsigned log_extended = log_height + static_cast<unsigned>(added_bits);
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) return static_cast<int>(status);
    const bool fused = fused_expansion();
    const size_t column_bytes = (height + extended_height) * sizeof(uint64_t);
    const size_t extra = expansion_extra_bytes(height);
    const size_t budget = panel_budget_bytes();
    if (budget < column_bytes + extra) return static_cast<int>(cudaErrorInvalidValue);
    const size_t columns = panel_columns(width, column_bytes, budget - extra);
    uint64_t* scratch = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&scratch), columns * column_bytes + extra);
    if (status != cudaSuccess) return static_cast<int>(status);
    uint64_t* a = scratch;
    uint64_t* b = scratch + columns * height;
    uint64_t* powers = b + columns * extended_height;
    int result = 0;
    if (fused) {
        reverse_powers<<<blocks_for_total(height), PANEL_THREADS, 0, cudaStreamPerThread>>>(
            shift_powers, height, log_height, powers);
        result = static_cast<int>(cudaGetLastError());
    }
    StageTimer timer;
    for (size_t first = 0; result == 0 && first < width; first += columns) {
        const size_t count = width - first < columns ? width - first : columns;
        timer.mark(0);
        result = static_cast<int>(gather_panel(trace, height, width, first, count, height, a));
        timer.mark(1);
        if (result == 0) result = transform_columns(device, a, log_height, 1, 1, count, height);
        timer.mark(2);
        const dim3 column_grid(blocks_for_total(extended_height), static_cast<unsigned>(count));
        if (result == 0) {
            if (fused)
                spread_columns<<<column_grid, PANEL_THREADS, 0, cudaStreamPerThread>>>(
                    a, powers, height, extended_height, static_cast<unsigned>(added_bits), b);
            else
                shift_columns<<<column_grid, PANEL_THREADS, 0, cudaStreamPerThread>>>(
                    a, shift_powers, height, log_height, extended_height, b);
            result = static_cast<int>(cudaGetLastError());
        }
        timer.mark(3);
        if (result == 0) result = transform_columns(device, b, log_extended, fused ? 2 : 1, 0, count, extended_height);
        timer.mark(4);
        if (result == 0)
            result = static_cast<int>(
                scatter_panel(b, extended_height, extended_height, width, first, count, log_extended, fused, values));
        timer.mark(5);
        if (result == 0) timer.report(height, count, added_bits);
    }
    // The scratch outlives every kernel that reads it: the free is ordered
    // behind them on the same stream.
    const cudaError_t freed = cudaFreeAsync(scratch, cudaStreamPerThread);
    if (result == 0) result = static_cast<int>(freed);
    return result;
}

// A forward transform in place on `values` (height x width, natural row
// order) leaving bit-reversed rows: gather, upstream forward per column,
// scatter. The quotient's transforms and the general DFT take this path.
extern "C" int multi_stark_sppark_forward(int device, uint64_t* values, size_t height, size_t width) {
    transforms_run.fetch_add(1, std::memory_order_relaxed);
    if (!values || height == 0 || width == 0 || (height & (height - 1)))
        return static_cast<int>(cudaErrorInvalidValue);
    const unsigned log_height = log2_exact(height);
    if (log_height == 0) return 0;
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) return static_cast<int>(status);
    const size_t columns = panel_columns(width, height * sizeof(uint64_t), panel_budget_bytes());
    if (columns == 0) return static_cast<int>(cudaErrorInvalidValue);
    uint64_t* panel = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&panel), columns * height * sizeof(uint64_t));
    if (status != cudaSuccess) return static_cast<int>(status);
    int result = 0;
    for (size_t first = 0; result == 0 && first < width; first += columns) {
        const size_t count = width - first < columns ? width - first : columns;
        result = static_cast<int>(gather_panel(values, height, width, first, count, height, panel));
        if (result == 0) result = transform_columns(device, panel, log_height, 1, 0, count, height);
        if (result == 0)
            result = static_cast<int>(scatter_panel(panel, height, height, width, first, count, log_height, false, values));
    }
    const cudaError_t freed = cudaFreeAsync(panel, cudaStreamPerThread);
    if (result == 0) result = static_cast<int>(freed);
    return result;
}
