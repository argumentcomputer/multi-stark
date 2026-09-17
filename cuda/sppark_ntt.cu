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
#include <cstdlib>
#include <cstring>

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

// One in-place transform of 2^lg field elements at `d_inout` on `device`.
// `order` is NTT::InputOutputOrder (NN, NR, RN, RR), `direction` 0 forward
// or 1 inverse, `coset` 1 for the multiplicative coset by the field
// generator. Inverse transforms are normalized by 1/2^lg upstream.
extern "C" int multi_stark_sppark_ntt_device(int device, uint64_t* d_inout, uint32_t lg,
                                              int order, int direction, int coset) {
    if (!valid_arguments(d_inout, lg, order, direction, coset))
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
    NTT::Base_dev_ptr(stream, reinterpret_cast<fr_t*>(d_inout), lg,
                      static_cast<NTT::InputOutputOrder>(order),
                      static_cast<NTT::Direction>(direction),
                      static_cast<NTT::Type>(coset));
    stream.record(events.after);
    status = cudaStreamWaitEvent(cudaStreamPerThread, events.after, 0);
    // Only once the caller's stream waits on the transform may the private
    // stream go; if that wait could not be installed, drain the stream here
    // so nothing still runs against the caller's buffer on return.
    if (status != cudaSuccess) stream.sync();
    return static_cast<int>(status);
}

// The same transform on host memory: uploads, transforms, downloads and
// synchronizes. For contract checks and small inputs, not the prover.
extern "C" int multi_stark_sppark_ntt_host(int device, uint64_t* inout, uint32_t lg, int order,
                                            int direction, int coset) {
    if (!valid_arguments(inout, lg, order, direction, coset))
        return static_cast<int>(cudaErrorInvalidValue);
    const size_t bytes = (size_t(1) << lg) * sizeof(uint64_t);
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) return static_cast<int>(status);
    uint64_t* d_inout = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&d_inout), bytes);
    if (status != cudaSuccess) return static_cast<int>(status);
    status = cudaMemcpyAsync(d_inout, inout, bytes, cudaMemcpyHostToDevice, cudaStreamPerThread);
    int result = static_cast<int>(status);
    if (result == 0) result = multi_stark_sppark_ntt_device(device, d_inout, lg, order, direction, coset);
    if (result == 0)
        result = static_cast<int>(
            cudaMemcpyAsync(inout, d_inout, bytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    const cudaError_t synced = cudaStreamSynchronize(cudaStreamPerThread);
    if (result == 0) result = static_cast<int>(synced);
    cudaFree(d_inout);
    return result;
}

// --- Resident coset LDE through sppark ---------------------------------
//
// The prover's matrices are row-major with the transform along the column,
// while upstream transforms one contiguous vector. A panel of columns is
// gathered into column-major scratch, transformed column by column, and
// scattered back in the bit-reversed row order the commitment expects:
//
//   A[c][r]  = canonical(trace[r][f + c])              r < N
//   inverse NR on A[c], normalized upstream            (bit-reversed coeffs)
//   B[c][i]  = A[c][rev(i)] * shift^i                  i < N, zero beyond
//   forward NR on B[c] over M = N << added_bits         (bit-reversed evals)
//   values[r][f + c] = B[c][r]                          r < M
//
// The gather reduces representatives at or above the modulus, which upstream
// does not accept. Two panels of C columns cost 16 * M * C bytes; C is sized
// from MULTI_STARK_SPPARK_PANEL_BYTES (default 4 GiB) and the width.

#include "goldilocks.cuh"

namespace {

constexpr unsigned PANEL_THREADS = 256;
constexpr size_t MAX_BLOCKS = 65535;

unsigned blocks_for_total(size_t total) {
    const size_t blocks = (total + PANEL_THREADS - 1) / PANEL_THREADS;
    return static_cast<unsigned>(blocks < MAX_BLOCKS ? blocks : MAX_BLOCKS);
}

__device__ __forceinline__ size_t reverse_bits(size_t index, unsigned log) {
    return log == 0 ? 0 : static_cast<size_t>(__brev(static_cast<unsigned>(index)) >> (32 - log));
}

__global__ void gather_columns(const uint64_t* __restrict__ trace, size_t height, size_t width,
                               size_t first, size_t columns, size_t extended_height,
                               uint64_t* __restrict__ panel) {
    const size_t total = height * columns;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        const size_t row = index / columns;
        const size_t column = index - row * columns;
        panel[column * extended_height + row] = multi_stark_cuda::canonicalize(trace[row * width + first + column]);
    }
}

__global__ void shift_columns(const uint64_t* __restrict__ coefficients, uint64_t* __restrict__ panel,
                              size_t height, unsigned log_height, size_t extended_height,
                              size_t columns, const uint64_t* __restrict__ shift_powers) {
    const size_t total = extended_height * columns;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        const size_t column = index / extended_height;
        const size_t row = index - column * extended_height;
        uint64_t value = 0;
        if (row < height) {
            const uint64_t coefficient = coefficients[column * extended_height + reverse_bits(row, log_height)];
            value = multi_stark_cuda::goldilocks_mul(coefficient, shift_powers[row]);
        }
        panel[index] = value;
    }
}

__global__ void scatter_columns(const uint64_t* __restrict__ panel, size_t extended_height, size_t width,
                                size_t first, size_t columns, uint64_t* __restrict__ values) {
    const size_t total = extended_height * columns;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += stride) {
        const size_t row = index / columns;
        const size_t column = index - row * columns;
        values[row * width + first + column] = panel[column * extended_height + row];
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
    return unsigned(decimal_setting("MULTI_STARK_SPPARK_MIN_LOG_HEIGHT", 20));
}

// The columns one panel holds at `column_bytes` each: as many as the
// budget admits, at most the width. Zero when one column does not fit,
// which the dispatch rules decline before any allocation.
size_t panel_columns(size_t width, size_t column_bytes) {
    const size_t columns = panel_budget_bytes() / column_bytes;
    return columns < width ? columns : width;
}

// Whether a transform of 2^log rows is within upstream's compiled domain.
bool within_domain(size_t height, size_t added_bits) {
    if (height == 0 || (height & (height - 1))) return false;
    const unsigned log = log2_exact(height);
    return added_bits <= MAX_LG_DOMAIN_SIZE && log + added_bits <= MAX_LG_DOMAIN_SIZE;
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
// path. Short transforms are launch-bound on the per-column baseline and
// stay on the first-party kernels below MULTI_STARK_SPPARK_MIN_LOG_HEIGHT
// (20).
extern "C" int multi_stark_sppark_takes(size_t height) {
    const int flag = multi_stark_sppark_backend_selected();
    if (flag == 2) return 1;
    if (flag != 1) return 0;
    return height >= (size_t(1) << min_log_height());
}

// Whether a resident coset LDE of the shape takes the sppark path: tall
// enough, the extended height within upstream's compiled domain, and one
// column's scratch within the panel budget.
extern "C" int multi_stark_sppark_takes_lde(size_t height, size_t width, size_t added_bits) {
    if (width == 0 || !multi_stark_sppark_takes(height) || !within_domain(height, added_bits)) return 0;
    const size_t extended_height = height << added_bits;
    return 2 * extended_height * sizeof(uint64_t) <= panel_budget_bytes();
}

// Whether a forward transform of the shape takes the sppark path.
extern "C" int multi_stark_sppark_takes_forward(size_t height, size_t width) {
    if (width == 0 || !multi_stark_sppark_takes(height) || !within_domain(height, 0)) return 0;
    return height * sizeof(uint64_t) <= panel_budget_bytes();
}

// The scratch the sppark path allocates for one LDE: two panels of the
// columns the budget admits, sized for the extended height. Zero when the
// shape does not take the path.
extern "C" size_t multi_stark_sppark_panel_bytes(size_t height, size_t width, size_t added_bits) {
    if (!multi_stark_sppark_takes_lde(height, width, added_bits)) return 0;
    const size_t column_bytes = 2 * (height << added_bits) * sizeof(uint64_t);
    return panel_columns(width, column_bytes) * column_bytes;
}

// The scratch the sppark path allocates for one forward transform.
extern "C" size_t multi_stark_sppark_forward_panel_bytes(size_t height, size_t width) {
    if (!multi_stark_sppark_takes_forward(height, width)) return 0;
    const size_t column_bytes = height * sizeof(uint64_t);
    return panel_columns(width, column_bytes) * column_bytes;
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
    const size_t column_bytes = 2 * extended_height * sizeof(uint64_t);
    const size_t columns = panel_columns(width, column_bytes);
    if (columns == 0) return static_cast<int>(cudaErrorInvalidValue);
    uint64_t* scratch = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&scratch), columns * column_bytes);
    if (status != cudaSuccess) return static_cast<int>(status);
    uint64_t* a = scratch;
    uint64_t* b = scratch + columns * extended_height;
    int result = 0;
    for (size_t first = 0; result == 0 && first < width; first += columns) {
        const size_t count = width - first < columns ? width - first : columns;
        gather_columns<<<blocks_for_total(height * count), PANEL_THREADS, 0, cudaStreamPerThread>>>(
            trace, height, width, first, count, extended_height, a);
        result = static_cast<int>(cudaGetLastError());
        for (size_t column = 0; result == 0 && column < count && log_height > 0; ++column)
            result = multi_stark_sppark_ntt_device(device, a + column * extended_height, log_height, 1, 1, 0);
        if (result == 0) {
            shift_columns<<<blocks_for_total(extended_height * count), PANEL_THREADS, 0, cudaStreamPerThread>>>(
                a, b, height, log_height, extended_height, count, shift_powers);
            result = static_cast<int>(cudaGetLastError());
        }
        for (size_t column = 0; result == 0 && column < count && log_extended > 0; ++column)
            result = multi_stark_sppark_ntt_device(device, b + column * extended_height, log_extended, 1, 0, 0);
        if (result == 0) {
            scatter_columns<<<blocks_for_total(extended_height * count), PANEL_THREADS, 0, cudaStreamPerThread>>>(
                b, extended_height, width, first, count, values);
            result = static_cast<int>(cudaGetLastError());
        }
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
    const size_t columns = panel_columns(width, height * sizeof(uint64_t));
    if (columns == 0) return static_cast<int>(cudaErrorInvalidValue);
    uint64_t* panel = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&panel), columns * height * sizeof(uint64_t));
    if (status != cudaSuccess) return static_cast<int>(status);
    int result = 0;
    for (size_t first = 0; result == 0 && first < width; first += columns) {
        const size_t count = width - first < columns ? width - first : columns;
        gather_columns<<<blocks_for_total(height * count), PANEL_THREADS, 0, cudaStreamPerThread>>>(
            values, height, width, first, count, height, panel);
        result = static_cast<int>(cudaGetLastError());
        for (size_t column = 0; result == 0 && column < count; ++column)
            result = multi_stark_sppark_ntt_device(device, panel + column * height, log_height, 1, 0, 0);
        if (result == 0) {
            scatter_columns<<<blocks_for_total(height * count), PANEL_THREADS, 0, cudaStreamPerThread>>>(
                panel, height, width, first, count, values);
            result = static_cast<int>(cudaGetLastError());
        }
    }
    const cudaError_t freed = cudaFreeAsync(panel, cudaStreamPerThread);
    if (result == 0) result = static_cast<int>(freed);
    return result;
}
