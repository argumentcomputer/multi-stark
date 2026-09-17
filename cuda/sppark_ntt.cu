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

#include <cstdint>

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
