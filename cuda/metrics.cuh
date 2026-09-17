// SPDX-License-Identifier: MIT OR Apache-2.0
#pragma once
#include <atomic>
#include <time.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

namespace multi_stark_metrics {
constexpr size_t DEVICES = 64;
constexpr size_t NTT_OFFSET = 16;
constexpr size_t WORDS = NTT_OFFSET + 33 * 4;
enum Counter : size_t {
    UploadCalls, UploadRequestedBytes, UploadChunks, UploadFailures, UploadHostNs,
    CosetHits, CosetMisses, CosetUploadedBytes, ConstantBytes,
    DriverFreeBytes, TotalBytes, MemorySamples
};
std::atomic<uint64_t> counters[DEVICES][WORDS]{};
inline bool enabled() {
    static const bool value = std::getenv("AIUR_METRICS") != nullptr;
    return value;
}
inline void add(int device, size_t key, uint64_t value) {
    if (device >= 0 && device < int(DEVICES))
        counters[device][key].fetch_add(value, std::memory_order_relaxed);
}
inline void sample(int device, size_t key, uint64_t value) {
    if (device >= 0 && device < int(DEVICES))
        counters[device][key].store(value, std::memory_order_relaxed);
}
inline uint64_t host_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000 + uint64_t(ts.tv_nsec);
}
struct Upload {
    bool active = enabled();
    int device;
    size_t bytes;
    uint64_t chunks = 0;
    cudaError_t& status;
    uint64_t start = 0;
    Upload(int d, size_t b, cudaError_t& s) : device(d), bytes(b), status(s) {
        if (active) start = host_ns();
    }
    ~Upload() {
        if (!active) return;
        add(device, UploadCalls, 1);
        add(device, UploadRequestedBytes, bytes);
        add(device, UploadChunks, chunks);
        add(device, UploadFailures, status != cudaSuccess);
        add(device, UploadHostNs, host_ns() - start);
    }
};
}
