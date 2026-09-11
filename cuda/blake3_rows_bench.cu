// SPDX-License-Identifier: MIT OR Apache-2.0
// Standalone resident-kernel comparison; see docs/cuda-blake3-short-rows.md.
// Both kernels run against the same resident bytes, with alternating order.
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "kernels.cu"

void checked(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}
void launch(bool old, uint8_t* output, const uint8_t* input,
            size_t bytes, size_t rows, bool force_stride) {
    if (old) {
        size_t blocks = std::min<size_t>((rows + THREADS / 32 - 1) / (THREADS / 32), MAX_BLOCKS);
        blake3_hash_rows_kernel<<<blocks, THREADS, 0, cudaStreamPerThread>>>(output, input, bytes, rows);
        checked(cudaGetLastError());
    } else if (force_stride && bytes <= BLAKE3_CHUNK_BYTES) {
        // Exercise multiple grid-stride iterations without a 16M-row fixture.
        blake3_hash_short_rows_kernel<<<2, THREADS, 0, cudaStreamPerThread>>>(output, input, bytes, rows);
        checked(cudaGetLastError());
    } else {
        checked(launch_blake3_rows(output, input, bytes, rows));
    }
}
int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--check") != 0)) {
        fprintf(stderr, "usage: %s [--check]\n", argv[0]);
        return 1;
    }
    const bool small = argc == 2;
    checked(cudaSetDevice(0));
    printf("row_bytes,rows,iteration,warp_ms,dispatch_ms,speedup\n");
    for (size_t bytes : {size_t(1), size_t(8), size_t(16), size_t(32), size_t(64),
                         size_t(128), size_t(256), size_t(320), size_t(512),
                         size_t(1023), size_t(1024), size_t(1025), size_t(4264), size_t(7400)}) {
        size_t rows = small ? 777 : std::min<size_t>(1 << 20, (128 << 20) / bytes);
        std::vector<uint8_t> input(bytes * rows);
        uint64_t state = 0x81726ab567846217ULL;
        for (auto& byte : input) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            byte = state;
        }
        DeviceBuffer source, first, second;
        checked(source.allocate((input.size() + 7) / 8));
        checked(first.allocate(rows * 4));
        checked(second.allocate(rows * 4));
        auto* d_source = reinterpret_cast<uint8_t*>(source.get());
        auto* d_first = reinterpret_cast<uint8_t*>(first.get());
        auto* d_second = reinterpret_cast<uint8_t*>(second.get());
        checked(cudaMemcpy(d_source, input.data(), input.size(), cudaMemcpyHostToDevice));
        launch(true, d_first, d_source, bytes, rows, false);
        launch(false, d_second, d_source, bytes, rows, small);
        std::vector<uint8_t> expected(rows * 32), actual(rows * 32);
        checked(cudaMemcpy(expected.data(), d_first, expected.size(), cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(actual.data(), d_second, actual.size(), cudaMemcpyDeviceToHost));
        if (expected != actual) { fprintf(stderr, "mismatch for %zu-byte rows\n", bytes); return 2; }
        cudaEvent_t start, stop;
        checked(cudaEventCreate(&start)); checked(cudaEventCreate(&stop));
        for (int iteration = 0; iteration < (small ? 1 : 7); ++iteration) {
            float ms[2];
            for (int order = 0; order < 2; ++order) {
                int which = (order + iteration) % 2;
                checked(cudaEventRecord(start, cudaStreamPerThread));
                launch(which == 0, which == 0 ? d_first : d_second, d_source, bytes, rows, small);
                checked(cudaEventRecord(stop, cudaStreamPerThread));
                checked(cudaEventSynchronize(stop));
                checked(cudaEventElapsedTime(&ms[which], start, stop));
            }
            printf("%zu,%zu,%d,%.6f,%.6f,%.3f\n", bytes, rows, iteration, ms[0], ms[1], ms[0]/ms[1]);
        }
        checked(cudaEventDestroy(start)); checked(cudaEventDestroy(stop));
    }
}
