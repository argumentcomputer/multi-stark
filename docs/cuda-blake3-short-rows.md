# Short-row BLAKE3 dispatch: 2026-09-11

Base: multi-stark `231942a` on `sb/trace-sharding-gpu`.
Isolated branch: `codex/cuda-blake3-short-rows`.
GPU: RTX PRO 6000 Blackwell Server Edition (96 GB), CUDA 13.3, native `sm_120`.
CPU-side benchmarks used `RAYON_NUM_THREADS=8`.

## Change

Messages of 1–1024 bytes use one thread per row, reusing our existing
BLAKE3 chunk compression and raw little-endian digest encoding. Longer
messages keep the existing warp-per-row kernel. The dispatch boundary is
BLAKE3's chunk size; there is no machine-dependent threshold or new runtime
flag. No upstream implementation was copied and no dependency was added.

The new kernel uses 64 registers/thread versus 96 for the old kernel, with
no register spills in either on `sm_120`. It also avoids the old kernel's
8 KiB shared chunk-value array. Both report a 64-byte local stack frame.

## Measurements

The resident benchmark runs both kernels against the same uploaded bytes,
alternates their order, checks all output digests, and times with CUDA events.
Medians of seven samples; at most 1,048,576 rows / 128 MiB input per shape:

| Row bytes | Existing kernel | New dispatch | Speedup |
| ---: | ---: | ---: | ---: |
| 8 | 1.321 ms | 0.384 ms | 3.44× |
| 64 | 1.353 ms | 0.333 ms | 4.07× |
| 320 | 2.351 ms | 0.413 ms | 5.70× |
| 1023 | 2.488 ms | 0.347 ms | 7.16× |
| 1024 | 2.480 ms | 0.472 ms | 5.26× |
| 1025 | 2.632 ms | 2.629 ms | 1.00× |
| 4264 | 0.801 ms | 0.800 ms | 1.00× |
| 7400 | 0.678 ms | 0.677 ms | 1.00× |

Complete PCS commitments include host upload, LDEs, row gathering/hashing,
Merkle construction, and commitment return. Allocation/filling of the input
witness precedes timing. Three baseline/candidate processes per variant,
five warm samples each, with later process order reversed; first iterations
are excluded. Commitments matched byte-for-byte for every shape/run.

| Input shape (height × field columns) | Existing | New | Wall reduction |
| --- | ---: | ---: | ---: |
| 2^20 × 8 (64-byte rows) | 11.403 ms | 9.011 ms | 21.0% |
| 2^18 × 40 (320-byte rows) | 12.871 ms | 10.594 ms | 17.7% |
| 2^18 × 128 (1024-byte rows) | 43.977 ms | 35.930 ms | 18.3% |
| 2^18 × 129 (1032-byte rows) | 44.759 ms | 43.986 ms | 1.7% |
| 2^16 × 925 (7400-byte rows) | 94.049 ms | 91.704 ms | 2.5% |
| Mixed heights/widths | 71.689 ms | 68.297 ms | 4.7% |

The mixed case combines `(2^18,4)`, `(2^18,12)`, `(2^18,24)`, `(2^17,129)`,
and `(2^16,533)`. Equal-height rows concatenate before hashing. The small
wide-only commitment differences are timing variation: their row kernels
are unchanged. These synthetic measurements establish a useful local win;
they do not establish an Init/Mathlib or Stage 2 wall-time improvement.

## Validation and reproduction

- All 70 CUDA-enabled library tests passed (5.46 seconds).
- CPU BLAKE3 comparisons cover 25 row lengths, 1–32768 bytes, and eight row
  counts across warp/block boundaries, using seeded random messages.
- Merkle roots/openings cover both sides of the 1024-byte boundary;
  existing mixed-height, resident, hybrid, and batch-proof tests passed.
- CUDA memcheck reported zero errors on both kernels, with the new kernel
  deliberately launched with two blocks to exercise its grid-stride loop.
- CPU, baseline CUDA, and new CUDA `proof_compatibility` outputs were identical
  (17,213 bytes), and each proof verified. SHA-256:
  `25564a01d1d352b1ec2de56b019b641d24acc81083133e79274a86829b2a5dd5`.
- Clippy completed without warnings for the library and new commit benchmark.

```sh
MULTI_STARK_CUDA_ARCHS=120 cargo test --release --locked --features parallel,cuda --lib -- --test-threads=1
nvcc -O3 --std=c++17 --default-stream=per-thread -arch=sm_120 cuda/blake3_rows_bench.cu -o /tmp/blake3-rows-bench
/tmp/blake3-rows-bench
compute-sanitizer --tool memcheck --error-exitcode 1 /tmp/blake3-rows-bench --check
RAYON_NUM_THREADS=8 cargo run --release --locked --features parallel,cuda --example cuda_commit_bench
```

Select the native architecture appropriate to the test GPU. To compare full
commitments, build the same `cuda_commit_bench.rs` against the base and this
change, preserve both binaries, and alternate runs on an idle GPU. The
benchmark reports cold iteration zero separately and the exact commitment
beside each timing. Existing `cuda_blake3_bench` includes host transfers and
CPU hashing; it is not a resident-kernel timer.
