# CUDA benchmark record

This file records dated downstream and microbenchmark results without making
hardware-specific numbers part of multi-stark's stable API documentation.

## 2026-10-09: Init FRI compression and recursive KZG

The four-Blackwell pipeline measured **14m29s including FRI compression** with
populated fixed-preprocessing and development-SRS caches, or 14m57s including
independent CPU verification. Fresh fixed preprocessing with cached SRS took
20m01s. The regenerated root has a 19-circuit profile; it is not the historical
22-circuit workload. The SRS remains development-only.

See the [KZG CUDA handoff](kzg-cuda-handoff.md) for preserved shutdown/recovery
artifacts, validation scope, implementation status and resumption commands,
and the [measurement record](../experiments/kzg-cuda-gpu-feeding-results.json)
for exact phase times and hashes.

## 2026-08-26: Ix recursive proving

- GPU: NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB
- Driver: 595.84
- Toolkit/compiler: CUDA 13.3 (`nvcc`)
- Workload: Ix `Vector.extract_append`, recursive proving, 50 FRI queries
- CPU combined inner + outer STARK proving: 71.87 s
- CUDA combined inner + outer STARK proving: 8.84 s
- Speedup: 8.13x
- Inner proof: 11,783,606 bytes
- Outer proof: 4,162,775 bytes
- CPU verification: passed

The 50-query setting was selected for development iteration and is not a
recommended production security parameter. These figures include a downstream
workload and should be re-measured after changes to Ix, multi-stark, the CUDA
toolkit, or the GPU architecture.

## 2026-10-08 to 2026-10-09: Init KZG compression

- GPUs: 4 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each
- Driver 595.91.07, CUDA 13.3 (`nvcc`), `MULTI_STARK_CUDA_ARCHS=120`
- Host: 96 vCPU Xeon Platinum 8559C, 999 GB RAM, NVMe for checkpoints
- Feature: `kzg-cuda`; development SRS with known trapdoor throughout

| Input and cache state | Boundary | Wall time |
| --- | --- | ---: |
| Historical 1,530,149-byte compressed FRI proof; fresh SRS and fixed preprocessing | Both KZG stages, FRI compression excluded | 1,740.2 s (29m00s) |
| Regenerated 19-circuit root; cached SRS, fresh fixed preprocessing | FRI compression plus both KZG stages | 1,201.0 s (20m01s) |
| Regenerated 19-circuit root; cached SRS and fixed preprocessing, fresh witnesses | FRI compression plus both KZG stages | 869.5 s (14m29s) |

All runs reproduced the CPU proofs, packets and profile IDs byte for byte and
passed independent CPU verification. Sampled GPU utilization over the whole
29m00s run was 3.0 to 3.7 percent per device with 90 to 92 percent zero
samples; the cached KZG proving phases sampled 6 to 9 percent. Two-second
sampling misses short kernels, so these bound idle time from above only.

Records: `experiments/kzg-cuda-results.json`,
`experiments/kzg-cuda-sub30-results.json`,
`experiments/kzg-cuda-gpu-feeding-results.json`, and the reports under
`experiments/kzg-cuda-validation/`. Analysis:
`experiments/kzg-pipeline-review.md`. Upstream comparison:
[upstream GPU orchestration](upstream-gpu-orchestration.md).

## Reproducing in-repository measurements

The Criterion benchmark exercises the full proving pipeline:

```sh
cargo bench --release --locked --features parallel,cuda --bench multi_stark
```

Transfer-inclusive DFT/LDE and BLAKE3 comparisons are available separately:

```sh
cargo run --release --locked --features parallel,cuda --example cuda_dft_bench
cargo run --release --locked --features parallel,cuda --example cuda_blake3_bench
```

The CSV DFT benchmark labels shapes below the production CUDA thresholds as
`cpu-fallback`, warms both implementations, uses the same iteration count, and
checks each output against the CPU reference outside the timing window.
