# CUDA benchmark record

This file records dated downstream and microbenchmark results without making
hardware-specific numbers part of multi-stark's stable API documentation.

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
