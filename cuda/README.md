# CUDA backend

This directory contains multi-stark's first-party CUDA implementation of
Goldilocks arithmetic, DFTs, coset low-degree extensions, BLAKE3 Merkle
commitments, lookup construction, quotient evaluation, and FRI proving. It
does not use ICICLE or copy code from it.

The `cuda` Cargo feature selects `CudaDft` for the production
`GoldilocksBlake3Config`. BabyBear tests remain on their CPU DFT. Without the
feature, `build.rs` exits before looking for nvcc and the normal crate remains
independent of CUDA.

## Current architecture

The backend preserves Plonky3's public PCS interfaces while keeping the hot
prover pipeline device-resident:

1. Rust validates dimensions and builds exact P3-compatible twiddle tables.
2. Trace matrices are uploaded and transformed with fused radix-4/radix-8
   DIF kernels into resident coset LDEs.
3. First-party BLAKE3 kernels commit mixed-height matrices without copying
   LDEs back to the host.
4. Lookup traces and quotient LDEs are constructed from resident commitments.
5. Batched openings, reductions, FRI folding, and Merkle authentication remain
   resident; only protocol-visible openings and proofs return to Rust.

Large traces can be retained in pinned host memory and uploaded in bounded
windows when keeping every source matrix resident would exceed device memory.
The spill policy reserves one quarter of reported device memory by default and
uses `cudaMemGetInfo` at runtime instead of assuming a particular GPU size.
This retains the exact CPU storage layout and proof bytes.

The ordinary `TwoAdicSubgroupDft` implementation remains available for direct
host-matrix users and transfer-inclusive microbenchmarks.

## Build configuration

By default, the build emits native cubins for the common architectures among
`sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, and `sm_120` that the installed
`nvcc` reports as supported. Override the comma-separated list for smaller
deployment artifacts:

```sh
MULTI_STARK_CUDA_ARCHS=80,90 \
  cargo test --release --locked --features parallel,cuda
```

Useful environment variables:

- `NVCC`: path to nvcc (default: `$CUDA_HOME/bin/nvcc`, then `nvcc` on `PATH`)
- `CUDA_HOME` or `CUDA_PATH`: toolkit root used to find nvcc and `libcudart`
- `MULTI_STARK_CUDA_ARCHS`: numeric compute capabilities, e.g. `80,90`
- `MULTI_STARK_CUDA_DEVICE`: device used by the prover and microbenchmarks
  (default: `0`; `CUDA_VISIBLE_DEVICES` remapping is also honored)
- `MULTI_STARK_CUDA_MIN_FREE_BYTES`: device headroom retained by spilling
  source traces (default: one quarter of total device memory)

Only native cubins are emitted. This avoids a newer toolkit's PTX being chosen
by a slightly older driver instead of a compatible native cubin.

The current native backend supports Linux x86_64, one selected device per
prover, Merkle caps of height 0, and binary FRI folding
(`max_log_arity <= 1`). Select a device with `MULTI_STARK_CUDA_DEVICE` or remap
devices with `CUDA_VISIBLE_DEVICES`. Unsupported cap/arity configurations fail
during configuration construction instead of producing malformed commitments.

## First GPU run

From the repository root:

```sh
./cuda/smoke.sh
```

The script prints the git/toolchain/GPU environment, then runs tests in the
failure-localizing order: field arithmetic, DFT, coset LDE, and finally the
entire release suite. It then generates a deterministic wide proof on CPU and
CUDA, crossing the resident-FRI, GPU-MMCS, and parallel-LDE thresholds, and
compares the complete serialized files. Together with the pinned BLAKE3/Merkle
vectors and small-proof digest, this demonstrates production-path protocol
compatibility rather than only transform round trips.

After correctness passes, run the transfer-inclusive microbenchmark:

```sh
cargo run --release --locked --features parallel,cuda --example cuda_dft_bench
```

It emits CSV to stdout. Shape/iteration controls are documented at the top of
`examples/cuda_dft_bench.rs`. The full proving Criterion benchmark and dated
Ix measurements are documented in [`docs/cuda-benchmarks.md`](../docs/cuda-benchmarks.md).

## Safety and protocol invariants

- Goldilocks values cross the ABI as `u64`; P3 declares the type
  `repr(transparent)` and permits every `u64` bit pattern.
- CUDA outputs are canonical field representatives.
- Rust checks power-of-two sizes, two-adicity, integer overflow, and buffer
  lengths before each synchronous FFI call.
- The CUDA source is licensed under the repository's MIT/Apache-2.0 terms and
  depends only on the CUDA runtime/toolkit when enabled.
- CUDA affects prover execution only. Fields, BLAKE3 hashing, transcripts,
  proof format, and the CPU verifier are unchanged. Canonical representation
  changed newly generated proof bytes relative to pre-CUDA revisions as
  documented in the top-level changelog.
