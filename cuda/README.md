# CUDA backend

This directory contains multi-stark's CUDA prover: Goldilocks arithmetic,
BLAKE3 Merkle commitments, lookup construction, quotient evaluation, and FRI
proving. All GPU DFTs and coset low-degree extensions use the pinned sppark
fork through `sppark_ntt.cu`.

The `cuda` Cargo feature selects `CudaDft` for the production
`GoldilocksBlake3Config`. BabyBear tests remain on their CPU DFT. Without the
feature, `build.rs` exits before looking for nvcc and the normal crate remains
independent of CUDA.

## Current architecture

The backend preserves Plonky3's public PCS interfaces while keeping the hot
prover pipeline device-resident:

1. Rust validates dimensions and caches immutable transform plans, including
   panel scratch, batch groups and coset powers. Admission uses the same plan.
2. Trace matrices are uploaded and transformed through sppark on the caller's
   stream into resident coset LDEs.
3. First-party BLAKE3 kernels commit mixed-height matrices without copying
   LDEs back to the host. Rows of at most one BLAKE3 chunk (1024 bytes) hash
   with one thread per row; longer rows take a warp per row.
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

The `cuda` feature builds `libmulti_stark_cuda.so`. It exposes a C ABI and
privately links the CUDA runtime, GNU C++ runtime, and exception unwinder.
Cargo supplies its location when running tests or examples. Applications
installed outside Cargo must package the library and set an appropriate
runtime search path. A direct dependent's build script receives its path as
`DEP_MULTI_STARK_CUDA_LIBRARY`.

By default, the build detects installed GPUs with `nvidia-smi` and emits
native cubins only for their distinct architectures. Set the comma-separated
architecture list explicitly when cross-compiling or building without a GPU:

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

After correctness passes, run the full proving benchmark or the
transfer-inclusive DFT/LDE and BLAKE3 comparisons:

```sh
cargo bench --locked --features parallel,cuda --bench multi_stark
cargo run --release --locked --features parallel,cuda --example cuda_dft_bench
cargo run --release --locked --features parallel,cuda --example cuda_blake3_bench
```

The DFT benchmark emits CSV to stdout, labels CPU fallbacks, and checks each
output against the CPU reference outside the timing window. Shape/iteration
controls are documented at the top of `examples/cuda_dft_bench.rs`.

The host-backed LDE benchmark compares CPU transforms with auxiliary GPU
offload, including upload and download. Reserve at least one auxiliary GPU
distinct from the primary device (device 0 by default):

```sh
MULTI_STARK_CUDA_AUX_DEVICES=1 \
  cargo bench --locked --features parallel,cuda --bench cuda_host_lde_bench
```

`MULTI_STARK_CUDA_BENCH_SHAPES` accepts semicolon-separated
`log_height,width,added_bits` triples (default `20,533,2`).
`MULTI_STARK_CUDA_BENCH_COPIES` controls concurrent transforms (default `1`).

## Safety and protocol invariants

- Goldilocks values cross the ABI as `u64`; P3 declares the type
  `repr(transparent)` and permits every `u64` bit pattern.
- CUDA outputs are canonical field representatives.
- Rust checks power-of-two sizes, two-adicity, integer overflow, and buffer
  lengths before each synchronous FFI call.
- First-party CUDA sources use the repository's MIT/Apache-2.0 terms. The
  sppark dependency retains its own license; see the pinned fork's `LICENSE`.
- CUDA affects prover execution only. Fields, BLAKE3 hashing, transcripts,
  proof format, and the CPU verifier are unchanged. Canonical representation
  changed newly generated proof bytes relative to pre-CUDA revisions as
  documented in the top-level changelog.
