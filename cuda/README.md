# multi-stark CUDA backend

GPU acceleration of the multi-stark prover. Goldilocks-only by design.
This document is the single source of truth for the GPU work — read top
to bottom before touching anything.

## Status

| Phase | Scope | Status |
|---|---|---|
| **1** | NTT/LDE via sppark, `dft_batch` only, single H↔D copy per call | ✅ Done locally, untested on GPU |
| **2** | `coset_lde_batch` override with explicit-shift kernel | ✅ Done locally, untested on GPU |
| **3a** | Goldilocks-Poseidon2 permutation kernels (widths 8 + 16) | ✅ Done locally, untested on GPU |
| **3b** | Merkle tree kernels (leaf-hash sponge, pair-compress, mix-compress) + Rust orchestration | ✅ Done locally, untested on GPU |
| **3c** | `CudaPoseidon2MerkleMmcs` impl of `p3_commit::Mmcs<Goldilocks>` | ⏳ Trait wiring only (algorithmic core done) |
| **4** | Full GPU `Pcs` with device-resident trace across all phases | Not started |

**The agent's first job on Blackwell is to run the test suite.** If it
passes, Phases 1–3b are kernel-validated. Phase 3c is the next piece of
real work: wrapping the `cuda::mmcs` helpers in a `Mmcs<Goldilocks>` impl
so it slots into `TwoAdicFriPcs`. Detailed wiring guide in §"Phase 3c
wiring guide" below.

## Quick orientation

multi-stark is a Plonky3-based STARK prover. The `Pcs` is
`TwoAdicFriPcs<Val, Dft, Mmcs, ExtMmcs>`. Per-commit cost is dominated
by:

1. Coset LDE (FFT-bound) — `Dft`
2. Merkle leaf hashing + tree build (hash-bound) — `Mmcs`
3. FRI fold + queries (mixed) — internal to `TwoAdicFriPcs`

The plan is to swap the CPU `Dft` (`Radix2DitParallel`) and the CPU
`Mmcs` (`MerkleTreeMmcs<…Poseidon2…>`) for CUDA equivalents under a
`cuda` Cargo feature, leaving the verifier and AIR machinery on CPU.

## Source-of-truth per concern (the donor map)

Every kernel piece comes from the donor that owns the *correct* version
of that concern. Mixing donors per-concern keeps both protocol fidelity
(Plonky3 byte-for-byte) and battle-tested kernel code (sppark/pil2):

| Concern | Donor | Why |
|---|---|---|
| Goldilocks Mont arithmetic | sppark `ff/gl64_t.cuh` | PTX-asm Mont, ~10y production. Even pil2 vendors this verbatim. |
| Batched NTT kernels | sppark `ntt/{ntt.cuh, kernels/*.cu, parameters/goldilocks.h}` | Hand-tuned narrow/wide butterflies + bit reversal. |
| Coset shift | written fresh (see `coset_shift_columns_kernel`) | Plonky3 passes `Val::GENERATOR = 7`; sppark's `partial_group_gen_powers` uses the 2-adic generator instead. Fresh kernel takes an explicit `shift: u64`. |
| Poseidon2 round constants + matrix-diag | Plonky3 `p3-goldilocks/src/poseidon2.rs` | Protocol-truth — CPU and GPU must agree byte-for-byte. |
| Poseidon2 kernel structure (mds4x4, pow7) | pil2 `poseidon2_goldilocks.cuh` patterns | Only Goldilocks-specific GPU Poseidon2 reference. We don't pull in pil2's `goldilocks_trace_layout` machinery. |
| External / internal layer algorithm | Plonky3 `p3-poseidon2/src/{external,internal}.rs` | Spec-defining: `mds_light`, partial-round dispatch, round ordering. |
| 4-kernel Merkle tree decomp | SP1 `sp1-gpu/crates/sys/lib/merkle_tree/` | Cleaner separation than pil2's monolithic `merkletree(...)`. Substitute Goldilocks-Poseidon2 hash. |
| Mmcs trait + multi-matrix tree topology | Plonky3 `p3-merkle-tree/src/mmcs.rs` | Plonky3 invented the multi-height-matrix interleaving. |
| FRI prover structure (Phase 4) | Plonky3 `p3-fri/src/prover.rs` | Match protocol exactly so existing CPU verifier accepts GPU-produced proofs. |
| CUDA stream / device management | sppark `util/{exception.cuh, gpu_t.cuh, rusterror.h}` | Already used for Phase 1; battle-tested. |

| Reference (not a source) | Use |
|---|---|
| icicle `icicle-goldilocks/src/{poseidon2, fri, merkle}` | Cross-check oracle: when our hand-rolled kernel disagrees with icicle on a fixed input, one of us is wrong. Don't depend on it. |

## Building on the Blackwell box

Prerequisites: CUDA 12.9+, `nvcc` on `PATH`, Blackwell sm_120 GPU.

```bash
cargo build --release --features parallel,cuda
```

First build:
1. Downloads `sppark = "0.1.14"` (~4 MB).
2. Compiles sppark's own CUDA library (`libsppark_cuda.a`) — sppark's
   `build.rs` auto-detects `nvcc` and builds `select_gpu` and friends.
3. Compiles `cuda/gpu_api.cu` and `cuda/poseidon2.cu` against sppark's
   headers (via `DEP_SPPARK_ROOT`), producing `libmulti_stark_cuda.a`.
4. Links both into the multi-stark rlib.

If `nvcc` is missing, `cc::Build::cuda(true)` returns a hard error
(`failed to find tool "nvcc"`). This is intentional — building with
`cuda` feature without nvcc should never silently produce a stub binary.

## Validation: kernel-correctness test order

Run these in order. **If any test in this list fails, fix it before
moving to the next** — later tests build on earlier ones, so an early
bug masquerades as a downstream bug.

```bash
cargo test --release --features parallel,cuda
```

The full suite runs everything below in ~one invocation. The order
matters when something fails — debug from the top:

| # | Test name(s) | What it validates | If it fails, the bug is in… |
|---|---|---|---|
| 1 | `dft_matches_cpu_*` (4 tests) | sppark NTT FFI + bit-reversal ordering | `cuda/gpu_api.cu::compute_ntt_batch` or sppark integration |
| 2 | `coset_lde_matches_cpu_*` (5 tests) | `coset_shift_columns_kernel` + iDFT/DFT pipeline | `cuda/gpu_api.cu::compute_coset_lde_with_shift` (likely the shift exponentiation or the iDFT layout) |
| 3 | `poseidon2_permute_8_matches_cpu` | Width-8 permutation: round constants, MDS4, internal-layer matrix-diag | `cuda/poseidon2.cu`: round-constant byte order, `mds_mat4`, or `MATRIX_DIAG_8` |
| 4 | `poseidon2_permute_16_matches_cpu` | Width-16 permutation; same checks as #3 plus group-mixing in `mds_light<16>` | `mds_light<16>` group-sum logic, or width-16 constants |
| 5 | `leaf_hash_partial_block_widths` | Sponge absorb logic across rate-12 boundaries | `sponge_hash_row` partial-block branch |
| 6 | `leaf_hash_matches_cpu_*` (3 tests) | Full sponge over multi-row matrices | `compute_poseidon2_leaf_hash_rows` / kernel block configuration |
| 7 | `compress_pairs_matches_cpu` | Width-8 `TruncatedPermutation` 2-to-1 compression | `compress_pair_to` or its caller |
| 8 | `compress_paired_matches_cpu` | Same, two-input variant | `compress_paired_kernel` indexing |
| 9 | `build_merkle_tree_single_matrix_matches_cpu` | Recursive pair-compress topology | `build_merkle_tree` orchestration |
| 10 | `build_merkle_tree_two_matrices_matches_cpu` | Multi-matrix injection (mix-compress at right level) | Multi-matrix loop in `build_merkle_tree` |
| 11 | `build_merkle_tree_three_matrices_matches_cpu` | Two consecutive mix steps | Same |
| 12 | `build_merkle_tree_height_one`, `build_merkle_tree_unbalanced_columns` | Edge cases | Same |
| 13 | `end_to_end_lde_then_leaf_hash_matches_cpu` | Full pipeline: GPU LDE → GPU leaf-hash matches CPU pipeline | Anything between #1 and #6; serves as a regression for the integration |
| 14 | The 14 prover tests inherited from CPU | Real prover/verifier roundtrip with GPU `Dft` | `coset_lde_batch` correctness (most likely) or FRI Pcs interaction |

## Benchmarking

Two relevant harnesses:

```bash
# In-tree microbench: U32Add + ByteTable circuits at heights 2^12-2^14.
cargo bench --features parallel,cuda --bench multi_stark
```

For Aiur-driven benches (the ones that motivated the GPU work), Ix
needs to consume multi-stark with `parallel,cuda`:

```toml
# In ix/Cargo.toml:
multi-stark = { path = "../multi-stark.sb-gpu", features = ["parallel", "cuda"] }
```

Then from `~/repos/ix/` (under `direnv`):

```bash
lake exe bench-blake3 --num-hashes 1000 --input-size 1024
```

Compare per-phase wall-times to the CPU baseline (in project memory).
The expected wins:

- `stage1_commit`: Phase 1 helps (FFT moves to GPU). Phase 3c (when wired)
  closes the gap to the Keccak baseline by GPU-accelerating the leaf
  hashing.
- `stage2_commit`: Same.
- `quotient`: Phase 1 helps (the quotient commit FFT). Phase 3c helps
  (the quotient leaf hashing).
- `fri_open`: Phase 1 helps marginally (some FRI work uses Dft). Phase 4
  helps fully (FRI on device).

## Phase 3c wiring guide

The kernel-side primitives are done and tested at the equivalence level
(see test #9-13). What remains is wrapping them in a `Mmcs<Goldilocks>`
impl that `TwoAdicFriPcs` accepts. The structural shape:

```rust
// src/cuda/mmcs.rs — extend the existing module

use p3_commit::{BatchOpening, Mmcs};
use p3_field::Field;
use p3_matrix::{Dimensions, Matrix};

#[derive(Clone, Default)]
pub struct CudaPoseidon2MerkleMmcs {
    pub cap_height: usize,  // matches Plonky3's MerkleTreeMmcs::cap_height
}

pub struct CudaProverData<M> {
    pub leaves: Vec<M>,                    // original matrices, kept for openings
    pub digest_layers: Vec<Vec<Digest>>,   // bottom-to-top tree
    pub matrix_index_for_layer: Vec<Option<usize>>, // which matrix (if any) was injected at each layer
}

impl Mmcs<Goldilocks> for CudaPoseidon2MerkleMmcs {
    type ProverData<M> = CudaProverData<M>;
    type Commitment = Vec<Digest>;  // the cap = top `cap_height` layers worth of digests
    type Proof = Vec<Digest>;       // sibling chain for one query
    type Error = ();                // or a richer error type

    fn commit<M: Matrix<Goldilocks>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
        // 1. Sort inputs by height descending (caller may already have done this; we redo to be safe)
        // 2. Call build_merkle_tree(&inputs) — already implemented
        // 3. Take top cap_height layers as commitment
        // 4. Build matrix_index_for_layer mapping
        // 5. Return
    }

    fn open_batch<M: Matrix<Goldilocks>>(
        &self, index: usize, prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<Goldilocks, Self> {
        // Pure CPU: traverse the tree, extract sibling at each level,
        // also extract the matrix rows at the queried indices.
        //
        // Each layer halves except where a matrix is injected — at those
        // layers the proof sibling is the matrix's leaf hash, not the
        // pair-sibling.
    }

    fn verify_batch(
        &self, commitment: &Self::Commitment, dimensions: &[Dimensions],
        index: usize, opening: &BatchOpening<Goldilocks, Self>,
    ) -> Result<(), Self::Error> {
        // Pure CPU: replay leaf hashes from the matrix rows in the opening,
        // then walk up the path applying compress at each step. At injection
        // layers, mix in the corresponding matrix's leaf hash.
    }
}
```

Then in `src/types.rs`:

```rust
#[cfg(not(feature = "cuda"))]
pub type Mmcs = MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, Hash, Compress, 2, 4>;

#[cfg(feature = "cuda")]
pub type Mmcs = crate::cuda::mmcs::CudaPoseidon2MerkleMmcs;
```

And `ExtMmcs` follows analogously.

The `BatchOpening` type is `p3_commit::BatchOpening<T, M: Mmcs<T>>`,
which is `{ opened_values: Vec<Vec<T>>, opening_proof: M::Proof }`. So
the proof type is just the sibling chain.

**Validation tests for Phase 3c:**

Once wired, add tests like:

```rust
#[test]
fn cuda_mmcs_root_matches_cpu_mmcs() {
    // Build a CPU MerkleTreeMmcs with the same Hash + Compress, commit
    // the same matrices, compare commitments.
}

#[test]
fn cuda_mmcs_open_verify_roundtrip() {
    // commit → open at some index → verify. Should pass.
}

#[test]
fn cuda_mmcs_tampered_proof_rejected() {
    // Tamper with a sibling, verify must reject.
}
```

If those pass, the prover tests in #14 above (which run the full
prove/verify roundtrip with the swapped `Mmcs`) should pass automatically.

## Phase 4 — full GPU `Pcs` with resident trace

Goal: eliminate the per-phase H↔D copy. The trace lives on the device
from witness upload through stage-1 commit, stage-2 commit, quotient
commit, FRI commit, and FRI queries. Only the final root + opening
proof crosses back.

### Why it's a separate phase

Requires custom matrix types that hold device pointers, and a custom
`Pcs` impl that bypasses Plonky3's `TwoAdicFriPcs` to keep data
device-resident. The Plonky3 `Mmcs` and `Dft` traits are already
host-centric — they pass `RowMajorMatrix<F>` by value, which forces
materialization on the host. A device-resident Pcs has to override at
the `Pcs` layer, not at `Mmcs`/`Dft`.

### Estimated scope

- New module `src/cuda/pcs.rs` implementing `Pcs<Challenge, Challenger>`.
- Device-resident matrix wrapper type (`DeviceMatrix<F>` holding a
  `*mut fr_t` + `(rows, cols)`).
- FRI prover that operates on device buffers (keeps fold + Merkle on
  GPU). Probably means re-implementing the FRI commit phase rather
  than reusing `p3_fri::prover::prove_fri`.
- ~2-3 weeks of careful work.

### Prerequisite

Phase 3c must work first.

## Build / Cargo / dependency notes

### Cargo features

```toml
[features]
parallel = ["p3-maybe-rayon/parallel"]
cuda = ["dep:sppark"]
```

Both are independent. `parallel` enables CPU rayon. `cuda` enables the
GPU backend (and pulls sppark).

### Dependencies

- `sppark = "0.1.14"` — published crate. Auto-detects `nvcc`/`hipcc`
  and builds its own kernels. We use only the CUDA path. Provides
  `DEP_SPPARK_ROOT` env var to our `build.rs` so we can find sppark's
  C++ headers.
- `cc = "1"` — build-dep, always pulled even on CPU builds (Cargo
  doesn't allow optional build-deps). Our `build.rs` is a no-op
  without the `cuda` feature, so `nvcc` is not invoked unless feature
  is on.

### nvcc flags (set in `build.rs`)

- `-arch=sm_120` — Blackwell native code (primary target).
- `-gencode arch=compute_80,code=sm_80` — Ampere PTX fallback.
- `-Xcompiler -Wno-unused-{function,parameter}` — quiets sppark
  header warnings.
- `-DFEATURE_GOLDILOCKS` — sppark conditional include for Goldilocks.
- `-DGL64_NO_REDUCTION_KLUDGE` — skip slow-path reduction; use Solinas.
- `-DTAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE` — sppark's `RustError`
  carries the message string.

### Field constraints

Hardcoded `FEATURE_GOLDILOCKS`. There is no plan to support
BabyBear/KoalaBear. The `TwoAdicSubgroupDft` impl is also
Goldilocks-only by type signature.

## File map

```
multi-stark.sb-gpu/
├── Cargo.toml                  # cuda feature + sppark optional dep + cc build-dep
├── build.rs                    # nvcc compile gated on CARGO_FEATURE_CUDA
├── cuda/
│   ├── README.md               # this file
│   ├── gpu_api.cu              # NTT/LDE FFI: compute_ntt_batch, compute_coset_lde_with_shift
│   └── poseidon2.cu            # Poseidon2 + Merkle FFI: permute_8/16, leaf_hash_rows, compress_pairs/paired
└── src/
    ├── cuda/                   # mod cuda (gated on feature cuda)
    │   ├── mod.rs              # re-exports
    │   ├── ffi.rs              # extern "C" declarations for all FFI entry points
    │   ├── dft.rs              # CudaRadix2Dft impl of TwoAdicSubgroupDft<Goldilocks>
    │   ├── poseidon2.rs        # safe Rust wrappers: permute_8_gpu, permute_16_gpu
    │   ├── mmcs.rs             # leaf_hash_matrix, compress_pairs/paired, build_merkle_tree
    │   └── tests.rs            # full equivalence test suite (compiles only with cuda feature)
    ├── types.rs                # conditional Dft swap (Phase 1+2 wired); Mmcs swap pending Phase 3c
    └── ... rest unchanged
```

## Donor codebase pointers

For an agent reading this on the GPU box:

| What | Where (relative to typical clone layout) |
|---|---|
| sppark NTT public API | `sppark/ntt/ntt.cuh` |
| sppark Goldilocks NTT params (`GOLDILOCKS_PLONKY2` shift) | `sppark/ntt/parameters/goldilocks.h` |
| sppark `gl64_t` PTX-asm Mont arithmetic | `sppark/ff/gl64_t.cuh` |
| pil2 Poseidon2-Goldilocks GPU (Phase 3 inspiration) | `pil2-proofman/pil2-stark/src/goldilocks/src/poseidon2_goldilocks.{cuh,cu}` |
| SP1 GPU architecture (KoalaBear, structural reference) | `sp1-6.1.0/sp1/sp1-gpu/crates/` |
| plonky3-accelerate FFI template (Phase 1 lifted FFI shape from here) | `plonky3-accelerate/cuda/` |

Repos:
- sppark: <https://github.com/supranational/sppark>
- pil2-proofman: <https://github.com/0xPolygonHermez/pil2-proofman>
- SP1: <https://github.com/succinctlabs/sp1>
- Plonky3: <https://github.com/Plonky3/Plonky3>
- icicle (oracle for cross-check, not a dep): <https://github.com/ingonyama-zk/icicle>
- plonky3-accelerate: <https://github.com/argumentcomputer/plonky3-accelerate>
  *(stale relative to SP1; useful only for the FFI shape of Phase 1, not for kernel quality)*

## Open issues / things to watch

1. **Plonky3-Goldilocks vs sppark canonical form.** sppark's `gl64_t`
   default (no `GL64_PARTIALLY_REDUCED`) stores values in canonical
   `[0, p)` form, matching Plonky3's `Goldilocks::value`. We've
   confirmed this in headers but only kernel testing on hardware will
   confirm wire-format compatibility. If equivalence tests fail with
   "values are reduced versions of each other" patterns, this is the
   suspect.

2. **Block-size tuning.** `leaf_hash_rows_kernel`, `compress_*_kernel`
   all use `bsize = 256`. This is a guess; profile on Blackwell and
   tune. The width-16 sponge holds 16 u64s in registers per thread,
   so register pressure may push us to 128 or 64.

3. **Multi-matrix tree topology edge cases.** `build_merkle_tree`
   assumes heights are powers of 2 and strictly decreasing. Plonky3's
   `MerkleTreeMmcs` allows equal heights (matrices get hashed
   together at their level); our impl currently asserts. If the
   prover passes equal-height matrices, this needs to be relaxed —
   the fix is in the orchestration loop, not in the kernels.

4. **The `coset_lde_batch` shift.** Plonky3 typically passes
   `Val::GENERATOR = 7`, but the trait signature accepts any `F`. The
   kernel handles arbitrary `shift: u64` correctly via binary
   exponentiation, but the equivalence tests only cover the
   `Val::GENERATOR` case explicitly. If the prover passes a different
   shift somewhere, add a test for that case.

5. **Cap height = 0 currently.** multi-stark's bench config uses
   `cap_height = 0` (root-only commitment). Phase 3c wiring should
   support `cap_height > 0` (commit to top N layers) for FRI's
   commit-phase Merkle caps. Straightforward extraction once the tree
   is built.

## What to NOT do

- Don't enable any kernel without running the equivalence tests above
  and getting them to pass. Wrong evaluations = silently invalid
  proofs.
- Don't add multi-GPU support. Per project memory, single-GPU only.
- Don't add BabyBear/KoalaBear support. Per project memory,
  Goldilocks-only.
- Don't refactor multi-stark to a Cargo workspace just for the GPU
  module. The inline-feature-gated structure was intentional.
- Don't pin pil2's `goldilocks_trace_layout` machinery into our
  kernels — we only borrow algorithmic patterns from pil2, not its
  data-layout abstractions.
