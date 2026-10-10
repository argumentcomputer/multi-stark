# Upstream GPU prover orchestration

How SP1, ZisK and pil2-proofman organize GPU work, read from their sources on
2026-10-09, and what carries over to the KZG backend in
[`src/ark_adapter/cuda.rs`](../src/ark_adapter/cuda.rs). Every claim below
names the file it comes from at the pinned revision. Timings are the
upstream projects' own records, on their hardware, and are not comparable
with this repository's measurements without rerunning them.

| Project | Revision read | Branch |
| --- | --- | --- |
| succinctlabs/sp1 | `4ed918fe19c98db041066ea7a2c5f3201a4f03db` (2026-10-07) | `main` |
| succinctlabs/sp1-cluster | `e878e6d5d03613de39ab574d6dbeef0b76f0dfab` (2026-09-24) | `main` |
| 0xPolygonHermez/zisk | `2644d20f26222414d0ca210c0e07c2ac407647fa` (2026-10-08) | `pre-develop-1.4.0-alpha` |
| 0xPolygonHermez/pil2-proofman | `27119b4ef3bdd2f74ebd4a0c17c374503ab90837` (2026-10-08) | `pre-develop-1.4.0-alpha` |
| 0xPolygonHermez/pil2-proofman | `d65236a3e720fdbe11ddfdcae07bc030628b0eef` (2026-10-06) | `feature/pilfflonk`, open PR #610 |

The common pattern: no project splits one proof across devices. Each proof
runs on one GPU with its data resident for the proof's life, and additional
GPUs take independent proofs. Multi-GPU is a scheduling layer above the
prover, not inside it.

## SP1

The GPU prover is open source in the main repository under `sp1-gpu/`
(19 crates, 109 CUDA files under `sp1-gpu/crates/sys`). The separate
`succinctlabs/sp1-gpu` repository is deprecated and points there.

**Device model.** `sp1-gpu/crates/server/src/main.rs` reads
`CUDA_VISIBLE_DEVICES` and requires exactly one device id; the SDK connects
to one Unix socket per device (`crates/cuda/src/client.rs`). A `TaskPool` is
built per device (`sp1-gpu/crates/cuda/src/task.rs`). Multi-GPU is
`sp1-cluster`: one worker per GPU (`bin/coordinator/src/lib.rs:70`), a
coordinator assigning core-shard, recursion and wrap tasks, and concurrency
bounded by host RAM because each in-flight task keeps its shard data on the
host (`crates/worker/src/config.rs`).

**Runtime (`sp1-gpu/crates/cuda`, crate `sp1-gpu-cudart`).**

- `TaskPool` holds up to 64 `Task`s; each `Task` owns a CUDA stream and an
  end event. `TaskScope` is both the async unit of work and an allocator:
  device memory comes from the device's `cudaMallocAsync` pool with a
  release threshold (`crates/sys/lib/runtime/mem_pool.cu`), not from a raw
  allocation per call.
- `StreamCallbackFuture` (`crates/cuda/src/stream.rs:128`) resolves when the
  stream drains, through a host callback, so no thread blocks in a
  synchronize.
- `PinnedBuffer` and `WorkerQueue`: `new_cuda_prover` in
  `crates/prover_components/src/components.rs` allocates four pinned trace
  buffers sized to the largest trace. A shard takes one, generates its traces
  on the device (`crates/tracegen`), uploads, and proves under
  `ProverSemaphore::new(1)` (`crates/prover_components/src/builder.rs:107`):
  one shard proves while the next ones stage.
- The per-shard pipeline is device-resident end to end
  (`crates/shard_prover/src/prover.rs`): commit, logup GKR, zerocheck,
  jagged sumcheck, basefold evaluation proof. Only the proof returns.

**Terminal stage.** The wrap is an outer STARK over Poseidon2 on BN254
scalars, proven by the same machinery (`CudaProverWrapComponents` with
`Poseidon2Bn254CudaProver`), then Groth16 or PLONK through gnark on the CPU
(`crates/recursion/gnark-ffi/go/sp1/prove_groth16.go`), or through ICICLE
with the `groth16-cuda` feature (`prove_groth16_icicle.go`). SP1 has no KZG
prover of its own, and its wrap hash is Poseidon2, so it informs the runtime
layer here and not the terminal circuit.

## ZisK and pil2-proofman, STARK path

ZisK's prover is pil2-proofman; the GPU code is C++/CUDA under
`pil2-stark/src`, driven from Rust in `proofman/src`.

**Streams as lanes.** Each GPU holds one unified buffer carved into aux-trace
slots by size class (`common/src/gpu_stream_layout.rs`), giving `n_streams`
basic streams plus aggregation streams per GPU (`proofman/src/proofman.rs`,
`n_streams_per_gpu`). One proof runs on one stream from witness staging to
proof download. `StreamData` (`pil2-stark/src/goldilocks/src/goldilocks_tooling.cuh:370`)
carries the stream, two low-priority rebuild lanes for fixed and custom
commits, per-stream pinned staging, an atomic status, and a `trace_copy_event`
so the host trace buffer is recycled as soon as the H2D copy finishes rather
than when the proof ends.

**Scheduler (`proofman/src/scheduler.rs`).** A key-affinity scheduler
reserves streams in three passes: a free stream already warm with the same
const tree, then a fresh load of an unloaded key, then a reload. Reservations
are guards that release on drop. Witnesses are staged into pinned prefetch
zones (`starks_api.cu`, `PrefetchZone`) with one chunk in flight so the copy
engine is shared across streams. Completed proofs are collected from a harvest
ring without a host sync at reserve time.

**Buffer layout.** Open PR #616 reorganizes the per-GPU unified buffer so the
final SNARK and ZisK's memory operations borrow regions of it between proofs,
reloading only the fixed data that the borrow overwrote, from pinned host
copies.

**Peer access.** Not used by either project. On this host `nvidia-smi topo
-p2p r` reports P2P only within the pairs (0,1) and (2,3)
(`experiments/kzg-cuda-gpu-feeding-results.json`, `peer_access`), so any
cross-device redistribution here goes through the host for cross-pair moves.

## pilfflonk, the KZG analog

PR #610 adds `pilfflonk`: a PIL2 program proven with KZG over BN128, columns
packed into a few polynomials `f_i`, one SHPLONK opening, one pairing. Its
GPU design rules (`pilfflonk/docs/performance.md`, section "Rules of the
device path", at `d65236a`) are the closest published analog to this
repository's stage one:

- **The key is resident for its life**: SRS powers, fixed coefficients
  already interpolated by the setup, bytecode. A proof uploads its witness
  once; at `2^21` rows the key copies 3.3 GB up and 200 bytes down, and a
  proof copies 604 MB up and 4.7 KB down.
- **One device, one arena per proof.** Phases reuse each other's bytes. A key
  or proof that does not fit is refused at load; nothing falls back to the
  host mid-proof.
- **Everything but the transcript runs on the device**: stage columns and
  lookup hints, intermediate polynomials, all of `Q` through a bytecode
  interpreter (`pilfflonk_expressions.cu`), the coset LDE, SHPLONK
  evaluations and quotients (`pilfflonk_opening_gpu.hpp`).
- **Repeated scalars**: every MSM adds `ρ_i = h^(i+1)` to its scalars and
  subtracts a per-length precomputed `Σ ρ_i [τ^i]`, computed once at key load
  (`GpuKey::addShiftSums`). This is a general form of the structured-scalar
  normalization in `cuda.rs` and costs nothing per proof.
- **Measured** (RTX 5090, CUDA 13.0, 32 host threads): the `L1 2^21` fixture
  proves in 2.1 s on the GPU against 61.5 s on the CPU; `fibonacci 2^22` in
  0.9 s. The BLAKE3 wrap key (`2^20` rows, 65 fixed columns, 30.4M SRS
  powers) loads in 0.72 s and restores into a borrowed buffer in 0.17 s. Its
  opening spent 9.04 s before the interpolant fix described in the same
  document. These are phase and key figures; the document's end-to-end
  wrap chain timings are for a Poseidon2 key.

Scope limits stated in `pilfflonk/docs/README.md`: one instance of one AIR,
no air or airgroup values, no global constraints, no custom commits. Adopting
its layout requires an explicit lowering of this repository's multi-AIR
output.

## BLAKE3 verifier geometry in pil2-proofman

Two different AIRs, both BLAKE3, both in the `feature/pilfflonk` tree:

| AIR | Rows per compression | Columns | Where |
| --- | ---: | --- | --- |
| Goldilocks recursion aggregator (`blake3.pil`) | 56 (7 rounds × 8 G) per block, `k` lanes per row | 53 witness per lane + 2 shared, 8 fixed; stage 2 is 42/21/12 per lane at blowup 2/4/8 | `setup/stark-recurser/docs-research/README.md` |
| BN128 final wrap (`blake3_bn128/wrap.pil`) | 64: 56 G rows plus 8 feed-forward rows | `st[16]` opened at the row and the next, every other column at its row only; six PLONK gates per row on free rows | header comment of `setup/stark-recurser/plonk2pil/pil/blake3_bn128/wrap.pil` |

Both rely on an XOR lookup table and a range checker, and both recompute the
block interior on the device from boundary rows (`gate_bands_blake3.hpp`)
instead of storing it in the witness. In the circom verifier that feeds the
wrap (`circuits.gl/hash/blake3/blake3.circom`), a Merkle node is a dedicated
custom gate costing 5 linear constraints at the gate boundary, a general
compression 17.

Related data from the same tree:

- `common/src/hash_family.rs`: BLAKE3 recursion circuits settle at `2^19`
  rows against `2^17` for Poseidon, because arity-2 Merkle paths dominate at
  56 rows per compression; binary-tree families use a solved FRI folding
  schedule (`stark_struct::optimal_fri_steps`) instead of a fixed fold.
- `setup/stark-recurser/docs-research/README.md`: Merkle paths are 70 to 75
  percent of verifier hashes; the hashes-example verifiers count 21,576 to
  29,208 compressions. Query counts come from the JBR-regime closed form in
  `setup/pil2-stark/src/types/security/pcs/fri.rs`,
  `t = ceil((128 - grinding) / -log2(1 - pp))` with
  `pp = 1 - sqrt(rate) - 1/300`: 110 at blowup 4 and 73 at blowup 8 with
  20 grinding bits. This is a provable-soundness count; it is not the
  conjectured-security count this repository's `FriParameters` discussion
  uses, and the two must not be mixed in a comparison.

Release status matters: the ZisK `pre-develop-1.4.0-alpha` installation
guide (`book/getting_started/installation.md`) still says the BN128 wrap is
Poseidon-only and that `setup-snark` refuses a BLAKE3 key. The BLAKE3 wrap
exists only on the `feature/pilfflonk` branch.

## What applies to `cuda.rs`

Already present in the in-flight backend as of 2026-10-09 (see the CUDA
section of [the Init pipeline README](../experiments/kzg-wrap/README.md)):
pinned transfer rings, direct limb borrowing without packing, two-slot
per-device FFT queues, resident coefficients after interpolation with
per-device admission, resident-input coset FFTs, MSMs and evaluations.

Not yet present, in the order the upstream designs suggest:

1. **A per-device pool of stream-backed tasks** in place of one lease per
   device, so independent columns overlap uploads, transforms and MSMs on the
   same GPU (SP1 `TaskPool`, pil2-proofman streams).
2. **Device memory from a pool with a release threshold** rather than
   allocation per call (SP1 `mem_pool.cu`, sppark's `cudaMallocAsync`).
3. **Completion through stream callbacks into futures**, so host workers do
   not park in synchronizes (SP1 `StreamCallbackFuture`).
4. **A resident key**: SRS chunks and fixed coefficients uploaded once per
   process (pilfflonk `GpuKey`). The resident MSM path here still re-uploads
   SRS chunks per commit call.
5. **Lookup and constraint evaluation on the device** against resident
   evaluations, so the row-major downloads that the CPU sweeps need
   disappear (pil2-proofman expressions and lookup kernels, pilfflonk `Q`
   interpreter). The corrected one-trace diagnostic in
   [`experiments/kzg-cuda-sub30-results.json`](../experiments/kzg-cuda-sub30-results.json)
   shows evaluation reconstruction, not the arithmetic sweep, as the larger
   cost, which favors this over a standalone constraint kernel.

Before applying items 1 to 3, read the constraints in
[the KZG scheduling and memory plan](kzg-performance.md#cuda-scheduling-and-memory-constraints): sppark's
`msm_t` captures the selected device's zero and flip-flop streams, the pinned
rings are keyed by device and lane, completion callbacks must not call CUDA,
and retained coefficients, SRS points, scratch and pool reserve must share one
memory budget. Replacing the busy flag with a counter alone is unsafe.

Lane-style scheduling of whole partitions, as the ix prover does for shard
and join proofs, applies to stage one, whose 21 partitions are independent
until their commitments are observed in transcript order. It does not apply
inside the single-circuit recursive stage, where column fan-out is the only
intra-proof parallelism and a row-wise kernel would need an all-to-all over
the host for cross-pair devices.

## Local copies

Shallow clones of the revisions above were made into the session scratchpad
and should be moved to `~/repos/clones/zk/` for reuse. Fetch the upstream
default branch before browsing, as the repository conventions require.
