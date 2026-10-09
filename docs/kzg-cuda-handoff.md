# KZG CUDA handoff — 2026-10-09

Implementation is paused at a verified result. The checkout is
`/home/sam/repos/multi-stark`, on `ap/init-recursive-kzg-minimal`; the GPU work
is uncommitted and has not been pushed. The measured base commit is
`6d9d5ddad4fb9bfbabf8238ef87af6e9d7b3cab6`. At handoff, all 31 files in the
latest report's source manifest matched the working tree. No scheduler changes
were made after that measurement. No builds or benchmarks were started for
this handoff.

## Preserve before shutting down

`/opt/dlami/nvme` is mounted from `/dev/mapper/vg.01-lv_ephemeral`. Treat its
contents as disposable across instance shutdown or replacement. The checkout
is on `/dev/root`; retain that volume or copy the checkout elsewhere before
deleting the instance or its root volume. A local commit alone is not an
off-instance backup, and untracked files must be included in any copy.

The compact [recovery directory](../experiments/kzg-cuda-validation/resume-20261009/)
preserves 146 files, 58.02 MiB, copied out of the ephemeral volume:

- `root-artifacts/`: the regenerated root proof, verifying key and independently
  exported claims needed to rerun FRI compression.
- `compressed-fri/`: the measured compression output and profile metadata.
- `measured-binaries/`: the three exact benchmark executables.
- `sources/`: the 31 source files captured by the benchmark harness.
- `fri-to-kzg/`, `intermediate/`, `recursive/`: compact proof, packet, setup and
  statement metadata; these are not complete staged proving directories.
- Phase logs, GPU samples, the original report, and `manifest.json` with every
  preserved file's length and SHA-256. All copied files were hash-checked;
  binaries, source snapshots and root inputs also matched the measured report.

Large witness traces, coefficient checkpoints, fixed preprocessing and SRS
caches are deliberately excluded. They can be regenerated from the saved root
and sources. The recovery directory is a local copy on the checkout's volume,
not a remote backup. Keep the whole checkout to retain the rest of the source,
Git history, documentation and validation records; `sources/` is only the
benchmark's selected source snapshot.

## Verified result and scope

On four RTX PRO 6000 Blackwell 96 GB GPUs, with 96 logical CPUs and about
1 TiB RAM:

| Phase | Cached run | Peak host RAM |
| --- | ---: | ---: |
| FRI compression | 171.880 s | 232.4 GiB |
| FRI → KZG staging | 199.039 s | 63.6 GiB |
| FRI → KZG proving | 156.030 s | 190.7 GiB |
| Recursive KZG staging | 138.489 s | 84.2 GiB |
| Recursive KZG proving | 204.015 s | 464.2 GiB |

The complete boundary took **869.471 s (14m29s)**, including FRI compression,
both staging/proving steps and in-process verification. Additional independent
CPU verification brought it to **897.394 s (14m57s)**. Fresh fixed preprocessing
with a populated development-SRS cache took **1200.957 s (20m01s)**, or 20m29s
including the separate CPU checks. Upstream root aggregation is outside both
measurements. Neither number includes fresh SRS generation.

Both KZG proofs, packets and profile identifiers match the preceding 23m10s
run and the fresh-fixed run byte for byte. External pairings and tampering
checks passed. The final proof is 2,053 bytes; the complete packet is 2,709
bytes. The regenerated Init root uses a 19-circuit FRI profile, versus the
historical 22-circuit fixture with a 2,757-byte final packet. Do not describe
these newer timings as the same workload as the original five-hour CPU run.

The SRS has a known development trapdoor. The fixed cache binds the executable
digest, and the recursive cache also binds the public statement. Reuse across
arbitrary changed statements has not been demonstrated; never bypass those
cache identities to obtain a hit after rebuilding.

Authoritative records:

- [Latest results](../experiments/kzg-cuda-gpu-feeding-results.json).
- [Cached report](../experiments/kzg-cuda-validation/gpu-feeding-cached-report.json)
  and [fresh-fixed report](../experiments/kzg-cuda-validation/gpu-feeding-cold-report.json).
- [Build, controls and protocol details](../experiments/kzg-wrap/README.md).
- [Earlier implementation review](../experiments/kzg-pipeline-review.md), which
  predates the final measurements and contains projections, not current timings;
  its "Revisions, 2026-10-09" section lists the corrections.
- [Upstream GPU orchestration](upstream-gpu-orchestration.md): how SP1,
  ZisK/pil2-proofman and PILFFLONK place a proof on one device, with pinned
  revisions, and which of their patterns this adapter already has.

The 47 focused adapter tests passed, with three ignored scale/benchmark tests;
the explicit 2^29 FFT and chunked-MSM checks also passed. The CPU-only adapter
suite passed 36 tests. Both manifests' formatting and the whitespace check
passed. Logs are under `experiments/kzg-cuda-validation/gpu-feeding-*`.

## Implemented GPU path

- Immutable polynomial coefficients survive interpolation on their assigned
  device and feed subsequent MSMs, coset FFTs and opening evaluations. Host
  mirrors remain for checkpoints and spill recovery; constants need no device
  allocation. The resident budget is capped at half of each device's VRAM.
- Each GPU has one exclusive operation lease. An FFT batch uses up to two
  stream/work slots, overlapping columns and reusing its scratch allocation
  within that batch. Other operations on that GPU still wait for the lease.
- Each lane has persistent pinned upload/download rings, four 16 MiB chunks
  per direction. Two lanes use 256 MiB pinned memory per device. Events protect
  reuse; host copies run in parallel. These are bounded staging rings, not
  whole-column pinned storage.
- Scalars borrow arkworks Montgomery limbs directly with compile-time layout
  checks. Canonical decoding and host first-touch allocation are parallel.
  Affine point conversion is parallel; frequent nonunit scalars are normalized
  to avoid degenerate Pippenger buckets.
- An MSM point chunk is reused for the columns in its current local batch.
  There is no SRS point cache spanning all calls. Sppark allocation already
  uses `cudaMallocAsync`/`cudaFreeAsync`; no persistent pool release threshold
  has been configured.
- 2^29 MSMs are chunked. Whole-device FFTs support 2^29 and enforce log size
  at most 31, despite compiling with `MAX_LG_DOMAIN_SIZE=32`.
- Lookup and constraint sweeps, opening-polynomial folding, and division's
  carry scan remain on the CPU. The pinned sppark multi-point evaluation
  kernel has a recorded shared-memory race; the adapter instead launches
  single-point kernels against one retained coefficient buffer.

The recursive run retained about 160 GiB of coefficients; peak sampled KZG
device memory was 56.7 GiB per card. Four large columns spilled under the
half-VRAM limit, causing 32 GiB of uploads in recursive FFTs and another
32 GiB in evaluations. Stage-one forward FFTs and evaluations uploaded no
coefficients. Mean sampled utilization during KZG proving remained about
6–9% per GPU. Staging, CPU consumers, host copies and repeated SRS transfers
still leave idle periods. Two-second samples miss short kernels; CUDA-event
totals overlap across devices and cannot be summed into wall time. MSM events
currently record transfers only, and division's compute interval includes its
CPU carry pass. The old 23m10s report's constraint timer preceded the loop;
use the corrected diagnostics and latest run for constraint attribution.

## Next work, still unimplemented

The SP1-inspired scheduling proposal was only investigated; there is no
partially applied scheduler patch. The next substantial change should make
mixed FFT/MSM/transfer tasks safe on independent stream sets per device, with
shared memory admission. Do not merely replace the busy flag with a counter:

1. Sppark's `msm_t` captures `select_gpu(device_id)` and uses that runtime's
   zero and three flip-flop streams. Concurrent MSM contexts need independent
   stream sets. `transfer_lane` also keys shared mutable rings by device and
   lane, so concurrent tasks must not reuse the same rings.
2. Retained coefficients, SRS points, active scratch and cached free memory
   must share one budget. When retaining CUDA pool allocations, account for
   reserved-but-unused pool bytes; `cudaMemGetInfo` alone can understate
   reusable capacity. Add a bounded pool retention policy without consuming
   the transient reserve or breaking 2^29 admission.
3. A whole-column pinned pool is an alternative to measure against the
   existing rings. Avoid repeatedly registering/unregistering 8–16 GiB vectors.
   Bound pinned memory and preserve buffer lifetimes through every DMA and
   error path.
4. Stream completion callbacks may signal a future, but must not call CUDA
   APIs. Borrowed FFI buffers must stay alive through completion and unwinding.
   Waiting on that future from the same Rayon worker would still occupy the
   worker; the synchronous PCS interface needs an explicit dispatch/completion
   design. Current native `gpu.sync()` calls and the failure path's
   `cudaDeviceSynchronize()` also need review before enabling concurrency.

The relevant code is [the Rust adapter](../src/ark_adapter/cuda.rs),
[native primitives](../cuda/kzg.cu), [transfer rings](../cuda/kzg_transfer.cuh),
[runtime](../cuda/kzg_runtime.cpp), and [PCS integration](../src/ark_adapter/pcs.rs).
The existing FRI backend has useful admission and pipeline patterns in
`src/cuda/pcs.rs`, `src/cuda/witness.rs` and `cuda/kernels.cu`.
The pinned sppark revision is `e10e107673aa22861f0f8b9758fc62169ab919ae`.
Its runtime-header context-selection patch and symbol-renaming shim remain
integration debt; upstream namespace/stream injection would be cleaner.

Follow mixed-task changes with overlapping-operation parity, buffer reuse,
failure/lifetime and memory-pressure checks, then a same-fixture diagnostic
and verified complete run. Preserve transcript order and the existing FRI
security/profile parameters. Treat projected speedups as unmeasured.

## Resume commands

Run from the checkout root on a machine with the CUDA toolkit and sufficient
host memory. These commands are for resumption, not shutdown preparation.

```sh
MULTI_STARK_CUDA_ARCHS=120 cargo build --release --locked \
  --features parallel,kzg-cuda,cuda \
  --example init_fri_kzg_prove --example ix_root --example kzg_cuda_prepare_bench
MULTI_STARK_CUDA_ARCHS=120 cargo build --release --locked \
  --manifest-path experiments/kzg-wrap/Cargo.toml --features kzg-cuda

MULTI_STARK_CUDA_ARCHS=120 cargo test --release --locked \
  --features parallel,kzg-cuda,cuda --lib ark_adapter:: -- --nocapture
```

Alternatively, the three preserved executables can be restored to
`target/release/examples/{init_fri_kzg_prove,ix_root}` and
`experiments/kzg-wrap/target/release/init-kzg-wrap` on a compatible host. They
use native CPU code and Blackwell cubins; rebuild for a different architecture.
Their SHA-256 values are in the recovery manifest and original report.

For a complete run including FRI compression, use the preserved root:

```sh
MULTI_STARK_KZG_CUDA_PROFILE=1 python3 experiments/kzg-cuda-bench.py \
  --root-artifacts experiments/kzg-cuda-validation/resume-20261009/root-artifacts \
  --fixed-cache /opt/dlami/nvme/kzg-resume/fixed-cache \
  --srs-cache /opt/dlami/nvme/kzg-resume/dev-srs-cache \
  --output /opt/dlami/nvme/kzg-resume/run-1
```

Substitute a sufficiently large fast volume if that NVMe mount is absent.
The output directory must be new. An empty-cache first run repopulates fixed
preprocessing and SRS; repeat with a new output directory and unchanged
binaries for the cached boundary. Do not compare the first run to 14m29s
without reporting that difference. The driver performs independent CPU
verification and retains source/binary hashes, phase logs and GPU samples.

Original disposable paths, if the NVMe contents survive:

| Contents | Path under `/opt/dlami/nvme/` |
| --- | --- |
| Cached complete run | `multi-stark-kzg-resident-cached-20261009` |
| Fresh fixed preprocessing run | `multi-stark-kzg-resident-cold-20261009` |
| Fixed cache | `multi-stark-kzg-resident-fixed-cache-20261009` |
| Development SRS cache | `multi-stark-kzg-sub30-srs-cache` |
| Original regenerated root | `multi-stark-init-root-20261008/root` |
| One-trace diagnostic input | `multi-stark-kzg-sub30-cache-setup-20261009/fri-to-kzg` |

The saved root eliminates the need to repeat the upstream Ix export merely
to resume these measurements. Existing cached fixed files are immutable and
may be hard-linked into run directories; never edit them in place.
