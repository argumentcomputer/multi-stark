# Init KZG Pipeline Review

Read-only review of the uncommitted `kzg-cuda` work on
`ap/init-recursive-kzg-minimal`, the saved run records, and a second
reviewer's findings, as of 2026-10-09. No builds or GPU runs were made for
this document. File references point at the working tree on that date.

## Revisions, 2026-10-09

Measurements and code moved after this review was written. The sections
below it are kept as the record of the 29m00s run; read them with these
corrections.

- **Measured since.** With the regenerated 19-circuit root, FRI compression
  included: 1,201 s (20m01s) with cached SRS and fresh fixed preprocessing,
  and 869 s (14m29s) with both caches populated and fresh witnesses. Both
  reproduced the previous proofs byte for byte. Records:
  `experiments/kzg-cuda-gpu-feeding-results.json` and
  `experiments/kzg-cuda-validation/gpu-feeding-*-report.json`. The
  "per-commit floor" table below was a projection; the cached run is its
  measurement.
- **Finding 1 is misattributed.** The corrected timer diagnostic
  (`corrected_timer_diagnostic` in `experiments/kzg-cuda-sub30-results.json`)
  shows that at 2^24 rows quotient commitment spends 9.04 s of 13.15 s
  reconstructing evaluations and 1.53 s preparing and sweeping constraints;
  lookup commitment spends 3.35 s of 6.61 s on reconstruction. In the
  recursive stage the streaming LDE is 136 s of the quotient phase. The
  cost is data movement and re-FFT, not the arithmetic sweep, so resident
  evaluations and device-side lookups come before a constraint kernel.
- **Findings 2 and 3 are partly addressed.** The adapter now borrows limb
  storage without packing, moves transfers through pinned rings, runs
  two-slot FFT queues per device, and keeps interpolated coefficients
  resident within a per-device budget. Still open: lookup and constraint
  evaluation read downloaded row-major evaluations, host coefficient copies
  are retained for checkpoints, the half-VRAM budget leaves four recursive
  columns on the host, and the resident MSM path re-uploads SRS chunks per
  commit.
- **Finding 7 is fixed in source** after the measured binaries were frozen;
  the archived `constraint_seconds` values remain invalid as sweep timings.
- **Facts corrected.** The recursive fixed trace has 15 columns, not 17.
  Stage-one main widths are not uniformly 3; four traces are 32 or 40 wide.
  FRI compression already runs on the CUDA Goldilocks backend, so roadmap
  step 3 is void. P2P access exists only within GPU pairs (0,1) and (2,3).
- **Query-count claim withdrawn.** "Blowup 8 brings queries to about 34 for
  the same soundness" used a conjectured-security heuristic. pil2-proofman's
  provable-regime calculator gives 73 queries at blowup 8 with 20 grinding
  bits (110 at blowup 4). Any change to the compression profile needs its own
  soundness analysis before the stage-one saving is claimed.
- **Upstream designs** for the device model, resident keys and per-device
  stream pools are summarized in
  [`docs/upstream-gpu-orchestration.md`](../docs/upstream-gpu-orchestration.md).
  SP1's GPU prover is open source in the main repository, contrary to an
  earlier statement in this session.
- **Related design work** on replacing both KZG stages with one BLAKE3
  terminal proof is in [`docs/blake3-terminal-statement.md`](../docs/blake3-terminal-statement.md)
  and [`docs/plonk-terminal-backend-review.md`](../docs/plonk-terminal-backend-review.md).

## Verdict

The 29m00s run is bounded by CPU work and data movement, not by GPU kernels.
Across the whole pipeline each GPU averaged 3.0 to 3.7 percent sampled
utilization and reported zero in 90 to 92 percent of samples. The MSM, FFT,
evaluation and division primitives are solved. What remains is lookup and
constraint evaluation on the CPU, two serialization round trips per stage, and
a second recursive stage that takes more than half the total and exists only
to shrink a 21-trace first-stage proof.

For a per-commit compressor the relevant number is the floor with fixed data
cached, about 18 minutes today. Reaching under 10 minutes needs GPU lookups
and constraint evaluation, removal of the staging round trips, and one
structural change: proving stage one as a single tall trace so the wrap
verifies about 30 commitments instead of about 600. With fewer FRI queries in
the compression step, about 4 minutes is plausible. BLAKE3 stays throughout.

## Measured state

The `sub30-v1` run on 2026-10-08 used four RTX PRO 6000 Blackwell GPUs, a
96-vCPU host with 999 GB RAM, and NVMe for all checkpoints. It starts from the
saved compressed FRI proof and includes fresh development SRS generation,
checkpoint writes, in-process verification and tampering checks. Both stages
matched the saved CPU artifacts byte for byte.

| Phase | 54m30s run | 29m00s run | Peak RSS |
| --- | ---: | ---: | ---: |
| FRI compression (outside the timed pipeline) | 349 s historical | 172 s regenerated | 232 GiB |
| FRI → KZG staging | 365 | 270 | 72 GiB |
| FRI → KZG proving | 867 | 537 | 206 GiB |
| Recursive KZG staging | 386 | 226 | 184 GiB |
| Recursive KZG proving | 1653 | 707 | 462 GiB |
| Pipeline, excluding FRI compression | 3270 | 1740 | |

The regenerated FRI compression used a newer ix root with 19 circuits and 7
arithmetic partitions instead of the historical 22 and 10, so its 172 s is not
an identical-workload comparison. The results file's target is under 1800 s
including FRI compression; the run is at roughly 1912 s with the regenerated
figure.

### Recursive prove, 707 s

| Segment | Seconds | GPU busy |
| --- | ---: | --- |
| Development SRS at 2^28 (fixed-base table) | 100 | no |
| Load 146 GB fixed matrix (single-threaded zstd pipe) | 121 | no |
| Commit and checkpoint fixed trace, load main trace | 132 | partly |
| Commit main trace | 19 | yes |
| Lookup construction and commitment | 95 | ~1% |
| Quotient construction and commitment (two cosets) | 198 | ~5% |
| Openings, verification, tampering checks | 42 | partly |

Source: `experiments/kzg-cuda-validation/sub30-v1-recursive-prove.txt`.

### Stage 1 prove, 537 s

SRS 6 s, then a serial prepare loop of 254 s over 21 traces (read fixed,
commit, checkpoint, read main, commit, checkpoint), lookups 62 s, and about
204 s of quotient, openings and verification. The prepare loop is dominated by
fixed-trace I/O that is input-independent.

### Staging, 496 s combined

Stage 1: scalar translation 61 s, scalar witness 53 s, lowering 27 s, trace
writes about 123 s. Recursive: native verification 7 s, circuit build about
30 s, witness 57 s, lowering and writes 130 s. All CPU, all before any GPU
work starts.

## What varies per commit and what does not

Every circuit downstream of the aggregation root is a fixed-profile verifier.
Its shape is set by a verifying key plus a `ProofProfile` (envelope,
activation bitmap, log heights, claim lengths), and `VerifierPlan::identity`
excludes public values so one circuit is reused. The FRI compression verifies
the root's profile, stage 1 verifies the compression proof's profile, and the
wrap verifies the stage-1 system. The 21 traces at 2^24, the 2^28 recursive
trace, the 174.9M gates and 62.5M lookups are constants of the
recursion-system version. Adding a Lean definition changes the aggregation
tree upstream and the root's bytes, but none of these shapes.

Two consequences:

- The per-commit witness is only the main traces, about 3 columns wide in
  both stages, plus the lookup and quotient polynomials derived from them.
  Everything with "fixed" in its name, the SRS, and circuit construction can
  be cached per recursion-system version.
- The compressor cannot be incremental. A new root is an entirely new witness
  for the verifier circuit, so every commit pays the full per-update floor.

| Per-commit floor with fixed data cached | Today | After findings 1 to 3 |
| --- | ---: | ---: |
| FRI compression (witness, traces, prove) | ~170 s | ~60 s on GPU |
| Stage 1 witness and main traces | ~80 s | ~60 s |
| Stage 1 prove without fixed preparation | ~310 s | ~100 s |
| Recursive witness, native verify, main trace | ~150 s | ~100 s |
| Recursive prove without SRS and fixed trace | ~380 s | ~150 s |
| Floor | ~18 min | ~8 min |

These are estimates from the measured breakdowns. The cached full pipeline is
still listed as pending in `experiments/kzg-cuda-sub30-results.json`.

## Findings

Merged from two independent read-only reviews and ranked by seconds
recoverable. Each item was checked against the current files and the run
logs. "Verified" marks a claim confirmed in code or logs; savings are
estimates.

### 1. Lookup and constraint evaluation run on the CPU while the GPUs idle (verified)

The recursive run spends 95 s in lookup construction and 198 s in quotient
construction; stage 1 spends 62 s and roughly 150 s. Both paths first
reconstruct evaluations from stored coefficients with forward FFTs (lookups at
the trace domain, quotient once per coset), transpose them into row-major
matrices, then sweep rows with Rayon workers. GPU utilization during these
windows is 1 to 5 percent. The Goldilocks backend already has a compiled
constraint-DAG kernel and lookup kernels in `cuda/kernels.cu` to model
BLS12-381 versions on. Keeping evaluations resident on device would also
remove the re-FFT and the duplicate matrices behind the 462 GiB peak.

Evidence: lookup path `src/ark_adapter/config.rs:271`, quotient path
`src/ark_adapter/config.rs:310`, evaluation reconstruction
`src/ark_adapter/pcs.rs:514`. Saving: about 450 s across both stages.

### 2. Each stage serializes everything to disk and reads it back (verified)

Staging costs 496 s and ends with compressed writes of every trace; proving
then decompresses them, transposes, and writes uncompressed polynomial
checkpoints even though the data stays resident. In the recursive prove, 121 s
pass between SRS readiness and the first commit, loading the 146 GB fixed
matrix through a single-threaded zstd pipe; the following 132 s include its
146 GB checkpoint write. The parallel-frame diagnostic improved one 9 GB
matrix load only from 9.58 s to 9.38 s, so decompression is not the limit;
materializing row-major matrices and the checkpoint traffic are. A fused
stage-and-prove path with optional checkpoints, or column-major raw files that
stream straight to the device, removes most of it. The stage-1 prepare loop is
strictly serial, so GPU work never overlaps the next trace's read.

Evidence: driver `experiments/kzg-wrap/src/outer.rs:310`, stage-1 prepare
loop in `examples/init_fri_kzg_prove.rs`, diagnostic
`experiments/kzg-cuda-sub30-results.json:131`. Saving: about 250 s of proving
plus most of the fixed-trace staging.

### 3. Host copies happen inside exclusive device reservations (verified)

The FFT wrapper acquires a device, then serially packs host field elements
into limb buffers, calls a CUDA entry that allocates, uploads, transforms,
downloads and synchronizes, and unpacks before releasing. At 2^28 elements one
column is 8 GiB each way. With four devices leased one operation each, the
packing time is GPU time lost to the other columns. Persistent device buffers,
packing outside the lease, and fewer matrix conversions are the fixes; the
explicit limb representation should stay for ABI safety.

Evidence: `src/ark_adapter/cuda.rs:311`, `cuda/kzg.cu:119`. Saving: tens of
seconds per stage.

### 4. The recursive stage is over half the total and its size is set by commitment count, not rows (verified)

Staging plus proving the wrap is 933 s of the 1740 s. From the staging log its
174.9M gates split into 45.7M for curve-point inputs, 16.9M for the
transcript, and 112.3M for MSMs. Each witness commitment costs about 76K gates
for on-curve and subgroup checks and about 80K gates per MSM base appearance;
the 357 fixed commitments are already free constants. The roughly 600 witness
points and 1,360 base appearances exist because stage 1 is capped at 2^24 rows
and splits into 21 traces, each with its own main, lookup, quotient and
shifted-degree commitments.

Proving stage 1 as one 2^28 trace keeps roughly the same total rows and
shrinks the wrap to an estimated 15M to 20M gates, a 2^25 domain, about a
minute. It is also the enabling step for a single terminal design: a
30-commitment KZG proof is a few kilobytes and verifies with a handful of MSMs
and pairings. Cost: stage-1 host RAM rises toward the 462 GiB the recursive
stage uses now.

Evidence: height cap `examples/init_fri_kzg_prove.rs:105`, gate attribution
`experiments/kzg-cuda-validation/sub30-v1-recursive-stage.txt`, 237,408,925
rows before padding. Saving: about 600 s.

### 5. Stage 1 is sized by the compression proof's query count, which is adjustable (verified)

The stage-1 source circuit checks 60,297 BLAKE3 compressions, about 600 per
query, and they are most of its 39M gates. The compression step runs 100
queries at blowup 2 with 20-bit grinding, root-only Merkle trees and binary
FRI. Blowup 8 brings queries to about 34 for the same soundness and cuts
stage 1 by roughly two thirds. Merkle caps and higher folding arity would
shorten each query's paths further, but the in-circuit verifier currently
rejects caps and asserts binary folding. The hash stays BLAKE3; nothing here
depends on changing it.

Evidence: parameters `examples/support/root_profile.rs:44`, cap rejection
`src/plonkish/verifier/plan.rs:333`, binary-FRI assert
`src/plonkish/verifier/pcs.rs:54`. Saving: about 200 s after finding 4.

### 6. The new caches work but do not yet give reusable preprocessing (verified)

The fixed preprocessing cache stores circuit metadata, fixed commitments and
coefficients keyed by executable digest plus plan identity, and the SRS cache
reloads 2^24 in 1.5 s. Neither has been measured in a full pipeline. On a
stage-1 cache hit the driver still builds and lowers the whole circuit before
restoring. In the wrap builder, claim values are allocated as constants and
only afterwards replaced by public variables, which leaves
statement-dependent constant gates in the circuit, so its fixed-data cache
correctly binds the expected statement and is useless across commits.
Allocate public slots first, then version the circuit and keys; weakening the
cache key would be wrong.

Evidence: rebuild on hit `examples/init_fri_kzg_prove.rs:47` onward, constant
allocation `experiments/kzg-wrap/src/native_verifier.rs:237`, statement
binding `experiments/kzg-wrap/src/saved.rs:81`.

### 7. Two instrumentation defects would misdirect the next round (verified)

The `KZG quotient constraints evaluated` event records `constraint_seconds`
before the parallel constraint loop runs, so it excludes the work it names;
the outer per-coset event is correct but merges the LDE with the sweep. The
benchmark summary keeps only peak GPU utilization, which hides the idle
periods visible in the raw CSV. Fix both before using the new timers to
prioritize finding 1.

Evidence: `src/ark_adapter/config.rs:175`, `experiments/kzg-cuda-bench.py:253`.

### 8. SRS generation is no longer a bottleneck but the 2^28 cache load is unmeasured (verified)

Fixed-base generation took 6 s at 2^24 and 100 s at 2^28 in the run, down
from 40 s and 642 s. The 24 GiB cache file has not been timed. Production
loads a ceremony SRS anyway, so this is a development-only cost.

### Relation to the second review

Adopted unchanged: the GPU utilization statistics, the 121 s fixed-matrix
load, the decompression diagnostic, the timer bug, the cache binding defect,
and the architectural weight of the recursive stage.

One refinement: its 13m27s stage-1 figure is not the right comparison point
for a terminal single-proof design, because a single tall stage-1 trace
behaves like today's recursive prove. That design's cost is closer to today's
recursive prove without fixed work, roughly 6 minutes now and 2 to 3 minutes
with GPU constraint evaluation.

## Path to a per-commit compressor

Estimated latency from root to packet with fixed data cached, in the order the
steps should land. Each step keeps BLAKE3 and preserves the CPU transcript and
proof bytes.

| Step | Change | Latency |
| --- | --- | ---: |
| 0 | Today, fixed data cached | ~18 min |
| 1 | Findings 1 to 3: GPU lookups and constraints, fused staging, device buffers | ~8 min |
| 2 | Finding 4: stage 1 as one 2^28 trace; wrap shrinks to ~2^25 or becomes the terminal proof | ~6 min |
| 3 | FRI compression prover on the existing Goldilocks CUDA backend | ~4 min |
| 4 | Finding 5: compression step at blowup 8, ~34 queries | ~3 min |

Pipelining across commits adds throughput on top: three stages on four GPUs
give throughput near the slowest stage, roughly 100 s after step 3, while
latency stays at the sum. The binding constraint is host RAM, since the
compression prover peaked at 232 GiB and stage 1 at 2^28 would be near
460 GiB. If commits arrive faster than the floor, compress the latest root and
skip the ones it supersedes.

### Validate first, in this order

1. Fix the quotient timer and record the LDE-versus-sweep split on one 2^24
   trace; `examples/kzg_cuda_prepare_bench.rs` runs it in under a minute.
2. Stage stage 1 with a 2^28 height cap and read the wrap builder's gate
   count from its stats line, before any proving.
3. Build the stage-1 source circuit with the compression profile at blowup 8
   and read its stats line.
4. Run the cached full pipeline the results file lists as pending, so the
   per-commit floor is measured rather than derived.

## Evidence

- Run report `experiments/kzg-cuda-validation/sub30-v1-pipeline-report.json`;
  phase logs `sub30-v1-*.txt` in the same directory; raw output and GPU
  samples under `/opt/dlami/nvme/multi-stark-kzg-cuda-sub30-v1-20261008/`.
- Prior fresh run for comparison:
  `experiments/kzg-cuda-validation/final-pipeline-report.json` (54m30s).
- Iteration record and pending list: `experiments/kzg-cuda-sub30-results.json`;
  README: `experiments/kzg-wrap/README.md`.
- GPU utilization recomputed from the CSV: means 3.7, 3.5, 3.2, 3.0 percent;
  zero samples 90, 90, 91, 92 percent; 925 samples per device over 30m48s.
