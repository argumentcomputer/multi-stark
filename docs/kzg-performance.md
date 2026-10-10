# KZG performance: roadmap to under five minutes

Updated 2026-10-10. This document combines the performance roadmap, measured
CUDA handoff, trace/witness design, and recursive circuit-size proposals.
The latest [complete prepared-worker development-v4 request](#integrated-development-v4-worker-pipeline)
takes **122.455 seconds (2m02s)** from an aggregation root to a verified final
packet, including fresh FRI compression and fresh witnesses for both KZG stages.
Both workers' preparation takes **404.353 seconds** separately. The earlier
839.988-second (14m00s) cached process pipeline remains a historical baseline;
its startup boundary differs. The objective is **less than 300 seconds** with
authentic Filecoin parameters and fresh witness generation. The development
request meets the time and size limits; the ceremony acceptance gate remains
open. Proposed improvements and circuit-size estimates below are unmeasured
unless identified as results.

The timing boundaries are:

| Measurement | Seconds | Minutes | Included work |
| --- | ---: | ---: | --- |
| Prepare both workers | 404.353 | **6m44s** | Circuit construction, cached development SRS loading, cold fixed preprocessing, and a complete warmup proof at each KZG stage |
| Fresh request with prepared workers | 122.455 | **2m02s** | Fresh FRI compression, fresh witnesses, both KZG proofs and in-process verification |
| Preparation followed by a fresh request | 526.810 | **8m47s** | Two warmup proofs, then another complete root-to-packet request |
| Whole benchmark command | 535.832 | **8m56s** | The preceding work, worker shutdown, separate CPU verification and driver overhead |
| One cold root-to-packet request | Unmeasured | — | No measured single-request cold-start result on this version |

**8m47s is warmup followed by a second request, not the measured latency of one
cold request.** The preparation itself proves and verifies both stages; it is
more than loading caches. Under five minutes has been demonstrated only for
the prepared development service, using known-trapdoor parameters. The latest
[direct fixed-input optimization](#feed-fixed-preprocessing-directly-into-fused-proving)
postdates this integrated run; its effect on production startup and total time
is unmeasured. See the [shutdown handoff](#preserve-before-shutting-down) for
the current source, evidence and cache locations.

Git retains the implementation, documentation and reproduction scripts. New
benchmark outputs, binaries, proof fixtures and duplicated source snapshots
are excluded. Evidence links below identify local validation files; those
outputs must be regenerated or restored separately on a fresh checkout.

The implementation target is a **maximum committed trace length of
`2^27`**, one recursive computation trace, and Filecoin ceremony parameters
for both KZG stages. The complete final packet must stay **below 3,000 bytes**;
the previous 2,053-byte proof and 2,709-byte packet remain comparison points. The
[migration plan](#migration-to-a-227-trace-ceiling) separates circuit fit,
the public-setup degree policy, and parameter import. Historical `2^28`
measurements remain evidence; new work should target the smaller trace cap.
The [sizing report](#sizing-and-degree-policy-report) recommends evaluating
mixed heights before uniform padding, which adds 85% to stage-one matrix
volume. The [protocol argument](#full-public-degree-soundness-argument) states
the generic bilinear group, random oracle and ceremony assumptions explicitly.
The integrated development measurement establishes full-request timing and
correctness for its known-trapdoor setup. It does not establish a proof using
the real ceremony artifact.

The latest [genuine development-v4 recursive proof](#complete-development-v4-recursive-proving)
uses **118,634,362 rows**, leaving **15,583,366 rows (11.61%)** below `2^27`
without wrapper sharding. Complete fresh assignments, native MSM/pairing checks
and independent CPU packet verification pass. The actual development proof is
**1,909 bytes**, with a **2,181-byte complete packet**. The earlier
[shared-builder census](../experiments/kzg-cuda-validation/recursive-worker-20261009/census/report.json)
uses one fewer row with its different development profile. Actual ceremony fit
and packet size still need validation with authentic parameters.

The earlier [FRI coexistence replay](#fri-with-two-idle-kzg-contexts) verifies
identical proof/key/claim bytes in **56.280 seconds** while two small KZG
fixtures keep their CUDA contexts alive. The earlier isolated replay took
**73.171 seconds**, with a different executable; these samples do not measure
the cost of coexistence. Both are separate from the 839.988-second full-chain
baseline. See [FRI memory placement](#fri-memory-placement) for the earlier
matched diagnostics.

An isolated [first-stage fused run](#first-stage-fused-run-with-cold-fixed-preprocessing)
verifies identical proof/packet/profile bytes in **279.87 seconds**, including
cold fixed preprocessing and a warm development SRS. Its frontend takes
93.145 seconds and proof computation takes 13.848 seconds. This affected-stage
measurement excludes fresh FRI compression and the recursive KZG stage.

The latest [fresh scalar-assignment comparison](#evaluate-independent-scalar-advice-in-parallel)
reduces that operation from **27.742 to 3.624 seconds**, including input
conversion and constraint checks. It uses the preserved production input and
the default CPU parallelism. This is one affected-operation A/B pair, with no
new lowering, proving or complete-chain timing.

The earlier [retained first-stage request](#production-first-stage-worker-measurement)
takes **36.647 seconds**, including fresh witness generation, proving, native
verification and GPU release. Its genuine preparation proof takes **240.717
seconds** separately, with cold fixed preprocessing and a warm development
SRS. Both requests match the prior first-stage proof/packet/profile bytes;
independent CPU verification passes. This measures one prepared first-stage
request, without fresh FRI compression or a recursive proof.

The complete [retained recursive request](#complete-development-v4-recursive-proving)
takes **34.412 seconds**, including fresh assignment, fused trace generation and
proving, verification and GPU release. Its first request takes **197.284 seconds**
with cold fixed preprocessing and a newly generated `2^27` development SRS;
compilation/lowering take another 14.375/4.042 seconds separately. Both requests
produce identical proof, packet and profile bytes. The separate
[frontend diagnostic](#measure-the-complete-v4-recursive-frontend) measures
9.884–9.922 seconds for fresh assignment and both traces alone.

The [integrated run](#integrated-development-v4-worker-pipeline) now measures the
three phases together: **57.173 seconds FRI**, **30.691 seconds first KZG**, and
**34.554 seconds recursive KZG**. Both complete workers remain alive throughout
the fresh request, release their GPU caches between phases, and reuse their
host keys. The 2,181-byte packet passes separate CPU verification and matches
the prior recursive fixture exactly.

The next work follows these measured boundaries:

1. Obtain the authenticated Filecoin artifact, regenerate both
   stages' keys, and [recount the saved proof](#count-a-saved-proof-with-the-filecoin-setup)
   with real constants. The complete development request now has a measurement;
   the missing ceremony artifact remains the acceptance blocker. The actual
   2,181-byte development packet does not establish a Filecoin result.
2. Repeat the integrated request with those authenticated keys and the existing
   Filecoin acceptance gate. The development result is 122.455 seconds after
   404.353 seconds of preparation; warmup plus a second request is **526.810
   seconds (8m47s)**. One cold request remains unmeasured, so this does not
   establish a cold-start bound below five minutes. Preserve the
   independent expected statement, fresh witness generation, actual final-packet
   verification and four-device kernel evidence when changing parameters.
3. Measure the [direct fixed-input path](#feed-fixed-preprocessing-directly-into-fused-proving)
   in the next authenticated cold startup. Both fused KZG stages now generate
   fixed matrices directly into preprocessing while retaining coefficient
   checkpoint publication. The bounded handoff comparison falls from 0.578 to
   0.086 seconds; production startup savings remain unmeasured. The prior
   [isolated recursive run](#complete-development-v4-recursive-proving) spent
   36.183 seconds materializing/compressing fixed traces and 31.256 seconds
   reading/decoding them. Materialization remains necessary, so the full
   67.439 seconds is not removable overhead. Prepared requests already skip it.
4. Let the integrated profile guide further transfer and kernel work. The fresh
   recursive request uses all four GPUs for the `2^27` lookup and quotient,
   taking 4.154 and 4.598 seconds respectively. Sampled utilization averages
   12–30% across that 34.554-second request, which includes CPU work and transfers.
   FRI is now the largest request phase at 57.173 seconds and nearly fills GPU 0.
   Inspect dependency gaps and repeated uploads before adding streams; the
   measured development request already fits the five-minute budget.

Keep targeted measurements separate from integrated request measurements. Standalone
MSM/FFT tuning and additional stream concurrency follow evidence that they
limit the remaining critical path.

- [Goal, priorities, and first experiments](#goal-and-priorities)
- [Measured results](#measured-results)
- [Cache contents and setup cost](#cache-contents-and-setup-cost)
- [Implemented GPU path](#implemented-gpu-path)
- [Host preparation and compiled plans](#host-preparation-and-compiled-plans)
- [FRI memory placement](#fri-memory-placement)
- [GPU trace and witness design](#gpu-trace-and-witness-design)
- [Migration to a 2^27 trace ceiling](#migration-to-a-227-trace-ceiling)
- [Recursive circuit reductions](#recursive-circuit-reductions)
- [Alternative profiles and terminal backends](#alternative-profiles-and-terminal-backends)
- [CUDA scheduling and memory constraints](#cuda-scheduling-and-memory-constraints)
- [Sppark primitives to adopt](#sppark-primitives-to-adopt)
- [Implementation review, 2026-10-10](#implementation-review-2026-10-10)
- [Targeted validation](#targeted-validation)
- [Recovery and run commands](#recovery-and-run-commands)

## Goal and priorities

The [historical cached report](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/report.json)
records four RTX PRO 6000 Blackwell GPUs, 96 logical CPUs, and approximately
1 TiB of host RAM. SRS and fixed preprocessing are cached on disk; the
executables still start separately and rebuild frontend plans. Upstream root
aggregation and additional independent CPU verification are outside the
pipeline boundary. In-process verification is included.

| Work | Measured time | Engineering target |
| --- | ---: | ---: |
| FRI compression | 171.686 s | 60–70 s |
| First KZG staging + proving | 325.911 s | 100–110 s |
| Recursive KZG staging + proving | 342.361 s | 90–100 s |
| Complete pipeline | 839.988 s | 250–280 s |

The targets allocate a budget with 20–50 seconds of margin; they are not
predicted timings. About 540 seconds had to disappear from that critical
path. The two KZG prove phases total 326.188 seconds. Even eliminating them
would leave 513.770 seconds of FRI and staging, so work across the pipeline
was necessary to reach that target. The new retained-worker result above uses
a different preparation boundary and v4 profile; it is not a paired speedup
measurement against this table.

For a persistent service, report startup/key loading separately from warmed
request latency, and retain a separate one-shot measurement. Reusing a plan
or key still requires a fresh assignment and proof. A warmed service result
does not establish a cold-start bound.

| Priority | Change | First local experiment |
| --- | --- | --- |
| 1 | Retain compiled circuits, witness plans, layouts, loaded keys, and shared per-proof preparation | First-stage and recursive reuse are measured together; repeat with authenticated keys and extend compatible reuse to FRI |
| 2 | Fuse staging/proving and keep lookup, quotient, and opening work with device data | Feed one CPU-generated trace directly into commitment, then evaluate one resident quotient partition/coset; separately measure device opening folds and division |
| 3 | Address FRI memory pressure and distribute suitable work/data across GPUs | Profile one affected FRI commitment/opening operation and its host fallback paths |
| 4 | Improve bounded hint arithmetic and batch independent witness operations on CUDA | Compare representative hint batches and exact advice outputs, including failure cases |
| 5 | Reduce recursive commitments and arithmetic constraints | Count commitments, gates, lookups, and padded domains before generating a proof |

Do the circuit-size comparison early, before extensive recursive-specific
kernel work. The ordering favors measured removal of work and data movement;
local experiments determine implementation order within each area.

Several proposals target the same work. Fixed-evaluation caching, retaining
original trace evaluations, and device consumers can each avoid reconstruction
or transfers; their potential savings must not be added independently. Likewise,
subgroup-check removal, limb changes, and fewer commitments interact in the
recursive circuit. Recompute costs after each combined design change.

The cached path still includes these compilation/setup regions:

| Region | Measured time |
| --- | ---: |
| Stage-one scalar translation | 50.903 s |
| Stage-one lowering | 25.880 s |
| Recursive circuit construction | 27.256 s |
| FRI circuit construction, lowering, and setup | 29.913 s |
| Total of these regions | 133.952 s |

Sources: [stage-one staging](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/fri_stage.log),
[recursive staging](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/recursive_stage.log),
and [FRI compression](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/fri_compress.log).
These are regions to investigate, not guaranteed removable seconds. The
recursive prover reaches its first main commitment after roughly 57 seconds.
That interval includes 23.34 seconds loading the SRS, fixed coefficients and
metadata loading, and reading/decoding the first witness matrix in
[`outer.rs`](../experiments/kzg-wrap/src/outer.rs). It is not all fixed setup
that a key cache can eliminate. Reuse must respect profile and statement
identities, particularly the recursive builder's claim-dependent constants.

The two KZG proves download **822.604 GiB of forward-FFT evaluations**:
326.561 GiB in stage one and 496.043 GiB in recursion. A useful device path is:

```text
fresh assignment + reusable layout
    -> GPU trace columns
    -> interpolation and MSM
    -> resident lookup construction
    -> resident quotient evaluation
    -> commitment and openings
```

Generate selectors inside the quotient consumer, evaluate the constraint DAG,
and pass quotient values directly to commitment. A selector-only kernel that
downloads its output does not establish the value of the complete path.
Operation timers overlap: transfer, copy, CPU, and CUDA totals must not be
added into an elapsed-time saving.


## Measured results

The preserved 14m29s measurement below was made from base commit
`6d9d5ddad4fb9bfbabf8238ef87af6e9d7b3cab6`; its 31-file source snapshot is in
the recovery directory. The work is now on `sb/kzg` at `a904bce` in
`/home/sam/repos/multi-stark`. The first follow-up added MSM compute events,
NVTX ranges, explicit regenerated-fixture parity, and host timer summaries.
The subsequent partition pipeline overlaps stage-one host work with device
operations while preserving transcript order and proof construction.

### First-stage fused run with cold fixed preprocessing

One `stage-and-prove` invocation consumes the preserved compressed-FRI proof
without rerunning the compressor or recursive KZG stage. It uses a new empty
fixed cache tied to the measured executable, the existing `2^24` known-trapdoor
development SRS, four GPUs, and resident SRS caching. Distributed quotient and
lookup consumers are unset for these independently scheduled partitions.
The external process takes **279.87 seconds**, including current fixed
preprocessing, and peaks at **213.12 GiB host memory**. The internal stage
timer is 274.722 seconds; startup and shutdown account for the different
boundaries. GPU memory peaks range from 43.52 to 44.89 GiB.

The measured frontend preserves the archived gate, value, lookup and trace
layout. Hint calls fall from 40,938,272 to 28,875,135 while hint outputs remain
40,938,272. Its 93.145-second boundary compares with 128.456 seconds in the
archived run using the same circuit shape:

| Frontend region | Seconds |
| --- | ---: |
| Source circuit construction | 3.211 |
| Source assignment | 0.765 |
| Scalar translation | 43.976 |
| Scalar assignment | 28.412 |
| Lowering | 16.781 |

After lowering, the cache boundaries differ. This run measures 56.126 seconds
of fixed-matrix generation and staging intervals. Prover preparation reaches
all nineteen partitions at 108.271 seconds after its own start, including the
1.476-second SRS load. The fixed-preparation intervals total 98.629 seconds,
including 43.984 seconds loading fixed matrices and 30.027 seconds committing
them; the remainder includes metadata and checkpoint handling. Fresh main
trace intervals total 12.912 seconds and main commitments total 6.215 seconds.
Trace generation overlaps preparation, and load/commit timers are nested
inside preparation: these totals must not be added together or subtracted
from the run to predict a warmed latency.

**Proof computation takes 13.848 seconds.** Seventeen lookup and seventeen
quotient jobs use the resident GPU paths; two jobs in each category use the
CPU path. The 49,557-byte proof, 49,733-byte packet and 32-byte profile match
the archived stage-one SHA256 identities exactly. The executable rejects all
eighteen altered claim words and malformed proof encodings. Independent CPU
verification passes in 4.036 seconds outside the measured stage boundary.
Fused output contains no witness files or main-polynomial checkpoints.

The immutable binary, ninety-five source/native inputs, raw logs, all
per-partition timers, GPU samples and verified artifacts are in
[`first-stage-fused-20261009`](../experiments/kzg-cuda-validation/first-stage-fused-20261009/).
Postprocessing initially encountered two CUDA records interleaved with Rust
logging. The proof process had already succeeded; report recovery preserves
the original failure receipt and raw log, skips those two complete records
out of 1,921, and marks event totals incomplete. No proof was rerun. The
canonical parser now reports that condition explicitly; its five focused
tests pass, including compatibility with older callers.

### Resident quotient evaluation on admitted traces

The KZG backend now evaluates a complete quotient on one GPU when its input
columns and scratch fit. It uses sppark's in-place NTT for trace-sized cosets,
generates the unnormalized selectors on device, evaluates the constraint DAG
and grouped LogUp equations, interleaves the cosets, and performs the final
inverse NTT. Only the quotient coefficient slices return to the host; the
existing commitment and opening code consumes those same slices.

A single `2^24` trace-0 diagnostic used the preserved fixed/main coefficients
and circuit DAG, with synthetic lookup coefficients. Both paths received the
same inputs and produced exactly equal coefficients. Coefficients were already
resident on GPU 0 before the timed regions. This measures quotient computation
and final coefficient download, excluding commitments, lookup construction,
SRS loading, and witness generation.

| Measurement | GPU FFT + CPU sweep | Resident GPU quotient |
| --- | ---: | ---: |
| Wall time | 4.666797 s | 0.586064 s |
| Host-to-device bytes during quotient | 1 GiB | 5,320 bytes |
| Device-to-host bytes | 21 GiB | 1 GiB |
| Device-to-device bytes | 10 GiB | 20 GiB |
| CUDA kernel intervals | 246.067 ms | 299.595 ms |
| Host-copy intervals | 1,550.559 ms | 30.731 ms |

The observed local speedup is **7.96×**. The trace has widths 17 fixed, 3 main,
and 5 lookup columns, 20 nonconstant columns, 39 DAG nodes, and quotient degree
two. Preparation took 3.123 s; the complete diagnostic took 8.60 s. This is one
ordered sample, with the baseline first and initial BLS NTT table setup charged
to that baseline. The [report and raw logs](../experiments/kzg-cuda-validation/resident-quotient-20261009/report.json)
record the inputs, source digests, commands and comparison boundary.

Small CPU/device coefficient fixtures cover selectors, all graph operations,
next-row wrapping, grouped and empty lookups, constant columns, all-constant
input, and `2^16` resident copies/grid-stride execution. A fresh-process public
protocol proof remains 1,205 bytes with BLAKE3
`8164e8f1acee3f94dbb48d29908d4e99085af60f3a8a96f4d09b8c6405c69bca`,
matching the saved CPU fixture.

Admission includes evaluation columns, quotient output, selector, recycled
DAG scratch and metadata, plus 1 GiB headroom covering the static sppark NTT
tables (under 21 MiB). The NTT itself is in-place. The memory query excludes
live coefficient and pool allocations and credits unused pages reusable by
the current CUDA pool; a quarter of VRAM remains reserved. Admission is checked
again under the exclusive operation lease. Admitted partitions compute and commit concurrently, then
enter the transcript in input order. Rejected partitions use the serial
trace-sized host fallback, preserving its memory bound. Unexpected CUDA
runtime failures surface as errors. Set `MULTI_STARK_KZG_CUDA_QUOTIENT=cpu` to
select the previous path for comparison.

Wide `2^27` inputs can now use the opt-in distributed path below. Coefficients
spread across GPUs still cause uploads from host mirrors when a partition runs
on another card. These local results do not establish an integrated
five-minute proof.

### Distributed quotient at `2^27`

The opt-in four-device quotient path uses two runtime-discovered mutual peer
pairs, one coset per pair. Each card owns half the full-column NTTs and sweeps
half the coset's rows through bounded tiles. Global selector coordinates,
next-row halos and the final `row * 2 + coset` merge preserve the existing
equations. The final inverse transform remains sppark's stock `2^28` NTT.
The [implementation and validation report](../experiments/kzg-cuda-validation/distributed-quotient-20261009/report.json)
records every coefficient comparison, source/binary hashes and commands.

One target-size diagnostic used the actual saved outer circuit graph, widths
15 fixed, 3 main and 4 lookup, with **22 dense synthetic nonconstant columns**
at `2^27`. This is not a real wrapper witness. Both paths received identical
coefficients and challenges; all `2^28` output coefficients matched. The
baseline evaluates two trace-sized cosets with existing GPU FFTs and the
bounded CPU sweep. Both timings include all coefficient reuploads, canonical
interleaving, final inverse NTT and coefficient downloads. Input preparation,
initial NTT tables, coefficient retention, commitments and SRS work are excluded.

| Measurement | GPU FFT + bounded CPU sweep | Distributed GPU quotient |
| --- | ---: | ---: |
| Wall time | 32.834030 s | 6.066316 s |
| Host-to-device traffic | 120 GiB | 160 GiB + 18,608 bytes |
| Device-to-host traffic | 184 GiB | 12 GiB |
| CUDA pool used-memory high water, per card | 16.016 GiB | 58.381 GiB |
| Host RSS high water in the phase | 217.576 GiB | 105.670 GiB |

This is a **5.41× local speedup**, reducing this quotient by 26.768 s in one
ordered sample with the baseline first. Shared preparation took 2.856 s; the
complete diagnostic took 43.74 s. Host peak accounting was reset before each
phase, and the distributed phase retained the 8 GiB reference output solely
for comparison. Pool high-water counters include retained coefficients;
sampled total live device memory reached 59.053 GiB per card, including
runtime allocations outside the pool. Event intervals overlap across streams
and devices; the native `call_ms=0` field is unavailable, not zero elapsed time.

The 8 GiB coefficient cap per card retained 32 GiB before timing. Distribution
reuploaded **156 GiB of coefficients**, versus 112 GiB in the baseline, because
the two pairs each need a complete coset and some retained columns belong to
another owner. It also made 20 GiB of local coefficient copies and about
88 GiB each of local and peer tile copies, including halos. Final merging used
2 GiB of local copies, 2 GiB of peer copies, and 4 GiB through the pinned ring
(4 GiB D2H plus 4 GiB H2D). Final coefficient download was 8 GiB. Including
the extra coefficient uploads, total host traffic fell from 304 GiB to
172 GiB plus metadata. Better coefficient placement remains a possible
improvement; the reported speedup already includes the current placement cost.

Enable the path with `MULTI_STARK_KZG_CUDA_DISTRIBUTED_QUOTIENT=1` and set
`MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8` **before device initialization** in the
outer process. Mode `1` tries distribution when one-device admission fails;
`force` also exercises eligible small traces for diagnostics. Both modes
require quotient ratio two, natural trace domains, four compatible devices,
the retention cap and complete peak-memory admission. Unsupported inputs
retain the existing one-device or bounded CPU path. All selected devices are
reserved atomically; pool peer permissions and all budget checks precede
native job allocations. CUDA failures drain every selected device before
cleanup and surface as errors.

Focused fixtures passed at `2^10` and `2^16`, covering tile boundaries, final
wrap, constants, grouped lookups, actual async-pool peer buffers, forced
two-slot host staging, and rejection before and after group acquisition.
The existing single-device quotient regression also passed. A fresh proof
with distribution forced and the SRS cache enabled exercised two distributed
quotients and retained the saved **1,205-byte** proof digest
`8164e8f1acee3f94dbb48d29908d4e99085af60f3a8a96f4d09b8c6405c69bca`.
This establishes the local target-size quotient result, not an affected-stage
or end-to-end timing. The distributed LogUp result below covers the other
wide-input consumer separately.

### Resident lookup construction on admitted traces

The KZG backend now builds LogUp coefficient columns on one GPU when the input
evaluations and scan buffers fit. It evaluates the lookup-prefix DAG using
Boolean trace selectors, combines each group into a numerator and product,
and uses sppark's `Multiply` prefix scans, `batch_inversion`, and `Add` prefix
scan to construct the accumulator. A device transpose preserves the exact
row/group order before sppark inverse NTTs. Only the coefficient columns and
one scalar accumulator total return to the host. Zero messages reproduce the
CPU convention `inverse(0) = 0`; negative multiplicities remain field elements.

The trace-0 diagnostic used the actual preserved development-profile
`setup-0.bin`, `fixed-0.bin`, and `main-0.bin`, with `beta = 23` and `gamma = 31`
on both paths. Every coefficient and the final total matched exactly. The
trace has `2^24` rows, widths 17 fixed, 3 main, and 5 lookup columns, 10 lookup
messages grouped in pairs, and 25 lookup-prefix DAG nodes. Fifteen nonconstant
input columns were resident on GPU 0 before timing; BLS NTT initialization was
warmed outside both regions.

| Measurement | GPU FFT + CPU lookup + GPU IFFT | Resident GPU lookup |
| --- | ---: | ---: |
| Wall time | 3.178567 s | 0.735625 s |
| Host-to-device bytes during lookup | 2.5 GiB | 3,812 bytes |
| Device-to-host bytes | 10 GiB | 2.5 GiB + 32 bytes |
| Device-to-device bytes | 7.5 GiB | 7.5 GiB |
| CUDA kernel intervals | 82.865 ms | 145.697 ms |
| Host-copy intervals | 707.871 ms | 73.993 ms |

The local speedup is **4.32×**, including final coefficient downloads and the
baseline's existing transpose/interpolation path. Preparation took 2.949 s;
the complete diagnostic took 7.27 s. This is one ordered sample with the
baseline first, excluding commitments, SRS loading, main-witness generation,
and checkpoint reads. The [report, raw logs, and source snapshot](../experiments/kzg-cuda-validation/resident-lookup-20261009/report.json)
record that boundary and the missing measured-binary digest; a later rebuild's
digest is not assigned to this run.

Small fixtures compare every coefficient and total against the portable CPU
implementation for zero/empty messages, zero and nonzero challenges, negative
multiplicities, selectors, next-row wrap, partial lookup groups, constant
inputs, and `2^16` scans. They passed in 1.64 s, and the shared quotient
regression passed in 1.71 s. A fresh-process proof across all four GPUs with
both resident consumers enabled still verifies and matches the saved CPU
fixture: 1,205 bytes and BLAKE3
`8164e8f1acee3f94dbb48d29908d4e99085af60f3a8a96f4d09b8c6405c69bca`.

Admission includes evaluated input columns, four `rows × lookup_groups`
scalar buffers, recycled DAG scratch, metadata, and 1 GiB headroom. SRS/cache
workspace is evicted if necessary before the final under-lease memory check;
live coefficients and allocator pages remain charged, while reusable unused
pool pages count toward available memory. Unsupported
shapes or insufficient capacity use the serial bounded host path. Set
`MULTI_STARK_KZG_CUDA_LOOKUP=cpu` for an explicit comparison. Unexpected CUDA
runtime failures surface as errors. The opt-in tiled implementation below
also covers wide `2^27` inputs; neither local result is an affected-stage or
end-to-end timing.

### Distributed lookup at `2^27`

The opt-in distributed LogUp builder reuses the quotient path's atomic leases,
pool peer permissions and bounded transfer rings. It assigns consecutive
global row quarters to four devices. After evaluating grouped numerators and
denominators through bounded tiles, it frees the input evaluations before
running stock sppark product scans, inversion and additive scans. Ordered
quarter totals restore the global exclusive accumulator. Output columns are
gathered in batches of at most four, then interpolated with stock full-length
inverse NTTs. The proof layout and cross-circuit accumulator order are unchanged.

The [target-size report](../experiments/kzg-cuda-validation/distributed-lookup-20261009/report.json)
uses the actual saved outer graph with **18 full-length synthetic nonconstant
input columns**, eight lookup messages in four groups, and `2^27` rows.
Every one of the `4 × 2^27` output coefficients and the final accumulator total
matches the bounded CPU builder. This is not a valid wrapper witness. A mixed
row-to-palette index avoids short-period coefficient repetition in the
original-domain FFTs. Both paths include input reuploads, trace construction,
transpose/interpolation and output downloads; shared input preparation,
coefficient retention, NTT initialization, commitments and SRS work are excluded.

| Measurement | GPU FFT + bounded CPU lookup + GPU IFFT | Distributed GPU lookup |
| --- | ---: | ---: |
| Wall time | 19.123629 s | 6.875674 s |
| Host-to-device traffic | 56 GiB | 148 GiB + 13,088 bytes |
| Device-to-host traffic | 88 GiB | 24 GiB + 288 bytes |
| CUDA pool used-memory high water, per card | 16.016 GiB | 52.315 GiB |
| Host RSS high water in the phase | 193.434 GiB | 105.647 GiB |

The local speedup is **2.78×**, a 12.248 s reduction in one ordered sample
with the baseline first. Preparation took 2.662 s; the complete process took
31.61 s. The distributed phase keeps 16 GiB of CPU reference coefficients
solely for comparison. Pool high-water counters include retained coefficients.
Sampled live device memory, including runtime allocations outside the pool,
reached 52.987 GiB per card; this is distinct from continuous high-water
telemetry. Native event intervals overlap, and `call_ms=0` is unavailable.

This speedup occurs despite **28 GiB more host traffic**. Preparation retained
32 GiB, two 4 GiB columns per GPU. Retention assigns devices by current retained
bytes while processing columns in parallel; the distributed consumer assigns
evaluation owners by column slot within each pair. Only one retained column
matched its evaluation owner in this sample. The baseline follows all retained
owners and uploads 40 GiB of coefficients. Distribution uploaded 140 GiB for
two complete 72 GiB input copies, reusing only 4 GiB locally. It additionally
staged 8 GiB of output between pairs, with both DMA legs counted, and downloaded
16 GiB of final coefficients. The complete host traffic is 144 GiB versus
172 GiB plus metadata and scalar totals. The improvement removes CPU lookup
work; it does not establish a transfer reduction. Coefficient lengths are
unchanged, and all 32 GiB of retained data stayed allocated. The measured
coefficient-reuse follow-up below addresses this placement mismatch.

Both distributed consumers now copy immutable resident coefficients from the
other GPU in their acquired pair into the existing evaluation buffer before
the NTT. Same-device copies retain priority; other resident locations use
the host mirror. Pool peer rights, stream ordering and the all-device drain
also cover these reads. Forced host staging disables this branch.

The [coefficient-reuse report](../experiments/kzg-cuda-validation/distributed-coefficient-peers-20261009/report.json)
repeats the same lookup-only comparison with unchanged synthetic inputs:

| Measurement | Initial distributed sample | With coefficient peer reads |
| --- | ---: | ---: |
| Distributed lookup wall time | 6.875674 s | 6.680026 s |
| Coefficient uploads | 140 GiB | 112 GiB |
| Same-device coefficient copies | 4 GiB | 16 GiB |
| Same-pair coefficient copies | 0 GiB | 16 GiB |
| Total host traffic | 172 GiB + 13,376 bytes | 144 GiB + 13,376 bytes |

Every output coefficient and the total still match. The fresh bounded CPU
reference took 19.231905 s, giving a 2.88× local speedup. Pool high water
remained 52.315 GiB per card, and sampled live memory remained 52.987 GiB.
The 0.196 s reduction from the earlier distributed sample is **2.85% in one
sample**. Parallel retention changed local placement between samples: the
28 GiB upload reduction comprises 16 GiB of explicit peer reads and 12 GiB
of additional local matches. The new path reuses all 32 GiB of retained
inputs regardless of which partner owns them. Per-device coefficient uploads
still span 20–36 GiB, so fewer total bytes need not reduce the slowest lane's
time proportionally. This does not establish a fixed-placement timing gain
or a new target-size quotient timing.

A deliberate partner-owned fixture uses 65,549 coefficients padded to
`2^17`, reversed device order and actual async-pool buffers. Exact coefficient
and total parity pass; counters show 2,097,568 peer bytes plus the same number
of uploaded bytes. Forced host staging changes those to 4,195,136 uploaded
bytes and zero peer bytes. All leases and retained allocations are released.
Distributed quotient, lookup, cross-circuit accumulator and fresh proof
regressions also pass after the shared loader change.

Small fixtures passed with reversed GPU order, one through five lookup groups,
partial groups, zero challenges/messages, negative multiplicities, constants,
global selectors and `Next`, forced host staging, and both admission rejection
boundaries. An unaligned used-row boundary and explicitly nonzero unequal
quarter totals exercise offset reconstruction. A 1,573-byte cross-circuit
proof is byte-identical to its reference; the accelerated API also preserves
a nonzero initial accumulator. With both distributed consumers forced and
the SRS cache enabled, the saved public proof remains **1,205 bytes** with
BLAKE3 `8164e8f1acee3f94dbb48d29908d4e99085af60f3a8a96f4d09b8c6405c69bca`.
Existing single-device lookup/quotient and distributed quotient regressions pass.

Set `MULTI_STARK_KZG_CUDA_DISTRIBUTED_LOOKUP=1` and initialize the outer process
with `MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8`; the benchmark harness's
`--distributed-wrapper` option enables both distributed consumers for that
process. Unsupported shapes, topology or memory retain the existing fallback.
This local result does not establish a complete-stage or five-minute proof.

### Retained first-stage compilation and proving keys

The first KZG executable now supports `stage-and-prove-many`. It processes
compatible requests sequentially in one process, retaining the Goldilocks
verifier circuit, scalar input mapping, lowered layout, KZG system, loaded
SRS, fixed coefficients, and proof codec. Each request generates and checks
fresh assignments and main commitments. See the [worker usage](#make-a-cached-run-reuse-compilation-too).

The [isolated comparison](../experiments/kzg-cuda-validation/retained-worker-20261009/report.json)
uses a 27-circuit fixture with maximum height `2^17`. Two paired comparisons
alternate execution order and use distinct valid inner proofs, changing a
private bit and then returning to the original input. Mean request time is
**11.425 s rebuilding versus 10.317 s retaining live state**, a **9.7%
reduction**. Rebuilding includes compilation and disk fixed-cache restoration;
both paths include fresh witness generation, proving, verification and output
writes. The initial frontend compilation took 0.507 s and the initial request,
including fixed-cache population, took 14.247 s; both are outside those means.
Native inner-proof generation and verification are outside the comparison
timers. The CLI additionally loads and verifies every saved input.

Proof, packet and profile bytes match for both pairs. Changing the private
input changes the proof; returning to the original input reproduces the
original artifacts. Warm requests write no fixed matrices, fixed-coefficient
checkpoints, witness files or main checkpoints. Their output still supports
standalone verification. Incompatible verifier-plan identities are rejected.
The existing checkpointed/fused/resume checks also pass and reproduce the
previously archived fixture's artifact digests. Four focused layout tests pass.

The worker comparison took 64.25 s and used an isolated temporary fixed cache
on tmpfs. It does not establish production disk savings, production peak
memory, or a full-pipeline speedup. The latest complete-chain measurement
remains **839.988 s**. Reproduce the comparison with the ignored
`worker::tests::retained_worker_benchmark` example test and an isolated
`MULTI_STARK_KZG_FIXED_CACHE`; exact commands and boundaries are in the report.

### Fused first-stage trace generation and proving

The first KZG executable now supports `stage-and-prove`, feeding generated
trace matrices directly into commitment with bounded lookahead. It skips
witness-file compression and decoding, and main-polynomial checkpoint writes.
The checkpointed commands remain available for intermediate resume.

The [isolated comparison](../experiments/kzg-cuda-validation/fused-stage-20261009/report.json)
uses a 12-circuit fixture with maximum height `2^16`, compact hashing, and
cached fixed preprocessing. Three comparisons alternate execution order and
use fresh assignments with different private bytes. The mean elapsed time
from assignment generation through proving and verification is **3.188 s
checkpointed versus 3.097 s fused**, a **2.8% reduction**. Each fused run
avoids approximately **8.45 MiB** of compressed witness and main-checkpoint
files. The fixture writes to tmpfs and is much smaller and more compressible
than production traces; it does not establish production NVMe or full-pipeline
savings. Compilation, lowering and fixed-cache restoration are outside these
timers; SRS initialization, fixed-coefficient loading and the example's
verification checks are included in both paths.

Proofs, packets and profile IDs are byte-identical in every comparison.
The same check exercises checkpointed resume, independent verification,
fresh witnesses in a directory containing older main checkpoints, and
rejection of mismatched trace dimensions. It also asserts that fused outputs
contain neither witness files nor main checkpoints. The complete check took
38.15 seconds; no production chain or production cache regeneration ran.

Reproduce it with the ignored `tests::fused_staging_benchmark` test in the
`init_fri_kzg_prove` example. See the
[usage and memory constraints](#remove-the-compulsory-file-boundary-before-requiring-gpu-witnesses)
below. First-stage plans and loaded keys can now be
[retained across requests](#retained-first-stage-compilation-and-proving-keys);
recursive-stage fusion is also implemented and covered by the targeted
correctness check below. The timings in this subsection cover stage one only.

### Shared staging preparation and byte expansion

The [isolated staging comparison](../experiments/kzg-cuda-validation/shared-staging-20261009/report.json)
generates 1,024 independent compact BLAKE3 calls with 64-byte messages over
the KZG scalar field. Each iteration uses a fresh assignment. Timings below
average iterations 1–3 after an initial iteration and sum assignment binding
and individual trace-generation timers. Circuit construction, lowering,
initial witness generation/validation, trace serialization, and file I/O are
outside these timers. No full pipeline or SRS generation was run.

| Operation | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Assignment binding and all 13 trace matrices | 485.597 ms | 84.371 ms | 82.6% |
| Nine compact hash traces, included above | 426.871 ms | 58.262 ms | 86.4% |
| Duplicate assignment validation | 32.372 ms | Constant-time owner check | — |

Sharing the compact words/table counts and removing duplicate validation
alone brought total trace generation to 429.954 ms, an 11.5% reduction.
Most of the additional improvement came from reusing the 256 byte-to-field
embeddings during matrix expansion, avoiding repeated Montgomery conversions.
These are nested measurements; their reductions must not be added together.

All four before/after serialized trace hashes match, covering 12,878,880 field
cells per assignment. The focused frontend suite passes 40 tests, including
concurrent and out-of-order trace requests, fresh assignments, foreign-owner
rejection, block/tree boundaries, and proof rejection for altered hash traces.
The fixture measures host trace generation; the complete pipeline remains
unmeasured since the 839.988-second baseline. This result does not establish
an 82.6% reduction in complete staging or end-to-end latency.

Reproduce this narrow check with the ignored
`plonkish::tests::compact_hash::compact_hash_staging_benchmark` library test
under `--release --features parallel,kzg-cuda,cuda -- --exact --ignored --nocapture`.
It uses CPU staging even in the CUDA-linked test binary. Raw logs, source
snapshots, binary hashes, and exact commands are in the comparison directory.

### Sppark division and point uploads

The [isolated comparison](../experiments/kzg-cuda-validation/sppark-primitives-20261009/report.json)
uses 2^24 coefficients and identical synthetic inputs on the four Blackwell
GPUs. These are operation timings, not a new full-pipeline measurement.
Each case ran three iterations; the table averages iterations 1–2 after
initialization. Opening timings include division and MSM, excluding polynomial
folding and evaluation. The point fixture repeats 256 generator multiples;
it does not load or generate the production SRS.

| Operation | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Division, including host transfers | 49.871 ms | 44.693 ms | 10.4% |
| Division + MSM on four GPUs, with direct affine uploads | 165.379 ms | 131.265 ms | 20.6% |
| One-GPU division + MSM, retaining the quotient | 224.755 ms | 184.508 ms | 17.9% |

The division comparison starts from the custom tiled implementation. The
four-GPU comparison already uses sppark division in both binaries and measures
the affine-import and host-buffer changes. The one-GPU comparison uses the
same final binary for both paths. These reductions must not be added together.

The changes are:

- `div_by_x_minus_z<true>` replaces the custom division kernels and CPU carry
  scan. The quotient comes first and the remainder is discarded. The host
  path borrows its input and first-touches a separate output in parallel.
- MSM uploads borrow the layout-checked 104-byte arkworks affine storage.
  A small device conversion produces sppark's 96-byte coordinates and maps
  the infinity flag to zero coordinates. This eliminates CPU repacking and
  its large temporary vector. SRS traffic grows by 8.3%; the four-GPU
  comparison's process peak host RSS falls from 5.049 to 3.554 GiB.
  GPU staging uses an additional 104 bytes per chunk point during import,
  freed before MSM scratch allocation; it fits the existing admission bound.
- With one selected KZG device, the quotient stays on that device through
  chunked MSM. A 128-scalar sample preserves normalization of structured
  inputs. Admission includes the quotient, MSM workspace, and transient
  reserve; insufficient space falls back to the host quotient path. Multiple
  selected devices keep the partitioned MSM path.

The pinned division launcher needs two guards: reserve at least one warp of
shared field elements, and choose exact cooperative tiles (or one block) to
avoid the partial-tail carry race. Host launches are serialized around its
mutable cached block size; device execution can still overlap across cards.
The upstream kernel and dependency checkout are unchanged. These guards are
integration debt to revisit with a corrected upstream launcher.

A per-device resident-fold prototype passed correctness checks but increased
the 16-column opening fixture from 324.213 to 363.856 ms, about 12.2%.
It repeated a full quotient MSM and SRS upload on every card. That path is
excluded from the implementation; its source and measurements are archived.
Revisit distributed folding together with quotient reduction/data placement
and SRS reuse, rather than independently multiplying MSM work.

Validation: 53 CUDA-enabled adapter tests and 39 CPU-only adapter tests pass.
Serialized opening proofs match an independent CPU calculation, including
repeated points and constant columns; the production one-GPU dispatch also
passes that fixture. CUDA memcheck reports zero errors for division boundaries
and affine uploads with exceptional points. The tests cover chunk offsets,
structured scalars, and memory admission. GPU peak memory was not sampled in
these microbenchmarks. No full-chain run or fixed-cache regeneration was used;
839.988 s remains the last measured complete pipeline.

Reproduce only the affected operations:

```bash
MULTI_STARK_CUDA_ARCHS=120 cargo test --release --locked \
  --features parallel,kzg-cuda,cuda --lib ark_adapter::
MULTI_STARK_CUDA_ARCHS=120 MULTI_STARK_KZG_CUDA_PROFILE=1 \
  MULTI_STARK_KZG_BENCH_LOG_N=24 cargo test --release --locked \
  --features parallel,kzg-cuda,cuda --lib cuda_division_benchmark \
  -- --ignored --nocapture
MULTI_STARK_CUDA_ARCHS=120 MULTI_STARK_KZG_CUDA_PROFILE=1 \
  MULTI_STARK_KZG_BENCH_LOG_N=24 cargo test --release --locked \
  --features parallel,kzg-cuda,cuda --lib cuda_opening_benchmark \
  -- --ignored --nocapture
```

Set `MULTI_STARK_KZG_CUDA_DEVICES=0` for the one-device comparison. Saved
sources, binary hashes, raw timings, and validation logs are in the linked
report directory. The opening benchmark compares both paths explicitly;
production selects the resident quotient only when one device is configured.

### Bounded partition pipeline

The [cached full-chain run](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/report.json)
took **839.988 s (14m00s)**, or **868.525 s (14m29s)** including independent
CPU checks. Against the instrumented baseline, stage-one proving fell
**20.34%**, saving 31.657 s; the complete boundary fell **3.65%**, saving
31.847 s. This is one cached sample per implementation on identical hardware,
CPU affinity, root inputs and compressed-FRI inputs. The unchanged phases
varied by a net 0.190 s.

| Phase | Instrumented baseline | Partition pipeline |
| --- | ---: | ---: |
| FRI compression | 175.189 s | 171.686 s |
| FRI → KZG staging | 198.688 s | 201.942 s |
| FRI → KZG proving | 155.627 s | 123.970 s |
| Recursive KZG staging | 140.543 s | 140.143 s |
| Recursive KZG proving | 201.759 s | 202.218 s |

Stage-one preparation fell from 53.226 s to 41.116 s. Peak host RAM during
that prove increased from 190.9 GiB to 218.8 GiB; recursive proving remained
at 464.2 GiB. Both proofs, packets and profile identifiers are byte-identical
to the preserved fixture. Independent CPU verification, external pairings
and tampering checks passed in both full runs. All four GPUs were idle after
completion.

The stage-one example prefetches one witness while committing its predecessor.
Lookup and full-domain quotient processing overlap one CPU sweep with the
preceding commitment and next evaluation reconstruction. Those two device
operations share one branch, avoiding nested Rayon waits for the same device
lease. Row-level parallelism keeps its default width. Lookup local totals are
folded in circuit order, and the quotient challenge still follows all lookup
observations. Recursive trace-sized quotient cosets retain their existing
schedule.

`KzgConfig::with_partition_pipeline(bytes)` enables evaluation overlap; its
library default is disabled. The stage-one example and diagnostic default to
`MULTI_STARK_KZG_PREFETCH_GIB=32`; set it to `0` for a same-binary comparison.
The limit bounds the extra prepared payload, not total host memory, retained
coefficients, compute scratch or commitment buffers. Oversized evaluation
partitions drain the pipeline.

The [four-partition diagnostic](../experiments/kzg-cuda-validation/partition-pipeline-20261009/diagnostic/diagnostic-summary.json)
used the same executable with prefetch disabled and with a 32 GiB limit:

| Measurement | Disabled | 32 GiB lookahead |
| --- | ---: | ---: |
| Lookup commitments | 12.774 s | 11.108 s |
| Quotient commitments | 17.668 s | 14.496 s |
| Complete diagnostic | 73.82 s | 69.99 s |
| Peak host RAM | 120.5 GiB | 148.8 GiB |

Lookup plus quotient time fell 15.9% in this diagnostic; it does not measure
the complete proof chain. Nsight confirms GPU kernels or transfers overlapped
1.734 s of lookup CPU work and 1.654 s of quotient CPU work. The raw timeline
is at `target/kzg-partition-pipeline-20261009/pipeline.nsys-rep`; summary tables
and the command logs are preserved beside the diagnostic report.

The [fresh-fixed full-chain run](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cold/report.json)
took 1166.561 s (19m27s), or 1195.669 s (19m56s) including independent CPU
checks. It reused the populated development-SRS cache and regenerated fixed
preprocessing for the exact new executables. Both stages' proofs, packets
and profile identifiers match the preserved fixture byte for byte; independent
CPU verification, external pairings and tampering checks passed.

Validation covers 51 focused GPU adapter tests (50 in the suite plus one
separate CUDA-size parity test, with three scale tests ignored), 39 CPU-only
adapter tests without the parallel feature, two codec/prove-resume fixture
tests, and three benchmark-harness tests. The new cases cover unequal circuit
heights, nonzero cross-circuit lookup totals, inactive circuits, oversized
payloads, ordered consumption and cleanup after each pipeline stage panics.

The [partition-pipeline summary](../experiments/kzg-cuda-validation/partition-pipeline-20261009/summary.json)
preserves the comparisons, reports, exact executables, source snapshots,
compact proofs, test logs and Nsight tables, with a verified file-hash
manifest. Full staged traces and fixed preprocessing remain on NVMe under
`/opt/dlami/nvme/multi-stark-kzg-prefetch-20261009/`.

### Instrumented baseline

The [instrumented cold run](../experiments/kzg-cuda-validation/msm-profile-20261009/cold/report.json)
repopulated the wiped NVMe caches and verified both proofs byte for byte
against the recovery fixture. It took 1291.405 s (21m31s), or 1319.513 s
including independent CPU checks, with fresh SRS and fixed preprocessing.
The [Nsight diagnostic](../experiments/kzg-cuda-validation/msm-profile-20261009/nsight/summary.json)
measured 2.211365 s of MSM kernels versus 2.214828 s from the new event
intervals, a 0.16% difference. These are overlapping totals across four GPUs.
Its timed application phases total 27.490 s; profiler collection and export
bring the command to 40.19 s. The raw timeline is retained on the root volume
at `target/kzg-nsys-20261009/one-trace.nsys-rep`.

The [instrumented cached run](../experiments/kzg-cuda-validation/msm-profile-20261009/cached/report.json)
took **871.835 s (14m32s)**, or **900.567 s (15m01s)** including independent
CPU checks. This single sample is 0.27% above the preserved 14m29s run;
it establishes comparable instrumentation, not a performance improvement.
All six proof/packet/profile artifacts match in both runs. The 47 focused
adapter tests and three benchmark-harness tests passed, as did both manifests'
format checks. All GPUs were idle after completion.

| Phase | Instrumented cached run | Preserved cached run |
| --- | ---: | ---: |
| FRI compression | 175.189 s | 171.880 s |
| FRI → KZG staging | 198.688 s | 199.039 s |
| FRI → KZG proving | 155.627 s | 156.030 s |
| Recursive KZG staging | 140.543 s | 138.489 s |
| Recursive KZG proving | 201.759 s | 204.015 s |

MSM GPU intervals total 24.518 s in stage one and 18.390 s in the recursive
prove, overlapping across streams/devices. Recursive forward FFTs still
download 496.043 GiB, with 63.757 s of aggregate host-copy intervals; the SRS
still uploads 154 times for 216.082 GiB. The CPU lookup, selector and constraint
timers total 50.584 s in stage one and 60.303 s in recursion. These measurements
support work on host/device data flow before assuming kernel execution or
stream count is the pipeline limit.

The [measurement summary](../experiments/kzg-cuda-validation/msm-profile-20261009/summary.json)
links both runs and their constraints. That directory also preserves exact
instrumented executables, source snapshots, logs, proof artifacts, Nsight
summary tables and a file-hash manifest. The large staged traces and caches
remain under `/opt/dlami/nvme/multi-stark-kzg-perf-20261009/`.

### Preserved 14m29s result and scope

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

- [Preserved 14m29s results](../experiments/kzg-cuda-gpu-feeding-results.json).
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

### Cache contents and setup cost

The populated disk caches avoid parameter generation and fixed-polynomial
preprocessing. They do not contain reusable witnesses or eliminate frontend
compilation. The first-stage worker additionally retains live objects:

| State | Contents | Reuse boundary |
| --- | --- | --- |
| Development SRS cache | Powers and verifier parameters for the development setup | Format, curve, seed, and exact degree range; independent of the witness |
| Fixed preprocessing cache | Fixed polynomial coefficients, commitments, and circuit metadata | Exact executable and circuit/profile identity; the recursive entry currently also binds the public statement |
| Resident device coefficients | Immutable coefficients retained by the running prover | Reused by operations within that process; witness-dependent columns belong to one proof |
| First-stage compiled worker state | Circuits, witness plans, lowered layouts, loaded SRS, fixed coefficients, KZG system and proof codec | One compatible profile per process; assignments and proof outputs remain fresh |

The [cache implementation](../examples/support/kzg_fixed_cache.rs) binds the
executable digest, while the stage-one profile additionally binds the FRI
verifier plan. Rebuilding or changing a bound profile requires a new fixed
entry. The recursive statement restriction remains until its constant gates
and cache identity are corrected together.

The original 14m29s cached and 20m01s fresh-fixed runs differ by **331.486 s
(5m31s)**. The latest partition-pipeline pair differs by **326.573 s (5m27s)**.
These differences include run variation and work across staging/proving;
they are not separately timed preprocessing jobs or extra time to add to the
fresh-fixed totals. Both pairs reused an already populated SRS cache.

[Development SRS generation and publication](../experiments/kzg-wrap/README.md)
took **7.32 s at `2^24`** and **119.34 s at `2^28`** in the preserved
measurements. Those costs are outside the fresh-fixed totals above. Reloading
the saved parameters still took about **1.50 s** and **23.33 s**, respectively;
a live worker can reuse the loaded objects. These are known-trapdoor
development parameters, not a production ceremony import.

### Staging work and timing scope

Witness generation computes logical wire values. Trace generation places
those values into padded rows and columns and constructs auxiliary traces.
They have different dependencies and different opportunities for parallelism.

| Operation | Existing KZG path | Proposed GPU work |
| --- | --- | --- |
| Logical assignment | `Witness::generate` evaluates recipes sequentially, invokes Rust hints, and checks the assignment | Batched arithmetic and typed hint kernels following a compiled dependency schedule |
| Computation trace | `main_trace` gathers assignment values according to the lowered layout | Gather directly into device columns |
| Auxiliary traces | Host construction of lookup multiplicities and compact BLAKE3 traces | Specialized counting and hash trace kernels |
| Main commitment | Host matrix decoding and transpose, followed by GPU iFFT and MSM | Interpolate and commit generated device columns |

The relevant implementations are [the witness evaluator](../src/plonkish/witness.rs),
[trace lowering and materialization](../src/plonkish/stark.rs),
[compact hash traces](../src/plonkish/hash.rs), and
[KzgPcs::interpolate_columns](../src/ark_adapter/pcs.rs).

The latest run's two KZG staging phases took 342.085 seconds, about 41% of
the pipeline, with almost no GPU activity. The following segments identify
potential targets within that staging time:

| Segment | Recorded time | Timing scope |
| --- | ---: | --- |
| First stage scalar translation | 50.90 s | Difference between cumulative source-witness and scalar-translation timers |
| First stage scalar witness | 46.87 s | Assignment generation, checking, and associated host work between cumulative timers |
| First stage lowering | 25.88 s | Difference between scalar-witness and lowered-circuit timers |
| Recursive circuit construction | 27.26 s | Circuit build timer |
| Recursive witness | 52.69 s | Witness generation/checking and associated output checks |

Sources: [cached pipeline report](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/report.json),
[first stage log](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/fri_stage.log),
and [recursive staging log](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/recursive_stage.log).
The witness segments are not isolated hint-kernel measurements. Eliminating
their entire cost would account for about 100 seconds; trace generation and
I/O offer additional opportunities, while circuit construction and lowering
need their own reuse strategy.

Use the [instrumented baseline](#instrumented-baseline) and
[partition comparison](#bounded-partition-pipeline) to distinguish kernel
execution, host work, transfers, and waiting. Aggregate operation intervals
overlap and cannot be added as elapsed pipeline time.


## Implemented GPU path

- Immutable polynomial coefficients survive interpolation on their assigned
  device and feed subsequent MSMs, coset FFTs and opening evaluations. Host
  mirrors remain for checkpoints and spill recovery; constants need no device
  allocation. The default resident budget is capped at half of each device's
  VRAM; the distributed quotient requires a pre-initialization cap of 8 GiB.
- Each GPU has one exclusive operation lease. An FFT batch uses up to two
  stream/work slots, overlapping columns and reusing its scratch allocation
  within that batch. Other operations on that GPU still wait for the lease.
- Each lane has persistent pinned upload/download rings, four 16 MiB chunks
  per direction. Two lanes use 256 MiB pinned memory per device. Events protect
  reuse; host copies run in parallel. These are bounded staging rings, not
  whole-column pinned storage.
- Scalars borrow arkworks Montgomery limbs directly with compile-time layout
  checks. Canonical decoding and host first-touch allocation are parallel.
  Affine uploads borrow arkworks storage and convert on the GPU, including
  infinity handling; frequent nonunit scalars are normalized to avoid
  degenerate Pippenger buckets.
- An MSM point chunk is reused for the columns in its current local batch.
  `MULTI_STARK_KZG_CUDA_SRS_CACHE=1` additionally retains immutable SRS ranges
  and a reusable sppark MSM workspace across calls. The opt-in cache has a
  12 GiB point limit per device and shares admission with other operations;
  stale owners and least-recently-used ranges can be evicted. Memory admission
  includes the current CUDA pool's unused reusable pages while preserving
  the quarter-VRAM reserve. Sppark allocation uses
  `cudaMallocAsync`/`cudaFreeAsync`; the KZG adapter does not set a persistent
  pool release threshold. See the [cache validation report](../experiments/kzg-cuda-validation/resident-srs-20261009/report.json).
- 2^29 MSMs are chunked. Whole-device FFTs support 2^29 and enforce log size
  at most 31, despite compiling with `MAX_LG_DOMAIN_SIZE=32`.
- Lookup construction uses device DAG evaluation, sppark scans/inversion,
  transpose and IFFT when its evaluated columns and scan buffers fit.
  The opt-in distributed lookup extends this to four global row quarters,
  with ordered offsets and bounded gathering of full coefficient columns.
  Quotient selectors, constraint and LogUp sweeps run on device when a complete
  trace-sized coset fits. An opt-in four-device quotient distributes wider
  ratio-two cosets through bounded row tiles; rejected partitions keep the
  bounded CPU path.
  Opening-polynomial folding remains on the CPU.
  Division uses sppark's in-place kernel with guarded launch geometry;
  one-device openings retain its quotient through MSM. Multi-device openings
  retain partitioned MSMs. The pinned sppark multi-point evaluation
  kernel has a recorded shared-memory race; the adapter instead launches
  single-point kernels against one retained coefficient buffer.
- Stage-one witness loading and evaluation reconstruction use bounded
  lookahead. CPU partition sweeps overlap neighboring device operations;
  commitments and lookup accumulator totals remain in transcript order.

The preserved 14m29s run retained about 160 GiB of coefficients; peak sampled KZG
device memory was 56.7 GiB per card. Four large columns spilled under the
half-VRAM limit, causing 32 GiB of uploads in recursive FFTs and another
32 GiB in evaluations. Stage-one forward FFTs and evaluations uploaded no
coefficients. Mean sampled utilization during KZG proving remained about
6–9% per GPU. Staging, CPU consumers, host copies and repeated SRS transfers
still leave idle periods. Two-second samples miss short kernels; CUDA-event
totals overlap across devices and cannot be summed into wall time. MSM events
in that archived run record transfers only. The current profiler also records
`msm-compute`: GPU intervals exclude explicit result downloads and host bucket
reduction, while `call_ms` covers the synchronous invocation. Intervals can
overlap within one invocation. Division's compute interval includes its CPU
carry pass in the archived runs; current division has no CPU carry pass.
The old 23m10s report's constraint timer preceded the loop;
use the corrected diagnostics and latest run for constraint attribution.


## Host preparation and compiled plans

These changes develop the priorities above. Shared immutable state and
bounded caches can remove repeated work before a general GPU witness compiler
is available. Their benefits need isolated measurements and joint host/device
memory accounting.

### Remove repeated work within a proof

Implemented: `Witness::generate` still checks all gate, lookup and hash
relations before constructing an `Assignment`. Its owner, values and publics
are private, with read-only accessors and no alternative constructor or
deserialization path. `MultiStarkCircuit::trace_shards` checks the exact circuit
owner without repeating validation. The proof constraints and generation
errors are unchanged.

The assignment lazily retains a thread-safe `OnceLock` containing the compact
`u32` word vector and all three table histograms, counted together. Individual
traces, whole auxiliary shards, and eager trace generation share that data;
each expanded field matrix is still generated on demand. The cache is local
to one assignment and is freed with it. It retains four bytes per compact word
plus 518 KiB of histogram counters on a 64-bit host, excluding allocator and
container overhead. It does not retain input-message copies or expanded traces.
Circuits that never request a hash trace do not allocate this cache.

Each expanded hash matrix reuses a 256-entry table of field-encoded bytes for
word bytes, carries, XOR outputs, and split limbs. This avoids repeated scalar
Montgomery conversion while preserving every field value. See the
[isolated measurements](#shared-staging-preparation-and-byte-expansion) above;
production staging also includes compilation, witness generation and writing.

### Parallel witness checks and grouped range hints

Implemented in [`builder.rs`](../src/plonkish/builder.rs): independent gate
checks run in parallel, and lookup checks reuse a row buffer within each
worker task. Both return the first failing constraint in the original order.
Hash errors still take precedence over gate errors, which precede lookup
errors. The serial feature configuration retains the same behavior.

[`foreign.rs`](../src/plonkish/foreign.rs) now extracts all 16-bit limbs for
one Goldilocks range check in a single hint. Lookup registration does not
allocate values, so grouping these outputs preserves wire indices, gate and
lookup order, and every assignment value. The range relations and their
existing rounded bit bounds are unchanged. Recipe evaluation and hash,
gate and lookup validation now have separate elapsed-time fields in the
witness logs, allowing the remaining serial work to be measured directly.

The [isolated CPU report](../experiments/kzg-cuda-validation/witness-validation-20261009/report.json)
records five alternating-order pairs after warmup, with all 96 available
logical CPUs and default Rayon scheduling:

| Workload | Reference median | Implemented median | Timing scope |
| --- | ---: | ---: | --- |
| 393,217 scalar gates and 262,144 lookups | 56.347 ms | 1.354 ms | Serial reference versus parallel relation checks; excludes circuit and assignment construction |
| 16,384 range checks with two, four or nine limbs | 9.858 ms | 8.614 ms | Single-output versus grouped hints; includes assignment generation and frontend relation checks, excludes circuit construction |

The range fixture reduces hint calls from **81,917 to 16,384**, with a
**12.6%** reduction in its measured assignment time. These short, warmed
synthetic workloads do not establish production staging or full-pipeline
speedups. In particular, they do not show how much of the preserved
46.87-second first-stage scalar-witness segment or 52.69-second recursive
witness segment belongs to validation. Production memory traffic and hint
mix differ from these fixtures.

Five targeted correctness tests pass with and without parallelism. They
check malformed assignments across chunk boundaries, deterministic error
precedence, exact range wire/constraint/assignment parity, and rejection of
out-of-range limbs even when their packed sum is correct. A small native
KZG fixture produces byte-identical proofs before and after hint grouping;
the seven existing foreign-translation tests also pass. The two ignored
tests `witness_validation_benchmark` and `grouped_range_hint_benchmark`
reproduce the isolated comparisons. Raw logs and source snapshots are
retained beside the report. Arithmetic recipe specialization is measured
below; a parallel recipe scheduler remains separate work.

### Specialize arithmetic witness coefficients

Implemented in [`witness.rs`](../src/plonkish/witness.rs): arithmetic recipes
use addition or subtraction for coefficients `1` and `-1`, and skip zero
terms. The general field multiplication remains for other coefficients.
Recipe order, dependency reads, hints, frontend relation checks and emitted
constraints are unchanged; gate validation still uses the original polynomial.

The [isolated CPU report](../experiments/kzg-cuda-validation/arithmetic-witness-20261009/report.json)
compares five alternating-order pairs after warmup on a mixed arithmetic
circuit with 163,846 logical values and 163,844 gates. Full scalar witness
generation, including allocation and relation checks, falls from a median
**11.230 ms to 6.913 ms**, a **38.4%** reduction. The same Goldilocks circuit
falls from **1.715 ms to 1.535 ms**, a **10.5%** reduction. Circuit construction,
lowering and proving are outside these timings; this short synthetic circuit
does not establish a production staging saving.

Three focused tests cover zero, unit, negative-unit and general coefficients
over both fields, exact assignment and error parity, and identical verified
KZG fixture bytes. All seven foreign-translation regressions pass. The report
retains the original evaluator, measured source and binary, commands and raw
timings.

### Bounded Goldilocks quotient advice

Implemented in [`foreign/quotient.rs`](../src/plonkish/foreign/quotient.rs):
Goldilocks gate quotient hints use four integer limbs for canonical source
values instead of allocating arbitrary-precision temporaries. Arkworks carry
primitives accumulate positive and negative terms separately, then exact
division by the Goldilocks modulus checks the remainder and preserves the
quotient's sign. Inputs whose canonical scalar representation exceeds 64 bits
retain the previous `BigInt` calculation, including its error behavior.

The coefficients have magnitude below `2^63`, each accepted input is below
`2^64`, and the cached offset is below `2^193`. Each signed accumulator is
therefore below `2^194`; after division the magnitude fits below `2^131`, well
inside Fr. This changes the advice calculation only. Wire indices, range
checks, constraints, lookup registration and public statements are unchanged.

The [isolated CPU report](../experiments/kzg-cuda-validation/quotient-hints-20261009/report.json)
records five alternating-order pairs after parity warmup. Computing 65,536
hints with mixed full-width coefficients takes a median **45.566 ms** with
the arbitrary-precision reference and **9.002 ms** with bounded arithmetic:
**5.06× faster**, or **80.2% less time**. Both timings include the output
vector allocation and exclude fixture/plan construction, frontend validation,
lowering and proving. This synthetic hint mix does not establish a production
staging speedup or the proportion of staging spent on these hints.

Three focused tests pass: exact integer-reference results for 2,048 valid
relations and their invalid residuals; signed, wide-input and top-limb
boundaries; and identical assignments, gates and verified KZG fixture bytes.
The seven existing translation tests also pass, including canonical bounds,
forged quotient rejection and exact circuit counts. The report retains the
measured binary, its 73 source/build inputs, raw timings and commands.
The same ten correctness tests pass with the `parallel` feature disabled,
including the identical verified proof digest.

### Compute translation intervals without heap integers

Implemented in [`foreign/interval.rs`](../src/plonkish/foreign/interval.rs):
modular gate construction computes its signed interval, ceiling offset and
quotient bit count with four arkworks integer limbs. Signed coefficients
convert directly into scalars. Both the strict direct-lift test `low > -P`
and `high < P`, and the existing quotient range limits, remain unchanged.
All interval magnitudes fit below `2^192`; the scalar offset is below `2^193`
and therefore embeds exactly in Fr.

The [isolated construction report](../experiments/kzg-cuda-validation/translation-interval-20261009/report.json)
records five alternating-order pairs after warmup for 16,384 modular source
gates. Constructing their translated hints, range constraints and residual
relations falls from a median **36.795 ms to 24.736 ms**, a **32.8%** reduction.
The comparison excludes source-bound propagation, the preallocation census,
witness generation, lowering and proving. It does not measure the complete
production translation region.

An independent arbitrary-precision reference matches 16,384 random plans and
signed, strict-boundary and quotient-bit transitions, including a zero-bit
quotient bound and negative upper endpoints. Exact gate, lookup, hint and
assignment layouts agree; fresh KZG fixture proofs verify and match byte for
byte. All 16 focused foreign-translation, range and quotient tests pass. The
existing quotient-advice proof digest is unchanged.

### Defer diagnostic names until an error

The first-stage source has **45,135,828 nonconstant values**. Both the counting
and materializing translation passes previously formatted `goldilocks[index]`
for every value: **90,271,656 diagnostic name formats per fresh compilation**.
The private input-name representation now retains a static prefix and source
recipe index, formatting only for a missing-input error. Ordinary caller-owned
names remain supported. Exact counting, preallocation, recipe order and every
constraint remain unchanged.

The [isolated production-input comparison](../experiments/kzg-cuda-validation/translation-input-names-20261009/report.json)
uses the preserved compressed FRI proof with its independently expected statement:

| Boundary | Reference | Indexed names |
| --- | ---: | ---: |
| Complete two-pass translation | 47.082 s | 42.950 s |
| Process, including input verification, source construction and cleanup | 55.059 s | 50.020 s |
| Peak process RSS | 45.62 GiB | 44.62 GiB |

This single A/B pair records **8.8% less translation time**, with identical
source/scalar counts and plan identity. It excludes assignment, lowering,
SRS loading and proving. In particular, it omits the live source-assignment
buffer present during staged proving; its process RSS is not full-stage RSS.
The source-construction timer includes plan validation and identity derivation.
The earlier full-stage and complete-chain measurements remain unchanged.

Five alternating-order pairs on a bounded mixed circuit give median total
construction times of **576.403 ms versus 538.157 ms (6.6% less)**. That eager
reference uses the current enum representation; the separate archived production
binaries compare the complete representation change. `InputName` occupies
32 bytes versus 24 for `String`, but avoids the owned string allocation. For
294,912 translated inputs, retained name capacity falls from a computed legacy
14,155,776 bytes to 9,437,184 bytes, excluding allocator overhead.

Nineteen foreign-translation tests pass, including exact recipes, constraints,
assignments, missing first/middle/final source-index errors, and identical
verified KZG fixture bytes. Two shared AIR/codec projection tests also pass.
To measure this boundary without initializing a setup:

```sh
MULTI_STARK_INIT_EXPECTED_CLAIMS=/path/to/root-claims.bin \
target/release/examples/init_fri_kzg_prove frontend-bench /path/to/compressed-fri
```

### Evaluate independent scalar advice in parallel

The compact Goldilocks translation supplies every nonconstant source value as
a scalar input. Each canonical-range or modular-gate advice block reads those
inputs, constants and its own intermediate values. The translator now records
coarse chunks of at least `2^16` recipes, except the final chunk, and certifies
those dependencies before enabling parallel evaluation. Each worker owns a disjoint output slice;
external reads resolve directly from immutable supplied inputs or constants.
Multi-output hints stay within a block. The earliest original-recipe error is
returned, and all existing constraint checks run after evaluation.

The schedule is private to internally generated pure advice. General hint
circuits, expanded hashes, unsupported dependencies and circuits given extra
witness recipes after certification retain serial evaluation. Nonparallel builds skip production
schedule construction. Constraints, value indices, trace layout and proof
format are unchanged.

The [production-input comparison](../experiments/kzg-cuda-validation/parallel-advice-20261009/report.json)
verifies the same saved FRI input against its independent expected statement,
then constructs and assigns the first-stage frontend:

| Boundary | Serial reference | Certified parallel advice |
| --- | ---: | ---: |
| Scalar recipe evaluation, including output allocation | 25.138 s | 1.065 s |
| Complete scalar assignment, including input conversion and relation checks | 27.742 s | 3.624 s |
| Two-pass translation, including schedule construction/certification | 42.432 s | 42.611 s |
| Whole diagnostic process, including source work and cleanup | 77.983 s | 54.022 s |
| Peak process RSS | 50.185 GiB | 50.183 GiB |

This single A/B pair saves **24.118 seconds (86.9%)** in fresh scalar
assignment. The candidate uses 2,027 certified chunks; dependency certification
takes 0.105 seconds inside translation. The source/scalar counts, plan identity,
public values and input hashes match. Both candidates perform full frontend
relation checks. The diagnostic includes a live source assignment and therefore
has a different memory boundary from the earlier translation-only experiment.
It excludes lowering, SRS loading, KZG proving and GPU work. The complete-chain
baseline remains unchanged.

Twenty-seven focused tests pass both with and without the `parallel` feature.
They include exact assignment and verified KZG-byte parity, canonical boundary
rejection, multi-output hints, all combinations of missing inputs and failing
hints in three blocks, certificate rejection, generic callback order and
serial fallback after a circuit is extended. The archive retains the source
delta, measured binaries, raw logs and feature-specific test results.

```sh
MULTI_STARK_INIT_EXPECTED_CLAIMS=/path/to/root-claims.bin \
target/release/examples/init_fri_kzg_prove assignment-bench /path/to/compressed-fri
```

### Reuse recursive field-advice conversions

Implemented in [`native_field/advice.rs`](../experiments/kzg-wrap/src/native_field/advice.rs):
the recursive Fq quotient/carry hint converts each scalar dependency once and
reuses it throughout the limb convolution. Each signed left limb is scaled
once before multiplication. Immutable modulus and bias constants are
shared across hints. Range hints extract their 16-bit limbs directly from
canonical scalar words, and integer conversion uses a stack byte buffer.

The signed `BigInt` quotient/carry arithmetic and its divisibility, overflow
and final-carry checks remain in place. Five 80-bit Fq limbs, the existing
carry bounds, wire order and constraints are unchanged. Independent review
and differential tests confirm the same advice values and rejection behavior.

The [isolated recursive-advice report](../experiments/kzg-cuda-validation/recursive-advice-20261009/report.json)
records five alternating-order pairs after warmup. For 4,096 Fq relations,
median quotient/carry advice time falls from **53.684 ms to 40.712 ms**,
a **24.2% reduction**. This excludes range hints, circuit construction,
witness validation and proving, so it neither measures the full benefit of
direct range extraction nor establishes a recursive-staging speedup.

Seven native-field tests, three curve tests and two complete-MSM tests pass.
They cover signed and redundant representatives, malformed inputs, carry
rejection, canonical ranges and exact frontend assignments/constraints. Two
CPU KZG proofs of the same field primitive, including its complete `2^16`
range table, verify and have identical bytes. The measured binary, 60
source/build inputs, commands and raw logs are retained with the report.

### Use exact shifts for recursive carries

The [division comparison](../experiments/kzg-cuda-validation/recursive-advice-division-20261010/report.json)
removes repeated work from the same recursive Fq advice routine. A relation now
gets its signed quotient and remainder together through the already-locked
`num-integer` primitive. Carry construction multiplies by `2^80` using a left
shift. Each division by `2^160` first checks exact divisibility, then shifts
right. Zero passes; negative nonmultiples are rejected before shifting, so
rounding cannot change their behavior. Quotient bias, carry bounds, check
ordering, rejection strings and final-zero checks remain unchanged.

For the same 4,096 deterministic relations, the median across five samples
falls from **40.444 ms to 33.455 ms**, a **17.3% reduction** against the
implementation immediately before this change. The archived current and new
executables run separately, each alternating its implementation with the
unchanged older reference after parity warmup. That reference's median differs
by 0.9% between runs. Timing includes advice and result-vector allocation;
it excludes input construction, range hints, full witness checks and proving.
No production-stage speedup is inferred from these samples.

All **25 focused correctness tests** pass: nine field tests, ten recursive
verifier tests, three curve tests, two complete-MSM tests and one proof-byte
comparison. Coverage includes signed exact multiples and adjacent nonmultiples,
zero, quotient overflow, malformed limbs, redundant representatives and the
genuine v4 development frontend's fresh/retained A/B/A assignments with native
MSM and pairing checks. The field primitive's two verified CPU KZG proofs retain
the same 2,245-byte encoding and BLAKE3
`508fe257542e77aff92fe7c096b7524a49a535c8e7158cf4f33468594bbc0d04`.

Only the direct dependency edge to the already-locked `num-integer 0.1.47` is
new; external dependency versions and checksums are unchanged. The CUDA wrapper
was rebuilt and passes capability preflight. The archive contains both source
snapshots, the current/new CPU executables, the rebuilt CUDA executable, commands
and raw results. This is a CPU witness optimization; no GPU or full-chain
performance run was needed for this arithmetic comparison.

### Make a cached run reuse compilation too

Implemented for the first KZG stage in
[`init_kzg_worker.rs`](../examples/support/init_kzg_worker.rs):
`stage-and-prove-many` accepts pairs of saved FRI input directories and new
output directories. It compiles once and retains the loaded proving state
after the first request. Subsequent requests avoid rebuilding the source
verifier circuit, both Goldilocks translation passes, and the lowered layout,
as well as restoring and decoding fixed coefficients or loading the SRS.

```sh
target/release/examples/init_fri_kzg_prove stage-and-prove-many \
  <saved-fri-dir-1> <new-output-dir-1> \
  <saved-fri-dir-2> <new-output-dir-2>
target/release/examples/init_fri_kzg_prove verify <new-output-dir-2>
```

One process handles one compatible verifier key, proof profile and constant
statement schema. The verifier-plan identity binds those inputs; loaded
prover reuse additionally checks trace dimensions. Public values and proof
witnesses are assigned afresh, and the existing expected Init statement checks
still apply. Requests are sequential. Frontend startup and each request have
separate timers; first-request key loading remains part of initial latency.
Warm outputs retain the metadata needed for standalone verification, while
the witness and main coefficients exist only for that proof.

The source circuit, scalar mapping/layout and fixed key remain live for the
process lifetime. Production peak host/device memory is unmeasured. Existing
device handles for fixed coefficients survive when admitted by the coefficient
budget; the loaded SRS remains in host memory and MSM point uploads still
occur. Selected fixed-evaluation caching and resident SRS points are separate
optimizations. The [small-fixture result](#retained-first-stage-compilation-and-proving-keys)
measures repeated first-stage requests, not production or cold-start latency.

Recursive reuse is implemented below; FRI still needs equivalent reuse. The
[recursive proving log](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/recursive_prove.log)
records 23.335 seconds loading an already populated development-SRS cache.
Disk preprocessing, a warm filesystem cache, and live reusable objects are
different benchmark states. Report startup separately from repeated proofs;
each proof still needs a fresh assignment.

For an initial proof, known-profile SRS/key loading can also overlap earlier
independent CPU preparation once the process exposes that scheduling boundary.
Charge startup work to the initial-proof measurement even when overlapped.

Opaque Rust hint closures make keeping live objects an earlier option than
serializing a portable execution plan.

### Retain the recursive frontend and outer key

Implemented in [`native_verifier::Plan`](../experiments/kzg-wrap/src/native_verifier/plan.rs),
[`Bindings`](../experiments/kzg-wrap/src/native_verifier/bindings.rs), and the
[`recursive worker`](../experiments/kzg-wrap/src/saved/worker.rs). Compilation
consumes an authenticated inner verifier profile, an exact proof shape, and a
typed claim schema: `Constant(Scalar)` or `PublicU64`. Constants are allocated
only for fixed slots. Public slots preserve their exposure order and 64-bit
range without capturing the first request's values. The compiled input map
retains typed paths to commitments, opening witnesses, opened values,
accumulators and claims; every assignment resolves those paths against its
current proof. Canonical encoding, infinity handling, on-curve/subgroup checks,
and the constrained witness-selected MSM offset remain in place.

The versioned compatibility identity binds:

- Ordered circuit graphs and metadata, fixed commitments and preprocessed mapping.
- The complete inner KZG transcript seed, degree policy, exact trace logs,
  generator, G2/τG2 and applicable degree keys from the same authenticated setup.
- Exact commitment/opening dimensions, round order and distinct-point batch schedule.
- Claim row lengths, constant values, ordered public slots and their range policy.
- Lowering namespace, one-trace/table limits and frontend/lowering version.

Actual public values and proof payloads are excluded from that identity.
Persisted fixed caches additionally bind the executable digest. Before
assignment, exact commitment/opening dimensions, witness count, trace logs,
claim shape and constant slots are checked. The all-active restriction,
canonical one-fixed-matrix-per-circuit mapping and native input-proof
verification still apply. MSM cross-checks rebuild their point list from the
current request.

Single-proof `stage` and `stage-and-prove` commands use the same
[`Input::plan`](../experiments/kzg-wrap/src/saved.rs) helper and build from
exactly the plan whose identity keys their fixed cache. This avoids a cache
miss merely because public words change. The Filecoin worker and single-proof
paths both select the `2^27` policy, while the explicit legacy development
single-proof path keeps its `2^29` limit and a distinct identity. The outer
setup identity and executable digest remain additional cache keys. Native
verification and the independently expected statement are checked before
reuse; dense matrices and the full SRS are not rehashed per request.

The native `Circuit` moves into one retained `MultiStarkCircuit`; the worker
also retains its input map, pairing metadata, KZG system, proving key, proof
codec and shared SRS allocation. It permits **one assignment in flight** and
produces expanded traces on demand. The admitted layout is exactly one
computation trace of at most `2^27` rows plus one merged table with the same
ceiling. Oversized layouts fail admission. Reusing a key requires matching
setup/frontend identities, trace dimensions and pairing metadata, and a fresh
trace source. Warm outputs contain setup metadata for independent verification
without witness/main checkpoints or copies of the retained fixed coefficients.

Each request supplies an independent expected-claims file. Its strict 160-byte
frame is loaded once into an owned array before proving, then used for input
verification, assignment checks, packet construction and verification:

```sh
target/release/init-kzg-wrap stage-and-prove-many \
  <saved-stage-one-1> <new-recursive-output-1> <expected-claims-1.bin> \
  <saved-stage-one-2> <new-recursive-output-2> <expected-claims-2.bin>
MULTI_STARK_INIT_EXPECTED_CLAIMS=<expected-claims-2.bin> \
  target/release/init-kzg-wrap verify <new-recursive-output-2>
```

The executable path above assumes `CARGO_TARGET_DIR=target` at the repository
root. Select the authenticated setup/cache as described in the recovery
commands. Frontend startup includes the first input's load and native
verification; the first request still includes outer key loading. Later
request timers include input loading and native verification. Per-request
JSON separately reports assignment/check time and whether the key was reused.
No warmed production latency or cold-start improvement has been measured.

The [focused validation report](../experiments/kzg-cuda-validation/recursive-worker-20261009/report.json)
separates two correctness checks. A native recursive fixture compares complete
relations and constrained assignments between fresh and retained compilation
for statements A, B, then A, including assignment after lowering. A separate
small outer fixture compares actual proof, packet and profile bytes for the
same sequence, checks SRS allocation reuse, and verifies each saved result
independently. Wrong statements, stale trace sources and incompatible profiles
are rejected. This composition is not a production-size recursive proof.
The [cache-profile follow-up](../experiments/kzg-cuda-validation/recursive-cache-profile-20261009/report.json)
passes all nineteen focused tests: four statement-loader checks, six plan/input
checks, five outer/storage/setup checks, three existing native wrapper fixtures
covering both degree policies and identity pairing outputs, and a saved-profile
check using two independently verified eighteen-word statements. It also builds
the CUDA-enabled executable. The native A/B/A check compares the retained
frontend directly against the single-proof `Plan::build` adapter. These test
and build times are validation observations, with no production speedup claim.

Removing unused public-claim constants changes the **frontend/cache profile**;
outer fixed keys and artifacts must be regenerated. Within the new profile,
fresh and retained proof bytes match in the small outer fixture. Old outer
proof-byte parity is not an acceptance condition for this circuit change.

### Schedule requests through retained workers

Both KZG executables implement `serve <new-response-jsonl-path>` using the
[shared request protocol](../examples/support/kzg_worker_protocol.rs). Standard
input accepts one JSON object per line, with one request in flight:

```json
{"id":"request-1","input":"/saved/input","output":"/new/output","expected_claims":"/independent/claims.bin"}
```

The dedicated response file uses `multi-stark-worker/v1` and flushed
`listening`, `request_started`, `proved` or terminal `failed` events. Listening
explicitly means `prepared=false`. Only a successful genuine proof, native
verification and idle release establish preparation. Responses record frontend
and loaded-key reuse, frontend compilation time, complete request time and
per-device idle evidence. Native logs are flushed before completion so the
driver can isolate each request's CUDA events by byte offsets. Malformed or
duplicate requests, nonempty outputs and incompatible profiles fail closed.

Each expected-claims file is read once into an owned eighteen-word statement,
which flows through input verification, assignment, proving and packet checks.
The first-stage service keeps its original verifier plan alive because the
input map binds its process-local identity. Both services retain host plans,
SRS allocations and fixed coefficients; they generate fresh assignments and
proofs for subsequent compatible statements.

The first-stage packet's existing `profile-id.bin` hashes the manifest,
including public claims. It therefore changes for different statements even
when frontend and setup identities remain compatible. Record it for every
proof; compare it across first-stage requests only when the expected statement
is the same. The recursive packet profile is independent of public words.

The [pipeline driver](../experiments/kzg-cuda-bench.py) has opt-in
`--retained-workers`, requiring `--root-artifacts`, `--worker-seed-fri` and an
independently supplied `--worker-seed-claims`. It prepares the first worker
with a genuine seed proof, exports that newly generated KZG proof to prepare
the recursive worker, and records all preparation separately. It then starts
the request timer before fresh FRI compression, feeds the new proof through
both retained workers, and stops after final native verification and idle
release. Independent CPU verification and worker shutdown have separate times.
`retained_worker_fresh_request` and `cold_process_pipeline` are distinct latency
boundaries; startup plus the measured request is retained in the report.

GPU ownership has an explicit idle boundary. Exclusive
`KzgProverData::release_device_residency(&mut self)` takes the `OnceLock`
coefficient handles while preserving their authoritative host representation.
After request temporaries are dropped, `KzgPcs::release_idle_device_memory()`
acquires all configured device leases, evicts SRS ranges and MSM workspaces,
synchronizes, and trims unused current/default CUDA pool pages. Dropping the
coefficient handles precedes acquiring all leases, since their destructors
also acquire a lease. Stock sppark streams and live NTT tables remain valid;
no device reset is used. A later request can hydrate coefficients again.
CPU execution and never-initialized CUDA return explicit no-op reports.

The [four-device lifecycle fixture](../experiments/kzg-cuda-validation/request-workers-20261010/cuda-lifecycle/report.json)
passes with a nonconstant `2^16` column on every card and a constant column.
It checks complete opening-byte parity and verification before/after release,
unchanged host checkpoints/SRS identity, preservation of live handles, cache
eviction, repeated release and admission recovery. After final release,
coefficient, SRS and MSM workspace counters are zero on all cards. Each retains
**32 MiB of pool pages**, of which **16.78125 MiB** holds live NTT tables; raw
driver accounting reports **718.1875 MiB used per card** while the process is
alive. This is a small lifecycle test, not a production latency result.

The [focused worker validation](../experiments/kzg-cuda-validation/request-workers-20261010/report.json)
also passes eleven CPU Rust tests and twenty-two Python tests. First-stage
and recursive A/B/A fixtures use fresh assignments, preserve return-A bytes
and match independently rebuilt proofs for B. First-stage expected files are
deleted after loading to check that later proving uses the owned statement.
The Python integration uses fake proving subprocesses to test protocol,
fresh-input flow, timing, log isolation, reuse and cleanup; it is not a GPU
performance measurement. Both actual CUDA executables also pass EOF and missing
setup CLI checks, and all three chain executables pass capability preflight.

The earlier isolated FRI run peaks at **96,893 MiB on GPU 0**. Two idle
processes still retain two contexts/modules/NTT allocations, and their lease
locks are local to each process. The [coexistence diagnostic](#fri-with-two-idle-kzg-contexts)
below now passes with two small verified KZG fixtures. It leaves both complete
production workers and their retained host keys to test. The real Filecoin
setup, trusted current profile, actual recursive fit and complete packet
remain final acceptance requirements.

#### Production first-stage worker measurement

The [production-input report](../experiments/kzg-cuda-validation/request-workers-20261010/production-first-stage/report.json)
uses the preserved compressed FRI proof and independent root claims. It starts
one worker with a warm development SRS and a new fixed-cache directory; the
cache binds the new executable. It proves the same input twice with fresh
assignments, releasing GPU allocations after each request. This is one sample
of the first-stage service boundary, not a new full-chain result or a cold-start
speedup comparison.

| Measured boundary | Seconds |
| --- | ---: |
| Spawn, frontend compilation, initial fixed/key preparation, genuine proof and GPU release | 240.717 |
| Fresh request through the prepared worker, including GPU rehydration and release | **36.647** |
| Graceful worker shutdown | 4.569 |
| Independent CPU verification of the fresh result | 4.055 |

Initial frontend compilation takes 58.098 seconds. The fresh request reports
both frontend and loaded-key reuse, with no fixed-cache or SRS-file reads. Its
prover preparation reaches all nineteen main commitments in 13.901 seconds;
proof computation then takes 15.393 seconds. The remaining request work
includes native input verification, fresh source/scalar assignments, relation
checks, packet/negative checks and idle release. These inner timers do not
replace the complete 36.647-second external request boundary.

All **1,637 CUDA profile records parse**, with positive compute events on all
four cards. Kernel totals per card are 7.451–7.520 seconds, which overlap across
devices. One-second samples during the fresh request show mean utilization of
29.2–35.1% and peak memory of 35,347–35,795 MiB per card. After release, each
card reports 723 MiB through `nvidia-smi`, with zero coefficient/SRS/workspace
counters and 32 MiB of retained pool pages. Sampled utilization immediately
after release still reflects its preceding sampling interval.

The worker's lifetime host peak is **219.462 GiB**; its RSS after the fresh
request is **144.639 GiB**. The lifetime peak spans preparation and proving.
The 49,557-byte intermediate proof, 49,733-byte packet and profile identifier
match the historical first-stage artifacts exactly. Both native verifications
reject all eighteen altered claim words; independent CPU verification
preserves the saved artifacts. The archive records recovery of an optional-file
collection error after successful proving and CPU verification; neither was
rerun. All four GPUs report zero allocated memory after worker shutdown.

The next production measurement is a compatible `2^27` recursive request with
authenticated Filecoin keys, followed by fresh FRI compression with both
complete workers alive. The full fresh root-to-packet benchmark remains
outstanding.

#### FRI with two idle KZG contexts

The [coexistence report](../experiments/kzg-cuda-validation/fri-worker-coexistence-20261010/coexistence/report.json)
records one fresh FRI replay in **56.280 seconds**. Two separate
[`kzg_cuda_idle`](../examples/kzg_cuda_idle.rs) processes first commit and verify
four nonconstant `2^16` columns each, release coefficient/SRS/workspace
allocations, and remain alive throughout FRI. Afterwards each rehydrates the
same data, verifies its evaluations and openings, checks exact opening-byte
parity, releases allocations again and exits successfully. The fixture uses
explicit known-trapdoor development parameters and retains about 1 GiB of host
memory per process; it does not contain a production frontend or host key.

| Measured FRI phase | Seconds |
| --- | ---: |
| Circuit build | 6.438 |
| Root verifier witness | 1.579 |
| Lowering | 5.220 |
| Trace construction | 4.311 |
| Setup | 7.777 |
| Proving | 29.796 |
| Complete FRI command, including native verification | **56.280** |

Before FRI starts, one fixture accounts for **719 MiB per card**, close to the
production first-stage worker's measured 721–723 MiB. Both fixtures together
account for **1,436 MiB per card**. Each retains 32 MiB of pool pages including
16.78125 MiB of live NTT tables, with zero coefficient/SRS/workspace counters.
These are observed fixture footprints, not a guarantee of the final two
workers' footprint.

Existing FRI admission and spilling handle this pressure without a backend
change. Sampled total GPU-0 usage peaks at **96,725 MiB**; the final FRI
admission releases one 256 MiB LDE in 0.020 seconds. The FRI process's host RSS
peaks at **204.164 GiB**. GPU 0 performs FRI computation; the other cards retain
contexts and show zero sampled utilization during this phase. FRI's own
executable initializes contexts on those cards too, bringing sampled idle-card
memory to 2,153 MiB while all three processes exist.

The 1,418,855-byte FRI proof, verification key and claims match the preserved
reference exactly, and native verification rejects the altered public claim.
Both KZG fixtures preserve their 104-byte opening with BLAKE3
`8a40e53643cec7a070ffe3e435b162eb36097c27f0fd359aa7619bd82b18e7ea`.
The [archived evidence check](../experiments/kzg-cuda-validation/fri-worker-coexistence-20261010/analysis.json)
parses all 144 CUDA records across preparation and post-FRI checks, with
positive compute events on every card in each boundary and no skipped records.
All child processes exit and all four cards return to zero allocated memory.

The complete diagnostic, including fixture startup and checks, takes 60.331
seconds. Its archived FRI executable differs from the earlier 73.171-second
sample; no same-executable isolated reference was run. This establishes one
successful GPU coexistence measurement, without attributing a speedup or
overhead to idle workers. The later [integrated development request](#integrated-development-v4-worker-pipeline)
measures coexistence with both complete retained workers and the full fresh
request. Authenticated Filecoin proving remains unmeasured.

### Overlap downstream preparation with an upstream proof

First-stage frontend compilation consumes a verifier key, proof profile and
statement schema; `Frontend::compile` does not consume the proof witness.
Expose that preparation boundary independently of input-proof loading.
Once FRI setup has produced its trusted verifier metadata, downstream scalar
translation and lowering can run while FRI generates the proof. The saved
50.903-second translation and 25.880-second lowering regions identify work
that could overlap; their sum is not a predicted latency reduction because
both stages compete for host compute and memory bandwidth.

Publish only complete metadata with its plan/setup identities. Before assigning
the resulting proof, require the actual key/profile/schema to match the prepared
frontend and perform the existing input verification. Challenge-dependent
witness work must still wait for the upstream proof. Keep this as a bounded
producer/consumer schedule, with the combined live circuit, assignment and
device working sets admitted before overlap begins.

The recursive plan now separates construction from proof-input assignment.
Its constant claim slots and public-slot schema must be known before
construction; public values can arrive with the proof. The saved-input worker
still starts after loading an input proof, so preparation overlap requires a
metadata publication boundary. First compare
one prepared first-stage frontend plus FRI proving against sequential execution
using preserved inputs, and measure the combined wall time and peak memory.
Charge all preparation to the initial request even when it overlaps. That
first-stage experiment is separate from the warmed-worker measurement and
requires no change to its proof relation or format; the recursive frontend
cleanup separately requires a new outer profile.

### Spend host memory on reusable data

[Selectors](../src/ark_adapter/domain.rs) depend only on the trace domain and
quotient coset. [Quotient evaluation](../src/ark_adapter/config.rs) recreates
them for each circuit/coset, including repeated stage-one domain shapes.
A bounded cache keyed by both domains can reuse them within and across
proofs. Some selectors admit compact representations: inverse vanishing is
constant on a trace-sized coset, yet the current interface expands it to an
4 GiB vector at the target `2^27` rows. Preserve this periodic structure in a tiled
consumer instead of storing full vectors where possible.

Fixed polynomial evaluations on lookup domains and quotient cosets are also
invariant for a fixed profile. The existing cache stores coefficients and
commitments; caching selected evaluated matrices could avoid repeated FFTs,
downloads and transposes. Use shared immutable views so a cache hit does not
copy the matrix. Bind the fixed polynomial identity, domain and field
representation, not merely the row count.

These caches need a joint host-memory budget with live assignments, pipeline
lookahead, row matrices and scratch. For scale, fifteen fixed scalar columns
at `2^27` occupy 60 GiB on one evaluation grid; three grids consume 180 GiB
before other data. Peak process RSS alone does not measure available memory
or filesystem cache. Prefer demonstrated reuse per byte and bounded eviction
over retaining every expanded representation.

### Remove the compulsory file boundary before requiring GPU witnesses

Both KZG stages now provide `stage-and-prove`. The first-stage command keeps
the compiled circuit and assignment in one process and generates each trace
directly into commitment. Its bounded prefetch prepares the next trace while
committing the current one. The recursive command passes a lazy trace source
to the outer prover, generating each computation/table trace when committed.
Fresh proving skips
witness compression, witness-file writes/reads, and main-polynomial checkpoint
writes. It always uses the supplied assignment, even if an older main
checkpoint exists. Fixed preprocessing, cache identity checks, output packets
and standalone verification use the same paths as checkpointed proving.

```sh
target/release/examples/init_fri_kzg_prove stage-and-prove <saved-fri-dir> <new-output-dir>
target/release/examples/init_fri_kzg_prove verify <new-output-dir>
experiments/kzg-wrap/target/release/init-kzg-wrap stage-and-prove <saved-stage-one-dir> <new-recursive-dir>
experiments/kzg-wrap/target/release/init-kzg-wrap verify <new-recursive-dir>
```

Use the existing `stage` then `prove` commands when intermediate witness and
main-polynomial checkpoints are needed for resume. The fused mode retains
the compiled layout, assignment and compact preparation alongside prover
data; the prefetch limit only bounds the additional trace matrix. Recursive
staging retains its compiled layout and assignment; generated row matrices
are consumed one at a time. Production peak memory still needs measurement.
The benchmark driver's
`--fused-staging` selects this mode for both stages.

The targeted recursive regression compares checkpointed and fused proof,
packet and profile bytes, changes the private witness, then supplies that fresh
witness in a directory containing stale main-polynomial checkpoints. It also
verifies each result independently without witness files. All checks pass;
the small fixture is a correctness gate, not a production speedup estimate:

```sh
cargo test --release --manifest-path experiments/kzg-wrap/Cargo.toml \
  fused_recursive_traces_preserve_proof_bytes_and_resume_verification
```

Where the budget permits, retaining original main trace evaluations for
lookup construction is still a separate opportunity; this mode commits and
discards each generated row matrix as before.

Respect transcript barriers: all main commitments precede lookup challenges,
and subsequent challenge-dependent work cannot start early. Bounded production
and consumption within each phase can still overlap. The current partition
pipeline already demonstrates the benefit; deeper lookahead should be driven
by elapsed-time measurements because concurrent CPU work can contend for
memory bandwidth.

For repeated proofs, separately evaluate overlapping CPU staging for one job
with device proving for another. This targets throughput, not necessarily
one-proof latency. Reuse compatible worker state and admit jobs against the
combined host/device working set, including FRI allocations. Higher aggregate
GPU utilization is useful only if completed proofs per unit time improves.

### Choose device data placement together with the consumer

The archived full-chain stage-one and recursive logs record approximately 823 GiB of
forward-FFT downloads and 370 GiB of SRS-point uploads in total. These are
traffic volumes, not wall-time savings. The implemented device lookup and
quotient consumers remove intermediate evaluation downloads on admitted
shapes and generate selectors where they are consumed. Shared accounting for coefficients,
SRS, scratch and transfer buffers is required before enlarging the current
coefficient retention cap.

The current allocator spreads retained columns among GPUs. A row-wise
constraint consumer needs several columns together, so column residency does
not imply that its inputs are local to one device. Admitted stage-one consumers
run a complete bounded partition on one GPU, with independent partitions on
the other GPUs. At the target `2^27` height, fifteen fixed columns alone occupy
60 GiB, before main/lookup evaluations, resident coefficients and scratch.
The distributed consumers use the measured pair topology, bounded row tiles
and an 8 GiB per-device coefficient cap to fit that working set. Same-pair
coefficient reads remove uploads caused by mismatched owners within a pair;
unequal per-device work remains a scheduling opportunity. More streams alone
do not resolve the placement problem.

### Specialize arithmetic advice before building a general compiler

Use the separate witness phase timers above to identify the remaining
evaluation cost, then profile hint families within that phase. Goldilocks
range hints now share canonical limb extraction, and bounded quotient advice
avoids `BigInt` temporaries on canonical source inputs. The recursive
field gadget now shares scalar conversions and immutable constants and
extracts range limbs directly. Its [combined division and exact carry shifts](#use-exact-shifts-for-recursive-carries)
remove redundant arithmetic while retaining arbitrary precision. Profile the remaining advice families before
replacing those calculations with bounded limbs or typed GPU operations.

Preserve signed quotient/carry behavior and prove intermediate bounds before
replacing arbitrary-precision arithmetic. Rewriting hints without changing
their outputs can preserve the circuit; changing limb widths or wire order
requires its own profile and fixture review. A general scheduler is justified
only after measuring the residual cost and supported hint coverage.

### Parallelize independent recursive point checks

The native wrapper can now evaluate its independent point-input prefix in
parallel, then continue the transcript and MSM recipes serially. The
[candidate census](../experiments/kzg-cuda-validation/recursive-worker-20261009/census/count.log)
separates 35,025,875 values for curve inputs, 19,513,430 for transcript/AIR work
and 60,970,802 for the two MSMs. The 335 dynamic point inputs have independent
canonical-coordinate, infinity, curve, subgroup and encoding work: 30.3% of
values and 2,337,965 hints. These counts do not measure their runtime share.

`Context::point` records each complete point-and-encoding block, skipping
duplicate boundaries when a fixed point allocates no new values. The wrapper
seals the prefix after the final fixed generator. The builder checks that each
chunk reads only its own earlier outputs or shared inputs/constants, with
complete multi-output hints and valid ownership. Dependent or malformed blocks
retain serial execution. The transcript consumes point encodings only after
the parallel prefix joins.

The public `try_parallel_witness_prefix` API seals exactly the recipes emitted
so far. Every hint in that prefix must explicitly use `hint_pure` or
`hint_many_pure`: its outputs and errors must depend only on declared inputs
and immutable captures, with no observable callback-order effects. Ordinary
hints remain opaque and prevent public certification. A single builder flag
tracks that condition; no metadata is added to each of the candidate's
9.9 million hints. Ordinary callbacks appended after the seal run in the
serial suffix. The existing private whole-circuit scheduler retains its
fallback when recipes are appended.

Witness generation reserves one complete assignment, initializes only the
parallel prefix, joins its disjoint slices, then appends the serial suffix in
the same allocation. It selects the earliest original recipe error before
starting the suffix; full relation checks remain after all recipes. There is
no second assignment copy, per-value dependency graph or additional suffix
zero-fill. Subgroup and canonical-encoding constraints, request-specific native
MSM comparisons and pairing checks remain intact. The purity contract changes
execution permission; it never replaces a constraint or validation check.

The [isolated point-prefix report](../experiments/kzg-cuda-validation/recursive-point-prefix-20261010/report.json)
measures 335 dynamic points using the native point gadgets, including identity,
duplicates, opposite points and deterministic scalar multiples. Five alternating
serial/parallel pairs give medians of **4.048256 seconds and 0.316232 seconds**:
**12.8× faster**, or **92.2% less time** for this operation. Both versions produce
the same complete assignment and canonical point encodings. There is no untimed
warmup; the first execution is included.

The fixture has 337 prefix blocks, 35,023,984 prefix values, and a dependent
ordinary-hint suffix. Its public encodings and sum expose 16,273 values for
comparison, unlike the actual wrapper's 22 public values. Timing includes input
assignment, witness allocation and evaluation, complete frontend relation checks,
and public extraction. Construction, parity checks, checksums, assignment
destruction, lowering and proving are excluded. The two equivalent circuits take
10.018 seconds to construct; the full diagnostic takes 39.409 seconds and peaks
at 16.916 GiB RSS, using default CPU parallelism. These coordinates and suffix are
synthetic, so the result does not establish a production recursive-stage saving.

Forty-eight distinct focused correctness tests pass, with eleven repeated in a
build without the `parallel` feature. They cover dependency and purity admission,
ordinary callback order, prefix/suffix error precedence, exact assignments and
encodings, subgroup rejection, genuine v4 A/B/A plan reuse, and native MSM/pairing
checks. A small fully constrained KZG prefix/suffix fixture verifies identical
proof bytes; its development setup and proof size are unrelated to the final
ceremony packet. All three CUDA pipeline executables are rebuilt with the change.

The MSM term loop interleaves table/digit construction with a running sum;
its 64-window accumulator is also chained. Arbitrary term chunks are therefore
not independent. Table multiplicities are another fresh-request cost; the
[bounded histogram implementation](#build-table-multiplicities-in-parallel)
addresses their serial scan separately. IR construction, copy-successor
lowering and cold fixed preprocessing belong to startup, which retained
requests already skip.

Production `2^27` timing cannot replay the archived v3 proof: its shifted-degree
wrapper needs 225,238,337 rows. The smaller v4 census changes metadata for
counting and is not an assignable proof. The following diagnostic supplies a
fresh, valid development-v4 inner proof for repeatable frontend measurements.
Authentic Filecoin input remains the acceptance path.

### Measure the complete v4 recursive frontend

The [complete frontend report](../experiments/kzg-cuda-validation/recursive-v4-frontend-20261010/report.json)
measures a genuine, newly proved development-v4 inner proof through recursive
compilation, fresh assignment and trace construction. The one-time inner-proof
bootstrap takes **79.245 seconds**. Subsequent frontend diagnostics load that
small fixture and require no first-stage reproving, outer SRS or outer proof.

| Boundary | First sample | Second sample |
| --- | ---: | ---: |
| Fresh assignment, complete relation and native MSM/pairing checks | 7.571 s | 7.593 s |
| Main trace, `2^27 × 3` | 2.261 s | 2.278 s |
| Merged table trace, `2^17 × 1` | 0.053 s | 0.052 s |
| Assignment and both traces | **9.884 s** | **9.922 s** |

Compilation takes **14.363 seconds**, lowering **4.191 seconds**, and loading
plus native verification **0.076 seconds**, separately. The full two-sample
diagnostic takes 46.296 seconds including checksums, comparisons and destruction.
It peaks at **46.607 GiB RSS**, including the retained **3.442 GiB** first
assignment used for complete equality comparison; this is not single-request
peak memory. No untimed warmup or CPU thread override is used.

The circuit has **118,634,362 rows**, **115,510,108 values**, 86,046,657 gates,
32,587,682 lookups and 22 public values. It lowers to one computation trace and
one merged table, with **15,583,366 rows (11.61%)** remaining below the main cap.
The point prefix admits 505 blocks and 35,025,875 values, including cheap fixed
point blocks. Both fresh samples satisfy every relation, check both native MSMs
(583 and 9 terms) and external pairings, and produce equal complete assignments
and canonical trace hashes. Hashing the main trace takes another 2.481–2.482
seconds per sample; diagnostic checksums, comparisons and destruction are outside
the operation timers. These timings measure the current implementation, with
no before/after speedup claim.

The preserved first-stage checkpoint set contains **31.943 GiB of main
coefficients and 84.772 GiB of fixed coefficients**. All nineteen compiled
metadata files match the [sizing archive](../experiments/kzg-cuda-validation/trace27-sizing-20261009/report.json).
For this layout, switching from 96-byte legacy commitments to 48-byte v4
commitments does not change any selected lookup group, stage-two width or
quotient ratio. Recheck this metadata condition before reuse; it is not a
general property of every circuit.
The bootstrap reads and hashes **116.715 GiB** across 38 coefficient checkpoints.
Restore, canonical decoding and hashing take **45.820 seconds**, recomputing
commitments **11.788 seconds**, and fresh v4 proving **17.321 seconds**. All four
GPUs record MSM compute events; peak host RSS is 158.688 GiB. Native and decoded
proof verification pass, all eighteen altered claims reject, and truncated,
extended and damaged proof checks pass. GPU coefficient, SRS and MSM workspace
allocations are quiesced afterward. The **38,229-byte inner proof** is input to
the recursive wrapper; it is not the final packet. This checkpoint route avoids
first-stage witness reconstruction and inverse FFTs, but its input exceeds the
2.853 GiB of compressed raw matrices. No comparison against a raw-trace bootstrap
is claimed.

[`dev-v4-bootstrap`](../experiments/kzg-wrap/src/saved/development.rs) uses a
separate known-trapdoor development setup identity, derived internally from the
seed and full public degree `2^28 - 2`. Its loaded first-stage prefix and trace
cap are `2^24`; recursive layout is capped at `2^27`. The setup identity is
independent of the loaded prefix, so frontend verification needs only two G1
anchors. No `2^28` trace or SRS allocation is used. Normal Filecoin selection and
authenticated import remain unchanged; the diagnostic rejects Filecoin mode.

`KzgPcs::recommit` checks every coefficient matrix before any MSM, regenerates
ordinary and applicable shifted commitments under the selected setup, and retains
the coefficient allocations. The bootstrap checks old checkpoint commitments
against the verified legacy proof, then requires recomputed ordinary commitments
to match and all v4 shifted vectors to be empty. Every transcript-dependent lookup,
quotient and opening is regenerated with a fresh challenger. Compiled metadata and
local checkpoints remain trusted inputs: this diagnostic does not reconstruct or
independently authenticate the intended Init AIR from FRI.

Twenty-four distinct focused correctness tests pass, covering recommitment across
setups and degree policies, cache-prefix identity, metadata compatibility, saved
input validation, and native v4 request reuse. Two CLI checks reject incompatible
setup modes before creating output. The library tests and CUDA wrapper are built
from recorded sources; a diagnostic-only Rust type-inference compile failure and
its successful correction are preserved in the archive.

The archive contains the small valid fixture and CUDA executable. A frontend-only
replay uses a new output directory:

```bash
MULTI_STARK_KZG_SETUP=development \
MULTI_STARK_KZG_BACKEND=cpu \
MULTI_STARK_INIT_EXPECTED_CLAIMS=experiments/kzg-cuda-validation/recursive-v4-frontend-20261010/root-claims.bin \
experiments/kzg-cuda-validation/recursive-v4-frontend-20261010/init-kzg-wrap \
  dev-v4-frontend-bench \
  experiments/kzg-cuda-validation/recursive-v4-frontend-20261010/development-fixture \
  /tmp/kzg-v4-frontend-replay
```

The following measurement feeds this saved inner proof through the shared
fused/retained outer prover. It separates outer SRS and fixed-preprocessing costs
from a fresh request using the loaded key.

### Complete development-v4 recursive proving

The [recursive proving report](../experiments/kzg-cuda-validation/recursive-v4-proving-20261010/report.json)
records **34.412 seconds** for a fresh recursive request on the retained host
frontend and key. It includes fresh assignment, full relation and native
MSM/pairing checks, both traces, all commitments and openings, native and negative
proof/packet verification, assignment destruction and GPU quiescence. Each request
generates its own outer assignment and proof.

| Boundary | First request | Retained request |
| --- | ---: | ---: |
| Fresh assignment and checks | 7.595 s | 7.664 s |
| Fused stage and prove | 189.426 s | 26.585 s |
| Device quiescence | 0.252 s | 0.153 s |
| Complete request, including destruction | **197.284 s** | **34.412 s** |

Loading the saved inner proof, compilation and lowering are outside these
request timers. Compilation takes **14.375 seconds**, lowering **4.042 seconds**;
the complete process with both requests takes **252.487 seconds** and peaks at
**226.559 GiB RSS**. That memory peak includes cold preparation, not an isolated
retained request. All 96 default CPU threads are available, with no thread cap
or simultaneous build/benchmark.

Both requests produce exactly the same **1,909-byte proof**, **2,181-byte packet**
and profile bytes. The packet comprises 32 profile bytes, 144 claim bytes,
96 pairing-point bytes and the proof. Separate CPU verification of the copied
saved packet passes in **0.126 seconds**, including negative checks. It derives
only two development G1 anchors, uses no SRS cache, and leaves the proof, packet
and profile unchanged. The layout remains one `2^27 × 3` computation trace and
one `2^17 × 1` merged table, with 118,634,362 unpadded computation rows.

Both large-domain consumers are admitted on all four GPUs. The first request's
lookup takes **3.811 seconds** and quotient **4.390 seconds**; the retained
request takes **4.075 seconds** and **4.588 seconds**. These are consumer wall
times inside the stage/prove timer. They include the native consumer's transfers
and final coefficient downloads, and are not additive savings relative to prior
synthetic comparisons. Lookup uses 18 evaluated columns and four groups; quotient
uses 22 evaluated columns. Both use `2^18` row tiles and runtime peer pairs.
The small merged table uses the one-device consumers.

GPU sampling shows **21–26% mean utilization** across the complete retained
request, with up to **69.706 GiB** used on a card. All four record MSM compute
events. After each request, tracked coefficient, SRS and MSM workspace bytes are
zero on all four cards, while roughly 720–722 MiB per card remains for contexts,
NTT tables and small pool allocations. This measures a retained **host** key with
device allocations released between requests, matching the worker admission
boundary. It does not measure a permanently device-resident key or coexistence
with both complete workers and FRI.

The first request has no fixed-cache hit and generates a new `2^27` development
SRS. Fixed staging takes **36.183 seconds**, SRS generation/cache writing
**56.382 seconds**, and reading/decoding the main fixed trace **31.256 seconds**.
The raw fixed matrix is 60 GiB; its staged compressed file is 1.456 GiB. The
retained request skips that materialization, disk read and key construction:
staging takes 2.476 ms and SRS/config loading reports exactly zero. Its main
traces are generated and committed by 5.591 seconds into proving; subsequent
lookup, quotient, openings and verification take another 20.991 seconds.

[`outer::parameters`](../experiments/kzg-wrap/src/outer/parameters.rs) separates
public-degree policy from setup provenance. Normal `SetupSource` environment
selection is unchanged; the explicit diagnostic adapter supplies the internally
derived known-trapdoor identity, full degree `2^28 - 2` and `2^27` cap. The normal
Filecoin and diagnostic public-degree paths use the same staging and proving
implementation, with strict setup/profile binding, matching inner/outer G2 anchors, zero degree
outputs, v4 profile hashing and a packet strictly below 3,000 bytes. Diagnostic
reports identify development parameters, set Filecoin acceptance false and leave
the ceremony-ID field empty.

Eight focused outer tests pass. They cover unchanged legacy/Filecoin selection,
two-anchor verification, strict setup binding, fresh-versus-retained A/B/A proof,
packet and profile parity, direct eager-prover parity, and wrong claims, seeds
and pairing anchors. The initial direct-comparison fixture needed fixed matrices;
that correction and a temporary-directory collision are preserved with the failed
attempts. Neither required a runtime prover change. Two command checks reject
Filecoin mode or a CPU proving backend before output creation.

The [exact command and environment](../experiments/kzg-cuda-validation/recursive-v4-proving-20261010/recursive-proving/report.json)
enable four devices, 8 GiB coefficient residency per card, resident SRS caching,
and distributed lookup/quotient. The small saved packet can be verified without
starting the GPU prover. Copy it to a new directory because verification writes
its own timing report:

```bash
cp -a experiments/kzg-cuda-validation/recursive-v4-proving-20261010/outputs/request-1 \
  /tmp/kzg-v4-verify-replay
MULTI_STARK_KZG_SETUP=development \
MULTI_STARK_KZG_BACKEND=cpu \
MULTI_STARK_INIT_EXPECTED_CLAIMS=experiments/kzg-cuda-validation/recursive-v4-proving-20261010/root-claims.bin \
experiments/kzg-cuda-validation/recursive-v4-proving-20261010/init-kzg-wrap \
  dev-v4-verify \
  /tmp/kzg-v4-verify-replay
```

This is a valid known-trapdoor v4 diagnostic from the same saved inner proof, with
two fresh outer witnesses and proofs. It establishes the actual recursive-stage
cost and development packet size. The integrated measurement below additionally
regenerates FRI and the first-stage witness and proof. Authentic Filecoin parameters,
both regenerated keys, real-constant recounting and an authenticated integrated
run remain necessary.

### Integrated development-v4 worker pipeline

The [integrated report](../experiments/kzg-cuda-validation/integrated-v4-workers-20261010/report.json)
measures **122.455 seconds** from fresh aggregation-root compression through the
verified final packet. Both complete retained workers are alive throughout this
request. It includes a newly compressed FRI proof, fresh source/scalar and recursive
assignments, trace generation, commitments, openings, native verification and
negative checks, intermediate proof export, and device quiescence after each KZG
request. No saved witness or coefficient bootstrap supplies a fresh-request proof.

| Timed fresh-request phase | Seconds |
| --- | ---: |
| FRI compression, including its process startup and frontend | 57.173 |
| First KZG stage, including fresh witness and GPU release | 30.691 |
| Recursive KZG stage, including fresh witness and GPU release | 34.554 |
| Export, artifact recording and other orchestration | 0.037 |
| Complete fresh request | **122.455** |

The first worker's genuine preparation request takes **234.832 seconds (3m55s)**,
and the recursive worker's takes **169.434 seconds (2m49s)**. Including worker
startup and orchestration, preparation totals **404.353 seconds (6m44s)**. Each
worker compiles and lowers its verifier circuit, loads its cached development
SRS, generates cold fixed preprocessing and proving keys, and generates and
verifies a full warmup proof. Frontend preparation accounts for 57.941 and
18.420 seconds respectively within those requests. Stage one's warmup uses
the preserved FRI seed; recursion wraps that newly generated first-stage proof.
The workers then retain their host circuit plans and keys for fresh requests,
releasing GPU caches between phases. This preparation does not include fresh
FRI compression or development SRS generation.

Preparation plus the subsequent fresh request takes **526.810 seconds (8m47s)**.
That sum includes two warmup proofs followed by another complete root-to-packet
request; it is **not a measurement of one cold request**. Worker shutdown takes
another **7.637 seconds**, followed by independent CPU verification in **0.364
seconds** for stage one and **0.164 seconds** for recursion. The complete external
driver takes **535.832 seconds (8m56s)**. The 122.455-second result is a prepared
service latency. A single cold request remains unmeasured on this version, and
a cold-start pipeline below five minutes has not been established.

The actual final proof is **1,909 bytes** and the complete packet **2,181 bytes**:
32 profile bytes, 144 claim bytes, 96 pairing-point bytes and the proof. The
recursive layout remains one `2^27 × 3` computation trace and one `2^17 × 1`
merged table, with **118,634,362** unpadded computation rows and no sharding.
The freshly generated **38,229-byte** first-stage proof exactly matches the
previous valid development-v4 bootstrap. Fresh FRI proof/key/claim bytes match
the preserved compressor fixture; both stages' proof/packet/profile bytes match
their warmups; the final artifacts also match the prior recursive measurement.
Both separate CPU verifiers pass all native and negative checks using two G1
anchors without loading a large SRS cache.

All four GPUs record positive compute events in both KZG stages. The recursive
lookup and quotient run across all four, taking **4.154 seconds** and **4.598
seconds** inside the recursive phase. Mean sampled utilization ranges from
20–25% across cards during stage one and 12–30% during recursion. These phase
means include witness work and transfers; event durations overlap and are not
additive elapsed-time savings.

FRI uses the primary GPU and peaks at **96,731 MiB (94.464 GiB)** on GPU 0 while
the two complete KZG workers remain alive. The recursive phase peaks at
**70,688 MiB (69.031 GiB)** on a card. After each KZG request, all four cards
report zero tracked coefficient, SRS-point and MSM-workspace bytes. The retained
host footprints after preparation are about **144.256 GiB** for stage one and
**102.336 GiB** for recursion. The sampled sum of process RSS peaks at **449.249
GiB** during the fresh request; it is a two-second sample and can double-count
shared pages. FRI's own measured peak is **204.102 GiB**. This verifies coexistence
with both full workers, beyond the earlier small-context fixture experiment.

The integration uses the existing JSONL worker protocol, fresh-request checks,
staging/proving cores and GPU-release paths. An explicitly constructed
[`KnownTrapdoorPublicDegree`](../examples/support/kzg_setup.rs) setup source supplies
the same development seed and full degree to both workers. `dev-v4-serve` requires
explicit `MULTI_STARK_KZG_SETUP=development`; `dev-v4-verify` selects diagnostic
verification. Both public-degree paths enforce strict setup markers and the
`2^27` cap. Normal Filecoin and legacy environment selection remain unchanged.
The [benchmark harness](../experiments/kzg-cuda-bench.py) exposes this only through
`--development-public-degree --setup development --retained-workers`, rejects
combination with `--acceptance`, and checks matching neutral public-setup IDs
across warmup, fresh and CPU reports. Diagnostic reports identify the known
trapdoor and leave ceremony/receipt fields empty.

The [exact command and environment](../experiments/kzg-cuda-validation/integrated-v4-workers-20261010/process-lifecycle.json)
select all four KZG devices, resident SRS caching, the standard stage-one budget,
and an 8 GiB/card coefficient cap with distributed consumers for recursion.
All 96 default CPU threads are available. Builds and tests finish before the
measurement. **48 distinct focused tests pass in 52 executions**: 24 Python
harness tests, seven first-stage setup/worker tests and 21 wrapper tests, with
four shared setup tests repeated in both Rust binaries. The archive retains
source and executable hashes, raw request logs, device events, GPU samples,
process-memory samples, all small proof artifacts and independent reviews.
The raw recursive `circuit-report.json` retains a stale frontend-only
`outer_proof_generated: false` field. The generated proof, `prove-report.json`
and separate verification establish successful outer proving. A subsequent
[report-only correction](../experiments/kzg-cuda-validation/integrated-v4-workers-20261010/post-measurement/source-delta.patch)
sets that field after successful worker proving and passes a 6.572-second
incremental CUDA wrapper build. The archive preserves the measured sources,
executables and raw outputs, with the correction recorded separately.

The analyzer rechecks the ordinary acceptance predicates against this run. The
only unmet predicate is **authenticated Filecoin parameters**; development
provenance deliberately fails it. Real ceremony artifact acquisition, regenerated
keys, real-constant sizing and the authenticated run remain the next acceptance
steps. The new result supplies a complete development measurement, not a sum of
the earlier isolated stage timings.

### Feed fixed preprocessing directly into fused proving

Both fused KZG stages now generate each cold fixed matrix directly from the
same compiled circuit as the fresh witness, immediately before committing that
partition. They record dimensions during staging and omit the intermediate
`.fixed.zst` encoding, compression, disk write, read and decoding. Plain
checkpointed staging still writes its raw fixed traces. Small circuit metadata,
setup commitments and fixed coefficient checkpoints retain their formats and
are published to the trusted local cache only after successful verification.

The [targeted report](../experiments/kzg-cuda-validation/fused-fixed-input-20261010/report.json)
compares a **480 MiB** synthetic fixed matrix (`2^20` rows, 15 columns) across
five alternating disk/direct pairs. Median generation plus handoff falls from
**0.578065 to 0.086197 seconds**, a **6.71×** improvement for that operation.
Both timers include generation. The disk path also includes canonical encoding,
`pzstd` compression, dropping the original matrix, reading, decompression and
decoding. Full matrix equality and the canonical checksum are checked outside
the timers. Each compressed file contains 15,515,084 bytes.

The fixture uses small signed coefficients, sequential/permuted copy labels
and sparse selectors. It does not reproduce the complete recursive circuit.
Filesystem cache state is uncontrolled and the read follows the write in the
same process. The complete comparison takes **4.143 seconds**, with **998.875
MiB** peak process RSS; that peak includes both leg outputs retained for parity.
These numbers do not establish a production-stage speedup, a lower production
memory peak or an updated full-pipeline time. The 122.455-second integrated
request and its 404.353-second preparation above predate this change.

**Eight focused correctness tests pass**, covering exact metadata, fixed
coefficient, setup, proof, packet and profile parity with checkpointed staging;
cache publication and restoration; fresh legacy and public-degree worker
requests; and malformed dimensions or stale raw files. Fresh geometry checks
run before parameter loading. The generated definition uses the existing
namespace, partition index, trace cap and degree policy. All three CUDA binaries
build successfully; the timed handoff runs after builds and tests finish, with
all 96 default CPU threads available.

This keeps one generated fixed matrix at a time. `System::new` still makes its
existing transient preprocessing copy, and stage one keeps its bounded next
witness prefetch. Loaded coefficient keys, setup bindings and cache identity
rules remain intact. Fixed caches include the executable digest, so rebuilding
selects a new cache namespace. Production startup and authentic Filecoin
acceptance still require measurement with the actual ceremony parameters.

### Build table multiplicities in parallel

The [isolated comparison](../experiments/kzg-cuda-validation/lookup-multiplicities-20261010/report.json)
measures the candidate's **32,587,682 lookups** over four native-table shapes.
Across five alternating serial/automatic pairs, median counting time falls
from **4.202520 seconds to 0.066370 seconds**, a **63.3× speedup** for this
operation. Every sample produces identical complete count vectors and their
canonical checksum. The lookup distribution is synthetic, with 25 range, three
nibble, three XOR and one split query per 32 lookups, reusing 65,536 assignment
values. It does not measure the production wrapper's much larger assignment
or establish a recursive-stage saving.

The timer includes allocation, scanning, reduction and field conversion.
Fixture construction, final-result destruction and parity checks are excluded;
construction takes 1.637 seconds separately. The whole diagnostic takes
28.111 seconds and peaks at 2.347 GiB RSS. It uses all 96 default CPU threads
without a parallelism override, and runs separately from builds and other
benchmarks.

[`table_counts`](../src/plonkish/stark/multiplicities.rs) divides sufficiently
large lookup lists into contiguous chunks. Each chunk counts into a private
`usize` histogram over genuine table rows, and an in-place reduction combines
the counts before converting each count to a field element once. Table indices
still select the first occurrence of a duplicate row. Table order, individual
padding and merged-table padding remain unchanged. Lowering already ensures
that the field's prime order exceeds the complete layout's cell count, so the
integer counts embed exactly. Assignment validation remains in place.

The planner limits chunks by the default Rayon pool size, a 65,536-lookup work
threshold, initialization/reduction work and a 64 MiB temporary-buffer
budget. It includes argument buffers and table offsets in that budget; allocator
metadata and existing circuit/output storage are separate. The recursive
candidate's four tables have 65,824 genuine rows and admit all 96 workers with
50,566,696 bytes of planned temporary storage. Each chunk allocates at most one
histogram, and reductions reuse an operand rather than allocating another.
Smaller jobs, one-worker configurations, unsupported sizes or unavailable
temporary storage retain the original serial field-add loop.

Twelve distinct focused tests pass, with nine repeated under a build without
the `parallel` feature: both fields, duplicate and unused rows, counts above
65,535, chunk boundaries, budget/overflow handling, merged and unmerged table
layouts, partitioned layouts, assignment ownership and validation error order.
Small FRI and KZG fixtures verify and preserve exact proof bytes. The KZG
fixtures use explicit development parameters; their sizes are unrelated to
the final ceremony packet. All three CUDA pipeline executables are rebuilt
with the change; no GPU kernel or complete-chain timing is claimed here.

The individual-table `TraceShards::trace` API still computes all tables before
selecting one. Callers requesting several unmerged tables separately can repeat
the lookup scan. Avoiding those repeated scans is separate work; the recursive
merged-table path performs one scan.


## FRI memory placement

[GoldilocksBlake3Config](../src/types.rs) binds a prover to one CUDA device.
The verified primary-only compression diagnostic now takes **73.171 seconds**,
down from 160.166 seconds on the same preserved root. Correct async-allocation
release accounting and producer-stream ordering first reduced it to 121.168
seconds by admitting all nineteen GPU quotient jobs. Parallel eager lookup
materialization and Plonky3's existing zero-allocation primitive then removed
another 47.997 seconds (39.61%). Proving now takes 28.532 seconds. Proof,
verifier-key, and public-claim bytes remain identical. These are isolated
FRI-phase observations, not a full-pipeline measurement.

[The FRI PCS](../src/cuda/pcs.rs) keeps resident data on its primary GPU.
Host-backed LDE jobs can use reserved auxiliary devices through
`MULTI_STARK_CUDA_AUX_DEVICES=1,2,3`, or
`CudaDft::with_auxiliary_devices`. This is opt-in: the listed CUDA ordinals
must be unique, exclude the primary device, and be reserved by the caller.
They follow `CUDA_VISIBLE_DEVICES`. Each auxiliary device serializes its
jobs; clones share its worker and immutable sppark plans. Admission counts
the complete input, output, panel scratch, coset constants, and free-memory
reserve before any transform allocation. Inadmissible matrices retain the
CPU path; admitted CUDA failures propagate.

The offload covers urgent, deferred, remaining, and all-host fallback LDE
jobs. It returns canonical, bit-reversed host matrices in the original
matrix order, so the combined Merkle commitment and transcript do not
change. It does not distribute resident FRI codewords, Merkle trees, lookup
construction, or quotient evaluation across devices. Deferred materialization
still recomputes a transient LDE; retaining those evaluations on another
device would avoid that work and the host transfer.

Large resident-LDE downloads now share the existing per-device pinned
transfer ring with uploads. A leased ring holds four 16 MiB chunks, queues
DMA ahead of host copies, and drains before reuse by another thread. This
avoids registering and unregistering a multi-gigabyte output allocation.
The pool retains at most four 64 MiB slots per device; the serialized
auxiliary path normally needs one slot per device. The transfer change also
applies to large downloads without auxiliary devices enabled.

The isolated diagnostic on 2026-10-09 used all 96 host CPUs and the four RTX
PRO 6000 Blackwell devices. Timings include GPU upload, computation, host
output allocation, download, and device release; input construction and
the final element-by-element comparison are outside the timed interval.
These are single measurements of representative original Init shapes,
not established spill shapes from the compression proof: its saved log
did not record matrix dimensions.

| Host-backed LDE workload, blowup 4 | CPU | GPU, whole-output registration | GPU, bounded ring |
| --- | ---: | ---: | ---: |
| One `2^20 × 533` matrix, 16.66 GiB output | 2.071 s | 4.164 s | 1.620 s |
| Three concurrent `2^19 × 129` matrices, 6.05 GiB output total | 0.454 s | 1.334 s | 0.368 s |

For the wide matrix, download time fell from 3.807 to 1.260 seconds; upload
and sppark computation took 0.356 seconds with the bounded ring. Every
output element matched the CPU reference. Small parity tests also cover
matrix ordering, arbitrary cosets, lazy input representatives, repeated
ring wraps, a partial final chunk, shared worker ownership, and restoration
of the caller's CUDA device. These results establish a local improvement,
not a reduction of the 171.686-second compression phase or the full pipeline.
Capture `MULTI_STARK_CUDA_MEMORY_LOG=1` on the next compression measurement
to identify actual spill shapes and the remaining critical path.

Reproduce the narrow diagnostic without KZG stages:

```sh
MULTI_STARK_CUDA_ARCHS=120 cargo build --release --locked --offline \
  --features parallel,cuda --example cuda_host_lde_bench
MULTI_STARK_CUDA_AUX_DEVICES=1,2,3 MULTI_STARK_CUDA_MEMORY_LOG=1 \
  MULTI_STARK_CUDA_BENCH_SHAPES=20,533,2 \
  target/release/examples/cuda_host_lde_bench
MULTI_STARK_CUDA_AUX_DEVICES=1,2,3 MULTI_STARK_CUDA_MEMORY_LOG=1 \
  MULTI_STARK_CUDA_BENCH_SHAPES=19,129,2 MULTI_STARK_CUDA_BENCH_COPIES=3 \
  target/release/examples/cuda_host_lde_bench
```

Raw timings and machine metadata are in
[`fri-aux-lde-20261009`](../experiments/kzg-cuda-validation/fri-aux-lde-20261009/).

Before the allocator correction, a FRI-compression-only diagnostic used the identical snapshotted
binary and preserved root input for both legs. With the bounded download
ring enabled in both, primary-only compression took 160.166 seconds and
auxiliary LDE offload took 157.166 seconds. Setup was 8.443 versus 6.699
seconds; proving was 115.455 versus 114.152 seconds. This single ordered
pair shows a modest 3.000-second difference (1.9%), including setup
variance, so auxiliary offload remains opt-in. Proof, verifier key, and
public claims matched each other and the historical saved artifacts
byte-for-byte.

The actual spills were six preprocessing matrices of `2^24 × 17`, followed
by main matrices of `2^23 × 3`, two of `2^23 × 32`, and two of `2^23 × 40`,
all with blowup four. The auxiliary leg returned 87.75 GiB across these
eleven LDEs. GPU 0 still peaked at 95,257 MiB; auxiliary peaks were 15,631,
17,551, and 17,551 MiB. This confirms that the principal remaining limitation
is outside these offloaded transforms. Full logs, per-second GPU samples,
artifact hashes, and the one-leg replay script are in
[`fri-aux-compression-20261009`](../experiments/kzg-cuda-validation/fri-aux-compression-20261009/).

The same logs expose a larger admission problem: the first of nineteen GPU
quotient jobs requests 14.49 GB while reported free memory stays at 3.89 GB
after many LDE evictions. That rejects the entire GPU quotient batch. A
256 MiB allocator diagnostic reproduces the cause on this driver:
`cudaFree` on a `cudaMallocAsync` allocation leaves the pool's used-byte
counter charged even after device synchronization. Reusing those physical
pages can then report more used bytes than reserved bytes. In a fresh
process, `cudaFreeAsync` followed by stream synchronization correctly returns
the used-byte counter to zero over three reuse cycles. The existing memory
query already adds unused pool reservation to driver-free memory; it cannot
recover capacity from the stale used counter. The focused correction frees
async-owned LDE buffers asynchronously and observes completion before
admission rechecks, preserving the pool retention policy. Allocation-counter
evidence is in
[`fri-admission-20261009`](../experiments/kzg-cuda-validation/fri-admission-20261009/);
a three-cycle release/admission regression and the existing auxiliary LDE
parity tests pass.

An intermediate run with only corrected release accounting admitted all
nineteen GPU quotient jobs but failed verification with
`OodEvaluationMismatch`. Its 121.167-second elapsed time and 76.420-second
proving time remain archived as **failed diagnostics, not a speedup**.
Mixed quotient execution had no dependency from selector generation and
metadata uploads on the per-thread producer stream to its two nonblocking
consumer streams. Large selector generation can outlive the first bounded
host-pack operation. Both consumers now wait on a producer event and drain
before temporary storage is released, including error paths.

The local reproducer runs concurrent CPU-reference parity fixtures at the
default test parallelism. It covers every host/resident input combination,
small and full-field challenge coordinates, all nineteen saved circuit
graphs at sixteen and 256 rows, and one graph at `2^20` rows with a `2^22`
quotient domain spanning three staging chunks. Two fixtures failed before
the event and all three passed afterward. The subsequent primary-only phase
replay verifies and matches the original 1,418,855-byte proof, verifier key,
and public claims byte-for-byte; altered public claims are rejected.

| Verified FRI-only measurement | Before admission/order corrections | Admission/order corrected | Parallel lookups and zero allocation |
| --- | ---: | ---: | ---: |
| Total elapsed | 160.166 s | 121.168 s | 73.171 s |
| FRI proving | 115.455 s | 75.697 s | 28.532 s |
| Completed GPU quotient jobs | 0; whole batch fell back | 19 | 19 |
| Peak host RSS | 232.47 GiB | 204.02 GiB | 204.34 GiB |
| GPU 0 peak memory | 95,257 MiB | 96,893 MiB | 96,893 MiB |

The first quotient job's reported free capacity rises from 3.89 GB to
14.49 GB after six evictions. In the latest replay, the nineteen quotient
jobs take 7.204 seconds in total, including their local admission and
materialization; lookup jobs take 10.401 seconds and streamed FRI opening
takes 4.938 seconds. These are
host elapsed intervals, not device-kernel-only measurements. Auxiliary LDE
offload was disabled, so its earlier 3-second difference must not be added
to this result without a new comparison. Binary/source identities, complete
logs, GPU samples, the failing and passing fixture commands, and golden
artifact hashes are in
[`fri-stream-order-20261009`](../experiments/kzg-cuda-validation/fri-stream-order-20261009/).

### Parallel eager lookup materialization

[SystemWitness::from_stage_1](../src/system.rs) retains circuit order and
materializes each circuit's independent lookup rows in parallel. Indexed row
writers borrow disjoint slices of the final buffers, with reusable expression
scratch per worker and a minimum split of 4,096 rows. Partial ranges retain
the full trace's selector indices and next-row wraparound. There is no
per-row writer vector: removing its 48-byte entries saves 768 MiB of
temporary metadata at `2^24` rows without enlarging the final payload.

Parallel evaluation alone left a serial buffer-initialization cost. The
[Plonky3 field adapter](../src/p3_adapter/field.rs) now delegates `zero_vec`
to the upstream field implementation. Goldilocks can then request zeroed
integer storage directly instead of filling a field-element vector in
userspace. This uses the pinned upstream primitive and preserves its field
layout and zero-length behavior.

The lookup-only diagnostic uses saved production graph 0, with `2^20` rows,
three main columns, seventeen fixed columns, ten lookups, and twenty-five
lookup-expression nodes. Values are deterministic synthetic inputs, and
timing includes final-buffer allocation. Five alternating samples per binary
give these medians:

| Row evaluation | Zero allocation | Elapsed |
| --- | --- | ---: |
| Serial | Generic field fill | 243.969 ms |
| Parallel | Generic field fill | 92.575 ms |
| Serial | Upstream primitive | 183.511 ms |
| Parallel | Upstream primitive | 7.421 ms |

The final row sweep is 24.73× faster than serial evaluation with the same
allocator, and the combined change is 32.88× faster than the original local
fixture. Exact lookup payloads match. Focused tests cover Goldilocks and BLS
scalar fields, row ranges and wraparound, selectors, argumentless and empty
lookups, circuit ordering, global LogUp totals, and identical small proof
bytes. Both parallel and serial feature configurations pass. Commands,
source identities, immutable test binaries, and separate pre/post-allocation
samples are in
[`fri-lookup-rows-20261009`](../experiments/kzg-cuda-validation/fri-lookup-rows-20261009/).

One affected-phase replay measures the combined change against the verified
121.168-second result: total FRI compression is 73.171 seconds and proving
is 28.532 seconds. The new `Lookup expressions materialized` timer reports
**2.283 seconds across all nineteen circuits**. GPU lookup and quotient
routing is unchanged, and the original 1,418,855-byte proof, verifier key,
and claims match exactly; altered claims are rejected. Peak host memory is
essentially unchanged at 204.34 GiB because the final lookup payload remains
materialized. The binary, fifty-six source/dependency inputs, preserved root,
logs, GPU samples, and parity hashes are archived in
[`fri-lookup-materialization-20261009`](../experiments/kzg-cuda-validation/fri-lookup-materialization-20261009/).
This is one phase observation per version; no whole-chain speedup is inferred.

The next FRI targets are lowering (14.787 seconds), trace construction
(11.065 seconds), and the lookup stage (10.401 seconds). Omitting eager host
payloads requires a lazy consumer that can materialize them when residency
admission selects the concrete path: eleven large jobs still need that path.
With the eager sweep at 2.283 seconds, this is primarily a memory reduction
opportunity until another phase measurement establishes a larger latency gain.

### Bound compact-hash copy-cycle storage

The 73.171-second replay predates this follow-up; its phase total has not been
remeasured. Compact-hash lowering previously collected a separate vector of
`(trace, row, slot, label)` tuples for each word. At the observed 118,978
compressions, the arithmetic recipes alone create 54,253,968 words and
162,761,904 occurrences. That implies at least 54.25 million small allocations
and 4.85 GiB of tuple payload, plus 1.21 GiB of vector headers, before spare
capacity and input/boundary words.

[Compact-hash lowering](../src/plonkish/hash.rs) now retains one 24-byte
endpoint record per word: its first label and the location of its last
successor cell. The same trace/row/slot traversal assigns labels; each new
occurrence fills its predecessor's successor cell, and a final pass closes
the cycles in word order. Unused words require no writes. Repeated slots,
singletons, cross-trace links, and input/output boundary aliases retain the
same fixed columns. That diagnostic keeps matrix allocation and trace packing
unchanged, isolating the representation change.

The local diagnostic contains 2,048 compressions, 966,668 words, and 2,842,636
occurrences. Five alternating samples reduce copy-cycle construction from
**126.523 to 18.555 milliseconds (6.82×)** and eliminate 966,668 small
allocations. Three focused tests compare every fixed cell with an occurrence-list
reference for Goldilocks and BLS scalars, then verify fresh FRI proof bytes are
identical. They pass in 0.44 seconds; the complete diagnostic takes 1.07 seconds.
Commands, binary/source identities, timings, and parity results are in
[`hash-copy-endpoints-20261009`](../experiments/kzg-cuda-validation/hash-copy-endpoints-20261009/).

### Field allocation and parallel hash row packing

Large main, fixed, table-multiplicity, and hash buffers now use `F::zero_vec`.
Goldilocks reaches the pinned Plonky3 zero-allocation primitive through the
field adapter; scalar fields retain the existing safe generic fallback. Tiny
per-row scratch allocations are unchanged. Hash advice packing writes indexed,
disjoint 4,096-row tiles into the final matrix. Row order, byte embeddings,
carries, rotations, histogram order, and zero padding are unchanged. There are
no intermediate trace copies or additional allocation primitives; builds
without `parallel` use Plonky3's serial iterator fallback.

An isolated allocation-plus-packing diagnostic uses hash kind 1, width 40,
three padding rows, and an 80 MiB output. Goldilocks has 2^18 padded rows; BLS
scalars have 2^16. Five interleaved samples give:

| Field | Serial packing and generic allocation | Indexed tiles and field allocation |
| --- | ---: | ---: |
| Goldilocks | 13.684 ms | **1.247 ms (10.98×)** |
| BLS scalar | 10.414 ms | **5.754 ms (1.81×)** |

The Goldilocks serial intermediate with field allocation measures 11.086 ms.
Scalar allocation still uses the same fallback, so no scalar allocation gain
is claimed. The fixture uses deterministic prepared words and measures only
construction; it does not include witness preparation, fixed construction,
commitments, or a full FRI phase. The complete diagnostic takes 0.425 seconds
and peaks at 192.74 MiB host memory.

All nine hash/table traces match the serial reference cell for cell for empty
inputs, small padded inputs, carry overflow patterns, and 4,096-row tile
boundaries, for both fields. Fresh FRI proof bytes match the serial trace and
occurrence-list references. Existing dense/lazy partition and native hash
boundary checks pass, with and without `parallel`. Baseline and final sources,
immutable binaries, commands, samples, and results are archived in
[`hash-trace-packing-20261009`](../experiments/kzg-cuda-validation/hash-trace-packing-20261009/).

The verified 73.171-second FRI result predates these construction follow-ups;
their aggregate phase effect remains unmeasured. The trace-construction timer
also includes all fixed circuit inputs and dropping the logical circuit.
The pinned Arkworks 0.5 scalar implementation has no
equivalent zero-vector allocation helper; any further allocation experiment
should include subsequent writes and retain safe allocation semantics.

### Select only the required row output

Sharded lowering and lazy main-trace generation now request only logical
wires; fixed-partition generation requests only fixed cells. A shared row
classifier uses a compile-time selector, preserving gate/public/lookup/hash
ordering, global copy labels, partition activation, and padding. Callers no
longer allocate scratch for the discarded output.

This work survived compiler optimization before the change. Archived
Goldilocks and scalar production binaries call the shared routine out of
line; the scalar routine has four field-conversion callsites. The isolated
reference retains the same call boundary, code sizes, and conversion pattern.
The compiled wires-only scalar routine contains none of those conversions.

A valid mixed fixture has 165,038 used rows, padded to 262,144, with width 5,
33,224 gates, 34 publics, 131,392 lookups, and five compact hash calls. Five
interleaved, allocation-inclusive samples give these copy-link sweep medians:

| Field | Both row outputs | Wires only |
| --- | ---: | ---: |
| Goldilocks | 5.755 ms | 5.498 ms (4.5% lower) |
| BLS scalar | 9.153 ms | **5.793 ms (36.7% lower)** |

This sweep includes endpoint allocation, row scanning, and cycle closing;
it excludes hash definitions and the rest of lowering. Whole main-trace and
fixed-matrix timings overlap, so this fixture establishes no improvement for
those paths. The complete diagnostic takes 0.671 seconds and peaks at
378.06 MiB. No phase or pipeline saving is inferred.

Both fields match every wire, fixed cell, successor, main-trace cell, and
claim against the original row routine and independent dense lowering.
Multiple partitions, partial tiles, padding, empty hash inputs, and chained
hash boundaries are covered. Fresh dense/lazy FRI proof bytes and existing
partition/proof regressions match with and without `parallel`. Independent
source review also passes.
Sources, immutable binaries, disassembly, commands, raw samples, and results
are in [`row-layout-selection-20261009`](../experiments/kzg-cuda-validation/row-layout-selection-20261009/).


## GPU trace and witness design

The initial execution changes preserve circuit relations, trace layout,
public statements, transcript order, and proof format. Compiled layouts and
execution schedules are reusable preprocessing; assignments and trace values
remain specific to each proof.

### What the FRI implementation provides

Aiur's CUDA trace writers expand compact seeds derived from finalized query
records into device rows. Their inputs already contain the execution results
needed to regenerate those rows. They provide a useful model for deterministic
generation, seed residency, bounded tiles, padding, and regeneration after
eviction.

See the [Aiur device writer](../../ix/crates/aiur/src/trace_codegen/cuda.rs)
and [seed preparation](../../ix/crates/aiur/src/trace_codegen.rs) in the sibling
`ix` checkout, reviewed at `91348190f091d53ab0838440eaeaf0cb36f2737f`.
These links assume the usual sibling `ix` and `multi-stark` repositories.

In this repository, [TraceGenerator and TraceSource](../src/witness.rs)
connect producers to [the FRI device commitment path](../src/cuda/witness.rs).
The device callback and [DeviceTraceView](../src/cuda/mod.rs) currently use
Goldilocks values represented by one `u64`. KZG needs BLS12-381 scalar values
represented by four limbs, with the Montgomery representation expected by
the existing CUDA adapter. The producer interface and commitment integration
therefore need an explicit scalar-field implementation.

The measured FRI compression example is a separate Plonkish path:
[ix_root](../examples/ix_root.rs) calls `witness.generate()` and
`compiled.traces()` on the host before CUDA proving. Its traces do not already
use the Aiur device writers. The same Plonkish trace-generation work could
eventually benefit that example as well.

### Generate KZG columns directly on the device

The proposed data flow is:

```text
Measured path
CPU assignment -> host trace rows -> compressed checkpoint
               -> host decode and transpose -> GPU iFFT -> GPU MSM

Proposed path
CPU assignment + reusable layout -> GPU trace columns -> GPU iFFT -> GPU MSM
```

1. Compile an immutable layout describing wire indices, public and activation
   rows, padding, partition boundaries, and auxiliary trace placement. Reuse
   that layout for compatible proofs. Upload the assignment or compact seeds
   needed by each partition and gather directly into column buffers.
2. Add a KZG commitment entry point for generated device columns. The existing
   `Config::commit_main` interface accepts trace sources, but its default
   implementation materializes them on the host. A KZG override alone is
   insufficient for the example programs: both their checkpointed and fused
   paths still pass host row matrices to the PCS. Extend that boundary to
   carry generated device columns without materializing those rows first.
3. Connect those buffers to resident interpolation and MSM. Preserve the
   coefficient recovery source needed by host consumers, checkpointing, and
   spill handling. Host coefficient mirrors can remain initially; direct
   trace input does not require making the entire prover device-only.
4. Extend generation to auxiliary traces. Arithmetic wire gathering is a
   small first target, but a complete stage-one path must also handle compact
   BLAKE3 traces and table multiplicities. Hash kernels must emit the circuit's
   intermediate trace values, and multiplicities must match the exact table
   indexing. A host fallback permits incremental coverage.
5. Fuse trace production with commitment in the normal execution path while
   retaining explicit checkpoint/resume support. Writing every generated row
   back to the existing trace files and reading it again would retain much
   of the traffic this design aims to remove.

The initial benefit is avoided materialization and movement of expanded
traces. Assignment uploads, layout/seed uploads, host checks, and coefficient
downloads must still be included in measurements.

Generation tiles bound producer workspace; they do not remove the full
column allocation needed by the current FFT implementation. A scalar column
at `2^27` rows is 4 GiB. Admission must account together for assignment seeds,
layout data, trace buffers, coefficients, SRS chunks, and FFT/MSM scratch.
Coordinate ownership and buffer lifetimes with the bounded prefetch and
partition overlap described in the [bounded partition pipeline](#bounded-partition-pipeline).

### Generate the scalar witness on the GPU

#### First stage

[GoldilocksCircuit::translate](../src/plonkish/foreign.rs) imports every
nonconstant source value as an input to the scalar circuit. The Goldilocks
assignment has already been computed, so much of the remaining work derives
canonical limbs, range-check advice, and modular-reduction quotients from
known values. This exposes parallelism across source values and gates.

Start with typed operations for those recurring patterns and batch independent
instances into kernels. Keep their exact output wire placement and arithmetic
bounds. Cache the dependency schedule and layout per compatible circuit;
execute them against fresh input values for every proof. Avoid a kernel
launch per gate, and group local dependent operations where possible.

The present [HintDefinition](../src/plonkish/builder.rs) stores an opaque
Rust closure. A general scheduler cannot translate that closure to CUDA or
serialize it as an executable plan. Add explicit operation metadata for
supported hints, with CPU fallback at planned boundaries for the remainder.
Keeping a compiled circuit in a long-lived process is an earlier option than
persisting a portable witness program to disk.

Assignment checking is part of the recorded witness cost. Profile generation
and checking separately, including the additional `check_values` call in
`trace_shards`. GPU generation alone does not remove those host sweeps.
Parallel or device checks must preserve the existing failure semantics and
continue to rely on proof constraints to enforce the relation.

#### Recursive stage

The recursive circuit additionally needs foreign BLS12-381 base-field
arithmetic, quotient/carry advice, inverses, and curve-operation intermediates.
Several hints use host `BigInt` arithmetic. See
[native field advice](../experiments/kzg-wrap/src/native_field.rs) and
[curve advice](../experiments/kzg-wrap/src/native_curve.rs).

Implement typed kernels for these operations and schedule independent work
around the circuit's dependencies. An existing GPU MSM supplies a final
point; the recursive witness also needs the intermediate values constrained
by its circuit. Producing those values requires more than substituting a
library MSM call.

The [recursive fixed profile](../experiments/kzg-wrap/src/saved.rs) currently
binds the public statement because constant gates remain after public slots
are rebound. Preserve that identity until the builder and its invariance
across statements have been corrected and validated. Retain the
[fixed cache's executable binding](../examples/support/kzg_fixed_cache.rs)
when comparing implementations. A reusable execution plan must bind every
constant that affects its outputs, as well as circuit layout, field
representation, and operation versions. Cache the plan; compute a fresh
assignment for each proof.

### Questions for review

- Which live compiled objects and evaluated fixed matrices give the greatest
  repeat-proof time reduction within the host-memory budget?
- Can the full-public-degree argument support the mixed-height candidate
  identified by the migration sizing report?
- Should device trace input extend the shared generator interface with typed
  field/layout metadata, or use a dedicated KZG column source first?
- Which assignment/layout data should stay resident across partitions, and
  when are compact seeds cheaper than gathering from a complete assignment?
- Should the first witness implementation specialize the Goldilocks lifting
  patterns or introduce a general typed witness operation representation?
- Can assignment checks be shared between generation and trace construction
  without weakening diagnostics or accepting an unchecked assignment?
- How much recursive-witness work should precede the
  [terminal backend decision](plonk-terminal-backend-review.md), which may
  replace the second KZG wrapper? Trace production and first-stage scalar
  advice have broader reuse than that wrapper's curve-specific witness.


## Migration to a 2^27 trace ceiling

This migration plan now has a measured development implementation; authenticated
Filecoin acceptance remains pending. Keep the current two-stage KZG
architecture and **one recursive computation trace plus the existing merged
table**. Wrapper sharding is excluded because it increases the final proof.
The wrapper must actually fit within `N = 2^27 = 134,217,728` used rows;
reject an oversized layout rather than splitting it. All other committed
matrices must also fit the cap. The measured reference sizes are **2,053 bytes
for the outer proof and 2,709 bytes for the complete packet**, from the
[historical full report](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/report.json).
The [integrated development-v4 result](#integrated-development-v4-worker-pipeline)
measures a 1,909-byte proof and 2,181-byte packet at the smaller cap.
Measure both for each candidate and reject complete packets of 3,000 bytes or more. Trusted-setup
deployment additionally requires resolving the degree policy below and
importing authenticated parameters into both stages.

The cap applies to computation traces, fixed matrices, lookup accumulators,
auxiliary hash traces, merged tables and committed quotient slices. It does
not cap temporary evaluation domains: with quotient degree two, an `N`-row
trace can require a `2^28` FFT while each committed slice still has length
`N`. Preserve that transform support; stop designing for `2^28` base traces.

### Decide the public-setup degree policy first

The [ceremony description](https://github.com/arielgabizon/perpetualpowersoftau)
and [parameter definitions](https://github.com/filecoin-project/powersoftau/blob/master/src/parameters.rs)
give `M = 2^28 - 1` public G1 powers and `N = 2^27` G2 powers. Working with
only a prefix does not remove the remaining powers from an adversarial
prover. Under the repository's [degree policy](pcs-abstraction.md), a
length-`n` polynomial needs a shifted commitment with shift `M - n`, and
the matching G2 power. For power-of-two `n <= N`, that power is present only
when `n = N`.

Consequently the existing mixed-height profile cannot be made production
ready merely by reducing the wrapper and importing the first `N` powers.
The original `Srs::max_len()` conflated stored G1 length, the trace cap
and the degree-check range. The public-policy implementation separates these
concepts and retains power-of-two honest prefixes. Even an
`N`-row trace needs a shifted commitment against the public range `M`; the
current branch that omits shifted commitments at `max_len` would change.

Choose and document one supported protocol before finalizing production
commitment counts:

| Route | Required work | Effect on the layout plan |
| --- | --- | --- |
| Keep explicit per-trace degree bounds | Supply the necessary full-range degree-check keys or another reviewed degree-bound proof | Filecoin's documented G2 powers only cover the existing shifted construction when every committed trace is `N` tall; mixed-height support needs a different mechanism |
| Prove soundness with the full public degree allowance | Analyze AIR identities, LogUp, quotient recombination and opening batching together; version the resulting protocol and add adversarial tests | Selected under the explicit model below; retains efficient mixed heights |
| Change the commitment/terminal backend | Select a backend with an applicable setup and degree argument, then measure the complete replacement | A separate architecture decision; not an implicit consequence of this migration |

The selected mixed-height Filecoin policy uses the argument below, v4
transcript binding and an authenticated importer. Retain current subgroup
checks and limb bounds throughout that work. The subgroup-removal and 96-bit
limb proposals are optional follow-ups, not prerequisites silently assumed by
the `2^27` plan.

### Sizing and degree-policy report

**The metadata report prioritizes the full-public-degree policy while keeping
the current mixed stage-one heights.** Removing shifted checks offers most
of the wrapper reduction without increasing stage-one matrix sizes. Uniform
`2^27` padding exceeds the earlier reference sizes and multiplies first-stage
matrix work. This frozen report predates construction and protocol tests;
implementation evidence is recorded below separately.

The [machine-readable report](../experiments/kzg-cuda-validation/trace27-sizing-20261009/report.json)
records input/source SHA-256 hashes, all circuit dimensions, conditional
candidate counts and log-based row attribution. The
[metadata counter](../examples/kzg_profile_counts.rs) reads the preserved
manifests and compiled circuits; the
[report generator](../experiments/kzg-sizing-report.py) combines those counts
with the preserved recursive build log. No SRS, witness, physical trace or
candidate recursive circuit is constructed. The counter reproduces both
saved compact proof sizes exactly: **49,557 and 2,053 bytes**. That checks
the counting model against existing artifacts; it does not verify those
proofs again.

#### Exact preserved shape

Widths below are per circuit. `Lookup` is the committed accumulator width,
after grouping; `Q` is the number of committed quotient slices. Grouping is
two for the arithmetic/hash circuits and one for the tables.

| Stage-one indices | Kind | Height | Main | Fixed | Lookup | Q |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0–8 | Nine arithmetic partitions | `2^24` | 3 | 17 | 5 | 2 |
| 9 | Merged table | `2^17` | 1 | 4 | 1 | 1 |
| 10, 12 | Wide hash | `2^22` | 32 | 14 | 9 | 2 |
| 11, 13 | Wide hash | `2^22` | 40 | 14 | 11 | 2 |
| 14 | Hash auxiliary | `2^19` | 12 | 8 | 6 | 2 |
| 15 | Hash auxiliary | `2^21` | 4 | 18 | 6 | 2 |
| 16 | Hash table | `2^16` | 1 | 3 | 1 | 1 |
| 17 | Hash table | `2^9` | 1 | 4 | 1 | 1 |
| 18 | Hash table | `2^8` | 1 | 1 | 1 | 1 |
| **Column totals** | **19 circuits** | | **191** | **247** | **101** | **34** |

This gives **573 unshifted commitments**, plus 330 shifted commitments
(236 witness, 94 fixed), and nine opening witnesses. The recursive verifier
allocates **571 witness curve inputs and 341 fixed curve inputs**, then
exposes ten pairing points through 20 public scalar values alongside the
18 Init words. Its opening MSM has 583 terms. Stage one opens 674 scalar
values and transports 18 intermediate accumulators. Main/fixed matrices
have no next-row openings in this profile; lookup matrices always have two.
All 19 circuits stay active in the candidates. The 34 stage-one claims
contain 127 scalar fields: nine arithmetic and six hash activation claims,
the public-value sentinel and all 18 words. Preserve that statement.

The current outer profile is a `2^28` computation trace with widths
`(main, fixed, lookup, Q) = (3, 15, 4, 2)` and a `2^17` merged table with
widths `(1, 4, 1, 1)`. Candidate byte counts below assume these widths
remain unchanged and the computation fits a single `2^27` trace. An actual
layout count must establish that assumption.

Retuning every saved graph at both 48 and 96 bytes per column commitment,
with quotient budget two, selects **the same lookup groups and widths**.
Uniform heights alone therefore do not remove lookup columns. Temporary
quotient domains reach `2^25` in stage one and `2^28` in the proposed
outer computation trace.

#### Padding and final-byte costs

Dense GiB count one 32-byte scalar representation of every main, fixed,
lookup and quotient matrix. These are total matrix volumes, not peak
memory, transferred bytes or time estimates. Fixed-key caching changes
when some of this work is paid.

| Candidate | Stage-one GiB | Both stages GiB | Inner proof bytes | Outer proof bytes | Packet bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved reference, outer `2^28` | 154.852 | 346.879 | 49,557 | 2,053 | 2,709 |
| Uniform `2^24`, development degree convention | 286.500 | 382.527 | 37,893 | 2,053 | 2,325 |
| Mixed heights, hypothetical full-public-degree policy | 154.852 | 250.879 | 38,229 | 1,909 | 2,181 |
| Uniform `2^24`, hypothetical full-public-degree policy | 286.500 | 382.527 | 37,893 | 1,909 | 2,181 |
| Uniform `2^27` in both stages, strict full-range shifted checks | 2,292.000 | 2,416.000 | 53,541 | **2,437** | **2,805** |

These are exact arithmetic for the stated shapes, not generated candidate
proof sizes. The full-public-degree rows assume removal of shifted checks
in **both** stages. The strict row retains shifted commitments for every
matrix, including `N`-row matrices, and pads the outer table too; it exceeds
the outer-proof budget by 384 bytes and packet budget by 96 bytes. This
rejects that particular layout/codec combination, not every possible
degree-bound construction.

Uniform `2^24` padding adds **131.648 GiB (85.0%)** to stage one:

| Matrix kind | Preserved GiB | Uniform `2^24` GiB |
| --- | ---: | ---: |
| Main | 31.943 | 95.500 |
| Fixed | 84.772 | 123.500 |
| Lookup | 27.975 | 50.500 |
| Quotient | 10.162 | 17.000 |

It increases combined matrix volume by about 10.3% even after halving the
outer domain. Under the same full-public-degree policy, uniform padding
only removes seven additional opening witnesses compared with mixed
heights: 328 versus 335 witness curve inputs, and 576 versus 583 terms in
the main opening MSM. That marginal recursive saving must justify the
extra 131.648 GiB; it is not a reason to select uniform padding by default.

The byte model is `5 + 48 * witness_points + 32 * scalar_fields`, including
opening witnesses and excluding fixed commitments from the proof. The
packet adds 32 profile bytes, 144 Init bytes and 48 bytes per external
pairing point. Both no-shift candidates expose two pairing points and
22 recursive public scalars. Wrapper sharding is absent from every row.

#### What the existing row profile supports

| Baseline component | Used rows attributed by the build log |
| --- | ---: |
| Curve-input allocation and encoding | 62,164,376 |
| Transcript | 25,945,039 |
| AIR checks | 5,604 |
| Shifted-commitment MSM | 35,559,560 |
| Seven unshifted degree-group MSMs | 38,307,138 |
| Main opening MSM | 57,292,158 |
| Opening-witness MSM | 1,353,567 |
| Final padding row | 1 |
| **Total** | **220,627,443** |

Consecutive 64-point allocation blocks establish **108,864 rows per witness
curve input**, including subgroup checks and canonical encoding. Removing
shifted checks at mixed heights removes 236 such inputs; uniform `2^24`
additionally removes seven opening witnesses, for 243 inputs total.

Subtracting those inputs and the 73,866,698 rows logged inside the eight
degree MSMs leaves 121,068,841 rows for mixed heights or 120,306,793 for
uniform heights. **Those subtotals are not valid final circuit counts.**
[`Base::table`](../experiments/kzg-wrap/src/native_msm.rs) caches each
point's table across MSMs. The remaining opening MSM must rebuild the
tables for 236 unshifted witness points whose first use moves out of a
deleted degree MSM. Eight doublings and seven additions per table cost
approximately **3.25M rows** in total, using the existing affine-cost
estimates. Fixed-point table allocation, infinity selection and constant
interning also need recounting.

That gives a planning subtotal of **about 124.3M rows for mixed heights**
or **123.6M for uniform heights**, before reduced transcript work and other
MSM/scalar changes. These are neither measurements nor upper bounds. The
older ~116M estimate and its claimed 18M-row margin did not follow from this
arithmetic. The subsequent [exact count](#decision-and-next-evidence) measures
118,634,361 used rows with complete MSMs and the compiled frontend's claim
schema; real ceremony constants still need
recounting. Subgroup checks and five 80-bit limbs remain intact.

#### Degree-policy findings

Let `M = 2^28 - 1` be the public G1 power count and `D = M - 1`. The
published G2 exponents run from zero through `N - 1`. For strict length
`n`, the existing shifted construction needs exponent `M - n` in G2.
For power-of-two `n <= N`, only `n = N` has that key. This follows from
the [ceremony parameter definitions](https://github.com/filecoin-project/powersoftau/blob/master/src/parameters.rs)
and its [power-27 description](https://github.com/arielgabizon/perpetualpowersoftau).

A prefix-relative check is demonstrably insufficient. Choose a working
prefix length `L < M` and claimed length `n < L`. The out-of-bound
polynomial `p(X) = X^n` has commitment `[tau^n]G`; its shifted commitment
is `[tau^L]G`, available in the public tail. The check with key
`[tau^(L-n)]H` passes because both pairing exponents are `tau^L`.
For `n = L`, the current verifier omits the shifted check altogether.
This disproves that proposed degree bound; by itself it is not a forged
AIR proof. Include this case in the eventual protocol tests.

There is a concrete reason to investigate a full-degree identity argument.
Conditionally assume an extractor fixes each witness and quotient-slice
polynomial with degree at most `D` before the evaluation challenge, and
fixed polynomials retain degree at most `n - 1`. Propagate degrees through
the saved constraint DAG: addition takes the maximum, multiplication adds,
first/last selectors have degree `n - 1`, and the transition selector has
degree one. Apply the same propagation to the grouped LogUp product and
multiplicity terms. Recombining `q` quotient slices gives degree at most
`D + (q - 1)n`; multiplying by `X^n - 1` gives `D + qn`.

For the preserved 19 stage-one graphs and the two outer graphs evaluated
at heights capped at `2^27`, the resulting identity-degree bounds sum to
**14,227,407,510**. For fixed nonzero identities and a uniform independent
evaluation challenge over BLS12-381 Fr, the union root bound is about
`2^-221.13`. This is only the polynomial-identity term, **not a security
claim for the protocol**. It suggests the field-size margin is not the
obvious obstacle to this route.

The protocol argument below covers knowledge/extraction under the complete
ceremony parameters (including auxiliary alpha/beta powers), challenge
ordering and Fiat–Shamir adaptivity, alpha cancellation, LogUp fingerprint
collisions and zero denominators, and batched openings at multiple points.
It must show that identities on each original trace domain still imply
the intended statement even when committed polynomials exceed that domain's
interpolation degree. Legacy v3 retains explicit degree bounds; v4 uses the
full public degree allowance under that argument's stated model.

#### Full-public-degree soundness argument

**The mixed-height construction supports a soundness argument in the generic
bilinear-group model with a random oracle, using the full public degree
`D = 2^28 - 2`. It does not require a separate degree bound for each trace.**
This is a protocol-specific argument in an idealized model, not a claim that
ordinary KZG evaluation binding alone proves the resulting SNARK secure in
the standard model. A concrete-curve security claim must state the group
model, random-oracle assumption and authenticated-ceremony assumption.

The distinction matters: [Chmel, Hubacek and Stejskal](https://eprint.iacr.org/2026/284)
show that the usual AGM knowledge-soundness notion for polynomial commitments
does not establish immediate extraction. The argument below tracks the
**entire protocol**, including when each polynomial is fixed, instead of
invoking that PCS property as an online extractor.

**Public inputs to the group model.** Let `T` denote the hidden ceremony
tau, and `U,V` its independent alpha and beta secrets. Give the adversary
all published powers, including G1 powers through `T^D`, G2 powers through
`T^(N-1)`, the G1 sequences `U T^j` and `V T^j` for `j < N`, and `V` in
G2. These are the sections of the
[Filecoin accumulator](https://github.com/filecoin-project/powersoftau/blob/master/src/accumulator.rs).
G2 and GT operations do not map back to G1 in this type-3 group model.
An adversary's G1 handle therefore has a formal exponent of the form
`P(T) + U A(T) + V B(T)`, with `deg P <= D` and
`deg A, deg B < N`, before accounting for other public or sampled handles.

The model must also include the public update history and independently
sampled group points; it must not silently discard them. The
[contribution public keys](https://github.com/filecoin-project/powersoftau/blob/master/src/keypair.rs)
contain random bases and their tau/alpha/beta multiples. Under a validated
history with an honest contribution that sampled and erased independent
secrets, take its tau update as an indeterminate. Earlier accumulators are
independent of that indeterminate; subsequent power sequences have degree
at most `D` in it, and the public-key terms have degree at most one.
Specialize the other setup indeterminates to constants and rescale the
tau variable so the final ordinary powers map to `1,T,...,T^D`.
Independent sampled handles can be assigned additional indeterminates.
This extends the degree bound to the complete modeled transcript. An
unvalidated ceremony, retained toxic waste, or arbitrary additional
tau-dependent auxiliary information is outside this argument.

Generic group operations maintain formal exponent expressions. Except for
the generic collision event, an accepted pairing equation is a formal
polynomial identity in these indeterminates. Apply a ring homomorphism
`Phi` that preserves the final ordinary powers and sets the auxiliary
indeterminates to constants; for the final accumulator alone, set
`U = V = 0`. Every G1 commitment and opening witness then has a projected
polynomial of degree at most `D`. Applying `Phi` to a valid formal identity
preserves it. This step neither reveals alpha/beta nor asserts that they
are zero in the real setup. It also does not assume auxiliary components
of individual commitments vanish. The projected polynomials provide the
extracted trace; fixed commitments project to their known fixed polynomials.

**Commitments and challenges.** The required order, implemented by the
[native verifier](../src/verifier.rs),
[PCS](../src/ark_adapter/pcs.rs) and
[recursive verifier](../experiments/kzg-wrap/src/native_verifier.rs), is:

1. Bind the version, setup identity, public degree allowance, trace cap,
   circuit shape, activation, fixed/main commitments, heights and
   length-prefixed claims; then sample the lookup challenges `beta,gamma`.
2. Bind lookup commitments and all intermediate accumulators; then sample
   constraint-folding challenge `alpha`.
3. Bind each quotient-slice commitment; then sample `zeta`.
4. Bind the opening-round shape and every claimed opened scalar; then
   sample column-batching challenge `v`.
5. Bind every opening witness; then sample point-batching challenge `r`.

In the generic model a handle's formal expression is fixed when it is
created. Thus the main trace is fixed before the lookup challenges, the
lookup trace before `alpha`, and the quotient slices before `zeta`.
Adaptively chosen later polynomials do not change those earlier handles.
Fiat-Shamir grinding is accounted for by the number of random-oracle
queries; the independent-challenge bounds below are not final
noninteractive security bounds.

**Batched openings.** For point batch `b`, write its point as
`z_b = g_b zeta`, its projected commitments as `P_bj(T)`, its claimed
values as `y_bj`, and its projected opening witness as `W_b(T)`. The
projected final pairing equation is

```text
sum_b r^b [sum_j v^j (P_bj(T) - y_bj) - (T - z_b) W_b(T)] = 0.
```

The expressions in square brackets are fixed before `r`. Unless all are
zero polynomials, a random `r` cancels them with probability at most
`(number_of_batches - 1) / |Fr|`. For a zero bracket, substituting
`T = z_b` eliminates its witness and leaves
`sum_j v^j (P_bj(z_b) - y_bj) = 0`. These discrepancies are fixed before
`v`; an incorrect opened value survives with probability at most
`(batch_width - 1) / |Fr|`. This allows the opening witness to depend on
`v` and `zeta`. The same argument works for the distinct rotations caused
by mixed trace heights.

Every used committed column, including **each quotient slice**, is opened
explicitly at the common random `zeta`. There is no hidden quotient
linearization whose polynomial changes with that point. This also permits
a special-soundness route: distinct `r` forks separate point equations,
distinct `v` forks recover individual opening equations by Vandermonde
inversion, and sufficiently many `zeta` forks permit interpolation.
[Lipmaa, Parisella and Siim](https://eprint.iacr.org/2024/994) establish the
relevant Batch-KZG special-soundness technique. Applying their
standard-model route here would require establishing its PCS hypotheses
for the **augmented Filecoin setup**; their minimal-SRS assumption cannot
simply be cited as covering all the extra G2 and auxiliary powers.

**AIR identities and trace extraction.** For circuit `i` with domain
`H_i` of size `n_i`, let `C_i(X)` be its folded constraint polynomial.
Let its `q_i` projected quotient slices be `Q_ij(X)`. The tested identity is

```text
R_i(X) = C_i(X) - (X^n_i - 1) sum_j X^(j n_i) Q_ij(X) = 0.
```

Each `Q_ij` may have degree `D`; the honest prover still uses degree below
`n_i`. Thus the quotient side has degree at most `D + q_i n_i`.
Propagating `D` through witness columns and actual selector/fixed degrees
gives the `B_i` bounds recorded in the sizing report. Since all these
polynomials are fixed before `zeta`, a false identity passes with
probability at most `B_i / |Fr|`. On `H_i`, the quotient side vanishes
regardless of its slice degrees. Before `alpha`, any violated row supplies
a nonzero polynomial in `alpha` of degree at most `k_i - 1`, where `k_i`
is that circuit's constraint count. Consequently folding cannot hide a
violated individual constraint except with probability
`(k_i - 1) / |Fr|`.

The witness is the collection of projected main polynomials evaluated on
their **original** domains. Their degree need not be below the domain size
for these row values to exist or satisfy the AIR. Replacing a polynomial
by its remainder modulo `X^n_i - 1` preserves current and next-row values
on that domain. Keep fixed matrices, their heights and the canonical
circuit/profile authenticated; the prover cannot choose a different table
or interpret it on another height.

**LogUp, multiplicities and activation.** Off the zero-denominator event,
each grouped LogUp constraint divides to the required sum of
`multiplicity / message`. Summing over groups and rows telescopes the
lookup trace. The normalized last-row injection binds that sum to the
declared incoming/outgoing accumulators. Chaining all active circuits,
the public-claim accumulator and the checked final zero therefore yields
the intended global rational identity.

For lookup messages fixed before `beta,gamma`, unequal weighted message
multisets give a nonzero rational function. Clearing denominators gives
the usual degree-dependent root bound. If `L` is the total number of row
and public message terms and `W` bounds their fingerprint width, a
conservative field-error allowance is `(W + 2)L / |Fr|`, including poles.
This requires unambiguous channel/arity encodings and a nonzero imbalance
**in Fr**. Integer-count semantics additionally require the existing
multiplicity/range invariants; field equality alone does not rule out
counts differing by the field characteristic. The
[LogUp analysis](https://eprint.iacr.org/2022/1530) explains this distinction.
Removing degree checks does not permit dropping those invariants.

Activation must be bound before lookup challenges, with at least one
active circuit and the canonical widths for every active slot. Inactive
circuits contribute no rows. The application must continue to bind its
required activation/public-value messages: generic sparse-proof validity
does not independently require every configured circuit to be active.
For the recursive profile, retain its all-active shape and the Init claim
locations. Native final verification must check both external pairing
points produced by the recursive circuit against the same setup anchors.

**Completeness and implementation conditions.** The honest prover's
interpolated columns and quotient slices still have degree below their
trace height, so the original FFT, quotient splitting and MSM algorithms
remain complete under the available prefix. No padding to `D + 1` is
needed. As with the existing protocol, lookup poles and evaluation points
on a trace domain are negligible challenge failure events; this is
overwhelming completeness, not a promise that every possible challenge
works. Explicitly reject visible zero claim denominators, `zeta = 0` and
`zeta` on an active trace domain in native verification, matching the
recursive inverse constraints. Hidden row poles belong in the statistical
error term; checking public denominators does not eliminate them.

Use a distinct protocol version and bind the public degree allowance and
setup identity. Keep subgroup/canonical-encoding checks, exact commitment
and opening shapes, public claims and accumulator checks. Reject shifted
commitments in the no-shift format rather than silently ignoring them.
The `2^27` trace cap remains an independently checked resource/profile
limit, not a claimed algebraic bound on a malicious committed polynomial.

The recorded `2^-221.13` is only `sum_i B_i / |Fr|` for the preserved
shapes. A complete bound also includes constraint and opening batching,
LogUp terms and poles, degenerate evaluation points, generic-group
collisions, random-oracle queries and hash/encoding failures. The generic
collision bound itself depends on the number and degrees of all modeled
public handles and adversarial operations. It is not a concrete security
estimate for BLS12-381. The original FRI statement's security also remains
part of the end-to-end guarantee. The argument supports implementing and
testing this policy under the stated model; it does not constitute an
independent cryptographic audit or a standard-model security theorem.

#### Decision and next evidence

The selected implementation keeps the existing mixed stage-one heights and
uses full-public-degree v4 in both stages. `PublicSetup` separates the full
ceremony degree/identity from `Srs::max_len()`, which now describes loaded
G1 powers. Both native and recursive verification omit shifted checks under
this policy and bind the same v4 transcript. Legacy v3 remains explicit.

The exact shared-builder census after the MSM completeness repair is
**118,634,361 used rows**, comprising 86,046,656 gates, 32,587,682 lookups,
22 public values and the sentinel. It pads to one `2^27` computation trace
and leaves **15,583,367 rows (11.61%)** spare. The two remaining MSMs have
583 and 9 terms and expose two pairing points. See the
[current construction report](../experiments/kzg-cuda-validation/recursive-worker-20261009/census/report.json).
This count uses the actual ceremony ID and stage-one `2^24` cap, but preserved
development commitment coordinates; recount real ceremony constants before
accepting the production layout. Counting allocates no full IR or witness.
The recount took 31.152 seconds and removes sixteen unused public-claim
constant gates from the [previous complete-MSM census](../experiments/kzg-cuda-validation/trace27-complete-msm-20261009/report.json).
Lookup counts, MSM terms and padded domains are unchanged. This is a profile
compatibility result, with no claimed proving-time improvement from those
sixteen rows.

The affine MSM formerly excluded valid exceptional intermediates and identity
outputs. A constrained subgroup offset now has a deterministic retry hint;
the circuit still checks every affine inverse and cancels the offset
algebraically. A complete final group operation supports identity results.
The additional approximately 922,000 rows fit inside the measured margin.
Subgroup checks and five 80-bit field limbs are retained. The projected
**1,909-byte outer proof / 2,181-byte packet** now matches the
[actual development-v4 proof](#complete-development-v4-recursive-proving).
A proof using the real ceremony parameters remains unmeasured.

Targeted checks cover genuine high-degree commitments that restrict to valid
trace values, invalid AIR identities with otherwise valid KZG openings, mixed
heights with anchor-only verification, altered degree/setup/trace profiles,
unexpected shifted commitments, malformed points and transport, public
denominator failures, and MSM identity/duplicate/opposite-point cases.
Native verification now explicitly rejects visible poles consistently with
the recursive verifier. Nondegenerate legacy proof bytes are unchanged.

The targeted v4 CPU/CUDA fixture uses two traces of `2^15` and `2^16` rows,
streaming lookup/quotient evaluation and partition pipelining. Both backends
produce the same 1,205-byte compact proof with BLAKE3
`8164e8f1acee3f94dbb48d29908d4e99085af60f3a8a96f4d09b8c6405c69bca`;
both decode and verify it. CUDA event logs show work on all four cards.
This fixture uses known-trapdoor test points with the public-degree policy;
it establishes backend parity, not authenticated ceremony provenance or
production-size throughput. The staged/resumed and retained-worker fixtures
also pass with explicit development setup selection.

Remaining gates are:

1. Obtain and authenticate the actual `challenge_19` bytes, import the required
   prefix, and regenerate both stages' fixed commitments. The original host
   was unreachable from the benchmark machine during provisioning; no local
   development cache substitutes for this step.
2. Finish integrated setup/cache tests and within-profile CPU/GPU proof parity;
   recount the wrapper with the real constants and enforce the single-trace
   cap before allocating physical traces.
3. Measure affected stages, then benchmark a fresh full chain with the selected
   setup and explicit startup/warm timing boundaries. Continue GPU residency,
   witness and FRI work against the measured critical path.

The narrow hash redesign remains optional. Its theoretical main-column savings
are insufficient evidence without fixed, lookup and routing column counts.

The five-minute target still needs independent performance work. The latest
839.988-second run spends 171.686 seconds in FRI alone, leaving only
128.314 seconds for every other phase in a 300-second pipeline. The cap
and the conditional row savings do not establish that budget. Keeping
stage-one padding out of the critical path is the report's strongest
performance conclusion.

Reproduce into a **new** artifact directory:

```bash
MULTI_STARK_CUDA_ARCHS=120 cargo build --release --locked --offline \
  --features parallel,kzg-cuda,cuda --example kzg_profile_counts
python3 experiments/kzg-sizing-report.py /tmp/kzg-sizing-report-new
```

The recorded run used the existing build cache: the narrow helper build
took 0.74 seconds. A cold environment must budget the build separately.

### Count the single-wrapper candidates

Use the preserved 19-circuit profile. Count all commitment rounds, fixed and
main cells, lookup groups, degree groups, opening points, public/activation
claims and packet bytes. The report above completes the metadata counts;
the shared-builder census also counts the candidate wrapper. Physical witness
generation and a recount with actual ceremony coordinates remain. Apply the
selected degree policy consistently to both stages.

| Candidate | Stage one | Recursive computation trace | Decision value |
| --- | --- | --- | --- |
| Preserved reference | Existing mixed heights, maximum `2^24` | Measured 220.6M used rows padded to `2^28` | Comparison only; fails the new cap |
| Mixed heights without shifted checks | Keep the current stage-one dimensions | Exact builder census 118,634,361 used rows including complete MSMs and the typed claim schema; real ceremony coordinates still need recount | Selected; fits one `2^27` computation trace without padding stage one |
| Uniform stage-one heights | Pad every stage-one trace to `2^24` | Planning subtotal ~123.6M before transcript and other changes; actual count pending | Development-protocol comparison; production applicability depends on the degree policy |
| Narrower hash traces | Redesign the wide compact BLAKE3 traces | Requires a new combined count | Candidate for reducing stage-one movement and recursive commitment cost; count all new routing/fixed/lookup columns |

The four wide hash traces alone add **54 GiB of main-trace cells** when
padded from `2^22` to `2^24` at a total width of 144 scalar columns. Fixed,
lookup and quotient work add further cost. That is storage arithmetic, not
a peak-memory or runtime prediction. Reaching one `2^27` wrapper therefore
does not by itself make uniform padding the fastest end-to-end choice.

Use uniform `2^24` as a comparison, not the selected production layout.
The report does not establish the older ~116M prediction or its claimed
headroom. Exact counting must include MSM table reuse and the actual
transcript under the chosen protocol. Keep subgroup checks intact and
select using total first-stage plus wrapper cost and the final packet limit.
If the mixed-height argument fails, choose an alternative degree-bound
construction and recount; the uniform `2^27` candidate with the current
columns/codec exceeds the historical reference sizes and greatly increases
matrix work. Wrapper sharding stays excluded.

### Count a saved proof with the Filecoin setup

[`count-saved`](../experiments/kzg-wrap/src/saved/count.rs) counts the actual
saved first-stage proof and its fixed commitments. It shares the recursive
worker's validated `Plan` and typed statement schema. The historical `count`
command still serves development-profile comparisons; `count-saved` requires
Filecoin mode so counting cannot silently generate a development SRS.

With the [imported cache and pinned receipt](#parameter-import-and-artifact-migration)
configured, run:

```sh
MULTI_STARK_KZG_SETUP=filecoin \
MULTI_STARK_INIT_EXPECTED_CLAIMS=/path/to/root-claims.bin \
target/release/init-kzg-wrap count-saved \
  /path/to/filecoin-stage-one /path/to/recursive-census.json
```

The loader authenticates the setup, checks the independently expected statement,
validates circuit metadata and checked proof length before allocating the codec,
and records
hashes of the exact input bytes consumed. The Filecoin receipt authenticates
the SRS. Supply AIR metadata and fixed commitments from the trusted first-stage
build; recording their hashes alone does not establish the intended Init
circuit. Counting retains small fixed gadget
tables and AIR metadata, without constructing the full recursive gate IR,
assignment, physical traces or outer proof.

Admission requires one computation trace and one merged table, each at most
`2^27`, with the table also fitting the computation domain. Shared lowering
relations and lookup tuning feed the compact codec's exact size calculation;
the report includes the 32-byte profile, 144-byte statement and all pairing
points in the projected packet. The limit is strictly **below 3,000 bytes**.
An oversized counted layout produces a report and a failing exit status.

`filecoin_census_passed` means the authenticated input, counted layout and codec
projection pass these gates. `actual_outer_proof_generated` remains false.
Actual ceremony constants, a generated and verified final packet, and the
fresh-witness full-chain timing remain required acceptance evidence.

The [focused validation report](../experiments/kzg-cuda-validation/saved-census-20261009/report.json)
records 23 wrapper tests, two shared AIR/codec tests, CPU/CUDA executable builds
and three CLI rejection checks. The real input count remains pending the
authenticated ceremony artifact. Tests cover exact metadata/codec parity,
raw merged-table padding, over-cap rows, packet limits, malformed dimensions
before allocation, and unchanged compatible-statement cache identities.

### Implementation sequence and acceptance checks

| Step | Concrete change | Evidence required before proceeding |
| --- | --- | --- |
| 1. Finish policy and exact layout counting | Use the completed metadata report; settle the degree argument and add count-only wrapper construction; inspect `multi_stark_layout()` without generating assignments or physical traces | Actual used/padded rows, one computation trace at most `2^27`, and final proof/packet counts under the reviewed policy |
| 2. Reduce stage-one commitments as needed | Apply the chosen degree construction; add narrower hash traces or coherent height controls only where the complete count justifies them | Preserve hash relations, active-row selectors and table multiplicities; CPU/GPU parity within each new profile; no added outer traces |
| 3. Enforce the single-wrapper limits | Reject wrapper layouts above `2^27` before lowering/allocation; retain one computation trace and the merged table; bind exact dimensions in the profile | Actual wrapper row count fits; complete packet below 3,000 bytes with proof/packet sizes reported separately; reject over-cap dimensions and unexpected trace counts; measure both affected stages |
| 4. Implement the selected setup policy | Separate public parameter capacity, admitted trace heights and degree-check keys; update both native and recursive verification and transcript/profile versions as required | Small adversarial examples using powers beyond the working prefix, altered degree/shift claims, missing G2 keys, and over-cap dimensions behave according to the reviewed argument |
| 5. Import and validate ceremony data | Build an explicit ceremony importer and a separate authenticated cache format; inject its setup into both KZG stages and verification | Pinned artifact/source identity and hashes; checked point decoding, subgroup/anchor/progression/key consistency; round-trip and malformed-input tests; no implicit development-setup fallback |
| 6. Regenerate affected artifacts | Generate fixed keys, codec/profile IDs, recursive circuit constants and fixtures for the selected setup/layout | Setup identity is bound through caches, worker reuse, packets and trusted verifier metadata; old/mismatched profiles fail explicitly |
| 7. Validate and tune the target shape | Run isolated lookup/quotient/MSM checks at representative sizes and affected-stage comparisons; do a final integrated run only after local gates pass | Fresh witnesses, independent verification and pairings, within-profile CPU/GPU parity, peak memory/transfer counters, separate startup and warmed timings, and a measured total against the 300 s goal |

Sizing can use explicit development parameters while the degree policy is
resolved; implement the selected production profile only after its design is
settled. Development-profile fit is not the production acceptance condition.
The existing `native-verify` command calls `Built::check_and_assign`; it is
not a count-only command. Separate construction/layout statistics from
assignment generation before using it for this workflow. If accurate wrapper
construction needs a proof for a changed stage-one profile, generate only
that affected fixture; FRI compression can reuse its preserved input.

The code changes are concentrated in these boundaries:

- [`stark.rs`](../src/plonkish/stark.rs) and
  [`hash.rs`](../src/plonkish/hash.rs): one coherent height policy shared by
  layout counts, preprocessing, witness traces and table multiplicities.
  `merge_table_traces(max_height)` currently imposes a maximum, not a minimum.
- [`init_fri_kzg_prove.rs`](../examples/init_fri_kzg_prove.rs) and
  [`init_kzg_worker.rs`](../examples/support/init_kzg_worker.rs): use the same
  profile for one-shot and retained requests; include setup and layout identity
  in compatibility checks and fixed-cache keys.
- [`outer.rs`](../experiments/kzg-wrap/src/outer.rs): retain the exact
  computation-plus-table shape; enforce the cap for both entries and reject
  an oversized computation layout before calling the sharded lowering API.
  Derive packet/public dimensions from the selected trusted profile and assert
  the final size budgets.
- [`saved.rs`](../experiments/kzg-wrap/src/saved.rs),
  [`native_verifier.rs`](../experiments/kzg-wrap/src/native_verifier.rs),
  [`srs.rs`](../src/ark_adapter/srs.rs),
  [`config.rs`](../src/ark_adapter/config.rs) and
  [`pcs.rs`](../src/ark_adapter/pcs.rs): replace development-seed loading with
  explicit setup selection and apply the selected degree policy consistently.

### Parameter import and artifact migration

Identify the exact finalized ceremony artifact and its authentication chain
before downloading it. The upstream
[accumulator format](https://github.com/filecoin-project/powersoftau/blob/master/src/accumulator.rs)
distinguishes uncompressed challenges from compressed responses with a
contribution key. Pin the format and required tau-power sections; do not
interpret the existing development cache as a ceremony file. Encoding
conversion needs checked fixtures rather than assuming arkworks and the
ceremony library serialize points identically.

The implemented importer pins the final Golem attestation's BLAKE2b-512 of
stock uncompressed `challenge_19`: 77,309,411,488 bytes. Filecoin uses this
phase-one output without an additional beacon; its provenance still assumes
at least one honest contribution. The importer reads and hashes the complete
artifact, validates every retained point's canonical coordinates, curve and
subgroup membership, and checks its tau progression against the G2 anchors.
The retained maximum prefix is approximately 12 GiB. Both import and generic
SRS progression checking use bounded-memory chunks.

The cache manifest binds the official source digest, public degree, anchors,
stored prefix and per-chunk BLAKE3 digests. Its receipt must be pinned externally
after a successful trusted import. Loads authenticate only needed chunks;
verifiers load the two G1 anchors and two G2 anchors from the authenticated
manifest. Reading a digest beside an untrusted cache is not authentication.
Eleven importer regressions cover encoding, subgroup/progression failures,
wrong receipts, corrupted metadata/chunks and deliberately skipped unused tails.

The 2026-10-10 [recovery metadata review](../experiments/kzg-cuda-validation/fri-worker-coexistence-20261010/filecoin-recovery-lead.json)
found a concrete preservation lead in the
[Slingshot Restore catalog](https://github.com/data-preservation-programs/slingshot/blob/a485ae2d38cc966deecb36995854d5ba05e4d589/recovery/cids/restore/filecoin-trusted-setup.csv):
five payload CIDs named `challenge_19.00` through `.04`. Its
[client registry](https://github.com/data-preservation-programs/slingshot/blob/master/recovery/files/client-list.json)
identifies TechGreedy/Xinan Xu, and the
[assignment sheet](https://docs.google.com/spreadsheets/d/1LWVndxGyegTdz5cPU86UZ5Y9vqN2n-VlK1kC0OeJHC8/edit)
names storage providers. The review found no reachable retrieval endpoint.
Historical storage assignments do not establish surviving copies; the catalog's
piece sizes and filenames do not establish reconstruction of the pinned source.
Any recovered bytes still require the complete source length/digest and import
checks above. No artifact was downloaded and no provider was contacted.

Provision once, outside the per-proof timing boundary:

```sh
target/release/examples/kzg_filecoin_import import /path/to/challenge_19 \
  /path/to/new-filecoin-cache 27
```

Set `MULTI_STARK_KZG_FILECOIN_CACHE` to that cache and
`MULTI_STARK_KZG_FILECOIN_DIGEST` to the trusted `manifest_blake3` receipt.
`MULTI_STARK_KZG_SETUP=filecoin` is the pipeline default. It fails when the
cache or receipt is missing; known-trapdoor runs require explicit
`MULTI_STARK_KZG_SETUP=development`. The same setup selection is used by
stage one, recursive staging, outer proving and independent verification.

After provisioning and targeted validation, run the integrated benchmark into
a new directory. `--fused-staging` feeds each stage's generated witness traces
directly into commitment and omits witness/main-checkpoint round trips:

```sh
python3 experiments/kzg-cuda-bench.py --setup filecoin --fused-staging \
  --distributed-wrapper \
  --filecoin-cache /path/to/filecoin-cache \
  --filecoin-digest "$MULTI_STARK_KZG_FILECOIN_DIGEST" \
  --fixed-cache /path/to/filecoin-fixed-cache \
  --root-artifacts experiments/kzg-cuda-validation/resume-20261009/root-artifacts \
  --output /path/to/new-filecoin-run
```

The harness records the receipt and source/binary identities and performs
independent CPU verification after the measured chain. A first run with new
ceremony keys populates fixed preprocessing; record it separately from warmed
measurements. Development proof bytes are not a baseline for this new profile.

`--distributed-wrapper` enables the distributed lookup and quotient consumers
only for the outer GPU process and sets its retained coefficient cap to
8 GiB per card before initialization. First-stage memory settings remain
independent. Every phase records its effective CUDA/KZG environment, including
the independent CPU verifier's backend. Unsupported geometry, topology or
memory budgets still use the consumer's existing fallback.

Both stages must use the authenticated setup. Rebuild the inner commitments
and recursive fixed constants when replacing the development SRS. Bind the
ceremony digest, point ranges, degree policy, layout and protocol version to
fixed-cache entries, retained-worker state and packet profiles. Verification
must use pinned setup metadata rather than keys supplied by the packet.
Keep historical artifacts readable through their explicit old profiles and
write new artifacts into distinct directories.

Changing layout or setup changes proof bytes. Preserve deterministic CPU/GPU
parity within the new profile and semantic/public-statement checks across
profiles; do not require equality with the old development proof. The 18 Init
words, external pairing checks and rejection of malformed statements remain
part of the acceptance boundary.

### GPU work after the layout decision

At `2^27`, one scalar column occupies 4 GiB and a prefix of `2^27` raw sppark
G1 points occupies 12 GiB. Those figures exclude shifted point ranges required
by the selected degree policy, FFT/MSM scratch and allocator reservations.
Fifteen fixed columns still occupy 60 GiB on one evaluation grid; the smaller
cap does not establish that all quotient inputs fit together on one card.

Resident quotient and lookup construction handle complete bounded partitions
on one GPU where admitted. Their local comparisons are
[7.96× for quotient](#resident-quotient-evaluation-on-admitted-traces) and
[4.32× for lookup](#resident-lookup-construction-on-admitted-traces), with exact
coefficient parity. The opt-in distributed quotient also has a
[5.41× target-size local result](#distributed-quotient-at-227), retaining the
explicit host fallback. Distributed lookup has a separate
[2.78× target-size local result](#distributed-lookup-at-227), using bounded
tiles and stock sppark scans/inversion. Keep selectors periodic and share
the memory budget across coefficients, SRS,
evaluations and scratch. Compare shared SRS chunks with residency after the
required point ranges are fixed. Tune for `2^24`–`2^27` base traces and retain
larger temporary quotient transforms only where the profile needs them.

#### Bounded quotient distribution for the `2^27` wrapper

The implemented opt-in path keeps one recursive computation trace and
distributes its evaluation work. It preserves the circuit, proof shape,
transcript and quotient coefficient order. The
[target-size diagnostic](#distributed-quotient-at-227) passed every coefficient
and measured 32.834 s to 6.066 s with the actual saved graph and synthetic
`2^27` inputs. The resident-SRS and FRI correctness gates have also passed.

The observed topology permits peer reads within ordinal pairs `(0, 1)` and
`(2, 3)`; cross-pair links report no peer-read support. Discover and validate
the directed peer-access matrix at runtime, retain stable CUDA ordinals in
the plan, and choose compatible pairs from that matrix. Do not hardcode the
observed ordinals or assume that four-card peer copies work. Small explicit
`cudaMemcpyPeerAsync` parity/transfer measurements validate the selected
directions on this host; unavailable peer transfers use bounded pinned host
staging.

The native distributed path must also configure access to each owner's
**current CUDA memory pool**. Sppark's `dev_ptr_t` allocates through
`cudaMallocAsync`, while the standalone diagnostic below used `cudaMalloc`.
Pool allocations require `cudaMemPoolSetAccess`; legacy
`cudaDeviceEnablePeerAccess` alone does not cover them. Validate parity on
actual pool allocations and retain the access setting for the pool's lifetime.
This distinction is documented in NVIDIA's
[stream-ordered allocator guidance](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/).

The [standalone transfer diagnostic](../experiments/kzg-cuda-validation/peer-transfers-20261009/report.json)
queried that directed CUDA matrix, enabled only supported directions, and
verified every byte of each 128 MiB destination after poisoning it. It used
one 128 MiB buffer per GPU and a two-slot, 16 MiB-per-slot portable pinned ring
for the unsupported `2 → 0` route. Every copy passed:

| Directed copy | Wall time for 128 MiB | Logical throughput |
| --- | ---: | ---: |
| Peer `0 → 1` | 2.416 ms | 51.74 GiB/s |
| Peer `1 → 0` | 2.381 ms | 52.49 GiB/s |
| Peer `2 → 3` | 2.407 ms | 51.94 GiB/s |
| Peer `3 → 2` | 2.530 ms | 49.41 GiB/s |
| Pinned ring `2 → 0` | 2.789 ms | 44.81 GiB/s |

These are one sequential sample per edge, including launch/completion
synchronization and both DMA legs for staging. Allocation, input preparation,
and parity downloads are outside the copy timings; the complete process took
1.37 s. The staged route transfers 128 MiB D2H plus 128 MiB H2D. Its 2.367 ms
download and 2.452 ms upload event intervals overlap, so their sum is not wall
time. Slot reuse waits for the destination-consumption event, and destination
DMA starts only after source production completes. The result validates these
bounded transfer primitives, without establishing simultaneous four-GPU
bandwidth or distributed-quotient correctness. Source, binary hashes, commands,
PCI identities, topology and raw output are archived with the report.

For the two-coset quotient, assign one coset to each pair. Split its full
column NTTs between the two cards and split its sweep rows in half. Each card
then gathers bounded row tiles from its own columns and its partner's columns,
while the existing sppark NTT and constraint DAG arithmetic remain unchanged.
Include one next-row halo, wrapping the final global row to row zero. Generate
the same unnormalized selector column per device and retain the scalar inverse
vanishing value per coset. Use global row indices for the domain point and
selectors; only column addressing becomes tile-relative.

The saved outer shape has 15 fixed, 3 main, and 4 lookup columns. Let `C` be
the actual number of nonconstant columns, validated from the selected profile,
and `S` the encoder's live DAG-slot count. With `C = 22`, `2^18` rows per tile,
and a separate outer worker capped at 8 GiB of retained coefficients per GPU,
the estimated per-device sweep peak is:

| Allocation | Per-device size |
| --- | ---: |
| Eleven owned coset-evaluation columns | 44 GiB |
| Retained coefficients | At most 8 GiB |
| Full first-selector column | 4 GiB |
| Half of one coset's quotient values | 2 GiB |
| Two all-column row tiles, including halos | About 0.344 GiB |
| DAG scratch, at 256 blocks of 128 threads | `S` MiB |
| Metadata and runtime/NTT headroom | Metadata plus 1 GiB |

This is about **59.35 GiB plus DAG scratch and metadata**, before any SRS or
MSM workspace that remains cached. With 25 nonconstant columns the analogous
worst card holds 13 columns and needs about 67.40 GiB plus scratch/metadata.
These are admission estimates, not measured peaks. The memory query must
charge live CUDA-pool pages, credit only the current pool's reported reusable
unused pages, and retain the existing quarter-VRAM reserve. Evict the
SRS/workspace cache as needed. The current half-VRAM
coefficient budget cannot be assumed compatible: coefficient handles are held
by `OnceLock` and are not evictable through the SRS cache. Select the smaller
outer-worker budget before loading coefficients, or implement a separately
reviewed coefficient-lifetime policy; changing an environment variable after
device initialization does not reclaim allocations.

Reserve every selected device atomically under the common lease mutex, with a
single stable lock order. Compute every phase's peak and check all devices
after any permitted cache eviction, before allocating native job buffers.
If any device fails admission, release the complete reservation and use the
bounded CPU path. Never hold one device while waiting to acquire another, or
enter Rayon/nested lease acquisition while the group is reserved. Build host
plans and output buffers beforehand. Unexpected CUDA runtime failures still
surface as errors rather than silently restarting proof work.

After both coset sweeps, synchronize readers and release the large evaluation,
selector, and tile buffers. Merge four 2 GiB partial outputs into the exact
`output[row * 2 + coset]` layout on one card, then use sppark's existing
`2^28` inverse NTT and download two contiguous `2^27` coefficient slices.
The selected merge card temporarily holds its own 2 GiB partial **and** the
8 GiB combined array, plus bounded copy/scatter buffers and retained data.
Each sender keeps its partial alive until its last transfer completes; release
events must cover those lifetimes. Account for this final phase independently,
including stream-ordered frees and allocator reuse, rather than assuming the
sweep's buffers vanish when host scopes end.

Only the other pair's **4 GiB of quotient output** must cross the unsupported
peer boundary in this design. A shared bounded pinned ring stages that as
4 GiB D2H plus 4 GiB H2D, with source-completion and destination-consumption
events protecting every slot. Within each pair, the dense `C = 22` case moves
about 44 GiB of remote evaluation tiles per coset, plus local gather copies.
The 8 GiB coefficient-retention cap can cause additional coefficient uploads:
record H2D, local D2D, peer D2D, cross-pair staging, and final D2H separately
and compare their complete totals against the existing path. The design is
useful only if the net transfer and wall-time measurements improve.

Small fixtures passed with both cosets, tile-boundary/final-row `Next`,
constant columns, deliberate admission rejection, forced host staging, and
exact coefficient plus fresh-proof-byte parity. The target-size quotient-only
diagnostic also passed and records peak memory and complete transfer totals.
An affected-stage or integrated run remains necessary to establish the total
pipeline saving.

#### Bounded LogUp distribution for the `2^27` wrapper

This implemented opt-in extension reuses the quotient path's atomic device
leases, pool-aware peer transfers and bounded pinned staging. The
[target-size diagnostic](#distributed-lookup-at-227) measured 19.124 s to
6.876 s with exact coefficients and total. The circuit, lookup grouping,
single computation trace, transcript and proof format stay unchanged.

Let `n = 2^27`, `C` be the actual nonconstant fixed/main column count, `G` the
lookup-group count, and `R = n/4`. For the documented shape, `C = 18` and
`G = 4`; validate these quantities from the circuit rather than embedding them
in admission. Duplicate original-domain evaluations across the two peer pairs,
splitting their complete column NTTs between each pair's two cards. Give the
four cards consecutive global row intervals of length `R`. Each pair then
serves half the rows, with each card gathering bounded tiles from its own and
its partner's columns. The column copies remain immutable until all tile
readers finish. Boolean first/last/transition selectors and the final `Next`
wrap use the **global** row index, including at device and tile boundaries.

Construct each partition's grouped numerator and denominator arrays in
row/group order. Preserve the existing zero-message convention by replacing
zero denominators by one and masking their numerator contributions. This
makes every stored denominator nonzero. Only these two arrays are needed
while reading the input evaluations; defer the forward/reverse product-scan
allocations until those reads finish and the full evaluations, tile buffers
and DAG scratch have been released.

With `2^18` rows per tile and the separate outer worker's 8 GiB retained
coefficient cap, the estimated sweep peak per card is:

| Allocation | Per-device size for `C = 18`, `G = 4` |
| --- | ---: |
| Nine owned original-domain evaluation columns | 36 GiB |
| Local numerator and denominator arrays | 8 GiB |
| Retained coefficients | At most 8 GiB |
| Two all-column row tiles, including halos | About 0.282 GiB |
| Lookup-prefix DAG scratch | `S_lookup` MiB at 256 blocks of 128 threads |
| Metadata and runtime/NTT headroom | Metadata plus 1 GiB |

The total is about **53.3 GiB plus DAG scratch and metadata**. General
admission uses `ceil(C/2) * n * 32` bytes for owned evaluations and
`2 * R * G * 32` bytes for the rational arrays. Once input evaluations and
scratch are freed, the four-array scan phase needs 16 GiB of arrays, plus
retained coefficients and headroom: about 25 GiB for this shape. These are
separate peaks, not measurements. Charge live allocations through the common
free-memory query, evict SRS/workspace entries where needed, retain the
quarter-VRAM reserve, and synchronize stream-ordered frees before crediting
their storage to a later phase.

Run the existing sppark forward/reverse `Multiply` scans and one local
total-product inversion independently on each partition. Their combination
recovers every local rational contribution; no cross-device product reduction
is needed. Apply the stock `Add` scan to those contributions. If `P_i[k]` is
partition `i`'s inclusive row/group prefix and `T_i` its final value, download
the four totals and compute `O_i = sum(T_j, j < i)`. The exclusive witness is
`O_i` at local slot zero and `O_i + P_i[k - 1]` elsewhere. This is exactly the
serial row/group prefix. The circuit's total is `sum(T_i)`; the existing host
code continues chaining circuit totals in transcript order.

Transpose the adjusted prefixes into local column quarters, then release the
remaining scan arrays. Gather each group's four quarters in global row order
onto a selected output owner before its usual full-length inverse NTT. For
four groups, assign one 4 GiB output column per card. A card holds its own
4 GiB of local quarters alongside the gathered 4 GiB column until every
recipient has finished reading those quarters. The dense gather moves 4 GiB
locally, 4 GiB between peers and **8 GiB across the unsupported pair boundary**
in total; the latter costs 8 GiB D2H plus 8 GiB H2D through the pinned ring.
Input tile exchange adds about 36 GiB of peer traffic plus local gathers,
excluding halos. Duplicate input NTTs and coefficient uploads also belong in
the comparison. Final coefficient downloads total 16 GiB, as in the existing
lookup path.

Require all selected devices to pass every phase's admission before starting
native work. Unsupported geometry, topology or memory returns the entire job
to the bounded host path. After execution starts, runtime errors propagate;
all workers and transfer streams drain before borrowed host buffers, peer
sources, events or leases are released. A sender's local output must survive
every destination's completion event, and a pinned slot cannot be reused
before its destination consumes it.

The local correctness gates passed with nonzero unequal carries across all
four row partitions, one through five output groups including a partial final
group, negative multiplicities, zero messages, empty arguments, constants,
global selectors and `Next` boundaries. Forced pinned transfers, real async-pool
buffers, admission rejection, exact coefficients/totals and fresh proof bytes
also passed. Full coefficient downloads, additional uploads and cross-pair
staging are included in the reported local time.

The two target-size diagnostics required no chain, production cache rebuild
or ceremony download. They establish local consumer improvements. The latest
complete pipeline time is still 839.988 seconds; neither these local timings
nor the `2^27` cap establishes the five-minute goal.

## Recursive circuit reductions

The historical v3 recursive staging/proving phases total 342.361 seconds.
That circuit has 162,558,992 gates and 220,627,443 used rows. Curve-input checks
account for 43,381,902 gates; MSM construction adds 103,621,353, together
about 90.4% of gates. Reaching 2^27 requires removing at least 86,409,715 used
rows, about 39.2%, before padding.

The proposals below retain their separate setup-capacity and soundness
questions. Their timing and row-reduction estimates are planning figures.
Execution optimizations can proceed independently of decisions to change
subgroup or degree checks.

The uniform-height proposal below removes shifted-degree commitments and
associated recursive MSMs by making stage-one trace heights equal to the
development SRS length. The selected v4 mixed-height policy instead has a
shared-builder census of 118,634,361 used rows, including complete MSMs and
the typed claim schema, and
fits one `2^27` computation trace with the preserved development coordinates.
Real ceremony constants still need a recount. That measured conditional fit
supersedes the earlier ~116M-row prediction; uniform stage-one padding and
its added matrix work are not part of the selected implementation.

Halving the domain does not promise half the elapsed time. Uniform padding also
increases stage-one work, especially its wide hash traces. Measure the total
change to both stages, including compilation, witness generation, committed
cells, transfers, memory and final packet size. A smaller recursive domain
can still lose overall if it requires too much padding in stage one.

The narrow, taller BLAKE3 layout below is especially worth
evaluating: fewer committed columns can reduce first-stage movement and
recursive commitment/MSM work together while using rows productively. Count
fixed and auxiliary columns, copy constraints and quotient degree as well as
main columns. Compare layouts at power-of-two thresholds; reducing logical
rows without crossing a threshold may leave dense FFT/MSM sizes unchanged.

Keep subgroup-check removal and changes to arithmetic bounds under their
separate soundness reviews. Execution-only optimizations above can proceed
without those decisions. Layout changes need new fixed keys and parity
fixtures. Before investing heavily in recursive-specific GPU advice, also
resolve whether the [terminal backend proposal](plonk-terminal-backend-review.md)
will replace that wrapper. Removing it would avoid its present work, but the
replacement terminal's full cost remains unmeasured.

### Setup capacity

Filecoin's
[parameters](https://github.com/filecoin-project/powersoftau/blob/master/src/parameters.rs)
define `TAU_POWERS_LENGTH = 2^27` G2 powers and
`TAU_POWERS_G1_LENGTH = 2 * TAU_POWERS_LENGTH - 1` G1 powers.

The legacy shifted-degree adapter sizes the SRS to the tallest committed matrix
and requires the G2 degree key `τ^(max_len - 2^k)·H` for every `k`, including `τ^(max_len - 1)·H`
([`src/ark_adapter/srs.rs`](../src/ark_adapter/srs.rs)).

| `max_len` | G1 powers needed | Highest G2 exponent needed | Filecoin |
| --- | ---: | ---: | --- |
| 2^27 | 2^27 | 2^27 - 1 | both present |
| 2^28 | 2^28 | 2^28 - 1 | G1 is one power short; G2 keys absent |

This table checks point availability for the legacy prefix-based degree policy,
not soundness of its degree bounds against the complete public setup. The
selected v4 policy loads the required G1 prefix and two G2 anchors, with no
shifted-degree keys.

Every committed matrix in a stage must fit `max_len`: main, stage-2 and fixed
columns, quotient slices of trace size, and merged tables. Stage 1 is capped
at 2^24 and already fits.

Both KZG stages need the trusted setup. The final packet checks the stage-1
commitments by pairing against stage-1 G2 keys, so stage 1 cannot use a
development SRS in production either.

#### Degree bounds under a public setup

[commitment-backend degree policy](pcs-abstraction.md) states that truncating a
larger public SRS does not establish a smaller degree bound. A prover holding
Filecoin's public `2^28 - 1` G1 powers can commit polynomials of degree up to
`2^28 - 2`, so a shifted check relative to `max_len = 2^27` does not establish
the claimed smaller degree bound. Making
it one would require the G2 key `τ^(2^28 - 1 - n)·H`, which Filecoin holds only
for `n = 2^27`. Every trace in both stages would have to be exactly 2^27 tall.

The historical reduction below removes shifted commitments under a configured
development SRS length; that alone establishes no smaller production degree
bound when the prover has access to the complete public setup. The selected
v4 implementation instead admits the full public degree in its
[soundness argument](#full-public-degree-soundness-argument) and binds the
ceremony identity, degree allowance and trace cap in a separate transcript.
Its assumptions and error terms apply independently of the performance
projections below. Legacy v3 keeps its strict degree-check behavior.

### Where the wrap's rows go

The 19-circuit wrap has 162,558,992 gates and 58,068,412 lookups, 220,627,443
rows, padded to 2^28. Getting under 2^27 = 134,217,728 rows requires removing
at least 86.4M rows.

#### Measured unit costs

| Unit | Rows | Source |
| --- | ---: | --- |
| Witness curve-point allocation and encoding | 108,864 | 64 points cost 4,862,080 gates and 2,105,216 lookups |
| of which the subgroup check | ~107K | two 64-bit constant multiplications: 126 doublings, 10 additions |
| of which on-curve, bounds, encoding | ~2K | two relations plus byte constraints |
| MSM term, four-bit Straus | 98K to 116K | 64 affine additions, table build, 960 limb selects |
| Affine addition | ~1,060 | four Fq relations |
| Affine doubling | ~795 | three Fq relations |
| Fq relation, five 80-bit limbs | ~265 | 52 lookups, 25 to 50 limb products, 5 equations |
| BLAKE3 compression, generic byte gadget | ~24K | 25.9M rows over roughly 68 KB of transcript input |

Gadgets: [`experiments/kzg-wrap/src/native_curve.rs`](../experiments/kzg-wrap/src/native_curve.rs),
[`native_msm.rs`](../experiments/kzg-wrap/src/native_msm.rs),
[`native_field.rs`](../experiments/kzg-wrap/src/native_field.rs),
[`native_transcript.rs`](../experiments/kzg-wrap/src/native_transcript.rs).

#### Attribution

| Component | Rows | Share |
| --- | ---: | ---: |
| 571 witness points, dominated by subgroup checks | 62.2M | 28% |
| Transcript, generic BLAKE3 gadget | 25.9M | 12% |
| Shifted-degree MSM, 330 terms | 35.6M | 16% |
| Seven degree-group MSMs, 330 terms | 38.3M | 17% |
| Opening-batch MSM, 583 terms | 57.3M | 26% |
| Witness MSM and AIR checks | 1.4M | 1% |

The 583 opening-batch terms are 573 commitments, the generator and 9 opening
witnesses. The 330 shifted commitments exist because stage-1 traces have eight
distinct heights: nine arithmetic partitions at 2^24, the merged lookup table
at 2^17, the six BLAKE3 traces at 2^22, 2^22, 2^22, 2^22, 2^19 and 2^21, and
the three hash tables at 2^16, 2^9 and 2^8. Of the 330, 236 are witness points
and 94 are fixed constants.

### Reductions

The numbering defines the projection sequence below, not an implementation
priority or approval of the protocol changes. Estimates use the unit costs
above; the stats line printed by the wrap builder is the measurement.

#### 1. Make every stage-1 trace 2^24 tall

Removes the shifted-degree MSM, the seven degree-group MSMs, 236 shifted
witness points and seven opening witnesses. The report above counts 243
removed witness inputs and corrects for tables reused by the opening MSM.
Its planning subtotal is about 123.6M rows before transcript and other
changes; neither the earlier ~116M total nor a specific margin is measured.

Under the current development-setup degree convention, the verifier in
[`src/ark_adapter/pcs.rs`](../src/ark_adapter/pcs.rs) already requires empty
shifted commitments for a matrix equal to the SRS length, and the wrap builder
in
[`experiments/kzg-wrap/src/native_verifier.rs`](../experiments/kzg-wrap/src/native_verifier.rs)
emits the shifted and degree-group MSMs only when shifted commitments exist.
This uses existing verifier behavior; it does not resolve the public-setup
degree issue above. Recount these savings under the selected Filecoin policy.

Lowering changes:

- A minimum-height option for the six BLAKE3 traces, whose heights come from
  `rows[t].len().next_power_of_two()` in `layouts()` in
  [`src/plonkish/hash.rs`](../src/plonkish/hash.rs), and for the three hash
  tables fixed there at 65536, 512 and 256 rows.
- The same minimum for the merged lookup table in `merge_table_traces` in
  [`src/plonkish/stark.rs`](../src/plonkish/stark.rs).
- The arithmetic partitions already all equal 2^24 in the current profile.

Effects outside the wrap:

- The external pairing count drops from 10 to 2. The packet loses 8 points,
  384 bytes, and 16 public values. `degree_output_count` in
  [`experiments/kzg-wrap/src/outer.rs`](../experiments/kzg-wrap/src/outer.rs)
  becomes zero and the packet profile changes.
- Stage-1 proof bytes change; they are internal to the pipeline. New CPU/GPU
  parity fixtures are required, as for any profile change.
- Stage 1 pays for the padding. The four wide BLAKE3 traces grow from 2^22 to
  2^24 rows at 32 or 40 main columns each. The earlier one-to-two-minute
  stage-one penalty was an unmeasured estimate. The recursive padded domain
  would halve; its latest staging/proving times are 140.143 s and 202.218 s,
  with 464.2 GiB peak proving RSS. These times and peak memory do not
  automatically halve: loading, circuit construction, witness work, auxiliary
  tables, scratch, and data movement scale differently. Measure both stages.

Lowering the arithmetic cap to 2^22 instead of padding up would add 27
arithmetic partitions. The current estimate adds about 190 witness
commitments and predicts a larger wrapper; a combined layout count would
need to demonstrate an offsetting reduction before prioritizing this route.

#### 2. Drop G1 subgroup checks on witness points

Keep the on-curve check, limb bounds and canonical encoding. Removes about
107K rows per remaining witness point, roughly 36M, leaving about 80M.

Argument: a cofactor component in a G1 input is invisible to a pairing against
an r-torsion G2 key, because the reduced pairing is defined on
`E(Fq)/rE(Fq)`, and the cofactor is coprime to `r`. The in-circuit affine
formulas are the correct group law on all of `E(Fq)` and reject exceptional
cases through constrained inverses, so the circuit's MSM output projects to the
honest r-torsion combination. `check_pairings` in `outer.rs` additionally
deserializes the output points with arkworks validation, which rejects any
output carrying a cofactor component.

This is a soundness decision. It requires a written argument reviewed before
it ships. The change itself is the `p.subgroup(b, f)` call in
`PointInput::new`.

#### 3. Four 96-bit limbs instead of five 80-bit limbs

`native_field.rs` uses `W = 80`, `L = 5`, `CARRY_BITS = 89`. With four 96-bit
limbs a relation has 16 limb products per term instead of 25, four carry
equations instead of five, and about 45 lookups instead of 52. Roughly 30%
off every nonnative operation, about 18M rows after reductions 1 and 2,
leaving about 62M. That is under 2^26.

Requires redoing the bound analysis: product magnitudes, biased quotient
range and carry width must stay below the scalar field, as the existing
`integer_bounds_do_not_wrap_fr` test checks for the current layout.

#### 4. Narrow, tall BLAKE3 layout

The four 32- and 40-column BLAKE3 main traces account for 144 of the 573
commitments. A layout using four times as many rows with a quarter of the
main columns could remove 108 main commitments while avoiding unused main
cells introduced by uniform padding. This requires an AIR redesign in
`hash.rs`; count any additional fixed, lookup, routing, or quotient columns.

The earlier roughly 25M-row saving used per-commitment costs that include
checks and arithmetic also targeted by reductions 2 and 3. It cannot be
subtracted from their already reduced circuit. Rebuild or count the combined
layout before assigning a row saving or padded domain. Explore this layout
early because it may improve the cost of uniform heights without changing
subgroup policy.

#### Smaller levers

- Signed-digit four-bit windows: table build of 7 operations instead of 15,
  about 5% of MSM rows.
- 20-bit range table instead of 16-bit: 40 lookups per relation instead of 52,
  about 5% of relation rows, with a 2^20-row table.
- Merging the nine arithmetic partitions by raising the stage-1 cap to 2^27
  saves about 10M rows but conflicts with reduction 1, since the BLAKE3 traces
  would then need 2^27 padding. Not recommended.

#### Other layout alternatives

- Sharding the wrap into two 2^27 traces adds commitments and two activation
  claims. It is excluded from the current migration plan because the final
  proof must not grow.
- Moving the wrap transcript to the current compact BLAKE3 layout would add
  nine hash traces with about 160 commitments to the outer proof, several
  kilobytes. The proposal therefore retains the generic gadget and reduces
  its work through fewer hashed bytes; alternate gadget layouts are unmeasured.
- Pippenger bucketing in-circuit needs constrained indexing or selection.
  No circuit-size advantage over the existing Straus construction has been
  established at this term count; a fast native Pippenger implementation is
  not evidence of a cheaper verification circuit.

### Historical arithmetic projections

The following cumulative estimates predate the migration sizing report's
MSM-table correction. They are retained as arithmetic research hypotheses;
they are not acceptance counts. Subgroup removal and new limb bounds are
excluded from the current baseline plan. Recount combinations before using
any of these predicted power-of-two thresholds.

| Step | Earlier estimated rows | Earlier predicted padded trace |
| --- | ---: | ---: |
| Today | 220.6M | 2^28 |
| 1. Uniform 2^24 stage-1 heights | ~116M | 2^27 |
| 2. No subgroup checks | ~80M | 2^27 |
| 3. Four 96-bit limbs | ~62M | 2^26 |
| 4. Narrow BLAKE3 layout after the preceding changes | Needs a combined layout count | Not established |

### Ceremony import

The authenticated importer and explicit setup selection are implemented in
the [parameter migration](#parameter-import-and-artifact-migration). Provisioning
the real stock artifact, regenerating ceremony-bound fixed keys, and the
production recount remain outstanding. The importer retains a G1 prefix for
honest proving while binding the full public degree allowance used by v4;
it does not claim that truncating the public setup enforces a smaller bound.
Development caches remain explicitly separate from ceremony provenance.

### Sizing and validation order

1. Count layout sizes, shifted commitments, recursive MSM terms, and padded
   domains without producing a proof. Include the added committed cells and
   memory cost in stage one.
2. When proof-dependent wrapper construction is needed, run only the affected
   stage-one fixture and wrap builder. Inspect the staged manifest, commitment
   shape, and builder statistics before considering outer proving.
3. Validate a changed layout with new fixed keys, profile identifiers, and
   CPU/GPU parity fixtures. Measure both affected stages separately. A full
   integrated run is reserved for a demonstrated cross-stage validation need
   or an explicit request.
4. Record the separate soundness review for subgroup/degree-check changes and
   the intermediate-bound analysis for a new limb layout before shipping them.

The older sizing workflow budgeted about four minutes for stage-one staging
and about a minute for wrapper construction. These are estimates for that
workflow, not authorization or a requirement to run it for every change.


## Alternative profiles and terminal backends

These ideas remain separate from optimizations that preserve the measured
profile. Include them when comparing architectural options, without counting
their potential savings on top of an incompatible wrapper design.

- **FRI profile tuning:** compare query count, blowup, grinding, Merkle caps,
  and folding arity using a supported soundness analysis and the required
  verifier changes. The [earlier review](../experiments/kzg-pipeline-review.md)
  withdrew its claim that increasing blowup would permit about 34 queries
  at the same soundness. No such saving is included in this roadmap's
  measurements. Record new profiles and proof-size/proving-cost tradeoffs
  separately; keep the current security parameters for execution benchmarks.
- **One terminal proof:** the
  [terminal backend review](plonk-terminal-backend-review.md) considers
  replacing both current KZG stages with a conventional PLONK/KZG proof of
  the compressed FRI statement. Reuse of CUDA primitives is possible, but
  circuit fit, hash layout, setup capacity, and complete proving cost remain
  unresolved. Compare that full replacement against the complete two-stage
  pipeline before investing heavily in recursive-only kernels.

## CUDA scheduling and memory constraints

Use bounded queues with explicit ownership: prepare the next input while
another device computes, retain reusable data near its consumers, and overlap
transfers where dependencies permit. Commitments and transcript observations
remain ordered. Challenge dependencies impose barriers between main
commitment, lookup, and quotient. Judge scheduling by reduced critical-path
time, not average utilization alone.

Measure SRS residency within shared memory admission. A replicated SRS must
fit beside coefficients, evaluations, scratch, and pool reservations; the
aggregate SRS transfer/copy intervals are not promised wall-time savings.

Mixed FFT/MSM/transfer work must preserve these constraints; do not merely
replace the busy flag with a counter:

1. Sppark's `msm_t` captures `select_gpu(device_id)` and uses that runtime's
   zero and three flip-flop streams. Concurrent MSM contexts need independent
   stream sets. `transfer_lane` also keys shared mutable rings by device and
   lane, so concurrent tasks must not reuse the same rings.
2. Retained coefficients, SRS points, active scratch and cached free memory
   must share one budget. When retaining CUDA pool allocations, account for
   reserved-but-unused pool bytes; `cudaMemGetInfo` alone can understate
   reusable capacity. Add a bounded pool retention policy without consuming
   the transient reserve needed by `2^27` traces and their temporary quotient
   transforms.
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
Its runtime-header context-selection patch, scoped MSM profiling-header patch
and symbol-renaming shim remain integration debt; upstream namespace/stream
injection would be cleaner.


## Sppark primitives to adopt

The pinned fork at `e10e107673aa22861f0f8b9758fc62169ab919ae` supplies the
primitives below. Several now remove host work in admitted device paths;
their complete-stage and integrated speedups remain unmeasured.

The old roughly 7 s recursive GPU estimate excluded MSM compute. The
[historical recursive log](../experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/recursive_prove.log)
records 18.401 s of MSM compute intervals alone, summed across devices and
streams. These intervals overlap and are not a critical-path breakdown.
Use the Nsight timeline for device activity; do not subtract aggregate event
times from the 202.218 s recursive proving time.

### Already native

- `msm_t::invoke` with `dev_ptr_t` arguments binds to the `is_device_ptr`
  overload, so Pippenger already runs on resident points and scalars with
  sppark's internal double-buffering across the flip-flop streams.
- `NTT::Base_dev_ptr` transforms resident buffers in place.
- `prefix_op<Multiply>`, `batch_inversion`, and `prefix_op<Add>` construct
  admitted LogUp traces on device, followed by a device transpose and IFFTs.
  The [local comparison](#resident-lookup-construction-on-admitted-traces)
  is 4.32× faster with exact coefficients and accumulator total.
- `polynomial/evaluate.cuh` single-point kernels serve the openings.
- `polynomial/div_by_x_minus_z.cuh` now serves division, with small-grid and
  partial-tail launcher guards. A resident quotient feeds MSM directly when
  one device is selected. See [the measured comparison](#sppark-division-and-point-uploads).

### Ranked adoption

| # | Primitive | Target | Recorded scope, not expected saving |
| --- | --- | --- | --- |
| 1 | `polynomial/prefix_op.cuh` `Add` and `Multiply`; `ff/batch_inversion.hpp` | Bounded multi-device LogUp with ordered row offsets | Implemented locally: 3.179 s to 0.736 s on one GPU; latest comparison 19.232 s to 6.680 s for synthetic `2^27` inputs and the saved wrapper graph, including all transfers |
| 2 | Persistent MSM contexts and resident SRS points/chunks | Repeated context allocation and SRS uploads | 216.08 GiB of SRS uploads; about 19.14 s of aggregate upload/host-copy intervals, overlapping across devices |
| 3 | `polynomial/div_by_x_minus_z.cuh` with resident folding and MSM | Opening folds and polynomial transfers; the host carry scan is removed | Division and one-device quotient retention are implemented; multi-device folding needs a better reduction/SRS strategy |
| 4 | `stream_t::notify(semaphore_t)`, `channel_t`, non-owning `stream_t` | Completion notification and queue plumbing | Enablers for a bounded scheduler; no standalone saving measured |

1. **Prefix scans and batch inversion.** These now form the admitted LogUp
   builder. `prefix_op<Multiply>` supplies a
   forward product scan for Montgomery batch inversion and can generate power
   vectors from a constant fill; `prefix_op<Add>` supplies the LogUp running
   sum. The
   [batch-inversion helper](https://github.com/argumentcomputer/sppark/blob/e10e107673aa22861f0f8b9758fc62169ab919ae/ff/batch_inversion.hpp)
   inverts a compile-time-sized batch with one field inversion and preserves
   zero entries, matching the CPU helper's semantics. Choose a per-thread,
   warp, or tile batch size by measuring inversion count and register/shared
   memory pressure. Do not default to one inversion per row: the CPU already
   batches, and the vanishing inverse is periodic across the coset. Fingerprint
   construction, constraint sweeps, and device trace storage use the custom
   DAG and layout kernels around those stock primitives. The implemented
   wide-trace path uses bounded row partitions and ordered prefix offsets.
   Peer reads now reuse every retained coefficient within its pair; per-device
   upload imbalance remains measurable.
2. **Persistent `msm_t` with resident SRS.** An optional
   [resident point cache](../src/ark_adapter/cuda/srs_cache.rs) is implemented
   behind `MULTI_STARK_KZG_CUDA_SRS_CACHE=1`; the default remains disabled.
   Immutable point chunks are separate from one mutable sppark workspace per
   device. The workspace is keyed by sppark's actual window size, so retaining
   eight point chunks does not retain eight bucket arrays. Every invocation
   rebinds non-owning point and scalar `dev_ptr_t` views while holding the
   device lease, including shifted ranges and temporary normalized scalars.
   The captured sppark streams remain serialized by that lease.

   A nonreused owner token retains the immutable `Arc<Srs>`; cache entries use
   its identity and exact range, rather than host addresses or a caller-supplied
   label. Prefix and contained subrange requests reuse the allocation. Expired
   owners and overlapping replacement ranges are removed before admission.
   Points use at most 12 GiB per device, enough for a `2^27` prefix, with at most
   `2^24` points per chunk. Upload staging is bounded to `2^18` layout-checked
   104-byte arkworks points, or 26 MiB, and the device conversion preserves
   infinity flags in the retained 96-byte representation.

   SRS storage yields to coefficient, FFT, lookup and quotient allocations.
   Initial eligibility includes evictable point/workspace bytes; under the
   lease, LRU eviction is followed by a fresh usable-memory check. That check
   includes driver-free bytes and unused pages in the current CUDA async pool,
   capped at total memory, before subtracting the quarter-device reserve.
   Raw `cudaMemGetInfo` alone can reject a job after successful eviction because
   freed pages remain reusable inside the pool. The coefficient cap and
   [shared memory constraints](#cuda-scheduling-and-memory-constraints) still
   apply; 12 GiB is a cache ceiling, not capacity reserved against scratch.

   The isolated [resident-SRS report](../experiments/kzg-cuda-validation/resident-srs-20261009/report.json)
   measures `2^22` points and two host-backed full-field scalar columns across
   four GPUs. Five alternating warmed pairs gave medians **50.268 ms uncached
   versus 42.131 ms cached**, a **16.2% reduction**. Each card held 96 MiB of
   points and a 9,119,744-byte workspace. Warmup uploaded 104 MiB per card;
   the five timed cache calls per card added five hits and no point uploads or
   evictions. This combines upload and workspace-reuse savings. It does not
   establish `2^27` residency, production eviction behavior, or a stage-level
   saving.

   Focused tests cover immutable ownership, direct prefix/subrange hits,
   scalar rebinding after normalization, odd chunk tails, exceptional points,
   distinct SRS owners, forced eviction and re-upload, and single-device
   openings. The four-GPU public-degree fixture still matches the saved CPU
   proof exactly: **1,205 bytes**, BLAKE3
   `8164e8f1acee3f94dbb48d29908d4e99085af60f3a8a96f4d09b8c6405c69bca`.
   Correctness logs separately time the bounded point-conversion kernels;
   event profiling was disabled for the repeated-MSM timing.
3. **Device-side division and folding.** Division now uses
   [sppark's cooperative kernel](https://github.com/argumentcomputer/sppark/blob/e10e107673aa22861f0f8b9758fc62169ab919ae/polynomial/div_by_x_minus_z.cuh)
   with validated launch guards. The CPU carry scan is gone. One-device
   openings retain the quotient through MSM; multiple devices currently use
   a host quotient to preserve MSM partitioning. GPU folding is still open:
   simply committing each device's partial fold repeated enough SRS uploads
   and MSM work to regress the local comparison. Include quotient reduction,
   point reuse, and spilled inputs when designing the complete resident path.
   Preserve transcript weights and mixed-degree padding throughout.
4. **Pipeline plumbing.** The fork's non-owning `stream_t(id, cudaStream_t)`
   can wrap an independently owned stream. It does not redirect the zero and
   flip-flop streams captured internally by `msm_t`; concurrent MSM contexts
   still need stream isolation. `stream_t::notify` uses `cudaLaunchHostFunc`
   to notify a semaphore without calling CUDA from the callback. Its
   [channel implementation](https://github.com/argumentcomputer/sppark/blob/e10e107673aa22861f0f8b9758fc62169ab919ae/util/thread_pool_t.hpp)
   is an unbounded `std::deque` with a mutex and condition variable. Add
   capacity/byte admission and backpressure in the scheduler rather than
   treating `channel_t` as a bounded queue. Completion notification alone
   also does not provide the Rust dispatch/future and buffer-lifetime design.

### Lower priority or blocked candidates

- `NTT::Base_dev_ptr_batch` is the fork's own addition and the Goldilocks path
  uses it, but it targets batches of short vectors. The expensive arithmetic
  target traces have 2^24 through 2^27 rows; smaller hash and table traces also exist.
  Establish launch overhead for those smaller groups before prioritizing it.
- `NTT::LDE_expand` fuses the coset spread that `scale_coset` performs, and the
  `NR`/`RN` ordering can avoid bit-reversal passes. These target transform
  overhead; their effect on this pipeline has not been measured.
- The multi-point `evaluate` kernel has the recorded shared-memory race; the
  single-point path records about 0.16 s of compute intervals, excluding its
  upload and host-copy costs. Retain the correct path while prioritizing larger
  measured costs.
- `msm/batch_addition.cuh` and the thread pool's `par_map` have no consumer
  here; Rayon owns host parallelism.

The archived 496 GiB of recursive evaluation downloads motivated the bounded
device consumers now measured at `2^27`. Their stock scans and NTTs surround
the circuit-specific lookup and quotient kernels. Validation with the actual
ceremony wrapper and the complete pipeline remains outstanding.


## Implementation review, 2026-10-10

This review covers the uncommitted `sb/kzg` tree above `a904bce`: the
resident and distributed consumers, the SRS cache, retained workers, the
Filecoin importer and the `2^27` frontend. It judges direction against the
measured profile, not against new timings. No build or test ran for it; four
source files changed after the last compiled executables.

### Confirmed direction

- Host preparation holds the largest measured reductions on the branch:
  stage-one frontend 128.456 s to 93.145 s, scalar assignment 27.742 s to
  3.624 s, recursive witness about 52.7 s to roughly 10 s. These target the
  342 s of staging that the [profile](#goal-and-priorities) identifies as the
  main cost, and they should stay first.
- The [v4 recursive frontend](#measure-the-complete-v4-recursive-frontend)
  fits `2^27` at 118,634,362 rows with native MSM and pairing checks passing,
  close to the lowering-only projection in the
  [reduction plan](#recursive-circuit-reductions).
- The [full-public-degree policy](#full-public-degree-soundness-argument)
  is the appropriate choice: it is how KZG-based PLONK systems commit, the
  Schwartz–Zippel loss at degree `2^28` over the BLS12-381 scalar field is
  negligible, and it removes the shifted-degree MSM from the wrap. The
  [Filecoin importer](../src/ark_adapter/srs/filecoin.rs) authenticates the
  full artifact digest and checks canonical coordinates, subgroup membership
  and the power progression, which is the correct shape for that import.
- Resident lookup already uses sppark's prefix scans and batch inversion,
  division already uses `div_by_x_minus_z`, and the SRS cache exists behind
  `MULTI_STARK_KZG_CUDA_SRS_CACHE=1`. The
  [adoption table](#ranked-adoption) records scope; it no longer predicts a
  saving, and earlier summed estimates for adopting those primitives are
  superseded.

### Findings

1. **No integrated measurement follows the isolated results.** The 839.988 s
   baseline predates roughly twenty local changes, and several of them
   conflict on device memory. The distributed consumers require
   `MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8`, the SRS cache may hold 12 GiB of
   points per card, and the archived run retained about 160 GiB of
   coefficients. Under the 8 GiB cap the distributed quotient re-uploaded
   156 GiB of coefficients against 112 GiB in its baseline. The local
   speedups therefore cannot be added, and the next full-chain run should
   enable every implemented path together before further kernel work.
2. **The distributed layer is the largest risk for a bounded gain.**
   [`kzg_distributed.cuh`](../cuda/kzg_distributed.cuh),
   [`kzg_quotient_distributed.cuh`](../cuda/kzg_quotient_distributed.cuh) and
   [`kzg_lookup_distributed.cuh`](../cuda/kzg_lookup_distributed.cuh) total
   about 900 lines, plus 388 lines of Rust glue, covering peer pool access,
   pinned staging rings, next-row halos, atomic group leases and drain
   guards. The path is fixed at four devices in two mutual peer pairs and
   has no CI coverage. Its measured gain is 26.768 s for the quotient and
   12.248 s for the lookup on synthetic `2^27` inputs. The path is forced by
   memory: 22 nonconstant columns at `2^27` occupy 88 GiB per coset in
   evaluation form, more than one card holds with scratch. Keep it, but do
   not extend it until an integrated run shows it on the critical path.
3. **The sweep kernels are latency-bound by design.** `quotient_sweep`,
   `quotient_tile_sweep` and the lookup rational kernels launch at most
   `MAX_BLOCKS = 256` blocks of 128 threads, about five warps per SM on
   Blackwell, and the DAG interpreter stores every node value in global
   scratch (`values[node.out * blockDim.x]`). Kernel intervals already exceed
   host-copy intervals in the resident quotient comparison (299.595 ms
   against 30.731 ms). Raising the block cap costs `blocks × 128 × slots × 32`
   bytes of scratch, under 1 GiB at 4,096 blocks for typical slot counts;
   keeping short DAG prefixes in registers is the larger change.
4. **The coset `L_first` selector is materialized unnecessarily.** Each
   resident or distributed quotient job fills an `n`-element buffer with ones,
   runs a trace-sized coset NTT, and stores the result: 4 GiB per card per
   coset at `2^27`, charged against the admission that decides whether
   distribution is needed. The unnormalized selector has the closed form
   `(shift^n - 1) / (point - 1)`, and `point` is already tracked per row;
   a per-tile batch inversion replaces the buffer and the NTT.
5. **The recursive fixed profile binds the public statement.** Claim-derived
   constant gates remain in the recursive circuit, so its fixed commitments
   change with the statement. A verifier of the final packet cannot recompute
   fifteen `2^27` fixed commitments, so no ceremony-backed proof is meaningful
   until the builder is statement-invariant. This item also decides what the
   fixed cache may reuse; schedule it ahead of further recursive performance
   work rather than preserving the binding indefinitely.
6. **Configuration lives in the environment.** The tree reads 45 distinct
   `MULTI_STARK_*` variables, 21 of them inside the library, and some are
   order-dependent: the resident cap must be set before device
   initialization, and distributed mode silently depends on it. Library
   behavior should come from `KzgConfig` and explicit constructor arguments;
   environment parsing belongs in the examples and benchmark harness.
7. **`build.rs` edits sppark by string substitution.** The MSM profiling
   timers are injected into a copy of `msm/pippenger.cuh` with six exact-match
   replacements. The pinned dependency is already a maintained fork; the hooks
   belong there, where the compiler checks them.
8. **Repository hygiene.** The working tree holds 56 modified and about 60
   untracked paths with 7,755 uncommitted insertions. About 1.1 GiB of
   measured executables sit under `experiments/kzg-cuda-validation`, neither
   tracked nor ignored, and `a904bce` already committed 52 MB of binaries
   under `resume-20261009/measured-binaries`. The hash manifests identify
   those binaries; store the files outside the tree. The pipeline itself now
   lives in `examples/support`, including the 784-line request worker and its
   protocol, and belongs in a binary crate.

### Recommended order

1. Make the recursive builder statement-invariant and rebind the fixed cache
   identity to circuit and profile only.
2. Run the complete chain with resident consumers, the SRS cache, retained
   workers and the `2^27` layout enabled, recording peak device and host
   memory per phase. Treat that run as the new baseline.
3. Apply the block-cap and selector changes above inside the existing kernels
   and re-measure only the affected consumers.
4. Move environment parsing to the examples, commit the branch in reviewable
   pieces, and relocate measured binaries.
5. Return to distribution and transfer placement only if the new baseline
   shows the recursive quotient or lookup on the critical path.


## Targeted validation

For each optimization, run focused correctness checks and an isolated A/B
comparison using preserved inputs for the affected operation or stage. Reuse
the same binary with runtime controls where practical; respect cache identities
after rebuilding. Record wall time, peak host/device memory, transferred bytes,
cache state, and output parity. Run authorized work at the machine's default
parallelism.

Full-chain runs are reserved for changes that require validation across stage
boundaries or an explicitly requested integrated measurement. A targeted
optimization does not require regenerating unrelated FRI proofs, witnesses,
or fixed caches. Expensive builds and experiments still require agreement on
their scope and expected cost. Preserve the current FRI security parameters.

| Step | Reviewable result |
| --- | --- |
| Shared host preparation | One compact-hash preparation per assignment, shared validation, and isolated timing for each eliminated repeat |
| Reusable worker state | Compiled layouts and loaded keys reused for compatible fresh proofs; separate startup and repeated-proof timings |
| Direct CPU trace input | Staging connected to proving with optional checkpoints, bounded memory, and unchanged commitments |
| Reused evaluations and device consumers | Domain-selector reuse, measured fixed-evaluation caching, and a bounded lookup/quotient placement plan |
| Device computation traces | One KZG computation partition generated from a CPU assignment, committed directly, with CPU/GPU cell and commitment parity |
| Complete trace coverage | Matching auxiliary hash traces and table multiplicities, bounded memory, and checkpoint/resume parity |
| First stage witness kernels | Typed Goldilocks-to-scalar advice operations and a reusable dependency schedule, with host fallback for unsupported operations |
| Recursive witness kernels | Exact foreign-field and curve advice, scheduled without changing the circuit relation or public statement binding |

Use small fixtures first. Compare logical values, canonical trace cells,
padding, partition boundaries, table multiplicities, and commitments against
the CPU implementation. Exercise modular boundary values, carries, inverses,
empty/padded rows, regeneration, and memory-pressure fallback. Invalid inputs
must still fail the relevant checks.

When a complete pipeline comparison is required, retain the same input profile and record
cache state, source/binary identities, and fresh witness generation. Require
the existing deterministic fixture's proof/packet/profile parity, independent
CPU verification, and altered-claim checks. Report assignment generation,
checking, trace construction, transfers, checkpoint I/O, and commitment
separately, alongside end-to-end wall time and peak host/device memory.

The target is reduced elapsed time. Higher GPU utilization alone does not
establish a benefit, and the approximately 100 seconds in the two witness
segments is not a projected saving. Circuit construction, lowering, and
later lookup/quotient consumers remain separate costs.

Follow mixed CUDA scheduling changes with overlapping-operation parity,
buffer reuse, failure/lifetime, and memory-pressure checks. Layout changes
require new keys and fixtures; implementation changes preserving the existing
circuit retain its deterministic proof-byte comparison.


## Recovery and run commands

The commands here support recovery or an explicitly scoped measurement.
Use the targeted-validation policy above when developing local optimizations.

### Preserve before shutting down

The current checkpoint is the **2026-10-10 development-v4 pipeline plus the
direct fixed-input optimization**. Preserve the commit containing this handoff,
including `docs/kzg-performance.md`, the new implementation modules and the
reproduction scripts. Generated validation artifacts are excluded from that
commit and remain local. The earlier branch base
`a904bceecd514f0781e69ef3d8fb395bbd2aea87` alone does not recover this work.
Also retain any subsequent working-tree changes, including untracked files.

`/opt/dlami/nvme` is mounted from `/dev/mapper/vg.01-lv_ephemeral`. Treat its
contents as disposable across instance shutdown or replacement. The checkout
is on `/dev/root`; retain that volume or copy the checkout elsewhere before
deleting the instance or its root volume. A local commit alone is not an
off-instance backup, and untracked files must be included in any copy.

The latest sealed local evidence archives are listed below. They are optional
historical evidence, separate from the code commit; only their reproduction
scripts are tracked. Keeping the source commit does not preserve their binaries,
reports, proof fixtures or captured source trees.

| Archive | What it preserves |
| --- | --- |
| [Integrated development-v4 workers](../experiments/kzg-cuda-validation/integrated-v4-workers-20261010/) | The 2m02s prepared request and 6m44s warmup, measured source snapshots and binaries, exact command/environment, logs, device and memory samples, small proof artifacts, parity checks and independent CPU verification |
| [Direct fixed input](../experiments/kzg-cuda-validation/fused-fixed-input-20261010/) | The current 186 source inputs, three current CUDA binaries and two test binaries, eight passing correctness tests, and the five-pair synthetic 0.578-to-0.086-second handoff comparison |

Both contain `SHA256SUMS`, `source-sha256.json`, `report.json` and
`verification.json`. Their manifests contain 620 and 412 files respectively.
The SHA-256 of each `SHA256SUMS` file is:

```text
integrated-v4-workers-20261010: 355c7db74446728541588a133ad4cf4dd0f0e7aee8710a906390c9ac5385a4e0
fused-fixed-input-20261010:    0187a38826d9626f54b27db6bcfd2e33b8da11d4d23e554f906055fbab395a16
```

Keep these archives unchanged. Their recorded document hashes describe the
documentation at validation time, before this shutdown handoff. The integrated
archive's measured executables predate direct fixed input; use the newer
archive for the current implementation and the integrated archive for the
measured full-request result. The latest optimization has no new full-pipeline
or production cold-start measurement.

The current reproducible inputs are the
[root artifacts](../experiments/kzg-cuda-validation/fri-lookup-materialization-20261009/root-artifacts/),
the [warmup FRI seed](../experiments/kzg-cuda-validation/first-stage-fused-20261009/compressed-fri/),
and the [independent seed claims](../experiments/kzg-cuda-validation/integrated-v4-workers-20261010/root-claims.bin).
The integrated archive's `outputs/` contains the small resulting proofs and
reports. Its [large-output inventory](../experiments/kzg-cuda-validation/integrated-v4-workers-20261010/large-output-files.json)
lists 44 files excluded from that copy. Preserve the following separately if
avoiding cache regeneration matters; the small evidence archives do not include
these large NVMe contents:

| Contents | Absolute path |
| --- | --- |
| Development SRS cache, including the `2^27` prefix | `/opt/dlami/nvme/multi-stark-kzg-perf-20261009/dev-srs-cache/` |
| Integrated outputs and fixed coefficient cache | `/opt/dlami/nvme/multi-stark-kzg-integrated-v4-20261010/` (cache: `fixed-cache/`) |
| Earlier isolated recursive-v4 proving outputs | `/opt/dlami/nvme/multi-stark-kzg-recursive-v4-20261010/` |

Those three paths were present at the shutdown handoff. The development caches
and staged artifacts are rebuildable from preserved inputs and matching source.
Fixed caches bind the exact executable digest: rebuilding or selecting the
newer direct-input binary selects a different namespace. Do not relabel an old
fixed cache for a new executable. SRS cache identity binds its parameter recipe
separately.

The archived benchmark workers shut down and their runs, builds and tests
finished. No new build, benchmark or cache regeneration was started for this
documentation handoff. Stopping a worker loses its retained host plans and keys;
the prepared-request latency does not include restoring that in-memory state.

The goal remains **blocked on authentic Filecoin parameters**, not complete.
The official `challenge_19` artifact, or an authenticated imported cache with an
externally pinned receipt, is still absent. On resumption, first verify the
preserved source and binary fingerprints, then obtain/import those parameters,
regenerate both stages' keys, recount with real constants, and run the Filecoin
acceptance command below. Keep one recursive computation trace capped at `2^27`
plus its merged table; do not shard the wrapper. Report prepared-request and
single cold-request timing separately. The latter, and the production savings
from direct fixed input, still need measurement.

For a new measurement, use the current
[benchmark harness](../experiments/kzg-cuda-bench.py) and the build/acceptance
commands below with newly generated or independently preserved root inputs.
The historical `run.py`, `run_pipeline.py`, `commands.sh` and `analyze.py` files
preserve experiment recipes; some require the original local outputs, pinned
source hashes or machine paths. They are not portable replacements for the
current harness and should not be replayed blindly against different sources.

The older compact [recovery directory](../experiments/kzg-cuda-validation/resume-20261009/)
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

### Resume commands

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

Alternatively, the three full-chain executables preserved under
`experiments/kzg-cuda-validation/partition-pipeline-20261009/measured-binaries/`
can be restored to
`target/release/examples/{init_fri_kzg_prove,ix_root}` and
`experiments/kzg-wrap/target/release/init-kzg-wrap` on a compatible host. They
use native CPU code and Blackwell cubins; rebuild for a different architecture.
Their SHA-256 values are in that archive's manifest and measured reports.
The diagnostic executable is preserved in the same directory. Those historical
executables predate the compiled-capability command; use their archived driver
for a historical replay, or rebuild for the current driver's admission checks.

For a complete run including FRI compression, use the preserved root:

```sh
MULTI_STARK_KZG_CUDA_PROFILE=1 MULTI_STARK_KZG_PREFETCH_GIB=32 \
  python3 experiments/kzg-cuda-bench.py \
  --setup development \
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
It checks each snapshotted executable's `capabilities` response before starting
CUDA work. Requesting `MULTI_STARK_KZG_BACKEND=cuda` does not establish that
the executable was built with CUDA. Explicit `--first-stage-binary`,
`--wrapper-binary` and `--fri-binary` paths also avoid selecting a stale nested
target directory after a build with `CARGO_TARGET_DIR=target`.

After provisioning the authenticated Filecoin cache and validating the current
profile, the full goal uses a distinct acceptance mode:

```sh
python3 experiments/kzg-cuda-bench.py \
  --acceptance --setup filecoin --kzg-devices 0,1,2,3 \
  --fused-staging --distributed-wrapper \
  --first-stage-binary target/release/examples/init_fri_kzg_prove \
  --wrapper-binary target/release/init-kzg-wrap \
  --fri-binary target/release/examples/ix_root \
  --root-artifacts /path/to/trusted-root \
  --filecoin-cache /path/to/authenticated-filecoin-cache \
  --filecoin-digest <externally-pinned-import-receipt> \
  --fixed-cache /path/to/current-profile-fixed-cache \
  --output /path/to/new-run
```

Build the wrapper at the explicit path selected above, or supply its actual
location. Correctness status and `goal_acceptance` are separate. Acceptance
requires the fresh root-to-packet boundary, elapsed time below 300 seconds,
matching Filecoin ceremony/receipt evidence from both stages and their CPU
verifiers, actual trace heights within the ceiling, one wrapper plus its merged
table, and an actual packet below 3,000 bytes. Each KZG prove phase must show
positive parsed kernel events on all four selected devices; malformed event
records and utilization samples are insufficient. All runs hash both stages'
proof, packet and profile artifacts, including runs without a parity baseline.
The [admission validation](../experiments/kzg-cuda-validation/parallel-advice-20261009/report.json)
passes twelve focused harness tests and rejects an actual CPU-only first-stage
binary before any proving phase. Both small staged/proving fixtures pass, and
all three rebuilt CUDA executables pass compiled-feature preflight. These are
local correctness and build checks; they do not establish runtime GPU dispatch
or final goal acceptance.

For the separately labeled prepared-service boundary, add the following to
the acceptance command after validating FRI coexistence with both idle workers:

```sh
  --retained-workers \
  --worker-seed-fri /path/to/compatible-compressed-fri \
  --worker-seed-claims /path/to/independent-seed-claims.bin
```

The seed proves both initial requests using the selected setup. Their work is
reported in `worker_startup_wall_seconds`; `pipeline_wall_seconds` starts with
fresh FRI compression after preparation. Startup plus that fresh pipeline is
reported too: it includes the two warmup proofs and a second request, rather
than timing a single cold request. Existing one-shot commands continue to
measure cold processes.
Retained-mode acceptance additionally checks genuine preparation, request-local
GPU evidence, quiescence on all selected devices, fresh input linkage and key
compatibility. The measured request cannot borrow warmup CUDA events.

`--baseline` requires identical input hashes and both stages' proof, packet
and profile bytes. Supply a run from the same current profile; the historical
recursive profile predates the public-claim constant cleanup and cannot serve
as a byte-parity baseline for the current source. The populated partition-pipeline fixed cache is at
`/opt/dlami/nvme/multi-stark-kzg-prefetch-20261009/fixed-cache`; it uses the
development SRS at
`/opt/dlami/nvme/multi-stark-kzg-perf-20261009/dev-srs-cache`.
The earlier instrumented baseline retains its own fixed cache under
`multi-stark-kzg-perf-20261009/fixed-cache`. Fixed caches bind their exact
executables; the SRS cache binds its parameter recipe independently.

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
