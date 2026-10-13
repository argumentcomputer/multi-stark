# Remaining performance opportunities on `sb/kzg`

Review date: 2026-10-13. Source revision: `1e2182f` on `sb/kzg`.
The review includes the branch changes, performance documentation, CUDA and
Rust execution paths, and the pinned sppark dependency at
`e10e107673aa22861f0f8b9758fc62169ab919ae`. No builds or benchmarks were run.
Timings below are recorded results from the repository documentation, not
new measurements. The latest integrated run's raw artifacts are not included
in this checkout; its reproduction scripts are present.

**The selected architecture remains the recursive KZG verifier. GPU residency
is a hypothesis to validate, not an established largest latency saving.**
Sppark's main algorithms are already in use. The host-backed PCS creates
avoidable transfers, but their removable wall time needs attribution against
allocation, host copies, reuploads, and MSM placement. FRI frontend reuse
remains a separate candidate for complete root-to-packet latency.

The [single-GPU residency design](kzg-gpu-residency-design.md) develops the
capacity question for the user's 96 GB RTX PRO 6000 Blackwell. Current dense
representations do not fit. A proposed recursive schedule with compact fixed
inputs and buffer reuse has a modeled **78.44 GiB subtotal**, including its
12 GiB SRS, before runtime/scratch overhead. It needs new consumer scheduling
and a measured memory plan; output handoff alone does not establish that fit.

The latest documented run used four RTX PRO 6000 Blackwell GPUs and 96
logical CPUs:

| Prepared request phase | Time |
| --- | ---: |
| FRI compression, including process startup and frontend | 57.173 s |
| First KZG stage, including fresh witness and GPU release | 30.691 s |
| Recursive KZG stage, including fresh witness and GPU release | 34.554 s |
| Export and other orchestration | 0.037 s |
| Complete request | **122.455 s** |

Worker preparation took another **404.353 seconds**, including a complete
warmup proof at each KZG stage. These are development-parameter measurements;
one cold request and acceptance with authentic Filecoin parameters remain
unmeasured. Preparation plus the subsequent fresh request is not the latency
of one cold request. The latest direct fixed-input change postdates the
integrated run. See the [recorded timing boundaries](kzg-performance.md#integrated-development-v4-worker-pipeline).

Several older sections discuss the superseded 14-minute pipeline. Historical
transfer volumes and isolated speedups should not be added to estimate savings
against the current 122.455-second request.

## Hardware and measurement tracks

| Track | Evidence and scope |
| --- | --- |
| Current single-card target | User-confirmed 96 GB RTX PRO 6000 Blackwell; reviewer reports 249 GiB host RAM. Validate admitted first-stage partitions and a complete first-stage proof before attempting the large wrapper. |
| Single-GPU pilot | Reviewer-reported cold, legacy-profile run: 38.1 GiB lookup/quotient output downloads, 0.79 s aggregate GPU download events. The raw report was not supplied; host-copy, allocation, and subsequent-upload costs still need attribution. |
| Prepared v4 reference | The four-GPU timings above; sampled sum of process RSS peaks at 449.249 GiB and can double-count shared pages. This does not establish fit on the current host or performance on one GPU. |
| Distributed wrapper | Current large lookup/quotient paths require four devices and their topology/memory admission. Output ownership and MSM placement changes need validation on that hardware. |

Keep the cold legacy pilot and prepared v4 reference separate. Preserving
host mirrors in the first handoff experiment removes neither the pilot's
downloads nor their host-RAM allocation. The 0.79 s event sum cannot justify
a large latency claim, and overlapping event sums are not critical-path time.

## 1. Establish the live-set and placement plan

Before changing an output API, account for persistent coefficients or compact
sources, evaluation buffers, consumer outputs, SRS points, MSM/FFT/scan
workspace, and allocator/runtime overhead at each phase. Admission must cover
the output's subsequent commitment and lifetime, not only its construction.
Current coefficient residency is not individually evictable; only SRS points
and MSM workspace participate in `prepare_budget()` eviction.

The single-card design starts with same-owner commitments and bounded scratch.
The distributed design must choose MSM placement explicitly: quotient output
currently ends on the primary card, while resident MSMs execute on the buffer's
owner. A direct handoff could concentrate work on that card and lose the
host-backed path's coefficient-range parallelism. Location, SRS ranges,
admission, and eviction are prerequisites, not follow-up optimizations.

Global transcript barriers also prevent an arbitrary per-partition pipeline:
all lookup commitments/totals precede the constraint challenge, and all
quotient commitments precede the opening point. Retention must be planned
across those boundaries. See the [worked schedule and interface design](kzg-gpu-residency-design.md).

## 2. Carry device-owned polynomial buffers into commitment

Both GPU lookup and quotient consumers finish by downloading coefficient
columns. Rust wraps those outputs in `Vec<Vec<Fr>>`, then commitment uploads
them again. At the recursive `2^27` shape, the four lookup columns and two
quotient chunks alone produce **24 GiB of downloaded coefficients**, before
their subsequent uploads and opening work.

For admitted single-device first-stage consumers, return owned device handles
and pass them to a planned same-owner MSM. Initially, retain host mirrors for
compatibility while eliminating the reupload and duplicate device allocation.
Measure the complete producer-to-consumer path before expanding that change.
Make mirrors lazy only once openings, spill handling and checkpoint paths
support the representation. That later step can avoid allocating and
initializing large host output vectors.

The important representation change is a polynomial with explicit device
ownership, domain/layout information, completion dependencies and an optional
host mirror. Transcript barriers require commitments and small scalar values
to reach the CPU; they do not require entire polynomial columns to do so.

Source evidence:

- [`CommittedMatrix`](../src/ark_adapter/pcs.rs) stores host coefficient
  vectors and optional resident copies. `commit_trace_coefficients` creates a
  new host-backed matrix, and `commit_matrix_columns` hydrates residency.
- [Lookup construction](../cuda/kzg_lookup.cuh) downloads every output
  coefficient column after its IFFT.
- [Distributed quotient construction](../cuda/kzg_quotient_distributed.cuh)
  downloads both final coefficient chunks and releases the device buffer.
- The Rust [lookup](../src/ark_adapter/cuda/lookup.rs) and
  [quotient](../src/ark_adapter/cuda/quotient.rs) APIs return host vectors.

This is the first ownership experiment after the admission/placement design.
Keep the existing distributed handoff until its MSM decomposition has an
explicit replacement. It is not the complete-residency implementation: the
large recursive consumers need the capacity-changing work described in the
[single-card design](kzg-gpu-residency-design.md#worked-recursive-schedule).

## 3. Finish the resident opening pipeline

The desired sequence is device folding, division, then a partitioned witness
MSM. Currently, `open()` evaluates polynomials on GPU but folds every
polynomial into a host `combined` vector. With multiple GPUs, division uploads
that vector, downloads the quotient, and MSM uploads it again. The resident
division-to-MSM shortcut runs only with one selected device.

Design folding together with coefficient-range ownership and SRS placement,
so the final witness MSM is partitioned once. Preserve transcript weights,
mixed-degree padding and repeated opening points. A device fold alone does
not establish a faster opening.

On the single-card target, first extend the existing division/MSM path to
accept a device-produced fold. Process point batches under a memory budget.
Compact fixed data also permits folding original-domain fixed values before
one IFFT, instead of expanding every fixed coefficient column. The separate
design explains the ordering and mixed-domain requirements.

The docs record an earlier per-device folding prototype that **regressed
12.2%**, from 324.213 to 363.856 milliseconds on its opening fixture, because
it repeated full MSMs and SRS uploads. Reuse that result as a constraint on
the next design, rather than repeating the same decomposition.

Sources: [`open` and `fold_polynomial`](../src/ark_adapter/pcs.rs),
[`single_device_opening` and division](../src/ark_adapter/cuda.rs), the
[cached-SRS opening path](../src/ark_adapter/cuda/srs_cache.rs), and the
[prototype result](kzg-performance.md#sppark-division-and-point-uploads).

## Other opportunities: retain the FRI frontend and key

Both KZG stages retain compiled state, but
[`ix_root`](../examples/ix_root.rs) still constructs its circuit, lowers it,
generates fixed inputs and calls `System::new` on each invocation. Reuse
compatible verifier plans, lowered layouts and preprocessing while generating
a fresh assignment. Compatibility must include the trusted verifier key,
proof shape, constant message bindings and implementation/profile identity.

FRI is now the largest request phase. Its current source still exposes this
repeated preparation, although the exact removable portion of the latest
57.173-second phase is not established by the older frontend timings.

Beyond frontend reuse, distribute resident FRI data together with its
consumers. The existing auxiliary-GPU path computes host-backed LDEs and
returns them to host; it does not distribute resident codewords, Merkle trees,
lookups or quotient work. GPU 0 reached approximately **94.5 GiB** in the
integrated run. Retaining useful evaluations on other devices could avoid
reconstruction and transfers, but requires extending consumer ownership too.

Sources: [FRI placement and measurements](kzg-performance.md#fri-memory-placement),
[FRI PCS](../src/cuda/pcs.rs), and
[auxiliary LDE offload](../src/cuda/offload.rs).

## Generate expanded trace columns on GPU

The fused KZG path already removes witness-file round trips. It still
generates host row-major matrices, transposes them into columns, uploads them
and interpolates. Lookup construction later FFTs those coefficients back
into the original evaluation domain.

Start with CPU-generated assignments plus reusable wire-layout metadata,
gathering directly into GPU columns. Retain or regenerate original
evaluations where that is cheaper than reconstruction. This is a smaller
first step than porting all witness advice to CUDA; compact BLAKE3
intermediates and table multiplicities can follow.

Retention must be selective. At `2^27`, each scalar column occupies 4 GiB,
and original evaluations compete with coefficients, SRS points and consumer
scratch. Tiled generation bounds producer scratch but does not remove the
full column required by the current FFT.

Sources: [transpose and interpolation](../src/ark_adapter/pcs.rs),
[lookup reconstruction](../cuda/kzg_lookup.cuh), and the existing
[trace/witness design](kzg-performance.md#gpu-trace-and-witness-design).

## Extend placement and cross-request reuse

Per-proof ownership, MSM placement, and admission belong to the prerequisites
above. The additional opportunities here require their own comparisons.

Coefficient retention currently chooses the device with the fewest retained
bytes. Distributed consumers use a different column/pair ownership scheme.
Same-pair peer reads help, but placement still causes replicated uploads and
uneven work. A plan that considers the next lookup, quotient and MSM consumers
can do better than balancing retained bytes alone.

There is also cross-request opportunity. Prepared workers retain host keys
but explicitly discard device coefficients, SRS points and MSM workspaces
after requests. Selective retention on cards with available capacity could
avoid rehydration. It needs joint accounting for both KZG workers and FRI;
keeping everything resident is incompatible with FRI's current primary-GPU
footprint. The measured distributed wrapper uses an 8 GiB coefficient cap per
card to admit its working set, so increasing residency independently can
force consumer fallback.

After placement is settled, consider bounded stream tasks, completion events
and persistent scratch/pool capacity. An FFT batch already has two work slots,
but other operations wait for an exclusive device lease. Sppark MSM captures
shared internal streams, and transfer rings are keyed by device and lane;
increasing concurrent leases alone would be unsafe. Completion callbacks also
need an explicit host dispatch and buffer-lifetime design.

Sources: [`Devices::reserve` and memory release](../src/ark_adapter/cuda.rs),
[distributed column loading](../cuda/kzg_distributed.cuh),
[recursive worker release](../experiments/kzg-wrap/src/outer.rs), and
[scheduling constraints](kzg-performance.md#cuda-scheduling-and-memory-constraints).

## Improve the existing lookup and quotient kernels

These are concrete, bounded opportunities:

- **Replace the full `L_first` buffer and its NTT.** On a trace-sized coset,
  the unnormalized selector is `(shift^n - 1) / (point - 1)`. Tiled batch
  inversion could replace a 4 GiB selector buffer per GPU at `2^27`, also
  improving memory admission. Preserve the existing last-row selector
  normalization and next-row semantics.
- **Measure larger sweep grids.** The current cap is 256 blocks of 128
  threads. More blocks may improve latency hiding, at the cost of more
  per-block DAG scratch. Update admission together with launch geometry.
- **Specialize common constraint graphs.** The interpreter already reuses
  slots according to liveness, but every intermediate still passes through
  global scratch. Keeping short expression chains in registers could reduce
  traffic; register pressure must be measured.
- **Compare tiled or hierarchical batch inversion with the two full product
  scans used by lookup construction.** This trades inversion work against
  scratch and memory passes. It is an experiment, not an established faster
  replacement.

Sources: [grid cap and scratch admission](../src/ark_adapter/cuda/quotient.rs),
[interpreter and selector generation](../cuda/kzg_quotient.cuh), and
[distributed lookup scans](../cuda/kzg_lookup_distributed.cuh).

The latest recursive lookup and quotient consumers take **4.154 and 4.598
seconds**, respectively, including transfers. Their combined 8.752 seconds
bounds the direct latency opportunity in those consumers. Memory savings
may additionally help other phases; isolated arithmetic improvements cannot
be extrapolated to the whole request.

## What remains to adopt from sppark

The branch already uses `dev_ptr_t`, resident-input `msm_t`, in-place NTTs,
prefix scans, batch inversion, polynomial evaluation and `div_by_x_minus_z`.
It also has optional resident SRS ranges and reusable MSM workspaces. Those
are implemented capabilities, not remaining algorithm migrations.

The valuable next step is exposing native buffers and lifetimes through the
PCS. `NR`/`RN` transform ordering, compatible fused LDE paths and batching
small transforms are secondary candidates after the larger transfers
disappear. The KZG transform currently requests natural-to-natural ordering;
avoiding permutation passes requires carrying the chosen ordering through
every consumer. Fused coset operations must also preserve arbitrary shifts.

Stock stream notifications can support completion plumbing, but sppark's
`channel_t` is unbounded and does not supply memory admission or backpressure.
The multi-point evaluation kernel has a recorded shared-memory race; the
current single-point path is intentional.

Sources: [native integration](../cuda/kzg.cu),
[resident SRS integration](../cuda/kzg_srs.cuh), and
[sppark adoption notes](kzg-performance.md#sppark-primitives-to-adopt).

## Larger circuit and architecture opportunities

Fewer first-stage commitments, particularly through a narrower BLAKE3 layout,
could reduce first-stage movement and recursive point checks/MSM terms
together. Count the combined circuit first: the current wrapper uses
118,634,362 rows, so modest reductions leave its padded `2^27` FFT/MSM sizes
unchanged. Reaching `2^26` requires approximately 43.4% fewer used rows.

Alternative foreign-field limb widths and recursive MSM layouts merit sizing,
with their arithmetic bounds checked separately. The older estimates include
degree-check work already removed by the selected v4 policy and cannot be
reused unchanged. Subgroup-check removal is a separate soundness decision.

Terminal backend replacement is deferred. The active design retains both KZG
stages and their recursive verifier; earlier PLONK/PILFFLONK research remains
background rather than part of this implementation plan.

Sources: [current circuit-size discussion](kzg-performance.md#recursive-circuit-reductions)
and [terminal backend review](plonk-terminal-backend-review.md).

## Documentation corrections and review focus

The older implementation review still lists statement-dependent recursive
constants as unfinished. Current code separates public slots from constants,
excludes public values from the frontend identity and tests reuse across
different statements. That is already addressed. See the
[current construction](../experiments/kzg-wrap/src/native_verifier.rs),
[plan identity](../experiments/kzg-wrap/src/native_verifier/plan.rs), and
[A/B/A reuse tests](../experiments/kzg-wrap/src/native_verifier/reuse_tests.rs).

Review the [capacity calculation](kzg-gpu-residency-design.md#capacity-conclusion)
and the complete producer/consumer schedule before the ownership API. Check
fixed reconstruction, the lookup row/group prefix, quotient constraint
dependencies, and representation lifetimes. Keep the single-GPU first-stage
work distinct from distributed-wrapper validation. Comparisons must include
allocation, host copies, transfers, additional FFTs, SRS misses, consumer/MSM
work, fallback counts, and complete-proof wall time, while preserving proof
parity and reporting peak GPU/host memory. Aggregate CUDA-event intervals
overlap and are not an additive wall-time saving.
