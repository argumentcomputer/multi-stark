# Single-GPU KZG handoff — 2026-10-13

On `sb/kzg`, the saved Init root-to-packet pipeline completes on one RTX PRO
6000 Blackwell under a 230 GiB host-memory limit. Bounded GPU constraint passes
reduce the matched complete-pipeline median from **431.609 s to 336.485 s**
(95.124 s, **22.04%**). The implementation, benchmark driver, reports and
correctness evidence accompany this handoff.

## What the benchmark proves

The input is the saved Ix Init aggregation-root proof and its independently
supplied claims in
`experiments/kzg-cuda-validation/resume-20261009/root-artifacts`. Each trial
creates three fresh proofs: FRI compression of that root, a KZG proof verifying
the FRI proof, and a recursive KZG proof verifying the first KZG proof. Original
application/root proving is outside this boundary. This uses the development
public-degree v4 profile and development SRS; production Filecoin ceremony
acceptance has not been established.

The measured host has one 97,887 MiB GPU, 32 logical CPUs and approximately
249 GiB RAM, with no swap. CPU parallelism is unchanged. A user-systemd scope
enforces `MemoryMax=230G` and `MemorySwapMax=0`.

## Measured result

One excluded warmup populated fixed preprocessing. Four measured trials ran
in baseline/candidate/candidate/baseline order, with identical executables,
sources and inputs. Both variants used cached SRS/fixed data, cold worker
processes and fresh witnesses. Only `MULTI_STARK_KZG_CUDA_PASSES` differed:
`0` for the existing fallback and `1` for the new passes.

| Process-lifetime median | Existing fallback | GPU passes |
| --- | ---: | ---: |
| FRI compression | 60.565 s | 60.590 s |
| First KZG proof | 163.445 s | 164.166 s |
| Recursive KZG proof | 207.593 s | 111.723 s |
| Complete pipeline | **431.609 s** | **336.485 s** |

The timing includes preparation, cache loading, fresh traces, commitments,
proving, in-process verification and worker shutdown. Independent CPU
verification adds about 0.4 s. There are two measured samples per variant on
one fixture; this is not a throughput or workload-distribution study.

Recursive peak RSS falls from 226.083 GiB to about 144.63 GiB. The first KZG
stage still peaks near 217.4 GiB, and cold fixed preprocessing can raise the
recursive peak; **144.63 GiB is not the pipeline's RAM requirement**. File-cache
reclaim occurs under the 230 GiB cap. Recursive GPU usage reaches about
69.3 GiB; FRI remains the largest GPU consumer, near the card's capacity.
GPU samples are two seconds apart and can miss shorter peaks. All five runs
completed with zero OOM events.

The earlier 521.094 s capacity run had cold fixed preprocessing. Use the
matched 431.609 s baseline above to attribute the speedup. See the
[paired report](../experiments/kzg-cuda-validation/single-gpu-passes-20261013/report.json)
and [performance notes](kzg-performance.md#single-gpu-constraint-pass-comparison)
for raw samples, memory attribution and timing boundaries.

## Implementation map

| Area | Entry points | Behavior |
| --- | --- | --- |
| Stage scheduling | [benchmark harness](../experiments/kzg-cuda-bench.py) | `--sequential-workers` starts and closes each worker before the next stage; disables partition prefetch and caps recursive coefficient residency at 8 GiB for one selected GPU. |
| Host evaluation reconstruction | [PCS adapter](../src/ark_adapter/pcs.rs), [domain selectors](../src/ark_adapter/domain.rs) | Limits transformed-column scratch to 1 GiB or one column, whichever is larger; computes selectors by tile. A `2^27` scalar column is 4 GiB. |
| Quotient spill | [prover configuration](../src/ark_adapter/config.rs), [scratch storage](../src/ark_adapter/quotient_scratch.rs), [example setup](../examples/support/kzg_setup.rs) | Spills completed cosets to NVMe, releases evaluation matrices, then interleaves cosets. Scratch files have RAII cleanup. |
| GPU pass planning | [pass planner](../src/ark_adapter/cuda/passes.rs), [quotient encoder](../src/ark_adapter/cuda/quotient.rs), [lookup dispatch](../src/ark_adapter/cuda/lookup.rs) | Prunes transitive dependencies and admits one bounded operation under the existing device lease and memory policy. |
| GPU execution | [pass kernels](../cuda/kzg_passes.cuh), [shared quotient arithmetic](../cuda/kzg_quotient.cuh) | Reuses evaluation and scan scratch across lookup groups and quotient passes, retaining the existing proof layout. |

The recursive profile has 15 fixed, 3 main and 4 lookup columns. A lookup pass
needs at most 7 input columns; a quotient pass needs at most 9. Conservative
operation bounds are 57.010 GiB and 49.010 GiB respectively, including 1 GiB
of headroom. Existing residency is accounted for, the SRS cache can be
evicted, and the quarter-VRAM reserve remains in effect. The implementation
retains dense host coefficients and host-backed openings.

`MULTI_STARK_KZG_CUDA_PASSES=1` is the default and tries the pass layout when
the dense operation fails admission. `0` retains the previous fallback;
`force` exercises the pass path on small admitted test fixtures. New passes
require one selected GPU. Unsupported shapes and failed preflight admission
retain the existing fallback; native execution errors remain errors.

Sequential mode requires `--kzg-devices` or
`MULTI_STARK_KZG_CUDA_DEVICES`. This prevents implicit selection from bypassing
the recursive residency cap and allowing a roughly 48 GiB coefficient cache
to block pass admission. The guard was added after the benchmark; all measured
runs explicitly selected GPU 0, and 26 harness tests cover the final code.

## Reproduce and inspect

Build and single-proof commands are in
[single-GPU execution](kzg-performance.md#single-gpu-execution-with-bounded-host-memory).
The measured toolchain was Rust 1.98.1, CUDA 13.3 and
`MULTI_STARK_CUDA_ARCHS=120`. Run from the repository root, explicitly select
GPU 0, use sequential workers, and put output/scratch on NVMe. Fixed caches
bind executable digests; rebuilding can invalidate them.

Useful artifacts on the measured host:

| Path | Contents |
| --- | --- |
| `/opt/dlami/nvme/kzg-gpu-passes-20261013` | Five complete runs, pinned executables, proof artifacts, logs, memory samples and analysis. |
| `/opt/dlami/nvme/kzg-gpu-passes-20261013/fixed-cache` | Fixed preprocessing for the measured executables. |
| `/opt/dlami/nvme/kzg-single-gpu-pilot-20261013/srs-cache` | Cached development SRS, including the `2^27` recursive domain. |
| `/opt/dlami/nvme/kzg-single-gpu-full-spill-20261013` | Earlier successful cold-fixed capacity run. |

Compact JSON reports and the comparison scripts are committed under
`experiments/kzg-cuda-validation/single-gpu-passes-20261013`; large binaries,
caches and logs remain machine-local. The
[comparison driver](../experiments/kzg-cuda-validation/single-gpu-passes-20261013/run.py)
copies executables, applies the memory scope, checks GPU dispatch and proof
parity, and runs the warmup plus both pairs. It accepts a fresh `--output`
directory and `--budget-seconds` (default 2700). Repository and SRS paths are
machine-specific. The recorded five-run comparison took about 33 minutes;
obtain permission before starting another long build or benchmark sweep.
The [analysis script](../experiments/kzg-cuda-validation/single-gpu-passes-20261013/analyze.py)
also embeds the original result directory; update it for a new run.

## Validation and invariants

The [validation record](../experiments/kzg-cuda-validation/single-gpu-passes-20261013/validation.json)
records 3 planner tests, 4 forced-pass GPU parity tests, 2 dense-path GPU
regressions, 2 CUDA memcheck cases with zero errors, and 26 Python harness
tests. Earlier targeted checks cover bounded FFT reconstruction, tiled
selectors, scalar/portable parity, and spill proof parity/cleanup.

All five full runs passed independent CPU verification, negative checks and
recursive external pairing checks. KZG proof, packet and profile bytes agree
across variants and with the preceding verified single-GPU baseline. The
final proof is 1,909 bytes and the packet is 2,181 bytes. The reports retain
executable, input and proof hashes.

Preserve lookup prefix order across rows/groups, the zero-message convention,
exact quotient Horner weights, next-row wrapping, absolute selector row
indices and constant-column elision. Admission arithmetic must include all
scratch and resident allocations. A failed native operation must not silently
retry through the host fallback after partial execution.

## Remaining work

The compact fixed-input representation, sole-copy device storage, GPU opening
folds and complete-residency design are unimplemented. The new pass layout is
limited to one GPU; existing multi-GPU paths remain separate. No production
SRS acceptance or original root-generation benchmark is included here.

Start further optimization by attributing the first KZG stage's approximately
164 s and 217.4 GiB, then remaining recursive host preparation. Treat the
first-stage RAM peak and FRI VRAM peak as independent capacity constraints.
Require a complete-proof comparison with matched caches, binaries, inputs,
memory limits and correctness checks before accepting more complex storage
or scheduling changes. Kernel milliseconds or transferred bytes alone do not
establish an end-to-end benefit.
