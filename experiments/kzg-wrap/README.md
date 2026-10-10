Recursive verification of the saved Init KZG proof, using native Fr Plonkish
arithmetic and bounded Fq limbs. The circuit checks the transcript, AIR,
lookups, curve membership, subgroups and both KZG batching equations. Its
public claim contains the 18 Init words and the profile's compressed pairing inputs.
The final verifier checks both external pairing equations against fixed keys.

The current Filecoin v4 migration uses two pairing inputs and a `2^27` maximum
computation trace. Its setup requirements, exact construction counts and
validation status are maintained in [the KZG performance document](../../docs/kzg-performance.md#decision-and-next-evidence).
The current `stage-and-prove-many` command retains a compatible frontend and
outer key across fresh requests; see its [statement-file interface and validation scope](../../docs/kzg-performance.md#retain-the-recursive-frontend-and-outer-key).
The `serve` command accepts sequential JSONL requests and releases idle GPU
memory between proofs; see the [worker protocol and measured boundaries](../../docs/kzg-performance.md#schedule-requests-through-retained-workers).
The `count-saved` command checks a Filecoin saved proof's recursive layout and
projected packet size before staging; see its [usage and acceptance scope](../../docs/kzg-performance.md#count-a-saved-proof-with-the-filecoin-setup).
The measurements and commands below describe legacy development profiles;
set `MULTI_STARK_KZG_SETUP=development` explicitly to reproduce them.

The input is the saved 1,530,149-byte FRI compression of the Init root proof in
`../init-fri-artifacts`. Rebuild its intermediate KZG proof with:

```sh
export MULTI_STARK_KZG_SETUP=development
cargo build --release --features kzg,parallel --example init_fri_kzg_prove
target/release/examples/init_fri_kzg_prove stage experiments/init-fri-artifacts target/init-fri-kzg
target/release/examples/init_fri_kzg_prove prove target/init-fri-kzg
target/release/examples/init_fri_kzg_prove verify target/init-fri-kzg
```

The equivalent saved intermediate proof is in `../init-fri-kzg-artifacts`.
Build and run its recursive wrapper with:

```sh
cargo test --offline --release --manifest-path experiments/kzg-wrap/Cargo.toml
cargo build --offline --release --manifest-path experiments/kzg-wrap/Cargo.toml
experiments/kzg-wrap/target/release/init-kzg-wrap stage experiments/init-fri-kzg-artifacts target/init-kzg-recursive
experiments/kzg-wrap/target/release/init-kzg-wrap prove target/init-kzg-recursive
experiments/kzg-wrap/target/release/init-kzg-wrap verify target/init-kzg-recursive
```

The 53,157-byte input passes the complete circuit and pairing checks:
237,408,925 rows, one padded 2^28 computation trace plus a shared lookup table.
The outer proof is 2,053 bytes; its complete packet is 2,757 bytes. Setup and
proving took 10,011 seconds with 485.5 GiB peak RSS. Separate verification
passed, including external pairings and tampering tests (13.8 ms after setup).
See `../init-kzg-recursive-circuit.json` and `../init-kzg-recursive-proof.json`.

The saved packet and verification metadata are in `../init-kzg-recursive-artifacts`.
Run `init-kzg-wrap verify experiments/init-kzg-recursive-artifacts` to recheck
them with explicit development setup selection. Without a populated development
cache, verification regenerates those parameters; the historical fixed-base
setup took about 100 seconds for this domain on the Blackwell host. The original
wrapper's affine formulas rejected exceptional intermediate sums. The current
[MSM completeness repair](../../docs/kzg-performance.md#decision-and-next-evidence)
supports those cases while retaining subgroup checks.

Known-trapdoor development SRS only; these artifacts are not production-secure.

## CUDA proving

Measurements and the exact validation scope are recorded in
[`../kzg-cuda-results.json`](../kzg-cuda-results.json).

The complete aggregation-root-to-packet run took **14m29s including FRI
compression**, using populated fixed-preprocessing and development-SRS caches.
That is 37.4% less elapsed time than the preceding 23m10s run on the same
fixture and four RTX PRO 6000 Blackwell GPUs:

| Phase | Previous cached run | Resident-coefficient run | Peak host RAM |
| --- | ---: | ---: | ---: |
| FRI compression | 2m51s | 2m52s | 232.4 GiB |
| FRI → KZG staging | 3m18s | 3m19s | 63.6 GiB |
| FRI → KZG proving | 6m27s | 2m36s | 190.7 GiB |
| Recursive KZG staging | 2m19s | 2m18s | 84.2 GiB |
| Recursive KZG proving | 8m15s | 3m24s | 464.2 GiB |

Both stages passed independent CPU verification, external pairings and
tampering checks. Including the independent CPU checks, the run took 14m57s.
The final proof is 2,053 bytes; its complete packet is 2,709 bytes. Both KZG
proofs, packets and profile IDs match the previous verified run and the fresh
fixed-preprocessing run byte for byte. All witnesses and proofs were regenerated.
The same backend took **20m01s** with fresh fixed preprocessing and cached SRS;
including its separate CPU verification processes took 20m29s.

The [GPU-feeding results](../kzg-cuda-gpu-feeding-results.json) retain the
comparison, primitive checks and transfer counters. The
[cached full report](../kzg-cuda-validation/gpu-feeding-cached-report.json) and
[fresh-fixed full report](../kzg-cuda-validation/gpu-feeding-cold-report.json)
identify frozen binaries, input hashes, source snapshots and phase logs.
The preceding [23m10s report](../kzg-cuda-validation/sub30-complete-report.json)
and [earlier iteration record](../kzg-cuda-sub30-results.json) remain available.

The recursive run retains about 160 GiB of coefficients across the devices;
sampled peak device memory was 56.7 GiB per GPU. The conservative half-VRAM
budget leaves four large columns on the host. Reused stage-one forward FFTs
and opening evaluations upload no coefficients; recursive FFTs and evaluations
each upload 32 GiB for the columns that did not fit. CPU staging and graph
sweeps still leave substantial idle time: sampled mean utilization in the
cached KZG proving phases was about 6–9% per GPU. Residency and transfer
improvements reduced wall time; this is not a fully device-only prover.

This is a cached Init fixture measurement. Its recursive cache binds the
public statement; it predates the reusable frontend's typed claim schema.
The current worker's small-fixture parity checks do not establish reuse or
latency for this production-size profile. Cache population and upstream root aggregation are
outside the timing boundary. The regenerated root certifies the same Init
claim but produces a 19-circuit FRI profile, versus the historical 22; these
timings are not an identical-workload comparison with the older runs below.
The SRS remains development-only.

The archived 23m10s run's `constraint_seconds` events precede the constraint loop
and must not be used as sweep timings; total phase and per-coset wall times
remain valid. A [corrected one-trace diagnostic](../kzg-cuda-validation/sub30-corrected-timer-trace.txt)
at 2^24 rows measured 13.15 seconds for quotient commitment, including 9.04
seconds reconstructing evaluations and 1.53 seconds preparing and evaluating
constraints. Lookup commitment took 6.61 seconds: evaluation reconstruction
3.35, trace construction 2.03, and commitment 1.20. The result supports
prioritizing reconstruction and data movement over a standalone constraint
kernel. GPU sample summaries now include mean utilization and the zero-sample
share, with the sampling boundary stated explicitly.

A subsequent GPU-feeding diagnostic used the same staged 2^24 trace and SRS
recipe for all three builds:

| Diagnostic phase | Before | Direct buffers and parallel decoding | Residency and pinned queues |
| --- | ---: | ---: | ---: |
| Whole diagnostic | 44.63 s | 27.48 s | 25.06 s |
| Lookup commitment | 6.63 s | 3.85 s | 3.20 s |
| Quotient commitment | 13.21 s | 6.48 s | 4.83 s |

The final diagnostic includes CUDA-event profiling. All 55 forward FFTs read
resident coefficients, eliminating 27.5 GiB of coefficient uploads. Downloads
remain for CPU consumers. This is a one-trace diagnostic, not an end-to-end
speedup claim. The retained-coefficient tests cover FFT/MSM parity, queue reuse,
cosets and repeated opening points. The explicit 2^29 coset FFT/inverse check
passed in 3.58 seconds, with about 46–48 GB/s measured DMA throughput.

An earlier optimization pass reduced the same historical compressed-FRI
workload to **29m00s with fresh development SRS**, preserving both CPU packets
byte for byte. That measurement excludes FRI compression and does not meet
the complete 30-minute target. Its report is in
[`../kzg-cuda-sub30-results.json`](../kzg-cuda-sub30-results.json).

On four RTX PRO 6000 Blackwell GPUs, the fresh two-stage run took **54m30s**
from the saved compressed FRI proof to the verified final packet:

| Phase | Wall time | Peak host RAM |
| --- | ---: | ---: |
| FRI → KZG staging | 6m05s | 72.4 GiB |
| FRI → KZG proving | 14m27s | 205.6 GiB |
| Recursive KZG staging | 6m26s | 184.1 GiB |
| Recursive KZG proving | 27m33s | 466.5 GiB |

Both proofs, packets, and profile identifiers match the saved CPU artifacts
byte for byte. The final proof is 2,053 bytes, and its packet is 2,757 bytes.
Proving includes fresh development SRS generation, checkpoint writes,
verification, external pairings, and tampering checks. No SRS cache was used.
The preceding FRI compression was not rerun; adding the reported six minutes
gives approximately 60.5 minutes overall. Independent CPU verification is
recorded separately in the results file.

The first-stage lookup commitments took 100.7 seconds in total, with the
large partitions at 7.33–8.01 seconds each. The recursive 2^28-row lookup
commitment took 110.9 seconds. Fresh SRS generation took 40.3 seconds and
641.8 seconds respectively. Those historical figures used fresh SRS
generation; the optional development cache below avoids repeating it.

`kzg-cuda` accelerates BLS12-381 MSMs, scalar-field FFTs, polynomial
evaluations, and opening-witness division. It uses the pinned
`argumentcomputer/sppark` fork and preserves the CPU transcript, verifier,
checkpoint format, and proof bytes. Development SRS generation and circuit
construction still run on the CPU.

Build both stages for the RTX PRO 6000 Blackwell (`sm_120`):

```sh
MULTI_STARK_CUDA_ARCHS=120 cargo build --release --locked \
  --features parallel,kzg-cuda --example init_fri_kzg_prove
MULTI_STARK_CUDA_ARCHS=120 cargo build --release --locked \
  --manifest-path experiments/kzg-wrap/Cargo.toml --features kzg-cuda
```

Run from the repository root. The output directory must be new; the driver
records separate staging/proving times, peak RSS, binary/source hashes, and
exact comparisons against the saved CPU packets. It then verifies both
packets in separate processes with the CPU backend:

```sh
python3 experiments/kzg-cuda-bench.py --output target/kzg-cuda-run
```

The input is the saved compressed FRI proof. This does not rerun the preceding
FRI compression. Proving times include development SRS generation,
checkpoint writes, and the harness's verification and tampering checks.
Independent CPU verification is timed separately from the four-stage pipeline.
The staged traces and coefficient checkpoints require substantial disk space.
On machines with local NVMe, put the output directory on that volume; the
default-throughput EBS root disk can dominate checkpoint writes. For example,
use `--output /opt/dlami/nvme/kzg-cuda-run` on this box. GPU memory and
utilization are sampled every two seconds in `gpu-samples.csv`.
Interrupting the driver with Ctrl-C or SIGTERM stops its active proving
process group and GPU monitor; partial logs remain in the output directory.

All visible supported GPUs are used by default. Interpolation retains immutable
coefficient columns on their assigned device when memory admission permits.
MSMs, subsequent coset FFTs and opening evaluations reuse those coefficients;
checkpoint-loaded columns enter the same cache lazily. Constants consume no
resident allocation. Host coefficients remain available for checkpoints and
for columns that do not fit the resident budget.

Each device has an exclusive queue with up to two FFT work slots. Uploading
the next column can overlap the preceding transform, and the work allocation
is reused within a queue. Jobs reading resident coefficients return to the
owning device; host inputs are distributed among queues. Each MSM queue
uploads an SRS chunk once for its local columns. Resident scalars feed sppark
directly, with any scalar normalization performed on a temporary device buffer.

Transfers use persistent pinned rings, adapted from the FRI backend: four
16 MiB chunks in each direction per lane, with at most two lanes per device
(256 MiB pinned host memory per device). Parallel host copies and CUDA events
protect each chunk until its DMA completes. This avoids registering an entire
8–16 GiB host vector on every call. Scalar storage is borrowed directly through
an explicit limb-array ABI guarded by compile-time size, alignment and offset
checks; FFT/evaluation calls no longer pack or unpack full columns.

Opening evaluation retains point and column order. The pinned sppark
multi-point evaluation kernel has a shared-memory race, so each point uses a
single-point kernel against the same resident coefficients. CPU lookup and
constraint sweeps still need downloaded evaluations; opening-polynomial folding
and the tiled division carry pass also remain on the CPU.

The stage-one example prefetches one witness while committing its predecessor.
Lookup and full-domain quotient processing also keep one partition's evaluations
ahead: host computation overlaps the preceding commitment and next evaluation
reconstruction. Both device operations occupy the same pipeline branch, so
nested Rayon work cannot recursively request another partition's device lease.
The shared CPU pool retains its default width. Lookup totals and all commitments
are returned in transcript order.

The lookahead cap applies to the extra row-major evaluation or witness payload;
retained coefficients, current compute scratch and commitment buffers are
additional. Oversized evaluation partitions drain the pipeline. The library
enables this through `KzgConfig::with_partition_pipeline(bytes)`; the stage-one
example and diagnostic expose the environment control below. Recursive
trace-sized quotient cosets retain their existing schedule.

On the regenerated 19-circuit Init fixture, a cached run with a 32 GiB
lookahead reduced stage-one proving from 155.627 s to 123.970 s (20.34%).
The complete root-to-packet boundary fell from 871.835 s to 839.988 s (3.65%);
peak stage-one host RAM rose from 190.9 GiB to 218.8 GiB. Both stages retain
byte-identical proofs, packets and profile identifiers. These are single
full-chain samples; a separate four-partition comparison using one binary
reduced lookup-plus-quotient time by 15.9%, and Nsight confirmed overlap.
See the [KZG performance document](../../docs/kzg-performance.md#bounded-partition-pipeline)
and [preserved reports](../kzg-cuda-validation/partition-pipeline-20261009/summary.json).

Controls:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MULTI_STARK_KZG_BACKEND` | `cuda` | `cpu` bypasses KZG GPU operations in the same binary. |
| `MULTI_STARK_KZG_CUDA_DEVICES` | all | Comma-separated CUDA ordinals, after `CUDA_VISIBLE_DEVICES` filtering; consistent with the Goldilocks backend. |
| `MULTI_STARK_KZG_MSM_CHUNK_POINTS` | `16777216` | Maximum points in each MSM chunk; runtime admission can reduce it further. |
| `MULTI_STARK_KZG_CUDA_RESIDENT_GIB` | half of device VRAM | Per-device coefficient budget, capped at half of total VRAM; `0` disables retention. Runtime admission also checks free memory and reserves scratch space. |
| `MULTI_STARK_KZG_CUDA_PROFILE` | unset | `1` emits CUDA-event transfer/compute intervals, host-copy wall time, byte counts, and NVTX ranges. `msm-compute` separates GPU intervals (`kernel_ms`) from the synchronous invocation (`call_ms`). Intervals across streams/devices overlap; division's compute interval includes its host carry pass. |
| `MULTI_STARK_KZG_PREFETCH_GIB` | `32` | Stage-one example and diagnostic: extra prefetched payload budget in GiB. `0` disables witness/evaluation overlap for a comparison using the same binary and cache identity. This is not a total host-memory limit. |
| `MULTI_STARK_KZG_DEV_SRS_CACHE` | unset | Optional trusted local directory for reusing known-trapdoor development parameters. |
| `MULTI_STARK_KZG_FIXED_CACHE` | unset | Trusted local fixed coefficients, commitments and circuit metadata; no witness or opening data. |
| `MULTI_STARK_INIT_EXPECTED_CLAIMS` | historical Init statement | Independently supplied root-claims file for verifying a regenerated input. |
| `MULTI_STARK_KZG_TRACE_CODEC` | `auto` | `pzstd` uses parallel compression and decompression; `zstd` uses the legacy stream. Automatic selection prefers installed `pzstd`. |
| `MULTI_STARK_CUDA_ARCHS` | toolkit-supported architectures | Native cubins to compile; set `120` on Blackwell. |

Fixed preprocessing is published atomically after successful proving and
verification. On reuse, staging only materializes witness traces, and proving
loads the fixed coefficients and commitments. The cache key includes the
exact executable digest, which conservatively binds dependencies, compiler,
lowering, SRS recipe and layout options. The first KZG stage also binds the
FRI verifier plan identity. The recursive stage binds the inner system's
metadata and statement: its current builder emits constant gates before
rebinding public claim slots, so different statements cannot share that
cache without a circuit-version change. Cache directories are trusted local
parameters, not an import format for parameters supplied by an adversary.
Files on the same volume are shared by hard links; they must remain immutable.

For a complete measured run, first regenerate or obtain the independently
verified aggregation root's `root-vk.bin`, `root-proof.bin` and
`root-claims.bin`. Build the FRI compressor with `--features parallel,cuda`
and `--example ix_root` (or include `cuda` in the first KZG build), then run:

```sh
python3 experiments/kzg-cuda-bench.py \
  --root-artifacts /path/to/verified-root \
  --fixed-cache /opt/dlami/nvme/kzg-fixed \
  --srs-cache /opt/dlami/nvme/kzg-dev-srs \
  --output /opt/dlami/nvme/kzg-full-run
```

The first run populates the caches; subsequent runs with the same binaries
and profile measure per-proof cost. Reports include each cache hit/miss and
the exact input hashes. Root aggregation is outside the timing boundary;
FRI circuit construction, FRI proving, both KZG staging/proving steps and
their in-process verification are inside it. Independent CPU verification
is an additional check timed separately. A regenerated root can have a
different trace profile despite certifying the same Init statement, so its
packet is verified rather than compared to the historical CPU packet.

For byte parity on a regenerated root, pass `--baseline /path/to/verified-run`.
The baseline directory must contain `report.json` and both stages' compact
proofs, packets and profile identifiers. The preserved
`experiments/kzg-cuda-validation/resume-20261009` directory is a compatible
baseline for its saved root. The driver requires identical root/compressed-FRI
input hashes, compares all six output artifacts, and still runs independent
CPU verification. Its report records the baseline report digest and expected
artifact hashes. Without `--baseline`, regenerated inputs receive verification
but no comparison to a previous regenerated proof.

With `MULTI_STARK_KZG_DEV_SRS_CACHE=/opt/dlami/nvme/kzg-dev-srs`, proving
and verification reuse development parameters after the first generation.
Cache keys bind the format, curve, seed, and exact degree range. The files use
canonical uncompressed points (about 1.5 GiB for 2^24 and 24 GiB for 2^28),
bounded I/O buffers, a checksum, and atomic publication. Loads check canonical
coordinates, curve membership, and seed-dependent anchors. They trust the local
producer for G1 subgroup membership and the complete power progression; this
cache is not a loader for ceremony parameters. Damaged files fail explicitly.
The known development trapdoor is unchanged. Leave the variable unset when
measuring fresh SRS generation; compare cached and fresh timings separately.

With bounded fixed-base generation, setup plus cache publication took 7.32
seconds at 2^24 and 119.34 seconds at 2^28. The complete cached run loaded
them in 1.50 and 23.33 seconds respectively. The operating-system page cache
may be warm on reload. Cache corruption, seed/size separation, concurrent
writers, and a GPU prove/resume/verify fixture also passed.

MSMs of 2^29 points are split into independent sums and reduced on the host.
Chunk admission accounts for resident points, scalars, sorting scratch, and
buckets while reserving one quarter of total VRAM. FFTs remain whole on one
device: a 2^29 scalar-field vector uses 16 GiB. The KZG instantiation sets
sppark's `MAX_LG_DOMAIN_SIZE=32`, overriding the fork's 2^28 default; actual
transforms are capped at log size 31 because its 32-bit stride cannot encode
2^32. Admission also depends on available device memory. No single 96 GiB device
has to hold all the points of a 2^29 MSM.

Fixed-selector polynomials can have many identical nonunit coefficients.
Those coefficients overload serial Pippenger buckets even when random-scalar
MSMs are fast. A bounded sample detects frequent values up to sign; scaling
the scalar chunk maps that value to +/-1 for sppark's parallel addition
kernel. Multiplying the partial result by the same value preserves the MSM.
Sampling affects performance only, not the mathematical result.

Lookup traces are constructed in parallel CPU tiles, with at most one scratch
tile per worker in each wave. An ordered prefix of tile totals restores the
row accumulators before commitment. This preserves the serial trace and proof
bytes while bounding temporary lookup payloads independently of trace height.

Fresh development SRS generation uses a bounded fixed-base multiplication
table and independent power ranges. Quotient evaluation reuses graph and
constraint buffers across rows, evaluates base-field lookups without heap
allocations, and computes selectors in parallel tiles. Trace materialization,
matrix transposes, and canonical field encoding also run in parallel. The
field codec uses bounded 32 MiB conversion buffers and preserves the existing
trace and checkpoint formats; it rejects noncanonical field elements.
When `pzstd` is available, staging emits independent Zstandard frames so trace
loading can decompress in parallel. Both tools can read the resulting files,
and legacy single-frame traces remain readable. The `zstd` fallback compresses
on all available cores but decodes a legacy stream on one core.

Use `CUDA_VISIBLE_DEVICES` to isolate GPUs from this process. The KZG device
list controls scheduling; sppark discovery and twiddle initialization still
touch every visible eligible GPU. A missing driver or empty visible-device
list produces an explicit CPU-backend hint. The CPU override retains CPU
parallelism when the `parallel` feature is enabled.

Affine points use explicit limb buffers; scalar borrowing is guarded by concrete
layout assertions. The KZG runtime and FFT symbols are isolated from the Goldilocks `cuda`
feature, so both features can be enabled together. The build makes a scoped
copy of sppark's runtime header to select the owning CUDA context before
synchronizing a device's streams. MSM chunks preload their points before
execution. Synthetic division uses independent tiles and a bounded host
carry pass over sppark field arithmetic.

Without `kzg-cuda` or `cuda`, builds do not require the CUDA toolkit or GPU.
Enable the existing `cuda` feature separately for Goldilocks FRI acceleration.

Primitive parity and explicit scale checks:

```sh
MULTI_STARK_CUDA_ARCHS=120 cargo test --release --locked \
  --features parallel,kzg-cuda --lib ark_adapter::cuda::tests
# Approximately 100 GiB host memory when these two tests run concurrently.
MULTI_STARK_CUDA_ARCHS=120 cargo test --release --locked \
  --features parallel,kzg-cuda --lib 2pow29 -- --ignored --nocapture
```

The ignored `cuda_primitive_benchmark` test measures CPU/GPU crossover sizes
and resident-column MSM reuse. Set `MULTI_STARK_KZG_BENCH_LOG_N=24` for its
real 2^24-point development SRS case. A single staged trace can be profiled
without rebuilding the whole proof:

```sh
MULTI_STARK_CUDA_ARCHS=120 cargo run --release --locked \
  --features parallel,kzg-cuda --example kzg_cuda_prepare_bench -- \
  /path/to/staged/fri-to-kzg 0
```

This reports loading, fixed/main/lookup commitment, and checkpoint-encoding costs;
append `quotient` to include quotient evaluation and commitment. Checkpoint bytes
go to `/dev/null`, so the encoding measurement excludes disk latency.

Use comma-separated indices to measure partition overlap. A single trace uses
trace-sized quotient cosets; multiple traces use the stage-one full quotient
domains. For a comparison with identical binaries and inputs:

```sh
MULTI_STARK_KZG_PREFETCH_GIB=0 target/release/examples/kzg_cuda_prepare_bench \
  /path/to/staged/fri-to-kzg 0,1,2,3 quotient
MULTI_STARK_KZG_PREFETCH_GIB=32 target/release/examples/kzg_cuda_prepare_bench \
  /path/to/staged/fri-to-kzg 0,1,2,3 quotient
```

After building the diagnostic, capture its CUDA, NVTX and host-runtime timeline
with Nsight Systems:

```sh
MULTI_STARK_KZG_CUDA_PROFILE=1 nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-memory-usage=true --output=/path/to/kzg-one-trace \
  target/release/examples/kzg_cuda_prepare_bench \
  /path/to/staged/fri-to-kzg 0 quotient
```

CPU sampling and context-switch tracing are disabled here; application CPU
parallelism is unchanged. Run the diagnostic separately from timing runs.
`msm-compute.kernel_ms` sums event intervals around scalar digit processing,
bucket accumulation/integration, and optional scalar normalization. These
intervals exclude explicit result downloads and host bucket reduction, but
can contain launch gaps and overlap across streams. `call_ms` measures the
whole synchronous invocation, including its waits and host reduction. Use the
Nsight kernel timeline to distinguish launch gaps from device execution; do
not subtract summed intervals from pipeline wall time. Other NVTX ranges mark
FFTs, opening operations, evaluation reconstruction, and CPU lookup/quotient
sweeps. The report's `host_operation_totals_overlap` also contains nested wall
timers, so it is an attribution aid rather than an additive wall-time ledger.
