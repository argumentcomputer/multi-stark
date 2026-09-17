# Resident LDE and wide NTT experiment: 2026-09-11

Historical measurements of the first-party NTT removed in September 2026.
Current GPU transforms use sppark; see [CUDA acceleration](../README.md#cuda-acceleration).

Base: `ff3237c` (the short-row BLAKE3 change). Same RTX PRO 6000 Blackwell
Server Edition, CUDA 13.3 / native `sm_120`, and eight Rayon threads as the
[BLAKE3 experiment](cuda-blake3-short-rows.md). Timings below measure the
incremental effect of this change; BLAKE3 is enabled in both binaries.

## Change retained

- Use the existing fused radix-8/radix-4 stages for matrices with at least
  eight columns regardless of height. Previously a separate `height >= 2^18`
  condition left short, wide matrices on individual radix-2 stages even when
  their total cell count was large. The existing narrow-column path remains.
- Zero only the padded tail of the resident LDE allocation. The input prefix
  is overwritten by the trace copy and does not need a preceding clear.
- Remove the resident LDE's final canonicalization pass. Its normalization
  and butterfly arithmetic already produce canonical Goldilocks values,
  including height-one transforms. Tests inspect the actual stored u64
  representation, rather than an accessor that canonicalizes values.

No new NTT implementation, matrix layout, workspace allocation, dependency,
or machine-specific threshold was introduced. Trace retention and output
storage order are unchanged. The NTT dispatch change applies to other callers
of the same transform helper; memory-pass removal is scoped to
`multi_stark_cuda_coset_lde_create`.

An earlier trial fused bit reversal, coset scaling, virtual zero padding,
and the first forward stage in place, using bit-reversal pairs to avoid
races. It passed correctness checks but did not improve complete commitments
consistently, so that kernel and its transform-continuation helper were
removed. The simpler change above is the measured result.

## Measurements

Resident LDE timings include upload, inverse transform, coset handling,
forward transform, and completion, excluding construction of the host input.
Medians of 14 warm samples per variant across two processes; first use of
each shape is reported separately and excluded. Baseline and candidate
processes were interleaved on an idle GPU.

| Input height | Columns | Blowup | BLAKE3-only | + LDE change | Reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2^20 | 1 | 2 | 0.992 ms | 0.965 ms | 2.7% |
| 2^20 | 2 | 2 | 1.805 ms | 1.833 ms | -1.6% |
| 2^20 | 8 | 2 | 6.563 ms | 6.435 ms | 1.9% |
| 2^18 | 40 | 2 | 8.582 ms | 8.451 ms | 1.5% |
| 2^18 | 128 | 2 | 30.389 ms | 30.093 ms | 1.0% |
| 2^18 | 129 | 2 | 30.533 ms | 30.099 ms | 1.4% |
| 2^16 | 925 | 2 | 79.048 ms | 57.660 ms | 27.1% |
| 2^18 | 40 | 4 | 10.664 ms | 10.155 ms | 4.8% |

Complete PCS commitments include LDEs, gathering, hashing, Merkle construction,
and root return. Medians of 15 warm samples per variant across three processes;
all commitment bytes matched for every shape/run.

| Shape | BLAKE3-only | + LDE change | Reduction |
| --- | ---: | ---: | ---: |
| 2^20 × 8 | 9.011 ms | 8.938 ms | 0.8% |
| 2^18 × 40 | 10.666 ms | 10.478 ms | 1.8% |
| 2^18 × 128 | 36.102 ms | 35.332 ms | 2.1% |
| 2^18 × 129 | 43.792 ms | 43.195 ms | 1.4% |
| 2^16 × 925 | 91.071 ms | 69.281 ms | 23.9% |
| Mixed heights/widths | 68.184 ms | 51.253 ms | 24.8% |

The mixed shape is the same five-matrix fixture as the BLAKE3 experiment.
The substantial gains occur in shapes affected by the removed NTT height
cutoff. Small percentage differences on unaffected/narrow shapes should not
be treated as established wins; one narrow LDE case was 1.6% slower. These
are synthetic component timings, not an Init/Mathlib or Stage 2 wall-time
measurement, and are not a general 2× NTT claim.

## Validation and reproduction

- All 71 CUDA-enabled library tests passed (6.46 seconds).
- CPU comparisons exercise blowups 1/2/4/8, three cosets, height-one inputs,
  wide/narrow dispatch, fused-stage residues modulo three, and noncanonical
  input representations around the field modulus.
- Raw output field words remain below the modulus. Mixed-height commitments,
  resident/hybrid openings, hashing, and batch-proof tests pass.
- CUDA memcheck and initcheck report zero errors for the padding/coset/raw-
  representation suite, including the case with no zero-padded tail.
- The complete 17,213-byte compatibility proof verifies and matches both
  CPU and BLAKE3-only output exactly. SHA-256:
  `25564a01d1d352b1ec2de56b019b641d24acc81083133e79274a86829b2a5dd5`.
- Clippy completed without warnings for the library and both new examples.

```sh
MULTI_STARK_CUDA_ARCHS=120 RAYON_NUM_THREADS=8 cargo test --release --locked --features parallel,cuda --lib -- --test-threads=1
RAYON_NUM_THREADS=8 cargo run --release --locked --features parallel,cuda --example cuda_resident_lde_bench
RAYON_NUM_THREADS=8 cargo run --release --locked --features parallel,cuda --example cuda_commit_bench
```

Build the identical benchmark source against `ff3237c` and this commit,
preserve both binaries, and alternate runs. `MULTI_STARK_CUDA_BENCH_ITERATIONS`
controls the warm sample count. The resident benchmark uses the production
resident-LDE interface; the older `cuda_dft_bench` exercises a different,
transfer-inclusive host-return interface and does not isolate these changes.
