# Conventional PLONK terminal backends

As of 2026-10-09, Jellyfish UltraPlonk is a credible BLS12-381 candidate for
the [BLAKE3 terminal statement](blake3-terminal-statement.md). Axiom's
Halo2/KZG fork is a candidate for a more flexible custom-gate layout if
BN254's security margin is acceptable. Prefer these conventional libraries
over adopting Proofman, PIL, or PILFFLONK. An in-house backend remains a
fallback after concrete library feasibility checks.

The terminal relation stays fixed: verify the existing Goldilocks/BLAKE3 FRI
proof against a trusted inner key and the independently expected 18-word
root claim. One terminal PLONK/KZG proof replaces the existing first KZG
proof and its additional recursive KZG wrapper. The preceding BLAKE3/FRI
compression remains in the initial comparison. Poseidon and Poseidon2 are
excluded from every selected proof layer and transcript.

## Library shortlist

| Candidate | Relevant capabilities | Main unresolved cost |
| --- | --- | --- |
| Jellyfish UltraPlonk on BLS12-381 | Rust, arkworks, KZG, Merlin transcript, lookups, canonical serialization | Its fixed selector family may substantially expand the BLAKE3 layout; BLS12-381 GPU integration needs work |
| Axiom Halo2 with BN254 KZG | Flexible polynomial gates, rotations, equality constraints, lookups, BLAKE2b transcript | Full circuit and quotient domains, GPU integration, and the lower curve security margin |
| gnark PLONK | Conventional terminal reference, including the SP1 architectural precedent | Keep as a reference rather than the leading candidate for this compact hash layout; full-size cost is unmeasured |

### Jellyfish

The reviewed source revision is
[`b1581c26f7195962d17fc5ff3768ac55db610ca1`](https://github.com/EspressoSystems/jellyfish/tree/b1581c26f7195962d17fc5ff3768ac55db610ca1).
Its SNARK tests exercise BLS12-381 UltraPlonk with `StandardTranscript`,
including serialization. The published
[SNARK source](https://jellyfish.docs.espressosys.com/src/jf_plonk/proof_system/snark.rs.html)
and [Merlin wrapper](https://jellyfish.docs.espressosys.com/src/jf_plonk/transcript/standard.rs.html)
substantiate a non-Poseidon terminal path.

An open gate trait does not mean arbitrary polynomial constraints. The
[gate interface](https://jellyfish.docs.espressosys.com/src/jf_relation/gates/mod.rs.html)
selects coefficients from a fixed family of linear, multiplication, hash,
and elliptic-curve terms. BLAKE3 needs a mapping into those terms and the
lookup machinery. A trait implementation alone cannot add a wide
quarter-round constraint to the proof protocol.

The pinned source assessment reports six UltraPlonk wire types, a lookup
slot per row, key/value tables with domain separation, and configurable
range-check chunks. These are useful building blocks for byte XOR and
modular additions. They do not yet establish a competitive BLAKE3 gadget.
The pinned tree uses arkworks 0.5, matching this repository's KZG stack;
sharing field types still requires a backend adapter and a GPU execution
integration.

The approximately 1.5 KB UltraPlonk proof-size estimate from its point and
scalar counts is a planning figure, not a serialized terminal fixture.
Proof size does not predict proving cost. The backend's zero-knowledge
masking must remain included in degree and setup calculations. No
BLS12-381 GPU path has been established for the pinned Jellyfish backend.

### Axiom Halo2

The published `halo2-axiom` 0.5.1
[serialization example](https://docs.rs/crate/halo2-axiom/0.5.1/source/examples/serialization.rs)
uses BN254, `KZGCommitmentScheme`, and `Blake2bRead`/`Blake2bWrite`.
Its [layout example](https://docs.rs/crate/halo2-axiom/0.5.1/source/examples/circuit-layout.rs)
demonstrates polynomial gates, previous/current/next-row queries, equality
constraints, and lookup tables. These interfaces make a compact BLAKE3
layout plausible without adopting an AIR/PIL backend. They do not establish
that the existing layout maps one-to-one or retains its row count.

Use a specific fork and release in comparisons. The
[PSE repository](https://github.com/privacy-ethereum/halo2) was archived on
2026-08-17. Its README describes maintenance mode starting in January 2025
and directs substantial development toward
[Axiom's fork](https://github.com/axiom-crypto/halo2). Deployment or audit
claims for one fork do not establish the properties of every fork.

The demonstrated path here is BN254. BLS12-381 support in a separately
ported Halo2 stack would require its own source, codec, transcript, and
prover review. BLS curve arithmetic gadgets inside a circuit are not
evidence that the terminal PCS uses BLS12-381.

### gnark

Read at `3ac4de3` (2026-10-08, tagged v0.16.4, Go 1.26.8), with
gnark-crypto alongside; both are cloned under `~/repos/clones/zk/`. gnark's
PLONK backend supports BLS12-377, BLS12-381, BN254 and BW6-761 with a
SHA-256 Fiat-Shamir transcript by default, a fixed three-wire gate plus
bsb22 commitments, and logarithmic-derivative lookups. Range checks and
byte operations go through `std/rangecheck` and `std/math/uints` tables;
the latter carries its own soundness disclaimer. There is no BLAKE3 or u32
rotate gadget. The only measured hash cost in the tree is the Keccak-f
permutation at 158,486 PLONK constraints, so a BLAKE3 compression on this
gate family would land well above the Jellyfish estimate below, which is
why it is not a full-size candidate.

Integration facts: Go, so Rust reaches it through a cgo `c-archive` as SP1
does; the PLONK `Accelerator` interface has only a WebGPU implementation and
ICICLE covers Groth16 only; the verifier subgroup-checks proof points; there
is no powers-of-tau importer, and SP1 adds its own Aztec Ignition verifier.
The repository's advisory history includes a critical 2023 PLONK verifier
fix, a critical 2026-08 `std` soundness fix and a high 2026-10 fake-GLV
fix, with audit reports in `audits/`. Those are reasons to pin and review a
release, not reasons to exclude it as a reference verifier or Solidity
exporter.

## Setup capacity

Filecoin's power-27 ceremony label describes its Groth16 constraint target,
not its total G1 tau-power count. The
[ceremony description](https://github.com/arielgabizon/perpetualpowersoftau)
specifies `2^28 - 1` G1 tau powers. The
[parameter implementation](https://github.com/filecoin-project/powersoftau/blob/master/src/parameters.rs)
defines `TAU_POWERS_G1_LENGTH = 2 * TAU_POWERS_LENGTH - 1`.

Jellyfish's
[arithmetization](https://jellyfish.docs.espressosys.com/src/jf_relation/constraint_system.rs.html)
requests a maximum degree of `n + 2`, and its
[preprocessor](https://jellyfish.docs.espressosys.com/src/jf_plonk/proof_system/snark.rs.html)
checks that against the SRS maximum degree. Consequently:

| Quantity | Value |
| --- | ---: |
| Documented Filecoin G1 tau-power count | `2^28 - 1 = 268,435,455` |
| Highest represented exponent, counting from zero | `2^28 - 2 = 268,435,454` |
| Required maximum degree for Jellyfish with `n = 2^27` | `2^27 + 2 = 134,217,730` |
| Largest power-of-two Jellyfish domain supported by that degree capacity | `2^27` |

The claim that Filecoin necessarily limits Jellyfish to `2^26` rows is
therefore incorrect. This arithmetic does not authenticate an artifact,
establish a working importer, or demonstrate that the full verifier fits.
Production selection must identify the exact usable powers, provenance,
point validation and conversion rules. No claim that this is the largest
possible or currently available ceremony is needed for the comparison.

The existing multi-stark KZG adapter's shifted degree checks require
additional G2 powers. A conventional PLONK backend follows its own degree
policy; it does not inherit that requirement merely because it uses KZG.
Universal setup avoids a fresh circuit-specific ceremony within capacity,
but each compiled circuit still has its own proving and verification keys.

## FFT domain limits

BN254's scalar field has two-adicity 28. The relevant Halo2 limit also
includes the extended quotient domain. Its
[domain constructor](https://raw.githubusercontent.com/axiom-crypto/halo2/main/halo2_proofs/src/poly/domain.rs)
grows the domain until it covers `(d - 1) * 2^k`, then checks that the
extended exponent is within the field's two-adicity. Thus:

```text
k + ceil(log2(d - 1)) <= 28
```

Here `d` is the configured constraint-system degree, including its arguments,
and `2^k` is the base domain. Degree 5 permits a base domain at most `2^26`;
degree 9 permits at most `2^25`. Usable witness rows are smaller after the
backend's reserved and blinding rows. A setup advertised at power 28 does
not imply a `2^28`-row circuit fits this implementation.

Check the selected Jellyfish implementation's quotient-domain requirements
as well as its SRS capacity. Setup degree, circuit domain, and extended FFT
domain are distinct resource limits.

## BLAKE3 sizing evidence

The historical compressed-FRI workload checks 60,297 BLAKE3 compressions.
The layout figures below are planning estimates, not completed conventional
PLONK layouts or comparative prover benchmarks:

| Layout estimate | Rows per compression | Hash rows for 60,297 compressions |
| --- | ---: | ---: |
| Existing compact layout, using wide rows | Approximately 240 | Approximately 14.5 million |
| Proposed Jellyfish byte-table mapping | Approximately 5,000 to 6,000 | Approximately 301 to 362 million |

At the midpoint, `60,297 * 5,500 = 331,633,500` hash rows would require a
`2^29` base domain before adding other verifier work. That would exceed the
Filecoin/Jellyfish capacity above and BN254's radix-2 limit. This establishes
the consequence of that estimate, not a lower bound on every possible
Jellyfish mapping.

Wide rows and narrow rows are not equal units of proving work. A comparison
must include committed field elements, fixed and auxiliary columns, lookup
tables and arguments, copy constraints, quotient degree and padding. The
existing 126.8-million-gate translated IR is also not automatically the row
count of a conventional PLONK layout. Conversely, counting only BLAKE3 rows
does not establish that the full terminal fits.

Study inner FRI tuning alongside the layout experiment while retaining the
unchanged input as a baseline. An order-of-magnitude reduction in compression
count is not a proven requirement. Any change to blowup, query count,
folding, grinding, caps, or final polynomial needs its own supported profile
and soundness analysis; it cannot silently alter the archived verifier.

## GPU integration and measurement boundaries

The current checkout contains BLS12-381 GPU MSM, FFT, polynomial evaluation,
and division in the [KZG adapter](../src/ark_adapter/cuda.rs) and
[CUDA implementation](../cuda/kzg.cu). A conventional backend could reuse
those primitives with additional integration. Switching the terminal curve
does not require switching the existing baseline's curve, but it changes
the field/curve buffers, parameters and validation needed by the terminal.
How SP1's open GPU prover and pil2-proofman's PILFFLONK keep a proof's key
and polynomials device-resident, and which of those patterns the KZG adapter
already has, is recorded in
[upstream GPU orchestration](upstream-gpu-orchestration.md).

The [GPU data-movement results](../experiments/kzg-cuda-gpu-feeding-results.json)
record a 869.47-second cached pipeline on the regenerated 19-circuit profile.
It includes FRI compression, both KZG staging/proving phases and in-process
verification, with fresh witnesses and proofs. Root generation, cache
population and additional independent CPU checks are outside that boundary.
The same record reports 1,200.96 seconds with fresh fixed preprocessing and
a populated development SRS cache.

The earlier [1,740.20-second run](../experiments/kzg-cuda-validation/sub30-v1-pipeline-report.json)
starts from the historical 1,530,149-byte compressed FRI proof and includes
fresh development SRS generation. Its 15m33s recursive-stage cost is a
historical observation, not the current cached cost or a promised saving
from replacing that stage. Use matched input, cache, security and privacy
settings for a terminal comparison.

## Feasibility decision

Keep BLS12-381 as the initial baseline and evaluate Jellyfish before assuming
an in-house prover is necessary. Evaluate Axiom Halo2 on BN254 if the
consumer accepts that curve's security margin. A lower pairing-security
margin is a separate decision from excluding Poseidon; removing Poseidon
does not resolve the curve-security choice.

The first feasibility record should contain:

1. A complete BLAKE3 compression and representative Goldilocks modular
   arithmetic, with correct ranges, carries, XORs, rotations and copy links.
2. Jellyfish's initial byte XOR table and 16-bit range-chunk configuration,
   or the precise Halo2 gates and lookup arguments used.
3. Logical and padded rows, all column counts, table sizes, quotient degree,
   extended-domain size, SRS powers, and memory estimates for the full input.
4. A valid terminal proof for the small circuit, its actual encoded size,
   and negative cases that reject altered advice and public values.

Select the full-size backend from that evidence. A small proof size alone,
a ceremony's advertised constraint count, or an uncompiled hash-row estimate
does not determine the winning terminal architecture.
