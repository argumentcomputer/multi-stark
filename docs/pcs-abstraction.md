# Commitment backends

`ProofConfig` selects crate-owned field, transcript, domain and PCS traits.
`p3_adapter` implements the Goldilocks FRI configuration; `ark_adapter` provides
BLS12-381 KZG behind the `kzg` feature. `StarkGenericConfig` remains an alias.

KZG uses the scalar field as both base and challenge field. Commitments contain
one G1 point per column, and openings are batched by point. Quotients are split
into trace-sized slices before commitment. Unused next-row openings and inactive
preprocessing openings are omitted; every required column remains explicitly opened.

There are two degree policies. Legacy `multi-stark/kzg/v3` parameters require
shifted commitments for polynomials shorter than their complete SRS range.
Authenticated public parameters use `multi-stark/kzg/v4`, binding the ceremony
identity, full public degree allowance and admitted trace cap. This policy has
no per-trace shifted commitments: its whole-protocol argument uses the full
public degree in the generic bilinear group and random oracle models. See the
[soundness argument](kzg-performance.md#full-public-degree-soundness-argument).
Loading a shorter prefix never reduces an adversary's public degree allowance.

Supply authenticated parameters with checked curve/subgroup membership and
validate their progression with `Srs::validate`. Consistency alone does not
establish provenance or an unknown trapdoor. The Filecoin importer authenticates
stock `challenge_19` and writes a cache whose manifest digest must be pinned in
trusted configuration. `KzgConfig::with_max_trace_len` separates the loaded
prefix from admitted trace heights; a public-policy verifier needs two G1 anchors.
`unsafe_dev_setup` is only for tests: its seed reveals the setup secret.

To prove a Goldilocks circuit with KZG, translate it with
`plonkish::foreign::GoldilocksCircuit`. The translation enforces canonical values
and bounded modular quotients over the scalar field. Existing bounded integer
relations and compact hash traces can be retained directly.
`GoldilocksCircuit::estimate` counts the same translation without storing its IR.
Boolean, lookup and equality bounds avoid redundant integer range checks.
`new_preallocated` counts first to avoid geometric IR allocation growth.

After lowering, `kzg_circuit_inputs(srs_len, quotient_budget)` selects lookup
groups to minimize commitment/evaluation bytes within the quotient budget.
The SRS length must match the legacy parameters used for proving. For public
parameters, use `kzg_circuit_input_with_degree_policy(..., false)` so lookup
tuning accounts for the absence of shifted commitments at every trace height.
`merge_table_traces(max_height)` combines fixed tables using a fixed table-ID
column. `ark_adapter::compact::FixedProofCodec` omits redundant shape metadata
for balanced, all-active proofs; the verifier supplies the trusted system and
heights and must still verify the decoded proof normally.

For bounded trace memory, use `lower_to_multi_stark_sharded` and
`trace_shards` to stage matrices before proving. Copy links remain global.
`KzgConfig::with_streaming_lookups` constructs accumulators in row chunks;
`with_streaming_quotient` evaluates one trace-sized coset at a time.
`KzgProverData` checkpoints and `System::prove_committed` reuse committed traces.

## GPU acceleration

The `kzg-cuda` feature runs BLS12-381 MSMs, scalar-field FFTs, polynomial
evaluations and opening-witness division on CUDA through the pinned
`argumentcomputer/sppark` fork, preserving the CPU transcript, checkpoint
format and proof bytes. It coexists with the Goldilocks `cuda` feature.
Interpolated columns stay resident on their device within a per-device
budget and feed later FFTs, MSMs and evaluations without re-upload. Lookup
and quotient evaluation use resident CUDA consumers when their working sets
fit; larger shapes retain bounded host fallback. See the
[implemented GPU path](kzg-performance.md#implemented-gpu-path) for admission,
transfer behavior and targeted validation.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MULTI_STARK_KZG_BACKEND` | `cuda` | `cpu` bypasses the GPU in the same binary |
| `MULTI_STARK_KZG_CUDA_DEVICES` | all | Comma-separated CUDA ordinals to use |
| `MULTI_STARK_KZG_MSM_CHUNK_POINTS` | `16777216` | Maximum SRS points per MSM chunk |
| `MULTI_STARK_KZG_CUDA_RESIDENT_GIB` | half of VRAM | Per-device budget for resident coefficients |
| `MULTI_STARK_KZG_CUDA_SRS_CACHE` | `0` | `1` enables bounded, evictable resident SRS ranges and reusable MSM workspaces |
| `MULTI_STARK_KZG_CUDA_LOOKUP` | `cuda` | `cpu` selects the host lookup construction reference |
| `MULTI_STARK_KZG_CUDA_QUOTIENT` | `cuda` | `cpu` selects the host quotient construction reference |
| `MULTI_STARK_KZG_DEV_SRS_CACHE` | unset | Trusted local cache of development SRS files |
| `MULTI_STARK_KZG_SETUP` | `filecoin` | Pipeline setup selection; `development` explicitly enables known-trapdoor experiments |
| `MULTI_STARK_KZG_FILECOIN_CACHE` | required for Filecoin | Authenticated normalized ceremony cache |
| `MULTI_STARK_KZG_FILECOIN_DIGEST` | required for Filecoin | Externally pinned manifest digest emitted by a successful import |
| `MULTI_STARK_KZG_FIXED_CACHE` | unset | Trusted local cache of fixed preprocessing, keyed by executable and plan identity |

The fixed cache binds the executable, setup/profile and recursive public statement;
the development SRS cache binds its seed and degree range independently. See
[the Init pipeline](../experiments/kzg-wrap/README.md) for commands and measured
results, [the benchmark record](cuda-benchmarks.md) for dated figures, and
[upstream GPU orchestration](upstream-gpu-orchestration.md) for how SP1 and
ZisK structure the same work. Wrapping a proof does not raise its underlying
security level.
