# Commitment backends

`ProofConfig` selects crate-owned field, transcript, domain and PCS traits.
`p3_adapter` implements the Goldilocks FRI configuration; `ark_adapter` provides
BLS12-381 KZG behind the `kzg` feature. `StarkGenericConfig` remains an alias.

KZG uses the scalar field as both base and challenge field. Commitments contain
one G1 point per column, plus a shifted commitment for polynomials shorter than
the SRS. The verifier checks these degree bounds and batches openings by point.
Quotients are split into trace-degree slices before commitment.
KZG omits unused next-row openings and inactive preprocessing openings; required
openings and degree checks remain mandatory. Transcript: `multi-stark/kzg/v3`.

Supply a trusted SRS and validate imported parameters with `Srs::validate`.
The SRS includes degree-check G2 powers and must cover the full available degree
range; truncating a larger public SRS does not establish a smaller degree bound.
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
The SRS length must match the parameters used for proving.
`merge_table_traces(max_height)` combines fixed tables using a fixed table-ID
column. `ark_adapter::compact::FixedProofCodec` omits redundant shape metadata
for balanced, all-active proofs; the verifier supplies the trusted system and
heights and must still verify the decoded proof normally.

```sh
cargo test --release --features kzg,parallel
cargo run --release --features kzg,parallel --example fri_kzg -- --compare
```

The example proves the same FRI statement with baseline and tuned lookup groups.
Add `--compact` to compare compact hashing. It uses development security
parameters and an insecure test SRS; the proof system is not zero-knowledge.
Add `--merge-tables` to consolidate tables and report the compact encoding.
Use `--estimate` to count translated gates without allocating the scalar circuit.

For bounded main-trace memory, use `lower_to_multi_stark_sharded`, then
`circuit_input` / `kzg_circuit_input` and `trace_shards`. Integer copy links remain
global. `ark_adapter::sharded::ShardedKzg` regenerates preprocessing and witnesses
per shard, checking them against setup and round-one commitments. Verification
requires the expected claims and a schedule covering each circuit exactly once.
The frontend, assignment, copy links and custom fixed traces remain resident.
`TraceShards::trace(index)` generates each table or hash circuit separately.
Staging these matrices permits dropping the frontend before proving.
`commit_shard` and `prove_shard` support checkpointed runs; verify the complete
batch against its expected claims and schedule.

The example accepts `--partition-log=18 --batch --stream-preprocessing`.
Sharding reduces peak trace memory but adds proof bytes. The merged-table generic
fixture is 1,989 B in compact transport (ordinary: 2,405 B); these are small-fixture
measurements, excluding the statement and verification key.
An end-to-end security target also applies to the inner proof: field size alone
does not account for degree and lookup soundness losses. Wrapping a weaker
proof cannot raise its security level.
