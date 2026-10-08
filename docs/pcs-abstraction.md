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

For bounded trace memory, use `lower_to_multi_stark_sharded` and
`trace_shards` to stage matrices before proving. Copy links remain global.
`KzgConfig::with_streaming_lookups` constructs accumulators in row chunks;
`with_streaming_quotient` evaluates one trace-sized coset at a time.
`KzgProverData` checkpoints and `System::prove_committed` reuse committed traces.

See [the Init pipeline](../experiments/kzg-wrap/README.md) for commands and measured
results. Wrapping a proof does not raise its underlying security level.
