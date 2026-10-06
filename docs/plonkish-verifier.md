# Plonkish FRI verifier

`plonkish::verifier::VerifierPlan` builds a verifier circuit for a trusted key
and fixed proof profile. Supported: Goldilocks with quadratic challenges,
BLAKE3, mixed two-adic heights, root-only Merkle caps, binary FRI with a constant
final polynomial, and ordinary or single-shard batch proofs. Preprocessing must
be present and its circuits active. Field sampling rejects retry-budget exhaustion.

```rust
let key = VerifierKey::from_system(&system);
let plan = VerifierPlan::validate(&key, profile, VerifierLimits::default())?;
let (circuit, inputs) = plan.build(schema, ImplementationOptions::default())?;
let prepared = plan.expand_witness(ProofEnvelope::SingleBatch(&batch))?;
let mut witness = circuit.witness();
inputs.assign_statement(&mut witness, &expected_statement)?;
prepared.assign_proof(&mut witness, &inputs)?;
let assignment = witness.generate()?;
```

`ProofProfile` fixes the envelope, activation bitmap in key order, log heights
in active-circuit order, claim/message lengths, and field-sampling retry budget.
Validation derives opening dimensions and checks structural limits before
circuit construction; these limits do not bound total prover memory.

`Statement` holds claims and batch messages (arguments and multiplicities):

- `plan.build` accepts `StatementSlot::{Constant, Public}`. Public order is claims,
  then each message's arguments and multiplicity, omitting constants.
- `plan.constrain` accepts `StatementBinding::{Constant, Wire}` for composition.
  Callers must expose or constrain the wires to bind the intended statement.
- Supply independently expected statement values. Proof expansion is untrusted;
  `assign_proof` assigns only proof inputs and cannot overwrite the statement.

Keys and profiles stay fixed across witnesses. `VerifierKey::to_bytes/from_bytes`
serializes metadata and commitments without prover matrices; decoded keys still
require authentication. `plan.identity` binds the key, profile, schema and
constants; `build_identity` adds implementation and supplied build versions.
Compact BLAKE3 is the default; disable it through `ImplementationOptions`.
See the [builder guide](plonkish.md) for lowering and domain requirements.

[Plan tests](../tests/plonkish_plan.rs) contain ordinary and dynamic-batch usage.
The self-contained recursive example uses development security parameters:

```sh
cargo run --release --example parity_recursive -- --check-only
```

Omit `--check-only` to also prove the verifier circuit; this requires more memory.

For a KZG outer proof, see [commitment backends](pcs-abstraction.md) and
[the FRI-to-KZG example](../examples/fri_kzg.rs). Goldilocks constraints require
explicit field translation; changing the commitment scheme alone is insufficient.

With `groth16`, `QueryShardPlan` splits an ordinary or single-batch verifier into one global
shard and fixed query groups. All shards constrain the same BLAKE3 context;
only the complete ordered bundle under trusted keys verifies the statement.
`verify_encoded_query_bundle` checks the compact binary packet against an
independently expected statement. Development setup example:

```sh
cargo run --release --features groth16,parallel --example fri_groth16 -- --shards
```

Add `--ordinary` to exercise ordinary FRI proofs. `init_fri_wrap` checks the
saved recursive Init proof and counts its R1CS. Append `10 <shard> --prove`
to prove one shard with a development setup; `--bundle` verifies all 11 saved
shards. `experiments/run-init-fri-groth16.py` runs them sequentially with a memory
cap and checkpoints. Measurements are in `experiments/init-fri-wrap.json`.

`r1cs::streaming` consumes rows directly into the standard Groth16 QAP, without
retaining constraint matrices. `--stream-check` checks the full QAP witness;
`--stream-prove` uses this path for setup and proving. Set
`INIT_GROTH16_STREAMING=1` for the runner and give it a fresh output directory.

The saved 2,324-byte Init bundle and its development verifying keys are in
`experiments/init-fri-groth16-artifacts`. Verify it and reject altered statements
and packets with:

```sh
cargo run --release --features groth16,parallel --example verify_init_bundle
```

These deterministic setup keys are insecure. Completed CPU measurements are in
`experiments/init-fri-groth16-streaming.json`.

`init_fri_kzg <recovered-artifacts> <output-dir>` (features `kzg,parallel`)
counts KZG layouts for the same saved proof. `experiments/measure-init-fri-kzg.py`
records sizes and dense-array memory estimates; it does not run full-size setup
or proving. Results: `experiments/init-fri-kzg.json`.
