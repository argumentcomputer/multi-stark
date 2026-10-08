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

For KZG compression, see [commitment backends](pcs-abstraction.md) and
[the Init pipeline](../experiments/kzg-wrap/README.md). Goldilocks constraints
require explicit field translation.
