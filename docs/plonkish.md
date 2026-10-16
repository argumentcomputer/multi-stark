# Plonkish builder

`plonkish::CircuitBuilder<F>` builds a fixed native-field circuit with arithmetic,
fixed-table lookups, and public bindings. Construct once, then assign inputs for
each witness. Handles belong to one builder; reusing a value enforces equality.

```rust
let mut builder = CircuitBuilder::<Val>::new();
let x = builder.input("x");
let square = builder.mul(x, x);
builder.expose_public(square);
let compiled = builder.finish().lower_to_multi_stark(namespace)?;
let mut witness = compiled.witness();
witness.set(x, value)?;
let assignment = witness.generate()?;
let traces = compiled.traces(&assignment)?;
let claims = compiled.claims(&[expected_square])?;
```

Use `compiled.circuit_inputs()` for setup. Pass all returned claims to proving
and verification, even with no public values. Expected public values must come
from the verifier. Use distinct lookup namespaces when combining lowerings.
See [plonkish_proof.rs](../examples/plonkish_proof.rs) for a complete example:

```sh
cargo run --release --example plonkish_proof
```

- Gadgets take a mutable builder and return handles; callers choose public exposure.
- `hint` and `hint_many` only compute witness values; constrain their outputs.
- `stats()` reports circuit counts; `multi_stark_layout()` reports padded traces.
- `lower_to_multi_stark_with_max_height` partitions computation traces; the cap
  must be a power of two, at least two. Tables and custom traces must fit it.
  Partitioning limits individual domains, not total prover memory.
- `plonkish::gadgets` provides byte operations and BLAKE3. Call
  `enable_compact_blake3()` before hashing to use custom gates.
- The lowering requires FRI blowup at least two, or four with compact BLAKE3.
  Include blowup when checking trace domain limits. Layout byte counts exclude
  LDEs and prover scratch. The backend is not zero-knowledge.

External circuit translators can inspect `gates`, `tables`, `lookups`,
`blake3_calls`, `public_values` and `value_sources`. Preserve every relation;
value sources and hints are witness metadata, not constraints. `Circuit::id`
and `Assignment::values(id)` check assignment ownership without copying values.
`constrain_blake3` binds existing byte wires to a digest. `counting` and `reserve`
support sizing before allocation. Counting builders cannot produce circuits.

`lower_to_multi_stark_sharded` and `trace_shards` allow generating one trace at a
time; copy constraints span partitions. `merge_table_traces` combines lookup
tables. Backend implementations use the [backend interfaces](pcs-abstraction.md).
