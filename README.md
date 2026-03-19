# multi-STARK

A multi-circuit STARK proving system built on [Plonky3](https://github.com/Plonky3/Plonky3).

Prove and verify multiple AIR circuits in a single proof, with cross-circuit
lookup arguments for shared state.

## Features

- **Multi-circuit proofs** — bundle multiple AIR circuits into one proof with
  independent trace heights
- **Lookup arguments** — push/pull interactions of arbitrary length between
  circuits, enforced via accumulator-based multiset checks
- **Preprocessed tables** — commit to fixed tables once, reuse across proofs
- **Serialization** — `Proof::to_bytes` / `Proof::from_bytes` via bincode
- **Parallel proving** — opt-in via the `parallel` feature flag

## Cryptographic setup

| Component | Choice |
|-----------|--------|
| Field | Goldilocks (p = 2^64 - 2^32 + 1) |
| Extension | Degree-2 binomial extension (~2^128 elements) |
| Hash | Keccak-256 |
| PCS | FRI over Merkle trees |

Security level is configurable via `FriParameters`. With `log_blowup = 1` and
`num_queries = 100`, FRI provides ~2^(-100) soundness error. See the
[verifier module docs](src/verifier.rs) for the full soundness argument.

## Examples

**Minimal prove-and-verify** (no lookups):
```sh
cargo run --example simple_proof --release
```

**Preprocessed trace with lookups** (byte range-check table):
```sh
cargo run --example preprocessed_proof --release
```

**Multi-circuit with lookup arguments**:
```sh
cargo run --example lookup_proof --release
```

## Benchmarks

```sh
cargo bench --bench multi_stark --features parallel
```

Benchmarks cover `prove` and `verify` at 2^12, 2^13, and 2^14 trace rows
using a U32 addition circuit with lookups and a preprocessed byte table.
Use `--features parallel` for representative numbers. Native SIMD instructions
are enabled by default via `.cargo/config.toml`.

## License

MIT or Apache-2.0
