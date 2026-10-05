# Commitment backends

`ProofConfig` selects crate-owned field, transcript, domain and PCS traits.
`p3_adapter` implements the Goldilocks FRI configuration; `ark_adapter` provides
BLS12-381 KZG behind the `kzg` feature. `StarkGenericConfig` remains an alias.

KZG uses the scalar field as both base and challenge field. Commitments contain
one G1 point per column, plus a shifted commitment for polynomials shorter than
the SRS. The verifier checks these degree bounds and batches openings by point.
Quotients are split into trace-degree slices before commitment.

Supply a trusted SRS and validate imported parameters with `Srs::validate`.
The SRS includes degree-check G2 powers and must cover the full available degree
range; truncating a larger public SRS does not establish a smaller degree bound.
`unsafe_dev_setup` is only for tests: its seed reveals the setup secret.

To prove a Goldilocks circuit with KZG, translate it with
`plonkish::foreign::GoldilocksCircuit`. The translation enforces canonical values
and bounded modular quotients over the scalar field. Existing bounded integer
relations and compact hash traces can be retained directly.

After lowering, `kzg_circuit_inputs(srs_len, quotient_budget)` selects lookup
groups to minimize commitment/evaluation bytes within the quotient budget.
The SRS length must match the parameters used for proving.

```sh
cargo test --release --features kzg,parallel
cargo run --release --features kzg,parallel --example fri_kzg -- --compare
```

The example proves the same FRI statement with baseline and tuned lookup groups.
Add `--compact` to compare compact hashing. It uses development security
parameters and an insecure test SRS; the proof system is not zero-knowledge.

For the example's two-row, one-query inner proof, serialized KZG proof sizes
(excluding the verification key and SRS) were:

| Hash implementation | Baseline groups | Tuned groups |
| --- | ---: | ---: |
| Generic gates | 6,147 B | 5,635 B |
| Compact traces | 57,661 B | 47,741 B |

Generic hashing used a 2^22-row main trace; compact hashing used 2^20 rows.
These fixture measurements do not establish costs for larger proof profiles.
