Recursive verification of the saved Init KZG proof, using native Fr Plonkish
arithmetic and bounded Fq limbs. The circuit checks the transcript, AIR,
lookups, curve membership, subgroups and both KZG batching equations. Its
public claim contains the 18 Init words and 11 compressed pairing inputs.
The final verifier checks both external pairing equations against fixed keys.

```sh
python3 experiments/flock/prepare_terminal.py
cargo test --offline --release --manifest-path experiments/kzg-wrap/Cargo.toml
cargo build --offline --release --manifest-path experiments/kzg-wrap/Cargo.toml
experiments/kzg-wrap/target/release/init-kzg-wrap stage experiments/init-fri-kzg-artifacts target/init-kzg-recursive
experiments/kzg-wrap/target/release/init-kzg-wrap prove target/init-kzg-recursive
experiments/kzg-wrap/target/release/init-kzg-wrap verify target/init-kzg-recursive
```

The 53,157-byte input passes the complete circuit and pairing checks:
237,408,925 rows, one padded 2^28 computation trace plus a shared lookup table.
The outer proof is 2,053 bytes; its complete packet is 2,757 bytes. Setup and
proving took 10,011 seconds with 485.5 GiB peak RSS. Separate verification
passed, including external pairings and tampering tests (13.8 ms after setup).
See `../init-kzg-recursive-{circuit,proof,size}.json`.

The saved packet and verification metadata are in `../init-kzg-recursive-artifacts`.
Run `init-kzg-wrap verify experiments/init-kzg-recursive-artifacts` to recheck
them; the current harness regenerates the development SRS (about 11 minutes).
Affine formulas constrain nonzero denominators; exceptional intermediate
sums reject. This restriction was tested against the saved full proof.

Known-trapdoor development SRS only; these artifacts are not production-secure.
The older R1CS curve baseline remains in `curve.rs` and `measurements.json`.
Building that baseline requires the terminal crates from
`experiments/flock/prepare_terminal.py`.
