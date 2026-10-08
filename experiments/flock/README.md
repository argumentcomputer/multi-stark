Experimental Plonkish-to-Flock backend for Goldilocks arithmetic, supported
lookups, BLAKE3, and copy wiring. Unsupported tables are rejected.

```sh
python3 experiments/flock/prepare.py
python3 experiments/flock/prepare_terminal.py --ix-source ../ix
RUSTFLAGS='-C target-cpu=native' cargo build --release --no-default-features --manifest-path experiments/flock/Cargo.toml
RUSTFLAGS='-C target-cpu=native' cargo test --release --manifest-path experiments/flock/Cargo.toml
python3 experiments/flock/run.py --init-artifacts <ix-artifacts>
python3 experiments/flock/run_init.py --init-artifacts <ix-artifacts> --output target/flock-experiment/init-run --low-memory
```

`run.py` measures small fixtures and an Init census with operation counters.
Build with `--no-default-features` for timing full proofs without counter
contention; unavailable operation counts are reported as null. `run_init.py` attempts full
Init proving and verification with its original proof and 18 public words.
It defaults to strict Fast parameters, 32 threads, and a 400 GiB RSS limit.
Results, phase logs, binary/source hashes, and process peak RAM go in the output
directory. `--low-memory` drops returned scratch buffers and schedules the
forked Boolean/wiring protocols sequentially. PCS buffers are regenerated
after those protocols, with commitment equality checked; the transcript is
unchanged.
Create `STOP` there to terminate the attempt. Use a fresh directory
for each run.

`--initial-k 4` or `5` rederives the strict Init PCS schedule with narrower
opening rows. Selection freezes before the first profile lookup; each run saves
`pcs-profile.toml` and a binary snapshot. The default remains unchanged.

`--coalesced` uses one table per operation type. The census reports its padding
cost alongside the original layout. Terminal runs write completed phase row
counts to `phase-progress.json`; unfinished or deferred checks are excluded.

The lowering reuses computed field wires, specializes bounded integer sums and
scalings, eliminates internal XOR wires, and packs operation batches into rows.
Goldilocks multiplication uses unsigned Karatsuba with constrained carries and
borrows. The census reports each operation's live witness cost.
It preserves existing output bindings and derives bounds from constraints.
[Lowering measurements](lowering-results.json) separate these savings from the
outstanding terminal-verifier work. Witness generation runs in place across CPU
workers. IO advice is constrained by relations and
copy wiring. Native verification discharges matrix, wiring, and layout claims;
small-proof tests reject mutations of each. The terminal census constrains fixed matrix and layout evaluations, live-cell
masks, and a copy-permutation opening against a circuit-derived commitment.

`prepare.py` applies pinned patches under `target/`: derived Fast/Slim
profiles through `m=40`, size-balanced wiring joins, and low-memory scheduling. It leaves Cargo's cache
unchanged. Init's three padded witness arrays total 96 GiB with the default
packing and 192 GiB with coalescing; PCS, wiring, and
scratch are additional. Array sizes are not measured process peaks.

The complete Groth16 Flock verifier and composed 128-bit security validation
remain outstanding. The Groth16 field gadget uses a development setup. Small
FRI fixtures use development security. [Earlier measurements](results.json)
include their source hashes. [Optimization measurements](optimization-results.json)
record the verified 453,555-byte Init Flock proof and terminal sizing attempts.
[Opening-width trials](width-results.json) verified 426,995-byte (`k=5`) and
436,219-byte (`k=4`) Init proofs. Both increased native peak RAM and still hit
the KZG FFT limit before finishing PCS verification; the default is retained. Fixed-table discharges are additional work; no
complete KZG/Groth16 wrapper has been proved.

[Terminal optimizations](terminal-optimization-results.json) complete the
conditional Init replay at 1,286,877,147 rows (domain `2^31`). File products use
bounded transforms; preprocessing uses small root tables; gates share their
coefficients. Compact transport is 640 bytes plus the unchanged claim, verified
on development fixtures. The full Init terminal proof remains unproduced:
full-size proving integration, adequate storage, and production setup remain. The complete Init census includes all 192 fixed claims: 3,897,983,003 rows
on domain `2^32`, 45.5 minutes total, and 227 GiB peak RSS. This is a census,
not terminal proving. [Fixed-check measurements](fixed-discharge-results.json)
record both fixtures and the backend storage changes. File witness lowering streams A/B and retains C for auxiliary
references, saving two full-column allocations; this is not total proving RAM.
The file key stores eight coefficient columns; permutation evaluations and
packed polynomials are reconstructed from existing data.

[Verifier coverage](verifier-audit.json) maps native acceptance checks to circuit
constraints and records the query-count/domain fixes and adversarial tests.

KZG experiments use pinned terminal-verifier crates materialized by
`prepare_terminal.py` under `target/`. Build with
`--no-default-features --features terminal`, then run `terminal-fri <out>` or
`run_init.py --init-artifacts <ix-artifacts> --output <out> --low-memory --terminal-census`.
These commands regenerate the intermediate Flock proof with chained BLAKE3 and
measure the terminal circuit without retaining its matrices. The original FRI
proof and public claim are unchanged. `terminal.json` records completed checks; `all_fixed_claims_projected` means
the census reached every fixed-table discharge. Partial runs are lower bounds.
Neither status is a terminal proof. Use `--profile slim` to measure the strict Slim profile. The
sizing pass stops when completed rows exceed the base-domain FFT limit;
`projection-progress.json` records live counts. The KZG arithmetic test uses a
development SRS. [KZG measurements](terminal-results.json) include both Init profiles.
