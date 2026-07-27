//! Criterion benchmarks for the multi-circuit STARK prover and verifier.
//!
//! Uses a U32 addition circuit:
//! - a **byte table** backed by a preprocessed byte table (256 rows) and
//!   lookup pulls,
//! - a **U32 add** that decomposes additions into bytes via lookup pushes.
//!
//! Run with:
//! ```sh
//! cargo bench --bench multi_stark --features parallel
//! ```
//!
//! Before each criterion measurement, one prove/verify runs under a
//! `tracing-texray`-examined span: every `info_span!` in the prover streams
//! a `[texray] <span>: <dur>  ── RAM Δ +X peak Y` line to stderr as it
//! closes, followed by the full span-tree timeline when the root exits.
//! By default only multi-stark's `stark/*` spans are rendered; set
//! `TEXRAY_PREFIXES=` (empty) to also render Plonky3's internal spans.

use criterion::{BenchmarkId, Criterion, criterion_group};

use multi_stark::expr::Expr;
use multi_stark::lookup::Lookup;
use multi_stark::system::{CircuitInputs, System, SystemWitness};
use multi_stark::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
use multi_stark::{p3_field::PrimeCharacteristicRing, p3_matrix::dense::RowMajorMatrix};

const BYTE_INDEX: u32 = 0;
const U32_INDEX: u32 = 1;

/// Install `tracing-texray` as the global subscriber, with per-span RAM
/// sampling and streaming output.
///
/// Spans are filtered by name prefix, like Ix's `rs_texray_init`:
/// `TEXRAY_PREFIXES` is a comma-separated list of span-name prefixes to
/// render, defaulting to `stark/,bench/` (multi-stark's own spans only).
/// An explicitly empty list (`TEXRAY_PREFIXES=`) disables filtering and
/// renders everything — including Plonky3's internal spans at every level
/// (FRI folds, merkle-tree builds, per-matrix DFTs, …).
fn init_texray() {
    use tracing_subscriber::{
        Layer, Registry, filter::FilterFn, layer::SubscriberExt, util::SubscriberInitExt,
    };
    let prefixes: Vec<String> = std::env::var("TEXRAY_PREFIXES")
        .unwrap_or_else(|_| "stark/,bench/".to_string())
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let layer = tracing_texray::TeXRayLayer::new().track_ram().streaming();
    let filter = FilterFn::new(move |metadata| {
        if !metadata.is_span() || prefixes.is_empty() {
            return true;
        }
        let name = metadata.name();
        prefixes.iter().any(|p| name.starts_with(p.as_str()))
    });
    Registry::default()
        .with(layer.with_filter(filter))
        .try_init()
        .ok();
}

fn config() -> GoldilocksBlake3Config {
    GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 100,
            commit_proof_of_work_bits: 10,
            query_proof_of_work_bits: 10,
        },
    )
}

fn c(value: u32) -> Expr<Val> {
    Expr::constant(Val::from_u32(value))
}

/// Little-endian 4-byte word starting at main column `base`.
fn word(base: u32) -> Expr<Val> {
    Expr::main(base)
        + Expr::main(base + 1) * c(256)
        + Expr::main(base + 2) * c(256 * 256)
        + Expr::main(base + 3) * c(256 * 256 * 256)
}

fn byte_table() -> CircuitInputs<Val> {
    CircuitInputs {
        main_width: 1,
        preprocessed: Some(RowMajorMatrix::new(
            (0..256).map(Val::from_u32).collect(),
            1,
        )),
        lookups: vec![Lookup {
            multiplicity: -Expr::main(0),
            args: vec![c(BYTE_INDEX), Expr::preprocessed(0)],
        }],
        ..Default::default()
    }
}

fn u32_add() -> CircuitInputs<Val> {
    let carry = Expr::main(12);
    let expr1 = word(0) + word(4);
    let expr2 = word(8) + carry.clone() * Expr::constant(Val::from_u64(256 * 256 * 256 * 256));
    let constraints = vec![
        carry.clone() * (carry.clone() - Expr::constant(Val::ONE)),
        expr1 - expr2,
    ];
    let mut lookups = vec![Lookup {
        multiplicity: -Expr::main(13),
        args: vec![c(U32_INDEX), word(0), word(4), word(8)],
    }];
    lookups.extend((0..12).map(|i| Lookup {
        multiplicity: Expr::constant(Val::ONE),
        args: vec![c(BYTE_INDEX), Expr::main(i)],
    }));
    CircuitInputs {
        main_width: 14,
        constraints,
        lookups,
        ..Default::default()
    }
}

fn build_traces(num_adds: usize) -> Vec<RowMajorMatrix<Val>> {
    let byte_width = 1;
    let add_width = 14;
    let add_height = num_adds.next_power_of_two();

    let mut byte_trace = RowMajorMatrix::new(vec![Val::ZERO; byte_width * 256], byte_width);
    let mut add_trace = RowMajorMatrix::new(vec![Val::ZERO; add_width * add_height], add_width);

    let mut a: u32 = 0xdead_beef;
    let mut b: u32 = 0xcafe_babe;
    for row_index in 0..num_adds {
        a ^= a << 13;
        a ^= a >> 17;
        a ^= a << 5;
        b ^= b << 13;
        b ^= b >> 17;
        b ^= b << 5;
        let (x, y) = (a, b);
        let (z, carry) = x.overflowing_add(y);
        let (xb, yb, zb) = (x.to_le_bytes(), y.to_le_bytes(), z.to_le_bytes());

        let row = add_trace.row_mut(row_index);
        for (col, &v) in row[0..4].iter_mut().zip(&xb) {
            *col = Val::from_u8(v);
        }
        for (col, &v) in row[4..8].iter_mut().zip(&yb) {
            *col = Val::from_u8(v);
        }
        for (col, &v) in row[8..12].iter_mut().zip(&zb) {
            *col = Val::from_u8(v);
        }
        row[12] = Val::from_u8(u8::from(carry));
        row[13] = Val::ONE;

        for &byte in xb.iter().chain(yb.iter()).chain(zb.iter()) {
            byte_trace.row_mut(byte as usize)[0] += Val::ONE;
        }
    }
    vec![byte_trace, add_trace]
}

fn build_claims(num_adds: usize) -> Vec<[Val; 4]> {
    let f = Val::from_u32;
    let mut a: u32 = 0xdead_beef;
    let mut b: u32 = 0xcafe_babe;
    let mut claims = Vec::with_capacity(num_adds);
    for _ in 0..num_adds {
        a ^= a << 13;
        a ^= a >> 17;
        a ^= a << 5;
        b ^= b << 13;
        b ^= b >> 17;
        b ^= b << 5;
        let (x, y) = (a, b);
        let (z, _carry) = x.overflowing_add(y);
        claims.push([f(1), f(x), f(y), f(z)]);
    }
    claims
}

fn bench_prove(criterion: &mut Criterion) {
    let (system, key) = System::new(config(), [byte_table(), u32_add()]);

    let mut group = criterion.benchmark_group("prove");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));

    for log_height in [12, 13, 14] {
        let num_adds = 1 << log_height;
        let claims = build_claims(num_adds);
        let claim_refs: Vec<&[Val]> = claims.iter().map(|c| c.as_slice()).collect();

        // One examined proof outside the measurement loop; spans opened during
        // criterion's iterations have no examined root and print nothing.
        let witness = SystemWitness::from_stage_1(build_traces(num_adds), &system);
        eprintln!("\n═══ texray: prove/u32_add/2^{log_height} (single traced run) ═══");
        tracing_texray::examine(tracing::info_span!("bench/prove", log_height))
            .in_scope(|| system.prove_multiple_claims(&key, &claim_refs, witness));
        eprintln!();

        group.bench_function(
            BenchmarkId::new("u32_add", format!("2^{log_height}")),
            |b| {
                b.iter_batched(
                    || SystemWitness::from_stage_1(build_traces(num_adds), &system),
                    |witness| system.prove_multiple_claims(&key, &claim_refs, witness),
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_verify(criterion: &mut Criterion) {
    let (system, key) = System::new(config(), [byte_table(), u32_add()]);

    let mut group = criterion.benchmark_group("verify");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    for log_height in [12, 13, 14] {
        let num_adds = 1 << log_height;
        let claims = build_claims(num_adds);
        let claim_refs: Vec<&[Val]> = claims.iter().map(|c| c.as_slice()).collect();
        let witness = SystemWitness::from_stage_1(build_traces(num_adds), &system);
        let proof = system.prove_multiple_claims(&key, &claim_refs, witness);

        eprintln!("\n═══ texray: verify/u32_add/2^{log_height} (single traced run) ═══");
        tracing_texray::examine(tracing::info_span!("bench/verify", log_height))
            .in_scope(|| system.verify_multiple_claims(&claim_refs, &proof).unwrap());
        eprintln!();

        group.bench_function(
            BenchmarkId::new("u32_add", format!("2^{log_height}")),
            |b| {
                b.iter(|| system.verify_multiple_claims(&claim_refs, &proof).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_prove, bench_verify);

// Expanded `criterion_main!` so the texray subscriber installs before any
// span is created.
fn main() {
    init_texray();
    benches();
    Criterion::default().configure_from_args().final_summary();
}
