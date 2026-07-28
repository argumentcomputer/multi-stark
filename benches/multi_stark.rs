//! Criterion benchmarks for the multi-circuit STARK prover and verifier.
//!
//! Uses a U32 addition circuit with:
//! - A **ByteTable** backed by a preprocessed byte table (256 rows) and lookup pulls
//! - A **U32Add** that decomposes additions into bytes via lookup pushes
//!
//! This exercises lookups, preprocessed traces, and regular constraints –
//! giving a more representative cost profile than plain arithmetic AIRs.
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
use multi_stark::builder::symbolic::{SymbolicExpression, preprocessed_var, var};
use multi_stark::lookup::{Lookup, LookupAir};
use multi_stark::system::{ProverKey, System, SystemWitness};
use multi_stark::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
use multi_stark::{
    p3_air::{Air, AirBuilder, BaseAir, WindowAccess},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::dense::RowMajorMatrix,
};

type SymbExpr = SymbolicExpression<Val>;

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

// ---------------------------------------------------------------------------
// Circuits
// ---------------------------------------------------------------------------

enum U32CS {
    /// Preprocessed byte table (256 rows). Main trace column: multiplicity.
    ByteTable,
    /// U32 addition: 4 bytes x + 4 bytes y + 4 bytes z + carry + multiplicity = 14 cols.
    U32Add,
}

impl<F: Field> BaseAir<F> for U32CS {
    fn width(&self) -> usize {
        match self {
            Self::ByteTable => 1,
            Self::U32Add => 14,
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            Self::ByteTable => Some(RowMajorMatrix::new((0..256).map(F::from_u32).collect(), 1)),
            Self::U32Add => None,
        }
    }
}

impl<AB> Air<AB> for U32CS
where
    AB: AirBuilder,
    AB::Var: Copy,
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::ByteTable => {}
            Self::U32Add => {
                let main = builder.main();
                let local = main.current_slice();
                let x = &local[0..4];
                let y = &local[4..8];
                let z = &local[8..12];
                let carry = local[12];
                builder.assert_bool(carry);

                let expr1 = x[0]
                    + x[1] * AB::Expr::from_u32(256)
                    + x[2] * AB::Expr::from_u32(256 * 256)
                    + x[3] * AB::Expr::from_u32(256 * 256 * 256)
                    + y[0]
                    + y[1] * AB::Expr::from_u32(256)
                    + y[2] * AB::Expr::from_u32(256 * 256)
                    + y[3] * AB::Expr::from_u32(256 * 256 * 256);
                let expr2 = z[0]
                    + z[1] * AB::Expr::from_u32(256)
                    + z[2] * AB::Expr::from_u32(256 * 256)
                    + z[3] * AB::Expr::from_u32(256 * 256 * 256)
                    + carry * AB::Expr::from_u64(256 * 256 * 256 * 256);
                builder.assert_eq(expr1, expr2);
            }
        }
    }
}

impl U32CS {
    fn lookups(&self) -> Vec<Lookup<SymbExpr>> {
        let byte_index = SymbExpr::from_u8(0);
        let u32_index = SymbExpr::from_u8(1);
        match self {
            Self::ByteTable => vec![Lookup::pull(var(0), vec![byte_index, preprocessed_var(0)])],
            Self::U32Add => {
                let mut lookups = vec![Lookup::pull(
                    var(13),
                    vec![
                        u32_index,
                        var(0)
                            + var(1) * SymbExpr::from_u32(256)
                            + var(2) * SymbExpr::from_u32(256 * 256)
                            + var(3) * SymbExpr::from_u32(256 * 256 * 256),
                        var(4)
                            + var(5) * SymbExpr::from_u32(256)
                            + var(6) * SymbExpr::from_u32(256 * 256)
                            + var(7) * SymbExpr::from_u32(256 * 256 * 256),
                        var(8)
                            + var(9) * SymbExpr::from_u32(256)
                            + var(10) * SymbExpr::from_u32(256 * 256)
                            + var(11) * SymbExpr::from_u32(256 * 256 * 256),
                    ],
                )];
                lookups.extend(
                    (0..12).map(|i| Lookup::push(SymbExpr::ONE, vec![byte_index.clone(), var(i)])),
                );
                lookups
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Witness generation
// ---------------------------------------------------------------------------

fn build_witness(
    num_adds: usize,
    system: &System<GoldilocksBlake3Config, U32CS>,
) -> SystemWitness<Val> {
    let byte_width = 1;
    let add_width = 14;
    let add_height = num_adds.next_power_of_two();

    let mut byte_trace = RowMajorMatrix::new(vec![Val::ZERO; byte_width * 256], byte_width);
    let mut add_trace = RowMajorMatrix::new(vec![Val::ZERO; add_width * add_height], add_width);

    // Fill with pseudo-random additions (deterministic for reproducibility).
    let mut a: u32 = 0xdead_beef;
    let mut b: u32 = 0xcafe_babe;
    for row_index in 0..num_adds {
        // Simple xorshift-style PRNG
        a ^= a << 13;
        a ^= a >> 17;
        a ^= a << 5;
        b ^= b << 13;
        b ^= b >> 17;
        b ^= b << 5;

        let x = a;
        let y = b;
        let (z, carry) = x.overflowing_add(y);
        let x_bytes = x.to_le_bytes();
        let y_bytes = y.to_le_bytes();
        let z_bytes = z.to_le_bytes();

        let row = add_trace.row_mut(row_index);
        for (col, &val) in row[0..4].iter_mut().zip(&x_bytes) {
            *col = Val::from_u8(val);
        }
        for (col, &val) in row[4..8].iter_mut().zip(&y_bytes) {
            *col = Val::from_u8(val);
        }
        for (col, &val) in row[8..12].iter_mut().zip(&z_bytes) {
            *col = Val::from_u8(val);
        }
        row[12] = Val::from_u8(u8::from(carry));
        row[13] = Val::ONE;

        for &byte in x_bytes.iter().chain(y_bytes.iter()).chain(z_bytes.iter()) {
            byte_trace.row_mut(byte as usize)[0] += Val::ONE;
        }
    }

    SystemWitness::from_stage_1(vec![byte_trace, add_trace], system)
}

/// Build claims for the first `num_adds` additions (same PRNG seed as `build_witness`).
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
        let x = a;
        let y = b;
        let (z, _carry) = x.overflowing_add(y);
        claims.push([f(1), f(x), f(y), f(z)]);
    }
    claims
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_config() -> GoldilocksBlake3Config {
    GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 2,
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

fn build_system() -> (
    System<GoldilocksBlake3Config, U32CS>,
    ProverKey<GoldilocksBlake3Config>,
) {
    let byte_table = LookupAir::new(U32CS::ByteTable, U32CS::ByteTable.lookups());
    let u32_add = LookupAir::new(U32CS::U32Add, U32CS::U32Add.lookups());
    System::new(bench_config(), [byte_table, u32_add])
}

fn bench_prove(c: &mut Criterion) {
    let (system, key) = build_system();

    let mut group = c.benchmark_group("prove");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));

    for log_height in [12, 13, 14] {
        let num_adds = 1 << log_height;
        let claims = build_claims(num_adds);
        let claim_refs: Vec<&[Val]> = claims.iter().map(|c| c.as_slice()).collect();

        // One examined proof outside the measurement loop; spans opened during
        // criterion's iterations have no examined root and print nothing.
        let witness = build_witness(num_adds, &system);
        eprintln!("\n═══ texray: prove/u32_add/2^{log_height} (single traced run) ═══");
        tracing_texray::examine(tracing::info_span!("bench/prove", log_height))
            .in_scope(|| system.prove_multiple_claims(&key, &claim_refs, witness));
        eprintln!();

        group.bench_function(
            BenchmarkId::new("u32_add", format!("2^{log_height}")),
            |b| {
                b.iter_batched(
                    || build_witness(num_adds, &system),
                    |witness| system.prove_multiple_claims(&key, &claim_refs, witness),
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let (system, key) = build_system();

    let mut group = c.benchmark_group("verify");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    for log_height in [12, 13, 14] {
        let num_adds = 1 << log_height;
        let claims = build_claims(num_adds);
        let claim_refs: Vec<&[Val]> = claims.iter().map(|c| c.as_slice()).collect();
        let witness = build_witness(num_adds, &system);
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
