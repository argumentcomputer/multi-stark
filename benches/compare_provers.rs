//! Head-to-head benchmark of the original AIR-based prover/verifier
//! (`multi_stark::{system, prover, verifier}`) against the new
//! constraint-IR prover/verifier (`multi_stark::constraint::*`), on the
//! *same* U32-addition problem (a byte table with a preprocessed trace plus
//! a U32-add circuit that decomposes into bytes via lookups).
//!
//! Run with:
//! ```sh
//! cargo bench --bench compare_provers --features parallel
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use multi_stark::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
use multi_stark::{
    p3_air::{Air, AirBuilder, BaseAir, WindowAccess},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::dense::RowMajorMatrix,
};

// Original AIR-based path.
use multi_stark::builder::symbolic::{SymbolicExpression, preprocessed_var, var};
use multi_stark::lookup::{Lookup as AirLookup, LookupAir};
use multi_stark::system::{System as AirSystem, SystemWitness as AirWitness};

// New constraint-IR path.
use multi_stark::constraint::expr::Expr;
use multi_stark::constraint::lookup::Lookup as IrLookup;
use multi_stark::constraint::system::{
    CircuitInputs, System as IrSystem, SystemWitness as IrWitness,
};

type SymbExpr = SymbolicExpression<Val>;

const BYTE_INDEX: u32 = 0;
const U32_INDEX: u32 = 1;

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

fn byte_table_trace() -> RowMajorMatrix<Val> {
    RowMajorMatrix::new((0..256).map(Val::from_u32).collect(), 1)
}

// ---------------------------------------------------------------------------
// Original AIR circuits
// ---------------------------------------------------------------------------

enum U32CS {
    ByteTable,
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
    fn lookups(&self) -> Vec<AirLookup<SymbExpr>> {
        let byte_index = SymbExpr::from_u32(BYTE_INDEX);
        let u32_index = SymbExpr::from_u32(U32_INDEX);
        match self {
            Self::ByteTable => vec![AirLookup::pull(
                var(0),
                vec![byte_index, preprocessed_var(0)],
            )],
            Self::U32Add => {
                let mut lookups = vec![AirLookup::pull(
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
                    (0..12)
                        .map(|i| AirLookup::push(SymbExpr::ONE, vec![byte_index.clone(), var(i)])),
                );
                lookups
            }
        }
    }
}

// ---------------------------------------------------------------------------
// New constraint-IR circuits (same problem)
// ---------------------------------------------------------------------------

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

fn ir_byte_table() -> CircuitInputs<Val> {
    CircuitInputs {
        main_width: 1,
        preprocessed: Some(byte_table_trace()),
        lookups: vec![IrLookup {
            multiplicity: -Expr::main(0),
            args: vec![c(BYTE_INDEX), Expr::preprocessed(0)],
        }],
        ..Default::default()
    }
}

fn ir_u32_add() -> CircuitInputs<Val> {
    let carry = Expr::main(12);
    let expr1 = word(0) + word(4);
    let expr2 = word(8) + carry.clone() * Expr::constant(Val::from_u64(256 * 256 * 256 * 256));
    let constraints = vec![
        carry.clone() * (carry.clone() - Expr::constant(Val::ONE)),
        expr1 - expr2,
    ];
    let mut lookups = vec![IrLookup {
        multiplicity: -Expr::main(13),
        args: vec![c(U32_INDEX), word(0), word(4), word(8)],
    }];
    lookups.extend((0..12).map(|i| IrLookup {
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

// ---------------------------------------------------------------------------
// Shared trace / claim generation
// ---------------------------------------------------------------------------

/// The two stage-1 traces (byte table, then U32-add), identical for both
/// systems.
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

const LOG_HEIGHTS: [usize; 3] = [12, 13, 14];

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_prove(criterion: &mut Criterion) {
    let (air_system, air_key) = AirSystem::new(
        config(),
        [
            LookupAir::new(U32CS::ByteTable, U32CS::ByteTable.lookups()),
            LookupAir::new(U32CS::U32Add, U32CS::U32Add.lookups()),
        ],
    );
    let (ir_system, ir_key) = IrSystem::new(config(), [ir_byte_table(), ir_u32_add()]);

    let mut group = criterion.benchmark_group("prove");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));

    for log_height in LOG_HEIGHTS {
        let num_adds = 1 << log_height;
        let claims = build_claims(num_adds);
        let claim_refs: Vec<&[Val]> = claims.iter().map(|c| c.as_slice()).collect();
        let size = format!("2^{log_height}");

        group.bench_function(BenchmarkId::new("air", &size), |b| {
            b.iter_batched(
                || AirWitness::from_stage_1(build_traces(num_adds), &air_system),
                |witness| air_system.prove_multiple_claims(&air_key, &claim_refs, witness),
                criterion::BatchSize::LargeInput,
            );
        });
        group.bench_function(BenchmarkId::new("ir", &size), |b| {
            b.iter_batched(
                || IrWitness::from_stage_1(build_traces(num_adds), &ir_system),
                |witness| ir_system.prove_multiple_claims(&ir_key, &claim_refs, witness),
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_verify(criterion: &mut Criterion) {
    let (air_system, air_key) = AirSystem::new(
        config(),
        [
            LookupAir::new(U32CS::ByteTable, U32CS::ByteTable.lookups()),
            LookupAir::new(U32CS::U32Add, U32CS::U32Add.lookups()),
        ],
    );
    let (ir_system, ir_key) = IrSystem::new(config(), [ir_byte_table(), ir_u32_add()]);

    let mut group = criterion.benchmark_group("verify");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    for log_height in LOG_HEIGHTS {
        let num_adds = 1 << log_height;
        let claims = build_claims(num_adds);
        let claim_refs: Vec<&[Val]> = claims.iter().map(|c| c.as_slice()).collect();
        let size = format!("2^{log_height}");

        let air_proof = {
            let witness = AirWitness::from_stage_1(build_traces(num_adds), &air_system);
            air_system.prove_multiple_claims(&air_key, &claim_refs, witness)
        };
        let ir_proof = {
            let witness = IrWitness::from_stage_1(build_traces(num_adds), &ir_system);
            ir_system.prove_multiple_claims(&ir_key, &claim_refs, witness)
        };
        // Sanity: both proofs verify before timing.
        air_system
            .verify_multiple_claims(&claim_refs, &air_proof)
            .unwrap();
        ir_system
            .verify_multiple_claims(&claim_refs, &ir_proof)
            .unwrap();

        group.bench_function(BenchmarkId::new("air", &size), |b| {
            b.iter(|| {
                air_system
                    .verify_multiple_claims(&claim_refs, &air_proof)
                    .unwrap()
            });
        });
        group.bench_function(BenchmarkId::new("ir", &size), |b| {
            b.iter(|| {
                ir_system
                    .verify_multiple_claims(&claim_refs, &ir_proof)
                    .unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_prove, bench_verify);
criterion_main!(benches);
