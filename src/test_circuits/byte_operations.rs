//! Byte-operation table proved against the constraint-IR API.
//!
//! This is a port of the original AIR-based `ByteCS` test. It proves the same
//! statement: a single circuit that owns a *preprocessed* byte table with
//! columns `[A, B, A xor B, A and B, A or B]` covering all 65536 byte pairs,
//! and a *main* trace of per-operation multiplicities. The circuit has no
//! ordinary polynomial constraints — correctness rests entirely on the lookup
//! argument. Each table row PULLs its operation (weighted by how many times it
//! was requested), and the public claims passed to `prove_multiple_claims`
//! act as the matching PUSHes, so the global lookup accumulator balances iff
//! every claimed `(op, A, B, result)` really appears in the table.

use crate::expr::Expr;
use crate::lookup::Lookup;
use crate::system::{CircuitInputs, System, SystemWitness};
use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

// Preprocessed columns are: [A, B, A xor B, A and B, A or B], where A and B are bytes.
const PREPROCESSED_TRACE_WIDTH: usize = 5;
// Main trace consists of multiplicities for each operation: `xor`, `and`, `or` and range check.
const TRACE_WIDTH: usize = 4;
const BYTE_VALUES_NUM: usize = 256;

/// The operations the byte table can answer. The discriminant doubles as the
/// operation's position in the main (multiplicity) trace *and* as the leading
/// argument in every lookup, so pushes and pulls only match for the same op.
enum ByteOperation {
    Xor,
    And,
    Or,
    PairU8Range,
}

impl ByteOperation {
    fn position(&self) -> usize {
        match self {
            Self::Xor => 0,
            Self::And => 1,
            Self::Or => 2,
            Self::PairU8Range => 3,
        }
    }
}

/// Builds the preprocessed byte table: one row per `(i, j)` byte pair, with the
/// precomputed xor/and/or results alongside the operands.
fn preprocessed_trace() -> RowMajorMatrix<Val> {
    let mut trace_values =
        Vec::with_capacity(BYTE_VALUES_NUM * BYTE_VALUES_NUM * PREPROCESSED_TRACE_WIDTH);
    for i in 0..BYTE_VALUES_NUM {
        for j in 0..BYTE_VALUES_NUM {
            let a = u8::try_from(i).expect("byte value fits u8");
            let b = u8::try_from(j).expect("byte value fits u8");
            trace_values.push(Val::from_u8(a));
            trace_values.push(Val::from_u8(b));
            trace_values.push(Val::from_u8(a ^ b));
            trace_values.push(Val::from_u8(a & b));
            trace_values.push(Val::from_u8(a | b));
        }
    }
    RowMajorMatrix::new(trace_values, PREPROCESSED_TRACE_WIDTH)
}

/// The lookups exposed by the byte table. Every one is a PULL (negated
/// multiplicity), drawing its multiplicity from the corresponding main column.
/// The args are `[op_index, A, B, result]` for the binary ops and
/// `[op_index, A, B]` for the range check (which has no return value).
fn byte_table_lookups() -> Vec<Lookup<Expr<Val>>> {
    let xor_idx = ByteOperation::Xor.position();
    let and_idx = ByteOperation::And.position();
    let or_idx = ByteOperation::Or.position();
    let pair_range_check_idx = ByteOperation::PairU8Range.position();

    let mut lookups = [xor_idx, and_idx, or_idx]
        .into_iter()
        .map(|i| {
            let col = u32::try_from(i).expect("column index fits u32");
            Lookup {
                // pull: negative multiplicity, taken from main column `i`.
                multiplicity: -Expr::main(col),
                args: vec![
                    Expr::constant(Val::from_usize(i)),
                    Expr::preprocessed(0),
                    Expr::preprocessed(1),
                    Expr::preprocessed(2 + col),
                ],
            }
        })
        .collect::<Vec<_>>();
    // Range checks do not have a return value.
    let range_col = u32::try_from(pair_range_check_idx).expect("column index fits u32");
    lookups.push(Lookup {
        multiplicity: -Expr::main(range_col),
        args: vec![
            Expr::constant(Val::from_usize(pair_range_check_idx)),
            Expr::preprocessed(0),
            Expr::preprocessed(1),
        ],
    });
    lookups
}

/// The single circuit: a preprocessed byte table with multiplicity columns and
/// lookup pulls, and no ordinary constraints.
fn byte_table_circuit() -> CircuitInputs<Val> {
    CircuitInputs {
        main_width: TRACE_WIDTH,
        preprocessed: Some(preprocessed_trace()),
        // No regular polynomial constraints — correctness relies entirely on lookups.
        lookups: byte_table_lookups(),
        ..Default::default()
    }
}

/// Builds the main (multiplicity) trace for a set of requested operations. For
/// each call `(op, x, y)` we bump the `op`-th multiplicity column of the row
/// keyed by `(x, y)` (`row_index = 256 * x + y`), mirroring the table layout.
fn build_witness(
    calls: &[(ByteOperation, u8, u8)],
    system: &System<GoldilocksBlake3Config>,
) -> SystemWitness<Val> {
    let mut byte_trace = RowMajorMatrix::new(
        vec![Val::ZERO; TRACE_WIDTH * BYTE_VALUES_NUM * BYTE_VALUES_NUM],
        TRACE_WIDTH,
    );
    for (op, x, y) in calls.iter() {
        let row_index = BYTE_VALUES_NUM * (*x as usize) + *y as usize;
        let row = byte_trace.row_mut(row_index);
        let position = op.position();
        row[position] += Val::ONE;
    }
    SystemWitness::from_stage_1(vec![byte_trace], system)
}

#[test]
fn byte_test() {
    let config = GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 64,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    );

    let (system, key) = System::new(config, [byte_table_circuit()]);

    // One request per operation; each bumps a single multiplicity by one.
    let calls = vec![
        (ByteOperation::Xor, 10, 5),
        (ByteOperation::And, 30, 20),
        (ByteOperation::Or, 100, 40),
        (ByteOperation::PairU8Range, 200, 100),
    ];
    let witness = build_witness(&calls, &system);

    // The claims are the PUSH side of the lookup argument: each must match a
    // table row so the accumulator balances to zero. Binary ops carry the
    // result; the range check does not.
    let f = Val::from_u32;
    let claim1 = &[f(0), f(10), f(5), f(10 ^ 5)];
    let claim2 = &[f(1), f(30), f(20), f(30 & 20)];
    let claim3 = &[f(2), f(100), f(40), f(100 | 40)];
    let claim4 = &[f(3), f(200), f(100)];
    let claims: &[&[Val]] = &[claim1, claim2, claim3, claim4];

    let proof = system.prove_multiple_claims(&key, claims, witness);
    system.verify_multiple_claims(claims, &proof).unwrap();
}
