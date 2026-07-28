//! U32 addition circuit, authored against the constraint-IR API.
//!
//! This is a port of the original AIR-based `test_circuits::u32_add` test to
//! the new frontend where circuits are described declaratively with
//! [`CircuitInputs`] (main/preprocessed widths, polynomial constraints, and
//! lookup pushes/pulls) rather than by implementing the `Air` trait.
//!
//! Two circuits form the system:
//!
//! - **byte table**: a read-only table backed by a preprocessed column holding
//!   the byte values `0..256`. Its single main column is a *multiplicity*
//!   counter. Each row *pulls* the argument `(BYTE_INDEX, byte_value)` from the
//!   lookup channel weighted by its multiplicity (a negated push), so the table
//!   supplies exactly as many copies of each byte as are consumed elsewhere.
//!
//! - **u32 add**: decomposes `x + y = z` into little-endian bytes, constrains
//!   the arithmetic (with an overflow carry), *pushes* every byte into the byte
//!   table for range-checking (proving each column really is a byte), and pulls
//!   a `(U32_INDEX, x, y, z)` tuple from the channel so that external *claims*
//!   can push matching tuples and balance the argument.
//!
//! The lookup argument balances iff:
//! - every byte pushed by the add rows is accounted for by the byte table's
//!   multiplicity column, and
//! - the number of active add rows (multiplicity column = 1) equals the number
//!   of `(U32_INDEX, x, y, z)` claims, each matching a computed row.

use crate::expr::Expr;
use crate::lookup::Lookup;
use crate::system::{CircuitInputs, System, SystemWitness};
use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

/// Lookup channel discriminant for the byte (range-check) table.
pub(super) const BYTE_INDEX: u32 = 0;
/// Lookup channel discriminant for u32-addition claims.
pub(super) const U32_INDEX: u32 = 1;

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

/// The byte table: preprocessed column `0..256`, main column = multiplicity.
///
/// | multiplicity | byte |
/// |            9 |    0 |
/// |            4 |    1 |
/// |            8 |    2 |
/// |          ... |  ... |
pub(super) fn byte_table() -> CircuitInputs<Val> {
    CircuitInputs {
        main_width: 1,
        preprocessed: Some(RowMajorMatrix::new(
            (0..256).map(Val::from_u32).collect(),
            1,
        )),
        // Pull (negated multiplicity): supply `main(0)` copies of this byte.
        lookups: vec![Lookup {
            multiplicity: -Expr::main(0),
            args: vec![c(BYTE_INDEX), Expr::preprocessed(0)],
        }],
        ..Default::default()
    }
}

/// The u32-add circuit.
///
/// Main columns (width 14):
/// - `0..4`  : bytes of `x` (little-endian)
/// - `4..8`  : bytes of `y`
/// - `8..12` : bytes of `z`
/// - `12`    : overflow carry (boolean)
/// - `13`    : multiplicity (1 on active rows, 0 on padding)
pub(super) fn u32_add() -> CircuitInputs<Val> {
    let carry = Expr::main(12);
    let expr1 = word(0) + word(4);
    let expr2 = word(8) + carry.clone() * Expr::constant(Val::from_u64(256 * 256 * 256 * 256));
    let constraints = vec![
        // carry must be boolean.
        carry.clone() * (carry.clone() - Expr::constant(Val::ONE)),
        // x + y == z + carry * 2^32.
        expr1 - expr2,
    ];
    // Pull the (index, x, y, z) tuple so an external claim can balance it.
    let mut lookups = vec![Lookup {
        multiplicity: -Expr::main(13),
        args: vec![c(U32_INDEX), word(0), word(4), word(8)],
    }];
    // Push every byte column into the byte table for range-checking.
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

/// Deterministic (x, y) pair for add row `i`, via an xorshift PRNG. Kept
/// identical between trace and claim construction so they stay consistent.
fn add_inputs(num_adds: usize) -> Vec<(u32, u32)> {
    let mut a: u32 = 0xdead_beef;
    let mut b: u32 = 0xcafe_babe;
    let mut out = Vec::with_capacity(num_adds);
    for _ in 0..num_adds {
        a ^= a << 13;
        a ^= a >> 17;
        a ^= a << 5;
        b ^= b << 13;
        b ^= b >> 17;
        b ^= b << 5;
        out.push((a, b));
    }
    out
}

/// Build the two stage-1 traces (byte table, then u32 add) for `num_adds`
/// additions. The add trace is padded to a power-of-two height with zero rows;
/// padding rows have multiplicity 0 so they contribute nothing to the argument.
pub(super) fn build_traces(num_adds: usize) -> Vec<RowMajorMatrix<Val>> {
    let byte_width = 1;
    let add_width = 14;
    let add_height = num_adds.next_power_of_two();

    let mut byte_trace = RowMajorMatrix::new(vec![Val::ZERO; byte_width * 256], byte_width);
    let mut add_trace = RowMajorMatrix::new(vec![Val::ZERO; add_width * add_height], add_width);

    for (row_index, (x, y)) in add_inputs(num_adds).into_iter().enumerate() {
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

        // Count each pushed byte in the table's multiplicity column.
        for &byte in xb.iter().chain(yb.iter()).chain(zb.iter()) {
            byte_trace.row_mut(byte as usize)[0] += Val::ONE;
        }
    }
    vec![byte_trace, add_trace]
}

/// External claims: one `[U32_INDEX, x, y, z]` per active add row, matching
/// exactly the tuples pulled by the u32-add circuit so the argument balances.
pub(super) fn build_claims(num_adds: usize) -> Vec<[Val; 4]> {
    let f = Val::from_u32;
    add_inputs(num_adds)
        .into_iter()
        .map(|(x, y)| {
            let (z, _carry) = x.overflowing_add(y);
            [f(U32_INDEX), f(x), f(y), f(z)]
        })
        .collect()
}

pub(super) fn config() -> GoldilocksBlake3Config {
    GoldilocksBlake3Config::new(
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
    )
}

#[test]
fn u32_add_proof() {
    // A small trace: 2^4 = 16 additions (already a power of two, so every row
    // is active and each pairs with exactly one claim).
    let num_adds = 1 << 4;

    let (system, key) = System::new(config(), [byte_table(), u32_add()]);

    let witness = SystemWitness::from_stage_1(build_traces(num_adds), &system);

    let claims = build_claims(num_adds);
    let claim_refs: Vec<&[Val]> = claims.iter().map(|c| c.as_slice()).collect();

    let proof = system.prove_multiple_claims(&key, &claim_refs, witness);
    system.verify_multiple_claims(&claim_refs, &proof).unwrap();
}
