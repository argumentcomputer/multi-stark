//! Example with a preprocessed trace and lookups.
//!
//! Defines two circuits:
//! - **RangeTable**: a read-only byte table (0..256) committed as a
//!   preprocessed trace. Each row pulls a lookup weighted by its
//!   multiplicity column.
//! - **Squares**: computes `x * x` and range-checks both bytes of the result
//!   via lookup pushes into the RangeTable.
//!
//! Run with:
//! ```sh
//! cargo run --example preprocessed_proof --release
//! ```

use multi_stark::expr::Expr;
use multi_stark::lookup::Lookup;
use multi_stark::system::{CircuitInputs, System, SystemWitness};
use multi_stark::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
use multi_stark::{p3_field::PrimeCharacteristicRing, p3_matrix::dense::RowMajorMatrix};

fn range_table() -> CircuitInputs<Val> {
    // Preprocessed column: bytes 0..256. Main column: multiplicity.
    CircuitInputs {
        main_width: 1,
        preprocessed: Some(RowMajorMatrix::new(
            (0..256).map(Val::from_u32).collect(),
            1,
        )),
        // Pull: multiplicity × (preprocessed byte value).
        lookups: vec![Lookup {
            multiplicity: -Expr::main(0),
            args: vec![Expr::preprocessed(0)],
        }],
        ..Default::default()
    }
}

fn squares() -> CircuitInputs<Val> {
    // Columns: [x, x², low_byte, high_byte, multiplicity].
    let x = Expr::main(0);
    let x_squared = Expr::main(1);
    let low = Expr::main(2);
    let high = Expr::main(3);
    let mult = Expr::main(4);
    CircuitInputs {
        main_width: 5,
        constraints: vec![
            // x² == x * x
            x_squared.clone() - x.clone() * x,
            // x² == low + 256 * high
            x_squared - (low.clone() + high.clone() * Expr::constant(Val::from_u32(256))),
        ],
        // Push each byte into the range table for validation.
        lookups: vec![
            Lookup {
                multiplicity: mult.clone(),
                args: vec![low],
            },
            Lookup {
                multiplicity: mult,
                args: vec![high],
            },
        ],
        ..Default::default()
    }
}

fn main() {
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

    let (system, key) = System::new(config, [range_table(), squares()]);

    // Build traces: square every value 0..16.
    let n = 16u32;
    let f = Val::from_u32;

    let mut range_mults = vec![Val::ZERO; 256];
    let mut sq_values = Vec::with_capacity(5 * n as usize);
    for x in 0..n {
        let sq = x * x;
        let low = sq & 0xFF;
        let high = (sq >> 8) & 0xFF;
        sq_values.extend([f(x), f(sq), f(low), f(high), Val::ONE]);
        range_mults[low as usize] += Val::ONE;
        range_mults[high as usize] += Val::ONE;
    }

    let range_trace = RowMajorMatrix::new(range_mults, 1);
    let squares_trace = RowMajorMatrix::new(sq_values, 5);
    let witness = SystemWitness::from_stage_1(vec![range_trace, squares_trace], &system);

    let no_claims: &[&[Val]] = &[];
    let proof = system.prove_multiple_claims(&key, no_claims, witness);
    system.verify_multiple_claims(no_claims, &proof).unwrap();
    println!("Preprocessed proof verified successfully!");

    let bytes = proof.to_bytes().expect("serialization failed");
    println!("Proof size: {} bytes", bytes.len());
}
