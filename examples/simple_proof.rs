//! Minimal prove-and-verify example (no lookups).
//!
//! Defines a circuit that checks Pythagorean triples: `a² + b² == c²`,
//! creates a small trace, and proves + verifies it.
//!
//! Run with:
//! ```sh
//! cargo run --example simple_proof --release
//! ```

use multi_stark::expr::Expr;
use multi_stark::system::{CircuitInputs, System, SystemWitness};
use multi_stark::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
use multi_stark::{p3_field::PrimeCharacteristicRing, p3_matrix::dense::RowMajorMatrix};

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

    // Width is 3 columns: [a, b, c], with the constraint a² + b² == c².
    let (a, b, c) = (Expr::main(0), Expr::main(1), Expr::main(2));
    let circuit = CircuitInputs {
        main_width: 3,
        constraints: vec![a.clone() * a + b.clone() * b - c.clone() * c],
        ..Default::default()
    };
    let (system, key) = System::new(config, [circuit]);

    // Build a trace with 4 rows of Pythagorean triples.
    let f = Val::from_u32;
    #[rustfmt::skip]
    let trace = RowMajorMatrix::new(
        vec![
            f(3), f(4), f(5),
            f(5), f(12), f(13),
            f(8), f(15), f(17),
            f(7), f(24), f(25),
        ],
        3,
    );
    let witness = SystemWitness::from_stage_1(vec![trace], &system);

    // Prove
    let no_claims: &[&[Val]] = &[];
    let proof = system.prove_multiple_claims(&key, no_claims, witness);

    // Verify
    system.verify_multiple_claims(no_claims, &proof).unwrap();
    println!("Proof verified successfully!");

    // Show proof size
    let bytes = proof.to_bytes().expect("serialization failed");
    println!("Proof size: {} bytes", bytes.len());
}
