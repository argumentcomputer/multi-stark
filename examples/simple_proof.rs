//! Minimal prove-and-verify example (no lookups).
//!
//! Defines a simple AIR that checks Pythagorean triples: `a² + b² == c²`.
//! Wraps it in `LookupAir` with no lookups, creates a small trace, and
//! proves + verifies it.
//!
//! Run with:
//! ```sh
//! cargo run --example simple_proof --release
//! ```

use multi_stark::lookup::LookupAir;
use multi_stark::system::{System, SystemWitness};
use multi_stark::types::{CommitmentParameters, FriParameters, Val};
use multi_stark::{
    p3_air::{Air, AirBuilder, BaseAir, WindowAccess},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::dense::RowMajorMatrix,
};

/// A simple AIR checking Pythagorean triples: a² + b² == c².
/// Width is 3 columns: [a, b, c].
struct PythagoreanAir;

impl<F> BaseAir<F> for PythagoreanAir {
    fn width(&self) -> usize {
        3
    }
}

impl<AB> Air<AB> for PythagoreanAir
where
    AB: AirBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        // Constraint: a² + b² == c²
        let lhs = local[0] * local[0] + local[1] * local[1];
        let rhs = local[2] * local[2];
        builder.assert_eq(lhs, rhs);
    }
}

fn main() {
    let commitment_parameters = CommitmentParameters {
        log_blowup: 1,
        cap_height: 0,
    };

    // Wrap the AIR with empty lookups
    let air = LookupAir::new(PythagoreanAir, vec![]);
    let (system, key) = System::new(commitment_parameters, [air]);

    // Build a trace with 4 rows of Pythagorean triples
    let f = Val::from_u32;
    let trace = RowMajorMatrix::new(
        vec![
            f(3),
            f(4),
            f(5),
            f(5),
            f(12),
            f(13),
            f(8),
            f(15),
            f(17),
            f(7),
            f(24),
            f(25),
        ],
        3,
    );
    let witness = SystemWitness::from_stage_1(vec![trace], &system);

    let fri_parameters = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 64,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };

    // Prove
    let no_claims: &[Val] = &[];
    let proof = system.prove(fri_parameters, &key, no_claims, witness);

    // Verify
    system.verify(fri_parameters, no_claims, &proof).unwrap();
    println!("Proof verified successfully!");

    // Show proof size
    let bytes = proof.to_bytes().expect("serialization failed");
    println!("Proof size: {} bytes", bytes.len());
}
