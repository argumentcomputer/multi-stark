//! Multi-circuit example with lookup arguments.
//!
//! Defines two circuits (Even and Odd) that compute whether an input number
//! is even or odd using a recursive lookup argument:
//!   - Even(n) pulls a lookup claim and, if n > 0, pushes to Odd(n-1).
//!   - Odd(n) pulls a lookup claim and, if n > 0, pushes to Even(n-1).
//!
//! The claim encodes the initial query: is_even(4) == 1.
//!
//! Run with:
//! ```sh
//! cargo run --example lookup_proof --release
//! ```

use multi_stark::expr::Expr;
use multi_stark::lookup::Lookup;
use multi_stark::system::{CircuitInputs, System, SystemWitness};
use multi_stark::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
use multi_stark::{
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::dense::RowMajorMatrix,
};

/// Width: 6 columns
/// [multiplicity, input, input_inverse, input_is_zero, input_not_zero, recursion_output].
fn parity_constraints() -> Vec<Expr<Val>> {
    let m = Expr::main(0);
    let input = Expr::main(1);
    let input_inv = Expr::main(2);
    let is_zero = Expr::main(3);
    let not_zero = Expr::main(4);
    let one = || Expr::constant(Val::ONE);
    vec![
        is_zero.clone() * (one() - is_zero.clone()),
        not_zero.clone() * (one() - not_zero.clone()),
        m * (is_zero.clone() + not_zero.clone() - one()),
        is_zero * input.clone(),
        not_zero * (input * input_inv - one()),
    ]
}

fn even_lookups() -> Vec<Lookup<Expr<Val>>> {
    let (m, input) = (Expr::main(0), Expr::main(1));
    let (is_zero, not_zero, rec) = (Expr::main(3), Expr::main(4), Expr::main(5));
    let even = Expr::constant(Val::ZERO);
    let odd = Expr::constant(Val::ONE);
    vec![
        // pull: negated multiplicity.
        Lookup {
            multiplicity: -m,
            args: vec![
                even,
                input.clone(),
                not_zero.clone() * rec.clone() + is_zero,
            ],
        },
        Lookup {
            multiplicity: not_zero,
            args: vec![odd, input - Expr::constant(Val::ONE), rec],
        },
    ]
}

fn odd_lookups() -> Vec<Lookup<Expr<Val>>> {
    let (m, input) = (Expr::main(0), Expr::main(1));
    let (not_zero, rec) = (Expr::main(4), Expr::main(5));
    let even = Expr::constant(Val::ZERO);
    let odd = Expr::constant(Val::ONE);
    vec![
        Lookup {
            multiplicity: -m,
            args: vec![odd, input.clone(), not_zero.clone() * rec.clone()],
        },
        Lookup {
            multiplicity: not_zero,
            args: vec![even, input - Expr::constant(Val::ONE), rec],
        },
    ]
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

    let even = CircuitInputs {
        main_width: 6,
        constraints: parity_constraints(),
        lookups: even_lookups(),
        ..Default::default()
    };
    let odd = CircuitInputs {
        main_width: 6,
        constraints: parity_constraints(),
        lookups: odd_lookups(),
        ..Default::default()
    };
    let (system, key) = System::new(config, [even, odd]);

    let f = Val::from_u32;
    #[rustfmt::skip]
    let witness = SystemWitness::from_stage_1(
        vec![
            // Even circuit trace
            RowMajorMatrix::new(
                vec![
                    f(1), f(4), f(4).inverse(), f(0), f(1), f(1),
                    f(1), f(2), f(2).inverse(), f(0), f(1), f(1),
                    f(1), f(0), f(0),            f(1), f(0), f(0),
                    f(0), f(0), f(0),            f(0), f(0), f(0),
                ],
                6,
            ),
            // Odd circuit trace
            RowMajorMatrix::new(
                vec![
                    f(1), f(3), f(3).inverse(), f(0), f(1), f(1),
                    f(1), f(1), f(1).inverse(), f(0), f(1), f(1),
                    f(0), f(0), f(0),            f(0), f(0), f(0),
                    f(0), f(0), f(0),            f(0), f(0), f(0),
                ],
                6,
            ),
        ],
        &system,
    );

    // Claim: [even_index=0, input=4, expected_output=1] — is_even(4) should be 1.
    let claim = &[f(0), f(4), f(1)];

    let proof = system.prove(&key, claim, witness);
    system.verify(claim, &proof).unwrap();
    println!("Lookup proof verified successfully!");

    let bytes = proof.to_bytes().expect("serialization failed");
    println!("Proof size: {} bytes", bytes.len());
}
