//! Multi-circuit example with lookup arguments.
//!
//! Defines two AIR circuits (Even and Odd) that compute whether an input
//! number is even or odd using a recursive lookup argument:
//!   - Even(n) pulls a lookup claim and, if n > 0, pushes to Odd(n-1).
//!   - Odd(n) pulls a lookup claim and, if n > 0, pushes to Even(n-1).
//!
//! The claim encodes the initial query: is_even(4) == 1.
//!
//! Run with:
//! ```sh
//! cargo run --example lookup_proof --release
//! ```

use multi_stark::builder::symbolic::{SymbolicExpression, var};
use multi_stark::lookup::{Lookup, LookupAir};
use multi_stark::system::{System, SystemWitness};
use multi_stark::types::{CommitmentParameters, FriParameters, Val};
use multi_stark::{
    p3_air::{Air, AirBuilder, BaseAir, WindowAccess},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::dense::RowMajorMatrix,
};

/// Circuit for the Even/Odd parity check.
/// Width: 6 columns [multiplicity, input, input_inverse, input_is_zero, input_not_zero, recursion_output]
enum ParityAir {
    Even,
    Odd,
}

impl ParityAir {
    fn lookups(&self) -> Vec<Lookup<SymbolicExpression<Val>>> {
        let multiplicity = var(0);
        let input = var(1);
        let input_is_zero = var(3);
        let input_not_zero = var(4);
        let recursion_output = var(5);
        let even_index = Val::ZERO.into();
        let odd_index = Val::ONE.into();
        let one: SymbolicExpression<_> = Val::ONE.into();
        match self {
            Self::Even => vec![
                Lookup::pull(
                    multiplicity,
                    vec![
                        even_index,
                        input.clone(),
                        input_not_zero.clone() * recursion_output.clone() + input_is_zero,
                    ],
                ),
                Lookup::push(
                    input_not_zero,
                    vec![odd_index, input - one, recursion_output],
                ),
            ],
            Self::Odd => vec![
                Lookup::pull(
                    multiplicity,
                    vec![
                        odd_index,
                        input.clone(),
                        input_not_zero.clone() * recursion_output.clone(),
                    ],
                ),
                Lookup::push(
                    input_not_zero,
                    vec![even_index, input - one, recursion_output],
                ),
            ],
        }
    }
}

impl<F> BaseAir<F> for ParityAir {
    fn width(&self) -> usize {
        6
    }
}

impl<AB> Air<AB> for ParityAir
where
    AB: AirBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let multiplicity = local[0];
        let input = local[1];
        let input_inverse = local[2];
        let input_is_zero = local[3];
        let input_not_zero = local[4];
        builder.assert_bools([input_is_zero, input_not_zero]);
        builder
            .when(multiplicity)
            .assert_one(input_is_zero + input_not_zero);
        builder.when(input_is_zero).assert_zero(input);
        builder
            .when(input_not_zero)
            .assert_one(input * input_inverse);
    }
}

fn main() {
    let commitment_parameters = CommitmentParameters {
        log_blowup: 1,
        cap_height: 0,
    };

    let even = LookupAir::new(ParityAir::Even, ParityAir::Even.lookups());
    let odd = LookupAir::new(ParityAir::Odd, ParityAir::Odd.lookups());
    let (system, key) = System::new(commitment_parameters, [even, odd]);

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

    // Claim: [even_index=0, input=4, expected_output=1] — is_even(4) should be 1
    let claim = &[f(0), f(4), f(1)];
    let fri_parameters = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 64,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };

    let proof = system.prove(fri_parameters, &key, claim, witness);
    system.verify(fri_parameters, claim, &proof).unwrap();
    println!("Lookup proof verified successfully!");

    let bytes = proof.to_bytes().expect("serialization failed");
    println!("Proof size: {} bytes", bytes.len());
}
