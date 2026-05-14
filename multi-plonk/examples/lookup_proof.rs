//! Multi-circuit PLONK example with lookup arguments.
//!
//! Mirrors the multi-stark `lookup_proof` example. Two circuits (Even, Odd)
//! compute parity via a recursive lookup: Even(n) → Odd(n-1), Odd(n) → Even(n-1).
//! The claim encodes the initial query: is_even(4) == 1.
//!
//! Run with:
//! ```sh
//! cargo run --example lookup_proof --release
//! ```

use ark_ff::{Field, One, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::test_rng;

use multi_plonk::air::{Air, AirBuilder, BaseAir};
use multi_plonk::builder::symbolic::{SymbolicExpression, var};
use multi_plonk::lookup::{Lookup, LookupAir};
use multi_plonk::matrix::Matrix;
use multi_plonk::system::{System, SystemWitness};
use multi_plonk::types::{PlonkConfig, Val};

/// Width: [multiplicity, input, input_inverse, input_is_zero, input_not_zero, recursion_output]
enum ParityAir {
    Even,
    Odd,
}

impl ParityAir {
    fn lookups(&self) -> Vec<Lookup<SymbolicExpression>> {
        let multiplicity = var(0);
        let input = var(1);
        let input_is_zero = var(3);
        let input_not_zero = var(4);
        let recursion_output = var(5);
        let even_index: SymbolicExpression = Val::zero().into();
        let odd_index: SymbolicExpression = Val::one().into();
        let one: SymbolicExpression = Val::one().into();
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

impl BaseAir for ParityAir {
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
        let local = builder.main_local();
        let multiplicity = local[0];
        let input = local[1];
        let input_inverse = local[2];
        let input_is_zero = local[3];
        let input_not_zero = local[4];
        builder.assert_bool(input_is_zero);
        builder.assert_bool(input_not_zero);
        builder
            .when(multiplicity)
            .assert_one(input_is_zero.into() + input_not_zero.into());
        builder.when(input_is_zero).assert_zero(input);
        builder
            .when(input_not_zero)
            .assert_one(input.into() * input_inverse.into());
    }
}

fn main() {
    let mut rng = test_rng();
    // max_constraint_degree = 3 (not_zero * input * input_inverse - 1)
    // trace size = 4, q_factor = next_pow2(3) = 4, quotient_degree ≤ 3*4 = 12
    let config = PlonkConfig::setup(64, &mut rng);

    let even = LookupAir::new(ParityAir::Even, ParityAir::Even.lookups());
    let odd = LookupAir::new(ParityAir::Odd, ParityAir::Odd.lookups());
    let (system, key) = System::new(&config, [even, odd]);

    let f = |x: u64| Val::from(x);
    // When input = 0 the constraint is gated by input_not_zero = 0, so the
    // inverse column is left as 0 (any value is valid).
    let inv = |x: u64| -> Val {
        if x == 0 {
            Val::zero()
        } else {
            Val::from(x).inverse().unwrap()
        }
    };

    #[rustfmt::skip]
    let witness = SystemWitness::from_stage_1(
        vec![
            // Even circuit trace (4 rows × 6 cols):
            // [multiplicity, input, input_inverse, input_is_zero, input_not_zero, recursion_output]
            Matrix::new(
                vec![
                    f(1), f(4), inv(4), f(0), f(1), f(1),
                    f(1), f(2), inv(2), f(0), f(1), f(1),
                    f(1), f(0), f(0),   f(1), f(0), f(0),
                    f(0), f(0), f(0),   f(0), f(0), f(0),
                ],
                6,
            ),
            // Odd circuit trace (4 rows × 6 cols)
            Matrix::new(
                vec![
                    f(1), f(3), inv(3), f(0), f(1), f(1),
                    f(1), f(1), inv(1), f(0), f(1), f(1),
                    f(0), f(0), f(0),   f(0), f(0), f(0),
                    f(0), f(0), f(0),   f(0), f(0), f(0),
                ],
                6,
            ),
        ],
        &system,
    );

    // claim: [circuit_index=0, input=4, output=1] — is_even(4) should be 1
    let claim = &[f(0), f(4), f(1)];
    let proof = system.prove(&config, &key, claim, &witness);
    system
        .verify(&config, claim, &proof)
        .expect("verify failed");
    println!("Lookup proof verified successfully!");

    let mut bytes = vec![];
    proof.serialize_uncompressed(&mut bytes).expect("serialize");
    println!(
        "Proof size: {} bytes (uncompressed), {} bytes (compressed)",
        bytes.len(),
        proof.compressed_size()
    );
}
