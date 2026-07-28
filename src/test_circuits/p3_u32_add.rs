//! The u32-add circuit authored Plonky3-style, through the `p3_adapter`.
//!
//! Same circuit as [`super::u32_add`], but the constraints are written by
//! implementing `Air`/`BaseAir` and evaluating against the recording
//! builder. The lookups (which have no `p3_air` vocabulary) are attached
//! to the resulting [`CircuitInputs`] afterwards. Beyond the end-to-end
//! proof, the test asserts the compiled [`crate::graph::ConstraintGraph`]
//! is identical to the hand-authored circuit's — the two authoring styles
//! converge to the same canonical artifact.

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::PrimeCharacteristicRing;

use crate::expr::Expr;
use crate::lookup::Lookup;
use crate::p3_adapter::circuit_inputs_from_air;
use crate::system::{CircuitInputs, System, SystemWitness};
use crate::types::Val;

use super::u32_add::{
    BYTE_INDEX, U32_INDEX, build_claims, build_traces, byte_table, config, u32_add,
};

/// See [`super::u32_add::u32_add`] for the column layout: bytes of `x`,
/// `y`, `z` (little-endian), overflow carry, multiplicity.
struct U32AddAir;

impl BaseAir<Val> for U32AddAir {
    fn width(&self) -> usize {
        14
    }
}

impl<AB: AirBuilder<F = Val>> Air<AB> for U32AddAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        // Little-endian 4-byte word starting at column `base`.
        let word = |base: usize| -> AB::Expr {
            let limb = |i: usize, m: u32| row[base + i] * Val::from_u32(m);
            limb(0, 1) + limb(1, 256) + limb(2, 256 * 256) + limb(3, 256 * 256 * 256)
        };
        let carry = row[12];

        builder.assert_bool(carry);
        // x + y == z + carry * 2^32.
        builder.assert_eq(
            word(0) + word(4),
            word(8) + carry * AB::Expr::from(Val::from_u64(1 << 32)),
        );
    }
}

/// The adapter-built equivalent of [`super::u32_add::u32_add`].
fn u32_add_via_p3() -> CircuitInputs<Val> {
    let mut inputs = circuit_inputs_from_air(&U32AddAir);
    let c = |value: u32| Expr::constant(Val::from_u32(value));
    let word = |base: u32| {
        Expr::main(base)
            + Expr::main(base + 1) * c(256)
            + Expr::main(base + 2) * c(256 * 256)
            + Expr::main(base + 3) * c(256 * 256 * 256)
    };
    inputs.lookups.push(Lookup {
        multiplicity: -Expr::main(13),
        args: vec![c(U32_INDEX), word(0), word(4), word(8)],
    });
    inputs.lookups.extend((0..12).map(|i| Lookup {
        multiplicity: Expr::constant(Val::ONE),
        args: vec![c(BYTE_INDEX), Expr::main(i)],
    }));
    inputs
}

#[test]
fn p3_adapter_u32_add_proof() {
    let num_adds = 1 << 4;

    let (system, key) = System::new(config(), [byte_table(), u32_add_via_p3()]);

    // Both authoring styles must compile to the same canonical graph.
    let (reference, _) = System::new(config(), [byte_table(), u32_add()]);
    assert_eq!(system.circuits[1].graph, reference.circuits[1].graph);

    let witness = SystemWitness::from_stage_1(build_traces(num_adds), &system);

    let claims = build_claims(num_adds);
    let claim_refs: Vec<&[Val]> = claims.iter().map(|c| c.as_slice()).collect();

    let proof = system.prove_multiple_claims(&key, &claim_refs, witness);
    system.verify_multiple_claims(&claim_refs, &proof).unwrap();
}
