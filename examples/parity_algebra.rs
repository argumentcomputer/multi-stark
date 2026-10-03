//! Prove the algebraic checks of the fixed-height parity verifier.
//! NOT full recursive verification: transcript and PCS are still native.
//! Run with `cargo run --release --example parity_algebra`.

#[path = "support/parity_algebra.rs"]
mod adapter;
#[path = "support/parity.rs"]
mod parity;

use std::time::Instant;

use multi_stark::p3_field::PrimeCharacteristicRing;
use multi_stark::plonkish::CircuitBuilder;
use multi_stark::plonkish::verifier::{AlgebraicInputs, constrain_algebraic_checks};
use multi_stark::system::{System, SystemWitness};
use multi_stark::types::Val;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (inner, key) = System::new(parity::config(), parity::circuit_inputs());
    let mut builder = CircuitBuilder::new();
    let inputs = AlgebraicInputs::allocate(&mut builder, &inner.circuits, &[3]);
    constrain_algebraic_checks(
        &mut builder,
        &inner.circuits,
        &[parity::LOG_HEIGHT; 2],
        &inputs,
    );
    // Until the transcript and PCS gadgets exist, expose the complete
    // algebraic instance, not just the parity result.
    adapter::expose_boundary(&mut builder, &inputs);
    let circuit = builder.finish();
    println!(
        "Algebraic checks: {} gates, {} public coordinates",
        circuit.gates().len(),
        circuit.public_values().len()
    );
    let compiled = circuit.lower_to_multi_stark(Val::from_u32(101))?;
    println!(
        "Outer trace: {} rows × {} columns",
        compiled.main_height(),
        compiled.main_width()
    );
    let (outer, outer_key) = System::new(parity::config(), compiled.circuit_inputs());
    for (function, n) in [(parity::Function::Even, 100), (parity::Function::Odd, 127)] {
        let claim = parity::claim(function, n, function.result(n));
        let proof = inner.prove(
            &key,
            &claim,
            SystemWitness::from_stage_1(parity::traces(function, n), &inner),
        );
        inner.verify(&claim, &proof).expect("native inner verifier");
        let challenges = adapter::challenges(&inner, &proof, &[&claim]);
        let expected = adapter::statement(&proof, &[&claim], challenges);
        let start = Instant::now();
        let mut witness = compiled.witness();
        adapter::assign(&mut witness, &inputs, &proof, &[&claim], challenges)?;
        let assignment = witness.generate()?;
        assert_eq!(assignment.public_values(), expected);
        let witness_time = start.elapsed();
        let claims = compiled.claims(&expected)?;
        let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
        let start = Instant::now();
        let outer_proof = outer.prove_multiple_claims(
            &outer_key,
            &refs,
            SystemWitness::from_stage_1(compiled.traces(&assignment)?, &outer),
        );
        let prove_time = start.elapsed();
        outer
            .verify_multiple_claims(&refs, &outer_proof)
            .expect("algebraic instance proof");
        println!(
            "{function:?}({n}): witness {witness_time:?}, outer proving {prove_time:?}, {} proof bytes",
            outer_proof.to_bytes()?.len()
        );
    }
    println!(
        "Algebra only; no in-circuit transcript or PCS yet. {} queries: test security only.",
        parity::NUM_QUERIES
    );
    Ok(())
}
