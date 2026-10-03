//! Constrained transcript + algebra, still NOT a complete recursive verifier.
//! Run with `cargo run --release --example parity_transcript`.

#[path = "support/parity_algebra.rs"]
mod adapter;
#[path = "support/parity.rs"]
mod parity;

use std::time::Instant;

use multi_stark::p3_field::PrimeCharacteristicRing;
use multi_stark::plonkish::CircuitBuilder;
use multi_stark::plonkish::verifier::{
    AlgebraicInputs, ByteGadgets, TranscriptCommitments, constrain_algebraic_checks,
    constrain_transcript,
};
use multi_stark::system::{System, SystemWitness};
use multi_stark::types::Val;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (inner, key) = System::new(parity::config(), parity::circuit_inputs());
    let start = Instant::now();
    let mut builder = CircuitBuilder::new();
    let inputs = AlgebraicInputs::allocate(&mut builder, &inner.circuits, &[3]);
    let bytes = ByteGadgets::new(&mut builder);
    let roots = TranscriptCommitments::allocate(&mut builder, &bytes);
    constrain_transcript(
        &mut builder,
        &bytes,
        &inner,
        &[parity::LOG_HEIGHT; 2],
        &inputs,
        &roots,
    );
    constrain_algebraic_checks(
        &mut builder,
        &inner.circuits,
        &[parity::LOG_HEIGHT; 2],
        &inputs,
    );
    // This intermediate example omits PCS, so expose the complete instance
    // and all commitments, not merely the claimed parity result.
    adapter::expose_boundary(&mut builder, &inputs);
    for byte in roots
        .stage1
        .iter()
        .chain(&roots.stage2)
        .chain(&roots.quotient)
    {
        builder.expose_public(byte.value());
    }
    let circuit = builder.finish();
    println!(
        "{} gates, {} lookups, {} public coordinates; build {:?}",
        circuit.gates().len(),
        circuit.lookups().len(),
        circuit.public_values().len(),
        start.elapsed()
    );
    let compiled = circuit.lower_to_multi_stark(Val::from_u32(102))?;
    println!(
        "Outer computation trace: {} rows × {} columns",
        compiled.main_height(),
        compiled.main_width()
    );
    let start = Instant::now();
    let (outer, outer_key) = System::new(parity::config(), compiled.circuit_inputs());
    println!("Outer setup: {:?}", start.elapsed());
    for (function, n) in [(parity::Function::Even, 100), (parity::Function::Odd, 127)] {
        let claim = parity::claim(function, n, function.result(n));
        let proof = inner.prove(
            &key,
            &claim,
            SystemWitness::from_stage_1(parity::traces(function, n), &inner),
        );
        inner.verify(&claim, &proof).expect("native inner verifier");
        let challenges = adapter::challenges(&inner, &proof, &[&claim]);
        let digests = [
            proof.commitments.stage_1_trace.roots()[0],
            proof.commitments.stage_2_trace.roots()[0],
            proof.commitments.quotient_chunks.roots()[0],
        ];
        let mut expected = adapter::statement(&proof, &[&claim], challenges);
        expected.extend(digests.into_iter().flatten().map(Val::from_u8));
        let start = Instant::now();
        let mut witness = compiled.witness();
        adapter::assign(&mut witness, &inputs, &proof, &[&claim], challenges)?;
        for (wires, digest) in [&roots.stage1, &roots.stage2, &roots.quotient]
            .into_iter()
            .zip(digests)
        {
            for (wire, value) in wires.iter().zip(digest) {
                witness.set(wire.value(), Val::from_u8(value))?;
            }
        }
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
            .expect("transcript/algebra instance proof");
        println!(
            "{function:?}({n}): witness {witness_time:?}, outer proving {prove_time:?}, {} bytes",
            outer_proof.to_bytes()?.len()
        );
    }
    println!(
        "This example omits PCS/Merkle/FRI; see parity_recursive for full verification. Bounded field-sampling retries; {} queries: development security only.",
        parity::NUM_QUERIES
    );
    Ok(())
}
