//! A static circuit with native arithmetic, enforced copies, a fixed lookup
//! table, and a public output. Run with:
//! `cargo run --release --example plonkish_proof`

use multi_stark::p3_field::PrimeCharacteristicRing;
use multi_stark::plonkish::CircuitBuilder;
use multi_stark::system::{System, SystemWitness};
use multi_stark::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = CircuitBuilder::<Val>::new();
    let x = builder.input("x");
    let y = builder.input("y");
    let bytes = builder.fixed_table("byte", (0..256).map(|n| vec![Val::from_u32(n)]).collect());
    builder.lookup(bytes, &[x]);
    builder.lookup(bytes, &[y]);
    // A gadget accepts existing wires and leaves public-input policy to its
    // caller. measure() adds no constraints or persistent profiling data.
    let (sum, cost) = builder.measure(|builder| {
        let x2 = builder.mul(x, x);
        builder.mul_add(y, y, x2)
    });
    println!("Sum-of-squares gadget: {cost:?}");
    builder.expose_public(sum);

    // One application-assigned namespace for this lowering's lookup messages.
    let circuit = builder.finish();
    println!(
        "Circuit: {:?}; layout: {:?}",
        circuit.stats(),
        circuit.multi_stark_layout()?
    );
    let compiled = circuit.lower_to_multi_stark(Val::from_u32(42))?;
    let config = GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 100,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    );
    let (system, key) = System::new(config, compiled.circuit_inputs());
    let mut witness = compiled.witness();
    witness.set(x, Val::from_u32(3))?;
    witness.set(y, Val::from_u32(4))?;
    let assignment = witness.generate()?;
    let traces = compiled.traces(&assignment)?;

    // The verifier chooses the expected output independently of the witness.
    // claims() includes the mandatory activation anchor, even with no publics.
    let claims = compiled.claims(&[Val::from_u32(25)])?;
    let claims: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let proof =
        system.prove_multiple_claims(&key, &claims, SystemWitness::from_stage_1(traces, &system));
    system
        .verify_multiple_claims(&claims, &proof)
        .expect("valid Plonkish proof");
    println!(
        "Verified x² + y² = 25 with byte-constrained inputs ({} bytes)",
        proof.to_bytes()?.len()
    );
    Ok(())
}
