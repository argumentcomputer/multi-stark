//! The fixed-height inner proof for the Plonkish verifier experiment.
//! This example uses the native verifier; the in-circuit verifier is next.
//! Run with `cargo run --release --example parity_proof`.

#[path = "support/parity.rs"]
mod parity;

use multi_stark::system::{System, SystemWitness};
use parity::{Function, HEIGHT, LOG_HEIGHT, NUM_QUERIES};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (system, key) = System::new(parity::config(), parity::circuit_inputs());
    for (function, n) in [(Function::Even, 100), (Function::Odd, 127)] {
        let expected = function.result(n);
        let claim = parity::claim(function, n, expected);
        let witness = SystemWitness::from_stage_1(parity::traces(function, n), &system);
        let proof = system.prove(&key, &claim, witness);
        assert_eq!(proof.active, vec![true; 2]);
        assert_eq!(proof.log_degrees, vec![LOG_HEIGHT; 2]);
        system.verify(&claim, &proof).expect("valid parity proof");
        println!(
            "Verified is_{function:?}({n}) = {expected}: two {HEIGHT}-row traces, {} bytes",
            proof.to_bytes()?.len()
        );
    }
    println!("Development fixture only: {NUM_QUERIES} FRI queries, not production security.");
    Ok(())
}
