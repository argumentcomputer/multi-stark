//! `cargo run --release --example root_profile`
//! `cargo run --release --example root_profile -- --prove-outer`
//! `cargo run --release --example root_profile -- --full-size --native-only`
//! --full-size --allow-large-circuit builds the 100-query circuit;
//! this explicit opt-in may exceed 16 GiB of RAM.

#[path = "support/root_profile.rs"]
mod fixture;

use std::time::Instant;

use multi_stark::p3_field::PrimeCharacteristicRing;
use multi_stark::plonkish::verifier::expand_pcs_witness;
use multi_stark::system::{System, SystemWitness};
use multi_stark::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut profile = fixture::Profile::Smoke;
    let mut native_only = false;
    let mut allow_large_circuit = false;
    let mut prove_outer = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--full-size" => profile = fixture::Profile::FullSize,
            "--native-only" => native_only = true,
            "--allow-large-circuit" => allow_large_circuit = true,
            "--prove-outer" => prove_outer = true,
            _ => {
                return Err(
                    "usage: root_profile [--full-size] [--native-only | --prove-outer] [--allow-large-circuit]"
                        .into(),
                );
            }
        }
    }
    if native_only && prove_outer {
        return Err("--native-only and --prove-outer are mutually exclusive".into());
    }
    if matches!(profile, fixture::Profile::FullSize) && !native_only && !allow_large_circuit {
        return Err("The 100-query Plonkish circuit may exceed 16 GiB. Use --native-only, or explicitly opt in with --allow-large-circuit on a larger machine.".into());
    }
    let start = Instant::now();
    let (system, key) = System::new(profile.config(), fixture::circuit_inputs(profile));
    let subject = [0x5a; 32];
    let claim = fixture::claim(subject);
    let proof = system.prove(
        &key,
        &claim,
        SystemWitness::from_stage_1(fixture::traces(profile, subject), &system),
    );
    system.verify(&claim, &proof).unwrap();
    println!(
        "{profile:?}: native proof verified in {:?}, {} bytes",
        start.elapsed(),
        proof.to_bytes()?.len()
    );
    println!(
        "Active {:?}; heights {:?}; quotient slices {:?}",
        proof.active,
        proof.log_degrees,
        system
            .circuits
            .iter()
            .map(|c| c.quotient_degree())
            .collect::<Vec<_>>()
    );
    let shape = profile.shape(&system);
    let expanded = expand_pcs_witness(&system, &shape, &proof, &[&claim])?;
    println!(
        "Expanded {} queries and {} FRI rounds",
        shape.queries, shape.log_trace
    );
    if native_only {
        return Ok(());
    }
    let start = Instant::now();
    let verifier = fixture::build_statement_verifier(&system, profile);
    println!(
        "Plonkish build {:?}: {:?}",
        start.elapsed(),
        verifier.circuit.stats()
    );
    println!("STARK layout: {:?}", verifier.circuit.multi_stark_layout()?);
    let mut witness = verifier.circuit.witness();
    expanded.assign_proof(&mut witness, &verifier.inputs)?;
    for (wire, byte) in verifier.subject.iter().zip(subject) {
        witness.set(wire.value(), Val::from_u8(byte))?;
    }
    let assignment = witness.generate()?;
    assert_eq!(assignment.public_values(), subject.map(Val::from_u8));
    println!("Complete statement-bound verifier constraints satisfied (synthetic workload).");
    if !prove_outer {
        return Ok(());
    }

    // Release inner proof/key data and the frontend before allocating the
    // large outer commitments. The outer profile matches parity_recursive:
    // blowup 2, four queries, binary FRI, zero grinding (development only).
    drop(expanded);
    drop(proof);
    drop(key);
    drop(system);
    let start = Instant::now();
    let compiled = verifier.circuit.lower_to_multi_stark(Val::from_u8(106))?;
    println!(
        "Outer lowering: {:?}; {} rows × {} advice columns",
        start.elapsed(),
        compiled.main_height(),
        compiled.main_width()
    );
    let public = subject.map(Val::from_u8);
    let claims = compiled.claims(&public)?;
    let mut wrong_public = public;
    wrong_public[0] += Val::ONE;
    let wrong_claims = compiled.claims(&wrong_public)?;
    let traces = compiled.traces(&assignment)?;
    let definitions = compiled.circuit_inputs();
    drop(compiled);
    drop(assignment);
    let config = GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 4,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    );
    let start = Instant::now();
    let (outer, outer_key) = System::new(config, definitions);
    println!("Outer setup: {:?}", start.elapsed());
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let start = Instant::now();
    let outer_proof = outer.prove_multiple_claims(
        &outer_key,
        &refs,
        SystemWitness::from_stage_1(traces, &outer),
    );
    let proving_time = start.elapsed();
    let size = outer_proof.to_bytes()?.len();
    let start = Instant::now();
    outer
        .verify_multiple_claims(&refs, &outer_proof)
        .expect("outer root-profile proof");
    let verification_time = start.elapsed();
    assert!(
        outer
            .verify_multiple_claims(
                &wrong_claims.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &outer_proof
            )
            .is_err()
    );
    println!(
        "Outer proof: {size} bytes; proving {proving_time:?}; verification {verification_time:?}"
    );
    println!(
        "Changed public subject rejected. Outer parameters: blowup 2, 4 queries, zero grinding; development security only."
    );
    if let Ok(status) = std::fs::read_to_string("/proc/self/status")
        && let Some(peak) = status.lines().find(|line| line.starts_with("VmHWM:"))
    {
        println!(
            "Whole-process peak RSS: {}",
            peak.trim_start_matches("VmHWM:").trim()
        );
    }
    Ok(())
}
