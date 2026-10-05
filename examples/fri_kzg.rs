//! Compare KZG proofs of the same Goldilocks FRI verification circuit.
//! Development parameters and a deterministic, insecure test SRS.

#[cfg(not(feature = "kzg"))]
fn main() {
    eprintln!("enable --features kzg,parallel");
}

#[cfg(feature = "kzg")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use multi_stark::{
        ark_adapter::{config::KzgConfig, field::Scalar, srs::Srs},
        expr::Expr,
        lookup::Lookup,
        plonkish::{foreign::GoldilocksCircuit, verifier::*},
        system::{CircuitInputs, System, SystemWitness},
        traits::{Algebra, Field},
        types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val},
    };
    use p3_matrix::{Matrix, dense::RowMajorMatrix};
    use std::{sync::Arc, time::Instant};
    let args: Vec<_> = std::env::args().skip(1).collect();
    let generic = !args.iter().any(|s| s == "--compact");
    let check_only = args.iter().any(|s| s == "--check-only");
    let compare = args.iter().any(|s| s == "--compare");
    if args
        .iter()
        .any(|s| !["--compact", "--check-only", "--compare"].contains(&s.as_str()))
    {
        return Err("usage: fri_kzg [--compact] [--check-only] [--compare]".into());
    }
    let start = Instant::now();
    let config = GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 1,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    );
    let (inner, pk) = System::new(
        config,
        [CircuitInputs {
            main_width: 1,
            preprocessed: Some(RowMajorMatrix::new_col(vec![Val::ONE, Val::ZERO])),
            lookups: vec![Lookup::pull(Expr::preprocessed(0), vec![Expr::main(0)])],
            ..Default::default()
        }],
    );
    let statement = Statement {
        claims: vec![vec![Val::from_u8(13)]],
        messages: vec![],
    };
    let claim_refs: Vec<_> = statement.claims.iter().map(Vec::as_slice).collect();
    let proof = inner.prove_multiple_claims(
        &pk,
        &claim_refs,
        SystemWitness::from_stage_1(
            vec![RowMajorMatrix::new_col(vec![Val::from_u8(13); 2])],
            &inner,
        ),
    );
    inner.verify_multiple_claims(&claim_refs, &proof).unwrap();
    let key = VerifierKey::from_system(&inner);
    let plan = VerifierPlan::validate(
        &key,
        ProofProfile {
            envelope: Envelope::Ordinary,
            active: vec![true],
            log_degrees: vec![1],
            claim_lengths: vec![1],
            message_lengths: vec![],
            max_field_retries: 2,
        },
        VerifierLimits::default(),
    )?;
    let (circuit, inputs) = plan.build(
        Statement {
            claims: vec![vec![StatementSlot::Public]],
            messages: vec![],
        },
        ImplementationOptions {
            compact_blake3: !generic,
        },
    )?;
    let prepared = plan.expand_witness(ProofEnvelope::Ordinary {
        proof: &proof,
        claims: &statement.claims,
    })?;
    let mut witness = circuit.witness();
    inputs.assign_statement(&mut witness, &statement)?;
    prepared.assign_proof(&mut witness, &inputs)?;
    let assignment = witness.generate()?;
    let foreign = GoldilocksCircuit::new(&circuit);
    let mut witness = foreign.circuit.witness();
    foreign.assign(&assignment, &mut witness)?;
    let assignment = witness.generate()?;
    println!(
        "hash={}, inner_bytes={}, scalar_circuit={:?}, build_and_witness={:?}",
        if generic { "generic" } else { "compact" },
        proof.to_bytes()?.len(),
        foreign.circuit.stats(),
        start.elapsed()
    );
    let compiled = foreign.circuit.lower_to_multi_stark(Scalar::from_u8(93))?;
    let definitions = compiled.circuit_inputs();
    let max_height = definitions
        .iter()
        .map(|c| c.preprocessed.as_ref().unwrap().height())
        .max()
        .unwrap();
    println!("max_height={max_height}, traces={}", definitions.len());
    if check_only {
        return Ok(());
    }
    let start = Instant::now();
    let srs = Arc::new(Srs::unsafe_dev_setup(max_height, b"fri-kzg-example"));
    println!("SRS: {:?}", start.elapsed());
    let claims = compiled.claims(&[Scalar::from_u8(13)])?;
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    for &group in if compare { &[1, 0][..] } else { &[0][..] } {
        let mut circuits = if group == 0 {
            compiled.kzg_circuit_inputs(max_height, 8)?
        } else {
            definitions.clone()
        };
        if group != 0 {
            for c in &mut circuits {
                c.lookup_group_size = group;
            }
        }
        let start = Instant::now();
        let (outer, pk) = System::new(KzgConfig::new(srs.clone(), 8), circuits);
        let widths: Vec<_> = outer
            .circuits
            .iter()
            .map(|c| {
                (
                    c.main_width,
                    c.preprocessed_width,
                    c.stage_2_width,
                    c.quotient_degree(),
                )
            })
            .collect();
        let proof = outer.prove_multiple_claims(
            &pk,
            &refs,
            SystemWitness::from_stage_1(compiled.traces(&assignment)?, &outer),
        );
        outer.verify_multiple_claims(&refs, &proof).unwrap();
        let mut wrong = claims.clone();
        wrong[1][3] += Scalar::ONE;
        assert!(
            outer
                .verify_multiple_claims(
                    &wrong.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &proof
                )
                .is_err()
        );
        println!(
            "group={group}, columns(main,fixed,lookup,quotient)={widths:?}, proof_bytes={}, setup_and_prove={:?}; verified, altered claim rejected",
            proof.to_bytes()?.len(),
            start.elapsed()
        );
    }
    Ok(())
}
