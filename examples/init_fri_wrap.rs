//! Measure verification of the saved recursive Init FRI proof in Groth16.
#[cfg(feature = "groth16")]
#[path = "support/init_fri.rs"]
mod init_fri;
#[cfg(not(feature = "groth16"))]
fn main() {
    eprintln!("enable --features groth16,parallel");
}

#[cfg(feature = "groth16")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ark_bls12_381::{Fr, G1Affine, G2Affine};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use multi_stark::{
        plonkish::{r1cs, verifier::*},
        types::Val,
    };
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use std::{fs, path::PathBuf, time::Instant};

    let args: Vec<_> = std::env::args().skip(1).collect();
    use tracing_subscriber::prelude::*;
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_filter(
                    tracing_subscriber::filter::Targets::new()
                        .with_target("multi_stark::streaming", tracing::Level::INFO),
                ),
        )
        .init();
    let streamed = args
        .last()
        .is_some_and(|a| matches!(a.as_str(), "--stream-prove" | "--stream-check"));
    let prove = args
        .last()
        .is_some_and(|a| matches!(a.as_str(), "--prove" | "--stream-prove"));
    let preflight = args
        .last()
        .is_some_and(|a| matches!(a.as_str(), "--preflight" | "--stream-check"));
    let bundle = args.len() == 3 && args[2] == "--bundle";
    if args.len() != 2 && args.len() != 4 && !(args.len() == 5 && (prove || preflight)) && !bundle {
        return Err("usage: init_fri_wrap <recovered-artifacts> <output-dir> [queries-per-shard shard-index [--prove] | --bundle]".into());
    }
    let dir = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    fs::create_dir_all(&out)?;
    fs::write(
        out.join("binary-blake3.txt"),
        ::blake3::hash(&fs::read(std::env::current_exe()?)?)
            .to_hex()
            .as_str(),
    )?;
    if prove && !streamed && args[3] != "0" {
        // Reserve the measured dense key footprint before testing the proving
        // R1CS. The parent has not allocated a circuit or proving key yet.
        let budget_kib = ((450u64 << 30) - 154_970_806_392) / 1024;
        let mut child_args = args.clone();
        *child_args.last_mut().unwrap() = "--preflight".into();
        println!(
            "Checking proving R1CS under {budget_kib} KiB, reserving 154970806392 bytes for key arrays"
        );
        let status = std::process::Command::new("bash")
            .arg("-c")
            .arg(format!("ulimit -v {budget_kib}; exec \"$@\""))
            .arg("--")
            .arg(std::env::current_exe()?)
            .args(child_args)
            .status()?;
        if !status.success() {
            return Err(format!(
                "proving R1CS memory preflight failed ({status}); full setup was not attempted"
            )
            .into());
        }
    }
    let init_fri::Fixture {
        key,
        proof,
        public,
        claims,
        profile,
        schema,
    } = init_fri::load(&dir)?;
    let plan = VerifierPlan::validate(&key, profile, VerifierLimits::default())?;
    if bundle {
        return verify_bundle(&out, &public);
    }
    let start = Instant::now();
    let options = ImplementationOptions {
        compact_blake3: true,
    };
    let partition = if args.len() >= 4 {
        Some(QueryShardPlan::new(
            &plan,
            schema.clone(),
            options,
            args[2].parse()?,
        )?)
    } else {
        None
    };
    let shard = if partition.is_some() {
        args[3].parse()?
    } else {
        0
    };
    let counts = if let Some(p) = &partition {
        println!(
            "Partition: {} proofs; measuring shard {shard}, queries {:?}; packet {} bytes",
            p.shard_count(),
            p.query_range(shard)?,
            4 + 32 + 18 * 8 + 32 + 192 * p.shard_count()
        );
        p.estimate(shard)?
    } else {
        plan.estimate(schema.clone(), options)?
    };
    println!("Frontend census: {counts:?}; {:?}", start.elapsed());
    if counts.values > 200_000_000 {
        return Err("frontend exceeds census memory budget".into());
    }
    let (circuit, inputs) = if let Some(p) = &partition {
        p.build(shard)?
    } else {
        plan.build(schema, options)?
    };
    assert_eq!(counts, circuit.stats());
    let prepared = plan.expand_witness(ProofEnvelope::Ordinary {
        proof: &proof,
        claims: &claims,
    })?;
    let mut witness = circuit.witness();
    inputs.assign_statement(
        &mut witness,
        &Statement {
            claims: claims.clone(),
            messages: vec![],
        },
    )?;
    prepared.assign_proof(&mut witness, &inputs)?;
    let assignment = witness.generate()?;
    assert_eq!(&assignment.public_values()[..18], public);
    if partition.is_some() {
        assert_eq!(assignment.public_values().len(), 51);
        assert_eq!(assignment.public_values()[50], Val::from_usize(shard));
        fs::write(
            out.join("context.bin"),
            assignment.public_values()[18..50]
                .iter()
                .map(|v| u8::try_from(v.as_canonical_u64()).unwrap())
                .collect::<Vec<_>>(),
        )?;
    }
    println!(
        "Full compact verifier witness satisfied; public values equal the original 18-word Init claim; {:?}",
        start.elapsed()
    );
    drop(prepared);
    drop(inputs);
    let start = Instant::now();
    let cost = r1cs::estimate_goldilocks(&circuit)?;
    assert_eq!(
        cost.public_inputs,
        if partition.is_some() { 51 } else { 18 }
    );
    let domain = GeneralEvaluationDomain::<Fr>::new(cost.constraints + cost.public_inputs + 1)
        .ok_or("unsupported QAP domain")?
        .size();
    let vars = (cost.witnesses + cost.public_inputs + 1) as u128;
    let g1 = 2 * vars + cost.witnesses as u128 + domain as u128 - 1;
    let g2 = vars;
    let compressed = 48 * g1 + 96 * g2;
    let resident = size_of::<G1Affine>() as u128 * g1 + size_of::<G2Affine>() as u128 * g2;
    let report = format!(
        "Frontend: {counts:?}\nR1CS: {cost:?}\nQAP domain: {domain}\nDense key queries: {compressed} compressed bytes, {resident} resident bytes\nOne QAP field array: {} bytes\nCensus duration: {:?}\nCensus only; successful proving separately writes development-proof.bin and VERIFIED.\n",
        domain * 32,
        start.elapsed()
    );
    print!("{report}");
    fs::write(out.join("counts.txt"), report)?;
    if prove || preflight {
        if args[2] != "10" || cost.constraints > 300_000_000 {
            return Err("proving run is limited to the measured ten-query partition".into());
        }
        prove_shard(circuit, assignment, &out, preflight, streamed, cost)?;
    }
    Ok(())
}

#[cfg(feature = "groth16")]
fn prove_shard(
    circuit: multi_stark::plonkish::Circuit<multi_stark::types::Val>,
    assignment: multi_stark::plonkish::Assignment<multi_stark::types::Val>,
    out: &std::path::Path,
    preflight: bool,
    streamed: bool,
    cost: multi_stark::plonkish::r1cs::R1csStats,
) -> Result<(), Box<dyn std::error::Error>> {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_groth16::{Groth16, prepare_verifying_key};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use ark_std::rand::{SeedableRng, rngs::StdRng};
    use multi_stark::plonkish::{
        foreign::GoldilocksCircuit,
        r1cs::{R1csCircuit, streaming},
    };
    use p3_field::PrimeField64;
    use std::{fs, time::Instant};

    let expected: Vec<_> = assignment
        .public_values()
        .iter()
        .map(|v| Fr::from(v.as_canonical_u64()))
        .collect();
    let start = Instant::now();
    println!("Expanding foreign-field circuit for proving");
    let foreign = GoldilocksCircuit::new_with_expanded_hashes(&circuit);
    let mut witness = foreign.circuit.witness();
    foreign.assign(&assignment, &mut witness)?;
    let assigned = witness.generate()?;
    drop(assignment);
    drop(circuit);
    println!(
        "Foreign circuit and witness checked: {:?}; {:?}",
        start.elapsed(),
        foreign.circuit.stats()
    );
    if preflight {
        if streamed {
            println!("Starting streamed R1CS/QAP witness check");
            streaming::check(R1csCircuit::new(&foreign.circuit, Some(&assigned))?, cost)?;
            println!(
                "Streamed R1CS/QAP check passed: {} constraints",
                cost.constraints
            );
            return Ok(());
        }
        use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
        println!("Starting proving R1CS memory preflight");
        let cs = ConstraintSystem::<Fr>::new_ref();
        R1csCircuit::new(&foreign.circuit, Some(&assigned))?.generate_constraints(cs.clone())?;
        cs.finalize();
        assert!(cs.is_satisfied()?);
        println!(
            "Proving R1CS preflight passed: {} constraints",
            cs.num_constraints()
        );
        return Ok(());
    }
    let mut rng = StdRng::seed_from_u64(42); // Insecure development setup.
    let start = Instant::now();
    println!("Starting Groth16 development setup");
    let pk = if streamed {
        streaming::setup(R1csCircuit::new(&foreign.circuit, None)?, cost, &mut rng)?
    } else {
        Groth16::<Bls12_381>::generate_random_parameters_with_reduction(
            R1csCircuit::new(&foreign.circuit, None)?,
            &mut rng,
        )?
    };
    println!("Development setup finished: {:?}", start.elapsed());
    let mut vk = Vec::new();
    pk.vk.serialize_compressed(&mut vk)?;
    fs::write(out.join("development-vk.bin"), &vk)?;
    let start = Instant::now();
    println!("Starting Groth16 proof");
    let proof = if streamed {
        streaming::prove(
            R1csCircuit::new(&foreign.circuit, Some(&assigned))?,
            cost,
            &pk,
            &mut rng,
        )?
    } else {
        Groth16::<Bls12_381>::create_random_proof_with_reduction(
            R1csCircuit::new(&foreign.circuit, Some(&assigned))?,
            &pk,
            &mut rng,
        )?
    };
    let pvk = prepare_verifying_key(&pk.vk);
    assert!(Groth16::<Bls12_381>::verify_proof(&pvk, &proof, &expected)?);
    let mut wrong = expected.clone();
    wrong[0] += Fr::from(1u64);
    assert!(!Groth16::<Bls12_381>::verify_proof(&pvk, &proof, &wrong)?);
    let mut bytes = Vec::new();
    proof.serialize_compressed(&mut bytes)?;
    assert_eq!(bytes.len(), 192);
    let decoded = ark_groth16::Proof::<Bls12_381>::deserialize_compressed(bytes.as_slice())?;
    assert!(Groth16::<Bls12_381>::verify_proof(
        &pvk, &decoded, &expected
    )?);
    fs::write(out.join("development-proof.bin"), bytes)?;
    fs::write(
        out.join("VERIFIED"),
        "192-byte shard verified; wrong Init claim rejected; insecure development setup\n",
    )?;
    println!("Shard proof verified and saved: {:?}", start.elapsed());
    Ok(())
}

#[cfg(feature = "groth16")]
fn verify_bundle(
    out: &std::path::Path,
    public: &[multi_stark::types::Val],
) -> Result<(), Box<dyn std::error::Error>> {
    use ark_bls12_381::Bls12_381;
    use ark_groth16::{Proof, VerifyingKey, prepare_verifying_key};
    use ark_serialize::CanonicalDeserialize;
    use multi_stark::plonkish::verifier::{encode_query_bundle, verify_encoded_query_bundle};
    use std::fs;
    let context: [u8; 32] = fs::read(out.join("shard-0/context.bin"))?
        .try_into()
        .map_err(|bytes: Vec<u8>| format!("context length {}, expected 32", bytes.len()))?;
    let mut keys = Vec::new();
    let mut proofs = Vec::new();
    for shard in 0..11 {
        let dir = out.join(format!("shard-{shard}"));
        assert_eq!(fs::read(dir.join("context.bin"))?, context);
        keys.push(prepare_verifying_key(
            &VerifyingKey::<Bls12_381>::deserialize_compressed(
                fs::read(dir.join("development-vk.bin"))?.as_slice(),
            )?,
        ));
        proofs.push(Proof::<Bls12_381>::deserialize_compressed(
            fs::read(dir.join("development-proof.bin"))?.as_slice(),
        )?);
    }
    let bytes = encode_query_bundle(&keys, &proofs, public, &context)?;
    assert_eq!(bytes.len(), 2324);
    assert!(verify_encoded_query_bundle(&keys, public, &bytes)?);
    fs::write(out.join("development-bundle.bin"), bytes)?;
    fs::write(
        out.join("VERIFIED"),
        "11 Init shard proofs verified; insecure development setup\n",
    )?;
    println!("Complete Init bundle verified: 2324 bytes; insecure development setup");
    Ok(())
}
