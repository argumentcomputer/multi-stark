//! Check/prove verification of an exported Ix root with its unchanged 100-query profile.
#[cfg(feature = "groth16")]
#[path = "../experiments/init-kzg/src/support/ix_vk.rs"]
mod ix_vk;
#[cfg(feature = "groth16")]
use multi_stark::{
    batch::BatchProof,
    plonkish::verifier::{
        Envelope, ImplementationOptions, Message, ProofEnvelope, ProofProfile, Statement,
        StatementSlot, VerifierKey, VerifierLimits, VerifierPlan,
    },
    system::System,
    types::{GoldilocksBlake3Config, Val},
};
#[cfg(feature = "groth16")]
use p3_blake3::Blake3;
#[cfg(feature = "groth16")]
use p3_field::{PrimeCharacteristicRing, PrimeField64};
#[cfg(feature = "groth16")]
use p3_symmetric::CryptographicHasher;
#[cfg(feature = "groth16")]
use std::{fs, path::PathBuf, time::Instant};

#[cfg(feature = "groth16")]
fn claims_from_bytes(bytes: &[u8]) -> Result<Vec<Vec<Val>>, String> {
    let (words, remainder) = bytes.as_chunks::<8>();
    if !remainder.is_empty() {
        return Err("unaligned claims".into());
    }
    let mut words = words.iter();
    let n = u64::from_le_bytes(*words.next().ok_or("missing claims count")?);
    let mut claims = Vec::new();
    for _ in 0..n {
        let len = u64::from_le_bytes(*words.next().ok_or("missing claim length")?);
        let mut claim = Vec::new();
        for _ in 0..len {
            let v = u64::from_le_bytes(*words.next().ok_or("missing claim value")?);
            if v >= Val::ORDER_U64 {
                return Err("noncanonical claim".into());
            }
            claim.push(Val::from_u64(v));
        }
        claims.push(claim);
    }
    if words.next().is_some() {
        return Err("trailing claims".into());
    }
    Ok(claims)
}

#[cfg(feature = "groth16")]
fn check_aiur_policy(
    system: &System<GoldilocksBlake3Config>,
    proof: &BatchProof<GoldilocksBlake3Config>,
    claims: &[Vec<Val>],
) -> Result<(), String> {
    if proof.proofs.len() != 1
        || proof.preamble.headers.len() != 1
        || claims.len() != 1
        || claims[0].len() != 18
        || claims[0][0] != Val::ZERO
    {
        return Err("expected the Ix aggregate's one-shard, one 18-word claim profile".into());
    }
    let h = &proof.preamble.headers[0];
    if h.claims != claims {
        return Err("proof claim differs from independently exported expected claim".into());
    }
    let mut logs = h.log_degrees.iter();
    let mut rows = 0u128;
    let mut lookups = 1u128;
    for (c, &active) in system.circuits.iter().zip(&h.active) {
        if active {
            let log = *logs.next().ok_or("missing height")?;
            let n = 1u128.checked_shl(u32::from(log)).ok_or("height overflow")?;
            rows = rows.checked_add(n).ok_or("row overflow")?;
            lookups = lookups
                .checked_add(
                    n.checked_mul(c.num_lookups as u128)
                        .ok_or("lookup overflow")?,
                )
                .ok_or("lookup overflow")?;
        }
    }
    if logs.next().is_some()
        || rows >= u128::from(Val::ORDER_U64)
        || lookups >= u128::from(Val::ORDER_U64)
    {
        return Err("Aiur trace/lookup bound".into());
    }
    let mut previous = (0, 0);
    let (pairs, remainder) = proof.preamble.messages.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err("unpaired memory closures".into());
    }
    for pair in pairs {
        let (pull, push) = (&pair[0], &pair[1]);
        if pull.multiplicity != Val::NEG_ONE
            || push.multiplicity != Val::ONE
            || pull.args.len() != 3
            || push.args.len() != 3
            || pull.args[0] != Val::from_u8(16)
            || push.args[0] != Val::from_u8(16)
            || pull.args[1] != push.args[1]
        {
            return Err("invalid memory closure shape".into());
        }
        let width = pull.args[1].as_canonical_u64();
        let start = pull.args[2].as_canonical_u64();
        let end = push.args[2].as_canonical_u64();
        if start > end || !(previous.0 < width || (previous.0 == width && previous.1 <= start)) {
            return Err("memory closures not sorted and disjoint".into());
        }
        previous = (width, end);
    }
    Ok(())
}

#[cfg(feature = "groth16")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err(
            "usage: init_groth16 <ix-artifacts> <output-dir> [--estimate | --r1cs-estimate | --prove | --shard-estimate <group-size> <index>]".into(),
        );
    }
    let dir = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    let mode = args.get(2).map_or("--estimate", String::as_str);
    if (mode == "--shard-estimate" && args.len() != 5)
        || (mode != "--shard-estimate" && args.len() > 3)
        || !matches!(
            mode,
            "--estimate" | "--r1cs-estimate" | "--prove" | "--shard-estimate"
        )
    {
        return Err("invalid mode".into());
    }
    fs::create_dir_all(&out)?;
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_target(false)
        .with_max_level(tracing_subscriber::filter::LevelFilter::INFO)
        .init();
    let vk_bytes = fs::read(dir.join("root-vk.bin"))?;
    let (system, cp, fp) = ix_vk::from_bytes(&vk_bytes)?;
    let bytes = fs::read(dir.join("root-proof.bin"))?;
    let proof = BatchProof::<GoldilocksBlake3Config>::from_bytes(&bytes)?;
    if proof.to_bytes()? != bytes {
        return Err("noncanonical proof transport".into());
    }
    let claims = claims_from_bytes(&fs::read(dir.join("root-claims.bin"))?)?;
    check_aiur_policy(&system, &proof, &claims)?;
    let start = Instant::now();
    system
        .verify_batch(&proof)
        .map_err(|e| format!("native verification: {e:?}"))?;
    println!(
        "Native Ix root batch verified in {:?}: {} bytes, {} circuits, {} shards",
        start.elapsed(),
        bytes.len(),
        system.circuits.len(),
        proof.proofs.len()
    );
    println!(
        "log_blowup={} cap={} final_log={} arity={} queries={} commit_pow={} query_pow={}",
        cp.log_blowup,
        cp.cap_height,
        fp.log_final_poly_len,
        fp.max_log_arity,
        fp.num_queries,
        fp.commit_proof_of_work_bits,
        fp.query_proof_of_work_bits
    );
    println!("Trusted VK BLAKE3: {:02x?}", Blake3.hash_iter(vk_bytes));
    println!("Expected public claims: {claims:?}");
    println!(
        "Fixed messages (validated Aiur memory-closure policy): {}",
        proof.preamble.messages.len()
    );
    let h = &proof.preamble.headers[0];
    let verifier_key = VerifierKey::from_system(&system);
    let plan = VerifierPlan::validate(
        &verifier_key,
        ProofProfile {
            envelope: Envelope::SingleBatch,
            active: h.active.clone(),
            log_degrees: h.log_degrees.clone(),
            claim_lengths: claims.iter().map(Vec::len).collect(),
            message_lengths: proof
                .preamble
                .messages
                .iter()
                .map(|m| m.args.len())
                .collect(),
            max_field_retries: 2,
        },
        VerifierLimits::default(),
    )?;
    let shape = plan.shape();
    println!(
        "Fixed shape: {} active circuits, max trace log {}, {} queries; batch messages are circuit constants",
        shape.log_degrees.len(),
        shape.log_trace,
        shape.queries
    );
    let schema = Statement {
        claims: claims
            .iter()
            .map(|c| vec![StatementSlot::Public; c.len()])
            .collect(),
        messages: proof
            .preamble
            .messages
            .iter()
            .map(|m| Message {
                args: m
                    .args
                    .iter()
                    .copied()
                    .map(StatementSlot::Constant)
                    .collect(),
                multiplicity: StatementSlot::Constant(m.multiplicity),
            })
            .collect(),
    };
    let options = ImplementationOptions {
        compact_blake3: mode != "--estimate",
    };
    let partition = if mode == "--shard-estimate" {
        Some(multi_stark::plonkish::verifier::QueryShardPlan::new(
            &plan,
            schema.clone(),
            options,
            args[3].parse()?,
        )?)
    } else {
        None
    };
    let shard: usize = if partition.is_some() {
        args[4].parse()?
    } else {
        0
    };
    let start = Instant::now();
    let counts = if let Some(p) = &partition {
        println!(
            "Query partition: {} shards; measuring shard {shard}, queries {:?}",
            p.shard_count(),
            p.query_range(shard)?
        );
        p.estimate(shard)?
    } else {
        plan.estimate(schema.clone(), options)?
    };
    println!(
        "Exact native frontend census: {counts:?}; {:?}",
        start.elapsed()
    );
    fs::write(
        out.join("frontend-counts.txt"),
        format!("{counts:?}\nNative frontend only; excludes foreign-field and R1CS expansion.\n"),
    )?;
    if mode == "--estimate" {
        return Ok(());
    }
    if mode == "--r1cs-estimate" || mode == "--shard-estimate" {
        if counts.values > 200_000_000 {
            return Err("compact frontend exceeds the census memory budget".into());
        }
        let start = Instant::now();
        let (circuit, _) = if let Some(p) = &partition {
            p.build(shard)?
        } else {
            plan.build(schema, options)?
        };
        assert_eq!(counts, circuit.stats());
        println!(
            "Compact frontend built: {:?}; {:?}",
            circuit.stats(),
            start.elapsed()
        );
        let start = Instant::now();
        let cost = multi_stark::plonkish::r1cs::estimate_goldilocks(&circuit)?;
        use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
        let size = cost
            .constraints
            .checked_add(cost.public_inputs + 1)
            .ok_or("domain overflow")?;
        let domain = GeneralEvaluationDomain::<Fr>::new(size);
        println!(
            "EXACT R1CS CENSUS: {cost:?}; {:?}; QAP domain={:?}",
            start.elapsed(),
            domain.map(|d| d.size())
        );
        if let Some(domain) = domain {
            use ark_bls12_381::{G1Affine, G2Affine};
            let variables = (cost.witnesses + cost.public_inputs + 1) as u128;
            let g1 = 2 * variables + cost.witnesses as u128 + domain.size() as u128 - 1;
            let g2 = variables;
            let compressed = 48 * g1 + 96 * g2;
            let resident = size_of::<G1Affine>() as u128 * g1 + size_of::<G2Affine>() as u128 * g2;
            println!(
                "Dense proving-key queries: {compressed} compressed bytes, {resident} resident bytes; excludes R1CS, FFT workspace and witnesses (G1={} bytes, G2={} bytes)",
                size_of::<G1Affine>(),
                size_of::<G2Affine>()
            );
        }
        fs::write(
            out.join("r1cs-counts.txt"),
            format!("{cost:?}\nQAP domain={:?}\n", domain.map(|d| d.size())),
        )?;
        return Ok(());
    }
    // This prototype holds the frontend, assignment, and R1CS in memory.
    if counts.values > 10_000_000 {
        return Err("frontend exceeds this in-memory prototype's 10-million-value limit; use the census to plan a streamed lowering".into());
    }
    let (circuit, inputs) = plan.build(schema, options)?;
    let expanded = plan.expand_witness(ProofEnvelope::SingleBatch(&proof))?;
    let mut w = circuit.witness();
    inputs.assign_statement(
        &mut w,
        &Statement {
            claims: claims.clone(),
            messages: proof
                .preamble
                .messages
                .iter()
                .map(|m| Message {
                    args: m.args.clone(),
                    multiplicity: m.multiplicity,
                })
                .collect(),
        },
    )?;
    expanded.assign_proof(&mut w, &inputs)?;
    let a = w.generate()?;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_groth16::{Groth16, prepare_verifying_key};
    use ark_serialize::CanonicalSerialize;
    use ark_std::rand::{SeedableRng, rngs::StdRng};
    use multi_stark::plonkish::{foreign::GoldilocksCircuit, r1cs::R1csCircuit};
    let foreign = GoldilocksCircuit::new_with_expanded_hashes(&circuit);
    let mut w = foreign.circuit.witness();
    foreign.assign(&a, &mut w)?;
    let a = w.generate()?;
    let mut rng = StdRng::seed_from_u64(42); // Development setup only.
    let pk = Groth16::<Bls12_381>::generate_random_parameters_with_reduction(
        R1csCircuit::new(&foreign.circuit, None)?,
        &mut rng,
    )?;
    let outer = Groth16::<Bls12_381>::create_random_proof_with_reduction(
        R1csCircuit::new(&foreign.circuit, Some(&a))?,
        &pk,
        &mut rng,
    )?;
    let expected: Vec<_> = claims
        .iter()
        .flatten()
        .map(|v| Fr::from(v.as_canonical_u64()))
        .collect();
    let vk = prepare_verifying_key(&pk.vk);
    assert!(Groth16::<Bls12_381>::verify_proof(&vk, &outer, &expected)?);
    let mut wrong = expected;
    wrong[0] += Fr::from(1u64);
    assert!(!Groth16::<Bls12_381>::verify_proof(&vk, &outer, &wrong)?);
    let mut bytes = vec![];
    outer.serialize_compressed(&mut bytes)?;
    fs::write(out.join("development-proof.bin"), &bytes)?;
    let mut key = vec![];
    pk.vk.serialize_compressed(&mut key)?;
    fs::write(out.join("development-vk.bin"), key)?;
    println!(
        "Init proof verification wrapped in {} bytes; altered claim rejected; insecure development setup",
        bytes.len()
    );
    Ok(())
}

#[cfg(feature = "groth16")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
#[cfg(not(feature = "groth16"))]
fn main() {
    eprintln!("enable --features groth16,parallel");
}
