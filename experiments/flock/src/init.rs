//! Lower and prove verification of the unchanged Init aggregate.
#[path = "../../../examples/support/init_claim.rs"]
mod init_claim;
#[path = "../../init-kzg/src/support/ix_vk.rs"]
mod ix_vk;
use crate::adapter::Program;
use multi_stark::{
    batch::BatchProof,
    plonkish::verifier::*,
    types::{GoldilocksBlake3Config, Val},
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::{fs, path::Path, time::Instant};

pub fn census(dir: &Path, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    run(dir, out, None)
}
pub fn run(
    dir: &Path,
    out: &Path,
    profile: Option<flock_prover::pcs::ligerito::LigeritoProfile>,
) -> Result<(), Box<dyn std::error::Error>> {
    run_inner(dir, out, profile, false, false, None)
}
pub fn run_inner(
    dir: &Path,
    out: &Path,
    profile: Option<flock_prover::pcs::ligerito::LigeritoProfile>,
    terminal: bool,
    coalesced: bool,
    initial_k: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(k) = initial_k {
        flock_prover::pcs::ligerito::select_initial_k(
            38,
            profile.ok_or("opening width requires a proving profile")?,
            k,
        )?;
    }
    let _ = flock_prover::init_perf_thread_pool();
    let start = Instant::now();
    let vk_bytes = fs::read(dir.join("root-vk.bin"))?;
    let (system, _, _) = ix_vk::from_bytes(&vk_bytes)?;
    let bytes = fs::read(dir.join("root-proof.bin"))?;
    let proof = BatchProof::<GoldilocksBlake3Config>::from_bytes(&bytes)?;
    if proof.to_bytes()? != bytes || proof.proofs.len() != 1 || proof.preamble.headers.len() != 1 {
        return Err("expected canonical one-shard Init aggregate".into());
    }
    let expected = init_claim::INIT_PUBLIC_WORDS.map(Val::from_u64);
    let h = &proof.preamble.headers[0];
    if h.claims != vec![expected.to_vec()] {
        return Err("Init public claim mismatch".into());
    }
    system
        .verify_batch(&proof)
        .map_err(|e| format!("native root verification: {e:?}"))?;
    // Preserve the aggregate consumer's trace/lookup bounds and memory closures.
    let mut logs = h.log_degrees.iter();
    let (mut rows, mut lookups) = (0u128, 1u128);
    for (c, &active) in system.circuits.iter().zip(&h.active) {
        if active {
            let n = 1u128
                .checked_shl(u32::from(*logs.next().ok_or("missing height")?))
                .ok_or("height overflow")?;
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
        return Err("aggregate field bound".into());
    }
    let (pairs, remainder) = proof.preamble.messages.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err("unpaired memory closures".into());
    }
    let mut previous = (0, 0);
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
            return Err("invalid memory closure".into());
        }
        let (width, start, end) = (
            pull.args[1].as_canonical_u64(),
            pull.args[2].as_canonical_u64(),
            push.args[2].as_canonical_u64(),
        );
        if start > end || !(previous.0 < width || (previous.0 == width && previous.1 <= start)) {
            return Err("unordered memory closures".into());
        }
        previous = (width, end);
    }
    let key = VerifierKey::from_system(&system);
    let plan = VerifierPlan::validate(
        &key,
        ProofProfile {
            envelope: Envelope::SingleBatch,
            active: h.active.clone(),
            log_degrees: h.log_degrees.clone(),
            claim_lengths: vec![18],
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
    let schema = Statement {
        claims: vec![vec![StatementSlot::Public; 18]],
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
    eprintln!("Unchanged Init verified; building compact verifier");
    let (circuit, inputs) = plan.build(
        schema,
        ImplementationOptions {
            compact_blake3: true,
        },
    )?;
    let frontend_seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "Frontend ready after {frontend_seconds:.1}s: {:?}",
        circuit.stats()
    );
    let start = Instant::now();
    let mut program = if profile.is_some() {
        Program::compile(&circuit)?
    } else {
        Program::census(&circuit)?
    };
    if coalesced {
        program.coalesce();
    }
    let mut report = serde_json::json!({"scope":"Exact Flock lowering for unchanged Init; proving not completed",
        "source_proof_bytes":bytes.len(),"source_proof_blake3":blake3::hash(&bytes).to_hex().to_string(),
        "source_vk_blake3":blake3::hash(&vk_bytes).to_hex().to_string(),"public_claim_words":init_claim::INIT_PUBLIC_WORDS,
        "queries":plan.shape().queries,"frontend":format!("{:?}",circuit.stats()),
        "frontend_seconds":frontend_seconds,"lowering_seconds":start.elapsed().as_secs_f64(),"lowering":program.stats(),
        "geometry":program.geometry(), "packed_geometry":program.packed_geometry(),
        "coalesced_geometry":program.coalesced_geometry()});
    fs::create_dir_all(out)?;
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    if let Some(profile) = profile {
        use flock_prover::{
            challenger::FsChallenger,
            field::F128,
            pcs::{PcsParams, ligerito::embedded_initial_k},
            union::UnionInstance,
            verifier::verify_ligerito_union_circuit,
        };
        const DOMAIN: &[u8] = b"multi-stark-flock-init-v1";
        let start = Instant::now();
        eprintln!("Lowering complete; generating the unchanged Init verifier witness");
        let statement = Statement {
            claims: h.claims.clone(),
            messages: proof
                .preamble
                .messages
                .iter()
                .map(|m| Message {
                    args: m.args.clone(),
                    multiplicity: m.multiplicity,
                })
                .collect(),
        };
        let prepared = plan.expand_witness(ProofEnvelope::SingleBatch(&proof))?;
        let mut witness = circuit.witness();
        inputs.assign_statement(&mut witness, &statement)?;
        prepared.assign_proof(&mut witness, &inputs)?;
        let assignment = witness.generate()?;
        eprintln!(
            "Source witness generated in {:.1}s; building packed Flock shape",
            start.elapsed().as_secs_f64()
        );
        let mut ready = program.build_packed(&assignment)?.into_ready();
        report["prepare_witness_and_shape_seconds"] = start.elapsed().as_secs_f64().into();
        drop(assignment);
        drop(program);
        drop(circuit);
        drop(prepared);
        drop(proof);
        let rows = ready.take_rows();
        let union = UnionInstance::new(&ready.registry, ready.counts.clone());
        let m = union.dense_m();
        if initial_k.is_some() && m != 38 {
            return Err(format!("experimental Init profile expects m38, got m{m}").into());
        }
        let config = flock_prover::pcs::ligerito::embedded_security_config(m, profile)
            .ok_or("missing strict PCS profile")?;
        fs::write(out.join("pcs-profile.toml"), config)?;
        report["pcs_config_blake3"] = blake3::hash(config.as_bytes()).to_hex().to_string().into();
        report["pcs_m"] = m.into();
        let batch = embedded_initial_k(m, profile).ok_or("missing strict PCS profile")?;
        report["initial_k"] = batch.into();
        let params = PcsParams {
            m,
            log_inv_rate: profile.log_inv_rate(),
            log_batch_size: batch,
            profile,
            num_lanes: union.commit_lanes(batch),
            merkle_hash: flock_prover::merkle::HashKind::Blake3,
        };
        params.ligerito_prover_config()?;
        report["profile"] = profile.as_str().into();
        report["counters_enabled"] = cfg!(feature = "counters").into();
        report["status"] = "proving".into();
        fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
        eprintln!(
            "Flock shape ready: m={m}, profile={}; starting full Init proving",
            profile.as_str()
        );
        let start = Instant::now();
        let challenger = || {
            if terminal {
                FsChallenger::with_chained_blake3(DOMAIN)
            } else {
                FsChallenger::new(DOMAIN)
            }
        };
        let (outer, commitment) = if terminal {
            ready.prove_with_challenger(rows, &params, &mut challenger())
        } else {
            ready.prove(rows, &params, DOMAIN)
        };
        report["prove_seconds"] = start.elapsed().as_secs_f64().into();
        let mut expected = ready.public.clone();
        let claim_start = expected
            .len()
            .checked_sub(18)
            .ok_or("missing public claim")?;
        for (v, word) in expected[claim_start..]
            .iter_mut()
            .zip(init_claim::INIT_PUBLIC_WORDS)
        {
            *v = F128::new(word, 0);
        }
        let circuits = ready.circuits();
        let start = Instant::now();
        verify_ligerito_union_circuit(
            &union,
            &ready.circuit,
            &expected,
            &circuits,
            &commitment,
            &outer,
            &params,
            &mut challenger(),
        )
        .map_err(|e| format!("full Init Flock verification: {e:?}"))?;
        report["verify_seconds"] = start.elapsed().as_secs_f64().into();
        let encoded = bincode::serialize(&(&commitment, &outer))?;
        let (decoded_commitment, decoded): (
            flock_prover::pcs::Commitment,
            flock_prover::proof::R1csProofCircuitMerged,
        ) = bincode::deserialize(&encoded)?;
        verify_ligerito_union_circuit(
            &union,
            &ready.circuit,
            &expected,
            &circuits,
            &decoded_commitment,
            &decoded,
            &params,
            &mut challenger(),
        )
        .map_err(|e| format!("serialized Init Flock verification: {e:?}"))?;
        expected[claim_start] += F128::ONE;
        assert!(
            verify_ligerito_union_circuit(
                &union,
                &ready.circuit,
                &expected,
                &circuits,
                &commitment,
                &outer,
                &params,
                &mut challenger()
            )
            .is_err()
        );
        fs::write(out.join("proof.bin"), &encoded)?;
        report["proof_and_commitment_bytes"] = encoded.len().into();
        report["proof_blake3"] = blake3::hash(&encoded).to_hex().to_string().into();
        report["status"] = "verified".into();
        report["serialized_roundtrip_verified"] = true.into();
        report["altered_claim_rejected"] = true.into();
        report["scope"] = "Full unchanged Init Flock proof; Groth16 wrapper and composed security validation outstanding".into();
        report["transcript"] = if terminal { "chained-blake3" } else { "blake3" }.into();
        fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
        #[cfg(feature = "terminal")]
        if terminal {
            drop(circuits);
            drop(union);
            crate::terminal::census(
                out,
                ready,
                &params,
                DOMAIN,
                &commitment,
                &outer,
                &init_claim::INIT_PUBLIC_WORDS,
            )?;
        }
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
