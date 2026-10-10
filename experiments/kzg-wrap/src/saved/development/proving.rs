use super::*;
use crate::outer::{Frontend, parameters::Parameters};

fn artifacts(dir: &Path) -> Result<[Vec<u8>; 3]> {
    Ok([
        fs::read(dir.join("kzg/proof.compact.bin"))?,
        fs::read(dir.join("kzg/packet.bin"))?,
        fs::read(dir.join("kzg/profile-id.bin"))?,
    ])
}

pub(crate) fn run(input_dir: &Path, cache: &Path, output: &Path) -> Result<()> {
    check_mode()?;
    if !multi_stark::BuildCapabilities::compiled().kzg_cuda
        || std::env::var("MULTI_STARK_KZG_BACKEND").as_deref() != Ok("cuda")
    {
        return Err(
            "recursive proving diagnostic requires a CUDA build and explicit CUDA backend".into(),
        );
    }
    let expected = init_claim::expected_words()?;
    fs::create_dir(output)?;
    fs::write(
        output.join("SECURITY.txt"),
        "Known-trapdoor development v4 recursive proving diagnostic. Not Filecoin ceremony parameters or final-pipeline acceptance.\n",
    )?;
    let started = Instant::now();
    let input = development_input(input_dir, &expected)?;
    let compile_started = Instant::now();
    let plan = input.plan(OUTER_HEIGHT)?;
    let identity = plan.identity();
    let compiled = plan.compile()?;
    let compile_seconds = compile_started.elapsed().as_secs_f64();
    let stats = compiled.circuit.stats();
    let lower_started = Instant::now();
    let mut frontend = Frontend::new(
        compiled.circuit,
        compiled.bindings.pairing_keys(),
        compiled.bindings.degree_output_count(),
        &identity,
        OUTER_HEIGHT,
    )?;
    let lowering_seconds = lower_started.elapsed().as_secs_f64();
    let parameters = Parameters::KnownTrapdoorPublicDegree {
        seed: SEED,
        cache: Some(cache),
    };
    let mut reference = None;
    let mut samples = vec![];
    for sample in 0..2 {
        let dir = output.join(format!("request-{sample}"));
        let request_started = Instant::now();
        let request_started_unix_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs_f64();
        let loaded_key_reused = frontend.has_loaded_key();
        assert_eq!(loaded_key_reused, sample > 0);
        eprintln!(
            "DEVELOPMENT_RECURSIVE_PROVING_REQUEST_STARTED sample={sample} loaded_key_reused={loaded_key_reused} unix_seconds={request_started_unix_seconds:.6}"
        );
        let assignment_started = Instant::now();
        let (checks, assignment) = compiled.bindings.check_and_assign(
            identity,
            &input.proof,
            &input.claims,
            frontend.witness(),
        )?;
        let assignment_seconds = assignment_started.elapsed().as_secs_f64();
        let proving_started = Instant::now();
        frontend.stage_with_parameters(&assignment, &dir, true, &parameters, &expected)?;
        let stage_and_prove_seconds = proving_started.elapsed().as_secs_f64();
        drop(assignment);
        let release_started = Instant::now();
        let gpu_release = frontend.release_device_memory();
        let release_seconds = release_started.elapsed().as_secs_f64();
        let request_seconds = request_started.elapsed().as_secs_f64();
        let request_finished_unix_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs_f64();
        if !gpu_release.quiesced
            || gpu_release.devices.len() != 4
            || gpu_release.devices.iter().any(|device| {
                device.after.resident_coefficient_bytes != 0
                    || device.after.srs_point_bytes != 0
                    || device.after.msm_workspace_bytes != 0
            })
        {
            return Err("recursive diagnostic requires quiescing all four CUDA devices".into());
        }
        let actual = artifacts(&dir)?;
        if let Some(previous) = &reference {
            assert_eq!(&actual, previous);
        }
        let proof_report: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("kzg/prove-report.json"))?)?;
        if proof_report["known_trapdoor"] != true
            || proof_report["filecoin_acceptance"] != false
            || proof_report["public_max_degree"] != PUBLIC_DEGREE
            || proof_report["public_setup_id"]
                != blake3::Hash::from_bytes(input.setup_identity)
                    .to_hex()
                    .to_string()
            || proof_report["trace_heights"] != serde_json::json!([OUTER_HEIGHT, 1 << 17])
            || proof_report["loaded_key_reused"] != loaded_key_reused
            || actual[1].len() >= 3_000
        {
            return Err("recursive diagnostic differs from the development-v4 profile".into());
        }
        let record = serde_json::json!({
            "sample": sample, "loaded_key_reused": loaded_key_reused,
            "fresh_assignment_seconds": assignment_seconds,
            "stage_and_prove_seconds": stage_and_prove_seconds,
            "gpu_release_seconds": release_seconds, "request_seconds": request_seconds,
            "request_started_unix_seconds": request_started_unix_seconds,
            "request_finished_unix_seconds": request_finished_unix_seconds,
            "checks": checks, "proof_report": proof_report, "gpu_release": gpu_release,
            "proof_blake3": blake3::hash(&actual[0]).to_hex().to_string(),
            "packet_blake3": blake3::hash(&actual[1]).to_hex().to_string(),
            "profile_blake3": blake3::hash(&actual[2]).to_hex().to_string(),
            "reference_sample": sample == 0,
            "proof_packet_profile_equal": (sample > 0).then_some(true),
        });
        eprintln!("DEVELOPMENT_RECURSIVE_PROVING_SAMPLE {record}");
        save_json(
            &output.join(format!("request-{sample}-report.json")),
            &record,
        )?;
        samples.push(record);
        if reference.is_none() {
            reference = Some(actual);
        }
    }
    let report = serde_json::json!({
        "scope": "Genuine known-trapdoor v4 inner and outer proofs; one cold preparation request and one fresh retained request",
        "known_trapdoor": true, "filecoin_acceptance": false, "outer_proof_generated": true,
        "full_pipeline_run": false, "input": input_dir, "input_blake3": input.input_blake3,
        "input_proof_bytes": input.proof_bytes.len(),
        "setup_id": blake3::Hash::from_bytes(input.setup_identity).to_hex().to_string(),
        "frontend_identity": blake3::Hash::from_bytes(identity).to_hex().to_string(),
        "load_and_native_verify_seconds": input.load_seconds, "compile_seconds": compile_seconds,
        "lowering_seconds": lowering_seconds, "wall_seconds": started.elapsed().as_secs_f64(),
        "values": stats.values, "gates": stats.gates, "lookups": stats.lookups, "publics": stats.publics,
        "rows": stats.gates + stats.lookups + stats.publics + 1,
        "max_trace_len": OUTER_HEIGHT, "public_max_degree": PUBLIC_DEGREE,
        "samples": samples, "repeated_proof_packet_profile_parity": true,
        "request_timing_includes": ["fresh assignment and full checks", "fresh fused trace generation and proving", "native and negative verification", "assignment destruction", "device quiescence"],
        "request_timing_excludes": ["inner-proof bootstrap", "input loading", "compilation", "lowering", "artifact parity comparisons"],
        "cold_request_includes": ["outer SRS generation or cache loading", "fixed preprocessing load or generation and checkpointing"],
        "fixed_cache_directory": std::env::var_os("MULTI_STARK_KZG_FIXED_CACHE"),
        "retained_request_releases_device_residency_between_requests": true,
    });
    save_json(&output.join("recursive-proving-report.json"), &report)?;
    println!("DEVELOPMENT_V4_RECURSIVE_PROVING {report}");
    Ok(())
}

pub(crate) fn verify(dir: &Path) -> Result<()> {
    check_mode()?;
    let expected = init_claim::expected_words()?;
    crate::outer::verify_with_parameters(
        dir,
        &Parameters::KnownTrapdoorPublicDegree {
            seed: SEED,
            cache: None,
        },
        &expected,
    )
}
