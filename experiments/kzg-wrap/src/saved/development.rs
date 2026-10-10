use super::*;
use crate::native_verifier::development_metadata;
use multi_stark::{
    ark_adapter::{KzgPcs, Srs, pcs::KzgProverData},
    config::ProofConfig,
    lookup::LookupValues,
    prover::Stage1,
    system::ProverKey,
    traits::Pcs,
};
use p3_matrix::Matrix;
use std::{
    fs::File,
    io::{self, BufReader, Read},
    sync::Arc,
};

mod proving;
pub(crate) use proving::{run as prove, verify};

const SEED: &[u8] = b"init-fri-kzg-ordinary-v1";
const PUBLIC_DEGREE: usize = (1 << 28) - 2;
const INNER_HEIGHT: usize = 1 << 24;
const OUTER_HEIGHT: usize = 1 << 27;
const FORMAT: &str = "multi-stark/known-trapdoor-v4-diagnostic/v1";

fn check_mode() -> Result<()> {
    if std::env::var_os("MULTI_STARK_KZG_FILECOIN_CACHE").is_some()
        || std::env::var_os("MULTI_STARK_KZG_FILECOIN_DIGEST").is_some()
        || std::env::var("MULTI_STARK_KZG_SETUP").is_ok_and(|mode| mode != "development")
    {
        return Err("development diagnostic cannot select Filecoin parameters".into());
    }
    Ok(())
}

fn save_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn save<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    fs::write(
        path,
        bincode::serde::encode_to_vec(value, bincode::config::standard())?,
    )?;
    Ok(())
}

struct HashedReader<R> {
    inner: R,
    hash: blake3::Hasher,
    bytes: u64,
}

impl<R: Read> Read for HashedReader<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read(output)?;
        self.hash.update(&output[..count]);
        self.bytes += count as u64;
        Ok(count)
    }
}

fn restore_and_recommit(
    config: &KzgConfig,
    path: &Path,
    height: usize,
    width: usize,
    expected: KzgCommitment,
) -> Result<(KzgProverData, serde_json::Value)> {
    let started = Instant::now();
    let file = File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut reader = HashedReader {
        inner: BufReader::with_capacity(1 << 20, file),
        hash: blake3::Hasher::new(),
        bytes: 0,
    };
    let data = KzgProverData::read_checkpoint(&mut reader)?;
    if reader.bytes != file_bytes
        || data.matrices.len() != 1
        || data.matrices[0].domain() != config.pcs().natural_domain_for_degree(height)
        || data.matrices[0].width() != width
    {
        return Err("checkpoint dimensions differ from the saved circuit".into());
    }
    let (previous, data) = KzgProverData::concatenate([data]);
    if previous != expected {
        return Err("checkpoint commitment differs from the verified legacy proof".into());
    }
    let restore_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let (commitment, data) = config
        .pcs()
        .recommit(data)
        .map_err(|error| format!("recommit: {error:?}"))?;
    if commitment.0 != expected.0 || commitment.1.len() != 1 || !commitment.1[0].is_empty() {
        return Err(
            "regenerated coefficients differ from the verified development commitment".into(),
        );
    }
    let record = serde_json::json!({
        "path": path, "bytes": file_bytes,
        "blake3": reader.hash.finalize().to_hex().to_string(),
        "height": height, "width": width,
        "restore_seconds": restore_seconds,
        "recommit_seconds": started.elapsed().as_secs_f64(),
        "ordinary_commitment_matches_verified_legacy": true,
        "shifted_commitments": false,
    });
    eprintln!("DEVELOPMENT_RECOMMIT {record}");
    Ok((data, record))
}

fn part(commitment: &KzgCommitment, index: usize) -> KzgCommitment {
    KzgCommitment(
        vec![commitment.0[index].clone()],
        vec![commitment.1[index].clone()],
    )
}

pub(crate) fn bootstrap(stage: &Path, cache: &Path, output: &Path) -> Result<()> {
    check_mode()?;
    let started = Instant::now();
    let expected = init_claim::expected_words()?;
    fs::create_dir(output)?;
    fs::write(
        output.join("SECURITY.txt"),
        "Known-trapdoor development v4 diagnostic. Not Filecoin ceremony parameters. No outer proof or final-pipeline acceptance.\n",
    )?;
    let legacy_dir = output.join("legacy-input");
    fs::create_dir(&legacy_dir)?;
    let mut source_hashes = BTreeMap::new();
    let manifest: Manifest = load(stage, "manifest.bin", &mut source_hashes)?;
    if manifest.heights.len() != 19 {
        return Err("diagnostic requires the preserved nineteen-circuit Init layout".into());
    }
    for name in ["manifest.bin", "plan-id.bin", "inner-vk.bin"] {
        let bytes = read_recorded(stage, name, &mut source_hashes)?;
        fs::write(legacy_dir.join(name), &bytes)?;
    }
    for name in std::iter::once("proof.compact.bin".to_owned())
        .chain((0..19).map(|i| format!("setup-{i}.bin")))
    {
        let bytes = read_recorded(&stage.join("kzg"), &name, &mut source_hashes)?;
        fs::write(legacy_dir.join(&name), bytes)?;
    }
    let marker = stage.join(crate::outer::setup::BINDING_FILE);
    if marker.try_exists()? {
        let bytes = read_recorded(stage, crate::outer::setup::BINDING_FILE, &mut source_hashes)?;
        fs::write(legacy_dir.join(crate::outer::setup::BINDING_FILE), bytes)?;
    }
    let legacy = Input::load(
        &legacy_dir,
        &crate::outer::setup::SetupSource::Development {
            cache: Some(cache.to_owned()),
        },
        &expected,
    )?;
    let audit =
        development_metadata::audit(&legacy.system.circuits, &manifest.widths, &manifest.heights)?;
    let legacy_load_seconds = started.elapsed().as_secs_f64();
    let legacy_fixed = legacy.system.preprocessed_commit.as_ref().unwrap().clone();
    let legacy_main = legacy.proof.commitments.stage_1_trace.clone();
    let logs = legacy.logs;
    let claims = legacy.claims;
    let circuits = legacy.system.circuits;
    drop(legacy.system.config);

    let srs_started = Instant::now();
    let srs =
        Srs::unsafe_dev_public_setup_with_cache(INNER_HEIGHT, PUBLIC_DEGREE, SEED, Some(cache))?;
    let setup = srs.public_setup().ok_or("missing development v4 policy")?;
    let config = KzgConfig::with_max_trace_len(Arc::new(srs), INNER_HEIGHT, 2)
        .with_streaming_lookups()
        .with_partition_pipeline(32 << 30);
    let srs_seconds = srs_started.elapsed().as_secs_f64();
    let mut fixed_parts = Vec::new();
    let mut main_parts = Vec::new();
    let mut checkpoint_records = Vec::new();
    for i in 0..circuits.len() {
        for (kind, width, expected_commitment, parts) in [
            (
                "fixed",
                circuits[i].preprocessed_width,
                part(&legacy_fixed, i),
                &mut fixed_parts,
            ),
            (
                "main",
                circuits[i].main_width,
                part(&legacy_main, i),
                &mut main_parts,
            ),
        ] {
            let (data, record) = restore_and_recommit(
                &config,
                &stage.join("kzg").join(format!("{kind}-{i}.bin")),
                manifest.heights[i],
                width,
                expected_commitment,
            )?;
            parts.push(data);
            checkpoint_records.push(record);
        }
    }
    let (fixed_commitment, fixed_data) = KzgProverData::concatenate(fixed_parts);
    let (main_commitment, main_data) = KzgProverData::concatenate(main_parts);
    let system = System {
        config: config.clone(),
        circuits,
        preprocessed_commit: Some(fixed_commitment),
        preprocessed_indices: (0..19).map(Some).collect(),
    };
    let key = ProverKey {
        preprocessed_data: Some(fixed_data),
    };
    let lookups = system
        .circuits
        .iter()
        .zip(&manifest.heights)
        .map(|(circuit, &height)| {
            LookupValues::shape_only(
                height,
                &circuit
                    .graph
                    .lookups
                    .iter()
                    .map(|lookup| lookup.args.len())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let stage_one = Stage1 {
        active: vec![true; 19],
        active_indices: (0..19).collect(),
        log_degrees: logs.iter().map(|&log| usize::from(log)).collect(),
        stage_1_trace_commit: main_commitment,
        stage_1_trace_data: main_data,
        lookups,
    };
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let proof_started = Instant::now();
    let proof = system.prove_committed(&key, &refs, stage_one);
    let proving_seconds = proof_started.elapsed().as_secs_f64();
    system
        .verify_multiple_claims(&refs, &proof)
        .map_err(|error| format!("v4 verification: {error:?}"))?;
    let codec = FixedProofCodec::new(&system, &logs)?;
    let bytes = codec.encode(&proof)?;
    let decoded = codec.decode(&bytes)?;
    system
        .verify_multiple_claims(&refs, &decoded)
        .map_err(|error| format!("decoded v4 verification: {error:?}"))?;
    for i in 1..=18 {
        let mut changed = claims.clone();
        changed[i][3] += Scalar::ONE;
        assert!(
            system
                .verify_multiple_claims(
                    &changed.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &decoded
                )
                .is_err()
        );
    }
    assert!(codec.decode(&bytes[..bytes.len() - 1]).is_err());
    let mut extended = bytes.clone();
    extended.push(0);
    assert!(codec.decode(&extended).is_err());
    let mut damaged = bytes.clone();
    *damaged.last_mut().unwrap() ^= 1;
    assert!(match codec.decode(&damaged) {
        Err(_) => true,
        Ok(proof) => system.verify_multiple_claims(&refs, &proof).is_err(),
    });
    let plan = Input::metadata_plan(&system, &logs, &claims, OUTER_HEIGHT)?;
    plan.validate_request(&decoded, &claims)?;
    plan.validate_compact_len(bytes.len())?;
    for i in 0..system.circuits.len() {
        save(
            &output.join(format!("setup-{i}.bin")),
            &(
                &system.circuits[i],
                part(system.preprocessed_commit.as_ref().unwrap(), i),
            ),
        )?;
    }
    fs::copy(legacy_dir.join("manifest.bin"), output.join("manifest.bin"))?;
    fs::write(output.join("proof.compact.bin"), &bytes)?;
    let marker = serde_json::json!({
        "format": FORMAT, "known_trapdoor": true, "filecoin_acceptance": false,
        "setup_id": blake3::Hash::from_bytes(setup.id).to_hex().to_string(),
        "public_max_degree": setup.max_degree, "max_trace_len": INNER_HEIGHT,
        "recursive_max_trace_len": OUTER_HEIGHT,
        "proof_blake3": blake3::hash(&bytes).to_hex().to_string(),
    });
    save_json(&output.join("development-v4.json"), &marker)?;
    drop(key);
    drop(system);
    drop(config);
    let release = KzgPcs::release_idle_device_memory();
    let report = serde_json::json!({
        "scope": "Fresh known-trapdoor v4 proof from preserved first-stage coefficients; bootstrap only",
        "parameters": marker, "source": stage, "source_metadata_blake3": source_hashes,
        "source_coefficients": checkpoint_records, "metadata_audit": audit,
        "legacy_load_and_verify_seconds": legacy_load_seconds, "public_srs_load_seconds": srs_seconds,
        "v4_proving_seconds": proving_seconds, "wall_seconds": started.elapsed().as_secs_f64(),
        "proof_bytes": bytes.len(), "native_verification_passed": true,
        "altered_claims_rejected": 18, "encoding_negative_checks_passed": true,
        "gpu_release": release, "outer_proof_generated": false, "full_pipeline_run": false,
    });
    save_json(&output.join("bootstrap-report.json"), &report)?;
    println!("DEVELOPMENT_V4_BOOTSTRAP {report}");
    Ok(())
}

fn expected_claims(claims: &[Vec<Scalar>], count: usize, expected: &[u64; 18]) -> bool {
    let ns = Scalar::from_u8(93);
    let mut required: Vec<_> = std::iter::once(0)
        .chain(*expected)
        .enumerate()
        .map(|(i, word)| {
            vec![
                ns,
                Scalar::ONE,
                Scalar::from_usize(i),
                Scalar::from_u64(word),
            ]
        })
        .collect();
    required.extend((0..count - 10).map(|i| vec![ns, Scalar::from_u8(3), Scalar::from_usize(i)]));
    required.extend((0..6).map(|i| {
        vec![
            ns,
            Scalar::from_u8(4),
            Scalar::from_u8(3),
            Scalar::from_usize(i),
        ]
    }));
    required == claims
}

fn development_input(dir: &Path, expected: &[u64; 18]) -> Result<Input> {
    let started = Instant::now();
    let mut hashes = BTreeMap::new();
    let marker: serde_json::Value =
        serde_json::from_slice(&read_recorded(dir, "development-v4.json", &mut hashes)?)?;
    let srs = Srs::unsafe_dev_public_setup_with_cache(2, PUBLIC_DEGREE, SEED, None)?;
    let setup = srs.public_setup().unwrap();
    if marker["format"] != FORMAT
        || marker["known_trapdoor"] != true
        || marker["filecoin_acceptance"] != false
        || marker["setup_id"] != blake3::Hash::from_bytes(setup.id).to_hex().to_string()
        || marker["public_max_degree"] != PUBLIC_DEGREE
        || marker["max_trace_len"] != INNER_HEIGHT
        || marker["recursive_max_trace_len"] != OUTER_HEIGHT
    {
        return Err("not a compatible known-trapdoor development v4 fixture".into());
    }
    let manifest: Manifest = load(dir, "manifest.bin", &mut hashes)?;
    if manifest.heights.len() != 19
        || manifest.widths.len() != 19
        || !expected_claims(&manifest.claims, 19, expected)
    {
        return Err("development fixture differs from the expected Init statement".into());
    }
    let mut circuits = Vec::new();
    let mut fixed = KzgCommitment(vec![], vec![]);
    for i in 0..19 {
        let (circuit, mut commitment): (Circuit<Scalar>, KzgCommitment) =
            load(dir, &format!("setup-{i}.bin"), &mut hashes)?;
        if commitment.0.len() != 1 || commitment.1.len() != 1 || !commitment.1[0].is_empty() {
            return Err("development fixture has a non-v4 fixed commitment".into());
        }
        circuits.push(circuit);
        fixed.0.append(&mut commitment.0);
        fixed.1.append(&mut commitment.1);
    }
    development_metadata::audit(&circuits, &manifest.widths, &manifest.heights)?;
    let system = System {
        config: KzgConfig::with_max_trace_len(Arc::new(srs), INNER_HEIGHT, 2),
        circuits,
        preprocessed_commit: Some(fixed),
        preprocessed_indices: (0..19).map(Some).collect(),
    };
    let logs: Vec<_> = manifest
        .heights
        .iter()
        .map(|height| height.ilog2() as u8)
        .collect();
    let bytes = read_recorded(dir, "proof.compact.bin", &mut hashes)?;
    if marker["proof_blake3"] != blake3::hash(&bytes).to_hex().to_string() {
        return Err("development fixture proof checksum differs".into());
    }
    let plan = Input::metadata_plan(&system, &logs, &manifest.claims, OUTER_HEIGHT)?;
    plan.validate_compact_len(bytes.len())?;
    let proof = FixedProofCodec::new(&system, &logs)?.decode(&bytes)?;
    plan.validate_request(&proof, &manifest.claims)?;
    system
        .verify_multiple_claims(
            &manifest
                .claims
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>(),
            &proof,
        )
        .map_err(|error| format!("development fixture verification: {error:?}"))?;
    Ok(Input {
        system,
        proof,
        claims: manifest.claims,
        logs,
        proof_bytes: bytes,
        input_blake3: hashes,
        setup_identity: setup.id,
        load_seconds: started.elapsed().as_secs_f64(),
    })
}

fn scalar_digest(values: &[Scalar]) -> String {
    let mut hash = blake3::Hasher::new();
    multi_stark::ark_adapter::encoding::write_scalars(&mut hash, values)
        .expect("hash writer is infallible");
    hash.finalize().to_hex().to_string()
}

pub(crate) fn frontend(input_dir: &Path, output: &Path) -> Result<()> {
    check_mode()?;
    fs::create_dir(output)?;
    let started = Instant::now();
    let expected = init_claim::expected_words()?;
    let input = development_input(input_dir, &expected)?;
    let compile_started = Instant::now();
    let plan = input.plan(OUTER_HEIGHT)?;
    let identity = plan.identity();
    let compiled = plan.compile()?;
    let compile_seconds = compile_started.elapsed().as_secs_f64();
    let stats = compiled.circuit.stats();
    let height = compiled.circuit.multi_stark_layout()?.main_height;
    let lower_started = Instant::now();
    let circuit = compiled
        .circuit
        .lower_to_multi_stark_sharded(Scalar::from_u8(94), height)?
        .merge_table_traces(height)?;
    let lowering_seconds = lower_started.elapsed().as_secs_f64();
    if circuit.num_circuits() != 2 || circuit.main_heights() != [height] || height > OUTER_HEIGHT {
        return Err("recursive development layout exceeds one computation and merged table".into());
    }
    let mut reference = None::<multi_stark::plonkish::Assignment<Scalar>>;
    let mut assignment_digest = String::new();
    let mut samples = Vec::new();
    let mut trace_digests = Vec::new();
    for sample in 0..2 {
        let assignment_started = Instant::now();
        let (checks, assignment) = compiled.bindings.check_and_assign(
            identity,
            &input.proof,
            &input.claims,
            circuit.witness(),
        )?;
        let assignment_seconds = assignment_started.elapsed().as_secs_f64();
        let comparison_started = Instant::now();
        if let Some(previous) = &reference {
            assert!(assignment.values() == previous.values());
        } else {
            assignment_digest = scalar_digest(assignment.values());
        }
        let assignment_comparison_seconds = comparison_started.elapsed().as_secs_f64();
        let shards = circuit.trace_shards(&assignment)?;
        let mut traces = Vec::new();
        for index in 0..2 {
            let trace_started = Instant::now();
            let trace = shards.trace(index)?;
            let seconds = trace_started.elapsed().as_secs_f64();
            let checksum_started = Instant::now();
            let digest = scalar_digest(&trace.values);
            let checksum_seconds = checksum_started.elapsed().as_secs_f64();
            if sample == 0 {
                trace_digests.push(digest.clone());
            } else {
                assert_eq!(digest, trace_digests[index]);
            }
            traces.push(serde_json::json!({"index": index, "height": trace.height(), "width": trace.width(), "seconds": seconds, "checksum_seconds": checksum_seconds, "blake3": digest}));
        }
        let record = serde_json::json!({"sample": sample, "fresh_assignment_seconds": assignment_seconds,
            "assignment_comparison_seconds": assignment_comparison_seconds,
            "checks": checks, "traces": traces, "reference_sample": sample == 0,
            "complete_assignment_equal": (sample > 0).then_some(true),
            "trace_checksums_equal": (sample > 0).then_some(true)});
        eprintln!("DEVELOPMENT_RECURSIVE_SAMPLE {record}");
        samples.push(record);
        drop(shards);
        if reference.is_none() {
            reference = Some(assignment);
        }
    }
    let report = serde_json::json!({
        "scope": "Genuine known-trapdoor v4 inner proof; recursive frontend only, no outer SRS or proof",
        "known_trapdoor": true, "filecoin_acceptance": false, "outer_proof_generated": false,
        "full_pipeline_run": false, "input": input_dir, "input_blake3": input.input_blake3,
        "input_proof_bytes": input.proof_bytes.len(), "setup_id": blake3::Hash::from_bytes(input.setup_identity).to_hex().to_string(),
        "frontend_identity": blake3::Hash::from_bytes(identity).to_hex().to_string(),
        "load_and_native_verify_seconds": input.load_seconds, "compile_seconds": compile_seconds,
        "lowering_seconds": lowering_seconds, "wall_seconds": started.elapsed().as_secs_f64(),
        "values": stats.values, "gates": stats.gates, "lookups": stats.lookups, "publics": stats.publics,
        "rows": stats.gates + stats.lookups + stats.publics + 1, "main_heights": circuit.main_heights(),
        "circuit_count": circuit.num_circuits(), "assignment_blake3": assignment_digest, "samples": samples,
        "repeated_assignment_and_trace_parity": true,
        "reference_storage_bytes": stats.values * size_of::<Scalar>(),
        "operation_timing_excludes": ["first-stage bootstrap", "assignment comparisons", "canonical checksums", "assignment and trace destruction"],
    });
    save_json(&output.join("frontend-report.json"), &report)?;
    println!("DEVELOPMENT_V4_FRONTEND {report}");
    Ok(())
}
