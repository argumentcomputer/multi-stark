//! Checkpointed KZG compression of the saved recursive Init FRI proof.
#[cfg(feature = "kzg")]
#[path = "support/init_fri.rs"]
mod init_fri;
#[cfg(feature = "kzg")]
#[allow(unreachable_pub)]
#[path = "../experiments/init-kzg/src/storage.rs"]
mod storage;

#[cfg(not(feature = "kzg"))]
fn main() {
    eprintln!("enable --features kzg,parallel");
}

#[cfg(feature = "kzg")]
fn main() -> storage::Result<()> {
    use multi_stark::{
        ark_adapter::Scalar,
        plonkish::{foreign::GoldilocksCircuit, verifier::*},
        traits::Field,
    };
    use p3_field::PrimeField64;
    use p3_matrix::Matrix;
    use std::{fs, path::PathBuf, time::Instant};
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 2 && (args[0] == "prove" || args[0] == "verify") {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
        return prove(std::path::Path::new(&args[1]), args[0] == "verify");
    }
    if args.len() != 3 || args[0] != "stage" {
        return Err("usage: init_fri_kzg_prove stage <saved-fri-dir> <output-dir>".into());
    }
    let out = PathBuf::from(&args[2]);
    fs::create_dir_all(&out)?;
    if out.join("manifest.bin").exists() {
        return Err("staging already complete".into());
    }
    let fixture = init_fri::load(&PathBuf::from(&args[1]))?;
    let plan = VerifierPlan::validate(&fixture.key, fixture.profile, VerifierLimits::default())?;
    fs::write(out.join("plan-id.bin"), plan.identity(&fixture.schema)?)?;
    fs::write(out.join("inner-vk.bin"), fixture.key.to_bytes()?)?;
    let start = Instant::now();
    let (source, inputs) = plan.build(
        fixture.schema,
        ImplementationOptions {
            compact_blake3: true,
        },
    )?;
    println!(
        "Source circuit: {:?}; {:?}",
        start.elapsed(),
        source.stats()
    );
    let expanded = plan.expand_witness(ProofEnvelope::Ordinary {
        proof: &fixture.proof,
        claims: &fixture.claims,
    })?;
    let mut witness = source.witness();
    inputs.assign_statement(
        &mut witness,
        &Statement {
            claims: fixture.claims,
            messages: vec![],
        },
    )?;
    expanded.assign_proof(&mut witness, &inputs)?;
    let assignment = witness.generate()?;
    assert_eq!(assignment.public_values(), fixture.public);
    drop(expanded);
    drop(inputs);
    println!("Source witness satisfied: {:?}", start.elapsed());
    let foreign = GoldilocksCircuit::new_preallocated(&source);
    println!(
        "Scalar translation: {:?}; {:?}",
        start.elapsed(),
        foreign.circuit.stats()
    );
    let mut witness = foreign.circuit.witness();
    foreign.assign(&assignment, &mut witness)?;
    let scalar = witness.generate()?;
    let public: Vec<_> = fixture
        .public
        .iter()
        .map(|v| Scalar::from_u64(v.as_canonical_u64()))
        .collect();
    assert_eq!(scalar.public_values(), public);
    drop(assignment);
    drop(source);
    drop(foreign.inputs);
    println!("Scalar witness satisfied: {:?}", start.elapsed());
    let compiled = foreign
        .circuit
        .lower_to_multi_stark_sharded(Scalar::from_u8(93), 1 << 24)?
        .merge_table_traces(1 << 22)?;
    println!(
        "Lowered {} circuits: {:?}",
        compiled.num_circuits(),
        start.elapsed()
    );
    let shards = compiled.trace_shards(&scalar)?;
    let mut manifest = storage::Manifest {
        widths: vec![],
        heights: vec![],
        claims: compiled.claims(&public)?,
    };
    for i in 0..compiled.num_circuits() {
        let mut definition = compiled.kzg_circuit_input(i, 1 << 24, 2)?.unwrap();
        let fixed = definition.preprocessed.take().unwrap();
        manifest.widths.push(definition.main_width);
        manifest.heights.push(fixed.height());
        storage::save(&out.join(format!("{i}.meta")), &definition)?;
        storage::write_matrix(&out.join(format!("{i}.fixed.zst")), &fixed)?;
        drop(fixed);
        let trace = shards.trace(i)?;
        storage::write_matrix(&out.join(format!("{i}.witness.zst")), &trace)?;
        println!(
            "Staged {i}/{}: {:?}",
            compiled.num_circuits(),
            start.elapsed()
        );
    }
    storage::save(&out.join("manifest.bin"), &manifest)?;
    println!("STAGING COMPLETE: {:?}", start.elapsed());
    Ok(())
}

#[cfg(feature = "kzg")]
fn prove(dir: &std::path::Path, verify_only: bool) -> storage::Result<()> {
    use multi_stark::{
        ark_adapter::{KzgConfig, Scalar, Srs, compact::FixedProofCodec, pcs::KzgProverData},
        config::ProofConfig,
        lookup::LookupValues,
        prover::Stage1,
        system::{Circuit, ProverKey, System},
        traits::{Algebra, Field, Pcs},
    };
    use std::{
        fs::{self, File},
        io::{BufReader, BufWriter, Write},
        sync::Arc,
        time::Instant,
    };
    let manifest: storage::Manifest = storage::load(&dir.join("manifest.bin"))?;
    let count = manifest.heights.len();
    if count < 12 || manifest.widths.len() != count {
        return Err("unexpected trace layout".into());
    }
    let namespace = Scalar::from_u8(93);
    let mut expected: Vec<_> = std::iter::once(0)
        .chain(init_fri::INIT_PUBLIC_WORDS)
        .enumerate()
        .map(|(i, word)| {
            vec![
                namespace,
                Scalar::ONE,
                Scalar::from_usize(i),
                Scalar::from_u64(word),
            ]
        })
        .collect();
    expected.extend(
        (0..count - 10).map(|i| vec![namespace, Scalar::from_u8(3), Scalar::from_usize(i)]),
    );
    expected.extend((0..6).map(|i| {
        vec![
            namespace,
            Scalar::from_u8(4),
            Scalar::from_u8(3),
            Scalar::from_usize(i),
        ]
    }));
    if manifest.claims != expected {
        return Err(
            "staged claims differ from expected Init statement and activation anchors".into(),
        );
    }
    let height = *manifest.heights.iter().max().ok_or("empty manifest")?;
    let out = dir.join("kzg");
    fs::create_dir_all(&out)?;
    fs::write(
        out.join("SECURITY.txt"),
        "Development SRS with known trapdoor. Correctness and cost experiment only.\n",
    )?;
    let start = Instant::now();
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(height, b"init-fri-kzg-ordinary-v1")),
        2,
    )
    .with_streaming_lookups();
    println!("Development SRS: {:?}", start.elapsed());
    let mut circuits = Vec::new();
    let mut fixed_parts = Vec::new();
    let mut main_parts = Vec::new();
    let mut commits = Vec::new();
    let save = |path: &std::path::Path, data: &KzgProverData| -> storage::Result<()> {
        let temp = path.with_extension("partial");
        let mut file = BufWriter::with_capacity(1 << 20, File::create(&temp)?);
        data.write_checkpoint(&mut file)?;
        file.flush()?;
        drop(file);
        fs::rename(temp, path)?;
        Ok(())
    };
    for i in 0..count {
        let meta = out.join(format!("setup-{i}.bin"));
        let fixed_path = out.join(format!("fixed-{i}.bin"));
        let main_path = out.join(format!("main-{i}.bin"));
        let (circuit, commitment): (Circuit<Scalar>, _) =
            if meta.exists() && (verify_only || fixed_path.exists()) {
                if !verify_only {
                    fixed_parts.push(KzgProverData::read_checkpoint(BufReader::with_capacity(
                        1 << 20,
                        File::open(&fixed_path)?,
                    ))?);
                }
                storage::load(&meta)?
            } else {
                if verify_only {
                    return Err("missing setup checkpoint".into());
                }
                let (mut local, key) = System::new(config.clone(), [storage::definition(dir, i)?]);
                let mut circuit = local.circuits.remove(0);
                circuit.preprocessed = None;
                let data = key.preprocessed_data.unwrap();
                save(&fixed_path, &data)?;
                fixed_parts.push(data);
                let pair = (circuit, local.preprocessed_commit.unwrap());
                storage::save(&meta, &pair)?;
                pair
            };
        assert_eq!(circuit.main_width, manifest.widths[i]);
        assert_eq!(circuit.preprocessed_height, manifest.heights[i]);
        circuits.push(circuit);
        commits.push(commitment);
        if !verify_only {
            let data = if main_path.exists() {
                KzgProverData::read_checkpoint(BufReader::with_capacity(
                    1 << 20,
                    File::open(&main_path)?,
                ))?
            } else {
                let main = storage::read_matrix(&dir.join(format!("{i}.witness.zst")))?;
                let (_, data) = config.pcs().commit(vec![(
                    config.pcs().natural_domain_for_degree(manifest.heights[i]),
                    main,
                )]);
                save(&main_path, &data)?;
                data
            };
            main_parts.push(data);
        }
        println!("Prepared {i}/{count}: {:?}", start.elapsed());
    }
    let mut commitment = multi_stark::ark_adapter::KzgCommitment(vec![], vec![]);
    for mut c in commits {
        commitment.0.append(&mut c.0);
        commitment.1.append(&mut c.1);
    }
    let system = System {
        config,
        circuits,
        preprocessed_commit: Some(commitment),
        preprocessed_indices: (0..count).map(Some).collect(),
    };
    let logs: Vec<_> = manifest
        .heights
        .iter()
        .map(|h| u8::try_from(h.ilog2()).unwrap())
        .collect();
    let codec = FixedProofCodec::new(&system, &logs)?;
    let refs: Vec<_> = manifest.claims.iter().map(Vec::as_slice).collect();
    let proof_path = out.join("proof.compact.bin");
    let bytes = if verify_only {
        fs::read(&proof_path)?
    } else {
        let (fixed_commitment, fixed) = KzgProverData::concatenate(fixed_parts);
        assert_eq!(Some(fixed_commitment), system.preprocessed_commit);
        let (main_commitment, main) = KzgProverData::concatenate(main_parts);
        let lookups = system
            .circuits
            .iter()
            .zip(&manifest.heights)
            .map(|(c, &h)| {
                LookupValues::shape_only(
                    h,
                    &c.graph
                        .lookups
                        .iter()
                        .map(|l| l.args.len())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let stage = Stage1 {
            active: vec![true; count],
            active_indices: (0..count).collect(),
            log_degrees: logs.iter().map(|&l| usize::from(l)).collect(),
            stage_1_trace_commit: main_commitment,
            stage_1_trace_data: main,
            lookups,
        };
        println!(
            "Proving all {count} partitions together: {:?}",
            start.elapsed()
        );
        let proof = system.prove_committed(
            &ProverKey {
                preprocessed_data: Some(fixed),
            },
            &refs,
            stage,
        );
        system
            .verify_multiple_claims(&refs, &proof)
            .map_err(|e| format!("proof: {e:?}"))?;
        let bytes = codec.encode(&proof)?;
        fs::write(&proof_path, &bytes)?;
        bytes
    };
    let proof = codec.decode(&bytes)?;
    let verify_start = Instant::now();
    system
        .verify_multiple_claims(&refs, &proof)
        .map_err(|e| format!("decoded proof: {e:?}"))?;
    let verify_seconds = verify_start.elapsed().as_secs_f64();
    for i in 0..18 {
        let mut wrong = manifest.claims.clone();
        wrong[i + 1][3] += Scalar::ONE;
        assert!(
            system
                .verify_multiple_claims(
                    &wrong.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &proof
                )
                .is_err()
        );
    }
    let mut corrupted = bytes.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 1;
    assert!(match codec.decode(&corrupted) {
        Err(_) => true,
        Ok(p) => system.verify_multiple_claims(&refs, &p).is_err(),
    });
    assert!(codec.decode(&bytes[..bytes.len() - 1]).is_err());
    let mut extended = bytes.clone();
    extended.push(0);
    assert!(codec.decode(&extended).is_err());
    let mut profile = blake3::Hasher::new();
    profile.update(b"init-fri-kzg-ordinary-v1/development-srs/quotient2");
    profile.update(&fs::read(dir.join("plan-id.bin"))?);
    profile.update(&fs::read(dir.join("manifest.bin"))?);
    for i in 0..count {
        profile.update(&fs::read(out.join(format!("setup-{i}.bin")))?);
    }
    let mut packet = profile.finalize().as_bytes().to_vec();
    fs::write(out.join("profile-id.bin"), &packet)?;
    for claim in &manifest.claims[1..19] {
        packet.extend_from_slice(&claim[3].canonical_limbs_le()[0].to_le_bytes());
    }
    packet.extend_from_slice(&bytes);
    fs::write(out.join("packet.bin"), &packet)?;
    let report = format!(
        "VERIFIED: proof_bytes={} packet_bytes={} verify_seconds={verify_seconds} total_seconds={} altered_claims_rejected=18 development_srs=true\n",
        bytes.len(),
        packet.len(),
        start.elapsed().as_secs_f64()
    );
    fs::write(out.join("VERIFIED.txt"), &report)?;
    print!("{report}");
    Ok(())
}

#[cfg(all(test, feature = "kzg"))]
mod tests {
    use super::*;
    use multi_stark::{
        ark_adapter::Scalar,
        plonkish::{
            CircuitBuilder,
            gadgets::{ByteGadgets, blake3},
        },
        traits::Field,
    };
    use p3_matrix::Matrix;

    #[test]
    fn staged_hash_fixture_proves_and_resumes_verification() -> storage::Result<()> {
        let dir = std::env::temp_dir().join(format!("init-fri-kzg-fixture-{}", std::process::id()));
        std::fs::create_dir(&dir)?;
        let mut builder = CircuitBuilder::<Scalar>::new();
        builder.enable_compact_blake3();
        let bytes = ByteGadgets::new(&mut builder);
        let input = bytes.input(&mut builder, "byte");
        let _digest = blake3(&mut builder, &bytes, &[input]);
        let seven = builder.constant(Scalar::from_u8(7));
        for _ in 0..65540 {
            builder.assert_equal(input.value(), seven);
        }
        let public: Vec<_> = init_fri::INIT_PUBLIC_WORDS
            .into_iter()
            .map(Scalar::from_u64)
            .collect();
        for &word in &public {
            let value = builder.constant(word);
            builder.expose_public(value);
        }
        let compiled = builder
            .finish()
            .lower_to_multi_stark_sharded(Scalar::from_u8(93), 1 << 16)?
            .merge_table_traces(1 << 22)?;
        let mut witness = compiled.witness();
        witness.set(input.value(), Scalar::from_u8(7))?;
        let assignment = witness.generate()?;
        let traces = compiled.trace_shards(&assignment)?;
        let height = compiled
            .circuit_inputs()
            .iter()
            .map(|c| c.preprocessed.as_ref().unwrap().height())
            .max()
            .unwrap();
        let mut manifest = storage::Manifest {
            widths: vec![],
            heights: vec![],
            claims: compiled.claims(&public)?,
        };
        for i in 0..compiled.num_circuits() {
            let mut input = compiled.kzg_circuit_input(i, height, 2)?.unwrap();
            let fixed = input.preprocessed.take().unwrap();
            manifest.widths.push(input.main_width);
            manifest.heights.push(fixed.height());
            storage::save(&dir.join(format!("{i}.meta")), &input)?;
            storage::write_matrix(&dir.join(format!("{i}.fixed.zst")), &fixed)?;
            storage::write_matrix(&dir.join(format!("{i}.witness.zst")), &traces.trace(i)?)?;
        }
        storage::save(&dir.join("manifest.bin"), &manifest)?;
        std::fs::write(dir.join("plan-id.bin"), [0u8; 32])?;
        prove(&dir, false)?;
        let first = std::fs::read(dir.join("kzg/proof.compact.bin"))?;
        prove(&dir, false)?;
        assert_eq!(first, std::fs::read(dir.join("kzg/proof.compact.bin"))?);
        prove(&dir, true)?;
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }
}
