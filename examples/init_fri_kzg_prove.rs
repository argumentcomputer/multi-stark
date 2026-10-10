//! KZG compression of the saved recursive Init FRI proof with staged or direct traces.
#[path = "support/binary_capabilities.rs"]
mod binary_capabilities;
#[cfg(feature = "kzg")]
#[path = "support/kzg_fixed_cache.rs"]
mod fixed_cache;
#[cfg(feature = "kzg")]
#[path = "support/init_fri.rs"]
mod init_fri;
#[cfg(feature = "kzg")]
#[path = "support/kzg_setup.rs"]
mod setup;
#[cfg(feature = "kzg")]
#[allow(unreachable_pub)]
#[path = "support/kzg_storage.rs"]
mod storage;
#[cfg(feature = "kzg")]
#[path = "support/init_kzg_worker.rs"]
mod worker;
#[cfg(feature = "kzg")]
#[path = "support/kzg_worker_protocol.rs"]
mod worker_protocol;

#[cfg(feature = "kzg")]
const DEV_SEED: &[u8] = setup::DEVELOPMENT_PUBLIC_SEED;

#[cfg(not(feature = "kzg"))]
fn main() {
    if std::env::args().skip(1).eq(["capabilities"]) {
        binary_capabilities::print("init_fri_kzg_prove");
        return;
    }
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
    if args == ["capabilities"] {
        binary_capabilities::print("init_fri_kzg_prove");
        return Ok(());
    }
    if args.first().is_some_and(|arg| arg == "assignment-bench") {
        if args.len() != 2 {
            return Err("usage: init_fri_kzg_prove assignment-bench <saved-fri-dir>".into());
        }
        tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .init();
        return assignment_bench(std::path::Path::new(&args[1]));
    }
    if args.first().is_some_and(|arg| arg == "frontend-bench") {
        if args.len() != 2 {
            return Err("usage: init_fri_kzg_prove frontend-bench <saved-fri-dir>".into());
        }
        return frontend_bench(std::path::Path::new(&args[1]));
    }
    if args
        .first()
        .is_some_and(|arg| matches!(arg.as_str(), "serve" | "dev-v4-serve"))
    {
        if args.len() != 2 {
            return Err(
                "usage: init_fri_kzg_prove <serve|dev-v4-serve> <new-response-jsonl-path>".into(),
            );
        }
        tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .init();
        return if args[0] == "dev-v4-serve" {
            worker::serve_development_public_degree(std::path::Path::new(&args[1]))
        } else {
            worker::serve(std::path::Path::new(&args[1]))
        };
    }
    if args.first().is_some_and(|arg| arg == "dev-v4-verify") {
        if args.len() != 2 {
            return Err(
                "usage: init_fri_kzg_prove dev-v4-verify <saved-development-v4-dir>".into(),
            );
        }
        let setup = setup::SetupSource::development_public_degree_from_env()?;
        tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .init();
        return prove_with_state(
            std::path::Path::new(&args[1]),
            ProveMode::Verify,
            &mut None,
            &setup,
        );
    }
    let setup = setup::SetupSource::from_env()?;
    if args
        .first()
        .is_some_and(|arg| arg == "stage-and-prove-many")
    {
        if args.len() < 3 || args.len() % 2 != 1 {
            return Err("usage: init_fri_kzg_prove stage-and-prove-many <saved-fri-dir> <new-output-dir> [<saved-fri-dir> <new-output-dir> ...]".into());
        }
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
        return worker::run(&args[1..]);
    }
    if args.len() == 2 && (args[0] == "prove" || args[0] == "verify") {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
        let mode = if args[0] == "verify" {
            ProveMode::Verify
        } else {
            ProveMode::Checkpointed
        };
        return prove(std::path::Path::new(&args[1]), mode);
    }
    if args.len() != 3 || !matches!(args[0].as_str(), "stage" | "stage-and-prove") {
        return Err("usage: init_fri_kzg_prove <stage|stage-and-prove> <saved-fri-dir> <output-dir>, or <prove|verify> <output-dir>".into());
    }
    let fused = args[0] == "stage-and-prove";
    if fused {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }
    let out = PathBuf::from(&args[2]);
    fs::create_dir_all(&out)?;
    if out.join("manifest.bin").exists() {
        return Err("staging already complete".into());
    }
    let expected_words = init_fri::expected_words()?;
    let fixture = init_fri::load_with_expected(&PathBuf::from(&args[1]), &expected_words)?;
    let plan = VerifierPlan::validate(&fixture.key, fixture.profile, VerifierLimits::default())?;
    let plan_id = plan.identity(&fixture.schema)?;
    fs::write(out.join("plan-id.bin"), plan_id)?;
    let setup_id = setup.identity(DEV_SEED, 1 << 24, 2)?;
    setup.bind_stage(&out, &setup_id)?;
    let cache_profile = [plan_id, setup_id].concat();
    let cache = fixed_cache::FixedCache::for_stage(&out, &cache_profile)?;
    let cached = cache
        .as_ref()
        .map(|c| c.restore(&out))
        .transpose()?
        .flatten();
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
    if cached.as_ref().is_some_and(|shape| {
        shape.widths.len() != compiled.num_circuits()
            || shape.heights.len() != compiled.num_circuits()
    }) {
        return Err("cached trace count differs from the compiled circuit".into());
    }
    for i in 0..compiled.num_circuits() {
        let (trace_height, trace_width) = compiled
            .trace_dimensions(i)
            .ok_or("missing compiled trace")?;
        if let Some(shape) = &cached {
            if shape.widths[i] != trace_width || shape.heights[i] != trace_height {
                return Err(
                    "cached fixed trace dimensions differ from the compiled circuit".into(),
                );
            }
            manifest.widths.push(shape.widths[i]);
            manifest.heights.push(shape.heights[i]);
        } else if fused {
            manifest.widths.push(trace_width);
            manifest.heights.push(trace_height);
        } else {
            let fixed_started = Instant::now();
            let mut definition = compiled
                .kzg_circuit_input_with_degree_policy(i, 1 << 24, 2, !setup.uses_public_degree())?
                .unwrap();
            let fixed = definition.preprocessed.take().unwrap();
            manifest.widths.push(definition.main_width);
            manifest.heights.push(fixed.height());
            storage::save(&out.join(format!("{i}.meta")), &definition)?;
            storage::write_matrix(&out.join(format!("{i}.fixed.zst")), &fixed)?;
            println!(
                "Fixed trace staged circuit={i} seconds={:.9}",
                fixed_started.elapsed().as_secs_f64()
            );
        }
        if !fused {
            let trace = shards.trace(i)?;
            if trace.width() != manifest.widths[i] || trace.height() != manifest.heights[i] {
                return Err("cached fixed trace and witness have different shapes".into());
            }
            storage::write_matrix(&out.join(format!("{i}.witness.zst")), &trace)?;
        }
        println!(
            "Staged {i}/{}: {:?}",
            compiled.num_circuits(),
            start.elapsed()
        );
    }
    storage::save(&out.join("manifest.bin"), &manifest)?;
    if fused {
        println!(
            "Metadata staging complete; generating fixed and witness traces directly into commitment"
        );
        prove_with_state_with_expected(
            &out,
            ProveMode::Fresh(&shards),
            &mut None,
            &setup,
            &expected_words,
        )?;
        println!("STAGE-AND-PROVE COMPLETE: {:?}", start.elapsed());
    } else {
        println!("STAGING COMPLETE: {:?}", start.elapsed());
    }
    Ok(())
}

#[cfg(feature = "kzg")]
fn frontend_bench(dir: &std::path::Path) -> storage::Result<()> {
    use multi_stark::plonkish::{foreign::GoldilocksCircuit, verifier::*};
    use std::time::Instant;

    let load_started = Instant::now();
    let fixture = init_fri::load(dir)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let source_started = Instant::now();
    let plan = VerifierPlan::validate(&fixture.key, fixture.profile, VerifierLimits::default())?;
    let plan_id = plan.identity(&fixture.schema)?;
    let (source, inputs) = plan.build(
        fixture.schema,
        ImplementationOptions {
            compact_blake3: true,
        },
    )?;
    let source_seconds = source_started.elapsed().as_secs_f64();
    let source_stats = source.stats();
    drop(inputs);
    let translation_started = Instant::now();
    let foreign = GoldilocksCircuit::new_preallocated(&source);
    let translation_seconds = translation_started.elapsed().as_secs_f64();
    let scalar_stats = std::hint::black_box(&foreign).circuit.stats();
    println!(
        concat!(
            "FRONTEND_BENCH {{\"format\":\"init-fri-translation/v1\",",
            "\"load_and_verify_seconds\":{:.9},\"source_build_seconds\":{:.9},",
            "\"translation_seconds\":{:.9},\"plan_id\":{:?},",
            "\"source_stats\":{},\"scalar_stats\":{}}}"
        ),
        load_seconds,
        source_seconds,
        translation_seconds,
        plan_id,
        stats_json(source_stats),
        stats_json(scalar_stats)
    );
    Ok(())
}

#[cfg(feature = "kzg")]
fn stats_json(s: multi_stark::plonkish::CircuitStats) -> String {
    format!(
        concat!(
            "{{\"hash_calls\":{},\"hash_compressions\":{},\"values\":{},",
            "\"inputs\":{},\"gates\":{},\"lookups\":{},\"tables\":{},",
            "\"publics\":{},\"hint_calls\":{},\"hint_outputs\":{}}}"
        ),
        s.hash_calls,
        s.hash_compressions,
        s.values,
        s.inputs,
        s.gates,
        s.lookups,
        s.tables,
        s.publics,
        s.hint_calls,
        s.hint_outputs
    )
}

#[cfg(feature = "kzg")]
fn assignment_bench(dir: &std::path::Path) -> storage::Result<()> {
    use multi_stark::{
        ark_adapter::Scalar,
        plonkish::{foreign::GoldilocksCircuit, verifier::*},
        traits::Field,
    };
    use p3_field::PrimeField64;
    use std::time::Instant;

    let started = Instant::now();
    let fixture = init_fri::load(dir)?;
    let load_seconds = started.elapsed().as_secs_f64();
    let source_started = Instant::now();
    let plan = VerifierPlan::validate(&fixture.key, fixture.profile, VerifierLimits::default())?;
    let plan_id = plan.identity(&fixture.schema)?;
    let (source, inputs) = plan.build(
        fixture.schema,
        ImplementationOptions {
            compact_blake3: true,
        },
    )?;
    let source_seconds = source_started.elapsed().as_secs_f64();
    let source_assignment_started = Instant::now();
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
    let source_assignment_seconds = source_assignment_started.elapsed().as_secs_f64();
    let translation_started = Instant::now();
    let foreign = GoldilocksCircuit::new_preallocated(&source);
    let translation_seconds = translation_started.elapsed().as_secs_f64();
    let scalar_started = Instant::now();
    let mut witness = foreign.circuit.witness();
    foreign.assign(&assignment, &mut witness)?;
    let input_seconds = scalar_started.elapsed().as_secs_f64();
    let generation_started = Instant::now();
    let scalar = witness.generate()?;
    let generation_seconds = generation_started.elapsed().as_secs_f64();
    let assignment_seconds = scalar_started.elapsed().as_secs_f64();
    let expected: Vec<_> = fixture
        .public
        .iter()
        .map(|value| Scalar::from_u64(value.as_canonical_u64()))
        .collect();
    assert_eq!(scalar.public_values(), expected);
    let public_words: Vec<_> = fixture
        .public
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect();
    println!(
        concat!(
            "ASSIGNMENT_BENCH {{\"format\":\"init-fri-assignment/v1\",",
            "\"load_and_verify_seconds\":{:.9},\"source_build_seconds\":{:.9},",
            "\"source_assignment_seconds\":{:.9},\"translation_seconds\":{:.9},",
            "\"scalar_input_seconds\":{:.9},\"scalar_generation_seconds\":{:.9},",
            "\"scalar_assignment_seconds\":{:.9},\"total_seconds\":{:.9},",
            "\"plan_id\":{:?},\"public_words\":{:?},",
            "\"source_stats\":{},\"scalar_stats\":{}}}"
        ),
        load_seconds,
        source_seconds,
        source_assignment_seconds,
        translation_seconds,
        input_seconds,
        generation_seconds,
        assignment_seconds,
        started.elapsed().as_secs_f64(),
        plan_id,
        public_words,
        stats_json(source.stats()),
        stats_json(foreign.circuit.stats())
    );
    Ok(())
}

#[cfg(feature = "kzg")]
#[derive(Clone, Copy)]
enum ProveMode<'a> {
    Checkpointed,
    Fresh(&'a multi_stark::plonkish::TraceShards<'a, multi_stark::ark_adapter::Scalar>),
    Verify,
}

#[cfg(feature = "kzg")]
fn prove(dir: &std::path::Path, mode: ProveMode<'_>) -> storage::Result<()> {
    prove_with_state(dir, mode, &mut None, &setup::SetupSource::from_env()?)
}

#[cfg(feature = "kzg")]
struct LoadedProver {
    system: multi_stark::system::System<multi_stark::ark_adapter::KzgConfig>,
    key: multi_stark::system::ProverKey<multi_stark::ark_adapter::KzgConfig>,
    codec: multi_stark::ark_adapter::compact::FixedProofCodec,
    widths: Vec<usize>,
    heights: Vec<usize>,
    logs: Vec<u8>,
    plan_id: [u8; 32],
    setup_id: [u8; 32],
    setup_bytes: Vec<Vec<u8>>,
    prefetch_bytes: usize,
}

#[cfg(feature = "kzg")]
fn prove_with_state(
    dir: &std::path::Path,
    mode: ProveMode<'_>,
    retained: &mut Option<LoadedProver>,
    setup: &setup::SetupSource,
) -> storage::Result<()> {
    prove_with_state_with_expected(dir, mode, retained, setup, &init_fri::expected_words()?)
}

#[cfg(feature = "kzg")]
fn prove_with_state_with_expected(
    dir: &std::path::Path,
    mode: ProveMode<'_>,
    retained: &mut Option<LoadedProver>,
    setup: &setup::SetupSource,
    expected_words: &[u64; 18],
) -> storage::Result<()> {
    use multi_stark::{
        ark_adapter::{Scalar, compact::FixedProofCodec, pcs::KzgProverData},
        config::ProofConfig,
        lookup::LookupValues,
        prover::Stage1,
        system::{Circuit, ProverKey, System},
        traits::{Algebra, Field, Pcs},
    };
    use p3_matrix::Matrix;
    use std::{
        fs::{self, File},
        io::{BufReader, BufWriter, Write},
        time::Instant,
    };
    let verify_only = matches!(mode, ProveMode::Verify);
    let fresh = matches!(mode, ProveMode::Fresh(_));
    let cold = retained.is_none();
    let manifest: storage::Manifest = storage::load(&dir.join("manifest.bin"))?;
    let count = manifest.heights.len();
    if count < 12 || manifest.widths.len() != count || manifest.widths.contains(&0) {
        return Err("unexpected trace layout".into());
    }
    for &height in &manifest.heights {
        setup.check_height(height)?;
    }
    if let ProveMode::Fresh(traces) = mode {
        let compiled = traces.compiled();
        if compiled.num_circuits() != count
            || (0..count).any(|i| {
                compiled.trace_dimensions(i) != Some((manifest.heights[i], manifest.widths[i]))
            })
        {
            return Err("generated trace dimensions differ from the staged manifest".into());
        }
    }
    let height = *manifest.heights.iter().max().ok_or("empty manifest")?;
    let setup_id = setup.identity(DEV_SEED, height, 2)?;
    setup.check_binding(dir, &setup_id)?;
    let plan_id: [u8; 32] = fs::read(dir.join("plan-id.bin"))?
        .try_into()
        .map_err(|_| "invalid verifier plan identity")?;
    if let Some(loaded) = retained {
        if loaded.plan_id != plan_id
            || loaded.setup_id != setup_id
            || loaded.widths != manifest.widths
            || loaded.heights != manifest.heights
        {
            return Err("request differs from the loaded prover profile".into());
        }
        if !verify_only && loaded.key.preprocessed_data.is_none() {
            return Err("loaded verifier has no proving key".into());
        }
    }
    let namespace = Scalar::from_u8(93);
    let mut expected: Vec<_> = std::iter::once(0)
        .chain(*expected_words)
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
    let out = dir.join("kzg");
    fs::create_dir_all(&out)?;
    fs::write(out.join("SECURITY.txt"), setup.security_description())?;
    let start = Instant::now();
    let (config, prefetch_bytes) = if let Some(loaded) = retained {
        for (i, bytes) in loaded.setup_bytes.iter().enumerate() {
            fs::write(out.join(format!("setup-{i}.bin")), bytes)?;
        }
        println!("Reusing loaded SRS, fixed coefficients and compiled prover");
        (loaded.system.config.clone(), loaded.prefetch_bytes)
    } else {
        let prefetch_gib: usize =
            std::env::var("MULTI_STARK_KZG_PREFETCH_GIB").map_or(Ok(32), |value| value.parse())?;
        let prefetch_bytes = prefetch_gib
            .checked_mul(1 << 30)
            .ok_or("prefetch budget overflow")?;
        let config = setup
            .config(DEV_SEED, height, 2, verify_only)?
            .with_streaming_lookups()
            .with_partition_pipeline(prefetch_bytes);
        println!("KZG parameters ready: {:?}", start.elapsed());
        (config, prefetch_bytes)
    };
    println!("Partition prefetch budget: {} GiB", prefetch_bytes >> 30);
    let mut circuits = Vec::new();
    let mut fixed_parts = Vec::new();
    let mut main_parts = Vec::new();
    let mut commits = Vec::new();
    let mut setup_bytes = Vec::new();
    let save = |path: &std::path::Path, data: &KzgProverData| -> storage::Result<()> {
        let temp = path.with_extension("partial");
        let mut file = BufWriter::with_capacity(1 << 20, File::create(&temp)?);
        data.write_checkpoint(&mut file)?;
        file.flush()?;
        drop(file);
        fs::rename(temp, path)?;
        Ok(())
    };
    let load_witness = |i: usize| -> Result<_, String> {
        if let ProveMode::Fresh(traces) = mode {
            let trace_started = Instant::now();
            let trace = traces.trace(i).map_err(|error| error.to_string())?;
            if trace.width() != manifest.widths[i] || trace.height() != manifest.heights[i] {
                return Err("generated trace dimensions differ from the staged manifest".into());
            }
            println!(
                "KZG fresh trace generated circuit={i} seconds={:.9} overlap_possible=true",
                trace_started.elapsed().as_secs_f64()
            );
            Ok(Some(trace))
        } else if verify_only || out.join(format!("main-{i}.bin")).exists() {
            Ok(None)
        } else {
            storage::read_matrix_with_shape(
                &dir.join(format!("{i}.witness.zst")),
                manifest.widths[i],
                manifest.heights[i],
            )
            .map(Some)
            .map_err(|error| error.to_string())
        }
    };
    let mut prefetched = None;
    for i in 0..count {
        let witness = prefetched.take().unwrap_or_else(|| load_witness(i))?;
        let prepare = || -> storage::Result<()> {
            let meta = out.join(format!("setup-{i}.bin"));
            let fixed_path = out.join(format!("fixed-{i}.bin"));
            let main_path = out.join(format!("main-{i}.bin"));
            if cold {
                let fixed_started = Instant::now();
                let from_checkpoint = meta.exists() && (verify_only || fixed_path.exists());
                let (circuit, commitment): (Circuit<Scalar>, _) = if from_checkpoint {
                    if !verify_only {
                        fixed_parts.push(KzgProverData::read_checkpoint(
                            BufReader::with_capacity(1 << 20, File::open(&fixed_path)?),
                        )?);
                    }
                    storage::load(&meta)?
                } else {
                    if verify_only {
                        return Err("missing setup checkpoint".into());
                    }
                    let load_started = Instant::now();
                    let mut definition = match mode {
                        ProveMode::Fresh(traces) => traces
                            .compiled()
                            .kzg_circuit_input_with_degree_policy(
                                i,
                                height,
                                2,
                                !setup.uses_public_degree(),
                            )?
                            .ok_or("missing compiled fixed trace")?,
                        _ => storage::definition(dir, i)?,
                    };
                    let fixed = definition
                        .preprocessed
                        .as_ref()
                        .ok_or("missing preprocessing")?;
                    if definition.main_width != manifest.widths[i]
                        || fixed.height() != manifest.heights[i]
                    {
                        return Err("fixed trace dimensions differ from the staged manifest".into());
                    }
                    if fresh {
                        let fixed = definition.preprocessed.take();
                        storage::save(&dir.join(format!("{i}.meta")), &definition)?;
                        definition.preprocessed = fixed;
                    }
                    println!(
                        "KZG fixed trace {} circuit={i} seconds={:.9} overlap_possible=true",
                        if fresh { "generated" } else { "loaded" },
                        load_started.elapsed().as_secs_f64()
                    );
                    let commit_started = Instant::now();
                    let (mut local, key) = System::new(config.clone(), [definition]);
                    println!(
                        "KZG fixed committed circuit={i} seconds={:.9} overlap_possible=true",
                        commit_started.elapsed().as_secs_f64()
                    );
                    let mut circuit = local.circuits.remove(0);
                    circuit.preprocessed = None;
                    let data = key.preprocessed_data.unwrap();
                    save(&fixed_path, &data)?;
                    fixed_parts.push(data);
                    let pair = (circuit, local.preprocessed_commit.unwrap());
                    storage::save(&meta, &pair)?;
                    pair
                };
                if circuit.main_width != manifest.widths[i]
                    || circuit.preprocessed_height != manifest.heights[i]
                {
                    return Err(
                        "setup checkpoint dimensions differ from the staged manifest".into(),
                    );
                }
                circuits.push(circuit);
                commits.push(commitment);
                setup_bytes.push(fs::read(&meta)?);
                println!(
                    "KZG fixed prepared circuit={i} seconds={:.9} source={} overlap_possible=true",
                    fixed_started.elapsed().as_secs_f64(),
                    if from_checkpoint {
                        "checkpoint"
                    } else {
                        "generated"
                    },
                );
            }
            if !verify_only {
                let main_started = Instant::now();
                let from_checkpoint = !fresh && main_path.exists();
                let data = if from_checkpoint {
                    KzgProverData::read_checkpoint(BufReader::with_capacity(
                        1 << 20,
                        File::open(&main_path)?,
                    ))?
                } else {
                    let main = witness.ok_or("missing prefetched witness")?;
                    let commit_started = Instant::now();
                    let (_, data) = config.pcs().commit(vec![(
                        config.pcs().natural_domain_for_degree(manifest.heights[i]),
                        main,
                    )]);
                    println!(
                        "KZG main committed circuit={i} seconds={:.9} overlap_possible=true",
                        commit_started.elapsed().as_secs_f64()
                    );
                    if !fresh {
                        save(&main_path, &data)?;
                    }
                    data
                };
                main_parts.push(data);
                println!(
                    "KZG main prepared circuit={i} seconds={:.9} source={} overlap_possible=true",
                    main_started.elapsed().as_secs_f64(),
                    if from_checkpoint {
                        "checkpoint"
                    } else {
                        "generated"
                    },
                );
            }
            println!("Prepared {i}/{count}: {:?}", start.elapsed());
            Ok(())
        };
        let can_prefetch = !verify_only
            && prefetch_bytes > 0
            && i + 1 < count
            && manifest.widths[i + 1]
                .checked_mul(manifest.heights[i + 1])
                .and_then(|values| values.checked_mul(size_of::<Scalar>()))
                .is_some_and(|bytes| bytes <= prefetch_bytes);
        if can_prefetch {
            let (prepared, next) = p3_maybe_rayon::prelude::join(
                || prepare().map_err(|error| error.to_string()),
                || load_witness(i + 1),
            );
            prepared?;
            prefetched = Some(next);
        } else {
            prepare()?;
        }
    }
    if cold {
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
        let key = if verify_only {
            ProverKey {
                preprocessed_data: None,
            }
        } else {
            let (commitment, data) = KzgProverData::concatenate(fixed_parts);
            assert_eq!(Some(commitment), system.preprocessed_commit);
            ProverKey {
                preprocessed_data: Some(data),
            }
        };
        let logs: Vec<_> = manifest
            .heights
            .iter()
            .map(|h| u8::try_from(h.ilog2()).unwrap())
            .collect();
        let codec = FixedProofCodec::new(&system, &logs)?;
        *retained = Some(LoadedProver {
            system,
            key,
            codec,
            widths: manifest.widths.clone(),
            heights: manifest.heights.clone(),
            logs,
            plan_id,
            setup_id,
            setup_bytes,
            prefetch_bytes,
        });
    }
    let loaded = retained.as_ref().unwrap();
    let system = &loaded.system;
    let codec = &loaded.codec;
    let logs = &loaded.logs;
    let refs: Vec<_> = manifest.claims.iter().map(Vec::as_slice).collect();
    let proof_path = out.join("proof.compact.bin");
    let bytes = if verify_only {
        fs::read(&proof_path)?
    } else {
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
        let proof_started = Instant::now();
        let proof = system.prove_committed(&loaded.key, &refs, stage);
        println!(
            "KZG proof computed seconds={:.9}",
            proof_started.elapsed().as_secs_f64()
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
    if !setup.uses_public_degree() {
        profile.update(b"init-fri-kzg-ordinary-v1/development-srs/quotient2");
    } else {
        profile.update(b"init-fri-kzg-ordinary/v4");
        profile.update(system.config.transcript_seed());
    }
    profile.update(&fs::read(dir.join("plan-id.bin"))?);
    profile.update(&fs::read(dir.join("manifest.bin"))?);
    for i in 0..count {
        profile.update(&fs::read(out.join(format!("setup-{i}.bin")))?);
    }
    let profile = *profile.finalize().as_bytes();
    let mut packet = profile.to_vec();
    for claim in &manifest.claims[1..19] {
        packet.extend_from_slice(&claim[3].canonical_limbs_le()[0].to_le_bytes());
    }
    packet.extend_from_slice(&bytes);
    if verify_only {
        if fs::read(out.join("profile-id.bin"))? != profile
            || fs::read(out.join("packet.bin"))? != packet
        {
            return Err("saved packet or profile differs".into());
        }
    } else {
        fs::write(out.join("profile-id.bin"), profile)?;
        fs::write(out.join("packet.bin"), &packet)?;
    }
    let report = format!(
        "VERIFIED: proof_bytes={} packet_bytes={} verify_seconds={verify_seconds} total_seconds={} altered_claims_rejected=18 development_srs={}\n",
        bytes.len(),
        packet.len(),
        start.elapsed().as_secs_f64(),
        setup.is_development()
    );
    fs::write(out.join("VERIFIED.txt"), &report)?;
    let (setup_name, manifest_digest) = match setup {
        setup::SetupSource::Filecoin { digest, .. } => (
            "filecoin",
            Some(blake3::Hash::from_bytes(*digest).to_string()),
        ),
        setup::SetupSource::Development { .. }
        | setup::SetupSource::KnownTrapdoorPublicDegree { .. } => ("development", None),
    };
    let public_setup = system.config.srs().public_setup();
    let public_setup_id =
        public_setup.map(|parameters| blake3::Hash::from_bytes(parameters.id).to_string());
    let ceremony_id = if setup.is_development() {
        None
    } else {
        public_setup_id.clone()
    };
    let evidence = serde_json::json!({
        "proof_bytes": bytes.len(),
        "packet_bytes": packet.len(),
        "trace_heights": manifest.heights,
        "verify_seconds": verify_seconds,
        "total_seconds": start.elapsed().as_secs_f64(),
        "native_verification_passed": true,
        "negative_tests_pass": true,
        "altered_claims_rejected": 18,
        "development_srs": setup.is_development(),
        "known_trapdoor": setup.is_development(),
        "filecoin_acceptance": !setup.is_development(),
        "setup": setup_name,
        "setup_id": blake3::Hash::from_bytes(setup_id).to_string(),
        "filecoin_manifest_digest": manifest_digest,
        "ceremony_id": ceremony_id,
        "public_setup_id": public_setup_id,
        "public_max_degree": public_setup.map(|parameters| parameters.max_degree),
    });
    fs::write(
        out.join(if verify_only {
            "verify-report.json"
        } else {
            "prove-report.json"
        }),
        format!("{evidence}\n"),
    )?;
    if cold
        && !verify_only
        && let Some(cache) = fixed_cache::FixedCache::for_prove(dir)?
    {
        cache.publish(dir)?;
    }
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

    fn prove(dir: &std::path::Path, mode: ProveMode<'_>) -> storage::Result<()> {
        prove_with_state(
            dir,
            mode,
            &mut None,
            &setup::SetupSource::Development { cache: None },
        )
    }

    #[test]
    fn staged_hash_fixture_proves_and_resumes_verification() -> storage::Result<()> {
        exercise_staged_and_fused_fixture(1)
    }

    #[test]
    fn malformed_stage_shapes_fail_before_loading_parameters() -> storage::Result<()> {
        let dir =
            std::env::temp_dir().join(format!("init-fri-kzg-invalid-shape-{}", std::process::id()));
        std::fs::create_dir(&dir)?;
        let mut manifest = storage::Manifest {
            widths: vec![3; 12],
            heights: vec![1 << 24; 12],
            claims: vec![],
        };
        for invalid in [0, 1, 3] {
            manifest.heights[1] = invalid;
            storage::save(&dir.join("manifest.bin"), &manifest)?;
            let error = prove(&dir, ProveMode::Verify).unwrap_err();
            assert!(error.to_string().contains("trace height"));
        }
        manifest.heights[1] = 1 << 16;
        manifest.widths[1] = 0;
        storage::save(&dir.join("manifest.bin"), &manifest)?;
        let error = prove(&dir, ProveMode::Verify).unwrap_err();
        assert!(error.to_string().contains("unexpected trace layout"));
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    #[ignore = "isolated comparison of checkpointed and direct trace proving"]
    fn fused_staging_benchmark() -> storage::Result<()> {
        exercise_staged_and_fused_fixture(3)
    }

    fn exercise_staged_and_fused_fixture(iterations: u8) -> storage::Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "init-fri-kzg-fixture-{}-{iterations}",
            std::process::id()
        ));
        std::fs::create_dir(&dir)?;
        let mut builder = CircuitBuilder::<Scalar>::new();
        builder.enable_compact_blake3();
        let bytes = ByteGadgets::new(&mut builder);
        let input = bytes.input(&mut builder, "byte");
        let private = bytes.input(&mut builder, "private byte");
        let _digest = blake3(&mut builder, &bytes, &[input, private]);
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
        let assign = |private_byte| -> storage::Result<_> {
            let mut witness = compiled.witness();
            witness.set(input.value(), Scalar::from_u8(7))?;
            witness.set(private.value(), Scalar::from_u8(private_byte))?;
            Ok(witness.generate()?)
        };
        let assignment = assign(9)?;
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
        prove(&dir, ProveMode::Checkpointed)?;
        let first = std::fs::read(dir.join("kzg/proof.compact.bin"))?;
        let artifacts = |directory: &std::path::Path| -> storage::Result<Vec<Vec<u8>>> {
            Ok(["proof.compact.bin", "packet.bin", "profile-id.bin"]
                .into_iter()
                .map(|name| std::fs::read(directory.join("kzg").join(name)))
                .collect::<Result<_, _>>()?)
        };
        let first_artifacts = artifacts(&dir)?;
        let direct = dir.join("direct-fixed");
        std::fs::create_dir(&direct)?;
        for name in ["manifest.bin", "plan-id.bin"] {
            std::fs::copy(dir.join(name), direct.join(name))?;
        }
        std::fs::write(direct.join("0.meta"), b"invalid metadata")?;
        std::fs::write(direct.join("0.fixed.zst"), b"unused fixed trace")?;
        prove(&direct, ProveMode::Fresh(&traces))?;
        assert_eq!(first_artifacts, artifacts(&direct)?);
        for i in 0..compiled.num_circuits() {
            assert_eq!(
                std::fs::read(dir.join(format!("{i}.meta")))?,
                std::fs::read(direct.join(format!("{i}.meta")))?
            );
            assert_eq!(
                std::fs::read(dir.join(format!("kzg/setup-{i}.bin")))?,
                std::fs::read(direct.join(format!("kzg/setup-{i}.bin")))?
            );
            assert_eq!(
                std::fs::read(dir.join(format!("kzg/fixed-{i}.bin")))?,
                std::fs::read(direct.join(format!("kzg/fixed-{i}.bin")))?
            );
            assert!(!direct.join(format!("kzg/main-{i}.bin")).exists());
            assert!(!direct.join(format!("{i}.witness.zst")).exists());
            if i != 0 {
                assert!(!direct.join(format!("{i}.fixed.zst")).exists());
            }
        }
        assert_eq!(
            std::fs::read(direct.join("0.fixed.zst"))?,
            b"unused fixed trace"
        );

        for case in 0..3 {
            let mut shape = storage::Manifest {
                widths: manifest.widths.clone(),
                heights: manifest.heights.clone(),
                claims: manifest.claims.clone(),
            };
            match case {
                0 => shape.widths[0] += 1,
                1 => shape.heights[0] *= 2,
                _ => {
                    shape.widths.push(1);
                    shape.heights.push(2);
                }
            }
            let malformed = dir.join(format!("invalid-fresh-shape-{case}"));
            std::fs::create_dir(&malformed)?;
            storage::save(&malformed.join("manifest.bin"), &shape)?;
            let unavailable = setup::SetupSource::Filecoin {
                cache: malformed.join("unprovisioned-filecoin"),
                digest: [0; 32],
            };
            let error = prove_with_state(
                &malformed,
                ProveMode::Fresh(&traces),
                &mut None,
                &unavailable,
            )
            .unwrap_err();
            assert!(error.to_string().contains("trace dimensions differ"));
            assert!(!malformed.join("kzg").exists());
        }
        let cache = fixed_cache::FixedCache::from_profile(&dir.join("cache"), b"fixture-v1")?;
        cache.publish(&direct)?;
        let reused = dir.join("reused");
        std::fs::create_dir(&reused)?;
        let shape = cache.restore(&reused)?.ok_or("missing populated cache")?;
        assert_eq!(shape.heights, manifest.heights);
        for name in ["manifest.bin", "plan-id.bin"] {
            std::fs::copy(dir.join(name), reused.join(name))?;
        }
        for i in 0..manifest.heights.len() {
            assert!(!reused.join(format!("{i}.fixed.zst")).exists());
            assert!(!reused.join(format!("kzg/main-{i}.bin")).exists());
            std::fs::copy(
                dir.join(format!("{i}.witness.zst")),
                reused.join(format!("{i}.witness.zst")),
            )?;
        }
        prove(&reused, ProveMode::Checkpointed)?;
        assert_eq!(first, std::fs::read(reused.join("kzg/proof.compact.bin"))?);
        let different =
            fixed_cache::FixedCache::from_profile(&dir.join("cache"), b"different-profile")?;
        assert!(different.restore(&dir.join("miss"))?.is_none());
        prove(&dir, ProveMode::Checkpointed)?;
        assert_eq!(first, std::fs::read(dir.join("kzg/proof.compact.bin"))?);
        prove(&dir, ProveMode::Verify)?;

        for name in ["packet.bin", "profile-id.bin"] {
            let path = dir.join("kzg").join(name);
            let original = std::fs::read(&path)?;
            let mut altered = original.clone();
            altered[0] ^= 1;
            std::fs::write(&path, &altered)?;
            let error = prove(&dir, ProveMode::Verify).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("saved packet or profile differs")
            );
            assert_eq!(std::fs::read(&path)?, altered);
            std::fs::write(path, original)?;
        }

        let changed = assign(231)?;
        let changed_traces = compiled.trace_shards(&changed)?;
        prove(&reused, ProveMode::Fresh(&changed_traces))?;
        assert_ne!(first, std::fs::read(reused.join("kzg/proof.compact.bin"))?);
        prove(&reused, ProveMode::Verify)?;

        manifest.widths[0] += 1;
        storage::save(&reused.join("manifest.bin"), &manifest)?;
        let error = prove(&reused, ProveMode::Fresh(&changed_traces)).unwrap_err();
        assert!(error.to_string().contains("trace dimensions differ"));
        manifest.widths[0] -= 1;

        for iteration in 0..iterations {
            let mut elapsed = [0.0; 2];
            let mut artifacts: [Vec<Vec<u8>>; 2] = std::array::from_fn(|_| Vec::new());
            let mut avoided_bytes = 0;
            let order = if iteration % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for fused in order {
                let output = dir.join(format!("comparison-{iteration}-{fused}"));
                std::fs::create_dir(&output)?;
                cache.restore(&output)?.ok_or("missing populated cache")?;
                for name in ["manifest.bin", "plan-id.bin"] {
                    std::fs::copy(dir.join(name), output.join(name))?;
                }
                let start = std::time::Instant::now();
                let fresh = assign(10 + iteration)?;
                let source = compiled.trace_shards(&fresh)?;
                if fused {
                    prove(&output, ProveMode::Fresh(&source))?;
                } else {
                    for i in 0..manifest.heights.len() {
                        storage::write_matrix(
                            &output.join(format!("{i}.witness.zst")),
                            &source.trace(i)?,
                        )?;
                    }
                    prove(&output, ProveMode::Checkpointed)?;
                }
                elapsed[usize::from(fused)] = start.elapsed().as_secs_f64();
                artifacts[usize::from(fused)] =
                    ["proof.compact.bin", "packet.bin", "profile-id.bin"]
                        .map(|name| std::fs::read(output.join("kzg").join(name)))
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()?;
                for i in 0..manifest.heights.len() {
                    for name in [format!("{i}.witness.zst"), format!("kzg/main-{i}.bin")] {
                        let path = output.join(name);
                        if fused {
                            assert!(!path.exists());
                        } else {
                            avoided_bytes += std::fs::metadata(path)?.len();
                        }
                    }
                }
            }
            assert_eq!(artifacts[0], artifacts[1]);
            println!(
                "fused_staging_sample iteration={iteration} checkpointed_seconds={:.6} fused_seconds={:.6} avoided_file_bytes={avoided_bytes} proof_digest={} packet_digest={} profile_digest={}",
                elapsed[0],
                elapsed[1],
                ::blake3::hash(&artifacts[0][0]).to_hex(),
                ::blake3::hash(&artifacts[0][1]).to_hex(),
                ::blake3::hash(&artifacts[0][2]).to_hex(),
            );
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }
}
