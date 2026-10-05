//! Check/prove verification of an exported Ix root with its unchanged 100-query profile.
mod storage;
#[path = "support/ix_vk.rs"]
mod ix_vk;
use multi_stark::{
    batch::BatchProof,
    plonkish::{
        CircuitBuilder,
        verifier::{
            Envelope, ImplementationOptions, Message, ProofEnvelope, ProofProfile, Statement,
            StatementBinding, StatementSlot, VerifierKey, VerifierLimits, VerifierPlan,
        },
    },
    system::System,
    types::{GoldilocksBlake3Config, Val},
};
use p3_blake3::Blake3;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_symmetric::CryptographicHasher;
use std::{fs, path::PathBuf, time::Instant};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::thread::spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(30));
        if let Ok(s)=std::fs::read_to_string("/proc/self/status") {
            eprintln!("{}",s.lines().filter(|l|l.starts_with("VmRSS:") || l.starts_with("VmSize:")).collect::<Vec<_>>().join(" "));
        }
    });
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("usage: ix_root <ix-artifacts> <output-dir> [--native-only | --check-only | --prove-outer]".into());
    }
    let dir = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    let mode = args.get(2).map_or("--check-only", String::as_str);
    if args.len() > 3 || !matches!(mode, "--native-only" | "--check-only" | "--prove-outer") {
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
    let start = Instant::now();
    let expanded = plan.expand_witness(ProofEnvelope::SingleBatch(&proof))?;
    println!("Untrusted witness expansion: {:?}", start.elapsed());
    if mode == "--native-only" {
        return Ok(());
    }
    let start = Instant::now();
    let mut builder = CircuitBuilder::new();
    let compact = std::env::var_os("IX_ROOT_GENERIC_HASHES").is_none();
    if compact {
        builder.enable_compact_blake3();
    }
    println!("Compact BLAKE3: {compact}");
    let claim_wires = claims
        .iter()
        .enumerate()
        .map(|(i, claim)| {
            (0..claim.len())
                .map(|j| StatementBinding::Wire(builder.public_input(format!("claim[{i}][{j}]"))))
                .collect()
        })
        .collect();
    let inputs = plan.constrain(
        &mut builder,
        Statement {
            claims: claim_wires,
            messages: proof
                .preamble
                .messages
                .iter()
                .map(|m| Message {
                    args: m
                        .args
                        .iter()
                        .copied()
                        .map(StatementBinding::Constant)
                        .collect(),
                    multiplicity: StatementBinding::Constant(m.multiplicity),
                })
                .collect(),
        },
        ImplementationOptions {
            compact_blake3: compact,
        },
    )?;
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
    fs::write(out.join("inner-vk.bin"), verifier_key.to_bytes()?)?;
    fs::write(out.join("plan-id.bin"), plan.identity(&schema)?)?;
    let compiler = std::process::Command::new("rustc")
        .arg("--version")
        .output()?;
    if !compiler.status.success() {
        return Err("could not identify Rust compiler".into());
    }
    let compiler = String::from_utf8(compiler.stdout)?;
    fs::write(
        out.join("build-id.bin"),
        plan.build_identity(
            &schema,
            ImplementationOptions {
                compact_blake3: compact,
            },
            compiler.trim(),
            "plonkish-multi-stark/v1",
            "max-height=16777216;lookup-group=3",
        )?,
    )?;
    fs::write(
        out.join("statement-schema.txt"),
        format!(
            "Public order: claims in order, then message arguments and multiplicity; constants omitted.\nProfile: {:?}\nSchema: {schema:?}\nCompiler: {compiler}Compact BLAKE3: {compact}\n",
            plan.profile()
        ),
    )?;
    let circuit = builder.finish();
    println!(
        "Circuit build: {:?}; {:?}",
        start.elapsed(),
        circuit.stats()
    );
    use multi_stark::{ark_adapter::Scalar,plonkish::foreign::GoldilocksCircuit};
    use multi_stark::traits::Field as _;
    let start=Instant::now();
    let mut witness=circuit.witness();
    inputs.assign_statement(&mut witness,&Statement {
        claims:claims.clone(),messages:proof.preamble.messages.iter().map(|m| Message {args:m.args.clone(),multiplicity:m.multiplicity}).collect()
    })?;
    expanded.assign_proof(&mut witness,&inputs)?;
    let assignment=witness.generate()?;
    assert_eq!(assignment.public_values(),claims.iter().flatten().copied().collect::<Vec<_>>());
    println!("Source witness verified: {:?}",start.elapsed());
    let start=Instant::now();
    let foreign=GoldilocksCircuit::new_preallocated(&circuit);
    println!("Scalar translation: {:?}; {:?}",start.elapsed(),foreign.circuit.stats());
    let start=Instant::now();
    let mut witness=foreign.circuit.witness();foreign.assign(&assignment,&mut witness)?;
    let scalar=witness.generate()?;
    let public:Vec<_>=claims.iter().flatten().map(|v| Scalar::from_u64(v.as_canonical_u64())).collect();
    assert_eq!(scalar.public_values(),public);
    println!("Scalar witness verified: {:?}",start.elapsed());
    drop(assignment);drop(circuit);drop(foreign.inputs);
    let start=Instant::now();
    let compiled=foreign.circuit.lower_to_multi_stark_sharded(Scalar::from_u8(93),1<<24)?;
    println!("Lowered {} circuits: {:?}",compiled.num_circuits(),start.elapsed());
    let shards=compiled.trace_shards(&scalar)?;
    let mut manifest=storage::Manifest {widths:vec![],heights:vec![],claims:compiled.claims(&public)?};
    for i in 0..compiled.num_circuits() {
        let start=Instant::now();
        let mut definition=compiled.kzg_circuit_input(i,1<<24,8)?.unwrap();
        let fixed=definition.preprocessed.take().unwrap();
        manifest.widths.push(definition.main_width);manifest.heights.push(fixed.values.len()/fixed.width);
        storage::save(&out.join(format!("{i}.meta")),&definition)?;
        storage::write_matrix(&out.join(format!("{i}.fixed.zst")),&fixed)?;drop(fixed);
        let trace=shards.trace(i)?;
        storage::write_matrix(&out.join(format!("{i}.witness.zst")),&trace)?;
        println!("Staged circuit {i}/{}: {:?}",compiled.num_circuits(),start.elapsed());
    }
    storage::save(&out.join("manifest.bin"),&manifest)?;
    println!("STAGING COMPLETE: {} circuits; source and witness may now be released",manifest.widths.len());
    Ok(())
}
