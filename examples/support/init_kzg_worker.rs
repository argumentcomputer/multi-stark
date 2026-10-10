use super::{
    DEV_SEED, LoadedProver, ProveMode, fixed_cache, init_fri, prove_with_state_with_expected,
    setup::SetupSource, storage, worker_protocol,
};
use multi_stark::{
    ark_adapter::Scalar,
    plonkish::{
        Assignment, Circuit, MultiStarkCircuit,
        foreign::{GoldilocksCircuit, GoldilocksInputs},
        verifier::{
            ImplementationOptions, ProofEnvelope, Statement, VerifierInputs, VerifierLimits,
            VerifierPlan,
        },
    },
    traits::Field,
    types::Val,
};
use p3_field::PrimeField64;
use std::{fs, io::BufRead, path::Path, time::Instant};

pub(super) struct Frontend {
    source: Circuit<Val>,
    scalar_inputs: GoldilocksInputs,
    compiled: MultiStarkCircuit<Scalar>,
}

impl Frontend {
    pub(super) fn compile(
        source: Circuit<Val>,
        max_main_height: usize,
        max_table_height: usize,
    ) -> storage::Result<Self> {
        let translated = GoldilocksCircuit::new_preallocated(&source);
        let compiled = translated
            .circuit
            .lower_to_multi_stark_sharded(Scalar::from_u8(93), max_main_height)?
            .merge_table_traces(max_table_height)?;
        Ok(Self {
            source,
            scalar_inputs: translated.inputs,
            compiled,
        })
    }

    pub(super) fn source(&self) -> &Circuit<Val> {
        &self.source
    }

    pub(super) fn compiled(&self) -> &MultiStarkCircuit<Scalar> {
        &self.compiled
    }

    pub(super) fn assign(&self, source: &Assignment<Val>) -> storage::Result<Assignment<Scalar>> {
        let mut witness = self.compiled.witness();
        self.scalar_inputs.assign(source, &mut witness)?;
        Ok(witness.generate()?)
    }
}

struct Worker<'a> {
    plan: VerifierPlan<'a>,
    plan_id: [u8; 32],
    inputs: VerifierInputs,
    frontend: Frontend,
    key_bytes: Vec<u8>,
    loaded: Option<LoadedProver>,
    setup: SetupSource,
}

impl<'a> Worker<'a> {
    fn compile(fixture: &'a init_fri::Fixture) -> storage::Result<Self> {
        Self::compile_with_layout(fixture, 1 << 24, 1 << 22, SetupSource::from_env()?)
    }

    fn compile_with_layout(
        fixture: &'a init_fri::Fixture,
        max_main_height: usize,
        max_table_height: usize,
        setup: SetupSource,
    ) -> storage::Result<Self> {
        let plan = VerifierPlan::validate(
            &fixture.key,
            fixture.profile.clone(),
            VerifierLimits::default(),
        )?;
        let plan_id = plan.identity(&fixture.schema)?;
        let (source, inputs) = plan.build(
            fixture.schema.clone(),
            ImplementationOptions {
                compact_blake3: true,
            },
        )?;
        let frontend = Frontend::compile(source, max_main_height, max_table_height)?;
        Ok(Self {
            plan,
            plan_id,
            inputs,
            frontend,
            key_bytes: fixture.key.to_bytes()?,
            loaded: None,
            setup,
        })
    }

    fn prove_with_expected(
        &mut self,
        fixture: &init_fri::Fixture,
        output: &Path,
        expected: &[u64; 18],
    ) -> storage::Result<()> {
        if fixture.public.map(|value| value.as_canonical_u64()) != *expected {
            return Err("request differs from the independently expected statement".into());
        }
        let candidate = VerifierPlan::validate(
            &fixture.key,
            fixture.profile.clone(),
            VerifierLimits::default(),
        )?;
        if candidate.identity(&fixture.schema)? != self.plan_id {
            return Err(
                "request uses a different verifier key, profile or constant statement".into(),
            );
        }
        worker_protocol::ensure_empty_output(output)?;
        fs::create_dir_all(output)?;
        fs::write(output.join("plan-id.bin"), self.plan_id)?;
        fs::write(output.join("inner-vk.bin"), &self.key_bytes)?;

        let expanded = self.plan.expand_witness(ProofEnvelope::Ordinary {
            proof: &fixture.proof,
            claims: &fixture.claims,
        })?;
        let mut witness = self.frontend.source().witness();
        self.inputs.assign_statement(
            &mut witness,
            &Statement {
                claims: fixture.claims.clone(),
                messages: vec![],
            },
        )?;
        expanded.assign_proof(&mut witness, &self.inputs)?;
        let source = witness.generate()?;
        if source.public_values() != fixture.public {
            return Err("source witness public values differ from the expected statement".into());
        }
        drop(expanded);
        let assignment = self.frontend.assign(&source)?;
        drop(source);
        let public: Vec<_> = fixture
            .public
            .iter()
            .map(|v| Scalar::from_u64(v.as_canonical_u64()))
            .collect();
        if assignment.public_values() != public {
            return Err("scalar witness public values differ from the expected statement".into());
        }
        let compiled = self.frontend.compiled();
        let srs_len = (0..compiled.num_circuits())
            .map(|i| compiled.trace_dimensions(i).unwrap().0)
            .max()
            .ok_or("empty compiled circuit")?;
        let setup_id = self.setup.identity(DEV_SEED, srs_len, 2)?;
        self.setup.bind_stage(output, &setup_id)?;
        let mut manifest = storage::Manifest {
            widths: vec![],
            heights: vec![],
            claims: compiled.claims(&public)?,
        };
        if let Some(loaded) = &self.loaded {
            manifest.widths.clone_from(&loaded.widths);
            manifest.heights.clone_from(&loaded.heights);
        } else {
            let cache_profile = [self.plan_id, setup_id].concat();
            let cache = fixed_cache::FixedCache::for_stage(output, &cache_profile)?;
            let cached = cache
                .as_ref()
                .map(|c| c.restore(output))
                .transpose()?
                .flatten();
            if let Some(shape) = cached {
                if shape.widths.len() != compiled.num_circuits()
                    || shape.heights.len() != compiled.num_circuits()
                    || (0..compiled.num_circuits()).any(|i| {
                        compiled.trace_dimensions(i) != Some((shape.heights[i], shape.widths[i]))
                    })
                {
                    return Err("cached trace dimensions differ from the compiled circuit".into());
                }
                manifest.widths = shape.widths;
                manifest.heights = shape.heights;
            } else {
                for i in 0..compiled.num_circuits() {
                    let (height, width) = compiled
                        .trace_dimensions(i)
                        .ok_or("missing compiled trace")?;
                    manifest.widths.push(width);
                    manifest.heights.push(height);
                }
            }
        }
        storage::save(&output.join("manifest.bin"), &manifest)?;
        let traces = compiled.trace_shards(&assignment)?;
        prove_with_state_with_expected(
            output,
            ProveMode::Fresh(&traces),
            &mut self.loaded,
            &self.setup,
            expected,
        )
    }

    fn quiesce(&mut self) -> storage::Result<serde_json::Value> {
        if let Some(data) = self
            .loaded
            .as_mut()
            .and_then(|loaded| loaded.key.preprocessed_data.as_mut())
        {
            data.release_device_residency();
        }
        Ok(serde_json::to_value(
            multi_stark::ark_adapter::pcs::KzgPcs::release_idle_device_memory(),
        )?)
    }
}

pub(super) fn run(args: &[String]) -> storage::Result<()> {
    let started = Instant::now();
    let expected = init_fri::expected_words()?;
    let first = init_fri::load_with_expected(Path::new(&args[0]), &expected)?;
    let mut worker = Worker::compile(&first)?;
    println!(
        "WORKER FRONTEND READY: startup_seconds={:.6}",
        started.elapsed().as_secs_f64()
    );
    let request = Instant::now();
    worker.prove_with_expected(&first, Path::new(&args[1]), &expected)?;
    println!(
        "WORKER REQUEST 0: seconds={:.6} loaded_key_reused=false",
        request.elapsed().as_secs_f64()
    );
    for (index, pair) in args[2..].chunks_exact(2).enumerate() {
        let request = Instant::now();
        let expected = init_fri::expected_words()?;
        let fixture = init_fri::load_with_expected(Path::new(&pair[0]), &expected)?;
        worker.prove_with_expected(&fixture, Path::new(&pair[1]), &expected)?;
        println!(
            "WORKER REQUEST {}: seconds={:.6} loaded_key_reused=true",
            index + 1,
            request.elapsed().as_secs_f64()
        );
    }
    println!(
        "WORKER COMPLETE: requests={} total_seconds={:.6}",
        args.len() / 2,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

pub(super) fn serve(path: &Path) -> storage::Result<()> {
    serve_with_setup(path, SetupSource::from_env)
}

pub(super) fn serve_development_public_degree(path: &Path) -> storage::Result<()> {
    serve_with_setup(path, SetupSource::development_public_degree_from_env)
}

fn serve_with_setup(
    path: &Path,
    load_setup: impl FnOnce() -> storage::Result<SetupSource>,
) -> storage::Result<()> {
    let mut protocol = worker_protocol::Protocol::open(path, "init_fri_kzg_prove")?;
    let setup = match load_setup() {
        Ok(setup) => setup,
        Err(error) => {
            protocol.failed(None, &error.to_string())?;
            return Err(error);
        }
    };
    serve_requests(
        &mut protocol,
        setup,
        1 << 24,
        1 << 22,
        |request, expected| init_fri::load_with_expected(&request.input, expected),
    )
}

fn serve_requests<R: BufRead>(
    protocol: &mut worker_protocol::Protocol<R>,
    setup: SetupSource,
    max_main_height: usize,
    max_table_height: usize,
    mut load: impl FnMut(&worker_protocol::Request, &[u64; 18]) -> storage::Result<init_fri::Fixture>,
) -> storage::Result<()> {
    let mut current = None;
    let result = (|| -> storage::Result<()> {
        let Some(request) = protocol.next_request()? else {
            return Ok(());
        };
        current = Some(request.id.clone());
        let started = Instant::now();
        worker_protocol::ensure_empty_output(&request.output)?;
        let expected = init_fri::expected_words_from_path(&request.expected_claims)?;
        let first = load(&request, &expected)?;
        let preparing = Instant::now();
        let mut worker =
            Worker::compile_with_layout(&first, max_main_height, max_table_height, setup)?;
        let preparation_seconds = preparing.elapsed().as_secs_f64();
        worker.prove_with_expected(&first, &request.output, &expected)?;
        let idle = worker.quiesce()?;
        protocol.proved(
            &request,
            false,
            false,
            preparation_seconds,
            started.elapsed().as_secs_f64(),
            idle,
        )?;
        loop {
            current = None;
            let Some(request) = protocol.next_request()? else {
                break;
            };
            current = Some(request.id.clone());
            let started = Instant::now();
            worker_protocol::ensure_empty_output(&request.output)?;
            let expected = init_fri::expected_words_from_path(&request.expected_claims)?;
            let fixture = load(&request, &expected)?;
            let loaded_key_reused = worker.loaded.is_some();
            worker.prove_with_expected(&fixture, &request.output, &expected)?;
            drop(fixture);
            let idle = worker.quiesce()?;
            protocol.proved(
                &request,
                true,
                loaded_key_reused,
                0.0,
                started.elapsed().as_secs_f64(),
                idle,
            )?;
        }
        Ok(())
    })();
    if let Err(error) = &result {
        protocol.failed(current.as_deref(), &error.to_string())?;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use multi_stark::{
        plonkish::{
            CircuitBuilder,
            verifier::{Envelope, ProofProfile, StatementSlot, VerifierKey},
        },
        system::{System, SystemWitness},
        types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config},
    };

    #[test]
    fn worker_reuses_compilation_and_keys_for_fresh_proofs() -> storage::Result<()> {
        exercise_service(SetupSource::Development { cache: None })
    }

    #[test]
    fn development_public_degree_worker_reuses_compilation_and_keys_for_fresh_proofs()
    -> storage::Result<()> {
        exercise_service(SetupSource::KnownTrapdoorPublicDegree { cache: None })
    }

    #[test]
    #[ignore = "isolated comparison of rebuilding and retaining frontend/prover state"]
    fn retained_worker_benchmark() -> storage::Result<()> {
        exercise_worker(2)
    }

    fn exercise_service(setup: SetupSource) -> storage::Result<()> {
        let root = std::env::temp_dir().join(format!(
            "init-kzg-service-{}-{}",
            std::process::id(),
            setup.is_diagnostic(),
        ));
        fs::create_dir(&root)?;
        let mut fixtures = fixtures(&[0, 1, 0, 1, 1], true)?.into_iter();
        let mut served: std::collections::VecDeque<_> = fixtures.by_ref().take(3).collect();
        let fresh = fixtures.next().unwrap();
        let mut incompatible = fixtures.next().unwrap();
        let mut records = Vec::new();
        for (id, fixture) in ["a", "b", "a-again"].into_iter().zip(&served) {
            let expected = root.join(format!("{id}.claims"));
            fs::write(
                &expected,
                [1, 18]
                    .into_iter()
                    .chain(fixture.public.map(|value| value.as_canonical_u64()))
                    .flat_map(u64::to_le_bytes)
                    .collect::<Vec<_>>(),
            )?;
            records.push(serde_json::json!({
                "id": id, "input": id, "output": root.join(id), "expected_claims": expected,
            }));
        }
        records.push(serde_json::json!({
            "id": "duplicate-output", "input": "must-not-load", "output": root.join("a"),
            "expected_claims": root.join("missing-claims"),
        }));
        let bytes = records
            .iter()
            .map(|record| format!("{record}\n"))
            .collect::<String>();
        let responses = root.join("responses.jsonl");
        let mut protocol = worker_protocol::Protocol::with_reader(
            &responses,
            "init_fri_kzg_prove",
            std::io::Cursor::new(bytes),
        )?;
        let error = serve_requests(
            &mut protocol,
            setup.clone(),
            1 << 16,
            1 << 22,
            |request, expected| {
                let fixture = served.pop_front().expect("only three inputs may load");
                assert_eq!(
                    fixture.public.map(|value| value.as_canonical_u64()),
                    *expected
                );
                fs::remove_file(&request.expected_claims)?;
                Ok(fixture)
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("worker output must be empty"));
        assert!(served.is_empty());
        let events: Vec<serde_json::Value> = fs::read_to_string(&responses)?
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        assert_eq!(events.len(), 9);
        assert_eq!(events[0]["status"], "listening");
        assert_eq!(events[0]["prepared"], false);
        for (index, id) in ["a", "b", "a-again"].into_iter().enumerate() {
            let started = &events[2 * index + 1];
            let proved = &events[2 * index + 2];
            assert_eq!(started["id"], id);
            assert_eq!(started["status"], "request_started");
            assert_eq!(started["prepared"], index != 0);
            assert_eq!(proved["id"], id);
            assert_eq!(proved["status"], "proved");
            assert_eq!(proved["prepared"], true);
            assert_eq!(proved["frontend_reused"], index != 0);
            assert_eq!(proved["loaded_key_reused"], index != 0);
            assert!(proved["idle"]["quiesced"].is_boolean());
            if proved["idle"]["initialized"] == true {
                assert_eq!(proved["idle"]["quiesced"], true);
                for device in proved["idle"]["devices"].as_array().unwrap() {
                    assert_eq!(device["after"]["resident_coefficient_bytes"], 0);
                }
            }
            println!("served_worker_sample {proved}");
        }
        assert_eq!(events[8]["id"], "duplicate-output");
        assert_eq!(events[8]["status"], "failed");
        assert_eq!(events[8]["prepared"], true);

        let initial_artifacts = read_artifacts(&root.join("a"))?;
        let second_artifacts = read_artifacts(&root.join("b"))?;
        assert_eq!(initial_artifacts, read_artifacts(&root.join("a-again"))?);
        assert_ne!(initial_artifacts[0], second_artifacts[0]);
        assert_ne!(initial_artifacts[2], second_artifacts[2]);
        for file in ["plan-id.bin", "kzg-setup-id.bin"] {
            assert_eq!(
                fs::read(root.join("a").join(file))?,
                fs::read(root.join("b").join(file))?
            );
        }

        let expected = fresh.public.map(|value| value.as_canonical_u64());
        let mut rebuilt = Worker::compile_with_layout(&fresh, 1 << 16, 1 << 22, setup.clone())?;
        rebuilt.prove_with_expected(&fresh, &root.join("fresh-b"), &expected)?;
        rebuilt.quiesce()?;
        assert_eq!(second_artifacts, read_artifacts(&root.join("fresh-b"))?);
        for id in ["a", "fresh-b"] {
            for index in 0..rebuilt.frontend.compiled().num_circuits() {
                assert!(!root.join(id).join(format!("{index}.fixed.zst")).exists());
                assert!(!root.join(id).join(format!("{index}.witness.zst")).exists());
                assert!(!root.join(id).join(format!("kzg/main-{index}.bin")).exists());
                assert!(root.join(id).join(format!("{index}.meta")).exists());
                assert!(
                    root.join(id)
                        .join(format!("kzg/setup-{index}.bin"))
                        .exists()
                );
                assert!(
                    root.join(id)
                        .join(format!("kzg/fixed-{index}.bin"))
                        .exists()
                );
            }
        }
        prove_with_state_with_expected(
            &root.join("b"),
            ProveMode::Verify,
            &mut None,
            &rebuilt.setup,
            &expected,
        )?;
        for report in ["prove-report.json", "verify-report.json"] {
            let evidence: serde_json::Value =
                serde_json::from_slice(&fs::read(root.join("b/kzg").join(report))?)?;
            assert_eq!(evidence["setup"], "development");
            assert_eq!(evidence["development_srs"], true);
            assert_eq!(evidence["known_trapdoor"], true);
            assert_eq!(evidence["filecoin_acceptance"], false);
            assert!(evidence["ceremony_id"].is_null());
            assert!(evidence["filecoin_manifest_digest"].is_null());
            assert_eq!(evidence["native_verification_passed"], true);
            assert_eq!(evidence["negative_tests_pass"], true);
            if setup.is_diagnostic() {
                let config = rebuilt.loaded.as_ref().unwrap().system.config.clone();
                assert!(config.transcript_seed().starts_with(b"multi-stark/kzg/v4"));
                assert_eq!(evidence["public_max_degree"], (1u64 << 28) - 2);
                assert_eq!(
                    evidence["public_setup_id"],
                    blake3::Hash::from_bytes(config.srs().public_setup().unwrap().id).to_string()
                );
                assert!(config.srs().degree_keys.is_empty());
            } else {
                assert!(evidence["public_max_degree"].is_null());
                assert!(evidence["public_setup_id"].is_null());
            }
        }

        let mut wrong = expected;
        wrong[1] += 1;
        assert!(
            prove_with_state_with_expected(
                &root.join("b"),
                ProveMode::Verify,
                &mut None,
                &setup,
                &wrong,
            )
            .unwrap_err()
            .to_string()
            .contains("staged claims differ")
        );
        assert!(
            rebuilt
                .prove_with_expected(&fresh, &root.join("wrong-statement"), &wrong)
                .unwrap_err()
                .to_string()
                .contains("independently expected statement")
        );
        assert!(!root.join("wrong-statement").exists());
        incompatible.schema.claims[0][0] = StatementSlot::Constant(Val::from_u8(108));
        assert!(
            rebuilt
                .prove_with_expected(&incompatible, &root.join("wrong-profile"), &expected)
                .unwrap_err()
                .to_string()
                .contains("different verifier key, profile or constant")
        );
        assert!(!root.join("wrong-profile").exists());
        for id in ["b", "a-again"] {
            for index in 0..rebuilt.frontend.compiled().num_circuits() {
                for file in [
                    format!("{index}.fixed.zst"),
                    format!("{index}.witness.zst"),
                    format!("kzg/fixed-{index}.bin"),
                    format!("kzg/main-{index}.bin"),
                ] {
                    assert!(!root.join(id).join(file).exists());
                }
            }
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }

    fn read_artifacts(dir: &Path) -> storage::Result<Vec<Vec<u8>>> {
        Ok(["proof.compact.bin", "packet.bin", "profile-id.bin"]
            .into_iter()
            .map(|name| fs::read(dir.join("kzg").join(name)))
            .collect::<Result<Vec<_>, _>>()?)
    }

    #[test]
    fn empty_input_never_prepares_or_loads_a_worker() -> storage::Result<()> {
        let path = std::env::temp_dir().join(format!("init-kzg-worker-eof-{}", std::process::id()));
        let mut protocol = worker_protocol::Protocol::with_reader(
            &path,
            "init_fri_kzg_prove",
            std::io::Cursor::new(Vec::<u8>::new()),
        )?;
        serve_requests(
            &mut protocol,
            SetupSource::Development { cache: None },
            1 << 16,
            1 << 22,
            |_, _| panic!("empty input must not load a fixture"),
        )?;
        let lines = fs::read_to_string(&path)?;
        assert_eq!(lines.lines().count(), 1);
        let response: serde_json::Value = serde_json::from_str(lines.trim())?;
        assert_eq!(response["status"], "listening");
        assert_eq!(response["prepared"], false);
        fs::remove_file(path)?;
        Ok(())
    }

    fn fixtures(values: &[u8], vary_public: bool) -> storage::Result<Vec<init_fri::Fixture>> {
        let mut builder = CircuitBuilder::<Val>::new();
        let bit = builder.input("private bit");
        builder.assert_bool(bit);
        let public_inputs: Vec<_> = init_fri::INIT_PUBLIC_WORDS
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let input = vary_public.then(|| builder.input(format!("public {index}")));
                let wire = input.unwrap_or_else(|| builder.constant(Val::from_u64(value)));
                builder.expose_public(wire);
                input
            })
            .collect();
        let inner = builder
            .finish()
            .lower_to_multi_stark_with_max_height(Val::from_u8(107), 8)?;
        let config = GoldilocksBlake3Config::new(
            CommitmentParameters {
                log_blowup: 1,
                cap_height: 0,
            },
            FriParameters {
                log_final_poly_len: 0,
                max_log_arity: 1,
                num_queries: 2,
                commit_proof_of_work_bits: 0,
                query_proof_of_work_bits: 0,
            },
        );
        let (system, key) = System::new(config, inner.circuit_inputs());
        values
            .iter()
            .map(|&value| -> storage::Result<init_fri::Fixture> {
                let mut public = init_fri::INIT_PUBLIC_WORDS.map(Val::from_u64);
                if vary_public {
                    public[1] += Val::from_u8(value);
                }
                let mut witness = inner.witness();
                witness.set(bit, Val::from_u8(value))?;
                for (input, value) in public_inputs.iter().zip(public) {
                    if let Some(input) = input {
                        witness.set(*input, value)?;
                    }
                }
                let assignment = witness.generate()?;
                let claims = inner.claims(&public)?;
                let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
                let proof = system.prove_multiple_claims(
                    &key,
                    &refs,
                    SystemWitness::from_stage_1(inner.traces(&assignment)?, &system),
                );
                system
                    .verify_multiple_claims(&refs, &proof)
                    .map_err(|e| format!("inner proof: {e:?}"))?;
                let profile = ProofProfile {
                    envelope: Envelope::Ordinary,
                    active: proof.active.clone(),
                    log_degrees: proof.log_degrees.clone(),
                    claim_lengths: claims.iter().map(Vec::len).collect(),
                    message_lengths: vec![],
                    max_field_retries: 2,
                };
                let mut schema = Statement {
                    claims: claims
                        .iter()
                        .map(|claim| {
                            claim
                                .iter()
                                .copied()
                                .map(StatementSlot::Constant)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                    messages: vec![],
                };
                for claim in &mut schema.claims[1..19] {
                    claim[3] = StatementSlot::Public;
                }
                Ok(init_fri::Fixture {
                    key: VerifierKey::from_system(&system),
                    proof,
                    public,
                    claims,
                    profile,
                    schema,
                })
            })
            .collect()
    }

    fn exercise_worker(iterations: usize) -> storage::Result<()> {
        let root = std::env::temp_dir().join(format!(
            "init-kzg-worker-{}-{iterations}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let mut fixtures = fixtures(&[0, 1, 1], false)?.into_iter();
        let first = fixtures.next().unwrap();
        let second = fixtures.next().unwrap();
        let expected = init_fri::INIT_PUBLIC_WORDS;
        let started = Instant::now();
        let mut worker = Worker::compile_with_layout(
            &first,
            1 << 16,
            1 << 22,
            SetupSource::Development { cache: None },
        )?;
        let compile_seconds = started.elapsed().as_secs_f64();
        let compiled = worker.frontend.compiled();
        let dimensions: Vec<_> = (0..compiled.num_circuits())
            .map(|i| compiled.trace_dimensions(i).unwrap())
            .collect();
        assert!(dimensions.iter().all(|&(height, _)| height <= 1 << 18));
        println!(
            "worker_fixture compile_seconds={compile_seconds:.6} source_gates={} scalar_gates={} dimensions={dimensions:?}",
            worker.frontend.source().stats().gates,
            compiled.circuit().stats().gates
        );

        let started = Instant::now();
        let initial = root.join("initial");
        worker.prove_with_expected(&first, &initial, &expected)?;
        worker.quiesce()?;
        println!(
            "worker_startup first_request_seconds={:.6}",
            started.elapsed().as_secs_f64()
        );
        let read_artifacts = |dir: &Path| -> storage::Result<Vec<Vec<u8>>> {
            Ok(["proof.compact.bin", "packet.bin", "profile-id.bin"]
                .into_iter()
                .map(|name| fs::read(dir.join("kzg").join(name)))
                .collect::<Result<Vec<_>, _>>()?)
        };
        let initial_artifacts = read_artifacts(&initial)?;
        for iteration in 0..iterations {
            let fixture = if iteration % 2 == 0 { &second } else { &first };
            let mut times = [0.0; 2];
            let mut artifacts: [Vec<Vec<u8>>; 2] = std::array::from_fn(|_| vec![]);
            let order = if iteration % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for reuse in order {
                let output = root.join(format!("comparison-{iteration}-{reuse}"));
                let started = Instant::now();
                if reuse {
                    worker.prove_with_expected(fixture, &output, &expected)?;
                    worker.quiesce()?;
                } else {
                    let mut rebuilt = Worker::compile_with_layout(
                        fixture,
                        1 << 16,
                        1 << 22,
                        SetupSource::Development { cache: None },
                    )?;
                    rebuilt.prove_with_expected(fixture, &output, &expected)?;
                    rebuilt.quiesce()?;
                }
                times[usize::from(reuse)] = started.elapsed().as_secs_f64();
                artifacts[usize::from(reuse)] = read_artifacts(&output)?;
                if reuse {
                    for index in 0..dimensions.len() {
                        for file in [
                            format!("{index}.fixed.zst"),
                            format!("{index}.witness.zst"),
                            format!("kzg/fixed-{index}.bin"),
                            format!("kzg/main-{index}.bin"),
                        ] {
                            assert!(!output.join(file).exists());
                        }
                    }
                    if iteration == 0 {
                        prove_with_state_with_expected(
                            &output,
                            ProveMode::Verify,
                            &mut None,
                            &SetupSource::Development { cache: None },
                            &expected,
                        )?;
                    }
                }
            }
            assert_eq!(artifacts[0], artifacts[1]);
            if iteration % 2 == 0 {
                assert_ne!(initial_artifacts[0], artifacts[1][0]);
            } else {
                assert_eq!(initial_artifacts, artifacts[1]);
            }
            println!(
                "retained_worker_sample iteration={iteration} rebuilt_seconds={:.6} retained_seconds={:.6} proof_digest={} packet_digest={} profile_digest={}",
                times[0],
                times[1],
                ::blake3::hash(&artifacts[0][0]).to_hex(),
                ::blake3::hash(&artifacts[0][1]).to_hex(),
                ::blake3::hash(&artifacts[0][2]).to_hex()
            );
        }

        let mut incompatible = fixtures.next().unwrap();
        incompatible.schema.claims[0][0] = StatementSlot::Constant(Val::from_u8(108));
        assert!(
            worker
                .prove_with_expected(&incompatible, &root.join("wrong-schema"), &expected)
                .unwrap_err()
                .to_string()
                .contains("different verifier key, profile or constant")
        );
        assert!(!root.join("wrong-schema").exists());

        let output = root.join("wrong-profile");
        fs::create_dir(&output)?;
        fs::copy(initial.join("manifest.bin"), output.join("manifest.bin"))?;
        fs::write(output.join("plan-id.bin"), [255u8; 32])?;
        assert!(
            prove_with_state_with_expected(
                &output,
                ProveMode::Verify,
                &mut worker.loaded,
                &worker.setup,
                &expected,
            )
            .unwrap_err()
            .to_string()
            .contains("loaded prover profile")
        );
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
