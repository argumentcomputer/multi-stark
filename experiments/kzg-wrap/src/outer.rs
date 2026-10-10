use crate::native_verifier::Built;
use multi_stark::{
    ark_adapter::{KzgConfig, Scalar, compact::FixedProofCodec},
    plonkish::{Assignment, MultiStarkCircuit},
    system::{ProverKey, System},
    traits::Field,
};
use p3_matrix::Matrix;
use std::{fs, path::Path};
#[path = "../../../examples/support/kzg_fixed_cache.rs"]
mod fixed_cache;
pub(crate) mod parameters;
#[path = "../../../examples/support/kzg_setup.rs"]
pub(crate) mod setup;
#[allow(dead_code)]
#[path = "../../../examples/support/kzg_storage.rs"]
mod storage;
#[derive(serde::Serialize, serde::Deserialize)]
struct Pairings {
    degree_count: usize,
    keys: Vec<u8>,
}

#[cfg(test)]
fn stage_with_setup(
    built: Built,
    assignment: Assignment<Scalar>,
    dir: &Path,
    profile: &[u8],
    fused: bool,
    setup: &setup::SetupSource,
) -> storage::Result<()> {
    let expected = init_claim::expected_words()?;
    stage_with_expected(built, assignment, dir, profile, fused, setup, &expected)
}

pub(crate) fn stage_with_expected(
    built: Built,
    assignment: Assignment<Scalar>,
    dir: &Path,
    profile: &[u8],
    fused: bool,
    setup: &setup::SetupSource,
    expected: &[u64; 18],
) -> storage::Result<()> {
    let max_height = if setup.uses_public_degree() {
        1 << 27
    } else {
        1 << 29
    };
    let mut frontend = Frontend::new(
        built.circuit,
        &built.pairing_keys,
        built.degree_output_count,
        profile,
        max_height,
    )?;
    frontend.stage(&assignment, dir, fused, setup, expected)
}

pub(crate) struct Frontend {
    compiled: MultiStarkCircuit<Scalar>,
    pairings: Pairings,
    profile: Vec<u8>,
    loaded: Option<LoadedProver>,
}

impl Frontend {
    pub(crate) fn new(
        circuit: multi_stark::plonkish::Circuit<Scalar>,
        keys: &[ark_bls12_381::G2Affine],
        degree_count: usize,
        profile: &[u8],
        max_height: usize,
    ) -> storage::Result<Self> {
        use ark_serialize::CanonicalSerialize;
        if keys.len().checked_sub(2) != Some(degree_count) {
            return Err("invalid recursive pairing profile".into());
        }
        let height = circuit.multi_stark_layout()?.main_height;
        if height > max_height {
            return Err("recursive computation exceeds its single-trace height cap".into());
        }
        let mut encoded = vec![];
        keys.serialize_compressed(&mut encoded)?;
        eprintln!("Lowering recursive circuit into one {height}-row computation trace");
        let compiled = circuit
            .lower_to_multi_stark_sharded(Scalar::from_u8(94), height)?
            .merge_table_traces(height)?;
        if compiled.num_circuits() != 2 || compiled.main_heights() != [height] {
            return Err("expected one computation trace and one merged table".into());
        }
        Ok(Self {
            compiled,
            pairings: Pairings {
                degree_count,
                keys: encoded,
            },
            profile: profile.to_vec(),
            loaded: None,
        })
    }

    pub(crate) fn witness(&self) -> multi_stark::plonkish::Witness<'_, Scalar> {
        self.compiled.witness()
    }

    pub(crate) fn has_loaded_key(&self) -> bool {
        self.loaded.is_some()
    }

    pub(crate) fn release_device_memory(
        &mut self,
    ) -> multi_stark::ark_adapter::pcs::KzgIdleMemoryRelease {
        if let Some(loaded) = &mut self.loaded
            && let Some(data) = &mut loaded.key.preprocessed_data
        {
            data.release_device_residency();
        }
        multi_stark::ark_adapter::KzgPcs::release_idle_device_memory()
    }

    pub(crate) fn stage(
        &mut self,
        assignment: &Assignment<Scalar>,
        dir: &Path,
        fused: bool,
        setup: &setup::SetupSource,
        expected: &[u64; 18],
    ) -> storage::Result<()> {
        self.stage_with_parameters(
            assignment,
            dir,
            fused,
            &parameters::Parameters::Selected(setup),
            expected,
        )
    }

    pub(crate) fn stage_with_parameters(
        &mut self,
        assignment: &Assignment<Scalar>,
        dir: &Path,
        fused: bool,
        parameters: &parameters::Parameters<'_>,
        expected: &[u64; 18],
    ) -> storage::Result<()> {
        if dir.join("manifest.bin").exists() {
            return Err("already staged".into());
        }
        if assignment.public_values().len() != 18 + 2 * (self.pairings.degree_count + 2)
            || assignment.public_values()[..18]
                .iter()
                .zip(expected)
                .any(|(&value, &word)| value != Scalar::from_u64(word))
        {
            return Err(
                "recursive assignment differs from the independently expected statement".into(),
            );
        }
        let start = std::time::Instant::now();
        let height = self.compiled.main_height();
        parameters.check_height(height)?;
        let setup_id = parameters.identity(height, 2)?;
        let traces = self.compiled.trace_shards(assignment)?;
        fs::create_dir_all(dir)?;
        parameters.bind_stage(dir, &setup_id)?;
        fs::write(
            dir.join("frontend-id.bin"),
            blake3::hash(&self.profile).as_bytes(),
        )?;
        storage::save(&dir.join("pairings.bin"), &self.pairings)?;
        let mut manifest = storage::Manifest {
            widths: vec![],
            heights: vec![],
            claims: self.compiled.claims(assignment.public_values())?,
        };
        if let Some(loaded) = &self.loaded {
            if !fused {
                return Err("retained recursive requests require fused proving".into());
            }
            manifest.widths.clone_from(&loaded.widths);
            manifest.heights.clone_from(&loaded.heights);
        } else {
            let cache_profile = [self.profile.as_slice(), &setup_id].concat();
            let cache = fixed_cache::FixedCache::for_stage(dir, &cache_profile)?;
            let cached = cache
                .as_ref()
                .map(|cache| cache.restore(dir))
                .transpose()?
                .flatten();
            if cached.as_ref().is_some_and(|shape| {
                shape.widths.len() != self.compiled.num_circuits()
                    || shape.heights.len() != self.compiled.num_circuits()
            }) {
                return Err("cached trace count differs from the compiled circuit".into());
            }
            for i in 0..self.compiled.num_circuits() {
                let (expected_height, expected_width) = self
                    .compiled
                    .trace_dimensions(i)
                    .ok_or("missing compiled trace")?;
                if let Some(shape) = &cached {
                    if shape.widths[i] != expected_width || shape.heights[i] != expected_height {
                        return Err(
                            "cached fixed trace dimensions differ from the compiled circuit".into(),
                        );
                    }
                    manifest.widths.push(shape.widths[i]);
                    manifest.heights.push(shape.heights[i]);
                } else if fused {
                    manifest.widths.push(expected_width);
                    manifest.heights.push(expected_height);
                } else {
                    let mut definition = self
                        .compiled
                        .kzg_circuit_input_with_degree_policy(
                            i,
                            height,
                            2,
                            !parameters.uses_public_degree(),
                        )?
                        .ok_or("missing circuit")?;
                    let fixed = definition
                        .preprocessed
                        .take()
                        .ok_or("missing preprocessing")?;
                    manifest.widths.push(definition.main_width);
                    manifest.heights.push(fixed.height());
                    storage::save(&dir.join(format!("{i}.meta")), &definition)?;
                    storage::write_matrix(&dir.join(format!("{i}.fixed.zst")), &fixed)?;
                }
                if !fused {
                    let trace = traces.trace(i)?;
                    if trace.width() != manifest.widths[i] || trace.height() != manifest.heights[i]
                    {
                        return Err("cached fixed trace and witness have different shapes".into());
                    }
                    storage::write_matrix(&dir.join(format!("{i}.witness.zst")), &trace)?;
                }
                eprintln!("Staged recursive trace {i}: {:?}", start.elapsed());
            }
        }
        storage::save(&dir.join("manifest.bin"), &manifest)?;
        fs::write(dir.join("SECURITY.txt"), parameters.security_description())?;
        eprintln!("Recursive staging complete: {:?}", start.elapsed());
        if fused {
            prove_with_parameters(
                dir,
                false,
                Some(&traces),
                &mut self.loaded,
                parameters,
                expected,
            )?;
            eprintln!("Recursive stage-and-prove complete: {:?}", start.elapsed());
        }
        Ok(())
    }
}

const OUTER_SEED: &[u8] = b"init-kzg-recursive-v1";
use crate::init_claim;

#[cfg(test)]
#[path = "outer/worker_tests.rs"]
mod worker_tests;

fn pairing_keys(dir: &Path) -> storage::Result<(usize, Vec<ark_bls12_381::G2Affine>)> {
    use ark_serialize::CanonicalDeserialize;
    let profile: Pairings = storage::load(&dir.join("pairings.bin"))?;
    let mut input = profile.keys.as_slice();
    let keys = Vec::<ark_bls12_381::G2Affine>::deserialize_compressed(&mut input)?;
    if !input.is_empty()
        || keys.len() != profile.degree_count + 2
        || keys.iter().any(|point| !point.is_on_curve())
    {
        return Err("invalid pairing profile".into());
    }
    Ok((profile.degree_count, keys))
}
fn pairing_bytes(publics: &[Scalar]) -> storage::Result<Vec<u8>> {
    let mut bytes = vec![];
    for value in publics {
        let full: Vec<_> = value
            .canonical_limbs_le()
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        if full[24..].iter().any(|&v| v != 0) {
            return Err("noncanonical packed point".into());
        }
        bytes.extend_from_slice(&full[..24]);
    }
    Ok(bytes)
}
fn check_pairings(
    bytes: &[u8],
    degree_count: usize,
    keys: &[ark_bls12_381::G2Affine],
) -> storage::Result<()> {
    use ark_ec::pairing::Pairing;
    use ark_ff::Zero;
    use ark_serialize::CanonicalDeserialize;
    if bytes.len() != keys.len() * 48 || keys.len() != degree_count + 2 {
        return Err("pairing point count".into());
    }
    let points = bytes
        .as_chunks::<48>()
        .0
        .iter()
        .map(|bytes| ark_bls12_381::G1Affine::deserialize_compressed(bytes.as_slice()))
        .collect::<Result<Vec<_>, _>>()?;
    if points.iter().any(|point| !point.is_on_curve()) {
        return Err("invalid external pairing point".into());
    }
    for range in [0..degree_count, degree_count..keys.len()] {
        if !ark_bls12_381::Bls12_381::multi_pairing(
            points[range.clone()].to_vec(),
            keys[range].to_vec(),
        )
        .is_zero()
        {
            return Err("external pairing equation failed".into());
        }
    }
    Ok(())
}
fn packet_claims(
    packet: &[u8],
    profile: &[u8; 32],
    pairing_count: usize,
    expected: &[u64; 18],
) -> storage::Result<(Vec<Vec<Scalar>>, usize)> {
    use multi_stark::traits::Algebra;
    let offset = 32 + 18 * 8 + pairing_count * 48;
    if packet.len() < offset || packet[..32] != *profile {
        return Err("packet profile mismatch".into());
    }
    let mut publics = vec![];
    for (bytes, expected) in packet[32..32 + 18 * 8]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(*expected)
    {
        let word = u64::from_le_bytes(*bytes);
        if word != expected {
            return Err("unexpected Init claim".into());
        }
        publics.push(Scalar::from_u64(word));
    }
    for bytes in packet[32 + 18 * 8..offset].as_chunks::<24>().0.iter() {
        let mut limbs = [0u64; 4];
        for (i, chunk) in bytes.as_chunks::<8>().0.iter().enumerate() {
            limbs[i] = u64::from_le_bytes(*chunk);
        }
        publics.push(Scalar::from_limbs_le(limbs));
    }
    let claims = std::iter::once(Scalar::ZERO)
        .chain(publics)
        .enumerate()
        .map(|(i, value)| {
            vec![
                Scalar::from_u8(94),
                Scalar::ONE,
                Scalar::from_usize(i),
                value,
            ]
        })
        .collect();
    Ok((claims, offset))
}
fn verify_packet(
    system: &multi_stark::system::System<multi_stark::ark_adapter::KzgConfig>,
    codec: &multi_stark::ark_adapter::compact::FixedProofCodec,
    packet: &[u8],
    profile: &[u8; 32],
    degree_count: usize,
    keys: &[ark_bls12_381::G2Affine],
    expected: &[u64; 18],
) -> storage::Result<()> {
    let (claims, offset) = packet_claims(packet, profile, keys.len(), expected)?;
    let proof = codec.decode(&packet[offset..])?;
    system
        .verify_multiple_claims(
            &claims.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &proof,
        )
        .map_err(|e| format!("outer KZG proof: {e:?}"))?;
    check_pairings(&packet[32 + 18 * 8..offset], degree_count, keys)
}

pub fn prove(dir: &Path, verify_only: bool) -> storage::Result<()> {
    prove_with_traces(dir, verify_only, None, &setup::SetupSource::from_env()?)
}

fn prove_with_traces(
    dir: &Path,
    verify_only: bool,
    traces: Option<&multi_stark::plonkish::TraceShards<'_, Scalar>>,
    setup: &setup::SetupSource,
) -> storage::Result<()> {
    let expected = init_claim::expected_words()?;
    prove_with_state(dir, verify_only, traces, &mut None, setup, &expected)
}

struct LoadedProver {
    system: System<KzgConfig>,
    key: ProverKey<KzgConfig>,
    codec: FixedProofCodec,
    widths: Vec<usize>,
    heights: Vec<usize>,
    logs: Vec<u8>,
    setup_id: [u8; 32],
    frontend_id: Option<[u8; 32]>,
    pairing_metadata: Vec<u8>,
    setup_bytes: Vec<Vec<u8>>,
}

fn prove_with_state(
    dir: &Path,
    verify_only: bool,
    traces: Option<&multi_stark::plonkish::TraceShards<'_, Scalar>>,
    retained: &mut Option<LoadedProver>,
    setup: &setup::SetupSource,
    expected: &[u64; 18],
) -> storage::Result<()> {
    prove_with_parameters(
        dir,
        verify_only,
        traces,
        retained,
        &parameters::Parameters::Selected(setup),
        expected,
    )
}

pub(crate) fn verify_with_parameters(
    dir: &Path,
    parameters: &parameters::Parameters<'_>,
    expected: &[u64; 18],
) -> storage::Result<()> {
    prove_with_parameters(dir, true, None, &mut None, parameters, expected)
}

fn prove_with_parameters(
    dir: &Path,
    verify_only: bool,
    traces: Option<&multi_stark::plonkish::TraceShards<'_, Scalar>>,
    retained: &mut Option<LoadedProver>,
    parameters: &parameters::Parameters<'_>,
    expected: &[u64; 18],
) -> storage::Result<()> {
    use multi_stark::{
        ark_adapter::{KzgCommitment, compact::FixedProofCodec, pcs::KzgProverData},
        config::ProofConfig,
        lookup::LookupValues,
        prover::Stage1,
        system::{Circuit, CircuitInputs, ProverKey, System},
        traits::{Algebra, Pcs},
    };
    use std::{
        fs::File,
        io::{BufReader, BufWriter, Write},
        time::Instant,
    };
    let _ = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing_subscriber::filter::LevelFilter::INFO)
        .try_init();
    let start = Instant::now();
    let cold = retained.is_none();
    let manifest: storage::Manifest = storage::load(&dir.join("manifest.bin"))?;
    if manifest.heights.len() != 2
        || manifest.widths.len() != 2
        || manifest
            .heights
            .iter()
            .any(|height| *height < 2 || !height.is_power_of_two())
        || manifest.widths.contains(&0)
        || manifest.heights[0] > 1 << 29
        || manifest.heights[1] > manifest.heights[0]
    {
        return Err("expected one computation trace and one merged table".into());
    }
    let count = manifest.heights.len();
    let height = manifest.heights[0];
    if let Some(traces) = traces {
        let compiled = traces.compiled();
        if compiled.num_circuits() != count
            || compiled.main_heights() != [height]
            || (0..count).any(|i| {
                compiled.trace_dimensions(i) != Some((manifest.heights[i], manifest.widths[i]))
            })
        {
            return Err("compiled trace dimensions differ from the staged manifest".into());
        }
    }
    let setup_id = parameters.identity(height, 2)?;
    parameters.check_binding(dir, &setup_id)?;
    let frontend_id = match fs::read(dir.join("frontend-id.bin")) {
        Ok(bytes) => Some(
            bytes
                .try_into()
                .map_err(|_| "invalid recursive frontend identity")?,
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let pairing_metadata = fs::read(dir.join("pairings.bin"))?;
    if let Some(loaded) = retained.as_ref() {
        if loaded.setup_id != setup_id
            || loaded.frontend_id.is_none()
            || loaded.frontend_id != frontend_id
            || loaded.widths != manifest.widths
            || loaded.heights != manifest.heights
            || loaded.pairing_metadata != pairing_metadata
        {
            return Err("request differs from the retained recursive prover profile".into());
        }
        if verify_only || traces.is_none() || loaded.key.preprocessed_data.is_none() {
            return Err("retained recursive proving requires a fresh trace assignment".into());
        }
    }
    let (degree_count, keys) = pairing_keys(dir)?;
    if parameters.uses_public_degree() && degree_count != 0 {
        return Err("public-degree v4 requires exactly the two opening pairing points".into());
    }
    if manifest.claims.len() != 19 + 2 * keys.len()
        || manifest.claims.iter().any(|claim| claim.len() != 4)
    {
        return Err("wrong recursive public profile".into());
    }
    let publics: Vec<_> = manifest.claims[1..].iter().map(|row| row[3]).collect();
    let point_bytes = pairing_bytes(&publics[18..])?;
    check_pairings(&point_bytes, degree_count, &keys)?;
    for (&value, &word) in publics[..18].iter().zip(expected) {
        if value != Scalar::from_u64(word) {
            return Err("wrong Init claim".into());
        }
    }
    let out = dir.join("kzg");
    fs::create_dir_all(&out)?;
    fs::write(out.join("SECURITY.txt"), parameters.security_description())?;
    eprintln!("Preparing KZG parameters for 2^{} rows", height.ilog2());
    let mut srs_config_load_seconds = 0.0;
    let config = if let Some(loaded) = retained.as_ref() {
        for (i, bytes) in loaded.setup_bytes.iter().enumerate() {
            fs::write(out.join(format!("setup-{i}.bin")), bytes)?;
        }
        eprintln!("Reusing recursive SRS, fixed coefficients and compiled prover");
        loaded.system.config.clone()
    } else {
        let config_started = Instant::now();
        let config = parameters
            .config(height, 2, verify_only)?
            .with_streaming_lookups()
            .with_streaming_quotient();
        srs_config_load_seconds = config_started.elapsed().as_secs_f64();
        config
    };
    if parameters.uses_public_degree() && keys != [config.srs().g2, config.srs().tau_g2] {
        return Err("inner pairing keys differ from the selected public-degree parameters".into());
    }
    eprintln!("Outer KZG parameters ready: {:?}", start.elapsed());
    let checkpoint = |path: &Path, data: &KzgProverData| -> storage::Result<()> {
        let temp = path.with_extension("partial");
        let mut writer = BufWriter::with_capacity(1 << 20, File::create(&temp)?);
        data.write_checkpoint(&mut writer)?;
        writer.flush()?;
        drop(writer);
        fs::rename(temp, path)?;
        Ok(())
    };
    let mut circuits = vec![];
    let mut commits = vec![];
    let mut fixed_parts = vec![];
    let mut main_parts = vec![];
    let mut setup_bytes = vec![];
    for i in 0..count {
        let meta = out.join(format!("setup-{i}.bin"));
        let fixed_path = out.join(format!("fixed-{i}.bin"));
        let main_path = out.join(format!("main-{i}.bin"));
        if cold {
            let (circuit, commit): (Circuit<Scalar>, KzgCommitment) =
                if meta.exists() && (verify_only || fixed_path.exists()) {
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
                    let mut definition: CircuitInputs<Scalar> = if let Some(traces) = traces {
                        traces
                            .compiled()
                            .kzg_circuit_input_with_degree_policy(
                                i,
                                height,
                                2,
                                !parameters.uses_public_degree(),
                            )?
                            .ok_or("missing compiled circuit")?
                    } else {
                        let mut definition: CircuitInputs<Scalar> =
                            storage::load(&dir.join(format!("{i}.meta")))?;
                        definition.preprocessed = Some(storage::read_matrix_bounded(
                            &dir.join(format!("{i}.fixed.zst")),
                            height,
                        )?);
                        definition
                    };
                    if definition.main_width != manifest.widths[i]
                        || definition.preprocessed.as_ref().is_none_or(|fixed| {
                            fixed.width() == 0 || fixed.height() != manifest.heights[i]
                        })
                    {
                        return Err("fixed trace dimensions differ from the staged manifest".into());
                    }
                    if traces.is_some() {
                        let fixed = definition
                            .preprocessed
                            .take()
                            .ok_or("missing preprocessing")?;
                        storage::save(&dir.join(format!("{i}.meta")), &definition)?;
                        definition.preprocessed = Some(fixed);
                    }
                    eprintln!(
                        "Committing recursive fixed trace {i}: {:?}",
                        start.elapsed()
                    );
                    let (mut local, key) = System::new(config.clone(), [definition]);
                    let mut circuit = local.circuits.remove(0);
                    circuit.preprocessed = None;
                    let data = key.preprocessed_data.unwrap();
                    checkpoint(&fixed_path, &data)?;
                    fixed_parts.push(data);
                    let pair = (circuit, local.preprocessed_commit.unwrap());
                    storage::save(&meta, &pair)?;
                    pair
                };
            if circuit.preprocessed_height != manifest.heights[i]
                || circuit.main_width != manifest.widths[i]
            {
                return Err("setup shape mismatch".into());
            }
            circuits.push(circuit);
            commits.push(commit);
            setup_bytes.push(fs::read(&meta)?);
        }
        if !verify_only {
            let data = if traces.is_none() && main_path.exists() {
                KzgProverData::read_checkpoint(BufReader::with_capacity(
                    1 << 20,
                    File::open(&main_path)?,
                ))?
            } else {
                let main = match traces {
                    Some(traces) => traces.trace(i)?,
                    None => storage::read_matrix_with_shape(
                        &dir.join(format!("{i}.witness.zst")),
                        manifest.widths[i],
                        manifest.heights[i],
                    )?,
                };
                if main.width() != manifest.widths[i] || main.height() != manifest.heights[i] {
                    return Err("generated trace dimensions differ from the staged manifest".into());
                }
                eprintln!("Committing recursive main trace {i}: {:?}", start.elapsed());
                let (_, data) = config.pcs().commit(vec![(
                    config.pcs().natural_domain_for_degree(manifest.heights[i]),
                    main,
                )]);
                if traces.is_none() {
                    checkpoint(&main_path, &data)?;
                }
                data
            };
            main_parts.push(data);
        }
        eprintln!("Prepared recursive trace {i}: {:?}", start.elapsed());
    }
    if cold {
        let mut fixed_commit = KzgCommitment(vec![], vec![]);
        for mut commit in commits {
            fixed_commit.0.append(&mut commit.0);
            fixed_commit.1.append(&mut commit.1);
        }
        let system = System {
            config,
            circuits,
            preprocessed_commit: Some(fixed_commit),
            preprocessed_indices: (0..count).map(Some).collect(),
        };
        let key = if verify_only {
            ProverKey {
                preprocessed_data: None,
            }
        } else {
            let (commit, fixed) = KzgProverData::concatenate(fixed_parts);
            assert_eq!(Some(commit), system.preprocessed_commit);
            ProverKey {
                preprocessed_data: Some(fixed),
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
            logs,
            widths: manifest.widths.clone(),
            heights: manifest.heights.clone(),
            setup_id,
            frontend_id,
            pairing_metadata,
            setup_bytes,
        });
    }
    let loaded = retained.as_ref().unwrap();
    let system = &loaded.system;
    let codec = &loaded.codec;
    let logs = &loaded.logs;
    let refs: Vec<_> = manifest.claims.iter().map(Vec::as_slice).collect();
    let path = out.join("proof.compact.bin");
    let bytes = if verify_only {
        fs::read(&path)?
    } else {
        let (commit, main) = KzgProverData::concatenate(main_parts);
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
            log_degrees: logs.iter().map(|&v| usize::from(v)).collect(),
            stage_1_trace_commit: commit,
            stage_1_trace_data: main,
            lookups,
        };
        eprintln!("Proving recursive KZG wrapper: {:?}", start.elapsed());
        let proof = system.prove_committed(&loaded.key, &refs, stage);
        system
            .verify_multiple_claims(&refs, &proof)
            .map_err(|e| format!("new outer proof: {e:?}"))?;
        let bytes = codec.encode(&proof)?;
        fs::write(&path, &bytes)?;
        bytes
    };
    let mut hash = blake3::Hasher::new();
    if !parameters.uses_public_degree() {
        hash.update(b"init-kzg-recursive-profile/v1");
        hash.update(OUTER_SEED);
    } else {
        hash.update(b"init-kzg-recursive-profile/v4");
        hash.update(system.config.transcript_seed());
    }
    hash.update(&(height as u64).to_le_bytes());
    for i in 0..count {
        hash.update(&fs::read(out.join(format!("setup-{i}.bin")))?);
    }
    hash.update(&fs::read(dir.join("pairings.bin"))?);
    let profile = *hash.finalize().as_bytes();
    let mut packet = profile.to_vec();
    for word in *expected {
        packet.extend(word.to_le_bytes());
    }
    packet.extend(point_bytes);
    packet.extend(&bytes);
    if parameters.uses_public_degree() && packet.len() >= 3_000 {
        return Err("final public-degree proof packet must be smaller than 3000 bytes".into());
    }
    if verify_only
        && (fs::read(out.join("packet.bin"))? != packet
            || fs::read(out.join("profile-id.bin"))? != profile)
    {
        return Err("saved packet or profile differs".into());
    }
    let verify_start = Instant::now();
    verify_packet(
        system,
        codec,
        &packet,
        &profile,
        degree_count,
        &keys,
        expected,
    )?;
    let verify_seconds = verify_start.elapsed().as_secs_f64();
    let decoded = codec.decode(&bytes)?;
    for i in 0..18 {
        let mut wrong = manifest.claims.clone();
        wrong[i + 1][3] += Scalar::ONE;
        assert!(
            system
                .verify_multiple_claims(
                    &wrong.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &decoded
                )
                .is_err()
        );
    }
    for offset in [0, 32, 32 + 18 * 8, packet.len() - 1] {
        let mut wrong = packet.clone();
        wrong[offset] ^= 1;
        assert!(
            verify_packet(
                system,
                codec,
                &wrong,
                &profile,
                degree_count,
                &keys,
                expected
            )
            .is_err()
        );
    }
    assert!(
        verify_packet(
            &system,
            &codec,
            &packet[..packet.len() - 1],
            &profile,
            degree_count,
            &keys,
            expected,
        )
        .is_err()
    );
    let mut extra = packet.clone();
    extra.push(0);
    assert!(
        verify_packet(
            system,
            codec,
            &extra,
            &profile,
            degree_count,
            &keys,
            expected
        )
        .is_err()
    );
    if !verify_only {
        fs::write(out.join("packet.bin"), &packet)?;
        fs::write(out.join("profile-id.bin"), profile)?;
    }
    let development_srs = parameters.is_development();
    let setup_name = if development_srs {
        "development"
    } else {
        "filecoin"
    };
    let filecoin_manifest_digest = parameters
        .filecoin_manifest_digest()
        .map(|digest| blake3::Hash::from_bytes(digest).to_hex().to_string());
    let public_setup = system.config.srs().public_setup();
    let public_setup_id =
        public_setup.map(|setup| blake3::Hash::from_bytes(setup.id).to_hex().to_string());
    let report = serde_json::json!({
        "proof_bytes": bytes.len(),
        "packet_bytes": packet.len(),
        "claim_bytes": 144,
        "pairing_bytes": keys.len() * 48,
        "profile_bytes": 32,
        "total_seconds": start.elapsed().as_secs_f64(),
        "verify_seconds": verify_seconds,
        "native_verification_passed": true,
        "external_pairings_pass": true,
        "negative_tests_pass": true,
        "development_srs": development_srs,
        "known_trapdoor": development_srs,
        "filecoin_acceptance": !development_srs,
        "loaded_key_reused": !cold,
        "srs_config_load_seconds": srs_config_load_seconds,
        "setup": setup_name,
        "filecoin_manifest_digest": filecoin_manifest_digest,
        "ceremony_id": if parameters.is_diagnostic() { None } else { public_setup_id.clone() },
        "public_setup_id": public_setup_id,
        "public_max_degree": public_setup.map(|setup| setup.max_degree),
        "setup_id": blake3::Hash::from_bytes(setup_id).to_hex().to_string(),
        "trace_heights": manifest.heights,
    });
    fs::write(
        out.join(if verify_only {
            "verify-report.json"
        } else {
            "prove-report.json"
        }),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if cold
        && !verify_only
        && let Some(cache) = fixed_cache::FixedCache::for_prove(dir)?
    {
        cache.publish(dir)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::CurveGroup;
    use ark_serialize::CanonicalSerialize;
    use multi_stark::{
        ark_adapter::{KzgConfig, Srs, compact::FixedProofCodec},
        plonkish::CircuitBuilder,
        system::{System, SystemWitness},
        traits::Algebra,
    };
    use std::sync::Arc;

    #[test]
    fn fused_recursive_traces_preserve_proof_bytes_and_resume_verification() -> storage::Result<()>
    {
        let inner = Srs::unsafe_dev_setup(8, b"fused-recursive-inner");
        let mut packet = vec![0u8; 32];
        let expected = init_claim::expected_words()?;
        for word in expected {
            packet.extend(word.to_le_bytes());
        }
        inner.g1[1].serialize_compressed(&mut packet)?;
        (-inner.g1[0]).serialize_compressed(&mut packet)?;
        let (claims, _) = packet_claims(&packet, &[0; 32], 2, &expected)?;
        let make = |private| -> storage::Result<(Built, Assignment<Scalar>)> {
            let mut builder = CircuitBuilder::new();
            let input = builder.input("private");
            let table =
                builder.fixed_table("small", (0..16).map(|v| vec![Scalar::from_u8(v)]).collect());
            builder.lookup(table, &[input]);
            for _ in 0..8 {
                builder.mul(input, input);
            }
            for claim in &claims[1..] {
                let wire = builder.constant(claim[3]);
                builder.expose_public(wire);
            }
            let circuit = builder.finish();
            let mut witness = circuit.witness();
            witness.set(input, Scalar::from_u8(private))?;
            let assignment = witness.generate()?;
            Ok((
                Built {
                    circuit,
                    inputs: vec![],
                    points: vec![],
                    pairing_outputs: vec![],
                    pairing_keys: vec![inner.g2, inner.tau_g2],
                    degree_output_count: 0,
                    terms: vec![],
                },
                assignment,
            ))
        };
        let root = std::env::temp_dir().join(format!("kzg-recursive-fused-{}", std::process::id()));
        fs::create_dir(&root)?;
        let setup = setup::SetupSource::Development { cache: None };
        let mut outputs = vec![];
        for (name, fused, private) in [
            ("checkpointed", false, 7),
            ("fused", true, 7),
            ("fresh", true, 9),
        ] {
            let dir = root.join(name);
            fs::create_dir(&dir)?;
            let (built, assignment) = make(private)?;
            stage_with_setup(built, assignment, &dir, b"recursive-fixture", fused, &setup)?;
            if !fused {
                prove_with_traces(&dir, false, None, &setup)?;
            }
            prove_with_traces(&dir, true, None, &setup)?;
            for name in ["prove-report.json", "verify-report.json"] {
                let report: serde_json::Value =
                    serde_json::from_slice(&fs::read(dir.join("kzg").join(name))?)?;
                assert_eq!(report["setup"], "development");
                assert_eq!(report["development_srs"], true);
                assert_eq!(report["native_verification_passed"], true);
                assert!(report["filecoin_manifest_digest"].is_null());
                assert!(report["ceremony_id"].is_null());
                assert!(report["public_max_degree"].is_null());
            }
            let manifest: storage::Manifest = storage::load(&dir.join("manifest.bin"))?;
            for i in 0..manifest.heights.len() {
                assert_eq!(dir.join(format!("{i}.witness.zst")).exists(), !fused);
                assert_eq!(dir.join(format!("kzg/main-{i}.bin")).exists(), !fused);
            }
            outputs.push(
                ["proof.compact.bin", "packet.bin", "profile-id.bin"]
                    .map(|name| fs::read(dir.join("kzg").join(name)).unwrap()),
            );
        }
        assert_eq!(outputs[0], outputs[1]);
        assert_ne!(outputs[1][0], outputs[2][0]);
        assert_eq!(outputs[1][2], outputs[2][2]);
        let resumed = root.join("checkpointed");
        let old_checkpoint = fs::read(resumed.join("kzg/main-0.bin"))?;
        let (built, assignment) = make(9)?;
        let height = built.circuit.multi_stark_layout()?.main_height;
        let compiled = built
            .circuit
            .lower_to_multi_stark_sharded(Scalar::from_u8(94), height)?
            .merge_table_traces(height)?;
        let traces = compiled.trace_shards(&assignment)?;
        prove_with_traces(&resumed, false, Some(&traces), &setup)?;
        prove_with_traces(&resumed, true, None, &setup)?;
        assert_eq!(
            fs::read(resumed.join("kzg/proof.compact.bin"))?,
            outputs[2][0]
        );
        assert_eq!(fs::read(resumed.join("kzg/main-0.bin"))?, old_checkpoint);
        for name in ["packet.bin", "profile-id.bin"] {
            let path = resumed.join("kzg").join(name);
            let original = fs::read(&path)?;
            let mut altered = original.clone();
            altered[0] ^= 1;
            fs::write(&path, &altered)?;
            let error = prove_with_traces(&resumed, true, None, &setup).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("saved packet or profile differs")
            );
            assert_eq!(fs::read(&path)?, altered);
            fs::write(path, original)?;
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn packet_requires_outer_proof_claim_and_external_pairings() {
        let inner = Srs::unsafe_dev_setup(8, b"packet-inner");
        let keys = vec![inner.g2, inner.tau_g2];
        let profile = [17u8; 32];
        let packet_prefix = |wrong: bool| {
            let mut packet = profile.to_vec();
            for v in init_claim::INIT_PUBLIC_WORDS {
                packet.extend(v.to_le_bytes());
            }
            let lhs = if wrong {
                (inner.g1[1] * ark_bls12_381::Fr::from(2u8)).into_affine()
            } else {
                inner.g1[1]
            };
            lhs.serialize_compressed(&mut packet).unwrap();
            (-inner.g1[0]).serialize_compressed(&mut packet).unwrap();
            packet
        };
        let (claims, _) = packet_claims(
            &packet_prefix(false),
            &profile,
            2,
            &init_claim::INIT_PUBLIC_WORDS,
        )
        .unwrap();
        let mut b = CircuitBuilder::new();
        let inputs: Vec<_> = (1..claims.len())
            .map(|i| {
                let v = b.input(format!("public{i}"));
                b.expose_public(v);
                v
            })
            .collect();
        let circuit = b.finish();
        let lowered = circuit.lower_to_multi_stark(Scalar::from_u8(94)).unwrap();
        let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(64, b"packet-outer")), 2);
        let (system, key) = System::new(config, lowered.kzg_circuit_inputs(64, 2).unwrap());
        let codec = FixedProofCodec::new(
            &system,
            &[system.circuits[0].preprocessed_height.ilog2() as u8],
        )
        .unwrap();
        for wrong in [false, true] {
            let mut packet = packet_prefix(wrong);
            let (claims, _) =
                packet_claims(&packet, &profile, 2, &init_claim::INIT_PUBLIC_WORDS).unwrap();
            let mut w = lowered.witness();
            for (&input, claim) in inputs.iter().zip(&claims[1..]) {
                w.set(input, claim[3]).unwrap();
            }
            let a = w.generate().unwrap();
            let proof = system.prove_multiple_claims(
                &key,
                &claims.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                SystemWitness::from_stage_1(lowered.traces(&a).unwrap(), &system),
            );
            packet.extend(codec.encode(&proof).unwrap());
            assert_eq!(
                verify_packet(
                    &system,
                    &codec,
                    &packet,
                    &profile,
                    0,
                    &keys,
                    &init_claim::INIT_PUBLIC_WORDS
                )
                .is_ok(),
                !wrong
            );
            if !wrong {
                let mut changed = packet.clone();
                changed[32 + 8] ^= 1;
                assert!(
                    verify_packet(
                        &system,
                        &codec,
                        &changed,
                        &profile,
                        0,
                        &keys,
                        &init_claim::INIT_PUBLIC_WORDS
                    )
                    .is_err()
                );
                let mut changed_claims = claims.clone();
                changed_claims[2][3] += Scalar::ONE;
                assert!(
                    system
                        .verify_multiple_claims(
                            &changed_claims.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                            &proof
                        )
                        .is_err()
                );
                assert!(
                    verify_packet(
                        &system,
                        &codec,
                        &packet[..packet.len() - 1],
                        &profile,
                        0,
                        &keys,
                        &init_claim::INIT_PUBLIC_WORDS,
                    )
                    .is_err()
                );
                packet.push(0);
                assert!(
                    verify_packet(
                        &system,
                        &codec,
                        &packet,
                        &profile,
                        0,
                        &keys,
                        &init_claim::INIT_PUBLIC_WORDS
                    )
                    .is_err()
                );
            }
        }
    }
}
