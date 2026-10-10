use crate::{
    init_claim,
    native_verifier::{ClaimSchema, LayoutPolicy, Plan},
};
use multi_stark::{
    ark_adapter::{KzgCommitment, KzgConfig, Scalar, compact::FixedProofCodec},
    prover::Proof,
    system::{Circuit, System},
    traits::{Algebra, Field},
};
use serde::de::DeserializeOwned;
use std::{collections::BTreeMap, fs, path::Path, time::Instant};

mod count;
mod development;
#[cfg(test)]
mod tests;
mod worker;
pub(crate) use count::run as count_saved;
pub(crate) use development::{
    bootstrap as development_bootstrap, frontend as development_frontend_bench,
    prove as development_prove_bench, verify as development_verify,
};
pub(crate) use worker::{development_serve, run_many, serve};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(serde::Deserialize)]
struct Manifest {
    widths: Vec<usize>,
    heights: Vec<usize>,
    claims: Vec<Vec<Scalar>>,
}

fn read_recorded(dir: &Path, name: &str, hashes: &mut BTreeMap<String, String>) -> Result<Vec<u8>> {
    let bytes = fs::read(dir.join(name))?;
    hashes.insert(name.into(), blake3::hash(&bytes).to_hex().to_string());
    Ok(bytes)
}

fn load<T: DeserializeOwned>(
    dir: &Path,
    name: &str,
    hashes: &mut BTreeMap<String, String>,
) -> Result<T> {
    let bytes = read_recorded(dir, name, hashes)?;
    let (result, used) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
    if used != bytes.len() {
        return Err("trailing metadata".into());
    }
    Ok(result)
}

struct Input {
    system: System<KzgConfig>,
    proof: Proof<KzgConfig>,
    claims: Vec<Vec<Scalar>>,
    logs: Vec<u8>,
    proof_bytes: Vec<u8>,
    input_blake3: BTreeMap<String, String>,
    setup_identity: [u8; 32],
    load_seconds: f64,
}

impl Input {
    fn plan(&self, max_height: usize) -> Result<Plan<'_>> {
        Self::metadata_plan(&self.system, &self.logs, &self.claims, max_height)
    }

    fn metadata_plan<'a>(
        system: &'a System<KzgConfig>,
        logs: &[u8],
        claims: &[Vec<Scalar>],
        max_height: usize,
    ) -> Result<Plan<'a>> {
        let public: Vec<_> = (1..=18).map(|i| (i, 3)).collect();
        Ok(Plan::new(
            system,
            logs,
            ClaimSchema::from_claims(claims, &public)?,
            LayoutPolicy {
                namespace: Scalar::from_u8(94),
                max_computation_height: max_height,
                max_table_height: max_height,
            },
        )?)
    }

    fn load(
        dir: &Path,
        setup: &crate::outer::setup::SetupSource,
        expected_words: &[u64; 18],
    ) -> Result<Self> {
        let start = Instant::now();
        let mut input_blake3 = BTreeMap::new();
        let manifest: Manifest = load(dir, "manifest.bin", &mut input_blake3)?;
        let count = manifest.heights.len();
        if count < 12 || manifest.widths.len() != count || manifest.widths.contains(&0) {
            return Err("wrong Init profile".into());
        }
        for &height in &manifest.heights {
            setup.check_height(height)?;
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
            return Err("Init public claim/activation mismatch".into());
        }
        let height = *manifest.heights.iter().max().unwrap();
        if height != 1 << 24 {
            return Err("wrong SRS length".into());
        }
        eprintln!("Loading authenticated verifier parameters and saved proof");
        let setup_id = setup.identity(b"init-fri-kzg-ordinary-v1", height, 2)?;
        setup.check_binding(dir, &setup_id)?;
        match read_recorded(dir, crate::outer::setup::BINDING_FILE, &mut input_blake3) {
            Ok(bytes) if bytes == setup_id => (),
            Err(error)
                if matches!(setup, crate::outer::setup::SetupSource::Development { .. })
                    && error
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound) =>
            {
                ()
            }
            Err(error) => return Err(error),
            _ => return Err("setup binding changed while loading saved metadata".into()),
        }
        let config = setup.config(b"init-fri-kzg-ordinary-v1", height, 2, true)?;
        let mut circuits = vec![];
        let mut commitment = KzgCommitment(vec![], vec![]);
        for i in 0..count {
            let (circuit, mut part): (Circuit<Scalar>, KzgCommitment) =
                load(dir, &format!("setup-{i}.bin"), &mut input_blake3)?;
            if circuit.preprocessed_height != manifest.heights[i]
                || circuit.main_width != manifest.widths[i]
                || circuit.preprocessed.is_some()
                || part.0.len() != 1
                || part.1.len() != 1
            {
                return Err("inconsistent setup metadata".into());
            }
            circuits.push(circuit);
            commitment.0.append(&mut part.0);
            commitment.1.append(&mut part.1);
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
        let max_height = if setup.uses_public_degree() {
            1 << 27
        } else {
            1 << 29
        };
        let plan = Self::metadata_plan(&system, &logs, &manifest.claims, max_height)?;
        let proof_bytes = read_recorded(dir, "proof.compact.bin", &mut input_blake3)?;
        plan.validate_compact_len(proof_bytes.len())?;
        let proof = FixedProofCodec::new(&system, &logs)?.decode(&proof_bytes)?;
        plan.validate_request(&proof, &manifest.claims)?;
        let refs: Vec<_> = manifest.claims.iter().map(Vec::as_slice).collect();
        system
            .verify_multiple_claims(&refs, &proof)
            .map_err(|e| format!("native verification: {e:?}"))?;
        eprintln!(
            "Saved {}-byte proof verifies natively: {:?}",
            proof_bytes.len(),
            start.elapsed()
        );
        Ok(Self {
            system,
            proof,
            claims: manifest.claims,
            logs,
            proof_bytes,
            input_blake3,
            setup_identity: setup_id,
            load_seconds: start.elapsed().as_secs_f64(),
        })
    }
}

pub fn check(dir: &Path, report: &Path, stage: Option<&Path>, fused: bool) -> Result<()> {
    let start = Instant::now();
    let setup = crate::outer::setup::SetupSource::from_env()?;
    let expected = init_claim::expected_words()?;
    let input = Input::load(dir, &setup, &expected)?;
    let build_start = Instant::now();
    let max_height = if setup.uses_public_degree() {
        1 << 27
    } else {
        1 << 29
    };
    let plan = input.plan(max_height)?;
    let fixed_profile = plan.identity();
    let built = plan.build(&input.proof, &input.claims)?;
    let build_seconds = build_start.elapsed().as_secs_f64();
    let (mut result, assignment) = built.check_and_assign()?;
    result["input_proof_bytes"] = input.proof_bytes.len().into();
    result["input_proof_blake3"] = blake3::hash(&input.proof_bytes).to_hex().to_string().into();
    result["frontend_identity"] = blake3::Hash::from_bytes(fixed_profile)
        .to_hex()
        .to_string()
        .into();
    result["load_and_native_verify_seconds"] = input.load_seconds.into();
    result["build_seconds"] = build_seconds.into();
    result["total_seconds"] = start.elapsed().as_secs_f64().into();
    result["development_srs"] = setup.is_development().into();
    result["scope"] = "Full saved Init KZG proof checked by recursive constraints and external pairings; no outer proof yet".into();
    fs::write(report, serde_json::to_vec_pretty(&result)?)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if let Some(stage) = stage {
        crate::outer::stage_with_expected(
            built,
            assignment,
            stage,
            &fixed_profile,
            fused,
            &setup,
            &expected,
        )?;
    }
    Ok(())
}
