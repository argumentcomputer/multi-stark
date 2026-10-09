use crate::init_claim;
use multi_stark::{
    ark_adapter::{KzgCommitment, KzgConfig, Scalar, Srs, compact::FixedProofCodec},
    system::{Circuit, System},
    traits::{Algebra, Field},
};
use serde::de::DeserializeOwned;
use std::{
    fs,
    io::{BufReader, Read},
    path::Path,
    sync::Arc,
    time::Instant,
};
#[derive(serde::Deserialize)]
struct Manifest {
    widths: Vec<usize>,
    heights: Vec<usize>,
    claims: Vec<Vec<Scalar>>,
}
fn load<T: DeserializeOwned>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(fs::File::open(path)?);
    let result = bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())?;
    let mut tail = [0];
    if reader.read(&mut tail)? != 0 {
        return Err("trailing metadata".into());
    }
    Ok(result)
}
pub fn check(
    dir: &Path,
    report: &Path,
    stage: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let manifest: Manifest = load(&dir.join("manifest.bin"))?;
    let count = manifest.heights.len();
    if count < 12 || manifest.widths.len() != count {
        return Err("wrong Init profile".into());
    }
    let namespace = Scalar::from_u8(93);
    let mut expected: Vec<_> = std::iter::once(0)
        .chain(init_claim::expected_words()?)
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
    eprintln!("Loading development SRS and saved proof");
    let cache = std::env::var_os("MULTI_STARK_KZG_DEV_SRS_CACHE").map(std::path::PathBuf::from);
    let srs = Arc::new(Srs::unsafe_dev_setup_with_cache(
        height,
        b"init-fri-kzg-ordinary-v1",
        cache.as_deref(),
    )?);
    let config = KzgConfig::new(srs.clone(), 2);
    let mut circuits = vec![];
    // Public slots currently leave constant gates in the native verifier's
    // allocation sequence, so this cache must also bind the expected statement.
    let mut fixed_profile = blake3::Hasher::new();
    fixed_profile.update(b"init-kzg-recursive/fixed-profile/v1");
    fixed_profile.update(&fs::read(dir.join("manifest.bin"))?);
    let mut commitment = KzgCommitment(vec![], vec![]);
    for i in 0..count {
        fixed_profile.update(&fs::read(dir.join(format!("setup-{i}.bin")))?);
        let (circuit, mut part): (Circuit<Scalar>, KzgCommitment) =
            load(&dir.join(format!("setup-{i}.bin")))?;
        if circuit.preprocessed_height != manifest.heights[i]
            || circuit.main_width != manifest.widths[i]
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
    let proof_bytes = fs::read(dir.join("proof.compact.bin"))?;
    let proof = FixedProofCodec::new(&system, &logs)?.decode(&proof_bytes)?;
    let refs: Vec<_> = manifest.claims.iter().map(Vec::as_slice).collect();
    system
        .verify_multiple_claims(&refs, &proof)
        .map_err(|e| format!("native verification: {e:?}"))?;
    eprintln!(
        "Saved {}-byte proof verifies natively: {:?}",
        proof_bytes.len(),
        start.elapsed()
    );
    let srs_seconds = start.elapsed().as_secs_f64();
    let build_start = Instant::now();
    let public: Vec<_> = (1..=18).map(|i| (i, 3)).collect();
    let built = crate::native_verifier::build(&system, &srs, &proof, &manifest.claims, &public)?;
    let build_seconds = build_start.elapsed().as_secs_f64();
    let (mut result, assignment) = built.check_and_assign()?;
    result["input_proof_bytes"] = proof_bytes.len().into();
    result["input_proof_blake3"] = blake3::hash(&proof_bytes).to_hex().to_string().into();
    result["load_and_native_verify_seconds"] = srs_seconds.into();
    result["build_seconds"] = build_seconds.into();
    result["total_seconds"] = start.elapsed().as_secs_f64().into();
    result["development_srs"] = true.into();
    result["scope"]="Full saved Init KZG proof checked by recursive constraints and external pairings; no outer proof yet".into();
    fs::write(report, serde_json::to_vec_pretty(&result)?)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if let Some(stage) = stage {
        crate::outer::stage(
            built,
            assignment,
            stage,
            fixed_profile.finalize().as_bytes(),
        )?;
    }
    Ok(())
}
