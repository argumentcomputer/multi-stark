use crate::{init_claim, native_verifier};
use ark_serialize::CanonicalSerialize;
use multi_stark::{
    ark_adapter::{
        KzgCommitment, Scalar, Srs,
        compact::FixedProofCodec,
        srs::filecoin::{FILECOIN_PUBLIC_MAX_DEGREE, filecoin_setup_id},
    },
    system::Circuit,
    traits::{Algebra, Field},
};
use serde::de::DeserializeOwned;
use std::{
    collections::BTreeMap,
    fs,
    io::{BufReader, Read},
    path::Path,
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
    let value = bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())?;
    let mut tail = [0];
    if reader.read(&mut tail)? != 0 {
        return Err("trailing metadata".into());
    }
    Ok(value)
}

fn seed(srs: &Srs, max_log: usize, public_setup: Option<(usize, [u8; 32])>) -> Vec<u8> {
    let mut seed = match public_setup {
        None => b"multi-stark/kzg/v3".to_vec(),
        Some(_) => b"multi-stark/kzg/v4".to_vec(),
    };
    seed.extend((max_log as u64).to_le_bytes());
    seed.extend(2u64.to_le_bytes());
    if let Some((max_degree, setup_id)) = public_setup {
        seed.extend((max_degree as u64).to_le_bytes());
        seed.extend(setup_id);
    }
    srs.g1[0].serialize_compressed(&mut seed).unwrap();
    srs.g2.serialize_compressed(&mut seed).unwrap();
    srs.g1[1].serialize_compressed(&mut seed).unwrap();
    srs.tau_g2.serialize_compressed(&mut seed).unwrap();
    seed
}

fn report(counts: native_verifier::Counts, seconds: f64) -> serde_json::Value {
    let s = counts.stats;
    let rows = s.gates + s.lookups + s.publics + 1;
    serde_json::json!({
        "gates": s.gates,
        "lookups": s.lookups,
        "values": s.values,
        "inputs": s.inputs,
        "tables": s.tables,
        "hint_calls": s.hint_calls,
        "hint_outputs": s.hint_outputs,
        "hash_calls": s.hash_calls,
        "hash_compressions": s.hash_compressions,
        "rows": rows,
        "main_height": rows.next_power_of_two(),
        "public_values": s.publics,
        "degree_pairing_points": counts.degree_output_count,
        "pairing_points": counts.msm_terms.len(),
        "msm_terms": counts.msm_terms,
        "fits_one_2_27_trace": rows <= 1 << 27,
        "row_margin_below_2_27": (1i64 << 27) - rows as i64,
        "count_seconds": seconds,
    })
}

pub fn run(dir: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
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
    if manifest.claims != expected || manifest.heights.iter().any(|h| !h.is_power_of_two()) {
        return Err("unexpected Init claim or trace height".into());
    }
    let mut input_blake3 = BTreeMap::new();
    let mut record_input = |name: String| -> Result<(), Box<dyn std::error::Error>> {
        let digest = blake3::hash(&fs::read(dir.join(&name))?)
            .to_hex()
            .to_string();
        input_blake3.insert(name, digest);
        Ok(())
    };
    record_input("manifest.bin".into())?;
    record_input("proof.compact.bin".into())?;
    let mut circuits = vec![];
    let mut fixed = KzgCommitment(vec![], vec![]);
    for (i, (&width, &height)) in manifest.widths.iter().zip(&manifest.heights).enumerate() {
        let name = format!("setup-{i}.bin");
        record_input(name.clone())?;
        let (circuit, mut part): (Circuit<Scalar>, KzgCommitment) = load(&dir.join(name))?;
        if circuit.preprocessed.is_some()
            || circuit.preprocessed_height != height
            || circuit.main_width != width
            || part.0.len() != 1
            || part.1.len() != 1
            || part.0[0].len() != circuit.preprocessed_width
        {
            return Err("inconsistent setup metadata".into());
        }
        circuits.push(circuit);
        fixed.0.append(&mut part.0);
        fixed.1.append(&mut part.1);
    }
    let logs: Vec<_> = manifest.heights.iter().map(|h| h.ilog2() as u8).collect();
    let baseline_max_log = *logs.iter().max().unwrap() as usize;
    if baseline_max_log != 24 {
        return Err("expected the preserved 2^24 Init profile".into());
    }
    let proof_bytes = fs::read(dir.join("proof.compact.bin"))?;
    let mut proof = FixedProofCodec::for_profile(&circuits, &logs, baseline_max_log, true)?
        .decode(&proof_bytes)?;
    let public: Vec<_> = (1..=18).map(|i| (i, 3)).collect();
    let anchors = Srs::unsafe_dev_setup(2, b"init-fri-kzg-ordinary-v1");
    let baseline_seed = seed(&anchors, baseline_max_log, None);
    // Pairing keys are external to the circuit and cannot affect its row count.
    let degree_keys = vec![anchors.g2; baseline_max_log + 1];
    let profile = native_verifier::Profile {
        circuits: &circuits,
        fixed: &fixed,
        transcript_seed: &baseline_seed,
        generator: anchors.g1[0],
        g2: anchors.g2,
        tau_g2: anchors.tau_g2,
        degree_keys: &degree_keys,
        max_log_degree: baseline_max_log,
        shifted_degree_bounds: true,
    };
    eprintln!("Counting the saved v3 wrapper without retaining its IR");
    let counted_at = Instant::now();
    let baseline = native_verifier::count(&profile, &proof, &manifest.claims, &public)?;
    let baseline = report(baseline, counted_at.elapsed().as_secs_f64());
    let shifted_fixed: usize = fixed.1.iter().map(Vec::len).sum();
    let mut shifted_witness = 0;
    for commitment in [
        &mut proof.commitments.stage_1_trace,
        &mut proof.commitments.stage_2_trace,
        &mut proof.commitments.quotient_chunks,
    ] {
        for row in &mut commitment.1 {
            shifted_witness += row.len();
            row.clear();
        }
    }
    for row in &mut fixed.1 {
        row.clear();
    }
    let public_max_degree = FILECOIN_PUBLIC_MAX_DEGREE;
    let setup_id = filecoin_setup_id();
    let public_seed = seed(
        &anchors,
        baseline_max_log,
        Some((public_max_degree, setup_id)),
    );
    let profile = native_verifier::Profile {
        circuits: &circuits,
        fixed: &fixed,
        transcript_seed: &public_seed,
        generator: anchors.g1[0],
        g2: anchors.g2,
        tau_g2: anchors.tau_g2,
        degree_keys: &[],
        max_log_degree: baseline_max_log,
        shifted_degree_bounds: false,
    };
    eprintln!("Counting the hypothetical v4 wrapper with mixed trace heights");
    let counted_at = Instant::now();
    let candidate = native_verifier::count(&profile, &proof, &manifest.claims, &public)?;
    let candidate = report(candidate, counted_at.elapsed().as_secs_f64());
    let candidate_proof_bytes =
        FixedProofCodec::for_profile(&circuits, &logs, baseline_max_log, false)?
            .encode(&proof)?
            .len();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut source_blake3 = BTreeMap::new();
    for path in [
        "src/plonkish/builder.rs",
        "src/ark_adapter/compact.rs",
        "src/ark_adapter/config.rs",
        "src/ark_adapter/srs/filecoin.rs",
        "experiments/kzg-wrap/src/count.rs",
        "experiments/kzg-wrap/src/native_verifier.rs",
        "experiments/kzg-wrap/src/native_verifier/plan.rs",
        "experiments/kzg-wrap/src/native_verifier/shape.rs",
        "experiments/kzg-wrap/src/native_verifier/bindings.rs",
        "experiments/kzg-wrap/src/native_msm.rs",
        "experiments/kzg-wrap/src/native_curve.rs",
        "experiments/kzg-wrap/src/native_field.rs",
        "experiments/kzg-wrap/src/native_field/advice.rs",
        "experiments/kzg-wrap/src/native_transcript.rs",
    ] {
        source_blake3.insert(
            path,
            blake3::hash(&fs::read(root.join(path))?)
                .to_hex()
                .to_string(),
        );
    }
    let result = serde_json::json!({
        "scope": "Current shared-builder census for the saved v3 profile and hypothetical v4 profile. Public claim slots are payload-free inputs under the compiled frontend. No circuit IR, witness, trace or proof was generated; this is not proof verification or a soundness result.",
        "candidate_assumptions": [
            "Mixed trace heights and all column widths are unchanged.",
            "Shifted commitments, their degree challenge and their pairing MSMs are omitted.",
            "Subgroup checks and five 80-bit field limbs are unchanged.",
            "Both profiles use the compiled frontend's typed claim schema, without unused constants for public claim values.",
            "The v4 inner transcript uses the saved stage-one trace ceiling 2^24, maximum public degree 2^28-2 and the authenticated Filecoin ceremony identity.",
            "Fixed commitment coordinates and anchors are from the preserved development setup. Actual ceremony constants must be recounted before release."
        ],
        "candidate_setup_id": blake3::Hash::from(setup_id).to_hex().to_string(),
        "public_max_degree": public_max_degree,
        "inner_max_trace_log": baseline_max_log,
        "executable_blake3": blake3::hash(&fs::read(std::env::current_exe()?)?).to_hex().to_string(),
        "input_directory": dir,
        "input_blake3": input_blake3,
        "source_blake3": source_blake3,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_input_proof_bytes": proof_bytes.len(),
        "candidate_input_proof_shape_bytes": candidate_proof_bytes,
        "removed_shifted_fixed_commitments": shifted_fixed,
        "removed_shifted_witness_commitments": shifted_witness,
        "total_seconds": start.elapsed().as_secs_f64(),
    });
    fs::write(output, serde_json::to_vec_pretty(&result)?)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
