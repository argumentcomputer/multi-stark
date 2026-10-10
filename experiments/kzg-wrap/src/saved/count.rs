use super::*;
use crate::{native_verifier::Counts, outer::setup::SetupSource};
use multi_stark::{config::ProofConfig, plonkish::MultiStarkCircuit, traits::TwoAdicField};

const HEIGHT_CAP: usize = 1 << 27;
const PACKET_LIMIT: usize = 3_000;

#[derive(Debug, serde::Serialize)]
struct Layout {
    ordinary_rows: usize,
    row_count_complete: bool,
    main_height: usize,
    main_width: usize,
    main_fixed_width: usize,
    merged_table_rows: usize,
    merged_table_height: usize,
    merged_table_fixed_width: usize,
    max_computation_height: usize,
    max_table_height: usize,
    layout_admissible: bool,
    rejection_reasons: Vec<&'static str>,
}

impl Layout {
    fn from_counts(counts: &Counts, cap: usize) -> Result<Self> {
        if cap < 2 || !cap.is_power_of_two() {
            return Err("invalid census trace cap".into());
        }
        let stats = counts.stats;
        if counts.table_dimensions.len() != stats.tables
            || counts
                .table_dimensions
                .iter()
                .any(|&(rows, width)| rows == 0 || width == 0)
        {
            return Err("invalid counted table dimensions".into());
        }
        let ordinary_rows = stats
            .gates
            .checked_add(stats.lookups)
            .and_then(|rows| rows.checked_add(stats.publics))
            .and_then(|rows| rows.checked_add(1))
            .ok_or("computation row count overflow")?;
        let main_height = ordinary_rows
            .max(2)
            .checked_next_power_of_two()
            .ok_or("computation height overflow")?;
        let main_width = counts
            .table_dimensions
            .iter()
            .map(|&(_, width)| width)
            .max()
            .unwrap_or(0)
            .max(3);
        let main_fixed_width = main_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(9))
            .ok_or("computation width overflow")?;
        let merged_table_rows =
            counts
                .table_dimensions
                .iter()
                .try_fold(0usize, |sum, &(rows, _)| {
                    sum.checked_add(rows)
                        .ok_or("merged table row count overflow")
                })?;
        let merged_table_height = if merged_table_rows == 0 {
            0
        } else {
            merged_table_rows
                .max(2)
                .checked_next_power_of_two()
                .ok_or("merged table height overflow")?
        };
        let merged_table_fixed_width = main_width
            .checked_add(1)
            .ok_or("merged table width overflow")?;
        let row_count_complete = stats.hash_calls == 0 && stats.hash_compressions == 0;
        let mut rejection_reasons = vec![];
        if !row_count_complete {
            rejection_reasons
                .push("custom hash traces are outside the ordinary single-wrapper profile");
        }
        if main_height > cap {
            rejection_reasons.push("computation exceeds the single-trace height cap");
        }
        if merged_table_height == 0 {
            rejection_reasons.push("profile requires one merged table trace");
        }
        if merged_table_height > cap {
            rejection_reasons.push("merged table exceeds its height cap");
        }
        if merged_table_height > main_height {
            rejection_reasons.push("merged table exceeds the computation-domain merge cap");
        }
        Ok(Self {
            ordinary_rows,
            row_count_complete,
            main_height,
            main_width,
            main_fixed_width,
            merged_table_rows,
            merged_table_height,
            merged_table_fixed_width,
            max_computation_height: cap,
            max_table_height: cap,
            layout_admissible: rejection_reasons.is_empty(),
            rejection_reasons,
        })
    }
}

fn projection(layout: &Layout, counts: &Counts, shifted: bool) -> Result<serde_json::Value> {
    if !layout.layout_admissible {
        return Ok(serde_json::json!({
            "status": "unavailable_for_rejected_layout",
            "projected_packet_below_limit": false,
        }));
    }
    if counts.msm_terms.len().checked_sub(2) != Some(counts.degree_output_count)
        || counts.stats.publics != 18 + 2 * counts.msm_terms.len()
    {
        return Err("counted public/pairing framing differs from the recursive packet".into());
    }
    let circuits = MultiStarkCircuit::<Scalar>::ordinary_merged_kzg_profile(
        Scalar::from_u8(94),
        layout.main_width,
        layout.main_height,
        layout.merged_table_height,
        2,
        shifted,
    )?;
    let logs = [
        layout.main_height.ilog2() as u8,
        layout.merged_table_height.ilog2() as u8,
    ];
    let mut generators = vec![Scalar::ONE];
    for log in logs {
        let generator = Scalar::two_adic_generator(usize::from(log));
        if !generators.contains(&generator) {
            generators.push(generator);
        }
    }
    let codec = FixedProofCodec::for_profile(&circuits, &logs, usize::from(logs[0]), shifted)?;
    let proof_bytes = codec.encoded_len(generators.len())?;
    let pairing_bytes = counts
        .msm_terms
        .len()
        .checked_mul(48)
        .ok_or("pairing bytes overflow")?;
    let packet_bytes = proof_bytes
        .checked_add(pairing_bytes)
        .and_then(|bytes| bytes.checked_add(32 + 18 * 8))
        .ok_or("packet bytes overflow")?;
    let shapes: Vec<_> = circuits
        .iter()
        .map(|circuit| {
            serde_json::json!({
                "height": circuit.preprocessed_height,
                "main_width": circuit.main_width,
                "fixed_width": circuit.preprocessed_width,
                "lookup_group_size": circuit.lookup_group_size,
                "lookup_width": circuit.stage_2_width,
                "quotient_width": circuit.quotient_degree(),
            })
        })
        .collect();
    Ok(serde_json::json!({
        "status": "exact_codec_projection",
        "circuits": shapes,
        "outer_quotient_budget": 2,
        "shifted_degree_bounds": shifted,
        "opening_witnesses": generators.len(),
        "projected_proof_bytes": proof_bytes,
        "projected_packet_bytes": packet_bytes,
        "profile_bytes": 32,
        "claim_bytes": 18 * 8,
        "pairing_points": counts.msm_terms.len(),
        "pairing_bytes": pairing_bytes,
        "packet_limit_exclusive": PACKET_LIMIT,
        "projected_packet_below_limit": packet_bytes < PACKET_LIMIT,
        "actual_outer_proof_generated": false,
    }))
}

pub(super) fn report(
    input: &Input,
    setup: &SetupSource,
    counts: Counts,
    identity: [u8; 32],
) -> Result<serde_json::Value> {
    let layout = Layout::from_counts(&counts, HEIGHT_CAP)?;
    let projected = projection(&layout, &counts, !setup.uses_public_degree())?;
    let packet_ok = projected["projected_packet_below_limit"] == true;
    let stats = counts.stats;
    let public_setup = input.system.config.srs().public_setup();
    let ceremony = match setup {
        SetupSource::Filecoin { digest, .. } => Some(serde_json::json!({
            "trusted_import_receipt": blake3::Hash::from_bytes(*digest).to_hex().to_string(),
            "ceremony_id": public_setup.map(|setup| blake3::Hash::from_bytes(setup.id).to_hex().to_string()),
            "public_max_degree": public_setup.map(|setup| setup.max_degree),
        })),
        SetupSource::Development { .. } | SetupSource::KnownTrapdoorPublicDegree { .. } => None,
    };
    Ok(serde_json::json!({
        "format": "init-kzg-recursive/saved-census/v1",
        "scope": "Saved-input census and exact codec projection using authenticated SRS parameters. Native verification is relative to the supplied, recorded AIR/fixed verifier profile; this command does not independently reconstruct or authenticate its Init-circuit provenance. No recursive gate IR, assignment, physical trace, outer proof or complete pipeline execution is produced. Small AIR metadata graphs and fixed gadget tables are retained for counting.",
        "authenticated_input": {
            "native_verification_passed": true,
            "verifier_profile_provenance": "Supplied setup metadata and fixed commitments, recorded by input hashes and frontend identity; independent Init-circuit provenance is outside this census.",
            "development_known_trapdoor": setup.is_development(),
            "ceremony": ceremony,
            "setup_binding": blake3::Hash::from_bytes(input.setup_identity).to_hex().to_string(),
            "input_proof_bytes": input.proof_bytes.len(),
            "input_blake3": input.input_blake3,
            "inner_max_trace_log": input.system.config.max_log_degree(),
            "inner_quotient_budget": input.system.config.max_quotient_degree(),
            "transcript_seed_blake3": blake3::hash(input.system.config.transcript_seed()).to_hex().to_string(),
            "loaded_g1_points": input.system.config.srs().g1.len(),
        },
        "frontend_identity": blake3::Hash::from_bytes(identity).to_hex().to_string(),
        "layout_admissible": layout.layout_admissible,
        "projected_packet_below_limit": packet_ok,
        "filecoin_census_passed": !setup.is_development() && layout.layout_admissible && packet_ok,
        "actual_outer_proof_generated": false,
        "layout": layout,
        "table_dimensions_rows_width": counts.table_dimensions,
        "codec_projection": projected,
        "stats": {
            "gates": stats.gates, "lookups": stats.lookups, "values": stats.values,
            "inputs": stats.inputs, "tables": stats.tables, "publics": stats.publics,
            "hash_calls": stats.hash_calls, "hash_compressions": stats.hash_compressions,
            "hint_calls": stats.hint_calls, "hint_outputs": stats.hint_outputs,
            "degree_pairing_points": counts.degree_output_count, "msm_terms": counts.msm_terms,
        },
    }))
}

pub(crate) fn run(dir: &Path, output: &Path) -> Result<()> {
    let started = Instant::now();
    let setup = SetupSource::from_env()?;
    if setup.is_development() {
        return Err("count-saved requires authenticated Filecoin parameters; use count for preserved development-profile experiments".into());
    }
    let expected = init_claim::expected_words()?;
    let input = Input::load(dir, &setup, &expected)?;
    let plan = input.plan(HEIGHT_CAP)?;
    plan.validate_request(&input.proof, &input.claims)?;
    let counted_at = Instant::now();
    let counts = plan.count()?;
    let count_seconds = counted_at.elapsed().as_secs_f64();
    let mut result = report(&input, &setup, counts, plan.identity())?;
    let expected_bytes: Vec<_> = [1u64, 18]
        .into_iter()
        .chain(expected)
        .flat_map(u64::to_le_bytes)
        .collect();
    result["expected_statement_blake3"] = blake3::hash(&expected_bytes).to_hex().to_string().into();
    result["input_directory"] = serde_json::to_value(dir)?;
    result["load_and_native_verify_seconds"] = input.load_seconds.into();
    result["count_seconds"] = count_seconds.into();
    result["executable_blake3"] = blake3::hash(&fs::read(std::env::current_exe()?)?)
        .to_hex()
        .to_string()
        .into();
    result["runtime_source_provenance"] = source_hashes();
    result["total_seconds"] = started.elapsed().as_secs_f64().into();
    fs::write(output, serde_json::to_vec_pretty(&result)?)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if result["filecoin_census_passed"] != true {
        return Err(
            "saved census failed the single-wrapper layout or projected packet limit; see report"
                .into(),
        );
    }
    Ok(())
}

fn source_hashes() -> serde_json::Value {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    source_hashes_at(&root)
}

fn source_hashes_at(root: &Path) -> serde_json::Value {
    let mut hashes = BTreeMap::new();
    let mut unavailable = BTreeMap::new();
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "experiments/kzg-wrap/Cargo.toml",
        "experiments/kzg-wrap/Cargo.lock",
        "experiments/kzg-wrap/src/main.rs",
        "experiments/kzg-wrap/src/saved.rs",
        "experiments/kzg-wrap/src/saved/count.rs",
        "experiments/kzg-wrap/src/native_verifier.rs",
        "experiments/kzg-wrap/src/native_verifier/plan.rs",
        "experiments/kzg-wrap/src/native_verifier/shape.rs",
        "experiments/kzg-wrap/src/native_verifier/bindings.rs",
        "experiments/kzg-wrap/src/native_msm.rs",
        "experiments/kzg-wrap/src/native_curve.rs",
        "experiments/kzg-wrap/src/native_field.rs",
        "experiments/kzg-wrap/src/native_field/advice.rs",
        "experiments/kzg-wrap/src/native_transcript.rs",
        "experiments/kzg-wrap/src/outer.rs",
        "examples/support/init_claim.rs",
        "examples/support/kzg_setup.rs",
        "src/plonkish/builder.rs",
        "src/plonkish/stark.rs",
        "src/plonkish/stark/profile.rs",
        "src/plonkish/foreign.rs",
        "src/plonkish/gadgets/bytes.rs",
        "src/plonkish/gadgets/blake3.rs",
        "src/system.rs",
        "src/graph.rs",
        "src/lookup.rs",
        "src/ark_adapter/compact.rs",
        "src/ark_adapter/config.rs",
        "src/ark_adapter/srs.rs",
        "src/ark_adapter/srs/filecoin.rs",
    ] {
        match fs::read(root.join(path)) {
            Ok(bytes) => {
                hashes.insert(path, blake3::hash(&bytes).to_hex().to_string());
            }
            Err(error) => {
                unavailable.insert(path, error.to_string());
            }
        }
    }
    serde_json::json!({
        "scope": "Runtime checkout hashes at report time, not authenticated build provenance. Archive the corresponding source and executable together for reproducibility.",
        "checkout_root": root,
        "complete": unavailable.is_empty(),
        "hashes_blake3": hashes,
        "unavailable": unavailable,
    })
}

#[cfg(test)]
mod tests;
