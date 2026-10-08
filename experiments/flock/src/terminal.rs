//! KZG capacity test using the native Flock verifier's recorded execution.
use crate::adapter::Ready;
use ark_bls12_381::Fr;
use flock_prover::{
    challenger::FsChallenger,
    pcs::{Commitment, PcsParams},
    proof::R1csProofCircuitMerged,
    transcript_record::RecordingChallenger,
    union::UnionInstance,
};
use flock_terminal_exporter::*;
use ix_terminal_circuit::*;
use serde_json::json;
use std::{cell::Cell, fs, path::Path, rc::Rc, time::Instant};

pub fn gadgets(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    let mut reports = serde_json::Map::new();
    for name in [
        "multiply",
        "constant_multiply",
        "frobenius",
        "equality",
        "blake3",
    ] {
        let start = Instant::now();
        let (relation, witness) = if name == "blake3" {
            let (r, w, _) = build_blake3_compression_r1cs(Blake3CompressionInputV1 {
                chaining_value: BLAKE3_IV,
                message: [0xfedcba98; 16],
                counter: 0,
                block_length: 64,
                flags: 3,
            })?;
            (r, w)
        } else {
            let mut b = R1csBuilder::new();
            let x = alloc_f128_private(&mut b, [0x93; 16], ConstraintPhase::Pcs)?;
            let y = alloc_f128_private(
                &mut b,
                if name == "equality" {
                    [0x93; 16]
                } else {
                    [0x75; 16]
                },
                ConstraintPhase::Pcs,
            )?;
            match name {
                "multiply" => {
                    constrain_f128_multiply(&mut b, &x, &y, ConstraintPhase::Pcs)?;
                }
                "constant_multiply" => {
                    constrain_f128_multiply_constant(&mut b, &x, [0x75; 16], ConstraintPhase::Pcs)?;
                }
                "frobenius" => {
                    constrain_f128_frobenius(&mut b, &x, 1, ConstraintPhase::Pcs)?;
                }
                "equality" => enforce_f128_equal(&mut b, &x, &y, ConstraintPhase::Pcs),
                _ => unreachable!(),
            }
            b.finish()?
        };
        relation.check(&witness)?;
        let gates = ix_fflonk::arithmetize_r1cs(&relation)?;
        ix_fflonk::lower_plonk_witness(&gates, &relation, &witness)?;
        reports.insert(
            name.into(),
            json!({"constraints":relation.census().constraints,
            "plonk_rows":gates.census().active_rows(), "variables":witness.assignment().len(),
            "gate_storage_payload_bytes":gates.gates().storage_payload_bytes(),
            "selector_records":gates.gates().selector_records(),
            "seconds":start.elapsed().as_secs_f64()}),
        );
    }
    fs::write(
        out.join("gadgets.json"),
        serde_json::to_vec_pretty(&reports)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}

#[derive(Debug)]
struct FftCapacityExceeded;

fn check_capacity(active_rows: u64, max_domain: u64) {
    if active_rows + ix_fflonk::FFLONK_BLINDING_ROWS > max_domain {
        std::panic::resume_unwind(Box::new(FftCapacityExceeded));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn census(
    out: &Path,
    ready: Ready,
    params: &PcsParams,
    domain: &[u8],
    commitment: &Commitment,
    proof: &R1csProofCircuitMerged,
    application_words: &[u64],
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    flock_prover::scratch::clear();
    let circuit_digest = ready.circuit.digest();
    let union = UnionInstance::new(&ready.registry, ready.counts.clone());
    let circuits = ready.circuits();
    let fixed_matrices = crate::fixed_matrices::FixedMatrices::new(&ready.registry)?;
    fs::write(
        out.join("fixed-matrix-program.json"),
        serde_json::to_vec_pretty(&fixed_matrices.stats())?,
    )?;
    let matrix_geometry: Vec<_> = ready
        .registry
        .boolean_types()
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            json!({"table":i, "variables":ty.k_log,
            "a_nonzeros":ty.a_0.rows.iter().map(Vec::len).sum::<usize>(),
            "b_nonzeros":ty.b_0.rows.iter().map(Vec::len).sum::<usize>()})
        })
        .collect();
    fs::write(
        out.join("fixed-matrices.json"),
        serde_json::to_vec_pretty(&matrix_geometry)?,
    )?;
    let mut recording = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));
    // Verify all native obligations before translating. These host checks are
    // validation of the experiment, never substitutes for terminal constraints.
    flock_prover::verifier::verify_ligerito_union_circuit(
        &union,
        &ready.circuit,
        &ready.public,
        &circuits,
        commitment,
        proof,
        params,
        &mut FsChallenger::with_chained_blake3(domain),
    )
    .map_err(|e| format!("native Flock verifier: {e:?}"))?;
    let (_, deferred, sigma) = flock_prover::verifier::verify_ligerito_union_circuit_deferred(
        &union,
        &ready.circuit,
        &ready.public,
        &circuits,
        commitment,
        proof,
        params,
        &mut recording,
    )
    .map_err(|e| format!("recorded Flock verifier: {e:?}"))?;
    if deferred.element.is_some() {
        return Err("element PIOP is not supported".into());
    }
    let wiring = export_wiring_f128_algebra(
        &recording,
        &ready.circuit,
        &ready.public,
        &proof.wiring,
        &sigma,
    )?;
    let fixed_live = crate::fixed_tables::FixedLiveMask::new(
        circuit_digest,
        &ready.circuit.live_mask(),
        wiring.trace().structure_base_variables as usize,
    )?;
    fs::write(
        out.join("fixed-live-mask.json"),
        serde_json::to_vec_pretty(&fixed_live.stats())?,
    )?;
    let fixed_sigma = crate::fixed_sigma::FixedSigma::prepare(
        &ready.circuit,
        &sigma,
        wiring.trace().structure_matrix(),
        out,
    )?;
    let algebra = export_boolean_piop_f128_algebra(
        &recording,
        &union,
        proof.boolean.as_ref().ok_or("missing Boolean PIOP")?,
        params.zerocheck_grinding(),
        params.lincheck_grinding(),
    )?;
    let frontend = export_merged_pcs_frontend(
        &recording,
        &union,
        commitment,
        &proof.pcs_open,
        wiring.trace(),
        &algebra,
    )?;
    let assist = export_multipoint_twisted_assist(
        &recording,
        circuit_digest,
        &union,
        &proof.pcs_open,
        wiring.trace(),
        &algebra,
        &frontend,
        &deferred.jagged,
    )?;
    let ligerito = export_inner_ligerito(&recording, commitment, &proof.pcs_open, &frontend)?;
    let fixed_layout = crate::fixed_tables::FixedLayout::new(
        circuit_digest,
        &flock_prover::pcs::jagged::JaggedParams::from_heights(
            &union.jagged_heights(),
            union.n_log(),
            union.dense_m() - 7,
        ),
    )?;
    fs::write(
        out.join("fixed-layout.json"),
        serde_json::to_vec_pretty(&fixed_layout.stats())?,
    )?;
    let payload_indices = statement_prefix_payloads(&recording)?;
    let transcript = Stage4FlockTranscriptWitnessV1::from_recording_with_algebra(
        &recording,
        domain,
        algebra.trace,
        &algebra.private_values,
    )?;
    let prefix_len = ready
        .public
        .len()
        .checked_sub(application_words.len())
        .ok_or("application count")?;
    let public_values: Vec<_> = ready
        .public
        .iter()
        .map(|v| (u128::from(v.lo) | (u128::from(v.hi) << 64)).to_le_bytes())
        .collect();
    for (actual, expected) in public_values[prefix_len..].iter().zip(application_words) {
        if u128::from_le_bytes(*actual) != u128::from(*expected) {
            return Err("application public mismatch".into());
        }
    }
    let bundle = bincode::serialize(&(commitment, proof))?;
    fs::write(out.join("terminal-input.bin"), &bundle)?;
    let mut report = json!({
        "status":"projecting", "terminal_proof_generated":false,
        "source_revision":"8fdb3eab26491f2e79a6004016c2b7b96d0b4bda",
        "flock_circuit_digest":blake3::Hash::from_bytes(circuit_digest).to_hex().to_string(),
        "flock_transcript":"chained-blake3", "flock_proof_bytes":bundle.len(),
        "pcs_profile":params.profile.as_str(),
        "compact_proof_bytes_if_completed":ix_fflonk::FFLONK_COMPACT_PROOF_BYTES,
        "qr_proof_bytes_if_completed":ix_fflonk::FFLONK_QR_PROOF_BYTES,
        "flock_proof_blake3":blake3::hash(&bundle).to_hex().to_string(),
        "public_words":application_words, "export_seconds":start.elapsed().as_secs_f64(),
        "scope":"terminal circuit census; completed_phases records fixed discharges reached; no terminal proof generated or composed security certification",
    });
    let save = |report: &serde_json::Value| -> Result<(), Box<dyn std::error::Error>> {
        fs::write(
            out.join("terminal.json"),
            serde_json::to_vec_pretty(report)?,
        )?;
        Ok(())
    };
    save(&report)?;
    drop(circuits);
    drop(union);
    drop(ready);
    eprintln!("Terminal export complete; released native circuit before projection");
    let rows = Rc::new(Cell::new(0u64));
    let stage_rows = rows.clone();
    let gate_rows = rows.clone();
    let gates = ix_fflonk::PlonkGateProjectionV1::new_gate_observed(move |_| {
        gate_rows.set(gate_rows.get() + 1);
    });
    let mut observe = gates.observer();
    let mut constraints = 0u64;
    let progress_path = out.join("projection-progress.json");
    let public_count = application_words.len() as u64;
    let maximum_fft_domain = 1u64 << <Fr as ark_ff::FftField>::TWO_ADICITY;
    let maximum_domain = maximum_fft_domain;
    let mut builder = R1csBuilder::new_projection_observed(move |constraint| {
        observe(constraint);
        constraints += 1;
        if constraints.is_multiple_of(1_000_000) {
            let progress = json!({"constraints":constraints,"plonk_rows":rows.get()+public_count,"phase":format!("{:?}",constraint.phase),"elapsed_seconds":start.elapsed().as_secs_f64()});
            // Monitoring failure must not alter circuit construction.
            let _ = fs::write(&progress_path, progress.to_string());
        }
        // Both observers have finished this entire constraint. Their counts
        // remain consistent when we unwind this sizing-only construction.
        // File products split exactly; only the base domain needs roots of unity.
        check_capacity(rows.get() + public_count, maximum_domain);
    });
    let public = application_words
        .iter()
        .map(|v| builder.alloc_public(Fr::from(*v)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut last = Instant::now();
    let mut completed_phases = serde_json::Map::new();
    let mut previous_rows = 0;
    let mut phase = |name: &str| {
        let elapsed = last.elapsed().as_secs_f64();
        let cumulative = stage_rows.get();
        let phase_rows = cumulative - previous_rows;
        previous_rows = cumulative;
        completed_phases.insert(
            name.to_owned(),
            json!({
                "plonk_rows": phase_rows,
                "cumulative_plonk_rows_excluding_public_bindings": cumulative,
                "seconds": elapsed,
            }),
        );
        let _ = fs::write(
            out.join("phase-progress.json"),
            json!(completed_phases).to_string(),
        );
        eprintln!("terminal {name}: {elapsed:.2}s, {phase_rows} rows");
        last = Instant::now();
        (name.to_owned(), json!(elapsed))
    };
    let mut times = serde_json::Map::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> anyhow::Result<_> {
        let tape = constrain_chained_blake3_transcript(
            &mut builder,
            transcript.chained_blake3(),
            transcript.observed_values(),
            transcript.byte_payloads(),
            transcript.challenges(),
        )?;
        let (k, v) = phase("transcript");
        times.insert(k, v);
        let public_values = constrain_application_public_values(
            &mut builder,
            circuit_digest,
            &public_values[..prefix_len],
            &public,
            &public_values,
            payload_indices,
            &tape.byte_payloads,
        )?;
        let (k, v) = phase("statement");
        times.insert(k, v);
        let wiring = constrain_f128_wiring(
            &mut builder,
            wiring.trace(),
            F128WiringCircuitInputsV1 {
                public_values: &public_values,
                observed_values: &tape.observed_values,
                challenges: &tape.challenges,
                private_values: wiring.private_values(),
            },
        )?;
        let (k, v) = phase("wiring");
        times.insert(k, v);
        fixed_live.constrain(&mut builder, &wiring.circuit_structure_claims)?;
        let (k, v) = phase("fixed_live_mask");
        times.insert(k, v);
        let private = transcript
            .f128_private_values()
            .iter()
            .map(|v| alloc_f128_private(&mut builder, *v, ConstraintPhase::Lincheck))
            .collect::<Result<Vec<_>, _>>()?;
        let algebra = constrain_f128_algebra_trace_deferred(
            &mut builder,
            transcript.f128_algebra(),
            F128AlgebraCircuitInputsV1 {
                public_values: &public_values,
                observed_values: &tape.observed_values,
                challenges: &tape.challenges,
                private_values: &private,
            },
        )?;
        let (k, v) = phase("boolean_piop");
        times.insert(k, v);
        let frontend = constrain_f128_merged_pcs_frontend(
            &mut builder,
            &frontend,
            F128MergedPcsFrontendCircuitInputsV1 {
                public_values: &public_values,
                observed_values: &tape.observed_values,
                challenges: &tape.challenges,
                private_values: &private,
                algebra_operations: &algebra.operations,
                byte_payloads: &tape.byte_payloads,
                packed_direct_claims: &wiring.gather_claims,
            },
        )?;
        let (k, v) = phase("merged_pcs");
        times.insert(k, v);
        let assist = constrain_f128_multipoint_twisted_assist(
            &mut builder,
            assist.trace(),
            F128MultipointTwistedAssistCircuitInputsV1 {
                observed_values: &tape.observed_values,
                challenges: &tape.challenges,
                private_values: assist.private_values(),
                frontend: &frontend,
            },
        )?;
        let (k, v) = phase("multipoint");
        times.insert(k, v);
        constrain_f128_inner_ligerito(
            &mut builder,
            ligerito.trace(),
            F128InnerLigeritoCircuitInputsV1 {
                observed_values: &tape.observed_values,
                challenges: &tape.challenges,
                byte_payloads: &tape.byte_payloads,
                private_values: ligerito.private_values(),
                private_digests: ligerito.private_digests(),
                frontend: &frontend,
            },
        )?;
        let (k, v) = phase("inner_ligerito");
        times.insert(k, v);
        fixed_layout.constrain(&mut builder, &assist.jagged_assertion)?;
        let (k, v) = phase("fixed_layout");
        times.insert(k, v);
        fixed_matrices.constrain(&mut builder, &algebra.deferred_matrix_claims)?;
        let (k, v) = phase("fixed_matrices");
        times.insert(k, v);
        fixed_sigma.constrain(&mut builder, &wiring.circuit_structure_claims[2])?;
        let (k, v) = phase("fixed_sigma");
        times.insert(k, v);
        Ok((0usize, 0usize, 0usize))
    }));
    let (result, stopped_at_limit) = match result {
        Ok(value) => (value.map(Some), false),
        Err(reason) if reason.is::<FftCapacityExceeded>() => (Ok(None), true),
        Err(reason) => std::panic::resume_unwind(reason),
    };
    let projection = builder.finish_projection()?;
    let census = gates.finish_for_sizing(&projection)?;
    let capacity = ix_fflonk::plan_fflonk_capacity(&census)?;
    report["completed_phases"] = completed_phases.into();
    report["phase_seconds"] = times.into();
    report["r1cs_constraints"] = projection.census().constraints.into();
    report["r1cs_projection_digest"] = blake3::Hash::from_bytes(projection.digest())
        .to_hex()
        .to_string()
        .into();
    report["r1cs_variables"] =
        (1 + u64::from(projection.public_variables()) + u64::from(projection.private_variables()))
            .into();
    report["r1cs_constraints_by_phase"] = projection
        .census()
        .constraints_by_phase
        .iter()
        .map(|(phase, count)| (format!("{phase:?}"), json!(count)))
        .collect::<serde_json::Map<_, _>>()
        .into();
    report["plonk_rows_by_phase"] = census
        .rows_by_phase
        .iter()
        .map(|(phase, count)| (format!("{phase:?}"), json!(count)))
        .collect::<serde_json::Map<_, _>>()
        .into();
    report["plonk_rows"] = census.active_rows().into();
    report["domain"] = capacity.domain_size.into();
    report["product_fft_domain"] = capacity.polynomial_fft_domain_size.into();
    report["maximum_product_fft_domain"] = maximum_fft_domain.into();
    report["supported_fft_domain"] = capacity.supported_file_workspace_domain.into();
    report["monolithic_product_fft_supported"] = capacity.supported_polynomial_fft_domain.into();
    report["file_product_fft_domain"] = capacity.file_product_fft_domain_size.into();
    report["polynomial_backend"] = "file-backed coefficient splitting".into();
    report["stopped_at_fft_limit"] = stopped_at_limit.into();
    report["srs_degree"] = capacity.required_srs_degree.into();
    report["compressed_srs_bytes"] = capacity.compressed_file_srs_bytes.into();
    report["file_key_polynomial_bytes"] = capacity.file_key_polynomial_bytes.into();
    report["file_srs_key_and_fft_minimum_bytes"] =
        capacity.file_srs_key_and_fft_minimum_bytes.into();
    report["elapsed_seconds"] = start.elapsed().as_secs_f64().into();
    match result {
        Ok(Some((matrix, structure, jagged))) => {
            report["pending_claims"] =
                json!({"matrix":matrix,"structure":structure,"jagged":jagged});
            report["status"] = "all_fixed_claims_projected".into();
        }
        Ok(None) => {
            report["status"] = "blocked_fft_domain".into();
            report["scope"] = "partial census: replay stopped at the KZG FFT limit; see completed_phases for checks reached".into();
        }
        Err(error) => {
            report["status"] = "projection_failed".into();
            report["error"] = error.to_string().into();
            save(&report)?;
            return Err(error.into());
        }
    }
    save(&report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{AdditiveGroup, Field};

    #[test]
    fn capacity_stop_preserves_exact_completed_constraint_counts() {
        let rows = Rc::new(Cell::new(0));
        let count = rows.clone();
        let gates = ix_fflonk::PlonkGateProjectionV1::new_gate_observed(move |_| {
            count.set(count.get() + 1)
        });
        let mut observe = gates.observer();
        let mut builder = R1csBuilder::new_projection_observed(move |constraint| {
            observe(constraint);
            check_capacity(rows.get(), 8);
        });
        let bit = builder.alloc_private(Fr::ONE).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for _ in 0..20 {
                builder.enforce_boolean(ConstraintPhase::Pcs, bit);
            }
        }));
        assert!(result.unwrap_err().is::<FftCapacityExceeded>());
        let projection = builder.finish_projection().unwrap();
        let census = gates.finish_for_sizing(&projection).unwrap();
        assert_eq!(projection.census().constraints, 7);
        assert_eq!(census.active_rows(), 7);
        assert_eq!(census.domain_size, 16);
    }

    #[test]
    fn application_claim_and_fixed_prefix_are_constrained() {
        use flock_prover::{field::F128, union::publics_digest};
        let values = [F128::new(9, 17), F128::new(13, 0), F128::new(42, 0)];
        let raw: Vec<_> = values
            .iter()
            .map(|v| (u128::from(v.lo) | (u128::from(v.hi) << 64)).to_le_bytes())
            .collect();
        let digest = publics_digest(&values);
        let circuit_digest = [7; 32];
        let mut builder = R1csBuilder::new();
        let public = [13, 42].map(|v| builder.alloc_public(Fr::from(v)).unwrap());
        let payloads: Vec<Vec<_>> = [circuit_digest, digest]
            .iter()
            .map(|bytes| {
                bytes
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .map(|word| {
                        F128TranscriptWordV1::from_f128_variables(
                            &alloc_f128_private(&mut builder, *word, ConstraintPhase::Transcript)
                                .unwrap(),
                        )
                    })
                    .collect()
            })
            .collect();
        let bound = constrain_application_public_values(
            &mut builder,
            circuit_digest,
            &raw[..1],
            &public,
            &raw,
            [0, 1],
            &payloads,
        )
        .unwrap();
        let (r1cs, witness) = builder.finish().unwrap();
        r1cs.check(&witness).unwrap();
        // Bypass the witness generator and host digest checks.
        for variable in public
            .into_iter()
            .chain(bound.iter().flat_map(|v| v.bit_variables().iter().copied()))
        {
            let mut changed = witness.clone();
            let original = witness.assignment()[variable.index() as usize];
            changed.set(variable, original + Fr::ONE).unwrap();
            assert!(
                r1cs.check(&changed).is_err(),
                "unbound variable {}",
                variable.index()
            );
        }
    }

    #[test]
    fn development_kzg_proof_binds_binary_field_multiplication() {
        use ark_bls12_381::{G1Affine, G2Affine};
        use ark_ec::{AffineRepr, CurveGroup};
        use ark_ff::PrimeField;
        use flock_prover::field::F128;
        use ix_fflonk::*;
        use rayon::prelude::*;
        let (a, b) = (
            F128::new(0x12345678, 0xabcdef),
            F128::new(0xcafebabe, 0x87654321),
        );
        let product = a * b;
        let encode = |v: F128| (u128::from(v.lo) | (u128::from(v.hi) << 64)).to_le_bytes();
        let mut builder = R1csBuilder::new();
        let output = builder
            .alloc_public(Fr::from_le_bytes_mod_order(&encode(product)))
            .unwrap();
        let a = alloc_f128_private(&mut builder, encode(a), ConstraintPhase::Pcs).unwrap();
        let b = alloc_f128_private(&mut builder, encode(b), ConstraintPhase::Pcs).unwrap();
        let product = constrain_f128_multiply(&mut builder, &a, &b, ConstraintPhase::Pcs).unwrap();
        let mut weight = Fr::ONE;
        let packed = LinearCombination::from_terms(product.bit_variables().iter().map(|&v| {
            let term = (v, weight);
            weight.double_in_place();
            term
        }));
        builder.enforce_zero(
            ConstraintPhase::Pcs,
            packed.minus(&LinearCombination::from_variable(output)),
        );
        let (r1cs, witness) = builder.finish().unwrap();
        r1cs.check(&witness).unwrap();
        let arithmetization = arithmetize_r1cs(&r1cs).unwrap();
        let degree = required_fflonk_srs_degree(arithmetization.census().domain_size).unwrap();
        // Known tau is ONLY for this reproducible backend test.
        let tau = Fr::from(29u64);
        let mut power = Fr::ONE;
        let scalars: Vec<_> = (0..=degree)
            .map(|_| {
                let p = power;
                power *= tau;
                p
            })
            .collect();
        let powers = scalars
            .par_iter()
            .map(|p| {
                G1Affine::generator()
                    .mul_bigint(p.into_bigint())
                    .into_affine()
            })
            .collect();
        let srs = KzgUniversalSrsV1::new(
            powers,
            G2Affine::generator(),
            G2Affine::generator()
                .mul_bigint(tau.into_bigint())
                .into_affine(),
        )
        .unwrap();
        let key = preprocess_fflonk(&srs, arithmetization).unwrap();
        let proof = prove_fflonk(&srs, &key, &r1cs, &witness, FflonkBlindingV1::default()).unwrap();
        assert_eq!(proof.proof.to_bytes().len(), 992);
        assert!(
            verify_fflonk(&key.verification_key(), &proof.proof, &proof.public_inputs).unwrap()
        );
        let mut wrong = proof.public_inputs.clone();
        wrong[0] += Fr::ONE;
        assert!(!verify_fflonk(&key.verification_key(), &proof.proof, &wrong).unwrap());
        for i in 0..15 {
            let mut wrong = proof.proof.clone();
            wrong.evaluations[i] += Fr::ONE;
            assert!(!verify_fflonk(&key.verification_key(), &wrong, &proof.public_inputs).unwrap());
        }
    }
}
