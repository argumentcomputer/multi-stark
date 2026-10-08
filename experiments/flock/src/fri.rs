use crate::adapter::Program;
use flock_prover::{
    challenger::FsChallenger,
    field::F128,
    pcs::{
        PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k_or_default},
    },
    union::UnionInstance,
    verifier::verify_ligerito_union_circuit,
};
use multi_stark::{
    batch::ShardInput,
    expr::Expr,
    lookup::Lookup,
    plonkish::verifier::*,
    system::{CircuitInputs, System, SystemWitness},
    types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val},
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use serde_json::json;
use std::{fs, path::Path, time::Instant};
const DOMAIN: &[u8] = b"multi-stark-flock-complete-fri-v1";

pub fn run(out: &Path, prove: bool) -> Result<(), Box<dyn std::error::Error>> {
    run_inner(out, prove, false)
}

pub fn run_inner(
    out: &Path,
    prove: bool,
    terminal: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = flock_prover::init_perf_thread_pool();
    let start = Instant::now();
    let (inner, pk) = System::new(
        GoldilocksBlake3Config::new(
            CommitmentParameters {
                log_blowup: 1,
                cap_height: 0,
            },
            FriParameters {
                log_final_poly_len: 0,
                max_log_arity: 1,
                num_queries: 1,
                commit_proof_of_work_bits: 0,
                query_proof_of_work_bits: 0,
            },
        ),
        [CircuitInputs {
            main_width: 1,
            preprocessed: Some(RowMajorMatrix::new_col(vec![Val::ONE, Val::ZERO])),
            lookups: vec![Lookup::pull(Expr::preprocessed(0), vec![Expr::main(0)])],
            ..Default::default()
        }],
    );
    let statement = Statement {
        claims: vec![vec![Val::from_u8(13)]],
        messages: vec![],
    };
    let proof = inner.prove_batch(
        &pk,
        vec![ShardInput {
            claims: statement.claims.clone(),
            witness: SystemWitness::from_stage_1(
                vec![RowMajorMatrix::new_col(vec![Val::from_u8(13); 2])],
                &inner,
            ),
        }],
        vec![],
    );
    inner
        .verify_batch(&proof)
        .map_err(|e| format!("inner: {e:?}"))?;
    let key = VerifierKey::from_system(&inner);
    let plan = VerifierPlan::validate(
        &key,
        ProofProfile {
            envelope: Envelope::SingleBatch,
            active: vec![true],
            log_degrees: vec![1],
            claim_lengths: vec![1],
            message_lengths: vec![],
            max_field_retries: 2,
        },
        VerifierLimits::default(),
    )?;
    let (circuit, inputs) = plan.build(
        Statement {
            claims: vec![vec![StatementSlot::Public]],
            messages: vec![],
        },
        ImplementationOptions {
            compact_blake3: true,
        },
    )?;
    let prepared = plan.expand_witness(ProofEnvelope::SingleBatch(&proof))?;
    let mut witness = circuit.witness();
    inputs.assign_statement(&mut witness, &statement)?;
    prepared.assign_proof(&mut witness, &inputs)?;
    let assignment = witness.generate()?;
    let program = Program::compile(&circuit)?;
    let census = Program::census(&circuit)?;
    assert_eq!(program.stats(), census.stats());
    fs::create_dir_all(out)?;
    fs::write(
        out.join("lowering.json"),
        serde_json::to_vec_pretty(&program.stats())?,
    )?;
    eprintln!("FRI verifier lowered: {}", program.stats());
    if !prove {
        return Ok(());
    }
    let geometry = program.packed_geometry();
    let mut built = program.build_packed(&assignment)?.into_ready();
    let rows = built.take_rows();
    let shape = &built;
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let profile = LigeritoProfile::Slim;
    let m = union.dense_m();
    let batch = embedded_initial_k_or_default(m, profile);
    assert_eq!(geometry["dense_m"], json!(m));
    assert_eq!(
        geometry["padded_witness_array_bytes"],
        json!((union.packed_len() as u128) * 16)
    );
    let params = PcsParams {
        m,
        log_inv_rate: profile.log_inv_rate(),
        log_batch_size: batch,
        profile,
        num_lanes: union.commit_lanes(batch),
        merkle_hash: flock_prover::merkle::HashKind::Blake3,
    };
    let prepare_seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "Flock shape ready: m={m}, public segment {}",
        built.public.len()
    );
    let start = Instant::now();
    let (outer, commitment) = if terminal {
        built.prove_with_challenger(
            rows,
            &params,
            &mut FsChallenger::with_chained_blake3(DOMAIN),
        )
    } else {
        built.prove(rows, &params, DOMAIN)
    };
    #[cfg(feature = "terminal")]
    if terminal {
        drop(union);
        return crate::terminal::census(out, built, &params, DOMAIN, &commitment, &outer, &[13]);
    }
    let circuits = built.circuits();
    let prove_seconds = start.elapsed().as_secs_f64();
    // Fixed constants precede application publics. Expected claim is independent
    // of both the FRI witness and Flock's proof-supplied data.
    let mut expected = built.public.clone();
    let last = expected.len() - 1;
    expected[last] = F128::new(13, 0);
    use crate::counts as op_count;
    op_count::reset();
    let start = Instant::now();
    verify_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &expected,
        &circuits,
        &commitment,
        &outer,
        &params,
        &mut FsChallenger::new(DOMAIN),
    )
    .map_err(|e| format!("outer: {e:?}"))?;
    let verify_seconds = start.elapsed().as_secs_f64();
    let full_ops = op_count::snapshot();
    let ((_, deferred, sigma), deferred_ops) = op_count::measure(|| {
        flock_prover::verifier::verify_ligerito_union_circuit_deferred(
            &union,
            &shape.circuit,
            &expected,
            &circuits,
            &commitment,
            &outer,
            &params,
            &mut FsChallenger::new(DOMAIN),
        )
        .unwrap()
    });
    let ((), matrix_ops) = op_count::measure(|| {
        assert!(deferred.element.is_none());
        deferred
            .boolean
            .as_ref()
            .unwrap()
            .check(&union, &circuits)
            .unwrap();
    });
    let (valid, wiring_ops) = op_count::measure(|| sigma.check(&shape.circuit));
    assert!(valid);
    let fixed_wiring = crate::static_wiring::measure(&shape.circuit, &sigma)?;
    let jagged_params = flock_prover::pcs::jagged::JaggedParams::from_heights(
        &union.jagged_heights(),
        union.n_log(),
        m - 7,
    );
    let (valid, layout_ops) = op_count::measure(|| deferred.jagged.check(&jagged_params));
    assert!(valid);
    // Deferred acceptance is conditional: corrupt each returned obligation
    // independently and require the corresponding discharge to reject it.
    let mut bad_matrix = deferred.boolean.clone().unwrap();
    bad_matrix.target += F128::ONE;
    assert!(bad_matrix.check(&union, &circuits).is_err());
    let mut bad_sigma = sigma;
    bad_sigma.value += F128::ONE;
    assert!(!bad_sigma.check(&shape.circuit));
    let mut bad_layout = deferred.jagged;
    bad_layout.m += 1;
    assert!(!bad_layout.check(&jagged_params));
    expected[last].lo += 1;
    assert!(
        verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &expected,
            &circuits,
            &commitment,
            &outer,
            &params,
            &mut FsChallenger::new(DOMAIN)
        )
        .is_err()
    );
    let bytes = bincode::serialize(&(&commitment, &outer))?;
    let (decoded_commitment, decoded): (
        flock_prover::pcs::Commitment,
        flock_prover::proof::R1csProofCircuitMerged,
    ) = bincode::deserialize(&bytes)?;
    expected[last] = F128::new(13, 0);
    verify_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &expected,
        &circuits,
        &decoded_commitment,
        &decoded,
        &params,
        &mut FsChallenger::new(DOMAIN),
    )
    .map_err(|e| format!("decoded: {e:?}"))?;
    let mut bad = decoded;
    bad.boolean.as_mut().unwrap().lincheck.z_partial[0].lo ^= 1;
    assert!(
        verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &expected,
            &circuits,
            &decoded_commitment,
            &bad,
            &params,
            &mut FsChallenger::new(DOMAIN)
        )
        .is_err()
    );
    fs::write(out.join("proof.bin"), &bytes)?;
    let report = json!({"scope":"Complete single-batch FRI verifier fixture in Flock; inner fixture has development security, not Init",
        "flock_rev":"b684b1258e4b1f202bec24afd660ace851b09e5e","profile":"slim","dense_m":m,"initial_k":batch,
        "source":format!("{:?}",circuit.stats()),"lowering":program.stats(),"geometry":geometry,"prepare_seconds":prepare_seconds,
        "prove_seconds":prove_seconds,"verify_seconds":verify_seconds,"proof_and_commitment_bytes":bytes.len(),
        "expected_claim":statement.claims[0][0].as_canonical_u64(),"verified":true,"altered_claim_rejected":true,
        "serialized_roundtrip_verified":true,"corrupted_lincheck_rejected":true,
        "full_verifier_f128_multiplications":full_ops.muls_excluding_inv(),"full_verifier_f128_inversions":full_ops.invs,
        "deferred_verifier_f128_multiplications":deferred_ops.muls_excluding_inv(),"deferred_verifier_f128_inversions":deferred_ops.invs,
        "deferred_obligations_discharged":true,"altered_obligations_rejected":true,
        "discharge_f128_multiplications":{"matrix":matrix_ops.muls_excluding_inv(),"wiring":wiring_ops.muls_excluding_inv(),"layout":layout_ops.muls_excluding_inv()},
        "discharge_f128_inversions":{"matrix":matrix_ops.invs,"wiring":wiring_ops.invs,"layout":layout_ops.invs},
        "groth16_flock_verifier_implemented":false,
        "counters_enabled":cfg!(feature = "counters"),
        "counter_scope":"Native operations when counters are enabled; null otherwise. Not a Groth16 constraint census. Deferred verification alone leaves obligations outstanding."});
    let mut report = report;
    report["fixed_wiring_pcs"] = fixed_wiring;
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/support/verifier_audit.rs"]
mod audit_fixture;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_multicircuit_fri_verifier_proves_in_flock_and_binds_publics() {
        let (inner, pk) = audit_fixture::setup();
        let claims = audit_fixture::claims();
        let proof = inner.prove_batch(
            &pk,
            vec![ShardInput {
                claims: claims.clone(),
                witness: audit_fixture::witness(&inner),
            }],
            vec![],
        );
        inner.verify_batch(&proof).unwrap();
        let key = VerifierKey::from_system(&inner);
        let plan = VerifierPlan::validate(
            &key,
            ProofProfile {
                envelope: Envelope::SingleBatch,
                active: audit_fixture::ACTIVE.to_vec(),
                log_degrees: audit_fixture::LOGS.to_vec(),
                claim_lengths: vec![2],
                message_lengths: vec![],
                max_field_retries: 2,
            },
            VerifierLimits::default(),
        )
        .unwrap();
        let (circuit, inputs) = plan
            .build(
                Statement {
                    claims: vec![vec![StatementSlot::Public; 2]],
                    messages: vec![],
                },
                ImplementationOptions::default(),
            )
            .unwrap();
        let statement = Statement {
            claims,
            messages: vec![],
        };
        let prepared = plan
            .expand_witness(ProofEnvelope::SingleBatch(&proof))
            .unwrap();
        let mut w = circuit.witness();
        inputs.assign_statement(&mut w, &statement).unwrap();
        prepared.assign_proof(&mut w, &inputs).unwrap();
        let assignment = w.generate().unwrap();
        for coalesced in [false, true, false, true] {
            let mut program = Program::compile(&circuit).unwrap();
            if coalesced {
                program.coalesce();
            }
            let mut ready = program.build_packed(&assignment).unwrap().into_ready();
            let rows = ready.take_rows();
            let union = UnionInstance::new(&ready.registry, ready.counts.clone());
            let profile = LigeritoProfile::Fast;
            let m = union.dense_m();
            let batch = flock_prover::pcs::ligerito::embedded_initial_k(m, profile).unwrap();
            let params = PcsParams {
                m,
                profile,
                log_inv_rate: profile.log_inv_rate(),
                log_batch_size: batch,
                num_lanes: union.commit_lanes(batch),
                merkle_hash: flock_prover::merkle::HashKind::Blake3,
            };
            params.ligerito_prover_config().unwrap();
            let (outer, commitment) = ready.prove(rows, &params, DOMAIN);
            let circuits = ready.circuits();
            let verify = |public: &[F128]| {
                verify_ligerito_union_circuit(
                    &union,
                    &ready.circuit,
                    public,
                    &circuits,
                    &commitment,
                    &outer,
                    &params,
                    &mut FsChallenger::new(DOMAIN),
                )
            };
            let mut expected = ready.public.clone();
            let start = expected.len() - 2;
            for (v, word) in expected[start..].iter_mut().zip(&statement.claims[0]) {
                *v = F128::new(word.as_canonical_u64(), 0);
            }
            verify(&expected).unwrap();
            // Includes circuit-fixed constants as well as both application words.
            for index in 0..expected.len() {
                let mut wrong = expected.clone();
                wrong[index] += F128::ONE;
                assert!(verify(&wrong).is_err(), "unbound public {index}");
            }
            println!(
                "mixed-height Flock audit: m={m}, {} bytes, {} bound public words",
                bincode::serialize(&(&commitment, &outer)).unwrap().len(),
                expected.len()
            );
        }
    }
}
