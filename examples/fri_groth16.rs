//! Direct Groth16 proof of a small FRI verification fixture. Insecure test setup.
#[cfg(not(feature = "groth16"))]
fn main() {
    eprintln!("enable --features groth16,parallel");
}
#[cfg(feature = "groth16")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_groth16::{Groth16, prepare_verifying_key};
    use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use ark_std::rand::{SeedableRng, rngs::StdRng};
    use multi_stark::{
        batch::ShardInput,
        expr::Expr,
        lookup::Lookup,
        plonkish::{foreign::GoldilocksCircuit, r1cs::R1csCircuit, verifier::*},
        system::{CircuitInputs, System, SystemWitness},
        traits::{Algebra, Field},
        types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val},
    };
    use p3_matrix::dense::RowMajorMatrix;
    use std::time::Instant;
    let start = Instant::now();
    let sharded = std::env::args().any(|a| a == "--shards");
    let ordinary = std::env::args().any(|a| a == "--ordinary");
    let config = GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: if sharded { 2 } else { 1 },
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    );
    let (inner, pk) = System::new(
        config,
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
    inner.verify_batch(&proof).unwrap();
    let ordinary_proof = ordinary.then(|| {
        let proof = inner.prove_multiple_claims(
            &pk,
            &[&statement.claims[0]],
            SystemWitness::from_stage_1(
                vec![RowMajorMatrix::new_col(vec![Val::from_u8(13); 2])],
                &inner,
            ),
        );
        inner
            .verify_multiple_claims(&[&statement.claims[0]], &proof)
            .unwrap();
        proof
    });
    let key = VerifierKey::from_system(&inner);
    let plan = VerifierPlan::validate(
        &key,
        ProofProfile {
            envelope: if ordinary {
                Envelope::Ordinary
            } else {
                Envelope::SingleBatch
            },
            active: vec![true],
            log_degrees: vec![1],
            claim_lengths: vec![1],
            message_lengths: vec![],
            max_field_retries: 2,
        },
        VerifierLimits::default(),
    )?;
    let schema = Statement {
        claims: vec![vec![StatementSlot::Public]],
        messages: vec![],
    };
    let options = ImplementationOptions {
        compact_blake3: true,
    };
    if sharded {
        if let Some(proof) = &ordinary_proof {
            let alternate = inner.prove_multiple_claims(
                &pk,
                &[&statement.claims[0]],
                SystemWitness::from_stage_1(
                    vec![RowMajorMatrix::new_col(vec![
                        Val::from_u8(13),
                        Val::from_u8(14),
                    ])],
                    &inner,
                ),
            );
            inner
                .verify_multiple_claims(&[&statement.claims[0]], &alternate)
                .unwrap();
            return prove_shards(
                &plan,
                schema,
                options,
                &statement,
                ProofEnvelope::Ordinary {
                    proof,
                    claims: &statement.claims,
                },
                ProofEnvelope::Ordinary {
                    proof: &alternate,
                    claims: &statement.claims,
                },
            );
        }
        let alternate = inner.prove_batch(
            &pk,
            vec![ShardInput {
                claims: statement.claims.clone(),
                witness: SystemWitness::from_stage_1(
                    vec![RowMajorMatrix::new_col(vec![
                        Val::from_u8(13),
                        Val::from_u8(14),
                    ])],
                    &inner,
                ),
            }],
            vec![],
        );
        inner.verify_batch(&alternate).unwrap();
        return prove_shards(
            &plan,
            schema,
            options,
            &statement,
            ProofEnvelope::SingleBatch(&proof),
            ProofEnvelope::SingleBatch(&alternate),
        );
    }
    let estimate = plan.estimate(schema.clone(), options)?;
    let (circuit, inputs) = plan.build(schema, options)?;
    assert_eq!(estimate, circuit.stats());
    let prepared = plan.expand_witness(match &ordinary_proof {
        Some(proof) => ProofEnvelope::Ordinary {
            proof,
            claims: &statement.claims,
        },
        None => ProofEnvelope::SingleBatch(&proof),
    })?;
    let mut witness = circuit.witness();
    inputs.assign_statement(&mut witness, &statement)?;
    prepared.assign_proof(&mut witness, &inputs)?;
    let assignment = witness.generate()?;
    let count = multi_stark::plonkish::r1cs::estimate_goldilocks(&circuit)?;
    let foreign = GoldilocksCircuit::new_with_expanded_hashes(&circuit);
    let mut witness = foreign.circuit.witness();
    foreign.assign(&assignment, &mut witness)?;
    let assignment = witness.generate()?;

    println!("FRI verifier built and assigned: {:?}", start.elapsed());
    let r1cs = R1csCircuit::new(&foreign.circuit, Some(&assignment))?;
    let cs = ConstraintSystem::<Fr>::new_ref();
    r1cs.generate_constraints(cs.clone())?;
    assert!(cs.is_satisfied()?);
    assert_eq!(count.constraints, cs.num_constraints());
    assert_eq!(count.witnesses, cs.num_witness_variables());
    assert_eq!(count.public_inputs + 1, cs.num_instance_variables());
    println!(
        "R1CS: {} constraints, {} witnesses, {} public inputs",
        cs.num_constraints(),
        cs.num_witness_variables(),
        cs.num_instance_variables() - 1
    );
    drop(cs);
    if std::env::args().any(|a| a == "--check-only") {
        return Ok(());
    }
    let mut rng = StdRng::seed_from_u64(42);
    let start = Instant::now();
    let pk = Groth16::<Bls12_381>::generate_random_parameters_with_reduction(
        R1csCircuit::new(&foreign.circuit, None)?,
        &mut rng,
    )?;
    println!("Development setup: {:?}", start.elapsed());
    let start = Instant::now();
    let proof = Groth16::<Bls12_381>::create_random_proof_with_reduction(r1cs, &pk, &mut rng)?;
    let pvk = prepare_verifying_key(&pk.vk);
    let expected = [Fr::from(13u64)];
    assert!(Groth16::<Bls12_381>::verify_proof(&pvk, &proof, &expected)?);
    let mut bytes = vec![];
    proof.serialize_compressed(&mut bytes)?;
    let decoded = ark_groth16::Proof::<Bls12_381>::deserialize_compressed(bytes.as_slice())?;
    assert!(Groth16::<Bls12_381>::verify_proof(
        &pvk, &decoded, &expected
    )?);
    assert!(!Groth16::<Bls12_381>::verify_proof(
        &pvk,
        &decoded,
        &[Fr::from(14u64)]
    )?);
    println!(
        "Small FRI fixture (ordinary={ordinary}) proved and verified: {} bytes, {:?}; altered claim rejected. Development setup only; not Init.",
        bytes.len(),
        start.elapsed()
    );
    Ok(())
}

#[cfg(feature = "groth16")]
fn prove_shards(
    plan: &multi_stark::plonkish::verifier::VerifierPlan<'_>,
    schema: multi_stark::plonkish::verifier::Statement<
        multi_stark::plonkish::verifier::StatementSlot,
    >,
    options: multi_stark::plonkish::verifier::ImplementationOptions,
    statement: &multi_stark::plonkish::verifier::Statement<multi_stark::types::Val>,
    proof: multi_stark::plonkish::verifier::ProofEnvelope<'_>,
    alternate: multi_stark::plonkish::verifier::ProofEnvelope<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_groth16::{Groth16, prepare_verifying_key};
    use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use ark_std::rand::{SeedableRng, rngs::StdRng};
    use multi_stark::plonkish::{
        foreign::GoldilocksCircuit,
        r1cs::{R1csCircuit, estimate_goldilocks},
        verifier::*,
    };
    use p3_field::PrimeField64;
    let partition = QueryShardPlan::new(plan, schema, options, 1)?;
    assert_eq!(partition.shard_count(), 3);
    assert_eq!(partition.query_range(0)?, 0..0);
    assert_eq!(partition.query_range(1)?, 0..1);
    assert_eq!(partition.query_range(2)?, 1..2);
    assert!(partition.build(3).is_err());
    let prepared = plan.expand_witness(proof)?;
    let alternate = plan.expand_witness(alternate)?;
    let mut rng = StdRng::seed_from_u64(42); // Insecure development setup.
    let mut keys = Vec::new();
    let mut proofs = Vec::new();
    let mut context = None;
    let mut mixed = None;
    for shard in 0..partition.shard_count() {
        let start = std::time::Instant::now();
        let (c, inputs) = partition.build(shard)?;
        assert_eq!(partition.estimate(shard)?, c.stats());
        let mut w = c.witness();
        inputs.assign_statement(&mut w, statement)?;
        prepared.assign_proof(&mut w, &inputs)?;
        let a = w.generate()?;
        let digest: [u8; 32] = a.public_values()[1..33]
            .iter()
            .map(|v| u8::try_from(v.as_canonical_u64()).unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        if let Some(expected) = context {
            assert_eq!(digest, expected);
        } else {
            context = Some(digest);
        }
        let stats = estimate_goldilocks(&c)?;
        let foreign = GoldilocksCircuit::new_with_expanded_hashes(&c);
        let mut w = foreign.circuit.witness();
        foreign.assign(&a, &mut w)?;
        let a = w.generate()?;
        let circuit = R1csCircuit::new(&foreign.circuit, Some(&a))?;
        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone())?;
        assert!(cs.is_satisfied()?);
        assert_eq!(stats.constraints, cs.num_constraints());
        assert_eq!(stats.witnesses, cs.num_witness_variables());
        drop(cs);
        println!(
            "Shard {shard}, queries {:?}: {stats:?}",
            partition.query_range(shard)?
        );
        let pk = Groth16::<Bls12_381>::generate_random_parameters_with_reduction(
            R1csCircuit::new(&foreign.circuit, None)?,
            &mut rng,
        )?;
        let proof =
            Groth16::<Bls12_381>::create_random_proof_with_reduction(circuit, &pk, &mut rng)?;
        let mut serialized = vec![];
        proof.serialize_compressed(&mut serialized)?;
        assert_eq!(serialized.len(), 192);
        proofs.push(ark_groth16::Proof::deserialize_compressed(
            serialized.as_slice(),
        )?);
        keys.push(prepare_verifying_key(&pk.vk));
        if shard == 1 {
            let mut w = c.witness();
            inputs.assign_statement(&mut w, statement)?;
            alternate.assign_proof(&mut w, &inputs)?;
            let a = w.generate()?;
            let other: [u8; 32] = a.public_values()[1..33]
                .iter()
                .map(|v| u8::try_from(v.as_canonical_u64()).unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            assert_ne!(other, digest);
            let mut w = foreign.circuit.witness();
            foreign.assign(&a, &mut w)?;
            let a = w.generate()?;
            let p = Groth16::<Bls12_381>::create_random_proof_with_reduction(
                R1csCircuit::new(&foreign.circuit, Some(&a))?,
                &pk,
                &mut rng,
            )?;
            let mut public = vec![Fr::from(13u64)];
            public.extend(other.map(Fr::from));
            public.push(Fr::from(1u64));
            assert!(Groth16::<Bls12_381>::verify_proof(&keys[1], &p, &public)?);
            mixed = Some(p);
        }
        println!("Shard {shard} proved in {:?}", start.elapsed());
    }
    let context = context.unwrap();
    let expected = [Fr::from(13u64)];
    assert!(verify_query_bundle(&keys, &proofs, &expected, &context)?);
    assert!(!verify_query_bundle(
        &keys,
        &proofs[..2],
        &expected,
        &context
    )?);
    assert!(!verify_query_bundle(
        &keys,
        &proofs,
        &[Fr::from(14u64)],
        &context
    )?);
    let mut bad_context = context;
    bad_context[0] ^= 1;
    assert!(!verify_query_bundle(
        &keys,
        &proofs,
        &expected,
        &bad_context
    )?);
    let mut bad = proofs.clone();
    bad.swap(1, 2);
    assert!(!verify_query_bundle(&keys, &bad, &expected, &context)?);
    bad = proofs.clone();
    bad[2] = bad[1].clone();
    assert!(!verify_query_bundle(&keys, &bad, &expected, &context)?);
    bad = proofs.clone();
    bad[1] = mixed.unwrap();
    assert!(!verify_query_bundle(&keys, &bad, &expected, &context)?);
    let values = &statement.claims[0];
    let encoded = encode_query_bundle(&keys, &proofs, values, &context)?;
    assert_eq!(encoded.len(), 652);
    assert!(verify_encoded_query_bundle(&keys, values, &encoded)?);
    for index in [0, 4, 36, 44, 76] {
        let mut bad = encoded.clone();
        bad[index] ^= 1;
        assert!(!verify_encoded_query_bundle(&keys, values, &bad)?);
    }
    assert!(!verify_encoded_query_bundle(
        &keys,
        values,
        &encoded[..encoded.len() - 1]
    )?);
    let mut trailing = encoded.clone();
    trailing.push(0);
    assert!(!verify_encoded_query_bundle(&keys, values, &trailing)?);
    std::fs::write("target/fri-groth16-shards.bin", &encoded)?;
    println!(
        "Binary bundle: {} bytes including statement, context, version, and trusted key-set identifier",
        encoded.len()
    );
    println!(
        "Three-shard FRI bundle verified: 576 proof bytes + 32 context bytes; missing, reordered, duplicated, mixed-proof shards and altered statement/context rejected. Insecure development setup; not Init."
    );
    Ok(())
}
