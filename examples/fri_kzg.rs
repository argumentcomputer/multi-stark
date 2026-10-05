//! Compare KZG proofs of the same Goldilocks FRI verification circuit.
//! Development parameters and a deterministic, insecure test SRS.

#[cfg(not(feature = "kzg"))]
fn main() {
    eprintln!("enable --features kzg,parallel");
}

#[cfg(feature = "kzg")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use multi_stark::{
        ark_adapter::{config::KzgConfig, field::Scalar, srs::Srs},
        batch::{BatchProof, Retention},
        expr::Expr,
        lookup::Lookup,
        plonkish::{foreign::GoldilocksCircuit, verifier::*},
        system::{CircuitInputs, System, SystemWitness},
        traits::{Algebra, Field},
        types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val},
    };
    use p3_matrix::{Matrix, dense::RowMajorMatrix};
    use std::{sync::Arc, time::Instant};
    let args: Vec<_> = std::env::args().skip(1).collect();
    let generic = !args.iter().any(|s| s == "--compact");
    let check_only = args.iter().any(|s| s == "--check-only");
    let estimate = args.iter().any(|s| s == "--estimate");
    let compare = args.iter().any(|s| s == "--compare");
    let merge_tables = args.iter().any(|s| s == "--merge-tables");
    let batch = args.iter().any(|s| s == "--batch");
    let stream = args.iter().any(|s| s == "--stream-preprocessing");
    if stream && !batch {
        return Err("--stream-preprocessing requires --batch".into());
    }
    let partition_log = args
        .iter()
        .filter_map(|s| s.strip_prefix("--partition-log="))
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if partition_log.len() > 1 {
        return Err("supply --partition-log only once".into());
    }
    let partition_height = partition_log
        .first()
        .map(|&log| 1usize.checked_shl(log).ok_or("invalid partition log"))
        .transpose()?;
    if args.iter().any(|s| {
        ![
            "--compact",
            "--check-only",
            "--estimate",
            "--compare",
            "--merge-tables",
            "--batch",
            "--stream-preprocessing",
        ]
        .contains(&s.as_str())
            && !s.starts_with("--partition-log=")
    }) {
        return Err(
            "usage: fri_kzg [--compact] [--check-only] [--estimate] [--compare] [--merge-tables] [--partition-log=N] [--batch] [--stream-preprocessing]"
                .into(),
        );
    }
    let start = Instant::now();
    let config = GoldilocksBlake3Config::new(
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
    let claim_refs: Vec<_> = statement.claims.iter().map(Vec::as_slice).collect();
    let proof = inner.prove_multiple_claims(
        &pk,
        &claim_refs,
        SystemWitness::from_stage_1(
            vec![RowMajorMatrix::new_col(vec![Val::from_u8(13); 2])],
            &inner,
        ),
    );
    inner.verify_multiple_claims(&claim_refs, &proof).unwrap();
    let key = VerifierKey::from_system(&inner);
    let plan = VerifierPlan::validate(
        &key,
        ProofProfile {
            envelope: Envelope::Ordinary,
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
            compact_blake3: !generic,
        },
    )?;
    if estimate {
        println!(
            "source={:?}, scalar={:?}",
            circuit.stats(),
            GoldilocksCircuit::estimate(&circuit)
        );
        return Ok(());
    }
    let prepared = plan.expand_witness(ProofEnvelope::Ordinary {
        proof: &proof,
        claims: &statement.claims,
    })?;
    let mut witness = circuit.witness();
    inputs.assign_statement(&mut witness, &statement)?;
    prepared.assign_proof(&mut witness, &inputs)?;
    let assignment = witness.generate()?;
    let foreign = GoldilocksCircuit::new(&circuit);
    let mut witness = foreign.circuit.witness();
    foreign.assign(&assignment, &mut witness)?;
    let assignment = witness.generate()?;
    println!(
        "hash={}, inner_bytes={}, scalar_circuit={:?}, build_and_witness={:?}",
        if generic { "generic" } else { "compact" },
        proof.to_bytes()?.len(),
        foreign.circuit.stats(),
        start.elapsed()
    );
    let compiled = if stream {
        foreign.circuit.lower_to_multi_stark_sharded(
            Scalar::from_u8(93),
            partition_height.unwrap_or(1 << 18),
        )?
    } else if let Some(height) = partition_height {
        foreign
            .circuit
            .lower_to_multi_stark_with_max_height(Scalar::from_u8(93), height)?
    } else {
        foreign.circuit.lower_to_multi_stark(Scalar::from_u8(93))?
    };
    let compiled = if merge_tables {
        compiled.merge_table_traces(1 << 22)?
    } else {
        compiled
    };
    let max_height = (0..compiled.num_circuits())
        .map(|i| compiled.circuit_input(i).unwrap())
        .map(|c| c.preprocessed.as_ref().unwrap().height())
        .max()
        .unwrap();
    println!(
        "max_height={max_height}, traces={}",
        compiled.num_circuits()
    );
    if check_only {
        return Ok(());
    }
    let start = Instant::now();
    let srs = Arc::new(Srs::unsafe_dev_setup(max_height, b"fri-kzg-example"));
    println!("SRS: {:?}", start.elapsed());
    let claims = compiled.claims(&[Scalar::from_u8(13)])?;
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    for &group in if compare { &[1, 0][..] } else { &[0][..] } {
        if stream {
            use multi_stark::ark_adapter::sharded::ShardedKzg;
            let definition = |i| {
                if group == 0 {
                    compiled
                        .kzg_circuit_input(i, max_height, 8)
                        .unwrap()
                        .unwrap()
                } else {
                    let mut input = compiled.circuit_input(i).unwrap();
                    input.lookup_group_size = group;
                    input
                }
            };
            let start = Instant::now();
            let mut prover = ShardedKzg::new(
                KzgConfig::new(srs.clone(), 8),
                (0..compiled.num_circuits()).map(definition),
            );
            let shards = compiled.trace_shards(&assignment)?;
            let mut schedule: Vec<_> = (0..compiled.main_heights().len())
                .map(|i| vec![i])
                .collect();
            if schedule.len() < shards.len() {
                schedule.push((compiled.main_heights().len()..compiled.num_circuits()).collect());
            }
            let mut shard_claims = vec![vec![]; shards.len()];
            shard_claims[0] = claims.clone();
            let proof = prover.prove(&shard_claims, &schedule, definition, |i| {
                shards.traces(i).unwrap()
            })?;
            prover.verify(&proof, &shard_claims, &schedule)?;
            let bytes = proof.to_bytes()?;
            let mut decoded = BatchProof::<KzgConfig>::from_bytes(&bytes)?;
            prover.verify(&decoded, &shard_claims, &schedule)?;
            decoded.preamble.headers[0].claims[1][3] += Scalar::ONE;
            assert!(prover.verify(&decoded, &shard_claims, &schedule).is_err());
            assert!(
                prover
                    .system
                    .circuits
                    .iter()
                    .all(|c| c.preprocessed.is_none())
            );
            println!(
                "group={group}, shards={}, partition_heights={:?}, batch_bytes={}, setup_and_prove={:?}; verified, altered claim rejected, preprocessing and witnesses regenerated",
                shards.len(),
                compiled.main_heights(),
                bytes.len(),
                start.elapsed()
            );
            continue;
        }
        let mut circuits = if group == 0 {
            compiled.kzg_circuit_inputs(max_height, 8)?
        } else {
            compiled.circuit_inputs()
        };
        if group != 0 {
            for c in &mut circuits {
                c.lookup_group_size = group;
            }
        }
        let start = Instant::now();
        let (outer, pk) = System::new(KzgConfig::new(srs.clone(), 8), circuits);
        let widths: Vec<_> = outer
            .circuits
            .iter()
            .map(|c| {
                (
                    c.main_width,
                    c.preprocessed_width,
                    c.stage_2_width,
                    c.quotient_degree(),
                )
            })
            .collect();
        if batch {
            let shards = compiled.trace_shards(&assignment)?;
            let mut shard_claims = vec![vec![]; shards.len()];
            shard_claims[0] = claims.clone();
            let mut calls = vec![0; shards.len()];
            let proof = outer.prove_batch_with(
                &pk,
                &shard_claims,
                vec![],
                Retention::Regenerate,
                |shard| {
                    calls[shard] += 1;
                    SystemWitness::from_stage_1(
                        shards.traces(shard).expect("validated assignment"),
                        &outer,
                    )
                },
            );
            assert!(calls.iter().all(|&n| n == 2));
            // Application policy is trusted independently of the proof:
            // every partition once, fixed claims, no extra balance messages.
            let accepts = |proof: &BatchProof<KzgConfig>| {
                proof.preamble.headers.len() == shards.len()
                    && proof.preamble.messages.is_empty()
                    && proof
                        .preamble
                        .headers
                        .iter()
                        .enumerate()
                        .all(|(shard, header)| {
                            let active: Vec<_> = (0..outer.circuits.len())
                                .map(|ci| {
                                    if shard < compiled.main_heights().len() {
                                        ci == shard
                                    } else {
                                        ci >= compiled.main_heights().len()
                                    }
                                })
                                .collect();
                            let logs: Vec<_> = outer
                                .circuits
                                .iter()
                                .zip(&active)
                                .filter(|(_, on)| **on)
                                .map(|(c, _)| u8::try_from(c.preprocessed_height.ilog2()).unwrap())
                                .collect();
                            header.active == active
                                && header.log_degrees == logs
                                && header.claims == shard_claims[shard]
                        })
                    && outer.verify_batch(proof).is_ok()
            };
            assert!(accepts(&proof));
            let bytes = proof.to_bytes()?;
            let mut decoded = BatchProof::<KzgConfig>::from_bytes(&bytes)?;
            assert!(accepts(&decoded));
            decoded.preamble.headers[0].claims[1][3] += Scalar::ONE;
            assert!(!accepts(&decoded));
            println!(
                "group={group}, shards={}, partition_heights={:?}, batch_bytes={}, setup_and_prove={:?}; verified, altered claim rejected, witnesses regenerated",
                shards.len(),
                compiled.main_heights(),
                bytes.len(),
                start.elapsed()
            );
            continue;
        }
        let proof = outer.prove_multiple_claims(
            &pk,
            &refs,
            SystemWitness::from_stage_1(compiled.traces(&assignment)?, &outer),
        );
        outer.verify_multiple_claims(&refs, &proof).unwrap();
        let codec =
            multi_stark::ark_adapter::compact::FixedProofCodec::new(&outer, &proof.log_degrees)?;
        let compact_bytes = codec.encode(&proof)?;
        let decoded = codec.decode(&compact_bytes)?;
        outer.verify_multiple_claims(&refs, &decoded).unwrap();
        assert_eq!(decoded.to_bytes()?, proof.to_bytes()?);
        let mut wrong = claims.clone();
        wrong[1][3] += Scalar::ONE;
        assert!(
            outer
                .verify_multiple_claims(
                    &wrong.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &proof
                )
                .is_err()
        );
        println!(
            "group={group}, columns(main,fixed,lookup,quotient)={widths:?}, proof_bytes={}, compact_bytes={}, setup_and_prove={:?}; verified, altered claim rejected",
            proof.to_bytes()?.len(),
            compact_bytes.len(),
            start.elapsed()
        );
    }
    Ok(())
}
