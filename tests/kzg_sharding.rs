#![cfg(feature = "kzg")]

use std::sync::Arc;

use multi_stark::{
    ark_adapter::{KzgConfig, Scalar, Srs},
    batch::Retention,
    plonkish::{CircuitBuilder, WitnessError},
    system::{System, SystemWitness},
    traits::{Algebra, Field},
};
use p3_matrix::Matrix;

#[test]
fn regenerated_kzg_shards_preserve_global_copy_constraints() {
    let f = Scalar::from_u8;
    let mut b = CircuitBuilder::new();
    let x = b.input("x");
    let double = b.add(x, x);
    b.mul(x, x);
    b.expose_public(double);
    let compiled = b.finish().lower_to_multi_stark_sharded(f(81), 2).unwrap();
    assert_eq!(compiled.main_heights(), [2, 2, 2]);
    let mut witness = compiled.witness();
    witness.set(x, f(5)).unwrap();
    let assignment = witness.generate().unwrap();
    let shards = compiled.trace_shards(&assignment).unwrap();
    assert_eq!(shards.len(), 3);
    assert!(matches!(
        shards.traces(3),
        Err(WitnessError::ShardIndex { .. })
    ));
    let claims = vec![compiled.claims(&[f(10)]).unwrap(), vec![], vec![]];
    let (system, key) = System::new(
        KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(2, b"shards")), 8),
        compiled.kzg_circuit_inputs(2, 8).unwrap(),
    );
    let mut calls = [0; 3];
    let proof = system.prove_batch_with(&key, &claims, vec![], Retention::Regenerate, |shard| {
        calls[shard] += 1;
        SystemWitness::from_stage_1(shards.traces(shard).unwrap(), &system)
    });
    assert_eq!(calls, [2; 3]);
    system.verify_batch(&proof).unwrap();
    assert!(
        proof
            .proofs
            .iter()
            .any(|p| p.intermediate_accumulators.last() != Some(&Scalar::ZERO))
    );
    let retained = system.prove_batch_with(&key, &claims, vec![], Retention::Retain, |shard| {
        SystemWitness::from_stage_1(shards.traces(shard).unwrap(), &system)
    });
    assert_eq!(retained.to_bytes().unwrap(), proof.to_bytes().unwrap());

    let forged = system.prove_batch_with(&key, &claims, vec![], Retention::Regenerate, |shard| {
        let mut traces = shards.traces(shard).unwrap();
        if shard == 1 {
            // 7*7=49 satisfies this shard's arithmetic, but x=5 elsewhere.
            assert_eq!(&traces[1].values[..3], &[f(5), f(5), f(25)]);
            traces[1].values[..3].copy_from_slice(&[f(7), f(7), f(49)]);
        }
        SystemWitness::from_stage_1(traces, &system)
    });
    assert!(system.verify_batch(&forged).is_err());

    // Reprove with a disconnected partition omitted, rather than merely
    // corrupting an existing proof. The required activation claim rejects it.
    let omitted =
        system.prove_batch_with(&key, &claims[..2], vec![], Retention::Regenerate, |shard| {
            SystemWitness::from_stage_1(shards.traces(shard).unwrap(), &system)
        });
    assert!(system.verify_batch(&omitted).is_err());
    let mut wrong = proof.clone();
    wrong.preamble.headers[0].claims[1][3] += Scalar::ONE;
    assert!(system.verify_batch(&wrong).is_err());
}

#[test]
fn shared_tables_are_generated_in_one_shard() {
    let f = Scalar::from_u8;
    let mut b = CircuitBuilder::new();
    let x = b.input("x");
    let table = b.fixed_table("small", (0..4).map(|v| vec![f(v)]).collect());
    for _ in 0..20 {
        b.lookup(table, &[x]);
    }
    b.expose_public(x);
    let compiled = b
        .finish()
        .lower_to_multi_stark_with_max_height(f(82), 8)
        .unwrap()
        .merge_table_traces(8)
        .unwrap();
    let mut witness = compiled.witness();
    witness.set(x, f(3)).unwrap();
    let assignment = witness.generate().unwrap();
    let shards = compiled.trace_shards(&assignment).unwrap();
    let all = compiled.traces(&assignment).unwrap();
    for ci in 0..all.len() {
        let mut copies = 0;
        for shard in 0..shards.len() {
            let traces = shards.traces(shard).unwrap();
            if traces[ci].height() != 0 {
                assert_eq!(traces[ci].values, all[ci].values);
                copies += 1;
            }
        }
        assert_eq!(copies, 1);
    }
    let (system, key) = System::new(
        KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(8, b"shard-table")), 8),
        compiled.kzg_circuit_inputs(8, 8).unwrap(),
    );
    let mut claims = vec![vec![]; shards.len()];
    claims[0] = compiled.claims(&[f(3)]).unwrap();
    let proof = system.prove_batch_with(&key, &claims, vec![], Retention::Regenerate, |shard| {
        SystemWitness::from_stage_1(shards.traces(shard).unwrap(), &system)
    });
    system.verify_batch(&proof).unwrap();
}

#[test]
fn regenerated_preprocessing_matches_resident_proof_and_binds_sources() {
    use multi_stark::{ark_adapter::sharded::ShardedKzg, expr::Expr, system::CircuitInputs};
    use p3_matrix::dense::RowMajorMatrix;
    let f = Scalar::from_u8;
    let definitions = vec![
        CircuitInputs {
            main_width: 1,
            constraints: vec![Expr::main(0) - Expr::constant(f(9))],
            ..Default::default()
        },
        CircuitInputs {
            main_width: 1,
            preprocessed: Some(RowMajorMatrix::new_col(vec![f(7); 4])),
            constraints: vec![Expr::main(0) - Expr::preprocessed(0)],
            ..Default::default()
        },
    ];
    let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(8, b"streamed-fixed")), 4);
    let mut streamed = ShardedKzg::new(config.clone(), definitions.clone());
    assert!(
        streamed
            .system
            .circuits
            .iter()
            .all(|c| c.preprocessed.is_none())
    );
    let (resident, key) = System::new(config, definitions.clone());
    let traces = |shard: usize| {
        vec![
            RowMajorMatrix::new_col(if shard == 0 { vec![f(9); 2] } else { vec![] }),
            RowMajorMatrix::new_col(if shard == 1 { vec![f(7); 4] } else { vec![] }),
        ]
    };
    let claims = vec![vec![], vec![]];
    let schedule = vec![vec![0], vec![1]];
    let proof = streamed
        .prove(&claims, &schedule, |ci| definitions[ci].clone(), traces)
        .unwrap();
    streamed.verify(&proof, &claims, &schedule).unwrap();
    resident.verify_batch(&proof).unwrap();
    let expected =
        resident.prove_batch_with(&key, &claims, vec![], Retention::Regenerate, |shard| {
            SystemWitness::from_stage_1(traces(shard), &resident)
        });
    assert_eq!(proof.to_bytes().unwrap(), expected.to_bytes().unwrap());
    assert!(
        streamed
            .system
            .circuits
            .iter()
            .all(|c| c.preprocessed.is_none())
    );
    assert!(
        streamed
            .verify(&proof, &claims, &[vec![1], vec![0]])
            .is_err()
    );
    assert!(
        streamed
            .verify(&proof, &claims, &[vec![0], vec![0]])
            .is_err()
    );
    let mut wrong_claims = claims.clone();
    wrong_claims[0].push(vec![f(9)]);
    assert!(streamed.verify(&proof, &wrong_claims, &schedule).is_err());
    for changed_graph in [false, true] {
        let result = streamed.prove(
            &claims,
            &schedule,
            |ci| {
                let mut input = definitions[ci].clone();
                if ci == 1 {
                    if changed_graph {
                        input.constraints = vec![Expr::main(0) - Expr::constant(f(7))];
                    } else {
                        input.preprocessed.as_mut().unwrap().values[0] += Scalar::ONE;
                    }
                }
                input
            },
            traces,
        );
        assert!(result.is_err());
        assert!(
            streamed
                .system
                .circuits
                .iter()
                .all(|c| c.preprocessed.is_none())
        );
    }
    let mut calls = [0; 2];
    let changed_witness = streamed.prove(
        &claims,
        &schedule,
        |ci| definitions[ci].clone(),
        |shard| {
            calls[shard] += 1;
            let mut result = traces(shard);
            if calls[shard] == 2 {
                result[shard].values[0] += Scalar::ONE;
            }
            result
        },
    );
    assert!(matches!(
        changed_witness,
        Err("regenerated witness differs from round one")
    ));
}

#[test]
fn individual_hash_shards_bind_the_digest() {
    use multi_stark::{
        ark_adapter::sharded::ShardedKzg,
        batch::BatchPreamble,
        plonkish::gadgets::{ByteGadgets, blake3},
    };
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::CryptographicHasher;
    let mut b = CircuitBuilder::<Scalar>::new();
    b.enable_compact_blake3();
    let bytes = ByteGadgets::new(&mut b);
    let input = bytes.input(&mut b, "byte");
    let digest = blake3(&mut b, &bytes, &vec![input; 1025]);
    for byte in digest {
        b.expose_public(byte.value());
    }
    let c = b
        .finish()
        .lower_to_multi_stark_sharded(Scalar::from_u8(91), 1 << 16)
        .unwrap();
    let mut w = c.witness();
    w.set(input.value(), Scalar::from_u8(5)).unwrap();
    let assignment = w.generate().unwrap();
    let native: [u8; 32] = p3_blake3::Blake3.hash_iter(std::iter::repeat_n(5u8, 1025));
    let public = native.map(Scalar::from_u8);
    assert_eq!(assignment.public_values(), public);
    let source = c.trace_shards(&assignment).unwrap();
    let count = c.num_circuits();
    let definition = |i| c.kzg_circuit_input(i, 1 << 16, 8).unwrap().unwrap();
    let widths: Vec<_> = (0..count)
        .map(|i| c.circuit_input(i).unwrap().main_width)
        .collect();
    let traces = |i| {
        let mut result: Vec<_> = widths
            .iter()
            .map(|&w| RowMajorMatrix::new(vec![], w))
            .collect();
        result[i] = source.trace(i).unwrap();
        result
    };
    let mut prover = ShardedKzg::new(
        KzgConfig::new(
            Arc::new(Srs::unsafe_dev_setup(1 << 16, b"individual-hash")),
            8,
        ),
        (0..count).map(definition),
    );
    let schedule: Vec<_> = (0..count).map(|i| vec![i]).collect();
    let mut claims = vec![vec![]; count];
    claims[0] = c.claims(&public).unwrap();
    let headers = (0..count)
        .map(|i| {
            prover
                .commit_shard(&claims[i], &schedule[i], traces(i))
                .unwrap()
        })
        .collect();
    let preamble = BatchPreamble {
        headers,
        messages: vec![],
    };
    // Resume from a serialized preamble, as the disk-backed runner does.
    let bytes = bincode::serde::encode_to_vec(&preamble, bincode::config::standard()).unwrap();
    let (preamble, _): (BatchPreamble<KzgConfig>, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    let proofs = (0..count)
        .map(|i| {
            prover
                .prove_shard(&preamble, i, definition, traces(i))
                .unwrap()
        })
        .collect();
    let proof = multi_stark::batch::BatchProof { preamble, proofs };
    prover.verify(&proof, &claims, &schedule).unwrap();
    claims[0][1][3] += Scalar::ONE;
    assert!(prover.verify(&proof, &claims, &schedule).is_err());
    assert!(
        prover
            .prove_shard(&proof.preamble, count, definition, traces(0))
            .is_err()
    );
}
