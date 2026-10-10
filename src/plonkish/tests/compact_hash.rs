use super::*;
use crate::plonkish::gadgets::{ByteGadgets, blake3};
use p3_blake3::Blake3;
use p3_symmetric::CryptographicHasher;

fn hash_config() -> GoldilocksBlake3Config {
    GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 2,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 32,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    )
}

#[test]
fn compact_hash_all_block_and_tree_boundaries_match_native() {
    let mut b = CircuitBuilder::<Val>::new();
    b.enable_compact_blake3();
    let bytes = ByteGadgets::new(&mut b);
    let mut inputs = vec![];
    for len in [0, 1, 63, 64, 65, 1023, 1024, 1025, 2048, 2049, 4096, 5000] {
        let input: Vec<_> = (0..len)
            .map(|i| bytes.input(&mut b, &format!("{len}/{i}")))
            .collect();
        let digest = blake3(&mut b, &bytes, &input);
        for byte in digest {
            b.expose_public(byte.value());
        }
        inputs.push(input);
    }
    let circuit = b.finish();
    let layout = circuit.multi_stark_layout().unwrap();
    assert_eq!(circuit.stats().hash_calls, 12);
    assert_eq!(layout.custom_traces.len(), 9);
    let compiled = circuit.lower_to_multi_stark(Val::from_u8(93)).unwrap();
    let defs = compiled.circuit_inputs();
    for seed in [0u8, 197] {
        let mut w = compiled.witness();
        let mut expected = vec![];
        for input in &inputs {
            let message: Vec<_> = (0..input.len())
                .map(|i| u8::try_from(i % 251).unwrap().wrapping_add(seed))
                .collect();
            for (wire, &byte) in input.iter().zip(&message) {
                w.set(wire.value(), Val::from_u8(byte)).unwrap();
            }
            let digest: [u8; 32] = Blake3.hash_iter(message);
            expected.extend(digest.map(Val::from_u8));
        }
        let a = w.generate().unwrap();
        assert_eq!(a.public_values(), expected);
        let shards = compiled.trace_shards(&a).unwrap();
        let streamed = std::thread::scope(|scope| {
            let pending: Vec<_> = (0..compiled.num_circuits())
                .rev()
                .map(|index| {
                    let shards = &shards;
                    scope.spawn(move || (index, shards.trace(index).unwrap()))
                })
                .collect();
            pending
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        let traces = compiled.traces(&a).unwrap();
        for (index, trace) in traces.iter().enumerate() {
            assert_eq!(
                compiled.trace_dimensions(index),
                Some((trace.height(), trace.width()))
            );
        }
        for (index, trace) in streamed {
            assert_eq!(trace, traces[index]);
        }
        let regenerated = compiled.trace_shards(&a).unwrap();
        let auxiliary = regenerated.traces(regenerated.len() - 1).unwrap();
        for index in compiled.num_circuits() - 9..compiled.num_circuits() {
            assert_eq!(regenerated.trace(index).unwrap(), traces[index]);
            assert_eq!(auxiliary[index], traces[index]);
        }
        let claims = compiled.claims(&expected).unwrap();
        assert!(relations_hold(&defs, &traces, &claims));
        assert_eq!(
            layout.advice_cells,
            traces.iter().map(|t| t.values.len()).sum::<usize>()
        );
        assert_eq!(
            layout.preprocessed_cells,
            defs.iter()
                .map(|d| d.preprocessed.as_ref().unwrap().values.len())
                .sum::<usize>()
        );
    }
}

#[test]
fn compact_hash_bridge_rejects_valid_hash_traces_for_a_different_witness() {
    let mut b = CircuitBuilder::<Val>::new();
    b.enable_compact_blake3();
    let bytes = ByteGadgets::new(&mut b);
    let input: Vec<_> = (0..65)
        .map(|i| bytes.input(&mut b, &format!("input{i}")))
        .collect();
    let digest = blake3(&mut b, &bytes, &input);
    // Feed a custom hash result back into a second custom call through normal
    // ByteValue wires, as the challenger and Merkle verifier do.
    let digest = blake3(&mut b, &bytes, &digest);
    for byte in digest {
        b.expose_public(byte.value());
    }
    let compiled = b.finish().lower_to_multi_stark(Val::from_u8(94)).unwrap();
    let defs = compiled.circuit_inputs();
    let (system, key) = System::new(hash_config(), defs.clone());
    let mut saved = vec![];
    for seed in [0u8, 23] {
        let message: Vec<_> = (0..65)
            .map(|i| u8::try_from(i).unwrap().wrapping_add(seed))
            .collect();
        let first: [u8; 32] = Blake3.hash_iter(message.iter().copied());
        let expected: [u8; 32] = Blake3.hash_iter(first);
        let mut w = compiled.witness();
        for (wire, byte) in input.iter().zip(message) {
            w.set(wire.value(), Val::from_u8(byte)).unwrap();
        }
        let a = w.generate().unwrap();
        let traces = compiled.traces(&a).unwrap();
        let claims = compiled.claims(&expected.map(Val::from_u8)).unwrap();
        assert!(relations_hold(&defs, &traces, &claims));
        let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
        let proof = system.prove_multiple_claims(
            &key,
            &refs,
            SystemWitness::from_stage_1(traces.clone(), &system),
        );
        system.verify_multiple_claims(&refs, &proof).unwrap();
        saved.push((traces, claims));
    }
    let custom_start = defs.len() - 9;
    let mut spliced = saved[0].0.clone();
    spliced[custom_start..].clone_from_slice(&saved[1].0[custom_start..]);
    let claims = &saved[0].1;
    assert!(!relations_hold(&defs, &spliced, claims));
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let proof =
        system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(spliced, &system));
    assert!(system.verify_multiple_claims(&refs, &proof).is_err());
    for t in custom_start..custom_start + 6 {
        let mut bad = saved[0].0.clone();
        bad[t].values.clear();
        assert!(!relations_hold(&defs, &bad, claims));
    }
}

#[cfg(feature = "kzg")]
#[test]
#[ignore = "isolated staging timing; generates no proof or SRS"]
fn compact_hash_staging_benchmark() {
    use crate::ark_adapter::Scalar;
    use crate::traits::Field;
    use std::{hint::black_box, time::Instant};

    let calls = 1024;
    let mut b = CircuitBuilder::<Scalar>::new();
    b.enable_compact_blake3();
    let bytes = ByteGadgets::new(&mut b);
    let inputs: Vec<_> = (0..calls)
        .map(|call| {
            let input: Vec<_> = (0..64)
                .map(|i| bytes.input(&mut b, &format!("{call}/{i}")))
                .collect();
            for byte in blake3(&mut b, &bytes, &input) {
                b.expose_public(byte.value());
            }
            input
        })
        .collect();
    let compiled = b
        .finish()
        .lower_to_multi_stark(Scalar::from_u8(95))
        .unwrap();
    println!("staging_fixture calls={calls} message_bytes=64 field_bytes=32");
    for iteration in 0..4 {
        let mut witness = compiled.witness();
        for (call, input) in inputs.iter().enumerate() {
            for (offset, byte) in input.iter().enumerate() {
                let value = u8::try_from((call + offset + iteration) % 256).unwrap();
                witness.set(byte.value(), Scalar::from_u8(value)).unwrap();
            }
        }
        let assignment = witness.generate().unwrap();
        let start = Instant::now();
        let shards = compiled.trace_shards(&assignment).unwrap();
        let validation_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut trace_ms = Vec::new();
        let mut digest = ::blake3::Hasher::new();
        let mut cells = 0;
        for index in 0..compiled.num_circuits() {
            let start = Instant::now();
            let trace = shards.trace(index).unwrap();
            trace_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            cells += trace.values.len();
            bincode::serde::encode_into_std_write(
                &(trace.width, &trace.values),
                &mut digest,
                bincode::config::standard(),
            )
            .unwrap();
            black_box(&trace);
        }
        let hash_ms: f64 = trace_ms[trace_ms.len() - 9..].iter().sum();
        let total_ms = validation_ms + trace_ms.iter().sum::<f64>();
        println!(
            "staging_sample iteration={iteration} validation_ms={validation_ms:.6} hash_ms={hash_ms:.6} total_ms={total_ms:.6} cells={cells} digest={} trace_ms={trace_ms:?}",
            digest.finalize().to_hex()
        );
    }
}
