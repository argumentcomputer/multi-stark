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
        let traces = compiled.traces(&a).unwrap();
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
