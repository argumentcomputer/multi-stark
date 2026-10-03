#[path = "../examples/support/compact_blake3.rs"]
mod compact;
use compact::{Compact, check};
use multi_stark::{
    system::{System, SystemWitness},
    types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val},
};
use p3_blake3::Blake3;
use p3_field::PrimeCharacteristicRing;
use p3_symmetric::CryptographicHasher;

#[test]
fn custom_gate_hashes_match_native_and_reject_corruption() {
    let c = Compact::new(2);
    let defs = c.definitions();
    for seed in [0u8, 1, 197, 255] {
        let messages = [
            std::array::from_fn(|j| u8::try_from(j).unwrap().wrapping_mul(31).wrapping_add(seed)),
            [seed; 64],
        ];
        let digests = messages.map(|m| Blake3.hash_iter(m));
        let claims = c.claims(&messages, &digests);
        let traces = c.traces(&messages, &defs);
        assert!(check(&defs, &traces, &claims), "seed {seed}");
        for (t, col) in [(0, 24), (0, 28), (1, 20), (3, 32), (5, 0), (6, 0)] {
            let mut bad = traces.clone();
            bad[t].values[col] += Val::ONE;
            assert!(
                !check(&defs, &bad, &claims),
                "seed {seed}, trace {t}, column {col}"
            );
        }
        for t in 0..6 {
            let mut bad = traces.clone();
            bad[t].values.clear();
            assert!(!check(&defs, &bad, &claims));
        }
        let mut wrong = claims.clone();
        *wrong.last_mut().unwrap().last_mut().unwrap() += Val::ONE;
        assert!(!check(&defs, &traces, &wrong));
    }
}
#[test]
fn custom_gate_outer_proof_binds_cross_row_copies() {
    let c = Compact::new(1);
    let defs = c.definitions();
    let messages = [std::array::from_fn(|j| u8::try_from(j * 3).unwrap())];
    let digests = messages.map(|m| Blake3.hash_iter(m));
    let claims = c.claims(&messages, &digests);
    let traces = c.traces(&messages, &defs);
    assert!(check(&defs, &traces, &claims));
    let cfg = GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 2,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 16,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    );
    let (system, key) = System::new(cfg, defs);
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let proof = system.prove_multiple_claims(
        &key,
        &refs,
        SystemWitness::from_stage_1(traces.clone(), &system),
    );
    system.verify_multiple_claims(&refs, &proof).unwrap();
    let mut bad = traces;
    let width = bad[0].width;
    // Swap complete same-kind gate rows: local arithmetic and all table
    // query multiplicities are unchanged. Only fixed copy wiring disagrees.
    assert_ne!(&bad[0].values[..width], &bad[0].values[width..2 * width]);
    for i in 0..width {
        bad[0].values.swap(i, width + i);
    }
    // Check that this attack really leaves each local gate equation valid.
    use multi_stark::eval::{VarValues, eval_expr};
    use p3_matrix::Matrix;
    let definitions = c.definitions();
    for (def, trace) in definitions.iter().zip(&bad) {
        for r in 0..trace.height() {
            let main = trace.row_slice(r).unwrap();
            let prep = def.preprocessed.as_ref().unwrap().row_slice(r).unwrap();
            let view = VarValues {
                main: [&main, &main],
                preprocessed: [&prep, &prep],
                stage2: [&[], &[]],
                publics: &[],
                is_first_row: Val::from_bool(r == 0),
                is_last_row: Val::from_bool(r + 1 == trace.height()),
                is_transition: Val::from_bool(r + 1 < trace.height()),
            };
            assert!(
                def.constraints
                    .iter()
                    .all(|e| eval_expr(e, &view) == Val::ZERO)
            );
        }
    }
    assert!(!check(&c.definitions(), &bad, &claims));
    let proof =
        system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(bad, &system));
    assert!(system.verify_multiple_claims(&refs, &proof).is_err());
}
