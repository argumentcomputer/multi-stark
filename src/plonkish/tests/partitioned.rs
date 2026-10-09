use super::*;

#[test]
fn partitioned_proof_preserves_cross_trace_copies() {
    let f = Val::from_u32;
    let mut b = CircuitBuilder::new();
    let x = b.input("x");
    let double = b.add(x, x);
    b.mul(x, x); // unused result, but the input must agree with the other trace
    b.expose_public(double);
    let c = b.finish();
    let layout = c.multi_stark_layout_with_max_height(2).unwrap();
    let compiled = c.lower_to_multi_stark_with_max_height(f(81), 2).unwrap();
    assert_eq!(layout.main_heights, compiled.main_heights());
    assert_eq!(layout.main_height, compiled.main_height());
    assert_eq!(compiled.main_heights(), [2, 2, 2]);
    let mut w = compiled.witness();
    w.set(x, f(5)).unwrap();
    let a = w.generate().unwrap();
    let traces = compiled.traces(&a).unwrap();
    let claims = compiled.claims(&[f(10)]).unwrap();
    let defs = compiled.circuit_inputs();
    assert!(relations_hold(&defs, &traces, &claims));
    let (system, key) = System::new(config(), defs);
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let proof = system.prove_multiple_claims(
        &key,
        &refs,
        SystemWitness::from_stage_1(traces.clone(), &system),
    );
    system.verify_multiple_claims(&refs, &proof).unwrap();
    let wrong = compiled.claims(&[f(11)]).unwrap();
    assert!(
        system
            .verify_multiple_claims(&wrong.iter().map(Vec::as_slice).collect::<Vec<_>>(), &proof)
            .is_err()
    );

    let mut bad = traces;
    assert_eq!(&bad[1].values[..3], &[f(5), f(5), f(25)]);
    // 7*7=49 is locally valid and does not affect the public double=10.
    // Only the global copy cycle can reject this disagreement on x.
    bad[1].values[..3].copy_from_slice(&[f(7), f(7), f(49)]);
    let defs = compiled.circuit_inputs();
    for (def, trace) in defs.iter().zip(&bad) {
        for row in 0..trace.height() {
            let main = trace.row_slice(row).unwrap();
            let prep = def.preprocessed.as_ref().unwrap().row_slice(row).unwrap();
            let view = VarValues {
                main: [&main, &main],
                preprocessed: [&prep, &prep],
                stage2: [&[], &[]],
                publics: &[],
                is_first_row: f(u32::from(row == 0)),
                is_last_row: f(u32::from(row + 1 == trace.height())),
                is_transition: f(u32::from(row + 1 < trace.height())),
            };
            assert!(
                def.constraints
                    .iter()
                    .all(|e| eval_expr(e, &view) == Val::ZERO)
            );
        }
    }
    assert!(!relations_hold(&defs, &bad, &claims));
    let proof =
        system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(bad, &system));
    assert!(system.verify_multiple_claims(&refs, &proof).is_err());
}

#[test]
fn every_partition_is_required_even_without_user_publics() {
    let mut b = CircuitBuilder::<Val>::new();
    for n in 0..5 {
        let x = b.input(format!("zero {n}"));
        b.assert_zero(x);
    }
    let compiled = b
        .finish()
        .lower_to_multi_stark_with_max_height(Val::from_u8(82), 2)
        .unwrap();
    let claims = compiled.claims(&[]).unwrap();
    let defs = compiled.circuit_inputs();
    let traces: Vec<_> = compiled
        .main_heights()
        .iter()
        .map(|&height| RowMajorMatrix::new(vec![Val::ZERO; 3 * height], 3))
        .collect();
    assert!(relations_hold(&defs, &traces, &claims));
    let (system, key) = System::new(config(), defs);
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    for omitted in 0..traces.len() {
        let mut bad = traces.clone();
        bad[omitted].values.clear();
        assert!(!relations_hold(&compiled.circuit_inputs(), &bad, &claims));
        let proof =
            system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(bad, &system));
        assert!(
            system.verify_multiple_claims(&refs, &proof).is_err(),
            "omitted partition {omitted}"
        );
    }
}

#[test]
fn partitioned_traces_share_table_multiplicities() {
    let f = Val::from_u32;
    let mut b = CircuitBuilder::new();
    let x = b.input("x");
    let table = b.fixed_table("five or seven", vec![vec![f(5)], vec![f(7)]]);
    for _ in 0..5 {
        b.lookup(table, &[x]);
    }
    let c = b.finish();
    let unpartitioned = c.multi_stark_layout().unwrap();
    let layout = c.multi_stark_layout_with_max_height(4).unwrap();
    assert_eq!(layout.main_heights, [4, 4]);
    assert_eq!(
        layout.trace_field_bytes,
        unpartitioned.trace_field_bytes + layout.main_height * size_of::<Val>()
    );
    let compiled = c.lower_to_multi_stark_with_max_height(f(83), 4).unwrap();
    let mut w = compiled.witness();
    w.set(x, f(5)).unwrap();
    let a = w.generate().unwrap();
    let traces = compiled.traces(&a).unwrap();
    assert_eq!(traces.last().unwrap().values, [f(5), Val::ZERO]);
    let claims = compiled.claims(&[]).unwrap();
    let defs = compiled.circuit_inputs();
    assert!(relations_hold(&defs, &traces, &claims));
    let (system, key) = System::new(config(), defs);
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let proof =
        system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(traces, &system));
    system.verify_multiple_claims(&refs, &proof).unwrap();
}

#[test]
fn oversized_tables_are_rejected_explicitly() {
    let mut b = CircuitBuilder::<Val>::new();
    b.fixed_table("three", (0..3).map(|i| vec![Val::from_u8(i)]).collect());
    assert_eq!(
        b.finish().multi_stark_layout_with_max_height(2),
        Err(LoweringError::TableExceedsTraceHeight)
    );
}

#[test]
fn partitioned_mixed_heights_prove_the_same_statement() {
    for cap in [4, 8, 16] {
        let mut b = CircuitBuilder::<Val>::new();
        let x = b.input("x");
        let mut output = x;
        for _ in 0..6 {
            output = b.add(output, output);
        }
        b.expose_public(output);
        let compiled = b
            .finish()
            .lower_to_multi_stark_with_max_height(Val::from_u8(84), cap)
            .unwrap();
        if cap == 4 {
            assert_eq!(compiled.main_heights(), [4, 4, 2]);
        }
        if cap == 8 {
            assert_eq!(compiled.main_heights(), [8, 2]);
        }
        if cap == 16 {
            assert_eq!(compiled.main_heights(), [16]);
        }
        let mut w = compiled.witness();
        w.set(x, Val::from_u8(3)).unwrap();
        let a = w.generate().unwrap();
        let traces = compiled.traces(&a).unwrap();
        let claims = compiled.claims(&[Val::from_u8(192)]).unwrap();
        let defs = compiled.circuit_inputs();
        assert!(relations_hold(&defs, &traces, &claims));
        let (system, key) = System::new(config(), defs);
        let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
        let proof =
            system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(traces, &system));
        system.verify_multiple_claims(&refs, &proof).unwrap();
    }
}
