use std::collections::HashMap;

use p3_field::PrimeCharacteristicRing;
use p3_matrix::{Matrix, dense::RowMajorMatrix};

use super::*;
use crate::eval::{VarValues, eval_expr};
use crate::system::{CircuitInputs, System, SystemWitness};
use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};

mod compact_hash;
mod hardening;
mod partitioned;

fn config() -> GoldilocksBlake3Config {
    GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 20,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    )
}

/// Independently check the emitted AIR and exact lookup multiset, bypassing
/// every frontend witness check. Used to attack the physical trace directly.
fn relations_hold<F: crate::traits::Field>(
    inputs: &[CircuitInputs<F>],
    traces: &[RowMajorMatrix<F>],
    claims: &[Vec<F>],
) -> bool {
    let mut balance: HashMap<Vec<F>, F> = HashMap::new();
    for claim in claims {
        *balance.entry(claim.clone()).or_insert(F::ZERO) += F::ONE;
    }
    for (input, trace) in inputs.iter().zip(traces) {
        let prep = input.preprocessed.as_ref().unwrap();
        if trace.height() != 0 && trace.height() != prep.height() {
            return false;
        }
        for row in 0..trace.height() {
            let main = trace.row_slice(row).unwrap();
            let fixed = prep.row_slice(row).unwrap();
            let view = VarValues {
                main: [&main, &main],
                preprocessed: [&fixed, &fixed],
                stage2: [&[], &[]],
                publics: &[],
                is_first_row: F::from_bool(row == 0),
                is_last_row: F::from_bool(row + 1 == trace.height()),
                is_transition: F::from_bool(row + 1 < trace.height()),
            };
            if input
                .constraints
                .iter()
                .any(|e| eval_expr(e, &view) != F::ZERO)
            {
                return false;
            }
            for lookup in &input.lookups {
                let args = lookup.args.iter().map(|e| eval_expr(e, &view)).collect();
                *balance.entry(args).or_insert(F::ZERO) += eval_expr(&lookup.multiplicity, &view);
            }
        }
    }
    balance.values().all(|&v| v == F::ZERO)
}

#[test]
fn plonkish_proves_arithmetic_selection_hints_and_wide_lookups() {
    let f = Val::from_u32;
    let mut builder = CircuitBuilder::<Val>::new();
    let a = builder.public_input("a");
    let b = builder.input("b");
    let choose = builder.input("choose sum");
    let bit = builder.assert_bool(choose);
    let sum = builder.add(a, b);
    let product = builder.mul(a, b);
    let result = builder.select(bit, sum, product);
    builder.expose_public(result);
    let inverse = builder.inverse(b);
    let identity = builder.mul(b, inverse);
    builder.expose_public(identity);
    let difference = builder.sub(a, b);
    let zero = builder.is_zero(difference);
    builder.expose_public(zero.value());
    let table = builder.fixed_table(
        "arithmetic tuples",
        vec![
            vec![f(1), f(2), f(2), f(3)],
            vec![f(2), f(3), f(6), f(5)],
            vec![f(3), f(4), f(12), f(7)],
        ],
    );
    builder.lookup(table, &[a, b, product, sum]);
    // Repeated queries must get counted, not deduplicated.
    builder.lookup(table, &[a, b, product, sum]);
    let compiled = builder.finish().lower_to_multi_stark(f(17)).unwrap();
    assert_eq!(compiled.main_width(), 4);
    let mut witness = compiled.witness();
    witness.set(a, f(2)).unwrap();
    witness.set(b, f(3)).unwrap();
    witness.set(choose, Val::ONE).unwrap();
    let assignment = witness.generate().unwrap();
    let expected = [f(2), f(5), Val::ONE, Val::ZERO];
    assert_eq!(assignment.public_values(), expected);
    let traces = compiled.traces(&assignment).unwrap();
    assert_eq!(
        traces[1].values,
        vec![Val::ZERO, f(2), Val::ZERO, Val::ZERO]
    );
    let claims = compiled.claims(&expected).unwrap();
    let inputs = compiled.circuit_inputs();
    assert!(relations_hold(&inputs, &traces, &claims));
    let (system, key) = System::new(config(), inputs);
    let witness = SystemWitness::from_stage_1(traces, &system);
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let proof = system.prove_multiple_claims(&key, &refs, witness);
    system.verify_multiple_claims(&refs, &proof).unwrap();
    // The computation and table both have fixed, independently pinned
    // heights. Reject a different claimed domain before invoking the PCS.
    for position in 0..proof.log_degrees.len() {
        let mut resized = proof.clone();
        resized.log_degrees[position] -= 1;
        assert!(matches!(
            system.verify_multiple_claims(&refs, &resized),
            Err(crate::verifier::VerificationError::InvalidProofShape)
        ));
    }
    let wrong = compiled.claims(&[f(2), f(6), Val::ONE, Val::ZERO]).unwrap();
    assert!(
        system
            .verify_multiple_claims(&wrong.iter().map(Vec::as_slice).collect::<Vec<_>>(), &proof)
            .is_err()
    );
}

#[test]
fn plonkish_rejects_broken_copy_even_when_local_arithmetic_is_valid() {
    let f = Val::from_u32;
    let mut builder = CircuitBuilder::new();
    let x = builder.input("x");
    let double = builder.add(x, x);
    builder.expose_public(double);
    let compiled = builder.finish().lower_to_multi_stark(f(23)).unwrap();
    let mut witness = compiled.witness();
    witness.set(x, f(5)).unwrap();
    let assignment = witness.generate().unwrap();
    let mut traces = compiled.traces(&assignment).unwrap();
    // Gate row 1 says 5 + 7 = 12; both input cells represent the same x.
    // Also change the public output to 12, so only copy consistency rejects.
    traces[0].values[3..6].copy_from_slice(&[f(5), f(7), f(12)]);
    traces[0].values[9] = f(12);
    let claims = compiled.claims(&[f(12)]).unwrap();
    let inputs = compiled.circuit_inputs();
    assert!(!relations_hold(&inputs, &traces, &claims));
    let (system, key) = System::new(config(), inputs);
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let proof =
        system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(traces, &system));
    assert!(system.verify_multiple_claims(&refs, &proof).is_err());
}

#[test]
fn plonkish_anchor_prevents_deactivating_a_computation_without_publics() {
    let mut builder = CircuitBuilder::<Val>::new();
    let x = builder.input("must be zero");
    builder.assert_zero(x);
    builder.fixed_table("unused", vec![vec![Val::ONE]]);
    let compiled = builder
        .finish()
        .lower_to_multi_stark(Val::from_u32(24))
        .unwrap();
    let claims = compiled.claims(&[]).unwrap();
    assert_eq!(claims.len(), 1);
    // Malicious prover leaves only the unused fixed-table circuit active.
    let traces = vec![
        RowMajorMatrix::new(vec![], 3),
        RowMajorMatrix::new_col(vec![Val::ZERO; 2]),
    ];
    let inputs = compiled.circuit_inputs();
    assert!(!relations_hold(&inputs, &traces, &claims));
    let (system, key) = System::new(config(), inputs);
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let proof =
        system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(traces, &system));
    assert!(system.verify_multiple_claims(&refs, &proof).is_err());
}

#[test]
fn plonkish_every_physical_cell_is_bound_including_padding() {
    let mut builder = CircuitBuilder::<Val>::new();
    let x = builder.input("x");
    let square = builder.mul(x, x);
    builder.expose_public(square);
    builder.expose_public(x); // makes five used rows, padded to eight
    let table = builder.fixed_table(
        "odd",
        vec![
            vec![Val::ONE],
            vec![Val::from_u32(3)],
            vec![Val::from_u32(5)],
        ],
    );
    builder.lookup(table, &[x]);
    let compiled = builder
        .finish()
        .lower_to_multi_stark(Val::from_u32(9))
        .unwrap();
    let mut witness = compiled.witness();
    witness.set(x, Val::from_u32(3)).unwrap();
    let assignment = witness.generate().unwrap();
    let traces = compiled.traces(&assignment).unwrap();
    let claims = compiled
        .claims(&[Val::from_u32(9), Val::from_u32(3)])
        .unwrap();
    let inputs = compiled.circuit_inputs();
    assert!(relations_hold(&inputs, &traces, &claims));
    // Padding repeats a real row, and does not accidentally admit zero.
    assert_eq!(inputs[1].preprocessed.as_ref().unwrap().values[9], Val::ONE);
    for circuit in 0..traces.len() {
        for cell in 0..traces[circuit].values.len() {
            let mut changed = traces.clone();
            changed[circuit].values[cell] += Val::ONE;
            assert!(
                !relations_hold(&inputs, &changed, &claims),
                "unbound cell {circuit}:{cell}"
            );
        }
    }
}

#[test]
fn plonkish_fixed_tables_and_namespaces_do_not_cross_match() {
    let mut builder = CircuitBuilder::<Val>::new();
    let x = builder.input("x");
    let first = builder.fixed_table("one", vec![vec![Val::ONE]]);
    builder.fixed_table("two", vec![vec![Val::TWO]]);
    builder.lookup(first, &[x]);
    let compiled = builder
        .finish()
        .lower_to_multi_stark(Val::from_u32(10))
        .unwrap();
    let mut witness = compiled.witness();
    witness.set(x, Val::ONE).unwrap();
    let assignment = witness.generate().unwrap();
    let mut traces = compiled.traces(&assignment).unwrap();
    let claims = compiled.claims(&[]).unwrap();
    let inputs = compiled.circuit_inputs();
    assert!(relations_hold(&inputs, &traces, &claims));
    // Change the query to two everywhere x occurs, and move its multiplicity
    // into the wrong table. Arithmetic and copies still hold.
    for cell in &mut traces[0].values {
        if *cell == Val::ONE {
            *cell = Val::TWO;
        }
    }
    traces[1].values[0] = Val::ZERO;
    traces[2].values[0] = Val::ONE;
    assert!(!relations_hold(&inputs, &traces, &claims));
    let mut wrong_namespace = claims;
    wrong_namespace[0][0] += Val::ONE;
    assert!(!relations_hold(
        &inputs,
        &compiled.traces(&assignment).unwrap(),
        &wrong_namespace
    ));
}

#[test]
fn plonkish_same_layout_supports_multiple_witnesses_and_zero_tests() {
    let mut builder = CircuitBuilder::<Val>::new();
    let x = builder.input("x");
    let is_zero = builder.is_zero(x);
    let one = builder.constant(Val::ONE);
    let result = builder.select(is_zero, one, x);
    builder.expose_public(result);
    let compiled = builder
        .finish()
        .lower_to_multi_stark(Val::from_u32(11))
        .unwrap();
    let inputs = compiled.circuit_inputs();
    for value in [Val::ZERO, Val::from_u32(7)] {
        let mut witness = compiled.witness();
        witness.set(x, value).unwrap();
        let assignment = witness.generate().unwrap();
        let expected = if value == Val::ZERO { Val::ONE } else { value };
        assert_eq!(assignment.public_values(), &[expected]);
        let traces = compiled.traces(&assignment).unwrap();
        assert_eq!(traces[0].height(), compiled.main_height());
        assert!(relations_hold(
            &inputs,
            &traces,
            &compiled.claims(&[expected]).unwrap()
        ));
    }
}

#[test]
fn plonkish_witness_errors_and_invalid_hints() {
    let mut builder = CircuitBuilder::<Val>::new();
    let x = builder.input("x");
    let wrong_inverse = builder.hint("bad inverse", &[x], |_| Ok(Val::ZERO));
    let product = builder.mul(x, wrong_inverse);
    let one = builder.constant(Val::ONE);
    builder.assert_equal(product, one);
    let circuit = builder.finish();
    assert!(matches!(
        circuit.witness().generate(),
        Err(WitnessError::MissingInput { .. })
    ));
    let mut witness = circuit.witness();
    witness.set(x, Val::TWO).unwrap();
    assert_eq!(witness.set(x, Val::ONE), Err(WitnessError::AlreadyAssigned));
    assert_eq!(
        witness.set(product, Val::ONE),
        Err(WitnessError::NotAnInput)
    );
    assert!(matches!(
        witness.generate(),
        Err(WitnessError::UnsatisfiedGate { .. })
    ));
    let mut other = CircuitBuilder::<Val>::new();
    let foreign = other.input("foreign");
    assert_eq!(
        circuit.witness().set(foreign, Val::ONE),
        Err(WitnessError::ForeignValue)
    );

    let mut builder = CircuitBuilder::<Val>::new();
    let x = builder.input("zero");
    builder.inverse(x);
    let circuit = builder.finish();
    let mut witness = circuit.witness();
    witness.set(x, Val::ZERO).unwrap();
    assert!(matches!(
        witness.generate(),
        Err(WitnessError::HintFailed { .. })
    ));
}

#[test]
fn plonkish_invalid_boolean_and_lookup_are_rejected() {
    let mut builder = CircuitBuilder::<Val>::new();
    let x = builder.input("bit");
    builder.assert_bool(x);
    let circuit = builder.finish();
    let mut witness = circuit.witness();
    witness.set(x, Val::TWO).unwrap();
    assert!(matches!(
        witness.generate(),
        Err(WitnessError::UnsatisfiedGate { .. })
    ));
    let mut builder = CircuitBuilder::<Val>::new();
    let x = builder.input("byte");
    let table = builder.fixed_table(
        "nonzero",
        vec![vec![Val::ONE], vec![Val::TWO], vec![Val::from_u32(3)]],
    );
    builder.lookup(table, &[x]);
    let circuit = builder.finish();
    let mut witness = circuit.witness();
    witness.set(x, Val::ZERO).unwrap();
    assert!(matches!(
        witness.generate(),
        Err(WitnessError::LookupMissing { .. })
    ));
}

#[test]
#[should_panic(expected = "value belongs to another circuit")]
fn plonkish_foreign_handles_are_rejected_during_construction() {
    let mut a = CircuitBuilder::<Val>::new();
    let mut b = CircuitBuilder::<Val>::new();
    let x = a.input("x");
    let y = b.input("y");
    a.add(x, y);
}

#[test]
fn plonkish_public_count_and_foreign_assignment_are_checked() {
    let mut a = CircuitBuilder::<Val>::new();
    let one = a.constant(Val::ONE);
    a.expose_public(one);
    let a = a.finish().lower_to_multi_stark(Val::from_u32(12)).unwrap();
    assert!(matches!(
        a.claims(&[]),
        Err(WitnessError::PublicCount { .. })
    ));
    let b = CircuitBuilder::<Val>::new().finish();
    let assignment = b.witness().generate().unwrap();
    assert!(matches!(
        a.traces(&assignment),
        Err(WitnessError::ForeignAssignment)
    ));
}
