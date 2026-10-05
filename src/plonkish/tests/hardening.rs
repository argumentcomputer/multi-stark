use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use super::*;

#[test]
fn arithmetic_folds_constants_and_has_predictable_costs() {
    let mut b = CircuitBuilder::<Val>::new();
    let x = b.input("x");
    let y = b.input("y");
    let z = b.input("z");
    let zero = b.constant(Val::ZERO);
    let one = b.constant(Val::ONE);
    let seven = b.constant(Val::from_u32(7));
    let bit = b.assert_bool(one);
    let (_, identities) = b.measure(|b| {
        assert_eq!(b.add(x, zero), x);
        assert_eq!(b.sub(x, zero), x);
        assert_eq!(b.sub(x, x), zero);
        assert_eq!(b.mul(x, one), x);
        assert_eq!(b.mul(zero, x), zero);
        assert_eq!(b.scale(x, Val::ZERO), zero);
        assert_eq!(b.scale(x, Val::ONE), x);
        assert_eq!(b.select(bit, x, y), x);
        assert_eq!(b.select(bit, x, x), x);
        b.assert_equal(x, x);
    });
    assert_eq!(identities, CircuitStats::default());
    let (affine, cost) =
        b.measure(|b| b.affine([x, y], [Val::TWO, -Val::from_u32(3)], Val::from_u32(5)));
    assert_eq!(
        (cost.gates, cost.values, cost.hint_calls, cost.publics),
        (1, 1, 0, 0)
    );
    let (constant_factor, cost) = b.measure(|b| b.mul_add(x, seven, y));
    assert_eq!(cost.gates, 1);
    let (constant_addend, cost) = b.measure(|b| b.mul_add(x, y, seven));
    assert_eq!(cost.gates, 1);
    let (alias_addend, cost) = b.measure(|b| b.mul_add(x, y, x));
    assert_eq!(cost.gates, 1);
    let (alias_right, cost) = b.measure(|b| b.mul_add(x, y, y));
    assert_eq!(cost.gates, 1);
    let (variable, cost) = b.measure(|b| b.mul_add(x, y, z));
    assert_eq!(cost.gates, 2);
    let (constant, cost) = b.measure(|b| b.mul(seven, seven));
    assert_eq!((cost.gates, cost.values), (1, 1));
    let (_, reuse) = b.measure(|b| assert_eq!(b.constant(Val::from_u32(49)), constant));
    assert_eq!(reuse, CircuitStats::default());
    let values = [
        affine,
        constant_factor,
        constant_addend,
        alias_addend,
        alias_right,
        variable,
        constant,
    ];
    for value in values {
        b.expose_public(value);
    }
    let circuit = b.finish();
    for (a, c, d) in [(0, 0, 0), (1, 2, 3), (7, 11, 19), (u32::MAX, 31, 4)] {
        let [a, c, d] = [a, c, d].map(Val::from_u32);
        let mut witness = circuit.witness();
        for (wire, value) in [(x, a), (y, c), (z, d)] {
            witness.set(wire, value).unwrap();
        }
        let assignment = witness.generate().unwrap();
        assert_eq!(
            assignment.public_values(),
            [
                a * Val::TWO - c * Val::from_u32(3) + Val::from_u32(5),
                a * Val::from_u32(7) + c,
                a * c + Val::from_u32(7),
                a * c + a,
                a * c + c,
                a * c + d,
                Val::from_u32(49),
            ]
        );
    }
}

#[test]
fn linear_combinations_normalize_terms_without_public_side_effects() {
    let mut b = CircuitBuilder::<Val>::new();
    let x = b.input("x");
    let y = b.input("y");
    let c = b.constant(Val::from_u32(7));
    let (sum, cost) = b.measure(|b| {
        b.linear_combination(
            &[
                (Val::from_u32(3), x),
                (Val::from_u32(5), y),
                (-Val::from_u32(3), x),
                (Val::ZERO, x),
                (Val::TWO, c),
            ],
            Val::ONE,
        )
    });
    assert_eq!((cost.gates, cost.values, cost.publics), (1, 1, 0));
    let (same, cost) =
        b.measure(|b| b.linear_combination(&[(Val::TWO, x), (Val::NEG_ONE, x)], Val::ZERO));
    assert_eq!(same, x);
    assert_eq!(cost, CircuitStats::default());
    let empty = b.linear_combination(&[], Val::from_u32(9));
    assert_eq!(empty, b.constant(Val::from_u32(9)));
    b.expose_public(sum);
    let compiled = b.finish().lower_to_multi_stark(Val::from_u32(201)).unwrap();
    let mut witness = compiled.witness();
    witness.set(x, Val::from_u32(999)).unwrap();
    witness.set(y, Val::from_u32(4)).unwrap();
    let assignment = witness.generate().unwrap();
    assert_eq!(assignment.public_values(), [Val::from_u32(35)]);
    let traces = compiled.traces(&assignment).unwrap();
    assert!(relations_hold(
        &compiled.circuit_inputs(),
        &traces,
        &compiled.claims(&[Val::from_u32(35)]).unwrap()
    ));
}

#[test]
fn simplifications_never_hide_foreign_handles() {
    for attack in 0..10 {
        let mut foreign = CircuitBuilder::<Val>::new();
        let other = foreign.input("foreign");
        let other_bit = foreign.assert_bool(other);
        let mut b = CircuitBuilder::<Val>::new();
        let x = b.input("x");
        let zero = b.constant(Val::ZERO);
        let one = b.constant(Val::ONE);
        let bit = b.assert_bool(one);
        assert!(
            catch_unwind(AssertUnwindSafe(|| match attack {
                0 => {
                    b.mul(zero, other);
                }
                1 => {
                    b.sub(other, other);
                }
                2 => {
                    b.affine([x, other], [Val::ONE, Val::ZERO], Val::ZERO);
                }
                3 => {
                    b.linear_combination(&[(Val::ZERO, other)], Val::ZERO);
                }
                4 => {
                    b.select(bit, x, other);
                }
                5 => {
                    b.select(other_bit, x, x);
                }
                6 => {
                    b.assert_equal(other, other);
                }
                7 => {
                    b.mul_add(zero, other, x);
                }
                8 => {
                    b.scale(other, Val::ZERO);
                }
                _ => {
                    b.hint_many("foreign dependency", &[other], |_| Ok([Val::ONE; 2]));
                }
            }))
            .is_err(),
            "accepted foreign handle in shortcut {attack}"
        );
    }
}

#[test]
fn batched_hints_execute_once_and_preserve_input_and_output_roles() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut b = CircuitBuilder::<Val>::new();
    let x = b.input("x");
    let counter = calls.clone();
    let ([plus, square], cost) = b.measure(|b| {
        b.hint_many("pair", &[x], move |args| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok([args[0] + Val::ONE, args[0] * args[0]])
        })
    });
    assert_eq!(
        (
            cost.values,
            cost.hint_calls,
            cost.hint_outputs,
            cost.gates,
            cost.publics
        ),
        (2, 1, 2, 0, 0)
    );
    let expected_plus = b.affine([x, x], [Val::ONE, Val::ZERO], Val::ONE);
    let expected_square = b.mul(x, x);
    b.assert_equal(plus, expected_plus);
    b.assert_equal(square, expected_square);
    // Input slots and logical value indices are deliberately far apart.
    let y = b.input("input after batch");
    let sum = b.hint("dependent", &[plus, square, y], |a| Ok(a[0] + a[1] + a[2]));
    let expected_sum = b.linear_combination(
        &[(Val::ONE, plus), (Val::ONE, square), (Val::ONE, y)],
        Val::ZERO,
    );
    b.assert_equal(sum, expected_sum);
    b.expose_public(sum);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let compiled = b.finish().lower_to_multi_stark(Val::from_u32(202)).unwrap();
    let inputs = compiled.circuit_inputs();
    let (system, key) = System::new(config(), compiled.circuit_inputs());
    for n in [3, 5] {
        let mut witness = compiled.witness();
        witness.set(x, Val::from_u32(n)).unwrap();
        witness.set(y, Val::from_u32(7)).unwrap();
        assert_eq!(
            witness.set(y, Val::ZERO),
            Err(WitnessError::AlreadyAssigned)
        );
        assert_eq!(
            witness.set(square, Val::ZERO),
            Err(WitnessError::NotAnInput)
        );
        let assignment = witness.generate().unwrap();
        let claims = compiled
            .claims(&[Val::from_u32(n + 1 + n * n + 7)])
            .unwrap();
        let traces = compiled.traces(&assignment).unwrap();
        assert!(relations_hold(&inputs, &traces, &claims));
        let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
        let proof = system.prove_multiple_claims(
            &key,
            &refs,
            SystemWitness::from_stage_1(traces.clone(), &system),
        );
        system.verify_multiple_claims(&refs, &proof).unwrap();
        // Bypass every recipe and corrupt individual logical values.
        for output in [plus, square, sum] {
            let mut values = assignment.values.clone();
            values[output.index()] += Val::ONE;
            assert!(compiled.circuit().check_values(&values).is_err());
        }
        // Independently attack every physical cell, including copied outputs
        // and padding, against the emitted AIR and lookup messages.
        for cell in 0..traces[0].values.len() {
            let mut bad = traces.clone();
            bad[0].values[cell] += Val::ONE;
            assert!(
                !relations_hold(&inputs, &bad, &claims),
                "unbound cell {cell}"
            );
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn hint_batches_are_not_implicit_constraints() {
    let mut b = CircuitBuilder::<Val>::new();
    let [a, c] = b.hint_many("unconstrained", &[], |_| Ok([Val::ONE, Val::TWO]));
    let circuit = b.finish();
    let assignment = circuit.witness().generate().unwrap();
    let mut changed = assignment.values;
    changed[a.index()] = Val::from_u32(99);
    changed[c.index()] = Val::from_u32(123);
    assert!(
        circuit.check_values(&changed).is_ok(),
        "recipes must never imply relations"
    );

    let mut b = CircuitBuilder::<Val>::new();
    b.hint_many::<2>("failure", &[], |_| Err("intentional".into()));
    assert!(
        matches!(b.finish().witness().generate(), Err(WitnessError::HintFailed { name, message })
        if name == "failure" && message == "intentional")
    );
    let mut b = CircuitBuilder::<Val>::new();
    assert!(
        catch_unwind(AssertUnwindSafe(
            || b.hint_many::<0>("empty", &[], |_| Ok([]))
        ))
        .is_err()
    );
}

#[test]
fn layout_inspection_matches_allocations_and_inclusive_measurements() {
    let mut b = CircuitBuilder::<Val>::new();
    let ((x, product, nested), outer) = b.measure(|b| {
        let x = b.input("x");
        let (product, nested) = b.measure(|b| b.mul(x, x));
        (x, product, nested)
    });
    assert_eq!((outer.inputs, outer.values, outer.gates), (1, 2, 1));
    assert_eq!((nested.inputs, nested.values, nested.gates), (0, 1, 1));
    let table = b.fixed_table("duplicates", vec![vec![Val::TWO; 4]; 3]);
    b.fixed_table("unused singleton", vec![vec![Val::ONE]]);
    b.lookup(table, &[x; 4]);
    b.expose_public(product);
    b.expose_public(product);
    let stats = b.stats();
    let circuit = b.finish();
    assert_eq!(stats, circuit.stats());
    let layout = circuit.multi_stark_layout().unwrap();
    assert_eq!(
        layout.used_rows,
        stats.gates + stats.lookups + stats.publics + 1
    );
    assert_eq!(layout.table_heights, [4, 2]);
    assert_eq!(layout.main_width, 4);
    let compiled = circuit.lower_to_multi_stark(Val::from_u32(203)).unwrap();
    assert_eq!(layout.main_height, compiled.main_height());
    let mut witness = compiled.witness();
    witness.set(x, Val::TWO).unwrap();
    let traces = compiled.traces(&witness.generate().unwrap()).unwrap();
    let inputs = compiled.circuit_inputs();
    let advice: usize = traces.iter().map(|m| m.values.len()).sum();
    let fixed: usize = inputs
        .iter()
        .map(|c| c.preprocessed.as_ref().unwrap().values.len())
        .sum();
    assert_eq!(
        (layout.advice_cells, layout.preprocessed_cells),
        (advice, fixed)
    );
    assert_eq!(
        layout.trace_field_bytes,
        (advice + fixed) * size_of::<Val>()
    );
    assert_eq!(
        traces[1].values,
        [Val::ONE, Val::ZERO, Val::ZERO, Val::ZERO]
    );
    let claims = compiled.claims(&[Val::from_u32(4); 2]).unwrap();
    assert!(relations_hold(&inputs, &traces, &claims));
    let mut wrong = claims;
    wrong[2][3] += Val::ONE;
    assert!(!relations_hold(&inputs, &traces, &wrong));
}

#[test]
fn invalid_constant_booleans_cannot_be_optimized_away() {
    let mut b = CircuitBuilder::<Val>::new();
    let two = b.constant(Val::TWO);
    let bit = b.assert_bool(two);
    assert_eq!(b.select(bit, two, two), two);
    assert!(matches!(
        b.finish().witness().generate(),
        Err(WitnessError::UnsatisfiedGate { .. })
    ));
}

#[test]
fn arithmetic_matches_native_fields_at_wrapping_boundaries() {
    fn check<F: crate::traits::Field>() {
        let scalars = [F::ZERO, F::ONE, F::NEG_ONE, F::TWO];
        for a in scalars {
            for c in scalars {
                for k in scalars {
                    let mut b = CircuitBuilder::<F>::new();
                    let x = b.input("x");
                    let y = b.input("y");
                    let z = b.input("z");
                    let affine = b.affine([x, y], [a, c], k);
                    let linear =
                        b.linear_combination(&[(a, x), (c, y), (F::ONE, x), (F::NEG_ONE, x)], k);
                    let product = b.mul_add(x, y, z);
                    let circuit = b.finish();
                    for xv in scalars {
                        for yv in scalars {
                            let mut witness = circuit.witness();
                            witness.set(x, xv).unwrap();
                            witness.set(y, yv).unwrap();
                            witness.set(z, k).unwrap();
                            let assignment = witness.generate().unwrap();
                            assert_eq!(assignment.value(affine).unwrap(), a * xv + c * yv + k);
                            assert_eq!(assignment.value(linear).unwrap(), a * xv + c * yv + k);
                            assert_eq!(assignment.value(product).unwrap(), xv * yv + k);
                        }
                    }
                }
            }
        }
    }
    check::<Val>();
    check::<p3_baby_bear::BabyBear>();
}

#[test]
fn measurements_reject_replacing_the_builder() {
    let mut b = CircuitBuilder::<Val>::new();
    assert!(
        catch_unwind(AssertUnwindSafe(|| b.measure(|b| {
            *b = CircuitBuilder::new();
        })))
        .is_err()
    );
}
