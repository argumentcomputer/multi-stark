use super::*;
use crate::{plonkish::CircuitBuilder, traits::Algebra};
use p3_baby_bear::BabyBear;

fn pure_blocks() -> (CircuitBuilder<BabyBear>, Vec<usize>) {
    let mut b = CircuitBuilder::<BabyBear>::new();
    let mut ends = Vec::new();
    let root = b.input("first");
    let constant = b.constant(BabyBear::from_u8(7));
    for (i, name) in ["first hint", "middle hint", "last hint"]
        .into_iter()
        .enumerate()
    {
        let input = if i == 0 { root } else { b.input(name) };
        let outputs = b.hint_many(name, &[input, root, constant], |args| {
            if args[0] == BabyBear::ZERO {
                Err("zero advice input".into())
            } else {
                Ok([args[0], args[1], args[2]])
            }
        });
        b.assert_equal(outputs[0], input);
        b.assert_equal(outputs[1], root);
        b.assert_equal(outputs[2], constant);
        let derived = b.add(outputs[0], constant);
        b.expose_public(derived);
        ends.push(b.stats().values);
    }
    (b, ends)
}

#[test]
fn independent_advice_preserves_original_recipe_error_precedence() {
    let (mut builder, ends) = pure_blocks();
    assert!(builder.certify_witness_chunks(ends));
    let circuit = builder.finish();
    for first in [None, Some(BabyBear::ZERO), Some(BabyBear::ONE)] {
        for middle in [None, Some(BabyBear::ZERO), Some(BabyBear::ONE)] {
            for last in [None, Some(BabyBear::ZERO), Some(BabyBear::ONE)] {
                let witness = || {
                    let mut witness = circuit.witness();
                    witness.inputs = vec![first, middle, last];
                    witness
                };
                let expected = witness().generate_with(arithmetic_value);
                for actual in [witness().generate(), witness().generate_scheduled()] {
                    match (&expected, actual) {
                        (Ok(expected), Ok(actual)) => {
                            assert_eq!(
                                actual.values(circuit.id()).unwrap(),
                                expected.values(circuit.id()).unwrap()
                            );
                            assert_eq!(actual.public_values(), expected.public_values());
                        }
                        (Err(expected), Err(actual)) => assert_eq!(&actual, expected),
                        _ => panic!("scheduled and serial error behavior differs"),
                    }
                }
            }
        }
    }
}

#[test]
fn independent_advice_certificate_rejects_incomplete_or_dependent_chunks() {
    let fixture = || {
        let (builder, ends) = pure_blocks();
        (builder.finish(), ends)
    };
    let (circuit, ends) = fixture();
    assert!(circuit.independent_witness_chunks(&ends));
    for invalid in [
        vec![],
        vec![circuit.num_values()],
        vec![0, circuit.num_values()],
        vec![ends[0], ends[0], circuit.num_values()],
        vec![ends[0], circuit.num_values() - 1],
        vec![circuit.hints[0].start + 1, circuit.num_values()],
    ] {
        assert!(!circuit.independent_witness_chunks(&invalid));
    }
    for invalid in 0..10 {
        let (mut circuit, ends) = fixture();
        let hint = circuit.hints[0].start;
        let arithmetic = circuit
            .recipes
            .iter()
            .position(|recipe| matches!(recipe, Recipe::Arithmetic(_)))
            .unwrap();
        let Recipe::Arithmetic(gate) = circuit.recipes[arithmetic] else {
            unreachable!()
        };
        match invalid {
            0 => circuit.hints[0].len = 0,
            1 => circuit.hints[0].len = usize::MAX,
            2 => circuit.hints[0].start += 1,
            3 => circuit.hints[0].dependencies[0] = hint,
            4 => circuit.recipes[hint + 1] = Recipe::Constant(BabyBear::ZERO),
            5 => circuit.hints[1].dependencies[0] = hint,
            6 => circuit.gates[gate].wires[0].owner = u64::MAX,
            7 => circuit.gates[gate].wires[2].index += 1,
            8 => circuit.gates[gate].coefficients[3] = BabyBear::ZERO,
            9 => circuit.gates[gate].wires[0].index = arithmetic,
            _ => unreachable!(),
        }
        assert!(
            !circuit.independent_witness_chunks(&ends),
            "invalid case {invalid}"
        );
    }
}

#[test]
fn independent_advice_keeps_opaque_and_appended_plans_serial() {
    use std::sync::{Arc, Mutex};
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut b = CircuitBuilder::<BabyBear>::new();
    for i in 0..3 {
        let calls = calls.clone();
        b.hint("ordered callback", &[], move |_| {
            calls.lock().unwrap().push(i);
            Ok(BabyBear::ZERO)
        });
    }
    let circuit = b.finish();
    assert!(circuit.witness_chunks.is_empty());
    circuit.witness().generate().unwrap();
    assert_eq!(*calls.lock().unwrap(), [0, 1, 2]);

    let (mut b, ends) = pure_blocks();
    assert!(b.certify_witness_chunks(ends));
    let appended = b.hint("appended advice", &[], |_| Ok(BabyBear::from_u8(23)));
    let constant = b.constant(BabyBear::from_u8(23));
    b.assert_equal(appended, constant);
    b.expose_public(appended);
    let circuit = b.finish();
    assert_ne!(circuit.witness_chunks.last(), Some(&circuit.num_values()));
    let witness = || {
        let mut witness = circuit.witness();
        witness.inputs.fill(Some(BabyBear::ONE));
        witness
    };
    let expected = witness().generate_with(arithmetic_value).unwrap();
    let actual = witness().generate().unwrap();
    assert_eq!(
        actual.values(circuit.id()).unwrap(),
        expected.values(circuit.id()).unwrap()
    );
    assert_eq!(actual.public_values().last(), Some(&BabyBear::from_u8(23)));
}
