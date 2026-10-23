use super::*;
use crate::{plonkish::CircuitBuilder, traits::Algebra};
use p3_baby_bear::BabyBear;
use std::sync::{Arc, Mutex, atomic::AtomicUsize};

fn blocks() -> (CircuitBuilder<BabyBear>, [Value; 2], [Value; 2], Vec<usize>) {
    let mut b = CircuitBuilder::<BabyBear>::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut ends = Vec::new();
    for name in ["first", "second"] {
        let input = b.input(name);
        let pair = b.hint_many_pure::<2>(name, &[input], |v| {
            if v[0] == BabyBear::NEG_ONE {
                return Err("negative sentinel".into());
            }
            Ok([v[0] + BabyBear::ONE, v[0] * v[0]])
        });
        let one = b.constant(BabyBear::ONE);
        let incremented = b.add(input, one);
        let squared = b.mul(input, input);
        b.assert_equal(pair[0], incremented);
        b.assert_equal(pair[1], squared);
        let output = b.add(pair[0], pair[1]);
        inputs.push(input);
        outputs.push(output);
        ends.push(b.stats().values);
    }
    (
        b,
        inputs.try_into().unwrap(),
        outputs.try_into().unwrap(),
        ends,
    )
}

#[test]
fn ordinary_hints_keep_order_and_execute_batches_once() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut b = CircuitBuilder::<BabyBear>::new();
    let mut outputs = Vec::new();
    for i in 0..3 {
        let calls = calls.clone();
        let output = b.hint_many::<2>("ordered pair", &[], move |_| {
            calls.lock().unwrap().push(i);
            Ok([BabyBear::from_usize(i), BabyBear::from_usize(i + 1)])
        });
        outputs.extend(output);
    }
    let circuit = b.finish();
    assert!(circuit.witness_chunks.is_empty());
    assert!(!circuit.witness_prefix);
    let assignment = circuit.witness().generate().unwrap();
    assert_eq!(*calls.lock().unwrap(), [0, 1, 2]);
    for (i, pair) in outputs.as_chunks::<2>().0.iter().enumerate() {
        assert_eq!(assignment.value(pair[0]).unwrap(), BabyBear::from_usize(i));
        assert_eq!(
            assignment.value(pair[1]).unwrap(),
            BabyBear::from_usize(i + 1)
        );
    }
}

#[test]
fn purity_marking_alone_does_not_install_a_schedule() {
    let (mut b, inputs, outputs, _) = blocks();
    let tail = b.hint_pure("pure tail", &outputs, |v| Ok(v[0] + v[1]));
    b.expose_public(tail);
    let circuit = b.finish();
    assert!(circuit.witness_chunks.is_empty());
    assert!(!circuit.witness_prefix);
    let witness = || {
        let mut w = circuit.witness();
        w.set(inputs[0], BabyBear::from_u8(2)).unwrap();
        w.set(inputs[1], BabyBear::from_u8(3)).unwrap();
        w
    };
    let expected = witness().generate_with(arithmetic_value).unwrap();
    let actual = witness().generate().unwrap();
    assert_eq!(
        actual.values(circuit.id()).unwrap(),
        expected.values(circuit.id()).unwrap()
    );
    assert_eq!(actual.public_values(), [BabyBear::from_u8(20)]);
}

#[test]
fn prefix_certification_rejects_opaque_hints_and_invalid_boundaries() {
    for invalid in 0..8 {
        let (mut b, _, _, ends) = blocks();
        let last = *ends.last().unwrap();
        let boundaries = match invalid {
            0 => vec![],
            1 => vec![last],
            2 => vec![0, last],
            3 => vec![ends[0], ends[0], last],
            4 => vec![last, ends[0]],
            5 => vec![ends[0], last - 1],
            // Zero, input, then the first output of a two-output hint.
            6 => vec![3, last],
            7 => vec![ends[0], last + 1],
            _ => unreachable!(),
        };
        assert!(!b.try_parallel_witness_prefix(boundaries));
        let circuit = b.finish();
        assert!(circuit.witness_chunks.is_empty());
        assert!(!circuit.witness_prefix);
    }
    for batched in [false, true] {
        let (mut b, _, _, mut ends) = blocks();
        if batched {
            b.hint_many::<2>("opaque", &[], |_| Ok([BabyBear::ZERO; 2]));
        } else {
            b.hint("opaque", &[], |_| Ok(BabyBear::ZERO));
        }
        *ends.last_mut().unwrap() = b.stats().values;
        assert!(!b.try_parallel_witness_prefix(ends));
    }
    let (mut b, _, outputs, mut ends) = blocks();
    b.hint_pure("cross-chunk derived read", &[outputs[0]], |v| Ok(v[0]));
    *ends.last_mut().unwrap() = b.stats().values;
    assert!(!b.try_parallel_witness_prefix(ends));

    let mut census = CircuitBuilder::<BabyBear>::counting();
    census.hint_pure("one", &[], |_| Ok(BabyBear::ONE));
    let first = census.stats().values;
    census.hint_many_pure::<2>("two", &[], |_| Ok([BabyBear::ONE; 2]));
    let before = census.stats();
    assert!(!census.try_parallel_witness_prefix(vec![first, before.values]));
    assert_eq!(census.stats(), before);
}

#[test]
fn prefix_barrier_preserves_values_original_errors_and_serial_suffix() {
    let (mut b, inputs, outputs, ends) = blocks();
    let prefix_end = *ends.last().unwrap();
    assert!(b.try_parallel_witness_prefix(ends));
    let suffix_input = b.input("suffix input");
    let calls = Arc::new(AtomicUsize::new(0));
    let suffix_calls = calls.clone();
    let suffix = b.hint_many::<2>(
        "ordinary suffix",
        &[outputs[0], outputs[1], suffix_input],
        move |v| {
            suffix_calls.fetch_add(1, Ordering::Relaxed);
            if v[2] == BabyBear::NEG_ONE {
                return Err("suffix sentinel".into());
            }
            Ok([v[0] + v[1] + v[2], v[0] * v[1]])
        },
    );
    b.expose_public(suffix[0]);
    b.expose_public(suffix[1]);
    let circuit = b.finish();
    assert!(circuit.witness_prefix);
    assert_eq!(circuit.witness_chunks.last(), Some(&prefix_end));
    assert!(prefix_end < circuit.num_values());
    for first in [None, Some(BabyBear::NEG_ONE), Some(BabyBear::from_u8(2))] {
        for second in [None, Some(BabyBear::NEG_ONE), Some(BabyBear::from_u8(3))] {
            for tail in [None, Some(BabyBear::NEG_ONE), Some(BabyBear::from_u8(5))] {
                let witness = || {
                    let mut w = circuit.witness();
                    for (input, value) in [inputs[0], inputs[1], suffix_input]
                        .into_iter()
                        .zip([first, second, tail])
                    {
                        if let Some(value) = value {
                            w.set(input, value).unwrap();
                        }
                    }
                    w
                };
                calls.store(0, Ordering::Relaxed);
                let expected = witness().generate_with(arithmetic_value);
                let expected_calls = calls.load(Ordering::Relaxed);
                calls.store(0, Ordering::Relaxed);
                let actual = witness().generate();
                assert_eq!(calls.load(Ordering::Relaxed), expected_calls);
                match (expected, actual) {
                    (Ok(expected), Ok(actual)) => {
                        assert_eq!(
                            actual.values(circuit.id()).unwrap(),
                            expected.values(circuit.id()).unwrap()
                        );
                        assert_eq!(actual.public_values(), expected.public_values());
                        assert_eq!(
                            actual.public_values(),
                            [BabyBear::from_u8(25), BabyBear::from_u8(91)]
                        );
                        assert_eq!(expected_calls, 1);
                    }
                    (Err(expected), Err(actual)) => assert_eq!(actual, expected),
                    _ => panic!("scheduled prefix differs from serial evaluation"),
                }
            }
        }
    }
}

#[test]
fn suffix_recipe_errors_precede_prefix_relation_failures() {
    for failure in 0..3 {
        let (mut b, inputs, outputs, ends) = blocks();
        let bad_gate = b.stats().gates;
        b.assert_zero(outputs[0]);
        assert!(b.try_parallel_witness_prefix(ends));
        let tail = b.input("tail");
        b.hint("tail hint", &[tail], |v| {
            if v[0] == BabyBear::NEG_ONE {
                Err("tail error".into())
            } else {
                Ok(v[0])
            }
        });
        let circuit = b.finish();
        let witness = || {
            let mut w = circuit.witness();
            w.set(inputs[0], BabyBear::from_u8(2)).unwrap();
            w.set(inputs[1], BabyBear::from_u8(3)).unwrap();
            if failure != 0 {
                w.set(
                    tail,
                    if failure == 1 {
                        BabyBear::NEG_ONE
                    } else {
                        BabyBear::ONE
                    },
                )
                .unwrap();
            }
            w
        };
        let expected = match failure {
            0 => WitnessError::MissingInput {
                name: "tail".into(),
            },
            1 => WitnessError::HintFailed {
                name: "tail hint".into(),
                message: "tail error".into(),
            },
            _ => WitnessError::UnsatisfiedGate { index: bad_gate },
        };
        assert_eq!(
            witness().generate_with(arithmetic_value).err(),
            Some(expected.clone())
        );
        assert_eq!(witness().generate().err(), Some(expected));
    }
}
