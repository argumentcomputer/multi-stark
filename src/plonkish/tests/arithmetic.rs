use super::*;
use crate::{plonkish::CircuitBuilder, types::Val};

fn reference<F: Field>(coefficients: [F; 5], a: F, b: F) -> F {
    let [qm, qa, qb, _, k] = coefficients;
    let mut result = k;
    if qm != F::ZERO {
        result += qm * a * b;
    }
    if qa != F::ZERO {
        result += qa * a;
    }
    if qb != F::ZERO {
        result += qb * b;
    }
    result
}

fn coefficient_parity<F: Field>() {
    let coefficients = [F::ZERO, F::ONE, F::NEG_ONE, F::from_u8(7), -F::from_u8(19)];
    let values = [
        F::ZERO,
        F::ONE,
        F::NEG_ONE,
        F::from_u64(u64::MAX),
        F::from_u64(0xab7e_0956_dcad_2347).exp_u64(13),
    ];
    for qm in coefficients {
        for qa in coefficients {
            for qb in coefficients {
                for k in coefficients {
                    for a in values {
                        for b in values {
                            let coefficients = [qm, qa, qb, F::NEG_ONE, k];
                            assert_eq!(
                                arithmetic_value(coefficients, a, b),
                                reference(coefficients, a, b)
                            );
                        }
                    }
                }
            }
        }
    }
}

fn fixture<F: Field>(rounds: usize) -> (Circuit<F>, [Value; 3]) {
    let mut builder = CircuitBuilder::<F>::new();
    let inputs = [builder.input("a"), builder.input("b"), builder.input("bit")];
    let bit = builder.assert_bool(inputs[2]);
    let mut value = inputs[0];
    for i in 0..rounds {
        value = match i % 8 {
            0 => builder.add(value, inputs[1]),
            1 => builder.sub(value, inputs[1]),
            2 => builder.mul(value, inputs[1]),
            3 => builder.mul_add(value, inputs[1], value),
            4 => builder.affine(
                [value, inputs[1]],
                [F::ONE, F::from_u64((1 << 32) + 15)],
                F::NEG_ONE,
            ),
            5 => builder.affine(
                [value, inputs[1]],
                [F::NEG_ONE, F::from_u8(7)],
                F::from_u8(5),
            ),
            6 => builder.select(bit, value, inputs[1]),
            _ => builder.affine(
                [value, inputs[1]],
                [F::from_u8(13), -F::from_u64((1 << 48) + 27)],
                F::from_u8(19),
            ),
        };
    }
    let reciprocal = builder.inverse(inputs[1]);
    builder.expose_public(value);
    builder.expose_public(reciprocal);
    (builder.finish(), inputs)
}

fn witness<F: Field>(circuit: &Circuit<F>, inputs: [Value; 3], bit: F) -> Witness<'_, F> {
    let mut witness = circuit.witness();
    for (input, value) in inputs.into_iter().zip([F::NEG_ONE, F::from_u8(7), bit]) {
        witness.set(input, value).unwrap();
    }
    witness
}

fn assignment_parity<F: Field>() {
    let (circuit, inputs) = fixture::<F>(257);
    for bit in [F::ZERO, F::ONE] {
        let expected = witness(&circuit, inputs, bit)
            .generate_with(reference)
            .unwrap();
        let actual = witness(&circuit, inputs, bit).generate().unwrap();
        assert_eq!(
            actual.values(circuit.id()).unwrap(),
            expected.values(circuit.id()).unwrap()
        );
        assert_eq!(actual.public_values(), expected.public_values());
    }
    let expected = witness(&circuit, inputs, F::TWO)
        .generate_with(reference)
        .err();
    let actual = witness(&circuit, inputs, F::TWO).generate().err();
    assert!(matches!(actual, Some(WitnessError::UnsatisfiedGate { .. })));
    assert_eq!(actual, expected);
    let expected = circuit.witness().generate_with(reference).err();
    let actual = circuit.witness().generate().err();
    assert_eq!(actual, expected);

    let mut builder = CircuitBuilder::<F>::new();
    builder.hint("failed advice", &[], |_| Err("invalid input".into()));
    let circuit = builder.finish();
    assert_eq!(
        circuit.witness().generate().err(),
        circuit.witness().generate_with(reference).err()
    );
}

#[test]
fn specialized_goldilocks_arithmetic_preserves_values_and_errors() {
    coefficient_parity::<Val>();
    assignment_parity::<Val>();
}
#[test]
fn specialized_baby_bear_arithmetic_preserves_values_and_errors() {
    coefficient_parity::<p3_baby_bear::BabyBear>();
    assignment_parity::<p3_baby_bear::BabyBear>();
}
