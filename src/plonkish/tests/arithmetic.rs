use super::*;
use crate::{plonkish::CircuitBuilder, traits::Algebra, types::Val};

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
        assert_eq!(actual.values(), expected.values());
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

#[cfg(feature = "kzg")]
#[test]
fn specialized_scalar_arithmetic_preserves_values_and_errors() {
    coefficient_parity::<crate::ark_adapter::field::Scalar>();
    assignment_parity::<crate::ark_adapter::field::Scalar>();
}

#[cfg(feature = "kzg")]
#[test]
fn specialized_scalar_arithmetic_preserves_kzg_proof_bytes() {
    use crate::{
        ark_adapter::{KzgConfig, Srs, compact::FixedProofCodec, field::Scalar},
        system::{System, SystemWitness},
    };
    use std::sync::Arc;

    let (circuit, inputs) = fixture::<Scalar>(16);
    let assignments = [
        witness(&circuit, inputs, Scalar::ONE)
            .generate_with(reference)
            .unwrap(),
        witness(&circuit, inputs, Scalar::ONE).generate().unwrap(),
    ];
    let compiled = circuit.lower_to_multi_stark(Scalar::from_u8(86)).unwrap();
    let claims = compiled.claims(assignments[0].public_values()).unwrap();
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(128, b"arithmetic-witness-parity")),
        4,
    );
    let (system, key) = System::new(config, compiled.circuit_inputs());
    let refs = claims.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut proofs = Vec::new();
    for assignment in &assignments {
        let traces = compiled.traces(assignment).unwrap();
        let proof =
            system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(traces, &system));
        system.verify_multiple_claims(&refs, &proof).unwrap();
        proofs.push(
            FixedProofCodec::new(&system, &proof.log_degrees)
                .unwrap()
                .encode(&proof)
                .unwrap(),
        );
    }
    assert_eq!(proofs[0], proofs[1]);
    println!(
        "ARITHMETIC_PROOF bytes={} blake3={}",
        proofs[0].len(),
        blake3::hash(&proofs[0])
    );
}

fn benchmark<F: Field>(field: &str) {
    use std::{hint::black_box, time::Instant};
    let (circuit, inputs) = fixture::<F>(1 << 17);
    let expected = witness(&circuit, inputs, F::ONE)
        .generate_with(reference)
        .unwrap();
    let evaluate = |specialized| {
        let witness = witness(black_box(&circuit), inputs, F::ONE);
        let started = Instant::now();
        let assignment = if specialized {
            witness.generate().unwrap()
        } else {
            witness.generate_with(reference).unwrap()
        };
        let seconds = started.elapsed().as_secs_f64();
        assert_eq!(black_box(assignment.values()), expected.values());
        seconds
    };
    evaluate(false);
    evaluate(true);
    for iteration in 0..5 {
        let mut seconds = [0.0; 2];
        for specialized in if iteration % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            seconds[usize::from(specialized)] = evaluate(specialized);
        }
        println!(
            "ARITHMETIC_BENCH field={field} iteration={iteration} values={} gates={} reference_seconds={:.9} specialized_seconds={:.9}",
            circuit.num_values(),
            circuit.gates.len(),
            seconds[0],
            seconds[1]
        );
    }
}

#[cfg(feature = "kzg")]
#[test]
#[ignore = "isolated arithmetic witness comparison"]
fn specialized_arithmetic_benchmark() {
    benchmark::<Val>("goldilocks");
    benchmark::<crate::ark_adapter::field::Scalar>("scalar");
}
