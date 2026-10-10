use super::*;

fn single_limb_range(b: &mut CircuitBuilder<Scalar>, table: Table, value: Value, bits: usize) {
    if bits == 0 {
        let zero = b.constant(Scalar::ZERO);
        b.assert_equal(value, zero);
        return;
    }
    if bits == 1 {
        b.assert_bool(value);
        return;
    }
    let mut terms = Vec::new();
    let mut coefficient = Scalar::ONE;
    for i in 0..bits.div_ceil(16) {
        let limb = b.hint("integer limb", &[value], move |v| {
            let words = v[0].canonical_limbs_le();
            Ok(Scalar::from_u64((words[i / 4] >> (16 * (i % 4))) & 0xffff))
        });
        b.lookup(table, &[limb]);
        terms.push((coefficient, limb));
        coefficient *= Scalar::from_u32(1 << 16);
    }
    let packed = b.linear_combination(&terms, Scalar::ZERO);
    b.assert_equal(value, packed);
}

fn fixture(bits: &[usize], grouped: bool, table_size: u32) -> (Circuit<Scalar>, Vec<Value>) {
    let mut b = CircuitBuilder::<Scalar>::new();
    let table = b.fixed_table(
        "limbs",
        (0..table_size).map(|v| vec![Scalar::from_u32(v)]).collect(),
    );
    let inputs = bits.iter().map(|_| b.input("integer")).collect::<Vec<_>>();
    for (&bits, &value) in bits.iter().zip(&inputs) {
        if grouped {
            range(&mut b, table, value, bits);
        } else {
            single_limb_range(&mut b, table, value, bits);
        }
        b.expose_public(value);
    }
    (b.finish(), inputs)
}

fn assign(circuit: &Circuit<Scalar>, inputs: &[Value], values: &[Scalar]) -> Assignment<Scalar> {
    let mut witness = circuit.witness();
    for (&input, &value) in inputs.iter().zip(values) {
        witness.set(input, value).unwrap();
    }
    witness.generate().unwrap()
}

#[test]
fn grouped_range_hints_preserve_wires_constraints_and_boundary_values() {
    let bits = [
        0, 1, 2, 16, 17, 32, 33, 48, 64, 65, 80, 96, 112, 128, 129, 130,
    ];
    let (single, old_inputs) = fixture(&bits, false, 1 << 16);
    let (grouped, new_inputs) = fixture(&bits, true, 1 << 16);
    assert_eq!(
        old_inputs.iter().map(|v| v.index).collect::<Vec<_>>(),
        new_inputs.iter().map(|v| v.index).collect::<Vec<_>>()
    );
    let mut expected = single.stats();
    assert!(grouped.stats().hint_calls < expected.hint_calls);
    expected.hint_calls = grouped.stats().hint_calls;
    assert_eq!(grouped.stats(), expected);
    for (old, new) in single.gates.iter().zip(&grouped.gates) {
        assert_eq!(old.wires.map(|v| v.index), new.wires.map(|v| v.index));
        assert_eq!(old.coefficients, new.coefficients);
    }
    for (old, new) in single.lookups.iter().zip(&grouped.lookups) {
        assert_eq!(old.table.index, new.table.index);
        assert_eq!(
            old.values.iter().map(|v| v.index).collect::<Vec<_>>(),
            new.values.iter().map(|v| v.index).collect::<Vec<_>>()
        );
    }
    for digit in [0u64, 1, 0x5555, 0xffff] {
        let values = bits.map(|bits| {
            if bits <= 1 {
                return Scalar::from_u64(digit & bits as u64);
            }
            (0..bits.div_ceil(16)).fold(Scalar::ZERO, |value, _| {
                value * Scalar::from_u32(1 << 16) + Scalar::from_u64(digit)
            })
        });
        let old = assign(&single, &old_inputs, &values);
        let new = assign(&grouped, &new_inputs, &values);
        assert_eq!(old.values(), new.values());
        assert_eq!(old.public_values(), new.public_values());
    }
}

#[test]
fn grouped_range_hints_still_reject_non_limbs_with_valid_packing() {
    let (circuit, inputs) = fixture(&[32], true, 1 << 16);
    let assignment = assign(&circuit, &inputs, &[Scalar::ZERO]);
    let mut forged = assignment.values().to_vec();
    let hint = circuit
        .hints
        .iter()
        .find(|hint| hint.name == "integer limbs")
        .unwrap();
    assert_eq!(hint.len, 2);
    forged[hint.start] = Scalar::from_u32(1 << 16);
    forged[hint.start + 1] = Scalar::NEG_ONE;
    assert!(
        circuit
            .gates
            .iter()
            .all(|gate| gate.evaluate(&forged) == Scalar::ZERO)
    );
    assert_eq!(
        circuit.check_values(&forged),
        Err(WitnessError::LookupMissing {
            table: "limbs".into()
        })
    );
}

#[test]
fn grouped_range_hints_preserve_kzg_proof_bytes() {
    use crate::{
        ark_adapter::{KzgConfig, Srs, compact::FixedProofCodec},
        system::{System, SystemWitness},
    };
    use std::sync::Arc;
    let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(64, b"range-hint-parity")), 4);
    let mut proofs = Vec::new();
    for grouped in [false, true] {
        let (circuit, inputs) = fixture(&[17, 32], grouped, 16);
        let assignment = assign(
            &circuit,
            &inputs,
            &[
                Scalar::from_u32((1 << 16) + 9),
                Scalar::from_u32((7 << 16) + 3),
            ],
        );
        let compiled = circuit.lower_to_multi_stark(Scalar::from_u8(77)).unwrap();
        let traces = compiled.traces(&assignment).unwrap();
        let claims = compiled.claims(assignment.public_values()).unwrap();
        let (system, key) = System::new(config.clone(), compiled.circuit_inputs());
        let refs = claims.iter().map(Vec::as_slice).collect::<Vec<_>>();
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
}

#[test]
#[ignore = "isolated CPU range-hint assignment comparison"]
fn grouped_range_hint_benchmark() {
    use std::{hint::black_box, time::Instant};
    let bits = (0..1 << 14)
        .map(|i| [32, 64, 130][i % 3])
        .collect::<Vec<_>>();
    let single = fixture(&bits, false, 1 << 16);
    let grouped = fixture(&bits, true, 1 << 16);
    let values = bits
        .iter()
        .map(|&bits| {
            (0..bits.div_ceil(16)).fold(Scalar::ZERO, |value, _| {
                value * Scalar::from_u32(1 << 16) + Scalar::from_u16(0x5555)
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        assign(&single.0, &single.1, &values).values(),
        assign(&grouped.0, &grouped.1, &values).values()
    );
    for iteration in 0..5 {
        let mut seconds = [0.0; 2];
        for is_grouped in if iteration % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let (circuit, inputs) = if is_grouped { &grouped } else { &single };
            let started = Instant::now();
            let assignment = assign(black_box(circuit), black_box(inputs), black_box(&values));
            seconds[usize::from(is_grouped)] = started.elapsed().as_secs_f64();
            black_box(assignment);
        }
        println!(
            "RANGE_HINT_BENCH iteration={iteration} hints_single={} hints_grouped={} single_seconds={:.6} grouped_seconds={:.6}",
            single.0.stats().hint_calls,
            grouped.0.stats().hint_calls,
            seconds[0],
            seconds[1]
        );
    }
}
