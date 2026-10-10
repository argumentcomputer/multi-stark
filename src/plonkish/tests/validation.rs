use super::*;

fn serial_check<F: Field>(circuit: &Circuit<F>, values: &[F]) -> Result<(), WitnessError> {
    if let Some(hashes) = &circuit.hashes {
        hashes.check_values(values)?;
    }
    for (index, gate) in circuit.gates.iter().enumerate() {
        if gate.evaluate(values) != F::ZERO {
            return Err(WitnessError::UnsatisfiedGate { index });
        }
    }
    let mut row = Vec::new();
    for lookup in &circuit.lookups {
        let table = &circuit.tables[lookup.table.index];
        row.clear();
        row.extend(lookup.values.iter().map(|v| values[v.index]));
        if !table.indices.contains_key(&row) {
            return Err(WitnessError::LookupMissing {
                table: table.name.clone(),
            });
        }
    }
    Ok(())
}

fn fixture<F: Field>(rows: usize) -> (Circuit<F>, Vec<F>, Vec<[Value; 4]>) {
    let mut b = CircuitBuilder::<F>::new();
    let bytes = b.fixed_table("byte", (0..256).map(|v| vec![F::from_u16(v)]).collect());
    let products = b.fixed_table(
        "product",
        (0..16)
            .flat_map(|x| {
                (0..16).map(move |y| vec![F::from_u16(x), F::from_u16(y), F::from_u16(x * y)])
            })
            .collect(),
    );
    let mut wires = Vec::with_capacity(rows);
    for _ in 0..rows {
        let x = b.input("x");
        let y = b.input("y");
        let product = b.mul(x, y);
        let affine = b.affine([x, y], [F::from_u8(7), -F::from_u8(3)], F::from_u8(5));
        b.constrain_gate(
            [x, y, product],
            [F::from_u8(11), F::ZERO, F::ZERO, -F::from_u8(11), F::ZERO],
        );
        b.lookup(bytes, &[x]);
        b.lookup(products, &[x, y, product]);
        wires.push([x, y, product, affine]);
    }
    let circuit = b.finish();
    let mut witness = circuit.witness();
    for (i, &[x, y, _, _]) in wires.iter().enumerate() {
        witness.set(x, F::from_usize(i % 16)).unwrap();
        witness.set(y, F::from_usize((i * 7) % 16)).unwrap();
    }
    let values = witness.generate().unwrap().values().to_vec();
    (circuit, values, wires)
}

fn replace_row<F: Field>(values: &mut [F], wires: [Value; 4], x: F, y: F) {
    for (wire, value) in wires.into_iter().zip([
        x,
        y,
        x * y,
        F::from_u8(7) * x - F::from_u8(3) * y + F::from_u8(5),
    ]) {
        values[wire.index] = value;
    }
}

fn malformed_assignments<F: Field>() {
    let (circuit, values, wires) = fixture::<F>(4097);
    assert_eq!(circuit.check_values(&values), Ok(()));

    let mut lookup_errors = values.clone();
    replace_row(&mut lookup_errors, wires[7], F::from_u8(17), F::ONE);
    replace_row(&mut lookup_errors, wires[3075], F::from_u16(256), F::ONE);
    assert_eq!(
        serial_check(&circuit, &lookup_errors),
        Err(WitnessError::LookupMissing {
            table: "product".into()
        })
    );
    for _ in 0..4 {
        assert_eq!(
            circuit.check_values(&lookup_errors),
            serial_check(&circuit, &lookup_errors)
        );
    }

    let mut gate_and_lookup_errors = lookup_errors;
    gate_and_lookup_errors[wires[13][2].index] += F::ONE;
    gate_and_lookup_errors[wires[3075][3].index] += F::ONE;
    let first = serial_check(&circuit, &gate_and_lookup_errors);
    assert!(matches!(first, Err(WitnessError::UnsatisfiedGate { .. })));
    for _ in 0..4 {
        assert_eq!(circuit.check_values(&gate_and_lookup_errors), first);
    }

    for row in [0, 1023, 1024, 1365, 1366, 4096] {
        for wire in wires[row] {
            let mut forged = values.clone();
            forged[wire.index] += F::ONE;
            let expected = serial_check(&circuit, &forged);
            assert!(expected.is_err());
            assert_eq!(circuit.check_values(&forged), expected);
        }
    }

    let mut b = CircuitBuilder::<F>::new();
    b.enable_compact_blake3();
    let x = b.input("hash byte");
    b.record_hash(vec![x], &[x; 32]);
    b.assert_zero(x);
    let table = b.fixed_table("zero", vec![vec![F::ZERO]]);
    b.lookup(table, &[x]);
    let circuit = b.finish();
    let forged = vec![F::ONE; circuit.num_values()];
    assert_eq!(
        circuit.check_values(&forged),
        Err(WitnessError::HashMismatch)
    );
    assert_eq!(
        circuit.check_values(&forged),
        serial_check(&circuit, &forged)
    );
}

#[test]
fn parallel_validation_preserves_errors_goldilocks() {
    malformed_assignments::<crate::types::Val>();
}

#[cfg(feature = "kzg")]
#[test]
fn parallel_validation_preserves_errors_scalar() {
    malformed_assignments::<crate::ark_adapter::Scalar>();
}

#[cfg(feature = "kzg")]
#[test]
#[ignore = "isolated CPU scalar gate and lookup validation comparison"]
fn witness_validation_benchmark() {
    use std::{hint::black_box, time::Instant};
    let (circuit, values, _) = fixture::<crate::ark_adapter::Scalar>(1 << 17);
    serial_check(&circuit, &values).unwrap();
    circuit.check_values(&values).unwrap();
    for iteration in 0..5 {
        let mut seconds = [0.0; 2];
        for parallel in if iteration % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let started = Instant::now();
            let result = if parallel {
                black_box(&circuit).check_values(black_box(&values))
            } else {
                serial_check(black_box(&circuit), black_box(&values))
            };
            result.unwrap();
            seconds[usize::from(parallel)] = started.elapsed().as_secs_f64();
        }
        println!(
            "WITNESS_VALIDATION_BENCH iteration={iteration} gates={} lookups={} serial_seconds={:.6} parallel_seconds={:.6}",
            circuit.gates.len(),
            circuit.lookups.len(),
            seconds[0],
            seconds[1]
        );
    }
}
