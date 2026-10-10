use super::*;
use crate::plonkish::{CircuitBuilder, gadgets::ByteGadgets, gadgets::blake3};
use crate::traits::PrimeField;
use crate::types::Val;

// Keep both outputs behind a call boundary, as in the shared row implementation.
#[inline(never)]
fn both_outputs<F: Field>(
    compiled: &MultiStarkCircuit<F>,
    row: usize,
    wires: &mut [Value],
    fixed: &mut [F],
) {
    wires.fill(compiled.circuit.zero);
    fixed.fill(F::ZERO);
    let c = &compiled.circuit;
    let public_enable = 5 + 2 * compiled.width;
    let public_index = public_enable + 1;
    let table_enable = public_enable + 2;
    let table_index = public_enable + 3;
    if let Some(gate) = c.gates.get(row) {
        wires[..3].copy_from_slice(&gate.wires);
        fixed[..5].copy_from_slice(&gate.coefficients);
    } else if row < c.gates.len() + 1 + c.publics.len() {
        let index = row - c.gates.len();
        wires[0] = if index == 0 {
            c.zero
        } else {
            c.publics[index - 1]
        };
        fixed[public_enable] = F::ONE;
        fixed[public_index] = F::from_usize(index);
    } else {
        let index = row - c.gates.len() - 1 - c.publics.len();
        if let Some(lookup) = c.lookups.get(index) {
            wires[..lookup.values.len()].copy_from_slice(&lookup.values);
            fixed[table_enable] = F::ONE;
            fixed[table_index] = F::from_usize(lookup.table.index);
        } else {
            let index = index - c.lookups.len();
            let call_index = compiled.hash_row_ends.partition_point(|&end| end <= index);
            if let Some(call) = c.hashes.as_ref().and_then(|h| h.calls.get(call_index)) {
                let offset = index
                    - call_index
                        .checked_sub(1)
                        .map_or(0, |i| compiled.hash_row_ends[i]);
                wires[0] = if offset < call.input.len() {
                    call.input[offset]
                } else {
                    call.output[offset - call.input.len()]
                };
                fixed[table_index] = F::from_usize(call_index);
                fixed[public_index] = F::from_usize(offset);
                fixed[table_index + 1] = if offset < call.input.len() {
                    F::ONE
                } else {
                    F::NEG_ONE
                };
            }
        }
    }
}

fn mixed_circuit<F: PrimeField>(gates: usize, lookups: usize) -> (Circuit<F>, Value) {
    let mut builder = CircuitBuilder::<F>::new();
    let seed = builder.input("seed");
    let mut value = seed;
    let one = builder.constant(F::ONE);
    for index in 0..gates {
        value = if index % 3 == 0 {
            builder.mul(value, seed)
        } else {
            builder.add(value, one)
        };
    }
    builder.expose_public(seed);
    builder.expose_public(value);
    let wide_row: Vec<_> = (0..5).map(F::from_usize).collect();
    let wide_values: Vec<_> = wide_row
        .iter()
        .map(|&value| builder.constant(value))
        .collect();
    let wide = builder.fixed_table("wide", vec![wide_row]);
    let narrow = builder.fixed_table("narrow", vec![vec![F::ONE]]);
    for index in 0..lookups {
        if index % 3 == 0 {
            builder.lookup(narrow, &[one]);
        } else {
            builder.lookup(wide, &wide_values);
        }
    }
    builder.enable_compact_blake3();
    let bytes = ByteGadgets::new(&mut builder);
    for len in [0_usize, 1, 65, 129] {
        let input: Vec<_> = (0..len)
            .map(|index| bytes.constant(&mut builder, (index * 17 + 11) as u8))
            .collect();
        let digest = blake3(&mut builder, &bytes, &input);
        if len == 65 {
            let second = blake3(&mut builder, &bytes, &digest);
            for byte in second {
                builder.expose_public(byte.value());
            }
        }
    }
    (builder.finish(), seed)
}

fn assigned<F: Field>(compiled: &MultiStarkCircuit<F>, seed: Value) -> Assignment<F> {
    let mut witness = compiled.witness();
    witness.set(seed, F::from_u8(7)).unwrap();
    witness.generate().unwrap()
}

fn copy_links<F: Field, const SELECTED: bool>(compiled: &MultiStarkCircuit<F>) -> Vec<usize> {
    let mut successors = vec![0; compiled.main_height() * compiled.width];
    let mut endpoints = vec![(usize::MAX, usize::MAX); compiled.circuit.num_values()];
    let mut wires = vec![compiled.circuit.zero; compiled.width];
    let mut fixed = if SELECTED {
        vec![]
    } else {
        F::zero_vec(compiled.fixed.width())
    };
    for row in 0..compiled.main_height() {
        if SELECTED {
            compiled.layout_wires(row, &mut wires);
        } else {
            both_outputs(compiled, row, &mut wires, &mut fixed);
        }
        for (col, value) in wires.iter().enumerate() {
            let cell = row * compiled.width + col;
            let (first, last) = &mut endpoints[value.index];
            if *first == usize::MAX {
                *first = cell;
            } else {
                successors[*last] = cell;
            }
            *last = cell;
        }
    }
    for (first, last) in endpoints {
        if first != usize::MAX {
            successors[last] = first;
        }
    }
    successors
}

fn both_output_main<F: Field>(
    compiled: &MultiStarkCircuit<F>,
    assignment: &Assignment<F>,
    index: usize,
) -> RowMajorMatrix<F> {
    let start = compiled.main_heights[..index].iter().sum::<usize>();
    let mut values = F::zero_vec(compiled.main_heights[index] * compiled.width);
    values
        .par_chunks_mut(compiled.width * (1 << 12))
        .enumerate()
        .for_each(|(tile, output)| {
            let mut wires = vec![compiled.circuit.zero; compiled.width];
            let mut fixed = vec![F::ZERO; compiled.fixed.width()];
            for (offset, row_values) in output.chunks_exact_mut(compiled.width).enumerate() {
                both_outputs(
                    compiled,
                    start + tile * (1 << 12) + offset,
                    &mut wires,
                    &mut fixed,
                );
                for (output, wire) in row_values.iter_mut().zip(&wires) {
                    *output = assignment.values()[wire.index];
                }
            }
        });
    RowMajorMatrix::new(values, compiled.width)
}

fn both_output_fixed<F: Field>(compiled: &MultiStarkCircuit<F>, index: usize) -> RowMajorMatrix<F> {
    let start = compiled.main_heights[..index].iter().sum::<usize>();
    let width = compiled.fixed.width();
    let successors = compiled.successors.as_ref().unwrap();
    let mut values = F::zero_vec(compiled.main_heights[index] * width);
    values
        .par_chunks_mut(width * (1 << 12))
        .enumerate()
        .for_each(|(tile, output)| {
            let mut wires = vec![compiled.circuit.zero; compiled.width];
            for (offset, fixed) in output.chunks_exact_mut(width).enumerate() {
                let row = start + tile * (1 << 12) + offset;
                both_outputs(compiled, row, &mut wires, fixed);
                for col in 0..compiled.width {
                    let cell = row * compiled.width + col;
                    fixed[5 + col] = F::from_usize(cell);
                    fixed[5 + compiled.width + col] = F::from_usize(successors[cell]);
                }
            }
        });
    if compiled.main_heights.len() > 1 {
        values[width - 1] = F::ONE;
    }
    RowMajorMatrix::new(values, width)
}

fn check_rows<F: PrimeField>() {
    let (circuit, seed) = mixed_circuit::<F>(5003, 65_497);
    let lazy = circuit
        .lower_to_multi_stark_sharded(F::from_u8(143), 1 << 16)
        .unwrap();
    let assignment = assigned(&lazy, seed);
    let (circuit, seed) = mixed_circuit::<F>(5003, 65_497);
    let dense = circuit
        .lower_to_multi_stark_with_max_height(F::from_u8(143), 1 << 16)
        .unwrap();
    let dense_assignment = assigned(&dense, seed);
    assert!(lazy.main_heights.len() > 1);
    assert!(lazy.main_heights.last().unwrap() < &(1 << 16));
    assert_eq!(lazy.main_heights, dense.main_heights);
    assert_eq!(
        copy_links::<_, false>(&lazy),
        *lazy.successors.as_ref().unwrap()
    );
    assert_eq!(
        copy_links::<_, true>(&lazy),
        *lazy.successors.as_ref().unwrap()
    );
    let mut wires = vec![seed; lazy.width];
    let mut expected_wires = wires.clone();
    let mut fixed = vec![F::ONE; lazy.fixed.width()];
    let mut expected_fixed = fixed.clone();
    for row in 0..lazy.main_height() {
        both_outputs(&lazy, row, &mut expected_wires, &mut expected_fixed);
        lazy.layout_wires(row, &mut wires);
        lazy.layout_fixed(row, &mut fixed);
        assert_eq!(wires, expected_wires, "wire row {row}");
        assert_eq!(fixed, expected_fixed, "fixed row {row}");
    }
    for index in 0..lazy.main_heights.len() {
        let fixed = lazy.fixed_partition(index);
        assert_eq!(fixed, both_output_fixed(&lazy, index));
        assert_eq!(fixed, dense.fixed_partition(index));
        let main = lazy.main_trace(&assignment, index);
        assert_eq!(main, both_output_main(&lazy, &assignment, index));
        assert_eq!(main, dense.main_trace(&dense_assignment, index));
    }
    assert_eq!(
        lazy.claims(assignment.public_values()).unwrap(),
        dense.claims(dense_assignment.public_values()).unwrap()
    );
}

#[test]
fn selected_rows_match_dense_and_shared_goldilocks() {
    check_rows::<Val>();
}

#[cfg(feature = "kzg")]
#[test]
fn selected_rows_match_dense_and_shared_scalar() {
    check_rows::<crate::ark_adapter::Scalar>();
}

#[test]
fn selected_rows_preserve_fresh_proof_bytes() {
    use crate::system::{System, SystemWitness};
    use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config};
    let config = || {
        GoldilocksBlake3Config::new(
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
        )
    };
    let (circuit, seed) = mixed_circuit::<Val>(17, 31);
    let lazy = circuit
        .lower_to_multi_stark_sharded(Val::from_u8(143), 1 << 16)
        .unwrap();
    let assignment = assigned(&lazy, seed);
    let (circuit, seed) = mixed_circuit::<Val>(17, 31);
    let dense = circuit
        .lower_to_multi_stark_with_max_height(Val::from_u8(143), 1 << 16)
        .unwrap();
    let dense_assignment = assigned(&dense, seed);
    let claims = lazy.claims(assignment.public_values()).unwrap();
    assert_eq!(
        claims,
        dense.claims(dense_assignment.public_values()).unwrap()
    );
    let claims: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let (actual, actual_key) = System::new(config(), lazy.circuit_inputs());
    let (reference, reference_key) = System::new(config(), dense.circuit_inputs());
    let actual_proof = actual.prove_multiple_claims(
        &actual_key,
        &claims,
        SystemWitness::from_stage_1(lazy.traces(&assignment).unwrap(), &actual),
    );
    let reference_proof = reference.prove_multiple_claims(
        &reference_key,
        &claims,
        SystemWitness::from_stage_1(dense.traces(&dense_assignment).unwrap(), &reference),
    );
    actual
        .verify_multiple_claims(&claims, &actual_proof)
        .unwrap();
    reference
        .verify_multiple_claims(&claims, &reference_proof)
        .unwrap();
    assert_eq!(
        actual_proof.to_bytes().unwrap(),
        reference_proof.to_bytes().unwrap()
    );
}

fn measure<T: PartialEq>(name: &str, mut before: impl FnMut() -> T, mut after: impl FnMut() -> T) {
    use std::{hint::black_box, time::Instant};
    assert!(
        black_box(before()) == black_box(after()),
        "{name} warmup differs"
    );
    let mut old = Vec::new();
    let mut new = Vec::new();
    for sample in 0..5 {
        let mut timed = |selected| {
            let start = Instant::now();
            let value = black_box(if selected { after() } else { before() });
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if selected {
                new.push(elapsed)
            } else {
                old.push(elapsed)
            }
            value
        };
        let first = timed(sample % 2 != 0);
        let second = timed(sample % 2 == 0);
        assert!(first == second, "{name} sample {sample} differs");
    }
    println!("row_selection_result name={name} before_ms={old:?} after_ms={new:?}");
}

fn benchmark<F: PrimeField>(name: &str) {
    let (circuit, seed) = mixed_circuit::<F>(1 << 15, 1 << 17);
    let layout = circuit.multi_stark_layout().unwrap();
    let compiled = circuit
        .lower_to_multi_stark_sharded(F::from_u8(143), 1 << 18)
        .unwrap();
    let assignment = assigned(&compiled, seed);
    assert_eq!(compiled.main_heights, [1 << 18]);
    println!(
        "row_selection_shape field={name} used_rows={} padded_rows={} gates={} publics={} lookups={} hash_calls={} main_width={} fixed_width={} main_bytes={} fixed_bytes={}",
        layout.used_rows,
        compiled.main_height(),
        compiled.circuit.gates.len(),
        compiled.circuit.publics.len(),
        compiled.circuit.lookups.len(),
        compiled.circuit.hashes.as_ref().unwrap().calls.len(),
        compiled.width,
        compiled.fixed.width(),
        compiled.main_height() * compiled.width * size_of::<F>(),
        compiled.main_height() * compiled.fixed.width() * size_of::<F>()
    );
    measure(
        &format!("{name}/copy_links"),
        || copy_links::<_, false>(&compiled),
        || copy_links::<_, true>(&compiled),
    );
    measure(
        &format!("{name}/main_trace"),
        || both_output_main(&compiled, &assignment, 0),
        || compiled.main_trace(&assignment, 0),
    );
    measure(
        &format!("{name}/fixed_partition"),
        || both_output_fixed(&compiled, 0),
        || compiled.fixed_partition(0),
    );
}

#[test]
#[ignore = "isolated allocation-inclusive row selection comparison"]
fn selected_rows_benchmark() {
    benchmark::<Val>("goldilocks");
    #[cfg(feature = "kzg")]
    benchmark::<crate::ark_adapter::Scalar>("scalar");
}
