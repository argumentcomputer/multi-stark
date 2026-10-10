use super::*;
use crate::plonkish::{CircuitBuilder, gadgets::ByteGadgets, gadgets::blake3};
use crate::system::{System, SystemWitness};
use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};

fn occurrence_fixed<F: Field>(hashes: &Compact) -> Vec<Vec<F>> {
    let mut fixed: Vec<_> = (0..6)
        .map(|t| vec![F::ZERO; hashes.rows[t].len().max(2).next_power_of_two() * fw(t)])
        .collect();
    let mut occurrences = vec![vec![]; hashes.recipes.len()];
    let mut label = 0usize;
    for t in 0..6 {
        fixed[t][1] = F::ONE;
        for (r, row) in hashes.rows[t].iter().enumerate() {
            fixed[t][r * fw(t)] = F::ONE;
            for (slot, &word) in row.iter().enumerate() {
                fixed[t][r * fw(t) + 2 + slot] = f(label);
                occurrences[word].push((t, r, slot, label));
                label += 1;
            }
        }
    }
    assert!(F::prime_order_exceeds(label));
    for uses in occurrences {
        for (i, &(t, r, slot, _)) in uses.iter().enumerate() {
            fixed[t][r * fw(t) + 2 + slots(t) + slot] = f(uses[(i + 1) % uses.len()].3);
        }
    }
    fixed
}

fn serial_trace<F: Field>(
    hashes: &Compact,
    prepared: &Prepared,
    defs: &[CircuitInputs<F>],
    index: usize,
    field_allocation: bool,
) -> RowMajorMatrix<F> {
    let definition = &defs[index];
    if index >= 6 {
        return RowMajorMatrix::new(
            prepared.multiplicities[index - 6]
                .iter()
                .map(|&n| F::from_usize(n))
                .collect(),
            definition.main_width,
        );
    }
    let words = &prepared.words;
    let bytes: [F; 256] = std::array::from_fn(F::from_usize);
    let len = definition.preprocessed.as_ref().unwrap().height() * definition.main_width;
    let mut trace = RowMajorMatrix::new(
        if field_allocation {
            F::zero_vec(len)
        } else {
            vec![F::ZERO; len]
        },
        definition.main_width,
    );
    for (r, slots) in hashes.rows[index].iter().enumerate() {
        let row = &mut trace.values[r * mw(index)..(r + 1) * mw(index)];
        for (slot, &word) in slots.iter().enumerate() {
            for (i, b) in words[word].to_le_bytes().into_iter().enumerate() {
                row[slot * 4 + i] = bytes[usize::from(b)];
            }
        }
        if index < 4 {
            let mut carry = 0u16;
            for i in 0..4 {
                let sum = u16::from(words[slots[0]].to_le_bytes()[i])
                    + u16::from(words[slots[1]].to_le_bytes()[i])
                    + u16::from(words[slots[2]].to_le_bytes()[i])
                    + carry;
                carry = sum >> 8;
                row[24 + i] = bytes[usize::from(carry)];
                let x = words[slots[3]].to_le_bytes()[i] ^ words[slots[4]].to_le_bytes()[i];
                row[28 + i] = bytes[usize::from(x)];
                let bits = ROT[index] % 8;
                if bits != 0 {
                    row[32 + i] = bytes[usize::from(x & ((1 << bits) - 1))];
                    row[36 + i] = bytes[usize::from(x >> bits)];
                }
            }
        }
    }
    trace
}

fn packing_fixture(rows: usize, kinds: &[usize]) -> (Compact, Prepared) {
    let mut hashes = Compact::default();
    for &kind in kinds {
        hashes.rows[kind] = (0..rows)
            .map(|row| (0..slots(kind)).map(|slot| row * 6 + slot).collect())
            .collect();
    }
    let mut state = 0x3141_5926_u32;
    let words = (0..rows * 6)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            match index % 257 {
                0..=2 => u32::MAX,
                3 => 0,
                _ => state,
            }
        })
        .collect();
    let multiplicities = [65536, 512, 256].map(|len| (0..len).map(|i| i * 17 + 3).collect());
    (
        hashes,
        Prepared {
            words,
            multiplicities,
        },
    )
}

fn packing_definitions<F: Field>(hashes: &Compact) -> Vec<CircuitInputs<F>> {
    (0..6)
        .map(|kind| CircuitInputs {
            main_width: mw(kind),
            preprocessed: Some(RowMajorMatrix::new_col(F::zero_vec(
                hashes.rows[kind].len().max(2).next_power_of_two(),
            ))),
            ..Default::default()
        })
        .chain([65536, 512, 256].into_iter().map(|len| CircuitInputs {
            main_width: 1,
            preprocessed: Some(RowMajorMatrix::new_col(F::zero_vec(len))),
            ..Default::default()
        }))
        .collect()
}

fn check_trace_packing<F: Field>() {
    for rows in [0, 1, 2, 3, 4095, 4096, 4097] {
        let (hashes, prepared) = packing_fixture(rows, &[0, 1, 2, 3, 4, 5]);
        let definitions = packing_definitions::<F>(&hashes);
        for kind in 0..9 {
            let expected = serial_trace(&hashes, &prepared, &definitions, kind, false);
            let actual = hashes.trace(&prepared, &definitions, kind);
            assert_eq!(actual, expected, "rows={rows}, kind={kind}");
        }
    }
}

#[test]
fn trace_tiles_preserve_all_rows_goldilocks() {
    check_trace_packing::<Val>();
}

#[cfg(feature = "kzg")]
#[test]
fn trace_tiles_preserve_all_rows_scalar() {
    check_trace_packing::<crate::ark_adapter::Scalar>();
}

fn check_copy_cycles<F: Field>() {
    let mut hashes = Compact {
        recipes: (0..11).map(Recipe::Constant).collect(),
        rows: [
            vec![vec![0, 1, 0, 2, 3, 4], vec![4, 5, 6, 4, 7, 1]],
            vec![vec![7, 0, 8, 4, 3, 2]],
            vec![],
            vec![vec![8, 7, 3, 0, 1, 9]],
            vec![vec![1, 8, 8]],
            vec![vec![5], vec![5], vec![9]],
        ],
        boundaries: vec![
            Boundary::Bridge {
                call: 0,
                offset: 0,
                len: 4,
                output: false,
            },
            Boundary::Bridge {
                call: 0,
                offset: 4,
                len: 4,
                output: true,
            },
            Boundary::Constant(0xdeadbeef),
        ],
        ..Default::default()
    };
    assert_eq!(hashes.copy_fixed::<F>(), occurrence_fixed::<F>(&hashes));
    for rows in [0, 1, 3, 4097] {
        hashes.rows[0] = vec![vec![0; 6]; rows];
        assert_eq!(hashes.copy_fixed::<F>(), occurrence_fixed::<F>(&hashes));
    }
    hashes.rows.iter_mut().for_each(Vec::clear);
    assert_eq!(hashes.copy_fixed::<F>(), occurrence_fixed::<F>(&hashes));
}

#[test]
fn copy_endpoints_preserve_every_fixed_cell_goldilocks() {
    check_copy_cycles::<Val>();
}

#[cfg(feature = "kzg")]
#[test]
fn copy_endpoints_preserve_every_fixed_cell_scalar() {
    check_copy_cycles::<crate::ark_adapter::Scalar>();
}

fn config() -> GoldilocksBlake3Config {
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
}

#[test]
fn copy_and_trace_construction_preserve_fresh_proof_bytes() {
    let mut builder = CircuitBuilder::<Val>::new();
    builder.enable_compact_blake3();
    let gadgets = ByteGadgets::new(&mut builder);
    let input: Vec<_> = (0..65)
        .map(|i| gadgets.input(&mut builder, &format!("byte-{i}")))
        .collect();
    let digest = blake3(&mut builder, &gadgets, &input);
    for byte in digest {
        builder.expose_public(byte.value());
    }
    let circuit = builder.finish();
    let reference_fixed = occurrence_fixed::<Val>(circuit.hashes.as_ref().unwrap());
    let compiled = circuit.lower_to_multi_stark(Val::from_u8(131)).unwrap();
    let mut witness = compiled.witness();
    for (i, byte) in input.into_iter().enumerate() {
        witness
            .set(byte.value(), Val::from_usize((i * 17 + 11) % 256))
            .unwrap();
    }
    let assignment = witness.generate().unwrap();
    let claims = compiled.claims(assignment.public_values()).unwrap();
    let claim_refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let traces = compiled.traces(&assignment).unwrap();
    let actual_definitions = compiled.circuit_inputs();
    let mut reference_definitions = actual_definitions.clone();
    let hash_start = compiled.num_circuits() - 9;
    let mut reference_traces = traces[..hash_start].to_vec();
    reference_traces.extend((0..9).map(|index| {
        serial_trace(
            compiled.circuit().hashes.as_ref().unwrap(),
            assignment.prepared_hash(compiled.circuit()).unwrap(),
            &actual_definitions[hash_start..],
            index,
            false,
        )
    }));
    assert_eq!(traces, reference_traces);
    for (t, fixed) in reference_fixed.iter().enumerate() {
        let target = reference_definitions[hash_start + t]
            .preprocessed
            .as_mut()
            .unwrap();
        let copy_width = 2 + 2 * slots(t);
        for (reference_row, row) in fixed
            .chunks_exact(fw(t))
            .zip(target.values.chunks_exact_mut(fw(t)))
        {
            row[..copy_width].copy_from_slice(&reference_row[..copy_width]);
        }
    }
    for (actual, reference) in actual_definitions.iter().zip(&reference_definitions) {
        assert_eq!(actual.preprocessed, reference.preprocessed);
    }
    let (actual, actual_key) = System::new(config(), actual_definitions);
    let (reference, reference_key) = System::new(config(), reference_definitions);
    let actual_proof = actual.prove_multiple_claims(
        &actual_key,
        &claim_refs,
        SystemWitness::from_stage_1(traces, &actual),
    );
    let reference_proof = reference.prove_multiple_claims(
        &reference_key,
        &claim_refs,
        SystemWitness::from_stage_1(reference_traces, &reference),
    );
    actual
        .verify_multiple_claims(&claim_refs, &actual_proof)
        .unwrap();
    assert_eq!(
        actual_proof.to_bytes().unwrap(),
        reference_proof.to_bytes().unwrap()
    );
}

#[test]
#[ignore = "isolated CPU compact-hash copy-cycle construction comparison"]
fn copy_endpoints_benchmark() {
    use std::{hint::black_box, time::Instant};
    let value = Value { owner: 1, index: 0 };
    let mut hashes = Compact::default();
    for _ in 0..1024 {
        hashes.add_call(HashCall {
            input: vec![value; 128],
            output: [value; 32],
        });
    }
    let occurrences: usize = hashes.rows.iter().flatten().map(Vec::len).sum();
    let mut used = vec![false; hashes.recipes.len()];
    for &word in hashes.rows.iter().flatten().flatten() {
        used[word] = true;
    }
    let nonempty_words = used.into_iter().filter(|used| *used).count();
    let expected = occurrence_fixed::<Val>(&hashes);
    let actual = hashes.copy_fixed::<Val>();
    assert!(actual == expected, "compact hash fixed cells differ");
    drop(expected);
    drop(actual);
    for iteration in 0..5 {
        let mut seconds = [0.0; 2];
        for endpoints in if iteration % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let started = Instant::now();
            let fixed = if endpoints {
                black_box(&hashes).copy_fixed::<Val>()
            } else {
                occurrence_fixed::<Val>(black_box(&hashes))
            };
            seconds[usize::from(endpoints)] = started.elapsed().as_secs_f64();
            black_box(fixed);
        }
        println!(
            "COPY_ENDPOINTS_BENCH iteration={iteration} compressions={} words={} occurrences={occurrences} removed_small_allocations={nonempty_words} occurrence_seconds={:.9} endpoint_seconds={:.9}",
            hashes.compressions,
            hashes.recipes.len(),
            seconds[0],
            seconds[1],
        );
    }
}

fn trace_packing_benchmark_for<F: Field>(field: &str, log_height: usize) {
    use std::{hint::black_box, time::Instant};
    let height = 1 << log_height;
    let kind = 1;
    let (hashes, prepared) = packing_fixture(height - 3, &[kind]);
    let definitions = packing_definitions::<F>(&hashes);
    let expected = serial_trace(&hashes, &prepared, &definitions, kind, false);
    assert_eq!(
        expected,
        serial_trace(&hashes, &prepared, &definitions, kind, true)
    );
    assert_eq!(expected, hashes.trace(&prepared, &definitions, kind));
    drop(expected);
    for (iteration, order) in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [0, 2, 1], [1, 0, 2]]
        .into_iter()
        .enumerate()
    {
        let mut seconds = [0.0; 3];
        for variant in order {
            let started = Instant::now();
            let trace = match variant {
                0 => serial_trace(black_box(&hashes), &prepared, &definitions, kind, false),
                1 => serial_trace(black_box(&hashes), &prepared, &definitions, kind, true),
                _ => black_box(&hashes).trace(&prepared, &definitions, kind),
            };
            seconds[variant] = started.elapsed().as_secs_f64();
            black_box(trace);
        }
        println!(
            "HASH_TRACE_PACKING_BENCH field={field} iteration={iteration} kind={kind} used_rows={} height={height} width={} output_bytes={} serial_generic_seconds={:.9} serial_field_seconds={:.9} parallel_field_seconds={:.9}",
            height - 3,
            mw(kind),
            height * mw(kind) * size_of::<F>(),
            seconds[0],
            seconds[1],
            seconds[2],
        );
    }
}

#[test]
#[ignore = "isolated CPU hash advice allocation and indexed row-packing comparison"]
fn trace_packing_benchmark() {
    trace_packing_benchmark_for::<Val>("goldilocks", 18);
    #[cfg(feature = "kzg")]
    trace_packing_benchmark_for::<crate::ark_adapter::Scalar>("scalar", 16);
}
