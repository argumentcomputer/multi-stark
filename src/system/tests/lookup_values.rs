use super::super::*;
use crate::traits::Algebra;
use crate::types::{CommitmentParameters, ExtVal, FriParameters, GoldilocksBlake3Config, Val};
use std::ops::Range;

fn serial_values<F: Field>(
    circuit: &Circuit<F>,
    trace: &RowMajorMatrix<F>,
    preprocessed: Option<&RowMajorMatrix<F>>,
    rows: Range<usize>,
) -> LookupValues<F> {
    let height = trace.height();
    assert!(rows.start <= rows.end && rows.end <= height);
    let widths: Vec<_> = circuit.graph.lookups.iter().map(|l| l.args.len()).collect();
    let mut builder = LookupValues::builder(rows.len(), &widths);
    if height == 0 || widths.is_empty() {
        return builder.finish();
    }
    let mut buf = Vec::new();
    let mut args = Vec::new();
    let mut writers = builder.rows_mut();
    for (r, writer) in rows.zip(writers.iter_mut()) {
        let r_next = (r + 1) % height;
        let main_cur = trace.row_slice(r).unwrap();
        let main_next = trace.row_slice(r_next).unwrap();
        let fixed = preprocessed.map(|p| (p.row_slice(r).unwrap(), p.row_slice(r_next).unwrap()));
        let (pp_cur, pp_next): (&[F], &[F]) = match &fixed {
            Some((cur, next)) => (cur, next),
            None => (&[], &[]),
        };
        let view = VarValues {
            preprocessed: [pp_cur, pp_next],
            main: [&main_cur, &main_next],
            stage2: [&[], &[]],
            publics: &[],
            is_first_row: F::from_bool(r == 0),
            is_last_row: F::from_bool(r == height - 1),
            is_transition: F::from_bool(r != height - 1),
        };
        circuit.graph.sweep_lookup_prefix(&view, &mut buf);
        for (slot, lookup) in circuit.graph.lookups.iter().enumerate() {
            args.clear();
            args.extend(lookup.args.iter().map(|arg| buf[arg.index()]));
            writer.push(slot, buf[lookup.multiplicity.index()], &args);
        }
    }
    drop(writers);
    builder.finish()
}

fn fixture<F: Field>(height: usize, fixed: bool) -> (Circuit<F>, RowMajorMatrix<F>) {
    let trace = RowMajorMatrix::new(
        (0..height * 2).map(|i| F::from_usize(i * 13 + 7)).collect(),
        2,
    );
    let preprocessed = fixed.then(|| {
        RowMajorMatrix::new(
            (0..height * 2)
                .map(|i| F::from_usize(i * 17 + 11))
                .collect(),
            2,
        )
    });
    let spec = CircuitSpec {
        main_width: 2,
        preprocessed_width: usize::from(fixed) * 2,
        lookups: vec![
            Lookup::push(
                Expr::IsFirstRow + Expr::main(0),
                vec![
                    Expr::main_next(1),
                    if fixed {
                        Expr::preprocessed_next(1)
                    } else {
                        Expr::main_next(0)
                    },
                    Expr::main(0) * Expr::main(1)
                        + if fixed {
                            Expr::preprocessed(0)
                        } else {
                            Expr::constant(F::from_u8(19))
                        },
                ],
            ),
            Lookup::pull(Expr::IsTransition, vec![Expr::IsLastRow, Expr::main(1)]),
            Lookup::push(Expr::IsLastRow - Expr::IsFirstRow, vec![]),
        ],
        ..Default::default()
    };
    let graph = compile(
        &spec,
        &ExtensionParams {
            degree: 1,
            w: F::ONE,
            karatsuba: false,
        },
    )
    .unwrap();
    let circuit = Circuit {
        graph,
        main_width: 2,
        preprocessed,
        preprocessed_width: spec.preprocessed_width,
        preprocessed_height: if fixed { height } else { 0 },
        num_lookups: 3,
        stage_2_width: 4,
        num_publics: 8,
        lookup_group_size: 2,
        constraint_count: 4,
        max_constraint_degree: 5,
    };
    (circuit, trace)
}

fn expected_values<F: Field>(height: usize, fixed: bool, rows: Range<usize>) -> LookupValues<F> {
    if rows.is_empty() {
        return LookupValues::builder(0, &[3, 2, 0]).finish();
    }
    LookupValues::from_rows(
        rows.map(|r| {
            let next = (r + 1) % height;
            let x = F::from_usize(2 * r * 13 + 7);
            let y = F::from_usize((2 * r + 1) * 13 + 7);
            vec![
                Lookup::push(
                    F::from_bool(r == 0) + x,
                    vec![
                        F::from_usize((2 * next + 1) * 13 + 7),
                        F::from_usize(if fixed {
                            (2 * next + 1) * 17 + 11
                        } else {
                            2 * next * 13 + 7
                        }),
                        x * y + F::from_usize(if fixed { 2 * r * 17 + 11 } else { 19 }),
                    ],
                ),
                Lookup::pull(
                    F::from_bool(r + 1 != height),
                    vec![F::from_bool(r + 1 == height), y],
                ),
                Lookup::push(F::from_bool(r + 1 == height) - F::from_bool(r == 0), vec![]),
            ]
        })
        .collect(),
    )
}

fn check_row_windows<F: Field>() {
    for fixed in [false, true] {
        for height in [0, 1, 17, (1 << 14) + 1] {
            let (circuit, trace) = fixture::<F>(height, fixed);
            let ranges = [
                0..height,
                height / 3..height * 2 / 3,
                height.saturating_sub(3)..height,
                0..0,
                height / 2..height / 2,
                height..height,
            ];
            for rows in ranges {
                let actual = compute_lookup_values_range(
                    &circuit,
                    &trace,
                    circuit.preprocessed.as_ref(),
                    rows.clone(),
                );
                assert!(
                    actual == expected_values(height, fixed, rows.clone()),
                    "lookup payload mismatch: height={height}, fixed={fixed}, rows={rows:?}"
                );
            }
        }
    }
}

#[test]
fn row_windows_preserve_payload_goldilocks() {
    check_row_windows::<Val>();
}

#[cfg(feature = "kzg")]
#[test]
fn row_windows_preserve_payload_scalar() {
    check_row_windows::<crate::ark_adapter::Scalar>();
}

#[test]
fn argumentless_and_empty_lookup_layouts() {
    let (mut circuit, trace) = fixture::<Val>((1 << 14) + 1, false);
    let height = trace.height();
    circuit
        .graph
        .lookups
        .retain(|lookup| lookup.args.is_empty());
    let actual = compute_lookup_values(&circuit, &trace);
    let expected = LookupValues::from_rows(
        (0..height)
            .map(|r| {
                vec![Lookup::push(
                    Val::from_bool(r + 1 == height) - Val::from_bool(r == 0),
                    vec![],
                )]
            })
            .collect(),
    );
    assert_eq!(actual, expected);
    circuit.graph.lookups.clear();
    assert_eq!(
        compute_lookup_values(&circuit, &trace),
        LookupValues::builder(height, &[]).finish()
    );
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
fn witness_circuit_order_and_lookup_totals() {
    let (circuits, traces): (Vec<_>, Vec<_>) = [65, 0, 513]
        .into_iter()
        .map(|height| fixture::<Val>(height, true))
        .unzip();
    let system = System {
        config: config(),
        circuits,
        preprocessed_commit: None,
        preprocessed_indices: vec![None; 3],
    };
    let expected: Vec<_> = traces
        .iter()
        .zip(&system.circuits)
        .map(|(trace, circuit)| {
            serial_values(
                circuit,
                trace,
                circuit.preprocessed.as_ref(),
                0..trace.height(),
            )
        })
        .collect();
    let witness = SystemWitness::from_stage_1(traces.clone(), &system);
    assert_eq!(witness.lookups, expected);
    for (actual, expected) in witness.traces.iter().zip(&traces) {
        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(actual.values, expected.values);
    }
    let challenges = [
        ExtVal::from_u8(29),
        ExtVal::from_u8(31),
        ExtVal::from_u8(37),
    ];
    let (actual_traces, actual_totals) = LookupValues::stage_2_traces(
        &witness.lookups,
        &[2, 1, 3],
        challenges[0],
        &challenges[1],
        challenges[2],
    );
    let (expected_traces, expected_totals) = LookupValues::stage_2_traces(
        &expected,
        &[2, 1, 3],
        challenges[0],
        &challenges[1],
        challenges[2],
    );
    assert_eq!(actual_totals, expected_totals);
    for (actual, expected) in actual_traces.iter().zip(&expected_traces) {
        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(actual.values, expected.values);
    }
}

#[test]
fn lookup_witness_preserves_proof_bytes() {
    let inputs = [false, true].map(|pull| CircuitInputs {
        main_width: 2,
        lookups: vec![if pull {
            Lookup::pull(
                Expr::constant(Val::ONE),
                vec![Expr::main(0), Expr::main_next(1)],
            )
        } else {
            Lookup::push(
                Expr::constant(Val::ONE),
                vec![Expr::main(0), Expr::main_next(1)],
            )
        }],
        ..Default::default()
    });
    let (system, key) = System::new(config(), inputs);
    let (_, trace) = fixture::<Val>(64, false);
    let traces = vec![trace.clone(), trace];
    let serial_lookups = traces
        .iter()
        .zip(&system.circuits)
        .map(|(trace, circuit)| serial_values(circuit, trace, None, 0..trace.height()))
        .collect();
    let parallel_witness = SystemWitness::from_stage_1(traces.clone(), &system);
    let no_claims: &[&[Val]] = &[];
    let serial_proof = system.prove_multiple_claims(
        &key,
        no_claims,
        SystemWitness {
            traces,
            lookups: serial_lookups,
        },
    );
    let parallel_proof = system.prove_multiple_claims(&key, no_claims, parallel_witness);
    system
        .verify_multiple_claims(no_claims, &parallel_proof)
        .unwrap();
    assert_eq!(
        bincode::serde::encode_to_vec(&parallel_proof, bincode::config::standard()).unwrap(),
        bincode::serde::encode_to_vec(&serial_proof, bincode::config::standard()).unwrap()
    );
}

#[test]
#[ignore = "requires MULTI_STARK_LOOKUP_VK; isolated saved-graph lookup row comparison"]
fn saved_lookup_rows_benchmark() {
    use std::{hint::black_box, time::Instant};
    let path = std::env::var("MULTI_STARK_LOOKUP_VK").unwrap();
    let key =
        crate::plonkish::verifier::VerifierKey::from_bytes(&std::fs::read(path).unwrap()).unwrap();
    let circuit = &key.system().circuits[0];
    let height = 1 << 20;
    let matrix = |width: usize, seed: usize| {
        RowMajorMatrix::new(
            (0..height * width)
                .map(|i| Val::from_usize((i * 17 + seed) % 1009))
                .collect(),
            width,
        )
    };
    let trace = matrix(circuit.main_width, 23);
    let fixed = (circuit.preprocessed_width != 0).then(|| matrix(circuit.preprocessed_width, 41));
    let expected = serial_values(circuit, &trace, fixed.as_ref(), 0..height);
    let actual = compute_lookup_values_range(circuit, &trace, fixed.as_ref(), 0..height);
    assert!(actual == expected, "saved graph lookup payload mismatch");
    drop(actual);
    drop(expected);
    for iteration in 0..5 {
        let mut seconds = [0.0; 2];
        for parallel in if iteration % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let started = Instant::now();
            let values = if parallel {
                compute_lookup_values_range(black_box(circuit), &trace, fixed.as_ref(), 0..height)
            } else {
                serial_values(black_box(circuit), &trace, fixed.as_ref(), 0..height)
            };
            seconds[usize::from(parallel)] = started.elapsed().as_secs_f64();
            black_box(values);
        }
        println!(
            "LOOKUP_ROWS_BENCH iteration={iteration} height={height} main_width={} fixed_width={} lookups={} graph_nodes={} lookup_nodes={} removed_writer_bytes={} serial_seconds={:.9} parallel_seconds={:.9}",
            circuit.main_width,
            circuit.preprocessed_width,
            circuit.num_lookups,
            circuit.graph.nodes.len(),
            circuit.graph.lookup_prefix_len,
            height * size_of::<crate::lookup::LookupRowMut<'_, Val>>(),
            seconds[0],
            seconds[1],
        );
    }
}
