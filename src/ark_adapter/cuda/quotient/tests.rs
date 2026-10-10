use super::*;
use crate::ark_adapter::{Srs, domain::Radix2Coset};
use crate::expr::Expr;
use crate::lookup::Lookup as CircuitLookup;
use crate::system::{CircuitInputs, System};
use crate::traits::{EvaluationDomain, Field as _, Pcs};
use ark_poly::{EvaluationDomain as ArkDomain, Radix2EvaluationDomain};
use p3_matrix::dense::RowMajorMatrix;

mod benchmark;

fn evaluate(matrix: &CommittedMatrix, domain: Radix2Coset) -> RowMajorMatrix<Scalar> {
    let fft = Radix2EvaluationDomain::<Fr>::new(domain.size())
        .unwrap()
        .get_coset(domain.shift.0)
        .unwrap();
    let columns: Vec<_> = matrix
        .cuda_columns()
        .iter()
        .map(|(column, _, _)| {
            let mut values = column.to_vec();
            values.resize(domain.size(), Fr::ZERO);
            fft.fft_in_place(&mut values);
            values
        })
        .collect();
    RowMajorMatrix::new(
        (0..domain.size())
            .flat_map(|row| columns.iter().map(move |column| Scalar(column[row])))
            .collect(),
        columns.len(),
    )
}

#[test]
fn quotient_instruction_layout_and_liveness() {
    assert_eq!(size_of::<Instruction>(), 64);
    assert_eq!(size_of::<Column>(), 96);
    assert_eq!(std::mem::offset_of!(Column, constant), 32);
    assert_eq!(size_of::<Parameters>(), 192);
    let graph = ConstraintGraph {
        nodes: vec![
            Node::Public(0),
            Node::Neg(crate::graph::NodeId(0)),
            Node::Neg(crate::graph::NodeId(1)),
            Node::Neg(crate::graph::NodeId(2)),
        ],
        degrees: vec![0; 4],
        zeros: vec![crate::graph::NodeId(3)],
        lookups: vec![],
        lookup_prefix_len: 0,
        max_constraint_degree: 0,
    };
    let program = encode(&graph, [0; 3], 1).unwrap();
    assert_eq!(program.slots, 2);
    assert_eq!(program.nodes[2].out, program.nodes[0].out);
    assert!(encode(&graph, [0; 3], 0).is_none());
    assert!(memory_bound(&program, usize::MAX, 4, 1, 1, 4).is_none());
}

#[test]
#[ignore = "requires a CUDA GPU; compares device coefficients with the portable CPU evaluator"]
fn resident_quotient_matches_cpu_coefficients() {
    use crate::config::ProofConfig;
    if !super::super::enabled() {
        return;
    }
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(1 << 16, b"resident-quotient")),
        8,
    );
    for (log_n, lookups, group_size, empty_graph, constant) in [
        (10, 0, 1, true, false),
        (10, 0, 1, false, false),
        (10, 1, 1, false, false),
        (10, 3, 2, false, false),
        (10, 3, 3, false, false),
        (10, 5, 5, false, false),
        (10, 9, 8, false, false),
        (16, 5, 5, false, false),
        (16, 5, 5, false, true),
    ] {
        let n = 1 << log_n;
        let fixed = (!empty_graph).then(|| {
            RowMajorMatrix::new(
                (0..n)
                    .flat_map(|row| {
                        [
                            Scalar::from_usize(if constant { 3 } else { row % 37 }),
                            Scalar::from_u8(7),
                        ]
                    })
                    .collect(),
                2,
            )
        });
        let definition = CircuitInputs {
            main_width: 3,
            preprocessed: fixed,
            constraints: if empty_graph {
                vec![]
            } else {
                vec![
                    Expr::IsFirstRow * Expr::main(0) + Expr::IsLastRow * Expr::main_next(1),
                    Expr::IsTransition * (Expr::preprocessed_next(0) - Expr::main_next(0)),
                    -(Expr::main(0) * Expr::main(1))
                        + Expr::public(0)
                        + Expr::constant(Scalar::from_u8(17)),
                    Expr::preprocessed(1) * Expr::main_next(2),
                ]
            },
            ext_constraints: if empty_graph {
                vec![]
            } else {
                vec![
                    crate::expr::ExtExpr::stage2(0, 1, RowOffset::Next)
                        - crate::expr::ExtExpr::stage2(0, 1, RowOffset::Current),
                ]
            },
            lookups: (0..lookups)
                .map(|i| {
                    CircuitLookup::push(
                        Expr::preprocessed(0) - Expr::constant(Scalar::from_usize(i)),
                        if i % 3 == 0 {
                            vec![]
                        } else {
                            vec![
                                Expr::main_next(0),
                                Expr::preprocessed_next(1),
                                Expr::main(2),
                            ]
                        },
                    )
                })
                .collect(),
            lookup_group_size: group_size,
            ..Default::default()
        };
        let (system, key) = System::new(config.clone(), [definition]);
        let circuit = &system.circuits[0];
        let trace = config.pcs().natural_domain_for_degree(n);
        let quotient = trace.create_disjoint_domain(n * circuit.quotient_degree());
        let (_, main) = config.pcs().commit(vec![(
            trace,
            RowMajorMatrix::new(
                (0..n)
                    .flat_map(|row| {
                        let row = if constant { 0 } else { row };
                        [
                            Scalar::from_usize(row * 11 + 1),
                            Scalar::from_usize(row % 23),
                            Scalar::from_u8(13),
                        ]
                    })
                    .collect(),
                3,
            ),
        )]);
        let (_, stage2) = config.pcs().commit(vec![(
            trace,
            RowMajorMatrix::new(
                (0..n)
                    .flat_map(|row| {
                        let row = if constant { 0 } else { row };
                        (0..circuit.stage_2_width).map(move |column| {
                            Scalar::from_usize(if column % 2 == 0 {
                                row * 5 + column + 9
                            } else {
                                19
                            })
                        })
                    })
                    .collect(),
                circuit.stage_2_width,
            ),
        )]);
        let mut input: QuotientCommitInput<'_, KzgConfig> = QuotientCommitInput {
            circuit,
            lookup_publics: [23, 31, 5, 47].map(Scalar::from_u8).to_vec(),
            trace_domain: trace,
            quotient_domain: quotient,
            preprocessed: key.preprocessed_data.as_ref().map(|data| (data, 0)),
            stage_1: (&main, 0),
            stage_2: (&stage2, 0),
            constraint_count: circuit.constraint_count(),
        };
        let alpha = Scalar::from_u8(41);
        let fixed_values = input
            .preprocessed
            .map(|(data, index)| evaluate(&data.matrices[index], quotient));
        let main_values = evaluate(&main.matrices[0], quotient);
        let stage2_values = evaluate(&stage2.matrices[0], quotient);
        let expected_values = crate::prover::quotient_values::<KzgConfig>(
            circuit,
            &input.lookup_publics,
            trace,
            quotient,
            &fixed_values,
            &main_values,
            &stage2_values,
            alpha,
            input.constraint_count,
        );
        let mut expected: Vec<_> = expected_values.into_iter().map(|value| value.0).collect();
        Radix2EvaluationDomain::<Fr>::new(quotient.size())
            .unwrap()
            .get_coset(quotient.shift.0)
            .unwrap()
            .ifft_in_place(&mut expected);
        let actual = coefficients(&input, alpha).expect("small fixture must fit a GPU");
        assert_eq!(
            actual.concat(),
            expected,
            "lookups={lookups}, group={group_size}, empty={empty_graph}"
        );
        input.trace_domain.shift = Scalar::from_u8(7);
        assert!(coefficients(&input, alpha).is_none());
        input.trace_domain = trace;
        input.quotient_domain.shift = Scalar::ONE;
        assert!(coefficients(&input, alpha).is_none());
        input.quotient_domain = quotient;
        input.quotient_domain.log_size = 29;
        assert!(coefficients(&input, alpha).is_none());
        input.quotient_domain = quotient;
        input.lookup_publics.pop();
        assert!(coefficients(&input, alpha).is_none());
    }
}

#[test]
#[ignore = "requires four paired CUDA GPUs and MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8"]
fn distributed_quotient_matches_cpu_coefficients() {
    use crate::config::ProofConfig;
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(1 << 16, b"distributed-quotient")),
        2,
    );
    for (log_n, constant) in [(10, false), (16, false), (10, true)] {
        let n = 1 << log_n;
        let fixed = RowMajorMatrix::new(
            (0..n)
                .flat_map(|row| {
                    [
                        Scalar::from_usize(if constant { 3 } else { row % 37 }),
                        Scalar::from_u8(7),
                    ]
                })
                .collect(),
            2,
        );
        let (system, key) = System::new(
            config.clone(),
            [CircuitInputs {
                main_width: 3,
                preprocessed: Some(fixed),
                constraints: vec![
                    Expr::IsFirstRow * Expr::main(0) + Expr::IsLastRow * Expr::main_next(1),
                    Expr::IsTransition * (Expr::preprocessed_next(0) - Expr::main_next(0)),
                    -(Expr::main(0) * Expr::main_next(1)) + Expr::public(0),
                    Expr::main(0) * Expr::main(1) * Expr::main_next(2),
                ],
                ext_constraints: vec![
                    crate::expr::ExtExpr::stage2(0, 1, RowOffset::Next)
                        - crate::expr::ExtExpr::stage2(0, 1, RowOffset::Current),
                ],
                lookups: (0..3)
                    .map(|i| {
                        CircuitLookup::push(
                            Expr::preprocessed(0) - Expr::constant(Scalar::from_usize(i)),
                            if i == 0 {
                                vec![]
                            } else {
                                vec![Expr::main_next(0), Expr::preprocessed_next(1)]
                            },
                        )
                    })
                    .collect(),
                lookup_group_size: 2,
                ..Default::default()
            }],
        );
        let circuit = &system.circuits[0];
        assert_eq!(circuit.quotient_degree(), 2);
        let trace = config.pcs().natural_domain_for_degree(n);
        let quotient = trace.create_disjoint_domain(n * 2);
        let (_, main) = config.pcs().commit(vec![(
            trace,
            RowMajorMatrix::new(
                (0..n)
                    .flat_map(|row| {
                        let row = if constant { 0 } else { row };
                        [
                            Scalar::from_usize(row * 11 + 1),
                            Scalar::from_usize(row % 23),
                            Scalar::from_u8(13),
                        ]
                    })
                    .collect(),
                3,
            ),
        )]);
        let (_, stage2) = config.pcs().commit(vec![(
            trace,
            RowMajorMatrix::new(
                (0..n)
                    .flat_map(|row| {
                        (0..circuit.stage_2_width).map(move |column| {
                            Scalar::from_usize(if constant { 19 } else { row * 5 + column + 9 })
                        })
                    })
                    .collect(),
                circuit.stage_2_width,
            ),
        )]);
        let input = QuotientCommitInput::<KzgConfig> {
            circuit,
            lookup_publics: [23, 31, 5, 47].map(Scalar::from_u8).to_vec(),
            trace_domain: trace,
            quotient_domain: quotient,
            preprocessed: key.preprocessed_data.as_ref().map(|data| (data, 0)),
            stage_1: (&main, 0),
            stage_2: (&stage2, 0),
            constraint_count: circuit.constraint_count(),
        };
        let fixed_values = input
            .preprocessed
            .map(|(data, index)| evaluate(&data.matrices[index], quotient));
        let main_values = evaluate(&main.matrices[0], quotient);
        let stage2_values = evaluate(&stage2.matrices[0], quotient);
        let alpha = Scalar::from_u8(41);
        let mut expected: Vec<_> = crate::prover::quotient_values::<KzgConfig>(
            circuit,
            &input.lookup_publics,
            trace,
            quotient,
            &fixed_values,
            &main_values,
            &stage2_values,
            alpha,
            input.constraint_count,
        )
        .into_iter()
        .map(|value| value.0)
        .collect();
        Radix2EvaluationDomain::<Fr>::new(quotient.size())
            .unwrap()
            .get_coset(quotient.shift.0)
            .unwrap()
            .ifft_in_place(&mut expected);
        let options = distributed::Options {
            tile_rows: 256,
            force_host: false,
            budget_limit: usize::MAX,
            reject_after_lease: false,
        };
        assert!(
            coefficients_impl(
                &input,
                alpha,
                Some(distributed::Options {
                    budget_limit: 0,
                    ..options
                })
            )
            .is_none()
        );
        assert!(
            coefficients_impl(
                &input,
                alpha,
                Some(distributed::Options {
                    reject_after_lease: true,
                    ..options
                }),
            )
            .is_none()
        );
        assert!(
            Devices::get()
                .busy
                .lock()
                .unwrap()
                .iter()
                .all(|&busy| !busy)
        );
        for force_host in [false, true] {
            let actual = coefficients_impl(
                &input,
                alpha,
                Some(distributed::Options {
                    force_host,
                    ..options
                }),
            )
            .expect("distributed fixture must pass admission");
            assert_eq!(
                actual.concat(),
                expected,
                "log_n={log_n} constant={constant} host={force_host}"
            );
            assert!(
                Devices::get()
                    .busy
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|&busy| !busy)
            );
        }
    }
}

#[test]
#[ignore = "reads a saved trace and times one quotient; requires MULTI_STARK_KZG_QUOTIENT_CHECKPOINTS and a CUDA GPU"]
fn resident_quotient_checkpoint_benchmark() {
    use crate::ark_adapter::pcs::{KzgCommitment, KzgProverData};
    use crate::config::ProofConfig;
    use std::io::BufReader;
    use std::time::Instant;
    let directory =
        std::path::PathBuf::from(std::env::var("MULTI_STARK_KZG_QUOTIENT_CHECKPOINTS").unwrap());
    let index: usize =
        std::env::var("MULTI_STARK_KZG_QUOTIENT_PARTITION").map_or(0, |s| s.parse().unwrap());
    let started = Instant::now();
    let (circuit, _): (crate::system::Circuit<Scalar>, KzgCommitment) =
        bincode::serde::decode_from_std_read(
            &mut BufReader::new(
                std::fs::File::open(directory.join(format!("setup-{index}.bin"))).unwrap(),
            ),
            bincode::config::standard(),
        )
        .unwrap();
    let load = |prefix| {
        KzgProverData::read_checkpoint(BufReader::with_capacity(
            1 << 20,
            std::fs::File::open(directory.join(format!("{prefix}-{index}.bin"))).unwrap(),
        ))
        .unwrap()
    };
    let (fixed, main) = rayon::join(|| load("fixed"), || load("main"));
    let trace = main.matrices[0].domain;
    let n = trace.size();
    let ratio = circuit.quotient_degree();
    let quotient = trace.create_disjoint_domain(n * ratio);
    let stage2 = KzgProverData::coefficient_fixture(
        trace,
        (0..circuit.stage_2_width)
            .into_par_iter()
            .map(|column| {
                let pattern: Vec<_> = (0..4096)
                    .map(|i| Fr::from(((i + 1) * 731 + column * 179 + 11) as u64))
                    .collect();
                let mut values = vec![Fr::ZERO; n];
                values
                    .par_chunks_mut(pattern.len())
                    .for_each(|chunk| chunk.copy_from_slice(&pattern[..chunk.len()]));
                values
            })
            .collect(),
    );
    for data in [&fixed, &main, &stage2] {
        data.matrices[0].resident_columns();
    }
    println!(
        "quotient_checkpoint partition={index} rows={n} ratio={ratio} widths={},{},{} nodes={} preparation_seconds={:.6} synthetic_lookup_coefficients=true",
        fixed.matrices[0].width(),
        main.matrices[0].width(),
        stage2.matrices[0].width(),
        circuit.graph.nodes.len(),
        started.elapsed().as_secs_f64()
    );
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(2, b"quotient-diagnostic-unused-srs")),
        ratio,
    );
    let input: QuotientCommitInput<'_, KzgConfig> = QuotientCommitInput {
        circuit: &circuit,
        lookup_publics: [23, 31, 5, 47].map(Scalar::from_u8).to_vec(),
        trace_domain: trace,
        quotient_domain: quotient,
        preprocessed: Some((&fixed, 0)),
        stage_1: (&main, 0),
        stage_2: (&stage2, 0),
        constraint_count: circuit.constraint_count(),
    };
    let alpha = Scalar::from_u8(41);
    let started = Instant::now();
    let values = config
        .accelerated_quotient_values(
            &circuit,
            &input.lookup_publics,
            trace,
            quotient,
            input.preprocessed,
            input.stage_1,
            input.stage_2,
            alpha,
            input.constraint_count,
        )
        .unwrap();
    let mut expected: Vec<_> = values.into_iter().map(|value| value.0).collect();
    fft(&mut expected, true, quotient.shift.0);
    let cpu_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let actual = coefficients(&input, alpha).expect("selected trace must fit on one GPU");
    let gpu_seconds = started.elapsed().as_secs_f64();
    assert!(actual.iter().flatten().eq(expected.iter()));
    println!(
        "quotient_checkpoint cpu_sweep_seconds={cpu_seconds:.6} resident_seconds={gpu_seconds:.6} speedup={:.3} coefficient_parity=true",
        cpu_seconds / gpu_seconds
    );
}
