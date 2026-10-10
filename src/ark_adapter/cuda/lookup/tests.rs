use super::*;
use crate::ark_adapter::Srs;
use crate::config::ProofConfig;
use crate::expr::Expr;
use crate::lookup::{Lookup as CircuitLookup, LookupValues};
use crate::system::{CircuitInputs, System};
use crate::traits::{EvaluationDomain, Field as _, Pcs};
use ark_poly::{EvaluationDomain as ArkDomain, Radix2EvaluationDomain};
use p3_matrix::dense::RowMajorMatrix;

mod benchmark;
mod distributed;

#[test]
#[ignore = "requires a CUDA GPU; compares grouped scans, zero messages and lookup coefficients"]
fn resident_lookup_matches_cpu_coefficients() {
    assert_eq!(size_of::<Parameters>(), 96);
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(1 << 16, b"resident-lookup")),
        16,
    );
    for (log_n, count, group_size, constant) in [
        (10, 0, 1, false),
        (10, 1, 1, false),
        (10, 3, 2, false),
        (10, 9, 8, false),
        (16, 9, 2, false),
        (16, 9, 2, true),
    ] {
        let n = 1 << log_n;
        let fixed = RowMajorMatrix::new(
            (0..n)
                .flat_map(|row| {
                    [
                        Scalar::from_usize(if constant { 1 } else { row % 3 }),
                        Scalar::ZERO,
                    ]
                })
                .collect(),
            2,
        );
        let main_values = RowMajorMatrix::new(
            (0..n)
                .flat_map(|row| {
                    [
                        Scalar::from_usize(if constant { 0 } else { row % 7 }),
                        Scalar::from_usize(if constant { 0 } else { row % 2 }),
                        Scalar::from_u8(3),
                    ]
                })
                .collect(),
            3,
        );
        let definition = CircuitInputs {
            main_width: 3,
            preprocessed: Some(fixed),
            lookups: (0..count)
                .map(|i| {
                    let multiplicity = Expr::main(2) + Expr::IsFirstRow + Expr::IsLastRow
                        - Expr::IsTransition * Expr::preprocessed_next(0);
                    let args = match i % 3 {
                        0 => vec![],
                        1 => vec![Expr::main(0)],
                        _ => vec![
                            Expr::main_next(1),
                            Expr::preprocessed_next(1),
                            Expr::main(0),
                        ],
                    };
                    if i % 2 == 0 {
                        CircuitLookup::push(multiplicity, args)
                    } else {
                        CircuitLookup::pull(multiplicity, args)
                    }
                })
                .collect(),
            lookup_group_size: group_size,
            ..Default::default()
        };
        let (system, key) = System::new(config.clone(), [definition]);
        let circuit = &system.circuits[0];
        let domain = config.pcs().natural_domain_for_degree(n);
        let (_, main) = config.pcs().commit(vec![(domain, main_values.clone())]);
        let values = crate::system::compute_lookup_values_range(
            circuit,
            &main_values,
            circuit.preprocessed.as_ref(),
            0..n,
        );
        let input: LookupCommitInput<'_, KzgConfig> = LookupCommitInput {
            circuit,
            lookup_values: &values,
            preprocessed: key.preprocessed_data.as_ref().map(|data| (data, 0)),
            stage_1: (&main, 0),
        };
        for beta in [Scalar::ZERO, Scalar::from_u8(11)] {
            let gamma = Scalar::from_u8(7);
            let (traces, totals) = LookupValues::stage_2_traces(
                core::slice::from_ref(&values),
                &[group_size],
                beta,
                &gamma,
                Scalar::ZERO,
            );
            let trace = &traces[0];
            let fft = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
            let expected: Vec<Vec<_>> = (0..trace.width)
                .map(|column| {
                    let mut coefficients: Vec<_> = (0..n)
                        .map(|row| trace.values[row * trace.width + column].0)
                        .collect();
                    fft.ifft_in_place(&mut coefficients);
                    coefficients
                })
                .collect();
            let (actual, total) = coefficients(&input, beta, gamma).expect("fixture fits one GPU");
            assert_eq!(
                total, totals[0],
                "total log={log_n} count={count} group={group_size} constant={constant} beta={beta:?}"
            );
            assert!(
                actual == expected,
                "coefficients log={log_n} count={count} group={group_size} constant={constant} beta={beta:?}"
            );
        }
    }
}

#[test]
#[ignore = "reads a saved trace and times one lookup; requires MULTI_STARK_KZG_QUOTIENT_CHECKPOINTS and a CUDA GPU"]
fn resident_lookup_checkpoint_benchmark() {
    use crate::ark_adapter::pcs::{KzgCommitment, KzgProverData};
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
    for data in [&fixed, &main] {
        data.matrices[0].resident_columns();
    }
    // Parameter initialization is outside both timed regions.
    let mut warmup = vec![Fr::ONE; 1024];
    fft(&mut warmup, false, Fr::ONE);
    println!(
        "lookup_checkpoint partition={index} rows={n} widths={},{},{} lookup_count={} prefix_nodes={} preparation_seconds={:.6}",
        fixed.matrices[0].width(),
        main.matrices[0].width(),
        circuit.stage_2_width,
        circuit.graph.lookups.len(),
        circuit.graph.lookup_prefix_len,
        started.elapsed().as_secs_f64()
    );
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(2, b"lookup-diagnostic-unused-srs")),
        2,
    );
    let beta = Scalar::from_u8(23);
    let gamma = Scalar::from_u8(31);
    let started = Instant::now();
    let (trace_values, expected_total) = {
        let fixed_values = config.pcs().get_evaluations_on_domain(&fixed, 0, trace);
        let main_values = config.pcs().get_evaluations_on_domain(&main, 0, trace);
        crate::ark_adapter::config::lookup_trace(
            &circuit,
            &main_values,
            Some(&fixed_values),
            beta,
            gamma,
            1 << 16,
        )
    };
    let expected = crate::ark_adapter::pcs::KzgPcs::interpolate_columns(trace, &trace_values);
    drop(trace_values);
    let cpu_seconds = started.elapsed().as_secs_f64();
    let shape = LookupValues::shape_only(
        n,
        &circuit
            .graph
            .lookups
            .iter()
            .map(|lookup| lookup.args.len())
            .collect::<Vec<_>>(),
    );
    let input: LookupCommitInput<'_, KzgConfig> = LookupCommitInput {
        circuit: &circuit,
        lookup_values: &shape,
        preprocessed: Some((&fixed, 0)),
        stage_1: (&main, 0),
    };
    let started = Instant::now();
    let (actual, actual_total) =
        coefficients(&input, beta, gamma).expect("selected trace fits on one GPU");
    let gpu_seconds = started.elapsed().as_secs_f64();
    assert_eq!(actual_total, expected_total);
    for (column, (actual, (expected, _, _))) in
        actual.iter().zip(expected.cuda_columns()).enumerate()
    {
        assert!(
            actual.as_slice() == expected,
            "lookup coefficient column {column}"
        );
    }
    println!(
        "lookup_checkpoint cpu_trace_seconds={cpu_seconds:.6} resident_seconds={gpu_seconds:.6} speedup={:.3} coefficient_and_total_parity=true",
        cpu_seconds / gpu_seconds
    );
}
