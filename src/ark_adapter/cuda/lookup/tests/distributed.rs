use super::*;
use crate::ark_adapter::cuda::lookup::distributed as device;

#[test]
#[ignore = "requires four paired CUDA GPUs and MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8"]
fn distributed_lookup_matches_cpu_coefficients_and_totals() {
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(1 << 16, b"distributed-lookup")),
        16,
    );
    for (log_n, groups, group_size, constant) in [
        (10, 1, 1, false),
        (10, 2, 2, false),
        (10, 3, 2, false),
        (16, 4, 2, false),
        (10, 5, 2, false),
        (10, 4, 2, true),
    ] {
        let n = 1 << log_n;
        let used_rows = n / 2 + 13;
        let count = groups * group_size - usize::from(group_size > 1);
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
                    if constant {
                        [Scalar::ZERO, Scalar::ZERO, Scalar::from_u8(3)]
                    } else if row >= used_rows {
                        [Scalar::ZERO; 3]
                    } else {
                        [
                            Scalar::from_usize(row % 7),
                            Scalar::from_usize(row % 2),
                            Scalar::from_usize(row + 3),
                        ]
                    }
                })
                .collect(),
            3,
        );
        let definition = CircuitInputs {
            main_width: 3,
            preprocessed: Some(fixed),
            lookups: (0..count)
                .map(|i| {
                    let multiplicity = (Expr::main(2) + Expr::IsFirstRow + Expr::IsLastRow
                        - Expr::IsTransition * Expr::preprocessed_next(0))
                        * Expr::constant(Scalar::from_usize(i + 1));
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
        assert_eq!(circuit.stage_2_width, groups);
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
        let options = device::Options {
            tile_rows: 256,
            force_host: false,
            budget_limit: usize::MAX,
            reject_after_lease: false,
        };
        for (beta, gamma) in [(0, 0), (11, 7)] {
            let beta = Scalar::from_u8(beta);
            let gamma = Scalar::from_u8(gamma);
            let (traces, totals) = LookupValues::stage_2_traces(
                core::slice::from_ref(&values),
                &[group_size],
                beta,
                &gamma,
                Scalar::ZERO,
            );
            let trace = &traces[0];
            if log_n == 16 && beta != Scalar::ZERO {
                let mut prefixes: Vec<_> = (0..4)
                    .map(|quarter| trace.values[quarter * (n / 4) * groups])
                    .collect();
                prefixes.push(totals[0]);
                let quarter_totals: Vec<_> =
                    prefixes.windows(2).map(|pair| pair[1] - pair[0]).collect();
                assert!(quarter_totals.iter().all(|value| *value != Scalar::ZERO));
                assert!(quarter_totals.windows(2).all(|pair| pair[0] != pair[1]));
                println!(
                    "distributed_lookup_fixture quarter_totals={quarter_totals:?} used_rows={used_rows}"
                );
            }
            let fft = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
            let expected: Vec<Vec<_>> = (0..groups)
                .map(|column| {
                    let mut coefficients: Vec<_> = (0..n)
                        .map(|row| trace.values[row * groups + column].0)
                        .collect();
                    fft.ifft_in_place(&mut coefficients);
                    coefficients
                })
                .collect();
            for rejected in [
                device::Options {
                    budget_limit: 0,
                    ..options
                },
                device::Options {
                    reject_after_lease: true,
                    ..options
                },
            ] {
                assert!(coefficients_impl(&input, beta, gamma, Some(rejected)).is_none());
                assert!(Devices::get().busy.lock().unwrap().iter().all(|busy| !busy));
            }
            for force_host in [false, true] {
                let (actual, total) = coefficients_impl(
                    &input,
                    beta,
                    gamma,
                    Some(device::Options {
                        force_host,
                        ..options
                    }),
                )
                .expect("distributed lookup fixture must pass admission");
                assert_eq!(
                    total, totals[0],
                    "total log={log_n} groups={groups} constant={constant} host={force_host}"
                );
                assert!(
                    actual == expected,
                    "coefficients log={log_n} groups={groups} constant={constant} host={force_host}"
                );
                assert!(Devices::get().busy.lock().unwrap().iter().all(|busy| !busy));
            }
        }
    }
}

#[test]
#[ignore = "requires distributed lookup force mode, four paired CUDA GPUs and 8 GiB coefficient cap"]
fn distributed_lookup_preserves_accumulators_and_proof_bytes() {
    use crate::system::SystemWitness;
    assert!(
        device::enabled(false),
        "set distributed lookup mode to force"
    );
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(1 << 12, b"distributed-lookup-order")),
        8,
    );
    let definitions = vec![
        CircuitInputs {
            main_width: 1,
            preprocessed: Some(RowMajorMatrix::new_col(vec![Scalar::ONE; 1 << 11])),
            constraints: vec![Expr::main_next(0) - Expr::main(0)],
            lookups: vec![CircuitLookup::push(
                Expr::preprocessed(0),
                vec![Expr::main(0), Expr::main_next(0)],
            )],
            ..Default::default()
        },
        CircuitInputs {
            main_width: 1,
            preprocessed: Some(RowMajorMatrix::new_col(vec![Scalar::from_u8(2); 1 << 10])),
            lookups: vec![CircuitLookup::pull(
                Expr::preprocessed(0),
                vec![Expr::main(0), Expr::main_next(0)],
            )],
            ..Default::default()
        },
    ];
    let traces = vec![
        RowMajorMatrix::new_col(vec![Scalar::from_u8(7); 1 << 11]),
        RowMajorMatrix::new_col(vec![Scalar::from_u8(7); 1 << 10]),
    ];
    let (reference, key) = System::new(config.clone(), definitions.clone());
    let expected = reference.prove_multiple_claims(
        &key,
        &[],
        SystemWitness::from_stage_1(traces.clone(), &reference),
    );
    assert_ne!(expected.intermediate_accumulators[0], Scalar::ZERO);
    let streamed_config = config.clone().with_streaming_lookups();
    let (streamed, streamed_key) = System::new(streamed_config.clone(), definitions);
    let actual = streamed.prove_multiple_claims(
        &streamed_key,
        &[],
        SystemWitness::from_stage_1(traces.clone(), &streamed),
    );
    assert_eq!(actual.to_bytes().unwrap(), expected.to_bytes().unwrap());
    reference.verify_multiple_claims(&[], &actual).unwrap();

    let (_, main) = config.pcs().commit(
        traces
            .into_iter()
            .zip([1 << 11, 1 << 10])
            .map(|(values, rows)| (config.pcs().natural_domain_for_degree(rows), values))
            .collect(),
    );
    let shapes = [
        LookupValues::shape_only(1 << 11, &[2]),
        LookupValues::shape_only(1 << 10, &[2]),
    ];
    let inputs: Vec<_> = (0..2)
        .map(|i| LookupCommitInput {
            circuit: &reference.circuits[i],
            lookup_values: &shapes[i],
            preprocessed: Some((key.preprocessed_data.as_ref().unwrap(), i)),
            stage_1: (&main, i),
        })
        .collect();
    let beta = Scalar::from_u8(23);
    let gamma = Scalar::from_u8(31);
    let initial = Scalar::from_u8(13);
    let local = Scalar::from_usize(1 << 11)
        * (beta + Scalar::from_u8(7) + gamma * Scalar::from_u8(7)).inverse();
    let (_, _, accumulators) = streamed_config
        .accelerated_lookup_commit(&inputs, beta, gamma, initial)
        .unwrap();
    assert_eq!(accumulators, vec![initial + local, initial]);
    println!(
        "distributed_lookup_proof bytes={} nonzero_initial_accumulator=true",
        actual.to_bytes().unwrap().len()
    );
}

#[test]
#[ignore = "requires four paired CUDA GPUs, profiling, and an 8 GiB coefficient cap"]
fn distributed_lookup_uses_partner_resident_coefficients() {
    let log_n = 17;
    let n = 1 << log_n;
    let count = (n / 2) + 13;
    let coefficients: Vec<_> = (0..count)
        .map(|row| Fr::from((row * 731 + 3) as u64))
        .collect();
    let options = device::Options {
        tile_rows: 256,
        force_host: false,
        budget_limit: usize::MAX,
        reject_after_lease: false,
    };
    let plan = crate::ark_adapter::cuda::distributed::Plan::new(&[0; 4], options)
        .expect("fixture requires two mutual peer pairs");
    let owner = Devices::get()
        .ids
        .iter()
        .position(|&ordinal| ordinal == plan.ordinals[1])
        .unwrap();
    let mut reservations = Vec::new();
    let resident = loop {
        assert!(reservations.len() < Devices::get().ids.len() * 2);
        let candidate = retain(&coefficients).expect("small coefficient fixture must be retained");
        if candidate.reservation.index == owner {
            break candidate;
        }
        reservations.push(candidate);
    };
    assert_ne!(
        plan.ordinals[0],
        Devices::get().ids[resident.reservation.index]
    );
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(2, b"peer-coefficients-unused-srs")),
        2,
    );
    let (system, _) = System::new(
        config,
        [CircuitInputs {
            main_width: 1,
            lookups: vec![CircuitLookup::push(
                Expr::constant(Scalar::ONE) + Expr::IsFirstRow + Expr::IsLastRow,
                vec![Expr::main(0), Expr::main_next(0)],
            )],
            ..Default::default()
        }],
    );
    let circuit = &system.circuits[0];
    let program = quotient::encode_lookup(&circuit.graph, [0, 1, 0]).unwrap();
    let beta = Scalar::from_u8(23);
    let gamma = Scalar::from_u8(31);
    let parameters = Parameters {
        beta: beta.0.0.0,
        gamma: gamma.0.0.0,
        trace_log: log_n,
        slots: program.slots as u32,
        group_size: 1,
        blocks: 256,
    };
    let fft = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
    let mut values = coefficients.clone();
    values.resize(n, Fr::ZERO);
    fft.fft_in_place(&mut values);
    let main = RowMajorMatrix::new_col(values.into_iter().map(Scalar).collect());
    let lookup_values = crate::system::compute_lookup_values_range(circuit, &main, None, 0..n);
    let (traces, totals) =
        LookupValues::stage_2_traces(&[lookup_values], &[1], beta, &gamma, Scalar::ZERO);
    let mut expected: Vec<_> = traces[0].values.iter().map(|value| value.0).collect();
    fft.ifft_in_place(&mut expected);
    let columns = [(&coefficients[..], Some(resident.as_ref()), false)];
    for force_host in [false, true] {
        eprintln!(
            "peer_coefficient_fixture force_host={force_host} coefficient_bytes={} evaluation_owner={} resident_owner={} padded_rows={n}",
            count * size_of::<Fr>(),
            plan.ordinals[0],
            plan.ordinals[1]
        );
        let (actual, total) = device::coefficients(
            &parameters,
            &program,
            &columns,
            1,
            device::Options {
                force_host,
                ..options
            },
        )
        .expect("peer coefficient fixture must pass admission");
        assert_eq!(actual, vec![expected.clone()]);
        assert_eq!(total, totals[0]);
        assert!(Devices::get().busy.lock().unwrap().iter().all(|busy| !busy));
    }
    drop(resident);
    drop(reservations);
    assert!(
        Devices::get()
            .resident_bytes
            .lock()
            .unwrap()
            .iter()
            .all(|&bytes| bytes == 0)
    );
}
