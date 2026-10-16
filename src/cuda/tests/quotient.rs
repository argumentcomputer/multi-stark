use super::*;
use crate::expr::{CircuitSpec, Expr, ExtExpr};
use crate::graph::compile;
use crate::lookup::{Lookup, logup_constraint_count, stage2_width};
use crate::system::Circuit;
use crate::types::{ExtVal, GoldilocksBlake3Config};
use p3_field::BasedVectorSpace;
use p3_field::coset::TwoAdicMultiplicativeCoset;

#[test]
#[ignore = "requires a CUDA GPU; compares resident and spilled quotient LDEs with the CPU evaluator"]
fn mixed_quotient_matches_cpu() {
    let device = configured_device();
    let gpu = CudaDft::without_auxiliary_devices(device);
    let n = 1usize << 8;
    let ext = crate::system::extension_params::<GoldilocksBlake3Config>();
    for (lookup_count, group_size) in [(0, 1), (1, 1), (3, 2), (5, 5), (9, 8)] {
        let stage2_width = stage2_width(lookup_count, group_size, 2);
        let spec = CircuitSpec {
            main_width: 3,
            preprocessed_width: 2,
            stage2_width,
            num_publics: 8,
            constraints: vec![
                Expr::IsFirstRow * Expr::main(0) + Expr::IsLastRow * Expr::main_next(1),
                Expr::IsTransition * (Expr::preprocessed_next(0) - Expr::main_next(0)),
                -(Expr::main(0) * Expr::main(1))
                    + Expr::public(4)
                    + Expr::constant(Goldilocks::from_u8(17)),
                Expr::preprocessed(1) * Expr::main_next(2),
            ],
            ext_constraints: vec![
                ExtExpr::stage2(0, 2, RowOffset::Next) - ExtExpr::stage2(0, 2, RowOffset::Current),
            ],
            lookups: (0..lookup_count)
                .map(|index| {
                    Lookup::push(
                        Expr::preprocessed(0) - Expr::constant(Goldilocks::from_usize(index)),
                        if index % 3 == 0 {
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
        };
        let graph = compile(&spec, &ext).unwrap();
        let constraint_count =
            graph.zeros.len() + logup_constraint_count(lookup_count, group_size, 2);
        let circuit = Circuit {
            graph,
            main_width: 3,
            preprocessed: None,
            preprocessed_width: 2,
            preprocessed_height: n,
            num_lookups: lookup_count,
            stage_2_width: stage2_width,
            num_publics: 8,
            lookup_group_size: group_size,
            constraint_count,
            max_constraint_degree: 5,
        };
        for n in [16, 256] {
            check_quotient(&gpu, &circuit, n, 2, &[1, 2, 4]);
        }
    }
}

/// A circuit with neither constraints nor lookups compiles to an empty
/// graph. The device evaluators reject it, so both dispatchers must hand
/// it to the portable evaluator instead of the kernels.
fn constraint_free_circuit() -> crate::system::CircuitInputs<Goldilocks> {
    crate::system::CircuitInputs {
        main_width: 1,
        ..Default::default()
    }
}

#[test]
fn constraint_free_graph_has_no_device_evaluation() {
    let ext = crate::system::extension_params::<GoldilocksBlake3Config>();
    let x = Expr::main(0);
    let spec = |constraints| CircuitSpec {
        main_width: 1,
        preprocessed_width: 0,
        stage2_width: stage2_width(0, 1, 2),
        num_publics: 8,
        constraints,
        ext_constraints: vec![],
        lookups: vec![],
    };
    let empty = compile(&spec(vec![]), &ext).unwrap();
    let constrained = compile(&spec(vec![x.clone() * x.clone() - x]), &ext).unwrap();
    assert!(!quotient_graph_supported(&empty));
    assert!(quotient_graph_supported(&constrained));
}

#[test]
#[ignore = "requires a CUDA GPU; proves circuits without constraints or lookups through the device configuration"]
fn constraint_free_circuits_prove_through_the_portable_quotient() {
    use crate::system::{System, SystemWitness};
    let x = Expr::main(0);
    let constrained = || crate::system::CircuitInputs {
        main_width: 1,
        constraints: vec![x.clone() * x.clone() - x.clone()],
        ..Default::default()
    };
    // Alone, the empty graph is the whole batch; mixed, it sends the fused
    // commit to the per-circuit path, where the constrained circuit still
    // evaluates on the device.
    let systems: [Vec<crate::system::CircuitInputs<Goldilocks>>; 3] = [
        vec![constraint_free_circuit()],
        vec![constraint_free_circuit(), constrained()],
        vec![constrained(), constraint_free_circuit(), constrained()],
    ];
    for inputs in systems {
        let config = GoldilocksBlake3Config::with_device(
            COMMITMENT_PARAMETERS,
            FRI_PARAMETERS,
            Some(configured_device()),
        );
        let circuit_count = inputs.len();
        let (system, key) = System::new(config, inputs);
        let witness = SystemWitness::from_stage_1(
            vec![RowMajorMatrix::new_col(vec![Goldilocks::ONE; 4]); circuit_count],
            &system,
        );
        let no_claims = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness);
        system
            .verify_multiple_claims(no_claims, &proof)
            .unwrap_or_else(|error| panic!("{circuit_count} circuits: {error:?}"));
    }
}

const COMMITMENT_PARAMETERS: crate::types::CommitmentParameters =
    crate::types::CommitmentParameters {
        log_blowup: 1,
        cap_height: 0,
    };

const FRI_PARAMETERS: crate::types::FriParameters = crate::types::FriParameters {
    log_final_poly_len: 0,
    max_log_arity: 1,
    num_queries: 8,
    commit_proof_of_work_bits: 0,
    query_proof_of_work_bits: 0,
};

fn assert_matrix_eq(
    actual: &RowMajorMatrix<Goldilocks>,
    expected: &RowMajorMatrix<Goldilocks>,
    label: &str,
) {
    assert_eq!(actual.dimensions(), expected.dimensions(), "{label}");
    if let Some(index) = actual
        .values
        .iter()
        .zip(&expected.values)
        .position(|(a, e)| a != e)
    {
        panic!(
            "{label}: first mismatch at row={}, column={}: actual={}, expected={}",
            index / actual.width(),
            index % actual.width(),
            actual.values[index],
            expected.values[index]
        );
    }
}

fn check_quotient(
    gpu: &CudaDft,
    circuit: &Circuit<Goldilocks>,
    n: usize,
    log_blowup: usize,
    quotient_degrees: &[usize],
) {
    let device = gpu.device_id;
    let cpu = Radix2DitParallel::<Goldilocks>::default();
    let ext = crate::system::extension_params::<GoldilocksBlake3Config>();
    let trace = TwoAdicMultiplicativeCoset::new(Goldilocks::ONE, n.ilog2() as usize).unwrap();
    let lookup_count = circuit.num_lookups;
    let group_size = circuit.lookup_group_size;
    let constraint_count = circuit.constraint_count();
    let matrices: Vec<_> = [
        circuit.preprocessed_width.max(1),
        circuit.main_width.max(1),
        circuit.stage_2_width,
    ]
    .into_iter()
    .enumerate()
    .map(|(source, width)| {
        let trace = RowMajorMatrix::new(
            (0..n)
                .flat_map(|row| {
                    (0..width).map(move |column| {
                        Goldilocks::from_usize(
                            (row * row + row * 17 + column * 23 + source * 41 + 11) % 1009,
                        )
                    })
                })
                .collect(),
            width,
        );
        cpu.coset_lde_batch(trace, log_blowup, Goldilocks::GENERATOR)
            .bit_reverse_rows()
            .to_row_major_matrix()
    })
    .collect();
    let resident: Vec<_> = matrices
        .iter()
        .map(|matrix| CudaLde::from_row_major_matrix(device, matrix))
        .collect();
    for full_field in [false, true] {
        let mut rng = SmallRng::seed_from_u64(0x71756f7469656e74);
        let mut scalar = |small| {
            if full_field {
                rng.random::<Goldilocks>()
            } else {
                Goldilocks::from_u8(small)
            }
        };
        let publics = [23, 31, 5, 47, 59, 61, 67, 71].map(&mut scalar);
        let alpha = ExtVal::from_basis_coefficients_slice(&[scalar(41), scalar(43)]).unwrap();
        let powers: Vec<_> = alpha.powers().take(constraint_count).collect();
        let alpha_flat: Vec<_> = (0..2)
            .flat_map(|coordinate| {
                powers
                    .iter()
                    .rev()
                    .map(move |power| power.as_basis_coefficients_slice()[coordinate])
            })
            .collect();
        let normalization = (Goldilocks::from_usize(n) * trace.subgroup_generator()).inverse();
        let delta = [
            (publics[6] - publics[4]) * normalization,
            (publics[7] - publics[5]) * normalization,
        ];
        for &quotient_degree in quotient_degrees {
            let quotient_size = n * quotient_degree;
            let quotient = TwoAdicMultiplicativeCoset::new(
                Goldilocks::GENERATOR,
                quotient_size.ilog2() as usize,
            )
            .unwrap();
            let selectors = CudaCosetSelectors {
                coset_shift: quotient.shift(),
                coset_generator: quotient.subgroup_generator(),
                trace_last: trace.subgroup_generator().inverse(),
                vanishing_start: quotient.shift().exp_u64(n as u64),
                vanishing_step: Goldilocks::two_adic_generator(quotient_degree.ilog2() as usize),
            };
            let evaluations = |index: usize| {
                matrices[index]
                    .split_rows(quotient_size)
                    .0
                    .as_cow()
                    .bit_reverse_rows()
            };
            let expected_values = crate::prover::quotient_values::<GoldilocksBlake3Config>(
                circuit,
                &publics,
                trace,
                quotient,
                &(circuit.preprocessed_width != 0).then(|| evaluations(0)),
                &evaluations(1),
                &evaluations(2),
                alpha,
                constraint_count,
            );
            let expected_values = RowMajorMatrix::new(
                expected_values
                    .iter()
                    .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
                    .collect(),
                2,
            );
            let actual_values = quotient_values_resident(
                &circuit.graph,
                (circuit.preprocessed_width != 0).then_some(&resident[0]),
                &resident[1],
                &resident[2],
                &publics,
                selectors,
                &alpha_flat,
                &delta,
                ext.w,
                quotient_size,
                quotient_degree,
                group_size,
            );
            assert_matrix_eq(
                &actual_values,
                &expected_values,
                &format!(
                    "quotient values: lookups={lookup_count}, group={group_size}, degree={quotient_degree}, full_field={full_field}"
                ),
            );
            let coefficients = cpu.coset_idft_batch(expected_values, quotient.shift());
            let width = 2 * quotient_degree;
            let mut sliced =
                RowMajorMatrix::new(Goldilocks::zero_vec((n << log_blowup) * width), width);
            for row in 0..n {
                for chunk in 0..quotient_degree {
                    for coordinate in 0..2 {
                        sliced.values[row * width + 2 * chunk + coordinate] =
                            coefficients.values[(chunk * n + row) * 2 + coordinate];
                    }
                }
            }
            let expected = cpu
                .coset_dft_batch(sliced, Goldilocks::GENERATOR)
                .bit_reverse_rows()
                .to_row_major_matrix();
            for host_mask in 0..8 {
                let source = |index| {
                    if host_mask & (1 << index) == 0 {
                        mmcs::CudaMatrixSource::Resident(&resident[index])
                    } else {
                        mmcs::CudaMatrixSource::Host(&matrices[index])
                    }
                };
                let actual = quotient_lde_mixed(
                    gpu,
                    &circuit.graph,
                    (circuit.preprocessed_width != 0).then(|| source(0)),
                    source(1),
                    source(2),
                    &publics,
                    selectors,
                    &alpha_flat,
                    &delta,
                    ext.w,
                    quotient_size,
                    quotient_degree,
                    group_size,
                    quotient_degree,
                    log_blowup,
                )
                .to_row_major_matrix();
                assert_matrix_eq(
                    &actual,
                    &expected,
                    &format!(
                        "quotient LDE: lookups={lookup_count}, group={group_size}, degree={quotient_degree}, host_mask={host_mask}, full_field={full_field}"
                    ),
                );
            }
        }
    }
}
