use super::*;
use crate::{
    plonkish::{Assignment, CircuitBuilder, LookupConstraint, MultiStarkCircuit},
    types::Val,
};
use p3_matrix::dense::RowMajorMatrix;

fn fixture<F: Field>(lookups: usize) -> (Circuit<F>, Vec<F>) {
    let mut b = CircuitBuilder::<F>::new();
    let a = b.fixed_table(
        "duplicates",
        [1, 2, 1, 3, 2].map(|v| vec![F::from_u8(v)]).to_vec(),
    );
    let b_table = b.fixed_table(
        "pairs",
        [[1, 0], [2, 1], [1, 0]]
            .map(|row| row.map(F::from_u8).to_vec())
            .to_vec(),
    );
    b.fixed_table("unused singleton", vec![vec![F::from_u8(7)]]);
    let d = b.fixed_table(
        "triples",
        [[1, 0, 1], [2, 1, 0]]
            .map(|row| row.map(F::from_u8).to_vec())
            .to_vec(),
    );
    let wires = [0, 1, 2, 3].map(|v| b.constant(F::from_u8(v)));
    b.expose_public(wires[1]);
    let templates = [
        (a, vec![wires[1]]),
        (a, vec![wires[2]]),
        (a, vec![wires[3]]),
        (b_table, vec![wires[1], wires[0]]),
        (b_table, vec![wires[2], wires[1]]),
        (d, vec![wires[1], wires[0], wires[1]]),
        (d, vec![wires[2], wires[1], wires[0]]),
    ];
    let mut circuit = b.finish();
    let values = circuit
        .witness()
        .generate()
        .unwrap()
        .values(circuit.id())
        .unwrap()
        .to_vec();
    circuit.lookups = (0..lookups)
        .map(|i| {
            let index = match i % 16 {
                0..=9 => 0,
                n => n - 9,
            };
            let (table, values) = &templates[index];
            LookupConstraint {
                table: *table,
                values: values.clone(),
            }
        })
        .collect();
    (circuit, values)
}

fn forced_counts<F: Field>(circuit: &Circuit<F>, values: &[F]) -> Vec<Vec<F>> {
    #[cfg(feature = "parallel")]
    if !circuit.lookups.is_empty() {
        let mut plan = histogram_plan::<F>(
            3 * MIN_LOOKUPS_PER_JOB,
            circuit.tables.iter().map(|t| (t.rows.len(), t.width())),
            3,
            MAX_WORKING_BYTES,
        )
        .unwrap();
        // Small proof fixtures exercise the same counting kernel without
        // enlarging their AIR merely to meet the production work threshold.
        plan.chunk_len = circuit.lookups.len().div_ceil(3);
        plan.jobs = circuit.lookups.len().div_ceil(plan.chunk_len);
        return parallel_counts(circuit, values, &plan).unwrap();
    }
    table_counts(circuit, values)
}

fn differential<F: Field>() {
    for n in [
        0,
        1,
        17,
        2 * MIN_LOOKUPS_PER_JOB - 1,
        2 * MIN_LOOKUPS_PER_JOB,
        3 * MIN_LOOKUPS_PER_JOB + 17,
    ] {
        let (circuit, values) = fixture::<F>(n);
        let expected = serial_counts(&circuit, &values);
        assert_eq!(table_counts(&circuit, &values), expected);
        assert_eq!(forced_counts(&circuit, &values), expected);
        assert_eq!(
            expected.iter().map(Vec::len).collect::<Vec<_>>(),
            [8, 4, 2, 2]
        );
        let hot = (n / 16) * 10 + (n % 16).min(10);
        assert_eq!(expected[0][0], F::from_usize(hot));
        assert_eq!(expected[0][2], F::ZERO);
        assert_eq!(expected[0][4..], [F::ZERO; 4]);
        assert_eq!(expected[1][2..], [F::ZERO; 2]);
        assert_eq!(expected[2], [F::ZERO; 2]);
        assert_eq!(
            expected
                .iter()
                .flatten()
                .copied()
                .fold(F::ZERO, |a, b| a + b),
            F::from_usize(n)
        );
        if n > 3 * MIN_LOOKUPS_PER_JOB {
            assert!(hot > usize::from(u16::MAX));
        }
    }
    let circuit = CircuitBuilder::<F>::new().finish();
    let assignment = circuit.witness().generate().unwrap();
    assert!(table_counts(&circuit, assignment.values(circuit.id()).unwrap()).is_empty());
}

#[test]
fn goldilocks_multiplicities_match_original_counts() {
    differential::<Val>();
}
#[test]
fn baby_bear_multiplicities_match_original_counts() {
    differential::<p3_baby_bear::BabyBear>();
}

#[test]
fn histogram_planner_bounds_memory_work_and_integer_sizes() {
    let n = 96 * MIN_LOOKUPS_PER_JOB;
    let dimensions = [(5, 1), (3, 2), (1, 1), (2, 3)];
    let plan = histogram_plan::<Val>(n, dimensions.into_iter(), 96, MAX_WORKING_BYTES).unwrap();
    assert_eq!(plan.offsets, [0, 5, 8, 9, 11]);
    assert_eq!(plan.jobs, 96);
    assert_eq!(plan.chunk_len, MIN_LOOKUPS_PER_JOB);
    assert_eq!(plan.rows, 11);
    assert_eq!(plan.row_width, 3);
    let two = histogram_plan::<Val>(n, dimensions.into_iter(), 2, MAX_WORKING_BYTES).unwrap();
    let limited = histogram_plan::<Val>(n, dimensions.into_iter(), 96, two.working_bytes).unwrap();
    assert_eq!(limited.jobs, 2);
    assert_eq!(limited.working_bytes, two.working_bytes);
    assert!(histogram_plan::<Val>(n, dimensions.into_iter(), 96, two.working_bytes - 1).is_none());
    assert!(histogram_plan::<Val>(n, dimensions.into_iter(), 1, MAX_WORKING_BYTES).is_none());
    assert!(histogram_plan::<Val>(n, dimensions.into_iter(), 96, 0).is_none());
    assert!(histogram_plan::<Val>(n, [].into_iter(), 96, MAX_WORKING_BYTES).is_none());
    assert!(histogram_plan::<Val>(n, [(0, 1)].into_iter(), 96, MAX_WORKING_BYTES).is_none());
    assert!(histogram_plan::<Val>(n, [(1, 0)].into_iter(), 96, MAX_WORKING_BYTES).is_none());
    assert!(histogram_plan::<Val>(n, [(1, usize::MAX)].into_iter(), 96, usize::MAX).is_none());
    assert!(
        histogram_plan::<Val>(n, [(usize::MAX, 1), (1, 1)].into_iter(), 96, usize::MAX).is_none()
    );
    assert!(
        histogram_plan::<Val>(n, std::iter::repeat_n((1, 1), usize::MAX), 96, usize::MAX).is_none()
    );
    let work_limited = histogram_plan::<Val>(
        4 * MIN_LOOKUPS_PER_JOB,
        [(2 * MIN_LOOKUPS_PER_JOB, 1)].into_iter(),
        96,
        MAX_WORKING_BYTES,
    )
    .unwrap();
    assert_eq!(work_limited.jobs, 2);
    for n in [2 * MIN_LOOKUPS_PER_JOB, 3 * MIN_LOOKUPS_PER_JOB + 17] {
        let plan = histogram_plan::<Val>(n, dimensions.into_iter(), 96, MAX_WORKING_BYTES).unwrap();
        assert_eq!(n.div_ceil(plan.chunk_len), plan.jobs);
        assert!(plan.working_bytes <= MAX_WORKING_BYTES);
        assert!(plan.jobs * plan.rows <= n);
        assert!(plan.jobs <= 96);
    }
}

fn compiled<F: Field>(merged: bool, sharded: bool) -> (MultiStarkCircuit<F>, Assignment<F>) {
    let (circuit, _) = fixture::<F>(47);
    let compiled = if sharded {
        circuit.lower_to_multi_stark_sharded(F::from_u8(151), 16)
    } else {
        circuit.lower_to_multi_stark(F::from_u8(151))
    }
    .unwrap();
    let compiled = if merged {
        compiled.merge_table_traces(16).unwrap()
    } else {
        compiled
    };
    let assignment = compiled.witness().generate().unwrap();
    (compiled, assignment)
}

fn traces_with_counts<F: Field>(
    compiled: &MultiStarkCircuit<F>,
    assignment: &Assignment<F>,
    counts: Vec<Vec<F>>,
) -> Vec<RowMajorMatrix<F>> {
    let mut traces: Vec<_> = (0..compiled.main_heights.len())
        .map(|i| compiled.main_trace(assignment, i))
        .collect();
    if let Some(height) = compiled.merged_table_height {
        let mut values = Vec::with_capacity(height);
        for (table, counts) in compiled.circuit.tables.iter().zip(counts) {
            values.extend_from_slice(&counts[..table.rows.len()]);
        }
        values.resize(height, F::ZERO);
        traces.push(RowMajorMatrix::new_col(values));
    } else {
        traces.extend(counts.into_iter().map(RowMajorMatrix::new_col));
    }
    traces
}

fn layout_parity<F: Field>() {
    for merged in [false, true] {
        for sharded in [false, true] {
            let (compiled, assignment) = compiled::<F>(merged, sharded);
            let reference = traces_with_counts(
                &compiled,
                &assignment,
                serial_counts(
                    &compiled.circuit,
                    assignment.values(compiled.circuit.id()).unwrap(),
                ),
            );
            let parallel = traces_with_counts(
                &compiled,
                &assignment,
                forced_counts(
                    &compiled.circuit,
                    assignment.values(compiled.circuit.id()).unwrap(),
                ),
            );
            let actual = compiled.traces(&assignment).unwrap();
            let shards = compiled.trace_shards(&assignment).unwrap();
            for (i, ((expected, parallel), actual)) in
                reference.iter().zip(&parallel).zip(&actual).enumerate()
            {
                assert_eq!(actual.width, expected.width);
                assert_eq!(actual.values, expected.values);
                assert_eq!(parallel.values, expected.values);
                assert_eq!(shards.trace(i).unwrap().values, expected.values);
            }
            if merged {
                assert_eq!(reference.last().unwrap().values[11..], [F::ZERO; 5]);
            }
        }
    }
}

#[test]
fn goldilocks_merged_unmerged_and_partitioned_multiplicities_match() {
    layout_parity::<Val>();
}
#[test]
fn baby_bear_merged_unmerged_and_partitioned_multiplicities_match() {
    layout_parity::<p3_baby_bear::BabyBear>();
}

#[test]
fn multiplicities_preserve_verified_fri_proof_bytes() {
    use crate::{
        system::{System, SystemWitness},
        types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config},
    };
    let config = GoldilocksBlake3Config::new(
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
    );
    let (compiled, assignment) = compiled::<Val>(true, true);
    let claims = compiled.claims(assignment.public_values()).unwrap();
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let (system, key) = System::new(config, compiled.circuit_inputs());
    let mut encoded = Vec::new();
    for counts in [
        serial_counts(
            &compiled.circuit,
            assignment.values(compiled.circuit.id()).unwrap(),
        ),
        forced_counts(
            &compiled.circuit,
            assignment.values(compiled.circuit.id()).unwrap(),
        ),
    ] {
        let traces = traces_with_counts(&compiled, &assignment, counts);
        let proof =
            system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(traces, &system));
        system.verify_multiple_claims(&refs, &proof).unwrap();
        encoded.push(proof.to_bytes().unwrap());
    }
    assert_eq!(encoded[0], encoded[1]);
}
