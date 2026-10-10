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
    let values = circuit.witness().generate().unwrap().values().to_vec();
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
    assert!(table_counts(&circuit, assignment.values()).is_empty());
}

#[test]
fn goldilocks_multiplicities_match_original_counts() {
    differential::<Val>();
}

#[cfg(feature = "kzg")]
#[test]
fn scalar_multiplicities_match_original_counts() {
    differential::<crate::ark_adapter::Scalar>();
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
                serial_counts(&compiled.circuit, assignment.values()),
            );
            let parallel = traces_with_counts(
                &compiled,
                &assignment,
                forced_counts(&compiled.circuit, assignment.values()),
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

#[cfg(feature = "kzg")]
#[test]
fn scalar_merged_unmerged_and_partitioned_multiplicities_match() {
    layout_parity::<crate::ark_adapter::Scalar>();
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
        serial_counts(&compiled.circuit, assignment.values()),
        forced_counts(&compiled.circuit, assignment.values()),
    ] {
        let traces = traces_with_counts(&compiled, &assignment, counts);
        let proof =
            system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(traces, &system));
        system.verify_multiple_claims(&refs, &proof).unwrap();
        encoded.push(proof.to_bytes().unwrap());
    }
    assert_eq!(encoded[0], encoded[1]);
}

#[cfg(feature = "kzg")]
#[test]
fn multiplicities_preserve_verified_kzg_proof_bytes() {
    use crate::{
        ark_adapter::{KzgConfig, PublicSetup, Scalar, Srs, compact::FixedProofCodec},
        system::{System, SystemWitness},
    };
    use std::sync::Arc;
    let powers = Srs::unsafe_dev_setup(128, b"table-multiplicity-development-only");
    let srs = Srs::from_public_powers(
        powers.g1,
        powers.g2,
        powers.tau_g2,
        PublicSetup {
            max_degree: 254,
            id: *blake3::hash(b"table-multiplicity-development-only").as_bytes(),
        },
    )
    .unwrap();
    let config = KzgConfig::new(Arc::new(srs), 4);
    for merged in [false, true] {
        let (compiled, assignment) = compiled::<Scalar>(merged, false);
        let claims = compiled.claims(assignment.public_values()).unwrap();
        let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
        let (system, key) = System::new(config.clone(), compiled.circuit_inputs());
        let mut encoded = Vec::new();
        for counts in [
            serial_counts(&compiled.circuit, assignment.values()),
            forced_counts(&compiled.circuit, assignment.values()),
        ] {
            let traces = traces_with_counts(&compiled, &assignment, counts);
            let proof = system.prove_multiple_claims(
                &key,
                &refs,
                SystemWitness::from_stage_1(traces, &system),
            );
            system.verify_multiple_claims(&refs, &proof).unwrap();
            encoded.push(
                FixedProofCodec::new(&system, &proof.log_degrees)
                    .unwrap()
                    .encode(&proof)
                    .unwrap(),
            );
        }
        assert_eq!(encoded[0], encoded[1]);
        println!(
            "MULTIPLICITY_PROOF {}",
            serde_json::json!({
                "merged": merged, "bytes": encoded[0].len(),
                "blake3": blake3::hash(&encoded[0]).to_hex().to_string(), "verified": true, "parity": true,
            })
        );
    }
}

#[cfg(all(feature = "kzg", feature = "parallel"))]
fn benchmark_fixture(
    lookups: usize,
) -> (
    Circuit<crate::ark_adapter::Scalar>,
    Vec<crate::ark_adapter::Scalar>,
) {
    use crate::ark_adapter::Scalar;
    let mut b = CircuitBuilder::<Scalar>::new();
    let range = b.fixed_table(
        "Fq u16",
        (0..=u16::MAX).map(|v| vec![Scalar::from_u16(v)]).collect(),
    );
    let nibble = b.fixed_table(
        "nibble",
        (0..16).map(|v| vec![Scalar::from_u8(v)]).collect(),
    );
    let xor = b.fixed_table(
        "nibble XOR",
        (0..16)
            .flat_map(|a| {
                (0..16).map(move |c| {
                    vec![
                        Scalar::from_u8(a),
                        Scalar::from_u8(c),
                        Scalar::from_u8(a ^ c),
                    ]
                })
            })
            .collect(),
    );
    let split = b.fixed_table(
        "nibble split",
        (0..16)
            .map(|v| {
                vec![
                    Scalar::from_u8(v),
                    Scalar::from_u8(v & 7),
                    Scalar::from_u8(v >> 3),
                ]
            })
            .collect(),
    );
    let wires: Vec<_> = (0..=u16::MAX)
        .map(|v| b.constant(Scalar::from_u16(v)))
        .collect();
    let mut circuit = b.finish();
    let values = circuit.witness().generate().unwrap().values().to_vec();
    circuit.lookups = (0..lookups)
        .map(|i| {
            let mixed = (i as u64)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left(23);
            let a = (mixed & 15) as usize;
            let c = ((mixed >> 16) & 15) as usize;
            let (table, values) = match i % 32 {
                0..=24 => (range, vec![wires[(mixed & 0xffff) as usize]]),
                25..=27 => (nibble, vec![wires[a]]),
                28..=30 => (xor, vec![wires[a], wires[c], wires[a ^ c]]),
                _ => (split, vec![wires[a], wires[a & 7], wires[a >> 3]]),
            };
            LookupConstraint { table, values }
        })
        .collect();
    (circuit, values)
}

#[cfg(all(feature = "kzg", feature = "parallel"))]
#[test]
#[ignore = "isolated table-multiplicity comparison; default fixture has 32,587,682 lookups"]
fn table_multiplicity_operation_benchmark() {
    use crate::ark_adapter::Scalar;
    use std::{hint::black_box, time::Instant};
    let lookups = std::env::var("MULTI_STARK_MULTIPLICITY_LOOKUPS")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("valid MULTI_STARK_MULTIPLICITY_LOOKUPS")
        })
        .unwrap_or(32_587_682);
    let fixture_start = Instant::now();
    let (circuit, values) = benchmark_fixture(lookups);
    let fixture_seconds = fixture_start.elapsed().as_secs_f64();
    let threads = p3_maybe_rayon::prelude::current_num_threads();
    let dimensions: Vec<_> = circuit
        .tables
        .iter()
        .map(|t| (t.rows.len(), t.width()))
        .collect();
    assert_eq!(dimensions, [(65_536, 1), (16, 1), (256, 3), (16, 3)]);
    let plan = histogram_plan::<Scalar>(
        lookups,
        dimensions.iter().copied(),
        threads,
        MAX_WORKING_BYTES,
    )
    .expect("benchmark requires sufficient work and at least two default-pool threads");
    let lookup_entry_bytes = circuit.lookups.capacity() * size_of::<LookupConstraint>();
    let lookup_wire_bytes: usize = circuit
        .lookups
        .iter()
        .map(|l| l.values.capacity() * size_of::<crate::plonkish::Value>())
        .sum();
    let reference = serial_counts(&circuit, &values);
    assert_eq!(table_counts(&circuit, &values), reference);
    let checksum = |counts: &[Vec<Scalar>]| {
        let mut hash = blake3::Hasher::new();
        for row in counts {
            hash.update(&(row.len() as u64).to_le_bytes());
            for value in row {
                for limb in value.canonical_limbs_le() {
                    hash.update(&limb.to_le_bytes());
                }
            }
        }
        hash.finalize().to_hex().to_string()
    };
    let digest = checksum(&reference);
    println!(
        "MULTIPLICITY_BENCH {}",
        serde_json::json!({
            "type": "fixture",
            "lookups": lookups, "table_dimensions": dimensions, "real_table_rows": plan.rows,
            "assignment_values": values.len(), "default_pool_threads": threads,
            "jobs": plan.jobs, "chunk_len": plan.chunk_len, "histogram_working_bytes": plan.working_bytes,
            "histogram_budget_bytes": MAX_WORKING_BYTES, "fixture_seconds_excluded": fixture_seconds,
            "lookup_entry_bytes": lookup_entry_bytes, "lookup_wire_capacity_bytes": lookup_wire_bytes,
            "allocation_overhead_included": false, "checksum_blake3": digest,
            "lookup_mix_per_32": [25, 3, 3, 1],
            "checksum_encoding": "table padded length u64le, then canonical Scalar limbs u64le",
            "warmup_parity": true, "samples": 5,
            "scope": "Synthetic native-table distribution and small reused assignment pool; count allocation, scan, reduction and field conversion only; no lowering or proof timing. Histogram payload cap excludes allocator overhead, Rayon stacks and output traces.",
        })
    );
    for sample in 0..5 {
        let mut seconds = [0.0; 2];
        for optimized in [sample % 2 != 0, sample % 2 == 0] {
            let started = Instant::now();
            let counts = black_box(if optimized {
                table_counts(black_box(&circuit), black_box(&values))
            } else {
                serial_counts(black_box(&circuit), black_box(&values))
            });
            seconds[usize::from(optimized)] = started.elapsed().as_secs_f64();
            assert_eq!(counts, reference);
            assert_eq!(checksum(&counts), digest);
        }
        println!(
            "MULTIPLICITY_BENCH {}",
            serde_json::json!({
                "type": "sample",
                "sample": sample, "first": if sample % 2 == 0 { "serial" } else { "automatic" },
                "serial_seconds": seconds[0], "automatic_seconds": seconds[1],
                "parity": true, "checksum_blake3": digest,
            })
        );
    }
}
