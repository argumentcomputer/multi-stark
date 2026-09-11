//! Complete PCS commitment timings: host upload, LDEs, and mixed-height Merkle tree.
//! Run identical binaries before/after a kernel change and compare commitments.
//! Witness allocation is outside the timer; cold iteration zero is reported separately.
//!
//! cargo run --release --locked --features parallel,cuda --example cuda_commit_bench
//! MULTI_STARK_CUDA_BENCH_ITERATIONS controls the number of warm iterations (default 5).

use std::hint::black_box;
use std::time::Instant;

use multi_stark::config::StarkGenericConfig;
use multi_stark::types::{
    Challenger, CommitmentParameters, ExtVal, FriParameters, GoldilocksBlake3Config, Pcs, Val,
};
use p3_commit::Pcs as PcsTrait;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

fn main() {
    let iterations: usize = std::env::var("MULTI_STARK_CUDA_BENCH_ITERATIONS")
        .map_or(5, |s| s.parse().expect("invalid iteration count"));
    let config = GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    );
    let pcs = config.pcs();
    // Equal-height matrices are hashed as concatenated rows. The mixed case
    // covers both dispatches and lower-height row injections in one commitment.
    let shapes: &[(&str, &[(usize, usize)])] = &[
        ("narrow", &[(20, 8)]),
        ("medium", &[(18, 40)]),
        ("chunk_boundary", &[(18, 128)]),
        ("above_boundary", &[(18, 129)]),
        ("wide", &[(16, 925)]),
        (
            "mixed",
            &[(18, 4), (18, 12), (18, 24), (17, 129), (16, 533)],
        ),
    ];
    println!("shape,iteration,seconds,commitment");
    for &(name, shape) in shapes {
        let mut expected = None;
        for iteration in 0..=iterations {
            let inputs = shape
                .iter()
                .enumerate()
                .map(|(matrix_index, &(log_height, width))| {
                    let height = 1 << log_height;
                    let values = (0..height * width)
                        .map(|index| {
                            let value = (index as u64)
                                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                                .wrapping_add(matrix_index as u64);
                            Val::from_u64(value)
                        })
                        .collect();
                    let domain = <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                        pcs, height,
                    );
                    (domain, RowMajorMatrix::new(values, width))
                })
                .collect::<Vec<_>>();
            let started = Instant::now();
            let (commitment, data) = <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, inputs);
            let seconds = started.elapsed().as_secs_f64();
            if let Some(expected) = &expected {
                assert_eq!(&commitment, expected);
            } else {
                expected = Some(commitment.clone());
            }
            let bytes =
                bincode::serde::encode_to_vec(&commitment, bincode::config::standard()).unwrap();
            let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
            println!("{name},{iteration},{seconds:.9},{hex}");
            // Retain all prover data until after timing: committing must not
            // silently include materialization or opening work from a consumer.
            black_box(&data);
            drop(data);
        }
    }
}
