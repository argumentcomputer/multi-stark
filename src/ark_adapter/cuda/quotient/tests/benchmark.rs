use super::*;
use crate::ark_adapter::cuda::diagnostic::{memory_kib, pool_high_water, warm_ntts};
use crate::ark_adapter::pcs::{KzgCommitment, KzgProverData};
use crate::config::ProofConfig;
use crate::traits::TwoAdicField;
use std::io::BufReader;
use std::time::Instant;

#[test]
#[ignore = "target-size synthetic quotient diagnostic; requires explicit memory/GPU window and saved setup path"]
fn distributed_quotient_synthetic_benchmark() {
    let setup = std::env::var("MULTI_STARK_KZG_DISTRIBUTED_SETUP").unwrap();
    let log_n: usize =
        std::env::var("MULTI_STARK_KZG_DISTRIBUTED_BENCH_LOG").map_or(27, |s| s.parse().unwrap());
    assert!((10..=27).contains(&log_n));
    let preparation = Instant::now();
    let (circuit, _): (crate::system::Circuit<Scalar>, KzgCommitment) =
        bincode::serde::decode_from_std_read(
            &mut BufReader::new(std::fs::File::open(&setup).unwrap()),
            bincode::config::standard(),
        )
        .unwrap();
    assert_eq!(circuit.quotient_degree(), 2);
    let n = 1usize << log_n;
    let widths = [
        circuit.preprocessed_width,
        circuit.main_width,
        circuit.stage_2_width,
    ];
    let total_columns: usize = widths.iter().sum();
    let available = memory_kib("/proc/meminfo", "MemAvailable:").unwrap() * 1024;
    let host_bound = (total_columns * 3 + 16) * n * size_of::<Fr>();
    assert!(
        available > host_bound as u64,
        "insufficient available host memory for bounded CPU reference"
    );
    println!(
        "distributed_benchmark preparation rows={n} widths={widths:?} nonconstant_columns={total_columns} available_host_bytes={available} conservative_host_bound={host_bound} setup={setup}"
    );
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(
            2,
            b"distributed-diagnostic-unused-srs",
        )),
        2,
    );
    let trace = Radix2Coset {
        log_size: log_n,
        shift: Scalar::ONE,
    };
    let quotient = trace.create_disjoint_domain(2 * n);
    let make = |width: usize, seed: usize| {
        KzgProverData::coefficient_fixture(
            trace,
            (0..width)
                .into_par_iter()
                .map(|column| {
                    let palette: Vec<_> = (0..4096)
                        .map(|i| Fr::from((1 + i * 731 + (column + seed) * 179) as u64))
                        .collect();
                    crate::ark_adapter::buffer::generate(n, |row| palette[row % palette.len()])
                })
                .collect(),
        )
    };
    let fixed = make(widths[0], 0);
    let main = make(widths[1], widths[0]);
    let stage2 = make(widths[2], widths[0] + widths[1]);
    for data in [&fixed, &main, &stage2] {
        data.matrices[0].resident_columns();
    }
    let evaluated: usize = [&fixed, &main, &stage2]
        .into_iter()
        .map(|data| {
            data.matrices[0]
                .cuda_columns()
                .iter()
                .filter(|(_, _, constant)| !constant)
                .count()
        })
        .sum();
    assert_eq!(evaluated, total_columns);
    let public_count = circuit
        .graph
        .nodes
        .iter()
        .filter_map(|node| {
            if let Node::Public(index) = node {
                Some(*index as usize + 1)
            } else {
                None
            }
        })
        .max()
        .unwrap_or(0)
        .max(4);
    let mut publics: Vec<_> = (0..public_count)
        .map(|i| Scalar::from_usize(53 + i * 17))
        .collect();
    publics[..4].copy_from_slice(&[23, 31, 5, 47].map(Scalar::from_u8));
    let input = QuotientCommitInput::<KzgConfig> {
        circuit: &circuit,
        lookup_publics: publics,
        trace_domain: trace,
        quotient_domain: quotient,
        preprocessed: Some((&fixed, 0)),
        stage_1: (&main, 0),
        stage_2: (&stage2, 0),
        constraint_count: circuit.constraint_count(),
    };
    let alpha = Scalar::from_u8(41);
    warm_ntts();
    println!(
        "distributed_benchmark prepared seconds={:.6} publics={:?}",
        preparation.elapsed().as_secs_f64(),
        input.lookup_publics
    );
    let rss_reset = std::fs::write("/proc/self/clear_refs", b"5").is_ok();
    pool_high_water(true);
    println!("distributed_benchmark phase=cpu_start rss_peak_reset={rss_reset}");
    let started = Instant::now();
    let mut expected = vec![Fr::ZERO; 2 * n];
    let mut shift = quotient.shift;
    for coset in 0..2 {
        let domain = Radix2Coset {
            log_size: log_n,
            shift,
        };
        let values = config
            .accelerated_quotient_values(
                &circuit,
                &input.lookup_publics,
                trace,
                domain,
                input.preprocessed,
                input.stage_1,
                input.stage_2,
                alpha,
                input.constraint_count,
            )
            .unwrap();
        expected
            .par_chunks_mut(2)
            .zip(values.into_par_iter())
            .for_each(|(row, value)| row[coset] = value.0);
        shift *= Scalar::two_adic_generator(quotient.log_size);
    }
    fft(&mut expected, true, quotient.shift.0);
    let baseline_seconds = started.elapsed().as_secs_f64();
    println!(
        "distributed_benchmark phase=cpu_done seconds={baseline_seconds:.6} pool_peak_bytes={:?} rss_peak_kib={:?}",
        pool_high_water(false),
        memory_kib("/proc/self/status", "VmHWM:")
    );
    let rss_reset = std::fs::write("/proc/self/clear_refs", b"5").is_ok();
    pool_high_water(true);
    println!("distributed_benchmark phase=distributed_start rss_peak_reset={rss_reset}");
    let started = Instant::now();
    let actual = coefficients_impl(&input, alpha, Some(distributed::options()))
        .expect("target-size distributed quotient must pass admission");
    let distributed_seconds = started.elapsed().as_secs_f64();
    println!(
        "distributed_benchmark phase=distributed_done seconds={distributed_seconds:.6} pool_peak_bytes={:?} rss_peak_kib={:?}",
        pool_high_water(false),
        memory_kib("/proc/self/status", "VmHWM:")
    );
    assert!(actual.iter().flatten().eq(expected.iter()));
    println!(
        "distributed_benchmark coefficient_parity=true baseline_seconds={baseline_seconds:.6} distributed_seconds={distributed_seconds:.6} speedup={:.3}",
        baseline_seconds / distributed_seconds
    );
}
