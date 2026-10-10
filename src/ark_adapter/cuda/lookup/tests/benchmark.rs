use super::*;
use crate::ark_adapter::cuda::diagnostic::{memory_kib, pool_high_water, warm_ntts};
use crate::ark_adapter::cuda::lookup::distributed as device;
use crate::ark_adapter::domain::Radix2Coset;
use crate::ark_adapter::pcs::{KzgCommitment, KzgProverData};
use std::io::BufReader;
use std::time::Instant;

#[test]
#[ignore = "target-size synthetic lookup diagnostic; requires explicit memory/GPU window and saved setup path"]
fn distributed_lookup_synthetic_benchmark() {
    let setup = std::env::var("MULTI_STARK_KZG_DISTRIBUTED_SETUP").unwrap();
    let log_n: usize = std::env::var("MULTI_STARK_KZG_DISTRIBUTED_BENCH_LOG")
        .map_or(27, |value| value.parse().unwrap());
    assert!((10..=27).contains(&log_n));
    let preparation = Instant::now();
    let (circuit, _): (crate::system::Circuit<Scalar>, KzgCommitment) =
        bincode::serde::decode_from_std_read(
            &mut BufReader::new(std::fs::File::open(&setup).unwrap()),
            bincode::config::standard(),
        )
        .unwrap();
    let n = 1usize << log_n;
    let widths = [circuit.preprocessed_width, circuit.main_width];
    let groups = circuit.stage_2_width;
    let total_columns: usize = widths.iter().sum();
    let available = memory_kib("/proc/meminfo", "MemAvailable:").unwrap() * 1024;
    let host_bound = (3 * total_columns + 4 * groups + 16) * n * size_of::<Fr>();
    assert!(
        available > host_bound as u64,
        "insufficient host memory for bounded CPU lookup reference"
    );
    println!(
        "distributed_lookup_benchmark preparation rows={n} widths={widths:?} groups={groups} nonconstant_columns={total_columns} messages={} prefix_nodes={} available_host_bytes={available} conservative_host_bound={host_bound} setup={setup}",
        circuit.graph.lookups.len(),
        circuit.graph.lookup_prefix_len
    );
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(2, b"distributed-lookup-unused-srs")),
        2,
    );
    let trace = Radix2Coset {
        log_size: log_n,
        shift: Scalar::ONE,
    };
    let make = |width: usize, seed: usize| {
        KzgProverData::coefficient_fixture(
            trace,
            (0..width)
                .into_par_iter()
                .map(|column| {
                    let palette: Vec<_> = (0..4096)
                        .map(|i| Fr::from((1 + i * 731 + (column + seed) * 179) as u64))
                        .collect();
                    crate::ark_adapter::buffer::generate(n, |row| {
                        let mixed = (row as u64).wrapping_mul(0x9e3779b185ebca87);
                        palette[((mixed ^ (mixed >> 29)) & 4095) as usize]
                    })
                })
                .collect(),
        )
    };
    let fixed = make(widths[0], 0);
    let main = make(widths[1], widths[0]);
    for data in [&fixed, &main] {
        data.matrices[0].resident_columns();
    }
    let evaluated: usize = [&fixed, &main]
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
    let beta = Scalar::from_u8(23);
    let gamma = Scalar::from_u8(31);
    warm_ntts();
    println!(
        "distributed_lookup_benchmark prepared seconds={:.6} beta=23 gamma=31",
        preparation.elapsed().as_secs_f64()
    );
    let rss_reset = std::fs::write("/proc/self/clear_refs", b"5").is_ok();
    pool_high_water(true);
    println!("distributed_lookup_benchmark phase=cpu_start rss_peak_reset={rss_reset}");
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
    let baseline_seconds = started.elapsed().as_secs_f64();
    println!(
        "distributed_lookup_benchmark phase=cpu_done seconds={baseline_seconds:.6} pool_peak_bytes={:?} rss_peak_kib={:?} total={expected_total:?}",
        pool_high_water(false),
        memory_kib("/proc/self/status", "VmHWM:")
    );
    let rss_reset = std::fs::write("/proc/self/clear_refs", b"5").is_ok();
    pool_high_water(true);
    println!("distributed_lookup_benchmark phase=distributed_start rss_peak_reset={rss_reset}");
    let started = Instant::now();
    let (actual, actual_total) = coefficients_impl(&input, beta, gamma, Some(device::options()))
        .expect("target-size distributed lookup must pass admission");
    let distributed_seconds = started.elapsed().as_secs_f64();
    println!(
        "distributed_lookup_benchmark phase=distributed_done seconds={distributed_seconds:.6} pool_peak_bytes={:?} rss_peak_kib={:?} total={actual_total:?}",
        pool_high_water(false),
        memory_kib("/proc/self/status", "VmHWM:")
    );
    assert_eq!(actual_total, expected_total);
    for (actual, (expected, _, _)) in actual.iter().zip(expected.cuda_columns()) {
        assert_eq!(actual.as_slice(), expected);
    }
    println!(
        "distributed_lookup_benchmark coefficient_parity=true total_parity=true baseline_seconds={baseline_seconds:.6} distributed_seconds={distributed_seconds:.6} speedup={:.3}",
        baseline_seconds / distributed_seconds
    );
}
