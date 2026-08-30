//! Transfer-inclusive CPU/CUDA DFT and coset-LDE comparison.
//!
//! Environment controls (comma-separated where applicable):
//! - `MULTI_STARK_CUDA_BENCH_LOG_HEIGHTS` (default `12,16,19`)
//! - `MULTI_STARK_CUDA_BENCH_WIDTHS` (default `2,8,32`)
//! - `MULTI_STARK_CUDA_BENCH_ITERATIONS` (default `3`)
//! - `MULTI_STARK_CUDA_BENCH_LOG_BLOWUP` (default `1`)
//! - `MULTI_STARK_CUDA_DEVICE` (default `0`)

#[cfg(feature = "cuda")]
mod enabled {
    use std::env;
    use std::hint::black_box;
    use std::time::Instant;

    use multi_stark::cuda::CudaDft;
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_goldilocks::Goldilocks;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;

    pub(super) fn run() {
        let log_heights = list("MULTI_STARK_CUDA_BENCH_LOG_HEIGHTS", "12,16,19");
        let widths = list("MULTI_STARK_CUDA_BENCH_WIDTHS", "2,8,32");
        let iterations = scalar("MULTI_STARK_CUDA_BENCH_ITERATIONS", 3);
        let log_blowup = scalar("MULTI_STARK_CUDA_BENCH_LOG_BLOWUP", 1);
        let device = scalar::<i32>("MULTI_STARK_CUDA_DEVICE", 0);
        let cpu = Radix2DitParallel::<Goldilocks>::default();
        let gpu = CudaDft::new(device);

        println!("backend,operation,log_height,width,log_blowup,iteration,seconds,elements");
        for log_height in log_heights {
            let height = 1usize << log_height;
            for &width in &widths {
                let input = deterministic_matrix(height, width);

                let expected_dft = cpu.dft_batch(input.clone()).to_row_major_matrix();
                let warm_dft = gpu.dft_batch(input.clone()).to_row_major_matrix();
                assert_eq!(
                    warm_dft, expected_dft,
                    "DFT correctness at 2^{log_height}x{width}"
                );
                for iteration in 0..iterations {
                    let run_input = input.clone();
                    let started = Instant::now();
                    let output = cpu.dft_batch(run_input).to_row_major_matrix();
                    let elapsed = started.elapsed().as_secs_f64();
                    assert_eq!(output, expected_dft);
                    black_box(output);
                    emit(
                        "cpu",
                        "dft",
                        log_height,
                        width,
                        0,
                        iteration,
                        elapsed,
                        height * width,
                    );

                    let run_input = input.clone();
                    let started = Instant::now();
                    let output = gpu.dft_batch(run_input).to_row_major_matrix();
                    let elapsed = started.elapsed().as_secs_f64();
                    assert_eq!(output, expected_dft);
                    black_box(output);
                    emit(
                        if CudaDft::uses_cuda_dft(height, width) {
                            "cuda"
                        } else {
                            "cpu-fallback"
                        },
                        "dft",
                        log_height,
                        width,
                        0,
                        iteration,
                        elapsed,
                        height * width,
                    );
                }

                let expected_lde = cpu
                    .coset_lde_batch(input.clone(), log_blowup, Goldilocks::GENERATOR)
                    .to_row_major_matrix();
                let warm_lde = gpu
                    .coset_lde_batch(input.clone(), log_blowup, Goldilocks::GENERATOR)
                    .to_row_major_matrix();
                assert_eq!(
                    warm_lde, expected_lde,
                    "LDE correctness at 2^{log_height}x{width}"
                );
                for iteration in 0..iterations {
                    let run_input = input.clone();
                    let started = Instant::now();
                    let output = cpu
                        .coset_lde_batch(run_input, log_blowup, Goldilocks::GENERATOR)
                        .to_row_major_matrix();
                    let elapsed = started.elapsed().as_secs_f64();
                    assert_eq!(output, expected_lde);
                    black_box(output);
                    emit(
                        "cpu",
                        "coset_lde",
                        log_height,
                        width,
                        log_blowup,
                        iteration,
                        elapsed,
                        (height << log_blowup) * width,
                    );

                    let run_input = input.clone();
                    let started = Instant::now();
                    let output = gpu
                        .coset_lde_batch(run_input, log_blowup, Goldilocks::GENERATOR)
                        .to_row_major_matrix();
                    let elapsed = started.elapsed().as_secs_f64();
                    assert_eq!(output, expected_lde);
                    black_box(output);
                    emit(
                        if CudaDft::uses_cuda_coset_lde(height, width, log_blowup) {
                            "cuda"
                        } else {
                            "cpu-fallback"
                        },
                        "coset_lde",
                        log_height,
                        width,
                        log_blowup,
                        iteration,
                        elapsed,
                        (height << log_blowup) * width,
                    );
                }
            }
        }
    }

    fn deterministic_matrix(height: usize, width: usize) -> RowMajorMatrix<Goldilocks> {
        RowMajorMatrix::new(
            (0..height * width)
                .map(|index| {
                    let index = u64::try_from(index).expect("matrix index exceeds u64");
                    Goldilocks::from_u64(index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xdead_beef)
                })
                .collect(),
            width,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn emit(
        backend: &str,
        operation: &str,
        log_height: usize,
        width: usize,
        log_blowup: usize,
        iteration: usize,
        seconds: f64,
        elements: usize,
    ) {
        println!(
            "{backend},{operation},{log_height},{width},{log_blowup},{iteration},{seconds:.9},{elements}"
        );
    }

    fn list(name: &str, default: &str) -> Vec<usize> {
        env::var(name)
            .unwrap_or_else(|_| default.to_owned())
            .split(',')
            .map(|value| {
                value
                    .trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("invalid {name}"))
            })
            .collect()
    }

    fn scalar<T>(name: &str, default: T) -> T
    where
        T: std::str::FromStr + Copy,
    {
        env::var(name).map_or(default, |value| {
            value.parse().unwrap_or_else(|_| panic!("invalid {name}"))
        })
    }
}

#[cfg(feature = "cuda")]
fn main() {
    enabled::run();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("enable the `cuda` feature to run this benchmark");
    std::process::exit(2);
}
