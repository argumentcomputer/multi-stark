//! CPU versus auxiliary-GPU host-backed LDEs, including upload and download.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("enable --features parallel,cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use multi_stark::cuda::{CudaDft, pcs::CudaPcsDft};
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
    use p3_goldilocks::Goldilocks;
    use p3_matrix::Matrix;
    use p3_matrix::bitrev::BitReversibleMatrix;
    use p3_matrix::dense::RowMajorMatrix;
    use rayon::prelude::*;
    use std::time::Instant;

    let shape =
        std::env::var("MULTI_STARK_CUDA_BENCH_SHAPES").unwrap_or_else(|_| "20,533,2".into());
    let shapes = shape
        .split(';')
        .map(|shape| {
            let dimensions = shape
                .split(',')
                .map(|part| part.trim().parse::<usize>().expect("invalid shape"))
                .collect::<Vec<_>>();
            let [log_height, width, added_bits] = dimensions[..] else {
                panic!("each shape must be log_height,width,added_bits");
            };
            (log_height, width, added_bits)
        })
        .collect::<Vec<_>>();
    let copies: usize = std::env::var("MULTI_STARK_CUDA_BENCH_COPIES")
        .map_or(1, |value| value.parse().expect("invalid copy count"));
    assert!(copies > 0);
    let gpu = CudaDft::default();
    let warmup = RowMajorMatrix::new(vec![Goldilocks::ONE; (1 << 10) * 33], 33);
    assert!(
        gpu.try_coset_lde_batch_host(&warmup, 2, Goldilocks::GENERATOR)
            .is_some(),
        "set MULTI_STARK_CUDA_AUX_DEVICES to reserved CUDA ordinals"
    );
    println!(
        "log_height,width,added_bits,copies,input_bytes,output_bytes,cpu_seconds,gpu_seconds,speedup"
    );
    for (log_height, width, added_bits) in shapes {
        let height = 1usize
            .checked_shl(log_height.try_into().unwrap())
            .expect("invalid height");
        let matrix = RowMajorMatrix::new(
            (0..height * width)
                .into_par_iter()
                .map(|index| {
                    Goldilocks::from_u64((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                })
                .collect(),
            width,
        );
        let cpu_inputs = (0..copies).map(|_| matrix.clone()).collect::<Vec<_>>();
        let cpu = Radix2DitParallel::<Goldilocks>::default();
        let started = Instant::now();
        let expected = cpu_inputs
            .into_par_iter()
            .map(|input| {
                let mut result = cpu
                    .coset_lde_batch(input, added_bits, Goldilocks::GENERATOR)
                    .bit_reverse_rows()
                    .to_row_major_matrix();
                result
                    .values
                    .par_iter_mut()
                    .for_each(|value| *value = Goldilocks::from_u64(value.as_canonical_u64()));
                result
            })
            .collect::<Vec<_>>();
        let cpu_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let actual = (0..copies)
            .into_par_iter()
            .map(|_| {
                gpu.try_coset_lde_batch_host(&matrix, added_bits, Goldilocks::GENERATOR)
                    .expect("matrix did not fit on any auxiliary device")
            })
            .collect::<Vec<_>>();
        let gpu_seconds = started.elapsed().as_secs_f64();
        expected
            .par_iter()
            .zip(&actual)
            .for_each(|(expected, actual)| {
                assert_eq!(expected.dimensions(), actual.dimensions());
                expected
                    .values
                    .par_chunks(1 << 16)
                    .zip(actual.values.par_chunks(1 << 16))
                    .for_each(|(expected, actual)| assert_eq!(expected, actual));
            });
        let input_bytes = matrix.values.len() * size_of::<Goldilocks>() * copies;
        let output_bytes = input_bytes << added_bits;
        println!(
            "{log_height},{width},{added_bits},{copies},{input_bytes},{output_bytes},{cpu_seconds:.6},{gpu_seconds:.6},{:.3}",
            cpu_seconds / gpu_seconds
        );
    }
}
