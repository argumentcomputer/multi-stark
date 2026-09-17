//! Resident LDE timings including upload, excluding host input construction.
//! Run the identical benchmark against both revisions. Correctness is covered
//! by resident_coset_lde_matches_cpu_storage and proof_compatibility.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("enable --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use multi_stark::cuda::{CudaDft, pcs::CudaPcsDft};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;
    use std::hint::black_box;
    use std::time::Instant;

    let iterations: usize = std::env::var("MULTI_STARK_CUDA_BENCH_ITERATIONS")
        .map_or(7, |s| s.parse().expect("invalid iteration count"));
    let gpu = CudaDft::default();
    // MULTI_STARK_CUDA_BENCH_SHAPES="20,533,2;24,6,2" runs those shapes only.
    let only: Vec<(usize, usize, usize)> = std::env::var("MULTI_STARK_CUDA_BENCH_SHAPES")
        .map(|list| {
            list.split(';')
                .map(|shape| {
                    let mut parts = shape
                        .split(',')
                        .map(|part| part.trim().parse().expect("invalid shape"));
                    (
                        parts.next().expect("log height"),
                        parts.next().expect("width"),
                        parts.next().expect("added bits"),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    println!("log_height,width,added_bits,iteration,seconds");
    // The last five are the shapes the profiled Init proofs commit most:
    // BLAKE3 pieces, wide and narrow IxVM circuits at the height cap, and
    // the width-2 codewords of the quotient and narrow lookups.
    for (log_height, width, added_bits) in [
        (20, 1, 1),
        (20, 2, 1),
        (20, 8, 1),
        (18, 40, 1),
        (18, 128, 1),
        (18, 129, 1),
        (16, 925, 1),
        (18, 40, 2),
        (20, 533, 2),
        (24, 6, 2),
        (24, 17, 2),
        (22, 2, 2),
        (20, 2, 2),
        // Short shapes below the sppark height threshold: the aggregation
        // circuits' tables and narrow lookup and quotient codewords.
        (8, 20, 2),
        (10, 33, 2),
        (12, 8, 2),
        (14, 40, 2),
        (15, 129, 2),
        (16, 2, 2),
        (17, 16, 2),
        (18, 2, 2),
        (19, 2, 2),
        (19, 49, 2),
    ] {
        if !only.is_empty() && !only.contains(&(log_height, width, added_bits)) {
            continue;
        }
        let height = 1 << log_height;
        let values = (0..height * width)
            .map(|index| Goldilocks::from_u64((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)))
            .collect();
        let matrix = RowMajorMatrix::new(values, width);
        for iteration in 0..=iterations {
            let started = Instant::now();
            let lde = CudaPcsDft::coset_lde_batch_resident(
                &gpu,
                &matrix,
                added_bits,
                Goldilocks::GENERATOR,
            );
            let seconds = started.elapsed().as_secs_f64();
            println!("{log_height},{width},{added_bits},{iteration},{seconds:.9}");
            black_box(&lde);
            drop(lde);
        }
    }
}
