//! sppark's Goldilocks NTT through the device-pointer adapter in
//! `cuda/sppark_ntt.cu`: the comparison backend behind `cuda-sppark`.
//!
//! The contracts the prover relies on, checked by the tests below against
//! the CPU reference: a forward transform in natural order equals the DFT,
//! the `R` orders are the bit-reversed permutation of the `N` orders,
//! inverse transforms are normalized upstream, the coset variant is the
//! DFT on the coset shifted by the field generator, and outputs are stored
//! as canonical words. Inputs must be canonical: upstream does not reduce
//! representatives at or above the modulus.

use core::ffi::c_int;

use p3_goldilocks::Goldilocks;

use super::check_cuda;

/// `NTT::InputOutputOrder`: whether the input and the output are in natural
/// (`N`) or bit-reversed (`R`) order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Order {
    NN = 0,
    NR = 1,
    RN = 2,
    RR = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Direction {
    Forward = 0,
    Inverse = 1,
}

unsafe extern "C" {
    fn multi_stark_sppark_max_lg_domain() -> c_int;
    fn multi_stark_sppark_backend_selected() -> c_int;
    fn multi_stark_sppark_select_backend(selected: c_int);
    fn multi_stark_sppark_takes(height: usize) -> c_int;
    fn multi_stark_sppark_transforms_run() -> u64;
    fn multi_stark_sppark_takes_lde(height: usize, width: usize, added_bits: usize) -> c_int;
    fn multi_stark_sppark_takes_forward(height: usize, width: usize) -> c_int;
    fn multi_stark_sppark_panel_bytes(height: usize, width: usize, added_bits: usize) -> usize;
    fn multi_stark_sppark_forward_panel_bytes(height: usize, width: usize) -> usize;
    fn multi_stark_sppark_ntt_device(
        device: c_int,
        d_inout: *mut u64,
        lg: u32,
        order: c_int,
        direction: c_int,
        coset: c_int,
    ) -> c_int;
    fn multi_stark_sppark_ntt_host(
        device: c_int,
        inout: *mut u64,
        lg: u32,
        order: c_int,
        direction: c_int,
        coset: c_int,
    ) -> c_int;
    fn multi_stark_sppark_ntt_batch_host(
        device: c_int,
        inout: *mut u64,
        lg: u32,
        order: c_int,
        direction: c_int,
        coset: c_int,
        batch: u32,
        stride: usize,
    ) -> c_int;
}

/// Whether the prover's resident LDEs take the sppark path: selected by
/// `MULTI_STARK_CUDA_NTT=sppark` or by [`select_backend`].
pub fn backend_selected() -> bool {
    unsafe { multi_stark_sppark_backend_selected() != 0 }
}

/// Which transforms take the sppark path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The first-party kernels only.
    Legacy,
    /// sppark for LDEs at or above the height threshold
    /// (`MULTI_STARK_SPPARK_MIN_LOG_HEIGHT`, 18), where the panel path
    /// wins; the first-party kernels below it.
    Sppark,
    /// sppark at every height: for comparing the paths on small shapes.
    SpparkAllHeights,
}

/// Selects the backend for the rest of the process. Comparisons in one
/// process switch between constructions; concurrent constructions all see
/// the latest value.
pub fn select_backend(backend: Backend) {
    let flag = match backend {
        Backend::Legacy => 0,
        Backend::Sppark => 1,
        Backend::SpparkAllHeights => 2,
    };
    unsafe { multi_stark_sppark_select_backend(flag) }
}

/// How many transforms the adapter has run in this process.
pub fn transforms_run() -> u64 {
    unsafe { multi_stark_sppark_transforms_run() }
}

/// Whether a transform of `height` input rows is tall enough for the sppark
/// path; the shape rules below decide a dispatch.
pub fn takes(height: usize) -> bool {
    unsafe { multi_stark_sppark_takes(height) != 0 }
}

/// Whether a resident coset LDE of the shape takes the sppark path: tall
/// enough, within upstream's compiled domain after expansion, and one
/// column's scratch within the panel budget.
pub fn takes_lde(height: usize, width: usize, added_bits: usize) -> bool {
    unsafe { multi_stark_sppark_takes_lde(height, width, added_bits) != 0 }
}

/// Whether a forward transform of the shape takes the sppark path.
pub fn takes_forward(height: usize, width: usize) -> bool {
    unsafe { multi_stark_sppark_takes_forward(height, width) != 0 }
}

/// The scratch one forward transform of the shape allocates on the sppark
/// path; zero when the shape stays on the first-party kernels.
pub fn forward_panel_bytes(height: usize, width: usize) -> usize {
    unsafe { multi_stark_sppark_forward_panel_bytes(height, width) }
}

/// Serializes tests that switch the process-wide backend, so a comparison
/// sees the backend it selected on both of its constructions.
#[cfg(test)]
pub(crate) fn backend_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The scratch the sppark path allocates for that LDE, to admit alongside
/// the trace and the LDE; zero when the first-party kernels take it.
pub fn panel_bytes(height: usize, width: usize, added_bits: usize) -> usize {
    unsafe { multi_stark_sppark_panel_bytes(height, width, added_bits) }
}

/// The largest log domain size the compiled upstream parameters support.
pub fn max_log_domain() -> usize {
    usize::try_from(unsafe { multi_stark_sppark_max_lg_domain() }).expect("domain limit")
}

fn log_len(values: usize) -> u32 {
    assert!(
        values.is_power_of_two() && values > 1,
        "sppark transforms take 2^lg elements, lg > 0"
    );
    values.trailing_zeros()
}

/// Transforms `values` in place on `device`: uploaded, transformed and
/// downloaded within the call. `coset` selects the coset by the field
/// generator. An error is the CUDA status the adapter returned; `device`
/// must be the CUDA ordinal of a device upstream supports.
pub fn try_ntt_host(
    device: i32,
    values: &mut [Goldilocks],
    order: Order,
    direction: Direction,
    coset: bool,
) -> Result<(), i32> {
    let lg = log_len(values.len());
    let status = unsafe {
        multi_stark_sppark_ntt_host(
            device,
            values.as_mut_ptr().cast(),
            lg,
            order as c_int,
            direction as c_int,
            c_int::from(coset),
        )
    };
    if status == 0 { Ok(()) } else { Err(status) }
}

/// `count` vectors of `2^lg` elements laid out `stride` elements apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Batch {
    pub lg: u32,
    pub count: u32,
    pub stride: usize,
}

/// Transforms the vectors of `batch` in `values` (`count * stride` long) in
/// one batched launch sequence, uploaded, transformed and downloaded within
/// the call.
pub fn try_ntt_batch_host(
    device: i32,
    values: &mut [Goldilocks],
    batch: Batch,
    order: Order,
    direction: Direction,
    coset: bool,
) -> Result<(), i32> {
    assert_eq!(values.len(), batch.count as usize * batch.stride);
    let status = unsafe {
        multi_stark_sppark_ntt_batch_host(
            device,
            values.as_mut_ptr().cast(),
            batch.lg,
            order as c_int,
            direction as c_int,
            c_int::from(coset),
            batch.count,
            batch.stride,
        )
    };
    if status == 0 { Ok(()) } else { Err(status) }
}

/// [`try_ntt_host`], panicking on a CUDA status like the other backends.
pub fn ntt_host(
    device: i32,
    values: &mut [Goldilocks],
    order: Order,
    direction: Direction,
    coset: bool,
) {
    if let Err(status) = try_ntt_host(device, values, order, direction, coset) {
        check_cuda(status, "sppark host transform");
    }
}

/// Transforms `2^lg` field elements at `d_inout` in place on the calling
/// thread's stream.
///
/// # Safety
///
/// `d_inout` must be a device allocation of `2^lg` 64-bit words on
/// `device`, and no other stream may access it until work enqueued after
/// this call on the calling thread's stream has completed.
pub unsafe fn ntt_device(
    device: i32,
    d_inout: *mut u64,
    lg: u32,
    order: Order,
    direction: Direction,
    coset: bool,
) {
    let status = unsafe {
        multi_stark_sppark_ntt_device(
            device,
            d_inout,
            lg,
            order as c_int,
            direction as c_int,
            c_int::from(coset),
        )
    };
    check_cuda(status, "sppark device transform");
}

/// The raw stored words of `values`; Goldilocks is `repr(transparent)`.
pub fn raw_words(values: &[Goldilocks]) -> &[u64] {
    // SAFETY: Goldilocks is a transparent wrapper over u64.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::reverse_slice_index_bits;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    const LOGS: [usize; 6] = [1, 5, 10, 11, 16, 20];

    fn random(lg: usize, seed: u64) -> Vec<Goldilocks> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..1usize << lg).map(|_| rng.random()).collect()
    }

    fn cpu_dft(values: &[Goldilocks]) -> Vec<Goldilocks> {
        Radix2DitParallel::<Goldilocks>::default()
            .dft_batch(RowMajorMatrix::new(values.to_vec(), 1))
            .to_row_major_matrix()
            .values
    }

    fn cpu_idft(values: &[Goldilocks]) -> Vec<Goldilocks> {
        Radix2DitParallel::<Goldilocks>::default()
            .idft_batch(RowMajorMatrix::new(values.to_vec(), 1))
            .values
    }

    fn cpu_coset_dft(values: &[Goldilocks]) -> Vec<Goldilocks> {
        Radix2DitParallel::<Goldilocks>::default()
            .coset_dft_batch(
                RowMajorMatrix::new(values.to_vec(), 1),
                Goldilocks::GENERATOR,
            )
            .to_row_major_matrix()
            .values
    }

    fn assert_canonical(values: &[Goldilocks], what: &str) {
        for (i, &word) in raw_words(values).iter().enumerate() {
            assert!(
                word < Goldilocks::ORDER_U64,
                "{what}: word {i} is not canonical: {word:#x}"
            );
        }
    }

    #[test]
    fn an_unknown_device_ordinal_is_rejected_before_any_launch() {
        // cudaErrorInvalidDevice is 101; the adapter answers it for an
        // ordinal upstream did not enumerate, without touching the buffer.
        let input = random(8, 0xde);
        let mut values = input.clone();
        let status = try_ntt_host(1 << 20, &mut values, Order::NN, Direction::Forward, false);
        assert_eq!(status, Err(101));
        assert_eq!(values, input);
    }

    /// One resident LDE both ways, comparing stored words and the Merkle root
    /// of a commitment over the matrix.
    fn resident_lde_both_ways(
        matrix: &RowMajorMatrix<Goldilocks>,
        added_bits: usize,
        shift: Goldilocks,
    ) {
        let dft = super::super::CudaDft::new(0);
        select_backend(Backend::Legacy);
        let legacy = dft.coset_lde_batch_resident(matrix, added_bits, shift);
        let legacy_rows = legacy.to_row_major_matrix();
        select_backend(Backend::SpparkAllHeights);
        let candidate = dft.coset_lde_batch_resident(matrix, added_bits, shift);
        let candidate_rows = candidate.to_row_major_matrix();
        select_backend(Backend::Legacy);
        assert_eq!(
            raw_words(&candidate_rows.values),
            raw_words(&legacy_rows.values),
            "height {} width {} blowup {added_bits}",
            matrix.height(),
            matrix.width()
        );
    }

    #[test]
    fn resident_lde_matches_the_first_party_kernels_bit_for_bit() {
        let _guard = backend_lock();
        let mut rng = SmallRng::seed_from_u64(0x1de5);
        for log_height in [0usize, 1, 2, 5, 8, 12, 14] {
            for added_bits in [0usize, 1, 2, 3] {
                for width in [1usize, 2, 3, 7, 8, 33] {
                    let height = 1 << log_height;
                    let matrix = RowMajorMatrix::new(
                        (0..height * width).map(|_| rng.random()).collect(),
                        width,
                    );
                    resident_lde_both_ways(&matrix, added_bits, Goldilocks::GENERATOR);
                }
            }
        }
    }

    #[test]
    fn resident_lde_reduces_raw_representatives_like_the_first_party_kernels() {
        let _guard = backend_lock();
        let p = Goldilocks::ORDER_U64;
        let words = [0u64, 1, p - 1, p, p + 1, u64::MAX, 7, p + 7];
        let height = 1usize << 10;
        let width = 9;
        // SAFETY: the test reinterprets raw words as field elements on purpose.
        let values: Vec<Goldilocks> = (0..height * width)
            .map(|i| unsafe { core::mem::transmute::<u64, Goldilocks>(words[i % words.len()]) })
            .collect();
        for shift in [
            Goldilocks::GENERATOR,
            Goldilocks::ONE,
            Goldilocks::from_u64(11),
        ] {
            resident_lde_both_ways(&RowMajorMatrix::new(values.clone(), width), 2, shift);
        }
    }

    #[test]
    fn resident_lde_panels_narrower_than_the_matrix_cover_every_column() {
        let _guard = backend_lock();
        // A 2^12 x 33 matrix at blowup 2 needs 16 KiB x 2 per column, so a
        // 256 KiB budget forces panels of a few columns.
        let mut rng = SmallRng::seed_from_u64(0x9a7e);
        let previous = std::env::var("MULTI_STARK_SPPARK_PANEL_BYTES").ok();
        unsafe { std::env::set_var("MULTI_STARK_SPPARK_PANEL_BYTES", "262144") };
        let matrix = RowMajorMatrix::new((0..(1 << 12) * 33).map(|_| rng.random()).collect(), 33);
        resident_lde_both_ways(&matrix, 2, Goldilocks::GENERATOR);
        match previous {
            Some(value) => unsafe { std::env::set_var("MULTI_STARK_SPPARK_PANEL_BYTES", value) },
            None => unsafe { std::env::remove_var("MULTI_STARK_SPPARK_PANEL_BYTES") },
        }
    }

    #[test]
    fn general_dft_matches_the_first_party_kernels_bit_for_bit() {
        let _guard = backend_lock();
        // Shapes above the CUDA DFT threshold of 2^15 cells.
        let mut rng = SmallRng::seed_from_u64(0xdf7);
        let dft = super::super::CudaDft::new(0);
        for (log_height, width) in [(15usize, 1usize), (12, 8), (10, 33), (16, 2), (11, 129)] {
            let matrix = RowMajorMatrix::new(
                (0..(1 << log_height) * width)
                    .map(|_| rng.random())
                    .collect(),
                width,
            );
            select_backend(Backend::Legacy);
            let legacy = dft.dft_batch(matrix.clone()).to_row_major_matrix();
            select_backend(Backend::SpparkAllHeights);
            let candidate = dft.dft_batch(matrix).to_row_major_matrix();
            select_backend(Backend::Legacy);
            assert_eq!(
                raw_words(&candidate.values),
                raw_words(&legacy.values),
                "2^{log_height} x {width}"
            );
        }
    }

    #[test]
    fn host_coset_lde_matches_the_first_party_kernels_bit_for_bit() {
        let _guard = backend_lock();
        // The host entry takes matrices of width at most two whose extended
        // height reaches 2^15.
        let mut rng = SmallRng::seed_from_u64(0x1e5);
        let dft = super::super::CudaDft::new(0);
        let generator = Goldilocks::GENERATOR;
        for (log_height, width, added_bits) in [
            (14usize, 1usize, 1usize),
            (14, 2, 2),
            (15, 2, 3),
            (16, 1, 1),
        ] {
            let matrix = RowMajorMatrix::new(
                (0..(1 << log_height) * width)
                    .map(|_| rng.random())
                    .collect(),
                width,
            );
            select_backend(Backend::Legacy);
            let legacy = dft
                .coset_lde_batch(matrix.clone(), added_bits, generator)
                .to_row_major_matrix();
            select_backend(Backend::SpparkAllHeights);
            let candidate = dft
                .coset_lde_batch(matrix, added_bits, generator)
                .to_row_major_matrix();
            select_backend(Backend::Legacy);
            assert_eq!(
                raw_words(&candidate.values),
                raw_words(&legacy.values),
                "2^{log_height} x {width} blowup {added_bits}"
            );
        }
    }

    /// The fork's batched launch sequence transforms every vector of a
    /// batch exactly as the single-vector entry does, for each order,
    /// direction and coset setting, with padding between the vectors.
    #[test]
    fn a_batch_of_vectors_matches_the_vectors_one_by_one() {
        let mut rng = SmallRng::seed_from_u64(0xba7c);
        for (lg, count, padding) in [(4u32, 3u32, 0usize), (10, 7, 8), (12, 5, 0), (16, 3, 16)] {
            let length = 1usize << lg;
            let stride = length + padding;
            let batch = Batch { lg, count, stride };
            let values: Vec<Goldilocks> =
                (0..count as usize * stride).map(|_| rng.random()).collect();
            for order in [Order::NN, Order::NR, Order::RN, Order::RR] {
                for direction in [Direction::Forward, Direction::Inverse] {
                    for coset in [false, true] {
                        let mut batched = values.clone();
                        try_ntt_batch_host(0, &mut batched, batch, order, direction, coset)
                            .unwrap();
                        let mut expected = values.clone();
                        for vector in expected.chunks_exact_mut(stride) {
                            ntt_host(0, &mut vector[..length], order, direction, coset);
                        }
                        assert_eq!(
                            raw_words(&batched),
                            raw_words(&expected),
                            "2^{lg} x {count} stride {stride} {order:?} {direction:?} coset {coset}"
                        );
                    }
                }
            }
        }
    }

    /// With column batching switched off the panel path launches its
    /// columns one by one and still matches the first-party kernels.
    #[test]
    fn unbatched_columns_match_the_first_party_kernels() {
        let _guard = backend_lock();
        let previous = std::env::var("MULTI_STARK_SPPARK_BATCH_BYTES").ok();
        unsafe { std::env::set_var("MULTI_STARK_SPPARK_BATCH_BYTES", "0") };
        let mut rng = SmallRng::seed_from_u64(0x0ff);
        for (log_height, width, added_bits) in [(10usize, 5usize, 2usize), (12, 33, 1), (8, 3, 3)] {
            let matrix = RowMajorMatrix::new(
                (0..(1 << log_height) * width)
                    .map(|_| rng.random())
                    .collect(),
                width,
            );
            resident_lde_both_ways(&matrix, added_bits, Goldilocks::GENERATOR);
        }
        match previous {
            Some(value) => unsafe { std::env::set_var("MULTI_STARK_SPPARK_BATCH_BYTES", value) },
            None => unsafe { std::env::remove_var("MULTI_STARK_SPPARK_BATCH_BYTES") },
        }
    }

    #[test]
    fn the_height_threshold_and_the_panel_budget_decide_dispatch_and_scratch() {
        let _guard = backend_lock();
        select_backend(Backend::Sppark);
        assert!(
            !takes(1 << 12),
            "short transforms stay on the first-party kernels"
        );
        assert!(!takes(1 << 17));
        assert!(takes(1 << 18));
        assert!(takes(1 << 20));
        assert!(takes_lde(1 << 20, 533, 2));
        assert!(takes_forward(1 << 20, 2));
        assert!(
            !takes_lde(1 << 24, 6, 5),
            "2^29 output rows are beyond the compiled domain"
        );
        assert!(!takes_forward(1 << 29, 2));
        assert_eq!(panel_bytes(1 << 12, 533, 2), 0);
        assert_eq!(panel_bytes(1 << 24, 6, 5), 0);
        // A forward transform's panel is one column set at the height.
        assert_eq!(forward_panel_bytes(1 << 22, 2), 2 * (1 << 22) * 8);
        // 2^20 rows, 533 columns, blowup 4: (2^20 + 2^22) x 8 bytes per
        // column is 40 MiB, so a 4 GiB budget admits 102 columns, plus the
        // reversed coset powers.
        let column_bytes = ((1 << 20) + (1 << 22)) * 8;
        assert_eq!(panel_bytes(1 << 20, 533, 2), 102 * column_bytes + (1 << 20) * 8);
        select_backend(Backend::SpparkAllHeights);
        assert!(takes(2));
        select_backend(Backend::Legacy);
        assert!(!takes(1 << 24));
        assert_eq!(panel_bytes(1 << 24, 6, 2), 0);
    }

    #[test]
    fn domain_limit_covers_the_trace_cap_and_its_lde() {
        assert!(
            max_log_domain() >= 26,
            "the prover extends 2^24 traces by 4"
        );
    }

    #[test]
    fn forward_natural_matches_the_cpu_dft() {
        for lg in LOGS {
            let input = random(lg, 0xf0 + lg as u64);
            let expected = cpu_dft(&input);
            let mut actual = input.clone();
            ntt_host(0, &mut actual, Order::NN, Direction::Forward, false);
            assert_eq!(actual, expected, "lg {lg}");
            assert_canonical(&actual, "forward NN");
        }
    }

    #[test]
    fn reversed_orders_are_the_bit_reversal_of_natural_ones() {
        for lg in LOGS {
            let input = random(lg, 0xb1 + lg as u64);
            let mut natural = cpu_dft(&input);
            let mut nr = input.clone();
            ntt_host(0, &mut nr, Order::NR, Direction::Forward, false);
            reverse_slice_index_bits(&mut natural);
            assert_eq!(nr, natural, "NR at lg {lg}");
            // RN: bit-reversed input yields the natural-order output.
            let mut reversed_input = input.clone();
            reverse_slice_index_bits(&mut reversed_input);
            ntt_host(0, &mut reversed_input, Order::RN, Direction::Forward, false);
            assert_eq!(reversed_input, cpu_dft(&input), "RN at lg {lg}");
        }
    }

    #[test]
    fn inverse_is_normalized_upstream() {
        for lg in LOGS {
            let input = random(lg, 0x1d + lg as u64);
            let mut actual = input.clone();
            ntt_host(0, &mut actual, Order::NN, Direction::Inverse, false);
            assert_eq!(actual, cpu_idft(&input), "inverse NN at lg {lg}");
            // A forward NR followed by an inverse RN is the identity, so the
            // prover's inverse-then-forward pair needs no scaling of its own.
            let mut round_trip = input.clone();
            ntt_host(0, &mut round_trip, Order::NR, Direction::Forward, false);
            ntt_host(0, &mut round_trip, Order::RN, Direction::Inverse, false);
            assert_eq!(round_trip, input, "round trip at lg {lg}");
        }
    }

    #[test]
    fn coset_transform_uses_the_field_generator() {
        for lg in LOGS {
            let input = random(lg, 0xc0 + lg as u64);
            let mut actual = input.clone();
            ntt_host(0, &mut actual, Order::NN, Direction::Forward, true);
            assert_eq!(actual, cpu_coset_dft(&input), "coset forward NN at lg {lg}");
        }
    }

    #[test]
    fn extreme_canonical_values_transform_and_stay_canonical() {
        // Upstream takes canonical words only: raw representatives at or
        // above the modulus, which the first-party kernels reduce lazily,
        // transform to different values, so the adapter's callers reduce
        // first. Canonical extremes must survive unchanged.
        let p = Goldilocks::ORDER_U64;
        let words: Vec<u64> = [0, 1, p - 1, 7, p - 7, 1 << 32, (1 << 32) - 1, p - (1 << 32)]
            .into_iter()
            .cycle()
            .take(1 << 10)
            .collect();
        let input: Vec<Goldilocks> = words.iter().map(|&w| Goldilocks::from_u64(w)).collect();
        assert_eq!(
            raw_words(&input),
            &words[..],
            "the inputs are stored canonically"
        );
        let mut actual = input.clone();
        ntt_host(0, &mut actual, Order::NN, Direction::Forward, false);
        assert_eq!(actual, cpu_dft(&input));
        assert_canonical(&actual, "forward of canonical extremes");
        ntt_host(0, &mut actual, Order::NN, Direction::Inverse, false);
        assert_eq!(actual, input);
        assert_canonical(&actual, "inverse of canonical extremes");
    }
}
