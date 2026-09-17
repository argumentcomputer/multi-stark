//! sppark's Goldilocks NTT through the device-pointer adapter in
//! `cuda/sppark_ntt.cu`, with shared immutable transform plans.
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

/// The allocation and launch shape shared by admission and CUDA execution.
#[derive(Debug)]
#[repr(C)]
pub(crate) struct RawPlan {
    height: usize,
    width: usize,
    extended_height: usize,
    columns: usize,
    inverse_group: usize,
    forward_group: usize,
    scratch_bytes: usize,
    shift_powers: *const u64,
}

#[derive(Debug)]
pub(crate) struct TransformPlan {
    raw: RawPlan,
    powers: Option<std::sync::Arc<[Goldilocks]>>,
}

// SAFETY: the pointer references immutable host powers owned by this plan.
// CUDA entry points borrow it only for their synchronous host-side upload.
unsafe impl Send for TransformPlan {}
unsafe impl Sync for TransformPlan {}

impl TransformPlan {
    pub(crate) fn raw(&self) -> &RawPlan {
        &self.raw
    }

    pub(crate) fn scratch_bytes(&self) -> usize {
        self.raw.scratch_bytes
    }

    pub(crate) fn constant_bytes(&self) -> usize {
        self.powers.as_ref().map_or(0, |powers| powers.len() * 8)
    }
}

type PlanKey = (usize, usize, usize, Option<u64>);
type PowerCache = std::collections::BTreeMap<(usize, u64), std::sync::Arc<[Goldilocks]>>;

#[derive(Default, Debug)]
struct Plans {
    shapes: std::collections::BTreeMap<PlanKey, std::sync::Arc<TransformPlan>>,
    powers: PowerCache,
}

/// Immutable per-device settings and shared plans; environment changes cannot
/// change the workspace between admission and execution of a transform.
#[derive(Clone, Debug)]
pub(crate) struct Planner {
    panel_bytes: usize,
    batch_bytes: usize,
    plans: std::sync::Arc<std::sync::RwLock<Plans>>,
}

impl Planner {
    pub(crate) fn new(device: i32) -> Self {
        fn setting(name: &str) -> Option<usize> {
            std::env::var(name).ok().map(|value| {
                value
                    .parse()
                    .unwrap_or_else(|_| panic!("{name} must be a non-negative byte count"))
            })
        }
        let mut l2_bytes = 0;
        check_cuda(
            unsafe { multi_stark_sppark_l2_bytes(device, &mut l2_bytes) },
            "NTT device properties",
        );
        Self::with_budgets(
            setting("MULTI_STARK_SPPARK_PANEL_BYTES")
                .filter(|&n| n != 0)
                .unwrap_or(4usize << 30),
            setting("MULTI_STARK_SPPARK_BATCH_BYTES").unwrap_or(l2_bytes),
        )
    }

    pub(crate) fn with_budgets(panel_bytes: usize, batch_bytes: usize) -> Self {
        Self {
            panel_bytes,
            batch_bytes,
            plans: Default::default(),
        }
    }

    pub(crate) fn plan(
        &self,
        height: usize,
        width: usize,
        added_bits: usize,
        shift: Option<Goldilocks>,
    ) -> std::sync::Arc<TransformPlan> {
        use p3_field::{PrimeCharacteristicRing, PrimeField64};
        use std::sync::Arc;
        assert!(
            height.is_power_of_two(),
            "NTT height must be a power of two"
        );
        assert!(width > 0, "NTT width must be positive");
        let log = height.trailing_zeros() as usize;
        assert!(
            added_bits <= max_log_domain() && log + added_bits <= max_log_domain(),
            "NTT exceeds sppark's compiled domain"
        );
        assert!(shift.is_some() || added_bits == 0);
        let extended_height = height << added_bits;
        extended_height
            .checked_mul(width)
            .and_then(|n| n.checked_mul(8))
            .expect("NTT matrix byte count overflows usize");
        let column_bytes = extended_height
            .checked_add(if shift.is_some() { height } else { 0 })
            .and_then(|n| n.checked_mul(8))
            .expect("NTT column byte count overflows usize");
        assert!(
            column_bytes <= self.panel_bytes,
            "MULTI_STARK_SPPARK_PANEL_BYTES cannot hold one NTT column: need {column_bytes} bytes, budget {}",
            self.panel_bytes
        );
        let key = (
            height,
            width,
            added_bits,
            shift.map(|s| s.as_canonical_u64()),
        );
        if let Some(plan) = self
            .plans
            .read()
            .expect("NTT plan cache poisoned")
            .shapes
            .get(&key)
        {
            return Arc::clone(plan);
        }
        let mut cache = self.plans.write().expect("NTT plan cache poisoned");
        if let Some(plan) = cache.shapes.get(&key) {
            return Arc::clone(plan);
        }
        let powers = shift.map(|shift| {
            Arc::clone(
                cache
                    .powers
                    .entry((height, shift.as_canonical_u64()))
                    .or_insert_with(|| shift.powers().take(height).collect().into()),
            )
        });
        let columns = (self.panel_bytes / column_bytes).min(width).min(65535);
        let group = |rows: usize| (self.batch_bytes / (rows * 8)).max(1).min(columns);
        let raw = RawPlan {
            height,
            width,
            extended_height,
            columns,
            inverse_group: group(height),
            forward_group: group(extended_height),
            scratch_bytes: columns * column_bytes,
            shift_powers: powers
                .as_ref()
                .map_or(core::ptr::null(), |p| p.as_ptr().cast()),
        };
        let plan = Arc::new(TransformPlan { raw, powers });
        cache.shapes.insert(key, Arc::clone(&plan));
        plan
    }
}

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
    fn multi_stark_sppark_l2_bytes(device: c_int, bytes: *mut usize) -> c_int;
    #[cfg(test)]
    fn multi_stark_sppark_borrowed_round_trip(device: c_int, values: *mut u64, lg: u32) -> c_int;
    fn multi_stark_sppark_transforms_run() -> u64;
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

/// Number of transform operations launched, including identity shapes.
pub fn transforms_run() -> u64 {
    unsafe { multi_stark_sppark_transforms_run() }
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
    let words = (batch.count as usize)
        .checked_mul(batch.stride)
        .expect("batch words overflow usize");
    assert_eq!(values.len(), words);
    assert!(
        u32::try_from(batch.stride).is_ok(),
        "the adapter indexes vectors with 32 bits"
    );
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
    use p3_matrix::bitrev::BitReversibleMatrix;
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

    fn resident_matches_cpu(
        dft: &super::super::CudaDft,
        matrix: &RowMajorMatrix<Goldilocks>,
        added_bits: usize,
        shift: Goldilocks,
    ) {
        let expected = Radix2DitParallel::<Goldilocks>::default()
            .coset_lde_batch(matrix.clone(), added_bits, shift)
            .bit_reverse_rows()
            .to_row_major_matrix();
        let actual = dft
            .coset_lde_batch_resident(matrix, added_bits, shift)
            .to_row_major_matrix();
        assert_eq!(actual, expected);
        assert_canonical(&actual.values, "resident LDE");
    }

    #[test]
    fn resident_lde_reduces_raw_representatives() {
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
            resident_matches_cpu(
                &super::super::CudaDft::new(0),
                &RowMajorMatrix::new(values.clone(), width),
                2,
                shift,
            );
        }
    }

    #[test]
    fn panel_and_batch_boundaries_match_cpu_without_global_settings() {
        let mut rng = SmallRng::seed_from_u64(0x9a7e);
        let matrix = RowMajorMatrix::new((0..(1 << 12) * 33).map(|_| rng.random()).collect(), 33);
        for batch_bytes in [0, 64 << 10, 64 << 20] {
            let mut dft = super::super::CudaDft::new(0);
            dft.planner = Planner::with_budgets(256 << 10, batch_bytes);
            let plan = dft.lde_plan(matrix.height(), matrix.width(), 2, Goldilocks::GENERATOR);
            assert!(plan.raw.columns < matrix.width());
            resident_matches_cpu(&dft, &matrix, 2, Goldilocks::GENERATOR);
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

    #[test]
    fn plan_reuses_constants_and_rejects_unrepresentable_shapes() {
        use std::sync::Arc;
        let planner = Planner::with_budgets(256 << 10, 64 << 10);
        let plan = planner.plan(1 << 12, 33, 2, Some(Goldilocks::GENERATOR));
        assert!(Arc::ptr_eq(
            &plan,
            &planner.plan(1 << 12, 33, 2, Some(Goldilocks::GENERATOR))
        ));
        assert!(plan.scratch_bytes() <= 256 << 10);
        let narrow = planner.plan(1 << 12, 1, 2, Some(Goldilocks::GENERATOR));
        assert_eq!(plan.raw.shift_powers, narrow.raw.shift_powers);
        for (height, width, bits) in [
            (1 << 29, 1, 0),
            (1 << 24, 1, 5),
            (1 << 20, 1, 2),
            (2, usize::MAX, 0),
        ] {
            assert!(
                std::panic::catch_unwind(|| planner.plan(
                    height,
                    width,
                    bits,
                    Some(Goldilocks::ONE)
                ))
                .is_err()
            );
        }
    }

    #[test]
    fn ffi_rejects_mismatched_transform_plans() {
        let dft = super::super::CudaDft::new(0);
        let input = vec![Goldilocks::ONE; 8 * 3];
        for plan in [
            dft.forward_plan(4, 3),
            dft.forward_plan(8, 4),
            dft.lde_plan(8, 3, 1, Goldilocks::GENERATOR),
            dft.lde_plan(8, 3, 0, Goldilocks::GENERATOR),
        ] {
            let mut values = input.clone();
            let status = unsafe {
                super::super::multi_stark_cuda_dft_batch(
                    0,
                    values.as_mut_ptr().cast(),
                    8,
                    3,
                    plan.raw(),
                )
            };
            assert_eq!(status, 1, "cudaErrorInvalidValue");
            assert_eq!(values, input);
        }
        for plan in [
            dft.lde_plan(4, 3, 2, Goldilocks::GENERATOR),
            dft.lde_plan(8, 4, 1, Goldilocks::GENERATOR),
            dft.lde_plan(8, 3, 2, Goldilocks::GENERATOR),
        ] {
            let mut output = vec![Goldilocks::TWO; 16 * 3];
            let status = unsafe {
                super::super::multi_stark_cuda_coset_lde_batch(
                    0,
                    output.as_mut_ptr().cast(),
                    input.as_ptr().cast(),
                    8,
                    3,
                    1,
                    plan.raw(),
                )
            };
            assert_eq!(status, 1, "cudaErrorInvalidValue");
            assert!(output.iter().all(|v| *v == Goldilocks::TWO));
            let mut handle = core::ptr::null_mut();
            let status = unsafe {
                super::super::multi_stark_cuda_coset_lde_create(
                    0,
                    &mut handle,
                    input.as_ptr().cast(),
                    8,
                    3,
                    1,
                    plan.raw(),
                )
            };
            assert_eq!(status, 1, "cudaErrorInvalidValue");
            assert!(handle.is_null());
        }
    }

    #[test]
    fn borrowed_stream_preserves_order_and_caller_ownership() {
        let mut values = random(12, 0xb0770);
        let expected = values.clone();
        check_cuda(
            unsafe { multi_stark_sppark_borrowed_round_trip(0, values.as_mut_ptr().cast(), 12) },
            "borrowed stream round trip",
        );
        assert_eq!(values, expected);
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
