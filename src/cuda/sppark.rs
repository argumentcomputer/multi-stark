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

use p3_field::PrimeField64;
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
/// generator.
pub fn ntt_host(
    device: i32,
    values: &mut [Goldilocks],
    order: Order,
    direction: Direction,
    coset: bool,
) {
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
    check_cuda(status, "sppark host transform");
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
    use p3_field::{Field, PrimeCharacteristicRing};
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
