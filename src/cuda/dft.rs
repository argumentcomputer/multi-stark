//! [`TwoAdicSubgroupDft`] backed by sppark's batched Goldilocks NTT.

use core::ffi::c_void;
use core::marker::PhantomData;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{PrimeField64, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use sppark::{NTTDirection, NTTInputOutputOrder, NTTType};

use super::ffi::{compute_coset_lde_with_shift, compute_ntt_batch};

/// CUDA-backed batched forward DFT.
///
/// Implements [`TwoAdicSubgroupDft`] for [`Goldilocks`] only. The `F`
/// type parameter exists so this can sit in the same `Dft` slot Plonky3
/// expects, but only the Goldilocks impl is provided.
///
/// `dft_batch` is the only required method on the trait; coset-LDE,
/// inverse, and shift variants compose through the default impls and
/// will route back through `dft_batch` here.
#[derive(Clone, Default, Debug)]
pub struct CudaRadix2Dft<F: TwoAdicField> {
    _phantom: PhantomData<F>,
}

impl TwoAdicSubgroupDft<Goldilocks> for CudaRadix2Dft<Goldilocks> {
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<Goldilocks>>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<Goldilocks>) -> Self::Evaluations {
        let height = mat.height();
        let width = mat.width();
        if height <= 1 {
            return BitReversalPerm::new_view(mat);
        }
        let lg_domain_size = log2_strict_usize(height) as u32;

        // sppark in-place NTT with natural-input / bit-reversed-output.
        // The result matrix is wrapped in `BitReversedMatrixView` so that
        // downstream code reads rows in natural domain-element order.
        let err = unsafe {
            compute_ntt_batch(
                0,
                mat.values.as_mut_ptr().cast::<c_void>(),
                lg_domain_size,
                u32::try_from(width).expect("matrix width exceeds u32::MAX"),
                NTTInputOutputOrder::NR,
                NTTDirection::Forward,
                NTTType::Standard,
            )
        };
        if err.code != 0 {
            panic!("CUDA NTT failed: {}", String::from(err));
        }

        BitReversalPerm::new_view(mat)
    }

    /// Override the trait default to do iDFT + coset shift + zero-pad +
    /// forward DFT entirely on device, saving one host↔device round trip
    /// per commit vs the default impl that composes through `dft_batch`.
    ///
    /// Plonky3 passes `shift: F` (typically `Val::GENERATOR = 7` for
    /// Goldilocks). We forward it as the canonical `u64` to the kernel,
    /// which builds `shift^row` per-row via binary exponentiation on
    /// device.
    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<Goldilocks>,
        added_bits: usize,
        shift: Goldilocks,
    ) -> Self::Evaluations {
        use p3_field::PrimeCharacteristicRing;

        let height = mat.height();
        let width = mat.width();

        // Trivial cases handled in Rust — sppark's NTT assumes
        // `lg_domain_size > 0` and we'd rather not test that boundary.
        if height == 0 {
            let ext_height = 1usize << added_bits;
            return BitReversalPerm::new_view(RowMajorMatrix::new(
                vec![Goldilocks::ZERO; ext_height * width],
                width,
            ));
        }
        if height == 1 {
            // DFT([a]) = [a]; coset shift on row 0 is multiply by
            // shift^0 = 1 (no-op); zero-pad to ext_height; DFT of
            // [a, 0, 0, ...] is the constant vector [a, a, ..., a].
            let ext_height = 1usize << added_bits;
            let mut out = Vec::with_capacity(ext_height * width);
            for _ in 0..ext_height {
                out.extend_from_slice(&mat.values);
            }
            return BitReversalPerm::new_view(RowMajorMatrix::new(out, width));
        }

        let lg_domain_size = log2_strict_usize(height) as u32;
        let ext_height = height << added_bits;
        let mut out = vec![Goldilocks::ZERO; ext_height * width];

        let err = unsafe {
            compute_coset_lde_with_shift(
                0,
                out.as_mut_ptr().cast::<c_void>(),
                mat.values.as_ptr().cast::<c_void>(),
                lg_domain_size,
                u32::try_from(added_bits).expect("added_bits exceeds u32::MAX"),
                u32::try_from(width).expect("matrix width exceeds u32::MAX"),
                shift.as_canonical_u64(),
            )
        };
        if err.code != 0 {
            panic!("CUDA coset LDE failed: {}", String::from(err));
        }

        BitReversalPerm::new_view(RowMajorMatrix::new(out, width))
    }
}
