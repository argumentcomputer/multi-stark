//! Field traits for Plonky3 scalars and transparent SIMD packing wrappers.
//! The wrappers keep implementations disjoint from external field adapters.

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{
    Algebra as P3Algebra, BasedVectorSpace, ExtensionField as P3ExtensionField, Field as P3Field,
    PackedValue, PrimeCharacteristicRing,
};
use p3_matrix::Matrix;

use crate::traits::{Algebra, Field, Packed, PackedExtension};

/// A p3 SIMD packing, newtyped so impls stay constructor-headed.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct P3Packing<F: P3Field>(pub F::Packing);

/// A p3 packed extension (D base packings), newtyped likewise.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct P3ExtPacking<F: P3Field, EF: P3ExtensionField<F>>(pub EF::ExtensionPacking);

// ---- ring ops for P3Packing, by delegation ----

impl<F: P3Field> Add for P3Packing<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}
impl<F: P3Field> Sub for P3Packing<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}
impl<F: P3Field> Mul for P3Packing<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}
impl<F: P3Field> Neg for P3Packing<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
impl<F: P3Field> AddAssign for P3Packing<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl<F: P3Field> SubAssign for P3Packing<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}
impl<F: P3Field> MulAssign for P3Packing<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}
impl<F: P3Field> From<F> for P3Packing<F> {
    #[inline]
    fn from(f: F) -> Self {
        Self(F::Packing::from(f))
    }
}
impl<F: P3Field> Mul<F> for P3Packing<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self(self.0 * rhs)
    }
}

impl<F: P3Field + Field> Algebra<F> for P3Packing<F> {
    const ZERO: Self = Self(<<F as P3Field>::Packing as PrimeCharacteristicRing>::ZERO);
    const ONE: Self = Self(<<F as P3Field>::Packing as PrimeCharacteristicRing>::ONE);
}

impl<F: P3Field + Field> Packed<F> for P3Packing<F> {
    const WIDTH: usize = <<F as P3Field>::Packing as PackedValue>::WIDTH;

    #[inline]
    fn from_slice(slice: &[F]) -> &Self {
        let inner = <<F as P3Field>::Packing as PackedValue>::from_slice(slice);
        // SAFETY: repr(transparent) over F::Packing.
        unsafe { &*(inner as *const <F as P3Field>::Packing).cast::<Self>() }
    }

    #[inline]
    fn as_slice(&self) -> &[F] {
        self.0.as_slice()
    }

    #[inline]
    fn batched_linear_combination(vecs: &[Self], coeffs: &[F]) -> Self {
        // SAFETY: repr(transparent) — &[P3Packing<F>] and &[F::Packing]
        // have identical layout.
        let inner: &[<F as P3Field>::Packing] = unsafe {
            core::slice::from_raw_parts(vecs.as_ptr().cast::<<F as P3Field>::Packing>(), vecs.len())
        };
        Self(<<F as P3Field>::Packing as P3Algebra<F>>::batched_linear_combination(inner, coeffs))
    }

    #[inline]
    fn packed_row_pair<M: Matrix<F>>(matrix: &M, i: usize, step: usize) -> Vec<Self> {
        let width = matrix.width();
        let mut out = vec![Self::ZERO; 2 * width];
        Self::fill_packed_row(matrix, i, &mut out[..width]);
        Self::fill_packed_row(matrix, i + step, &mut out[width..]);
        out
    }

    #[inline]
    fn fill_packed_row<M: Matrix<F>>(matrix: &M, i: usize, out: &mut [Self]) {
        let cols = out.len();
        debug_assert!(cols <= matrix.width());
        if cols == 0 {
            return;
        }
        match Self::WIDTH {
            1 => fill_packed_row_fixed::<F, M, 1>(matrix, i, out),
            2 => fill_packed_row_fixed::<F, M, 2>(matrix, i, out),
            4 => fill_packed_row_fixed::<F, M, 4>(matrix, i, out),
            8 => fill_packed_row_fixed::<F, M, 8>(matrix, i, out),
            16 => fill_packed_row_fixed::<F, M, 16>(matrix, i, out),
            32 => fill_packed_row_fixed::<F, M, 32>(matrix, i, out),
            _ => {
                let rows = matrix.wrapping_row_slices(i, Self::WIDTH);
                for (c, dst) in out.iter_mut().enumerate() {
                    *dst = Self(<<F as P3Field>::Packing as PackedValue>::from_fn(|lane| {
                        rows[lane][c]
                    }));
                }
            }
        }
    }
}

#[inline]
fn fill_packed_row_fixed<F: P3Field + Field, M: Matrix<F>, const W: usize>(
    matrix: &M,
    i: usize,
    out: &mut [P3Packing<F>],
) {
    debug_assert_eq!(W, P3Packing::<F>::WIDTH);
    let cols = out.len();
    let height = matrix.height();
    let mask = height.is_power_of_two().then_some(height - 1);
    let rows = core::array::from_fn::<_, W, _>(|lane| {
        let r = match mask {
            Some(m) => (i + lane) & m,
            None => (i + lane) % height,
        };
        // SAFETY: `r < height` and `cols <= matrix.width()`.
        unsafe { matrix.row_subslice_unchecked(r, 0, cols) }
    });
    let slices: [&[F]; W] = core::array::from_fn(|lane| {
        let slice: &[F] = &rows[lane];
        assert!(slice.len() >= cols);
        slice
    });
    for (c, dst) in out.iter_mut().enumerate() {
        // SAFETY: `c < cols <= slices[lane].len()` for all `lane < W == WIDTH`.
        *dst = P3Packing(<<F as P3Field>::Packing as PackedValue>::from_fn(
            |lane| unsafe { *slices[lane].get_unchecked(c) },
        ));
    }
}

// ---- packed extension, by delegation ----

impl<F: P3Field + Field, EF: P3ExtensionField<F>> Mul<P3Packing<F>> for P3ExtPacking<F, EF> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: P3Packing<F>) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl<F, EF> PackedExtension<F, EF> for P3ExtPacking<F, EF>
where
    F: P3Field + Field<Packing = P3Packing<F>>,
    EF: P3ExtensionField<F> + Field,
{
    #[inline]
    fn from_basis_coefficients_fn(mut f: impl FnMut(usize) -> P3Packing<F>) -> Self {
        Self(<EF::ExtensionPacking as BasedVectorSpace<
            <F as P3Field>::Packing,
        >>::from_basis_coefficients_fn(|i| f(i).0))
    }

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[P3Packing<F>] {
        let inner: &[<F as P3Field>::Packing] = <EF::ExtensionPacking as BasedVectorSpace<
            <F as P3Field>::Packing,
        >>::as_basis_coefficients_slice(&self.0);
        // SAFETY: repr(transparent) — identical slice layout.
        unsafe { core::slice::from_raw_parts(inner.as_ptr().cast::<P3Packing<F>>(), inner.len()) }
    }
}

// ---- scalar field impls, macro-generated delegation ----

/// Implement the crate field traits for a concrete p3 field.
/// Absolute paths throughout: `macro_rules` resolves names at the
/// expansion site.
macro_rules! impl_field_via_p3 {
    ($f:ty) => {
        impl $crate::traits::Algebra<$f> for $f {
            const ZERO: Self = <$f as p3_field::PrimeCharacteristicRing>::ZERO;
            const ONE: Self = <$f as p3_field::PrimeCharacteristicRing>::ONE;
        }

        impl $crate::traits::Field for $f {
            type Packing = $crate::p3_adapter::field::P3Packing<$f>;
            const TWO: Self = <$f as p3_field::PrimeCharacteristicRing>::TWO;
            const NEG_ONE: Self = <$f as p3_field::PrimeCharacteristicRing>::NEG_ONE;
            fn prime_order_exceeds(bound: usize) -> bool {
                <<$f as p3_field::PrimeCharacteristicRing>::PrimeSubfield as p3_field::Field>::order() > bound.into()
            }

            #[inline]
            fn inverse(&self) -> Self {
                p3_field::Field::inverse(self)
            }

            #[inline]
            fn exp_u64(&self, exp: u64) -> Self {
                p3_field::PrimeCharacteristicRing::exp_u64(self, exp)
            }

            #[inline]
            fn from_bool(b: bool) -> Self {
                p3_field::PrimeCharacteristicRing::from_bool(b)
            }

            #[inline]
            fn from_u8(x: u8) -> Self {
                p3_field::PrimeCharacteristicRing::from_u8(x)
            }

            #[inline]
            fn from_u32(x: u32) -> Self {
                p3_field::PrimeCharacteristicRing::from_u32(x)
            }

            #[inline]
            fn from_u64(x: u64) -> Self {
                p3_field::PrimeCharacteristicRing::from_u64(x)
            }

            #[inline]
            fn from_usize(x: usize) -> Self {
                p3_field::PrimeCharacteristicRing::from_usize(x)
            }

            #[inline]
            fn zero_vec(len: usize) -> Vec<Self> {
                p3_field::PrimeCharacteristicRing::zero_vec(len)
            }
        }
    };
}

/// Implement [`crate::traits::TwoAdicField`] for a concrete p3 field.
macro_rules! impl_two_adic_via_p3 {
    ($f:ty) => {
        impl $crate::traits::TwoAdicField for $f {
            const TWO_ADICITY: usize = <$f as p3_field::TwoAdicField>::TWO_ADICITY;

            #[inline]
            fn two_adic_generator(bits: usize) -> Self {
                <$f as p3_field::TwoAdicField>::two_adic_generator(bits)
            }
        }
    };
}

/// Implement [`ExtensionOf`] (and the base-field [`Algebra`]) for a
/// concrete p3 binomial extension over a concrete base.
macro_rules! impl_extension_via_p3 {
    ($base:ty, $ext:ty, $d:literal) => {
        impl $crate::traits::Algebra<$base> for $ext {
            const ZERO: Self = <$ext as p3_field::PrimeCharacteristicRing>::ZERO;
            const ONE: Self = <$ext as p3_field::PrimeCharacteristicRing>::ONE;
        }

        impl $crate::traits::ExtensionOf<$base> for $ext {
            const D: usize = $d;
            const MODULUS: $crate::traits::ExtensionModulus<$base> =
                $crate::traits::ExtensionModulus::Binomial {
                    w: <$base as p3_field::extension::BinomiallyExtendable<$d>>::W,
                };
            type ExtPacking = $crate::p3_adapter::field::P3ExtPacking<$base, $ext>;

            #[inline]
            fn as_basis_coefficients_slice(&self) -> &[$base] {
                p3_field::BasedVectorSpace::<$base>::as_basis_coefficients_slice(self)
            }

            #[inline]
            fn from_basis_coefficients_fn(f: impl FnMut(usize) -> $base) -> Self {
                <$ext as p3_field::BasedVectorSpace<$base>>::from_basis_coefficients_fn(f)
            }

            #[inline]
            fn flatten_vec_to_base(values: Vec<Self>) -> Vec<$base> {
                <$ext as p3_field::BasedVectorSpace<$base>>::flatten_to_base(values)
            }
        }
    };
}

/// Implement [`ExtensionOf`] (and the base-field [`Algebra`]) for a
/// concrete p3 cubic trinomial extension `X^3 - X - 1` over a concrete base.
macro_rules! impl_cubic_trinomial_extension_via_p3 {
    ($base:ty, $ext:ty) => {
        impl $crate::traits::Algebra<$base> for $ext {
            const ZERO: Self = <$ext as p3_field::PrimeCharacteristicRing>::ZERO;
            const ONE: Self = <$ext as p3_field::PrimeCharacteristicRing>::ONE;
        }

        impl $crate::traits::ExtensionOf<$base> for $ext {
            const D: usize = 3;
            const MODULUS: $crate::traits::ExtensionModulus<$base> =
                $crate::traits::ExtensionModulus::CubicTrinomial;
            type ExtPacking = $crate::p3_adapter::field::P3ExtPacking<$base, $ext>;

            #[inline]
            fn as_basis_coefficients_slice(&self) -> &[$base] {
                p3_field::BasedVectorSpace::<$base>::as_basis_coefficients_slice(self)
            }

            #[inline]
            fn from_basis_coefficients_fn(f: impl FnMut(usize) -> $base) -> Self {
                <$ext as p3_field::BasedVectorSpace<$base>>::from_basis_coefficients_fn(f)
            }

            #[inline]
            fn flatten_vec_to_base(values: Vec<Self>) -> Vec<$base> {
                <$ext as p3_field::BasedVectorSpace<$base>>::flatten_to_base(values)
            }
        }
    };
}

#[cfg(test)]
pub(crate) use {impl_extension_via_p3, impl_field_via_p3, impl_two_adic_via_p3};

impl_field_via_p3!(p3_goldilocks::Goldilocks);
impl_two_adic_via_p3!(p3_goldilocks::Goldilocks);
impl_field_via_p3!(p3_field::extension::BinomialExtensionField<p3_goldilocks::Goldilocks, 2>);
impl_extension_via_p3!(
    p3_goldilocks::Goldilocks,
    p3_field::extension::BinomialExtensionField<p3_goldilocks::Goldilocks, 2>,
    2
);
impl_field_via_p3!(p3_field::extension::CubicTrinomialExtensionField<p3_goldilocks::Goldilocks>);
impl_cubic_trinomial_extension_via_p3!(
    p3_goldilocks::Goldilocks,
    p3_field::extension::CubicTrinomialExtensionField<p3_goldilocks::Goldilocks>
);

impl crate::traits::PrimeField for p3_goldilocks::Goldilocks {
    fn canonical_u64(self) -> Option<u64> {
        Some(p3_field::PrimeField64::as_canonical_u64(&self))
    }
    fn modulus_u64() -> Option<u64> {
        Some(<Self as p3_field::PrimeField64>::ORDER_U64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;

    #[test]
    fn fill_packed_row_matches_vertically_packed_row_pair() {
        type P = P3Packing<Goldilocks>;
        for (height, width) in [(1usize, 5usize), (4, 7), (7, 9), (16, 11)] {
            let values: Vec<Goldilocks> = (0..height * width)
                .map(|i| Goldilocks::new((i * 37 + 13) as u64))
                .collect();
            let mat = RowMajorMatrix::new(values, width);
            for i in [0usize, 1, height / 2, height - 1] {
                for step in [1usize, 2, height] {
                    let expected: Vec<Goldilocks> = mat
                        .vertically_packed_row_pair::<<Goldilocks as P3Field>::Packing>(i, step)
                        .into_iter()
                        .flat_map(|p| p.as_slice().to_vec())
                        .collect();
                    let pair = P::packed_row_pair(&mat, i, step);
                    let got_pair: Vec<Goldilocks> = pair
                        .iter()
                        .flat_map(|p| p.as_slice().iter().copied())
                        .collect();
                    assert_eq!(got_pair, expected);

                    // Verify prefix fill (`cols < width`) matches the first `cols` columns.
                    let prefix_cols = 2.min(width);
                    let mut prefix = vec![P::ZERO; prefix_cols];
                    P::fill_packed_row(&mat, i + step, &mut prefix);
                    for (got, exp) in prefix.iter().zip(&pair[width..width + prefix_cols]) {
                        assert_eq!(got.as_slice(), exp.as_slice());
                    }
                }
            }
        }
    }
}
