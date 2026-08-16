//! Crate field traits ([`crate::traits::Field`] and friends) for the
//! Plonky3 fields.
//!
//! Scalars implement the traits directly (macro-generated delegation —
//! local trait on a foreign type, no wrappers needed). Packings are
//! wrapped in the generic newtypes [`P3Packing`] / [`P3ExtPacking`] so
//! every impl is constructor-headed: coherence never has to reason
//! about which foreign types might implement which foreign traits, and
//! the arkworks adapter's scalar packings can never collide with these.
//! Both newtypes are `repr(transparent)`, so the slice reinterpretation
//! `from_slice`/`as_basis_coefficients_slice` rely on is layout-sound
//! and the delegation is zero-cost.

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
        matrix
            .vertically_packed_row_pair::<<F as P3Field>::Packing>(i, step)
            .into_iter()
            .map(Self)
            .collect()
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
            const W: $base = <$base as p3_field::extension::BinomiallyExtendable<$d>>::W;
            type ExtPacking = $crate::p3_adapter::field::P3ExtPacking<$base, $ext>;

            #[inline]
            fn as_basis_coefficients_slice(&self) -> &[$base] {
                p3_field::BasedVectorSpace::<$base>::as_basis_coefficients_slice(self)
            }

            #[inline]
            fn from_basis_coefficients_fn(f: impl FnMut(usize) -> $base) -> Self {
                <$ext as p3_field::BasedVectorSpace<$base>>::from_basis_coefficients_fn(f)
            }
        }
    };
}

pub(crate) use {impl_extension_via_p3, impl_field_via_p3, impl_two_adic_via_p3};

impl_field_via_p3!(p3_goldilocks::Goldilocks);
impl_two_adic_via_p3!(p3_goldilocks::Goldilocks);
impl_field_via_p3!(p3_field::extension::BinomialExtensionField<p3_goldilocks::Goldilocks, 2>);
impl_extension_via_p3!(
    p3_goldilocks::Goldilocks,
    p3_field::extension::BinomialExtensionField<p3_goldilocks::Goldilocks, 2>,
    2
);
