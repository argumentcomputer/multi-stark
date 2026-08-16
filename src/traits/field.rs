//! The field surface the core actually uses.
//!
//! Three layers, measured off the call sites (see docs/pcs-abstraction.md):
//!
//! - [`Field`]: the scalar base/challenge fields — ring ops, constants,
//!   inversion, small-integer embeddings, powers, and an associated
//!   SIMD packing for the prover's constraint sweep.
//! - [`Algebra`]: the sweep's working types (`W` in `eval.rs`): the base
//!   field itself (witness paths), its packing (quotient domain), or the
//!   challenge field (verifier at zeta). Ring ops among `W`, embedding
//!   from `F`, and scaling by `F`.
//! - [`ExtensionOf`]: the challenge field over the base field, as a
//!   BINOMIAL extension `X^D = W` with `D >= 1`. `D = 1` is first-class
//!   (the field is its own challenge field — the KZG/BLS12-381 case);
//!   the constants replace the old runtime recovery of `W`, which
//!   required `D >= 2`.
//!
//! Backends adapt in: `p3_adapter/field.rs` for Plonky3 fields (packings
//! wrapped in a generic newtype so every impl stays constructor-headed),
//! the arkworks adapter with scalar (width-1) packing.

use core::fmt::Debug;
use core::iter::Iterator;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_matrix::Matrix;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Ring operations shared by every layer, over `Rhs`.
pub trait RingOps<Rhs = Self>:
    Sized
    + Add<Rhs, Output = Self>
    + Sub<Rhs, Output = Self>
    + Mul<Rhs, Output = Self>
    + AddAssign<Rhs>
    + SubAssign<Rhs>
    + MulAssign<Rhs>
{
}
impl<T, Rhs> RingOps<Rhs> for T where
    T: Sized
        + Add<Rhs, Output = Self>
        + Sub<Rhs, Output = Self>
        + Mul<Rhs, Output = Self>
        + AddAssign<Rhs>
        + SubAssign<Rhs>
        + MulAssign<Rhs>
{
}

/// A working type for the constraint sweep: closed ring ops, an
/// embedding from the base field, and scaling by base-field constants.
pub trait Algebra<F>:
    Copy + Send + Sync + RingOps + Neg<Output = Self> + From<F> + Mul<F, Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
}

/// A scalar field.
pub trait Field:
    'static
    + Copy
    + Send
    + Sync
    + Eq
    + core::hash::Hash
    + Debug
    + Algebra<Self>
    + Serialize
    + DeserializeOwned
{
    /// SIMD packing for the prover's sweep; scalar (width 1) is a valid
    /// implementation.
    type Packing: Packed<Self>;

    /// Multiplicative inverse; implementations may define `inverse(0)`
    /// freely (the protocol never inverts zero on honest paths).
    fn inverse(&self) -> Self;

    fn exp_u64(&self, exp: u64) -> Self;

    /// `self^(2^log)` by repeated squaring.
    fn exp_power_of_2(&self, log: usize) -> Self {
        let mut r = *self;
        for _ in 0..log {
            r = r * r;
        }
        r
    }

    fn from_bool(b: bool) -> Self;
    fn from_u8(x: u8) -> Self;
    fn from_u32(x: u32) -> Self;
    fn from_u64(x: u64) -> Self;
    fn from_usize(x: usize) -> Self;

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    /// The infinite iterator `1, x, x^2, ...`.
    fn powers(&self) -> Powers<Self> {
        Powers {
            base: *self,
            current: Self::ONE,
        }
    }

    fn zero_vec(len: usize) -> Vec<Self> {
        vec![Self::ZERO; len]
    }
}

/// `1, x, x^2, ...` — see [`Field::powers`].
#[derive(Clone, Debug)]
pub struct Powers<F> {
    pub base: F,
    pub current: F,
}

impl<F: Field> Iterator for Powers<F> {
    type Item = F;

    fn next(&mut self) -> Option<F> {
        let result = self.current;
        self.current *= self.base;
        Some(result)
    }
}

/// A field with a large power-of-two subgroup.
pub trait TwoAdicField: Field {
    const TWO_ADICITY: usize;

    /// A generator of the order-`2^bits` subgroup.
    fn two_adic_generator(bits: usize) -> Self;
}

/// SIMD packet of base-field values. Scalar fields implement this for
/// themselves with `WIDTH = 1`.
pub trait Packed<F: Field>: Algebra<F> {
    const WIDTH: usize;

    /// Reinterpret `WIDTH` contiguous scalars as one packet (no copy).
    fn from_slice(slice: &[F]) -> &Self;

    /// The packet's scalars.
    fn as_slice(&self) -> &[F];

    /// `sum_i coeffs[i] * vecs[i]` — the alpha-fold inner loop.
    fn batched_linear_combination(vecs: &[Self], coeffs: &[F]) -> Self {
        let mut acc = Self::ZERO;
        for (v, c) in vecs.iter().zip(coeffs) {
            acc += *v * *c;
        }
        acc
    }

    /// The packed two-row window `[rows i.., rows (i+step)..]` of a
    /// matrix, `WIDTH` vertically adjacent rows per packet — the sweep's
    /// input gather. Row indices wrap modulo the height.
    fn packed_row_pair<M: Matrix<F>>(matrix: &M, i: usize, step: usize) -> Vec<Self>;
}

/// The challenge field as a binomial extension `X^D = W` of the base
/// field, `D >= 1`. For `D = 1` the field is its own challenge field:
/// `W` is unused and the basis is the identity.
pub trait ExtensionOf<F: Field>: Field + Algebra<F> {
    const D: usize;
    /// The binomial modulus constant (`X^D = W`); unused when `D = 1`.
    const W: F;

    /// Packed representation: `D` base packings.
    type ExtPacking: PackedExtension<F, Self>;

    fn as_basis_coefficients_slice(&self) -> &[F];
    fn from_basis_coefficients_fn(f: impl FnMut(usize) -> F) -> Self;
}

/// Packed challenge values: `D` base packets, used to reassemble the
/// sweep's folded accumulator and scale it by packed base values.
pub trait PackedExtension<F: Field, EF>:
    Copy + Send + Sync + Mul<F::Packing, Output = Self>
{
    fn from_basis_coefficients_fn(f: impl FnMut(usize) -> F::Packing) -> Self;
    fn as_basis_coefficients_slice(&self) -> &[F::Packing];
}

/// Flatten a challenge-valued matrix to base columns (each value becomes
/// its `D` basis coordinates). Identity-shaped when `D = 1`.
pub fn flatten_to_base<F: Field, EF: ExtensionOf<F>>(
    m: p3_matrix::dense::RowMajorMatrix<EF>,
) -> p3_matrix::dense::RowMajorMatrix<F> {
    let width = m.width * EF::D;
    let values = m
        .values
        .into_iter()
        .flat_map(|e| e.as_basis_coefficients_slice().to_vec())
        .collect();
    p3_matrix::dense::RowMajorMatrix::new(values, width)
}

/// Batch inversion by Montgomery's trick: 3 multiplications per element
/// and a single field inversion. Zero entries are passed through as zero
/// (matching the reference semantics the lookup witness relies on).
pub fn batch_inverse<F: Field>(values: &[F]) -> Vec<F> {
    let mut prefix = Vec::with_capacity(values.len());
    let mut acc = F::ONE;
    for &v in values {
        prefix.push(acc);
        if !v.is_zero() {
            acc *= v;
        }
    }
    let mut inv = acc.inverse();
    let mut out = vec![F::ZERO; values.len()];
    for i in (0..values.len()).rev() {
        if !values[i].is_zero() {
            out[i] = prefix[i] * inv;
            inv *= values[i];
        }
    }
    out
}
