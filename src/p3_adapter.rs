//! Plonky3 adapter: build [`CircuitInputs`] from a Plonky3-style AIR.
//!
//! [`circuit_inputs_from_air`] runs [`Air::eval`] against a recording
//! builder whose expression type wraps the frontend [`Expr`], so an AIR
//! written against the `p3_air` traits (`builder.main()`, `assert_zero`,
//! `when_transition`, ...) compiles down to the same declarative
//! [`CircuitInputs`] that hand-authored circuits use. The main-trace width
//! and the optional preprocessed trace are taken from the AIR's
//! [`p3_air::BaseAir`] impl.
//!
//! Scope:
//! - Base constraints only. Extension constraints and public inputs are
//!   owned by the lookup argument in this system (`System::new` derives
//!   and synthesizes them), so `ExtensionBuilder` is not implemented and
//!   `public_values()` returns the empty default.
//! - Lookups have no `p3_air` vocabulary; push them onto the returned
//!   `CircuitInputs::lookups` afterwards.
//! - Periodic columns are not supported.

use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_air::{Air, AirBuilder, WindowAccess};
use p3_field::{Algebra, Dup, Field, PrimeCharacteristicRing};

use crate::expr::{Expr, RowOffset, Source};
use crate::system::CircuitInputs;

/// A trace variable: a column reference in the current or next row of the
/// main or preprocessed trace.
#[derive(Debug)]
pub struct P3Var<F> {
    source: Source,
    offset: RowOffset,
    index: u32,
    _phantom: PhantomData<F>,
}

// Manual impls: `derive` would put a bound on `F`, but the phantom
// parameter is only there to tie the variable to its expression type.
impl<F> Copy for P3Var<F> {}
impl<F> Clone for P3Var<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> P3Var<F> {
    const fn new(source: Source, offset: RowOffset, index: u32) -> Self {
        Self {
            source,
            offset,
            index,
            _phantom: PhantomData,
        }
    }
}

/// Expression wrapper implementing the `p3_air`/`p3_field` operator and
/// ring traits over the frontend [`Expr`] tree.
#[derive(Clone, Debug)]
pub struct P3Expr<F>(pub Expr<F>);

impl<F: Field> From<F> for P3Expr<F> {
    fn from(value: F) -> Self {
        Self(Expr::Const(value))
    }
}

impl<F: Field> From<P3Var<F>> for P3Expr<F> {
    fn from(value: P3Var<F>) -> Self {
        Self(Expr::var(value.source, value.offset, value.index))
    }
}

impl<F: Field> Dup for P3Expr<F> {
    fn dup(&self) -> Self {
        self.clone()
    }
}

impl<F: Field> Default for P3Expr<F> {
    fn default() -> Self {
        Self(Expr::Const(F::ZERO))
    }
}

impl<F: Field, T: Into<Self>> Add<T> for P3Expr<F> {
    type Output = Self;

    fn add(self, rhs: T) -> Self {
        Self(self.0 + rhs.into().0)
    }
}

impl<F: Field, T: Into<Self>> AddAssign<T> for P3Expr<F> {
    fn add_assign(&mut self, rhs: T) {
        *self = self.clone() + rhs.into();
    }
}

impl<F: Field, T: Into<Self>> Sum<T> for P3Expr<F> {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|x, y| x + y)
            .unwrap_or(Self::ZERO)
    }
}

impl<F: Field, T: Into<Self>> Sub<T> for P3Expr<F> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self {
        Self(self.0 - rhs.into().0)
    }
}

impl<F: Field, T: Into<Self>> SubAssign<T> for P3Expr<F> {
    fn sub_assign(&mut self, rhs: T) {
        *self = self.clone() - rhs.into();
    }
}

impl<F: Field> Neg for P3Expr<F> {
    type Output = Self;

    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl<F: Field, T: Into<Self>> Mul<T> for P3Expr<F> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Self(self.0 * rhs.into().0)
    }
}

impl<F: Field, T: Into<Self>> MulAssign<T> for P3Expr<F> {
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone() * rhs.into();
    }
}

impl<F: Field, T: Into<Self>> Product<T> for P3Expr<F> {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|x, y| x * y)
            .unwrap_or(Self::ONE)
    }
}

impl<F: Field> PrimeCharacteristicRing for P3Expr<F> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self(Expr::Const(F::ZERO));
    const ONE: Self = Self(Expr::Const(F::ONE));
    const TWO: Self = Self(Expr::Const(F::TWO));
    const NEG_ONE: Self = Self(Expr::Const(F::NEG_ONE));

    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        F::from_prime_subfield(f).into()
    }
}

impl<F: Field> Algebra<F> for P3Expr<F> {}

impl<F: Field> Algebra<P3Var<F>> for P3Expr<F> {}

impl<F: Field, T: Into<P3Expr<F>>> Add<T> for P3Var<F> {
    type Output = P3Expr<F>;

    fn add(self, rhs: T) -> P3Expr<F> {
        P3Expr::from(self) + rhs.into()
    }
}

impl<F: Field, T: Into<P3Expr<F>>> Sub<T> for P3Var<F> {
    type Output = P3Expr<F>;

    fn sub(self, rhs: T) -> P3Expr<F> {
        P3Expr::from(self) - rhs.into()
    }
}

impl<F: Field, T: Into<P3Expr<F>>> Mul<T> for P3Var<F> {
    type Output = P3Expr<F>;

    fn mul(self, rhs: T) -> P3Expr<F> {
        P3Expr::from(self) * rhs.into()
    }
}

/// Two-row window of symbolic variables over one trace.
#[derive(Clone, Debug)]
pub struct P3Window<F> {
    current: Vec<P3Var<F>>,
    next: Vec<P3Var<F>>,
}

impl<F> P3Window<F> {
    fn new(source: Source, width: usize) -> Self {
        let width = u32::try_from(width).expect("trace width exceeds u32");
        let vars = |offset| {
            (0..width)
                .map(|index| P3Var::new(source, offset, index))
                .collect()
        };
        Self {
            current: vars(RowOffset::Current),
            next: vars(RowOffset::Next),
        }
    }
}

impl<F> WindowAccess<P3Var<F>> for P3Window<F> {
    fn current_slice(&self) -> &[P3Var<F>] {
        &self.current
    }

    fn next_slice(&self) -> &[P3Var<F>] {
        &self.next
    }
}

/// An `AirBuilder` that records the asserted constraints as [`Expr`] trees.
pub struct P3AirBuilder<F> {
    main: P3Window<F>,
    preprocessed: P3Window<F>,
    constraints: Vec<Expr<F>>,
}

impl<F: Field> AirBuilder for P3AirBuilder<F> {
    type F = F;
    type Expr = P3Expr<F>;
    type Var = P3Var<F>;
    type PreprocessedWindow = P3Window<F>;
    type MainWindow = P3Window<F>;
    // Public inputs are reserved for the lookup argument in this system, so
    // the builder exposes none (the default `public_values()` is empty).
    type PublicVar = P3Var<F>;

    fn main(&self) -> Self::MainWindow {
        self.main.clone()
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    fn is_first_row(&self) -> Self::Expr {
        P3Expr(Expr::IsFirstRow)
    }

    fn is_last_row(&self) -> Self::Expr {
        P3Expr(Expr::IsLastRow)
    }

    /// # Panics
    /// Panics if `size` is not `2`; only two-row windows are supported.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert_eq!(size, 2, "multi-stark only supports a window size of 2");
        P3Expr(Expr::IsTransition)
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into().0);
    }
}

/// Evaluates a Plonky3-style AIR into [`CircuitInputs`]: the main width and
/// preprocessed trace come from [`p3_air::BaseAir`], the constraints from
/// [`Air::eval`]. Lookups can be pushed onto the result afterwards.
///
/// # Panics
/// Panics if the AIR declares periodic columns or public values, which
/// this system does not support.
pub fn circuit_inputs_from_air<F: Field, A: Air<P3AirBuilder<F>>>(air: &A) -> CircuitInputs<F> {
    assert_eq!(
        air.num_periodic_columns(),
        0,
        "periodic columns are not supported"
    );
    assert_eq!(
        air.num_public_values(),
        0,
        "public inputs are reserved for the lookup argument"
    );
    let main_width = air.width();
    let preprocessed = air.preprocessed_trace();
    let preprocessed_width = preprocessed.as_ref().map_or(0, |m| m.width);
    let mut builder = P3AirBuilder {
        main: P3Window::new(Source::Main, main_width),
        preprocessed: P3Window::new(Source::Preprocessed, preprocessed_width),
        constraints: vec![],
    };
    air.eval(&mut builder);
    CircuitInputs {
        main_width,
        preprocessed,
        constraints: builder.constraints,
        ..Default::default()
    }
}
