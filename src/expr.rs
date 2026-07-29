//! Frontend expression trees.
//!
//! These exist only while a circuit is being described; compilation (see
//! [`crate::circuit`]) flattens them into the interned format everything
//! else consumes. Operators fold constants as they go.

use std::ops::{Add, Mul, Neg, Sub};

use p3_field::Field;

use crate::lookup::Lookup;

/// Which committed matrix a column variable refers to.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Source {
    Preprocessed,
    Main,
    /// The stage-2 trace in its flattened base-column layout.
    Stage2,
}

/// The evaluation window: only the current and next row are addressable.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum RowOffset {
    Current,
    Next,
}

/// A column of a committed matrix, at the current or next row.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ColRef {
    pub source: Source,
    pub offset: RowOffset,
    pub index: u32,
}

/// A base-field expression.
#[derive(Clone, Debug)]
pub enum Expr<F> {
    Const(F),
    Var(ColRef),
    /// A public input, as a base-field coordinate.
    Public(u32),
    IsFirstRow,
    IsLastRow,
    IsTransition,
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Neg(Box<Self>),
}

/// An extension-field expression. The primitive is `Coords`: an array of
/// D base-field coordinates representing `Σ_j coord_j · b_j`.
#[derive(Clone, Debug)]
pub enum ExtExpr<F> {
    /// The D-tuple primitive; length is checked at compile time.
    Coords(Vec<Expr<F>>),
    /// Implicit embedding of a base expression, so mixed arithmetic like
    /// `m * inv` needs no explicit lift.
    Base(Box<Expr<F>>),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Neg(Box<Self>),
}

/// What the constraint compiler consumes, per circuit: the authored
/// [`crate::system::CircuitInputs`] with the derived widths filled in and
/// the synthesized lookup constraints appended. Built internally by
/// `System::new`; not part of the public API.
#[derive(Clone, Debug)]
pub(crate) struct CircuitSpec<F> {
    pub main_width: usize,
    pub preprocessed_width: usize,
    /// Width of the stage-2 trace in flattened base columns.
    pub stage2_width: usize,
    /// Number of public inputs (base-field coordinates).
    pub num_publics: usize,
    /// Base-field constraints: each must evaluate to 0 on every row.
    pub constraints: Vec<Expr<F>>,
    /// Extension-field constraints, same vanishing semantics.
    pub ext_constraints: Vec<ExtExpr<F>>,
    /// Lookup specifications.
    pub lookups: Vec<Lookup<Expr<F>>>,
}

impl<F> Default for CircuitSpec<F> {
    fn default() -> Self {
        Self {
            main_width: 0,
            preprocessed_width: 0,
            stage2_width: 0,
            num_publics: 0,
            constraints: vec![],
            ext_constraints: vec![],
            lookups: vec![],
        }
    }
}

impl<F: Field> Expr<F> {
    pub fn constant(value: F) -> Self {
        Self::Const(value)
    }

    pub fn var(source: Source, offset: RowOffset, index: u32) -> Self {
        Self::Var(ColRef {
            source,
            offset,
            index,
        })
    }

    pub fn main(index: u32) -> Self {
        Self::var(Source::Main, RowOffset::Current, index)
    }

    pub fn main_next(index: u32) -> Self {
        Self::var(Source::Main, RowOffset::Next, index)
    }

    pub fn preprocessed(index: u32) -> Self {
        Self::var(Source::Preprocessed, RowOffset::Current, index)
    }

    pub fn preprocessed_next(index: u32) -> Self {
        Self::var(Source::Preprocessed, RowOffset::Next, index)
    }

    pub fn stage2(index: u32) -> Self {
        Self::var(Source::Stage2, RowOffset::Current, index)
    }

    pub fn stage2_next(index: u32) -> Self {
        Self::var(Source::Stage2, RowOffset::Next, index)
    }

    pub fn public(index: u32) -> Self {
        Self::Public(index)
    }
}

impl<F: Field> ExtExpr<F> {
    /// An extension constant from its D basis coordinates.
    pub fn constant(coords: Vec<F>) -> Self {
        Self::Coords(coords.into_iter().map(Expr::Const).collect())
    }

    /// The extension value stored in stage-2 slot `slot`: D consecutive
    /// flattened base columns starting at `slot * d`.
    pub fn stage2(slot: u32, d: u32, offset: RowOffset) -> Self {
        Self::Coords(
            (0..d)
                .map(|j| Expr::var(Source::Stage2, offset, slot * d + j))
                .collect(),
        )
    }

    /// The extension public value `k`: D consecutive base coordinates
    /// starting at `k * d`.
    pub fn public(k: u32, d: u32) -> Self {
        Self::Coords((0..d).map(|j| Expr::Public(k * d + j)).collect())
    }
}

impl<F: Field> From<F> for Expr<F> {
    fn from(value: F) -> Self {
        Self::Const(value)
    }
}

impl<F: Field> From<Expr<F>> for ExtExpr<F> {
    fn from(value: Expr<F>) -> Self {
        Self::Base(Box::new(value))
    }
}

impl<F: Field> Add for Expr<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Const(a), Self::Const(b)) => Self::Const(a + b),
            (Self::Const(z), x) | (x, Self::Const(z)) if z == F::ZERO => x,
            (a, b) => Self::Add(Box::new(a), Box::new(b)),
        }
    }
}

impl<F: Field> Sub for Expr<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Const(a), Self::Const(b)) => Self::Const(a - b),
            (x, Self::Const(z)) if z == F::ZERO => x,
            (Self::Const(z), x) if z == F::ZERO => -x,
            (a, b) => Self::Sub(Box::new(a), Box::new(b)),
        }
    }
}

impl<F: Field> Mul for Expr<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Const(a), Self::Const(b)) => Self::Const(a * b),
            (Self::Const(z), _) | (_, Self::Const(z)) if z == F::ZERO => Self::Const(F::ZERO),
            (Self::Const(o), x) | (x, Self::Const(o)) if o == F::ONE => x,
            (a, b) => Self::Mul(Box::new(a), Box::new(b)),
        }
    }
}

impl<F: Field> Neg for Expr<F> {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            Self::Const(c) => Self::Const(-c),
            Self::Neg(inner) => *inner,
            e => Self::Neg(Box::new(e)),
        }
    }
}

impl<F: Field> Add for ExtExpr<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::Add(Box::new(self), Box::new(rhs))
    }
}

impl<F: Field> Sub for ExtExpr<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::Sub(Box::new(self), Box::new(rhs))
    }
}

impl<F: Field> Mul for ExtExpr<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(rhs))
    }
}

impl<F: Field> Neg for ExtExpr<F> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::Neg(Box::new(self))
    }
}

/// Mixed base/extension operators, routed through `ExtExpr::Base` so a
/// lone base value combines with extension values without a visible lift.
macro_rules! mixed_ops {
    ($($trait:ident :: $method:ident),*) => {
        $(
            impl<F: Field> $trait<ExtExpr<F>> for Expr<F> {
                type Output = ExtExpr<F>;

                fn $method(self, rhs: ExtExpr<F>) -> ExtExpr<F> {
                    ExtExpr::from(self).$method(rhs)
                }
            }

            impl<F: Field> $trait<Expr<F>> for ExtExpr<F> {
                type Output = ExtExpr<F>;

                fn $method(self, rhs: Expr<F>) -> ExtExpr<F> {
                    self.$method(ExtExpr::from(rhs))
                }
            }
        )*
    };
}

mixed_ops!(Add::add, Sub::sub, Mul::mul);

impl<F: Field> ExtExpr<F> {
    /// True if the expression is built solely from base embeddings — its
    /// expansion would be one real constraint plus D−1 trivial zeros, so
    /// it should have been stated as a base constraint.
    pub fn is_purely_base(&self) -> bool {
        match self {
            Self::Coords(_) => false,
            Self::Base(_) => true,
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => {
                a.is_purely_base() && b.is_purely_base()
            }
            Self::Neg(a) => a.is_purely_base(),
        }
    }
}

// NOTE: deep frontend trees drop recursively; the production version needs
// an iterative `Drop` (which in turn requires the operators to avoid
// by-value destructuring, E0509). Out of scope for this lab.
