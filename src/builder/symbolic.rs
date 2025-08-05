/// Adapted from Plonky3's `https://github.com/Plonky3/Plonky3/blob/main/uni-stark/src/symbolic_builder.rs`
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder};
use p3_field::{Algebra, Field, InjectiveMonomial, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use std::fmt::Debug;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::types::{ExtVal, Val};

use super::{PreprocessedBuilder, TwoStagedBuilder};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Entry {
    Preprocessed { offset: usize },
    Main { offset: usize },
    Stage2 { offset: usize },
    Public,
    Stage2Public,
    Challenge,
}

/// A variable within the evaluation window, i.e. a column in either the local or next row.
#[derive(Copy, Clone, Debug)]
pub struct SymbolicVariable<F> {
    pub entry: Entry,
    pub index: usize,
    pub(crate) _phantom: PhantomData<F>,
}

impl<F> SymbolicVariable<F> {
    pub const fn new(entry: Entry, index: usize) -> Self {
        Self {
            entry,
            index,
            _phantom: PhantomData,
        }
    }

    pub const fn degree_multiple(&self) -> usize {
        match self.entry {
            Entry::Preprocessed { .. } | Entry::Main { .. } | Entry::Stage2 { .. } => 1,
            Entry::Public | Entry::Challenge | Entry::Stage2Public => 0,
        }
    }
}

impl<F: Field> From<SymbolicVariable<F>> for SymbolicExpression<F> {
    fn from(value: SymbolicVariable<F>) -> Self {
        Self::Variable(value)
    }
}

impl<F: Field, T> Add<T> for SymbolicVariable<F>
where
    T: Into<SymbolicExpression<F>>,
{
    type Output = SymbolicExpression<F>;

    fn add(self, rhs: T) -> Self::Output {
        SymbolicExpression::from(self) + rhs.into()
    }
}

impl<F: Field, T> Sub<T> for SymbolicVariable<F>
where
    T: Into<SymbolicExpression<F>>,
{
    type Output = SymbolicExpression<F>;

    fn sub(self, rhs: T) -> Self::Output {
        SymbolicExpression::from(self) - rhs.into()
    }
}

impl<F: Field, T> Mul<T> for SymbolicVariable<F>
where
    T: Into<SymbolicExpression<F>>,
{
    type Output = SymbolicExpression<F>;

    fn mul(self, rhs: T) -> Self::Output {
        SymbolicExpression::from(self) * rhs.into()
    }
}

/// An expression over `SymbolicVariable`s.
#[derive(Clone, Debug)]
pub enum SymbolicExpression<F> {
    Variable(SymbolicVariable<F>),
    IsFirstRow,
    IsLastRow,
    IsTransition,
    Constant(F),
    Add {
        x: Box<Self>,
        y: Box<Self>,
        degree_multiple: usize,
    },
    Sub {
        x: Box<Self>,
        y: Box<Self>,
        degree_multiple: usize,
    },
    Neg {
        x: Box<Self>,
        degree_multiple: usize,
    },
    Mul {
        x: Box<Self>,
        y: Box<Self>,
        degree_multiple: usize,
    },
}

impl<F: Field> SymbolicExpression<F> {
    pub fn interpret<Expr: Algebra<F>, Var: Into<Expr> + Clone>(
        &self,
        row: &[Var],
        preprocessed: Option<&[Var]>,
    ) -> Expr {
        match self {
            Self::Variable(var) => match &var.entry {
                Entry::Main { offset: 0 } => row[var.index].clone().into(),
                Entry::Preprocessed { offset: 0 } => {
                    preprocessed.unwrap()[var.index].clone().into()
                }
                _ => unimplemented!(),
            },
            Self::Constant(c) => (*c).into(),
            Self::Add { x, y, .. } => {
                x.interpret(row, preprocessed) + y.interpret(row, preprocessed)
            }
            Self::Sub { x, y, .. } => {
                x.interpret(row, preprocessed) - y.interpret(row, preprocessed)
            }
            Self::Neg { x, .. } => -x.interpret(row, preprocessed),
            Self::Mul { x, y, .. } => {
                x.interpret(row, preprocessed) * y.interpret(row, preprocessed)
            }
            _ => unimplemented!(),
        }
    }

    /// Returns the multiple of `n` (the trace length) in this expression's degree.
    pub const fn degree_multiple(&self) -> usize {
        match self {
            Self::Variable(v) => v.degree_multiple(),
            Self::IsFirstRow | Self::IsLastRow => 1,
            Self::IsTransition | Self::Constant(_) => 0,
            Self::Add {
                degree_multiple, ..
            }
            | Self::Sub {
                degree_multiple, ..
            }
            | Self::Neg {
                degree_multiple, ..
            }
            | Self::Mul {
                degree_multiple, ..
            } => *degree_multiple,
        }
    }
}

impl<F: Field> Default for SymbolicExpression<F> {
    fn default() -> Self {
        Self::Constant(F::ZERO)
    }
}

impl<F: Field> From<F> for SymbolicExpression<F> {
    fn from(value: F) -> Self {
        Self::Constant(value)
    }
}

impl<F: Field> PrimeCharacteristicRing for SymbolicExpression<F> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self::Constant(F::ZERO);
    const ONE: Self = Self::Constant(F::ONE);
    const TWO: Self = Self::Constant(F::TWO);
    const NEG_ONE: Self = Self::Constant(F::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        F::from_prime_subfield(f).into()
    }
}

impl<F: Field> Algebra<F> for SymbolicExpression<F> {}

impl<F: Field> Algebra<SymbolicVariable<F>> for SymbolicExpression<F> {}

impl<F: Field + InjectiveMonomial<N>, const N: u64> InjectiveMonomial<N> for SymbolicExpression<F> {}

impl<F: Field, T> Add<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self {
        match (self, rhs.into()) {
            (Self::Constant(lhs), rhs) if lhs == F::ZERO => rhs,
            (lhs, Self::Constant(rhs)) if rhs == F::ZERO => lhs,
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs + rhs),
            (lhs, rhs) => Self::Add {
                degree_multiple: lhs.degree_multiple().max(rhs.degree_multiple()),
                x: Box::new(lhs),
                y: Box::new(rhs),
            },
        }
    }
}

impl<F: Field, T> AddAssign<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn add_assign(&mut self, rhs: T) {
        *self = self.clone() + rhs.into();
    }
}

impl<F: Field, T> Sum<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|x, y| x + y)
            .unwrap_or(Self::ZERO)
    }
}

impl<F: Field, T: Into<Self>> Sub<T> for SymbolicExpression<F> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self {
        match (self, rhs.into()) {
            (Self::Constant(lhs), rhs) if lhs == F::ZERO => -rhs,
            (lhs, Self::Constant(rhs)) if rhs == F::ZERO => lhs,
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs - rhs),
            (lhs, rhs) => Self::Sub {
                degree_multiple: lhs.degree_multiple().max(rhs.degree_multiple()),
                x: Box::new(lhs),
                y: Box::new(rhs),
            },
        }
    }
}

impl<F: Field, T> SubAssign<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn sub_assign(&mut self, rhs: T) {
        *self = self.clone() - rhs.into();
    }
}

impl<F: Field> Neg for SymbolicExpression<F> {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            Self::Constant(c) => Self::Constant(-c),
            expr => Self::Neg {
                degree_multiple: expr.degree_multiple(),
                x: Box::new(expr),
            },
        }
    }
}

impl<F: Field, T: Into<Self>> Mul<T> for SymbolicExpression<F> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        match (self, rhs.into()) {
            (Self::Constant(lhs), rhs) if lhs == F::ONE => rhs,
            (lhs, Self::Constant(rhs)) if rhs == F::ONE => lhs,
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs * rhs),
            (lhs, rhs) => Self::Mul {
                degree_multiple: lhs.degree_multiple() + rhs.degree_multiple(),
                x: Box::new(lhs),
                y: Box::new(rhs),
            },
        }
    }
}

impl<F: Field, T> MulAssign<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone() * rhs.into();
    }
}

impl<F: Field, T: Into<Self>> Product<T> for SymbolicExpression<F> {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|x, y| x * y)
            .unwrap_or(Self::ONE)
    }
}

pub fn get_log_quotient_degree<F, A>(
    air: &A,
    preprocessed_width: usize,
    stage_1_width: usize,
    stage_2_width: usize,
    num_public_values: usize,
    num_stage_2_public_values: usize,

    is_zk: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder>,
{
    assert!(is_zk <= 1, "is_zk must be either 0 or 1");
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = (get_max_constraint_degree(
        air,
        preprocessed_width,
        stage_1_width,
        stage_2_width,
        num_public_values,
        num_stage_2_public_values,
    ) + is_zk)
        .max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the vanishing polynomial.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

pub fn get_max_constraint_degree<A>(
    air: &A,
    preprocessed_width: usize,
    stage_1_width: usize,
    stage_2_width: usize,
    num_public_values: usize,
    num_stage_2_public_values: usize,
) -> usize
where
    A: Air<SymbolicAirBuilder>,
{
    get_symbolic_constraints(
        air,
        preprocessed_width,
        stage_1_width,
        stage_2_width,
        num_public_values,
        num_stage_2_public_values,
    )
    .iter()
    .map(|c| c.degree_multiple())
    .max()
    .unwrap_or(0)
}

pub fn get_symbolic_constraints<A>(
    air: &A,
    preprocessed_width: usize,
    stage_1_width: usize,
    stage_2_width: usize,
    num_public_values: usize,
    num_stage_2_public_values: usize,
) -> Vec<SymbolicExpression<ExtVal>>
where
    A: Air<SymbolicAirBuilder>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        stage_1_width,
        stage_2_width,
        num_public_values,
        num_stage_2_public_values,
    );
    air.eval(&mut builder);
    builder.constraints
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder {
    preprocessed: Option<RowMajorMatrix<SymbolicVariable<Val>>>,
    stage_1: RowMajorMatrix<SymbolicVariable<Val>>,
    stage_2: RowMajorMatrix<SymbolicVariable<ExtVal>>,
    public_values: Vec<SymbolicVariable<Val>>,
    stage_2_public_values: Vec<SymbolicVariable<ExtVal>>,
    constraints: Vec<SymbolicExpression<ExtVal>>,
}

impl SymbolicAirBuilder {
    pub(crate) fn new(
        preprocessed_width: usize,
        stage_1_width: usize,
        stage_2_width: usize,
        num_public_values: usize,
        num_stage_2_public_values: usize,
    ) -> Self {
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width)
                    .map(move |index| SymbolicVariable::new(Entry::Preprocessed { offset }, index))
            })
            .collect();
        let stage_1_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..stage_1_width)
                    .map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
            })
            .collect();
        let stage_2_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..stage_2_width)
                    .map(move |index| SymbolicVariable::new(Entry::Stage2 { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();
        let stage_2_public_values = (0..num_stage_2_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Stage2Public, index))
            .collect();
        Self {
            preprocessed: if preprocessed_width == 0 {
                None
            } else {
                Some(RowMajorMatrix::new(prep_values, preprocessed_width))
            },
            stage_1: RowMajorMatrix::new(stage_1_values, stage_1_width),
            stage_2: RowMajorMatrix::new(stage_2_values, stage_2_width),
            public_values,
            stage_2_public_values,
            constraints: vec![],
        }
    }
}

impl AirBuilder for SymbolicAirBuilder {
    type F = Val;
    type Expr = SymbolicExpression<Val>;
    type Var = SymbolicVariable<Val>;
    type M = RowMajorMatrix<Self::Var>;

    fn main(&self) -> Self::M {
        self.stage_1.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition
        } else {
            panic!("multi-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into().into());
    }
}

impl AirBuilderWithPublicValues for SymbolicAirBuilder {
    type PublicVar = SymbolicVariable<Val>;
    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl PreprocessedBuilder for SymbolicAirBuilder {
    fn preprocessed(&self) -> Option<Self::M> {
        self.preprocessed.clone()
    }
}

impl Algebra<SymbolicExpression<Val>> for SymbolicExpression<ExtVal> {}

impl From<SymbolicExpression<Val>> for SymbolicExpression<ExtVal> {
    fn from(value: SymbolicExpression<Val>) -> Self {
        match value {
            SymbolicExpression::Variable(SymbolicVariable {
                entry,
                index,
                _phantom,
            }) => Self::Variable(SymbolicVariable {
                entry,
                index,
                _phantom: PhantomData,
            }),
            SymbolicExpression::IsFirstRow => Self::IsFirstRow,
            SymbolicExpression::IsLastRow => Self::IsLastRow,
            SymbolicExpression::IsTransition => Self::IsTransition,
            SymbolicExpression::Constant(f) => Self::Constant(f.into()),
            SymbolicExpression::Add {
                x,
                y,
                degree_multiple,
            } => Self::Add {
                x: Self::from(*x).into(),
                y: Self::from(*y).into(),
                degree_multiple,
            },
            SymbolicExpression::Sub {
                x,
                y,
                degree_multiple,
            } => Self::Sub {
                x: Self::from(*x).into(),
                y: Self::from(*y).into(),
                degree_multiple,
            },
            SymbolicExpression::Mul {
                x,
                y,
                degree_multiple,
            } => Self::Mul {
                x: Self::from(*x).into(),
                y: Self::from(*y).into(),
                degree_multiple,
            },
            SymbolicExpression::Neg { x, degree_multiple } => Self::Neg {
                x: Self::from(*x).into(),
                degree_multiple,
            },
        }
    }
}

impl ExtensionBuilder for SymbolicAirBuilder {
    type EF = ExtVal;
    type ExprEF = SymbolicExpression<ExtVal>;
    type VarEF = SymbolicVariable<ExtVal>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.constraints.push(x.into());
    }
}

impl TwoStagedBuilder for SymbolicAirBuilder {
    type MP = RowMajorMatrix<Self::VarEF>;

    type Stage2PublicVar = Self::VarEF;

    fn stage_2(&self) -> Self::MP {
        self.stage_2.clone()
    }

    fn stage_2_public_values(&self) -> &[Self::Stage2PublicVar] {
        &self.stage_2_public_values
    }
}
