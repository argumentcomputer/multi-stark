/// Symbolic constraint builder and expressions, adapted from Plonky3.
use p3_air::{Air, AirBuilder, ExtensionBuilder};
use p3_field::{Algebra, Dup, ExtensionField, Field, InjectiveMonomial, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use std::fmt::Debug;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::TwoStagedBuilder;

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

#[inline]
pub fn var<F: Field>(i: usize) -> SymbolicExpression<F> {
    SymbolicExpression::from(SymbolicVariable::new(Entry::Main { offset: 0 }, i))
}

#[inline]
pub fn preprocessed_var<F: Field>(i: usize) -> SymbolicExpression<F> {
    SymbolicExpression::from(SymbolicVariable::new(Entry::Preprocessed { offset: 0 }, i))
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

impl<F: Field> Dup for SymbolicExpression<F> {
    #[inline(always)]
    fn dup(&self) -> Self {
        self.clone()
    }
}

impl<F: Field> Default for SymbolicExpression<F> {
    fn default() -> Self {
        Self::Constant(F::ZERO)
    }
}

/// Base-field values embed into expressions over any extension of the base
/// field (including the base field itself, via the reflexive
/// `ExtensionField<F> for F`). This single impl replaces both the same-field
/// `From<F>` conversion and the old concrete `Val` → `ExtVal` lifting.
impl<F: Field, EF: ExtensionField<F>> From<F> for SymbolicExpression<EF> {
    fn from(value: F) -> Self {
        Self::Constant(value.into())
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

impl<F: Field, EF: ExtensionField<F>> Algebra<F> for SymbolicExpression<EF> {}

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

pub fn get_symbolic_constraints<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    stage_1_width: usize,
    stage_2_width: usize,
    num_public_values: usize,
    num_stage_2_public_values: usize,
) -> Vec<SymbolicExpression<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
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

pub fn get_max_constraint_degree<EF: Field>(constraints: &[SymbolicExpression<EF>]) -> usize {
    constraints
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0)
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
///
/// All variables and expressions are tagged with the extension field `EF`,
/// even those referring to base-field trace columns. The tag only affects the
/// type of embedded constants — using a single tag lets `Expr` and `ExprEF`
/// be the same type, which sidesteps the coherence problems of converting
/// between `SymbolicExpression<F>` and `SymbolicExpression<EF>`.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field, EF: ExtensionField<F>> {
    preprocessed: RowMajorMatrix<SymbolicVariable<EF>>,
    stage_1: RowMajorMatrix<SymbolicVariable<EF>>,
    stage_2: RowMajorMatrix<SymbolicVariable<EF>>,
    public_values: Vec<SymbolicVariable<EF>>,
    stage_2_public_values: Vec<SymbolicVariable<EF>>,
    constraints: Vec<SymbolicExpression<EF>>,
    _phantom: PhantomData<F>,
}

impl<F: Field, EF: ExtensionField<F>> SymbolicAirBuilder<F, EF> {
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
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            stage_1: RowMajorMatrix::new(stage_1_values, stage_1_width),
            stage_2: RowMajorMatrix::new(stage_2_values, stage_2_width),
            public_values,
            stage_2_public_values,
            constraints: vec![],
            _phantom: PhantomData,
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilder for SymbolicAirBuilder<F, EF> {
    type F = F;
    type Expr = SymbolicExpression<EF>;
    type Var = SymbolicVariable<EF>;
    type PreprocessedWindow = RowMajorMatrix<Self::Var>;
    type MainWindow = RowMajorMatrix<Self::Var>;
    type PublicVar = SymbolicVariable<EF>;

    fn main(&self) -> Self::MainWindow {
        self.stage_1.clone()
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
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
        self.constraints.push(x.into());
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for SymbolicAirBuilder<F, EF> {
    type EF = EF;
    type ExprEF = SymbolicExpression<EF>;
    type VarEF = SymbolicVariable<EF>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.constraints.push(x.into());
    }
}

impl<F: Field, EF: ExtensionField<F>> TwoStagedBuilder for SymbolicAirBuilder<F, EF> {
    type MP = RowMajorMatrix<Self::VarEF>;

    type Stage2PublicVar = Self::VarEF;

    fn stage_2(&self) -> Self::MP {
        self.stage_2.clone()
    }

    fn stage_2_public_values(&self) -> &[Self::Stage2PublicVar] {
        &self.stage_2_public_values
    }
}
