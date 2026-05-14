//! Symbolic constraint builder + expressions.
//!
//! Used to (a) record constraints once at [`Circuit`](crate::system::Circuit)
//! construction time, then derive their max degree, and (b) re-interpret lookup
//! argument expressions against concrete trace rows in both witness generation
//! and the in-circuit constraint evaluation.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_ff::{One, Zero};

use crate::air::{Air, AirBuilder};
use crate::types::Val;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Entry {
    Preprocessed { offset: usize },
    Main { offset: usize },
    Stage2 { offset: usize },
    Stage2Public,
}

#[derive(Copy, Clone, Debug)]
pub struct SymbolicVariable {
    pub entry: Entry,
    pub index: usize,
}

impl SymbolicVariable {
    pub const fn new(entry: Entry, index: usize) -> Self {
        Self { entry, index }
    }

    pub const fn degree_multiple(&self) -> usize {
        match self.entry {
            Entry::Preprocessed { .. } | Entry::Main { .. } | Entry::Stage2 { .. } => 1,
            Entry::Stage2Public => 0,
        }
    }
}

#[derive(Clone, Debug)]
pub enum SymbolicExpression {
    Variable(SymbolicVariable),
    IsFirstRow,
    IsLastRow,
    IsTransition,
    Constant(Val),
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

impl SymbolicExpression {
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

    /// Evaluate against a concrete trace row + (optional) preprocessed row.
    /// Used by lookup argument expressions, which can only refer to current-row
    /// columns.
    pub fn interpret<E>(&self, row: &[E], preprocessed: Option<&[E]>) -> E
    where
        E: Clone
            + From<Val>
            + Add<Output = E>
            + Sub<Output = E>
            + Mul<Output = E>
            + Neg<Output = E>,
    {
        match self {
            Self::Variable(v) => match v.entry {
                Entry::Main { offset: 0 } => row[v.index].clone(),
                Entry::Preprocessed { offset: 0 } => preprocessed
                    .expect("preprocessed row required but not provided")[v.index]
                    .clone(),
                _ => panic!(
                    "symbolic expression in lookup args may only reference offset-0 main or preprocessed columns"
                ),
            },
            Self::Constant(c) => E::from(*c),
            Self::Add { x, y, .. } => {
                x.interpret(row, preprocessed) + y.interpret(row, preprocessed)
            }
            Self::Sub { x, y, .. } => {
                x.interpret(row, preprocessed) - y.interpret(row, preprocessed)
            }
            Self::Mul { x, y, .. } => {
                x.interpret(row, preprocessed) * y.interpret(row, preprocessed)
            }
            Self::Neg { x, .. } => -x.interpret(row, preprocessed),
            Self::IsFirstRow | Self::IsLastRow | Self::IsTransition => {
                panic!("row selectors are not allowed in lookup expressions")
            }
        }
    }
}

impl From<SymbolicVariable> for SymbolicExpression {
    fn from(v: SymbolicVariable) -> Self {
        Self::Variable(v)
    }
}

impl From<Val> for SymbolicExpression {
    fn from(v: Val) -> Self {
        Self::Constant(v)
    }
}

impl Default for SymbolicExpression {
    fn default() -> Self {
        Self::Constant(Val::zero())
    }
}

#[inline]
pub fn var(i: usize) -> SymbolicExpression {
    SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, i))
}

#[inline]
pub fn preprocessed_var(i: usize) -> SymbolicExpression {
    SymbolicExpression::Variable(SymbolicVariable::new(Entry::Preprocessed { offset: 0 }, i))
}

impl Add for SymbolicExpression {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Constant(a), b) if a.is_zero() => b,
            (a, Self::Constant(b)) if b.is_zero() => a,
            (Self::Constant(a), Self::Constant(b)) => Self::Constant(a + b),
            (a, b) => {
                let degree_multiple = a.degree_multiple().max(b.degree_multiple());
                Self::Add {
                    x: Box::new(a),
                    y: Box::new(b),
                    degree_multiple,
                }
            }
        }
    }
}

impl Sub for SymbolicExpression {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        match (self, rhs) {
            (a, Self::Constant(b)) if b.is_zero() => a,
            (Self::Constant(a), Self::Constant(b)) => Self::Constant(a - b),
            (a, b) => {
                let degree_multiple = a.degree_multiple().max(b.degree_multiple());
                Self::Sub {
                    x: Box::new(a),
                    y: Box::new(b),
                    degree_multiple,
                }
            }
        }
    }
}

impl Mul for SymbolicExpression {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Constant(a), _) if a.is_zero() => Self::Constant(Val::zero()),
            (_, Self::Constant(b)) if b.is_zero() => Self::Constant(Val::zero()),
            (Self::Constant(a), b) if a.is_one() => b,
            (a, Self::Constant(b)) if b.is_one() => a,
            (Self::Constant(a), Self::Constant(b)) => Self::Constant(a * b),
            (a, b) => {
                let degree_multiple = a.degree_multiple() + b.degree_multiple();
                Self::Mul {
                    x: Box::new(a),
                    y: Box::new(b),
                    degree_multiple,
                }
            }
        }
    }
}

impl Neg for SymbolicExpression {
    type Output = Self;
    fn neg(self) -> Self {
        match self {
            Self::Constant(c) => Self::Constant(-c),
            other => {
                let degree_multiple = other.degree_multiple();
                Self::Neg {
                    x: Box::new(other),
                    degree_multiple,
                }
            }
        }
    }
}

impl AddAssign for SymbolicExpression {
    fn add_assign(&mut self, rhs: Self) {
        let lhs = std::mem::take(self);
        *self = lhs + rhs;
    }
}
impl SubAssign for SymbolicExpression {
    fn sub_assign(&mut self, rhs: Self) {
        let lhs = std::mem::take(self);
        *self = lhs - rhs;
    }
}
impl MulAssign for SymbolicExpression {
    fn mul_assign(&mut self, rhs: Self) {
        let lhs = std::mem::take(self);
        *self = lhs * rhs;
    }
}

/// Records constraints emitted by `air.eval(...)` for degree analysis.
pub struct SymbolicAirBuilder {
    pub preprocessed_local: Vec<SymbolicVariable>,
    pub preprocessed_next: Vec<SymbolicVariable>,
    pub main_local: Vec<SymbolicVariable>,
    pub main_next: Vec<SymbolicVariable>,
    pub stage_2_local: Vec<SymbolicVariable>,
    pub stage_2_next: Vec<SymbolicVariable>,
    pub stage_2_public: Vec<SymbolicVariable>,
    pub constraints: Vec<SymbolicExpression>,
}

impl SymbolicAirBuilder {
    pub fn new(
        preprocessed_width: usize,
        stage_1_width: usize,
        stage_2_width: usize,
        stage_2_public_count: usize,
    ) -> Self {
        let make = |entry_at_offset: fn(usize) -> Entry, w: usize| -> (Vec<_>, Vec<_>) {
            let local = (0..w)
                .map(|i| SymbolicVariable::new(entry_at_offset(0), i))
                .collect();
            let next = (0..w)
                .map(|i| SymbolicVariable::new(entry_at_offset(1), i))
                .collect();
            (local, next)
        };
        let (preprocessed_local, preprocessed_next) =
            make(|o| Entry::Preprocessed { offset: o }, preprocessed_width);
        let (main_local, main_next) = make(|o| Entry::Main { offset: o }, stage_1_width);
        let (stage_2_local, stage_2_next) = make(|o| Entry::Stage2 { offset: o }, stage_2_width);
        let stage_2_public = (0..stage_2_public_count)
            .map(|i| SymbolicVariable::new(Entry::Stage2Public, i))
            .collect();
        Self {
            preprocessed_local,
            preprocessed_next,
            main_local,
            main_next,
            stage_2_local,
            stage_2_next,
            stage_2_public,
            constraints: vec![],
        }
    }
}

impl AirBuilder for SymbolicAirBuilder {
    type Var = SymbolicVariable;
    type Expr = SymbolicExpression;

    fn main_local(&self) -> &[Self::Var] {
        &self.main_local
    }
    fn main_next(&self) -> &[Self::Var] {
        &self.main_next
    }
    fn preprocessed_local(&self) -> &[Self::Var] {
        &self.preprocessed_local
    }
    fn preprocessed_next(&self) -> &[Self::Var] {
        &self.preprocessed_next
    }
    fn stage_2_local(&self) -> &[Self::Var] {
        &self.stage_2_local
    }
    fn stage_2_next(&self) -> &[Self::Var] {
        &self.stage_2_next
    }
    fn stage_2_public_values(&self) -> &[Self::Var] {
        &self.stage_2_public
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }
    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }
    fn is_transition(&self) -> Self::Expr {
        SymbolicExpression::IsTransition
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into());
    }
}

pub fn get_symbolic_constraints<A>(
    air: &A,
    preprocessed_width: usize,
    stage_1_width: usize,
    stage_2_width: usize,
    stage_2_public_count: usize,
) -> Vec<SymbolicExpression>
where
    A: Air<SymbolicAirBuilder>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        stage_1_width,
        stage_2_width,
        stage_2_public_count,
    );
    air.eval(&mut builder);
    builder.constraints
}

pub fn get_max_constraint_degree(constraints: &[SymbolicExpression]) -> usize {
    constraints
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// SymbolicExpression interop with `Val` so that AIR code can write `expr + 1`,
// `expr * Val::from(2)`, etc. directly.
// ---------------------------------------------------------------------------

impl Add<Val> for SymbolicExpression {
    type Output = Self;
    fn add(self, rhs: Val) -> Self {
        self + Self::Constant(rhs)
    }
}
impl Sub<Val> for SymbolicExpression {
    type Output = Self;
    fn sub(self, rhs: Val) -> Self {
        self - Self::Constant(rhs)
    }
}
impl Mul<Val> for SymbolicExpression {
    type Output = Self;
    fn mul(self, rhs: Val) -> Self {
        self * Self::Constant(rhs)
    }
}

impl Add<SymbolicVariable> for SymbolicExpression {
    type Output = Self;
    fn add(self, rhs: SymbolicVariable) -> Self {
        self + Self::Variable(rhs)
    }
}
impl Sub<SymbolicVariable> for SymbolicExpression {
    type Output = Self;
    fn sub(self, rhs: SymbolicVariable) -> Self {
        self - Self::Variable(rhs)
    }
}
impl Mul<SymbolicVariable> for SymbolicExpression {
    type Output = Self;
    fn mul(self, rhs: SymbolicVariable) -> Self {
        self * Self::Variable(rhs)
    }
}

impl Add<SymbolicExpression> for SymbolicVariable {
    type Output = SymbolicExpression;
    fn add(self, rhs: SymbolicExpression) -> SymbolicExpression {
        SymbolicExpression::from(self) + rhs
    }
}
impl Sub<SymbolicExpression> for SymbolicVariable {
    type Output = SymbolicExpression;
    fn sub(self, rhs: SymbolicExpression) -> SymbolicExpression {
        SymbolicExpression::from(self) - rhs
    }
}
impl Mul<SymbolicExpression> for SymbolicVariable {
    type Output = SymbolicExpression;
    fn mul(self, rhs: SymbolicExpression) -> SymbolicExpression {
        SymbolicExpression::from(self) * rhs
    }
}

impl Add for SymbolicVariable {
    type Output = SymbolicExpression;
    fn add(self, rhs: Self) -> SymbolicExpression {
        SymbolicExpression::from(self) + SymbolicExpression::from(rhs)
    }
}
impl Sub for SymbolicVariable {
    type Output = SymbolicExpression;
    fn sub(self, rhs: Self) -> SymbolicExpression {
        SymbolicExpression::from(self) - SymbolicExpression::from(rhs)
    }
}
impl Mul for SymbolicVariable {
    type Output = SymbolicExpression;
    fn mul(self, rhs: Self) -> SymbolicExpression {
        SymbolicExpression::from(self) * SymbolicExpression::from(rhs)
    }
}
impl Neg for SymbolicVariable {
    type Output = SymbolicExpression;
    fn neg(self) -> SymbolicExpression {
        -SymbolicExpression::from(self)
    }
}

impl Add<Val> for SymbolicVariable {
    type Output = SymbolicExpression;
    fn add(self, rhs: Val) -> SymbolicExpression {
        SymbolicExpression::from(self) + SymbolicExpression::Constant(rhs)
    }
}
impl Sub<Val> for SymbolicVariable {
    type Output = SymbolicExpression;
    fn sub(self, rhs: Val) -> SymbolicExpression {
        SymbolicExpression::from(self) - SymbolicExpression::Constant(rhs)
    }
}
impl Mul<Val> for SymbolicVariable {
    type Output = SymbolicExpression;
    fn mul(self, rhs: Val) -> SymbolicExpression {
        SymbolicExpression::from(self) * SymbolicExpression::Constant(rhs)
    }
}
