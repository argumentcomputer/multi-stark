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
        self.interpret_with(&|var| match &var.entry {
            Entry::Main { offset: 0 } => row[var.index].clone().into(),
            Entry::Preprocessed { offset: 0 } => preprocessed.unwrap()[var.index].clone().into(),
            _ => unimplemented!(),
        })
    }

    /// Evaluates the expression with a caller-supplied variable resolver,
    /// which decides what each [`SymbolicVariable`] stands for.
    ///
    /// # Panics
    /// Panics on the row selector expressions (`IsFirstRow`, `IsLastRow`,
    /// `IsTransition`), which have no value outside a constraint context.
    pub fn interpret_with<Expr: Algebra<F>>(
        &self,
        resolve: &impl Fn(&SymbolicVariable<F>) -> Expr,
    ) -> Expr {
        match self {
            Self::Variable(var) => resolve(var),
            Self::Constant(c) => (*c).into(),
            Self::Add { x, y, .. } => x.interpret_with(resolve) + y.interpret_with(resolve),
            Self::Sub { x, y, .. } => x.interpret_with(resolve) - y.interpret_with(resolve),
            Self::Neg { x, .. } => -x.interpret_with(resolve),
            Self::Mul { x, y, .. } => x.interpret_with(resolve) * y.interpret_with(resolve),
            _ => unimplemented!("row selectors cannot be interpreted"),
        }
    }

    /// Re-tags the expression from `F` to an extension field `EF`. The
    /// structure (variables, degrees) is unchanged; only embedded constants
    /// are lifted.
    pub fn lift<EF: ExtensionField<F>>(&self) -> SymbolicExpression<EF> {
        match self {
            Self::Variable(var) => SymbolicVariable::new(var.entry, var.index).into(),
            Self::IsFirstRow => SymbolicExpression::IsFirstRow,
            Self::IsLastRow => SymbolicExpression::IsLastRow,
            Self::IsTransition => SymbolicExpression::IsTransition,
            Self::Constant(c) => SymbolicExpression::Constant((*c).into()),
            Self::Add {
                x,
                y,
                degree_multiple,
            } => SymbolicExpression::Add {
                x: Box::new(x.lift()),
                y: Box::new(y.lift()),
                degree_multiple: *degree_multiple,
            },
            Self::Sub {
                x,
                y,
                degree_multiple,
            } => SymbolicExpression::Sub {
                x: Box::new(x.lift()),
                y: Box::new(y.lift()),
                degree_multiple: *degree_multiple,
            },
            Self::Neg { x, degree_multiple } => SymbolicExpression::Neg {
                x: Box::new(x.lift()),
                degree_multiple: *degree_multiple,
            },
            Self::Mul {
                x,
                y,
                degree_multiple,
            } => SymbolicExpression::Mul {
                x: Box::new(x.lift()),
                y: Box::new(y.lift()),
                degree_multiple: *degree_multiple,
            },
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

/// Accumulator for the `x = expr` definitions created by [`trim`], together
/// with the allocator for the fresh variables: definitions are placed at
/// `entry`, taking consecutive indices starting from the one given at
/// construction.
///
/// A definition is semantically just the zero constraint `x - expr`, but it
/// is kept apart so that provers know to fill the column `x` with the value
/// of `expr` (and so that a Plonkish backend could treat it as a wire
/// assignment). Definitions may reference variables defined by earlier
/// entries of the list, never later ones.
#[derive(Debug)]
pub struct Definitions<F> {
    entry: Entry,
    next_index: usize,
    pub definitions: Vec<(SymbolicVariable<F>, SymbolicExpression<F>)>,
}

impl<F: Field> Definitions<F> {
    pub fn new(entry: Entry, first_index: usize) -> Self {
        Self {
            entry,
            next_index: first_index,
            definitions: vec![],
        }
    }

    /// The index the next extracted variable would take.
    pub fn next_index(&self) -> usize {
        self.next_index
    }

    fn fresh(&mut self) -> SymbolicVariable<F> {
        let variable = SymbolicVariable::new(self.entry, self.next_index);
        self.next_index += 1;
        variable
    }
}

/// Trims `expr` to degree (multiple) at most `budget`, assuming constraints
/// may have degree up to `max_degree`. Subexpressions that cannot be fit
/// structurally are *extracted*: replaced by a fresh degree-1 variable `x`
/// whose definition `x = subexpr` is appended to `definitions` (the
/// definition itself being trimmed to `max_degree`, so that its zero
/// constraint `x - subexpr` is admissible).
///
/// The structural rules are: an addition is trimmed by trimming both sides
/// to the same budget, and a multiplication by splitting the budget between
/// its sides (a side of degree zero gets budget zero for free). Among all
/// ways of interleaving these rules with extraction, the algorithm picks one
/// minimizing the **number of extractions** — each costs a column and a
/// constraint — via dynamic programming over the expression tree: every node
/// gets the minimal cost of reaching each budget in `1..=max_degree`, and
/// the result is rebuilt following the optimal choices. Ties are broken in
/// favor of not extracting, and multiplication splits give the left side the
/// least budget among optimal splits.
///
/// The trimmed expression evaluates to the same value as `expr` on any row
/// extended with the definition values in order.
///
/// # Panics
/// Panics if `max_degree < 2` (extraction could never terminate: a product
/// of two variables would not even fit a definition) or if `budget` is not
/// in `1..=max_degree`.
pub fn trim<F: Field>(
    expr: &SymbolicExpression<F>,
    max_degree: usize,
    budget: usize,
    definitions: &mut Definitions<F>,
) -> SymbolicExpression<F> {
    assert!(max_degree > 1, "the maximum degree must be at least 2");
    assert!(
        (1..=max_degree).contains(&budget),
        "the budget must be in 1..={max_degree}"
    );
    let plan = Plan::build(expr, max_degree);
    plan.rebuild(expr, budget, max_degree, definitions)
}

/// The cost tables of [`trim`]'s dynamic program, mirroring the expression
/// tree: `costs[b - 1]` is the minimal number of extractions needed to bring
/// the node to degree at most `b`.
struct Plan {
    costs: Vec<usize>,
    children: Vec<Plan>,
}

impl Plan {
    fn build<F: Field>(expr: &SymbolicExpression<F>, max_degree: usize) -> Self {
        use SymbolicExpression::{Add, Mul, Neg, Sub};
        let children = match expr {
            Add { x, y, .. } | Sub { x, y, .. } | Mul { x, y, .. } => {
                vec![Self::build(x, max_degree), Self::build(y, max_degree)]
            }
            Neg { x, .. } => vec![Self::build(x, max_degree)],
            _ => vec![],
        };
        let mut plan = Self {
            costs: vec![0; max_degree],
            children,
        };
        let degree = expr.degree_multiple();
        // Downwards, so that the extraction option (which defines the node at
        // `max_degree`) can read `costs[max_degree - 1]` already computed. At
        // `max_degree` itself the option is void: the definition would face
        // the very same problem.
        for b in (1..=max_degree).rev() {
            plan.costs[b - 1] = if degree <= b {
                0
            } else {
                let structural = plan.structural_cost(expr, b);
                if b < max_degree {
                    structural.min(1 + plan.costs[max_degree - 1])
                } else {
                    structural
                }
            };
        }
        plan
    }

    /// Minimal cost of fitting `expr` into `b` without extracting `expr`
    /// itself (subexpressions may still be extracted). `usize::MAX` when
    /// impossible (a product at budget 1).
    fn structural_cost<F: Field>(&self, expr: &SymbolicExpression<F>, b: usize) -> usize {
        use SymbolicExpression::{Add, Mul, Neg, Sub};
        match expr {
            Add { .. } | Sub { .. } => {
                let x = self.children[0].costs[b - 1];
                let y = self.children[1].costs[b - 1];
                x.saturating_add(y)
            }
            Neg { .. } => self.children[0].costs[b - 1],
            Mul { x, y, .. } => self
                .mul_splits(x, y, b)
                .map(|(_, _, cost)| cost)
                .min()
                .unwrap_or(usize::MAX),
            // Leaves have degree at most 1 and never reach this point.
            _ => unreachable!("structural_cost called on a leaf that already fits"),
        }
    }

    /// The admissible budget splits `(bx, by)` of a product, with their
    /// costs. A degree-0 side takes budget 0; any other side needs at least
    /// budget 1.
    fn mul_splits<'a, F: Field>(
        &'a self,
        x: &SymbolicExpression<F>,
        y: &SymbolicExpression<F>,
        b: usize,
    ) -> impl Iterator<Item = (usize, usize, usize)> + 'a {
        let lo_x = usize::from(x.degree_multiple() != 0);
        let lo_y = usize::from(y.degree_multiple() != 0);
        let hi_x = b.saturating_sub(lo_y);
        (lo_x..=hi_x).map(move |bx| {
            let by = b - bx;
            let cost_x = if bx == 0 {
                0
            } else {
                self.children[0].costs[bx - 1]
            };
            let cost_y = if by == 0 {
                0
            } else {
                self.children[1].costs[by - 1]
            };
            (bx, by, cost_x.saturating_add(cost_y))
        })
    }

    fn rebuild<F: Field>(
        &self,
        expr: &SymbolicExpression<F>,
        b: usize,
        max_degree: usize,
        definitions: &mut Definitions<F>,
    ) -> SymbolicExpression<F> {
        use SymbolicExpression::{Add, Mul, Neg, Sub};
        if expr.degree_multiple() <= b {
            return expr.clone();
        }
        let structural = self.structural_cost(expr, b);
        let extraction = if b < max_degree {
            1 + self.costs[max_degree - 1]
        } else {
            usize::MAX
        };
        if extraction < structural {
            // The definition itself is built at full degree; its inner
            // extractions are appended first, so definitions stay in
            // dependency order.
            let definition = self.rebuild(expr, max_degree, max_degree, definitions);
            let variable = definitions.fresh();
            definitions.definitions.push((variable, definition));
            return variable.into();
        }
        match expr {
            Add { x, y, .. } => {
                self.children[0].rebuild(x, b, max_degree, definitions)
                    + self.children[1].rebuild(y, b, max_degree, definitions)
            }
            Sub { x, y, .. } => {
                self.children[0].rebuild(x, b, max_degree, definitions)
                    - self.children[1].rebuild(y, b, max_degree, definitions)
            }
            Neg { x, .. } => -self.children[0].rebuild(x, b, max_degree, definitions),
            Mul { x, y, .. } => {
                let (bx, by, _) = self
                    .mul_splits(x, y, b)
                    .min_by_key(|&(_, _, cost)| cost)
                    .expect("a product with no admissible split is always extracted");
                self.children[0].rebuild(x, bx, max_degree, definitions)
                    * self.children[1].rebuild(y, by, max_degree, definitions)
            }
            _ => unreachable!("leaves always fit their budget"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Val;

    fn definitions_at(width: usize) -> Definitions<Val> {
        Definitions::new(Entry::Main { offset: 0 }, width)
    }

    /// Extends a row with the values of the definitions, in order.
    fn extended_row(mut row: Vec<Val>, definitions: &Definitions<Val>) -> Vec<Val> {
        for (variable, definition) in &definitions.definitions {
            assert_eq!(variable.index, row.len(), "unexpected variable index");
            let value: Val = definition.interpret(&row, None);
            row.push(value);
        }
        row
    }

    fn assert_trim(
        expr: &SymbolicExpression<Val>,
        max_degree: usize,
        budget: usize,
        expected_definitions: usize,
    ) {
        let width = 4;
        let mut definitions = definitions_at(width);
        let trimmed = trim(expr, max_degree, budget, &mut definitions);
        assert_eq!(definitions.definitions.len(), expected_definitions);
        assert!(trimmed.degree_multiple() <= budget);
        for (_, definition) in &definitions.definitions {
            assert!(definition.degree_multiple() <= max_degree);
        }
        // The trimmed expression must evaluate to the same value once the
        // row is extended with the definition values.
        let base = [3, 5, 7, 11].map(Val::from_u32).to_vec();
        let row = extended_row(base.clone(), &definitions);
        let original: Val = expr.interpret(&row, None);
        let evaluated: Val = trimmed.interpret(&row, None);
        assert_eq!(original, evaluated);
    }

    #[test]
    fn fitting_expressions_are_untouched() {
        let expr = var::<Val>(0) * var(1) + var(2) * var(3);
        let mut definitions = definitions_at(4);
        let trimmed = trim(&expr, 3, 2, &mut definitions);
        assert!(definitions.definitions.is_empty());
        assert_eq!(trimmed.degree_multiple(), 2);
    }

    #[test]
    fn product_splits_with_one_extraction() {
        // Degree 4 at budget 3: extract one side of the top product and keep
        // the other inline.
        let expr = (var::<Val>(0) * var(1)) * (var(2) * var(3));
        assert_trim(&expr, 3, 3, 1);
    }

    #[test]
    fn extracting_a_sum_beats_extracting_its_terms() {
        // Degree 2 at budget 1: extracting each product would cost 2;
        // extracting the whole sum costs 1.
        let expr = var::<Val>(0) * var(1) + var(2) * var(3);
        assert_trim(&expr, 3, 1, 1);
    }

    #[test]
    fn degree_zero_factors_are_free() {
        // The constant costs no budget: the degree-4 right side gets all of
        // it, needing a single extraction.
        let expr =
            SymbolicExpression::from(Val::TWO) * (var::<Val>(0) * (var(1) * (var(2) * var(3))));
        assert_trim(&expr, 3, 3, 1);
    }

    #[test]
    fn nested_extractions_stay_in_dependency_order() {
        // Degree 4 at budget 1: the whole product is extracted, and its
        // definition (degree 4 > max 3) needs one inner extraction, appended
        // before it.
        let expr = (var::<Val>(0) * var(1)) * (var(2) * var(3));
        assert_trim(&expr, 3, 1, 2);
    }
}
