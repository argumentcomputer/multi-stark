//! Lookup argument based on a grand product (GPA) with counters.
//!
//! Every lookup interaction is expressed with two message *sides*: a **push**
//! side that multiplies one global grand product and a **pull** side that
//! multiplies another; the protocol enforces that both products agree. A
//! message is the lookup challenge plus a fingerprint of the interaction's
//! arguments *prefixed by a counter*, and the two sides of one interaction
//! share the arguments but carry different counters:
//!
//! - [`Lookup::provide`]`(m, args)` pushes the message at counter `0` and
//!   pulls it at counter `m`, where `m` is the total number of times the
//!   claim is required.
//! - [`Lookup::require`]`(c, m, args)` pulls the message at counter `c` and
//!   pushes it at counter `c + m`, where `c` is the witnessed number of
//!   previous requires of the claim.
//!
//! For a claim provided once and required `m` times with counters
//! `0, 1, ..., m - 1`, the pushed multiset is `{0, 1, ..., m}` and the pulled
//! multiset is `{0, 1, ..., m}` as well — the counter chain telescopes and
//! the products balance. Counted operations are self-gating: a provide with
//! `m = 0` or a require with multiplicity `0` pushes and pulls the very same
//! message, contributing equally to both sides regardless of the values of
//! its arguments (or of a garbage counter on an inactive row).
//!
//! Within each circuit, one stage 2 column carries the running accumulator
//! `z`, constrained by the division-free transition
//! `pull_product * z_next = push_product * z`. The row products are built
//! symbolically from the lookup expressions — whose degrees are arbitrary —
//! and [trimmed](crate::builder::symbolic::trim) to the configuration's
//! constraint degree budget, extracting subexpressions into additional
//! stage 2 columns (*definitions*) when they do not fit.

use p3_air::{Air, BaseAir, ExtensionBuilder, WindowAccess};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, batch_multiplicative_inverse};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;

use crate::builder::{
    TwoStagedBuilder,
    symbolic::{Definitions, Entry, SymbolicExpression, SymbolicVariable, trim},
};

/// Each circuit is required to have 4 arguments for the second stage. Namely,
/// the lookup challenge, fingerprint challenge, current accumulator and next
/// accumulator.
pub const LOOKUP_PUBLIC_SIZE: usize = 4;

/// A lookup interaction in counter form: the same arguments are pushed at
/// counter `push_count` and pulled at counter `pull_count`. Use
/// [`Lookup::provide`] and [`Lookup::require`] instead of building the
/// counters by hand.
#[derive(Clone)]
pub struct Lookup<Expr> {
    pub push_count: Expr,
    pub pull_count: Expr,
    pub args: Vec<Expr>,
}

impl<Expr> Lookup<Expr> {
    /// Returns a [`Lookup`] with both counters zero and no arguments, whose
    /// sides cancel each other out.
    #[inline]
    pub fn empty() -> Self
    where
        Expr: PrimeCharacteristicRing,
    {
        Self {
            push_count: Expr::ZERO,
            pull_count: Expr::ZERO,
            args: vec![],
        }
    }

    /// "Providing" creates a claim that can be required `multiplicity`
    /// times: the message is pushed at counter `0` and pulled back at
    /// counter `multiplicity`, the value the require chain must reach.
    #[inline]
    pub fn provide(multiplicity: Expr, args: Vec<Expr>) -> Self
    where
        Expr: PrimeCharacteristicRing,
    {
        Self {
            push_count: Expr::ZERO,
            pull_count: multiplicity,
            args,
        }
    }

    /// "Requiring" consumes a provided claim: the message is pulled at the
    /// witnessed counter `count` (the number of previous requires of the
    /// claim) and pushed back at `count + multiplicity`. A boolean
    /// `multiplicity` acts as an activation guard: at `0` the two sides
    /// cancel.
    #[inline]
    pub fn require(count: Expr, multiplicity: Expr, args: Vec<Expr>) -> Self
    where
        Expr: Clone + std::ops::Add<Output = Expr>,
    {
        Self {
            push_count: count.clone() + multiplicity,
            pull_count: count,
            args,
        }
    }
}

pub struct LookupAir<A, F: Field> {
    pub inner_air: A,
    pub lookups: Vec<Lookup<SymbolicExpression<F>>>,
    pub preprocessed: Option<RowMajorMatrix<F>>,
}

impl<A: BaseAir<F>, F: Field> LookupAir<A, F> {
    pub fn new(inner_air: A, lookups: Vec<Lookup<SymbolicExpression<F>>>) -> Self {
        let preprocessed = inner_air.preprocessed_trace();
        Self {
            inner_air,
            lookups,
            preprocessed,
        }
    }
}

/// Computes a fingerprint of the coefficients using Horner's method.
#[inline]
pub(crate) fn fingerprint<F, I, Iter>(r: &F, coeffs: Iter) -> F
where
    F: PrimeCharacteristicRing,
    I: Into<F>,
    Iter: DoubleEndedIterator<Item = I>,
{
    coeffs
        .rev()
        .fold(F::ZERO, |acc, coeff| acc * r.clone() + coeff.into())
}

/// The message of one side of a lookup: the lookup challenge plus the
/// fingerprint of the counter followed by the arguments. The counter comes
/// *first* so that zero padding at the end of the arguments does not change
/// the message (Horner evaluation is transparent to trailing zeros).
#[inline]
fn message<F, EF, I>(lookup_challenge: EF, fingerprint_challenge: &EF, count: F, args: I) -> EF
where
    F: Into<EF>,
    EF: PrimeCharacteristicRing,
    I: DoubleEndedIterator<Item = F>,
{
    lookup_challenge
        + fingerprint(
            fingerprint_challenge,
            std::iter::once(count.into()).chain(args.map(Into::into)),
        )
}

/// Folds the claims into the two sides of the grand product. The claim set
/// behaves as one *require* of each claim at counter zero: the verifier
/// pulls each claim message at counter `0` and pushes it at counter `1`.
///
/// Returns `(initial, expected_final)`: the accumulator chain across the
/// circuits starts at the pushed product and, when every lookup balances,
/// must end at the pulled product.
pub fn fold_claims<F: Field, EF: ExtensionField<F>>(
    claims: &[&[F]],
    lookup_challenge: EF,
    fingerprint_challenge: &EF,
) -> (EF, EF) {
    let mut pushed = EF::ONE;
    let mut pulled = EF::ONE;
    for claim in claims {
        let args = || claim.iter().map(|&x| EF::from(x));
        pulled *= message(lookup_challenge, fingerprint_challenge, EF::ZERO, args());
        pushed *= message(lookup_challenge, fingerprint_challenge, EF::ONE, args());
    }
    (pushed, pulled)
}

/// A circuit's lookups lowered for the grand-product argument: the symbolic
/// per-row products of the pulled and pushed messages, trimmed to the
/// constraint degree budget, together with the definitions of the stage 2
/// columns extracted by the trimming.
pub struct GpaLookups<EF> {
    /// Product of the row's pulled messages, of degree at most
    /// `max_degree - 2` (it multiplies an accumulator column inside a
    /// row-selector-gated constraint).
    pub pull_product: SymbolicExpression<EF>,
    /// Product of the row's pushed messages, same degree bound.
    pub push_product: SymbolicExpression<EF>,
    /// Extracted definitions: `definitions[i]` defines stage 2 column
    /// `1 + i` (column 0 is the running accumulator). Definitions only
    /// reference stage 1 values, the challenges, and *earlier* definitions.
    pub definitions: Vec<(SymbolicVariable<EF>, SymbolicExpression<EF>)>,
}

impl<EF: Field> GpaLookups<EF> {
    /// Lowers a circuit's lookups, trimming both message products to fit
    /// constraints of degree at most `max_degree`. Both products share one
    /// definition list, so common extractions cost a single column.
    pub fn lower<F: Field>(lookups: &[Lookup<SymbolicExpression<F>>], max_degree: usize) -> Self
    where
        EF: ExtensionField<F>,
    {
        assert!(
            max_degree >= 3,
            "the grand-product argument needs a constraint degree budget of at least 3"
        );
        let beta: SymbolicExpression<EF> = SymbolicVariable::new(Entry::Stage2Public, 0).into();
        let gamma: SymbolicExpression<EF> = SymbolicVariable::new(Entry::Stage2Public, 1).into();
        let side = |count_of: fn(&Lookup<SymbolicExpression<F>>) -> &SymbolicExpression<F>| {
            lookups
                .iter()
                .map(|lookup| {
                    message(
                        beta.clone(),
                        &gamma,
                        count_of(lookup).lift(),
                        lookup.args.iter().map(SymbolicExpression::lift),
                    )
                })
                .product::<SymbolicExpression<EF>>()
        };
        let pull_product = side(|lookup| &lookup.pull_count);
        let push_product = side(|lookup| &lookup.push_count);
        // The products multiply a stage 2 column (degree 1) inside a
        // row-selector-gated constraint (one more degree), so they must fit
        // in `max_degree - 2`.
        let budget = max_degree - 2;
        let mut definitions = Definitions::new(Entry::Stage2 { offset: 0 }, 1);
        let pull_product = trim(&pull_product, max_degree, budget, &mut definitions);
        let push_product = trim(&push_product, max_degree, budget, &mut definitions);
        Self {
            pull_product,
            push_product,
            definitions: definitions.definitions,
        }
    }

    /// One column for the running accumulator plus one per extracted
    /// definition.
    pub fn stage_2_width(&self) -> usize {
        1 + self.definitions.len()
    }
}

/// A circuit's AIR together with its lowered lookup argument. This is the
/// AIR the prover and verifier actually evaluate: the inner AIR's stage 1
/// constraints plus the grand-product constraints over the stage 2 trace.
pub(crate) struct GpaAir<'a, A, F: Field, EF> {
    pub air: &'a LookupAir<A, F>,
    pub gpa: &'a GpaLookups<EF>,
}

impl<A, F, EF> BaseAir<F> for GpaAir<'_, A, F, EF>
where
    A: BaseAir<F>,
    F: Field,
    EF: Sync,
{
    fn width(&self) -> usize {
        self.air.inner_air.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.air.preprocessed.clone()
    }
}

impl<A, F, EF, AB> Air<AB> for GpaAir<'_, A, F, EF>
where
    A: Air<AB>,
    F: Field,
    EF: ExtensionField<F>,
    AB: TwoStagedBuilder<F = F, EF = EF>,
{
    fn eval(&self, builder: &mut AB) {
        if self.air.preprocessed.is_some() {
            let preprocessed = builder.preprocessed().clone();
            let preprocessed_row = preprocessed.current_slice();
            self.eval_with_preprocessed_row(builder, Some(preprocessed_row))
        } else {
            self.eval_with_preprocessed_row(builder, None)
        }
    }
}

impl<A, F: Field, EF: ExtensionField<F>> GpaAir<'_, A, F, EF> {
    fn eval_with_preprocessed_row<AB>(&self, builder: &mut AB, preprocessed_row: Option<&[AB::Var]>)
    where
        A: Air<AB>,
        AB: TwoStagedBuilder<F = F, EF = EF>,
    {
        // Call `eval` for regular stage 1 constraints.
        self.air.inner_air.eval(builder);

        // Extract challenges and accumulators from stage 2 public values.
        let stage_2_public_values = builder.stage_2_public_values().to_vec();
        debug_assert_eq!(stage_2_public_values.len(), LOOKUP_PUBLIC_SIZE);
        let acc_in = stage_2_public_values[2];
        let acc_out: AB::ExprEF = stage_2_public_values[3].into();

        // Bind relevant variables to construct the stage 2 constraints.
        let stage_2 = builder.stage_2();
        let stage_2_row = stage_2.row_slice(0).unwrap();
        let stage_2_next_row = stage_2.row_slice(1).unwrap();
        let acc = stage_2_row[0];
        let acc_expr: AB::ExprEF = acc.into();
        let acc_next: AB::ExprEF = stage_2_next_row[0].into();

        let main = builder.main();
        let main_row = main.current_slice();

        // Resolves the variables of the lowered lookup expressions within
        // the constraint evaluation windows. Only current-row (offset 0)
        // entries can appear in lookup expressions and their definitions.
        let resolve = |variable: &SymbolicVariable<EF>| -> AB::ExprEF {
            match variable.entry {
                Entry::Main { offset: 0 } => {
                    let value: AB::Expr = main_row[variable.index].into();
                    value.into()
                }
                Entry::Preprocessed { offset: 0 } => {
                    let row = preprocessed_row.expect("circuit has no preprocessed trace");
                    let value: AB::Expr = row[variable.index].into();
                    value.into()
                }
                Entry::Stage2 { offset: 0 } => stage_2_row[variable.index].into(),
                Entry::Stage2Public => stage_2_public_values[variable.index].into(),
                _ => unimplemented!("unsupported variable entry in lookup expressions"),
            }
        };

        // Each extracted stage 2 column must equal its definition.
        let definitions = self
            .gpa
            .definitions
            .iter()
            .map(|(variable, definition)| {
                let column: AB::ExprEF = stage_2_row[variable.index].into();
                (column, definition.interpret_with(&resolve))
            })
            .collect::<Vec<_>>();
        let pull_product: AB::ExprEF = self.gpa.pull_product.interpret_with(&resolve);
        let push_product: AB::ExprEF = self.gpa.push_product.interpret_with(&resolve);
        for (column, definition) in definitions {
            builder.assert_eq_ext(column, definition);
        }

        // The initial accumulator value must be set correctly.
        builder.when_first_row().assert_eq_ext(acc, acc_in);

        // Running product transition, division-free by cross-multiplication:
        // the pulled messages divide the accumulator and the pushed messages
        // multiply it.
        builder.when_transition().assert_eq_ext(
            pull_product.clone() * acc_next,
            push_product.clone() * acc_expr.clone(),
        );

        // On the last row the completed product must reach the expected
        // final accumulator.
        builder
            .when_last_row()
            .assert_eq_ext(pull_product * acc_out, push_product * acc_expr);
    }
}

/// Concrete lookup values of one circuit, stored flat.
///
/// Every row of a circuit has the same lookups (each with a fixed number of
/// arguments), so all counters and argument values are kept in row-major
/// vectors instead of nested per-row, per-lookup allocations.
#[derive(Clone)]
pub struct LookupValues<F> {
    /// Number of trace rows.
    height: usize,
    /// Number of lookups per row.
    num_lookups: usize,
    /// Push-side counter of each (row, lookup); `height * num_lookups` values.
    push_counts: Vec<F>,
    /// Pull-side counter of each (row, lookup); `height * num_lookups` values.
    pull_counts: Vec<F>,
    /// Offset of each lookup's arguments within a row's argument block;
    /// `num_lookups + 1` entries, the last being the block width.
    arg_offsets: Vec<usize>,
    /// Concatenated argument values, row-major; `height * arg_offsets.last()`
    /// values.
    args: Vec<F>,
}

impl<F: Field> LookupValues<F> {
    /// Evaluates the symbolic lookups of a circuit on every row of its trace.
    pub fn compute(
        lookups: &[Lookup<SymbolicExpression<F>>],
        trace: &RowMajorMatrix<F>,
        preprocessed: Option<&RowMajorMatrix<F>>,
    ) -> Self {
        let height = trace.height();
        let num_lookups = lookups.len();
        let mut arg_offsets = Vec::with_capacity(num_lookups + 1);
        arg_offsets.push(0);
        for lookup in lookups {
            arg_offsets.push(arg_offsets.last().unwrap() + lookup.args.len());
        }
        let args_width = *arg_offsets.last().unwrap();
        let mut push_counts = Vec::with_capacity(height * num_lookups);
        let mut pull_counts = Vec::with_capacity(height * num_lookups);
        let mut args = Vec::with_capacity(height * args_width);
        let mut eval_row = |row: &[F], preprocessed_row: Option<&[F]>| {
            for lookup in lookups {
                push_counts.push(lookup.push_count.interpret(row, preprocessed_row));
                pull_counts.push(lookup.pull_count.interpret(row, preprocessed_row));
                for arg in &lookup.args {
                    args.push(arg.interpret(row, preprocessed_row));
                }
            }
        };
        match preprocessed {
            Some(preprocessed) => trace
                .row_slices()
                .zip(preprocessed.row_slices())
                .for_each(|(row, preprocessed_row)| eval_row(row, Some(preprocessed_row))),
            None => trace.row_slices().for_each(|row| eval_row(row, None)),
        }
        Self {
            height,
            num_lookups,
            push_counts,
            pull_counts,
            arg_offsets,
            args,
        }
    }

    /// Builds flat lookup values from per-row, per-lookup concrete lookups.
    ///
    /// Every row must have the same number of lookups. Argument counts may
    /// vary across rows within a lookup slot (e.g. rows holding
    /// [`Lookup::empty`]); shorter argument lists are zero-padded to the
    /// slot's maximum width. Padding is transparent to the protocol: the
    /// message fingerprint is a Horner evaluation and the counter comes
    /// first, so trailing zero coefficients do not change its value.
    ///
    /// # Panics
    /// Panics if rows have differing numbers of lookups.
    pub fn from_rows(rows: Vec<Vec<Lookup<F>>>) -> Self {
        let height = rows.len();
        let num_lookups = rows.first().map_or(0, |row| row.len());
        let mut slot_widths = vec![0usize; num_lookups];
        for row in &rows {
            assert_eq!(
                row.len(),
                num_lookups,
                "every row must have the same number of lookups"
            );
            for (width, lookup) in slot_widths.iter_mut().zip(row) {
                *width = (*width).max(lookup.args.len());
            }
        }
        let mut arg_offsets = Vec::with_capacity(num_lookups + 1);
        arg_offsets.push(0);
        for width in &slot_widths {
            arg_offsets.push(arg_offsets.last().unwrap() + width);
        }
        let args_width = *arg_offsets.last().unwrap();
        let mut push_counts = Vec::with_capacity(height * num_lookups);
        let mut pull_counts = Vec::with_capacity(height * num_lookups);
        let mut args = Vec::with_capacity(height * args_width);
        for row in rows {
            for (lookup, &width) in row.into_iter().zip(&slot_widths) {
                push_counts.push(lookup.push_count);
                pull_counts.push(lookup.pull_count);
                let padding = width - lookup.args.len();
                args.extend(lookup.args);
                args.extend(std::iter::repeat_n(F::ZERO, padding));
            }
        }
        Self {
            height,
            num_lookups,
            push_counts,
            pull_counts,
            arg_offsets,
            args,
        }
    }

    /// Returns an allocation-free builder; see [`LookupValuesBuilder`].
    pub fn builder(height: usize, slot_arg_widths: &[usize]) -> LookupValuesBuilder<F> {
        LookupValuesBuilder::new(height, slot_arg_widths)
    }

    /// The arguments of the given lookup on the given row.
    #[inline]
    fn args_at(&self, row: usize, lookup: usize) -> &[F] {
        let start = row * self.arg_offsets[self.num_lookups];
        &self.args[start + self.arg_offsets[lookup]..start + self.arg_offsets[lookup + 1]]
    }
}

/// Per-circuit inputs for stage 2 trace generation.
pub struct Stage2Inputs<'a, F: Field, EF> {
    /// The circuit's concrete lookup values.
    pub lookups: &'a LookupValues<F>,
    /// The circuit's lowered lookups.
    pub gpa: &'a GpaLookups<EF>,
    /// The circuit's main trace; required when the lowering has definitions
    /// (their values are functions of the stage 1 values).
    pub main: Option<&'a RowMajorMatrix<F>>,
    /// The circuit's preprocessed trace, if any.
    pub preprocessed: Option<&'a RowMajorMatrix<F>>,
}

/// Computes the stage 2 traces and the intermediate accumulators for each
/// circuit given a lookup challenge, a fingerprint challenge and the initial
/// accumulator value (the pushed product of the claims).
///
/// Each circuit's trace has one column for the running accumulator followed
/// by one column per extracted definition. The accumulator satisfies
/// `z_next = z * push_product / pull_product` row by row, entering the next
/// circuit where it left the previous one.
pub fn stage_2_traces<F: Field, EF: ExtensionField<F>>(
    circuits: &[Stage2Inputs<'_, F, EF>],
    lookup_challenge: EF,
    fingerprint_challenge: &EF,
    mut accumulator: EF,
) -> (Vec<RowMajorMatrix<EF>>, Vec<EF>) {
    let mut traces = Vec::with_capacity(circuits.len());
    let mut intermediate_accumulators = Vec::with_capacity(circuits.len());
    for circuit in circuits {
        let lookups = circuit.lookups;
        let height = lookups.height;
        let num_definitions = circuit.gpa.definitions.len();

        if lookups.num_lookups == 0 {
            // No lookups: both products are 1 (and there is nothing to
            // extract), so the accumulator just repeats.
            debug_assert_eq!(num_definitions, 0);
            traces.push(RowMajorMatrix::new(vec![accumulator; height], 1));
            intermediate_accumulators.push(accumulator);
            continue;
        }

        // Pull- and push-side message products of every row.
        let _g = tracing::info_span!("stark/lookup_messages").entered();
        let (pull_products, push_products): (Vec<EF>, Vec<EF>) = (0..height)
            .into_par_iter()
            .map(|row| {
                let mut pull = EF::ONE;
                let mut push = EF::ONE;
                for lookup in 0..lookups.num_lookups {
                    let index = row * lookups.num_lookups + lookup;
                    let args = lookups.args_at(row, lookup);
                    pull *= message(
                        lookup_challenge,
                        fingerprint_challenge,
                        lookups.pull_counts[index],
                        args.iter().cloned(),
                    );
                    push *= message(
                        lookup_challenge,
                        fingerprint_challenge,
                        lookups.push_counts[index],
                        args.iter().cloned(),
                    );
                }
                (pull, push)
            })
            .unzip();
        drop(_g);

        // Only the pulled products need inverting, one element per row.
        let pull_inverses = tracing::info_span!("stark/batch_inverse")
            .in_scope(|| batch_multiplicative_inverse(&pull_products));
        drop(pull_products);

        // Values of the extracted definition columns, row-major.
        let definition_values: Vec<Vec<EF>> = if num_definitions == 0 {
            vec![]
        } else {
            let _g = tracing::info_span!("stark/lookup_definitions").entered();
            let main = circuit
                .main
                .expect("main trace required to fill definition columns");
            assert_eq!(main.height(), height, "main trace height mismatch");
            let challenges = [lookup_challenge, *fingerprint_challenge];
            (0..height)
                .into_par_iter()
                .map(|row| {
                    let main_row = main.row_slice(row).unwrap();
                    let preprocessed_row = circuit
                        .preprocessed
                        .map(|preprocessed| preprocessed.row_slice(row).unwrap());
                    let mut values: Vec<EF> = Vec::with_capacity(num_definitions);
                    for (variable, definition) in &circuit.gpa.definitions {
                        debug_assert_eq!(variable.index, values.len() + 1);
                        let value = definition.interpret_with(&|v: &SymbolicVariable<EF>| {
                            match v.entry {
                                Entry::Main { offset: 0 } => main_row[v.index].into(),
                                Entry::Preprocessed { offset: 0 } => preprocessed_row
                                    .as_ref()
                                    .expect("circuit has no preprocessed trace")[v.index]
                                    .into(),
                                // Column 0 is the accumulator; definitions
                                // only reference earlier definitions.
                                Entry::Stage2 { offset: 0 } => values[v.index - 1],
                                Entry::Stage2Public => challenges[v.index],
                                _ => unimplemented!(
                                    "unsupported variable entry in lookup expressions"
                                ),
                            }
                        });
                        values.push(value);
                    }
                    values
                })
                .collect()
        };

        // Assemble the trace: running accumulator followed by definitions.
        let _g = tracing::info_span!("stark/lookup_traces").entered();
        let width = 1 + num_definitions;
        let mut values = Vec::with_capacity(height * width);
        for row in 0..height {
            values.push(accumulator);
            if num_definitions != 0 {
                values.extend_from_slice(&definition_values[row]);
            }
            accumulator *= push_products[row] * pull_inverses[row];
        }
        drop(_g);
        traces.push(RowMajorMatrix::new(values, width));
        intermediate_accumulators.push(accumulator);
    }
    (traces, intermediate_accumulators)
}

/// Incremental, allocation-free constructor for [`LookupValues`].
///
/// Rows start zeroed — both counters zero and zero arguments in every slot,
/// i.e. [`Lookup::empty`] — and are filled in place through the parallel row
/// writers of [`Self::rows_mut`]. Values are written directly into the
/// final flat storage, so no per-lookup allocation ever happens.
pub struct LookupValuesBuilder<F> {
    height: usize,
    num_lookups: usize,
    arg_offsets: Vec<usize>,
    /// Row stride of `args`: the total argument width, with a minimum of 1
    /// so row chunking stays well-defined when every slot is argument-less.
    row_stride: usize,
    push_counts: Vec<F>,
    pull_counts: Vec<F>,
    args: Vec<F>,
}

impl<F: Field> LookupValuesBuilder<F> {
    /// `slot_arg_widths[j]` is the maximum number of arguments of lookup
    /// slot `j`. Rows writing fewer arguments leave zero padding, which does
    /// not change the message fingerprint (a Horner evaluation with the
    /// counter as the leading coefficient).
    pub fn new(height: usize, slot_arg_widths: &[usize]) -> Self {
        let num_lookups = slot_arg_widths.len();
        let mut arg_offsets = Vec::with_capacity(num_lookups + 1);
        arg_offsets.push(0);
        for width in slot_arg_widths {
            arg_offsets.push(arg_offsets.last().unwrap() + width);
        }
        let row_stride = (*arg_offsets.last().unwrap()).max(1);
        Self {
            height,
            num_lookups,
            arg_offsets,
            row_stride,
            push_counts: F::zero_vec(height * num_lookups),
            pull_counts: F::zero_vec(height * num_lookups),
            args: F::zero_vec(height * row_stride),
        }
    }

    /// Row writers, one per row, in row order. Returned as a `Vec` so the
    /// caller can parallelize with its own runtime (e.g. rayon's
    /// `into_par_iter`, zipped with a parallel iterator over trace rows)
    /// regardless of how this crate's `parallel` feature is set.
    ///
    /// # Panics
    /// Panics if the builder has no lookup slots (nothing to write; call
    /// [`Self::finish`] directly).
    pub fn rows_mut(&mut self) -> Vec<LookupRowMut<'_, F>> {
        assert!(self.num_lookups > 0, "builder has no lookup slots");
        let Self {
            num_lookups,
            arg_offsets,
            row_stride,
            push_counts,
            pull_counts,
            args,
            ..
        } = self;
        let arg_offsets = arg_offsets.as_slice();
        push_counts
            .chunks_exact_mut(*num_lookups)
            .zip(pull_counts.chunks_exact_mut(*num_lookups))
            .zip(args.chunks_exact_mut(*row_stride))
            .map(|((push_counts, pull_counts), args)| LookupRowMut {
                push_counts,
                pull_counts,
                args,
                arg_offsets,
            })
            .collect()
    }

    /// Finalizes into [`LookupValues`].
    pub fn finish(mut self) -> LookupValues<F> {
        if self.arg_offsets[self.num_lookups] == 0 {
            // Every slot is argument-less; drop the dummy stride storage.
            self.args = vec![];
        }
        LookupValues {
            height: self.height,
            num_lookups: self.num_lookups,
            push_counts: self.push_counts,
            pull_counts: self.pull_counts,
            arg_offsets: self.arg_offsets,
            args: self.args,
        }
    }
}

/// Mutable view of one row of a [`LookupValuesBuilder`].
pub struct LookupRowMut<'a, F> {
    push_counts: &'a mut [F],
    pull_counts: &'a mut [F],
    args: &'a mut [F],
    arg_offsets: &'a [usize],
}

impl<F: Field> LookupRowMut<'_, F> {
    /// Writes lookup slot `slot` with "provide" semantics; see
    /// [`Lookup::provide`]. Fewer arguments than the slot's width leave zero
    /// padding.
    ///
    /// # Panics
    /// Panics if `args` exceeds the slot's width.
    #[inline]
    pub fn provide(&mut self, slot: usize, multiplicity: F, args: &[F]) {
        self.write(slot, F::ZERO, multiplicity, args);
    }

    /// Writes lookup slot `slot` with "require" semantics; see
    /// [`Lookup::require`].
    ///
    /// # Panics
    /// Panics if `args` exceeds the slot's width.
    #[inline]
    pub fn require(&mut self, slot: usize, count: F, multiplicity: F, args: &[F]) {
        self.write(slot, count + multiplicity, count, args);
    }

    #[inline]
    fn write(&mut self, slot: usize, push_count: F, pull_count: F, args: &[F]) {
        let start = self.arg_offsets[slot];
        let end = self.arg_offsets[slot + 1];
        assert!(
            args.len() <= end - start,
            "too many arguments for lookup slot {slot}"
        );
        self.push_counts[slot] = push_count;
        self.pull_counts[slot] = pull_count;
        self.args[start..start + args.len()].copy_from_slice(args);
        self.args[start + args.len()..end].fill(F::ZERO);
    }
}

impl<A, F> BaseAir<F> for LookupAir<A, F>
where
    A: BaseAir<F>,
    F: Field,
{
    fn width(&self) -> usize {
        self.inner_air.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.preprocessed.clone()
    }
}

#[cfg(test)]
mod tests {
    use p3_air::{AirBuilder, WindowAccess};
    use p3_field::Field;

    use crate::{
        builder::symbolic::var,
        system::{ProverKey, System, SystemWitness},
        types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val},
    };

    use super::*;

    enum CS {
        Even,
        Odd,
        /// A circuit on its own channel that nothing references; used by the
        /// sparse-activation tests as the deactivated table.
        Dead,
    }
    impl<F> BaseAir<F> for CS {
        fn width(&self) -> usize {
            6
        }
    }
    impl CS {
        fn lookups(&self) -> Vec<Lookup<SymbolicExpression<Val>>> {
            let multiplicity = var(0);
            let input = var(1);
            let input_is_zero = var(3);
            let input_not_zero = var(4);
            let recursion_output = var(5);
            let even_index = Val::ZERO.into();
            let odd_index = Val::ONE.into();
            let one: SymbolicExpression<_> = Val::ONE.into();
            let zero = SymbolicExpression::ZERO;
            // Every claim in this system is required exactly once (either by
            // the initial claim or by the recursion of the next step), so
            // provide multiplicities are boolean and require counters are
            // all zero.
            match self {
                Self::Even => vec![
                    Lookup::provide(
                        multiplicity,
                        vec![
                            even_index,
                            input.clone(),
                            input_not_zero.clone() * recursion_output.clone() + input_is_zero,
                        ],
                    ),
                    Lookup::require(
                        zero,
                        input_not_zero,
                        vec![odd_index, input - one, recursion_output],
                    ),
                ],
                Self::Odd => vec![
                    Lookup::provide(
                        multiplicity,
                        vec![
                            odd_index,
                            input.clone(),
                            input_not_zero.clone() * recursion_output.clone(),
                        ],
                    ),
                    Lookup::require(
                        zero,
                        input_not_zero,
                        vec![even_index, input - one, recursion_output],
                    ),
                ],
                Self::Dead => vec![Lookup::provide(
                    multiplicity,
                    vec![Val::from_u32(2).into(), input],
                )],
            }
        }
    }
    impl<AB> Air<AB> for CS
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            // both even and odd have the same constraints, they only differ on the lookups
            let main = builder.main();
            let local = main.current_slice();
            let multiplicity = local[0];
            let input = local[1];
            let input_inverse = local[2];
            let input_is_zero = local[3];
            let input_not_zero = local[4];
            builder.assert_bools([input_is_zero, input_not_zero]);
            builder
                .when(multiplicity)
                .assert_one(input_is_zero + input_not_zero);
            builder.when(input_is_zero).assert_zero(input);
            builder
                .when(input_not_zero)
                .assert_one(input * input_inverse);
        }
    }
    const COMMITMENT_PARAMETERS: CommitmentParameters = CommitmentParameters {
        log_blowup: 1,
        cap_height: 0,
    };

    const FRI_PARAMETERS: FriParameters = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 64,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };

    fn system() -> (
        System<GoldilocksBlake3Config, CS>,
        ProverKey<GoldilocksBlake3Config>,
    ) {
        let config = GoldilocksBlake3Config::new(COMMITMENT_PARAMETERS, FRI_PARAMETERS);
        let even = LookupAir::new(CS::Even, CS::Even.lookups());
        let odd = LookupAir::new(CS::Odd, CS::Odd.lookups());
        System::new(config, [even, odd])
    }

    fn witness_traces() -> Vec<RowMajorMatrix<Val>> {
        let f = Val::from_u32;
        #[rustfmt::skip]
        let traces = vec![
            RowMajorMatrix::new(
                vec![
                    // row 1
                    f(1), f(4), f(4).inverse(), f(0), f(1), f(1),
                    // row 2
                    f(1), f(2), f(2).inverse(), f(0), f(1), f(1),
                    // row 3
                    f(1), f(0), f(0), f(1), f(0), f(0),
                    // row 4
                    f(0), f(0), f(0), f(0), f(0), f(0),
                ],
                6,
            ),
            RowMajorMatrix::new(
                vec![
                    // row 1
                    f(1), f(3), f(3).inverse(), f(0), f(1), f(1),
                    // row 2
                    f(1), f(1), f(1).inverse(), f(0), f(1), f(1),
                    // row 3
                    f(0), f(0), f(0), f(0), f(0), f(0),
                    // row 4
                    f(0), f(0), f(0), f(0), f(0), f(0),
                ],
                6,
            ),
        ];
        traces
    }

    fn witness(system: &System<GoldilocksBlake3Config, CS>) -> SystemWitness<Val> {
        SystemWitness::from_stage_1(witness_traces(), system)
    }

    /// The builder must produce the exact same flat storage as the nested
    /// conversion, including zero padding for short and empty slots.
    #[test]
    fn builder_matches_from_rows() {
        let f = Val::from_u32;
        let rows = vec![
            vec![
                Lookup::provide(f(1), vec![f(2), f(3)]),
                Lookup::require(f(4), f(1), vec![f(5), f(6), f(7)]),
            ],
            vec![Lookup::empty(), Lookup::provide(f(8), vec![f(9)])],
        ];
        let a = LookupValues::from_rows(rows);
        let mut builder = LookupValues::builder(2, &[2, 3]);
        for (i, row) in builder.rows_mut().iter_mut().enumerate() {
            if i == 0 {
                row.provide(0, f(1), &[f(2), f(3)]);
                row.require(1, f(4), f(1), &[f(5), f(6), f(7)]);
            } else {
                row.provide(1, f(8), &[f(9)]);
            }
        }
        let b = builder.finish();
        assert_eq!(a.height, b.height);
        assert_eq!(a.num_lookups, b.num_lookups);
        assert_eq!(a.arg_offsets, b.arg_offsets);
        assert_eq!(a.push_counts, b.push_counts);
        assert_eq!(a.pull_counts, b.pull_counts);
        assert_eq!(a.args, b.args);
    }

    #[test]
    fn lookup_test() {
        let (system, key) = system();
        let witness = witness(&system);
        let f = Val::from_u32;
        let claim = &[f(0), f(4), f(1)];
        let proof = system.prove(&key, claim, witness);
        system.verify(claim, &proof).unwrap();
    }

    /// A three-circuit system where the third circuit (`Dead`, its own
    /// channel, referenced by nothing) gets an empty trace: the prover must
    /// deactivate it, the proof must carry per-circuit data for the two
    /// active circuits only, and verification must accept.
    #[test]
    fn sparse_inactive_circuit_accepted() {
        let config = GoldilocksBlake3Config::new(COMMITMENT_PARAMETERS, FRI_PARAMETERS);
        let even = LookupAir::new(CS::Even, CS::Even.lookups());
        let odd = LookupAir::new(CS::Odd, CS::Odd.lookups());
        let dead = LookupAir::new(CS::Dead, CS::Dead.lookups());
        let (system, key) = System::new(config, [even, odd, dead]);
        let mut witness = witness_traces();
        witness.push(RowMajorMatrix::new(vec![], 6));
        let witness = SystemWitness::from_stage_1(witness, &system);
        let f = Val::from_u32;
        let claim = &[f(0), f(4), f(1)];
        let proof = system.prove(&key, claim, witness);
        assert_eq!(proof.active, vec![true, true, false]);
        assert_eq!(proof.log_degrees.len(), 2);
        assert_eq!(proof.intermediate_accumulators.len(), 2);
        assert_eq!(proof.stage_1_opened_values.len(), 2);
        system.verify(claim, &proof).unwrap();
    }

    /// Tampering with the activation bitmap must be rejected: activating a
    /// circuit the proof carries no data for, or deactivating one it does.
    #[test]
    fn sparse_bitmap_tamper_rejected() {
        let config = GoldilocksBlake3Config::new(COMMITMENT_PARAMETERS, FRI_PARAMETERS);
        let even = LookupAir::new(CS::Even, CS::Even.lookups());
        let odd = LookupAir::new(CS::Odd, CS::Odd.lookups());
        let dead = LookupAir::new(CS::Dead, CS::Dead.lookups());
        let (system, key) = System::new(config, [even, odd, dead]);
        let mut witness = witness_traces();
        witness.push(RowMajorMatrix::new(vec![], 6));
        let witness = SystemWitness::from_stage_1(witness, &system);
        let f = Val::from_u32;
        let claim = &[f(0), f(4), f(1)];
        let mut proof = system.prove(&key, claim, witness);
        proof.active[2] = true;
        assert!(system.verify(claim, &proof).is_err());
        proof.active[2] = false;
        proof.active[1] = false;
        assert!(system.verify(claim, &proof).is_err());
    }

    /// Deactivating a circuit the claim's execution NEEDS (here: emptying the
    /// Odd table while Even still requires from the odd channel) must leave
    /// the grand product unbalanced and be rejected.
    #[test]
    fn sparse_needed_circuit_rejected() {
        let (system, key) = system();
        let mut traces = witness_traces();
        // Empty the Odd table: Even's requires from the odd channel and the
        // claim's chain can no longer be matched.
        traces[1] = RowMajorMatrix::new(vec![], 6);
        let witness = SystemWitness::from_stage_1(traces, &system);
        let f = Val::from_u32;
        let claim = &[f(0), f(4), f(1)];
        let proof = system.prove(&key, claim, witness);
        assert_eq!(proof.active, vec![true, false]);
        assert!(system.verify(claim, &proof).is_err());
    }

    #[test]
    fn test_claim_split_rejected() {
        let (system, key) = system();
        let witness = witness(&system);
        let f = Val::from_u32;
        // Prove a single claim, then attempt to verify with the same values
        // regrouped into two claims. The transcript is length-prefixed, so the
        // regrouped claims must yield different challenges and fail.
        let claim = &[f(0), f(4), f(1)];
        let proof = system.prove(&key, claim, witness);
        let split_claims: &[&[Val]] = &[&[f(0), f(4)], &[f(1)]];
        let result = system.verify_multiple_claims(split_claims, &proof);
        assert!(result.is_err());
    }
}
