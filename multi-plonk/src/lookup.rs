//! LogUp lookup argument.
//!
//! Each circuit may push or pull lookup tuples each row. The argument compresses
//! tuples into single field elements via a fingerprint (Horner evaluation under
//! a per-proof challenge `gamma`), then offsets each by another challenge
//! `beta`. The running accumulator
//!
//! ```text
//!   acc_{i+1} = acc_i + Σ_j  multiplicity_{i,j} / (beta + fingerprint(gamma, args_{i,j}))
//! ```
//!
//! must telescope to zero across all circuits if the multiset balance holds.
//!
//! Stage 2 trace per circuit: column 0 is the running accumulator at the start
//! of each row; columns 1..=L_i are the per-lookup message inverses.

use ark_ff::{Field, One, Zero};

use crate::air::{Air, AirBuilder, BaseAir};
use crate::builder::symbolic::SymbolicExpression;
use crate::matrix::Matrix;
use crate::types::Val;

/// Stage-2 public values used by the lookup argument: lookup challenge,
/// fingerprint challenge, current accumulator, next accumulator.
pub const LOOKUP_PUBLIC_SIZE: usize = 4;

#[derive(Clone)]
pub struct Lookup<Expr> {
    pub multiplicity: Expr,
    pub args: Vec<Expr>,
}

impl<Expr> Lookup<Expr> {
    #[inline]
    pub fn empty() -> Self
    where
        Expr: From<Val>,
    {
        Self {
            multiplicity: Expr::from(Val::zero()),
            args: vec![],
        }
    }

    /// Adds a claim to the lookup channel.
    #[inline]
    pub fn push(multiplicity: Expr, args: Vec<Expr>) -> Self {
        Self { multiplicity, args }
    }

    /// Removes a claim from the lookup channel.
    #[inline]
    pub fn pull(multiplicity: Expr, args: Vec<Expr>) -> Self
    where
        Expr: std::ops::Neg<Output = Expr>,
    {
        Self {
            multiplicity: -multiplicity,
            args,
        }
    }
}

pub struct LookupAir<A> {
    pub inner_air: A,
    pub lookups: Vec<Lookup<SymbolicExpression>>,
    pub preprocessed: Option<Matrix<Val>>,
}

impl<A: BaseAir> LookupAir<A> {
    pub fn new(inner_air: A, lookups: Vec<Lookup<SymbolicExpression>>) -> Self {
        let preprocessed = inner_air.preprocessed_trace();
        Self {
            inner_air,
            lookups,
            preprocessed,
        }
    }

    /// Stage 2 width: 1 accumulator column + 1 inverse column per lookup.
    pub fn stage_2_width(&self) -> usize {
        1 + self.lookups.len()
    }
}

/// Horner-style fingerprint of a sequence of coefficients.
#[inline]
pub fn fingerprint<I, Iter>(r: &Val, coeffs: Iter) -> Val
where
    I: Into<Val>,
    Iter: DoubleEndedIterator<Item = I>,
{
    coeffs
        .rev()
        .fold(Val::zero(), |acc, coeff| acc * r + coeff.into())
}

impl Lookup<SymbolicExpression> {
    /// Evaluate symbolic lookup args against a concrete trace row + optional
    /// preprocessed row.
    pub fn compute_expr(&self, row: &[Val], preprocessed: Option<&[Val]>) -> Lookup<Val> {
        let multiplicity = self.multiplicity.interpret::<Val>(row, preprocessed);
        let args = self
            .args
            .iter()
            .map(|arg| arg.interpret::<Val>(row, preprocessed))
            .collect();
        Lookup { multiplicity, args }
    }
}

impl Lookup<Val> {
    /// Build the stage-2 trace for every circuit + the per-circuit intermediate
    /// accumulator value (the running sum at the **end** of each circuit).
    pub fn stage_2_traces(
        lookups: &[Vec<Vec<Self>>],
        lookup_challenge: Val,
        fingerprint_challenge: &Val,
        mut accumulator: Val,
    ) -> (Vec<Matrix<Val>>, Vec<Val>) {
        // count messages per circuit + total
        let mut num_lookups_per_circuit = Vec::with_capacity(lookups.len());
        let mut total_num_lookups = 0usize;
        for circuit_lookups in lookups {
            let num_rows = circuit_lookups.len();
            let num_row_lookups = circuit_lookups[0].len();
            let n = num_rows * num_row_lookups;
            num_lookups_per_circuit.push(n);
            total_num_lookups += n;
        }

        // compute messages
        let mut messages = Vec::with_capacity(total_num_lookups);
        for circuit_lookups in lookups {
            for lookup in circuit_lookups.iter().flatten() {
                messages.push(lookup.compute_message(lookup_challenge, fingerprint_challenge));
            }
        }

        // batch invert
        let messages_inverses = batch_inverse(&messages);

        let mut intermediate_accumulators = Vec::with_capacity(lookups.len());
        let mut traces = Vec::with_capacity(lookups.len());
        let mut offset = 0;
        for (circuit_lookups, num_circuit_messages) in lookups.iter().zip(num_lookups_per_circuit) {
            let circuit_messages_inverses =
                &messages_inverses[offset..offset + num_circuit_messages];
            offset += num_circuit_messages;

            let num_row_lookups = circuit_lookups[0].len();
            let values = if num_row_lookups == 0 {
                vec![accumulator; circuit_lookups.len()]
            } else {
                circuit_lookups
                    .iter()
                    .zip(circuit_messages_inverses.chunks_exact(num_row_lookups))
                    .flat_map(|(row_lookups, row_inverses)| {
                        let mut row = Vec::with_capacity(1 + row_lookups.len());
                        row.push(accumulator);
                        for (lookup, &inv) in row_lookups.iter().zip(row_inverses) {
                            accumulator += lookup.multiplicity * inv;
                            row.push(inv);
                        }
                        row
                    })
                    .collect::<Vec<Val>>()
            };
            let width = 1 + num_row_lookups;
            debug_assert_eq!(values.len() % width, 0);
            traces.push(Matrix::new(values, width));
            intermediate_accumulators.push(accumulator);
        }
        (traces, intermediate_accumulators)
    }

    fn compute_message(&self, lookup_challenge: Val, fingerprint_challenge: &Val) -> Val {
        lookup_challenge + fingerprint(fingerprint_challenge, self.args.iter().copied())
    }
}

fn batch_inverse(elems: &[Val]) -> Vec<Val> {
    // Standard Montgomery batch-inversion trick. Avoids a per-element
    // `inverse()` call.
    let n = elems.len();
    if n == 0 {
        return vec![];
    }
    let mut prefix = Vec::with_capacity(n);
    let mut acc = Val::one();
    for x in elems {
        prefix.push(acc);
        acc *= *x;
    }
    let mut inv_acc = acc.inverse().expect("batch inversion: zero element");
    let mut out = vec![Val::zero(); n];
    for i in (0..n).rev() {
        out[i] = prefix[i] * inv_acc;
        inv_acc *= elems[i];
    }
    out
}

impl<A: BaseAir> BaseAir for LookupAir<A> {
    fn width(&self) -> usize {
        self.inner_air.width()
    }
    fn preprocessed_trace(&self) -> Option<Matrix<Val>> {
        self.preprocessed.clone()
    }
}

impl<A, AB> Air<AB> for LookupAir<A>
where
    A: Air<AB>,
    AB: AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // 1. inner AIR constraints
        self.inner_air.eval(builder);

        // 2. lookup constraints (stage 2)
        let (lookup_challenge, fingerprint_challenge, acc, next_acc) = {
            let pubs = builder.stage_2_public_values();
            debug_assert_eq!(pubs.len(), LOOKUP_PUBLIC_SIZE);
            (pubs[0], pubs[1], pubs[2], pubs[3])
        };
        let stage_2_local = builder.stage_2_local().to_vec();
        let acc_col: AB::Var = stage_2_local[0];
        let messages_inverses: Vec<AB::Var> = stage_2_local[1..].to_vec();
        debug_assert_eq!(messages_inverses.len(), self.lookups.len());

        // Convert row data into Expr form so symbolic interpretation can run
        // over the builder's expression type.
        let main_local_expr: Vec<AB::Expr> =
            builder.main_local().iter().map(|v| (*v).into()).collect();
        let preprocessed_local_expr: Option<Vec<AB::Expr>> = if self.preprocessed.is_some() {
            Some(
                builder
                    .preprocessed_local()
                    .iter()
                    .map(|v| (*v).into())
                    .collect(),
            )
        } else {
            None
        };

        let mut acc_expr: AB::Expr = acc_col.into();
        for (lookup, &inv_var) in self.lookups.iter().zip(messages_inverses.iter()) {
            let multiplicity: AB::Expr = lookup
                .multiplicity
                .interpret::<AB::Expr>(&main_local_expr, preprocessed_local_expr.as_deref());
            let args_iter = lookup.args.iter().map(|arg| {
                arg.interpret::<AB::Expr>(&main_local_expr, preprocessed_local_expr.as_deref())
            });
            let fp_r: AB::Expr = fingerprint_challenge.into();
            let fingerprint_expr = fingerprint_horner::<AB::Expr>(&fp_r, args_iter);
            let message_expr: AB::Expr = AB::Expr::from(lookup_challenge) + fingerprint_expr;
            let inv_expr: AB::Expr = inv_var.into();
            // m * m^{-1} == 1
            let one: AB::Expr = Val::one().into();
            builder.assert_zero(message_expr * inv_expr.clone() - one);
            acc_expr = acc_expr + multiplicity * inv_expr;
        }

        // initial accumulator value
        let acc_pub: AB::Expr = acc.into();
        builder.when_first_row().assert_eq(acc_col, acc_pub);

        // transition: acc_expr matches next-row accumulator column
        let next_acc_col: AB::Var = builder.stage_2_next()[0];
        builder
            .when_transition()
            .assert_eq(acc_expr.clone(), next_acc_col);

        // last-row final value matches the public next_acc
        let next_acc_pub: AB::Expr = next_acc.into();
        builder.when_last_row().assert_eq(acc_expr, next_acc_pub);
    }
}

/// Horner over a generic ring (used in-circuit; element type need not be `Val`).
fn fingerprint_horner<E>(r: &E, args: impl DoubleEndedIterator<Item = E>) -> E
where
    E: Clone + From<Val> + std::ops::Add<Output = E> + std::ops::Mul<Output = E>,
{
    args.rev()
        .fold(E::from(Val::zero()), |acc, coeff| acc * r.clone() + coeff)
}
