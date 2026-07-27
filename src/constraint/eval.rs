//! Evaluation.
//!
//! Compiled circuits are evaluated by a single dense forward sweep: one
//! buffer slot per node, children always at smaller indices. The frontend
//! trees also get a direct recursive evaluator, used as the reference in
//! tests (with genuine extension-field arithmetic for `ExtExpr`).

use p3_field::{Algebra, Field};

use super::circuit::{Circuit, ExtensionParams, Node, NodeId};
use super::expr::{ColRef, Expr, ExtExpr, RowOffset, Source};
use super::lookup::Lookup;

/// Concrete leaf values for a sweep, in the working type `W`: the two-row
/// window of each trace matrix, the public inputs and the selector values.
///
/// `W` is the type the sweep computes in — `F` for the debug/witness paths,
/// `PackedVal` on the quotient domain (prover), `EF` at ζ (verifier). The
/// compiled node constants (base field `F`) embed into `W` via `Algebra<F>`.
#[derive(Clone, Debug)]
pub struct VarValues<'a, W> {
    /// `[current_row, next_row]` of the preprocessed trace.
    pub preprocessed: [&'a [W]; 2],
    /// `[current_row, next_row]` of the main trace.
    pub main: [&'a [W]; 2],
    /// `[current_row, next_row]` of the stage-2 trace (flattened base columns).
    pub stage2: [&'a [W]; 2],
    /// Public inputs, as base-field coordinates embedded into `W`.
    pub publics: &'a [W],
    pub is_first_row: W,
    pub is_last_row: W,
    pub is_transition: W,
}

impl<W: Copy> VarValues<'_, W> {
    fn var(&self, col: &ColRef) -> W {
        let rows = match col.source {
            Source::Preprocessed => &self.preprocessed,
            Source::Main => &self.main,
            Source::Stage2 => &self.stage2,
        };
        let row = match col.offset {
            RowOffset::Current => rows[0],
            RowOffset::Next => rows[1],
        };
        row[col.index as usize]
    }
}

impl<F: Field> Circuit<F> {
    /// Dense forward sweep over the whole node vector, in working type `W`;
    /// fills `buf` with one value per node.
    pub fn sweep<W: Algebra<F> + Copy>(&self, values: &VarValues<'_, W>, buf: &mut Vec<W>) {
        self.sweep_range(values, buf, self.nodes.len());
    }

    /// Sweeps only the lookup prefix (partial evaluation for the lookup
    /// witness).
    pub fn sweep_lookup_prefix<W: Algebra<F> + Copy>(
        &self,
        values: &VarValues<'_, W>,
        buf: &mut Vec<W>,
    ) {
        self.sweep_range(values, buf, self.lookup_prefix_len);
    }

    fn sweep_range<W: Algebra<F> + Copy>(
        &self,
        values: &VarValues<'_, W>,
        buf: &mut Vec<W>,
        len: usize,
    ) {
        buf.clear();
        buf.reserve(len);
        for node in &self.nodes[..len] {
            let value = match node {
                Node::Const(c) => (*c).into(),
                Node::Var(col) => values.var(col),
                Node::Public(i) => values.publics[*i as usize],
                Node::IsFirstRow => values.is_first_row,
                Node::IsLastRow => values.is_last_row,
                Node::IsTransition => values.is_transition,
                Node::Add(a, b) => buf[a.index()] + buf[b.index()],
                Node::Sub(a, b) => buf[a.index()] - buf[b.index()],
                Node::Mul(a, b) => buf[a.index()] * buf[b.index()],
                Node::Neg(a) => -buf[a.index()],
            };
            buf.push(value);
        }
    }

    /// The constraint values, read off a full sweep buffer.
    pub fn constraint_values<W: Copy>(&self, buf: &[W]) -> Vec<W> {
        self.zeros.iter().map(|z| buf[z.index()]).collect()
    }

    /// The concrete lookup values, read off a (prefix or full) sweep buffer.
    pub fn lookup_values<W: Copy>(&self, buf: &[W]) -> Vec<Lookup<W>> {
        self.lookups
            .iter()
            .map(|lookup| Lookup {
                multiplicity: buf[lookup.multiplicity.index()],
                args: lookup.args.iter().map(|a| buf[a.index()]).collect(),
            })
            .collect()
    }

    /// Convenience: sweep and return the constraint values.
    pub fn evaluate_constraints<W: Algebra<F> + Copy>(&self, values: &VarValues<'_, W>) -> Vec<W> {
        let mut buf = Vec::new();
        self.sweep(values, &mut buf);
        self.constraint_values(&buf)
    }
}

/// Reference evaluation of a frontend base expression (recursive).
pub fn eval_expr<F: Field>(expr: &Expr<F>, values: &VarValues<'_, F>) -> F {
    match expr {
        Expr::Const(c) => *c,
        Expr::Var(col) => values.var(col),
        Expr::Public(i) => values.publics[*i as usize],
        Expr::IsFirstRow => values.is_first_row,
        Expr::IsLastRow => values.is_last_row,
        Expr::IsTransition => values.is_transition,
        Expr::Add(a, b) => eval_expr(a, values) + eval_expr(b, values),
        Expr::Sub(a, b) => eval_expr(a, values) - eval_expr(b, values),
        Expr::Mul(a, b) => eval_expr(a, values) * eval_expr(b, values),
        Expr::Neg(a) => -eval_expr(a, values),
    }
}

/// Reference evaluation of a frontend extension expression, in genuine
/// extension-field arithmetic: coordinates as `Vec<F>` of length D,
/// products reduced mod `X^D − W` (schoolbook — deliberately independent
/// of the compiled Karatsuba path).
pub fn eval_ext_expr<F: Field>(
    expr: &ExtExpr<F>,
    values: &VarValues<'_, F>,
    params: &ExtensionParams<F>,
) -> Vec<F> {
    let d = params.degree;
    match expr {
        ExtExpr::Coords(coords) => {
            assert_eq!(coords.len(), d, "reference eval: bad Coords length");
            coords.iter().map(|c| eval_expr(c, values)).collect()
        }
        ExtExpr::Base(base) => {
            let mut out = vec![F::ZERO; d];
            out[0] = eval_expr(base, values);
            out
        }
        ExtExpr::Add(a, b) => {
            let a = eval_ext_expr(a, values, params);
            let b = eval_ext_expr(b, values, params);
            a.into_iter().zip(b).map(|(x, y)| x + y).collect()
        }
        ExtExpr::Sub(a, b) => {
            let a = eval_ext_expr(a, values, params);
            let b = eval_ext_expr(b, values, params);
            a.into_iter().zip(b).map(|(x, y)| x - y).collect()
        }
        ExtExpr::Neg(a) => {
            let a = eval_ext_expr(a, values, params);
            a.into_iter().map(|x| -x).collect()
        }
        ExtExpr::Mul(a, b) => {
            let a = eval_ext_expr(a, values, params);
            let b = eval_ext_expr(b, values, params);
            let mut out = vec![F::ZERO; d];
            for i in 0..d {
                for j in 0..d {
                    let term = a[i] * b[j];
                    if i + j < d {
                        out[i + j] += term;
                    } else {
                        out[i + j - d] += params.w * term;
                    }
                }
            }
            out
        }
    }
}

/// Checks the topological invariant: every node's children have strictly
/// smaller indices. Test helper, but useful as a deserialization check too.
pub fn check_topological_order<F: Field>(circuit: &Circuit<F>) -> bool {
    circuit.nodes.iter().enumerate().all(|(i, node)| {
        let ok = |child: NodeId| child.index() < i;
        match *node {
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => ok(a) && ok(b),
            Node::Neg(a) => ok(a),
            _ => true,
        }
    })
}
