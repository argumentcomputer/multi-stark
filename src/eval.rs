//! Evaluation.
//!
//! Compiled circuits are evaluated by a single dense forward sweep: one
//! buffer slot per node, children always at smaller indices. The frontend
//! trees also get a direct recursive evaluator, used as the reference in
//! tests (with genuine extension-field arithmetic for `ExtExpr`).

use crate::traits::{Algebra, Field};

use crate::expr::{ColRef, Expr, ExtExpr, RowOffset, Source};
use crate::graph::{ConstraintGraph, ExtensionParams, Node, NodeId};
use crate::lookup::Lookup;

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

impl<F: Field> ConstraintGraph<F> {
    /// Dense forward sweep over the whole node vector, in working type `W`;
    /// fills `buf` with one value per node.
    pub fn sweep<W: Algebra<F>>(&self, values: &VarValues<'_, W>, buf: &mut Vec<W>) {
        self.sweep_range(values, buf, self.nodes.len());
    }

    /// Sweeps only the lookup prefix (partial evaluation for the lookup
    /// witness).
    pub fn sweep_lookup_prefix<W: Algebra<F>>(&self, values: &VarValues<'_, W>, buf: &mut Vec<W>) {
        self.sweep_range(values, buf, self.lookup_prefix_len);
    }

    fn sweep_range<W: Algebra<F>>(&self, values: &VarValues<'_, W>, buf: &mut Vec<W>, len: usize) {
        buf.clear();
        buf.reserve(len);
        // Raw pointer into the reserved storage; captured by the `child`
        // reader (a `*mut` is `Copy`, so it borrows nothing) and used for the
        // writes, sidestepping the borrow checker in the hot loop.
        let ptr = buf.spare_capacity_mut().as_mut_ptr();
        // SAFETY throughout: the node vector is topologically ordered — every
        // child's index is strictly less than its parent's — so when node `i`
        // is processed, every index it reads (`< i`) has already been written
        // and initialized; the storage has room for `len` values (reserved,
        // no realloc); and `W: Copy` means the reads take copies with no drop
        // obligations. This is the constraint-evaluation hot loop (quotient
        // domain, per packet), so eliding per-node bounds/capacity checks and
        // the shared-subexpression recomputation matters.
        let child = |id: NodeId| unsafe { (*ptr.add(id.index())).assume_init_read() };
        for i in 0..len {
            let node = unsafe { self.nodes.get_unchecked(i) };
            let value = match node {
                Node::Const(c) => (*c).into(),
                Node::Var(col) => values.var(col),
                Node::Public(idx) => values.publics[*idx as usize],
                Node::IsFirstRow => values.is_first_row,
                Node::IsLastRow => values.is_last_row,
                Node::IsTransition => values.is_transition,
                Node::Add(a, b) => child(*a) + child(*b),
                Node::Sub(a, b) => child(*a) - child(*b),
                Node::Mul(a, b) => child(*a) * child(*b),
                Node::Neg(a) => -child(*a),
            };
            unsafe { (*ptr.add(i)).write(value) };
        }
        // SAFETY: the loop initialized exactly `len` values.
        unsafe { buf.set_len(len) };
    }

    /// The constraint values, read off a full sweep buffer.
    pub fn constraint_values<W: Copy>(&self, buf: &[W]) -> Vec<W> {
        self.zeros.iter().map(|z| buf[z.index()]).collect()
    }

    /// Folds the user-constraint roots (`self.zeros`) with interleaved degree-2
    /// alpha-power pairs `[c0, c1]` directly off the sweep buffer, avoiding any
    /// intermediate `constraint_values` allocation.
    #[inline]
    pub fn fold_constraints_d2<W: Algebra<F>>(&self, buf: &[W], alpha_pairs: &[[F; 2]]) -> (W, W) {
        debug_assert!(alpha_pairs.len() >= self.zeros.len());
        let mut a0 = W::ZERO;
        let mut a1 = W::ZERO;
        let mut a2 = W::ZERO;
        let mut a3 = W::ZERO;
        let mut b0 = W::ZERO;
        let mut b1 = W::ZERO;
        let mut b2 = W::ZERO;
        let mut b3 = W::ZERO;

        let (z_chunks, z_rem) = self.zeros.as_chunks::<4>();
        let (p_chunks, p_rem) = alpha_pairs[..self.zeros.len()].as_chunks::<4>();
        for (zc, pc) in z_chunks.iter().zip(p_chunks) {
            let v0 = buf[zc[0].index()];
            let v1 = buf[zc[1].index()];
            let v2 = buf[zc[2].index()];
            let v3 = buf[zc[3].index()];
            a0 += v0 * pc[0][0];
            b0 += v0 * pc[0][1];
            a1 += v1 * pc[1][0];
            b1 += v1 * pc[1][1];
            a2 += v2 * pc[2][0];
            b2 += v2 * pc[2][1];
            a3 += v3 * pc[3][0];
            b3 += v3 * pc[3][1];
        }
        let mut acc0 = (a0 + a1) + (a2 + a3);
        let mut acc1 = (b0 + b1) + (b2 + b3);
        for (z, p) in z_rem.iter().zip(p_rem) {
            let v = buf[z.index()];
            acc0 += v * p[0];
            acc1 += v * p[1];
        }
        (acc0, acc1)
    }

    /// Folds the user-constraint roots (`self.zeros`) with decomposed alpha
    /// powers into `acc` (one accumulator per basis coordinate), avoiding any
    /// intermediate `constraint_values` allocation.
    #[inline]
    pub fn fold_constraints<W: Algebra<F>>(
        &self,
        buf: &[W],
        decomposed_alpha_powers: &[Vec<F>],
        acc: &mut [W],
    ) {
        debug_assert_eq!(acc.len(), decomposed_alpha_powers.len());
        for (idx, z) in self.zeros.iter().enumerate() {
            let v = buf[z.index()];
            for (slot, coeffs) in acc.iter_mut().zip(decomposed_alpha_powers) {
                *slot += v * coeffs[idx];
            }
        }
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
    pub fn evaluate_constraints<W: Algebra<F>>(&self, values: &VarValues<'_, W>) -> Vec<W> {
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
pub fn check_topological_order<F: Field>(circuit: &ConstraintGraph<F>) -> bool {
    circuit.nodes.iter().enumerate().all(|(i, node)| {
        let ok = |child: NodeId| child.index() < i;
        match *node {
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => ok(a) && ok(b),
            Node::Neg(a) => ok(a),
            _ => true,
        }
    })
}
