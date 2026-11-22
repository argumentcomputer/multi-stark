//! Compilation: frontend trees → flat, interned, base-only node vector.
//!
//! Properties established here (see the design doc):
//! - children by index, topological order (children before parents);
//! - hash-consing with commutative normalization: index equality ⇔
//!   structural equality;
//! - constant folding, so the zero coordinates of base embeddings vanish;
//! - coordinate expansion of extension constraints (scalar detection,
//!   Karatsuba for D=2, schoolbook otherwise);
//! - lookup expressions interned first, so they occupy a prefix of the
//!   vector (partial evaluation for the lookup witness);
//! - per-node degree multiples.

use std::collections::HashMap;

use crate::traits::{ExtensionModulus, ExtensionOf, Field};

use crate::expr::{CircuitSpec, ColRef, Expr, ExtExpr, RowOffset, Source};
use crate::lookup::Lookup;

/// Index of a node in the compiled vector.
#[derive(
    serde::Serialize, serde::Deserialize, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug,
)]
pub struct NodeId(pub u32);

impl NodeId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A compiled expression node. Base-field only: extension constraints
/// have been coordinate-expanded away by the time nodes exist.
#[derive(serde::Serialize, serde::Deserialize, Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Node<F> {
    Const(F),
    Var(ColRef),
    Public(u32),
    IsFirstRow,
    IsLastRow,
    IsTransition,
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Neg(NodeId),
}

/// Extension-field parameters controlling coordinate expansion of [`ExtExpr`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExtensionParams<F> {
    pub degree: usize,
    pub modulus: ExtensionModulus<F>,
    /// Emit Karatsuba products for D=2 binomial (3 muls) and D=3 cubic
    /// trinomial (6 muls, 0 constant muls). Schoolbook otherwise; both are
    /// exposed so tests can cross-check them.
    pub karatsuba: bool,
}

impl<F> ExtensionParams<F> {
    /// Degree-1 base-field extension parameters.
    pub const fn base() -> Self {
        Self {
            degree: 1,
            modulus: ExtensionModulus::Base,
            karatsuba: false,
        }
    }

    /// Binomial extension `X^degree - w`.
    pub const fn binomial(degree: usize, w: F, karatsuba: bool) -> Self {
        Self {
            degree,
            modulus: ExtensionModulus::Binomial { w },
            karatsuba,
        }
    }

    /// Cubic trinomial extension `X^3 - X - 1`.
    pub const fn cubic_trinomial(karatsuba: bool) -> Self {
        Self {
            degree: 3,
            modulus: ExtensionModulus::CubicTrinomial,
            karatsuba,
        }
    }

    /// Read extension parameters from an [`ExtensionOf`] implementation.
    pub fn for_extension<EF: ExtensionOf<F>>() -> Self
    where
        F: Field,
    {
        let degree = EF::D;
        let modulus = EF::MODULUS;
        let karatsuba = matches!(
            (degree, modulus),
            (2, ExtensionModulus::Binomial { .. }) | (3, ExtensionModulus::CubicTrinomial)
        );
        Self {
            degree,
            modulus,
            karatsuba,
        }
    }
}

/// A compiled constraint graph: the hash-consed DAG of expression nodes
/// shared by all constraints and lookups of one circuit.
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct ConstraintGraph<F> {
    /// The unique expression vector; all constraints and lookups share it.
    pub nodes: Vec<Node<F>>,
    /// Degree multiple of each node.
    pub degrees: Vec<u32>,
    /// Constraint roots, in canonical order: base constraints, then each
    /// extension constraint's D coordinate roots.
    pub zeros: Vec<NodeId>,
    /// Compiled lookups; all their nodes live in the prefix.
    pub lookups: Vec<Lookup<NodeId>>,
    /// `nodes[..lookup_prefix_len]` covers everything reachable from the
    /// lookup expressions.
    pub lookup_prefix_len: usize,
    pub max_constraint_degree: u32,
}

impl<F> ConstraintGraph<F> {
    /// Returns `true` if any lookup expression (multiplicity or argument)
    /// reads at least one preprocessed column.
    pub fn lookup_prefix_uses_preprocessed(&self) -> bool {
        self.nodes[..self.lookup_prefix_len]
            .iter()
            .any(|node| matches!(node, Node::Var(col) if col.source == Source::Preprocessed))
    }

    /// Returns `true` if any node in the graph reads a column from `source` at `offset`.
    pub fn uses_row(&self, source: Source, offset: RowOffset) -> bool {
        self.nodes.iter().any(
            |node| matches!(node, Node::Var(col) if col.source == source && col.offset == offset),
        )
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CompileError {
    CoordsLength {
        constraint: usize,
        expected: usize,
        got: usize,
    },
    ColumnOutOfRange {
        source: Source,
        index: u32,
        width: usize,
    },
    PublicOutOfRange {
        index: u32,
        count: usize,
    },
    /// Stage-2 columns are extension coordinates; base constraints and
    /// lookup expressions must not touch them.
    Stage2InBaseContext,
    /// A fully base-field extension constraint: state it as a base
    /// constraint instead.
    PurelyBaseExtConstraint {
        constraint: usize,
    },
    /// A constraint (or extension-constraint coordinate) that compiles to
    /// a nonzero constant: it can never vanish, so the system would be
    /// unprovable. `coordinate` is `None` for a base constraint, `Some(k)`
    /// for coordinate `k` of extension constraint `constraint`.
    UnsatisfiableConstant {
        constraint: usize,
        coordinate: Option<usize>,
    },
}

/// Compiles a circuit spec. Interning order: lookups first (they form the
/// prefix), then base constraints, then expanded extension constraints.
///
/// Constraint roots are canonicalized: a root that compiles to a constant
/// is dropped if zero (vacuous) or rejected if nonzero (unsatisfiable),
/// and the survivors are sorted by node id and deduplicated. The compiled
/// constraint set is therefore independent of the order and multiplicity
/// in which constraints were authored.
pub(crate) fn compile<F: Field>(
    spec: &CircuitSpec<F>,
    params: &ExtensionParams<F>,
) -> Result<ConstraintGraph<F>, CompileError> {
    let mut interner = Interner::new();

    let mut lookups = Vec::with_capacity(spec.lookups.len());
    for lookup in &spec.lookups {
        let multiplicity = interner.compile_expr(&lookup.multiplicity, spec, false)?;
        let args = lookup
            .args
            .iter()
            .map(|arg| interner.compile_expr(arg, spec, false))
            .collect::<Result<Vec<_>, _>>()?;
        lookups.push(Lookup { multiplicity, args });
    }
    let lookup_prefix_len = interner.nodes.len();

    let mut zeros = Vec::new();
    for (i, constraint) in spec.constraints.iter().enumerate() {
        let root = interner.compile_expr(constraint, spec, false)?;
        record_zero(&interner, &mut zeros, root, i, None)?;
    }
    for (i, constraint) in spec.ext_constraints.iter().enumerate() {
        if constraint.is_purely_base() {
            return Err(CompileError::PurelyBaseExtConstraint { constraint: i });
        }
        let coords = interner.expand_ext(constraint, spec, params, i)?;
        for (coord, root) in coords.into_iter().enumerate() {
            record_zero(&interner, &mut zeros, root, i, Some(coord))?;
        }
    }
    // Canonicalize: sort by node id and drop duplicate roots. Index
    // equality is structural equality, so this deduplicates constraints
    // that became equal under interning (e.g. `a + b` and `b + a`).
    zeros.sort_unstable();
    zeros.dedup();

    // Bottom-up interning guarantees children precede parents. The dense
    // evaluator's unchecked sweep (`ConstraintGraph::sweep_range`) relies on
    // this, so pin it here.
    debug_assert!(
        interner
            .nodes
            .iter()
            .enumerate()
            .all(|(i, node)| match *node {
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) =>
                    a.index() < i && b.index() < i,
                Node::Neg(a) => a.index() < i,
                _ => true,
            }),
        "compiled nodes must be topologically ordered"
    );

    let max_constraint_degree = zeros
        .iter()
        .map(|z| interner.degrees[z.index()])
        .max()
        .unwrap_or(0);
    Ok(ConstraintGraph {
        nodes: interner.nodes,
        degrees: interner.degrees,
        zeros,
        lookups,
        lookup_prefix_len,
        max_constraint_degree,
    })
}

/// Records a constraint root, handling the constant cases: a nonzero
/// constant is rejected (unsatisfiable), a zero constant is dropped
/// (vacuous), anything else is appended.
fn record_zero<F: Field>(
    interner: &Interner<F>,
    zeros: &mut Vec<NodeId>,
    root: NodeId,
    constraint: usize,
    coordinate: Option<usize>,
) -> Result<(), CompileError> {
    match interner.as_const(root) {
        Some(c) if c == F::ZERO => {}
        Some(_) => {
            return Err(CompileError::UnsatisfiableConstant {
                constraint,
                coordinate,
            });
        }
        None => zeros.push(root),
    }
    Ok(())
}

/// Bottom-up interner. Children are canonical ids before a parent is
/// interned, so the map lookup is O(1) and sharing is complete.
struct Interner<F> {
    nodes: Vec<Node<F>>,
    degrees: Vec<u32>,
    map: HashMap<Node<F>, NodeId>,
}

impl<F: Field> Interner<F> {
    fn new() -> Self {
        Self {
            nodes: vec![],
            degrees: vec![],
            map: HashMap::new(),
        }
    }

    fn intern(&mut self, node: Node<F>) -> NodeId {
        if let Some(&id) = self.map.get(&node) {
            return id;
        }
        let id = NodeId(u32::try_from(self.nodes.len()).expect("node count exceeds u32"));
        let degree = self.degree_of(&node);
        self.nodes.push(node);
        self.degrees.push(degree);
        self.map.insert(node, id);
        id
    }

    fn degree_of(&self, node: &Node<F>) -> u32 {
        match *node {
            Node::Const(_) | Node::Public(_) | Node::IsTransition => 0,
            Node::Var(_) | Node::IsFirstRow | Node::IsLastRow => 1,
            Node::Add(a, b) | Node::Sub(a, b) => {
                self.degrees[a.index()].max(self.degrees[b.index()])
            }
            Node::Mul(a, b) => self.degrees[a.index()] + self.degrees[b.index()],
            Node::Neg(a) => self.degrees[a.index()],
        }
    }

    fn as_const(&self, id: NodeId) -> Option<F> {
        match self.nodes[id.index()] {
            Node::Const(c) => Some(c),
            _ => None,
        }
    }

    fn constant(&mut self, value: F) -> NodeId {
        self.intern(Node::Const(value))
    }

    fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        match (self.as_const(a), self.as_const(b)) {
            (Some(x), Some(y)) => return self.constant(x + y),
            (Some(x), None) if x == F::ZERO => return b,
            (None, Some(y)) if y == F::ZERO => return a,
            _ => {}
        }
        // Commutative normalization: sorted operand ids.
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        self.intern(Node::Add(a, b))
    }

    fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if a == b {
            return self.constant(F::ZERO);
        }
        match (self.as_const(a), self.as_const(b)) {
            (Some(x), Some(y)) => return self.constant(x - y),
            (None, Some(y)) if y == F::ZERO => return a,
            (Some(x), None) if x == F::ZERO => return self.neg(b),
            _ => {}
        }
        self.intern(Node::Sub(a, b))
    }

    fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        match (self.as_const(a), self.as_const(b)) {
            (Some(x), Some(y)) => return self.constant(x * y),
            (Some(x), None) => {
                if x == F::ZERO {
                    return a;
                }
                if x == F::ONE {
                    return b;
                }
            }
            (None, Some(y)) => {
                if y == F::ZERO {
                    return b;
                }
                if y == F::ONE {
                    return a;
                }
            }
            _ => {}
        }
        // Commutative normalization: sorted operand ids.
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        self.intern(Node::Mul(a, b))
    }

    fn neg(&mut self, a: NodeId) -> NodeId {
        if let Some(x) = self.as_const(a) {
            return self.constant(-x);
        }
        if let Node::Neg(inner) = self.nodes[a.index()] {
            return inner;
        }
        self.intern(Node::Neg(a))
    }

    fn compile_expr(
        &mut self,
        expr: &Expr<F>,
        spec: &CircuitSpec<F>,
        allow_stage2: bool,
    ) -> Result<NodeId, CompileError> {
        Ok(match expr {
            Expr::Const(c) => self.constant(*c),
            Expr::Var(col) => {
                let width = match col.source {
                    Source::Preprocessed => spec.preprocessed_width,
                    Source::Main => spec.main_width,
                    Source::Stage2 => {
                        if !allow_stage2 {
                            return Err(CompileError::Stage2InBaseContext);
                        }
                        spec.stage2_width
                    }
                };
                if col.index as usize >= width {
                    return Err(CompileError::ColumnOutOfRange {
                        source: col.source,
                        index: col.index,
                        width,
                    });
                }
                self.intern(Node::Var(*col))
            }
            Expr::Public(index) => {
                if *index as usize >= spec.num_publics {
                    return Err(CompileError::PublicOutOfRange {
                        index: *index,
                        count: spec.num_publics,
                    });
                }
                self.intern(Node::Public(*index))
            }
            Expr::IsFirstRow => self.intern(Node::IsFirstRow),
            Expr::IsLastRow => self.intern(Node::IsLastRow),
            Expr::IsTransition => self.intern(Node::IsTransition),
            Expr::Add(a, b) => {
                let a = self.compile_expr(a, spec, allow_stage2)?;
                let b = self.compile_expr(b, spec, allow_stage2)?;
                self.add(a, b)
            }
            Expr::Sub(a, b) => {
                let a = self.compile_expr(a, spec, allow_stage2)?;
                let b = self.compile_expr(b, spec, allow_stage2)?;
                self.sub(a, b)
            }
            Expr::Mul(a, b) => {
                let a = self.compile_expr(a, spec, allow_stage2)?;
                let b = self.compile_expr(b, spec, allow_stage2)?;
                self.mul(a, b)
            }
            Expr::Neg(a) => {
                let a = self.compile_expr(a, spec, allow_stage2)?;
                self.neg(a)
            }
        })
    }

    /// Coordinate expansion: returns the D coordinates of an extension
    /// expression as interned base nodes.
    fn expand_ext(
        &mut self,
        expr: &ExtExpr<F>,
        spec: &CircuitSpec<F>,
        params: &ExtensionParams<F>,
        constraint: usize,
    ) -> Result<Vec<NodeId>, CompileError> {
        let d = params.degree;
        Ok(match expr {
            ExtExpr::Coords(coords) => {
                if coords.len() != d {
                    return Err(CompileError::CoordsLength {
                        constraint,
                        expected: d,
                        got: coords.len(),
                    });
                }
                coords
                    .iter()
                    .map(|c| self.compile_expr(c, spec, true))
                    .collect::<Result<Vec<_>, _>>()?
            }
            ExtExpr::Base(base) => {
                let zero = self.constant(F::ZERO);
                let mut coords = vec![zero; d];
                coords[0] = self.compile_expr(base, spec, true)?;
                coords
            }
            ExtExpr::Add(a, b) => {
                let a = self.expand_ext(a, spec, params, constraint)?;
                let b = self.expand_ext(b, spec, params, constraint)?;
                (0..d).map(|k| self.add(a[k], b[k])).collect()
            }
            ExtExpr::Sub(a, b) => {
                let a = self.expand_ext(a, spec, params, constraint)?;
                let b = self.expand_ext(b, spec, params, constraint)?;
                (0..d).map(|k| self.sub(a[k], b[k])).collect()
            }
            ExtExpr::Neg(a) => {
                let a = self.expand_ext(a, spec, params, constraint)?;
                a.into_iter().map(|c| self.neg(c)).collect()
            }
            ExtExpr::Mul(a, b) => {
                let a = self.expand_ext(a, spec, params, constraint)?;
                let b = self.expand_ext(b, spec, params, constraint)?;
                self.ext_mul(&a, &b, params)
            }
        })
    }

    /// True if all coordinates but the first are the interned zero — the
    /// shape of an embedded base value.
    fn is_scalar(&self, coords: &[NodeId]) -> bool {
        coords[1..]
            .iter()
            .all(|&c| self.as_const(c) == Some(F::ZERO))
    }

    fn ext_mul(&mut self, a: &[NodeId], b: &[NodeId], params: &ExtensionParams<F>) -> Vec<NodeId> {
        let d = params.degree;
        // Scalar structure is preserved by construction: an embedded base
        // value multiplies coordinate-wise, D base multiplications.
        if self.is_scalar(a) {
            return b.iter().map(|&bk| self.mul(a[0], bk)).collect();
        }
        if self.is_scalar(b) {
            return a.iter().map(|&ak| self.mul(b[0], ak)).collect();
        }
        match params.modulus {
            ExtensionModulus::Base => {
                debug_assert_eq!(d, 1);
                vec![self.mul(a[0], b[0])]
            }
            ExtensionModulus::CubicTrinomial => {
                debug_assert_eq!(d, 3);
                if params.karatsuba {
                    // (a0 + a1 X + a2 X^2)(b0 + b1 X + b2 X^2) mod (X^3 - X - 1),
                    // 6 multiplications, 0 constant multiplications:
                    //   v0 = a0 b0, v1 = a1 b1, v2 = a2 b2
                    //   m01 = (a0 + a1)(b0 + b1) - v0 - v1
                    //   m02 = (a0 + a2)(b0 + b2) - v0 - v2
                    //   m12 = (a1 + a2)(b1 + b2) - v1 - v2
                    //   c0 = v0 + m12
                    //   c1 = m01 + m12 + v2
                    //   c2 = m02 + v1 + v2
                    let v0 = self.mul(a[0], b[0]);
                    let v1 = self.mul(a[1], b[1]);
                    let v2 = self.mul(a[2], b[2]);

                    let s01_a = self.add(a[0], a[1]);
                    let s01_b = self.add(b[0], b[1]);
                    let p01 = self.mul(s01_a, s01_b);
                    let t01 = self.sub(p01, v0);
                    let m01 = self.sub(t01, v1);

                    let s02_a = self.add(a[0], a[2]);
                    let s02_b = self.add(b[0], b[2]);
                    let p02 = self.mul(s02_a, s02_b);
                    let t02 = self.sub(p02, v0);
                    let m02 = self.sub(t02, v2);

                    let s12_a = self.add(a[1], a[2]);
                    let s12_b = self.add(b[1], b[2]);
                    let p12 = self.mul(s12_a, s12_b);
                    let t12 = self.sub(p12, v1);
                    let m12 = self.sub(t12, v2);

                    let c0 = self.add(v0, m12);
                    let c1_0 = self.add(m01, m12);
                    let c1 = self.add(c1_0, v2);
                    let c2_0 = self.add(m02, v1);
                    let c2 = self.add(c2_0, v2);
                    vec![c0, c1, c2]
                } else {
                    // Schoolbook mod (X^3 - X - 1): X^3 = X + 1, X^4 = X^2 + X.
                    let t0 = self.mul(a[0], b[0]);
                    let a0b1 = self.mul(a[0], b[1]);
                    let a1b0 = self.mul(a[1], b[0]);
                    let t1 = self.add(a0b1, a1b0);
                    let a0b2 = self.mul(a[0], b[2]);
                    let a1b1 = self.mul(a[1], b[1]);
                    let a2b0 = self.mul(a[2], b[0]);
                    let t2_0 = self.add(a0b2, a1b1);
                    let t2 = self.add(t2_0, a2b0);
                    let a1b2 = self.mul(a[1], b[2]);
                    let a2b1 = self.mul(a[2], b[1]);
                    let t3 = self.add(a1b2, a2b1);
                    let t4 = self.mul(a[2], b[2]);

                    let c0 = self.add(t0, t3);
                    let c1_0 = self.add(t1, t3);
                    let c1 = self.add(c1_0, t4);
                    let c2 = self.add(t2, t4);
                    vec![c0, c1, c2]
                }
            }
            ExtensionModulus::Binomial { w: modulus_w } => {
                if d == 2 && params.karatsuba {
                    // (a0 + a1 X)(b0 + b1 X) mod (X² − W), 3 multiplications:
                    //   c0 = a0 b0 + W · a1 b1
                    //   c1 = (a0 + a1)(b0 + b1) − a0 b0 − a1 b1
                    let p0 = self.mul(a[0], b[0]);
                    let p1 = self.mul(a[1], b[1]);
                    let sa = self.add(a[0], a[1]);
                    let sb = self.add(b[0], b[1]);
                    let s = self.mul(sa, sb);
                    let w = self.constant(modulus_w);
                    let wp1 = self.mul(w, p1);
                    let c0 = self.add(p0, wp1);
                    let t = self.sub(s, p0);
                    let c1 = self.sub(t, p1);
                    return vec![c0, c1];
                }
                // Schoolbook: c_k = Σ_{i+j=k} a_i b_j + W · Σ_{i+j=k+D} a_i b_j.
                let w = self.constant(modulus_w);
                (0..d)
                    .map(|k| {
                        let mut low: Option<NodeId> = None;
                        let mut high: Option<NodeId> = None;
                        for (i, &ai) in a.iter().enumerate() {
                            for (j, &bj) in b.iter().enumerate() {
                                if i + j == k {
                                    let term = self.mul(ai, bj);
                                    low = Some(match low {
                                        Some(acc) => self.add(acc, term),
                                        None => term,
                                    });
                                } else if i + j == k + d {
                                    let term = self.mul(ai, bj);
                                    high = Some(match high {
                                        Some(acc) => self.add(acc, term),
                                        None => term,
                                    });
                                }
                            }
                        }
                        let low = low.expect("k < d always has low terms");
                        match high {
                            Some(high) => {
                                let high = self.mul(w, high);
                                self.add(low, high)
                            }
                            None => low,
                        }
                    })
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{VarValues, eval_ext_expr};
    use crate::traits::{Algebra, ExtensionOf};
    use p3_field::extension::CubicTrinomialExtensionField;
    use p3_goldilocks::Goldilocks;

    #[test]
    fn cubic_trinomial_graph_compilation_matches_reference_and_p3() {
        type Ext3 = CubicTrinomialExtensionField<Goldilocks>;

        let a = ExtExpr::Coords(vec![Expr::main(0), Expr::main(1), Expr::main(2)]);
        let b = ExtExpr::Coords(vec![Expr::main(3), Expr::main(4), Expr::main(5)]);
        let c = ExtExpr::Coords(vec![
            Expr::main_next(0),
            Expr::main_next(1),
            Expr::main_next(2),
        ]);
        let expr = a * b + c * ExtExpr::from(Expr::main(3));

        let spec = CircuitSpec {
            main_width: 6,
            preprocessed_width: 0,
            stage2_width: 0,
            num_publics: 0,
            constraints: vec![],
            ext_constraints: vec![expr.clone()],
            lookups: vec![],
        };
        let karatsuba = compile(&spec, &ExtensionParams::cubic_trinomial(true)).unwrap();
        let schoolbook = compile(&spec, &ExtensionParams::cubic_trinomial(false)).unwrap();

        let main_cur: Vec<Goldilocks> = (1..=6u64)
            .map(|i| Goldilocks::from_u64(i * 19 + 3))
            .collect();
        let main_next: Vec<Goldilocks> = (1..=6u64)
            .map(|i| Goldilocks::from_u64(i * 31 + 7))
            .collect();
        let empty: [Goldilocks; 0] = [];
        let view = VarValues {
            preprocessed: [&empty, &empty],
            main: [&main_cur, &main_next],
            stage2: [&empty, &empty],
            publics: &empty,
            is_first_row: Goldilocks::ZERO,
            is_last_row: Goldilocks::ZERO,
            is_transition: Goldilocks::ONE,
        };

        let ref_vals = eval_ext_expr(&expr, &view, &ExtensionParams::cubic_trinomial(false));
        let k_vals = karatsuba.evaluate_constraints(&view);
        let s_vals = schoolbook.evaluate_constraints(&view);

        // Compare against native Plonky3 CubicTrinomialExtensionField arithmetic.
        let p3_a = Ext3::from_basis_coefficients_fn(|i| main_cur[i]);
        let p3_b = Ext3::from_basis_coefficients_fn(|i| main_cur[3 + i]);
        let p3_c = Ext3::from_basis_coefficients_fn(|i| main_next[i]);
        let p3_res = p3_a * p3_b + p3_c * main_cur[3];
        assert_eq!(ref_vals.as_slice(), p3_res.as_basis_coefficients_slice());

        // `zeros` are canonicalized (sorted by node id), so compare as sorted sets.
        let mut sorted_ref = ref_vals;
        let mut sorted_k = k_vals;
        let mut sorted_s = s_vals;
        sorted_ref.sort();
        sorted_k.sort();
        sorted_s.sort();
        assert_eq!(sorted_k, sorted_ref);
        assert_eq!(sorted_s, sorted_ref);
    }
}
