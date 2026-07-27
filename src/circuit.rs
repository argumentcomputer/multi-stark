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

use p3_field::Field;

use crate::expr::{CircuitSpec, ColRef, Expr, ExtExpr, Source};
use crate::lookup::Lookup;

/// Index of a node in the compiled vector.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NodeId(pub u32);

impl NodeId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A compiled expression node. Base-field only: extension constraints
/// have been coordinate-expanded away by the time nodes exist.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
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

/// Binomial extension parameters: `X^degree = w`.
#[derive(Copy, Clone, Debug)]
pub struct ExtensionParams<F> {
    pub degree: usize,
    pub w: F,
    /// Emit the 3-multiplication Karatsuba form for D=2 products
    /// (matching native extension-multiplication cost). Schoolbook
    /// otherwise; both are exposed so tests can cross-check them.
    pub karatsuba: bool,
}

/// A compiled circuit.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Circuit<F> {
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
pub fn compile<F: Field>(
    spec: &CircuitSpec<F>,
    params: &ExtensionParams<F>,
) -> Result<Circuit<F>, CompileError> {
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

    let max_constraint_degree = zeros
        .iter()
        .map(|z| interner.degrees[z.index()])
        .max()
        .unwrap_or(0);
    Ok(Circuit {
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
        if d == 2 && params.karatsuba {
            // (a0 + a1 X)(b0 + b1 X) mod (X² − W), 3 multiplications:
            //   c0 = a0 b0 + W · a1 b1
            //   c1 = (a0 + a1)(b0 + b1) − a0 b0 − a1 b1
            let p0 = self.mul(a[0], b[0]);
            let p1 = self.mul(a[1], b[1]);
            let sa = self.add(a[0], a[1]);
            let sb = self.add(b[0], b[1]);
            let s = self.mul(sa, sb);
            let w = self.constant(params.w);
            let wp1 = self.mul(w, p1);
            let c0 = self.add(p0, wp1);
            let t = self.sub(s, p0);
            let c1 = self.sub(t, p1);
            return vec![c0, c1];
        }
        // Schoolbook: c_k = Σ_{i+j=k} a_i b_j + W · Σ_{i+j=k+D} a_i b_j.
        let w = self.constant(params.w);
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
