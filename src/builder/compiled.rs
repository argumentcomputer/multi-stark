//! Compiled constraint evaluation for the prover's quotient phase.
//!
//! [`SymbolicExpression`] trees are what circuits hand the system, but they
//! are a poor runtime representation: evaluation recurses over `Box`ed heap
//! nodes, and expression trees built by constraint generators typically
//! contain massive structural duplication (the same selector sums and
//! compound sub-expressions cloned into many constraints and lookup
//! arguments), all of which the tree walker re-evaluates from scratch on
//! every row of the quotient domain.
//!
//! This module lowers the symbolic constraints of a circuit once per proof
//! into a flat program that the quotient loop then executes per packed row
//! chunk:
//!
//! 1. **Intern** every sub-expression into a hash-consed DAG: a node is keyed
//!    by `(operator, child id, child id)`, so structurally identical subtrees
//!    collapse to a single node no matter how many times they were cloned
//!    during construction. Commutative operands are sorted by id first, which
//!    also merges `a + b` with `b + a`.
//! 2. **Reference-count** the DAG from the constraint roots.
//! 3. **Linearize** into an accumulator machine: a chain of operations whose
//!    left operand is implicitly the accumulator (the value computed by the
//!    preceding instruction) and whose right operand is a variable, constant,
//!    or scratch slot. Nodes used once are fused into their consumer's chain
//!    and never touch memory; nodes used more than once are computed once and
//!    stored in a scratch slot.
//!
//! Values live in one of two lanes: the **base** lane (`F`, packed as
//! [`Field::Packing`]) for stage-1/preprocessed columns and base constants,
//! and the **extension** lane (`EF`, packed as
//! [`ExtensionField::ExtensionPacking`]) for stage-2 columns, stage-2 public
//! values, and genuinely-extension constants. Operations are lane-typed at
//! compile time; a base value is promoted to the extension lane only where an
//! operation actually mixes lanes, so the (much cheaper) base arithmetic is
//! preserved wherever the original constraints were base-only.
//!
//! Each constraint root is folded into a running total with its
//! `alpha_powers` coefficient (same reversed indexing as the verifier's
//! Horner fold), so a single program execution yields the full constraint
//! accumulator of one packed row chunk.

use std::collections::HashMap;

use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};

use super::symbolic::{Entry, SymbolicExpression, SymbolicVariable};

/// Packed base-field values of `F`.
type PackedVal<F> = <F as Field>::Packing;
/// Packed extension-field values of `EF` over `F`.
type PackedExt<F, EF> = <EF as ExtensionField<F>>::ExtensionPacking;

/// Operand reference: 4-bit kind tag in the high bits, 28-bit index below.
const KIND_SHIFT: u32 = 28;
const IDX_MASK: u32 = (1 << KIND_SHIFT) - 1;

/// Stage-1 (main) trace column; index = `offset * width + column` with
/// offset 0 = local row, 1 = next row.
const K_S1: u32 = 0;
/// Preprocessed trace column, flattened like [`K_S1`].
const K_PREP: u32 = 1;
/// Stage-2 trace column in extension-element units, flattened like [`K_S1`].
const K_S2: u32 = 2;
/// Base-field constant table index.
const K_CONST_B: u32 = 3;
/// Extension-field constant table index.
const K_CONST_E: u32 = 4;
/// Base-lane scratch slot.
const K_SLOT_B: u32 = 5;
/// Extension-lane scratch slot.
const K_SLOT_E: u32 = 6;
/// Row selectors: 0 = is_first_row, 1 = is_last_row, 2 = is_transition.
const K_SPECIAL: u32 = 7;
/// Stage-1 public value index.
const K_PUB_B: u32 = 8;
/// Stage-2 public value index.
const K_PUB_E: u32 = 9;

#[inline]
#[allow(clippy::cast_possible_truncation)] // idx is asserted to fit in 28 bits
const fn operand(kind: u32, idx: usize) -> u32 {
    assert!(idx <= IDX_MASK as usize, "operand index exceeds 28 bits");
    (kind << KIND_SHIFT) | idx as u32
}

/// One instruction of the linearized constraint program. `B`/`E` suffixes
/// denote the lane of the accumulator after the instruction; `EB` denotes an
/// extension-lane accumulator combined with a base-lane operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum OpCode {
    /// `acc_b = operand`
    LoadB,
    /// `acc_b += operand`
    AddB,
    /// `acc_b -= operand`
    SubB,
    /// `acc_b = operand - acc_b`
    RSubB,
    /// `acc_b *= operand`
    MulB,
    /// `acc_b = -acc_b`
    NegB,
    /// `scratch_b[arg] = acc_b`
    StoreB,
    /// `acc_e = embed(acc_b)`
    Promote,
    /// `acc_e = operand`
    LoadE,
    /// `acc_e += operand` (extension operand)
    AddE,
    /// `acc_e -= operand` (extension operand)
    SubE,
    /// `acc_e = operand - acc_e` (extension operand)
    RSubE,
    /// `acc_e *= operand` (extension operand)
    MulE,
    /// `acc_e += operand` (base operand)
    AddEB,
    /// `acc_e -= operand` (base operand)
    SubEB,
    /// `acc_e = embed(operand) - acc_e` (base operand)
    RSubEB,
    /// `acc_e *= operand` (base operand)
    MulEB,
    /// `acc_e = -acc_e`
    NegE,
    /// `scratch_e[arg] = acc_e`
    StoreE,
    /// `total += alpha_powers[arg] * acc_b`
    FoldB,
    /// `total += alpha_powers[arg] * acc_e`
    FoldE,
}

#[derive(Clone, Copy, Debug)]
struct Instr {
    op: OpCode,
    arg: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lane {
    Base,
    Ext,
}

/// Interned DAG node. Leaves are keyed directly by their encoded operand, so
/// two occurrences of the same column/constant dedupe automatically.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Node {
    Leaf(u32),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Neg(u32),
}

/// A circuit's constraints, compiled to a linear program.
///
/// Compilation depends only on the circuit (not on the proof instance), but
/// per-instance values — public values and `alpha` powers — enter through
/// [`CompiledConstraints::prepare`].
pub struct CompiledConstraints<F: Field, EF: ExtensionField<F>> {
    code: Vec<Instr>,
    base_consts: Vec<F>,
    ext_consts: Vec<EF>,
    /// Number of base-lane scratch slots the program uses.
    num_slots_b: usize,
    /// Number of extension-lane scratch slots the program uses.
    num_slots_e: usize,
    num_constraints: usize,
}

/// Per-proof, per-circuit values pre-packed for the evaluation loop:
/// constants and public values splatted across packing lanes, and the
/// (reversed) constraint-fold coefficients.
pub struct PreparedConstraints<F: Field, EF: ExtensionField<F>> {
    base_consts: Vec<PackedVal<F>>,
    ext_consts: Vec<PackedExt<F, EF>>,
    base_publics: Vec<PackedVal<F>>,
    ext_publics: Vec<PackedExt<F, EF>>,
    alpha_powers: Vec<PackedExt<F, EF>>,
}

/// Reusable per-worker scratch buffers for [`CompiledConstraints::eval`].
pub struct Scratch<F: Field, EF: ExtensionField<F>> {
    slots_b: Vec<PackedVal<F>>,
    slots_e: Vec<PackedExt<F, EF>>,
}

/// Packed values of one row-pair chunk of the quotient domain.
pub struct RowChunk<'a, F: Field, EF: ExtensionField<F>> {
    /// Stage-1 columns: local row then next row (`2 * stage_1_width`).
    pub stage_1: &'a [PackedVal<F>],
    /// Preprocessed columns, laid out like `stage_1` (empty if none).
    pub preprocessed: &'a [PackedVal<F>],
    /// Stage-2 columns in extension elements, laid out like `stage_1`.
    pub stage_2: &'a [PackedExt<F, EF>],
    pub is_first_row: PackedVal<F>,
    pub is_last_row: PackedVal<F>,
    pub is_transition: PackedVal<F>,
}

struct Compiler<F: Field, EF: ExtensionField<F>> {
    stage_1_width: usize,
    preprocessed_width: usize,
    stage_2_width: usize,
    nodes: Vec<(Node, Lane)>,
    dedup: HashMap<Node, u32>,
    base_consts: Vec<F>,
    ext_consts: Vec<EF>,
    /// Number of direct references to each node (parents + root uses).
    refs: Vec<u32>,
    /// Scratch slot holding each materialized shared node, if any.
    slot_of: Vec<Option<u32>>,
    code: Vec<Instr>,
    num_slots_b: usize,
    num_slots_e: usize,
    /// Freed temporary slots available for reuse, per lane. Only anonymous
    /// temporaries are recycled; slots holding shared nodes stay live for the
    /// whole program.
    free_b: Vec<u32>,
    free_e: Vec<u32>,
}

impl<F: Field, EF: ExtensionField<F>> Compiler<F, EF> {
    fn intern(&mut self, node: Node, lane: Lane) -> u32 {
        // Sort commutative operands so `a + b` and `b + a` share a node.
        let node = match node {
            Node::Add(a, b) if a > b => Node::Add(b, a),
            Node::Mul(a, b) if a > b => Node::Mul(b, a),
            node => node,
        };
        if let Some(&id) = self.dedup.get(&node) {
            return id;
        }
        let id = u32::try_from(self.nodes.len()).expect("constraint DAG exceeds u32 nodes");
        self.nodes.push((node, lane));
        self.dedup.insert(node, id);
        id
    }

    fn intern_leaf(&mut self, encoded: u32, lane: Lane) -> u32 {
        self.intern(Node::Leaf(encoded), lane)
    }

    fn intern_var(&mut self, var: &SymbolicVariable<EF>) -> u32 {
        let (kind, flat, width, lane) = match var.entry {
            Entry::Main { offset } => (K_S1, offset, self.stage_1_width, Lane::Base),
            Entry::Preprocessed { offset } => (K_PREP, offset, self.preprocessed_width, Lane::Base),
            Entry::Stage2 { offset } => (K_S2, offset, self.stage_2_width, Lane::Ext),
            Entry::Public => (K_PUB_B, 0, 0, Lane::Base),
            Entry::Stage2Public => (K_PUB_E, 0, 0, Lane::Ext),
            Entry::Challenge => unimplemented!("challenge variables are not used"),
        };
        let idx = match var.entry {
            Entry::Public | Entry::Stage2Public => var.index,
            _ => {
                assert!(flat < 2, "only a two-row window is supported");
                assert!(var.index < width, "variable index out of range");
                flat * width + var.index
            }
        };
        self.intern_leaf(operand(kind, idx), lane)
    }

    fn intern_const(&mut self, value: EF) -> u32 {
        let coefficients = value.as_basis_coefficients_slice();
        if coefficients[1..].iter().all(|c| c.is_zero()) {
            let base = coefficients[0];
            let idx = self
                .base_consts
                .iter()
                .position(|c| *c == base)
                .unwrap_or_else(|| {
                    self.base_consts.push(base);
                    self.base_consts.len() - 1
                });
            self.intern_leaf(operand(K_CONST_B, idx), Lane::Base)
        } else {
            let idx = self
                .ext_consts
                .iter()
                .position(|c| *c == value)
                .unwrap_or_else(|| {
                    self.ext_consts.push(value);
                    self.ext_consts.len() - 1
                });
            self.intern_leaf(operand(K_CONST_E, idx), Lane::Ext)
        }
    }

    /// Interns an expression tree bottom-up with an explicit stack (the trees
    /// can be deep enough to threaten the call stack).
    fn intern_expr(&mut self, expr: &SymbolicExpression<EF>) -> u32 {
        enum Task<'e, EF> {
            Visit(&'e SymbolicExpression<EF>),
            Combine(u8),
        }
        let mut tasks = vec![Task::Visit(expr)];
        let mut values: Vec<u32> = vec![];
        while let Some(task) = tasks.pop() {
            match task {
                Task::Visit(expr) => match expr {
                    SymbolicExpression::Variable(var) => {
                        let id = self.intern_var(var);
                        values.push(id);
                    }
                    SymbolicExpression::IsFirstRow => {
                        let id = self.intern_leaf(operand(K_SPECIAL, 0), Lane::Base);
                        values.push(id);
                    }
                    SymbolicExpression::IsLastRow => {
                        let id = self.intern_leaf(operand(K_SPECIAL, 1), Lane::Base);
                        values.push(id);
                    }
                    SymbolicExpression::IsTransition => {
                        let id = self.intern_leaf(operand(K_SPECIAL, 2), Lane::Base);
                        values.push(id);
                    }
                    SymbolicExpression::Constant(c) => {
                        let id = self.intern_const(*c);
                        values.push(id);
                    }
                    SymbolicExpression::Add { x, y, .. } => {
                        tasks.push(Task::Combine(0));
                        tasks.push(Task::Visit(y));
                        tasks.push(Task::Visit(x));
                    }
                    SymbolicExpression::Sub { x, y, .. } => {
                        tasks.push(Task::Combine(1));
                        tasks.push(Task::Visit(y));
                        tasks.push(Task::Visit(x));
                    }
                    SymbolicExpression::Mul { x, y, .. } => {
                        tasks.push(Task::Combine(2));
                        tasks.push(Task::Visit(y));
                        tasks.push(Task::Visit(x));
                    }
                    SymbolicExpression::Neg { x, .. } => {
                        tasks.push(Task::Combine(3));
                        tasks.push(Task::Visit(x));
                    }
                },
                Task::Combine(op) => {
                    if op == 3 {
                        let x = values.pop().unwrap();
                        let lane = self.nodes[x as usize].1;
                        let id = self.intern(Node::Neg(x), lane);
                        values.push(id);
                    } else {
                        let y = values.pop().unwrap();
                        let x = values.pop().unwrap();
                        let lane = if self.nodes[x as usize].1 == Lane::Ext
                            || self.nodes[y as usize].1 == Lane::Ext
                        {
                            Lane::Ext
                        } else {
                            Lane::Base
                        };
                        let node = match op {
                            0 => Node::Add(x, y),
                            1 => Node::Sub(x, y),
                            _ => Node::Mul(x, y),
                        };
                        let id = self.intern(node, lane);
                        values.push(id);
                    }
                }
            }
        }
        debug_assert_eq!(values.len(), 1);
        values.pop().unwrap()
    }

    fn lane(&self, id: u32) -> Lane {
        self.nodes[id as usize].1
    }

    /// The encoded operand for a node that can be referenced directly: a leaf,
    /// or a shared node already computed into a scratch slot.
    fn operand_of(&self, id: u32) -> Option<u32> {
        match self.nodes[id as usize].0 {
            Node::Leaf(encoded) => Some(encoded),
            _ => self.slot_of[id as usize].map(|slot| {
                let kind = match self.lane(id) {
                    Lane::Base => K_SLOT_B,
                    Lane::Ext => K_SLOT_E,
                };
                operand(kind, slot as usize)
            }),
        }
    }

    fn push(&mut self, op: OpCode, arg: u32) {
        self.code.push(Instr { op, arg });
    }

    fn alloc_slot(&mut self, lane: Lane) -> u32 {
        let (counter, free) = match lane {
            Lane::Base => (&mut self.num_slots_b, &mut self.free_b),
            Lane::Ext => (&mut self.num_slots_e, &mut self.free_e),
        };
        if let Some(slot) = free.pop() {
            return slot;
        }
        let slot = u32::try_from(*counter).unwrap();
        *counter += 1;
        slot
    }

    fn free_slot(&mut self, lane: Lane, slot: u32) {
        match lane {
            Lane::Base => self.free_b.push(slot),
            Lane::Ext => self.free_e.push(slot),
        }
    }

    /// Promotes the accumulator to the extension lane if the value just
    /// computed (of lane `from`) must participate in an extension operation.
    fn promote_if_needed(&mut self, from: Lane, to: Lane) {
        if from == Lane::Base && to == Lane::Ext {
            self.push(OpCode::Promote, 0);
        }
    }

    /// Emits an operation combining the accumulator (left) with an encoded
    /// operand (right). `node_lane` is the lane of the result; the
    /// accumulator must already be in that lane.
    fn push_bin(&mut self, node: &Node, node_lane: Lane, arg: u32, reversed: bool) {
        let arg_lane = if arg >> KIND_SHIFT == K_S2
            || arg >> KIND_SHIFT == K_CONST_E
            || arg >> KIND_SHIFT == K_SLOT_E
            || arg >> KIND_SHIFT == K_PUB_E
        {
            Lane::Ext
        } else {
            Lane::Base
        };
        let op = match (node, node_lane, arg_lane, reversed) {
            (Node::Add(..), Lane::Base, Lane::Base, _) => OpCode::AddB,
            (Node::Mul(..), Lane::Base, Lane::Base, _) => OpCode::MulB,
            (Node::Sub(..), Lane::Base, Lane::Base, false) => OpCode::SubB,
            (Node::Sub(..), Lane::Base, Lane::Base, true) => OpCode::RSubB,
            (Node::Add(..), Lane::Ext, Lane::Ext, _) => OpCode::AddE,
            (Node::Mul(..), Lane::Ext, Lane::Ext, _) => OpCode::MulE,
            (Node::Sub(..), Lane::Ext, Lane::Ext, false) => OpCode::SubE,
            (Node::Sub(..), Lane::Ext, Lane::Ext, true) => OpCode::RSubE,
            (Node::Add(..), Lane::Ext, Lane::Base, _) => OpCode::AddEB,
            (Node::Mul(..), Lane::Ext, Lane::Base, _) => OpCode::MulEB,
            (Node::Sub(..), Lane::Ext, Lane::Base, false) => OpCode::SubEB,
            (Node::Sub(..), Lane::Ext, Lane::Base, true) => OpCode::RSubEB,
            _ => unreachable!("push_bin on non-binary node or base node with ext operand"),
        };
        self.push(op, arg);
    }

    /// Emits code leaving the value of `id` in the accumulator of its lane.
    /// If the node is shared and not yet materialized, it is also stored into
    /// a fresh scratch slot for later uses (the accumulator still holds the
    /// value afterwards, so the current consumer's chain continues unbroken).
    fn emit_acc(&mut self, id: u32) {
        let (node, node_lane) = self.nodes[id as usize];
        match node {
            Node::Leaf(encoded) => {
                let op = match node_lane {
                    Lane::Base => OpCode::LoadB,
                    Lane::Ext => OpCode::LoadE,
                };
                self.push(op, encoded);
            }
            Node::Neg(x) => {
                self.emit_operand_free(x, node_lane);
                let op = match node_lane {
                    Lane::Base => OpCode::NegB,
                    Lane::Ext => OpCode::NegE,
                };
                self.push(op, 0);
            }
            Node::Add(x, y) | Node::Sub(x, y) | Node::Mul(x, y) => {
                let commutative = !matches!(node, Node::Sub(..));
                match (self.operand_of(x), self.operand_of(y)) {
                    (Some(ox), Some(oy)) => {
                        // Both are direct operands: load one, combine the
                        // other. In the extension lane, load an extension
                        // operand first, so the combine step gets the cheaper
                        // mixed-lane form when the other operand is base.
                        if node_lane == Lane::Ext && self.lane(x) == Lane::Base {
                            // x base ⇒ y is the extension operand.
                            self.push(OpCode::LoadE, oy);
                            self.push_bin(&node, node_lane, ox, !commutative);
                        } else {
                            let load = match node_lane {
                                Lane::Base => OpCode::LoadB,
                                Lane::Ext => OpCode::LoadE,
                            };
                            self.push(load, ox);
                            self.push_bin(&node, node_lane, oy, false);
                        }
                    }
                    (None, Some(oy)) => {
                        // Chain x through the accumulator, combine y.
                        self.emit_operand_free(x, node_lane);
                        self.push_bin(&node, node_lane, oy, false);
                    }
                    (Some(ox), None) => {
                        // Chain y through the accumulator, combine x (reversed
                        // for subtraction).
                        self.emit_operand_free(y, node_lane);
                        self.push_bin(&node, node_lane, ox, !commutative);
                    }
                    (None, None) => {
                        // Both children are compound: compute y first. If y is
                        // shared it lands in a persistent slot as a side
                        // effect; otherwise park it in a temporary slot,
                        // recycled once the combine consumes it.
                        self.emit_acc(y);
                        let (oy, freed) = match self.operand_of(y) {
                            Some(oy) => (oy, None),
                            None => {
                                let y_lane = self.lane(y);
                                let slot = self.alloc_slot(y_lane);
                                let (store, kind) = match y_lane {
                                    Lane::Base => (OpCode::StoreB, K_SLOT_B),
                                    Lane::Ext => (OpCode::StoreE, K_SLOT_E),
                                };
                                self.push(store, slot);
                                (operand(kind, slot as usize), Some((y_lane, slot)))
                            }
                        };
                        self.emit_operand_free(x, node_lane);
                        self.push_bin(&node, node_lane, oy, false);
                        if let Some((lane, slot)) = freed {
                            self.free_slot(lane, slot);
                        }
                    }
                }
            }
        }
        if self.refs[id as usize] > 1
            && self.slot_of[id as usize].is_none()
            && !matches!(node, Node::Leaf(_))
        {
            let slot = self.alloc_slot(node_lane);
            let store = match node_lane {
                Lane::Base => OpCode::StoreB,
                Lane::Ext => OpCode::StoreE,
            };
            self.push(store, slot);
            self.slot_of[id as usize] = Some(slot);
        }
    }

    /// Emits `id` into the accumulator and promotes it to `target_lane` if
    /// the consuming operation lives in the extension lane.
    fn emit_operand_free(&mut self, id: u32, target_lane: Lane) {
        match self.operand_of(id) {
            Some(encoded) => {
                let op = match self.lane(id) {
                    Lane::Base => OpCode::LoadB,
                    Lane::Ext => OpCode::LoadE,
                };
                self.push(op, encoded);
            }
            None => self.emit_acc(id),
        }
        self.promote_if_needed(self.lane(id), target_lane);
    }
}

impl<F: Field, EF: ExtensionField<F>> CompiledConstraints<F, EF> {
    /// Compiles a circuit's symbolic constraints. The widths describe the
    /// trace windows the variables index into (`stage_2_width` in extension
    /// elements) and are used to flatten `(offset, column)` variable
    /// references; out-of-range references panic here rather than at
    /// evaluation time.
    pub fn compile(
        constraints: &[SymbolicExpression<EF>],
        preprocessed_width: usize,
        stage_1_width: usize,
        stage_2_width: usize,
    ) -> Self {
        let mut compiler = Compiler::<F, EF> {
            stage_1_width,
            preprocessed_width,
            stage_2_width,
            nodes: vec![],
            dedup: HashMap::new(),
            base_consts: vec![],
            ext_consts: vec![],
            refs: vec![],
            slot_of: vec![],
            code: vec![],
            num_slots_b: 0,
            num_slots_e: 0,
            free_b: vec![],
            free_e: vec![],
        };

        let roots: Vec<u32> = constraints
            .iter()
            .map(|c| compiler.intern_expr(c))
            .collect();

        // Reference counts: one per DAG edge plus one per root use.
        compiler.refs = vec![0; compiler.nodes.len()];
        for (node, _) in &compiler.nodes {
            match *node {
                Node::Leaf(_) => {}
                Node::Neg(x) => compiler.refs[x as usize] += 1,
                Node::Add(x, y) | Node::Sub(x, y) | Node::Mul(x, y) => {
                    compiler.refs[x as usize] += 1;
                    compiler.refs[y as usize] += 1;
                }
            }
        }
        for &root in &roots {
            compiler.refs[root as usize] += 1;
        }
        compiler.slot_of = vec![None; compiler.nodes.len()];

        for (k, &root) in roots.iter().enumerate() {
            match compiler.operand_of(root) {
                // A root that is itself a direct operand (constant, lone
                // column, or already-materialized shared node) still needs a
                // load so the fold sees it in the accumulator.
                Some(encoded) => {
                    let op = match compiler.lane(root) {
                        Lane::Base => OpCode::LoadB,
                        Lane::Ext => OpCode::LoadE,
                    };
                    compiler.push(op, encoded);
                }
                None => compiler.emit_acc(root),
            }
            let fold = match compiler.lane(root) {
                Lane::Base => OpCode::FoldB,
                Lane::Ext => OpCode::FoldE,
            };
            compiler.push(fold, u32::try_from(k).unwrap());
        }

        Self {
            code: compiler.code,
            base_consts: compiler.base_consts,
            ext_consts: compiler.ext_consts,
            num_slots_b: compiler.num_slots_b,
            num_slots_e: compiler.num_slots_e,
            num_constraints: constraints.len(),
        }
    }

    /// Number of instructions in the compiled program (diagnostics).
    pub fn code_len(&self) -> usize {
        self.code.len()
    }

    /// Pre-packs the per-proof values: constants, public values and the
    /// reversed `alpha` powers (`alpha_powers[k]` multiplies constraint `k`,
    /// exactly like the folder's `constraint_index` walk).
    pub fn prepare(
        &self,
        public_values: &[F],
        stage_2_public_values: &[EF],
        alpha_powers: &[EF],
    ) -> PreparedConstraints<F, EF> {
        assert_eq!(alpha_powers.len(), self.num_constraints);
        PreparedConstraints {
            base_consts: self.base_consts.iter().map(|&c| c.into()).collect(),
            ext_consts: self
                .ext_consts
                .iter()
                .map(|&c| PackedExt::<F, EF>::from(c))
                .collect(),
            base_publics: public_values.iter().map(|&c| c.into()).collect(),
            ext_publics: stage_2_public_values
                .iter()
                .map(|&c| PackedExt::<F, EF>::from(c))
                .collect(),
            alpha_powers: alpha_powers
                .iter()
                .map(|&a| PackedExt::<F, EF>::from(a))
                .collect(),
        }
    }

    /// Allocates scratch buffers sized for this program. Reuse across rows
    /// (e.g. per rayon worker); `eval` overwrites what it needs.
    pub fn scratch(&self) -> Scratch<F, EF> {
        Scratch {
            slots_b: vec![PackedVal::<F>::ZERO; self.num_slots_b],
            slots_e: vec![PackedExt::<F, EF>::ZERO; self.num_slots_e],
        }
    }

    /// Evaluates all constraints on one packed row chunk, returning
    /// `Σ_k alpha_powers[k] · constraint_k` — the same accumulator the
    /// constraint folder produces for that chunk.
    pub fn eval(
        &self,
        prepared: &PreparedConstraints<F, EF>,
        chunk: &RowChunk<'_, F, EF>,
        scratch: &mut Scratch<F, EF>,
    ) -> PackedExt<F, EF> {
        let mut acc_b = PackedVal::<F>::ZERO;
        let mut acc_e = PackedExt::<F, EF>::ZERO;
        let mut total = PackedExt::<F, EF>::ZERO;

        #[inline(always)]
        fn fetch_b<F: Field, EF: ExtensionField<F>>(
            arg: u32,
            prepared: &PreparedConstraints<F, EF>,
            chunk: &RowChunk<'_, F, EF>,
            scratch: &Scratch<F, EF>,
        ) -> PackedVal<F> {
            let idx = (arg & IDX_MASK) as usize;
            match arg >> KIND_SHIFT {
                K_S1 => chunk.stage_1[idx],
                K_PREP => chunk.preprocessed[idx],
                K_CONST_B => prepared.base_consts[idx],
                K_SLOT_B => scratch.slots_b[idx],
                K_PUB_B => prepared.base_publics[idx],
                K_SPECIAL => match idx {
                    0 => chunk.is_first_row,
                    1 => chunk.is_last_row,
                    _ => chunk.is_transition,
                },
                _ => unreachable!("extension operand in base fetch"),
            }
        }

        #[inline(always)]
        fn fetch_e<F: Field, EF: ExtensionField<F>>(
            arg: u32,
            prepared: &PreparedConstraints<F, EF>,
            chunk: &RowChunk<'_, F, EF>,
            scratch: &Scratch<F, EF>,
        ) -> PackedExt<F, EF> {
            let idx = (arg & IDX_MASK) as usize;
            match arg >> KIND_SHIFT {
                K_S2 => chunk.stage_2[idx],
                K_CONST_E => prepared.ext_consts[idx],
                K_SLOT_E => scratch.slots_e[idx],
                K_PUB_E => prepared.ext_publics[idx],
                _ => unreachable!("base operand in extension fetch"),
            }
        }

        for instr in &self.code {
            let arg = instr.arg;
            match instr.op {
                OpCode::LoadB => acc_b = fetch_b(arg, prepared, chunk, scratch),
                OpCode::AddB => acc_b += fetch_b(arg, prepared, chunk, scratch),
                OpCode::SubB => acc_b -= fetch_b(arg, prepared, chunk, scratch),
                OpCode::RSubB => acc_b = fetch_b(arg, prepared, chunk, scratch) - acc_b,
                OpCode::MulB => acc_b *= fetch_b(arg, prepared, chunk, scratch),
                OpCode::NegB => acc_b = -acc_b,
                OpCode::StoreB => scratch.slots_b[arg as usize] = acc_b,
                OpCode::Promote => acc_e = PackedExt::<F, EF>::from(acc_b),
                OpCode::LoadE => acc_e = fetch_e(arg, prepared, chunk, scratch),
                OpCode::AddE => acc_e += fetch_e(arg, prepared, chunk, scratch),
                OpCode::SubE => acc_e -= fetch_e(arg, prepared, chunk, scratch),
                OpCode::RSubE => acc_e = fetch_e(arg, prepared, chunk, scratch) - acc_e,
                OpCode::MulE => acc_e *= fetch_e(arg, prepared, chunk, scratch),
                OpCode::AddEB => acc_e += fetch_b(arg, prepared, chunk, scratch),
                OpCode::SubEB => acc_e -= fetch_b(arg, prepared, chunk, scratch),
                OpCode::RSubEB => {
                    acc_e = PackedExt::<F, EF>::from(fetch_b(arg, prepared, chunk, scratch)) - acc_e
                }
                OpCode::MulEB => acc_e *= fetch_b(arg, prepared, chunk, scratch),
                OpCode::NegE => acc_e = -acc_e,
                OpCode::StoreE => scratch.slots_e[arg as usize] = acc_e,
                OpCode::FoldB => total += prepared.alpha_powers[arg as usize] * acc_b,
                OpCode::FoldE => total += prepared.alpha_powers[arg as usize] * acc_e,
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{BasedVectorSpace, PackedValue, extension::BinomialExtensionField};
    use p3_goldilocks::Goldilocks;

    use super::*;
    use crate::builder::symbolic::{SymbolicExpression, SymbolicVariable, var};

    type F = Goldilocks;
    type EF = BinomialExtensionField<Goldilocks, 2>;
    type Expr = SymbolicExpression<EF>;

    const W1: usize = 3;
    const WP: usize = 2;
    const W2: usize = 2;
    const NUM_PUB_B: usize = 1;
    const NUM_PUB_E: usize = 4;

    fn v(entry: Entry, index: usize) -> Expr {
        SymbolicVariable::new(entry, index).into()
    }

    /// Reference evaluation: a direct recursive walk with scalar `EF`
    /// arithmetic, independent of the compiler's lane analysis.
    fn ref_eval(
        expr: &Expr,
        s1: &[Vec<F>; 2],
        prep: &[Vec<F>; 2],
        s2: &[Vec<EF>; 2],
        pubs_b: &[F],
        pubs_e: &[EF],
        specials: &[F; 3],
    ) -> EF {
        let rec = |e| ref_eval(e, s1, prep, s2, pubs_b, pubs_e, specials);
        match expr {
            Expr::Variable(var) => match var.entry {
                Entry::Main { offset } => s1[offset][var.index].into(),
                Entry::Preprocessed { offset } => prep[offset][var.index].into(),
                Entry::Stage2 { offset } => s2[offset][var.index],
                Entry::Public => pubs_b[var.index].into(),
                Entry::Stage2Public => pubs_e[var.index],
                Entry::Challenge => unimplemented!(),
            },
            Expr::IsFirstRow => specials[0].into(),
            Expr::IsLastRow => specials[1].into(),
            Expr::IsTransition => specials[2].into(),
            Expr::Constant(c) => *c,
            Expr::Add { x, y, .. } => rec(x) + rec(y),
            Expr::Sub { x, y, .. } => rec(x) - rec(y),
            Expr::Neg { x, .. } => -rec(x),
            Expr::Mul { x, y, .. } => rec(x) * rec(y),
        }
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn random_f(state: &mut u64) -> F {
        F::from_u64(xorshift(state))
    }

    fn random_ef(state: &mut u64) -> EF {
        let a = random_f(state);
        let b = random_f(state);
        EF::from_basis_coefficients_fn(|i| if i == 0 { a } else { b })
    }

    /// Expressions covering every emission path: base and extension lanes,
    /// lane promotion, reversed subtraction, shared compound sub-expressions
    /// (including across constraints), double-compound operands needing
    /// temporary slots, negation, constants both base and extension,
    /// next-row offsets, and duplicated constraint roots.
    fn test_constraints() -> Vec<Expr> {
        let s1 = |i| v(Entry::Main { offset: 0 }, i);
        let s1n = |i| v(Entry::Main { offset: 1 }, i);
        let prep = |i| v(Entry::Preprocessed { offset: 0 }, i);
        let s2 = |i| v(Entry::Stage2 { offset: 0 }, i);
        let s2n = |i| v(Entry::Stage2 { offset: 1 }, i);
        let pub_b = v(Entry::Public, 0);
        let pub_e = |i| v(Entry::Stage2Public, i);
        let c_b = Expr::Constant(EF::from_u64(77));
        let c_e = Expr::Constant(EF::from_basis_coefficients_fn(|i| {
            F::from_u64(3 + i as u64)
        }));

        // A compound shared within and across constraints.
        let shared = s1(0) + s1(1) * c_b.clone();
        // Double-compound product: both children need evaluation.
        let double =
            (s1(1) + s1(2)) * (s1n(0) - prep(0)) * ((prep(1) + s1(0)) * (s1n(2) + pub_b.clone()));

        let base_constraint = shared.clone() * (shared.clone() - Expr::ONE) + double.clone()
            - Expr::IsTransition * s1n(1);
        // Extension-lane constraint with base/ext mixing in both operand
        // positions, promotion of a base chain, and reversed subtraction.
        let ext_constraint = (shared.clone() - s2(0)) * s2(1)
            + (s2n(0) * c_e.clone() - shared.clone()) * pub_e(1)
            + Expr::IsFirstRow * (s2(0) - pub_e(2));
        let neg_constraint = -(shared * c_e - pub_e(3)) + (-double) * pub_e(0);

        vec![
            base_constraint.clone(),
            ext_constraint,
            neg_constraint,
            // Duplicate root: must fold twice with different alpha powers.
            base_constraint,
        ]
    }

    #[test]
    fn compiled_matches_reference() {
        let constraints = test_constraints();
        let compiled = CompiledConstraints::<F, EF>::compile(&constraints, WP, W1, W2);

        let mut state = 0xDEAD_BEEF_1234_5678u64;
        let width = PackedVal::<F>::WIDTH;

        // Per-lane scalar inputs.
        let gen_rows_f = |state: &mut u64, w: usize| -> Vec<Vec<Vec<F>>> {
            (0..2)
                .map(|_| {
                    (0..width)
                        .map(|_| (0..w).map(|_| random_f(state)).collect())
                        .collect()
                })
                .collect()
        };
        // rows[offset][lane][col]
        let s1_rows = gen_rows_f(&mut state, W1);
        let prep_rows = gen_rows_f(&mut state, WP);
        let s2_rows: Vec<Vec<Vec<EF>>> = (0..2)
            .map(|_| {
                (0..width)
                    .map(|_| (0..W2).map(|_| random_ef(&mut state)).collect())
                    .collect()
            })
            .collect();
        let specials: Vec<[F; 3]> = (0..width)
            .map(|_| {
                [
                    random_f(&mut state),
                    random_f(&mut state),
                    random_f(&mut state),
                ]
            })
            .collect();
        let pubs_b: Vec<F> = (0..NUM_PUB_B).map(|_| random_f(&mut state)).collect();
        let pubs_e: Vec<EF> = (0..NUM_PUB_E).map(|_| random_ef(&mut state)).collect();
        let alpha_powers: Vec<EF> = (0..constraints.len())
            .map(|_| random_ef(&mut state))
            .collect();

        // Packed inputs, flattened local-then-next.
        let pack_f = |rows: &Vec<Vec<Vec<F>>>, w: usize| -> Vec<PackedVal<F>> {
            (0..2)
                .flat_map(|offset| {
                    (0..w).map(move |col| PackedVal::<F>::from_fn(|lane| rows[offset][lane][col]))
                })
                .collect()
        };
        let s1_packed = pack_f(&s1_rows, W1);
        let prep_packed = pack_f(&prep_rows, WP);
        let s2_packed: Vec<PackedExt<F, EF>> = (0..2)
            .flat_map(|offset| {
                let s2_rows = &s2_rows;
                (0..W2).map(move |col| {
                    PackedExt::<F, EF>::from_basis_coefficients_fn(|coeff| {
                        PackedVal::<F>::from_fn(|lane| {
                            s2_rows[offset][lane][col].as_basis_coefficients_slice()[coeff]
                        })
                    })
                })
            })
            .collect();

        let chunk = RowChunk {
            stage_1: &s1_packed,
            preprocessed: &prep_packed,
            stage_2: &s2_packed,
            is_first_row: PackedVal::<F>::from_fn(|lane| specials[lane][0]),
            is_last_row: PackedVal::<F>::from_fn(|lane| specials[lane][1]),
            is_transition: PackedVal::<F>::from_fn(|lane| specials[lane][2]),
        };

        let prepared = compiled.prepare(&pubs_b, &pubs_e, &alpha_powers);
        let mut scratch = compiled.scratch();
        let total = compiled.eval(&prepared, &chunk, &mut scratch);

        for lane in 0..width {
            let s1: [Vec<F>; 2] = [s1_rows[0][lane].clone(), s1_rows[1][lane].clone()];
            let prep: [Vec<F>; 2] = [prep_rows[0][lane].clone(), prep_rows[1][lane].clone()];
            let s2: [Vec<EF>; 2] = [s2_rows[0][lane].clone(), s2_rows[1][lane].clone()];
            let expected: EF = constraints
                .iter()
                .zip(&alpha_powers)
                .map(|(c, &alpha)| {
                    alpha * ref_eval(c, &s1, &prep, &s2, &pubs_b, &pubs_e, &specials[lane])
                })
                .sum();
            let actual = EF::from_basis_coefficients_fn(|coeff| {
                <PackedExt<F, EF> as BasedVectorSpace<PackedVal<F>>>::as_basis_coefficients_slice(
                    &total,
                )[coeff]
                    .as_slice()[lane]
            });
            assert_eq!(actual, expected, "lane {lane}");
        }
    }

    /// Duplicated subtrees must collapse: the compiled program for `N`
    /// identical clones of one constraint must stay within a constant number
    /// of instructions of the single-constraint program (one extra
    /// load-from-slot + fold per clone), rather than re-emitting the tree.
    #[test]
    fn hash_consing_collapses_duplicates() {
        let constraints = test_constraints();
        let single = CompiledConstraints::<F, EF>::compile(
            std::slice::from_ref(&constraints[0]),
            WP,
            W1,
            W2,
        );
        let many: Vec<Expr> = (0..16).map(|_| constraints[0].clone()).collect();
        let repeated = CompiledConstraints::<F, EF>::compile(&many, WP, W1, W2);
        assert!(
            repeated.code_len() <= single.code_len() + 2 * 16,
            "duplicate constraints were not deduplicated: {} vs {}",
            repeated.code_len(),
            single.code_len(),
        );
    }

    /// The reference-count/slot machinery must also produce correct results
    /// with an empty constraint list and constant-only constraints.
    #[test]
    fn degenerate_programs() {
        let empty = CompiledConstraints::<F, EF>::compile(&[], WP, W1, W2);
        let prepared = empty.prepare(&[], &[], &[]);
        let chunk = RowChunk {
            stage_1: &[],
            preprocessed: &[],
            stage_2: &[],
            is_first_row: PackedVal::<F>::ZERO,
            is_last_row: PackedVal::<F>::ZERO,
            is_transition: PackedVal::<F>::ZERO,
        };
        let mut scratch = empty.scratch();
        assert_eq!(
            empty.eval(&prepared, &chunk, &mut scratch),
            PackedExt::<F, EF>::ZERO
        );

        let constant = vec![Expr::Constant(EF::from_u64(5)), var::<EF>(0) * var::<EF>(0)];
        let compiled = CompiledConstraints::<F, EF>::compile(&constant, 0, 1, 0);
        let alpha_powers = [EF::from_u64(9), EF::from_u64(11)];
        let prepared = compiled.prepare(&[], &[], &alpha_powers);
        let x = PackedVal::<F>::from_fn(|lane| F::from_u64(lane as u64));
        let chunk = RowChunk {
            stage_1: &[x, x],
            preprocessed: &[],
            stage_2: &[],
            is_first_row: PackedVal::<F>::ZERO,
            is_last_row: PackedVal::<F>::ZERO,
            is_transition: PackedVal::<F>::ZERO,
        };
        let mut scratch = compiled.scratch();
        let total = compiled.eval(&prepared, &chunk, &mut scratch);
        for lane in 0..PackedVal::<F>::WIDTH {
            let expected = EF::from_u64(9) * EF::from_u64(5)
                + EF::from_u64(11) * EF::from_u64(lane as u64) * EF::from_u64(lane as u64);
            let actual = EF::from_basis_coefficients_fn(|coeff| {
                <PackedExt<F, EF> as BasedVectorSpace<PackedVal<F>>>::as_basis_coefficients_slice(
                    &total,
                )[coeff]
                    .as_slice()[lane]
            });
            assert_eq!(actual, expected, "lane {lane}");
        }
    }
}
