use p3_field::{Field, PrimeCharacteristicRing};

use super::circuit::{Circuit, CompileError, ExtensionParams, Node, compile};
use super::eval::{VarValues, check_topological_order, eval_expr, eval_ext_expr};
use super::expr::{CircuitSpec, Expr, ExtExpr, Lookup, RowOffset, Source};
use crate::types::Val;

type G = Val;

fn gl(x: u64) -> G {
    Val::from_u64(x)
}

/// Goldilocks quadratic extension parameter: X² = 7, matching
/// `BinomialExtensionField<Goldilocks, 2>`.
fn params(karatsuba: bool) -> ExtensionParams<G> {
    ExtensionParams {
        degree: 2,
        w: gl(7),
        karatsuba,
    }
}

/// Tiny xorshift generator so tests need no dependencies.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn g(&mut self) -> G {
        Val::from_u64(self.next_u64())
    }

    fn row(&mut self, width: usize) -> Vec<G> {
        (0..width).map(|_| self.g()).collect()
    }
}

/// Owned random leaf values matching a spec's shape.
struct OwnedValues {
    preprocessed: [Vec<G>; 2],
    main: [Vec<G>; 2],
    stage2: [Vec<G>; 2],
    publics: Vec<G>,
    is_first_row: G,
    is_last_row: G,
    is_transition: G,
}

impl OwnedValues {
    fn random(spec: &CircuitSpec<G>, rng: &mut Rng) -> Self {
        Self {
            preprocessed: [
                rng.row(spec.preprocessed_width),
                rng.row(spec.preprocessed_width),
            ],
            main: [rng.row(spec.main_width), rng.row(spec.main_width)],
            stage2: [rng.row(spec.stage2_width), rng.row(spec.stage2_width)],
            publics: rng.row(spec.num_publics),
            is_first_row: rng.g(),
            is_last_row: rng.g(),
            is_transition: rng.g(),
        }
    }

    fn view(&self) -> VarValues<'_, G> {
        VarValues {
            preprocessed: [&self.preprocessed[0], &self.preprocessed[1]],
            main: [&self.main[0], &self.main[1]],
            stage2: [&self.stage2[0], &self.stage2[1]],
            publics: &self.publics,
            is_first_row: self.is_first_row,
            is_last_row: self.is_last_row,
            is_transition: self.is_transition,
        }
    }
}

fn count_nodes<F: Field>(circuit: &Circuit<F>, pred: impl Fn(&Node<F>) -> bool) -> usize {
    circuit.nodes.iter().filter(|n| pred(n)).count()
}

// -- extension-field reference arithmetic (quadratic, X² = W) --

fn ext_mul(a: [G; 2], b: [G; 2], w: G) -> [G; 2] {
    [a[0] * b[0] + w * a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn ext_inv(a: [G; 2], w: G) -> [G; 2] {
    let denom = a[0] * a[0] - w * a[1] * a[1];
    let denom_inv = denom.inverse();
    [a[0] * denom_inv, -(a[1] * denom_inv)]
}

// -- interning ----------------------------------------------------------

#[test]
fn interning_shares_across_constraints() {
    // `a*b + c` and `c + b*a` must compile to the SAME root: commutative
    // normalization makes b*a ≡ a*b, interning makes the adds identical.
    let (a, b, c) = (Expr::main(0), Expr::main(1), Expr::main(2));
    let spec = CircuitSpec {
        main_width: 3,
        constraints: vec![a.clone() * b.clone() + c.clone(), c + b * a],
        ..Default::default()
    };
    let circuit = compile(&spec, &params(true)).unwrap();
    assert_eq!(circuit.zeros[0], circuit.zeros[1]);
    // var a, var b, mul, var c, add — nothing else.
    assert_eq!(circuit.nodes.len(), 5);
}

#[test]
fn constant_folding_in_interner() {
    // Raw tree constructors bypass the frontend operators' own folding,
    // so this exercises the interner's folding rules.
    let x = || Box::new(Expr::<G>::main(0));
    let zero = || Box::new(Expr::Const(G::ZERO));
    let one = || Box::new(Expr::Const(G::ONE));
    let spec = CircuitSpec {
        main_width: 1,
        constraints: vec![
            Expr::Add(x(), zero()),              // x + 0 = x
            Expr::Mul(x(), one()),               // x * 1 = x
            Expr::Mul(x(), zero()),              // x * 0 = 0
            Expr::Sub(x(), x()),                 // x - x = 0
            Expr::Neg(Box::new(Expr::Neg(x()))), // -(-x) = x
            Expr::Add(Box::new(Expr::Const(gl(2))), Box::new(Expr::Const(gl(3)))),
        ],
        ..Default::default()
    };
    let circuit = compile(&spec, &params(true)).unwrap();
    let var = circuit.zeros[0];
    assert_eq!(
        circuit.nodes[var.index()],
        Node::Var(super::expr::ColRef {
            source: Source::Main,
            offset: RowOffset::Current,
            index: 0,
        })
    );
    assert_eq!(circuit.zeros[1], var);
    assert_eq!(circuit.zeros[4], var);
    let zero_node = circuit.zeros[2];
    assert_eq!(circuit.nodes[zero_node.index()], Node::Const(G::ZERO));
    assert_eq!(circuit.zeros[3], zero_node);
    assert_eq!(circuit.nodes[circuit.zeros[5].index()], Node::Const(gl(5)));
}

#[test]
fn frontend_operators_fold_constants() {
    let e = Expr::constant(gl(2)) + Expr::constant(gl(3));
    assert!(matches!(e, Expr::Const(c) if c == gl(5)));
    let e = Expr::<G>::main(0) * Expr::constant(G::ONE);
    assert!(matches!(e, Expr::Var(_)));
    let e = Expr::<G>::main(0) * Expr::constant(G::ZERO);
    assert!(matches!(e, Expr::Const(c) if c == G::ZERO));
}

#[test]
fn topological_order_holds() {
    let x = Expr::main(0);
    let y = Expr::main_next(1);
    let spec = CircuitSpec {
        main_width: 2,
        stage2_width: 4,
        num_publics: 4,
        constraints: vec![(x.clone() + y.clone()) * (x.clone() - y.clone()) * x.clone()],
        ext_constraints: vec![
            (ExtExpr::public(0, 2) + x * ExtExpr::public(1, 2))
                * ExtExpr::stage2(1, 2, RowOffset::Current)
                - Expr::constant(G::ONE),
        ],
        lookups: vec![Lookup {
            multiplicity: Expr::main(0),
            args: vec![y],
        }],
        ..Default::default()
    };
    let circuit = compile(&spec, &params(true)).unwrap();
    assert!(check_topological_order(&circuit));
}

#[test]
fn degree_accounting() {
    let (x, y, z) = (Expr::main(0), Expr::main(1), Expr::main(2));
    let spec = CircuitSpec {
        main_width: 3,
        num_publics: 1,
        constraints: vec![
            Expr::IsFirstRow * (x.clone() * y.clone() - z.clone()), // 1 + 2 = 3
            Expr::IsTransition * (x.clone() * y),                   // 0 + 2 = 2
            Expr::public(0) * x,                                    // 0 + 1 = 1
            Expr::constant(gl(42)),                                 // 0
        ],
        ..Default::default()
    };
    let circuit = compile(&spec, &params(true)).unwrap();
    let degree = |i: usize| circuit.degrees[circuit.zeros[i].index()];
    assert_eq!(degree(0), 3);
    assert_eq!(degree(1), 2);
    assert_eq!(degree(2), 1);
    assert_eq!(degree(3), 0);
    assert_eq!(circuit.max_constraint_degree, 3);
}

// -- evaluation ---------------------------------------------------------

#[test]
fn sweep_matches_reference_eval() {
    let (x0, x1) = (Expr::main(0), Expr::main(1));
    let constraints = vec![
        (x0.clone() + x1.clone()) * (x0.clone() - Expr::preprocessed(0)),
        Expr::IsFirstRow * (x0.clone() - Expr::public(0)) + Expr::IsLastRow * x1.clone(),
        Expr::IsTransition * (Expr::main_next(0) - x0.clone() * x0.clone()),
        // shares (x0 + x1) with the first constraint
        (x0 + x1) * Expr::public(1) - Expr::preprocessed_next(0),
    ];
    let spec = CircuitSpec {
        main_width: 2,
        preprocessed_width: 1,
        num_publics: 2,
        constraints: constraints.clone(),
        ..Default::default()
    };
    let circuit = compile(&spec, &params(true)).unwrap();
    let mut rng = Rng::new(0xC0FFEE);
    for _ in 0..5 {
        let values = OwnedValues::random(&spec, &mut rng);
        let view = values.view();
        let compiled = circuit.evaluate_constraints(&view);
        let reference: Vec<G> = constraints.iter().map(|c| eval_expr(c, &view)).collect();
        assert_eq!(compiled, reference);
    }
}

fn ext_test_constraints() -> (CircuitSpec<G>, Vec<ExtExpr<G>>) {
    let s2a = ExtExpr::stage2(0, 2, RowOffset::Current);
    let s2b = ExtExpr::stage2(1, 2, RowOffset::Next);
    let beta = ExtExpr::public(0, 2);
    let gamma = ExtExpr::public(1, 2);
    let ext_constraints = vec![
        (beta.clone() + Expr::main(0) * gamma.clone()) * s2a.clone()
            - ExtExpr::constant(vec![gl(5), gl(9)]),
        -(s2a * s2b.clone()) + Expr::main_next(1) * (gamma.clone() * gamma),
        ExtExpr::Coords(vec![Expr::main(0) * Expr::main(1), Expr::preprocessed(0)]) * beta
            + Expr::IsFirstRow * s2b,
    ];
    let spec = CircuitSpec {
        main_width: 2,
        preprocessed_width: 1,
        stage2_width: 4,
        num_publics: 4,
        ext_constraints: ext_constraints.clone(),
        ..Default::default()
    };
    (spec, ext_constraints)
}

#[test]
fn expansion_matches_extension_field_reference() {
    // Compiled coordinates (Karatsuba and schoolbook) must both agree
    // with a direct recursive evaluation in genuine EF arithmetic.
    let (spec, ext_constraints) = ext_test_constraints();
    for karatsuba in [true, false] {
        let p = params(karatsuba);
        let circuit = compile(&spec, &p).unwrap();
        assert_eq!(circuit.zeros.len(), 2 * ext_constraints.len());
        let mut rng = Rng::new(0xDEADBEEF);
        for _ in 0..5 {
            let values = OwnedValues::random(&spec, &mut rng);
            let view = values.view();
            let compiled = circuit.evaluate_constraints(&view);
            for (i, constraint) in ext_constraints.iter().enumerate() {
                let reference = eval_ext_expr(constraint, &view, &p);
                assert_eq!(
                    &compiled[2 * i..2 * i + 2],
                    reference.as_slice(),
                    "constraint {i}, karatsuba={karatsuba}"
                );
            }
        }
    }
}

#[test]
fn scalar_mul_expands_to_d_multiplications() {
    // Base(m) * stage2_slot must compile to exactly D multiplications
    // [m·inv_0, m·inv_1] — no Karatsuba detour, no subtractions.
    let spec = CircuitSpec {
        main_width: 1,
        stage2_width: 4,
        ext_constraints: vec![Expr::main(0) * ExtExpr::stage2(1, 2, RowOffset::Current)],
        ..Default::default()
    };
    let circuit = compile(&spec, &params(true)).unwrap();
    assert_eq!(count_nodes(&circuit, |n| matches!(n, Node::Mul(..))), 2);
    assert_eq!(count_nodes(&circuit, |n| matches!(n, Node::Sub(..))), 0);
    assert_eq!(count_nodes(&circuit, |n| matches!(n, Node::Add(..))), 0);
}

#[test]
fn lookup_prefix_segmentation() {
    let m = Expr::main(0);
    let arg0 = Expr::main(1) + Expr::main(2);
    let arg1 = Expr::main(2) * Expr::main(2);
    let spec = CircuitSpec {
        main_width: 3,
        lookups: vec![Lookup {
            multiplicity: m.clone(),
            args: vec![arg0.clone(), arg1.clone()],
        }],
        // Reuses arg0's expression: must resolve to the prefix node.
        constraints: vec![arg0.clone() * m.clone()],
        ..Default::default()
    };
    let circuit = compile(&spec, &params(true)).unwrap();
    // Prefix: var m, var 1, var 2, add, mul(var2, var2) — five nodes.
    assert_eq!(circuit.lookup_prefix_len, 5);
    // The constraint adds exactly one node (its outer mul); its operands
    // are prefix nodes.
    assert_eq!(circuit.nodes.len(), 6);
    let lookup = &circuit.lookups[0];
    assert!(lookup.multiplicity.index() < circuit.lookup_prefix_len);
    assert!(
        lookup
            .args
            .iter()
            .all(|a| a.index() < circuit.lookup_prefix_len)
    );

    // Prefix sweep computes the concrete lookup values.
    let mut rng = Rng::new(42);
    let values = OwnedValues::random(&spec, &mut rng);
    let view = values.view();
    let mut buf = Vec::new();
    circuit.sweep_lookup_prefix(&view, &mut buf);
    assert_eq!(buf.len(), circuit.lookup_prefix_len);
    let lookups = circuit.lookup_values(&buf);
    assert_eq!(lookups[0].multiplicity, eval_expr(&m, &view));
    assert_eq!(lookups[0].args[0], eval_expr(&arg0, &view));
    assert_eq!(lookups[0].args[1], eval_expr(&arg1, &view));
}

#[test]
fn lookup_style_constraint_vanishes_on_consistent_values() {
    // The real use-case shape: msg·inv − 1 with msg = β + (a0 + γ·a1),
    // where inv is the actual EF inverse of msg. Both coordinate
    // constraints must evaluate to zero.
    let w = gl(7);
    let msg = ExtExpr::public(0, 2) + (Expr::main(0) + ExtExpr::public(1, 2) * Expr::main(1));
    let constraint = msg * ExtExpr::stage2(1, 2, RowOffset::Current) - Expr::constant(G::ONE);
    let spec = CircuitSpec {
        main_width: 2,
        stage2_width: 4,
        num_publics: 4,
        ext_constraints: vec![constraint],
        ..Default::default()
    };
    let circuit = compile(&spec, &params(true)).unwrap();

    let mut rng = Rng::new(7);
    let (a0, a1) = (rng.g(), rng.g());
    let beta = [rng.g(), rng.g()];
    let gamma = [rng.g(), rng.g()];
    let fingerprint = [a0 + gamma[0] * a1, gamma[1] * a1];
    let msg_value = [beta[0] + fingerprint[0], beta[1] + fingerprint[1]];
    let inv_value = ext_inv(msg_value, w);
    assert_eq!(ext_mul(msg_value, inv_value, w), [G::ONE, G::ZERO]);

    let main = vec![a0, a1];
    let stage2 = vec![rng.g(), rng.g(), inv_value[0], inv_value[1]];
    let publics = vec![beta[0], beta[1], gamma[0], gamma[1]];
    let empty: Vec<G> = vec![];
    let view = VarValues {
        preprocessed: [&empty, &empty],
        main: [&main, &main],
        stage2: [&stage2, &stage2],
        publics: &publics,
        is_first_row: G::ONE,
        is_last_row: G::ZERO,
        is_transition: G::ONE,
    };
    assert_eq!(circuit.evaluate_constraints(&view), vec![G::ZERO, G::ZERO]);
}

// -- validation ---------------------------------------------------------

#[test]
fn validation_errors() {
    let base = CircuitSpec::<G> {
        main_width: 2,
        preprocessed_width: 1,
        stage2_width: 4,
        num_publics: 4,
        ..Default::default()
    };
    let p = params(true);

    let spec = CircuitSpec {
        ext_constraints: vec![ExtExpr::Coords(vec![Expr::main(0)])],
        ..base.clone()
    };
    assert_eq!(
        compile(&spec, &p),
        Err(CompileError::CoordsLength {
            constraint: 0,
            expected: 2,
            got: 1,
        })
    );

    let spec = CircuitSpec {
        constraints: vec![Expr::main(5)],
        ..base.clone()
    };
    assert_eq!(
        compile(&spec, &p),
        Err(CompileError::ColumnOutOfRange {
            source: Source::Main,
            index: 5,
            width: 2,
        })
    );

    let spec = CircuitSpec {
        constraints: vec![Expr::public(9)],
        ..base.clone()
    };
    assert_eq!(
        compile(&spec, &p),
        Err(CompileError::PublicOutOfRange { index: 9, count: 4 })
    );

    // Stage-2 columns are off-limits to base constraints and lookups.
    let spec = CircuitSpec {
        constraints: vec![Expr::stage2(0)],
        ..base.clone()
    };
    assert_eq!(compile(&spec, &p), Err(CompileError::Stage2InBaseContext));
    let spec = CircuitSpec {
        lookups: vec![Lookup {
            multiplicity: Expr::main(0),
            args: vec![Expr::stage2(0)],
        }],
        ..base.clone()
    };
    assert_eq!(compile(&spec, &p), Err(CompileError::Stage2InBaseContext));

    // An ext constraint built solely from base embeddings is rejected.
    let spec = CircuitSpec {
        ext_constraints: vec![ExtExpr::from(Expr::main(0)) * Expr::main(1)],
        ..base
    };
    assert_eq!(
        compile(&spec, &p),
        Err(CompileError::PurelyBaseExtConstraint { constraint: 0 })
    );
}
