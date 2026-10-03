//! Isolated static custom-gate Plonkish prototype for 64-byte BLAKE3 hashes.
//! Not wired into the production verifier. Copy wiring, table queries, and
//! public bindings are constrained; native word arithmetic is witness advice.
use multi_stark::eval::{VarValues, eval_expr};
use multi_stark::{expr::Expr, lookup::Lookup, system::CircuitInputs, types::Val};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use std::collections::HashMap;

const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const ROT: [u32; 4] = [16, 12, 8, 7];
const NS: u8 = 91;
fn f(x: usize) -> Val {
    Val::from_usize(x)
}
fn c(x: usize) -> Expr<Val> {
    Expr::constant(f(x))
}
fn m(x: usize) -> Expr<Val> {
    Expr::main(u32::try_from(x).unwrap())
}
fn p(x: usize) -> Expr<Val> {
    Expr::preprocessed(u32::try_from(x).unwrap())
}
fn mw(t: usize) -> usize {
    match t {
        0 | 2 => 32,
        1 | 3 => 40,
        4 => 12,
        _ => 4,
    }
}
fn slots(t: usize) -> usize {
    match t {
        0..=3 => 6,
        4 => 3,
        _ => 1,
    }
}
fn fw(t: usize) -> usize {
    if t == 5 { 11 } else { 2 + 2 * slots(t) }
}
fn table_query(id: usize, args: Vec<Expr<Val>>) -> Lookup<Expr<Val>> {
    Lookup::pull(p(0), [vec![c(NS.into()), c(2), c(id)], args].concat())
}

enum Recipe {
    Input(usize),
    Constant(u32),
    Sum([usize; 3]),
    Rotate(usize, usize, u32),
    Xor(usize, usize),
}
struct Boundary {
    constant: Option<u32>,
    public: Option<usize>,
}
pub(crate) struct Compact {
    recipes: Vec<Recipe>,
    rows: [Vec<Vec<usize>>; 6],
    boundaries: Vec<Boundary>,
    instances: usize,
}
impl Compact {
    fn word(&mut self, r: Recipe) -> usize {
        let i = self.recipes.len();
        self.recipes.push(r);
        i
    }
    fn constant(&mut self, x: u32) -> usize {
        let w = self.word(Recipe::Constant(x));
        self.rows[5].push(vec![w]);
        self.boundaries.push(Boundary {
            constant: Some(x),
            public: None,
        });
        w
    }
    fn step(&mut self, t: usize, a: usize, b: usize, msg: usize, d: usize) -> (usize, usize) {
        let sum = self.word(Recipe::Sum([a, b, msg]));
        let rotated = self.word(Recipe::Rotate(sum, d, ROT[t]));
        self.rows[t].push(vec![a, b, msg, d, sum, rotated]);
        (sum, rotated)
    }
    pub(crate) fn new(instances: usize) -> Self {
        assert!(instances > 0);
        let mut s = Self {
            recipes: vec![],
            rows: std::array::from_fn(|_| vec![]),
            boundaries: vec![],
            instances,
        };
        for instance in 0..instances {
            let mut msg = std::array::from_fn::<_, 16, _>(|i| {
                let w = s.word(Recipe::Input(instance * 16 + i));
                s.rows[5].push(vec![w]);
                s.boundaries.push(Boundary {
                    constant: None,
                    public: Some(instance * 24 + i),
                });
                w
            });
            let iv = IV.map(|v| s.constant(v));
            let zero = s.constant(0);
            let len = s.constant(64);
            let flags = s.constant(11); // CHUNK_START|CHUNK_END|ROOT
            let mut state = std::array::from_fn::<_, 16, _>(|i| match i {
                0..8 => iv[i],
                8..12 => iv[i - 8],
                12 | 13 => zero,
                14 => len,
                _ => flags,
            });
            for _ in 0..7 {
                for (j, [a, b, cc, d]) in [
                    [0, 4, 8, 12],
                    [1, 5, 9, 13],
                    [2, 6, 10, 14],
                    [3, 7, 11, 15],
                    [0, 5, 10, 15],
                    [1, 6, 11, 12],
                    [2, 7, 8, 13],
                    [3, 4, 9, 14],
                ]
                .into_iter()
                .enumerate()
                {
                    (state[a], state[d]) = s.step(0, state[a], state[b], msg[2 * j], state[d]);
                    (state[cc], state[b]) = s.step(1, state[cc], state[d], zero, state[b]);
                    (state[a], state[d]) = s.step(2, state[a], state[b], msg[2 * j + 1], state[d]);
                    (state[cc], state[b]) = s.step(3, state[cc], state[d], zero, state[b]);
                }
                msg = PERM.map(|i| msg[i]);
            }
            for i in 0..8 {
                let w = s.word(Recipe::Xor(state[i], state[i + 8]));
                s.rows[4].push(vec![state[i], state[i + 8], w]);
                s.rows[5].push(vec![w]);
                s.boundaries.push(Boundary {
                    constant: None,
                    public: Some(instance * 24 + 16 + i),
                });
            }
        }
        s
    }
    pub(crate) fn definitions(&self) -> Vec<CircuitInputs<Val>> {
        let mut fixed: Vec<_> = (0..6)
            .map(|t| vec![Val::ZERO; self.rows[t].len().max(2).next_power_of_two() * fw(t)])
            .collect();
        let mut occurrences = vec![vec![]; self.recipes.len()];
        let mut label = 0usize;
        for t in 0..6 {
            fixed[t][1] = Val::ONE; // mandatory activation marker
            for (r, row) in self.rows[t].iter().enumerate() {
                fixed[t][r * fw(t)] = Val::ONE;
                for (slot, &word) in row.iter().enumerate() {
                    fixed[t][r * fw(t) + 2 + slot] = f(label);
                    occurrences[word].push((t, r, slot, label));
                    label += 1;
                }
            }
        }
        assert!((label as u128) < u128::from(Val::ORDER_U64));
        for uses in occurrences {
            for (i, &(t, r, slot, _)) in uses.iter().enumerate() {
                fixed[t][r * fw(t) + 2 + slots(t) + slot] = f(uses[(i + 1) % uses.len()].3);
            }
        }
        for (r, boundary) in self.boundaries.iter().enumerate() {
            let row = &mut fixed[5][r * fw(5)..(r + 1) * fw(5)];
            if let Some(value) = boundary.constant {
                row[4] = Val::ONE;
                for (i, b) in value.to_le_bytes().into_iter().enumerate() {
                    row[5 + i] = Val::from_u8(b);
                }
            }
            if let Some(index) = boundary.public {
                row[9] = Val::ONE;
                row[10] = f(index);
            }
        }
        let mut defs = vec![];
        for (t, data) in fixed.into_iter().enumerate() {
            let mut lookups = vec![Lookup::pull(p(1), vec![c(NS.into()), c(3), c(t)])];
            for slot in 0..slots(t) {
                let args = |label| {
                    let mut a = vec![c(NS.into()), c(0), p(label)];
                    a.extend((0..4).map(|i| m(slot * 4 + i)));
                    a
                };
                lookups.push(Lookup::push(p(0), args(2 + slot)));
                lookups.push(Lookup::pull(p(0), args(2 + slots(t) + slot)));
            }
            let mut constraints = vec![];
            if t < 4 {
                for i in 0..4 {
                    let carry = m(24 + i);
                    constraints
                        .push(carry.clone() * (carry.clone() - c(1)) * (carry.clone() - c(2)));
                    constraints.push(
                        m(i) + m(4 + i) + m(8 + i) + if i == 0 { c(0) } else { m(23 + i) }
                            - m(16 + i)
                            - c(256) * carry,
                    );
                    lookups.push(table_query(0, vec![m(12 + i), m(16 + i), m(28 + i)]));
                }
                let shift = usize::try_from(ROT[t] / 8).unwrap();
                let bits = usize::try_from(ROT[t] % 8).unwrap();
                for i in 0..4 {
                    if bits == 0 {
                        constraints.push(m(20 + i) - m(28 + (i + shift) % 4));
                    } else {
                        lookups.push(table_query(
                            1,
                            vec![c(bits), m(28 + i), m(32 + i), m(36 + i)],
                        ));
                        constraints.push(
                            m(20 + i)
                                - m(36 + (i + shift) % 4)
                                - c(1 << (8 - bits)) * m(32 + (i + shift + 1) % 4),
                        );
                    }
                }
            } else if t == 4 {
                for i in 0..4 {
                    lookups.push(table_query(0, vec![m(i), m(4 + i), m(8 + i)]));
                }
            } else {
                for i in 0..4 {
                    constraints.push(p(4) * (m(i) - p(5 + i)));
                    lookups.push(table_query(2, vec![m(i)]));
                }
                let mut args = vec![c(NS.into()), c(1), p(10)];
                args.extend((0..4).map(m));
                lookups.push(Lookup::pull(p(9), args));
            }
            defs.push(CircuitInputs {
                main_width: mw(t),
                preprocessed: Some(RowMajorMatrix::new(data, fw(t))),
                constraints,
                lookups,
                lookup_group_size: 3,
                ..Default::default()
            });
        }
        let xor: Vec<_> = (0..256)
            .flat_map(|a| (0..256).flat_map(move |b| [f(a), f(b), f(a ^ b)]))
            .collect();
        let split: Vec<_> = [4, 7]
            .into_iter()
            .flat_map(|bits| {
                (0..256).flat_map(move |x| [f(bits), f(x), f(x & ((1 << bits) - 1)), f(x >> bits)])
            })
            .collect();
        for (id, (data, width)) in [(xor, 3), (split, 4), ((0..256).map(f).collect(), 1)]
            .into_iter()
            .enumerate()
        {
            let mut args = vec![c(NS.into()), c(2), c(id)];
            args.extend((0..width).map(p));
            defs.push(CircuitInputs {
                main_width: 1,
                preprocessed: Some(RowMajorMatrix::new(data, width)),
                lookups: vec![Lookup::push(m(0), args)],
                ..Default::default()
            });
        }
        defs
    }
    pub(crate) fn traces(
        &self,
        messages: &[[u8; 64]],
        defs: &[CircuitInputs<Val>],
    ) -> Vec<RowMajorMatrix<Val>> {
        assert_eq!(messages.len(), self.instances);
        let mut words: Vec<u32> = vec![];
        for recipe in &self.recipes {
            let value = match *recipe {
                Recipe::Input(i) => u32::from_le_bytes(
                    messages[i / 16][(i % 16) * 4..(i % 16 + 1) * 4]
                        .try_into()
                        .unwrap(),
                ),
                Recipe::Constant(v) => v,
                Recipe::Sum([a, b, c]) => words[a].wrapping_add(words[b]).wrapping_add(words[c]),
                Recipe::Rotate(a, b, r) => (words[a] ^ words[b]).rotate_right(r),
                Recipe::Xor(a, b) => words[a] ^ words[b],
            };
            words.push(value);
        }
        let mut traces: Vec<_> = defs
            .iter()
            .map(|d| {
                RowMajorMatrix::new(
                    vec![Val::ZERO; d.preprocessed.as_ref().unwrap().height() * d.main_width],
                    d.main_width,
                )
            })
            .collect();
        for t in 0..6 {
            for (r, slots) in self.rows[t].iter().enumerate() {
                let row = &mut traces[t].values[r * mw(t)..(r + 1) * mw(t)];
                for (slot, &word) in slots.iter().enumerate() {
                    for (i, b) in words[word].to_le_bytes().into_iter().enumerate() {
                        row[slot * 4 + i] = Val::from_u8(b);
                    }
                }
                if t < 4 {
                    let mut carry = 0u16;
                    for i in 0..4 {
                        let sum = u16::from(words[slots[0]].to_le_bytes()[i])
                            + u16::from(words[slots[1]].to_le_bytes()[i])
                            + u16::from(words[slots[2]].to_le_bytes()[i])
                            + carry;
                        carry = sum >> 8;
                        row[24 + i] = Val::from_u16(carry);
                        let x = words[slots[3]].to_le_bytes()[i] ^ words[slots[4]].to_le_bytes()[i];
                        row[28 + i] = Val::from_u8(x);
                        let bits = ROT[t] % 8;
                        if bits != 0 {
                            row[32 + i] = Val::from_u8(x & ((1 << bits) - 1));
                            row[36 + i] = Val::from_u8(x >> bits);
                        }
                    }
                }
            }
        }
        let indices: Vec<HashMap<Vec<Val>, usize>> = defs[6..]
            .iter()
            .map(|d| {
                d.preprocessed
                    .as_ref()
                    .unwrap()
                    .rows()
                    .enumerate()
                    .map(|(i, r)| (r.collect(), i))
                    .collect()
            })
            .collect();
        for t in 0..6 {
            for r in 0..self.rows[t].len() {
                let main = traces[t].row_slice(r).unwrap();
                let prep = defs[t].preprocessed.as_ref().unwrap().row_slice(r).unwrap();
                let view = view(&main, &prep, r, traces[t].height());
                let mut increments = vec![];
                for lookup in &defs[t].lookups {
                    let args: Vec<_> = lookup.args.iter().map(|e| eval_expr(e, &view)).collect();
                    if args[1] == f(2) {
                        let id = usize::try_from(args[2].as_canonical_u64()).unwrap();
                        increments.push((id, indices[id][&args[3..]]));
                    }
                }
                drop(main);
                for (id, index) in increments {
                    traces[6 + id].values[index] += Val::ONE;
                }
            }
        }
        traces
    }
    pub(crate) fn claims(&self, messages: &[[u8; 64]], digests: &[[u8; 32]]) -> Vec<Vec<Val>> {
        assert_eq!(messages.len(), self.instances);
        assert_eq!(digests.len(), self.instances);
        let mut claims: Vec<_> = (0..6).map(|t| vec![f(NS.into()), f(3), f(t)]).collect();
        for (instance, (msg, digest)) in messages.iter().zip(digests).enumerate() {
            for (i, bytes) in msg
                .as_chunks::<4>()
                .0
                .iter()
                .chain(digest.as_chunks::<4>().0.iter())
                .enumerate()
            {
                let mut claim = vec![f(NS.into()), f(1), f(instance * 24 + i)];
                claim.extend(bytes.iter().copied().map(Val::from_u8));
                claims.push(claim);
            }
        }
        claims
    }
}
fn view<'a>(main: &'a [Val], prep: &'a [Val], r: usize, height: usize) -> VarValues<'a, Val> {
    VarValues {
        main: [main, main],
        preprocessed: [prep, prep],
        stage2: [&[], &[]],
        publics: &[],
        is_first_row: Val::from_bool(r == 0),
        is_last_row: Val::from_bool(r + 1 == height),
        is_transition: Val::from_bool(r + 1 < height),
    }
}
/// Independent AIR + exact lookup multiset check, also usable with corrupt traces.
pub(crate) fn check(
    defs: &[CircuitInputs<Val>],
    traces: &[RowMajorMatrix<Val>],
    claims: &[Vec<Val>],
) -> bool {
    if defs.len() != traces.len() {
        return false;
    }
    let mut balance: HashMap<Vec<Val>, Val> = HashMap::new();
    for claim in claims {
        *balance.entry(claim.clone()).or_insert(Val::ZERO) += Val::ONE;
    }
    for (def, trace) in defs.iter().zip(traces) {
        if trace.height() != 0 && trace.height() != def.preprocessed.as_ref().unwrap().height() {
            return false;
        }
        for r in 0..trace.height() {
            let main = trace.row_slice(r).unwrap();
            let prep = def.preprocessed.as_ref().unwrap().row_slice(r).unwrap();
            let view = view(&main, &prep, r, trace.height());
            if def
                .constraints
                .iter()
                .any(|e| eval_expr(e, &view) != Val::ZERO)
            {
                return false;
            }
            for l in &def.lookups {
                let args = l.args.iter().map(|e| eval_expr(e, &view)).collect();
                *balance.entry(args).or_insert(Val::ZERO) += eval_expr(&l.multiplicity, &view);
            }
        }
    }
    balance.values().all(|v| *v == Val::ZERO)
}
