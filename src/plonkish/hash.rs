//! Compact BLAKE3 custom gates, linked to the generic Plonkish circuit by
//! fixed per-call byte lookups. Native hashing supplies advice only.
//!
//! Layout: four add/XOR/rotate kinds (16, 12, 8, 7 bits), final XOR,
//! and word boundaries, followed by shared XOR/split/byte tables. An add
//! row contains six byte-decomposed words (a,b,message,d,sum,rotated), four
//! carries, four XOR bytes, and, for non-byte rotations, four low/high pairs.
//! Word occurrences form fixed global copy cycles across all gate kinds.
//!
//! Range soundness starts at byte-checked input/constant boundaries. Sum
//! bytes are bounded by XOR lookups; rotated bytes are bounded combinations
//! of checked splits. Thus each add equation is an integer equation (not
//! a field-overflow alias), with carries constrained to {0,1,2}.
//!
//! Boundary fixed columns: enable, activation, own/next copy labels, four
//! constant flags, four constant bytes, four signed bridge multiplicities,
//! call index, byte offset. Padding bytes are fixed to zero. Input bindings
//! pull from generic wires; digest bindings push back. Channel 4 contains
//! this entire vocabulary, disjoint from generic copy/public/table channels.
use super::{Value, WitnessError};
use crate::traits::Field;
use crate::{expr::Expr, lookup::Lookup, system::CircuitInputs};
use p3_blake3::Blake3;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_symmetric::CryptographicHasher;
use std::collections::HashMap;
const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const ROT: [u32; 4] = [16, 12, 8, 7];
fn f<F: Field>(x: usize) -> F {
    F::from_usize(x)
}
fn c<F: Field>(x: usize) -> Expr<F> {
    Expr::constant(f(x))
}
fn m<F: Field>(x: usize) -> Expr<F> {
    Expr::main(u32::try_from(x).unwrap())
}
fn p<F: Field>(x: usize) -> Expr<F> {
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
    if t == 5 { 18 } else { 2 + 2 * slots(t) }
}
fn table_query<F: Field>(namespace: F, id: usize, args: Vec<Expr<F>>) -> Lookup<Expr<F>> {
    Lookup::pull(
        p(0),
        [vec![Expr::constant(namespace), c(4), c(2), c(id)], args].concat(),
    )
}

enum Recipe {
    Input(usize, usize, usize),
    Constant(u32),
    Sum([usize; 3]),
    Rotate(usize, usize, u32),
    Xor(usize, usize),
}
enum Boundary {
    Constant(u32),
    Bridge {
        call: usize,
        offset: usize,
        len: usize,
        output: bool,
    },
}
#[derive(Clone, Copy)]
struct Output {
    cv: [usize; 8],
    block: [usize; 16],
    counter: u64,
    len: u32,
    flags: u32,
}
pub(super) struct HashCall {
    pub input: Vec<Value>,
    pub output: [Value; 32],
}
#[derive(Default)]
pub(super) struct Compact {
    pub calls: Vec<HashCall>,
    pub compressions: usize,
    recipes: Vec<Recipe>,
    rows: [Vec<Vec<usize>>; 6],
    boundaries: Vec<Boundary>,
    constants: HashMap<u32, usize>,
}
impl Compact {
    fn word(&mut self, r: Recipe) -> usize {
        let i = self.recipes.len();
        self.recipes.push(r);
        i
    }
    fn constant(&mut self, x: u32) -> usize {
        if let Some(&i) = self.constants.get(&x) {
            return i;
        }
        let w = self.word(Recipe::Constant(x));
        self.constants.insert(x, w);
        self.rows[5].push(vec![w]);
        self.boundaries.push(Boundary::Constant(x));
        w
    }
    fn step(&mut self, t: usize, a: usize, b: usize, msg: usize, d: usize) -> (usize, usize) {
        let sum = self.word(Recipe::Sum([a, b, msg]));
        let rot = self.word(Recipe::Rotate(sum, d, ROT[t]));
        self.rows[t].push(vec![a, b, msg, d, sum, rot]);
        (sum, rot)
    }
    fn compress(&mut self, out: Output, root: bool) -> [usize; 8] {
        self.compressions += 1;
        let iv = IV.map(|v| self.constant(v));
        let zero = self.constant(0);
        let low = self.constant(u32::try_from(out.counter & 0xffff_ffff).unwrap());
        let high = self.constant(u32::try_from(out.counter >> 32).unwrap());
        let len = self.constant(out.len);
        let flags = self.constant(out.flags | if root { 8 } else { 0 });
        let mut state: [usize; 16] = std::array::from_fn(|i| match i {
            0..8 => out.cv[i],
            8..12 => iv[i - 8],
            12 => low,
            13 => high,
            14 => len,
            _ => flags,
        });
        let mut msg = out.block;
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
                (state[a], state[d]) = self.step(0, state[a], state[b], msg[2 * j], state[d]);
                (state[cc], state[b]) = self.step(1, state[cc], state[d], zero, state[b]);
                (state[a], state[d]) = self.step(2, state[a], state[b], msg[2 * j + 1], state[d]);
                (state[cc], state[b]) = self.step(3, state[cc], state[d], zero, state[b]);
            }
            msg = PERM.map(|i| msg[i]);
        }
        std::array::from_fn(|i| {
            let w = self.word(Recipe::Xor(state[i], state[i + 8]));
            self.rows[4].push(vec![state[i], state[i + 8], w]);
            w
        })
    }
    fn subtree(&mut self, input: &[usize], len: usize, counter: u64) -> Output {
        let iv = IV.map(|x| self.constant(x));
        if len > 1024 {
            let chunks = len.div_ceil(1024);
            let left = 1usize << (usize::BITS - 1 - (chunks - 1).leading_zeros());
            let split = left * 1024;
            let lo = self.subtree(&input[..split / 4], split, counter);
            let lo = self.compress(lo, false);
            let hi = self.subtree(
                &input[split / 4..],
                len - split,
                counter + u64::try_from(left).unwrap(),
            );
            let hi = self.compress(hi, false);
            return Output {
                cv: iv,
                block: std::array::from_fn(|i| if i < 8 { lo[i] } else { hi[i - 8] }),
                counter: 0,
                len: 64,
                flags: 4,
            };
        }
        let zero = self.constant(0);
        let blocks = len.div_ceil(64).max(1);
        let mut cv = iv;
        for i in 0..blocks {
            let block = std::array::from_fn(|j| input.get(i * 16 + j).copied().unwrap_or(zero));
            let out = Output {
                cv,
                block,
                counter,
                len: u32::try_from(len.saturating_sub(i * 64).min(64)).unwrap(),
                flags: if i == 0 { 1 } else { 0 } | if i + 1 == blocks { 2 } else { 0 },
            };
            if i + 1 == blocks {
                return out;
            }
            cv = self.compress(out, false);
        }
        unreachable!()
    }
    pub(super) fn add_call(&mut self, call: HashCall) {
        let id = self.calls.len();
        let len = call.input.len();
        let mut input = vec![];
        for offset in (0..len).step_by(4) {
            let n = (len - offset).min(4);
            let w = self.word(Recipe::Input(id, offset, n));
            input.push(w);
            self.rows[5].push(vec![w]);
            self.boundaries.push(Boundary::Bridge {
                call: id,
                offset,
                len: n,
                output: false,
            });
        }
        let out = self.subtree(&input, len, 0);
        let digest = self.compress(out, true);
        for (i, w) in digest.into_iter().enumerate() {
            self.rows[5].push(vec![w]);
            self.boundaries.push(Boundary::Bridge {
                call: id,
                offset: len + 4 * i,
                len: 4,
                output: true,
            });
        }
        self.calls.push(call);
    }
    pub(super) fn layouts(&self) -> Vec<(usize, usize, usize)> {
        if self.calls.is_empty() {
            return vec![];
        }
        let mut result: Vec<_> = (0..6)
            .map(|t| (self.rows[t].len().max(2).next_power_of_two(), mw(t), fw(t)))
            .collect();
        result.extend([(65536, 1, 3), (512, 1, 4), (256, 1, 1)]);
        result
    }
    fn messages<F: Field>(&self, values: &[F]) -> Result<Vec<Vec<u8>>, WitnessError> {
        let bytes: HashMap<_, _> = (0..=255u8).map(|x| (F::from_u8(x), x)).collect();
        self.calls
            .iter()
            .map(|call| {
                call.input
                    .iter()
                    .map(|v| {
                        bytes
                            .get(&values[v.index])
                            .copied()
                            .ok_or(WitnessError::HashMismatch)
                    })
                    .collect()
            })
            .collect()
    }
    pub(super) fn check_values<F: Field>(&self, values: &[F]) -> Result<(), WitnessError> {
        for (call, msg) in self.calls.iter().zip(self.messages(values)?) {
            let digest: [u8; 32] = Blake3.hash_iter(msg);
            if call
                .output
                .iter()
                .zip(digest)
                .any(|(v, b)| values[v.index] != F::from_u8(b))
            {
                return Err(WitnessError::HashMismatch);
            }
        }
        Ok(())
    }
    pub(super) fn claims<F: Field>(&self, namespace: F) -> Vec<Vec<F>> {
        if self.calls.is_empty() {
            return vec![];
        }
        (0..6).map(|t| vec![namespace, f(4), f(3), f(t)]).collect()
    }
    pub(super) fn definitions<F: Field>(&self, namespace: F) -> Vec<CircuitInputs<F>> {
        let mut fixed: Vec<_> = (0..6)
            .map(|t| vec![F::ZERO; self.rows[t].len().max(2).next_power_of_two() * fw(t)])
            .collect();
        let mut occurrences = vec![vec![]; self.recipes.len()];
        let mut label = 0usize;
        for t in 0..6 {
            fixed[t][1] = F::ONE; // mandatory activation marker
            for (r, row) in self.rows[t].iter().enumerate() {
                fixed[t][r * fw(t)] = F::ONE;
                for (slot, &word) in row.iter().enumerate() {
                    fixed[t][r * fw(t) + 2 + slot] = f(label);
                    occurrences[word].push((t, r, slot, label));
                    label += 1;
                }
            }
        }
        assert!(F::prime_order_exceeds(label));
        for uses in occurrences {
            for (i, &(t, r, slot, _)) in uses.iter().enumerate() {
                fixed[t][r * fw(t) + 2 + slots(t) + slot] = f(uses[(i + 1) % uses.len()].3);
            }
        }

        for (r, boundary) in self.boundaries.iter().enumerate() {
            let row = &mut fixed[5][r * fw(5)..(r + 1) * fw(5)];
            match *boundary {
                Boundary::Constant(value) => {
                    for (i, byte) in value.to_le_bytes().into_iter().enumerate() {
                        row[4 + i] = F::ONE;
                        row[8 + i] = F::from_u8(byte);
                    }
                }
                Boundary::Bridge {
                    call,
                    offset,
                    len,
                    output,
                } => {
                    row[16] = f(call);
                    row[17] = f(offset);
                    for i in 0..4 {
                        if i < len {
                            row[12 + i] = if output { F::NEG_ONE } else { F::ONE };
                        } else {
                            row[4 + i] = F::ONE;
                        }
                    }
                }
            }
        }
        let mut defs = vec![];
        for (t, data) in fixed.into_iter().enumerate() {
            let mut lookups = vec![Lookup::pull(
                p(1),
                vec![Expr::constant(namespace), c(4), c(3), c(t)],
            )];
            for slot in 0..slots(t) {
                let args = |label| {
                    let mut a = vec![Expr::constant(namespace), c(4), c(0), p(label)];
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
                    lookups.push(table_query(
                        namespace,
                        0,
                        vec![m(12 + i), m(16 + i), m(28 + i)],
                    ));
                }
                let shift = usize::try_from(ROT[t] / 8).unwrap();
                let bits = usize::try_from(ROT[t] % 8).unwrap();
                for i in 0..4 {
                    if bits == 0 {
                        constraints.push(m(20 + i) - m(28 + (i + shift) % 4));
                    } else {
                        lookups.push(table_query(
                            namespace,
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
                    lookups.push(table_query(namespace, 0, vec![m(i), m(4 + i), m(8 + i)]));
                }
            } else {
                for i in 0..4 {
                    constraints.push(p(4 + i) * (m(i) - p(8 + i)));
                    lookups.push(table_query(namespace, 2, vec![m(i)]));
                }
                for i in 0..4 {
                    lookups.push(Lookup::pull(
                        p(12 + i),
                        vec![
                            Expr::constant(namespace),
                            c(4),
                            c(4),
                            p(16),
                            p(17) + c(i),
                            m(i),
                        ],
                    ));
                }
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
            let mut args = vec![Expr::constant(namespace), c(4), c(2), c(id)];
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
    pub(super) fn traces<F: Field>(
        &self,
        values: &[F],
        defs: &[CircuitInputs<F>],
    ) -> Result<Vec<RowMajorMatrix<F>>, WitnessError> {
        self.selected_traces(values, defs, None)
    }

    pub(super) fn trace<F: Field>(
        &self,
        values: &[F],
        defs: &[CircuitInputs<F>],
        index: usize,
    ) -> Result<RowMajorMatrix<F>, WitnessError> {
        Ok(self
            .selected_traces(values, defs, Some(index))?
            .remove(index))
    }

    fn selected_traces<F: Field>(
        &self,
        values: &[F],
        defs: &[CircuitInputs<F>],
        selected: Option<usize>,
    ) -> Result<Vec<RowMajorMatrix<F>>, WitnessError> {
        let messages = self.messages(values)?;
        let mut words: Vec<u32> = Vec::with_capacity(self.recipes.len());
        for recipe in &self.recipes {
            let value = match *recipe {
                Recipe::Input(call, offset, len) => {
                    let mut bytes = [0u8; 4];
                    bytes[..len].copy_from_slice(&messages[call][offset..offset + len]);
                    u32::from_le_bytes(bytes)
                }
                Recipe::Constant(v) => v,
                Recipe::Sum([a, b, c]) => words[a].wrapping_add(words[b]).wrapping_add(words[c]),
                Recipe::Rotate(a, b, r) => (words[a] ^ words[b]).rotate_right(r),
                Recipe::Xor(a, b) => words[a] ^ words[b],
            };
            words.push(value);
        }
        let mut traces: Vec<_> = defs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let height = if selected.is_none_or(|s| s == i) {
                    d.preprocessed.as_ref().unwrap().height()
                } else {
                    0
                };
                RowMajorMatrix::new(vec![F::ZERO; height * d.main_width], d.main_width)
            })
            .collect();
        for t in 0..6 {
            if selected.is_some_and(|s| s < 6 && s != t) {
                continue;
            }
            for (r, slots) in self.rows[t].iter().enumerate() {
                if selected.is_none_or(|s| s == t) {
                    let row = &mut traces[t].values[r * mw(t)..(r + 1) * mw(t)];
                    for (slot, &word) in slots.iter().enumerate() {
                        for (i, b) in words[word].to_le_bytes().into_iter().enumerate() {
                            row[slot * 4 + i] = F::from_u8(b);
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
                            row[24 + i] = F::from_u16(carry);
                            let x =
                                words[slots[3]].to_le_bytes()[i] ^ words[slots[4]].to_le_bytes()[i];
                            row[28 + i] = F::from_u8(x);
                            let bits = ROT[t] % 8;
                            if bits != 0 {
                                row[32 + i] = F::from_u8(x & ((1 << bits) - 1));
                                row[36 + i] = F::from_u8(x >> bits);
                            }
                        }
                    }
                }
                // The three fixed tables use canonical byte-indexed row order.
                // Count directly from word recipes without materializing other traces.
                for i in 0..4 {
                    if t < 5 {
                        let (a, b) = if t < 4 {
                            (slots[3], slots[4])
                        } else {
                            (slots[0], slots[1])
                        };
                        let (a, b) = (words[a].to_le_bytes()[i], words[b].to_le_bytes()[i]);
                        if selected.is_none_or(|s| s == 6) {
                            traces[6].values[usize::from(a) * 256 + usize::from(b)] += F::ONE;
                        }
                        if (t == 1 || t == 3) && selected.is_none_or(|s| s == 7) {
                            let offset = if t == 1 { 0 } else { 256 };
                            traces[7].values[offset + usize::from(a ^ b)] += F::ONE;
                        }
                    } else if selected.is_none_or(|s| s == 8) {
                        traces[8].values[usize::from(words[slots[0]].to_le_bytes()[i])] += F::ONE;
                    }
                }
            }
        }
        Ok(traces)
    }
}
