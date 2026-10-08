//! Experimental complete lowering of the supported Plonkish relations to Flock.
use crate::{
    goldilocks::{GoldGate, Operation, P, Relation},
    hash_chain::Blake3Gate,
};
use flock_prover::{
    circuit::builder::{CircuitShape, CircuitWitness, ShapeBuilder, SlotId},
    field::F128,
    lincheck::LincheckCircuit,
    prover::UnionSlotProverInput,
    r1cs::BlockR1cs,
    r1cs_hashes::blake3::{build_block_r1cs, generate_witness_batch_major_partial_into},
};
use multi_stark::{
    plonkish::{Assignment, Circuit, Value},
    types::Val,
};
use p3_field::PrimeField64;
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Kind {
    Gold(Operation),
    Blake3,
}
enum WireSource {
    Source(Value),
    Constant(F128),
    Computed,
}
struct Step {
    kind: Kind,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}
#[derive(Clone, Copy)]
struct HashOutput {
    cv: [usize; 2],
    block: [usize; 4],
    counter: u64,
    len: u32,
    flags: u32,
}
pub struct Program {
    words: Vec<WireSource>,
    source: Vec<Option<usize>>,
    constants: HashMap<F128, usize>,
    steps: Vec<Step>,
    equal: Vec<(usize, usize)>,
    publics: Vec<usize>,
    count_only: bool,
    next_word: usize,
    counts: HashMap<Kind, usize>,
    equalities: usize,
    scales: HashMap<u64, usize>,
    bounds: Vec<u8>,
    kind_order: Vec<Kind>,
    known_sources: HashMap<usize, u64>,
    known_words: HashMap<usize, F128>,
    word_bounds: Vec<u8>,
    canonical: Vec<bool>,
    needs_canonical: Vec<usize>,
    coalesced: bool,
}
impl Program {
    fn packed_nu(&self) -> usize {
        let hash_log = self
            .counts
            .get(&Kind::Blake3)
            .copied()
            .unwrap_or(1)
            .next_power_of_two()
            .ilog2() as usize;
        let max_log = self
            .counts
            .values()
            .copied()
            .max()
            .unwrap_or(1)
            .next_power_of_two()
            .ilog2() as usize;
        // Avoid trading a small witness for enormous replicated base matrices.
        hash_log.max(max_log.saturating_sub(10)).max(12)
    }
    fn batches(&self, kind: Kind, nu: usize) -> Vec<(usize, usize)> {
        let capacity = 1usize << nu;
        if self.coalesced {
            let count = self.counts[&kind];
            return vec![(count.div_ceil(capacity).next_power_of_two(), count)];
        }
        let mut remaining = self.counts[&kind];
        let mut batches = Vec::new();
        while remaining >= capacity {
            let lanes = 1usize << (remaining / capacity).ilog2();
            let take = lanes * capacity;
            batches.push((lanes, take));
            remaining -= take;
        }
        if remaining != 0 {
            batches.push((1, remaining));
        }
        batches
    }
    fn word(&mut self, source: WireSource) -> usize {
        let w = self.next_word;
        self.next_word += 1;
        self.word_bounds.push(128);
        self.canonical.push(false);
        if !self.count_only {
            self.words.push(source);
        }
        w
    }
    fn connect(&mut self, a: usize, b: usize) {
        self.equalities += 1;
        if a == b {
            return;
        }
        if let (Some(av), Some(bv)) = (self.known_words.get(&a), self.known_words.get(&b)) {
            assert_eq!(av, bv, "inconsistent constant equality");
        } else if !self.count_only {
            self.equal.push((a, b));
        }
    }
    fn constant(&mut self, v: F128) -> usize {
        if let Some(&w) = self.constants.get(&v) {
            return w;
        }
        let w = self.word(WireSource::Constant(v));
        let integer = u128::from(v.lo) | (u128::from(v.hi) << 64);
        self.word_bounds[w] = (128 - integer.leading_zeros()) as u8;
        self.canonical[w] = integer < u128::from(P);
        self.constants.insert(v, w);
        self.known_words.insert(w, v);
        w
    }
    fn number(&mut self, v: u64) -> usize {
        self.constant(F128::new(v, 0))
    }
    fn source(&mut self, v: Value) -> usize {
        if let Some(w) = self.source[v.index()] {
            return w;
        }
        if let Some(&c) = self.known_sources.get(&v.index()) {
            let w = self.number(c);
            self.source[v.index()] = Some(w);
            return w;
        }
        let w = self.word(WireSource::Source(v));
        self.source[v.index()] = Some(w);
        let bits = self.bounds[v.index()] as usize;
        self.word_bounds[w] = bits as u8;
        // Smaller bounds already follow from a boolean/range constraint or
        // a bounded sum below p. Those constraints are emitted by lower().
        if bits == 64 {
            self.needs_canonical.push(w);
        }
        w
    }
    fn gate(&mut self, kind: Kind, inputs: Vec<usize>, n: usize) -> Vec<usize> {
        // These relations themselves enforce canonical input encodings. A
        // separate range gate is needed only for otherwise uncovered sources.
        let checked_inputs = match kind {
            Kind::Gold(
                Operation::Add
                | Operation::Mul
                | Operation::MulSmall(_)
                | Operation::Mask
                | Operation::Linear { .. }
                | Operation::Xor4,
            ) => 2,
            Kind::Gold(Operation::Pack(n)) if n <= 32 => 2,
            Kind::Gold(Operation::Canonical) => 1,
            Kind::Gold(Operation::Range(n)) if n < 64 => 1,
            _ => 0,
        };
        for &input in &inputs[..checked_inputs] {
            self.canonical[input] = true;
        }
        let outputs = (0..n)
            .map(|_| self.word(WireSource::Computed))
            .collect::<Vec<_>>();
        let bits = match kind {
            Kind::Blake3 => 128,
            Kind::Gold(Operation::Pack(n)) => 2 * n,
            Kind::Gold(Operation::Xor4) => 4,
            Kind::Gold(Operation::Mask) => usize::from(self.word_bounds[inputs[0]]).min(64),
            Kind::Gold(Operation::Linear {
                a,
                b,
                a_bits,
                b_bits,
            }) => {
                let max = ((1u128 << a_bits) - 1) * u128::from(a)
                    + ((1u128 << b_bits) - 1) * u128::from(b);
                (128 - max.leading_zeros()) as usize
            }
            Kind::Gold(_) => 64,
        };
        for &output in &outputs {
            self.word_bounds[output] = bits as u8;
        }
        if !self.counts.contains_key(&kind) {
            self.kind_order.push(kind);
        }
        *self.counts.entry(kind).or_insert(0) += 1;
        if !self.count_only {
            self.steps.push(Step {
                kind,
                inputs,
                outputs: outputs.clone(),
            });
        }
        outputs
    }
    fn bind_result(&mut self, value: Value, result: usize) {
        // A computed field result is already canonical. Reuse its wire unless
        // another constraint has already assigned this source a wire.
        if self.source[value.index()].is_none() && !self.known_sources.contains_key(&value.index())
        {
            self.source[value.index()] = Some(result);
        } else {
            let source = self.source(value);
            self.connect(result, source);
        }
    }
    fn gold(&mut self, op: Operation, a: usize, b: usize) -> usize {
        if op.is_assertion() {
            if let (Some(&av), Some(&bv)) = (self.known_words.get(&a), self.known_words.get(&b)) {
                Relation::new(op).evaluate_words(av, bv);
            } else {
                self.gate(Kind::Gold(op), vec![a, b], 0);
            }
            return a;
        }
        if let (Some(&av), Some(&bv)) = (self.known_words.get(&a), self.known_words.get(&b)) {
            if matches!(op, Operation::Add | Operation::Mul | Operation::MulSmall(_)) {
                assert!(av.hi == 0 && bv.hi == 0 && av.lo < P && bv.lo < P);
                if let Operation::MulSmall(n) = op {
                    assert!(u128::from(bv.lo) < (1u128 << n));
                }
            }
            let v = match op {
                Operation::Add => F128::new(
                    ((u128::from(av.lo) + u128::from(bv.lo)) % u128::from(P)) as u64,
                    0,
                ),
                Operation::Mul | Operation::MulSmall(_) => F128::new(
                    ((u128::from(av.lo) * u128::from(bv.lo)) % u128::from(P)) as u64,
                    0,
                ),
                Operation::Linear {
                    a,
                    b,
                    a_bits,
                    b_bits,
                } => {
                    assert!(av.hi == 0 && bv.hi == 0);
                    assert!(u128::from(av.lo) < (1u128 << a_bits));
                    assert!(u128::from(bv.lo) < (1u128 << b_bits));
                    let value =
                        u128::from(av.lo) * u128::from(a) + u128::from(bv.lo) * u128::from(b);
                    assert!(value < u128::from(P));
                    F128::new(value as u64, 0)
                }
                // Tiny packing/range relations; construct only when folding constants.
                _ => Relation::new(op).evaluate_words(av, bv)[2],
            };
            return self.constant(v);
        }
        self.gate(Kind::Gold(op), vec![a, b], 1)[0]
    }
    fn plus(&mut self, a: usize, b: usize) -> usize {
        let zero = self.number(0);
        if a == zero {
            b
        } else if b == zero {
            a
        } else if self.word_bounds[a] <= 32 && self.word_bounds[b] <= 32 {
            self.gold(
                Operation::Linear {
                    a: 1,
                    b: 1,
                    a_bits: self.word_bounds[a] as usize,
                    b_bits: self.word_bounds[b] as usize,
                },
                a,
                b,
            )
        } else {
            self.gold(Operation::Add, a, b)
        }
    }
    fn scale(&mut self, a: usize, k: u64) -> usize {
        if k == 1 {
            return a;
        }
        if k == 0 {
            return self.number(0);
        }
        *self.scales.entry(k).or_insert(0) += 1;
        if k <= 16 && self.word_bounds[a] <= 32 {
            let zero = self.number(0);
            return self.gold(
                Operation::Linear {
                    a: k,
                    b: 0,
                    a_bits: self.word_bounds[a] as usize,
                    b_bits: 0,
                },
                a,
                zero,
            );
        }
        let bits = (64 - k.leading_zeros()) as usize;
        let op = if bits <= 16 {
            Operation::MulSmall(bits.next_power_of_two())
        } else {
            Operation::Mul
        };
        let k = self.number(k);
        self.gold(op, a, k)
    }
    fn pack(&mut self, bytes: &[usize]) -> usize {
        assert!(bytes.len() <= 16);
        let zero = self.number(0);
        let mut words = bytes.to_vec();
        words.resize(16, zero);
        for n in [8, 16, 32, 64] {
            words = words
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| self.gold(Operation::Pack(n), p[0], p[1]))
                .collect();
        }
        words[0]
    }
    fn iv(&mut self) -> [usize; 2] {
        [
            self.constant(F128::new(0xbb67ae856a09e667, 0xa54ff53a3c6ef372)),
            self.constant(F128::new(0x9b05688c510e527f, 0x5be0cd191f83d9ab)),
        ]
    }
    fn compress(&mut self, h: HashOutput, root: bool) -> [usize; 2] {
        let params = self.constant(F128::new(
            h.counter,
            u64::from(h.len) | (u64::from(h.flags | if root { 8 } else { 0 }) << 32),
        ));
        let words = self.gate(
            Kind::Blake3,
            vec![
                h.cv[0], h.cv[1], h.block[0], h.block[1], h.block[2], h.block[3], params,
            ],
            4,
        );
        [words[0], words[1]]
    }
    fn subtree(&mut self, packed: &[usize], len: usize, counter: u64) -> HashOutput {
        let iv = self.iv();
        if len > 1024 {
            let chunks = len.div_ceil(1024);
            let left = 1usize << (usize::BITS - 1 - (chunks - 1).leading_zeros());
            let split = left * 1024;
            let lo = self.subtree(&packed[..split / 16], split, counter);
            let lo = self.compress(lo, false);
            let hi = self.subtree(&packed[split / 16..], len - split, counter + left as u64);
            let hi = self.compress(hi, false);
            return HashOutput {
                cv: iv,
                block: [lo[0], lo[1], hi[0], hi[1]],
                counter: 0,
                len: 64,
                flags: 4,
            };
        }
        let zero = self.number(0);
        let mut cv = iv;
        let n = len.div_ceil(64).max(1);
        for i in 0..n {
            let h = HashOutput {
                cv,
                block: std::array::from_fn(|j| packed.get(4 * i + j).copied().unwrap_or(zero)),
                counter,
                len: len.saturating_sub(i * 64).min(64) as u32,
                flags: if i == 0 { 1 } else { 0 } | if i + 1 == n { 2 } else { 0 },
            };
            if i + 1 == n {
                return h;
            }
            cv = self.compress(h, false);
        }
        unreachable!()
    }
    pub fn compile(circuit: &Circuit<Val>) -> Result<Self, String> {
        Self::lower(circuit, false)
    }
    pub fn census(circuit: &Circuit<Val>) -> Result<Self, String> {
        Self::lower(circuit, true)
    }
    fn lower(circuit: &Circuit<Val>, count_only: bool) -> Result<Self, String> {
        let boolean = |g: &multi_stark::plonkish::Gate<Val>| {
            let q = g.coefficients.map(|x| x.as_canonical_u64());
            g.wires[0] == g.wires[1]
                && q[0] == 1
                && q[3] == 0
                && q[4] == 0
                && (u128::from(q[1]) + u128::from(q[2])) % u128::from(P) == u128::from(P - 1)
        };
        let mut bounds = vec![64u8; circuit.num_values()];
        let mut known_sources = HashMap::new();
        for g in circuit.gates() {
            if boolean(g) {
                bounds[g.wires[0].index()] = 1;
            }
            let q = g.coefficients.map(|x| x.as_canonical_u64());
            if q[..4] == [0, 1, 0, 0] {
                let v = if q[4] == 0 { 0 } else { P - q[4] };
                bounds[g.wires[0].index()] =
                    bounds[g.wires[0].index()].min((64 - v.leading_zeros()) as u8);
                if known_sources
                    .insert(g.wires[0].index(), v)
                    .is_some_and(|old| old != v)
                {
                    return Err("inconsistent constant constraints".into());
                }
            }
        }
        for l in circuit.lookups() {
            let rows = circuit.tables()[l.table.index()].rows();
            if rows.len().is_power_of_two()
                && rows.len() <= 65536
                && rows
                    .iter()
                    .enumerate()
                    .all(|(i, r)| r.len() == 1 && r[0].as_canonical_u64() == i as u64)
            {
                bounds[l.values[0].index()] =
                    bounds[l.values[0].index()].min(rows.len().ilog2() as u8);
            }
        }
        // Integer sums below p cannot wrap. Their field equality therefore
        // propagates a range bound to the canonical output as well.
        for g in circuit.gates() {
            let q = g.coefficients.map(|x| x.as_canonical_u64());
            let [a, b, c] = g.wires.map(|v| v.index());
            if q[0] == 0 && q[3] == P - 1 && q[4] == 0 && q[1] <= 16 && q[2] <= 16 {
                let max = ((1u128 << bounds[a]) - 1) * u128::from(q[1])
                    + ((1u128 << bounds[b]) - 1) * u128::from(q[2]);
                if max < u128::from(P) {
                    bounds[c] = bounds[c].min((128 - max.leading_zeros()) as u8);
                }
            }
        }
        let mut p = Self {
            words: vec![],
            source: vec![None; circuit.num_values()],
            constants: HashMap::new(),
            steps: vec![],
            equal: vec![],
            publics: vec![],
            count_only,
            next_word: 0,
            counts: HashMap::new(),
            equalities: 0,
            scales: HashMap::new(),
            bounds,
            kind_order: vec![],
            known_sources,
            known_words: HashMap::new(),
            word_bounds: vec![],
            canonical: vec![],
            needs_canonical: vec![],
            coalesced: false,
        };
        let zero = p.number(0);
        for g in circuit.gates() {
            if boolean(g) {
                let a = p.source(g.wires[0]);
                p.gold(Operation::Range(1), a, zero);
                continue;
            }
            let q = g.coefficients.map(|x| x.as_canonical_u64());
            // Zero-coefficient wires do not participate in the relation.
            let a = if q[0] != 0 || q[1] != 0 {
                p.source(g.wires[0])
            } else {
                zero
            };
            let b = if q[0] != 0 || q[2] != 0 {
                p.source(g.wires[1])
            } else {
                zero
            };
            let ba = p.bounds[g.wires[0].index()] as usize;
            let bb = p.bounds[g.wires[1].index()] as usize;
            if q[0] == 0
                && q[3] == P - 1
                && q[4] == 0
                && q[1] <= 16
                && q[2] <= 16
                && ba <= 32
                && bb <= 32
                && q[1] != 0
                && q[2] != 0
            {
                let result = p.gold(
                    Operation::Linear {
                        a: q[1],
                        b: q[2],
                        a_bits: ba,
                        b_bits: bb,
                    },
                    a,
                    b,
                );
                p.bind_result(g.wires[2], result);
                continue;
            }
            let product = if q[0] == 0 {
                zero
            } else if ba <= 1 {
                p.gold(Operation::Mask, b, a)
            } else if bb <= 1 {
                p.gold(Operation::Mask, a, b)
            } else if ba <= 16 {
                p.gold(Operation::MulSmall(ba.next_power_of_two()), b, a)
            } else if bb <= 16 {
                p.gold(Operation::MulSmall(bb.next_power_of_two()), a, b)
            } else {
                p.gold(Operation::Mul, a, b)
            };
            if q[3] == P - 1 && [q[0], q[1], q[2], q[4]].iter().all(|&k| k <= P / 2) {
                let mut result = p.number(q[4]);
                for (k, v) in [q[0], q[1], q[2]].into_iter().zip([product, a, b]) {
                    let term = p.scale(v, k);
                    result = p.plus(result, term);
                }
                p.bind_result(g.wires[2], result);
                continue;
            }
            let c = if q[3] != 0 {
                p.source(g.wires[2])
            } else {
                zero
            };
            let mut positive = zero;
            let mut negative = zero;
            let one = p.number(1);
            for (k, v) in q.into_iter().zip([product, a, b, c, one]) {
                if k == 0 {
                    continue;
                }
                if k > P / 2 {
                    let term = p.scale(v, P - k);
                    negative = p.plus(negative, term);
                } else {
                    let term = p.scale(v, k);
                    positive = p.plus(positive, term);
                }
            }
            p.connect(positive, negative);
        }
        for lookup in circuit.lookups() {
            let rows = circuit.tables()[lookup.table.index()].rows();
            let values: Vec<_> = lookup.values.iter().map(|&v| p.source(v)).collect();
            let is_row = |r: &Vec<Val>, e: [u64; 3]| {
                r.len() == 3 && r.iter().zip(e).all(|(v, e)| v.as_canonical_u64() == e)
            };
            if rows.len().is_power_of_two()
                && rows.len() <= 65536
                && rows
                    .iter()
                    .enumerate()
                    .all(|(i, r)| r.len() == 1 && r[0].as_canonical_u64() == i as u64)
            {
                p.gold(
                    Operation::Range(rows.len().ilog2() as usize),
                    values[0],
                    zero,
                );
            } else if rows.len() == 256
                && rows.iter().enumerate().all(|(i, r)| {
                    is_row(
                        r,
                        [
                            (i / 16) as u64,
                            (i % 16) as u64,
                            ((i / 16) ^ (i % 16)) as u64,
                        ],
                    )
                })
            {
                let out = p.gold(Operation::Xor4, values[0], values[1]);
                p.connect(out, values[2]);
            } else if rows.len() == 16
                && rows
                    .iter()
                    .enumerate()
                    .all(|(i, r)| is_row(r, [i as u64, (i & 7) as u64, (i >> 3) as u64]))
            {
                p.gold(Operation::Range(1), values[2], zero);
                let out = p.gold(Operation::Pack(3), values[1], values[2]);
                p.connect(out, values[0]);
            } else {
                return Err(format!("unsupported table {}", lookup.table.index()));
            }
        }
        for (input, output) in circuit.blake3_calls() {
            let bytes: Vec<_> = input.iter().map(|&v| p.source(v)).collect();
            let packed: Vec<_> = bytes.chunks(16).map(|chunk| p.pack(chunk)).collect();
            let h = p.subtree(&packed, input.len(), 0);
            let digest = p.compress(h, true);
            for (i, chunk) in output.as_chunks::<16>().0.iter().enumerate() {
                let bytes: Vec<_> = chunk.iter().map(|&v| p.source(v)).collect();
                let word = p.pack(&bytes);
                p.connect(digest[i], word);
            }
        }
        for &v in circuit.public_values() {
            let w = p.source(v);
            p.publics.push(w);
        }
        for word in std::mem::take(&mut p.needs_canonical) {
            if !p.canonical[word] {
                p.gold(Operation::Canonical, word, zero);
            }
        }
        Ok(p)
    }
    pub fn stats(&self) -> serde_json::Value {
        let counts: std::collections::BTreeMap<_, _> = self
            .counts
            .iter()
            .map(|(k, &v)| (format!("{k:?}"), v))
            .collect();
        let mut scales: Vec<_> = self.scales.iter().map(|(&k, &n)| (k, n)).collect();
        scales.sort_by_key(|&(k, n)| (std::cmp::Reverse(n), k));
        scales.truncate(20);
        let operation_costs: std::collections::BTreeMap<_, _> = self
            .counts
            .iter()
            .map(|(kind, &n)| {
                let bits = match kind {
                    Kind::Gold(op) => Relation::new(*op).useful_bits().div_ceil(128) * 128,
                    Kind::Blake3 => {
                        flock_prover::r1cs_hashes::blake3::USEFUL_BITS.div_ceil(128) * 128
                    }
                };
                (
                    format!("{kind:?}"),
                    (n, bits, (bits as u128) * (n as u128) / 8),
                )
            })
            .collect();
        let packed_bytes: u128 = operation_costs.values().map(|&(_, _, bytes)| bytes).sum();
        let packed_bits = packed_bytes * 8;
        let operation_costs: std::collections::BTreeMap<_, _> = operation_costs
            .into_iter()
            .map(|(kind, (instances, bits, bytes))| {
                (
                    kind,
                    serde_json::json!({
                        "instances": instances,
                        "bits_per_instance": bits,
                        "live_witness_bytes": bytes,
                    }),
                )
            })
            .collect();
        serde_json::json!({"words":self.next_word,"steps":self.counts.values().sum::<usize>(),"tables":counts,"equalities":self.equalities,"publics":self.publics.len(),
            "packed_witness_bytes_before_padding":packed_bytes,"operation_costs":operation_costs,"dense_m_lower_bound":(packed_bits.max(2)-1).ilog2()+1,
            "constant_multiplications":self.scales.values().sum::<usize>(),"most_frequent_scalars":scales,
            "memory_scope":"One packed witness only; excludes a/b, PCS, wiring, frontend and padding"})
    }
    pub fn geometry(&self) -> serde_json::Value {
        self.layout_geometry(false)
    }
    pub fn packed_geometry(&self) -> serde_json::Value {
        self.layout_geometry(true)
    }
    pub fn coalesced_geometry(&mut self) -> serde_json::Value {
        let original = self.coalesced;
        self.coalesced = true;
        let geometry = self.packed_geometry();
        self.coalesced = original;
        geometry
    }
    pub fn coalesce(&mut self) {
        self.coalesced = true;
    }
    fn layout_geometry(&self, packed: bool) -> serde_json::Value {
        use flock_prover::{
            circuit::CellSpace,
            pcs::ligerito::{LigeritoProfile, embedded_security_config},
            schedule::{Registry, TableType},
            union::UnionInstance,
        };
        let nu = if packed {
            self.packed_nu()
        } else {
            self.counts
                .values()
                .copied()
                .max()
                .unwrap_or(1)
                .max(8)
                .next_power_of_two()
                .ilog2() as usize
        };
        let mut types: Vec<_> = self
            .kind_order
            .iter()
            .flat_map(|kind| {
                let batches = if packed {
                    self.batches(*kind, nu)
                } else {
                    vec![(1, self.counts[kind])]
                };
                batches.into_iter().map(move |(lanes, logical_count)| {
                    let table = match kind {
                        Kind::Gold(op) => {
                            use flock_prover::circuit::builder::GateType;
                            let relation = Arc::new(Relation::new(*op));
                            if packed {
                                crate::packed::PackedGoldGate {
                                    block: Arc::new(crate::packed::block(&relation, lanes, nu)),
                                    relation,
                                    lanes,
                                }
                                .table()
                            } else {
                                GoldGate { relation, nu }.table()
                            }
                        }
                        Kind::Blake3 => TableType::from_block_r1cs(&build_block_r1cs(nu))
                            .with_io_schema(flock_prover::r1cs_hashes::blake3::io_schema()),
                    };
                    (table, logical_count.div_ceil(lanes))
                })
            })
            .collect();
        types.sort_by_key(|(t, _)| std::cmp::Reverse(t.k_log));
        let (types, counts): (Vec<_>, Vec<_>) = types.into_iter().unzip();
        let registry = Registry::new(types, nu);
        let union = UnionInstance::new(&registry, counts);
        let cells = CellSpace::new(&registry, self.constants.len() + self.publics.len());
        let padded = (union.packed_len() as u128) * 16;
        let mut pcs_buffers = serde_json::Map::new();
        for profile in [LigeritoProfile::Fast, LigeritoProfile::Slim] {
            if let Some(batch) =
                flock_prover::pcs::ligerito::embedded_initial_k(union.dense_m(), profile)
            {
                let params = flock_prover::pcs::PcsParams {
                    m: union.dense_m(),
                    log_inv_rate: profile.log_inv_rate(),
                    log_batch_size: batch,
                    profile,
                    num_lanes: union.commit_lanes(batch),
                    merkle_hash: flock_prover::merkle::HashKind::Blake3,
                };
                // These buffers survive commitment while the PIOPs run.
                pcs_buffers.insert(
                    profile.as_str().into(),
                    serde_json::json!({
                        "codeword_bytes": params.codeword_len_f128() as u128 * 16,
                        "merkle_tree_bytes": 2 * params.n_leaves() as u128 * 32,
                        "dense_message_allocation_bytes": union.committed_words() as u128 * 16,
                    }),
                );
            }
        }
        serde_json::json!({"packed_rows":packed,"uniform_row_log":nu,"dense_m":union.dense_m(),
            "table_slots":registry.num_types(), "coalesced":packed && self.coalesced,
            "strict_slim_profile_available":embedded_security_config(union.dense_m(),LigeritoProfile::Slim).is_some(),
            "live_packed_witness_bytes":(union.dense_words() as u128)*16,
            "padded_witness_array_bytes":padded,"three_padded_witness_arrays_bytes":3*padded,
            "wiring_cell_log":cells.mu(),"one_wiring_field_array_bytes":(1u128<<cells.mu())*16,
            "pcs_buffers":pcs_buffers,
            "memory_scope":"Named array sizes, not a peak-RAM prediction; excludes frontend, layout, PCS and other scratch"})
    }
    pub fn build_packed(&self, assignment: &Assignment<Val>) -> Result<Built, String> {
        use crate::packed::PackedGoldGate;
        use flock_prover::circuit::builder::GateType;
        if self.count_only {
            return Err("cannot build a counting-only program".into());
        }
        let start = std::time::Instant::now();
        let trace = |phase| {
            if std::env::var_os("PCS_TRACE").is_some() {
                eprintln!(
                    "  [packed shape] {phase}: {:.1}s",
                    start.elapsed().as_secs_f64()
                );
            }
        };
        let nu = self.packed_nu();
        let relations: HashMap<_, _> = self
            .kind_order
            .iter()
            .filter_map(|&kind| match kind {
                Kind::Gold(op) => Some((kind, Arc::new(Relation::new(op)))),
                _ => None,
            })
            .collect();
        let mut values = Vec::with_capacity(self.words.len());
        for word in &self.words {
            values.push(match word {
                WireSource::Source(v) => F128::new(
                    assignment
                        .value(*v)
                        .map_err(|e| e.to_string())?
                        .as_canonical_u64(),
                    0,
                ),
                WireSource::Constant(v) => *v,
                WireSource::Computed => F128::ZERO,
            });
        }
        let mut grouped: HashMap<Kind, Vec<&Step>> = HashMap::new();
        for step in &self.steps {
            let inputs: Vec<_> = step.inputs.iter().map(|&w| values[w]).collect();
            let mut outputs = Vec::new();
            match step.kind {
                Kind::Gold(op) => {
                    let row = relations[&step.kind].evaluate_words(inputs[0], inputs[1]);
                    if !op.is_assertion() {
                        outputs.push(row[2]);
                    }
                }
                Kind::Blake3 => {
                    Blake3Gate { nu }.eval(&inputs, &(), &mut outputs);
                }
            }
            for (&word, output) in step.outputs.iter().zip(outputs) {
                values[word] = output;
            }
            grouped.entry(step.kind).or_default().push(step);
        }
        let mut builder = ShapeBuilder::new(nu);
        trace("values evaluated");
        let words: Vec<_> = self
            .words
            .iter()
            .map(|source| match source {
                WireSource::Constant(v) => builder.fixed_public_input(*v),
                _ => builder.input(),
            })
            .collect();
        let zero = words[self.constants[&F128::ZERO]];
        let mut tables = Vec::new();
        for &kind in &self.kind_order {
            match kind {
                Kind::Gold(_) => {
                    let relation = relations[&kind].clone();
                    let mut offset = 0;
                    for (lanes, take) in self.batches(kind, nu) {
                        let block = Arc::new(crate::packed::block(&relation, lanes, nu));
                        let slot = builder.slot(PackedGoldGate {
                            relation: relation.clone(),
                            block: block.clone(),
                            lanes,
                        });
                        for chunk in grouped[&kind][offset..offset + take].chunks(lanes) {
                            let arity = if relation.operation.is_assertion() {
                                1
                            } else {
                                3
                            };
                            let mut inputs = vec![zero; lanes * arity];
                            for (i, step) in chunk.iter().enumerate() {
                                inputs[arity * i] = words[step.inputs[0]];
                                if arity == 3 {
                                    inputs[arity * i + 1] = words[step.inputs[1]];
                                    inputs[arity * i + 2] = words[step.outputs[0]];
                                }
                            }
                            builder.gate(slot, &inputs);
                        }
                        tables.push((slot, (*block).clone(), Some(relation.clone()), lanes));
                        offset += take;
                    }
                    assert_eq!(offset, grouped[&kind].len());
                }
                Kind::Blake3 => {
                    let slot = builder.slot(crate::packed::AdviceBlake3Gate { nu });
                    for step in &grouped[&kind] {
                        let inputs: Vec<_> = step
                            .inputs
                            .iter()
                            .chain(&step.outputs)
                            .map(|&w| words[w])
                            .collect();
                        builder.gate(slot, &inputs);
                    }
                    tables.push((slot, build_block_r1cs(nu), None, 0));
                }
            }
        }
        trace("gates allocated");
        for &(a, b) in &self.equal {
            builder.connect(words[a], words[b]);
        }
        for &word in &self.publics {
            builder.publish(words[word]);
        }
        trace("wires connected");
        let shape = builder
            .finish()
            .map_err(|e| format!("packed shape: {e:?}"))?;
        tables.sort_by_key(|(slot, _, _, _)| shape.registry_slot(*slot));
        trace("shape finalized");
        let witness = shape.run(&values, &[]);
        trace("rows evaluated");
        Ok(Built {
            shape,
            witness,
            tables,
            nu,
        })
    }
}
pub struct Built {
    pub shape: CircuitShape,
    pub witness: CircuitWitness,
    tables: Vec<(SlotId, BlockR1cs, Option<Arc<Relation>>, usize)>,
    nu: usize,
}
impl Built {
    pub fn into_ready(mut self) -> Ready {
        let rows = self
            .tables
            .iter()
            .map(|(slot, _, relation, _)| {
                if relation.is_some() {
                    Rows::Gold(self.witness.take_rows_of::<Vec<[F128; 4]>>(*slot))
                } else {
                    Rows::Hash(
                        self.witness
                            .take_rows_of::<flock_prover::r1cs_hashes::blake3::Compression>(*slot),
                    )
                }
            })
            .collect();
        Ready {
            registry: self.shape.registry,
            circuit: self.shape.circuit,
            counts: self.shape.counts,
            public: self.witness.public,
            tables: self.tables,
            nu: self.nu,
            rows,
        }
    }
}

pub enum Rows {
    Gold(Vec<Vec<[F128; 4]>>),
    Hash(Vec<flock_prover::r1cs_hashes::blake3::Compression>),
}
pub struct Ready {
    pub registry: flock_prover::schedule::Registry,
    pub circuit: flock_prover::circuit::Circuit,
    pub counts: Vec<usize>,
    pub public: Vec<F128>,
    tables: Vec<(SlotId, BlockR1cs, Option<Arc<Relation>>, usize)>,
    nu: usize,
    rows: Vec<Rows>,
}
impl Ready {
    pub fn take_rows(&mut self) -> Vec<Rows> {
        std::mem::take(&mut self.rows)
    }
    pub fn circuits(&self) -> Vec<&dyn LincheckCircuit> {
        self.tables
            .iter()
            .map(|(_, block, _, _)| block.csc_lincheck_circuit() as &dyn LincheckCircuit)
            .collect()
    }
    pub fn prove(
        &self,
        rows: Vec<Rows>,
        params: &flock_prover::pcs::PcsParams,
        domain: &[u8],
    ) -> (
        flock_prover::proof::R1csProofCircuitMerged,
        flock_prover::pcs::Commitment,
    ) {
        self.prove_with_challenger(
            rows,
            params,
            &mut flock_prover::challenger::FsChallenger::new(domain),
        )
    }
    pub fn prove_with_challenger(
        &self,
        rows: Vec<Rows>,
        params: &flock_prover::pcs::PcsParams,
        challenger: &mut impl flock_prover::challenger::Challenger,
    ) -> (
        flock_prover::proof::R1csProofCircuitMerged,
        flock_prover::pcs::Commitment,
    ) {
        assert_eq!(rows.len(), self.tables.len());
        let slots = self
            .tables
            .iter()
            .zip(rows)
            .map(|((_, block, relation, lanes), rows)| {
                let nu = self.nu;
                let lanes = *lanes;
                let relation = relation.clone();
                UnionSlotProverInput::in_place(
                    move |dst| {
                        let start = std::time::Instant::now();
                        let trace = std::env::var_os("PCS_TRACE").is_some();
                        if trace {
                            eprintln!(
                                "  [witness] start {:?}, {lanes} lanes",
                                relation.as_ref().map(|r| r.operation)
                            );
                        }
                        let stripe = match rows {
                            Rows::Gold(rows) => crate::packed::witness_into(
                                relation.as_ref().unwrap(),
                                lanes,
                                &rows,
                                nu,
                                dst,
                            ),
                            Rows::Hash(rows) => {
                                generate_witness_batch_major_partial_into(&rows, nu, dst)
                            }
                        };
                        if trace {
                            eprintln!("  [witness] done in {:.1}s", start.elapsed().as_secs_f64());
                        }
                        stripe
                    },
                    block.csc_lincheck_circuit(),
                )
            })
            .collect();
        let union = flock_prover::union::UnionInstance::new(&self.registry, self.counts.clone());
        let (proof, commitment, _) = flock_prover::prover::prove_fast_ligerito_union_circuit(
            &union,
            &self.circuit,
            &self.public,
            params,
            slots,
            vec![],
            challenger,
        );
        (proof, commitment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use multi_stark::plonkish::{
        CircuitBuilder,
        gadgets::{ByteGadgets, blake3 as hash_gadget},
    };
    use p3_field::PrimeCharacteristicRing;
    #[test]
    fn coalesced_packing_preserves_partial_lanes_and_public_outputs() {
        let mut b = CircuitBuilder::<Val>::new();
        let x = b.input("x");
        let mut out = x;
        for _ in 0..9001 {
            out = b.mul(x, x);
        }
        b.expose_public(out);
        let circuit = b.finish();
        let mut program = Program::compile(&circuit).unwrap();
        let kind = Kind::Gold(Operation::Mul);
        assert_eq!(
            program.batches(kind, program.packed_nu()),
            [(2, 8192), (1, 809)]
        );
        program.coalesce();
        assert_eq!(program.batches(kind, program.packed_nu()), [(4, 9001)]);
        let mut w = circuit.witness();
        w.set(x, Val::from_u8(7)).unwrap();
        let assignment = w.generate().unwrap();
        let built = program.build_packed(&assignment).unwrap();
        assert_eq!(built.witness.public.last(), Some(&F128::new(49, 0)));
        let mut ready = built.into_ready();
        let rows = ready.take_rows();
        let Rows::Gold(rows) = &rows[0] else {
            panic!("expected multiplication rows")
        };
        assert_eq!(rows.len(), 2251);
        let relation = Relation::new(Operation::Mul);
        for (lane, row) in rows.last().unwrap().iter().enumerate() {
            assert_eq!(
                row[2],
                if lane == 0 {
                    F128::new(49, 0)
                } else {
                    F128::ZERO
                }
            );
            assert!(relation.satisfied(&relation.row(*row)));
        }
    }
    #[test]
    fn canonical_checks_cover_sources_without_arithmetic_consumers() {
        let mut b = CircuitBuilder::<Val>::new();
        let x = b.input("x");
        let y = b.input("y");
        let free = b.public_input("otherwise unconstrained");
        let product = b.mul(x, y);
        b.expose_public(product);
        let circuit = b.finish();
        let program = Program::compile(&circuit).unwrap();
        assert_eq!(program.stats(), Program::census(&circuit).unwrap().stats());
        let standalone: Vec<_> = program
            .steps
            .iter()
            .filter(|s| s.kind == Kind::Gold(Operation::Canonical))
            .collect();
        assert_eq!(standalone.len(), 1);
        assert_eq!(
            standalone[0].inputs[0],
            program.source[free.index()].unwrap()
        );
        for value in [0, 1, P - 1] {
            let mut w = circuit.witness();
            for input in [x, y, free] {
                w.set(input, Val::from_u64(value)).unwrap();
            }
            let assignment = w.generate().unwrap();
            let built = program.build_packed(&assignment).unwrap();
            assert_eq!(
                built.witness.public[built.witness.public.len() - 2],
                F128::new(value, 0)
            );
        }
    }
    #[test]
    fn computed_results_reuse_wires_but_preserve_existing_constraints() {
        let mut b = CircuitBuilder::<Val>::new();
        let x = b.input("x");
        let y = b.input("y");
        let sum = b.add(x, y);
        let product = b.mul(sum, x);
        // This output already has a wire before its defining relation.
        let reused = b.input("reused");
        b.assert_equal(reused, product);
        b.constrain_gate(
            [sum, x, reused],
            [Val::ONE, Val::ZERO, Val::ZERO, Val::NEG_ONE, Val::ZERO],
        );
        // A constant output must stay pinned, even if its source is not yet used.
        let fixed = b.constant(Val::from_u8(3));
        b.constrain_gate(
            [x, y, fixed],
            [Val::ZERO, Val::ONE, Val::ONE, Val::NEG_ONE, Val::ZERO],
        );
        for v in [sum, product, reused, fixed] {
            b.expose_public(v);
        }
        let circuit = b.finish();
        let program = Program::compile(&circuit).unwrap();
        assert_eq!(program.stats(), Program::census(&circuit).unwrap().stats());
        for value in [sum, product] {
            assert!(matches!(
                program.words[program.source[value.index()].unwrap()],
                WireSource::Computed
            ));
        }
        assert!(matches!(
            program.words[program.source[reused.index()].unwrap()],
            WireSource::Source(_)
        ));
        for (xv, yv) in [(1, 2), (2, 1), (P - 1, 4)] {
            let product_value = Val::from_u64(xv) * Val::from_u8(3);
            let mut w = circuit.witness();
            w.set(x, Val::from_u64(xv)).unwrap();
            w.set(y, Val::from_u64(yv)).unwrap();
            w.set(reused, product_value).unwrap();
            let assignment = w.generate().unwrap();
            let built = program.build_packed(&assignment).unwrap();
            let public = &built.witness.public[built.witness.public.len() - 4..];
            assert_eq!(
                public,
                [
                    F128::new(3, 0),
                    F128::new(product_value.as_canonical_u64(), 0),
                    F128::new(product_value.as_canonical_u64(), 0),
                    F128::new(3, 0)
                ]
            );
        }
    }
    #[test]
    fn shared_constant_joins_preserve_shape_and_check_all_inputs() {
        let build = |reverse| {
            let mut builder = ShapeBuilder::new(10);
            let fixed = builder.fixed_public_input(F128::ONE);
            let slot = builder.slot(GoldGate {
                relation: Arc::new(Relation::new(Operation::Range(1))),
                nu: 10,
            });
            let wires: Vec<_> = (0..1024).map(|_| builder.input()).collect();
            for &wire in &wires {
                builder.gate(slot, &[wire]);
            }
            for &wire in &wires {
                if reverse {
                    builder.connect(fixed, wire);
                } else {
                    builder.connect(wire, fixed);
                }
            }
            builder.publish(wires[0]);
            builder.finish().unwrap()
        };
        let a = build(false);
        let b = build(true);
        assert_eq!(a.circuit.digest(), b.circuit.digest());
        let mut values = vec![F128::ONE; 1025];
        assert_eq!(a.run(&values, &[]).public, b.run(&values, &[]).public);
        values[513] = F128::ZERO;
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { a.run(&values, &[]) }))
                .is_err()
        );
    }
    #[test]
    fn hashing_boundaries_and_witness_independent_shape() {
        for len in [0, 1, 15, 16, 63, 64, 65, 1023, 1024, 1025, 2050] {
            let mut b = CircuitBuilder::<Val>::new();
            b.enable_compact_blake3();
            let bytes = ByteGadgets::new(&mut b);
            let inputs: Vec<_> = (0..len).map(|_| bytes.input(&mut b, "message")).collect();
            let digest = hash_gadget(&mut b, &bytes, &inputs);
            for d in digest {
                b.expose_public(d.value());
            }
            let circuit = b.finish();
            let program = Program::compile(&circuit).unwrap();
            assert_eq!(program.stats(), Program::census(&circuit).unwrap().stats());
            let mut previous = None;
            for offset in [0u8, 91] {
                let values: Vec<_> = (0..len).map(|i| (i as u8).wrapping_add(offset)).collect();
                let mut witness = circuit.witness();
                for (&v, &x) in inputs.iter().zip(&values) {
                    witness.set(v.value(), Val::from_u8(x)).unwrap();
                }
                let assignment = witness.generate().unwrap();
                let expected: Vec<_> = blake3::hash(&values)
                    .as_bytes()
                    .iter()
                    .map(|&x| F128::new(u64::from(x), 0))
                    .collect();
                let built = program.build_packed(&assignment).unwrap();
                let public = &built.witness.public[built.witness.public.len() - 32..];
                assert_eq!(public, expected, "length {len}");
                let digest = built.shape.circuit.digest();
                if let Some(previous) = previous {
                    assert_eq!(previous, digest);
                }
                previous = Some(digest);
            }
        }
    }
    #[test]
    fn table_contents_are_checked_and_bad_constants_rejected() {
        let mut b = CircuitBuilder::<Val>::new();
        let a = b.input("a");
        let table = b.fixed_table("nibble", vec![vec![Val::ZERO], vec![Val::from_u8(7)]]);
        b.lookup(table, &[a]);
        assert!(Program::compile(&b.finish()).is_err());
        let mut b = CircuitBuilder::<Val>::new();
        let a = b.input("a");
        for value in [3, 4] {
            b.constrain_gate(
                [a; 3],
                [
                    Val::ZERO,
                    Val::ONE,
                    Val::ZERO,
                    Val::ZERO,
                    -Val::from_u8(value),
                ],
            );
        }
        assert!(Program::compile(&b.finish()).is_err());
    }
}
