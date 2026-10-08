//! Boolean relations for canonical Goldilocks addition and multiplication.
use flock_prover::{
    circuit::builder::{GateType, SlotWitness},
    field::F128,
    r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
    schedule::{IoWord, TableType},
};
use std::sync::OnceLock;

pub const P: u64 = 0xffff_ffff_0000_0001;
const ONE: usize = 512;
const ZERO: usize = 513;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    Add,
    Mul,
    MulSmall(usize),
    Mask,
    Pack(usize),
    Range(usize),
    Xor4,
    Canonical,
    #[cfg(test)]
    Equal,
    Linear {
        a: u64,
        b: u64,
        a_bits: usize,
        b_bits: usize,
    },
}
impl Operation {
    pub fn is_assertion(self) -> bool {
        matches!(self, Self::Range(_) | Self::Canonical)
    }
}
#[derive(Clone)]
enum Recipe {
    Input(usize),
    One,
    Zero,
    Xor(usize, usize),
    And(usize, usize),
    Product(Vec<usize>, Vec<usize>),
    Check,
}

pub struct Relation {
    recipes: Vec<Recipe>,
    a: Vec<Vec<usize>>,
    b: Vec<Vec<usize>>,
    pub operation: Operation,
}

pub struct GoldGate {
    pub relation: std::sync::Arc<Relation>,
    pub nu: usize,
}
impl GateType for GoldGate {
    type Row = [F128; 4];
    type Hint = ();
    fn table(&self) -> TableType {
        let schema = if self.relation.operation.is_assertion() {
            vec![IoWord::input(0)]
        } else {
            vec![IoWord::input(0), IoWord::input(1), IoWord::output(2)]
        };
        TableType::from_block_r1cs(&self.relation.block(self.nu)).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let row = self
            .relation
            .evaluate_words(inputs[0], inputs.get(1).copied().unwrap_or(F128::ZERO));
        if !self.relation.operation.is_assertion() {
            outputs.push(row[2]);
        }
        row
    }
    fn witness(&self, _: &[Self::Row], _: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

pub fn packed_witness(
    relation: &Relation,
    rows: &[[F128; 4]],
    nu: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    let block = relation.block(nu);
    let words = 1usize << (block.k_log - 7);
    let capacity = 1usize << nu;
    assert!(rows.len() <= capacity);
    let mut z = vec![F128::ZERO; words * capacity];
    for (j, &row) in rows.iter().enumerate() {
        let bits = relation.row(row);
        assert!(relation.satisfied(&bits));
        for (i, v) in bits.into_iter().enumerate() {
            if v {
                let w = &mut z[j * words + i / 128];
                if i % 128 < 64 {
                    w.lo |= 1u64 << (i % 64);
                } else {
                    w.hi |= 1u64 << (i % 64);
                }
            }
        }
    }
    let a = block.apply_a_packed(&z);
    let b = block.apply_b_packed(&z);
    let stripe = flock_prover::lincheck::pack_z_lincheck_from_packed(&z, block.m, block.k_log);
    let transpose = |row_major: Vec<F128>| {
        let mut batch = vec![F128::ZERO; row_major.len()];
        for j in 0..capacity {
            for c in 0..words {
                batch[c * capacity + j] = row_major[j * words + c];
            }
        }
        batch
    };
    (transpose(z), transpose(a), transpose(b), stripe)
}

#[cfg(test)]
pub fn packed_witness_into(
    relation: &Relation,
    rows: &[[F128; 4]],
    nu: usize,
    dst: flock_prover::union::SlotWitnessDest<'_>,
) -> Vec<u8> {
    let bits_per_row = relation.useful_bits().next_power_of_two();
    let capacity = 1usize << nu;
    let words = bits_per_row / 128;
    assert!(nu >= 3 && rows.len() <= capacity);
    assert_eq!(dst.z.len(), words * capacity);
    assert_eq!(dst.a.len(), dst.z.len());
    assert_eq!(dst.b.len(), dst.z.len());
    if !dst.elide_padding_writes {
        dst.z.fill(F128::ZERO);
        dst.a.fill(F128::ZERO);
        dst.b.fill(F128::ZERO);
    }
    let mut stripe = vec![0u8; bits_per_row * capacity / 8];
    for (row_index, &row) in rows.iter().enumerate() {
        let z = relation.row(row);
        assert!(relation.satisfied(&z));
        for word in 0..relation.useful_bits().div_ceil(128) {
            let mut packed = [0u128; 3];
            for bit in word * 128..((word + 1) * 128).min(relation.useful_bits()) {
                let av = relation.a[bit].iter().fold(false, |v, &i| v ^ z[i]);
                let bv = relation.b[bit].iter().fold(false, |v, &i| v ^ z[i]);
                for (p, v) in packed.iter_mut().zip([z[bit], av, bv]) {
                    *p |= u128::from(v) << (bit % 128);
                }
                stripe[(row_index / 8) * bits_per_row + bit] |= u8::from(z[bit]) << (row_index % 8);
            }
            let [zv, av, bv] = packed.map(|v| F128::new(v as u64, (v >> 64) as u64));
            let address = word * capacity + row_index;
            dst.z[address] = zv;
            dst.a[address] = av;
            dst.b[address] = bv;
        }
    }
    stripe
}

impl Relation {
    fn xor(&mut self, a: usize, b: usize) -> usize {
        if a == b {
            return ZERO;
        }
        if a == ZERO {
            return b;
        }
        if b == ZERO {
            return a;
        }
        let i = self.recipes.len();
        self.recipes.push(Recipe::Xor(a, b));
        self.a.push(vec![a, b]);
        self.b.push(vec![ONE]);
        i
    }
    fn and(&mut self, a: usize, b: usize) -> usize {
        if a == ZERO || b == ZERO {
            return ZERO;
        }
        if a == ONE {
            return b;
        }
        if b == ONE || a == b {
            return a;
        }
        let i = self.recipes.len();
        self.recipes.push(Recipe::And(a, b));
        self.a.push(vec![a]);
        self.b.push(vec![b]);
        i
    }
    fn equal(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        // z = (a + b + z) * 1 enforces a=b; z is assigned zero.
        let i = self.recipes.len();
        self.recipes.push(Recipe::Check);
        self.a.push(vec![a, b, i]);
        self.b.push(vec![ONE]);
    }
    fn full_add(&mut self, a: usize, b: usize, c: usize) -> (usize, usize) {
        let x = self.xor(a, b);
        let sum = self.xor(x, c);
        // For bits: majority(a,b,c) = (a+b)(a+c)+a in characteristic two.
        let y = self.xor(a, c);
        let product = self.and(x, y);
        (sum, self.xor(product, a))
    }
    fn add(&mut self, a: &[usize], b: &[usize]) -> Vec<usize> {
        assert_eq!(a.len(), b.len());
        let mut carry = ZERO;
        let mut out = Vec::with_capacity(a.len() + 1);
        for (&a, &b) in a.iter().zip(b) {
            let (sum, next) = self.full_add(a, b, carry);
            out.push(sum);
            carry = next;
        }
        out.push(carry);
        out
    }
    fn sub(&mut self, a: &[usize], b: &[usize]) -> Vec<usize> {
        let mut borrow = ZERO;
        let mut out = Vec::with_capacity(a.len());
        for (&a, &b) in a.iter().zip(b) {
            let na = self.xor(a, ONE);
            let (complement, next) = self.full_add(na, b, borrow);
            out.push(self.xor(complement, ONE));
            borrow = next;
        }
        self.equal(borrow, ZERO);
        out
    }
    fn multiply(&mut self, a: &[usize], b: &[usize]) -> Vec<usize> {
        if a.len() != b.len() || a.len() <= 16 {
            let width = a.len() + b.len();
            let mut columns = vec![Vec::new(); width + 1];
            for (i, &a) in a.iter().enumerate() {
                for (j, &b) in b.iter().enumerate() {
                    columns[i + j].push(self.and(a, b));
                }
            }
            let mut out = Vec::with_capacity(width);
            for i in 0..width {
                while columns[i].len() >= 2 {
                    let a = columns[i].pop().unwrap();
                    let b = columns[i].pop().unwrap();
                    let c = columns[i].pop().unwrap_or(ZERO);
                    let (sum, carry) = self.full_add(a, b, c);
                    columns[i].push(sum);
                    columns[i + 1].push(carry);
                }
                out.push(columns[i].pop().unwrap_or(ZERO));
            }
            for &overflow in &columns[width] {
                self.equal(overflow, ZERO);
            }
            return out;
        }

        // Unsigned Karatsuba: all carries and the two nonnegative
        // subtractions are Boolean relations, without native-field aliases.
        let half = a.len().div_ceil(2);
        let mut ah = a[half..].to_vec();
        let mut bh = b[half..].to_vec();
        ah.resize(half, ZERO);
        bh.resize(half, ZERO);
        let low = self.multiply(&a[..half], &b[..half]);
        let high = self.multiply(&ah, &bh);
        let sa = self.add(&a[..half], &ah);
        let sb = self.add(&b[..half], &bh);
        let product = self.multiply(&sa, &sb);
        let mut padded_low = low.clone();
        let mut padded_high = high.clone();
        padded_low.resize(product.len(), ZERO);
        padded_high.resize(product.len(), ZERO);
        let middle = self.sub(&product, &padded_low);
        let middle = self.sub(&middle, &padded_high);
        let ends: Vec<_> = low.into_iter().chain(high).collect();
        let mut shifted = vec![ZERO; ends.len()];
        shifted[half..half + middle.len()].copy_from_slice(&middle);
        let mut out = self.add(&ends, &shifted);
        for &overflow in &out[2 * a.len()..] {
            self.equal(overflow, ZERO);
        }
        out.truncate(2 * a.len());
        out
    }
    fn canonical(&mut self, base: usize) {
        // x>=p iff its upper 32 bits are all one and its lower 32 are nonzero.
        let mut high_ones = ONE;
        let mut low_zero = ONE;
        for i in 0..32 {
            high_ones = self.and(high_ones, base + 32 + i);
            let n = self.xor(base + i, ONE);
            low_zero = self.and(low_zero, n);
        }
        let nonzero = self.xor(low_zero, ONE);
        let bad = self.and(high_ones, nonzero);
        self.equal(bad, ZERO);
    }
    pub fn new(operation: Operation) -> Self {
        let mut r = Self {
            operation,
            recipes: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        };
        // [a,b,c,quotient], each at a 128-bit word boundary. Inputs and output
        // are canonical 64-bit integers; quotient is 1 bit for add, 64 for mul.
        for i in 0..512 {
            let word = i / 128;
            let width = match operation {
                Operation::Add => {
                    if word == 3 {
                        1
                    } else {
                        64
                    }
                }
                Operation::Mul => 64,
                Operation::MulSmall(n) => {
                    if word == 1 || word == 3 {
                        n
                    } else {
                        64
                    }
                }
                Operation::Mask => match word {
                    0 | 2 => 64,
                    1 => 1,
                    _ => 0,
                },
                Operation::Pack(n) => match word {
                    0 | 1 => n,
                    2 => 2 * n,
                    _ => 0,
                },
                Operation::Range(n) => match word {
                    0 => n,
                    _ => 0,
                },
                Operation::Xor4 => {
                    if word < 3 {
                        4
                    } else {
                        0
                    }
                }
                Operation::Canonical => {
                    if word == 0 {
                        64
                    } else {
                        0
                    }
                }
                #[cfg(test)]
                Operation::Equal => {
                    if word < 2 {
                        128
                    } else {
                        0
                    }
                }
                Operation::Linear {
                    a,
                    b,
                    a_bits,
                    b_bits,
                } => match word {
                    0 => a_bits,
                    1 => b_bits,
                    2 => linear_width(a, b, a_bits, b_bits),
                    _ => 0,
                },
            };
            r.recipes.push(Recipe::Input(i));
            if i % 128 < width {
                r.a.push(vec![i]);
                r.b.push(vec![ONE]);
            } else {
                r.a.push(vec![]);
                r.b.push(vec![]);
            }
        }
        r.recipes.push(Recipe::One);
        r.a.push(vec![ONE]);
        r.b.push(vec![ONE]);
        r.recipes.push(Recipe::Zero);
        r.a.push(vec![]);
        r.b.push(vec![]);
        if matches!(
            operation,
            Operation::Add | Operation::Mul | Operation::MulSmall(_)
        ) {
            for base in [0, 128, 256] {
                r.canonical(base);
            }
        }
        let a: Vec<_> = (0..64).collect();
        let b: Vec<_> = (128..192).collect();
        let c: Vec<_> = (256..320).collect();
        match operation {
            Operation::Linear {
                a: ka,
                b: kb,
                a_bits,
                b_bits,
            } => {
                let width = linear_width(ka, kb, a_bits, b_bits).max(1);
                let mut sum = vec![ZERO; width];
                for (coefficient, input, bits) in [(ka, &a, a_bits), (kb, &b, b_bits)] {
                    if bits == 0 {
                        continue;
                    }
                    for shift in 0..64 {
                        if coefficient >> shift & 1 == 0 {
                            continue;
                        }
                        let mut term = vec![ZERO; width];
                        term[shift..shift + bits].copy_from_slice(&input[..bits]);
                        let next = r.add(&sum, &term);
                        r.equal(next[width], ZERO);
                        sum.copy_from_slice(&next[..width]);
                    }
                }
                for (i, &bit) in sum.iter().enumerate() {
                    r.equal(bit, 256 + i);
                }
            }
            Operation::Mask => {
                r.canonical(0);
                for i in 0..64 {
                    let v = r.and(i, 128);
                    r.equal(v, 256 + i);
                }
            }
            #[cfg(test)]
            Operation::Equal => {
                for i in 0..128 {
                    r.equal(i, 128 + i);
                }
            }
            Operation::Canonical => {
                r.canonical(0);
            }
            Operation::Pack(n) => {
                assert!([1, 2, 3, 4, 8, 16, 32, 64].contains(&n));
                for i in 0..n {
                    r.equal(256 + i, i);
                    r.equal(256 + n + i, 128 + i);
                }
            }
            Operation::Range(n) => {
                assert!(n <= 64);
            }
            Operation::Xor4 => {
                for i in 0..4 {
                    let x = r.xor(i, 128 + i);
                    r.equal(256 + i, x);
                }
            }
            Operation::Add => {
                let lhs = r.add(&a, &b);
                let qp: Vec<_> = (0..64)
                    .map(|i| if (P >> i) & 1 == 1 { 384 } else { ZERO })
                    .collect();
                let rhs = r.add(&c, &qp);
                for (a, b) in lhs.into_iter().zip(rhs) {
                    r.equal(a, b);
                }
            }
            Operation::Mul | Operation::MulSmall(_) => {
                let width = match operation {
                    Operation::MulSmall(n) => n,
                    _ => 64,
                };
                assert!((1..=64).contains(&width));
                let mut lhs = r.multiply(&a, &b[..width]);
                lhs.resize(129, ZERO);
                let mut q64 = vec![ZERO; 129];
                let mut q32 = vec![ZERO; 129];
                for i in 0..width {
                    q64[64 + i] = 384 + i;
                    q32[32 + i] = 384 + i;
                }
                let difference = r.sub(&q64, &q32);
                let mut q = vec![ZERO; 129];
                let mut cwide = vec![ZERO; 129];
                for (i, v) in q.iter_mut().enumerate().take(width) {
                    *v = 384 + i;
                }
                cwide[..64].copy_from_slice(&c);
                let qp = r.add(&difference, &q);
                r.equal(qp[129], ZERO);
                let rhs = r.add(&qp[..129], &cwide);
                r.equal(rhs[129], ZERO);
                for (a, b) in lhs.into_iter().zip(rhs) {
                    r.equal(a, b);
                }
            }
        }
        r.compact_linear()
    }

    fn compact_linear(self) -> Self {
        // With the constant pinned to one, XOR rows are definitions of linear
        // forms. Substitute them into A/B and keep only the nonlinear rows,
        // boundary words, and checks. Each removed row has a unique extension.
        fn parity(mut indices: Vec<usize>) -> Vec<usize> {
            indices.sort_unstable();
            let mut out = Vec::new();
            let mut i = 0;
            while i < indices.len() {
                let mut end = i + 1;
                while end < indices.len() && indices[end] == indices[i] {
                    end += 1;
                }
                if (end - i) % 2 == 1 {
                    out.push(indices[i]);
                }
                i = end;
            }
            out
        }
        let mut forms: Vec<Vec<usize>> = vec![Vec::new(); self.recipes.len()];
        // Boundary constraints refer forward to the fixed ONE column.
        let boundary = if self.operation.is_assertion() {
            128
        } else {
            512
        };
        for (i, form) in forms.iter_mut().enumerate().take(boundary) {
            form.push(i);
        }
        forms[ONE] = vec![boundary];
        forms[ZERO] = vec![boundary + 1];
        let mut result = Self {
            recipes: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
            operation: self.operation,
        };
        for (old, recipe) in self.recipes.into_iter().enumerate() {
            if (boundary..ONE).contains(&old) {
                assert!(self.a[old].is_empty() && self.b[old].is_empty());
                continue;
            }
            if let Recipe::Xor(a, b) = recipe {
                forms[old] = parity(forms[a].iter().chain(&forms[b]).copied().collect());
                continue;
            }
            let new = result.recipes.len();
            forms[old] = vec![new];
            let expand = |row: &[usize]| {
                parity(row.iter().flat_map(|&i| forms[i].iter().copied()).collect())
            };
            let a = expand(&self.a[old]);
            let b = expand(&self.b[old]);
            let recipe = match recipe {
                Recipe::And(_, _) => Recipe::Product(a.clone(), b.clone()),
                Recipe::Product(_, _) => unreachable!("already compacted"),
                other => other,
            };
            result.a.push(a);
            result.b.push(b);
            result.recipes.push(recipe);
        }
        result
    }
    pub fn useful_bits(&self) -> usize {
        self.recipes.len()
    }
    pub fn block(&self, nu: usize) -> BlockR1cs {
        let k = self.recipes.len().next_power_of_two();
        let mut a = self.a.clone();
        a.resize(k, vec![]);
        let mut b = self.b.clone();
        b.resize(k, vec![]);
        BlockR1cs {
            m: k.ilog2() as usize + nu,
            k_log: k.ilog2() as usize,
            k_skip: 6,
            useful_bits: self.recipes.len(),
            a_0: SparseBinaryMatrix::new(k, k, a),
            b_0: SparseBinaryMatrix::new(k, k, b),
            c_0: SparseBinaryMatrix::new(k, k, (0..k).map(|i| vec![i]).collect()),
            layout: WitnessLayout::BatchMajor,
            const_pin: Some(if self.operation.is_assertion() {
                128
            } else {
                ONE
            }),
            digest_cache: OnceLock::new(),
            csc_cache: OnceLock::new(),
        }
    }
    pub fn row(&self, raw: [F128; 4]) -> Vec<bool> {
        let mut z = Vec::with_capacity(self.recipes.len());
        for recipe in &self.recipes {
            let value = match *recipe {
                Recipe::Input(i) => {
                    let v = raw[i / 128];
                    (if i % 128 < 64 { v.lo } else { v.hi }) >> (i % 64) & 1 == 1
                }
                Recipe::One => true,
                Recipe::Zero | Recipe::Check => false,
                Recipe::Xor(a, b) => z[a] ^ z[b],
                Recipe::And(a, b) => z[a] & z[b],
                Recipe::Product(ref a, ref b) => {
                    a.iter().fold(false, |v, &i| v ^ z[i]) & b.iter().fold(false, |v, &i| v ^ z[i])
                }
            };
            z.push(value);
        }
        z.resize(z.len().next_power_of_two(), false);
        z
    }
    pub fn batch64(&self, rows: &[[F128; 4]]) -> Vec<[u64; 3]> {
        assert!(!rows.is_empty() && rows.len() <= 64);
        let mask = u64::MAX >> (64 - rows.len());
        let mut out = vec![[0; 3]; self.useful_bits().div_ceil(128) * 128];
        let one = if self.operation.is_assertion() {
            128
        } else {
            ONE
        };
        out[one][0] = mask; // Boundary rows refer forward to ONE.
        for (i, recipe) in self.recipes.iter().enumerate() {
            let parity = |indices: &[usize]| indices.iter().fold(0u64, |v, &j| v ^ out[j][0]);
            let z = match recipe {
                Recipe::Input(bit) => rows.iter().enumerate().fold(0, |v, (j, row)| {
                    let word = row[bit / 128];
                    let half = if bit % 128 < 64 { word.lo } else { word.hi };
                    v | (((half >> (bit % 64)) & 1) << j)
                }),
                Recipe::One => mask,
                Recipe::Zero | Recipe::Check => 0,
                Recipe::Product(a, b) => parity(a) & parity(b),
                Recipe::And(_, _) | Recipe::Xor(_, _) => unreachable!("uncompacted relation"),
            };
            out[i][0] = z;
            let a = self.a[i].iter().fold(0, |v, &j| v ^ out[j][0]);
            let b = self.b[i].iter().fold(0, |v, &j| v ^ out[j][0]);
            assert_eq!(a & b, z, "invalid packed relation row {i}");
            out[i] = [z, a, b];
        }
        out
    }
    pub fn evaluate(&self, a: u64, b: u64) -> [F128; 4] {
        assert!(a < P && b < P);
        let wide = match self.operation {
            Operation::Add => u128::from(a) + u128::from(b),
            Operation::Mul | Operation::MulSmall(_) => u128::from(a) * u128::from(b),
            _ => panic!("use evaluate_words for packing"),
        };
        [
            a,
            b,
            (wide % u128::from(P)) as u64,
            (wide / u128::from(P)) as u64,
        ]
        .map(|x| F128::new(x, 0))
    }
    pub fn evaluate_words(&self, a: F128, b: F128) -> [F128; 4] {
        let av = u128::from(a.lo) | (u128::from(a.hi) << 64);
        let bv = u128::from(b.lo) | (u128::from(b.hi) << 64);
        let result = match self.operation {
            Operation::Linear {
                a: ka,
                b: kb,
                a_bits,
                b_bits,
            } => {
                assert!(av < (1u128 << a_bits) && bv < (1u128 << b_bits));
                av * u128::from(ka) + bv * u128::from(kb)
            }
            Operation::Add | Operation::Mul | Operation::MulSmall(_) => {
                assert_eq!(a.hi | b.hi, 0);
                if let Operation::MulSmall(n) = self.operation {
                    assert!(bv < (1u128 << n));
                }
                return self.evaluate(a.lo, b.lo);
            }
            Operation::Pack(n) => {
                assert!(av < (1u128 << n) && bv < (1u128 << n));
                av | (bv << n)
            }
            Operation::Range(n) => {
                assert!(av < (1u128 << n) && bv == 0);
                0
            }
            Operation::Xor4 => {
                assert!(av < 16 && bv < 16);
                av ^ bv
            }
            Operation::Canonical => {
                assert!(av < u128::from(P) && bv == 0);
                0
            }
            #[cfg(test)]
            Operation::Equal => {
                assert_eq!(av, bv);
                0
            }
            Operation::Mask => {
                assert!(av < u128::from(P) && bv < 2);
                av * bv
            }
        };
        [
            a,
            b,
            F128::new(result as u64, (result >> 64) as u64),
            F128::ZERO,
        ]
    }
    pub fn satisfied(&self, z: &[bool]) -> bool {
        (0..self.recipes.len()).all(|i| {
            let a = self.a[i].iter().fold(false, |v, &j| v ^ z[j]);
            let b = self.b[i].iter().fold(false, |v, &j| v ^ z[j]);
            (a & b) == z[i]
        }) && z[if self.operation.is_assertion() {
            128
        } else {
            ONE
        }] && z[self.recipes.len()..].iter().all(|x| !x)
    }
}

fn linear_width(a: u64, b: u64, a_bits: usize, b_bits: usize) -> usize {
    assert!(a_bits <= 32 && b_bits <= 32 && a <= 16 && b <= 16);
    let max = ((1u128 << a_bits) - 1) * u128::from(a) + ((1u128 << b_bits) - 1) * u128::from(b);
    assert!(max < u128::from(P));
    (128 - max.leading_zeros()) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_linear_and_in_place_witness() {
        for (ka, kb, na, nb) in [
            (16, 1, 4, 4),
            (7, 15, 1, 4),
            (16, 1, 32, 4),
            (16, 1, 0, 0),
            (16, 0, 1, 0),
            (15, 0, 4, 0),
            (1, 1, 32, 32),
        ] {
            let r = Relation::new(Operation::Linear {
                a: ka,
                b: kb,
                a_bits: na,
                b_bits: nb,
            });
            let max_a = (1u64 << na) - 1;
            let max_b = (1u64 << nb) - 1;
            let rows: Vec<_> = (0..11)
                .map(|i| {
                    r.evaluate_words(
                        F128::new(max_a.saturating_sub(i), 0),
                        F128::new(max_b.saturating_sub(i), 0),
                    )
                })
                .collect();
            for &raw in &rows {
                assert!(r.satisfied(&r.row(raw)));
                for word in 0..4 {
                    let mut bad = raw;
                    bad[word].hi ^= 1;
                    assert!(!r.satisfied(&r.row(bad)));
                }
                let mut bad = raw;
                bad[2].lo ^= 1;
                assert!(!r.satisfied(&r.row(bad)));
                for (word, n) in [(0, na), (1, nb)] {
                    let mut bad = raw;
                    bad[word].lo |= 1 << n;
                    assert!(!r.satisfied(&r.row(bad)));
                }
            }
            let old = packed_witness(&r, &rows, 4);
            for dirty in [false, true] {
                let mut z = vec![if dirty { F128::ONE } else { F128::ZERO }; old.0.len()];
                let mut a = z.clone();
                let mut b = z.clone();
                let stripe = packed_witness_into(
                    &r,
                    &rows,
                    4,
                    flock_prover::union::SlotWitnessDest {
                        z: &mut z,
                        a: &mut a,
                        b: &mut b,
                        elide_padding_writes: !dirty,
                        dead_padding_unread: false,
                    },
                );
                assert_eq!((z, a, b, stripe), old);
            }
        }
    }
    #[test]
    fn recursive_multiplication_carries_and_raw_witness_are_constrained() {
        let relation = Relation::new(Operation::Mul);
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % P
        };
        for _ in 0..256 {
            let (a, b) = (next(), next());
            let raw = relation.evaluate(a, b);
            let product = u128::from(a) * u128::from(b);
            assert_eq!(raw[2].lo, (product % u128::from(P)) as u64);
            assert_eq!(raw[3].lo, (product / u128::from(P)) as u64);
            assert!(relation.satisfied(&relation.row(raw)), "{a} * {b}");
        }
        let mut row = relation.row(relation.evaluate(P - 2, P - 3));
        for (index, recipe) in relation.recipes.iter().enumerate() {
            if matches!(recipe, Recipe::And(..) | Recipe::Product(..)) {
                row[index] ^= true;
                assert!(
                    !relation.satisfied(&row),
                    "unconstrained product/carry {index}"
                );
                row[index] ^= true;
            }
        }
        eprintln!(
            "Goldilocks multiplication: {} useful bits",
            relation.useful_bits()
        );
    }

    #[test]
    fn canonical_arithmetic_and_adversarial_rows() {
        for op in [Operation::Add, Operation::Mul] {
            let r = Relation::new(op);
            for a in [0, 1, 1 << 32, P - 1, P - 2] {
                for b in [0, 1, 1 << 32, P - 1, P - 2] {
                    let raw = r.evaluate(a, b);
                    assert!(r.satisfied(&r.row(raw)), "{op:?}: {a} {b}");
                    for word in 0..4 {
                        let mut bad = raw;
                        bad[word].hi = 1;
                        assert!(!r.satisfied(&r.row(bad)), "high bit word {word}");
                    }
                    for word in [2, 3] {
                        let mut bad = raw;
                        bad[word].lo ^= 1;
                        assert!(!r.satisfied(&r.row(bad)), "wrong output/quotient");
                    }
                }
            }
            // Integer aliases must not be accepted as field encodings.
            for value in [P, P + 1, u64::MAX] {
                let mut raw = r.evaluate(0, 0);
                raw[0].lo = value;
                assert!(!r.satisfied(&r.row(raw)));
            }
        }
    }
    #[test]
    fn bounded_operations_reject_aliases_and_wrong_outputs() {
        for op in [
            Operation::MulSmall(2),
            Operation::MulSmall(4),
            Operation::MulSmall(8),
            Operation::MulSmall(16),
            Operation::Mask,
            Operation::Pack(3),
            Operation::Pack(8),
            Operation::Pack(64),
            Operation::Range(1),
            Operation::Range(4),
            Operation::Canonical,
            Operation::Equal,
            Operation::Xor4,
        ] {
            let r = Relation::new(op);
            let (a, b) = match op {
                Operation::MulSmall(n) => (F128::new(P - 1, 0), F128::new((1 << n) - 1, 0)),
                Operation::Mask => (F128::new(P - 1, 0), F128::ONE),
                Operation::Pack(n) => (
                    F128::new(if n == 64 { u64::MAX } else { (1 << n) - 1 }, 0),
                    F128::new(1, 0),
                ),
                Operation::Range(n) => (F128::new((1 << n) - 1, 0), F128::ZERO),
                Operation::Canonical => (F128::new(P - 1, 0), F128::ZERO),
                Operation::Equal => (F128::new(7, 8), F128::new(7, 8)),
                Operation::Xor4 => (F128::new(7, 0), F128::new(13, 0)),
                _ => unreachable!(),
            };
            let raw = r.evaluate_words(a, b);
            assert!(r.satisfied(&r.row(raw)), "{op:?}");
            let mut bad = raw;
            if let Operation::Range(n) = op {
                bad[0].lo |= 1 << n;
            } else if op == Operation::Canonical {
                bad[0].lo = P;
            } else {
                bad[2].lo ^= 1;
            }
            assert!(!r.satisfied(&r.row(bad)), "wrong {op:?} output");
            if !matches!(op, Operation::Equal) {
                let mut bad = raw;
                bad[0].hi |= 1;
                assert!(!r.satisfied(&r.row(bad)), "{op:?} high input bit");
            }
            if let Operation::MulSmall(n) = op {
                let mut bad = raw;
                bad[1].lo |= 1 << n;
                assert!(!r.satisfied(&r.row(bad)), "coefficient range");
                let mut bad = raw;
                bad[3].lo ^= 1;
                assert!(!r.satisfied(&r.row(bad)), "quotient");
            }
        }
    }
}
