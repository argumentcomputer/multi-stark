use crate::goldilocks::Relation;
use flock_prover::{
    circuit::builder::{GateType, SlotWitness},
    field::F128,
    r1cs::{BlockR1cs, SparseBinaryMatrix},
    schedule::{IoWord, TableType},
    union::SlotWitnessDest,
};
use rayon::prelude::*;
use std::sync::Arc;

pub struct PackedGoldGate {
    pub relation: Arc<Relation>,
    pub block: Arc<BlockR1cs>,
    pub lanes: usize,
}

pub struct AdviceBlake3Gate {
    pub nu: usize,
}
impl GateType for AdviceBlake3Gate {
    type Row = flock_prover::r1cs_hashes::blake3::Compression;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut table = crate::hash_chain::Blake3Gate { nu: self.nu }.table();
        for word in &mut table.io_schema {
            word.dir = flock_prover::schedule::IoDirection::In;
        }
        table
    }
    fn eval(&self, inputs: &[F128], _: &(), _: &mut Vec<F128>) -> Self::Row {
        assert_eq!(inputs.len(), 11);
        let mut outputs = Vec::new();
        let row =
            crate::hash_chain::Blake3Gate { nu: self.nu }.eval(&inputs[..7], &(), &mut outputs);
        assert_eq!(outputs, inputs[7..]);
        row
    }
    fn witness(&self, _: &[Self::Row], _: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

pub fn block(relation: &Relation, lanes: usize, nu: usize) -> BlockR1cs {
    assert!(lanes.is_power_of_two());
    let base = relation.block(nu);
    let stride = relation.useful_bits().div_ceil(128) * 128;
    let useful = stride * lanes;
    let k = useful.next_power_of_two();
    let mut a = vec![vec![]; k];
    let mut b = vec![vec![]; k];
    let one = base.const_pin.unwrap();
    for lane in 0..lanes {
        let offset = lane * stride;
        for i in 0..relation.useful_bits() {
            a[offset + i] = base.a_0.rows[i].iter().map(|j| offset + j).collect();
            b[offset + i] = base.b_0.rows[i].iter().map(|j| offset + j).collect();
        }
        // A single lincheck pin must force every lane's constant to one.
        a[offset + one] = vec![one];
        b[offset + one] = vec![one];
    }
    BlockR1cs {
        m: k.ilog2() as usize + nu,
        k_log: k.ilog2() as usize,
        useful_bits: useful,
        a_0: SparseBinaryMatrix::new(k, k, a),
        b_0: SparseBinaryMatrix::new(k, k, b),
        c_0: SparseBinaryMatrix::new(k, k, (0..k).map(|i| vec![i]).collect()),
        ..base
    }
}

pub fn schema(relation: &Relation, lanes: usize) -> Vec<IoWord> {
    let stride = relation.useful_bits().div_ceil(128);
    let arity = if relation.operation.is_assertion() {
        1
    } else {
        3
    };
    (0..lanes)
        .flat_map(|lane| (0..arity).map(move |word| IoWord::input(lane * stride + word)))
        .collect()
}

impl GateType for PackedGoldGate {
    type Row = Vec<[F128; 4]>;
    type Hint = ();
    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&self.block).with_io_schema(schema(&self.relation, self.lanes))
    }
    fn eval(&self, inputs: &[F128], _: &(), _: &mut Vec<F128>) -> Self::Row {
        if self.relation.operation.is_assertion() {
            assert_eq!(inputs.len(), self.lanes);
            return inputs
                .iter()
                .map(|&v| self.relation.evaluate_words(v, F128::ZERO))
                .collect();
        }
        assert_eq!(inputs.len(), self.lanes * 3);
        inputs
            .as_chunks::<3>()
            .0
            .iter()
            .map(|v| {
                let row = self.relation.evaluate_words(v[0], v[1]);
                assert_eq!(row[2], v[2]);
                row
            })
            .collect()
    }
    fn witness(&self, _: &[Self::Row], _: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

#[cfg(test)]
fn witness_into_scalar(
    relation: &Relation,
    lanes: usize,
    rows: &[Vec<[F128; 4]>],
    nu: usize,
    dst: SlotWitnessDest<'_>,
) -> Vec<u8> {
    let stride = relation.useful_bits().div_ceil(128) * 128;
    let useful = lanes * stride;
    let k = useful.next_power_of_two();
    let capacity = 1usize << nu;
    assert!(nu >= 3 && rows.len() <= capacity);
    assert_eq!(dst.z.len(), k / 128 * capacity);
    if !dst.elide_padding_writes {
        dst.z.fill(F128::ZERO);
        dst.a.fill(F128::ZERO);
        dst.b.fill(F128::ZERO);
    }
    let base = relation.block(3);
    let mut stripe = vec![0; k * capacity / 8];
    for (j, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), lanes);
        for (lane, &raw) in row.iter().enumerate() {
            let z = relation.row(raw);
            assert!(relation.satisfied(&z));
            for word in 0..stride / 128 {
                let mut packed = [0u128; 3];
                for bit in word * 128..((word + 1) * 128).min(relation.useful_bits()) {
                    let av = base.a_0.rows[bit].iter().fold(false, |v, &i| v ^ z[i]);
                    let bv = base.b_0.rows[bit].iter().fold(false, |v, &i| v ^ z[i]);
                    for (p, v) in packed.iter_mut().zip([z[bit], av, bv]) {
                        *p |= u128::from(v) << (bit % 128);
                    }
                    stripe[(j / 8) * k + lane * stride + bit] |= u8::from(z[bit]) << (j % 8);
                }
                let [zv, av, bv] = packed.map(|v| F128::new(v as u64, (v >> 64) as u64));
                let address = (lane * stride / 128 + word) * capacity + j;
                dst.z[address] = zv;
                dst.a[address] = av;
                dst.b[address] = bv;
            }
        }
    }
    stripe
}

// LSB-first transpose: output row i, bit j = input row j, bit i.
fn transpose(a: &mut [u64; 64]) {
    let mut shift = 32;
    let mut mask = 0x0000_0000_ffff_ffff;
    while shift != 0 {
        let mut k = 0;
        while k < 64 {
            let t = ((a[k] >> shift) ^ a[k + shift]) & mask;
            a[k] ^= t << shift;
            a[k + shift] ^= t;
            k = (k + shift + 1) & !shift;
        }
        shift >>= 1;
        mask ^= mask << shift;
    }
}

pub fn witness_into(
    relation: &Relation,
    lanes: usize,
    rows: &[Vec<[F128; 4]>],
    nu: usize,
    dst: SlotWitnessDest<'_>,
) -> Vec<u8> {
    let stride = relation.useful_bits().div_ceil(128) * 128;
    let k = (lanes * stride).next_power_of_two();
    let capacity = 1usize << nu;
    assert!(nu >= 3 && rows.len() <= capacity);
    assert_eq!(dst.z.len(), k / 128 * capacity);
    assert_eq!(dst.a.len(), dst.z.len());
    assert_eq!(dst.b.len(), dst.z.len());
    if !dst.elide_padding_writes {
        dst.z.fill(F128::ZERO);
        dst.a.fill(F128::ZERO);
        dst.b.fill(F128::ZERO);
    }
    let mut stripe = vec![0; k * capacity / 8];
    if rows.is_empty() {
        return stripe;
    }
    // The first packed fold reads complete groups of eight rows.
    let padded_rows = rows.len().next_multiple_of(8).min(capacity);
    for buffer in [&mut *dst.z, &mut *dst.a, &mut *dst.b] {
        for column in buffer.chunks_mut(capacity).take(lanes * stride / 128) {
            column[rows.len()..padded_rows].fill(F128::ZERO);
        }
    }
    // Split each column into disjoint row ranges. Workers own matching stripe
    // ranges, so both layouts can be filled in parallel without extra buffers.
    let rows_per_task = rows
        .len()
        .div_ceil(rayon::current_num_threads())
        .div_ceil(64)
        * 64;
    fn split(
        buffer: &mut [F128],
        capacity: usize,
        count: usize,
        rows_per_task: usize,
    ) -> Vec<Vec<&mut [F128]>> {
        let task_count = count.div_ceil(rows_per_task);
        let mut tasks: Vec<Vec<&mut [F128]>> = (0..task_count).map(|_| Vec::new()).collect();
        for column in buffer.chunks_mut(capacity) {
            for (task, chunk) in tasks
                .iter_mut()
                .zip(column[..count].chunks_mut(rows_per_task))
            {
                task.push(chunk);
            }
        }
        tasks
    }
    let z_tasks = split(dst.z, capacity, rows.len(), rows_per_task);
    let a_tasks = split(dst.a, capacity, rows.len(), rows_per_task);
    let b_tasks = split(dst.b, capacity, rows.len(), rows_per_task);
    z_tasks
        .into_par_iter()
        .zip(a_tasks)
        .zip(b_tasks)
        .zip(stripe.par_chunks_mut(k * rows_per_task / 8))
        .zip(rows.par_chunks(rows_per_task))
        .for_each(|((((mut z, mut a), mut b), stripe), task_rows)| {
            for (batch, physical_rows) in task_rows.chunks(64).enumerate() {
                for lane in 0..lanes {
                    let raw: Vec<_> = physical_rows
                        .iter()
                        .map(|row| {
                            assert_eq!(row.len(), lanes);
                            row[lane]
                        })
                        .collect();
                    let bits = relation.batch64(&raw);
                    for (bit, v) in bits.iter().enumerate() {
                        for (group, byte) in v[0]
                            .to_le_bytes()
                            .into_iter()
                            .enumerate()
                            .take(physical_rows.len().div_ceil(8))
                        {
                            stripe[(batch * 8 + group) * k + lane * stride + bit] = byte;
                        }
                    }
                    for (word, bits) in bits.as_chunks::<128>().0.iter().enumerate() {
                        for (column, dst) in [&mut z, &mut a, &mut b].into_iter().enumerate() {
                            let mut lo = std::array::from_fn(|i| bits[i][column]);
                            let mut hi = std::array::from_fn(|i| bits[64 + i][column]);
                            transpose(&mut lo);
                            transpose(&mut hi);
                            let start = batch * 64;
                            let dst = &mut dst[lane * stride / 128 + word];
                            for (j, value) in dst[start..start + physical_rows.len()]
                                .iter_mut()
                                .enumerate()
                            {
                                *value = F128::new(lo[j], hi[j]);
                            }
                        }
                    }
                }
            }
        });
    stripe
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goldilocks::Operation;

    #[test]
    fn blake3_advice_is_constrained_by_matrices_without_witness_checks() {
        use flock_prover::r1cs_hashes::blake3::{build_block_r1cs, build_block_witness};
        let gate = AdviceBlake3Gate { nu: 3 };
        let table = gate.table();
        assert_eq!(table.io_schema.len(), 11);
        assert!(
            table
                .io_schema
                .iter()
                .all(|w| w.dir == flock_prover::schedule::IoDirection::In)
        );
        let block = build_block_r1cs(3);
        let row = build_block_witness(&[0x12345678; 8], &[0x9abcdef0; 16], 17, 61, 11);
        let satisfied = |z: &[bool]| {
            z[block.const_pin.unwrap()]
                && (0..z.len()).all(|i| {
                    let a = block.a_0.rows[i].iter().fold(false, |v, &j| v ^ z[j]);
                    let b = block.b_0.rows[i].iter().fold(false, |v, &j| v ^ z[j]);
                    (a & b) == z[i]
                })
        };
        assert!(satisfied(&row));
        for word in &table.io_schema {
            for bit in 0..128 {
                let mut forged = row.clone();
                forged[word.word_col * 128 + bit] ^= true;
                assert!(
                    !satisfied(&forged),
                    "unconstrained BLAKE3 word {}, bit {bit}",
                    word.word_col
                );
            }
        }
        assert!(!satisfied(&vec![false; row.len()]));
    }

    #[test]
    fn bit_sliced_witness_matches_scalar_at_boundaries() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for op in [
            Operation::Add,
            Operation::Mul,
            Operation::Range(4),
            Operation::Canonical,
        ] {
            let r = Relation::new(op);
            for n in [0usize, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 257] {
                let nu = n.max(8).next_power_of_two().ilog2() as usize;
                let rows: Vec<_> = (0..n)
                    .map(|i| {
                        (0..2)
                            .map(|j| {
                                r.evaluate_words(
                                    F128::new(((i + j) % 16) as u64, 0),
                                    F128::new(if op.is_assertion() { 0 } else { 13 }, 0),
                                )
                            })
                            .collect()
                    })
                    .collect();
                let len = (r.useful_bits().div_ceil(128) * 2).next_power_of_two() << nu;
                let run = |scalar: bool| {
                    let mut z = vec![F128::ONE; len];
                    let mut a = z.clone();
                    let mut b = z.clone();
                    let dst = SlotWitnessDest {
                        z: &mut z,
                        a: &mut a,
                        b: &mut b,
                        elide_padding_writes: false,
                        dead_padding_unread: false,
                    };
                    let stripe = if scalar {
                        witness_into_scalar(&r, 2, &rows, nu, dst)
                    } else {
                        pool.install(|| witness_into(&r, 2, &rows, nu, dst))
                    };
                    (z, a, b, stripe)
                };
                assert_eq!(run(false), run(true), "{op:?}, {n}");
                if n > 0 {
                    let expected = run(true);
                    let mut z = vec![F128::ONE; len];
                    let mut a = z.clone();
                    let mut b = z.clone();
                    let stripe = pool.install(|| {
                        witness_into(
                            &r,
                            2,
                            &rows,
                            nu,
                            SlotWitnessDest {
                                z: &mut z,
                                a: &mut a,
                                b: &mut b,
                                elide_padding_writes: true,
                                dead_padding_unread: true,
                            },
                        )
                    });
                    assert_eq!(stripe, expected.3);
                    let capacity = 1 << nu;
                    for (actual, expected) in
                        [&z, &a, &b]
                            .into_iter()
                            .zip([&expected.0, &expected.1, &expected.2])
                    {
                        for column in 0..r.useful_bits().div_ceil(128) * 2 {
                            let start = column * capacity;
                            let end = start + n.next_multiple_of(8);
                            assert_eq!(
                                actual[start..end],
                                expected[start..end],
                                "dirty partial group: {op:?}, {n}"
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn every_lane_is_constrained_and_pinned() {
        let relation = Relation::new(Operation::MulSmall(4));
        let block = block(&relation, 4, 3);
        let rows = vec![
            (0..4)
                .map(|i| relation.evaluate(i + 4, 7))
                .collect::<Vec<_>>(),
        ];
        let len = 1 << (block.m - 7);
        let mut z = vec![F128::ZERO; len];
        let mut a = z.clone();
        let mut b = z.clone();
        witness_into(
            &relation,
            4,
            &rows,
            3,
            SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes: true,
                dead_padding_unread: false,
            },
        );
        let width = 1 << (block.k_log - 7);
        let row: Vec<_> = (0..width * 128)
            .map(|bit| {
                let v = z[(bit / 128) * 8];
                (if bit % 128 < 64 { v.lo } else { v.hi }) >> (bit % 64) & 1 != 0
            })
            .collect();
        let satisfied = |z: &[bool]| {
            (0..z.len()).all(|i| {
                let a = block.a_0.rows[i].iter().fold(false, |v, &j| v ^ z[j]);
                let b = block.b_0.rows[i].iter().fold(false, |v, &j| v ^ z[j]);
                (a & b) == z[i]
            })
        };
        assert!(satisfied(&row));
        let stride = relation.useful_bits().div_ceil(128) * 128;
        for lane in 0..4 {
            let mut bad = row.clone();
            bad[lane * stride + 256] ^= true;
            assert!(!satisfied(&bad));
            if lane > 0 {
                let mut bad = row.clone();
                bad[lane * stride..(lane + 1) * stride].fill(false);
                assert!(!satisfied(&bad), "unpinned lane {lane}");
            }
        }
    }
}
