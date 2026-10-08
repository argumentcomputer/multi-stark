//! Registry-owned binary matrices, contracted through shared 64-bit blocks.
use anyhow::{Result, ensure};
use flock_prover::{r1cs::SparseBinaryMatrix, schedule::Registry};
use ix_stage4_trace::{F128MatrixSideV1, F128StaticMatrixIdV1};
use ix_terminal_circuit::*;
use std::collections::{BTreeMap, HashMap};

const PHASE: ConstraintPhase = ConstraintPhase::Lincheck;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Reference {
    Leaf(usize),
    Node(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Node {
    variable: usize,
    low: Reference,
    high: Reference,
}
struct Matrix {
    id: F128StaticMatrixIdV1,
    nodes: Vec<Node>,
    root: Reference,
}

impl Matrix {
    fn build(
        id: F128StaticMatrixIdV1,
        matrix: &SparseBinaryMatrix,
        patterns: &mut Vec<[u64; 64]>,
        pool: &mut HashMap<[u64; 64], usize>,
    ) -> Result<Self> {
        let variables = id.variables as usize;
        ensure!((6..32).contains(&variables), "fixed matrix arity");
        ensure!(
            matrix.num_rows == 1usize << variables
                && matrix.num_cols == 1usize << variables
                && matrix.rows.len() == matrix.num_rows,
            "fixed matrix dimensions"
        );
        let high = variables - 6;
        let mut cells = Vec::new();
        for (block_row, rows) in matrix.rows.as_chunks::<64>().0.iter().enumerate() {
            let mut blocks: HashMap<usize, [u64; 64]> = HashMap::new();
            for (row, columns) in rows.iter().enumerate() {
                for &column in columns {
                    ensure!(column < matrix.num_cols, "fixed matrix column");
                    blocks.entry(column >> 6).or_insert([0; 64])[row] ^= 1u64 << (column & 63);
                }
            }
            // Canonical iteration keeps shape and identifiers reproducible.
            let mut blocks: Vec<_> = blocks.into_iter().collect();
            blocks.sort_unstable_by_key(|b| b.0);
            for (column, block) in blocks {
                if block == [0; 64] {
                    continue;
                }
                let pattern = *pool.entry(block).or_insert_with(|| {
                    let index = patterns.len();
                    patterns.push(block);
                    index
                });
                cells.push(((block_row as u64) | ((column as u64) << high), pattern));
            }
        }
        let mut best = None;
        for order in [
            (0..high).flat_map(|b| [b, high + b]).collect::<Vec<_>>(),
            (0..2 * high).collect(),
            (0..high).rev().flat_map(|b| [b, high + b]).collect(),
        ] {
            let mut keys: Vec<_> = cells
                .iter()
                .map(|&(address, pattern)| {
                    (
                        order
                            .iter()
                            .fold(0u64, |key, bit| (key << 1) | ((address >> bit) & 1)),
                        pattern,
                    )
                })
                .collect();
            keys.sort_unstable_by_key(|e| e.0);
            let mut nodes = Vec::new();
            let mut intern = HashMap::new();
            fn descend(
                keys: &[(u64, usize)],
                order: &[usize],
                level: usize,
                nodes: &mut Vec<Node>,
                intern: &mut HashMap<Node, Reference>,
            ) -> Reference {
                if keys.is_empty() {
                    return Reference::Leaf(0);
                }
                if level == order.len() {
                    return Reference::Leaf(keys[0].1);
                }
                let bit = 1u64 << (order.len() - level - 1);
                let mid = keys.partition_point(|e| e.0 & bit == 0);
                let low = descend(&keys[..mid], order, level + 1, nodes, intern);
                let high = descend(&keys[mid..], order, level + 1, nodes, intern);
                if low == high {
                    return low;
                }
                let node = Node {
                    variable: order[level],
                    low,
                    high,
                };
                *intern.entry(node).or_insert_with(|| {
                    let id = Reference::Node(nodes.len());
                    nodes.push(node);
                    id
                })
            }
            let root = descend(&keys, &order, 0, &mut nodes, &mut intern);
            if best
                .as_ref()
                .is_none_or(|(old, _): &(Vec<Node>, Reference)| nodes.len() < old.len())
            {
                best = Some((nodes, root));
            }
        }
        let (nodes, root) = best.unwrap();
        Ok(Self { id, nodes, root })
    }
}

fn tensor_children(rows: u64, columns: u64) -> ((u64, u64), (u64, u64)) {
    let split = |mask: u64| {
        let low = mask.trailing_zeros();
        let high = 63 - mask.leading_zeros();
        let width = 1u32 << (low ^ high).ilog2();
        let boundary = high / width * width;
        let left = mask & ((1u64 << boundary) - 1);
        (left, mask ^ left)
    };
    if !rows.is_power_of_two() {
        let (a, b) = split(rows);
        ((a, columns), (b, columns))
    } else {
        let (a, b) = split(columns);
        ((rows, a), (rows, b))
    }
}

#[derive(Default)]
struct Contractions {
    leaves: HashMap<usize, F128VariablesV1>,
    products: HashMap<(u64, u64), F128VariablesV1>,
}
impl Contractions {
    fn tensor(
        &mut self,
        builder: &mut R1csBuilder,
        rows: u64,
        columns: u64,
        row: &[F128VariablesV1],
        column: &[F128VariablesV1],
    ) -> Result<F128VariablesV1> {
        if let Some(value) = self.products.get(&(rows, columns)) {
            return Ok(value.clone());
        }
        let value = if rows.is_power_of_two() && columns.is_power_of_two() {
            constrain_f128_multiply(
                builder,
                &row[rows.trailing_zeros() as usize],
                &column[columns.trailing_zeros() as usize],
                PHASE,
            )?
        } else {
            let (a, b) = tensor_children(rows, columns);
            let a = self.tensor(builder, a.0, a.1, row, column)?;
            let b = self.tensor(builder, b.0, b.1, row, column)?;
            constrain_f128_add(builder, &a, &b, PHASE)?
        };
        self.products.insert((rows, columns), value.clone());
        Ok(value)
    }
    fn leaf(
        &mut self,
        builder: &mut R1csBuilder,
        index: usize,
        patterns: &[[u64; 64]],
        row: &[F128VariablesV1],
        column: &[F128VariablesV1],
    ) -> Result<F128VariablesV1> {
        if let Some(value) = self.leaves.get(&index) {
            return Ok(value.clone());
        }
        let mut groups = BTreeMap::<u64, u64>::new();
        for (r, &mask) in patterns[index].iter().enumerate() {
            if mask != 0 {
                *groups.entry(mask).or_default() |= 1u64 << r;
            }
        }
        let mut sum = alloc_f128_constant(builder, [0; 16], PHASE)?;
        for (columns, rows) in groups {
            let product = self.tensor(builder, rows, columns, row, column)?;
            sum = constrain_f128_add(builder, &sum, &product, PHASE)?;
        }
        self.leaves.insert(index, sum.clone());
        Ok(sum)
    }
}

pub struct FixedMatrices {
    matrices: Vec<Matrix>,
    patterns: Vec<[u64; 64]>,
}
impl FixedMatrices {
    pub fn new(registry: &Registry) -> Result<Self> {
        ensure!(
            registry.num_element() == 0,
            "fixed matrices support Boolean registries"
        );
        let digest = registry.digest();
        let mut patterns = vec![[0; 64]];
        let mut pool = HashMap::from([([0; 64], 0)]);
        let mut matrices = Vec::new();
        for (table, ty) in registry.boolean_types().iter().enumerate() {
            for (side, matrix) in [
                (F128MatrixSideV1::A, &ty.a_0),
                (F128MatrixSideV1::B, &ty.b_0),
            ] {
                matrices.push(Matrix::build(
                    F128StaticMatrixIdV1 {
                        registry_digest: digest,
                        table: table as u64,
                        side,
                        variables: ty.k_log as u32,
                    },
                    matrix,
                    &mut patterns,
                    &mut pool,
                )?);
            }
        }
        Ok(Self { matrices, patterns })
    }
    pub fn stats(&self) -> serde_json::Value {
        let mut pairs = std::collections::HashSet::new();
        for pattern in &self.patterns {
            let mut groups = BTreeMap::<u64, u64>::new();
            for (i, &columns) in pattern.iter().enumerate() {
                if columns != 0 {
                    *groups.entry(columns).or_default() |= 1u64 << i;
                }
            }
            pairs.extend(groups);
        }
        let mut schedule = std::collections::HashSet::new();
        let mut pending: Vec<_> = pairs
            .iter()
            .map(|&(columns, rows)| (rows, columns))
            .collect();
        while let Some((rows, columns)) = pending.pop() {
            if !schedule.insert((rows, columns)) {
                continue;
            }
            if !rows.is_power_of_two() || !columns.is_power_of_two() {
                let (a, b) = tensor_children(rows, columns);
                pending.extend([a, b]);
            }
        }
        let products = schedule
            .iter()
            .filter(|(r, c)| r.is_power_of_two() && c.is_power_of_two())
            .count();
        serde_json::json!({"claims":self.matrices.len(),"patterns":self.patterns.len(),"distinct_block_products":pairs.len(),"tensor_products":products,"tensor_additions":schedule.len()-products,
            "nodes":self.matrices.iter().map(|m|m.nodes.len()).sum::<usize>(),
            "matrices":self.matrices.iter().map(|m|serde_json::json!({"table":m.id.table,"side":format!("{:?}",m.id.side),"nodes":m.nodes.len()})).collect::<Vec<_>>()})
    }
    pub fn constrain(
        &self,
        builder: &mut R1csBuilder,
        claims: &[F128DeferredMatrixClaimVariablesV1],
    ) -> Result<()> {
        ensure!(
            claims.len() == self.matrices.len(),
            "fixed matrix claim count"
        );
        let mut contractions = HashMap::<Vec<u32>, Contractions>::new();
        for (matrix, claim) in self.matrices.iter().zip(claims) {
            ensure!(matrix.id == claim.matrix, "fixed matrix identity/order");
            ensure!(
                claim.row.low.len() == 64 && claim.column.low.len() == 64,
                "fixed matrix low weights"
            );
            let high = matrix.id.variables as usize - 6;
            ensure!(
                claim.row.point.len() == high && claim.column.point.len() == high,
                "fixed matrix high weights"
            );
            let key = claim
                .row
                .low
                .iter()
                .chain(&claim.column.low)
                .flat_map(|v| v.bit_variables().iter().map(|v| v.index()))
                .collect();
            let cache = contractions.entry(key).or_default();
            let point: Vec<_> = claim.row.point.iter().chain(&claim.column.point).collect();
            let mut values = Vec::with_capacity(matrix.nodes.len());
            let resolve = |builder: &mut R1csBuilder,
                           reference: Reference,
                           cache: &mut Contractions,
                           values: &Vec<F128VariablesV1>|
             -> Result<F128VariablesV1> {
                match reference {
                    Reference::Node(i) => Ok(values[i].clone()),
                    Reference::Leaf(i) => cache.leaf(
                        builder,
                        i,
                        &self.patterns,
                        &claim.row.low,
                        &claim.column.low,
                    ),
                }
            };
            for node in &matrix.nodes {
                let low = resolve(builder, node.low, cache, &values)?;
                let high = resolve(builder, node.high, cache, &values)?;
                let delta = constrain_f128_add(builder, &low, &high, PHASE)?;
                let product =
                    constrain_f128_multiply(builder, point[node.variable], &delta, PHASE)?;
                values.push(constrain_f128_add(builder, &low, &product, PHASE)?);
            }
            let actual = resolve(builder, matrix.root, cache, &values)?;
            enforce_f128_equal(builder, &actual, &claim.value, PHASE);
            ensure!(
                actual.value() == claim.value.value(),
                "fixed matrix evaluation mismatch"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use flock_prover::{
        field::F128,
        matrix_fold::{Weight, bilinear},
    };
    fn raw(v: F128) -> [u8; 16] {
        (u128::from(v.lo) | (u128::from(v.hi) << 64)).to_le_bytes()
    }
    fn weight(builder: &mut R1csBuilder, w: &Weight) -> F128StructuredWeightVariablesV1 {
        F128StructuredWeightVariablesV1 {
            low: w
                .low
                .iter()
                .map(|&v| alloc_f128_private(builder, raw(v), PHASE).unwrap())
                .collect(),
            point: w
                .point
                .iter()
                .map(|&v| alloc_f128_private(builder, raw(v), PHASE).unwrap())
                .collect(),
        }
    }
    #[test]
    fn block_contraction_matches_sparse_oracle_and_binds_every_claim() {
        let mut rows = vec![vec![]; 512];
        rows[0] = vec![0, 63, 64];
        rows[63] = vec![127];
        rows[64] = vec![0, 63, 64];
        rows[127] = vec![1, 1, 127];
        rows[169] = vec![350, 511];
        rows[511] = vec![169, 350];
        let native = SparseBinaryMatrix::new(512, 512, rows);
        let mut patterns = vec![[0; 64]];
        let mut pool = HashMap::from([([0; 64], 0)]);
        let ids = [F128MatrixSideV1::A, F128MatrixSideV1::B].map(|side| F128StaticMatrixIdV1 {
            registry_digest: [5; 32],
            table: 0,
            side,
            variables: 9,
        });
        let matrices = ids
            .iter()
            .map(|id| Matrix::build(*id, &native, &mut patterns, &mut pool).unwrap())
            .collect();
        let fixed = FixedMatrices { matrices, patterns };
        let mut builder = R1csBuilder::new();
        let mut claims = Vec::new();
        for (i, &id) in ids.iter().enumerate() {
            let row = Weight::low_eq(
                (0..64)
                    .map(|j| F128::new(j + 7 + i as u64, 2 * j + 1))
                    .collect(),
                vec![F128::new(3, 8), F128::new(4, 2), F128::new(6, 11)],
            );
            let column = Weight::low_eq(
                (0..64).map(|j| F128::new(j + 13, 4 * j + 9)).collect(),
                vec![F128::new(17, 6), F128::new(13, 9), F128::new(5, 14)],
            );
            let value = bilinear(&row, &column, &native);
            claims.push(F128DeferredMatrixClaimVariablesV1 {
                matrix: id,
                row: weight(&mut builder, &row),
                column: weight(&mut builder, &column),
                value: alloc_f128_private(&mut builder, raw(value), PHASE).unwrap(),
            });
        }
        fixed.constrain(&mut builder, &claims).unwrap();
        let (r, w) = builder.finish().unwrap();
        r.check(&w).unwrap();
        for claim in &claims {
            let mut bad = w.clone();
            let bit = claim.value.bit_variables()[0];
            bad.set(bit, Fr::from(1u64) - w.assignment()[bit.index() as usize])
                .unwrap();
            assert!(r.check(&bad).is_err());
        }
        let mut wrong = claims.clone();
        wrong[0].matrix.registry_digest[0] ^= 1;
        assert!(fixed.constrain(&mut R1csBuilder::new(), &wrong).is_err());
        assert!(
            fixed
                .constrain(&mut R1csBuilder::new(), &claims[..1])
                .is_err()
        );
    }
}
