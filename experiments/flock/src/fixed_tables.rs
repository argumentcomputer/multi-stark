//! Exact evaluation of verifier-owned fixed tables.
use anyhow::{Result, ensure};
use flock_prover::pcs::jagged::JaggedParams;
use ix_stage4_trace::F128JaggedMatrixIdV1;
use ix_terminal_circuit::*;
use std::collections::HashMap;

const PHASE: ConstraintPhase = ConstraintPhase::Pcs;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Node {
    variable: usize,
    low: usize,
    high: usize,
}

struct Diagram {
    nodes: Vec<Node>,
    root: usize,
}

impl Diagram {
    fn build(entries: &[u128], order: &[usize], _row_variables: usize) -> Self {
        let mut keys: Vec<u128> = entries
            .iter()
            .map(|entry| {
                order
                    .iter()
                    .fold(0, |key, bit| (key << 1) | ((entry >> bit) & 1))
            })
            .collect();
        keys.sort_unstable();
        keys.dedup();
        let mut nodes = Vec::new();
        let mut intern = HashMap::new();
        fn descend(
            keys: &[u128],
            order: &[usize],
            level: usize,
            nodes: &mut Vec<Node>,
            intern: &mut HashMap<Node, usize>,
        ) -> usize {
            if keys.is_empty() {
                return 0;
            }
            if level == order.len() {
                return 1;
            }
            let mask = 1u128 << (order.len() - level - 1);
            let middle = keys.partition_point(|key| key & mask == 0);
            let low = descend(&keys[..middle], order, level + 1, nodes, intern);
            let high = descend(&keys[middle..], order, level + 1, nodes, intern);
            if low == high {
                return low;
            }
            let node = Node {
                variable: order[level],
                low,
                high,
            };
            *intern.entry(node).or_insert_with(|| {
                let id = nodes.len() + 2;
                nodes.push(node);
                id
            })
        }
        let root = descend(&keys, order, 0, &mut nodes, &mut intern);
        Self { nodes, root }
    }

    fn evaluate(
        &self,
        builder: &mut R1csBuilder,
        point: &[F128VariablesV1],
    ) -> Result<F128VariablesV1> {
        let mut values = constants(builder)?;
        for node in &self.nodes {
            let delta = constrain_f128_add(builder, &values[node.low], &values[node.high], PHASE)?;
            let product = constrain_f128_multiply(builder, &point[node.variable], &delta, PHASE)?;
            values.push(constrain_f128_add(
                builder,
                &values[node.low],
                &product,
                PHASE,
            )?);
        }
        Ok(values[self.root].clone())
    }
}

fn constants(builder: &mut R1csBuilder) -> Result<Vec<F128VariablesV1>> {
    Ok(vec![
        alloc_f128_constant(builder, 0u128.to_le_bytes(), PHASE)?,
        alloc_f128_constant(builder, 1u128.to_le_bytes(), PHASE)?,
    ])
}

struct BoundaryProducts<'a> {
    point: &'a [F128VariablesV1],
    one: F128VariablesV1,
    cache: HashMap<(u128, u128), F128VariablesV1>,
}
impl BoundaryProducts<'_> {
    fn product(
        &mut self,
        builder: &mut R1csBuilder,
        mask: u128,
        boundary: u128,
    ) -> Result<F128VariablesV1> {
        if mask == 0 {
            return Ok(self.one.clone());
        }
        let key = (mask, boundary & mask);
        if let Some(value) = self.cache.get(&key) {
            return Ok(value.clone());
        }
        let value = if mask.is_power_of_two() {
            let coordinate = &self.point[mask.trailing_zeros() as usize];
            if boundary & mask == 0 {
                constrain_f128_add(builder, &self.one, coordinate, PHASE)?
            } else {
                coordinate.clone()
            }
        } else {
            let low = mask.trailing_zeros();
            let high = 127 - mask.leading_zeros();
            let width = 1u32 << (low ^ high).ilog2();
            let split = high / width * width;
            let left = mask & ((1u128 << split) - 1);
            let a = self.product(builder, left, boundary)?;
            let b = self.product(builder, mask ^ left, boundary)?;
            constrain_f128_multiply(builder, &a, &b, PHASE)?
        };
        self.cache.insert(key, value.clone());
        Ok(value)
    }
}

pub struct FixedLayout {
    id: F128JaggedMatrixIdV1,
    diagram: Diagram,
    shift: usize,
    boundaries: Vec<u128>,
}

impl FixedLayout {
    /// Call with circuit-derived parameters, never proof-supplied boundaries.
    pub fn new(circuit_digest: [u8; 32], params: &JaggedParams) -> Result<Self> {
        let k = params.k;
        let pairs = 2 * (params.m + 1);
        ensure!(
            k < 32 && params.m < 63 && k + pairs <= 128,
            "layout dimensions"
        );
        ensure!(
            params.col_prefix_sums.len() == (1usize << k) + 1,
            "layout length"
        );
        ensure!(params.col_prefix_sums[0] == 0, "layout origin");
        ensure!(
            params.col_prefix_sums.windows(2).all(|w| w[0] <= w[1]),
            "layout order"
        );
        ensure!(
            *params.col_prefix_sums.last().unwrap() <= 1u64 << params.m,
            "layout area"
        );
        let entries: Vec<_> = params
            .col_prefix_sums
            .windows(2)
            .enumerate()
            .map(|(row, bounds)| {
                let mut entry = row as u128;
                for bit in 0..=params.m {
                    entry |= u128::from((bounds[0] >> bit) & 1) << (k + 2 * bit);
                    entry |= u128::from((bounds[1] >> bit) & 1) << (k + 2 * bit + 1);
                }
                entry
            })
            .collect();
        // Interleave address bits with the boundary bits they commonly control.
        // Selection uses only the fixed layout and cannot depend on proof values.
        let mut best: Option<(usize, Diagram)> = None;
        for shift in 0..=params.m + 1 {
            let mut order = Vec::new();
            for bit in 0..=(params.m + 1).max(shift + k) {
                if bit >= shift && bit - shift < k {
                    order.push(bit - shift);
                }
                if bit <= params.m {
                    order.extend([k + 2 * bit, k + 2 * bit + 1]);
                }
            }
            let diagram = Diagram::build(&entries, &order, k);
            if best
                .as_ref()
                .is_none_or(|(_, old)| diagram.nodes.len() < old.nodes.len())
            {
                best = Some((shift, diagram));
            }
        }
        let (shift, diagram) = best.unwrap();
        Ok(Self {
            id: F128JaggedMatrixIdV1 {
                circuit_digest,
                row_variables: k as u32,
                column_variables: pairs as u32,
            },
            diagram,
            shift,
            boundaries: entries.iter().map(|entry| entry >> k).collect(),
        })
    }

    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({"nodes":self.diagram.nodes.len(), "shift":self.shift,
            "row_variables":self.id.row_variables, "column_variables":self.id.column_variables})
    }

    pub fn constrain(
        &self,
        builder: &mut R1csBuilder,
        assertion: &F128JaggedAssertionVariablesV1,
    ) -> Result<()> {
        ensure!(assertion.matrix == self.id, "fixed layout identity");
        ensure!(assertion.claims.len() == 3, "fixed layout claim count");
        let k = self.id.row_variables as usize;
        for claim in &assertion.claims {
            ensure!(
                claim.column_point.len() == self.id.column_variables as usize,
                "layout column point"
            );
            let computed = match &claim.row {
                F128JaggedRowWeightVariablesV1::Eq { scale, point } => {
                    ensure!(point.len() == k, "layout row point");
                    let mut combined = point.clone();
                    combined.extend_from_slice(&claim.column_point);
                    let value = self.diagram.evaluate(builder, &combined)?;
                    constrain_f128_multiply(builder, scale, &value, PHASE)?
                }
                F128JaggedRowWeightVariablesV1::Combo { terms } => {
                    let constants = constants(builder)?;
                    let mut groups = std::collections::BTreeMap::new();
                    for term in terms {
                        ensure!(u64::from(term.address) < 1u64 << k, "layout combo address");
                        let boundary = self.boundaries[term.address as usize];
                        let entry = groups
                            .entry(boundary)
                            .or_insert_with(|| constants[0].clone());
                        *entry = constrain_f128_add(builder, entry, &term.coefficient, PHASE)?;
                    }
                    let first = groups.keys().next().copied().unwrap_or(0);
                    let varying = groups
                        .keys()
                        .fold(0, |mask, boundary| mask | (boundary ^ first));
                    let all = u128::MAX >> (128 - claim.column_point.len());
                    let mut products = BoundaryProducts {
                        point: &claim.column_point,
                        one: constants[1].clone(),
                        cache: HashMap::new(),
                    };
                    let common = products.product(builder, all ^ varying, first)?;
                    let mut sum = constants[0].clone();
                    for (boundary, coefficient) in groups {
                        let value = products.product(builder, varying, boundary)?;
                        let product =
                            constrain_f128_multiply(builder, &coefficient, &value, PHASE)?;
                        sum = constrain_f128_add(builder, &sum, &product, PHASE)?;
                    }
                    constrain_f128_multiply(builder, &common, &sum, PHASE)?
                }
            };
            enforce_f128_equal(builder, &computed, &claim.value, PHASE);
            ensure!(
                computed.value() == claim.value.value(),
                "fixed layout evaluation mismatch"
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
        matrix_fold::{JaggedRowWeight, JaggedTable, jagged_bilinear},
    };

    fn raw(value: F128) -> [u8; 16] {
        (u128::from(value.lo) | (u128::from(value.hi) << 64)).to_le_bytes()
    }
    fn alloc(builder: &mut R1csBuilder, value: F128) -> F128VariablesV1 {
        alloc_f128_private(builder, raw(value), PHASE).unwrap()
    }

    #[test]
    fn diagram_matches_truth_table_for_every_variable_order() {
        let entries = [0, 1, 7, 8, 9, 15];
        for order in [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]] {
            let diagram = Diagram::build(&entries, &order, 2);
            for input in 0..16 {
                let mut values = vec![false, true];
                for node in &diagram.nodes {
                    values.push(
                        values[if (input >> node.variable) & 1 == 0 {
                            node.low
                        } else {
                            node.high
                        }],
                    );
                }
                assert_eq!(values[diagram.root], entries.contains(&input));
            }
        }
    }

    #[test]
    fn factored_boundary_products_match_native_for_shared_and_empty_combinations() {
        let heights: Vec<_> = (0..16).map(|i| (i * 7) % 5).collect();
        let params = JaggedParams::from_heights(&heights, 2, 7);
        let fixed = FixedLayout::new([3; 32], &params).unwrap();
        let table = JaggedTable::from_params(&params);
        let column: Vec<_> = (0..16).map(|i| F128::new(i * 3 + 7, i * 5 + 11)).collect();
        let combinations = [
            vec![],
            vec![(F128::new(7, 5), 3), (F128::new(9, 11), 3)],
            (0..16)
                .map(|i| (F128::new(3 * i + 1, 7 * i + 3), i as u32))
                .collect(),
        ];
        let mut builder = R1csBuilder::new();
        let column_point: Vec<_> = column.iter().map(|&v| alloc(&mut builder, v)).collect();
        let claims = combinations
            .iter()
            .map(|terms| {
                let native = JaggedRowWeight::Combo(terms.clone());
                let value = alloc(&mut builder, jagged_bilinear(&native, &column, &table));
                F128JaggedClaimVariablesV1 {
                    row: F128JaggedRowWeightVariablesV1::Combo {
                        terms: terms
                            .iter()
                            .map(|&(coefficient, address)| F128JaggedComboTermVariablesV1 {
                                coefficient: alloc(&mut builder, coefficient),
                                address,
                            })
                            .collect(),
                    },
                    column_point: column_point.clone(),
                    value,
                }
            })
            .collect();
        fixed
            .constrain(
                &mut builder,
                &F128JaggedAssertionVariablesV1 {
                    matrix: fixed.id,
                    claims,
                },
            )
            .unwrap();
        let (r1cs, witness) = builder.finish().unwrap();
        r1cs.check(&witness).unwrap();
    }

    #[test]
    fn fixed_layout_shape_does_not_depend_on_private_values() {
        let params = JaggedParams::from_heights(&[1, 0, 2, 1], 1, 3);
        let fixed = FixedLayout::new([4; 32], &params).unwrap();
        let table = JaggedTable::from_params(&params);
        let mut digests = Vec::new();
        for seed in [0, 17] {
            let column: Vec<_> = (0..8)
                .map(|i| F128::new(seed * (i + 1), seed * (i + 3)))
                .collect();
            let rows = [
                JaggedRowWeight::Eq(F128::new(seed, 0), vec![F128::new(seed, seed); 2]),
                JaggedRowWeight::Combo(vec![
                    (F128::new(seed, seed), 0),
                    (F128::new(seed + 1, seed), 2),
                ]),
                JaggedRowWeight::Combo(vec![(F128::new(seed, 1), 3)]),
            ];
            let mut builder = R1csBuilder::new();
            let column_point: Vec<_> = column.iter().map(|&v| alloc(&mut builder, v)).collect();
            let claims = rows
                .iter()
                .map(|row| {
                    let value = alloc(&mut builder, jagged_bilinear(row, &column, &table));
                    let row = match row {
                        JaggedRowWeight::Eq(scale, point) => F128JaggedRowWeightVariablesV1::Eq {
                            scale: Box::new(alloc(&mut builder, *scale)),
                            point: point.iter().map(|&v| alloc(&mut builder, v)).collect(),
                        },
                        JaggedRowWeight::Combo(terms) => F128JaggedRowWeightVariablesV1::Combo {
                            terms: terms
                                .iter()
                                .map(|&(coefficient, address)| F128JaggedComboTermVariablesV1 {
                                    coefficient: alloc(&mut builder, coefficient),
                                    address,
                                })
                                .collect(),
                        },
                    };
                    F128JaggedClaimVariablesV1 {
                        row,
                        column_point: column_point.clone(),
                        value,
                    }
                })
                .collect();
            fixed
                .constrain(
                    &mut builder,
                    &F128JaggedAssertionVariablesV1 {
                        matrix: fixed.id,
                        claims,
                    },
                )
                .unwrap();
            let (r1cs, witness) = builder.finish().unwrap();
            r1cs.check(&witness).unwrap();
            digests.push(r1cs.digest());
        }
        assert_eq!(digests[0], digests[1]);
    }

    #[test]
    fn fixed_layout_checks_native_claims_and_rejects_mutation() {
        let params = JaggedParams::from_heights(&[1, 0, 1, 0], 0, 1);
        let fixed = FixedLayout::new([9; 32], &params).unwrap();
        let table = JaggedTable::from_params(&params);
        let column: Vec<_> = (0..4).map(|i| F128::new(9 + i, 3 + i)).collect();
        let native_rows = [
            JaggedRowWeight::Eq(F128::new(7, 2), vec![F128::new(3, 9), F128::new(5, 1)]),
            JaggedRowWeight::Eq(F128::ZERO, vec![F128::new(4, 8), F128::new(6, 2)]),
            JaggedRowWeight::Combo(vec![
                (F128::new(11, 3), 1),
                (F128::new(17, 9), 2),
                (F128::new(5, 1), 1),
            ]),
        ];
        let mut builder = R1csBuilder::new();
        let column_point: Vec<_> = column.iter().map(|&v| alloc(&mut builder, v)).collect();
        let mut claims = Vec::new();
        for row in &native_rows {
            let value = alloc(&mut builder, jagged_bilinear(row, &column, &table));
            let row = match row {
                JaggedRowWeight::Eq(scale, point) => F128JaggedRowWeightVariablesV1::Eq {
                    scale: Box::new(alloc(&mut builder, *scale)),
                    point: point.iter().map(|&v| alloc(&mut builder, v)).collect(),
                },
                JaggedRowWeight::Combo(terms) => F128JaggedRowWeightVariablesV1::Combo {
                    terms: terms
                        .iter()
                        .map(|&(coefficient, address)| F128JaggedComboTermVariablesV1 {
                            coefficient: alloc(&mut builder, coefficient),
                            address,
                        })
                        .collect(),
                },
            };
            claims.push(F128JaggedClaimVariablesV1 {
                row,
                column_point: column_point.clone(),
                value,
            });
        }
        let assertion = F128JaggedAssertionVariablesV1 {
            matrix: fixed.id,
            claims,
        };
        fixed.constrain(&mut builder, &assertion).unwrap();
        let (r1cs, witness) = builder.finish().unwrap();
        r1cs.check(&witness).unwrap();
        for claim in &assertion.claims {
            let mut bad = witness.clone();
            let bit = claim.value.bit_variables()[0];
            bad.set(
                bit,
                Fr::from(1u64) - witness.assignment()[bit.index() as usize],
            )
            .unwrap();
            assert!(r1cs.check(&bad).is_err());
        }
        let mut wrong_identity = assertion.clone();
        wrong_identity.matrix.circuit_digest[0] ^= 1;
        assert!(
            fixed
                .constrain(&mut R1csBuilder::new(), &wrong_identity)
                .is_err()
        );
        let mut missing = assertion.clone();
        missing.claims.pop();
        assert!(fixed.constrain(&mut R1csBuilder::new(), &missing).is_err());
    }
}

pub struct FixedLiveMask {
    id: ix_stage4_trace::F128CircuitStructureMatrixIdV1,
    diagram: Diagram,
}

impl FixedLiveMask {
    pub fn new(
        digest: [u8; 32],
        mask: &flock_prover::product_gkr::LiveMask,
        base_bits: usize,
    ) -> Result<Self> {
        ensure!(
            mask.nu < 63 && mask.nu + base_bits <= 64 && mask.counts.len().is_power_of_two(),
            "live-mask dimensions"
        );
        ensure!(
            mask.counts.len().ilog2() as usize <= base_bits,
            "live-mask slots"
        );
        ensure!(
            mask.counts.iter().all(|&n| n <= 1usize << mask.nu),
            "live-mask count"
        );
        let mut nodes = Vec::new();
        let mut intern = HashMap::new();
        fn branch(
            nodes: &mut Vec<Node>,
            intern: &mut HashMap<Node, usize>,
            variable: usize,
            low: usize,
            high: usize,
        ) -> usize {
            if low == high {
                return low;
            }
            let node = Node {
                variable,
                low,
                high,
            };
            *intern.entry(node).or_insert_with(|| {
                let id = nodes.len() + 2;
                nodes.push(node);
                id
            })
        }
        fn prefix(
            nodes: &mut Vec<Node>,
            intern: &mut HashMap<Node, usize>,
            bits: usize,
            n: usize,
        ) -> usize {
            if n == 0 {
                return 0;
            }
            if n == 1usize << bits {
                return 1;
            }
            let half = 1usize << (bits - 1);
            let (low, high) = if n <= half {
                (prefix(nodes, intern, bits - 1, n), 0)
            } else {
                (1, prefix(nodes, intern, bits - 1, n - half))
            };
            branch(nodes, intern, bits - 1, low, high)
        }
        let mut level: Vec<_> = mask
            .counts
            .iter()
            .map(|&n| prefix(&mut nodes, &mut intern, mask.nu, n))
            .collect();
        let slot_bits = mask.counts.len().ilog2() as usize;
        for bit in 0..slot_bits {
            level = level
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| branch(&mut nodes, &mut intern, mask.nu + bit, pair[0], pair[1]))
                .collect();
        }
        let mut root = level[0];
        for bit in slot_bits..base_bits {
            root = branch(&mut nodes, &mut intern, mask.nu + bit, root, 0);
        }
        Ok(Self {
            id: ix_stage4_trace::F128CircuitStructureMatrixIdV1 {
                circuit_digest: digest,
                row_variables: mask.nu as u32,
                column_variables: (base_bits + 3) as u32,
            },
            diagram: Diagram { nodes, root },
        })
    }

    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({"nodes":self.diagram.nodes.len(),"claims":2})
    }

    fn evaluate(
        &self,
        builder: &mut R1csBuilder,
        point: &[F128VariablesV1],
    ) -> Result<(F128VariablesV1, F128VariablesV1)> {
        let mut values = constants(builder)?;
        let zero = values[0].clone();
        let mut tags = vec![zero.clone(), zero.clone()];
        let mut id_prefix = vec![zero.clone()];
        for (bit, value) in point.iter().enumerate() {
            let term = constrain_f128_multiply_constant(
                builder,
                value,
                (1u128 << bit).to_le_bytes(),
                PHASE,
            )?;
            id_prefix.push(constrain_f128_add(
                builder,
                id_prefix.last().unwrap(),
                &term,
                PHASE,
            )?);
        }
        let extend = |builder: &mut R1csBuilder,
                      id: usize,
                      end: usize,
                      values: &[F128VariablesV1],
                      tags: &[F128VariablesV1]|
         -> Result<F128VariablesV1> {
            let start = if id < 2 {
                0
            } else {
                self.diagram.nodes[id - 2].variable + 1
            };
            if start == end {
                return Ok(tags[id].clone());
            }
            let gap = constrain_f128_add(builder, &id_prefix[end], &id_prefix[start], PHASE)?;
            let gap = constrain_f128_multiply(builder, &gap, &values[id], PHASE)?;
            Ok(constrain_f128_add(builder, &tags[id], &gap, PHASE)?)
        };
        for node in &self.diagram.nodes {
            let low_tag = extend(builder, node.low, node.variable, &values, &tags)?;
            let high_tag = extend(builder, node.high, node.variable, &values, &tags)?;
            let high_bit = constrain_f128_multiply_constant(
                builder,
                &values[node.high],
                (1u128 << node.variable).to_le_bytes(),
                PHASE,
            )?;
            let high_tag = constrain_f128_add(builder, &high_tag, &high_bit, PHASE)?;
            let tag_delta = constrain_f128_add(builder, &low_tag, &high_tag, PHASE)?;
            let tag_delta =
                constrain_f128_multiply(builder, &point[node.variable], &tag_delta, PHASE)?;
            tags.push(constrain_f128_add(builder, &low_tag, &tag_delta, PHASE)?);
            let delta = constrain_f128_add(builder, &values[node.low], &values[node.high], PHASE)?;
            let delta = constrain_f128_multiply(builder, &point[node.variable], &delta, PHASE)?;
            values.push(constrain_f128_add(
                builder,
                &values[node.low],
                &delta,
                PHASE,
            )?);
        }
        let tag = extend(builder, self.diagram.root, point.len(), &values, &tags)?;
        Ok((values[self.diagram.root].clone(), tag))
    }

    pub fn constrain(
        &self,
        builder: &mut R1csBuilder,
        claims: &[F128CircuitStructureClaimVariablesV1],
    ) -> Result<()> {
        ensure!(claims.len() == 3, "structure claim count");
        let base = self.id.column_variables as usize - 3;
        for (plane, claim) in claims[..2].iter().enumerate() {
            ensure!(claim.matrix == self.id, "live-mask identity");
            ensure!(
                claim.row_point.len() == self.id.row_variables as usize
                    && claim.column_point.len() == base + 3,
                "live-mask point"
            );
            for (bit, variable) in claim.column_point[base..].iter().enumerate() {
                let expected = alloc_f128_constant(
                    builder,
                    (((plane >> bit) & 1) as u128).to_le_bytes(),
                    PHASE,
                )?;
                enforce_f128_equal(builder, variable, &expected, PHASE);
            }
        }
        let point: Vec<_> = claims[0]
            .row_point
            .iter()
            .chain(&claims[0].column_point[..base])
            .cloned()
            .collect();
        let other: Vec<_> = claims[1]
            .row_point
            .iter()
            .chain(&claims[1].column_point[..base])
            .collect();
        ensure!(
            point
                .iter()
                .zip(other)
                .all(|(a, b)| a.bit_variables() == b.bit_variables()),
            "fresh structure points differ"
        );
        let (live, tag) = self.evaluate(builder, &point)?;
        enforce_f128_equal(builder, &tag, &claims[0].value, PHASE);
        enforce_f128_equal(builder, &live, &claims[1].value, PHASE);
        ensure!(
            tag.value() == claims[0].value.value() && live.value() == claims[1].value.value(),
            "live-mask evaluation mismatch"
        );
        Ok(())
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;
    use ark_bls12_381::Fr;
    use flock_prover::{
        field::F128,
        product_gkr::{LiveMask, s_id_basis},
    };

    #[test]
    fn live_and_masked_id_match_native_with_skipped_bits_and_padding() {
        for counts in [
            vec![0, 0, 0, 0],
            vec![4, 4, 4, 4],
            vec![4, 0, 3, 1],
            vec![2, 2, 2, 2],
        ] {
            let mask = LiveMask { nu: 2, counts };
            let fixed = FixedLiveMask::new([7; 32], &mask, 3).unwrap();
            let mut padded = mask.clone();
            padded.counts.resize(8, 0);
            let native: Vec<_> = (0..5).map(|i| F128::new(i + 3, 2 * i + 9)).collect();
            let mut b = R1csBuilder::new();
            let point: Vec<_> = native
                .iter()
                .map(|v| {
                    alloc_f128_private(
                        &mut b,
                        (u128::from(v.lo) | (u128::from(v.hi) << 64)).to_le_bytes(),
                        PHASE,
                    )
                    .unwrap()
                })
                .collect();
            let (live, tag) = fixed.evaluate(&mut b, &point).unwrap();
            let expected = padded.live_eval(&native);
            assert_eq!(
                *live.value(),
                (u128::from(expected.lo) | (u128::from(expected.hi) << 64)).to_le_bytes()
            );
            let expected = padded.masked_id_eval(&s_id_basis(5), &native);
            assert_eq!(
                *tag.value(),
                (u128::from(expected.lo) | (u128::from(expected.hi) << 64)).to_le_bytes()
            );
            let (r, w) = b.finish().unwrap();
            r.check(&w).unwrap();
            for value in [live, tag] {
                let bit = value.bit_variables()[0];
                let mut wrong = w.clone();
                wrong
                    .set(bit, Fr::from(1u64) - w.assignment()[bit.index() as usize])
                    .unwrap();
                assert!(r.check(&wrong).is_err());
            }
        }
    }
}
