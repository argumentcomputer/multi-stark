//! Goldilocks constraints over the BLS12-381 scalar field.
//! Each modular relation has a bounded quotient; small integer relations lift directly.

use num_bigint::{BigInt, BigUint, Sign};
use p3_field::PrimeField64;

use super::builder::Recipe;
use super::{Assignment, Circuit, CircuitBuilder, Gate, Table, Value, Witness, WitnessError};
use super::{LoweringError, MultiStarkCircuit};
use crate::ark_adapter::field::Scalar;
use crate::system::CircuitInputs;
use crate::traits::{Algebra, Field};
use crate::types::Val;
use p3_matrix::Matrix;

const P: u64 = <Val as PrimeField64>::ORDER_U64;

impl MultiStarkCircuit<Scalar> {
    /// Minimize KZG commitment/evaluation bytes by choosing lookup groups per
    /// trace, within the supplied quotient budget. Include the actual SRS length
    /// because smaller traces also need shifted degree commitments.
    pub fn kzg_circuit_inputs(
        &self,
        srs_len: usize,
        quotient_budget: usize,
    ) -> Result<Vec<CircuitInputs<Scalar>>, LoweringError> {
        use crate::{
            expr::CircuitSpec,
            graph::{ExtensionParams, compile},
            lookup::{MAX_LOOKUP_GROUP, logup_max_degree, stage2_width},
        };
        let mut inputs = self.circuit_inputs();
        for input in &mut inputs {
            let fixed = input
                .preprocessed
                .as_ref()
                .expect("lowering has fixed columns");
            if fixed.height() > srs_len {
                return Err(LoweringError::InvalidTraceHeight);
            }
            let graph = compile(
                &CircuitSpec {
                    main_width: input.main_width,
                    preprocessed_width: fixed.width(),
                    stage2_width: stage2_width(input.lookups.len(), 1, 1),
                    num_publics: 4,
                    constraints: input.constraints.clone(),
                    ext_constraints: input.ext_constraints.clone(),
                    lookups: input.lookups.clone(),
                },
                &ExtensionParams {
                    degree: 1,
                    w: Scalar::ZERO,
                    karatsuba: false,
                },
            )
            .expect("valid lowering");
            let commitment_bytes = if fixed.height() == srs_len { 48 } else { 96 };
            let best = (1..=MAX_LOOKUP_GROUP)
                .filter_map(|group| {
                    let degree = graph
                        .max_constraint_degree
                        .max(logup_max_degree(&graph, group))
                        as usize;
                    let quotient = (degree.max(2) - 1).next_power_of_two();
                    if quotient > quotient_budget {
                        return None;
                    }
                    let width = stage2_width(input.lookups.len(), group, 1);
                    let bytes =
                        width * (commitment_bytes + 64) + quotient * (commitment_bytes + 32);
                    Some(((bytes, quotient, group), group))
                })
                .min_by_key(|(cost, _)| *cost)
                .ok_or(LoweringError::QuotientBudgetTooSmall)?;
            input.lookup_group_size = best.1;
        }
        Ok(inputs)
    }
}

/// A fixed translation. Public values retain their original order and canonical encoding.
pub struct GoldilocksCircuit {
    pub circuit: Circuit<Scalar>,
    pub inputs: GoldilocksInputs,
}

/// Retain this mapping when consuming the circuit for lowering, to reuse its key.
pub struct GoldilocksInputs {
    source_owner: u64,
    wires: Vec<Value>,
    is_input: Vec<bool>,
}

fn signed(value: Val) -> i128 {
    let n = value.as_canonical_u64();
    if n > P / 2 {
        i128::from(n) - i128::from(P)
    } else {
        i128::from(n)
    }
}

fn scalar(n: &BigInt) -> Scalar {
    let (sign, limbs) = n.to_u64_digits();
    let radix = Scalar::from_u64(u64::MAX) + Scalar::ONE;
    let v = limbs
        .iter()
        .rev()
        .fold(Scalar::ZERO, |a, &b| a * radix + Scalar::from_u64(b));
    if sign == Sign::Minus { -v } else { v }
}

fn integer(value: Scalar) -> BigInt {
    let bytes: Vec<_> = value
        .canonical_limbs_le()
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect();
    BigInt::from_biguint(Sign::Plus, BigUint::from_bytes_le(&bytes))
}

fn interval(coefficients: [i128; 5], bounds: [u64; 3]) -> (BigInt, BigInt) {
    let [a, b, c] = bounds.map(BigInt::from);
    let maxima = [&a * &b, a, b, c, BigInt::from(1)];
    let mut low = BigInt::from(0);
    let mut high = BigInt::from(0);
    for (coefficient, maximum) in coefficients.into_iter().zip(maxima) {
        let term = coefficient * maximum;
        if coefficient < 0 {
            low += term;
        } else {
            high += term;
        }
    }
    (low, high)
}

/// Range checks use one shared 16-bit table. Rounding up the bit count remains
/// sound: even the largest gate and quotient products are below 2^209 < Fr.
fn range(b: &mut CircuitBuilder<Scalar>, table: Table, value: Value, bits: usize) {
    if bits == 0 {
        let zero = b.constant(Scalar::ZERO);
        b.assert_equal(value, zero);
        return;
    }
    if bits == 1 {
        b.assert_bool(value);
        return;
    }
    let count = bits.div_ceil(16);
    let mut terms = Vec::with_capacity(count);
    let radix = Scalar::from_u32(1 << 16);
    let mut coefficient = Scalar::ONE;
    for i in 0..count {
        let limb = b.hint("integer limb", &[value], move |v| {
            let limbs = v[0].canonical_limbs_le();
            Ok(Scalar::from_u64((limbs[i / 4] >> (16 * (i % 4))) & 0xffff))
        });
        b.lookup(table, &[limb]);
        terms.push((coefficient, limb));
        coefficient *= radix;
    }
    let packed = b.linear_combination(&terms, Scalar::ZERO);
    b.assert_equal(value, packed);
}

fn canonical(b: &mut CircuitBuilder<Scalar>, table: Table, value: Value) {
    let hi = b.hint("Goldilocks high word", &[value], |v| {
        Ok(Scalar::from_u64(v[0].canonical_low_u64() >> 32))
    });
    let lo = b.hint("Goldilocks low word", &[value], |v| {
        Ok(Scalar::from_u32(
            u32::try_from(v[0].canonical_low_u64() & u64::from(u32::MAX)).unwrap(),
        ))
    });
    range(b, table, hi, 32);
    range(b, table, lo, 32);
    let packed = b.affine(
        [hi, lo],
        [Scalar::from_u64(1 << 32), Scalar::ONE],
        Scalar::ZERO,
    );
    b.assert_equal(value, packed);
    // x < 0xffffffff00000001 iff the high word is not 0xffffffff or lo = 0.
    let delta = b.affine(
        [hi, hi],
        [Scalar::ONE, Scalar::ZERO],
        -Scalar::from_u32(u32::MAX),
    );
    let max_hi = b.is_zero(delta);
    let invalid = b.mul(max_hi.value(), lo);
    let zero = b.constant(Scalar::ZERO);
    b.assert_equal(invalid, zero);
}

impl GoldilocksCircuit {
    pub fn new(source: &Circuit<Val>) -> Self {
        let mut b = CircuitBuilder::<Scalar>::new();
        let range_table = b.fixed_table(
            "u16",
            (0..=u16::MAX).map(|v| vec![Scalar::from_u16(v)]).collect(),
        );
        let mut bounds = vec![P - 1; source.num_values()];
        let mut checked = vec![false; source.num_values()];
        // Membership checks give independently enforced bounds.
        for lookup in &source.lookups {
            let table = &source.tables[lookup.table.index];
            for (column, value) in lookup.values.iter().enumerate() {
                let maximum = table
                    .rows
                    .iter()
                    .map(|r| r[column].as_canonical_u64())
                    .max()
                    .unwrap();
                bounds[value.index] = bounds[value.index].min(maximum);
                checked[value.index] = true;
            }
        }
        if let Some(hashes) = &source.hashes {
            for call in &hashes.calls {
                for value in call.input.iter().chain(&call.output) {
                    bounds[value.index] = bounds[value.index].min(255);
                    checked[value.index] = true;
                }
            }
        }
        let mut wires = Vec::with_capacity(source.num_values());
        let mut inputs = Vec::with_capacity(source.num_values());
        for (i, recipe) in source.recipes.iter().enumerate() {
            if let Recipe::Constant(value) = recipe {
                let n = value.as_canonical_u64();
                bounds[i] = n;
                checked[i] = true;
                wires.push(b.constant(Scalar::from_u64(n)));
                inputs.push(false);
            } else {
                // Acyclic arithmetic with no modular wrap derives a range from
                // earlier wires. The translated gate enforces that equality.
                if let Recipe::Arithmetic(gate) = recipe {
                    let gate = &source.gates[*gate];
                    let mut coefficients = gate.coefficients.map(signed);
                    coefficients[3] = 0;
                    let (low, high) = interval(coefficients, gate.wires.map(|w| bounds[w.index]));
                    if low >= BigInt::from(0) && high < BigInt::from(P) {
                        let (_, digits) = high.to_u64_digits();
                        bounds[i] = bounds[i].min(digits.first().copied().unwrap_or(0));
                        checked[i] = true;
                    }
                }
                let value = b.input(format!("goldilocks[{i}]"));
                if !checked[i] {
                    canonical(&mut b, range_table, value);
                }
                wires.push(value);
                inputs.push(true);
            }
        }
        let tables: Vec<_> = source
            .tables
            .iter()
            .map(|table| {
                b.fixed_table(
                    table.name(),
                    table
                        .rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(|v| Scalar::from_u64(v.as_canonical_u64()))
                                .collect()
                        })
                        .collect(),
                )
            })
            .collect();
        for lookup in &source.lookups {
            b.lookup(
                tables[lookup.table.index],
                &lookup
                    .values
                    .iter()
                    .map(|v| wires[v.index])
                    .collect::<Vec<_>>(),
            );
        }
        if let Some(hashes) = &source.hashes {
            b.enable_compact_blake3();
            for call in &hashes.calls {
                b.record_hash(
                    call.input.iter().map(|v| wires[v.index]).collect(),
                    &call.output.map(|v| wires[v.index]),
                );
            }
        }
        for gate in &source.gates {
            constrain_gate(&mut b, range_table, gate, &wires, &bounds);
        }
        for value in &source.publics {
            b.expose_public(wires[value.index]);
        }
        Self {
            circuit: b.finish(),
            inputs: GoldilocksInputs {
                source_owner: source.owner,
                wires,
                is_input: inputs,
            },
        }
    }

    pub fn assign(
        &self,
        source: &Assignment<Val>,
        witness: &mut Witness<'_, Scalar>,
    ) -> Result<(), WitnessError> {
        self.inputs.assign(source, witness)
    }
}

impl GoldilocksInputs {
    pub fn assign(
        &self,
        source: &Assignment<Val>,
        witness: &mut Witness<'_, Scalar>,
    ) -> Result<(), WitnessError> {
        if source.owner != self.source_owner {
            return Err(WitnessError::ForeignAssignment);
        }
        for (i, &input) in self.is_input.iter().enumerate() {
            if input {
                witness.set(
                    self.wires[i],
                    Scalar::from_u64(source.values[i].as_canonical_u64()),
                )?;
            }
        }
        Ok(())
    }
}

fn constrain_gate(
    b: &mut CircuitBuilder<Scalar>,
    table: Table,
    gate: &Gate<Val>,
    wires: &[Value],
    bounds: &[u64],
) {
    let coefficients = gate.coefficients.map(signed);
    let values = gate.wires.map(|v| wires[v.index]);
    let (low, high) = interval(coefficients, gate.wires.map(|v| bounds[v.index]));
    let p = BigInt::from(P);
    if low > -&p && high < p {
        b.constrain_gate(values, coefficients.map(|c| scalar(&BigInt::from(c))));
        return;
    }
    let shift = if low < BigInt::from(0) {
        (-low + &p - 1) / &p
    } else {
        BigInt::from(0)
    };
    let offset = &shift * &p;
    let max_q = (high + &offset) / &p;
    let bits = usize::try_from(max_q.bits()).expect("quotient fits usize");
    assert!(bits <= 130, "Goldilocks gate quotient bound");
    let hint_offset = offset.clone();
    let quotient = b.hint("Goldilocks quotient", &values, move |v| {
        let [a, c, d] = [integer(v[0]), integer(v[1]), integer(v[2])];
        let [qm, qa, qb, qc, k] = coefficients;
        let sum = qm * &a * &c + qa * a + qb * c + qc * d + k + &hint_offset;
        if &sum % P != BigInt::from(0) {
            return Err("invalid Goldilocks gate".into());
        }
        Ok(scalar(&(sum / P)))
    });
    range(b, table, quotient, bits);
    let [qm, qa, qb, qc, k] = coefficients.map(|c| scalar(&BigInt::from(c)));
    let product = b.mul(values[0], values[1]);
    let residual = b.linear_combination(
        &[
            (qm, product),
            (qa, values[0]),
            (qb, values[1]),
            (qc, values[2]),
            (-Scalar::from_u64(P), quotient),
        ],
        k + scalar(&offset),
    );
    let zero = b.constant(Scalar::ZERO);
    b.assert_equal(residual, zero);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modular_arithmetic_and_canonical_boundaries() {
        let mut b = CircuitBuilder::<Val>::new();
        let x = b.input("x");
        let y = b.input("y");
        let product = b.mul(x, y);
        let result = b.sub(product, x);
        b.expose_public(result);
        let source = b.finish();
        let mapped = GoldilocksCircuit::new(&source);
        for (a, c) in [
            (0, 0),
            (1, 2),
            (P - 1, P - 1),
            (P - 1, 2),
            (1 << 63, 1 << 63),
        ] {
            let mut witness = source.witness();
            witness.set(x, Val::from_u64(a)).unwrap();
            witness.set(y, Val::from_u64(c)).unwrap();
            let assignment = witness.generate().unwrap();
            let mut witness = mapped.circuit.witness();
            mapped.assign(&assignment, &mut witness).unwrap();
            let output = witness.generate().unwrap();
            assert_eq!(
                output.public_values(),
                &[Scalar::from_u64(
                    assignment.public_values()[0].as_canonical_u64()
                )]
            );
            let mut forged = output.values;
            let quotient = mapped
                .circuit
                .hints
                .iter()
                .find(|h| h.name == "Goldilocks quotient")
                .unwrap();
            forged[quotient.start] += Scalar::ONE;
            assert!(mapped.circuit.check_values(&forged).is_err());
        }
        // Bypass the source witness API: noncanonical scalar inputs must fail
        // the actual translated constraints, not just a conversion helper.
        for invalid in [P, u64::MAX] {
            let mut witness = mapped.circuit.witness();
            for (i, &is_input) in mapped.inputs.is_input.iter().enumerate() {
                if is_input {
                    let value = if i == x.index {
                        Scalar::from_u64(invalid)
                    } else {
                        Scalar::ZERO
                    };
                    witness.set(mapped.inputs.wires[i], value).unwrap();
                }
            }
            assert!(witness.generate().is_err());
        }
    }

    #[test]
    fn arbitrary_gate_coefficients_and_lookup_bounds() {
        let mut b = CircuitBuilder::<Val>::new();
        let x = b.input("x");
        let y = b.input("y");
        let table = b.fixed_table("small", (0..16).map(|i| vec![Val::from_u8(i)]).collect());
        b.lookup(table, &[x]);
        b.lookup(table, &[y]);
        let product = b.mul(x, y);
        let coefficient = Val::from_u64(P / 2);
        b.constrain_gate(
            [x, y, product],
            [coefficient, Val::ZERO, Val::ZERO, -coefficient, Val::ZERO],
        );
        b.expose_public(product);
        let source = b.finish();
        let mapped = GoldilocksCircuit::new(&source);
        let mut witness = source.witness();
        witness.set(x, Val::from_u8(15)).unwrap();
        witness.set(y, Val::from_u8(13)).unwrap();
        let assignment = witness.generate().unwrap();
        let mut witness = mapped.circuit.witness();
        mapped.assign(&assignment, &mut witness).unwrap();
        let output = witness.generate().unwrap();
        assert_eq!(output.public_values(), &[Scalar::from_u8(195)]);
        let mut forged = output.values;
        forged[mapped.inputs.wires[product.index].index] += Scalar::ONE;
        assert!(mapped.circuit.check_values(&forged).is_err());
    }

    #[test]
    fn lookup_tuning_respects_the_quotient_budget() {
        let mut b = CircuitBuilder::<Scalar>::new();
        let x = b.input("x");
        let y = b.mul(x, x);
        b.expose_public(y);
        let compiled = b.finish().lower_to_multi_stark(Scalar::from_u8(1)).unwrap();
        let height = compiled.main_height();
        let tuned = compiled.kzg_circuit_inputs(height, 4).unwrap();
        assert_eq!(tuned[0].lookup_group_size, 4);
        assert!(matches!(
            compiled.kzg_circuit_inputs(height, 1),
            Err(LoweringError::QuotientBudgetTooSmall)
        ));
    }
}
