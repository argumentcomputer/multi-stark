//! Direct R1CS lowering. Table contents, never names, select the supported
//! range/XOR/split relations. Compact hash traces must first be expanded.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use ark_bls12_381::Fr;
use ark_ff::{AdditiveGroup, Field, PrimeField};
use ark_relations::{
    lc,
    r1cs::{
        ConstraintSynthesizer, ConstraintSystemRef, LinearCombination, SynthesisError, Variable,
    },
};

use super::{Assignment, Circuit, TableDefinition, builder::Recipe};
use crate::ark_adapter::Scalar;

pub mod streaming;

trait ConstraintSink {
    fn new_lc(&self, lc: LinearCombination<Fr>) -> Result<Variable, SynthesisError>;
    fn new_witness_variable(
        &self,
        value: impl FnOnce() -> Result<Fr, SynthesisError>,
    ) -> Result<Variable, SynthesisError>;
    fn new_input_variable(
        &self,
        value: impl FnOnce() -> Result<Fr, SynthesisError>,
    ) -> Result<Variable, SynthesisError>;
    fn enforce_constraint(
        &self,
        a: LinearCombination<Fr>,
        b: LinearCombination<Fr>,
        c: LinearCombination<Fr>,
    ) -> Result<(), SynthesisError>;
}

impl ConstraintSink for ConstraintSystemRef<Fr> {
    fn new_lc(&self, lc: LinearCombination<Fr>) -> Result<Variable, SynthesisError> {
        self.new_lc(lc)
    }
    fn new_witness_variable(
        &self,
        value: impl FnOnce() -> Result<Fr, SynthesisError>,
    ) -> Result<Variable, SynthesisError> {
        self.new_witness_variable(value)
    }
    fn new_input_variable(
        &self,
        value: impl FnOnce() -> Result<Fr, SynthesisError>,
    ) -> Result<Variable, SynthesisError> {
        self.new_input_variable(value)
    }
    fn enforce_constraint(
        &self,
        a: LinearCombination<Fr>,
        b: LinearCombination<Fr>,
        c: LinearCombination<Fr>,
    ) -> Result<(), SynthesisError> {
        self.enforce_constraint(a, b, c)
    }
}

/// Exact counts for this adapter, before allocating constraint matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct R1csStats {
    pub constraints: usize,
    pub witnesses: usize,
    pub public_inputs: usize,
    pub scalar_values: usize,
    pub scalar_gates: usize,
    pub scalar_lookups: usize,
    pub bit_variables: usize,
    pub inlined_linear_gates: usize,
}

#[derive(Default)]
struct LookupCensus {
    relations: HashMap<usize, Relation>,
    ranges: Vec<u32>,
    constraints: usize,
    variables: usize,
    unsupported: bool,
    linear_terms: Vec<u8>,
    inlined: usize,
}

// Bound expanded LC weight: unrestricted inlining can make sparse matrices dense.
const MAX_INLINE_TERMS: usize = 16;

fn linear_terms(coefficients: [Fr; 3], inputs: [usize; 2]) -> usize {
    usize::from(coefficients[0] != Fr::ZERO) * inputs[0]
        + usize::from(coefficients[1] != Fr::ZERO) * inputs[1]
        + usize::from(coefficients[2] != Fr::ZERO)
}

impl LookupCensus {
    fn linear(&mut self, out: super::Value, inputs: [super::Value; 2], coefficients: [Scalar; 3]) {
        let terms = linear_terms(
            coefficients.map(|c| c.0),
            inputs.map(|v| {
                if v.index == 0 {
                    0
                } else {
                    usize::from(self.linear_terms.get(v.index).copied().unwrap_or(1))
                }
            }),
        );
        if terms <= MAX_INLINE_TERMS {
            if self.linear_terms.len() <= out.index {
                self.linear_terms.resize(out.index + 1, 1);
            }
            self.linear_terms[out.index] = u8::try_from(terms).unwrap();
            self.inlined += 1;
        }
    }

    fn register(&mut self, index: usize, width: usize) {
        if self.ranges.len() <= index {
            self.ranges.resize(index + 1, 0);
        }
        self.ranges[index] |= 1u32 << width;
    }

    fn bits(&mut self, index: usize, width: usize, boolean: bool) {
        let previous = self.ranges.get(index).copied().unwrap_or(0);
        if previous.trailing_zeros() as usize > width {
            self.register(index, width);
            self.variables += width;
            self.constraints += usize::from(boolean) * width + 1;
        }
    }

    fn lookup(
        &mut self,
        table: usize,
        definition: &TableDefinition<Scalar>,
        values: &[super::Value],
    ) {
        let r = if let Some(&r) = self.relations.get(&table) {
            r
        } else {
            let Ok(r) = relation(definition) else {
                self.unsupported = true;
                return;
            };
            self.relations.insert(table, r);
            r
        };
        match r {
            Relation::Range(n) => self.bits(values[0].index, n, true),
            Relation::Xor => {
                self.bits(values[0].index, 4, true);
                self.bits(values[1].index, 4, true);
                self.bits(values[2].index, 4, false);
                self.constraints += 4;
            }
            Relation::Split => {
                self.bits(values[0].index, 4, true);
                self.register(values[1].index, 3);
                self.register(values[2].index, 1);
                self.constraints += 2;
            }
        }
    }
}

/// Count the direct adapter after foreign-field translation and generic hash
/// expansion. Stores range-use flags, not expanded gates, recipes or R1CS rows.
pub fn estimate_goldilocks(
    source: &Circuit<crate::types::Val>,
) -> Result<R1csStats, SynthesisError> {
    let lookup = Rc::new(RefCell::new(LookupCensus::default()));
    let observer = lookup.clone();
    let mut b = super::CircuitBuilder::<Scalar>::counting();
    b.observe_counted_lookups(Box::new(move |table, definition, values| {
        observer.borrow_mut().lookup(table, definition, values);
    }));
    let observer = lookup.clone();
    b.observe_counted_linear(Box::new(move |out, inputs, coefficients| {
        observer.borrow_mut().linear(out, inputs, coefficients);
    }));
    super::foreign::GoldilocksCircuit::translate(source, &mut b, true);
    for table in b.tables() {
        relation(table)?;
    }
    let stats = b.stats();
    let lookup = lookup.borrow();
    if lookup.unsupported {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(R1csStats {
        constraints: stats.gates + stats.publics + lookup.constraints - lookup.inlined,
        witnesses: stats.values - b.constant_count() + lookup.variables - lookup.inlined,
        public_inputs: stats.publics,
        scalar_values: stats.values,
        scalar_gates: stats.gates,
        scalar_lookups: stats.lookups,
        bit_variables: lookup.variables,
        inlined_linear_gates: lookup.inlined,
    })
}

#[derive(Clone, Copy)]
enum Relation {
    Range(usize),
    Xor,
    Split,
}

fn relation(table: &TableDefinition<Scalar>) -> Result<Relation, SynthesisError> {
    let rows = table.rows();
    if rows.len().is_power_of_two()
        && rows.len() <= 65536
        && rows
            .iter()
            .enumerate()
            .all(|(i, row)| row.len() == 1 && row[0].0 == Fr::from(u64::try_from(i).unwrap()))
    {
        return Ok(Relation::Range(rows.len().ilog2() as usize));
    }
    let row_is = |row: &[Scalar], expected: [u64; 3]| {
        row.len() == 3 && row.iter().zip(expected).all(|(v, e)| v.0 == Fr::from(e))
    };
    if rows.len() == 256
        && rows.iter().enumerate().all(|(i, row)| {
            let a = u64::try_from(i / 16).unwrap();
            let b = u64::try_from(i % 16).unwrap();
            row_is(row, [a, b, a ^ b])
        })
    {
        return Ok(Relation::Xor);
    }
    if rows.len() == 16
        && rows.iter().enumerate().all(|(i, row)| {
            let n = u64::try_from(i).unwrap();
            row_is(row, [n, n & 7, n >> 3])
        })
    {
        return Ok(Relation::Split);
    }
    Err(SynthesisError::Unsatisfiable)
}

/// Public inputs follow `Circuit::public_values`. Setup uses `None`; proving
/// uses an assignment belonging to that exact circuit. The verifier supplies
/// independently expected public inputs and a trusted verification key.
#[derive(Clone, Copy)]
pub struct R1csCircuit<'a> {
    circuit: &'a Circuit<Scalar>,
    assignment: Option<&'a Assignment<Scalar>>,
}

impl<'a> R1csCircuit<'a> {
    pub fn new(
        circuit: &'a Circuit<Scalar>,
        assignment: Option<&'a Assignment<Scalar>>,
    ) -> Result<Self, SynthesisError> {
        if circuit.stats().hash_calls != 0
            || assignment
                .is_some_and(|a| a.owner != circuit.owner || a.values.len() != circuit.num_values())
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        for table in &circuit.tables {
            relation(table)?;
        }
        Ok(Self {
            circuit,
            assignment,
        })
    }
}

struct Lowering<'a, S> {
    cs: &'a S,
    input: R1csCircuit<'a>,
    wires: Vec<Variable>,
    bits: HashMap<(usize, usize), Vec<Variable>>,
}

impl<S: ConstraintSink> Lowering<'_, S> {
    fn wire(&self, index: usize) -> LinearCombination<Fr> {
        match self.input.circuit.recipes[index] {
            Recipe::Constant(c) => lc!() + (c.0, Variable::One),
            _ => lc!() + self.wires[index],
        }
    }

    fn equal(
        &self,
        a: LinearCombination<Fr>,
        b: LinearCombination<Fr>,
    ) -> Result<(), SynthesisError> {
        self.cs
            .enforce_constraint(a - b, lc!() + Variable::One, lc!())
    }

    fn bits(
        &mut self,
        index: usize,
        width: usize,
        boolean: bool,
    ) -> Result<Vec<Variable>, SynthesisError> {
        for n in 0..=width {
            if let Some(bits) = self.bits.get(&(index, n)) {
                let mut bits = bits.clone();
                bits.resize(width, Variable::Zero);
                return Ok(bits);
            }
        }
        let mut packed = lc!();
        let mut bits = Vec::with_capacity(width);
        let mut weight = Fr::ONE;
        for i in 0..width {
            let bit = self.cs.new_witness_variable(|| {
                let a = self
                    .input
                    .assignment
                    .ok_or(SynthesisError::AssignmentMissing)?;
                Ok(Fr::from((a.values[index].0.into_bigint().0[0] >> i) & 1))
            })?;
            if boolean {
                self.cs
                    .enforce_constraint(lc!() + bit, lc!() + bit - Variable::One, lc!())?;
            }
            packed += (weight, bit);
            weight.double_in_place();
            bits.push(bit);
        }
        self.equal(packed, self.wire(index))?;
        self.bits.insert((index, width), bits.clone());
        Ok(bits)
    }
}

impl ConstraintSynthesizer<Fr> for R1csCircuit<'_> {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        self.emit(&cs)
    }
}

impl R1csCircuit<'_> {
    fn emit(self, cs: &impl ConstraintSink) -> Result<(), SynthesisError> {
        let relations = self
            .circuit
            .tables
            .iter()
            .map(relation)
            .collect::<Result<Vec<_>, _>>()?;
        let mut l = Lowering {
            cs,
            input: self,
            wires: Vec::with_capacity(self.circuit.num_values()),
            bits: HashMap::new(),
        };
        let mut inlined = vec![false; self.circuit.gates.len()];
        let mut terms = Vec::with_capacity(self.circuit.num_values());
        for (i, recipe) in self.circuit.recipes.iter().enumerate() {
            if let Recipe::Arithmetic(gate_index) = recipe {
                let gate = &self.circuit.gates[*gate_index];
                let [qm, qa, qb, qc, k] = gate.coefficients.map(|c| c.0);
                let [a, b, c] = gate.wires;
                let count = linear_terms([qa, qb, k], [terms[a.index], terms[b.index]]);
                if qm == Fr::ZERO && qc == -Fr::ONE && c.index == i && count <= MAX_INLINE_TERMS {
                    let lc = l.wire(a.index) * qa + l.wire(b.index) * qb + (k, Variable::One);
                    l.wires.push(cs.new_lc(lc)?);
                    terms.push(count);
                    inlined[*gate_index] = true;
                    continue;
                }
            }
            terms.push(match recipe {
                Recipe::Constant(c) if c.0 == Fr::ZERO => 0,
                _ => 1,
            });
            l.wires.push(match recipe {
                Recipe::Constant(_) => Variable::One,
                _ => cs.new_witness_variable(|| {
                    Ok(self
                        .assignment
                        .ok_or(SynthesisError::AssignmentMissing)?
                        .values[i]
                        .0)
                })?,
            });
        }
        drop(terms);
        for (index, gate) in self.circuit.gates.iter().enumerate() {
            if inlined[index] {
                continue;
            }
            let [a, b, c] = gate.wires.map(|w| l.wire(w.index));
            let [qm, qa, qb, qc, k] = gate.coefficients.map(|v| v.0);
            cs.enforce_constraint(
                a.clone(),
                b.clone() * qm,
                a * -qa + b * -qb + c * -qc + (-k, Variable::One),
            )?;
        }
        for lookup in &self.circuit.lookups {
            let v: Vec<_> = lookup.values.iter().map(|v| v.index).collect();
            match relations[lookup.table.index] {
                Relation::Range(n) => {
                    l.bits(v[0], n, true)?;
                }
                Relation::Xor => {
                    let a = l.bits(v[0], 4, true)?;
                    let b = l.bits(v[1], 4, true)?;
                    // Boolean inputs and c=a+b-2ab enforce Boolean outputs.
                    let c = l.bits(v[2], 4, false)?;
                    for i in 0..4 {
                        cs.enforce_constraint(
                            lc!() + (Fr::from(2u64), a[i]),
                            lc!() + b[i],
                            lc!() + a[i] + b[i] - c[i],
                        )?;
                    }
                }
                Relation::Split => {
                    let bits = l.bits(v[0], 4, true)?;
                    l.equal(
                        l.wire(v[1]),
                        lc!() + bits[0] + (Fr::from(2u64), bits[1]) + (Fr::from(4u64), bits[2]),
                    )?;
                    l.equal(l.wire(v[2]), lc!() + bits[3])?;
                    l.bits
                        .entry((v[1], 3))
                        .or_insert_with(|| bits[..3].to_vec());
                    l.bits.entry((v[2], 1)).or_insert_with(|| vec![bits[3]]);
                }
            }
        }
        for public in &self.circuit.publics {
            let var = cs.new_input_variable(|| {
                Ok(self
                    .assignment
                    .ok_or(SynthesisError::AssignmentMissing)?
                    .values[public.index]
                    .0)
            })?;
            l.equal(lc!() + var, l.wire(public.index))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plonkish::{
        CircuitBuilder,
        gadgets::{ByteGadgets, blake3},
    };
    use ark_bls12_381::Bls12_381;
    use ark_groth16::{Groth16, prepare_verifying_key};
    use ark_relations::r1cs::{ConstraintSystem, SynthesisMode};
    use ark_serialize::CanonicalSerialize;
    use ark_std::rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn expanded_hash_census_matches_r1cs_and_rejects_forged_digest() {
        use crate::{plonkish::foreign::GoldilocksCircuit, traits::Field as _, types::Val};
        for compact in [false, true] {
            let mut b = CircuitBuilder::<Val>::new();
            if compact {
                b.enable_compact_blake3();
            }
            let x = b.input("large field element");
            let square = b.mul(x, x);
            b.expose_public(square);
            let bytes = ByteGadgets::new(&mut b);
            let byte = bytes.input(&mut b, "byte");
            for v in blake3(&mut b, &bytes, &[byte; 65]) {
                b.expose_public(v.value());
            }
            let source = b.finish();
            let cost = estimate_goldilocks(&source).unwrap();
            let mut w = source.witness();
            w.set(x, Val::from_u64(0xffff_ffff_0000_0000)).unwrap();
            w.set(byte.value(), Val::from_u8(7)).unwrap();
            let a = w.generate().unwrap();
            let foreign = GoldilocksCircuit::new_with_expanded_hashes(&source);
            let mut w = foreign.circuit.witness();
            foreign.assign(&a, &mut w).unwrap();
            let mut a = w.generate().unwrap();
            let cs = ConstraintSystem::new_ref();
            R1csCircuit::new(&foreign.circuit, Some(&a))
                .unwrap()
                .generate_constraints(cs.clone())
                .unwrap();
            assert!(cs.is_satisfied().unwrap());
            assert_eq!(cost.constraints, cs.num_constraints());
            assert_eq!(cost.witnesses, cs.num_witness_variables());
            assert_eq!(cost.public_inputs + 1, cs.num_instance_variables());
            let index = foreign.circuit.publics.last().unwrap().index;
            a.values[index].0 += Fr::ONE;
            let bad = ConstraintSystem::new_ref();
            R1csCircuit::new(&foreign.circuit, Some(&a))
                .unwrap()
                .generate_constraints(bad.clone())
                .unwrap();
            assert!(!bad.is_satisfied().unwrap());
        }
    }

    #[test]
    fn hash_proof_binds_public_digest() {
        let mut b = CircuitBuilder::<Scalar>::new();
        let bytes = ByteGadgets::new(&mut b);
        let x = bytes.input(&mut b, "x");
        let hash = blake3(&mut b, &bytes, &[x]);
        for byte in hash {
            b.expose_public(byte.value());
        }
        let c = b.finish();
        let mut w = c.witness();
        w.set(x.value(), Scalar(Fr::from(7u64))).unwrap();
        let a = w.generate().unwrap();
        let expected: Vec<_> = ::blake3::hash(&[7])
            .as_bytes()
            .iter()
            .map(|&v| Fr::from(u64::from(v)))
            .collect();
        assert_eq!(
            a.public_values().iter().map(|v| v.0).collect::<Vec<_>>(),
            expected
        );
        let circuit = R1csCircuit::new(&c, Some(&a)).unwrap();
        let cs = ConstraintSystem::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();
        assert!(cs.is_satisfied().unwrap());
        let setup = ConstraintSystem::new_ref();
        setup.set_mode(SynthesisMode::Setup);
        R1csCircuit::new(&c, None)
            .unwrap()
            .generate_constraints(setup.clone())
            .unwrap();
        assert_eq!(cs.num_constraints(), setup.num_constraints());
        assert_eq!(cs.num_witness_variables(), setup.num_witness_variables());
        let mut rng = StdRng::seed_from_u64(23);
        let pk = Groth16::<Bls12_381>::generate_random_parameters_with_reduction(
            R1csCircuit::new(&c, None).unwrap(),
            &mut rng,
        )
        .unwrap();
        let proof =
            Groth16::<Bls12_381>::create_random_proof_with_reduction(circuit, &pk, &mut rng)
                .unwrap();
        let vk = prepare_verifying_key(&pk.vk);
        assert!(Groth16::<Bls12_381>::verify_proof(&vk, &proof, &expected).unwrap());
        let mut wrong = expected;
        wrong[0] += Fr::ONE;
        assert!(!Groth16::<Bls12_381>::verify_proof(&vk, &proof, &wrong).unwrap());
        assert_eq!(proof.compressed_size(), 192);
    }

    #[test]
    fn lookup_relations_reject_forged_values() {
        for kind in 0..3 {
            let mut b = CircuitBuilder::<Scalar>::new();
            let rows: Vec<Vec<Scalar>> = match kind {
                0 => (0..65536u64).map(|n| vec![Scalar(Fr::from(n))]).collect(),
                1 => (0..16u64)
                    .flat_map(|a| {
                        (0..16u64).map(move |c| {
                            vec![a, c, a ^ c]
                                .into_iter()
                                .map(|n| Scalar(Fr::from(n)))
                                .collect()
                        })
                    })
                    .collect(),
                _ => (0..16u64)
                    .map(|n| {
                        vec![n, n & 7, n >> 3]
                            .into_iter()
                            .map(|n| Scalar(Fr::from(n)))
                            .collect()
                    })
                    .collect(),
            };
            let t = b.fixed_table("arbitrary name", rows);
            let values = if kind == 0 {
                vec![5u64]
            } else if kind == 1 {
                vec![5, 9, 12]
            } else {
                vec![13, 5, 1]
            };
            let wires: Vec<_> = (0..values.len()).map(|i| b.input(format!("{i}"))).collect();
            b.lookup(t, &wires);
            let c = b.finish();
            let mut w = c.witness();
            for (&wire, value) in wires.iter().zip(values) {
                w.set(wire, Scalar(Fr::from(value))).unwrap();
            }
            let mut a = w.generate().unwrap();
            let check = |a: &Assignment<Scalar>| {
                let cs = ConstraintSystem::new_ref();
                R1csCircuit::new(&c, Some(a))
                    .unwrap()
                    .generate_constraints(cs.clone())
                    .unwrap();
                cs.is_satisfied().unwrap()
            };
            assert!(check(&a));
            if kind == 0 {
                a.values[wires[0].index] = Scalar(Fr::from(65536u64));
            } else {
                a.values[wires[2].index].0 += Fr::ONE;
            }
            assert!(!check(&a));
        }
    }

    #[test]
    fn rejects_unknown_tables_compact_hashes_and_foreign_assignments() {
        let mut b = CircuitBuilder::<Scalar>::new();
        b.fixed_table("nibble", vec![vec![Scalar(Fr::from(99u64))]]);
        assert!(R1csCircuit::new(&b.finish(), None).is_err());
        let mut b = CircuitBuilder::<Scalar>::new();
        b.enable_compact_blake3();
        let bytes = ByteGadgets::new(&mut b);
        blake3(&mut b, &bytes, &[]);
        assert!(R1csCircuit::new(&b.finish(), None).is_err());
        let a = CircuitBuilder::<Scalar>::new().finish();
        let b = CircuitBuilder::<Scalar>::new().finish();
        let assignment = a.witness().generate().unwrap();
        assert!(R1csCircuit::new(&b, Some(&assignment)).is_err());
    }
}
