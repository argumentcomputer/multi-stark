//! Groth16 with streamed constraint rows and bounded curve-operation scratch.
//! Uses the same variable order and Libsnark QAP reduction as `R1csCircuit`.
use super::{ConstraintSink, R1csCircuit, R1csStats};
use ark_bls12_381::{Bls12_381, Fr, G1Projective, G2Projective};
use ark_ec::{CurveGroup, scalar_mul::BatchMulPreprocessing};
use ark_ff::{AdditiveGroup, FftField, Field, PrimeField, UniformRand};
use ark_groth16::{Proof, ProvingKey, VerifyingKey};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{LinearCombination as Lc, SynthesisError, Variable};
use ark_std::rand::Rng;
use std::cell::RefCell;

type Domain = GeneralEvaluationDomain<Fr>;
// Private emitter handles; never passed to Arkworks' constraint system.
const SYMBOL: usize = 1usize << (usize::BITS - 1);

enum Mode {
    Setup { weights: Vec<Fr>, abc: [Vec<Fr>; 3] },
    Witness { abc: [Vec<Fr>; 3] },
}

struct State {
    mode: Mode,
    symbols: Vec<Lc<Fr>>,
    scratch: Vec<(Fr, Variable)>,
    inputs: Vec<Fr>,
    witnesses: Vec<Fr>,
    witness_count: usize,
    row: usize,
    cost: R1csStats,
}

// Expand one row transiently. Duplicate terms add normally, as in the dense
// reduction. Iteration avoids recursion through long chains of affine aliases.
fn visit(
    lc: Lc<Fr>,
    symbols: &[Lc<Fr>],
    scratch: &mut Vec<(Fr, Variable)>,
    mut term: impl FnMut(Fr, Variable) -> Result<(), SynthesisError>,
) -> Result<(), SynthesisError> {
    scratch.clear();
    scratch.extend(lc.0);
    while let Some((c, v)) = scratch.pop() {
        if c == Fr::ZERO {
            continue;
        }
        if let Variable::Instance(i) = v
            && i >= SYMBOL
        {
            let lc = symbols
                .get(i - SYMBOL)
                .ok_or(SynthesisError::Unsatisfiable)?;
            scratch.extend(
                lc.0.iter()
                    .map(|&(a, v)| (if c == Fr::ONE { a } else { c * a }, v)),
            );
            continue;
        }
        term(c, v)?;
    }
    Ok(())
}

fn index(v: Variable, inputs: usize) -> Result<Option<usize>, SynthesisError> {
    match v {
        Variable::Zero => Ok(None),
        Variable::One => Ok(Some(0)),
        Variable::Instance(i) if i < inputs => Ok(Some(i)),
        Variable::Witness(i) => inputs
            .checked_add(i)
            .map(Some)
            .ok_or(SynthesisError::Unsatisfiable),
        _ => Err(SynthesisError::Unsatisfiable),
    }
}

impl ConstraintSink for RefCell<State> {
    fn new_lc(&self, lc: Lc<Fr>) -> Result<Variable, SynthesisError> {
        let mut s = self.borrow_mut();
        let i = s.symbols.len();
        if i >= SYMBOL {
            return Err(SynthesisError::Unsatisfiable);
        }
        s.symbols.push(lc);
        Ok(Variable::Instance(SYMBOL + i))
    }

    fn new_witness_variable(
        &self,
        value: impl FnOnce() -> Result<Fr, SynthesisError>,
    ) -> Result<Variable, SynthesisError> {
        let mut s = self.borrow_mut();
        let i = s.witness_count;
        if i >= s.cost.witnesses {
            return Err(SynthesisError::Unsatisfiable);
        }
        if matches!(s.mode, Mode::Witness { .. }) {
            s.witnesses.push(value()?);
        }
        s.witness_count += 1;
        Ok(Variable::Witness(i))
    }

    fn new_input_variable(
        &self,
        value: impl FnOnce() -> Result<Fr, SynthesisError>,
    ) -> Result<Variable, SynthesisError> {
        let mut s = self.borrow_mut();
        let i = s.inputs.len();
        if i > s.cost.public_inputs {
            return Err(SynthesisError::Unsatisfiable);
        }
        let v = if matches!(s.mode, Mode::Witness { .. }) {
            value()?
        } else {
            Fr::ZERO
        };
        s.inputs.push(v);
        Ok(Variable::Instance(i))
    }

    fn enforce_constraint(&self, a: Lc<Fr>, b: Lc<Fr>, c: Lc<Fr>) -> Result<(), SynthesisError> {
        let mut s = self.borrow_mut();
        let State {
            mode,
            symbols,
            scratch,
            inputs,
            witnesses,
            row,
            cost,
            ..
        } = &mut *s;
        if *row >= cost.constraints {
            return Err(SynthesisError::Unsatisfiable);
        }
        let ni = cost.public_inputs + 1;
        match mode {
            Mode::Setup { weights, abc } => {
                let u = weights[*row];
                for (lc, result) in [a, b, c].into_iter().zip(abc) {
                    visit(lc, symbols, scratch, |coeff, var| {
                        if let Some(i) = index(var, ni)? {
                            *result.get_mut(i).ok_or(SynthesisError::Unsatisfiable)? +=
                                if coeff == Fr::ONE { u } else { u * coeff };
                        }
                        Ok(())
                    })?;
                }
            }
            Mode::Witness { abc } => {
                for (lc, result) in [a, b, c].into_iter().zip(abc.iter_mut()) {
                    let mut sum = Fr::ZERO;
                    visit(lc, symbols, scratch, |coeff, var| {
                        if let Some(i) = index(var, ni)? {
                            let value = if i < ni {
                                inputs.get(i)
                            } else {
                                witnesses.get(i - ni)
                            }
                            .ok_or(SynthesisError::AssignmentMissing)?;
                            sum += if coeff == Fr::ONE {
                                *value
                            } else {
                                coeff * value
                            };
                        }
                        Ok(())
                    })?;
                    result[*row] = sum;
                }
                if abc[0][*row] * abc[1][*row] != abc[2][*row] {
                    return Err(SynthesisError::Unsatisfiable);
                }
            }
        }
        *row += 1;
        if row.is_multiple_of(10_000_000) {
            tracing::info!(target: "multi_stark::streaming", rows = *row, total = cost.constraints, "R1CS rows consumed");
        }
        Ok(())
    }
}

fn domain(cost: R1csStats) -> Result<Domain, SynthesisError> {
    let n = cost
        .constraints
        .checked_add(cost.public_inputs)
        .and_then(|n| n.checked_add(1))
        .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    Domain::new(n).ok_or(SynthesisError::PolynomialDegreeTooLarge)
}

fn emit(input: R1csCircuit<'_>, cost: R1csStats, mode: Mode) -> Result<State, SynthesisError> {
    if input.circuit.publics.len() != cost.public_inputs {
        return Err(SynthesisError::Unsatisfiable);
    }
    let witnesses = if matches!(mode, Mode::Witness { .. }) {
        Vec::with_capacity(cost.witnesses)
    } else {
        Vec::new()
    };
    let state = RefCell::new(State {
        mode,
        symbols: Vec::new(),
        scratch: Vec::new(),
        inputs: vec![Fr::ONE],
        witnesses,
        witness_count: 0,
        row: 0,
        cost,
    });
    input.emit(&state)?;
    let s = state.into_inner();
    if s.row != cost.constraints
        || s.witness_count != cost.witnesses
        || s.inputs.len() != cost.public_inputs + 1
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(s)
}

fn instance_map(
    input: R1csCircuit<'_>,
    cost: R1csStats,
    t: Fr,
) -> Result<[Vec<Fr>; 3], SynthesisError> {
    let d = domain(cost)?;
    if d.evaluate_vanishing_polynomial(t) == Fr::ZERO {
        return Err(SynthesisError::Unsatisfiable);
    }
    let n = cost
        .witnesses
        .checked_add(cost.public_inputs + 1)
        .ok_or(SynthesisError::Unsatisfiable)?;
    let weights = d.evaluate_all_lagrange_coefficients(t);
    let mut abc = std::array::from_fn(|_| vec![Fr::ZERO; n]);
    abc[0][..cost.public_inputs + 1]
        .copy_from_slice(&weights[cost.constraints..cost.constraints + cost.public_inputs + 1]);
    let State { mode, .. } = emit(input, cost, Mode::Setup { weights, abc })?;
    let Mode::Setup { abc, .. } = mode else {
        unreachable!()
    };
    Ok(abc)
}

fn witness_map(
    input: R1csCircuit<'_>,
    cost: R1csStats,
) -> Result<(Vec<Fr>, Vec<Fr>), SynthesisError> {
    if input.assignment.is_none() {
        return Err(SynthesisError::AssignmentMissing);
    }
    let d = domain(cost)?;
    let abc = std::array::from_fn(|_| vec![Fr::ZERO; d.size()]);
    let State {
        mode,
        mut inputs,
        witnesses,
        ..
    } = emit(input, cost, Mode::Witness { abc })?;
    tracing::info!(target: "multi_stark::streaming", "Witness rows checked; computing QAP quotient");
    let Mode::Witness {
        abc: [mut a, mut b, mut c],
    } = mode
    else {
        unreachable!()
    };
    a[cost.constraints..cost.constraints + inputs.len()].copy_from_slice(&inputs);
    inputs.extend(witnesses);
    let coset = d
        .get_coset(Fr::GENERATOR)
        .ok_or(SynthesisError::Unsatisfiable)?;
    d.ifft_in_place(&mut a);
    d.ifft_in_place(&mut b);
    coset.fft_in_place(&mut a);
    coset.fft_in_place(&mut b);
    for (a, b) in a.iter_mut().zip(b) {
        *a *= b;
    }
    d.ifft_in_place(&mut c);
    coset.fft_in_place(&mut c);
    let inverse = d
        .evaluate_vanishing_polynomial(Fr::GENERATOR)
        .inverse()
        .ok_or(SynthesisError::Unsatisfiable)?;
    for (a, c) in a.iter_mut().zip(c) {
        *a = (*a - c) * inverse;
    }
    coset.ifft_in_place(&mut a);
    if a.last() != Some(&Fr::ZERO) {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok((a, inputs))
}

fn batch<G: CurveGroup<ScalarField = Fr>>(
    table: &BatchMulPreprocessing<G>,
    scalars: &[Fr],
) -> Vec<G::Affine> {
    let mut result = Vec::with_capacity(scalars.len());
    for chunk in scalars.chunks(1 << 16) {
        use p3_maybe_rayon::prelude::*;
        let points: Vec<G> = chunk
            .par_iter()
            .map(|s| {
                if *s == Fr::ZERO {
                    return G::zero();
                }
                let bits = s.into_bigint();
                let words = bits.as_ref();
                let mut point = G::zero();
                for (outer, row) in table.table.iter().enumerate() {
                    let bit = outer * table.window;
                    let limb = bit / 64;
                    let shift = bit % 64;
                    let mut word = words.get(limb).copied().unwrap_or(0) >> shift;
                    if shift != 0 {
                        word |= words.get(limb + 1).copied().unwrap_or(0) << (64 - shift);
                    }
                    let i = usize::try_from(word & ((1u64 << table.window) - 1)).unwrap();
                    point += row[i];
                }
                point
            })
            .collect();
        result.extend(G::normalize_batch(&points));
    }
    result
}

/// Ordinary Groth16 setup with streamed QAP evaluation. Randomness must follow
/// the application's trusted-setup policy. Counts are checked against emission.
pub fn setup(
    input: R1csCircuit<'_>,
    cost: R1csStats,
    rng: &mut impl Rng,
) -> Result<ProvingKey<Bls12_381>, SynthesisError> {
    let alpha = Fr::rand(rng);
    let beta = Fr::rand(rng);
    let gamma = Fr::rand(rng);
    let delta = Fr::rand(rng);
    let g1 = G1Projective::rand(rng);
    let g2 = G2Projective::rand(rng);
    let d = domain(cost)?;
    let t = d.sample_element_outside_domain(rng);
    let zt = d.evaluate_vanishing_polynomial(t);
    let [a, b, mut c] = instance_map(input, cost, t)?;
    tracing::info!(target: "multi_stark::streaming", "Setup QAP evaluated; constructing curve queries");
    let gamma_inv = gamma.inverse().ok_or(SynthesisError::UnexpectedIdentity)?;
    let delta_inv = delta.inverse().ok_or(SynthesisError::UnexpectedIdentity)?;
    let ni = cost.public_inputs + 1;
    let gamma_abc: Vec<_> = (0..ni)
        .map(|i| (beta * a[i] + alpha * b[i] + c[i]) * gamma_inv)
        .collect();
    for (i, c) in c.iter_mut().enumerate().skip(ni) {
        *c = (beta * a[i] + alpha * b[i] + *c) * delta_inv;
    }
    c.drain(..ni);
    let l = c;
    let non_zero_a = a.iter().filter(|s| **s != Fr::ZERO).count();
    let non_zero_b = b.iter().filter(|s| **s != Fr::ZERO).count();
    let g2_table = BatchMulPreprocessing::new(g2, non_zero_b);
    let b_g2_query = batch(&g2_table, &b);
    tracing::info!(target: "multi_stark::streaming", "G2 query complete");
    drop(g2_table);
    let g1_table = BatchMulPreprocessing::new(g1, non_zero_a + non_zero_b + a.len() + d.size());
    let a_query = batch(&g1_table, &a);
    drop(a);
    let b_g1_query = batch(&g1_table, &b);
    tracing::info!(target: "multi_stark::streaming", "A and B G1 queries complete");
    drop(b);
    // Generate consecutive powers without retaining a domain-sized scalar vector.
    let mut h_query = Vec::with_capacity(d.size() - 1);
    let mut power = zt * delta_inv;
    while h_query.len() < d.size() - 1 {
        let n = (d.size() - 1 - h_query.len()).min(1 << 16);
        let scalars: Vec<_> = (0..n)
            .map(|_| {
                let v = power;
                power *= t;
                v
            })
            .collect();
        h_query.extend(batch(&g1_table, &scalars));
    }
    let l_query = batch(&g1_table, &l);
    tracing::info!(target: "multi_stark::streaming", "H and L queries complete");
    let vk = VerifyingKey {
        alpha_g1: (g1 * alpha).into_affine(),
        beta_g2: (g2 * beta).into_affine(),
        gamma_g2: (g2 * gamma).into_affine(),
        delta_g2: (g2 * delta).into_affine(),
        gamma_abc_g1: batch(&g1_table, &gamma_abc),
    };
    Ok(ProvingKey {
        vk,
        beta_g1: (g1 * beta).into_affine(),
        delta_g1: (g1 * delta).into_affine(),
        a_query,
        b_g1_query,
        b_g2_query,
        h_query,
        l_query,
    })
}

fn msm<G: CurveGroup<ScalarField = Fr>>(bases: &[G::Affine], scalars: &[Fr]) -> G {
    assert_eq!(bases.len(), scalars.len());
    bases
        .chunks(1 << 18)
        .zip(scalars.chunks(1 << 18))
        .map(|(b, s)| G::msm_unchecked(b, s))
        .sum()
}

/// Prove with standard Groth16 keys, without retaining the R1CS matrices.
pub fn prove(
    input: R1csCircuit<'_>,
    cost: R1csStats,
    pk: &ProvingKey<Bls12_381>,
    rng: &mut impl Rng,
) -> Result<Proof<Bls12_381>, SynthesisError> {
    let ni = cost.public_inputs + 1;
    let n = cost.witnesses + ni;
    let d = domain(cost)?;
    if pk.a_query.len() != n
        || pk.b_g1_query.len() != n
        || pk.b_g2_query.len() != n
        || pk.l_query.len() != cost.witnesses
        || pk.h_query.len() != d.size() - 1
        || pk.vk.gamma_abc_g1.len() != ni
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let r = Fr::rand(rng);
    let s = Fr::rand(rng);
    let (h, assignment) = witness_map(input, cost)?;
    tracing::info!(target: "multi_stark::streaming", "QAP quotient ready; computing proof commitments");
    let h_acc = msm::<G1Projective>(&pk.h_query, &h[..h.len() - 1]);
    drop(h);
    let l_acc = msm::<G1Projective>(&pk.l_query, &assignment[ni..]);
    let a = pk.delta_g1 * r + pk.vk.alpha_g1 + msm::<G1Projective>(&pk.a_query, &assignment);
    let b1 = pk.delta_g1 * s + pk.beta_g1 + msm::<G1Projective>(&pk.b_g1_query, &assignment);
    let b2 = pk.vk.delta_g2 * s + pk.vk.beta_g2 + msm::<G2Projective>(&pk.b_g2_query, &assignment);
    let c = a * s + b1 * r - pk.delta_g1 * (r * s) + l_acc + h_acc;
    Ok(Proof {
        a: a.into_affine(),
        b: b2.into_affine(),
        c: c.into_affine(),
    })
}

/// Run the streamed witness/QAP path without generating a key or proof.
pub fn check(input: R1csCircuit<'_>, cost: R1csStats) -> Result<(), SynthesisError> {
    witness_map(input, cost).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plonkish::{
            CircuitBuilder,
            foreign::GoldilocksCircuit,
            gadgets::{ByteGadgets, blake3},
            r1cs::estimate_goldilocks,
        },
        types::Val,
    };
    use ark_groth16::{
        Groth16, prepare_verifying_key,
        r1cs_to_qap::{LibsnarkReduction, R1CSToQAP},
    };
    use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
    use ark_std::rand::{SeedableRng, rngs::StdRng};
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn fixed_base_windows_match_reference_at_limb_boundaries() {
        let mut rng = StdRng::seed_from_u64(51);
        let mut scalars = vec![Fr::ZERO, Fr::ONE, -Fr::ONE];
        for bit in [1, 63, 64, 65, 127, 128, 191, 192, 254] {
            let x = Fr::from(2u64).pow([bit]);
            scalars.extend([x - Fr::ONE, x, x + Fr::ONE]);
        }
        scalars.extend((0..64).map(|_| Fr::rand(&mut rng)));
        for size in [1, 1024, 65536] {
            let table = BatchMulPreprocessing::new(G1Projective::rand(&mut rng), size);
            assert_eq!(batch(&table, &scalars), table.batch_mul(&scalars));
            let table = BatchMulPreprocessing::new(G2Projective::rand(&mut rng), size);
            assert_eq!(batch(&table, &scalars), table.batch_mul(&scalars));
        }
    }

    #[test]
    fn matches_dense_qap_keys_and_proofs_and_rejects_bad_assignments() {
        let mut b = CircuitBuilder::<Val>::new();
        b.enable_compact_blake3();
        let x = b.input("field");
        let y = b.mul(x, x);
        b.expose_public(y);
        let bytes = ByteGadgets::new(&mut b);
        let byte = bytes.input(&mut b, "byte");
        for v in blake3(&mut b, &bytes, &[byte; 65]) {
            b.expose_public(v.value());
        }
        let c = b.finish();
        let cost = estimate_goldilocks(&c).unwrap();
        let mut w = c.witness();
        w.set(x, Val::from_u64(0xffff_ffff_0000_0000)).unwrap();
        w.set(byte.value(), Val::from_u8(7)).unwrap();
        let a = w.generate().unwrap();
        let foreign = GoldilocksCircuit::new_with_expanded_hashes(&c);
        let mut w = foreign.circuit.witness();
        foreign.assign(&a, &mut w).unwrap();
        let mut a = w.generate().unwrap();
        let circuit = R1csCircuit::new(&foreign.circuit, Some(&a)).unwrap();
        let cs = ConstraintSystem::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();
        assert!(cs.is_satisfied().unwrap());
        cs.finalize();
        for t in [Fr::from(17u64), Fr::from(123u64)] {
            let (a, b, c, _, _, _) =
                LibsnarkReduction::instance_map_with_evaluation::<Fr, Domain>(cs.clone(), &t)
                    .unwrap();
            assert_eq!(instance_map(circuit, cost, t).unwrap(), [a, b, c]);
        }
        let expected_h = LibsnarkReduction::witness_map::<Fr, Domain>(cs.clone()).unwrap();
        let (actual_h, assignment) = witness_map(circuit, cost).unwrap();
        assert_eq!(actual_h, expected_h);
        let dense_assignment = {
            let cs = cs.borrow().unwrap();
            [
                cs.instance_assignment.clone(),
                cs.witness_assignment.clone(),
            ]
            .concat()
        };
        assert_eq!(assignment, dense_assignment);
        drop(cs);
        let dense_pk = Groth16::<Bls12_381>::generate_random_parameters_with_reduction(
            R1csCircuit::new(&foreign.circuit, None).unwrap(),
            &mut StdRng::seed_from_u64(19),
        )
        .unwrap();
        let pk = setup(
            R1csCircuit::new(&foreign.circuit, None).unwrap(),
            cost,
            &mut StdRng::seed_from_u64(19),
        )
        .unwrap();
        assert!(pk == dense_pk, "streaming and dense keys differ");
        let actual = prove(circuit, cost, &pk, &mut StdRng::seed_from_u64(29)).unwrap();
        let expected = Groth16::<Bls12_381>::create_random_proof_with_reduction(
            circuit,
            &dense_pk,
            &mut StdRng::seed_from_u64(29),
        )
        .unwrap();
        assert_eq!(actual, expected);
        let vk = prepare_verifying_key(&pk.vk);
        let public: Vec<_> = a.public_values().iter().map(|v| v.0).collect();
        assert!(Groth16::<Bls12_381>::verify_proof(&vk, &actual, &public).unwrap());
        let mut wrong = public;
        wrong[0] += Fr::ONE;
        assert!(!Groth16::<Bls12_381>::verify_proof(&vk, &actual, &wrong).unwrap());
        let mut bad_cost = cost;
        bad_cost.constraints -= 1;
        assert!(check(circuit, bad_cost).is_err());
        bad_cost = cost;
        bad_cost.witnesses += 1;
        assert!(check(circuit, bad_cost).is_err());
        let bad_index = foreign.circuit.publics.last().unwrap().index;
        a.values[bad_index].0 += Fr::ONE;
        assert!(check(R1csCircuit::new(&foreign.circuit, Some(&a)).unwrap(), cost).is_err());
    }
}
