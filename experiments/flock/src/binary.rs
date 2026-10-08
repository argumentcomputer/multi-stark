//! Correctness-first GF(2^128) multiplication in the Groth16 scalar field.
use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::{AdditiveGroup, Field};
use ark_groth16::{Groth16, prepare_verifying_key};
use ark_relations::{
    lc,
    r1cs::{
        ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, LinearCombination,
        SynthesisError, Variable,
    },
};
use ark_serialize::CanonicalSerialize;
use flock_prover::field::F128;
use serde_json::json;
use std::time::Instant;

#[derive(Clone)]
struct Bit {
    lc: LinearCombination<Fr>,
    value: bool,
}
impl Bit {
    fn zero() -> Self {
        Self {
            lc: lc!(),
            value: false,
        }
    }
    fn allocate(cs: &ConstraintSystemRef<Fr>, value: bool) -> Result<Self, SynthesisError> {
        let v = cs.new_witness_variable(|| Ok(Fr::from(u64::from(value))))?;
        cs.enforce_constraint(lc!() + v, lc!() + Variable::One - v, lc!())?;
        Ok(Self {
            lc: lc!() + v,
            value,
        })
    }
    fn xor(&self, cs: &ConstraintSystemRef<Fr>, rhs: &Self) -> Result<Self, SynthesisError> {
        if self.lc.0.is_empty() {
            return Ok(rhs.clone());
        }
        if rhs.lc.0.is_empty() {
            return Ok(self.clone());
        }
        let value = self.value ^ rhs.value;
        let v = cs.new_witness_variable(|| Ok(Fr::from(u64::from(value))))?;
        cs.enforce_constraint(
            self.lc.clone() * Fr::from(2u64),
            rhs.lc.clone(),
            self.lc.clone() + &rhs.lc - v,
        )?;
        Ok(Self {
            lc: lc!() + v,
            value,
        })
    }
    fn and(&self, cs: &ConstraintSystemRef<Fr>, rhs: &Self) -> Result<Self, SynthesisError> {
        let value = self.value & rhs.value;
        let v = cs.new_witness_variable(|| Ok(Fr::from(u64::from(value))))?;
        cs.enforce_constraint(self.lc.clone(), rhs.lc.clone(), lc!() + v)?;
        Ok(Self {
            lc: lc!() + v,
            value,
        })
    }
}

fn polynomial_mul(
    cs: &ConstraintSystemRef<Fr>,
    a: &[Bit],
    b: &[Bit],
) -> Result<Vec<Bit>, SynthesisError> {
    if a.len() == 1 {
        return Ok(vec![a[0].and(cs, &b[0])?]);
    }
    let n = a.len() / 2;
    let lo = polynomial_mul(cs, &a[..n], &b[..n])?;
    let hi = polynomial_mul(cs, &a[n..], &b[n..])?;
    let ax = (0..n)
        .map(|i| a[i].xor(cs, &a[n + i]))
        .collect::<Result<Vec<_>, _>>()?;
    let bx = (0..n)
        .map(|i| b[i].xor(cs, &b[n + i]))
        .collect::<Result<Vec<_>, _>>()?;
    let middle = polynomial_mul(cs, &ax, &bx)?;
    let mut out = vec![Bit::zero(); 4 * n - 1];
    for i in 0..2 * n - 1 {
        out[i] = out[i].xor(cs, &lo[i])?;
        out[2 * n + i] = out[2 * n + i].xor(cs, &hi[i])?;
        let cross = middle[i].xor(cs, &lo[i])?.xor(cs, &hi[i])?;
        out[n + i] = out[n + i].xor(cs, &cross)?;
    }
    Ok(out)
}

// Integer convolution coefficients are at most 128. Constrain eight bits per
// coefficient and the polynomial identity at 255 distinct scalar-field points.
// Degree <=254 makes this deterministic. Since inputs are bits and coefficients
// are <256, equality modulo Fr also gives equality over the integers. Their
// low bits are precisely the carry-less product coefficients.
fn convolution_mul(
    cs: &ConstraintSystemRef<Fr>,
    a: &[Bit],
    b: &[Bit],
) -> Result<Vec<Bit>, SynthesisError> {
    let mut counts = [0u64; 255];
    for i in 0..128 {
        for j in 0..128 {
            counts[i + j] += u64::from(a[i].value & b[j].value);
        }
    }
    let mut coefficients = Vec::with_capacity(255);
    let mut parity = Vec::with_capacity(255);
    for count in counts {
        let bits = (0..8)
            .map(|i| Bit::allocate(cs, (count >> i) & 1 == 1))
            .collect::<Result<Vec<_>, _>>()?;
        let mut coefficient = lc!();
        for (i, bit) in bits.iter().enumerate() {
            coefficient = coefficient + &(bit.lc.clone() * Fr::from(1u64 << i));
        }
        let v = cs.new_witness_variable(|| Ok(Fr::from(count)))?;
        cs.enforce_constraint(coefficient - v, lc!() + Variable::One, lc!())?;
        coefficients.push(lc!() + v);
        parity.push(bits[0].clone());
    }
    for point in 1u64..=255 {
        let x = Fr::from(point);
        let mut power = Fr::ONE;
        let (mut al, mut bl, mut cl) = (lc!(), lc!(), lc!());
        for i in 0..255 {
            if i < 128 {
                al = al + &(a[i].lc.clone() * power);
                bl = bl + &(b[i].lc.clone() * power);
            }
            cl = cl + &(coefficients[i].clone() * power);
            power *= x;
        }
        cs.enforce_constraint(al, bl, cl)?;
    }
    Ok(parity)
}

fn bits(cs: &ConstraintSystemRef<Fr>, v: F128) -> Result<Vec<Bit>, SynthesisError> {
    (0..128)
        .map(|i| {
            Bit::allocate(
                cs,
                ((if i < 64 { v.lo } else { v.hi }) >> (i % 64)) & 1 == 1,
            )
        })
        .collect()
}
fn packed(v: F128) -> Fr {
    Fr::from(v.lo) + Fr::from(v.hi) * Fr::from(2u64).pow([64])
}
fn bind_public(cs: &ConstraintSystemRef<Fr>, bits: &[Bit], v: F128) -> Result<(), SynthesisError> {
    let input = cs.new_input_variable(|| Ok(packed(v)))?;
    let mut lc = lc!();
    let mut weight = Fr::ONE;
    for bit in bits {
        lc = lc + &(bit.lc.clone() * weight);
        weight.double_in_place();
    }
    cs.enforce_constraint(lc - input, lc!() + Variable::One, lc!())
}

#[derive(Clone)]
struct Multiplication {
    a: F128,
    b: F128,
    output: F128,
    convolution: bool,
}
impl ConstraintSynthesizer<Fr> for Multiplication {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        let a = bits(&cs, self.a)?;
        let b = bits(&cs, self.b)?;
        bind_public(&cs, &a, self.a)?;
        bind_public(&cs, &b, self.b)?;
        let mut product = if self.convolution {
            convolution_mul(&cs, &a, &b)?
        } else {
            polynomial_mul(&cs, &a, &b)?
        };
        // GHASH polynomial x^128 + x^7 + x^2 + x + 1.
        for i in (128..255).rev() {
            let high = product[i].clone();
            for j in [0, 1, 2, 7] {
                product[i - 128 + j] = product[i - 128 + j].xor(&cs, &high)?;
            }
        }
        bind_public(&cs, &product[..128], self.output)
    }
}

pub fn benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let a = F128::new(0xa5a5010203040506, 0xfefdfcfbdeadbeef);
    let b = F128::new(0x1122334455667788, 0x99aabbccddeeff00);
    let circuit = Multiplication {
        a,
        b,
        output: a * b,
        convolution: true,
    };
    let cs = ConstraintSystem::<Fr>::new_ref();
    circuit.clone().generate_constraints(cs.clone())?;
    assert!(cs.is_satisfied()?);
    let mut rng = ark_std::test_rng();
    let start = Instant::now();
    let pk =
        Groth16::<Bls12_381>::generate_random_parameters_with_reduction(circuit.clone(), &mut rng)?;
    let setup_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let proof =
        Groth16::<Bls12_381>::create_random_proof_with_reduction(circuit.clone(), &pk, &mut rng)?;
    let prove_seconds = start.elapsed().as_secs_f64();
    let vk = prepare_verifying_key(&pk.vk);
    let public = [packed(a), packed(b), packed(a * b)];
    assert!(Groth16::<Bls12_381>::verify_proof(&vk, &proof, &public)?);
    for i in 0..3 {
        let mut wrong = public;
        wrong[i] += Fr::ONE;
        assert!(!Groth16::<Bls12_381>::verify_proof(&vk, &proof, &wrong)?);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "scope": "One constrained GF(2^128) multiplication; not a Flock verifier",
            "method": "bounded integer convolution, deterministic 255-point identity",
            "constraints": cs.num_constraints(), "witnesses": cs.num_witness_variables(),
            "public_inputs": cs.num_instance_variables()-1,
            "setup_seconds": setup_seconds, "prove_seconds": prove_seconds,
            "proof_bytes": proof.compressed_size(), "verified": true,
            "altered_inputs_and_output_rejected": true, "setup": "insecure development randomness"
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn multiplication_matches_flock_and_rejects_wrong_output() {
        let values = [
            F128::ZERO,
            F128::ONE,
            F128::new(0, 1 << 63),
            F128::new(u64::MAX, u64::MAX),
            F128::new(0x918203746523ff00, 0x554433221100abcd),
        ];
        for convolution in [false, true] {
            for a in values {
                for b in values {
                    let circuit = Multiplication {
                        a,
                        b,
                        output: a * b,
                        convolution,
                    };
                    let cs = ConstraintSystem::<Fr>::new_ref();
                    circuit.clone().generate_constraints(cs.clone()).unwrap();
                    assert!(cs.is_satisfied().unwrap());
                    let cs = ConstraintSystem::<Fr>::new_ref();
                    let mut bad = circuit;
                    bad.output.lo ^= 1;
                    bad.generate_constraints(cs.clone()).unwrap();
                    assert!(!cs.is_satisfied().unwrap());
                }
            }
        }
    }
}
