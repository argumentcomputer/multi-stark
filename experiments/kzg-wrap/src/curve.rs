use ark_bls12_381::{Fq, Fr, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::PrimeField;
use ark_r1cs_std::{fields::emulated_fp::EmulatedFpVar, prelude::*};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

type F = EmulatedFpVar<Fq, Fr>;

/// Affine formulas with explicit identity, doubling and vertical-line cases.
/// Inputs are on-curve; subgroup membership is a separate constraint.
#[derive(Clone)]
pub struct Point {
    pub x: F,
    pub y: F,
    pub infinity: Boolean<Fr>,
}

impl Point {
    fn normalize(&self) -> Result<Self, SynthesisError> {
        let normalize = |v: &F| -> Result<F, SynthesisError> {
            if v.is_constant() {
                return Ok(v.clone());
            }
            let reduced = F::new_witness(v.cs(), || v.value())?;
            reduced.enforce_equal(v)?;
            Ok(reduced)
        };
        Ok(Self {
            x: normalize(&self.x)?,
            y: normalize(&self.y)?,
            infinity: self.infinity.clone(),
        })
    }
    pub fn constant(p: G1Affine) -> Self {
        Self {
            x: F::constant(p.x),
            y: F::constant(p.y),
            infinity: Boolean::constant(p.infinity),
        }
    }

    pub fn witness(cs: ConstraintSystemRef<Fr>, p: G1Affine) -> Result<Self, SynthesisError> {
        let result = Self {
            x: F::new_witness(cs.clone(), || Ok(p.x))?,
            y: F::new_witness(cs.clone(), || Ok(p.y))?,
            infinity: Boolean::new_witness(cs, || Ok(p.infinity))?,
        };
        result
            .x
            .conditional_enforce_equal(&F::zero(), &result.infinity)?;
        result
            .y
            .conditional_enforce_equal(&F::zero(), &result.infinity)?;
        result.y.square()?.conditional_enforce_equal(
            &(&result.x.square()? * &result.x + Fq::from(4)),
            &!&result.infinity,
        )?;
        Ok(result)
    }

    pub fn select(bit: &Boolean<Fr>, a: &Self, b: &Self) -> Result<Self, SynthesisError> {
        Self {
            x: bit.select(&a.x, &b.x)?,
            y: bit.select(&a.y, &b.y)?,
            infinity: bit.select(&a.infinity, &b.infinity)?,
        }
        .normalize()
    }

    pub fn double(&self) -> Result<Self, SynthesisError> {
        let regular = !(&self.infinity | self.y.is_zero()?);
        let denominator = regular.select(&self.y.double()?, &F::one())?;
        let slope = (&self.x.square()? * Fq::from(3)) * denominator.inverse()?;
        let x = slope.square()? - self.x.double()?;
        let y = slope * (&self.x - &x) - &self.y;
        Self::select(
            &regular,
            &Self {
                x,
                y,
                infinity: Boolean::FALSE,
            },
            &Self::constant(G1Affine::identity()),
        )
    }

    pub fn add(&self, other: &Self) -> Result<Self, SynthesisError> {
        let same_x = self.x.is_eq(&other.x)?;
        let same_y = self.y.is_eq(&other.y)?;
        let doubling = &same_x & &same_y;
        let regular =
            !(&self.infinity | &other.infinity) & (!&same_x | (&same_y & !self.y.is_zero()?));
        let numerator =
            doubling.select(&(&self.x.square()? * Fq::from(3)), &(&other.y - &self.y))?;
        let denominator = doubling.select(&self.y.double()?, &(&other.x - &self.x))?;
        let denominator = regular.select(&denominator, &F::one())?;
        let slope = numerator * denominator.inverse()?;
        let x = slope.square()? - &self.x - &other.x;
        let y = slope * (&self.x - &x) - &self.y;
        let sum = Self::select(
            &regular,
            &Self {
                x,
                y,
                infinity: Boolean::FALSE,
            },
            &Self::constant(G1Affine::identity()),
        )?;
        let sum = Self::select(&other.infinity, self, &sum)?;
        Self::select(&self.infinity, other, &sum)
    }

    pub fn scalar_mul(&self, bits: &[Boolean<Fr>]) -> Result<Self, SynthesisError> {
        let mut acc = Self::constant(G1Affine::identity());
        for bit in bits.iter().rev() {
            acc = acc.double()?;
            let added = acc.add(self)?;
            acc = Self::select(bit, &added, &acc)?;
        }
        Ok(acc)
    }

    pub fn constant_mul(&self, limbs: &[u64]) -> Result<Self, SynthesisError> {
        let bits = ark_ff::BitIteratorBE::without_leading_zeros(limbs);
        let mut acc = Self::constant(G1Affine::identity());
        for bit in bits {
            acc = acc.double()?;
            if bit {
                acc = acc.add(self)?;
            }
        }
        Ok(acc)
    }

    pub fn enforce_equal(&self, other: &Self) -> Result<(), SynthesisError> {
        self.infinity.enforce_equal(&other.infinity)?;
        self.x.enforce_equal(&other.x)?;
        self.y.enforce_equal(&other.y)
    }

    /// Enforce membership without assuming that the supplied affine point is
    /// already in the prime-order subgroup. This deliberately uses the full
    /// group-order test as a conservative baseline.
    pub fn enforce_subgroup(&self) -> Result<(), SynthesisError> {
        self.constant_mul(Fr::MODULUS.as_ref())?
            .enforce_equal(&Self::constant(G1Affine::identity()))
    }

    /// Same endomorphism criterion and exceptional-point rejection as the
    /// pinned ark-bls12-381 native subgroup checker (g1.rs).
    pub fn enforce_subgroup_fast(&self) -> Result<(), SynthesisError> {
        let x_p = self.constant_mul(&[0xd201000000010000])?;
        let same =
            self.x.is_eq(&x_p.x)? & self.y.is_eq(&x_p.y)? & self.infinity.is_eq(&x_p.infinity)?;
        (same & !&self.infinity).enforce_equal(&Boolean::FALSE)?;
        let x2_p = x_p.constant_mul(&[0xd201000000010000])?;
        let minus_x2_p = Self {
            x: x2_p.x,
            y: x2_p.y.negate()?,
            infinity: x2_p.infinity,
        };
        let phi = Self {
            x: &self.x * ark_bls12_381::g1::BETA,
            y: self.y.clone(),
            infinity: self.infinity.clone(),
        };
        minus_x2_p.enforce_equal(&phi)
    }

    // Only used by the fixed positive X and X^2 addition chains. Every prefix
    // is strictly between 0 and r-1. For a nonzero subgroup input, doubling
    // and adding the base therefore never encounters an exceptional point.
    // Inversion constraints reject exceptional inputs rather than leaving an
    // unconstrained slope. Identity inputs are replaced with the generator.
    fn fixed_x_mul(&self) -> Result<Self, SynthesisError> {
        let mut acc = self.clone();
        for bit in ark_ff::BitIteratorBE::without_leading_zeros([0xd201000000010000u64]).skip(1) {
            let slope = (&acc.x.square()? * Fq::from(3)) * acc.y.double()?.inverse()?;
            let x = slope.square()? - acc.x.double()?;
            let y = slope * (&acc.x - &x) - &acc.y;
            acc = Self {
                x,
                y,
                infinity: Boolean::FALSE,
            }
            .normalize()?;
            if bit {
                let slope = (&acc.y - &self.y) * (&acc.x - &self.x).inverse()?;
                let x = slope.square()? - &acc.x - &self.x;
                let y = slope * (&self.x - &x) - &self.y;
                acc = Self {
                    x,
                    y,
                    infinity: Boolean::FALSE,
                }
                .normalize()?;
            }
        }
        Ok(acc)
    }

    pub fn enforce_subgroup_fixed_chain(&self) -> Result<(), SynthesisError> {
        let p = Self::select(&self.infinity, &Self::constant(G1Affine::generator()), self)?;
        let x_p = p.fixed_x_mul()?;
        let same = p.x.is_eq(&x_p.x)? & p.y.is_eq(&x_p.y)?;
        same.enforce_equal(&Boolean::FALSE)?;
        let x2_p = x_p.fixed_x_mul()?;
        x2_p.x.enforce_equal(&(&p.x * ark_bls12_381::g1::BETA))?;
        x2_p.y.negate()?.enforce_equal(&p.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_ff::AdditiveGroup;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn addition_handles_identity_doubling_inverse_and_distinct_points() {
        let g = G1Affine::generator();
        for (p, q) in [
            (g, g),
            (g, -g),
            (g, (g * Fr::from(7)).into_affine()),
            (g, G1Affine::identity()),
            (G1Affine::identity(), g),
            (G1Affine::identity(), G1Affine::identity()),
        ] {
            let cs = ConstraintSystem::new_ref();
            let a = Point::witness(cs.clone(), p).unwrap();
            let b = Point::witness(cs.clone(), q).unwrap();
            a.add(&b)
                .unwrap()
                .enforce_equal(&Point::constant((p + q).into_affine()))
                .unwrap();
            assert!(cs.is_satisfied().unwrap());
        }
    }

    #[test]
    fn malformed_point_and_wrong_sum_are_rejected() {
        let cs = ConstraintSystem::new_ref();
        Point::witness(cs.clone(), G1Affine::new_unchecked(Fq::ZERO, Fq::ZERO)).unwrap();
        assert!(!cs.is_satisfied().unwrap());
        let cs = ConstraintSystem::new_ref();
        let p = Point::witness(cs.clone(), G1Affine::generator()).unwrap();
        p.double().unwrap().enforce_equal(&p).unwrap();
        assert!(!cs.is_satisfied().unwrap());
    }

    #[test]
    fn scalar_multiplication_matches_native() {
        for value in [0u64, 1, 2, 3, 7, 123] {
            let cs = ConstraintSystem::new_ref();
            cs.set_optimization_goal(ark_relations::r1cs::OptimizationGoal::Weight);
            let g = G1Affine::generator();
            let p = Point::witness(cs.clone(), g).unwrap();
            let bits = (0..8)
                .map(|i| Boolean::new_witness(cs.clone(), || Ok((value >> i) & 1 == 1)).unwrap())
                .collect::<Vec<_>>();
            let result = p.scalar_mul(&bits).unwrap();
            let expected = (g * Fr::from(value)).into_affine();
            assert_eq!(result.x.value().unwrap(), expected.x, "x for {value}");
            assert_eq!(result.y.value().unwrap(), expected.y, "y for {value}");
            result.enforce_equal(&Point::constant(expected)).unwrap();
            assert!(
                cs.is_satisfied().unwrap(),
                "constraints for {value}: {:?}",
                cs.which_is_unsatisfied()
            );
        }
    }

    #[test]
    fn full_width_scalar_preserves_reduced_coordinate_bounds() {
        let cs = ConstraintSystem::new_ref();
        cs.set_optimization_goal(ark_relations::r1cs::OptimizationGoal::Weight);
        let g = G1Affine::generator();
        let p = Point::witness(cs.clone(), g).unwrap();
        let value = Fr::from(123456789);
        let scalar =
            ark_r1cs_std::fields::fp::FpVar::new_witness(cs.clone(), || Ok(value)).unwrap();
        p.scalar_mul(&scalar.to_bits_le().unwrap())
            .unwrap()
            .enforce_equal(&Point::constant((g * value).into_affine()))
            .unwrap();
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn fixed_chain_subgroup_check_matches_native_acceptance() {
        let off_subgroup = (1..100u64)
            .filter_map(|x| G1Affine::get_point_from_x_unchecked(Fq::from(x), false))
            .find(|p| !p.is_in_correct_subgroup_assuming_on_curve())
            .unwrap();
        for p in [
            G1Affine::generator(),
            G1Affine::identity(),
            G1Affine::new_unchecked(Fq::ZERO, Fq::from(2)),
            off_subgroup,
        ] {
            assert!(p.is_on_curve());
            let cs = ConstraintSystem::new_ref();
            cs.set_optimization_goal(ark_relations::r1cs::OptimizationGoal::Weight);
            let point = Point::witness(cs.clone(), p).unwrap();
            let accepted =
                point.enforce_subgroup_fixed_chain().is_ok() && cs.is_satisfied().unwrap();
            assert_eq!(accepted, p.is_in_correct_subgroup_assuming_on_curve());
        }
    }
}
