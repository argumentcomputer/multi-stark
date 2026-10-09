//! The crate evaluation domain over [`Scalar`]: a multiplicative coset
//! `shift · H` of the order-`2^log_size` subgroup `H`.
//!
//! The selector formulas reproduce the p3 coset semantics the core's
//! constraint math was written against (unnormalized selectors; the
//! logUp boundary injection pre-absorbs the `n·g` normalization):
//! with `s` the shift, `g` the subgroup generator, and `u = X/s`,
//!
//! - vanishing:      `Z(X) = u^n − 1`
//! - first row:      `Z(X)/(u − 1)`
//! - last row:       `Z(X)/(u − g⁻¹)`
//! - transition:     `u − g⁻¹`

use crate::traits::{
    Algebra, EvaluationDomain, Field, LagrangeSelectors, TwoAdicField, batch_inverse,
};

use super::field::Scalar;

const ONE: Scalar = <Scalar as Algebra<Scalar>>::ONE;

/// A coset `shift · H`, `|H| = 2^log_size`, over the BLS12-381 scalar
/// field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Radix2Coset {
    pub log_size: usize,
    pub shift: Scalar,
}

impl Radix2Coset {
    /// The subgroup generator `g`.
    #[inline]
    pub fn generator(&self) -> Scalar {
        Scalar::two_adic_generator(self.log_size)
    }

    /// The coset's points in natural order: `shift · g^i`.
    pub fn points(&self) -> Vec<Scalar> {
        use p3_maybe_rayon::prelude::*;
        let generator = self.generator();
        let mut points = vec![ONE; self.size()];
        points
            .par_chunks_mut(1 << 14)
            .enumerate()
            .for_each(|(tile, output)| {
                let mut point = self.shift * generator.exp_u64((tile * (1 << 14)) as u64);
                for value in output {
                    *value = point;
                    point *= generator;
                }
            });
        points
    }
}

impl EvaluationDomain for Radix2Coset {
    type F = Scalar;
    type Challenge = Scalar;

    #[inline]
    fn size(&self) -> usize {
        1 << self.log_size
    }

    #[inline]
    fn first_point(&self) -> Scalar {
        self.shift
    }

    #[inline]
    fn next_point(&self, x: Scalar) -> Scalar {
        x * self.generator()
    }

    fn create_disjoint_domain(&self, min_size: usize) -> Self {
        // Multiplying the shift by the field's multiplicative generator
        // leaves the subgroup (and every subgroup coset reachable by
        // repeated application), exactly as in p3.
        Self {
            log_size: p3_util::log2_ceil_usize(min_size),
            shift: self.shift * Scalar(<ark_bls12_381::Fr as ark_ff::FftField>::GENERATOR),
        }
    }

    fn selectors_at_point(&self, point: Scalar) -> LagrangeSelectors<Scalar> {
        let unshifted = point * self.shift.inverse();
        let z_h = unshifted.exp_power_of_2(self.log_size) - ONE;
        let g_inv = self.generator().inverse();
        LagrangeSelectors {
            is_first_row: z_h * (unshifted - ONE).inverse(),
            is_last_row: z_h * (unshifted - g_inv).inverse(),
            is_transition: unshifted - g_inv,
            inv_vanishing: z_h.inverse(),
        }
    }

    fn selectors_on_coset(&self, coset: Self) -> LagrangeSelectors<Vec<Scalar>> {
        assert_eq!(self.shift, ONE, "selectors_on_coset needs the group itself");
        assert_ne!(coset.shift, ONE, "coset must be disjoint from the group");
        assert!(coset.log_size >= self.log_size);
        let rate_bits = coset.log_size - self.log_size;

        // Z_H(X) = X^n − 1 is periodic over the coset with period
        // 2^rate_bits: (s·w^j·h)^n = s^n·w^{jn} and h^n = 1.
        let s_pow_n = coset.shift.exp_power_of_2(self.log_size);
        let vanishing: Vec<Scalar> = Scalar::two_adic_generator(rate_bits)
            .powers()
            .take(1 << rate_bits)
            .map(|x| s_pow_n * x - ONE)
            .collect();

        use p3_maybe_rayon::prelude::*;
        let subgroup_last = self.generator().inverse();
        let generator = coset.generator();
        let inverses = batch_inverse(&vanishing);
        let mut first = vec![ONE; coset.size()];
        let mut last = vec![ONE; coset.size()];
        let mut transition = vec![ONE; coset.size()];
        let mut inv_vanishing = vec![ONE; coset.size()];
        first
            .par_chunks_mut(1 << 14)
            .zip(last.par_chunks_mut(1 << 14))
            .zip(transition.par_chunks_mut(1 << 14))
            .zip(inv_vanishing.par_chunks_mut(1 << 14))
            .enumerate()
            .for_each(|(tile, (((first, last), transition), inv_vanishing))| {
                let start = tile * (1 << 14);
                let mut point = coset.shift * generator.exp_u64(start as u64);
                for row in 0..first.len() {
                    first[row] = point - ONE;
                    last[row] = point - subgroup_last;
                    transition[row] = last[row];
                    inv_vanishing[row] = inverses[(start + row) % inverses.len()];
                    point *= generator;
                }
                let first_inverses = batch_inverse(first);
                let last_inverses = batch_inverse(last);
                for row in 0..first.len() {
                    let z = vanishing[(start + row) % vanishing.len()];
                    first[row] = z * first_inverses[row];
                    last[row] = z * last_inverses[row];
                }
            });
        LagrangeSelectors {
            is_first_row: first,
            is_last_row: last,
            is_transition: transition,
            inv_vanishing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selectors_and_points_agree_across_parallel_tiles() {
        for extra_bits in [0, 1, 2] {
            let trace = Radix2Coset {
                log_size: 14,
                shift: ONE,
            };
            let coset = trace.create_disjoint_domain(1 << (14 + extra_bits));
            let points = coset.points();
            let selectors = trace.selectors_on_coset(coset);
            for i in [0, 1, (1 << 14) - 1, 1 << 14, coset.size() - 1] {
                if i >= coset.size() {
                    continue;
                }
                let point = coset.shift * coset.generator().exp_u64(i as u64);
                assert_eq!(points[i], point);
                let reference = trace.selectors_at_point(point);
                assert_eq!(selectors.is_first_row[i], reference.is_first_row);
                assert_eq!(selectors.is_last_row[i], reference.is_last_row);
                assert_eq!(selectors.is_transition[i], reference.is_transition);
                assert_eq!(selectors.inv_vanishing[i], reference.inv_vanishing);
            }
        }
    }

    /// `selectors_on_coset` must agree pointwise with `selectors_at_point`
    /// evaluated at each coset point.
    #[test]
    fn coset_selectors_match_pointwise() {
        for log_size in [0usize, 1, 3] {
            for extra_bits in [0usize, 1, 2] {
                let trace = Radix2Coset {
                    log_size,
                    shift: ONE,
                };
                let coset = trace.create_disjoint_domain(1 << (log_size + extra_bits));
                let on_coset = trace.selectors_on_coset(coset);
                for (i, x) in coset.points().into_iter().enumerate() {
                    let at_point = trace.selectors_at_point(x);
                    assert_eq!(on_coset.is_first_row[i], at_point.is_first_row);
                    assert_eq!(on_coset.is_last_row[i], at_point.is_last_row);
                    assert_eq!(on_coset.is_transition[i], at_point.is_transition);
                    assert_eq!(on_coset.inv_vanishing[i], at_point.inv_vanishing);
                }
            }
        }
    }

    /// The disjoint domain must be disjoint: the vanishing polynomial of
    /// the trace domain is nonzero on every coset point.
    #[test]
    fn disjoint_domain_is_disjoint() {
        let trace = Radix2Coset {
            log_size: 4,
            shift: ONE,
        };
        let coset = trace.create_disjoint_domain(1 << 6);
        for x in coset.points() {
            assert!(!(x.exp_power_of_2(4) - ONE).is_zero());
        }
    }

    #[test]
    fn next_point_walks_the_domain() {
        let domain = Radix2Coset {
            log_size: 3,
            shift: ONE,
        };
        let points = domain.points();
        for i in 0..points.len() - 1 {
            assert_eq!(domain.next_point(points[i]), points[i + 1]);
        }
    }
}
