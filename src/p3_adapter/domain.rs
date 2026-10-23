//! [`EvaluationDomain`] for the p3 two-adic coset domain used by the production
//! (Goldilocks/FRI) configuration.

use p3_commit::{LagrangeSelectors as P3LagrangeSelectors, PolynomialSpace};
use p3_field::coset::TwoAdicMultiplicativeCoset;

use crate::traits::{EvaluationDomain, LagrangeSelectors};
use crate::types::{ExtVal, Val};

fn convert<T>(sels: P3LagrangeSelectors<T>) -> LagrangeSelectors<T> {
    LagrangeSelectors {
        is_first_row: sels.is_first_row,
        is_last_row: sels.is_last_row,
        is_transition: sels.is_transition,
        inv_vanishing: sels.inv_vanishing,
    }
}

impl EvaluationDomain for TwoAdicMultiplicativeCoset<Val> {
    type F = Val;
    type Challenge = ExtVal;

    #[inline]
    fn size(&self) -> usize {
        PolynomialSpace::size(self)
    }

    #[inline]
    fn first_point(&self) -> Val {
        PolynomialSpace::first_point(self)
    }

    #[inline]
    fn next_point(&self, x: ExtVal) -> ExtVal {
        PolynomialSpace::next_point(self, x).expect("two-adic domain has a next point")
    }

    #[inline]
    fn create_disjoint_domain(&self, min_size: usize) -> Self {
        PolynomialSpace::create_disjoint_domain(self, min_size)
    }

    #[inline]
    fn selectors_at_point(&self, point: ExtVal) -> LagrangeSelectors<ExtVal> {
        convert(PolynomialSpace::selectors_at_point(self, point))
    }

    #[inline]
    fn selectors_on_coset(&self, coset: Self) -> LagrangeSelectors<Vec<Val>> {
        convert(PolynomialSpace::selectors_on_coset(self, coset))
    }
}
