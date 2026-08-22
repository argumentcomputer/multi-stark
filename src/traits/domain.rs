//! The evaluation-domain surface the core actually uses.
//!
//! Both backends run over two-adic multiplicative subgroups; what differs
//! is who hands the domains out (the PCS) and how the quotient-sweep
//! coset is realised. Associated types, for the same inference reasons as
//! [`super::Transcript`]: one (base field, challenge field) pair per
//! domain type per configuration.

/// Lagrange selector values over some evaluation context `T` (a point's
/// worth of challenge-field values, or per-coset-row base-field vectors).
/// Mirrors the classic quadruple: first row, last row, transition, and
/// the inverse vanishing polynomial.
pub struct LagrangeSelectors<T> {
    pub is_first_row: T,
    pub is_last_row: T,
    pub is_transition: T,
    pub inv_vanishing: T,
}

/// A (sub)group evaluation domain of power-of-two size.
pub trait EvaluationDomain: Copy {
    /// Base (trace) field.
    type F;
    /// Challenge field for out-of-domain points.
    type Challenge;

    fn size(&self) -> usize;

    /// The domain's first point (its coset shift; the subgroup identity
    /// for unshifted domains).
    fn first_point(&self) -> Self::F;

    /// `x * g` for the domain's group generator — the "next row" map used
    /// to build the (zeta, zeta*g) opening pairs.
    fn next_point(&self, x: Self::Challenge) -> Self::Challenge;

    /// A domain of at least `min_size` points disjoint from this one, for
    /// the quotient sweep (the vanishing polynomial of `self` must be
    /// nonzero on every point of the result).
    fn create_disjoint_domain(&self, min_size: usize) -> Self;

    /// Selector values at one out-of-domain point.
    fn selectors_at_point(&self, point: Self::Challenge) -> LagrangeSelectors<Self::Challenge>;

    /// Selector values at every point of `coset` (a disjoint domain as
    /// produced by [`Self::create_disjoint_domain`]).
    fn selectors_on_coset(&self, coset: Self) -> LagrangeSelectors<Vec<Self::F>>;
}
