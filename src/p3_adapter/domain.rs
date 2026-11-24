//! [`EvaluationDomain`] for the p3 two-adic coset domains used by the
//! production (Goldilocks/FRI) configurations.
//!
//! - `TwoAdicMultiplicativeCoset<Val>` implements [`EvaluationDomain`] with
//!   `Challenge = ExtVal` for legacy `v5` [`crate::types::GoldilocksBlake3Config`].
//! - [`CosetDomain<F, EF>`] wraps `TwoAdicMultiplicativeCoset<F>` transparently
//!   so protocol `v6` configurations can instantiate [`EvaluationDomain`] for
//!   any challenge extension `EF` (including degree-3 `ExtVal3` and degree-2
//!   `ExtVal2`).

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use p3_commit::{
    LagrangeSelectors as P3LagrangeSelectors, OpenedValues, Pcs as P3Pcs, PeriodicLdeTable,
    PolynomialSpace,
};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

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

/// Extension-parameterized wrapper around [`TwoAdicMultiplicativeCoset<F>`].
#[derive(Debug)]
#[repr(transparent)]
pub struct CosetDomain<F: TwoAdicField, EF> {
    pub inner: TwoAdicMultiplicativeCoset<F>,
    _marker: PhantomData<EF>,
}

impl<F: TwoAdicField, EF> Copy for CosetDomain<F, EF> {}

impl<F: TwoAdicField, EF> Clone for CosetDomain<F, EF> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: TwoAdicField, EF> CosetDomain<F, EF> {
    #[inline]
    pub const fn new(inner: TwoAdicMultiplicativeCoset<F>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub const fn into_inner(self) -> TwoAdicMultiplicativeCoset<F> {
        self.inner
    }

    #[inline]
    pub const fn log_size(&self) -> usize {
        self.inner.log_size()
    }

    #[inline]
    pub const fn shift(&self) -> F {
        self.inner.shift()
    }

    #[inline]
    pub fn subgroup_generator(&self) -> F {
        self.inner.subgroup_generator()
    }
}

impl<F: TwoAdicField, EF> PolynomialSpace for CosetDomain<F, EF> {
    type Val = F;

    #[inline]
    fn size(&self) -> usize {
        PolynomialSpace::size(&self.inner)
    }

    #[inline]
    fn first_point(&self) -> Self::Val {
        PolynomialSpace::first_point(&self.inner)
    }

    #[inline]
    fn next_point<Ext: ExtensionField<Self::Val>>(&self, x: Ext) -> Option<Ext> {
        PolynomialSpace::next_point(&self.inner, x)
    }

    #[inline]
    fn try_create_disjoint_domain(&self, min_size: usize) -> Option<Self> {
        PolynomialSpace::try_create_disjoint_domain(&self.inner, min_size).map(Self::new)
    }

    #[inline]
    fn split_domains(&self, num_chunks: usize) -> Vec<Self> {
        PolynomialSpace::split_domains(&self.inner, num_chunks)
            .into_iter()
            .map(Self::new)
            .collect()
    }

    #[inline]
    fn split_evals(
        &self,
        num_chunks: usize,
        evals: RowMajorMatrix<Self::Val>,
    ) -> Vec<RowMajorMatrix<Self::Val>> {
        PolynomialSpace::split_evals(&self.inner, num_chunks, evals)
    }

    #[inline]
    fn vanishing_poly_at_point<Ext: ExtensionField<Self::Val>>(&self, point: Ext) -> Ext {
        PolynomialSpace::vanishing_poly_at_point(&self.inner, point)
    }

    #[inline]
    fn selectors_at_point<Ext: ExtensionField<Self::Val>>(
        &self,
        point: Ext,
    ) -> P3LagrangeSelectors<Ext> {
        PolynomialSpace::selectors_at_point(&self.inner, point)
    }

    #[inline]
    fn selectors_on_coset(&self, coset: Self) -> P3LagrangeSelectors<Vec<Self::Val>> {
        PolynomialSpace::selectors_on_coset(&self.inner, coset.inner)
    }

    #[inline]
    fn evaluate_polynomial_at<Ext: ExtensionField<Self::Val>>(
        &self,
        evals: &[Self::Val],
        point: Ext,
    ) -> Ext {
        PolynomialSpace::evaluate_polynomial_at(&self.inner, evals, point)
    }

    #[inline]
    fn evaluate_periodic_column_at<Ext: ExtensionField<Self::Val>>(
        &self,
        col: &[Self::Val],
        point: Ext,
    ) -> Ext {
        PolynomialSpace::evaluate_periodic_column_at(&self.inner, col, point)
    }

    #[inline]
    fn evaluate_periodic_columns_at<Ext: ExtensionField<Self::Val>>(
        &self,
        periodic_columns: &[Vec<Self::Val>],
        point: Ext,
    ) -> Vec<Ext> {
        PolynomialSpace::evaluate_periodic_columns_at(&self.inner, periodic_columns, point)
    }
}

impl<F: TwoAdicField, EF: ExtensionField<F>> EvaluationDomain for CosetDomain<F, EF> {
    type F = F;
    type Challenge = EF;

    #[inline]
    fn size(&self) -> usize {
        PolynomialSpace::size(&self.inner)
    }

    #[inline]
    fn first_point(&self) -> F {
        PolynomialSpace::first_point(&self.inner)
    }

    #[inline]
    fn next_point(&self, x: EF) -> EF {
        PolynomialSpace::next_point(&self.inner, x).expect("two-adic domain has a next point")
    }

    #[inline]
    fn create_disjoint_domain(&self, min_size: usize) -> Self {
        Self::new(PolynomialSpace::create_disjoint_domain(
            &self.inner,
            min_size,
        ))
    }

    #[inline]
    fn selectors_at_point(&self, point: EF) -> LagrangeSelectors<EF> {
        convert(PolynomialSpace::selectors_at_point(&self.inner, point))
    }

    #[inline]
    fn selectors_on_coset(&self, coset: Self) -> LagrangeSelectors<Vec<F>> {
        convert(PolynomialSpace::selectors_on_coset(
            &self.inner,
            coset.inner,
        ))
    }
}

/// Transparent PCS adapter mapping `TwoAdicMultiplicativeCoset<F>` domains
/// into [`CosetDomain<F, EF>`], allowing any two-adic PCS backend (CPU
/// `SpillableTwoAdicFriPcs` or CUDA `CudaTwoAdicFriPcs`) to serve multiple
/// challenge extensions (`ExtVal2`, `ExtVal3`) without coherence conflicts.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct DomainMappedPcs<Inner> {
    inner: Inner,
}

impl<Inner> DomainMappedPcs<Inner> {
    #[inline]
    pub const fn new(inner: Inner) -> Self {
        Self { inner }
    }

    #[inline]
    pub const fn inner(&self) -> &Inner {
        &self.inner
    }

    #[inline]
    pub fn inner_mut(&mut self) -> &mut Inner {
        &mut self.inner
    }
}

impl<Inner> Deref for DomainMappedPcs<Inner> {
    type Target = Inner;
    #[inline]
    fn deref(&self) -> &Inner {
        &self.inner
    }
}

impl<Inner> DerefMut for DomainMappedPcs<Inner> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Inner {
        &mut self.inner
    }
}

impl<F, EF, Challenger, Inner> P3Pcs<EF, Challenger> for DomainMappedPcs<Inner>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Inner: P3Pcs<EF, Challenger, Domain = TwoAdicMultiplicativeCoset<F>>,
{
    type Domain = CosetDomain<F, EF>;
    type Commitment = Inner::Commitment;
    type ProverData = Inner::ProverData;
    type EvaluationsOnDomain<'a> = Inner::EvaluationsOnDomain<'a>;
    type Proof = Inner::Proof;
    type Error = Inner::Error;
    const ZK: bool = Inner::ZK;

    #[inline]
    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        CosetDomain::new(self.inner.natural_domain_for_degree(degree))
    }

    #[inline]
    fn log_max_lde_height(&self) -> usize {
        self.inner.log_max_lde_height()
    }

    #[inline]
    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<F>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        self.inner.commit(
            evaluations
                .into_iter()
                .map(|(domain, evals)| (domain.inner, evals)),
        )
    }

    #[inline]
    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<F>)>,
        num_chunks: usize,
    ) -> Vec<RowMajorMatrix<F>> {
        self.inner.get_quotient_ldes(
            evaluations
                .into_iter()
                .map(|(domain, evals)| (domain.inner, evals)),
            num_chunks,
        )
    }

    #[inline]
    fn commit_ldes(&self, ldes: Vec<RowMajorMatrix<F>>) -> (Self::Commitment, Self::ProverData) {
        self.inner.commit_ldes(ldes)
    }

    #[inline]
    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        self.inner
            .get_evaluations_on_domain(prover_data, idx, domain.inner)
    }

    #[inline]
    fn open(
        &self,
        rounds: Vec<(&Self::ProverData, Vec<Vec<EF>>)>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<EF>, Self::Proof) {
        self.inner.open(rounds, challenger)
    }

    #[inline]
    fn verify(
        &self,
        rounds: Vec<(Self::Commitment, Vec<(Self::Domain, Vec<(EF, Vec<EF>)>)>)>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let mapped = rounds
            .into_iter()
            .map(|(com, mats)| {
                (
                    com,
                    mats.into_iter()
                        .map(|(domain, pts)| (domain.inner, pts))
                        .collect(),
                )
            })
            .collect();
        self.inner.verify(mapped, proof, challenger)
    }

    #[inline]
    fn build_periodic_lde_table(
        &self,
        periodic_cols: &[Vec<F>],
        trace_domain: Self::Domain,
        quotient_domain: Self::Domain,
    ) -> PeriodicLdeTable<F> {
        self.inner.build_periodic_lde_table(
            periodic_cols,
            trace_domain.inner,
            quotient_domain.inner,
        )
    }
}
