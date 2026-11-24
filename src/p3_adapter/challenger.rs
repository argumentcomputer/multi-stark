//! [`Transcript`] for the p3-backed production challengers.
//!
//! - [`Challenger`]: legacy `v5` unkeyed BLAKE3 challenger for `ExtVal` (degree-2).
//! - [`DomainChallenger<EF>`]: protocol `v6` domain-separated BLAKE3 challenger
//!   parameterized over any challenge extension `EF` (supporting both degree-2
//!   and degree-3 cubic trinomial extensions).

use core::marker::PhantomData;

use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger, HashChallenger,
    SerializingChallenger64,
};
use p3_field::{BasedVectorSpace, ExtensionField};
use p3_symmetric::Hash;

use crate::p3_adapter::domain_blake3::{Blake3Domain, DomainBlake3};
use crate::traits::Transcript;
use crate::types::{Challenger, Commitment, DeterministicPow, ExtVal, Val};

impl Transcript for Challenger {
    type F = Val;
    type Challenge = ExtVal;
    type Commitment = Commitment;

    #[inline]
    fn observe_field(&mut self, x: Val) {
        self.observe(x);
    }

    #[inline]
    fn observe_field_slice(&mut self, xs: &[Val]) {
        self.observe_slice(xs);
    }

    #[inline]
    fn observe_challenge(&mut self, x: ExtVal) {
        self.observe_algebra_element(x);
    }

    #[inline]
    fn observe_commitment(&mut self, c: Commitment) {
        self.observe(c);
    }

    #[inline]
    fn observe_digest(&mut self, digest: [u8; 32]) {
        self.observe(Hash::<Val, u8, 32>::from(digest));
    }

    #[inline]
    fn sample_challenge(&mut self) -> ExtVal {
        self.sample_algebra_element()
    }
}

/// Protocol `v6` domain-separated Goldilocks/BLAKE3 challenger parameterized
/// by the challenge extension field `EF` (e.g. `ExtVal3` or `ExtVal2`).
#[derive(Clone, Debug)]
pub struct DomainChallenger<EF> {
    inner: DeterministicPow<SerializingChallenger64<Val, HashChallenger<u8, DomainBlake3, 32>>>,
    _marker: PhantomData<EF>,
}

impl<EF> DomainChallenger<EF> {
    /// Creates a fresh domain-separated challenger initialized with `initial_state`
    /// under `domain` (`TranscriptSingle` or `TranscriptBatch`).
    pub fn from_domain(initial_state: Vec<u8>, domain: Blake3Domain) -> Self {
        Self {
            inner: DeterministicPow(SerializingChallenger64::from_hasher(
                initial_state,
                DomainBlake3::new(domain),
            )),
            _marker: PhantomData,
        }
    }
}

impl<EF, T> CanObserve<T> for DomainChallenger<EF>
where
    DeterministicPow<SerializingChallenger64<Val, HashChallenger<u8, DomainBlake3, 32>>>:
        CanObserve<T>,
{
    #[inline]
    fn observe(&mut self, value: T) {
        self.inner.observe(value);
    }
}

impl<EF, T> CanSample<T> for DomainChallenger<EF>
where
    DeterministicPow<SerializingChallenger64<Val, HashChallenger<u8, DomainBlake3, 32>>>:
        CanSample<T>,
{
    #[inline]
    fn sample(&mut self) -> T {
        self.inner.sample()
    }
}

impl<EF> CanSampleBits<usize> for DomainChallenger<EF> {
    #[inline]
    fn sample_bits(&mut self, bits: usize) -> usize {
        self.inner.sample_bits(bits)
    }
}

impl<EF: Clone + Send + Sync> FieldChallenger<Val> for DomainChallenger<EF> {}

impl<EF: Clone + Send + Sync> GrindingChallenger for DomainChallenger<EF> {
    type Witness = Val;

    #[inline]
    fn grind(&mut self, bits: usize) -> Self::Witness {
        self.inner.grind(bits)
    }
}

impl<EF> Transcript for DomainChallenger<EF>
where
    EF: ExtensionField<Val> + BasedVectorSpace<Val> + Clone + Send + Sync,
{
    type F = Val;
    type Challenge = EF;
    type Commitment = Commitment;

    #[inline]
    fn observe_field(&mut self, x: Val) {
        self.observe(x);
    }

    #[inline]
    fn observe_field_slice(&mut self, xs: &[Val]) {
        self.observe_slice(xs);
    }

    #[inline]
    fn observe_challenge(&mut self, x: EF) {
        self.observe_algebra_element(x);
    }

    #[inline]
    fn observe_commitment(&mut self, c: Commitment) {
        self.observe(c);
    }

    #[inline]
    fn observe_digest(&mut self, digest: [u8; 32]) {
        self.observe(Hash::<Val, u8, 32>::from(digest));
    }

    #[inline]
    fn sample_challenge(&mut self) -> EF {
        self.sample_algebra_element()
    }
}
