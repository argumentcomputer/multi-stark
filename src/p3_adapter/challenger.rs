//! [`Transcript`] for the p3-backed production challenger.
//!
//! Delegates to the Plonky3 challenger traits the concrete type already
//! implements; the transcript bytes are identical by construction (the
//! `proof_bytes_pin` test enforces it).

use p3_challenger::{CanObserve, FieldChallenger};

use crate::traits::Transcript;
use crate::types::{Challenger, Commitment, ExtVal, Val};

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
    fn sample_challenge(&mut self) -> ExtVal {
        self.sample_algebra_element()
    }
}
