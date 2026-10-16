//! The Fiat-Shamir transcript surface the core actually uses.
//!
//! Associated types (not generics): a transcript serves exactly one
//! (field, challenge, commitment) triple per configuration, so
//! associated types keep every call site inference-unambiguous —
//! `observe_field(x)` needs no turbofish — and sidestep the
//! unconstrained-parameter (E0207) problems a blanket impl over an
//! erased base field would hit.
//!
//! Semantics each implementation must honour (they define the
//! transcript format, which is consensus-critical):
//! - `observe_field`: absorb one base-field element canonically.
//! - `observe_field_slice`: absorb elements in order; MUST equal
//!   observing each element individually.
//! - `observe_challenge`: absorb a challenge-field element by its basis
//!   coefficients over the base field, in order (degree 1: the element
//!   itself).
//! - `observe_commitment`: absorb a commitment by its canonical byte
//!   encoding (FRI: Merkle cap digest limbs; KZG: compressed G1 bytes).
//! - `sample_challenge`: squeeze one challenge-field element.

/// Transcript operations for one proof configuration.
pub trait Transcript {
    /// Base (trace) field.
    type F;
    /// Challenge field (a based vector space over `F`; degree may be 1).
    type Challenge;
    /// Commitment type absorbed into the transcript.
    type Commitment;

    fn observe_field(&mut self, x: Self::F);
    fn observe_field_slice(&mut self, xs: &[Self::F]);
    fn observe_challenge(&mut self, x: Self::Challenge);
    fn observe_commitment(&mut self, c: Self::Commitment);
    fn sample_challenge(&mut self) -> Self::Challenge;
}
