//! Constraint restructure: a first-class expression IR for circuits.
//!
//! Replaces the Plonky3-inherited builder traits with data. Circuits are
//! described by frontend [`expr`] trees, compiled by [`circuit`] into a
//! flat, hash-consed, base-only node vector, and evaluated by [`eval`]'s
//! dense forward sweep. See `../../multi-stark-restructure.org` for the
//! design. Not yet wired into the prover/verifier.

pub mod circuit;
pub mod eval;
pub mod expr;
pub mod synth;

#[cfg(test)]
mod tests;
