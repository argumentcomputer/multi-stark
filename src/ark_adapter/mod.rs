//! The arkworks/KZG backend: crate traits implemented over BLS12-381.
//!
//! The counterpart of [`crate::p3_adapter`] for the second PCS the
//! abstraction exists for (docs/pcs-abstraction.md, Phase 1). The
//! challenge field is the base field itself (`D = 1` — BLS12-381's
//! scalar field is already ~2^255, so no extension is needed), packing
//! is scalar (width 1), and commitments are monomial-basis KZG: one
//! G1 point per column, opening proofs one G1 point per distinct
//! opening point regardless of query count.
//!
//! Modules:
//! - [`field`]: [`field::Scalar`], a serde-able newtype over `Fr`
//!   carrying the crate field traits.
//! - [`domain`]: [`domain::Radix2Coset`], the crate-owned evaluation
//!   domain (the p3 coset semantics, reimplemented over `Scalar`).
//! - [`transcript`]: [`transcript::Blake3Transcript`], a byte-oriented
//!   Fiat-Shamir challenger.
//! - [`srs`]: the structured reference string.
//! - [`pcs`]: [`pcs::KzgPcs`] and the proof/commitment types.
//! - [`config`]: [`config::KzgConfig`], the [`ProofConfig`]
//!   instantiation tying it all together.
//!
//! [`ProofConfig`]: crate::config::ProofConfig

pub mod config;
pub mod domain;
pub mod field;
pub mod pcs;
pub mod srs;
pub mod transcript;

pub use config::KzgConfig;
pub use domain::Radix2Coset;
pub use field::Scalar;
pub use pcs::{KzgCommitment, KzgPcs, KzgProof};
pub use srs::Srs;
pub use transcript::Blake3Transcript;
