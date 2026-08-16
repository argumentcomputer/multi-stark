//! Plonky3 adapter: everything that binds this crate to the Plonky3
//! stack lives under this module (see `docs/pcs-abstraction.md`).
//!
//! - [`air`] (re-exported at this level for compatibility): build
//!   [`crate::system::CircuitInputs`] from a Plonky3-style AIR — the
//!   frontend adapter.
//! - [`challenger`]: [`crate::traits::Transcript`] for the p3-backed
//!   challengers.

mod air;
pub mod challenger;
pub mod domain;

pub use air::*;
