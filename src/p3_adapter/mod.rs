//! Plonky3 AIR, field, transcript and FRI adapters.
//! AIR helpers are re-exported here for compatibility.

mod air;
#[cfg(not(feature = "cuda"))]
pub(crate) mod blake3_simd;
pub mod cap;
pub mod challenger;
pub mod domain;
pub mod domain_blake3;
pub mod field;
pub mod fri_batching;
pub mod pcs;
#[cfg(not(feature = "cuda"))]
pub mod spill;
#[cfg(not(feature = "cuda"))]
pub mod spill_zk;
pub mod zk;

pub use air::*;
