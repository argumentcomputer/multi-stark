//! Reusable constrained byte arithmetic and fixed-length BLAKE3 hashing.
//!
//! These gadgets compose with caller-owned values and leave public exposure
//! to the enclosing circuit. Hashing supports the generic gates and the
//! builder's optional compact BLAKE3 lowering.

mod blake3;
mod bytes;

pub use crate::p3_adapter::domain_blake3::Blake3Domain;
pub use blake3::{blake3, blake3_domain, blake3_xof};
pub use bytes::{ByteGadgets, ByteValue};
