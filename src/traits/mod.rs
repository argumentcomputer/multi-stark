//! Crate-owned trait layer: the abstraction boundary the prover and
//! verifier are written against, so a polynomial commitment backend
//! (Plonky3 FRI today, arkworks KZG next) plugs in via an adapter
//! module instead of leaking its traits through the core. See
//! `docs/pcs-abstraction.md` for the design and migration plan.
//!
//! Names are the natural ones for this crate; where they collide with a
//! backend's (p3 also has a `Pcs`), the backend import is renamed at the
//! adapter (`use p3_commit::Pcs as P3Pcs`), never the other way around.

mod transcript;

pub use transcript::Transcript;
