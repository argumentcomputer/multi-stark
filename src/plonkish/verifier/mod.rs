//! Fixed-profile multi-stark verification with transcript, algebra, PCS and FRI checks.
//! [`VerifierPlan`] validates a trusted key and profile; callers bind the statement.
//! Proof assignment never overwrites statement wires. See `docs/plonkish-verifier.md`.
//!
//! Standalone algebraic and transcript helpers do not verify a complete proof.

mod algebra;
mod batch;
pub use batch::{
    build_single_batch_verifier, constrain_single_batch_verifier, expand_single_batch_witness,
};
mod pcs;
mod pcs_witness;
mod quadratic;
mod transcript;

pub use super::gadgets::{ByteGadgets, ByteValue, blake3};
pub use algebra::{
    AlgebraicChallenges, AlgebraicInputs, AlgebraicOutputs, CircuitEvaluation, CircuitOpenings,
    Selectors, constrain_algebraic_checks,
};
pub use pcs::{
    FixedPcsShape, FixedVerifierInputs, PcsInputs, PcsQuery, build_fixed_verifier,
    build_profile_verifier, constrain_fixed_verifier,
};
pub use pcs_witness::{ExpandedPcsWitness, expand_pcs_witness};
pub use quadratic::QuadraticValue;
pub use transcript::{
    Blake3Challenger, DEFAULT_FIELD_RETRIES, TranscriptCommitments, constrain_transcript,
};

mod key_codec;
mod plan;
pub use plan::{
    Envelope, ImplementationOptions, Message, PreparedProof, ProofEnvelope, ProofProfile,
    ResourceEstimate, Statement, StatementBinding, StatementSlot, VerifierError, VerifierInputs,
    VerifierKey, VerifierLimits, VerifierPlan,
};

#[cfg(feature = "groth16")]
mod query_shards;
#[cfg(feature = "groth16")]
pub use query_shards::{
    QueryShardPlan, encode_query_bundle, verify_encoded_query_bundle, verify_query_bundle,
};
