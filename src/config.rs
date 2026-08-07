//! Generic STARK configuration.
//!
//! [`StarkGenericConfig`] bundles the three choices that instantiate the
//! protocol: the polynomial commitment scheme, the challenge (extension)
//! field, and the Fiat-Shamir challenger. The base field is determined
//! transitively by the PCS ([`Val`]). The prover, verifier and system are
//! generic over an implementation of this trait; see
//! [`crate::types::GoldilocksBlake3Config`] for the reference instantiation.

use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{ExtensionField, Field};

/// The base field of a configuration, as determined by its PCS domain.
pub type Val<SC> = <<<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain as PolynomialSpace>::Val;

/// The evaluation domain type of a configuration's PCS.
pub type Domain<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain;

/// The commitment type of a configuration's PCS.
pub type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;

/// The opening proof type of a configuration's PCS.
pub type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;

/// The error type of a configuration's PCS.
pub type PcsError<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Error;

/// The prover data type of a configuration's PCS.
pub type PcsData<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::ProverData;

/// Evaluations of committed polynomials over a domain.
pub type EvaluationsOnDomain<'a, SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::EvaluationsOnDomain<'a>;

/// Packed base-field values of a configuration.
pub type PackedVal<SC> = <Val<SC> as Field>::Packing;

/// Packed challenge-field values of a configuration.
pub type PackedChallenge<SC> =
    <<SC as StarkGenericConfig>::Challenge as ExtensionField<Val<SC>>>::ExtensionPacking;

/// Configuration of a STARK system.
pub trait StarkGenericConfig {
    /// The PCS used to commit to trace polynomials.
    type Pcs: Pcs<Self::Challenge, Self::Challenger>;

    /// The field from which random challenges are drawn. Its size bounds the
    /// Schwartz-Zippel terms of the soundness error, so it must be large
    /// enough for the target security level (see the soundness argument in
    /// the verifier module docs).
    type Challenge: ExtensionField<Val<Self>>;

    /// The Fiat-Shamir challenger.
    type Challenger: FieldChallenger<Val<Self>> + CanObserve<Com<Self>> + CanSample<Self::Challenge>;

    /// Returns a reference to the PCS.
    fn pcs(&self) -> &Self::Pcs;

    /// Returns a fresh challenger.
    ///
    /// # Transcript contract
    /// Implementations must seed the challenger with a domain-separation tag
    /// and a digest of all protocol parameters (PCS configuration, security
    /// parameters), so that transcripts produced under different parameters
    /// never collide. The circuit shape is bound separately via
    /// `System::observe_shape`.
    fn initialise_challenger(&self) -> Self::Challenger;

    /// The largest log2 polynomial degree the PCS can commit to and open.
    ///
    /// The verifier rejects proofs whose claimed trace degree, multiplied by
    /// the quotient degree, exceeds this bound. For a FRI-based PCS this is
    /// the field's two-adicity minus the log blowup.
    fn max_log_degree(&self) -> usize;

    /// The largest quotient degree — as a multiple of the trace degree —
    /// that the PCS can serve trace evaluations for.
    ///
    /// The prover evaluates the constraints on a domain `quotient_degree`
    /// times larger than the trace domain, obtained from the PCS via
    /// `get_evaluations_on_domain`. For a FRI-based PCS this only works up
    /// to the blowup factor: the committed low-degree extension has
    /// `2^log_blowup · N` evaluations, and asking for a larger domain
    /// produces invalid proofs. Since the quotient degree is
    /// `next_power_of_two(max_constraint_degree - 1)`, this bounds the
    /// constraint degree: `2^log_blowup + 1` (degree 3 at `log_blowup = 1`).
    ///
    /// [`System::new`](crate::system::System::new) rejects circuits whose
    /// constraint degree requires a larger quotient degree.
    fn max_quotient_degree(&self) -> usize;

    /// Log2 of the blowup the PCS applies when committing: a degree-`N`
    /// trace is stored as a low-degree extension with `2^log_blowup · N`
    /// evaluations.
    ///
    /// This must be the blowup `Pcs::commit` ACTUALLY applies, not a bound:
    /// the prover uses it to rebuild committed LDEs directly from
    /// polynomial coefficients (see `lde_from_coefficients` in the prover
    /// module), and a mismatch produces commitments to the wrong
    /// evaluations.
    fn log_blowup(&self) -> usize;
}
