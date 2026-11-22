//! Proof-system configuration, parameter limits and backend acceleration hooks.
//! Associated types use the interfaces in [`crate::traits`].

use crate::traits::{ExtensionOf, Field, Pcs, Transcript};

/// The base (trace) field of a configuration's PCS.
pub type Val<SC> = <<SC as ProofConfig>::Pcs as Pcs>::F;

/// The evaluation domain type of a configuration's PCS.
pub type Domain<SC> = <<SC as ProofConfig>::Pcs as Pcs>::Domain;

/// The commitment type of a configuration's PCS.
pub type Com<SC> = <<SC as ProofConfig>::Pcs as Pcs>::Commitment;

/// The opening proof type of a configuration's PCS.
pub type PcsProof<SC> = <<SC as ProofConfig>::Pcs as Pcs>::Proof;

/// The error type of a configuration's PCS.
pub type PcsError<SC> = <<SC as ProofConfig>::Pcs as Pcs>::Error;

/// The prover data type of a configuration's PCS.
pub type PcsData<SC> = <<SC as ProofConfig>::Pcs as Pcs>::ProverData;

/// Result produced by an accelerated lookup-trace constructor.
pub type AcceleratedLookupTraces<SC> = (
    Vec<p3_matrix::dense::RowMajorMatrix<<SC as ProofConfig>::Challenge>>,
    Vec<<SC as ProofConfig>::Challenge>,
);

/// Result produced by an accelerated lookup commitment.
pub type AcceleratedLookupCommitment<SC> =
    (Com<SC>, PcsData<SC>, Vec<<SC as ProofConfig>::Challenge>);

/// Reference to the committed cross-circuit LogUp accumulator boundary mask
/// matrix `U` (matrix `#num_active` inside `stage_2_trace`) for multi-circuit
/// `v6-zk` proofs (`num_active > 1`).
pub struct ZkBoundaryMask<'a, SC: ProofConfig> {
    pub data: &'a PcsData<SC>,
    pub matrix_index: usize,
    pub active_pos: usize,
    pub num_active: usize,
}

impl<SC: ProofConfig> Clone for ZkBoundaryMask<'_, SC> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<SC: ProofConfig> Copy for ZkBoundaryMask<'_, SC> {}

/// One circuit's inputs to an optional fused quotient commitment backend.
/// Keeping this protocol-level description free of CUDA types lets the
/// generic CPU prover remain entirely independent of accelerator support.
pub struct QuotientCommitInput<'a, SC: ProofConfig> {
    pub circuit: &'a crate::system::Circuit<Val<SC>>,
    pub lookup_publics: Vec<Val<SC>>,
    pub trace_domain: Domain<SC>,
    pub quotient_domain: Domain<SC>,
    pub preprocessed: Option<(&'a PcsData<SC>, usize)>,
    pub stage_1: (&'a PcsData<SC>, usize),
    pub stage_2: (&'a PcsData<SC>, usize),
    pub constraint_count: usize,
    pub zk_boundary_mask: Option<ZkBoundaryMask<'a, SC>>,
}

/// One circuit's inputs to an optional fused lookup construction and
/// commitment backend. The committed stage-1 data lets an accelerator reuse
/// the witness already uploaded for the main-trace commitment.
pub struct LookupCommitInput<'a, SC: ProofConfig> {
    pub circuit: &'a crate::system::Circuit<Val<SC>>,
    pub lookup_values: &'a crate::lookup::LookupValues<Val<SC>>,
    pub preprocessed: Option<(&'a PcsData<SC>, usize)>,
    pub stage_1: (&'a PcsData<SC>, usize),
}

/// The borrowed evaluations view of a configuration's PCS.
pub type EvaluationsOnDomain<'a, SC> = <<SC as ProofConfig>::Pcs as Pcs>::Evaluations<'a>;

/// Packed (SIMD) representation of the base field.
pub type PackedVal<SC> = <Val<SC> as Field>::Packing;

/// Packed (SIMD) representation of the challenge field.
pub type PackedChallenge<SC> = <<SC as ProofConfig>::Challenge as ExtensionOf<Val<SC>>>::ExtPacking;

/// Protocol wire & cryptographic specification version.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ProtocolVersion {
    /// Legacy `v5` protocol: unkeyed BLAKE3, Horner FRI batching, shape-only
    /// transcript binding.
    V5,
    /// Hardened `v6` protocol: BLAKE3 `derive_key` domain separation across
    /// Merkle leaves / compressions / single-vs-batch transcripts / VK digest,
    /// canonical VK & constraint-graph digest plus extension degree/modulus
    /// bound in the transcript, and configurable multilinear tensor FRI batching.
    V6,
    /// Zero-knowledge `v6-zk` protocol: 2× interleaved odd-coset hiding PCS
    /// with zero-storage BLAKE3-CTR leaf salting (`multi-stark/v6-zk/*`),
    /// `W_R >= 4` base-field randomizer polynomial merged into quotient
    /// commitment, telescoping half-stride quotient blinding, minimum trace
    /// height floor, and boundary-mask blinding of `intermediate_accumulators`
    /// when multiple circuits are active.
    V6Zk,
}

/// Transcript mode separating standalone single-proof transcripts from
/// multi-shard batch transcripts.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TranscriptMode {
    /// Standalone single proof (`prove` / `verify`).
    Single,
    /// Multi-shard batch proof (`batch_challenger` / `prove_batch_shard` / `verify_batch`).
    Batch,
}

pub trait ProofConfig {
    /// The PCS used to commit to trace polynomials.
    type Pcs: Pcs<F: Field, Challenge = Self::Challenge, Challenger = Self::Challenger>;

    /// The field from which random challenges are drawn. Its size bounds the
    /// Schwartz-Zippel terms of the soundness error, so it must be large
    /// enough for the target security level (see the soundness argument in
    /// the verifier module docs).
    type Challenge: ExtensionOf<Val<Self>>;

    /// The Fiat-Shamir challenger.
    type Challenger: Transcript<F = Val<Self>, Challenge = Self::Challenge, Commitment = Com<Self>>;

    /// Returns a reference to the PCS.
    fn pcs(&self) -> &Self::Pcs;

    /// Protocol specification version governing transcript & hash domain
    /// separation. Defaults to [`ProtocolVersion::V5`] for backwards
    /// compatibility with existing configurations.
    fn protocol_version(&self) -> ProtocolVersion {
        ProtocolVersion::V5
    }

    /// Returns `true` when zero-knowledge mode (`ProtocolVersion::V6Zk`) is
    /// enabled. Non-ZK `v5` and `v6` configurations return `false` with zero
    /// runtime overhead.
    fn is_zk(&self) -> bool {
        self.protocol_version() == ProtocolVersion::V6Zk
    }

    /// Minimum admissible `log2(trace_height)` for active circuits (`0` in
    /// non-ZK modes; enforces `N >= next_pow2(2Q + 2D)` in `v6-zk`).
    fn min_log_degree(&self) -> usize {
        0
    }

    /// Base-field column width `W_R` of the randomizer polynomial `R(X)`
    /// merged into `quotient_chunks` (`0` in non-ZK modes; `>= 4` in `v6-zk`).
    fn zk_randomizer_width(&self) -> usize {
        0
    }

    /// Samples `count` uniformly random base-field elements from a CSPRNG for
    /// `v6-zk` blinding (e.g. cross-circuit accumulator boundary masking).
    fn sample_zk_base_randomness(&self, _count: usize) -> Vec<Val<Self>> {
        unimplemented!("ProofConfig does not support v6-zk CSPRNG sampling")
    }

    /// Commits ZK blinded quotient polynomials (`2 * q_zk` telescoping slices
    /// per active circuit) together with the `W_R`-column randomizer `R(X)` in
    /// one hiding MMCS round.
    fn commit_quotient_zk(
        &self,
        _quotients: Vec<(
            Domain<Self>,
            p3_matrix::dense::RowMajorMatrix<Val<Self>>,
            usize,
        )>,
    ) -> (Com<Self>, PcsData<Self>)
    where
        Self: Sized,
    {
        unimplemented!("ProofConfig does not support v6-zk quotient commitment")
    }

    /// Returns a fresh challenger.
    ///
    /// # Transcript contract
    /// Implementations must seed the challenger with a domain-separation tag
    /// and a digest of all protocol parameters (PCS configuration, security
    /// parameters), so that transcripts produced under different parameters
    /// never collide. The circuit shape is bound separately via
    /// `System::observe_shape`.
    fn initialise_challenger(&self) -> Self::Challenger;

    /// Returns a fresh challenger domain-separated for `mode` (single proof
    /// vs batch proof). Legacy `v5` configurations fall back to
    /// [`Self::initialise_challenger`]; `v6` configurations use distinct
    /// BLAKE3 `derive_key` contexts for `Single` and `Batch`.
    fn initialise_challenger_for_mode(&self, _mode: TranscriptMode) -> Self::Challenger {
        self.initialise_challenger()
    }

    /// Largest log2 length of a committed polynomial.
    fn max_log_degree(&self) -> usize;

    /// Omit next-row openings of main/fixed matrices when no graph node reads
    /// them. Configurations enabling this must bind it in their transcript tag.
    fn omit_unused_next_row_openings(&self) -> bool {
        false
    }

    /// Permit empty opening lists for inactive fixed matrices. The PCS must
    /// authenticate matrix boundaries without opening their values. Bind this
    /// choice in the transcript; row-batched Merkle commitments cannot use it.
    fn omit_inactive_preprocessed_openings(&self) -> bool {
        false
    }

    /// Largest log2 domain for computing the unsliced quotient.
    fn max_log_quotient_domain(&self) -> usize {
        self.max_log_degree()
    }

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

    /// Validate that `commit` has the exact expected structure (e.g. exact
    /// Merkle cap root count) for a commitment whose tallest committed LDE has
    /// height `2^max_log_lde_height`.
    ///
    /// Called by the verifier before observing any commitment into the
    /// Fiat-Shamir transcript so variable-length caps cannot collide with
    /// subsequent transcript fields or enable free challenge grinding.
    fn validate_commitment_cap(&self, _commit: &Com<Self>, _max_log_lde_height: usize) -> bool {
        true
    }

    /// Commit deterministic main-trace sources.
    fn commit_main(
        &self,
        evaluations: Vec<(Domain<Self>, crate::witness::TraceSource<Val<Self>>)>,
    ) -> (Com<Self>, PcsData<Self>)
    where
        Self: Sized,
    {
        self.pcs().commit(
            evaluations
                .into_iter()
                .map(|(domain, trace)| (domain, trace.materialize()))
                .collect(),
        )
    }

    /// Commit preprocessed evaluation matrices from a streaming iterator.
    ///
    /// Each item is `(domain, matrix, retain_1x_for_lookups)`, where
    /// `retain_1x_for_lookups` is `true` when the system does not retain
    /// `circuit.preprocessed` in RAM and the circuit's lookup expressions read
    /// preprocessed columns.
    ///
    /// Backends that spill LDEs to disk can process one matrix at a time so
    /// all preprocessed matrices never coexist in RAM simultaneously. Returns
    /// `None` if the iterator yields no matrices.
    fn commit_preprocessed(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (
                Domain<Self>,
                p3_matrix::dense::RowMajorMatrix<Val<Self>>,
                bool,
            ),
        >,
    ) -> Option<(Com<Self>, PcsData<Self>)>
    where
        Self: Sized,
    {
        let evals: Vec<_> = evaluations
            .map(|(domain, matrix, _retain_1x)| (domain, matrix))
            .collect();
        if evals.is_empty() {
            None
        } else {
            Some(self.pcs().commit(evals))
        }
    }

    /// Commit deterministic main-trace sources from a streaming iterator.
    ///
    /// Each item is `(domain, trace_source, retain_1x_for_lookups)`, where
    /// `retain_1x_for_lookups` is `true` when the circuit's lookup witness is
    /// unmaterialized (`shape_only`) and the circuit has at least one lookup.
    ///
    /// Returns `None` if the iterator yields no active traces.
    fn commit_main_streaming(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (Domain<Self>, crate::witness::TraceSource<Val<Self>>, bool),
        >,
    ) -> Option<(Com<Self>, PcsData<Self>)>
    where
        Self: Sized,
    {
        let evals: Vec<_> = evaluations
            .map(|(domain, trace, _retain_1x)| (domain, trace))
            .collect();
        if evals.is_empty() {
            None
        } else {
            Some(self.commit_main(evals))
        }
    }

    /// Normalize any backend-dependent field representatives before proof
    /// serialization. Most fields have canonical in-memory representations;
    /// configurations whose field permits lazy reduction can override this.
    fn canonicalize_proof(_proof: &mut crate::prover::Proof<Self>)
    where
        Self: Sized,
    {
    }

    /// Optional device-resident quotient evaluator. Implementations return
    /// `None` to use the portable host evaluator. Keeping this hook on the
    /// configuration preserves a CUDA-free PCS and prover for every other
    /// field/backend.
    #[allow(clippy::too_many_arguments)]
    fn accelerated_quotient_values(
        &self,
        _circuit: &crate::system::Circuit<Val<Self>>,
        _lookup_publics: &[Val<Self>],
        _trace_domain: Domain<Self>,
        _quotient_domain: Domain<Self>,
        _preprocessed: Option<(&PcsData<Self>, usize)>,
        _stage_1: (&PcsData<Self>, usize),
        _stage_2: (&PcsData<Self>, usize),
        _alpha: Self::Challenge,
        _constraint_count: usize,
    ) -> Option<Vec<Self::Challenge>>
    where
        Self: Sized,
    {
        None
    }

    /// Optional fused quotient evaluation, coefficient slicing, LDE and PCS
    /// commitment. Implementations returning `None` use the portable path.
    /// The result must commit to exactly the matrices produced by
    /// `shifted_quotient_slices` followed by
    /// `lde_from_shifted_coefficients`.
    fn accelerated_quotient_commit(
        &self,
        _inputs: &[QuotientCommitInput<'_, Self>],
        _alpha: Self::Challenge,
    ) -> Option<(Com<Self>, PcsData<Self>)>
    where
        Self: Sized,
    {
        None
    }

    /// Optional accelerator for the logUp message inversion and accumulator
    /// scan. The returned matrices retain the protocol's extension-valued
    /// row-major layout; the generic prover remains the reference fallback.
    fn accelerated_lookup_traces(
        &self,
        _circuits: &[crate::lookup::LookupValues<Val<Self>>],
        _group_sizes: &[usize],
        _lookup_challenge: Self::Challenge,
        _fingerprint_challenge: Self::Challenge,
        _accumulator: Self::Challenge,
    ) -> Option<AcceleratedLookupTraces<Self>>
    where
        Self: Sized,
    {
        None
    }

    /// Optional fused lookup construction, LDE and commitment. This avoids
    /// forcing accelerator-native trace storage through host matrices merely
    /// to satisfy the portable `Pcs::commit` interface.
    fn accelerated_lookup_commit(
        &self,
        _inputs: &[LookupCommitInput<'_, Self>],
        _lookup_challenge: Self::Challenge,
        _fingerprint_challenge: Self::Challenge,
        _accumulator: Self::Challenge,
    ) -> Option<AcceleratedLookupCommitment<Self>>
    where
        Self: Sized,
    {
        None
    }
}

// Compatibility name for existing configurations.
pub use ProofConfig as StarkGenericConfig;
