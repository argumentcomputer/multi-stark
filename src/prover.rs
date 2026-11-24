//! Multi-circuit STARK prover, constraint-IR version.
//!
//! The composition polynomial is built by running the compiled circuit's
//! dense sweep in `PackedVal` on the quotient domain and folding the
//! constraint values with the constraint challenge. Stage-2 columns are
//! treated as ordinary base columns (they were committed flattened), so no
//! extension packing is needed here.
//!
//! The proving protocol proceeds in several stages sharing one Fiat-Shamir
//! transcript. The challenger starts from a seed binding a domain tag and all
//! protocol parameters (see the transcript contract on
//! [`ProofConfig::initialise_challenger`]), and the system shape
//! (circuit count, widths, constraint counts, degrees)
//! is observed before any commitment.
//!
//! 0. **Sparse activation**: circuits whose stage-1 trace is empty are
//!    deactivated for this proof — skipped by every stage below — and the
//!    activation bitmap over the canonical circuit set is observed
//!    immediately after the shape, before any commitment or challenge. See
//!    [`Proof::active`] for the soundness contract.
//!
//! 1. **Stage 1 — Main traces**: Each circuit's execution trace is committed via the
//!    configuration's PCS. The preprocessed commitment (if any), stage-1 commitment,
//!    trace heights, and length-prefixed claims are observed into the challenger.
//!    Claims must be observed before lookup challenges are sampled; otherwise the
//!    prover could choose claims adaptively to balance the lookup accumulator.
//!
//! 2. **Lookup challenges**: The challenger samples two independent challenges:
//!    `lookup_argument_challenge` (β) and `fingerprint_challenge` (γ). An initial
//!    accumulator is computed from the claims:
//!    `acc = Σ (β + fingerprint(γ, claim_i))⁻¹`.
//!
//! 3. **Stage 2 — Lookup traces**: For each circuit, the lookup traces are computed
//!    (one chained partial accumulator per lookup; no message inverses are
//!    committed) and committed via PCS. Each
//!    circuit produces an intermediate accumulator value recording where its running
//!    sum ended up; these are observed into the challenger, and the verifier will
//!    check that the last one is zero.
//!
//! 4. **Quotient polynomial**: A constraint challenge (α) is sampled and used to fold
//!    all constraints via powers of α. The folded constraint polynomial is divided by
//!    the vanishing polynomial, split into degree-bounded chunks, and committed.
//!
//! 5. **Opening**: An out-of-domain point (ζ) is sampled. All polynomials are opened
//!    at ζ and ζ·g (where g is the trace domain generator) via the PCS, producing
//!    the FRI opening proof.
//!
//! The resulting [`Proof`] can be serialized with [`Proof::to_bytes`] and deserialized
//! with [`Proof::from_bytes`] for transport or storage.
//!
//! # Prover cost analysis
//!
//! The prover's computational work divides into polynomial commitments (FFT +
//! Merkle hashing), lookup trace construction, constraint evaluation, and the
//! FRI opening protocol. This section gives a concrete cost breakdown.
//!
//! ## Notation
//!
//! We use the following notation throughout:
//! - C — number of circuits in the system
//! - n_i — trace height (rows, power of two) of circuit i
//! - w_i — stage 1 trace width (columns) of circuit i
//! - p_i — preprocessed trace width of circuit i (0 if none)
//! - L_i — number of lookups in circuit i
//! - k_i — number of constraints in circuit i (after lookup expansion)
//! - d_i — maximum constraint degree multiple of circuit i
//! - q_i = next_pow2(max(d_i, 2) − 1) — quotient polynomial degree
//! - D — dimension of the challenge (extension) field over the base field
//!   (2 in the reference config)
//! - B = 2^log_blowup — FRI blowup factor
//! - Q = num_queries — FRI query repetitions
//! - a = max_log_arity — FRI folding arity (log₂)
//!
//! Derived quantities:
//! - s_i — lookup group size of circuit i (1 = ungrouped)
//! - w2_i = max(⌈L_i/s_i⌉, 1) — stage 2 width in extension field elements
//! - W_i = w_i + w2_i · D + q_i · D — total committed width per circuit (base field)
//! - H = max_i(n_i) · B — largest LDE height
//! - R = ⌈(log₂ H − log_final_poly_len) / a⌉ — FRI folding rounds
//!
//! ## Stage 1 commit
//!
//! Each circuit's main trace (n_i × w_i) undergoes a coset LDE (FFT-based)
//! expanding from n_i to n_i · B evaluations, followed by Merkle tree
//! construction over the LDE rows. All circuits are committed together.
//!
//! ```text
//! FFT work:   Σ_i  w_i · (B+1) · n_i · log₂(n_i)   field multiplications
//! Hashing:    Σ_i  n_i · B                            Merkle leaf hashes
//! ```
//!
//! ## Lookup trace construction
//!
//! For each circuit with L_i > 0 lookups, the prover computes per-row
//! fingerprints (Horner evaluations in the extension field), batch-inverts
//! all messages via Montgomery's trick (≈ 3 extension multiplications per
//! element), and builds the running accumulator.
//!
//! ```text
//! Fingerprints:  Σ_i  n_i · L_i · |args|   extension field multiplications
//! Inversions:    Σ_i  3 · n_i · L_i         extension field multiplications
//! Accumulator:   Σ_i  n_i · L_i             extension field multiply-adds
//! ```
//!
//! ## Stage 2 commit
//!
//! Stage 2 traces (n_i × w2_i extension elements, flattened to
//! n_i × w2_i · D base elements) are committed identically to stage 1.
//!
//! ```text
//! FFT work:   Σ_i  w2_i · D · (B+1) · n_i · log₂(n_i)   field multiplications
//! Hashing:    Σ_i  n_i · B                                 Merkle leaf hashes
//! ```
//!
//! ## Quotient computation and commit
//!
//! For each circuit, the prover evaluates all k_i constraints at every point
//! of the quotient domain (size n_i · q_i) and divides by the vanishing
//! polynomial. If q_i ≤ B, the trace evaluations on the quotient domain are
//! obtained by subsetting the LDE (essentially free); otherwise an additional
//! iFFT + FFT is required.
//!
//! The quotient polynomial is split into q_i coefficient slices (each of
//! degree n_i) and committed as a single q_i·D-column matrix per circuit on
//! the trace domain, so the opening phase pays its per-matrix costs once per
//! circuit rather than once per slice. The committed LDE is built directly
//! from the slice coefficients, skipping the trace-domain DFT whose output
//! `Pcs::commit` would immediately invert: one forward DFT recovers the
//! coefficients, a fused parallel gather slices them with the LDE's coset
//! shift folded in (`shifted_quotient_slices`), and one zero-padded DFT
//! produces the committed LDE (`lde_from_shifted_coefficients`).
//!
//! ```text
//! Constraint eval:  Σ_i  n_i · q_i · eval_cost(k_i)         field operations
//! Quotient iFFT:    Σ_i  D · q_i · n_i · log₂(q_i · n_i)    field multiplications
//! Quotient LDE:     Σ_i  q_i · D · B · n_i · log₂(B · n_i)  field multiplications
//! Hashing:          Σ_i  q_i · n_i · B                        Merkle leaf hashes
//! ```
//!
//! Here eval_cost(k_i) denotes the per-row cost of evaluating all k_i folded
//! constraints, which depends on the compiled constraint sweep.
//!
//! ## FRI opening
//!
//! All polynomials (stage 1, stage 2, quotient, and preprocessed if present)
//! are opened at ζ and ζ·g via barycentric interpolation, then verified
//! through FRI. The preprocessed traces' LDE was computed at setup time
//! ([`System::new`]); only their opening contributes to per-proof cost.
//!
//! ```text
//! Interpolation:  Σ_i  n_i · B · W_i     field multiply-adds (barycentric)
//! FRI folding:    ≈ H · 2^a / (2^a − 1)  extension field operations
//! FRI queries:    Q · R · log₂ H          hash operations (Merkle paths)
//! FRI grinding:   2^pow_commit · R + 2^pow_query   hash invocations
//! ```
//!
//! ## Overall cost
//!
//! The total per-proof cost is approximately:
//!
//! ```text
//! C_prove  ≈  Σ_i  (B+1) · n_i · log₂(n_i) · W_i     (FFT — all commit rounds)
//!           + Σ_i  n_i · q_i · eval_cost(k_i)          (constraint evaluation)
//!           + Σ_i  n_i · B · W_i                        (barycentric interpolation)
//!           + Q · R · log₂ H                            (FRI query phase)
//!           + 2^pow_commit · R + 2^pow_query            (FRI grinding)
//! ```
//!
//! For typical parameters (B = 2, Q = 100, a = 1):
//! - **FFT dominates** when traces have large n_i · W_i products.
//! - **Constraint evaluation** can dominate for circuits with many complex
//!   constraints (large k_i) or high quotient degree (large q_i).
//! - **FRI queries** grow logarithmically in trace height but linearly in Q.
//! - **Lookup computation** is subdominant unless L_i is very large.
//! - **Grinding** is a constant overhead per FRI round (or once for queries).
//!
//! Cost scales **linearly** in trace height n_i for fixed circuit structure,
//! with a logarithmic factor from the FFT. Doubling n_i approximately doubles
//! the prover's work.

use crate::config::{
    Com, Domain, EvaluationsOnDomain, PackedChallenge, PackedVal, PcsData, PcsProof, ProofConfig,
    Val,
};
use crate::eval::VarValues;
use crate::lookup::{LookupValues, try_claims_accumulator};
use crate::system::{ProverKey, System, SystemWitness};
use crate::traits::{
    Algebra, EvaluationDomain, ExtensionOf, Field, LagrangeSelectors, OpenedValuesForRound, Packed,
    PackedExtension, Pcs, Transcript, TwoAdicField, flatten_to_base,
};

use bincode::config::{Configuration, Fixint, LittleEndian, standard};
use bincode::error::{DecodeError, EncodeError};
use bincode::serde::{decode_from_slice, encode_to_vec};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

/// Polynomial commitments included in the proof.
#[derive(Clone, Serialize, Deserialize)]
pub struct Commitments<Com> {
    /// Commitment to the stage 1 (main) execution traces.
    pub stage_1_trace: Com,
    /// Commitment to the stage 2 (lookup) execution traces.
    pub stage_2_trace: Com,
    /// Commitment to the quotient polynomial chunks.
    pub quotient_chunks: Com,
}

/// A STARK proof for a multi-circuit system.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Proof<SC: ProofConfig> {
    /// Activation bitmap over the system's canonical circuit set: circuit i
    /// is covered by this proof iff `active[i]`. Inactive circuits (empty
    /// execution traces) are not committed, opened, accumulated, or
    /// constraint-evaluated; the lookup accumulator polices dishonest
    /// deactivation (an omitted-but-needed circuit leaves an unmatched
    /// channel send, so the final accumulator cannot be zero). Bound into
    /// the Fiat-Shamir transcript before any commitment or challenge.
    /// Every other per-circuit sequence in this proof (accumulators,
    /// log_degrees, opened values) is indexed by ACTIVE position.
    pub active: Vec<bool>,
    pub commitments: Commitments<Com<SC>>,
    /// Per-active-circuit intermediate accumulator values for the lookup
    /// argument.
    pub intermediate_accumulators: Vec<SC::Challenge>,
    /// Log2 of the trace degree for each active circuit.
    pub log_degrees: Vec<u8>,
    /// PCS opening proof covering all rounds.
    pub opening_proof: PcsProof<SC>,
    pub quotient_opened_values: OpenedValuesForRound<SC::Challenge>,
    pub preprocessed_opened_values: Option<OpenedValuesForRound<SC::Challenge>>,
    pub stage_1_opened_values: OpenedValuesForRound<SC::Challenge>,
    pub stage_2_opened_values: OpenedValuesForRound<SC::Challenge>,
}

impl<SC: ProofConfig> Clone for Proof<SC> {
    fn clone(&self) -> Self {
        Self {
            active: self.active.clone(),
            commitments: self.commitments.clone(),
            intermediate_accumulators: self.intermediate_accumulators.clone(),
            log_degrees: self.log_degrees.clone(),
            opening_proof: self.opening_proof.clone(),
            quotient_opened_values: self.quotient_opened_values.clone(),
            preprocessed_opened_values: self.preprocessed_opened_values.clone(),
            stage_1_opened_values: self.stage_1_opened_values.clone(),
            stage_2_opened_values: self.stage_2_opened_values.clone(),
        }
    }
}

/// Round-one state of a proof: the stage-1 (main) trace commitment together
/// with everything the remaining rounds need. Produced by
/// [`System::prove_stage_1`], consumed by [`System::prove_after_challenges`].
///
/// Holding a `Stage1` across a batch's challenge barrier keeps the committed
/// LDE and Merkle tree resident; a caller that cannot afford that may drop it
/// after reading [`Stage1::active`], [`Stage1::log_degrees`] and
/// [`Stage1::stage_1_trace_commit`], and recommit the same witness later —
/// the commitment is deterministic.
pub struct Stage1<SC: ProofConfig> {
    /// Activation bitmap over the canonical circuit set (see [`Proof::active`]).
    pub active: Vec<bool>,
    /// Canonical index of each active circuit, in order.
    pub active_indices: Vec<usize>,
    /// Log2 trace height of each active circuit.
    pub log_degrees: Vec<usize>,
    /// The stage-1 commitment.
    pub stage_1_trace_commit: Com<SC>,
    /// PCS prover data behind the commitment (LDE and Merkle tree).
    pub stage_1_trace_data: PcsData<SC>,
    /// Lookup witness of each active circuit, in active order.
    pub lookups: Vec<LookupValues<Val<SC>>>,
}

/// Observes length-prefixed claims: the claim count, then each claim's
/// length and elements, so distinct claim structures (e.g. `[[a, b]]` vs
/// `[[a], [b]]`) yield distinct transcripts.
pub fn observe_claims<SC: ProofConfig>(challenger: &mut SC::Challenger, claims: &[&[Val<SC>]]) {
    challenger.observe_field(Val::<SC>::from_usize(claims.len()));
    for claim in claims {
        challenger.observe_field(Val::<SC>::from_usize(claim.len()));
        challenger.observe_field_slice(claim);
    }
}

/// Samples the lookup argument challenge β and the fingerprint challenge γ,
/// observing each back into the transcript.
pub fn sample_lookup_challenges<SC: ProofConfig>(
    challenger: &mut SC::Challenger,
) -> (SC::Challenge, SC::Challenge) {
    let lookup_argument_challenge: SC::Challenge = challenger.sample_challenge();
    challenger.observe_challenge(lookup_argument_challenge);
    let fingerprint_challenge: SC::Challenge = challenger.sample_challenge();
    challenger.observe_challenge(fingerprint_challenge);
    (lookup_argument_challenge, fingerprint_challenge)
}

impl<SC: ProofConfig> Proof<SC> {
    pub(crate) fn serde_config() -> Configuration<LittleEndian, Fixint> {
        standard().with_little_endian().with_fixed_int_encoding()
    }

    #[inline]
    pub fn to_bytes(&self) -> Result<Vec<u8>, EncodeError> {
        let mut proof = self.clone();
        SC::canonicalize_proof(&mut proof);
        encode_to_vec(&proof, Self::serde_config())
    }

    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let (proof, num_bytes) = decode_from_slice(bytes, Self::serde_config())?;
        if num_bytes != bytes.len() {
            return Err(DecodeError::Other("trailing bytes after proof"));
        }
        Ok(proof)
    }
}

impl<SC> System<SC>
where
    SC: ProofConfig,
    // Two-adicity is needed to slice the quotient into coefficient slices
    // and rebuild their committed LDE from those coefficients; every
    // FRI-based config is two-adic anyway.
    Val<SC>: TwoAdicField + Ord,
{
    /// Builds the stage-two traces and intermediate accumulators on the host,
    /// one circuit at a time. A materialized witness is dropped as soon as its
    /// stage-two trace is built; a shape-only witness streams directly from the
    /// committed stage-one trace (via [`ProofConfig::committed_trace`], falling
    /// back to [`Pcs::get_evaluations_on_domain`] only when `committed_trace`
    /// returns `None`) into [`crate::lookup::stage_2_trace_from_trace`] without
    /// ever materializing full [`LookupValues`] in memory. Circuits only
    /// interact through the accumulator chain, which is carried between
    /// iterations, so the traces equal those of a single pass over every
    /// circuit.
    #[allow(clippy::too_many_arguments)]
    fn host_lookup_traces(
        &self,
        key: &ProverKey<SC>,
        data: &PcsData<SC>,
        active_indices: &[usize],
        log_degrees: &[usize],
        lookups: Vec<LookupValues<Val<SC>>>,
        group_sizes: &[usize],
        lookup_argument_challenge: SC::Challenge,
        fingerprint_challenge: SC::Challenge,
        mut acc: SC::Challenge,
    ) -> (Vec<RowMajorMatrix<SC::Challenge>>, Vec<SC::Challenge>) {
        let pcs = self.config.pcs();
        let mut traces = Vec::with_capacity(lookups.len());
        let mut intermediate_accumulators = Vec::with_capacity(lookups.len());
        for (position, values) in lookups.into_iter().enumerate() {
            if values.is_materialized() {
                let values = [values];
                let group_size = &group_sizes[position..=position];
                let (circuit_traces, accumulators) = self
                    .config
                    .accelerated_lookup_traces(
                        &values,
                        group_size,
                        lookup_argument_challenge,
                        fingerprint_challenge,
                        acc,
                    )
                    .unwrap_or_else(|| {
                        LookupValues::stage_2_traces(
                            &values,
                            group_size,
                            lookup_argument_challenge,
                            &fingerprint_challenge,
                            acc,
                        )
                    });
                drop(values);
                acc = *accumulators.last().expect("one accumulator per circuit");
                traces.extend(circuit_traces);
                intermediate_accumulators.extend(accumulators);
            } else {
                drop(values);
                let circuit_index = active_indices[position];
                let circuit = &self.circuits[circuit_index];
                let degree = 1 << log_degrees[position];
                let (trace, delta) = if circuit.graph.lookups.is_empty() {
                    (
                        RowMajorMatrix::new(
                            vec![<SC::Challenge as Algebra<SC::Challenge>>::ZERO; degree],
                            1,
                        ),
                        <SC::Challenge as Algebra<SC::Challenge>>::ZERO,
                    )
                } else {
                    let _g = tracing::info_span!("stark/materialize_lookups", position).entered();
                    let main_trace = self
                        .config
                        .committed_trace(data, position, degree)
                        .unwrap_or_else(|| {
                            let trace_domain = pcs.natural_domain_for_degree(degree);
                            pcs.get_evaluations_on_domain(data, position, trace_domain)
                                .to_row_major_matrix()
                        });
                    // A system built without its fixed matrices reads each one
                    // back from the preprocessed commitment when it is needed.
                    let owned_prep;
                    let prep_ref = if circuit.graph.lookup_prefix_uses_preprocessed() {
                        if let Some(pp) = circuit.preprocessed.as_ref() {
                            Some(pp)
                        } else {
                            let (prep_data, prep_idx) = key
                                .preprocessed_data
                                .as_ref()
                                .zip(self.preprocessed_indices[circuit_index])
                                .expect("circuit lookup expressions require preprocessed trace");
                            owned_prep = self
                                .config
                                .committed_trace(prep_data, prep_idx, degree)
                                .unwrap_or_else(|| {
                                    let trace_domain = pcs.natural_domain_for_degree(degree);
                                    pcs.get_evaluations_on_domain(prep_data, prep_idx, trace_domain)
                                        .to_row_major_matrix()
                                });
                            Some(&owned_prep)
                        }
                    } else {
                        None
                    };
                    crate::lookup::stage_2_trace_from_trace(
                        circuit,
                        &main_trace,
                        prep_ref,
                        lookup_argument_challenge,
                        &fingerprint_challenge,
                    )
                };
                acc += delta;
                traces.push(trace);
                intermediate_accumulators.push(acc);
            }
        }
        (traces, intermediate_accumulators)
    }

    /// Generates a STARK proof for the system with a single claim.
    ///
    /// This is a convenience wrapper around [`Self::prove_multiple_claims`].
    pub fn prove(
        &self,
        key: &ProverKey<SC>,
        claim: &[Val<SC>],
        witness: SystemWitness<Val<SC>>,
    ) -> Proof<SC> {
        self.prove_multiple_claims(key, &[claim], witness)
    }

    /// Generates a STARK proof for the system with multiple claims.
    ///
    /// Each claim is a slice of field elements that is observed by the challenger
    /// before lookup challenges are sampled, binding the proof to the claimed values.
    ///
    /// Circuits whose stage-1 trace is empty are deactivated for this proof:
    /// they are neither committed, opened, accumulated, nor
    /// constraint-evaluated, and the activation bitmap is recorded in
    /// [`Proof::active`].
    ///
    /// # Panics
    /// Panics if every circuit's trace is empty (nothing to prove), or if the
    /// sampled lookup challenge makes a public claim denominator zero.
    #[tracing::instrument(level = "info", skip_all, name = "stark/prove")]
    pub fn prove_multiple_claims(
        &self,
        key: &ProverKey<SC>,
        claims: &[&[Val<SC>]],
        witness: SystemWitness<Val<SC>>,
    ) -> Proof<SC> {
        let stage_1 = self.prove_stage_1(witness);
        self.prove_committed(key, claims, stage_1)
    }

    /// Generates a STARK proof with a single claim by streaming stage-1 traces
    /// lazily and evaluating lookups on demand.
    ///
    /// Produces proofs byte-identical to [`Self::prove`].
    pub fn prove_streaming<I>(&self, key: &ProverKey<SC>, claim: &[Val<SC>], traces: I) -> Proof<SC>
    where
        I: IntoIterator,
        I::Item: Into<crate::witness::TraceSource<Val<SC>>>,
    {
        self.prove_multiple_claims_streaming(key, &[claim], traces)
    }

    /// Generates a STARK proof with multiple claims by streaming stage-1 traces
    /// lazily without materializing [`LookupValues`] upfront.
    ///
    /// Produces proofs byte-identical to [`Self::prove_multiple_claims`].
    #[tracing::instrument(level = "info", skip_all, name = "stark/prove_streaming")]
    pub fn prove_multiple_claims_streaming<I>(
        &self,
        key: &ProverKey<SC>,
        claims: &[&[Val<SC>]],
        traces: I,
    ) -> Proof<SC>
    where
        I: IntoIterator,
        I::Item: Into<crate::witness::TraceSource<Val<SC>>>,
    {
        let stage_1 = self.prove_stage_1_streaming(traces);
        self.prove_committed(key, claims, stage_1)
    }

    /// Continue an ordinary proof from independently prepared stage-one data.
    ///
    /// # Panics
    /// Panics if the sampled lookup challenge makes a public claim denominator zero.
    pub fn prove_committed(
        &self,
        key: &ProverKey<SC>,
        claims: &[&[Val<SC>]],
        stage_1: Stage1<SC>,
    ) -> Proof<SC> {
        let mut challenger = self
            .config
            .initialise_challenger_for_mode(crate::config::TranscriptMode::Single);

        // Bind the system shape into the transcript. The protocol parameters
        // are already bound via the challenger seed.
        self.observe_shape(&mut challenger);

        // The activation bitmap is bound into the transcript before any
        // commitment or challenge (see `Stage1::active`).
        for &is_active in &stage_1.active {
            challenger.observe_field(Val::<SC>::from_bool(is_active));
        }

        if let Some(commit) = &self.preprocessed_commit {
            challenger.observe_commitment(commit.clone());
        }
        challenger.observe_commitment(stage_1.stage_1_trace_commit.clone());

        // Observe the traces' heights. This binds the proof to specific domain
        // sizes; the verifier reads these from the (untrusted) proof, so they
        // must influence every subsequent challenge.
        for log_degree in &stage_1.log_degrees {
            challenger.observe_field(Val::<SC>::from_usize(*log_degree));
        }

        // Observe the claims, length-prefixed so that distinct claim
        // structures (e.g. [[a, b]] vs [[a], [b]]) yield distinct transcripts.
        // This has to be done before generating the lookup argument challenge,
        // otherwise the lookup argument can be attacked.
        observe_claims::<SC>(&mut challenger, claims);

        let (lookup_argument_challenge, fingerprint_challenge) =
            sample_lookup_challenges::<SC>(&mut challenger);
        let acc = try_claims_accumulator(lookup_argument_challenge, &fingerprint_challenge, claims)
            .expect("lookup challenge makes a claim denominator zero");

        self.prove_after_challenges(
            key,
            stage_1,
            challenger,
            lookup_argument_challenge,
            fingerprint_challenge,
            acc,
        )
    }

    /// Round one of a proof: commits the stage-1 (main) traces and retains
    /// what the remaining rounds need. No transcript is touched here — the
    /// caller decides which transcript the commitment enters (a single
    /// proof's own, or a batch's shared one — see [`crate::batch`]).
    ///
    /// # Panics
    /// Panics if every circuit's trace is empty (nothing to prove).
    #[tracing::instrument(level = "info", skip_all, name = "stark/stage1_commit")]
    pub fn prove_stage_1(
        &self,
        witness: impl Into<crate::witness::PreparedWitness<Val<SC>>>,
    ) -> Stage1<SC> {
        let witness = witness.into();
        let pcs = self.config.pcs();

        // Sparse activation: a circuit whose stage-1 trace is empty is
        // INACTIVE for this proof — nothing of it is committed, opened,
        // accumulated, or constraint-evaluated. Soundness of the omission
        // rests on the lookup accumulator: an inactive circuit has no rows,
        // hence no sends or receives, so omitting it leaves the global
        // balance unchanged — while dishonestly deactivating a circuit the
        // execution needs leaves an unmatched channel send and the final
        // accumulator cannot be zero.
        let active: Vec<bool> = witness.traces.iter().map(|t| t.height() > 0).collect();
        // Canonical index of each active circuit, in order; matrix position
        // within every per-proof commitment == position in this list.
        let active_indices: Vec<usize> = active
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| a.then_some(i))
            .collect();
        assert!(
            !active_indices.is_empty(),
            "cannot prove with every circuit deactivated (all traces empty)"
        );

        // Only active circuits enter the accumulator chain; the chain (and
        // `intermediate_accumulators`) is indexed by active position.
        let lookups: Vec<_> = witness
            .lookups
            .into_iter()
            .zip(&active)
            .filter_map(|(l, &is_active)| is_active.then_some(l))
            .collect();

        let is_zk = self.config.is_zk();
        let min_log_degree = self.config.min_log_degree();
        let max_log_degree = self.config.max_log_degree();

        // Cost: "Stage 1 commit" — coset LDE (FFT) of each trace from n_i to
        // n_i·B rows (an iDFT plus B coset DFTs per column), then Merkle
        // tree. FFT work: Σ w_i · (B+1) · n_i · log₂(n_i).
        let mut log_degrees = Vec::with_capacity(active_indices.len());
        let mut evaluations = witness
            .traces
            .into_iter()
            .enumerate()
            .zip(&active)
            .filter_map(|((ci, trace), &is_active)| is_active.then_some((ci, trace)))
            .zip(&lookups)
            .map(|((ci, trace), lookup)| {
                let degree = trace.height();
                let log_degree = log2_strict_usize(degree);
                if is_zk {
                    assert!(
                        log_degree >= min_log_degree,
                        "circuit {ci}: trace height {degree} (2^{log_degree}) is below v6-zk minimum height {} (2^{min_log_degree})",
                        1usize << min_log_degree,
                    );
                    assert!(
                        log_degree <= max_log_degree,
                        "circuit {ci}: trace log_degree {log_degree} exceeds v6-zk max_log_degree {max_log_degree}",
                    );
                }
                let trace_domain = pcs.natural_domain_for_degree(degree);
                log_degrees.push(log_degree);
                let retain_1x = !lookup.is_materialized() && self.circuits[ci].num_lookups > 0;
                (trace_domain, trace, retain_1x)
            });
        let (stage_1_trace_commit, stage_1_trace_data) = self
            .config
            .commit_main_streaming(&mut evaluations)
            .expect("cannot prove with every circuit deactivated (all traces empty)");

        Stage1 {
            active,
            active_indices,
            log_degrees,
            stage_1_trace_commit,
            stage_1_trace_data,
            lookups,
        }
    }

    /// Streaming round one of a proof: pulls each circuit's stage-1 trace
    /// lazily from `traces` without holding all traces or [`LookupValues`] in
    /// memory simultaneously.
    ///
    /// Lookup expressions are evaluated on demand in bounded row chunks during
    /// stage-2 trace construction. Produces proofs byte-identical to
    /// [`Self::prove_stage_1`] with [`SystemWitness::from_stage_1`].
    ///
    /// # Panics
    /// Panics if `traces` does not yield one trace per circuit, if an active
    /// circuit's height disagrees with its preprocessed trace height, or if
    /// every circuit's trace is empty.
    #[tracing::instrument(level = "info", skip_all, name = "stark/stage1_commit_streaming")]
    pub fn prove_stage_1_streaming<I>(&self, traces: I) -> Stage1<SC>
    where
        I: IntoIterator,
        I::Item: Into<crate::witness::TraceSource<Val<SC>>>,
    {
        let pcs = self.config.pcs();
        let is_zk = self.config.is_zk();
        let min_log_degree = self.config.min_log_degree();
        let max_log_degree = self.config.max_log_degree();
        let mut active = Vec::with_capacity(self.circuits.len());
        let mut active_indices = Vec::new();
        let mut log_degrees = Vec::new();
        let mut lookups = Vec::new();

        let mut evaluations = traces.into_iter().enumerate().filter_map(|(ci, trace)| {
            assert!(
                ci < self.circuits.len(),
                "expected one trace per circuit (got more than {})",
                self.circuits.len()
            );
            let trace: crate::witness::TraceSource<Val<SC>> = trace.into();
            let height = trace.height();
            let circuit = &self.circuits[ci];
            if height == 0 {
                active.push(false);
                return None;
            }
            let log_degree = log2_strict_usize(height);
            if is_zk {
                assert!(
                    log_degree >= min_log_degree,
                    "circuit {ci}: trace height {height} (2^{log_degree}) is below v6-zk minimum height {} (2^{min_log_degree})",
                    1usize << min_log_degree,
                );
                assert!(
                    log_degree <= max_log_degree,
                    "circuit {ci}: trace log_degree {log_degree} exceeds v6-zk max_log_degree {max_log_degree}",
                );
            }
            if circuit.preprocessed_height != 0 {
                assert_eq!(
                    height, circuit.preprocessed_height,
                    "circuit {ci}: main trace height must equal preprocessed trace height"
                );
            }
            active.push(true);
            active_indices.push(ci);
            log_degrees.push(log_degree);
            let slot_widths: Vec<usize> = circuit
                .graph
                .lookups
                .iter()
                .map(|lookup| lookup.args.len())
                .collect();
            lookups.push(LookupValues::shape_only(height, &slot_widths));
            let trace_domain = pcs.natural_domain_for_degree(height);
            let retain_1x = circuit.num_lookups > 0;
            Some((trace_domain, trace, retain_1x))
        });

        let committed = self.config.commit_main_streaming(&mut evaluations);
        for _ in evaluations {}
        assert_eq!(
            active.len(),
            self.circuits.len(),
            "expected one trace per circuit"
        );
        let (stage_1_trace_commit, stage_1_trace_data) =
            committed.expect("cannot prove with every circuit deactivated (all traces empty)");

        Stage1 {
            active,
            active_indices,
            log_degrees,
            stage_1_trace_commit,
            stage_1_trace_data,
            lookups,
        }
    }

    /// Rounds two onward: given a committed [`Stage1`], a transcript that has
    /// just sampled the lookup challenges, and the initial accumulator the
    /// verifier will derive, builds the stage-2 traces, the quotient and the
    /// FRI opening. Shared by the single-proof protocol and the batch
    /// protocol, which differ only in how the transcript reached this point.
    pub fn prove_after_challenges(
        &self,
        key: &ProverKey<SC>,
        stage_1: Stage1<SC>,
        mut challenger: SC::Challenger,
        lookup_argument_challenge: SC::Challenge,
        fingerprint_challenge: SC::Challenge,
        mut acc: SC::Challenge,
    ) -> Proof<SC> {
        let pcs = self.config.pcs();
        let is_zk = self.config.is_zk();
        let Stage1 {
            active,
            active_indices,
            log_degrees,
            stage_1_trace_commit,
            stage_1_trace_data,
            lookups: active_lookups,
        } = stage_1;
        let num_active = active_indices.len();
        let use_zk_acc_mask = is_zk && num_active > 1;
        // Canonical index -> active position (None = inactive).
        let mut active_pos: Vec<Option<usize>> = vec![None; active.len()];
        for (pos, &ci) in active_indices.iter().enumerate() {
            active_pos[ci] = Some(pos);
        }

        // Cost: "Lookup trace construction" — fingerprint (Horner), batch
        // inversion, and accumulator update. Total: Σ n_i·L_i extension field ops.
        let _g = tracing::info_span!("stark/lookup_construction").entered();
        let group_sizes: Vec<usize> = active_indices
            .iter()
            .map(|&ci| self.circuits[ci].lookup_group_size)
            .collect();
        let lookup_inputs: Vec<_> = active_indices
            .iter()
            .enumerate()
            .zip(&active_lookups)
            .map(
                |((position, &circuit_index), lookup_values)| crate::config::LookupCommitInput {
                    circuit: &self.circuits[circuit_index],
                    lookup_values,
                    preprocessed: key
                        .preprocessed_data
                        .as_ref()
                        .zip(self.preprocessed_indices[circuit_index]),
                    stage_1: (&stage_1_trace_data, position),
                },
            )
            .collect();
        let accelerated_lookup_commit = self.config.accelerated_lookup_commit(
            &lookup_inputs,
            lookup_argument_challenge,
            fingerprint_challenge,
            acc,
        );
        drop(lookup_inputs);
        let (stage_2_trace_commit, stage_2_trace_data, intermediate_accumulators) =
            if let Some(result) = accelerated_lookup_commit {
                drop(active_lookups);
                drop(_g);
                result
            } else {
                let (stage_2_traces, intermediate_accumulators) = self.host_lookup_traces(
                    key,
                    &stage_1_trace_data,
                    &active_indices,
                    &log_degrees,
                    active_lookups,
                    &group_sizes,
                    lookup_argument_challenge,
                    fingerprint_challenge,
                    acc,
                );
                drop(_g);

                // Cost: "Stage 2 commit" — LDE + Merkle for flattened extension traces.
                // FFT work: Σ w2_i · D · (B+1) · n_i · log₂(n_i).
                let _g = tracing::info_span!("stark/stage2_commit").entered();
                let mut stage_2_traces = stage_2_traces;
                let mut intermediate_accumulators = intermediate_accumulators;
                let evaluations: Vec<_> = if use_zk_acc_mask {
                    let ext_d = <SC::Challenge as ExtensionOf<Val<SC>>>::D;
                    // Move boundary injection to row 0 (X = 1) so every active
                    // circuit shares the common point X = 1 with the boundary
                    // mask matrix `U`.
                    let mut prev_true = acc;
                    for (trace, &true_acc) in
                        stage_2_traces.iter_mut().zip(&intermediate_accumulators)
                    {
                        let delta_true = true_acc - prev_true;
                        crate::lookup::apply_first_row_delta_shift(trace, delta_true);
                        prev_true = true_acc;
                    }
                    // Blind interior accumulators `0..num_active - 1` with fresh
                    // CSPRNG extension masks `mu_p`, leaving the final accumulator
                    // exact (`0` for single proofs, shard residual for batches).
                    let mask_cols = (num_active - 1) * ext_d;
                    let mu_base = self.config.sample_zk_base_randomness(mask_cols);
                    for p in 0..(num_active - 1) {
                        let mu =
                            SC::Challenge::from_basis_coefficients_fn(|c| mu_base[p * ext_d + c]);
                        intermediate_accumulators[p] += mu;
                    }
                    let mut evals: Vec<_> = stage_2_traces
                        .into_iter()
                        .map(|trace| {
                            let degree = trace.height();
                            let trace_domain = pcs.natural_domain_for_degree(degree);
                            (trace_domain, flatten_to_base(trace))
                        })
                        .collect();
                    let min_height = 1usize << log_degrees.iter().copied().min().unwrap();
                    let mut mask_values = Vec::with_capacity(min_height * mask_cols);
                    mask_values.extend_from_slice(&mu_base);
                    if min_height > 1 {
                        mask_values.extend(
                            self.config
                                .sample_zk_base_randomness((min_height - 1) * mask_cols),
                        );
                    }
                    let mask_mat = RowMajorMatrix::new(mask_values, mask_cols);
                    let min_domain = pcs.natural_domain_for_degree(min_height);
                    evals.push((min_domain, mask_mat));
                    evals
                } else {
                    stage_2_traces
                        .into_iter()
                        .map(|trace| {
                            let degree = trace.height();
                            let trace_domain = pcs.natural_domain_for_degree(degree);
                            (trace_domain, flatten_to_base(trace))
                        })
                        .collect()
                };
                let (commitment, data) = pcs.commit(evaluations);
                drop(_g);
                (commitment, data, intermediate_accumulators)
            };
        challenger.observe_commitment(stage_2_trace_commit.clone());

        // Observe the intermediate accumulators. They enter the constraints as
        // public values, so later challenges (α, ζ) must depend on them
        // directly rather than only through the quotient commitment.
        for acc in &intermediate_accumulators {
            challenger.observe_challenge(*acc);
        }

        // Constraint challenge.
        let constraint_challenge: SC::Challenge = challenger.sample_challenge();

        // Cost: "Quotient computation and commit" — constraint evaluation on
        // the quotient domain (Σ n_i·q_i·eval_cost(k_i)), the forward DFT of
        // the flattened quotient (Σ D·q_i·n_i·log₂(q_i·n_i)), then LDE +
        // Merkle of the sub-polynomials (Σ q_i·D·B·n_i·log₂(B·n_i)).
        let _g = tracing::info_span!("stark/quotient").entered();
        debug_assert_eq!(intermediate_accumulators.len(), active_indices.len());
        debug_assert_eq!(log_degrees.len(), active_indices.len());
        let mut quotient_inputs = Vec::with_capacity(active_indices.len());
        let zk_shift = usize::from(is_zk);
        for (pos, ((&ci, log_degree), next_acc)) in active_indices
            .iter()
            .zip(log_degrees.iter())
            .zip(intermediate_accumulators.iter())
            .enumerate()
        {
            let circuit = &self.circuits[ci];
            let quotient_degree = circuit.quotient_degree_for(is_zk);
            let log_quotient_degree = log2_strict_usize(quotient_degree);
            let trace_domain = pcs.natural_domain_for_degree(1 << log_degree);
            let quotient_domain = trace_domain
                .create_disjoint_domain(1 << (log_degree + zk_shift + log_quotient_degree));
            // The four lookup publics (β, γ, acc, next_acc) as flat base
            // coordinates, in the layout the synthesized constraints read.
            let mut lookup_publics: Vec<Val<SC>> = Vec::new();
            for ef in [
                lookup_argument_challenge,
                fingerprint_challenge,
                acc,
                *next_acc,
            ] {
                lookup_publics.extend_from_slice(ef.as_basis_coefficients_slice());
            }

            quotient_inputs.push(crate::config::QuotientCommitInput {
                circuit,
                lookup_publics,
                trace_domain,
                quotient_domain,
                preprocessed: key
                    .preprocessed_data
                    .as_ref()
                    .zip(self.preprocessed_indices[ci]),
                stage_1: (&stage_1_trace_data, pos),
                stage_2: (&stage_2_trace_data, pos),
                constraint_count: circuit.constraint_count(),
                zk_boundary_mask: use_zk_acc_mask.then_some(crate::config::ZkBoundaryMask {
                    data: &stage_2_trace_data,
                    matrix_index: num_active,
                    active_pos: pos,
                    num_active,
                }),
            });
            acc = *next_acc;
        }

        let (quotient_commit, quotient_data) = if let Some(committed) = self
            .config
            .accelerated_quotient_commit(&quotient_inputs, constraint_challenge)
        {
            committed
        } else {
            let quotient_ldes: Vec<_> = quotient_inputs
                .into_iter()
                .map(|input| {
                    let circuit = input.circuit;
                    let quotient_degree = circuit.quotient_degree_for(is_zk);

                    // compute the quotient values which are elements of the extension field and flatten it to the base field
                    let quotient_values = if input.zk_boundary_mask.is_none() {
                        self.config.accelerated_quotient_values(
                            circuit,
                            &input.lookup_publics,
                            input.trace_domain,
                            input.quotient_domain,
                            input.preprocessed,
                            input.stage_1,
                            input.stage_2,
                            constraint_challenge,
                            input.constraint_count,
                        )
                    } else {
                        None
                    }
                    .unwrap_or_else(|| {
                        let preprocessed_trace_on_quotient_domain =
                            input.preprocessed.map(|(data, idx)| {
                                pcs.get_evaluations_on_domain(data, idx, input.quotient_domain)
                            });
                        let stage_1_trace_on_quotient_domain = pcs.get_evaluations_on_domain(
                            input.stage_1.0,
                            input.stage_1.1,
                            input.quotient_domain,
                        );
                        let stage_2_trace_on_quotient_domain = pcs.get_evaluations_on_domain(
                            input.stage_2.0,
                            input.stage_2.1,
                            input.quotient_domain,
                        );
                        let mask_evals_owned = input.zk_boundary_mask.map(|mask| {
                            (
                                pcs.get_evaluations_on_domain(
                                    mask.data,
                                    mask.matrix_index,
                                    input.quotient_domain,
                                ),
                                mask.active_pos,
                                mask.num_active,
                            )
                        });
                        let mask_ref = mask_evals_owned
                            .as_ref()
                            .map(|(evals, active_pos, n_active)| (evals, *active_pos, *n_active));
                        quotient_values::<SC>(
                            circuit,
                            &input.lookup_publics,
                            input.trace_domain,
                            input.quotient_domain,
                            &preprocessed_trace_on_quotient_domain,
                            &stage_1_trace_on_quotient_domain,
                            &stage_2_trace_on_quotient_domain,
                            mask_ref,
                            constraint_challenge,
                            input.constraint_count,
                        )
                    });
                    let quotient_flat =
                        flatten_to_base::<Val<SC>, _>(RowMajorMatrix::new_col(quotient_values));
                    (input.quotient_domain, quotient_flat, quotient_degree)
                })
                .collect();
            if is_zk {
                self.config.commit_quotient_zk(quotient_ldes)
            } else {
                pcs.commit_quotient(quotient_ldes)
            }
        };
        challenger.observe_commitment(quotient_commit.clone());
        drop(_g);

        let commitments = Commitments {
            stage_1_trace: stage_1_trace_commit,
            stage_2_trace: stage_2_trace_commit,
            quotient_chunks: quotient_commit,
        };

        // Cost: "FRI opening" — barycentric interpolation (Σ n_i·B·W_i),
        // FRI folding (≈ H), and FRI queries (Q·R·log₂ H hash ops).
        let _g = tracing::info_span!("stark/fri_open").entered();
        let zeta: SC::Challenge = challenger.sample_challenge();
        let mut round0_openings = vec![];
        let mut round1_openings = vec![];
        let mut round2_openings = vec![];
        let mut round3_openings = vec![];
        for (&ci, &log_degree) in active_indices.iter().zip(&log_degrees) {
            let trace_domain = pcs.natural_domain_for_degree(1 << log_degree);
            let zeta_next = trace_domain.next_point(zeta);
            round1_openings.push(if self.opens_next_row(ci, crate::expr::Source::Main) {
                vec![zeta, zeta_next]
            } else {
                vec![zeta]
            });
            round2_openings.push(vec![zeta, zeta_next]);
            // One wide matrix per circuit holds all its quotient slices.
            round3_openings.push(vec![zeta]);
        }
        // In multi-circuit `v6-zk`, open the committed accumulator boundary mask
        // matrix `U` (matrix `#num_active` in `stage_2_trace`) at `zeta`.
        if use_zk_acc_mask {
            round2_openings.push(vec![zeta]);
        }
        // In `v6-zk`, open the `W_R`-column randomizer polynomial `R(X)`
        // (matrix `#num_active` in `quotient_chunks`) at `zeta`.
        if is_zk {
            round3_openings.push(vec![zeta]);
        }
        // The preprocessed commitment is built once over ALL preprocessed
        // traces at system construction, so its round must carry one entry
        // per preprocessed matrix regardless of activation. An inactive
        // circuit's preprocessed matrix is normally opened at ζ alone: no constraint
        // reads the value, but the PCS pins every matrix's width to the
        // claimed values at its first opening point (the leaf hash flattens
        // same-height rows into one stream, so a matrix opened nowhere would
        // leave its row boundary unauthenticated) and rejects a matrix with
        // no opening points. A PCS with independent column commitments can
        // explicitly opt out of these otherwise-unused openings.
        for (ci, (prep_index, &pos)) in self
            .preprocessed_indices
            .iter()
            .zip(&active_pos)
            .enumerate()
        {
            if prep_index.is_some() {
                match pos {
                    Some(pos) => {
                        let trace_domain = pcs.natural_domain_for_degree(1 << log_degrees[pos]);
                        let zeta_next = trace_domain.next_point(zeta);
                        round0_openings.push(
                            if self.opens_next_row(ci, crate::expr::Source::Preprocessed) {
                                vec![zeta, zeta_next]
                            } else {
                                vec![zeta]
                            },
                        );
                    }
                    None => {
                        round0_openings.push(if self.config.omit_inactive_preprocessed_openings() {
                            vec![]
                        } else {
                            vec![zeta]
                        })
                    }
                }
            }
        }
        let mut rounds = vec![
            (&stage_1_trace_data, round1_openings),
            (&stage_2_trace_data, round2_openings),
            (&quotient_data, round3_openings),
        ];
        if self.preprocessed_commit.is_some() {
            rounds.push((key.preprocessed_data.as_ref().unwrap(), round0_openings));
        }
        let (opened_values, opening_proof) = pcs.open(rounds, &mut challenger);
        drop(_g);
        let mut opened_values_iter = opened_values.into_iter();
        let stage_1_opened_values = opened_values_iter.next().unwrap();
        let stage_2_opened_values = opened_values_iter.next().unwrap();
        let quotient_opened_values = opened_values_iter.next().unwrap();
        let preprocessed_opened_values = opened_values_iter.next();
        debug_assert!(opened_values_iter.next().is_none());
        let log_degrees = log_degrees
            .into_iter()
            .map(|n| n.try_into().unwrap())
            .collect();
        Proof {
            active,
            commitments,
            intermediate_accumulators,
            log_degrees,
            opening_proof,
            quotient_opened_values,
            preprocessed_opened_values,
            stage_1_opened_values,
            stage_2_opened_values,
        }
    }
}

/// Evaluates the folded constraints on the quotient domain and divides by
/// the vanishing polynomial, producing the quotient values.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quotient_values<SC>(
    circuit: &crate::system::Circuit<Val<SC>>,
    lookup_publics: &[Val<SC>],
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    preprocessed_on_quotient_domain: &Option<EvaluationsOnDomain<'_, SC>>,
    stage_1_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    stage_2_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    mask_on_quotient_domain: Option<(&EvaluationsOnDomain<'_, SC>, usize, usize)>,
    alpha: SC::Challenge,
    constraint_count: usize,
) -> Vec<SC::Challenge>
where
    SC: ProofConfig,
    Val<SC>: TwoAdicField + Ord,
{
    let quotient_size = quotient_domain.size();
    let main_width = circuit.main_width;
    let stage_2_width = circuit.stage_2_width;
    let preprocessed_width = circuit.preprocessed_width;
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);
    // The logUp wrap constraint consumes the boundary selector ADDITIVELY, so
    // its normalization matters:
    // - Non-ZK / 1-circuit ZK: `is_last_row` at `g^{-1}` has value `n·g`, so
    //   `inj_norm = 1/(n·g)`.
    // - Multi-circuit ZK (`mask_on_quotient_domain.is_some()`): `is_first_row`
    //   at `1` has value `n`, so `inj_norm = 1/n`.
    let n_val = Val::<SC>::from_usize(trace_domain.size());
    let g = Val::<SC>::two_adic_generator(log2_strict_usize(trace_domain.size()));
    let inj_norm = if mask_on_quotient_domain.is_some() {
        n_val.inverse()
    } else {
        (n_val * g).inverse()
    };

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    let width = PackedVal::<SC>::WIDTH;
    let padded_len = quotient_size.next_multiple_of(width);
    for _ in quotient_size..padded_len {
        sels.is_first_row.push(Val::<SC>::ZERO);
        sels.is_last_row.push(Val::<SC>::ZERO);
        sels.is_transition.push(Val::<SC>::ZERO);
        sels.inv_vanishing.push(Val::<SC>::ZERO);
    }

    // α powers in reverse (constraint i of k weighted by α^{k-1-i}),
    // decomposed per basis coordinate for the fused base-field fold.
    let ext_d = <SC::Challenge as ExtensionOf<Val<SC>>>::D;
    let mut alpha_powers: Vec<SC::Challenge> = alpha.powers().take(constraint_count).collect();
    alpha_powers.reverse();
    let alpha_table = if ext_d == 2 {
        let pairs: Vec<[Val<SC>; 2]> = alpha_powers
            .iter()
            .map(|x| {
                let coords = x.as_basis_coefficients_slice();
                [coords[0], coords[1]]
            })
            .collect();
        AlphaTable::Degree2(pairs)
    } else if ext_d == 3 {
        let triples: Vec<[Val<SC>; 3]> = alpha_powers
            .iter()
            .map(|x| {
                let coords = x.as_basis_coefficients_slice();
                [coords[0], coords[1], coords[2]]
            })
            .collect();
        AlphaTable::Degree3(triples)
    } else {
        let decomposed: Vec<Vec<Val<SC>>> = (0..ext_d)
            .map(|i| {
                alpha_powers
                    .iter()
                    .map(|x| x.as_basis_coefficients_slice()[i])
                    .collect()
            })
            .collect();
        AlphaTable::Generic(decomposed)
    };
    drop(alpha_powers);

    // Public coordinates broadcast to packed base values.
    let publics_packed: Vec<PackedVal<SC>> = lookup_publics
        .iter()
        .map(|&c| PackedVal::<SC>::from(c))
        .collect();

    // Unmasked constant `delta_scaled` (non-ZK / 1-circuit ZK) or unscaled
    // `delta_tilde` (multi-circuit ZK with `U` mask subtracted per pack).
    let delta_base: Vec<PackedVal<SC>> = (0..ext_d)
        .map(|k| {
            let diff = lookup_publics[3 * ext_d + k] - lookup_publics[2 * ext_d + k];
            if mask_on_quotient_domain.is_some() {
                PackedVal::<SC>::from(diff)
            } else {
                PackedVal::<SC>::from(diff * inj_norm)
            }
        })
        .collect();
    let inj_norm_packed = PackedVal::<SC>::from(inj_norm);
    let mask_width = mask_on_quotient_domain.map_or(0, |(_, _, n_active)| (n_active - 1) * ext_d);

    let ext_params = crate::system::extension_params::<SC>();

    // Determine once per circuit which row slices are actually referenced,
    // skipping `next` (and any unused `current`) row gathers inside the hot loop.
    let has_prep = preprocessed_on_quotient_domain.is_some();
    let prep_cur_len = if has_prep
        && circuit.uses_row(
            crate::expr::Source::Preprocessed,
            crate::expr::RowOffset::Current,
        ) {
        preprocessed_width
    } else {
        0
    };
    let prep_next_len = if has_prep
        && circuit.uses_row(
            crate::expr::Source::Preprocessed,
            crate::expr::RowOffset::Next,
        ) {
        preprocessed_width
    } else {
        0
    };
    let main_cur_len =
        if circuit.uses_row(crate::expr::Source::Main, crate::expr::RowOffset::Current) {
            main_width
        } else {
            0
        };
    let main_next_len = if circuit.uses_row(crate::expr::Source::Main, crate::expr::RowOffset::Next)
    {
        main_width
    } else {
        0
    };
    let s2_next_len = if circuit.uses_row(crate::expr::Source::Stage2, crate::expr::RowOffset::Next)
    {
        stage_2_width
    } else {
        // `logup_constraint_values` only reads the wrap slot `stage_2_next[0..ext_d]`.
        ext_d.min(stage_2_width)
    };
    let node_count = circuit.graph.nodes.len();
    let accs_len = if ext_d == 2 || ext_d == 3 { 0 } else { ext_d };

    const PACKS_PER_CHUNK: usize = 16;
    let chunk_rows = PACKS_PER_CHUNK * width;
    let mut out = vec![<SC::Challenge as Algebra<SC::Challenge>>::ZERO; quotient_size];

    out.par_chunks_mut(chunk_rows).enumerate().for_each_init(
        || QuotientScratch {
            prep_cur: vec![PackedVal::<SC>::ZERO; prep_cur_len],
            prep_next: vec![PackedVal::<SC>::ZERO; prep_next_len],
            main_cur: vec![PackedVal::<SC>::ZERO; main_cur_len],
            main_next: vec![PackedVal::<SC>::ZERO; main_next_len],
            s2_cur: vec![PackedVal::<SC>::ZERO; stage_2_width],
            s2_next: vec![PackedVal::<SC>::ZERO; s2_next_len],
            mask_cur: vec![PackedVal::<SC>::ZERO; mask_width],
            delta_eff: vec![PackedVal::<SC>::ZERO; if mask_width > 0 { ext_d } else { 0 }],
            buf: Vec::with_capacity(node_count),
            accs: vec![PackedVal::<SC>::ZERO; accs_len],
        },
        |scratch, (chunk_idx, chunk_out)| {
            let chunk_start = chunk_idx * chunk_rows;
            for (pack_idx, pack_out) in chunk_out.chunks_mut(width).enumerate() {
                let i_start = chunk_start + pack_idx * width;
                eval_quotient_pack::<SC>(
                    circuit,
                    &sels,
                    preprocessed_on_quotient_domain,
                    stage_1_on_quotient_domain,
                    stage_2_on_quotient_domain,
                    mask_on_quotient_domain,
                    &publics_packed,
                    &delta_base,
                    inj_norm_packed,
                    &alpha_table,
                    next_step,
                    i_start,
                    ext_params.modulus,
                    ext_params.degree,
                    constraint_count,
                    scratch,
                    pack_out,
                );
            }
        },
    );
    out
}

enum AlphaTable<F> {
    Degree2(Vec<[F; 2]>),
    Degree3(Vec<[F; 3]>),
    Generic(Vec<Vec<F>>),
}

struct QuotientScratch<P> {
    prep_cur: Vec<P>,
    prep_next: Vec<P>,
    main_cur: Vec<P>,
    main_next: Vec<P>,
    s2_cur: Vec<P>,
    s2_next: Vec<P>,
    mask_cur: Vec<P>,
    delta_eff: Vec<P>,
    buf: Vec<P>,
    accs: Vec<P>,
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn eval_quotient_pack<SC>(
    circuit: &crate::system::Circuit<Val<SC>>,
    sels: &LagrangeSelectors<Vec<Val<SC>>>,
    preprocessed_on_quotient_domain: &Option<EvaluationsOnDomain<'_, SC>>,
    stage_1_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    stage_2_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    mask_on_quotient_domain: Option<(&EvaluationsOnDomain<'_, SC>, usize, usize)>,
    publics_packed: &[PackedVal<SC>],
    delta_base: &[PackedVal<SC>],
    inj_norm_packed: PackedVal<SC>,
    alpha_table: &AlphaTable<Val<SC>>,
    next_step: usize,
    i_start: usize,
    ext_modulus: crate::traits::ExtensionModulus<Val<SC>>,
    ext_degree: usize,
    constraint_count: usize,
    scratch: &mut QuotientScratch<PackedVal<SC>>,
    pack_out: &mut [SC::Challenge],
) where
    SC: ProofConfig,
    Val<SC>: TwoAdicField + Ord,
{
    let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;
    let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
    let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
    let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
    let inv_vanishing = *PackedVal::<SC>::from_slice(&sels.inv_vanishing[i_range]);

    if let Some(prep) = preprocessed_on_quotient_domain.as_ref() {
        if !scratch.prep_cur.is_empty() {
            PackedVal::<SC>::fill_packed_row(prep, i_start, &mut scratch.prep_cur);
        }
        if !scratch.prep_next.is_empty() {
            PackedVal::<SC>::fill_packed_row(prep, i_start + next_step, &mut scratch.prep_next);
        }
    }
    if !scratch.main_cur.is_empty() {
        PackedVal::<SC>::fill_packed_row(
            stage_1_on_quotient_domain,
            i_start,
            &mut scratch.main_cur,
        );
    }
    if !scratch.main_next.is_empty() {
        PackedVal::<SC>::fill_packed_row(
            stage_1_on_quotient_domain,
            i_start + next_step,
            &mut scratch.main_next,
        );
    }
    PackedVal::<SC>::fill_packed_row(stage_2_on_quotient_domain, i_start, &mut scratch.s2_cur);
    PackedVal::<SC>::fill_packed_row(
        stage_2_on_quotient_domain,
        i_start + next_step,
        &mut scratch.s2_next,
    );

    let (logup_selector, delta_scaled): (PackedVal<SC>, &[PackedVal<SC>]) =
        if let Some((mask_evals, active_pos, num_active)) = mask_on_quotient_domain {
            PackedVal::<SC>::fill_packed_row(mask_evals, i_start, &mut scratch.mask_cur);
            for (k, (&d_tilde, dst)) in delta_base
                .iter()
                .zip(scratch.delta_eff.iter_mut())
                .enumerate()
            {
                let mut val = d_tilde;
                if active_pos + 1 < num_active {
                    val -= scratch.mask_cur[active_pos * ext_degree + k];
                }
                if active_pos > 0 {
                    val += scratch.mask_cur[(active_pos - 1) * ext_degree + k];
                }
                *dst = val * inj_norm_packed;
            }
            (is_first_row, &scratch.delta_eff)
        } else {
            (is_last_row, delta_base)
        };

    let view = VarValues {
        preprocessed: [&scratch.prep_cur, &scratch.prep_next],
        main: [&scratch.main_cur, &scratch.main_next],
        stage2: [&scratch.s2_cur, &scratch.s2_next],
        publics: publics_packed,
        is_first_row,
        is_last_row,
        is_transition,
    };
    circuit.graph.sweep(&view, &mut scratch.buf);

    // Fuse user-constraint and logUp-constraint α folding directly without
    // allocating a per-pack `constraint_values` vector.
    let accumulator = match alpha_table {
        AlphaTable::Degree2(alpha_pairs) => {
            let (mut acc0, mut acc1) = circuit.graph.fold_constraints_d2(&scratch.buf, alpha_pairs);
            let mut idx = circuit.graph.zeros.len();
            crate::lookup::for_each_logup_constraint_value(
                &circuit.graph.lookups,
                &scratch.buf,
                &scratch.s2_cur,
                &scratch.s2_next,
                publics_packed,
                delta_scaled,
                logup_selector,
                ext_modulus,
                ext_degree,
                circuit.lookup_group_size,
                |val| {
                    let pair = alpha_pairs[idx];
                    acc0 += val * pair[0];
                    acc1 += val * pair[1];
                    idx += 1;
                },
            );
            debug_assert_eq!(idx, constraint_count);
            PackedChallenge::<SC>::from_basis_coefficients_fn(|coeff_idx| {
                if coeff_idx == 0 { acc0 } else { acc1 }
            })
        }
        AlphaTable::Degree3(alpha_triples) => {
            let (mut acc0, mut acc1, mut acc2) = circuit
                .graph
                .fold_constraints_d3(&scratch.buf, alpha_triples);
            let mut idx = circuit.graph.zeros.len();
            crate::lookup::for_each_logup_constraint_value(
                &circuit.graph.lookups,
                &scratch.buf,
                &scratch.s2_cur,
                &scratch.s2_next,
                publics_packed,
                delta_scaled,
                logup_selector,
                ext_modulus,
                ext_degree,
                circuit.lookup_group_size,
                |val| {
                    let triple = alpha_triples[idx];
                    acc0 += val * triple[0];
                    acc1 += val * triple[1];
                    acc2 += val * triple[2];
                    idx += 1;
                },
            );
            debug_assert_eq!(idx, constraint_count);
            PackedChallenge::<SC>::from_basis_coefficients_fn(|coeff_idx| match coeff_idx {
                0 => acc0,
                1 => acc1,
                _ => acc2,
            })
        }
        AlphaTable::Generic(decomposed_alpha_powers) => {
            scratch.accs.fill(PackedVal::<SC>::ZERO);
            circuit.graph.fold_constraints(
                &scratch.buf,
                decomposed_alpha_powers,
                &mut scratch.accs,
            );
            let mut idx = circuit.graph.zeros.len();
            crate::lookup::for_each_logup_constraint_value(
                &circuit.graph.lookups,
                &scratch.buf,
                &scratch.s2_cur,
                &scratch.s2_next,
                publics_packed,
                delta_scaled,
                logup_selector,
                ext_modulus,
                ext_degree,
                circuit.lookup_group_size,
                |val| {
                    for (acc, coeffs) in scratch.accs.iter_mut().zip(decomposed_alpha_powers) {
                        *acc += val * coeffs[idx];
                    }
                    idx += 1;
                },
            );
            debug_assert_eq!(idx, constraint_count);
            PackedChallenge::<SC>::from_basis_coefficients_fn(|coeff_idx| scratch.accs[coeff_idx])
        }
    };

    let quotient = accumulator * inv_vanishing;
    let basis = PackedExtension::<Val<SC>, SC::Challenge>::as_basis_coefficients_slice(&quotient);
    for (idx_in_packing, slot) in pack_out.iter_mut().enumerate() {
        *slot = SC::Challenge::from_basis_coefficients_fn(|coeff_idx| {
            basis[coeff_idx].as_slice()[idx_in_packing]
        });
    }
}
