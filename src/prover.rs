//! Multi-circuit STARK prover.
//!
//! The proving protocol proceeds in several stages sharing one Fiat-Shamir
//! transcript. The challenger starts from a seed binding a domain tag and all
//! protocol parameters (see the transcript contract on
//! [`StarkGenericConfig::initialise_challenger`]), and the system shape
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
//!    (running accumulator and message inverses per row) and committed via PCS. Each
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
//! - w2_i = 1 + L_i — stage 2 width in extension field elements
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
//! FFT work:   Σ_i  w_i · n_i · B · log₂(n_i · B)   field multiplications
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
//! FFT work:   Σ_i  w2_i · D · n_i · B · log₂(n_i · B)   field multiplications
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
//! circuit rather than once per slice.
//!
//! ```text
//! Constraint eval:  Σ_i  n_i · q_i · eval_cost(k_i)         field operations
//! Quotient FFT:     Σ_i  q_i · D · n_i · B · log₂(n_i · B)  field multiplications
//! Hashing:          Σ_i  q_i · n_i · B                        Merkle leaf hashes
//! ```
//!
//! Here eval_cost(k_i) denotes the per-row cost of evaluating all k_i folded
//! constraints, which depends on the specific AIR.
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
//! C_prove  ≈  Σ_i  n_i · B · log₂(n_i · B) · W_i     (FFT — all commit rounds)
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

use std::ops::Deref;

use crate::{
    builder::compiled::{CompiledConstraints, PreparedConstraints, RowChunk, Scratch},
    builder::symbolic::{SymbolicAirBuilder, get_symbolic_constraints},
    config::{
        Com, Domain, EvaluationsOnDomain, PackedChallenge, PackedVal, PcsProof, StarkGenericConfig,
        Val,
    },
    lookup::{LOOKUP_PUBLIC_SIZE, LookupValues, fingerprint},
    system::{ProverKey, System, SystemWitness},
};
use bincode::{
    config::{Configuration, Fixint, LittleEndian, standard},
    error::{DecodeError, EncodeError},
    serde::{decode_from_slice, encode_to_vec},
};
use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{LagrangeSelectors, OpenedValuesForRound, Pcs, PolynomialSpace};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

/// Polynomial commitments included in the proof.
#[derive(Serialize, Deserialize)]
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
pub struct Proof<SC: StarkGenericConfig> {
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

impl<SC: StarkGenericConfig> Proof<SC> {
    fn serde_config() -> Configuration<LittleEndian, Fixint> {
        standard().with_little_endian().with_fixed_int_encoding()
    }

    #[inline]
    pub fn to_bytes(&self) -> Result<Vec<u8>, EncodeError> {
        encode_to_vec(self, Self::serde_config())
    }

    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let (proof, _num_bytes) = decode_from_slice(bytes, Self::serde_config())?;
        Ok(proof)
    }
}

impl<SC, A> System<SC, A>
where
    SC: StarkGenericConfig,
    // Two-adicity is needed to re-base the quotient's coefficient slices
    // onto the trace domain; every FRI-based config is two-adic anyway.
    Val<SC>: TwoAdicField + Ord,
    // The prover evaluates constraints through their compiled form, obtained
    // by running the AIR once over the symbolic builder (same bound as
    // system setup) — not through a folder over the AIR itself.
    A: BaseAir<Val<SC>> + Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>,
{
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
    /// Panics if every circuit's trace is empty (nothing to prove).
    #[tracing::instrument(level = "info", skip_all, name = "stark/prove")]
    pub fn prove_multiple_claims(
        &self,
        key: &ProverKey<SC>,
        claims: &[&[Val<SC>]],
        witness: SystemWitness<Val<SC>>,
    ) -> Proof<SC> {
        // initialize pcs and challenger
        let pcs = self.config.pcs();
        let mut challenger = self.config.initialise_challenger();

        // Bind the system shape into the transcript. The protocol parameters
        // are already bound via the challenger seed.
        self.observe_shape(&mut challenger);

        // Sparse activation: a circuit whose stage-1 trace is empty is
        // INACTIVE for this proof — nothing of it is committed, opened,
        // accumulated, or constraint-evaluated. Soundness of the omission
        // rests on the lookup accumulator: an inactive circuit has no rows,
        // hence no sends or receives, so omitting it leaves the global
        // balance unchanged — while dishonestly deactivating a circuit the
        // execution needs leaves an unmatched channel send and the final
        // accumulator cannot be zero. The activation bitmap is bound into
        // the transcript here, before any commitment or challenge.
        let active: Vec<bool> = witness.traces.iter().map(|t| t.height() > 0).collect();
        for &is_active in &active {
            challenger.observe(Val::<SC>::from_bool(is_active));
        }
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
        // Canonical index -> active position (None = inactive).
        let mut active_pos: Vec<Option<usize>> = vec![None; active.len()];
        for (pos, &ci) in active_indices.iter().enumerate() {
            active_pos[ci] = Some(pos);
        }

        // Cost: "Stage 1 commit" — coset LDE (FFT) of each trace from n_i to
        // n_i·B rows, then Merkle tree. FFT work: Σ w_i · n_i · B · log₂(n_i·B).
        let _g = tracing::info_span!("stark/stage1_commit").entered();
        let mut log_degrees = vec![];
        let evaluations = witness
            .traces
            .into_iter()
            .zip(active.clone())
            .filter_map(|(trace, is_active)| is_active.then_some(trace))
            .map(|trace| {
                let degree = trace.height();
                let log_degree = log2_strict_usize(degree);
                let trace_domain = pcs.natural_domain_for_degree(degree);
                log_degrees.push(log_degree);
                (trace_domain, trace)
            });
        let (stage_1_trace_commit, stage_1_trace_data) = pcs.commit(evaluations);
        drop(_g);

        if let Some(commit) = &self.preprocessed_commit {
            challenger.observe(commit.clone());
        }
        challenger.observe(stage_1_trace_commit.clone());

        // Observe the traces' heights. This binds the proof to specific domain
        // sizes; the verifier reads these from the (untrusted) proof, so they
        // must influence every subsequent challenge.
        for log_degree in &log_degrees {
            challenger.observe(Val::<SC>::from_usize(*log_degree));
        }

        // Observe the claims, length-prefixed so that distinct claim
        // structures (e.g. [[a, b]] vs [[a], [b]]) yield distinct transcripts.
        // This has to be done before generating the lookup argument challenge,
        // otherwise the lookup argument can be attacked.
        challenger.observe(Val::<SC>::from_usize(claims.len()));
        for claim in claims {
            challenger.observe(Val::<SC>::from_usize(claim.len()));
            challenger.observe_slice(claim);
        }

        // generate lookup challenges
        let lookup_argument_challenge: SC::Challenge = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: SC::Challenge = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);

        // construct the accumulator from the claims
        let mut acc = SC::Challenge::ZERO;
        for claim in claims {
            let message = lookup_argument_challenge
                + fingerprint(&fingerprint_challenge, claim.iter().cloned());
            acc += message.inverse();
        }

        // Cost: "Lookup trace construction" — fingerprint (Horner), batch
        // inversion, and accumulator update. Total: Σ n_i·L_i extension field ops.
        let _g = tracing::info_span!("stark/lookup_construction").entered();
        // Only active circuits enter the accumulator chain; the chain (and
        // `intermediate_accumulators`) is indexed by active position.
        let active_lookups: Vec<_> = witness
            .lookups
            .into_iter()
            .zip(&active)
            .filter_map(|(l, &is_active)| is_active.then_some(l))
            .collect();
        let (stage_2_traces, intermediate_accumulators) = LookupValues::stage_2_traces(
            &active_lookups,
            lookup_argument_challenge,
            &fingerprint_challenge,
            acc,
        );
        // The lookup witness can be as large as the traces themselves; free it
        // now instead of holding it through the commit/quotient/FRI stages.
        drop(active_lookups);
        drop(_g);

        // Cost: "Stage 2 commit" — LDE + Merkle for flattened extension traces.
        // FFT work: Σ w2_i · D · n_i · B · log₂(n_i·B).
        let _g = tracing::info_span!("stark/stage2_commit").entered();
        let evaluations = stage_2_traces.into_iter().map(|trace| {
            let degree = trace.height();
            let trace_domain = pcs.natural_domain_for_degree(degree);
            (trace_domain, trace.flatten_to_base())
        });
        let (stage_2_trace_commit, stage_2_trace_data) = pcs.commit(evaluations);
        drop(_g);
        challenger.observe(stage_2_trace_commit.clone());

        // Observe the intermediate accumulators. They enter the constraints as
        // public values, so later challenges (α, ζ) must depend on them
        // directly rather than only through the quotient commitment.
        for acc in &intermediate_accumulators {
            challenger.observe_algebra_element(*acc);
        }

        // generate constraint challenge
        let constraint_challenge: SC::Challenge = challenger.sample_algebra_element();

        // Cost: "Quotient computation and commit" — constraint evaluation on the
        // quotient domain (Σ n_i·q_i·eval_cost(k_i)) plus LDE + Merkle of the
        // quotient sub-polynomials (Σ q_i·D·n_i·B·log₂(n_i·B)).
        let _g = tracing::info_span!("stark/quotient").entered();
        debug_assert_eq!(intermediate_accumulators.len(), active_indices.len());
        debug_assert_eq!(log_degrees.len(), active_indices.len());
        let dft = Radix2DitParallel::<Val<SC>>::default();
        let quotient_evaluations = active_indices
            .iter()
            .zip(log_degrees.iter())
            .zip(intermediate_accumulators.iter())
            .enumerate()
            .map(|(pos, ((&ci, log_degree), next_acc))| {
                let circuit = &self.circuits[ci];
                let air = &circuit.air;
                let quotient_degree = circuit.quotient_degree();
                let log_quotient_degree = log2_strict_usize(quotient_degree);
                let trace_domain = pcs.natural_domain_for_degree(1 << log_degree);
                let quotient_domain =
                    trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));
                let preprocessed_trace_on_quotient_domain = key
                    .preprocessed_data
                    .as_ref()
                    .zip(self.preprocessed_indices[ci])
                    .map(|(preprocessed_trace_data, preprocessed_idx)| {
                        pcs.get_evaluations_on_domain(
                            preprocessed_trace_data,
                            preprocessed_idx,
                            quotient_domain,
                        )
                    });
                let stage_1_trace_on_quotient_domain =
                    pcs.get_evaluations_on_domain(&stage_1_trace_data, pos, quotient_domain);
                let stage_2_trace_on_quotient_domain =
                    pcs.get_evaluations_on_domain(&stage_2_trace_data, pos, quotient_domain);
                // compute the quotient values which are elements of the extension field and flatten it to the base field
                let public_values: [Val<SC>; 0] = [];
                let stage_2_public_values = [
                    lookup_argument_challenge,
                    fingerprint_challenge,
                    acc,
                    *next_acc,
                ];
                // Lower the circuit's symbolic constraints into the linear
                // program the evaluation loop executes. The hash-consing pass
                // collapses duplicated sub-expressions, so this once-per-proof
                // cost buys a much cheaper per-row evaluation below.
                let _g = tracing::info_span!("quotient/compile").entered();
                let symbolic_constraints = get_symbolic_constraints::<Val<SC>, SC::Challenge, _>(
                    air,
                    circuit.preprocessed_width,
                    circuit.stage_1_width,
                    circuit.stage_2_width,
                    public_values.len(),
                    LOOKUP_PUBLIC_SIZE,
                );
                debug_assert_eq!(symbolic_constraints.len(), circuit.constraint_count);
                let compiled = CompiledConstraints::<Val<SC>, SC::Challenge>::compile(
                    &symbolic_constraints,
                    circuit.preprocessed_width,
                    circuit.stage_1_width,
                    circuit.stage_2_width,
                );
                drop(_g);
                let _g = tracing::info_span!("quotient/values").entered();
                let quotient_values = quotient_values::<SC>(
                    &compiled,
                    &public_values,
                    &stage_2_public_values,
                    trace_domain,
                    quotient_domain,
                    &preprocessed_trace_on_quotient_domain,
                    &stage_1_trace_on_quotient_domain,
                    &stage_2_trace_on_quotient_domain,
                    constraint_challenge,
                    circuit.constraint_count,
                );
                drop(_g);
                let _g = tracing::info_span!("quotient/split").entered();
                let quotient_flat =
                    RowMajorMatrix::new_col(quotient_values).flatten_to_base::<Val<SC>>();
                // The quotient has degree greater than the trace polynomials,
                // so for FRI to work it must be split into `quotient_degree`
                // sub-polynomials of trace degree. Slice its COEFFICIENTS:
                // `Q(X) = Σᵢ X^{i·n}·cᵢ(X)` with each `cᵢ` of degree < n, and
                // commit all slices as ONE `q·D`-column matrix on the trace
                // domain — instead of one matrix per slice on the split
                // cosets — so the opening phase pays its per-matrix costs
                // once per circuit rather than once per slice.
                let coefficients =
                    dft.coset_idft_batch(quotient_flat, quotient_domain.first_point());
                let ext_degree = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
                let n = 1 << log_degree;
                let width = quotient_degree * ext_degree;
                let mut sliced = Vec::with_capacity(n * width);
                for row in 0..n {
                    for chunk in 0..quotient_degree {
                        sliced.extend_from_slice(
                            &coefficients.values[(chunk * n + row) * ext_degree
                                ..(chunk * n + row + 1) * ext_degree],
                        );
                    }
                }
                let quotient_chunks_evals = dft
                    .coset_dft_batch(
                        RowMajorMatrix::new(sliced, width),
                        trace_domain.first_point(),
                    )
                    .to_row_major_matrix();
                drop(_g);
                acc = *next_acc;
                (trace_domain, quotient_chunks_evals)
            });
        let (quotient_commit, quotient_data) = pcs.commit(quotient_evaluations);
        challenger.observe(quotient_commit.clone());
        drop(_g);

        // save the commitments
        let commitments = Commitments {
            stage_1_trace: stage_1_trace_commit,
            stage_2_trace: stage_2_trace_commit,
            quotient_chunks: quotient_commit,
        };

        // Cost: "FRI opening" — barycentric interpolation (Σ n_i·B·W_i),
        // FRI folding (≈ H), and FRI queries (Q·R·log₂ H hash ops).
        let _g = tracing::info_span!("stark/fri_open").entered();
        let zeta: SC::Challenge = challenger.sample_algebra_element();
        let mut round0_openings = vec![];
        let mut round1_openings = vec![];
        let mut round2_openings = vec![];
        let mut round3_openings = vec![];
        for &log_degree in log_degrees.iter() {
            let trace_domain = pcs.natural_domain_for_degree(1 << log_degree);
            let zeta_next = trace_domain
                .next_point(zeta)
                .expect("domain has no next point");
            round1_openings.push(vec![zeta, zeta_next]);
            round2_openings.push(vec![zeta, zeta_next]);
            // One wide matrix per circuit holds all its quotient slices.
            round3_openings.push(vec![zeta]);
        }
        // The preprocessed commitment is built once over ALL preprocessed
        // traces at system construction, so its round must carry one entry
        // per preprocessed matrix regardless of activation: an inactive
        // circuit's preprocessed matrix is opened at no points.
        for (prep_index, &pos) in self.preprocessed_indices.iter().zip(&active_pos) {
            if prep_index.is_some() {
                match pos {
                    Some(pos) => {
                        let trace_domain = pcs.natural_domain_for_degree(1 << log_degrees[pos]);
                        let zeta_next = trace_domain
                            .next_point(zeta)
                            .expect("domain has no next point");
                        round0_openings.push(vec![zeta, zeta_next]);
                    }
                    None => round0_openings.push(vec![]),
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

#[allow(clippy::too_many_arguments)]
fn quotient_values<SC>(
    compiled: &CompiledConstraints<Val<SC>, SC::Challenge>,
    public_values: &[Val<SC>],
    stage_2_public_values: &[SC::Challenge],
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    preprocessed_on_quotient_domain: &Option<EvaluationsOnDomain<'_, SC>>,
    stage_1_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    stage_2_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    alpha: SC::Challenge,
    constraint_count: usize,
) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
{
    let quotient_size = quotient_domain.size();
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_vanishing.push(Val::<SC>::default());
    }

    let mut alpha_powers = alpha.powers().collect_n(constraint_count);
    alpha_powers.reverse();

    let prepared = compiled.prepare(public_values, stage_2_public_values, &alpha_powers);

    #[cfg(feature = "parallel")]
    {
        (0..quotient_size)
            .into_par_iter()
            .step_by(PackedVal::<SC>::WIDTH)
            .map_init(
                || compiled.scratch(),
                |scratch, i_start| {
                    quotient_values_inner::<SC>(
                        compiled,
                        &prepared,
                        &sels,
                        quotient_size,
                        preprocessed_on_quotient_domain,
                        stage_1_on_quotient_domain,
                        stage_2_on_quotient_domain,
                        next_step,
                        i_start,
                        scratch,
                    )
                },
            )
            .flatten_iter()
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = compiled.scratch();
        let mut values = Vec::with_capacity(quotient_size);
        for i_start in (0..quotient_size).step_by(PackedVal::<SC>::WIDTH) {
            values.extend(quotient_values_inner::<SC>(
                compiled,
                &prepared,
                &sels,
                quotient_size,
                preprocessed_on_quotient_domain,
                stage_1_on_quotient_domain,
                stage_2_on_quotient_domain,
                next_step,
                i_start,
                &mut scratch,
            ));
        }
        values
    }
}

#[allow(clippy::too_many_arguments)]
fn quotient_values_inner<SC>(
    compiled: &CompiledConstraints<Val<SC>, SC::Challenge>,
    prepared: &PreparedConstraints<Val<SC>, SC::Challenge>,
    sels: &LagrangeSelectors<Vec<Val<SC>>>,
    quotient_size: usize,
    preprocessed_on_quotient_domain: &Option<EvaluationsOnDomain<'_, SC>>,
    stage_1_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    stage_2_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    next_step: usize,
    i_start: usize,
    scratch: &mut Scratch<Val<SC>, SC::Challenge>,
    // The returned iterator owns its data (`use<SC>`): it must not capture
    // the input borrows, so each worker's scratch can be reused while
    // previously returned iterators are still being drained.
) -> impl Iterator<Item = SC::Challenge> + use<SC>
where
    SC: StarkGenericConfig,
{
    let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

    let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
    let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
    let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
    let inv_vanishing = *PackedVal::<SC>::from_slice(&sels.inv_vanishing[i_range]);

    // Local row followed by next row, matching the compiled program's flat
    // `offset * width + column` operand indexing.
    let preprocessed: Option<Vec<PackedVal<SC>>> =
        preprocessed_on_quotient_domain
            .as_ref()
            .map(|on_quotient_domain| {
                on_quotient_domain.vertically_packed_row_pair(i_start, next_step)
            });
    let stage_1: Vec<PackedVal<SC>> =
        stage_1_on_quotient_domain.vertically_packed_row_pair(i_start, next_step);
    let stage_2: Vec<PackedChallenge<SC>> = pack::<SC>(
        stage_2_on_quotient_domain.width(),
        stage_2_on_quotient_domain.wrapping_row_slices(i_start, PackedVal::<SC>::WIDTH),
    )
    .chain(pack::<SC>(
        stage_2_on_quotient_domain.width(),
        stage_2_on_quotient_domain.wrapping_row_slices(i_start + next_step, PackedVal::<SC>::WIDTH),
    ))
    .collect();

    let chunk = RowChunk {
        stage_1: &stage_1,
        preprocessed: preprocessed.as_deref().unwrap_or(&[]),
        stage_2: &stage_2,
        is_first_row,
        is_last_row,
        is_transition,
    };
    let accumulator = compiled.eval(prepared, &chunk, scratch);

    let quotient = accumulator * inv_vanishing;

    (0..quotient_size.min(PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
        SC::Challenge::from_basis_coefficients_fn(|coeff_idx| {
            <PackedChallenge<SC> as BasedVectorSpace<PackedVal<SC>>>::as_basis_coefficients_slice(
                &quotient,
            )[coeff_idx]
                .as_slice()[idx_in_packing]
        })
    })
}

fn pack<SC: StarkGenericConfig>(
    n: usize,
    rows: Vec<impl Deref<Target = [Val<SC>]>>,
) -> impl Iterator<Item = PackedChallenge<SC>> {
    let extension_d = <PackedChallenge<SC> as BasedVectorSpace<PackedVal<SC>>>::DIMENSION;
    (0..n).step_by(extension_d).map(move |c| {
        PackedChallenge::<SC>::from_basis_coefficients_fn(|j| {
            PackedVal::<SC>::from_fn(|i| rows[i][c + j])
        })
    })
}
