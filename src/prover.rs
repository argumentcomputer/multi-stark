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
//! - w2_i = max(L_i, 1) — stage 2 width in extension field elements
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
    Com, Domain, EvaluationsOnDomain, PackedChallenge, PackedVal, PcsProof, StarkGenericConfig, Val,
};
use crate::eval::VarValues;
use crate::lookup::{LookupValues, fingerprint};
use crate::system::{ProverKey, System, SystemWitness};

use bincode::config::{Configuration, Fixint, LittleEndian, standard};
use bincode::error::{DecodeError, EncodeError};
use bincode::serde::{decode_from_slice, encode_to_vec};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{LagrangeSelectors, OpenedValuesForRound, Pcs, PolynomialSpace};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{
    Algebra, BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::{Matrix, bitrev::BitReversibleMatrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits_len};
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

impl<SC> System<SC>
where
    SC: StarkGenericConfig,
    // Two-adicity is needed to slice the quotient into coefficient slices
    // and rebuild their committed LDE from those coefficients; every
    // FRI-based config is two-adic anyway.
    Val<SC>: TwoAdicField + Ord,
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
        // n_i·B rows (an iDFT plus B coset DFTs per column), then Merkle
        // tree. FFT work: Σ w_i · (B+1) · n_i · log₂(n_i).
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

        // Lookup challenges.
        let lookup_argument_challenge: SC::Challenge = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: SC::Challenge = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);

        // Initial accumulator from the claims.
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
        // FFT work: Σ w2_i · D · (B+1) · n_i · log₂(n_i).
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

        // Constraint challenge.
        let constraint_challenge: SC::Challenge = challenger.sample_algebra_element();

        // Cost: "Quotient computation and commit" — constraint evaluation on
        // the quotient domain (Σ n_i·q_i·eval_cost(k_i)), the forward DFT of
        // the flattened quotient (Σ D·q_i·n_i·log₂(q_i·n_i)), then LDE +
        // Merkle of the sub-polynomials (Σ q_i·D·B·n_i·log₂(B·n_i)).
        let _g = tracing::info_span!("stark/quotient").entered();
        debug_assert_eq!(intermediate_accumulators.len(), active_indices.len());
        debug_assert_eq!(log_degrees.len(), active_indices.len());
        let dft = Radix2DitParallel::<Val<SC>>::default();
        let log_blowup = self.config.log_blowup();
        let quotient_ldes: Vec<_> = active_indices
            .iter()
            .zip(log_degrees.iter())
            .zip(intermediate_accumulators.iter())
            .enumerate()
            .map(|(pos, ((&ci, log_degree), next_acc))| {
                let circuit = &self.circuits[ci];
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

                // compute the quotient values which are elements of the extension field and flatten it to the base field
                let quotient_values = quotient_values::<SC>(
                    circuit,
                    &lookup_publics,
                    trace_domain,
                    quotient_domain,
                    &preprocessed_trace_on_quotient_domain,
                    &stage_1_trace_on_quotient_domain,
                    &stage_2_trace_on_quotient_domain,
                    constraint_challenge,
                    circuit.constraint_count(),
                );
                let quotient_flat =
                    RowMajorMatrix::new_col(quotient_values).flatten_to_base::<Val<SC>>();
                // The quotient has degree greater than the trace polynomials,
                // so for FRI to work it must be split into `quotient_degree`
                // sub-polynomials of trace degree. Slice its COEFFICIENTS:
                // `Q(X) = Σᵢ X^{i·n}·cᵢ(X)` with each `cᵢ` of degree < n, and
                // commit all slices as ONE `q·D`-column matrix on the trace
                // domain — instead of one matrix per slice on the split
                // cosets — so the opening phase pays its per-matrix costs
                // once per circuit rather than once per slice. The committed
                // LDE is built straight from these coefficients: evaluating
                // them onto the trace domain only for `Pcs::commit` to
                // inverse-DFT that evaluation right back would waste two
                // size-n transforms per column. The slicing itself is fused
                // with the iDFT's scaling passes into one parallel gather
                // (see `shifted_quotient_slices`).
                acc = *next_acc;
                let sliced = shifted_quotient_slices(
                    &dft,
                    quotient_flat,
                    quotient_domain.first_point(),
                    quotient_degree,
                );
                lde_from_shifted_coefficients(&dft, sliced, log_blowup)
            })
            .collect();
        // `commit_ldes` skips the randomization a hiding PCS applies inside
        // `commit`; this prover targets non-hiding configurations only.
        assert!(
            !<SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::ZK,
            "committing the quotient from coefficients bypasses hiding-PCS randomization"
        );
        let (quotient_commit, quotient_data) = pcs.commit_ldes(quotient_ldes);
        challenger.observe(quotient_commit.clone());
        drop(_g);

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

/// From the quotient's evaluations on its disjoint domain — `q·n` rows of
/// `D` base columns on the coset `shift·H` with `|H| = q·n` — produce the
/// `n`-row, `q·D`-column matrix of slice coefficients with the committed
/// LDE's `GENERATOR` shift already folded in: the input
/// [`lde_from_shifted_coefficients`] expects.
///
/// Semantically this is three steps: coset iDFT to coefficients, slicing
/// `Q(X) = Σₖ X^{k·n}·cₖ(X)` into rows `[c₀ | … | c_{q−1}]`, and
/// pre-scaling row `r` by `GENERATOR^r`. Executed literally (the library
/// entry points) those steps cost a bit-reversal materialization, a serial
/// row-swap pass, two serial full-matrix scaling passes, and a serial
/// gather. But the composition collapses: with `N = q·n` and `S` the raw
/// bit-reversed storage of the forward DFT,
///
/// ```text
///   idft(f)ⱼ = N⁻¹ · dft(f)_{(N−j) mod N} = N⁻¹ · S[rev((N−j) mod N)]
/// ```
///
/// and the coset-unscale factor `shift^{−j}` at `j = k·n + r` splits into
/// `shift^{−k·n} · shift^{−r}`, whose row-dependent part cancels the LDE
/// pre-scale `GENERATOR^r` exactly, because the disjoint quotient domain's
/// shift IS the generator (asserted below; `create_disjoint_domain` on a
/// natural trace domain guarantees it). What survives is a single parallel
/// gather off the DFT storage with ONE constant weight per slice:
/// `wₖ = N⁻¹ · GENERATOR^{−k·n}`.
fn shifted_quotient_slices<F: TwoAdicField + Ord>(
    dft: &Radix2DitParallel<F>,
    quotient_evals: RowMajorMatrix<F>,
    domain_shift: F,
    quotient_degree: usize,
) -> RowMajorMatrix<F> {
    assert_eq!(
        domain_shift,
        F::GENERATOR,
        "quotient domain shift must equal the LDE shift for the scalings to cancel"
    );
    let ext_degree = quotient_evals.width();
    let big_height = quotient_evals.height();
    let log_big_height = log2_strict_usize(big_height);
    debug_assert_eq!(big_height % quotient_degree, 0);
    let n = big_height / quotient_degree;
    let width = quotient_degree * ext_degree;
    // Raw storage of the forward DFT: natural index `k` lives at row
    // `rev(k)`, and the unwrap out of the bit-reversed view is copy-free.
    let storage = dft.dft_batch(quotient_evals).bit_reverse_rows();
    let n_inv = F::ONE.div_2exp_u64(log_big_height as u64);
    let weight_step = F::GENERATOR.exp_u64(n as u64).inverse();
    let weights: Vec<F> = weight_step
        .powers()
        .take(quotient_degree)
        .map(|w| w * n_inv)
        .collect();
    let mut values = F::zero_vec(n * width);
    values
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out)| {
            for (chunk, weight) in weights.iter().enumerate() {
                let j = chunk * n + row;
                let src = reverse_bits_len(
                    big_height.wrapping_sub(j) & (big_height - 1),
                    log_big_height,
                );
                let src = &storage.values[src * ext_degree..(src + 1) * ext_degree];
                for (out, src) in out[chunk * ext_degree..(chunk + 1) * ext_degree]
                    .iter_mut()
                    .zip(src)
                {
                    *out = *src * *weight;
                }
            }
        });
    RowMajorMatrix::new(values, width)
}

/// Low-degree extension of column polynomials given by their COEFFICIENTS
/// with the `GENERATOR` coset shift already folded in (row `j`
/// pre-multiplied by `GENERATOR^j`, which [`shifted_quotient_slices`]
/// produces for free), in the exact layout `Pcs::commit` stores for
/// evaluations on the natural domain: `2^log_blowup` row blocks, where
/// block `b` holds the evaluations on the coset `GENERATOR · w^rev(b) · H`
/// in bit-reversed row order (`H` is the size-`n` subgroup, `w` generates
/// the size-`2^log_blowup · n` subgroup, and `rev` reverses `log_blowup`
/// bits). Globally that is the bit-reversal of the natural order of the
/// whole blown-up coset — i.e.
/// `coset_lde_batch(evals, log_blowup, GENERATOR).bit_reverse_rows()`,
/// which is what `TwoAdicFriPcs::commit` computes.
///
/// Committing the result via `Pcs::commit_ldes` is therefore bit-identical
/// to `Pcs::commit` on the columns' trace-domain evaluations — field
/// arithmetic is exact, so equal polynomials give equal evaluations no
/// matter which transform produced them — while skipping both that
/// evaluation DFT and the inverse DFT `commit` would open with.
///
/// The whole extension is ONE size-`2^log_blowup · n` transform: zero-pad
/// the shifted coefficients to the LDE height (which leaves the column
/// polynomials unchanged) and DFT. That spends `log_blowup` more butterfly
/// layers than `2^log_blowup` separate size-`n` coset DFTs would, but one
/// batched transform is what the memory traffic wants: no per-coset matrix
/// clones, no per-coset serial shift-scaling passes inside
/// `coset_dft_batch`, no reassembly copies, and `Radix2DitParallel`'s
/// native output order is already the bit-reversed storage order, so the
/// final unwrap is copy-free.
fn lde_from_shifted_coefficients<F: TwoAdicField + Ord>(
    dft: &Radix2DitParallel<F>,
    mut coefficients: RowMajorMatrix<F>,
    log_blowup: usize,
) -> RowMajorMatrix<F> {
    let height = coefficients.height();
    coefficients.pad_to_height(height << log_blowup, F::ZERO);
    dft.dft_batch(coefficients).bit_reverse_rows()
}

/// Reference form of [`lde_from_shifted_coefficients`] taking PLAIN
/// coefficients: folds the `GENERATOR` shift in explicitly. Only the
/// pinning tests need it; the prover gets the shift for free inside
/// [`shifted_quotient_slices`].
#[cfg(test)]
fn lde_from_coefficients<F: TwoAdicField + Ord>(
    dft: &Radix2DitParallel<F>,
    mut coefficients: RowMajorMatrix<F>,
    log_blowup: usize,
) -> RowMajorMatrix<F> {
    scale_rows_by_powers(&mut coefficients, F::GENERATOR);
    lde_from_shifted_coefficients(dft, coefficients, log_blowup)
}

/// Multiplies row `j` of `mat` by `base^j`, in parallel: each chunk of rows
/// pays one exponentiation and steps serially from there.
#[cfg(test)]
fn scale_rows_by_powers<F: Field>(mat: &mut RowMajorMatrix<F>, base: F) {
    const ROWS_PER_CHUNK: usize = 512;
    let width = mat.width();
    mat.values
        .par_chunks_mut(ROWS_PER_CHUNK * width)
        .enumerate()
        .for_each(|(chunk, rows)| {
            let mut weight = base.exp_u64((chunk * ROWS_PER_CHUNK) as u64);
            for row in rows.chunks_mut(width) {
                for value in row {
                    *value *= weight;
                }
                weight *= base;
            }
        });
}

/// Evaluates the folded constraints on the quotient domain and divides by
/// the vanishing polynomial, producing the quotient values.
#[allow(clippy::too_many_arguments)]
fn quotient_values<SC>(
    circuit: &crate::system::Circuit<Val<SC>>,
    lookup_publics: &[Val<SC>],
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
    Val<SC>: TwoAdicField + Ord,
{
    let quotient_size = quotient_domain.size();
    let main_width = circuit.main_width;
    let stage_2_width = circuit.stage_2_width;
    let preprocessed_width = circuit.preprocessed_width;
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);
    // The logUp wrap constraint consumes the last-row selector ADDITIVELY,
    // so its normalization matters — but the selector multiplies Δ, which
    // is constant across the domain, so the normalization constant
    // 1/(n·g) is absorbed into Δ once per circuit instead of rescaling
    // the selector vectors (p3's unnormalized L_last has value n·g at the
    // last row; see the selector-normalization pin test).
    let n_val = Val::<SC>::from_usize(trace_domain.size());
    let g = Val::<SC>::two_adic_generator(log2_strict_usize(trace_domain.size()));
    let inj_norm = (n_val * g).inverse();

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_vanishing.push(Val::<SC>::default());
    }

    // α powers in reverse (constraint i of k weighted by α^{k-1-i}),
    // decomposed per basis coordinate for the batched base-field fold.
    let mut alpha_powers = alpha.powers().collect_n(constraint_count);
    alpha_powers.reverse();
    let decomposed_alpha_powers: Vec<Vec<Val<SC>>> =
        (0..<SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION)
            .map(|i| {
                alpha_powers
                    .iter()
                    .map(|x| x.as_basis_coefficients_slice()[i])
                    .collect()
            })
            .collect();

    // Public coordinates broadcast to packed base values.
    let publics_packed: Vec<PackedVal<SC>> = lookup_publics
        .iter()
        .map(|&c| PackedVal::<SC>::from(c))
        .collect();

    // Δ/(n·g) per coordinate, broadcast: the logUp boundary injection with
    // the last-row selector's normalization constant pre-absorbed.
    let ext_d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
    let delta_scaled: Vec<PackedVal<SC>> = (0..ext_d)
        .map(|k| {
            PackedVal::<SC>::from(
                (lookup_publics[3 * ext_d + k] - lookup_publics[2 * ext_d + k]) * inj_norm,
            )
        })
        .collect();

    let ext_params = crate::system::extension_params::<SC>();
    let inner = |i_start: usize| {
        quotient_values_inner::<SC>(
            circuit,
            &sels,
            quotient_size,
            preprocessed_on_quotient_domain,
            stage_1_on_quotient_domain,
            stage_2_on_quotient_domain,
            main_width,
            stage_2_width,
            preprocessed_width,
            &publics_packed,
            &delta_scaled,
            &decomposed_alpha_powers,
            next_step,
            i_start,
            ext_params.w,
            ext_params.degree,
        )
    };
    #[cfg(feature = "parallel")]
    {
        (0..quotient_size)
            .into_par_iter()
            .step_by(PackedVal::<SC>::WIDTH)
            .flat_map_iter(inner)
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..quotient_size)
            .step_by(PackedVal::<SC>::WIDTH)
            .flat_map_iter(inner)
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn quotient_values_inner<SC>(
    circuit: &crate::system::Circuit<Val<SC>>,
    sels: &LagrangeSelectors<Vec<Val<SC>>>,
    quotient_size: usize,
    preprocessed_on_quotient_domain: &Option<EvaluationsOnDomain<'_, SC>>,
    stage_1_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    stage_2_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    main_width: usize,
    stage_2_width: usize,
    preprocessed_width: usize,
    publics_packed: &[PackedVal<SC>],
    delta_scaled: &[PackedVal<SC>],
    decomposed_alpha_powers: &[Vec<Val<SC>>],
    next_step: usize,
    i_start: usize,
    ext_w: Val<SC>,
    ext_degree: usize,
) -> impl Iterator<Item = SC::Challenge>
where
    SC: StarkGenericConfig,
    Val<SC>: TwoAdicField + Ord,
{
    let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;
    let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
    let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
    let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
    let inv_vanishing = *PackedVal::<SC>::from_slice(&sels.inv_vanishing[i_range]);

    // Packed two-row windows of each trace, as base columns.
    let preprocessed_pair: Option<Vec<PackedVal<SC>>> = preprocessed_on_quotient_domain
        .as_ref()
        .map(|m| m.vertically_packed_row_pair::<PackedVal<SC>>(i_start, next_step));
    let stage_1_pair =
        stage_1_on_quotient_domain.vertically_packed_row_pair::<PackedVal<SC>>(i_start, next_step);
    let stage_2_pair =
        stage_2_on_quotient_domain.vertically_packed_row_pair::<PackedVal<SC>>(i_start, next_step);

    let (stage_1_cur, stage_1_next) = stage_1_pair.split_at(main_width);
    let (stage_2_cur, stage_2_next) = stage_2_pair.split_at(stage_2_width);
    let (preprocessed_cur, preprocessed_next): (&[PackedVal<SC>], &[PackedVal<SC>]) =
        match &preprocessed_pair {
            Some(pair) => pair.split_at(preprocessed_width),
            None => (&[], &[]),
        };

    let view = VarValues {
        preprocessed: [preprocessed_cur, preprocessed_next],
        main: [stage_1_cur, stage_1_next],
        stage2: [stage_2_cur, stage_2_next],
        publics: publics_packed,
        is_first_row,
        is_last_row,
        is_transition,
    };
    let mut buf = Vec::new();
    circuit.graph.sweep(&view, &mut buf);
    let mut constraint_values = circuit.graph.constraint_values(&buf);
    // The logUp constraint values are evaluated directly (they are not
    // compiled into the graph), appended after the user roots in the
    // canonical protocol order. Coordinate-expanded logUp constraints are
    // base-field-only, so they evaluate in `PackedVal` like everything else;
    // the lookup expressions are read straight out of the sweep buffer.
    crate::lookup::logup_constraint_values(
        &circuit.graph.lookups,
        &buf,
        stage_2_cur,
        stage_2_next,
        publics_packed,
        delta_scaled,
        // p3's (unnormalized) last-row selector; the normalization
        // constant is pre-absorbed into `delta_scaled`.
        is_last_row,
        ext_w,
        ext_degree,
        &mut constraint_values,
    );
    debug_assert_eq!(constraint_values.len(), circuit.constraint_count());

    // Fold the base constraint values with α through the decomposed path,
    // reassembling one packed extension accumulator.
    let accumulator = PackedChallenge::<SC>::from_basis_coefficients_fn(|coeff_idx| {
        PackedVal::<SC>::batched_linear_combination(
            &constraint_values,
            &decomposed_alpha_powers[coeff_idx],
        )
    });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Val;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    /// `lde_from_coefficients` must reproduce, value for value, the matrix
    /// `TwoAdicFriPcs::commit` stores for the same polynomials given as
    /// trace-domain evaluations: `coset_lde_batch` with the generator shift
    /// (the natural domain's shift is one), then a bit-reversal. This pins
    /// the exact substitution the quotient commit relies on.
    #[test]
    fn lde_from_coefficients_matches_commit_transform() {
        let mut rng = SmallRng::seed_from_u64(0);
        let dft = Radix2DitParallel::<Val>::default();
        for log_height in [0usize, 1, 2, 5, 8] {
            for log_blowup in [1usize, 2, 3] {
                for width in [1usize, 2, 7] {
                    let height = 1 << log_height;
                    let coefficients = RowMajorMatrix::new(
                        (0..height * width).map(|_| rng.random()).collect(),
                        width,
                    );
                    let evaluations = dft
                        .coset_dft_batch(coefficients.clone(), Val::ONE)
                        .to_row_major_matrix();
                    let expected = dft
                        .coset_lde_batch(evaluations, log_blowup, Val::GENERATOR)
                        .bit_reverse_rows()
                        .to_row_major_matrix();
                    let got = lde_from_coefficients(&dft, coefficients, log_blowup);
                    assert_eq!(got, expected, "h=2^{log_height} B=2^{log_blowup} w={width}");
                }
            }
        }
    }

    /// `shifted_quotient_slices` must reproduce, value for value, the naive
    /// composition it replaces: coset iDFT off the quotient domain, slicing
    /// the coefficients into `q` chunks per row, and folding the committed
    /// LDE's `GENERATOR` shift into the rows. This pins the scaling
    /// cancellation the fused gather relies on.
    #[test]
    fn shifted_quotient_slices_matches_naive_composition() {
        let mut rng = SmallRng::seed_from_u64(1);
        let dft = Radix2DitParallel::<Val>::default();
        for log_n in [0usize, 1, 2, 5, 7] {
            for quotient_degree in [1usize, 2, 4] {
                for ext_degree in [1usize, 2] {
                    let n = 1 << log_n;
                    let big_height = n * quotient_degree;
                    let evals = RowMajorMatrix::new(
                        (0..big_height * ext_degree).map(|_| rng.random()).collect(),
                        ext_degree,
                    );
                    // Naive path: coset iDFT, slice, fold the LDE shift in.
                    let coefficients = dft.coset_idft_batch(evals.clone(), Val::GENERATOR);
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
                    let mut expected = RowMajorMatrix::new(sliced, width);
                    scale_rows_by_powers(&mut expected, Val::GENERATOR);
                    let got = shifted_quotient_slices(&dft, evals, Val::GENERATOR, quotient_degree);
                    assert_eq!(
                        got, expected,
                        "n=2^{log_n} q={quotient_degree} D={ext_degree}"
                    );
                }
            }
        }
    }
}
