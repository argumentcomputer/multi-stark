//! Multi-circuit STARK verifier (constraint-IR version).
//!
//! The out-of-domain check recomputes each circuit's composition polynomial by
//! running the compiled circuit's dense constraint sweep in the challenge field
//! at ζ, then folding the constraint values with the constraint challenge.
//! Opened stage-2 columns are fed as base columns directly (no `from_ext_basis`
//! reassembly): the coordinate-expanded constraints are base-field polynomial
//! identities, so evaluating them at the ζ-openings is correct.
//!
//! # Verification steps
//!
//! 1. **Shape check** ([`System::verify_shape`]): Validate the activation bitmap
//!    (it must cover the canonical circuit set and activate at least one circuit)
//!    and that the proof's array dimensions (opened values, accumulators, quotient
//!    chunks) match the ACTIVE circuit count and the canonical column widths.
//!
//! 2. **Accumulator balance**: Assert that the last intermediate accumulator is zero,
//!    ensuring that all lookup pushes and pulls cancel out across circuits.
//!
//! 3. **Fiat-Shamir replay**: Reconstruct the challenger state identically to the
//!    prover: starting from the parameter-seeded challenger, observe the system
//!    shape, the activation bitmap, commitments, trace heights, length-prefixed
//!    claims, and intermediate accumulators, sampling the same challenges (lookup,
//!    fingerprint, constraint alpha, OOD zeta) at the same points in the
//!    transcript.
//!
//! 4. **PCS verification**: Verify the FRI opening proofs against the committed
//!    polynomials at the sampled points.
//!
//! 5. **OOD evaluation**: For each ACTIVE circuit, recompute the composition
//!    polynomial at zeta from the opened values and verify that
//!    `composition(zeta) * inv_vanishing(zeta) == quotient(zeta)`.
//!
//! See [`VerificationError`] for the possible failure modes.
//!
//! # Soundness argument
//!
//! The protocol is sound in the random oracle model, instantiated by the
//! configuration's Fiat-Shamir challenger.
//! Informally: if a prover produces a proof that the verifier accepts, then with
//! overwhelming probability the claimed computation is correct.
//!
//! We use the following notation throughout:
//! - |F_ext| — size of the challenge (extension) field. All Schwartz-Zippel
//!   terms below scale with 1/|F_ext|, so the configuration must pick an
//!   extension large enough for the target security level: the reference
//!   config's degree-2 Goldilocks extension gives |F_ext| ≈ 2^128; a degree-4
//!   BabyBear extension gives ≈ 2^124; a degree-2 BabyBear extension (≈ 2^62)
//!   would be far too small.
//! - ρ = 2^(-log_blowup) — FRI rate parameter (inverse of the blowup factor)
//! - n — number of FRI queries (`num_queries`)
//! - k — number of constraints (after lookup expansion)
//! - N — total number of lookup rows across all circuits
//! - D — maximum degree of the quotient polynomial (trace_degree × quotient_degree)
//!
//! ## FRI proximity test
//!
//! The FRI-based PCS guarantees that committed polynomials are close to polynomials
//! of the claimed degree. Two regimes must be distinguished when picking parameters:
//!
//! - **Conjectured soundness** (up to list-decoding capacity): each query catches a
//!   cheating prover with probability ≈ 1 - ρ, giving a query-phase error of ≈ **ρ^n**.
//!   With `log_blowup = 1` and `num_queries = 100`, this is ≈ 2^(-100).
//! - **Proven soundness** (Johnson bound): each query only provably catches a
//!   cheating prover with probability ≈ 1 - √ρ, giving ≈ **(√ρ)^n** — roughly half
//!   the bits of the conjectured bound. The same parameters provably give only
//!   ≈ 2^(-50).
//!
//! Like most production STARKs, parameters here are typically chosen against the
//! conjectured bound; be aware of the distinction when quoting a security level.
//! For the precise bounds, see the FRI soundness analysis in the Plonky3
//! documentation.
//!
//! The proof-of-work (PoW) phases add grinding cost: a cheating prover must perform
//! 2^(commit_proof_of_work_bits) work per batching challenge and
//! 2^(query_proof_of_work_bits) work before query sampling. This increases the
//! concrete cost of attack without affecting honest verification time.
//!
//! ## Constraint folding (α)
//!
//! All k constraints are folded into a single composition polynomial using
//! powers of a random challenge α. The folded polynomial Σ α^i · C_i(x) has degree
//! k - 1 in α. By the Schwartz-Zippel lemma, if any individual constraint C_i is
//! nonzero, the folded sum is nonzero with probability at least
//! **1 - (k - 1) / |F_ext|**, which is negligible for practical constraint counts
//! when |F_ext| is around 2^128.
//!
//! ## Out-of-domain evaluation (ζ)
//!
//! The verifier checks `composition(ζ) · inv_vanishing(ζ) = quotient(ζ)` at a
//! random point ζ. If the composition polynomial is not divisible by the vanishing
//! polynomial (i.e. some constraint is violated on the trace domain), the
//! difference `composition · inv_vanishing - quotient` is a nonzero polynomial of
//! degree at most D. By Schwartz-Zippel, this check passes incorrectly with
//! probability at most **D / |F_ext|**, which is negligible.
//!
//! ## Lookup argument
//!
//! The accumulator-based lookup argument uses two random challenges (β, γ) to
//! compress lookup messages into field elements. For each lookup interaction, the
//! message `m = β + fingerprint(γ, args)` is a random affine function of the
//! challenges. If the multiset of "pushed" values differs from the multiset of
//! "pulled" values, the running accumulator `Σ multiplicity_i / m_i` is a nonzero
//! rational function of the challenges. By Schwartz-Zippel (applied to the
//! numerator after clearing denominators), the accumulator evaluates to zero with
//! probability at most **N / |F_ext|**. Crucially, the challenges are sampled
//! *after* the prover has committed to the stage-1 traces and the claims have been
//! observed, so the prover cannot adapt them.
//!
//! ## Fiat-Shamir (random oracle model)
//!
//! All challenges (α, ζ, β, γ) are derived from the transcript via the
//! configuration's challenger. Security relies on the underlying hash behaving
//! as a random oracle. The ordering of
//! observations is critical: in particular, claims must be observed *before* lookup
//! challenges are sampled, otherwise the prover could choose claims adaptively to
//! make the accumulator balance.
//!
//! ## Overall soundness
//!
//! By a union bound, the total soundness error is at most:
//!
//! ```text
//! ε ≤ ε_FRI + (k - 1 + D + N) / |F_ext|
//! ```
//!
//! where ε_FRI is the FRI soundness error (see above for the conjectured vs proven
//! regimes). With a sufficiently large challenge field (|F_ext| ≈ 2^128) the
//! second term is negligible for any practical parameters, so **FRI dominates**.
//! With `log_blowup = 1` and
//! `num_queries = 100`, the protocol provides approximately 100 bits of conjectured
//! security (≈ 50 bits proven) from FRI alone, plus additional grinding cost from
//! PoW.
//!
//! ## Sparse activation
//!
//! A proof covers only the circuits its activation bitmap marks active
//! ([`Proof::active`]); inactive circuits are neither committed, opened,
//! accumulated, nor constraint-evaluated. An accepted sparse proof is
//! equivalent to a dense proof in which every inactive circuit has an empty
//! table: an inactive circuit contributes no lookup sends or receives, so
//! honest deactivation leaves the global balance unchanged, while
//! dishonestly deactivating a circuit the execution needs leaves an
//! unmatched channel message and the final accumulator is a nonzero
//! rational function of (β, γ) — caught by the balance check with the same
//! N / |F_ext| bound as above. The bitmap cannot be chosen adaptively: it
//! is observed before any commitment or challenge, and the verifier takes
//! every active circuit's widths, constraints, and lookup structure from
//! the canonical system by bitmap index. The one contract change relative
//! to dense proofs: a circuit can no longer force its own presence through
//! must-hold padding rows — sparse verification guarantees the constraints
//! of ACTIVE circuits plus global lookup balance, and nothing about
//! inactive ones.
//!
//! ## Not zero-knowledge
//!
//! This protocol is a succinct argument of knowledge, **not** a zero-knowledge
//! proof: traces are committed without blinding, and FRI query responses reveal
//! actual low-degree-extension values of the witness. Do not use it when the
//! witness must remain hidden from the verifier.

use crate::config::{PcsError, StarkGenericConfig, Val};
use crate::ensure_eq;
use crate::eval::VarValues;
use crate::lookup::fingerprint;
use crate::prover::Proof;
use crate::system::System;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_util::log2_strict_usize;

/// Errors that can occur during proof verification.
#[derive(Debug)]
pub enum VerificationError<PcsErr> {
    /// A provided claim is invalid.
    ///
    /// Note: this variant is not currently returned by any verification path.
    /// It is reserved for future claim validation checks.
    InvalidClaim,
    /// The PCS opening proof failed to verify.
    InvalidOpeningArgument(PcsErr),
    /// The proof has an unexpected shape (wrong number of opened values, etc.).
    InvalidProofShape,
    /// The system configuration is invalid (e.g. no circuits).
    InvalidSystem,
    /// The recomputed composition polynomial does not match the quotient.
    OodEvaluationMismatch,
    /// The lookup accumulator did not balance to zero.
    UnbalancedChannel,
}

impl<SC: StarkGenericConfig> System<SC> {
    /// Verifies a STARK proof against a single claim.
    pub fn verify(
        &self,
        claim: &[Val<SC>],
        proof: &Proof<SC>,
    ) -> Result<(), VerificationError<PcsError<SC>>>
    where
        Val<SC>: TwoAdicField,
    {
        self.verify_multiple_claims(&[claim], proof)
    }

    /// Verifies a STARK proof against multiple claims.
    pub fn verify_multiple_claims(
        &self,
        claims: &[&[Val<SC>]],
        proof: &Proof<SC>,
    ) -> Result<(), VerificationError<PcsError<SC>>>
    where
        Val<SC>: TwoAdicField,
    {
        let Proof {
            active,
            commitments,
            intermediate_accumulators,
            log_degrees,
            opening_proof,
            quotient_opened_values,
            preprocessed_opened_values,
            stage_1_opened_values,
            stage_2_opened_values,
        } = proof;
        // first, verify the proof shape (also validates the activation bitmap)
        let quotient_degrees = self.verify_shape(proof)?;
        // Canonical index of each active circuit; every per-circuit sequence
        // in the proof is indexed by position in this list.
        let active_indices: Vec<usize> = active
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| a.then_some(i))
            .collect();

        // Soundness: lookup argument. The accumulator was computed by the prover
        // under challenges (β, γ) that were sampled after the traces and claims were
        // committed. If the pushed and pulled multisets differ, the accumulator is a
        // nonzero rational function of (β, γ) and evaluates to zero with probability
        // ≤ N / |F_ext| (Schwartz-Zippel on the numerator polynomial).
        ensure_eq!(
            intermediate_accumulators.last(),
            Some(&SC::Challenge::ZERO),
            VerificationError::UnbalancedChannel
        );

        // Soundness: Fiat-Shamir. All challenges below are derived deterministically
        // from the transcript via the configuration's challenger, whose hash is
        // modeled as a random oracle. The verifier replays exactly the same
        // observations as the prover, so any divergence (e.g. different
        // commitments) produces different challenges, making it infeasible for a
        // cheating prover to predict them.
        let pcs = self.config.pcs();
        let mut challenger = self.config.initialise_challenger();

        // Bind the system shape into the transcript. The protocol parameters
        // are already bound via the challenger seed.
        self.observe_shape(&mut challenger);

        // Bind the activation bitmap — before any commitment or challenge, so
        // every sample depends on which circuits this proof covers.
        for &is_active in active {
            challenger.observe(Val::<SC>::from_bool(is_active));
        }

        // observe preprocessed and stage_1 commitment
        if let Some(commit) = &self.preprocessed_commit {
            challenger.observe(commit.clone());
        }
        challenger.observe(commitments.stage_1_trace.clone());

        // Observe trace heights to bind the proof to specific domain sizes.
        for log_degree in log_degrees {
            challenger.observe(Val::<SC>::from_u8(*log_degree));
        }

        // Soundness: claims must be observed BEFORE lookup challenges are sampled.
        // Otherwise, the prover could choose claims adaptively after seeing the
        // challenges, breaking the lookup argument's binding property. The claims
        // are length-prefixed so that distinct claim structures (e.g. [[a, b]]
        // vs [[a], [b]]) yield distinct transcripts.
        challenger.observe(Val::<SC>::from_usize(claims.len()));
        for claim in claims {
            challenger.observe(Val::<SC>::from_usize(claim.len()));
            challenger.observe_slice(claim);
        }

        // Soundness: lookup argument. The challenges are random elements of F_ext.
        // The message m_i = lookup_challenge + fingerprint(fingerprint_challenge, args_i)
        // is an affine function of the challenges, ensuring that distinct argument
        // tuples produce distinct messages with probability ≥ 1 - 1/|F_ext|.
        let lookup_argument_challenge: SC::Challenge = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: SC::Challenge = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);

        // observe stage_2 commitment
        challenger.observe(commitments.stage_2_trace.clone());

        // Observe the intermediate accumulators. They enter the constraints as
        // public values, so later challenges (α, ζ) must depend on them
        // directly rather than only through the quotient commitment.
        for acc in intermediate_accumulators {
            challenger.observe_algebra_element(*acc);
        }

        // construct the accumulator from the claims
        let mut acc = SC::Challenge::ZERO;
        for claim in claims {
            let message = lookup_argument_challenge
                + fingerprint(&fingerprint_challenge, claim.iter().cloned());
            acc += message.inverse();
        }

        // Soundness: constraint folding. All k constraints are combined via powers
        // of α. The folded sum has degree k-1 in α, so by Schwartz-Zippel a violated
        // constraint survives folding with probability ≥ 1 - (k-1)/|F_ext|.
        let constraint_challenge: SC::Challenge = challenger.sample_algebra_element();

        // observe quotient commitment
        challenger.observe(commitments.quotient_chunks.clone());

        // Soundness: OOD evaluation. ζ is sampled after all commitments are fixed.
        // A nonzero polynomial of degree ≤ D vanishes at ζ with probability ≤ D/|F_ext|.
        let zeta: SC::Challenge = challenger.sample_algebra_element();

        // Reconstruct the PCS opening rounds (identical to the prover).
        let mut stage_1_trace_evaluations = vec![];
        let mut stage_2_trace_evaluations = vec![];
        let mut quotient_chunks_evaluations = vec![];
        for pos in 0..active_indices.len() {
            let log_degree = log_degrees[pos];
            let trace_domain = pcs.natural_domain_for_degree(1 << log_degree);
            let zeta_next = trace_domain
                .next_point(zeta)
                .ok_or(VerificationError::InvalidProofShape)?;
            stage_1_trace_evaluations.push((
                trace_domain,
                vec![
                    (zeta, stage_1_opened_values[pos][0].clone()),
                    (zeta_next, stage_1_opened_values[pos][1].clone()),
                ],
            ));
            stage_2_trace_evaluations.push((
                trace_domain,
                vec![
                    (zeta, stage_2_opened_values[pos][0].clone()),
                    (zeta_next, stage_2_opened_values[pos][1].clone()),
                ],
            ));
            // All of a circuit's quotient coefficient slices live in one
            // matrix on the trace domain, opened once at ζ.
            quotient_chunks_evaluations.push((
                trace_domain,
                vec![(zeta, quotient_opened_values[pos][0].clone())],
            ));
        }
        // The preprocessed commitment covers ALL preprocessed matrices, in
        // canonical slot order; inactive circuits' matrices are opened at no
        // points (mirroring the prover's round construction).
        let mut preprocessed_trace_evaluations = vec![];
        {
            let mut active_pos: Vec<Option<usize>> = vec![None; active.len()];
            for (pos, &ci) in active_indices.iter().enumerate() {
                active_pos[ci] = Some(pos);
            }
            for (ci, &slot) in self.preprocessed_indices.iter().enumerate() {
                if let Some(slot) = slot {
                    match active_pos[ci] {
                        Some(pos) => {
                            let trace_domain = pcs.natural_domain_for_degree(1 << log_degrees[pos]);
                            let zeta_next = trace_domain
                                .next_point(zeta)
                                .ok_or(VerificationError::InvalidProofShape)?;
                            let preprocessed_opened_values =
                                preprocessed_opened_values.as_ref().unwrap();
                            preprocessed_trace_evaluations.push((
                                trace_domain,
                                vec![
                                    (zeta, preprocessed_opened_values[slot][0].clone()),
                                    (zeta_next, preprocessed_opened_values[slot][1].clone()),
                                ],
                            ));
                        }
                        None => {
                            let domain = pcs
                                .natural_domain_for_degree(self.circuits[ci].preprocessed_height);
                            preprocessed_trace_evaluations.push((domain, vec![]));
                        }
                    }
                }
            }
        }
        let mut coms_to_verify = vec![
            (commitments.stage_1_trace.clone(), stage_1_trace_evaluations),
            (commitments.stage_2_trace.clone(), stage_2_trace_evaluations),
            (
                commitments.quotient_chunks.clone(),
                quotient_chunks_evaluations,
            ),
        ];
        if let Some(preprocessed_commitment) = &self.preprocessed_commit {
            coms_to_verify.push((
                preprocessed_commitment.clone(),
                preprocessed_trace_evaluations,
            ));
        }
        // Soundness: FRI proximity test. Verifies that the committed polynomials
        // are close to low-degree polynomials and that the claimed evaluations are
        // consistent with the commitments. Soundness error ≤ ρ^num_queries, where
        // ρ = 2^(-log_blowup). This is the dominant term in the overall bound.
        pcs.verify(coms_to_verify, opening_proof, &mut challenger)
            .map_err(VerificationError::InvalidOpeningArgument)?;

        // use the opened values to compute the composition polynomial for each circuit
        // and check that the evaluation of the composition polynomial equals the
        // product of the zerofier with the quotient
        let extension_d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
        let empty: [SC::Challenge; 0] = [];
        for (pos, &ci) in active_indices.iter().enumerate() {
            let circuit = &self.circuits[ci];
            let degree = 1 << log_degrees[pos];
            let quotient_degree = quotient_degrees[pos];
            let next_acc = intermediate_accumulators[pos];
            let trace_domain = pcs.natural_domain_for_degree(degree);
            let sels = trace_domain.selectors_at_point(zeta);
            // The logUp boundary injection absorbs the last-row selector's
            // normalization constant into Δ (mirroring the prover): p3's
            // unnormalized L_last has value n·g at the last row.
            let n_val = Val::<SC>::from_usize(degree);
            let g = Val::<SC>::two_adic_generator(usize::from(log_degrees[pos]));
            let inj_norm = SC::Challenge::from((n_val * g).inverse());

            // The four lookup publics (β, γ, acc, next_acc) as base
            // coordinates embedded into the challenge field.
            let mut publics: Vec<SC::Challenge> = Vec::with_capacity(4 * extension_d);
            for ef in [
                lookup_argument_challenge,
                fingerprint_challenge,
                acc,
                next_acc,
            ] {
                for &coord in ef.as_basis_coefficients_slice() {
                    publics.push(SC::Challenge::from(coord));
                }
            }

            let (preprocessed_cur, preprocessed_next): (&[SC::Challenge], &[SC::Challenge]) =
                if let Some(slot) = self.preprocessed_indices[ci] {
                    let values = preprocessed_opened_values.as_ref().unwrap();
                    (&values[slot][0], &values[slot][1])
                } else {
                    (&empty, &empty)
                };
            let view = VarValues {
                preprocessed: [preprocessed_cur, preprocessed_next],
                main: [
                    &stage_1_opened_values[pos][0],
                    &stage_1_opened_values[pos][1],
                ],
                stage2: [
                    &stage_2_opened_values[pos][0],
                    &stage_2_opened_values[pos][1],
                ],
                publics: &publics,
                is_first_row: sels.is_first_row,
                is_last_row: sels.is_last_row,
                is_transition: sels.is_transition,
            };
            // One sweep serves both the user-constraint roots and the
            // evaluated lookup expressions; the logUp constraint values are
            // then computed directly (never compiled) and folded after the
            // user roots, in the canonical protocol order.
            let mut buf = Vec::new();
            circuit.graph.sweep(&view, &mut buf);
            let mut constraint_values = circuit.graph.constraint_values(&buf);
            let delta_scaled: Vec<SC::Challenge> = (0..extension_d)
                .map(|k| (publics[3 * extension_d + k] - publics[2 * extension_d + k]) * inj_norm)
                .collect();
            crate::lookup::logup_constraint_values(
                &circuit.graph.lookups,
                &buf,
                view.stage2[0],
                view.stage2[1],
                &publics,
                &delta_scaled,
                // p3's (unnormalized) last-row selector; the normalization
                // constant is pre-absorbed into `delta_scaled`.
                view.is_last_row,
                crate::system::extension_params::<SC>().w,
                extension_d,
                circuit.lookup_group_size,
                &mut constraint_values,
            );
            debug_assert_eq!(constraint_values.len(), circuit.constraint_count());
            // Fold with α (Horner): Σ_i α^{k-1-i} · value_i, matching the
            // prover's reversed α-power weighting.
            let composition = constraint_values
                .iter()
                .fold(SC::Challenge::ZERO, |acc, &v| {
                    acc * constraint_challenge + v
                });

            // Recombine the quotient from its coefficient slices:
            // `Q(ζ) = Σᵢ ζ^{i·n}·cᵢ(ζ)`, with each slice's value read from
            // the wide quotient matrix opened at ζ.
            let quotient_row = &quotient_opened_values[pos][0];
            debug_assert_eq!(quotient_row.len(), quotient_degree * extension_d);
            let zeta_pow_n = zeta.exp_power_of_2(log2_strict_usize(degree));
            let quotient = quotient_row
                .chunks_exact(extension_d)
                .zip(zeta_pow_n.powers())
                .map(|(chunk, zeta_pow)| zeta_pow * from_ext_basis::<Val<SC>, SC::Challenge>(chunk))
                .sum::<SC::Challenge>();

            // Soundness: OOD check. If any constraint is violated on the trace
            // domain, the composition polynomial is not divisible by the vanishing
            // polynomial, so their ratio differs from the committed quotient. At
            // the random point ζ, this difference is nonzero with probability
            // ≥ 1 - D/|F_ext| (Schwartz-Zippel). Combined with the FRI check above,
            // this ensures that the opened values are consistent with actually
            // low-degree polynomials satisfying all constraints.
            ensure_eq!(
                composition * sels.inv_vanishing,
                quotient,
                VerificationError::OodEvaluationMismatch
            );
            // the accumulator must become the next accumulator for the next iteration
            acc = next_acc;
        }
        Ok(())
    }

    /// Validates the structural shape of the proof without checking any cryptographic
    /// properties. Returns the quotient degrees per active circuit on success.
    pub fn verify_shape(
        &self,
        proof: &Proof<SC>,
    ) -> Result<Vec<usize>, VerificationError<PcsError<SC>>> {
        let Proof {
            active,
            intermediate_accumulators,
            log_degrees,
            quotient_opened_values,
            preprocessed_opened_values,
            stage_1_opened_values,
            stage_2_opened_values,
            ..
        } = proof;
        let num_circuits = self.circuits.len();
        crate::ensure!(num_circuits > 0, VerificationError::InvalidSystem);
        ensure_eq!(
            active.len(),
            num_circuits,
            VerificationError::InvalidProofShape
        );
        let active_indices: Vec<usize> = active
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| a.then_some(i))
            .collect();
        let num_active = active_indices.len();
        crate::ensure!(num_active > 0, VerificationError::InvalidProofShape);
        ensure_eq!(
            log_degrees.len(),
            num_active,
            VerificationError::InvalidProofShape
        );
        let num_preprocessed = self
            .preprocessed_indices
            .iter()
            .map(|i| usize::from(i.is_some()))
            .sum::<usize>();
        ensure_eq!(
            self.preprocessed_commit.is_none(),
            num_preprocessed == 0,
            VerificationError::InvalidSystem
        );
        // Stage 0 round: the preprocessed commitment is built once over ALL
        // preprocessed traces at system construction, so its round carries one
        // entry per preprocessed matrix regardless of activation; an inactive
        // circuit's matrix must be opened at no points.
        ensure_eq!(
            preprocessed_opened_values
                .as_ref()
                .map_or(0, |values| values.len()),
            num_preprocessed,
            VerificationError::InvalidProofShape
        );
        for (&prep_index, &is_active) in self.preprocessed_indices.iter().zip(active) {
            if let Some(slot) = prep_index
                && !is_active
            {
                ensure_eq!(
                    preprocessed_opened_values.as_ref().unwrap()[slot].len(),
                    0,
                    VerificationError::InvalidProofShape
                );
            }
        }
        ensure_eq!(
            stage_1_opened_values.len(),
            num_active,
            VerificationError::InvalidProofShape
        );
        ensure_eq!(
            stage_2_opened_values.len(),
            num_active,
            VerificationError::InvalidProofShape
        );
        for (pos, &ci) in active_indices.iter().enumerate() {
            let circuit = &self.circuits[ci];
            let preprocessed_i = self.preprocessed_indices[ci];
            // zeta and zeta_next
            let num_openings = 2;
            ensure_eq!(
                stage_1_opened_values[pos].len(),
                num_openings,
                VerificationError::InvalidProofShape
            );
            ensure_eq!(
                stage_2_opened_values[pos].len(),
                num_openings,
                VerificationError::InvalidProofShape
            );
            if let Some(slot) = preprocessed_i {
                ensure_eq!(
                    preprocessed_opened_values.as_ref().unwrap()[slot].len(),
                    num_openings,
                    VerificationError::InvalidProofShape
                );
            }
            for j in 0..num_openings {
                if let Some(slot) = preprocessed_i {
                    ensure_eq!(
                        preprocessed_opened_values.as_ref().unwrap()[slot][j].len(),
                        circuit.preprocessed_width,
                        VerificationError::InvalidProofShape
                    );
                }
                ensure_eq!(
                    stage_1_opened_values[pos][j].len(),
                    circuit.main_width,
                    VerificationError::InvalidProofShape
                );
                // Stage-2 is committed as flattened base columns, so the
                // opened width is already the flattened width.
                ensure_eq!(
                    stage_2_opened_values[pos][j].len(),
                    circuit.stage_2_width,
                    VerificationError::InvalidProofShape
                );
            }
        }
        let mut quotient_degrees = vec![];
        for (&ci, log_degree) in active_indices.iter().zip(log_degrees) {
            let quotient_degree = self.circuits[ci].quotient_degree();
            // The claimed log degree must be small enough that the quotient
            // domain can still be committed and opened by the PCS. This also
            // guards the `1 << log_degree` shifts used during verification
            // against overflow on adversarial proofs.
            crate::ensure!(
                usize::from(*log_degree) + log2_strict_usize(quotient_degree)
                    <= self.config.max_log_degree(),
                VerificationError::InvalidProofShape
            );
            quotient_degrees.push(quotient_degree);
        }
        // One wide quotient matrix per active circuit, holding all its
        // coefficient slices, opened at the single point ζ.
        ensure_eq!(
            quotient_opened_values.len(),
            num_active,
            VerificationError::InvalidProofShape
        );
        let extension_d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
        for (pos, quotient_degree) in quotient_degrees.iter().enumerate() {
            ensure_eq!(
                quotient_opened_values[pos].len(),
                1,
                VerificationError::InvalidProofShape
            );
            ensure_eq!(
                quotient_opened_values[pos][0].len(),
                quotient_degree * extension_d,
                VerificationError::InvalidProofShape
            );
        }
        ensure_eq!(
            intermediate_accumulators.len(),
            num_active,
            VerificationError::InvalidProofShape
        );
        Ok(quotient_degrees)
    }
}

/// Reassembles an extension element from its base coordinates.
fn from_ext_basis<F: Field, EF: ExtensionField<F>>(coeffs: &[EF]) -> EF {
    coeffs
        .iter()
        .enumerate()
        .map(|(i, c)| *c * <EF as BasedVectorSpace<F>>::ith_basis_element(i).unwrap())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        p3_adapter::LookupAir,
        prover::Proof,
        system::{ProverKey, SystemWitness},
        types::{CommitmentParameters, ExtVal, FriParameters, GoldilocksBlake3Config, Val},
    };
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_matrix::dense::RowMajorMatrix;

    enum CS {
        Pythagorean,
        Complex,
    }
    impl<F> BaseAir<F> for CS {
        fn width(&self) -> usize {
            match self {
                Self::Pythagorean => 3,
                Self::Complex => 6,
            }
        }
    }
    impl<AB> Air<AB> for CS
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                Self::Pythagorean => {
                    let main = builder.main();
                    let local = main.current_slice();
                    let expr1 = local[0] * local[0] + local[1] * local[1];
                    let expr2 = local[2] * local[2];
                    // this extra `local[0]` multiplication is there to increase the maximum constraint degree
                    builder.assert_eq(local[0] * expr1, local[0] * expr2);
                }
                Self::Complex => {
                    let main = builder.main();
                    let local = main.current_slice();
                    // (a + ib)(c + id) = (ac - bd) + i(ad + bc)
                    let expr1 = local[0] * local[2] - local[1] * local[3];
                    let expr2 = local[4];
                    let expr3 = local[0] * local[3] + local[1] * local[2];
                    let expr4 = local[5];
                    builder.assert_eq(expr1, expr2);
                    builder.assert_eq(expr3, expr4);
                }
            }
        }
    }
    const COMMITMENT_PARAMETERS: CommitmentParameters = CommitmentParameters {
        log_blowup: 1,
        cap_height: 0,
    };

    const FRI_PARAMETERS: FriParameters = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 64,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };

    fn system() -> (
        System<GoldilocksBlake3Config>,
        ProverKey<GoldilocksBlake3Config>,
    ) {
        let config = GoldilocksBlake3Config::new(COMMITMENT_PARAMETERS, FRI_PARAMETERS);
        let pythagorean_circuit = LookupAir::new(CS::Pythagorean, vec![]);
        let complex_circuit = LookupAir::new(CS::Complex, vec![]);
        System::new(config, [pythagorean_circuit, complex_circuit])
    }

    #[test]
    fn multi_stark_test() {
        let (system, key) = system();
        let f = Val::from_u32;
        let witness = SystemWitness::from_stage_1(
            vec![
                RowMajorMatrix::new(
                    [3, 4, 5, 5, 12, 13, 8, 15, 17, 7, 24, 25].map(f).to_vec(),
                    3,
                ),
                RowMajorMatrix::new([4, 2, 3, 1, 10, 10, 3, 2, 5, 1, 13, 13].map(f).to_vec(), 6),
            ],
            &system,
        );
        let no_claims = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness);
        system.verify_multiple_claims(no_claims, &proof).unwrap();
    }

    #[test]
    fn multi_stark_prove_verify_serialize() {
        let (system, key) = system();
        let f = Val::from_u32;
        // 2^4 = 16 rows — small enough for fast CI
        let mut pythagorean_trace = [3, 4, 5].map(f).to_vec();
        let mut complex_trace = [4, 2, 3, 1, 10, 10].map(f).to_vec();
        for _ in 0..4 {
            pythagorean_trace.extend(pythagorean_trace.clone());
            complex_trace.extend(complex_trace.clone());
        }
        let witness = SystemWitness::from_stage_1(
            vec![
                RowMajorMatrix::new(pythagorean_trace, 3),
                RowMajorMatrix::new(complex_trace, 6),
            ],
            &system,
        );
        let no_claims = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness);
        // Serialization round-trip
        let proof_bytes = proof.to_bytes().expect("Failed to serialize proof");
        let proof2 = Proof::from_bytes(&proof_bytes).expect("Failed to deserialize proof");
        system.verify_multiple_claims(no_claims, &proof2).unwrap();
    }

    // -- Negative / adversarial tests --

    /// Helper: creates a small system and valid proof for negative tests.
    fn small_system_and_proof() -> (
        System<GoldilocksBlake3Config>,
        Proof<GoldilocksBlake3Config>,
    ) {
        let (system, key) = system();
        let f = Val::from_u32;
        let witness = SystemWitness::from_stage_1(
            vec![
                RowMajorMatrix::new(
                    [3, 4, 5, 5, 12, 13, 8, 15, 17, 7, 24, 25].map(f).to_vec(),
                    3,
                ),
                RowMajorMatrix::new([4, 2, 3, 1, 10, 10, 3, 2, 5, 1, 13, 13].map(f).to_vec(), 6),
            ],
            &system,
        );
        let no_claims = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness);
        (system, proof)
    }

    #[test]
    fn test_wrong_claim_rejected() {
        let (system, proof) = small_system_and_proof();
        let f = Val::from_u32;
        // Verify with a bogus claim — the prover used no claims, so any claim should fail.
        let result = system.verify(&[f(42)], &proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_stage_1_values_rejected() {
        let (system, mut proof) = small_system_and_proof();
        // Mutate a value in the stage 1 opened values — FRI should catch this.
        proof.stage_1_opened_values[0][0][0] += ExtVal::ONE;
        let no_claims: &[&[Val]] = &[];
        let result = system.verify_multiple_claims(no_claims, &proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_accumulator_rejected() {
        let (system, mut proof) = small_system_and_proof();
        // Set the last intermediate accumulator to non-zero.
        let last = proof.intermediate_accumulators.len() - 1;
        proof.intermediate_accumulators[last] = ExtVal::ONE;
        let no_claims: &[&[Val]] = &[];
        let result = system.verify_multiple_claims(no_claims, &proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_log_degrees_rejected() {
        let (system, mut proof) = small_system_and_proof();
        // Remove a log degree — the shape check must reject this instead of
        // the verifier panicking on an out-of-bounds index.
        proof.log_degrees.pop();
        let no_claims: &[&[Val]] = &[];
        let result = system.verify_multiple_claims(no_claims, &proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_oversized_log_degree_rejected() {
        let (system, mut proof) = small_system_and_proof();
        // A log degree beyond the field's two-adicity must be rejected instead
        // of causing a shift overflow or a panic inside the PCS.
        proof.log_degrees[0] = 200;
        let no_claims: &[&[Val]] = &[];
        let result = system.verify_multiple_claims(no_claims, &proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_proof_rejected() {
        let (system, mut proof) = small_system_and_proof();
        // Remove a quotient opened value — shape check should fail.
        proof.quotient_opened_values.pop();
        let no_claims: &[&[Val]] = &[];
        let result = system.verify_multiple_claims(no_claims, &proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization_round_trip() {
        let (system, proof) = small_system_and_proof();
        let bytes = proof.to_bytes().expect("serialize");
        let proof2 = Proof::from_bytes(&bytes).expect("deserialize");
        let no_claims: &[&[Val]] = &[];
        system.verify_multiple_claims(no_claims, &proof2).unwrap();
    }
}
