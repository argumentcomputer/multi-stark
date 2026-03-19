//! Multi-circuit STARK verifier.
//!
//! # Verification steps
//!
//! 1. **Shape check** ([`System::verify_shape`]): Validate that the proof's array
//!    dimensions (opened values, accumulators, quotient chunks) match the system's
//!    circuit count and column widths.
//!
//! 2. **Accumulator balance**: Assert that the last intermediate accumulator is zero,
//!    ensuring that all lookup pushes and pulls cancel out across circuits.
//!
//! 3. **Fiat-Shamir replay**: Reconstruct the challenger state identically to the
//!    prover by observing commitments, trace heights, claims, and sampling the same
//!    challenges (lookup, fingerprint, constraint alpha, OOD zeta).
//!
//! 4. **PCS verification**: Verify the FRI opening proofs against the committed
//!    polynomials at the sampled points.
//!
//! 5. **OOD evaluation**: For each circuit, recompute the composition polynomial at
//!    zeta from the opened values and verify that
//!    `composition(zeta) * inv_vanishing(zeta) == quotient(zeta)`.
//!
//! See [`VerificationError`] for the possible failure modes.
//!
//! # Soundness argument
//!
//! The protocol is sound in the random oracle model (instantiated by Keccak-256 via
//! the Fiat-Shamir challenger). Informally: if a prover produces a proof that the
//! verifier accepts, then with overwhelming probability the claimed computation is
//! correct.
//!
//! We use the following notation throughout:
//! - |F_ext| ≈ 2^128 — size of the extension field (GoldilocksBinomialExtension<2>)
//! - ρ = 2^(-log_blowup) — FRI rate parameter (inverse of the blowup factor)
//! - n — number of FRI queries (`num_queries`)
//! - k — number of AIR constraints (after lookup expansion)
//! - N — total number of lookup rows across all circuits
//! - D — maximum degree of the quotient polynomial (trace_degree × quotient_degree)
//!
//! ## FRI proximity test
//!
//! The FRI-based PCS guarantees that committed polynomials are close to polynomials
//! of the claimed degree. The exact soundness bound depends on the proximity
//! parameter, number of folding rounds, and folding arity; a commonly cited
//! approximation is **ρ^n** (each of the n queries independently catches a cheating
//! prover with probability ≈ 1 - ρ). With `log_blowup = 1` and `num_queries = 100`,
//! this gives ≈ 2^(-100). For the precise bound, see the FRI soundness analysis in
//! the Plonky3 documentation.
//!
//! The proof-of-work (PoW) phases add grinding cost: a cheating prover must perform
//! 2^(commit_proof_of_work_bits) work per batching challenge and
//! 2^(query_proof_of_work_bits) work before query sampling. This increases the
//! concrete cost of attack without affecting honest verification time.
//!
//! ## Constraint folding (α)
//!
//! All k AIR constraints are folded into a single composition polynomial using
//! powers of a random challenge α. The folded polynomial Σ α^i · C_i(x) has degree
//! k - 1 in α. By the Schwartz-Zippel lemma, if any individual constraint C_i is
//! nonzero, the folded sum is nonzero with probability at least
//! **1 - (k - 1) / |F_ext|**, which is negligible for practical constraint counts
//! since |F_ext| ≈ 2^128.
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
//! All challenges (α, ζ, β, γ) are derived from the transcript via Keccak-256.
//! Security relies on Keccak-256 behaving as a random oracle. The ordering of
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
//! where ε_FRI ≈ ρ^n is the FRI soundness error. The second term is negligible
//! for any practical parameters since |F_ext| ≈ 2^128, so **FRI dominates**. With
//! `log_blowup = 1` and `num_queries = 100`, the protocol provides approximately
//! 100 bits of security from FRI alone, plus additional grinding cost from PoW.

use crate::{
    builder::folder::VerifierConstraintFolder,
    ensure, ensure_eq,
    lookup::fingerprint,
    prover::Proof,
    system::System,
    types::{Challenger, ExtVal, FriParameters, Pcs, PcsError, StarkConfig, Val},
};
use p3_air::{Air, BaseAir, RowWindow};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs as PcsTrait, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};
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

impl<A: BaseAir<Val> + for<'a> Air<VerifierConstraintFolder<'a>>> System<A> {
    /// Verifies a STARK proof against a single claim.
    ///
    /// Returns `Ok(())` if the proof is valid, or a [`VerificationError`] describing
    /// the first check that failed.
    pub fn verify(
        &self,
        fri_parameters: FriParameters,
        claim: &[Val],
        proof: &Proof,
    ) -> Result<(), VerificationError<PcsError>> {
        self.verify_multiple_claims(fri_parameters, &[claim], proof)
    }

    /// Verifies a STARK proof against multiple claims.
    pub fn verify_multiple_claims(
        &self,
        fri_parameters: FriParameters,
        claims: &[&[Val]],
        proof: &Proof,
    ) -> Result<(), VerificationError<PcsError>> {
        let Proof {
            commitments,
            intermediate_accumulators,
            log_degrees,
            opening_proof,
            quotient_opened_values,
            preprocessed_opened_values,
            stage_1_opened_values,
            stage_2_opened_values,
        } = proof;
        // first, verify the proof shape
        let quotient_degrees = self.verify_shape(proof)?;

        // Soundness: lookup argument. The accumulator was computed by the prover
        // under challenges (β, γ) that were sampled after the traces and claims were
        // committed. If the pushed and pulled multisets differ, the accumulator is a
        // nonzero rational function of (β, γ) and evaluates to zero with probability
        // ≤ N / |F_ext| (Schwartz-Zippel on the numerator polynomial).
        ensure_eq!(
            intermediate_accumulators.last(),
            Some(&ExtVal::ZERO),
            VerificationError::UnbalancedChannel
        );

        // Soundness: Fiat-Shamir. All challenges below are derived deterministically
        // from the transcript via Keccak-256 (random oracle model). The verifier
        // replays exactly the same observations as the prover, so any divergence
        // (e.g. different commitments) produces different challenges, making it
        // infeasible for a cheating prover to predict them.
        let config = StarkConfig::new(self.commitment_parameters, fri_parameters);
        let pcs = config.pcs();
        let mut challenger = config.initialise_challenger();

        // observe preprocessed and stage_1 commitment
        if let Some(commit) = &self.preprocessed_commit {
            challenger.observe(commit);
        }
        challenger.observe(commitments.stage_1_trace.clone());

        // Observe trace heights to bind the proof to specific domain sizes.
        for log_degree in log_degrees {
            challenger.observe(Val::from_u8(*log_degree));
        }

        // Soundness: claims must be observed BEFORE lookup challenges are sampled.
        // Otherwise, the prover could choose claims adaptively after seeing the
        // challenges, breaking the lookup argument's binding property.
        for claim in claims {
            challenger.observe_slice(claim);
        }

        // Soundness: lookup argument. The challenges are random elements of F_ext.
        // The message m_i = lookup_challenge + fingerprint(fingerprint_challenge, args_i)
        // is an affine function of the challenges, ensuring that distinct argument
        // tuples produce distinct messages with probability ≥ 1 - 1/|F_ext|.
        let lookup_argument_challenge: ExtVal = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: ExtVal = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);

        // observe stage_2 commitment
        challenger.observe(commitments.stage_2_trace.clone());

        // construct the accumulator from the claims
        let mut acc = ExtVal::ZERO;
        for claim in claims {
            let message = lookup_argument_challenge
                + fingerprint(&fingerprint_challenge, claim.iter().cloned());
            acc += message.inverse();
        }

        // Soundness: constraint folding. All k constraints are combined via powers
        // of α. The folded sum has degree k-1 in α, so by Schwartz-Zippel a violated
        // constraint survives folding with probability ≥ 1 - (k-1)/|F_ext|.
        let constraint_challenge: ExtVal = challenger.sample_algebra_element();

        // observe quotient commitment
        challenger.observe(commitments.quotient_chunks.clone());

        // Soundness: OOD evaluation. ζ is sampled after all commitments are fixed.
        // A nonzero polynomial of degree ≤ D vanishes at ζ with probability ≤ D/|F_ext|.
        let zeta: ExtVal = challenger.sample_algebra_element();
        let mut preprocessed_trace_evaluations = vec![];
        let mut stage_1_trace_evaluations = vec![];
        let mut stage_2_trace_evaluations = vec![];
        let mut quotient_chunks_evaluations = vec![];
        let mut last_quotient_i = 0;
        for i in 0..self.circuits.len() {
            let log_degree = log_degrees[i];
            let quotient_degree = quotient_degrees[i];
            let log_quotient_degree = log2_strict_usize(quotient_degree);
            let trace_domain = <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                pcs,
                1 << log_degree,
            );
            let quotient_domain =
                trace_domain.create_disjoint_domain((1 << log_degree) << log_quotient_degree);
            let quotient_chunks_domains = quotient_domain.split_domains(quotient_degree);
            let unshifted_quotient_chunks_domains = quotient_chunks_domains
                .iter()
                .map(|domain| {
                    <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                        pcs,
                        domain.size(),
                    )
                })
                .collect::<Vec<_>>();
            let zeta_next = zeta * trace_domain.subgroup_generator();
            if let Some(i) = self.preprocessed_indices[i] {
                let preprocessed_opened_values = preprocessed_opened_values.as_ref().unwrap();
                preprocessed_trace_evaluations.push((
                    trace_domain,
                    vec![
                        (zeta, preprocessed_opened_values[i][0].clone()),
                        (zeta_next, preprocessed_opened_values[i][1].clone()),
                    ],
                ));
            }
            stage_1_trace_evaluations.push((
                trace_domain,
                vec![
                    (zeta, stage_1_opened_values[i][0].clone()),
                    (zeta_next, stage_1_opened_values[i][1].clone()),
                ],
            ));
            stage_2_trace_evaluations.push((
                trace_domain,
                vec![
                    (zeta, stage_2_opened_values[i][0].clone()),
                    (zeta_next, stage_2_opened_values[i][1].clone()),
                ],
            ));
            let iter = unshifted_quotient_chunks_domains
                .into_iter()
                .zip(
                    quotient_opened_values[last_quotient_i..last_quotient_i + quotient_degree]
                        .iter(),
                )
                .map(|(domain, opened_values)| (domain, vec![(zeta, opened_values[0].clone())]));
            quotient_chunks_evaluations.extend(iter);
            last_quotient_i += quotient_degree;
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
            coms_to_verify.extend([(
                preprocessed_commitment.clone(),
                preprocessed_trace_evaluations,
            )])
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
        let mut last_quotient_i = 0;
        for i in 0..self.circuits.len() {
            let circuit = &self.circuits[i];
            let degree = 1 << log_degrees[i];
            let quotient_degree = quotient_degrees[i];
            let next_acc = intermediate_accumulators[i];
            let stage_1_row = &stage_1_opened_values[i][0];
            let stage_1_next_row = &stage_1_opened_values[i][1];
            let stage_2_row = &stage_2_opened_values[i][0];
            let stage_2_next_row = &stage_2_opened_values[i][1];
            let quotient_chunks = quotient_opened_values
                [last_quotient_i..last_quotient_i + quotient_degree]
                .iter()
                .map(|values| &values[0]);
            last_quotient_i += quotient_degree;

            // compute the composition polynomial evaluation
            let trace_domain =
                <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);
            let sels = trace_domain.selectors_at_point(zeta);
            let preprocessed = if let Some(i) = self.preprocessed_indices[i] {
                let preprocessed_opened_values = preprocessed_opened_values.as_ref().unwrap();
                let preprocessed_row = &preprocessed_opened_values[i][0];
                let preprocessed_next_row = &preprocessed_opened_values[i][1];
                RowWindow::from_two_rows(preprocessed_row, preprocessed_next_row)
            } else {
                RowWindow::from_two_rows(&[], &[])
            };
            let stage_1 = VerticalPair::new(
                RowMajorMatrixView::new_row(stage_1_row),
                RowMajorMatrixView::new_row(stage_1_next_row),
            );
            let extension_d = <ExtVal as BasedVectorSpace<Val>>::DIMENSION;
            let stage_2_row = &stage_2_row
                .chunks_exact(extension_d)
                .map(from_ext_basis)
                .collect::<Vec<_>>();
            let stage_2_next_row = &stage_2_next_row
                .chunks_exact(extension_d)
                .map(from_ext_basis)
                .collect::<Vec<_>>();
            let stage_2 = VerticalPair::new(
                RowMajorMatrixView::new_row(stage_2_row),
                RowMajorMatrixView::new_row(stage_2_next_row),
            );
            let stage_1_public_values = &[];
            let stage_2_public_values = &[
                lookup_argument_challenge,
                fingerprint_challenge,
                acc,
                next_acc,
            ];
            let mut folder = VerifierConstraintFolder {
                preprocessed,
                stage_1,
                stage_2,
                stage_1_public_values,
                stage_2_public_values,
                is_first_row: sels.is_first_row,
                is_last_row: sels.is_last_row,
                is_transition: sels.is_transition,
                alpha: constraint_challenge,
                accumulator: ExtVal::ZERO,
            };
            circuit.air.eval(&mut folder);
            let composition_polynomial = folder.accumulator;
            // compute the quotient evaluation
            let quotient_domain = trace_domain.create_disjoint_domain(degree * quotient_degree);
            let quotient_chunks_domains = quotient_domain.split_domains(quotient_degree);
            let zps = quotient_chunks_domains
                .iter()
                .enumerate()
                .map(|(i, domain)| {
                    quotient_chunks_domains
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, other_domain)| {
                            other_domain.vanishing_poly_at_point(zeta)
                                * other_domain
                                    .vanishing_poly_at_point(domain.first_point())
                                    .inverse()
                        })
                        .product::<ExtVal>()
                })
                .collect::<Vec<_>>();
            let quotient = quotient_chunks
                .enumerate()
                .map(|(ch_i, ch)| zps[ch_i] * from_ext_basis(ch))
                .sum::<ExtVal>();

            // Soundness: OOD check. If any constraint is violated on the trace
            // domain, the composition polynomial is not divisible by the vanishing
            // polynomial, so their ratio differs from the committed quotient. At
            // the random point ζ, this difference is nonzero with probability
            // ≥ 1 - D/|F_ext| (Schwartz-Zippel). Combined with the FRI check above,
            // this ensures that the opened values are consistent with actually
            // low-degree polynomials satisfying all constraints.
            ensure_eq!(
                composition_polynomial * sels.inv_vanishing,
                quotient,
                VerificationError::OodEvaluationMismatch
            );
            // the accumulator must become the next accumulator for the next iteration
            acc = next_acc;
        }

        Ok(())
    }

    /// Validates the structural shape of the proof without checking any cryptographic
    /// properties. Returns the quotient degrees per circuit on success.
    pub fn verify_shape(&self, proof: &Proof) -> Result<Vec<usize>, VerificationError<PcsError>> {
        let Proof {
            intermediate_accumulators,
            quotient_opened_values,
            preprocessed_opened_values,
            stage_1_opened_values,
            stage_2_opened_values,
            ..
        } = proof;
        // The following are proof shape checks
        let num_circuits = self.circuits.len();
        // there must be at least one circuit
        ensure!(num_circuits > 0, VerificationError::InvalidSystem);
        // the preprocessed commitment is empty if and only if there are zero preprocessed circuits
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
        // stage 0 round
        ensure_eq!(
            preprocessed_opened_values
                .as_ref()
                .map_or(0, |values| values.len()),
            num_preprocessed,
            VerificationError::InvalidProofShape
        );
        // stage 1 round
        ensure_eq!(
            stage_1_opened_values.len(),
            num_circuits,
            VerificationError::InvalidProofShape
        );
        // stage 2 round
        ensure_eq!(
            stage_2_opened_values.len(),
            num_circuits,
            VerificationError::InvalidProofShape
        );
        for (i, circuit) in self.circuits.iter().enumerate() {
            let preprocessed_i = self.preprocessed_indices[i];
            // zeta and zeta_next
            let num_openings = 2;
            ensure_eq!(
                stage_1_opened_values[i].len(),
                num_openings,
                VerificationError::InvalidProofShape
            );
            ensure_eq!(
                stage_2_opened_values[i].len(),
                num_openings,
                VerificationError::InvalidProofShape
            );
            for j in 0..num_openings {
                if let Some(i) = preprocessed_i {
                    ensure_eq!(
                        preprocessed_opened_values.as_ref().unwrap()[i][j].len(),
                        circuit.preprocessed_width,
                        VerificationError::InvalidProofShape
                    );
                }
                ensure_eq!(
                    stage_1_opened_values[i][j].len(),
                    circuit.stage_1_width,
                    VerificationError::InvalidProofShape
                );
                let extension_d = <ExtVal as BasedVectorSpace<Val>>::DIMENSION;
                ensure_eq!(
                    stage_2_opened_values[i][j].len(),
                    circuit.stage_2_width * extension_d,
                    VerificationError::InvalidProofShape
                );
            }
        }
        // quotient round
        let mut quotient_degrees = vec![];
        for circuit in self.circuits.iter() {
            let quotient_degree = (circuit.max_constraint_degree.max(2) - 1).next_power_of_two();
            quotient_degrees.push(quotient_degree);
        }
        let quotient_size: usize = quotient_degrees.iter().sum();
        ensure_eq!(
            quotient_opened_values.len(),
            quotient_size,
            VerificationError::InvalidProofShape
        );
        #[allow(clippy::needless_range_loop)]
        for i in 0..quotient_size {
            // zeta
            let num_openings = 1;
            ensure_eq!(
                quotient_opened_values[i].len(),
                num_openings,
                VerificationError::InvalidProofShape
            );
            ensure_eq!(
                quotient_opened_values[i][0].len(),
                <ExtVal as BasedVectorSpace<Val>>::DIMENSION,
                VerificationError::InvalidProofShape
            );
        }
        // there must be as many intermediate accumulators as circuits
        ensure_eq!(
            intermediate_accumulators.len(),
            self.circuits.len(),
            VerificationError::InvalidProofShape
        );
        Ok(quotient_degrees)
    }
}

fn from_ext_basis(coeffs: &[ExtVal]) -> ExtVal {
    coeffs
        .iter()
        .enumerate()
        .map(|(i, c)| *c * <ExtVal as BasedVectorSpace<Val>>::ith_basis_element(i).unwrap())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lookup::LookupAir,
        prover::Proof,
        system::{ProverKey, SystemWitness},
        types::{CommitmentParameters, FriParameters},
    };
    use p3_air::{AirBuilder, BaseAir, WindowAccess};
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
    fn system(commitment_parameters: CommitmentParameters) -> (System<CS>, ProverKey) {
        let pythagorean_circuit = LookupAir::new(CS::Pythagorean, vec![]);
        let complex_circuit = LookupAir::new(CS::Complex, vec![]);
        System::new(
            commitment_parameters,
            [pythagorean_circuit, complex_circuit],
        )
    }

    #[test]
    fn multi_stark_test() {
        let commitment_parameters = CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        };
        let (system, key) = system(commitment_parameters);
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
        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 64,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        };
        let no_claims = &[];
        let proof = system.prove_multiple_claims(fri_parameters, &key, no_claims, witness);
        system
            .verify_multiple_claims(fri_parameters, no_claims, &proof)
            .unwrap();
    }

    #[test]
    fn multi_stark_prove_verify_serialize() {
        let commitment_parameters = CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        };
        let (system, key) = system(commitment_parameters);
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
        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 64,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        };
        let no_claims = &[];
        let proof = system.prove_multiple_claims(fri_parameters, &key, no_claims, witness);
        // Serialization round-trip
        let proof_bytes = proof.to_bytes().expect("Failed to serialize proof");
        let proof2 = Proof::from_bytes(&proof_bytes).expect("Failed to deserialize proof");
        system
            .verify_multiple_claims(fri_parameters, no_claims, &proof2)
            .unwrap();
    }

    // -- Negative / adversarial tests --

    /// Helper: creates a small system and valid proof for negative tests.
    fn small_system_and_proof() -> (System<CS>, FriParameters, Proof) {
        let commitment_parameters = CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        };
        let (system, key) = system(commitment_parameters);
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
        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 64,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        };
        let no_claims = &[];
        let proof = system.prove_multiple_claims(fri_parameters, &key, no_claims, witness);
        (system, fri_parameters, proof)
    }

    #[test]
    fn test_wrong_claim_rejected() {
        let (system, fri_parameters, proof) = small_system_and_proof();
        let f = Val::from_u32;
        // Verify with a bogus claim — the prover used no claims, so any claim should fail.
        let result = system.verify(fri_parameters, &[f(42)], &proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_stage_1_values_rejected() {
        let (system, fri_parameters, mut proof) = small_system_and_proof();
        // Mutate a value in the stage 1 opened values — FRI should catch this.
        proof.stage_1_opened_values[0][0][0] += ExtVal::ONE;
        let no_claims: &[&[Val]] = &[];
        let result = system.verify_multiple_claims(fri_parameters, no_claims, &proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_accumulator_rejected() {
        let (system, fri_parameters, mut proof) = small_system_and_proof();
        // Set the last intermediate accumulator to non-zero.
        let last = proof.intermediate_accumulators.len() - 1;
        proof.intermediate_accumulators[last] = ExtVal::ONE;
        let no_claims: &[&[Val]] = &[];
        let result = system.verify_multiple_claims(fri_parameters, no_claims, &proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_proof_rejected() {
        let (system, fri_parameters, mut proof) = small_system_and_proof();
        // Remove a quotient opened value — shape check should fail.
        proof.quotient_opened_values.pop();
        let no_claims: &[&[Val]] = &[];
        let result = system.verify_multiple_claims(fri_parameters, no_claims, &proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization_round_trip() {
        let (system, fri_parameters, proof) = small_system_and_proof();
        let bytes = proof.to_bytes().expect("serialize");
        let proof2 = Proof::from_bytes(&bytes).expect("deserialize");
        let no_claims: &[&[Val]] = &[];
        system
            .verify_multiple_claims(fri_parameters, no_claims, &proof2)
            .unwrap();
    }
}
