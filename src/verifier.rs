//! Multi-circuit STARK verifier, constraint-IR version.
//!
//! Parallel to [`crate::verifier`]. The transcript and PCS-verification are
//! identical; the out-of-domain check recomputes each circuit's composition
//! polynomial by running the compiled circuit's dense sweep in the challenge
//! field at ζ, then folding the constraint values with the constraint
//! challenge. Opened stage-2 columns are fed as base columns directly (no
//! `from_ext_basis` reassembly): the coordinate-expanded constraints are
//! base-field polynomial identities, so evaluating them at the ζ-openings is
//! correct.

use crate::config::{PcsError, StarkGenericConfig, Val};
use crate::ensure_eq;
use crate::eval::VarValues;
use crate::lookup::fingerprint;
use crate::prover::Proof;
use crate::system::System;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing};
use p3_util::log2_strict_usize;

/// Errors that can occur during proof verification.
#[derive(Debug)]
pub enum VerificationError<PcsErr> {
    /// A provided claim is invalid. Reserved for future claim validation.
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
    ) -> Result<(), VerificationError<PcsError<SC>>> {
        self.verify_multiple_claims(&[claim], proof)
    }

    /// Verifies a STARK proof against multiple claims.
    pub fn verify_multiple_claims(
        &self,
        claims: &[&[Val<SC>]],
        proof: &Proof<SC>,
    ) -> Result<(), VerificationError<PcsError<SC>>> {
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
        let quotient_degrees = self.verify_shape(proof)?;
        let active_indices: Vec<usize> = active
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| a.then_some(i))
            .collect();

        // Lookup argument must balance to zero overall.
        ensure_eq!(
            intermediate_accumulators.last(),
            Some(&SC::Challenge::ZERO),
            VerificationError::UnbalancedChannel
        );

        let pcs = self.config.pcs();
        let mut challenger = self.config.initialise_challenger();

        self.observe_shape(&mut challenger);
        for &is_active in active {
            challenger.observe(Val::<SC>::from_bool(is_active));
        }
        if let Some(commit) = &self.preprocessed_commit {
            challenger.observe(commit.clone());
        }
        challenger.observe(commitments.stage_1_trace.clone());
        for log_degree in log_degrees {
            challenger.observe(Val::<SC>::from_u8(*log_degree));
        }
        challenger.observe(Val::<SC>::from_usize(claims.len()));
        for claim in claims {
            challenger.observe(Val::<SC>::from_usize(claim.len()));
            challenger.observe_slice(claim);
        }

        let lookup_argument_challenge: SC::Challenge = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: SC::Challenge = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);

        challenger.observe(commitments.stage_2_trace.clone());
        for acc in intermediate_accumulators {
            challenger.observe_algebra_element(*acc);
        }

        let mut acc = SC::Challenge::ZERO;
        for claim in claims {
            let message = lookup_argument_challenge
                + fingerprint(&fingerprint_challenge, claim.iter().cloned());
            acc += message.inverse();
        }

        let constraint_challenge: SC::Challenge = challenger.sample_algebra_element();
        challenger.observe(commitments.quotient_chunks.clone());
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
            quotient_chunks_evaluations.push((
                trace_domain,
                vec![(zeta, quotient_opened_values[pos][0].clone())],
            ));
        }
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
        pcs.verify(coms_to_verify, opening_proof, &mut challenger)
            .map_err(VerificationError::InvalidOpeningArgument)?;

        // Out-of-domain check per active circuit.
        let extension_d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
        let empty: [SC::Challenge; 0] = [];
        for (pos, &ci) in active_indices.iter().enumerate() {
            let circuit = &self.circuits[ci];
            let degree = 1 << log_degrees[pos];
            let quotient_degree = quotient_degrees[pos];
            let next_acc = intermediate_accumulators[pos];
            let trace_domain = pcs.natural_domain_for_degree(degree);
            let sels = trace_domain.selectors_at_point(zeta);

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
            let constraint_values = circuit.compiled.evaluate_constraints(&view);
            // Fold with α (Horner): Σ_i α^{k-1-i} · value_i, matching the
            // prover's reversed α-power weighting.
            let composition = constraint_values
                .iter()
                .fold(SC::Challenge::ZERO, |acc, &v| {
                    acc * constraint_challenge + v
                });

            // Recombine the quotient from its coefficient slices at ζ.
            let quotient_row = &quotient_opened_values[pos][0];
            debug_assert_eq!(quotient_row.len(), quotient_degree * extension_d);
            let zeta_pow_n = zeta.exp_power_of_2(log2_strict_usize(degree));
            let quotient = quotient_row
                .chunks_exact(extension_d)
                .zip(zeta_pow_n.powers())
                .map(|(chunk, zeta_pow)| zeta_pow * from_ext_basis::<Val<SC>, SC::Challenge>(chunk))
                .sum::<SC::Challenge>();

            ensure_eq!(
                composition * sels.inv_vanishing,
                quotient,
                VerificationError::OodEvaluationMismatch
            );
            acc = next_acc;
        }
        Ok(())
    }

    /// Validates the structural shape of the proof. Returns the quotient
    /// degrees per active circuit on success.
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
            crate::ensure!(
                usize::from(*log_degree) + log2_strict_usize(quotient_degree)
                    <= self.config.max_log_degree(),
                VerificationError::InvalidProofShape
            );
            quotient_degrees.push(quotient_degree);
        }
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
    use super::super::expr::Expr;
    use super::super::lookup::Lookup;
    use super::super::system::{CircuitInputs, System, SystemWitness};
    use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;

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

    /// The shared even/odd constraints (a mirror of the `CS` AIR in
    /// `crate::lookup`'s tests), authored with the new frontend.
    fn cs_constraints() -> Vec<Expr<Val>> {
        let m = Expr::main(0);
        let input = Expr::main(1);
        let input_inv = Expr::main(2);
        let is_zero = Expr::main(3);
        let not_zero = Expr::main(4);
        let one = || Expr::constant(Val::ONE);
        vec![
            // is_zero and not_zero are boolean.
            is_zero.clone() * (one() - is_zero.clone()),
            not_zero.clone() * (one() - not_zero.clone()),
            // when active, exactly one of the two flags is set.
            m * (is_zero.clone() + not_zero.clone() - one()),
            // is_zero ⇒ input == 0.
            is_zero * input.clone(),
            // not_zero ⇒ input has the claimed inverse.
            not_zero * (input * input_inv - one()),
        ]
    }

    fn even_lookups() -> Vec<Lookup<Expr<Val>>> {
        let m = Expr::main(0);
        let input = Expr::main(1);
        let is_zero = Expr::main(3);
        let not_zero = Expr::main(4);
        let rec = Expr::main(5);
        let even = Expr::constant(Val::ZERO);
        let odd = Expr::constant(Val::ONE);
        let one = Expr::constant(Val::ONE);
        vec![
            // pull: negative multiplicity.
            Lookup {
                multiplicity: -m,
                args: vec![
                    even,
                    input.clone(),
                    not_zero.clone() * rec.clone() + is_zero,
                ],
            },
            Lookup {
                multiplicity: not_zero,
                args: vec![odd, input - one, rec],
            },
        ]
    }

    fn odd_lookups() -> Vec<Lookup<Expr<Val>>> {
        let m = Expr::main(0);
        let input = Expr::main(1);
        let not_zero = Expr::main(4);
        let rec = Expr::main(5);
        let even = Expr::constant(Val::ZERO);
        let odd = Expr::constant(Val::ONE);
        let one = Expr::constant(Val::ONE);
        vec![
            Lookup {
                multiplicity: -m,
                args: vec![odd, input.clone(), not_zero.clone() * rec.clone()],
            },
            Lookup {
                multiplicity: not_zero,
                args: vec![even, input - one, rec],
            },
        ]
    }

    fn witness_traces() -> Vec<RowMajorMatrix<Val>> {
        let f = Val::from_u32;
        #[rustfmt::skip]
        let traces = vec![
            RowMajorMatrix::new(
                vec![
                    f(1), f(4), f(4).inverse(), f(0), f(1), f(1),
                    f(1), f(2), f(2).inverse(), f(0), f(1), f(1),
                    f(1), f(0), f(0),           f(1), f(0), f(0),
                    f(0), f(0), f(0),           f(0), f(0), f(0),
                ],
                6,
            ),
            RowMajorMatrix::new(
                vec![
                    f(1), f(3), f(3).inverse(), f(0), f(1), f(1),
                    f(1), f(1), f(1).inverse(), f(0), f(1), f(1),
                    f(0), f(0), f(0),           f(0), f(0), f(0),
                    f(0), f(0), f(0),           f(0), f(0), f(0),
                ],
                6,
            ),
        ];
        traces
    }

    fn even_odd_system() -> (
        System<GoldilocksBlake3Config>,
        super::super::system::ProverKey<GoldilocksBlake3Config>,
    ) {
        let config = GoldilocksBlake3Config::new(COMMITMENT_PARAMETERS, FRI_PARAMETERS);
        let even = CircuitInputs {
            main_width: 6,
            constraints: cs_constraints(),
            lookups: even_lookups(),
            ..Default::default()
        };
        let odd = CircuitInputs {
            main_width: 6,
            constraints: cs_constraints(),
            lookups: odd_lookups(),
            ..Default::default()
        };
        System::new(config, [even, odd])
    }

    #[test]
    fn even_odd_prove_verify() {
        let (system, key) = even_odd_system();
        let witness = SystemWitness::from_stage_1(witness_traces(), &system);
        let f = Val::from_u32;
        let claim = &[f(0), f(4), f(1)];
        let proof = system.prove(&key, claim, witness);
        system.verify(claim, &proof).unwrap();
    }

    #[test]
    fn wrong_claim_rejected() {
        let (system, key) = even_odd_system();
        let witness = SystemWitness::from_stage_1(witness_traces(), &system);
        let f = Val::from_u32;
        let claim = &[f(0), f(4), f(1)];
        let proof = system.prove(&key, claim, witness);
        // A different claimed output must fail (unbalanced accumulator).
        let bad_claim = &[f(0), f(4), f(2)];
        assert!(system.verify(bad_claim, &proof).is_err());
    }
}
