use crate::{
    builder::folder::VerifierConstraintFolder,
    ensure, ensure_eq,
    prover::Proof,
    system::System,
    types::{Challenger, ExtVal, FriParameters, Pcs, PcsError, StarkConfig, Val},
};
use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs as PcsTrait, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};
use p3_util::log2_strict_usize;

#[derive(Debug)]
pub enum VerificationError<PcsErr> {
    InvalidClaim,
    InvalidOpeningArgument(PcsErr),
    InvalidProofShape,
    InvalidSystem,
    OodEvaluationMismatch,
    UnbalancedChannel,
}

impl<A: BaseAir<Val> + for<'a> Air<VerifierConstraintFolder<'a>>> System<A> {
    pub fn verify(
        &self,
        fri_parameters: FriParameters,
        claim: &[Val],
        proof: &Proof,
    ) -> Result<(), VerificationError<PcsError>> {
        self.verify_multiple_claims(fri_parameters, &[claim], proof)
    }

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

        // the last accumulator should be 0
        ensure_eq!(
            intermediate_accumulators.last(),
            Some(&ExtVal::ZERO),
            VerificationError::UnbalancedChannel
        );

        // initialize pcs and challenger
        let config = StarkConfig::new(self.commitment_parameters, fri_parameters);
        let pcs = config.pcs();
        let mut challenger = config.initialise_challenger();

        // observe preprocessed and stage_1 commitment
        if let Some(commit) = self.preprocessed_commit {
            challenger.observe(commit);
        }
        challenger.observe(commitments.stage_1_trace);

        // observe the traces' heights. TODO: is this necessary?
        for log_degree in log_degrees {
            challenger.observe(Val::from_u8(*log_degree));
        }

        // observe the claims
        for claim in claims {
            challenger.observe_slice(claim);
        }

        // generate lookup challenges
        let lookup_argument_challenge: ExtVal = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: ExtVal = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);

        // observe stage_2 commitment
        challenger.observe(commitments.stage_2_trace);

        // construct the accumulator from the claims
        let mut acc = ExtVal::ZERO;
        for claim in claims {
            let message = lookup_argument_challenge
                + claim.iter().rev().fold(ExtVal::ZERO, |acc, &coeff| {
                    acc * fingerprint_challenge + coeff
                });
            acc += message.inverse();
        }

        // generate constraint challenge
        let constraint_challenge: ExtVal = challenger.sample_algebra_element();

        // observe quotient commitment
        challenger.observe(commitments.quotient_chunks);

        // generate out of domain points and verify the PCS opening
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
            (commitments.stage_1_trace, stage_1_trace_evaluations),
            (commitments.stage_2_trace, stage_2_trace_evaluations),
            (commitments.quotient_chunks, quotient_chunks_evaluations),
        ];
        if let Some(preprocessed_commitment) = self.preprocessed_commit {
            coms_to_verify.extend([(preprocessed_commitment, preprocessed_trace_evaluations)])
        }
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
                Some(VerticalPair::new(
                    RowMajorMatrixView::new_row(preprocessed_row),
                    RowMajorMatrixView::new_row(preprocessed_next_row),
                ))
            } else {
                None
            };
            let stage_1 = VerticalPair::new(
                RowMajorMatrixView::new_row(stage_1_row),
                RowMajorMatrixView::new_row(stage_1_next_row),
            );
            let extension_d = <ExtVal as BasedVectorSpace<Val>>::DIMENSION;
            let stage_2_row = &stage_2_row
                .chunks_exact(extension_d)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .map(|(i, c)| {
                            *c * <ExtVal as BasedVectorSpace<Val>>::ith_basis_element(i).unwrap()
                        })
                        .sum()
                })
                .collect::<Vec<_>>();
            let stage_2_next_row = &stage_2_next_row
                .chunks_exact(extension_d)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .map(|(i, c)| {
                            *c * <ExtVal as BasedVectorSpace<Val>>::ith_basis_element(i).unwrap()
                        })
                        .sum()
                })
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
                .map(|(ch_i, ch)| {
                    zps[ch_i]
                        * ch.iter()
                            .enumerate()
                            .map(|(e_i, &c)| {
                                <ExtVal as BasedVectorSpace<Val>>::ith_basis_element(e_i).unwrap()
                                    * c
                            })
                            .sum::<ExtVal>()
                })
                .sum::<ExtVal>();

            // finally, check that the composition polynomial
            // is divisible by the quotient polynomial
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
        // the preprocessed commitment is empty if and only if there are zero preprocessed chips
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        benchmark,
        lookup::LookupAir,
        system::{ProverKey, SystemWitness},
        types::{CommitmentParameters, FriParameters},
    };
    use p3_air::{AirBuilderWithPublicValues, BaseAir};
    use p3_matrix::{Matrix, dense::RowMajorMatrix};

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
        AB: AirBuilderWithPublicValues,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                Self::Pythagorean => {
                    let main = builder.main();
                    let local = main.row_slice(0).unwrap();
                    let expr1 = local[0] * local[0] + local[1] * local[1];
                    let expr2 = local[2] * local[2];
                    // this extra `local[0]` multiplication is there to increase the maximum constraint degree
                    builder.assert_eq(local[0] * expr1, local[0] * expr2);
                }
                Self::Complex => {
                    let main = builder.main();
                    let local = main.row_slice(0).unwrap();
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
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
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
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let no_claims = &[];
        let proof = system.prove_multiple_claims(fri_parameters, &key, no_claims, witness);
        system
            .verify_multiple_claims(fri_parameters, no_claims, &proof)
            .unwrap();
    }

    #[test]
    #[ignore]
    fn multi_stark_benchmark_test() {
        // To run this benchmark effectively, run the following command
        // RUSTFLAGS="-Ctarget-cpu=native" cargo test multi_stark_benchmark_test --release --features parallel -- --include-ignored --nocapture
        const LOG_HEIGHT: usize = 20;
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let (system, key) = system(commitment_parameters);
        let f = Val::from_u32;
        let mut pythagorean_trace = [3, 4, 5].map(f).to_vec();
        let mut complex_trace = [4, 2, 3, 1, 10, 10].map(f).to_vec();
        for _ in 0..LOG_HEIGHT {
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
            num_queries: 100,
            proof_of_work_bits: 20,
        };
        let no_claims = &[];
        let proof = benchmark!(
            system.prove_multiple_claims(fri_parameters, &key, no_claims, witness),
            "proof: "
        );
        let proof_bytes = proof.to_bytes().expect("Failed to serialize proof");
        println!("Proof size: {} bytes", proof_bytes.len());
        benchmark!(
            system
                .verify_multiple_claims(fri_parameters, no_claims, &proof)
                .unwrap(),
            "verification: "
        );
    }
}
