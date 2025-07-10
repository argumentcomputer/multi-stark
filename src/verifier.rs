use crate::{
    builder::folder::VerifierConstraintFolder,
    ensure_eq,
    prover::{Proof, fingerprint_reverse},
    system::System,
    types::{Challenger, ExtVal, Pcs, PcsError, StarkConfig, Val},
};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs as PcsTrait, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};
use p3_util::log2_strict_usize;
use std::iter::once;

#[derive(Debug)]
pub enum VerificationError<PcsErr> {
    InvalidClaim,
    InvalidProofShape,
    InvalidOpeningArgument(PcsErr),
    OodEvaluationMismatch,
}

impl<A: BaseAirWithPublicValues<Val> + for<'a> Air<VerifierConstraintFolder<'a>>> System<A> {
    pub fn verify(
        &self,
        config: &StarkConfig,
        proof: &Proof,
    ) -> Result<(), VerificationError<PcsError>> {
        let Proof {
            commitments,
            stage_1_opened_values,
            stage_2_opened_values,
            quotient_opened_values,
            opening_proof,
            claim,
            log_degrees,
        } = proof;
        let num_circuits = self.circuits.len();

        // check the claim and proof shape
        let circuit_index = Val::from_usize(
            *self
                .circuit_names
                .get(&claim.circuit_name)
                .ok_or(VerificationError::InvalidClaim)?,
        );
        // stage 1 round
        ensure_eq!(
            stage_1_opened_values.len(),
            num_circuits,
            VerificationError::InvalidProofShape
        );
        for (i, circuit) in self.circuits.iter().enumerate() {
            // zeta and zeta_next
            let num_openings = 2;
            ensure_eq!(
                stage_1_opened_values[i].len(),
                num_openings,
                VerificationError::InvalidProofShape
            );
            for j in 0..num_openings {
                ensure_eq!(
                    stage_1_opened_values[i][j].len(),
                    circuit.stage_1_width,
                    VerificationError::InvalidProofShape
                );
            }
        }
        // TODO missing stage 2 round
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

        // initialize pcs and challenger
        let pcs = config.pcs();
        let mut challenger = config.initialise_challenger();

        // observe stage_1 commitment
        challenger.observe(commitments.stage_1_trace);

        // generate lookup challenges
        // TODO use `ExtVal` instead of `Val`
        let lookup_argument_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);
        // TODO commit to stage 2 traces

        // observe the claim
        challenger.observe(circuit_index);
        challenger.observe_slice(&claim.args);
        // construct the accumulator from the claim
        let claim_iter = claim.args.iter().rev().copied().chain(once(circuit_index));
        let init_acc = fingerprint_reverse(fingerprint_challenge, claim_iter);
        let public_values = vec![init_acc, lookup_argument_challenge, fingerprint_challenge];
        // TODO stage 2

        // generate constraint challenge
        let constraint_challenge: ExtVal = challenger.sample_algebra_element();

        // observe quotient commitment
        challenger.observe(commitments.quotient_chunks);

        // generate out of domain points and verify the PCS opening
        let zeta: ExtVal = challenger.sample_algebra_element();
        let mut stage_1_trace_evaluations = vec![];
        let mut quotient_chunks_evaluations = vec![];
        let mut last_quotient_i = 0;
        log_degrees
            .iter()
            .zip(quotient_degrees.iter())
            .enumerate()
            .for_each(|(i, (log_degree, quotient_degree))| {
                let log_quotient_degree = log2_strict_usize(*quotient_degree);
                let trace_domain = <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                    pcs,
                    1 << log_degree,
                );
                let quotient_domain =
                    trace_domain.create_disjoint_domain((1 << log_degree) << log_quotient_degree);
                let quotient_chunks_domains = quotient_domain.split_domains(*quotient_degree);
                let unshifted_quotient_chunks_domains = quotient_chunks_domains
                    .iter()
                    .map(|domain| {
                        <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                            pcs,
                            domain.size(),
                        )
                    })
                    .collect::<Vec<_>>();
                let zeta_next = trace_domain.next_point(zeta).unwrap();
                stage_1_trace_evaluations.push((
                    trace_domain,
                    vec![
                        (zeta, stage_1_opened_values[i][0].clone()),
                        (zeta_next, stage_1_opened_values[i][1].clone()),
                    ],
                ));
                let iter = unshifted_quotient_chunks_domains
                    .into_iter()
                    .zip(
                        quotient_opened_values[last_quotient_i..last_quotient_i + quotient_degree]
                            .iter(),
                    )
                    .map(|(domain, opened_values)| {
                        (domain, vec![(zeta, opened_values[0].clone())])
                    });
                quotient_chunks_evaluations.extend(iter);
                last_quotient_i += quotient_degree;
            });
        let coms_to_verify = vec![
            (commitments.stage_1_trace, stage_1_trace_evaluations),
            (commitments.quotient_chunks, quotient_chunks_evaluations),
        ];
        pcs.verify(coms_to_verify, opening_proof, &mut challenger)
            .map_err(VerificationError::InvalidOpeningArgument)?;

        // use the opened values to compute the composition polynomial for each circuit
        // and check that the evaluation of the composition polynomial equals the
        // product of the zerofier with the quotient
        let mut last_quotient_i = 0;
        for i in 0..num_circuits {
            let circuit = &self.circuits[i];
            let degree = 1 << log_degrees[i];
            let quotient_degree = quotient_degrees[i];
            let row = &stage_1_opened_values[i][0];
            let next_row = &stage_1_opened_values[i][0];
            let quotient_chunks = quotient_opened_values
                [last_quotient_i..last_quotient_i + quotient_degree]
                .iter()
                .map(|values| &values[0]);
            last_quotient_i += quotient_degree;

            // compute the composition polynomial evaluation
            let trace_domain =
                <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);
            let sels = trace_domain.selectors_at_point(zeta);
            let main = VerticalPair::new(
                RowMajorMatrixView::new_row(row),
                RowMajorMatrixView::new_row(next_row),
            );
            let mut folder = VerifierConstraintFolder {
                // TODO fix preprocessed and stage_2
                preprocessed: main,
                stage_1: main,
                stage_2: main,
                public_values: &public_values,
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
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        benchmark,
        prover::Claim,
        system::{Circuit, CircuitWitness, MIN_IO_SIZE, SystemWitness},
        types::{FriParameters, new_stark_config},
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
                CS::Pythagorean => 3,
                CS::Complex => 6,
            }
        }
    }
    impl<F> BaseAirWithPublicValues<F> for CS {
        fn num_public_values(&self) -> usize {
            MIN_IO_SIZE
        }
    }
    impl<AB> Air<AB> for CS
    where
        AB: AirBuilderWithPublicValues,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                CS::Pythagorean => {
                    let main = builder.main();
                    let local = main.row_slice(0).unwrap();
                    let expr1 = local[0] * local[0] + local[1] * local[1];
                    let expr2 = local[2] * local[2];
                    // this extra `local[0]` multiplication is there to increase the maximum constraint degree
                    builder.assert_eq(local[0] * expr1, local[0] * expr2);
                }
                CS::Complex => {
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
    fn system() -> System<CS> {
        let pythagorean_circuit = Circuit::from_air_single_stage(CS::Pythagorean).unwrap();
        let complex_circuit = Circuit::from_air_single_stage(CS::Complex).unwrap();
        System::new(
            [
                ("pythagorean", pythagorean_circuit),
                ("complex", complex_circuit),
            ]
            .into_iter(),
        )
    }

    #[test]
    fn multi_stark_test() {
        let system = system();
        let f = Val::from_u32;
        let witness = SystemWitness {
            circuits: vec![
                CircuitWitness {
                    trace: RowMajorMatrix::new(
                        [3, 4, 5, 5, 12, 13, 8, 15, 17, 7, 24, 25].map(f).to_vec(),
                        3,
                    ),
                },
                CircuitWitness {
                    trace: RowMajorMatrix::new(
                        [4, 2, 3, 1, 10, 10, 3, 2, 5, 1, 13, 13].map(f).to_vec(),
                        6,
                    ),
                },
            ],
        };
        // lookup arguments not yet implemented so the claim doesn't matter
        let dummy_claim = Claim {
            circuit_name: "complex".into(),
            args: vec![],
        };
        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let config = new_stark_config(fri_parameters);
        let proof = system.prove(&config, dummy_claim, witness);
        system.verify(&config, &proof).unwrap();
    }

    #[test]
    #[ignore]
    fn multi_stark_benchmark_test() {
        // To run this benchmark effectively, run the following command
        // RUSTFLAGS="-Ctarget-cpu=native" cargo test multi_stark_benchmark_test --release --features parallel -- --include-ignored --nocapture
        const LOG_HEIGHT: usize = 20;
        let system = system();
        let f = Val::from_u32;
        let mut pythagorean_trace = [3, 4, 5].map(f).to_vec();
        let mut complex_trace = [4, 2, 3, 1, 10, 10].map(f).to_vec();
        for _ in 0..LOG_HEIGHT {
            pythagorean_trace.extend(pythagorean_trace.clone());
            complex_trace.extend(complex_trace.clone());
        }
        let witness = SystemWitness {
            circuits: vec![
                CircuitWitness {
                    trace: RowMajorMatrix::new(pythagorean_trace, 3),
                },
                CircuitWitness {
                    trace: RowMajorMatrix::new(complex_trace, 6),
                },
            ],
        };
        // lookup arguments not yet implemented so the claim doesn't matter
        let dummy_claim = Claim {
            circuit_name: "complex".into(),
            args: vec![],
        };
        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 100,
            proof_of_work_bits: 20,
        };
        let config = new_stark_config(fri_parameters);
        let proof = benchmark!(system.prove(&config, dummy_claim, witness), "proof: ");
        let bincode_config = bincode::config::standard()
            .with_little_endian()
            .with_fixed_int_encoding();
        let proof_bytes = bincode::serde::encode_to_vec(&proof, bincode_config)
            .expect("Failed to serialize proof");
        println!("Proof size: {} bytes", proof_bytes.len());
        benchmark!(system.verify(&config, &proof).unwrap(), "verification: ");
    }
}
