use crate::types::{Challenger, ExtVal, PackedExtVal, PackedVal, Pcs, StarkConfig, Val};

use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs as PcsTrait, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing};
use p3_matrix::{
    Matrix,
    dense::{RowMajorMatrix, RowMajorMatrixView},
    stack::VerticalPair,
};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{
    Domain, PcsError, ProverConstraintFolder, StarkGenericConfig, SymbolicAirBuilder,
    VerificationError, VerifierConstraintFolder, get_log_quotient_degree, get_symbolic_constraints,
};
use p3_util::{log2_strict_usize, zip_eq::zip_eq};

pub struct Proof {
    pub commitments: Commitments,
    pub opened_values: OpenedValues,
    pub opening_proof: PcsProof,
    pub degree_bits: usize,
}

type Commitment = <Pcs as PcsTrait<ExtVal, Challenger>>::Commitment;
type PcsProof = <Pcs as PcsTrait<ExtVal, Challenger>>::Proof;

pub struct Commitments {
    pub trace: Commitment,
    pub quotient_chunks: Commitment,
    pub random: Option<Commitment>,
}

pub struct OpenedValues {
    pub trace_local: Vec<ExtVal>,
    pub trace_next: Vec<ExtVal>,
    pub quotient_chunks: Vec<Vec<ExtVal>>,
    pub random: Option<Vec<ExtVal>>,
}

pub fn prove<A>(
    config: &StarkConfig,
    air: &A,
    trace: RowMajorMatrix<Val>,
    public_values: &Vec<Val>,
) -> Proof
where
    A: Air<SymbolicAirBuilder<Val>> + for<'a> Air<ProverConstraintFolder<'a, StarkConfig>>,
{
    let degree = trace.height();
    let log_degree = log2_strict_usize(degree);

    let log_quotient_degree = get_log_quotient_degree::<Val, A>(air, 0, public_values.len(), 0);
    let quotient_degree = 1 << log_quotient_degree;

    let pcs: &Pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    let trace_domain =
        <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);
    let (trace_commit, trace_data) =
        <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, [(trace_domain, trace)]);

    challenger.observe(Val::from_u8(log_degree as u8));
    challenger.observe(trace_commit);
    challenger.observe_slice(public_values);

    let alpha: ExtVal = challenger.sample_algebra_element();

    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));

    let trace_on_quotient_domain = <Pcs as PcsTrait<ExtVal, Challenger>>::get_evaluations_on_domain(
        pcs,
        &trace_data,
        0,
        quotient_domain,
    );

    let constraint_count = get_symbolic_constraints(air, 0, public_values.len()).len();
    let quotient_values = quotient_values(
        air,
        public_values,
        trace_domain,
        quotient_domain,
        trace_on_quotient_domain,
        alpha,
        constraint_count,
    );

    let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();

    let (quotient_commit, quotient_data) = <Pcs as PcsTrait<ExtVal, Challenger>>::commit_quotient(
        pcs,
        quotient_domain,
        quotient_flat,
        quotient_degree,
    );
    challenger.observe(quotient_commit);

    let commitments = Commitments {
        trace: trace_commit,
        quotient_chunks: quotient_commit,
        random: None,
    };

    let zeta: ExtVal = challenger.sample_algebra_element();
    let zeta_next = trace_domain.next_point(zeta).unwrap();

    let (opened_values, opening_proof) = {
        let round1 = (&trace_data, vec![vec![zeta, zeta_next]]);
        let round2 = (&quotient_data, vec![vec![zeta]; quotient_degree]);

        let rounds = vec![round1, round2];

        pcs.open(rounds, &mut challenger)
    };
    let trace_idx = 0;
    let quotient_idx = 1;
    let trace_local = opened_values[trace_idx][0][0].clone();
    let trace_next = opened_values[trace_idx][0][1].clone();
    let quotient_chunks = opened_values[quotient_idx]
        .iter()
        .map(|v| v[0].clone())
        .collect();
    let random = None;
    let opened_values = OpenedValues {
        trace_local,
        trace_next,
        quotient_chunks,
        random,
    };
    Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits: log_degree,
    }
}

fn quotient_values<A, Mat>(
    air: &A,
    public_values: &Vec<Val>,
    trace_domain: Domain<StarkConfig>,
    quotient_domain: Domain<StarkConfig>,
    trace_on_quotient_domain: Mat,
    alpha: ExtVal,
    constraint_count: usize,
) -> Vec<ExtVal>
where
    A: for<'a> Air<ProverConstraintFolder<'a, StarkConfig>>,
    Mat: Matrix<Val> + Sync,
{
    let quotient_size = quotient_domain.size();
    let width = trace_on_quotient_domain.width();
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    for _ in quotient_size..PackedVal::WIDTH {
        sels.is_first_row.push(Val::default());
        sels.is_last_row.push(Val::default());
        sels.is_transition.push(Val::default());
        sels.inv_vanishing.push(Val::default());
    }

    let mut alpha_powers = alpha.powers().collect_n(constraint_count);
    alpha_powers.reverse();

    let decomposed_alpha_powers: Vec<_> = (0..<ExtVal as BasedVectorSpace<Val>>::DIMENSION)
        .map(|i| {
            alpha_powers
                .iter()
                .map(|x| x.as_basis_coefficients_slice()[i])
                .collect()
        })
        .collect();
    (0..quotient_size)
        .into_par_iter()
        .step_by(PackedVal::WIDTH)
        .flat_map_iter(|i_start| {
            let i_range = i_start..i_start + PackedVal::WIDTH;

            let is_first_row = *PackedVal::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_vanishing = *PackedVal::from_slice(&sels.inv_vanishing[i_range]);

            let main = RowMajorMatrix::new(
                trace_on_quotient_domain.vertically_packed_row_pair(i_start, next_step),
                width,
            );

            let accumulator = PackedExtVal::ZERO;
            let mut folder = ProverConstraintFolder {
                main: main.as_view(),
                public_values,
                is_first_row,
                is_last_row,
                is_transition,
                alpha_powers: &alpha_powers,
                decomposed_alpha_powers: &decomposed_alpha_powers,
                accumulator,
                constraint_index: 0,
            };
            air.eval(&mut folder);

            let quotient = folder.accumulator * inv_vanishing;

            (0..core::cmp::min(quotient_size, PackedVal::WIDTH)).map(move |idx_in_packing| {
                ExtVal::from_basis_coefficients_fn(|coeff_idx| {
                    <PackedExtVal as BasedVectorSpace<PackedVal>>::as_basis_coefficients_slice(
                        &quotient,
                    )[coeff_idx]
                        .as_slice()[idx_in_packing]
                })
            })
        })
        .collect()
}

pub fn verify<A>(
    config: &StarkConfig,
    air: &A,
    proof: &Proof,
    public_values: &Vec<Val>,
) -> Result<(), VerificationError<PcsError<StarkConfig>>>
where
    A: Air<SymbolicAirBuilder<Val>> + for<'a> Air<VerifierConstraintFolder<'a, StarkConfig>>,
{
    let Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits,
    } = proof;

    let pcs = config.pcs();

    let degree = 1 << degree_bits;
    let log_quotient_degree = get_log_quotient_degree::<Val, A>(air, 0, public_values.len(), 0);
    let quotient_degree = 1 << log_quotient_degree;

    let mut challenger = config.initialise_challenger();
    let trace_domain =
        <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);

    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (degree_bits + log_quotient_degree));
    let quotient_chunks_domains = quotient_domain.split_domains(quotient_degree);

    let randomized_quotient_chunks_domains = quotient_chunks_domains
        .iter()
        .map(|domain| {
            <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, domain.size())
        })
        .collect::<Vec<_>>();

    if opened_values.random.is_some() || commitments.random.is_some() {
        return Err(VerificationError::RandomizationError);
    }

    let air_width = A::width(air);
    let valid_shape = opened_values.trace_local.len() == air_width
        && opened_values.trace_next.len() == air_width
        && opened_values.quotient_chunks.len() == quotient_degree
        && opened_values
            .quotient_chunks
            .iter()
            .all(|qc| qc.len() == <ExtVal as BasedVectorSpace<Val>>::DIMENSION)
        && if let Some(r_comm) = &opened_values.random {
            r_comm.len() == <ExtVal as BasedVectorSpace<Val>>::DIMENSION
        } else {
            true
        };
    if !valid_shape {
        return Err(VerificationError::InvalidProofShape);
    }

    challenger.observe(Val::from_usize(proof.degree_bits));
    challenger.observe(commitments.trace);
    challenger.observe_slice(public_values);

    let alpha: ExtVal = challenger.sample_algebra_element();
    challenger.observe(commitments.quotient_chunks);

    let zeta: ExtVal = challenger.sample_algebra_element();
    let zeta_next: ExtVal = trace_domain.next_point(zeta).unwrap();

    let coms_to_verify = vec![
        (
            commitments.trace,
            vec![(
                trace_domain,
                vec![
                    (zeta, opened_values.trace_local.clone()),
                    (zeta_next, opened_values.trace_next.clone()),
                ],
            )],
        ),
        (
            commitments.quotient_chunks,
            zip_eq(
                randomized_quotient_chunks_domains.iter(),
                &opened_values.quotient_chunks,
                VerificationError::InvalidProofShape,
            )?
            .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
            .collect::<Vec<_>>(),
        ),
    ];

    pcs.verify(coms_to_verify, opening_proof, &mut challenger)
        .map_err(VerificationError::InvalidOpeningArgument)?;

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

    let quotient = opened_values
        .quotient_chunks
        .iter()
        .enumerate()
        .map(|(ch_i, ch)| {
            zps[ch_i]
                * ch.iter()
                    .enumerate()
                    .map(|(e_i, &c)| {
                        <ExtVal as BasedVectorSpace<Val>>::ith_basis_element(e_i).unwrap() * c
                    })
                    .sum::<ExtVal>()
        })
        .sum::<ExtVal>();

    let sels = trace_domain.selectors_at_point(zeta);

    let main = VerticalPair::new(
        RowMajorMatrixView::new_row(&opened_values.trace_local),
        RowMajorMatrixView::new_row(&opened_values.trace_next),
    );

    let mut folder = VerifierConstraintFolder {
        main,
        public_values,
        is_first_row: sels.is_first_row,
        is_last_row: sels.is_last_row,
        is_transition: sels.is_transition,
        alpha,
        accumulator: ExtVal::ZERO,
    };
    air.eval(&mut folder);
    let folded_constraints = folder.accumulator;

    if folded_constraints * sels.inv_vanishing != quotient {
        return Err(VerificationError::OodEvaluationMismatch);
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::{FriParameters, new_stark_config};
    use p3_air::{AirBuilderWithPublicValues, BaseAir};

    fn demo_trace() -> RowMajorMatrix<Val> {
        let f = Val::from_u32;
        let trace = [3, 4, 5, 5, 12, 13, 8, 15, 17, 7, 24, 25].map(f).to_vec();
        RowMajorMatrix::new(trace, 3)
    }

    struct CS {}

    impl<F> BaseAir<F> for CS {
        fn width(&self) -> usize {
            3
        }
    }

    impl<AB: AirBuilderWithPublicValues> Air<AB> for CS {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.row_slice(0).unwrap();
            let expr1 = local[0] * local[0] + local[1] * local[1];
            let expr2 = local[2] * local[2];
            builder.assert_eq(expr1, expr2);
        }
    }

    #[test]
    fn prove_verify_test() {
        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let config = new_stark_config(fri_parameters);

        let trace = demo_trace();
        let pis = vec![];
        let proof = prove(&config, &CS {}, trace, &pis);
        verify(&config, &CS {}, &proof, &pis).expect("verification failed");
    }
}
