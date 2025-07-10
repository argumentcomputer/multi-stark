use crate::{
    builder::folder::ProverConstraintFolder,
    system::{Name, System, SystemWitness},
    types::{Challenger, Domain, ExtVal, PackedExtVal, PackedVal, Pcs, StarkConfig, Val},
};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{OpenedValuesForRound, Pcs as PcsTrait, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use std::{cmp::min, iter::once};

#[derive(Serialize, Deserialize)]
pub struct Claim {
    pub circuit_name: Name,
    pub args: Vec<Val>,
}

type Commitment = <Pcs as PcsTrait<ExtVal, Challenger>>::Commitment;
type PcsProof = <Pcs as PcsTrait<ExtVal, Challenger>>::Proof;

#[derive(Serialize, Deserialize)]
pub struct Commitments {
    // TODO add stage 2
    pub stage_1_trace: Commitment,
    pub quotient_chunks: Commitment,
}

#[derive(Serialize, Deserialize)]
pub struct Proof {
    pub claim: Claim,
    pub commitments: Commitments,
    pub log_degrees: Vec<u8>,
    pub opening_proof: PcsProof,
    pub stage_1_opened_values: OpenedValuesForRound<ExtVal>,
    pub quotient_opened_values: OpenedValuesForRound<ExtVal>,
}

impl<A: BaseAirWithPublicValues<Val> + for<'a> Air<ProverConstraintFolder<'a>>> System<A> {
    pub fn prove(&self, config: &StarkConfig, claim: Claim, witness: SystemWitness) -> Proof {
        // initialize pcs and challenger
        let pcs = config.pcs();
        let mut challenger = config.initialise_challenger();

        // commit to stage 1 traces
        let mut log_degrees = vec![];
        let evaluations = witness.circuits.into_iter().map(|witness| {
            let trace = witness.trace;
            let degree = trace.height();
            let log_degree = log2_strict_usize(degree);
            let trace_domain =
                <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);
            log_degrees.push(log_degree);
            (trace_domain, trace)
        });
        let (stage_1_trace_commit, stage_1_trace_data) =
            <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, evaluations);
        // TODO: do we have to observe the log_degrees?
        challenger.observe(stage_1_trace_commit);

        // generate lookup challenges
        // TODO use `ExtVal` instead of `Val`
        let lookup_argument_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);
        // TODO commit to stage 2 traces

        // observe the claim
        let circuit_index = Val::from_usize(*self.circuit_names.get(&claim.circuit_name).unwrap());
        challenger.observe(circuit_index);
        challenger.observe_slice(&claim.args);
        // construct the accumulator from the claim
        let claim_iter = claim.args.iter().rev().copied().chain(once(circuit_index));
        let init_acc = fingerprint_reverse(fingerprint_challenge, claim_iter);
        let public_values = vec![init_acc, lookup_argument_challenge, fingerprint_challenge];

        // generate constraint challenge
        let constraint_challenge: ExtVal = challenger.sample_algebra_element();

        // commit to evaluations of the quotient polynomials
        let mut quotient_degrees = vec![];
        let quotient_evaluations = self
            .circuits
            .iter()
            .zip(log_degrees.iter())
            .enumerate()
            .flat_map(|(idx, (circuit, log_degree))| {
                let air = &circuit.air;
                // quotient degree is at most 1 less than the max degree, padded to a power of two
                let quotient_degree =
                    (circuit.max_constraint_degree.max(2) - 1).next_power_of_two();
                let log_quotient_degree = log2_strict_usize(quotient_degree);
                let trace_domain = <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                    pcs,
                    1 << log_degree,
                );
                let quotient_domain =
                    trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));
                // TODO add stage 2 traces
                let stage_1_trace_on_quotient_domain =
                    <Pcs as PcsTrait<ExtVal, Challenger>>::get_evaluations_on_domain(
                        pcs,
                        &stage_1_trace_data,
                        idx,
                        quotient_domain,
                    );
                // compute the quotient values which are elements of the extension field and flatten it to the base field
                let quotient_values = quotient_values(
                    air,
                    &public_values,
                    trace_domain,
                    quotient_domain,
                    stage_1_trace_on_quotient_domain,
                    constraint_challenge,
                    circuit.constraint_count,
                );
                let quotient_flat =
                    RowMajorMatrix::new_col(quotient_values).flatten_to_base::<Val>();
                // note that, in general, the quotients have a degree that is greater than the trace polynomials,
                // so for FRI to work so we must split into smaller polynomials
                let quotient_sub_evaluations =
                    quotient_domain.split_evals(quotient_degree, quotient_flat);
                let quotient_sub_domains = quotient_domain.split_domains(quotient_degree);
                // need to save the quotient degree for later
                quotient_degrees.push(quotient_degree);
                quotient_sub_domains
                    .into_iter()
                    .zip(quotient_sub_evaluations)
            });
        let (quotient_commit, quotient_data) =
            <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, quotient_evaluations);
        challenger.observe(quotient_commit);

        // save the commitments
        let commitments = Commitments {
            stage_1_trace: stage_1_trace_commit,
            quotient_chunks: quotient_commit,
        };

        // generate the out of domain point and prove polynomial evaluations
        let zeta: ExtVal = challenger.sample_algebra_element();
        let mut round1_openings = vec![];
        let mut round2_openings = vec![];
        log_degrees.iter().zip(quotient_degrees.iter()).for_each(
            |(log_degree, quotient_degree)| {
                let trace_domain = <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                    pcs,
                    1 << log_degree,
                );
                let zeta_next = trace_domain.next_point(zeta).unwrap();
                round1_openings.push(vec![zeta, zeta_next]);
                round2_openings.extend(vec![vec![zeta]; *quotient_degree]);
            },
        );
        let rounds = vec![
            (&stage_1_trace_data, round1_openings),
            (&quotient_data, round2_openings),
        ];
        let (opened_values, opening_proof) = pcs.open(rounds, &mut challenger);
        let mut opened_values_iter = opened_values.into_iter();
        let stage_1_opened_values = opened_values_iter.next().unwrap();
        let quotient_opened_values = opened_values_iter.next().unwrap();
        debug_assert!(opened_values_iter.next().is_none());
        let log_degrees = log_degrees.into_iter().map(|n| n as u8).collect();
        Proof {
            claim,
            commitments,
            stage_1_opened_values,
            quotient_opened_values,
            opening_proof,
            log_degrees,
        }
    }
}

// Compute a fingerprint of the coefficients in reverse using Horner's method:
pub(crate) fn fingerprint_reverse<F: Field, Iter: Iterator<Item = F>>(r: F, coeffs: Iter) -> F {
    coeffs.fold(F::ZERO, |acc, coeff| acc * r + coeff)
}

// TODO take stage 2 traces and update the accumulator
fn quotient_values<A, Mat>(
    air: &A,
    public_values: &Vec<Val>,
    trace_domain: Domain,
    quotient_domain: Domain,
    trace_on_quotient_domain: Mat,
    alpha: ExtVal,
    constraint_count: usize,
) -> Vec<ExtVal>
where
    A: for<'a> Air<ProverConstraintFolder<'a>>,
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
                // TODO fix preprocessed and stage_2
                preprocessed: main.as_view(),
                stage_1: main.as_view(),
                stage_2: main.as_view(),
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

            (0..min(quotient_size, PackedVal::WIDTH)).map(move |idx_in_packing| {
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
