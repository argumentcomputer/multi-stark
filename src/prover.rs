use crate::{
    builder::folder::ProverConstraintFolder,
    lookup::Lookup,
    system::{ProverKey, System, SystemWitness},
    types::{
        Challenger, Commitment, Domain, ExtVal, PackedExtVal, PackedVal, Pcs, PcsProof,
        StarkConfig, Val,
    },
};
use bincode::{
    config::{Configuration, Fixint, LittleEndian, standard},
    error::{DecodeError, EncodeError},
    serde::{decode_from_slice, encode_to_vec},
};
use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{LagrangeSelectors, OpenedValuesForRound, Pcs as PcsTrait, PolynomialSpace};
use p3_field::{
    BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing,
    extension::BinomialExtensionField,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use std::cmp::min;

#[derive(Serialize, Deserialize)]
pub struct Commitments {
    pub stage_1_trace: Commitment,
    pub stage_2_trace: Commitment,
    pub quotient_chunks: Commitment,
}

#[derive(Serialize, Deserialize)]
pub struct Proof {
    pub commitments: Commitments,
    pub intermediate_accumulators: Vec<Val>,
    pub log_degrees: Vec<u8>,
    pub opening_proof: PcsProof,
    pub quotient_opened_values: OpenedValuesForRound<ExtVal>,
    pub stage_1_opened_values: OpenedValuesForRound<ExtVal>,
    pub stage_2_opened_values: OpenedValuesForRound<ExtVal>,
}

impl Proof {
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

impl<A: BaseAir<Val> + for<'a> Air<ProverConstraintFolder<'a>>> System<A> {
    pub fn prove(
        &self,
        config: &StarkConfig,
        key: ProverKey,
        claim: &[Val],
        witness: SystemWitness,
    ) -> Proof {
        let multiplicity = Val::ONE;
        self.prove_with_claim_multiplicy(config, key, multiplicity, claim, witness)
    }

    pub fn prove_with_claim_multiplicy(
        &self,
        config: &StarkConfig,
        key: ProverKey,
        multiplicity: Val,
        claim: &[Val],
        witness: SystemWitness,
    ) -> Proof {
        // initialize pcs and challenger
        let pcs = config.pcs();
        let mut challenger = config.initialise_challenger();

        // commit to stage 1 traces
        let mut log_degrees = vec![];
        let evaluations = witness.traces.into_iter().map(|trace| {
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

        // observe the claim
        // this has to be done before generating the lookup argument challenge
        // otherwise the lookup argument can be attacked
        challenger.observe_slice(claim);

        // generate lookup challenges
        // TODO use `ExtVal` instead of `Val`
        let lookup_argument_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);

        // construct the accumulator from the claim
        let message = lookup_argument_challenge
            + claim
                .iter()
                .rev()
                .fold(Val::ZERO, |acc, &coeff| acc * fingerprint_challenge + coeff);
        let mut acc = multiplicity * message.inverse();
        // commit to stage 2 traces
        let (stage_2_traces, intermediate_accumulators) = Lookup::stage_2_traces(
            &witness.lookups,
            &[lookup_argument_challenge, fingerprint_challenge, acc],
        );
        let evaluations = stage_2_traces.into_iter().map(|trace| {
            let degree = trace.height();
            let trace_domain =
                <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);
            (trace_domain, trace)
        });
        let (stage_2_trace_commit, stage_2_trace_data) =
            <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, evaluations);
        challenger.observe(stage_2_trace_commit);

        // generate constraint challenge
        let constraint_challenge: ExtVal = challenger.sample_algebra_element();

        // commit to evaluations of the quotient polynomials
        debug_assert_eq!(intermediate_accumulators.len(), self.circuits.len());
        debug_assert_eq!(log_degrees.len(), self.circuits.len());
        let mut quotient_degrees = vec![];
        let quotient_evaluations = self
            .circuits
            .iter()
            .zip(log_degrees.iter())
            .zip(intermediate_accumulators.iter())
            .enumerate()
            .flat_map(|(idx, ((circuit, log_degree), next_acc))| {
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
                let stage_1_trace_on_quotient_domain =
                    <Pcs as PcsTrait<ExtVal, Challenger>>::get_evaluations_on_domain(
                        pcs,
                        &stage_1_trace_data,
                        idx,
                        quotient_domain,
                    );
                let stage_2_trace_on_quotient_domain =
                    <Pcs as PcsTrait<ExtVal, Challenger>>::get_evaluations_on_domain(
                        pcs,
                        &stage_2_trace_data,
                        idx,
                        quotient_domain,
                    );
                // compute the quotient values which are elements of the extension field and flatten it to the base field
                let public_values = [
                    lookup_argument_challenge,
                    fingerprint_challenge,
                    acc,
                    *next_acc,
                ];
                let quotient_values = quotient_values(
                    air,
                    &public_values,
                    trace_domain,
                    quotient_domain,
                    &stage_1_trace_on_quotient_domain,
                    &stage_2_trace_on_quotient_domain,
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
                acc = *next_acc;
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
            stage_2_trace: stage_2_trace_commit,
            quotient_chunks: quotient_commit,
        };

        // generate the out of domain point and prove polynomial evaluations
        let zeta: ExtVal = challenger.sample_algebra_element();
        let mut round1_openings = vec![];
        let mut round2_openings = vec![];
        let mut round3_openings = vec![];
        log_degrees.iter().zip(quotient_degrees.iter()).for_each(
            |(log_degree, quotient_degree)| {
                let trace_domain = <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                    pcs,
                    1 << log_degree,
                );
                let zeta_next = trace_domain.next_point(zeta).unwrap();
                round1_openings.push(vec![zeta, zeta_next]);
                round2_openings.push(vec![zeta, zeta_next]);
                round3_openings.extend(vec![vec![zeta]; *quotient_degree]);
            },
        );
        let rounds = vec![
            (&stage_1_trace_data, round1_openings),
            (&stage_2_trace_data, round2_openings),
            (&quotient_data, round3_openings),
        ];
        let (opened_values, opening_proof) = pcs.open(rounds, &mut challenger);
        let mut opened_values_iter = opened_values.into_iter();
        let stage_1_opened_values = opened_values_iter.next().unwrap();
        let stage_2_opened_values = opened_values_iter.next().unwrap();
        let quotient_opened_values = opened_values_iter.next().unwrap();
        debug_assert!(opened_values_iter.next().is_none());
        let log_degrees = log_degrees
            .into_iter()
            .map(|n| n.try_into().unwrap())
            .collect();
        Proof {
            commitments,
            intermediate_accumulators,
            log_degrees,
            opening_proof,
            quotient_opened_values,
            stage_1_opened_values,
            stage_2_opened_values,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn quotient_values<A, Mat>(
    air: &A,
    public_values: &[Val],
    trace_domain: Domain,
    quotient_domain: Domain,
    stage_1_on_quotient_domain: &Mat,
    stage_2_on_quotient_domain: &Mat,
    alpha: ExtVal,
    constraint_count: usize,
) -> Vec<ExtVal>
where
    A: for<'a> Air<ProverConstraintFolder<'a>>,
    Mat: Matrix<Val> + Sync,
{
    let quotient_size = quotient_domain.size();
    let stage_1_width = stage_1_on_quotient_domain.width();
    let stage_2_width = stage_2_on_quotient_domain.width();
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
    #[cfg(feature = "parallel")]
    {
        (0..quotient_size)
            .into_par_iter()
            .step_by(PackedVal::WIDTH)
            .flat_map_iter(|i_start| {
                quotient_values_inner(
                    air,
                    public_values,
                    &sels,
                    quotient_size,
                    stage_1_on_quotient_domain,
                    stage_2_on_quotient_domain,
                    stage_1_width,
                    stage_2_width,
                    &alpha_powers,
                    &decomposed_alpha_powers,
                    next_step,
                    i_start,
                )
            })
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..quotient_size)
            .step_by(PackedVal::WIDTH)
            .flat_map_iter(|i_start| {
                quotient_values_inner(
                    air,
                    public_values,
                    &sels,
                    quotient_size,
                    stage_1_on_quotient_domain,
                    stage_2_on_quotient_domain,
                    stage_1_width,
                    stage_2_width,
                    &alpha_powers,
                    &decomposed_alpha_powers,
                    next_step,
                    i_start,
                )
            })
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn quotient_values_inner<A, Mat>(
    air: &A,
    public_values: &[Val],
    sels: &LagrangeSelectors<Vec<Val>>,
    quotient_size: usize,
    stage_1_on_quotient_domain: &Mat,
    stage_2_on_quotient_domain: &Mat,
    stage_1_width: usize,
    stage_2_width: usize,
    alpha_powers: &[BinomialExtensionField<Val, 2>],
    decomposed_alpha_powers: &[Vec<Val>],
    next_step: usize,
    i_start: usize,
) -> impl Iterator<Item = BinomialExtensionField<Val, 2>>
where
    A: for<'a> Air<ProverConstraintFolder<'a>>,
    Mat: Matrix<Val> + Sync,
{
    let i_range = i_start..i_start + PackedVal::WIDTH;

    let is_first_row = *PackedVal::from_slice(&sels.is_first_row[i_range.clone()]);
    let is_last_row = *PackedVal::from_slice(&sels.is_last_row[i_range.clone()]);
    let is_transition = *PackedVal::from_slice(&sels.is_transition[i_range.clone()]);
    let inv_vanishing = *PackedVal::from_slice(&sels.inv_vanishing[i_range]);

    // TODO fix preprocessed
    let preprocessed = RowMajorMatrix::new(vec![], 0);
    let stage_1 = RowMajorMatrix::new(
        stage_1_on_quotient_domain.vertically_packed_row_pair(i_start, next_step),
        stage_1_width,
    );
    let stage_2 = RowMajorMatrix::new(
        stage_2_on_quotient_domain.vertically_packed_row_pair(i_start, next_step),
        stage_2_width,
    );

    let accumulator = PackedExtVal::ZERO;
    let mut folder = ProverConstraintFolder {
        preprocessed: preprocessed.as_view(),
        stage_1: stage_1.as_view(),
        stage_2: stage_2.as_view(),
        public_values,
        is_first_row,
        is_last_row,
        is_transition,
        alpha_powers,
        decomposed_alpha_powers,
        accumulator,
        constraint_index: 0,
    };
    air.eval(&mut folder);

    let quotient = folder.accumulator * inv_vanishing;

    (0..min(quotient_size, PackedVal::WIDTH)).map(move |idx_in_packing| {
        ExtVal::from_basis_coefficients_fn(|coeff_idx| {
            <PackedExtVal as BasedVectorSpace<PackedVal>>::as_basis_coefficients_slice(&quotient)
                [coeff_idx]
                .as_slice()[idx_in_packing]
        })
    })
}
