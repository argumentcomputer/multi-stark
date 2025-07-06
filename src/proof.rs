use crate::types::{Challenger, ExtVal, PackedExtVal, PackedVal, Pcs, StarkConfig, Val};

use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs as PcsTrait, PolynomialSpace};
use p3_field::{BasedVectorSpace, PackedValue, PrimeCharacteristicRing};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{
    Domain, ProverConstraintFolder, StarkGenericConfig, SymbolicAirBuilder, SymbolicExpression,
    get_symbolic_constraints,
};
use p3_util::{log2_ceil_usize, log2_strict_usize};

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

    let symbolic_constraints = get_symbolic_constraints(air, 0, public_values.len());
    let constraint_count = symbolic_constraints.len();
    let constraint_degree = symbolic_constraints
        .iter()
        .map(SymbolicExpression::degree_multiple)
        .max()
        .unwrap_or(0);

    let log_quotient_degree = log2_ceil_usize(constraint_degree - 1);
    let quotient_degree = 1 << log_quotient_degree;

    let pcs: &Pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    let trace_domain =
        <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);
    let (trace_commit, trace_data) =
        <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, [(trace_domain, trace)]);

    challenger.observe(Val::from_u8(log_degree as u8));
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
