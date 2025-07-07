use crate::{
    ensure,
    types::{Challenger, ExtVal, PackedExtVal, PackedVal, Pcs, StarkConfig, Val},
};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs as PcsTrait, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{
    Domain, ProverConstraintFolder, StarkGenericConfig, SymbolicAirBuilder,
    get_max_constraint_degree, get_symbolic_constraints,
};
use p3_util::log2_strict_usize;
use std::{collections::BTreeMap as Map, iter::once};

pub type Name = &'static str;

/// Each circuit is required to have at least 3 arguments. Namely,
/// the accumulator, the lookup challenge and the fingerprint challenge.
pub const MIN_IO_SIZE: usize = 3;

pub struct System<A> {
    pub circuits: Vec<Circuit<A>>,
    pub circuit_names: Map<Name, usize>,
}

pub struct Circuit<A> {
    pub air: A,
    pub constraint_count: usize,
    pub max_constraint_degree: usize,
    pub preprocessed_width: usize,
    pub stage1_width: usize,
    pub stage2_width: usize,
}

pub struct CircuitWitness {
    pub stage1: RowMajorMatrix<Val>,
    // TODO use `ExtVal` instead of `Val`
    pub stage2: RowMajorMatrix<Val>,
}

pub struct SystemWitness {
    pub circuits: Vec<CircuitWitness>,
}

pub struct Claim {
    pub circuit_name: Name,
    pub args: Vec<Val>,
}

pub struct Proof {
    pub claim: Claim,
}

impl<A: BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>> Circuit<A> {
    pub fn is_well_formed(&self) -> Result<(), String> {
        let io_size = self.air.num_public_values();
        let preprocessed_width = self.air.preprocessed_trace().map_or(0, |mat| mat.width());
        let constraint_count =
            get_symbolic_constraints(&self.air, preprocessed_width, io_size).len();
        let max_constraint_count =
            get_max_constraint_degree(&self.air, preprocessed_width, io_size);
        let width = self.air.width();
        // As of now, only the minimum IO size is supported.
        ensure!(io_size == MIN_IO_SIZE, "Incompatible IO size");
        ensure!(
            self.constraint_count == constraint_count,
            "Incompatible constraint count"
        );
        ensure!(
            self.max_constraint_degree == max_constraint_count,
            "Incompatible constraint degree"
        );
        ensure!(
            self.preprocessed_width == preprocessed_width,
            "Incompatible widths"
        );
        ensure!(
            self.stage1_width + self.stage2_width + self.preprocessed_width == width,
            "Incompatible widths"
        );
        Ok(())
    }
}

impl<A: BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>> System<A> {
    pub fn is_well_formed(&self) -> Result<(), String> {
        ensure!(
            self.circuits.len() == self.circuit_names.len(),
            "Map of names is not well-formed"
        );
        let mut idxs = self.circuit_names.values().copied().collect::<Vec<_>>();
        idxs.sort();
        ensure!(
            idxs == (0..self.circuits.len()).collect::<Vec<_>>(),
            "Map of names is not well-formed"
        );
        self.circuits.iter().try_for_each(|c| c.is_well_formed())?;
        Ok(())
    }
}

impl<A: BaseAirWithPublicValues<Val> + for<'a> Air<ProverConstraintFolder<'a, StarkConfig>>>
    System<A>
{
    pub fn prove(&self, config: &StarkConfig, claim: Claim, witness: SystemWitness) -> Proof {
        // initialize pcs and challenger
        let pcs = config.pcs();
        let mut challenger = config.initialise_challenger();
        // commit to stage 1 traces
        let mut stage1_info = vec![];
        let mut stage2_traces = vec![];
        for witness in witness.circuits.into_iter() {
            let trace = witness.stage1;
            let degree = trace.height();
            let log_degree = log2_strict_usize(degree);
            let trace_domain =
                <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);
            let (trace_commit, trace_data) =
                <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, [(trace_domain, trace)]);
            stage2_traces.push(witness.stage2);
            stage1_info.push((log_degree, trace_domain, trace_data));
            challenger.observe(Val::from_u8(log_degree as u8));
            challenger.observe(trace_commit);
        }
        // generate lookup challenges
        // TODO use `ExtVal` instead of `Val`
        let lookup_argument_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);
        // commit to stage 2 traces
        let mut stage2_info = vec![];
        for trace in stage2_traces.into_iter() {
            let degree = trace.height();
            let log_degree = log2_strict_usize(degree);
            let trace_domain =
                <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);
            let (trace_commit, trace_data) =
                <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, [(trace_domain, trace)]);
            stage2_info.push((log_degree, trace_domain, trace_data));
            challenger.observe(Val::from_u8(log_degree as u8));
            challenger.observe(trace_commit);
        }
        // observe the claim
        let circuit_index = Val::from_usize(*self.circuit_names.get(claim.circuit_name).unwrap());
        challenger.observe(circuit_index);
        challenger.observe_slice(&claim.args);
        // construct the accumulator from the claim
        let claim_iter = claim.args.iter().rev().copied().chain(once(circuit_index));
        let init_acc = fingerprint_reverse(fingerprint_challenge, claim_iter);
        let public_values = vec![lookup_argument_challenge, fingerprint_challenge, init_acc];

        // generate constraint challenge
        let constraint_challenge: ExtVal = challenger.sample_algebra_element();

        // TODO add stage2 traces
        for (circuit, (log_degree, trace_domain, trace_data)) in
            self.circuits.iter().zip(stage1_info.iter())
        {
            let air = &circuit.air;
            // the quotient degree is at most the maximum degree of the constraint minus 1, to
            // account for the zerofier, times the degree of the trace. It is padded to a power of two.
            let quotient_degree = (circuit.max_constraint_degree.max(2) - 1).next_power_of_two();
            let log_quotient_degree = quotient_degree.trailing_zeros() as usize;
            let quotient_domain =
                trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));
            let trace_on_quotient_domain =
                <Pcs as PcsTrait<ExtVal, Challenger>>::get_evaluations_on_domain(
                    pcs,
                    &trace_data,
                    0,
                    quotient_domain,
                );
            let quotient_values = quotient_values(
                air,
                &public_values,
                *trace_domain,
                quotient_domain,
                trace_on_quotient_domain,
                constraint_challenge,
                circuit.constraint_count,
            );
            let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();
            let (quotient_commit, quotient_data) =
                <Pcs as PcsTrait<ExtVal, Challenger>>::commit_quotient(
                    pcs,
                    quotient_domain,
                    quotient_flat,
                    quotient_degree,
                );
            challenger.observe(quotient_commit);
        }
        todo!()
    }

    #[allow(unused_variables)]
    pub fn verify(&self, proof: Proof) {}
}

// Compute a fingerprint of the coefficients in reverse using Horner's method:
fn fingerprint_reverse<F: Field, Iter: Iterator<Item = F>>(r: F, coeffs: Iter) -> F {
    coeffs.fold(F::ZERO, |acc, coeff| acc * r + coeff)
}

// TODO take stage 2 traces and update the accumulator
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
