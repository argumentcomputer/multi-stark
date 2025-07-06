use crate::{
    ensure,
    types::{Challenger, ExtVal, Pcs, StarkConfig, Val},
};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::Pcs as PcsTrait;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_uni_stark::{
    ProverConstraintFolder, StarkGenericConfig, SymbolicAirBuilder, get_log_quotient_degree,
};
use p3_util::log2_strict_usize;
use std::collections::BTreeMap as Map;

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
    pub stage1_width: usize,
    pub stage2_width: usize,
}

pub struct CircuitWitness {
    pub stage1: RowMajorMatrix<Val>,
    pub stage2: RowMajorMatrix<ExtVal>,
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

impl<A: BaseAirWithPublicValues<Val>> Circuit<A> {
    pub fn is_well_formed(&self) -> Result<(), String> {
        let width = self.air.width();
        // As of now, only the minimum IO size is supported.
        ensure!(
            self.air.num_public_values() == MIN_IO_SIZE,
            "Incompatible IO size"
        );
        ensure!(
            self.stage1_width + self.stage2_width == width,
            "Incompatible widths"
        );
        Ok(())
    }
}

impl<A: BaseAirWithPublicValues<Val>> System<A> {
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

impl<
    A: BaseAirWithPublicValues<Val>
        + Air<SymbolicAirBuilder<Val>>
        + for<'a> Air<ProverConstraintFolder<'a, StarkConfig>>,
> System<A>
{
    #[allow(unused_variables)]
    #[allow(unused_mut)]
    pub fn prove(&self, config: &StarkConfig, claim: Claim, witness: SystemWitness) -> Proof {
        // initialize pcs and challenger
        let pcs = config.pcs();
        let mut challenger = config.initialise_challenger();
        // compute domains for all circuits
        let mut stage2_traces = vec![];
        let mut log_degrees = vec![];
        for (circuit, witness) in self.circuits.iter().zip(witness.circuits.into_iter()) {
            let air = &circuit.air;
            let trace = witness.stage1;
            // TODO: allow preprocessed tables
            let preprocessed_width = 0;
            // TODO: perhaps implement zero-knowledge. Although the better idea might be
            // to get zero-knowledge through the compression SNARK.
            let is_zk = 0;

            let degree = trace.height();
            let log_degree = log2_strict_usize(degree);
            let log_quotient_degree = get_log_quotient_degree::<Val, A>(
                air,
                preprocessed_width,
                air.num_public_values(),
                is_zk,
            );
            let quotient_degree = 1 << log_quotient_degree;
            let trace_domain =
                <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);
            let (trace_commit, trace_data) =
                <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, [(trace_domain, trace)]);

            challenger.observe(Val::from_u8(log_degree as u8));
            challenger.observe(trace_commit);

            stage2_traces.push(witness.stage2);
            log_degrees.push(log_degree);
        }
        // generate lookup challenges
        let lookup_argument_challenge: ExtVal = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: ExtVal = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);
        // TODO: implement stage 2

        // observe the claim
        challenger.observe(Val::from_usize(
            *self.circuit_names.get(claim.circuit_name).unwrap(),
        ));
        challenger.observe_slice(&claim.args);

        // generate constraint challenge
        let constraint_challenge: ExtVal = challenger.sample_algebra_element();
        todo!()
    }

    #[allow(unused_variables)]
    pub fn verify(&self, proof: Proof) {}
}
