use crate::{
    builder::symbolic::{SymbolicAirBuilder, get_max_constraint_degree, get_symbolic_constraints},
    ensure_eq,
    lookup::{Lookup, LookupAir},
    types::{Commitment, CommitmentParameters, Committer, ProverData, Val},
};
use p3_air::{Air, BaseAir, BaseAirWithPublicValues};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

/// Each circuit is required to have at least 4 arguments. Namely, the lookup challenge,
/// fingerprint challenge, current accumulator and next accumulator
pub const MIN_IO_SIZE: usize = 4;

pub struct System<A> {
    pub circuits: Vec<Circuit<A>>,
    pub preprocessed_commit: Option<Commitment>,
}

pub struct ProverKey {
    pub preprocessed_data: Option<ProverData>,
}

impl<A: BaseAir<Val> + Air<SymbolicAirBuilder<Val>>> System<A> {
    #[inline]
    pub fn new(
        commitment_parameters: &CommitmentParameters,
        circuits: impl IntoIterator<Item = LookupAir<A>>,
    ) -> (Self, ProverKey) {
        let (circuits, maybe_preprocessed_traces): (Vec<_>, Vec<_>) = circuits
            .into_iter()
            .map(|air| Circuit::from_air(air).unwrap())
            .unzip();
        let committer = Committer::new(commitment_parameters);
        let preprocessed_traces = maybe_preprocessed_traces
            .into_iter()
            .filter_map(|maybe_trace| {
                maybe_trace.map(|trace| {
                    let domain = committer.natural_domain_for_degree(trace.height());
                    (domain, trace)
                })
            })
            .collect::<Vec<_>>();
        let (preprocessed_commit, preprocessed_data) = if !preprocessed_traces.is_empty() {
            let (commit, data) = committer.commit(preprocessed_traces);
            (Some(commit), Some(data))
        } else {
            (None, None)
        };
        let system = Self {
            circuits,
            preprocessed_commit,
        };
        let key = ProverKey { preprocessed_data };
        (system, key)
    }
}

pub struct Circuit<A> {
    pub air: LookupAir<A>,
    pub constraint_count: usize,
    pub max_constraint_degree: usize,
    pub preprocessed_width: usize,
    pub stage_1_width: usize,
    pub stage_2_width: usize,
}

#[derive(Clone)]
pub struct SystemWitness {
    pub traces: Vec<RowMajorMatrix<Val>>,
    pub lookups: Vec<Vec<Vec<Lookup<Val>>>>,
}

impl SystemWitness {
    pub fn from_stage_1<A>(traces: Vec<RowMajorMatrix<Val>>, system: &System<A>) -> Self {
        let lookups = traces
            .iter()
            .zip(system.circuits.iter())
            .map(|(trace, circuit)| {
                trace
                    .row_slices()
                    .map(|row| {
                        circuit
                            .air
                            .lookups
                            .iter()
                            .map(|lookup| lookup.compute_expr(row))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self { traces, lookups }
    }
}

impl<A: BaseAir<Val> + Air<SymbolicAirBuilder<Val>>> Circuit<A> {
    pub fn from_air(air: LookupAir<A>) -> Result<(Self, Option<RowMajorMatrix<Val>>), String> {
        let io_size = air.num_public_values();
        ensure_eq!(io_size, MIN_IO_SIZE, "Incompatible IO size");
        let stage_1_width = air.inner_air.width();
        let stage_2_width = air.stage_2_width();
        let preprocessed_trace = air.preprocessed_trace();
        let preprocessed_width = preprocessed_trace.as_ref().map_or(0, |mat| mat.width());
        let constraint_count = get_symbolic_constraints(
            &air,
            preprocessed_width,
            stage_1_width,
            stage_2_width,
            io_size,
        )
        .len();
        let max_constraint_degree = get_max_constraint_degree(
            &air,
            preprocessed_width,
            stage_1_width,
            stage_2_width,
            io_size,
        );
        let circuit = Self {
            air,
            max_constraint_degree,
            preprocessed_width,
            constraint_count,
            stage_1_width,
            stage_2_width,
        };
        Ok((circuit, preprocessed_trace))
    }
}
