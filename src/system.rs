use crate::{
    builder::symbolic::{SymbolicAirBuilder, get_max_constraint_degree, get_symbolic_constraints},
    lookup::{LOOKUP_PUBLIC_SIZE, Lookup, LookupAir},
    types::{
        Challenger, Commitment, CommitmentParameters, Committer, FriParameters, ProverData, Val,
    },
};
use p3_air::{Air, BaseAir};
use p3_challenger::CanObserve;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{Matrix, dense::RowMajorMatrix};

/// A multi-circuit STARK system. Contains all circuits together with their
/// shared preprocessed commitment and commitment parameters.
pub struct System<A> {
    pub commitment_parameters: CommitmentParameters,
    pub circuits: Vec<Circuit<A>>,
    /// Commitment to all preprocessed traces (if any circuit has one).
    pub preprocessed_commit: Option<Commitment>,
    /// Maps each circuit index to its position within the preprocessed commitment.
    /// `None` if that circuit has no preprocessed trace.
    pub preprocessed_indices: Vec<Option<usize>>,
}

/// Prover-side data that must be retained between system setup and proving.
pub struct ProverKey {
    /// PCS prover data for the preprocessed traces.
    pub preprocessed_data: Option<ProverData>,
}

impl<A: BaseAir<Val> + Air<SymbolicAirBuilder>> System<A> {
    #[inline]
    pub fn new(
        commitment_parameters: CommitmentParameters,
        airs: impl IntoIterator<Item = LookupAir<A>>,
    ) -> (Self, ProverKey) {
        let committer = Committer::new(commitment_parameters);
        let mut circuits = vec![];
        let mut preprocessed_traces = vec![];
        let mut preprocessed_indices = vec![];
        for air in airs {
            let (circuit, maybe_preprocessed_trace) = Circuit::from_air(air);
            circuits.push(circuit);
            if let Some(preprocessed_trace) = maybe_preprocessed_trace {
                preprocessed_indices.push(Some(preprocessed_traces.len()));
                let domain = committer.natural_domain_for_degree(preprocessed_trace.height());
                preprocessed_traces.push((domain, preprocessed_trace));
            } else {
                preprocessed_indices.push(None);
            }
        }
        let (preprocessed_commit, preprocessed_data) = if !preprocessed_traces.is_empty() {
            let (commit, data) = committer.commit(preprocessed_traces);
            (Some(commit), Some(data))
        } else {
            (None, None)
        };
        let system = Self {
            commitment_parameters,
            circuits,
            preprocessed_commit,
            preprocessed_indices,
        };
        let key = ProverKey { preprocessed_data };
        (system, key)
    }
}

impl<A> System<A> {
    /// Binds the protocol parameters and the system shape into the Fiat-Shamir
    /// transcript. The prover and the verifier must call this identically,
    /// before observing any commitment, so that transcripts of systems with
    /// different parameters or circuit shapes never collide.
    pub fn observe_shape(&self, fri_parameters: &FriParameters, challenger: &mut Challenger) {
        let mut observe_usize = |x: usize| challenger.observe(Val::from_usize(x));
        observe_usize(self.commitment_parameters.log_blowup);
        observe_usize(self.commitment_parameters.cap_height);
        observe_usize(fri_parameters.log_final_poly_len);
        observe_usize(fri_parameters.max_log_arity);
        observe_usize(fri_parameters.num_queries);
        observe_usize(fri_parameters.commit_proof_of_work_bits);
        observe_usize(fri_parameters.query_proof_of_work_bits);
        observe_usize(self.circuits.len());
        for circuit in &self.circuits {
            observe_usize(circuit.constraint_count);
            observe_usize(circuit.max_constraint_degree);
            observe_usize(circuit.preprocessed_height);
            observe_usize(circuit.preprocessed_width);
            observe_usize(circuit.stage_1_width);
            observe_usize(circuit.stage_2_width);
        }
    }
}

/// A single circuit within the system, wrapping an AIR together with
/// precomputed metadata used by the prover and verifier.
pub struct Circuit<A> {
    pub air: LookupAir<A>,
    pub constraint_count: usize,
    pub max_constraint_degree: usize,
    pub preprocessed_height: usize,
    pub preprocessed_width: usize,
    pub stage_1_width: usize,
    pub stage_2_width: usize,
}

/// Witness data for the multi-circuit system, comprising stage 1 traces and
/// the concrete lookup values derived from them.
#[derive(Clone)]
pub struct SystemWitness {
    /// Stage 1 (main) execution traces, one per circuit.
    pub traces: Vec<RowMajorMatrix<Val>>,
    /// Lookup values per circuit, per row, per lookup.
    pub lookups: Vec<Vec<Vec<Lookup<Val>>>>,
}

impl SystemWitness {
    /// Builds the witness from the stage 1 traces, computing the concrete
    /// lookup values for each row of each circuit.
    ///
    /// # Panics
    /// Panics if the number of traces differs from the number of circuits, or
    /// if a circuit with a preprocessed trace receives a main trace of a
    /// different height (both traces are opened on the same domain, so their
    /// heights must match; the rows would otherwise be silently truncated).
    pub fn from_stage_1<A>(traces: Vec<RowMajorMatrix<Val>>, system: &System<A>) -> Self {
        assert_eq!(
            traces.len(),
            system.circuits.len(),
            "expected one trace per circuit"
        );
        let lookups = traces
            .iter()
            .zip(system.circuits.iter())
            .enumerate()
            .map(|(circuit_idx, (trace, circuit))| {
                if let Some(preprocessed) = &circuit.air.preprocessed {
                    assert_eq!(
                        trace.height(),
                        preprocessed.height(),
                        "circuit {circuit_idx}: main trace height must equal preprocessed trace height"
                    );
                    trace
                        .row_slices()
                        .zip(preprocessed.row_slices())
                        .map(|(row, preprocessed_row)| {
                            circuit
                                .air
                                .lookups
                                .iter()
                                .map(|lookup| lookup.compute_expr(row, Some(preprocessed_row)))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                } else {
                    trace
                        .row_slices()
                        .map(|row| {
                            circuit
                                .air
                                .lookups
                                .iter()
                                .map(|lookup| lookup.compute_expr(row, None))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                }
            })
            .collect::<Vec<_>>();
        Self { traces, lookups }
    }
}

impl<A: BaseAir<Val> + Air<SymbolicAirBuilder>> Circuit<A> {
    pub fn from_air(air: LookupAir<A>) -> (Self, Option<RowMajorMatrix<Val>>) {
        // as of now, we assume no public values apart from the lookup values
        let io_size = 0;
        let stage_1_width = air.inner_air.width();
        let stage_2_width = air.stage_2_width();
        let preprocessed_trace = air.preprocessed_trace();
        let preprocessed_height = preprocessed_trace.as_ref().map_or(0, |mat| mat.height());
        let preprocessed_width = preprocessed_trace.as_ref().map_or(0, |mat| mat.width());
        let symbolic_constraints = get_symbolic_constraints(
            &air,
            preprocessed_width,
            stage_1_width,
            stage_2_width,
            io_size,
            LOOKUP_PUBLIC_SIZE,
        );
        let constraint_count = symbolic_constraints.len();
        let max_constraint_degree = get_max_constraint_degree(&symbolic_constraints);
        let circuit = Self {
            air,
            max_constraint_degree,
            preprocessed_height,
            preprocessed_width,
            constraint_count,
            stage_1_width,
            stage_2_width,
        };
        (circuit, preprocessed_trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_air::AirBuilder;

    /// A trivial AIR with a preprocessed trace of 4 rows and no constraints.
    struct Preprocessed;

    impl<F: p3_field::Field> BaseAir<F> for Preprocessed {
        fn width(&self) -> usize {
            1
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            Some(RowMajorMatrix::new(vec![F::ZERO; 4], 1))
        }
    }

    impl<AB: AirBuilder> Air<AB> for Preprocessed
    where
        AB::F: p3_field::Field,
    {
        fn eval(&self, _builder: &mut AB) {}
    }

    #[test]
    #[should_panic(expected = "preprocessed trace height")]
    fn mismatched_preprocessed_height_panics() {
        let commitment_parameters = CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        };
        let (system, _key) = System::new(
            commitment_parameters,
            [LookupAir::new(Preprocessed, vec![])],
        );
        // The main trace has 8 rows but the preprocessed trace has 4. This
        // must panic instead of silently truncating the lookup rows.
        let trace = RowMajorMatrix::new(vec![Val::ZERO; 8], 1);
        SystemWitness::from_stage_1(vec![trace], &system);
    }
}
