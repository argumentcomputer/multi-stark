use crate::{
    builder::symbolic::{SymbolicAirBuilder, get_max_constraint_degree, get_symbolic_constraints},
    config::{Com, PcsData, StarkGenericConfig, Val},
    lookup::{LOOKUP_PUBLIC_SIZE, Lookup, LookupAir},
};
use p3_air::{Air, BaseAir};
use p3_challenger::CanObserve;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

/// A multi-circuit STARK system. Contains all circuits together with their
/// shared preprocessed commitment and the protocol configuration.
pub struct System<SC: StarkGenericConfig, A> {
    pub config: SC,
    pub circuits: Vec<Circuit<A, Val<SC>>>,
    /// Commitment to all preprocessed traces (if any circuit has one).
    pub preprocessed_commit: Option<Com<SC>>,
    /// Maps each circuit index to its position within the preprocessed commitment.
    /// `None` if that circuit has no preprocessed trace.
    pub preprocessed_indices: Vec<Option<usize>>,
}

/// Prover-side data that must be retained between system setup and proving.
pub struct ProverKey<SC: StarkGenericConfig> {
    /// PCS prover data for the preprocessed traces.
    pub preprocessed_data: Option<PcsData<SC>>,
}

impl<SC, A> System<SC, A>
where
    SC: StarkGenericConfig,
    A: BaseAir<Val<SC>> + Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>,
{
    #[inline]
    pub fn new(
        config: SC,
        airs: impl IntoIterator<Item = LookupAir<A, Val<SC>>>,
    ) -> (Self, ProverKey<SC>) {
        let pcs = config.pcs();
        let mut circuits = vec![];
        let mut preprocessed_traces = vec![];
        let mut preprocessed_indices = vec![];
        for air in airs {
            let (circuit, maybe_preprocessed_trace) = Circuit::from_air::<SC::Challenge>(air);
            circuits.push(circuit);
            if let Some(preprocessed_trace) = maybe_preprocessed_trace {
                preprocessed_indices.push(Some(preprocessed_traces.len()));
                let domain = pcs.natural_domain_for_degree(preprocessed_trace.height());
                preprocessed_traces.push((domain, preprocessed_trace));
            } else {
                preprocessed_indices.push(None);
            }
        }
        let (preprocessed_commit, preprocessed_data) = if !preprocessed_traces.is_empty() {
            let (commit, data) = pcs.commit(preprocessed_traces);
            (Some(commit), Some(data))
        } else {
            (None, None)
        };
        let system = Self {
            config,
            circuits,
            preprocessed_commit,
            preprocessed_indices,
        };
        let key = ProverKey { preprocessed_data };
        (system, key)
    }
}

impl<SC: StarkGenericConfig, A> System<SC, A> {
    /// Binds the system shape into the Fiat-Shamir transcript. The prover and
    /// the verifier must call this identically, before observing any
    /// commitment, so that transcripts of systems with different circuit
    /// shapes never collide. The protocol parameters are bound separately,
    /// via the challenger seed (see
    /// [`StarkGenericConfig::initialise_challenger`]).
    pub fn observe_shape(&self, challenger: &mut SC::Challenger) {
        let mut observe_usize = |x: usize| challenger.observe(Val::<SC>::from_usize(x));
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
pub struct Circuit<A, F: Field> {
    pub air: LookupAir<A, F>,
    pub constraint_count: usize,
    pub max_constraint_degree: usize,
    pub preprocessed_height: usize,
    pub preprocessed_width: usize,
    pub stage_1_width: usize,
    pub stage_2_width: usize,
}

impl<A, F: Field> Circuit<A, F> {
    /// Degree of the quotient polynomial as a multiple of the trace degree.
    /// Division by the vanishing polynomial reduces the composition
    /// polynomial's degree by 1; the result is padded to a power of two so
    /// the quotient can be split into equally-sized chunks.
    pub fn quotient_degree(&self) -> usize {
        (self.max_constraint_degree.max(2) - 1).next_power_of_two()
    }
}

/// Witness data for the multi-circuit system, comprising stage 1 traces and
/// the concrete lookup values derived from them.
#[derive(Clone)]
pub struct SystemWitness<F: Field> {
    /// Stage 1 (main) execution traces, one per circuit.
    pub traces: Vec<RowMajorMatrix<F>>,
    /// Lookup values per circuit, per row, per lookup.
    pub lookups: Vec<Vec<Vec<Lookup<F>>>>,
}

impl<F: Field> SystemWitness<F> {
    /// Builds the witness from the stage 1 traces, computing the concrete
    /// lookup values for each row of each circuit.
    ///
    /// # Panics
    /// Panics if the number of traces differs from the number of circuits, or
    /// if a circuit with a preprocessed trace receives a main trace of a
    /// different height (both traces are opened on the same domain, so their
    /// heights must match; the rows would otherwise be silently truncated).
    pub fn from_stage_1<SC, A>(traces: Vec<RowMajorMatrix<F>>, system: &System<SC, A>) -> Self
    where
        SC: StarkGenericConfig,
        SC::Pcs: Pcs<SC::Challenge, SC::Challenger, Domain: PolynomialSpace<Val = F>>,
    {
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

impl<A, F: Field> Circuit<A, F> {
    pub fn from_air<EF>(air: LookupAir<A, F>) -> (Self, Option<RowMajorMatrix<F>>)
    where
        EF: ExtensionField<F>,
        A: BaseAir<F> + Air<SymbolicAirBuilder<F, EF>>,
    {
        // as of now, we assume no public values apart from the lookup values
        let io_size = 0;
        let stage_1_width = air.inner_air.width();
        let stage_2_width = air.stage_2_width();
        let preprocessed_trace = air.preprocessed_trace();
        let preprocessed_height = preprocessed_trace.as_ref().map_or(0, |mat| mat.height());
        let preprocessed_width = preprocessed_trace.as_ref().map_or(0, |mat| mat.width());
        let symbolic_constraints = get_symbolic_constraints::<F, EF, _>(
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
    use crate::types::{CommitmentParameters, FriParameters, GoldilocksKeccakConfig, Val};
    use p3_air::AirBuilder;

    /// A trivial AIR with a preprocessed trace of 4 rows and no constraints.
    struct Preprocessed;

    impl<F: Field> BaseAir<F> for Preprocessed {
        fn width(&self) -> usize {
            1
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            Some(RowMajorMatrix::new(vec![F::ZERO; 4], 1))
        }
    }

    impl<AB: AirBuilder> Air<AB> for Preprocessed
    where
        AB::F: Field,
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
        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 64,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        };
        let config = GoldilocksKeccakConfig::new(commitment_parameters, fri_parameters);
        let (system, _key) = System::new(config, [LookupAir::new(Preprocessed, vec![])]);
        // The main trace has 8 rows but the preprocessed trace has 4. This
        // must panic instead of silently truncating the lookup rows.
        let trace = RowMajorMatrix::new(vec![Val::ZERO; 8], 1);
        SystemWitness::from_stage_1(vec![trace], &system);
    }
}
