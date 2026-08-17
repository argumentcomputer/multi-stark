//! Multi-circuit STARK system.
//!
//! Circuits are compiled constraint data ([`crate::graph::ConstraintGraph`]) rather
//! than AIRs, so there is no `A` type parameter. `System::new` derives each
//! circuit's stage-2 and public-input layout from its lookups and the
//! challenge field's extension degree, and compiles the user constraints
//! plus the lookup-expression prefix. The logUp constraints are NOT
//! compiled: prover and verifier evaluate them directly
//! ([`crate::lookup::logup_constraint_values`]), folding their values after
//! the user roots.

use crate::traits::{ExtensionOf, Field};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

use crate::config::{Com, PcsData, ProofConfig, Val};
use crate::lookup::LookupValues;

use crate::eval::VarValues;
use crate::expr::{CircuitSpec, Expr, ExtExpr};
use crate::graph::{ConstraintGraph, ExtensionParams, compile};
use crate::lookup::{Lookup, logup_constraint_count, logup_max_degree, num_publics, stage2_width};
use crate::traits::{Pcs, Transcript};

/// User-facing definition of one circuit: main-trace width, optional
/// preprocessed trace, base and extension constraints, and lookups. The
/// stage-2 width and public-input count are derived (from the lookups and
/// the challenge field's extension degree), not supplied here.
pub struct CircuitInputs<F: Field> {
    pub main_width: usize,
    pub preprocessed: Option<RowMajorMatrix<F>>,
    pub constraints: Vec<Expr<F>>,
    pub ext_constraints: Vec<ExtExpr<F>>,
    pub lookups: Vec<Lookup<Expr<F>>>,
    /// How many logUp messages share one chained-accumulator step (and one
    /// committed accumulator column): stage-2 width becomes `⌈L/k⌉·D` at
    /// the cost of raising the logUp constraint degree to roughly
    /// `k·deg(args) + 1`. Circuit-local because the degree budget depends
    /// on each circuit's message degrees; `1` (the default) is the
    /// ungrouped chained argument. Bounded by
    /// [`crate::lookup::MAX_LOOKUP_GROUP`].
    pub lookup_group_size: usize,
}

impl<F: Field> Default for CircuitInputs<F> {
    fn default() -> Self {
        Self {
            main_width: 0,
            preprocessed: None,
            constraints: vec![],
            ext_constraints: vec![],
            lookups: vec![],
            lookup_group_size: 1,
        }
    }
}

/// A compiled circuit within the system, with the metadata the prover and
/// verifier need. The preprocessed trace is retained for witness-time
/// lookup evaluation (it is also committed at setup).
pub struct Circuit<F: Field> {
    pub graph: ConstraintGraph<F>,
    pub main_width: usize,
    pub preprocessed: Option<RowMajorMatrix<F>>,
    pub preprocessed_width: usize,
    pub preprocessed_height: usize,
    pub num_lookups: usize,
    /// Stage-2 trace width in flattened base columns: `(1 + num_lookups)·D`.
    pub stage_2_width: usize,
    /// Public-input width in base coordinates: `4·D`.
    pub num_publics: usize,
    /// Lookup group size (see [`CircuitInputs::lookup_group_size`]).
    pub lookup_group_size: usize,
    /// Total constraint count folded with α: the graph's user-constraint
    /// roots plus the directly-evaluated logUp values, `⌈L/k⌉·D`.
    pub constraint_count: usize,
    /// Maximum degree multiple over user constraints AND the logUp
    /// constraints (the latter computed analytically, not compiled).
    pub max_constraint_degree: usize,
}

impl<F: Field> Circuit<F> {
    /// Number of constraint values folded with α (user roots + logUp).
    pub fn constraint_count(&self) -> usize {
        self.constraint_count
    }

    pub fn max_constraint_degree(&self) -> usize {
        self.max_constraint_degree
    }

    /// Degree of the quotient polynomial as a multiple of the trace degree.
    /// Division by the vanishing polynomial reduces the composition
    /// polynomial's degree by 1; the result is padded to a power of two so
    /// the quotient can be split into equally-sized chunks.
    pub fn quotient_degree(&self) -> usize {
        (self.max_constraint_degree().max(2) - 1).next_power_of_two()
    }
}

/// A multi-circuit STARK system over compiled constraint circuits. Contains
/// all circuits together with their shared preprocessed commitment and the
/// protocol configuration.
pub struct System<SC: ProofConfig> {
    pub config: SC,
    pub circuits: Vec<Circuit<Val<SC>>>,
    /// Commitment to all preprocessed traces (if any circuit has one).
    pub preprocessed_commit: Option<Com<SC>>,
    /// Maps each circuit index to its position within the preprocessed
    /// commitment; `None` if that circuit has no preprocessed trace.
    pub preprocessed_indices: Vec<Option<usize>>,
}

/// Prover-side data retained between system setup and proving.
pub struct ProverKey<SC: ProofConfig> {
    /// PCS prover data for the preprocessed traces.
    pub preprocessed_data: Option<PcsData<SC>>,
}

impl<SC: ProofConfig> System<SC> {
    /// Builds the system from per-circuit inputs.
    ///
    /// # Panics
    /// Panics if a circuit fails to compile, or if its constraint degree
    /// exceeds what the PCS can serve (`max_quotient_degree`).
    pub fn new(
        config: SC,
        inputs: impl IntoIterator<Item: Into<CircuitInputs<Val<SC>>>>,
    ) -> (Self, ProverKey<SC>) {
        let pcs = config.pcs();
        let params = extension_params::<SC>();
        let d = params.degree;

        let mut circuits = vec![];
        let mut preprocessed_traces = vec![];
        let mut preprocessed_indices = vec![];
        for (i, input) in inputs.into_iter().enumerate() {
            let input = input.into();
            let num_lookups = input.lookups.len();
            let group_size = input.lookup_group_size.max(1);
            assert!(
                group_size <= crate::lookup::MAX_LOOKUP_GROUP,
                "circuit {i}: lookup_group_size {group_size} exceeds MAX_LOOKUP_GROUP",
            );
            let preprocessed_width = input.preprocessed.as_ref().map_or(0, |m| m.width());
            let preprocessed_height = input.preprocessed.as_ref().map_or(0, |m| m.height());
            let s2_width = stage2_width(num_lookups, group_size, d);
            let n_publics = num_publics(d);

            // The logUp constraints are NOT compiled into the graph: the
            // prover and verifier evaluate them directly (see
            // `lookup::logup_constraint_values`), folded after the user
            // roots in the canonical protocol order. The graph carries only
            // the user constraints and the lookup-expression prefix.
            let spec = CircuitSpec {
                main_width: input.main_width,
                preprocessed_width,
                stage2_width: s2_width,
                num_publics: n_publics,
                constraints: input.constraints,
                ext_constraints: input.ext_constraints,
                lookups: input.lookups,
            };
            let graph = compile(&spec, &params)
                .unwrap_or_else(|e| panic!("circuit {i}: constraint compilation failed: {e:?}"));

            let constraint_count =
                graph.zeros.len() + logup_constraint_count(num_lookups, group_size, d);
            let max_constraint_degree = (graph
                .max_constraint_degree
                .max(logup_max_degree(&graph, group_size)))
                as usize;
            let circuit = Circuit {
                graph,
                main_width: input.main_width,
                preprocessed: input.preprocessed,
                preprocessed_width,
                preprocessed_height,
                num_lookups,
                stage_2_width: s2_width,
                num_publics: n_publics,
                lookup_group_size: group_size,
                constraint_count,
                max_constraint_degree,
            };
            // The prover obtains trace evaluations on the quotient domain
            // from the PCS, which can only serve domains up to
            // `max_quotient_degree` times the trace domain (the blowup
            // factor for FRI). Beyond that, proving would silently produce
            // invalid proofs, so reject the circuit upfront.
            assert!(
                circuit.quotient_degree() <= config.max_quotient_degree(),
                "circuit {i}: constraint degree {} needs quotient degree {}, but the PCS only \
                 supports {}; increase log_blowup or lower the constraint degree",
                circuit.max_constraint_degree(),
                circuit.quotient_degree(),
                config.max_quotient_degree(),
            );

            if let Some(preprocessed) = &circuit.preprocessed {
                preprocessed_indices.push(Some(preprocessed_traces.len()));
                let domain = pcs.natural_domain_for_degree(preprocessed.height());
                preprocessed_traces.push((domain, preprocessed.clone()));
            } else {
                preprocessed_indices.push(None);
            }
            circuits.push(circuit);
        }

        let (preprocessed_commit, preprocessed_data) = if preprocessed_traces.is_empty() {
            (None, None)
        } else {
            let (commit, data) = pcs.commit(preprocessed_traces);
            (Some(commit), Some(data))
        };
        let system = Self {
            config,
            circuits,
            preprocessed_commit,
            preprocessed_indices,
        };
        (system, ProverKey { preprocessed_data })
    }

    /// Binds the system shape into the Fiat-Shamir transcript. The prover and
    /// the verifier must call this identically, before observing any
    /// commitment, so that transcripts of systems with different circuit
    /// shapes never collide. The protocol parameters are bound separately,
    /// via the challenger seed (see
    /// [`ProofConfig::initialise_challenger`]).
    pub fn observe_shape(&self, challenger: &mut SC::Challenger) {
        let mut observe = |x: usize| challenger.observe_field(Val::<SC>::from_usize(x));
        observe(self.circuits.len());
        for circuit in &self.circuits {
            observe(circuit.constraint_count());
            observe(circuit.max_constraint_degree());
            observe(circuit.preprocessed_height);
            observe(circuit.preprocessed_width);
            observe(circuit.main_width);
            observe(circuit.stage_2_width);
            // The group size changes the constraint STRUCTURE (count and
            // degree alone don't determine it), so bind it like the widths.
            observe(circuit.lookup_group_size);
        }
    }
}

/// Witness data for the system: stage-1 traces and the concrete lookup
/// values derived from them.
#[derive(Clone)]
pub struct SystemWitness<F: Field> {
    /// Stage 1 (main) execution traces, one per circuit.
    pub traces: Vec<RowMajorMatrix<F>>,
    /// Lookup values per circuit, stored flat.
    pub lookups: Vec<LookupValues<F>>,
}

impl<F: Field> SystemWitness<F> {
    /// Builds the witness from the stage-1 traces, computing each circuit's
    /// lookup values by sweeping the compiled lookup prefix over its rows.
    ///
    /// # Panics
    /// Panics if the number of traces differs from the number of circuits, or
    /// if a circuit with a preprocessed trace receives a main trace of a
    /// different height (both traces are opened on the same domain, so their
    /// heights must match; the rows would otherwise be silently truncated).
    pub fn from_stage_1<SC>(traces: Vec<RowMajorMatrix<F>>, system: &System<SC>) -> Self
    where
        SC: ProofConfig,
        SC::Pcs: Pcs<F = F>,
    {
        assert_eq!(
            traces.len(),
            system.circuits.len(),
            "expected one trace per circuit"
        );
        let lookups = traces
            .iter()
            .zip(&system.circuits)
            .enumerate()
            .map(|(i, (trace, circuit))| {
                if let Some(preprocessed) = &circuit.preprocessed {
                    assert_eq!(
                        trace.height(),
                        preprocessed.height(),
                        "circuit {i}: main trace height must equal preprocessed trace height"
                    );
                }
                compute_lookup_values(circuit, trace)
            })
            .collect();
        Self { traces, lookups }
    }
}

/// Computes a circuit's concrete lookup values by sweeping the compiled
/// lookup prefix over each row (with wrap-around for the next-row window).
fn compute_lookup_values<F: Field>(
    circuit: &Circuit<F>,
    trace: &RowMajorMatrix<F>,
) -> LookupValues<F> {
    let height = trace.height();
    let slot_widths: Vec<usize> = circuit
        .graph
        .lookups
        .iter()
        .map(|lookup| lookup.args.len())
        .collect();
    // No rows, or no lookups: nothing to sweep, but preserve num_lookups.
    if height == 0 || slot_widths.is_empty() {
        return LookupValues::builder(height, &slot_widths).finish();
    }

    let preprocessed = circuit.preprocessed.as_ref();
    let empty: [F; 0] = [];
    let mut builder = LookupValues::builder(height, &slot_widths);
    let mut buf = Vec::new();
    let mut args = Vec::new();
    let mut writers = builder.rows_mut();
    for (r, writer) in writers.iter_mut().enumerate() {
        let r_next = (r + 1) % height;
        let main_cur = trace.row_slice(r).unwrap();
        let main_next = trace.row_slice(r_next).unwrap();
        let preprocessed_rows =
            preprocessed.map(|pp| (pp.row_slice(r).unwrap(), pp.row_slice(r_next).unwrap()));
        let (pp_cur, pp_next): (&[F], &[F]) = match &preprocessed_rows {
            Some((cur, next)) => (cur, next),
            None => (&empty, &empty),
        };
        let view = VarValues {
            preprocessed: [pp_cur, pp_next],
            main: [&main_cur, &main_next],
            stage2: [&empty, &empty],
            publics: &empty,
            is_first_row: if r == 0 { F::ONE } else { F::ZERO },
            is_last_row: if r == height - 1 { F::ONE } else { F::ZERO },
            is_transition: if r == height - 1 { F::ZERO } else { F::ONE },
        };
        circuit.graph.sweep_lookup_prefix(&view, &mut buf);
        for (slot, lookup) in circuit.graph.lookups.iter().enumerate() {
            let multiplicity = buf[lookup.multiplicity.index()];
            args.clear();
            args.extend(lookup.args.iter().map(|a| buf[a.index()]));
            // The multiplicity expression already carries its sign, so store
            // it as-is (push semantics).
            writer.push(slot, multiplicity, &args);
        }
    }
    drop(writers);
    builder.finish()
}

/// The binomial extension parameters of the challenge field, read off
/// the `ExtensionOf` constants (`X^D = W`; `W` is unused when `D = 1`).
pub(crate) fn extension_params<SC: ProofConfig>() -> ExtensionParams<Val<SC>> {
    let d = <SC::Challenge as ExtensionOf<Val<SC>>>::D;
    ExtensionParams {
        degree: d,
        w: <SC::Challenge as ExtensionOf<Val<SC>>>::W,
        karatsuba: d == 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p3_adapter::LookupAir;
    use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};

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

    /// A degree-5 constraint: `x^5 == y`.
    struct HighDegreeAir;

    impl<F> BaseAir<F> for HighDegreeAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<AB> Air<AB> for HighDegreeAir
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            let x = local[0];
            builder.assert_eq(x * x * x * x * x, local[1]);
        }
    }

    /// The prover can only evaluate constraints on a domain `2^log_blowup`
    /// times the trace domain, so the constraint degree is bounded by
    /// `2^log_blowup + 1` (degree 3 at `log_blowup = 1`). Exceeding it used
    /// to silently produce invalid proofs; it must be rejected at setup.
    #[test]
    #[should_panic(expected = "needs quotient degree 4, but the PCS only supports 2")]
    fn excessive_constraint_degree_rejected() {
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
        let config = GoldilocksBlake3Config::new(commitment_parameters, fri_parameters);
        System::new(config, [LookupAir::new(HighDegreeAir, vec![])]);
    }

    /// The same degree-5 circuit is fine at `log_blowup = 2` (quotient
    /// degree 4 = blowup factor 4), end to end.
    #[test]
    fn high_degree_constraint_with_larger_blowup() {
        let commitment_parameters = CommitmentParameters {
            log_blowup: 2,
            cap_height: 0,
        };
        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 40,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        };
        let config = GoldilocksBlake3Config::new(commitment_parameters, fri_parameters);
        let (system, key) = System::new(config, [LookupAir::new(HighDegreeAir, vec![])]);
        let f = Val::from_u32;
        let trace = RowMajorMatrix::new(vec![f(2), f(32), f(1), f(1), f(3), f(243), f(0), f(0)], 2);
        let witness = SystemWitness::from_stage_1(vec![trace], &system);
        let no_claims: &[&[Val]] = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness);
        system.verify_multiple_claims(no_claims, &proof).unwrap();
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
        let config = GoldilocksBlake3Config::new(commitment_parameters, fri_parameters);
        let (system, _key) = System::new(config, [LookupAir::new(Preprocessed, vec![])]);
        // The main trace has 8 rows but the preprocessed trace has 4. This
        // must panic instead of silently truncating the lookup rows.
        let trace =
            RowMajorMatrix::new(vec![<Val as p3_field::PrimeCharacteristicRing>::ZERO; 8], 1);
        SystemWitness::from_stage_1(vec![trace], &system);
    }
}
