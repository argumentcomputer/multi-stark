//! Multi-circuit STARK system.
//!
//! Circuits are compiled constraint data ([`crate::graph::ConstraintGraph`]) rather
//! than AIRs, so there is no `A` type parameter. `System::new` derives each
//! circuit's stage-2 and public-input layout from its lookups and the
//! challenge field's extension degree, appends the synthesized lookup
//! constraints, and compiles.

use p3_challenger::CanObserve;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

use crate::config::{Com, PcsData, StarkGenericConfig, Val};
use crate::lookup::LookupValues;

use crate::eval::VarValues;
use crate::expr::{CircuitSpec, Expr, ExtExpr, RowOffset, Source};
use crate::graph::{ConstraintGraph, ExtensionParams, Node, compile};
use crate::lookup::{Lookup, num_publics, stage2_width, synthesize_lookups};

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
    /// Whether this circuit's constraints or lookups reference the NEXT row
    /// of the main or preprocessed trace. When `false` (the default), those
    /// traces are opened at ζ only, halving their share of the opening cost;
    /// `System::new` rejects the circuit if the compiled constraints
    /// reference a next-row column anyway. Stage-2 columns are unaffected:
    /// the lookup accumulator's transition constraint always needs ζ·g, so
    /// stage-2 is always opened at both points.
    pub uses_next_row: bool,
}

impl<F: Field> Default for CircuitInputs<F> {
    fn default() -> Self {
        Self {
            main_width: 0,
            preprocessed: None,
            constraints: vec![],
            ext_constraints: vec![],
            lookups: vec![],
            uses_next_row: false,
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
    /// Whether the main/preprocessed traces are opened at ζ·g in addition
    /// to ζ (see [`CircuitInputs::uses_next_row`]).
    pub uses_next_row: bool,
}

impl<F: Field> Circuit<F> {
    /// Number of constraint roots (after canonicalization).
    pub fn constraint_count(&self) -> usize {
        self.graph.zeros.len()
    }

    pub fn max_constraint_degree(&self) -> usize {
        self.graph.max_constraint_degree as usize
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
pub struct System<SC: StarkGenericConfig> {
    pub config: SC,
    pub circuits: Vec<Circuit<Val<SC>>>,
    /// Commitment to all preprocessed traces (if any circuit has one).
    pub preprocessed_commit: Option<Com<SC>>,
    /// Maps each circuit index to its position within the preprocessed
    /// commitment; `None` if that circuit has no preprocessed trace.
    pub preprocessed_indices: Vec<Option<usize>>,
}

/// Prover-side data retained between system setup and proving.
pub struct ProverKey<SC: StarkGenericConfig> {
    /// PCS prover data for the preprocessed traces.
    pub preprocessed_data: Option<PcsData<SC>>,
}

impl<SC: StarkGenericConfig> System<SC> {
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
            let preprocessed_width = input.preprocessed.as_ref().map_or(0, |m| m.width());
            let preprocessed_height = input.preprocessed.as_ref().map_or(0, |m| m.height());
            let s2_width = stage2_width(num_lookups, d);
            let n_publics = num_publics(d);

            let mut ext_constraints = input.ext_constraints;
            ext_constraints.extend(synthesize_lookups(&input.lookups, d));
            let spec = CircuitSpec {
                main_width: input.main_width,
                preprocessed_width,
                stage2_width: s2_width,
                num_publics: n_publics,
                constraints: input.constraints,
                ext_constraints,
                lookups: input.lookups,
            };
            let graph = compile(&spec, &params)
                .unwrap_or_else(|e| panic!("circuit {i}: constraint compilation failed: {e:?}"));

            // With `uses_next_row = false` the main/preprocessed traces are
            // opened at ζ only, so a compiled constraint referencing a
            // next-row column would have no opened value to evaluate
            // against — reject at setup rather than produce invalid proofs.
            if !input.uses_next_row {
                let refs_next = graph.nodes.iter().any(|node| {
                    matches!(node, Node::Var(col)
                        if col.offset == RowOffset::Next
                            && matches!(col.source, Source::Main | Source::Preprocessed))
                });
                assert!(
                    !refs_next,
                    "circuit {i}: constraints or lookups reference the next row of the \
                     main/preprocessed trace, but `uses_next_row` is false; set it to true",
                );
            }

            let circuit = Circuit {
                graph,
                main_width: input.main_width,
                preprocessed: input.preprocessed,
                preprocessed_width,
                preprocessed_height,
                num_lookups,
                stage_2_width: s2_width,
                num_publics: n_publics,
                uses_next_row: input.uses_next_row,
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
    /// [`StarkGenericConfig::initialise_challenger`]).
    pub fn observe_shape(&self, challenger: &mut SC::Challenger) {
        let mut observe = |x: usize| challenger.observe(Val::<SC>::from_usize(x));
        observe(self.circuits.len());
        for circuit in &self.circuits {
            observe(circuit.constraint_count());
            observe(circuit.max_constraint_degree());
            observe(circuit.preprocessed_height);
            observe(circuit.preprocessed_width);
            observe(circuit.main_width);
            observe(circuit.stage_2_width);
            // The opening shape depends on it, so bind it like the widths.
            observe(usize::from(circuit.uses_next_row));
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

/// Extracts the binomial extension parameters of the challenge field
/// generically: the degree is `Challenge::DIMENSION`, and the modulus
/// constant `W` (with `X^D = W`) is recovered by evaluating `X^D` and
/// reading its base coordinate — no dependence on a concrete field type.
fn extension_params<SC: StarkGenericConfig>() -> ExtensionParams<Val<SC>> {
    let d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
    let x = <SC::Challenge as BasedVectorSpace<Val<SC>>>::ith_basis_element(1)
        .expect("challenge field must have extension degree >= 2");
    let x_pow_d = x.powers().nth(d).expect("powers iterator is infinite");
    let coords = x_pow_d.as_basis_coefficients_slice();
    debug_assert!(
        coords[1..].iter().all(|c| c.is_zero()),
        "challenge field is not a binomial extension: X^D is not a base element"
    );
    ExtensionParams {
        degree: d,
        w: coords[0],
        karatsuba: d == 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p3_adapter::LookupAir;
    use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};

    /// A Fibonacci-style transition AIR: next row = (b, a + b). Reads the
    /// next row, so the adapter must derive `uses_next_row = true` and the
    /// prover must open its main trace at ζ·g.
    struct FibAir;

    impl<F> BaseAir<F> for FibAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<AB> Air<AB> for FibAir
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            let next = main.next_slice();
            let (a, b) = (local[0], local[1]);
            let (a_next, b_next) = (next[0], next[1]);
            let mut when_transition = builder.when_transition();
            when_transition.assert_eq(a_next, b);
            when_transition.assert_eq(b_next, a + b);
        }
    }

    /// Next-row transition constraints prove and verify end to end: the
    /// adapter derives `uses_next_row`, and the main trace is opened at
    /// both ζ and ζ·g while single-row circuits in the same system are not.
    #[test]
    fn next_row_circuit_proves() {
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
        let (system, key) = System::new(
            config,
            [
                CircuitInputs::from(LookupAir::new(FibAir, vec![])),
                // A single-row circuit alongside, to cover the mixed case.
                CircuitInputs::from(LookupAir::new(Preprocessed, vec![])),
            ],
        );
        assert!(system.circuits[0].uses_next_row);
        assert!(!system.circuits[1].uses_next_row);
        let f = Val::from_u32;
        let fib = RowMajorMatrix::new(
            vec![
                f(1),
                f(1),
                f(1),
                f(2),
                f(2),
                f(3),
                f(3),
                f(5),
                f(5),
                f(8),
                f(8),
                f(13),
                f(13),
                f(21),
                f(21),
                f(34),
            ],
            2,
        );
        let other = RowMajorMatrix::new(vec![Val::ZERO; 4], 1);
        let witness = SystemWitness::from_stage_1(vec![fib, other], &system);
        let no_claims: &[&[Val]] = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness);
        system.verify_multiple_claims(no_claims, &proof).unwrap();
    }

    /// A constraint referencing the next row while `uses_next_row` is false
    /// must be rejected at setup (there would be no ζ·g opening to evaluate
    /// it against).
    #[test]
    #[should_panic(expected = "uses_next_row")]
    fn undeclared_next_row_rejected() {
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
        let inputs = CircuitInputs {
            main_width: 1,
            constraints: vec![Expr::IsTransition * (Expr::main_next(0) - Expr::main(0))],
            uses_next_row: false,
            ..Default::default()
        };
        System::new(config, [inputs]);
    }

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
        let trace = RowMajorMatrix::new(vec![Val::ZERO; 8], 1);
        SystemWitness::from_stage_1(vec![trace], &system);
    }
}
