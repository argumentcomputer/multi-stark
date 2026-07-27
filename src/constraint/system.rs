//! Multi-circuit STARK system, constraint-IR version.
//!
//! Parallel to [`crate::system`], but circuits are compiled constraint data
//! ([`super::circuit::Circuit`]) instead of AIRs, so there is no `A` type
//! parameter. `System::new` derives each circuit's stage-2 and public-input
//! layout from its lookups and the challenge field's extension degree,
//! appends the synthesized lookup constraints, and compiles.

use p3_challenger::CanObserve;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

use crate::config::{Com, PcsData, StarkGenericConfig, Val};
use crate::lookup::LookupValues;

use super::circuit::{Circuit as CompiledCircuit, ExtensionParams, compile};
use super::eval::VarValues;
use super::expr::{CircuitSpec, Expr, ExtExpr};
use super::lookup::{Lookup, num_publics, stage2_width, synthesize_lookups};

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
}

impl<F: Field> Default for CircuitInputs<F> {
    fn default() -> Self {
        Self {
            main_width: 0,
            preprocessed: None,
            constraints: vec![],
            ext_constraints: vec![],
            lookups: vec![],
        }
    }
}

/// A compiled circuit within the system, with the metadata the prover and
/// verifier need. The preprocessed trace is retained for witness-time
/// lookup evaluation (it is also committed at setup).
pub struct Circuit<F: Field> {
    pub compiled: CompiledCircuit<F>,
    pub main_width: usize,
    pub preprocessed: Option<RowMajorMatrix<F>>,
    pub preprocessed_width: usize,
    pub preprocessed_height: usize,
    pub num_lookups: usize,
    /// Stage-2 trace width in flattened base columns: `(1 + num_lookups)·D`.
    pub stage_2_width: usize,
    /// Public-input width in base coordinates: `4·D`.
    pub num_publics: usize,
}

impl<F: Field> Circuit<F> {
    /// Number of constraint roots (after canonicalization).
    pub fn constraint_count(&self) -> usize {
        self.compiled.zeros.len()
    }

    pub fn max_constraint_degree(&self) -> usize {
        self.compiled.max_constraint_degree as usize
    }

    /// Degree of the quotient polynomial as a multiple of the trace degree.
    pub fn quotient_degree(&self) -> usize {
        (self.max_constraint_degree().max(2) - 1).next_power_of_two()
    }
}

/// A multi-circuit STARK system over compiled constraint circuits.
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
        inputs: impl IntoIterator<Item = CircuitInputs<Val<SC>>>,
    ) -> (Self, ProverKey<SC>) {
        let pcs = config.pcs();
        let params = extension_params::<SC>();
        let d = params.degree;

        let mut circuits = vec![];
        let mut preprocessed_traces = vec![];
        let mut preprocessed_indices = vec![];
        for (i, input) in inputs.into_iter().enumerate() {
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
            let compiled = compile(&spec, &params)
                .unwrap_or_else(|e| panic!("circuit {i}: constraint compilation failed: {e:?}"));

            let circuit = Circuit {
                compiled,
                main_width: input.main_width,
                preprocessed: input.preprocessed,
                preprocessed_width,
                preprocessed_height,
                num_lookups,
                stage_2_width: s2_width,
                num_publics: n_publics,
            };
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

    /// Binds the system shape into the Fiat-Shamir transcript. Prover and
    /// verifier call this identically, before observing any commitment.
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
        }
    }
}

/// Witness data for the system: stage-1 traces and the concrete lookup
/// values derived from them.
#[derive(Clone)]
pub struct SystemWitness<F: Field> {
    pub traces: Vec<RowMajorMatrix<F>>,
    pub lookups: Vec<LookupValues<F>>,
}

impl<F: Field> SystemWitness<F> {
    /// Builds the witness from the stage-1 traces, computing each circuit's
    /// lookup values by sweeping the compiled lookup prefix over its rows.
    ///
    /// # Panics
    /// Panics if the number of traces differs from the number of circuits,
    /// or if a circuit with a preprocessed trace receives a main trace of a
    /// different height.
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
        .compiled
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
        circuit.compiled.sweep_lookup_prefix(&view, &mut buf);
        for (slot, lookup) in circuit.compiled.lookups.iter().enumerate() {
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
    use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config};

    const COMMITMENT_PARAMETERS: CommitmentParameters = CommitmentParameters {
        log_blowup: 1,
        cap_height: 0,
    };
    const FRI_PARAMETERS: FriParameters = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 64,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };

    fn config() -> GoldilocksBlake3Config {
        GoldilocksBlake3Config::new(COMMITMENT_PARAMETERS, FRI_PARAMETERS)
    }

    #[test]
    fn extension_params_goldilocks_quadratic() {
        let p = extension_params::<GoldilocksBlake3Config>();
        assert_eq!(p.degree, 2);
        assert_eq!(p.w, crate::types::Val::from_u64(7));
        assert!(p.karatsuba);
    }

    #[test]
    fn builds_and_lays_out_circuit() {
        let input = CircuitInputs {
            main_width: 3,
            constraints: vec![Expr::main(0) * Expr::main(1) - Expr::main(2)],
            lookups: vec![Lookup {
                multiplicity: Expr::main(0),
                args: vec![Expr::main(1), Expr::main(2)],
            }],
            ..Default::default()
        };
        let (system, key) = System::new(config(), [input]);
        assert_eq!(system.circuits.len(), 1);
        assert!(key.preprocessed_data.is_none());
        let c = &system.circuits[0];
        // one lookup at D=2: stage-2 width (1+1)*2 = 4, publics 4*2 = 8.
        assert_eq!(c.stage_2_width, 4);
        assert_eq!(c.num_publics, 8);
        assert_eq!(c.num_lookups, 1);
        // 1 base constraint + (1 message + 3 accumulator) ext constraints,
        // each expanded to D=2 coordinate roots: 1 + 4*2 = 9.
        assert_eq!(c.constraint_count(), 9);
        // shape binding must not panic.
        let mut challenger = system.config.initialise_challenger();
        system.observe_shape(&mut challenger);
    }
}
