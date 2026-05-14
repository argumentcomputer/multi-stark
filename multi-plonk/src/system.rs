//! Multi-circuit system + per-circuit metadata + setup of preprocessed
//! commitments.

use ark_ff::Zero;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain, univariate::DensePolynomial};
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial, PolynomialCommitment};

use crate::air::{Air, BaseAir};
use crate::builder::symbolic::{
    SymbolicAirBuilder, get_max_constraint_degree, get_symbolic_constraints,
};
use crate::lookup::{LOOKUP_PUBLIC_SIZE, Lookup, LookupAir};
use crate::matrix::Matrix;
use crate::types::{Commitment, CommitmentState, PC, PlonkConfig, UniPoly, Val};

pub struct System<A> {
    pub circuits: Vec<Circuit<A>>,
    /// Maps circuit index → index into the preprocessed bundle (None if circuit
    /// has no preprocessed trace).
    pub preprocessed_indices: Vec<Option<usize>>,
    /// Public per-preprocessed-circuit data (commitment + label).
    pub preprocessed_commitments: Vec<LabeledCommitment<Commitment>>,
}

pub struct ProverKey {
    /// Coefficient-form preprocessed polynomials (one per preprocessed circuit).
    pub preprocessed_polys: Vec<LabeledPolynomial<Val, UniPoly>>,
    /// PCS commitment-state (randomness) for each preprocessed polynomial.
    pub preprocessed_states: Vec<CommitmentState>,
    /// Original eval-form preprocessed traces (kept for row evaluation).
    pub preprocessed_traces: Vec<Matrix<Val>>,
}

pub struct Circuit<A> {
    pub air: LookupAir<A>,
    pub constraint_count: usize,
    pub max_constraint_degree: usize,
    pub preprocessed_height: usize,
    pub preprocessed_width: usize,
    pub stage_1_width: usize,
    pub stage_2_width: usize,
}

#[derive(Clone)]
pub struct SystemWitness {
    pub traces: Vec<Matrix<Val>>,
    /// Per circuit, per row, per lookup.
    pub lookups: Vec<Vec<Vec<Lookup<Val>>>>,
}

impl SystemWitness {
    pub fn from_stage_1<A>(traces: Vec<Matrix<Val>>, system: &System<A>) -> Self {
        let lookups = traces
            .iter()
            .zip(system.circuits.iter())
            .map(|(trace, circuit)| {
                let preprocessed = circuit.air.preprocessed.as_ref();
                trace
                    .rows()
                    .enumerate()
                    .map(|(row_i, row)| {
                        circuit
                            .air
                            .lookups
                            .iter()
                            .map(|lookup| {
                                lookup.compute_expr(row, preprocessed.map(|m| m.row(row_i)))
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self { traces, lookups }
    }
}

impl<A> System<A>
where
    A: BaseAir + Air<SymbolicAirBuilder>,
{
    pub fn new(
        config: &PlonkConfig,
        airs: impl IntoIterator<Item = LookupAir<A>>,
    ) -> (Self, ProverKey) {
        let mut circuits = vec![];
        let mut preprocessed_traces = vec![];
        let mut preprocessed_indices = vec![];
        for air in airs {
            let (circuit, maybe_preprocessed_trace) = Circuit::from_air(air);
            circuits.push(circuit);
            if let Some(t) = maybe_preprocessed_trace {
                preprocessed_indices.push(Some(preprocessed_traces.len()));
                preprocessed_traces.push(t);
            } else {
                preprocessed_indices.push(None);
            }
        }
        // Commit preprocessed polynomials.
        let mut preprocessed_polys: Vec<LabeledPolynomial<Val, UniPoly>> = Vec::new();
        for (i, trace) in preprocessed_traces.iter().enumerate() {
            for (col_idx, col) in trace.columns().into_iter().enumerate() {
                let poly = ifft_column(&col);
                let label = format!("preprocessed_c{i}_col{col_idx}");
                preprocessed_polys.push(LabeledPolynomial::new(label, poly, None, None));
            }
        }
        let (preprocessed_commitments, preprocessed_states) = if preprocessed_polys.is_empty() {
            (vec![], vec![])
        } else {
            PC::commit(&config.committer_key, &preprocessed_polys, None)
                .expect("preprocessed commit failed")
        };

        let system = Self {
            circuits,
            preprocessed_indices,
            preprocessed_commitments,
        };
        let key = ProverKey {
            preprocessed_polys,
            preprocessed_states,
            preprocessed_traces,
        };
        (system, key)
    }
}

impl<A> Circuit<A>
where
    A: BaseAir + Air<SymbolicAirBuilder>,
{
    pub fn from_air(air: LookupAir<A>) -> (Self, Option<Matrix<Val>>) {
        let stage_1_width = air.inner_air.width();
        let stage_2_width = air.stage_2_width();
        let preprocessed_trace = air.preprocessed_trace();
        let preprocessed_height = preprocessed_trace.as_ref().map_or(0, |m| m.height());
        let preprocessed_width = preprocessed_trace.as_ref().map_or(0, |m| m.width());
        let constraints = get_symbolic_constraints(
            &air,
            preprocessed_width,
            stage_1_width,
            stage_2_width,
            LOOKUP_PUBLIC_SIZE,
        );
        let constraint_count = constraints.len();
        let max_constraint_degree = get_max_constraint_degree(&constraints);
        let circuit = Self {
            air,
            constraint_count,
            max_constraint_degree,
            preprocessed_height,
            preprocessed_width,
            stage_1_width,
            stage_2_width,
        };
        (circuit, preprocessed_trace)
    }
}

/// Convenience: iFFT a single column from evaluation form (on the
/// canonical subgroup of size `n = col.len()`) into coefficient form.
pub(crate) fn ifft_column(col: &[Val]) -> UniPoly {
    let n = col.len();
    if n == 0 {
        return DensePolynomial { coeffs: vec![] };
    }
    let domain =
        GeneralEvaluationDomain::<Val>::new(n).expect("trace size must be supported by FFT domain");
    let coeffs = domain.ifft(col);
    let mut poly = DensePolynomial { coeffs };
    while poly.coeffs.last().is_some_and(|c| c.is_zero()) {
        poly.coeffs.pop();
    }
    poly
}
