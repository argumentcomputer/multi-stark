use std::fmt;

use crate::traits::Field;

use super::builder::Recipe;
use super::{Circuit, CircuitId, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessError {
    HashMismatch,
    ForeignValue,
    NotAnInput,
    AlreadyAssigned,
    MissingInput { name: String },
    HintFailed { name: String, message: String },
    UnsatisfiedGate { index: usize },
    LookupMissing { table: String },
    ForeignAssignment,
    ValueCount { expected: usize, actual: usize },
    ShardIndex { index: usize, count: usize },
    PublicCount { expected: usize, actual: usize },
}

impl fmt::Display for WitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for WitnessError {}

/// A single execution of the circuit's witness plan.
pub struct Witness<'a, F: Field> {
    circuit: &'a Circuit<F>,
    inputs: Vec<Option<F>>,
}

/// Computed logical values, before backend row placement.
pub struct Assignment<F> {
    pub(super) owner: u64,
    pub(super) values: Vec<F>,
    pub(super) publics: Vec<F>,
}

impl<F: Field> Assignment<F> {
    /// Borrow logical values after checking the source circuit's identity.
    pub fn values(&self, circuit: CircuitId) -> Result<&[F], WitnessError> {
        if circuit != CircuitId(self.owner) {
            return Err(WitnessError::ForeignAssignment);
        }
        Ok(&self.values)
    }

    pub fn value(&self, value: Value) -> Result<F, WitnessError> {
        if value.owner != self.owner {
            return Err(WitnessError::ForeignValue);
        }
        self.values
            .get(value.index)
            .copied()
            .ok_or(WitnessError::ForeignValue)
    }

    /// Useful for constructing a prover's statement. A verifier must obtain
    /// the expected public values independently of this private assignment.
    pub fn public_values(&self) -> &[F] {
        &self.publics
    }
}

impl<'a, F: Field> Witness<'a, F> {
    pub(super) fn new(circuit: &'a Circuit<F>) -> Self {
        Self {
            circuit,
            inputs: vec![None; circuit.input_names.len()],
        }
    }

    /// Assigns each external input exactly once. Derived values cannot be
    /// overwritten: their recipes will be evaluated by [`Self::generate`].
    pub fn set(&mut self, input: Value, value: F) -> Result<(), WitnessError> {
        if input.owner != self.circuit.owner || input.index >= self.circuit.num_values() {
            return Err(WitnessError::ForeignValue);
        }
        let Recipe::Input(slot) = self.circuit.recipes[input.index] else {
            return Err(WitnessError::NotAnInput);
        };
        if self.inputs[slot].is_some() {
            return Err(WitnessError::AlreadyAssigned);
        }
        self.inputs[slot] = Some(value);
        Ok(())
    }

    /// Evaluates the topological witness plan and checks all frontend
    /// relations. These checks improve diagnostics; backend constraints must
    /// enforce the same relations even for an adversarially generated trace.
    pub fn generate(self) -> Result<Assignment<F>, WitnessError> {
        let mut values = Vec::with_capacity(self.circuit.num_values());
        let mut args = Vec::new();
        while values.len() < self.circuit.num_values() {
            let recipe = &self.circuit.recipes[values.len()];
            let value = match recipe {
                Recipe::Input(slot) => {
                    self.inputs[*slot].ok_or_else(|| WitnessError::MissingInput {
                        name: self.circuit.input_names[*slot].clone(),
                    })?
                }
                Recipe::Constant(c) => *c,
                Recipe::Arithmetic(index) => {
                    let gate = &self.circuit.gates[*index];
                    let [a, b, _] = gate.wires;
                    let [qm, qa, qb, _, k] = gate.coefficients;
                    let (a, b) = (values[a.index], values[b.index]);
                    let mut result = k;
                    if qm != F::ZERO {
                        result += qm * a * b;
                    }
                    if qa != F::ZERO {
                        result += qa * a;
                    }
                    if qb != F::ZERO {
                        result += qb * b;
                    }
                    result
                }
                Recipe::Hint(index) => {
                    let hint = &self.circuit.hints[*index];
                    debug_assert_eq!(values.len(), hint.start);
                    args.clear();
                    args.extend(hint.dependencies.iter().map(|&i| values[i]));
                    values.resize(hint.start + hint.len, F::ZERO);
                    (hint.compute)(&args, &mut values[hint.start..]).map_err(|message| {
                        WitnessError::HintFailed {
                            name: hint.name.clone(),
                            message,
                        }
                    })?;
                    continue;
                }
            };
            values.push(value);
        }
        self.circuit.check_values(&values)?;
        let publics = self
            .circuit
            .publics
            .iter()
            .map(|v| values[v.index])
            .collect();
        Ok(Assignment {
            owner: self.circuit.owner,
            values,
            publics,
        })
    }
}
