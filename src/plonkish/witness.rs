use std::{fmt, sync::OnceLock};

#[cfg(feature = "kzg")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "kzg")]
use p3_maybe_rayon::prelude::*;

use crate::traits::Field;

use super::builder::Recipe;
use super::hash::Prepared;
use super::{Circuit, Value};

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

/// Immutable logical values checked against the circuit that generated them,
/// before backend row placement.
pub struct Assignment<F> {
    owner: u64,
    values: Vec<F>,
    publics: Vec<F>,
    hash_preparation: OnceLock<Result<Prepared, WitnessError>>,
}

impl<F: Field> Assignment<F> {
    pub(super) fn owner(&self) -> u64 {
        self.owner
    }

    /// Immutable witness values in recipe order, indexed by [`Value::index`].
    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub(super) fn prepared_hash(&self, circuit: &Circuit<F>) -> Result<&Prepared, WitnessError> {
        if self.owner != circuit.owner {
            return Err(WitnessError::ForeignAssignment);
        }
        self.hash_preparation
            .get_or_init(|| circuit.hashes.as_ref().unwrap().prepare(&self.values))
            .as_ref()
            .map_err(Clone::clone)
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
        #[cfg(feature = "kzg")]
        if cfg!(feature = "parallel")
            && self.circuit.witness_chunks.len() > 1
            && self.circuit.witness_chunks.last().is_some_and(|&end| {
                end <= self.circuit.num_values()
                    && (self.circuit.witness_prefix || end == self.circuit.num_values())
            })
        {
            return self.generate_scheduled();
        }
        self.generate_with(arithmetic_value)
    }

    fn generate_with(
        self,
        arithmetic: impl Fn([F; 5], F, F) -> F,
    ) -> Result<Assignment<F>, WitnessError> {
        let started = std::time::Instant::now();
        let mut values = Vec::with_capacity(self.circuit.num_values());
        self.evaluate_serial(&mut values, arithmetic)?;
        self.finish(values, started)
    }

    fn evaluate_serial(
        &self,
        values: &mut Vec<F>,
        arithmetic: impl Fn([F; 5], F, F) -> F,
    ) -> Result<(), WitnessError> {
        let mut args = Vec::new();
        while values.len() < self.circuit.num_values() {
            let recipe = &self.circuit.recipes[values.len()];
            let value = match recipe {
                Recipe::Input(slot) => {
                    self.inputs[*slot].ok_or_else(|| WitnessError::MissingInput {
                        name: self.circuit.input_names[*slot].to_string(),
                    })?
                }
                Recipe::Constant(c) => *c,
                Recipe::Arithmetic(index) => {
                    let gate = &self.circuit.gates[*index];
                    let [a, b, _] = gate.wires;
                    let (a, b) = (values[a.index], values[b.index]);
                    arithmetic(gate.coefficients, a, b)
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
        Ok(())
    }

    #[cfg(feature = "kzg")]
    fn generate_scheduled(self) -> Result<Assignment<F>, WitnessError> {
        let started = std::time::Instant::now();
        let prefix = *self.circuit.witness_chunks.last().unwrap();
        let mut values = Vec::with_capacity(self.circuit.num_values());
        values.resize(prefix, F::ZERO);
        let mut tail = values.as_mut_slice();
        let mut start = 0;
        let mut pieces = Vec::with_capacity(self.circuit.witness_chunks.len());
        for &end in &self.circuit.witness_chunks {
            let (piece, remainder) = tail.split_at_mut(end - start);
            pieces.push((start, piece, None));
            tail = remainder;
            start = end;
        }
        let first_error = AtomicUsize::new(usize::MAX);
        pieces
            .par_iter_mut()
            .for_each_init(Vec::new, |args, (start, values, error)| {
                *error = self.evaluate_chunk(*start, values, args, &first_error);
            });
        if let Some((_, error)) = pieces
            .into_iter()
            .filter_map(|(_, _, error)| error)
            .min_by_key(|(index, _)| *index)
        {
            return Err(error);
        }
        self.evaluate_serial(&mut values, arithmetic_value)?;
        self.finish(values, started)
    }

    #[cfg(feature = "kzg")]
    fn chunk_value(
        &self,
        index: usize,
        start: usize,
        values: &[F],
    ) -> Result<F, (usize, WitnessError)> {
        if index >= start {
            return Ok(values[index - start]);
        }
        // A certified external dependency never reads another worker's output.
        match self.circuit.recipes[index] {
            Recipe::Input(slot) => self.input_value(index, slot),
            Recipe::Constant(value) => Ok(value),
            _ => unreachable!("certified independent witness dependency"),
        }
    }

    #[cfg(feature = "kzg")]
    fn input_value(&self, index: usize, slot: usize) -> Result<F, (usize, WitnessError)> {
        self.inputs[slot].ok_or_else(|| {
            (
                index,
                WitnessError::MissingInput {
                    name: self.circuit.input_names[slot].to_string(),
                },
            )
        })
    }

    #[cfg(feature = "kzg")]
    fn evaluate_chunk(
        &self,
        start: usize,
        values: &mut [F],
        args: &mut Vec<F>,
        first_error: &AtomicUsize,
    ) -> Option<(usize, WitnessError)> {
        let result: Result<(), (usize, WitnessError)> = (|| {
            let end = start + values.len();
            let mut index = start;
            while index < end {
                if index >= first_error.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let value = match self.circuit.recipes[index] {
                    Recipe::Input(slot) => self.input_value(index, slot)?,
                    Recipe::Constant(value) => value,
                    Recipe::Arithmetic(gate) => {
                        let gate = &self.circuit.gates[gate];
                        let [a, b, _] = gate.wires;
                        arithmetic_value(
                            gate.coefficients,
                            self.chunk_value(a.index, start, values)?,
                            self.chunk_value(b.index, start, values)?,
                        )
                    }
                    Recipe::Hint(hint) => {
                        let hint = &self.circuit.hints[hint];
                        debug_assert_eq!(index, hint.start);
                        args.clear();
                        for &dependency in &hint.dependencies {
                            args.push(self.chunk_value(dependency, start, values)?);
                        }
                        (hint.compute)(args, &mut values[index - start..index - start + hint.len])
                            .map_err(|message| {
                                (
                                    index,
                                    WitnessError::HintFailed {
                                        name: hint.name.clone(),
                                        message,
                                    },
                                )
                            })?;
                        index += hint.len;
                        continue;
                    }
                };
                values[index - start] = value;
                index += 1;
            }
            Ok(())
        })();
        if let Err((index, error)) = result {
            first_error.fetch_min(index, Ordering::Relaxed);
            Some((index, error))
        } else {
            None
        }
    }

    fn finish(
        self,
        values: Vec<F>,
        started: std::time::Instant,
    ) -> Result<Assignment<F>, WitnessError> {
        tracing::info!(
            values = values.len(),
            hints = self.circuit.hints.len(),
            evaluation_seconds = started.elapsed().as_secs_f64(),
            "Plonkish witness evaluated"
        );
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
            hash_preparation: OnceLock::new(),
        })
    }
}

fn add_scaled<F: Field>(accumulator: F, coefficient: F, value: F) -> F {
    if coefficient == F::ZERO {
        accumulator
    } else if coefficient == F::ONE {
        accumulator + value
    } else if coefficient == F::NEG_ONE {
        accumulator - value
    } else {
        accumulator + coefficient * value
    }
}

fn arithmetic_value<F: Field>(coefficients: [F; 5], a: F, b: F) -> F {
    let [qm, qa, qb, _, k] = coefficients;
    let product = if qm == F::ZERO {
        k
    } else {
        add_scaled(k, qm, a * b)
    };
    add_scaled(add_scaled(product, qa, a), qb, b)
}

#[cfg(test)]
#[path = "tests/arithmetic.rs"]
mod arithmetic_tests;

#[cfg(all(test, feature = "kzg"))]
#[path = "tests/independent_witness.rs"]
mod independent_witness_tests;

#[cfg(all(test, feature = "kzg"))]
#[path = "tests/parallel_prefix.rs"]
mod parallel_prefix_tests;
