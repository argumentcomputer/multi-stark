use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::traits::Field;

use super::{Witness, WitnessError};

static NEXT_CIRCUIT: AtomicU64 = AtomicU64::new(0);

/// A symbolic native-field value. Handles belong to exactly one builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Value {
    pub(super) owner: u64,
    pub(super) index: usize,
}

impl Value {
    /// Position in the circuit's value vector, independent of backend layout.
    pub fn index(self) -> usize {
        self.index
    }
}

/// A value whose booleanity has been constrained by its builder.
#[derive(Clone, Copy, Debug)]
pub struct Bool(pub(super) Value);

impl Bool {
    pub fn value(self) -> Value {
        self.0
    }
}

/// A handle to an immutable lookup table owned by one circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Table {
    pub(super) owner: u64,
    pub(super) index: usize,
}

impl Table {
    pub fn index(self) -> usize {
        self.index
    }
}

/// The relation `qm*a*b + qa*a + qb*b + qc*c + k = 0`.
///
/// Coefficients are circuit constants; wires refer to logical values, not
/// physical cells. This relation is shared by witness checking and lowering.
#[derive(Clone, Debug)]
pub struct Gate<F> {
    pub wires: [Value; 3],
    /// `[qm, qa, qb, qc, k]`.
    pub coefficients: [F; 5],
}

impl<F: Field> Gate<F> {
    pub(super) fn evaluate(&self, values: &[F]) -> F {
        let [a, b, c] = self.wires.map(|v| values[v.index]);
        let [qm, qa, qb, qc, k] = self.coefficients;
        qm * a * b + qa * a + qb * b + qc * c + k
    }
}

/// A fixed table. Duplicate rows are allowed and denote the same membership.
pub struct TableDefinition<F> {
    pub(super) name: String,
    pub(super) rows: Vec<Vec<F>>,
    pub(super) indices: HashMap<Vec<F>, usize>,
}

impl<F> TableDefinition<F> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn rows(&self) -> &[Vec<F>] {
        &self.rows
    }

    pub fn width(&self) -> usize {
        self.rows[0].len()
    }
}

/// A tuple of values constrained to belong to a fixed table.
#[derive(Clone, Debug)]
pub struct LookupConstraint {
    pub table: Table,
    pub values: Vec<Value>,
}

type Hint<F> = dyn Fn(&[F], &mut [F]) -> Result<(), String> + Send + Sync;

pub(super) struct HintDefinition<F> {
    pub name: String,
    pub dependencies: Box<[usize]>,
    pub start: usize,
    pub len: usize,
    pub compute: Box<Hint<F>>,
}

pub(super) enum Recipe<F> {
    Input(usize),
    Constant(F),
    /// Output of a builder-generated gate with output coefficient -1.
    /// Reuse the gate's coefficients rather than retaining a second copy.
    Arithmetic(usize),
    /// All outputs of a hint reference the same batch definition.
    Hint(usize),
}

/// Exact frontend counts, independent of any backend's row placement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CircuitStats {
    pub hash_calls: usize,
    pub hash_compressions: usize,
    pub values: usize,
    pub inputs: usize,
    pub gates: usize,
    pub lookups: usize,
    pub tables: usize,
    pub publics: usize,
    pub hint_calls: usize,
    pub hint_outputs: usize,
}

impl CircuitStats {
    fn since(self, before: Self) -> Self {
        Self {
            hash_calls: self.hash_calls - before.hash_calls,
            hash_compressions: self.hash_compressions - before.hash_compressions,
            values: self.values - before.values,
            inputs: self.inputs - before.inputs,
            gates: self.gates - before.gates,
            lookups: self.lookups - before.lookups,
            tables: self.tables - before.tables,
            publics: self.publics - before.publics,
            hint_calls: self.hint_calls - before.hint_calls,
            hint_outputs: self.hint_outputs - before.hint_outputs,
        }
    }
}

/// Immutable circuit description, independent of its row layout and witness.
///
/// Inspecting this IR is sufficient to lower its relations to another backend.
/// The witness recipes are an execution aid, never part of the trusted relation.
pub struct Circuit<F: Field> {
    pub(super) hashes: Option<super::hash::Compact>,
    pub(super) owner: u64,
    pub(super) recipes: Vec<Recipe<F>>,
    pub(super) input_names: Vec<String>,
    pub(super) hints: Vec<HintDefinition<F>>,
    hint_outputs: usize,
    pub(super) gates: Vec<Gate<F>>,
    pub(super) tables: Vec<TableDefinition<F>>,
    pub(super) lookups: Vec<LookupConstraint>,
    pub(super) publics: Vec<Value>,
    pub(super) zero: Value,
}

impl<F: Field> Circuit<F> {
    pub fn stats(&self) -> CircuitStats {
        CircuitStats {
            hash_calls: self.hashes.as_ref().map_or(0, |h| h.calls.len()),
            hash_compressions: self.hashes.as_ref().map_or(0, |h| h.compressions),
            values: self.recipes.len(),
            inputs: self.input_names.len(),
            gates: self.gates.len(),
            lookups: self.lookups.len(),
            tables: self.tables.len(),
            publics: self.publics.len(),
            hint_calls: self.hints.len(),
            hint_outputs: self.hint_outputs,
        }
    }

    pub fn gates(&self) -> &[Gate<F>] {
        &self.gates
    }

    pub fn tables(&self) -> &[TableDefinition<F>] {
        &self.tables
    }

    pub fn lookups(&self) -> &[LookupConstraint] {
        &self.lookups
    }

    /// Compact BLAKE3 calls that a backend must constrain, including byte bindings.
    pub fn blake3_calls(&self) -> impl Iterator<Item = (&[Value], &[Value; 32])> {
        self.hashes
            .iter()
            .flat_map(|h| &h.calls)
            .map(|c| (c.input.as_slice(), &c.output))
    }

    /// Public bindings in the order they were exposed.
    pub fn public_values(&self) -> &[Value] {
        &self.publics
    }

    pub fn num_values(&self) -> usize {
        self.recipes.len()
    }

    pub fn witness(&self) -> Witness<'_, F> {
        Witness::new(self)
    }
}

/// Constructs a fixed computation over `F`, without observing any witness data.
///
/// Foreign handles and malformed table declarations are programmer errors and
/// panic. Assignment and unsatisfied-constraint errors are returned by the
/// witness API. No method branches on a witness value or changes the layout
/// during witness generation.
pub struct CircuitBuilder<F: Field> {
    circuit: Circuit<F>,
    constants: HashMap<F, Value>,
    census: Option<Census<F>>,
}

struct Census<F> {
    stats: CircuitStats,
    constants: HashMap<usize, F>,
    #[cfg(feature = "groth16")]
    lookup_observer: Option<LookupObserver<F>>,
    #[cfg(feature = "groth16")]
    linear_observer: Option<LinearObserver<F>>,
}

#[cfg(feature = "groth16")]
type LookupObserver<F> = Box<dyn FnMut(usize, &TableDefinition<F>, &[Value])>;
#[cfg(feature = "groth16")]
type LinearObserver<F> = Box<dyn FnMut(Value, [Value; 2], [F; 3])>;

impl<F: Field> Default for CircuitBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> CircuitBuilder<F> {
    pub fn new() -> Self {
        let owner = NEXT_CIRCUIT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .expect("circuit handle space exhausted");
        let zero = Value { owner, index: 0 };
        Self {
            circuit: Circuit {
                hashes: None,
                owner,
                recipes: vec![Recipe::Constant(F::ZERO)],
                input_names: vec![],
                hints: vec![],
                hint_outputs: 0,
                gates: vec![Gate {
                    wires: [zero; 3],
                    coefficients: [F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO],
                }],
                tables: vec![],
                lookups: vec![],
                publics: vec![],
                zero,
            },
            constants: HashMap::from([(F::ZERO, zero)]),
            census: None,
        }
    }

    /// Executes the same builder operations without retaining constraints or recipes.
    #[cfg(feature = "kzg")]
    pub(super) fn counting() -> Self {
        let mut builder = Self::new();
        builder.census = Some(Census {
            stats: builder.stats(),
            constants: HashMap::from([(0, F::ZERO)]),
            #[cfg(feature = "groth16")]
            lookup_observer: None,
            #[cfg(feature = "groth16")]
            linear_observer: None,
        });
        builder
    }

    #[cfg(feature = "groth16")]
    pub(super) fn observe_counted_lookups(&mut self, observer: LookupObserver<F>) {
        self.census
            .as_mut()
            .expect("counting builder")
            .lookup_observer = Some(observer);
    }

    #[cfg(feature = "groth16")]
    pub(super) fn observe_counted_linear(&mut self, observer: LinearObserver<F>) {
        self.census
            .as_mut()
            .expect("counting builder")
            .linear_observer = Some(observer);
    }

    #[cfg(feature = "groth16")]
    pub(super) fn constant_count(&self) -> usize {
        self.constants.len()
    }

    #[cfg(feature = "groth16")]
    pub(super) fn tables(&self) -> &[TableDefinition<F>] {
        &self.circuit.tables
    }

    #[cfg(feature = "kzg")]
    pub(super) fn reserve(&mut self, stats: CircuitStats) {
        let c = &mut self.circuit;
        c.recipes
            .reserve_exact(stats.values.saturating_sub(c.recipes.len()));
        c.gates
            .reserve_exact(stats.gates.saturating_sub(c.gates.len()));
        c.lookups
            .reserve_exact(stats.lookups.saturating_sub(c.lookups.len()));
        c.hints
            .reserve_exact(stats.hint_calls.saturating_sub(c.hints.len()));
        c.input_names
            .reserve_exact(stats.inputs.saturating_sub(c.input_names.len()));
        c.publics
            .reserve_exact(stats.publics.saturating_sub(c.publics.len()));
    }

    fn check(&self, value: Value) {
        assert_eq!(
            value.owner, self.circuit.owner,
            "value belongs to another circuit"
        );
        assert!(value.index < self.stats().values, "invalid value handle");
    }

    /// Opt into fixed custom-gate BLAKE3 traces. The lowering binds their
    /// input/output bytes to ordinary wires; native hash hints are not trusted.
    /// This lowering uses lookup groups of three and requires blowup >= 4.
    pub fn enable_compact_blake3(&mut self) {
        self.circuit.hashes.get_or_insert_with(Default::default);
    }

    pub(super) fn compact_blake3_enabled(&self) -> bool {
        self.circuit.hashes.is_some()
    }

    pub(super) fn record_hash(&mut self, input: Vec<Value>, output: &[Value; 32]) {
        for &v in input.iter().chain(output) {
            self.check(v);
        }
        if let Some(census) = &mut self.census {
            assert!(self.circuit.hashes.is_some(), "compact hash enabled");
            census.stats.hash_calls += 1;
            census.stats.hash_compressions +=
                input.len().max(1).div_ceil(64) + input.len().max(1).div_ceil(1024) - 1;
            return;
        }
        self.circuit
            .hashes
            .as_mut()
            .expect("compact hash enabled")
            .add_call(super::hash::HashCall {
                input,
                output: *output,
            });
    }

    fn allocate(&mut self, recipe: Recipe<F>) -> Value {
        let value = Value {
            owner: self.circuit.owner,
            index: self.stats().values,
        };
        if let Some(census) = &mut self.census {
            census.stats.values += 1;
            if let Recipe::Constant(c) = recipe {
                census.constants.insert(value.index, c);
            }
        } else {
            self.circuit.recipes.push(recipe);
        }
        value
    }

    pub fn stats(&self) -> CircuitStats {
        self.census
            .as_ref()
            .map_or_else(|| self.circuit.stats(), |c| c.stats)
    }

    /// Measure an ordinary gadget call without retaining profiling metadata.
    /// Nested measurements are inclusive. Reused constants cost nothing.
    pub fn measure<R>(&mut self, build: impl FnOnce(&mut Self) -> R) -> (R, CircuitStats) {
        let owner = self.circuit.owner;
        let before = self.stats();
        let result = build(self);
        assert_eq!(
            owner, self.circuit.owner,
            "measurement must not replace its builder"
        );
        (result, self.stats().since(before))
    }

    fn known_constant(&self, value: Value) -> Option<F> {
        if let Some(census) = &self.census {
            return census.constants.get(&value.index).copied();
        }
        match self.circuit.recipes[value.index] {
            Recipe::Constant(c) => Some(c),
            _ => None,
        }
    }

    fn arithmetic(&mut self, a: Value, b: Value, qm: F, qa: F, qb: F, k: F) -> Value {
        let out = self.allocate(Recipe::Arithmetic(self.stats().gates));
        #[cfg(feature = "groth16")]
        if qm == F::ZERO
            && let Some(observer) = self
                .census
                .as_mut()
                .and_then(|c| c.linear_observer.as_mut())
        {
            observer(out, [a, b], [qa, qb, k]);
        }
        self.constrain_gate([a, b, out], [qm, qa, qb, F::NEG_ONE, k]);
        out
    }

    /// Allocates an externally assigned private input. Expose it separately if
    /// it is part of the public statement.
    pub fn input(&mut self, name: impl Into<String>) -> Value {
        let slot = self.stats().inputs;
        if let Some(census) = &mut self.census {
            census.stats.inputs += 1;
        } else {
            self.circuit.input_names.push(name.into());
        }
        self.allocate(Recipe::Input(slot))
    }

    /// Allocates and immediately exposes a public input.
    pub fn public_input(&mut self, name: impl Into<String>) -> Value {
        let value = self.input(name);
        self.expose_public(value);
        value
    }

    pub fn constant(&mut self, constant: F) -> Value {
        if let Some(&value) = self.constants.get(&constant) {
            return value;
        }
        let value = self.allocate(Recipe::Constant(constant));
        let zero = self.circuit.zero;
        self.constrain_gate(
            [value, zero, zero],
            [F::ZERO, F::ONE, F::ZERO, F::ZERO, -constant],
        );
        self.constants.insert(constant, value);
        value
    }

    /// Adds a low-level arithmetic relation without allocating a witness.
    pub fn constrain_gate(&mut self, wires: [Value; 3], coefficients: [F; 5]) {
        for value in wires {
            self.check(value);
        }
        if let Some(census) = &mut self.census {
            census.stats.gates += 1;
        } else {
            self.circuit.gates.push(Gate {
                wires,
                coefficients,
            });
        }
    }

    pub fn add(&mut self, a: Value, b: Value) -> Value {
        self.affine([a, b], [F::ONE, F::ONE], F::ZERO)
    }

    pub fn sub(&mut self, a: Value, b: Value) -> Value {
        self.affine([a, b], [F::ONE, F::NEG_ONE], F::ZERO)
    }

    /// `coefficients[0]*values[0] + coefficients[1]*values[1] + constant`.
    /// Uses at most one arithmetic gate, with coefficients fixed in the circuit.
    /// All handles are checked even when a coefficient is zero.
    pub fn affine(
        &mut self,
        values: [Value; 2],
        mut coefficients: [F; 2],
        mut constant: F,
    ) -> Value {
        for &value in &values {
            self.check(value);
        }
        if values[0] == values[1] && coefficients[0] + coefficients[1] == F::ZERO {
            return self.constant(constant);
        }
        let mut values = values;
        for i in 0..2 {
            if let Some(c) = self.known_constant(values[i]) {
                constant += coefficients[i] * c;
                coefficients[i] = F::ZERO;
            }
            if coefficients[i] == F::ZERO {
                values[i] = self.circuit.zero;
            }
        }
        match coefficients {
            [a, b] if a == F::ZERO && b == F::ZERO => self.constant(constant),
            [a, b] if a == F::ONE && b == F::ZERO && constant == F::ZERO => values[0],
            [a, b] if a == F::ZERO && b == F::ONE && constant == F::ZERO => values[1],
            [a, b] => self.arithmetic(values[0], values[1], F::ZERO, a, b, constant),
        }
    }

    /// Native-field scaling without allocating a separate constant wire.
    pub fn scale(&mut self, value: Value, scalar: F) -> Value {
        self.affine([value, self.circuit.zero], [scalar, F::ZERO], F::ZERO)
    }

    /// Sum of fixed-coefficient terms plus a constant. Combines repeated
    /// handles, folds constants, and emits a deterministic chain of affine
    /// gates. It never inspects witness values or infers equality from hints.
    pub fn linear_combination(&mut self, terms: &[(F, Value)], mut constant: F) -> Value {
        let mut terms = terms.to_vec();
        for &(_, value) in &terms {
            self.check(value);
        }
        terms.sort_by_key(|&(_, value)| value.index);
        let mut normalized: Vec<(F, Value)> = Vec::with_capacity(terms.len());
        for (coefficient, value) in terms {
            if let Some(c) = self.known_constant(value) {
                constant += coefficient * c;
            } else if let Some((previous, last)) = normalized.last_mut()
                && *last == value
            {
                *previous += coefficient;
            } else {
                normalized.push((coefficient, value));
            }
        }
        normalized.retain(|&(c, _)| c != F::ZERO);
        let mut terms = normalized.into_iter();
        let Some((c, first)) = terms.next() else {
            return self.constant(constant);
        };
        let (d, second) = terms.next().unwrap_or((F::ZERO, self.circuit.zero));
        let mut sum = self.affine([first, second], [c, d], constant);
        for (c, value) in terms {
            sum = self.affine([sum, value], [F::ONE, c], F::ZERO);
        }
        sum
    }

    pub fn mul(&mut self, a: Value, b: Value) -> Value {
        self.check(a);
        self.check(b);
        if let Some(c) = self.known_constant(a) {
            return self.scale(b, c);
        }
        if let Some(c) = self.known_constant(b) {
            return self.scale(a, c);
        }
        self.arithmetic(a, b, F::ONE, F::ZERO, F::ZERO, F::ZERO)
    }

    /// `a*b + c`. Constant factors fuse into one affine gate; three
    /// distinct variable operands require two gates in the current three-wire
    /// relation. A constant addend or an addend equal to a factor uses one.
    pub fn mul_add(&mut self, a: Value, b: Value, c: Value) -> Value {
        for value in [a, b, c] {
            self.check(value);
        }
        if let Some(k) = self.known_constant(a) {
            return self.affine([b, c], [k, F::ONE], F::ZERO);
        }
        if let Some(k) = self.known_constant(b) {
            return self.affine([a, c], [k, F::ONE], F::ZERO);
        }
        if let Some(k) = self.known_constant(c) {
            return self.arithmetic(a, b, F::ONE, F::ZERO, F::ZERO, k);
        }
        if c == a || c == b {
            return self.arithmetic(
                a,
                b,
                F::ONE,
                F::from_bool(c == a),
                F::from_bool(c != a),
                F::ZERO,
            );
        }
        let product = self.mul(a, b);
        self.add(product, c)
    }

    pub fn assert_equal(&mut self, a: Value, b: Value) {
        self.check(a);
        self.check(b);
        if a == b {
            return;
        }
        self.constrain_gate(
            [a, b, self.circuit.zero],
            [F::ZERO, F::ONE, F::NEG_ONE, F::ZERO, F::ZERO],
        );
    }

    pub fn assert_zero(&mut self, value: Value) {
        self.assert_equal(value, self.circuit.zero);
    }

    pub fn assert_bool(&mut self, value: Value) -> Bool {
        self.constrain_gate(
            [value, value, self.circuit.zero],
            [F::ONE, F::NEG_ONE, F::ZERO, F::ZERO, F::ZERO],
        );
        Bool(value)
    }

    /// Selects `when_true` when the constrained bit is one.
    pub fn select(&mut self, bit: Bool, when_true: Value, when_false: Value) -> Value {
        for value in [bit.0, when_true, when_false] {
            self.check(value);
        }
        if when_true == when_false {
            return when_true;
        }
        if let Some(c) = self.known_constant(bit.0) {
            // For an invalid constant bit, retain the general relation;
            // assert_bool's constraint still rejects the circuit.
            if c == F::ONE {
                return when_true;
            }
            if c == F::ZERO {
                return when_false;
            }
        }
        if let (Some(yes), Some(no)) = (
            self.known_constant(when_true),
            self.known_constant(when_false),
        ) {
            return self.affine([bit.0, self.circuit.zero], [yes - no, F::ZERO], no);
        }
        // Keep intermediates nonnegative for bounded integer branches. This
        // uses three gates and avoids a modular subtraction before selection.
        let yes = self.mul(bit.0, when_true);
        let no = if let Some(c) = self.known_constant(when_false) {
            self.affine([bit.0, self.circuit.zero], [-c, F::ZERO], c)
        } else {
            self.arithmetic(when_false, bit.0, F::NEG_ONE, F::ONE, F::ZERO, F::ZERO)
        };
        self.add(yes, no)
    }

    /// Records a witness-only computation. **This does not constrain its
    /// output.** The caller must add relations validating the hinted result.
    /// The closure runs only after inputs are assigned, never during lowering.
    pub fn hint(
        &mut self,
        name: impl Into<String>,
        dependencies: &[Value],
        compute: impl Fn(&[F]) -> Result<F, String> + Send + Sync + 'static,
    ) -> Value {
        self.hint_many(name, dependencies, move |args| compute(args).map(|v| [v]))[0]
    }

    /// One witness-only call with a fixed, nonzero output count. The callback
    /// runs once per assignment; EVERY output needs its own appropriate
    /// constraints. Batching adds no relations and exposes no public values.
    pub fn hint_many<const N: usize>(
        &mut self,
        name: impl Into<String>,
        dependencies: &[Value],
        compute: impl Fn(&[F]) -> Result<[F; N], String> + Send + Sync + 'static,
    ) -> [Value; N] {
        assert!(N > 0, "hint must have at least one output");
        for &value in dependencies {
            self.check(value);
        }
        let index = self.stats().hint_calls;
        let start = self.stats().values;
        let outputs = std::array::from_fn(|_| self.allocate(Recipe::Hint(index)));
        if let Some(census) = &mut self.census {
            census.stats.hint_calls += 1;
            census.stats.hint_outputs += N;
            return outputs;
        }
        self.circuit.hints.push(HintDefinition {
            name: name.into(),
            dependencies: dependencies.iter().map(|v| v.index).collect(),
            start,
            len: N,
            compute: Box::new(move |args, outputs| {
                outputs.copy_from_slice(&compute(args)?);
                Ok(())
            }),
        });
        self.circuit.hint_outputs += N;
        outputs
    }

    /// Nonzero inverse: the hint is pinned by `value * inverse = 1`.
    /// Zero inputs return a witness error.
    pub fn inverse(&mut self, value: Value) -> Value {
        let inverse = self.hint("inverse", &[value], |values| {
            values[0]
                .try_inverse()
                .ok_or_else(|| "cannot invert zero".to_owned())
        });
        let one = self.constant(F::ONE);
        self.constrain_gate(
            [value, inverse, one],
            [F::ONE, F::ZERO, F::ZERO, F::NEG_ONE, F::ZERO],
        );
        inverse
    }

    /// Returns a constrained zero-test bit, including at zero.
    pub fn is_zero(&mut self, value: Value) -> Bool {
        let inverse = self.hint("zero-test inverse", &[value], |values| {
            Ok(values[0].try_inverse().unwrap_or(F::ZERO))
        });
        let product = self.mul(value, inverse);
        let one = self.constant(F::ONE);
        let result = self.sub(one, product);
        let bit = self.assert_bool(result);
        self.constrain_gate(
            [value, result, self.circuit.zero],
            [F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        );
        bit
    }

    /// Adds a public binding and returns its index in the public value vector.
    /// Repeated exposure is allowed and produces separate indexed bindings.
    pub fn expose_public(&mut self, value: Value) -> usize {
        self.check(value);
        let index = self.stats().publics;
        if let Some(census) = &mut self.census {
            census.stats.publics += 1;
        } else {
            self.circuit.publics.push(value);
        }
        index
    }

    /// Registers a nonempty, rectangular fixed table of nonzero width.
    pub fn fixed_table(&mut self, name: impl Into<String>, rows: Vec<Vec<F>>) -> Table {
        assert!(!rows.is_empty(), "lookup table must not be empty");
        let width = rows[0].len();
        assert!(width > 0, "lookup table width must be positive");
        assert!(
            rows.iter().all(|row| row.len() == width),
            "lookup table must be rectangular"
        );
        let mut indices = HashMap::new();
        for (index, row) in rows.iter().enumerate() {
            indices.entry(row.clone()).or_insert(index);
        }
        let table = Table {
            owner: self.circuit.owner,
            index: self.circuit.tables.len(),
        };
        self.circuit.tables.push(TableDefinition {
            name: name.into(),
            rows,
            indices,
        });
        if let Some(census) = &mut self.census {
            census.stats.tables += 1;
        }
        table
    }

    pub fn lookup(&mut self, table: Table, values: &[Value]) {
        assert_eq!(
            table.owner, self.circuit.owner,
            "table belongs to another circuit"
        );
        assert_eq!(
            values.len(),
            self.circuit.tables[table.index].width(),
            "lookup arity mismatch"
        );
        for &value in values {
            self.check(value);
        }
        if let Some(census) = &mut self.census {
            census.stats.lookups += 1;
            #[cfg(feature = "groth16")]
            if let Some(observer) = &mut census.lookup_observer {
                observer(table.index, &self.circuit.tables[table.index], values);
            }
        } else {
            self.circuit.lookups.push(LookupConstraint {
                table,
                values: values.to_vec(),
            });
        }
    }

    pub fn finish(self) -> Circuit<F> {
        assert!(
            self.census.is_none(),
            "a counting builder cannot produce a circuit"
        );
        self.circuit
    }
}

impl<F: Field> Circuit<F> {
    pub(super) fn check_values(&self, values: &[F]) -> Result<(), WitnessError> {
        if let Some(hashes) = &self.hashes {
            hashes.check_values(values)?;
        }
        for (index, gate) in self.gates.iter().enumerate() {
            if gate.evaluate(values) != F::ZERO {
                return Err(WitnessError::UnsatisfiedGate { index });
            }
        }
        let mut row = Vec::new();
        for lookup in &self.lookups {
            let table = &self.tables[lookup.table.index];
            row.clear();
            row.extend(lookup.values.iter().map(|v| values[v.index]));
            if !table.indices.contains_key(&row) {
                return Err(WitnessError::LookupMissing {
                    table: table.name.clone(),
                });
            }
        }
        Ok(())
    }
}
