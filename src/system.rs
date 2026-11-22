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

use crate::traits::{ExtensionModulus, ExtensionOf, Field};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;

use crate::config::{Com, PcsData, ProofConfig, ProtocolVersion, Val};
use crate::lookup::LookupValues;

use crate::eval::VarValues;
use crate::expr::{CircuitSpec, Expr, ExtExpr, RowOffset, Source};
use crate::graph::{ConstraintGraph, ExtensionParams, Node, compile};
use crate::lookup::{Lookup, logup_constraint_count, logup_max_degree, num_publics, stage2_width};
use crate::p3_adapter::domain_blake3::Blake3Domain;
use crate::traits::{Pcs, Transcript};

/// User-facing definition of one circuit: main-trace width, optional
/// preprocessed trace, base and extension constraints, and lookups. The
/// stage-2 width and public-input count are derived (from the lookups and
/// the challenge field's extension degree), not supplied here.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
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

impl<F: Field> CircuitInputs<F> {
    /// Compile constraints for inspection without committing preprocessing.
    /// Lookup constraints remain implicit and are evaluated by the protocol.
    pub fn compile_graph(
        &self,
        params: &ExtensionParams<F>,
    ) -> Result<ConstraintGraph<F>, crate::graph::CompileError> {
        assert!((1..=crate::lookup::MAX_LOOKUP_GROUP).contains(&self.lookup_group_size));
        compile(
            &CircuitSpec {
                main_width: self.main_width,
                preprocessed_width: self.preprocessed.as_ref().map_or(0, |m| m.width()),
                stage2_width: stage2_width(
                    self.lookups.len(),
                    self.lookup_group_size,
                    params.degree,
                ),
                num_publics: num_publics(params.degree),
                constraints: self.constraints.clone(),
                ext_constraints: self.ext_constraints.clone(),
                lookups: self.lookups.clone(),
            },
            params,
        )
    }
}

/// A compiled circuit within the system, with the metadata the prover and
/// verifier need. The preprocessed trace is retained for witness-time
/// lookup evaluation (it is also committed at setup).
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
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

    /// Quotient degree multiple for non-ZK (`is_zk == false`) or `v6-zk`
    /// (`is_zk == true`).
    ///
    /// In `v6-zk`, 2× odd-coset witness blinding yields trace polynomials of
    /// degree `< 2N` so `deg(Q) <= (2d - 1)N - d + 1 < 2N · q_zk` where
    /// `q_zk = max(d, 1).next_power_of_two()` (matching Plonky3's
    /// `get_log_num_quotient_chunks(..., is_zk = 1)`).
    pub fn quotient_degree_for(&self, is_zk: bool) -> usize {
        if is_zk {
            self.max_constraint_degree().max(1).next_power_of_two()
        } else {
            self.quotient_degree()
        }
    }

    /// Returns `true` if any compiled node reads a column from `source` at `offset`.
    pub fn uses_row(&self, source: Source, offset: RowOffset) -> bool {
        self.graph.uses_row(source, offset)
    }
}

/// A multi-circuit STARK system over compiled constraint circuits. Contains
/// all circuits together with their shared preprocessed commitment, canonical
/// verifying-key / constraint-graph digest, and the protocol configuration.
pub struct System<SC: ProofConfig> {
    pub config: SC,
    pub circuits: Vec<Circuit<Val<SC>>>,
    /// Commitment to all preprocessed traces (if any circuit has one).
    pub preprocessed_commit: Option<Com<SC>>,
    /// Maps each circuit index to its position within the preprocessed
    /// commitment; `None` if that circuit has no preprocessed trace.
    pub preprocessed_indices: Vec<Option<usize>>,
    /// Canonical BLAKE3 `derive_key("multi-stark/v6/vk")` digest committing to
    /// the extension degree/modulus and every circuit's full constraint DAG,
    /// lookup definitions, and metadata (independent of whether preprocessed
    /// trace matrices are retained in RAM).
    pub vk_digest: [u8; 32],
}

/// Prover-side data retained between system setup and proving.
pub struct ProverKey<SC: ProofConfig> {
    /// PCS prover data for the preprocessed traces.
    pub preprocessed_data: Option<PcsData<SC>>,
}

impl<SC: ProofConfig> System<SC> {
    pub fn opens_next_row(&self, circuit: usize, source: Source) -> bool {
        !self.config.omit_unused_next_row_openings()
            || self.circuits[circuit].uses_row(source, RowOffset::Next)
    }

    /// Builds the system from per-circuit inputs.
    ///
    /// # Panics
    /// Panics if a circuit fails to compile, or if its constraint degree
    /// exceeds what the PCS can serve (`max_quotient_degree`).
    pub fn new(
        config: SC,
        inputs: impl IntoIterator<Item: Into<CircuitInputs<Val<SC>>>>,
    ) -> (Self, ProverKey<SC>) {
        Self::new_inner(config, inputs, true)
    }

    /// Commit preprocessing without retaining its evaluation matrices in the
    /// system.
    ///
    /// When disk spilling is enabled on the configuration, preprocessed traces
    /// are streamed one circuit at a time through commitment without holding
    /// all preprocessed matrices in memory simultaneously. Works with both
    /// lazy lookup evaluation ([`SystemWitness::from_stage_1_lazy`],
    /// [`crate::witness::PreparedWitness::from_stage_1_lazy`],
    /// [`Self::prove_stage_1_streaming`]) and `prove_committed`.
    pub fn new_without_preprocessed(
        config: SC,
        inputs: impl IntoIterator<Item: Into<CircuitInputs<Val<SC>>>>,
    ) -> (Self, ProverKey<SC>) {
        Self::new_inner(config, inputs, false)
    }

    fn new_inner(
        config: SC,
        inputs: impl IntoIterator<Item: Into<CircuitInputs<Val<SC>>>>,
        retain_preprocessed: bool,
    ) -> (Self, ProverKey<SC>) {
        let pcs = config.pcs();
        let is_zk = config.is_zk();
        if is_zk {
            assert!(
                config.zk_randomizer_width() >= crate::p3_adapter::zk::ZK_RANDOMIZER_WIDTH,
                "v6-zk requires randomizer width W_R >= {} (got {})",
                crate::p3_adapter::zk::ZK_RANDOMIZER_WIDTH,
                config.zk_randomizer_width()
            );
            assert!(
                config.log_blowup() >= 1,
                "v6-zk requires log_blowup >= 1 (got {})",
                config.log_blowup()
            );
        }
        let min_zk_height = if is_zk {
            1usize << config.min_log_degree()
        } else {
            0
        };
        let max_quotient_degree = config.max_quotient_degree();
        let params = extension_params::<SC>();
        let d = params.degree;

        let mut circuits = vec![];
        let mut preprocessed_indices = vec![];
        let mut next_preprocessed_index = 0usize;

        let mut prep_iter = inputs.into_iter().enumerate().filter_map(|(i, input)| {
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
            let mut circuit = Circuit {
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
            // `max_quotient_degree` times the committed trace domain (the
            // blowup factor for FRI). Beyond that, proving would silently
            // produce invalid proofs, so reject the circuit upfront.
            let required_quotient_degree = circuit.quotient_degree_for(is_zk);
            assert!(
                required_quotient_degree <= max_quotient_degree,
                "circuit {i}: constraint degree {} needs quotient degree {}, but the PCS only \
                 supports {}; increase log_blowup or lower the constraint degree",
                circuit.max_constraint_degree(),
                required_quotient_degree,
                max_quotient_degree,
            );
            if is_zk && circuit.preprocessed_height > 0 {
                assert!(
                    circuit.preprocessed_height >= min_zk_height,
                    "circuit {i}: preprocessed trace height {} is below v6-zk minimum height {}",
                    circuit.preprocessed_height,
                    min_zk_height,
                );
            }

            let item = if let Some(preprocessed) = circuit.preprocessed.take() {
                preprocessed_indices.push(Some(next_preprocessed_index));
                next_preprocessed_index += 1;
                let domain = pcs.natural_domain_for_degree(preprocessed.height());
                let retain_1x =
                    !retain_preprocessed && circuit.graph.lookup_prefix_uses_preprocessed();
                if retain_preprocessed {
                    circuit.preprocessed = Some(preprocessed.clone());
                }
                Some((domain, preprocessed, retain_1x))
            } else {
                preprocessed_indices.push(None);
                None
            };
            circuits.push(circuit);
            item
        });

        let (preprocessed_commit, preprocessed_data) =
            match config.commit_preprocessed(&mut prep_iter) {
                Some((commit, data)) => (Some(commit), Some(data)),
                None => (None, None),
            };
        // Ensure all inputs are drained even if a custom backend stopped early.
        for _ in prep_iter {}

        let vk_domain = match config.protocol_version() {
            ProtocolVersion::V6Zk => Blake3Domain::ZkVerifyingKey,
            ProtocolVersion::V5 | ProtocolVersion::V6 => Blake3Domain::VerifyingKey,
        };
        let vk_digest =
            compute_vk_digest_with_domain::<Val<SC>, SC::Challenge>(&circuits, vk_domain);
        let system = Self {
            config,
            circuits,
            preprocessed_commit,
            preprocessed_indices,
            vk_digest,
        };
        (system, ProverKey { preprocessed_data })
    }

    /// Binds the system shape (and, under protocol `v6` / `v6-zk`, the
    /// extension degree/modulus and canonical VK / constraint-graph digest)
    /// into the Fiat-Shamir transcript. The prover and the verifier must call
    /// this identically, before observing any commitment, so that transcripts
    /// of systems with different constraint graphs or circuit shapes never
    /// collide. The protocol parameters are bound separately via the
    /// challenger seed (see [`ProofConfig::initialise_challenger_for_mode`]).
    pub fn observe_shape(&self, challenger: &mut SC::Challenger) {
        if matches!(
            self.config.protocol_version(),
            ProtocolVersion::V6 | ProtocolVersion::V6Zk
        ) {
            challenger.observe_field(Val::<SC>::from_usize(
                <SC::Challenge as ExtensionOf<Val<SC>>>::D,
            ));
            let modulus = <SC::Challenge as ExtensionOf<Val<SC>>>::MODULUS;
            challenger.observe_field(Val::<SC>::from_u8(modulus.tag()));
            if let ExtensionModulus::Binomial { w } = modulus {
                challenger.observe_field(w);
            }
            challenger.observe_digest(self.vk_digest);
        }
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

/// Computes the canonical protocol-`v6` verifying-key / constraint-graph digest
/// under `BLAKE3_derive_key("multi-stark/v6/vk")`.
///
/// Commits to the extension degree, extension modulus, and every circuit's
/// metadata, compiled DAG nodes (including canonical field constants), zero
/// constraint roots, and lookup expressions. Intentionally excludes the
/// in-memory `circuit.preprocessed` matrix so `System::new` and
/// `System::new_without_preprocessed` produce identical digests (the
/// preprocessed traces themselves are bound via `preprocessed_commit`).
pub fn compute_vk_digest<F: Field, EF: ExtensionOf<F>>(circuits: &[Circuit<F>]) -> [u8; 32] {
    compute_vk_digest_with_domain::<F, EF>(circuits, Blake3Domain::VerifyingKey)
}

/// Computes the canonical verifying-key / constraint-graph digest under the
/// requested BLAKE3 `derive_key` domain (`VerifyingKey` for `v6`,
/// `ZkVerifyingKey` for `v6-zk`).
pub fn compute_vk_digest_with_domain<F: Field, EF: ExtensionOf<F>>(
    circuits: &[Circuit<F>],
    domain: Blake3Domain,
) -> [u8; 32] {
    let mut hasher = domain.hasher();
    let bincode_cfg = bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding();

    let hash_u32 = |h: &mut blake3::Hasher, x: u32| {
        h.update(&x.to_le_bytes());
    };
    let hash_u64 = |h: &mut blake3::Hasher, x: u64| {
        h.update(&x.to_le_bytes());
    };
    let hash_usize = |h: &mut blake3::Hasher, x: usize| {
        hash_u64(h, u64::try_from(x).expect("usize exceeds u64"));
    };
    let hash_field = |h: &mut blake3::Hasher, x: F| {
        let bytes =
            bincode::serde::encode_to_vec(x, bincode_cfg).expect("field serialization cannot fail");
        h.update(&bytes);
    };

    hash_usize(&mut hasher, EF::D);
    hasher.update(&[EF::MODULUS.tag()]);
    if let ExtensionModulus::Binomial { w } = EF::MODULUS {
        hash_field(&mut hasher, w);
    }

    hash_usize(&mut hasher, circuits.len());
    for circuit in circuits {
        hash_usize(&mut hasher, circuit.main_width);
        hash_usize(&mut hasher, circuit.preprocessed_width);
        hash_usize(&mut hasher, circuit.preprocessed_height);
        hash_usize(&mut hasher, circuit.num_lookups);
        hash_usize(&mut hasher, circuit.stage_2_width);
        hash_usize(&mut hasher, circuit.num_publics);
        hash_usize(&mut hasher, circuit.lookup_group_size);
        hash_usize(&mut hasher, circuit.constraint_count);
        hash_usize(&mut hasher, circuit.max_constraint_degree);
        hash_usize(&mut hasher, circuit.graph.lookup_prefix_len);
        hash_u32(&mut hasher, circuit.graph.max_constraint_degree);

        hash_usize(&mut hasher, circuit.graph.nodes.len());
        for node in &circuit.graph.nodes {
            match *node {
                Node::Const(c) => {
                    hasher.update(&[0u8]);
                    hash_field(&mut hasher, c);
                }
                Node::Var(col) => {
                    let src_tag = match col.source {
                        Source::Preprocessed => 0u8,
                        Source::Main => 1u8,
                        Source::Stage2 => 2u8,
                    };
                    let off_tag = match col.offset {
                        RowOffset::Current => 0u8,
                        RowOffset::Next => 1u8,
                    };
                    hasher.update(&[1u8, src_tag, off_tag]);
                    hash_u32(&mut hasher, col.index);
                }
                Node::Public(idx) => {
                    hasher.update(&[2u8]);
                    hash_u32(&mut hasher, idx);
                }
                Node::IsFirstRow => {
                    hasher.update(&[3u8]);
                }
                Node::IsLastRow => {
                    hasher.update(&[4u8]);
                }
                Node::IsTransition => {
                    hasher.update(&[5u8]);
                }
                Node::Add(a, b) => {
                    hasher.update(&[6u8]);
                    hash_u32(&mut hasher, a.0);
                    hash_u32(&mut hasher, b.0);
                }
                Node::Sub(a, b) => {
                    hasher.update(&[7u8]);
                    hash_u32(&mut hasher, a.0);
                    hash_u32(&mut hasher, b.0);
                }
                Node::Mul(a, b) => {
                    hasher.update(&[8u8]);
                    hash_u32(&mut hasher, a.0);
                    hash_u32(&mut hasher, b.0);
                }
                Node::Neg(a) => {
                    hasher.update(&[9u8]);
                    hash_u32(&mut hasher, a.0);
                }
            }
        }

        hash_usize(&mut hasher, circuit.graph.zeros.len());
        for z in &circuit.graph.zeros {
            hash_u32(&mut hasher, z.0);
        }

        hash_usize(&mut hasher, circuit.graph.lookups.len());
        for lookup in &circuit.graph.lookups {
            hash_u32(&mut hasher, lookup.multiplicity.0);
            hash_usize(&mut hasher, lookup.args.len());
            for arg in &lookup.args {
                hash_u32(&mut hasher, arg.0);
            }
        }
    }

    hasher.finalize().into()
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
    /// if a circuit with a preprocessed trace receives a non-empty main trace
    /// of a different height (both traces are opened on the same domain, so
    /// their heights must match; the rows would otherwise be silently
    /// truncated). An empty main trace deactivates the circuit for this
    /// witness, preprocessed or not.
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
        let started = std::time::Instant::now();
        let lookups = traces
            .iter()
            .zip(&system.circuits)
            .enumerate()
            .map(|(i, (trace, circuit))| {
                if let Some(preprocessed) = &circuit.preprocessed
                    && trace.height() != 0
                {
                    assert_eq!(
                        trace.height(),
                        preprocessed.height(),
                        "circuit {i}: main trace height must equal preprocessed trace height"
                    );
                }
                compute_lookup_values(circuit, trace)
            })
            .collect();
        tracing::info!(
            circuits = traces.len(),
            seconds = started.elapsed().as_secs_f64(),
            "Lookup expressions materialized"
        );
        Self { traces, lookups }
    }

    /// Builds a lazy witness from the stage-1 traces without materializing
    /// [`LookupValues`] upfront.
    ///
    /// Lookup expressions are evaluated on demand in bounded row chunks during
    /// stage-2 trace construction instead of holding concrete lookup arguments
    /// and multiplicities in RAM alongside the traces and LDEs. Produces proofs
    /// byte-identical to [`Self::from_stage_1`].
    ///
    /// # Panics
    /// Panics under the same shape checks as [`Self::from_stage_1`]: trace
    /// count must match circuit count, and any active circuit with a
    /// preprocessed trace must match its preprocessed height.
    pub fn from_stage_1_lazy<SC>(traces: Vec<RowMajorMatrix<F>>, system: &System<SC>) -> Self
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
                let height = trace.height();
                if circuit.preprocessed_height != 0 && height != 0 {
                    assert_eq!(
                        height, circuit.preprocessed_height,
                        "circuit {i}: main trace height must equal preprocessed trace height"
                    );
                }
                let slot_widths: Vec<usize> = circuit
                    .graph
                    .lookups
                    .iter()
                    .map(|lookup| lookup.args.len())
                    .collect();
                LookupValues::shape_only(height, &slot_widths)
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
    compute_lookup_values_with_fixed(circuit, trace, circuit.preprocessed.as_ref())
}

pub(crate) fn compute_lookup_values_with_fixed<F: Field>(
    circuit: &Circuit<F>,
    trace: &RowMajorMatrix<F>,
    preprocessed: Option<&RowMajorMatrix<F>>,
) -> LookupValues<F> {
    compute_lookup_values_range(&circuit.graph, trace, preprocessed, 0..trace.height())
}

/// Evaluate lookup messages for a row range. Selectors and next-row reads use
/// the full trace height, including wraparound at its last row.
pub fn compute_lookup_values_range<F: Field>(
    graph: &ConstraintGraph<F>,
    trace: &RowMajorMatrix<F>,
    preprocessed: Option<&RowMajorMatrix<F>>,
    rows: std::ops::Range<usize>,
) -> LookupValues<F> {
    let height = trace.height();
    assert!(rows.start <= rows.end && rows.end <= height);
    let slot_widths: Vec<usize> = graph
        .lookups
        .iter()
        .map(|lookup| lookup.args.len())
        .collect();
    // No rows, or no lookups: nothing to sweep, but preserve num_lookups.
    if height == 0 || slot_widths.is_empty() {
        return LookupValues::builder(rows.len(), &slot_widths).finish();
    }

    let empty: [F; 0] = [];
    let mut builder = LookupValues::builder(rows.len(), &slot_widths);
    builder
        .par_rows_mut()
        .enumerate()
        .with_min_len(1 << 12)
        .for_each_init(
            || (Vec::new(), Vec::new()),
            |(buf, args), (offset, mut writer)| {
                // Selectors and the next-row window refer to the full trace,
                // even when only a range of lookup rows is materialized.
                let r = rows.start + offset;
                let r_next = (r + 1) % height;
                let main_cur = trace.row_slice(r).unwrap();
                let main_next = trace.row_slice(r_next).unwrap();
                let preprocessed_rows = preprocessed
                    .map(|pp| (pp.row_slice(r).unwrap(), pp.row_slice(r_next).unwrap()));
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
                graph.sweep_lookup_prefix(&view, buf);
                for (slot, lookup) in graph.lookups.iter().enumerate() {
                    let multiplicity = buf[lookup.multiplicity.index()];
                    args.clear();
                    args.extend(lookup.args.iter().map(|a| buf[a.index()]));
                    writer.push(slot, multiplicity, args);
                }
            },
        );
    builder.finish()
}

/// The extension parameters of the challenge field, read off [`ExtensionOf`].
pub fn extension_params<SC: ProofConfig>() -> ExtensionParams<Val<SC>> {
    ExtensionParams::for_extension::<SC::Challenge>()
}

#[cfg(test)]
mod tests {
    mod lookup_values;

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
