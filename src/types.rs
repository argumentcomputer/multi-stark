//! The reference STARK configuration: Goldilocks field with a degree-2
//! binomial extension, Blake3 hashing, and a FRI-based PCS.
//!
//! The generic protocol lives in [`crate::config`], [`crate::system`],
//! [`crate::prover`] and [`crate::verifier`]; this module only provides a
//! concrete, batteries-included instantiation.

use crate::config::StarkGenericConfig;
use p3_blake3::Blake3;
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger, HashChallenger,
    SerializingChallenger64,
};
use p3_commit::{ExtensionMmcs, Pcs as PcsTrait};
use p3_dft::Radix2DitParallel;
use p3_field::BasedVectorSpace;
use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField,
    extension::BinomialExtensionField,
};
use p3_fri::FriParameters as InnerFriParameters;
#[cfg(not(feature = "cuda"))]
use p3_fri::TwoAdicFriPcs;
use p3_goldilocks::Goldilocks;
#[cfg(feature = "cuda")]
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

#[cfg(feature = "cuda")]
use crate::cuda::CudaDft;
#[cfg(feature = "cuda")]
use crate::cuda::pcs::CudaPcsDft;

pub type Val = Goldilocks;
pub type PackedVal = <Val as Field>::Packing;
pub type ExtVal = BinomialExtensionField<Val, 2>;
pub type PackedExtVal = <ExtVal as ExtensionField<Val>>::ExtensionPacking;
pub type Challenger =
    DeterministicPow<SerializingChallenger64<Val, HashChallenger<u8, Blake3, 32>>>;

/// Forwarding challenger wrapper that makes zero-bit proof-of-work
/// deterministic.
///
/// p3's `SerializingChallenger64::grind` races `find_any` over every
/// candidate witness even at 0 bits — where EVERY candidate passes — so the
/// returned witness is whichever a rayon thread reports first: a
/// thread-race-dependent value that lands in the proof bytes and makes
/// zero-PoW proofs nondeterministic run to run. (The duplex challenger
/// special-cases this; the serializing one doesn't.) Zero bits mean any
/// witness verifies, so return the canonical ZERO; positive bit counts
/// delegate to the inner grind unchanged. Transcript semantics are
/// untouched — `check_witness` at 0 bits observes nothing on either side.
#[derive(Clone, Debug)]
pub struct DeterministicPow<C>(pub C);

impl Challenger {
    pub fn from_hasher(initial_state: Vec<u8>, hasher: Blake3) -> Self {
        Self(SerializingChallenger64::from_hasher(initial_state, hasher))
    }
}

impl<T, C: CanObserve<T>> CanObserve<T> for DeterministicPow<C> {
    fn observe(&mut self, value: T) {
        self.0.observe(value);
    }
}

impl<T, C: CanSample<T>> CanSample<T> for DeterministicPow<C> {
    fn sample(&mut self) -> T {
        self.0.sample()
    }
}

impl<C: CanSampleBits<usize>> CanSampleBits<usize> for DeterministicPow<C> {
    fn sample_bits(&mut self, bits: usize) -> usize {
        self.0.sample_bits(bits)
    }
}

impl<F: Field, C: FieldChallenger<F>> FieldChallenger<F> for DeterministicPow<C> {}

impl<C: GrindingChallenger> GrindingChallenger for DeterministicPow<C> {
    type Witness = C::Witness;

    fn grind(&mut self, bits: usize) -> Self::Witness {
        if bits == 0 {
            return Self::Witness::ZERO;
        }
        self.0.grind(bits)
    }
}
type CpuMmcs = MerkleTreeMmcs<Val, u8, SerializingHasher<Blake3>, Blake3CompressionFunction, 2, 32>;
#[cfg(not(feature = "cuda"))]
pub type Mmcs = CpuMmcs;
#[cfg(feature = "cuda")]
pub type Mmcs = crate::cuda::mmcs::CudaMmcs;
pub type ExtMmcs = ExtensionMmcs<Val, ExtVal, Mmcs>;
#[cfg(not(feature = "cuda"))]
pub type Pcs = TwoAdicFriPcs<Val, PcsDft, Mmcs, ExtMmcs>;
#[cfg(feature = "cuda")]
pub type Pcs = crate::cuda::pcs::CudaTwoAdicFriPcs<Val, PcsDft, Mmcs, ExtMmcs>;

pub type Commitment = <Pcs as PcsTrait<ExtVal, Challenger>>::Commitment;
pub type Domain = <Pcs as PcsTrait<ExtVal, Challenger>>::Domain;
pub type ProverData = <Pcs as PcsTrait<ExtVal, Challenger>>::ProverData;
pub type EvaluationsOnDomain<'a> = <Pcs as PcsTrait<ExtVal, Challenger>>::EvaluationsOnDomain<'a>;
pub type PcsError = <Pcs as PcsTrait<ExtVal, Challenger>>::Error;
pub type PcsProof = <Pcs as PcsTrait<ExtVal, Challenger>>::Proof;

#[cfg(feature = "cuda")]
fn cuda_coset_selectors(
    trace_domain: Domain,
    quotient_domain: Domain,
) -> crate::cuda::CudaCosetSelectors {
    let rate_bits = quotient_domain.log_size() - trace_domain.log_size();
    crate::cuda::CudaCosetSelectors {
        coset_shift: quotient_domain.shift(),
        coset_generator: quotient_domain.subgroup_generator(),
        trace_last: trace_domain.subgroup_generator().inverse(),
        vanishing_start: quotient_domain
            .shift()
            .exp_power_of_2(trace_domain.log_size()),
        vanishing_step: Val::two_adic_generator(rate_bits),
    }
}

#[cfg(feature = "cuda")]
fn cuda_worker_count(variable: &str, numerator: usize, denominator: usize) -> usize {
    let default_threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .saturating_mul(numerator)
        .div_ceil(denominator);
    std::env::var(variable)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads != 0)
        .unwrap_or(default_threads)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_stage1_worker_count(prioritize_deferred: bool) -> usize {
    std::env::var("MULTI_STARK_CUDA_DEFERRED_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads != 0)
        .unwrap_or_else(|| {
            let (numerator, denominator) = if prioritize_deferred { (5, 8) } else { (3, 8) };
            cuda_worker_count("MULTI_STARK_CUDA_STAGE1_THREADS", numerator, denominator)
        })
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_lookup_worker_count() -> usize {
    cuda_worker_count("MULTI_STARK_CUDA_LOOKUP_THREADS", 1, 4)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_host_pool(name: &'static str, threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |index| format!("cuda-{name}-{index}"))
        .build()
        .expect("failed to build CUDA host worker pool")
}

/// The reference [`StarkGenericConfig`] implementation.
pub struct GoldilocksBlake3Config {
    /// The PCS used to commit polynomials and prove opening proofs.
    pcs: Pcs,
    /// The same transform implementation used inside `pcs`, exposed for
    /// prover-side quotient transforms which live outside the PCS API.
    dft: Dft,
    /// Seed for fresh challengers: a domain-separation tag followed by a
    /// digest of all protocol parameters.
    challenger_seed: Vec<u8>,
    /// Largest log2 degree the PCS can commit to and open.
    max_log_degree: usize,
    /// Largest quotient degree the PCS can serve trace evaluations for
    /// (the FRI blowup factor).
    max_quotient_degree: usize,
    /// Log2 of the blowup the PCS applies when committing.
    log_blowup: usize,
}

impl GoldilocksBlake3Config {
    pub fn new(commitment_parameters: CommitmentParameters, fri_parameters: FriParameters) -> Self {
        #[cfg(feature = "cuda")]
        {
            assert_eq!(
                commitment_parameters.cap_height, 0,
                "the CUDA backend currently supports only cap_height = 0"
            );
            assert!(
                fri_parameters.max_log_arity <= 1,
                "the CUDA backend currently supports only binary FRI folds"
            );
        }
        let (pcs, dft) = new_pcs(commitment_parameters, fri_parameters);
        // Seed the challenger with a protocol tag for domain separation,
        // followed by every protocol parameter. Binding the parameters into
        // the seed means transcripts produced under different parameters
        // never collide (see the transcript contract on
        // [`StarkGenericConfig::initialise_challenger`]).
        let mut challenger_seed = b"multi-stark/v0".to_vec();
        for parameter in [
            commitment_parameters.log_blowup,
            commitment_parameters.cap_height,
            fri_parameters.log_final_poly_len,
            fri_parameters.max_log_arity,
            fri_parameters.num_queries,
            fri_parameters.commit_proof_of_work_bits,
            fri_parameters.query_proof_of_work_bits,
        ] {
            let parameter = u64::try_from(parameter).expect("parameter exceeds u64");
            challenger_seed.extend_from_slice(&parameter.to_le_bytes());
        }
        let max_log_degree = Val::TWO_ADICITY - commitment_parameters.log_blowup;
        let max_quotient_degree = 1 << commitment_parameters.log_blowup;
        Self {
            pcs,
            dft,
            challenger_seed,
            max_log_degree,
            max_quotient_degree,
            log_blowup: commitment_parameters.log_blowup,
        }
    }
}

impl StarkGenericConfig for GoldilocksBlake3Config {
    type Pcs = Pcs;
    type Dft = Dft;
    type Challenge = ExtVal;
    type Challenger = Challenger;

    fn pcs(&self) -> &Pcs {
        &self.pcs
    }

    fn dft(&self) -> &Dft {
        &self.dft
    }

    fn initialise_challenger(&self) -> Challenger {
        Challenger::from_hasher(self.challenger_seed.clone(), Blake3)
    }

    fn max_log_degree(&self) -> usize {
        self.max_log_degree
    }

    fn max_quotient_degree(&self) -> usize {
        self.max_quotient_degree
    }

    fn log_blowup(&self) -> usize {
        self.log_blowup
    }

    fn canonicalize_proof(proof: &mut crate::prover::Proof<Self>) {
        fn canonical_base(value: &mut Val) {
            *value = Val::from_u64(value.as_canonical_u64());
        }

        fn canonical_ext(value: &mut ExtVal) {
            let coefficients: &[Val] = value.as_basis_coefficients_slice();
            *value = ExtVal::from_basis_coefficients_slice(&[
                Val::from_u64(coefficients[0].as_canonical_u64()),
                Val::from_u64(coefficients[1].as_canonical_u64()),
            ])
            .expect("quadratic extension element");
        }

        fn canonical_opened(round: &mut p3_commit::OpenedValuesForRound<ExtVal>) {
            for matrix in round {
                for point in matrix {
                    point.iter_mut().for_each(canonical_ext);
                }
            }
        }

        proof
            .intermediate_accumulators
            .iter_mut()
            .for_each(canonical_ext);
        canonical_opened(&mut proof.quotient_opened_values);
        if let Some(round) = &mut proof.preprocessed_opened_values {
            canonical_opened(round);
        }
        canonical_opened(&mut proof.stage_1_opened_values);
        canonical_opened(&mut proof.stage_2_opened_values);

        let fri = &mut proof.opening_proof;
        fri.commit_pow_witnesses.iter_mut().for_each(canonical_base);
        canonical_base(&mut fri.query_pow_witness);
        fri.final_poly.iter_mut().for_each(canonical_ext);
        for opening in &mut fri.input_openings {
            for query in &mut opening.opened_values {
                for row in query {
                    row.iter_mut().for_each(canonical_base);
                }
            }
        }
        for step in &mut fri.commit_phase_openings {
            for siblings in &mut step.sibling_values {
                siblings.iter_mut().for_each(canonical_ext);
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn accelerated_quotient_values(
        &self,
        circuit: &crate::system::Circuit<Val>,
        lookup_publics: &[Val],
        trace_domain: crate::config::Domain<Self>,
        quotient_domain: crate::config::Domain<Self>,
        preprocessed: Option<(&crate::config::PcsData<Self>, usize)>,
        stage_1: (&crate::config::PcsData<Self>, usize),
        stage_2: (&crate::config::PcsData<Self>, usize),
        alpha: ExtVal,
        constraint_count: usize,
    ) -> Option<Vec<ExtVal>> {
        let main = stage_1.0.resident(stage_1.1)?;
        let s2 = stage_2.0.resident(stage_2.1)?;
        let prep = match preprocessed {
            Some((data, index)) => Some(data.resident(index)?),
            None => None,
        };
        let qsize = quotient_domain.size();
        let selectors = cuda_coset_selectors(trace_domain, quotient_domain);
        let mut powers = Vec::with_capacity(constraint_count);
        let mut power = ExtVal::ONE;
        for _ in 0..constraint_count {
            powers.push(power);
            power *= alpha;
        }
        powers.reverse();
        let mut alpha_flat = Vec::with_capacity(2 * constraint_count);
        for coordinate in 0..2 {
            alpha_flat.extend(powers.iter().map(|x| {
                <ExtVal as BasedVectorSpace<Val>>::as_basis_coefficients_slice(x)[coordinate]
            }));
        }
        let n = Val::from_usize(trace_domain.size());
        let g = Val::two_adic_generator(trace_domain.size().ilog2() as usize);
        let norm = (n * g).inverse();
        let delta = [
            (lookup_publics[6] - lookup_publics[4]) * norm,
            (lookup_publics[7] - lookup_publics[5]) * norm,
        ];
        let next_step = 1usize << (quotient_domain.size().ilog2() - trace_domain.size().ilog2());
        let flat = crate::cuda::quotient_values_resident(
            &circuit.graph,
            prep,
            main,
            s2,
            lookup_publics,
            selectors,
            &alpha_flat,
            &delta,
            crate::system::extension_params::<Self>().w,
            qsize,
            next_step,
            circuit.lookup_group_size,
        );
        Some(
            flat.values
                .as_chunks::<2>()
                .0
                .iter()
                .map(|coords| {
                    ExtVal::from_basis_coefficients_slice(coords)
                        .expect("CUDA quotient has two coordinates")
                })
                .collect(),
        )
    }

    #[cfg(feature = "cuda")]
    fn accelerated_quotient_commit(
        &self,
        inputs: &[crate::config::QuotientCommitInput<'_, Self>],
        alpha: ExtVal,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        use crate::cuda::mmcs::CudaCommitMmcs;

        let (_, total_device_bytes) =
            crate::cuda::device_memory_info(self.pcs.mmcs.cuda_device_id());
        let matrix_width = |data: &crate::config::PcsData<Self>, index: usize| {
            let dimensions = self.pcs.mmcs.matrix_dimensions(data);
            dimensions
                .get(index)
                .expect("matrix index out of bounds")
                .width
        };
        let staging_bytes = |input: &crate::config::QuotientCommitInput<'_, Self>| {
            let mut host_width = 0usize;
            for (data, index) in [Some(input.stage_1), Some(input.stage_2), input.preprocessed]
                .into_iter()
                .flatten()
            {
                if !self.pcs.mmcs.is_matrix_cuda_resident(data, index) {
                    host_width = host_width.saturating_add(matrix_width(data, index));
                }
            }
            if host_width == 0 {
                return 0;
            }
            let row_bytes = host_width
                .saturating_mul(2)
                .saturating_mul(size_of::<Val>());
            let per_buffer = input
                .quotient_domain
                .size()
                .saturating_mul(row_bytes)
                .min(512usize << 20);
            2usize.saturating_mul(per_buffer)
        };
        let mut quotient_jobs = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let quotient_size = input.quotient_domain.size();
                let quotient_degree = input.circuit.quotient_degree();
                let trace_height = quotient_size / quotient_degree;
                let lde_height = trace_height
                    .checked_shl(u32::try_from(self.log_blowup).expect("LDE blowup exceeds u32"))
                    .unwrap_or(usize::MAX);
                let (output_bytes, kernel_workspace) = crate::cuda::quotient_lde_memory_upper_bound(
                    &input.circuit.graph,
                    input.lookup_publics.len(),
                    input.constraint_count,
                    quotient_size,
                    quotient_degree,
                    self.log_blowup,
                );
                let current_staging = staging_bytes(input);
                let constant_bytes = quotient_size
                    .saturating_add(lde_height)
                    .saturating_div(2)
                    .saturating_add(quotient_degree)
                    .saturating_mul(size_of::<Val>());
                (
                    index,
                    output_bytes,
                    kernel_workspace,
                    constant_bytes,
                    current_staging
                        .saturating_add(kernel_workspace)
                        .saturating_add(output_bytes)
                        .saturating_add(constant_bytes),
                    lde_height,
                )
            })
            .collect::<Vec<_>>();
        quotient_jobs.sort_unstable_by_key(|&(index, _, _, _, peak, _)| {
            let input = &inputs[index];
            let ready = [Some(input.stage_1), Some(input.stage_2), input.preprocessed]
                .into_iter()
                .flatten()
                .all(|(data, matrix)| self.pcs.mmcs.is_matrix_source_ready(data, matrix));
            (!ready, core::cmp::Reverse(peak))
        });
        let accumulated_output = quotient_jobs
            .iter()
            .map(|&(_, output, _, _, _, _)| output)
            .sum::<usize>();
        let max_lde_height = quotient_jobs
            .iter()
            .map(|&(_, _, _, _, _, lde_height)| lde_height)
            .max()
            .unwrap_or(0);
        if crate::cuda::memory_diagnostics_enabled() {
            eprintln!(
                "[multi-stark/cuda] quotient jobs={} output={}",
                quotient_jobs.len(),
                accumulated_output,
            );
        }

        let mut ldes = (0..inputs.len()).map(|_| None).collect::<Vec<_>>();
        for (index, output_bytes, kernel_workspace, constant_bytes, _, lde_height) in quotient_jobs
        {
            let job_started = std::time::Instant::now();
            let input = &inputs[index];
            let current_staging = || staging_bytes(input);
            let required = || {
                current_staging()
                    .saturating_add(kernel_workspace)
                    .saturating_add(output_bytes)
                    .saturating_add(constant_bytes)
                    .saturating_add(total_device_bytes / 64)
            };
            let mut target = required();
            self.pcs
                .mmcs
                .ensure_device_headroom(input.stage_1.0, target, Some(input.stage_1.1));
            target = required();
            self.pcs
                .mmcs
                .ensure_device_headroom(input.stage_2.0, target, Some(input.stage_2.1));
            if let Some((data, matrix)) = input.preprocessed {
                target = required();
                self.pcs
                    .mmcs
                    .ensure_device_headroom(data, target, Some(matrix));
            }
            target = required();
            let free_bytes = crate::cuda::device_memory_info(self.pcs.mmcs.cuda_device_id()).0;
            if crate::cuda::memory_diagnostics_enabled() {
                eprintln!(
                    "[multi-stark/cuda] quotient job {index}: lde_height={lde_height} output={output_bytes} workspace={kernel_workspace} staging={} target={} free={free_bytes}",
                    current_staging(),
                    target
                );
            }
            if free_bytes < target {
                return None;
            }
            let lde =
                {
                    let quotient_size = input.quotient_domain.size();
                    let quotient_degree = input.circuit.quotient_degree();
                    let selectors = cuda_coset_selectors(input.trace_domain, input.quotient_domain);

                    let mut powers = Vec::with_capacity(input.constraint_count);
                    let mut power = ExtVal::ONE;
                    for _ in 0..input.constraint_count {
                        powers.push(power);
                        power *= alpha;
                    }
                    powers.reverse();
                    let mut alpha_flat = Vec::with_capacity(2 * input.constraint_count);
                    for coordinate in 0..2 {
                        alpha_flat.extend(powers.iter().map(|value| {
                            <ExtVal as BasedVectorSpace<Val>>::as_basis_coefficients_slice(value)
                                [coordinate]
                        }));
                    }
                    let trace_size = input.trace_domain.size();
                    let n = Val::from_usize(trace_size);
                    let generator = Val::two_adic_generator(trace_size.ilog2() as usize);
                    let normalization = (n * generator).inverse();
                    let delta = [
                        (input.lookup_publics[6] - input.lookup_publics[4]) * normalization,
                        (input.lookup_publics[7] - input.lookup_publics[5]) * normalization,
                    ];
                    let next_step = quotient_size / trace_size;
                    let evaluate = |main: crate::cuda::mmcs::CudaMatrixSource<'_>,
                                    stage2: crate::cuda::mmcs::CudaMatrixSource<'_>,
                                    preprocessed: Option<
                        crate::cuda::mmcs::CudaMatrixSource<'_>,
                    >| {
                        crate::cuda::quotient_lde_mixed(
                            &self.pcs.dft,
                            &input.circuit.graph,
                            preprocessed,
                            main,
                            stage2,
                            &input.lookup_publics,
                            selectors,
                            &alpha_flat,
                            &delta,
                            crate::system::extension_params::<Self>().w,
                            quotient_size,
                            next_step,
                            input.circuit.lookup_group_size,
                            quotient_degree,
                            self.log_blowup,
                        )
                    };
                    self.pcs
                        .mmcs
                        .with_matrix_source(input.stage_1.0, input.stage_1.1, |main| {
                            self.pcs.mmcs.with_matrix_source(
                                input.stage_2.0,
                                input.stage_2.1,
                                |stage2| match input.preprocessed {
                                    Some((data, index)) => self.pcs.mmcs.with_matrix_source(
                                        data,
                                        index,
                                        |preprocessed| evaluate(main, stage2, Some(preprocessed)),
                                    ),
                                    None => evaluate(main, stage2, None),
                                },
                            )
                        })
                };
            ldes[index] = Some(lde);
            if crate::cuda::memory_diagnostics_enabled() {
                eprintln!(
                    "[multi-stark/cuda] quotient job {index} complete: {:.3}s",
                    job_started.elapsed().as_secs_f64()
                );
            }
        }
        let tree_headroom = max_lde_height
            .saturating_mul(96)
            .saturating_add(total_device_bytes / 64);
        if let Some(input) = inputs.first() {
            self.pcs
                .mmcs
                .ensure_device_headroom(input.stage_1.0, tree_headroom, None);
            self.pcs
                .mmcs
                .ensure_device_headroom(input.stage_2.0, tree_headroom, None);
        }
        if let Some((data, _)) = inputs.iter().find_map(|input| input.preprocessed) {
            self.pcs
                .mmcs
                .ensure_device_headroom(data, tree_headroom, None);
        }
        if crate::cuda::device_memory_info(self.pcs.mmcs.cuda_device_id()).0 < tree_headroom {
            return None;
        }
        Some(
            self.pcs.mmcs.commit_cuda_spillable(
                ldes.into_iter()
                    .map(|lde| lde.expect("quotient job did not run"))
                    .collect(),
            ),
        )
    }

    #[cfg(feature = "cuda")]
    fn accelerated_lookup_commit(
        &self,
        inputs: &[crate::config::LookupCommitInput<'_, Self>],
        lookup_challenge: ExtVal,
        fingerprint_challenge: ExtVal,
        mut accumulator: ExtVal,
    ) -> Option<(
        crate::config::Com<Self>,
        crate::config::PcsData<Self>,
        Vec<ExtVal>,
    )> {
        use crate::cuda::mmcs::CudaCommitMmcs;

        let mut lookup_jobs = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let (height, num_lookups, _, _, arg_offsets) = input.lookup_values.cuda_parts();
                let group_size = input.circuit.lookup_group_size.max(1);
                let groups = num_lookups.div_ceil(group_size).max(1);
                let extended_height = height
                    .checked_shl(u32::try_from(self.log_blowup).expect("LDE blowup exceeds u32"))
                    .unwrap_or(usize::MAX);
                let output_bytes = extended_height
                    .saturating_mul(2)
                    .saturating_mul(groups)
                    .saturating_mul(size_of::<Val>());
                let direct_temporary_bytes = if num_lookups == 0 {
                    0
                } else {
                    let args_width = *arg_offsets.last().expect("lookup offsets are empty");
                    let chunk_rows = height.min(1 << 16);
                    let messages = chunk_rows.saturating_mul(num_lookups);
                    messages
                        // Multiplicity, conjugate, norm, and inverse-norm arrays.
                        .saturating_mul(5 * size_of::<Val>())
                        .saturating_add(
                            chunk_rows
                                .saturating_mul(args_width)
                                .saturating_mul(size_of::<Val>()),
                        )
                        // Extension-field deltas before the exclusive scan.
                        .saturating_add(
                            height
                                .saturating_mul(groups)
                                .saturating_mul(2 * size_of::<Val>()),
                        )
                        .saturating_add(arg_offsets.len().saturating_mul(size_of::<usize>()))
                        // Device-cached inverse/forward twiddles and coset
                        // powers may be cold for this height.
                        .saturating_add(
                            height
                                .saturating_add(height / 2)
                                .saturating_add(extended_height / 2)
                                .saturating_mul(size_of::<Val>()),
                        )
                };
                let main_width = self
                    .pcs
                    .mmcs
                    .matrix_dimensions(input.stage_1.0)
                    .get(input.stage_1.1)
                    .expect("stage-1 matrix index out of bounds")
                    .width;
                let graph_memory = (num_lookups != 0)
                    .then(|| {
                        crate::cuda::lookup_graph_lde_memory_upper_bound(
                            &input.circuit.graph,
                            height,
                            main_width,
                            group_size,
                            self.log_blowup,
                        )
                    })
                    .flatten();
                (
                    index,
                    output_bytes,
                    direct_temporary_bytes,
                    graph_memory.map(|(_, temporary)| temporary),
                    extended_height,
                )
            })
            .collect::<Vec<_>>();
        // Permanent lookup LDEs accumulate until their MMCS commitment. Run
        // the initially largest allocation first, while its trace is still
        // resident, then allow that trace to become an eviction candidate for
        // later jobs.
        lookup_jobs.sort_unstable_by_key(|&(index, output, direct, graph, _)| {
            let temporary = if self
                .pcs
                .mmcs
                .is_matrix_cuda_resident(inputs[index].stage_1.0, inputs[index].stage_1.1)
            {
                graph.unwrap_or(direct)
            } else {
                direct
            };
            core::cmp::Reverse(output.saturating_add(temporary))
        });
        let accumulated_output = lookup_jobs
            .iter()
            .map(|&(_, output, _, _, _)| output)
            .sum::<usize>();
        let max_lde_height = lookup_jobs
            .iter()
            .map(|&(_, _, _, _, extended_height)| extended_height)
            .max()
            .unwrap_or(0);
        let (_, total_device_bytes) =
            crate::cuda::device_memory_info(self.pcs.mmcs.cuda_device_id());
        if crate::cuda::memory_diagnostics_enabled() {
            eprintln!(
                "[multi-stark/cuda] lookup jobs={} output={}",
                lookup_jobs.len(),
                accumulated_output,
            );
        }

        let pair = |value: ExtVal| {
            let coordinates = value.as_basis_coefficients_slice();
            [coordinates[0], coordinates[1]]
        };
        let beta = pair(lookup_challenge);
        let gamma = pair(fingerprint_challenge);
        let extension_generator = ExtVal::from_basis_coefficients_slice(&[Val::ZERO, Val::ONE])?;
        let ext_w = pair(extension_generator * extension_generator)[0];
        let lookup_pool =
            std::sync::Arc::new(cuda_host_pool("lookup-rows", cuda_lookup_worker_count()));
        let evaluate = |input: &crate::config::LookupCommitInput<'_, Self>, cooperative: bool| {
            let (height, num_lookups, multiplicities, args, arg_offsets) =
                input.lookup_values.cuda_parts();
            let group_size = input.circuit.lookup_group_size.max(1);
            let main = input.stage_1.0.resident_with_trace(input.stage_1.1);
            let result = if let Some(main) = main.filter(|_| num_lookups != 0) {
                let preprocessed = match input.preprocessed {
                    Some((data, index)) => Some(data.resident(index)?),
                    None => None,
                };
                crate::cuda::lookup_graph_lde_resident(
                    &self.pcs.dft,
                    &input.circuit.graph,
                    preprocessed,
                    main,
                    height,
                    group_size,
                    beta,
                    gamma,
                    ext_w,
                    self.log_blowup,
                )
            } else if cooperative {
                let pool = std::sync::Arc::clone(&lookup_pool);
                Some(crate::cuda::lookup_lde_resident_partitioned(
                    &self.pcs.dft,
                    multiplicities,
                    args,
                    arg_offsets,
                    height,
                    num_lookups,
                    group_size,
                    beta,
                    gamma,
                    ext_w,
                    self.log_blowup,
                    move |rows| {
                        pool.install(|| {
                            input.lookup_values.cuda_stage_2_deltas::<ExtVal>(
                                rows,
                                group_size,
                                lookup_challenge,
                                &fingerprint_challenge,
                            )
                        })
                    },
                ))
            } else {
                Some(crate::cuda::lookup_lde_resident(
                    &self.pcs.dft,
                    multiplicities,
                    args,
                    arg_offsets,
                    height,
                    num_lookups,
                    group_size,
                    beta,
                    gamma,
                    ext_w,
                    self.log_blowup,
                ))
            };
            // SAFETY: evaluation is synchronous and complete, while the
            // retained matrix remains owned by the prover data.
            if let Some(main) = main {
                unsafe { main.release_trace() };
            }
            result
        };
        let mut results = (0..inputs.len()).map(|_| None).collect::<Vec<_>>();
        for (index, output_bytes, direct_temporary_bytes, graph_temporary_bytes, _) in lookup_jobs {
            let job_started = std::time::Instant::now();
            let input = &inputs[index];
            let graph_path = self
                .pcs
                .mmcs
                .is_matrix_cuda_resident(input.stage_1.0, input.stage_1.1)
                && graph_temporary_bytes.is_some()
                && input.preprocessed.is_none_or(|(data, matrix)| {
                    self.pcs.mmcs.is_matrix_cuda_resident(data, matrix)
                });
            let temporary_bytes = if graph_path {
                graph_temporary_bytes.unwrap()
            } else {
                direct_temporary_bytes
            };
            let target = output_bytes
                .saturating_add(temporary_bytes)
                .saturating_add(total_device_bytes / 64);
            if target > total_device_bytes {
                return None;
            }
            let mut free_bytes = self.pcs.mmcs.ensure_device_headroom(
                input.stage_1.0,
                target,
                Some(input.stage_1.1),
            );
            if free_bytes < target {
                // This circuit alone does not fit beside its resident trace.
                // Spill it as a last resort and use the direct lookup-values
                // path, which remains protocol-identical.
                free_bytes = self
                    .pcs
                    .mmcs
                    .ensure_device_headroom(input.stage_1.0, target, None);
            }
            if free_bytes < target {
                return None;
            }
            if crate::cuda::memory_diagnostics_enabled() {
                eprintln!(
                    "[multi-stark/cuda] lookup job {index}: graph={graph_path} output={output_bytes} temporary={temporary_bytes} target={target} free={free_bytes}"
                );
            }
            let (_, num_lookups, _, _, arg_offsets) = input.lookup_values.cuda_parts();
            let cooperative = !graph_path
                && output_bytes >= (8usize << 30)
                && num_lookups >= 64
                && arg_offsets.last().copied().unwrap_or(0) >= 256;
            results[index] = Some(evaluate(&inputs[index], cooperative)?);
            if crate::cuda::memory_diagnostics_enabled() {
                eprintln!(
                    "[multi-stark/cuda] lookup job {index} complete: {:.3}s",
                    job_started.elapsed().as_secs_f64()
                );
            }
        }
        let tree_headroom = max_lde_height
            .saturating_mul(96)
            .saturating_add(total_device_bytes / 64);
        if let Some(input) = inputs.first() {
            let free_bytes =
                self.pcs
                    .mmcs
                    .ensure_device_headroom(input.stage_1.0, tree_headroom, None);
            if free_bytes < tree_headroom {
                return None;
            }
        }
        let mut ldes = Vec::with_capacity(results.len());
        let mut intermediates = Vec::with_capacity(inputs.len());
        for result in results {
            let (lde, total) = result.expect("lookup job did not run");
            accumulator += ExtVal::from_basis_coefficients_slice(&total)?;
            intermediates.push(accumulator);
            ldes.push(lde);
        }
        let (commitment, data) = self.pcs.mmcs.commit_cuda_spillable(ldes);
        Some((commitment, data, intermediates))
    }
}

/// Parameters of the polynomial commitment: Reed-Solomon rate and Merkle
/// tree shape.
#[derive(Clone, Copy)]
pub struct CommitmentParameters {
    pub log_blowup: usize,
    /// Height of the Merkle cap (number of top layers included in the commitment).
    /// A cap height of 0 means only the root is committed.
    pub cap_height: usize,
}

/// Parameters controlling the FRI protocol.
///
/// These parameters determine the concrete security level. The FRI soundness
/// error is approximately `ρ^num_queries` (conjectured; `√ρ^num_queries`
/// proven) where `ρ = 2^(-log_blowup)` (set in [`CommitmentParameters`]).
/// See the verifier module docs for the full soundness argument.
#[derive(Clone, Copy)]
pub struct FriParameters {
    /// Log2 of the degree of the final polynomial (0 means a constant).
    pub log_final_poly_len: usize,
    /// Maximum folding arity per FRI round (log2). A value of 1 means binary folding.
    pub max_log_arity: usize,
    /// Number of query repetitions for soundness amplification.
    pub num_queries: usize,
    /// Number of bits for the PoW phase before sampling _each_ batching challenge.
    pub commit_proof_of_work_bits: usize,
    /// Number of bits for the PoW phase before sampling the queries.
    pub query_proof_of_work_bits: usize,
}

pub(crate) type Blake3CompressionFunction = CompressionFunctionFromHasher<Blake3, 2, 32>;

#[cfg(feature = "cuda")]
impl CudaPcsDft<Val> for CudaDft {
    fn prepare_coset_lde_constants(&self, height: usize, added_bits: usize, shift: Val) {
        self.prepare_coset_lde_constants(height, added_bits, shift);
    }

    fn coset_lde_batch_resident(
        &self,
        matrix: &RowMajorMatrix<Val>,
        added_bits: usize,
        shift: Val,
    ) -> crate::cuda::CudaLde {
        self.coset_lde_batch_resident(matrix, added_bits, shift)
    }
}

type Dft = Radix2DitParallel<Val>;
#[cfg(not(feature = "cuda"))]
type PcsDft = Dft;
#[cfg(feature = "cuda")]
type PcsDft = CudaDft;

fn new_mmcs(cap_height: usize) -> Mmcs {
    let byte_hash = Blake3;
    let field_hash = SerializingHasher::new(byte_hash);
    let compress = Blake3CompressionFunction::new(byte_hash);
    let cpu = CpuMmcs::new(field_hash, compress, cap_height);
    #[cfg(not(feature = "cuda"))]
    return cpu;
    #[cfg(feature = "cuda")]
    crate::cuda::mmcs::CudaMmcs::new(cpu)
}

fn new_pcs(
    commitment_parameters: CommitmentParameters,
    fri_parameters: FriParameters,
) -> (Pcs, Dft) {
    let val_mmcs = new_mmcs(commitment_parameters.cap_height);
    let mmcs = ExtensionMmcs::new(val_mmcs.clone());
    let inner_parameters = InnerFriParameters {
        log_blowup: commitment_parameters.log_blowup,
        log_final_poly_len: fri_parameters.log_final_poly_len,
        max_log_arity: fri_parameters.max_log_arity,
        num_queries: fri_parameters.num_queries,
        commit_proof_of_work_bits: fri_parameters.commit_proof_of_work_bits,
        query_proof_of_work_bits: fri_parameters.query_proof_of_work_bits,
        mmcs,
    };
    let dft = Dft::default();
    let pcs = Pcs::new(PcsDft::default(), val_mmcs, inner_parameters);
    (pcs, dft)
}

#[cfg(test)]
mod pcs_ref_gen {
    use super::*;
    use p3_commit::Mmcs as _;
    use p3_field::{
        BasedVectorSpace, PrimeCharacteristicRing, PrimeField64, batch_multiplicative_inverse,
    };
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

    #[cfg(feature = "cuda")]
    fn test_fri_parameters(max_log_arity: usize) -> FriParameters {
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity,
            num_queries: 1,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[should_panic(expected = "cap_height = 0")]
    fn cuda_rejects_nonzero_merkle_caps() {
        let _ = GoldilocksBlake3Config::new(
            CommitmentParameters {
                log_blowup: 1,
                cap_height: 1,
            },
            test_fri_parameters(1),
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[should_panic(expected = "binary FRI folds")]
    fn cuda_rejects_multi_bit_fri_folds() {
        let _ = GoldilocksBlake3Config::new(
            CommitmentParameters {
                log_blowup: 1,
                cap_height: 0,
            },
            test_fri_parameters(2),
        );
    }

    fn limbs(d: [u8; 32]) -> [u64; 4] {
        core::array::from_fn(|i| u64::from_le_bytes(d[i * 8..i * 8 + 8].try_into().unwrap()))
    }
    fn dig(xs: [u64; 4]) -> [u8; 32] {
        let mut o = [0u8; 32];
        for i in 0..4 {
            o[i * 8..i * 8 + 8].copy_from_slice(&xs[i].to_le_bytes());
        }
        o
    }

    /// Pins canonical Goldilocks base/extension arithmetic and Montgomery
    /// batch inversion at values chosen around the Solinas modulus. These are
    /// compact host-side oracles for the first CUDA field kernels.
    #[test]
    fn goldilocks_arithmetic_contract() {
        let canonical = |value: Val| value.as_canonical_u64();
        for (a, b, sum, difference, product) in [
            (
                1_311_768_467_463_790_320,
                1_147_797_409_030_816_545,
                2_459_565_876_494_606_865,
                163_971_058_432_973_775,
                14_965_091_924_900_821_934,
            ),
            (
                18_446_744_069_414_584_319,
                4_294_967_296,
                4_294_967_294,
                18_446_744_065_119_617_023,
                18_446_744_060_824_649_729,
            ),
            (
                4_294_967_295,
                4_294_967_297,
                8_589_934_592,
                18_446_744_069_414_584_319,
                4_294_967_294,
            ),
        ] {
            let a = Val::from_u64(a);
            let b = Val::from_u64(b);
            assert_eq!(canonical(a + b), sum);
            assert_eq!(canonical(a - b), difference);
            assert_eq!(canonical(a * b), product);
        }

        let values = [2, 3, 4_294_967_296, 1_311_768_467_463_790_320].map(Val::from_u64);
        let inverses = batch_multiplicative_inverse(&values);
        assert_eq!(
            inverses.into_iter().map(canonical).collect::<Vec<_>>(),
            vec![
                9_223_372_034_707_292_161,
                12_297_829_379_609_722_881,
                18_446_744_065_119_617_026,
                14_736_413_637_906_284_881,
            ]
        );

        let left = ExtVal::from_basis_coefficients_fn(|i| [Val::from_u8(3), Val::from_u8(5)][i]);
        let right = ExtVal::from_basis_coefficients_fn(|i| [Val::from_u8(11), Val::from_u8(13)][i]);
        let product = left * right;
        assert_eq!(
            product
                .as_basis_coefficients_slice()
                .iter()
                .copied()
                .map(canonical)
                .collect::<Vec<_>>(),
            vec![488, 94]
        );
    }

    /// Pins the Blake3 and Merkle values shared with `Ix/MultiStark/Tests.lean`
    /// (`pcs_hash_test`, `pcs_merkle_test`). A future GPU implementation must
    /// match these byte-for-byte, including field serialization and tree
    /// layout.
    #[test]
    fn blake3_and_merkle_contract() {
        let f = Val::from_u32;
        let fh = SerializingHasher::new(Blake3);
        for (n, expected) in [
            (
                3u32,
                [
                    4163513704854067712,
                    9384471110237386207,
                    13671380075168847140,
                    1533933974187331481,
                ],
            ),
            (
                17,
                [
                    8431665677194841246,
                    4495111673672851816,
                    7709594803249897978,
                    12683511314940902790,
                ],
            ),
            (
                22,
                [
                    14017803411919507972,
                    9236340131056405306,
                    11356520758956579629,
                    2008168271701183309,
                ],
            ),
            (
                20,
                [
                    8822819174011220231,
                    9835070768970864367,
                    9646176123001837413,
                    1210344881395534089,
                ],
            ),
        ] {
            let row: Vec<Val> = (1..=n).map(f).collect();
            assert_eq!(limbs(fh.hash_iter(row)), expected, "leaf width {n}");
        }
        let comp = Blake3CompressionFunction::new(Blake3);
        assert_eq!(
            limbs(comp.compress([dig([1, 2, 3, 4]), dig([5, 6, 7, 8])])),
            [
                16432952784711837466,
                12565756115161032165,
                6915939387221618258,
                11123773279136987111,
            ],
        );

        // Merkle tree: matrices of heights 8/4/2 and widths 2/3/1, opened at index 5.
        let mut m0 = vec![f(0); 16];
        m0[10] = f(11);
        m0[11] = f(12); // row 5 = [11, 12]
        let mut m1 = vec![f(0); 12];
        m1[6] = f(107);
        m1[7] = f(108);
        m1[8] = f(109); // row 2 = [107, 108, 109]
        let mut m2 = vec![f(0); 2];
        m2[1] = f(202); // row 1 = [202]
        let mmcs = new_mmcs(0);
        let (commit, pd) = mmcs.commit(vec![
            RowMajorMatrix::new(m0.clone(), 2),
            RowMajorMatrix::new(m1.clone(), 3),
            RowMajorMatrix::new(m2.clone(), 1),
        ]);
        let bo = mmcs.open_batch(5, &pd);
        assert_eq!(
            bo.opened_values,
            vec![
                vec![f(11), f(12)],
                vec![f(107), f(108), f(109)],
                vec![f(202)]
            ]
        );
        assert_eq!(
            bo.opening_proof
                .iter()
                .copied()
                .map(limbs)
                .collect::<Vec<_>>(),
            vec![
                [
                    824163284354560741,
                    10184227291309369989,
                    7314170388788081421,
                    2210258918235055872,
                ],
                [
                    16321412416894375658,
                    13817763133082311448,
                    4555362725758189505,
                    13946835461337436585,
                ],
                [
                    11035117895010660519,
                    10627114985553641692,
                    18209541265052796223,
                    11062544859664569990,
                ],
            ]
        );
        assert_eq!(
            commit.roots(),
            &[[
                45, 230, 248, 40, 61, 21, 136, 65, 180, 102, 50, 238, 76, 222, 102, 39, 123, 114,
                106, 220, 182, 223, 92, 68, 228, 55, 152, 7, 80, 209, 237, 16,
            ]]
        );

        let mmcs = new_mmcs(2);
        let (commit, pd) = mmcs.commit(vec![
            RowMajorMatrix::new(m0, 2),
            RowMajorMatrix::new(m1, 3),
            RowMajorMatrix::new(m2, 1),
        ]);
        let bo = mmcs.open_batch(5, &pd);
        assert_eq!(
            commit
                .roots()
                .iter()
                .copied()
                .map(limbs)
                .collect::<Vec<_>>(),
            vec![
                [
                    16321412416894375658,
                    13817763133082311448,
                    4555362725758189505,
                    13946835461337436585,
                ],
                [
                    16321412416894375658,
                    13817763133082311448,
                    4555362725758189505,
                    13946835461337436585,
                ],
                [
                    2755952710066137292,
                    16563663342057344133,
                    5946896676730904047,
                    10390238708790769607,
                ],
                [
                    16321412416894375658,
                    13817763133082311448,
                    4555362725758189505,
                    13946835461337436585,
                ],
            ]
        );
        assert_eq!(
            bo.opening_proof
                .iter()
                .copied()
                .map(limbs)
                .collect::<Vec<_>>(),
            vec![[
                824163284354560741,
                10184227291309369989,
                7314170388788081421,
                2210258918235055872,
            ]]
        );
    }

    /// Pins the Blake3-challenger reference values used by `sample_bits_test`
    /// and `pcs_challenger4_test` in Ix.
    #[test]
    fn blake3_challenger_contract() {
        use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
        use p3_field::{BasedVectorSpace, PrimeField64};
        let g = Val::from_u64;
        fn el(e: ExtVal) -> (u64, u64) {
            let s: &[Val] = e.as_basis_coefficients_slice();
            (s[0].as_canonical_u64(), s[1].as_canonical_u64())
        }
        // sample_bits_test: observe 0x0102030405060708, sample_bits(20).
        let mut ch = Challenger::from_hasher(vec![], Blake3);
        ch.observe(g(0x0102030405060708));
        assert_eq!(CanSampleBits::<usize>::sample_bits(&mut ch, 20), 1019203);
        // pcs_challenger4_test: the α_pcs/α_fri/β/index continuation.
        let mut ch = Challenger::from_hasher(vec![], Blake3);
        ch.observe(g(0x0102030405060708));
        ch.observe(g(0x1122334455667788));
        let apcs: ExtVal = ch.sample_algebra_element();
        let afri: ExtVal = ch.sample_algebra_element();
        assert_eq!(el(apcs), (17795849114622667264, 4116843485681689527));
        assert_eq!(el(afri), (11768399386651893439, 10948618071942561750));
        ch.observe(g(0x00000000deadbeef));
        let beta: ExtVal = ch.sample_algebra_element();
        assert_eq!(el(beta), (12096272534537655203, 11431251745744402868));
        ch.observe(g(0x0a0b0c0d01020304));
        ch.observe(g(0x0000000000000002));
        assert_eq!(CanSampleBits::<usize>::sample_bits(&mut ch, 20), 458922);
    }
}
