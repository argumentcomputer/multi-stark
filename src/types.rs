//! The reference STARK configuration: Goldilocks field with a degree-2
//! binomial extension, Blake3 hashing, and a FRI-based PCS.
//!
//! The generic protocol lives in [`crate::config`], [`crate::system`],
//! [`crate::prover`] and [`crate::verifier`]; this module only provides a
//! concrete, batteries-included instantiation.

use crate::config::{ProofConfig, ProtocolVersion, TranscriptMode};
use p3_blake3::Blake3;
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger, HashChallenger,
    SerializingChallenger64,
};
use p3_commit::{ExtensionMmcs, Pcs as PcsTrait};
use p3_dft::Radix2DitParallel;
#[cfg(feature = "cuda")]
use p3_field::BasedVectorSpace;
use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField,
    extension::{BinomialExtensionField, CubicTrinomialExtensionField},
};
use p3_fri::FriParameters as InnerFriParameters;
use p3_goldilocks::Goldilocks;
#[cfg(not(feature = "cuda"))]
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
#[cfg(test)]
use p3_symmetric::CompressionFunctionFromHasher;
#[cfg(test)]
use p3_symmetric::SerializingHasher;

#[cfg(feature = "cuda")]
use crate::cuda::CudaDft;
#[cfg(feature = "cuda")]
use crate::cuda::pcs::CudaPcsDft;
pub use crate::p3_adapter::challenger::DomainChallenger;
pub use crate::p3_adapter::domain::{CosetDomain, DomainMappedPcs};
pub use crate::p3_adapter::domain_blake3::{Blake3Domain, DomainBlake3};
pub use crate::p3_adapter::fri_batching::FriBatching;
use crate::traits::ExtensionModulus;

pub type Val = Goldilocks;
pub type PackedVal = <Val as Field>::Packing;
pub type ExtVal = BinomialExtensionField<Val, 2>;
pub type ExtVal2 = BinomialExtensionField<Val, 2>;
pub type ExtVal3 = CubicTrinomialExtensionField<Val>;
pub type PackedExtVal = <ExtVal as ExtensionField<Val>>::ExtensionPacking;
pub type PackedExtVal3 = <ExtVal3 as ExtensionField<Val>>::ExtensionPacking;
pub type Challenger =
    DeterministicPow<SerializingChallenger64<Val, HashChallenger<u8, Blake3, 32>>>;

/// Forwarding challenger wrapper that makes proof-of-work grinding
/// deterministic.
///
/// p3's `SerializingChallenger64::grind` races `find_any` over every
/// candidate witness, so the returned witness is whichever a rayon thread
/// reports first: a thread-race-dependent value that lands in the proof
/// bytes — and, since it is observed before the FRI query indices are
/// sampled, changes everything after it — making proofs nondeterministic
/// run to run. At 0 bits EVERY candidate passes, and the canonical ZERO is
/// returned without a search. At positive bit counts the search returns the
/// SMALLEST passing witness: candidates are checked in parallel one window
/// at a time, and a window's minimum is taken only once the whole window
/// has been checked, so no thread can resolve a candidate ahead of a lower
/// one. Transcript semantics are untouched — the witness is checked on the
/// wrapped challenger exactly as the inner grind would.
#[derive(Clone, Debug)]
pub struct DeterministicPow<C>(pub C);

/// Candidates checked per parallel window of the deterministic grind for a
/// `bits`-bit target: about a sixteenth of the `2^bits` checks such a grind
/// expects, so a low witness costs little extra work, within bounds that
/// keep a window worth splitting across threads and bounded in memory.
fn grind_window(bits: usize) -> u64 {
    (1u64 << bits.min(40)).div_ceil(16).clamp(64, 1 << 16)
}

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

impl<C: GrindingChallenger> GrindingChallenger for DeterministicPow<C>
where
    C::Witness: PrimeField64,
{
    type Witness = C::Witness;

    fn grind(&mut self, bits: usize) -> Self::Witness {
        if bits == 0 {
            return Self::Witness::ZERO;
        }
        let order = Self::Witness::ORDER_U64;
        let window = grind_window(bits);
        let mut base = 0u64;
        let witness = loop {
            assert!(base < order, "no proof-of-work witness in the field");
            let end = base.saturating_add(window).min(order);
            let found = (base..end)
                .into_par_iter()
                .filter(|&i| {
                    self.0
                        .clone()
                        .check_witness(bits, Self::Witness::from_u64(i))
                })
                .min();
            if let Some(i) = found {
                break Self::Witness::from_u64(i);
            }
            base = end;
        };
        assert!(
            self.0.check_witness(bits, witness),
            "grind witness fails its own check"
        );
        witness
    }
}

#[cfg(test)]
mod grind_tests {
    use super::*;
    use p3_challenger::GrindingChallenger;

    fn challenger(seed: u8) -> Challenger {
        Challenger::from_hasher(vec![seed; 4], Blake3)
    }

    #[test]
    fn grind_returns_the_smallest_witness() {
        for seed in 0..4u8 {
            let bits = 10;
            let witness = challenger(seed).grind(bits);
            // No smaller candidate passes, and the chosen one does, on a
            // fresh challenger in the same state.
            let w = witness.as_canonical_u64();
            for i in 0..w {
                assert!(!challenger(seed).check_witness(bits, Val::from_u64(i)));
            }
            assert!(challenger(seed).check_witness(bits, witness));
            assert_eq!(challenger(seed).grind(bits), witness, "repeatable");
        }
    }

    #[test]
    fn grind_leaves_the_challenger_where_the_verifier_lands() {
        // The prover grinds; the verifier checks the witness. Both must
        // continue from the same state, so their next samples agree.
        let bits = 8;
        let mut prover = challenger(7);
        let witness = prover.grind(bits);
        let mut verifier = challenger(7);
        assert!(verifier.check_witness(bits, witness));
        for _ in 0..4 {
            let p: Val = prover.sample();
            let v: Val = verifier.sample();
            assert_eq!(p, v);
        }
    }

    #[test]
    fn grind_crosses_windows() {
        // At 16 bits the window holds 4096 candidates; seed 0's smallest
        // witness lies well past the first window, so the step is taken.
        let bits = 16;
        let witness = challenger(0).grind(bits);
        assert!(witness.as_canonical_u64() >= grind_window(bits));
        for i in 0..witness.as_canonical_u64() {
            assert!(!challenger(0).check_witness(bits, Val::from_u64(i)));
        }
    }
}
#[cfg(not(feature = "cuda"))]
pub type Mmcs = crate::p3_adapter::spill::SpillableMmcs;
#[cfg(feature = "cuda")]
pub type Mmcs = crate::cuda::mmcs::CudaMmcs;
pub type ExtMmcs = ExtensionMmcs<Val, ExtVal, Mmcs>;
#[cfg(not(feature = "cuda"))]
pub type InnerPcs = crate::p3_adapter::spill::SpillableTwoAdicFriPcs<PcsDft, ExtMmcs>;
#[cfg(feature = "cuda")]
pub type InnerPcs = crate::cuda::pcs::CudaTwoAdicFriPcs<Val, PcsDft, Mmcs, ExtMmcs>;

pub type Pcs =
    crate::p3_adapter::pcs::FriPcs<Val, ExtVal, Challenger, Dft, Mmcs, ExtMmcs, InnerPcs>;

pub type Commitment = <InnerPcs as PcsTrait<ExtVal, Challenger>>::Commitment;
pub type Domain = <InnerPcs as PcsTrait<ExtVal, Challenger>>::Domain;
pub type ProverData = <InnerPcs as PcsTrait<ExtVal, Challenger>>::ProverData;
pub type EvaluationsOnDomain<'a> =
    <InnerPcs as PcsTrait<ExtVal, Challenger>>::EvaluationsOnDomain<'a>;
pub type PcsError = <InnerPcs as PcsTrait<ExtVal, Challenger>>::Error;
pub type PcsProof = <InnerPcs as PcsTrait<ExtVal, Challenger>>::Proof;

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
fn cuda_ext_params<EF: crate::traits::ExtensionOf<Val>>() -> (usize, Val) {
    let ext_degree = EF::D;
    assert!(
        ext_degree == 2 || ext_degree == 3,
        "CUDA backend supports Goldilocks extension degrees 2 and 3, got {ext_degree}"
    );
    let ext_w = match EF::MODULUS {
        ExtensionModulus::Binomial { w } => w,
        ExtensionModulus::CubicTrinomial => Val::ZERO,
        ExtensionModulus::Base => panic!("CUDA backend requires an extension field"),
    };
    (ext_degree, ext_w)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_accelerated_quotient_values<EF>(
    circuit: &crate::system::Circuit<Val>,
    lookup_publics: &[Val],
    trace_domain: Domain,
    quotient_domain: Domain,
    preprocessed: Option<(&ProverData, usize)>,
    stage_1: (&ProverData, usize),
    stage_2: (&ProverData, usize),
    alpha: EF,
    constraint_count: usize,
) -> Option<Vec<EF>>
where
    EF: TwoAdicField + ExtensionField<Val> + crate::traits::ExtensionOf<Val>,
{
    if !crate::cuda::quotient_graph_supported(&circuit.graph) {
        return None;
    }
    let main = stage_1.0.resident(stage_1.1)?;
    let s2 = stage_2.0.resident(stage_2.1)?;
    let prep = match preprocessed {
        Some((data, index)) => Some(data.resident(index)?),
        None => None,
    };
    let (ext_degree, ext_w) = cuda_ext_params::<EF>();
    let qsize = quotient_domain.size();
    let selectors = cuda_coset_selectors(trace_domain, quotient_domain);
    let mut powers = Vec::with_capacity(constraint_count);
    let mut power = <EF as PrimeCharacteristicRing>::ONE;
    for _ in 0..constraint_count {
        powers.push(power);
        power *= alpha;
    }
    powers.reverse();
    let mut alpha_flat = Vec::with_capacity(ext_degree * constraint_count);
    for coordinate in 0..ext_degree {
        alpha_flat.extend(
            powers
                .iter()
                .map(|x| <EF as BasedVectorSpace<Val>>::as_basis_coefficients_slice(x)[coordinate]),
        );
    }
    let n = Val::from_usize(trace_domain.size());
    let g = Val::two_adic_generator(trace_domain.size().ilog2() as usize);
    let norm = (n * g).inverse();
    let delta: Vec<Val> = (0..ext_degree)
        .map(|c| (lookup_publics[3 * ext_degree + c] - lookup_publics[2 * ext_degree + c]) * norm)
        .collect();
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
        ext_w,
        ext_degree,
        qsize,
        next_step,
        circuit.lookup_group_size,
    );
    Some(
        flat.values
            .chunks_exact(ext_degree)
            .map(|coords| {
                EF::from_basis_coefficients_slice(coords)
                    .expect("CUDA quotient coordinate count mismatch")
            })
            .collect(),
    )
}

#[cfg(feature = "cuda")]
struct CudaQuotientJob<'a> {
    circuit: &'a crate::system::Circuit<Val>,
    lookup_publics: &'a [Val],
    trace_domain: Domain,
    quotient_domain: Domain,
    preprocessed: Option<(&'a ProverData, usize)>,
    stage_1: (&'a ProverData, usize),
    stage_2: (&'a ProverData, usize),
    constraint_count: usize,
}

#[cfg(feature = "cuda")]
fn cuda_accelerated_quotient_commit<EF>(
    dft: &CudaDft,
    mmcs: &Mmcs,
    log_blowup: usize,
    inputs: &[CudaQuotientJob<'_>],
    alpha: EF,
) -> Option<(Commitment, ProverData)>
where
    EF: TwoAdicField + ExtensionField<Val> + crate::traits::ExtensionOf<Val>,
{
    use crate::cuda::mmcs::CudaCommitMmcs;

    if inputs
        .iter()
        .any(|input| !crate::cuda::quotient_graph_supported(&input.circuit.graph))
    {
        return None;
    }
    let (ext_degree, ext_w) = cuda_ext_params::<EF>();
    let (_, total_device_bytes) = crate::cuda::device_memory_info(mmcs.cuda_device_id());
    let matrix_width = |data: &ProverData, index: usize| {
        let dimensions = mmcs.matrix_dimensions(data);
        dimensions
            .get(index)
            .expect("matrix index out of bounds")
            .width
    };
    let staging_bytes = |input: &CudaQuotientJob<'_>| {
        let mut host_width = 0usize;
        for (data, index) in [Some(input.stage_1), Some(input.stage_2), input.preprocessed]
            .into_iter()
            .flatten()
        {
            if !mmcs.is_matrix_cuda_resident(data, index) {
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
                .checked_shl(u32::try_from(log_blowup).expect("LDE blowup exceeds u32"))
                .unwrap_or(usize::MAX);
            let (output_bytes, kernel_workspace) = crate::cuda::quotient_lde_memory_upper_bound(
                &input.circuit.graph,
                input.lookup_publics.len(),
                input.constraint_count,
                quotient_size,
                quotient_degree,
                log_blowup,
                ext_degree,
            );
            let current_staging = staging_bytes(input);
            let constant_bytes = quotient_degree.saturating_mul(size_of::<Val>());
            let quotient_plan = dft.forward_plan(quotient_size, ext_degree);
            let lde_plan = dft.forward_plan(lde_height, ext_degree * quotient_degree);
            let kernel_workspace = kernel_workspace
                .saturating_add(quotient_plan.scratch_bytes().max(lde_plan.scratch_bytes()));
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
            .all(|(data, matrix)| mmcs.is_matrix_source_ready(data, matrix));
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
    for (index, output_bytes, kernel_workspace, constant_bytes, _, lde_height) in quotient_jobs {
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
        mmcs.ensure_device_headroom(input.stage_1.0, target, Some(input.stage_1.1), "quotient");
        target = required();
        mmcs.ensure_device_headroom(input.stage_2.0, target, Some(input.stage_2.1), "quotient");
        if let Some((data, matrix)) = input.preprocessed {
            target = required();
            mmcs.ensure_device_headroom(data, target, Some(matrix), "quotient");
        }
        target = required();
        let free_bytes = crate::cuda::device_memory_info(mmcs.cuda_device_id()).0;
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
        let lde = {
            let quotient_size = input.quotient_domain.size();
            let quotient_degree = input.circuit.quotient_degree();
            let selectors = cuda_coset_selectors(input.trace_domain, input.quotient_domain);

            let mut powers = Vec::with_capacity(input.constraint_count);
            let mut power = <EF as PrimeCharacteristicRing>::ONE;
            for _ in 0..input.constraint_count {
                powers.push(power);
                power *= alpha;
            }
            powers.reverse();
            let mut alpha_flat = Vec::with_capacity(ext_degree * input.constraint_count);
            for coordinate in 0..ext_degree {
                alpha_flat.extend(powers.iter().map(|value| {
                    <EF as BasedVectorSpace<Val>>::as_basis_coefficients_slice(value)[coordinate]
                }));
            }
            let trace_size = input.trace_domain.size();
            let n = Val::from_usize(trace_size);
            let generator = Val::two_adic_generator(trace_size.ilog2() as usize);
            let normalization = (n * generator).inverse();
            let delta: Vec<Val> = (0..ext_degree)
                .map(|c| {
                    (input.lookup_publics[3 * ext_degree + c]
                        - input.lookup_publics[2 * ext_degree + c])
                        * normalization
                })
                .collect();
            let next_step = quotient_size / trace_size;
            let evaluate =
                |main: crate::cuda::mmcs::CudaMatrixSource<'_>,
                 stage2: crate::cuda::mmcs::CudaMatrixSource<'_>,
                 preprocessed: Option<crate::cuda::mmcs::CudaMatrixSource<'_>>| {
                    crate::cuda::quotient_lde_mixed(
                        dft,
                        &input.circuit.graph,
                        preprocessed,
                        main,
                        stage2,
                        input.lookup_publics,
                        selectors,
                        &alpha_flat,
                        &delta,
                        ext_w,
                        ext_degree,
                        quotient_size,
                        next_step,
                        input.circuit.lookup_group_size,
                        quotient_degree,
                        log_blowup,
                    )
                };
            mmcs.with_matrix_source(input.stage_1.0, input.stage_1.1, |main| {
                mmcs.with_matrix_source(input.stage_2.0, input.stage_2.1, |stage2| {
                    match input.preprocessed {
                        Some((data, index)) => {
                            mmcs.with_matrix_source(data, index, |preprocessed| {
                                evaluate(main, stage2, Some(preprocessed))
                            })
                        }
                        None => evaluate(main, stage2, None),
                    }
                })
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
        mmcs.ensure_device_headroom(input.stage_1.0, tree_headroom, None, "quotient_tree");
        mmcs.ensure_device_headroom(input.stage_2.0, tree_headroom, None, "quotient_tree");
    }
    if let Some((data, _)) = inputs.iter().find_map(|input| input.preprocessed) {
        mmcs.ensure_device_headroom(data, tree_headroom, None, "quotient_tree");
    }
    if crate::cuda::device_memory_info(mmcs.cuda_device_id()).0 < tree_headroom {
        return None;
    }
    Some(
        mmcs.commit_cuda_spillable(
            ldes.into_iter()
                .map(|lde| lde.expect("quotient job did not run"))
                .collect(),
        ),
    )
}

#[cfg(feature = "cuda")]
struct CudaLookupJob<'a> {
    circuit: &'a crate::system::Circuit<Val>,
    lookup_values: &'a crate::lookup::LookupValues<Val>,
    preprocessed: Option<(&'a ProverData, usize)>,
    stage_1: (&'a ProverData, usize),
}

#[cfg(feature = "cuda")]
fn cuda_accelerated_lookup_commit<EF>(
    dft: &CudaDft,
    mmcs: &Mmcs,
    log_blowup: usize,
    inputs: &[CudaLookupJob<'_>],
    lookup_challenge: EF,
    fingerprint_challenge: EF,
    mut accumulator: EF,
) -> Option<(Commitment, ProverData, Vec<EF>)>
where
    EF: TwoAdicField + ExtensionField<Val> + crate::traits::ExtensionOf<Val>,
{
    use crate::cuda::mmcs::CudaCommitMmcs;

    let (ext_degree, ext_w) = cuda_ext_params::<EF>();
    let mut lookup_jobs = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let (height, num_lookups, _, _, arg_offsets) = input.lookup_values.cuda_parts();
            let group_size = input.circuit.lookup_group_size.max(1);
            let groups = num_lookups.div_ceil(group_size).max(1);
            let extended_height = height
                .checked_shl(u32::try_from(log_blowup).expect("LDE blowup exceeds u32"))
                .unwrap_or(usize::MAX);
            let output_bytes = extended_height
                .saturating_mul(ext_degree)
                .saturating_mul(groups)
                .saturating_mul(size_of::<Val>());
            let direct_temporary_bytes = if num_lookups == 0 {
                0
            } else {
                let args_width = *arg_offsets.last().expect("lookup offsets are empty");
                let chunk_rows = height.min(1 << 16);
                let messages = chunk_rows.saturating_mul(num_lookups);
                messages
                    // Multiplicity, adjugate (ext_degree words), norm, and inverse-norm arrays.
                    .saturating_mul((ext_degree + 3) * size_of::<Val>())
                    .saturating_add(
                        chunk_rows
                            .saturating_mul(args_width)
                            .saturating_mul(size_of::<Val>()),
                    )
                    // Extension-field deltas before the exclusive scan.
                    .saturating_add(
                        height
                            .saturating_mul(groups)
                            .saturating_mul(ext_degree * size_of::<Val>()),
                    )
                    .saturating_add(arg_offsets.len().saturating_mul(size_of::<usize>()))
            };
            let main_width = mmcs
                .matrix_dimensions(input.stage_1.0)
                .get(input.stage_1.1)
                .expect("stage-1 matrix index out of bounds")
                .width;
            let preprocessed_width = input.preprocessed.map_or(0, |(data, matrix)| {
                mmcs.matrix_dimensions(data)
                    .get(matrix)
                    .expect("preprocessed matrix index out of bounds")
                    .width
            });
            let graph_memory = (num_lookups != 0)
                .then(|| {
                    crate::cuda::lookup_graph_lde_memory_upper_bound(
                        &input.circuit.graph,
                        height,
                        main_width,
                        preprocessed_width,
                        group_size,
                        log_blowup,
                        ext_degree,
                    )
                })
                .flatten();
            let transform_bytes = if num_lookups == 0 {
                0
            } else {
                let plan = dft.lde_plan(height, ext_degree * groups, log_blowup, Val::GENERATOR);
                plan.scratch_bytes().saturating_add(plan.constant_bytes())
            };
            (
                index,
                output_bytes,
                direct_temporary_bytes.saturating_add(transform_bytes),
                graph_memory.map(|(_, temporary)| temporary.saturating_add(transform_bytes)),
                extended_height,
            )
        })
        .collect::<Vec<_>>();
    lookup_jobs.sort_unstable_by_key(|&(index, output, direct, graph, _)| {
        let temporary = if inputs[index]
            .stage_1
            .0
            .has_resident_with_trace(inputs[index].stage_1.1)
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
    let (_, total_device_bytes) = crate::cuda::device_memory_info(mmcs.cuda_device_id());
    if crate::cuda::memory_diagnostics_enabled() {
        eprintln!(
            "[multi-stark/cuda] lookup jobs={} output={}",
            lookup_jobs.len(),
            accumulated_output,
        );
    }

    let beta = <EF as BasedVectorSpace<Val>>::as_basis_coefficients_slice(&lookup_challenge);
    let gamma = <EF as BasedVectorSpace<Val>>::as_basis_coefficients_slice(&fingerprint_challenge);
    let lookup_pool =
        std::sync::Arc::new(cuda_host_pool("lookup-rows", cuda_lookup_worker_count()));
    let evaluate = |input: &CudaLookupJob<'_>,
                    values: &crate::lookup::LookupValues<Val>,
                    graph_path: bool,
                    cooperative: bool| {
        let (height, num_lookups, multiplicities, args, arg_offsets) = values.cuda_parts();
        let group_size = input.circuit.lookup_group_size.max(1);
        let main = graph_path.then(|| {
            input
                .stage_1
                .0
                .resident_with_trace(input.stage_1.1)
                .expect("lookup graph admission requires a trace-backed resident LDE")
        });
        let result = if let Some(main) = main {
            let preprocessed = input.preprocessed.map(|(data, index)| {
                data.resident_with_host_trace(index).expect(
                    "lookup graph admission requires a trace-backed resident preprocessed LDE",
                )
            });
            crate::cuda::lookup_graph_lde_resident(
                dft,
                &input.circuit.graph,
                preprocessed,
                main,
                height,
                group_size,
                beta,
                gamma,
                ext_w,
                ext_degree,
                log_blowup,
            )
        } else if cooperative {
            let pool = std::sync::Arc::clone(&lookup_pool);
            Some(crate::cuda::lookup_lde_resident_partitioned(
                dft,
                multiplicities,
                args,
                arg_offsets,
                height,
                num_lookups,
                group_size,
                beta,
                gamma,
                ext_w,
                ext_degree,
                log_blowup,
                move |rows| {
                    pool.install(|| {
                        values.cuda_stage_2_deltas::<EF>(
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
                dft,
                multiplicities,
                args,
                arg_offsets,
                height,
                num_lookups,
                group_size,
                beta,
                gamma,
                ext_w,
                ext_degree,
                log_blowup,
            ))
        };
        // SAFETY: evaluation is synchronous and complete, while the retained
        // matrix remains owned by the prover data. A generator's device cache
        // stays: admission releases it under memory pressure, and later lookups
        // over the same record would otherwise rebuild it.
        if let Some(main) = main {
            unsafe { main.release_trace() };
        }
        result
    };
    let mut results = (0..inputs.len()).map(|_| None).collect::<Vec<_>>();
    for (index, output_bytes, direct_temporary_bytes, graph_temporary_bytes, _) in lookup_jobs {
        let job_started = std::time::Instant::now();
        let input = &inputs[index];
        // The graph kernel reads the stage-1 trace through `resident_with_trace`,
        // which also serves spilled LDEs that regenerate from a generator or a
        // retained host trace, and the preprocessed trace through
        // `resident_with_host_trace`. Admit it under the same predicates so the
        // budget matches the kernel that runs.
        let admits_graph = || {
            graph_temporary_bytes.is_some()
                && input.stage_1.0.has_resident_with_trace(input.stage_1.1)
                && input
                    .preprocessed
                    .is_none_or(|(data, matrix)| data.resident_with_host_trace(matrix).is_some())
        };
        // A graph lookup over a trace no longer on the device regenerates it,
        // which can re-upload generator state that admission released.
        let target_for = |graph_path: bool| {
            let temporary_bytes = if graph_path {
                graph_temporary_bytes
                    .unwrap()
                    .saturating_add(input.stage_1.0.regeneration_bytes(input.stage_1.1))
            } else {
                direct_temporary_bytes
            };
            (
                temporary_bytes,
                output_bytes
                    .saturating_add(temporary_bytes)
                    .saturating_add(total_device_bytes / 64),
            )
        };
        let mut graph_path = admits_graph();
        let (mut temporary_bytes, mut target) = target_for(graph_path);
        if target > total_device_bytes {
            return None;
        }
        let mut free_bytes =
            mmcs.ensure_device_headroom(input.stage_1.0, target, Some(input.stage_1.1), "lookup");
        if free_bytes < target {
            free_bytes = mmcs.ensure_device_headroom(input.stage_1.0, target, None, "lookup");
            graph_path = admits_graph();
            (temporary_bytes, target) = target_for(graph_path);
            if free_bytes < target {
                free_bytes = mmcs.ensure_device_headroom(input.stage_1.0, target, None, "lookup");
            }
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
        // The direct paths read lookup values on the host. A shape-only witness
        // gets them from the committed stage-one trace here. Circuits with zero
        // lookups build an all-zero LDE without reading any lookup values, so
        // skip trace rebuild when `num_lookups == 0`.
        let materialized = (!graph_path
            && num_lookups != 0
            && !input.lookup_values.is_materialized())
        .then(|| {
            let _g = tracing::info_span!("cuda/materialize_lookups", index).entered();
            let trace = mmcs.committed_trace(input.stage_1.0, input.stage_1.1, log_blowup);
            // A system built without its fixed matrices reads each one back
            // from the preprocessed commitment when it is needed.
            let fixed = (input.circuit.preprocessed.is_none()
                && input.circuit.graph.lookup_prefix_uses_preprocessed())
            .then_some(input.preprocessed)
            .flatten()
            .map(|(data, matrix)| mmcs.committed_trace(data, matrix, log_blowup));
            crate::system::compute_lookup_values_with_fixed(
                input.circuit,
                &trace,
                input.circuit.preprocessed.as_ref().or(fixed.as_ref()),
            )
        });
        let values = materialized.as_ref().unwrap_or(input.lookup_values);
        results[index] = Some(evaluate(input, values, graph_path, cooperative)?);
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
            mmcs.ensure_device_headroom(input.stage_1.0, tree_headroom, None, "lookup_tree");
        if free_bytes < tree_headroom {
            return None;
        }
    }
    let mut ldes = Vec::with_capacity(results.len());
    let mut intermediates = Vec::with_capacity(inputs.len());
    for result in results {
        let (lde, total) = result.expect("lookup job did not run");
        accumulator += EF::from_basis_coefficients_slice(&total)?;
        intermediates.push(accumulator);
        ldes.push(lde);
    }
    let (commitment, data) = mmcs.commit_cuda_spillable(ldes);
    Some((commitment, data, intermediates))
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

/// The reference [`ProofConfig`] implementation.
pub struct GoldilocksBlake3Config {
    /// The PCS used to commit polynomials and prove opening proofs.
    pcs: Pcs,
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
    cap_height: usize,
    fri_parameters: FriParameters,
}

impl GoldilocksBlake3Config {
    /// Public protocol-domain seed used by the reference challenger. Exposed
    /// for constrained transcript replay; this contains no prover secrets.
    pub fn challenger_seed(&self) -> &[u8] {
        &self.challenger_seed
    }

    pub fn cap_height(&self) -> usize {
        self.cap_height
    }

    pub fn fri_parameters(&self) -> FriParameters {
        self.fri_parameters
    }

    /// Enables disk-backed spilling of committed LDE matrices on the CPU
    /// prover, writing temporary spill files into `dir`.
    ///
    /// On Unix, spill files are unlinked immediately upon creation so their
    /// disk space is reclaimed automatically on drop, panic, or abnormal
    /// termination. Proofs produced with disk spilling enabled are
    /// byte-identical to proofs produced in memory.
    ///
    /// On CUDA builds this is a no-op (CUDA manages its own device offload).
    #[cfg_attr(feature = "cuda", allow(unused_mut))]
    pub fn with_disk_spill(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        #[cfg(not(feature = "cuda"))]
        self.pcs.inner_mut().set_spill_dir(Some(dir.into()));
        #[cfg(feature = "cuda")]
        let _ = dir.into();
        self
    }

    /// Disables disk spilling (forces in-memory LDE storage on CPU).
    #[cfg_attr(feature = "cuda", allow(unused_mut))]
    pub fn without_disk_spill(mut self) -> Self {
        #[cfg(not(feature = "cuda"))]
        self.pcs.inner_mut().set_spill_dir(None);
        self
    }

    /// Returns the configured disk-spill directory, if disk spilling is
    /// enabled on the CPU backend.
    pub fn spill_dir(&self) -> Option<&std::path::Path> {
        #[cfg(not(feature = "cuda"))]
        {
            self.pcs.spill_dir()
        }
        #[cfg(feature = "cuda")]
        {
            None
        }
    }

    pub fn new(commitment_parameters: CommitmentParameters, fri_parameters: FriParameters) -> Self {
        Self::with_device(commitment_parameters, fri_parameters, None)
    }

    /// A configuration whose prover lives on the given CUDA device (`None`:
    /// the device `MULTI_STARK_CUDA_DEVICE` names, or 0). Without the
    /// `cuda` feature only device 0 exists. Configurations on distinct
    /// devices prove concurrently in one process.
    ///
    /// On CPU builds, setting `MULTI_STARK_SPILL_DIR` to a non-empty path
    /// enables disk-backed LDE spilling by default; call
    /// [`Self::with_disk_spill`] or [`Self::without_disk_spill`] to override.
    pub fn with_device(
        commitment_parameters: CommitmentParameters,
        fri_parameters: FriParameters,
        device_id: Option<i32>,
    ) -> Self {
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
        #[cfg(feature = "cuda")]
        let (pcs, _dft) = new_pcs(
            commitment_parameters,
            fri_parameters,
            device_id.unwrap_or_else(crate::cuda::configured_device),
        );
        #[cfg(not(feature = "cuda"))]
        let (mut pcs, _dft) = {
            assert!(
                device_id.is_none_or(|device| device == 0),
                "no CUDA backend to place a prover on device {device_id:?}"
            );
            new_pcs(commitment_parameters, fri_parameters)
        };
        #[cfg(not(feature = "cuda"))]
        if let Some(dir) = std::env::var_os("MULTI_STARK_SPILL_DIR")
            && !dir.is_empty()
        {
            pcs.set_spill_dir(Some(std::path::PathBuf::from(dir)));
        }
        // Seed the challenger with a protocol tag for domain separation,
        // followed by every protocol parameter. Binding the parameters into
        // the seed means transcripts produced under different parameters
        // never collide (see the transcript contract on
        // [`ProofConfig::initialise_challenger`]).
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
            pcs: Pcs::new(pcs, commitment_parameters, fri_parameters),
            challenger_seed,
            max_log_degree,
            max_quotient_degree,
            log_blowup: commitment_parameters.log_blowup,
            cap_height: commitment_parameters.cap_height,
            fri_parameters,
        }
    }
}

impl ProofConfig for GoldilocksBlake3Config {
    type Pcs = Pcs;
    type Challenge = ExtVal;
    type Challenger = Challenger;

    fn pcs(&self) -> &Pcs {
        &self.pcs
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

    fn validate_commitment_cap(
        &self,
        commit: &crate::config::Com<Self>,
        max_log_lde_height: usize,
    ) -> bool {
        crate::p3_adapter::cap::expected_merkle_cap_roots(self.cap_height, max_log_lde_height)
            == Some(commit.num_roots())
    }

    /// The stage-one or preprocessed trace at `index` of `data`, on its trace
    /// domain of size `degree`. Delegates directly to
    /// [`crate::cuda::mmcs::CudaMmcs::committed_trace`] so only matrix `index`
    /// is read (or cloned from retained host traces) without calling
    /// `get_matrices`, which would spill all deferred GPU LDEs back to host
    /// RAM.
    #[cfg(feature = "cuda")]
    fn committed_trace(
        &self,
        data: &crate::config::PcsData<Self>,
        index: usize,
        degree: usize,
    ) -> Option<RowMajorMatrix<Val>> {
        use p3_matrix::Matrix;
        let trace = self.pcs.mmcs.committed_trace(data, index, self.log_blowup);
        assert_eq!(trace.height(), degree);
        Some(trace)
    }

    /// The stage-one or preprocessed trace at `index` of `data`, on its trace
    /// domain of size `degree`. Returns the retained 1× original trace without
    /// running any DFT when available; otherwise interpolates from the `gH`
    /// coset prefix of the committed LDE.
    #[cfg(not(feature = "cuda"))]
    fn committed_trace(
        &self,
        data: &crate::config::PcsData<Self>,
        index: usize,
        degree: usize,
    ) -> Option<RowMajorMatrix<Val>> {
        use crate::traits::{EvaluationDomain as _, Pcs as _};
        use p3_dft::TwoAdicSubgroupDft;
        if let Some(spilled) = data.spilled_original_trace(index) {
            let trace = spilled.read_all();
            assert_eq!(trace.height(), degree);
            return Some(trace);
        }
        if let Some(trace) = data.clone_in_memory_original_trace(index) {
            assert_eq!(trace.height(), degree);
            return Some(trace);
        }
        let coset = self
            .pcs
            .natural_domain_for_degree(degree)
            .create_disjoint_domain(degree);
        let evaluations = self
            .pcs
            .get_evaluations_on_domain(data, index, coset)
            .to_row_major_matrix();
        let dft = Radix2DitParallel::<Val>::default();
        let coefficients = dft.coset_idft_batch(evaluations, coset.first_point());
        Some(dft.dft_batch(coefficients).to_row_major_matrix())
    }
    #[cfg(feature = "cuda")]
    fn commit_main(
        &self,
        evaluations: Vec<(
            crate::config::Domain<Self>,
            crate::witness::TraceSource<Val>,
        )>,
    ) -> (crate::config::Com<Self>, crate::config::PcsData<Self>) {
        crate::cuda::witness::commit::<ExtVal, Challenger>(&self.pcs, evaluations)
    }

    #[cfg(not(feature = "cuda"))]
    fn commit_preprocessed(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (crate::config::Domain<Self>, RowMajorMatrix<Val>, bool),
        >,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        self.pcs.commit_streaming_with_originals(evaluations)
    }

    #[cfg(not(feature = "cuda"))]
    fn commit_main_streaming(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (
                crate::config::Domain<Self>,
                crate::witness::TraceSource<Val>,
                bool,
            ),
        >,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        let mut materialized =
            evaluations.map(|(domain, trace, retain_1x)| (domain, trace.materialize(), retain_1x));
        self.pcs.commit_streaming_with_originals(&mut materialized)
    }

    #[cfg(not(feature = "cuda"))]
    fn accelerated_lookup_commit(
        &self,
        inputs: &[crate::config::LookupCommitInput<'_, Self>],
        lookup_challenge: ExtVal,
        fingerprint_challenge: ExtVal,
        mut accumulator: ExtVal,
    ) -> Option<crate::config::AcceleratedLookupCommitment<Self>> {
        use crate::lookup::{
            InMemoryChunk, OwnedWindowChunk, TraceRowChunk, stage_2_trace_chunked,
        };
        use crate::p3_adapter::spill::{SpillPackWriter, SpilledMatrix};
        use crate::traits::{Pcs as _, flatten_to_base};

        enum CpuRowChunk<'a> {
            InMemory(InMemoryChunk<'a, Val>),
            Spilled(OwnedWindowChunk<Val>),
        }

        impl TraceRowChunk<Val> for CpuRowChunk<'_> {
            #[inline]
            fn row_pair(&self, local_row: usize) -> (&[Val], &[Val]) {
                match self {
                    Self::InMemory(chunk) => chunk.row_pair(local_row),
                    Self::Spilled(chunk) => chunk.row_pair(local_row),
                }
            }
        }

        enum TraceBacking<'a> {
            Borrowed(&'a RowMajorMatrix<Val>),
            Owned(RowMajorMatrix<Val>),
            Spilled(SpilledMatrix),
        }

        impl TraceBacking<'_> {
            fn load_chunk(&self, start: usize, len: usize) -> CpuRowChunk<'_> {
                match self {
                    Self::Borrowed(mat) => CpuRowChunk::InMemory(InMemoryChunk::new(mat, start)),
                    Self::Owned(mat) => CpuRowChunk::InMemory(InMemoryChunk::new(mat, start)),
                    Self::Spilled(spilled) => CpuRowChunk::Spilled(OwnedWindowChunk::new(
                        spilled.read_window_with_wrap(start, len),
                    )),
                }
            }
        }

        let spill_dir = self.spill_dir();
        let any_unmaterialized = inputs
            .iter()
            .any(|input| !input.lookup_values.is_materialized());
        if spill_dir.is_none() && !any_unmaterialized {
            return None;
        }

        let mut spill_pack = spill_dir.map(SpillPackWriter::create);
        let mut spilled_ldes =
            Vec::with_capacity(if spill_dir.is_some() { inputs.len() } else { 0 });
        let mut prehashed = Vec::with_capacity(if spill_dir.is_some() { inputs.len() } else { 0 });
        let height_counts: std::collections::HashMap<usize, usize> = if spill_dir.is_some() {
            let mut counts = std::collections::HashMap::new();
            for input in inputs {
                let lde_height = input.lookup_values.height() << self.log_blowup;
                *counts.entry(lde_height).or_insert(0) += 1;
            }
            counts
        } else {
            std::collections::HashMap::new()
        };
        let mut in_memory_ldes =
            Vec::with_capacity(if spill_dir.is_none() { inputs.len() } else { 0 });
        let mut intermediates = Vec::with_capacity(inputs.len());

        for input in inputs {
            let circuit = input.circuit;
            let height = input.lookup_values.height();
            let (stage_2_ext, delta) = if input.lookup_values.is_materialized() {
                input.lookup_values.stage_2_trace(
                    circuit.lookup_group_size,
                    lookup_challenge,
                    &fingerprint_challenge,
                )
            } else if circuit.graph.lookups.is_empty() {
                (
                    RowMajorMatrix::new(vec![ExtVal::ZERO; height], 1),
                    ExtVal::ZERO,
                )
            } else {
                let (s1_data, s1_idx) = input.stage_1;
                let main_backing = if let Some(spilled) = s1_data.spilled_original_trace(s1_idx) {
                    TraceBacking::Spilled(spilled)
                } else if let Some(owned) = s1_data.take_in_memory_original_trace(s1_idx) {
                    TraceBacking::Owned(owned)
                } else {
                    TraceBacking::Owned(
                        self.committed_trace(s1_data, s1_idx, height)
                            .expect("GoldilocksBlake3Config always rebuilds committed traces"),
                    )
                };

                let prep_backing = if circuit.graph.lookup_prefix_uses_preprocessed() {
                    if let Some(pp) = circuit.preprocessed.as_ref() {
                        Some(TraceBacking::Borrowed(pp))
                    } else if let Some((prep_data, prep_idx)) = input.preprocessed {
                        if let Some(spilled) = prep_data.spilled_original_trace(prep_idx) {
                            Some(TraceBacking::Spilled(spilled))
                        } else {
                            Some(TraceBacking::Owned(
                                self.committed_trace(prep_data, prep_idx, height).expect(
                                    "GoldilocksBlake3Config always rebuilds committed traces",
                                ),
                            ))
                        }
                    } else {
                        panic!(
                            "circuit lookup expressions read preprocessed columns, but no preprocessed trace is available"
                        );
                    }
                } else {
                    None
                };

                stage_2_trace_chunked(
                    circuit,
                    height,
                    lookup_challenge,
                    &fingerprint_challenge,
                    |start, len| main_backing.load_chunk(start, len),
                    prep_backing
                        .as_ref()
                        .map(|prep| move |start, len| prep.load_chunk(start, len)),
                )
            };

            accumulator += delta;
            intermediates.push(accumulator);

            let stage_2_base = flatten_to_base::<Val, ExtVal>(stage_2_ext);
            let domain = self.pcs.natural_domain_for_degree(stage_2_base.height());
            let lde = self.pcs.compute_lde(domain, stage_2_base);
            if let Some(pack) = &mut spill_pack {
                if height_counts.get(&lde.height()).copied() == Some(1) {
                    let (spilled, digests) =
                        pack.append_matrix_and_hash(&lde, self.pcs.leaf_domain());
                    spilled_ldes.push(spilled);
                    prehashed.push(Some(digests));
                } else {
                    spilled_ldes.push(pack.append_matrix(&lde));
                    prehashed.push(None);
                }
                // Drop `lde` immediately before building the next circuit's stage-2 trace.
            } else {
                in_memory_ldes.push(lde);
            }
        }

        // Release retained/spilled 1× main traces now that stage-2 is complete.
        // Preprocessed 1× traces stay attached to `ProverKey` for reuse across
        // proofs/shards.
        if let Some(first) = inputs.first() {
            first.stage_1.0.release_original_traces();
        }

        let (commitment, data) = if let Some(mut pack) = spill_pack {
            pack.flush();
            self.pcs
                .commit_spilled_ldes_prehashed(spilled_ldes, None, prehashed)
        } else {
            self.pcs.commit_in_memory_ldes(in_memory_ldes)
        };
        Some((commitment, data, intermediates))
    }

    #[cfg(not(feature = "cuda"))]
    fn accelerated_quotient_commit(
        &self,
        inputs: &[crate::config::QuotientCommitInput<'_, Self>],
        alpha: ExtVal,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        use crate::p3_adapter::pcs::{lde_from_shifted_coefficients, shifted_quotient_slices};
        use crate::p3_adapter::spill::SpillPackWriter;
        use crate::traits::{EvaluationDomain, Pcs as _, flatten_to_base};

        let spill_dir = self.spill_dir()?;
        let mut pack = SpillPackWriter::create(spill_dir);
        let mut spilled_ldes = Vec::with_capacity(inputs.len());
        let mut prehashed = Vec::with_capacity(inputs.len());
        let dft = self.pcs.quotient_dft();
        let log_blowup = self.log_blowup;
        let mut height_counts = std::collections::HashMap::new();
        for input in inputs {
            let lde_height =
                (input.quotient_domain.size() / input.circuit.quotient_degree()) << log_blowup;
            *height_counts.entry(lde_height).or_insert(0) += 1;
        }

        for input in inputs {
            let circuit = input.circuit;
            let quotient_degree = circuit.quotient_degree();
            let preprocessed_evals = input.preprocessed.map(|(data, idx)| {
                self.pcs
                    .get_evaluations_on_domain(data, idx, input.quotient_domain)
            });
            let stage_1_evals = self.pcs.get_evaluations_on_domain(
                input.stage_1.0,
                input.stage_1.1,
                input.quotient_domain,
            );
            let stage_2_evals = self.pcs.get_evaluations_on_domain(
                input.stage_2.0,
                input.stage_2.1,
                input.quotient_domain,
            );
            let q_vals = crate::prover::quotient_values::<Self>(
                circuit,
                &input.lookup_publics,
                input.trace_domain,
                input.quotient_domain,
                &preprocessed_evals,
                &stage_1_evals,
                &stage_2_evals,
                None,
                alpha,
                input.constraint_count,
            );
            // Release this circuit's trace evaluations before running quotient
            // DFTs and LDE expansion.
            drop(preprocessed_evals);
            drop(stage_1_evals);
            drop(stage_2_evals);

            let quotient_flat = flatten_to_base::<Val, ExtVal>(RowMajorMatrix::new_col(q_vals));
            let sliced = shifted_quotient_slices(
                dft,
                quotient_flat,
                EvaluationDomain::first_point(&input.quotient_domain),
                quotient_degree,
            );
            let lde = lde_from_shifted_coefficients(dft, sliced, log_blowup);
            if height_counts.get(&lde.height()).copied() == Some(1) {
                let (spilled, digests) = pack.append_matrix_and_hash(&lde, self.pcs.leaf_domain());
                spilled_ldes.push(spilled);
                prehashed.push(Some(digests));
            } else {
                spilled_ldes.push(pack.append_matrix(&lde));
                prehashed.push(None);
            }
        }

        pack.flush();
        Some(
            self.pcs
                .commit_spilled_ldes_prehashed(spilled_ldes, None, prehashed),
        )
    }

    fn canonicalize_proof(proof: &mut crate::prover::Proof<Self>) {
        canonicalize_goldilocks_fri_proof::<Self, ExtVal, Mmcs, ExtMmcs>(proof);
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
        cuda_accelerated_quotient_values(
            circuit,
            lookup_publics,
            trace_domain,
            quotient_domain,
            preprocessed,
            stage_1,
            stage_2,
            alpha,
            constraint_count,
        )
    }

    #[cfg(feature = "cuda")]
    fn accelerated_quotient_commit(
        &self,
        inputs: &[crate::config::QuotientCommitInput<'_, Self>],
        alpha: ExtVal,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        let jobs: Vec<CudaQuotientJob<'_>> = inputs
            .iter()
            .map(|input| CudaQuotientJob {
                circuit: input.circuit,
                lookup_publics: &input.lookup_publics,
                trace_domain: input.trace_domain,
                quotient_domain: input.quotient_domain,
                preprocessed: input.preprocessed,
                stage_1: input.stage_1,
                stage_2: input.stage_2,
                constraint_count: input.constraint_count,
            })
            .collect();
        cuda_accelerated_quotient_commit(
            &self.pcs.dft,
            &self.pcs.mmcs,
            self.log_blowup,
            &jobs,
            alpha,
        )
    }

    #[cfg(feature = "cuda")]
    fn accelerated_lookup_commit(
        &self,
        inputs: &[crate::config::LookupCommitInput<'_, Self>],
        lookup_challenge: ExtVal,
        fingerprint_challenge: ExtVal,
        accumulator: ExtVal,
    ) -> Option<(
        crate::config::Com<Self>,
        crate::config::PcsData<Self>,
        Vec<ExtVal>,
    )> {
        let jobs: Vec<CudaLookupJob<'_>> = inputs
            .iter()
            .map(|input| CudaLookupJob {
                circuit: input.circuit,
                lookup_values: input.lookup_values,
                preprocessed: input.preprocessed,
                stage_1: input.stage_1,
            })
            .collect();
        cuda_accelerated_lookup_commit(
            &self.pcs.dft,
            &self.pcs.mmcs,
            self.log_blowup,
            &jobs,
            lookup_challenge,
            fingerprint_challenge,
            accumulator,
        )
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

#[cfg(test)]
pub(crate) type Blake3CompressionFunction = CompressionFunctionFromHasher<Blake3, 2, 32>;

#[cfg(feature = "cuda")]
impl CudaPcsDft<Val> for CudaDft {
    fn coset_lde_workspace_bytes(
        &self,
        height: usize,
        width: usize,
        added_bits: usize,
        shift: Val,
    ) -> usize {
        if width == 0 {
            return 0;
        }
        let plan = self.lde_plan(height, width, added_bits, shift);
        plan.scratch_bytes().saturating_add(plan.constant_bytes())
    }

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

    fn try_coset_lde_batch_host(
        &self,
        matrix: &RowMajorMatrix<Val>,
        added_bits: usize,
        shift: Val,
    ) -> Option<RowMajorMatrix<Val>> {
        self.try_auxiliary_coset_lde(matrix, added_bits, shift)
    }
}

type Dft = Radix2DitParallel<Val>;
#[cfg(not(feature = "cuda"))]
type PcsDft = Dft;
#[cfg(feature = "cuda")]
type PcsDft = CudaDft;

fn new_mmcs(cap_height: usize, #[cfg(feature = "cuda")] device_id: i32) -> Mmcs {
    #[cfg(not(feature = "cuda"))]
    {
        crate::p3_adapter::spill::SpillableMmcs::new_plain(cap_height)
    }
    #[cfg(feature = "cuda")]
    {
        crate::cuda::mmcs::CudaMmcs::new_plain(cap_height, device_id)
    }
}

fn canonicalize_goldilocks_fri_proof<SC, EF, InputMmcs, FriMmcs>(
    proof: &mut crate::prover::Proof<SC>,
) where
    EF: ExtensionField<Val> + crate::traits::ExtensionOf<Val>,
    InputMmcs: p3_commit::Mmcs<Val>,
    FriMmcs: p3_commit::Mmcs<EF>,
    SC: ProofConfig<Challenge = EF>,
    SC::Pcs: crate::traits::Pcs<
            F = Val,
            Challenge = EF,
            Proof = p3_fri::FriProof<
                EF,
                FriMmcs,
                Val,
                Vec<p3_fri::BatchMultiOpening<Val, InputMmcs>>,
            >,
        >,
{
    fn canonical_base(value: &mut Val) {
        *value = Val::from_u64(value.as_canonical_u64());
    }

    fn canonical_ext<EF: ExtensionField<Val> + crate::traits::ExtensionOf<Val>>(value: &mut EF) {
        let coefficients: &[Val] =
            <EF as crate::traits::ExtensionOf<Val>>::as_basis_coefficients_slice(value);
        *value = <EF as crate::traits::ExtensionOf<Val>>::from_basis_coefficients_fn(|i| {
            Val::from_u64(coefficients[i].as_canonical_u64())
        });
    }

    fn canonical_opened<EF: ExtensionField<Val> + crate::traits::ExtensionOf<Val>>(
        round: &mut p3_commit::OpenedValuesForRound<EF>,
    ) {
        for matrix in round {
            for point in matrix {
                point.iter_mut().for_each(canonical_ext::<EF>);
            }
        }
    }

    proof
        .intermediate_accumulators
        .iter_mut()
        .for_each(canonical_ext::<EF>);
    canonical_opened(&mut proof.quotient_opened_values);
    if let Some(round) = &mut proof.preprocessed_opened_values {
        canonical_opened(round);
    }
    canonical_opened(&mut proof.stage_1_opened_values);
    canonical_opened(&mut proof.stage_2_opened_values);

    let fri = &mut proof.opening_proof;
    fri.commit_pow_witnesses.iter_mut().for_each(canonical_base);
    canonical_base(&mut fri.query_pow_witness);
    fri.final_poly.iter_mut().for_each(canonical_ext::<EF>);
    for opening in &mut fri.input_openings {
        for query in &mut opening.opened_values {
            for row in query {
                row.iter_mut().for_each(canonical_base);
            }
        }
    }
    for step in &mut fri.commit_phase_openings {
        for siblings in &mut step.sibling_values {
            siblings.iter_mut().for_each(canonical_ext::<EF>);
        }
    }
}

/// Extension MMCS for Protocol `v6` over any challenge extension `EF`.
pub type V6ExtMmcs<EF> = ExtensionMmcs<Val, EF, Mmcs>;

/// Inner PCS for Protocol `v6` over any challenge extension `EF` (CPU).
#[cfg(not(feature = "cuda"))]
pub type V6InnerPcs<EF> =
    DomainMappedPcs<crate::p3_adapter::spill::SpillableTwoAdicFriPcs<PcsDft, V6ExtMmcs<EF>>>;
/// Inner PCS for Protocol `v6` over any challenge extension `EF` (CUDA).
#[cfg(feature = "cuda")]
pub type V6InnerPcs<EF> =
    DomainMappedPcs<crate::cuda::pcs::CudaTwoAdicFriPcs<Val, PcsDft, Mmcs, V6ExtMmcs<EF>>>;

/// Protocol `v6` FRI PCS parameterized over any Goldilocks challenge extension
/// `EF` (degree-3 cubic trinomial [`ExtVal3`] or degree-2 binomial [`ExtVal2`]).
pub type V6Pcs<EF> = crate::p3_adapter::pcs::FriPcs<
    Val,
    EF,
    DomainChallenger<EF>,
    Dft,
    Mmcs,
    V6ExtMmcs<EF>,
    V6InnerPcs<EF>,
>;

/// Protocol `v6` Goldilocks/BLAKE3 configuration over the degree-3 cubic
/// trinomial extension $\mathbb{F}_{p^3} = \mathbb{F}_p[X]/(X^3 - X - 1)$
/// (192-bit field, 128-bit conjectured / 100+ bit provable soundness with
/// multilinear tensor FRI batching and domain-separated BLAKE3).
pub type GoldilocksBlake3V6Config = GoldilocksV6Config<ExtVal3>;

/// Protocol `v6` Goldilocks/BLAKE3 configuration over the degree-2 binomial
/// extension $\mathbb{F}_{p^2} = \mathbb{F}_p[X]/(X^2 - 7)$ (same hardened `v6`
/// transcript, VK digest, domain-separated BLAKE3, and tensor FRI batching as
/// [`GoldilocksBlake3V6Config`], useful for apples-to-apples benchmarking).
pub type GoldilocksBlake3V6Fp2Config = GoldilocksV6Config<ExtVal2>;

/// Hardened Protocol `v6` [`ProofConfig`] parameterized by Goldilocks
/// challenge extension `EF`.
///
/// Protocol `v6` cryptographic guarantees:
/// - Zero-overhead BLAKE3 `derive_key` domain separation across Merkle leaves
///   (`"multi-stark/v6/merkle/leaf"`), Merkle internal 2-to-1 compressions
///   (`"multi-stark/v6/merkle/compress"`), single-proof transcripts
///   (`"multi-stark/v6/transcript/single"`), batch-proof transcripts
///   (`"multi-stark/v6/transcript/batch"`), and VK/constraint-graph digests
///   (`"multi-stark/v6/vk"`).
/// - Explicit extension degree, extension modulus, FRI batching mode, and full
///   canonical VK/constraint-graph digest bound into the transcript.
/// - Multilinear tensor FRI batching (`FriBatching::Tensor`, default) or
///   Horner batching (`FriBatching::Horner`).
pub struct GoldilocksV6Config<EF>
where
    EF: TwoAdicField + ExtensionField<Val> + crate::traits::ExtensionOf<Val>,
{
    pcs: V6Pcs<EF>,
    challenger_seed: Vec<u8>,
    max_log_degree: usize,
    max_quotient_degree: usize,
    log_blowup: usize,
    cap_height: usize,
    commitment_parameters: CommitmentParameters,
    fri_parameters: FriParameters,
    fri_batching: FriBatching,
}

fn build_v6_challenger_seed<EF: crate::traits::ExtensionOf<Val>>(
    commitment_parameters: CommitmentParameters,
    fri_parameters: FriParameters,
    fri_batching: FriBatching,
) -> Vec<u8> {
    let modulus = EF::MODULUS;
    let modulus_param = match modulus {
        ExtensionModulus::Binomial { w } => w.as_canonical_u64(),
        ExtensionModulus::Base | ExtensionModulus::CubicTrinomial => 0u64,
    };
    let mut seed = b"multi-stark/v6".to_vec();
    let params = [
        u64::try_from(EF::D).expect("extension degree exceeds u64"),
        u64::from(modulus.tag()),
        modulus_param,
        fri_batching.tag(),
        u64::try_from(commitment_parameters.log_blowup).expect("parameter exceeds u64"),
        u64::try_from(commitment_parameters.cap_height).expect("parameter exceeds u64"),
        u64::try_from(fri_parameters.log_final_poly_len).expect("parameter exceeds u64"),
        u64::try_from(fri_parameters.max_log_arity).expect("parameter exceeds u64"),
        u64::try_from(fri_parameters.num_queries).expect("parameter exceeds u64"),
        u64::try_from(fri_parameters.commit_proof_of_work_bits).expect("parameter exceeds u64"),
        u64::try_from(fri_parameters.query_proof_of_work_bits).expect("parameter exceeds u64"),
    ];
    for p in params {
        seed.extend_from_slice(&p.to_le_bytes());
    }
    seed
}

impl<EF> GoldilocksV6Config<EF>
where
    EF: TwoAdicField + ExtensionField<Val> + crate::traits::ExtensionOf<Val>,
{
    /// Creates a Protocol `v6` configuration using multilinear tensor FRI
    /// batching ([`FriBatching::Tensor`]) by default.
    pub fn new(commitment_parameters: CommitmentParameters, fri_parameters: FriParameters) -> Self {
        Self::with_device(
            commitment_parameters,
            fri_parameters,
            FriBatching::Tensor,
            None,
        )
    }

    /// Creates a Protocol `v6` configuration with an explicit [`FriBatching`]
    /// strategy (`Tensor` or `Horner`).
    pub fn new_with_batching(
        commitment_parameters: CommitmentParameters,
        fri_parameters: FriParameters,
        fri_batching: FriBatching,
    ) -> Self {
        Self::with_device(commitment_parameters, fri_parameters, fri_batching, None)
    }

    /// Creates a Protocol `v6` configuration targeting an optional CUDA device.
    pub fn with_device(
        commitment_parameters: CommitmentParameters,
        fri_parameters: FriParameters,
        fri_batching: FriBatching,
        device_id: Option<i32>,
    ) -> Self {
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
        #[cfg(not(feature = "cuda"))]
        let pcs = {
            assert!(
                device_id.is_none_or(|device| device == 0),
                "no CUDA backend to place a prover on device {device_id:?}"
            );
            let val_mmcs =
                crate::p3_adapter::spill::SpillableMmcs::new_v6(commitment_parameters.cap_height);
            let ext_mmcs = ExtensionMmcs::new(val_mmcs.clone());
            let inner_parameters = InnerFriParameters {
                log_blowup: commitment_parameters.log_blowup,
                log_final_poly_len: fri_parameters.log_final_poly_len,
                max_log_arity: fri_parameters.max_log_arity,
                num_queries: fri_parameters.num_queries,
                commit_proof_of_work_bits: fri_parameters.commit_proof_of_work_bits,
                query_proof_of_work_bits: fri_parameters.query_proof_of_work_bits,
                mmcs: ext_mmcs,
            };
            let mut inner = crate::p3_adapter::spill::SpillableTwoAdicFriPcs::new(
                PcsDft::default(),
                val_mmcs,
                inner_parameters,
            )
            .with_batching(fri_batching);
            if let Some(dir) = std::env::var_os("MULTI_STARK_SPILL_DIR")
                && !dir.is_empty()
            {
                inner.set_spill_dir(Some(std::path::PathBuf::from(dir)));
            }
            DomainMappedPcs::new(inner)
        };
        #[cfg(feature = "cuda")]
        let pcs = {
            let dev = device_id.unwrap_or_else(crate::cuda::configured_device);
            let val_mmcs =
                crate::cuda::mmcs::CudaMmcs::new_v6(commitment_parameters.cap_height, dev);
            let ext_mmcs = ExtensionMmcs::new(val_mmcs.clone());
            let inner_parameters = InnerFriParameters {
                log_blowup: commitment_parameters.log_blowup,
                log_final_poly_len: fri_parameters.log_final_poly_len,
                max_log_arity: fri_parameters.max_log_arity,
                num_queries: fri_parameters.num_queries,
                commit_proof_of_work_bits: fri_parameters.commit_proof_of_work_bits,
                query_proof_of_work_bits: fri_parameters.query_proof_of_work_bits,
                mmcs: ext_mmcs,
            };
            let pcs_dft = CudaDft::new(dev);
            DomainMappedPcs::new(
                crate::cuda::pcs::CudaTwoAdicFriPcs::new(pcs_dft, val_mmcs, inner_parameters)
                    .with_batching(fri_batching),
            )
        };

        let challenger_seed =
            build_v6_challenger_seed::<EF>(commitment_parameters, fri_parameters, fri_batching);
        let max_log_degree = Val::TWO_ADICITY - commitment_parameters.log_blowup;
        let max_quotient_degree = 1 << commitment_parameters.log_blowup;
        Self {
            pcs: V6Pcs::new(pcs, commitment_parameters, fri_parameters),
            challenger_seed,
            max_log_degree,
            max_quotient_degree,
            log_blowup: commitment_parameters.log_blowup,
            cap_height: commitment_parameters.cap_height,
            commitment_parameters,
            fri_parameters,
            fri_batching,
        }
    }

    /// Public protocol-domain seed used by the `v6` challenger.
    pub fn challenger_seed(&self) -> &[u8] {
        &self.challenger_seed
    }

    pub fn cap_height(&self) -> usize {
        self.cap_height
    }

    pub fn fri_parameters(&self) -> FriParameters {
        self.fri_parameters
    }

    pub fn fri_batching(&self) -> FriBatching {
        self.fri_batching
    }

    /// Switches the FRI opening batching mode (`Horner` or `Tensor`), updating
    /// both the PCS and the protocol challenger seed.
    pub fn with_fri_batching(self, fri_batching: FriBatching) -> Self {
        let spill = self.spill_dir().map(std::path::Path::to_path_buf);
        #[cfg(feature = "cuda")]
        let device = Some(self.pcs.inner().dft.device_id());
        #[cfg(not(feature = "cuda"))]
        let device = None;
        let cfg = Self::with_device(
            self.commitment_parameters,
            self.fri_parameters,
            fri_batching,
            device,
        );
        match spill {
            Some(dir) => cfg.with_disk_spill(dir),
            None => cfg.without_disk_spill(),
        }
    }

    /// Enables disk-backed spilling of committed LDE matrices on the CPU prover.
    #[cfg_attr(feature = "cuda", allow(unused_mut))]
    pub fn with_disk_spill(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        #[cfg(not(feature = "cuda"))]
        self.pcs
            .inner_mut()
            .inner_mut()
            .set_spill_dir(Some(dir.into()));
        #[cfg(feature = "cuda")]
        let _ = dir.into();
        self
    }

    /// Disables disk spilling (forces in-memory LDE storage on CPU).
    #[cfg_attr(feature = "cuda", allow(unused_mut))]
    pub fn without_disk_spill(mut self) -> Self {
        #[cfg(not(feature = "cuda"))]
        self.pcs.inner_mut().inner_mut().set_spill_dir(None);
        self
    }

    /// Returns the configured disk-spill directory, if enabled on the CPU backend.
    pub fn spill_dir(&self) -> Option<&std::path::Path> {
        #[cfg(not(feature = "cuda"))]
        {
            self.pcs.spill_dir()
        }
        #[cfg(feature = "cuda")]
        {
            None
        }
    }
}

impl<EF> ProofConfig for GoldilocksV6Config<EF>
where
    EF: TwoAdicField + ExtensionField<Val> + crate::traits::ExtensionOf<Val>,
{
    type Pcs = V6Pcs<EF>;
    type Challenge = EF;
    type Challenger = DomainChallenger<EF>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn protocol_version(&self) -> ProtocolVersion {
        ProtocolVersion::V6
    }

    fn initialise_challenger(&self) -> Self::Challenger {
        self.initialise_challenger_for_mode(TranscriptMode::Single)
    }

    fn initialise_challenger_for_mode(&self, mode: TranscriptMode) -> Self::Challenger {
        let domain = match mode {
            TranscriptMode::Single => Blake3Domain::TranscriptSingle,
            TranscriptMode::Batch => Blake3Domain::TranscriptBatch,
        };
        DomainChallenger::from_domain(self.challenger_seed.clone(), domain)
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

    fn validate_commitment_cap(
        &self,
        commit: &crate::config::Com<Self>,
        max_log_lde_height: usize,
    ) -> bool {
        crate::p3_adapter::cap::expected_merkle_cap_roots(self.cap_height, max_log_lde_height)
            == Some(commit.num_roots())
    }

    #[cfg(feature = "cuda")]
    fn committed_trace(
        &self,
        data: &crate::config::PcsData<Self>,
        index: usize,
        degree: usize,
    ) -> Option<RowMajorMatrix<Val>> {
        use p3_matrix::Matrix;
        let trace = self
            .pcs
            .inner()
            .mmcs
            .committed_trace(data, index, self.log_blowup);
        assert_eq!(trace.height(), degree);
        Some(trace)
    }

    #[cfg(not(feature = "cuda"))]
    fn committed_trace(
        &self,
        data: &crate::config::PcsData<Self>,
        index: usize,
        degree: usize,
    ) -> Option<RowMajorMatrix<Val>> {
        use crate::traits::{EvaluationDomain as _, Pcs as _};
        use p3_dft::TwoAdicSubgroupDft;
        if let Some(spilled) = data.spilled_original_trace(index) {
            let trace = spilled.read_all();
            assert_eq!(trace.height(), degree);
            return Some(trace);
        }
        if let Some(trace) = data.clone_in_memory_original_trace(index) {
            assert_eq!(trace.height(), degree);
            return Some(trace);
        }
        let coset = self
            .pcs
            .natural_domain_for_degree(degree)
            .create_disjoint_domain(degree);
        let evaluations = self
            .pcs
            .get_evaluations_on_domain(data, index, coset)
            .to_row_major_matrix();
        let dft = Radix2DitParallel::<Val>::default();
        let coefficients = dft.coset_idft_batch(evaluations, coset.first_point());
        Some(dft.dft_batch(coefficients).to_row_major_matrix())
    }

    #[cfg(not(feature = "cuda"))]
    fn commit_preprocessed(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (crate::config::Domain<Self>, RowMajorMatrix<Val>, bool),
        >,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        let mut inner_evals =
            evaluations.map(|(domain, mat, retain_1x)| (domain.into_inner(), mat, retain_1x));
        self.pcs.commit_streaming_with_originals(&mut inner_evals)
    }

    #[cfg(not(feature = "cuda"))]
    fn commit_main_streaming(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (
                crate::config::Domain<Self>,
                crate::witness::TraceSource<Val>,
                bool,
            ),
        >,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        let mut inner_evals = evaluations.map(|(domain, trace, retain_1x)| {
            (domain.into_inner(), trace.materialize(), retain_1x)
        });
        self.pcs.commit_streaming_with_originals(&mut inner_evals)
    }

    #[cfg(not(feature = "cuda"))]
    fn accelerated_lookup_commit(
        &self,
        inputs: &[crate::config::LookupCommitInput<'_, Self>],
        lookup_challenge: EF,
        fingerprint_challenge: EF,
        mut accumulator: EF,
    ) -> Option<crate::config::AcceleratedLookupCommitment<Self>> {
        use crate::lookup::{
            InMemoryChunk, OwnedWindowChunk, TraceRowChunk, stage_2_trace_chunked,
        };
        use crate::p3_adapter::spill::{SpillPackWriter, SpilledMatrix};
        use crate::traits::{Pcs as _, flatten_to_base};

        enum CpuRowChunk<'a> {
            InMemory(InMemoryChunk<'a, Val>),
            Spilled(OwnedWindowChunk<Val>),
        }

        impl TraceRowChunk<Val> for CpuRowChunk<'_> {
            #[inline]
            fn row_pair(&self, local_row: usize) -> (&[Val], &[Val]) {
                match self {
                    Self::InMemory(chunk) => chunk.row_pair(local_row),
                    Self::Spilled(chunk) => chunk.row_pair(local_row),
                }
            }
        }

        enum TraceBacking<'a> {
            Borrowed(&'a RowMajorMatrix<Val>),
            Owned(RowMajorMatrix<Val>),
            Spilled(SpilledMatrix),
        }

        impl TraceBacking<'_> {
            fn load_chunk(&self, start: usize, len: usize) -> CpuRowChunk<'_> {
                match self {
                    Self::Borrowed(mat) => CpuRowChunk::InMemory(InMemoryChunk::new(mat, start)),
                    Self::Owned(mat) => CpuRowChunk::InMemory(InMemoryChunk::new(mat, start)),
                    Self::Spilled(spilled) => CpuRowChunk::Spilled(OwnedWindowChunk::new(
                        spilled.read_window_with_wrap(start, len),
                    )),
                }
            }
        }

        let spill_dir = self.spill_dir();
        let any_unmaterialized = inputs
            .iter()
            .any(|input| !input.lookup_values.is_materialized());
        if spill_dir.is_none() && !any_unmaterialized {
            return None;
        }

        let mut spill_pack = spill_dir.map(SpillPackWriter::create);
        let mut spilled_ldes =
            Vec::with_capacity(if spill_dir.is_some() { inputs.len() } else { 0 });
        let mut prehashed = Vec::with_capacity(if spill_dir.is_some() { inputs.len() } else { 0 });
        let height_counts: std::collections::HashMap<usize, usize> = if spill_dir.is_some() {
            let mut counts = std::collections::HashMap::new();
            for input in inputs {
                let lde_height = input.lookup_values.height() << self.log_blowup;
                *counts.entry(lde_height).or_insert(0) += 1;
            }
            counts
        } else {
            std::collections::HashMap::new()
        };
        let mut in_memory_ldes =
            Vec::with_capacity(if spill_dir.is_none() { inputs.len() } else { 0 });
        let mut intermediates = Vec::with_capacity(inputs.len());
        let leaf_domain = self.pcs.leaf_domain();

        for input in inputs {
            let circuit = input.circuit;
            let height = input.lookup_values.height();
            let (stage_2_ext, delta) = if input.lookup_values.is_materialized() {
                input.lookup_values.stage_2_trace(
                    circuit.lookup_group_size,
                    lookup_challenge,
                    &fingerprint_challenge,
                )
            } else if circuit.graph.lookups.is_empty() {
                (
                    RowMajorMatrix::new(vec![<EF as PrimeCharacteristicRing>::ZERO; height], 1),
                    <EF as PrimeCharacteristicRing>::ZERO,
                )
            } else {
                let (s1_data, s1_idx) = input.stage_1;
                let main_backing = if let Some(spilled) = s1_data.spilled_original_trace(s1_idx) {
                    TraceBacking::Spilled(spilled)
                } else if let Some(owned) = s1_data.take_in_memory_original_trace(s1_idx) {
                    TraceBacking::Owned(owned)
                } else {
                    let trace_domain = self.pcs.natural_domain_for_degree(height);
                    TraceBacking::Owned(
                        self.pcs
                            .get_evaluations_on_domain(s1_data, s1_idx, trace_domain)
                            .to_row_major_matrix(),
                    )
                };

                let prep_backing = if circuit.graph.lookup_prefix_uses_preprocessed() {
                    if let Some(pp) = circuit.preprocessed.as_ref() {
                        Some(TraceBacking::Borrowed(pp))
                    } else if let Some((prep_data, prep_idx)) = input.preprocessed {
                        if let Some(spilled) = prep_data.spilled_original_trace(prep_idx) {
                            Some(TraceBacking::Spilled(spilled))
                        } else {
                            let trace_domain = self.pcs.natural_domain_for_degree(height);
                            Some(TraceBacking::Owned(
                                self.pcs
                                    .get_evaluations_on_domain(prep_data, prep_idx, trace_domain)
                                    .to_row_major_matrix(),
                            ))
                        }
                    } else {
                        panic!(
                            "circuit lookup expressions read preprocessed columns, but no preprocessed trace is available"
                        );
                    }
                } else {
                    None
                };

                stage_2_trace_chunked(
                    circuit,
                    height,
                    lookup_challenge,
                    &fingerprint_challenge,
                    |start, len| main_backing.load_chunk(start, len),
                    prep_backing
                        .as_ref()
                        .map(|prep| move |start, len| prep.load_chunk(start, len)),
                )
            };

            accumulator += delta;
            intermediates.push(accumulator);

            let stage_2_base = flatten_to_base::<Val, EF>(stage_2_ext);
            let domain = self.pcs.natural_domain_for_degree(stage_2_base.height());
            let lde = self.pcs.compute_lde(domain.into_inner(), stage_2_base);
            if let Some(pack) = &mut spill_pack {
                if height_counts.get(&lde.height()).copied() == Some(1) {
                    let (spilled, digests) = pack.append_matrix_and_hash(&lde, leaf_domain);
                    spilled_ldes.push(spilled);
                    prehashed.push(Some(digests));
                } else {
                    spilled_ldes.push(pack.append_matrix(&lde));
                    prehashed.push(None);
                }
            } else {
                in_memory_ldes.push(lde);
            }
        }

        if let Some(first) = inputs.first() {
            first.stage_1.0.release_original_traces();
        }

        let (commitment, data) = if let Some(mut pack) = spill_pack {
            pack.flush();
            self.pcs
                .commit_spilled_ldes_prehashed(spilled_ldes, None, prehashed)
        } else {
            self.pcs.commit_in_memory_ldes(in_memory_ldes)
        };
        Some((commitment, data, intermediates))
    }

    #[cfg(not(feature = "cuda"))]
    fn accelerated_quotient_commit(
        &self,
        inputs: &[crate::config::QuotientCommitInput<'_, Self>],
        alpha: EF,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        use crate::p3_adapter::pcs::{lde_from_shifted_coefficients, shifted_quotient_slices};
        use crate::p3_adapter::spill::SpillPackWriter;
        use crate::traits::{EvaluationDomain, Pcs as _, flatten_to_base};

        let spill_dir = self.spill_dir()?;
        let mut pack = SpillPackWriter::create(spill_dir);
        let mut spilled_ldes = Vec::with_capacity(inputs.len());
        let mut prehashed = Vec::with_capacity(inputs.len());
        let dft = self.pcs.quotient_dft();
        let log_blowup = self.log_blowup;
        let leaf_domain = self.pcs.leaf_domain();
        let mut height_counts = std::collections::HashMap::new();
        for input in inputs {
            let lde_height =
                (input.quotient_domain.size() / input.circuit.quotient_degree()) << log_blowup;
            *height_counts.entry(lde_height).or_insert(0) += 1;
        }

        for input in inputs {
            let circuit = input.circuit;
            let quotient_degree = circuit.quotient_degree();
            let preprocessed_evals = input.preprocessed.map(|(data, idx)| {
                self.pcs
                    .get_evaluations_on_domain(data, idx, input.quotient_domain)
            });
            let stage_1_evals = self.pcs.get_evaluations_on_domain(
                input.stage_1.0,
                input.stage_1.1,
                input.quotient_domain,
            );
            let stage_2_evals = self.pcs.get_evaluations_on_domain(
                input.stage_2.0,
                input.stage_2.1,
                input.quotient_domain,
            );
            let q_vals = crate::prover::quotient_values::<Self>(
                circuit,
                &input.lookup_publics,
                input.trace_domain,
                input.quotient_domain,
                &preprocessed_evals,
                &stage_1_evals,
                &stage_2_evals,
                None,
                alpha,
                input.constraint_count,
            );
            drop(preprocessed_evals);
            drop(stage_1_evals);
            drop(stage_2_evals);

            let quotient_flat = flatten_to_base::<Val, EF>(RowMajorMatrix::new_col(q_vals));
            let sliced = shifted_quotient_slices(
                dft,
                quotient_flat,
                EvaluationDomain::first_point(&input.quotient_domain),
                quotient_degree,
            );
            let lde = lde_from_shifted_coefficients(dft, sliced, log_blowup);
            if height_counts.get(&lde.height()).copied() == Some(1) {
                let (spilled, digests) = pack.append_matrix_and_hash(&lde, leaf_domain);
                spilled_ldes.push(spilled);
                prehashed.push(Some(digests));
            } else {
                spilled_ldes.push(pack.append_matrix(&lde));
                prehashed.push(None);
            }
        }

        pack.flush();
        Some(
            self.pcs
                .commit_spilled_ldes_prehashed(spilled_ldes, None, prehashed),
        )
    }

    #[cfg(feature = "cuda")]
    fn commit_main(
        &self,
        evaluations: Vec<(
            crate::config::Domain<Self>,
            crate::witness::TraceSource<Val>,
        )>,
    ) -> (crate::config::Com<Self>, crate::config::PcsData<Self>) {
        let inner_evals = evaluations
            .into_iter()
            .map(|(domain, source)| (domain.into_inner(), source))
            .collect();
        crate::cuda::witness::commit::<EF, DomainChallenger<EF>>(self.pcs.inner(), inner_evals)
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
        alpha: EF,
        constraint_count: usize,
    ) -> Option<Vec<EF>> {
        cuda_accelerated_quotient_values(
            circuit,
            lookup_publics,
            trace_domain.into_inner(),
            quotient_domain.into_inner(),
            preprocessed,
            stage_1,
            stage_2,
            alpha,
            constraint_count,
        )
    }

    #[cfg(feature = "cuda")]
    fn accelerated_quotient_commit(
        &self,
        inputs: &[crate::config::QuotientCommitInput<'_, Self>],
        alpha: EF,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        let jobs: Vec<CudaQuotientJob<'_>> = inputs
            .iter()
            .map(|input| CudaQuotientJob {
                circuit: input.circuit,
                lookup_publics: &input.lookup_publics,
                trace_domain: input.trace_domain.into_inner(),
                quotient_domain: input.quotient_domain.into_inner(),
                preprocessed: input.preprocessed,
                stage_1: input.stage_1,
                stage_2: input.stage_2,
                constraint_count: input.constraint_count,
            })
            .collect();
        let inner = self.pcs.inner();
        cuda_accelerated_quotient_commit(&inner.dft, &inner.mmcs, self.log_blowup, &jobs, alpha)
    }

    #[cfg(feature = "cuda")]
    fn accelerated_lookup_commit(
        &self,
        inputs: &[crate::config::LookupCommitInput<'_, Self>],
        lookup_challenge: EF,
        fingerprint_challenge: EF,
        accumulator: EF,
    ) -> Option<crate::config::AcceleratedLookupCommitment<Self>> {
        let jobs: Vec<CudaLookupJob<'_>> = inputs
            .iter()
            .map(|input| CudaLookupJob {
                circuit: input.circuit,
                lookup_values: input.lookup_values,
                preprocessed: input.preprocessed,
                stage_1: input.stage_1,
            })
            .collect();
        let inner = self.pcs.inner();
        cuda_accelerated_lookup_commit(
            &inner.dft,
            &inner.mmcs,
            self.log_blowup,
            &jobs,
            lookup_challenge,
            fingerprint_challenge,
            accumulator,
        )
    }

    fn canonicalize_proof(proof: &mut crate::prover::Proof<Self>) {
        canonicalize_goldilocks_fri_proof::<Self, EF, Mmcs, V6ExtMmcs<EF>>(proof);
    }
}

/// Base hiding MMCS for Protocol `v6-zk` (CPU spillable or CUDA hiding MMCS).
#[cfg(not(feature = "cuda"))]
pub type ZkMmcs = crate::p3_adapter::spill_zk::HidingSpillableMmcs;
#[cfg(feature = "cuda")]
pub type ZkMmcs = crate::cuda::hiding_mmcs::HidingCudaMmcs;

/// Extension hiding MMCS for Protocol `v6-zk` over any challenge extension `EF`.
pub type V6ZkExtMmcs<EF> = ExtensionMmcs<Val, EF, ZkMmcs>;

/// Inner hiding PCS for Protocol `v6-zk` over any challenge extension `EF` (CPU).
#[cfg(not(feature = "cuda"))]
pub type V6ZkInnerPcs<EF> = DomainMappedPcs<
    crate::p3_adapter::spill_zk::HidingSpillableTwoAdicFriPcs<PcsDft, V6ZkExtMmcs<EF>>,
>;
/// Inner hiding PCS for Protocol `v6-zk` over any challenge extension `EF` (CUDA).
#[cfg(feature = "cuda")]
pub type V6ZkInnerPcs<EF> = DomainMappedPcs<
    crate::cuda::hiding_pcs::HidingCudaTwoAdicFriPcs<Val, PcsDft, ZkMmcs, V6ZkExtMmcs<EF>>,
>;

/// Protocol `v6-zk` hiding FRI PCS parameterized over any Goldilocks challenge
/// extension `EF`.
pub type V6ZkPcs<EF> = crate::p3_adapter::pcs::FriPcs<
    Val,
    EF,
    DomainChallenger<EF>,
    Dft,
    ZkMmcs,
    V6ZkExtMmcs<EF>,
    V6ZkInnerPcs<EF>,
>;

/// Protocol `v6-zk` zero-knowledge Goldilocks/BLAKE3 configuration over the
/// degree-3 cubic trinomial extension $\mathbb{F}_{p^3} = \mathbb{F}_p[X]/(X^3 - X - 1)$
/// (128-bit statistical/ROM post-quantum transparent ZK, `W_R = 4` base
/// randomizer columns, 2× odd-coset hiding PCS, zero-storage BLAKE3-CTR leaf
/// salts, multilinear tensor FRI batching).
pub type GoldilocksBlake3V6ZkConfig = GoldilocksV6ZkConfig<ExtVal3>;

/// Protocol `v6-zk` zero-knowledge Goldilocks/BLAKE3 configuration over the
/// degree-2 binomial extension $\mathbb{F}_{p^2} = \mathbb{F}_p[X]/(X^2 - 7)$.
pub type GoldilocksBlake3V6ZkFp2Config = GoldilocksV6ZkConfig<ExtVal2>;

/// Zero-knowledge Protocol `v6-zk` [`ProofConfig`] parameterized by Goldilocks
/// challenge extension `EF`.
pub struct GoldilocksV6ZkConfig<EF>
where
    EF: TwoAdicField + ExtensionField<Val> + crate::traits::ExtensionOf<Val>,
{
    pcs: V6ZkPcs<EF>,
    challenger_seed: Vec<u8>,
    min_log_degree: usize,
    max_log_degree: usize,
    max_quotient_degree: usize,
    log_blowup: usize,
    cap_height: usize,
    randomizer_width: usize,
    commitment_parameters: CommitmentParameters,
    fri_parameters: FriParameters,
    fri_batching: FriBatching,
}

fn build_v6_zk_challenger_seed<EF: crate::traits::ExtensionOf<Val>>(
    commitment_parameters: CommitmentParameters,
    fri_parameters: FriParameters,
    fri_batching: FriBatching,
    randomizer_width: usize,
    min_log_degree: usize,
) -> Vec<u8> {
    let modulus = EF::MODULUS;
    let modulus_param = match modulus {
        ExtensionModulus::Binomial { w } => w.as_canonical_u64(),
        ExtensionModulus::Base | ExtensionModulus::CubicTrinomial => 0u64,
    };
    let mut seed = b"multi-stark/v6-zk".to_vec();
    let params = [
        u64::try_from(EF::D).expect("extension degree exceeds u64"),
        u64::from(modulus.tag()),
        modulus_param,
        fri_batching.tag(),
        u64::try_from(crate::p3_adapter::zk::ZK_LEAF_SALT_BYTES).expect("parameter exceeds u64"),
        u64::try_from(randomizer_width).expect("parameter exceeds u64"),
        u64::try_from(min_log_degree).expect("parameter exceeds u64"),
        u64::try_from(commitment_parameters.log_blowup).expect("parameter exceeds u64"),
        u64::try_from(commitment_parameters.cap_height).expect("parameter exceeds u64"),
        u64::try_from(fri_parameters.log_final_poly_len).expect("parameter exceeds u64"),
        u64::try_from(fri_parameters.max_log_arity).expect("parameter exceeds u64"),
        u64::try_from(fri_parameters.num_queries).expect("parameter exceeds u64"),
        u64::try_from(fri_parameters.commit_proof_of_work_bits).expect("parameter exceeds u64"),
        u64::try_from(fri_parameters.query_proof_of_work_bits).expect("parameter exceeds u64"),
    ];
    for p in params {
        seed.extend_from_slice(&p.to_le_bytes());
    }
    seed
}

impl<EF> GoldilocksV6ZkConfig<EF>
where
    EF: TwoAdicField + ExtensionField<Val> + crate::traits::ExtensionOf<Val>,
{
    /// Creates a Protocol `v6-zk` configuration using multilinear tensor FRI
    /// batching ([`FriBatching::Tensor`]) and `W_R = 4` base randomizer
    /// columns by default.
    pub fn new(commitment_parameters: CommitmentParameters, fri_parameters: FriParameters) -> Self {
        Self::with_options(
            commitment_parameters,
            fri_parameters,
            FriBatching::Tensor,
            crate::p3_adapter::zk::ZK_RANDOMIZER_WIDTH,
            None,
        )
    }

    /// Creates a Protocol `v6-zk` configuration with an explicit [`FriBatching`]
    /// strategy (`Tensor` or `Horner`).
    pub fn new_with_batching(
        commitment_parameters: CommitmentParameters,
        fri_parameters: FriParameters,
        fri_batching: FriBatching,
    ) -> Self {
        Self::with_options(
            commitment_parameters,
            fri_parameters,
            fri_batching,
            crate::p3_adapter::zk::ZK_RANDOMIZER_WIDTH,
            None,
        )
    }

    /// Creates a Protocol `v6-zk` configuration targeting an optional CUDA device.
    pub fn with_device(
        commitment_parameters: CommitmentParameters,
        fri_parameters: FriParameters,
        fri_batching: FriBatching,
        device_id: Option<i32>,
    ) -> Self {
        Self::with_options(
            commitment_parameters,
            fri_parameters,
            fri_batching,
            crate::p3_adapter::zk::ZK_RANDOMIZER_WIDTH,
            device_id,
        )
    }

    /// Creates a Protocol `v6-zk` configuration with full explicit parameters,
    /// validating all ZK security floors fail-closed.
    pub fn with_options(
        commitment_parameters: CommitmentParameters,
        fri_parameters: FriParameters,
        fri_batching: FriBatching,
        randomizer_width: usize,
        device_id: Option<i32>,
    ) -> Self {
        assert!(
            commitment_parameters.log_blowup >= 1,
            "v6-zk requires log_blowup >= 1 (got {})",
            commitment_parameters.log_blowup
        );
        assert!(
            fri_parameters.num_queries >= 1,
            "v6-zk requires num_queries >= 1"
        );
        assert!(
            randomizer_width >= crate::p3_adapter::zk::ZK_RANDOMIZER_WIDTH,
            "v6-zk requires randomizer width W_R >= {} (got {randomizer_width})",
            crate::p3_adapter::zk::ZK_RANDOMIZER_WIDTH
        );
        let min_log_degree =
            crate::p3_adapter::zk::min_zk_log_degree(fri_parameters.num_queries, EF::D);
        let max_log_degree = Val::TWO_ADICITY
            .checked_sub(1 + commitment_parameters.log_blowup)
            .expect("log_blowup exceeds field two-adicity");
        assert!(
            min_log_degree <= max_log_degree,
            "v6-zk min_log_degree ({min_log_degree}) exceeds max_log_degree ({max_log_degree})"
        );
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
        #[cfg(not(feature = "cuda"))]
        let pcs = {
            assert!(
                device_id.is_none_or(|device| device == 0),
                "no CUDA backend to place a prover on device {device_id:?}"
            );
            let val_mmcs = crate::p3_adapter::spill_zk::HidingSpillableMmcs::new_v6_zk(
                commitment_parameters.cap_height,
            );
            let ext_mmcs = ExtensionMmcs::new(val_mmcs.clone());
            let inner_parameters = InnerFriParameters {
                log_blowup: commitment_parameters.log_blowup,
                log_final_poly_len: fri_parameters.log_final_poly_len,
                max_log_arity: fri_parameters.max_log_arity,
                num_queries: fri_parameters.num_queries,
                commit_proof_of_work_bits: fri_parameters.commit_proof_of_work_bits,
                query_proof_of_work_bits: fri_parameters.query_proof_of_work_bits,
                mmcs: ext_mmcs,
            };
            let mut inner = crate::p3_adapter::spill_zk::HidingSpillableTwoAdicFriPcs::new(
                PcsDft::default(),
                val_mmcs,
                inner_parameters,
            )
            .with_batching(fri_batching);
            if let Some(dir) = std::env::var_os("MULTI_STARK_SPILL_DIR")
                && !dir.is_empty()
            {
                inner.set_spill_dir(Some(std::path::PathBuf::from(dir)));
            }
            DomainMappedPcs::new(inner)
        };
        #[cfg(feature = "cuda")]
        let pcs = {
            let dev = device_id.unwrap_or_else(crate::cuda::configured_device);
            let val_mmcs = crate::cuda::hiding_mmcs::HidingCudaMmcs::new_v6_zk(
                commitment_parameters.cap_height,
                dev,
            );
            let ext_mmcs = ExtensionMmcs::new(val_mmcs.clone());
            let inner_parameters = InnerFriParameters {
                log_blowup: commitment_parameters.log_blowup,
                log_final_poly_len: fri_parameters.log_final_poly_len,
                max_log_arity: fri_parameters.max_log_arity,
                num_queries: fri_parameters.num_queries,
                commit_proof_of_work_bits: fri_parameters.commit_proof_of_work_bits,
                query_proof_of_work_bits: fri_parameters.query_proof_of_work_bits,
                mmcs: ext_mmcs,
            };
            let pcs_dft = CudaDft::new(dev);
            DomainMappedPcs::new(
                crate::cuda::hiding_pcs::HidingCudaTwoAdicFriPcs::new(
                    pcs_dft,
                    val_mmcs,
                    inner_parameters,
                )
                .with_batching(fri_batching),
            )
        };

        let challenger_seed = build_v6_zk_challenger_seed::<EF>(
            commitment_parameters,
            fri_parameters,
            fri_batching,
            randomizer_width,
            min_log_degree,
        );
        let max_quotient_degree = 1usize << commitment_parameters.log_blowup;
        Self {
            pcs: V6ZkPcs::new(pcs, commitment_parameters, fri_parameters),
            challenger_seed,
            min_log_degree,
            max_log_degree,
            max_quotient_degree,
            log_blowup: commitment_parameters.log_blowup,
            cap_height: commitment_parameters.cap_height,
            randomizer_width,
            commitment_parameters,
            fri_parameters,
            fri_batching,
        }
    }

    /// Public protocol-domain seed used by the `v6-zk` challenger.
    pub fn challenger_seed(&self) -> &[u8] {
        &self.challenger_seed
    }

    pub fn cap_height(&self) -> usize {
        self.cap_height
    }

    pub fn fri_parameters(&self) -> FriParameters {
        self.fri_parameters
    }

    pub fn fri_batching(&self) -> FriBatching {
        self.fri_batching
    }

    pub fn randomizer_width(&self) -> usize {
        self.randomizer_width
    }

    /// Switches the FRI opening batching mode (`Horner` or `Tensor`), updating
    /// both the PCS and the protocol challenger seed.
    pub fn with_fri_batching(self, fri_batching: FriBatching) -> Self {
        let spill = self.spill_dir().map(std::path::Path::to_path_buf);
        #[cfg(feature = "cuda")]
        let device = Some(self.pcs.inner().dft.device_id());
        #[cfg(not(feature = "cuda"))]
        let device = None;
        let cfg = Self::with_options(
            self.commitment_parameters,
            self.fri_parameters,
            fri_batching,
            self.randomizer_width,
            device,
        );
        match spill {
            Some(dir) => cfg.with_disk_spill(dir),
            None => cfg.without_disk_spill(),
        }
    }

    /// Overrides the randomizer width `W_R` (must be `>= 4`).
    pub fn with_randomizer_width(self, randomizer_width: usize) -> Self {
        let spill = self.spill_dir().map(std::path::Path::to_path_buf);
        #[cfg(feature = "cuda")]
        let device = Some(self.pcs.inner().dft.device_id());
        #[cfg(not(feature = "cuda"))]
        let device = None;
        let cfg = Self::with_options(
            self.commitment_parameters,
            self.fri_parameters,
            self.fri_batching,
            randomizer_width,
            device,
        );
        match spill {
            Some(dir) => cfg.with_disk_spill(dir),
            None => cfg.without_disk_spill(),
        }
    }

    /// Enables disk-backed spilling of committed LDE matrices on the CPU prover.
    #[cfg_attr(feature = "cuda", allow(unused_mut))]
    pub fn with_disk_spill(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        #[cfg(not(feature = "cuda"))]
        self.pcs
            .inner_mut()
            .inner_mut()
            .set_spill_dir(Some(dir.into()));
        #[cfg(feature = "cuda")]
        let _ = dir.into();
        self
    }

    /// Disables disk spilling (forces in-memory LDE storage on CPU).
    #[cfg_attr(feature = "cuda", allow(unused_mut))]
    pub fn without_disk_spill(mut self) -> Self {
        #[cfg(not(feature = "cuda"))]
        self.pcs.inner_mut().inner_mut().set_spill_dir(None);
        self
    }

    /// Returns the configured disk-spill directory, if enabled on the CPU backend.
    pub fn spill_dir(&self) -> Option<&std::path::Path> {
        #[cfg(not(feature = "cuda"))]
        {
            self.pcs.spill_dir()
        }
        #[cfg(feature = "cuda")]
        {
            None
        }
    }
}

impl<EF> ProofConfig for GoldilocksV6ZkConfig<EF>
where
    EF: TwoAdicField + ExtensionField<Val> + crate::traits::ExtensionOf<Val>,
{
    type Pcs = V6ZkPcs<EF>;
    type Challenge = EF;
    type Challenger = DomainChallenger<EF>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn protocol_version(&self) -> ProtocolVersion {
        ProtocolVersion::V6Zk
    }

    fn min_log_degree(&self) -> usize {
        self.min_log_degree
    }

    fn zk_randomizer_width(&self) -> usize {
        self.randomizer_width
    }

    fn sample_zk_base_randomness(&self, count: usize) -> Vec<Val> {
        crate::p3_adapter::zk::csprng_field_vec(count)
    }

    fn initialise_challenger(&self) -> Self::Challenger {
        self.initialise_challenger_for_mode(TranscriptMode::Single)
    }

    fn initialise_challenger_for_mode(&self, mode: TranscriptMode) -> Self::Challenger {
        let domain = match mode {
            TranscriptMode::Single => Blake3Domain::ZkTranscriptSingle,
            TranscriptMode::Batch => Blake3Domain::ZkTranscriptBatch,
        };
        DomainChallenger::from_domain(self.challenger_seed.clone(), domain)
    }

    fn max_log_degree(&self) -> usize {
        self.max_log_degree
    }

    fn max_log_quotient_domain(&self) -> usize {
        Val::TWO_ADICITY
    }

    fn max_quotient_degree(&self) -> usize {
        self.max_quotient_degree
    }

    fn log_blowup(&self) -> usize {
        self.log_blowup
    }

    fn validate_commitment_cap(
        &self,
        commit: &crate::config::Com<Self>,
        max_log_lde_height: usize,
    ) -> bool {
        crate::p3_adapter::cap::expected_merkle_cap_roots(self.cap_height, max_log_lde_height)
            == Some(commit.num_roots())
    }

    #[cfg(feature = "cuda")]
    fn committed_trace(
        &self,
        data: &crate::config::PcsData<Self>,
        index: usize,
        degree: usize,
    ) -> Option<RowMajorMatrix<Val>> {
        use crate::traits::Pcs as _;
        use p3_matrix::Matrix;
        let trace_domain = self.pcs.natural_domain_for_degree(degree);
        let trace = self
            .pcs
            .get_evaluations_on_domain(data, index, trace_domain)
            .to_row_major_matrix();
        assert_eq!(trace.height(), degree);
        Some(trace)
    }

    #[cfg(not(feature = "cuda"))]
    fn committed_trace(
        &self,
        data: &crate::config::PcsData<Self>,
        index: usize,
        degree: usize,
    ) -> Option<RowMajorMatrix<Val>> {
        use crate::traits::Pcs as _;
        if let Some(spilled) = data.spilled_original_trace(index) {
            let trace = spilled.read_all();
            assert_eq!(trace.height(), degree);
            return Some(trace);
        }
        if let Some(trace) = data.clone_in_memory_original_trace(index) {
            assert_eq!(trace.height(), degree);
            return Some(trace);
        }
        let trace_domain = self.pcs.natural_domain_for_degree(degree);
        let trace = self
            .pcs
            .get_evaluations_on_domain(data, index, trace_domain)
            .to_row_major_matrix();
        assert_eq!(trace.height(), degree);
        Some(trace)
    }

    #[cfg(not(feature = "cuda"))]
    fn commit_preprocessed(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (crate::config::Domain<Self>, RowMajorMatrix<Val>, bool),
        >,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        let mut inner_evals =
            evaluations.map(|(domain, mat, retain_1x)| (domain.into_inner(), mat, retain_1x));
        self.pcs.commit_preprocessed_streaming(&mut inner_evals)
    }

    #[cfg(feature = "cuda")]
    fn commit_preprocessed(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (crate::config::Domain<Self>, RowMajorMatrix<Val>, bool),
        >,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        let evals: Vec<_> = evaluations
            .map(|(domain, mat, _)| (domain.into_inner(), mat))
            .collect();
        if evals.is_empty() {
            None
        } else {
            Some(self.pcs.inner().commit_preprocessed(evals))
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn commit_main_streaming(
        &self,
        evaluations: &mut dyn Iterator<
            Item = (
                crate::config::Domain<Self>,
                crate::witness::TraceSource<Val>,
                bool,
            ),
        >,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        let mut inner_evals = evaluations.map(|(domain, trace, retain_1x)| {
            (domain.into_inner(), trace.materialize(), retain_1x)
        });
        self.pcs.commit_streaming_with_originals(&mut inner_evals)
    }

    fn commit_quotient_zk(
        &self,
        quotients: Vec<(crate::config::Domain<Self>, RowMajorMatrix<Val>, usize)>,
    ) -> (crate::config::Com<Self>, crate::config::PcsData<Self>) {
        use crate::p3_adapter::pcs::lde_from_shifted_coefficients;
        use crate::p3_adapter::zk::{csprng_field_vec, shifted_quotient_slices_zk};
        use crate::traits::EvaluationDomain;
        use p3_matrix::Matrix as _;

        let dft = Dft::default();
        let mut ldes: Vec<RowMajorMatrix<Val>> = Vec::with_capacity(quotients.len() + 1);
        let mut max_ext_height = 0usize;
        for (quotient_domain, evaluations, quotient_degree) in quotients {
            let sliced = shifted_quotient_slices_zk(
                &dft,
                evaluations,
                EvaluationDomain::first_point(&quotient_domain),
                quotient_degree,
            );
            max_ext_height = max_ext_height.max(sliced.height());
            ldes.push(lde_from_shifted_coefficients(&dft, sliced, self.log_blowup));
        }
        assert!(
            max_ext_height > 0,
            "commit_quotient_zk requires at least one quotient matrix"
        );
        // Sample `W_R` uniform base-field shifted coefficient polynomials of
        // degree `< 2 * N_max_active` and extend in one forward DFT.
        let r_shifted = RowMajorMatrix::new(
            csprng_field_vec(max_ext_height * self.randomizer_width),
            self.randomizer_width,
        );
        ldes.push(lde_from_shifted_coefficients(
            &dft,
            r_shifted,
            self.log_blowup,
        ));
        <V6ZkInnerPcs<EF> as PcsTrait<EF, DomainChallenger<EF>>>::commit_ldes(&self.pcs, ldes)
    }

    #[cfg(not(feature = "cuda"))]
    fn accelerated_lookup_commit(
        &self,
        inputs: &[crate::config::LookupCommitInput<'_, Self>],
        lookup_challenge: EF,
        fingerprint_challenge: EF,
        mut accumulator: EF,
    ) -> Option<crate::config::AcceleratedLookupCommitment<Self>> {
        use crate::lookup::{
            InMemoryChunk, OwnedWindowChunk, TraceRowChunk, apply_first_row_delta_shift,
            stage_2_trace_chunked,
        };
        use crate::p3_adapter::spill::{SpillPackWriter, SpilledMatrix};
        use crate::p3_adapter::zk::{csprng_field_vec, random_seed};
        use crate::traits::{Pcs as _, flatten_to_base};

        enum CpuRowChunk<'a> {
            InMemory(InMemoryChunk<'a, Val>),
            Spilled(OwnedWindowChunk<Val>),
        }

        impl TraceRowChunk<Val> for CpuRowChunk<'_> {
            #[inline]
            fn row_pair(&self, local_row: usize) -> (&[Val], &[Val]) {
                match self {
                    Self::InMemory(chunk) => chunk.row_pair(local_row),
                    Self::Spilled(chunk) => chunk.row_pair(local_row),
                }
            }
        }

        enum TraceBacking<'a> {
            Borrowed(&'a RowMajorMatrix<Val>),
            Owned(RowMajorMatrix<Val>),
            Spilled(SpilledMatrix),
        }

        impl TraceBacking<'_> {
            fn load_chunk(&self, start: usize, len: usize) -> CpuRowChunk<'_> {
                match self {
                    Self::Borrowed(mat) => CpuRowChunk::InMemory(InMemoryChunk::new(mat, start)),
                    Self::Owned(mat) => CpuRowChunk::InMemory(InMemoryChunk::new(mat, start)),
                    Self::Spilled(spilled) => CpuRowChunk::Spilled(OwnedWindowChunk::new(
                        spilled.read_window_with_wrap(start, len),
                    )),
                }
            }
        }

        let spill_dir = self.spill_dir();
        let any_unmaterialized = inputs
            .iter()
            .any(|input| !input.lookup_values.is_materialized());
        if spill_dir.is_none() && !any_unmaterialized {
            return None;
        }

        let num_active = inputs.len();
        let use_acc_mask = num_active > 1;
        let ext_d = EF::D;
        let mu_base = if use_acc_mask {
            csprng_field_vec((num_active - 1) * ext_d)
        } else {
            Vec::new()
        };
        let min_height = inputs
            .iter()
            .map(|input| input.lookup_values.height())
            .min()
            .unwrap_or(0);

        let salt_seed = random_seed();
        let mut spill_pack = spill_dir.map(SpillPackWriter::create);
        let total_mats = num_active + usize::from(use_acc_mask);
        let mut spilled_ldes = Vec::with_capacity(if spill_dir.is_some() { total_mats } else { 0 });
        let mut prehashed = Vec::with_capacity(if spill_dir.is_some() { total_mats } else { 0 });
        let height_counts: std::collections::HashMap<usize, usize> = if spill_dir.is_some() {
            let mut counts = std::collections::HashMap::new();
            for input in inputs {
                let lde_height = input.lookup_values.height() << (self.log_blowup + 1);
                *counts.entry(lde_height).or_insert(0) += 1;
            }
            if use_acc_mask {
                let mask_lde_height = min_height << (self.log_blowup + 1);
                *counts.entry(mask_lde_height).or_insert(0) += 1;
            }
            counts
        } else {
            std::collections::HashMap::new()
        };
        let mut in_memory_ldes =
            Vec::with_capacity(if spill_dir.is_none() { total_mats } else { 0 });
        let mut intermediates = Vec::with_capacity(num_active);
        let leaf_domain = self.pcs.leaf_domain();

        for (pos, input) in inputs.iter().enumerate() {
            let circuit = input.circuit;
            let height = input.lookup_values.height();
            let (mut stage_2_ext, delta) = if input.lookup_values.is_materialized() {
                input.lookup_values.stage_2_trace(
                    circuit.lookup_group_size,
                    lookup_challenge,
                    &fingerprint_challenge,
                )
            } else if circuit.graph.lookups.is_empty() {
                (
                    RowMajorMatrix::new(vec![<EF as PrimeCharacteristicRing>::ZERO; height], 1),
                    <EF as PrimeCharacteristicRing>::ZERO,
                )
            } else {
                let (s1_data, s1_idx) = input.stage_1;
                let main_backing = if let Some(spilled) = s1_data.spilled_original_trace(s1_idx) {
                    TraceBacking::Spilled(spilled)
                } else if let Some(owned) = s1_data.take_in_memory_original_trace(s1_idx) {
                    TraceBacking::Owned(owned)
                } else {
                    let trace_domain = self.pcs.natural_domain_for_degree(height);
                    TraceBacking::Owned(
                        self.pcs
                            .get_evaluations_on_domain(s1_data, s1_idx, trace_domain)
                            .to_row_major_matrix(),
                    )
                };

                let prep_backing = if circuit.graph.lookup_prefix_uses_preprocessed() {
                    if let Some(pp) = circuit.preprocessed.as_ref() {
                        Some(TraceBacking::Borrowed(pp))
                    } else if let Some((prep_data, prep_idx)) = input.preprocessed {
                        if let Some(spilled) = prep_data.spilled_original_trace(prep_idx) {
                            Some(TraceBacking::Spilled(spilled))
                        } else {
                            let trace_domain = self.pcs.natural_domain_for_degree(height);
                            Some(TraceBacking::Owned(
                                self.pcs
                                    .get_evaluations_on_domain(prep_data, prep_idx, trace_domain)
                                    .to_row_major_matrix(),
                            ))
                        }
                    } else {
                        panic!(
                            "circuit lookup expressions read preprocessed columns, but no preprocessed trace is available"
                        );
                    }
                } else {
                    None
                };

                stage_2_trace_chunked(
                    circuit,
                    height,
                    lookup_challenge,
                    &fingerprint_challenge,
                    |start, len| main_backing.load_chunk(start, len),
                    prep_backing
                        .as_ref()
                        .map(|prep| move |start, len| prep.load_chunk(start, len)),
                )
            };

            accumulator += delta;
            if use_acc_mask {
                apply_first_row_delta_shift(&mut stage_2_ext, delta);
                if pos + 1 < num_active {
                    let mu =
                        EF::from_basis_coefficients_slice(&mu_base[pos * ext_d..(pos + 1) * ext_d])
                            .expect("mask basis slice");
                    intermediates.push(accumulator + mu);
                } else {
                    intermediates.push(accumulator);
                }
            } else {
                intermediates.push(accumulator);
            }

            let stage_2_base = flatten_to_base::<Val, EF>(stage_2_ext);
            let domain = self.pcs.natural_domain_for_degree(stage_2_base.height());
            let lde = self
                .pcs
                .compute_blinded_lde(domain.into_inner(), &stage_2_base);
            if let Some(pack) = &mut spill_pack {
                if height_counts.get(&lde.height()).copied() == Some(1) {
                    let (spilled, digests) =
                        pack.append_matrix_and_hash_salted(&lde, leaf_domain, &salt_seed);
                    spilled_ldes.push(spilled);
                    prehashed.push(Some(digests));
                } else {
                    spilled_ldes.push(pack.append_matrix(&lde));
                    prehashed.push(None);
                }
            } else {
                in_memory_ldes.push(lde);
            }
        }

        if use_acc_mask {
            let mask_cols = (num_active - 1) * ext_d;
            let mut mask_values = Vec::with_capacity(min_height * mask_cols);
            mask_values.extend_from_slice(&mu_base);
            if min_height > 1 {
                mask_values.extend(csprng_field_vec((min_height - 1) * mask_cols));
            }
            let mask_mat = RowMajorMatrix::new(mask_values, mask_cols);
            let min_domain = self.pcs.natural_domain_for_degree(min_height);
            let mask_lde = self
                .pcs
                .compute_blinded_lde(min_domain.into_inner(), &mask_mat);
            if let Some(pack) = &mut spill_pack {
                if height_counts.get(&mask_lde.height()).copied() == Some(1) {
                    let (spilled, digests) =
                        pack.append_matrix_and_hash_salted(&mask_lde, leaf_domain, &salt_seed);
                    spilled_ldes.push(spilled);
                    prehashed.push(Some(digests));
                } else {
                    spilled_ldes.push(pack.append_matrix(&mask_lde));
                    prehashed.push(None);
                }
            } else {
                in_memory_ldes.push(mask_lde);
            }
        }

        if let Some(first) = inputs.first() {
            first.stage_1.0.release_original_traces();
        }

        let (commitment, data) = if let Some(mut pack) = spill_pack {
            pack.flush();
            self.pcs
                .commit_spilled_ldes_prehashed(spilled_ldes, None, prehashed, salt_seed)
        } else {
            self.pcs.commit_in_memory_ldes(in_memory_ldes)
        };
        Some((commitment, data, intermediates))
    }

    #[cfg(not(feature = "cuda"))]
    fn accelerated_quotient_commit(
        &self,
        inputs: &[crate::config::QuotientCommitInput<'_, Self>],
        alpha: EF,
    ) -> Option<(crate::config::Com<Self>, crate::config::PcsData<Self>)> {
        use crate::p3_adapter::pcs::lde_from_shifted_coefficients;
        use crate::p3_adapter::spill::SpillPackWriter;
        use crate::p3_adapter::zk::{csprng_field_vec, random_seed, shifted_quotient_slices_zk};
        use crate::traits::{EvaluationDomain, Pcs as _, flatten_to_base};

        let spill_dir = self.spill_dir()?;
        let salt_seed = random_seed();
        let mut pack = SpillPackWriter::create(spill_dir);
        let mut spilled_ldes = Vec::with_capacity(inputs.len() + 1);
        let mut prehashed = Vec::with_capacity(inputs.len() + 1);
        let dft = self.pcs.quotient_dft();
        let log_blowup = self.log_blowup;
        let leaf_domain = self.pcs.leaf_domain();
        let mut height_counts = std::collections::HashMap::new();
        let mut max_ext_height = 0usize;
        for input in inputs {
            let q_zk = input.circuit.quotient_degree_for(true);
            let ext_height = input.quotient_domain.size() / q_zk;
            max_ext_height = max_ext_height.max(ext_height);
            let lde_height = ext_height << log_blowup;
            *height_counts.entry(lde_height).or_insert(0) += 1;
        }
        // Account for the randomizer matrix `R(X)` at height `max_ext_height << log_blowup`.
        let r_lde_height = max_ext_height << log_blowup;
        *height_counts.entry(r_lde_height).or_insert(0) += 1;

        for input in inputs {
            let circuit = input.circuit;
            let q_zk = circuit.quotient_degree_for(true);
            let preprocessed_evals = input.preprocessed.map(|(data, idx)| {
                self.pcs
                    .get_evaluations_on_domain(data, idx, input.quotient_domain)
            });
            let stage_1_evals = self.pcs.get_evaluations_on_domain(
                input.stage_1.0,
                input.stage_1.1,
                input.quotient_domain,
            );
            let stage_2_evals = self.pcs.get_evaluations_on_domain(
                input.stage_2.0,
                input.stage_2.1,
                input.quotient_domain,
            );
            let mask_evals_owned = input.zk_boundary_mask.map(|mask| {
                (
                    self.pcs.get_evaluations_on_domain(
                        mask.data,
                        mask.matrix_index,
                        input.quotient_domain,
                    ),
                    mask.active_pos,
                    mask.num_active,
                )
            });
            let mask_ref = mask_evals_owned
                .as_ref()
                .map(|(evals, active_pos, num_active)| (evals, *active_pos, *num_active));
            let q_vals = crate::prover::quotient_values::<Self>(
                circuit,
                &input.lookup_publics,
                input.trace_domain,
                input.quotient_domain,
                &preprocessed_evals,
                &stage_1_evals,
                &stage_2_evals,
                mask_ref,
                alpha,
                input.constraint_count,
            );
            drop(mask_evals_owned);
            drop(preprocessed_evals);
            drop(stage_1_evals);
            drop(stage_2_evals);

            let quotient_flat = flatten_to_base::<Val, EF>(RowMajorMatrix::new_col(q_vals));
            let sliced = shifted_quotient_slices_zk(
                dft,
                quotient_flat,
                EvaluationDomain::first_point(&input.quotient_domain),
                q_zk,
            );
            let lde = lde_from_shifted_coefficients(dft, sliced, log_blowup);
            if height_counts.get(&lde.height()).copied() == Some(1) {
                let (spilled, digests) =
                    pack.append_matrix_and_hash_salted(&lde, leaf_domain, &salt_seed);
                spilled_ldes.push(spilled);
                prehashed.push(Some(digests));
            } else {
                spilled_ldes.push(pack.append_matrix(&lde));
                prehashed.push(None);
            }
        }

        let r_shifted = RowMajorMatrix::new(
            csprng_field_vec(max_ext_height * self.randomizer_width),
            self.randomizer_width,
        );
        let r_lde = lde_from_shifted_coefficients(dft, r_shifted, log_blowup);
        if height_counts.get(&r_lde.height()).copied() == Some(1) {
            let (spilled, digests) =
                pack.append_matrix_and_hash_salted(&r_lde, leaf_domain, &salt_seed);
            spilled_ldes.push(spilled);
            prehashed.push(Some(digests));
        } else {
            spilled_ldes.push(pack.append_matrix(&r_lde));
            prehashed.push(None);
        }

        pack.flush();
        Some(
            self.pcs
                .commit_spilled_ldes_prehashed(spilled_ldes, None, prehashed, salt_seed),
        )
    }

    fn canonicalize_proof(proof: &mut crate::prover::Proof<Self>) {
        canonicalize_goldilocks_fri_proof::<Self, EF, ZkMmcs, V6ZkExtMmcs<EF>>(proof);
    }
}

fn new_pcs(
    commitment_parameters: CommitmentParameters,
    fri_parameters: FriParameters,
    #[cfg(feature = "cuda")] device_id: i32,
) -> (InnerPcs, Dft) {
    #[cfg(feature = "cuda")]
    let val_mmcs = new_mmcs(commitment_parameters.cap_height, device_id);
    #[cfg(not(feature = "cuda"))]
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
    #[cfg(feature = "cuda")]
    let pcs_dft = CudaDft::new(device_id);
    #[cfg(not(feature = "cuda"))]
    let pcs_dft = PcsDft::default();
    let pcs = InnerPcs::new(pcs_dft, val_mmcs, inner_parameters);
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
        let mmcs = new_mmcs(
            0,
            #[cfg(feature = "cuda")]
            0,
        );
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

        let mmcs = new_mmcs(
            2,
            #[cfg(feature = "cuda")]
            0,
        );
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
