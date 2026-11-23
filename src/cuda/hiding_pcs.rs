//! CUDA hiding 2× interleaved odd-coset FRI PCS (`v6-zk`, ePrint 2024/1037).
//!
//! Mirrors the CPU [`crate::p3_adapter::spill_zk::HidingSpillableTwoAdicFriPcs`]
//! semantics and wire format byte-for-byte while executing coset LDEs, salted
//! Merkle tree construction, barycentric OOD interpolation, reduced-opening
//! accumulation, and FRI commit-phase folding on CUDA.

use core::iter;
use core::marker::PhantomData;
use std::borrow::Cow;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, PeriodicLdeTable};
use p3_dft::{Radix2DFTSmallBatch, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PrimeCharacteristicRing, PrimeField64,
    TwoAdicField, dot_product,
};
use p3_fri::verifier::FriError;
use p3_fri::{
    BatchMultiOpening, CommitPhaseMultiStep, CommitmentWithOpeningPoints, FriParameters, FriProof,
    ProverDataWithOpeningPoints, TwoAdicFriFolding, TwoAdicFriFoldingForMmcs,
    build_periodic_lde_table_two_adic, compute_log_arity_for_round, prover,
};
use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow};
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};
use tracing::debug_span;

use super::hiding_mmcs::{HidingCudaFriMmcs, HidingCudaMmcs};
use super::pcs::{
    CudaPcsDft, bit_reversed_coset, compute_inverse_denominators, cuda_ext_degree_and_w,
    ext_from_goldilocks, ext_to_goldilocks,
};
use super::{
    CudaFriWorkspace, CudaInterpolationTasks, CudaLde, CudaReducedOpening, CudaReductionTasks,
};
use crate::p3_adapter::cap::{CapValidatingMmcs, validate_fri_commitment_caps};
use crate::p3_adapter::fri_batching::{
    FriBatching, FriBatchingSchedule, pack_ext_weights, verify_fri_with_batching,
};
use crate::p3_adapter::zk::{interleave_random_rows, preprocessed_salt_seed, random_seed};

/// CUDA hiding 2× interleaved odd-coset FRI PCS (`const ZK: bool = true`) for
/// Protocol `v6-zk`.
#[derive(Clone, Debug)]
pub struct HidingCudaTwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub(crate) dft: Dft,
    pub(crate) mmcs: InputMmcs,
    pub(crate) fri: FriParameters<FriMmcs>,
    pub(crate) batching: FriBatching,
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, InputMmcs, FriMmcs> HidingCudaTwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub const fn new(dft: Dft, mmcs: InputMmcs, fri: FriParameters<FriMmcs>) -> Self {
        Self {
            dft,
            mmcs,
            fri,
            batching: FriBatching::Horner,
            _phantom: PhantomData,
        }
    }

    /// Configures the FRI opening batching strategy (`Horner` or `Tensor`).
    pub const fn with_batching(mut self, batching: FriBatching) -> Self {
        self.batching = batching;
        self
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_fri_cuda_resident_hiding<Challenge, InputMmcs, FriMmcs, Challenger>(
    params: &FriParameters<FriMmcs>,
    mut inputs: Vec<CudaReducedOpening>,
    challenger: &mut Challenger,
    log_global_max_height: usize,
    prover_data_with_opening_points: &[ProverDataWithOpeningPoints<
        '_,
        Challenge,
        InputMmcs::ProverData<RowMajorMatrix<Goldilocks>>,
    >],
    input_mmcs: &InputMmcs,
    ext_w: Goldilocks,
    ext_degree: usize,
) -> FriProof<Challenge, FriMmcs, Challenger::Witness, Vec<BatchMultiOpening<Goldilocks, InputMmcs>>>
where
    Challenge: ExtensionField<Goldilocks>,
    InputMmcs: Mmcs<Goldilocks>,
    FriMmcs: HidingCudaFriMmcs<Goldilocks, Challenge>,
    Challenger: FieldChallenger<Goldilocks> + GrindingChallenger + CanObserve<FriMmcs::Commitment>,
{
    assert!(!inputs.is_empty());
    assert!(ext_degree == 2 || ext_degree == 3);
    assert!(
        params.num_queries > 0,
        "num_queries must be at least 1 for FRI soundness"
    );
    assert!(
        params.max_log_arity > 0,
        "max_log_arity must be at least 1 to guarantee folding progress"
    );
    assert!(
        inputs
            .windows(2)
            .all(|pair| pair[0].height() > pair[1].height()),
        "inputs are not sorted in strictly descending order of height"
    );
    assert_eq!(
        log_global_max_height,
        log2_strict_usize(inputs[0].height()),
        "log_global_max_height must match the largest input height"
    );
    let log_min_height = log2_strict_usize(inputs.last().unwrap().height());
    if params.log_final_poly_len > 0 {
        assert!(log_min_height > params.log_final_poly_len + params.log_blowup);
    }

    let mut codeword = inputs.remove(0).into_lde();
    let mut commits = Vec::new();
    let mut rounds = Vec::new();
    let mut log_arities = Vec::new();
    let mut commit_pow_witnesses = Vec::new();
    let final_height = params.blowup() * params.final_poly_len();

    while codeword.height() > final_height {
        let log_height = log2_strict_usize(codeword.height());
        let next_log_height = inputs
            .first()
            .map(|input| log2_strict_usize(input.height()));
        let log_arity = compute_log_arity_for_round(
            log_height,
            next_log_height,
            params.log_blowup + params.log_final_poly_len,
            params.max_log_arity,
        );
        let (commitment, round) = tracing::info_span!("stark/fri_round_commit")
            .in_scope(|| params.mmcs.commit_cuda_fri_hiding(codeword, log_arity));
        challenger.observe(commitment.clone());
        commits.push(commitment);
        commit_pow_witnesses.push(
            tracing::info_span!("stark/fri_commit_grind")
                .in_scope(|| challenger.grind(params.commit_proof_of_work_bits)),
        );
        let beta: Challenge = challenger.sample_algebra_element();
        let mut beta_step = beta;
        let mut betas_flat = Vec::with_capacity(log_arity * ext_degree);
        for _ in 0..log_arity {
            let (buf, d) = ext_to_goldilocks::<Goldilocks, Challenge>(beta_step, ext_degree);
            betas_flat.extend_from_slice(&buf[..d]);
            beta_step = beta_step.square();
        }
        let folded_height = round.codeword.height() >> log_arity;
        let add_next = inputs
            .first()
            .is_some_and(|input| input.height() == folded_height);
        let (beta_power_buf, _) =
            ext_to_goldilocks::<Goldilocks, Challenge>(beta.exp_power_of_2(log_arity), ext_degree);
        let g_inv = Goldilocks::two_adic_generator(log_height).inverse();
        codeword = round.codeword.fri_fold(
            add_next.then(|| &inputs[0]),
            &betas_flat,
            log_arity,
            &beta_power_buf[..ext_degree],
            g_inv,
            ext_w,
            ext_degree,
        );
        if add_next {
            inputs.remove(0);
        }
        rounds.push(round);
        log_arities.push(log_arity);
    }
    assert!(inputs.is_empty());

    let final_matrix = codeword.to_row_major_matrix();
    let mut final_values = final_matrix
        .values
        .chunks_exact(ext_degree)
        .take(params.final_poly_len())
        .map(ext_from_goldilocks::<Goldilocks, Challenge>)
        .collect_vec();
    reverse_slice_index_bits(&mut final_values);
    let final_poly = Radix2DFTSmallBatch::default().idft_algebra(final_values);
    challenger.observe_algebra_slice(&final_poly);

    for &log_arity in &log_arities {
        challenger.observe(Goldilocks::from_usize(log_arity));
    }
    let _query_span = tracing::info_span!("stark/fri_queries").entered();
    let query_pow_witness = tracing::info_span!("stark/fri_query_grind")
        .in_scope(|| challenger.grind(params.query_proof_of_work_bits));
    let query_indices = iter::repeat_with(|| challenger.sample_bits(log_global_max_height))
        .take(params.num_queries)
        .collect_vec();
    let input_openings_span = tracing::info_span!("stark/fri_input_openings").entered();
    let input_openings = prover_data_with_opening_points
        .iter()
        .map(|(data, _)| {
            let log_height = log2_strict_usize(input_mmcs.get_max_height(data));
            let indices = query_indices
                .iter()
                .map(|&index| index >> (log_global_max_height - log_height))
                .collect_vec();
            let (opened_values, opening_proof) = input_mmcs.open_multi_batch(&indices, data);
            BatchMultiOpening {
                opened_values,
                opening_proof,
            }
        })
        .collect_vec();
    drop(input_openings_span);

    let mut current_indices = query_indices;
    let commit_phase_openings = rounds
        .iter()
        .zip(&log_arities)
        .map(|(round, &log_arity)| {
            let arity = 1usize << log_arity;
            let positions = current_indices
                .iter()
                .map(|&index| index % arity)
                .collect_vec();
            let group_indices = current_indices
                .iter()
                .map(|&index| index >> log_arity)
                .collect_vec();
            let (opened_rows, opening_proof) =
                tracing::info_span!("stark/fri_commit_phase_opening").in_scope(|| {
                    params
                        .mmcs
                        .open_cuda_fri_hiding_batch(round, &group_indices)
                });
            current_indices = group_indices;
            let sibling_values = positions
                .into_iter()
                .zip(opened_rows)
                .map(|(index_in_group, opened)| {
                    assert_eq!(opened.len(), arity, "FRI opening has the wrong arity");
                    opened
                        .into_iter()
                        .enumerate()
                        .filter_map(|(column, value)| (column != index_in_group).then_some(value))
                        .collect()
                })
                .collect_vec();
            CommitPhaseMultiStep {
                log_arity: u8::try_from(log_arity).expect("FRI arity exceeds u8"),
                sibling_values,
                opening_proof,
            }
        })
        .collect_vec();

    FriProof {
        commit_phase_commits: commits,
        commit_pow_witnesses,
        input_openings,
        commit_phase_openings,
        final_poly,
        query_pow_witness,
    }
}

impl<Dft, FriMmcs> HidingCudaTwoAdicFriPcs<Goldilocks, Dft, HidingCudaMmcs, FriMmcs>
where
    Dft: TwoAdicSubgroupDft<Goldilocks> + CudaPcsDft<Goldilocks> + Sync,
{
    /// Commits public preprocessed matrices onto the `2N · B` LDE coset WITHOUT
    /// random rows and using the deterministic public [`preprocessed_salt_seed`]
    /// so independent prover and verifier setups (including CPU and CUDA) produce
    /// identical `preprocessed_commit` roots.
    pub(crate) fn commit_preprocessed(
        &self,
        evaluations: impl IntoIterator<
            Item = (
                TwoAdicMultiplicativeCoset<Goldilocks>,
                RowMajorMatrix<Goldilocks>,
            ),
        >,
    ) -> (
        <HidingCudaMmcs as Mmcs<Goldilocks>>::Commitment,
        <HidingCudaMmcs as Mmcs<Goldilocks>>::ProverData<RowMajorMatrix<Goldilocks>>,
    ) {
        let evals: Vec<_> = evaluations.into_iter().collect();
        self.commit_inner(evals, false, preprocessed_salt_seed())
    }

    fn commit_inner(
        &self,
        evaluations: Vec<(
            TwoAdicMultiplicativeCoset<Goldilocks>,
            RowMajorMatrix<Goldilocks>,
        )>,
        blind_rows: bool,
        salt_seed: [u8; 32],
    ) -> (
        <HidingCudaMmcs as Mmcs<Goldilocks>>::Commitment,
        <HidingCudaMmcs as Mmcs<Goldilocks>>::ProverData<RowMajorMatrix<Goldilocks>>,
    ) {
        assert!(
            !evaluations.is_empty(),
            "commit requires at least one matrix"
        );
        let added_bits = if blind_rows {
            self.fri.log_blowup
        } else {
            self.fri.log_blowup + 1
        };
        let prepared: Vec<(
            TwoAdicMultiplicativeCoset<Goldilocks>,
            RowMajorMatrix<Goldilocks>,
        )> = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                let mat = if blind_rows {
                    interleave_random_rows(&evals)
                } else {
                    evals
                };
                (domain, mat)
            })
            .collect();

        let source_cells: usize = prepared
            .iter()
            .map(|(_, mat)| mat.height().saturating_mul(mat.width()))
            .sum();
        let max_transform_workspace = prepared
            .iter()
            .map(|(domain, mat)| {
                self.dft.coset_lde_workspace_bytes(
                    mat.height(),
                    mat.width(),
                    added_bits,
                    Goldilocks::GENERATOR / domain.shift(),
                )
            })
            .max()
            .unwrap_or(0);
        let (initial_free, total_bytes) = super::device_memory_info(self.mmcs.device_id());
        let minimum_free = super::minimum_free_bytes(total_bytes);
        let lde_bytes = source_cells
            .saturating_mul(size_of::<Goldilocks>())
            .checked_shl(u32::try_from(added_bits).expect("LDE blowup exceeds u32"))
            .unwrap_or(usize::MAX);

        if lde_bytes
            .saturating_add(minimum_free)
            .saturating_add(max_transform_workspace)
            > initial_free
        {
            let dft = &self.dft;
            let ldes: Vec<RowMajorMatrix<Goldilocks>> = prepared
                .into_par_iter()
                .map(|(domain, evals)| {
                    let shift = Goldilocks::GENERATOR / domain.shift();
                    if let Some(lde) = dft.try_coset_lde_batch_host(&evals, added_bits, shift) {
                        return lde;
                    }
                    let cpu = Radix2DitParallel::<Goldilocks>::default();
                    let mut lde = cpu
                        .coset_lde_batch(evals, added_bits, shift)
                        .bit_reverse_rows()
                        .to_row_major_matrix();
                    lde.values.par_iter_mut().for_each(|v| {
                        *v = Goldilocks::from_u64(v.as_canonical_u64());
                    });
                    lde
                })
                .collect();
            return self.mmcs.commit_cpu_with_seed(ldes, salt_seed);
        }

        let mut ldes = Vec::with_capacity(prepared.len());
        for (domain, evals) in &prepared {
            let shift = Goldilocks::GENERATOR / domain.shift();
            let lde = self.dft.coset_lde_batch_resident(evals, added_bits, shift);
            // SAFETY: `coset_lde_batch_resident` completed synchronously; `v6-zk`
            // does not retain separate 1× GPU trace buffers after LDE creation.
            unsafe { lde.release_trace() };
            ldes.push(lde);
        }
        self.mmcs.commit_cuda_resident_with_seed(ldes, salt_seed)
    }

    fn re_evaluate_from_lde<'a>(
        &self,
        lde_mat: RowMajorMatrix<Goldilocks>,
        poly_height: usize,
        domain: TwoAdicMultiplicativeCoset<Goldilocks>,
    ) -> BitReversedMatrixView<RowMajorMatrixCow<'a, Goldilocks>> {
        let mut coeffs = self.dft.coset_idft_batch(lde_mat, Goldilocks::GENERATOR);
        let width = coeffs.width();
        coeffs.values.truncate(poly_height * width);
        let target_size = domain.size();
        if target_size >= poly_height {
            coeffs.values.resize(target_size * width, Goldilocks::ZERO);
        } else {
            // Fold modulo `(X^m - shift^m)` so evaluating a degree `< 2N` blinded
            // polynomial on a smaller coset `shift * H_m` (e.g. `H_N`) is exact.
            debug_assert!(poly_height.is_multiple_of(target_size));
            let blocks = poly_height / target_size;
            let shift_pow_m = domain
                .shift()
                .exp_u64(u64::try_from(target_size).expect("domain size exceeds u64"));
            let mut factor = shift_pow_m;
            for b in 1..blocks {
                let src_off = b * target_size * width;
                for i in 0..(target_size * width) {
                    let v = coeffs.values[src_off + i];
                    coeffs.values[i] += factor * v;
                }
                factor *= shift_pow_m;
            }
            coeffs.values.truncate(target_size * width);
        }
        let result = self
            .dft
            .coset_dft_batch(coeffs, domain.shift())
            .bit_reverse_rows()
            .to_row_major_matrix();
        let result_width = result.width();
        RowMajorMatrixCow::new(Cow::Owned(result.values), result_width).bit_reverse_rows()
    }
}

impl<Dft, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for HidingCudaTwoAdicFriPcs<Goldilocks, Dft, HidingCudaMmcs, FriMmcs>
where
    Dft: TwoAdicSubgroupDft<Goldilocks> + CudaPcsDft<Goldilocks> + Sync,
    FriMmcs: CapValidatingMmcs<Challenge> + HidingCudaFriMmcs<Goldilocks, Challenge>,
    Challenge: ExtensionField<Goldilocks>,
    Challenger: FieldChallenger<Goldilocks>
        + CanObserve<FriMmcs::Commitment>
        + GrindingChallenger<Witness = Goldilocks>,
{
    type Domain = TwoAdicMultiplicativeCoset<Goldilocks>;
    type Commitment = <HidingCudaMmcs as Mmcs<Goldilocks>>::Commitment;
    type ProverData = <HidingCudaMmcs as Mmcs<Goldilocks>>::ProverData<RowMajorMatrix<Goldilocks>>;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixCow<'a, Goldilocks>>;
    type Proof = FriProof<
        Challenge,
        FriMmcs,
        Goldilocks,
        Vec<BatchMultiOpening<Goldilocks, HidingCudaMmcs>>,
    >;
    type Error = FriError<FriMmcs::Error, <HidingCudaMmcs as Mmcs<Goldilocks>>::Error>;
    const ZK: bool = true;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        TwoAdicMultiplicativeCoset::new(Goldilocks::ONE, log2_strict_usize(degree)).unwrap()
    }

    fn log_max_lde_height(&self) -> usize {
        Goldilocks::TWO_ADICITY
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Goldilocks>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let evals: Vec<_> = evaluations.into_iter().collect();
        self.commit_inner(evals, true, random_seed())
    }

    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Goldilocks>)>,
        _num_chunks: usize,
    ) -> Vec<RowMajorMatrix<Goldilocks>> {
        evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                let interleaved = interleave_random_rows(&evals);
                let shift = Goldilocks::GENERATOR / domain.shift();
                self.dft
                    .coset_lde_batch(interleaved, self.fri.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect()
    }

    fn commit_ldes(
        &self,
        ldes: Vec<RowMajorMatrix<Goldilocks>>,
    ) -> (Self::Commitment, Self::ProverData) {
        let min_height = 1usize << (self.fri.log_blowup + 1);
        for lde in &ldes {
            assert!(
                lde.height() >= min_height,
                "committed ZK LDE height {} is smaller than 2 * blowup ({min_height})",
                lde.height()
            );
        }
        self.mmcs.commit_cuda_storage_with_seed(ldes, random_seed())
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let lde = self.mmcs.get_matrices(prover_data)[idx];
        if domain.shift() == Goldilocks::GENERATOR && lde.height() >= domain.size() {
            return lde.split_rows(domain.size()).0.as_cow().bit_reverse_rows();
        }
        let poly_height = lde.height() >> self.fri.log_blowup;
        let lde_mat = lde.as_view().bit_reverse_rows().to_row_major_matrix();
        self.re_evaluate_from_lde(lde_mat, poly_height, domain)
    }

    fn open(
        &self,
        commitment_data_with_opening_points: Vec<
            ProverDataWithOpeningPoints<'_, Challenge, Self::ProverData>,
        >,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        // Fast path: keep all GPU-resident commitments on device through
        // barycentric OOD interpolation, reduced-opening accumulation, and
        // salted FRI commit-phase folding.
        if commitment_data_with_opening_points
            .iter()
            .all(|(data, _)| self.mmcs.is_cuda_resident(data))
        {
            let resident_rounds = commitment_data_with_opening_points
                .iter()
                .map(|(data, points)| (self.mmcs.resident_arc(data).unwrap(), points))
                .collect_vec();
            let resident_max_height = resident_rounds
                .iter()
                .flat_map(|(ldes, _)| ldes.iter())
                .map(CudaLde::height)
                .max()
                .unwrap_or(0);
            let final_fri_height = self.fri.blowup() * self.fri.final_poly_len();
            if resident_max_height > 1024 && resident_max_height > final_fri_height {
                let _resident_guard = tracing::info_span!("stark/fri_resident_zk").entered();
                let device_id = self.mmcs.device_id();
                let (ext_degree, ext_w) = cuda_ext_degree_and_w::<Goldilocks, Challenge>();
                let mut opened_max_width = 0usize;
                let mut openings_per_height = [0usize; 33];
                for (ldes, points) in &resident_rounds {
                    for (lde, ps) in ldes.iter().zip(points.iter()) {
                        if !ps.is_empty() {
                            opened_max_width = opened_max_width.max(lde.width());
                            let lh = log2_strict_usize(lde.height());
                            openings_per_height[lh] += ps.len();
                        }
                    }
                }
                let max_openings_per_height = openings_per_height.into_iter().max().unwrap_or(0);
                let log_global_max_height = log2_strict_usize(resident_max_height);
                let coset_gold = bit_reversed_coset::<Goldilocks>(log_global_max_height);
                let mut max_log: LinearMap<Challenge, usize> = LinearMap::new();
                for (ldes, points) in &resident_rounds {
                    for (lde, ps) in ldes.iter().zip(points.iter()) {
                        for &z in ps {
                            if let Some(h) = max_log.get_mut(&z) {
                                *h = (*h).max(log2_strict_usize(lde.height()));
                            } else {
                                max_log.insert(z, log2_strict_usize(lde.height()));
                            }
                        }
                    }
                }
                let mut inv_offsets = LinearMap::new();
                let mut inverse_points = Vec::new();
                let mut inverse_counts = Vec::new();
                let mut inverse_count = 0usize;
                for (point, lh) in max_log {
                    inv_offsets.insert(point, inverse_count);
                    let (buf, d) = ext_to_goldilocks::<Goldilocks, Challenge>(point, ext_degree);
                    inverse_points.extend_from_slice(&buf[..d]);
                    let count = 1usize << lh;
                    inverse_counts.push(count);
                    inverse_count += count;
                }
                let mut workspace = CudaFriWorkspace::new(
                    device_id,
                    &inverse_points,
                    &inverse_counts,
                    &coset_gold,
                    ext_w,
                    ext_degree,
                );
                let mut interpolation_tasks = CudaInterpolationTasks::new(ext_degree);
                let mut output_count = 0usize;
                let layouts = resident_rounds
                    .iter()
                    .map(|(ldes, points)| {
                        ldes.iter()
                            .zip(points.iter())
                            .map(|(lde, ps)| {
                                let h = lde.height() >> self.fri.log_blowup;
                                let lh = log2_strict_usize(h);
                                ps.iter()
                                    .map(|&point| {
                                        let offset = output_count;
                                        output_count += lde.width();
                                        let shift_pow = Goldilocks::GENERATOR.exp_power_of_2(lh);
                                        let scale = (point.exp_power_of_2(lh) - shift_pow)
                                            * (Goldilocks::from_usize(h) * shift_pow).inverse();
                                        let (scale_buf, d) =
                                            ext_to_goldilocks::<Goldilocks, Challenge>(
                                                scale, ext_degree,
                                            );
                                        interpolation_tasks.push(
                                            lde,
                                            h,
                                            *inv_offsets.get(&point).unwrap(),
                                            offset,
                                            &scale_buf[..d],
                                        );
                                        (offset, lde.width())
                                    })
                                    .collect_vec()
                            })
                            .collect_vec()
                    })
                    .collect_vec();
                let interpolated = tracing::info_span!("stark/fri_interpolate")
                    .in_scope(|| workspace.interpolate(&interpolation_tasks, output_count, ext_w));
                let all_opened_values = layouts
                    .into_iter()
                    .map(|round| {
                        round
                            .into_iter()
                            .map(|matrix| {
                                matrix
                                    .into_iter()
                                    .map(|(offset, width)| {
                                        interpolated
                                            [offset * ext_degree..(offset + width) * ext_degree]
                                            .chunks_exact(ext_degree)
                                            .map(ext_from_goldilocks::<Goldilocks, Challenge>)
                                            .collect_vec()
                                    })
                                    .collect_vec()
                            })
                            .collect_vec()
                    })
                    .collect_vec();
                for round in &all_opened_values {
                    for matrix in round {
                        for values in matrix {
                            challenger.observe_algebra_slice(values);
                        }
                    }
                }
                let schedule = FriBatchingSchedule::<Challenge>::sample(
                    self.batching,
                    opened_max_width,
                    max_openings_per_height,
                    challenger,
                );
                let col_weights: Vec<Challenge> = match &schedule {
                    FriBatchingSchedule::Horner { alpha } => {
                        alpha.powers().take(opened_max_width).collect()
                    }
                    FriBatchingSchedule::Tensor { col_weights, .. } => col_weights.clone(),
                };
                let mut col_weights_flat = Vec::with_capacity(col_weights.len() * ext_degree);
                for &w in &col_weights {
                    let (buf, d) = ext_to_goldilocks::<Goldilocks, Challenge>(w, ext_degree);
                    col_weights_flat.extend_from_slice(&buf[..d]);
                }
                let mut num_reduced = [0usize; 33];
                let mut opening_idx_by_height = [0usize; 33];
                let mut reduced: [Option<CudaReducedOpening>; 33] = core::array::from_fn(|_| None);
                let mut reduction_tasks = CudaReductionTasks::new(ext_degree);
                for ((ldes, points), openings_round) in
                    resident_rounds.iter().zip(all_opened_values.iter())
                {
                    for ((lde, ps), openings) in
                        ldes.iter().zip(points.iter()).zip(openings_round.iter())
                    {
                        if ps.is_empty() {
                            continue;
                        }
                        let lh = log2_strict_usize(lde.height());
                        let target = reduced[lh].get_or_insert_with(|| {
                            CudaReducedOpening::new_with_degree(device_id, lde.height(), ext_degree)
                        });
                        for (&point, ys) in ps.iter().zip(openings.iter()) {
                            let reduced_y: Challenge =
                                dot_product(col_weights.iter().copied(), ys.iter().copied());
                            let offset = match &schedule {
                                FriBatchingSchedule::Horner { alpha } => alpha.exp_u64(
                                    u64::try_from(num_reduced[lh])
                                        .expect("reduced column count exceeds u64"),
                                ),
                                FriBatchingSchedule::Tensor { open_weights, .. } => {
                                    open_weights[opening_idx_by_height[lh]]
                                }
                            };
                            let (y_buf, d) =
                                ext_to_goldilocks::<Goldilocks, Challenge>(reduced_y, ext_degree);
                            let (off_buf, _) =
                                ext_to_goldilocks::<Goldilocks, Challenge>(offset, ext_degree);
                            reduction_tasks.push(
                                target,
                                lde,
                                *inv_offsets.get(&point).unwrap(),
                                &y_buf[..d],
                                &off_buf[..d],
                            );
                            num_reduced[lh] += lde.width();
                            opening_idx_by_height[lh] += 1;
                        }
                    }
                }
                if !reduction_tasks.is_empty() {
                    tracing::info_span!("stark/fri_reduce").in_scope(|| {
                        workspace.reduce(&reduction_tasks, &col_weights_flat, ext_w);
                    });
                }
                let fri_input = reduced.into_iter().rev().flatten().collect_vec();
                let fri_proof = tracing::info_span!("stark/fri_prove").in_scope(|| {
                    prove_fri_cuda_resident_hiding(
                        &self.fri,
                        fri_input,
                        challenger,
                        log_global_max_height,
                        &commitment_data_with_opening_points,
                        &self.mmcs,
                        ext_w,
                        ext_degree,
                    )
                });
                return (all_opened_values, fri_proof);
            }
        }

        // Fallback path for small matrices (`height <= 1024`) or hybrid/CPU
        // commitments (e.g., wide height groups exceeding 32 KiB per leaf).
        let mats_and_points = commitment_data_with_opening_points
            .iter()
            .map(|(data, points)| {
                let mats = self
                    .mmcs
                    .get_matrices(data)
                    .into_iter()
                    .map(|m| m.as_view())
                    .collect_vec();
                debug_assert_eq!(
                    mats.len(),
                    points.len(),
                    "each matrix should have a corresponding set of evaluation points"
                );
                (mats, points)
            })
            .collect_vec();

        let global_max_height = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats.iter().map(Matrix::height))
            .max()
            .expect("No Matrices Supplied?");
        let mut opened_max_width = 0usize;
        let mut openings_per_height = [0usize; 33];
        for (mats, points) in &mats_and_points {
            for (mat, pts) in mats.iter().zip(points.iter()) {
                if !pts.is_empty() {
                    opened_max_width = opened_max_width.max(mat.width());
                    let lh = log2_strict_usize(mat.height());
                    openings_per_height[lh] += pts.len();
                }
            }
        }
        let max_openings_per_height = openings_per_height.into_iter().max().unwrap_or(0);
        let log_global_max_height = log2_strict_usize(global_max_height);

        let coset = {
            let coset =
                TwoAdicMultiplicativeCoset::new(Goldilocks::GENERATOR, log_global_max_height)
                    .unwrap();
            let mut coset_points = coset.iter().collect();
            reverse_slice_index_bits(&mut coset_points);
            coset_points
        };
        let inv_denoms = compute_inverse_denominators(&mats_and_points, &coset);
        let adjusted_weights: LinearMap<Challenge, Vec<Challenge>> = inv_denoms
            .iter()
            .map(|(point, denoms)| (*point, compute_adjusted_weights(*point, denoms)))
            .collect();

        let all_opened_values = mats_and_points
            .iter()
            .map(|(mats, points)| {
                izip!(mats.iter(), points.iter())
                    .map(|(mat, points_for_mat)| {
                        let h = mat.height() >> self.fri.log_blowup;
                        let (low_coset, _) = mat.split_rows(h);
                        points_for_mat
                            .iter()
                            .map(|&point| {
                                let _guard =
                                    debug_span!("evaluate matrix", dims = %mat.dimensions())
                                        .entered();
                                let adjusted = &adjusted_weights.get(&point).unwrap()[..h];
                                let ys = low_coset.interpolate_coset_with_precomputation(
                                    Goldilocks::GENERATOR,
                                    point,
                                    adjusted,
                                );
                                challenger.observe_algebra_slice(&ys);
                                ys
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();

        let schedule = FriBatchingSchedule::<Challenge>::sample(
            self.batching,
            opened_max_width,
            max_openings_per_height,
            challenger,
        );
        let packed_col_weights: Vec<Challenge::ExtensionPacking> = match &schedule {
            FriBatchingSchedule::Horner { alpha } => {
                Challenge::ExtensionPacking::packed_ext_powers_capped(*alpha, opened_max_width)
                    .collect_vec()
            }
            FriBatchingSchedule::Tensor { col_weights, .. } => {
                pack_ext_weights::<Goldilocks, Challenge>(col_weights)
            }
        };
        let col_weights: Vec<Challenge> = match &schedule {
            FriBatchingSchedule::Horner { alpha } => {
                alpha.powers().take(opened_max_width).collect()
            }
            FriBatchingSchedule::Tensor { col_weights, .. } => col_weights.clone(),
        };

        let mut num_reduced = [0usize; 33];
        let mut opening_idx_by_height = [0usize; 33];
        let mut reduced_openings: [Option<Vec<Challenge>>; 33] = core::array::from_fn(|_| None);

        for ((mats, points), openings_for_round) in
            mats_and_points.iter().zip(all_opened_values.iter())
        {
            for (mat, points_for_mat, openings_for_mat) in
                izip!(mats.iter(), points.iter(), openings_for_round.iter())
            {
                if points_for_mat.is_empty() {
                    continue;
                }
                let log_height = log2_strict_usize(mat.height());
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![Challenge::ZERO; mat.height()]);
                let mat_compressed = mat
                    .rowwise_packed_dot_product::<Challenge>(&packed_col_weights)
                    .collect::<Vec<_>>();
                for (&point, openings) in points_for_mat.iter().zip(openings_for_mat) {
                    let alpha_pow_offset = match &schedule {
                        FriBatchingSchedule::Horner { alpha } => alpha.exp_u64(
                            u64::try_from(num_reduced[log_height])
                                .expect("reduced column count exceeds u64"),
                        ),
                        FriBatchingSchedule::Tensor { open_weights, .. } => {
                            open_weights[opening_idx_by_height[log_height]]
                        }
                    };
                    let reduced_openings: Challenge =
                        dot_product(col_weights.iter().copied(), openings.iter().copied());
                    mat_compressed
                        .par_iter()
                        .zip(reduced_opening_for_log_height.par_iter_mut())
                        .zip(inv_denoms.get(&point).unwrap().par_iter())
                        .for_each(|((&reduced_row, ro), &inv_denom)| {
                            *ro += alpha_pow_offset * (reduced_openings - reduced_row) * inv_denom;
                        });
                    num_reduced[log_height] += mat.width();
                    opening_idx_by_height[log_height] += 1;
                }
            }
        }

        let fri_input = reduced_openings.into_iter().rev().flatten().collect_vec();
        let folding: TwoAdicFriFoldingForMmcs<Goldilocks, HidingCudaMmcs> =
            TwoAdicFriFolding(PhantomData);
        let fri_proof = prover::prove_fri(
            &folding,
            &self.fri,
            fri_input,
            challenger,
            log_global_max_height,
            &commitment_data_with_opening_points,
            &self.mmcs,
        );

        (all_opened_values, fri_proof)
    }

    fn verify(
        &self,
        commitments_with_opening_points: Vec<
            CommitmentWithOpeningPoints<Challenge, Self::Commitment, Self::Domain>,
        >,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // Every committed polynomial in `v6-zk` has degree `< 2 * domain.size()`
        // and LDE height `2 * domain.size() * 2^log_blowup`. Double each domain
        // before cap validation and FRI verification.
        let mut doubled_commitments = Vec::with_capacity(commitments_with_opening_points.len());
        for (comm, mats) in commitments_with_opening_points {
            let mut doubled_mats = Vec::with_capacity(mats.len());
            for (domain, pts) in mats {
                let ext_log_size = domain.log_size() + 1;
                let ext_domain = TwoAdicMultiplicativeCoset::new(domain.shift(), ext_log_size)
                    .ok_or(FriError::GlobalMaxHeightTooLarge {
                        log_global_max_height: ext_log_size + self.fri.log_blowup,
                        two_adicity: Goldilocks::TWO_ADICITY,
                    })?;
                doubled_mats.push((ext_domain, pts));
            }
            doubled_commitments.push((comm, doubled_mats));
        }

        validate_fri_commitment_caps(&self.fri, proof, &doubled_commitments, &self.mmcs)?;
        for (_, round) in &doubled_commitments {
            for (_, mat) in round {
                for (_, point) in mat {
                    challenger.observe_algebra_slice(point);
                }
            }
        }

        let folding: TwoAdicFriFoldingForMmcs<Goldilocks, HidingCudaMmcs> =
            TwoAdicFriFolding(PhantomData);
        verify_fri_with_batching(
            &folding,
            &self.fri,
            proof,
            challenger,
            &doubled_commitments,
            &self.mmcs,
            self.batching,
        )?;

        Ok(())
    }

    fn build_periodic_lde_table(
        &self,
        periodic_cols: &[Vec<Goldilocks>],
        trace_domain: Self::Domain,
        quotient_domain: Self::Domain,
    ) -> PeriodicLdeTable<Goldilocks> {
        build_periodic_lde_table_two_adic::<Goldilocks, Dft>(
            &self.dft,
            periodic_cols,
            &trace_domain,
            &quotient_domain,
        )
    }
}
