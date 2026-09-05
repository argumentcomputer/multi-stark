//! The FRI PCS protocol over two-adic fields.
//!
//! The following implements a slight variant of the usual FRI protocol. As usual we start
//! with a polynomial `F(x)` of degree `n` given as evaluations over the coset `gH` with `|H| = 2^n`.
//!
//! Now consider the polynomial `G(x) = F(gx)`. Note that `G(x)` has the same degree as `F(x)` and
//! the evaluations of `F(x)` over `gH` are identical to the evaluations of `G(x)` over `H`.
//!
//! Hence we can reinterpret our vector of evaluations as evaluations of `G(x)` over `H` and apply
//! the standard FRI protocol to this evaluation vector. This makes it easier to apply FRI to a collection
//! of polynomials defined over different cosets as we don't need to keep track of the coset shifts. We
//! can just assume that every polynomial is defined over the subgroup of the relevant size.
//!
//! If we changed our domain construction (e.g., using multiple cosets), we would need to carefully reconsider these assumptions.
//!
//! The CPU PCS control flow in this module is derived from Plonky3's
//! `two_adic_pcs.rs` at revision 3152b14a89067c83775a8076cc262ffc48a1fd7c
//! (MIT/Apache-2.0). CUDA-resident commitments, openings, and FRI are maintained
//! here so their transcript order can be reviewed directly against that source.

use core::iter;
use core::marker::PhantomData;
use core::mem::size_of;
use std::borrow::Cow;
use std::vec;
use std::vec::Vec;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, OpenedValues, Pcs, PeriodicLdeTable};
use p3_dft::{Radix2DFTSmallBatch, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, PackedFieldExtension, PrimeCharacteristicRing, PrimeField64,
    TwoAdicField, batch_multiplicative_inverse, dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow};
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};
use tracing::{debug_span, instrument};

use super::mmcs::{CudaCommitMmcs, hash_host_only_height_groups};
use super::{CudaFriWorkspace, CudaLde, CudaMixedMerkleTree, CudaReducedOpening};
use p3_goldilocks::Goldilocks;
use p3_symmetric::MerkleCap;

fn goldilocks_quadratic_inverse_denominators(
    point: [Goldilocks; 2],
    coset: &[Goldilocks],
    extension_nonresidue: Goldilocks,
) -> Vec<[Goldilocks; 2]> {
    let [point_0, point_1] = point;
    let point_1_norm = extension_nonresidue * point_1 * point_1;
    let norms = coset
        .par_iter()
        .map(|&x| {
            let real = point_0 - x;
            real * real - point_1_norm
        })
        .collect::<Vec<_>>();
    let inverse_norms = batch_multiplicative_inverse(&norms);
    let inverse_point_1 = -point_1;
    coset
        .par_iter()
        .zip(inverse_norms.par_iter())
        .map(|(&x, &inverse_norm)| [(point_0 - x) * inverse_norm, inverse_point_1 * inverse_norm])
        .collect()
}

pub trait CudaPcsDft<T: TwoAdicField>: TwoAdicSubgroupDft<T> {
    fn prepare_coset_lde_constants(&self, height: usize, added_bits: usize, shift: T);

    fn coset_lde_batch_resident(
        &self,
        matrix: &RowMajorMatrix<T>,
        added_bits: usize,
        shift: T,
    ) -> CudaLde;
}

use p3_fri::{
    BatchMultiOpening, CommitPhaseMultiStep, FriParameters, FriProof, TwoAdicFriFolding,
    TwoAdicFriFoldingForMmcs, build_periodic_lde_table_two_adic, compute_log_arity_for_round,
    prover,
    verifier::{self, FriError},
};

struct CudaFriRound {
    codeword: CudaLde,
    tree: CudaMixedMerkleTree,
    arity: usize,
}

fn select_gpu_items(
    items: &[(usize, u128)],
    byte_budget: usize,
    max_item_bytes: usize,
) -> (Vec<bool>, usize, u128) {
    const PLACEMENT_GRANULARITY: usize = 8 << 20;

    #[derive(Clone, Copy)]
    struct Node {
        matrix: usize,
        previous: Option<usize>,
    }

    // Round every matrix up, making the discretized capacity conservative.
    // Eight-MiB units keep placement overhead small even on large GPUs while
    // wasting at most one unit per selected matrix.
    let capacity = byte_budget / PLACEMENT_GRANULARITY;
    let mut states = vec![None::<(u128, Option<usize>)>; capacity + 1];
    let mut nodes = Vec::<Node>::new();
    states[0] = Some((0, None));
    for (item, &(bytes, work)) in items.iter().enumerate() {
        if bytes > max_item_bytes {
            continue;
        }
        let units = bytes.div_ceil(PLACEMENT_GRANULARITY);
        if units > capacity {
            continue;
        }
        for used in (units..=capacity).rev() {
            let Some((previous_work, previous_node)) = states[used - units] else {
                continue;
            };
            let candidate_work = previous_work.saturating_add(work);
            if states[used].is_some_and(|(current_work, _)| current_work >= candidate_work) {
                continue;
            }
            let node = nodes.len();
            nodes.push(Node {
                matrix: item,
                previous: previous_node,
            });
            states[used] = Some((candidate_work, Some(node)));
        }
    }

    let (_, (gpu_work, mut node)) = states
        .into_iter()
        .enumerate()
        .filter_map(|(used, state)| state.map(|state| (used, state)))
        .max_by_key(|&(used, (work, _))| (work, core::cmp::Reverse(used)))
        .unwrap();
    let mut selected = vec![false; items.len()];
    while let Some(index) = node {
        let choice = nodes[index];
        selected[choice.matrix] = true;
        node = choice.previous;
    }
    let gpu_bytes = items
        .iter()
        .zip(&selected)
        .filter_map(|(&(bytes, _), &selected)| selected.then_some(bytes))
        .sum();
    (selected, gpu_bytes, gpu_work)
}

type CosetLdeJob<F> = (usize, (TwoAdicMultiplicativeCoset<F>, RowMajorMatrix<F>));

fn cpu_coset_lde_jobs<F>(
    jobs: Vec<CosetLdeJob<F>>,
    log_blowup: usize,
) -> Vec<(usize, RowMajorMatrix<F>)>
where
    F: TwoAdicField + PrimeField64,
    Radix2DitParallel<F>: TwoAdicSubgroupDft<F>,
{
    let cpu = Radix2DitParallel::<F>::default();
    jobs.into_par_iter()
        .map(|(index, (domain, evaluations))| {
            let shift = F::GENERATOR / domain.shift();
            let mut lde = cpu
                .coset_lde_batch(evaluations, log_blowup, shift)
                .bit_reverse_rows()
                .to_row_major_matrix();
            lde.values.par_iter_mut().for_each(|value| {
                *value = F::from_u64(value.as_canonical_u64());
            });
            (index, lde)
        })
        .collect()
}

trait CudaFriMmcs<Val, Challenge: Send + Sync + Clone>: Mmcs<Challenge> {
    fn commit_cuda_fri(
        &self,
        codeword: CudaLde,
        log_arity: usize,
    ) -> (Self::Commitment, CudaFriRound);
    fn open_cuda_fri_batch(
        &self,
        round: &CudaFriRound,
        rows: &[usize],
    ) -> (Vec<Vec<Challenge>>, Self::MultiProof);
}

impl<Challenge> CudaFriMmcs<Goldilocks, Challenge>
    for ExtensionMmcs<Goldilocks, Challenge, super::mmcs::CudaMmcs>
where
    Challenge: ExtensionField<Goldilocks>,
{
    fn commit_cuda_fri(
        &self,
        codeword: CudaLde,
        log_arity: usize,
    ) -> (Self::Commitment, CudaFriRound) {
        let arity = 1 << log_arity;
        let tree = CudaMixedMerkleTree::from_fri_codeword(&codeword, arity);
        let commitment = MerkleCap::new(vec![tree.root()]);
        (
            commitment,
            CudaFriRound {
                codeword,
                tree,
                arity,
            },
        )
    }

    fn open_cuda_fri_batch(
        &self,
        round: &CudaFriRound,
        rows: &[usize],
    ) -> (Vec<Vec<Challenge>>, Self::MultiProof) {
        let codeword_rows = rows
            .iter()
            .flat_map(|&row| (0..round.arity).map(move |column| row * round.arity + column))
            .collect_vec();
        let opened = round.codeword.rows(&codeword_rows);
        let opened_rows = opened
            .chunks_exact(round.arity)
            .map(|query_rows| {
                query_rows
                    .iter()
                    .map(|pair| {
                        Challenge::from_basis_coefficients_slice(pair)
                            .expect("quadratic extension row")
                    })
                    .collect()
            })
            .collect();
        let opening_proof = round.tree.open_pruned_siblings(rows);
        (opened_rows, opening_proof)
    }
}

/// A polynomial commitment scheme using FRI to generate opening proofs.
///
/// We commit to a polynomial `f` via its evaluation vectors over a coset
/// `gH` where `|H| >= 2 * deg(f)`. A value `f(z)` is opened by using a FRI
/// proof to show that the evaluations of `(f(x) - f(z))/(x - z)` over
/// `gH` are low degree.
#[derive(Clone, Debug)]
pub struct CudaTwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub(crate) dft: Dft,
    pub(crate) mmcs: InputMmcs,
    pub(crate) fri: FriParameters<FriMmcs>,
    _phantom: PhantomData<Val>,
}

fn prove_fri_cuda_resident<Val, Challenge, InputMmcs, FriMmcs, Challenger>(
    params: &FriParameters<FriMmcs>,
    mut inputs: Vec<CudaReducedOpening>,
    challenger: &mut Challenger,
    log_global_max_height: usize,
    prover_data_with_opening_points: &[ProverDataWithOpeningPoints<
        '_,
        Challenge,
        InputMmcs::ProverData<RowMajorMatrix<Val>>,
    >],
    input_mmcs: &InputMmcs,
    ext_w: Goldilocks,
) -> FriProof<Challenge, FriMmcs, Challenger::Witness, Vec<BatchMultiOpening<Val, InputMmcs>>>
where
    Val: TwoAdicField + PrimeField64 + 'static,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: CudaFriMmcs<Val, Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<FriMmcs::Commitment>,
{
    assert!(!inputs.is_empty());
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
    let to_pair = |value: Challenge| {
        let coefficients = value.as_basis_coefficients_slice();
        [
            Goldilocks::from_u64(coefficients[0].as_canonical_u64()),
            Goldilocks::from_u64(coefficients[1].as_canonical_u64()),
        ]
    };
    let from_pair = |pair: [Goldilocks; 2]| {
        Challenge::from_basis_coefficients_slice(&[
            Val::from_u64(pair[0].as_canonical_u64()),
            Val::from_u64(pair[1].as_canonical_u64()),
        ])
        .expect("quadratic extension element")
    };

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
        let arity = 1 << log_arity;
        let (commitment, round) = params.mmcs.commit_cuda_fri(codeword, log_arity);
        challenger.observe(commitment.clone());
        commits.push(commitment);
        commit_pow_witnesses.push(challenger.grind(params.commit_proof_of_work_bits));
        let beta: Challenge = challenger.sample_algebra_element();
        let mut beta_step = beta;
        let betas = (0..log_arity)
            .map(|_| {
                let result = to_pair(beta_step);
                beta_step = beta_step.square();
                result
            })
            .collect_vec();
        let folded_height = round.codeword.height() >> log_arity;
        let add_next = inputs
            .first()
            .is_some_and(|input| input.height() == folded_height);
        let beta_power = to_pair(beta.exp_power_of_2(log_arity));
        let g_inv = Goldilocks::from_u64(
            Val::two_adic_generator(log_height)
                .inverse()
                .as_canonical_u64(),
        );
        codeword = round.codeword.fri_fold(
            add_next.then(|| &inputs[0]),
            &betas,
            beta_power,
            g_inv,
            ext_w,
        );
        if add_next {
            inputs.remove(0);
        }
        rounds.push(round);
        log_arities.push(log_arity);
        debug_assert_eq!(arity, 1 << log_arity);
    }
    assert!(inputs.is_empty());

    let final_matrix = codeword.to_row_major_matrix();
    let mut final_values = final_matrix
        .values
        .as_chunks::<2>()
        .0
        .iter()
        .take(params.final_poly_len())
        .map(|pair| from_pair([pair[0], pair[1]]))
        .collect_vec();
    reverse_slice_index_bits(&mut final_values);
    let final_poly = Radix2DFTSmallBatch::default().idft_algebra(final_values);
    challenger.observe_algebra_slice(&final_poly);

    for &log_arity in &log_arities {
        challenger.observe(Val::from_usize(log_arity));
    }
    let query_pow_witness = challenger.grind(params.query_proof_of_work_bits);
    let query_indices = iter::repeat_with(|| challenger.sample_bits(log_global_max_height))
        .take(params.num_queries)
        .collect_vec();
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
    let mut current_indices = query_indices;
    let commit_phase_openings = rounds
        .iter()
        .zip(&log_arities)
        .map(|(round, &log_arity)| {
            let arity = 1 << log_arity;
            let positions = current_indices
                .iter()
                .map(|&index| index % arity)
                .collect_vec();
            let group_indices = current_indices
                .iter()
                .map(|&index| index >> log_arity)
                .collect_vec();
            let (opened_rows, opening_proof) =
                params.mmcs.open_cuda_fri_batch(round, &group_indices);
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

impl<Val, Dft, InputMmcs, FriMmcs> CudaTwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub const fn new(dft: Dft, mmcs: InputMmcs, fri: FriParameters<FriMmcs>) -> Self {
        Self {
            dft,
            mmcs,
            fri,
            _phantom: PhantomData,
        }
    }
}

/// The Prover Data associated to a commitment to a collection of matrices
/// and a list of points to open each matrix at.
pub type ProverDataWithOpeningPoints<'a, EF, ProverData> = (
    // The matrices and auxiliary prover data
    &'a ProverData,
    // for each matrix,
    Vec<
        // points to open
        Vec<EF>,
    >,
);

/// A joint commitment to a collection of matrices and their opening at
/// a collection of points.
pub type CommitmentWithOpeningPoints<Challenge, Commitment, Domain> = (
    Commitment,
    // For each matrix in the commitment:
    Vec<(
        // The domain of the matrix
        Domain,
        // A vector of (point, claimed_evaluation) pairs
        Vec<(Challenge, Vec<Challenge>)>,
    )>,
);

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for CudaTwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField + PrimeField64,
    Dft: TwoAdicSubgroupDft<Val> + CudaPcsDft<Val> + Sync,
    InputMmcs: Mmcs<Val, MultiProof: Sync, Error: Sync> + CudaCommitMmcs<Val>,
    FriMmcs: Mmcs<Challenge> + CudaFriMmcs<Val, Challenge>,
    Challenge: ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixCow<'a, Val>>;
    type Proof = FriProof<Challenge, FriMmcs, Val, Vec<BatchMultiOpening<Val, InputMmcs>>>;
    type Error = FriError<FriMmcs::Error, InputMmcs::Error>;
    const ZK: bool = false;

    /// Get the unique subgroup `H` of size `|H| = degree`.
    ///
    /// # Panics:
    /// This function will panic if `degree` is not a power of 2 or `degree > (1 << Val::TWO_ADICITY)`.
    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        TwoAdicMultiplicativeCoset::new(Val::ONE, log2_strict_usize(degree)).unwrap()
    }

    fn log_max_lde_height(&self) -> usize {
        Val::TWO_ADICITY
    }

    /// Commit to a collection of evaluation matrices.
    ///
    /// Each element of `evaluations` contains a coset `shift * H` and a matrix `mat` with `mat.height() = |H|`.
    /// Interpreting each column of `mat` as the evaluations of a polynomial `p_i(x)` over `shift * H`,
    /// this computes the evaluations of `p_i` over `gK` where `g` is the chosen generator of the multiplicative group
    /// of `Val` and `K` is the unique subgroup of order `|H| << self.fri.log_blowup`.
    ///
    /// This then outputs a Merkle commitment to these evaluations.
    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let evaluations: Vec<_> = evaluations.into_iter().collect();
        let source_cells = evaluations
            .iter()
            .map(|(_, matrix)| matrix.height() * matrix.width())
            .sum::<usize>();
        let dft = &self.dft;
        let log_blowup = self.fri.log_blowup;
        let (initial_free, total_bytes) =
            crate::cuda::device_memory_info(self.mmcs.cuda_device_id());
        let minimum_free = std::env::var("MULTI_STARK_CUDA_MIN_FREE_BYTES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(total_bytes / 4);
        let source_bytes = source_cells.saturating_mul(size_of::<Val>());
        let lde_bytes = source_bytes
            .checked_shl(u32::try_from(log_blowup).expect("LDE blowup exceeds u32"))
            .unwrap_or(usize::MAX);
        let release_traces_during_construction = source_bytes
            .saturating_add(lde_bytes)
            .saturating_add(minimum_free)
            > initial_free;
        if lde_bytes.saturating_add(minimum_free) > initial_free {
            let max_lde_height = evaluations
                .iter()
                .map(|(_, matrix)| matrix.height() << log_blowup)
                .max()
                .unwrap();
            // A hybrid tree stores all binary digest layers plus one injected
            // digest frontier. Reserve these explicitly before assigning any
            // LDE height group to device memory.
            let tree_workspace_bytes = max_lde_height.saturating_mul(96);
            let gpu_lde_budget = initial_free
                .saturating_sub(minimum_free)
                .saturating_sub(tree_workspace_bytes);
            let matrix_resources = evaluations
                .iter()
                .map(|(_, matrix)| {
                    let lde_cells = (matrix.height() * matrix.width()) << log_blowup;
                    let bytes = lde_cells.saturating_mul(size_of::<Val>());
                    let work = u128::try_from(lde_cells).expect("LDE cells exceed u128")
                        * (u128::from(matrix.height().trailing_zeros())
                            + u128::try_from(log_blowup).expect("log blowup exceeds u128")
                            + 1);
                    (bytes, work)
                })
                .collect_vec();
            let total_work = matrix_resources.iter().map(|(_, work)| *work).sum::<u128>();
            let mut height_groups = std::collections::BTreeMap::<usize, (usize, u128)>::new();
            let mut height_indices = std::collections::BTreeMap::<usize, Vec<usize>>::new();
            for (index, ((_, matrix), &(bytes, work))) in
                evaluations.iter().zip(&matrix_resources).enumerate()
            {
                let group = height_groups.entry(matrix.height()).or_default();
                group.0 = group.0.saturating_add(bytes);
                group.1 = group.1.saturating_add(work);
                height_indices
                    .entry(matrix.height())
                    .or_default()
                    .push(index);
            }
            // Split allocatable device capacity between persistent LDEs and
            // later phase workspace. This prevents stage one from filling VRAM
            // with values which lookup immediately has to copy back and evict.
            let durable_budget = gpu_lde_budget / 2;
            // A Merkle leaf combines every matrix at a given height. Keeping
            // height groups intact avoids streaming the CPU half of a split
            // group across PCIe merely to hash it beside resident matrices.
            // Matrix-level hybrid hashing remains available to later stages,
            // where the output partition is intrinsic rather than introduced
            // by this scheduler.
            let grouped_resources = height_groups.values().copied().collect_vec();
            let (selected_groups, durable_bytes, durable_work) =
                select_gpu_items(&grouped_resources, durable_budget, durable_budget);
            let durable_heights = height_groups
                .keys()
                .zip(selected_groups)
                .filter_map(|(&height, selected)| selected.then_some(height))
                .collect::<std::collections::BTreeSet<_>>();
            let durable_matrices = evaluations
                .iter()
                .map(|(_, matrix)| durable_heights.contains(&matrix.height()))
                .collect_vec();
            // A group too large to retain can still borrow the otherwise-idle
            // GPU lane during commitment. Its selected matrices are transformed
            // temporarily for hashing, then recomputed by the CPU for durable
            // host storage while CUDA moves on to the retained groups.
            let transient_reserve = max_lde_height.saturating_mul(32).saturating_add(64 << 20);
            let select_transient_plan = |transient_budget| {
                height_indices
                    .iter()
                    .filter(|(height, _)| !durable_heights.contains(height))
                    .filter_map(|(&height, indices)| {
                        let resources = indices
                            .iter()
                            .map(|&index| {
                                let (lde_bytes, work) = matrix_resources[index];
                                let trace_bytes = lde_bytes >> log_blowup;
                                (lde_bytes.saturating_add(trace_bytes), work)
                            })
                            .collect_vec();
                        let (selected, bytes, work) =
                            select_gpu_items(&resources, transient_budget, transient_budget);
                        let selected = indices
                            .iter()
                            .zip(selected)
                            .filter_map(|(&index, selected)| selected.then_some(index))
                            .collect_vec();
                        (!selected.is_empty()).then_some((height, selected, bytes, work))
                    })
                    .max_by_key(|(_, _, _, work)| *work)
            };

            // The CUDA constant cache is persistent and populated lazily. A
            // free-memory snapshot taken before preparing the selected DFTs
            // therefore overstates the capacity available to their LDEs. Keep
            // replanning until every constant required by the resulting plan
            // is resident and reflected in the memory snapshot. This is most
            // visible on very wide outer proofs, where a stale snapshot can
            // otherwise leave no room for the Merkle digest frontier.
            let mut transient_budget = initial_free.saturating_sub(transient_reserve);
            let mut transient_plan = select_transient_plan(transient_budget);
            let mut prepared_constants = std::collections::BTreeSet::new();
            loop {
                let transient_indices: std::collections::BTreeSet<usize> = transient_plan
                    .as_ref()
                    .map(|(_, indices, _, _)| indices.iter().copied().collect())
                    .unwrap_or_default();
                for (index, (domain, evals)) in evaluations.iter().enumerate() {
                    if !durable_matrices[index] && !transient_indices.contains(&index) {
                        continue;
                    }
                    let shift = Val::GENERATOR / domain.shift();
                    if prepared_constants.insert((evals.height(), shift.as_canonical_u64())) {
                        dft.prepare_coset_lde_constants(evals.height(), log_blowup, shift);
                    }
                }
                let (prepared_free, _) =
                    crate::cuda::device_memory_info(self.mmcs.cuda_device_id());
                transient_budget = prepared_free.saturating_sub(transient_reserve);
                let prepared_plan = select_transient_plan(transient_budget);
                if prepared_plan == transient_plan {
                    break;
                }
                transient_plan = prepared_plan;
            }
            let transient_height = transient_plan.as_ref().map(|(height, _, _, _)| *height);
            let transient_matrices: std::collections::BTreeSet<usize> = transient_plan
                .as_ref()
                .map(|(_, indices, _, _)| indices.iter().copied().collect())
                .unwrap_or_default();
            let durable_count = durable_matrices
                .iter()
                .filter(|&&selected| selected)
                .count();
            if crate::cuda::memory_diagnostics_enabled() {
                let gpu_percent = durable_work
                    .saturating_mul(100)
                    .checked_div(total_work)
                    .unwrap_or(0);
                eprintln!(
                    "[multi-stark/cuda] stage1 placement: durable={durable_count}/{} heights={}/{} gpu_work={durable_work}/{total_work} ({gpu_percent}%) durable_bytes={durable_bytes}/{durable_budget} device_budget={gpu_lde_budget}",
                    evaluations.len(),
                    durable_heights.len(),
                    height_groups.len(),
                );
                if let Some((height, indices, bytes, work)) = &transient_plan {
                    eprintln!(
                        "[multi-stark/cuda] stage1 transient group: height={height} matrices={indices:?} footprint={bytes}/{transient_budget} work={work}"
                    );
                }
                for (index, ((_, matrix), &(bytes, work))) in
                    evaluations.iter().zip(&matrix_resources).enumerate()
                {
                    if !durable_matrices[index] {
                        eprintln!(
                            "[multi-stark/cuda] stage1 host matrix {index}: height={} width={} lde_bytes={bytes} work={work}",
                            matrix.height(),
                            matrix.width(),
                        );
                    }
                }
            }
            if durable_count != 0 {
                let matrix_count = evaluations.len();
                let mut durable_jobs = Vec::with_capacity(durable_count);
                let mut transient_jobs = Vec::new();
                let mut transient_aux_jobs = Vec::new();
                let mut remaining_cpu_jobs = Vec::new();
                for (index, evaluation) in evaluations.into_iter().enumerate() {
                    if durable_matrices[index] {
                        durable_jobs.push((index, evaluation));
                    } else if transient_matrices.contains(&index) {
                        transient_jobs.push((index, evaluation));
                    } else if transient_height == Some(evaluation.1.height()) {
                        transient_aux_jobs.push((index, evaluation));
                    } else {
                        remaining_cpu_jobs.push((index, evaluation));
                    }
                }
                // All host LDE work shares one bounded pool. Keeping the
                // urgent same-height matrices, deferred materialization, and
                // remaining CPU matrices on separate Rayon pools can
                // oversubscribe a many-core host early, then strand cores once
                // only a large deferred matrix remains.
                // A large durable set must be evicted in later stages, which
                // naturally gives deferred CPU transforms more overlap. Keep
                // those transforms narrower so they do not starve the GPU's
                // host-memory traffic. When little remains resident, use more
                // cores to prevent a deferred transform becoming the quotient
                // critical path.
                let prioritize_deferred = durable_bytes < total_bytes / 4;
                let stage1_threads = crate::types::cuda_stage1_worker_count(prioritize_deferred);
                let stage1_pool =
                    std::sync::Arc::new(crate::types::cuda_host_pool("stage1-lde", stage1_threads));
                let cpu_started = std::time::Instant::now();
                let (aux_results, transient_results) = std::thread::scope(|scope| {
                    let aux_pool = std::sync::Arc::clone(&stage1_pool);
                    let cpu_task = scope.spawn(move || {
                        aux_pool.install(|| cpu_coset_lde_jobs(transient_aux_jobs, log_blowup))
                    });
                    let transient_started = std::time::Instant::now();
                    let transient_results = transient_jobs
                        .into_iter()
                        .map(|(index, (domain, evals))| {
                            let shift = Val::GENERATOR / domain.shift();
                            let lde = dft.coset_lde_batch_resident(&evals, log_blowup, shift);
                            (index, lde, (domain, evals))
                        })
                        .collect_vec();
                    if crate::cuda::memory_diagnostics_enabled() && !transient_results.is_empty() {
                        eprintln!(
                            "[multi-stark/cuda] stage1 transient GPU LDE: {:.3}s",
                            transient_started.elapsed().as_secs_f64()
                        );
                    }
                    (cpu_task.join().unwrap(), transient_results)
                });

                let mut host_matrices = (0..matrix_count).map(|_| None).collect_vec();
                for (index, lde) in aux_results {
                    host_matrices[index] = Some(lde);
                }
                let mut transient_ldes = (0..matrix_count).map(|_| None).collect_vec();
                let mut deferred_matrices = (0..matrix_count).map(|_| None).collect_vec();
                for (index, lde, (domain, evaluations)) in transient_results {
                    let dimensions = p3_matrix::Dimensions {
                        width: evaluations.width(),
                        height: evaluations.height() << log_blowup,
                    };
                    let pool = std::sync::Arc::clone(&stage1_pool);
                    let worker = std::thread::spawn(move || {
                        pool.install(|| {
                            cpu_coset_lde_jobs(vec![(index, (domain, evaluations))], log_blowup)
                                .pop()
                                .expect("deferred LDE worker returned no matrix")
                                .1
                        })
                    });
                    transient_ldes[index] = Some(lde);
                    deferred_matrices[index] = Some((dimensions, worker));
                }
                if crate::cuda::memory_diagnostics_enabled() {
                    let deferred_count = deferred_matrices
                        .iter()
                        .filter(|matrix| matrix.is_some())
                        .count();
                    eprintln!(
                        "[multi-stark/cuda] stage1 deferred CPU LDE workers: {deferred_count} matrices on {stage1_threads} threads"
                    );
                }
                let mut host_digest_groups = Vec::new();
                let mut prehashed_heights = std::collections::BTreeSet::new();

                let (cpu_results, durable_results) = std::thread::scope(|scope| {
                    let remaining_pool = std::sync::Arc::clone(&stage1_pool);
                    let cpu_task = scope.spawn(move || {
                        remaining_pool
                            .install(|| cpu_coset_lde_jobs(remaining_cpu_jobs, log_blowup))
                    });
                    if let Some(height) = transient_height {
                        let indices = &height_indices[&height];
                        let resident_refs = indices
                            .iter()
                            .map(|&index| transient_ldes[index].as_ref())
                            .collect_vec();
                        let host_refs = indices
                            .iter()
                            .map(|&index| {
                                transient_ldes[index]
                                    .is_none()
                                    .then(|| host_matrices[index].as_ref().unwrap())
                            })
                            .collect_vec();
                        let hash_started = std::time::Instant::now();
                        let digests = self
                            .mmcs
                            .hash_cuda_hybrid_height_group(&resident_refs, &host_refs);
                        let lde_height = height << log_blowup;
                        prehashed_heights.insert(lde_height);
                        host_digest_groups.push((lde_height, digests));
                        if crate::cuda::memory_diagnostics_enabled() {
                            eprintln!(
                                "[multi-stark/cuda] stage1 transient group hash: {:.3}s",
                                hash_started.elapsed().as_secs_f64()
                            );
                        }
                    }
                    // The transient LDEs have supplied their digest frontier.
                    // Drop them before admitting the durable resident set.
                    drop(transient_ldes);
                    let durable_started = std::time::Instant::now();
                    let durable_results = durable_jobs
                        .into_iter()
                        .map(|(index, (domain, evals))| {
                            let shift = Val::GENERATOR / domain.shift();
                            let lde = dft.coset_lde_batch_resident(&evals, log_blowup, shift);
                            (index, lde, evals)
                        })
                        .collect_vec();
                    if crate::cuda::memory_diagnostics_enabled() {
                        eprintln!(
                            "[multi-stark/cuda] stage1 durable GPU LDE: {:.3}s",
                            durable_started.elapsed().as_secs_f64()
                        );
                    }
                    (cpu_task.join().unwrap(), durable_results)
                });
                for (index, lde) in cpu_results {
                    host_matrices[index] = Some(lde);
                }
                if crate::cuda::memory_diagnostics_enabled() {
                    eprintln!(
                        "[multi-stark/cuda] stage1 CPU LDE: {:.3}s",
                        cpu_started.elapsed().as_secs_f64()
                    );
                }

                let mut resident = (0..matrix_count).map(|_| None).collect_vec();
                let mut retained_traces = (0..matrix_count).map(|_| None).collect_vec();
                for (index, lde, trace) in durable_results {
                    resident[index] = Some(lde);
                    retained_traces[index] = Some(trace);
                }
                let digest_started = std::time::Instant::now();
                let deferred_dimensions = deferred_matrices
                    .iter()
                    .map(|matrix| matrix.as_ref().map(|(dimensions, _)| *dimensions))
                    .collect_vec();
                host_digest_groups.extend(hash_host_only_height_groups(
                    &host_matrices,
                    &resident,
                    &deferred_dimensions,
                    &prehashed_heights,
                ));
                if crate::cuda::memory_diagnostics_enabled() {
                    eprintln!(
                        "[multi-stark/cuda] stage1 partitioned host digests: {:.3}s",
                        digest_started.elapsed().as_secs_f64()
                    );
                }
                let tree_started = std::time::Instant::now();
                let committed = self.mmcs.commit_cuda_hybrid(
                    resident,
                    host_matrices,
                    deferred_matrices,
                    retained_traces,
                    host_digest_groups,
                );
                if crate::cuda::memory_diagnostics_enabled() {
                    eprintln!(
                        "[multi-stark/cuda] stage1 hybrid tree: {:.3}s",
                        tree_started.elapsed().as_secs_f64()
                    );
                }
                return committed;
            }
            let cpu = Radix2DitParallel::<Val>::default();
            let ldes = tracing::info_span!("cuda/cpu_lde_fallback").in_scope(|| {
                evaluations
                    .into_par_iter()
                    .map(|(domain, evals)| {
                        let shift = Val::GENERATOR / domain.shift();
                        let mut lde = cpu
                            .coset_lde_batch(evals, log_blowup, shift)
                            .bit_reverse_rows()
                            .to_row_major_matrix();
                        lde.values.par_iter_mut().for_each(|value| {
                            *value = Val::from_u64(value.as_canonical_u64());
                        });
                        lde
                    })
                    .collect()
            });
            return tracing::info_span!("cuda/cpu_mmcs_fallback")
                .in_scope(|| self.mmcs.commit_cpu_storage(ldes));
        }
        if source_cells >= 10_000_000 {
            for (domain, evals) in &evaluations {
                dft.prepare_coset_lde_constants(
                    evals.height(),
                    log_blowup,
                    Val::GENERATOR / domain.shift(),
                );
            }
        }
        // Each Rayon worker uses its own per-thread CUDA stream. A bounded
        // admission window keeps pageable upload staging within the driver's
        // reliable concurrency range while retaining substantial overlap.
        const CUDA_LDE_WAVE: usize = 16;
        let mut ldes = Vec::with_capacity(evaluations.len());
        let wave_size = if release_traces_during_construction {
            1
        } else {
            CUDA_LDE_WAVE
        };
        for wave in evaluations.chunks(wave_size) {
            let transform =
                |(domain, evals): &(TwoAdicMultiplicativeCoset<Val>, RowMajorMatrix<Val>)| {
                    assert_eq!(domain.size(), evals.height());
                    let shift = Val::GENERATOR / domain.shift();
                    dft.coset_lde_batch_resident(evals, log_blowup, shift)
                };
            // Ix's inner proof contains hundreds of tiny matrices. Entering
            // CUDA concurrently for that low-volume batch costs more than it
            // saves and can exhaust driver-side pageable-copy workers.
            let wave_ldes: Vec<_> = if source_cells < 10_000_000 {
                wave.iter().map(transform).collect()
            } else {
                wave.par_iter().map(transform).collect()
            };
            if release_traces_during_construction {
                for lde in &wave_ldes {
                    // SAFETY: this wave has completed synchronously. The host
                    // matrices remain owned below and are retained for every
                    // later consumer of the original trace.
                    unsafe { lde.release_trace() };
                }
            }
            ldes.extend(wave_ldes);
        }
        // Preserve enough device headroom for lookup, quotient, and FRI
        // allocations. Spill the largest retained traces first, deriving the
        // policy from this device rather than a 96-GiB development machine.
        let (free_bytes, total_bytes) = crate::cuda::device_memory_info(self.mmcs.cuda_device_id());
        // A free-memory snapshot immediately after commitment construction is
        // not enough: lookup construction temporarily needs buffers
        // proportional to the original trace.  Proactively spill large traces
        // once this commitment itself is a meaningful fraction of VRAM.  The
        // ratios retain the policy that was validated on a 96-GiB device while
        // scaling it to smaller cards.
        let proactive_spill = source_bytes >= total_bytes / 12;
        let large_matrix = total_bytes / 120;
        let mut projected_free = free_bytes;
        let mut by_size = evaluations
            .iter()
            .enumerate()
            .map(|(index, (_, matrix))| {
                (
                    index,
                    matrix
                        .height()
                        .saturating_mul(matrix.width())
                        .saturating_mul(size_of::<Val>()),
                )
            })
            .collect_vec();
        by_size.sort_unstable_by_key(|&(_, bytes)| core::cmp::Reverse(bytes));
        let mut spilled = release_traces_during_construction;
        for (index, bytes) in by_size {
            if release_traces_during_construction {
                continue;
            }
            let needs_headroom = projected_free < minimum_free;
            let crowds_later_stages = proactive_spill && bytes >= large_matrix;
            if !needs_headroom && !crowds_later_stages {
                break;
            }
            // SAFETY: construction and commitment are synchronous; no CUDA
            // operation can observe this trace while it is released.
            unsafe { ldes[index].release_trace() };
            projected_free = projected_free.saturating_add(bytes);
            spilled = true;
        }
        // Commit to the bit-reversed LDEs. Once host traces have been retained,
        // keep the LDEs individually spillable as well: later quotient and FRI
        // admission may need to trade a resident LDE for its host materialization.
        // Small all-resident commitments retain the lower-overhead fast path.
        let (commitment, mut data) = if spilled {
            self.mmcs.commit_cuda_spillable(ldes)
        } else {
            self.mmcs.commit_cuda_resident(ldes)
        };
        if spilled {
            self.mmcs.retain_matrices(
                &mut data,
                evaluations.into_iter().map(|(_, matrix)| matrix).collect(),
            );
        }
        (commitment, data)
    }

    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        _num_chunks: usize,
    ) -> Vec<RowMajorMatrix<Val>> {
        evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                // coset_lde_batch converts from evaluations over `xH` to evaluations over `shift * x * K`.
                // Hence, letting `shift = g/x` the output will be evaluations over `gK` as desired.
                // When `x = g`, we could just use the standard LDE but currently this doesn't seem
                // to give a meaningful performance boost.
                let shift = Val::GENERATOR / domain.shift();
                // Compute the LDE with blowup factor fri.log_blowup.
                // We bit reverse as this is required by our implementation of the FRI protocol.
                self.dft
                    .coset_lde_batch(evals, self.fri.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect()
    }

    fn commit_ldes(&self, ldes: Vec<RowMajorMatrix<Val>>) -> (Self::Commitment, Self::ProverData) {
        let min_height = 1 << self.fri.log_blowup;
        for lde in &ldes {
            assert!(
                lde.height() >= min_height,
                "committed LDE height {} is smaller than the blowup factor {min_height}",
                lde.height()
            );
        }
        self.mmcs.commit_cuda_storage(ldes)
    }

    /// Given the evaluations on a domain `gH`, return the evaluations on a different domain `g'K`.
    ///
    /// Arguments:
    /// - `prover_data`: The prover data containing all committed evaluation matrices.
    /// - `idx`: The index of the matrix containing the evaluations we want. These evaluations
    ///   are assumed to be over the coset `gH` where `g = Val::GENERATOR`.
    /// - `domain`: The domain `g'K` on which to get evaluations on.
    ///
    /// When `g' = g` (i.e. `Val::GENERATOR`) and `K` is a subgroup of `H`, this is a simple
    /// truncation of the bit-reversed LDE. Otherwise, we recover the polynomial coefficients
    /// from the committed LDE and re-evaluate on the requested domain.
    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let lde = self.mmcs.get_matrices(prover_data)[idx];
        if domain.shift() == Val::GENERATOR && lde.height() >= domain.size() {
            return lde.split_rows(domain.size()).0.as_cow().bit_reverse_rows();
        }

        // The committed LDE contains bit-reversed evaluations over `gH`.
        // Un-bit-reverse, coset iDFT to recover coefficients, truncate to
        // the original polynomial degree, then coset DFT onto the target domain.
        let poly_height = lde.height() >> self.fri.log_blowup;
        let lde_mat = lde.as_view().bit_reverse_rows().to_row_major_matrix();
        let mut coeffs = self.dft.coset_idft_batch(lde_mat, Val::GENERATOR);
        let width = coeffs.width();
        coeffs.values.truncate(poly_height * width);
        coeffs.values.resize(domain.size() * width, Val::ZERO);
        let result = self
            .dft
            .coset_dft_batch(coeffs, domain.shift())
            .bit_reverse_rows()
            .to_row_major_matrix();
        let result_width = result.width();

        RowMajorMatrixCow::new(Cow::Owned(result.values), result_width).bit_reverse_rows()
    }

    /// Open a batch of matrices at a collection of points.
    ///
    /// Returns the opened values along with a proof.
    ///
    /// This function assumes that all matrices correspond to evaluations over the
    /// coset `gH` where `g = Val::GENERATOR` and `H` is a subgroup of appropriate size depending on the
    /// matrix.
    fn open(
        &self,
        // For each multi-matrix commitment,
        commitment_data_with_opening_points: Vec<(
            // The matrices and auxiliary prover data
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        /*

        A quick rundown of the optimizations in this function:
        We are trying to compute sum_i alpha^i * (p(X) - y)/(X - z),
        for each z an opening point, y = p(z). Each p(X) is given as evaluations in bit-reversed order
        in the columns of the matrices. y is computed by barycentric interpolation.
        X and p(X) are in the base field; alpha, y and z are in the extension.
        The primary goal is to minimize extension multiplications.

        - Instead of computing all alpha^i, we just compute alpha^i for i up to the largest width
        of a matrix, then multiply by an "alpha offset" when accumulating.
              a^0 x0 + a^1 x1 + a^2 x2 + a^3 x3 + ...
            = ( a^0 x0 + a^1 x1 ) + a^2 ( a^0 x2 + a^1 x3 ) + ...
            (see `alpha_pows`, `alpha_pow_offset`, `num_reduced`)

        - For each unique point z, we precompute 1/(X-z) for the largest subgroup opened at this point.
        Since we compute it in bit-reversed order, smaller subgroups can simply truncate the vector.
            (see `inv_denoms`)

        - Then, for each matrix (with columns p_i) and opening point z, we want:
            for each row (corresponding to subgroup element X):
                reduced[X] += alpha_offset * sum_i [ alpha^i * inv_denom[X] * (p_i[X] - y[i]) ]

            We can factor out inv_denom, and expand what's left:
                reduced[X] += alpha_offset * inv_denom[X] * sum_i [ alpha^i * p_i[X] - alpha^i * y[i] ]

            And separate the sum:
                reduced[X] += alpha_offset * inv_denom[X] * [ sum_i [ alpha^i * p_i[X] ] - sum_i [ alpha^i * y[i] ] ]

            And now the last sum doesn't depend on X, so we can precompute that for the matrix, too.
            So the hot loop (that depends on both X and i) is just:
                sum_i [ alpha^i * p_i[X] ]

            with alpha^i an extension, p_i[X] a base

        */

        // Keep CUDA commitments resident through barycentric interpolation
        // and construction of the reduced FRI codewords. Only the opened
        // values and final extension codewords cross back to the host.
        if commitment_data_with_opening_points
            .iter()
            .all(|(data, _)| self.mmcs.is_cuda_resident(data))
        {
            let resident_rounds = debug_span!("cuda prepare resident rounds").in_scope(|| {
                commitment_data_with_opening_points
                    .iter()
                    .map(|(data, points)| (self.mmcs.resident_or_upload(data), points))
                    .collect_vec()
            });
            let resident_max_height = resident_rounds
                .iter()
                .flat_map(|(ldes, _)| ldes.iter())
                .map(CudaLde::height)
                .max()
                .unwrap_or(0);
            let final_fri_height = self.fri.blowup() * self.fri.final_poly_len();
            if resident_max_height > 1024 && resident_max_height > final_fri_height {
                let _resident_guard = debug_span!("cuda resident fri").entered();
                let rounds = resident_rounds;
                let device_id = self.mmcs.cuda_device_id();
                assert_eq!(<Challenge as BasedVectorSpace<Val>>::DIMENSION, 2);
                let to_gold = |v: Val| Goldilocks::from_u64(v.as_canonical_u64());
                let to_pair = |v: Challenge| {
                    let c = <Challenge as BasedVectorSpace<Val>>::as_basis_coefficients_slice(&v);
                    [to_gold(c[0]), to_gold(c[1])]
                };
                let from_pair = |v: [Goldilocks; 2]| {
                    Challenge::from_basis_coefficients_slice(&[
                        Val::from_u64(v[0].as_canonical_u64()),
                        Val::from_u64(v[1].as_canonical_u64()),
                    ])
                    .unwrap()
                };
                let global_max_height = rounds
                    .iter()
                    .flat_map(|(ldes, _)| ldes.iter().map(CudaLde::height))
                    .max()
                    .unwrap();
                let global_max_width = rounds
                    .iter()
                    .flat_map(|(ldes, _)| ldes.iter().map(CudaLde::width))
                    .max()
                    .unwrap();
                let log_global_max_height = log2_strict_usize(global_max_height);
                let coset_domain =
                    TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_global_max_height).unwrap();
                let mut coset: Vec<Val> = coset_domain.iter().collect();
                reverse_slice_index_bits(&mut coset);
                let mut max_log: LinearMap<Challenge, usize> = LinearMap::new();
                for (ldes, points) in &rounds {
                    for (lde, ps) in ldes.iter().zip(points.iter()) {
                        for &z in ps {
                            if let Some(h) = max_log.get_mut(&z) {
                                *h = (*h).max(log2_strict_usize(lde.height()))
                            } else {
                                max_log.insert(z, log2_strict_usize(lde.height()));
                            }
                        }
                    }
                }
                let ext_x =
                    Challenge::from_basis_coefficients_slice(&[Val::ZERO, Val::ONE]).unwrap();
                let ext_w = to_pair(ext_x * ext_x)[0];
                assert_eq!(ext_w, Goldilocks::from_u64(7));
                let coset_gold: Vec<_> = coset.iter().copied().map(to_gold).collect();
                let mut inv_offsets = LinearMap::new();
                let mut inverse_points = Vec::new();
                let mut inverse_counts = Vec::new();
                let mut inverse_count = 0usize;
                for (point, lh) in max_log {
                    inv_offsets.insert(point, inverse_count);
                    inverse_points.push(to_pair(point));
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
                );
                let mut interpolation_tasks = Vec::new();
                let mut output_count = 0usize;
                let layouts = rounds
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
                                        let shift_pow = Val::GENERATOR.exp_power_of_2(lh);
                                        let scale = (point.exp_power_of_2(lh) - shift_pow)
                                            * (Val::from_usize(h) * shift_pow).inverse();
                                        interpolation_tasks.push(lde.interpolation_task(
                                            h,
                                            *inv_offsets.get(&point).unwrap(),
                                            offset,
                                            to_pair(scale),
                                        ));
                                        (offset, lde.width())
                                    })
                                    .collect_vec()
                            })
                            .collect_vec()
                    })
                    .collect_vec();
                let interpolated = debug_span!("cuda interpolate openings")
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
                                        interpolated[offset..offset + width]
                                            .iter()
                                            .copied()
                                            .map(from_pair)
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
                let alpha: Challenge = challenger.sample_algebra_element();
                let alpha_powers: Vec<_> = alpha.powers().take(global_max_width).collect();
                let alpha_pairs: Vec<_> = alpha_powers.iter().copied().map(to_pair).collect();
                let mut num_reduced = [0usize; 33];
                let mut reduced: [Option<CudaReducedOpening>; 33] = core::array::from_fn(|_| None);
                let mut reduction_tasks = Vec::new();
                for ((ldes, points), openings_round) in rounds.iter().zip(all_opened_values.iter())
                {
                    for ((lde, ps), openings) in
                        ldes.iter().zip(points.iter()).zip(openings_round.iter())
                    {
                        let lh = log2_strict_usize(lde.height());
                        let target = reduced[lh].get_or_insert_with(|| {
                            CudaReducedOpening::new(device_id, lde.height())
                        });
                        for (&point, ys) in ps.iter().zip(openings.iter()) {
                            let reduced_y =
                                dot_product(alpha_powers.iter().copied(), ys.iter().copied());
                            let offset = alpha.exp_u64(num_reduced[lh] as u64);
                            reduction_tasks.push(target.reduction_task(
                                lde,
                                *inv_offsets.get(&point).unwrap(),
                                to_pair(reduced_y),
                                to_pair(offset),
                            ));
                            num_reduced[lh] += lde.width();
                        }
                    }
                }
                debug_span!("cuda reduce openings")
                    .in_scope(|| workspace.reduce(&reduction_tasks, &alpha_pairs, ext_w));
                let fri_input = reduced.into_iter().rev().flatten().collect_vec();
                let fri_proof = debug_span!("cuda prove fri").in_scope(|| {
                    prove_fri_cuda_resident(
                        &self.fri,
                        fri_input,
                        challenger,
                        log_global_max_height,
                        &commitment_data_with_opening_points,
                        &self.mmcs,
                        ext_w,
                    )
                });
                return (all_opened_values, fri_proof);
            }
        }

        // A commitment which did not fit in VRAM is stored by the CPU MMCS,
        // but that does not require the entire opening to fall back to the
        // CPU. Upload one such matrix at a time for interpolation and
        // reduction. Resident commitments stay in place, and temporary
        // uploads are dropped before the next matrix, bounding additional
        // device memory by the largest CPU-backed matrix instead of the sum
        // of all commitments.
        let dimensions = commitment_data_with_opening_points
            .iter()
            .map(|(data, points)| {
                let dimensions = self.mmcs.matrix_dimensions(data);
                assert_eq!(
                    dimensions.len(),
                    points.len(),
                    "each matrix should have a corresponding set of evaluation points"
                );
                dimensions
            })
            .collect_vec();
        let opening_points = commitment_data_with_opening_points
            .iter()
            .map(|(_, points)| points)
            .collect_vec();
        let (cuda_max_height, cuda_max_width) = dimensions
            .iter()
            .flatten()
            .map(|dims| (dims.height, dims.width))
            .reduce(|(hmax, wmax), (height, width)| (hmax.max(height), wmax.max(width)))
            .expect("No Matrices Supplied?");
        let final_fri_height = self.fri.blowup() * self.fri.final_poly_len();
        if cuda_max_height > 1024 && cuda_max_height > final_fri_height {
            let _resident_guard = debug_span!("cuda streamed fri").entered();
            let phase_started = std::time::Instant::now();
            let device_id = self.mmcs.cuda_device_id();
            assert_eq!(<Challenge as BasedVectorSpace<Val>>::DIMENSION, 2);
            let to_gold = |value: Val| Goldilocks::from_u64(value.as_canonical_u64());
            let to_pair = |value: Challenge| {
                let coefficients =
                    <Challenge as BasedVectorSpace<Val>>::as_basis_coefficients_slice(&value);
                [to_gold(coefficients[0]), to_gold(coefficients[1])]
            };
            let from_pair = |value: [Goldilocks; 2]| {
                Challenge::from_basis_coefficients_slice(&[
                    Val::from_u64(value[0].as_canonical_u64()),
                    Val::from_u64(value[1].as_canonical_u64()),
                ])
                .expect("quadratic extension element")
            };
            let log_global_max_height = log2_strict_usize(cuda_max_height);
            let coset_domain =
                TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_global_max_height).unwrap();
            let mut coset: Vec<Val> = coset_domain.iter().collect();
            reverse_slice_index_bits(&mut coset);

            let mut max_log: LinearMap<Challenge, usize> = LinearMap::new();
            for ((_, points), round_dimensions) in commitment_data_with_opening_points
                .iter()
                .zip(dimensions.iter())
            {
                for (points, dims) in points.iter().zip(round_dimensions) {
                    for &point in points {
                        let log_height = log2_strict_usize(dims.height);
                        if let Some(maximum) = max_log.get_mut(&point) {
                            *maximum = (*maximum).max(log_height);
                        } else {
                            max_log.insert(point, log_height);
                        }
                    }
                }
            }

            let ext_x = Challenge::from_basis_coefficients_slice(&[Val::ZERO, Val::ONE])
                .expect("quadratic extension generator");
            let ext_w = to_pair(ext_x * ext_x)[0];
            assert_eq!(ext_w, Goldilocks::from_u64(7));
            let coset_gold: Vec<_> = coset.iter().copied().map(to_gold).collect();
            let mut inv_offsets = LinearMap::new();
            let mut inverse_points = Vec::new();
            let mut inverse_counts = Vec::new();
            let mut inverse_count = 0usize;
            let point_logs = max_log.into_iter().collect_vec();
            let (_, total_device_bytes) = crate::cuda::device_memory_info(device_id);
            let inverse_elements = point_logs
                .iter()
                .map(|(_, log_height)| 1usize << log_height)
                .sum::<usize>();
            let max_inverse_elements = point_logs
                .iter()
                .map(|(_, log_height)| 1usize << log_height)
                .max()
                .unwrap_or(0);
            let reduced_elements = dimensions
                .iter()
                .flatten()
                .map(|dims| dims.height)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .sum::<usize>();
            let fri_workspace_bytes = inverse_elements
                .saturating_mul(2 * size_of::<Val>())
                .saturating_add(cuda_max_height.saturating_mul(size_of::<Val>()))
                .saturating_add(max_inverse_elements.saturating_mul(2 * size_of::<Val>()))
                .saturating_add(reduced_elements.saturating_mul(2 * size_of::<Val>()))
                .saturating_add(total_device_bytes / 64);
            let admission_started = std::time::Instant::now();
            let admission_data = commitment_data_with_opening_points
                .iter()
                .map(|entry| entry.0)
                .collect_vec();
            self.mmcs
                .ensure_device_headroom_batch(&admission_data, fri_workspace_bytes);
            if crate::cuda::memory_diagnostics_enabled() {
                eprintln!(
                    "[multi-stark/cuda] FRI admission: {:.3}s",
                    admission_started.elapsed().as_secs_f64()
                );
            }
            for &(point, log_height) in &point_logs {
                inv_offsets.insert(point, inverse_count);
                inverse_points.push(to_pair(point));
                let count = 1usize << log_height;
                inverse_counts.push(count);
                inverse_count += count;
            }
            let cpu_round_matrices = commitment_data_with_opening_points
                .iter()
                .map(|(data, _)| self.mmcs.cpu_matrices(data))
                .collect_vec();
            let mut cpu_max_log: LinearMap<Challenge, usize> = LinearMap::new();
            for ((matrices, &points), round_dimensions) in cpu_round_matrices
                .iter()
                .zip(opening_points.iter())
                .zip(dimensions.iter())
            {
                for ((matrix, points), dims) in
                    matrices.iter().zip(points.iter()).zip(round_dimensions)
                {
                    if matrix.is_none() {
                        continue;
                    }
                    let log_height = log2_strict_usize(dims.height);
                    for &point in points {
                        if let Some(maximum) = cpu_max_log.get_mut(&point) {
                            *maximum = (*maximum).max(log_height);
                        } else {
                            cpu_max_log.insert(point, log_height);
                        }
                    }
                }
            }
            let cpu_point_logs = cpu_max_log.into_iter().collect_vec();
            let fri_log_blowup = self.fri.log_blowup;
            let denominators_started = std::time::Instant::now();
            let ((inv_denoms, cpu_denominator_seconds), mut workspace, gpu_denominator_seconds) =
                std::thread::scope(|scope| {
                    let cpu_denominators = scope.spawn(|| {
                        let started = std::time::Instant::now();
                        let denominators: LinearMap<Challenge, Vec<Challenge>> = cpu_point_logs
                            .iter()
                            .map(|&(point, log_height)| {
                                let count = 1 << log_height;
                                let coset = coset_gold[..count].as_ref();
                                let inverses = goldilocks_quadratic_inverse_denominators(
                                    to_pair(point),
                                    coset,
                                    ext_w,
                                )
                                .into_par_iter()
                                .map(from_pair)
                                .collect();
                                (point, inverses)
                            })
                            .collect();
                        (denominators, started.elapsed().as_secs_f64())
                    });
                    let gpu_started = std::time::Instant::now();
                    let workspace = CudaFriWorkspace::new(
                        device_id,
                        &inverse_points,
                        &inverse_counts,
                        &coset_gold,
                        ext_w,
                    );
                    let gpu_seconds = gpu_started.elapsed().as_secs_f64();
                    let cpu_denominators = cpu_denominators.join().unwrap();
                    (cpu_denominators, workspace, gpu_seconds)
                });
            let adjusted_weights: LinearMap<Challenge, Vec<Challenge>> = inv_denoms
                .iter()
                .map(|(point, denoms)| (*point, compute_adjusted_weights(*point, denoms)))
                .collect();
            if crate::cuda::memory_diagnostics_enabled() {
                eprintln!(
                    "[multi-stark/cuda] FRI denominators: {:.3}s (CPU {:.3}s, GPU {:.3}s)",
                    denominators_started.elapsed().as_secs_f64(),
                    cpu_denominator_seconds,
                    gpu_denominator_seconds
                );
            }

            let interpolation_started = std::time::Instant::now();
            let ((cpu_opened, cpu_interpolation_seconds), gpu_opened, gpu_interpolation_seconds) =
                std::thread::scope(|scope| {
                    let cpu_opened = scope.spawn(|| {
                        let started = std::time::Instant::now();
                        let opened = cpu_round_matrices
                            .iter()
                            .zip(opening_points.iter())
                            .map(|(matrices, &points)| {
                                matrices
                                    .par_iter()
                                    .copied()
                                    .zip(points.par_iter())
                                    .map(|(matrix, points)| {
                                        let matrix = matrix?;
                                        let polynomial_height = matrix.height() >> fri_log_blowup;
                                        assert!(polynomial_height.is_power_of_two());
                                        let (low_coset, _) = matrix.split_rows(polynomial_height);
                                        Some(
                                            points
                                                .iter()
                                                .map(|&point| {
                                                    low_coset.interpolate_coset_with_precomputation(
                                                        Val::GENERATOR,
                                                        point,
                                                        &adjusted_weights.get(&point).unwrap()
                                                            [..polynomial_height],
                                                    )
                                                })
                                                .collect_vec(),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect_vec();
                        (opened, started.elapsed().as_secs_f64())
                    });
                    let gpu_started = std::time::Instant::now();
                    let gpu_opened = commitment_data_with_opening_points
                        .iter()
                        .zip(dimensions.iter())
                        .map(|((data, points), round_dimensions)| {
                            points
                                .iter()
                                .zip(round_dimensions)
                                .enumerate()
                                .map(|(matrix_index, (points, dims))| {
                                    if !self.mmcs.is_matrix_cuda_resident(data, matrix_index) {
                                        return None;
                                    }
                                    if points.is_empty() {
                                        return Some(Vec::new());
                                    }
                                    Some(self.mmcs.with_resident_matrix(
                                        data,
                                        matrix_index,
                                        |lde| {
                                            assert_eq!(lde.height(), dims.height);
                                            assert_eq!(lde.width(), dims.width);
                                            let polynomial_height =
                                                lde.height() >> self.fri.log_blowup;
                                            assert!(polynomial_height.is_power_of_two());
                                            let log_polynomial_height =
                                                log2_strict_usize(polynomial_height);
                                            let shift_power = Val::GENERATOR
                                                .exp_power_of_2(log_polynomial_height);
                                            let tasks = points
                                                .iter()
                                                .enumerate()
                                                .map(|(point_index, &point)| {
                                                    let scale = (point
                                                        .exp_power_of_2(log_polynomial_height)
                                                        - shift_power)
                                                        * (Val::from_usize(polynomial_height)
                                                            * shift_power)
                                                            .inverse();
                                                    lde.interpolation_task(
                                                        polynomial_height,
                                                        *inv_offsets.get(&point).unwrap(),
                                                        point_index * lde.width(),
                                                        to_pair(scale),
                                                    )
                                                })
                                                .collect_vec();
                                            workspace
                                                .interpolate(
                                                    &tasks,
                                                    points.len() * lde.width(),
                                                    ext_w,
                                                )
                                                .chunks_exact(lde.width())
                                                .map(|values| {
                                                    values
                                                        .iter()
                                                        .copied()
                                                        .map(from_pair)
                                                        .collect_vec()
                                                })
                                                .collect_vec()
                                        },
                                    ))
                                })
                                .collect_vec()
                        })
                        .collect_vec();
                    let gpu_seconds = gpu_started.elapsed().as_secs_f64();
                    let cpu_opened = cpu_opened.join().unwrap();
                    (cpu_opened, gpu_opened, gpu_seconds)
                });
            let all_opened_values = cpu_opened
                .into_iter()
                .zip(gpu_opened)
                .map(|(cpu_round, gpu_round)| {
                    cpu_round
                        .into_iter()
                        .zip(gpu_round)
                        .map(|(cpu, gpu)| cpu.or(gpu).expect("matrix has no opening backend"))
                        .collect_vec()
                })
                .collect_vec();
            if crate::cuda::memory_diagnostics_enabled() {
                eprintln!(
                    "[multi-stark/cuda] FRI interpolation: {:.3}s (CPU {:.3}s, GPU {:.3}s)",
                    interpolation_started.elapsed().as_secs_f64(),
                    cpu_interpolation_seconds,
                    gpu_interpolation_seconds
                );
            }

            for round in &all_opened_values {
                for matrix in round {
                    for values in matrix {
                        challenger.observe_algebra_slice(values);
                    }
                }
            }
            let alpha: Challenge = challenger.sample_algebra_element();
            let alpha_powers: Vec<_> = alpha.powers().take(cuda_max_width).collect();
            let alpha_pairs: Vec<_> = alpha_powers.iter().copied().map(to_pair).collect();
            let packed_alpha_powers =
                Challenge::ExtensionPacking::packed_ext_powers_capped(alpha, cuda_max_width)
                    .collect_vec();
            let mut num_reduced = [0usize; 33];
            let reduction_offsets = commitment_data_with_opening_points
                .iter()
                .zip(dimensions.iter())
                .map(|((_, points), round_dimensions)| {
                    points
                        .iter()
                        .zip(round_dimensions)
                        .map(|(points, dims)| {
                            let log_height = log2_strict_usize(dims.height);
                            points
                                .iter()
                                .map(|_| {
                                    let offset = num_reduced[log_height];
                                    num_reduced[log_height] += dims.width;
                                    offset
                                })
                                .collect_vec()
                        })
                        .collect_vec()
                })
                .collect_vec();

            let reduction_started = std::time::Instant::now();
            let ((cpu_reduced, cpu_reduction_seconds), mut gpu_reduced, gpu_reduction_seconds) =
                std::thread::scope(|scope| {
                    let cpu_reduced = scope.spawn(|| {
                        let started = std::time::Instant::now();
                        let mut reduced: [Option<Vec<Challenge>>; 33] =
                            core::array::from_fn(|_| None);
                        for (
                            (((matrices, &points), round_dimensions), openings_round),
                            offsets_round,
                        ) in cpu_round_matrices
                            .iter()
                            .zip(opening_points.iter())
                            .zip(dimensions.iter())
                            .zip(all_opened_values.iter())
                            .zip(reduction_offsets.iter())
                        {
                            for ((((matrix, points), dims), openings), offsets) in matrices
                                .iter()
                                .copied()
                                .zip(points)
                                .zip(round_dimensions)
                                .zip(openings_round)
                                .zip(offsets_round)
                            {
                                let Some(matrix) = matrix else {
                                    continue;
                                };
                                if points.is_empty() {
                                    continue;
                                }
                                let log_height = log2_strict_usize(dims.height);
                                let target = reduced[log_height]
                                    .get_or_insert_with(|| vec![Challenge::ZERO; dims.height]);
                                let compressed = matrix
                                    .rowwise_packed_dot_product::<Challenge>(&packed_alpha_powers)
                                    .collect::<Vec<_>>();
                                for ((&point, values), &offset) in
                                    points.iter().zip(openings).zip(offsets)
                                {
                                    let reduced_y: Challenge = dot_product(
                                        alpha_powers.iter().copied(),
                                        values.iter().copied(),
                                    );
                                    let alpha_offset = alpha.exp_u64(offset as u64);
                                    compressed
                                        .par_iter()
                                        .zip(target.par_iter_mut())
                                        .zip(inv_denoms.get(&point).unwrap().par_iter())
                                        .for_each(|((&row, output), &inv_denom)| {
                                            *output += alpha_offset * (reduced_y - row) * inv_denom;
                                        });
                                }
                            }
                        }
                        (reduced, started.elapsed().as_secs_f64())
                    });
                    let gpu_started = std::time::Instant::now();
                    let mut reduced: [Option<CudaReducedOpening>; 33] =
                        core::array::from_fn(|_| None);
                    let mut reduction_tasks = Vec::new();
                    for ((((data, points), round_dimensions), openings_round), offsets_round) in
                        commitment_data_with_opening_points
                            .iter()
                            .zip(dimensions.iter())
                            .zip(all_opened_values.iter())
                            .zip(reduction_offsets.iter())
                    {
                        for (matrix_index, (((points, dims), openings), offsets)) in points
                            .iter()
                            .zip(round_dimensions)
                            .zip(openings_round)
                            .zip(offsets_round)
                            .enumerate()
                        {
                            if !self.mmcs.is_matrix_cuda_resident(data, matrix_index) {
                                continue;
                            }
                            if points.is_empty() {
                                continue;
                            }
                            let log_height = log2_strict_usize(dims.height);
                            let target = reduced[log_height].get_or_insert_with(|| {
                                CudaReducedOpening::new(device_id, dims.height)
                            });
                            self.mmcs.with_resident_matrix(data, matrix_index, |lde| {
                                reduction_tasks.extend(
                                    points.iter().zip(openings).zip(offsets).map(
                                        |((&point, values), &offset)| {
                                            let reduced_y = dot_product(
                                                alpha_powers.iter().copied(),
                                                values.iter().copied(),
                                            );
                                            target.reduction_task(
                                                lde,
                                                *inv_offsets.get(&point).unwrap(),
                                                to_pair(reduced_y),
                                                to_pair(alpha.exp_u64(offset as u64)),
                                            )
                                        },
                                    ),
                                );
                            });
                        }
                    }
                    if !reduction_tasks.is_empty() {
                        workspace.reduce(&reduction_tasks, &alpha_pairs, ext_w);
                    }
                    let gpu_seconds = gpu_started.elapsed().as_secs_f64();
                    let cpu_reduced = cpu_reduced.join().unwrap();
                    (cpu_reduced, reduced, gpu_seconds)
                });
            drop(workspace);

            for (log_height, cpu_values) in cpu_reduced.into_iter().enumerate() {
                if let Some(cpu_values) = cpu_values {
                    let values = cpu_values.into_iter().map(to_pair).collect_vec();
                    gpu_reduced[log_height]
                        .get_or_insert_with(|| CudaReducedOpening::new(device_id, values.len()))
                        .add_host(&values);
                }
            }
            if crate::cuda::memory_diagnostics_enabled() {
                eprintln!(
                    "[multi-stark/cuda] FRI reduction: {:.3}s (CPU {:.3}s, GPU {:.3}s)",
                    reduction_started.elapsed().as_secs_f64(),
                    cpu_reduction_seconds,
                    gpu_reduction_seconds
                );
            }
            let fri_input = gpu_reduced.into_iter().rev().flatten().collect_vec();
            let folding_started = std::time::Instant::now();
            let fri_proof = debug_span!("cuda prove streamed fri").in_scope(|| {
                prove_fri_cuda_resident(
                    &self.fri,
                    fri_input,
                    challenger,
                    log_global_max_height,
                    &commitment_data_with_opening_points,
                    &self.mmcs,
                    ext_w,
                )
            });
            if crate::cuda::memory_diagnostics_enabled() {
                eprintln!(
                    "[multi-stark/cuda] FRI folding and queries: {:.3}s (streamed total {:.3}s)",
                    folding_started.elapsed().as_secs_f64(),
                    phase_started.elapsed().as_secs_f64()
                );
            }
            return (all_opened_values, fri_proof);
        }

        // Contained in each `Self::ProverData` is a list of matrices which have been committed to.
        // We extract those matrices to be able to refer to them directly.
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

        // Find the maximum height and the maximum width of matrices in the batch.
        // These do not need to correspond to the same matrix.
        let (global_max_height, global_max_width) = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats.iter().map(|m| (m.height(), m.width())))
            .reduce(|(hmax, wmax), (h, w)| (hmax.max(h), wmax.max(w)))
            .expect("No Matrices Supplied?");
        let log_global_max_height = log2_strict_usize(global_max_height);

        // Get all values of the coset `gH` for the largest necessary subgroup `H`.
        // We also bit reverse which means that coset has the nice property that
        // `coset[..2^i]` contains the values of `gK` for `|K| = 2^i`.
        let coset = {
            let coset =
                TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_global_max_height).unwrap();
            let mut coset_points = coset.iter().collect();
            reverse_slice_index_bits(&mut coset_points);
            coset_points
        };

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(z - X) for the largest subgroup (in bitrev order).
        let inv_denoms = compute_inverse_denominators(&mats_and_points, &coset);

        // Convert the inverse denominators into the adjusted barycentric weights expected by
        // the matrix interpolation API. Reuse them across every matrix opened at each point.
        let adjusted_weights: LinearMap<Challenge, Vec<Challenge>> = inv_denoms
            .iter()
            .map(|(point, denoms)| (*point, compute_adjusted_weights(*point, denoms)))
            .collect();

        // Evaluate coset representations and write openings to the challenger
        let all_opened_values = mats_and_points
            .iter()
            .map(|(mats, points)| {
                // For each collection of matrices
                izip!(mats.iter(), points.iter())
                    .map(|(mat, points_for_mat)| {
                        // TODO: This assumes that every input matrix has a blowup of at least self.fri.log_blowup.
                        // If the blow_up factor is smaller than self.fri.log_blowup, this will lead to errors.
                        // If it is bigger, we shouldn't get any errors but it will be slightly slower.
                        // Ideally, polynomials could be passed in with their blow_up factors known.

                        // The point of this correction is that each column of the matrix corresponds to a low degree polynomial.
                        // Hence we can save time by restricting the height of the matrix to be the minimal height which
                        // uniquely identifies the polynomial.
                        let h = mat.height() >> self.fri.log_blowup;

                        // `subgroup` and `mat` are both in bit-reversed order, so we can truncate.
                        let (low_coset, _) = mat.split_rows(h);

                        points_for_mat
                            .iter()
                            .map(|&point| {
                                let _guard =
                                    debug_span!("evaluate matrix", dims = %mat.dimensions())
                                        .entered();

                                // Use Barycentric interpolation to evaluate each column of the matrix at the given point.
                                let ys = debug_span!(
                                    "compute opened values with Lagrange interpolation"
                                )
                                .in_scope(|| {
                                    let adjusted = &adjusted_weights.get(&point).unwrap()[..h];
                                    low_coset.interpolate_coset_with_precomputation(
                                        Val::GENERATOR,
                                        point,
                                        adjusted,
                                    )
                                });

                                challenger.observe_algebra_slice(&ys);
                                ys
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();

        // Batch combination challenge

        // Soundness Error:
        // See the discussion in the doc comment of [`prove_fri`]. Essentially, the soundness error
        // for this sample is tightly tied to the soundness error of the FRI protocol.
        // Roughly speaking, at a minimum is it k/|EF| where `k` is the sum of, for each function, the number of
        // points it needs to be opened at. This comes from the fact that we are taking a large linear combination
        // of `(f(zeta) - f(x))/(zeta - x)` for each function `f` and all of `f`'s opening points.
        // In our setup, k is two times the trace width plus the number of quotient polynomials.
        let alpha: Challenge = challenger.sample_algebra_element();

        // We precompute powers of alpha as we need the same powers for each matrix.
        // We compute both a vector of unpacked powers and a vector of packed powers.
        // TODO: It should be possible to refactor this to only use the packed powers but
        // this is not a bottleneck so is not a priority.
        let packed_alpha_powers =
            Challenge::ExtensionPacking::packed_ext_powers_capped(alpha, global_max_width)
                .collect_vec();
        let alpha_powers =
            Challenge::ExtensionPacking::to_ext_iter(packed_alpha_powers.iter().copied())
                .collect_vec();

        // Now that we have sent the openings to the verifier, it remains to prove
        // that those openings are correct.

        // Given a low degree polynomial `f(x)` with claimed evaluation `f(zeta)`, we can check
        // that `f(zeta)` is correct by doing a low degree test on `(f(zeta) - f(x))/(zeta - x)`.
        // We will use `alpha` to batch together both different claimed openings `zeta` and
        // different polynomials `f` whose evaluation vectors have the same height.

        // TODO: If we allow different polynomials to have different blow_up factors
        // we may need to revisit this and to ensure it is safe to batch them together.

        // num_reduced records the number of (function, opening point) pairs for each `log_height`.
        // TODO: This should really be `[0; Val::TWO_ADICITY]` but that runs into issues with generics.
        let mut num_reduced = [0; 33];

        // For each `log_height` from 2^1 -> 2^32, reduced_openings will contain either `None`
        // if there are no matrices of that height, or `Some(vec)` where `vec` is equal to
        // a weighted sum of `(f(zeta) - f(x))/(zeta - x)` over all `f`'s of that height and
        // for each `f`, all opening points `zeta`. The sum is weighted by powers of the challenge alpha.
        let mut reduced_openings: [_; 33] = core::array::from_fn(|_| None);

        for ((mats, points), openings_for_round) in
            mats_and_points.iter().zip(all_opened_values.iter())
        {
            for (mat, points_for_mat, openings_for_mat) in
                izip!(mats.iter(), points.iter(), openings_for_round.iter())
            {
                let _guard =
                    debug_span!("reduce matrix quotient", dims = %mat.dimensions()).entered();

                let log_height = log2_strict_usize(mat.height());

                // If this is our first matrix at this height, initialise reduced_openings to zero.
                // Otherwise, get a mutable reference to it.
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![Challenge::ZERO; mat.height()]);
                debug_assert_eq!(reduced_opening_for_log_height.len(), mat.height());

                // Treating our matrix M as the evaluations of functions f_0, f_1, ...
                // Compute the evaluations of `Mred(x) = f_0(x) + alpha*f_1(x) + ...`
                let mat_compressed = debug_span!("compress mat").in_scope(|| {
                    // This will be reused for all points z which M is opened at so we collect into a vector.
                    mat.rowwise_packed_dot_product::<Challenge>(&packed_alpha_powers)
                        .collect::<Vec<_>>()
                });

                for (&point, openings) in points_for_mat.iter().zip(openings_for_mat) {
                    // If we have multiple matrices at the same height, we need to scale alpha to combine them.
                    // This means that reduced_openings will contain:
                    // Mred_0(x) + alpha^{M_0.width()}Mred_1(x) + alpha^{M_0.width() + M_1.width()}Mred_2(x) + ...
                    // Where M_0, M_1, ... are the matrices of the same height.
                    let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);

                    // As we have all the openings `f_i(z)`, we can combine them using `alpha`
                    // in an identical way to before to compute `Mred(z)`.
                    let reduced_openings: Challenge =
                        dot_product(alpha_powers.iter().copied(), openings.iter().copied());

                    mat_compressed
                        .par_iter()
                        .zip(reduced_opening_for_log_height.par_iter_mut())
                        // inv_denoms contains `1/(z - x)` for `x` in a coset `gK`.
                        // If `|K| =/= mat.height()` we actually want a subset of this
                        // corresponding to the evaluations over `gH` for `|H| = mat.height()`.
                        // As inv_denoms is bit reversed, the evaluations over `gH` are exactly
                        // the evaluations over `gK` at the indices `0..mat.height()`.
                        // So zip will truncate to the desired smaller length.
                        .zip(inv_denoms.get(&point).unwrap().par_iter())
                        // Map the function `Mred(x) -> (Mred(z) - Mred(x))/(z - x)`
                        // across the evaluation vector of `Mred(x)`. Adjust by alpha_pow_offset
                        // as needed.
                        .for_each(|((&reduced_row, ro), &inv_denom)| {
                            *ro += alpha_pow_offset * (reduced_openings - reduced_row) * inv_denom;
                        });
                    num_reduced[log_height] += mat.width();
                }
            }
        }

        // It remains to prove that all evaluation vectors in reduced_openings correspond to
        // low degree functions.
        let fri_input = reduced_openings.into_iter().rev().flatten().collect_vec();

        let folding: TwoAdicFriFoldingForMmcs<Val, InputMmcs> = TwoAdicFriFolding(PhantomData);

        // Produce the FRI proof.
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
        // For each commitment:
        commitments_with_opening_points: Vec<
            CommitmentWithOpeningPoints<Challenge, Self::Commitment, Self::Domain>,
        >,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // Write all evaluations to challenger.
        // Need to ensure to do this in the same order as the prover.
        for (_, round) in &commitments_with_opening_points {
            for (_, mat) in round {
                for (_, point) in mat {
                    challenger.observe_algebra_slice(point);
                }
            }
        }

        let folding: TwoAdicFriFoldingForMmcs<Val, InputMmcs> = TwoAdicFriFolding(PhantomData);

        verifier::verify_fri(
            &folding,
            &self.fri,
            proof,
            challenger,
            &commitments_with_opening_points,
            &self.mmcs,
        )?;

        Ok(())
    }

    fn build_periodic_lde_table(
        &self,
        periodic_cols: &[Vec<Val>],
        trace_domain: Self::Domain,
        quotient_domain: Self::Domain,
    ) -> PeriodicLdeTable<Val> {
        build_periodic_lde_table_two_adic::<Val, Dft>(
            &self.dft,
            periodic_cols,
            &trace_domain,
            &quotient_domain,
        )
    }
}

/// Compute vectors of inverse denominators for each unique opening point.
///
/// Arguments:
/// - `mats_and_points` is a list of matrices and for each matrix a list of points. We assume that
///    the total number of distinct points is very small as several methods contained herein are `O(n^2)`
///    in the number of points.
/// - `coset` is the set of points `gH` where `H` a two-adic subgroup such that `|H|` is greater
///     than or equal to the largest height of any matrix in `mats_and_points`. The values
///     in `coset` must be in bit-reversed order.
///
/// For each point `z`, let `M` be the matrix of largest height which opens at `z`.
/// let `H_z` be the unique subgroup of order `M.height()`. Compute the vector of
/// `1/(z - x)` for `x` in `gH_z`.
///
/// Return a LinearMap which allows us to recover the computed vectors for each `z`.
#[instrument(skip_all)]
fn compute_inverse_denominators<F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>>(
    mats_and_points: &[(Vec<M>, &Vec<Vec<EF>>)],
    coset: &[F],
) -> LinearMap<EF, Vec<EF>> {
    // For each `z`, find the maximal height of any matrix which we need to
    // open at `z`.
    let mut max_log_height_for_point: LinearMap<EF, usize> = LinearMap::new();
    for (mats, points) in mats_and_points {
        for (mat, points_for_mat) in izip!(mats, *points) {
            let log_height = log2_strict_usize(mat.height());
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = core::cmp::max(*lh, log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    // Compute the inverse denominators for each point `z`.
    max_log_height_for_point
        .into_iter()
        .map(|(z, log_height)| {
            (
                z,
                batch_multiplicative_inverse(
                    // As coset is stored in bit-reversed order,
                    // we can just take the first `2^log_height` elements.
                    &coset[..(1 << log_height)]
                        .iter()
                        .map(|&x| z - x)
                        .collect_vec(),
                ),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ExtVal;
    use p3_field::Field;

    #[test]
    fn quadratic_norm_inverses_match_extension_inverses() {
        let point_pair = [Goldilocks::from_u64(123), Goldilocks::from_u64(456)];
        let point = ExtVal::from_basis_coefficients_slice(&point_pair).unwrap();
        let coset = Goldilocks::GENERATOR.powers().take(4096).collect_vec();

        for count in [1, 3, 1024, 1025, 4096] {
            let actual = goldilocks_quadratic_inverse_denominators(
                point_pair,
                &coset[..count],
                Goldilocks::from_u64(7),
            );
            let expected = coset[..count].iter().map(|&x| (point - x).inverse());
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual, expected.as_basis_coefficients_slice());
            }
        }
    }
}
