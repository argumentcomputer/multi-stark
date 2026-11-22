//! Configurable FRI opening batching: Horner powers (`alpha^i`) and
//! multilinear tensor batching (`col_weights ⊗ open_weights`).
//!
//! Under Horner batching, combining $n_f$ opened functions with powers of a
//! single challenge $\alpha$ has soundness error $n_f / |\mathbb{F}_{\text{ext}}|$.
//! Under multilinear tensor batching, we sample:
//! - $k_{\text{col}} = \lceil \log_2 W_{\max} \rceil$ column challenges
//!   $(r^{\text{col}}_0, \dots, r^{\text{col}}_{k_{\text{col}}-1})$, expanding
//!   to column weights $w^{\text{col}}_c = \prod_b (r^{\text{col}}_b)^{c_b}$, and
//! - $k_{\text{open}} = \lceil \log_2 M_{\max} \rceil$ opening-pair challenges
//!   $(r^{\text{open}}_0, \dots, r^{\text{open}}_{k_{\text{open}}-1})$, expanding
//!   to opening weights $w^{\text{open}}_m = \prod_b (r^{\text{open}}_b)^{m_b}$,
//!
//! where $W_{\max}$ is the maximum column width across opened matrices and
//! $M_{\max}$ is the maximum number of `(matrix, point)` opening pairs at any
//! single LDE height.
//!
//! Each column $c$ of opening $m$ at a given height receives weight
//! $w^{\text{open}}_m \cdot w^{\text{col}}_c$ — a distinct multilinear monomial
//! of degree $\le 1$ in every challenge and total degree
//! $\le k_{\text{col}} + k_{\text{open}}$. By Schwartz-Zippel this reduces the
//! batching soundness error from $W_{\max} M_{\max} / |\mathbb{F}_{\text{ext}}|$
//! to $(k_{\text{col}} + k_{\text{open}}) / |\mathbb{F}_{\text{ext}}|$ (~9–10
//! extra bits of soundness) at **zero prover overhead**, because each matrix is
//! still compressed once via `rowwise_packed_dot_product(&packed_col_weights)`
//! and reused across all its opening points ($\zeta, \zeta g$).

use std::collections::BTreeMap;
use std::iter;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, HornerIter, PackedValue, TwoAdicField,
    batch_multiplicative_inverse,
};
use p3_fri::verifier::{self, FriError};
use p3_fri::{
    BatchMultiOpening, CommitPhaseMultiStep, CommitmentWithOpeningPoints, FriFoldingStrategy,
    FriParameters, FriProof,
};
use p3_matrix::Dimensions;
use p3_util::{log2_ceil_usize, log2_strict_usize, reverse_bits_len};
use serde::{Deserialize, Serialize};

/// Batching strategy for reducing opened trace columns and opening points into
/// per-height FRI input codewords.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FriBatching {
    /// Legacy Plonky3 Horner powers `1, α, α^2, ...` from a single challenge `α`.
    #[default]
    Horner,
    /// Protocol `v6` multilinear tensor batching `col_weights ⊗ open_weights`
    /// using `⌈log2(max_width)⌉ + ⌈log2(max_openings_per_height)⌉` challenges.
    Tensor,
}

impl FriBatching {
    /// Canonical protocol tag bound into the `v6` challenger seed.
    #[inline]
    pub const fn tag(self) -> u64 {
        match self {
            Self::Horner => 0,
            Self::Tensor => 1,
        }
    }
}

/// Number of multilinear tensor challenges required to index `count` items:
/// `0` when `count <= 1`, otherwise `⌈log2(count)⌉`.
#[inline]
pub const fn tensor_num_challenges(count: usize) -> usize {
    if count <= 1 {
        0
    } else {
        log2_ceil_usize(count)
    }
}

/// Expands `challenges` into `len` multilinear tensor weights where index
/// `j = Σ_b j_b 2^b` gets weight `∏_{b: j_b = 1} challenges[b]`.
///
/// Requires `len <= (1 << challenges.len())`. Uses exactly `len.saturating_sub(1)`
/// field multiplications (matching `alpha.powers().take(len)`).
pub fn tensor_weights<F: Field>(challenges: &[F], len: usize) -> Vec<F> {
    if len == 0 {
        return Vec::new();
    }
    let capacity_bound = 1usize
        .checked_shl(u32::try_from(challenges.len()).expect("too many tensor challenges"))
        .unwrap_or(usize::MAX);
    assert!(
        len <= capacity_bound,
        "tensor_weights: len {len} exceeds 2^{}",
        challenges.len()
    );
    let mut weights = Vec::with_capacity(len);
    weights.push(F::ONE);
    for &r in challenges {
        let prev_len = weights.len();
        for i in 0..prev_len {
            if weights.len() == len {
                break;
            }
            let w = weights[i] * r;
            weights.push(w);
        }
    }
    debug_assert_eq!(weights.len(), len);
    weights
}

/// Packs a slice of extension-field weights into `EF::ExtensionPacking` SIMD
/// vectors suitable for `Matrix::rowwise_packed_dot_product`.
pub fn pack_ext_weights<F: Field, EF: ExtensionField<F>>(
    weights: &[EF],
) -> Vec<EF::ExtensionPacking> {
    let pw = F::Packing::WIDTH;
    weights
        .chunks(pw)
        .map(|chunk| {
            EF::ExtensionPacking::from_basis_coefficients_fn(|c| {
                F::Packing::from_fn(|lane| {
                    chunk
                        .get(lane)
                        .map_or(F::ZERO, |w| w.as_basis_coefficients_slice()[c])
                })
            })
        })
        .collect()
}

/// Sampled batching weights shared between the FRI prover and verifier.
#[derive(Clone, Debug)]
pub enum FriBatchingSchedule<EF> {
    Horner {
        alpha: EF,
    },
    Tensor {
        col_weights: Vec<EF>,
        open_weights: Vec<EF>,
    },
}

impl<EF: Field> FriBatchingSchedule<EF> {
    /// Samples batching challenges from `challenger` in canonical protocol order.
    pub fn sample<Val, Challenger>(
        batching: FriBatching,
        max_width: usize,
        max_openings_per_height: usize,
        challenger: &mut Challenger,
    ) -> Self
    where
        Val: Field,
        EF: ExtensionField<Val>,
        Challenger: FieldChallenger<Val>,
    {
        match batching {
            FriBatching::Horner => Self::Horner {
                alpha: challenger.sample_algebra_element(),
            },
            FriBatching::Tensor => {
                let k_col = tensor_num_challenges(max_width);
                let col_challenges: Vec<EF> = (0..k_col)
                    .map(|_| challenger.sample_algebra_element())
                    .collect();
                let k_open = tensor_num_challenges(max_openings_per_height);
                let open_challenges: Vec<EF> = (0..k_open)
                    .map(|_| challenger.sample_algebra_element())
                    .collect();
                Self::Tensor {
                    col_weights: tensor_weights(&col_challenges, max_width),
                    open_weights: tensor_weights(&open_challenges, max_openings_per_height),
                }
            }
        }
    }
}

type FriOpenings<F> = Vec<(usize, F)>;

/// Verifies a FRI proof under the requested [`FriBatching`] strategy.
///
/// For [`FriBatching::Horner`], delegates directly to stock
/// [`p3_fri::verifier::verify_fri`] to guarantee 100% compatibility with
/// Plonky3. For [`FriBatching::Tensor`], performs the identical FRI commit,
/// PoW, Merkle multi-batch, fold, and final-polynomial checks using
/// `col_weights ⊗ open_weights` input reduction.
pub fn verify_fri_with_batching<Folding, Val, Challenge, InputMmcs, FriMmcs, Challenger>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    proof: &FriProof<Challenge, FriMmcs, Challenger::Witness, Folding::InputProof>,
    challenger: &mut Challenger,
    commitments_with_opening_points: &[CommitmentWithOpeningPoints<
        Challenge,
        InputMmcs::Commitment,
        TwoAdicMultiplicativeCoset<Val>,
    >],
    input_mmcs: &InputMmcs,
    batching: FriBatching,
) -> Result<(), FriError<FriMmcs::Error, InputMmcs::Error>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<FriMmcs::Commitment>,
    Folding: FriFoldingStrategy<
            Val,
            Challenge,
            InputError = InputMmcs::Error,
            InputProof = Vec<BatchMultiOpening<Val, InputMmcs>>,
        >,
{
    match batching {
        FriBatching::Horner => verifier::verify_fri(
            folding,
            params,
            proof,
            challenger,
            commitments_with_opening_points,
            input_mmcs,
        ),
        FriBatching::Tensor => verify_fri_with_batching_inner(
            folding,
            params,
            proof,
            challenger,
            commitments_with_opening_points,
            input_mmcs,
            FriBatching::Tensor,
        ),
    }
}

/// Core batching-generic FRI verifier (supports both `Horner` and `Tensor`;
/// exposed to tests so `Horner` is cross-checked against `p3_fri::verifier::verify_fri`).
pub(crate) fn verify_fri_with_batching_inner<
    Folding,
    Val,
    Challenge,
    InputMmcs,
    FriMmcs,
    Challenger,
>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    proof: &FriProof<Challenge, FriMmcs, Challenger::Witness, Folding::InputProof>,
    challenger: &mut Challenger,
    commitments_with_opening_points: &[CommitmentWithOpeningPoints<
        Challenge,
        InputMmcs::Commitment,
        TwoAdicMultiplicativeCoset<Val>,
    >],
    input_mmcs: &InputMmcs,
    batching: FriBatching,
) -> Result<(), FriError<FriMmcs::Error, InputMmcs::Error>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<FriMmcs::Commitment>,
    Folding: FriFoldingStrategy<
            Val,
            Challenge,
            InputError = InputMmcs::Error,
            InputProof = Vec<BatchMultiOpening<Val, InputMmcs>>,
        >,
{
    if params.num_queries == 0 {
        return Err(FriError::ZeroQueries);
    }

    // Derive max_width and max_openings_per_height from the authenticated
    // opening structure before sampling batching challenges.
    let mut max_width = 0usize;
    let mut openings_per_height: BTreeMap<usize, usize> = BTreeMap::new();
    for (batch, (_, mats)) in commitments_with_opening_points.iter().enumerate() {
        for (matrix, (domain, points_and_values)) in mats.iter().enumerate() {
            let Some((_, first_vals)) = points_and_values.first() else {
                return Err(FriError::MatrixWithoutOpeningPoints { batch, matrix });
            };
            let width = first_vals.len();
            for (point, (_, vals)) in points_and_values.iter().enumerate() {
                if vals.len() != width {
                    return Err(FriError::PointEvaluationCountMismatch {
                        batch,
                        matrix,
                        point,
                        expected: width,
                        got: vals.len(),
                    });
                }
            }
            max_width = max_width.max(width);
            let log_height = log2_strict_usize(domain.size()) + params.log_blowup;
            *openings_per_height.entry(log_height).or_insert(0) += points_and_values.len();
        }
    }
    let max_openings_per_height = openings_per_height.values().copied().max().unwrap_or(0);

    let schedule = FriBatchingSchedule::<Challenge>::sample(
        batching,
        max_width,
        max_openings_per_height,
        challenger,
    );

    let expected_rounds = proof.commit_phase_commits.len();
    if proof.commit_phase_openings.len() != expected_rounds {
        return Err(FriError::CommitPhaseOpeningsCountMismatch {
            expected: expected_rounds,
            got: proof.commit_phase_openings.len(),
        });
    }

    let log_arities: Vec<usize> = proof
        .commit_phase_openings
        .iter()
        .enumerate()
        .map(|(round, opening)| {
            let log_arity = usize::from(opening.log_arity);
            if (1..=params.max_log_arity).contains(&log_arity) {
                Ok(log_arity)
            } else {
                Err(FriError::InvalidLogArity {
                    round,
                    log_arity,
                    max: params.max_log_arity,
                })
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    for (round, (opening, &log_arity)) in proof
        .commit_phase_openings
        .iter()
        .zip(&log_arities)
        .enumerate()
    {
        if opening.sibling_values.len() != params.num_queries {
            return Err(FriError::CommitPhaseQueryCountMismatch {
                round,
                expected: params.num_queries,
                got: opening.sibling_values.len(),
            });
        }
        let arity = 1usize << log_arity;
        for siblings in &opening.sibling_values {
            if siblings.len() != arity - 1 {
                return Err(FriError::SiblingValuesLengthMismatch {
                    round,
                    expected: arity - 1,
                    got: siblings.len(),
                });
            }
        }
    }

    let total_log_reduction: usize = log_arities.iter().sum();
    let log_global_max_height = total_log_reduction + params.log_blowup + params.log_final_poly_len;

    if log_global_max_height > Val::TWO_ADICITY {
        return Err(FriError::GlobalMaxHeightTooLarge {
            log_global_max_height,
            two_adicity: Val::TWO_ADICITY,
        });
    }

    let expected_log_global_max_height = commitments_with_opening_points
        .iter()
        .flat_map(|(_, mats)| {
            mats.iter()
                .map(|(domain, _)| log2_strict_usize(domain.size()) + params.log_blowup)
        })
        .max();
    if let Some(expected) = expected_log_global_max_height
        && log_global_max_height != expected
    {
        return Err(FriError::GlobalMaxHeightMismatch {
            expected,
            got: log_global_max_height,
        });
    }

    if proof.commit_pow_witnesses.len() != proof.commit_phase_commits.len() {
        return Err(FriError::CommitPowWitnessCountMismatch {
            expected: proof.commit_phase_commits.len(),
            got: proof.commit_pow_witnesses.len(),
        });
    }

    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .zip(&proof.commit_pow_witnesses)
        .map(|(comm, witness)| {
            challenger.observe(comm.clone());
            if !challenger.check_witness(params.commit_proof_of_work_bits, *witness) {
                return Err(FriError::InvalidPowWitness);
            }
            Ok(challenger.sample_algebra_element())
        })
        .collect::<Result<Vec<_>, _>>()?;

    if proof.final_poly.len() != params.final_poly_len() {
        return Err(FriError::FinalPolyLengthMismatch {
            expected: params.final_poly_len(),
            got: proof.final_poly.len(),
        });
    }

    challenger.observe_algebra_slice(&proof.final_poly);

    for &log_arity in &log_arities {
        challenger.observe(Val::from_usize(log_arity));
    }

    if !challenger.check_witness(params.query_proof_of_work_bits, proof.query_pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    let log_final_height = params.log_blowup + params.log_final_poly_len;

    let indices: Vec<usize> = iter::repeat_with(|| {
        challenger.sample_bits(log_global_max_height + folding.extra_query_index_bits())
    })
    .take(params.num_queries)
    .collect();

    let reduced_openings = open_inputs_with_schedule(
        params,
        log_global_max_height,
        &indices,
        &proof.input_openings,
        &schedule,
        input_mmcs,
        commitments_with_opening_points,
    )?;

    let num_rounds = proof.commit_phase_commits.len();
    let mut group_indices_by_round: Vec<Vec<usize>> =
        vec![Vec::with_capacity(params.num_queries); num_rounds];
    let mut rows_by_round: Vec<Vec<Vec<Vec<Challenge>>>> =
        vec![Vec::with_capacity(params.num_queries); num_rounds];

    for (query, (&index, ro)) in indices.iter().zip(reduced_openings).enumerate() {
        let mut domain_index = index >> folding.extra_query_index_bits();
        let folded_eval = fold_query(
            folding,
            query,
            &mut domain_index,
            &betas,
            &log_arities,
            &proof.commit_phase_openings,
            ro,
            log_global_max_height,
            log_final_height,
            &mut group_indices_by_round,
            &mut rows_by_round,
        )?;

        let rev_idx = u64::try_from(reverse_bits_len(domain_index, log_global_max_height))
            .expect("query index exceeds u64");
        let x = Val::two_adic_generator(log_global_max_height).exp_u64(rev_idx);
        let eval: Challenge = proof.final_poly.iter().copied().horner(x);
        if eval != folded_eval {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    let mut log_current_height = log_global_max_height;
    for (round, ((commit, opening), &log_arity)) in proof
        .commit_phase_commits
        .iter()
        .zip(&proof.commit_phase_openings)
        .zip(&log_arities)
        .enumerate()
    {
        let arity = 1usize << log_arity;
        let log_folded_height = log_current_height - log_arity;
        let dims = &[Dimensions {
            width: arity,
            height: 1usize << log_folded_height,
        }];
        params
            .mmcs
            .verify_multi_batch(
                commit,
                dims,
                &group_indices_by_round[round],
                &rows_by_round[round],
                &opening.opening_proof,
            )
            .map_err(FriError::CommitPhaseMmcsError)?;
        log_current_height = log_folded_height;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn fold_query<Folding, F, EF, M>(
    folding: &Folding,
    query: usize,
    start_index: &mut usize,
    betas: &[EF],
    log_arities: &[usize],
    commit_phase_openings: &[CommitPhaseMultiStep<EF, M>],
    reduced_openings: FriOpenings<EF>,
    log_global_max_height: usize,
    log_final_height: usize,
    group_indices_by_round: &mut [Vec<usize>],
    rows_by_round: &mut [Vec<Vec<Vec<EF>>>],
) -> Result<EF, FriError<M::Error, Folding::InputError>>
where
    F: Field,
    EF: ExtensionField<F>,
    M: Mmcs<EF>,
    Folding: FriFoldingStrategy<F, EF>,
{
    let mut ro_iter = reduced_openings.into_iter().peekable();

    let Some((first_log_height, _)) = ro_iter.peek() else {
        return Err(FriError::MissingInitialReducedOpening {
            expected: log_global_max_height,
        });
    };
    if *first_log_height != log_global_max_height {
        return Err(FriError::InitialReducedOpeningHeightMismatch {
            expected: log_global_max_height,
            got: *first_log_height,
        });
    }
    let mut folded_eval = ro_iter.next().unwrap().1;
    let mut log_current_height = log_global_max_height;

    for (round, ((&beta, &log_arity), opening)) in betas
        .iter()
        .zip(log_arities)
        .zip(commit_phase_openings)
        .enumerate()
    {
        let arity = 1usize << log_arity;
        let index_in_group = *start_index % arity;
        let mut evals = EF::zero_vec(arity);
        evals[index_in_group] = folded_eval;

        let siblings = &opening.sibling_values[query];
        let mut sibling_idx = 0;
        for (j, eval) in evals.iter_mut().enumerate() {
            if j != index_in_group {
                *eval = siblings[sibling_idx];
                sibling_idx += 1;
            }
        }

        let log_folded_height = log_current_height - log_arity;
        *start_index >>= log_arity;

        folded_eval = folding.fold_row(
            *start_index,
            log_folded_height,
            log_arity,
            beta,
            evals.iter().copied(),
        );

        group_indices_by_round[round].push(*start_index);
        rows_by_round[round].push(vec![evals]);
        log_current_height = log_folded_height;

        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height) {
            let beta_pow = beta.exp_power_of_2(log_arity);
            folded_eval += beta_pow * ro;
        }
    }

    if log_current_height != log_final_height {
        return Err(FriError::FinalFoldHeightMismatch {
            expected: log_final_height,
            got: log_current_height,
        });
    }

    if let Some((next_log_height, _)) = ro_iter.next() {
        return Err(FriError::UnconsumedReducedOpenings {
            next_log_height,
            remaining: 1 + ro_iter.count(),
        });
    }

    Ok(folded_eval)
}

#[allow(clippy::type_complexity)]
fn open_inputs_with_schedule<Val, Challenge, InputMmcs, FriMmcs>(
    params: &FriParameters<FriMmcs>,
    log_global_max_height: usize,
    indices: &[usize],
    input_openings: &[BatchMultiOpening<Val, InputMmcs>],
    schedule: &FriBatchingSchedule<Challenge>,
    input_mmcs: &InputMmcs,
    commitments_with_opening_points: &[CommitmentWithOpeningPoints<
        Challenge,
        InputMmcs::Commitment,
        TwoAdicMultiplicativeCoset<Val>,
    >],
) -> Result<Vec<FriOpenings<Challenge>>, FriError<FriMmcs::Error, InputMmcs::Error>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
{
    if input_openings.len() != commitments_with_opening_points.len() {
        return Err(FriError::InputProofBatchCountMismatch {
            expected: commitments_with_opening_points.len(),
            got: input_openings.len(),
        });
    }

    for (batch, (batch_opening, (batch_commit, mats))) in input_openings
        .iter()
        .zip(commitments_with_opening_points.iter())
        .enumerate()
    {
        if batch_opening.opened_values.len() != indices.len() {
            return Err(FriError::InputOpeningsQueryCountMismatch {
                batch,
                expected: indices.len(),
                got: batch_opening.opened_values.len(),
            });
        }
        for opened_values in &batch_opening.opened_values {
            if opened_values.len() != mats.len() {
                return Err(FriError::BatchOpenedValuesCountMismatch {
                    batch,
                    expected: mats.len(),
                    got: opened_values.len(),
                });
            }
        }
        let batch_heights: Vec<usize> = mats
            .iter()
            .map(|(domain, _)| domain.size() << params.log_blowup)
            .collect();
        let batch_dims: Vec<Dimensions> = batch_heights
            .iter()
            .zip(mats)
            .enumerate()
            .map(|(matrix, (&height, (_, points_and_values)))| {
                let (_, values) = points_and_values
                    .first()
                    .ok_or(FriError::MatrixWithoutOpeningPoints { batch, matrix })?;
                Ok(Dimensions {
                    width: values.len(),
                    height,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let reduced_indices: Vec<usize> = batch_heights.iter().max().map_or_else(
            || vec![0; indices.len()],
            |&h| {
                let bits_reduced = log_global_max_height - log2_strict_usize(h);
                indices.iter().map(|&index| index >> bits_reduced).collect()
            },
        );

        input_mmcs
            .verify_multi_batch(
                batch_commit,
                &batch_dims,
                &reduced_indices,
                &batch_opening.opened_values,
                &batch_opening.opening_proof,
            )
            .map_err(FriError::InputError)?;
    }

    let mut denominators = Vec::new();
    for &index in indices {
        for (batch, (_, mats)) in commitments_with_opening_points.iter().enumerate() {
            for (matrix, (mat_domain, mat_points_and_values)) in mats.iter().enumerate() {
                let log_height = log2_strict_usize(mat_domain.size()) + params.log_blowup;
                let bits_reduced = log_global_max_height - log_height;
                let rev_reduced_index =
                    u64::try_from(reverse_bits_len(index >> bits_reduced, log_height))
                        .expect("reduced query index exceeds u64");
                let x =
                    Val::GENERATOR * Val::two_adic_generator(log_height).exp_u64(rev_reduced_index);

                for (point, (z, _)) in mat_points_and_values.iter().enumerate() {
                    let denominator = *z - x;
                    if denominator.is_zero() {
                        return Err(FriError::OpeningPointMatchesQueryPoint {
                            batch,
                            matrix,
                            point,
                        });
                    }
                    denominators.push(denominator);
                }
            }
        }
    }
    let mut inverse_denominators = batch_multiplicative_inverse(&denominators).into_iter();

    indices
        .iter()
        .enumerate()
        .map(|(query, _)| match schedule {
            FriBatchingSchedule::Horner { alpha } => {
                let mut reduced_openings = BTreeMap::<usize, (Challenge, Challenge)>::new();
                for (batch, (batch_opening, (_, mats))) in input_openings
                    .iter()
                    .zip(commitments_with_opening_points.iter())
                    .enumerate()
                {
                    for (matrix, (mat_opening, (mat_domain, mat_points_and_values))) in
                        batch_opening.opened_values[query]
                            .iter()
                            .zip(mats.iter())
                            .enumerate()
                    {
                        let log_height = log2_strict_usize(mat_domain.size()) + params.log_blowup;
                        let (alpha_pow, ro) = reduced_openings
                            .entry(log_height)
                            .or_insert((Challenge::ONE, Challenge::ZERO));

                        for (point, (_z, ps_at_z)) in mat_points_and_values.iter().enumerate() {
                            if mat_opening.len() != ps_at_z.len() {
                                return Err(FriError::PointEvaluationCountMismatch {
                                    batch,
                                    matrix,
                                    point,
                                    expected: mat_opening.len(),
                                    got: ps_at_z.len(),
                                });
                            }
                            let inv_denom = inverse_denominators
                                .next()
                                .expect("one inverse was precomputed per opening point");
                            for (&p_at_x, &p_at_z) in mat_opening.iter().zip(ps_at_z.iter()) {
                                *ro += *alpha_pow * (p_at_z - p_at_x) * inv_denom;
                                *alpha_pow *= *alpha;
                            }
                        }
                    }
                }
                if let Some((_, ro)) = reduced_openings.get(&params.log_blowup)
                    && !ro.is_zero()
                {
                    return Err(FriError::FinalPolyMismatch);
                }
                Ok(reduced_openings
                    .into_iter()
                    .rev()
                    .map(|(log_height, (_, ro))| (log_height, ro))
                    .collect())
            }
            FriBatchingSchedule::Tensor {
                col_weights,
                open_weights,
            } => {
                let mut reduced_openings = BTreeMap::<usize, (usize, Challenge)>::new();
                for (batch, (batch_opening, (_, mats))) in input_openings
                    .iter()
                    .zip(commitments_with_opening_points.iter())
                    .enumerate()
                {
                    for (matrix, (mat_opening, (mat_domain, mat_points_and_values))) in
                        batch_opening.opened_values[query]
                            .iter()
                            .zip(mats.iter())
                            .enumerate()
                    {
                        let log_height = log2_strict_usize(mat_domain.size()) + params.log_blowup;
                        let (opening_idx, ro) = reduced_openings
                            .entry(log_height)
                            .or_insert((0usize, Challenge::ZERO));

                        for (point, (_z, ps_at_z)) in mat_points_and_values.iter().enumerate() {
                            if mat_opening.len() != ps_at_z.len() {
                                return Err(FriError::PointEvaluationCountMismatch {
                                    batch,
                                    matrix,
                                    point,
                                    expected: mat_opening.len(),
                                    got: ps_at_z.len(),
                                });
                            }
                            let inv_denom = inverse_denominators
                                .next()
                                .expect("one inverse was precomputed per opening point");
                            let mut col_acc = Challenge::ZERO;
                            for ((&w_col, &p_at_x), &p_at_z) in col_weights
                                .iter()
                                .zip(mat_opening.iter())
                                .zip(ps_at_z.iter())
                            {
                                col_acc += w_col * (p_at_z - p_at_x);
                            }
                            let w_open = open_weights[*opening_idx];
                            *opening_idx += 1;
                            *ro += w_open * col_acc * inv_denom;
                        }
                    }
                }
                if let Some((_, ro)) = reduced_openings.get(&params.log_blowup)
                    && !ro.is_zero()
                {
                    return Err(FriError::FinalPolyMismatch);
                }
                Ok(reduced_openings
                    .into_iter()
                    .rev()
                    .map(|(log_height, (_, ro))| (log_height, ro))
                    .collect())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::CubicTrinomialExtensionField;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_maybe_rayon::prelude::*;

    type Ext3 = CubicTrinomialExtensionField<Goldilocks>;

    #[test]
    fn tensor_weights_match_explicit_multilinear_monomials() {
        let r = [
            Ext3::from_basis_coefficients_slice(&[
                Goldilocks::from_u64(3),
                Goldilocks::from_u64(5),
                Goldilocks::from_u64(7),
            ])
            .unwrap(),
            Ext3::from_basis_coefficients_slice(&[
                Goldilocks::from_u64(11),
                Goldilocks::from_u64(13),
                Goldilocks::from_u64(17),
            ])
            .unwrap(),
            Ext3::from_basis_coefficients_slice(&[
                Goldilocks::from_u64(19),
                Goldilocks::from_u64(23),
                Goldilocks::from_u64(29),
            ])
            .unwrap(),
        ];

        assert!(tensor_weights::<Ext3>(&[], 0).is_empty());
        assert_eq!(tensor_weights::<Ext3>(&[], 1), vec![Ext3::ONE]);

        for len in 1..=8 {
            let k = tensor_num_challenges(len);
            let got = tensor_weights(&r[..k], len);
            assert_eq!(got.len(), len);
            for (j, &w) in got.iter().enumerate() {
                let mut expected = Ext3::ONE;
                for (bit, &rb) in r[..k].iter().enumerate() {
                    if (j >> bit) & 1 == 1 {
                        expected *= rb;
                    }
                }
                assert_eq!(w, expected, "len={len} j={j}");
            }
        }
    }

    #[test]
    fn packed_ext_weights_dot_product_matches_scalar_for_cubic_extension() {
        let mut challenges = Vec::new();
        for i in 0..5u64 {
            challenges.push(
                Ext3::from_basis_coefficients_slice(&[
                    Goldilocks::from_u64(3 * i + 2),
                    Goldilocks::from_u64(5 * i + 7),
                    Goldilocks::from_u64(11 * i + 13),
                ])
                .unwrap(),
            );
        }
        let max_width = 25usize;
        let weights = tensor_weights(&challenges, max_width);
        let packed = pack_ext_weights::<Goldilocks, Ext3>(&weights);

        for width in [1usize, 3, 4, 7, 8, 17, 25] {
            let height = 16usize;
            let mat = RowMajorMatrix::new(
                (0..height * width)
                    .map(|i| Goldilocks::from_u64(u64::try_from(i * 37 + 11).unwrap()))
                    .collect(),
                width,
            );
            let got: Vec<Ext3> = mat
                .as_view()
                .rowwise_packed_dot_product::<Ext3>(&packed)
                .collect();
            for (r, &got_val) in got.iter().enumerate() {
                let mut expected = Ext3::ZERO;
                for (c, &w) in weights.iter().enumerate().take(width) {
                    expected += w * mat.values[r * width + c];
                }
                assert_eq!(got_val, expected, "width={width} row={r}");
            }
        }
    }

    fn run_pcs_horner_parity_and_tensor_roundtrip<EF>()
    where
        EF: ExtensionField<Goldilocks> + crate::traits::ExtensionOf<Goldilocks>,
    {
        use crate::p3_adapter::challenger::DomainChallenger;
        use crate::p3_adapter::domain_blake3::Blake3Domain;
        use crate::p3_adapter::spill::{SpillableMmcs, SpillableTwoAdicFriPcs};
        use p3_commit::{ExtensionMmcs, Pcs, PolynomialSpace};
        use p3_dft::Radix2DitParallel;
        use p3_fri::TwoAdicFriFolding;

        type Ch<EF> = DomainChallenger<EF>;

        let make_pcs = |batching: FriBatching| {
            let mmcs = SpillableMmcs::new_v6(1);
            let ext_mmcs = ExtensionMmcs::<Goldilocks, EF, _>::new(mmcs.clone());
            let fri_params = FriParameters {
                log_blowup: 2,
                log_final_poly_len: 1,
                max_log_arity: 2,
                num_queries: 12,
                commit_proof_of_work_bits: 2,
                query_proof_of_work_bits: 2,
                mmcs: ext_mmcs,
            };
            SpillableTwoAdicFriPcs::new(
                Radix2DitParallel::<Goldilocks>::default(),
                mmcs,
                fri_params,
            )
            .with_batching(batching)
        };

        let make_mat = |height: usize, width: usize, salt: u64| {
            RowMajorMatrix::new(
                (0..height * width)
                    .map(|i| {
                        Goldilocks::from_u64(
                            u64::try_from(i).unwrap().wrapping_mul(salt) + 17 + salt,
                        )
                    })
                    .collect(),
                width,
            )
        };

        // Two batches with distinct heights and widths (including width=1).
        let pcs_horner = make_pcs(FriBatching::Horner);
        let dom16 = Pcs::<EF, Ch<EF>>::natural_domain_for_degree(&pcs_horner, 16);
        let dom32 = Pcs::<EF, Ch<EF>>::natural_domain_for_degree(&pcs_horner, 32);
        let dom64 = Pcs::<EF, Ch<EF>>::natural_domain_for_degree(&pcs_horner, 64);

        let batch0 = vec![(dom32, make_mat(32, 5, 31)), (dom16, make_mat(16, 1, 53))];
        let batch1 = vec![(dom64, make_mat(64, 13, 97)), (dom32, make_mat(32, 7, 131))];

        let zeta: EF = <EF as BasedVectorSpace<Goldilocks>>::from_basis_coefficients_fn(|i| {
            Goldilocks::from_u64(1001 + i as u64 * 37)
        });
        let zeta_next_32 = zeta * dom32.first_point() * dom32.next_point(Goldilocks::ONE).unwrap();
        let zeta_next_64 = zeta * dom64.next_point(Goldilocks::ONE).unwrap();

        // 1. Horner parity: verify_fri_with_batching_inner(..., Horner) matches stock p3_fri::verifier::verify_fri
        let (c0, d0) = Pcs::<EF, Ch<EF>>::commit(&pcs_horner, batch0.clone());
        let (c1, d1) = Pcs::<EF, Ch<EF>>::commit(&pcs_horner, batch1.clone());
        let make_ch = |seed: &[u8], a: &_, b: &_| {
            let mut ch = Ch::<EF>::from_domain(seed.to_vec(), Blake3Domain::TranscriptSingle);
            ch.observe(Clone::clone(a));
            ch.observe(Clone::clone(b));
            ch
        };
        let mut p_ch = make_ch(b"fri-parity", &c0, &c1);

        let (opened_h, proof_h) = Pcs::<EF, Ch<EF>>::open(
            &pcs_horner,
            vec![
                (&d0, vec![vec![zeta, zeta_next_32], vec![zeta]]),
                (
                    &d1,
                    vec![vec![zeta, zeta_next_64], vec![zeta, zeta_next_32]],
                ),
            ],
            &mut p_ch,
        );

        let commit_and_openings = vec![
            (
                c0.clone(),
                vec![
                    (
                        dom32,
                        vec![
                            (zeta, opened_h[0][0][0].clone()),
                            (zeta_next_32, opened_h[0][0][1].clone()),
                        ],
                    ),
                    (dom16, vec![(zeta, opened_h[0][1][0].clone())]),
                ],
            ),
            (
                c1.clone(),
                vec![
                    (
                        dom64,
                        vec![
                            (zeta, opened_h[1][0][0].clone()),
                            (zeta_next_64, opened_h[1][0][1].clone()),
                        ],
                    ),
                    (
                        dom32,
                        vec![
                            (zeta, opened_h[1][1][0].clone()),
                            (zeta_next_32, opened_h[1][1][1].clone()),
                        ],
                    ),
                ],
            ),
        ];

        type OpeningSpec<EF> = CommitmentWithOpeningPoints<
            EF,
            p3_merkle_tree::MerkleCap<Goldilocks, [u8; 32]>,
            TwoAdicMultiplicativeCoset<Goldilocks>,
        >;
        let observe_openings = |ch: &mut Ch<EF>, openings: &[OpeningSpec<EF>]| {
            for (_, round) in openings {
                for (_, mat) in round {
                    for (_, vals) in mat {
                        ch.observe_algebra_slice(vals);
                    }
                }
            }
        };

        let mut v_stock = make_ch(b"fri-parity", &c0, &c1);
        observe_openings(&mut v_stock, &commit_and_openings);
        let mut v_inner = v_stock.clone();

        let fri_params_ref = FriParameters {
            log_blowup: 2,
            log_final_poly_len: 1,
            max_log_arity: 2,
            num_queries: 12,
            commit_proof_of_work_bits: 2,
            query_proof_of_work_bits: 2,
            mmcs: ExtensionMmcs::<Goldilocks, EF, _>::new(SpillableMmcs::new_v6(1)),
        };
        let input_mmcs = SpillableMmcs::new_v6(1);

        let folding = TwoAdicFriFolding(core::marker::PhantomData);

        verifier::verify_fri(
            &folding,
            &fri_params_ref,
            &proof_h,
            &mut v_stock,
            &commit_and_openings,
            &input_mmcs,
        )
        .expect("stock p3_fri::verifier::verify_fri should accept");

        verify_fri_with_batching_inner(
            &folding,
            &fri_params_ref,
            &proof_h,
            &mut v_inner,
            &commit_and_openings,
            &input_mmcs,
            FriBatching::Horner,
        )
        .expect("verify_fri_with_batching_inner(Horner) should accept");

        // Post-verification transcript states must be identical!
        let s1: EF = v_stock.sample_algebra_element();
        let s2: EF = v_inner.sample_algebra_element();
        assert_eq!(
            s1, s2,
            "Horner verifier transcript diverged from stock p3_fri"
        );

        // Tampered opened value must be rejected by both verifiers.
        let ef_one = <EF as PrimeCharacteristicRing>::ONE;
        let mut tampered = commit_and_openings.clone();
        tampered[1].1[0].1[0].1[0] += ef_one;
        let mut v_bad1 = make_ch(b"fri-parity", &c0, &c1);
        observe_openings(&mut v_bad1, &tampered);
        let mut v_bad2 = v_bad1.clone();
        assert!(
            verifier::verify_fri(
                &folding,
                &fri_params_ref,
                &proof_h,
                &mut v_bad1,
                &tampered,
                &input_mmcs,
            )
            .is_err()
        );
        assert!(
            verify_fri_with_batching_inner(
                &folding,
                &fri_params_ref,
                &proof_h,
                &mut v_bad2,
                &tampered,
                &input_mmcs,
                FriBatching::Horner,
            )
            .is_err()
        );

        // Cross-check: Horner proof must fail under Tensor verifier.
        let pcs_tensor = make_pcs(FriBatching::Tensor);
        let mut v_cross = make_ch(b"fri-parity", &c0, &c1);
        assert!(
            Pcs::<EF, Ch<EF>>::verify(
                &pcs_tensor,
                commit_and_openings.clone(),
                &proof_h,
                &mut v_cross,
            )
            .is_err(),
            "Horner proof must not verify under Tensor batching"
        );

        // 2. Tensor roundtrip: prove & verify, reject tampering & Horner cross-verify.
        let (tc0, td0) = Pcs::<EF, Ch<EF>>::commit(&pcs_tensor, batch0);
        let (tc1, td1) = Pcs::<EF, Ch<EF>>::commit(&pcs_tensor, batch1);
        assert_eq!(tc0, c0);
        assert_eq!(tc1, c1);

        let mut tp_ch = make_ch(b"fri-tensor", &tc0, &tc1);

        let (opened_t, proof_t) = Pcs::<EF, Ch<EF>>::open(
            &pcs_tensor,
            vec![
                (&td0, vec![vec![zeta, zeta_next_32], vec![zeta]]),
                (
                    &td1,
                    vec![vec![zeta, zeta_next_64], vec![zeta, zeta_next_32]],
                ),
            ],
            &mut tp_ch,
        );
        assert_eq!(
            opened_t, opened_h,
            "opened evaluations are independent of batching"
        );

        let t_commit_and_openings = commit_and_openings;
        let mut tv_ch = make_ch(b"fri-tensor", &tc0, &tc1);
        Pcs::<EF, Ch<EF>>::verify(
            &pcs_tensor,
            t_commit_and_openings.clone(),
            &proof_t,
            &mut tv_ch,
        )
        .expect("Tensor FRI proof should verify");

        // Tampered opened value fails under Tensor verifier.
        let mut tv_bad = make_ch(b"fri-tensor", &tc0, &tc1);
        assert!(
            Pcs::<EF, Ch<EF>>::verify(&pcs_tensor, tampered, &proof_t, &mut tv_bad).is_err(),
            "Tensor verifier must reject tampered opened value"
        );

        // Tampered final_poly fails under Tensor verifier.
        let mut bad_proof = proof_t.clone();
        bad_proof.final_poly[0] += ef_one;
        let mut tv_bad2 = make_ch(b"fri-tensor", &tc0, &tc1);
        assert!(
            Pcs::<EF, Ch<EF>>::verify(
                &pcs_tensor,
                t_commit_and_openings.clone(),
                &bad_proof,
                &mut tv_bad2,
            )
            .is_err(),
            "Tensor verifier must reject tampered final_poly"
        );

        // Tensor proof fails under Horner verifier.
        let mut hv_cross = make_ch(b"fri-tensor", &tc0, &tc1);
        assert!(
            Pcs::<EF, Ch<EF>>::verify(&pcs_horner, t_commit_and_openings, &proof_t, &mut hv_cross,)
                .is_err(),
            "Tensor proof must not verify under Horner batching"
        );
    }

    #[test]
    fn fri_batching_horner_parity_and_tensor_fp2_and_fp3() {
        run_pcs_horner_parity_and_tensor_roundtrip::<crate::types::ExtVal2>();
        run_pcs_horner_parity_and_tensor_roundtrip::<crate::types::ExtVal3>();
    }
}
