use core::marker::PhantomData;
use std::time::Instant;

use p3_commit::{ExtensionMmcs, Mmcs, Pcs};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, PackedFieldExtension, batch_multiplicative_inverse, dot_product};
use p3_fri::{
    FriFoldingStrategy, FriParameters as InnerFriParameters, TwoAdicFriFolding,
    TwoAdicFriFoldingForMmcs, prover,
};
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
use p3_maybe_rayon::prelude::*;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_util::linear_map::LinearMap;
use p3_util::reverse_slice_index_bits;

use super::*;
use crate::cuda::hiding_mmcs::{HidingCudaFriMmcs, HidingCudaFriRound, HidingCudaMmcs};
use crate::cuda::mmcs::CudaMmcs;
use crate::cuda::pcs::CudaTwoAdicFriPcs;
use crate::expr::Expr;
use crate::lookup::Lookup;
use crate::p3_adapter::domain_blake3::{Blake3Domain, DomainBlake3};
use crate::p3_adapter::fri_batching::{
    FriBatching, FriBatchingSchedule, pack_ext_weights, verify_fri_with_batching,
};
use crate::prover::Proof;
use crate::system::{CircuitInputs, System, SystemWitness};
use crate::types::{
    Challenger, CommitmentParameters, DomainChallenger, ExtVal2, ExtVal3, FriParameters,
    GoldilocksBlake3Config, GoldilocksBlake3V6Config, GoldilocksBlake3V6ZkConfig, Val,
};

fn random_ext<Challenge: ExtensionField<Goldilocks>>(rng: &mut SmallRng, d: usize) -> Challenge {
    let mut coeffs = [Goldilocks::ZERO; 3];
    for coeff in &mut coeffs[..d] {
        *coeff = rng.random::<Goldilocks>();
    }
    Challenge::from_basis_coefficients_slice(&coeffs[..d]).expect("extension element")
}

fn flatten_ext<Challenge: ExtensionField<Goldilocks>>(
    values: &[Challenge],
    d: usize,
) -> Vec<Goldilocks> {
    let mut flat = Vec::with_capacity(values.len() * d);
    for value in values {
        flat.extend_from_slice(&value.as_basis_coefficients_slice()[..d]);
    }
    flat
}

fn unflatten_ext<Challenge: ExtensionField<Goldilocks>>(
    matrix: &RowMajorMatrix<Goldilocks>,
    d: usize,
) -> Vec<Challenge> {
    assert_eq!(matrix.width(), d);
    matrix
        .values
        .chunks_exact(d)
        .map(|chunk| Challenge::from_basis_coefficients_slice(chunk).expect("extension row"))
        .collect()
}

fn check_fri_fold_parity<Challenge: ExtensionField<Goldilocks>>(
    device: i32,
    ext_degree: usize,
    ext_w: Goldilocks,
    seed: u64,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let folding: TwoAdicFriFolding<(), ()> = TwoAdicFriFolding(PhantomData);

    for log_arity in 1usize..=5 {
        let arity = 1usize << log_arity;
        let heights = [arity, arity << 4, 1usize << 13];
        for height in heights {
            if height < arity {
                continue;
            }
            let log_height = log2_strict_usize(height);
            let final_height = height >> log_arity;

            let input_ext: Vec<Challenge> = (0..height)
                .map(|_| random_ext(&mut rng, ext_degree))
                .collect();
            let input_flat = flatten_ext(&input_ext, ext_degree);
            let input_mat = RowMajorMatrix::new(input_flat, ext_degree);
            let input_gpu = CudaLde::from_row_major_matrix(device, &input_mat);

            let beta: Challenge = random_ext(&mut rng, ext_degree);
            let mut beta_step = beta;
            let mut betas_flat = Vec::with_capacity(log_arity * ext_degree);
            for _ in 0..log_arity {
                betas_flat
                    .extend_from_slice(&beta_step.as_basis_coefficients_slice()[..ext_degree]);
                beta_step = beta_step.square();
            }
            let beta_power = beta.exp_power_of_2(log_arity);
            let beta_power_slice = &beta_power.as_basis_coefficients_slice()[..ext_degree];
            let g_inv = Goldilocks::two_adic_generator(log_height).inverse();

            let cpu_folded = folding.fold_matrix(
                beta,
                log_arity,
                RowMajorMatrix::new(input_ext.clone(), arity).as_view(),
            );
            assert_eq!(cpu_folded.len(), final_height);

            // Case 1: without next_reduced.
            let gpu_folded = input_gpu.fri_fold(
                None,
                &betas_flat,
                log_arity,
                beta_power_slice,
                g_inv,
                ext_w,
                ext_degree,
            );
            assert_eq!(gpu_folded.height(), final_height);
            assert_eq!(gpu_folded.width(), ext_degree);
            let gpu_mat = gpu_folded.to_row_major_matrix();
            assert!(
                gpu_mat.values.iter().all(|&v| {
                    // SAFETY: Goldilocks is repr(transparent) over u64.
                    let raw = unsafe { core::mem::transmute::<Goldilocks, u64>(v) };
                    raw < Goldilocks::ORDER_U64
                }),
                "folded codeword must be canonical (d={ext_degree}, log_arity={log_arity}, h={height})"
            );
            let gpu_ext: Vec<Challenge> = unflatten_ext(&gpu_mat, ext_degree);
            assert_eq!(
                gpu_ext, cpu_folded,
                "GPU fri_fold mismatch without next_reduced (d={ext_degree}, log_arity={log_arity}, h={height})"
            );

            // Case 2: with next_reduced folded in via beta^(2^log_arity).
            let next_ext: Vec<Challenge> = (0..final_height)
                .map(|_| random_ext(&mut rng, ext_degree))
                .collect();
            let next_flat = flatten_ext(&next_ext, ext_degree);
            let mut next_gpu =
                CudaReducedOpening::new_with_degree(device, final_height, ext_degree);
            next_gpu.add_host_flat(&next_flat);

            let cpu_with_next: Vec<Challenge> = cpu_folded
                .iter()
                .zip(&next_ext)
                .map(|(&f, &n)| f + beta_power * n)
                .collect();
            let gpu_with_next = input_gpu.fri_fold(
                Some(&next_gpu),
                &betas_flat,
                log_arity,
                beta_power_slice,
                g_inv,
                ext_w,
                ext_degree,
            );
            let gpu_with_next_ext: Vec<Challenge> =
                unflatten_ext(&gpu_with_next.to_row_major_matrix(), ext_degree);
            assert_eq!(
                gpu_with_next_ext, cpu_with_next,
                "GPU fri_fold mismatch with next_reduced (d={ext_degree}, log_arity={log_arity}, h={height})"
            );
        }
    }
}

#[test]
fn cuda_fri_fold_matches_cpu_all_arities_ext2_and_ext3() {
    let device = configured_device();
    check_fri_fold_parity::<ExtVal2>(device, 2, Goldilocks::from_u64(7), 0xF102_0002);
    check_fri_fold_parity::<ExtVal3>(device, 3, Goldilocks::ZERO, 0xF103_0003);
}

fn check_fri_merkle_parity<Challenge: ExtensionField<Goldilocks>>(
    device: i32,
    ext_degree: usize,
    seed: u64,
) {
    type CpuDomMmcs = MerkleTreeMmcs<
        Goldilocks,
        u8,
        SerializingHasher<DomainBlake3>,
        CompressionFunctionFromHasher<DomainBlake3, 2, 32>,
        2,
        32,
    >;

    let mut rng = SmallRng::seed_from_u64(seed);
    let hiding_mmcs = HidingCudaMmcs::new_v6_zk(0, device);
    let hiding_ext_mmcs: ExtensionMmcs<Goldilocks, Challenge, HidingCudaMmcs> =
        ExtensionMmcs::new(hiding_mmcs.clone());

    for log_arity in 1usize..=4 {
        let arity = 1usize << log_arity;
        for height in [64usize, 4096usize] {
            let tree_height = height / arity;
            let codeword_ext: Vec<Challenge> = (0..height)
                .map(|_| random_ext(&mut rng, ext_degree))
                .collect();
            let codeword_flat = flatten_ext(&codeword_ext, ext_degree);
            let codeword_mat = RowMajorMatrix::new(codeword_flat.clone(), ext_degree);
            let codeword_gpu = CudaLde::from_row_major_matrix(device, &codeword_mat);

            let query_rows = [0usize, 1, tree_height / 2, 1, tree_height - 1, 0];
            let dims = [Dimensions {
                width: arity,
                height: tree_height,
            }];

            // Plain (v5) and domain-separated (v6) FRI Merkle trees.
            for domain_v6 in [false, true] {
                let (leaf_dom, comp_dom) = if domain_v6 {
                    (Blake3Domain::MerkleLeaf, Blake3Domain::MerkleCompress)
                } else {
                    (Blake3Domain::Plain, Blake3Domain::Plain)
                };
                let cpu_base = CpuDomMmcs::new(
                    SerializingHasher::new(DomainBlake3::new(leaf_dom)),
                    CompressionFunctionFromHasher::new(DomainBlake3::new(comp_dom)),
                    0,
                );
                let cpu_ext: ExtensionMmcs<Goldilocks, Challenge, CpuDomMmcs> =
                    ExtensionMmcs::new(cpu_base);
                let (cpu_commit, cpu_data) =
                    cpu_ext.commit_matrix(RowMajorMatrix::new(codeword_ext.clone(), arity));

                let gpu_tree = CudaMixedMerkleTree::from_fri_codeword_with_domain(
                    &codeword_gpu,
                    arity,
                    domain_v6,
                );
                assert_eq!(
                    gpu_tree.root(),
                    cpu_commit.roots()[0],
                    "FRI Merkle root mismatch (d={ext_degree}, arity={arity}, h={height}, v6={domain_v6})"
                );

                let (cpu_opened, cpu_proof) = cpu_ext.open_multi_batch(&query_rows, &cpu_data);
                let gpu_pruned = gpu_tree.open_pruned_siblings(&query_rows);
                assert_eq!(
                    gpu_pruned.sibling_hashes, cpu_proof.sibling_hashes,
                    "FRI Merkle pruned proof mismatch (d={ext_degree}, arity={arity}, h={height}, v6={domain_v6})"
                );

                // Gather rows from resident GPU codeword and verify with CPU verifier.
                let codeword_rows: Vec<usize> = query_rows
                    .iter()
                    .flat_map(|&row| (0..arity).map(move |col| row * arity + col))
                    .collect();
                let opened_flat = codeword_gpu.rows(&codeword_rows);
                let gpu_opened: Vec<Vec<Vec<Challenge>>> = opened_flat
                    .chunks_exact(arity)
                    .map(|rows| {
                        vec![
                            rows.iter()
                                .map(|r| {
                                    Challenge::from_basis_coefficients_slice(r)
                                        .expect("extension row")
                                })
                                .collect(),
                        ]
                    })
                    .collect();
                assert_eq!(gpu_opened, cpu_opened);
                cpu_ext
                    .verify_multi_batch(&cpu_commit, &dims, &query_rows, &gpu_opened, &gpu_pruned)
                    .expect("GPU FRI Merkle multi-batch opening must verify on CPU");
            }

            // Salted v6-zk FRI Merkle tree.
            let salt_seed: [u8; 32] = core::array::from_fn(|i| {
                u8::try_from((i * 37 + arity + ext_degree) & 0xFF).unwrap()
            });
            let flattened_leaves = RowMajorMatrix::new(codeword_flat, arity * ext_degree);
            let (cpu_zk_commit, cpu_zk_data) =
                hiding_mmcs.commit_cpu_with_seed(vec![flattened_leaves], salt_seed);
            let gpu_zk_tree =
                CudaMixedMerkleTree::from_fri_codeword_v6zk(&codeword_gpu, arity, &salt_seed);
            assert_eq!(
                gpu_zk_tree.root(),
                cpu_zk_commit.roots()[0],
                "v6-zk FRI Merkle root mismatch (d={ext_degree}, arity={arity}, h={height})"
            );

            let round = HidingCudaFriRound {
                codeword: CudaLde::from_row_major_matrix(device, &codeword_mat),
                tree: gpu_zk_tree,
                arity,
                salt_seed,
            };
            let (gpu_zk_rows, mut gpu_zk_proof) =
                hiding_ext_mmcs.open_cuda_fri_hiding_batch(&round, &query_rows);
            let (_, cpu_zk_proof) = hiding_mmcs.open_multi_batch(&query_rows, &cpu_zk_data);
            assert_eq!(
                gpu_zk_proof, cpu_zk_proof,
                "v6-zk FRI Merkle multi-proof mismatch (d={ext_degree}, arity={arity}, h={height})"
            );
            let wrapped_rows: Vec<Vec<Vec<Challenge>>> =
                gpu_zk_rows.into_iter().map(|r| vec![r]).collect();
            hiding_ext_mmcs
                .verify_multi_batch(
                    &cpu_zk_commit,
                    &dims,
                    &query_rows,
                    &wrapped_rows,
                    &gpu_zk_proof,
                )
                .expect("GPU v6-zk FRI Merkle opening must verify");

            // Tampered salt must fail verification.
            gpu_zk_proof.salts[0][0][0] ^= 1;
            assert!(
                hiding_ext_mmcs
                    .verify_multi_batch(
                        &cpu_zk_commit,
                        &dims,
                        &query_rows,
                        &wrapped_rows,
                        &gpu_zk_proof,
                    )
                    .is_err(),
                "tampered v6-zk FRI leaf salt must fail verification"
            );
        }
    }
}

#[test]
fn cuda_fri_merkle_matches_cpu_all_arities_plain_v6_and_v6zk() {
    let device = configured_device();
    check_fri_merkle_parity::<ExtVal2>(device, 2, 0xA117_0002);
    check_fri_merkle_parity::<ExtVal3>(device, 3, 0xA117_0003);
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn cpu_reference_open<Challenge, ChallengerType>(
    mmcs: &CudaMmcs,
    fri: &InnerFriParameters<ExtensionMmcs<Goldilocks, Challenge, CudaMmcs>>,
    batching: FriBatching,
    commitments_with_points: &[(
        &<CudaMmcs as Mmcs<Goldilocks>>::ProverData<RowMajorMatrix<Goldilocks>>,
        Vec<Vec<Challenge>>,
    )],
    challenger: &mut ChallengerType,
) -> (
    p3_commit::OpenedValues<Challenge>,
    p3_fri::FriProof<
        Challenge,
        ExtensionMmcs<Goldilocks, Challenge, CudaMmcs>,
        Goldilocks,
        Vec<p3_fri::BatchMultiOpening<Goldilocks, CudaMmcs>>,
    >,
)
where
    Challenge: ExtensionField<Goldilocks>,
    ChallengerType: p3_challenger::FieldChallenger<Goldilocks>
        + p3_challenger::GrindingChallenger<Witness = Goldilocks>
        + p3_challenger::CanObserve<p3_symmetric::MerkleCap<Goldilocks, [u8; 32]>>,
{
    let mats_and_points: Vec<_> = commitments_with_points
        .iter()
        .map(|(data, points)| {
            let mats = mmcs
                .get_matrices(data)
                .into_iter()
                .map(|m| m.as_view())
                .collect::<Vec<_>>();
            (mats, points)
        })
        .collect();
    let global_max_height = mats_and_points
        .iter()
        .flat_map(|(mats, _)| mats.iter().map(Matrix::height))
        .max()
        .unwrap();
    let log_global_max_height = log2_strict_usize(global_max_height);

    let mut opened_max_width = 0usize;
    let mut openings_per_height = [0usize; 33];
    for (mats, points) in &mats_and_points {
        for (mat, pts) in mats.iter().zip(points.iter()) {
            if !pts.is_empty() {
                opened_max_width = opened_max_width.max(mat.width());
                openings_per_height[log2_strict_usize(mat.height())] += pts.len();
            }
        }
    }
    let max_openings_per_height = openings_per_height.into_iter().max().unwrap_or(0);

    let mut coset: Vec<Goldilocks> =
        TwoAdicMultiplicativeCoset::new(Goldilocks::GENERATOR, log_global_max_height)
            .unwrap()
            .iter()
            .collect();
    reverse_slice_index_bits(&mut coset);

    let mut max_log: LinearMap<Challenge, usize> = LinearMap::new();
    for (mats, points) in &mats_and_points {
        for (mat, pts) in mats.iter().zip(points.iter()) {
            let lh = log2_strict_usize(mat.height());
            for &z in pts {
                if let Some(cur) = max_log.get_mut(&z) {
                    *cur = (*cur).max(lh);
                } else {
                    max_log.insert(z, lh);
                }
            }
        }
    }
    let inv_denoms: LinearMap<Challenge, Vec<Challenge>> = max_log
        .into_iter()
        .map(|(z, lh)| {
            let diffs: Vec<Challenge> = coset[..(1usize << lh)].iter().map(|&x| z - x).collect();
            (z, batch_multiplicative_inverse(&diffs))
        })
        .collect();
    let adjusted_weights: LinearMap<Challenge, Vec<Challenge>> = inv_denoms
        .iter()
        .map(|(point, denoms)| (*point, compute_adjusted_weights(*point, denoms)))
        .collect();

    let all_opened_values: p3_commit::OpenedValues<Challenge> = mats_and_points
        .iter()
        .map(|(mats, points)| {
            mats.iter()
                .zip(points.iter())
                .map(|(mat, pts)| {
                    let h = mat.height() >> fri.log_blowup;
                    let (low_coset, _) = mat.split_rows(h);
                    pts.iter()
                        .map(|&point| {
                            let adjusted = &adjusted_weights.get(&point).unwrap()[..h];
                            let ys = low_coset.interpolate_coset_with_precomputation(
                                Goldilocks::GENERATOR,
                                point,
                                adjusted,
                            );
                            challenger.observe_algebra_slice(&ys);
                            ys
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let schedule = FriBatchingSchedule::<Challenge>::sample(
        batching,
        opened_max_width,
        max_openings_per_height,
        challenger,
    );
    let packed_col_weights: Vec<Challenge::ExtensionPacking> = match &schedule {
        FriBatchingSchedule::Horner { alpha } => {
            Challenge::ExtensionPacking::packed_ext_powers_capped(*alpha, opened_max_width)
                .collect()
        }
        FriBatchingSchedule::Tensor { col_weights, .. } => {
            pack_ext_weights::<Goldilocks, Challenge>(col_weights)
        }
    };
    let col_weights: Vec<Challenge> = match &schedule {
        FriBatchingSchedule::Horner { alpha } => alpha.powers().take(opened_max_width).collect(),
        FriBatchingSchedule::Tensor { col_weights, .. } => col_weights.clone(),
    };

    let mut num_reduced = [0usize; 33];
    let mut opening_idx_by_height = [0usize; 33];
    let mut reduced_openings: [Option<Vec<Challenge>>; 33] = core::array::from_fn(|_| None);
    for ((mats, points), openings_round) in mats_and_points.iter().zip(&all_opened_values) {
        for ((mat, pts), openings_mat) in mats.iter().zip(points.iter()).zip(openings_round.iter())
        {
            if pts.is_empty() {
                continue;
            }
            let lh = log2_strict_usize(mat.height());
            let target =
                reduced_openings[lh].get_or_insert_with(|| vec![Challenge::ZERO; mat.height()]);
            let compressed: Vec<Challenge> = mat
                .rowwise_packed_dot_product::<Challenge>(&packed_col_weights)
                .collect();
            for (&point, ys) in pts.iter().zip(openings_mat.iter()) {
                let offset = match &schedule {
                    FriBatchingSchedule::Horner { alpha } => alpha.exp_u64(
                        u64::try_from(num_reduced[lh]).expect("reduced width exceeds u64"),
                    ),
                    FriBatchingSchedule::Tensor { open_weights, .. } => {
                        open_weights[opening_idx_by_height[lh]]
                    }
                };
                let reduced_y: Challenge =
                    dot_product(col_weights.iter().copied(), ys.iter().copied());
                let denoms = inv_denoms.get(&point).unwrap();
                for ((&row_val, out), &inv_d) in
                    compressed.iter().zip(target.iter_mut()).zip(denoms)
                {
                    *out += offset * (reduced_y - row_val) * inv_d;
                }
                num_reduced[lh] += mat.width();
                opening_idx_by_height[lh] += 1;
            }
        }
    }

    let fri_input: Vec<Vec<Challenge>> = reduced_openings.into_iter().rev().flatten().collect();
    let folding: TwoAdicFriFoldingForMmcs<Goldilocks, CudaMmcs> = TwoAdicFriFolding(PhantomData);
    let fri_proof = prover::prove_fri(
        &folding,
        fri,
        fri_input,
        challenger,
        log_global_max_height,
        commitments_with_points,
        mmcs,
    );
    (all_opened_values, fri_proof)
}

#[allow(clippy::too_many_arguments)]
fn check_pcs_open_cpu_gpu_parity<Challenge, ChallengerType>(
    device: i32,
    domain_v6: bool,
    batching: FriBatching,
    log_blowup: usize,
    log_final_poly_len: usize,
    max_log_arity: usize,
    trace_shapes: &[(usize, usize)],
    make_challenger: impl Fn() -> ChallengerType,
    ext_degree: usize,
    seed: u64,
) where
    Challenge: ExtensionField<Goldilocks>,
    ChallengerType: p3_challenger::FieldChallenger<Goldilocks>
        + p3_challenger::GrindingChallenger<Witness = Goldilocks>
        + p3_challenger::CanObserve<p3_symmetric::MerkleCap<Goldilocks, [u8; 32]>>
        + Clone,
{
    let dft = CudaDft::new(device);
    let mmcs = if domain_v6 {
        CudaMmcs::new_v6(0, device)
    } else {
        CudaMmcs::new_plain(0, device)
    };
    let fri_mmcs: ExtensionMmcs<Goldilocks, Challenge, CudaMmcs> = ExtensionMmcs::new(mmcs.clone());
    let fri_params = InnerFriParameters {
        log_blowup,
        log_final_poly_len,
        max_log_arity,
        num_queries: 16,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
        mmcs: fri_mmcs,
    };
    let pcs = CudaTwoAdicFriPcs::<Goldilocks, _, _, _>::new(dft, mmcs.clone(), fri_params.clone())
        .with_batching(batching);

    let mut rng = SmallRng::seed_from_u64(seed);
    let max_lde_height = trace_shapes
        .iter()
        .map(|&(h, _)| h << log_blowup)
        .max()
        .unwrap();
    let final_fri_height = (1usize << log_blowup) * (1usize << log_final_poly_len);
    assert!(
        max_lde_height > 1024 && max_lde_height > final_fri_height,
        "test must trigger resident CUDA FRI path (max_lde_height={max_lde_height}, final={final_fri_height})"
    );

    let domains_and_mats: Vec<(
        TwoAdicMultiplicativeCoset<Goldilocks>,
        RowMajorMatrix<Goldilocks>,
    )> = trace_shapes
        .iter()
        .map(|&(height, width)| {
            let domain =
                TwoAdicMultiplicativeCoset::new(Goldilocks::ONE, log2_strict_usize(height))
                    .unwrap();
            let mat = RowMajorMatrix::new(
                (0..height * width)
                    .map(|_| rng.random::<Goldilocks>())
                    .collect(),
                width,
            );
            (domain, mat)
        })
        .collect();

    let (commitment, prover_data) =
        Pcs::<Challenge, ChallengerType>::commit(&pcs, domains_and_mats);

    let zeta: Challenge = random_ext(&mut rng, ext_degree);
    let points_per_mat: Vec<Vec<Challenge>> = trace_shapes
        .iter()
        .map(|&(height, _)| {
            let g = Goldilocks::two_adic_generator(log2_strict_usize(height));
            vec![zeta, zeta * g]
        })
        .collect();

    let mut gpu_challenger = make_challenger();
    let mut cpu_challenger = gpu_challenger.clone();
    let mut ver_challenger = gpu_challenger.clone();

    let (gpu_opened, gpu_proof) = pcs.open(
        vec![(&prover_data, points_per_mat.clone())],
        &mut gpu_challenger,
    );
    let (cpu_opened, cpu_proof) = cpu_reference_open(
        &mmcs,
        &fri_params,
        batching,
        &[(&prover_data, points_per_mat.clone())],
        &mut cpu_challenger,
    );

    // Exact parity on opened values, commits, final poly, arities, siblings, and Merkle proofs.
    assert_eq!(gpu_opened, cpu_opened, "opened values mismatch");
    assert_eq!(
        gpu_proof.commit_phase_commits, cpu_proof.commit_phase_commits,
        "FRI commit_phase_commits mismatch"
    );
    assert_eq!(
        gpu_proof.final_poly, cpu_proof.final_poly,
        "FRI final_poly mismatch"
    );
    assert_eq!(
        gpu_proof.commit_pow_witnesses, cpu_proof.commit_pow_witnesses,
        "FRI commit PoW mismatch"
    );
    assert_eq!(
        gpu_proof.query_pow_witness, cpu_proof.query_pow_witness,
        "FRI query PoW mismatch"
    );
    assert_eq!(
        gpu_proof.commit_phase_openings.len(),
        cpu_proof.commit_phase_openings.len(),
        "FRI round count mismatch"
    );
    for (round_idx, (gpu_step, cpu_step)) in gpu_proof
        .commit_phase_openings
        .iter()
        .zip(&cpu_proof.commit_phase_openings)
        .enumerate()
    {
        assert_eq!(
            gpu_step.log_arity, cpu_step.log_arity,
            "round {round_idx} log_arity mismatch"
        );
        assert_eq!(
            gpu_step.sibling_values, cpu_step.sibling_values,
            "round {round_idx} sibling_values mismatch"
        );
        assert_eq!(
            gpu_step.opening_proof.sibling_hashes, cpu_step.opening_proof.sibling_hashes,
            "round {round_idx} Merkle sibling_hashes mismatch"
        );
    }
    assert_eq!(
        gpu_proof.input_openings.len(),
        cpu_proof.input_openings.len()
    );
    for (gpu_in, cpu_in) in gpu_proof
        .input_openings
        .iter()
        .zip(&cpu_proof.input_openings)
    {
        assert_eq!(gpu_in.opened_values, cpu_in.opened_values);
        assert_eq!(
            gpu_in.opening_proof.sibling_hashes,
            cpu_in.opening_proof.sibling_hashes
        );
    }

    // Confirm CPU verifier accepts the GPU proof.
    type OpenedMatrix<C> = (TwoAdicMultiplicativeCoset<Goldilocks>, Vec<(C, Vec<C>)>);
    let verify_mats: Vec<OpenedMatrix<Challenge>> = trace_shapes
        .iter()
        .zip(&points_per_mat)
        .zip(&gpu_opened[0])
        .map(|((&(height, _), pts), vals)| {
            let domain =
                TwoAdicMultiplicativeCoset::new(Goldilocks::ONE, log2_strict_usize(height))
                    .unwrap();
            let pairs = pts.iter().copied().zip(vals.iter().cloned()).collect();
            (domain, pairs)
        })
        .collect();
    for (_, vals) in &verify_mats {
        for (_, evals) in vals {
            p3_challenger::FieldChallenger::observe_algebra_slice(
                &mut ver_challenger,
                evals.as_slice(),
            );
        }
    }
    let folding: TwoAdicFriFoldingForMmcs<Goldilocks, CudaMmcs> = TwoAdicFriFolding(PhantomData);
    verify_fri_with_batching(
        &folding,
        &fri_params,
        &gpu_proof,
        &mut ver_challenger,
        &[(commitment, verify_mats)],
        &mmcs,
        batching,
    )
    .expect("CPU verifier must accept GPU multi-arity FRI proof");
}

#[test]
fn cuda_pcs_open_multi_arity_matches_cpu_prove_fri_v5_and_v6() {
    let device = configured_device();
    let v6_challenger = || {
        DomainChallenger::<ExtVal3>::from_domain(
            b"fri-arity-v6-test".to_vec(),
            Blake3Domain::TranscriptSingle,
        )
    };
    let v5_challenger =
        || Challenger::from_hasher(b"fri-arity-v5-test".to_vec(), p3_blake3::Blake3);

    // v6 (Ext3): uniform and mixed heights across max_log_arity in 1..=4, log_final_poly_len in {0, 6, 8}.
    for max_log_arity in [1usize, 2, 3, 4] {
        for log_final_poly_len in [0usize, 6, 8] {
            for batching in [FriBatching::Tensor, FriBatching::Horner] {
                // Uniform height: 2^12 trace -> 2^14 LDE (> 1024 and >= 2^14 so arity=4 hits full fold even with log_final=8).
                check_pcs_open_cpu_gpu_parity::<ExtVal3, _>(
                    device,
                    true,
                    batching,
                    2,
                    log_final_poly_len,
                    max_log_arity,
                    &[(4096, 5)],
                    v6_challenger,
                    3,
                    0x6000 + u64::try_from(max_log_arity * 100 + log_final_poly_len * 10).unwrap(),
                );
                // Mixed heights: 2^12, 2^10, 2^9 traces -> 2^14, 2^12, 2^11 LDEs (forces mixed log_arity steps!).
                check_pcs_open_cpu_gpu_parity::<ExtVal3, _>(
                    device,
                    true,
                    batching,
                    2,
                    log_final_poly_len,
                    max_log_arity,
                    &[(4096, 3), (1024, 4), (512, 2)],
                    v6_challenger,
                    3,
                    0x7000 + u64::try_from(max_log_arity * 100 + log_final_poly_len * 10).unwrap(),
                );
            }
        }
    }

    // v5 (Ext2): uniform and mixed heights at max_log_arity = 2, 3.
    for max_log_arity in [2usize, 3] {
        for log_final_poly_len in [0usize, 6] {
            check_pcs_open_cpu_gpu_parity::<ExtVal2, _>(
                device,
                false,
                FriBatching::Horner,
                2,
                log_final_poly_len,
                max_log_arity,
                &[(1024, 5)],
                v5_challenger,
                2,
                0x5000 + u64::try_from(max_log_arity * 100 + log_final_poly_len * 10).unwrap(),
            );
            check_pcs_open_cpu_gpu_parity::<ExtVal2, _>(
                device,
                false,
                FriBatching::Horner,
                2,
                log_final_poly_len,
                max_log_arity,
                &[(4096, 3), (1024, 4), (512, 2)],
                v5_challenger,
                2,
                0x5500 + u64::try_from(max_log_arity * 100 + log_final_poly_len * 10).unwrap(),
            );
        }
    }
}

fn single_circuit_cubic_inputs() -> Vec<CircuitInputs<Val>> {
    let a = Expr::main(0);
    let b = Expr::main(1);
    let target = Expr::main(2);
    let a3 = Expr::main(3);
    let mult = Expr::main(4);
    let target_next = Expr::main_next(2);
    vec![CircuitInputs {
        main_width: 5,
        preprocessed: None,
        constraints: vec![
            a.clone() + b - target.clone(),
            a.clone() * a.clone() * a - a3,
            Expr::IsTransition * (target_next - target.clone()),
        ],
        ext_constraints: vec![],
        lookups: vec![Lookup::pull(
            mult,
            vec![Expr::constant(Val::from_u32(77)), target],
        )],
        lookup_group_size: 1,
    }]
}

fn single_circuit_cubic_witness(
    rows: usize,
    target: u32,
    seed: u32,
) -> (Vec<RowMajorMatrix<Val>>, Vec<Vec<Val>>) {
    let mut values = Vec::with_capacity(rows * 5);
    for r in 0..rows {
        let a_u = seed.wrapping_add(u32::try_from(r).unwrap().wrapping_mul(17)) % target;
        let b_u = target - a_u;
        let a = Val::from_u32(a_u);
        let b = Val::from_u32(b_u);
        let t = Val::from_u32(target);
        let a3 = a * a * a;
        let mult = if r == 0 { Val::ONE } else { Val::ZERO };
        values.extend([a, b, t, a3, mult]);
    }
    (
        vec![RowMajorMatrix::new(values, 5)],
        vec![vec![Val::from_u32(77), Val::from_u32(target)]],
    )
}

fn mismatched_height_system_inputs(rom_rows: usize) -> Vec<CircuitInputs<Val>> {
    let rom_prep = RowMajorMatrix::new(
        (0..u32::try_from(rom_rows).unwrap())
            .map(Val::from_u32)
            .collect(),
        1,
    );
    let rom_circuit = CircuitInputs {
        main_width: 1,
        preprocessed: Some(rom_prep),
        constraints: vec![],
        ext_constraints: vec![],
        lookups: vec![Lookup::pull(
            Expr::main(0),
            vec![Expr::constant(Val::from_u32(10)), Expr::preprocessed(0)],
        )],
        lookup_group_size: 1,
    };

    let a = Expr::main(0);
    let b = Expr::main(1);
    let sum = Expr::main(2);
    let mult = Expr::main(3);
    let work_circuit = CircuitInputs {
        main_width: 4,
        preprocessed: None,
        constraints: vec![sum.clone() - (a.clone() + b.clone())],
        ext_constraints: vec![],
        lookups: vec![
            Lookup::push(
                Expr::constant(Val::ONE),
                vec![Expr::constant(Val::from_u32(10)), a],
            ),
            Lookup::push(
                Expr::constant(Val::ONE),
                vec![Expr::constant(Val::from_u32(10)), b],
            ),
            Lookup::pull(mult, vec![Expr::constant(Val::from_u32(20)), sum]),
        ],
        lookup_group_size: 2,
    };

    let sq_circuit = CircuitInputs {
        main_width: 2,
        preprocessed: None,
        constraints: vec![Expr::main(0) * Expr::main(0) - Expr::main(1)],
        ext_constraints: vec![],
        lookups: vec![],
        lookup_group_size: 1,
    };

    vec![rom_circuit, work_circuit, sq_circuit]
}

fn mismatched_height_system_witness(
    rom_rows: usize,
    work_rows: usize,
    sq_rows: usize,
) -> (Vec<RowMajorMatrix<Val>>, Vec<Vec<Val>>) {
    let mask = u32::try_from(rom_rows - 1).unwrap();
    let mut rom_mults = vec![0u32; rom_rows];
    let mut work_vals = Vec::with_capacity(work_rows * 4);
    let mut claims = Vec::new();
    for r in 0..work_rows {
        let r32 = u32::try_from(r).unwrap();
        let a = (r32.wrapping_mul(13) + 7) & mask;
        let b = (r32.wrapping_mul(29) + 19) & mask;
        let sum = a + b;
        let mult = u32::from(r == 0);
        if mult == 1 {
            claims.push(vec![Val::from_u32(20), Val::from_u32(sum)]);
        }
        rom_mults[usize::try_from(a).unwrap()] += 1;
        rom_mults[usize::try_from(b).unwrap()] += 1;
        work_vals.extend([
            Val::from_u32(a),
            Val::from_u32(b),
            Val::from_u32(sum),
            Val::from_u32(mult),
        ]);
    }
    let rom_trace = RowMajorMatrix::new(rom_mults.into_iter().map(Val::from_u32).collect(), 1);
    let work_trace = RowMajorMatrix::new(work_vals, 4);
    let sq_trace = RowMajorMatrix::new(
        (0..u32::try_from(sq_rows).unwrap())
            .flat_map(|i| {
                let x = Val::from_u32(i + 3);
                [x, x * x]
            })
            .collect(),
        2,
    );
    (vec![rom_trace, work_trace, sq_trace], claims)
}

#[test]
fn cuda_full_system_multi_arity_v6_and_v5() {
    let device = configured_device();
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: 0,
    };

    // v6 (Ext3): single-circuit uniform & multi-circuit mismatched heights across
    // max_log_arity in 1..=4 and log_final_poly_len in {0, 6, 8}.
    for max_log_arity in [1usize, 2, 3, 4] {
        for log_final_poly_len in [0usize, 6, 8] {
            let fri = FriParameters {
                log_final_poly_len,
                max_log_arity,
                num_queries: 20,
                commit_proof_of_work_bits: 0,
                query_proof_of_work_bits: 0,
            };

            // 1. Single-circuit uniform height (2^12 trace -> 2^14 LDE > 1024 and >= 2^14 so arity=4 hits full fold even with log_final=8).
            let (sys, key) = System::new(
                GoldilocksBlake3V6Config::with_device(comm, fri, FriBatching::Tensor, Some(device)),
                single_circuit_cubic_inputs(),
            );
            let (traces, claims) = single_circuit_cubic_witness(4096, 500, 11);
            let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();
            let proof = sys.prove_multiple_claims(
                &key,
                &claim_refs,
                SystemWitness::from_stage_1(traces.clone(), &sys),
            );
            sys.verify_multiple_claims(&claim_refs, &proof)
                .unwrap_or_else(|e| {
                    panic!("v6 single-circuit verify failed (arity={max_log_arity}, final={log_final_poly_len}): {e:?}")
                });
            let arities: Vec<usize> = proof
                .opening_proof
                .commit_phase_openings
                .iter()
                .map(|s| usize::from(s.log_arity))
                .collect();
            assert!(
                arities.contains(&max_log_arity),
                "expected log_arity={max_log_arity} in {arities:?}"
            );
            let bytes = proof.to_bytes().unwrap();
            let decoded = Proof::<GoldilocksBlake3V6Config>::from_bytes(&bytes).unwrap();
            sys.verify_multiple_claims(&claim_refs, &decoded).unwrap();

            // 2. Multi-circuit mismatched heights: rom=512 (LDE 2048), work=4096 (LDE 16384), sq=1024 (LDE 4096).
            let (msys, mkey) = System::new(
                GoldilocksBlake3V6Config::with_device(comm, fri, FriBatching::Tensor, Some(device)),
                mismatched_height_system_inputs(512),
            );
            let (mtraces, mclaims) = mismatched_height_system_witness(512, 4096, 1024);
            let mclaim_refs: Vec<&[Val]> = mclaims.iter().map(Vec::as_slice).collect();
            let mproof = msys.prove_multiple_claims(
                &mkey,
                &mclaim_refs,
                SystemWitness::from_stage_1(mtraces, &msys),
            );
            msys.verify_multiple_claims(&mclaim_refs, &mproof)
                .unwrap_or_else(|e| {
                    panic!("v6 multi-circuit verify failed (arity={max_log_arity}, final={log_final_poly_len}): {e:?}")
                });
            let marities: Vec<usize> = mproof
                .opening_proof
                .commit_phase_openings
                .iter()
                .map(|s| usize::from(s.log_arity))
                .collect();
            if max_log_arity >= 3 {
                // Heights are 2^14, 2^12, 2^11 => steps between injections are 2 and 1,
                // then folding down from 2^11 uses max_log_arity when final_height permits!
                assert!(
                    marities.contains(&1) && marities.contains(&2),
                    "expected mixed arities for mismatched heights, got {marities:?}"
                );
            }
        }
    }

    // v5 (Ext2): single-circuit and multi-circuit at max_log_arity = 2, 3.
    for max_log_arity in [2usize, 3] {
        for log_final_poly_len in [0usize, 6] {
            let fri = FriParameters {
                log_final_poly_len,
                max_log_arity,
                num_queries: 20,
                commit_proof_of_work_bits: 0,
                query_proof_of_work_bits: 0,
            };
            let (sys, key) = System::new(
                GoldilocksBlake3Config::with_device(comm, fri, Some(device)),
                single_circuit_cubic_inputs(),
            );
            let (traces, claims) = single_circuit_cubic_witness(1024, 500, 17);
            let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();
            let proof = sys.prove_multiple_claims(
                &key,
                &claim_refs,
                SystemWitness::from_stage_1(traces, &sys),
            );
            sys.verify_multiple_claims(&claim_refs, &proof)
                .unwrap_or_else(|e| {
                    panic!("v5 single-circuit verify failed (arity={max_log_arity}, final={log_final_poly_len}): {e:?}")
                });

            let (msys, mkey) = System::new(
                GoldilocksBlake3Config::with_device(comm, fri, Some(device)),
                mismatched_height_system_inputs(512),
            );
            let (mtraces, mclaims) = mismatched_height_system_witness(512, 2048, 1024);
            let mclaim_refs: Vec<&[Val]> = mclaims.iter().map(Vec::as_slice).collect();
            let mproof = msys.prove_multiple_claims(
                &mkey,
                &mclaim_refs,
                SystemWitness::from_stage_1(mtraces, &msys),
            );
            msys.verify_multiple_claims(&mclaim_refs, &mproof)
                .unwrap_or_else(|e| {
                    panic!("v5 multi-circuit verify failed (arity={max_log_arity}, final={log_final_poly_len}): {e:?}")
                });
        }
    }
}

#[test]
fn cuda_full_system_multi_arity_v6_zk() {
    let device = configured_device();
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: 0,
    };

    for max_log_arity in [1usize, 2, 3, 4] {
        for log_final_poly_len in [0usize, 6, 8] {
            let fri = FriParameters {
                log_final_poly_len,
                max_log_arity,
                num_queries: 24,
                commit_proof_of_work_bits: 0,
                query_proof_of_work_bits: 0,
            };

            // 1. Single-circuit uniform height: rows = 2048 -> ZK LDE height = 2 * 2048 * 4 = 16384 (2^14 > 1024).
            let (sys, key) = System::new(
                GoldilocksBlake3V6ZkConfig::with_device(
                    comm,
                    fri,
                    FriBatching::Tensor,
                    Some(device),
                ),
                single_circuit_cubic_inputs(),
            );
            let (traces, claims) = single_circuit_cubic_witness(2048, 500, 19);
            let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();
            let proof = sys.prove_multiple_claims(
                &key,
                &claim_refs,
                SystemWitness::from_stage_1(traces, &sys),
            );
            sys.verify_multiple_claims(&claim_refs, &proof)
                .unwrap_or_else(|e| {
                    panic!("v6-zk single-circuit verify failed (arity={max_log_arity}, final={log_final_poly_len}): {e:?}")
                });
            let arities: Vec<usize> = proof
                .opening_proof
                .commit_phase_openings
                .iter()
                .map(|s| usize::from(s.log_arity))
                .collect();
            assert!(
                arities.contains(&max_log_arity),
                "v6-zk expected log_arity={max_log_arity} in {arities:?}"
            );

            // Serialization roundtrip + tamper checks.
            let bytes = proof.to_bytes().unwrap();
            let decoded = Proof::<GoldilocksBlake3V6ZkConfig>::from_bytes(&bytes).unwrap();
            sys.verify_multiple_claims(&claim_refs, &decoded).unwrap();

            if max_log_arity == 3 && log_final_poly_len == 8 {
                let mut bad_sibling = decoded.clone();
                bad_sibling.opening_proof.commit_phase_openings[0].sibling_values[0][0] +=
                    ExtVal3::ONE;
                assert!(
                    sys.verify_multiple_claims(&claim_refs, &bad_sibling)
                        .is_err(),
                    "tampered v6-zk FRI sibling must fail verification"
                );
                let mut bad_salt = decoded;
                bad_salt.opening_proof.commit_phase_openings[0]
                    .opening_proof
                    .salts[0][0][0] ^= 1;
                assert!(
                    sys.verify_multiple_claims(&claim_refs, &bad_salt).is_err(),
                    "tampered v6-zk FRI leaf salt must fail verification"
                );
            }

            // 2. Multi-circuit mismatched heights: rom=256 (LDE 2048), work=2048 (LDE 16384), sq=512 (LDE 4096).
            // All ZK LDE heights > 2^(8+2) = 1024!
            let (msys, mkey) = System::new(
                GoldilocksBlake3V6ZkConfig::with_device(
                    comm,
                    fri,
                    FriBatching::Tensor,
                    Some(device),
                ),
                mismatched_height_system_inputs(256),
            );
            let (mtraces, mclaims) = mismatched_height_system_witness(256, 2048, 512);
            let mclaim_refs: Vec<&[Val]> = mclaims.iter().map(Vec::as_slice).collect();
            let mproof = msys.prove_multiple_claims(
                &mkey,
                &mclaim_refs,
                SystemWitness::from_stage_1(mtraces, &msys),
            );
            msys.verify_multiple_claims(&mclaim_refs, &mproof)
                .unwrap_or_else(|e| {
                    panic!("v6-zk multi-circuit verify failed (arity={max_log_arity}, final={log_final_poly_len}): {e:?}")
                });
            let marities: Vec<usize> = mproof
                .opening_proof
                .commit_phase_openings
                .iter()
                .map(|s| usize::from(s.log_arity))
                .collect();
            if max_log_arity >= 3 {
                assert!(
                    marities.contains(&1) && marities.contains(&2),
                    "v6-zk expected mixed arities for mismatched heights, got {marities:?}"
                );
            }
        }
    }
}

#[test]
#[ignore = "live GPU timing smoke test for 2^20 and 2^22 single-circuit v6 & v6-zk max_log_arity=3"]
fn cuda_smoke_timing_v6_and_v6zk_arity3() {
    let device = configured_device();
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: 0,
    };
    let fri = FriParameters {
        log_final_poly_len: 8,
        max_log_arity: 3,
        num_queries: 64,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };

    for &log_rows in &[20usize, 22usize] {
        let rows = 1usize << log_rows;
        let (traces, claims) = single_circuit_cubic_witness(rows, 500, 11);
        let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();

        // v6 (Ext3)
        let (sys_v6, key_v6) = System::new(
            GoldilocksBlake3V6Config::with_device(comm, fri, FriBatching::Tensor, Some(device)),
            single_circuit_cubic_inputs(),
        );
        let w_v6 = SystemWitness::from_stage_1(traces.clone(), &sys_v6);
        let t0 = Instant::now();
        let proof_v6 = sys_v6.prove_multiple_claims(&key_v6, &claim_refs, w_v6);
        let prove_v6 = t0.elapsed();
        let t1 = Instant::now();
        sys_v6
            .verify_multiple_claims(&claim_refs, &proof_v6)
            .expect("v6 smoke proof must verify");
        let verify_v6 = t1.elapsed();
        let bytes_v6 = proof_v6.to_bytes().unwrap().len();
        let arities_v6: Vec<u8> = proof_v6
            .opening_proof
            .commit_phase_openings
            .iter()
            .map(|s| s.log_arity)
            .collect();
        eprintln!(
            "[RTX PRO 6000] v6    2^{log_rows} (max_log_arity=3, log_final=8, q=64): prove={:.3}s verify={:.3}ms bytes={} arities={arities_v6:?}",
            prove_v6.as_secs_f64(),
            verify_v6.as_secs_f64() * 1e3,
            bytes_v6,
        );

        // v6-zk (Ext3 hiding)
        let (sys_zk, key_zk) = System::new(
            GoldilocksBlake3V6ZkConfig::with_device(comm, fri, FriBatching::Tensor, Some(device)),
            single_circuit_cubic_inputs(),
        );
        let w_zk = SystemWitness::from_stage_1(traces, &sys_zk);
        let t2 = Instant::now();
        let proof_zk = sys_zk.prove_multiple_claims(&key_zk, &claim_refs, w_zk);
        let prove_zk = t2.elapsed();
        let t3 = Instant::now();
        sys_zk
            .verify_multiple_claims(&claim_refs, &proof_zk)
            .expect("v6-zk smoke proof must verify");
        let verify_zk = t3.elapsed();
        let bytes_zk = proof_zk.to_bytes().unwrap().len();
        let arities_zk: Vec<u8> = proof_zk
            .opening_proof
            .commit_phase_openings
            .iter()
            .map(|s| s.log_arity)
            .collect();
        eprintln!(
            "[RTX PRO 6000] v6-zk 2^{log_rows} (max_log_arity=3, log_final=8, q=64): prove={:.3}s verify={:.3}ms bytes={} arities={arities_zk:?}",
            prove_zk.as_secs_f64(),
            verify_zk.as_secs_f64() * 1e3,
            bytes_zk,
        );
    }
}
