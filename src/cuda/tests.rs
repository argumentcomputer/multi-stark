//! GPU↔CPU equivalence tests for the CUDA backend.
//!
//! These tests only compile when `--features cuda` is enabled and only
//! pass on a host with a working CUDA toolchain + a GPU. Run on the
//! Blackwell box as the first sanity check after a fresh build:
//!
//! ```bash
//! cargo test --release --features parallel,cuda
//! ```

use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::{
    Goldilocks, default_goldilocks_poseidon2_8, default_goldilocks_poseidon2_16,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::Permutation;

use super::{CudaRadix2Dft, permute_8_gpu, permute_16_gpu};

/// Build a deterministic test matrix of `height × width` Goldilocks elements.
///
/// Filled with `Goldilocks::from_u64((row * width + col + 1) as u64)` so any
/// off-by-one in indexing or column ordering shows up as a clear mismatch.
fn fixture(height: usize, width: usize) -> RowMajorMatrix<Goldilocks> {
    let values = (0..height * width)
        .map(|i| Goldilocks::from_u64(i as u64 + 1))
        .collect::<Vec<_>>();
    RowMajorMatrix::new(values, width)
}

fn assert_dft_matches(height: usize, width: usize) {
    let mat = fixture(height, width);
    let cpu = Radix2DitParallel::<Goldilocks>::default()
        .dft_batch(mat.clone())
        .to_row_major_matrix();
    let gpu = CudaRadix2Dft::<Goldilocks>::default()
        .dft_batch(mat)
        .to_row_major_matrix();
    assert_eq!(
        cpu.values, gpu.values,
        "GPU dft_batch diverged from CPU for height={height}, width={width}",
    );
    assert_eq!(cpu.height(), gpu.height());
    assert_eq!(cpu.width(), gpu.width());
}

#[test]
fn dft_matches_cpu_single_column() {
    // Single-column path skips the on-device transpose. Important to test
    // separately because the FFI takes a different code path.
    assert_dft_matches(1 << 8, 1);
}

#[test]
fn dft_matches_cpu_narrow_matrix() {
    // 4 columns matches the Mmcs digest size; common path for stage-2 trace.
    assert_dft_matches(1 << 10, 4);
}

#[test]
fn dft_matches_cpu_wide_matrix() {
    // 32 columns approximates a typical AIR's main trace width.
    assert_dft_matches(1 << 12, 32);
}

#[test]
fn dft_matches_cpu_height_one_noop() {
    // Edge case: height-1 input should pass through unchanged (no NTT applies).
    assert_dft_matches(1, 4);
}

fn assert_coset_lde_matches(height: usize, width: usize, added_bits: usize) {
    let mat = fixture(height, width);
    let shift = Goldilocks::GENERATOR;
    let cpu = Radix2DitParallel::<Goldilocks>::default()
        .coset_lde_batch(mat.clone(), added_bits, shift)
        .to_row_major_matrix();
    let gpu = CudaRadix2Dft::<Goldilocks>::default()
        .coset_lde_batch(mat, added_bits, shift)
        .to_row_major_matrix();
    assert_eq!(
        cpu.values, gpu.values,
        "GPU coset_lde_batch diverged from CPU \
         (height={height}, width={width}, added_bits={added_bits})",
    );
    assert_eq!(cpu.height(), gpu.height());
    assert_eq!(cpu.width(), gpu.width());
}

#[test]
fn coset_lde_matches_cpu_typical_blowup() {
    // log_blowup = 1 is the default in multi-stark CommitmentParameters;
    // 4 columns matches the Mmcs digest size.
    assert_coset_lde_matches(1 << 10, 4, 1);
}

#[test]
fn coset_lde_matches_cpu_wide_matrix() {
    // 16-column main trace, 2× blowup. Exercises the multi-column
    // transpose path and the per-element shift kernel under load.
    assert_coset_lde_matches(1 << 12, 16, 1);
}

#[test]
fn coset_lde_matches_cpu_larger_blowup() {
    // log_blowup = 2 = 4× extension. Verifies the zero-pad + extended-domain
    // forward NTT cooperate correctly for non-trivial blowup factors.
    assert_coset_lde_matches(1 << 8, 4, 2);
}

#[test]
fn coset_lde_matches_cpu_single_column() {
    // Single-column path skips the on-device transpose. Important to
    // test separately because the FFI takes a different code path.
    assert_coset_lde_matches(1 << 8, 1, 1);
}

#[test]
fn coset_lde_matches_cpu_height_one() {
    // Edge case: height-1 input is handled in Rust without an FFI call;
    // the result must still match Plonky3's CPU semantics (constant vector
    // of `ext_height` copies of the single input row).
    assert_coset_lde_matches(1, 4, 3);
}

// ---------- Poseidon2 permutation equivalence ----------------------------
//
// These are the most important tests: if the GPU Poseidon2 disagrees with
// the CPU on a single permutation, every Merkle root is wrong, every FRI
// commitment is wrong, every proof is invalid. They must pass before any
// downstream Phase-3 work is trusted.

fn poseidon2_perm_8_inputs() -> [[Goldilocks; 8]; 4] {
    [
        // Plonky3's own test vector for default_goldilocks_poseidon2_8.
        Goldilocks::new_array([0, 1, 2, 3, 4, 5, 6, 7]),
        // All zeros: the most stripped-down sanity test.
        [Goldilocks::ZERO; 8],
        // All ones: catches sign-flip errors in the matrix-diag entries.
        [Goldilocks::ONE; 8],
        // Adversarial pattern that exercises every state position with a
        // distinct large value to surface index-mixing bugs.
        Goldilocks::new_array([
            0xfffffffeffffffff,
            0x7fffffff80000001,
            0xdeadbeefdeadbeef,
            0x0123456789abcdef,
            0xfedcba9876543210,
            0x55aa55aa55aa55aa,
            0xaa55aa55aa55aa55,
            0x123456789abcdef0,
        ]),
    ]
}

fn poseidon2_perm_16_inputs() -> [[Goldilocks; 16]; 4] {
    [
        // Direct extension of Plonky3's width-8 vector.
        Goldilocks::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        [Goldilocks::ZERO; 16],
        [Goldilocks::ONE; 16],
        // Plonky3's own test vector for default_goldilocks_poseidon2_16
        // (from poseidon2.rs::test_default_goldilocks_poseidon2_width_16).
        Goldilocks::new_array([
            0x4d3f967fab9d4979,
            0x57e1fba55677697e,
            0x57429a86e75a3774,
            0x31d379f3a592b5eb,
            0x497232e1b648e3f1,
            0x325a7db57173c39e,
            0xa802252d78bee916,
            0x8920f55e154adef8,
            0xa1225bc9c7913658,
            0xd687be5097ffd038,
            0x89f514ef0c913e48,
            0x21fd4a9cf548cd84,
            0x570a1586ada436ff,
            0x46bfbf38ccd740ae,
            0x23651b3f3ab26484,
            0xe90f3b02127fa552,
        ]),
    ]
}

#[test]
fn poseidon2_permute_8_matches_cpu() {
    let cpu_perm = default_goldilocks_poseidon2_8();
    for (idx, input) in poseidon2_perm_8_inputs().iter().enumerate() {
        let mut cpu = *input;
        let mut gpu = *input;
        cpu_perm.permute_mut(&mut cpu);
        permute_8_gpu(&mut gpu);
        assert_eq!(
            cpu, gpu,
            "Poseidon2 width-8 GPU/CPU mismatch on input vector #{idx}"
        );
    }
}

#[test]
fn poseidon2_permute_16_matches_cpu() {
    let cpu_perm = default_goldilocks_poseidon2_16();
    for (idx, input) in poseidon2_perm_16_inputs().iter().enumerate() {
        let mut cpu = *input;
        let mut gpu = *input;
        cpu_perm.permute_mut(&mut cpu);
        permute_16_gpu(&mut gpu);
        assert_eq!(
            cpu, gpu,
            "Poseidon2 width-16 GPU/CPU mismatch on input vector #{idx}"
        );
    }
}

// ---------- Merkle primitives equivalence -------------------------------
//
// These test the `cuda::mmcs` helpers against Plonky3's CPU
// `Hash` (PaddingFreeSponge<P16, 16, 12, 4>) and `Compress`
// (TruncatedPermutation<P8, 2, 4, 8>) — the CPU pieces that build the
// production multi-stark Mmcs.

use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, PseudoCompressionFunction, TruncatedPermutation};

use super::mmcs::{Digest, build_merkle_tree, compress_paired, compress_pairs, leaf_hash_matrix};
use crate::types::{Compress, Hash};

fn cpu_hash() -> Hash {
    PaddingFreeSponge::new(default_goldilocks_poseidon2_16())
}

fn cpu_compress() -> Compress {
    TruncatedPermutation::new(default_goldilocks_poseidon2_8())
}

fn fixture_digests(n: usize) -> Vec<Digest> {
    (0..n)
        .map(|i| {
            [
                Goldilocks::from_u64((4 * i + 1) as u64),
                Goldilocks::from_u64((4 * i + 2) as u64),
                Goldilocks::from_u64((4 * i + 3) as u64),
                Goldilocks::from_u64((4 * i + 4) as u64),
            ]
        })
        .collect()
}

#[test]
fn leaf_hash_matches_cpu_single_row() {
    // Width 8 — fits in a single rate block of size 12.
    let row: [Goldilocks; 8] = Goldilocks::new_array([1, 2, 3, 4, 5, 6, 7, 8]);
    let expected: [Goldilocks; 4] = cpu_hash().hash_iter(row.iter().copied());
    let mat = RowMajorMatrix::new(row.to_vec(), 8);
    let gpu = leaf_hash_matrix(&mat);
    assert_eq!(gpu.len(), 1);
    assert_eq!(gpu[0], expected);
}

#[test]
fn leaf_hash_matches_cpu_multiple_rate_blocks() {
    // Width 32 — spans two full rate-12 blocks plus a width-8 partial.
    // Exercises the partial-block "extra permute" branch in the kernel.
    let width = 32;
    let height = 5;
    let mat = fixture(height, width);
    let cpu_digests: Vec<[Goldilocks; 4]> = (0..height)
        .map(|r| {
            let row: Vec<Goldilocks> = (0..width)
                .map(|c| mat.values[r * width + c])
                .collect();
            cpu_hash().hash_iter(row.into_iter())
        })
        .collect();
    let gpu = leaf_hash_matrix(&mat);
    assert_eq!(gpu, cpu_digests);
}

#[test]
fn leaf_hash_matches_cpu_clean_block_boundary() {
    // Width 12 — exactly one rate block, no trailing permute.
    let width = 12;
    let height = 3;
    let mat = fixture(height, width);
    let cpu_digests: Vec<[Goldilocks; 4]> = (0..height)
        .map(|r| {
            let row: Vec<Goldilocks> = (0..width)
                .map(|c| mat.values[r * width + c])
                .collect();
            cpu_hash().hash_iter(row.into_iter())
        })
        .collect();
    let gpu = leaf_hash_matrix(&mat);
    assert_eq!(gpu, cpu_digests);
}

#[test]
fn compress_pairs_matches_cpu() {
    let digests = fixture_digests(8);
    let cpu: Vec<Digest> = digests
        .chunks_exact(2)
        .map(|chunk| cpu_compress().compress([chunk[0], chunk[1]]))
        .collect();
    let gpu = compress_pairs(&digests);
    assert_eq!(gpu, cpu);
}

#[test]
fn compress_paired_matches_cpu() {
    let left = fixture_digests(6);
    let right: Vec<Digest> = (0..6)
        .map(|i| {
            [
                Goldilocks::from_u64((100 + 4 * i) as u64),
                Goldilocks::from_u64((100 + 4 * i + 1) as u64),
                Goldilocks::from_u64((100 + 4 * i + 2) as u64),
                Goldilocks::from_u64((100 + 4 * i + 3) as u64),
            ]
        })
        .collect();
    let cpu: Vec<Digest> = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| cpu_compress().compress([*l, *r]))
        .collect();
    let gpu = compress_paired(&left, &right);
    assert_eq!(gpu, cpu);
}

#[test]
fn build_merkle_tree_single_matrix_matches_cpu() {
    // Single matrix, height 8, width 4. Compute the tree manually on
    // CPU using `Hash` + `Compress` and compare layer by layer.
    let mat = fixture(8, 4);

    // Reference: hash each row, then pair-compress until root.
    let mut cpu_layers: Vec<Vec<Digest>> = Vec::new();
    let mut layer: Vec<Digest> = (0..mat.height())
        .map(|r| {
            let row: Vec<Goldilocks> = (0..mat.width())
                .map(|c| mat.values[r * mat.width() + c])
                .collect();
            cpu_hash().hash_iter(row.into_iter())
        })
        .collect();
    cpu_layers.push(layer.clone());
    while layer.len() > 1 {
        layer = layer
            .chunks_exact(2)
            .map(|chunk| cpu_compress().compress([chunk[0], chunk[1]]))
            .collect();
        cpu_layers.push(layer.clone());
    }

    let gpu_layers = build_merkle_tree(std::slice::from_ref(&mat));
    assert_eq!(gpu_layers.len(), cpu_layers.len(), "layer count mismatch");
    for (i, (gpu, cpu)) in gpu_layers.iter().zip(cpu_layers.iter()).enumerate() {
        assert_eq!(gpu, cpu, "layer {i} mismatch");
    }
}

#[test]
fn leaf_hash_partial_block_widths() {
    // Sweep widths around rate boundaries to exercise the kernel's
    // partial-block branch. Rate is 12, so the interesting widths are
    // 1 (much smaller than rate), 11 (rate − 1), 12 (exact rate),
    // 13 (rate + 1), 23 (one rate + rate − 1), 24 (exactly two rate
    // blocks, no trailing permute).
    for &width in &[1usize, 11, 12, 13, 23, 24] {
        let height = 4;
        let mat = fixture(height, width);
        let cpu_digests: Vec<[Goldilocks; 4]> = (0..height)
            .map(|r| {
                let row: Vec<Goldilocks> = (0..width)
                    .map(|c| mat.values[r * width + c])
                    .collect();
                cpu_hash().hash_iter(row.into_iter())
            })
            .collect();
        let gpu = leaf_hash_matrix(&mat);
        assert_eq!(
            gpu, cpu_digests,
            "leaf-hash GPU/CPU mismatch at width {width}"
        );
    }
}

#[test]
fn build_merkle_tree_two_matrices_matches_cpu() {
    // Multi-matrix tree: A is height 8 width 4, B is height 2 width 6.
    // Tree topology (per Plonky3 MerkleTreeMmcs):
    //   layer 0: leaf_hash(A) — 8 digests
    //   layer 1: pair-compress  — 4
    //   layer 2: pair-compress  — 2 (pre-mix)
    //   layer 3: mix-compress with leaf_hash(B) — 2 (post-mix)
    //   layer 4: pair-compress  — 1 (root)
    let mat_a = fixture(8, 4);
    let mat_b_values: Vec<Goldilocks> = (0..2 * 6)
        .map(|i| Goldilocks::from_u64(1000 + i as u64))
        .collect();
    let mat_b = RowMajorMatrix::new(mat_b_values, 6);

    // Reference computation, replicating the multi-matrix algorithm
    // step-by-step on CPU.
    let leaves_a: Vec<Digest> = (0..mat_a.height())
        .map(|r| {
            let row: Vec<Goldilocks> = (0..mat_a.width())
                .map(|c| mat_a.values[r * mat_a.width() + c])
                .collect();
            cpu_hash().hash_iter(row.into_iter())
        })
        .collect();
    let leaves_b: Vec<Digest> = (0..mat_b.height())
        .map(|r| {
            let row: Vec<Goldilocks> = (0..mat_b.width())
                .map(|c| mat_b.values[r * mat_b.width() + c])
                .collect();
            cpu_hash().hash_iter(row.into_iter())
        })
        .collect();

    let layer1: Vec<Digest> = leaves_a
        .chunks_exact(2)
        .map(|c| cpu_compress().compress([c[0], c[1]]))
        .collect();
    let layer2_pre: Vec<Digest> = layer1
        .chunks_exact(2)
        .map(|c| cpu_compress().compress([c[0], c[1]]))
        .collect();
    let layer3_mixed: Vec<Digest> = layer2_pre
        .iter()
        .zip(leaves_b.iter())
        .map(|(l, r)| cpu_compress().compress([*l, *r]))
        .collect();
    let layer4_root: Vec<Digest> = layer3_mixed
        .chunks_exact(2)
        .map(|c| cpu_compress().compress([c[0], c[1]]))
        .collect();
    let cpu_layers = vec![leaves_a, layer1, layer2_pre, layer3_mixed, layer4_root];

    let gpu_layers = build_merkle_tree(&[mat_a, mat_b]);
    assert_eq!(
        gpu_layers.len(),
        cpu_layers.len(),
        "multi-matrix tree layer count mismatch"
    );
    for (i, (gpu, cpu)) in gpu_layers.iter().zip(cpu_layers.iter()).enumerate() {
        assert_eq!(gpu, cpu, "multi-matrix layer {i} mismatch");
    }
}

#[test]
fn build_merkle_tree_three_matrices_matches_cpu() {
    // Three-matrix variant exercises two distinct mix-compress steps in
    // sequence: A height 8, B height 4, C height 1.
    //
    // Topology:
    //   layer 0: leaf_hash(A)               — 8
    //   layer 1: pair-compress              — 4 (pre-mix with B)
    //   layer 2: mix-compress with H(B)     — 4 (post-mix B)
    //   layer 3: pair-compress              — 2 (pre-mix with C — but C has h=1)
    //   layer 4: pair-compress              — 1 (= 2*h_C; mix with H(C) follows)
    //   ... wait, h_C = 1, so we need 2*1 = 2 pre-mix nodes, then compress
    //       the 2 to 1 (which would be the post-mix layer with single C row).
    //
    // Easier: pick heights 8, 4, 2 and compute step-by-step.
    let mat_a = fixture(8, 4);
    let mat_b_values: Vec<Goldilocks> = (0..4 * 5)
        .map(|i| Goldilocks::from_u64(2000 + i as u64))
        .collect();
    let mat_b = RowMajorMatrix::new(mat_b_values, 5);
    let mat_c_values: Vec<Goldilocks> = (0..2 * 3)
        .map(|i| Goldilocks::from_u64(3000 + i as u64))
        .collect();
    let mat_c = RowMajorMatrix::new(mat_c_values, 3);

    let hash_rows = |mat: &RowMajorMatrix<Goldilocks>| -> Vec<Digest> {
        (0..mat.height())
            .map(|r| {
                let row: Vec<Goldilocks> = (0..mat.width())
                    .map(|c| mat.values[r * mat.width() + c])
                    .collect();
                cpu_hash().hash_iter(row.into_iter())
            })
            .collect()
    };
    let pair = |layer: &[Digest]| -> Vec<Digest> {
        layer
            .chunks_exact(2)
            .map(|c| cpu_compress().compress([c[0], c[1]]))
            .collect()
    };
    let mix = |left: &[Digest], right: &[Digest]| -> Vec<Digest> {
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| cpu_compress().compress([*l, *r]))
            .collect()
    };

    let l0 = hash_rows(&mat_a); // 8
    let l1 = pair(&l0); // 4 (pre-mix B at h=4)
    let l2 = mix(&l1, &hash_rows(&mat_b)); // 4 (post-mix B)
    let l3 = pair(&l2); // 2 (pre-mix C at h=2)
    let l4 = mix(&l3, &hash_rows(&mat_c)); // 2 (post-mix C)
    let l5 = pair(&l4); // 1 (root)
    let cpu_layers = vec![l0, l1, l2, l3, l4, l5];

    let gpu_layers = build_merkle_tree(&[mat_a, mat_b, mat_c]);
    assert_eq!(
        gpu_layers.len(),
        cpu_layers.len(),
        "three-matrix tree layer count mismatch"
    );
    for (i, (gpu, cpu)) in gpu_layers.iter().zip(cpu_layers.iter()).enumerate() {
        assert_eq!(gpu, cpu, "three-matrix layer {i} mismatch");
    }
}

#[test]
fn build_merkle_tree_height_one() {
    // Single matrix, single row. Tree is just the leaf hash; no compression.
    let mat = RowMajorMatrix::new(
        Goldilocks::new_array([1, 2, 3, 4, 5]).to_vec(),
        5,
    );
    let gpu_layers = build_merkle_tree(std::slice::from_ref(&mat));
    assert_eq!(gpu_layers.len(), 1, "height-1 tree should have one layer");
    assert_eq!(gpu_layers[0].len(), 1, "single root digest");

    let row: Vec<Goldilocks> = (0..5).map(|c| mat.values[c]).collect();
    let expected = cpu_hash().hash_iter(row.into_iter());
    assert_eq!(gpu_layers[0][0], expected);
}

#[test]
fn build_merkle_tree_unbalanced_columns() {
    // Multi-matrix with widely different column counts. Catches any
    // accidental coupling between matrix index and column stride.
    let mat_a = fixture(4, 32); // wide
    let mat_b_values: Vec<Goldilocks> = (0..2 * 1)
        .map(|i| Goldilocks::from_u64(7000 + i as u64))
        .collect();
    let mat_b = RowMajorMatrix::new(mat_b_values, 1); // narrow

    let hash_rows = |mat: &RowMajorMatrix<Goldilocks>| -> Vec<Digest> {
        (0..mat.height())
            .map(|r| {
                let row: Vec<Goldilocks> = (0..mat.width())
                    .map(|c| mat.values[r * mat.width() + c])
                    .collect();
                cpu_hash().hash_iter(row.into_iter())
            })
            .collect()
    };
    let pair = |layer: &[Digest]| -> Vec<Digest> {
        layer
            .chunks_exact(2)
            .map(|c| cpu_compress().compress([c[0], c[1]]))
            .collect()
    };
    let mix = |left: &[Digest], right: &[Digest]| -> Vec<Digest> {
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| cpu_compress().compress([*l, *r]))
            .collect()
    };

    let l0 = hash_rows(&mat_a); // 4
    let l1 = pair(&l0); // 2 (pre-mix B at h=2)
    let l2 = mix(&l1, &hash_rows(&mat_b)); // 2
    let l3 = pair(&l2); // 1 root
    let cpu_layers = vec![l0, l1, l2, l3];

    let gpu_layers = build_merkle_tree(&[mat_a, mat_b]);
    assert_eq!(gpu_layers.len(), cpu_layers.len());
    for (i, (gpu, cpu)) in gpu_layers.iter().zip(cpu_layers.iter()).enumerate() {
        assert_eq!(gpu, cpu, "unbalanced-columns layer {i} mismatch");
    }
}

// ---------- End-to-end: GPU DFT → GPU leaf hash ----------------------
//
// The integration sanity check: take a fresh trace, run the GPU LDE,
// hash the LDE rows on GPU, compare with the equivalent CPU pipeline.
// If this passes, every other piece is wired correctly.

#[test]
fn end_to_end_lde_then_leaf_hash_matches_cpu() {
    let mat = fixture(1 << 8, 4);
    let added_bits = 1;
    let shift = Goldilocks::GENERATOR;

    // CPU pipeline
    let cpu_lde = Radix2DitParallel::<Goldilocks>::default()
        .coset_lde_batch(mat.clone(), added_bits, shift)
        .to_row_major_matrix();
    let cpu_digests: Vec<[Goldilocks; 4]> = (0..cpu_lde.height())
        .map(|r| {
            let row: Vec<Goldilocks> = (0..cpu_lde.width())
                .map(|c| cpu_lde.values[r * cpu_lde.width() + c])
                .collect();
            cpu_hash().hash_iter(row.into_iter())
        })
        .collect();

    // GPU pipeline
    let gpu_lde = CudaRadix2Dft::<Goldilocks>::default()
        .coset_lde_batch(mat, added_bits, shift)
        .to_row_major_matrix();
    let gpu_digests = leaf_hash_matrix(&gpu_lde);

    assert_eq!(gpu_lde.values, cpu_lde.values, "LDE values diverge");
    assert_eq!(gpu_digests, cpu_digests, "leaf-hash digests diverge");
}
