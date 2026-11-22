//! Protocol `v6-zk` zero-knowledge cryptographic primitives, BLAKE3-CTR leaf
//! salting, odd-coset random row interleaving, telescoping quotient blinding,
//! and hiding Merkle verification shared across CPU and CUDA backends.

use core::cmp::Reverse;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_merkle_tree::{MerkleTreeError, PrunedMerklePaths, PrunedProofError};
use p3_symmetric::MerkleCap;
use p3_util::{log2_ceil_usize, log2_strict_usize, reverse_bits_len};
use serde::{Deserialize, Serialize};

use super::cap::verify_merkle_cap_len;
use super::domain_blake3::Blake3Domain;
use crate::types::Val;

/// Per-leaf salt length in bytes (128 bits = 16 bytes) for `v6-zk` hiding
/// Merkle commitments.
pub const ZK_LEAF_SALT_BYTES: usize = 16;

/// Minimum and default number of base-field (`Goldilocks`) columns in the
/// randomizer polynomial `R(X)` committed inside the quotient MMCS.
///
/// Under `FriBatching::Tensor` over the cubic extension $\mathbb{F}_{p^3}$,
/// 3 columns (`1, r_0, r_1`) fail to span $\mathbb{F}_{p^3}$ with probability
/// $1/p \approx 2^{-64}$. Using `W_R >= 4` columns (`1, r_0, r_1, r_0 r_1`)
/// reduces span-failure probability to $\le 2/p^2 \approx 2^{-127}$.
pub const ZK_RANDOMIZER_WIDTH: usize = 4;

/// Context string used to derive the deterministic public salt seed for
/// preprocessed trace commitments in `v6-zk`.
///
/// Preprocessed traces are public setup data; independent prover and verifier
/// setups MUST derive identical `preprocessed_commit` roots while sharing the
/// uniform hiding MMCS verification format.
const PREPROCESSED_SALT_CONTEXT: &str = "multi-stark/v6-zk/preprocessed-salt";

/// Draws a fresh 256-bit cryptographically secure random seed from the OS
/// CSPRNG (`getrandom`).
///
/// Never stored in the proof or observed into the Fiat-Shamir transcript.
pub fn random_seed() -> [u8; 32] {
    let mut seed = [0u8; 32];
    getrandom::fill(&mut seed).expect("OS CSPRNG unavailable for ZK randomness");
    seed
}

/// Returns the deterministic 256-bit salt seed used solely for public
/// preprocessed commitments so independent prover and verifier `System::new`
/// calls produce identical `preprocessed_commit` digests.
pub fn preprocessed_salt_seed() -> [u8; 32] {
    blake3::derive_key(PREPROCESSED_SALT_CONTEXT, b"preprocessed")
}

/// Deterministically derives the 128-bit leaf salt for `(height, row_index)`
/// on the fly from `salt_seed` via single-block keyed BLAKE3:
///
/// ```text
/// salt = BLAKE3_keyed(salt_seed, height_u64_le || row_index_u64_le)[0..16]
/// ```
///
/// Because this is a single 16-byte keyed BLAKE3 compression, both CPU and
/// CUDA compute leaf salts on the fly during hashing and opening with **zero**
/// salt storage in RAM, VRAM, or disk.
#[inline]
pub fn derive_leaf_salt(
    salt_seed: &[u8; 32],
    height: usize,
    row_index: usize,
) -> [u8; ZK_LEAF_SALT_BYTES] {
    let mut msg = [0u8; 16];
    msg[..8].copy_from_slice(
        &u64::try_from(height)
            .expect("height exceeds u64")
            .to_le_bytes(),
    );
    msg[8..16].copy_from_slice(
        &u64::try_from(row_index)
            .expect("row_index exceeds u64")
            .to_le_bytes(),
    );
    let mut hasher = blake3::Hasher::new_keyed(salt_seed);
    hasher.update(&msg);
    let digest = hasher.finalize();
    let mut salt = [0u8; ZK_LEAF_SALT_BYTES];
    salt.copy_from_slice(&digest.as_bytes()[..ZK_LEAF_SALT_BYTES]);
    salt
}

/// Samples `count` uniformly distributed canonical Goldilocks field elements
/// using a fresh 256-bit OS CSPRNG seed expanded via BLAKE3 XOF with exact
/// rejection sampling (`< Goldilocks::ORDER_U64`, zero modulo bias).
pub fn csprng_field_vec(count: usize) -> Vec<Val> {
    if count == 0 {
        return Vec::new();
    }
    let seed = random_seed();
    csprng_field_vec_from_seed(&seed, count)
}

/// Deterministic expansion helper behind [`csprng_field_vec`].
pub(crate) fn csprng_field_vec_from_seed(seed: &[u8; 32], count: usize) -> Vec<Val> {
    const CHUNK_ELEMS: usize = 4096;
    let mut out = vec![Val::ZERO; count];
    let fill_chunk = |chunk_idx: usize, dst: &mut [Val]| {
        let mut hasher = blake3::Hasher::new_keyed(seed);
        hasher.update(
            &u64::try_from(chunk_idx)
                .expect("chunk index exceeds u64")
                .to_le_bytes(),
        );
        let mut xof = hasher.finalize_xof();
        let mut buf = [0u8; 512];
        let mut pos = buf.len();
        for slot in dst.iter_mut() {
            loop {
                if pos == buf.len() {
                    xof.fill(&mut buf);
                    pos = 0;
                }
                let word = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
                pos += 8;
                if word < Val::ORDER_U64 {
                    *slot = Val::from_u64(word);
                    break;
                }
            }
        }
    };
    if count <= CHUNK_ELEMS {
        fill_chunk(0, &mut out);
    } else {
        out.par_chunks_mut(CHUNK_ELEMS)
            .enumerate()
            .for_each(|(chunk_idx, dst)| fill_chunk(chunk_idx, dst));
    }
    out
}

/// Samples `count` uniformly random extension field elements using the OS CSPRNG.
pub fn csprng_extension_vec<EF: ExtensionField<Val>>(count: usize) -> Vec<EF> {
    let d = EF::DIMENSION;
    let base = csprng_field_vec(count * d);
    base.chunks_exact(d)
        .map(|coords| {
            EF::from_basis_coefficients_slice(coords).expect("extension basis coefficient length")
        })
        .collect()
}

/// Minimum admissible `log2(trace_height)` for Protocol `v6-zk` (Trap 1).
///
/// In 2× odd-coset blinding (`T(X) = T_0(X) + Z_H(X) b(X)` with `deg(b) < N`),
/// `Q` FRI queries plus out-of-domain openings at `(zeta, zeta * g)` impose up
/// to `2 * Q + 2 * D` base-field linear constraints on `b` (each query can
/// expose `T(x_j)` directly and `T(g x_j)` indirectly via the quotient
/// relation, and each OOD opening over $\mathbb{F}_{p^D}$ imposes `D` base-field
/// equations). Enforcing `N >= (2 * Q + 2 * D).next_power_of_two()` strictly
/// exceeds both `next_pow2(Q + 2)` and the full transition + OOD degree budget.
#[inline]
pub const fn min_zk_log_degree(num_queries: usize, ext_degree: usize) -> usize {
    let min_rows = (2 * num_queries + 2 * ext_degree).next_power_of_two();
    log2_strict_usize(min_rows)
}

/// Doubles `mat` (height `N`, width `W`) to height `2N` by interleaving
/// witness rows on even indices (`2r`) and uniform CSPRNG blinding rows on odd
/// indices (`2r + 1`), matching the ePrint 2024/1037 / Plonky3 odd-coset
/// blinding polynomial `T(X) = T_0(X) + Z_H(X) b(X)`.
pub fn interleave_random_rows(mat: &RowMajorMatrix<Val>) -> RowMajorMatrix<Val> {
    let height = mat.height();
    let width = mat.width();
    if width == 0 {
        return RowMajorMatrix::new(Vec::new(), 0);
    }
    let rand_vals = csprng_field_vec(height * width);
    let mut out = vec![Val::ZERO; 2 * height * width];
    out.par_chunks_mut(2 * width)
        .enumerate()
        .for_each(|(r, pair)| {
            let src = &mat.values[r * width..(r + 1) * width];
            let rnd = &rand_vals[r * width..(r + 1) * width];
            pair[..width].copy_from_slice(src);
            pair[width..].copy_from_slice(rnd);
        });
    RowMajorMatrix::new(out, width)
}

/// Telescoping half-stride ZK quotient slicing with the committed LDE's
/// `GENERATOR` shift folded in.
///
/// Given evaluations of the quotient polynomial `Q(X)` (degree `< 2N * q_zk`)
/// on the disjoint quotient domain `GENERATOR * H_{2N * q_zk}` (flattened to
/// `ext_degree` base columns), decomposes
///
/// ```text
/// Q(X) = sum_{k=0}^{2q - 1} X^{k N} c_k(X),   deg(c_k) < N
/// ```
///
/// samples `2q - 1` uniform base-field blinding polynomials `rho_0..rho_{2q-2}`
/// of degree `< N` per extension coordinate, and forms `2q` blinded slices of
/// degree `< 2N`:
///
/// ```text
/// c'_k(X) = c_k(X) - rho_{k-1}(X) + X^N rho_k(X)   (rho_{-1} = rho_{2q-1} = 0)
/// ```
///
/// Telescoping guarantees `sum_{k=0}^{2q-1} X^{k N} c'_k(X) == Q(X)` identically
/// so verifier OOD recombination at `zeta` uses `zeta^N` with `2 * q_zk` slices.
///
/// Returns a `(2N) × (2 * q_zk * ext_degree)` matrix of `GENERATOR`-shifted
/// slice coefficients ready for `lde_from_shifted_coefficients`.
pub(crate) fn shifted_quotient_slices_zk<F: TwoAdicField + Ord>(
    dft: &impl TwoAdicSubgroupDft<F>,
    quotient_evals: RowMajorMatrix<F>,
    domain_shift: F,
    quotient_degree: usize,
) -> RowMajorMatrix<F>
where
    Val: Into<F>,
{
    assert_eq!(
        domain_shift,
        F::GENERATOR,
        "quotient domain shift must equal the LDE shift for the scalings to cancel"
    );
    let ext_degree = quotient_evals.width();
    let big_height = quotient_evals.height();
    let log_big_height = log2_strict_usize(big_height);
    let num_slices = 2 * quotient_degree;
    assert!(
        num_slices >= 2 && big_height.is_multiple_of(num_slices),
        "ZK quotient domain height {big_height} must be a multiple of 2 * q_zk ({num_slices})"
    );
    let n = big_height / num_slices;
    let width = num_slices * ext_degree;

    let storage = dft
        .dft_batch(quotient_evals)
        .bit_reverse_rows()
        .to_row_major_matrix();
    let n_inv = F::ONE.div_2exp_u64(u64::try_from(log_big_height).expect("log height exceeds u64"));
    let gen_n = F::GENERATOR.exp_u64(u64::try_from(n).expect("trace height exceeds u64"));
    let weight_step = gen_n.inverse();
    let weights: Vec<F> = weight_step
        .powers()
        .take(num_slices)
        .map(|w| w * n_inv)
        .collect();

    // Pre-scaled blinding coefficients `tilde_rho[k][r][c] = GENERATOR^r * rho_k[r][c]`.
    // Because `GENERATOR != 0`, sampling `tilde_rho` uniformly is identically
    // distributed to sampling `rho` uniformly and scaling row `r` by `GENERATOR^r`.
    let num_blinders = num_slices - 1;
    let rand_base = csprng_field_vec(num_blinders * n * ext_degree);
    let tilde_rho: Vec<F> = rand_base.into_iter().map(Into::into).collect();

    let mut values = F::zero_vec(2 * n * width);
    let (low_rows, high_rows) = values.split_at_mut(n * width);

    low_rows
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out)| {
            for (slice_idx, weight) in weights.iter().enumerate() {
                let j = slice_idx * n + row;
                let src_idx = reverse_bits_len(
                    big_height.wrapping_sub(j) & (big_height - 1),
                    log_big_height,
                );
                let src = &storage.values[src_idx * ext_degree..(src_idx + 1) * ext_degree];
                let dst = &mut out[slice_idx * ext_degree..(slice_idx + 1) * ext_degree];
                if slice_idx == 0 {
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d = s * *weight;
                    }
                } else {
                    let prev_rho_off = ((slice_idx - 1) * n + row) * ext_degree;
                    let prev_rho = &tilde_rho[prev_rho_off..prev_rho_off + ext_degree];
                    for ((d, &s), &r) in dst.iter_mut().zip(src).zip(prev_rho) {
                        *d = s * *weight - r;
                    }
                }
            }
        });

    high_rows
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out)| {
            for slice_idx in 0..num_blinders {
                let rho_off = (slice_idx * n + row) * ext_degree;
                let rho = &tilde_rho[rho_off..rho_off + ext_degree];
                let dst = &mut out[slice_idx * ext_degree..(slice_idx + 1) * ext_degree];
                for (d, &r) in dst.iter_mut().zip(rho) {
                    *d = gen_n * r;
                }
            }
            // Final slice `num_slices - 1` has degree < n (high rows stay zero).
        });

    RowMajorMatrix::new(values, width)
}

/// Single-query hiding Merkle opening proof carrying one 128-bit leaf salt per
/// height group encountered along the authentication path plus the sibling
/// hashes up to the Merkle cap.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HidingBatchProof {
    pub salts: Vec<[u8; ZK_LEAF_SALT_BYTES]>,
    pub siblings: Vec<[u8; 32]>,
}

/// Multi-query pruned hiding Merkle opening proof carrying deduplicated
/// height-group leaf salts (ordered tallest-first, ascending unique node index
/// within each height group) plus pruned binary Merkle sibling hashes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HidingMultiProof {
    pub salts: Vec<Vec<[u8; ZK_LEAF_SALT_BYTES]>>,
    pub pruned: PrunedMerklePaths<u8, 32>,
}

impl PartialEq for HidingMultiProof {
    fn eq(&self, other: &Self) -> bool {
        self.salts == other.salts && self.pruned.sibling_hashes == other.pruned.sibling_hashes
    }
}

impl Eq for HidingMultiProof {}

/// One distinct height group within a mixed-height Merkle tree.
#[derive(Clone, Debug)]
pub(crate) struct HeightGroup {
    pub(crate) height: usize,
    pub(crate) level: usize,
    pub(crate) matrices: Vec<usize>,
}

/// Validates reachable matrix heights (`ceil(max_height / 2^k)`) and groups
/// matrix indices stably by strictly descending height.
pub(crate) fn height_groups_tallest_first(
    heights: &[usize],
) -> Result<(usize, Vec<HeightGroup>), MerkleTreeError> {
    let max_height = heights
        .iter()
        .copied()
        .max()
        .filter(|&h| h > 0)
        .ok_or(MerkleTreeError::EmptyBatch)?;
    let log_max_height = log2_ceil_usize(max_height);
    for &h in heights {
        if h == 0 {
            return Err(MerkleTreeError::IncompatibleHeights {
                height: 0,
                max_height,
                expected_height: 1,
                bits_reduced: log_max_height,
            });
        }
        let bits_reduced = log_max_height - log2_ceil_usize(h);
        let expected_height = ((max_height - 1) >> bits_reduced) + 1;
        if h != expected_height {
            return Err(MerkleTreeError::IncompatibleHeights {
                height: h,
                max_height,
                expected_height,
                bits_reduced,
            });
        }
    }

    let mut order: Vec<usize> = (0..heights.len()).collect();
    order.sort_by_key(|&idx| Reverse(heights[idx]));

    let mut groups: Vec<HeightGroup> = Vec::new();
    let mut cursor = 0usize;
    while cursor < order.len() {
        let h = heights[order[cursor]];
        let start = cursor;
        while cursor < order.len() && heights[order[cursor]] == h {
            cursor += 1;
        }
        let level = log_max_height - log2_ceil_usize(h);
        groups.push(HeightGroup {
            height: h,
            level,
            matrices: order[start..cursor].to_vec(),
        });
    }
    Ok((max_height, groups))
}

/// Computes the single-query leaf salts on the fly from `salt_seed` (zero
/// salt storage).
pub(crate) fn open_hiding_batch_salts(
    salt_seed: &[u8; 32],
    heights: &[usize],
    index: usize,
    cap_height: usize,
) -> Vec<[u8; ZK_LEAF_SALT_BYTES]> {
    let (max_height, groups) = height_groups_tallest_first(heights).expect("valid tree heights");
    let log_max_height = log2_ceil_usize(max_height);
    let effective_cap_height = cap_height.min(log_max_height);
    let proof_levels = log_max_height - effective_cap_height;
    groups
        .iter()
        .filter(|g| g.level <= proof_levels)
        .map(|g| derive_leaf_salt(salt_seed, g.height, index >> g.level))
        .collect()
}

/// Computes the pruned multi-batch leaf salts on the fly from `salt_seed`
/// (zero salt storage), in tallest-first height-group order and ascending
/// unique node index order within each height group.
pub(crate) fn open_hiding_multi_salts(
    salt_seed: &[u8; 32],
    heights: &[usize],
    indices: &[usize],
    cap_height: usize,
) -> Vec<Vec<[u8; ZK_LEAF_SALT_BYTES]>> {
    if indices.is_empty() {
        return Vec::new();
    }
    let (max_height, groups) = height_groups_tallest_first(heights).expect("valid tree heights");
    let log_max_height = log2_ceil_usize(max_height);
    let effective_cap_height = cap_height.min(log_max_height);
    let proof_levels = log_max_height - effective_cap_height;

    groups
        .iter()
        .filter(|g| g.level <= proof_levels)
        .map(|g| {
            let mut nodes: Vec<usize> = indices.iter().map(|&idx| idx >> g.level).collect();
            nodes.sort_unstable();
            nodes.dedup();
            nodes
                .into_iter()
                .map(|node_idx| derive_leaf_salt(salt_seed, g.height, node_idx))
                .collect()
        })
        .collect()
}

#[inline]
fn hash_leaf_rows_with_salt<R: AsRef<[Val]>>(
    leaf_domain: Blake3Domain,
    opened_rows: &[R],
    matrix_indices: &[usize],
    salt: &[u8; ZK_LEAF_SALT_BYTES],
) -> [u8; 32] {
    let mut hasher = leaf_domain.hasher();
    for &mat_idx in matrix_indices {
        for &elem in opened_rows[mat_idx].as_ref() {
            hasher.update(&elem.as_canonical_u64().to_le_bytes());
        }
    }
    hasher.update(salt);
    hasher.finalize().into()
}

#[inline]
fn compress_pair_domain(
    compress_domain: Blake3Domain,
    left: &[u8; 32],
    right: &[u8; 32],
) -> [u8; 32] {
    let mut block = [0u8; 64];
    block[..32].copy_from_slice(left);
    block[32..].copy_from_slice(right);
    compress_domain.hash_bytes(&block)
}

/// Verifies a single-query salted binary Merkle batch opening, enforcing exact
/// `C2` `MerkleCap` root count, exact row widths, exact salt count, and exact
/// sibling count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_hiding_batch(
    cap_height: usize,
    leaf_domain: Blake3Domain,
    compress_domain: Blake3Domain,
    commit: &MerkleCap<Val, [u8; 32]>,
    dimensions: &[Dimensions],
    index: usize,
    opened_values: &[Vec<Val>],
    proof: &HidingBatchProof,
) -> Result<(), MerkleTreeError> {
    verify_merkle_cap_len(cap_height, commit, dimensions)?;
    if dimensions.len() != opened_values.len() {
        return Err(MerkleTreeError::WrongBatchSize);
    }
    for (matrix, (dim, row)) in dimensions.iter().zip(opened_values).enumerate() {
        if row.len() != dim.width {
            return Err(MerkleTreeError::WrongWidth {
                matrix,
                expected: dim.width,
                got: row.len(),
            });
        }
    }
    let heights: Vec<usize> = dimensions.iter().map(|d| d.height).collect();
    let (max_height, groups) = height_groups_tallest_first(&heights)?;
    if index >= max_height {
        return Err(MerkleTreeError::IndexOutOfBounds { max_height, index });
    }
    let log_max_height = log2_ceil_usize(max_height);
    let effective_cap_height = cap_height.min(log_max_height);
    let proof_levels = log_max_height - effective_cap_height;

    if proof.siblings.len() != proof_levels {
        return Err(MerkleTreeError::WrongHeight {
            expected_proof_len: proof_levels,
            num_siblings: proof.siblings.len(),
        });
    }
    let expected_salt_groups = groups.iter().filter(|g| g.level <= proof_levels).count();
    if proof.salts.len() != expected_salt_groups {
        return Err(MerkleTreeError::WrongBatchSize);
    }

    let mut digest = hash_leaf_rows_with_salt(
        leaf_domain,
        opened_values,
        &groups[0].matrices,
        &proof.salts[0],
    );
    let mut group_cursor = 1usize;
    let mut salt_cursor = 1usize;
    let mut idx = index;

    for (level, sibling) in proof.siblings.iter().enumerate() {
        digest = if idx & 1 == 0 {
            compress_pair_domain(compress_domain, &digest, sibling)
        } else {
            compress_pair_domain(compress_domain, sibling, &digest)
        };
        idx >>= 1;
        let next_level = level + 1;
        if group_cursor < groups.len() && groups[group_cursor].level == next_level {
            let injected = hash_leaf_rows_with_salt(
                leaf_domain,
                opened_values,
                &groups[group_cursor].matrices,
                &proof.salts[salt_cursor],
            );
            digest = compress_pair_domain(compress_domain, &digest, &injected);
            group_cursor += 1;
            salt_cursor += 1;
        }
    }

    if idx < commit.num_roots() && commit[idx] == digest {
        Ok(())
    } else {
        Err(MerkleTreeError::CapMismatch)
    }
}

/// Verifies a pruned multi-query salted binary Merkle batch opening, enforcing
/// exact `C2` `MerkleCap` root count, exact row widths, duplicate/merged row
/// consistency, exact deduplicated salt counts, and exact pruned sibling count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_hiding_multi_batch<R: AsRef<[Val]> + PartialEq>(
    cap_height: usize,
    leaf_domain: Blake3Domain,
    compress_domain: Blake3Domain,
    commit: &MerkleCap<Val, [u8; 32]>,
    dimensions: &[Dimensions],
    indices: &[usize],
    opened_values: &[Vec<R>],
    proof: &HidingMultiProof,
) -> Result<(), MerkleTreeError> {
    verify_merkle_cap_len(cap_height, commit, dimensions)?;
    let heights: Vec<usize> = dimensions.iter().map(|d| d.height).collect();
    let (max_height, groups) = height_groups_tallest_first(&heights)?;

    if indices.len() != opened_values.len() {
        return Err(MerkleTreeError::WrongBatchSize);
    }
    if indices.is_empty() {
        if !proof.salts.is_empty() {
            return Err(MerkleTreeError::WrongBatchSize);
        }
        if !proof.pruned.sibling_hashes.is_empty() {
            return Err(MerkleTreeError::MalformedPrunedProof(
                PrunedProofError::SiblingCountMismatch {
                    expected: 0,
                    got: proof.pruned.sibling_hashes.len(),
                },
            ));
        }
        return Ok(());
    }

    for (q_idx, (&leaf_idx, query_rows)) in indices.iter().zip(opened_values).enumerate() {
        let _ = q_idx;
        if leaf_idx >= max_height {
            return Err(MerkleTreeError::IndexOutOfBounds {
                max_height,
                index: leaf_idx,
            });
        }
        if query_rows.len() != dimensions.len() {
            return Err(MerkleTreeError::WrongBatchSize);
        }
        for (matrix, (dim, row)) in dimensions.iter().zip(query_rows).enumerate() {
            if row.as_ref().len() != dim.width {
                return Err(MerkleTreeError::WrongWidth {
                    matrix,
                    expected: dim.width,
                    got: row.as_ref().len(),
                });
            }
        }
    }

    let log_max_height = log2_ceil_usize(max_height);
    let effective_cap_height = cap_height.min(log_max_height);
    let proof_levels = log_max_height - effective_cap_height;

    let expected_salt_groups = groups.iter().filter(|g| g.level <= proof_levels).count();
    if proof.salts.len() != expected_salt_groups {
        return Err(MerkleTreeError::WrongBatchSize);
    }

    // Group queries by leaf index in ascending order, verifying that duplicate
    // queries agree on all opened rows.
    let mut query_order: Vec<(usize, usize)> = indices
        .iter()
        .enumerate()
        .map(|(q, &idx)| (idx, q))
        .collect();
    query_order.sort_unstable_by_key(|&(idx, _)| idx);

    struct FrontierNode {
        index: usize,
        digest: [u8; 32],
        queries: Vec<usize>,
    }

    let mut unique_leaves: Vec<(usize, Vec<usize>)> = Vec::new();
    for (leaf_idx, q) in query_order {
        if let Some((last_idx, qs)) = unique_leaves.last_mut()
            && *last_idx == leaf_idx
        {
            let lead = qs[0];
            if opened_values[q] != opened_values[lead] {
                let slot = unique_leaves.len() - 1;
                return Err(MerkleTreeError::MalformedPrunedProof(
                    PrunedProofError::InconsistentDuplicateOpenings { slot },
                ));
            }
            qs.push(q);
        } else {
            unique_leaves.push((leaf_idx, vec![q]));
        }
    }

    if proof.salts[0].len() != unique_leaves.len() {
        return Err(MerkleTreeError::WrongBatchSize);
    }
    let mut frontier: Vec<FrontierNode> = unique_leaves
        .into_iter()
        .zip(&proof.salts[0])
        .map(|((leaf_idx, qs), salt)| {
            let digest = hash_leaf_rows_with_salt(
                leaf_domain,
                &opened_values[qs[0]],
                &groups[0].matrices,
                salt,
            );
            FrontierNode {
                index: leaf_idx,
                digest,
                queries: qs,
            }
        })
        .collect();

    let siblings = &proof.pruned.sibling_hashes;
    let mut sib_cursor = 0usize;
    let mut group_cursor = 1usize;
    let mut salt_cursor = 1usize;

    for level in 0..proof_levels {
        let mut next_frontier: Vec<FrontierNode> = Vec::with_capacity(frontier.len());
        let mut i = 0usize;
        while i < frontier.len() {
            let parent_idx = frontier[i].index >> 1;
            let is_left = (frontier[i].index & 1) == 0;
            if is_left && i + 1 < frontier.len() && (frontier[i + 1].index >> 1) == parent_idx {
                let left = &frontier[i];
                let right = &frontier[i + 1];
                let digest = compress_pair_domain(compress_domain, &left.digest, &right.digest);
                let mut queries = Vec::with_capacity(left.queries.len() + right.queries.len());
                queries.extend_from_slice(&left.queries);
                queries.extend_from_slice(&right.queries);
                next_frontier.push(FrontierNode {
                    index: parent_idx,
                    digest,
                    queries,
                });
                i += 2;
            } else {
                let Some(sibling) = siblings.get(sib_cursor) else {
                    // Sibling list ended early; report mismatch.
                    return Err(MerkleTreeError::MalformedPrunedProof(
                        PrunedProofError::SiblingCountMismatch {
                            expected: sib_cursor + 1,
                            got: siblings.len(),
                        },
                    ));
                };
                sib_cursor += 1;
                let node = &frontier[i];
                let digest = if is_left {
                    compress_pair_domain(compress_domain, &node.digest, sibling)
                } else {
                    compress_pair_domain(compress_domain, sibling, &node.digest)
                };
                next_frontier.push(FrontierNode {
                    index: parent_idx,
                    digest,
                    queries: node.queries.clone(),
                });
                i += 1;
            }
        }

        let next_level = level + 1;
        if group_cursor < groups.len() && groups[group_cursor].level == next_level {
            let group = &groups[group_cursor];
            let group_salts = &proof.salts[salt_cursor];
            if group_salts.len() != next_frontier.len() {
                return Err(MerkleTreeError::WrongBatchSize);
            }
            for (slot, (node, salt)) in next_frontier.iter_mut().zip(group_salts).enumerate() {
                let lead = node.queries[0];
                for &q in &node.queries[1..] {
                    for &mat_idx in &group.matrices {
                        if opened_values[q][mat_idx] != opened_values[lead][mat_idx] {
                            return Err(MerkleTreeError::MalformedPrunedProof(
                                PrunedProofError::InconsistentGroupOpening {
                                    slot,
                                    matrix: mat_idx,
                                },
                            ));
                        }
                    }
                }
                let injected = hash_leaf_rows_with_salt(
                    leaf_domain,
                    &opened_values[lead],
                    &group.matrices,
                    salt,
                );
                node.digest = compress_pair_domain(compress_domain, &node.digest, &injected);
            }
            group_cursor += 1;
            salt_cursor += 1;
        }

        frontier = next_frontier;
    }

    if sib_cursor != siblings.len() {
        return Err(MerkleTreeError::MalformedPrunedProof(
            PrunedProofError::SiblingCountMismatch {
                expected: sib_cursor,
                got: siblings.len(),
            },
        ));
    }

    for node in &frontier {
        if node.index >= commit.num_roots() || commit[node.index] != node.digest {
            return Err(MerkleTreeError::CapMismatch);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p3_adapter::pcs::lde_from_shifted_coefficients;
    use crate::types::ExtVal3;
    use p3_dft::Radix2DitParallel;
    use p3_field::{BasedVectorSpace, Field};

    #[test]
    fn derive_leaf_salt_matches_single_block_keyed_blake3_and_is_distinct() {
        let seed: [u8; 32] = core::array::from_fn(|i| u8::try_from((i * 19 + 7) & 0xff).unwrap());
        let s0 = derive_leaf_salt(&seed, 1024, 0);
        let s1 = derive_leaf_salt(&seed, 1024, 1);
        let s2 = derive_leaf_salt(&seed, 512, 0);
        assert_ne!(s0, s1);
        assert_ne!(s0, s2);
        assert_ne!(s1, s2);
        assert_eq!(s0, derive_leaf_salt(&seed, 1024, 0));

        // Cross-check directly against raw BLAKE3 single-block KEYED_HASH compression
        // (matching `cuda/kernels.cu::blake3_derive_leaf_salt`).
        const KEYED_HASH: u32 = 1 << 4;
        const CHUNK_START: u32 = 1 << 0;
        const CHUNK_END: u32 = 1 << 1;
        const ROOT: u32 = 1 << 3;
        const IV: [u32; 8] = [
            0x6A09_E667,
            0xBB67_AE85,
            0x3C6E_F372,
            0xA54F_F53A,
            0x510E_527F,
            0x9B05_688C,
            0x1F83_D9AB,
            0x5BE0_CD19,
        ];
        const MSG_SCHEDULE: [[usize; 16]; 7] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
            [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
            [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
            [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
            [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
            [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
        ];
        let g = |mut st: [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32| {
            st[a] = st[a].wrapping_add(st[b]).wrapping_add(x);
            st[d] = (st[d] ^ st[a]).rotate_right(16);
            st[c] = st[c].wrapping_add(st[d]);
            st[b] = (st[b] ^ st[c]).rotate_right(12);
            st[a] = st[a].wrapping_add(st[b]).wrapping_add(y);
            st[d] = (st[d] ^ st[a]).rotate_right(8);
            st[c] = st[c].wrapping_add(st[d]);
            st[b] = (st[b] ^ st[c]).rotate_right(7);
            st
        };
        let cv: [u32; 8] = core::array::from_fn(|i| {
            u32::from_le_bytes(seed[4 * i..4 * i + 4].try_into().unwrap())
        });
        let height = 1024u64;
        let row = 37u64;
        let mut msg = [0u32; 16];
        msg[0] = u32::try_from(height & 0xffff_ffff).unwrap();
        msg[1] = u32::try_from(height >> 32).unwrap();
        msg[2] = u32::try_from(row & 0xffff_ffff).unwrap();
        msg[3] = u32::try_from(row >> 32).unwrap();
        let mut st = [
            cv[0],
            cv[1],
            cv[2],
            cv[3],
            cv[4],
            cv[5],
            cv[6],
            cv[7],
            IV[0],
            IV[1],
            IV[2],
            IV[3],
            0,
            0,
            16,
            KEYED_HASH | CHUNK_START | CHUNK_END | ROOT,
        ];
        for sched in &MSG_SCHEDULE {
            st = g(st, 0, 4, 8, 12, msg[sched[0]], msg[sched[1]]);
            st = g(st, 1, 5, 9, 13, msg[sched[2]], msg[sched[3]]);
            st = g(st, 2, 6, 10, 14, msg[sched[4]], msg[sched[5]]);
            st = g(st, 3, 7, 11, 15, msg[sched[6]], msg[sched[7]]);
            st = g(st, 0, 5, 10, 15, msg[sched[8]], msg[sched[9]]);
            st = g(st, 1, 6, 11, 12, msg[sched[10]], msg[sched[11]]);
            st = g(st, 2, 7, 8, 13, msg[sched[12]], msg[sched[13]]);
            st = g(st, 3, 4, 9, 14, msg[sched[14]], msg[sched[15]]);
        }
        let mut expected = [0u8; 16];
        for i in 0..4 {
            expected[4 * i..4 * i + 4].copy_from_slice(&(st[i] ^ st[i + 8]).to_le_bytes());
        }
        assert_eq!(
            derive_leaf_salt(
                &seed,
                usize::try_from(height).unwrap(),
                usize::try_from(row).unwrap()
            ),
            expected
        );
    }

    #[test]
    fn shifted_quotient_slices_zk_recombines_identically_and_randomizes() {
        let dft = Radix2DitParallel::<Val>::default();
        let zeta = ExtVal3::from_basis_coefficients_slice(&[
            Val::from_u64(12_345_678),
            Val::from_u64(98_765_432),
            Val::from_u64(55_555_555),
        ])
        .unwrap();

        for log_n in [2usize, 4, 6] {
            let n = 1usize << log_n;
            for q_zk in [1usize, 2, 4] {
                let big_height = 2 * n * q_zk;
                let ext_d = 3usize;
                let coeffs = RowMajorMatrix::new(csprng_field_vec(big_height * ext_d), ext_d);
                // Evaluate Q(zeta) directly from polynomial coefficients.
                let mut q_at_zeta = ExtVal3::ZERO;
                let mut zeta_pow = ExtVal3::ONE;
                for r in 0..big_height {
                    let c = ExtVal3::from_basis_coefficients_slice(
                        &coeffs.values[r * ext_d..(r + 1) * ext_d],
                    )
                    .unwrap();
                    q_at_zeta += c * zeta_pow;
                    zeta_pow *= zeta;
                }

                // Evaluate Q on the disjoint quotient domain `GENERATOR * H_{big_height}`.
                let q_evals = dft
                    .coset_dft_batch(coeffs, Val::GENERATOR)
                    .to_row_major_matrix();

                let slices_a =
                    shifted_quotient_slices_zk(&dft, q_evals.clone(), Val::GENERATOR, q_zk);
                let slices_b = shifted_quotient_slices_zk(&dft, q_evals, Val::GENERATOR, q_zk);
                assert_eq!(slices_a.height(), 2 * n);
                assert_eq!(slices_a.width(), 2 * q_zk * ext_d);
                assert_ne!(
                    slices_a.values, slices_b.values,
                    "telescoping quotient blinding must randomize slices across calls"
                );

                // Reconstruct LDE and verify recombination at zeta matches Q(zeta) exactly!
                for slices in [slices_a, slices_b] {
                    let lde = lde_from_shifted_coefficients(&dft, slices, 2);
                    // Un-bit-reverse and take coset iDFT to read actual slice polynomials.
                    let natural_lde = lde.as_view().bit_reverse_rows().to_row_major_matrix();
                    let recovered = dft.coset_idft_batch(natural_lde, Val::GENERATOR);
                    let width = recovered.width();
                    // High coefficients (rows >= 2n) must be identically zero (degree < 2n).
                    for r in (2 * n)..recovered.height() {
                        for c in 0..width {
                            assert_eq!(
                                recovered.values[r * width + c],
                                Val::ZERO,
                                "slice degree exceeded 2n at row {r} col {c}"
                            );
                        }
                    }
                    // Evaluate each slice polynomial at zeta and recombine with zeta^n.
                    let num_slices = 2 * q_zk;
                    let mut slice_evals = vec![ExtVal3::ZERO; width];
                    let mut zp = ExtVal3::ONE;
                    for r in 0..(2 * n) {
                        for (c, eval) in slice_evals.iter_mut().enumerate().take(width) {
                            *eval += zp * recovered.values[r * width + c];
                        }
                        zp *= zeta;
                    }
                    let zeta_pow_n = zeta.exp_power_of_2(log_n);
                    let mut recombined = ExtVal3::ZERO;
                    let mut zn_pow = ExtVal3::ONE;
                    for s in 0..num_slices {
                        let chunk = &slice_evals[s * ext_d..(s + 1) * ext_d];
                        let basis_val = (0..ext_d).fold(ExtVal3::ZERO, |acc, idx| {
                            let basis = ExtVal3::from_basis_coefficients_fn(|j| {
                                if j == idx { Val::ONE } else { Val::ZERO }
                            });
                            acc + chunk[idx] * basis
                        });
                        recombined += zn_pow * basis_val;
                        zn_pow *= zeta_pow_n;
                    }
                    assert_eq!(
                        recombined, q_at_zeta,
                        "recombined blinded quotient mismatch at n={n}, q_zk={q_zk}"
                    );
                }
            }
        }
    }
}
