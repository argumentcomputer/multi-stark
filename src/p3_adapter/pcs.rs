//! [`crate::traits::Pcs`] for the Plonky3 two-adic FRI PCS.
//!
//! [`FriPcs`] wraps [`TwoAdicFriPcs`] together with the blowup (which p3
//! keeps private) and a DFT engine for the quotient path. Everything
//! FRI-representation-specific about the quotient commit lives here: the
//! prover hands over quotient EVALUATIONS on the disjoint domain, and
//! this adapter slices coefficients with the committed LDE's coset shift
//! folded in ([`shifted_quotient_slices`]) and builds the committed LDE
//! with one zero-padded DFT ([`lde_from_shifted_coefficients`]) —
//! bit-identical to `Pcs::commit` on the slices' trace-domain
//! evaluations, while skipping two size-n transforms per column.

use std::marker::PhantomData;

use p3_commit::{Pcs as P3Pcs, PolynomialSpace};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_fri::TwoAdicFriPcs;
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::traits::{EvaluationDomain, OpenedValues, Pcs, VerifyRounds};

/// The p3 FRI PCS plus the two pieces of configuration the adapter needs
/// that p3 keeps internal: the blowup (the quotient-degree budget) and a
/// DFT engine for the quotient commit.
pub struct FriPcs<F, EF, Challenger, Dft, ValMmcs, ChallengeMmcs> {
    inner: TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs>,
    dft: Dft,
    log_blowup: usize,
    _marker: PhantomData<(EF, Challenger)>,
}

impl<F, EF, Challenger, Dft: Default, ValMmcs, ChallengeMmcs>
    FriPcs<F, EF, Challenger, Dft, ValMmcs, ChallengeMmcs>
{
    pub fn new(inner: TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs>, log_blowup: usize) -> Self {
        Self {
            inner,
            dft: Dft::default(),
            log_blowup,
            _marker: PhantomData,
        }
    }
}

impl<F, EF, Challenger, Dft, ValMmcs, ChallengeMmcs> Pcs
    for FriPcs<F, EF, Challenger, Dft, ValMmcs, ChallengeMmcs>
where
    F: TwoAdicField + Ord,
    EF: TwoAdicField + ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
    TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs>: P3Pcs<EF, Challenger>,
    <TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs> as P3Pcs<EF, Challenger>>::Domain:
        PolynomialSpace<Val = F> + EvaluationDomain<F = F, Challenge = EF>,
{
    type F = F;
    type Challenge = EF;
    type Domain = <TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs> as P3Pcs<EF, Challenger>>::Domain;
    type Challenger = Challenger;
    type Commitment =
        <TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs> as P3Pcs<EF, Challenger>>::Commitment;
    type ProverData =
        <TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs> as P3Pcs<EF, Challenger>>::ProverData;
    type Proof = <TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs> as P3Pcs<EF, Challenger>>::Proof;
    type Error = <TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs> as P3Pcs<EF, Challenger>>::Error;
    type Evaluations<'a> = <TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs> as P3Pcs<
        EF,
        Challenger,
    >>::EvaluationsOnDomain<'a>;

    #[inline]
    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        self.inner.natural_domain_for_degree(degree)
    }

    #[inline]
    fn max_quotient_degree(&self) -> usize {
        // The LDE-subsetting economy: trace evaluations on the quotient
        // domain come for free only up to the blowup.
        1 << self.log_blowup
    }

    #[inline]
    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<F>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        self.inner.commit(evaluations)
    }

    fn commit_quotient(
        &self,
        quotients: Vec<(Self::Domain, RowMajorMatrix<F>, usize)>,
    ) -> (Self::Commitment, Self::ProverData) {
        // `commit_ldes` skips the randomization a hiding PCS applies
        // inside `commit`; this path targets non-hiding configurations.
        assert!(
            !<TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs> as P3Pcs<EF, Challenger>>::ZK,
            "committing the quotient from coefficients bypasses hiding-PCS randomization"
        );
        let ldes = quotients
            .into_iter()
            .map(|(quotient_domain, evaluations, quotient_degree)| {
                let sliced = shifted_quotient_slices(
                    &self.dft,
                    evaluations,
                    EvaluationDomain::first_point(&quotient_domain),
                    quotient_degree,
                );
                lde_from_shifted_coefficients(&self.dft, sliced, self.log_blowup)
            })
            .collect();
        self.inner.commit_ldes(ldes)
    }

    #[inline]
    fn get_evaluations_on_domain<'a>(
        &self,
        data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::Evaluations<'a> {
        self.inner.get_evaluations_on_domain(data, idx, domain)
    }

    #[inline]
    fn open(
        &self,
        rounds: Vec<(&Self::ProverData, Vec<Vec<EF>>)>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<EF>, Self::Proof) {
        self.inner.open(rounds, challenger)
    }

    #[inline]
    fn verify(
        &self,
        rounds: VerifyRounds<Self::Commitment, Self::Domain, EF>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        self.inner.verify(rounds, proof, challenger)
    }
}

/// From the quotient's evaluations on its disjoint domain — `q·n` rows of
/// `D` base columns on the coset `shift·H` with `|H| = q·n` — produce the
/// `n`-row, `q·D`-column matrix of slice coefficients with the committed
/// LDE's `GENERATOR` shift already folded in: the input
/// [`lde_from_shifted_coefficients`] expects.
///
/// Semantically this is three steps: coset iDFT to coefficients, slicing
/// `Q(X) = Σₖ X^{k·n}·cₖ(X)` into rows `[c₀ | … | c_{q−1}]`, and
/// pre-scaling row `r` by `GENERATOR^r`. Executed literally (the library
/// entry points) those steps cost a bit-reversal materialization, a serial
/// row-swap pass, two serial full-matrix scaling passes, and a serial
/// gather. But the composition collapses: with `N = q·n` and `S` the raw
/// bit-reversed storage of the forward DFT,
///
/// ```text
///   idft(f)ⱼ = N⁻¹ · dft(f)_{(N−j) mod N} = N⁻¹ · S[rev((N−j) mod N)]
/// ```
///
/// and the coset-unscale factor `shift^{−j}` at `j = k·n + r` splits into
/// `shift^{−k·n} · shift^{−r}`, whose row-dependent part cancels the LDE
/// pre-scale `GENERATOR^r` exactly, because the disjoint quotient domain's
/// shift IS the generator (asserted below; `create_disjoint_domain` on a
/// natural trace domain guarantees it). What survives is a single parallel
/// gather off the DFT storage with ONE constant weight per slice:
/// `wₖ = N⁻¹ · GENERATOR^{−k·n}`.
fn shifted_quotient_slices<F: TwoAdicField + Ord>(
    dft: &impl TwoAdicSubgroupDft<F>,
    quotient_evals: RowMajorMatrix<F>,
    domain_shift: F,
    quotient_degree: usize,
) -> RowMajorMatrix<F> {
    assert_eq!(
        domain_shift,
        F::GENERATOR,
        "quotient domain shift must equal the LDE shift for the scalings to cancel"
    );
    let ext_degree = quotient_evals.width();
    let big_height = quotient_evals.height();
    let log_big_height = log2_strict_usize(big_height);
    debug_assert_eq!(big_height % quotient_degree, 0);
    let n = big_height / quotient_degree;
    let width = quotient_degree * ext_degree;
    // Raw storage of the forward DFT: natural index `k` lives at row
    // `rev(k)`, and the unwrap out of the bit-reversed view is copy-free.
    let storage = dft
        .dft_batch(quotient_evals)
        .bit_reverse_rows()
        .to_row_major_matrix();
    let n_inv = F::ONE.div_2exp_u64(log_big_height as u64);
    let weight_step = F::GENERATOR.exp_u64(n as u64).inverse();
    let weights: Vec<F> = weight_step
        .powers()
        .take(quotient_degree)
        .map(|w| w * n_inv)
        .collect();
    let mut values = F::zero_vec(n * width);
    values
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out)| {
            for (chunk, weight) in weights.iter().enumerate() {
                let j = chunk * n + row;
                let src = reverse_bits_len(
                    big_height.wrapping_sub(j) & (big_height - 1),
                    log_big_height,
                );
                let src = &storage.values[src * ext_degree..(src + 1) * ext_degree];
                for (out, src) in out[chunk * ext_degree..(chunk + 1) * ext_degree]
                    .iter_mut()
                    .zip(src)
                {
                    *out = *src * *weight;
                }
            }
        });
    RowMajorMatrix::new(values, width)
}

/// Low-degree extension of column polynomials given by their COEFFICIENTS
/// with the `GENERATOR` coset shift already folded in (row `j`
/// pre-multiplied by `GENERATOR^j`, which [`shifted_quotient_slices`]
/// produces for free), in the exact layout `Pcs::commit` stores for
/// evaluations on the natural domain: `2^log_blowup` row blocks, where
/// block `b` holds the evaluations on the coset `GENERATOR · w^rev(b) · H`
/// in bit-reversed row order (`H` is the size-`n` subgroup, `w` generates
/// the size-`2^log_blowup · n` subgroup, and `rev` reverses `log_blowup`
/// bits). Globally that is the bit-reversal of the natural order of the
/// whole blown-up coset — i.e.
/// `coset_lde_batch(evals, log_blowup, GENERATOR).bit_reverse_rows()`,
/// which is what `TwoAdicFriPcs::commit` computes.
///
/// Committing the result via `Pcs::commit_ldes` is therefore bit-identical
/// to `Pcs::commit` on the columns' trace-domain evaluations — field
/// arithmetic is exact, so equal polynomials give equal evaluations no
/// matter which transform produced them — while skipping both that
/// evaluation DFT and the inverse DFT `commit` would open with.
///
/// The whole extension is ONE size-`2^log_blowup · n` transform: zero-pad
/// the shifted coefficients to the LDE height (which leaves the column
/// polynomials unchanged) and DFT. That spends `log_blowup` more butterfly
/// layers than `2^log_blowup` separate size-`n` coset DFTs would, but one
/// batched transform is what the memory traffic wants: no per-coset matrix
/// clones, no per-coset serial shift-scaling passes inside
/// `coset_dft_batch`, no reassembly copies, and `Radix2DitParallel`'s
/// native output order is already the bit-reversed storage order, so the
/// final unwrap is copy-free.
fn lde_from_shifted_coefficients<F: TwoAdicField + Ord>(
    dft: &impl TwoAdicSubgroupDft<F>,
    mut coefficients: RowMajorMatrix<F>,
    log_blowup: usize,
) -> RowMajorMatrix<F> {
    let height = coefficients.height();
    coefficients.pad_to_height(height << log_blowup, F::ZERO);
    dft.dft_batch(coefficients)
        .bit_reverse_rows()
        .to_row_major_matrix()
}

/// Reference form of [`lde_from_shifted_coefficients`] taking PLAIN
/// coefficients: folds the `GENERATOR` shift in explicitly. Only the
/// pinning tests need it; the prover gets the shift for free inside
/// [`shifted_quotient_slices`].
#[cfg(test)]
fn lde_from_coefficients<F: TwoAdicField + Ord>(
    dft: &impl TwoAdicSubgroupDft<F>,
    mut coefficients: RowMajorMatrix<F>,
    log_blowup: usize,
) -> RowMajorMatrix<F> {
    scale_rows_by_powers(&mut coefficients, F::GENERATOR);
    lde_from_shifted_coefficients(dft, coefficients, log_blowup)
}

/// Multiplies row `j` of `mat` by `base^j`, in parallel: each chunk of rows
/// pays one exponentiation and steps serially from there.
#[cfg(test)]
fn scale_rows_by_powers<F: p3_field::Field>(mat: &mut RowMajorMatrix<F>, base: F) {
    const ROWS_PER_CHUNK: usize = 512;
    let width = mat.width();
    mat.values
        .par_chunks_mut(ROWS_PER_CHUNK * width)
        .enumerate()
        .for_each(|(chunk, rows)| {
            let mut weight = base.exp_u64((chunk * ROWS_PER_CHUNK) as u64);
            for row in rows.chunks_mut(width) {
                for value in row {
                    *value *= weight;
                }
                weight *= base;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Val;
    use p3_dft::Radix2DitParallel;
    use p3_field::{Field, PrimeCharacteristicRing};
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    /// `lde_from_coefficients` must reproduce, value for value, the matrix
    /// `TwoAdicFriPcs::commit` stores for the same polynomials given as
    /// trace-domain evaluations: `coset_lde_batch` with the generator shift
    /// (the natural domain's shift is one), then a bit-reversal. This pins
    /// the exact substitution the quotient commit relies on.
    #[test]
    fn lde_from_coefficients_matches_commit_transform() {
        let mut rng = SmallRng::seed_from_u64(0);
        let dft = Radix2DitParallel::<Val>::default();
        for log_height in [0usize, 1, 2, 5, 8] {
            for log_blowup in [1usize, 2, 3] {
                for width in [1usize, 2, 7] {
                    let height = 1 << log_height;
                    let coefficients = RowMajorMatrix::new(
                        (0..height * width).map(|_| rng.random()).collect(),
                        width,
                    );
                    let evaluations = dft
                        .coset_dft_batch(coefficients.clone(), Val::ONE)
                        .to_row_major_matrix();
                    let expected = dft
                        .coset_lde_batch(evaluations, log_blowup, Val::GENERATOR)
                        .bit_reverse_rows()
                        .to_row_major_matrix();
                    let got = lde_from_coefficients(&dft, coefficients, log_blowup);
                    assert_eq!(got, expected, "h=2^{log_height} B=2^{log_blowup} w={width}");
                }
            }
        }
    }

    /// `shifted_quotient_slices` must reproduce, value for value, the naive
    /// composition it replaces: coset iDFT off the quotient domain, slicing
    /// the coefficients into `q` chunks per row, and folding the committed
    /// LDE's `GENERATOR` shift into the rows. This pins the scaling
    /// cancellation the fused gather relies on.
    #[test]
    fn shifted_quotient_slices_matches_naive_composition() {
        let mut rng = SmallRng::seed_from_u64(1);
        let dft = Radix2DitParallel::<Val>::default();
        for log_n in [0usize, 1, 2, 5, 7] {
            for quotient_degree in [1usize, 2, 4] {
                for ext_degree in [1usize, 2] {
                    let n = 1 << log_n;
                    let big_height = n * quotient_degree;
                    let evals = RowMajorMatrix::new(
                        (0..big_height * ext_degree).map(|_| rng.random()).collect(),
                        ext_degree,
                    );
                    // Naive path: coset iDFT, slice, fold the LDE shift in.
                    let coefficients = dft.coset_idft_batch(evals.clone(), Val::GENERATOR);
                    let width = quotient_degree * ext_degree;
                    let mut sliced = Vec::with_capacity(n * width);
                    for row in 0..n {
                        for chunk in 0..quotient_degree {
                            sliced.extend_from_slice(
                                &coefficients.values[(chunk * n + row) * ext_degree
                                    ..(chunk * n + row + 1) * ext_degree],
                            );
                        }
                    }
                    let mut expected = RowMajorMatrix::new(sliced, width);
                    scale_rows_by_powers(&mut expected, Val::GENERATOR);
                    let got = shifted_quotient_slices(&dft, evals, Val::GENERATOR, quotient_degree);
                    assert_eq!(
                        got, expected,
                        "n=2^{log_n} q={quotient_degree} D={ext_degree}"
                    );
                }
            }
        }
    }
}
