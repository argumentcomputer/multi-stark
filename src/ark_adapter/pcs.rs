//! Monomial-basis KZG over BLS12-381 as a crate [`Pcs`].
//!
//! Commitments are per column: interpolate each column of a committed
//! matrix over its domain (radix-2 iFFT) and MSM the coefficients
//! against the SRS — a round's commitment is the vector of G1 points,
//! ~48 bytes per column, and proof size is independent of any query
//! count (there are no queries).
//!
//! Opening batches per distinct point: all polynomials opened at `z`
//! (across every round and matrix) are folded with powers of one
//! transcript challenge `v`, and a single witness commitment
//! `W_z = [ (Σᵢ vⁱ·pᵢ − Σᵢ vⁱ·pᵢ(z)) / (X − z) ]·G` covers them. The
//! multi-stark opens at `ζ` and the per-trace-height `ζ·gₖ`, so a whole
//! proof carries a handful of G1 points. Verification folds the same
//! combination over the commitments and checks all points with one
//! 2-pairing equation, cross-batched by a second challenge `r`:
//! `e(Σ_z r^z·(C_z − y_z·G + z·W_z), H) = e(Σ_z r^z·W_z, τH)`.
//!
//! The quotient commit follows the core's coefficient-slice convention
//! (`Q(X) = Σₖ X^{k·n}·cₖ(X)`, verifier recombines at ζ): one coset
//! iFFT off the quotient domain, then each length-`n` slice is just a
//! range of the coefficient vector — no evaluation representation ever
//! needed. Trace evaluations on the quotient domain
//! ([`Pcs::get_evaluations_on_domain`]) are coset FFTs from the stored
//! coefficients; that FFT budget is what [`Pcs::max_quotient_degree`]
//! bounds (there is no blowup wall — exceeding it is slow, not
//! unsound, but the build-time check keeps the cost model honest).

use std::sync::Arc;

use ark_bls12_381::{Bls12_381, Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, VariableBaseMSM, pairing::Pairing};
use ark_ff::{AdditiveGroup, Field as ArkField, Zero};
use ark_poly::{EvaluationDomain as ArkEvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::traits::{EvaluationDomain, OpenedValues, OpeningRounds, Pcs, Transcript, VerifyRounds};

use super::domain::Radix2Coset;
use super::field::Scalar;
use super::srs::Srs;
use super::transcript::Blake3Transcript;

/// One KZG commitment round: per matrix, one G1 point per column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KzgCommitment(pub Vec<Vec<G1Affine>>);

/// One opening proof: one witness point per distinct opening point, in
/// transcript (first-appearance) order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KzgProof(pub Vec<G1Affine>);

/// Serde via the arkworks canonical (compressed, validated) encoding.
macro_rules! serde_via_canonical {
    ($t:ty, $inner:ty) => {
        impl Serialize for $t {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut bytes = Vec::new();
                self.0
                    .serialize_compressed(&mut bytes)
                    .map_err(serde::ser::Error::custom)?;
                bytes.serialize(serializer)
            }
        }
        impl<'de> Deserialize<'de> for $t {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let bytes = Vec::<u8>::deserialize(deserializer)?;
                <$inner>::deserialize_compressed(&bytes[..])
                    .map(Self)
                    .map_err(serde::de::Error::custom)
            }
        }
    };
}
serde_via_canonical!(KzgCommitment, Vec<Vec<G1Affine>>);
serde_via_canonical!(KzgProof, Vec<G1Affine>);

/// A committed matrix on the prover side: its domain and each column
/// polynomial in coefficient form (length = domain size).
pub struct CommittedMatrix {
    domain: Radix2Coset,
    columns: Vec<Vec<Fr>>,
}

/// Prover-side retained data for one commitment round.
pub struct KzgProverData {
    pub matrices: Vec<CommittedMatrix>,
}

#[derive(Debug)]
pub enum KzgError {
    /// Commitment/opened-value dimensions disagree with the rounds.
    ShapeMismatch,
    /// The batched pairing equation does not hold.
    PairingCheckFailed,
}

/// See the module docs.
pub struct KzgPcs {
    srs: Arc<Srs>,
    max_quotient_degree: usize,
}

impl KzgPcs {
    pub fn new(srs: Arc<Srs>, max_quotient_degree: usize) -> Self {
        Self {
            srs,
            max_quotient_degree,
        }
    }

    /// The arkworks FFT domain realizing one of ours.
    fn ark_domain(domain: Radix2Coset) -> Radix2EvaluationDomain<Fr> {
        let base = Radix2EvaluationDomain::new(domain.size()).expect("size within Fr two-adicity");
        if domain.shift == crate::traits::Algebra::<Scalar>::ONE {
            base
        } else {
            base.get_coset(domain.shift.0).expect("nonzero coset shift")
        }
    }

    fn commit_columns(&self, columns: &[Vec<Fr>]) -> Vec<G1Affine> {
        let commits: Vec<G1Projective> = columns.iter().map(|c| self.msm(c)).collect();
        G1Projective::normalize_batch(&commits)
    }

    fn msm(&self, coeffs: &[Fr]) -> G1Projective {
        assert!(
            coeffs.len() <= self.srs.max_len(),
            "polynomial length {} exceeds the SRS ({})",
            coeffs.len(),
            self.srs.max_len()
        );
        if coeffs.is_empty() {
            return G1Projective::zero();
        }
        G1Projective::msm(&self.srs.g1[..coeffs.len()], coeffs).expect("equal lengths")
    }

    /// Interpolate each column of `matrix` over `domain`.
    fn interpolate_columns(domain: Radix2Coset, matrix: &RowMajorMatrix<Scalar>) -> Vec<Vec<Fr>> {
        assert_eq!(
            matrix.height(),
            domain.size(),
            "matrix height != domain size"
        );
        let width = matrix.width();
        let mut evals: Vec<Vec<Fr>> = vec![Vec::with_capacity(matrix.height()); width];
        for (i, value) in matrix.values.iter().enumerate() {
            evals[i % width].push(value.0);
        }
        let ark = Self::ark_domain(domain);
        evals.into_iter().map(|col| ark.ifft(&col)).collect()
    }
}

impl Pcs for KzgPcs {
    type F = Scalar;
    type Challenge = Scalar;
    type Domain = Radix2Coset;
    type Challenger = Blake3Transcript;
    type Commitment = KzgCommitment;
    type ProverData = KzgProverData;
    type Proof = KzgProof;
    type Error = KzgError;
    type Evaluations<'a> = RowMajorMatrix<Scalar>;

    fn natural_domain_for_degree(&self, degree: usize) -> Radix2Coset {
        Radix2Coset {
            log_size: p3_util::log2_strict_usize(degree),
            shift: crate::traits::Algebra::<Scalar>::ONE,
        }
    }

    fn max_quotient_degree(&self) -> usize {
        self.max_quotient_degree
    }

    fn commit(
        &self,
        evaluations: Vec<(Radix2Coset, RowMajorMatrix<Scalar>)>,
    ) -> (KzgCommitment, KzgProverData) {
        let matrices: Vec<CommittedMatrix> = evaluations
            .into_iter()
            .map(|(domain, matrix)| CommittedMatrix {
                domain,
                columns: Self::interpolate_columns(domain, &matrix),
            })
            .collect();
        let commitment = KzgCommitment(
            matrices
                .iter()
                .map(|m| self.commit_columns(&m.columns))
                .collect(),
        );
        (commitment, KzgProverData { matrices })
    }

    fn commit_quotient(
        &self,
        quotients: Vec<(Radix2Coset, RowMajorMatrix<Scalar>, usize)>,
    ) -> (KzgCommitment, KzgProverData) {
        let matrices: Vec<CommittedMatrix> = quotients
            .into_iter()
            .map(|(quotient_domain, evaluations, quotient_degree)| {
                let big = quotient_domain.size();
                debug_assert_eq!(big % quotient_degree, 0);
                let n = big / quotient_degree;
                let coefficient_columns = Self::interpolate_columns(quotient_domain, &evaluations);
                // Slice `Q(X) = Σₖ X^{k·n}·cₖ(X)`: slice k of coordinate d
                // is coefficient range [k·n, (k+1)·n), laid out as column
                // `k·D + d` — the order the verifier's ζ-recombination
                // reads.
                let columns: Vec<Vec<Fr>> = (0..quotient_degree)
                    .flat_map(|k| {
                        coefficient_columns
                            .iter()
                            .map(move |c| c[k * n..(k + 1) * n].to_vec())
                    })
                    .collect();
                CommittedMatrix {
                    domain: Radix2Coset {
                        log_size: p3_util::log2_strict_usize(n),
                        shift: crate::traits::Algebra::<Scalar>::ONE,
                    },
                    columns,
                }
            })
            .collect();
        let commitment = KzgCommitment(
            matrices
                .iter()
                .map(|m| self.commit_columns(&m.columns))
                .collect(),
        );
        (commitment, KzgProverData { matrices })
    }

    fn get_evaluations_on_domain(
        &self,
        data: &KzgProverData,
        idx: usize,
        domain: Radix2Coset,
    ) -> RowMajorMatrix<Scalar> {
        let matrix = &data.matrices[idx];
        assert!(
            domain.size() <= matrix.domain.size() * self.max_quotient_degree,
            "requested domain ({}) exceeds the coset-FFT budget ({}x trace); \
             raise max_quotient_degree if this cost is intended",
            domain.size(),
            self.max_quotient_degree
        );
        let ark = Self::ark_domain(domain);
        let column_evals: Vec<Vec<Fr>> = matrix.columns.iter().map(|c| ark.fft(c)).collect();
        let width = column_evals.len();
        let height = domain.size();
        let mut values = Vec::with_capacity(width * height);
        for row in 0..height {
            values.extend(column_evals.iter().map(|col| Scalar(col[row])));
        }
        RowMajorMatrix::new(values, width)
    }

    fn open(
        &self,
        rounds: OpeningRounds<'_, KzgProverData, Scalar>,
        challenger: &mut Blake3Transcript,
    ) -> (OpenedValues<Scalar>, KzgProof) {
        // Pass 1: evaluate everything, observing values in traversal
        // order, and batch (polynomial, value) pairs per distinct point.
        struct PointBatch<'a> {
            z: Fr,
            entries: Vec<(&'a [Fr], Fr)>,
        }
        let mut batches: Vec<PointBatch<'_>> = Vec::new();
        let mut opened: OpenedValues<Scalar> = Vec::new();
        for (data, points_per_matrix) in &rounds {
            debug_assert_eq!(data.matrices.len(), points_per_matrix.len());
            let mut round_values = Vec::new();
            for (matrix, points) in data.matrices.iter().zip(points_per_matrix) {
                let mut matrix_values = Vec::new();
                for &z in points {
                    let row: Vec<Scalar> = matrix
                        .columns
                        .iter()
                        .map(|c| Scalar(eval_poly(c, z.0)))
                        .collect();
                    for &value in &row {
                        challenger.observe_challenge(value);
                    }
                    let batch = match batches.iter().position(|b| b.z == z.0) {
                        Some(i) => &mut batches[i],
                        None => {
                            batches.push(PointBatch {
                                z: z.0,
                                entries: Vec::new(),
                            });
                            batches.last_mut().expect("just pushed")
                        }
                    };
                    for (column, value) in matrix.columns.iter().zip(&row) {
                        batch.entries.push((column, value.0));
                    }
                    matrix_values.push(row);
                }
                round_values.push(matrix_values);
            }
            opened.push(round_values);
        }

        let v = challenger.sample_challenge().0;

        // Pass 2: per point, fold with powers of v and commit the witness.
        let witnesses: Vec<G1Projective> = batches
            .iter()
            .map(|batch| {
                let max_len = batch
                    .entries
                    .iter()
                    .map(|(c, _)| c.len())
                    .max()
                    .unwrap_or(0);
                let mut combined = vec![Fr::ZERO; max_len];
                let mut power = Fr::ONE;
                for (coeffs, _value) in &batch.entries {
                    for (acc, c) in combined.iter_mut().zip(*coeffs) {
                        *acc += power * c;
                    }
                    power *= v;
                }
                // The constant offset −Σ vⁱ·yᵢ only shifts the remainder;
                // the witness quotient ignores it.
                self.msm(&divide_by_linear(&combined, batch.z))
            })
            .collect();
        let witnesses = G1Projective::normalize_batch(&witnesses);
        for w in &witnesses {
            challenger.observe_canonical(w);
        }
        // Mirror the verifier's cross-point batching sample to keep the
        // transcripts in lockstep (the prover has no use for r).
        let _r = challenger.sample_challenge();

        (opened, KzgProof(witnesses))
    }

    fn verify(
        &self,
        rounds: VerifyRounds<KzgCommitment, Radix2Coset, Scalar>,
        proof: &KzgProof,
        challenger: &mut Blake3Transcript,
    ) -> Result<(), KzgError> {
        // Mirror `open`'s traversal exactly: observe claimed values and
        // batch (commitment, value) pairs per distinct point.
        struct PointBatch {
            z: Fr,
            commitments: Vec<G1Affine>,
            values: Vec<Fr>,
        }
        let mut batches: Vec<PointBatch> = Vec::new();
        for (commitment, matrices) in &rounds {
            if commitment.0.len() != matrices.len() {
                return Err(KzgError::ShapeMismatch);
            }
            for (column_commits, (_domain, openings)) in commitment.0.iter().zip(matrices) {
                for (z, values) in openings {
                    if values.len() != column_commits.len() {
                        return Err(KzgError::ShapeMismatch);
                    }
                    for &value in values {
                        challenger.observe_challenge(value);
                    }
                    let batch = match batches.iter().position(|b| b.z == z.0) {
                        Some(i) => &mut batches[i],
                        None => {
                            batches.push(PointBatch {
                                z: z.0,
                                commitments: Vec::new(),
                                values: Vec::new(),
                            });
                            batches.last_mut().expect("just pushed")
                        }
                    };
                    batch.commitments.extend_from_slice(column_commits);
                    batch.values.extend(values.iter().map(|value| value.0));
                }
            }
        }
        if proof.0.len() != batches.len() {
            return Err(KzgError::ShapeMismatch);
        }

        let v = challenger.sample_challenge().0;
        for w in &proof.0 {
            challenger.observe_canonical(w);
        }
        let r = challenger.sample_challenge().0;

        // Per point z (with witness W and v-powers u):
        //   e(C_z − y_z·G + z·W, H) = e(W, τH)
        // where C_z = Σ uᵢ·Cᵢ and y_z = Σ uᵢ·yᵢ. Cross-batched over
        // points with powers of r into one 2-pairing product.
        let g = G1Projective::from(self.srs.g1[0]);
        let mut lhs = G1Projective::zero();
        let mut rhs = G1Projective::zero();
        let mut r_power = Fr::ONE;
        for (batch, &witness) in batches.iter().zip(&proof.0) {
            let mut v_powers = Vec::with_capacity(batch.values.len());
            let mut power = Fr::ONE;
            let mut y = Fr::ZERO;
            for &value in &batch.values {
                v_powers.push(power);
                y += power * value;
                power *= v;
            }
            let c = G1Projective::msm(&batch.commitments, &v_powers).expect("equal lengths");
            lhs += (c - g * y + witness * batch.z) * r_power;
            rhs += witness * r_power;
            r_power *= r;
        }
        let check = Bls12_381::multi_pairing(
            [lhs.into_affine(), (-rhs).into_affine()],
            [self.srs.g2, self.srs.tau_g2],
        );
        if check.is_zero() {
            Ok(())
        } else {
            Err(KzgError::PairingCheckFailed)
        }
    }
}

/// Horner evaluation.
fn eval_poly(coeffs: &[Fr], z: Fr) -> Fr {
    coeffs.iter().rev().fold(Fr::ZERO, |acc, c| acc * z + c)
}

/// The quotient of `p` by `(X − z)` (synthetic division; the remainder
/// — `p(z)` — is dropped).
fn divide_by_linear(p: &[Fr], z: Fr) -> Vec<Fr> {
    if p.len() <= 1 {
        return Vec::new();
    }
    let mut quotient = vec![Fr::ZERO; p.len() - 1];
    let mut carry = Fr::ZERO;
    for j in (1..p.len()).rev() {
        carry = p[j] + z * carry;
        quotient[j - 1] = carry;
    }
    quotient
}
