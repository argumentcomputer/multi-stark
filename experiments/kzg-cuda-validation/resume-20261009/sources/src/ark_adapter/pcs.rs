//! Monomial-basis KZG over BLS12-381 as a crate [`Pcs`].
//!
//! Commitments are per column: interpolate each column of a committed
//! matrix over its domain (radix-2 iFFT) and MSM the coefficients
//! against the SRS — a round's commitment is the vector of G1 points,
//! 48 bytes per column. Shorter polynomials also carry a shifted commitment
//! checked against the SRS degree key. There are no FRI queries.
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

fn map_columns<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Send + Sync) -> Vec<R> {
    use p3_maybe_rayon::prelude::*;
    items.into_par_iter().map(f).collect()
}

/// Per-matrix column commitments and optional shifted degree commitments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KzgCommitment(pub Vec<Vec<G1Affine>>, pub Vec<Vec<G1Affine>>);

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
                let mut input = bytes.as_slice();
                let value = <$inner>::deserialize_compressed(&mut input)
                    .map_err(serde::de::Error::custom)?;
                if !input.is_empty() {
                    return Err(serde::de::Error::custom("trailing proof bytes"));
                }
                Ok(Self(value))
            }
        }
    };
}
impl Serialize for KzgCommitment {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut bytes = Vec::new();
        self.0
            .serialize_compressed(&mut bytes)
            .map_err(serde::ser::Error::custom)?;
        self.1
            .serialize_compressed(&mut bytes)
            .map_err(serde::ser::Error::custom)?;
        bytes.serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for KzgCommitment {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        let mut input = bytes.as_slice();
        let main = Vec::<Vec<G1Affine>>::deserialize_compressed(&mut input)
            .map_err(serde::de::Error::custom)?;
        let shifted = Vec::<Vec<G1Affine>>::deserialize_compressed(&mut input)
            .map_err(serde::de::Error::custom)?;
        if !input.is_empty() {
            return Err(serde::de::Error::custom("trailing commitment bytes"));
        }
        Ok(Self(main, shifted))
    }
}
serde_via_canonical!(KzgProof, Vec<G1Affine>);

/// A committed matrix on the prover side: its domain and each column
/// polynomial in coefficient form (length = domain size).
pub struct CommittedMatrix {
    pub(crate) domain: Radix2Coset,
    columns: Vec<Vec<Fr>>,
    constants: std::sync::OnceLock<Vec<bool>>,
    #[cfg(feature = "kzg-cuda")]
    resident: std::sync::OnceLock<Vec<Option<Arc<super::cuda::ResidentPolynomial>>>>,
}

impl CommittedMatrix {
    fn new(domain: Radix2Coset, columns: Vec<Vec<Fr>>) -> Self {
        Self {
            domain,
            columns,
            constants: Default::default(),
            #[cfg(feature = "kzg-cuda")]
            resident: Default::default(),
        }
    }

    fn constant_columns(&self) -> &[bool] {
        self.constants.get_or_init(|| {
            map_columns(self.columns.iter().collect(), |column| {
                column.iter().skip(1).all(Zero::is_zero)
            })
        })
    }

    #[cfg(feature = "kzg-cuda")]
    fn resident_columns(&self) -> &[Option<Arc<super::cuda::ResidentPolynomial>>] {
        self.resident.get_or_init(|| {
            let constants = self.constant_columns();
            map_columns(
                self.columns.iter().enumerate().collect(),
                |(index, column)| {
                    if constants[index] {
                        None
                    } else {
                        super::cuda::retain(column)
                    }
                },
            )
        })
    }
}

/// Prover-side retained data for one commitment round.
pub struct KzgProverData {
    commitment: KzgCommitment,
    pub matrices: Vec<CommittedMatrix>,
}

impl KzgProverData {
    /// Save a local prover checkpoint. This is not a verifier-key format.
    pub fn write_checkpoint(
        &self,
        mut out: impl std::io::Write,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.commitment.0.serialize_compressed(&mut out)?;
        self.commitment.1.serialize_compressed(&mut out)?;
        (self.matrices.len() as u64).serialize_compressed(&mut out)?;
        for matrix in &self.matrices {
            (matrix.domain.log_size as u64).serialize_compressed(&mut out)?;
            matrix.domain.shift.0.serialize_compressed(&mut out)?;
            (matrix.columns.len() as u64).serialize_compressed(&mut out)?;
            for column in &matrix.columns {
                (column.len() as u64).serialize_compressed(&mut out)?;
                super::encoding::write_fields(&mut out, column, |value| *value)?;
            }
        }
        Ok(())
    }

    /// Load a trusted, locally produced prover checkpoint.
    pub fn read_checkpoint(
        mut input: impl std::io::Read,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let commitment = KzgCommitment(
            Vec::deserialize_compressed(&mut input)?,
            Vec::deserialize_compressed(&mut input)?,
        );
        let count = usize::try_from(u64::deserialize_compressed(&mut input)?)
            .map_err(|_error| ark_serialize::SerializationError::InvalidData)?;
        if count != commitment.0.len() || count != commitment.1.len() {
            return Err(ark_serialize::SerializationError::InvalidData);
        }
        let mut matrices = Vec::with_capacity(count);
        for i in 0..count {
            let log_size = usize::try_from(u64::deserialize_compressed(&mut input)?)
                .map_err(|_error| ark_serialize::SerializationError::InvalidData)?;
            if log_size > 32 {
                return Err(ark_serialize::SerializationError::InvalidData);
            }
            let shift = Scalar(Fr::deserialize_compressed(&mut input)?);
            let width = usize::try_from(u64::deserialize_compressed(&mut input)?)
                .map_err(|_| ark_serialize::SerializationError::InvalidData)?;
            if width != commitment.0[i].len() {
                return Err(ark_serialize::SerializationError::InvalidData);
            }
            let mut columns = Vec::with_capacity(width);
            for _ in 0..width {
                let length = u64::deserialize_compressed(&mut input)?;
                if length != 1u64 << log_size {
                    return Err(ark_serialize::SerializationError::InvalidData);
                }
                columns.push(super::encoding::read_fields(
                    &mut input,
                    length as usize,
                    |value| value,
                )?);
            }
            matrices.push(CommittedMatrix::new(
                Radix2Coset { log_size, shift },
                columns,
            ));
        }
        let mut tail = [0];
        if input.read(&mut tail)? != 0 {
            return Err(ark_serialize::SerializationError::InvalidData);
        }
        Ok(Self {
            commitment,
            matrices,
        })
    }

    /// Combine independently committed matrices in canonical circuit order.
    pub fn concatenate(parts: impl IntoIterator<Item = Self>) -> (KzgCommitment, Self) {
        let mut commitment = KzgCommitment(vec![], vec![]);
        let mut matrices = Vec::new();
        for mut part in parts {
            commitment.0.append(&mut part.commitment.0);
            commitment.1.append(&mut part.commitment.1);
            matrices.append(&mut part.matrices);
        }
        (
            commitment.clone(),
            Self {
                commitment,
                matrices,
            },
        )
    }
}

#[derive(Debug)]
pub enum KzgError {
    /// Commitment/opened-value dimensions disagree with the rounds.
    ShapeMismatch,
    /// The batched pairing equation does not hold.
    PairingCheckFailed,
    DegreeBoundFailed,
}

/// See the module docs.
#[derive(Clone)]
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

    #[cfg(test)]
    fn commit_columns(&self, columns: &[Vec<Fr>]) -> Vec<G1Affine> {
        self.commit_columns_at(columns, 0)
    }

    fn commit_columns_at(&self, columns: &[Vec<Fr>], shift: usize) -> Vec<G1Affine> {
        #[cfg(feature = "kzg-cuda")]
        if super::cuda::enabled() && columns.first().is_some_and(|c| c.len() >= 1 << 12) {
            let count = columns[0].len();
            assert!(columns.iter().all(|c| c.len() == count));
            let mut commits = vec![G1Projective::zero(); columns.len()];
            let mut active = Vec::new();
            for (index, column) in columns.iter().enumerate() {
                if column.iter().skip(1).all(Zero::is_zero) {
                    commits[index] = self.srs.g1[shift] * column[0];
                } else {
                    active.push(index);
                }
            }
            if !active.is_empty() {
                let scalars: Vec<_> = active.iter().map(|&i| columns[i].as_slice()).collect();
                let started = std::time::Instant::now();
                let points = super::cuda::msm_columns(&self.srs.g1[shift..shift + count], &scalars);
                tracing::debug!(
                    count,
                    columns = active.len(),
                    seconds = started.elapsed().as_secs_f64(),
                    "KZG column MSMs"
                );
                for (index, point) in active.into_iter().zip(points) {
                    commits[index] = point;
                }
            }
            return G1Projective::normalize_batch(&commits);
        }
        let commits = map_columns(columns.iter().collect(), |c| self.msm_at(c, shift));
        G1Projective::normalize_batch(&commits)
    }

    fn commit_matrix_columns(&self, matrix: &CommittedMatrix, shift: usize) -> Vec<G1Affine> {
        #[cfg(feature = "kzg-cuda")]
        if super::cuda::enabled() && matrix.domain.size() >= 1 << 16 {
            let constants = matrix.constant_columns();
            let resident = matrix.resident_columns();
            let mut commits = vec![G1Projective::zero(); matrix.columns.len()];
            let active: Vec<_> = (0..commits.len()).filter(|&i| !constants[i]).collect();
            for (i, column) in matrix.columns.iter().enumerate() {
                if constants[i] {
                    commits[i] = self.srs.g1[shift] * column[0];
                }
            }
            let inputs: Vec<_> = active
                .iter()
                .map(|&i| (matrix.columns[i].as_slice(), resident[i].as_deref()))
                .collect();
            let started = std::time::Instant::now();
            let count = matrix.domain.size();
            let points =
                super::cuda::msm_columns_resident(&self.srs.g1[shift..shift + count], &inputs);
            for (i, point) in active.iter().copied().zip(points) {
                commits[i] = point;
            }
            tracing::debug!(
                count,
                columns = active.len(),
                seconds = started.elapsed().as_secs_f64(),
                "KZG column MSMs"
            );
            return G1Projective::normalize_batch(&commits);
        }
        self.commit_columns_at(&matrix.columns, shift)
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
        self.msm_at(coeffs, 0)
    }

    fn msm_at(&self, coeffs: &[Fr], shift: usize) -> G1Projective {
        if coeffs.iter().skip(1).all(Zero::is_zero) {
            return self.srs.g1[shift] * coeffs.first().copied().unwrap_or(Fr::ZERO);
        }
        #[cfg(feature = "kzg-cuda")]
        if super::cuda::enabled() && coeffs.len() >= 1 << 12 {
            return super::cuda::msm(&self.srs.g1[shift..shift + coeffs.len()], coeffs);
        }
        G1Projective::msm(&self.srs.g1[shift..shift + coeffs.len()], coeffs).expect("equal lengths")
    }

    fn transform(values: &mut Vec<Fr>, domain: Radix2Coset, inverse: bool) {
        #[cfg(feature = "kzg-cuda")]
        if super::cuda::enabled() && domain.size() >= 1 << 10 {
            values.resize(domain.size(), Fr::ZERO);
            super::cuda::fft(values, inverse, domain.shift.0);
            return;
        }
        let ark = Self::ark_domain(domain);
        if inverse {
            ark.ifft_in_place(values);
        } else {
            ark.fft_in_place(values);
        }
    }

    /// Interpolate each column of `matrix` over `domain`.
    fn interpolate_columns(
        domain: Radix2Coset,
        matrix: &RowMajorMatrix<Scalar>,
    ) -> CommittedMatrix {
        assert_eq!(
            matrix.height(),
            domain.size(),
            "matrix height != domain size"
        );
        let width = matrix.width();
        let started = std::time::Instant::now();
        use p3_maybe_rayon::prelude::*;
        let evals: Vec<Vec<Fr>> = (0..width)
            .into_par_iter()
            .map(|column| {
                super::buffer::generate(matrix.height(), |row| {
                    matrix.values[row * width + column].0
                })
            })
            .collect();
        tracing::debug!(
            width,
            height = matrix.height(),
            seconds = started.elapsed().as_secs_f64(),
            "KZG transpose columns"
        );
        let started = std::time::Instant::now();
        let mut constants = vec![false; width];
        let columns = map_columns(evals, |col| {
            let first = col[0];
            (col.iter().all(|&v| v == first), col)
        });
        let mut output = vec![Vec::new(); width];
        let mut active = Vec::new();
        let mut inputs = Vec::new();
        for (i, (constant, mut col)) in columns.into_iter().enumerate() {
            constants[i] = constant;
            if constant {
                let first = col[0];
                col.par_iter_mut().for_each(|value| *value = Fr::ZERO);
                col[0] = first;
                output[i] = col;
            } else {
                active.push(i);
                inputs.push(col);
            }
        }
        let result = CommittedMatrix::new(domain, vec![]);
        let columns = {
            #[cfg(feature = "kzg-cuda")]
            if super::cuda::enabled() && domain.size() >= 1 << 10 {
                let (columns, residents) =
                    super::cuda::interpolate_columns(inputs, domain.log_size, domain.shift.0);
                let mut all_residents = vec![None; width];
                for (&i, resident) in active.iter().zip(residents) {
                    all_residents[i] = resident;
                }
                let _ = result.resident.set(all_residents);
                inputs = columns;
            } else {
                inputs = map_columns(inputs, |mut column| {
                    Self::transform(&mut column, domain, true);
                    column
                });
            }
            #[cfg(not(feature = "kzg-cuda"))]
            {
                inputs = map_columns(inputs, |mut column| {
                    Self::transform(&mut column, domain, true);
                    column
                });
            }
            for (i, column) in active.into_iter().zip(inputs) {
                output[i] = column;
            }
            output
        };
        tracing::debug!(
            width,
            height = matrix.height(),
            seconds = started.elapsed().as_secs_f64(),
            "KZG column interpolation"
        );
        let _ = result.constants.set(constants);
        CommittedMatrix { columns, ..result }
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
            .map(|(domain, matrix)| Self::interpolate_columns(domain, &matrix))
            .collect();
        let commitment = KzgCommitment(
            matrices
                .iter()
                .map(|m| self.commit_matrix_columns(m, 0))
                .collect(),
            matrices
                .iter()
                .map(|m| {
                    let shift = self.srs.max_len() - m.domain.size();
                    if shift == 0 {
                        return vec![];
                    }
                    self.commit_matrix_columns(m, shift)
                })
                .collect(),
        );
        (
            commitment.clone(),
            KzgProverData {
                matrices,
                commitment,
            },
        )
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
                let coefficient_columns =
                    Self::interpolate_columns(quotient_domain, &evaluations).columns;
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
                CommittedMatrix::new(
                    Radix2Coset {
                        log_size: p3_util::log2_strict_usize(n),
                        shift: crate::traits::Algebra::<Scalar>::ONE,
                    },
                    columns,
                )
            })
            .collect();
        let commitment = KzgCommitment(
            matrices
                .iter()
                .map(|m| self.commit_matrix_columns(m, 0))
                .collect(),
            matrices
                .iter()
                .map(|m| {
                    let shift = self.srs.max_len() - m.domain.size();
                    if shift == 0 {
                        return vec![];
                    }
                    self.commit_matrix_columns(m, shift)
                })
                .collect(),
        );
        (
            commitment.clone(),
            KzgProverData {
                matrices,
                commitment,
            },
        )
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
        let started = std::time::Instant::now();
        enum Column {
            Constant(Fr),
            Values(Vec<Fr>),
        }
        let constants = matrix.constant_columns();
        let mut column_evals: Vec<_> = matrix
            .columns
            .iter()
            .map(|c| Column::Constant(c.first().copied().unwrap_or(Fr::ZERO)))
            .collect();
        let active: Vec<_> = (0..matrix.columns.len())
            .filter(|&i| !constants[i])
            .collect();
        let transform_columns = || {
            #[cfg(feature = "kzg-cuda")]
            if super::cuda::enabled() && domain.size() >= 1 << 10 {
                let residents = matrix.resident_columns();
                let inputs: Vec<_> = active
                    .iter()
                    .map(|&i| (matrix.columns[i].as_slice(), residents[i].as_deref()))
                    .collect();
                return super::cuda::fft_columns(&inputs, domain.log_size, false, domain.shift.0);
            }
            map_columns(active.clone(), |i| {
                let c = &matrix.columns[i];
                let mut values = super::buffer::generate(c.len(), |i| c[i]);
                Self::transform(&mut values, domain, false);
                values
            })
        };
        let transformed = transform_columns();
        for (i, values) in active.into_iter().zip(transformed) {
            column_evals[i] = Column::Values(values);
        }
        let transform_seconds = started.elapsed().as_secs_f64();
        let width = column_evals.len();
        let height = domain.size();
        let started = std::time::Instant::now();
        let values = super::buffer::generate(width * height, |index| {
            Scalar(match &column_evals[index % width] {
                Column::Constant(value) => *value,
                Column::Values(column) => column[index / width],
            })
        });
        tracing::debug!(
            width,
            height,
            transform_seconds,
            transpose_seconds = started.elapsed().as_secs_f64(),
            "KZG evaluation reconstruction"
        );
        RowMajorMatrix::new(values, width)
    }

    fn open(
        &self,
        rounds: OpeningRounds<'_, KzgProverData, Scalar>,
        challenger: &mut Blake3Transcript,
    ) -> (OpenedValues<Scalar>, KzgProof) {
        for (data, _) in &rounds {
            challenger.observe_commitment(data.commitment.clone());
            for m in &data.matrices {
                challenger.observe_canonical(&(m.domain.log_size as u64));
            }
        }
        let _degree_challenge = challenger.sample_challenge();
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
            for (index, (matrix, points)) in data.matrices.iter().zip(points_per_matrix).enumerate()
            {
                if !points.is_empty() {
                    assert_eq!(
                        matrix.columns.len(),
                        data.commitment.0[index].len(),
                        "missing polynomial data for requested openings"
                    );
                }
                let mut matrix_values = Vec::new();
                let zs: Vec<_> = points.iter().map(|z| z.0).collect();
                let constants = matrix.constant_columns();
                #[cfg(feature = "kzg-cuda")]
                let residents =
                    (super::cuda::enabled() && !zs.is_empty()).then(|| matrix.resident_columns());
                let column_values =
                    map_columns(matrix.columns.iter().enumerate().collect(), |(i, c)| {
                        if constants[i] {
                            return vec![c.first().copied().unwrap_or(Fr::ZERO); zs.len()];
                        }
                        #[cfg(feature = "kzg-cuda")]
                        if super::cuda::enabled() && c.len() >= 1 << 12 {
                            return super::cuda::evaluate_resident(
                                c,
                                residents.and_then(|r| r[i].as_deref()),
                                &zs,
                            );
                        }
                        eval_poly_many(c, &zs)
                    });
                for (point_index, &z) in points.iter().enumerate() {
                    let row: Vec<_> = column_values
                        .iter()
                        .map(|values| Scalar(values[point_index]))
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
                    fold_polynomial(&mut combined, coeffs, power);
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
        for (commitment, matrices) in &rounds {
            challenger.observe_commitment(commitment.clone());
            for (domain, _) in matrices {
                challenger.observe_canonical(&(domain.log_size as u64));
            }
        }
        let degree_challenge = challenger.sample_challenge().0;
        let mut weight = Fr::ONE;
        let mut shifted_sum = G1Projective::zero();
        let mut by_degree = vec![G1Projective::zero(); self.srs.degree_keys.len()];
        for (commitment, matrices) in &rounds {
            if commitment.0.len() != matrices.len() || commitment.1.len() != matrices.len() {
                return Err(KzgError::ShapeMismatch);
            }
            for ((columns, shifted), (domain, _)) in
                commitment.0.iter().zip(&commitment.1).zip(matrices)
            {
                let log = domain.log_size;
                if log >= by_degree.len() {
                    return Err(KzgError::ShapeMismatch);
                }
                if domain.size() == self.srs.max_len() {
                    if !shifted.is_empty() {
                        return Err(KzgError::ShapeMismatch);
                    }
                } else {
                    if shifted.len() != columns.len() {
                        return Err(KzgError::ShapeMismatch);
                    }
                    for (&c, &s) in columns.iter().zip(shifted) {
                        by_degree[log] += c * weight;
                        shifted_sum += s * weight;
                        weight *= degree_challenge;
                    }
                }
            }
        }
        let mut g1 = vec![shifted_sum.into_affine()];
        let mut g2 = vec![self.srs.g2];
        for (sum, key) in by_degree.into_iter().zip(&self.srs.degree_keys) {
            if !sum.is_zero() {
                g1.push((-sum).into_affine());
                g2.push(*key);
            }
        }
        if !Bls12_381::multi_pairing(g1, g2).is_zero() {
            return Err(KzgError::DegreeBoundFailed);
        }
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

fn eval_poly_many(coeffs: &[Fr], points: &[Fr]) -> Vec<Fr> {
    #[cfg(feature = "kzg-cuda")]
    if super::cuda::enabled() && coeffs.len() >= 1 << 12 {
        return super::cuda::evaluate_many(coeffs, points);
    }
    points
        .iter()
        .map(|z| coeffs.iter().rev().fold(Fr::ZERO, |acc, c| acc * z + c))
        .collect()
}

fn fold_polynomial(combined: &mut [Fr], coefficients: &[Fr], power: Fr) {
    if coefficients.len() >= 1 << 16 {
        use p3_maybe_rayon::prelude::*;
        combined
            .par_iter_mut()
            .zip(coefficients)
            .for_each(|(acc, c)| *acc += power * c);
        return;
    }
    for (acc, c) in combined.iter_mut().zip(coefficients) {
        *acc += power * c;
    }
}

/// The quotient of `p` by `(X − z)` (synthetic division; the remainder
/// — `p(z)` — is dropped).
fn divide_by_linear(p: &[Fr], z: Fr) -> Vec<Fr> {
    #[cfg(feature = "kzg-cuda")]
    if super::cuda::enabled() && p.len() >= 1 << 16 {
        return super::cuda::divide(p, z);
    }
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

#[cfg(test)]
mod degree_tests {
    use super::*;
    use crate::traits::Field;

    #[test]
    fn parallel_checkpoint_encoding_preserves_canonical_format() {
        let pcs = KzgPcs::new(Arc::new(Srs::unsafe_dev_setup(8, b"checkpoint-format")), 4);
        let (_, data) = pcs.commit(
            [(8, 2), (2, 1)]
                .into_iter()
                .map(|(height, width)| {
                    (
                        pcs.natural_domain_for_degree(height),
                        RowMajorMatrix::new(
                            (0..height * width)
                                .map(|i| Scalar::from_usize(i * 7 + 3))
                                .collect(),
                            width,
                        ),
                    )
                })
                .collect(),
        );
        let mut reference = Vec::new();
        data.commitment
            .0
            .serialize_compressed(&mut reference)
            .unwrap();
        data.commitment
            .1
            .serialize_compressed(&mut reference)
            .unwrap();
        (data.matrices.len() as u64)
            .serialize_compressed(&mut reference)
            .unwrap();
        for matrix in &data.matrices {
            (matrix.domain.log_size as u64)
                .serialize_compressed(&mut reference)
                .unwrap();
            matrix
                .domain
                .shift
                .0
                .serialize_compressed(&mut reference)
                .unwrap();
            matrix.columns.serialize_compressed(&mut reference).unwrap();
        }
        let mut actual = Vec::new();
        data.write_checkpoint(&mut actual).unwrap();
        assert_eq!(actual, reference);
        let restored = KzgProverData::read_checkpoint(reference.as_slice()).unwrap();
        for (restored, original) in restored.matrices.iter().zip(&data.matrices) {
            assert_eq!(restored.domain, original.domain);
            assert_eq!(restored.columns, original.columns);
        }
        actual.pop();
        assert!(KzgProverData::read_checkpoint(actual.as_slice()).is_err());
    }

    #[test]
    fn batched_openings_preserve_point_column_and_transcript_order() {
        let n = 1 << 12;
        let pcs = KzgPcs::new(Arc::new(Srs::unsafe_dev_setup(n, b"opening-order")), 4);
        let domain = pcs.natural_domain_for_degree(n);
        let columns: Vec<Vec<Fr>> = [3u64, 11]
            .into_iter()
            .map(|factor| {
                (0..n)
                    .map(|i| Fr::from(factor * i as u64 + 7).square())
                    .collect()
            })
            .collect();
        let commits: Vec<_> = columns
            .iter()
            .map(|c| G1Projective::msm(&pcs.srs.g1, c).unwrap())
            .collect();
        let commitment = KzgCommitment(vec![G1Projective::normalize_batch(&commits)], vec![vec![]]);
        let data = KzgProverData {
            commitment: commitment.clone(),
            matrices: vec![CommittedMatrix::new(domain, columns)],
        };
        let points = [vec![0, 19, 5, 19], vec![], vec![5, 0]]
            .map(|round| round.into_iter().map(Scalar::from_u64).collect::<Vec<_>>());
        let mut prover = Blake3Transcript::new();
        let (opened, proof) = pcs.open(
            points
                .iter()
                .map(|points| (&data, vec![points.clone()]))
                .collect(),
            &mut prover,
        );
        let mut rounds = Vec::new();
        for (values, points) in opened.iter().zip(&points) {
            assert_eq!(values.len(), 1);
            assert_eq!(values[0].len(), points.len());
            let mut openings = Vec::new();
            for (row, &z) in values[0].iter().zip(points) {
                let expected: Vec<_> = data.matrices[0]
                    .columns
                    .iter()
                    .map(|coefficients| {
                        Scalar(coefficients.iter().rev().fold(Fr::ZERO, |v, c| v * z.0 + c))
                    })
                    .collect();
                assert_eq!(*row, expected);
                openings.push((z, expected));
            }
            rounds.push((commitment.clone(), vec![(domain, openings)]));
        }
        assert_eq!(proof.0.len(), 3);
        let mut verifier = Blake3Transcript::new();
        pcs.verify(rounds, &proof, &mut verifier).unwrap();
        assert_eq!(prover.sample_challenge(), verifier.sample_challenge());
    }

    #[test]
    fn mixed_degree_openings_enforce_every_bound() {
        let pcs = KzgPcs::new(Arc::new(Srs::unsafe_dev_setup(16, b"degree-test")), 4);
        let domains = [
            pcs.natural_domain_for_degree(2),
            pcs.natural_domain_for_degree(16),
        ];
        let (commitment, data) = pcs.commit(
            domains
                .into_iter()
                .map(|d| {
                    (
                        d,
                        RowMajorMatrix::new_col(vec![Scalar::from_u8(7); d.size()]),
                    )
                })
                .collect(),
        );
        let z = Scalar::from_u8(19);
        let (values, proof) = pcs.open(
            vec![(&data, vec![vec![z], vec![z]])],
            &mut Blake3Transcript::new(),
        );
        let rounds = vec![(
            commitment.clone(),
            domains
                .into_iter()
                .zip(values[0].iter())
                .map(|(d, v)| (d, vec![(z, v[0].clone())]))
                .collect(),
        )];
        pcs.verify(rounds.clone(), &proof, &mut Blake3Transcript::new())
            .unwrap();
        let mut missing = rounds.clone();
        missing[0].0.1[0].clear();
        assert!(
            pcs.verify(missing, &proof, &mut Blake3Transcript::new())
                .is_err()
        );
        let mut corrupt = rounds;
        corrupt[0].0.1[0][0] = pcs.srs.g1[0];
        assert!(matches!(
            pcs.verify(corrupt, &proof, &mut Blake3Transcript::new()),
            Err(KzgError::DegreeBoundFailed)
        ));

        // A valid KZG opening for a polynomial too large for its declared
        // two-row domain must be rejected, even though it fits the global SRS.
        let columns = vec![vec![Fr::ZERO, Fr::ZERO, Fr::ONE]];
        let commitment = KzgCommitment(
            pcs.commit_columns(&columns)
                .into_iter()
                .map(|c| vec![c])
                .collect(),
            vec![vec![pcs.srs.g1[0]]],
        );
        let data = KzgProverData {
            commitment: commitment.clone(),
            matrices: vec![CommittedMatrix::new(domains[0], columns)],
        };
        let (values, proof) = pcs.open(vec![(&data, vec![vec![z]])], &mut Blake3Transcript::new());
        assert!(matches!(
            pcs.verify(
                vec![(
                    commitment,
                    vec![(domains[0], vec![(z, values[0][0][0].clone())])]
                )],
                &proof,
                &mut Blake3Transcript::new()
            ),
            Err(KzgError::DegreeBoundFailed)
        ));
    }
}
