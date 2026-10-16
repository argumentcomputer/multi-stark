//! The polynomial-commitment surface the core actually uses.
//!
//! Six operations plus the degree-budget query (see
//! `docs/pcs-abstraction.md`). Associated types for the same reasons as
//! the other crate traits; the challenger is an associated type because
//! `open`/`verify` drive the transcript, and each backend requires its
//! own concrete challenger capabilities internally.

use core::fmt::Debug;

use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

use serde::Serialize;
use serde::de::DeserializeOwned;

use super::EvaluationDomain;

/// Opened values for one matrix: per opening point, the row of values.
pub type OpenedValuesForMatrix<EF> = Vec<Vec<EF>>;
/// Opened values for one commitment round: per matrix.
pub type OpenedValuesForRound<EF> = Vec<OpenedValuesForMatrix<EF>>;
/// Opened values for all rounds of one `open` call.
pub type OpenedValues<EF> = Vec<OpenedValuesForRound<EF>>;

/// Opening rounds: per committed round, per matrix, the opening points.
pub type OpeningRounds<'a, Data, EF> = Vec<(&'a Data, Vec<Vec<EF>>)>;

/// Verification rounds: per commitment, per matrix its domain and the
/// (point, claimed values) pairs.
pub type VerifyRounds<Com, Dom, EF> = Vec<(Com, Vec<(Dom, Vec<(EF, Vec<EF>)>)>)>;

/// A batch polynomial commitment scheme over matrices of trace
/// evaluations.
pub trait Pcs {
    /// Base (trace) field. The bounds are the container's needs (matrix
    /// storage is shared across prover threads), satisfied by any field
    /// element type — no proof-system semantics implied.
    type F: Clone + Send + Sync + 'static;
    /// Challenge field.
    type Challenge;
    /// Evaluation domains this PCS hands out.
    type Domain: EvaluationDomain<F = Self::F, Challenge = Self::Challenge>;
    /// The transcript driven by `open`/`verify`.
    type Challenger;
    /// Commitment type (FRI: one Merkle cap per round; KZG: one G1 per
    /// column, so a round is a vector of points).
    type Commitment: Clone + Serialize + DeserializeOwned;
    /// Prover-side retained data for committed rounds.
    type ProverData;
    /// Opening proof covering all rounds of one `open` call.
    type Proof: Clone + Serialize + DeserializeOwned;
    type Error: Debug;
    /// Prover-side view of a committed matrix's evaluations on a domain
    /// (borrowed from `ProverData`; the constraint sweep reads it).
    type Evaluations<'a>: Matrix<Self::F> + 'a;

    /// The canonical domain for traces of `degree` rows.
    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain;

    /// The largest quotient degree (as a multiple of the trace degree)
    /// this PCS serves ECONOMICALLY. Advisory: `System::new` validates
    /// every circuit against it at build time, and the lookup-grouping
    /// policy keys on it. FRI: the blowup (trace evaluations on the
    /// quotient domain subset the committed LDE). KZG: the coset-FFT
    /// budget. Implementations should fail loudly — not fall back to a
    /// silent slow path — if asked to exceed it (see
    /// [`Self::get_evaluations_on_domain`]).
    fn max_quotient_degree(&self) -> usize;

    /// Commit to matrices of evaluations over their domains, as one
    /// round.
    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Self::F>)>,
    ) -> (Self::Commitment, Self::ProverData);

    /// Commit the per-circuit quotient polynomials, given by their
    /// EVALUATIONS on the disjoint quotient domain (flattened to base
    /// columns) plus the circuit's quotient degree `q`. The backend owns
    /// everything representation-specific from here: slicing
    /// `Q(X) = Σₖ X^{k·n}·cₖ(X)` into `q` degree-`< n` coefficient
    /// slices and committing them as one `q·D`-column matrix per circuit
    /// on the trace domain (FRI: fused shifted gather + zero-padded DFT
    /// to the committed LDE; KZG: coset iDFT + per-slice MSM).
    fn commit_quotient(
        &self,
        quotients: Vec<(Self::Domain, RowMajorMatrix<Self::F>, usize)>,
    ) -> (Self::Commitment, Self::ProverData);

    /// A committed matrix's evaluations on `domain`. May panic if
    /// `domain` exceeds what this PCS serves economically (FRI: the
    /// committed LDE) — the build-time [`Self::max_quotient_degree`]
    /// check fires first.
    fn get_evaluations_on_domain<'a>(
        &self,
        data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::Evaluations<'a>;

    /// Open all rounds: per round, per matrix, the opening points.
    /// Returns the opened values in round order and the opening proof.
    fn open(
        &self,
        rounds: OpeningRounds<'_, Self::ProverData, Self::Challenge>,
        challenger: &mut Self::Challenger,
    ) -> (OpenedValues<Self::Challenge>, Self::Proof);

    /// Verify claimed opened values against commitments.
    fn verify(
        &self,
        rounds: VerifyRounds<Self::Commitment, Self::Domain, Self::Challenge>,
        proof: &Self::Proof,
        challenger: &mut Self::Challenger,
    ) -> Result<(), Self::Error>;
}
