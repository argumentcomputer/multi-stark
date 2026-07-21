//! Generic surface of the two-staged lookup protocol.
//!
//! Circuits describe their lookups as a list of [`Interaction`]s — a
//! scheme-independent frontend vocabulary. The [`LookupArgument`] trait
//! abstracts the lookup argument scheme that lowers those interactions into
//! committed claims and proves that the claim multiset balances across all
//! circuits (and the public claims). The surrounding protocol is
//! scheme-independent and fixes:
//!
//! - two extension-field challenges (the lookup challenge and the fingerprint
//!   challenge), sampled after the stage 1 traces and the public claims are
//!   bound to the transcript;
//! - one stage 2 trace per circuit, committed after the challenges are
//!   sampled, whose width and contents are scheme-defined;
//! - a per-circuit accumulator chain: each circuit receives an incoming
//!   accumulator value and produces an outgoing one. The chain is seeded by
//!   folding the public claims ([`LookupArgument::fold_claims`]) and must end
//!   at the scheme's balance target ([`LookupArgument::balance_target`]).
//!
//! [`LogUp`](crate::logup::LogUp) is the reference implementation.

use p3_air::{Air, BaseAir, WindowAccess};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use crate::builder::{TwoStagedBuilder, symbolic::SymbolicExpression};

/// Each circuit is required to have 4 arguments for the second stage. Namely,
/// the lookup challenge, fingerprint challenge, current accumulator and next
/// accumulator.
pub const LOOKUP_PUBLIC_SIZE: usize = 4;

/// One lookup interaction of a circuit, as described by its frontend.
///
/// Interactions come in two channels:
///
/// - **Permutation channel** ([`Push`](Interaction::Push)/
///   [`Pull`](Interaction::Pull)): claims are added and removed one at a
///   time; the multiset of pushed claims must equal the multiset of pulled
///   claims. `guard` switches the claim on and off (e.g. for padding or
///   branch rows) and is expected to be boolean; circuits should constrain
///   it.
/// - **Multiplicity channel** ([`Provide`](Interaction::Provide)/
///   [`Require`](Interaction::Require)): a claim required `n` times in total
///   must be provided with total multiplicity `n`, so a single trace row can
///   serve a claim made by many rows.
///
/// The two channels are semantically distinct: match pushes with pulls and
/// requires with provides. Whether claims can also match *across* channels
/// (a push cancelling a provide), or with a non-boolean guard, is
/// scheme-defined behavior: logUp folds everything into one signed
/// accumulator (so it happens to work), while other schemes may separate the
/// channels structurally or rely on guard booleanity for soundness. Portable
/// circuits must not depend on it.
pub enum Interaction<F: Field> {
    /// Adds a claim to the permutation channel once (when `guard` is one).
    Push {
        guard: SymbolicExpression<F>,
        args: Vec<SymbolicExpression<F>>,
    },
    /// Removes a claim from the permutation channel once (when `guard` is one).
    Pull {
        guard: SymbolicExpression<F>,
        args: Vec<SymbolicExpression<F>>,
    },
    /// Serves `multiplicity` matching requires in the multiplicity channel.
    Provide {
        multiplicity: SymbolicExpression<F>,
        args: Vec<SymbolicExpression<F>>,
    },
    /// Makes a claim served by a provide, `multiplicity` times.
    Require {
        multiplicity: SymbolicExpression<F>,
        args: Vec<SymbolicExpression<F>>,
    },
}

impl<F: Field> Interaction<F> {
    /// An unconditional [`Interaction::Push`].
    #[inline]
    pub fn push(args: Vec<SymbolicExpression<F>>) -> Self {
        Self::Push {
            guard: SymbolicExpression::ONE,
            args,
        }
    }

    /// An [`Interaction::Push`] switched by a boolean guard.
    #[inline]
    pub fn push_when(guard: SymbolicExpression<F>, args: Vec<SymbolicExpression<F>>) -> Self {
        Self::Push { guard, args }
    }

    /// An unconditional [`Interaction::Pull`].
    #[inline]
    pub fn pull(args: Vec<SymbolicExpression<F>>) -> Self {
        Self::Pull {
            guard: SymbolicExpression::ONE,
            args,
        }
    }

    /// An [`Interaction::Pull`] switched by a boolean guard.
    #[inline]
    pub fn pull_when(guard: SymbolicExpression<F>, args: Vec<SymbolicExpression<F>>) -> Self {
        Self::Pull { guard, args }
    }

    /// An [`Interaction::Provide`].
    #[inline]
    pub fn provide(multiplicity: SymbolicExpression<F>, args: Vec<SymbolicExpression<F>>) -> Self {
        Self::Provide { multiplicity, args }
    }

    /// An [`Interaction::Require`].
    #[inline]
    pub fn require(multiplicity: SymbolicExpression<F>, args: Vec<SymbolicExpression<F>>) -> Self {
        Self::Require { multiplicity, args }
    }
}

/// A circuit's claim set, main trace and optional preprocessed trace, as
/// consumed by [`LookupArgument::compute_witness`].
pub type CircuitWitnessInput<'a, F, L> =
    (&'a L, &'a RowMajorMatrix<F>, Option<&'a RowMajorMatrix<F>>);

/// A lookup argument scheme.
///
/// A value of an implementing type holds the scheme's *entire* claim set for
/// one circuit, lowered from the circuit's [`Interaction`]s by
/// [`LookupArgument::new`]. Lowering is not one-to-one in general: logUp maps
/// every interaction to a single signed-multiplicity term, while a
/// grand-product scheme may expand a provide/require into several raw claims
/// plus counter bookkeeping.
///
/// The remaining items drive the scheme-specific parts of the protocol:
/// stage 2 shape ([`stage_2_width`](Self::stage_2_width)), witness generation
/// ([`compute_witness`](Self::compute_witness)), stage 2 trace construction
/// ([`stage_2_traces`](Self::stage_2_traces)), stage 2 constraint evaluation
/// ([`eval`](Self::eval)), and the claim-folding/balance bookkeeping shared by
/// the prover and the verifier.
pub trait LookupArgument<F: Field>: Send + Sync + Sized {
    /// Concrete lookup data of one circuit, computed from its committed
    /// traces during witness generation.
    type Witness: Clone + Send + Sync;

    /// Lowers a circuit's interactions into the scheme's claim set.
    fn new(interactions: Vec<Interaction<F>>) -> Self;

    /// Width, in extension field columns, of this circuit's stage 2 trace.
    fn stage_2_width(&self) -> usize;

    /// Computes the concrete lookup data of every circuit, given each
    /// circuit's claim set, main trace and optional preprocessed trace.
    ///
    /// This is a single pass over all circuits (rather than a per-circuit
    /// function) so that schemes with globally ordered witness state — such
    /// as the counters of an offline-memory-checking lowering — can assign it
    /// consistently across circuits.
    fn compute_witness(circuits: &[CircuitWitnessInput<'_, F, Self>]) -> Vec<Self::Witness>;

    /// Computes the stage 2 traces and the outgoing accumulator value of each
    /// circuit, given the lookup challenges and the incoming accumulator of
    /// the first circuit (obtained from [`LookupArgument::fold_claims`]).
    fn stage_2_traces<EF: ExtensionField<F>>(
        witnesses: &[Self::Witness],
        lookup_challenge: EF,
        fingerprint_challenge: &EF,
        claims_accumulator: EF,
    ) -> (Vec<RowMajorMatrix<EF>>, Vec<EF>);

    /// Folds the public claims into the accumulator value that seeds the
    /// accumulator chain. Both the prover and the verifier compute this from
    /// public data.
    fn fold_claims<EF: ExtensionField<F>>(
        lookup_challenge: EF,
        fingerprint_challenge: &EF,
        claims: &[&[F]],
    ) -> EF;

    /// The value the final outgoing accumulator must equal for the claim
    /// multiset to balance: zero for logUp (a running sum), one for a
    /// grand-product argument (a running product).
    fn balance_target<EF: ExtensionField<F>>() -> EF;

    /// Evaluates the scheme's stage 2 constraints for this circuit. The four
    /// stage 2 public values are, in order: the lookup challenge, the
    /// fingerprint challenge, the circuit's incoming accumulator and its
    /// outgoing accumulator (see [`LOOKUP_PUBLIC_SIZE`]).
    fn eval<AB>(&self, builder: &mut AB, preprocessed_row: Option<&[AB::Var]>)
    where
        AB: TwoStagedBuilder<F = F>;
}

/// An AIR bundled with its lookup claim set under a lookup argument scheme `L`.
pub struct LookupAir<A, F: Field, L> {
    pub inner_air: A,
    pub lookups: L,
    pub preprocessed: Option<RowMajorMatrix<F>>,
}

impl<A: BaseAir<F>, F: Field, L: LookupArgument<F>> LookupAir<A, F, L> {
    pub fn new(inner_air: A, interactions: Vec<Interaction<F>>) -> Self {
        let preprocessed = inner_air.preprocessed_trace();
        Self {
            inner_air,
            lookups: L::new(interactions),
            preprocessed,
        }
    }

    /// Width of the circuit's stage 2 trace, as defined by the scheme.
    pub fn stage_2_width(&self) -> usize {
        self.lookups.stage_2_width()
    }
}

/// Computes a fingerprint of the coefficients using Horner's method.
#[inline]
pub fn fingerprint<F, I, Iter>(r: &F, coeffs: Iter) -> F
where
    F: PrimeCharacteristicRing,
    I: Into<F>,
    Iter: DoubleEndedIterator<Item = I>,
{
    coeffs
        .rev()
        .fold(F::ZERO, |acc, coeff| acc * r.clone() + coeff.into())
}

impl<A, F, L> BaseAir<F> for LookupAir<A, F, L>
where
    A: BaseAir<F>,
    F: Field,
    L: Send + Sync,
{
    fn width(&self) -> usize {
        self.inner_air.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.preprocessed.clone()
    }
}

impl<A, F, L, AB> Air<AB> for LookupAir<A, F, L>
where
    A: Air<AB>,
    F: Field,
    L: LookupArgument<F>,
    AB: TwoStagedBuilder<F = F>,
{
    fn eval(&self, builder: &mut AB) {
        // Call `eval` for regular stage 1 constraints.
        self.inner_air.eval(builder);

        // Then the scheme's stage 2 constraints.
        if self.preprocessed.is_some() {
            let preprocessed = builder.preprocessed().clone();
            let preprocessed_row = preprocessed.current_slice();
            self.lookups.eval(builder, Some(preprocessed_row));
        } else {
            self.lookups.eval(builder, None);
        }
    }
}
