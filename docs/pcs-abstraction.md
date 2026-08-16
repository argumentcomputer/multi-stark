# PCS abstraction: generalizing multi-stark over FRI and KZG

Design document. Goal: make the multi-stark prover/verifier generic over the
polynomial commitment scheme through a trait layer **owned by this crate**,
with Plonky3 (FRI/Goldilocks) and arkworks (KZG/BLS12-381) living behind
adapters. Everything above the PCS — the constraint IR, logUp, sparse
activation, claims binding, quotient slicing, Fiat-Shamir structure — stays
exactly as it is.

## 1. Why

The proving pipeline is a compression funnel:

| Stage | System | PCS | Proof size driver |
|---|---|---|---|
| 1. kernel | fat (~10k+ cols) | FRI | fast prover matters most |
| 2. recursive verifier | ~7.7k cols | FRI | `width x queries x 8B` (~4 MB @ q=50) |
| 3. final wrap | recursive verifier again | **KZG** | `width x 112B` (~0.9 MB), constant-ish opening |

FRI proof bytes are `total_width x num_queries x 8` for the opened base rows
plus ~20% Merkle/commit-phase overhead. KZG proof bytes are per *column*,
independent of queries:

- 48 B — one compressed BLS12-381 G1 commitment per column,
- 32 + 32 B — the two opened evaluations (zeta, zeta*g) per stage column
  (quotient columns open only at zeta: 32 B),
- ~2 G1 + a few Fr for the batched opening proof (amortized to zero).

So ~112 B/column: the same 7.7k-column system that costs 4.16 MB under FRI
costs ~0.87 MB under KZG, before any plonkish re-circuiting. Every column
removed upstream (inlining, grouping, packing) now pays twice — once per FRI
stage and 112 B in the wrap.

A KZG stage is terminal: its verifier runs natively (two pairings), never
in-circuit, so nothing on the ix/Aiur side changes.

## 2. Current coupling (measured, not guessed)

`StarkGenericConfig` (config.rs) binds Plonky3 traits directly:
`Pcs<Challenge, Challenger>`, `ExtensionField<Val>`, `FieldChallenger` +
`CanObserve<Com>`. The full surface the core actually uses:

- **PCS**: `natural_domain_for_degree`, `commit`, `commit_ldes` (quotient
  fast path), `open`, `verify`, `get_evaluations_on_domain` (8+2+1+1+1+3
  call sites across prover.rs/verifier.rs).
- **Domain** (`PolynomialSpace`): `selectors_at_point`, `next_point` /
  generator access, `size`.
- **Challenger**: `observe`, `observe_slice`, `observe_algebra_element`,
  `sample_algebra_element`, `CanObserve<Com>`. (FRI's `sample_bits` and
  grinding live *inside* the p3 PCS — already behind the boundary.)
- **Field/matrix plumbing**: `p3_field` (`Field`, `TwoAdicField`,
  `ExtensionField`, `PrimeCharacteristicRing`, `BasedVectorSpace`,
  packing), `p3_matrix::RowMajorMatrix`, `p3_dft` (quotient coefficient
  path), `p3_util` bit-reversal helpers.

Note: today's `p3_adapter.rs` is an **AIR frontend** adapter (p3_air ->
`CircuitInputs`), not a PCS adapter. It keeps that role and moves to
`p3_adapter/air.rs`.

## 3. The trait layer (crate-owned, in `src/traits/`)

Mirror the measured surface, nothing more. Naming convention (decided):
natural names, no prefixes — our traits are `Transcript`,
`EvaluationDomain`, `Pcs`; where a backend's name collides, the backend
import is renamed at the adapter (`use p3_commit::Pcs as P3Pcs`), never
ours. Associated types over generics throughout (inference-unambiguous,
no E0207 blanket-impl traps), no blanket impls — each config states its
instantiations concretely (impls must name concrete types anyway:
coherence cannot see through Pcs-projection aliases in impl headers).
The one deliberate fancy construct is the `Evaluations<'a>` GAT on
`Pcs` (borrowed LDE views beat copies in the constraint sweep).
Status: PHASE 0 IS COMPLETE — Transcript, EvaluationDomain, Pcs, and
the field layer (Field/TwoAdicField/Algebra/Packed/ExtensionOf/
PackedExtension) are all landed as behavioral no-ops under the
proof-bytes pin. Core imports no p3 proof-system traits; p3_matrix
(container), p3_util (log2), and p3_maybe_rayon (parallelism) remain as
utility libraries. Next: Phase 1, the ark_adapter. Original sketches
below (kept for the rationale; the landed signatures in src/traits/ are
authoritative and differ in detail):

```rust
// traits/field.rs
pub trait MsField:
    Copy + Send + Sync + Eq + Serialize + DeserializeOwned
    + Add + Sub + Mul + Neg + ... // ring ops
{
    const ZERO: Self; const ONE: Self;
    fn inverse(&self) -> Self;
    fn from_u64(x: u64) -> Self;
    /// Canonical little-endian bytes — the transcript encoding.
    fn to_canonical_bytes(&self) -> Vec<u8>; // fixed len per impl
}

pub trait MsTwoAdicField: MsField {
    const TWO_ADICITY: usize;
    fn two_adic_generator(bits: usize) -> Self;
}

/// Challenge field as a based vector space over the base field.
/// CRUCIAL: D = 1 must be a first-class case (KZG over BLS12-381 Fr —
/// |Fr| ~ 2^255 needs no extension; the identity impl makes logUp,
/// stage-2 flattening, and quotient slicing collapse their D factor).
pub trait MsChallenge<F: MsField>: MsField + From<F> {
    const D: usize;
    fn from_basis_coefficients(slice: &[F]) -> Self;
    fn as_basis_coefficients(&self) -> &[F];
}
```

```rust
// traits/pcs.rs
pub trait MsPcs {
    type F: MsTwoAdicField;
    type Challenge: MsChallenge<Self::F>;
    type Domain: MsDomain<Self::F, Self::Challenge>;
    type Commitment: Clone + Serialize + DeserializeOwned;
    type ProverData;
    type Proof: Serialize + DeserializeOwned;
    type Error: Debug;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain;

    /// The largest quotient degree (as a multiple of the trace degree)
    /// this PCS serves ECONOMICALLY. Advisory: `System::new` validates
    /// every circuit's quotient degree against it at build time (loud,
    /// per-circuit error), and the lookup-grouping policy keys on it.
    /// FRI: `1 << log_blowup` — the LDE-subsetting economy. KZG: the
    /// coset-FFT budget (SRS-independent thanks to quotient chunking).
    fn max_quotient_degree(&self) -> usize;

    /// Commit to matrices of evaluations over their domains.
    /// FRI: coset LDE + one Merkle tree over all matrices.
    /// KZG: per column, iFFT to coefficients + MSM against the monomial
    /// SRS; Commitment = Vec<G1Affine> in matrix-then-column order.
    fn commit(&self, evals: Vec<(Self::Domain, Matrix<Self::F>)>)
        -> (Self::Commitment, Self::ProverData);

    /// Commit the per-circuit quotients given by their EVALUATIONS on
    /// the disjoint quotient domain plus the quotient degree. (Chosen
    /// over a coefficients-based boundary: the evals->coeffs transform
    /// needs a DFT engine, which is backend property, so the whole
    /// conversion — slicing included — lives behind the trait. FRI:
    /// fused shifted gather + zero-padded DFT; KZG: coset iDFT + MSM.)
    fn commit_quotient(&self, quotients: Vec<(Self::Domain, Matrix<Self::F>, usize)>)
        -> (Self::Commitment, Self::ProverData);

    /// Prover-side access to committed evaluations (constraint sweep).
    fn get_evaluations_on_domain(&self, data: &Self::ProverData,
        idx: usize, domain: Self::Domain) -> impl Matrix<Self::F>;

    fn open(&self, rounds: Vec<(&Self::ProverData, Vec<Vec<Self::Challenge>>)>,
        challenger: &mut impl MsChallenger<...>)
        -> (OpenedValues<Self::Challenge>, Self::Proof);

    fn verify(&self, rounds: ..., proof: &Self::Proof,
        challenger: &mut impl MsChallenger<...>) -> Result<(), Self::Error>;
}
```

```rust
// traits/domain.rs — subgroup domains, both backends are two-adic
pub trait MsDomain<F, EF>: Copy {
    fn size(&self) -> usize;
    fn first_point(&self) -> F;
    fn next_point(&self, x: EF) -> EF;          // x * g
    fn selectors_at_point(&self, z: EF) -> LagrangeSelectors<EF>;
    fn vanishing_poly_at_point(&self, z: EF) -> EF;
}

// traits/challenger.rs
pub trait MsChallenger<F: MsField, EF, Com> {
    fn observe_field(&mut self, x: F);
    fn observe_slice(&mut self, xs: &[F]);
    fn observe_algebra(&mut self, x: EF);
    /// Commitments observe as canonical bytes: Merkle caps are digest
    /// limbs, G1 points are 48-byte compressed encodings. This is the one
    /// place the two backends' transcripts structurally differ.
    fn observe_commitment(&mut self, c: &Com);
    fn sample_algebra(&mut self) -> EF;
}

// traits/config.rs — replaces StarkGenericConfig
pub trait MsConfig {
    type F: MsTwoAdicField;
    type Challenge: MsChallenge<Self::F>;
    type Pcs: MsPcs<F = Self::F, Challenge = Self::Challenge>;
    type Challenger: MsChallenger<Self::F, Self::Challenge, Com<Self>>;
    fn pcs(&self) -> &Self::Pcs;
    fn initialise_challenger(&self) -> Self::Challenger;
    /// Protocol-parameter words bound into the challenger seed. FRI:
    /// blowup/queries/PoW/arity as today. KZG: SRS digest + degree bound.
    fn parameter_seed_words(&self) -> Vec<u64>;
}
```

**Matrix decision**: define a minimal crate-owned `Matrix<F>` (row-major
`Vec<F>` + width — what `RowMajorMatrix` is), so `ark_ff::Fr` needs no p3
trait impls. The p3 adapter converts by reinterpretation (same layout,
zero-copy where possible). Packing (`PackedVal`) is a prover-speed detail of
the constraint sweep; expose it as an associated `Packing` type on `MsField`
with a scalar (identity) default so the ark backend works unpacked first.

## 4. Adapter layout

```
src/
  traits/            field.rs  pcs.rs  domain.rs  challenger.rs  config.rs
  p3_adapter/
    mod.rs
    air.rs           (today's p3_adapter.rs, unchanged role)
    field.rs         MsField/MsTwoAdicField/MsChallenge for Goldilocks + deg-2 ext
    pcs.rs           TwoAdicFriPcs wrapped as MsPcs (commit_coefficients =
                     today's commit_ldes path)
    challenger.rs    SerializingChallenger64<Blake3> as MsChallenger
    domain.rs        TwoAdicMultiplicativeCoset as MsDomain
  ark_adapter/
    mod.rs
    field.rs         Fr (BLS12-381 scalar): MsField + MsTwoAdicField
                     (TWO_ADICITY = 32); MsChallenge with D = 1 (identity)
    domain.rs        radix-2 subgroup domain (ark-poly Radix2EvaluationDomain
                     or a ~50-line own impl to keep deps thin)
    pcs.rs           KZG (section 5)
    srs.rs           SRS loading, ceremony formats, dev-mode tau setup
    challenger.rs    same Blake3 transcript over 32-byte canonical Fr +
                     48-byte compressed G1 observations
```

Cargo features: `fri` (default, pulls p3), `kzg` (pulls ark-bls12-381,
ark-ec, ark-ff; NOT ark-poly-commit — the KZG we need is ~300 lines over
MSM + pairing, and ark-poly-commit's abstraction tax isn't worth it).
`p3_adapter/air.rs` stays under `fri` (it is the Aiur frontend's entry).

## 5. KZG backend specification

- **Commitment**: one G1 point per column. `commit` = per column: iFFT
  (n log n over Fr) to monomial coefficients, then an MSM of size n against
  the monomial SRS. `commit_coefficients` skips the iFFT (quotient slices
  arrive as coefficients — the existing slicing already caps every slice's
  degree at n, so **the SRS never needs to exceed the max trace height**).
- **Opening**: all stage-1/stage-2/preprocessed columns at {zeta, zeta*g},
  quotient columns at {zeta}. Two-point batch opening a la BDFG20 / the
  standard Plonk shape: sample batching challenge v, aggregate per point,
  one witness polynomial per point => proof = (W_zeta, W_zeta_g): 2 G1 =
  96 B + the opened evaluations. Verification: 2 pairings after combining
  commitments homomorphically (the verifier-side MSM over ~width points is
  the dominant native cost — fine for a terminal stage).
- **Transcript**: same Blake3 challenger; Fr observes as 32 canonical LE
  bytes, G1 as 48 compressed bytes. The parameter seed binds an SRS digest.
- **SRS**: needs monomial powers up to max trace height (2^22-ish for
  kernel-scale; the wrap stage proves the ~2^13-2^17 verifier system, so
  2^20 is comfortable). Source from a public ceremony (perpetual powers of
  tau covers 2^28); `srs.rs` parses that format and carries a dev-mode
  `unsafe_setup(tau)` for tests, feature-gated so it cannot ship.
- **Degrees**: Fr two-adicity 32 >> any trace height. `max_quotient_degree`
  is a FRI/blowup economy, not a protocol constant — it becomes a
  `MsPcs`-reported value (`fn max_quotient_degree(&self)`): FRI reports
  `1 << log_blowup` (today's types.rs:132), KZG reports its coset-FFT
  budget (SRS stays at `n_max` thanks to quotient chunking — see quirk 3).
  The lookup grouping policy reads the same cap, so KZG configs group
  deeper for free.

## 6. What stays identical / what simplifies

Identical: constraint graph + compilation, logUp chained accumulators and
grouping, sparse activation, claims observation, quotient construction and
slicing, verifier's OOD identity, `Commitments<Com>` / `Proof<SC>` shapes
(already generic over `Com`/`PcsProof`).

Simplifies under D = 1: stage-2 "flatten to base columns" becomes identity;
`from_ext_basis` reconstruction disappears; `stage2_width(L, k, 1) =
groups(L, k)`; opened-value rows halve. Keep all code generic over
`MsChallenge::D` — the D = 2 FRI path is unchanged, the D = 1 path just
takes the degenerate branches.

Serialization: `Proof` keeps bincode/serde; ark types get serde via
CanonicalSerialize newtype wrappers in the adapter. The ix manual codec is
untouched (it only ever reads FRI-stage proofs).

## 7. Phased plan (each phase lands green)

- **Phase 0 — carve the traits, behavioral no-op.** Introduce `traits/`,
  implement in `p3_adapter/`, port prover/verifier/system to the own traits.
  Gate: existing test suite passes AND proofs are **byte-identical** (pin a
  serialized proof hash in a test before starting; ix-side vk/codegen and
  all FFT pins must not move).
- **Phase 1 — ark adapter, KZG PCS.** Field/domain/SRS/pcs + unit tests
  (commit/open/verify roundtrip, batch opening against hand-computed
  pairings, transcript vectors).
- **Phase 2 — end-to-end.** Prove `test_circuits` and then the recursive
  verifier system under a `Bls12Kzg` config; differential test (same
  witness accepted by both configs; tampered claim rejected); bench row:
  proof size + prove/verify time vs the FRI stage. Target: ~112 B/column
  confirmed empirically.
- **Phase 3 (optional, separate design) — plonkish frontend** to collapse
  the wrap system's column count; and **linearization** (use commitment
  homomorphism to open a single linearized polynomial instead of every
  column at zeta) — halves opened values, at the cost of diverging the
  verifier logic between backends. Explicitly out of scope for v1: "the
  proof shrinks even if everything else looks exactly the same."

## 8. Quirk inventory (from the old `kzg` branch, origin/kzg `multi-plonk/`)

The branch is a full parallel rewrite whose merge-base predates the
constraint IR, logUp grouping, committed-inverse elimination, sparse
activation, quotient chunking, and deterministic PoW — do not rebase it;
mine it as a reference for the arkworks mechanics. The exact generalization
points it exposes, each of which the trait layer must own explicitly:

1. **Committed representation.** FRI commits Lagrange evaluations on a
   shifted coset (the LDE) behind one Merkle tree; KZG commits monomial
   coefficients per column (branch: `ifft_column` -> `DensePolynomial` ->
   MSM). The trait absorbs this via `commit` (evaluations in) +
   `commit_coefficients` (coefficients in) with each backend converting
   internally; `ProverData` is representation-private.
2. **Quotient-domain acquisition.** FRI reads trace evals on the quotient
   domain by subsetting the committed LDE — free, but it is what imposes
   `max_quotient_degree = 1 << log_blowup` (types.rs:132). KZG has no LDE:
   it pays a coset FFT to `qd*n` per column unconditionally (branch does
   exactly this), so no blowup-shaped cap exists. Generalization: the
   degree cap is a `Pcs`-reported economy
   (`fn max_quotient_degree(&self) -> usize`), not a protocol constant;
   FRI reports `B`, KZG reports its FFT budget. The lookup-grouping policy
   keys on it, so KZG configs group deeper automatically.
3. **Quotient chunking stays.** The branch commits the quotient un-chunked
   (degree `(qd-1)*n`), forcing `SRS >= (max_deg-1)*n_max` — its own setup
   comment admits this. Main's slicing into qd degree-<=n slices must
   survive under KZG: SRS pinned to `n_max`, smaller MSMs, slices cost
   48 B each. (Verifier recombination `Q(z) = sum z^(i*n) c_i(z)` is
   already PCS-agnostic.)
4. **Coset disjointness is shared logic.** Both backends must evaluate the
   composition on a coset disjoint from the trace subgroup (Z_H != 0 for
   the pointwise division; branch: GENERATOR coset + `zh_inv`, FRI: the
   disjoint LDE coset). The selector/vanishing-on-coset math and the
   `k_next = k * stride` next-row indexing belong in the shared sweep, not
   the adapters.
   Enforcement of quirk 2's cap is two-layered, both layers owned by the
   adapter: `max_quotient_degree()` is the advisory query (build-time
   validation + grouping policy), and `get_evaluations_on_domain` PANICS
   if asked for a domain larger than it serves economically (FRI: larger
   than the LDE). Deliberately no silent fallback: a transparent
   iFFT-and-bigger-FFT path would hide a serious prover regression behind
   a working proof. The panic is a should-never-happen backstop — the
   build-time check fires first with a per-circuit error.
5. **Commitment granularity and ordering.** FRI: one commitment per ROUND
   (all matrices, one tree). KZG: one G1 per COLUMN, a round is
   `Vec<G1>`. `Com` as an opaque associated type handles the data, but the
   proof layout and the challenger absorption need one canonical
   round -> matrix -> column order fixed by the core, and the opened-value
   indexing must not assume "one commitment therefore one Merkle proof".
6. **Opening points.** Both backends open stages at {zeta, zeta*g} and
   quotient at {zeta}; KZG batches all polynomials per point via a random
   LC into one witness commitment (branch uses Sonic's 1xG1 per point; we
   spec 2xG1 total). The `open`/`verify` trait shape (rounds of
   (data, points-per-matrix)) already covers both.
7. **Transcript.** The branch swapped to a Poseidon sponge — unnecessary:
   the KZG stage is terminal, nothing verifies its transcript in-circuit,
   so the Blake3 challenger stays for both backends (observe Fr as 32
   canonical LE bytes, G1 as 48 compressed bytes). Re-deriving the branch's
   Poseidon choice would only matter if the wrap were ever SNARK-verified.
8. **Dependency surface.** The branch pulls ark-poly-commit +
   ark-crypto-primitives (labeled polynomials, sponge traits) — the
   abstraction tax that motivated hand-rolling: MSM commit + two batched
   witness polynomials + 2 pairings is ~300 lines over ark-ec/ark-ff.

## 9. Open questions

1. Packing for the ark constraint sweep (start scalar; SIMD later if the
   wrap prover is ever hot).
2. `get_evaluations_on_domain` for KZG when the requested domain exceeds
   the trace domain (quotient sweep needs qd*n points): forward FFT from
   stored coefficients — decide whether ProverData caches evals, coeffs,
   or both.
3. Whether `natural_domain_for_degree` needs cosets at all for KZG (no —
   plain subgroup; the disjoint-coset machinery is FRI-only and stays in
   the adapter).
4. Blinding: the current system is not zero-knowledge under either PCS;
   KZG openings reveal evaluations exactly as FRI queries do. If ZK is
   ever wanted, that is a separate design (random row padding + blinded
   quotient), orthogonal to this abstraction.
5. Challenger uniformity: one Blake3 transcript implementation generic
   over "observe canonical bytes" would let both adapters share code —
   nice-to-have, not required.
