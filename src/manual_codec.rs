//! Manual (de)serializer for [`crate::prover::Proof`] — no serde, no bincode.
//!
//! The library serializes `Proof` with **bincode v2** under the config
//! `standard().with_little_endian().with_fixed_int_encoding()` (see
//! [`crate::prover::Proof::to_bytes`]). This module reimplements that exact wire
//! format by hand, against a set of plain mirror structs that store field elements
//! as raw `u64` and Merkle digests as `[u64; 4]`.
//!
//! Why mirror structs instead of the real types? The on-wire `Goldilocks` value is
//! the *raw, non-canonical* `u64` (the field's internal repr), but that field is
//! `pub(crate)` *inside the `p3_goldilocks` crate* and unreachable from here. Mirror
//! structs hold the raw words directly, so the round-trip is byte-exact. Use
//! [`deserialize`] to parse the bytes produced by [`crate::prover::Proof::to_bytes`].
//!
//! ## Wire format rules (this bincode config)
//! - every `Vec`/length prefix: `u64`, 8 bytes little-endian (fixed-int encoding)
//! - `Option`: 1 tag byte (`0` = None, `1` = Some), then the value
//! - `u8`: 1 byte
//! - struct: fields in declaration order, no framing
//! - array / tuple `[T; N]` (incl. digests and the degree-2 extension): N elements
//!   back-to-back, **no length prefix**
//! - `PhantomData` / unit: 0 bytes
//!
//! ## Leaf encodings
//! - `Val` = Goldilocks            -> raw `u64`, 8 bytes LE (NOT reduced mod p)
//! - `ExtVal` = ext field degree 2 -> `[u64; 2]`, 16 bytes LE, no length
//! - Merkle digest                 -> `[u64; 4]`, 32 bytes LE, no length
//! - `Commitment` = MerkleCap      -> `{ cap: Vec<[u64;4]> }` (PhantomData = 0 bytes)
//! - any Mmcs `Proof`              -> `Vec<[u64; 4]>`
//!
//! ## Type resolution
//! ```text
//! Pcs       = TwoAdicFriPcs<Val, Dft, Mmcs, ExtMmcs>
//! PcsProof  = FriProof<ExtVal, ExtMmcs, Witness = Val, InputProof = Vec<BatchOpening<Val, Mmcs>>>
//! ```

// ════════════════════════════════════════════════════════════════════════════
// Mirror structs: a transparent, serde-free view of the proof.
// Field elements are raw u64; digests are [u64; 4].
// ════════════════════════════════════════════════════════════════════════════

/// A single Goldilocks element on the wire (raw, possibly non-canonical).
pub type Val64 = u64;
/// `ExtVal = BinomialExtensionField<Goldilocks, 2>` -> `[u64; 2]`, no length prefix.
pub type Ext = [u64; 2];
/// A Merkle digest: `[u64; DIGEST_ELEMS]` with `DIGEST_ELEMS = 4` -> 32 bytes, no length.
pub type Digest = [u64; 4];

/// `Commitment = MerkleCap<Goldilocks, [u64; 4]>`. Only `cap` is on the wire;
/// the internal `PhantomData` contributes nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleCap {
    pub cap: Vec<Digest>,
}

/// The three trace/quotient commitments at the head of a `Proof`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitments {
    pub stage_1_trace: MerkleCap,
    pub stage_2_trace: MerkleCap,
    pub quotient_chunks: MerkleCap,
}

/// `BatchOpening<Val, Mmcs>` — the per-round input opening of a FRI query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchOpening {
    /// One inner vector per matrix in the batch; each is a row of base-field values.
    pub opened_values: Vec<Vec<Val64>>,
    /// Merkle authentication path: `Vec<[u64; 4]>`.
    pub opening_proof: Vec<Digest>,
}

/// `CommitPhaseProofStep<ExtVal, ExtMmcs>` — one folding step of a FRI query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitPhaseProofStep {
    pub log_arity: u8,
    /// `arity - 1` sibling values, each an extension element.
    pub sibling_values: Vec<Ext>,
    pub opening_proof: Vec<Digest>,
}

/// `QueryProof<ExtVal, ExtMmcs, Vec<BatchOpening<Val, Mmcs>>>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryProof {
    pub input_proof: Vec<BatchOpening>,
    pub commit_phase_openings: Vec<CommitPhaseProofStep>,
}

/// `PcsProof = FriProof<ExtVal, ExtMmcs, Val, Vec<BatchOpening<Val, Mmcs>>>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FriProof {
    pub commit_phase_commits: Vec<MerkleCap>,
    /// `Witness = Val`, so each is a raw `u64`.
    pub commit_pow_witnesses: Vec<Val64>,
    pub query_proofs: Vec<QueryProof>,
    pub final_poly: Vec<Ext>,
    /// `Witness = Val`.
    pub query_pow_witness: Val64,
}

/// `OpenedValuesForRound<ExtVal> = Vec<Vec<Vec<ExtVal>>>`.
pub type OpenedRound = Vec<Vec<Vec<Ext>>>;

/// Mirror of [`crate::prover::Proof`], in serialization (declaration) order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManualProof {
    pub commitments: Commitments,
    pub intermediate_accumulators: Vec<Ext>,
    pub log_degrees: Vec<u8>,
    pub opening_proof: FriProof,
    pub quotient_opened_values: OpenedRound,
    pub preprocessed_opened_values: Option<OpenedRound>,
    pub stage_1_opened_values: OpenedRound,
    pub stage_2_opened_values: OpenedRound,
}

// ════════════════════════════════════════════════════════════════════════════
// Deserializer: bytes -> ManualProof
// ════════════════════════════════════════════════════════════════════════════

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or("length overflow while reading")?;
        if end > self.buf.len() {
            return Err(format!(
                "unexpected end of input: need {n} bytes at offset {}, have {}",
                self.pos,
                self.buf.len() - self.pos
            ));
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    /// One raw `u64`, 8 bytes little-endian.
    fn u64(&mut self) -> Result<u64, String> {
        let bytes: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(u64::from_le_bytes(bytes))
    }

    /// A length prefix: `usize` is encoded as a fixed-width `u64` under this config.
    fn len(&mut self) -> Result<usize, String> {
        Ok(self.u64()? as usize)
    }

    fn ext(&mut self) -> Result<Ext, String> {
        Ok([self.u64()?, self.u64()?])
    }

    fn digest(&mut self) -> Result<Digest, String> {
        Ok([self.u64()?, self.u64()?, self.u64()?, self.u64()?])
    }

    /// Generic length-prefixed `Vec`.
    fn vec<T>(
        &mut self,
        mut elem: impl FnMut(&mut Self) -> Result<T, String>,
    ) -> Result<Vec<T>, String> {
        let n = self.len()?;
        let mut out = Vec::with_capacity(n.min(1 << 16));
        for _ in 0..n {
            out.push(elem(self)?);
        }
        Ok(out)
    }

    fn merkle_cap(&mut self) -> Result<MerkleCap, String> {
        Ok(MerkleCap {
            cap: self.vec(Self::digest)?,
        })
    }

    fn opened_round(&mut self) -> Result<OpenedRound, String> {
        // Vec<Vec<Vec<Ext>>>
        self.vec(|r| r.vec(|r| r.vec(Self::ext)))
    }

    fn batch_opening(&mut self) -> Result<BatchOpening, String> {
        Ok(BatchOpening {
            opened_values: self.vec(|r| r.vec(Self::u64))?,
            opening_proof: self.vec(Self::digest)?,
        })
    }

    fn commit_phase_step(&mut self) -> Result<CommitPhaseProofStep, String> {
        Ok(CommitPhaseProofStep {
            log_arity: self.u8()?,
            sibling_values: self.vec(Self::ext)?,
            opening_proof: self.vec(Self::digest)?,
        })
    }

    fn query_proof(&mut self) -> Result<QueryProof, String> {
        Ok(QueryProof {
            input_proof: self.vec(Self::batch_opening)?,
            commit_phase_openings: self.vec(Self::commit_phase_step)?,
        })
    }

    fn fri_proof(&mut self) -> Result<FriProof, String> {
        Ok(FriProof {
            commit_phase_commits: self.vec(Self::merkle_cap)?,
            commit_pow_witnesses: self.vec(Self::u64)?,
            query_proofs: self.vec(Self::query_proof)?,
            final_poly: self.vec(Self::ext)?,
            query_pow_witness: self.u64()?,
        })
    }

    fn proof(&mut self) -> Result<ManualProof, String> {
        Ok(ManualProof {
            commitments: Commitments {
                stage_1_trace: self.merkle_cap()?,
                stage_2_trace: self.merkle_cap()?,
                quotient_chunks: self.merkle_cap()?,
            },
            intermediate_accumulators: self.vec(Self::ext)?,
            log_degrees: self.vec(Self::u8)?,
            opening_proof: self.fri_proof()?,
            quotient_opened_values: self.opened_round()?,
            // Option: 1 tag byte, then the value when Some.
            preprocessed_opened_values: match self.u8()? {
                0 => None,
                1 => Some(self.opened_round()?),
                t => {
                    return Err(format!(
                        "invalid Option tag {t} for preprocessed_opened_values"
                    ));
                }
            },
            stage_1_opened_values: self.opened_round()?,
            stage_2_opened_values: self.opened_round()?,
        })
    }
}

/// Parse the full bincode byte stream into a [`ManualProof`], requiring that every
/// byte is consumed.
pub fn deserialize(bytes: &[u8]) -> Result<ManualProof, String> {
    let mut r = Reader::new(bytes);
    let proof = r.proof()?;
    if r.pos != bytes.len() {
        return Err(format!(
            "trailing data: consumed {} of {} bytes",
            r.pos,
            bytes.len()
        ));
    }
    Ok(proof)
}

// ════════════════════════════════════════════════════════════════════════════
// Serializer: ManualProof -> bytes
// ════════════════════════════════════════════════════════════════════════════

fn w_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}
fn w_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn w_len(out: &mut Vec<u8>, n: usize) {
    w_u64(out, n as u64);
}
fn w_ext(out: &mut Vec<u8>, e: &Ext) {
    w_u64(out, e[0]);
    w_u64(out, e[1]);
}
fn w_digest(out: &mut Vec<u8>, d: &Digest) {
    for &x in d {
        w_u64(out, x);
    }
}
/// Length-prefixed `Vec` writer.
fn w_vec<T>(out: &mut Vec<u8>, items: &[T], mut elem: impl FnMut(&mut Vec<u8>, &T)) {
    w_len(out, items.len());
    for item in items {
        elem(out, item);
    }
}

fn w_merkle_cap(out: &mut Vec<u8>, c: &MerkleCap) {
    w_vec(out, &c.cap, w_digest);
}

fn w_opened_round(out: &mut Vec<u8>, r: &OpenedRound) {
    w_vec(out, r, |out, matrix| {
        w_vec(out, matrix, |out, point| {
            w_vec(out, point, w_ext);
        });
    });
}

fn w_batch_opening(out: &mut Vec<u8>, b: &BatchOpening) {
    w_vec(out, &b.opened_values, |out, row| {
        w_vec(out, row, |out, &v| w_u64(out, v));
    });
    w_vec(out, &b.opening_proof, w_digest);
}

fn w_commit_phase_step(out: &mut Vec<u8>, s: &CommitPhaseProofStep) {
    w_u8(out, s.log_arity);
    w_vec(out, &s.sibling_values, w_ext);
    w_vec(out, &s.opening_proof, w_digest);
}

fn w_query_proof(out: &mut Vec<u8>, q: &QueryProof) {
    w_vec(out, &q.input_proof, w_batch_opening);
    w_vec(out, &q.commit_phase_openings, w_commit_phase_step);
}

fn w_fri_proof(out: &mut Vec<u8>, p: &FriProof) {
    w_vec(out, &p.commit_phase_commits, w_merkle_cap);
    w_vec(out, &p.commit_pow_witnesses, |out, &v| w_u64(out, v));
    w_vec(out, &p.query_proofs, w_query_proof);
    w_vec(out, &p.final_poly, w_ext);
    w_u64(out, p.query_pow_witness);
}

/// Serialize a [`ManualProof`] into bytes identical to
/// [`crate::prover::Proof::to_bytes`].
pub fn serialize(p: &ManualProof) -> Vec<u8> {
    let mut out = Vec::new();
    w_merkle_cap(&mut out, &p.commitments.stage_1_trace);
    w_merkle_cap(&mut out, &p.commitments.stage_2_trace);
    w_merkle_cap(&mut out, &p.commitments.quotient_chunks);
    w_vec(&mut out, &p.intermediate_accumulators, w_ext);
    w_vec(&mut out, &p.log_degrees, |out, &v| w_u8(out, v));
    w_fri_proof(&mut out, &p.opening_proof);
    w_opened_round(&mut out, &p.quotient_opened_values);
    match &p.preprocessed_opened_values {
        None => w_u8(&mut out, 0),
        Some(r) => {
            w_u8(&mut out, 1);
            w_opened_round(&mut out, r);
        }
    }
    w_opened_round(&mut out, &p.stage_1_opened_values);
    w_opened_round(&mut out, &p.stage_2_opened_values);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lookup::LookupAir;
    use crate::prover::Proof;
    use crate::system::{System, SystemWitness};
    use crate::types::{CommitmentParameters, FriParameters, Val};
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;

    /// A simple AIR checking Pythagorean triples: a² + b² == c².
    struct PythagoreanAir;

    impl<F> BaseAir<F> for PythagoreanAir {
        fn width(&self) -> usize {
            3
        }
    }

    impl<AB> Air<AB> for PythagoreanAir
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            let lhs = local[0] * local[0] + local[1] * local[1];
            let rhs = local[2] * local[2];
            builder.assert_eq(lhs, rhs);
        }
    }

    fn make_proof() -> Proof {
        let commitment_parameters = CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        };
        let air = LookupAir::new(PythagoreanAir, vec![]);
        let (system, key) = System::new(commitment_parameters, [air]);

        let f = Val::from_u32;
        let trace = RowMajorMatrix::new(
            vec![
                f(3),
                f(4),
                f(5),
                f(5),
                f(12),
                f(13),
                f(8),
                f(15),
                f(17),
                f(7),
                f(24),
                f(25),
            ],
            3,
        );
        let witness = SystemWitness::from_stage_1(vec![trace], &system);

        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 64,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        };

        let no_claims = &[];
        let proof = system.prove_multiple_claims(fri_parameters, &key, no_claims, witness);
        system
            .verify_multiple_claims(fri_parameters, no_claims, &proof)
            .expect("proof should verify");
        proof
    }

    #[test]
    fn manual_codec_matches_bincode() {
        let proof = make_proof();
        let canonical = proof.to_bytes().expect("library serialization failed");

        // Deserialize the bincode bytes by hand, consuming every byte.
        let manual = deserialize(&canonical).expect("manual deserialize failed");

        // Re-serialize by hand: must be byte-for-byte identical to bincode.
        let reencoded = serialize(&manual);
        assert_eq!(
            reencoded, canonical,
            "manual re-serialization does not match the bincode bytes"
        );

        // Manual decode -> encode -> decode is a fixpoint.
        let manual2 = deserialize(&reencoded).expect("second manual deserialize failed");
        assert_eq!(manual, manual2, "manual codec is not a fixpoint");

        // The library can still parse our hand-written bytes.
        Proof::from_bytes(&reencoded).expect("library could not parse hand-written bytes");
    }
}
