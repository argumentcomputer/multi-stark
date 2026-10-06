//! A global shard and fixed query groups, joined by a constrained BLAKE3 digest.
//! Only the complete ordered bundle verifies a FRI proof.

use ark_bls12_381::{Bls12_381, Fr};
use ark_groth16::{Groth16, PreparedVerifyingKey, Proof};
use ark_relations::r1cs::SynthesisError;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::*;
use crate::{
    plonkish::{Circuit, CircuitBuilder, CircuitStats, Value},
    types::Val,
};

pub struct QueryShardPlan<'p, 'k> {
    plan: &'p VerifierPlan<'k>,
    schema: Statement<StatementSlot>,
    options: ImplementationOptions,
    group_size: usize,
    identity: [u8; 32],
}

impl<'p, 'k> QueryShardPlan<'p, 'k> {
    pub fn new(
        plan: &'p VerifierPlan<'k>,
        schema: Statement<StatementSlot>,
        options: ImplementationOptions,
        queries_per_shard: usize,
    ) -> Result<Self, VerifierError> {
        if queries_per_shard == 0 {
            return Err(VerifierError::Profile("expected a nonzero query group"));
        }
        let identity = plan.identity(&schema)?;
        Ok(Self {
            plan,
            schema,
            options,
            group_size: queries_per_shard,
            identity,
        })
    }

    pub fn shard_count(&self) -> usize {
        1 + self.plan.shape().queries.div_ceil(self.group_size)
    }

    pub fn query_range(&self, shard: usize) -> Result<std::ops::Range<usize>, VerifierError> {
        if shard >= self.shard_count() {
            return Err(VerifierError::Profile("shard index out of range"));
        }
        if shard == 0 {
            return Ok(0..0);
        }
        let start = (shard - 1) * self.group_size;
        Ok(start
            ..start
                .saturating_add(self.group_size)
                .min(self.plan.shape().queries))
    }

    /// Public order: statement, 32 context bytes, shard index. The same context
    /// must be supplied to every proof under the prescribed trusted keys.
    pub fn build(&self, shard: usize) -> Result<(Circuit<Val>, VerifierInputs), VerifierError> {
        let mut b = CircuitBuilder::new();
        let inputs = self.constrain(&mut b, shard)?;
        Ok((b.finish(), inputs))
    }

    pub fn estimate(&self, shard: usize) -> Result<CircuitStats, VerifierError> {
        let mut b = CircuitBuilder::counting();
        self.constrain(&mut b, shard)?;
        Ok(b.stats())
    }

    fn constrain(
        &self,
        b: &mut CircuitBuilder<Val>,
        shard: usize,
    ) -> Result<VerifierInputs, VerifierError> {
        let range = self.query_range(shard)?;
        let mut bind = |slot: &StatementSlot| match slot {
            StatementSlot::Constant(c) => StatementBinding::Constant(*c),
            StatementSlot::Public => StatementBinding::Wire(b.public_input("statement")),
        };
        let statement = Statement {
            claims: self
                .schema
                .claims
                .iter()
                .map(|c| c.iter().map(&mut bind).collect())
                .collect(),
            messages: self
                .schema
                .messages
                .iter()
                .map(|m| Message {
                    args: m.args.iter().map(&mut bind).collect(),
                    multiplicity: bind(&m.multiplicity),
                })
                .collect(),
        };
        let inputs =
            self.plan
                .constrain_query_shard(b, statement, self.options, range, shard == 0)?;
        let bytes = ByteGadgets::new(b);
        let mut encoded: Vec<_> = b"multi-stark/query-shards/v1"
            .iter()
            .chain(&self.identity)
            .map(|&v| bytes.constant(b, v))
            .collect();
        // Bind the partition as well as the logical verifier identity.
        encoded.extend(
            u64::try_from(self.group_size)
                .unwrap()
                .to_le_bytes()
                .map(|v| bytes.constant(b, v)),
        );
        for value in shared_values(b, &inputs) {
            encoded.extend(bytes.encode_field64(b, value));
        }
        for byte in blake3(b, &bytes, &encoded) {
            b.expose_public(byte.value());
        }
        let index = b.constant(Val::from_u64(u64::try_from(shard).unwrap()));
        b.expose_public(index);
        Ok(inputs)
    }
}

// All non-query witness data, including values unused by a particular shard.
// Canonical field encoding and a fixed plan identity make this unambiguous.
fn shared_values(b: &mut CircuitBuilder<Val>, inputs: &VerifierInputs) -> Vec<Value> {
    let mut values = Vec::new();
    for binding in inputs.statement().claims.iter().flatten().chain(
        inputs
            .statement()
            .messages
            .iter()
            .flat_map(|m| m.args.iter().chain([&m.multiplicity])),
    ) {
        values.push(match binding {
            StatementBinding::Constant(c) => b.constant(*c),
            StatementBinding::Wire(w) => *w,
        });
    }
    let p = inputs.proof();
    let c = p.algebra.challenges;
    values.extend(
        [c.beta, c.gamma, c.alpha, c.zeta]
            .into_iter()
            .flat_map(|q| q.0),
    );
    for o in &p.algebra.openings {
        values.extend(
            o.preprocessed
                .iter()
                .chain(&o.main)
                .chain(&o.stage2)
                .flatten()
                .chain(&o.quotient)
                .chain([&o.accumulator])
                .flat_map(|q| q.0),
        );
    }
    values.extend(
        p.commitments
            .stage1
            .iter()
            .chain(&p.commitments.stage2)
            .chain(&p.commitments.quotient)
            .chain(p.pcs.roots.iter().flatten())
            .map(|v| v.value()),
    );
    values.extend(&p.pcs.commit_pow);
    values.push(p.pcs.query_pow);
    values.extend(p.pcs.final_poly.0);
    values
}

/// `keys` must be the complete ordered, trusted key set from one QueryShardPlan.
/// Never accept keys or the expected statement from the proof producer.
pub fn verify_query_bundle(
    keys: &[PreparedVerifyingKey<Bls12_381>],
    proofs: &[Proof<Bls12_381>],
    expected_statement: &[Fr],
    context: &[u8; 32],
) -> Result<bool, SynthesisError> {
    if keys.len() < 2 || keys.len() != proofs.len() {
        return Ok(false);
    }
    let mut public = expected_statement.to_vec();
    public.extend(context.iter().map(|&v| Fr::from(v)));
    public.push(Fr::from(0u64));
    for (index, (key, proof)) in keys.iter().zip(proofs).enumerate() {
        *public.last_mut().unwrap() = Fr::from(u64::try_from(index).unwrap());
        if !Groth16::<Bls12_381>::verify_proof(key, proof, &public)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn keyset_id(keys: &[PreparedVerifyingKey<Bls12_381>]) -> Result<[u8; 32], SerializationError> {
    let mut encoded = b"multi-stark/query-shard-keys/v1".to_vec();
    encoded.extend(u64::try_from(keys.len()).unwrap().to_le_bytes());
    for key in keys {
        key.vk.serialize_compressed(&mut encoded)?;
    }
    Ok(*::blake3::hash(&encoded).as_bytes())
}

/// Binary QR payload: version, key-set digest, canonical u64 statement, shared
/// context, then compressed proofs. Counts come from the trusted verifier.
pub fn encode_query_bundle(
    keys: &[PreparedVerifyingKey<Bls12_381>],
    proofs: &[Proof<Bls12_381>],
    statement: &[Val],
    context: &[u8; 32],
) -> Result<Vec<u8>, SerializationError> {
    if keys.len() < 2 || keys.len() != proofs.len() {
        return Err(SerializationError::InvalidData);
    }
    let mut bytes = b"MSQ1".to_vec();
    bytes.extend(keyset_id(keys)?);
    for value in statement {
        bytes.extend(value.as_canonical_u64().to_le_bytes());
    }
    bytes.extend(context);
    for proof in proofs {
        proof.serialize_compressed(&mut bytes)?;
    }
    Ok(bytes)
}

/// The caller supplies the expected statement and the complete trusted key set.
/// Rejects trailing data and validates compressed curve points before pairing.
pub fn verify_encoded_query_bundle(
    keys: &[PreparedVerifyingKey<Bls12_381>],
    expected_statement: &[Val],
    bytes: &[u8],
) -> Result<bool, SynthesisError> {
    let Some(length) = keys.len().checked_mul(192).and_then(|n| {
        expected_statement
            .len()
            .checked_mul(8)
            .and_then(|s| n.checked_add(s)?.checked_add(68))
    }) else {
        return Ok(false);
    };
    if keys.len() < 2 || bytes.len() != length || &bytes[..4] != b"MSQ1" {
        return Ok(false);
    }
    let id = keyset_id(keys).map_err(|_error| SynthesisError::Unsatisfiable)?;
    if bytes[4..36] != id {
        return Ok(false);
    }
    let mut cursor = 36;
    for value in expected_statement {
        if bytes[cursor..cursor + 8] != value.as_canonical_u64().to_le_bytes() {
            return Ok(false);
        }
        cursor += 8;
    }
    let context = bytes[cursor..cursor + 32].try_into().unwrap();
    cursor += 32;
    let mut proofs = Vec::with_capacity(keys.len());
    for chunk in bytes[cursor..].as_chunks::<192>().0 {
        let Ok(proof) = Proof::<Bls12_381>::deserialize_compressed(chunk.as_slice()) else {
            return Ok(false);
        };
        proofs.push(proof);
    }
    let expected: Vec<_> = expected_statement
        .iter()
        .map(|v| Fr::from(v.as_canonical_u64()))
        .collect();
    verify_query_bundle(keys, &proofs, &expected, context)
}
