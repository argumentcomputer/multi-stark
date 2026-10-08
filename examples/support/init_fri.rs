//! Fixed saved intermediate proof and independently expected Init claim.
#[path = "init_claim.rs"]
mod init_claim;
pub(crate) use init_claim::INIT_PUBLIC_WORDS;
use multi_stark::{
    plonkish::verifier::*,
    prover::Proof,
    types::{GoldilocksBlake3Config, Val},
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::{fs, path::Path};

pub(crate) struct Fixture {
    pub key: VerifierKey,
    pub proof: Proof<GoldilocksBlake3Config>,
    pub public: [Val; 18],
    pub claims: Vec<Vec<Val>>,
    pub profile: ProofProfile,
    pub schema: Statement<StatementSlot>,
}

pub(crate) fn load(dir: &Path) -> Result<Fixture, Box<dyn std::error::Error>> {
    let key = VerifierKey::from_bytes(&fs::read(dir.join("outer-vk.bin"))?)?;
    let bytes = fs::read(dir.join("outer-proof.bin"))?;
    let proof = Proof::<GoldilocksBlake3Config>::from_bytes(&bytes)?;
    if proof.to_bytes()? != bytes {
        return Err("noncanonical proof".into());
    }
    // This experiment's independently expected Init public values.
    let public = INIT_PUBLIC_WORDS.map(Val::from_u64);
    // Historical lowering: namespace 107, ten arithmetic traces, six hash anchors.
    let mut claims: Vec<_> = std::iter::once(Val::ZERO)
        .chain(public)
        .enumerate()
        .map(|(i, v)| vec![Val::from_u8(107), Val::ONE, Val::from_usize(i), v])
        .collect();
    claims.extend((0..10).map(|i| vec![Val::from_u8(107), Val::from_u8(3), Val::from_usize(i)]));
    claims.extend((0..6).map(|i| {
        vec![
            Val::from_u8(107),
            Val::from_u8(4),
            Val::from_u8(3),
            Val::from_usize(i),
        ]
    }));
    let mut encoded = Vec::new();
    encoded.extend_from_slice(&(claims.len() as u64).to_le_bytes());
    for claim in &claims {
        encoded.extend_from_slice(&(claim.len() as u64).to_le_bytes());
        for value in claim {
            encoded.extend_from_slice(&value.as_canonical_u64().to_le_bytes());
        }
    }
    assert_eq!(
        encoded,
        fs::read(dir.join("outer-claims.bin"))?,
        "claim mapping differs from historical lowering"
    );
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    key.system()
        .verify_multiple_claims(&refs, &proof)
        .map_err(|e| format!("saved recursive proof: {e:?}"))?;
    for i in 0..18 {
        let mut wrong = claims.clone();
        wrong[i + 1][3] += Val::ONE;
        assert!(
            key.system()
                .verify_multiple_claims(
                    &wrong.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &proof
                )
                .is_err()
        );
    }
    println!(
        "Saved {}-byte recursive FRI proof verified; all 18 altered Init claim words rejected",
        bytes.len()
    );
    assert_eq!(key.system().circuits.len(), 22);
    let active = vec![true; 22];
    let logs: Vec<_> = key
        .system()
        .circuits
        .iter()
        .map(|c| u8::try_from(c.preprocessed_height.ilog2()).unwrap())
        .collect();
    assert_eq!(proof.active, active);
    assert_eq!(proof.log_degrees, logs);
    let profile = ProofProfile {
        envelope: Envelope::Ordinary,
        active,
        log_degrees: logs,
        claim_lengths: claims.iter().map(Vec::len).collect(),
        message_lengths: vec![],
        max_field_retries: 2,
    };
    let mut schema = Statement {
        claims: claims
            .iter()
            .map(|c| {
                c.iter()
                    .copied()
                    .map(StatementSlot::Constant)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        messages: vec![],
    };
    for c in &mut schema.claims[1..19] {
        c[3] = StatementSlot::Public;
    }
    Ok(Fixture {
        key,
        proof,
        public,
        claims,
        profile,
        schema,
    })
}
