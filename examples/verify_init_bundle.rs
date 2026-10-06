//! Verify the checkpointed Init bundle without the inner proof or proving keys.
#[cfg(feature = "groth16")]
#[path = "support/init_claim.rs"]
mod init_claim;

#[cfg(not(feature = "groth16"))]
fn main() {
    eprintln!("enable --features groth16");
}

#[cfg(feature = "groth16")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ark_bls12_381::Bls12_381;
    use ark_groth16::{VerifyingKey, prepare_verifying_key};
    use ark_serialize::CanonicalDeserialize;
    use multi_stark::{plonkish::verifier::verify_encoded_query_bundle, types::Val};
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use std::{fs, path::PathBuf, time::Instant};
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "experiments/init-fri-groth16-artifacts".into()),
    );
    let keys = (0..11)
        .map(|i| {
            let bytes = fs::read(dir.join(format!("development-vk-{i}.bin")))?;
            Ok(prepare_verifying_key(
                &VerifyingKey::<Bls12_381>::deserialize_compressed(bytes.as_slice())?,
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let public = init_claim::INIT_PUBLIC_WORDS.map(Val::from_u64);
    let bytes = fs::read(dir.join("development-bundle.bin"))?;
    assert_eq!(bytes.len(), 2324);
    let start = Instant::now();
    assert!(verify_encoded_query_bundle(&keys, &public, &bytes)?);
    println!(
        "Complete {}-byte Init bundle verified in {:?}",
        bytes.len(),
        start.elapsed()
    );
    // Change both the packet's claim and the requested claim: rejection must
    // come from the Groth16 relation, not just the packet/header comparison.
    for i in 0..public.len() {
        let mut wrong = public;
        wrong[i] += Val::ONE;
        let mut altered = bytes.clone();
        altered[36 + i * 8..44 + i * 8].copy_from_slice(&wrong[i].as_canonical_u64().to_le_bytes());
        assert!(!verify_encoded_query_bundle(&keys, &wrong, &altered)?);
    }
    for offset in [0, 4, 180, 212] {
        let mut altered = bytes.clone();
        altered[offset] ^= 1;
        assert!(!verify_encoded_query_bundle(&keys, &public, &altered)?);
    }
    let mut reordered = bytes.clone();
    for i in 0..192 {
        reordered.swap(212 + i, 404 + i);
    }
    assert!(!verify_encoded_query_bundle(&keys, &public, &reordered)?);
    let mut duplicated = bytes.clone();
    duplicated.copy_within(212..404, 404);
    assert!(!verify_encoded_query_bundle(&keys, &public, &duplicated)?);
    assert!(!verify_encoded_query_bundle(
        &keys,
        &public,
        &bytes[..bytes.len() - 1]
    )?);
    let mut trailing = bytes;
    trailing.push(0);
    assert!(!verify_encoded_query_bundle(&keys, &public, &trailing)?);
    println!(
        "Altered claims/context/profile/proof, reordered/duplicated proofs, truncation and trailing bytes rejected. Insecure development setup."
    );
    Ok(())
}
