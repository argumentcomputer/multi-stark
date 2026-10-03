//! Verify an exported outer proof without repeating setup or building a circuit.
//! The artifact directory's VK and expected claims must be independently trusted.
use multi_stark::{
    plonkish::verifier::VerifierKey,
    prover::Proof,
    types::{GoldilocksBlake3Config, Val},
};
use std::{fs, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: verify_outer <trusted-artifact-directory>")?,
    );
    let key = VerifierKey::from_bytes(&fs::read(path.join("outer-vk.bin"))?)?;
    let data = fs::read(path.join("outer-claims.bin"))?;
    let (claims, used): (Vec<Vec<Val>>, usize) = bincode::serde::decode_from_slice(
        &data,
        bincode::config::standard().with_limit::<16777216>(),
    )?;
    if used != data.len() {
        return Err("trailing claim bytes".into());
    }
    let proof =
        Proof::<GoldilocksBlake3Config>::from_bytes(&fs::read(path.join("outer-proof.bin"))?)?;
    key.system()
        .verify_multiple_claims(
            &claims.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &proof,
        )
        .map_err(|e| format!("verification failed: {e:?}"))?;
    println!(
        "Outer proof verified with exported key {:02x?}",
        key.fingerprint()?
    );
    Ok(())
}
