//! Polynomial commitment scheme (PCS) using FRI over the Goldilocks field.
//!
//! Shows the three core PCS operations in sequence:
//!   - **Commit**: LDE + Merkle tree over polynomial evaluations.
//!   - **Open**: FRI opening proof for evaluation at a challenge point.
//!   - **Verify**: check the opening against the committed polynomials.
//!
//! All types (`Val`, `ExtVal`, `Pcs`, `Challenger`, ...) come from
//! [`multi_stark::types`], which bundles the concrete FRI/Goldilocks/Keccak
//! instantiation used by the STARK prover.
//!
//! Run with:
//! ```sh
//! cargo run --example pcs_example --release
//! ```

use bincode::{config::standard, serde::encode_to_vec};
use multi_stark::config::StarkGenericConfig;
use multi_stark::types::{
    Challenger, Commitment, CommitmentParameters, Domain, ExtVal, FriParameters,
    GoldilocksKeccakConfig, Pcs, PcsProof, ProverData, Val,
};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{OpenedValues, Pcs as PcsTrait};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

fn main() {
    // ── Parameters ──────────────────────────────────────────────────────────
    // log_blowup = 1 → blowup factor 2 (minimal Reed-Solomon rate).
    // num_queries = 100 → ~100 bits of conjectured FRI soundness at rate 1/2
    // (~50 bits under the proven Johnson bound).
    let commitment_params = CommitmentParameters {
        log_blowup: 1,
        cap_height: 0,
    };
    let fri_params = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 100,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };

    let config = GoldilocksKeccakConfig::new(commitment_params, fri_params);
    let pcs = config.pcs();

    // ── Build polynomial evaluations ─────────────────────────────────────────
    // Commit `num_polys` polynomials over a domain of size 32 (degree ≤ 31).
    // The evaluation matrix has one column per polynomial.
    let log_degree: u32 = 5; // domain size = 2^5 = 32
    let degree = 1usize << log_degree;
    let num_polys: usize = 1;

    let domain: Domain =
        <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);

    // Simple non-trivial evaluations: entry (i, j) = i * num_polys + j.
    let n = u32::try_from(degree * num_polys).unwrap();
    let matrix = RowMajorMatrix::new((0..n).map(Val::from_u32).collect::<Vec<_>>(), num_polys);

    // ── Commit ───────────────────────────────────────────────────────────────
    // The PCS computes a coset LDE (blowup × domain size evaluations), then
    // builds a Merkle tree over the extended rows. The root is the commitment.
    let (commitment, prover_data): (Commitment, ProverData) =
        <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, [(domain, matrix)]);
    println!(
        "Committed {} polynomials over domain of size {} (degree ≤ {})",
        num_polys,
        degree,
        degree - 1
    );

    // ── Open ─────────────────────────────────────────────────────────────────
    // Fiat-Shamir: observe the commitment, then sample a challenge point ζ
    // in the degree-2 extension field ExtVal = Goldilocks².
    let mut prover_challenger: Challenger = config.initialise_challenger();
    prover_challenger.observe(commitment.clone());
    let zeta: ExtVal = prover_challenger.sample_algebra_element();
    let num_open = 2;

    // Open all polynomials at ζ and produce a FRI opening proof.
    // Input: for each committed batch, a list of opening-point vectors (one per matrix).
    // Output: claimed evaluations + proof.
    let (opened_values, pcs_proof): (OpenedValues<_>, PcsProof) =
        <Pcs as PcsTrait<ExtVal, Challenger>>::open(
            pcs,
            vec![(&prover_data, vec![vec![zeta; num_open]])],
            &mut prover_challenger,
        );
    // opened_values[round][matrix_idx][point_idx] = Vec<ExtVal> (one per column)
    let evals_at_zeta = opened_values[0][0][0].clone();
    println!("Opened at ζ: {} polynomial values", evals_at_zeta.len());

    // ── Proof size ───────────────────────────────────────────────────────────
    let opened_values_bytes =
        encode_to_vec(&opened_values, standard()).expect("serialization failed");
    println!("opened_values size: {} bytes", opened_values_bytes.len());
    let proof_bytes = encode_to_vec(&pcs_proof, standard()).expect("serialization failed");
    println!("PCS proof size: {} bytes", proof_bytes.len());

    // ── Verify ───────────────────────────────────────────────────────────────
    // The verifier mirrors the prover's Fiat-Shamir transcript (observe commitment,
    // resample ζ), then checks the opening proof against the claimed evaluations.
    let mut verifier_challenger: Challenger = config.initialise_challenger();
    verifier_challenger.observe(commitment.clone());
    let verifier_zeta: ExtVal = verifier_challenger.sample_algebra_element();
    assert_eq!(verifier_zeta, zeta, "Fiat-Shamir transcript mismatch");

    <Pcs as PcsTrait<ExtVal, Challenger>>::verify(
        pcs,
        vec![(
            commitment,
            vec![(domain, vec![(zeta, evals_at_zeta); num_open])],
        )],
        &pcs_proof,
        &mut verifier_challenger,
    )
    .expect("PCS verification failed");

    println!("PCS proof verified successfully!");
}
