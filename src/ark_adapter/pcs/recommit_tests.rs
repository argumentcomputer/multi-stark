use super::*;
use crate::ark_adapter::PublicSetup;
use crate::traits::{Algebra, Field};

fn public_srs(len: usize, seed: &[u8]) -> Arc<Srs> {
    let powers = Srs::unsafe_dev_setup(len, seed);
    Arc::new(
        Srs::from_public_powers(
            powers.g1,
            powers.g2,
            powers.tau_g2,
            PublicSetup {
                max_degree: 62,
                id: *blake3::hash(seed).as_bytes(),
            },
        )
        .unwrap(),
    )
}

fn fixture() -> Vec<(Radix2Coset, RowMajorMatrix<Scalar>)> {
    [8usize, 2, 16, 4]
        .into_iter()
        .enumerate()
        .map(|(index, height)| {
            let values = (0..height)
                .flat_map(|row| {
                    [
                        Scalar::ZERO,
                        Scalar::from_usize(index + 7),
                        Scalar::from_usize(row * row + 3 * index + 1),
                        -Scalar::from_usize(row * (index + 2) + 11),
                    ]
                })
                .collect();
            (
                Radix2Coset {
                    log_size: height.ilog2() as usize,
                    shift: Scalar::ONE,
                },
                RowMajorMatrix::new(values, 4),
            )
        })
        .collect()
}

fn checkpoint(data: &KzgProverData) -> Vec<u8> {
    let mut bytes = Vec::new();
    data.write_checkpoint(&mut bytes).unwrap();
    bytes
}

fn open_verified(
    pcs: &KzgPcs,
    commitment: &KzgCommitment,
    data: &KzgProverData,
) -> (OpenedValues<Scalar>, Vec<u8>, Scalar) {
    let points: Vec<Vec<Vec<Scalar>>> = [
        vec![vec![19, 23, 19], vec![23], vec![0, 19], vec![29, 23]],
        vec![vec![], vec![19, 0], vec![29], vec![0, 23]],
    ]
    .into_iter()
    .map(|round| {
        round
            .into_iter()
            .map(|matrix| matrix.into_iter().map(Scalar::from_u64).collect())
            .collect()
    })
    .collect();
    let mut prover = Blake3Transcript::new();
    let (opened, proof) = pcs.open(
        points.iter().map(|points| (data, points.clone())).collect(),
        &mut prover,
    );
    assert_eq!(proof.0.len(), 4);
    let rounds: VerifyRounds<KzgCommitment, Radix2Coset, Scalar> = points
        .iter()
        .zip(&opened)
        .map(|(points, values)| {
            (
                commitment.clone(),
                data.matrices
                    .iter()
                    .zip(points)
                    .zip(values)
                    .map(|((matrix, points), values)| {
                        (
                            matrix.domain,
                            points.iter().copied().zip(values.clone()).collect(),
                        )
                    })
                    .collect(),
            )
        })
        .collect();
    let mut verifier = Blake3Transcript::new();
    pcs.verify(rounds.clone(), &proof, &mut verifier).unwrap();
    let next_challenge = prover.sample_challenge();
    assert_eq!(next_challenge, verifier.sample_challenge());
    let mut wrong = rounds;
    wrong[0].1[0].1[0].1[2] += Scalar::ONE;
    assert!(
        pcs.verify(wrong, &proof, &mut Blake3Transcript::new())
            .is_err()
    );
    (
        opened,
        bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap(),
        next_challenge,
    )
}

fn check_migration(source: &KzgPcs, target: &KzgPcs) -> (KzgCommitment, KzgCommitment) {
    let (before, data) = source.commit(fixture());
    let mut restored = KzgProverData::read_checkpoint(checkpoint(&data).as_slice()).unwrap();
    let allocations: Vec<_> = restored
        .matrices
        .iter()
        .map(|matrix| {
            let _ = matrix.constant_columns();
            #[cfg(feature = "kzg-cuda")]
            let _ = matrix.resident.set(vec![None; matrix.columns.len()]);
            matrix.columns.iter().map(Vec::as_ptr).collect::<Vec<_>>()
        })
        .collect();
    // Prior point vectors are irrelevant even if their shape is unusable.
    restored.commitment = KzgCommitment(vec![], vec![vec![source.srs.g1[0]; 7]]);
    let (actual_commitment, actual) = target.recommit(restored).unwrap();
    let (expected_commitment, expected) = target.commit(fixture());
    assert_eq!(actual_commitment, expected_commitment);
    assert_eq!(actual.commitment, actual_commitment);
    for ((matrix, expected), pointers) in actual
        .matrices
        .iter()
        .zip(&expected.matrices)
        .zip(allocations)
    {
        assert_eq!(matrix.domain(), expected.domain);
        assert_eq!(matrix.width(), 4);
        assert_eq!(matrix.columns, expected.columns);
        assert_eq!(
            matrix.constants.get(),
            Some(&vec![true, true, false, false])
        );
        assert_eq!(
            matrix.columns.iter().map(Vec::as_ptr).collect::<Vec<_>>(),
            pointers
        );
        #[cfg(feature = "kzg-cuda")]
        assert!(
            matrix
                .resident
                .get()
                .is_some_and(|columns| columns.len() == 4 && columns.iter().all(Option::is_none))
        );
    }
    for (matrix, shifted) in actual.matrices.iter().zip(&actual_commitment.1) {
        assert_eq!(
            shifted.len(),
            if target.requires_shifted_commitment(matrix.domain.size()) {
                matrix.columns.len()
            } else {
                0
            }
        );
    }
    assert_eq!(checkpoint(&actual), checkpoint(&expected));
    assert_eq!(
        open_verified(target, &actual_commitment, &actual),
        open_verified(target, &expected_commitment, &expected)
    );
    (before, actual_commitment)
}

#[test]
fn recommit_legacy_to_public_degree_matches_fresh_commit_and_opening() {
    let seed = b"recommit-known-trapdoor-same-seed";
    let source = KzgPcs::new(Arc::new(Srs::unsafe_dev_setup(16, seed)), 2);
    let target = KzgPcs::new(public_srs(16, seed), 2);
    let (before, after) = check_migration(&source, &target);
    assert_eq!(before.0, after.0);
    assert!(before.1.iter().any(|columns| !columns.is_empty()));
    assert!(after.1.iter().all(Vec::is_empty));
}

#[test]
fn recommit_changed_srs_and_shift_policy_matches_fresh_commit_and_opening() {
    let source = KzgPcs::new(public_srs(16, b"recommit-known-trapdoor-source"), 2);
    let target = KzgPcs::new(
        Arc::new(Srs::unsafe_dev_setup(32, b"recommit-known-trapdoor-target")),
        2,
    );
    let (before, after) = check_migration(&source, &target);
    assert_ne!(before.0, after.0);
    assert!(before.1.iter().all(Vec::is_empty));
    assert!(after.1.iter().all(|columns| columns.len() == 4));
}

fn coefficient_data() -> KzgProverData {
    KzgProverData {
        commitment: KzgCommitment(vec![], vec![]),
        matrices: vec![CommittedMatrix::new(
            Radix2Coset {
                log_size: 3,
                shift: Scalar::ONE,
            },
            vec![vec![Fr::ONE; 8]],
        )],
    }
}

#[test]
fn recommit_rejects_malformed_domains_and_columns() {
    let pcs = KzgPcs::new(public_srs(16, b"recommit-invalid-shapes"), 2);
    for case in 0..8 {
        let mut data = coefficient_data();
        let matrix = &mut data.matrices[0];
        match case {
            0 => matrix.columns.clear(),
            1 => matrix.columns[0].clear(),
            2 => {
                matrix.columns[0].pop();
            }
            3 => matrix.columns[0].push(Fr::ONE),
            4 => matrix.domain.shift = Scalar::ZERO,
            5 => matrix.domain.shift = Scalar::from_u8(2),
            6 => matrix.domain.log_size = <Scalar as crate::traits::TwoAdicField>::TWO_ADICITY + 1,
            7 => matrix.domain.log_size = usize::MAX,
            _ => unreachable!(),
        }
        assert!(matches!(pcs.recommit(data), Err(KzgError::ShapeMismatch)));
    }
    let mut data = coefficient_data();
    let mut later = coefficient_data().matrices.remove(0);
    later.columns.push(vec![]);
    data.matrices.push(later);
    assert!(matches!(pcs.recommit(data), Err(KzgError::ShapeMismatch)));
}

#[test]
fn recommit_rejects_trace_cap_and_loaded_prefix_independently() {
    let cap = KzgPcs::with_max_trace_len(public_srs(16, b"recommit-cap"), 4, 2);
    assert!(matches!(
        cap.recommit(coefficient_data()),
        Err(KzgError::ShapeMismatch)
    ));
    let prefix = KzgPcs::with_max_trace_len(public_srs(4, b"recommit-prefix"), 16, 2);
    assert!(matches!(
        prefix.recommit(coefficient_data()),
        Err(KzgError::ShapeMismatch)
    ));
}
