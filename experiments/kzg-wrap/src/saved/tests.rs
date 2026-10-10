use super::*;
use multi_stark::{
    ark_adapter::Srs,
    expr::Expr,
    lookup::Lookup,
    system::{CircuitInputs, ProverKey, SystemWitness},
};
use p3_matrix::dense::RowMajorMatrix;
use std::sync::Arc;

fn input(system: System<KzgConfig>, key: &ProverKey<KzgConfig>, words: [u64; 18]) -> Input {
    let claims: Vec<_> = std::iter::once(0)
        .chain(words)
        .enumerate()
        .map(|(index, value)| {
            vec![
                Scalar::from_u8(93),
                Scalar::ONE,
                Scalar::from_usize(index),
                Scalar::from_u64(value),
            ]
        })
        .collect();
    let main = RowMajorMatrix::new(
        words
            .into_iter()
            .map(Scalar::from_u64)
            .collect::<Vec<_>>()
            .repeat(4),
        18,
    );
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let proof =
        system.prove_multiple_claims(key, &refs, SystemWitness::from_stage_1(vec![main], &system));
    system.verify_multiple_claims(&refs, &proof).unwrap();
    let proof_bytes = FixedProofCodec::new(&system, &[2])
        .unwrap()
        .encode(&proof)
        .unwrap();
    Input {
        system,
        proof,
        claims,
        logs: vec![2],
        proof_bytes,
        input_blake3: BTreeMap::new(),
        setup_identity: [0; 32],
        load_seconds: 0.0,
    }
}

#[test]
fn saved_plan_excludes_public_words_and_binds_constants_and_layout() {
    let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(8, b"saved-plan-fixture")), 2);
    let definition = CircuitInputs {
        main_width: 18,
        preprocessed: Some(RowMajorMatrix::new_col(vec![
            Scalar::ONE,
            Scalar::ZERO,
            Scalar::ZERO,
            Scalar::ZERO,
        ])),
        constraints: (0..18)
            .map(|i| Expr::main_next(i) - Expr::main(i))
            .collect(),
        lookups: (0..19)
            .map(|index| {
                Lookup::pull(
                    Expr::preprocessed(0),
                    vec![
                        Expr::constant(Scalar::from_u8(93)),
                        Expr::constant(Scalar::ONE),
                        Expr::constant(Scalar::from_usize(index)),
                        if index == 0 {
                            Expr::constant(Scalar::ZERO)
                        } else {
                            Expr::main(u32::try_from(index - 1).unwrap())
                        },
                    ],
                )
            })
            .collect(),
        ..Default::default()
    };
    let (first_system, first_key) = System::new(config.clone(), [definition.clone()]);
    let (second_system, second_key) = System::new(config, [definition]);
    let first = input(first_system, &first_key, [0; 18]);
    let mut second = input(
        second_system,
        &second_key,
        std::array::from_fn(|i| i as u64),
    );
    assert_ne!(first.proof_bytes, second.proof_bytes);
    let identity = first.plan(1 << 27).unwrap().identity();
    assert_eq!(identity, second.plan(1 << 27).unwrap().identity());
    assert_ne!(identity, first.plan(1 << 29).unwrap().identity());
    let counted = first.plan(1 << 27).unwrap().count().unwrap();
    let report = super::count::report(
        &first,
        &crate::outer::setup::SetupSource::Development { cache: None },
        counted,
        identity,
    )
    .unwrap();
    assert_eq!(report["layout_admissible"], true);
    assert_eq!(report["projected_packet_below_limit"], true);
    assert_eq!(
        report["authenticated_input"]["development_known_trapdoor"],
        true
    );
    assert_eq!(report["filecoin_census_passed"], false);
    assert_eq!(report["actual_outer_proof_generated"], false);
    assert_eq!(
        report["frontend_identity"],
        blake3::Hash::from_bytes(identity).to_hex().to_string()
    );
    second.claims[0][3] = Scalar::ONE;
    assert_ne!(identity, second.plan(1 << 27).unwrap().identity());
    second.claims[0][3] = Scalar::ZERO;
    second.claims[18][3] = Scalar::from_limbs_le([0, 1, 0, 0]);
    assert!(second.plan(1 << 27).is_err());
}

#[test]
fn diagnostic_saved_loader_requires_binding_before_setup_metadata() -> Result<()> {
    use crate::outer::setup::{BINDING_FILE, SetupSource};
    let root =
        std::env::temp_dir().join(format!("saved-diagnostic-binding-{}", std::process::id()));
    fs::create_dir(&root)?;
    let cache = root.join("unprovisioned-cache");
    let source = SetupSource::KnownTrapdoorPublicDegree {
        cache: Some(cache.clone()),
    };
    let expected = init_claim::INIT_PUBLIC_WORDS;
    let ns = Scalar::from_u8(93);
    let mut claims: Vec<_> = std::iter::once(0)
        .chain(expected)
        .enumerate()
        .map(|(index, word)| {
            vec![
                ns,
                Scalar::ONE,
                Scalar::from_usize(index),
                Scalar::from_u64(word),
            ]
        })
        .collect();
    claims.extend((0..9).map(|i| vec![ns, Scalar::from_u8(3), Scalar::from_usize(i)]));
    claims.extend((0..6).map(|i| {
        vec![
            ns,
            Scalar::from_u8(4),
            Scalar::from_u8(3),
            Scalar::from_usize(i),
        ]
    }));
    let manifest = (vec![1usize; 19], vec![1usize << 24; 19], claims);
    fs::write(
        root.join("manifest.bin"),
        bincode::serde::encode_to_vec(&manifest, bincode::config::standard())?,
    )?;
    fs::write(root.join("setup-0.bin"), [0xff])?;
    let missing = Input::load(&root, &source, &expected).err().unwrap();
    assert_eq!(
        missing.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::NotFound
    );
    fs::write(root.join(BINDING_FILE), [0; 32])?;
    let wrong = Input::load(&root, &source, &expected).err().unwrap();
    assert!(wrong.to_string().contains("different KZG setup"));
    fs::remove_file(root.join(BINDING_FILE))?;
    let identity = source.identity(b"init-fri-kzg-ordinary-v1", 1 << 24, 2)?;
    source.bind_stage(&root, &identity)?;
    let malformed = Input::load(&root, &source, &expected).err().unwrap();
    assert!(
        malformed
            .downcast_ref::<bincode::error::DecodeError>()
            .is_some()
    );
    assert!(!cache.exists());
    fs::remove_dir_all(root)?;
    Ok(())
}
