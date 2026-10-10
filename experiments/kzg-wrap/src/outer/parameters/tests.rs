use super::*;
use crate::{
    init_claim,
    outer::{Frontend, Scalar, verify_with_parameters},
};
use ark_serialize::CanonicalSerialize;
use multi_stark::{
    plonkish::{Assignment, CircuitBuilder, Value},
    system::SystemWitness,
    traits::Field,
};

const SEED: &[u8] = b"outer-public-parameters-test";
const SHARED_SEED: &[u8] = b"init-fri-kzg-ordinary-v1";

#[test]
fn public_degree_parameters_bind_profile_and_verify_with_two_anchors() -> storage::Result<()> {
    let root = std::env::temp_dir().join(format!("outer-parameters-{}", std::process::id()));
    fs::create_dir(&root)?;
    let absent_cache = root.join("unprovisioned-cache");
    let parameters = Parameters::KnownTrapdoorPublicDegree {
        seed: SEED,
        cache: Some(&absent_cache),
    };
    let verifier = parameters.config(32, 2, true)?;
    assert_eq!(verifier.srs().max_len(), 2);
    assert!(!absent_cache.exists());
    let prover = Parameters::KnownTrapdoorPublicDegree {
        seed: SEED,
        cache: None,
    }
    .config(32, 2, false)?;
    assert_eq!(prover.srs().max_len(), 32);
    assert_eq!(prover.srs().public_setup(), verifier.srs().public_setup());
    assert_eq!(prover.transcript_seed(), verifier.transcript_seed());
    assert_eq!(
        verifier.srs().public_setup().unwrap().max_degree,
        PUBLIC_MAX_DEGREE
    );
    assert!(verifier.srs().degree_keys.is_empty());
    assert!(!verifier.requires_shifted_commitment(8));
    assert!(parameters.config(1, 2, true).is_err());
    assert!(parameters.config(3, 2, true).is_err());
    assert!(parameters.config(MAX_TRACE_LEN * 2, 2, true).is_err());
    assert!(parameters.config(32, 4, true).is_err());
    let identity = parameters.identity(32, 2)?;
    assert_ne!(identity, parameters.identity(64, 2)?);
    let other = Parameters::KnownTrapdoorPublicDegree {
        seed: b"different-seed",
        cache: None,
    };
    assert_ne!(identity, other.identity(32, 2)?);
    assert!(parameters.check_binding(&root, &identity).is_err());
    parameters.bind_stage(&root, &identity)?;
    parameters.check_binding(&root, &identity)?;
    assert!(other.check_binding(&root, &other.identity(32, 2)?).is_err());
    assert!(parameters.bind_stage(&root, &[0; 32]).is_err());
    fs::remove_file(root.join(crate::outer::setup::BINDING_FILE))?;
    fs::create_dir(root.join("kzg"))?;
    fs::write(root.join("kzg/setup-0.bin"), b"unidentified")?;
    assert!(parameters.bind_stage(&root, &identity).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn selected_parameters_preserve_legacy_and_filecoin_selection() -> storage::Result<()> {
    let legacy = SetupSource::Development { cache: None };
    let selected = Parameters::Selected(&legacy);
    let config = selected.config(8, 2, false)?;
    let expected = legacy.config(OUTER_SEED, 8, 2, false)?;
    assert_eq!(config.transcript_seed(), expected.transcript_seed());
    assert_eq!(config.srs().g1, expected.srs().g1);
    assert_eq!(config.srs().degree_keys, expected.srs().degree_keys);
    assert_eq!(selected.identity(8, 2)?, legacy.identity(OUTER_SEED, 8, 2)?);
    assert!(config.srs().public_setup().is_none());
    assert!(config.requires_shifted_commitment(4));
    assert!(!selected.uses_public_degree());
    assert!(selected.is_development());
    assert!(!selected.is_diagnostic());
    let filecoin = SetupSource::Filecoin {
        cache: "unopened-filecoin-cache".into(),
        digest: [17; 32],
    };
    let selected = Parameters::Selected(&filecoin);
    assert_eq!(
        selected.identity(8, 2)?,
        filecoin.identity(OUTER_SEED, 8, 2)?
    );
    assert!(selected.uses_public_degree());
    assert!(!selected.is_development());
    assert!(!selected.is_diagnostic());
    assert_eq!(selected.filecoin_manifest_digest(), Some([17; 32]));
    Ok(())
}

#[test]
fn selected_public_degree_source_matches_diagnostic_adapter() -> storage::Result<()> {
    let root = std::env::temp_dir().join(format!(
        "outer-selected-public-parameters-{}",
        std::process::id()
    ));
    fs::create_dir(&root)?;
    let cache = root.join("unprovisioned-cache");
    let source = SetupSource::KnownTrapdoorPublicDegree {
        cache: Some(cache.clone()),
    };
    let selected = Parameters::Selected(&source);
    let adapter = Parameters::KnownTrapdoorPublicDegree {
        seed: SHARED_SEED,
        cache: Some(&cache),
    };
    assert!(selected.uses_public_degree());
    assert!(selected.is_development());
    assert!(selected.is_diagnostic());
    assert!(selected.filecoin_manifest_digest().is_none());
    for height in [2, 32, MAX_TRACE_LEN] {
        let actual = selected.config(height, 2, true)?;
        let expected = adapter.config(height, 2, true)?;
        assert_eq!(actual.srs().max_len(), 2);
        assert_eq!(actual.srs().g1, expected.srs().g1);
        assert_eq!(actual.srs().g2, expected.srs().g2);
        assert_eq!(actual.srs().tau_g2, expected.srs().tau_g2);
        assert_eq!(actual.srs().public_setup(), expected.srs().public_setup());
        assert_eq!(actual.transcript_seed(), expected.transcript_seed());
        assert_eq!(selected.identity(height, 2)?, adapter.identity(height, 2)?);
        assert_eq!(
            source.identity(b"unused-caller-seed", height, 2)?,
            adapter.identity(height, 2)?
        );
    }
    assert!(!cache.exists());
    assert!(selected.config(MAX_TRACE_LEN * 2, 2, true).is_err());
    assert!(selected.config(32, 4, true).is_err());
    let identity = adapter.identity(32, 2)?;
    adapter.bind_stage(&root, &identity)?;
    selected.check_binding(&root, &identity)?;
    let legacy = SetupSource::Development { cache: None };
    assert_ne!(
        selected.identity(32, 2)?,
        legacy.identity(OUTER_SEED, 32, 2)?
    );
    assert!(selected.check_binding(&root, &[0; 32]).is_err());
    fs::remove_file(root.join(crate::outer::setup::BINDING_FILE))?;
    assert!(selected.check_binding(&root, &identity).is_err());
    assert!(legacy.check_binding(&root, &[0; 32]).is_ok());
    fs::remove_dir_all(root)?;
    Ok(())
}

fn frontend(inner: &Srs) -> storage::Result<(Frontend, Value, [Value; 18])> {
    let mut builder = CircuitBuilder::new();
    let private = builder.input("private");
    let table = builder.fixed_table("small", (0..16).map(|v| vec![Scalar::from_u8(v)]).collect());
    builder.lookup(table, &[private]);
    for _ in 0..8 {
        builder.mul(private, private);
    }
    let public = std::array::from_fn(|i| builder.public_input(format!("claim{i}")));
    let mut points = vec![];
    inner.g1[1].serialize_compressed(&mut points)?;
    (-inner.g1[0]).serialize_compressed(&mut points)?;
    for bytes in points.as_chunks::<24>().0 {
        let mut limbs = [0; 4];
        for (limb, bytes) in limbs.iter_mut().zip(bytes.as_chunks::<8>().0) {
            *limb = u64::from_le_bytes(*bytes);
        }
        let point = builder.constant(Scalar::from_limbs_le(limbs));
        builder.expose_public(point);
    }
    Ok((
        Frontend::new(
            builder.finish(),
            &[inner.g2, inner.tau_g2],
            0,
            b"public-degree-outer-parameters-test/v1",
            1 << 12,
        )?,
        private,
        public,
    ))
}

fn assignment(
    frontend: &Frontend,
    private: Value,
    public: &[Value; 18],
    expected: &[u64; 18],
    secret: u8,
) -> storage::Result<Assignment<Scalar>> {
    let mut witness = frontend.witness();
    witness.set(private, Scalar::from_u8(secret))?;
    for (&wire, &word) in public.iter().zip(expected) {
        witness.set(wire, Scalar::from_u64(word))?;
    }
    Ok(witness.generate()?)
}

fn artifacts(dir: &Path) -> storage::Result<[Vec<u8>; 3]> {
    Ok([
        fs::read(dir.join("kzg/proof.compact.bin"))?,
        fs::read(dir.join("kzg/packet.bin"))?,
        fs::read(dir.join("kzg/profile-id.bin"))?,
    ])
}

#[test]
fn diagnostic_public_degree_reuses_keys_with_fresh_a_b_a_proofs() -> storage::Result<()> {
    let root =
        std::env::temp_dir().join(format!("outer-public-degree-worker-{}", std::process::id()));
    fs::create_dir(&root)?;
    let inner = Srs::unsafe_dev_public_setup_with_cache(2, PUBLIC_MAX_DEGREE, SHARED_SEED, None)?;
    let parameters = Parameters::KnownTrapdoorPublicDegree {
        seed: SHARED_SEED,
        cache: None,
    };
    let source = SetupSource::KnownTrapdoorPublicDegree { cache: None };
    let selected = Parameters::Selected(&source);
    let wrong_parameters = Parameters::KnownTrapdoorPublicDegree {
        seed: b"wrong-retained-seed",
        cache: None,
    };
    let first = init_claim::INIT_PUBLIC_WORDS;
    let mut second = first;
    second[5] ^= 1;
    let (mut retained, private, public) = frontend(&inner)?;
    let mut retained_config = None;
    let mut results = vec![];
    for (index, (expected, secret)) in [(first, 7), (second, 9), (first, 7)]
        .into_iter()
        .enumerate()
    {
        let fresh_dir = root.join(format!("fresh-{index}"));
        let (mut fresh, fresh_private, fresh_public) = frontend(&inner)?;
        let fresh_assignment = assignment(&fresh, fresh_private, &fresh_public, &expected, secret)?;
        fresh.stage_with_parameters(&fresh_assignment, &fresh_dir, true, &parameters, &expected)?;

        let dir = root.join(format!("retained-{index}"));
        let witness = assignment(&retained, private, &public, &expected, secret)?;
        retained.stage(&witness, &dir, true, &source, &expected)?;
        let loaded = retained.loaded.as_ref().unwrap();
        if let Some(config) = &retained_config {
            assert!(std::ptr::eq(
                KzgConfig::srs(config),
                loaded.system.config.srs()
            ));
        } else {
            retained_config = Some(loaded.system.config.clone());
        }
        assert_eq!(
            loaded.system.config.srs().public_setup(),
            inner.public_setup()
        );
        assert_eq!(loaded.system.circuits.len(), 2);
        assert_eq!(retained.compiled.main_heights(), [loaded.heights[0]]);
        let actual = artifacts(&dir)?;
        assert_eq!(actual, artifacts(&fresh_dir)?);
        drop(fresh_assignment);
        drop(fresh);
        assert!(actual[1].len() < 3_000);
        if index == 0 {
            let claims = retained.compiled.claims(witness.public_values())?;
            let refs = claims.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let traces = retained.compiled.trace_shards(&witness)?;
            let matrices = (0..2)
                .map(|i| traces.trace(i))
                .collect::<Result<Vec<_>, _>>()?;
            let definitions = (0..2)
                .map(|i| -> storage::Result<_> {
                    Ok(retained
                        .compiled
                        .kzg_circuit_input_with_degree_policy(i, loaded.heights[0], 2, false)?
                        .ok_or("missing fixed trace definition")?)
                })
                .collect::<storage::Result<Vec<_>>>()?;
            let (direct, direct_key) =
                multi_stark::system::System::new(loaded.system.config.clone(), definitions);
            let proof = direct.prove_multiple_claims(
                &direct_key,
                &refs,
                SystemWitness::from_stage_1(matrices, &direct),
            );
            direct
                .verify_multiple_claims(&refs, &proof)
                .map_err(|error| format!("direct v4 proof: {error:?}"))?;
            assert_eq!(loaded.codec.encode(&proof)?, actual[0]);
        }
        assert!(!dir.join("0.witness.zst").exists());
        assert!(!dir.join("kzg/main-0.bin").exists());
        if index > 0 {
            assert!(!dir.join("0.fixed.zst").exists());
            assert!(!dir.join("kzg/fixed-0.bin").exists());
        }
        verify_with_parameters(&dir, &parameters, &expected)?;
        verify_with_parameters(&dir, &selected, &expected)?;
        for name in ["prove-report.json", "verify-report.json"] {
            let report: serde_json::Value =
                serde_json::from_slice(&fs::read(dir.join("kzg").join(name))?)?;
            assert_eq!(report["setup"], "development");
            assert_eq!(report["known_trapdoor"], true);
            assert_eq!(report["filecoin_acceptance"], false);
            assert_eq!(report["public_max_degree"], PUBLIC_MAX_DEGREE);
            assert_eq!(
                report["public_setup_id"],
                blake3::Hash::from_bytes(inner.public_setup().unwrap().id)
                    .to_hex()
                    .to_string()
            );
            assert!(report["ceremony_id"].is_null());
            assert!(report["filecoin_manifest_digest"].is_null());
            assert_eq!(
                report["loaded_key_reused"],
                name == "prove-report.json" && index > 0
            );
            if name == "prove-report.json" && index > 0 {
                assert_eq!(report["srs_config_load_seconds"], 0.0);
            }
        }
        let wrong = if expected == first { second } else { first };
        assert!(verify_with_parameters(&dir, &selected, &wrong).is_err());
        assert!(verify_with_parameters(&dir, &wrong_parameters, &expected).is_err());
        assert_eq!(actual, artifacts(&dir)?);
        let binding = fs::read(dir.join(crate::outer::setup::BINDING_FILE))?;
        fs::write(dir.join(crate::outer::setup::BINDING_FILE), [0; 32])?;
        assert!(verify_with_parameters(&dir, &selected, &expected).is_err());
        fs::remove_file(dir.join(crate::outer::setup::BINDING_FILE))?;
        assert!(verify_with_parameters(&dir, &selected, &expected).is_err());
        fs::write(dir.join(crate::outer::setup::BINDING_FILE), binding)?;
        let wrong_dir = root.join(format!("wrong-claim-{index}"));
        assert!(
            retained
                .stage(&witness, &wrong_dir, true, &source, &wrong)
                .is_err()
        );
        assert!(!wrong_dir.exists());
        let wrong_dir = root.join(format!("wrong-seed-{index}"));
        assert!(
            retained
                .stage_with_parameters(&witness, &wrong_dir, true, &wrong_parameters, &expected)
                .is_err()
        );
        assert_eq!(actual, artifacts(&dir)?);
        retained.release_device_memory();
        assert!(retained.has_loaded_key());
        results.push(actual);
    }
    assert_eq!(results[0], results[2]);
    assert_ne!(results[0][0], results[1][0]);
    assert_eq!(results[0][2], results[1][2]);
    let wrong_inner = Srs::unsafe_dev_public_setup_with_cache(
        2,
        PUBLIC_MAX_DEGREE,
        b"unmatched-pairing-seed",
        None,
    )?;
    let (mut mismatched, private, public) = frontend(&wrong_inner)?;
    let witness = assignment(&mismatched, private, &public, &first, 7)?;
    let error = mismatched
        .stage_with_parameters(
            &witness,
            &root.join("wrong-pairing"),
            true,
            &selected,
            &first,
        )
        .unwrap_err();
    assert!(error.to_string().contains("inner pairing keys differ"));
    fs::remove_dir_all(root)?;
    Ok(())
}
