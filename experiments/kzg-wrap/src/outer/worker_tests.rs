use super::*;
use ark_serialize::CanonicalSerialize;
use multi_stark::{
    ark_adapter::Srs,
    plonkish::{CircuitBuilder, Value},
};

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
        let mut words = [0; 4];
        for (word, bytes) in words.iter_mut().zip(bytes.as_chunks::<8>().0) {
            *word = u64::from_le_bytes(*bytes);
        }
        let point = builder.constant(Scalar::from_limbs_le(words));
        builder.expose_public(point);
    }
    Ok((
        Frontend::new(
            builder.finish(),
            &[inner.g2, inner.tau_g2],
            0,
            b"retained-outer-fixture/v1",
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
    for (&value, &word) in public.iter().zip(expected) {
        witness.set(value, Scalar::from_u64(word))?;
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
fn retained_recursive_keys_accept_fresh_statements_and_preserve_packets() -> storage::Result<()> {
    let root = std::env::temp_dir().join(format!("kzg-recursive-worker-{}", std::process::id()));
    fs::create_dir(&root)?;
    let inner = Srs::unsafe_dev_setup(8, b"retained-recursive-inner");
    let setup = setup::SetupSource::Development { cache: None };
    let first = init_claim::INIT_PUBLIC_WORDS;
    let mut second = first;
    second[5] ^= 1;
    let (mut retained, private, public) = frontend(&inner)?;
    let mut retained_config = None;
    let mut results = Vec::new();
    for (index, (expected, secret)) in [(first, 7), (second, 9), (first, 7)]
        .into_iter()
        .enumerate()
    {
        let fresh_dir = root.join(format!("fresh-{index}"));
        let (mut fresh, fresh_private, fresh_public) = frontend(&inner)?;
        let fresh_assignment = assignment(&fresh, fresh_private, &fresh_public, &expected, secret)?;
        fresh.stage(&fresh_assignment, &fresh_dir, true, &setup, &expected)?;

        let dir = root.join(format!("retained-{index}"));
        let witness = assignment(&retained, private, &public, &expected, secret)?;
        retained.stage(&witness, &dir, true, &setup, &expected)?;
        let loaded = retained.loaded.as_ref().unwrap();
        if let Some(config) = &retained_config {
            assert!(std::ptr::eq(
                KzgConfig::srs(config),
                loaded.system.config.srs()
            ));
        } else {
            retained_config = Some(loaded.system.config.clone());
        }
        let actual = artifacts(&dir)?;
        assert_eq!(actual, artifacts(&fresh_dir)?);
        assert!(!dir.join("0.witness.zst").exists());
        assert!(!dir.join("0.fixed.zst").exists());
        assert!(!dir.join("kzg/main-0.bin").exists());
        if index > 0 {
            assert!(!dir.join("kzg/fixed-0.bin").exists());
        }
        prove_with_state(&dir, true, None, &mut None, &setup, &expected)?;
        assert_eq!(actual, artifacts(&dir)?);
        let wrong = if expected == first { second } else { first };
        assert!(prove_with_state(&dir, true, None, &mut None, &setup, &wrong).is_err());
        assert_eq!(actual, artifacts(&dir)?);

        let identity = fs::read(dir.join("frontend-id.bin"))?;
        fs::write(dir.join("frontend-id.bin"), [0; 32])?;
        let traces = retained.compiled.trace_shards(&witness)?;
        let error = prove_with_state(
            &dir,
            false,
            Some(&traces),
            &mut retained.loaded,
            &setup,
            &expected,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("differs from the retained recursive prover profile")
        );
        assert_eq!(actual, artifacts(&dir)?);
        fs::write(dir.join("frontend-id.bin"), &identity)?;
        assert!(
            prove_with_state(&dir, false, None, &mut retained.loaded, &setup, &expected).is_err()
        );
        retained.release_device_memory();
        assert!(retained.has_loaded_key());
        results.push(actual);
    }
    assert_eq!(results[0], results[2]);
    assert_ne!(results[0][0], results[1][0]);
    assert_eq!(results[0][2], results[1][2]);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn fused_fixed_preprocessing_matches_checkpointed_and_cached_paths() -> storage::Result<()> {
    let root = std::env::temp_dir().join(format!(
        "kzg-recursive-fixed-handoff-{}",
        std::process::id()
    ));
    fs::create_dir(&root)?;
    let expected = init_claim::INIT_PUBLIC_WORDS;
    for setup in [
        setup::SetupSource::Development { cache: None },
        setup::SetupSource::KnownTrapdoorPublicDegree { cache: None },
    ] {
        let mode = root.join(if setup.uses_public_degree() {
            "v4"
        } else {
            "v3"
        });
        fs::create_dir(&mode)?;
        let inner = if setup.uses_public_degree() {
            Srs::unsafe_dev_public_setup_with_cache(
                2,
                parameters::PUBLIC_MAX_DEGREE,
                setup::DEVELOPMENT_PUBLIC_SEED,
                None,
            )?
        } else {
            Srs::unsafe_dev_setup(8, b"fixed-handoff-inner")
        };

        let staged = mode.join("staged");
        let (mut disk, private, public) = frontend(&inner)?;
        let witness = assignment(&disk, private, &public, &expected, 7)?;
        disk.stage(&witness, &staged, false, &setup, &expected)?;
        let metadata_path = staged.join("0.meta");
        let metadata = fs::read(&metadata_path)?;
        let mut malformed: multi_stark::system::CircuitInputs<Scalar> =
            storage::load(&metadata_path)?;
        malformed.main_width += 1;
        storage::save(&metadata_path, &malformed)?;
        let error =
            prove_with_state(&staged, false, None, &mut None, &setup, &expected).unwrap_err();
        assert!(error.to_string().contains("fixed trace dimensions differ"));
        assert!(!staged.join("kzg/setup-0.bin").exists());
        assert!(!staged.join("kzg/fixed-0.bin").exists());
        fs::write(&metadata_path, &metadata)?;
        prove_with_state(&staged, false, None, &mut None, &setup, &expected)?;
        let reference = artifacts(&staged)?;

        let fused = mode.join("fused");
        let (mut direct, private, public) = frontend(&inner)?;
        let witness = assignment(&direct, private, &public, &expected, 7)?;
        direct.stage(&witness, &fused, true, &setup, &expected)?;
        assert_eq!(artifacts(&fused)?, reference);
        let manifest: storage::Manifest = storage::load(&fused.join("manifest.bin"))?;
        assert_eq!(
            fs::read(fused.join("manifest.bin"))?,
            fs::read(staged.join("manifest.bin"))?
        );
        for i in 0..manifest.heights.len() {
            assert!(staged.join(format!("{i}.fixed.zst")).exists());
            assert!(staged.join(format!("{i}.witness.zst")).exists());
            assert!(!fused.join(format!("{i}.fixed.zst")).exists());
            assert!(!fused.join(format!("{i}.witness.zst")).exists());
            assert!(!fused.join(format!("kzg/main-{i}.bin")).exists());
            for name in [
                format!("{i}.meta"),
                format!("kzg/setup-{i}.bin"),
                format!("kzg/fixed-{i}.bin"),
            ] {
                assert_eq!(fs::read(fused.join(&name))?, fs::read(staged.join(&name))?);
            }
            let definition: multi_stark::system::CircuitInputs<Scalar> =
                storage::load(&fused.join(format!("{i}.meta")))?;
            assert!(definition.preprocessed.is_none());
        }
        prove_with_state(&fused, true, None, &mut None, &setup, &expected)?;
        assert_eq!(artifacts(&fused)?, reference);

        let copy_headers = |destination: &Path| -> storage::Result<()> {
            fs::create_dir(destination)?;
            for name in [
                "manifest.bin",
                "frontend-id.bin",
                "pairings.bin",
                setup::BINDING_FILE,
            ] {
                fs::copy(fused.join(name), destination.join(name))?;
            }
            Ok(())
        };
        let setup_id = setup.identity(OUTER_SEED, manifest.heights[0], 2)?;
        let cache_profile = [direct.profile.as_slice(), &setup_id].concat();
        let cache_root = mode.join("cache");
        let cache = fixed_cache::FixedCache::from_profile(&cache_root, &cache_profile)?;
        let cached = mode.join("cached");
        copy_headers(&cached)?;
        assert!(cache.restore(&cached)?.is_none());
        cache.publish(&fused)?;
        let restored = cache.restore(&cached)?.ok_or("missing published cache")?;
        assert_eq!(restored.widths, manifest.widths);
        assert_eq!(restored.heights, manifest.heights);
        let traces = direct.compiled.trace_shards(&witness)?;
        let cached_manifest = fs::read(cached.join("manifest.bin"))?;
        let mut malformed: storage::Manifest = storage::load(&cached.join("manifest.bin"))?;
        malformed.widths[1] += 1;
        storage::save(&cached.join("manifest.bin"), &malformed)?;
        let error = prove_with_state(&cached, false, Some(&traces), &mut None, &setup, &expected)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("compiled trace dimensions differ")
        );
        assert!(!cached.join("kzg/proof.compact.bin").exists());
        fs::write(cached.join("manifest.bin"), cached_manifest)?;
        prove_with_state(&cached, false, Some(&traces), &mut None, &setup, &expected)?;
        assert_eq!(artifacts(&cached)?, reference);
        assert!(!cached.join("0.fixed.zst").exists());

        fs::remove_file(cached.join("kzg/fixed-0.bin"))?;
        let sentinel = b"invalid compressed fixed trace must not be read";
        fs::write(cached.join("0.fixed.zst"), sentinel)?;
        prove_with_state(&cached, false, Some(&traces), &mut None, &setup, &expected)?;
        assert_eq!(artifacts(&cached)?, reference);
        assert_eq!(fs::read(cached.join("0.fixed.zst"))?, sentinel.as_slice());
        assert_eq!(
            fs::read(cached.join("kzg/fixed-0.bin"))?,
            fs::read(fused.join("kzg/fixed-0.bin"))?
        );

        let bad = mode.join("wrong-shape");
        copy_headers(&bad)?;
        let mut malformed: storage::Manifest = storage::load(&bad.join("manifest.bin"))?;
        malformed.widths[0] += 1;
        storage::save(&bad.join("manifest.bin"), &malformed)?;
        let error =
            prove_with_state(&bad, false, Some(&traces), &mut None, &setup, &expected).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("compiled trace dimensions differ")
        );
        assert!(!bad.join("kzg").exists());
        assert!(!bad.join("0.meta").exists());

        let entry = fs::read_dir(&cache_root)?
            .next()
            .ok_or("missing cache directory")??
            .path();
        fs::remove_file(entry.join("kzg/fixed-0.bin"))?;
        let damaged = mode.join("damaged-cache");
        fs::create_dir(&damaged)?;
        assert!(cache.restore(&damaged).is_err());
        assert!(!damaged.join("kzg").exists());
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
