//! Local caches of deterministic, known-trapdoor development parameters.

use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use ark_bls12_381::{G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use p3_maybe_rayon::prelude::*;

use super::{PublicSetup, Srs, dev_tau};

const MAGIC: &[u8] = b"multi-stark/bls12-381/dev-srs/v1\0";
const G1_BYTES: usize = 96;
const G2_BYTES: usize = 192;
const BLOCK_POINTS: usize = 1 << 18;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

impl Srs {
    /// Development parameters with a declared public degree independent of the
    /// loaded G1 prefix. TESTS AND DEVELOPMENT ONLY: the seed reveals the
    /// trapdoor, so these parameters do not provide binding commitments.
    ///
    /// The identity is derived from the seed and public degree, never from a
    /// caller-supplied ceremony identity or the loaded prefix. It is explicitly
    /// distinct from the Filecoin ceremony. The declared degree selects the
    /// public-degree protocol; it does not establish a trusted ceremony.
    ///
    /// Reuses [`Self::unsafe_dev_setup_with_cache`] and its trusted-local-cache
    /// contract, without repeating full subgroup or progression validation.
    /// The cache retains its legacy format and degree keys; the returned SRS
    /// contains only the G1 prefix, G2 anchors and public-degree metadata.
    pub fn unsafe_dev_public_setup_with_cache(
        max_len: usize,
        public_max_degree: usize,
        seed: &[u8],
        cache_dir: Option<&Path>,
    ) -> io::Result<Self> {
        let invalid_input = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        if max_len < 2 || !max_len.is_power_of_two() {
            return Err(invalid_input("SRS length must be a power of two >= 2"));
        }
        if public_max_degree < max_len - 1 {
            return Err(invalid_input("public degree is smaller than the G1 prefix"));
        }
        let public_len = public_max_degree
            .checked_add(1)
            .and_then(|length| u64::try_from(length).ok())
            .ok_or_else(|| invalid_input("public degree range overflows"))?;
        if max_len
            .checked_mul(size_of::<G1Affine>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(invalid_input("SRS allocation length overflows"));
        }
        let cache_overhead = MAGIC.len() + 8 + 32 + (max_len.ilog2() as usize + 3) * G2_BYTES + 32;
        u64::try_from(max_len)
            .ok()
            .and_then(|length| length.checked_mul(G1_BYTES as u64))
            .and_then(|bytes| bytes.checked_add(cache_overhead as u64))
            .ok_or_else(|| invalid_input("SRS cache length overflows"))?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"multi-stark/kzg/known-trapdoor-public-degree/v1");
        hasher.update(&(public_len - 1).to_le_bytes());
        hasher.update(blake3::hash(seed).as_bytes());
        let id = *hasher.finalize().as_bytes();
        if id == super::filecoin::filecoin_setup_id() {
            return Err(invalid_input(
                "development identity matches the Filecoin ceremony",
            ));
        }
        let mut srs = Self::unsafe_dev_setup_with_cache(max_len, seed, cache_dir)?;
        srs.degree_keys.clear();
        srs.public_setup = Some(PublicSetup {
            max_degree: public_max_degree,
            id,
        });
        Ok(srs)
    }

    /// Generate development parameters, or reuse a local cache if a directory
    /// is supplied. `None` always performs fresh generation.
    ///
    /// The directory must be trusted local storage. A checksum detects damaged
    /// files, but does not authenticate their producer. Loading checks canonical
    /// coordinates, curve membership and seed-dependent anchors; it omits the
    /// costly G1 subgroup and power-progression checks. This is not an import
    /// API for public ceremony parameters. Like [`Self::unsafe_dev_setup`], the
    /// resulting SRS has a known trapdoor and is for development only.
    ///
    /// A damaged existing cache returns an error; it does not silently start a
    /// potentially expensive regeneration. Writes become visible atomically.
    pub fn unsafe_dev_setup_with_cache(
        max_len: usize,
        seed: &[u8],
        cache_dir: Option<&Path>,
    ) -> io::Result<Self> {
        if max_len < 2 || !max_len.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "SRS length must be a power of two >= 2",
            ));
        }
        let Some(directory) = cache_dir else {
            return Ok(Self::unsafe_dev_setup(max_len, seed));
        };
        let header = header(max_len, seed);
        let path = directory.join(format!("{}.dev-srs", blake3::hash(&header).to_hex()));
        let start = Instant::now();
        match File::open(&path) {
            Ok(file) => {
                let srs = read_cache(file, &header, max_len, seed).map_err(|error| {
                    io::Error::new(error.kind(), format!("{}: {error}", path.display()))
                })?;
                eprintln!(
                    "Loaded development SRS cache {} in {:?}",
                    path.display(),
                    start.elapsed()
                );
                Ok(srs)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                fs::create_dir_all(directory)?;
                let srs = Self::unsafe_dev_setup(max_len, seed);
                write_cache(&path, &header, &srs)?;
                eprintln!(
                    "Generated and cached development SRS {} in {:?}",
                    path.display(),
                    start.elapsed()
                );
                Ok(srs)
            }
            Err(error) => Err(error),
        }
    }
}

fn header(max_len: usize, seed: &[u8]) -> Vec<u8> {
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&(max_len as u64).to_le_bytes());
    bytes.extend_from_slice(blake3::hash(seed).as_bytes());
    bytes
}

fn invalid(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

fn read_cache(
    mut file: File,
    expected_header: &[u8],
    max_len: usize,
    seed: &[u8],
) -> io::Result<Srs> {
    let keys = max_len.ilog2() as usize + 1;
    let g2_len = (keys + 2) * G2_BYTES;
    let expected_size = (max_len as u64)
        .checked_mul(G1_BYTES as u64)
        .and_then(|n| n.checked_add((expected_header.len() + g2_len + 32) as u64))
        .ok_or_else(|| invalid("SRS cache length overflow"))?;
    if file.metadata()?.len() != expected_size {
        return Err(invalid("SRS cache has the wrong length"));
    }
    let mut actual_header = vec![0; expected_header.len()];
    file.read_exact(&mut actual_header)?;
    if actual_header != expected_header {
        return Err(invalid("SRS cache seed, size or format mismatch"));
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(&actual_header);
    let mut bytes = vec![0; g2_len];
    file.read_exact(&mut bytes)?;
    hasher.update(&bytes);
    let mut g2_points = bytes
        .chunks_exact(G2_BYTES)
        .map(|point| G2Affine::deserialize_uncompressed(point).map_err(invalid))
        .collect::<io::Result<Vec<_>>>()?
        .into_iter();
    let g2 = g2_points.next().expect("two anchors");
    let tau_g2 = g2_points.next().expect("two anchors");
    let degree_keys: Vec<_> = g2_points.collect();
    let tau = dev_tau(seed);
    let generator = G2Projective::generator();
    if g2 != generator.into_affine()
        || tau_g2 != (generator * tau).into_affine()
        || degree_keys.iter().enumerate().any(|(k, point)| {
            *point != (generator * tau.pow([(max_len - (1 << k)) as u64])).into_affine()
        })
    {
        return Err(invalid(
            "SRS cache G2 keys differ from the development seed",
        ));
    }
    let mut g1 = Vec::new();
    g1.try_reserve_exact(max_len).map_err(io::Error::other)?;
    bytes.resize(max_len.min(BLOCK_POINTS) * G1_BYTES, 0);
    while g1.len() < max_len {
        let count = (max_len - g1.len()).min(BLOCK_POINTS);
        let block = &mut bytes[..count * G1_BYTES];
        file.read_exact(block)?;
        hasher.update(block);
        let points = block
            .par_chunks_exact(G1_BYTES)
            .map(|bytes| {
                let point = G1Affine::deserialize_uncompressed_unchecked(bytes).map_err(invalid)?;
                if point.is_zero() || !point.is_on_curve() {
                    return Err(invalid("invalid G1 point in development SRS cache"));
                }
                Ok(point)
            })
            .collect::<io::Result<Vec<_>>>()?;
        g1.extend(points);
    }
    let mut checksum = [0; 32];
    file.read_exact(&mut checksum)?;
    if hasher.finalize().as_bytes() != &checksum {
        return Err(invalid("SRS cache checksum mismatch"));
    }
    if file.read(&mut [0])? != 0 {
        return Err(invalid("trailing bytes in SRS cache"));
    }
    let generator = G1Projective::generator();
    if g1[0] != generator.into_affine()
        || g1[1] != (generator * tau).into_affine()
        || g1[max_len - 1] != (generator * tau.pow([(max_len - 1) as u64])).into_affine()
    {
        return Err(invalid(
            "SRS cache G1 anchors differ from the development seed",
        ));
    }
    Ok(Srs {
        g1,
        g2,
        tau_g2,
        degree_keys,
        public_setup: None,
    })
}

struct Temporary(PathBuf);

impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn write_cache(path: &Path, header: &[u8], srs: &Srs) -> io::Result<()> {
    let (temporary, mut file) = loop {
        let temporary = path.with_extension(format!(
            "{}.{}.partial",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => break (Temporary(temporary), file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    let mut hasher = blake3::Hasher::new();
    file.write_all(header)?;
    hasher.update(header);
    let mut bytes = Vec::new();
    for point in [&srs.g2, &srs.tau_g2].into_iter().chain(&srs.degree_keys) {
        point.serialize_uncompressed(&mut bytes).map_err(invalid)?;
    }
    file.write_all(&bytes)?;
    hasher.update(&bytes);
    bytes.resize(srs.g1.len().min(BLOCK_POINTS) * G1_BYTES, 0);
    for points in srs.g1.chunks(BLOCK_POINTS) {
        let block = &mut bytes[..points.len() * G1_BYTES];
        block
            .par_chunks_exact_mut(G1_BYTES)
            .zip(points.par_iter())
            .try_for_each(|(bytes, point)| point.serialize_uncompressed(bytes).map_err(invalid))?;
        file.write_all(block)?;
        hasher.update(block);
    }
    file.write_all(hasher.finalize().as_bytes())?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary.0, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "multi-stark-dev-srs-{}-{}",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self, max_len: usize, seed: &[u8]) -> PathBuf {
            self.0.join(format!(
                "{}.dev-srs",
                blake3::hash(&header(max_len, seed)).to_hex()
            ))
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn public_development_identity_is_prefix_independent_and_seed_degree_bound() {
        let directory = TestDirectory::new();
        let seed = b"public-development-prefix";
        let load = |length, degree, seed: &[u8]| {
            Srs::unsafe_dev_public_setup_with_cache(length, degree, seed, Some(&directory.0))
                .unwrap()
        };
        let prover = load(16, 31, seed);
        let verifier = load(2, 31, seed);
        let fresh = Srs::unsafe_dev_public_setup_with_cache(2, 31, seed, None).unwrap();
        for actual in [&verifier, &fresh] {
            assert_eq!(actual.public_setup(), prover.public_setup());
            assert_eq!(actual.g1, prover.g1[..2]);
            assert_eq!(actual.g2, prover.g2);
            assert_eq!(actual.tau_g2, prover.tau_g2);
            assert!(actual.degree_keys.is_empty());
            actual.validate().unwrap();
        }
        assert!(prover.degree_keys.is_empty());
        assert!(!prover.requires_shifted_commitment(2));
        prover.validate().unwrap();
        let metadata = prover.public_setup().unwrap();
        assert_eq!(metadata.max_degree, 31);
        assert_ne!(metadata.id, super::super::filecoin::filecoin_setup_id());

        let larger_degree = load(16, 63, seed);
        let minimum_degree = load(16, 15, seed);
        let another_seed = load(16, 31, b"another-public-development-seed");
        assert_eq!(larger_degree.g1, prover.g1);
        assert_eq!(larger_degree.g2, prover.g2);
        assert_eq!(larger_degree.tau_g2, prover.tau_g2);
        assert_eq!(minimum_degree.g1, prover.g1);
        for actual in [&larger_degree, &minimum_degree, &another_seed] {
            assert_ne!(actual.public_setup().unwrap().id, metadata.id);
            assert_ne!(
                actual.public_setup().unwrap().id,
                super::super::filecoin::filecoin_setup_id()
            );
        }
        assert_ne!(another_seed.g1[1], prover.g1[1]);
        assert_ne!(another_seed.tau_g2, prover.tau_g2);
    }

    #[test]
    fn public_development_setup_rejects_invalid_ranges_before_cache_io() {
        let directory = TestDirectory::new();
        let absent = directory.0.join("invalid-input-must-not-create-cache");
        for (length, degree) in [
            (0, 31),
            (1, 31),
            (3, 31),
            (2, 0),
            (16, 14),
            (2, usize::MAX),
            (1usize << (usize::BITS - 1), usize::MAX - 1),
        ] {
            let error = Srs::unsafe_dev_public_setup_with_cache(
                length,
                degree,
                b"invalid-public-development-range",
                Some(&absent),
            )
            .err()
            .expect("invalid public development range");
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(!absent.exists());
        }
    }

    #[test]
    fn public_development_setup_preserves_legacy_cache_and_damage_errors() {
        let directory = TestDirectory::new();
        let seed = b"shared-development-cache";
        let legacy = Srs::unsafe_dev_setup_with_cache(16, seed, Some(&directory.0)).unwrap();
        let path = directory.path(16, seed);
        let bytes = fs::read(&path).unwrap();
        let public =
            Srs::unsafe_dev_public_setup_with_cache(16, 31, seed, Some(&directory.0)).unwrap();
        assert_eq!(public.g1, legacy.g1);
        assert_eq!(public.g2, legacy.g2);
        assert_eq!(public.tau_g2, legacy.tau_g2);
        assert!(public.degree_keys.is_empty());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        let reloaded = Srs::unsafe_dev_setup_with_cache(16, seed, Some(&directory.0)).unwrap();
        assert_eq!(reloaded.degree_keys, legacy.degree_keys);
        assert!(reloaded.public_setup().is_none());
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);

        let mut damaged = bytes;
        *damaged.last_mut().unwrap() ^= 1;
        fs::write(&path, &damaged).unwrap();
        let error = Srs::unsafe_dev_public_setup_with_cache(16, 31, seed, Some(&directory.0))
            .err()
            .expect("damaged development cache");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(fs::read(&path).unwrap(), damaged);
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
    }

    #[test]
    fn public_development_proof_verifies_with_separately_loaded_anchors() {
        use crate::{
            ark_adapter::{KzgConfig, Scalar},
            expr::Expr,
            system::{CircuitInputs, System, SystemWitness},
            traits::Field as _,
        };
        use p3_matrix::dense::RowMajorMatrix;
        use std::sync::Arc;

        let directory = TestDirectory::new();
        let seed = b"public-development-proof";
        let load = |length, degree, seed: &[u8]| {
            Arc::new(
                Srs::unsafe_dev_public_setup_with_cache(length, degree, seed, Some(&directory.0))
                    .unwrap(),
            )
        };
        let prover = KzgConfig::new(load(16, 31, seed), 2);
        let verifier = KzgConfig::with_max_trace_len(load(2, 31, seed), 16, 2);
        assert_eq!(prover.transcript_seed(), verifier.transcript_seed());
        let traces: Vec<_> = [(4, 7), (2, 9)]
            .into_iter()
            .map(|(height, start)| {
                RowMajorMatrix::new_col(
                    (0..height)
                        .map(|row| Scalar::from_usize(start + row))
                        .collect(),
                )
            })
            .collect();
        let definitions: Vec<_> = traces
            .iter()
            .map(|trace| CircuitInputs {
                main_width: 1,
                preprocessed: Some(trace.clone()),
                constraints: vec![Expr::main(0) - Expr::preprocessed(0)],
                ..Default::default()
            })
            .collect();
        let (mut system, key) = System::new(prover, definitions);
        let proof =
            system.prove_multiple_claims(&key, &[], SystemWitness::from_stage_1(traces, &system));
        system.config = verifier;
        system.verify_multiple_claims(&[], &proof).unwrap();
        for (degree, seed) in [
            (63, seed.as_slice()),
            (31, b"different-proof-seed".as_slice()),
        ] {
            system.config = KzgConfig::with_max_trace_len(load(2, degree, seed), 16, 2);
            assert!(system.verify_multiple_claims(&[], &proof).is_err());
        }
    }

    #[test]
    fn cache_round_trip_and_key_separation() {
        let directory = TestDirectory::new();
        for (size, seed) in [
            (2, b"first".as_slice()),
            (16, b"first"),
            (32, b"first"),
            (16, b"second"),
        ] {
            let fresh = Srs::unsafe_dev_setup_with_cache(size, seed, Some(&directory.0)).unwrap();
            let cached = Srs::unsafe_dev_setup_with_cache(size, seed, Some(&directory.0)).unwrap();
            let expected = Srs::unsafe_dev_setup(size, seed);
            for actual in [fresh, cached] {
                assert_eq!(actual.g1, expected.g1);
                assert_eq!(actual.g2, expected.g2);
                assert_eq!(actual.tau_g2, expected.tau_g2);
                assert_eq!(actual.degree_keys, expected.degree_keys);
                actual.validate().unwrap();
            }
        }
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 4);
    }

    #[test]
    fn damaged_cache_fails_without_regeneration() {
        let directory = TestDirectory::new();
        let seed = b"damage";
        Srs::unsafe_dev_setup_with_cache(16, seed, Some(&directory.0)).unwrap();
        let path = directory.path(16, seed);
        let valid = fs::read(&path).unwrap();
        let mut variants = vec![
            valid[..valid.len() - 1].to_vec(),
            [valid.as_slice(), &[0]].concat(),
        ];
        for offset in [0, MAGIC.len(), valid.len() - 1] {
            let mut damaged = valid.clone();
            damaged[offset] ^= 1;
            variants.push(damaged);
        }
        // A valid but different interior point must still fail the checksum.
        let mut damaged = valid.clone();
        let start = header(16, seed).len() + (16usize.ilog2() as usize + 3) * G2_BYTES;
        let point = damaged[start + G1_BYTES..start + 2 * G1_BYTES].to_vec();
        damaged[start + 7 * G1_BYTES..start + 8 * G1_BYTES].copy_from_slice(&point);
        variants.push(damaged);
        for damaged in variants {
            fs::write(&path, &damaged).unwrap();
            assert!(Srs::unsafe_dev_setup_with_cache(16, seed, Some(&directory.0)).is_err());
            assert_eq!(fs::read(&path).unwrap(), damaged);
        }
    }

    #[test]
    fn cache_rejects_a_different_seed_even_with_valid_checksum() {
        let directory = TestDirectory::new();
        let expected_header = header(16, b"expected");
        let path = directory.path(16, b"expected");
        write_cache(
            &path,
            &expected_header,
            &Srs::unsafe_dev_setup(16, b"different"),
        )
        .unwrap();
        assert!(Srs::unsafe_dev_setup_with_cache(16, b"expected", Some(&directory.0)).is_err());
    }

    #[test]
    fn concurrent_cache_writers_publish_complete_files() {
        let directory = TestDirectory::new();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    Srs::unsafe_dev_setup_with_cache(32, b"concurrent", Some(&directory.0))
                        .unwrap();
                });
            }
        });
        let actual =
            Srs::unsafe_dev_setup_with_cache(32, b"concurrent", Some(&directory.0)).unwrap();
        assert_eq!(actual.g1, Srs::unsafe_dev_setup(32, b"concurrent").g1);
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
    }
}
