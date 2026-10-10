//! Explicit ceremony selection and checkpoint binding shared by both KZG stages.

use multi_stark::ark_adapter::{
    KzgConfig, Srs,
    srs::filecoin::{
        FILECOIN_MAX_TRACE_LEN, FILECOIN_PUBLIC_MAX_DEGREE, filecoin_setup_id, load_filecoin_cache,
    },
};
use std::{
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(crate) const BINDING_FILE: &str = "kzg-setup-id.bin";
pub(crate) const DEVELOPMENT_PUBLIC_SEED: &[u8] = b"init-fri-kzg-ordinary-v1";

#[derive(Clone)]
pub(crate) enum SetupSource {
    Filecoin { cache: PathBuf, digest: [u8; 32] },
    Development { cache: Option<PathBuf> },
    KnownTrapdoorPublicDegree { cache: Option<PathBuf> },
}

impl SetupSource {
    pub(crate) fn from_env() -> Result<Self> {
        match std::env::var("MULTI_STARK_KZG_SETUP")
            .as_deref()
            .unwrap_or("filecoin")
        {
            "filecoin" => {
                let cache = std::env::var_os("MULTI_STARK_KZG_FILECOIN_CACHE").ok_or(
                    "set MULTI_STARK_KZG_FILECOIN_CACHE to the authenticated Filecoin cache",
                )?;
                let pin = std::env::var("MULTI_STARK_KZG_FILECOIN_DIGEST").map_err(
                    |_| "set MULTI_STARK_KZG_FILECOIN_DIGEST to the trusted import receipt",
                )?;
                let digest = blake3::Hash::from_hex(&pin)
                    .map_err(|_| "Filecoin cache digest must be 64 hexadecimal digits")?;
                Ok(Self::Filecoin {
                    cache: cache.into(),
                    digest: *digest.as_bytes(),
                })
            }
            "development" => {
                if std::env::var_os("MULTI_STARK_KZG_FILECOIN_CACHE").is_some()
                    || std::env::var_os("MULTI_STARK_KZG_FILECOIN_DIGEST").is_some()
                {
                    return Err("development mode cannot also select a Filecoin setup".into());
                }
                Ok(Self::Development {
                    cache: std::env::var_os("MULTI_STARK_KZG_DEV_SRS_CACHE").map(PathBuf::from),
                })
            }
            _ => Err("MULTI_STARK_KZG_SETUP must be filecoin or development".into()),
        }
    }

    pub(crate) fn development_public_degree_from_env() -> Result<Self> {
        Self::development_public_degree_with_env(|name| std::env::var_os(name))
    }

    fn development_public_degree_with_env(
        mut get: impl FnMut(&str) -> Option<OsString>,
    ) -> Result<Self> {
        if get("MULTI_STARK_KZG_SETUP").as_deref() != Some(std::ffi::OsStr::new("development")) {
            return Err(
                "development v4 diagnostics require explicit MULTI_STARK_KZG_SETUP=development"
                    .into(),
            );
        }
        if get("MULTI_STARK_KZG_FILECOIN_CACHE").is_some()
            || get("MULTI_STARK_KZG_FILECOIN_DIGEST").is_some()
        {
            return Err("development v4 diagnostics cannot also select a Filecoin setup".into());
        }
        Ok(Self::KnownTrapdoorPublicDegree {
            cache: get("MULTI_STARK_KZG_DEV_SRS_CACHE").map(PathBuf::from),
        })
    }

    pub(crate) fn is_development(&self) -> bool {
        matches!(
            self,
            Self::Development { .. } | Self::KnownTrapdoorPublicDegree { .. }
        )
    }

    pub(crate) fn uses_public_degree(&self) -> bool {
        !matches!(self, Self::Development { .. })
    }

    pub(crate) fn is_diagnostic(&self) -> bool {
        matches!(self, Self::KnownTrapdoorPublicDegree { .. })
    }

    pub(crate) fn check_height(&self, height: usize) -> Result<()> {
        if height < 2
            || !height.is_power_of_two()
            || (self.uses_public_degree() && height > FILECOIN_MAX_TRACE_LEN)
        {
            return Err("trace height exceeds the selected setup profile".into());
        }
        Ok(())
    }

    pub(crate) fn identity(
        &self,
        dev_seed: &[u8],
        height: usize,
        quotient: usize,
    ) -> Result<[u8; 32]> {
        self.check_height(height)?;
        if self.is_diagnostic() {
            let config = self.config(dev_seed, height, quotient, true)?;
            let mut hash = blake3::Hasher::new();
            hash.update(b"multi-stark/kzg-known-trapdoor-outer-profile/v1");
            hash.update(config.transcript_seed());
            return Ok(*hash.finalize().as_bytes());
        }
        let mut hash = blake3::Hasher::new();
        hash.update(b"multi-stark/kzg-pipeline-setup/v1");
        hash.update(&(height as u64).to_le_bytes());
        hash.update(&(quotient as u64).to_le_bytes());
        match self {
            Self::Filecoin { digest, .. } => {
                hash.update(b"multi-stark/kzg/v4");
                hash.update(&filecoin_setup_id());
                hash.update(&(FILECOIN_PUBLIC_MAX_DEGREE as u64).to_le_bytes());
                hash.update(digest);
            }
            Self::Development { .. } => {
                hash.update(b"multi-stark/kzg/v3");
                hash.update(dev_seed);
            }
            Self::KnownTrapdoorPublicDegree { .. } => unreachable!(),
        }
        Ok(*hash.finalize().as_bytes())
    }

    pub(crate) fn bind_stage(&self, dir: &Path, identity: &[u8; 32]) -> Result<()> {
        let path = dir.join(BINDING_FILE);
        if path.try_exists()? {
            self.check_binding(dir, identity)?;
        } else {
            let checkpoints = dir.join("kzg");
            if checkpoints.try_exists()? && fs::read_dir(checkpoints)?.next().transpose()?.is_some()
            {
                return Err(
                    "cannot label existing KZG checkpoints without a setup identity".into(),
                );
            }
            fs::write(path, identity)?;
        }
        Ok(())
    }

    pub(crate) fn check_binding(&self, dir: &Path, identity: &[u8; 32]) -> Result<()> {
        match fs::read(dir.join(BINDING_FILE)) {
            Ok(bytes) if bytes == identity => Ok(()),
            // Legacy experiment archives predate setup markers; public-degree
            // profiles always require the marker, even with a known trapdoor.
            Err(error)
                if error.kind() == io::ErrorKind::NotFound
                    && matches!(self, Self::Development { .. }) =>
            {
                Ok(())
            }
            Err(error) => Err(error.into()),
            _ => Err("checkpoint belongs to a different KZG setup or degree profile".into()),
        }
    }

    pub(crate) fn config(
        &self,
        dev_seed: &[u8],
        height: usize,
        quotient: usize,
        verify_only: bool,
    ) -> Result<KzgConfig> {
        self.check_height(height)?;
        let srs = match self {
            Self::Filecoin { cache, digest } => {
                load_filecoin_cache(cache, *digest, if verify_only { 2 } else { height })?
            }
            Self::Development { cache } => {
                Srs::unsafe_dev_setup_with_cache(height, dev_seed, cache.as_deref())?
            }
            Self::KnownTrapdoorPublicDegree { cache } => {
                if quotient != 2 {
                    return Err(
                        "development public-degree profile requires quotient budget two".into(),
                    );
                }
                // Recursive verification checks the inner and outer G2 keys,
                // so both stages use this seed rather than their legacy seeds.
                Srs::unsafe_dev_public_setup_with_cache(
                    if verify_only { 2 } else { height },
                    FILECOIN_PUBLIC_MAX_DEGREE,
                    DEVELOPMENT_PUBLIC_SEED,
                    if verify_only { None } else { cache.as_deref() },
                )?
            }
        };
        Ok(KzgConfig::with_max_trace_len(
            Arc::new(srs),
            height,
            quotient,
        ))
    }

    pub(crate) fn security_description(&self) -> &'static str {
        match self {
            Self::Filecoin { .. } => {
                "Authenticated Filecoin challenge_19; full-public-degree KZG v4. Security assumes an honest ceremony contribution and the generic bilinear group/random oracle model described in docs/kzg-performance.md.\n"
            }
            Self::Development { .. } => {
                "Known-trapdoor development SRS. Correctness and cost experiment only.\n"
            }
            Self::KnownTrapdoorPublicDegree { .. } => {
                "Known-trapdoor development public-degree KZG v4 diagnostic. Not Filecoin ceremony parameters. Correctness and cost experiment only.\n"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn development_public_degree_requires_explicit_unambiguous_selection() {
        let select = |entries: &[(&str, &str)]| {
            SetupSource::development_public_degree_with_env(|name| {
                entries
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| OsString::from(*value))
            })
        };
        for mode in [None, Some("filecoin"), Some("development-v4"), Some("")] {
            let entries: Vec<_> = mode
                .map(|mode| ("MULTI_STARK_KZG_SETUP", mode))
                .into_iter()
                .collect();
            assert!(select(&entries).is_err());
        }
        for name in [
            "MULTI_STARK_KZG_FILECOIN_CACHE",
            "MULTI_STARK_KZG_FILECOIN_DIGEST",
        ] {
            assert!(select(&[("MULTI_STARK_KZG_SETUP", "development"), (name, "")]).is_err());
        }
        let source = select(&[
            ("MULTI_STARK_KZG_SETUP", "development"),
            ("MULTI_STARK_KZG_DEV_SRS_CACHE", "unused-cache"),
        ])
        .unwrap();
        assert!(source.is_development());
        assert!(source.uses_public_degree());
        assert!(source.is_diagnostic());
        assert!(
            matches!(source, SetupSource::KnownTrapdoorPublicDegree { cache: Some(path) }
            if path == Path::new("unused-cache"))
        );
    }

    #[test]
    fn development_public_degree_binds_policy_and_uses_two_verifier_anchors() {
        let dir = std::env::temp_dir().join(format!("kzg-public-setup-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let cache = dir.join("must-not-be-created");
        let source = SetupSource::KnownTrapdoorPublicDegree {
            cache: Some(cache.clone()),
        };
        let verifier = source.config(b"ignored-outer-seed", 16, 2, true).unwrap();
        let largest = source
            .config(b"ignored-inner-seed", 1 << 27, 2, true)
            .unwrap();
        let prover = SetupSource::KnownTrapdoorPublicDegree { cache: None }
            .config(b"ignored", 16, 2, false)
            .unwrap();
        assert!(!cache.exists());
        assert_eq!(verifier.srs().max_len(), 2);
        assert_eq!(largest.srs().max_len(), 2);
        assert_eq!(prover.srs().max_len(), 16);
        assert_eq!(verifier.srs().g1, prover.srs().g1[..2]);
        assert_eq!(verifier.srs().g2, prover.srs().g2);
        assert_eq!(verifier.srs().tau_g2, prover.srs().tau_g2);
        assert_eq!(verifier.transcript_seed(), prover.transcript_seed());
        assert!(
            verifier
                .transcript_seed()
                .starts_with(b"multi-stark/kzg/v4")
        );
        assert_eq!(verifier.srs().public_setup(), largest.srs().public_setup());
        assert_eq!(verifier.srs().public_setup(), prover.srs().public_setup());
        let public = verifier.srs().public_setup().unwrap();
        assert_eq!(public.max_degree, (1 << 28) - 2);
        assert_ne!(public.id, filecoin_setup_id());
        assert!(prover.srs().degree_keys.is_empty());
        assert!(!prover.requires_shifted_commitment(2));

        let identity = source.identity(b"ignored", 16, 2).unwrap();
        let mut hash = blake3::Hasher::new();
        hash.update(b"multi-stark/kzg-known-trapdoor-outer-profile/v1");
        hash.update(verifier.transcript_seed());
        assert_eq!(identity, *hash.finalize().as_bytes());
        assert_eq!(identity, source.identity(b"another caller", 16, 2).unwrap());
        assert_ne!(identity, source.identity(b"ignored", 32, 2).unwrap());
        for height in [0, 1, 3, 1 << 28] {
            assert!(source.config(b"ignored", height, 2, true).is_err());
            assert!(source.identity(b"ignored", height, 2).is_err());
        }
        for quotient in [0, 1, 4] {
            assert!(source.config(b"ignored", 16, quotient, false).is_err());
            assert!(source.identity(b"ignored", 16, quotient).is_err());
        }
        assert!(!cache.exists());
        assert!(source.check_binding(&dir, &identity).is_err());
        source.bind_stage(&dir, &identity).unwrap();
        source.check_binding(&dir, &identity).unwrap();
        assert!(source.check_binding(&dir, &[0; 32]).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn legacy_development_keeps_seed_degree_keys_and_unbound_archives() {
        let source = SetupSource::Development { cache: None };
        assert!(source.is_development());
        assert!(!source.uses_public_degree());
        assert!(!source.is_diagnostic());
        source.check_height(1 << 28).unwrap();
        let config = source.config(b"legacy-seed", 16, 2, true).unwrap();
        let expected = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(16, b"legacy-seed")), 2);
        assert_eq!(config.transcript_seed(), expected.transcript_seed());
        assert_eq!(config.srs().g1, expected.srs().g1);
        assert_eq!(config.srs().degree_keys, expected.srs().degree_keys);
        assert!(config.srs().public_setup().is_none());
        assert!(config.requires_shifted_commitment(2));
        let mut hash = blake3::Hasher::new();
        hash.update(b"multi-stark/kzg-pipeline-setup/v1");
        hash.update(&16u64.to_le_bytes());
        hash.update(&2u64.to_le_bytes());
        hash.update(b"multi-stark/kzg/v3");
        hash.update(b"legacy-seed");
        let identity = source.identity(b"legacy-seed", 16, 2).unwrap();
        assert_eq!(identity, *hash.finalize().as_bytes());
        let absent = std::env::temp_dir().join(format!("kzg-unbound-{}", std::process::id()));
        assert!(!absent.exists());
        source.check_binding(&absent, &identity).unwrap();
    }

    #[test]
    fn checkpoint_identity_binds_setup_degree_profile_and_receipt() {
        let source = SetupSource::Filecoin {
            cache: "unused".into(),
            digest: [1; 32],
        };
        assert!(!source.is_development());
        assert!(source.uses_public_degree());
        assert!(!source.is_diagnostic());
        let identity = source.identity(b"ignored", 16, 2).unwrap();
        assert_eq!(identity, source.identity(b"another stage", 16, 2).unwrap());
        assert_ne!(identity, source.identity(b"ignored", 32, 2).unwrap());
        assert_ne!(identity, source.identity(b"ignored", 16, 4).unwrap());
        assert_ne!(
            identity,
            SetupSource::Filecoin {
                cache: "unused".into(),
                digest: [2; 32]
            }
            .identity(b"ignored", 16, 2)
            .unwrap()
        );
        let dev = SetupSource::Development { cache: None };
        assert_ne!(identity, dev.identity(b"seed", 16, 2).unwrap());
        assert_ne!(
            dev.identity(b"seed", 16, 2).unwrap(),
            dev.identity(b"other", 16, 2).unwrap()
        );
        assert!(source.identity(b"ignored", 1 << 28, 2).is_err());
        let dir = std::env::temp_dir().join(format!("kzg-setup-binding-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        assert!(source.check_binding(&dir, &identity).is_err());
        source.bind_stage(&dir, &identity).unwrap();
        source.check_binding(&dir, &identity).unwrap();
        assert!(source.bind_stage(&dir, &[0; 32]).is_err());
        fs::remove_file(dir.join(BINDING_FILE)).unwrap();
        fs::create_dir(dir.join("kzg")).unwrap();
        fs::write(dir.join("kzg/setup-0.bin"), b"unidentified checkpoint").unwrap();
        assert!(source.bind_stage(&dir, &identity).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
