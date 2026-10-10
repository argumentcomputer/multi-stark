use super::{OUTER_SEED, setup::SetupSource, storage};
use multi_stark::ark_adapter::{KzgConfig, Srs};
use std::{fs, path::Path, sync::Arc};

pub(crate) const PUBLIC_MAX_DEGREE: usize = (1 << 28) - 2;
pub(crate) const MAX_TRACE_LEN: usize = 1 << 27;

pub(crate) enum Parameters<'a> {
    Selected(&'a SetupSource),
    KnownTrapdoorPublicDegree {
        seed: &'a [u8],
        cache: Option<&'a Path>,
    },
}

impl Parameters<'_> {
    pub(super) fn uses_public_degree(&self) -> bool {
        match self {
            Self::Selected(source) => source.uses_public_degree(),
            Self::KnownTrapdoorPublicDegree { .. } => true,
        }
    }

    pub(super) fn is_development(&self) -> bool {
        match self {
            Self::Selected(source) => source.is_development(),
            Self::KnownTrapdoorPublicDegree { .. } => true,
        }
    }

    pub(super) fn is_diagnostic(&self) -> bool {
        match self {
            Self::Selected(source) => source.is_diagnostic(),
            Self::KnownTrapdoorPublicDegree { .. } => true,
        }
    }

    pub(super) fn filecoin_manifest_digest(&self) -> Option<[u8; 32]> {
        match self {
            Self::Selected(SetupSource::Filecoin { digest, .. }) => Some(*digest),
            _ => None,
        }
    }

    pub(super) fn check_height(&self, height: usize) -> storage::Result<()> {
        match self {
            Self::Selected(source) => source.check_height(height),
            Self::KnownTrapdoorPublicDegree { .. } => {
                if height < 2 || !height.is_power_of_two() || height > MAX_TRACE_LEN {
                    return Err("trace height exceeds the development public-degree profile".into());
                }
                Ok(())
            }
        }
    }

    pub(crate) fn identity(&self, height: usize, quotient: usize) -> storage::Result<[u8; 32]> {
        match self {
            Self::Selected(source) => source.identity(OUTER_SEED, height, quotient),
            Self::KnownTrapdoorPublicDegree { .. } => {
                let config = self.config(height, quotient, true)?;
                let mut hash = blake3::Hasher::new();
                hash.update(b"multi-stark/kzg-known-trapdoor-outer-profile/v1");
                hash.update(config.transcript_seed());
                Ok(*hash.finalize().as_bytes())
            }
        }
    }

    pub(super) fn bind_stage(&self, dir: &Path, identity: &[u8; 32]) -> storage::Result<()> {
        if let Self::Selected(source) = self {
            return source.bind_stage(dir, identity);
        }
        let path = dir.join(super::setup::BINDING_FILE);
        if path.try_exists()? {
            self.check_binding(dir, identity)
        } else {
            let checkpoints = dir.join("kzg");
            if checkpoints.try_exists()? && fs::read_dir(checkpoints)?.next().transpose()?.is_some()
            {
                return Err(
                    "cannot label existing KZG checkpoints without a setup identity".into(),
                );
            }
            fs::write(path, identity)?;
            Ok(())
        }
    }

    pub(super) fn check_binding(&self, dir: &Path, identity: &[u8; 32]) -> storage::Result<()> {
        if let Self::Selected(source) = self {
            return source.check_binding(dir, identity);
        }
        if fs::read(dir.join(super::setup::BINDING_FILE))? != identity {
            return Err("checkpoint belongs to a different KZG setup or degree profile".into());
        }
        Ok(())
    }

    pub(crate) fn config(
        &self,
        height: usize,
        quotient: usize,
        verify_only: bool,
    ) -> storage::Result<KzgConfig> {
        self.check_height(height)?;
        match self {
            Self::Selected(source) => source.config(OUTER_SEED, height, quotient, verify_only),
            Self::KnownTrapdoorPublicDegree { seed, cache } => {
                if quotient != 2 {
                    return Err("development outer profile requires quotient budget two".into());
                }
                let srs = Srs::unsafe_dev_public_setup_with_cache(
                    if verify_only { 2 } else { height },
                    PUBLIC_MAX_DEGREE,
                    seed,
                    if verify_only { None } else { *cache },
                )?;
                Ok(KzgConfig::with_max_trace_len(
                    Arc::new(srs),
                    height,
                    quotient,
                ))
            }
        }
    }

    pub(super) fn security_description(&self) -> &'static str {
        match self {
            Self::Selected(source) => source.security_description(),
            Self::KnownTrapdoorPublicDegree { .. } => {
                "Known-trapdoor development public-degree KZG v4 diagnostic. Not Filecoin ceremony parameters. Correctness and cost experiment only.\n"
            }
        }
    }
}

#[cfg(test)]
mod tests;
