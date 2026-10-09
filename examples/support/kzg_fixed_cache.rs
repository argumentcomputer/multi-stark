//! Trusted local preprocessing, independent of each proof's witness and openings.

use super::storage::{self, Manifest, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    sync::OnceLock,
};

#[derive(Serialize, Deserialize)]
pub(crate) struct Shape {
    pub(crate) widths: Vec<usize>,
    pub(crate) heights: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
struct Entry {
    identity: [u8; 32],
    shape: Shape,
    files: Vec<(String, u64)>,
}

pub(crate) struct FixedCache {
    identity: [u8; 32],
    directory: PathBuf,
}

fn executable_digest() -> Result<[u8; 32]> {
    static DIGEST: OnceLock<[u8; 32]> = OnceLock::new();
    if let Some(digest) = DIGEST.get() {
        return Ok(*digest);
    }
    let mut file = File::open(std::env::current_exe()?)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let digest = *hasher.finalize().as_bytes();
    Ok(*DIGEST.get_or_init(|| digest))
}

impl FixedCache {
    /// The executable digest conservatively binds the compiler, lowering,
    /// dependency versions, SRS recipe, and all circuit implementation choices.
    pub(crate) fn for_stage(directory: &Path, profile: &[u8]) -> Result<Option<Self>> {
        let Some(root) = std::env::var_os("MULTI_STARK_KZG_FIXED_CACHE") else {
            return Ok(None);
        };
        let cache = Self::from_profile(Path::new(&root), profile)?;
        let mut marker = cache.identity.to_vec();
        marker.extend_from_slice(&executable_digest()?);
        fs::write(directory.join("fixed-cache-id.bin"), marker)?;
        Ok(Some(cache))
    }

    pub(crate) fn from_profile(root: &Path, profile: &[u8]) -> Result<Self> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"multi-stark/kzg-fixed-cache/v1");
        hasher.update(&executable_digest()?);
        hasher.update(profile);
        let identity = *hasher.finalize().as_bytes();
        Ok(Self::new(root, identity))
    }

    pub(crate) fn for_prove(directory: &Path) -> Result<Option<Self>> {
        let Some(root) = std::env::var_os("MULTI_STARK_KZG_FIXED_CACHE") else {
            return Ok(None);
        };
        let marker = match fs::read(directory.join("fixed-cache-id.bin")) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        if marker.len() != 64 || marker[32..] != executable_digest()? {
            return Err("fixed cache was staged by a different executable".into());
        }
        let identity = marker[..32].try_into().unwrap();
        Ok(Some(Self::new(Path::new(&root), identity)))
    }

    fn new(root: &Path, identity: [u8; 32]) -> Self {
        Self {
            identity,
            directory: root.join(blake3::Hash::from_bytes(identity).to_hex().as_str()),
        }
    }

    pub(crate) fn restore(&self, staged: &Path) -> Result<Option<Shape>> {
        let ready = self.directory.join("ready.bin");
        if !ready.try_exists()? {
            eprintln!(
                "Fixed preprocessing cache miss: {}",
                self.directory.display()
            );
            return Ok(None);
        }
        let entry: Entry = storage::load(&ready)?;
        if entry.identity != self.identity
            || entry.shape.widths.is_empty()
            || entry.shape.widths.len() != entry.shape.heights.len()
            || entry.files.len() != entry.shape.widths.len() * 3
        {
            return Err("fixed cache identity or shape mismatch".into());
        }
        let expected = file_names(entry.shape.widths.len());
        for ((name, size), expected) in entry.files.iter().zip(expected) {
            if *name != expected || fs::metadata(self.directory.join(name))?.len() != *size {
                return Err("fixed cache entry is incomplete or damaged".into());
            }
        }
        fs::create_dir_all(staged.join("kzg"))?;
        for (name, _) in &entry.files {
            link_or_copy(&self.directory.join(name), &staged.join(name))?;
        }
        eprintln!("Reused fixed preprocessing: {}", self.directory.display());
        Ok(Some(entry.shape))
    }

    /// Publish only after the proof using these parameters has verified.
    /// The cache directory must be trusted like the development SRS cache;
    /// metadata and field decoding detect truncation, not a malicious producer.
    pub(crate) fn publish(&self, staged: &Path) -> Result<()> {
        if self.directory.join("ready.bin").try_exists()? {
            return Ok(());
        }
        let manifest: Manifest = storage::load(&staged.join("manifest.bin"))?;
        let temporary = self
            .directory
            .with_extension(format!("partial-{}", std::process::id()));
        fs::create_dir_all(temporary.join("kzg"))?;
        let mut files = Vec::new();
        for name in file_names(manifest.widths.len()) {
            let source = staged.join(&name);
            let size = fs::metadata(&source)?.len();
            link_or_copy(&source, &temporary.join(&name))?;
            files.push((name, size));
        }
        storage::save(
            &temporary.join("ready.bin"),
            &Entry {
                identity: self.identity,
                shape: Shape {
                    widths: manifest.widths,
                    heights: manifest.heights,
                },
                files,
            },
        )?;
        match fs::rename(&temporary, &self.directory) {
            Ok(()) => (),
            Err(_) if self.directory.join("ready.bin").try_exists()? => {
                fs::remove_dir_all(temporary)?;
            }
            Err(error) => return Err(error.into()),
        }
        eprintln!(
            "Published fixed preprocessing: {}",
            self.directory.display()
        );
        Ok(())
    }
}

fn file_names(count: usize) -> impl Iterator<Item = String> {
    (0..count).flat_map(|i| {
        [
            format!("{i}.meta"),
            format!("kzg/setup-{i}.bin"),
            format!("kzg/fixed-{i}.bin"),
        ]
    })
}

fn link_or_copy(source: &Path, destination: &Path) -> Result<()> {
    if destination.try_exists()? {
        return Err(format!("refusing to replace {}", destination.display()).into());
    }
    match fs::hard_link(source, destination) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::CrossesDevices => {
            fs::copy(source, destination)?;
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}
