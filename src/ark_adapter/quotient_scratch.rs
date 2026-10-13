//! Temporary quotient cosets kept outside the live constraint working set.

use std::{
    fs::{self, File, OpenOptions},
    io::{self, Seek},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::{Scalar, encoding};

pub(super) enum Coset {
    Memory(Vec<Scalar>),
    File(ScratchFile),
}

pub(super) struct ScratchFile {
    file: Option<File>,
    path: PathBuf,
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        drop(self.file.take());
        if let Err(error) = fs::remove_file(&self.path) {
            tracing::warn!(path = %self.path.display(), %error, "KZG scratch cleanup failed");
        }
    }
}

impl Coset {
    pub(super) fn store(values: Vec<Scalar>, directory: Option<&Path>) -> io::Result<Self> {
        let Some(directory) = directory else {
            return Ok(Self::Memory(values));
        };
        static NEXT: AtomicU64 = AtomicU64::new(0);
        fs::create_dir_all(directory)?;
        let path = directory.join(format!(
            "quotient-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut scratch = ScratchFile {
            file: Some(file),
            path,
        };
        encoding::write_scalars(scratch.file.as_mut().unwrap(), &values)?;
        Ok(Self::File(scratch))
    }

    pub(super) fn load(self, count: usize) -> io::Result<Vec<Scalar>> {
        match self {
            Self::Memory(values) => {
                assert_eq!(values.len(), count);
                Ok(values)
            }
            Self::File(mut scratch) => {
                let file = scratch.file.as_mut().unwrap();
                if file.metadata()?.len() != (count as u64) * 32 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "quotient scratch length differs",
                    ));
                }
                file.rewind()?;
                encoding::read_scalars(file, count)
            }
        }
    }
}
