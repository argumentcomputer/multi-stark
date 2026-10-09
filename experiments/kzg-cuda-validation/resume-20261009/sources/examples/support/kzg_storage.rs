use multi_stark::{
    ark_adapter::{Scalar, encoding},
    system::CircuitInputs,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::OnceLock,
};

fn trace_codec() -> io::Result<&'static str> {
    match std::env::var("MULTI_STARK_KZG_TRACE_CODEC").as_deref() {
        Ok("pzstd") => Ok("pzstd"),
        Ok("zstd") => Ok("zstd"),
        Ok("auto") | Err(std::env::VarError::NotPresent) => {
            static AVAILABLE: OnceLock<&str> = OnceLock::new();
            Ok(AVAILABLE.get_or_init(|| {
                if Command::new("pzstd")
                    .arg("--version")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success())
                {
                    "pzstd"
                } else {
                    "zstd"
                }
            }))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "MULTI_STARK_KZG_TRACE_CODEC must be auto, pzstd, or zstd",
        )),
    }
}

struct ReapedChild(Child);

impl Drop for ReapedChild {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub widths: Vec<usize>,
    pub heights: Vec<usize>,
    pub claims: Vec<Vec<Scalar>>,
}
pub fn save<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let temp = path.with_extension("partial");
    let mut f = BufWriter::new(File::create(&temp)?);
    bincode::serde::encode_into_std_write(value, &mut f, bincode::config::standard())?;
    f.flush()?;
    drop(f);
    fs::rename(temp, path)?;
    Ok(())
}
pub fn load<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let mut f = BufReader::new(File::open(path)?);
    let value = bincode::serde::decode_from_std_read(&mut f, bincode::config::standard())?;
    let mut tail = [0];
    if f.read(&mut tail)? != 0 {
        return Err("trailing metadata".into());
    }
    Ok(value)
}
pub fn write_matrix(path: &Path, m: &RowMajorMatrix<Scalar>) -> Result<()> {
    write_matrix_with_codec(path, m, trace_codec()?)
}
fn write_matrix_with_codec(path: &Path, m: &RowMajorMatrix<Scalar>, codec: &str) -> Result<()> {
    let temp = path.with_extension("partial");
    let mut command = Command::new(codec);
    command.args(["-q", "-1", "-c"]);
    if codec == "zstd" {
        command.arg("-T0");
    }
    let mut child = ReapedChild(
        command
            .stdin(Stdio::piped())
            .stdout(File::create(&temp)?)
            .spawn()?,
    );
    let mut w = BufWriter::with_capacity(1 << 20, child.0.stdin.take().unwrap());
    w.write_all(&(m.width() as u64).to_le_bytes())?;
    w.write_all(&(m.height() as u64).to_le_bytes())?;
    encoding::write_scalars(&mut w, &m.values)?;
    w.flush()?;
    drop(w);
    if !child.0.wait()?.success() {
        return Err("zstd write failed".into());
    }
    fs::rename(temp, path)?;
    Ok(())
}
pub fn read_matrix(path: &Path) -> Result<RowMajorMatrix<Scalar>> {
    read_matrix_bounded(path, 1 << 24)
}
pub fn read_matrix_bounded(path: &Path, max_height: usize) -> Result<RowMajorMatrix<Scalar>> {
    read_matrix_with_codec(path, max_height, trace_codec()?)
}
fn read_matrix_with_codec(
    path: &Path,
    max_height: usize,
    codec: &str,
) -> Result<RowMajorMatrix<Scalar>> {
    let mut child = ReapedChild(
        Command::new(codec)
            .args(["-q", "-d", "-c"])
            .arg(path)
            .stdout(Stdio::piped())
            .spawn()?,
    );
    let mut r = BufReader::with_capacity(1 << 20, child.0.stdout.take().unwrap());
    let mut b = [0; 8];
    r.read_exact(&mut b)?;
    let width = usize::try_from(u64::from_le_bytes(b))?;
    r.read_exact(&mut b)?;
    let height = usize::try_from(u64::from_le_bytes(b))?;
    if width == 0 || !height.is_power_of_two() || width > 1024 || height > max_height {
        return Err("bad matrix dimensions".into());
    }
    let count = width.checked_mul(height).ok_or("matrix overflow")?;
    let values = encoding::read_scalars(&mut r, count)?;
    if io::copy(&mut r, &mut io::sink())? != 0 {
        return Err("trailing matrix bytes".into());
    }
    drop(r);
    if !child.0.wait()?.success() {
        return Err("zstd read failed".into());
    }
    Ok(RowMajorMatrix::new(values, width))
}
pub fn definition(dir: &Path, index: usize) -> Result<CircuitInputs<Scalar>> {
    let mut input: CircuitInputs<Scalar> = load(&dir.join(format!("{index}.meta")))?;
    input.preprocessed = Some(read_matrix(&dir.join(format!("{index}.fixed.zst")))?);
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use multi_stark::traits::{Algebra, Field};

    #[test]
    fn parallel_frames_and_legacy_streams_are_interoperable() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("kzg-trace-codecs-{}", std::process::id()));
        fs::create_dir(&dir)?;
        let matrix = RowMajorMatrix::new(
            (0..4 * (1 << 17))
                .map(|i| {
                    let value = Scalar::from_usize(i * 7919);
                    value * value - Scalar::ONE
                })
                .collect(),
            4,
        );
        let mut codecs = vec!["zstd"];
        if Command::new("pzstd")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
        {
            codecs.push("pzstd");
        }
        for encoder in &codecs {
            let path = dir.join(format!("{encoder}.zst"));
            write_matrix_with_codec(&path, &matrix, encoder)?;
            for decoder in &codecs {
                assert_eq!(matrix, read_matrix_with_codec(&path, 1 << 17, decoder)?);
            }
        }
        fs::remove_dir_all(dir)?;
        Ok(())
    }
}
