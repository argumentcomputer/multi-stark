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
    read_matrix_with_codec(path, max_height, trace_codec()?, None)
}
pub fn read_matrix_with_shape(
    path: &Path,
    width: usize,
    height: usize,
) -> Result<RowMajorMatrix<Scalar>> {
    read_matrix_with_codec(path, height, trace_codec()?, Some((width, height)))
}
fn read_matrix_with_codec(
    path: &Path,
    max_height: usize,
    codec: &str,
    expected: Option<(usize, usize)>,
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
    if expected.is_some_and(|shape| shape != (width, height)) {
        return Err("matrix dimensions differ from the staged manifest".into());
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
                assert_eq!(
                    matrix,
                    read_matrix_with_codec(&path, 1 << 17, decoder, None)?
                );
                assert!(
                    read_matrix_with_codec(&path, 1 << 17, decoder, Some((3, 1 << 17))).is_err()
                );
            }
        }
        fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    #[ignore = "480 MiB fixed-matrix handoff, five alternating disk/direct pairs"]
    fn fixed_matrix_handoff_benchmark() -> Result<()> {
        use std::time::Instant;

        const ROWS: usize = 1 << 20;
        const WIDTH: usize = 15;

        fn generate() -> RowMajorMatrix<Scalar> {
            let mut values = Scalar::zero_vec(ROWS * WIDTH);
            let rows_per_thread =
                ROWS.div_ceil(std::thread::available_parallelism().unwrap().get());
            std::thread::scope(|scope| {
                for (block, values) in values.chunks_mut(rows_per_thread * WIDTH).enumerate() {
                    scope.spawn(move || {
                        for (offset, values) in values.chunks_exact_mut(WIDTH).enumerate() {
                            let row = block * rows_per_thread + offset;
                            values[0] = Scalar::ONE;
                            values[1] = -Scalar::from_usize(row % 2);
                            values[2] = Scalar::from_usize(row % 8);
                            values[3] = -Scalar::from_usize(row % 4);
                            values[4] = Scalar::from_usize(row % 3);
                            for column in 0..3 {
                                values[5 + column] = Scalar::from_usize(row * 3 + column);
                                values[8 + column] =
                                    Scalar::from_usize(((row * 7919 + 17) % ROWS) * 3 + column);
                            }
                            if row.is_multiple_of(4096) {
                                values[11] = Scalar::ONE;
                                values[12] = Scalar::from_usize(row / 4096);
                            }
                            if row.is_multiple_of(64) {
                                values[13] = Scalar::ONE;
                                values[14] = Scalar::from_usize(row % 16384);
                            }
                        }
                    });
                }
            });
            RowMajorMatrix::new(values, WIDTH)
        }

        struct HashWriter(blake3::Hasher);
        impl Write for HashWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.0.update(bytes);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let directory =
            std::env::temp_dir().join(format!("kzg-fixed-matrix-handoff-{}", std::process::id()));
        fs::create_dir(&directory)?;
        let codec = trace_codec()?;
        let mut reference = None;
        for iteration in 0..5 {
            let path = directory.join(format!("{iteration}.fixed.zst"));
            let mut matrices: [Option<RowMajorMatrix<Scalar>>; 2] = [None, None];
            let mut timings = [0.0; 2];
            let mut generation = [0.0; 2];
            let mut write_seconds = 0.0;
            let mut read_seconds = 0.0;
            let order = if iteration % 2 == 0 { [0, 1] } else { [1, 0] };
            for leg in order {
                let started = Instant::now();
                let matrix = generate();
                generation[leg] = started.elapsed().as_secs_f64();
                matrices[leg] = Some(if leg == 0 {
                    let write_started = Instant::now();
                    write_matrix_with_codec(&path, &matrix, codec)?;
                    write_seconds = write_started.elapsed().as_secs_f64();
                    drop(matrix);
                    let read_started = Instant::now();
                    let restored = read_matrix_with_codec(&path, ROWS, codec, Some((WIDTH, ROWS)))?;
                    read_seconds = read_started.elapsed().as_secs_f64();
                    restored
                } else {
                    matrix
                });
                timings[leg] = started.elapsed().as_secs_f64();
            }
            assert_eq!(matrices[0], matrices[1]);
            let mut hash = HashWriter(blake3::Hasher::new());
            encoding::write_scalars(&mut hash, &matrices[0].as_ref().unwrap().values)?;
            let checksum = hash.0.finalize().to_hex().to_string();
            if let Some(expected) = &reference {
                assert_eq!(&checksum, expected);
            } else {
                reference = Some(checksum.clone());
            }
            println!(
                "fixed_input_handoff_pair={}",
                serde_json::json!({
                    "iteration": iteration, "order": order, "rows": ROWS, "width": WIDTH,
                    "field_bytes": ROWS * WIDTH * size_of::<Scalar>(), "codec": codec,
                    "disk_seconds": timings[0], "direct_seconds": timings[1],
                    "generation_seconds": generation, "write_seconds": write_seconds,
                    "read_seconds": read_seconds, "compressed_bytes": fs::metadata(&path)?.len(),
                    "checksum": checksum, "complete_matrix_parity": true,
                    "filesystem_cache_state": "uncontrolled; same-process write then read",
                    "scope": "synthetic fixed matrix; generation and handoff only; no commitment or proof"
                })
            );
            fs::remove_file(path)?;
        }
        fs::remove_dir(directory)?;
        Ok(())
    }
}
