//! Measure canonical trace I/O and verify the decoded round trip.

#[cfg(feature = "kzg")]
#[allow(dead_code, unreachable_pub)]
#[path = "support/kzg_storage.rs"]
mod storage;

#[cfg(feature = "kzg")]
fn main() -> storage::Result<()> {
    use p3_matrix::Matrix;
    use std::{path::PathBuf, time::Instant};

    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(args.next().ok_or("expected input matrix")?);
    let output = PathBuf::from(args.next().ok_or("expected new output matrix")?);
    if args.next().is_some() || output.exists() {
        return Err("usage: kzg_trace_io_bench <input.zst> <new-output.zst>".into());
    }
    let started = Instant::now();
    let matrix = storage::read_matrix_bounded(&input, 1 << 29)?;
    let input_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    storage::write_matrix(&output, &matrix)?;
    let write_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let restored = storage::read_matrix_bounded(&output, 1 << 29)?;
    let read_seconds = started.elapsed().as_secs_f64();
    assert_eq!(restored, matrix);
    println!("height={} width={}", matrix.height(), matrix.width());
    println!("input_read_seconds={input_seconds:.6}");
    println!("write_seconds={write_seconds:.6}");
    println!("output_read_seconds={read_seconds:.6}");
    println!("canonical_payload_bytes={}", 16 + matrix.values.len() * 32);
    println!("compressed_bytes={}", std::fs::metadata(output)?.len());
    println!("decoded_matrices_equal=true");
    Ok(())
}

#[cfg(not(feature = "kzg"))]
fn main() {
    eprintln!("Enable kzg and parallel to benchmark KZG trace I/O.");
}
