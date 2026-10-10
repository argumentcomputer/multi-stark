//! Hold a small verified KZG fixture's CUDA contexts idle across another phase.
//! Host allocations model a 2^16 development fixture, not production worker keys.

#[path = "support/binary_capabilities.rs"]
mod binary_capabilities;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "capabilities" {
        binary_capabilities::print("kzg_cuda_idle");
        return Ok(());
    }
    if args.len() != 1 {
        return Err("usage: kzg_cuda_idle <create-new-response.jsonl> | capabilities".into());
    }
    let compiled = multi_stark::BuildCapabilities::compiled();
    if !compiled.kzg || !compiled.kzg_cuda || !compiled.parallel {
        return Err("kzg_cuda_idle requires --features parallel,kzg-cuda".into());
    }
    #[cfg(feature = "kzg-cuda")]
    return diagnostic::run(std::path::Path::new(&args[0]));
    #[cfg(not(feature = "kzg-cuda"))]
    Err("KZG CUDA is not compiled into this binary".into())
}

#[cfg(feature = "kzg-cuda")]
mod diagnostic {
    use super::Result;
    use ark_serialize::CanonicalSerialize;
    use multi_stark::{
        ark_adapter::{
            Blake3Transcript, KzgCommitment, KzgPcs, Radix2Coset, Scalar, Srs,
            pcs::{KzgIdleMemoryRelease, KzgProverData},
        },
        traits::{Field, OpenedValues, Pcs},
    };
    use p3_matrix::dense::RowMajorMatrix;
    use serde_json::{Value, json};
    use std::{
        fs::{File, OpenOptions},
        io::{self, BufRead, BufReader, BufWriter, Read, Write},
        path::Path,
        sync::Arc,
        time::Instant,
    };

    const LOG_N: usize = 16;
    const FORMAT: &str = "kzg-cuda-idle/v1";

    struct Fixture {
        pcs: KzgPcs,
        domain: Radix2Coset,
        commitment: KzgCommitment,
        data: KzgProverData,
        values: RowMajorMatrix<Scalar>,
    }

    impl Fixture {
        fn new(columns: usize) -> Self {
            let n = 1 << LOG_N;
            let pcs = KzgPcs::new(
                Arc::new(Srs::unsafe_dev_setup(n, b"idle-coexistence-fixture-v1")),
                4,
            );
            let domain = pcs.natural_domain_for_degree(n);
            let values = RowMajorMatrix::new(
                (0..n)
                    .flat_map(|row| {
                        (0..columns).map(move |column| {
                            let value = Scalar::from_u64(
                                (row as u64 * 7919 + 17) ^ (column as u64 * 65537),
                            );
                            value * value
                        })
                    })
                    .collect(),
                columns,
            );
            let (commitment, data) = pcs.commit(vec![(domain, values.clone())]);
            Self {
                pcs,
                domain,
                commitment,
                data,
                values,
            }
        }

        fn check(&self) -> Result<(OpenedValues<Scalar>, Vec<u8>)> {
            let evaluated = self
                .pcs
                .get_evaluations_on_domain(&self.data, 0, self.domain);
            if evaluated.values != self.values.values {
                return Err("rehydrated evaluations differ from the fixture".into());
            }
            let points = vec![Scalar::from_u64(23), Scalar::from_u64(29)];
            let (opened, proof) = self.pcs.open(
                vec![(&self.data, vec![points.clone()])],
                &mut Blake3Transcript::new(),
            );
            self.pcs
                .verify(
                    vec![(
                        self.commitment.clone(),
                        vec![(
                            self.domain,
                            points.into_iter().zip(opened[0][0].clone()).collect(),
                        )],
                    )],
                    &proof,
                    &mut Blake3Transcript::new(),
                )
                .map_err(|error| format!("KZG opening verification failed: {error:?}"))?;
            let mut bytes = Vec::new();
            proof.0.serialize_compressed(&mut bytes)?;
            Ok((opened, bytes))
        }

        fn idle(&mut self, devices: &[i32]) -> Result<KzgIdleMemoryRelease> {
            self.data.release_device_residency();
            let idle = KzgPcs::release_idle_device_memory();
            if !idle.cuda_enabled
                || !idle.initialized
                || !idle.quiesced
                || idle
                    .devices
                    .iter()
                    .map(|device| device.device)
                    .collect::<Vec<_>>()
                    != devices
                || idle.devices.iter().any(|device| {
                    device.after.resident_coefficient_bytes != 0
                        || device.after.srs_point_bytes != 0
                        || device.after.msm_workspace_bytes != 0
                })
            {
                return Err(format!("CUDA idle release is incomplete: {idle:?}").into());
            }
            Ok(idle)
        }
    }

    fn write(response: &mut BufWriter<File>, mut value: Value) -> Result<()> {
        io::stdout().flush()?;
        io::stderr().flush()?;
        unsafe extern "C" {
            fn fflush(stream: *mut std::ffi::c_void) -> std::ffi::c_int;
        }
        if unsafe { fflush(std::ptr::null_mut()) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        value["format"] = FORMAT.into();
        value["pid"] = std::process::id().into();
        serde_json::to_writer(&mut *response, &value)?;
        response.write_all(b"\n")?;
        response.flush()?;
        Ok(())
    }

    fn completed(
        response: &mut BufWriter<File>,
        check_index: usize,
        columns: usize,
        opening: &[u8],
        operation_seconds: f64,
        idle_seconds: f64,
        idle: KzgIdleMemoryRelease,
    ) -> Result<()> {
        write(
            response,
            json!({
                "status": if check_index == 0 { "ready" } else { "checked" },
                "check_index": check_index,
                "scope": "small_fixture_cuda_context_and_ntt_residency",
                "known_trapdoor": true,
                "log_n": LOG_N,
                "nonconstant_columns": columns,
                "verified": true,
                "parity": true,
                "opening_bytes": opening.len(),
                "opening_blake3": blake3::hash(opening).to_hex().to_string(),
                "operation_seconds": operation_seconds,
                "idle_seconds": idle_seconds,
                "idle": idle,
            }),
        )
    }

    fn serve(response: &mut BufWriter<File>, devices: &[i32]) -> Result<()> {
        let columns = devices.len().max(4);
        let started = Instant::now();
        let mut fixture = Fixture::new(columns);
        let expected = fixture.check()?;
        let operation_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let idle = fixture.idle(devices)?;
        completed(
            response,
            0,
            columns,
            &expected.1,
            operation_seconds,
            started.elapsed().as_secs_f64(),
            idle,
        )?;

        let mut input = BufReader::new(io::stdin());
        let mut check_index = 0usize;
        loop {
            let mut command = Vec::new();
            Read::take(&mut input, 7).read_until(b'\n', &mut command)?;
            if command.is_empty() {
                return Ok(());
            }
            if command != b"check\n" {
                return Err("expected exactly check followed by a newline, or EOF".into());
            }
            check_index = check_index.checked_add(1).ok_or("check counter overflow")?;
            let started = Instant::now();
            let actual = fixture.check()?;
            if actual != expected {
                return Err(
                    "opening values or canonical proof bytes changed after rehydration".into(),
                );
            }
            let operation_seconds = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let idle = fixture.idle(devices)?;
            completed(
                response,
                check_index,
                columns,
                &actual.1,
                operation_seconds,
                started.elapsed().as_secs_f64(),
                idle,
            )?;
        }
    }

    pub(super) fn run(path: &Path) -> Result<()> {
        if std::env::var("MULTI_STARK_KZG_BACKEND").as_deref() != Ok("cuda") {
            return Err("set MULTI_STARK_KZG_BACKEND=cuda explicitly".into());
        }
        let devices: Vec<i32> = std::env::var("MULTI_STARK_KZG_CUDA_DEVICES")?
            .split(',')
            .map(|value| value.trim().parse())
            .collect::<std::result::Result<_, _>>()?;
        if devices.is_empty()
            || devices
                .iter()
                .enumerate()
                .any(|(index, device)| *device < 0 || devices[..index].contains(device))
        {
            return Err(
                "CUDA devices must be an explicit list of distinct nonnegative ordinals".into(),
            );
        }
        let output = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut response = BufWriter::new(output);
        let result = serve(&mut response, &devices);
        if let Err(error) = &result {
            let _ = write(
                &mut response,
                json!({"status": "failed", "error": error.to_string()}),
            );
        }
        result
    }
}
