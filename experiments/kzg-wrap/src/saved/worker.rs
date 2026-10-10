use super::*;
use crate::{
    kzg_worker_protocol::{Protocol, ensure_empty_output},
    native_verifier::Bindings,
    outer::{Frontend, setup::SetupSource},
};
use std::{io::BufRead, path::PathBuf};

struct Worker {
    bindings: Bindings,
    frontend: Frontend,
}

impl Worker {
    fn compile(input: &Input) -> Result<Self> {
        let plan = input.plan(1 << 27)?;
        let compiled = plan.compile()?;
        let frontend = Frontend::new(
            compiled.circuit,
            compiled.bindings.pairing_keys(),
            compiled.bindings.degree_output_count(),
            &compiled.bindings.identity(),
            1 << 27,
        )?;
        Ok(Self {
            bindings: compiled.bindings,
            frontend,
        })
    }

    fn prove(
        &mut self,
        input: &Input,
        output: &Path,
        setup: &SetupSource,
        expected: &[u64; 18],
    ) -> Result<()> {
        ensure_empty_output(output)?;
        let start = Instant::now();
        let candidate = input.plan(1 << 27)?;
        let loaded_key_reused = self.frontend.has_loaded_key();
        let (mut report, assignment) = self.bindings.check_and_assign(
            candidate.identity(),
            &input.proof,
            &input.claims,
            self.frontend.witness(),
        )?;
        let assignment_seconds = start.elapsed().as_secs_f64();
        self.frontend
            .stage(&assignment, output, true, setup, expected)?;
        report["outer_proof_generated"] = true.into();
        report["input_proof_bytes"] = input.proof_bytes.len().into();
        report["input_proof_blake3"] = blake3::hash(&input.proof_bytes).to_hex().to_string().into();
        report["frontend_identity"] = blake3::Hash::from_bytes(self.bindings.identity())
            .to_hex()
            .to_string()
            .into();
        report["load_and_native_verify_seconds"] = input.load_seconds.into();
        report["assignment_and_checks_seconds"] = assignment_seconds.into();
        report["prove_request_seconds"] = start.elapsed().as_secs_f64().into();
        report["loaded_key_reused"] = loaded_key_reused.into();
        report["development_srs"] = setup.is_development().into();
        report["known_trapdoor"] = setup.is_development().into();
        report["filecoin_acceptance"] = (!setup.is_development()).into();
        let public_setup_id = input
            .system
            .config
            .srs()
            .public_setup()
            .map(|public| blake3::Hash::from_bytes(public.id).to_hex().to_string());
        report["public_setup_id"] = serde_json::to_value(&public_setup_id)?;
        report["ceremony_id"] = serde_json::to_value(
            (!setup.is_diagnostic())
                .then_some(public_setup_id)
                .flatten(),
        )?;
        report["filecoin_manifest_digest"] = match setup {
            SetupSource::Filecoin { digest, .. } => blake3::Hash::from_bytes(*digest)
                .to_hex()
                .to_string()
                .into(),
            SetupSource::Development { .. } | SetupSource::KnownTrapdoorPublicDegree { .. } => {
                serde_json::Value::Null
            }
        };
        fs::write(
            output.join("circuit-report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        Ok(())
    }
}

pub(crate) fn serve(response_path: &Path) -> Result<()> {
    serve_with_setup(response_path, SetupSource::from_env)
}

pub(crate) fn development_serve(response_path: &Path) -> Result<()> {
    serve_with_setup(
        response_path,
        SetupSource::development_public_degree_from_env,
    )
}

fn serve_with_setup(
    response_path: &Path,
    load_setup: impl FnOnce() -> Result<SetupSource>,
) -> Result<()> {
    let mut protocol = Protocol::open(response_path, "init-kzg-wrap")?;
    let setup = match load_setup() {
        Ok(setup) => setup,
        Err(error) => {
            protocol.failed(None, &error.to_string())?;
            return Err(error);
        }
    };
    serve_requests(&mut protocol, &setup)
}

fn serve_requests<R: BufRead>(protocol: &mut Protocol<R>, setup: &SetupSource) -> Result<()> {
    let mut worker: Option<Worker> = None;
    loop {
        let request = match protocol.next_request() {
            Ok(Some(request)) => request,
            Ok(None) => return Ok(()),
            Err(error) => {
                protocol.failed(None, &error.to_string())?;
                return Err(error);
            }
        };
        let start = Instant::now();
        let result = (|| -> Result<_> {
            ensure_empty_output(&request.output)?;
            let expected = init_claim::expected_words_from_path(&request.expected_claims)?;
            let input = Input::load(&request.input, setup, &expected)?;
            let frontend_reused = worker.is_some();
            let mut preparation_seconds = 0.0;
            if worker.is_none() {
                let preparation = Instant::now();
                worker = Some(Worker::compile(&input)?);
                preparation_seconds = preparation.elapsed().as_secs_f64();
            }
            let worker = worker.as_mut().ok_or("missing recursive frontend")?;
            let loaded_key_reused = worker.frontend.has_loaded_key();
            worker.prove(&input, &request.output, setup, &expected)?;
            drop(input);
            let idle = serde_json::to_value(worker.frontend.release_device_memory())?;
            Ok((
                frontend_reused,
                loaded_key_reused,
                preparation_seconds,
                idle,
            ))
        })();
        match result {
            Ok((frontend_reused, loaded_key_reused, preparation_seconds, idle)) => {
                protocol.proved(
                    &request,
                    frontend_reused,
                    loaded_key_reused,
                    preparation_seconds,
                    start.elapsed().as_secs_f64(),
                    idle,
                )?;
            }
            Err(error) => {
                protocol.failed(Some(&request.id), &error.to_string())?;
                return Err(error);
            }
        }
    }
}

struct Request {
    input: PathBuf,
    output: PathBuf,
    expected: [u64; 18],
}

pub(crate) fn run_many(args: &[String]) -> Result<()> {
    if args.is_empty() || !args.len().is_multiple_of(3) {
        return Err(
            "usage: init-kzg-wrap stage-and-prove-many <input> <output> <expected-claims> [...]"
                .into(),
        );
    }
    let requests = args
        .chunks_exact(3)
        .map(|args| {
            Ok(Request {
                input: PathBuf::from(&args[0]),
                output: PathBuf::from(&args[1]),
                expected: init_claim::expected_words_from_path(Path::new(&args[2]))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let setup = SetupSource::from_env()?;
    let start = Instant::now();
    let first = &requests[0];
    let input = Input::load(&first.input, &setup, &first.expected)?;
    let mut worker = Worker::compile(&input)?;
    println!(
        "RECURSIVE WORKER FRONTEND READY: startup_seconds={:.6}",
        start.elapsed().as_secs_f64()
    );
    let request_start = Instant::now();
    worker.prove(&input, &first.output, &setup, &first.expected)?;
    println!(
        "RECURSIVE WORKER REQUEST 0: seconds={:.6} loaded_key_reused=false",
        request_start.elapsed().as_secs_f64()
    );
    drop(input);
    for (index, request) in requests.iter().enumerate().skip(1) {
        let request_start = Instant::now();
        let input = Input::load(&request.input, &setup, &request.expected)?;
        worker.prove(&input, &request.output, &setup, &request.expected)?;
        println!(
            "RECURSIVE WORKER REQUEST {index}: seconds={:.6} loaded_key_reused=true",
            request_start.elapsed().as_secs_f64()
        );
    }
    println!(
        "RECURSIVE WORKER COMPLETE: requests={} total_seconds={:.6}",
        requests.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn failed_request_preserves_existing_output_and_never_prepares_worker() -> Result<()> {
        let root = std::env::temp_dir().join(format!("kzg-serve-error-{}", std::process::id()));
        fs::create_dir(&root)?;
        let output = root.join("output");
        fs::create_dir(&output)?;
        fs::write(output.join("existing"), b"preserved")?;
        let response = root.join("responses.jsonl");
        let request = serde_json::json!({
            "id": "occupied-output",
            "input": root.join("missing-input"),
            "output": output,
            "expected_claims": root.join("missing-claims"),
        });
        let mut protocol = Protocol::with_reader(
            &response,
            "init-kzg-wrap",
            Cursor::new(format!("{request}\n")),
        )?;
        let error =
            serve_requests(&mut protocol, &SetupSource::Development { cache: None }).unwrap_err();
        assert!(error.to_string().contains("worker output must be empty"));
        assert_eq!(fs::read(output.join("existing"))?, b"preserved");
        let events: Vec<serde_json::Value> = fs::read_to_string(&response)?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        assert_eq!(events.len(), 3);
        assert_eq!(events[0]["status"], "listening");
        assert_eq!(events[1]["status"], "request_started");
        assert_eq!(events[2]["status"], "failed");
        assert_eq!(events[2]["id"], "occupied-output");
        assert!(events.iter().all(|event| event["prepared"] == false));
        drop(protocol);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn eof_without_request_does_not_compile_or_prepare() -> Result<()> {
        let path = std::env::temp_dir().join(format!("kzg-serve-eof-{}.jsonl", std::process::id()));
        let mut protocol = Protocol::with_reader(&path, "init-kzg-wrap", Cursor::new([]))?;
        serve_requests(&mut protocol, &SetupSource::Development { cache: None })?;
        let events: Vec<serde_json::Value> = fs::read_to_string(&path)?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["status"], "listening");
        assert_eq!(events[0]["prepared"], false);
        drop(protocol);
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn diagnostic_eof_does_not_load_parameters_or_prepare() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("kzg-diagnostic-serve-eof-{}", std::process::id()));
        fs::create_dir(&root)?;
        let response = root.join("responses.jsonl");
        let cache = root.join("unprovisioned-cache");
        let mut protocol = Protocol::with_reader(&response, "init-kzg-wrap", Cursor::new([]))?;
        serve_requests(
            &mut protocol,
            &SetupSource::KnownTrapdoorPublicDegree {
                cache: Some(cache.clone()),
            },
        )?;
        let events: Vec<serde_json::Value> = fs::read_to_string(&response)?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["status"], "listening");
        assert_eq!(events[0]["prepared"], false);
        assert!(!cache.exists());
        drop(protocol);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn setup_selection_failure_is_reported_before_reading_requests() -> Result<()> {
        let response = std::env::temp_dir().join(format!(
            "kzg-serve-setup-error-{}.jsonl",
            std::process::id()
        ));
        let error =
            serve_with_setup(&response, || Err("invalid diagnostic selection".into())).unwrap_err();
        assert_eq!(error.to_string(), "invalid diagnostic selection");
        let events: Vec<serde_json::Value> = fs::read_to_string(&response)?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["status"], "listening");
        assert_eq!(events[1]["status"], "failed");
        assert!(events[1]["id"].is_null());
        assert!(events.iter().all(|event| event["prepared"] == false));
        fs::remove_file(response)?;
        Ok(())
    }
}
