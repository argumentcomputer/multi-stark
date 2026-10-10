#[path = "../../../examples/support/binary_capabilities.rs"]
mod binary_capabilities;
mod count;
#[path = "../../../examples/support/init_claim.rs"]
mod init_claim;
#[path = "../../../examples/support/kzg_worker_protocol.rs"]
mod kzg_worker_protocol;
mod native_curve;
mod native_field;
mod native_msm;
mod native_transcript;
mod native_verifier;
mod outer;
mod saved;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "capabilities" {
        binary_capabilities::print("init-kzg-wrap");
        return Ok(());
    }
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing_subscriber::filter::LevelFilter::INFO)
        .init();
    if args.first().is_some_and(|s| s == "dev-v4-bootstrap") {
        if args.len() != 4 {
            return Err("usage: init-kzg-wrap dev-v4-bootstrap <legacy-stage-dir> <development-srs-cache> <new-output-dir>".into());
        }
        return saved::development_bootstrap(
            std::path::Path::new(&args[1]),
            std::path::Path::new(&args[2]),
            std::path::Path::new(&args[3]),
        );
    }
    if args.first().is_some_and(|s| s == "dev-v4-frontend-bench") {
        if args.len() != 3 {
            return Err("usage: init-kzg-wrap dev-v4-frontend-bench <saved-development-v4-dir> <new-output-dir>".into());
        }
        return saved::development_frontend_bench(
            std::path::Path::new(&args[1]),
            std::path::Path::new(&args[2]),
        );
    }
    if args.first().is_some_and(|s| s == "dev-v4-prove-bench") {
        if args.len() != 4 {
            return Err("usage: init-kzg-wrap dev-v4-prove-bench <saved-development-v4-dir> <development-srs-cache> <new-output-dir>".into());
        }
        return saved::development_prove_bench(
            std::path::Path::new(&args[1]),
            std::path::Path::new(&args[2]),
            std::path::Path::new(&args[3]),
        );
    }
    if args.first().is_some_and(|s| s == "dev-v4-verify") {
        if args.len() != 2 {
            return Err(
                "usage: init-kzg-wrap dev-v4-verify <saved-development-v4-outer-dir>".into(),
            );
        }
        return saved::development_verify(std::path::Path::new(&args[1]));
    }
    if args.first().is_some_and(|s| s == "dev-v4-serve") {
        if args.len() != 2 {
            return Err("usage: init-kzg-wrap dev-v4-serve <new-response-jsonl-path>".into());
        }
        return saved::development_serve(std::path::Path::new(&args[1]));
    }
    if args.first().is_some_and(|s| s == "serve") {
        if args.len() != 2 {
            return Err("usage: init-kzg-wrap serve <new-response-jsonl-path>".into());
        }
        return saved::serve(std::path::Path::new(&args[1]));
    }
    if args.first().is_some_and(|s| s == "stage-and-prove-many") {
        return saved::run_many(&args[1..]);
    }
    if args.first().is_some_and(|s| s == "count-saved") {
        if args.len() != 3 {
            return Err(
                "usage: init-kzg-wrap count-saved <saved-stage-one-dir> <report.json>".into(),
            );
        }
        return saved::count_saved(
            std::path::Path::new(&args[1]),
            std::path::Path::new(&args[2]),
        );
    }
    if args.first().is_some_and(|s| s == "count") {
        if args.len() != 3 {
            return Err("usage: init-kzg-wrap count <saved-artifact-dir> <report.json>".into());
        }
        return count::run(
            std::path::Path::new(&args[1]),
            std::path::Path::new(&args[2]),
        );
    }
    if args
        .first()
        .is_some_and(|s| matches!(s.as_str(), "native-verify" | "stage" | "stage-and-prove"))
    {
        if args.len() != 3 {
            return Err("usage: init-kzg-wrap <native-verify|stage|stage-and-prove> <saved-artifact-dir> <report-or-output-dir>".into());
        }
        let dir = std::path::Path::new(&args[2]);
        if args[0] != "native-verify" {
            std::fs::create_dir_all(dir)?;
            return saved::check(
                std::path::Path::new(&args[1]),
                &dir.join("circuit-report.json"),
                Some(dir),
                args[0] == "stage-and-prove",
            );
        }
        return saved::check(std::path::Path::new(&args[1]), dir, None, false);
    }
    if args.first().is_some_and(|s| s == "prove" || s == "verify") {
        if args.len() != 2 {
            return Err("usage: init-kzg-wrap <prove|verify> <staged-dir>".into());
        }
        std::thread::spawn(|| {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(30));
                if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                    eprintln!(
                        "{}",
                        status
                            .lines()
                            .filter(|l| l.starts_with("VmRSS:") || l.starts_with("VmHWM:"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                }
            }
        });
        return outer::prove(std::path::Path::new(&args[1]), args[0] == "verify");
    }
    if let Some(operation) = args.first().filter(|_| args.len() == 1)
        && matches!(
            operation.as_str(),
            "native-add" | "native-double" | "native-subgroup"
        )
    {
        println!("{}", native_curve::measure(operation)?);
        return Ok(());
    }
    Err("usage: init-kzg-wrap <count|count-saved|stage|stage-and-prove|stage-and-prove-many|serve|native-verify|prove|verify> ...".into())
}
