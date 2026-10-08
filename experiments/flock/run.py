#!/usr/bin/env python3
"""Build and measure the CPU experiments sequentially; keep large outputs in target."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from prepare import prepare

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "experiments/flock"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-artifacts", type=Path)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "target" / f"flock-{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RUSTFLAGS="-C target-cpu=native", RAYON_NUM_THREADS=str(args.threads))
    prepare()
    subprocess.run(["cargo", "build", "--locked", "--release", "--manifest-path",
                    str(PACKAGE / "Cargo.toml")], cwd=ROOT, env=env, check=True)
    exe = PACKAGE / "target/release/init-flock"
    jobs = [("field", ["field"]), ("chain-256", ["chain", "256", "slim"]),
            ("chain-4096", ["chain", "4096", "slim"]), ("fold", ["fold"]),
            ("fri", ["fri"])]
    if args.init_artifacts:
        jobs.append(("init", ["init-census", str(args.init_artifacts.resolve())]))
    sources = list(PACKAGE.glob("src/*.rs")) + list(PACKAGE.glob("configs/*.toml"))
    sources += [PACKAGE / name for name in ["Cargo.toml", "Cargo.lock", "prepare.py", "run.py", "flock-core.patch"]]
    sources += [ROOT / "src/plonkish/builder.rs"]
    results = {"threads": args.threads, "started_utc": stamp, "runs": {},
               "pipeline_complete": False,
               "missing_stages": ["full Init Flock proof", "complete Flock verifier in Groth16", "composed 128-bit security validation"],
               "binary_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
               "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(sources)}}
    for name, command in jobs:
        directory = output / name
        directory.mkdir(exist_ok=True)
        if name != "field":
            command += [str(directory)]
        if name == "fri":
            command += ["--prove"]
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            subprocess.run(["/usr/bin/time", "-f", "%e %M %x", "-o", str(directory / "time.txt"),
                            str(exe), *command], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)
        elapsed, rss, code = (directory / "time.txt").read_text().split()
        report_path = directory / ("stdout.log" if name == "field" else "report.json")
        results["runs"][name] = {"elapsed_seconds": float(elapsed), "peak_rss_bytes": int(rss) * 1024,
                                 "exit_code": int(code), "measurement": json.loads(report_path.read_text())}
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name}: {elapsed}s, {int(rss)/2**20:.3f} GiB peak RSS", flush=True)
    print(output / "results.json")


if __name__ == "__main__":
    main()
