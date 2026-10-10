#!/usr/bin/env python3
"""Measure one fused first stage with cold fixed preprocessing and a warm dev SRS."""

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUT = REPO / "experiments/kzg-cuda-validation/fri-lookup-materialization-20261009/primary/compressed-fri"
CLAIMS = REPO / "experiments/kzg-cuda-validation/resume-20261009/root-artifacts/root-claims.bin"
BASELINE = REPO / "experiments/kzg-cuda-validation/partition-pipeline-20261009/cached"
OUTPUT = Path("/opt/dlami/nvme/multi-stark-kzg-fused-stage1-20261009")
SRS = Path("/opt/dlami/nvme/multi-stark-kzg-perf-20261009/dev-srs-cache")
SRS_FILE = SRS / "50664d8c92de07b40c102be45cdb7c87837511bc767f775e0a24b06e841f3137.dev-srs"
BINARY = HERE / "init_fri_kzg_prove"

spec = importlib.util.spec_from_file_location("benchmark_helpers", HERE / "benchmark_helpers.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def now():
    return datetime.now(timezone.utc).isoformat()


def timer_totals(log):
    labels = [
        "Fixed trace staged", "KZG fixed trace loaded", "KZG fixed committed",
        "KZG fixed prepared", "KZG fresh trace generated", "KZG main committed",
        "KZG main prepared", "KZG proof computed",
    ]
    result = {}
    for label in labels:
        rows = []
        for line in log.splitlines():
            if line.startswith(label + " "):
                fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
                row = {"seconds": float(fields["seconds"])}
                if "circuit" in fields:
                    row["circuit"] = int(fields["circuit"])
                if "source" in fields:
                    row["source"] = fields["source"]
                rows.append(row)
        result[label] = {"calls": len(rows), "seconds_sum": sum(row["seconds"] for row in rows), "rows": rows}
    return result


def main():
    signal.signal(signal.SIGTERM, helpers.interrupt)
    signal.signal(signal.SIGINT, helpers.interrupt)
    if not BINARY.is_file():
        raise RuntimeError("immutable example binary has not been prepared")
    metadata = json.loads((HERE / "build-metadata.json").read_text())
    if helpers.sha256(BINARY) != metadata["binary_sha256"]:
        raise RuntimeError("immutable example binary digest differs")
    if SRS_FILE.stat().st_size != 1610618025:
        raise RuntimeError("expected populated 2^24 development SRS cache is missing or truncated")
    for variable in ("RAYON_NUM_THREADS", "RUST_TEST_THREADS", "CARGO_BUILD_JOBS"):
        if variable in os.environ:
            raise RuntimeError(f"unexpected thread override: {variable}")
    baseline_report = json.loads((BASELINE / "report.json").read_text())
    expected_input = baseline_report["baseline"]["fri_input"]
    actual_input = {name: helpers.sha256(INPUT / name) for name in expected_input}
    helpers.require_hashes(actual_input, expected_input, "compressed FRI input")
    expected_proof = helpers.proof_hashes(BASELINE / "fri-to-kzg/kzg")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "fixed-cache").mkdir()
    saved_input = HERE / "compressed-fri"
    saved_input.mkdir()
    for name in (
        "outer-vk.bin", "outer-proof.bin", "outer-claims.bin", "inner-vk.bin",
        "plan-id.bin", "build-id.bin", "statement-schema.txt",
    ):
        shutil.copyfile(INPUT / name, saved_input / name)
    shutil.copyfile(CLAIMS, HERE / "root-claims.bin")
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("MULTI_STARK_KZG_") or key == "MULTI_STARK_CUDA_AUX_DEVICES":
            env.pop(key)
    env.update(
        MULTI_STARK_KZG_SETUP="development",
        MULTI_STARK_KZG_BACKEND="cuda",
        MULTI_STARK_KZG_CUDA_DEVICES="0,1,2,3",
        MULTI_STARK_KZG_CUDA_PROFILE="1",
        MULTI_STARK_KZG_CUDA_SRS_CACHE="1",
        MULTI_STARK_KZG_PREFETCH_GIB="32",
        MULTI_STARK_KZG_DEV_SRS_CACHE=str(SRS),
        MULTI_STARK_KZG_FIXED_CACHE=str(OUTPUT / "fixed-cache"),
        MULTI_STARK_INIT_EXPECTED_CLAIMS=str(HERE / "root-claims.bin"),
        TZ="UTC",
    )
    report = {
        "status": "running", "recorded_at_utc": now(),
        "scope": "One fused first KZG stage; cold fixed preprocessing, warm known-trapdoor development SRS. No compressor or recursive stage.",
        "timing_boundary": "preserved compressed FRI to verified first-stage packet, including current fixed preprocessing",
        "independent_cpu_verification_outside_timing_boundary": True,
        "fixed_cache_state": "new empty current-executable cache; no imported legacy entries",
        "cached_speedup_claim": False,
        "environment": {key: value for key, value in env.items() if key.startswith("MULTI_STARK_") or key in ("CUDA_VISIBLE_DEVICES", "TZ")},
        "binary_sha256": helpers.sha256(BINARY),
        "source_archive_sha256": helpers.sha256(HERE / "source.tar.gz"),
        "runner_sha256": helpers.sha256(Path(__file__)),
        "helpers_sha256": helpers.sha256(HERE / "benchmark_helpers.py"),
        "input_hashes": actual_input,
        "expected_claims_sha256": helpers.sha256(HERE / "root-claims.bin"),
        "baseline_report_sha256": helpers.sha256(BASELINE / "report.json"),
        "baseline_cached_stage_plus_prove_seconds": sum(baseline_report["phases"][name]["wall_seconds"] for name in ("fri_stage", "fri_prove")),
        "expected_proof_artifacts": expected_proof,
        "output": str(OUTPUT), "phases": {},
        "logical_cpus": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "overlap_note": "Fixed and main prepared timers contain their load/commit timers. Fresh trace generation may overlap the prior circuit's preparation. Interval sums are not additive wall time.",
    }
    report["hardware"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv",
    ], text=True)

    def save():
        (HERE / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, command, cpu=False):
        log_path = HERE / (name + ".log")
        time_path = HERE / (name + ".time")
        started_at = now()
        started = time.monotonic()
        with log_path.open("w") as log:
            process = subprocess.Popen([
                "/usr/bin/time", "-f",
                '{"wall_seconds":%e,"peak_rss_kib":%M,"exit_code":%x}',
                "-o", str(time_path), *map(str, command),
            ], cwd=REPO, env=dict(env, MULTI_STARK_KZG_BACKEND="cpu") if cpu else env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print(f"{name}: {time.monotonic() - started:.0f}s; log: {log_path}", flush=True)
            except BaseException:
                helpers.terminate_process_group(process)
                report["phases"][name] = {
                    "wall_seconds": time.monotonic() - started, "exit_code": process.returncode,
                    "interrupted": True, "command": list(map(str, command)),
                }
                raise
        measured = json.loads(time_path.read_text().splitlines()[-1])
        measured.update(wall_seconds=time.monotonic() - started, started_at_utc=started_at,
                        finished_at_utc=now(), command=list(map(str, command)),
                        backend="cpu" if cpu else "cuda", log=str(log_path))
        measured["peak_rss_bytes"] = measured.pop("peak_rss_kib") * 1024
        log = log_path.read_text()
        measured.update(
            fixed_cache_hits=log.count("Reused fixed preprocessing:"),
            fixed_cache_misses=log.count("Fixed preprocessing cache miss:"),
            srs_cache_hits=log.count("Loaded development SRS cache"),
            srs_cache_misses=log.count("Generated and cached development SRS"),
            cuda_operation_totals_overlap=helpers.summarize_cuda_events(log_path),
            host_operation_totals_overlap=helpers.summarize_host_operations(log_path),
            preparation_interval_totals_overlap=timer_totals(log),
        )
        report["phases"][name] = measured
        save()
        if process.returncode:
            raise RuntimeError(f"{name} failed with exit code {process.returncode}")
        print(f"Finished {name}: {measured['wall_seconds']:.3f}s, {measured['peak_rss_bytes'] / 2**30:.2f} GiB RSS", flush=True)

    save()
    with (HERE / "gpu-samples.csv").open("w") as gpu_log:
        monitor = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits", "--loop=2",
        ], stdout=gpu_log, stderr=subprocess.DEVNULL, env=env, start_new_session=True)
        try:
            staged = OUTPUT / "fri-to-kzg"
            run("stage-and-prove", [BINARY, "stage-and-prove", saved_input, staged])
            primary = report["phases"]["stage-and-prove"]
            if (primary["fixed_cache_hits"], primary["fixed_cache_misses"], primary["srs_cache_hits"], primary["srs_cache_misses"]) != (0, 1, 1, 0):
                raise RuntimeError("run did not have the specified cold-fixed/warm-SRS cache state")
            proof = helpers.proof_hashes(staged / "kzg")
            helpers.require_hashes(proof, expected_proof, "legacy first-stage proof artifacts")
            report["proof_parity"] = proof
            artifact_dir = HERE / "proof-artifacts"
            artifact_dir.mkdir()
            for name in (*helpers.PROOF_FILES, "VERIFIED.txt", "SECURITY.txt"):
                shutil.copyfile(staged / "kzg" / name, artifact_dir / name)
            report["fused_output_checks"] = {
                "witness_files_absent": not any(staged.glob("*.witness.zst")),
                "main_checkpoints_absent": not any((staged / "kzg").glob("main-*.bin")),
            }
            if not all(report["fused_output_checks"].values()):
                raise RuntimeError("fused output unexpectedly contains witness/main checkpoints")
            run("cpu-verify", [BINARY, "verify", staged], cpu=True)
            helpers.require_hashes(helpers.proof_hashes(staged / "kzg"), expected_proof, "post-verification first-stage artifacts")
            shutil.copyfile(staged / "kzg/VERIFIED.txt", HERE / "CPU-VERIFIED.txt")
            report["status"] = "verified_legacy_stage1_artifact_parity_cold_fixed"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = repr(error)
            raise
        finally:
            helpers.terminate_process_group(monitor)
            gpu_log.flush()
            for measured in report["phases"].values():
                if "started_at_utc" in measured:
                    measured["gpu_samples_every_2_seconds"] = helpers.summarize_gpu_samples(
                        HERE / "gpu-samples.csv", datetime.fromisoformat(measured["started_at_utc"]),
                        datetime.fromisoformat(measured["finished_at_utc"]),
                    )
            report["finished_at_utc"] = now()
            save()


if __name__ == "__main__":
    main()
