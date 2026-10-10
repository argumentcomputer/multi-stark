#!/usr/bin/env python3
"""Measure a fresh first-stage request after genuine retained-worker preparation."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


archive = Path(__file__).resolve().parent
root = archive.parents[2]
spec = importlib.util.spec_from_file_location("kzg_bench", root / "experiments/kzg-cuda-bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
output = Path(sys.argv[1]).resolve()
output.mkdir()
record = archive / "production-first-stage"
record.mkdir()
binary = archive / "bin/init_fri_kzg_prove"
fixture = archive.parent / "first-stage-fused-20261009"
fri, claims, seed = bench.snapshot_worker_seed(
    fixture / "compressed-fri", fixture / "root-claims.bin", record / "seed")
environment = {key: value for key, value in os.environ.items()
               if not key.startswith("MULTI_STARK_")}
environment.update({
    "MULTI_STARK_KZG_SETUP": "development",
    "MULTI_STARK_KZG_BACKEND": "cuda",
    "MULTI_STARK_KZG_CUDA_DEVICES": "0,1,2,3",
    "MULTI_STARK_KZG_CUDA_PROFILE": "1",
    "MULTI_STARK_KZG_CUDA_SRS_CACHE": "1",
    "MULTI_STARK_KZG_PREFETCH_GIB": "32",
    "MULTI_STARK_KZG_DEV_SRS_CACHE": "/opt/dlami/nvme/multi-stark-kzg-perf-20261009/dev-srs-cache",
    "MULTI_STARK_KZG_FIXED_CACHE": str(output / "fixed-cache"),
    "TZ": "UTC",
})
report = {
    "status": "running",
    "scope": "Production-size first stage only; preserved compressed FRI, known-trapdoor development setup; fresh witness and proof per request. No FRI compression or recursive proof.",
    "latency_boundary": "Prepared first-stage worker request through native verification and GPU quiescence; startup and independent CPU verification separate.",
    "output": str(output),
    "binary_sha256": bench.sha256(binary),
    "harness_sha256": bench.sha256(root / "experiments/kzg-cuda-bench.py"),
    "capabilities": bench.read_capabilities(binary, "first_stage", environment),
    "seed": seed,
    "environment": bench.captured_environment(environment),
    "phases": {},
}


def save():
    (record / "report.json").write_text(json.dumps(report, indent=2) + "\n")


gpu_log = (record / "gpu-samples.csv").open("x")
monitor = subprocess.Popen([
    "nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits", "--loop-ms=1000",
], stdout=gpu_log, stderr=subprocess.STDOUT)
client = None
started = time.monotonic()
try:
    client = bench.WorkerClient(binary, "first_stage", record / "worker", environment, [0, 1, 2, 3])
    report["worker_startup"] = client.startup
    baseline = json.loads((fixture / "report.json").read_text())["expected_proof_artifacts"]
    for index, name in enumerate(("preparation", "fresh")):
        destination = output / name
        phase = client.request(name, fri, destination, claims, record / f"{name}.log", reused=index > 0)
        report["phases"][name] = phase
        proof = bench.proof_hashes(destination / "kzg")
        bench.require_hashes(proof, baseline, f"{name} historical first-stage artifact parity")
        verification = bench.read_verification_report(destination / "kzg", "prove")
        assert verification["native_verification_passed"] and verification["negative_tests_pass"]
        report.setdefault("artifacts", {})[name] = proof
        report.setdefault("native_verification", {})[name] = verification
        if index == 0:
            report["preparation_wall_seconds"] = time.monotonic() - started
        phase["idle_nvidia_smi"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()
        save()
    closing = time.monotonic()
    client.close(graceful=True)
    report["shutdown_seconds"] = time.monotonic() - closing
    report["worker_shutdown"] = client.shutdown
    cpu_environment = dict(environment, MULTI_STARK_KZG_BACKEND="cpu",
                           MULTI_STARK_INIT_EXPECTED_CLAIMS=str(claims))
    checking = time.monotonic()
    with (record / "cpu-verify.log").open("x") as log:
        checked = subprocess.run([str(binary), "verify", str(output / "fresh")],
                                 cwd=root, env=cpu_environment, stdout=log, stderr=subprocess.STDOUT)
    report["independent_cpu_verify"] = {"seconds": time.monotonic() - checking,
                                        "exit_code": checked.returncode}
    checked.check_returncode()
    report["independent_cpu_verify"]["report"] = bench.read_verification_report(output / "fresh/kzg", "verify")
    bench.require_hashes(bench.proof_hashes(output / "fresh/kzg"), baseline, "CPU verification artifact parity")
    for name in ("preparation", "fresh"):
        source = output / name
        target = record / "artifacts" / name
        (target / "kzg").mkdir(parents=True)
        metadata = [*source.glob("*.meta"), source / "manifest.bin", source / "plan-id.bin",
                    source / "inner-vk.bin", source / "kzg-setup-id.bin", *source.glob("SECURITY.txt")]
        for path in metadata:
            shutil.copyfile(path, target / path.name)
        for pattern in ("setup-*.bin", "*-report.json", *bench.PROOF_FILES):
            for path in (source / "kzg").glob(pattern):
                shutil.copyfile(path, target / "kzg" / path.name)
    report["status"] = "verified_retained_first_stage_with_gpu_idle_and_legacy_artifact_parity"
except BaseException as error:
    report["status"] = "failed"
    report["error"] = str(error)
    if client is not None and client.last_request is not None:
        report["last_request"] = client.last_request
    raise
finally:
    if client is not None:
        client.close(graceful=False)
    monitor.terminate()
    try:
        monitor.wait(timeout=5)
    except subprocess.TimeoutExpired:
        monitor.kill()
        monitor.wait()
    gpu_log.close()
    report["gpu_samples_every_1_second"] = bench.summarize_gpu_samples(record / "gpu-samples.csv")
    report["total_wall_seconds"] = time.monotonic() - started
    save()
print(json.dumps({key: report[key] for key in ("status", "preparation_wall_seconds", "total_wall_seconds")}, indent=2))
print(json.dumps({"fresh_request_seconds": report["phases"]["fresh"]["wall_seconds"]}, indent=2))
