#!/usr/bin/env python3
"""Measure the shared retained-worker pipeline with explicit development v4 parameters."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


archive = Path(__file__).resolve().parent
root = archive.parents[2]
output = Path("/opt/dlami/nvme/multi-stark-kzg-integrated-v4-20261010")
assert not output.exists(), output
assert not any(os.environ.get(key) for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS"))
environment = {key: value for key, value in os.environ.items() if not key.startswith("MULTI_STARK_")}
environment.update({
    "MULTI_STARK_KZG_BACKEND": "cuda",
    "MULTI_STARK_KZG_CUDA_DEVICES": "0,1,2,3",
    "MULTI_STARK_KZG_CUDA_PROFILE": "1",
    "MULTI_STARK_KZG_CUDA_SRS_CACHE": "1",
    "MULTI_STARK_KZG_PREFETCH_GIB": "32",
    "MULTI_STARK_CUDA_DEVICE": "0",
    "MULTI_STARK_CUDA_AUX_DEVICES": "",
    "MULTI_STARK_CUDA_MEMORY_LOG": "1",
    "RUST_LOG": "info,multi_stark::verifier=debug",
    "TZ": "UTC",
})
command = [
    sys.executable, str(root / "experiments/kzg-cuda-bench.py"),
    "--output", str(output), "--setup", "development", "--development-public-degree",
    "--retained-workers", "--distributed-wrapper", "--kzg-devices", "0,1,2,3",
    "--root-artifacts", str(archive.parent / "fri-lookup-materialization-20261009/root-artifacts"),
    "--worker-seed-fri", str(archive.parent / "first-stage-fused-20261009/compressed-fri"),
    "--worker-seed-claims", str(archive / "root-claims.bin"),
    "--srs-cache", "/opt/dlami/nvme/multi-stark-kzg-perf-20261009/dev-srs-cache",
    "--fixed-cache", str(output / "fixed-cache"),
    "--first-stage-binary", str(archive / "bin/init_fri_kzg_prove"),
    "--wrapper-binary", str(archive / "bin/init-kzg-wrap"),
    "--fri-binary", str(archive / "bin/ix_root"),
]


def gpu_snapshot():
    return {
        "devices": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv"], text=True),
        "processes": subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
            "--format=csv"], text=True),
    }


def process_tree(pid):
    pending, seen, records = [pid], set(), []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        proc = Path(f"/proc/{current}")
        try:
            pending.extend(map(int, (proc / f"task/{current}/children").read_text().split()))
            status = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines())
            records.append({
                "pid": current, "name": status["Name"].strip(),
                "rss_bytes": int(status.get("VmRSS", "0 kB").split()[0]) * 1024,
                "lifetime_peak_rss_bytes": int(status.get("VmHWM", "0 kB").split()[0]) * 1024,
            })
        except (OSError, ValueError, KeyError):
            continue
    return records


before = gpu_snapshot()
assert len(before["processes"].splitlines()) == 1, before
record = {
    "command": command, "output": str(output), "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "gpu_before": before, "logical_cpus": os.cpu_count(),
    "environment": {key: value for key, value in environment.items()
                    if key.startswith(("MULTI_STARK_", "RAYON_", "OMP_"))
                    or key in ("RUST_LOG", "TZ", "CUDA_VISIBLE_DEVICES")},
    "known_trapdoor": True, "filecoin_acceptance": False,
    "memory_scope": "Sampled sum of descendant process RSS every two seconds; shared pages may be counted more than once",
}
started = time.monotonic()
process = None
try:
    process = subprocess.Popen(command, cwd=root, env=environment)
    with (archive / "host-memory.jsonl").open("x") as samples:
        while process.poll() is None:
            processes = process_tree(process.pid)
            sample = {"unix_seconds": time.time(), "processes": processes,
                      "sum_rss_bytes": sum(item["rss_bytes"] for item in processes)}
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    record["exit_code"] = process.returncode
    if process.returncode == 0:
        pairs = []
        for name in ("outer-vk.bin", "outer-proof.bin", "outer-claims.bin"):
            pairs.append((output / "compressed-fri" / name,
                          archive.parent / "first-stage-fused-20261009/compressed-fri" / name))
        pairs.append((output / "fri-to-kzg/kzg/proof.compact.bin",
                      archive.parent / "recursive-v4-frontend-20261010/development-fixture/proof.compact.bin"))
        for name in ("proof.compact.bin", "packet.bin", "profile-id.bin"):
            pairs.append((output / "recursive/kzg" / name,
                          archive.parent / "recursive-v4-proving-20261010/outputs/request-1/kzg" / name))
            for stage in ("fri-to-kzg", "recursive"):
                pairs.append((output / stage / "kzg" / name, output / "worker-warmup" / stage / "kzg" / name))
        record["artifact_parity"] = []
        for actual, expected in pairs:
            actual_bytes, expected_bytes = actual.read_bytes(), expected.read_bytes()
            equal = actual_bytes == expected_bytes
            record["artifact_parity"].append({
                "actual": str(actual), "expected": str(expected), "equal": equal,
                "actual_sha256": hashlib.sha256(actual_bytes).hexdigest(),
                "expected_sha256": hashlib.sha256(expected_bytes).hexdigest(),
            })
        assert all(item["equal"] for item in record["artifact_parity"]), record["artifact_parity"]
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        process.wait(timeout=30)
    record["process_reaped"] = process is not None and process.poll() is not None
    record["wall_seconds"] = time.monotonic() - started
    record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    record["gpu_after"] = gpu_snapshot()
    (archive / "process-lifecycle.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({key: record[key] for key in ("exit_code", "wall_seconds", "process_reaped")}), flush=True)
assert len(record["gpu_after"]["processes"].splitlines()) == 1, record["gpu_after"]
raise SystemExit(record["exit_code"])
