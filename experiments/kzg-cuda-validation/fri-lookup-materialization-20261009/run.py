#!/usr/bin/env python3
"""Measure one FRI-compression leg against the same snapshotted binary/input."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("leg", choices=["primary", "auxiliary"])
args = parser.parse_args()
out = Path(__file__).resolve().parent
repo = out.parents[2]
root = Path("/opt/dlami/nvme/multi-stark-kzg-prefetch-20261009/cached/root-artifacts")
historical = root.parent / "compressed-fri"
baseline = out.parent / "fri-stream-order-20261009/primary/report.json"
binary = out / "ix_root"
snapshot = out / "root-artifacts"
if not binary.exists():
    shutil.copy2(repo / "target/release/examples/ix_root", binary)
    snapshot.mkdir()
    for name in ["root-vk.bin", "root-proof.bin", "root-claims.bin"]:
        shutil.copyfile(root / name, snapshot / name)
    metadata = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "binary_sha256": digest(binary),
        "root_sha256": {p.name: digest(p) for p in snapshot.iterdir()},
        "historical_fri_sha256": {name: digest(historical / name) for name in ["outer-proof.bin", "outer-vk.bin", "outer-claims.bin"]},
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "source_sha256": {
            name: digest(repo / name)
            for name in ["cuda/kernels.cu", "src/cuda/mod.rs", "src/cuda/mmcs.rs", "src/cuda/pcs.rs", "src/cuda/offload.rs", "src/types.rs", "src/verifier.rs", "src/system.rs", "src/lookup.rs", "src/p3_adapter/field.rs", "examples/ix_root.rs"]
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
metadata = json.loads((out / "metadata.json").read_text())
assert digest(binary) == metadata["binary_sha256"]
assert {p.name: digest(p) for p in snapshot.iterdir()} == metadata["root_sha256"]
leg = out / args.leg
leg.mkdir()
env = os.environ.copy()
env.update(
    MULTI_STARK_CUDA_DEVICE="0",
    MULTI_STARK_CUDA_AUX_DEVICES="" if args.leg == "primary" else "1,2,3",
    MULTI_STARK_CUDA_MEMORY_LOG="1",
    MULTI_STARK_INIT_EXPECTED_CLAIMS=str(snapshot / "root-claims.bin"),
    RUST_LOG="info,multi_stark::verifier=debug",
)
command = ["/usr/bin/time", "-v", "-o", str(leg / "time.txt"), str(binary), str(snapshot), str(leg / "compressed-fri"), "--prove-outer"]
report = {
    "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": command,
    "environment": {k: v for k, v in env.items() if k.startswith("MULTI_STARK_") or k in ["CUDA_VISIBLE_DEVICES", "RAYON_NUM_THREADS", "RUST_LOG"]},
    "gpu_before": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv"], text=True),
    "processes_before": subprocess.check_output(["ps", "-eo", "pid,pcpu,rss,comm", "--sort=-pcpu"], text=True).splitlines()[:12],
}
print(f"Starting {args.leg}: {command}", flush=True)
started = time.monotonic()
with (leg / "gpu-samples.csv").open("w") as samples, (leg / "fri_compress.log").open("w") as log:
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu", "--format=csv,noheader,nounits", "--loop=1"], stdout=samples, stderr=subprocess.DEVNULL, env=env)
    process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        report["exit_code"] = process.wait(timeout=420)
    except BaseException:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
report["wall_seconds"] = time.monotonic() - started
report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
text = (leg / "fri_compress.log").read_text()
report["phase_lines"] = [line for line in text.splitlines() if line.startswith(("Circuit build:", "Root verifier witness satisfied:", "FRI ", "VERIFIED:"))]
report["lookup_materialization_lines"] = [line for line in text.splitlines() if "Lookup expressions materialized" in line]
report["host_matrices"] = [dict(zip(["index", "height", "width", "lde_bytes", "work"], map(int, values))) for values in re.findall(r"stage1 host matrix (\d+): height=(\d+) width=(\d+) lde_bytes=(\d+) work=(\d+)", text)]
report["cuda_lines"] = [line for line in text.splitlines() if line.startswith("[multi-stark/cuda]")]
if report["exit_code"] == 0:
    assert "VERIFIED: FRI proof_bytes=" in text
    artifacts = {name: digest(leg / "compressed-fri" / name) for name in ["outer-proof.bin", "outer-vk.bin", "outer-claims.bin"]}
    report["artifacts_sha256"] = artifacts
    report["historical_parity"] = artifacts == metadata["historical_fri_sha256"]
    prior = json.loads(baseline.read_text())
    report["baseline_parity"] = artifacts == prior["artifacts_sha256"]
    report["baseline_wall_seconds"] = prior["wall_seconds"]
    report["baseline_delta_seconds"] = report["wall_seconds"] - prior["wall_seconds"]
    if args.leg == "auxiliary":
        primary = json.loads((out / "primary/report.json").read_text())
        report["primary_parity"] = artifacts == primary["artifacts_sha256"]
(leg / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k in ["exit_code", "wall_seconds", "phase_lines", "host_matrices", "historical_parity", "primary_parity", "baseline_parity", "baseline_wall_seconds", "baseline_delta_seconds", "lookup_materialization_lines"]}, indent=2), flush=True)
assert report["exit_code"] == 0, "FRI compression failed; inspect the saved log"
assert report["historical_parity"] and report["baseline_parity"], "FRI proof/VK/claims bytes changed"
if args.leg == "auxiliary":
    assert report["primary_parity"], "FRI proof/VK/claims bytes changed"
