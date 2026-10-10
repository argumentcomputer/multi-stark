#!/usr/bin/env python3
"""Record device telemetry while one isolated validation command runs."""

import json
from pathlib import Path
import subprocess
import sys


archive = Path(__file__).resolve().parent
label, *command = sys.argv[1:]
if not command:
    raise SystemExit("usage: run_gpu.py <label> <command> [arguments...]")


def probe(arguments):
    result = subprocess.run(["nvidia-smi", *arguments], capture_output=True, text=True, check=True)
    return result.stdout


identity = probe(["--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv"])
before = probe(["--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv"])
with (archive / f"{label}-gpu.csv").open("x") as log:
    monitor = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used",
        "--format=csv", "--loop-ms=500",
    ], stdout=log, stderr=subprocess.STDOUT)
    try:
        result = subprocess.run([sys.executable, str(archive / "run.py"), label, *command])
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait(timeout=5)
after = probe(["--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv"])
metadata = {"device_identity": identity, "compute_processes_before": before,
            "compute_processes_after": after, "monitor_returncode": monitor.returncode,
            "monitor_reaped": monitor.poll() is not None, "command_returncode": result.returncode}
(archive / f"{label}-gpu.json").write_text(json.dumps(metadata, indent=2) + "\n")
raise SystemExit(result.returncode)
