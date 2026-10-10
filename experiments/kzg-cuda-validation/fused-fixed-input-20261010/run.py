#!/usr/bin/env python3
"""Record a targeted validation command and its resource usage."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


archive = Path(__file__).resolve().parent
root = archive.parents[2]
label, *command = sys.argv[1:]
if not command or not re.fullmatch(r"[a-z0-9-]+", label):
    raise SystemExit("usage: run.py <label> <command> [arguments...]")
directory = archive / label
directory.mkdir()
environment = os.environ.copy()
source_hashes = json.loads((archive / "source-sha256.json").read_text())
if command[0] == "cargo" or command[0].endswith("cuda-library-tests"):
    source_hashes = {path: digest for path, digest in source_hashes.items() if not path.endswith(".py")}


def verify_sources():
    for path, expected in source_hashes.items():
        with (root / path).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"source changed during validation: {path}")


verify_sources()
started_at = datetime.now(timezone.utc).isoformat()
started = time.monotonic()
with (directory / "run.log").open("x") as log:
    result = subprocess.run(
        ["/usr/bin/time", "-v", "-o", str(directory / "time.txt"), *command],
        cwd=root,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
report = {
    "command": command,
    "cwd": str(root),
    "started_at_utc": started_at,
    "wall_seconds": time.monotonic() - started,
    "exit_code": result.returncode,
    "logical_cpus": os.cpu_count(),
    "source_input_count": len(source_hashes),
    "source_inputs_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
    "environment": {
        key: value for key, value in environment.items()
        if key.startswith(("MULTI_STARK_", "RAYON_", "OMP_"))
        or key in ("CARGO_TARGET_DIR", "RUSTFLAGS", "CUDA_VISIBLE_DEVICES", "TZ", "TMPDIR",
                   "RUST_TEST_THREADS", "CARGO_BUILD_JOBS")
    },
}
rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", (directory / "time.txt").read_text())
if rss:
    report["peak_rss_bytes"] = int(rss.group(1)) * 1024
try:
    verify_sources()
    report["source_hashes_unchanged"] = True
except RuntimeError as error:
    report["source_hashes_unchanged"] = False
    report["source_error"] = str(error)
(directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
if not report["source_hashes_unchanged"]:
    raise SystemExit(1)
raise SystemExit(result.returncode)
