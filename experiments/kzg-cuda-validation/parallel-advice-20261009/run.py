#!/usr/bin/env python3
"""Measure fresh scalar assignment on the preserved first-stage frontend."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


archive = Path(__file__).resolve().parent
root = archive.parents[2]
mode = sys.argv[1]
if mode not in ("reference", "optimized"):
    raise SystemExit("expected reference or optimized")
out = archive / mode
binary = out / "init_fri_kzg_prove"
fixture = archive.parent / "first-stage-fused-20261009"
inputs = fixture / "compressed-fri"
expected = fixture / "root-claims.bin"
paths = [inputs / name for name in ("outer-vk.bin", "outer-proof.bin", "outer-claims.bin")]
paths.append(expected)
input_hashes = {str(path.relative_to(root)): digest(path) for path in paths}
binary_hash = digest(binary)
command = [str(binary), "assignment-bench", str(inputs)]
environment = os.environ.copy()
environment.update(
    MULTI_STARK_INIT_EXPECTED_CLAIMS=str(expected),
    MULTI_STARK_KZG_SETUP="filecoin",
    MULTI_STARK_KZG_BACKEND="cpu",
    TZ="UTC",
)
started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
started = time.monotonic()
with (out / "run.log").open("x") as log:
    result = subprocess.run(
        ["/usr/bin/time", "-v", "-o", str(out / "time.txt"), *command],
        cwd=root,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
elapsed = time.monotonic() - started
report = {
    "mode": mode,
    "command": command,
    "exit_code": result.returncode,
    "started_at_utc": started_at,
    "process_seconds": elapsed,
    "binary_sha256": binary_hash,
    "input_sha256": input_hashes,
    "environment": {
        key: value
        for key, value in environment.items()
        if key.startswith(("MULTI_STARK_", "RAYON_", "OMP_")) or key == "TZ"
    },
    "logical_cpus": os.cpu_count(),
    "boundary": "load/verify input, source construction/assignment, translation census and IR emission, fresh scalar assignment and checks, cleanup; no lowering, SRS, proving or GPU",
}
lines = (out / "run.log").read_text().splitlines()
records = [line[len("ASSIGNMENT_BENCH "):] for line in lines if line.startswith("ASSIGNMENT_BENCH ")]
if result.returncode == 0:
    assert len(records) == 1, "missing or duplicated measurement"
    report["measurement"] = json.loads(records[0])
    report["recipe_evaluation_seconds"] = [float(value) for value in re.findall(r"evaluation_seconds=([0-9.]+)", "\n".join(lines))]
    report["schedule_certification"] = [
        {"chunks": int(chunks), "certified": certified == "true", "seconds": float(seconds)}
        for chunks, certified, seconds in re.findall(
            r"Goldilocks witness schedule checked chunks=(\d+) certified=(true|false) certificate_seconds=([0-9.]+)", "\n".join(lines))
    ]
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", (out / "time.txt").read_text())
    assert rss, "GNU time omitted RSS"
    report["peak_rss_bytes"] = int(rss.group(1)) * 1024
    assert digest(binary) == binary_hash, "binary changed during measurement"
    assert all(digest(root / path) == sha for path, sha in input_hashes.items()), "input changed during measurement"
    if mode == "optimized":
        certificates = report["schedule_certification"]
        assert len(certificates) == 1 and certificates[0]["certified"] and certificates[0]["chunks"] > 1, "parallel advice schedule did not activate"
        reference = json.loads((archive / "reference/measurement.json").read_text())
        assert reference["exit_code"] == 0, "reference measurement failed"
        assert reference["input_sha256"] == input_hashes, "reference input differs"
        same_fields = ("plan_id", "public_words", "source_stats", "scalar_stats")
        for field in same_fields:
            assert report["measurement"][field] == reference["measurement"][field], field + " differs"
        report["reference_equality_checked"] = ["input_sha256", *same_fields]
(out / "measurement.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(result.returncode)
