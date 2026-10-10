#!/usr/bin/env python3
"""Collect already verified worker artifacts after an optional-file copy failure."""

import hashlib
import json
from pathlib import Path
import shutil


archive = Path(__file__).resolve().parent
record = archive / "production-first-stage"
report = json.loads((record / "report-before-archive-recovery.json").read_text())
output = Path(report["output"])
files = ("proof.compact.bin", "packet.bin", "profile-id.bin")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


assert report["status"] == "failed" and "SECURITY.txt" in report["error"]
assert report["worker_shutdown"]["exit_code"] == 0
assert report["independent_cpu_verify"]["exit_code"] == 0
verified = report["independent_cpu_verify"]["report"]
assert verified["native_verification_passed"] and verified["negative_tests_pass"]
assert verified["development_srs"] and verified["setup"] == "development"
copied = {}
for name in ("preparation", "fresh"):
    phase = report["phases"][name]
    assert phase["exit_code"] == 0 and phase["worker_response"]["idle"]["quiesced"]
    assert phase["worker_response"]["loaded_key_reused"] == (name == "fresh")
    assert phase["worker_response"]["frontend_reused"] == (name == "fresh")
    source = output / name
    target = record / "artifacts" / name
    (target / "kzg").mkdir(parents=True, exist_ok=True)
    for filename in files:
        path = source / "kzg" / filename
        assert {"sha256": digest(path), "bytes": path.stat().st_size} == report["artifacts"][name][filename]
    metadata = [*source.glob("*.meta"), source / "manifest.bin", source / "plan-id.bin",
                source / "inner-vk.bin", source / "kzg-setup-id.bin", *source.glob("SECURITY.txt")]
    paths = [(path, target / path.name) for path in metadata]
    for pattern in ("setup-*.bin", "*-report.json", *files):
        paths.extend((path, target / "kzg" / path.name) for path in (source / "kzg").glob(pattern))
    for path, destination in paths:
        if destination.exists():
            assert digest(destination) == digest(path)
        else:
            shutil.copyfile(path, destination)
        assert digest(destination) == digest(path)
        copied[str(destination.relative_to(archive))] = digest(destination)
assert report["artifacts"]["preparation"] == report["artifacts"]["fresh"]
report["artifact_collection_recovery"] = {
    "original_error": report.pop("error"),
    "original_process_exit_code": 1,
    "proofs_rerun": False,
    "verification_rerun": False,
    "copied_file_sha256": copied,
}
report["status"] = "verified_retained_first_stage_with_gpu_idle_and_legacy_artifact_parity"
(record / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "copied_files": len(copied),
                  "fresh_request_seconds": report["phases"]["fresh"]["wall_seconds"]}, indent=2))
