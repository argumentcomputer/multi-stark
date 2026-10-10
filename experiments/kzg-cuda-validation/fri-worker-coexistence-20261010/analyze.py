#!/usr/bin/env python3
"""Validate archived coexistence evidence without executing a proving process."""

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys


sys.dont_write_bytecode = True
archive = Path(__file__).resolve().parent
repo = archive.parents[2]
run = archive / "coexistence"
report = json.loads((run / "report.json").read_text())
spec = importlib.util.spec_from_file_location("bench", run / "driver/kzg-cuda-bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


source_hashes = json.loads((archive / "source-sha256.json").read_text())
for path, expected in source_hashes.items():
    require(digest(archive / "source" / path) == expected, f"source snapshot changed: {path}")
for value in report["provenance"].values():
    require(digest(Path(value["snapshot"])) == value["sha256"], "provenance snapshot changed")
require(report["status"] == "verified_fri_with_two_quiescent_small_kzg_contexts", "run did not pass")
require(not report["errors"] and report["all_fixture_processes_reaped"], "incomplete cleanup")
require(report["fri"]["exact_proof_key_claims_parity"], "FRI artifact parity failed")
require(report["fri"]["native_verification_and_altered_claim_passed"], "FRI verification failed")
for name, expected in report["expected_fri_sha256"].items():
    require(digest(run / "compressed-fri" / name) == expected, f"FRI artifact changed: {name}")

fixtures = []
for index, fixture in enumerate(report["fixtures"]):
    require(fixture["exit_code"] == 0, "fixture failed")
    for status in ("ready", "checked"):
        event = fixture[status]
        require(event["verified"] and event["parity"], "fixture verification failed")
        bench.validate_worker_idle(event["idle"], [0, 1, 2, 3])
    for key in ("opening_bytes", "opening_blake3"):
        require(fixture["ready"][key] == fixture["checked"][key], "opening parity failed")
    log = (run / f"idle-{index}/fixture.log").read_bytes()
    require(fixture["ready_log_end_byte"] == fixture["check_log_start_byte"],
            "fixture emitted work logs while idle")
    check_path = archive / f"idle-{index}-check.log"
    check_path.write_bytes(log[fixture["check_log_start_byte"]:fixture["checked_log_end_byte"]])
    _, checked = bench.summarize_cuda_events(check_path, with_quality=True)
    for parsed in (fixture["cuda_profile_parse"], checked):
        require(parsed["records_skipped"] == 0 and parsed["records_parsed"] > 0,
                "missing or malformed CUDA evidence")
        require(set(parsed["devices"]) == {"0", "1", "2", "3"}, "missing selected device")
        require(all(device["kernel_ms"] > 0 for device in parsed["devices"].values()),
                "selected device had no compute events")
    fixtures.append({"ready_cuda_profile_parse": fixture["cuda_profile_parse"],
                     "checked_cuda_profile_parse": checked,
                     "opening_bytes": fixture["ready"]["opening_bytes"],
                     "opening_blake3": fixture["ready"]["opening_blake3"],
                     "no_log_output_while_idle": True})


def memory_snapshot(name):
    rows = list(csv.reader(io.StringIO(report["memory_snapshots"][name]["devices"])))
    return {row[0].strip(): int(row[4].strip().split()[0]) for row in rows[1:]}


final_memory = memory_snapshot("after_process_cleanup")
require(final_memory == {str(device): 0 for device in range(4)}, "GPU memory remains allocated")
result = {
    "status": "verified_archived_evidence",
    "source_snapshot_count": len(source_hashes),
    "source_snapshots_and_provenance_hashes_match": True,
    "fri_wall_seconds": report["fri"]["wall_seconds"],
    "fri_peak_rss_gib": report["fri"]["peak_rss_bytes"] / (1 << 30),
    "fixture_gpu_memory_mib": {name: memory_snapshot(name) for name in
                               ("after_idle_0_ready", "after_idle_1_ready", "after_process_cleanup")},
    "production_first_stage_idle_memory_mib_historical": [721, 723],
    "production_retained_host_keys_tested": False,
    "same_binary_isolated_reference_measured": False,
    "full_chain_measured": False,
    "fixtures": fixtures,
}
(archive / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
