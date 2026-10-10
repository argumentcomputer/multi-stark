#!/usr/bin/env python3
"""Check direct fixed-input evidence without extrapolating full-pipeline timing."""

import difflib
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics


archive = Path(__file__).resolve().parent


def load(name):
    return json.loads((archive / name).read_text())


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


before = load("before/source-sha256.json")
after = load("source-sha256.json")
assert before.keys() == after.keys() and len(after) == 186
changed, patch = [], []
for name, expected in after.items():
    old, new = archive / "before/source" / name, archive / "source" / name
    assert digest(old) == before[name] and digest(new) == expected
    if before[name] != expected:
        changed.append(name)
        patch.extend(difflib.unified_diff(old.read_text().splitlines(keepends=True),
                                         new.read_text().splitlines(keepends=True),
                                         fromfile=f"a/{name}", tofile=f"b/{name}"))
assert len(changed) == 6
assert (archive / "source-delta.patch").read_text() == "".join(patch)


def run(label):
    item = load(f"{label}/report.json")
    assert item["exit_code"] == 0 and item["source_hashes_unchanged"], label
    assert item["logical_cpus"] == 96
    assert not {"RAYON_NUM_THREADS", "OMP_NUM_THREADS", "RUST_TEST_THREADS",
                "CARGO_BUILD_JOBS"}.intersection(item["environment"])
    expected = after
    if item["command"][0] == "cargo":
        expected = {p: h for p, h in after.items() if not p.endswith(".py")}
    assert item["source_input_count"] == len(expected)
    assert item["source_inputs_sha256"] == manifest_digest(expected)
    return item


builds = {label: run(label) for label in (
    "first-test-build", "wrapper-test-build", "cuda-examples-build", "cuda-wrapper-build")}
tests = {label: run(label) for label in ("first-fixed-tests", "wrapper-fixed-tests")}
names = []
for label, count in (("first-fixed-tests", 5), ("wrapper-fixed-tests", 3)):
    log = (archive / label / "run.log").read_text()
    assert f"test result: ok. {count} passed; 0 failed;" in log
    current = re.findall(r"^test (\S+) \.\.\. ok$", log, re.MULTILINE)
    assert len(current) == count
    names.extend(current)
assert len(names) == len(set(names)) == 8
for name, metadata in load("test-binaries.json").items():
    assert digest(archive / "bin" / name) == metadata["sha256"]
for label, name in (("first-fixed-tests", "first-tests"), ("wrapper-fixed-tests", "wrapper-tests")):
    assert tests[label]["command"][0] == str(archive.relative_to(archive.parents[2]) / "bin" / name)
binaries = load("cuda-binaries.json")
for name, metadata in binaries.items():
    assert digest(archive / "bin" / name) == metadata["sha256"]
    assert all(metadata["capabilities"][key]
               for key in ("kzg", "kzg_cuda", "goldilocks_cuda", "parallel"))

measurement = run("fixed-handoff-benchmark")
assert measurement["command"][0] == tests["first-fixed-tests"]["command"][0]
started = datetime.fromisoformat(measurement["started_at_utc"]).timestamp()
for item in [*builds.values(), *tests.values()]:
    finished = datetime.fromisoformat(item["started_at_utc"]).timestamp() + item["wall_seconds"]
    assert finished < started
log = (archive / "fixed-handoff-benchmark/run.log").read_text()
assert "test result: ok. 1 passed; 0 failed;" in log
pairs = [json.loads(line.split("fixed_input_handoff_pair=", 1)[1])
         for line in log.splitlines() if "fixed_input_handoff_pair=" in line]
assert len(pairs) == 5
for i, pair in enumerate(pairs):
    assert pair["iteration"] == i and pair["order"] == ([0, 1] if i % 2 == 0 else [1, 0])
    assert (pair["rows"], pair["width"], pair["field_bytes"]) == (2**20, 15, 503316480)
    assert pair["complete_matrix_parity"] and len(pair["checksum"]) == 64
    assert pair["codec"] in ("zstd", "pzstd") and pair["compressed_bytes"] > 0
    assert pair["disk_seconds"] >= sum((pair["generation_seconds"][0],
                                       pair["write_seconds"], pair["read_seconds"]))
    assert pair["direct_seconds"] >= pair["generation_seconds"][1] > 0
assert len({pair["checksum"] for pair in pairs}) == 1
assert len({pair["codec"] for pair in pairs}) == 1
disk = statistics.median(pair["disk_seconds"] for pair in pairs)
direct = statistics.median(pair["direct_seconds"] for pair in pairs)
result = {
    "status": "verified_direct_fixed_input_handoff",
    "scope": "Cold fused preprocessing in both KZG stages; small complete-proof/cache parity and a synthetic fixed-matrix handoff comparison",
    "source_input_count": len(after), "source_inputs_sha256": manifest_digest(after),
    "changed_source_inputs": changed,
    "validation": {
        "distinct_correctness_tests_passed": len(names), "tests": sorted(names),
        "test_runs": tests, "builds": builds,
        "legacy_and_public_degree_worker_checks": True,
        "disk_direct_setup_coefficient_proof_packet_profile_parity": True,
        "direct_cache_publication_and_restore": True,
        "malformed_shape_and_stale_raw_file_checks": True,
    },
    "synthetic_handoff": {
        "measurement": measurement, "pairs": pairs,
        "median_disk_seconds": disk, "median_direct_seconds": direct,
        "median_seconds_removed": disk - direct, "handoff_speedup": disk / direct,
        "rows": 2**20, "width": 15, "field_bytes": 503316480,
        "generation_included_in_both_timers": True,
        "full_matrix_parity_and_canonical_checksum_outside_timers": True,
        "filesystem_cache_state": "uncontrolled; same-process write then read",
        "memory_scope": "Process peak includes both leg outputs retained for parity",
        "production_startup_speedup_measured": False,
    },
    "cuda_binaries": binaries,
    "production_proof_or_full_pipeline_rerun": False,
    "known_trapdoor_test_parameters": True,
    "filecoin_acceptance": False,
    "goal_complete": False,
}
(archive / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "distinct_tests_passed": len(names),
                  "median_disk_seconds": disk, "median_direct_seconds": direct,
                  "synthetic_handoff_speedup": disk / direct}, indent=2))
