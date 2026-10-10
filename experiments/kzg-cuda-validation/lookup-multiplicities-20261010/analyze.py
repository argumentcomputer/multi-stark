#!/usr/bin/env python3
"""Check and summarize the isolated lookup-multiplicity comparison."""

import difflib
import hashlib
import json
from pathlib import Path
import re
import statistics


archive = Path(__file__).resolve().parent


def load(path):
    return json.loads((archive / path).read_text())


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run(label):
    report = load(f"{label}/report.json")
    assert report["exit_code"] == 0 and report["source_hashes_unchanged"], label
    assert not any(key.startswith(("RAYON_", "OMP_")) for key in report["environment"])
    return report


def records(label, prefix):
    return [json.loads(line[len(prefix):])
            for line in (archive / label / "run.log").read_text().splitlines()
            if line.startswith(prefix)]


before = load("before/source-sha256.json")
after = load("source-sha256.json")
assert set(before) <= set(after)
changed, patch = [], []
for path, expected in after.items():
    current = archive / "source" / path
    assert digest(current) == expected, path
    previous = archive / "before/source" / path
    if path in before:
        assert digest(previous) == before[path], path
    if before.get(path) != expected:
        changed.append(path)
        patch.extend(difflib.unified_diff(
            previous.read_text().splitlines(keepends=True) if path in before else [],
            current.read_text().splitlines(keepends=True),
            fromfile=f"a/{path}" if path in before else "/dev/null", tofile=f"b/{path}"))
assert changed == ["src/plonkish/stark.rs", "src/plonkish/stark/multiplicities.rs",
                   "src/plonkish/stark/multiplicities/tests.rs"]
(archive / "source-delta.patch").write_text("".join(patch))

benchmark_run = run("operation-benchmark")
data = records("operation-benchmark", "MULTIPLICITY_BENCH ")
fixture, *samples = data
assert fixture["type"] == "fixture" and fixture["warmup_parity"]
assert fixture["lookups"] == 32_587_682 and fixture["real_table_rows"] == 65_824
assert fixture["jobs"] == fixture["default_pool_threads"] == 96
assert fixture["histogram_working_bytes"] == 50_566_696
assert fixture["histogram_working_bytes"] <= fixture["histogram_budget_bytes"] == 64 << 20
assert len(samples) == fixture["samples"] == 5
for index, sample in enumerate(samples):
    assert sample["type"] == "sample" and sample["sample"] == index
    assert sample["parity"] and sample["checksum_blake3"] == fixture["checksum_blake3"]
    assert sample["first"] == ("serial" if index % 2 == 0 else "automatic")
serial = statistics.median(sample["serial_seconds"] for sample in samples)
parallel = statistics.median(sample["automatic_seconds"] for sample in samples)

tests = {}
for label, expected_count in (("parallel-multiplicity-tests", 9),
                              ("validation-errors-tests", 2),
                              ("assignment-ownership-test", 1),
                              ("serial-multiplicity-tests", 9)):
    report = run(label)
    matched = re.search(r"test result: ok\. (\d+) passed; 0 failed; (\d+) ignored;",
                        (archive / label / "run.log").read_text())
    assert matched and int(matched[1]) == expected_count, label
    tests[label] = {"passed": int(matched[1]), "ignored": int(matched[2]),
                    "wall_seconds": report["wall_seconds"]}
proofs = records("parallel-multiplicity-tests", "MULTIPLICITY_PROOF ")
assert len(proofs) == 2 and all(p["verified"] and p["parity"] for p in proofs)
assert proofs == records("serial-multiplicity-tests", "MULTIPLICITY_PROOF ")
builds = {label: run(label) for label in (
    "parallel-test-build", "serial-test-build", "cuda-examples-build", "cuda-wrapper-build")}
binaries = {}
for label in ("parallel-binary", "serial-binary"):
    metadata = load(f"{label}.json")
    assert digest(Path(metadata["path"])) == metadata["sha256"], label
    binaries[label] = metadata
cuda_binaries = load("cuda-binaries.json")
assert len(cuda_binaries) == 3
for binary in cuda_binaries:
    assert digest(Path(binary["path"])) == binary["sha256"]
    assert all(binary["capabilities"][key]
               for key in ("kzg", "kzg_cuda", "goldilocks_cuda", "parallel"))

result = {
    "status": "verified_bounded_parallel_lookup_multiplicities",
    "scope": "Synthetic native-table distribution at the candidate lookup count; not a production recursive request or authenticated Filecoin proof",
    "source_input_count": len(after), "changed_source_inputs": changed,
    "fixture": fixture, "samples": samples,
    "median_serial_seconds": serial, "median_parallel_seconds": parallel,
    "speedup": serial / parallel,
    "time_reduction_percent": 100 * (1 - parallel / serial),
    "operation_benchmark_wall_seconds": benchmark_run["wall_seconds"],
    "operation_benchmark_peak_rss_bytes": benchmark_run["peak_rss_bytes"],
    "timing_includes": ["count allocation", "lookup scan", "integer histogram reduction", "field conversion"],
    "timing_excludes": ["fixture construction", "result checksums/comparison", "final result destruction", "assignment", "lowering", "proving"],
    "correctness_tests": tests,
    "correctness_executions_passed": sum(t["passed"] for t in tests.values()),
    "distinct_correctness_tests_passed": 12,
    "proof_parity": {"kzg_development_fixtures": proofs,
                     "fri_development_fixture_exact_bytes_and_verification": True,
                     "scope": "Small primitive fixtures; neither size is the final ceremony packet"},
    "independent_review": "Bounded live allocations, exact integer/field counts, duplicate mapping, serial fallback and benchmark scope reviewed without material issues",
    "builds": {label: {"wall_seconds": report["wall_seconds"],
                       "peak_rss_bytes": report["peak_rss_bytes"]} for label, report in builds.items()},
    "test_binaries": binaries, "cuda_binaries": cuda_binaries,
    "gpu_runtime_benchmark_performed": False,
    "production_recursive_latency_measured": False, "full_chain_measured": False,
    "authenticated_filecoin_artifact_available": False, "goal_complete": False,
}
(archive / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in (
    "status", "median_serial_seconds", "median_parallel_seconds", "speedup",
    "time_reduction_percent", "correctness_executions_passed")}, indent=2))
