#!/usr/bin/env python3
"""Verify and summarize the isolated recursive point-prefix comparison."""

import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


archive = Path(__file__).resolve().parent


def load(path):
    return json.loads((archive / path).read_text())


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


before = load("before/source-sha256.json")
after = load("source-sha256.json")
initial = load("initial-wrapper-fixture/source-sha256.json")
fixture_path = "experiments/kzg-wrap/src/native_verifier/parallel_prefix_tests.rs"
assert set(before) <= set(after) == set(initial)
assert [p for p in after if after[p] != initial[p]] == [fixture_path]
assert digest(archive / "initial-wrapper-fixture/parallel_prefix_tests.rs") == initial[fixture_path]
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
assert changed == [
    "experiments/kzg-wrap/src/native_curve.rs",
    "experiments/kzg-wrap/src/native_field.rs",
    "experiments/kzg-wrap/src/native_msm.rs",
    "experiments/kzg-wrap/src/native_transcript.rs",
    "experiments/kzg-wrap/src/native_verifier.rs",
    fixture_path,
    "src/plonkish/builder.rs",
    "src/plonkish/gadgets/bytes.rs",
    "src/plonkish/tests/parallel_prefix.rs",
    "src/plonkish/witness.rs",
]
(archive / "source-delta.patch").write_text("".join(patch))
initial_labels = {
    "parallel-library-build", "prefix-library-tests", "independent-witness-tests",
    "arithmetic-tests", "invalid-hint-tests", "wrapper-test-build-initial",
    "native-verifier-tests-missing-binary",
}


def run(label, exit_code=0):
    report = load(f"{label}/report.json")
    assert report["exit_code"] == exit_code and report["source_hashes_unchanged"], label
    assert not any(key.startswith(("RAYON_", "OMP_")) for key in report["environment"])
    expected = initial if label in initial_labels else after
    if report["command"][0] == "cargo":
        expected = {p: d for p, d in expected.items() if not p.endswith(".py")}
    assert report["source_input_count"] == len(expected), label
    assert report["source_inputs_sha256"] == manifest_digest(expected), label
    return report


def log(label):
    return (archive / label / "run.log").read_text()


def records(label, prefix):
    return [json.loads(line[len(prefix):]) for line in log(label).splitlines()
            if line.startswith(prefix)]


benchmark = run("point-benchmark")
setup, = records("point-benchmark", "NATIVE_POINT_PREFIX_SETUP ")
samples = records("point-benchmark", "NATIVE_POINT_PREFIX_SAMPLE ")
summary, = records("point-benchmark", "NATIVE_POINT_PREFIX_RESULT ")
assert setup["certified"] and setup["dynamic_points"] == 335
assert setup["prefix_blocks"] == 337 and setup["prefix_values"] == 35_023_984
assert setup["values"] == 35_040_064 and setup["inputs"] == 3_685
assert len(samples) == 10 and summary["paired_samples"] == 5
for i, sample in enumerate(samples):
    pair, position = divmod(i, 2)
    modes = ("serial", "parallel") if pair % 2 == 0 else ("parallel", "serial")
    assert sample["sample"] == pair and sample["mode"] == modes[position]
    assert all(sample[key] for key in ("full_assignment_equal", "input_mapping_equal",
                                       "public_encodings_equal", "full_relation_checked"))
    assert sample["assignment_blake3"] == summary["assignment_blake3"]
assert summary["assignment_blake3"] == "fe7f92ef9402a4300373c581ba8e9f36e10d09ca0b9b7bd31b83c46cbd1d39fb"
serial = statistics.median(s["seconds"] for s in samples if s["mode"] == "serial")
parallel = statistics.median(s["seconds"] for s in samples if s["mode"] == "parallel")
assert serial == summary["serial_median_seconds"]
assert parallel == summary["parallel_median_seconds"]
assert math.isclose(serial / parallel, summary["speedup"])

tests, test_names = {}, set()
for label, count in (
    ("prefix-library-tests", 6), ("independent-witness-tests", 5),
    ("arithmetic-tests", 9), ("invalid-hint-tests", 1),
    ("native-verifier-tests", 12), ("native-field-tests", 9),
    ("curve-tests", 3), ("msm-tests", 2), ("transcript-tests", 1),
    ("serial-prefix-tests", 6), ("serial-independent-tests", 5),
):
    report = run(label)
    matched = re.search(r"test result: ok\. (\d+) passed; 0 failed; (\d+) ignored;", log(label))
    assert matched and int(matched[1]) == count, label
    names = re.findall(r"^test (\S+) \.\.\. ok$", log(label), re.MULTILINE)
    assert len(names) == count, label
    test_names.update(names)
    tests[label] = {"passed": count, "ignored": int(matched[2]),
                    "wall_seconds": report["wall_seconds"]}
assert len(test_names) == 48 and sum(t["passed"] for t in tests.values()) == 59


def proof(label, prefix):
    match = re.search(rf"^{prefix} bytes=(\d+) blake3=([0-9a-f]{{64}})$", log(label), re.MULTILINE)
    assert match, label
    return {"bytes": int(match[1]), "blake3": match[2]}


prefix_proof = proof("prefix-library-tests", "PARALLEL_PREFIX_PROOF")
assert prefix_proof == proof("serial-prefix-tests", "PARALLEL_PREFIX_PROOF")
assert prefix_proof == {"bytes": 1877, "blake3": "fecddd92d7004991201440b6b9ef45739e29de422b295929032b1f3198b98136"}
independent_proof = proof("independent-witness-tests", "INDEPENDENT_ADVICE_PROOF")
assert independent_proof == proof("serial-independent-tests", "INDEPENDENT_ADVICE_PROOF")
builds = {label: run(label) for label in (
    "parallel-library-build", "serial-library-build", "wrapper-test-build",
    "cuda-examples-build", "cuda-wrapper-build")}
binaries = {}
for label in ("parallel-library-binary", "serial-library-binary", "wrapper-binary"):
    metadata = load(f"{label}.json")
    assert digest(Path(metadata["path"])) == metadata["sha256"], label
    binaries[label] = metadata
cuda_binaries = load("cuda-binaries.json")
assert len(cuda_binaries) == 3
for binary in cuda_binaries:
    assert digest(Path(binary["path"])) == binary["sha256"]
    assert all(binary["capabilities"][key]
               for key in ("kzg", "kzg_cuda", "goldilocks_cuda", "parallel"))
initial_build = run("wrapper-test-build-initial", exit_code=101)
missing_binary = run("native-verifier-tests-missing-binary", exit_code=127)

result = {
    "status": "verified_parallel_recursive_point_prefix",
    "scope": "Native point gadgets with synthetic coordinates at the candidate dynamic-point count; not a full recursive request or authenticated Filecoin proof",
    "source_input_count": len(after), "changed_source_inputs": changed,
    "fixture": setup, "samples": samples,
    "median_serial_seconds": serial, "median_parallel_seconds": parallel,
    "speedup": serial / parallel,
    "time_reduction_percent": 100 * (1 - parallel / serial),
    "operation_benchmark_wall_seconds": benchmark["wall_seconds"],
    "operation_benchmark_peak_rss_bytes": benchmark["peak_rss_bytes"],
    "untimed_warmup": False,
    "timing_includes": ["input assignment", "witness allocation", "recipe evaluation",
                        "complete frontend relation checks", "public extraction"],
    "timing_excludes": ["circuit construction", "relation/input mapping comparison",
                        "assignment comparisons and checksums", "assignment destruction",
                        "lowering", "proving"],
    "assignment_blake3": summary["assignment_blake3"],
    "correctness_tests": tests, "correctness_executions_passed": 59,
    "distinct_correctness_tests_passed": 48,
    "proof_parity": {
        "prefix_development_fixture": prefix_proof,
        "independent_advice_development_fixture": independent_proof,
        "arithmetic_development_fixture": proof("arithmetic-tests", "ARITHMETIC_PROOF"),
        "scope": "Small fully constrained fixtures; proof sizes are unrelated to the final ceremony packet",
    },
    "builds": {label: {"wall_seconds": report["wall_seconds"],
                       "peak_rss_bytes": report["peak_rss_bytes"]} for label, report in builds.items()},
    "initial_failed_attempts": {
        "wrapper_test_compile_exit_code": initial_build["exit_code"],
        "absent_binary_exit_code": missing_binary["exit_code"],
        "correction": "Scalar lacks Sum; two fixture-only sums use explicit field-add folds",
        "only_changed_input": fixture_path,
        "library_evidence_compiled_sources_unchanged": True,
        "counted_as_validation": False,
    },
    "test_binaries": binaries, "cuda_binaries": cuda_binaries,
    "gpu_runtime_benchmark_performed": False,
    "production_recursive_latency_measured": False, "full_chain_measured": False,
    "authenticated_filecoin_artifact_available": False, "goal_complete": False,
}
(archive / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in (
    "status", "median_serial_seconds", "median_parallel_seconds", "speedup",
    "time_reduction_percent", "correctness_executions_passed")}, indent=2))
