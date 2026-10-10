#!/usr/bin/env python3
"""Summarize the recorded recursive-advice comparison and correctness gates."""

import difflib
import hashlib
import json
from pathlib import Path
import re
import statistics
import tomllib


archive = Path(__file__).resolve().parent


def load(path):
    return json.loads((archive / path).read_text())


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def benchmark(label):
    report = load(f"{label}/report.json")
    assert report["exit_code"] == 0 and report["source_hashes_unchanged"]
    samples = [dict(iteration=int(index), relations=int(count), reference_seconds=float(reference),
                    implementation_seconds=float(implementation))
               for index, count, reference, implementation in re.findall(
                   r"RECURSIVE_ADVICE_BENCH iteration=(\d+) relations=(\d+) "
                   r"reference_seconds=([0-9.]+) optimized_seconds=([0-9.]+)",
                   (archive / label / "run.log").read_text())]
    assert len(samples) == 5 and all(sample["relations"] == 4096 for sample in samples)
    return {"samples": samples,
            "median_implementation_seconds": statistics.median(s["implementation_seconds"] for s in samples),
            "median_reference_seconds": statistics.median(s["reference_seconds"] for s in samples)}


before_hashes = load("before/source-sha256.json")
after_hashes = load("source-sha256.json")
assert set(before_hashes) == set(after_hashes)
patch = []
changed = []
for path in before_hashes:
    before = archive / "before/source" / path
    after = archive / "source" / path
    assert digest(before) == before_hashes[path], path
    assert digest(after) == after_hashes[path], path
    if before_hashes[path] != after_hashes[path]:
        changed.append(path)
        patch.extend(difflib.unified_diff(before.read_text().splitlines(keepends=True),
                                         after.read_text().splitlines(keepends=True),
                                         fromfile=f"a/{path}", tofile=f"b/{path}"))
(archive / "source-delta.patch").write_text("".join(patch))
before_lock = tomllib.loads((archive / "before/source/experiments/kzg-wrap/Cargo.lock").read_text())
after_lock = tomllib.loads((archive / "source/experiments/kzg-wrap/Cargo.lock").read_text())
external = lambda lock: [p for p in lock["package"] if p["name"] != "init-kzg-wrap"]
assert external(before_lock) == external(after_lock)

before = benchmark("before-advice-benchmark")
after = benchmark("after-advice-benchmark")
tests = {}
for label in ("native-field-tests", "native-verifier-tests", "curve-tests", "msm-tests", "advice-proof-parity"):
    report = load(f"{label}/report.json")
    assert report["exit_code"] == 0 and report["source_hashes_unchanged"], label
    matched = re.search(r"test result: ok\. (\d+) passed; 0 failed; (\d+) ignored;",
                        (archive / label / "run.log").read_text())
    assert matched, label
    tests[label] = {"passed": int(matched[1]), "ignored": int(matched[2]),
                    "wall_seconds": report["wall_seconds"], "peak_rss_bytes": report["peak_rss_bytes"]}
proof = re.search(r"RECURSIVE_ADVICE_PROOF bytes=(\d+) blake3=([0-9a-f]{64})",
                  (archive / "advice-proof-parity/run.log").read_text())
assert proof
assert proof[1] == "2245" and proof[2] == "508fe257542e77aff92fe7c096b7524a49a535c8e7158cf4f33468594bbc0d04"
for name in ("before/binary.json", "binary.json", "cuda-binary.json"):
    binary = load(name)
    assert digest(Path(binary["path"])) == binary["sha256"], name
cuda = load("cuda-binary.json")
assert all(cuda["capabilities"][key] for key in ("kzg", "kzg_cuda", "goldilocks_cuda", "parallel"))
for label in ("before-build", "after-build", "cuda-wrapper-build"):
    build = load(f"{label}/report.json")
    assert build["exit_code"] == 0 and build["source_hashes_unchanged"], label

result = {
    "status": "verified_recursive_advice_exact_division_optimization",
    "scope": "Known-trapdoor native-field and genuine v4 recursive fixtures; no production recursive timing or Filecoin proof",
    "changes": ["One combined signed quotient/remainder calculation per relation",
                "Exact signed multiplication by 2^80 and division by 2^160 use shifts with a divisibility guard"],
    "source_input_count": len(after_hashes), "changed_source_inputs": changed,
    "locked_external_dependencies_unchanged": True,
    "comparison": "Same 4096 deterministic relation cases in two CPU test executables with archived sources. Five samples per executable, before then after; each binary alternates old-reference/implementation order after exact parity warmup. Benchmarks run sequentially with default machine parallelism.",
    "timing_includes": ["relation advice", "result vector allocation"],
    "timing_excludes": ["case construction", "circuit construction", "range hints", "full witness validation", "lowering", "proving"],
    "before": before, "after": after,
    "speedup": before["median_implementation_seconds"] / after["median_implementation_seconds"],
    "time_reduction_percent": 100 * (1 - after["median_implementation_seconds"] / before["median_implementation_seconds"]),
    "unchanged_reference_time_ratio_after_over_before": after["median_reference_seconds"] / before["median_reference_seconds"],
    "correctness_tests": tests, "correctness_tests_passed": sum(t["passed"] for t in tests.values()),
    "proof_parity": {"bytes": int(proof[1]), "blake3": proof[2],
                     "exact_reference_and_optimized_bytes": True,
                     "scope": "Native Fq primitive with complete 2^16 range table; not final ceremony packet"},
    "independent_source_review": "Signed arithmetic, error ordering and strings preserved; only direct num-integer dependency edge added",
    "cuda_binary": cuda,
    "gpu_runtime_benchmark_performed": False,
    "production_recursive_latency_measured": False,
    "full_chain_measured": False,
    "authenticated_filecoin_artifact_available": False,
    "goal_complete": False,
}
(archive / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ("status", "speedup", "time_reduction_percent", "correctness_tests_passed")}, indent=2))
