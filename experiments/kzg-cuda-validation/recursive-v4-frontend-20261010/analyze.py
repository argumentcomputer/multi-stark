#!/usr/bin/env python3
"""Verify the development-v4 bootstrap and complete recursive frontend evidence."""

import csv
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
pre_review = load("pre-review/source-sha256.json")
initial = load("initial-wrapper/source-sha256.json")
diagnostic = "experiments/kzg-wrap/src/saved/development.rs"
assert len(before) == 180 and len(after) == 183
assert set(before) <= set(after) == set(initial) == set(pre_review)
for version, manifest in (("pre-review", pre_review), ("initial-wrapper", initial)):
    assert [p for p in after if after[p] != manifest[p]] == [diagnostic]
    assert digest(archive / version / "development.rs") == manifest[diagnostic]
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
assert set(changed) == {
    "experiments/kzg-wrap/src/main.rs",
    "experiments/kzg-wrap/src/native_verifier.rs",
    "experiments/kzg-wrap/src/native_verifier/development_metadata.rs",
    "experiments/kzg-wrap/src/native_verifier/shape.rs",
    "experiments/kzg-wrap/src/saved.rs", diagnostic,
    "src/ark_adapter/pcs.rs", "src/ark_adapter/pcs/recommit_tests.rs",
    "src/ark_adapter/srs/cache.rs", "src/plonkish/foreign.rs",
}
assert (archive / "source-delta.patch").read_text() == "".join(patch)


def run(label, exit_code=0):
    report = load(f"{label}/report.json")
    assert report["exit_code"] == exit_code and report["source_hashes_unchanged"], label
    assert report["logical_cpus"] == 96
    assert not any(key.startswith(("RAYON_", "OMP_")) for key in report["environment"])
    expected = pre_review if label == "library-test-build" else initial if label in {
        "recommit-tests", "public-development-tests", "wrapper-test-build"
    } else after
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


tests, test_names = {}, set()
for label, count in (("recommit-tests", 4), ("public-development-tests", 4),
                     ("native-verifier-tests", 15), ("saved-input-tests", 1)):
    report = run(label)
    matched = re.search(r"test result: ok\. (\d+) passed; 0 failed; (\d+) ignored;", log(label))
    assert matched and int(matched[1]) == count, label
    names = re.findall(r"^test (\S+) \.\.\. ok$", log(label), re.MULTILINE)
    assert len(names) == count, label
    test_names.update(names)
    tests[label] = {"passed": count, "ignored": int(matched[2]),
                    "wall_seconds": report["wall_seconds"]}
assert len(test_names) == 24
builds = {label: run(label) for label in (
    "library-test-build", "wrapper-test-build-retry", "cuda-wrapper-build")}
failed_build = run("wrapper-test-build", 101)
assert "E0283" in log("wrapper-test-build")
binaries = {}
for name in ("library-binary", "wrapper-binary", "cuda-binary"):
    binary = load(f"{name}.json")
    assert digest(archive / Path(binary["path"]).name) == binary["sha256"]
    binaries[name] = binary
assert all(binaries["cuda-binary"]["capabilities"][key]
           for key in ("kzg", "kzg_cuda", "goldilocks_cuda", "parallel"))
mode_checks = load("diagnostic-selection-checks.json")
assert len(mode_checks) == 2
assert all(check["exit_code"] != 0 and not check["output_created"] for check in mode_checks)

fixture = archive / "development-fixture"
fixture_files = load("fixture-files.json")
assert len(fixture_files) == 47
for path, expected in fixture_files.items():
    assert (fixture / path).stat().st_size == expected["bytes"], path
    assert digest(fixture / path) == expected["sha256"], path
bootstrap = load("development-fixture/bootstrap-report.json")
bootstrap_run = run("bootstrap")
bootstrap_logged, = records("bootstrap", "DEVELOPMENT_V4_BOOTSTRAP ")
assert bootstrap == bootstrap_logged
parameters = load("development-fixture/development-v4.json")
assert parameters == bootstrap["parameters"]
assert parameters["known_trapdoor"] and not parameters["filecoin_acceptance"]
assert parameters["public_max_degree"] == 2**28 - 2
assert parameters["max_trace_len"] == 2**24
assert parameters["recursive_max_trace_len"] == 2**27
assert parameters["setup_id"] == "15d4cd916ad4d2d0a82a0bec81101e011b5a6d597355790b03eda3ae2c6bd1ef"
assert parameters["proof_blake3"] == "b4ceea2c1b6f051108d05c12b76130d5a0cd09d9fd78aaeb2d4daafc3a11cade"
assert bootstrap["proof_bytes"] == (fixture / "proof.compact.bin").stat().st_size == 38_229
assert bootstrap["native_verification_passed"] and bootstrap["encoding_negative_checks_passed"]
assert bootstrap["altered_claims_rejected"] == 18
assert not bootstrap["outer_proof_generated"] and not bootstrap["full_pipeline_run"]
coefficients = bootstrap["source_coefficients"]
assert len(coefficients) == 38 and len({r["path"] for r in coefficients}) == 38
assert records("bootstrap", "DEVELOPMENT_RECOMMIT ") == coefficients
assert all(r["ordinary_commitment_matches_verified_legacy"] and not r["shifted_commitments"]
           and r["height"] <= 2**24 and r["width"] > 0 for r in coefficients)
audit = bootstrap["metadata_audit"]
assert audit["circuit_count"] == len(audit["circuits"]) == 19
assert all(c["lookup_group"] == c["current_v4_lookup_group"] for c in audit["circuits"])
release = bootstrap["gpu_release"]
assert release["quiesced"] and release["cuda_enabled"]
assert {d["device"] for d in release["devices"]} == {0, 1, 2, 3}
assert all(d["after"][key] == 0 for d in release["devices"]
           for key in ("resident_coefficient_bytes", "srs_point_bytes", "msm_workspace_bytes"))
gpu_run = load("bootstrap-gpu.json")
assert gpu_run["monitor_reaped"] and gpu_run["command_returncode"] == 0
assert len(gpu_run["compute_processes_before"].splitlines()) == 1
assert len(gpu_run["compute_processes_after"].splitlines()) == 1
gpu_samples = {device: [] for device in range(4)}
with (archive / "bootstrap-gpu.csv").open() as source:
    for row in csv.DictReader(source, skipinitialspace=True):
        gpu_samples[int(row["index"])].append({
            "utilization_percent": int(row["utilization.gpu [%]"].split()[0]),
            "memory_mib": int(row["memory.used [MiB]"].split()[0]),
        })
msm_events = []
for line in log("bootstrap").splitlines():
    if line.startswith("KZG CUDA profile "):
        event = dict(re.findall(r"(\w+)=([^ ]+)", line))
        if event["operation"] == "msm-compute":
            msm_events.append(event)
assert {int(e["device"]) for e in msm_events if float(e["kernel_ms"]) > 0} == {0, 1, 2, 3}
gpu_summary = {device: {
    "samples": len(samples),
    "mean_utilization_percent": statistics.mean(s["utilization_percent"] for s in samples),
    "peak_utilization_percent": max(s["utilization_percent"] for s in samples),
    "peak_memory_mib": max(s["memory_mib"] for s in samples),
    "msm_compute_events": sum(int(e["device"]) == device for e in msm_events),
    "msm_kernel_seconds": sum(float(e["kernel_ms"]) for e in msm_events
                              if int(e["device"]) == device) / 1000,
} for device, samples in gpu_samples.items()}

frontend = load("frontend-output/frontend-report.json")
frontend_run = run("frontend")
frontend_logged, = records("frontend", "DEVELOPMENT_V4_FRONTEND ")
assert frontend == frontend_logged
assert frontend["known_trapdoor"] and not frontend["filecoin_acceptance"]
assert not frontend["outer_proof_generated"] and not frontend["full_pipeline_run"]
assert frontend["setup_id"] == parameters["setup_id"]
assert frontend["input_blake3"]["proof.compact.bin"] == parameters["proof_blake3"]
assert frontend["input_proof_bytes"] == bootstrap["proof_bytes"]
assert frontend["circuit_count"] == 2 and frontend["main_heights"] == [2**27]
assert frontend["rows"] == frontend["gates"] + frontend["lookups"] + frontend["publics"] + 1
assert frontend["rows"] == 118_634_362
assert frontend["values"] == 115_510_108 and frontend["publics"] == 22
assert frontend["reference_storage_bytes"] == frontend["values"] * 32
assert frontend["repeated_assignment_and_trace_parity"]
samples = frontend["samples"]
assert len(samples) == 2 and records("frontend", "DEVELOPMENT_RECURSIVE_SAMPLE ") == samples
for index, sample in enumerate(samples):
    assert sample["sample"] == index and sample["reference_sample"] == (index == 0)
    assert sample["complete_assignment_equal"] == (True if index else None)
    assert sample["trace_checksums_equal"] == (True if index else None)
    checks = sample["checks"]
    assert checks["circuit_satisfied"] and checks["external_pairings_pass"]
    assert checks["msm_terms"] == [583, 9] and checks["pairing_points"] == 2
    assert checks["rows"] == frontend["rows"] and not checks["outer_proof_generated"]
    assert [(t["index"], t["height"], t["width"]) for t in sample["traces"]] == [
        (0, 2**27, 3), (1, 2**17, 1)]
    assert [t["blake3"] for t in sample["traces"]] == [t["blake3"] for t in samples[0]["traces"]]
    assert all(math.isfinite(t["seconds"]) and t["seconds"] > 0 for t in sample["traces"])
assert "Point witness prefix: admitted=true, blocks=505, values=35025875" in log("frontend")
assert frontend_run["environment"]["MULTI_STARK_KZG_BACKEND"] == "cpu"
assert bootstrap_run["environment"]["MULTI_STARK_KZG_CUDA_DEVICES"] == "0,1,2,3"

result = {
    "status": "verified_genuine_development_v4_recursive_frontend",
    "scope": "Complete production-shape recursive frontend from a freshly proved development-v4 inner proof; no outer proof or authenticated Filecoin parameters",
    "source_input_count": len(after), "source_inputs_sha256": manifest_digest(after),
    "changed_source_inputs": changed,
    "bootstrap": {
        "external_wall_seconds": bootstrap_run["wall_seconds"],
        "internal_wall_seconds": bootstrap["wall_seconds"],
        "peak_rss_bytes": bootstrap_run["peak_rss_bytes"],
        "coefficient_files": len(coefficients),
        "coefficient_bytes_read": sum(r["bytes"] for r in coefficients),
        "restore_decode_hash_seconds": sum(r["restore_seconds"] for r in coefficients),
        "recommit_seconds": sum(r["recommit_seconds"] for r in coefficients),
        "v4_proving_seconds": bootstrap["v4_proving_seconds"],
        "proof_bytes": bootstrap["proof_bytes"], "parameters": parameters,
        "native_verification_passed": True, "altered_claims_rejected": 18,
        "encoding_negative_checks_passed": True,
        "gpu": gpu_summary,
        "gpu_scope": "Whole bootstrap, including 116.715 GiB of host checkpoint restore; event durations overlap and cannot be added to infer wall time",
        "gpu_monitor_reaped": True, "gpu_memory_quiesced": True,
        "trust_scope": "Preserved first-stage metadata and coefficient checkpoints are trusted local inputs; FRI and first-stage witness generation are not repeated",
    },
    "frontend": {
        "external_wall_seconds": frontend_run["wall_seconds"],
        "internal_wall_seconds": frontend["wall_seconds"],
        "peak_rss_bytes": frontend_run["peak_rss_bytes"],
        "extra_retained_reference_bytes": frontend["reference_storage_bytes"],
        "load_and_native_verify_seconds": frontend["load_and_native_verify_seconds"],
        "compile_seconds": frontend["compile_seconds"],
        "lowering_seconds": frontend["lowering_seconds"],
        "fresh_assignment_and_traces_seconds": [s["fresh_assignment_seconds"] +
            sum(t["seconds"] for t in s["traces"]) for s in samples],
        "samples": samples, "rows": frontend["rows"],
        "row_headroom": 2**27 - frontend["rows"],
        "row_headroom_percent": 100 * (1 - frontend["rows"] / 2**27),
        "values": frontend["values"], "main_heights": frontend["main_heights"],
        "circuit_count": 2, "public_values": 22,
        "frontend_identity": frontend["frontend_identity"],
        "assignment_blake3": frontend["assignment_blake3"],
        "point_prefix_blocks": 505, "point_prefix_values": 35_025_875,
        "full_assignment_and_trace_parity": True,
        "untimed_warmup": False, "timing_excludes": frontend["operation_timing_excludes"],
        "rss_scope": "Two sequential fresh samples retain the first complete assignment for equality; this is not single-request peak RSS",
        "comparative_speedup_measured": False,
    },
    "correctness_tests": tests, "distinct_correctness_tests_passed": 24,
    "diagnostic_selection_negative_checks": mode_checks,
    "builds": {label: {"wall_seconds": report["wall_seconds"],
                       "peak_rss_bytes": report["peak_rss_bytes"]} for label, report in builds.items()},
    "initial_compile_failure": {
        "exit_code": failed_build["exit_code"], "counted_as_validation": False,
        "only_changed_input": diagnostic,
        "correction": "Ambiguous nested empty-Vec equality replaced by explicit vector length and is_empty checks",
        "library_evidence_compiled_sources_unchanged": True,
    },
    "binaries": binaries,
    "authenticated_filecoin_artifact_available": False,
    "production_recursive_proving_latency_measured": False,
    "outer_proof_generated": False, "final_packet_measured": False,
    "full_chain_measured": False, "goal_complete": False,
}
(archive / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "status": result["status"], "bootstrap_seconds": bootstrap_run["wall_seconds"],
    "recursive_compile_seconds": frontend["compile_seconds"],
    "recursive_lowering_seconds": frontend["lowering_seconds"],
    "recursive_fresh_frontend_seconds": result["frontend"]["fresh_assignment_and_traces_seconds"],
    "rows": frontend["rows"], "distinct_correctness_tests_passed": 24,
}, indent=2))
