#!/usr/bin/env python3
"""Verify integrated development-v4 pipeline evidence and request boundaries."""

from datetime import datetime
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re


archive = Path(__file__).resolve().parent


def load(name):
    return json.loads((archive / name).read_text())


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


before, after = load("before/source-sha256.json"), load("source-sha256.json")
assert len(before) == len(after) == 186 and before.keys() == after.keys()
changed, patch = [], []
for name, expected in after.items():
    old, new = archive / "before/source" / name, archive / "source" / name
    assert digest(old) == before[name] and digest(new) == expected
    if before[name] != expected:
        changed.append(name)
        patch.extend(difflib.unified_diff(old.read_text().splitlines(keepends=True),
                                         new.read_text().splitlines(keepends=True),
                                         fromfile=f"a/{name}", tofile=f"b/{name}"))
assert len(changed) == 13
assert (archive / "source-delta.patch").read_text() == "".join(patch)


def run(label, source_hashes=after):
    item = load(f"{label}/report.json")
    assert item["exit_code"] == 0 and item["source_hashes_unchanged"], label
    assert item["logical_cpus"] == 96
    assert not any(key.startswith(("RAYON_", "OMP_")) for key in item["environment"])
    expected = source_hashes
    if item["command"][0] == "cargo":
        expected = {p: h for p, h in source_hashes.items() if not p.endswith(".py")}
    assert item["source_input_count"] == len(expected)
    assert item["source_inputs_sha256"] == manifest_digest(expected)
    return item


builds = {label: run(label) for label in (
    "first-test-build", "wrapper-test-build", "cuda-examples-build", "cuda-wrapper-build")}
tests = {label: run(label) for label in ("first-worker-tests", "wrapper-tests")}
executed = []
for label, count in (("first-worker-tests", 7), ("wrapper-tests", 21)):
    log = (archive / label / "run.log").read_text()
    assert f"test result: ok. {count} passed; 0 failed;" in log
    names = re.findall(r"^test (\S+) \.\.\. ok$", log, re.MULTILINE)
    assert len(names) == count
    executed.extend(name.replace("outer::setup::tests::", "setup::tests::") for name in names)
python_tests = load("python-tests/report.json")
assert python_tests["exit_code"] == 0 and python_tests["source_hashes_unchanged"]
for name, expected in python_tests["source_inputs"].items():
    assert digest(archive / "python-tests/source" / name) == expected == after[name]
python_log = (archive / "python-tests/run.log").read_text()
assert "Ran 24 tests" in python_log and python_log.rstrip().endswith("OK")
python_names = re.findall(r"^(test_\S+) \([^\n]+\) \.\.\. ok$", python_log, re.MULTILINE)
assert len(python_names) == 24
executed.extend(python_names)
assert len(executed) == 52 and len(set(executed)) == 48
for name in ("first-test-binary", "wrapper-test-binary"):
    binary = load(name + ".json")
    assert digest(archive / "bin" / Path(binary["path"]).name) == binary["sha256"]
binaries = load("cuda-binaries.json")
for name, metadata in binaries.items():
    assert digest(archive / "bin" / name) == metadata["sha256"]
    assert all(metadata["capabilities"][key]
               for key in ("kzg", "kzg_cuda", "goldilocks_cuda", "parallel"))

post_sources = load("post-measurement/source-sha256.json")
assert post_sources.keys() == after.keys()
post_changed = [name for name in after if after[name] != post_sources[name]]
assert post_changed == ["experiments/kzg-wrap/src/saved/worker.rs"]
post_patch = []
for name, expected in post_sources.items():
    path = archive / ("post-measurement/source" if name in post_changed else "source") / name
    assert digest(path) == expected
    if name in post_changed:
        measured = (archive / "source" / name).read_text()
        corrected = path.read_text()
        correction = '        report["outer_proof_generated"] = true.into();\n'
        assert corrected.count(correction) == 1
        assert corrected.replace(correction, "", 1) == measured
        post_patch.extend(difflib.unified_diff(measured.splitlines(keepends=True),
                                              corrected.splitlines(keepends=True),
                                              fromfile=f"a/{name}", tofile=f"b/{name}"))
assert (archive / "post-measurement/source-delta.patch").read_text() == "".join(post_patch)
post_change = load("post-measurement/change.json")
assert post_change["changed_source_inputs"] == post_changed
assert post_change["measured_source_inputs_sha256"] == manifest_digest(after)
assert post_change["current_source_inputs_sha256"] == manifest_digest(post_sources)
assert not post_change["measured_timing_rerun"] and post_change["independent_review_passed"]
post_build = run("post-measurement/cuda-wrapper-build", post_sources)
post_binary = load("post-measurement/cuda-wrapper-binary.json")
assert digest(archive / "post-measurement/bin/init-kzg-wrap") == post_binary["sha256"]
assert post_binary["capabilities"] == binaries["init-kzg-wrap"]["capabilities"]

measurement = run("integrated-process")
process = load("process-lifecycle.json")
assert process["exit_code"] == 0 and process["process_reaped"]
assert all(len(process[when]["processes"].splitlines()) == 1 for when in ("gpu_before", "gpu_after"))
assert process["known_trapdoor"] and not process["filecoin_acceptance"]
assert len(process["artifact_parity"]) == 13
assert all(item["equal"] and item["actual_sha256"] == item["expected_sha256"]
           for item in process["artifact_parity"])
for name, metadata in load("output-files.json").items():
    path = archive / "outputs" / name
    assert path.stat().st_size == metadata["bytes"] and digest(path) == metadata["sha256"]
original_output = Path(process["output"])
for item in process["artifact_parity"]:
    actual = archive / "outputs" / Path(item["actual"]).relative_to(original_output)
    expected = Path(item["expected"])
    if expected.is_relative_to(original_output):
        expected = archive / "outputs" / expected.relative_to(original_output)
    else:
        assert expected.is_relative_to(archive.parent)
    assert digest(actual) == item["actual_sha256"]
    assert digest(expected) == item["expected_sha256"]
    assert actual.read_bytes() == expected.read_bytes()
report = load("outputs/report.json")
assert report["status"] == "verified_regenerated_chain"
assert report["sources"] == {p: h for p, h in after.items() if p != "experiments/test_kzg_cuda_bench.py"}
for field in ("binaries", "binary_snapshots"):
    assert {Path(path).name: value for path, value in report[field].items()} == {
        name: metadata["sha256"] for name, metadata in binaries.items()}
assert report["setup"] == "development" and report["development_public_degree"]
assert report["goal_acceptance"] == {"requested": False, "passed": False}
assert report["retained_workers"] and report["fri_compression_rerun"] and report["fused_staging"]
assert report["distributed_wrapper"] and report["kzg_devices"] == [0, 1, 2, 3]
assert report["fri_input"] == report["worker_startup"]["seed"]["input_hashes"]
assert report["root_artifacts"]["root-claims.bin"] == report["worker_startup"]["seed"]["expected_claims_sha256"]
assert report["worker_startup_wall_seconds"] == report["worker_startup"]["wall_seconds"]
assert report["worker_startup_plus_pipeline_wall_seconds"] > report["pipeline_wall_seconds"]

spec = importlib.util.spec_from_file_location("measured_benchmark", archive / "source/experiments/kzg-cuda-bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
for role, name in bench.CAPABILITY_BINARIES.items():
    assert report["capabilities"][role] == binaries[name]["capabilities"]
gpu_csv = archive / "outputs/gpu-samples.csv"
assert bench.summarize_gpu_samples(gpu_csv) == report["gpu_samples_every_2_seconds"]
checks = bench.evaluate_goal(report)
assert set(checks["failures"]) == {"both_stages_authenticated_filecoin"}
bench.validate_development_public_degree_reports([
    *report["worker_startup"]["verification"].values(),
    *[report[key] for key in ("first_stage_verification", "recursive_verification",
                             "first_stage_cpu_verification", "recursive_cpu_verification")],
])
for stage, directory in (("fri_to_kzg", "fri-to-kzg"), ("recursive_kzg", "recursive")):
    for name, metadata in report["proofs"][stage].items():
        actual = archive / "outputs" / directory / "kzg" / name
        warm = archive / "outputs/worker-warmup" / directory / "kzg" / name
        assert actual.read_bytes() == warm.read_bytes()
        assert digest(actual) == metadata["sha256"] and actual.stat().st_size == metadata["bytes"]
assert report["proofs"]["recursive_kzg"]["packet.bin"]["bytes"] == 2181
assert report["recursive_verification"]["trace_heights"] == [2**27, 2**17]

phases = {}
for name, phase in {**report["worker_startup"]["phases"], **report["phases"]}.items():
    path = archive / "outputs" / Path(phase["log"]).name
    operations, quality = bench.summarize_cuda_events(path, with_quality=True)
    assert operations == phase["cuda_operation_totals_overlap"]
    assert quality == phase["cuda_profile_parse"]
    assert bench.summarize_gpu_samples(
        gpu_csv, datetime.fromisoformat(phase["started_at_utc"]),
        datetime.fromisoformat(phase["finished_at_utc"])) == phase["gpu_samples_every_2_seconds"]
    if "stage_and_prove" in name:
        assert {device for device, item in quality["devices"].items() if item["kernel_records"] > 0} == {"0", "1", "2", "3"}
    phases[name] = {key: phase[key] for key in ("wall_seconds", "gpu_samples_every_2_seconds")}
    for key in ("peak_rss_bytes", "worker_response", "memory_before", "memory_after", "output_identity",
                "fixed_cache_hits", "fixed_cache_misses", "srs_cache_hits", "srs_cache_misses"):
        if key in phase:
            phases[name][key] = phase[key]
    phases[name]["cuda_profile_parse"] = quality
    phases[name]["cuda_event_totals"] = operations
    consumers = []
    for line in path.read_text().splitlines():
        match = re.search(r"KZG (lookup|quotient) distributed across peer pairs .*rows=(\d+) .*seconds=([0-9.eE+-]+)", line)
        if match:
            assert "ordinals=[0, 1, 2, 3]" in line and int(match[2]) == 2**27
            consumers.append({"operation": match[1], "rows": int(match[2]), "seconds": float(match[3])})
    if "recursive_stage_and_prove" in name:
        assert len(consumers) == 2 and {item["operation"] for item in consumers} == {"lookup", "quotient"}
    phases[name]["distributed_consumers"] = consumers

memory = [json.loads(line) for line in (archive / "host-memory.jsonl").read_text().splitlines()]
start = datetime.fromisoformat(report["pipeline_started_at_utc"]).timestamp()
end = datetime.fromisoformat(report["pipeline_finished_at_utc"]).timestamp()
fresh = [sample for sample in memory if start <= sample["unix_seconds"] <= end]
assert fresh
worker_pids = {worker["pid"] for worker in report["worker_startup"]["workers"].values()}
assert all(worker_pids <= {p["pid"] for p in sample["processes"]} for sample in fresh)
result = {
    "status": "verified_integrated_development_v4_pipeline",
    "scope": "Fresh root FRI compression, fresh first-stage and recursive assignments/proofs with both genuine retained workers; native verification included, worker preparation/shutdown and additional CPU verification separate",
    "source_input_count": len(after), "source_inputs_sha256": manifest_digest(after),
    "changed_source_inputs": changed,
    "pipeline_wall_seconds": report["pipeline_wall_seconds"],
    "worker_startup_wall_seconds": report["worker_startup_wall_seconds"],
    "worker_startup_plus_pipeline_wall_seconds": report["worker_startup_plus_pipeline_wall_seconds"],
    "worker_shutdown_wall_seconds": report["worker_shutdown_wall_seconds"],
    "external_wall_seconds": measurement["wall_seconds"],
    "phase_measurements": phases,
    "proof_bytes": report["recursive_verification"]["proof_bytes"],
    "packet_bytes": report["recursive_verification"]["packet_bytes"],
    "warmup_fresh_and_prior_artifact_parity": True,
    "memory": {
        "scope": process["memory_scope"], "samples": len(memory), "fresh_request_samples": len(fresh),
        "whole_run_sum_rss_peak_bytes": max(sample["sum_rss_bytes"] for sample in memory),
        "fresh_request_sum_rss_peak_bytes": max(sample["sum_rss_bytes"] for sample in fresh),
        "both_workers_alive_during_fresh_request": True,
    },
    "validation": {"test_executions_passed": len(executed), "distinct_tests_passed": len(set(executed)),
                   "tests": sorted(set(executed)), "shared_setup_tests_repeated_in_both_binaries": 4,
                   "python_test_seconds": python_tests["wall_seconds"],
                   "rust_tests": tests, "builds": builds},
    "binaries": binaries,
    "post_measurement_reporting_correction": {
        "change": post_change, "build": post_build, "binary": post_binary,
        "measured_sources_binaries_and_outputs_preserved": True,
    },
    "diagnostic_gate_checks": dict(checks, requested=False),
    "known_trapdoor": True, "filecoin_acceptance": False,
    "development_request_under_300_seconds": checks["checks"]["pipeline_under_300_seconds"]["passed"],
    "authenticated_filecoin_artifact_available": False,
    "ceremony_final_packet_measured": False,
    "full_development_chain_measured": True, "full_filecoin_chain_measured": False,
    "goal_complete": False,
}
(archive / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in (
    "status", "pipeline_wall_seconds", "worker_startup_wall_seconds", "external_wall_seconds",
    "proof_bytes", "packet_bytes", "development_request_under_300_seconds")}, indent=2))
