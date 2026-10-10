#!/usr/bin/env python3
"""Verify cold and retained recursive-v4 proving measurements and provenance."""

import csv
from datetime import datetime, timezone
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


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


before = load("before/source-sha256.json")
after = load("source-sha256.json")
initial = load("initial-test/source-sha256.json")
fixture = "experiments/kzg-wrap/src/outer/parameters/tests.rs"
assert len(before) == 183 and len(after) == 186
assert set(before) <= set(after) == set(initial)
assert [p for p in after if after[p] != initial[p]] == [fixture]
assert digest(archive / "initial-test/tests.rs") == initial[fixture]
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
    "experiments/kzg-wrap/src/main.rs", "experiments/kzg-wrap/src/outer.rs",
    "experiments/kzg-wrap/src/outer/parameters.rs", fixture,
    "experiments/kzg-wrap/src/saved.rs", "experiments/kzg-wrap/src/saved/development.rs",
    "experiments/kzg-wrap/src/saved/development/proving.rs",
}
assert (archive / "source-delta.patch").read_text() == "".join(patch)


def run(label, exit_code=0):
    report = load(f"{label}/report.json")
    assert report["exit_code"] == exit_code and report["source_hashes_unchanged"], label
    assert report["logical_cpus"] == 96
    assert not any(key.startswith(("RAYON_", "OMP_")) for key in report["environment"])
    expected = initial if label in {"wrapper-test-build", "outer-tests"} else after
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


builds = {label: run(label) for label in (
    "wrapper-test-build", "wrapper-test-build-retry", "cuda-wrapper-build")}
run("outer-tests", 101)
assert "index out of bounds: the len is 0 but the index is 5" in log("outer-tests")
run("outer-tests-retry", 101)
assert 'kind: AlreadyExists' in log("outer-tests-retry")
tests = run("outer-tests-isolated")
assert "test result: ok. 8 passed; 0 failed; 0 ignored;" in log("outer-tests-isolated")
assert "MULTI_STARK_KZG_BACKEND=cpu" in tests["command"]
assert any(c.startswith("TMPDIR=") for c in tests["command"])
test_names = re.findall(r"^test (\S+) \.\.\. ok$", log("outer-tests-isolated"), re.MULTILINE)
assert len(test_names) == 8
binaries = {}
for name in ("wrapper-binary", "cuda-binary"):
    binary = load(f"{name}.json")
    assert digest(archive / Path(binary["path"]).name) == binary["sha256"]
    binaries[name] = binary
initial_binary = load("initial-test/wrapper-binary.json")
assert digest(archive / "initial-test/wrapper-tests") == initial_binary["sha256"]
assert all(binaries["cuda-binary"]["capabilities"][key]
           for key in ("kzg", "kzg_cuda", "goldilocks_cuda", "parallel"))
mode_checks = load("diagnostic-selection-checks.json")
assert len(mode_checks) == 2
assert all(check["exit_code"] != 0 and not check["output_created"] for check in mode_checks)

measurement = run("recursive-proving")
independent = run("independent-cpu-verification")
assert independent["environment"]["MULTI_STARK_KZG_BACKEND"] == "cpu"
report = load("outputs/recursive-proving-report.json")
logged, = records("recursive-proving", "DEVELOPMENT_V4_RECURSIVE_PROVING ")
assert report == logged
assert report["known_trapdoor"] and not report["filecoin_acceptance"]
assert report["outer_proof_generated"] and not report["full_pipeline_run"]
assert report["max_trace_len"] == 2**27 and report["public_max_degree"] == 2**28 - 2
assert report["rows"] == 118_634_362 and report["values"] == 115_510_108
assert report["publics"] == 22 and report["input_proof_bytes"] == 38_229
assert report["repeated_proof_packet_profile_parity"]
assert report["retained_request_releases_device_residency_between_requests"]
assert report["fixed_cache_directory"] is None
assert report["input_blake3"]["proof.compact.bin"] == "b4ceea2c1b6f051108d05c12b76130d5a0cd09d9fd78aaeb2d4daafc3a11cade"
assert report["setup_id"] == "15d4cd916ad4d2d0a82a0bec81101e011b5a6d597355790b03eda3ae2c6bd1ef"
samples = report["samples"]
assert len(samples) == 2
assert records("recursive-proving", "DEVELOPMENT_RECURSIVE_PROVING_SAMPLE ") == samples
reference_artifacts = None
for index, sample in enumerate(samples):
    assert sample["sample"] == index and sample["loaded_key_reused"] == (index > 0)
    assert sample["reference_sample"] == (index == 0)
    assert sample["proof_packet_profile_equal"] == (True if index else None)
    assert sample["request_finished_unix_seconds"] > sample["request_started_unix_seconds"]
    checks = sample["checks"]
    assert checks["circuit_satisfied"] and checks["external_pairings_pass"]
    assert checks["msm_terms"] == [583, 9] and checks["pairing_points"] == 2
    outer = sample["proof_report"]
    assert outer == load(f"outputs/request-{index}/kzg/prove-report.json")
    assert outer["known_trapdoor"] and outer["development_srs"]
    assert not outer["filecoin_acceptance"] and outer["setup"] == "development"
    assert outer["ceremony_id"] is None and outer["filecoin_manifest_digest"] is None
    assert outer["public_setup_id"] == report["setup_id"]
    assert outer["public_max_degree"] == 2**28 - 2
    assert outer["trace_heights"] == [2**27, 2**17]
    assert outer["loaded_key_reused"] == (index > 0)
    assert outer["packet_bytes"] < 3_000
    assert all(outer[k] for k in ("native_verification_passed", "external_pairings_pass", "negative_tests_pass"))
    if index:
        assert outer["srs_config_load_seconds"] == 0
    release = sample["gpu_release"]
    assert release["quiesced"] and release["cuda_enabled"] and release["initialized"]
    assert {d["device"] for d in release["devices"]} == {0, 1, 2, 3}
    assert all(d["after"][key] == 0 for d in release["devices"]
               for key in ("resident_coefficient_bytes", "srs_point_bytes", "msm_workspace_bytes"))
    artifacts = [(archive / f"outputs/request-{index}/kzg" / name).read_bytes()
                 for name in ("proof.compact.bin", "packet.bin", "profile-id.bin")]
    assert len(artifacts[0]) == outer["proof_bytes"] and len(artifacts[1]) == outer["packet_bytes"]
    assert len(artifacts[2]) == 32 and artifacts[1].startswith(artifacts[2])
    if reference_artifacts is not None:
        assert artifacts == reference_artifacts
    reference_artifacts = artifacts
verified = load("outputs/request-1/kzg/verify-report.json")
assert verified["known_trapdoor"] and not verified["filecoin_acceptance"]
assert verified["packet_bytes"] == samples[1]["proof_report"]["packet_bytes"]
assert verified["public_setup_id"] == report["setup_id"]
assert all(verified[k] for k in ("native_verification_passed", "external_pairings_pass", "negative_tests_pass"))
for name, metadata in load("output-files.json").items():
    path = archive / "outputs" / name
    assert path.stat().st_size == metadata["bytes"] and digest(path) == metadata["sha256"]

gpu_metadata = load("recursive-proving-gpu.json")
assert gpu_metadata["monitor_reaped"] and gpu_metadata["command_returncode"] == 0
assert len(gpu_metadata["compute_processes_before"].splitlines()) == 1
assert len(gpu_metadata["compute_processes_after"].splitlines()) == 1
gpu_samples = []
with (archive / "recursive-proving-gpu.csv").open() as source:
    for row in csv.DictReader(source, skipinitialspace=True):
        gpu_samples.append({
            "unix_seconds": datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp(),
            "device": int(row["index"]),
            "utilization_percent": int(row["utilization.gpu [%]"].split()[0]),
            "memory_mib": int(row["memory.used [MiB]"].split()[0]),
        })
profiles = [{"msm_compute": [], "transfers": [], "distributed_consumers": [], "markers_seconds": {}} for _ in samples]
current = None
for line in log("recursive-proving").splitlines():
    if line.startswith("DEVELOPMENT_RECURSIVE_PROVING_REQUEST_STARTED "):
        current = int(re.search(r"sample=(\d+)", line)[1])
    marker = re.fullmatch(r"(Recursive staging complete|Outer KZG parameters ready|Committing recursive fixed trace [01]|Committing recursive main trace [01]|Prepared recursive trace [01]|Proving recursive KZG wrapper|Recursive stage-and-prove complete): ([0-9.]+)(s|ms|µs|ns)", line)
    if marker:
        assert current is not None
        profiles[current]["markers_seconds"][marker[1]] = float(marker[2]) * {
            "s": 1, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9,
        }[marker[3]]
    if line.startswith("KZG CUDA profile "):
        assert current is not None
        event = dict(re.findall(r"(\w+)=([^ ]+)", line))
        profiles[current]["msm_compute" if event["operation"] == "msm-compute" else "transfers"].append(event)
    consumer = re.search(r"KZG (lookup|quotient) distributed across peer pairs .*rows=(\d+) .*seconds=([0-9.eE+-]+)", line)
    if consumer:
        assert current is not None and "ordinals=[0, 1, 2, 3]" in line
        assert int(consumer[2]) == 2**27
        profiles[current]["distributed_consumers"].append({
            "operation": consumer[1], "rows": int(consumer[2]), "seconds": float(consumer[3]),
        })
gpu = []
for sample, profile in zip(samples, profiles):
    start, end = sample["request_started_unix_seconds"], sample["request_finished_unix_seconds"]
    phase = [row for row in gpu_samples if start <= row["unix_seconds"] <= end]
    assert {int(e["device"]) for e in profile["msm_compute"] if float(e["kernel_ms"]) > 0} == {0, 1, 2, 3}
    devices = {}
    for device in range(4):
        selected = [row for row in phase if row["device"] == device]
        devices[device] = {
            "samples": len(selected),
            "mean_utilization_percent": statistics.mean(row["utilization_percent"] for row in selected),
            "peak_utilization_percent": max(row["utilization_percent"] for row in selected),
            "peak_memory_mib": max(row["memory_mib"] for row in selected),
            "msm_compute_events": sum(int(e["device"]) == device for e in profile["msm_compute"]),
            "msm_kernel_seconds": sum(float(e["kernel_ms"]) for e in profile["msm_compute"] if int(e["device"]) == device) / 1000,
        }
    counters = {key: sum(float(e[key]) for e in profile["transfers"]) for key in (
        "upload_ms", "kernel_ms", "download_ms", "host_copy_ms", "upload_bytes", "download_bytes", "device_copy_bytes")}
    assert {c["operation"] for c in profile["distributed_consumers"]} == {"lookup", "quotient"}
    assert len(profile["distributed_consumers"]) == 2
    markers = profile["markers_seconds"]
    if sample["sample"] == 0:
        assert "Committing recursive fixed trace 0" in markers
    else:
        assert not any(k.startswith("Committing recursive fixed") for k in markers)
    gpu.append({"sample": sample["sample"], "devices": devices, "transfer_counters": counters,
                "distributed_consumers": profile["distributed_consumers"],
                "phase_markers_seconds": markers,
                "scope": "Complete request interval; event intervals overlap across devices and host/kernel regions"})

result = {
    "status": "verified_genuine_development_v4_recursive_proving",
    "scope": "Same saved valid v4 inner proof; two fresh outer assignments/proofs, cold process preparation then retained host frontend/key with device quiescence; no FRI or first-stage rerun",
    "source_input_count": len(after), "source_inputs_sha256": manifest_digest(after),
    "changed_source_inputs": changed,
    "measurement": report,
    "external_wall_seconds": measurement["wall_seconds"],
    "peak_rss_bytes": measurement["peak_rss_bytes"],
    "rss_scope": "Whole process including cold preparation; not isolated warm-request peak",
    "gpu": gpu, "gpu_monitor_reaped": True,
    "cold_fixed_read_decode_seconds": profiles[0]["markers_seconds"]["Committing recursive fixed trace 0"] - profiles[0]["markers_seconds"]["Outer KZG parameters ready"],
    "retained_post_commit_proving_verification_seconds": samples[1]["proof_report"]["total_seconds"] - profiles[1]["markers_seconds"]["Proving recursive KZG wrapper"],
    "proof_bytes": verified["proof_bytes"], "packet_bytes": verified["packet_bytes"],
    "proof_packet_profile_parity": True,
    "independent_cpu_verification_seconds": independent["wall_seconds"],
    "independent_cpu_verification_peak_rss_bytes": independent["peak_rss_bytes"],
    "correctness_tests": test_names, "distinct_correctness_tests_passed": 8,
    "diagnostic_selection_negative_checks": mode_checks,
    "builds": {label: {"wall_seconds": item["wall_seconds"], "peak_rss_bytes": item["peak_rss_bytes"]} for label, item in builds.items()},
    "initial_failed_attempts": {
        "direct_fixture": "Eager witness comparison initially used a loaded System with discarded fixed matrices; comparison now constructs a fresh System with real fixed traces",
        "retry_environment": "Reused sandbox process ID collided with a temporary directory from the failed test; final tests use a fresh TMPDIR without source changes",
        "only_changed_compiled_input": fixture, "runtime_prover_unchanged": True,
        "counted_as_validation": False,
    },
    "binaries": binaries,
    "known_trapdoor": True, "filecoin_acceptance": False,
    "authenticated_filecoin_artifact_available": False,
    "ceremony_final_packet_measured": False,
    "full_chain_measured": False, "goal_complete": False,
}
(archive / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "status": result["status"], "request_seconds": [s["request_seconds"] for s in samples],
    "compile_seconds": report["compile_seconds"], "lowering_seconds": report["lowering_seconds"],
    "proof_bytes": verified["proof_bytes"], "packet_bytes": verified["packet_bytes"],
    "external_wall_seconds": measurement["wall_seconds"],
    "independent_cpu_verification_seconds": independent["wall_seconds"],
}, indent=2))
