#!/usr/bin/env python3
"""Measure Init compression with fresh or cached fixed parameters."""

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


REPO = Path(__file__).resolve().parents[1]
PROOF_FILES = ("proof.compact.bin", "packet.bin", "profile-id.bin")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
CAPABILITY_BINARIES = {
    "first_stage": "init_fri_kzg_prove",
    "wrapper": "init-kzg-wrap",
    "fri": "ix_root",
}
FILECOIN_PUBLIC_MAX_DEGREE = (1 << 28) - 2
TRACE_HEIGHT_LIMIT = 1 << 27
FRI_FILES = ("outer-vk.bin", "outer-proof.bin", "outer-claims.bin")
WORKER_FORMAT = "multi-stark-worker/v1"


def validate_worker_options(*, retained, root_artifacts, seed_fri, seed_claims):
    if retained:
        if root_artifacts is None or seed_fri is None or seed_claims is None:
            raise ValueError("--retained-workers requires --root-artifacts, --worker-seed-fri "
                             "and independently supplied --worker-seed-claims")
    elif seed_fri is not None or seed_claims is not None:
        raise ValueError("worker seed options require --retained-workers")


def validate_development_public_degree_options(*, enabled, setup, retained, acceptance):
    if enabled and (setup != "development" or not retained or acceptance):
        raise ValueError("--development-public-degree requires --setup development and "
                         "--retained-workers; it cannot establish Filecoin --acceptance")


def validate_development_public_degree_reports(reports):
    identities = set()
    for report in reports:
        identity = report.get("public_setup_id")
        if (report.get("setup") != "development" or report.get("development_srs") is not True
                or report.get("known_trapdoor") is not True
                or report.get("filecoin_acceptance") is not False
                or report.get("ceremony_id") is not None
                or report.get("filecoin_manifest_digest") is not None
                or report.get("public_max_degree") != FILECOIN_PUBLIC_MAX_DEGREE
                or not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity)):
            raise ValueError("development public-degree proof has inconsistent setup provenance")
        identities.add(identity)
    if len(identities) != 1:
        raise ValueError("development public-degree stages use different public setup identities")


def expected_statement_bytes(path):
    with path.open("rb") as source:
        data = source.read(161)
    if len(data) != 160:
        raise ValueError(f"expected one independently supplied 18-word root claim: {path}")
    words = [int.from_bytes(data[i:i + 8], "little") for i in range(0, 160, 8)]
    if (words[:2] != [1, 18] or words[2] != 0
            or any(word >= 0xffffffff00000001 for word in words[2:])):
        raise ValueError(f"invalid canonical 18-word root claim: {path}")
    return data


def validate_worker_idle(idle, devices=None):
    if (not isinstance(idle, dict)
            or any(idle.get(key) is not True for key in ("cuda_enabled", "initialized", "quiesced"))
            or not isinstance(idle.get("devices"), list) or not idle["devices"]):
        raise ValueError("worker did not establish initialized CUDA quiescence")
    seen = set()
    byte_fields = ("driver_free_bytes", "total_bytes", "current_pool_reserved_bytes",
                   "current_pool_used_bytes", "default_pool_reserved_bytes", "default_pool_used_bytes",
                   "resident_coefficient_bytes", "srs_point_bytes", "msm_workspace_bytes")
    for item in idle["devices"]:
        if (not isinstance(item, dict) or type(item.get("device")) is not int
                or item["device"] < 0 or item["device"] in seen):
            raise ValueError("invalid or duplicate device in worker quiescence report")
        seen.add(item["device"])
        for key in ("released_srs_point_bytes", "released_msm_workspace_bytes"):
            if type(item.get(key)) is not int or item[key] < 0:
                raise ValueError(f"invalid worker quiescence counter: {key}")
        for when in ("before", "after"):
            snapshot = item.get(when)
            if (not isinstance(snapshot, dict)
                    or any(type(snapshot.get(key)) is not int or snapshot[key] < 0 for key in byte_fields)
                    or type(snapshot.get("current_pool_is_default")) is not bool
                    or snapshot["total_bytes"] <= 0
                    or snapshot["driver_free_bytes"] > snapshot["total_bytes"]
                    or any(snapshot[f"{pool}_used_bytes"] > snapshot[f"{pool}_reserved_bytes"]
                           for pool in ("current_pool", "default_pool"))):
                raise ValueError(f"invalid worker {when} memory snapshot")
        if any(item["after"][key] != 0 for key in
               ("resident_coefficient_bytes", "srs_point_bytes", "msm_workspace_bytes")):
            raise ValueError("worker retained coefficients, SRS points or MSM workspace while idle")
    if devices is not None and seen != set(devices):
        raise ValueError("worker quiescence devices differ from the selected KZG devices")
    return idle


def captured_environment(env):
    return {key: value for key, value in env.items()
            if key.startswith(("MULTI_STARK_KZG_", "MULTI_STARK_CUDA_")) or key in {
                "MULTI_STARK_INIT_EXPECTED_CLAIMS", "MULTI_STARK_SPPARK_PANEL_BYTES",
                "CUDA_VISIBLE_DEVICES", "RAYON_NUM_THREADS", "OMP_NUM_THREADS",
            }}


def phase_environment(env, *, cpu=False, recursive=False, distributed_wrapper=False):
    result = dict(env)
    if cpu:
        result["MULTI_STARK_KZG_BACKEND"] = "cpu"
    elif recursive and distributed_wrapper:
        result.update({
            "MULTI_STARK_KZG_CUDA_RESIDENT_GIB": "8",
            "MULTI_STARK_KZG_CUDA_DISTRIBUTED_QUOTIENT": "1",
            "MULTI_STARK_KZG_CUDA_DISTRIBUTED_LOOKUP": "1",
        })
    return result


def parse_kzg_devices(value):
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not re.fullmatch(r"[0-9]+", part) for part in parts):
        raise ValueError("KZG devices must be comma-separated nonnegative CUDA ordinals")
    devices = [int(part) for part in parts]
    if len(set(devices)) != len(devices) or any(device > (1 << 31) - 1 for device in devices):
        raise ValueError("KZG device ordinals must be distinct and fit in i32")
    return devices


def validate_acceptance_options(*, acceptance, setup, root_artifacts, devices):
    if not acceptance:
        return
    if setup != "filecoin" or root_artifacts is None:
        raise ValueError("--acceptance requires --setup filecoin and --root-artifacts")
    if (devices is None or len(devices) != 4 or len(set(devices)) != 4
            or any(type(device) is not int or not 0 <= device <= (1 << 31) - 1 for device in devices)):
        raise ValueError("--acceptance requires four distinct explicit --kzg-devices "
                         "or MULTI_STARK_KZG_CUDA_DEVICES ordinals")


def validate_capabilities(capabilities, role, *, acceptance=False):
    if (not isinstance(capabilities, dict)
            or capabilities.get("format") != "multi-stark-capabilities/v1"
            or capabilities.get("binary") != CAPABILITY_BINARIES[role]
            or any(type(capabilities.get(key)) is not bool for key in
                   ("kzg", "kzg_cuda", "goldilocks_cuda", "parallel"))):
        raise ValueError(f"{role}: missing or invalid compiled capability report")
    required = ["goldilocks_cuda"] if role == "fri" else ["kzg", "kzg_cuda"]
    if acceptance:
        required.append("parallel")
    missing = [key for key in required if not capabilities[key]]
    if missing:
        raise ValueError(f"{role}: executable lacks compiled features: {', '.join(missing)}")
    return capabilities


def read_capabilities(binary, role, env, *, acceptance=False):
    try:
        result = subprocess.run([str(binary), "capabilities"], cwd=REPO, env=env,
                                check=True, capture_output=True, text=True, timeout=15)
        if len(result.stdout.splitlines()) != 1:
            raise ValueError("expected one JSON line")
        capabilities = json.loads(result.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        raise ValueError(f"{role}: capability preflight failed for {binary}: {error}") from error
    return validate_capabilities(capabilities, role, acceptance=acceptance)


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def benchmark_sources():
    paths = {
        REPO / ".cargo/config.toml", REPO / "Cargo.toml", REPO / "Cargo.lock",
        REPO / "build.rs", REPO / "experiments/kzg-wrap/Cargo.toml",
        REPO / "experiments/kzg-wrap/Cargo.lock", Path(__file__).resolve(),
    }
    for directory in ("src", "examples", "experiments/kzg-wrap/src"):
        paths.update((REPO / directory).rglob("*.rs"))
    for suffix in ("cu", "cuh", "cpp", "h"):
        paths.update((REPO / "cuda").rglob(f"*.{suffix}"))
    return sorted(paths)


def summarize_cuda_events(path, *, with_quality=False):
    operations = {}
    devices = {}
    skipped = []
    parsed = 0
    prefix = "KZG CUDA profile "
    metrics = ("upload_ms", "kernel_ms", "download_ms", "host_copy_ms",
               "upload_bytes", "download_bytes", "device_copy_bytes", "call_ms")
    required = {"device", "operation", "elements", *metrics[:-1]}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if prefix not in line:
            continue
        pairs = re.findall(r"(\w+)=([^ ]+)", line)
        fields = dict(pairs)
        reason = None
        if not line.startswith(prefix) or line.count(prefix) != 1:
            reason = "interleaved record prefix"
        elif missing := required - fields.keys():
            reason = "missing fields: " + ", ".join(sorted(missing))
        else:
            invalid = [key for key in ("device", "elements", *metrics) if key in fields
                       and (not re.fullmatch(NUMBER, fields[key])
                            or not math.isfinite(float(fields[key])) or float(fields[key]) < 0)]
            if invalid:
                reason = "invalid numeric fields: " + ", ".join(invalid)
            elif any(not re.fullmatch(r"[0-9]+", fields[key]) for key in ("device", "elements")):
                reason = "device and elements must be integers"
            elif len(pairs) != len(fields):
                reason = "duplicate fields"
            elif not re.fullmatch(r"(?:\w+=[^\s=]+)(?: \w+=[^\s=]+)*", line[len(prefix):]):
                reason = "interleaved or malformed record text"
        if reason:
            if not with_quality:
                raise ValueError(f"invalid CUDA profile record at line {number}: {reason}")
            skipped.append({"line": number, "reason": reason, "text": line})
            continue
        parsed += 1
        operation = operations.setdefault(fields["operation"], {"calls": 0})
        operation["calls"] += 1
        for key in metrics:
            if key in fields:
                operation[key] = operation.get(key, 0) + float(fields[key])
        device = devices.setdefault(str(int(fields["device"])), {
            "records": 0, "kernel_records": 0, "kernel_ms": 0.0,
        })
        device["records"] += 1
        device["kernel_records"] += float(fields["kernel_ms"]) > 0
        device["kernel_ms"] += float(fields["kernel_ms"])
    quality = {
        "totals_complete": not skipped,
        "records_seen": parsed + len(skipped),
        "records_parsed": parsed,
        "records_skipped": len(skipped),
        "skipped": skipped,
        "devices": devices,
    }
    return (operations, quality) if with_quality else operations


def summarize_host_operations(path):
    timers = {
        "KZG lookup committed": ("evaluation_seconds", "trace_seconds", "commit_seconds"),
        "KZG quotient constraints evaluated": ("selector_seconds", "constraint_seconds"),
        "KZG quotient evaluations materialized": ("seconds",),
        "KZG quotient coset evaluated": ("lde_seconds",),
        "KZG evaluation reconstruction": ("transform_seconds", "transpose_seconds"),
    }
    operations = {}
    for line in path.read_text().splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        for label, keys in timers.items():
            if label not in line:
                continue
            fields = dict(re.findall(r"\b(\w+)=(" + NUMBER + r")(?=\s|$)", line))
            operation = operations.setdefault(label, {"calls": 0})
            operation["calls"] += 1
            for key in keys:
                operation[key] = operation.get(key, 0) + float(fields[key])
    return operations


def proof_hashes(directory):
    return {name: {"sha256": sha256(directory / name),
                   "bytes": (directory / name).stat().st_size} for name in PROOF_FILES}


def record_proof_evidence(report, name, directory):
    report.setdefault("proofs", {})[name] = proof_hashes(directory)


def read_verification_report(directory, name):
    path = directory / f"{name}-report.json"
    report = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(report, dict):
        raise ValueError(f"verification report must be a JSON object: {path}")
    return report


def add_log_evidence(measured, log_path):
    log_text = log_path.read_text()
    measured["fixed_cache_hits"] = log_text.count("Reused fixed preprocessing:")
    measured["fixed_cache_misses"] = log_text.count("Fixed preprocessing cache miss:")
    measured["srs_cache_hits"] = log_text.count("Loaded development SRS cache")
    measured["srs_cache_misses"] = log_text.count("Generated and cached development SRS")
    (measured["cuda_operation_totals_overlap"],
     measured["cuda_profile_parse"]) = summarize_cuda_events(log_path, with_quality=True)
    measured["host_operation_totals_overlap"] = summarize_host_operations(log_path)


def export_inner_proof(source, destination):
    destination.mkdir()
    files = [source / "manifest.bin", source / "kzg-setup-id.bin",
             *sorted((source / "kzg").glob("setup-*.bin")), source / "kzg/proof.compact.bin"]
    for path in files:
        shutil.copyfile(path, destination / path.name)


def worker_input_hashes(role, directory):
    names = (FRI_FILES if role == "first_stage" else
             ("manifest.bin", "kzg-setup-id.bin", "proof.compact.bin",
              *(path.name for path in sorted(directory.glob("setup-*.bin")))))
    return {name: sha256(directory / name) for name in names}


def worker_output_identity(role, directory):
    frontend = "plan-id.bin" if role == "first_stage" else "frontend-id.bin"
    identity = {}
    for name, path in (("frontend", directory / frontend),
                       ("setup", directory / "kzg-setup-id.bin"),
                       ("profile", directory / "kzg/profile-id.bin")):
        with path.open("rb") as source:
            value = source.read(33)
        if len(value) != 32:
            raise ValueError(f"worker {name} identity must have 32 bytes: {path}")
        identity[name] = value.hex()
    return identity


def worker_identity_matches(role, expected, actual, *, same_statement):
    keys = ("frontend", "setup")
    # The first-stage transport profile includes the public statement.
    if role != "first_stage" or same_statement:
        keys += ("profile",)
    return (isinstance(expected, dict) and isinstance(actual, dict)
            and all(key in expected and key in actual and expected[key] == actual[key] for key in keys))


def snapshot_worker_seed(seed_fri, seed_claims, directory):
    directory.mkdir()
    fri = directory / "seed-fri"
    fri.mkdir()
    expected_hashes = worker_input_hashes("first_stage", seed_fri)
    for name in FRI_FILES:
        shutil.copyfile(seed_fri / name, fri / name)
    require_hashes(worker_input_hashes("first_stage", fri), expected_hashes, "worker seed snapshot")
    claims = directory / "expected-claims.bin"
    claims.write_bytes(expected_statement_bytes(seed_claims))
    return fri, claims, {"fri_source": str(seed_fri), "expected_claims_source": str(seed_claims),
                         "fri_snapshot": str(fri), "expected_claims_snapshot": str(claims),
                         "input_hashes": expected_hashes, "expected_claims_sha256": sha256(claims)}


def evaluate_goal(report):
    checks = {}

    def check(name, passed, observed, required):
        checks[name] = {"passed": bool(passed), "observed": observed, "required": required}

    def hex_digest(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    def positive_integer(value):
        return type(value) is int and value > 0

    def admitted_heights(heights):
        return (isinstance(heights, list) and bool(heights)
                and all(positive_integer(height) and height >= 2 and height <= TRACE_HEIGHT_LIMIT
                        and height & (height - 1) == 0 for height in heights))

    def successful_phase(name, backend):
        phase = report.get("phases", {}).get(name, {})
        return phase.get("exit_code") == 0 and phase.get("backend") == backend

    check("fresh_root_boundary",
          report.get("fri_compression_rerun") is True
          and report.get("fresh_output_directory") is True
          and successful_phase("fri_compress", "cuda")
          and all(hex_digest(report.get("root_artifacts", {}).get(name)) for name in
                  ("root-vk.bin", "root-proof.bin", "root-claims.bin")),
          report.get("timing_boundary"), "fresh aggregation root to verified final packet")
    wall = report.get("pipeline_wall_seconds")
    check("pipeline_under_300_seconds",
          type(wall) in (int, float) and math.isfinite(wall) and 0 < wall < 300,
          wall, "0 < pipeline_wall_seconds < 300; independent CPU checks are additional")
    capabilities = report.get("capabilities", {})
    try:
        for role in CAPABILITY_BINARIES:
            validate_capabilities(capabilities.get(role), role, acceptance=True)
        capable = True
    except ValueError:
        capable = False
    check("compiled_cuda_capabilities", capable, capabilities,
          "parallel CUDA-capable first-stage, wrapper and FRI executables")
    devices = report.get("kzg_devices")
    device_ids_valid = (isinstance(devices, list) and len(devices) == 4
                        and all(type(device) is int and 0 <= device <= (1 << 31) - 1
                                for device in devices) and len(set(devices)) == 4)
    check("four_explicit_kzg_devices", device_ids_valid, devices,
          "four distinct CUDA ordinals under CUDA_VISIBLE_DEVICES")
    expected_devices = set(map(str, devices)) if device_ids_valid else set()
    fused = report.get("fused_staging") is True
    witness_phases = (("fri_stage_and_prove", "recursive_stage_and_prove") if fused
                      else ("fri_stage", "recursive_stage"))
    check("fresh_kzg_witness_stages", all(successful_phase(name, "cuda") for name in witness_phases),
          list(witness_phases), "both fresh witness-generation commands complete in the new output directory")
    for stage, phase in (("fri_to_kzg", "fri_stage_and_prove" if fused else "fri_prove"),
                         ("recursive_kzg", "recursive_stage_and_prove" if fused else "recursive_prove")):
        events = report.get("phases", {}).get(phase, {}).get("cuda_profile_parse", {})
        observed = events.get("devices", {})
        active = {device for device, data in observed.items()
                  if positive_integer(data.get("kernel_records"))
                  and type(data.get("kernel_ms")) in (int, float)
                  and math.isfinite(data["kernel_ms"]) and data["kernel_ms"] > 0}
        check(f"{stage}_four_device_kernels",
              device_ids_valid and successful_phase(phase, "cuda")
              and active == expected_devices,
              {"devices": observed, "records_skipped": events.get("records_skipped")},
              "valid positive kernel events on exactly the four selected devices; "
              "malformed records and utilization samples are not execution evidence")

    first = report.get("first_stage_verification", {})
    outer = report.get("recursive_verification", {})
    first_cpu = report.get("first_stage_cpu_verification", {})
    outer_cpu = report.get("recursive_cpu_verification", {})
    receipt = report.get("environment", {}).get("MULTI_STARK_KZG_FILECOIN_DIGEST")
    ceremony = first.get("ceremony_id")
    setup_reports = {"first": first, "outer": outer, "first_cpu": first_cpu, "outer_cpu": outer_cpu}
    authenticated = (report.get("setup") == "filecoin" and hex_digest(receipt)
                     and hex_digest(ceremony)
                     and all(item.get("setup") == "filecoin"
                             and item.get("development_srs") is False
                             and item.get("filecoin_manifest_digest") == receipt
                             and item.get("ceremony_id") == ceremony
                             and item.get("public_max_degree") == FILECOIN_PUBLIC_MAX_DEGREE
                             and hex_digest(item.get("setup_id"))
                             for item in setup_reports.values()))
    check("both_stages_authenticated_filecoin", authenticated,
          {name: {key: item.get(key) for key in
                  ("setup", "development_srs", "filecoin_manifest_digest", "ceremony_id",
                   "public_max_degree", "setup_id")} for name, item in setup_reports.items()},
          "both proving and CPU verification use the pinned receipt, same ceremony and full public degree")
    first_heights, outer_heights = first.get("trace_heights"), outer.get("trace_heights")
    check("stage_one_trace_cap", admitted_heights(first_heights), first_heights,
          "every actual stage-one trace is a power of two in [2, 2^27]")
    check("single_wrapper_trace_cap",
          admitted_heights(outer_heights) and len(outer_heights) == 2
          and outer_heights[1] <= outer_heights[0], outer_heights,
          "one computation and one merged table, each <=2^27; table <= computation")
    profile_keys = ("setup", "setup_id", "filecoin_manifest_digest", "ceremony_id",
                    "public_max_degree", "trace_heights", "proof_bytes", "packet_bytes")
    for stage, proved, verified in (("fri_to_kzg", first, first_cpu),
                                    ("recursive_kzg", outer, outer_cpu)):
        artifacts = report.get("proofs", {}).get(stage, {})
        recorded = all(isinstance(artifacts.get(name), dict)
                       and hex_digest(artifacts[name].get("sha256"))
                       and positive_integer(artifacts[name].get("bytes")) for name in PROOF_FILES)
        check(f"{stage}_artifacts_and_profile",
              recorded and artifacts["profile-id.bin"]["bytes"] == 32
              and proved.get("proof_bytes") == artifacts["proof.compact.bin"]["bytes"]
              and proved.get("packet_bytes") == artifacts["packet.bin"]["bytes"]
              and proved.get("setup_id") == report.get("setup_bindings", {}).get(stage)
              and all(key in proved and proved[key] == verified.get(key) for key in profile_keys),
              artifacts, "hash all three artifacts; lengths, native reports and that stage's setup binding agree")
    packet = report.get("proofs", {}).get("recursive_kzg", {}).get("packet.bin", {}).get("bytes")
    check("final_packet_below_3000_bytes", positive_integer(packet) and packet < 3000,
          packet, "actual complete packet, strictly <3000 bytes")
    check("native_verification_and_negative_checks",
          all(item.get("native_verification_passed") is True
              and item.get("negative_tests_pass") is True for item in setup_reports.values())
          and outer.get("external_pairings_pass") is True
          and outer_cpu.get("external_pairings_pass") is True,
          {name: {key: item.get(key) for key in
                  ("native_verification_passed", "negative_tests_pass", "external_pairings_pass")}
           for name, item in setup_reports.items()},
          "native verification, altered statements/transport and both external pairing checks")
    cpu_phases = ("fri_cpu_verify", "recursive_cpu_verify")
    check("independent_cpu_verification",
          all(successful_phase(name, "cpu")
              and report["phases"][name].get("environment", {}).get("MULTI_STARK_KZG_BACKEND") == "cpu"
              for name in cpu_phases), list(cpu_phases), "both independent CPU processes exit successfully")
    if report.get("retained_workers") is True:
        startup = report.get("worker_startup", {})
        valid = (startup.get("completed") is True
                 and report.get("latency_boundary") == "retained_worker_fresh_request"
                 and fused and type(startup.get("wall_seconds")) in (int, float)
                 and math.isfinite(startup["wall_seconds"]) and startup["wall_seconds"] > 0)
        request_ids, outputs = set(), set()
        for role, stage, phase in (("first_stage", "fri_to_kzg", "fri_stage_and_prove"),
                                   ("wrapper", "recursive_kzg", "recursive_stage_and_prove")):
            worker = startup.get("workers", {}).get(role, {})
            listening = worker.get("listening", {})
            cold = startup.get("phases", {}).get(f"startup_{phase}", {})
            warm = report.get("phases", {}).get(phase, {})
            valid = (valid and listening.get("format") == WORKER_FORMAT
                     and listening.get("binary") == CAPABILITY_BINARIES[role]
                     and listening.get("status") == "listening" and listening.get("prepared") is False
                     and listening.get("id") is None and positive_integer(worker.get("pid"))
                     and worker.get("shutdown", {}).get("exit_code") == 0
                     and worker_identity_matches(
                         role, cold.get("output_identity"), warm.get("output_identity"),
                         same_statement=cold.get("expected_claims_sha256") == warm.get("expected_claims_sha256"))
                     and warm.get("output_identity", {}).get("setup") == report.get("setup_bindings", {}).get(stage))
            for phase_record, reused in ((cold, False), (warm, True)):
                request = phase_record.get("worker_request", {})
                response = phase_record.get("worker_response", {})
                identity = phase_record.get("output_identity", {})
                request_id, destination = request.get("id"), request.get("output")
                valid = (valid and isinstance(request_id, str) and bool(request_id)
                         and request_id not in request_ids and isinstance(destination, str) and bool(destination)
                         and destination not in outputs and phase_record.get("exit_code") == 0
                         and phase_record.get("backend") == "cuda"
                         and phase_record.get("worker_pid") == worker.get("pid")
                         and all(hex_digest(identity.get(key)) for key in ("frontend", "setup", "profile"))
                         and response.get("format") == WORKER_FORMAT
                         and response.get("binary") == CAPABILITY_BINARIES[role]
                         and response.get("status") == "proved" and response.get("prepared") is True
                         and response.get("id") == request_id and response.get("output") == destination
                         and response.get("frontend_reused") is reused
                         and response.get("loaded_key_reused") is reused
                         and bool(phase_record.get("input_hashes"))
                         and all(hex_digest(value) for value in phase_record.get("input_hashes", {}).values())
                         and hex_digest(phase_record.get("expected_claims_sha256")))
                if isinstance(request_id, str):
                    request_ids.add(request_id)
                if isinstance(destination, str):
                    outputs.add(destination)
                try:
                    validate_worker_idle(response.get("idle"), devices)
                except ValueError:
                    valid = False
            start, end = warm.get("worker_log_start_byte"), cold.get("worker_log_end_byte")
            valid = (valid and type(start) is int and type(end) is int and start >= end >= 0
                     and warm.get("expected_claims_sha256") == report.get("root_artifacts", {}).get("root-claims.bin")
                     and cold.get("expected_claims_sha256") == startup.get("seed", {}).get("expected_claims_sha256"))
            if role == "first_stage":
                valid = (valid and warm.get("input_hashes") == report.get("fri_input")
                         and cold.get("input_hashes") == startup.get("seed", {}).get("input_hashes"))
            else:
                valid = (valid and warm.get("input_hashes", {}).get("proof.compact.bin")
                         == report.get("proofs", {}).get("fri_to_kzg", {}).get("proof.compact.bin", {}).get("sha256")
                         and cold.get("input_hashes", {}).get("proof.compact.bin")
                         == startup.get("proofs", {}).get("fri_to_kzg", {}).get("proof.compact.bin", {}).get("sha256"))
        check("retained_worker_preparation_and_reuse", valid,
              {"startup_wall_seconds": startup.get("wall_seconds"), "request_ids": sorted(request_ids),
               "outputs": sorted(outputs), "latency_boundary": report.get("latency_boundary")},
              "two separately recorded genuine warmups, same prepared worker/profile, fresh input hashes, "
              "disjoint request logs, successful GPU quiescence and clean worker exits")
    failures = [name for name, result in checks.items() if not result["passed"]]
    return {"requested": True, "passed": not failures, "checks": checks, "failures": failures}


def require_hashes(actual, expected, label):
    if actual != expected:
        different = sorted(name for name in actual.keys() | expected.keys()
                           if actual.get(name) != expected.get(name))
        raise RuntimeError(f"{label} mismatch: {', '.join(different)}")


def load_baseline(directory):
    path = directory / "report.json"
    report = json.loads(path.read_text())
    if not report["status"].startswith("verified_"):
        raise ValueError("baseline must be a successfully verified run")
    return {
        "directory": str(directory),
        "report_sha256": sha256(path),
        "root_artifacts": report.get("root_artifacts"),
        "fri_input": report["fri_input"],
        "proofs": {"fri_to_kzg": proof_hashes(directory / "fri-to-kzg/kzg"),
                   "recursive_kzg": proof_hashes(directory / "recursive/kzg")},
    }


def summarize_gpu_samples(path, started_at=None, finished_at=None):
    sampled = {}
    with path.open() as samples:
        for row in csv.reader(samples):
            if len(row) != 4:
                continue
            timestamp, index, memory, utilization = (value.strip() for value in row)
            if not memory.isdigit() or not utilization.isdigit():
                continue
            if started_at is not None:
                timestamp = datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                if not started_at <= timestamp <= finished_at:
                    continue
            memory, utilization = int(memory), int(utilization)
            gpu = sampled.setdefault(index, {
                "peak_memory_mib": 0, "peak_utilization_percent": 0,
                "sample_count": 0, "utilization_sum": 0, "zero_utilization_samples": 0,
            })
            gpu["peak_memory_mib"] = max(gpu["peak_memory_mib"], memory)
            gpu["peak_utilization_percent"] = max(gpu["peak_utilization_percent"], utilization)
            gpu["sample_count"] += 1
            gpu["utilization_sum"] += utilization
            gpu["zero_utilization_samples"] += utilization == 0
    for gpu in sampled.values():
        gpu["mean_utilization_percent"] = gpu.pop("utilization_sum") / gpu["sample_count"]
        gpu["zero_utilization_sample_percent"] = 100 * gpu["zero_utilization_samples"] / gpu["sample_count"]
    return sampled


def terminate_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A timing wrapper can exit before its child, so kill any descendants
        # still in the private process group even if the wrapper has finished.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


class WorkerClient:
    def __init__(self, binary, role, directory, env, devices=None, *, timeout=3600,
                 development_public_degree=False):
        self.role, self.env, self.devices, self.timeout = role, env, devices, timeout
        self.directory = directory
        directory.mkdir()
        self.log_path = directory / "worker.log"
        self.responses_path = directory / "responses.jsonl"
        self.requests_path = directory / "requests.jsonl"
        self.command = [str(binary), "dev-v4-serve" if development_public_degree else "serve",
                        str(self.responses_path)]
        self.response_offset = 0
        self.ids = set()
        self.prepared_identity = None
        self.statement_profiles = {}
        self.last_request = None
        self.closed = False
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.log = self.log_path.open("xb")
        try:
            self.process = subprocess.Popen(self.command, cwd=REPO, env=env, stdin=subprocess.PIPE,
                                            stdout=self.log, stderr=subprocess.STDOUT,
                                            text=True, start_new_session=True)
        except BaseException:
            self.log.close()
            raise
        try:
            ready = self.read_response(time.monotonic() + min(timeout, 30))
            if (ready.get("status") != "listening" or ready.get("id") is not None
                    or ready.get("prepared") is not False):
                raise ValueError("worker listening response must explicitly be unprepared")
            self.startup = {
                "command": self.command, "pid": self.process.pid,
                "environment": captured_environment(env), "listening": ready,
                "spawn_to_listening_seconds": time.monotonic() - self.started,
                "started_at_utc": self.started_at,
                "listening_at_utc": datetime.now(timezone.utc).isoformat(),
                "responses": str(self.responses_path), "requests": str(self.requests_path),
                "log": str(self.log_path), "memory_at_listening": self.memory_snapshot(),
            }
        except BaseException:
            self.close(graceful=False)
            raise

    def memory_snapshot(self):
        try:
            fields = dict(line.split(":", 1) for line in
                          Path(f"/proc/{self.process.pid}/status").read_text().splitlines() if ":" in line)
            return {key: int(fields[name].split()[0]) * 1024 for key, name in
                    (("rss_bytes", "VmRSS"), ("process_lifetime_peak_rss_bytes", "VmHWM")) if name in fields}
        except (OSError, ValueError):
            return {}

    def read_response(self, deadline):
        def unique_fields(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate worker response field: {key}")
                result[key] = value
            return result

        next_progress = time.monotonic() + 30
        while True:
            try:
                with self.responses_path.open("rb") as responses:
                    responses.seek(self.response_offset)
                    line = responses.readline(65537)
            except FileNotFoundError:
                line = b""
            if len(line) > 65536:
                raise ValueError("worker response exceeds 64 KiB")
            if line.endswith(b"\n"):
                response = json.loads(line, object_pairs_hook=unique_fields)
                self.response_offset += len(line)
                if (not isinstance(response, dict) or response.get("format") != WORKER_FORMAT
                        or response.get("binary") != CAPABILITY_BINARIES[self.role]):
                    raise ValueError("worker response has the wrong format or binary")
                if response.get("status") == "failed":
                    raise RuntimeError(f"worker failed: {response}")
                return response
            if self.process.poll() is not None:
                raise RuntimeError(f"worker exited {self.process.returncode} before a complete response; "
                                   f"see {self.log_path}")
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f"worker response timed out; see {self.log_path}")
            if now >= next_progress:
                print(f"Waiting for {self.role} worker; log: {self.log_path}", flush=True)
                next_progress = now + 30
            time.sleep(min(0.02, deadline - now))

    def request(self, request_id, input_dir, output_dir, expected_claims, log_path, *, reused):
        if request_id in self.ids:
            raise ValueError(f"duplicate worker request id: {request_id}")
        if output_dir.exists():
            raise ValueError(f"worker output must be new: {output_dir}")
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("worker is not running")
        self.ids.add(request_id)
        started = time.monotonic()
        log_start = self.log_path.stat().st_size
        request = {"id": request_id, "input": str(input_dir), "output": str(output_dir),
                   "expected_claims": str(expected_claims)}
        measured = {
            "worker_request": request, "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": self.command, "worker_pid": self.process.pid,
            "backend": "cuda", "environment": captured_environment(self.env),
            "completion_kind": "worker_response", "exit_code": None, "log": str(log_path),
            "worker_log": str(self.log_path), "worker_log_start_byte": log_start,
            "memory_before": self.memory_snapshot(),
        }
        self.last_request = measured
        try:
            claims = expected_statement_bytes(expected_claims)
            measured["expected_claims_sha256"] = hashlib.sha256(claims).hexdigest()
            measured["input_hashes"] = worker_input_hashes(self.role, input_dir)
            with self.requests_path.open("a") as requests:
                requests.write(json.dumps(request) + "\n")
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            deadline = started + self.timeout
            event = self.read_response(deadline)
            measured["request_started_response"] = event
            if (event.get("id") != request_id or event.get("status") != "request_started"
                    or event.get("prepared") is not reused or event.get("output") != str(output_dir)):
                raise ValueError("worker did not acknowledge the expected request id")
            response = self.read_response(deadline)
            measured["worker_response"] = response
            if (response.get("id") != request_id or response.get("status") != "proved"
                    or response.get("prepared") is not True
                    or response.get("output") != str(output_dir)
                    or response.get("frontend_reused") is not reused
                    or response.get("loaded_key_reused") is not reused
                    or any(type(response.get(key)) not in (int, float)
                           or not math.isfinite(response[key]) or response[key] < 0
                           for key in ("preparation_seconds", "request_seconds"))):
                raise ValueError("worker proof response has invalid request, reuse or timing evidence")
            validate_worker_idle(response.get("idle"), self.devices)
            if expected_statement_bytes(expected_claims) != claims:
                raise ValueError("independently supplied statement changed during the request")
            require_hashes(worker_input_hashes(self.role, input_dir), measured["input_hashes"],
                           "worker input changed during request")
            identity = worker_output_identity(self.role, output_dir)
            measured["output_identity"] = identity
            statement = measured["expected_claims_sha256"]
            expected_identity = dict(self.prepared_identity or {})
            if statement in self.statement_profiles:
                expected_identity["profile"] = self.statement_profiles[statement]
            if reused and not worker_identity_matches(
                    self.role, expected_identity, identity, same_statement=statement in self.statement_profiles):
                raise ValueError("worker frontend, setup or proof profile changed after preparation")
            self.prepared_identity = identity
            self.statement_profiles[statement] = identity["profile"]
            if self.process.poll() is not None:
                raise RuntimeError("worker exited after proof instead of retaining its prepared key")
            measured["exit_code"] = 0
            return measured
        except BaseException as error:
            measured["error"] = str(error)
            raise
        finally:
            measured["wall_seconds"] = time.monotonic() - started
            measured["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            measured["memory_after"] = self.memory_snapshot()
            end = self.log_path.stat().st_size
            measured["worker_log_end_byte"] = end
            with self.log_path.open("rb") as source, log_path.open("xb") as destination:
                source.seek(log_start)
                remaining = end - log_start
                while remaining:
                    chunk = source.read(min(remaining, 1 << 20))
                    if not chunk:
                        raise RuntimeError("worker log truncated while capturing request evidence")
                    destination.write(chunk)
                    remaining -= len(chunk)
            add_log_evidence(measured, log_path)

    def close(self, *, graceful=True):
        if self.closed:
            return
        self.closed = True
        self.shutdown = {"memory_before_exit": self.memory_snapshot()}
        try:
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
            if graceful:
                try:
                    self.process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    terminate_process_group(self.process)
                    raise RuntimeError("worker did not exit after request input closed")
                if self.process.returncode:
                    raise RuntimeError(f"worker exited {self.process.returncode}")
            else:
                terminate_process_group(self.process)
        except BaseException:
            if self.process.poll() is None:
                terminate_process_group(self.process)
            raise
        finally:
            self.shutdown.update(exit_code=self.process.poll(),
                                 process_lifetime_seconds=time.monotonic() - self.started)
            self.log.close()


def interrupt(signum, _frame):
    raise InterruptedError(f"Received {signal.Signals(signum).name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "target/kzg-cuda-run")
    parser.add_argument("--first-stage-binary", type=Path,
                        default=REPO / "target/release/examples/init_fri_kzg_prove")
    parser.add_argument("--wrapper-binary", type=Path,
                        default=REPO / "experiments/kzg-wrap/target/release/init-kzg-wrap")
    parser.add_argument("--fri-binary", type=Path, default=REPO / "target/release/examples/ix_root")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--root-artifacts", type=Path,
                        help="Measure root FRI compression before both KZG stages")
    inputs.add_argument("--fri-artifacts", type=Path,
                        help="Use a regenerated compressed FRI proof without historical byte comparison")
    parser.add_argument("--fixed-cache", type=Path, help="Trusted local fixed-preprocessing cache")
    parser.add_argument("--setup", choices=("filecoin", "development"),
                        default=os.environ.get("MULTI_STARK_KZG_SETUP", "filecoin"))
    parser.add_argument("--filecoin-cache", type=Path,
                        help="Authenticated normalized challenge_19 cache")
    parser.add_argument("--filecoin-digest", help="Externally pinned import receipt (64 hex digits)")
    parser.add_argument("--srs-cache", type=Path, help="Trusted local development SRS cache")
    parser.add_argument("--fused-staging", action="store_true",
                        help="Generate both KZG stages' witness traces directly into commitment")
    parser.add_argument("--retained-workers", action="store_true",
                        help="Warm two retained workers, then time a fresh root-to-packet request separately")
    parser.add_argument("--development-public-degree", action="store_true",
                        help="Explicit known-trapdoor v4 worker diagnostic; cannot establish Filecoin acceptance")
    parser.add_argument("--worker-seed-fri", type=Path,
                        help="Verified compressed FRI input for the first worker's genuine warmup proof")
    parser.add_argument("--worker-seed-claims", type=Path,
                        help="Independently supplied 18-word expected statement for the warmup input")
    parser.add_argument("--distributed-wrapper", action="store_true",
                        help="Enable four-GPU outer lookup/quotient consumers with an 8 GiB/card coefficient cap")
    parser.add_argument("--kzg-devices", help="Comma-separated CUDA ordinals under CUDA_VISIBLE_DEVICES")
    parser.add_argument("--acceptance", action="store_true",
                        help="Require the full Filecoin chain, four-device kernel evidence and the <300s goal")
    parser.add_argument("--baseline", type=Path,
                        help="Verified run or recovery directory with identical inputs and proof bytes")
    args = parser.parse_args()
    try:
        validate_development_public_degree_options(
            enabled=args.development_public_degree, setup=args.setup,
            retained=args.retained_workers, acceptance=args.acceptance)
        validate_worker_options(retained=args.retained_workers, root_artifacts=args.root_artifacts,
                                seed_fri=args.worker_seed_fri, seed_claims=args.worker_seed_claims)
        if args.retained_workers:
            args.worker_seed_fri = args.worker_seed_fri.resolve()
            args.worker_seed_claims = args.worker_seed_claims.resolve()
            expected_statement_bytes(args.worker_seed_claims)
            expected_statement_bytes(args.root_artifacts / "root-claims.bin")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.setup == "filecoin" and args.srs_cache:
        parser.error("--srs-cache requires --setup development")
    baseline = load_baseline(args.baseline.resolve()) if args.baseline else None
    if baseline and args.root_artifacts:
        actual = {name: sha256(args.root_artifacts / name) for name in
                  ("root-vk.bin", "root-proof.bin", "root-claims.bin")}
        require_hashes(actual, baseline["root_artifacts"] or {}, "baseline root input")
    signal.signal(signal.SIGTERM, interrupt)
    env = dict(os.environ, MULTI_STARK_KZG_BACKEND="cuda", MULTI_STARK_KZG_SETUP=args.setup)
    if args.filecoin_cache:
        env["MULTI_STARK_KZG_FILECOIN_CACHE"] = str(args.filecoin_cache.resolve())
    if args.filecoin_digest:
        env["MULTI_STARK_KZG_FILECOIN_DIGEST"] = args.filecoin_digest
    if args.setup == "filecoin" and not all(env.get(key) for key in (
            "MULTI_STARK_KZG_FILECOIN_CACHE", "MULTI_STARK_KZG_FILECOIN_DIGEST")):
        parser.error("Filecoin runs require an authenticated cache and an externally pinned digest")
    if args.setup == "development" and any(env.get(key) for key in (
            "MULTI_STARK_KZG_FILECOIN_CACHE", "MULTI_STARK_KZG_FILECOIN_DIGEST")):
        parser.error("development mode cannot also select a Filecoin setup")
    if args.setup == "filecoin":
        pin = env["MULTI_STARK_KZG_FILECOIN_DIGEST"]
        if not re.fullmatch(r"[0-9a-fA-F]{64}", pin):
            parser.error("Filecoin digest must be an externally pinned 64-digit hexadecimal receipt")
        env["MULTI_STARK_KZG_FILECOIN_DIGEST"] = pin.lower()
    device_selection = (args.kzg_devices if args.kzg_devices is not None
                        else env.get("MULTI_STARK_KZG_CUDA_DEVICES"))
    try:
        devices = parse_kzg_devices(device_selection) if device_selection is not None else None
        validate_acceptance_options(acceptance=args.acceptance, setup=args.setup,
                                    root_artifacts=args.root_artifacts, devices=devices)
    except ValueError as error:
        parser.error(str(error))
    if devices is not None:
        env["MULTI_STARK_KZG_CUDA_DEVICES"] = ",".join(map(str, devices))
    if args.acceptance:
        env["MULTI_STARK_KZG_CUDA_PROFILE"] = "1"
    if args.retained_workers:
        env["MULTI_STARK_KZG_CUDA_PROFILE"] = "1"
    if args.fixed_cache:
        env["MULTI_STARK_KZG_FIXED_CACHE"] = str(args.fixed_cache.resolve())
    if args.srs_cache:
        env["MULTI_STARK_KZG_DEV_SRS_CACHE"] = str(args.srs_cache.resolve())
    if args.root_artifacts:
        args.root_artifacts = args.root_artifacts.resolve()
        env["MULTI_STARK_INIT_EXPECTED_CLAIMS"] = str(args.root_artifacts / "root-claims.bin")
    binary_paths = {"first_stage": args.first_stage_binary.resolve(), "wrapper": args.wrapper_binary.resolve()}
    if args.root_artifacts:
        binary_paths["fri"] = args.fri_binary.resolve()
    for binary in binary_paths.values():
        if not binary.is_file():
            parser.error(f"Build or select the CUDA-capable executable first: {binary}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary_dir = output / "measured-binaries"
    binary_dir.mkdir()
    binary_hashes, snapshots, snapshot_hashes, capabilities = {}, {}, {}, {}
    try:
        for role, binary in binary_paths.items():
            digest = sha256(binary)
            label = str(binary.relative_to(REPO)) if binary.is_relative_to(REPO) else str(binary)
            binary_hashes[label] = digest
            snapshot = binary_dir / CAPABILITY_BINARIES[role]
            shutil.copy2(binary, snapshot)
            if sha256(snapshot) != digest:
                raise ValueError(f"{role}: executable changed while it was being archived")
            snapshots[role] = snapshot
            snapshot_hashes[str(snapshot)] = digest
            capabilities[role] = read_capabilities(snapshot, role, env, acceptance=args.acceptance)
    except BaseException as error:
        (output / "report.json").write_text(json.dumps({
            "status": "admission_failed", "error": str(error), "capabilities": capabilities,
            "binaries": binary_hashes, "binary_snapshots": snapshot_hashes,
            "goal_acceptance": {"requested": args.acceptance, "passed": False if args.acceptance else None},
        }, indent=2) + "\n")
        raise
    first, outer = snapshots["first_stage"], snapshots["wrapper"]
    compressor = snapshots.get("fri")
    historical = args.setup == "development" and not args.root_artifacts and not args.fri_artifacts
    report = {
        "status": "running",
        "development_srs": args.setup == "development",
        "setup": args.setup,
        "development_public_degree": args.development_public_degree,
        "fused_staging": args.fused_staging or args.retained_workers,
        "retained_workers": args.retained_workers,
        "latency_boundary": "retained_worker_fresh_request" if args.retained_workers else "cold_process_pipeline",
        "distributed_wrapper": args.distributed_wrapper,
        "kzg_devices": devices,
        "fresh_output_directory": True,
        "goal_acceptance": {"requested": args.acceptance,
                            "passed": False if args.development_public_degree else None},
        "capabilities": capabilities,
        "development_srs_cache": env.get("MULTI_STARK_KZG_DEV_SRS_CACHE"),
        "fixed_preprocessing_cache": env.get("MULTI_STARK_KZG_FIXED_CACHE"),
        "fri_compression_rerun": bool(args.root_artifacts),
        "timing_boundary": "aggregation root to verified final packet" if args.root_artifacts
                           else "compressed FRI to verified final packet",
        "historical_cpu_artifact_comparison": historical,
        "baseline": baseline,
        "gpu_sampling_scope": "whole run, including independent CPU verification after pipeline timing",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "hardware": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"
        ], text=True),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
        "storage": subprocess.check_output(["df", "-h", str(output)], text=True),
        "environment": captured_environment(env),
        "binaries": binary_hashes,
        "binary_snapshots": snapshot_hashes,
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in benchmark_sources()},
        "phases": {},
        "parity": {},
        "proofs": {},
        "setup_bindings": {},
    }
    workers = {}
    if args.retained_workers:
        report["worker_startup"] = {"completed": False, "workers": {}, "phases": {}, "proofs": {}}

    def save():
        temporary = output / "report.partial.json"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "report.json")

    def run(name, command, cpu=False, recursive=False):
        print(f"Starting {name}: {' '.join(map(str, command))}", flush=True)
        log_path, time_path = output / f"{name}.log", output / f"{name}.time"
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        phase_env = phase_environment(
            env, cpu=cpu, recursive=recursive, distributed_wrapper=args.distributed_wrapper)
        with log_path.open("w") as log:
            process = subprocess.Popen([
                "/usr/bin/time", "-f", '{"wall_seconds":%e,"peak_rss_kib":%M,"exit_code":%x}',
                "-o", str(time_path), *map(str, command),
            ], cwd=REPO, env=phase_env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print(f"{name}: {time.monotonic() - started:.0f}s; log: {log_path}", flush=True)
            except BaseException:
                terminate_process_group(process)
                report["phases"][name] = {
                    "wall_seconds": time.monotonic() - started,
                    "exit_code": process.returncode,
                    "interrupted": True,
                    "command": list(map(str, command)),
                    "log": str(log_path),
                    "backend": "cpu" if cpu else "cuda",
                    "environment": captured_environment(phase_env),
                }
                raise
        measured = json.loads(time_path.read_text().splitlines()[-1])
        measured["wall_seconds"] = time.monotonic() - started
        measured["started_at_utc"] = started_at
        measured["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        measured["peak_rss_bytes"] = measured.pop("peak_rss_kib") * 1024
        measured["command"] = list(map(str, command))
        measured["log"] = str(log_path)
        measured["backend"] = "cpu" if cpu else "cuda"
        measured["environment"] = captured_environment(phase_env)
        add_log_evidence(measured, log_path)
        report["phases"][name] = measured
        save()
        if process.returncode:
            raise RuntimeError(f"{name} exited {process.returncode}; see {log_path}")
        print(f"Finished {name}: {measured['wall_seconds']:.1f}s, "
              f"{measured['peak_rss_bytes'] / 2**30:.1f} GiB RSS", flush=True)

    def compare(name, actual, expected):
        files = proof_hashes(actual)
        require_hashes(files, expected, f"proof artifact {name}")
        report["parity"][name] = files
        save()

    def worker_request(role, phase, request_id, source, destination, claims, *, startup=False):
        worker = workers[role]
        scope = report["worker_startup"] if startup else report
        print(f"Starting {phase}: worker request {request_id}", flush=True)
        try:
            worker.request(request_id, source, destination, claims, output / f"{phase}.log", reused=not startup)
        finally:
            if worker.last_request is not None:
                scope["phases"][phase] = worker.last_request
                save()
        print(f"Finished {phase}: {worker.last_request['wall_seconds']:.1f}s; "
              "GPU idle memory reported", flush=True)

    def close_workers(*, graceful):
        errors = []
        for role, worker in workers.items():
            try:
                worker.close(graceful=graceful)
            except BaseException as error:
                errors.append(f"{role}: {error}")
            report["worker_startup"]["workers"][role]["shutdown"] = worker.shutdown
        if errors:
            raise RuntimeError("; ".join(errors))

    save()
    for relative in report["sources"]:
        snapshot = output / "sources" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, snapshot)
    if args.root_artifacts:
        root_snapshot = output / "root-artifacts"
        root_snapshot.mkdir()
        for name in ["root-vk.bin", "root-proof.bin", "root-claims.bin"]:
            shutil.copyfile(args.root_artifacts / name, root_snapshot / name)
        env["MULTI_STARK_INIT_EXPECTED_CLAIMS"] = str(root_snapshot / "root-claims.bin")
        report["root_artifacts"] = {name: sha256(root_snapshot / name) for name in
                                    ["root-vk.bin", "root-proof.bin", "root-claims.bin"]}
    if args.retained_workers:
        seed_fri, seed_claims, seed_evidence = snapshot_worker_seed(
            args.worker_seed_fri, args.worker_seed_claims, output / "worker-seed")
        report["worker_startup"]["seed"] = seed_evidence
    save()
    gpu_log = (output / "gpu-samples.csv").open("w")
    monitor = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits", "--loop=2",
    ], stdout=gpu_log, stderr=subprocess.DEVNULL, env=dict(env, TZ="UTC"))
    overall_started = time.monotonic()
    started = overall_started
    try:
        if args.retained_workers:
            startup = report["worker_startup"]
            startup["started_at_utc"] = datetime.now(timezone.utc).isoformat()
            startup["timing_boundary"] = "first process start through two sequential warmup proofs and GPU quiescence"
            warmup = output / "worker-warmup"
            warmup.mkdir()
            for role, binary, phase, source, destination in (
                    ("first_stage", first, "startup_fri_stage_and_prove", seed_fri, warmup / "fri-to-kzg"),
                    ("wrapper", outer, "startup_recursive_stage_and_prove", warmup / "intermediate", warmup / "recursive")):
                worker_env = phase_environment(env, recursive=role == "wrapper",
                                               distributed_wrapper=args.distributed_wrapper)
                worker = WorkerClient(binary, role, output / f"{role}-worker", worker_env, devices,
                                      development_public_degree=args.development_public_degree)
                workers[role] = worker
                startup["workers"][role] = worker.startup
                save()
                worker_request(role, phase, f"warmup-{role}", source, destination, seed_claims, startup=True)
                stage = "fri_to_kzg" if role == "first_stage" else "recursive_kzg"
                record_proof_evidence(startup, stage, destination / "kzg")
                verification = read_verification_report(destination / "kzg", "prove")
                startup.setdefault("verification", {})[stage] = verification
                if args.development_public_degree:
                    validate_development_public_degree_reports(startup["verification"].values())
                if (verification.get("native_verification_passed") is not True
                        or verification.get("negative_tests_pass") is not True
                        or (role == "wrapper" and verification.get("external_pairings_pass") is not True)):
                    raise ValueError(f"{role} warmup proof lacks native verification evidence")
                if role == "first_stage":
                    export_inner_proof(destination, warmup / "intermediate")
                save()
            startup["completed"] = True
            startup["wall_seconds"] = time.monotonic() - overall_started
            startup["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            report["worker_startup_wall_seconds"] = startup["wall_seconds"]
            save()
            started = time.monotonic()
        report["pipeline_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        fri, recursive = output / "fri-to-kzg", output / "recursive"
        input_fri = args.fri_artifacts.resolve() if args.fri_artifacts else REPO / "experiments/init-fri-artifacts"
        if args.root_artifacts:
            input_fri = output / "compressed-fri"
            run("fri_compress", [compressor, root_snapshot, input_fri, "--prove-outer"])
        report["fri_input"] = {name: sha256(input_fri / name) for name in
                               ["outer-vk.bin", "outer-proof.bin", "outer-claims.bin"]}
        if baseline:
            require_hashes(report["fri_input"], baseline["fri_input"], "baseline FRI input")
        if args.retained_workers:
            worker_request("first_stage", "fri_stage_and_prove", "fresh-first_stage", input_fri,
                           fri, root_snapshot / "root-claims.bin")
        elif args.fused_staging:
            run("fri_stage_and_prove", [first, "stage-and-prove", input_fri, fri])
        else:
            run("fri_stage", [first, "stage", input_fri, fri])
            run("fri_prove", [first, "prove", fri])
        record_proof_evidence(report, "fri_to_kzg", fri / "kzg")
        report["first_stage_verification"] = read_verification_report(fri / "kzg", "prove")
        report["setup_bindings"]["fri_to_kzg"] = (fri / "kzg-setup-id.bin").read_bytes().hex()
        save()
        if baseline:
            compare("fri_to_kzg", fri / "kzg", baseline["proofs"]["fri_to_kzg"])
        elif historical:
            compare("fri_to_kzg", fri / "kzg", proof_hashes(REPO / "experiments/init-fri-kzg-artifacts"))
        intermediate = output / "intermediate"
        export_inner_proof(fri, intermediate)
        if args.retained_workers:
            worker_request("wrapper", "recursive_stage_and_prove", "fresh-wrapper", intermediate,
                           recursive, root_snapshot / "root-claims.bin")
        elif args.fused_staging:
            run("recursive_stage_and_prove", [outer, "stage-and-prove", intermediate, recursive], recursive=True)
        else:
            run("recursive_stage", [outer, "stage", intermediate, recursive], recursive=True)
            run("recursive_prove", [outer, "prove", recursive], recursive=True)
        record_proof_evidence(report, "recursive_kzg", recursive / "kzg")
        report["setup_bindings"]["recursive_kzg"] = (recursive / "kzg-setup-id.bin").read_bytes().hex()
        if baseline:
            compare("recursive_kzg", recursive / "kzg", baseline["proofs"]["recursive_kzg"])
        elif historical:
            compare("recursive_kzg", recursive / "kzg", proof_hashes(REPO / "experiments/init-kzg-recursive-artifacts/kzg"))
        report["pipeline_wall_seconds"] = time.monotonic() - started
        report["pipeline_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if args.retained_workers:
            report["worker_startup_plus_pipeline_wall_seconds"] = time.monotonic() - overall_started
        report["recursive_verification"] = read_verification_report(recursive / "kzg", "prove")
        save()
        if args.retained_workers:
            shutdown_started = time.monotonic()
            close_workers(graceful=True)
            report["worker_shutdown_wall_seconds"] = time.monotonic() - shutdown_started
            save()
        verify_command = "dev-v4-verify" if args.development_public_degree else "verify"
        run("fri_cpu_verify", [first, verify_command, fri], cpu=True)
        run("recursive_cpu_verify", [outer, verify_command, recursive], cpu=True, recursive=True)
        report["first_stage_cpu_verification"] = read_verification_report(fri / "kzg", "verify")
        report["recursive_cpu_verification"] = read_verification_report(recursive / "kzg", "verify")
        if args.development_public_degree:
            validate_development_public_degree_reports([
                *report["worker_startup"]["verification"].values(),
                report["first_stage_verification"], report["recursive_verification"],
                report["first_stage_cpu_verification"], report["recursive_cpu_verification"],
            ])
        for name, directory in (("fri_to_kzg", fri / "kzg"), ("recursive_kzg", recursive / "kzg")):
            require_hashes(proof_hashes(directory), report["proofs"][name], f"{name} CPU verification artifacts")
        report["status"] = ("verified_baseline_artifact_parity" if baseline else
                            "verified_cpu_artifact_parity" if historical else "verified_regenerated_chain")
        if args.acceptance:
            report["goal_acceptance"] = evaluate_goal(report)
    except BaseException as error:
        report["status"] = "cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
        report["error"] = str(error)
        if args.acceptance:
            report["goal_acceptance"] = {"requested": True, "passed": False,
                                         "failures": ["pipeline_failed"], "checks": {}}
        raise
    finally:
        try:
            close_workers(graceful=False)
        except BaseException as error:
            report["worker_cleanup_error"] = str(error)
            report["status"] = "failed"
            if args.acceptance:
                report["goal_acceptance"] = {"requested": True, "passed": False,
                                             "failures": ["worker_cleanup_failed"], "checks": {}}
        if args.retained_workers and not report["worker_startup"]["completed"]:
            report["worker_startup"]["wall_seconds"] = time.monotonic() - overall_started
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
        gpu_log.close()
        report["gpu_samples_every_2_seconds"] = summarize_gpu_samples(output / "gpu-samples.csv")
        phase_records = list(report["phases"].values())
        if args.retained_workers:
            phase_records.extend(report["worker_startup"]["phases"].values())
        for phase in phase_records:
            if "started_at_utc" in phase:
                phase["gpu_samples_every_2_seconds"] = summarize_gpu_samples(
                    output / "gpu-samples.csv", datetime.fromisoformat(phase["started_at_utc"]),
                    datetime.fromisoformat(phase["finished_at_utc"]))
        report["total_wall_seconds"] = time.monotonic() - overall_started
        save()
    if report.get("worker_cleanup_error"):
        raise RuntimeError(report["worker_cleanup_error"])
    print(f"Verified results: {output / 'report.json'}", flush=True)
    if args.acceptance and not report["goal_acceptance"]["passed"]:
        raise SystemExit("Goal acceptance failed: " + ", ".join(report["goal_acceptance"]["failures"]))


if __name__ == "__main__":
    main()
