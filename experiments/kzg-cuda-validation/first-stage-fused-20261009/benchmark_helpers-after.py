#!/usr/bin/env python3
"""Measure Init compression with fresh or cached fixed parameters."""

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
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


def summarize_cuda_events(path):
    import math

    operations = {}
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
            elif len(pairs) != len(fields):
                reason = "duplicate fields"
            elif not re.fullmatch(r"(?:\w+=[^\s=]+)(?: \w+=[^\s=]+)*", line[len(prefix):]):
                reason = "interleaved or malformed record text"
        if reason:
            skipped.append({"line": number, "reason": reason, "text": line})
            continue
        parsed += 1
        operation = operations.setdefault(fields["operation"], {"calls": 0})
        operation["calls"] += 1
        for key in metrics:
            if key in fields:
                operation[key] = operation.get(key, 0) + float(fields[key])
    return operations, {
        "totals_complete": not skipped,
        "records_seen": parsed + len(skipped),
        "records_parsed": parsed,
        "records_skipped": len(skipped),
        "skipped": skipped,
    }


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


def interrupt(signum, _frame):
    raise InterruptedError(f"Received {signal.Signals(signum).name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "target/kzg-cuda-run")
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
    parser.add_argument("--distributed-wrapper", action="store_true",
                        help="Enable four-GPU outer lookup/quotient consumers with an 8 GiB/card coefficient cap")
    parser.add_argument("--baseline", type=Path,
                        help="Verified run or recovery directory with identical inputs and proof bytes")
    args = parser.parse_args()
    if args.setup == "filecoin" and args.srs_cache:
        parser.error("--srs-cache requires --setup development")
    baseline = load_baseline(args.baseline.resolve()) if args.baseline else None
    if baseline and args.root_artifacts:
        actual = {name: sha256(args.root_artifacts / name) for name in
                  ("root-vk.bin", "root-proof.bin", "root-claims.bin")}
        require_hashes(actual, baseline["root_artifacts"] or {}, "baseline root input")
    signal.signal(signal.SIGTERM, interrupt)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    first = REPO / "target/release/examples/init_fri_kzg_prove"
    outer = REPO / "experiments/kzg-wrap/target/release/init-kzg-wrap"
    compressor = REPO / "target/release/examples/ix_root"
    binaries = [first, outer] + ([compressor] if args.root_artifacts else [])
    for binary in binaries:
        if not binary.is_file():
            raise SystemExit(f"Build the kzg-cuda binary first: {binary}")
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
    if args.fixed_cache:
        env["MULTI_STARK_KZG_FIXED_CACHE"] = str(args.fixed_cache.resolve())
    if args.srs_cache:
        env["MULTI_STARK_KZG_DEV_SRS_CACHE"] = str(args.srs_cache.resolve())
    if args.root_artifacts:
        args.root_artifacts = args.root_artifacts.resolve()
        env["MULTI_STARK_INIT_EXPECTED_CLAIMS"] = str(args.root_artifacts / "root-claims.bin")
    historical = args.setup == "development" and not args.root_artifacts and not args.fri_artifacts
    report = {
        "status": "running",
        "development_srs": args.setup == "development",
        "setup": args.setup,
        "fused_staging": args.fused_staging,
        "distributed_wrapper": args.distributed_wrapper,
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
        "binaries": {str(p.relative_to(REPO)): sha256(p) for p in binaries},
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in benchmark_sources()},
        "phases": {},
        "parity": {},
    }

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
        log_text = log_path.read_text()
        measured["fixed_cache_hits"] = log_text.count("Reused fixed preprocessing:")
        measured["fixed_cache_misses"] = log_text.count("Fixed preprocessing cache miss:")
        measured["srs_cache_hits"] = log_text.count("Loaded development SRS cache")
        measured["srs_cache_misses"] = log_text.count("Generated and cached development SRS")
        (measured["cuda_operation_totals_overlap"],
         measured["cuda_profile_parse"]) = summarize_cuda_events(log_path)
        measured["host_operation_totals_overlap"] = summarize_host_operations(log_path)
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

    save()
    for relative in report["sources"]:
        snapshot = output / "sources" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, snapshot)
    binary_dir = output / "measured-binaries"
    binary_dir.mkdir()
    for binary in binaries:
        shutil.copy2(binary, binary_dir / binary.name)
    first, outer = binary_dir / first.name, binary_dir / outer.name
    compressor = binary_dir / compressor.name
    report["binary_snapshots"] = {str(binary_dir / p.name): sha256(binary_dir / p.name) for p in binaries}
    if args.root_artifacts:
        root_snapshot = output / "root-artifacts"
        root_snapshot.mkdir()
        for name in ["root-vk.bin", "root-proof.bin", "root-claims.bin"]:
            shutil.copyfile(args.root_artifacts / name, root_snapshot / name)
        env["MULTI_STARK_INIT_EXPECTED_CLAIMS"] = str(root_snapshot / "root-claims.bin")
        report["root_artifacts"] = {name: sha256(root_snapshot / name) for name in
                                    ["root-vk.bin", "root-proof.bin", "root-claims.bin"]}
    save()
    gpu_log = (output / "gpu-samples.csv").open("w")
    monitor = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits", "--loop=2",
    ], stdout=gpu_log, stderr=subprocess.DEVNULL, env=dict(env, TZ="UTC"))
    started = time.monotonic()
    try:
        fri, recursive = output / "fri-to-kzg", output / "recursive"
        input_fri = args.fri_artifacts.resolve() if args.fri_artifacts else REPO / "experiments/init-fri-artifacts"
        if args.root_artifacts:
            input_fri = output / "compressed-fri"
            run("fri_compress", [compressor, root_snapshot, input_fri, "--prove-outer"])
        report["fri_input"] = {name: sha256(input_fri / name) for name in
                               ["outer-vk.bin", "outer-proof.bin", "outer-claims.bin"]}
        if baseline:
            require_hashes(report["fri_input"], baseline["fri_input"], "baseline FRI input")
        if args.fused_staging:
            run("fri_stage_and_prove", [first, "stage-and-prove", input_fri, fri])
        else:
            run("fri_stage", [first, "stage", input_fri, fri])
            run("fri_prove", [first, "prove", fri])
        if baseline:
            compare("fri_to_kzg", fri / "kzg", baseline["proofs"]["fri_to_kzg"])
        elif historical:
            compare("fri_to_kzg", fri / "kzg", proof_hashes(REPO / "experiments/init-fri-kzg-artifacts"))
        intermediate = output / "intermediate"
        intermediate.mkdir()
        shutil.copyfile(fri / "manifest.bin", intermediate / "manifest.bin")
        shutil.copyfile(fri / "kzg-setup-id.bin", intermediate / "kzg-setup-id.bin")
        for path in (fri / "kzg").glob("setup-*.bin"):
            shutil.copyfile(path, intermediate / path.name)
        shutil.copyfile(fri / "kzg/proof.compact.bin", intermediate / "proof.compact.bin")
        if args.fused_staging:
            run("recursive_stage_and_prove", [outer, "stage-and-prove", intermediate, recursive], recursive=True)
        else:
            run("recursive_stage", [outer, "stage", intermediate, recursive], recursive=True)
            run("recursive_prove", [outer, "prove", recursive], recursive=True)
        if baseline:
            compare("recursive_kzg", recursive / "kzg", baseline["proofs"]["recursive_kzg"])
        elif historical:
            compare("recursive_kzg", recursive / "kzg", proof_hashes(REPO / "experiments/init-kzg-recursive-artifacts/kzg"))
        report["pipeline_wall_seconds"] = time.monotonic() - started
        report["recursive_verification"] = json.loads((recursive / "kzg/prove-report.json").read_text())
        save()
        run("fri_cpu_verify", [first, "verify", fri], cpu=True)
        run("recursive_cpu_verify", [outer, "verify", recursive], cpu=True, recursive=True)
        report["status"] = ("verified_baseline_artifact_parity" if baseline else
                            "verified_cpu_artifact_parity" if historical else "verified_regenerated_chain")
    except BaseException as error:
        report["status"] = "cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
        report["error"] = str(error)
        raise
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
        gpu_log.close()
        report["gpu_samples_every_2_seconds"] = summarize_gpu_samples(output / "gpu-samples.csv")
        for phase in report["phases"].values():
            if "started_at_utc" in phase:
                phase["gpu_samples_every_2_seconds"] = summarize_gpu_samples(
                    output / "gpu-samples.csv", datetime.fromisoformat(phase["started_at_utc"]),
                    datetime.fromisoformat(phase["finished_at_utc"]))
        report["total_wall_seconds"] = time.monotonic() - started
        save()
    print(f"Verified results: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
