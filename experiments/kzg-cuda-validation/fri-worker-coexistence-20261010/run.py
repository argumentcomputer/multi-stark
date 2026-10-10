#!/usr/bin/env python3
"""Replay FRI while two verified, quiescent KZG fixture processes remain alive."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


ARCHIVE = Path(__file__).resolve().parent
REPO = ARCHIVE.parents[2]
HISTORY = ARCHIVE.parent / "fri-lookup-materialization-20261009"
WORKERS = ARCHIVE.parent / "request-workers-20261010"
ROOT_FILES = ("root-vk.bin", "root-proof.bin", "root-claims.bin")
FRI_FILES = ("outer-vk.bin", "outer-proof.bin", "outer-claims.bin")
IDLE_FORMAT = "kzg-cuda-idle/v1"
DEVICES = [0, 1, 2, 3]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def hashes(directory, names):
    return {name: sha256(directory / name) for name in names}


def snapshot(source, destination):
    expected = sha256(source)
    shutil.copy2(source, destination)
    if sha256(destination) != expected:
        raise RuntimeError(f"snapshot changed: {source}")
    return {"source": str(source), "snapshot": str(destination), "sha256": expected}


def host_memory(pid):
    try:
        fields = dict(line.split(":", 1) for line in
                      Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
        return {"pid": pid, **{key: int(fields[name].split()[0]) * 1024 for key, name in
                (("rss_bytes", "VmRSS"), ("lifetime_peak_rss_bytes", "VmHWM")) if name in fields}}
    except (OSError, ValueError):
        return {"pid": pid, "unavailable": True}


def gpu_snapshot():
    return {
        "utc": utc(),
        "devices": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv"], text=True),
        "processes": subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
            "--format=csv"], text=True),
    }


def relevant_environment(environment):
    return {key: value for key, value in environment.items()
            if key.startswith(("MULTI_STARK_", "RAYON_", "OMP_"))
            or key in ("CUDA_VISIBLE_DEVICES", "RUST_LOG", "TZ")}


def environments(root_snapshot, historical_environment):
    for key in ("CUDA_VISIBLE_DEVICES", "RAYON_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(key) != historical_environment.get(key):
            raise ValueError(f"{key} differs from the historical default-width primary FRI run")
    base = {key: value for key, value in os.environ.items() if not key.startswith("MULTI_STARK_")}
    fri = dict(base, **historical_environment)
    fri["MULTI_STARK_INIT_EXPECTED_CLAIMS"] = str(root_snapshot / "root-claims.bin")
    if (fri.get("MULTI_STARK_CUDA_DEVICE") != "0"
            or fri.get("MULTI_STARK_CUDA_AUX_DEVICES") != ""
            or fri.get("MULTI_STARK_CUDA_MEMORY_LOG") != "1"):
        raise ValueError("historical FRI settings are not the expected primary-only configuration")
    idle = dict(base, MULTI_STARK_KZG_BACKEND="cuda", MULTI_STARK_KZG_CUDA_DEVICES="0,1,2,3",
                MULTI_STARK_KZG_CUDA_RESIDENT_GIB="8", MULTI_STARK_KZG_CUDA_SRS_CACHE="1",
                MULTI_STARK_KZG_CUDA_PROFILE="1")
    return fri, idle


def capabilities(binary, expected_name, environment):
    result = subprocess.run([str(binary), "capabilities"], env=environment, cwd=REPO,
                            capture_output=True, text=True, check=True, timeout=15)
    if len(result.stdout.splitlines()) != 1:
        raise ValueError(f"{binary}: expected one capability line")
    value = json.loads(result.stdout)
    required = ("goldilocks_cuda", "parallel") if expected_name == "ix_root" else ("kzg", "kzg_cuda", "parallel")
    if (not isinstance(value, dict) or value.get("format") != "multi-stark-capabilities/v1"
            or value.get("binary") != expected_name or any(value.get(key) is not True for key in required)):
        raise ValueError(f"{binary}: compiled capabilities do not support this diagnostic")
    return value


class IdleFixture:
    def __init__(self, binary, directory, environment, helper):
        self.helper = helper
        self.directory = directory
        directory.mkdir()
        self.responses = directory / "responses.jsonl"
        self.log_path = directory / "fixture.log"
        self.log = self.log_path.open("xb")
        self.offset = 0
        self.started = time.monotonic()
        self.record = {"started_at_utc": utc(), "environment": relevant_environment(environment),
                       "command": [str(binary), str(self.responses)], "log": str(self.log_path),
                       "responses": str(self.responses)}
        try:
            self.process = subprocess.Popen(self.record["command"], env=environment, cwd=REPO,
                                            stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
                                            text=True, start_new_session=True)
        except BaseException:
            self.log.close()
            raise
        self.record["pid"] = self.process.pid

    def event(self, expected_status, index, *, timeout=120):
        started = time.monotonic()
        next_progress = started + 30
        while True:
            try:
                with self.responses.open("rb") as source:
                    source.seek(self.offset)
                    line = source.readline(65537)
            except FileNotFoundError:
                line = b""
            if len(line) > 65536:
                raise ValueError("idle fixture response exceeds 64 KiB")
            if line.endswith(b"\n"):
                value = json.loads(line)
                self.offset += len(line)
                self.record[expected_status] = value
                if (not isinstance(value, dict) or value.get("format") != IDLE_FORMAT
                        or value.get("status") != expected_status or value.get("pid") != self.process.pid
                        or value.get("check_index") != index or value.get("known_trapdoor") is not True
                        or value.get("scope") != "small_fixture_cuda_context_and_ntt_residency"
                        or value.get("log_n") != 16 or type(value.get("nonconstant_columns")) is not int
                        or value["nonconstant_columns"] < 4
                        or value.get("verified") is not True or value.get("parity") is not True
                        or type(value.get("opening_bytes")) is not int or value["opening_bytes"] <= 0
                        or not isinstance(value.get("opening_blake3"), str)
                        or re.fullmatch(r"[0-9a-f]{64}", value["opening_blake3"]) is None
                        or any(type(value.get(key)) not in (int, float)
                               or not math.isfinite(value[key]) or value[key] < 0
                               for key in ("operation_seconds", "idle_seconds"))):
                    raise ValueError(f"invalid verified idle fixture event: {value}")
                self.helper.validate_worker_idle(value.get("idle"), DEVICES)
                self.assert_alive()
                return value
            if self.process.poll() is not None:
                raise RuntimeError(f"idle fixture exited {self.process.returncode}; see {self.log_path}")
            now = time.monotonic()
            if now - started >= timeout:
                raise TimeoutError(f"waiting for idle fixture {expected_status}; see {self.log_path}")
            if now >= next_progress:
                print(f"Waiting for {self.directory.name} {expected_status}; log: {self.log_path}", flush=True)
                next_progress = now + 30
            time.sleep(0.02)

    def assert_alive(self):
        if self.process.poll() is not None:
            raise RuntimeError(f"idle fixture {self.process.pid} exited before completing coexistence")

    def prepare(self):
        self.event("ready", 0)
        self.record.update(ready_at_utc=utc(), startup_wall_seconds=time.monotonic() - self.started,
                           ready_host_memory=host_memory(self.process.pid))
        self.record["ready_log_end_byte"] = self.log_path.stat().st_size
        self.helper.add_log_evidence(self.record, self.log_path)

    def check(self):
        started = time.monotonic()
        self.record["check_log_start_byte"] = self.log_path.stat().st_size
        self.process.stdin.write("check\n")
        self.process.stdin.flush()
        value = self.event("checked", 1)
        for key in ("opening_bytes", "opening_blake3", "log_n", "nonconstant_columns"):
            if value[key] != self.record["ready"][key]:
                raise ValueError(f"idle fixture opening changed after FRI: {key}")
        self.record.update(check_wall_seconds=time.monotonic() - started, checked_at_utc=utc(),
                           checked_host_memory=host_memory(self.process.pid),
                           checked_log_end_byte=self.log_path.stat().st_size)

    def close(self, graceful):
        try:
            if self.process.stdin and not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
            if graceful:
                self.process.wait(timeout=60)
                if self.process.returncode:
                    raise RuntimeError(f"idle fixture exited {self.process.returncode}")
            else:
                self.helper.terminate_process_group(self.process)
        except BaseException:
            if self.process.poll() is None:
                self.helper.terminate_process_group(self.process)
            raise
        finally:
            self.record.update(exit_code=self.process.poll(), exited_at_utc=utc(),
                               lifetime_wall_seconds=time.monotonic() - self.started)
            self.log.close()


def fri_log_evidence(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text())
    labels = ("Circuit build", "Root verifier witness satisfied", "FRI lowering", "FRI trace construction",
              "FRI setup", "FRI proving")
    phase_seconds = {}
    for label in labels:
        match = re.search(re.escape(label) + r": ([0-9.]+)(ns|µs|us|ms|s)", text)
        if match:
            phase_seconds[label] = float(match[1]) * {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1}[match[2]]
    verified = re.search(r"VERIFIED: FRI proof_bytes=(\d+) altered_public_claim_rejected=true", text)
    return {
        "native_verification_and_altered_claim_passed": verified is not None,
        "reported_proof_bytes": int(verified[1]) if verified else None,
        "phase_seconds": phase_seconds,
        "phase_lines": [line for line in text.splitlines() if line.startswith((*labels, "VERIFIED:"))],
        "lookup_materialization_lines": [line for line in text.splitlines() if "Lookup expressions materialized" in line],
        "cuda_lines": [line for line in text.splitlines() if line.startswith("[multi-stark/cuda]")],
        "host_matrices": [dict(zip(("index", "height", "width", "lde_bytes", "work"), map(int, row)))
                          for row in re.findall(r"stage1 host matrix (\d+): height=(\d+) width=(\d+) lde_bytes=(\d+) work=(\d+)", text)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle-binary", type=Path, required=True)
    parser.add_argument("--fri-binary", type=Path, default=WORKERS / "bin/ix_root")
    parser.add_argument("--output", type=Path, default=ARCHIVE / "coexistence")
    parser.add_argument("--root-artifacts", type=Path, default=HISTORY / "root-artifacts")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary_dir, root_snapshot, driver_dir = output / "bin", output / "root-artifacts", output / "driver"
    for path in (binary_dir, root_snapshot, driver_dir):
        path.mkdir()
    provenance = {"runner": snapshot(Path(__file__).resolve(), driver_dir / "run.py"),
                  "helper": snapshot(REPO / "experiments/kzg-cuda-bench.py", driver_dir / "kzg-cuda-bench.py")}
    spec = importlib.util.spec_from_file_location("coexistence_helper", driver_dir / "kzg-cuda-bench.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    historical = json.loads((HISTORY / "primary/report.json").read_text())
    metadata = json.loads((HISTORY / "metadata.json").read_text())
    if historical.get("exit_code") != 0 or historical.get("historical_parity") is not True:
        raise ValueError("historical FRI artifact reference is not verified")
    if hashes(args.root_artifacts, ROOT_FILES) != metadata["root_sha256"]:
        raise ValueError("root input differs from the verified historical FRI input")
    for name in ROOT_FILES:
        snapshot(args.root_artifacts / name, root_snapshot / name)
    golden = output / "golden-fri"
    golden.mkdir()
    for name in FRI_FILES:
        snapshot(HISTORY / "primary/compressed-fri" / name, golden / name)
    if hashes(golden, FRI_FILES) != historical["artifacts_sha256"]:
        raise ValueError("historical FRI artifacts disagree with their verified report")
    provenance["fri_binary"] = snapshot(args.fri_binary.resolve(), binary_dir / "ix_root")
    provenance["idle_binary"] = snapshot(args.idle_binary.resolve(), binary_dir / "kzg_cuda_idle")
    for name, path in (("historical-report.json", HISTORY / "primary/report.json"),
                       ("historical-metadata.json", HISTORY / "metadata.json"),
                       ("fri-source-sha256.json", WORKERS / "source-sha256.json"),
                       ("fri-build-environment.json", WORKERS / "build-environment.json"),
                       ("idle-source-sha256.json", ARCHIVE / "source-sha256.json"),
                       ("idle-build-report.json", ARCHIVE / "idle-build/report.json"),
                       ("archived-binaries.json", ARCHIVE / "binaries.json"),
                       ("kzg_cuda_idle.rs", REPO / "examples/kzg_cuda_idle.rs")):
        provenance[name] = snapshot(path, driver_dir / name)
    fri_env, idle_env = environments(root_snapshot, historical["environment"])
    report = {
        "status": "running", "started_at_utc": utc(), "provenance": provenance,
        "scope": "One fresh FRI replay with two small known-trapdoor KZG PCS opening fixtures held idle; "
                 "no production KZG frontend or host-key footprint and no full-chain timing",
        "root_sha256": hashes(root_snapshot, ROOT_FILES), "expected_fri_sha256": hashes(golden, FRI_FILES),
        "fri_environment": relevant_environment(fri_env), "idle_environment": relevant_environment(idle_env),
        "parent_environment": relevant_environment(os.environ),
        "capabilities": {"fri": capabilities(binary_dir / "ix_root", "ix_root", fri_env),
                         "idle": capabilities(binary_dir / "kzg_cuda_idle", "kzg_cuda_idle", idle_env)},
        "historical_context": {"wall_seconds": historical["wall_seconds"], "binary_sha256": metadata["binary_sha256"],
                               "same_binary_isolated_reference_measured": False,
                               "speedup_or_coexistence_penalty_established": False},
        "logical_cpus": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "fixtures": [], "memory_snapshots": {}, "errors": [],
    }

    def save():
        temporary = output / "report.partial.json"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "report.json")

    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    fixtures, process, monitor = [], None, None
    started = time.monotonic()
    gpu_log = (output / "gpu-samples.csv").open("x")
    save()
    try:
        report["memory_snapshots"]["before_fixtures"] = gpu_snapshot()
        monitor = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits", "--loop=1"], stdout=gpu_log, stderr=subprocess.DEVNULL,
            env=dict(fri_env, TZ="UTC"))
        for index in range(2):
            fixture = IdleFixture(binary_dir / "kzg_cuda_idle", output / f"idle-{index}", idle_env, helper)
            fixtures.append(fixture)
            report["fixtures"].append(fixture.record)
            save()
            fixture.prepare()
            report["memory_snapshots"][f"after_idle_{index}_ready"] = gpu_snapshot()
            save()
        report["fixture_startup_wall_seconds"] = time.monotonic() - started
        for fixture in fixtures:
            fixture.assert_alive()
        fri = report["fri"] = {"started_at_utc": utc(), "log": str(output / "fri_compress.log")}
        command = ["/usr/bin/time", "-v", "-o", str(output / "fri.time.txt"), str(binary_dir / "ix_root"),
                   str(root_snapshot), str(output / "compressed-fri"), "--prove-outer"]
        fri["command"] = command
        print(f"Starting one coexisting FRI replay; log: {fri['log']}", flush=True)
        phase_started = time.monotonic()
        with (output / "fri_compress.log").open("x") as log, (output / "host-samples.jsonl").open("x") as samples:
            process = subprocess.Popen(command, cwd=REPO, env=fri_env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            fri["timing_wrapper_pid"] = process.pid
            next_sample, next_progress = phase_started, phase_started + 30
            while process.poll() is None:
                for fixture in fixtures:
                    fixture.assert_alive()
                now = time.monotonic()
                if now >= next_sample:
                    children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
                    try:
                        child_pids = children.read_text().split()
                    except FileNotFoundError:
                        child_pids = []
                    samples.write(json.dumps({"utc": utc(), "idle": [host_memory(f.process.pid) for f in fixtures],
                                              "fri_children": [host_memory(int(pid)) for pid in child_pids]}) + "\n")
                    samples.flush()
                    next_sample = now + 1
                if now >= next_progress:
                    print(f"Coexisting FRI: {now - phase_started:.0f}s; both idle fixtures alive", flush=True)
                    next_progress = now + 30
                if now - phase_started > 420:
                    raise TimeoutError("FRI replay exceeded the 420-second diagnostic bound")
                time.sleep(0.05)
        fri.update(exit_code=process.wait(), wall_seconds=time.monotonic() - phase_started, finished_at_utc=utc())
        fri.update(fri_log_evidence(output / "fri_compress.log"))
        resource = (output / "fri.time.txt").read_text()
        if peak := re.search(r"Maximum resident set size \(kbytes\): (\d+)", resource):
            fri["peak_rss_bytes"] = int(peak[1]) * 1024
        report["memory_snapshots"]["after_fri_before_checks"] = gpu_snapshot()
        if fri["exit_code"] == 0:
            fri["artifacts_sha256"] = hashes(output / "compressed-fri", FRI_FILES)
            fri["exact_proof_key_claims_parity"] = fri["artifacts_sha256"] == report["expected_fri_sha256"]
            if (not fri["native_verification_and_altered_claim_passed"] or not fri["exact_proof_key_claims_parity"]
                    or fri["reported_proof_bytes"] != (output / "compressed-fri/outer-proof.bin").stat().st_size):
                report["errors"].append("FRI verification or exact proof/key/claims parity failed")
        else:
            report["errors"].append(f"FRI exited {fri['exit_code']}")
        save()
        for index, fixture in enumerate(fixtures):
            fixture.assert_alive()
            fixture.check()
            report["memory_snapshots"][f"after_idle_{index}_checked"] = gpu_snapshot()
            save()
        for fixture in fixtures:
            fixture.close(graceful=True)
        report["status"] = "failed" if report["errors"] else "verified_fri_with_two_quiescent_small_kzg_contexts"
    except BaseException as error:
        report["status"] = "cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
        report["errors"].append(str(error))
        raise
    finally:
        if process is not None:
            try:
                helper.terminate_process_group(process)
                report.setdefault("fri", {})["cleanup_exit_code"] = process.poll()
            except BaseException as error:
                report["errors"].append(f"FRI process-group cleanup: {error}")
        for fixture in fixtures:
            if fixture.process.poll() is None or not fixture.log.closed:
                try:
                    fixture.close(graceful=False)
                except BaseException as error:
                    report["errors"].append(f"fixture cleanup: {error}")
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        gpu_log.close()
        report["gpu_samples_every_1_second"] = helper.summarize_gpu_samples(output / "gpu-samples.csv")
        if "fri" in report and "finished_at_utc" in report["fri"]:
            report["fri"]["gpu_samples_every_1_second"] = helper.summarize_gpu_samples(
                output / "gpu-samples.csv", datetime.fromisoformat(report["fri"]["started_at_utc"]),
                datetime.fromisoformat(report["fri"]["finished_at_utc"]))
        try:
            report["memory_snapshots"]["after_process_cleanup"] = gpu_snapshot()
        except (OSError, subprocess.SubprocessError) as error:
            report["errors"].append(f"final memory snapshot: {error}")
        report["total_wall_seconds"] = time.monotonic() - started
        report["finished_at_utc"] = utc()
        report["all_fixture_processes_reaped"] = all(f.process.poll() is not None for f in fixtures)
        report["fri_process_reaped"] = process is None or process.poll() is not None
        if report["errors"] and report["status"].startswith("verified_"):
            report["status"] = "failed"
        save()
    print(json.dumps({key: report.get(key) for key in ("status", "fixture_startup_wall_seconds", "total_wall_seconds", "errors")}, indent=2))
    if report["status"] != "verified_fri_with_two_quiescent_small_kzg_contexts":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
