#!/usr/bin/env python3
"""Verify and archive the already generated first-stage artifacts on the CPU."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import run


def surviving_children():
    output = subprocess.check_output(["ps", "-eo", "pid=,comm=,args="], text=True)
    matches = []
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, name, args = fields
        if (name.startswith("init_fri_kzg") and str(run.BINARY) in args) or (
            name == "nvidia-smi" and "--loop=2" in args
        ):
            matches.append({"pid": int(pid), "name": name})
    return matches


def main():
    report_path = run.HERE / "report.json"
    receipt = json.loads(report_path.read_text())
    shutil.copyfile(report_path, run.HERE / "report-before-recovery.json")
    if run.helpers.sha256(run.BINARY) != receipt["binary_sha256"]:
        raise RuntimeError("measured binary digest differs")
    staged = run.OUTPUT / "fri-to-kzg"
    proof = run.helpers.proof_hashes(staged / "kzg")
    run.helpers.require_hashes(proof, receipt["expected_proof_artifacts"], "legacy artifact parity")
    children_before = surviving_children()
    if children_before:
        raise RuntimeError(f"measurement children remain: {children_before}")
    artifact_dir = run.HERE / "proof-artifacts"
    artifact_dir.mkdir()
    for name in (*run.helpers.PROOF_FILES, "VERIFIED.txt", "SECURITY.txt"):
        shutil.copyfile(staged / "kzg" / name, artifact_dir / name)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("MULTI_STARK_KZG_"):
            env.pop(key)
    env.update(receipt["environment"])
    env["MULTI_STARK_KZG_BACKEND"] = "cpu"
    command = [str(run.BINARY), "verify", str(staged)]
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    with (run.HERE / "cpu-verify.log").open("w") as log:
        process = subprocess.Popen([
            "/usr/bin/time", "-f", '{"wall_seconds":%e,"peak_rss_kib":%M,"exit_code":%x}',
            "-o", str(run.HERE / "cpu-verify.time"), *command,
        ], cwd=run.REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait()
        except BaseException:
            run.helpers.terminate_process_group(process)
            raise
    measured = json.loads((run.HERE / "cpu-verify.time").read_text().splitlines()[-1])
    measured.update(
        wall_seconds=time.monotonic() - started,
        started_at_utc=started_at, finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=command, backend="cpu", environment=receipt["environment"] | {"MULTI_STARK_KZG_BACKEND": "cpu"},
        log=str(run.HERE / "cpu-verify.log"), artifact_parity=proof,
        children_before_verification=children_before, children_after_verification=surviving_children(),
        verification_script_sha256=run.helpers.sha256(Path(__file__)),
    )
    measured["peak_rss_bytes"] = measured.pop("peak_rss_kib") * 1024
    (run.HERE / "cpu-verification.json").write_text(json.dumps(measured, indent=2) + "\n")
    if process.returncode:
        raise RuntimeError(f"CPU verification exited {process.returncode}")
    if measured["children_after_verification"]:
        raise RuntimeError("measurement or verification child remains")
    run.helpers.require_hashes(run.helpers.proof_hashes(staged / "kzg"), proof, "post-verification artifacts")
    shutil.copyfile(staged / "kzg/VERIFIED.txt", run.HERE / "CPU-VERIFIED.txt")
    print(json.dumps({key: measured[key] for key in ("wall_seconds", "exit_code", "peak_rss_bytes", "children_after_verification")}), flush=True)


if __name__ == "__main__":
    main()
