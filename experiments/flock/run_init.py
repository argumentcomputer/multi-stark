#!/usr/bin/env python3
"""Measure a full Init attempt, terminating cleanly at a process RSS limit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]


def process_rss(pid):
    """Sum the process tree, excluding threads (which share an address space)."""
    try:
        fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines())
        rss = int(fields.get("VmRSS", "0 kB").split()[0]) * 1024
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        return rss + sum(process_rss(int(child)) for child in children)
    except (FileNotFoundError, ProcessLookupError):
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["fast", "slim"], default="fast")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--rss-limit-gib", type=float, default=400)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--terminal-census", action="store_true")
    parser.add_argument("--coalesced", action="store_true")
    parser.add_argument("--initial-k", type=int, choices=[4, 5, 6])
    args = parser.parse_args()
    if args.threads < 1 or not args.rss_limit_gib > 0:
        parser.error("threads and RSS limit must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run-status.json").exists():
        parser.error("choose a new output directory to preserve previous measurements")
    exe = PACKAGE / "target/release/init-flock"
    shutil.copy2(exe, out / "init-flock-used")
    exe = out / "init-flock-used"
    command = ["/usr/bin/time", "-f", "%e %M %x", "-o", str(out / "time.txt"), str(exe),
               "init-prove", str(args.init_artifacts.resolve()), str(out), args.profile]
    if args.terminal_census:
        command[-4:] = ["terminal-init", str(args.init_artifacts.resolve()), str(out), args.profile]
    if args.coalesced:
        command.append("--coalesced")
    if args.initial_k is not None:
        command.extend(["--initial-k", str(args.initial_k)])
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), PCS_TRACE="1")
    if args.low_memory:
        env["FLOCK_LOW_MEMORY"] = "1"
        env["FLOCK_ZC_TIMING"] = "1"
    else:
        env.pop("FLOCK_LOW_MEMORY", None)
    build_info = json.loads(subprocess.check_output([str(exe), "build-info"], text=True))
    if args.terminal_census and not build_info.get("terminal_enabled", False):
        parser.error("rebuild with --features terminal before running the terminal census")
    status = {"initial_k": args.initial_k, "build_info": build_info, "low_memory": args.low_memory, "coalesced": args.coalesced, "command": command, "profile": args.profile, "threads": args.threads,
              "rss_limit_bytes": int(args.rss_limit_gib * 2**30), "status": "running",
              "started_unix": time.time(), "binary_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted([*PACKAGE.glob("src/*.rs"),
                                                 *PACKAGE.glob("*.py"),
                                                 *PACKAGE.glob("*.patch"),
                                                 *PACKAGE.glob("configs/*.toml"),
                                                 *ROOT.glob("src/**/*.rs"),
                                                 PACKAGE / "Cargo.lock", PACKAGE / "Cargo.toml",
                                                 ROOT / "Cargo.toml", ROOT / "Cargo.lock"])},
              "sampled_peak_process_tree_rss_bytes": 0}
    started = time.monotonic()
    with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        status["pid"] = process.pid
        while process.poll() is None:
            rss = process_rss(process.pid)
            status["sampled_peak_process_tree_rss_bytes"] = max(status["sampled_peak_process_tree_rss_bytes"], rss)
            status["elapsed_seconds"] = time.monotonic() - started
            status["current_process_tree_rss_bytes"] = rss
            if rss > status["rss_limit_bytes"] or (out / "STOP").exists():
                status["status"] = "rss_limit" if rss > status["rss_limit_bytes"] else "stopped"
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (out / "run-status.json").write_text(json.dumps(status, indent=2) + "\n")
            time.sleep(1)
        status["exit_code"] = process.returncode
    if status["status"] == "running":
        status["status"] = "completed" if process.returncode == 0 else "failed"
    status["elapsed_seconds"] = time.monotonic() - started
    if (out / "time.txt").exists():
        lines = (out / "time.txt").read_text().splitlines()
        if lines and len(lines[-1].split()) == 3:
            wall, rss, code = lines[-1].split()
            status["time_seconds"] = float(wall)
            status["peak_rss_bytes"] = int(rss) * 1024
    (out / "run-status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
