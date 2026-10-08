#!/usr/bin/env python3
"""Finish the staged Init KZG wrapper and verify its saved packet separately."""
import json
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "target/init-kzg-recursive"
BINARY = ROOT / "experiments/kzg-wrap/target/release/init-kzg-wrap"
STAGE_TIME = pathlib.Path("/tmp/init-kzg-recursive-stage.time")


def run(phase):
    started = time.time()
    timing = OUT / f"{phase}.time"
    with (OUT / f"{phase}.log").open("w") as log:
        status = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(timing), str(BINARY), phase, str(OUT)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        ).returncode
    text = timing.read_text()
    peak = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    result = {"exit_code": status, "wall_seconds": time.time() - started,
              "peak_rss_bytes": int(peak[1]) * 1024 if peak else None}
    (OUT / f"{phase}-resources.json").write_text(json.dumps(result, indent=2) + "\n")
    if status:
        raise RuntimeError(f"{phase} failed ({status}); see {OUT / (phase + '.log')}")
    return result


def main():
    deadline = time.monotonic() + 3 * 3600
    while True:
        text = STAGE_TIME.read_text() if STAGE_TIME.exists() else ""
        status = re.search(r"Exit status: (\d+)", text)
        if status:
            if status[1] != "0" or not (OUT / "manifest.bin").exists():
                raise RuntimeError("staging failed; see /tmp/init-kzg-recursive-stage.log")
            break
        if time.monotonic() > deadline:
            raise RuntimeError("staging did not finish within three hours")
        time.sleep(5)
    print("Staging complete; starting the outer KZG proof", flush=True)
    resources = {phase: run(phase) for phase in ("prove", "verify")}
    result = json.loads((OUT / "kzg/prove-report.json").read_text())
    result["resources"] = resources
    result["separate_verification"] = json.loads((OUT / "kzg/verify-report.json").read_text())
    (ROOT / "experiments/init-kzg-recursive-proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
