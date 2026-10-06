"""Sequential, resumable development proving of the measured 11-shard Init bundle."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

artifacts = Path("target/init-fri-wrap").resolve()
output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else artifacts / "prove-11"
streamed = os.environ.get("INIT_GROTH16_STREAMING") == "1"
output.mkdir(exist_ok=True)
binary = output / "init_fri_wrap"
if not binary.exists():
    shutil.copy2("target/release/examples/init_fri_wrap", binary)
fingerprint = {
    str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [binary, artifacts / "outer-vk.bin", artifacts / "outer-proof.bin", artifacts / "outer-claims.bin"]
}
if streamed:
    fingerprint["mode"] = "streaming"
identity = output / "inputs.json"
if identity.exists():
    assert json.loads(identity.read_text()) == fingerprint, "binary or inputs changed; choose a fresh output directory"
else:
    identity.write_text(json.dumps(fingerprint, indent=2) + "\n")


def limits():
    cap = 450 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def memory_status(pid):
    lines = []
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        lines.extend(f"pid {pid} {line}" for line in status if line.startswith(("VmRSS:", "VmPeak:")))
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        for child in children:
            lines.extend(memory_status(child))
    except FileNotFoundError:
        pass
    return lines


env = dict(os.environ, RAYON_NUM_THREADS="32")
for shard in range(11):
    directory = output / f"shard-{shard}"
    directory.mkdir(exist_ok=True)
    if (directory / "VERIFIED").exists():
        print(f"Shard {shard}: checkpoint exists; bundle verification will recheck it", flush=True)
        continue
    command = [str(binary), str(artifacts), str(directory), "10", str(shard), "--stream-prove" if streamed else "--prove"]
    started = time.monotonic()
    print(f"Starting shard {shard}: {command}", flush=True)
    with (directory / "run.log").open("w") as log, (directory / "time.txt").open("w") as timing:
        process = subprocess.Popen(["/usr/bin/time", "-v", *command], stdout=log, stderr=timing, env=env, preexec_fn=limits)
        while process.poll() is None:
            time.sleep(30)
            memory = memory_status(process.pid)
            print(f"Shard {shard}: {time.monotonic()-started:.0f}s; {'; '.join(memory)}", flush=True)
    result = {"shard": shard, "seconds": time.monotonic()-started, "exit_code": process.returncode, "verified": (directory / "VERIFIED").exists()}
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if process.returncode or not result["verified"]:
        raise SystemExit(f"Shard {shard} failed; see {directory}/time.txt and run.log")

subprocess.run([str(binary), str(artifacts), str(output), "--bundle"], check=True, env=env)
print(f"Verified bundle saved in {output}", flush=True)
