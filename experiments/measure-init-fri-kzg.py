"""Measure the saved intermediate Init proof without allocating full KZG traces."""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys

artifacts = Path("target/init-fri-wrap").resolve()
output = Path(sys.argv[1] if len(sys.argv) > 1 else "target/init-fri-kzg-census-final").resolve()
output.mkdir(parents=True, exist_ok=True)
assert not (output / "run.log").exists(), "choose a fresh output directory"
binary = output / "init_fri_kzg"
shutil.copy2("target/release/examples/init_fri_kzg", binary)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def limits():
    cap = 100 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


fingerprint = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in [binary, *(artifacts / name for name in ["outer-vk.bin", "outer-proof.bin", "outer-claims.bin"])]}
(output / "inputs.json").write_text(json.dumps(fingerprint, indent=2) + "\n")
with (output / "run.log").open("w") as log, (output / "time.txt").open("w") as timing:
    subprocess.run(["/usr/bin/time", "-v", str(binary), str(artifacts), str(output)], stdout=log, stderr=timing, env=dict(os.environ, RAYON_NUM_THREADS="32"), preexec_fn=limits, check=True)


def record(line):
    def value(v):
        if v in ("true", "false"):
            return v == "true"
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    return {k: value(v) for k, v in (item.split("=", 1) for item in line.split()[1:])}


lines = (output / "run.log").read_text().splitlines()
timing = (output / "time.txt").read_text()
report = {
    "status": "Full intermediate-proof census completed; no full-size KZG setup or proof attempted.",
    "artifacts": fingerprint,
    "source_witness_checked": True,
    "changed_inner_public_words_rejected": 18,
    "census_peak_rss_bytes": int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", timing)[1]) * 1024,
    "census": [record(line) for line in lines if line.startswith("CENSUS ")],
    "layouts": [record(line) for line in lines if line.startswith("TOTAL ")],
    "traces": [record(line) for line in lines if line.startswith("TRACE ")],
    "payload": "FixedProofCodec proof bytes plus 32-byte trusted profile identifier and the original 18 canonical u64 claim words. Projected transport, not an emitted Init KZG proof. Verifying key and SRS are external.",
    "memory": "base_arrays_bytes counts retained coefficient arrays for advice, fixed, lookup and quotient polynomials once. max_quotient_inputs_bytes is the largest trace's simultaneous dense fixed/advice/lookup quotient evaluations. These are not peak RSS; exclude IR, assignments, fixed matrices, FFT/MSM scratch and selectors. Partition figures describe one ordinary multi-trace proof, not separately proved shards.",
    "security": "No production security claim. Census uses a tiny insecure SRS only to compile representative constraint graphs. Full-size setup was not generated; BLAKE3 and original Init claim are unchanged.",
    "logs": str(output),
    "source_sha256": {name: sha(Path(name)) for name in ["examples/init_fri_kzg.rs", "examples/support/init_fri.rs", "src/plonkish/foreign.rs", "experiments/measure-init-fri-kzg.py"]},
}
(output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
# Keep the versioned report compact; per-trace detail is in the runtime report.
report.pop("traces")
Path("experiments/init-fri-kzg.json").write_text(json.dumps(report, indent=2) + "\n")
for layout in report["layouts"]:
    print(f"{layout['mode']}, quotient budget {layout['budget']}: {layout['packet_bytes']} bytes, quotient input arrays {layout['max_quotient_inputs_bytes'] / 2**30:.2f} GiB", flush=True)
