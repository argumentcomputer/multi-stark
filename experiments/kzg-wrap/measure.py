#!/usr/bin/env python3
"""Measure constrained primitives; stop short of claiming a complete verifier."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = root / "experiments/kzg-wrap/target/release/init-kzg-wrap"
    results = []
    for operation, weight in [("add", False), ("add", True), ("double", True),
                              ("scalar", True), ("subgroup-fast", True),
                              ("subgroup-fixed", True)]:
        name = operation + ("-weight" if weight else "-constraints")
        command = ["/usr/bin/time", "-f", "%e %M", "-o", str(output / (name + ".time")),
                   str(binary), operation] + (["--weight"] if weight else [])
        with (output / (name + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((output / (name + ".log")).read_text().splitlines()[-1])
        elapsed, rss = (output / (name + ".time")).read_text().split()
        report.update(elapsed_seconds=float(elapsed), peak_rss_bytes=int(rss) * 1024)
        results.append(report)
        print(json.dumps(report), flush=True)
    census_path = root / "experiments/init-fri-kzg.json"
    census = json.loads(census_path.read_text())
    layout = next(x for x in census["layouts"]
                  if x["mode"] == "compact_partition24" and x["budget"] == 2)
    fixed = next(x for x in results if x["operation"] == "subgroup-fixed")
    projected = layout["points"] * fixed["plonk_rows"]
    # Smallest power of two for the projected rows and two blinding rows.
    domain = 1 << (projected + 1).bit_length()
    report = {
        "status": "subgroup_preflight_exceeds_target",
        "complete_wrapper_implemented": False,
        "full_intermediate_proof_generated": False,
        "production_setup": False,
        "primitive_measurements": results,
        "intermediate_layout": {"points": layout["points"], "packet_bytes": layout["packet_bytes"]},
        "subgroup_projection": {
            "rows": projected, "domain": domain, "target_domain": 1 << 27,
            "srs_and_key_bytes_excluding_workspace": 688 * domain,
            "scope": "Per-point application of the tested fixed-chain gadget. Excludes transcript, MSMs, AIR, external-output encoding and bindings. Not a lower bound on all possible implementations.",
        },
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [*sorted((root / "experiments/kzg-wrap/src").glob("*.rs")),
                                    root / "experiments/kzg-wrap/Cargo.lock",
                                    root / "experiments/kzg-wrap/measure.py",
                                    root / "experiments/flock/terminal.patch", census_path]},
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
