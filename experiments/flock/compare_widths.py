#!/usr/bin/env python3
"""Compare saved Init width trials; reject mismatched inner statements."""
import argparse
import json
from pathlib import Path


def compare(paths):
    runs = []
    identity = None
    for path in paths:
        status = json.loads((path / "run-status.json").read_text())
        report = json.loads((path / "report.json").read_text())
        statement = {k: report[k] for k in ["source_proof_blake3", "source_vk_blake3", "public_claim_words"]}
        if identity is None:
            identity = statement
        if statement != identity:
            raise ValueError(f"inner statement differs: {path}")
        run = {
            "artifacts": str(path),
            "run_status": status["status"],
            "binary_sha256": status["binary_sha256"],
            "initial_k": report.get("initial_k", status.get("initial_k")),
            "native": {k: report.get(k) for k in ["status", "proof_and_commitment_bytes", "prove_seconds", "verify_seconds", "pcs_config_blake3", "serialized_roundtrip_verified", "altered_claim_rejected"]},
            "elapsed_seconds": status["elapsed_seconds"],
            "peak_rss_bytes": status.get("peak_rss_bytes"),
            "sampled_peak_rss_bytes": status["sampled_peak_process_tree_rss_bytes"],
        }
        for name in ["terminal.json", "phase-progress.json", "projection-progress.json"]:
            if (path / name).exists():
                run[name] = json.loads((path / name).read_text())
        runs.append(run)
    return {
        "pipeline_complete": False,
        "scope": "Native proofs and terminal censuses; each run records completed checks. No terminal proof or composed security validation.",
        "timing_scope": "Prove time excludes preparation/verification. Runs can overlap a terminal census; RSS belongs to each run, not total host RAM.",
        "inner_statement": identity,
        "runs": runs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.runs), indent=2) + "\n")
