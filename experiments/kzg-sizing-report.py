#!/usr/bin/env python3
"""Count preserved KZG profiles and conditional shapes without generating proofs."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent.parent
SAVED = Path("experiments/kzg-cuda-validation/resume-20261009")
LOG = Path("experiments/kzg-cuda-validation/partition-pipeline-20261009/cached/recursive_stage.log")
CAP = 1 << 27
FR = 52435875175126190479447740508185965837690552500527637822603658699938581184513


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def count(rows, height=None, policy="development"):
    heights = [height or row["height"] for row in rows]
    maximum = max(heights)
    totals = dict.fromkeys(("main", "fixed", "lookup", "quotient"), 0)
    cells = dict.fromkeys(totals, 0)
    shifted_fixed = shifted_witness = opened_fields = 0
    maximum_quotient_rows = 0
    shifted_heights = set()
    for row, n in zip(rows, heights):
        shifted = policy == "strict_public" or (policy == "development" and n < maximum)
        cost = 96 if shifted else 48
        widths = {
            "main": row["main_width"], "fixed": row["fixed_width"],
            "lookup": row[f"width_{cost}"], "quotient": row[f"quotient_{cost}"],
        }
        for name, width in widths.items():
            totals[name] += width
            cells[name] += width * n
        maximum_quotient_rows = max(maximum_quotient_rows, n * widths["quotient"])
        opened_fields += (widths["main"] * (1 + row["main_next"])
                          + widths["fixed"] * (1 + row["fixed_next"])
                          + 2 * widths["lookup"] + widths["quotient"])
        if shifted:
            shifted_heights.add(n)
            shifted_fixed += widths["fixed"]
            shifted_witness += widths["main"] + widths["lookup"] + widths["quotient"]
    witnesses = len(set(heights)) + 1
    witness_columns = sum(totals.values()) - totals["fixed"]
    degree_points = len(shifted_heights) + bool(shifted_heights)
    return {
        "policy": policy, "heights": heights, "circuits": len(rows),
        "columns": totals, "unshifted_commitments": sum(totals.values()),
        "shifted_fixed_commitments": shifted_fixed,
        "shifted_witness_commitments": shifted_witness,
        "opening_witnesses": witnesses,
        "witness_curve_inputs": witness_columns + shifted_witness + witnesses,
        "fixed_curve_inputs": totals["fixed"] + shifted_fixed,
        "degree_pairing_points": degree_points,
        "pairing_points": degree_points + 2,
        "pairing_output_scalar_values": 2 * (degree_points + 2),
        "opening_batch_msm_terms": sum(totals.values()) + 1 + witnesses,
        "opened_fields": opened_fields, "intermediate_accumulator_fields": len(rows) - 1,
        "compact_proof_bytes": 5 + 48 * (witness_columns + shifted_witness + witnesses)
                               + 32 * (opened_fields + len(rows) - 1),
        "dense_bytes_by_kind": {name: value * 32 for name, value in cells.items()},
        "dense_bytes_total": sum(cells.values()) * 32,
        "maximum_quotient_evaluation_rows": maximum_quotient_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new artifact directory")
    parser.add_argument("--counter", type=Path,
                        default=ROOT / "target/release/examples/kzg_profile_counts")
    args = parser.parse_args()
    raw = {}
    profiles = {}
    inputs = [LOG, SAVED / "recursive/circuit-report.json"]
    for stage, manifest_dir, setup_dir in [
        ("stage1", SAVED / "intermediate", SAVED / "intermediate"),
        ("outer", SAVED / "recursive", SAVED / "recursive/kzg"),
    ]:
        raw[stage] = subprocess.check_output(
            [str(args.counter), str(ROOT / manifest_dir), str(ROOT / setup_dir)], text=True)
        profiles[stage] = [{k: int(v) for k, v in row.items()}
                           for row in csv.DictReader(io.StringIO(raw[stage]))]
        inputs += [manifest_dir / "manifest.bin", setup_dir / "proof.compact.bin"]
        inputs += [setup_dir / f"setup-{row['index']}.bin" for row in profiles[stage]]
        baseline = count(profiles[stage])
        proof = (ROOT / setup_dir / "proof.compact.bin").read_bytes()
        if (proof[:4] != b"KQP1" or proof[4] != baseline["opening_witnesses"]
                or len(proof) != baseline["compact_proof_bytes"]):
            raise ValueError(f"{stage}: compact proof count differs from saved proof")

    first, outer = profiles["stage1"], profiles["outer"]
    target_outer = [dict(row, height=min(row["height"], CAP)) for row in outer]
    candidates = {
        "baseline": {"stage1": count(first), "outer": count(outer)},
        "uniform24_development": {"stage1": count(first, 1 << 24), "outer": count(target_outer)},
        "mixed_public_degree": {"stage1": count(first, policy="public_degree"),
                                "outer": count(target_outer, policy="public_degree")},
        "uniform24_public_degree": {"stage1": count(first, 1 << 24, "public_degree"),
                                    "outer": count(target_outer, policy="public_degree")},
        "uniform27_strict_public": {"stage1": count(first, CAP, "strict_public"),
                                    "outer": count(outer, CAP, "strict_public")},
    }
    for candidate in candidates.values():
        candidate["wrapper_public_values"] = 18 + candidate["stage1"]["pairing_output_scalar_values"]
        candidate["dense_bytes_both_stages"] = (candidate["stage1"]["dense_bytes_total"]
                                                + candidate["outer"]["dense_bytes_total"])
        candidate["packet_bytes"] = (32 + 18 * 8 + 48 * candidate["stage1"]["pairing_points"]
                                     + candidate["outer"]["compact_proof_bytes"])
        candidate["within_byte_budgets"] = (candidate["packet_bytes"] <= 2709
                                            and candidate["outer"]["compact_proof_bytes"] <= 2053)

    log = (ROOT / LOG).read_text()
    attribution = []
    previous = 0
    for label, body in re.findall(r"^(Curve inputs complete|Transcript complete|AIR checks complete|MSM \d+ complete): CircuitStats \{([^}]+)\}", log, re.M):
        fields = {k: int(v) for k, v in re.findall(r"(\w+): (\d+)", body)}
        cumulative = fields["gates"] + fields["lookups"] + fields["publics"]
        attribution.append({"label": label, "rows": cumulative - previous, "cumulative_rows": cumulative})
        previous = cumulative
    circuit_report = json.loads((ROOT / SAVED / "recursive/circuit-report.json").read_text())
    if previous + 1 != circuit_report["rows"]:
        raise ValueError("row attribution does not match the preserved circuit report")
    blocks = {}
    for number, body in re.findall(r"^Allocated (\d+) curve points: CircuitStats \{([^}]+)\}", log, re.M):
        fields = {k: int(v) for k, v in re.findall(r"(\w+): (\d+)", body)}
        blocks[int(number)] = fields["gates"] + fields["lookups"]
    point_costs = [(blocks[n] - blocks[n - 64]) // 64 for n in sorted(blocks) if n >= 448]
    if set(point_costs) != {108864}:
        raise ValueError("witness-point costs differ across allocation blocks")
    degree_rows = sum(part["rows"] for part in attribution
                      if part["label"] in {f"MSM {i} complete" for i in range(8)})
    projections = {}
    for name in ("uniform24_development", "mixed_public_degree"):
        removed_inputs = (candidates["baseline"]["stage1"]["witness_curve_inputs"]
                          - candidates[name]["stage1"]["witness_curve_inputs"])
        subtotal = circuit_report["rows"] - degree_rows - removed_inputs * point_costs[0]
        projections[name] = {
            "removed_witness_curve_inputs": removed_inputs,
            "remaining_rows_after_logged_degree_regions_and_point_inputs": subtotal,
            "approximate_rebuilt_witness_table_rows": 236 * (8 * 795 + 7 * 1060),
            "approximate_rows_before_transcript_and_other_changes": subtotal + 236 * (8 * 795 + 7 * 1060),
        }
    identity_sum = sum(row["cap27_identity_full_degree"] for rows in profiles.values() for row in rows)
    sources = [
        "examples/kzg_profile_counts.rs", "experiments/kzg-sizing-report.py",
        "src/ark_adapter/compact.rs", "src/ark_adapter/srs.rs", "src/ark_adapter/pcs.rs",
        "src/ark_adapter/domain.rs", "src/plonkish/foreign.rs", "src/lookup.rs", "src/system.rs",
        "experiments/kzg-wrap/src/native_verifier.rs", "experiments/kzg-wrap/src/native_msm.rs",
        "experiments/kzg-wrap/src/native_curve.rs", "experiments/kzg-wrap/src/outer.rs",
    ]
    report = {
        "scope": "Metadata counts and conditional shape arithmetic; no candidate circuit or proof generated.",
        "assumptions": [
            "Column layouts and next-row openings remain as saved; heights follow each candidate explicitly, including tables in the uniform candidates.",
            "Lookup grouping is retuned with the existing quotient budget of two.",
            "The public_degree policy is hypothetical and has not passed protocol review.",
            "Outer computation height 2^27 is assumed for byte counts, not established by this report.",
            "Dense bytes sum one representation of each matrix; they are not peak RSS or transferred bytes.",
            "Identity degrees assume fixed polynomials have degree below their domain height and extracted witness/quotient polynomials have degree at most 2^28-2.",
        ],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_sha256": {str(path): digest(path) for path in inputs},
        "source_sha256": {path: digest(Path(path)) for path in sources},
        "profiles": profiles, "candidates": candidates,
        "baseline_used_rows": circuit_report["rows"], "required_row_reduction": circuit_report["rows"] - CAP,
        "baseline_row_attribution": attribution, "final_padding_row": 1,
        "measured_rows_per_witness_curve_input": point_costs[0],
        "wrapper_projection": {
            "scope": "Planning arithmetic only, not an exact count or upper bound. Table costs use approximate affine operation costs; transcript savings, fixed-point tables, constant interning, infinity selection, scalar arithmetic and changes to opening MSMs are not recounted.",
            "removed_logged_degree_msm_rows": degree_rows,
            "candidates": projections,
        },
        "conditional_identity_root_bound": {
            "public_maximum_degree": (1 << 28) - 2, "sum_of_identity_degrees": identity_sum,
            "field_modulus": str(FR),
            "negative_log2_union_bound": math.log2(FR) - math.log2(identity_sum),
            "scope": "Fixed nonzero identities and a uniform independent challenge only; excludes extraction, Fiat-Shamir, alpha batching and LogUp errors.",
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for stage, text in raw.items():
        (args.output / f"{stage}.csv").write_text(text)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, candidate in candidates.items():
        first = candidate["stage1"]
        print(f"{name}: stage1={first['dense_bytes_total'] / (1 << 30):.6f} GiB, "
              f"inner={first['compact_proof_bytes']} B, outer={candidate['outer']['compact_proof_bytes']} B, "
              f"packet={candidate['packet_bytes']} B")


if __name__ == "__main__":
    main()
