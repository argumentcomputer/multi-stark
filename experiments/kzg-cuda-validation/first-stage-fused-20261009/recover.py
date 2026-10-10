#!/usr/bin/env python3
"""Rebuild the report from preserved timing, log and independent verification evidence."""

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import re

import run


spec = importlib.util.spec_from_file_location("recovery_helpers", run.HERE / "benchmark_helpers-quality.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main():
    receipt = json.loads((run.HERE / "report-before-recovery.json").read_text())
    report = dict(receipt)
    report.pop("error", None)
    for path, digest in ((run.BINARY, receipt["binary_sha256"]),
                         (run.HERE / "source.tar.gz", receipt["source_archive_sha256"]),
                         (run.HERE / "run.py", receipt["runner_sha256"]),
                         (run.HERE / "benchmark_helpers.py", receipt["helpers_sha256"])):
        if helpers.sha256(path) != digest:
            raise RuntimeError(f"immutable measurement input changed: {path}")
    log_path = run.HERE / "stage-and-prove.log"
    log = log_path.read_text()
    time_path = run.HERE / "stage-and-prove.time"
    measured = json.loads(time_path.read_text().splitlines()[-1])
    if measured["exit_code"]:
        raise RuntimeError("proof subprocess did not succeed")
    measured["peak_rss_bytes"] = measured.pop("peak_rss_kib") * 1024
    finished = datetime.fromtimestamp(time_path.stat().st_mtime, timezone.utc)
    started = finished - timedelta(seconds=measured["wall_seconds"])
    measured.update(
        wall_time_source="GNU time external subprocess boundary, rounded to0.01s; no proof rerun",
        started_at_utc=started.isoformat(), finished_at_utc=finished.isoformat(),
        utc_bounds_source="estimated from timing-file completion mtime and rounded external wall time",
        command=[str(run.BINARY), "stage-and-prove", str(run.HERE / "compressed-fri"), str(run.OUTPUT / "fri-to-kzg")],
        backend="cuda", environment=receipt["environment"], log=str(log_path),
        fixed_cache_hits=log.count("Reused fixed preprocessing:"),
        fixed_cache_misses=log.count("Fixed preprocessing cache miss:"),
        srs_cache_hits=log.count("Loaded development SRS cache"),
        srs_cache_misses=log.count("Generated and cached development SRS"),
        host_operation_totals_overlap=helpers.summarize_host_operations(log_path),
        preparation_interval_totals_overlap=run.timer_totals(log),
        gpu_samples_every_2_seconds=helpers.summarize_gpu_samples(run.HERE / "gpu-samples.csv"),
    )
    if (measured["fixed_cache_hits"], measured["fixed_cache_misses"], measured["srs_cache_hits"], measured["srs_cache_misses"]) != (0, 1, 1, 0):
        raise RuntimeError("cache state differs from the specified experiment")
    (measured["cuda_operation_totals_overlap"], measured["cuda_profile_parse"]) = helpers.summarize_cuda_events(log_path, with_quality=True)
    cumulative = {}
    for key, prefix in (
        ("source_circuit", "Source circuit:"),
        ("source_assignment", "Source witness satisfied:"),
        ("scalar_translation", "Scalar translation:"),
        ("scalar_assignment", "Scalar witness satisfied:"),
        ("lowering", "Lowered 19 circuits:"),
        ("fixed_staging_complete", "Staged 18/19:"),
        ("stage_complete", "STAGE-AND-PROVE COMPLETE:"),
    ):
        found = re.search(re.escape(prefix) + r" ([0-9.]+)s", log)
        if not found:
            raise RuntimeError(f"missing cumulative boundary: {prefix}")
        cumulative[key] = float(found.group(1))
    report["frontend_cumulative_seconds"] = cumulative
    last = 0
    report["frontend_intervals_seconds"] = {}
    for name in ("source_circuit", "source_assignment", "scalar_translation", "scalar_assignment", "lowering"):
        report["frontend_intervals_seconds"][name] = cumulative[name] - last
        last = cumulative[name]
    report["frontend_total_seconds"] = cumulative["lowering"]
    report["cold_fixed_staging_boundary_seconds"] = cumulative["fixed_staging_complete"] - cumulative["lowering"]
    report["prover_prepare_cumulative_seconds"] = float(re.search(r"Proving all 19 partitions together: ([0-9.]+)s", log).group(1))
    report["prover_total_seconds"] = float(re.search(r"VERIFIED:.* total_seconds=([0-9.]+)", log).group(1))
    report["gpu_sampling_scope"] = "primary stage only, including subsecond monitor startup/shutdown; CPU verification is outside sampling"
    cpu = json.loads((run.HERE / "cpu-verification.json").read_text())
    if cpu["exit_code"] or cpu["children_before_verification"] or cpu["children_after_verification"]:
        raise RuntimeError("independent verification or process cleanup did not pass")
    staged = run.OUTPUT / "fri-to-kzg"
    proof = helpers.proof_hashes(staged / "kzg")
    helpers.require_hashes(proof, receipt["expected_proof_artifacts"], "legacy stage1 artifacts")
    report["proof_parity"] = proof
    report["fused_output_checks"] = {
        "witness_files_absent": not any(staged.glob("*.witness.zst")),
        "main_checkpoints_absent": not any((staged / "kzg").glob("main-*.bin")),
    }
    if not all(report["fused_output_checks"].values()):
        raise RuntimeError("fused output has unexpected witness/main checkpoints")
    report["phases"] = {"stage-and-prove": measured, "cpu-verify": cpu}
    report["reporting_recovery"] = {
        "original_receipt": "report-before-recovery.json",
        "original_receipt_sha256": helpers.sha256(run.HERE / "report-before-recovery.json"),
        "original_error": receipt["error"],
        "original_helper_sha256": receipt["helpers_sha256"],
        "corrected_helper_file": "benchmark_helpers-quality.py",
        "corrected_helper_sha256": helpers.sha256(run.HERE / "benchmark_helpers-quality.py"),
        "recovery_script_sha256": helpers.sha256(Path(__file__)),
        "raw_log_sha256": helpers.sha256(log_path),
        "proof_rerun": False,
        "note": "The proof process had exit0 and a VERIFIED result before reporting failed. Invalid CUDA records are skipped wholly; their raw lines/reasons are retained and event totals marked incomplete.",
    }
    report["status"] = "verified_legacy_stage1_artifact_parity_cold_fixed"
    report["finished_at_utc"] = cpu["finished_at_utc"]
    report["recovered_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run.HERE / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "wall_seconds": measured["wall_seconds"],
                      "frontend_intervals_seconds": report["frontend_intervals_seconds"],
                      "cuda_records_skipped": measured["cuda_profile_parse"]["records_skipped"],
                      "cpu_verify_seconds": cpu["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
