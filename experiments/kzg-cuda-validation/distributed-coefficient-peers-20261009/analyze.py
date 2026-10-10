#!/usr/bin/env python3
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GIB = 1 << 30
previous = json.loads((ROOT.parent / "distributed-lookup-20261009/report.json").read_text())
log = (ROOT / "lookup-n27.log").read_text()
keys = (
    "upload_ms", "kernel_ms", "download_ms", "host_copy_ms",
    "upload_bytes", "download_bytes", "device_copy_bytes",
)


def values(line):
    return dict(re.findall(r"(\w+)=([^ ]+)", line))


def profiles(block):
    result = {key: 0 for key in keys}
    for line in block.splitlines():
        if not line.startswith("KZG CUDA profile "):
            continue
        fields = values(line)
        for key in keys:
            result[key] += float(fields[key]) if key.endswith("_ms") else int(fields[key])
    return {key: round(value, 3) if key.endswith("_ms") else value for key, value in result.items()}


def phase(name):
    line = next(line for line in log.splitlines() if f"phase={name}_done " in line)
    return {
        "seconds": float(re.search(r"seconds=([\d.]+)", line)[1]),
        "pool_peak_bytes": json.loads(re.search(r"pool_peak_bytes=(\[[^]]+\])", line)[1]),
        "rss_peak_kib": int(re.search(r"rss_peak_kib=Some\((\d+)\)", line)[1]),
        "total": re.search(r"total=Scalar\((\d+)\)", line)[1],
    }


cpu = phase("cpu")
gpu = phase("distributed")
assert cpu["total"] == gpu["total"] == previous["final_total_decimal"]
assert "coefficient_parity=true total_parity=true" in log
assert "test result: ok. 1 passed" in log
cpu_block = log.split("phase=cpu_start", 1)[1].split("phase=cpu_done", 1)[0]
gpu_block = log.split("phase=distributed_start", 1)[1].split("phase=distributed_done", 1)[0]
devices = [
    {key: int(value) for key, value in values(line).items()}
    for line in gpu_block.splitlines() if line.startswith("KZG CUDA distributed ")
]
details = {
    key: sum(device[key] for device in devices)
    for key in devices[0] if key not in ("device", "live_peak_bytes")
}
assert details["coefficient_upload_bytes"] == 112 * GIB
assert details["coefficient_local_bytes"] == details["coefficient_peer_bytes"] == 16 * GIB
assert sum(details[key] for key in ("coefficient_upload_bytes", "coefficient_local_bytes", "coefficient_peer_bytes")) == 144 * GIB
transfers = {"bounded_cpu_lookup": profiles(cpu_block), "distributed": profiles(gpu_block)}
assert transfers["distributed"]["upload_bytes"] == 120 * GIB + 13088
assert transfers["distributed"]["download_bytes"] == 24 * GIB + 288
assert transfers["distributed"]["device_copy_bytes"] == 112 * GIB + 294912

fixture_log = (ROOT / "kzg-resident-peer-parity.log").read_text()
blocks = re.split(r"peer_coefficient_fixture force_host=(false|true)[^\n]*\n", fixture_log)
fixture_counters = {}
for offset in range(1, len(blocks), 2):
    mode, block = blocks[offset:offset + 2]
    totals = dict.fromkeys(("coefficient_upload_bytes", "coefficient_local_bytes", "coefficient_peer_bytes"), 0)
    for line in block.splitlines():
        if line.startswith("KZG CUDA distributed "):
            fields = values(line)
            for key in totals:
                totals[key] += int(fields[key])
    expected = {
        "coefficient_upload_bytes": 2097568 if mode == "false" else 4195136,
        "coefficient_local_bytes": 0,
        "coefficient_peer_bytes": 2097568 if mode == "false" else 0,
    }
    assert totals == expected
    fixture_counters["force_host=" + mode] = totals

tests = {}
for path in ROOT.glob("kzg-resident-peer-*.log"):
    if path.name.endswith("build.log"):
        continue
    content = path.read_text()
    assert "test result: ok. 1 passed" in content, path
    tests[path.name] = float(re.search(r"finished in ([\d.]+)s", content)[1])
digest = "8164e8f1acee3f94dbb48d29908d4e99085af60f3a8a96f4d09b8c6405c69bca"
assert f"bytes=1205 blake3={digest}" in (ROOT / "kzg-resident-peer-public_degree_backend_parity_fixture.log").read_text()

report = {
    "schema": "multi-stark/distributed-coefficient-peers/v1",
    "date": "2026-10-09",
    "source_revision": (ROOT / "git-revision.txt").read_text().strip(),
    "test_binary_sha256": hashlib.sha256((ROOT / "multi_stark-tests").read_bytes()).hexdigest(),
    "source_snapshot_directory": "source",
    "source_sha256_file": "source-sha256.json",
    "change_file": "peer-change.patch",
    "commands_file": "commands.sh",
    "hardware_file": "gpus.csv",
    "hardware_metadata_source": "Copied from preceding measurements on the same four cards and driver; hardware unchanged.",
    "runtime_peer_pairs": [[0, 1], [2, 3]],
    "cpu_parallelism": "Default Rayon width and inherited CPU affinity; no manual thread cap.",
    "implementation": "Both distributed quotient and lookup use the shared input loader. It retains same-device D2D priority, then accepts an immutable resident source only on the actual acquired partner ordinal. The destination compute stream performs cudaMemcpyPeerAsync before zero filling and the stock NTT. No new allocations or column reordering. Forced host staging disables the peer coefficient branch. Resident handles remain borrowed until all selected devices drain, including errors.",
    "setup_source": previous["setup_source"],
    "setup_sha256": hashlib.sha256((ROOT / "setup-0.bin").read_bytes()).hexdigest(),
    "trace_rows": previous["trace_rows"],
    "widths": previous["widths"],
    "nonconstant_input_columns": 18,
    "lookup_messages": 8,
    "lookup_group_size": 2,
    "lookup_groups": 4,
    "lookup_prefix_nodes": 20,
    "tile_rows": 1 << 18,
    "coefficient_residency_cap_per_device": 8 * GIB,
    "input_semantics": previous["input_semantics"],
    "beta": 23,
    "gamma": 31,
    "comparison_semantics": previous["comparison_semantics"],
    "available_host_bytes_before_preparation": int(re.search(r"available_host_bytes=(\d+)", log)[1]),
    "conservative_host_admission_bytes": 344 * GIB,
    "preparation_seconds": float(re.search(r"prepared seconds=([\d.]+)", log)[1]),
    "bounded_cpu_lookup_seconds": cpu["seconds"],
    "distributed_seconds": gpu["seconds"],
    "speedup_over_current_cpu_reference": cpu["seconds"] / gpu["seconds"],
    "previous_report": "../distributed-lookup-20261009/report.json",
    "previous_distributed_seconds": previous["distributed_seconds"],
    "observed_distributed_reduction_seconds": previous["distributed_seconds"] - gpu["seconds"],
    "observed_distributed_reduction_percent": 100 * (1 - gpu["seconds"] / previous["distributed_seconds"]),
    "complete_process_seconds": 31.59,
    "coefficients_compared": 4 * (1 << 27),
    "exact_coefficient_parity": True,
    "exact_total_parity": True,
    "final_total_decimal": gpu["total"],
    "transfer_counters": transfers,
    "distributed_device_counters": devices,
    "distributed_detail_totals": details,
    "transfer_interpretation": {
        "initial_retained_bytes": 32 * GIB,
        "previous_coefficient_upload_bytes": 140 * GIB,
        "observed_coefficient_upload_reduction_bytes": 28 * GIB,
        "explicit_coefficient_peer_copy_bytes": 16 * GIB,
        "additional_local_matches_between_samples_bytes": 12 * GIB,
        "hypothetical_upload_without_peer_at_current_placement_bytes": 128 * GIB,
        "current_host_traffic_bytes": 144 * GIB + 13376,
        "current_cpu_host_traffic_bytes": 144 * GIB,
        "placement_caveat": "Retention runs in parallel, independently of slot ownership, so the two samples have different within-pair local matches. The current sample reuses 16 GiB locally and 16 GiB through peers; the previous sample reused 4 GiB locally. Of the observed 28 GiB H2D decrease, 16 GiB are explicit peer reads and 12 GiB are additional local matches. With this patch all 32 GiB retained inputs are reused once regardless of within-pair matching, giving 112 GiB coefficient H2D for two input copies totaling 144 GiB. Per-device H2D remains unbalanced at 28, 28, 20, 36 GiB; the largest lane still uploads its complete 36 GiB share.",
        "metadata_upload_bytes": 13088,
        "scalar_download_bytes": 288,
        "cross_pair_staged_logical_bytes": 8 * GIB,
        "cross_pair_staged_physical_bytes": 16 * GIB,
        "final_coefficient_download_bytes": 16 * GIB,
    },
    "memory": {
        "bounded_cpu_pool_used_high_water_bytes_per_device": cpu["pool_peak_bytes"],
        "distributed_pool_used_high_water_bytes_per_device": gpu["pool_peak_bytes"],
        "distributed_sampled_live_bytes_per_device": [device["live_peak_bytes"] for device in devices],
        "bounded_cpu_rss_peak_kib": cpu["rss_peak_kib"],
        "distributed_rss_peak_kib": gpu["rss_peak_kib"],
        "rss_peak_reset_succeeded_before_each_phase": True,
        "notes": previous["memory"]["notes"],
    },
    "validation": {
        "seconds_by_raw_log": tests,
        "deliberately_mismatched_resident_source": {"coefficient_count": 65549, "padded_rows": 131072, "selection": [3, 2, 1, 0], "coefficient_bytes": 2097568, "expected_and_observed_counters": fixture_counters},
        "coverage": "Exact coefficients and accumulator; partial coefficient length with odd rounded allocation tail and zero fill; async pool partner read; forced-host fallback; complete lease return and resident deallocation. Existing distributed lookup geometry, quarter offsets, admission, constants and forced-host cases; distributed quotient regression; cross-circuit proof and nonzero initial accumulator; fresh public proof with both distributed consumers forced and SRS cache enabled.",
        "cross_circuit_proof_bytes": 1573,
        "fresh_public_proof_bytes": 1205,
        "fresh_public_proof_blake3": digest,
        "independent_source_review": "No blocking arithmetic, ownership, stream-order, lifetime or accounting issue found.",
    },
    "event_timing_caveat": previous["event_timing_caveat"],
    "limitations": [
        "One affected lookup-only sample, not a paired fixed-placement performance experiment or a statistical speedup claim.",
        "Synthetic coefficients with the saved actual graph; no real n27 wrapper checkpoint, ceremony SRS or end-to-end proof timing.",
        "Both consumers share the changed helper and passed focused parity; no new target-size quotient timing was run.",
        "All retained bytes are reused, but immutable input placement still produces unequal per-device upload work.",
    ],
}
(ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
(ROOT / "transfer-counters.json").write_text(json.dumps({"profiles": transfers, "devices": devices, "detail_totals": details, "fixture_counters": fixture_counters}, indent=2) + "\n")
print(json.dumps({"seconds": gpu["seconds"], "reduction_percent": report["observed_distributed_reduction_percent"], "transfers": transfers, "details": details}, indent=2))
