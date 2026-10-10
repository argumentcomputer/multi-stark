#!/usr/bin/env python3
import json
from pathlib import Path
import shutil
import subprocess
import time

OUT = Path(__file__).resolve().parent
BINARY = OUT / "test-binary"
RUNS = [
    ("row-parity", ["plonkish::stark::row_selection_tests::", "--nocapture"]),
    ("partition-parity", ["plonkish::stark::partition_tests::", "--nocapture"]),
    (
        "existing-proof-parity",
        [
            "plonkish::hash::tests::copy_and_trace_construction_preserve_fresh_proof_bytes",
            "--exact",
            "--nocapture",
        ],
    ),
    (
        "benchmark",
        [
            "plonkish::stark::row_selection_tests::selected_rows_benchmark",
            "--exact",
            "--ignored",
            "--nocapture",
        ],
    ),
]
receipts = []
for name, arguments in RUNS:
    command = [str(BINARY), *arguments]
    measured = [shutil.which("time"), "-v", "-o", str(OUT / f"{name}.time"), *command]
    start = time.monotonic()
    with (OUT / f"{name}.log").open("w") as log:
        result = subprocess.run(measured, stdout=log, stderr=subprocess.STDOUT)
    receipt = {
        "name": name,
        "command": command,
        "elapsed_seconds": time.monotonic() - start,
        "exit_code": result.returncode,
    }
    receipts.append(receipt)
    (OUT / "runs.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    print((OUT / f"{name}.log").read_text(), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
