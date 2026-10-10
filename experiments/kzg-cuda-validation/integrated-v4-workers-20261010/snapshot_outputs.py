#!/usr/bin/env python3
"""Archive benchmark evidence without copying large trace and coefficient files."""

import hashlib
import json
from pathlib import Path
import shutil


archive = Path(__file__).resolve().parent
source = Path("/opt/dlami/nvme/multi-stark-kzg-integrated-v4-20261010")
target = archive / "outputs"
target.mkdir()
files, large = {}, {}
directories = ["", "compressed-fri", "root-artifacts", "worker-seed", "worker-seed/seed-fri",
               "first_stage-worker", "wrapper-worker", "intermediate", "worker-warmup/intermediate"]
for prefix in ("", "worker-warmup/"):
    for stage in ("fri-to-kzg", "recursive"):
        directories.extend([prefix + stage, prefix + stage + "/kzg"])
for directory in directories:
    for path in sorted((source / directory).iterdir()):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if path.name.endswith(".zst") or path.name.startswith(("fixed-", "main-")):
            large[relative] = {"path": str(path), "bytes": path.stat().st_size}
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        with destination.open("rb") as contents:
            digest = hashlib.file_digest(contents, "sha256").hexdigest()
        files[relative] = {"bytes": destination.stat().st_size, "sha256": digest}
(archive / "output-files.json").write_text(json.dumps(files, indent=2) + "\n")
(archive / "large-output-files.json").write_text(json.dumps(large, indent=2) + "\n")
print(json.dumps({"copied_files": len(files), "copied_bytes": sum(item["bytes"] for item in files.values()),
                  "large_files_kept_on_nvme": len(large)}, indent=2))
