#!/usr/bin/env python3
"""Materialize the pinned Flock patches, outside Cargo's cache."""
import argparse
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
REV = "b684b1258e4b1f202bec24afd660ace851b09e5e"
URL = "https://github.com/succinctlabs/flock"


def prepare(source=None):
    source = Path(source) if source else ROOT / "target/flock-upstream"
    if not source.exists():
        subprocess.run(["git", "clone", URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", REV], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REV:
        raise RuntimeError(f"expected Flock {REV}, found {revision}")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError("Flock source has tracked changes")
    for crate in ["flock-core", "flock-prover"]:
        dest = ROOT / f"target/{crate}-patched"
        shutil.copytree(source / "crates" / crate, dest, dirs_exist_ok=True)
        workspace = tomllib.loads((source / "Cargo.toml").read_text())["workspace"]
        manifest = (dest / "Cargo.toml").read_text()
        for key, value in workspace["package"].items():
            manifest = manifest.replace(f"{key}.workspace = true", f"{key} = {json.dumps(value)}")
        for key, value in workspace["dependencies"].items():
            if key.startswith("flock-"):
                value = {"git": URL, "rev": REV}
            if isinstance(value, str):
                value = {"version": value}
            entries = ", ".join(f"{k} = {json.dumps(v)}" for k, v in value.items())
            manifest = manifest.replace(f"{key} = {{ workspace = true }}", f"{key} = {{ {entries} }}")
            manifest = manifest.replace(f"{key} = {{ workspace = true, ", f"{key} = {{ {entries}, ")
        manifest = manifest.replace("[lints]\nworkspace = true", "[lints.clippy]\n" + "\n".join(
            f"{k} = {json.dumps(v)}" for k, v in workspace["lints"]["clippy"].items()))
        manifest = manifest.replace('path = "../flock-core"', 'path = "../flock-core-patched"')
        (dest / "Cargo.toml").write_text(manifest)
        if crate == "flock-core":
            for config in (PACKAGE / "configs").glob("*.toml"):
                shutil.copyfile(config, dest / "configs/ligerito" / config.name)
        subprocess.run(["patch", "--batch", "-p1", "-d", str(dest), "-i", str(PACKAGE / f"{crate}.patch")], check=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    prepare(parser.parse_args().source)
