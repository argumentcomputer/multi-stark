#!/usr/bin/env python3
"""Materialize the pinned terminal-verifier crates outside dependency caches."""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REV = "8fdb3eab26491f2e79a6004016c2b7b96d0b4bda"


def prepare(source):
    dest = ROOT / "target/flock-terminal"
    paths = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", "-r", "--name-only", REV, "flock-stage4"],
        text=True,
    ).splitlines()
    for name in paths:
        relative = Path(name).relative_to("flock-stage4")
        if relative.parts[0] not in {"Cargo.toml", "circuit", "trace", "fflonk"}:
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "-C", str(source), "show", f"{REV}:{name}"]))
    exporter = dest / "exporter"
    (exporter / "src").mkdir(parents=True, exist_ok=True)
    text = subprocess.check_output([
        "git", "-C", str(source), "show", f"{REV}:flock-stage3/host/src/stage4.rs"
    ], text=True).replace("pub(crate)", "pub")
    # The upstream application envelope is unused here. Keep it out of this
    # crate so an application-specific binding cannot become our statement.
    text = text.replace("use crate::{STAGE3_STATEMENT_BYTES, Stage3StatementV1};", "")
    for marker in [
        "pub struct Stage4FlockVerifierWitnessV1",
        "impl Stage4FlockVerifierWitnessV1",
        "pub fn export_statement_binding",
    ]:
        begin = text.index(marker)
        # Remove attributes/doc comments belonging to the deleted item.
        previous = text.rfind("\n\n", 0, begin)
        start = previous + 2
        opening = text.index("{", begin)
        depth, end = 1, opening + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        text = text[:start] + text[end:]
    (exporter / "src/lib.rs").write_text(text)
    (exporter / "Cargo.toml").write_text('''[package]
name = "flock-terminal-exporter"
version = "0.1.0"
edition = "2024"
publish = false
[dependencies]
anyhow = "1"
blake3 = "1"
ix-stage4-trace = { path = "../trace" }
flock-prover = { git = "https://github.com/succinctlabs/flock", rev = "b684b1258e4b1f202bec24afd660ace851b09e5e" }
''')
    manifest = dest / "Cargo.toml"
    manifest.write_text(manifest.read_text().replace('"trace"]', '"trace", "exporter"]'))
    patch = Path(__file__).with_name("terminal.patch")
    if patch.exists():
        subprocess.run(["patch", "--batch", "-p1", "-d", str(dest), "-i", str(patch)], check=True)
    print(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ix-source", type=Path, default=ROOT.parent / "ix")
    prepare(parser.parse_args().ix_source)
