#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
set -euo pipefail

git status --short --branch
git rev-parse HEAD
rustc --version
cargo --version
nvcc --version
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap --format=csv

if [[ -z "${MULTI_STARK_CUDA_ARCHS:-}" ]]; then
  MULTI_STARK_CUDA_ARCHS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1p' | tr -d '.[:space:]')"
  export MULTI_STARK_CUDA_ARCHS
fi
echo "MULTI_STARK_CUDA_ARCHS=${MULTI_STARK_CUDA_ARCHS}"

cargo clippy --release --locked --all-targets --features parallel,cuda -- -D warnings
cargo test --release --locked --features parallel,cuda \
  cuda::tests::goldilocks_field_kernels_match_cpu -- --test-threads=1
cargo test --release --locked --features parallel,cuda \
  cuda::tests::batched_dft_matches_cpu -- --test-threads=1
cargo test --release --locked --features parallel,cuda \
  cuda::tests::coset_lde_matches_cpu_including_storage_layout -- --test-threads=1
cargo test --release --locked --features parallel,cuda -- --test-threads=1

compat_dir="$(mktemp -d)"
trap 'rm -rf "$compat_dir"' EXIT
cargo run --release --locked --example proof_compatibility -- "$compat_dir/cpu.proof"
echo "25564a01d1d352b1ec2de56b019b641d24acc81083133e79274a86829b2a5dd5  $compat_dir/cpu.proof" \
  | sha256sum --check --status
cargo run --release --locked --features parallel,cuda \
  --example proof_compatibility -- "$compat_dir/cuda.proof"
cmp "$compat_dir/cpu.proof" "$compat_dir/cuda.proof"
