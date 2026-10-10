#!/usr/bin/env bash
set -euo pipefail
cd /home/sam/repos/multi-stark

# The archived binary is the measured build. Rebuilding is only needed to test current sources.
env MULTI_STARK_CUDA_ARCHS=120 cargo test --release --locked --offline --features parallel,kzg-cuda,cuda --lib distributed_quotient --no-run

target/release/deps/multi_stark-b565c7836c24a940 distributed_quotient_plan_bounds_and_pairs --nocapture

env MULTI_STARK_KZG_BACKEND=cuda MULTI_STARK_KZG_CUDA_DEVICES=0,1,2,3 MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8 MULTI_STARK_KZG_CUDA_PROFILE=1 target/release/deps/multi_stark-b565c7836c24a940 distributed_quotient_matches_cpu_coefficients --ignored --nocapture

env MULTI_STARK_KZG_BACKEND=cuda MULTI_STARK_KZG_CUDA_DEVICES=0,1,2,3 MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8 MULTI_STARK_KZG_CUDA_PROFILE=1 MULTI_STARK_KZG_CUDA_DISTRIBUTED_QUOTIENT=force MULTI_STARK_KZG_CUDA_SRS_CACHE=1 target/release/deps/multi_stark-b565c7836c24a940 public_degree_backend_parity_fixture --ignored --nocapture

env MULTI_STARK_KZG_BACKEND=cuda MULTI_STARK_KZG_CUDA_DEVICES=0,1,2,3 MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8 MULTI_STARK_KZG_CUDA_PROFILE=1 MULTI_STARK_KZG_DISTRIBUTED_BENCH_LOG=10 MULTI_STARK_KZG_DISTRIBUTED_SETUP=/opt/dlami/nvme/multi-stark-kzg-prefetch-20261009/cached/recursive/kzg/setup-0.bin target/release/deps/multi_stark-b565c7836c24a940 distributed_quotient_synthetic_benchmark --ignored --nocapture

env MULTI_STARK_KZG_BACKEND=cuda MULTI_STARK_KZG_CUDA_DEVICES=0,1,2,3 MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8 target/release/deps/multi_stark-b565c7836c24a940 resident_quotient_matches_cpu_coefficients --ignored --nocapture

env MULTI_STARK_KZG_BACKEND=cuda MULTI_STARK_KZG_CUDA_DEVICES=0,1,2,3 MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8 MULTI_STARK_KZG_CUDA_PROFILE=1 MULTI_STARK_KZG_DISTRIBUTED_BENCH_LOG=27 MULTI_STARK_KZG_DISTRIBUTED_SETUP=/opt/dlami/nvme/multi-stark-kzg-prefetch-20261009/cached/recursive/kzg/setup-0.bin /usr/bin/time -v experiments/kzg-cuda-validation/distributed-quotient-20261009/multi_stark-tests distributed_quotient_synthetic_benchmark --ignored --nocapture

nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version --format=csv,noheader
