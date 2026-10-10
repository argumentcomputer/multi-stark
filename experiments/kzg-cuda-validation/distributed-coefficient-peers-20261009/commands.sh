#!/usr/bin/env bash
set -euo pipefail
cd /home/sam/repos/multi-stark

env MULTI_STARK_CUDA_ARCHS=120 cargo test --release --locked --offline --features parallel,kzg-cuda,cuda --lib distributed_lookup_uses_partner_resident_coefficients --no-run

env MULTI_STARK_KZG_BACKEND=cuda MULTI_STARK_KZG_CUDA_DEVICES=3,2,1,0 MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8 MULTI_STARK_KZG_CUDA_PROFILE=1 target/release/deps/multi_stark-b565c7836c24a940 distributed_lookup_uses_partner_resident_coefficients --ignored --nocapture

for test in distributed_lookup_matches_cpu_coefficients_and_totals distributed_lookup_preserves_accumulators_and_proof_bytes distributed_quotient_matches_cpu_coefficients public_degree_backend_parity_fixture; do
    env MULTI_STARK_KZG_BACKEND=cuda MULTI_STARK_KZG_CUDA_DEVICES=3,2,1,0 MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8 MULTI_STARK_KZG_CUDA_PROFILE=1 MULTI_STARK_KZG_CUDA_DISTRIBUTED_LOOKUP=force MULTI_STARK_KZG_CUDA_DISTRIBUTED_QUOTIENT=force MULTI_STARK_KZG_CUDA_SRS_CACHE=1 target/release/deps/multi_stark-b565c7836c24a940 "$test" --ignored --nocapture
done

env MULTI_STARK_KZG_BACKEND=cuda MULTI_STARK_KZG_CUDA_DEVICES=0,1,2,3 MULTI_STARK_KZG_CUDA_RESIDENT_GIB=8 MULTI_STARK_KZG_CUDA_PROFILE=1 MULTI_STARK_KZG_DISTRIBUTED_BENCH_LOG=27 MULTI_STARK_KZG_DISTRIBUTED_SETUP=/opt/dlami/nvme/multi-stark-kzg-prefetch-20261009/cached/recursive/kzg/setup-0.bin /usr/bin/time -v experiments/kzg-cuda-validation/distributed-coefficient-peers-20261009/multi_stark-tests distributed_lookup_synthetic_benchmark --ignored --nocapture
