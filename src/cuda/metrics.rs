//! CPU-only snapshots of counters accumulated at CUDA operation boundaries.

pub(crate) fn emit_snapshot() {
    if !tracing::enabled!(target: "prover_metrics", tracing::Level::INFO) {
        return;
    }
    const SHAPES: usize = 33 * 4;
    const WORDS: usize = 16 + 2 * SHAPES;
    let mut values = vec![0u64; 64 * WORDS];
    unsafe {
        multi_stark_cuda_metrics_snapshot(values.as_mut_ptr(), values.len());
    }
    for (device, c) in values.chunks_exact(WORDS).enumerate() {
        if c.iter().all(|&v| v == 0) {
            continue;
        }
        tracing::info!(target: "prover_metrics", metric = "cuda_device_snapshot", device,
            scope = "process_device_cumulative", upload_calls = c[0], upload_requested_bytes = c[1],
            upload_chunks = c[2], upload_failures = c[3], upload_host_ns = c[4],
            coset_hits = c[5], coset_misses = c[6], coset_uploaded_bytes = c[7], constant_bytes = c[8],
            last_driver_free_bytes = c[9], total_bytes = c[10], memory_samples = c[11],
            sppark_taken = c[12], sppark_declined = c[13]);
        for (backend, shapes) in [
            ("legacy", &c[16..16 + SHAPES]),
            ("sppark", &c[16 + SHAPES..]),
        ] {
            for (shape, &count) in shapes.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                tracing::info!(target: "prover_metrics", metric = "ntt_snapshot", device,
                    scope = "process_device_cumulative", backend, log_height = shape / 4,
                    width_bucket = ["1", "2", "3-7", "8+"][shape % 4], transforms = count);
            }
        }
    }
}

unsafe extern "C" {
    fn multi_stark_cuda_metrics_snapshot(output: *mut u64, count: usize);
}
