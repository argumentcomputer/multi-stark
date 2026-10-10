//! Host-backed LDEs computed by exclusively reserved auxiliary CUDA devices.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, TryLockError, Weak};
use std::time::Instant;

use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    CudaDft, check_cuda, device_memory_info, memory_diagnostics_enabled, minimum_free_bytes,
};

type Worker = Mutex<CudaDft>;

#[derive(Clone, Debug, Default)]
pub(super) struct HostLdeDevices {
    workers: Arc<[Arc<Worker>]>,
    next: Arc<AtomicUsize>,
}

fn validate_devices(primary: i32, devices: &[i32]) {
    for (index, &device) in devices.iter().enumerate() {
        assert!(device >= 0, "auxiliary CUDA device id must be non-negative");
        assert_ne!(
            device, primary,
            "auxiliary CUDA device must not be the primary device"
        );
        assert!(
            !devices[..index].contains(&device),
            "duplicate auxiliary CUDA device"
        );
    }
}

pub(super) fn configured_devices(primary: i32) -> Vec<i32> {
    let Some(value) = std::env::var_os("MULTI_STARK_CUDA_AUX_DEVICES") else {
        return Vec::new();
    };
    let value = value
        .into_string()
        .expect("MULTI_STARK_CUDA_AUX_DEVICES must be UTF-8");
    let devices = if value.trim().is_empty() {
        Vec::new()
    } else {
        value
            .split(',')
            .map(|device| {
                device
                    .trim()
                    .parse()
                    .expect("MULTI_STARK_CUDA_AUX_DEVICES must be comma-separated CUDA ordinals")
            })
            .collect()
    };
    validate_devices(primary, &devices);
    devices
}

impl HostLdeDevices {
    pub(super) fn new(primary: i32, devices: &[i32]) -> Self {
        validate_devices(primary, devices);
        static WORKERS: OnceLock<Mutex<BTreeMap<i32, Weak<Worker>>>> = OnceLock::new();
        let mut registry = WORKERS
            .get_or_init(Mutex::default)
            .lock()
            .expect("auxiliary CUDA registry poisoned");
        let workers = devices
            .iter()
            .map(|&device| {
                if let Some(worker) = registry.get(&device).and_then(Weak::upgrade) {
                    worker
                } else {
                    let worker = Arc::new(Mutex::new(CudaDft::without_auxiliary_devices(device)));
                    registry.insert(device, Arc::downgrade(&worker));
                    worker
                }
            })
            .collect::<Vec<_>>()
            .into();
        Self {
            workers,
            next: Arc::default(),
        }
    }

    pub(super) fn try_coset_lde(
        &self,
        matrix: &RowMajorMatrix<Goldilocks>,
        added_bits: usize,
        shift: Goldilocks,
    ) -> Option<RowMajorMatrix<Goldilocks>> {
        // Small transforms cost less than auxiliary-device scheduling and transfers.
        if self.workers.is_empty() || matrix.width() == 0 || matrix.values.len() < 1 << 15 {
            return None;
        }
        let started = Instant::now();
        let start = self.next.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        let mut busy = None;
        for offset in 0..self.workers.len() {
            let index = (start + offset) % self.workers.len();
            match self.workers[index].try_lock() {
                Ok(worker) => {
                    if let Some(lde) = try_worker(&worker, matrix, added_bits, shift, started) {
                        return Some(lde);
                    }
                }
                Err(TryLockError::WouldBlock) => {
                    busy.get_or_insert(index);
                }
                Err(TryLockError::Poisoned(_)) => panic!("auxiliary CUDA worker poisoned"),
            }
        }
        // A busy device should not turn a GPU-admissible matrix into CPU work.
        let worker = self.workers[busy?]
            .lock()
            .expect("auxiliary CUDA worker poisoned");
        try_worker(&worker, matrix, added_bits, shift, started)
    }
}

fn admitted_bytes(
    input: usize,
    output: usize,
    scratch: usize,
    constants: usize,
    reserve: usize,
) -> Option<usize> {
    input
        .checked_add(output)?
        .checked_add(scratch)?
        .checked_add(constants)?
        .checked_add(reserve)
}

fn try_worker(
    worker: &CudaDft,
    matrix: &RowMajorMatrix<Goldilocks>,
    added_bits: usize,
    shift: Goldilocks,
    started: Instant,
) -> Option<RowMajorMatrix<Goldilocks>> {
    let _previous_device = RestoreDevice::capture();
    let plan = worker.lde_plan(matrix.height(), matrix.width(), added_bits, shift);
    let input_bytes = matrix.values.len().checked_mul(size_of::<Goldilocks>())?;
    let blowup = 1usize.checked_shl(u32::try_from(added_bits).ok()?)?;
    let output_bytes = input_bytes.checked_mul(blowup)?;
    let (free_bytes, total_bytes) = device_memory_info(worker.device_id);
    // Constants are counted even if their cached allocation is already included
    // in the snapshot. The reserve also covers lazy sppark device parameters.
    let required_bytes = admitted_bytes(
        input_bytes,
        output_bytes,
        plan.scratch_bytes(),
        plan.constant_bytes(),
        minimum_free_bytes(total_bytes),
    )?;
    if required_bytes > free_bytes {
        return None;
    }
    let compute_started = Instant::now();
    let resident = worker.coset_lde_batch_resident(matrix, added_bits, shift);
    let compute_seconds = compute_started.elapsed().as_secs_f64();
    let download_started = Instant::now();
    let result = resident.to_row_major_matrix();
    let download_seconds = download_started.elapsed().as_secs_f64();
    drop(resident);
    // Pool frees run on the calling thread's stream. Complete them before a
    // different thread can reuse this device's serialized worker and budget.
    // SAFETY: all allocations and launches above select the worker's device.
    check_cuda(unsafe { cudaDeviceSynchronize() }, "auxiliary LDE release");
    if memory_diagnostics_enabled() {
        eprintln!(
            "[multi-stark/cuda] host LDE offload: device={} height={} width={} added_bits={added_bits} input_bytes={input_bytes} output_bytes={output_bytes} admitted_bytes={required_bytes} upload_compute_s={compute_seconds:.6} download_s={download_seconds:.6} total_s={:.6}",
            worker.device_id,
            matrix.height(),
            matrix.width(),
            started.elapsed().as_secs_f64()
        );
    }
    Some(result)
}

struct RestoreDevice(i32);

impl RestoreDevice {
    fn capture() -> Self {
        let mut device = 0;
        check_cuda(
            unsafe { cudaGetDevice(&mut device) },
            "auxiliary LDE caller device",
        );
        Self(device)
    }
}

impl Drop for RestoreDevice {
    fn drop(&mut self) {
        // CUDA device selection is thread-local; no handle changes ownership.
        let _ = unsafe { cudaSetDevice(self.0) };
    }
}

unsafe extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn admission_counts_all_allocations_without_overflow() {
        assert_eq!(admitted_bytes(8, 32, 40, 8, 100), Some(188));
        assert_eq!(admitted_bytes(usize::MAX, 1, 0, 0, 0), None);
        assert_eq!(admitted_bytes(8, 32, 40, 8, usize::MAX), None);
    }

    #[test]
    fn auxiliary_device_selection_is_explicit_and_unique() {
        validate_devices(0, &[]);
        validate_devices(0, &[3, 1, 2]);
        for devices in [&[0][..], &[1, 1], &[-1]] {
            assert!(std::panic::catch_unwind(|| validate_devices(0, devices)).is_err());
        }
    }

    #[test]
    #[ignore = "requires two CUDA devices"]
    fn auxiliary_workers_share_device_lock_and_restore_context() {
        let first = HostLdeDevices::new(0, &[1]);
        let second = HostLdeDevices::new(0, &[1]);
        assert!(Arc::ptr_eq(&first.workers[0], &second.workers[0]));
        let _ = device_memory_info(0);
        let matrix = RowMajorMatrix::new(vec![Goldilocks::ONE; (1 << 10) * 32], 32);
        let result = first.try_coset_lde(&matrix, 2, Goldilocks::ONE).unwrap();
        assert!(result.values.iter().all(|&value| value == Goldilocks::ONE));
        let mut current = -1;
        check_cuda(unsafe { cudaGetDevice(&mut current) }, "test caller device");
        assert_eq!(current, 0);
    }
}
