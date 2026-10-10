//! Bounded, synchronous BLS12-381 operations on the visible sppark devices.

use ark_bls12_381::{Fq, Fr, G1Affine, G1Projective};
use ark_ff::{AdditiveGroup, BigInt, Field, Zero};
use blst as _;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

#[cfg(test)]
mod diagnostic;
mod distributed;
#[cfg(test)]
mod idle_tests;
pub(super) mod lookup;
pub(super) mod quotient;
pub(super) mod srs_cache;

#[repr(C)]
struct FftJob {
    input: *const [u64; 4],
    resident_input: *const c_void,
    output: *mut [u64; 4],
    input_count: usize,
    resident_output: *mut *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct NativeIdleMemory {
    driver_free_bytes: u64,
    total_bytes: u64,
    current_pool_reserved_bytes: u64,
    current_pool_used_bytes: u64,
    default_pool_reserved_bytes: u64,
    default_pool_used_bytes: u64,
    current_pool_is_default: u64,
}

unsafe extern "C" {
    fn multi_stark_kzg_range_push(name: *const std::ffi::c_char);
    fn multi_stark_kzg_range_pop();
    fn multi_stark_kzg_fft_batch(
        device: i32,
        jobs: *const FftJob,
        job_count: usize,
        lg: u32,
        inverse: bool,
        shift: *const [u64; 4],
        slots: usize,
    ) -> i32;
    fn multi_stark_kzg_polynomial_upload(
        device: i32,
        input: *const [u64; 4],
        count: usize,
        output: *mut *mut c_void,
    ) -> i32;
    fn multi_stark_kzg_polynomial_destroy(device: i32, pointer: *mut c_void) -> i32;

    fn multi_stark_kzg_devices(count: *mut i32) -> i32;
    fn multi_stark_kzg_device_ordinal(index: i32, ordinal: *mut i32) -> i32;
    fn multi_stark_kzg_memory(device: i32, free: *mut usize, total: *mut usize) -> i32;
    fn multi_stark_kzg_idle_memory(device: i32, trim: bool, report: *mut NativeIdleMemory) -> i32;
    fn multi_stark_kzg_peer_access(accessor: i32, owner: i32, supported: *mut i32) -> i32;
    fn multi_stark_kzg_msm_create(
        device: i32,
        out: *mut *mut c_void,
        points: *const c_void,
        count: usize,
    ) -> i32;
    fn multi_stark_kzg_msm_invoke(
        device: i32,
        context: *mut c_void,
        out: *mut [[u64; 6]; 3],
        scalars: *const [u64; 4],
        count: usize,
    ) -> i32;
    fn multi_stark_kzg_msm_invoke_resident(
        device: i32,
        context: *mut c_void,
        out: *mut [[u64; 6]; 3],
        polynomial: *const c_void,
        offset: usize,
        count: usize,
        normalization: *const [u64; 4],
    ) -> i32;
    fn multi_stark_kzg_msm_destroy(device: i32, context: *mut c_void) -> i32;
    fn multi_stark_kzg_fft(
        device: i32,
        values: *mut [u64; 4],
        lg: u32,
        inverse: bool,
        shift: *const [u64; 4],
    ) -> i32;
    #[cfg(test)]
    fn multi_stark_kzg_fft_from(
        device: i32,
        input: *const [u64; 4],
        input_count: usize,
        output: *mut [u64; 4],
        lg: u32,
        inverse: bool,
        shift: *const [u64; 4],
    ) -> i32;
    fn multi_stark_kzg_evaluate_resident(
        device: i32,
        coefficients: *const [u64; 4],
        resident: *const c_void,
        count: usize,
        points: *const [u64; 4],
        point_count: usize,
        results: *mut [u64; 4],
    ) -> i32;
    fn multi_stark_kzg_divide(
        device: i32,
        input: *const [u64; 4],
        output: *mut [u64; 4],
        count: usize,
        z: *const [u64; 4],
    ) -> i32;
    fn multi_stark_kzg_divide_resident(
        device: i32,
        input: *const [u64; 4],
        count: usize,
        z: *const [u64; 4],
        output: *mut *mut c_void,
    ) -> i32;
    fn multi_stark_kzg_polynomial_sample(
        device: i32,
        polynomial: *const c_void,
        offset: usize,
        count: usize,
        samples: *mut [u64; 4],
    ) -> i32;

}

pub(super) struct ProfileRange(std::marker::PhantomData<std::rc::Rc<()>>);

impl ProfileRange {
    pub(super) fn new(name: &std::ffi::CStr) -> Self {
        // NVTX copies the label and pairs ranges on the calling host thread.
        unsafe { multi_stark_kzg_range_push(name.as_ptr()) };
        Self(std::marker::PhantomData)
    }
}

impl Drop for ProfileRange {
    fn drop(&mut self) {
        unsafe { multi_stark_kzg_range_pop() };
    }
}

// The FFI names explicit limb arrays. These checks reject an arkworks layout
// change before its field storage can be borrowed without a packing copy.
const _: () = {
    assert!(size_of::<Fr>() == size_of::<[u64; 4]>());
    assert!(align_of::<Fr>() == align_of::<[u64; 4]>());
    assert!(std::mem::offset_of!(Fr, 0) == 0);
    assert!(std::mem::offset_of!(BigInt<4>, 0) == 0);
    assert!(size_of::<G1Affine>() == 104);
    assert!(std::mem::offset_of!(G1Affine, x) == 0);
    assert!(std::mem::offset_of!(G1Affine, y) == 48);
    assert!(std::mem::offset_of!(G1Affine, infinity) == 96);
    assert!(size_of::<Fq>() == size_of::<[u64; 6]>());
    assert!(std::mem::offset_of!(Fq, 0) == 0);
    assert!(std::mem::offset_of!(BigInt<6>, 0) == 0);
};

fn check(code: i32, operation: &str) {
    assert_eq!(code, 0, "KZG CUDA {operation} failed (CUDA status {code})");
}

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("MULTI_STARK_KZG_BACKEND").as_deref() {
            Ok("cpu") => false,
            Ok("cuda") | Err(std::env::VarError::NotPresent) => true,
            _ => panic!("MULTI_STARK_KZG_BACKEND must be cpu or cuda"),
        },
    )
}

struct Devices {
    ids: Vec<i32>,
    busy: Mutex<Vec<bool>>,
    available: Condvar,
    chunk_points: usize,
    resident_bytes: Mutex<Vec<usize>>,
    resident_limits: Vec<usize>,
}

static DEVICES: OnceLock<Devices> = OnceLock::new();

impl Devices {
    fn potential_budget(&self, index: usize) -> usize {
        let (mut free, mut total) = (0, 0);
        check(
            unsafe { multi_stark_kzg_memory(self.ids[index], &mut free, &mut total) },
            "memory query",
        );
        free.saturating_add(srs_cache::reclaimable(index))
            .saturating_sub(total / 4)
    }

    fn get() -> &'static Self {
        DEVICES.get_or_init(|| {
            let mut count = 0;
            // The FFI writes one integer and initializes sppark's device registry.
            check(unsafe { multi_stark_kzg_devices(&mut count) }, "device discovery");
            assert!(count > 0, "kzg-cuda requires a visible CUDA GPU; use MULTI_STARK_KZG_BACKEND=cpu for CPU execution");
            let supported: Vec<i32> = (0..count).map(|index| {
                let mut ordinal = -1;
                // Logical sppark indices need not match CUDA ordinals on mixed GPU hosts.
                check(unsafe { multi_stark_kzg_device_ordinal(index, &mut ordinal) }, "device ordinal");
                ordinal
            }).collect();
            let ids: Vec<i32> = std::env::var("MULTI_STARK_KZG_CUDA_DEVICES").map_or_else(
                |_| supported.clone(),
                |s| s.split(',').map(|id| id.trim().parse().expect("invalid KZG CUDA device")).collect(),
            );
            assert!(!ids.is_empty());
            for (i, &id) in ids.iter().enumerate() {
                assert!(supported.contains(&id) && !ids[..i].contains(&id), "invalid or duplicate KZG CUDA ordinal {id}");
            }
            let chunk_points = std::env::var("MULTI_STARK_KZG_MSM_CHUNK_POINTS")
                .map_or(1 << 24, |s| s.parse().expect("invalid MSM chunk size"));
            assert!((1..=1 << 28).contains(&chunk_points), "MSM chunk size must be in 1..=2^28");
            let limit = std::env::var("MULTI_STARK_KZG_CUDA_RESIDENT_GIB")
                .ok().map(|value| value.parse::<usize>().expect("invalid resident GiB limit")
                    .checked_mul(1 << 30).expect("resident limit overflow"));
            let resident_limits: Vec<_> = ids.iter().map(|&id| {
                let (mut free, mut total) = (0, 0);
                check(unsafe { multi_stark_kzg_memory(id, &mut free, &mut total) }, "memory query");
                limit.unwrap_or(total / 2).min(total / 2).min(free.saturating_sub(total / 4))
            }).collect();
            tracing::info!(?ids, chunk_points, ?resident_limits, "KZG sppark CUDA backend");
            Devices {
                busy: Mutex::new(vec![false; ids.len()]),
                resident_bytes: Mutex::new(vec![0; ids.len()]), resident_limits,
                ids, available: Condvar::new(), chunk_points,
            }
        })
    }

    fn acquire(&self) -> Device<'_> {
        let mut busy = self.busy.lock().unwrap();
        loop {
            if let Some(index) = busy.iter().position(|&v| !v) {
                busy[index] = true;
                return Device {
                    devices: self,
                    index,
                };
            }
            busy = self.available.wait(busy).unwrap();
        }
    }
    fn acquire_at(&self, index: usize) -> Device<'_> {
        let mut busy = self.busy.lock().unwrap();
        while busy[index] {
            busy = self.available.wait(busy).unwrap();
        }
        busy[index] = true;
        Device {
            devices: self,
            index,
        }
    }

    fn acquire_group(&self, indices: &[usize]) -> Vec<Device<'_>> {
        assert!(
            indices
                .iter()
                .enumerate()
                .all(|(i, &index)| { index < self.ids.len() && !indices[..i].contains(&index) })
        );
        let mut busy = self.busy.lock().unwrap();
        while indices.iter().any(|&index| busy[index]) {
            busy = self.available.wait(busy).unwrap();
        }
        for &index in indices {
            busy[index] = true;
        }
        indices
            .iter()
            .map(|&index| Device {
                devices: self,
                index,
            })
            .collect()
    }

    fn reserve(&self, bytes: usize) -> Option<Reservation> {
        let mut retained = self.resident_bytes.lock().unwrap();
        let index = (0..self.ids.len())
            .filter(|&i| bytes <= self.resident_limits[i].saturating_sub(retained[i]))
            .min_by_key(|&i| retained[i])?;
        retained[index] += bytes;
        Some(Reservation { index, bytes })
    }
}

struct Device<'a> {
    devices: &'a Devices,
    index: usize,
}

impl Device<'_> {
    fn prepare_budget(&self, required: usize) -> bool {
        srs_cache::prepare_budget(self, required)
    }

    fn id(&self) -> i32 {
        self.devices.ids[self.index]
    }

    fn budget(&self) -> usize {
        let (mut free, mut total) = (0, 0);
        // The lease excludes other KZG operations on these sppark streams.
        check(
            unsafe { multi_stark_kzg_memory(self.id(), &mut free, &mut total) },
            "memory query",
        );
        free.saturating_sub(total / 4)
    }

    fn idle_memory(&self, trim: bool) -> super::pcs::KzgDeviceMemorySnapshot {
        let mut memory = NativeIdleMemory::default();
        check(
            unsafe { multi_stark_kzg_idle_memory(self.id(), trim, &mut memory) },
            "idle memory release",
        );
        let (srs_point_bytes, msm_workspace_bytes) = srs_cache::resident_bytes(self.index);
        super::pcs::KzgDeviceMemorySnapshot {
            driver_free_bytes: memory.driver_free_bytes,
            total_bytes: memory.total_bytes,
            current_pool_reserved_bytes: memory.current_pool_reserved_bytes,
            current_pool_used_bytes: memory.current_pool_used_bytes,
            default_pool_reserved_bytes: memory.default_pool_reserved_bytes,
            default_pool_used_bytes: memory.default_pool_used_bytes,
            current_pool_is_default: memory.current_pool_is_default != 0,
            resident_coefficient_bytes: self.devices.resident_bytes.lock().unwrap()[self.index],
            srs_point_bytes,
            msm_workspace_bytes,
        }
    }
}

pub(super) fn release_idle_device_memory() -> super::pcs::KzgIdleMemoryRelease {
    let mut report = super::pcs::KzgIdleMemoryRelease {
        cuda_enabled: true,
        ..Default::default()
    };
    let Some(devices) = DEVICES.get() else {
        return report;
    };
    let indices: Vec<_> = (0..devices.ids.len()).collect();
    let leases = devices.acquire_group(&indices);
    report.initialized = true;
    for device in &leases {
        let before = device.idle_memory(false);
        let (released_srs_point_bytes, released_msm_workspace_bytes) =
            srs_cache::release_all(device);
        let after = device.idle_memory(true);
        report.devices.push(super::pcs::KzgDeviceMemoryRelease {
            device: device.id(),
            before,
            after,
            released_srs_point_bytes,
            released_msm_workspace_bytes,
        });
    }
    report.quiesced = true;
    report
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        self.devices.busy.lock().unwrap()[self.index] = false;
        self.devices.available.notify_all();
    }
}

struct Reservation {
    index: usize,
    bytes: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        Devices::get().resident_bytes.lock().unwrap()[self.index] -= self.bytes;
    }
}

/// Immutable device coefficients; every native use holds this device's lease.
/// Host coefficients remain the recovery source when the residency budget is full.
pub(super) struct ResidentPolynomial {
    reservation: Reservation,
    context: usize,
}

impl Drop for ResidentPolynomial {
    fn drop(&mut self) {
        let device = Devices::get().acquire_at(self.reservation.index);
        // Native destruction waits for all streams before the budget is returned.
        let code =
            unsafe { multi_stark_kzg_polynomial_destroy(device.id(), self.context as *mut c_void) };
        if code != 0 {
            tracing::error!(code, "KZG resident polynomial destruction failed");
        }
    }
}

pub(super) fn retain(coefficients: &[Fr]) -> Option<Arc<ResidentPolynomial>> {
    if coefficients.len() < 1 << 16 {
        return None;
    }
    let devices = Devices::get();
    let reservation = devices.reserve(coefficients.len().checked_mul(32).unwrap())?;
    let device = devices.acquire_at(reservation.index);
    if !device.prepare_budget(reservation.bytes + (1 << 30)) {
        return None;
    }
    let mut context = std::ptr::null_mut();
    // The native handle owns immutable storage and synchronizes its initial upload.
    check(
        unsafe {
            multi_stark_kzg_polynomial_upload(
                device.id(),
                coefficients.as_ptr().cast(),
                coefficients.len(),
                &mut context,
            )
        },
        "retain polynomial",
    );
    Some(Arc::new(ResidentPolynomial {
        reservation,
        context: context as usize,
    }))
}

/// Assign resident inputs to their owning device, balancing host inputs across
/// the remaining queues. Each queue has at most two full-size working buffers.
pub(super) fn fft_columns(
    inputs: &[(&[Fr], Option<&ResidentPolynomial>)],
    lg: usize,
    inverse: bool,
    shift: Fr,
) -> Vec<Vec<Fr>> {
    assert!(lg <= 31 && !shift.is_zero());
    let count = 1usize << lg;
    assert!(inputs.iter().all(|(input, _)| input.len() <= count));
    if inputs.is_empty() {
        return vec![];
    }
    let devices = Devices::get();
    let mut groups = vec![Vec::new(); devices.ids.len()];
    for (column, (_, resident)) in inputs.iter().enumerate() {
        if let Some(resident) = resident {
            groups[resident.reservation.index].push(column);
        }
    }
    for (column, (_, resident)) in inputs.iter().enumerate() {
        if resident.is_none() {
            let index = (0..groups.len()).min_by_key(|&i| groups[i].len()).unwrap();
            groups[index].push(column);
        }
    }
    let shift = if inverse {
        shift.inverse().unwrap()
    } else {
        shift
    };
    let groups: Vec<_> = groups
        .into_par_iter()
        .enumerate()
        .filter(|(_, g)| !g.is_empty())
        .map(|(index, group)| {
            let started = std::time::Instant::now();
            let device = devices.acquire_at(index);
            let wait_seconds = started.elapsed().as_secs_f64();
            device.prepare_budget(count * 32 * group.len().min(2) + (1 << 30));
            let workspace = device.budget().saturating_sub(1 << 30);
            let slots = (workspace / (count * 32)).min(2).min(group.len());
            assert!(slots > 0, "insufficient free VRAM for KZG FFT queue");
            let mut outputs: Vec<_> = group
                .iter()
                .map(|_| Vec::<Fr>::with_capacity(count))
                .collect();
            let jobs: Vec<_> = group
                .iter()
                .zip(&mut outputs)
                .map(|(&column, output)| {
                    let (input, resident) = inputs[column];
                    FftJob {
                        input: input.as_ptr().cast(),
                        resident_input: resident
                            .map_or(std::ptr::null(), |r| r.context as *const c_void),
                        output: output.spare_capacity_mut().as_mut_ptr().cast(),
                        input_count: input.len(),
                        resident_output: std::ptr::null_mut(),
                    }
                })
                .collect();
            let started = std::time::Instant::now();
            // Native queues drain both streams on success and failure. Only a
            // successful call initializes every field in each output allocation.
            check(
                unsafe {
                    multi_stark_kzg_fft_batch(
                        device.id(),
                        jobs.as_ptr(),
                        jobs.len(),
                        lg as u32,
                        inverse,
                        &shift.0.0,
                        slots,
                    )
                },
                "FFT queue",
            );
            for output in &mut outputs {
                unsafe { output.set_len(count) };
            }
            tracing::debug!(
                device = device.id(),
                columns = jobs.len(),
                elements = count,
                slots,
                wait_seconds,
                cuda_seconds = started.elapsed().as_secs_f64(),
                "KZG FFT queue"
            );
            group.into_iter().zip(outputs).collect::<Vec<_>>()
        })
        .collect();
    let mut outputs = vec![Vec::new(); inputs.len()];
    for (index, output) in groups.into_iter().flatten() {
        outputs[index] = output;
    }
    outputs
}

/// Interpolate owned columns in place on the host while retaining admitted
/// device outputs. A native queue owns unfinished outputs until it drains.
pub(super) fn interpolate_columns(
    columns: Vec<Vec<Fr>>,
    lg: usize,
    shift: Fr,
) -> (Vec<Vec<Fr>>, Vec<Option<Arc<ResidentPolynomial>>>) {
    assert!(lg <= 31 && !shift.is_zero());
    let count = 1usize << lg;
    let devices = Devices::get();
    let mut groups: Vec<Vec<_>> = (0..devices.ids.len()).map(|_| Vec::new()).collect();
    let width = columns.len();
    for (column, values) in columns.into_iter().enumerate() {
        assert_eq!(values.len(), count);
        let reservation = if count >= 1 << 16 {
            devices.reserve(count * 32)
        } else {
            None
        };
        let index = reservation.as_ref().map_or_else(
            || (0..groups.len()).min_by_key(|&i| groups[i].len()).unwrap(),
            |r| r.index,
        );
        groups[index].push((column, values, reservation));
    }
    let shift = shift.inverse().unwrap();
    let parts: Vec<_> = groups
        .into_par_iter()
        .enumerate()
        .filter(|(_, group)| !group.is_empty())
        .map(|(index, mut group)| {
            let device = devices.acquire_at(index);
            let bytes = count * 32;
            // All retained outputs survive the queue. Count them in addition
            // to the two transient slots, even though their lifetimes overlap.
            let retained = group.iter().filter(|(_, _, r)| r.is_some()).count() * bytes;
            device.prepare_budget(retained + bytes * group.len().min(2) + (1 << 30));
            let budget = device.budget().saturating_sub(1 << 30);
            if retained + bytes > budget {
                for (_, _, reservation) in &mut group {
                    *reservation = None;
                }
            }
            let retained = group.iter().filter(|(_, _, r)| r.is_some()).count() * bytes;
            let slots = ((budget - retained) / bytes).min(2).min(group.len());
            assert!(
                slots > 0,
                "insufficient free VRAM for KZG interpolation queue"
            );
            let mut handles = vec![std::ptr::null_mut(); group.len()];
            let jobs: Vec<_> = group
                .iter_mut()
                .zip(&mut handles)
                .map(|((_, values, reservation), handle)| FftJob {
                    input: values.as_ptr().cast(),
                    resident_input: std::ptr::null(),
                    output: values.as_mut_ptr().cast(),
                    input_count: count,
                    resident_output: if reservation.is_some() {
                        handle
                    } else {
                        std::ptr::null_mut()
                    },
                })
                .collect();
            let started = std::time::Instant::now();
            let code = unsafe {
                multi_stark_kzg_fft_batch(
                    device.id(),
                    jobs.as_ptr(),
                    jobs.len(),
                    lg as u32,
                    true,
                    &shift.0.0,
                    slots,
                )
            };
            // A later job can fail after an earlier retained output completed.
            // Reclaim those handles while the lease still excludes native users.
            if code != 0 {
                for handle in handles.into_iter().filter(|p| !p.is_null()) {
                    unsafe { multi_stark_kzg_polynomial_destroy(device.id(), handle) };
                }
                check(code, "interpolation queue");
                unreachable!();
            }
            tracing::debug!(
                device = device.id(),
                columns = group.len(),
                elements = count,
                slots,
                resident_bytes = retained,
                cuda_seconds = started.elapsed().as_secs_f64(),
                "KZG interpolation queue"
            );
            drop(device);
            group
                .into_iter()
                .zip(handles)
                .map(|((column, values, reservation), context)| {
                    let resident = reservation.map(|reservation| {
                        assert!(!context.is_null());
                        Arc::new(ResidentPolynomial {
                            reservation,
                            context: context as usize,
                        })
                    });
                    (column, values, resident)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let mut outputs = vec![Vec::new(); width];
    let mut residents = vec![None; width];
    for (column, values, resident) in parts.into_iter().flatten() {
        outputs[column] = values;
        residents[column] = resident;
    }
    (outputs, residents)
}

pub(super) fn msm(points: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    msm_chunked(points, scalars, Devices::get().chunk_points)
}

fn msm_chunked(points: &[G1Affine], scalars: &[Fr], chunk_limit: usize) -> G1Projective {
    msm_columns_chunked(points, &[scalars], chunk_limit).remove(0)
}

pub(super) fn msm_columns(points: &[G1Affine], columns: &[&[Fr]]) -> Vec<G1Projective> {
    msm_columns_chunked(points, columns, Devices::get().chunk_points)
}

pub(super) fn msm_columns_resident(
    points: &[G1Affine],
    columns: &[(&[Fr], Option<&ResidentPolynomial>)],
) -> Vec<G1Projective> {
    assert!(columns.iter().all(|(c, _)| c.len() == points.len()));
    if points.is_empty() {
        return vec![G1Projective::zero(); columns.len()];
    }
    let devices = Devices::get();
    let mut groups = vec![Vec::new(); devices.ids.len()];
    let mut host = Vec::new();
    for (i, (_, resident)) in columns.iter().enumerate() {
        if let Some(resident) = resident {
            groups[resident.reservation.index].push(i);
        } else {
            host.push(i);
        }
    }
    let parts: Vec<_> = groups
        .into_par_iter()
        .enumerate()
        .filter(|(_, g)| !g.is_empty())
        .map(|(index, group)| {
            let device = devices.acquire_at(index);
            device.prepare_budget(devices.chunk_points.min(points.len()) * 384 + (1 << 30));
            let chunk = devices
                .chunk_points
                .min(device.budget().saturating_sub(1 << 30) / 384);
            assert!(chunk > 0, "insufficient free VRAM for KZG MSM");
            let mut sums = vec![G1Projective::zero(); group.len()];
            for (part, points) in points.chunks(chunk).enumerate() {
                let resident_points = upload_msm_points(&device, points);
                let start = part * chunk;
                for (sum, &column) in sums.iter_mut().zip(&group) {
                    let (coefficients, resident) = columns[column];
                    let input = &coefficients[start..start + points.len()];
                    let normalization = msm_normalization(input);
                    let mut out = [[0; 6]; 3];
                    // Resident coefficients are read-only. Normalization uses a
                    // bounded temporary on the same device as this SRS chunk.
                    check(
                        unsafe {
                            multi_stark_kzg_msm_invoke_resident(
                                device.id(),
                                resident_points.context,
                                &mut out,
                                resident.unwrap().context as *const c_void,
                                start,
                                points.len(),
                                normalization
                                    .as_ref()
                                    .map_or(std::ptr::null(), |(_, inverse)| &inverse.0.0),
                            )
                        },
                        "resident scalar MSM",
                    );
                    let point = projective(out);
                    *sum += normalization.map_or(point, |(scale, _)| point * scale);
                }
            }
            group.into_iter().zip(sums).collect::<Vec<_>>()
        })
        .collect();
    let mut sums = vec![G1Projective::zero(); columns.len()];
    for (i, point) in parts.into_iter().flatten() {
        sums[i] = point;
    }
    if !host.is_empty() {
        let inputs: Vec<_> = host.iter().map(|&i| columns[i].0).collect();
        for (i, point) in host.into_iter().zip(msm_columns(points, &inputs)) {
            sums[i] = point;
        }
    }
    sums
}

fn projective(limbs: [[u64; 6]; 3]) -> G1Projective {
    G1Projective::new_unchecked(
        Fq::new_unchecked(BigInt(limbs[0])),
        Fq::new_unchecked(BigInt(limbs[1])),
        Fq::new_unchecked(BigInt(limbs[2])),
    )
}

fn upload_msm_points<'lease, 'devices>(
    device: &'lease Device<'devices>,
    points: &[G1Affine],
) -> ResidentMsm<'lease, 'devices> {
    let mut context = std::ptr::null_mut();
    // The layout-checked arkworks storage stays borrowed through the upload.
    // Native conversion preserves infinity without a polynomial-sized host copy.
    check(
        unsafe {
            multi_stark_kzg_msm_create(
                device.id(),
                &mut context,
                points.as_ptr().cast(),
                points.len(),
            )
        },
        "MSM point upload",
    );
    assert!(!context.is_null());
    ResidentMsm { device, context }
}

struct ResidentMsm<'lease, 'devices> {
    device: &'lease Device<'devices>,
    context: *mut c_void,
}

impl Drop for ResidentMsm<'_, '_> {
    fn drop(&mut self) {
        // The borrowed lease outlives the native points and bucket workspace.
        unsafe { multi_stark_kzg_msm_destroy(self.device.id(), self.context) };
    }
}

fn msm_normalization(scalars: &[Fr]) -> Option<(Fr, Fr)> {
    if scalars.len() < 4096 {
        return None;
    }
    sample_normalization(
        (0usize..128).map(|i| scalars[i.wrapping_mul(0x9e37_79b9_7f4a_7c15) % scalars.len()]),
    )
}

fn sample_normalization(samples: impl Iterator<Item = Fr>) -> Option<(Fr, Fr)> {
    let mut counts = HashMap::new();
    for value in samples {
        if !value.is_zero() {
            *counts.entry(value.min(-value)).or_insert(0usize) += 1;
        }
    }
    let (scale, count) = counts
        .into_iter()
        .max_by_key(|&(scale, count)| (count, scale))?;
    if count < 16 || scale == Fr::ONE || scale == -Fr::ONE {
        return None;
    }
    // Equal nonunit scalars overload one serial Pippenger bucket per window.
    // Scaling them to +/-1 uses sppark's parallel addition path; multiplying
    // the resulting point by `scale` restores the original MSM exactly.
    Some((scale, scale.inverse().unwrap()))
}

fn msm_columns_chunked(
    points: &[G1Affine],
    columns: &[&[Fr]],
    chunk_limit: usize,
) -> Vec<G1Projective> {
    assert!(columns.iter().all(|column| column.len() == points.len()));
    if points.is_empty() || columns.is_empty() {
        return vec![G1Projective::zero(); columns.len()];
    }
    assert!(chunk_limit > 0);
    let devices = Devices::get();
    let chunk = points
        .len()
        .div_ceil(devices.ids.len())
        .min(chunk_limit)
        .max(1);
    points
        .par_chunks(chunk)
        .enumerate()
        .map(|(chunk_index, points)| {
            let device = devices.acquire();
            device.prepare_budget(points.len() * 384 + (1 << 30));
            // The largest window uses < 1 GiB of buckets. Resident affine points,
            // scalars, signed digits and sorting scratch fit in 384 B/point.
            let admitted = device.budget().saturating_sub(1 << 30) / 384;
            assert!(admitted > 0, "insufficient free VRAM for KZG MSM");
            let mut sums = vec![G1Projective::zero(); columns.len()];
            for (part, points) in points.chunks(admitted).enumerate() {
                let start = chunk_index * chunk + part * admitted;
                let count = points.len();
                let resident = upload_msm_points(&device, points);
                for (sum, column) in sums.iter_mut().zip(columns) {
                    let input = &column[start..start + count];
                    let normalization = msm_normalization(input);
                    let normalized = normalization.map(|(scale, inverse)| {
                        let negative = -scale;
                        super::buffer::generate(input.len(), |i| {
                            let value = input[i];
                            if value == scale {
                                Fr::ONE
                            } else if value == negative {
                                -Fr::ONE
                            } else {
                                value * inverse
                            }
                        })
                    });
                    let scalars = normalized.as_deref().unwrap_or(input);
                    let mut out = [[0; 6]; 3];
                    // The resident point chunk is immutable across all column invocations.
                    check(
                        unsafe {
                            multi_stark_kzg_msm_invoke(
                                device.id(),
                                resident.context,
                                &mut out,
                                scalars.as_ptr().cast(),
                                count,
                            )
                        },
                        "MSM",
                    );
                    let point = projective(out);
                    *sum += match normalization {
                        Some((scale, _)) => point * scale,
                        None => point,
                    };
                }
            }
            sums
        })
        .reduce(
            || vec![G1Projective::zero(); columns.len()],
            |mut sums, parts| {
                for (sum, part) in sums.iter_mut().zip(parts) {
                    *sum += part;
                }
                sums
            },
        )
}

pub(super) fn fft(values: &mut [Fr], inverse: bool, shift: Fr) {
    // sppark's 32-bit domain stride cannot represent 2^32.
    assert!(values.len().is_power_of_two() && values.len().ilog2() <= 31);
    assert!(!shift.is_zero());
    if values.len() == 1 {
        return;
    }
    let started = std::time::Instant::now();
    let device = Devices::get().acquire();
    let wait_seconds = started.elapsed().as_secs_f64();
    assert!(
        device.prepare_budget(values.len().checked_mul(32).unwrap() + (1 << 30)),
        "insufficient free VRAM for KZG FFT"
    );
    let shift = if inverse {
        shift.inverse().unwrap()
    } else {
        shift
    };
    let started = std::time::Instant::now();
    // sppark normalizes inverse NTTs. Coset powers act on coefficients, before
    // the forward transform or after the inverse transform in natural order.
    check(
        // The asserted layout covers every byte of each field. The exclusive
        // slice remains borrowed until all native transfers have completed.
        unsafe {
            multi_stark_kzg_fft(
                device.id(),
                values.as_mut_ptr().cast(),
                values.len().ilog2(),
                inverse,
                &shift.0.0,
            )
        },
        "FFT",
    );
    let cuda_seconds = started.elapsed().as_secs_f64();
    tracing::debug!(
        device = device.id(),
        elements = values.len(),
        wait_seconds,
        cuda_seconds,
        "KZG FFT transfer profile"
    );
}

#[cfg(test)]
pub(super) fn fft_from(coefficients: &[Fr], lg: usize, shift: Fr) -> Vec<Fr> {
    assert!(lg <= 31 && !shift.is_zero());
    let count = 1usize << lg;
    assert!(coefficients.len() <= count);
    let started = std::time::Instant::now();
    let device = Devices::get().acquire();
    let wait_seconds = started.elapsed().as_secs_f64();
    assert!(
        device.prepare_budget(count * 32 + (1 << 30)),
        "insufficient free VRAM for KZG FFT"
    );
    let mut values = Vec::<Fr>::with_capacity(count);
    let started = std::time::Instant::now();
    // Input limbs have the checked layout above. CUDA initializes the entire
    // output before its length is set; failure leaves an empty, droppable Vec.
    check(
        unsafe {
            multi_stark_kzg_fft_from(
                device.id(),
                coefficients.as_ptr().cast(),
                coefficients.len(),
                values.spare_capacity_mut().as_mut_ptr().cast(),
                lg as u32,
                false,
                &shift.0.0,
            )
        },
        "out-of-place FFT",
    );
    unsafe { values.set_len(count) };
    tracing::debug!(
        device = device.id(),
        elements = count,
        input_elements = coefficients.len(),
        wait_seconds,
        cuda_seconds = started.elapsed().as_secs_f64(),
        "KZG FFT transfer profile"
    );
    values
}

pub(super) fn evaluate_many(coefficients: &[Fr], points: &[Fr]) -> Vec<Fr> {
    evaluate_resident(coefficients, None, points)
}

pub(super) fn evaluate_resident(
    coefficients: &[Fr],
    resident: Option<&ResidentPolynomial>,
    points: &[Fr],
) -> Vec<Fr> {
    if points.is_empty() {
        return vec![];
    }
    if coefficients.len() < 2 {
        return vec![coefficients.first().copied().unwrap_or(Fr::ZERO); points.len()];
    }
    let devices = Devices::get();
    let device = resident.map_or_else(
        || devices.acquire(),
        |r| devices.acquire_at(r.reservation.index),
    );
    let bytes = points
        .len()
        .checked_mul(64)
        .and_then(|bytes| {
            bytes.checked_add(if resident.is_some() {
                0
            } else {
                coefficients.len().checked_mul(32)?
            })
        })
        .and_then(|bytes| bytes.checked_add(1 << 30))
        .expect("KZG evaluation size overflow");
    assert!(
        device.prepare_budget(bytes),
        "insufficient free VRAM for KZG evaluation"
    );
    let mut results = Vec::<Fr>::with_capacity(points.len());
    // Coefficients are immutable for the entire call. The resident handle is
    // tied to its owning device and results retain point order, including repeats.
    check(
        unsafe {
            multi_stark_kzg_evaluate_resident(
                device.id(),
                coefficients.as_ptr().cast(),
                resident.map_or(std::ptr::null(), |r| r.context as *const c_void),
                coefficients.len(),
                points.as_ptr().cast(),
                points.len(),
                results.spare_capacity_mut().as_mut_ptr().cast(),
            )
        },
        "polynomial evaluation",
    );
    unsafe { results.set_len(points.len()) };
    results
}

pub(super) fn divide(coefficients: &[Fr], z: Fr) -> Vec<Fr> {
    if coefficients.len() < 2 {
        return vec![];
    }
    let device = Devices::get().acquire();
    assert!(
        device.prepare_budget(coefficients.len().checked_mul(32).unwrap() + (1 << 30)),
        "insufficient free VRAM for KZG division"
    );
    // Parallel first-touch keeps page faults out of the pinned-ring download.
    let mut values = super::buffer::generate(coefficients.len() - 1, |_| Fr::ZERO);
    // The call synchronizes its copies and kernels before releasing either buffer.
    check(
        unsafe {
            multi_stark_kzg_divide(
                device.id(),
                coefficients.as_ptr().cast(),
                values.as_mut_ptr().cast(),
                coefficients.len(),
                &z.0.0,
            )
        },
        "polynomial division",
    );
    values
}

pub(super) fn single_device_opening(
    points: &[G1Affine],
    coefficients: &[Fr],
    z: Fr,
) -> Option<G1Projective> {
    // A quotient on one card loses the multi-device MSM partitioning benefit.
    if Devices::get().ids.len() != 1 {
        return None;
    }
    divide_and_msm(points, coefficients, z)
}

/// Divide and commit without a polynomial-sized download or scalar re-upload.
/// The quotient and its MSM scratch must fit together under one device lease.
fn divide_and_msm(points: &[G1Affine], coefficients: &[Fr], z: Fr) -> Option<G1Projective> {
    divide_and_msm_chunked(points, coefficients, z, Devices::get().chunk_points)
}

fn divide_and_msm_chunked(
    points: &[G1Affine],
    coefficients: &[Fr],
    z: Fr,
    chunk_limit: usize,
) -> Option<G1Projective> {
    assert_eq!(points.len(), coefficients.len().saturating_sub(1));
    if points.is_empty() {
        return Some(G1Projective::zero());
    }
    let devices = Devices::get();
    let device = devices.acquire();
    device.prepare_budget(
        coefficients
            .len()
            .checked_mul(32)?
            .checked_add(1 << 30)?
            .checked_add(chunk_limit.min(points.len()).checked_mul(384)?)?,
    );
    let chunk = opening_chunk(coefficients.len(), chunk_limit, device.budget())?;
    let mut context = std::ptr::null_mut();
    check(
        unsafe {
            multi_stark_kzg_divide_resident(
                device.id(),
                coefficients.as_ptr().cast(),
                coefficients.len(),
                &z.0.0,
                &mut context,
            )
        },
        "resident polynomial division",
    );
    let quotient = LeasedPolynomial {
        device: &device,
        context,
    };
    Some(commit_leased_quotient(&quotient, points, chunk))
}

fn opening_chunk(count: usize, chunk_limit: usize, budget: usize) -> Option<usize> {
    let fixed = count.checked_mul(32)?.checked_add(1 << 30)?;
    let chunk = chunk_limit.min(budget.checked_sub(fixed)? / 384);
    (chunk > 0).then_some(chunk)
}

fn commit_leased_quotient(
    quotient: &LeasedPolynomial<'_, '_>,
    points: &[G1Affine],
    chunk: usize,
) -> G1Projective {
    let device = quotient.device;
    let mut result = G1Projective::zero();
    for (part, points) in points.chunks(chunk).enumerate() {
        let offset = part * chunk;
        let mut samples = [Fr::ZERO; 128];
        let normalization = if points.len() >= 4096 {
            check(
                unsafe {
                    multi_stark_kzg_polynomial_sample(
                        device.id(),
                        quotient.context,
                        offset,
                        points.len(),
                        samples.as_mut_ptr().cast(),
                    )
                },
                "resident scalar sample",
            );
            sample_normalization(samples.into_iter())
        } else {
            None
        };
        let resident_points = upload_msm_points(device, points);
        let mut out = [[0; 6]; 3];
        check(
            unsafe {
                multi_stark_kzg_msm_invoke_resident(
                    device.id(),
                    resident_points.context,
                    &mut out,
                    quotient.context,
                    offset,
                    points.len(),
                    normalization
                        .as_ref()
                        .map_or(std::ptr::null(), |(_, inverse)| &inverse.0.0),
                )
            },
            "opening witness MSM",
        );
        let point = projective(out);
        result += normalization.map_or(point, |(scale, _)| point * scale);
    }
    result
}

struct LeasedPolynomial<'lease, 'devices> {
    device: &'lease Device<'devices>,
    context: *mut c_void,
}

impl Drop for LeasedPolynomial<'_, '_> {
    fn drop(&mut self) {
        // The existing lease covers destruction; reacquiring it would deadlock.
        let code = unsafe { multi_stark_kzg_polynomial_destroy(self.device.id(), self.context) };
        if code != 0 {
            tracing::error!(code, "KZG temporary polynomial destruction failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
    use ark_ff::{AdditiveGroup, PrimeField};
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

    fn scalars(n: usize) -> Vec<Fr> {
        (0..n)
            .map(|i| {
                let mut bytes = [0; 64];
                blake3::Hasher::new()
                    .update(&(i as u64).to_le_bytes())
                    .finalize_xof()
                    .fill(&mut bytes);
                Fr::from_le_bytes_mod_order(&bytes)
            })
            .collect()
    }

    #[test]
    fn cuda_msm_matches_arkworks_with_chunks_and_exceptional_points() {
        let mut scalars = scalars(4099);
        let mut points: Vec<_> = scalars
            .iter()
            .map(|&s| (G1Affine::generator() * s).into_affine())
            .collect();
        points[0] = G1Affine::zero();
        points[1] = G1Affine::generator();
        points[2] = -points[1];
        points[4].infinity = true;
        scalars[1] = Fr::ONE;
        scalars[2] = Fr::ONE;
        scalars[3] = Fr::ZERO;
        scalars[4] = -Fr::ONE;
        for n in [0, 1, 31, 32, 33, 257, 4099] {
            assert_eq!(
                msm_chunked(&points[..n], &scalars[..n], 257),
                G1Projective::msm(&points[..n], &scalars[..n]).unwrap()
            );
        }
        assert!(msm_chunked(&points, &vec![Fr::ZERO; points.len()], 1024).is_zero());
        assert!(msm_chunked(&points[1..3], &scalars[1..3], 1).is_zero());
    }

    #[test]
    fn cuda_resident_msm_columns_match_arkworks() {
        let input = scalars(4099);
        let mut points: Vec<_> = input
            .iter()
            .map(|&s| (G1Affine::generator() * s).into_affine())
            .collect();
        points[0] = G1Affine::zero();
        let negative: Vec<_> = input.iter().map(|&s| -s).collect();
        let zeros = vec![Fr::ZERO; input.len()];
        let ones = vec![Fr::ONE; input.len()];
        let columns = [
            input.as_slice(),
            zeros.as_slice(),
            negative.as_slice(),
            ones.as_slice(),
        ];
        let actual = msm_columns_chunked(&points, &columns, 257);
        for (actual, coefficients) in actual.iter().zip(columns) {
            assert_eq!(*actual, G1Projective::msm(&points, coefficients).unwrap());
        }
        assert!((actual[0] + actual[2]).is_zero());
        points[1] = -points[1];
        assert_eq!(
            msm_chunked(&points, &input, 257),
            G1Projective::msm(&points, &input).unwrap()
        );
    }

    #[test]
    fn cuda_msm_normalizes_structured_columns() {
        let n = 65_537;
        let seeds = scalars(17);
        let mut basis: Vec<_> = seeds
            .iter()
            .map(|&s| (G1Affine::generator() * s).into_affine())
            .collect();
        basis[0] = G1Affine::zero();
        let points: Vec<_> = (0..n).map(|i| basis[i % basis.len()]).collect();
        let scale = Fr::from(19).inverse().unwrap();
        let structured: Vec<_> = (0..n)
            .map(|i| {
                if i % 11 == 0 {
                    Fr::ZERO
                } else if i % 257 == 0 {
                    seeds[i % seeds.len()]
                } else if i % 7 == 0 {
                    -scale
                } else {
                    scale
                }
            })
            .collect();
        let scaled: Vec<_> = structured.iter().map(|&s| s * Fr::from(7)).collect();
        let random = scalars(n);
        assert!(msm_normalization(&structured).is_some());
        assert!(msm_normalization(&random).is_none());
        let columns = [structured.as_slice(), scaled.as_slice(), random.as_slice()];
        let actual = msm_columns_chunked(&points, &columns, 1 << 15);
        for (actual, coefficients) in actual.into_iter().zip(columns) {
            assert_eq!(actual, G1Projective::msm(&points, coefficients).unwrap());
        }
    }

    #[test]
    fn cuda_fft_rejects_overflowing_domain_stride() {
        let mut value = Fr::ONE.0.0;
        let original = value;
        // The invalid size must be rejected before any transfer or allocation.
        let status = unsafe { multi_stark_kzg_fft(0, &mut value, 32, false, &original) };
        assert_ne!(status, 0);
        assert_eq!(value, original);
    }

    #[test]
    fn cuda_fft_matches_arkworks_on_subgroups_and_arbitrary_cosets() {
        for lg in [0, 1, 5, 10, 11, 16] {
            for shift in [Fr::ONE, Fr::from(7), Fr::from(12345)] {
                let domain = Radix2EvaluationDomain::<Fr>::new(1 << lg)
                    .unwrap()
                    .get_coset(shift)
                    .unwrap();
                let input = scalars(1 << lg);
                let mut actual = input.clone();
                fft(&mut actual, false, shift);
                assert_eq!(actual, domain.fft(&input), "forward lg={lg}");
                fft(&mut actual, true, shift);
                assert_eq!(actual, input, "inverse lg={lg}");
                let mut actual = input.clone();
                fft(&mut actual, true, shift);
                assert_eq!(actual, domain.ifft(&input), "inverse parity lg={lg}");
            }
        }
    }

    #[test]
    fn cuda_fft_from_pads_without_mutating_coefficients() {
        for lg in [0, 1, 10, 16] {
            let size = 1 << lg;
            let input = scalars(size);
            let saved = input.clone();
            for count in [0, 1, size / 3, size] {
                for shift in [Fr::ONE, Fr::from(7)] {
                    let domain = Radix2EvaluationDomain::<Fr>::new(size)
                        .unwrap()
                        .get_coset(shift)
                        .unwrap();
                    assert_eq!(
                        fft_from(&input[..count], lg, shift),
                        domain.fft(&input[..count])
                    );
                    assert_eq!(input, saved);
                }
            }
        }
    }

    #[test]
    fn cuda_fft_queues_reuse_resident_inputs_and_preserve_order() {
        let lg = 17;
        let input = scalars(1 << lg);
        let resident = retain(&input).expect("test requires resident VRAM budget");
        let expected = Radix2EvaluationDomain::<Fr>::new(input.len())
            .unwrap()
            .get_coset(Fr::from(7))
            .unwrap()
            .fft(&input);
        // More jobs than lanes forces buffer reuse; mixed inputs exercise both
        // device affinity and host-only queues, with repeated immutable handles.
        let jobs: Vec<_> = (0..11)
            .map(|i| {
                (
                    input.as_slice(),
                    if i % 2 == 0 {
                        Some(resident.as_ref())
                    } else {
                        None
                    },
                )
            })
            .collect();
        for _ in 0..2 {
            let actual = fft_columns(&jobs, lg, false, Fr::from(7));
            assert_eq!(actual.len(), jobs.len());
            assert!(actual.iter().all(|column| *column == expected));
        }
        let points = [Fr::ZERO, Fr::from(7), Fr::ONE, Fr::from(7)];
        let evaluations = evaluate_resident(&input, Some(&resident), &points);
        for (z, actual) in points.into_iter().zip(evaluations) {
            assert_eq!(
                actual,
                input.iter().rev().fold(Fr::ZERO, |acc, c| acc * z + c)
            );
        }
        drop(resident);
        assert_eq!(fft_columns(&[], lg, false, Fr::ONE), Vec::<Vec<Fr>>::new());
    }

    #[test]
    fn cuda_interpolation_retains_outputs_for_msm_and_cosets() {
        let count = 1 << 16;
        let shift = Fr::from(7);
        let domain = Radix2EvaluationDomain::<Fr>::new(count)
            .unwrap()
            .get_coset(shift)
            .unwrap();
        let coefficients = [scalars(count), vec![Fr::from(19).inverse().unwrap(); count]];
        let inputs = coefficients.iter().map(|c| domain.fft(c)).collect();
        let (actual, resident) = interpolate_columns(inputs, 16, shift);
        assert_eq!(actual, coefficients);
        assert!(resident.iter().all(Option::is_some));
        let points: Vec<_> = (0..count)
            .map(|i| {
                if i % 257 == 0 {
                    G1Affine::zero()
                } else {
                    (G1Affine::generator() * Fr::from((i % 31 + 1) as u64)).into_affine()
                }
            })
            .collect();
        let inputs: Vec<_> = actual
            .iter()
            .zip(&resident)
            .map(|(c, r)| (c.as_slice(), r.as_deref()))
            .collect();
        let commitments = msm_columns_resident(&points, &inputs);
        for (column, commitment) in actual.iter().zip(commitments) {
            assert_eq!(commitment, G1Projective::msm(&points, column).unwrap());
        }
        let output = fft_columns(&inputs, 16, false, Fr::from(123));
        let coset = Radix2EvaluationDomain::<Fr>::new(count)
            .unwrap()
            .get_coset(Fr::from(123))
            .unwrap();
        for (column, output) in coefficients.iter().zip(output) {
            assert_eq!(output, coset.fft(column));
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_goldilocks_and_kzg_runtime_coexist() {
        use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
        use p3_field::PrimeCharacteristicRing;
        use p3_goldilocks::Goldilocks;
        use p3_matrix::{Matrix, dense::RowMajorMatrix};

        let matrix = RowMajorMatrix::new(
            (0..4096 * 3)
                .map(|i| Goldilocks::from_u64(i as u64))
                .collect(),
            3,
        );
        let expected = Radix2DitParallel::default()
            .dft_batch(matrix.clone())
            .to_row_major_matrix();
        let gpu = crate::cuda::CudaDft::default();
        for _ in 0..2 {
            assert_eq!(
                gpu.dft_batch(matrix.clone()).to_row_major_matrix(),
                expected
            );
            cuda_fft_matches_arkworks_on_subgroups_and_arbitrary_cosets();
            cuda_msm_matches_arkworks_with_chunks_and_exceptional_points();
        }
    }

    #[test]
    fn cuda_polynomial_evaluation_and_division_match_horner() {
        for n in [0, 1, 2, 31, 32, 33, 1023, 1024, 1025, 65537] {
            let coefficients = scalars(n);
            for z in [Fr::ZERO, Fr::ONE, -Fr::ONE, Fr::from(918273)] {
                let expected = coefficients.iter().rev().fold(Fr::ZERO, |v, c| v * z + c);
                assert_eq!(
                    evaluate_many(&coefficients, &[z]),
                    [expected],
                    "evaluation n={n}"
                );
                let quotient = divide(&coefficients, z);
                assert_eq!(quotient.len(), n.saturating_sub(1));
                if n >= 2 {
                    assert_eq!(coefficients[0] + z * quotient[0], expected);
                    for i in 1..n {
                        assert_eq!(
                            quotient[i - 1] - z * quotient.get(i).copied().unwrap_or(Fr::ZERO),
                            coefficients[i],
                            "division n={n}, i={i}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cuda_polynomial_evaluates_all_points_in_order() {
        let mut points = scalars(41);
        points[..5].copy_from_slice(&[Fr::ZERO, Fr::ONE, -Fr::ONE, Fr::ONE, Fr::ZERO]);
        for n in [0, 1, 2, 31, 1025, 65_537] {
            let coefficients = scalars(n);
            for count in [0, 1, 2, 3, 4, 7, 8, 41] {
                let points = &points[..count];
                let expected: Vec<_> = points
                    .iter()
                    .map(|z| coefficients.iter().rev().fold(Fr::ZERO, |v, c| v * z + c))
                    .collect();
                assert_eq!(
                    evaluate_many(&coefficients, points),
                    expected,
                    "n={n}, points={count}"
                );
            }
        }
    }

    fn division_reference(coefficients: &[Fr], z: Fr) -> Vec<Fr> {
        let mut quotient = vec![Fr::ZERO; coefficients.len().saturating_sub(1)];
        let mut carry = Fr::ZERO;
        for i in (1..coefficients.len()).rev() {
            carry = coefficients[i] + z * carry;
            quotient[i - 1] = carry;
        }
        quotient
    }

    fn opening_points(count: usize) -> Vec<G1Affine> {
        let mut point = G1Projective::zero();
        let basis: Vec<_> = (0..256)
            .map(|_| {
                let output = point;
                point += G1Affine::generator();
                output
            })
            .collect();
        let basis = G1Projective::normalize_batch(&basis);
        super::super::buffer::generate(count, |i| basis[i % basis.len()])
    }

    #[test]
    fn cuda_resident_division_msm_matches_arkworks() {
        for n in [0usize, 1, 2, 33, 4097, 65537] {
            let points = opening_points(n.saturating_sub(1));
            for (coefficients, z) in [
                (scalars(n), Fr::from(918273)),
                (vec![Fr::from(19).inverse().unwrap(); n], Fr::ZERO),
                (vec![Fr::ZERO; n], Fr::ONE),
            ] {
                let expected =
                    G1Projective::msm(&points, &division_reference(&coefficients, z)).unwrap();
                assert_eq!(
                    divide_and_msm_chunked(&points, &coefficients, z, 16385),
                    Some(expected),
                    "n={n}, z={z}"
                );
            }
        }
        assert!(divide_and_msm_chunked(&opening_points(1), &[Fr::ONE; 2], Fr::ZERO, 0).is_none());
        assert_eq!(
            divide_and_msm(&opening_points(1), &[Fr::ONE; 2], Fr::ZERO),
            Some(G1Projective::zero())
        );
    }

    #[test]
    fn opening_memory_admission_keeps_scratch_reserve() {
        let fixed = (1 << 30) + 32 * 1024;
        assert_eq!(opening_chunk(1024, 16, fixed - 1), None);
        assert_eq!(opening_chunk(1024, 16, fixed + 383), None);
        assert_eq!(opening_chunk(1024, 16, fixed + 384), Some(1));
        assert_eq!(opening_chunk(1024, 16, fixed + 384 * 32), Some(16));
        assert_eq!(opening_chunk(usize::MAX, 16, usize::MAX), None);
        assert_eq!(opening_chunk(1024, 0, usize::MAX), None);
    }

    #[test]
    #[ignore = "compares quotient download/re-upload with a resident quotient MSM"]
    fn cuda_opening_benchmark() {
        let lg: usize =
            std::env::var("MULTI_STARK_KZG_BENCH_LOG_N").map_or(24, |value| value.parse().unwrap());
        assert!((16..=28).contains(&lg));
        let factor = scalars(1)[0];
        let coefficients =
            super::super::buffer::generate(1 << lg, |i| Fr::from(i as u64 + 1).pow([5]) * factor);
        let points = opening_points(coefficients.len() - 1);
        let z = Fr::from(918273);
        let _ = divide_and_msm(&points[..1023], &coefficients[..1024], z).unwrap();
        for iteration in 0..3 {
            let measure = |resident| {
                let started = std::time::Instant::now();
                let commitment = if resident {
                    divide_and_msm(&points, &coefficients, z).unwrap()
                } else {
                    msm(&points, &divide(&coefficients, z))
                };
                (commitment, started.elapsed().as_secs_f64())
            };
            let (first, second) = (measure(iteration % 2 == 0), measure(iteration % 2 != 0));
            assert_eq!(first.0, second.0);
            let (resident, host) = if iteration % 2 == 0 {
                (first.1, second.1)
            } else {
                (second.1, first.1)
            };
            eprintln!(
                "opening lg={lg} iteration={iteration} host_seconds={host:.6} resident_seconds={resident:.6}"
            );
        }
    }

    #[test]
    #[ignore = "measures one polynomial division without generating a proof or SRS"]
    fn cuda_division_benchmark() {
        let lg: usize =
            std::env::var("MULTI_STARK_KZG_BENCH_LOG_N").map_or(24, |value| value.parse().unwrap());
        assert!((16..=28).contains(&lg));
        let factor = scalars(1)[0];
        let coefficients =
            super::super::buffer::generate(1 << lg, |i| Fr::from(i as u64 + 1).pow([5]) * factor);
        let z = Fr::from(918273);
        let _ = divide(&coefficients[..1024], z);
        for iteration in 0..3 {
            let started = std::time::Instant::now();
            let quotient = divide(&coefficients, z);
            let seconds = started.elapsed().as_secs_f64();
            assert_eq!(quotient.len(), coefficients.len() - 1);
            assert!(quotient.par_iter().enumerate().all(|(i, &q)| {
                q - z * quotient.get(i + 1).copied().unwrap_or(Fr::ZERO) == coefficients[i + 1]
            }));
            eprintln!("division lg={lg} iteration={iteration} seconds={seconds:.6}");
        }
    }

    #[test]
    #[ignore = "measures CPU/GPU crossover and repeated-column MSM throughput"]
    fn cuda_primitive_benchmark() {
        use std::{hint::black_box, time::Instant};
        fn median(mut f: impl FnMut()) -> f64 {
            let mut samples: Vec<_> = (0..7)
                .map(|_| {
                    let start = Instant::now();
                    f();
                    start.elapsed().as_secs_f64()
                })
                .collect();
            samples.sort_by(f64::total_cmp);
            samples[3]
        }
        fft(&mut scalars(1024), false, Fr::ONE);
        for lg in [10, 12, 14, 16, 18] {
            let input = scalars(1 << lg);
            let domain = Radix2EvaluationDomain::<Fr>::new(input.len()).unwrap();
            let z = Fr::from(918273);
            let cpu_fft = median(|| {
                black_box(domain.fft(&input));
            });
            let gpu_fft = median(|| {
                let mut values = input.clone();
                fft(&mut values, false, Fr::ONE);
                black_box(values);
            });
            let cpu_eval = median(|| {
                black_box(input.iter().rev().fold(Fr::ZERO, |v, c| v * z + c));
            });
            let gpu_eval = median(|| {
                black_box(evaluate_many(&input, &[z]));
            });
            let cpu_divide = median(|| {
                let mut q = vec![Fr::ZERO; input.len() - 1];
                let mut carry = *input.last().unwrap();
                for i in (0..q.len()).rev() {
                    q[i] = carry;
                    carry = carry * z + input[i];
                }
                black_box(q);
            });
            let gpu_divide = median(|| {
                black_box(divide(&input, z));
            });
            eprintln!(
                "crossover lg={lg} cpu_fft={cpu_fft:.6} gpu_fft={gpu_fft:.6} cpu_eval={cpu_eval:.6} gpu_eval={gpu_eval:.6} cpu_divide={cpu_divide:.6} gpu_divide={gpu_divide:.6}"
            );
        }
        let lg: usize =
            std::env::var("MULTI_STARK_KZG_BENCH_LOG_N").map_or(20, |v| v.parse().unwrap());
        assert!((8..=24).contains(&lg));
        let srs = crate::ark_adapter::srs::Srs::unsafe_dev_setup(1 << lg, b"cuda-msm-benchmark");
        let columns: Vec<Vec<Fr>> = scalars(4)
            .iter()
            .map(|&factor| {
                (0..1 << lg)
                    .into_par_iter()
                    .map(|i| factor * Fr::from(i as u64 + 1))
                    .collect()
            })
            .collect();
        let start = Instant::now();
        let separate: Vec<_> = columns.iter().map(|c| msm(&srs.g1, c)).collect();
        let separate_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let batch = msm_columns(
            &srs.g1,
            &columns.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        );
        let batch_seconds = start.elapsed().as_secs_f64();
        assert_eq!(separate, batch);
        eprintln!(
            "resident_columns lg={lg} columns=4 separate_seconds={separate_seconds:.6} batch_seconds={batch_seconds:.6}"
        );
    }

    #[test]
    #[ignore = "allocates 32 GiB of host memory and a 16 GiB GPU transform"]
    fn cuda_fft_2pow29() {
        let n = 1usize << 29;
        let mut values = vec![Fr::ZERO; n];
        let terms = [
            (0, Fr::from(3)),
            (1, Fr::from(7)),
            (17, -Fr::ONE),
            (n - 1, Fr::from(11)),
        ];
        for &(i, coefficient) in &terms {
            values[i] = coefficient;
        }
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let shift = Fr::from(12345);
        let start = std::time::Instant::now();
        fft(&mut values, false, shift);
        for i in (0..1024)
            .map(|i| i * (n / 1024))
            .chain([n / 2 - 1, n / 2 + 1, n - 1])
        {
            let x = shift * domain.group_gen.pow([i as u64]);
            let expected: Fr = terms.iter().map(|&(k, c)| c * x.pow([k as u64])).sum();
            assert_eq!(values[i], expected, "2^29 FFT index {i}");
        }
        fft(&mut values, true, shift);
        assert!(values.par_iter().enumerate().all(|(i, &value)| {
            value
                == terms
                    .iter()
                    .find(|&&(k, _)| k == i)
                    .map_or(Fr::ZERO, |&(_, c)| c)
        }));
        eprintln!(
            "2^29 coset FFT and inverse: {:.3}s",
            start.elapsed().as_secs_f64()
        );
    }

    #[test]
    #[ignore = "allocates approximately 70 GiB of host memory; exercises all selected GPUs"]
    fn cuda_msm_2pow29() {
        let n = 1usize << 29;
        let factor = scalars(1)[0];
        let mut points = vec![G1Affine::generator(); n];
        points[0] = G1Affine::zero();
        points[n / 2] = -G1Affine::generator();
        let scalars: Vec<_> = (0..n)
            .into_par_iter()
            .map(|i| factor * Fr::from(i as u64 + 1))
            .collect();
        let total = factor
            * (Fr::from(n as u64) * Fr::from(n as u64 + 1) / Fr::from(2)
                - Fr::ONE
                - Fr::from(2) * Fr::from(n as u64 / 2 + 1));
        let start = std::time::Instant::now();
        assert_eq!(msm(&points, &scalars), G1Affine::generator() * total);
        eprintln!("2^29 chunked MSM: {:.3}s", start.elapsed().as_secs_f64());
    }
}
