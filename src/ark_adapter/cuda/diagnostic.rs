use super::*;

unsafe extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaDeviceGetMemPool(pool: *mut *mut c_void, device: i32) -> i32;
    fn cudaMemPoolGetAttribute(pool: *mut c_void, attribute: i32, value: *mut c_void) -> i32;
    fn cudaMemPoolSetAttribute(pool: *mut c_void, attribute: i32, value: *const c_void) -> i32;
}

pub(super) fn pool_high_water(reset: bool) -> Vec<u64> {
    Devices::get()
        .ids
        .iter()
        .map(|&id| {
            let mut pool = std::ptr::null_mut();
            let mut bytes = 0u64;
            // CUDA's cudaMemPoolAttrUsedMemHigh is 0x8 and permits reset to zero.
            check(unsafe { cudaSetDevice(id) }, "benchmark device selection");
            check(
                unsafe { cudaDeviceGetMemPool(&mut pool, id) },
                "benchmark pool query",
            );
            check(
                unsafe { cudaMemPoolGetAttribute(pool, 0x8, (&mut bytes as *mut u64).cast()) },
                "pool high water",
            );
            if reset {
                let zero = 0u64;
                check(
                    unsafe { cudaMemPoolSetAttribute(pool, 0x8, (&zero as *const u64).cast()) },
                    "reset pool high water",
                );
            }
            bytes
        })
        .collect()
}

pub(super) fn memory_kib(path: &str, label: &str) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix(label)?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}

pub(super) fn warm_ntts() {
    for index in 0..Devices::get().ids.len() {
        let device = Devices::get().acquire_at(index);
        let mut warm = vec![Fr::ONE; 1024];
        check(
            unsafe {
                multi_stark_kzg_fft(
                    device.id(),
                    warm.as_mut_ptr().cast(),
                    10,
                    false,
                    &Fr::ONE.0.0,
                )
            },
            "benchmark NTT warmup",
        );
    }
}
