//! Shared pair selection and atomic memory admission for distributed consumers.

use super::*;

const RESIDENT_LIMIT: usize = 8 << 30;
pub(super) const COPY_BYTES: usize = 16 << 20;

#[derive(Clone, Copy)]
pub(super) struct Options {
    pub(super) tile_rows: usize,
    pub(super) force_host: bool,
    pub(super) budget_limit: usize,
    #[cfg(test)]
    pub(super) reject_after_lease: bool,
}

impl Options {
    pub(super) fn from_env(tile_variable: &str, staging_variable: &str) -> Self {
        Self {
            tile_rows: std::env::var(tile_variable).map_or(1 << 18, |value| {
                value.parse().expect("invalid distributed tile size")
            }),
            force_host: std::env::var(staging_variable).is_ok_and(|value| value == "1"),
            budget_limit: usize::MAX,
            #[cfg(test)]
            reject_after_lease: false,
        }
    }
}

unsafe extern "C" {
    fn multi_stark_kzg_distributed_peers(devices: *const i32) -> i32;
}

pub(super) fn pair_indices(access: &[Vec<bool>]) -> Option<[usize; 4]> {
    let count = access.len();
    if access.iter().any(|row| row.len() != count) {
        return None;
    }
    let mutual = |a: usize, b: usize| access[a][b] && access[b][a];
    for a in 0..count {
        for b in a + 1..count {
            if !mutual(a, b) {
                continue;
            }
            for c in a + 1..count {
                if c == b {
                    continue;
                }
                for d in c + 1..count {
                    if d != b && mutual(c, d) {
                        return Some([a, b, c, d]);
                    }
                }
            }
        }
    }
    None
}

pub(super) struct Plan {
    indices: [usize; 4],
    pub(super) ordinals: [i32; 4],
}

impl Plan {
    pub(super) fn new(peaks: &[usize; 4], options: Options) -> Option<Self> {
        if peaks.iter().any(|&bytes| bytes > options.budget_limit) {
            return None;
        }
        let devices = Devices::get();
        let count = devices.ids.len();
        if count < 4 {
            return None;
        }
        let mut access = vec![vec![false; count]; count];
        for (accessor, row) in access.iter_mut().enumerate() {
            for (owner, supported) in row.iter_mut().enumerate() {
                if accessor == owner
                    || devices.resident_limits[accessor] > RESIDENT_LIMIT
                    || devices.resident_limits[owner] > RESIDENT_LIMIT
                {
                    continue;
                }
                let mut can_access = 0;
                check(
                    unsafe {
                        multi_stark_kzg_peer_access(
                            devices.ids[accessor],
                            devices.ids[owner],
                            &mut can_access,
                        )
                    },
                    "peer topology query",
                );
                *supported = can_access != 0;
            }
        }
        let indices = pair_indices(&access)?;
        if indices
            .iter()
            .zip(*peaks)
            .any(|(&index, peak)| devices.potential_budget(index) < peak)
        {
            return None;
        }
        Some(Self {
            indices,
            ordinals: indices.map(|index| devices.ids[index]),
        })
    }

    pub(super) fn acquire(
        &self,
        peaks: &[usize; 4],
        _options: Options,
    ) -> Option<Vec<Device<'static>>> {
        let group = Devices::get().acquire_group(&self.indices);
        check(
            unsafe { multi_stark_kzg_distributed_peers(self.ordinals.as_ptr()) },
            "distributed pool peer access",
        );
        #[cfg(test)]
        if _options.reject_after_lease {
            assert!(!group[0].prepare_budget(usize::MAX));
            return None;
        }
        if !group
            .iter()
            .zip(*peaks)
            .all(|(device, peak)| device.prepare_budget(peak))
        {
            return None;
        }
        Some(group)
    }
}
