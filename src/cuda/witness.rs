//! Device main-trace generation and commitment with bounded construction workspace.

use std::sync::{OnceLock, atomic::AtomicBool};

use p3_commit::Pcs;
use p3_field::Field;
use p3_matrix::{Dimensions, dense::RowMajorMatrix};
use p3_symmetric::MerkleCap;

use super::mmcs::CudaMmcsData;
use super::{CudaLde, CudaMixedMerkleTree, device_memory_info};
use crate::config::{Com, Domain, PcsData};
use crate::types::{GoldilocksBlake3Config as Config, InnerPcs as CudaPcs, Val};
use crate::witness::TraceSource;

pub(crate) fn record_lde_spill(device: i32, bytes: usize) {
    tracing::debug!(device, bytes, "spilled active LDE");
}

fn spill_lde(lde: &CudaLde) -> RowMajorMatrix<Val> {
    let matrix = lde.to_row_major_matrix();
    // Materialization synchronizes every use of the released values.
    unsafe { lde.release_values() };
    record_lde_spill(lde.device_id, lde.height() * lde.width() * 8);
    matrix
}

pub(crate) fn commit(
    pcs: &CudaPcs,
    evaluations: Vec<(Domain<Config>, TraceSource<Val>)>,
) -> (Com<Config>, PcsData<Config>) {
    commit_with(pcs, evaluations, None)
}

/// Stage-one LDEs with the per-matrix storage the Merkle tree is built over.
struct Constructed {
    resident: Vec<Option<CudaLde>>,
    host: Vec<Option<RowMajorMatrix<Val>>>,
    retained: Vec<Option<RowMajorMatrix<Val>>>,
}

/// `keep_traces` overrides the memory-derived choice between the resident
/// and the bounded construction, so tests can exercise both.
pub(crate) fn commit_with(
    pcs: &CudaPcs,
    evaluations: Vec<(Domain<Config>, TraceSource<Val>)>,
    keep_traces: Option<bool>,
) -> (Com<Config>, PcsData<Config>) {
    if evaluations
        .iter()
        .all(|(_, m)| matches!(m, TraceSource::Host(_)))
    {
        tracing::debug!(
            device = pcs.dft.device_id(),
            path = "host_pcs",
            "main commitment path"
        );
        return <CudaPcs as Pcs<crate::types::ExtVal, crate::types::Challenger>>::commit(
            pcs,
            evaluations.into_iter().map(|(d, m)| (d, m.materialize())),
        );
    }
    let device = pcs.dft.device_id();
    tracing::debug!(device, path = "prepared", "main commitment path");
    let blowup = pcs.fri.log_blowup;
    let dimensions: Vec<_> = evaluations
        .iter()
        .map(|(_, m)| Dimensions {
            width: m.width(),
            height: m
                .height()
                .checked_shl(blowup.try_into().expect("LDE blowup exceeds u32"))
                .expect("LDE height overflow"),
        })
        .collect();
    for (domain, source) in &evaluations {
        assert_eq!(
            domain.size(),
            source.height(),
            "main-trace domain height mismatch"
        );
    }
    let max_height = dimensions.iter().map(|d| d.height).max().unwrap();
    let (initial_free, total) = device_memory_info(device);
    let minimum_free = super::minimum_free_bytes(total);
    let reserve = minimum_free.saturating_add(128 << 20);
    let source_bytes = evaluations
        .iter()
        .map(|(_, m)| m.height().saturating_mul(m.width()))
        .sum::<usize>()
        .saturating_mul(size_of::<Val>());
    let lde_bytes = source_bytes
        .checked_shl(blowup.try_into().expect("LDE blowup exceeds u32"))
        .unwrap_or(usize::MAX);
    let max_workspace = evaluations
        .iter()
        .map(|(domain, source)| {
            let plan = pcs.dft.lde_plan(
                source.height(),
                source.width(),
                blowup,
                Val::GENERATOR / domain.shift(),
            );
            plan.scratch_bytes()
                .saturating_add(plan.constant_bytes())
                .saturating_add(writer_workspace(source))
        })
        .max()
        .unwrap_or(0);
    let restore = evaluations
        .iter()
        .map(|(_, source)| state_restore_bytes(source))
        .max()
        .unwrap_or(0);
    // Every trace and LDE stays resident through the tree build, as on the
    // all-host path when its batch fits beside the minimum free headroom.
    let resident_footprint = source_bytes
        .saturating_add(lde_bytes)
        .saturating_add(max_workspace)
        .saturating_add(restore)
        .saturating_add(max_height.saturating_mul(96))
        .saturating_add(reserve);
    let keep_traces = keep_traces.unwrap_or(resident_footprint <= initial_free);
    let Constructed {
        resident,
        mut host,
        retained,
    } = if keep_traces {
        let wave_size = initial_free
            .saturating_sub(source_bytes)
            .saturating_sub(lde_bytes)
            .saturating_sub(restore)
            .saturating_sub(reserve)
            .checked_div(max_workspace.max(1))
            .unwrap_or(1)
            .clamp(1, super::MAX_CONCURRENT_LDES);
        construct_resident(pcs, evaluations, source_bytes, total, wave_size)
    } else {
        construct_bounded(pcs, evaluations, reserve)
    };
    let _merkle_span = tracing::info_span!("stark/commit_merkle").entered();
    let needed = max_height.saturating_mul(96).saturating_add(reserve);
    for index in 0..resident.len() {
        if device_memory_info(device).0 >= needed {
            break;
        }
        if host[index].is_none() {
            host[index] = Some(spill_lde(resident[index].as_ref().unwrap()));
        }
    }
    assert!(
        device_memory_info(device).0 >= needed,
        "main Merkle tree exceeds device admission; reduce the shard cell budget"
    );
    let mut spilled_heights: std::collections::BTreeSet<_> = host
        .iter()
        .enumerate()
        .filter_map(|(index, h)| h.is_some().then_some(dimensions[index].height))
        .collect();
    // A height group wider than the device leaf kernel hashes is spilled
    // whole and hashed on the host, like a group spilled for memory.
    spilled_heights.extend(super::mmcs::host_hashed_heights(dimensions.iter().copied()));
    for index in 0..host.len() {
        if host[index].is_none() && spilled_heights.contains(&dimensions[index].height) {
            let lde = resident[index].as_ref().unwrap();
            host[index] = Some(spill_lde(lde));
        }
    }
    let resident_refs: Vec<_> = resident
        .iter()
        .zip(&host)
        .map(|(r, h)| if h.is_none() { r.as_ref() } else { None })
        .collect();
    let host_refs: Vec<_> = host.iter().map(Option::as_ref).collect();
    let deferred = vec![None; dimensions.len()];
    let digests = super::mmcs::hash_host_only_height_groups(
        &host,
        &resident_refs,
        &deferred,
        &std::collections::BTreeSet::new(),
    );
    let tree =
        CudaMixedMerkleTree::from_hybrid(device, &resident_refs, &host_refs, &deferred, &digests);
    let commitment = MerkleCap::new(vec![tree.root()]);
    let active = host.iter().map(|h| AtomicBool::new(h.is_none())).collect();
    let committed = host
        .into_iter()
        .map(|h| {
            let cell = OnceLock::new();
            if let Some(h) = h {
                cell.set(h).unwrap();
            }
            cell
        })
        .collect();
    let count = dimensions.len();
    (
        commitment,
        CudaMmcsData::Hybrid {
            resident,
            resident_active: active,
            materialize: Box::new(CudaLde::to_row_major_matrix),
            committed_matrices: committed,
            deferred_matrices: (0..count).map(|_| None).collect(),
            dimensions,
            retained_traces: retained,
            tree,
        },
    )
}

fn writer_workspace(source: &TraceSource<Val>) -> usize {
    match source {
        TraceSource::Host(_) => 0,
        TraceSource::Generated(generator) => generator.device_workspace_bytes(),
    }
}

/// Device bytes a generated source allocates to restore released state
/// before its first tile.
fn state_restore_bytes(source: &TraceSource<Val>) -> usize {
    match source {
        TraceSource::Generated(generator) if !generator.device_state_resident() => {
            generator.device_state_bytes()
        }
        _ => 0,
    }
}

fn lde_of(pcs: &CudaPcs, domain: &Domain<Config>, source: &TraceSource<Val>) -> CudaLde {
    let blowup = pcs.fri.log_blowup;
    let shift = Val::GENERATOR / domain.shift();
    let _source_span = tracing::info_span!(
        "stark/commit_source",
        kind = match source {
            TraceSource::Host(_) => "host",
            TraceSource::Generated(_) => "generated",
        },
        height = source.height(),
        width = source.width()
    )
    .entered();
    match source {
        TraceSource::Host(matrix) => pcs.dft.coset_lde_batch_resident(matrix, blowup, shift),
        TraceSource::Generated(generator) => {
            pcs.dft
                .generate_coset_lde(std::sync::Arc::clone(generator), blowup, shift)
        }
    }
}

/// Builds every LDE in concurrent waves and keeps each device trace for the
/// lookup pass, releasing traces only where the all-host path would: when
/// this commitment is large enough to crowd later stages, or free memory
/// falls below the minimum headroom.
fn construct_resident(
    pcs: &CudaPcs,
    evaluations: Vec<(Domain<Config>, TraceSource<Val>)>,
    source_bytes: usize,
    total_bytes: usize,
    wave_size: usize,
) -> Constructed {
    let device = pcs.dft.device_id();
    let blowup = pcs.fri.log_blowup;
    for (domain, source) in &evaluations {
        pcs.dft.prepare_coset_lde_constants(
            source.height(),
            blowup,
            Val::GENERATOR / domain.shift(),
        );
    }
    // Small batches of many tiny matrices lose more to concurrent CUDA entry
    // than they gain from overlap.
    let parallel = source_bytes >= 10_000_000 * size_of::<Val>();
    let transform =
        |(domain, source): &(Domain<Config>, TraceSource<Val>)| lde_of(pcs, domain, source);
    let ldes = if parallel {
        super::bounded_parallel_map(&evaluations, wave_size, transform)
    } else {
        evaluations.iter().map(transform).collect()
    };
    let (free_bytes, _) = device_memory_info(device);
    let minimum_free = super::minimum_free_bytes(total_bytes);
    let proactive_spill = source_bytes >= total_bytes / 12;
    let large_matrix = total_bytes / 120;
    let mut projected_free = free_bytes;
    let mut by_size: Vec<_> = evaluations
        .iter()
        .enumerate()
        .map(|(index, (_, source))| {
            (
                index,
                source
                    .height()
                    .saturating_mul(source.width())
                    .saturating_mul(size_of::<Val>()),
            )
        })
        .collect();
    by_size.sort_unstable_by_key(|&(_, bytes)| core::cmp::Reverse(bytes));
    for (index, bytes) in by_size {
        let needs_headroom = projected_free < minimum_free;
        let crowds_later_stages = proactive_spill && bytes >= large_matrix;
        if !needs_headroom && !crowds_later_stages {
            break;
        }
        // SAFETY: construction is synchronous and complete; a host source
        // stays retained below and a generated one regenerates its rows.
        unsafe { ldes[index].release_trace() };
        projected_free = projected_free.saturating_add(bytes);
    }
    let count = ldes.len();
    Constructed {
        resident: ldes.into_iter().map(Some).collect(),
        host: (0..count).map(|_| None).collect(),
        retained: evaluations
            .into_iter()
            .map(|(_, source)| match source {
                TraceSource::Host(matrix) => Some(matrix),
                TraceSource::Generated(_) => None,
            })
            .collect(),
    }
}

/// Builds one LDE at a time when the batch does not fit resident. Each
/// device trace is released after its LDE, and earlier LDEs spill to the
/// host whenever the next one would leave less than `reserve` free.
fn construct_bounded(
    pcs: &CudaPcs,
    evaluations: Vec<(Domain<Config>, TraceSource<Val>)>,
    reserve: usize,
) -> Constructed {
    let device = pcs.dft.device_id();
    let blowup = pcs.fri.log_blowup;
    let mut resident: Vec<Option<CudaLde>> = Vec::with_capacity(evaluations.len());
    let mut host: Vec<Option<RowMajorMatrix<Val>>> = Vec::with_capacity(evaluations.len());
    let mut retained = Vec::with_capacity(evaluations.len());
    for (domain, source) in evaluations {
        let shift = Val::GENERATOR / domain.shift();
        let plan = pcs
            .dft
            .lde_plan(source.height(), source.width(), blowup, shift);
        let needed = source
            .height()
            .saturating_mul(source.width())
            .saturating_mul(8)
            .saturating_mul((1 << blowup) + 1)
            .saturating_add(plan.scratch_bytes())
            .saturating_add(plan.constant_bytes())
            .saturating_add(writer_workspace(&source))
            .saturating_add(state_restore_bytes(&source))
            .saturating_add(reserve);
        // Generator caches go before any LDE spills: they are rebuilt from
        // host seeds on demand, a spilled LDE is uploaded again. A generated
        // source would rebuild a released cache at once, so it releases none.
        if matches!(source, TraceSource::Host(_)) {
            for slot in resident.iter() {
                if device_memory_info(device).0 >= needed {
                    break;
                }
                slot.as_ref().unwrap().release_generator_device();
            }
        }
        for index in 0..resident.len() {
            if device_memory_info(device).0 >= needed {
                break;
            }
            if host[index].is_none() {
                let lde = resident[index].as_ref().unwrap();
                host[index] = Some(spill_lde(lde));
            }
        }
        assert!(
            device_memory_info(device).0 >= needed,
            "generated trace commitment exceeds device admission; reduce the shard cell budget"
        );
        pcs.dft
            .prepare_coset_lde_constants(source.height(), blowup, shift);
        let lde = lde_of(pcs, &domain, &source);
        // Both kinds retain a bounded recovery source. Raw device rows need
        // not coexist with later lookup/quotient workspace.
        unsafe { lde.release_trace() };
        resident.push(Some(lde));
        host.push(None);
        retained.push(match source {
            TraceSource::Host(matrix) => Some(matrix),
            TraceSource::Generated(_) => None,
        });
    }
    Constructed {
        resident,
        host,
        retained,
    }
}
