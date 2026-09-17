//! Device main-trace generation and commitment with bounded construction workspace.

use std::sync::{OnceLock, atomic::AtomicBool};

use p3_commit::Pcs;
use p3_field::Field;
use p3_matrix::{Dimensions, dense::RowMajorMatrix};
use p3_symmetric::MerkleCap;

use super::mmcs::CudaMmcsData;
use super::{CudaLde, CudaMixedMerkleTree, device_memory_info};
use crate::config::{Com, Domain, PcsData};
use crate::types::{GoldilocksBlake3Config as Config, Pcs as CudaPcs, Val};
use crate::witness::TraceSource;

pub(crate) fn record_lde_spill(device: i32, bytes: usize) {
    tracing::debug!(device, bytes, "spilled active LDE");
    tracing::info!(target: "prover_metrics", metric = "lde_spill", device, bytes);
}

fn spill_lde(lde: &CudaLde) -> RowMajorMatrix<Val> {
    let matrix = lde.to_row_major_matrix();
    // Materialization synchronizes every use of the released values.
    unsafe { lde.release_values() };
    record_lde_spill(lde.device_id, lde.height() * lde.width() * 8);
    matrix
}

fn reserve_bytes(total: usize) -> usize {
    std::env::var("MULTI_STARK_CUDA_MIN_FREE_BYTES")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(total / 4)
}

pub(crate) fn commit(
    pcs: &CudaPcs,
    evaluations: Vec<(Domain<Config>, TraceSource<Val>)>,
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
    let (_, total) = device_memory_info(device);
    let reserve = reserve_bytes(total).saturating_add(128 << 20);
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
            .saturating_add(reserve);
        // Generator caches go before any LDE spills: they are rebuilt from
        // host seeds on demand, a spilled LDE is uploaded again.
        for slot in resident.iter() {
            if device_memory_info(device).0 >= needed {
                break;
            }
            slot.as_ref().unwrap().release_generator_device();
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
        let source_span = tracing::info_span!(
            "stark/commit_source",
            kind = match &source {
                TraceSource::Host(_) => "host",
                TraceSource::Generated(_) => "generated",
            },
            height = source.height(),
            width = source.width()
        )
        .entered();
        pcs.dft
            .prepare_coset_lde_constants(source.height(), blowup, shift);
        let (lde, trace) = match source {
            TraceSource::Host(matrix) => (
                pcs.dft.coset_lde_batch_resident(&matrix, blowup, shift),
                Some(matrix),
            ),
            TraceSource::Generated(source) => {
                (pcs.dft.generate_coset_lde(source, blowup, shift), None)
            }
        };
        // Both kinds retain a bounded recovery source. Raw device rows need
        // not coexist with later lookup/quotient workspace.
        unsafe { lde.release_trace() };
        let spilled = if std::env::var("MULTI_STARK_CUDA_TRACE_FORCE_SPILL").is_ok_and(|v| v == "1")
        {
            Some(spill_lde(&lde))
        } else {
            None
        };
        resident.push(Some(lde));
        host.push(spilled);
        retained.push(trace);
        drop(source_span);
    }
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
