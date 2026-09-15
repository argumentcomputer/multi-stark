//! Device main-trace generation and tree-only stage-one checkpoints.

use std::sync::{OnceLock, atomic::AtomicBool};

use p3_commit::Pcs;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::{Dimensions, dense::RowMajorMatrix};
use p3_symmetric::MerkleCap;

use super::mmcs::CudaMmcsData;
use super::{CudaLde, CudaMixedMerkleTree, device_memory_info};
use crate::config::{Com, Domain, PcsData};
use crate::types::{GoldilocksBlake3Config as Config, Pcs as CudaPcs, Val};
use crate::witness::{PreparedWitness, TraceSource, TreeCheckpoint};

struct Checkpoint {
    tree: CudaMixedMerkleTree,
    dimensions: Vec<Dimensions>,
}

pub(crate) fn checkpoint(data: PcsData<Config>, max_bytes: usize) -> Option<TreeCheckpoint> {
    let (tree, dimensions) = match data {
        CudaMmcsData::Cpu(_) => return None,
        CudaMmcsData::Hybrid {
            tree,
            dimensions,
            deferred_matrices,
            ..
        } => {
            // Joining releases deferred host LDEs before crossing the barrier.
            for deferred in deferred_matrices.into_iter().flatten() {
                if let Some(worker) = deferred
                    .into_inner()
                    .expect("deferred matrix lock poisoned")
                {
                    drop(worker.join().expect("deferred LDE worker panicked"));
                }
            }
            (tree, dimensions)
        }
        CudaMmcsData::Cuda { tree, resident, .. } => {
            let dimensions = resident
                .iter()
                .map(|m| Dimensions {
                    height: m.height(),
                    width: m.width(),
                })
                .collect();
            (tree, dimensions)
        }
    };
    let bytes = tree
        .row_count
        .saturating_mul(64)
        .saturating_add(
            dimensions
                .capacity()
                .saturating_mul(size_of::<Dimensions>()),
        )
        .saturating_add(size_of::<Checkpoint>());
    if bytes > max_bytes {
        return None;
    }
    tracing::debug!(bytes, "retained stage-one Merkle tree");
    Some(TreeCheckpoint {
        bytes,
        data: Box::new(Checkpoint { tree, dimensions }),
    })
}

fn reserve_bytes(total: usize) -> usize {
    std::env::var("MULTI_STARK_CUDA_MIN_FREE_BYTES")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(total / 4)
}

pub(crate) fn cache_headroom(pcs: &CudaPcs, witness: &PreparedWitness<Val>) -> usize {
    let (free, total) = device_memory_info(pcs.dft.device_id());
    let main = witness
        .traces
        .iter()
        .map(|m| m.height().saturating_mul(m.width()).saturating_mul(8))
        .sum::<usize>();
    let lde = main
        .checked_shl(pcs.fri.log_blowup as u32)
        .unwrap_or(usize::MAX);
    // Cache memory is expendable. Leave the active proof ample room for its
    // main and lookup LDEs, quotient, FRI, row tiles and transform constants.
    free.saturating_sub(
        lde.saturating_mul(4)
            .saturating_add(main)
            .saturating_add(reserve_bytes(total))
            .saturating_add(256 << 20),
    )
}

pub(crate) fn commit(
    pcs: &CudaPcs,
    evaluations: Vec<(Domain<Config>, TraceSource<Val>)>,
    checkpoint: Option<TreeCheckpoint>,
) -> (Com<Config>, PcsData<Config>) {
    if checkpoint.is_none()
        && evaluations
            .iter()
            .all(|(_, m)| matches!(m, TraceSource::Host(_)))
    {
        return <CudaPcs as Pcs<crate::types::ExtVal, crate::types::Challenger>>::commit(
            pcs,
            evaluations.into_iter().map(|(d, m)| (d, m.materialize())),
        );
    }
    let device = pcs.dft.device_id();
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
    let mut cached = checkpoint
        .and_then(|c| c.data.downcast::<Checkpoint>().ok())
        .filter(|c| {
            c.tree.device_id == device
                && c.dimensions == dimensions
                && evaluations.iter().all(|(d, _)| d.shift() == Val::ONE)
        });
    for (domain, source) in &evaluations {
        assert_eq!(
            domain.size(),
            source.height(),
            "main-trace domain height mismatch"
        );
        pcs.dft.prepare_coset_lde_constants(
            source.height(),
            blowup,
            Val::GENERATOR / domain.shift(),
        );
    }
    let max_height = dimensions.iter().map(|d| d.height).max().unwrap();
    let (_, total) = device_memory_info(device);
    let reserve = reserve_bytes(total)
        .saturating_add(max_height.saturating_mul(96))
        .saturating_add(128 << 20);
    let mut resident: Vec<Option<CudaLde>> = Vec::with_capacity(evaluations.len());
    let mut host: Vec<Option<RowMajorMatrix<Val>>> = Vec::with_capacity(evaluations.len());
    let mut retained = Vec::with_capacity(evaluations.len());
    for (domain, source) in evaluations {
        let needed = source
            .height()
            .saturating_mul(source.width())
            .saturating_mul(8)
            .saturating_mul((1 << blowup) + 1)
            .saturating_add(reserve);
        if device_memory_info(device).0 < needed {
            drop(cached.take());
        }
        for index in 0..resident.len() {
            if device_memory_info(device).0 >= needed {
                break;
            }
            if host[index].is_none() {
                let lde = resident[index].as_ref().unwrap();
                host[index] = Some(lde.to_row_major_matrix());
                // Synchronous materialization has finished every device use.
                unsafe { lde.release_values() };
            }
        }
        assert!(
            device_memory_info(device).0 >= needed,
            "generated trace commitment exceeds device admission; reduce the shard cell budget"
        );
        let shift = Val::GENERATOR / domain.shift();
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
            let matrix = lde.to_row_major_matrix();
            unsafe { lde.release_values() };
            Some(matrix)
        } else {
            None
        };
        resident.push(Some(lde));
        host.push(spilled);
        retained.push(trace);
    }
    let spilled_heights: std::collections::BTreeSet<_> = host
        .iter()
        .enumerate()
        .filter_map(|(index, h)| h.is_some().then_some(dimensions[index].height))
        .collect();
    for index in 0..host.len() {
        if host[index].is_none() && spilled_heights.contains(&dimensions[index].height) {
            let lde = resident[index].as_ref().unwrap();
            host[index] = Some(lde.to_row_major_matrix());
            unsafe { lde.release_values() };
        }
    }
    let resident_refs: Vec<_> = resident
        .iter()
        .zip(&host)
        .map(|(r, h)| if h.is_none() { r.as_ref() } else { None })
        .collect();
    let host_refs: Vec<_> = host.iter().map(Option::as_ref).collect();
    let tree = if let Some(cached) = cached {
        tracing::debug!(
            bytes = cached.tree.row_count * 64,
            "reused stage-one Merkle tree"
        );
        cached.tree
    } else {
        let deferred = vec![None; dimensions.len()];
        let digests = super::mmcs::hash_host_only_height_groups(
            &host,
            &resident_refs,
            &deferred,
            &std::collections::BTreeSet::new(),
        );
        CudaMixedMerkleTree::from_hybrid(device, &resident_refs, &host_refs, &deferred, &digests)
    };
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
