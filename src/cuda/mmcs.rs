use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use p3_matrix::{Dimensions, Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_merkle_tree::{MerkleTreeError, MerkleTreeMmcs};
use p3_symmetric::{CryptographicHasher, MerkleCap};

use super::{
    CudaLde, CudaMixedMerkleTree, mixed_lde_open_row, mixed_lde_open_row_at_height,
    mixed_lde_open_rows, mixed_lde_open_rows_at_height,
};
use crate::types::Blake3CompressionFunction;
use p3_blake3::Blake3;
use p3_symmetric::SerializingHasher;

type CpuMmcs =
    MerkleTreeMmcs<Goldilocks, u8, SerializingHasher<Blake3>, Blake3CompressionFunction, 2, 32>;
type CpuData<M> = <CpuMmcs as Mmcs<Goldilocks>>::ProverData<M>;
pub(crate) type DeferredMatrix<M> = Option<(Dimensions, std::thread::JoinHandle<M>)>;
type DeferredWorker<M> = Option<std::sync::Mutex<Option<std::thread::JoinHandle<M>>>>;

#[derive(Clone, Copy)]
pub enum CudaMatrixSource<'a> {
    Resident(&'a CudaLde),
    Host(&'a RowMajorMatrix<Goldilocks>),
}

pub enum CudaMmcsData<M> {
    Cpu(CpuData<M>),
    Hybrid {
        // Drop resident LDEs before their retained host traces.
        resident: Vec<Option<CudaLde>>,
        resident_active: Vec<std::sync::atomic::AtomicBool>,
        materialize: Box<dyn Fn(&CudaLde) -> M + Send + Sync>,
        committed_matrices: Vec<std::sync::OnceLock<M>>,
        deferred_matrices: Vec<DeferredWorker<M>>,
        dimensions: Vec<Dimensions>,
        retained_traces: Vec<Option<M>>,
        tree: CudaMixedMerkleTree,
    },
    Cuda {
        resident: std::sync::Arc<Vec<CudaLde>>,
        materialize: Option<Box<dyn Fn() -> Vec<M> + Send + Sync>>,
        // `materialize` can own the last Arc to `resident`; drop it before
        // retained host matrices so pinned trace pointers cannot outlive them.
        committed_matrices: std::sync::OnceLock<Vec<M>>,
        retained_traces: std::sync::OnceLock<Vec<M>>,
        tree: CudaMixedMerkleTree,
    },
}

fn hybrid_matrix<'a, M>(
    matrices: &'a [std::sync::OnceLock<M>],
    deferred: &[DeferredWorker<M>],
    index: usize,
) -> &'a M {
    matrices[index].get_or_init(|| {
        let started = std::time::Instant::now();
        let matrix = deferred[index]
            .as_ref()
            .expect("hybrid matrix has neither storage nor deferred computation")
            .lock()
            .expect("deferred matrix lock poisoned")
            .take()
            .expect("deferred matrix worker already consumed")
            .join()
            .expect("deferred matrix worker panicked");
        if super::memory_diagnostics_enabled() {
            eprintln!(
                "[multi-stark/cuda] waited for deferred matrix {index}: {:.3}s",
                started.elapsed().as_secs_f64()
            );
        }
        matrix
    })
}

fn hybrid_resident_candidates<M>(
    data: &CudaMmcsData<M>,
    protected_index: Option<usize>,
) -> Vec<(usize, usize, bool)> {
    let CudaMmcsData::Hybrid {
        resident,
        resident_active,
        committed_matrices,
        ..
    } = data
    else {
        return Vec::new();
    };
    resident
        .iter()
        .enumerate()
        .filter_map(|(index, lde)| {
            if protected_index == Some(index) {
                return None;
            }
            let lde = lde.as_ref()?;
            resident_active[index]
                .load(std::sync::atomic::Ordering::Acquire)
                .then_some((
                    index,
                    lde.height().saturating_mul(lde.width()),
                    committed_matrices[index].get().is_some(),
                ))
        })
        .collect()
}

fn evict_hybrid_resident<M>(data: &CudaMmcsData<M>, index: usize) {
    let CudaMmcsData::Hybrid {
        resident,
        resident_active,
        materialize,
        committed_matrices,
        ..
    } = data
    else {
        panic!("attempted to evict a non-hybrid CUDA commitment");
    };
    let lde = resident[index]
        .as_ref()
        .expect("hybrid resident matrix is missing its LDE");
    committed_matrices[index].get_or_init(|| materialize(lde));
    // SAFETY: admission transitions run between proving stages, when no CUDA
    // operation can access this LDE or its trace.
    unsafe { lde.release_values() };
    resident_active[index].store(false, std::sync::atomic::Ordering::Release);
}

impl<M> CudaMmcsData<M> {
    pub(crate) fn resident(&self, index: usize) -> Option<&CudaLde> {
        match self {
            Self::Cuda { resident, .. } => resident.get(index),
            Self::Hybrid {
                resident,
                resident_active,
                ..
            } => {
                if resident_active
                    .get(index)?
                    .load(std::sync::atomic::Ordering::Acquire)
                {
                    resident.get(index)?.as_ref()
                } else {
                    None
                }
            }
            Self::Cpu(_) => None,
        }
    }
}

impl CudaMmcsData<RowMajorMatrix<Goldilocks>> {
    pub(crate) fn resident_with_trace(&self, index: usize) -> Option<&CudaLde> {
        match self {
            Self::Cuda {
                resident,
                retained_traces,
                ..
            } => {
                let lde = resident.get(index)?;
                if let Some(retained) = retained_traces.get() {
                    // SAFETY: the matrix is retained in this prover data and
                    // drops after every Arc owner of `resident`. This proving
                    // path serializes attachment changes with CUDA use.
                    unsafe { lde.attach_trace(retained.get(index)?) };
                }
                Some(lde)
            }
            Self::Hybrid {
                resident,
                resident_active,
                retained_traces,
                ..
            } => {
                if !resident_active
                    .get(index)?
                    .load(std::sync::atomic::Ordering::Acquire)
                {
                    return None;
                }
                let lde = resident.get(index)?.as_ref()?;
                if let Some(trace) = retained_traces.get(index)?.as_ref() {
                    // SAFETY: the trace is owned by this prover data and drops
                    // after the resident LDE which holds the registered pointer.
                    unsafe { lde.attach_trace(trace) };
                }
                Some(lde)
            }
            Self::Cpu(_) => None,
        }
    }
}

#[cfg(test)]
pub(crate) fn hash_cpu_height_groups<F: p3_field::Field>(
    matrices: &[Option<RowMajorMatrix<F>>],
) -> Vec<(usize, Vec<[u8; 32]>)> {
    let mut groups = std::collections::BTreeMap::<usize, Vec<&RowMajorMatrix<F>>>::new();
    for matrix in matrices.iter().flatten() {
        groups.entry(matrix.height()).or_default().push(matrix);
    }
    groups
        .into_iter()
        .map(|(height, matrices)| {
            let hasher = SerializingHasher::new(Blake3);
            let digests = (0..height)
                .into_par_iter()
                .map(|row| {
                    hasher.hash_iter(matrices.iter().flat_map(|matrix| matrix.row(row).unwrap()))
                })
                .collect();
            (height, digests)
        })
        .collect()
}

pub(crate) fn hash_host_only_height_groups<F: PrimeField64>(
    matrices: &[Option<RowMajorMatrix<F>>],
    resident: &[Option<CudaLde>],
    deferred_dimensions: &[Option<Dimensions>],
    prehashed_heights: &std::collections::BTreeSet<usize>,
) -> Vec<(usize, Vec<[u8; 32]>)> {
    assert_eq!(matrices.len(), resident.len());
    assert_eq!(matrices.len(), deferred_dimensions.len());
    let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for (index, ((matrix, lde), deferred)) in matrices
        .iter()
        .zip(resident)
        .zip(deferred_dimensions)
        .enumerate()
    {
        assert!(
            usize::from(matrix.is_some())
                + usize::from(lde.is_some())
                + usize::from(deferred.is_some())
                == 1,
            "every partitioned matrix must have exactly one storage backend"
        );
        let height = matrix.as_ref().map_or_else(
            || {
                lde.as_ref()
                    .map_or_else(|| deferred.unwrap().height, CudaLde::height)
            },
            Matrix::height,
        );
        assert!(
            deferred.is_none() || prehashed_heights.contains(&height),
            "a deferred matrix requires a pre-hashed height group"
        );
        groups.entry(height).or_default().push(index);
    }

    let hasher = SerializingHasher::new(Blake3);
    groups
        .into_iter()
        .filter_map(|(height, indices)| {
            if prehashed_heights.contains(&height)
                || indices.iter().any(|&index| resident[index].is_some())
            {
                // CUDA hashes fully resident groups directly and assembles mixed
                // groups from resident and mapped host columns in bounded chunks.
                return None;
            }
            let sources = indices
                .iter()
                .map(|&index| matrices[index].as_ref().unwrap())
                .collect::<Vec<_>>();
            let digests = (0..height)
                .into_par_iter()
                .map(|row| {
                    hasher.hash_iter(sources.iter().flat_map(|matrix| matrix.row(row).unwrap()))
                })
                .collect();
            Some((height, digests))
        })
        .collect()
}

fn hybrid_max_height(dimensions: &[Dimensions]) -> usize {
    dimensions
        .iter()
        .map(|dimensions| dimensions.height)
        .max()
        .unwrap()
}

fn hybrid_open_row<M: Matrix<Goldilocks>>(
    resident: &[Option<CudaLde>],
    resident_active: &[std::sync::atomic::AtomicBool],
    matrices: &[std::sync::OnceLock<M>],
    deferred: &[DeferredWorker<M>],
    dimensions: &[Dimensions],
    index: usize,
) -> Vec<Vec<Goldilocks>> {
    let max_height = hybrid_max_height(dimensions);
    assert!(index < max_height, "MMCS opening index out of bounds");
    let active_ldes = resident
        .iter()
        .zip(resident_active)
        .filter_map(|(resident, active)| {
            active
                .load(std::sync::atomic::Ordering::Acquire)
                .then_some(resident.as_ref())
                .flatten()
        })
        .collect::<Vec<_>>();
    let mut gpu_rows = if !active_ldes.is_empty() {
        mixed_lde_open_row_at_height(active_ldes, max_height, index).into_iter()
    } else {
        Vec::new().into_iter()
    };
    resident
        .iter()
        .zip(resident_active)
        .enumerate()
        .map(|(matrix_index, (resident, active))| {
            if resident.is_some() && active.load(std::sync::atomic::Ordering::Acquire) {
                gpu_rows.next().unwrap()
            } else {
                let matrix = hybrid_matrix(matrices, deferred, matrix_index);
                let row = index >> (max_height.trailing_zeros() - matrix.height().trailing_zeros());
                matrix.row(row).unwrap().into_iter().collect()
            }
        })
        .collect()
}

fn hybrid_open_rows<M: Matrix<Goldilocks>>(
    resident: &[Option<CudaLde>],
    resident_active: &[std::sync::atomic::AtomicBool],
    matrices: &[std::sync::OnceLock<M>],
    deferred: &[DeferredWorker<M>],
    dimensions: &[Dimensions],
    indices: &[usize],
) -> Vec<Vec<Vec<Goldilocks>>> {
    if indices.is_empty() {
        return Vec::new();
    }
    let max_height = hybrid_max_height(dimensions);
    assert!(indices.iter().all(|&index| index < max_height));
    let active_ldes = resident
        .iter()
        .zip(resident_active)
        .filter_map(|(resident, active)| {
            active
                .load(std::sync::atomic::Ordering::Acquire)
                .then_some(resident.as_ref())
                .flatten()
        })
        .collect::<Vec<_>>();
    let mut gpu_queries = if !active_ldes.is_empty() {
        mixed_lde_open_rows_at_height(active_ldes, max_height, indices).into_iter()
    } else {
        (0..indices.len())
            .map(|_| Vec::new())
            .collect::<Vec<_>>()
            .into_iter()
    };
    indices
        .iter()
        .map(|&index| {
            let mut gpu_rows = gpu_queries.next().unwrap().into_iter();
            resident
                .iter()
                .zip(resident_active)
                .enumerate()
                .map(|(matrix_index, (resident, active))| {
                    if resident.is_some() && active.load(std::sync::atomic::Ordering::Acquire) {
                        gpu_rows.next().unwrap()
                    } else {
                        let matrix = hybrid_matrix(matrices, deferred, matrix_index);
                        let row = index
                            >> (max_height.trailing_zeros() - matrix.height().trailing_zeros());
                        matrix.row(row).unwrap().into_iter().collect()
                    }
                })
                .collect()
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct CudaMmcs {
    cpu: CpuMmcs,
    device_id: i32,
}

impl CudaMmcs {
    pub(crate) fn new(cpu: CpuMmcs) -> Self {
        Self {
            cpu,
            device_id: super::configured_device(),
        }
    }
}

pub trait CudaCommitMmcs<T: Send + Sync + Clone>: Mmcs<T> {
    fn cuda_device_id(&self) -> i32;

    fn hash_cuda_hybrid_height_group(
        &self,
        resident: &[Option<&CudaLde>],
        host: &[Option<&RowMajorMatrix<T>>],
    ) -> Vec<[u8; 32]>;

    fn is_cuda_resident<M: Matrix<T>>(&self, data: &Self::ProverData<M>) -> bool;

    fn is_matrix_cuda_resident<M: Matrix<T>>(
        &self,
        data: &Self::ProverData<M>,
        index: usize,
    ) -> bool;

    /// Reports whether a matrix can be accessed without waiting for deferred
    /// host materialization.
    fn is_matrix_source_ready<M: Matrix<T>>(
        &self,
        data: &Self::ProverData<M>,
        index: usize,
    ) -> bool;

    /// Materializes and releases the largest hybrid-resident matrices until
    /// the device has at least `target_free_bytes` available. Returns the
    /// measured free bytes after eviction.
    fn ensure_device_headroom<M: Matrix<T>>(
        &self,
        data: &Self::ProverData<M>,
        target_free_bytes: usize,
        protected_index: Option<usize>,
    ) -> usize;

    /// Selects spill candidates across all supplied commitments instead of
    /// letting iteration order decide which commitment gives up residency.
    fn ensure_device_headroom_batch(
        &self,
        data: &[&Self::ProverData<RowMajorMatrix<T>>],
        target_free_bytes: usize,
    ) -> usize;

    fn matrix_dimensions(&self, data: &Self::ProverData<RowMajorMatrix<T>>) -> Vec<Dimensions>;

    fn cpu_matrices<'a>(
        &self,
        data: &'a Self::ProverData<RowMajorMatrix<T>>,
    ) -> Vec<Option<&'a RowMajorMatrix<T>>>;

    fn with_resident_matrix<R>(
        &self,
        data: &Self::ProverData<RowMajorMatrix<T>>,
        index: usize,
        f: impl FnOnce(&CudaLde) -> R,
    ) -> R;

    fn with_matrix_source<R>(
        &self,
        data: &Self::ProverData<RowMajorMatrix<T>>,
        index: usize,
        f: impl FnOnce(CudaMatrixSource<'_>) -> R,
    ) -> R;

    fn resident_or_upload<M: Matrix<T>>(
        &self,
        data: &Self::ProverData<M>,
    ) -> std::sync::Arc<Vec<CudaLde>>;
    fn commit_cuda_resident(
        &self,
        ldes: Vec<CudaLde>,
    ) -> (Self::Commitment, Self::ProverData<RowMajorMatrix<T>>);

    fn commit_cuda_spillable(
        &self,
        ldes: Vec<CudaLde>,
    ) -> (Self::Commitment, Self::ProverData<RowMajorMatrix<T>>);

    fn commit_cuda_hybrid(
        &self,
        resident: Vec<Option<CudaLde>>,
        host_matrices: Vec<Option<RowMajorMatrix<T>>>,
        deferred_matrices: Vec<DeferredMatrix<RowMajorMatrix<T>>>,
        retained_traces: Vec<Option<RowMajorMatrix<T>>>,
        host_digest_groups: Vec<(usize, Vec<[u8; 32]>)>,
    ) -> (Self::Commitment, Self::ProverData<RowMajorMatrix<T>>);

    fn retain_matrices(
        &self,
        data: &mut Self::ProverData<RowMajorMatrix<T>>,
        matrices: Vec<RowMajorMatrix<T>>,
    );

    fn commit_cuda_storage(
        &self,
        ldes: Vec<RowMajorMatrix<T>>,
    ) -> (Self::Commitment, Self::ProverData<RowMajorMatrix<T>>);

    fn commit_cpu_storage(
        &self,
        ldes: Vec<RowMajorMatrix<T>>,
    ) -> (Self::Commitment, Self::ProverData<RowMajorMatrix<T>>);
}

impl CudaCommitMmcs<Goldilocks> for CudaMmcs {
    fn cuda_device_id(&self) -> i32 {
        self.device_id
    }

    fn hash_cuda_hybrid_height_group(
        &self,
        resident: &[Option<&CudaLde>],
        host: &[Option<&RowMajorMatrix<Goldilocks>>],
    ) -> Vec<[u8; 32]> {
        CudaMixedMerkleTree::hash_hybrid_height_group(self.device_id, resident, host)
    }

    fn is_cuda_resident<M: Matrix<Goldilocks>>(&self, data: &Self::ProverData<M>) -> bool {
        matches!(data, CudaMmcsData::Cuda { .. })
    }

    fn is_matrix_cuda_resident<M: Matrix<Goldilocks>>(
        &self,
        data: &Self::ProverData<M>,
        index: usize,
    ) -> bool {
        match data {
            CudaMmcsData::Cpu(_) => false,
            CudaMmcsData::Hybrid {
                resident,
                resident_active,
                ..
            } => {
                resident
                    .get(index)
                    .expect("hybrid matrix index out of bounds");
                resident_active[index].load(std::sync::atomic::Ordering::Acquire)
            }
            CudaMmcsData::Cuda { resident, .. } => {
                assert!(index < resident.len(), "CUDA matrix index out of bounds");
                true
            }
        }
    }

    fn is_matrix_source_ready<M: Matrix<Goldilocks>>(
        &self,
        data: &Self::ProverData<M>,
        index: usize,
    ) -> bool {
        match data {
            CudaMmcsData::Cpu(data) => {
                assert!(
                    index < self.cpu.get_matrix_heights(data).len(),
                    "CPU matrix index out of bounds"
                );
                true
            }
            CudaMmcsData::Cuda { resident, .. } => {
                assert!(index < resident.len(), "CUDA matrix index out of bounds");
                true
            }
            CudaMmcsData::Hybrid {
                resident_active,
                committed_matrices,
                deferred_matrices,
                ..
            } => {
                let active = resident_active
                    .get(index)
                    .expect("hybrid matrix index out of bounds")
                    .load(std::sync::atomic::Ordering::Acquire);
                active
                    || committed_matrices[index].get().is_some()
                    || deferred_matrices[index].as_ref().is_some_and(|worker| {
                        worker
                            .lock()
                            .expect("deferred matrix lock poisoned")
                            .as_ref()
                            .is_some_and(std::thread::JoinHandle::is_finished)
                    })
            }
        }
    }

    fn ensure_device_headroom<M: Matrix<Goldilocks>>(
        &self,
        data: &Self::ProverData<M>,
        target_free_bytes: usize,
        protected_index: Option<usize>,
    ) -> usize {
        let (mut free_bytes, _) = super::device_memory_info(self.device_id);
        if super::memory_diagnostics_enabled() {
            eprintln!(
                "[multi-stark/cuda] admission start: target={} free={}",
                target_free_bytes, free_bytes
            );
        }
        if free_bytes >= target_free_bytes {
            return free_bytes;
        }
        let mut candidates = hybrid_resident_candidates(data, protected_index);
        while free_bytes < target_free_bytes && !candidates.is_empty() {
            let deficit_cells = target_free_bytes
                .saturating_sub(free_bytes)
                .div_ceil(size_of::<Goldilocks>());
            let candidate = candidates
                .iter()
                .enumerate()
                .filter(|(_, (_, cells, _))| *cells >= deficit_cells)
                .min_by_key(|(_, (_, cells, _))| *cells)
                .or_else(|| {
                    candidates
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, (_, cells, _))| *cells)
                })
                .map(|(position, _)| position)
                .unwrap();
            let (index, cells, _) = candidates.swap_remove(candidate);
            evict_hybrid_resident(data, index);
            // Eviction is a synchronous device release. Re-read the allocator
            // instead of inferring availability from logical matrix sizes:
            // CUDA allocation granularity and per-thread memory pools make a
            // byte-sum projection unreliable across Rayon worker streams.
            free_bytes = super::device_memory_info(self.device_id).0;
            if super::memory_diagnostics_enabled() {
                eprintln!(
                    "[multi-stark/cuda] evicted matrix {index}: cells={cells} free={free_bytes}"
                );
            }
        }
        free_bytes
    }

    fn ensure_device_headroom_batch(
        &self,
        data: &[&Self::ProverData<RowMajorMatrix<Goldilocks>>],
        target_free_bytes: usize,
    ) -> usize {
        let (mut measured_free_bytes, _) = super::device_memory_info(self.device_id);
        if super::memory_diagnostics_enabled() {
            eprintln!(
                "[multi-stark/cuda] batch admission start: target={} free={}",
                target_free_bytes, measured_free_bytes
            );
        }
        if measured_free_bytes >= target_free_bytes {
            return measured_free_bytes;
        }
        let initial_free_bytes = measured_free_bytes;
        let mut released_bytes = 0usize;
        let mut candidates = data
            .iter()
            .enumerate()
            .flat_map(|(data_index, data)| {
                hybrid_resident_candidates(data, None).into_iter().map(
                    move |(matrix_index, cells, host_ready)| {
                        (data_index, matrix_index, cells, host_ready)
                    },
                )
            })
            .collect::<Vec<_>>();
        while initial_free_bytes
            .saturating_add(released_bytes)
            .max(measured_free_bytes)
            < target_free_bytes
            && !candidates.is_empty()
        {
            let available_bytes = initial_free_bytes
                .saturating_add(released_bytes)
                .max(measured_free_bytes);
            let deficit_cells = target_free_bytes
                .saturating_sub(available_bytes)
                .div_ceil(size_of::<Goldilocks>());
            let ready = candidates.iter().any(|candidate| candidate.3);
            let eligible = |candidate: &&(usize, usize, usize, bool)| !ready || candidate.3;
            let candidate = candidates
                .iter()
                .enumerate()
                .filter(|(_, candidate)| eligible(candidate))
                .filter(|(_, (_, _, cells, _))| *cells <= deficit_cells)
                .max_by_key(|(_, (_, _, cells, _))| *cells)
                .or_else(|| {
                    candidates
                        .iter()
                        .enumerate()
                        .filter(|(_, candidate)| eligible(candidate))
                        .min_by_key(|(_, (_, _, cells, _))| *cells)
                })
                .map(|(position, _)| position)
                .unwrap();
            let (data_index, matrix_index, cells, host_ready) = candidates.swap_remove(candidate);
            evict_hybrid_resident(data[data_index], matrix_index);
            // `release_values` synchronously frees this exact payload. CUDA's
            // free-memory counters advance at allocator granularity, however,
            // so requiring `cudaMemGetInfo` to reflect every small release can
            // evict hundreds of matrices to cover a sub-granule reporting gap.
            // The released payload is a conservative lower bound (the LDE may
            // also release interpolation scratch), while the fresh measurement
            // catches unrelated memory becoming available.
            released_bytes =
                released_bytes.saturating_add(cells.saturating_mul(size_of::<Goldilocks>()));
            measured_free_bytes = super::device_memory_info(self.device_id).0;
            if super::memory_diagnostics_enabled() {
                eprintln!(
                    "[multi-stark/cuda] batch evicted commitment {data_index} matrix {matrix_index}: cells={cells} host_ready={host_ready} measured_free={measured_free_bytes} projected_free={}",
                    initial_free_bytes.saturating_add(released_bytes)
                );
            }
        }
        initial_free_bytes
            .saturating_add(released_bytes)
            .max(measured_free_bytes)
    }

    fn matrix_dimensions(
        &self,
        data: &Self::ProverData<RowMajorMatrix<Goldilocks>>,
    ) -> Vec<Dimensions> {
        match data {
            CudaMmcsData::Cpu(data) => self
                .cpu
                .get_matrices(data)
                .into_iter()
                .map(Matrix::dimensions)
                .collect(),
            CudaMmcsData::Hybrid { dimensions, .. } => dimensions.clone(),
            CudaMmcsData::Cuda { resident, .. } => resident
                .iter()
                .map(|lde| Dimensions {
                    width: lde.width(),
                    height: lde.height(),
                })
                .collect(),
        }
    }

    fn cpu_matrices<'a>(
        &self,
        data: &'a Self::ProverData<RowMajorMatrix<Goldilocks>>,
    ) -> Vec<Option<&'a RowMajorMatrix<Goldilocks>>> {
        match data {
            CudaMmcsData::Cpu(data) => self.cpu.get_matrices(data).into_iter().map(Some).collect(),
            CudaMmcsData::Hybrid {
                resident,
                resident_active,
                committed_matrices,
                deferred_matrices,
                ..
            } => resident
                .iter()
                .zip(resident_active)
                .enumerate()
                .map(|(index, (resident, active))| {
                    (resident.is_none() || !active.load(std::sync::atomic::Ordering::Acquire))
                        .then(|| hybrid_matrix(committed_matrices, deferred_matrices, index))
                })
                .collect(),
            CudaMmcsData::Cuda { resident, .. } => (0..resident.len()).map(|_| None).collect(),
        }
    }

    fn with_resident_matrix<R>(
        &self,
        data: &Self::ProverData<RowMajorMatrix<Goldilocks>>,
        index: usize,
        f: impl FnOnce(&CudaLde) -> R,
    ) -> R {
        match data {
            CudaMmcsData::Cuda { resident, .. } => f(resident
                .get(index)
                .expect("CUDA matrix index out of bounds")),
            CudaMmcsData::Hybrid {
                resident,
                resident_active,
                committed_matrices,
                deferred_matrices,
                ..
            } => {
                let resident = resident
                    .get(index)
                    .expect("hybrid matrix index out of bounds");
                if resident_active[index].load(std::sync::atomic::Ordering::Acquire) {
                    let lde = resident.as_ref().expect("active hybrid matrix has no LDE");
                    f(lde)
                } else {
                    let matrix = hybrid_matrix(committed_matrices, deferred_matrices, index);
                    if super::memory_diagnostics_enabled() {
                        let (free_bytes, _) = super::device_memory_info(self.device_id);
                        eprintln!(
                            "[multi-stark/cuda] upload hybrid matrix {index}: bytes={} free={free_bytes}",
                            matrix
                                .height()
                                .saturating_mul(matrix.width())
                                .saturating_mul(size_of::<Goldilocks>())
                        );
                    }
                    let uploaded = CudaLde::from_row_major_matrix(self.device_id, matrix);
                    f(&uploaded)
                }
            }
            CudaMmcsData::Cpu(data) => {
                let matrices = self.cpu.get_matrices(data);
                let matrix = matrices.get(index).expect("CPU matrix index out of bounds");
                if super::memory_diagnostics_enabled() {
                    let (free_bytes, _) = super::device_memory_info(self.device_id);
                    eprintln!(
                        "[multi-stark/cuda] upload CPU matrix {index}: bytes={} free={free_bytes}",
                        matrix
                            .height()
                            .saturating_mul(matrix.width())
                            .saturating_mul(size_of::<Goldilocks>())
                    );
                }
                let uploaded = CudaLde::from_row_major_matrix(self.device_id, matrix);
                f(&uploaded)
            }
        }
    }

    fn with_matrix_source<R>(
        &self,
        data: &Self::ProverData<RowMajorMatrix<Goldilocks>>,
        index: usize,
        f: impl FnOnce(CudaMatrixSource<'_>) -> R,
    ) -> R {
        match data {
            CudaMmcsData::Cuda { resident, .. } => f(CudaMatrixSource::Resident(
                resident
                    .get(index)
                    .expect("CUDA matrix index out of bounds"),
            )),
            CudaMmcsData::Hybrid {
                resident,
                resident_active,
                committed_matrices,
                deferred_matrices,
                ..
            } => {
                if resident_active
                    .get(index)
                    .expect("hybrid matrix index out of bounds")
                    .load(std::sync::atomic::Ordering::Acquire)
                {
                    f(CudaMatrixSource::Resident(
                        resident[index]
                            .as_ref()
                            .expect("active hybrid matrix has no LDE"),
                    ))
                } else {
                    f(CudaMatrixSource::Host(hybrid_matrix(
                        committed_matrices,
                        deferred_matrices,
                        index,
                    )))
                }
            }
            CudaMmcsData::Cpu(data) => {
                let matrices = self.cpu.get_matrices(data);
                f(CudaMatrixSource::Host(
                    matrices.get(index).expect("CPU matrix index out of bounds"),
                ))
            }
        }
    }

    fn commit_cpu_storage(
        &self,
        ldes: Vec<RowMajorMatrix<Goldilocks>>,
    ) -> (
        Self::Commitment,
        Self::ProverData<RowMajorMatrix<Goldilocks>>,
    ) {
        let (commitment, data) = self.cpu.commit(ldes);
        (commitment, CudaMmcsData::Cpu(data))
    }

    fn retain_matrices(
        &self,
        data: &mut Self::ProverData<RowMajorMatrix<Goldilocks>>,
        retained: Vec<RowMajorMatrix<Goldilocks>>,
    ) {
        match data {
            CudaMmcsData::Cuda {
                retained_traces, ..
            } => retained_traces
                .set(retained)
                .expect("fresh CUDA trace matrix cell"),
            CudaMmcsData::Hybrid {
                retained_traces, ..
            } => {
                assert_eq!(retained_traces.len(), retained.len());
                assert!(retained_traces.iter().all(Option::is_none));
                for (slot, matrix) in retained_traces.iter_mut().zip(retained) {
                    *slot = Some(matrix);
                }
            }
            CudaMmcsData::Cpu(_) => {
                panic!("CPU commitment cannot retain CUDA trace matrices")
            }
        }
    }

    fn resident_or_upload<M: Matrix<Goldilocks>>(
        &self,
        data: &Self::ProverData<M>,
    ) -> std::sync::Arc<Vec<CudaLde>> {
        match data {
            CudaMmcsData::Cuda { resident, .. } => std::sync::Arc::clone(resident),
            CudaMmcsData::Hybrid { .. } => {
                panic!("hybrid commitments cannot be converted to all-resident storage")
            }
            CudaMmcsData::Cpu(cpu) => std::sync::Arc::new(
                self.cpu
                    .get_matrices(cpu)
                    .into_iter()
                    .map(|matrix| {
                        let values = (0..matrix.height())
                            .flat_map(|row| matrix.row(row).unwrap())
                            .collect();
                        CudaLde::from_row_major_matrix(
                            self.device_id,
                            &RowMajorMatrix::new(values, matrix.width()),
                        )
                    })
                    .collect(),
            ),
        }
    }
    fn commit_cuda_resident(
        &self,
        ldes: Vec<CudaLde>,
    ) -> (
        Self::Commitment,
        Self::ProverData<RowMajorMatrix<Goldilocks>>,
    ) {
        let ldes = std::sync::Arc::new(ldes);
        let tree = CudaMixedMerkleTree::from_ldes(self.device_id, &ldes);
        let commitment = MerkleCap::new(vec![tree.root()]);
        let materialize_ldes = std::sync::Arc::clone(&ldes);
        (
            commitment,
            CudaMmcsData::Cuda {
                resident: ldes,
                materialize: Some(Box::new(move || {
                    materialize_ldes
                        .iter()
                        .map(CudaLde::to_row_major_matrix)
                        .collect()
                })),
                committed_matrices: std::sync::OnceLock::new(),
                retained_traces: std::sync::OnceLock::new(),
                tree,
            },
        )
    }

    fn commit_cuda_spillable(
        &self,
        ldes: Vec<CudaLde>,
    ) -> (
        Self::Commitment,
        Self::ProverData<RowMajorMatrix<Goldilocks>>,
    ) {
        let matrix_count = ldes.len();
        self.commit_cuda_hybrid(
            ldes.into_iter().map(Some).collect(),
            (0..matrix_count).map(|_| None).collect(),
            (0..matrix_count).map(|_| None).collect(),
            (0..matrix_count).map(|_| None).collect(),
            Vec::new(),
        )
    }

    fn commit_cuda_hybrid(
        &self,
        resident: Vec<Option<CudaLde>>,
        host_matrices: Vec<Option<RowMajorMatrix<Goldilocks>>>,
        deferred_matrices: Vec<DeferredMatrix<RowMajorMatrix<Goldilocks>>>,
        retained_traces: Vec<Option<RowMajorMatrix<Goldilocks>>>,
        host_digest_groups: Vec<(usize, Vec<[u8; 32]>)>,
    ) -> (
        Self::Commitment,
        Self::ProverData<RowMajorMatrix<Goldilocks>>,
    ) {
        assert_eq!(resident.len(), host_matrices.len());
        assert_eq!(resident.len(), deferred_matrices.len());
        assert_eq!(resident.len(), retained_traces.len());
        assert!(
            resident
                .iter()
                .zip(&host_matrices)
                .zip(&deferred_matrices)
                .all(|((gpu, cpu), deferred)| usize::from(gpu.is_some())
                    + usize::from(cpu.is_some())
                    + usize::from(deferred.is_some())
                    == 1),
            "every hybrid matrix must have exactly one storage backend"
        );
        let deferred_dimensions = deferred_matrices
            .iter()
            .map(|deferred| deferred.as_ref().map(|(dimensions, _)| *dimensions))
            .collect::<Vec<_>>();
        let dimensions = resident
            .iter()
            .zip(&host_matrices)
            .zip(&deferred_dimensions)
            .map(|((resident, host), deferred)| {
                resident.as_ref().map_or_else(
                    || {
                        host.as_ref()
                            .map_or_else(|| deferred.unwrap(), Matrix::dimensions)
                    },
                    |lde| Dimensions {
                        width: lde.width(),
                        height: lde.height(),
                    },
                )
            })
            .collect::<Vec<_>>();
        let resident_refs = resident.iter().map(Option::as_ref).collect::<Vec<_>>();
        let host_refs = host_matrices.iter().map(Option::as_ref).collect::<Vec<_>>();
        let tree = CudaMixedMerkleTree::from_hybrid(
            self.device_id,
            &resident_refs,
            &host_refs,
            &deferred_dimensions,
            &host_digest_groups,
        );
        let commitment = MerkleCap::new(vec![tree.root()]);
        let committed_matrices = host_matrices
            .into_iter()
            .map(|matrix| {
                let cell = std::sync::OnceLock::new();
                if let Some(matrix) = matrix {
                    cell.set(matrix).expect("fresh hybrid matrix cell");
                }
                cell
            })
            .collect();
        let deferred_matrices = deferred_matrices
            .into_iter()
            .map(|deferred| deferred.map(|(_, worker)| std::sync::Mutex::new(Some(worker))))
            .collect();
        let resident_active = resident
            .iter()
            .map(|lde| std::sync::atomic::AtomicBool::new(lde.is_some()))
            .collect();
        (
            commitment,
            CudaMmcsData::Hybrid {
                resident,
                resident_active,
                materialize: Box::new(CudaLde::to_row_major_matrix),
                committed_matrices,
                deferred_matrices,
                dimensions,
                retained_traces,
                tree,
            },
        )
    }

    fn commit_cuda_storage(
        &self,
        ldes: Vec<RowMajorMatrix<Goldilocks>>,
    ) -> (
        Self::Commitment,
        Self::ProverData<RowMajorMatrix<Goldilocks>>,
    ) {
        let resident = ldes
            .iter()
            .map(|matrix| CudaLde::from_row_major_matrix(self.device_id, matrix))
            .collect();
        let (commitment, mut data) = self.commit_cuda_resident(resident);
        if let CudaMmcsData::Cuda {
            committed_matrices, ..
        } = &mut data
        {
            let _ = committed_matrices.set(ldes);
        }
        (commitment, data)
    }
}

impl Mmcs<Goldilocks> for CudaMmcs {
    type ProverData<M> = CudaMmcsData<M>;
    type Commitment = MerkleCap<Goldilocks, [u8; 32]>;
    type Proof = Vec<[u8; 32]>;
    type MultiProof = <CpuMmcs as Mmcs<Goldilocks>>::MultiProof;
    type Error = MerkleTreeError;

    fn commit<M: Matrix<Goldilocks>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        if inputs
            .iter()
            .map(|matrix| matrix.height().saturating_mul(matrix.width()))
            .sum::<usize>()
            > 1 << 18
        {
            let resident: Vec<_> = inputs
                .iter()
                .map(|matrix| {
                    let values = (0..matrix.height())
                        .flat_map(|row| matrix.row(row).unwrap())
                        .collect();
                    CudaLde::from_row_major_matrix(
                        self.device_id,
                        &RowMajorMatrix::new(values, matrix.width()),
                    )
                })
                .collect();
            let resident = std::sync::Arc::new(resident);
            let tree = CudaMixedMerkleTree::from_ldes(self.device_id, &resident);
            let commitment = MerkleCap::new(vec![tree.root()]);
            let committed_matrices = std::sync::OnceLock::new();
            committed_matrices
                .set(inputs)
                .ok()
                .expect("fresh CUDA matrix cell");
            return (
                commitment,
                CudaMmcsData::Cuda {
                    resident,
                    materialize: None,
                    committed_matrices,
                    retained_traces: std::sync::OnceLock::new(),
                    tree,
                },
            );
        }
        let (commitment, data) = self.cpu.commit(inputs);
        (commitment, CudaMmcsData::Cpu(data))
    }

    fn open_batch<M: Matrix<Goldilocks>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<Goldilocks, Self> {
        match prover_data {
            CudaMmcsData::Cpu(data) => {
                let opening = self.cpu.open_batch(index, data);
                BatchOpening::new(opening.opened_values, opening.opening_proof)
            }
            CudaMmcsData::Hybrid {
                resident,
                resident_active,
                committed_matrices,
                deferred_matrices,
                dimensions,
                tree,
                ..
            } => BatchOpening::new(
                hybrid_open_row(
                    resident,
                    resident_active,
                    committed_matrices,
                    deferred_matrices,
                    dimensions,
                    index,
                ),
                tree.open_siblings(index),
            ),
            CudaMmcsData::Cuda { resident, tree, .. } => {
                let opened_values = mixed_lde_open_row(resident, index);
                BatchOpening::new(opened_values, tree.open_siblings(index))
            }
        }
    }

    fn get_matrices<'a, M: Matrix<Goldilocks>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        match prover_data {
            CudaMmcsData::Cpu(data) => self.cpu.get_matrices(data),
            CudaMmcsData::Hybrid {
                resident,
                materialize,
                committed_matrices,
                deferred_matrices,
                ..
            } => resident
                .iter()
                .enumerate()
                .map(|(index, resident)| {
                    if resident.is_some() {
                        committed_matrices[index]
                            .get_or_init(|| materialize(resident.as_ref().unwrap()))
                    } else {
                        hybrid_matrix(committed_matrices, deferred_matrices, index)
                    }
                })
                .collect(),
            CudaMmcsData::Cuda {
                committed_matrices,
                materialize,
                ..
            } => committed_matrices
                .get_or_init(|| {
                    materialize
                        .as_ref()
                        .expect("CUDA matrices have no materializer")()
                })
                .iter()
                .collect(),
        }
    }

    fn get_matrix_heights<M: Matrix<Goldilocks>>(
        &self,
        prover_data: &Self::ProverData<M>,
    ) -> Vec<usize> {
        match prover_data {
            CudaMmcsData::Cpu(data) => self.cpu.get_matrix_heights(data),
            CudaMmcsData::Hybrid { dimensions, .. } => dimensions
                .iter()
                .map(|dimensions| dimensions.height)
                .collect(),
            CudaMmcsData::Cuda { resident, .. } => resident.iter().map(CudaLde::height).collect(),
        }
    }

    fn get_max_height<M: Matrix<Goldilocks>>(&self, prover_data: &Self::ProverData<M>) -> usize {
        match prover_data {
            CudaMmcsData::Cpu(data) => self.cpu.get_max_height(data),
            CudaMmcsData::Hybrid { dimensions, .. } => hybrid_max_height(dimensions),
            CudaMmcsData::Cuda { resident, .. } => {
                resident.iter().map(CudaLde::height).max().unwrap()
            }
        }
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, Goldilocks, Self>,
    ) -> Result<(), Self::Error> {
        self.cpu.verify_batch(
            commit,
            dimensions,
            index,
            BatchOpeningRef::new(batch_opening.opened_values, batch_opening.opening_proof),
        )
    }

    fn open_multi_batch<M: Matrix<Goldilocks>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> (Vec<Vec<Vec<Goldilocks>>>, Self::MultiProof) {
        match prover_data {
            CudaMmcsData::Cpu(data) => self.cpu.open_multi_batch(indices, data),
            CudaMmcsData::Hybrid {
                resident,
                resident_active,
                committed_matrices,
                deferred_matrices,
                dimensions,
                tree,
                ..
            } => {
                let opened_values = hybrid_open_rows(
                    resident,
                    resident_active,
                    committed_matrices,
                    deferred_matrices,
                    dimensions,
                    indices,
                );
                let opening_proof = tree.open_pruned_siblings(indices);
                (opened_values, opening_proof)
            }
            CudaMmcsData::Cuda { resident, tree, .. } => {
                let opened_values = if indices.is_empty() {
                    Vec::new()
                } else {
                    mixed_lde_open_rows(resident, indices)
                };
                let opening_proof = tree.open_pruned_siblings(indices);
                (opened_values, opening_proof)
            }
        }
    }

    fn verify_multi_batch<R: AsRef<[Goldilocks]> + PartialEq>(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<R>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error> {
        self.cpu
            .verify_multi_batch(commit, dimensions, indices, opened_values, proof)
    }
}

#[cfg(test)]
mod tests {
    use p3_commit::Mmcs;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    fn matrix(height: usize, width: usize, offset: usize) -> RowMajorMatrix<Goldilocks> {
        RowMajorMatrix::new(
            (0..height * width)
                .map(|index| Goldilocks::from_usize(offset + index))
                .collect(),
            width,
        )
    }

    #[test]
    fn hybrid_openings_survive_device_spill() {
        let matrices = vec![matrix(16, 2, 3), matrix(8, 3, 71), matrix(16, 1, 109)];
        let cpu = CpuMmcs::new(
            SerializingHasher::new(Blake3),
            Blake3CompressionFunction::new(Blake3),
            0,
        );
        let (expected_commitment, expected_data) = cpu.commit(matrices.clone());
        let expected_opening = cpu.open_batch(5, &expected_data);

        let mmcs = CudaMmcs::new(cpu);
        let resident = vec![
            Some(CudaLde::from_row_major_matrix(mmcs.device_id, &matrices[0])),
            None,
            None,
        ];
        let host_matrices = vec![None, Some(matrices[1].clone()), Some(matrices[2].clone())];
        let host_digest_groups = hash_host_only_height_groups(
            &host_matrices,
            &resident,
            &[None, None, None],
            &std::collections::BTreeSet::new(),
        );
        let retained_traces = vec![None, None, None];
        let deferred_matrices = vec![None, None, None];
        let (commitment, data) = mmcs.commit_cuda_hybrid(
            resident,
            host_matrices,
            deferred_matrices,
            retained_traces,
            host_digest_groups,
        );
        assert_eq!(commitment, expected_commitment);

        let resident_opening = mmcs.open_batch(5, &data);
        assert_eq!(
            resident_opening.opened_values,
            expected_opening.opened_values
        );
        assert_eq!(
            resident_opening.opening_proof,
            expected_opening.opening_proof
        );

        mmcs.ensure_device_headroom(&data, usize::MAX, None);
        assert!(!(0..matrices.len()).any(|index| mmcs.is_matrix_cuda_resident(&data, index)));
        let spilled_opening = mmcs.open_batch(5, &data);
        assert_eq!(
            spilled_opening.opened_values,
            expected_opening.opened_values
        );
        assert_eq!(
            spilled_opening.opening_proof,
            expected_opening.opening_proof
        );
    }
}
