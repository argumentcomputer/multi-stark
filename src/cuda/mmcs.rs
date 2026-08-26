use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_goldilocks::Goldilocks;
use p3_matrix::{Dimensions, Matrix, dense::RowMajorMatrix};
use p3_merkle_tree::{MerkleTreeError, MerkleTreeMmcs};
use p3_symmetric::MerkleCap;

use super::{CudaLde, CudaMixedMerkleTree, mixed_lde_open_row, mixed_lde_open_rows};
use crate::types::Blake3CompressionFunction;
use p3_blake3::Blake3;
use p3_symmetric::SerializingHasher;

type CpuMmcs =
    MerkleTreeMmcs<Goldilocks, u8, SerializingHasher<Blake3>, Blake3CompressionFunction, 2, 32>;
type CpuData<M> = <CpuMmcs as Mmcs<Goldilocks>>::ProverData<M>;

pub enum CudaMmcsData<M> {
    Cpu(CpuData<M>),
    Cuda {
        resident: std::sync::Arc<Vec<CudaLde>>,
        materialize: Option<Box<dyn Fn() -> Vec<M> + Send + Sync>>,
        // `materialize` can own the last Arc to `resident`; drop it before
        // retained host matrices so pinned trace pointers cannot outlive them.
        matrices: std::sync::OnceLock<Vec<M>>,
        tree: CudaMixedMerkleTree,
    },
}

impl<M> CudaMmcsData<M> {
    pub(crate) fn resident(&self, index: usize) -> Option<&CudaLde> {
        match self {
            Self::Cuda { resident, .. } => resident.get(index),
            Self::Cpu(_) => None,
        }
    }
}

impl CudaMmcsData<RowMajorMatrix<Goldilocks>> {
    pub(crate) fn resident_with_trace(&self, index: usize) -> Option<&CudaLde> {
        match self {
            Self::Cuda {
                resident, matrices, ..
            } => {
                let lde = resident.get(index)?;
                if let Some(retained) = matrices.get() {
                    // SAFETY: the matrix is retained in this prover data and
                    // drops after every Arc owner of `resident`. This proving
                    // path serializes attachment changes with CUDA use.
                    unsafe { lde.attach_trace(retained.get(index)?) };
                }
                Some(lde)
            }
            Self::Cpu(_) => None,
        }
    }
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

    fn resident_or_upload<M: Matrix<T>>(
        &self,
        data: &Self::ProverData<M>,
    ) -> std::sync::Arc<Vec<CudaLde>>;
    fn commit_cuda_resident(
        &self,
        ldes: Vec<CudaLde>,
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
}

pub trait CudaBatchOpenMmcs<T: Send + Sync + Clone>: Mmcs<T> {
    fn open_batches<M: Matrix<T>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> Vec<BatchOpening<T, Self>>
    where
        Self: Sized;
}

impl CudaBatchOpenMmcs<Goldilocks> for CudaMmcs {
    fn open_batches<M: Matrix<Goldilocks>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> Vec<BatchOpening<Goldilocks, Self>> {
        match prover_data {
            CudaMmcsData::Cpu(_) => indices
                .iter()
                .map(|&index| self.open_batch(index, prover_data))
                .collect(),
            CudaMmcsData::Cuda { resident, tree, .. } => {
                let rows = mixed_lde_open_rows(resident, indices);
                let paths = tree.open_siblings_batch(indices);
                rows.into_iter()
                    .zip(paths)
                    .map(|(opened_values, opening_proof)| {
                        BatchOpening::new(opened_values, opening_proof)
                    })
                    .collect()
            }
        }
    }
}

impl CudaCommitMmcs<Goldilocks> for CudaMmcs {
    fn cuda_device_id(&self) -> i32 {
        self.device_id
    }

    fn retain_matrices(
        &self,
        data: &mut Self::ProverData<RowMajorMatrix<Goldilocks>>,
        retained: Vec<RowMajorMatrix<Goldilocks>>,
    ) {
        if let CudaMmcsData::Cuda { matrices, .. } = data {
            matrices
                .set(retained)
                .expect("fresh CUDA trace matrix cell");
        }
    }

    fn resident_or_upload<M: Matrix<Goldilocks>>(
        &self,
        data: &Self::ProverData<M>,
    ) -> std::sync::Arc<Vec<CudaLde>> {
        match data {
            CudaMmcsData::Cuda { resident, .. } => std::sync::Arc::clone(resident),
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
                matrices: std::sync::OnceLock::new(),
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
        if let CudaMmcsData::Cuda { matrices, .. } = &mut data {
            let _ = matrices.set(ldes);
        }
        (commitment, data)
    }
}

impl Mmcs<Goldilocks> for CudaMmcs {
    type ProverData<M> = CudaMmcsData<M>;
    type Commitment = MerkleCap<Goldilocks, [u8; 32]>;
    type Proof = Vec<[u8; 32]>;
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
            let matrices = std::sync::OnceLock::new();
            matrices.set(inputs).ok().expect("fresh CUDA matrix cell");
            return (
                commitment,
                CudaMmcsData::Cuda {
                    resident,
                    materialize: None,
                    matrices,
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
            CudaMmcsData::Cuda {
                matrices,
                materialize,
                ..
            } => matrices
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
            CudaMmcsData::Cuda { resident, .. } => resident.iter().map(CudaLde::height).collect(),
        }
    }

    fn get_max_height<M: Matrix<Goldilocks>>(&self, prover_data: &Self::ProverData<M>) -> usize {
        match prover_data {
            CudaMmcsData::Cpu(data) => self.cpu.get_max_height(data),
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
}
