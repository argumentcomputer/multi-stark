//! CUDA-backed primitives for multi-stark.
//!
//! Phase 1: routes [`p3_dft::TwoAdicSubgroupDft`] through sppark's batched
//! Goldilocks NTT/LDE kernels via [`CudaRadix2Dft`]. Merkle hashing remains
//! on CPU until the dedicated Goldilocks-Poseidon2 kernel ships.

mod dft;
mod ffi;
pub mod mmcs;
mod poseidon2;
#[cfg(test)]
mod tests;

pub use dft::CudaRadix2Dft;
pub use poseidon2::{permute_8_gpu, permute_16_gpu};
