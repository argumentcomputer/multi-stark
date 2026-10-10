#[cfg(feature = "kzg")]
pub mod ark_adapter;
pub mod batch;
pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use cuda::pcs as cuda_pcs;
pub mod eval;
pub mod expr;
pub mod graph;
pub mod lookup;
pub mod p3_adapter;
pub mod plonkish;
pub mod prover;
pub mod system;
#[cfg(test)]
mod test_circuits;
pub mod traits;
pub mod types;
pub mod verifier;
pub mod witness;

pub use p3_air;
pub use p3_field;
pub use p3_goldilocks;
pub use p3_matrix;

/// Compiled library features, independent of runtime backend selection or device availability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildCapabilities {
    pub kzg: bool,
    pub kzg_cuda: bool,
    pub goldilocks_cuda: bool,
    pub parallel: bool,
}

impl BuildCapabilities {
    pub const fn compiled() -> Self {
        Self {
            kzg: cfg!(feature = "kzg"),
            kzg_cuda: cfg!(feature = "kzg-cuda"),
            goldilocks_cuda: cfg!(feature = "cuda"),
            parallel: cfg!(feature = "parallel"),
        }
    }
}

#[macro_export]
macro_rules! ensure {
    ($condition:expr, $err:expr) => {
        if !$condition {
            tracing::debug!(
                "verification check failed on file {} line {}",
                file!(),
                line!()
            );
            return std::result::Result::Err($err.into());
        }
    };
}

#[macro_export]
macro_rules! ensure_eq {
    ($a:expr, $b:expr, $err:expr) => {
        $crate::ensure!(($a) == ($b), $err);
    };
}
