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

/// Returns the converted error if the condition is false and logs at DEBUG.
/// After `;`, accepts `tracing::debug!` fields and an optional message.
#[macro_export]
macro_rules! ensure {
    ($condition:expr, $err:expr $(,)?) => {
        $crate::ensure!($condition, $err; "verification check failed");
    };
    ($condition:expr, $err:expr; $($context:tt)+) => {
        if !$condition {
            tracing::debug!(
                check = stringify!($condition),
                file = file!(),
                line = line!(),
                $($context)+
            );
            return std::result::Result::Err($err.into());
        }
    };
}

/// Checks equality with the same error conversion and diagnostics as [`ensure!`].
#[macro_export]
macro_rules! ensure_eq {
    ($a:expr, $b:expr, $err:expr $(,)?) => {
        $crate::ensure!(($a) == ($b), $err);
    };
    ($a:expr, $b:expr, $err:expr; $($context:tt)+) => {
        $crate::ensure!(($a) == ($b), $err; $($context)+);
    };
}
