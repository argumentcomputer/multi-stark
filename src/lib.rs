pub mod advice;
pub mod config;
pub mod eval;
pub mod expr;
pub mod graph;
pub mod lookup;
pub mod p3_adapter;
pub mod prover;
pub mod system;
#[cfg(test)]
mod test_circuits;
pub mod types;
pub mod verifier;

pub use p3_air;
pub use p3_field;
pub use p3_goldilocks;
pub use p3_matrix;

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
