pub mod prover;
pub mod system;
pub mod types;
pub mod verifier;

#[macro_export]
macro_rules! ensure {
    ($condition:expr, $err:expr) => {
        if !$condition {
            return core::result::Result::Err($err.into());
        }
    };
}
