pub mod builder;
pub mod prover;
pub mod system;
pub mod types;
pub mod verifier;

#[macro_export]
macro_rules! ensure {
    ($condition:expr, $err:expr) => {
        if !$condition {
            return std::result::Result::Err($err.into());
        }
    };
}
