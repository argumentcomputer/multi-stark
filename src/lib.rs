pub mod proof;
pub mod system;
pub mod types;

#[macro_export]
macro_rules! ensure {
    ($condition:expr, $err:expr) => {
        if !$condition {
            return core::result::Result::Err($err.into());
        }
    };
}
