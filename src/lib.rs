pub mod builder;
pub mod prover;
pub mod system;
pub mod types;
pub mod verifier;

#[macro_export]
macro_rules! ensure {
    ($condition:expr, $err:expr) => {
        if !$condition {
            eprintln!("assertion failed on file {} line {}", file!(), line!());
            return std::result::Result::Err($err.into());
        }
    };
}

#[macro_export]
macro_rules! ensure_eq {
    ($a:expr, $b:expr, $err:expr) => {
        $crate::ensure!($a == $b, $err);
    };
}
