pub mod builder;
mod chips;
pub mod lookup;
pub mod prover;
pub mod system;
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

#[macro_export]
macro_rules! benchmark {
    ($bench:expr, $msg:expr $(,$msg_args:expr)*) => {{
        let now = std::time::Instant::now();
        let result = std::hint::black_box($bench);
        print!($msg $(,$msg_args)*);
        println!("{:?}", now.elapsed());
        result
    }};
}
