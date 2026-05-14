//! multi-plonk: KZG-based variant of multi-stark.
//!
//! Mirrors the structure of multi-stark but uses BLS12-381 / SonicKZG10
//! polynomial commitments instead of FRI over Goldilocks. Trace polynomials
//! are committed in monomial basis after iFFT; constraints are evaluated on a
//! coset of size `next_pow2(max_constraint_degree) * trace_size` to allow
//! division by the vanishing polynomial. The quotient is committed as a single
//! polynomial (no chunking).
//!
//! No extension field is used: the BLS12-381 scalar field Fr is large enough
//! (~256 bits) that single-field Schwartz-Zippel bounds are already negligible.

pub mod air;
pub mod builder;
pub mod lookup;
pub mod matrix;
pub mod prover;
pub mod system;
pub mod types;
pub mod verifier;

#[macro_export]
macro_rules! ensure {
    ($condition:expr, $err:expr) => {
        if !$condition {
            return std::result::Result::Err($err);
        }
    };
}

#[macro_export]
macro_rules! ensure_eq {
    ($a:expr, $b:expr, $err:expr) => {
        $crate::ensure!(($a) == ($b), $err);
    };
}
