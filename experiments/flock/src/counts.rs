#[cfg(feature = "counters")]
pub use flock_prover::field::gf2_128::op_count::{measure, reset, snapshot};

#[cfg(not(feature = "counters"))]
mod disabled {
    pub struct Snapshot {
        pub invs: Option<u64>,
    }
    impl Snapshot {
        pub fn muls_excluding_inv(&self) -> Option<u64> {
            None
        }
    }
    pub fn reset() {}
    pub fn snapshot() -> Snapshot {
        Snapshot { invs: None }
    }
    pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Snapshot) {
        (f(), snapshot())
    }
}
#[cfg(not(feature = "counters"))]
pub use disabled::{measure, reset, snapshot};
