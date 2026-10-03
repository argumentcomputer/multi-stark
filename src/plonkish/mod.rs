//! Static native-field circuits with separate witness generation and STARK lowering.
//! Gadgets return handles; callers control public exposure. Witness hints add no
//! constraints. See `examples/plonkish_proof.rs` for a complete example.

mod builder;
pub mod gadgets;
mod hash;
mod stark;
pub mod verifier;
mod witness;

pub use builder::{
    Bool, Circuit, CircuitBuilder, CircuitStats, Gate, LookupConstraint, Table, TableDefinition,
    Value,
};
pub use stark::{LoweringError, MultiStarkCircuit, MultiStarkLayout};
pub use witness::{Assignment, Witness, WitnessError};

#[cfg(test)]
mod tests;
