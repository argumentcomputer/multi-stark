//! Static native-field circuits with separate witness generation and STARK lowering.
//! Gadgets return handles; callers control public exposure. Witness hints add no
//! constraints. See `examples/plonkish_proof.rs` for a complete example.

mod builder;
pub mod gadgets;
mod hash;

mod stark;
mod witness;

pub use builder::{
    Bool, Circuit, CircuitBuilder, CircuitId, CircuitStats, Gate, LookupConstraint, Table,
    TableDefinition, Value, ValueSource,
};
pub use stark::{LoweringError, MultiStarkCircuit, MultiStarkLayout, TraceShards};
pub use witness::{Assignment, Witness, WitnessError};

#[cfg(test)]
mod tests;
