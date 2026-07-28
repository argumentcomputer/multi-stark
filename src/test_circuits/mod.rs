//! Constraint-IR ports of the end-to-end circuit tests: preprocessed lookup
//! tables (byte operations, u32 addition), the Blake3 compression circuit with
//! its xor/add/rotate gadget tables, and a second `StarkGenericConfig` smoke
//! test (BabyBear + Poseidon2) that exercises the schoolbook coordinate-
//! expansion path a degree-2 extension never reaches.

mod baby_bear_config;
mod blake3;
mod byte_operations;
mod u32_add;
