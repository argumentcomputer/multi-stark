// SPDX-License-Identifier: MIT OR Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>

// Borrowed from the Rust plan; its host constants outlive the synchronous
// prover entry that uploads them. All sizes and groups are fixed at admission.
struct MultiStarkNttPlan {
    size_t height;
    size_t width;
    size_t extended_height;
    size_t columns;
    size_t inverse_group;
    size_t forward_group;
    size_t scratch_bytes;
    const uint64_t* shift_powers;
};

extern "C" int multi_stark_sppark_forward(int device, uint64_t* values,
                                          const MultiStarkNttPlan* plan);
extern "C" int multi_stark_sppark_coset_lde(int device, const uint64_t* trace,
                                            uint64_t* values, const MultiStarkNttPlan* plan,
                                            const uint64_t* device_shift_powers);
