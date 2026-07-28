//! Blake3 compression proved against the constraint-IR API.
//!
//! This is a port of the original AIR-based Blake3 circuit test to the new
//! frontend where circuits are described declaratively with [`CircuitInputs`]
//! (main/preprocessed widths, polynomial constraints, and lookup pushes/pulls)
//! rather than by implementing the `Air` trait. The statement proved is
//! unchanged, and so is the multi-circuit decomposition:
//!
//! - a **U8 xor / pair range-check** table backed by a preprocessed
//!   `[A, B, A xor B]` matrix over all 65536 byte pairs (two channels: xor and
//!   a pair range-check that only looks at `(A, B)`);
//! - **u32 xor**, **u32 add**, and four fixed **right-rotate** gadgets
//!   (`>>8`, `>>16`, `>>12`, `>>7`), each decomposing its 32-bit operands into
//!   bytes and delegating byte-level correctness to the U8 table via lookups;
//! - a **G function** circuit that chains add / xor / rotate through lookups to
//!   the gadgets above; and
//! - a **compression** circuit that pushes 56 G-function rounds and the final
//!   feed-forward xors, and constrains the message permutation between rounds.
//!
//! Every table exposes its rows as lookup PULLs (negated multiplicity) and the
//! higher-level circuits PUSH the tuples they need; the leading argument of
//! each lookup is a per-operation channel constant (`Blake3CompressionCircuit`
//! discriminants) so pushes and pulls only match within the same table. The
//! external `prove_multiple_claims` claims are the final PUSH that balances the
//! whole global accumulator.
//!
//! Two rotates (`>>12`, `>>7`) reconstruct the word via a div/rem
//! decomposition instead of a byte re-permutation, so they carry degree-2
//! polynomial constraints; the div/rem bytes are (as in the original) *not*
//! range-checked, so those circuits remain underconstrained on their own and
//! rely on the surrounding claims being honestly generated.

use crate::expr::Expr;
use crate::lookup::Lookup;
use crate::system::{CircuitInputs, ProverKey, System, SystemWitness};
use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use std::array;
use std::ops::Range;

struct CompressionInfo {
    cv: [u32; 8],
    block_words: [u32; 16],
    counter_low: u32,
    counter_high: u32,
    block_len: u32,
    flags: u32,
    output: [u32; 16],
}

// Blake3 reference hasher that additionally produces compression IO for claims construction.
// Tested to be compatible with: https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::needless_range_loop
)]
fn blake3_new_update_finalize(input: &[u8]) -> (Vec<CompressionInfo>, [u8; 32]) {
    const IV: [u32; 8] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
        0x5BE0CD19,
    ];
    const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
    const CHUNK_LEN: usize = 1024;
    const BLOCK_LEN: usize = 64;

    const CHUNK_START: u32 = 1 << 0;
    const CHUNK_END: u32 = 1 << 1;
    const PARENT: u32 = 1 << 2;
    const OUT_LEN: usize = 32;
    const ROOT: u32 = 1 << 3;

    fn compress(
        chaining_value: &[u32; 8],
        block_words: &[u32; 16],
        counter: u64,
        block_len: u32,
        flags: u32,
    ) -> [u32; 16] {
        let counter_low = counter as u32;
        let counter_high = (counter >> 32) as u32;

        #[rustfmt::skip]
        let mut state = [
            chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
            chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
            IV[0],             IV[1],             IV[2],             IV[3],
            counter_low,       counter_high,      block_len,         flags,
            block_words[0], block_words[1], block_words[2], block_words[3],
            block_words[4], block_words[5], block_words[6], block_words[7],
            block_words[8], block_words[9], block_words[10], block_words[11],
            block_words[12], block_words[13], block_words[14], block_words[15],
        ];

        let a = [0, 1, 2, 3, 0, 1, 2, 3];
        let b = [4, 5, 6, 7, 5, 6, 7, 4];
        let c = [8, 9, 10, 11, 10, 11, 8, 9];
        let d = [12, 13, 14, 15, 15, 12, 13, 14];
        let mx = [16, 18, 20, 22, 24, 26, 28, 30];
        let my = [17, 19, 21, 23, 25, 27, 29, 31];

        for round_idx in 0..7 {
            for j in 0..8 {
                let a_in = state[a[j]];
                let b_in = state[b[j]];
                let c_in = state[c[j]];
                let d_in = state[d[j]];
                let mx_in = state[mx[j]];
                let my_in = state[my[j]];

                let a_0 = a_in.wrapping_add(b_in).wrapping_add(mx_in);
                let d_0 = (d_in ^ a_0).rotate_right(16);
                let c_0 = c_in.wrapping_add(d_0);
                let b_0 = (b_in ^ c_0).rotate_right(12);

                let a_1 = a_0.wrapping_add(b_0).wrapping_add(my_in);
                let d_1 = (d_0 ^ a_1).rotate_right(8);
                let c_1 = c_0.wrapping_add(d_1);
                let b_1 = (b_0 ^ c_1).rotate_right(7);

                state[a[j]] = a_1;
                state[b[j]] = b_1;
                state[c[j]] = c_1;
                state[d[j]] = d_1;
            }

            if round_idx < 6 {
                let mut permuted = [0; 16];
                for i in 0..16 {
                    permuted[i] = state[16 + MSG_PERMUTATION[i]];
                }
                state[16..(16 + 16)].copy_from_slice(&permuted);
            }
        }

        for i in 0..8 {
            state[i] ^= state[i + 8];
            state[i + 8] ^= chaining_value[i];
        }

        array::from_fn(|i| state[i])
    }

    fn words_from_little_endian_bytes(bytes: &[u8], words: &mut [u32]) {
        debug_assert_eq!(bytes.len(), 4 * words.len());
        for (four_bytes, word) in bytes.chunks_exact(4).zip(words) {
            *word = u32::from_le_bytes(four_bytes.try_into().unwrap());
        }
    }

    fn first_8_words(compression_output: [u32; 16]) -> [u32; 8] {
        compression_output[0..8].try_into().unwrap()
    }

    fn start_flag(blocks_compressed: u8) -> u32 {
        if blocks_compressed == 0 {
            CHUNK_START
        } else {
            0
        }
    }

    let mut c_info = vec![];
    let mut input = input;
    let mut output = [0u8; 32];

    let hasher_key_words = IV;
    let mut hasher_cv_stack = [[0u32; 8]; 54];
    let mut hasher_cv_stack_len = 0u32;
    let hasher_flags = 0u32;

    let mut chunk_state_chaining_value = hasher_key_words;
    let mut chunk_state_chunk_counter = 0u64;
    let mut chunk_state_block = [0u8; BLOCK_LEN];
    let mut chunk_state_block_len = 0u8;
    let mut chunk_state_blocks_compressed = 0u8;
    let mut chunk_state_flags = hasher_flags;

    while !input.is_empty() {
        let chunk_state_len =
            BLOCK_LEN * chunk_state_blocks_compressed as usize + chunk_state_block_len as usize;
        if CHUNK_LEN == chunk_state_len {
            let mut block_words = [0; 16];
            words_from_little_endian_bytes(&chunk_state_block, &mut block_words);
            let chaining_value = chunk_state_chaining_value;
            let counter = chunk_state_chunk_counter;
            let block_len = chunk_state_block_len;
            let flags = chunk_state_flags | start_flag(chunk_state_blocks_compressed) | CHUNK_END;

            let cv = compress(
                &chaining_value,
                &block_words,
                counter,
                block_len as u32,
                flags,
            );
            c_info.push(CompressionInfo {
                cv: chaining_value,
                block_words,
                counter_low: counter as u32,
                counter_high: (counter >> 32) as u32,
                block_len: block_len as u32,
                flags,
                output: cv,
            });

            let chaining_value = first_8_words(cv);
            let chunk_cv = chaining_value;
            let total_chunks = chunk_state_chunk_counter + 1;

            let mut new_cv = chunk_cv;
            let mut total_chunks_inner = total_chunks;
            while total_chunks_inner & 1 == 0 {
                hasher_cv_stack_len -= 1;
                let pop_stack = hasher_cv_stack[hasher_cv_stack_len as usize];

                let left_child_cv = pop_stack;
                let right_child_cv = new_cv;

                let mut block_words = [0u32; 16];
                block_words[..8].copy_from_slice(&left_child_cv);
                block_words[8..].copy_from_slice(&right_child_cv);

                let input_chaining_value = hasher_key_words;
                let counter = 0u64;
                let block_len = BLOCK_LEN as u32;
                let flags = PARENT | hasher_flags;

                let cv = compress(
                    &input_chaining_value,
                    &block_words,
                    counter,
                    block_len,
                    flags,
                );
                c_info.push(CompressionInfo {
                    cv: input_chaining_value,
                    block_words,
                    counter_low: counter as u32,
                    counter_high: (counter >> 32) as u32,
                    block_len,
                    flags,
                    output: cv,
                });

                new_cv = first_8_words(cv);
                total_chunks_inner >>= 1;
            }

            hasher_cv_stack[hasher_cv_stack_len as usize] = new_cv;
            hasher_cv_stack_len += 1;

            chunk_state_chaining_value = hasher_key_words;
            chunk_state_chunk_counter = total_chunks;
            chunk_state_block = [0u8; BLOCK_LEN];
            chunk_state_block_len = 0u8;
            chunk_state_blocks_compressed = 0u8;
            chunk_state_flags = hasher_flags;
        }

        let chunk_state_len =
            BLOCK_LEN * chunk_state_blocks_compressed as usize + chunk_state_block_len as usize;
        let want = CHUNK_LEN - chunk_state_len;
        let take = std::cmp::min(want, input.len());

        let mut input_inner = &input[..take];

        while !input_inner.is_empty() {
            if chunk_state_block_len as usize == BLOCK_LEN {
                let mut block_words = [0; 16];
                words_from_little_endian_bytes(&chunk_state_block, &mut block_words);

                let cv = compress(
                    &chunk_state_chaining_value,
                    &block_words,
                    chunk_state_chunk_counter,
                    BLOCK_LEN as u32,
                    chunk_state_flags | start_flag(chunk_state_blocks_compressed),
                );
                c_info.push(CompressionInfo {
                    cv: chunk_state_chaining_value,
                    block_words,
                    counter_low: chunk_state_chunk_counter as u32,
                    counter_high: (chunk_state_chunk_counter >> 32) as u32,
                    block_len: BLOCK_LEN as u32,
                    flags: chunk_state_flags | start_flag(chunk_state_blocks_compressed),
                    output: cv,
                });

                chunk_state_chaining_value = first_8_words(cv);
                chunk_state_blocks_compressed += 1;
                chunk_state_block = [0u8; BLOCK_LEN];
                chunk_state_block_len = 0;
            }

            let want = BLOCK_LEN - chunk_state_block_len as usize;
            let take = std::cmp::min(want, input_inner.len());
            chunk_state_block[chunk_state_block_len as usize..][..take]
                .copy_from_slice(&input_inner[..take]);
            chunk_state_block_len += take as u8;
            input_inner = &input_inner[take..];
        }

        input = &input[take..];
    }

    let mut block_words = [0; 16];
    words_from_little_endian_bytes(&chunk_state_block, &mut block_words);
    let mut input_chaining_value = chunk_state_chaining_value;
    let mut counter = chunk_state_chunk_counter;
    let mut block_len = chunk_state_block_len as u32;
    let mut flags = chunk_state_flags | start_flag(chunk_state_blocks_compressed) | CHUNK_END;

    let mut parent_nodes_remaining = hasher_cv_stack_len as usize;
    while parent_nodes_remaining > 0 {
        parent_nodes_remaining -= 1;

        let left_child_cv = hasher_cv_stack[parent_nodes_remaining];

        let cv = compress(
            &input_chaining_value,
            &block_words,
            counter,
            block_len,
            flags,
        );
        c_info.push(CompressionInfo {
            cv: input_chaining_value,
            block_words,
            counter_low: counter as u32,
            counter_high: (counter >> 32) as u32,
            block_len,
            flags,
            output: cv,
        });

        let right_child_cv = first_8_words(cv);

        let mut block_words_inner = [0; 16];
        block_words_inner[..8].copy_from_slice(&left_child_cv);
        block_words_inner[8..].copy_from_slice(&right_child_cv);

        input_chaining_value = hasher_key_words;
        block_words = block_words_inner;
        counter = 0;
        block_len = BLOCK_LEN as u32;
        flags = PARENT | hasher_flags;
    }

    for (output_block_counter, out_block) in output.chunks_mut(2 * OUT_LEN).enumerate() {
        let output_block_counter = output_block_counter as u64;
        let cv = compress(
            &input_chaining_value,
            &block_words,
            output_block_counter,
            block_len,
            flags | ROOT,
        );
        c_info.push(CompressionInfo {
            cv: input_chaining_value,
            block_words,
            counter_low: output_block_counter as u32,
            counter_high: (output_block_counter >> 32) as u32,
            block_len,
            flags: flags | ROOT,
            output: cv,
        });

        let words = cv;
        for (word, out_word) in words.iter().zip(out_block.chunks_mut(4)) {
            out_word.copy_from_slice(&word.to_le_bytes()[..out_word.len()]);
        }
    }

    (c_info, output)
}

// Blake3-specific constants

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const A: [usize; 8] = [0, 1, 2, 3, 0, 1, 2, 3];
const B: [usize; 8] = [4, 5, 6, 7, 5, 6, 7, 4];
const C: [usize; 8] = [8, 9, 10, 11, 10, 11, 8, 9];
const D: [usize; 8] = [12, 13, 14, 15, 15, 12, 13, 14];
const MX: [usize; 8] = [16, 18, 20, 22, 24, 26, 28, 30];
const MY: [usize; 8] = [17, 19, 21, 23, 25, 27, 29, 31];

// Circuit constants

const BYTE_VALUES_NUM: usize = 256;

// Preprocessed columns are: [A, B, A xor B], where A and B are bytes
const PREPROCESSED_TRACE_WIDTH: usize = 3;

// Main trace consists of multiplicities for 'xor' and 'range_check' operations
const U8_XOR_PAIR_RANGE_CHECK_TRACE_WIDTH: usize = 2;

// multiplicity, a0, a1, a2, a3, b0, b1, b2, b3, a0^b0, a1^b1, a2^b2, a3^b3
const U32_XOR_TRACE_WIDTH: usize = 13;

// a0, a1, a2, a3, b0, b1, b2, b3, z0, z1, z2, z3, carry, multiplicity
const U32_ADD_TRACE_WIDTH: usize = 14;

// multiplicity, a0, a1, a2, a3, rot0, rot1, rot2, rot3
const U32_RIGHT_ROTATE_8_TRACE_WIDTH: usize = 9;
const U32_RIGHT_ROTATE_16_TRACE_WIDTH: usize = U32_RIGHT_ROTATE_8_TRACE_WIDTH;

// multiplicity,
// a0, a1, a2, a3,
// rot0, rot1, rot2, rot3,
// two_pow_k_0, two_pow_k_1, two_pow_k_2, two_pow_k_3,
// two_pow_32_minus_k_0, two_pow_32_minus_k_1, two_pow_32_minus_k_2, two_pow_32_minus_k_3,
// value_div_0, value_div_1, value_div_2, value_div_3,
// value_rem_0, value_rem_1, value_rem_2, value_rem_3
const U32_RIGHT_ROTATE_7_TRACE_WIDTH: usize = 25;
const U32_RIGHT_ROTATE_12_TRACE_WIDTH: usize = U32_RIGHT_ROTATE_7_TRACE_WIDTH;

// Totally 81 byte columns:
// multiplicity, a_in(4), b_in(4), c_in(4), d_in(4), mx_in(4), my_in(4),
// a_0_tmp(4), a_0(4), d_0_tmp(4), d_0(4), c_0(4), b_0_tmp(4), b_0(4),
// a_1_tmp(4), a_1(4), d_1_tmp(4), d_1(4), c_1(4), b_1_tmp(4), b_1(4)
const G_FUNCTION_TRACE_WIDTH: usize = 81;

// multiplicity,
// [state_in_0 (4), ... state_in_31 (4),
// [a_in (4), b_in (4), c_in (4), d_in (4), mx_in (4), my_in (4), a_1 (4), b_1 (4), c_1 (4), d_1 (4)] (x56),
// [state_i (4), state_i_8 (4), i_i8_xor (4)] (x8),
// [state_i_8 (4), chaining_value_i (4), i_cv_xor (4)] (x8),
// state_out_0 (4), ... state_out_16 (4)
//
// 1 + 32 * 4 + 40 * 56 + 12 * 8 * 2 + 16 * 4
const COMPRESSION_TRACE_WIDTH: usize = 2625;

// ---------------------------------------------------------------------------
// Constraint-IR helpers.
//
// The old AIR frontend addressed columns with `var(i)` / `preprocessed_var(i)`
// and built lookups with `Lookup::push` / `Lookup::pull`. The constraint-IR
// frontend uses `Expr::main(i)` / `Expr::preprocessed(i)` leaves and encodes a
// PULL as a negated multiplicity. These thin wrappers keep the port a
// (near-)mechanical translation of the original.
// ---------------------------------------------------------------------------

/// Main-trace column `i` as a base-field expression.
fn var(i: usize) -> Expr<Val> {
    Expr::main(u32::try_from(i).expect("main column index fits u32"))
}

/// Preprocessed-trace column `i` as a base-field expression.
fn preprocessed_var(i: usize) -> Expr<Val> {
    Expr::preprocessed(u32::try_from(i).expect("preprocessed column index fits u32"))
}

fn c_usize(value: usize) -> Expr<Val> {
    Expr::constant(Val::from_usize(value))
}

fn c_u32(value: u32) -> Expr<Val> {
    Expr::constant(Val::from_u32(value))
}

fn one() -> Expr<Val> {
    Expr::constant(Val::ONE)
}

fn zero() -> Expr<Val> {
    Expr::constant(Val::ZERO)
}

/// A PUSH lookup (positive multiplicity).
fn push(multiplicity: Expr<Val>, args: Vec<Expr<Val>>) -> Lookup<Expr<Val>> {
    Lookup { multiplicity, args }
}

/// A PULL lookup (negated multiplicity).
fn pull(multiplicity: Expr<Val>, args: Vec<Expr<Val>>) -> Lookup<Expr<Val>> {
    Lookup {
        multiplicity: -multiplicity,
        args,
    }
}

/// Pull the `[circuit_idx, state_in(32), state_out(16)]` tuple for the
/// compression circuit, reconstructing each u32 word from four byte columns.
fn pull_state_in_state_out(
    multiplicity: Expr<Val>,
    circuit_idx: usize,
    state_in_range: Range<usize>,
    state_out_range: Range<usize>,
) -> Lookup<Expr<Val>> {
    assert_eq!(state_in_range.len(), 128);
    assert_eq!(state_out_range.len(), 64);

    let in_i = state_in_range.collect::<Vec<usize>>();
    let out_i = state_out_range.collect::<Vec<usize>>();

    let state_in = in_i
        .chunks(4)
        .map(|i| {
            var(i[0])
                + var(i[1]) * c_u32(256)
                + var(i[2]) * c_u32(65536)
                + var(i[3]) * c_u32(16777216)
        })
        .collect::<Vec<Expr<Val>>>();

    let state_out = out_i
        .chunks(4)
        .map(|i| {
            var(i[0])
                + var(i[1]) * c_u32(256)
                + var(i[2]) * c_u32(65536)
                + var(i[3]) * c_u32(16777216)
        })
        .collect::<Vec<Expr<Val>>>();

    pull(
        multiplicity,
        [vec![c_usize(circuit_idx)], state_in, state_out].concat(),
    )
}

/// Push the 10-word tuple consumed by one G-function round.
fn push_round(
    multiplicity: Expr<Val>,
    circuit_idx: usize,
    v_ind: Range<usize>,
) -> Lookup<Expr<Val>> {
    assert_eq!(v_ind.len(), 40);

    let i = v_ind.collect::<Vec<usize>>();

    push(
        multiplicity,
        vec![
            c_usize(circuit_idx),
            var(i[0])
                + var(i[1]) * c_u32(256)
                + var(i[2]) * c_u32(65536)
                + var(i[3]) * c_u32(16777216),
            var(i[4])
                + var(i[5]) * c_u32(256)
                + var(i[6]) * c_u32(65536)
                + var(i[7]) * c_u32(16777216),
            var(i[8])
                + var(i[9]) * c_u32(256)
                + var(i[10]) * c_u32(65536)
                + var(i[11]) * c_u32(16777216),
            var(i[12])
                + var(i[13]) * c_u32(256)
                + var(i[14]) * c_u32(65536)
                + var(i[15]) * c_u32(16777216),
            var(i[16])
                + var(i[17]) * c_u32(256)
                + var(i[18]) * c_u32(65536)
                + var(i[19]) * c_u32(16777216),
            var(i[20])
                + var(i[21]) * c_u32(256)
                + var(i[22]) * c_u32(65536)
                + var(i[23]) * c_u32(16777216),
            var(i[24])
                + var(i[25]) * c_u32(256)
                + var(i[26]) * c_u32(65536)
                + var(i[27]) * c_u32(16777216),
            var(i[28])
                + var(i[29]) * c_u32(256)
                + var(i[30]) * c_u32(65536)
                + var(i[31]) * c_u32(16777216),
            var(i[32])
                + var(i[33]) * c_u32(256)
                + var(i[34]) * c_u32(65536)
                + var(i[35]) * c_u32(16777216),
            var(i[36])
                + var(i[37]) * c_u32(256)
                + var(i[38]) * c_u32(65536)
                + var(i[39]) * c_u32(16777216),
        ],
    )
}

/// A lookup constructor (`push` or `pull`), taken as a function pointer so the
/// u32-word helper below can build either polarity from one code path.
type LookupFn = fn(Expr<Val>, Vec<Expr<Val>>) -> Lookup<Expr<Val>>;

fn lookup_u32_inner(
    lookup_fn: LookupFn,
    multiplicity: Expr<Val>,
    circuit_idx: usize,
    v_ind: Range<usize>,
) -> Lookup<Expr<Val>> {
    assert_eq!(v_ind.len(), 12);

    let i = v_ind.collect::<Vec<usize>>();

    lookup_fn(
        multiplicity,
        vec![
            c_usize(circuit_idx),
            var(i[0])
                + var(i[1]) * c_u32(256)
                + var(i[2]) * c_u32(256 * 256)
                + var(i[3]) * c_u32(256 * 256 * 256),
            var(i[4])
                + var(i[5]) * c_u32(256)
                + var(i[6]) * c_u32(256 * 256)
                + var(i[7]) * c_u32(256 * 256 * 256),
            var(i[8])
                + var(i[9]) * c_u32(256)
                + var(i[10]) * c_u32(256 * 256)
                + var(i[11]) * c_u32(256 * 256 * 256),
        ],
    )
}

fn push_u32(multiplicity: Expr<Val>, circuit_idx: usize, v_ind: Range<usize>) -> Lookup<Expr<Val>> {
    lookup_u32_inner(push, multiplicity, circuit_idx, v_ind)
}

fn pull_u32(multiplicity: Expr<Val>, circuit_idx: usize, v_ind: Range<usize>) -> Lookup<Expr<Val>> {
    lookup_u32_inner(pull, multiplicity, circuit_idx, v_ind)
}

enum Blake3CompressionCircuit {
    U8Xor,
    U32Xor,
    U32Add,
    U32RightRotate8,
    U32RightRotate16,
    U32RightRotate12, // FIXME: currently underconstrained (range check is not performed).
    U32RightRotate7,  // FIXME: currently underconstrained (range check is not performed).
    U8PairRangeCheck,
    GFunction,
    Compression,
}

impl Blake3CompressionCircuit {
    /// The lookup channel discriminant. It doubles as the leading argument of
    /// every push/pull, so tables only match within themselves. Note that
    /// `U8Xor` and `U8PairRangeCheck` are two *channels* served by a single
    /// preprocessed circuit (see [`Self::circuit_inputs`]).
    fn position(&self) -> usize {
        match self {
            Self::U8Xor => 0,
            Self::U32Xor => 1,
            Self::U32Add => 2,
            Self::U32RightRotate8 => 3,
            Self::U32RightRotate16 => 4,
            Self::U32RightRotate12 => 5,
            Self::U32RightRotate7 => 6,
            Self::U8PairRangeCheck => 7,
            Self::GFunction => 8,
            Self::Compression => 9,
        }
    }

    fn width(&self) -> usize {
        match self {
            Self::U8Xor | Self::U8PairRangeCheck => U8_XOR_PAIR_RANGE_CHECK_TRACE_WIDTH,
            Self::U32Xor => U32_XOR_TRACE_WIDTH,
            Self::U32Add => U32_ADD_TRACE_WIDTH,
            Self::U32RightRotate8 => U32_RIGHT_ROTATE_8_TRACE_WIDTH,
            Self::U32RightRotate16 => U32_RIGHT_ROTATE_16_TRACE_WIDTH,
            Self::U32RightRotate12 => U32_RIGHT_ROTATE_12_TRACE_WIDTH,
            Self::U32RightRotate7 => U32_RIGHT_ROTATE_7_TRACE_WIDTH,
            Self::GFunction => G_FUNCTION_TRACE_WIDTH,
            Self::Compression => COMPRESSION_TRACE_WIDTH,
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val>> {
        match self {
            Self::U8Xor | Self::U8PairRangeCheck => {
                let bytes: [u8; BYTE_VALUES_NUM] = array::from_fn(|idx| u8::try_from(idx).unwrap());
                let mut trace_values = Vec::with_capacity(
                    BYTE_VALUES_NUM * BYTE_VALUES_NUM * PREPROCESSED_TRACE_WIDTH,
                );
                for i in 0..BYTE_VALUES_NUM {
                    for j in 0..BYTE_VALUES_NUM {
                        trace_values.push(Val::from_u8(bytes[i]));
                        trace_values.push(Val::from_u8(bytes[j]));
                        trace_values.push(Val::from_u8(bytes[i] ^ bytes[j]));
                    }
                }
                Some(RowMajorMatrix::new(trace_values, PREPROCESSED_TRACE_WIDTH))
            }
            Self::U32Xor
            | Self::U32Add
            | Self::U32RightRotate8
            | Self::U32RightRotate16
            | Self::U32RightRotate12
            | Self::U32RightRotate7
            | Self::GFunction
            | Self::Compression => None,
        }
    }

    /// Base-field polynomial constraints (each must vanish on every row).
    /// `assert_eq(a, b)` becomes `a - b`, `assert_bool(x)` becomes
    /// `x * (x - 1)`.
    #[allow(clippy::too_many_lines)]
    fn constraints(&self) -> Vec<Expr<Val>> {
        match self {
            Self::U8Xor
            | Self::U8PairRangeCheck
            | Self::U32Xor
            | Self::U32RightRotate8
            | Self::U32RightRotate16
            | Self::GFunction => vec![],
            Self::U32Add => {
                let x: Vec<Expr<Val>> = (0..4).map(var).collect();
                let y: Vec<Expr<Val>> = (4..8).map(var).collect();
                let z: Vec<Expr<Val>> = (8..12).map(var).collect();
                let carry = var(12);

                let expr1 = x[0].clone()
                    + x[1].clone() * c_u32(256)
                    + x[2].clone() * c_u32(256 * 256)
                    + x[3].clone() * c_u32(256 * 256 * 256)
                    + y[0].clone()
                    + y[1].clone() * c_u32(256)
                    + y[2].clone() * c_u32(256 * 256)
                    + y[3].clone() * c_u32(256 * 256 * 256);
                let expr2 = z[0].clone()
                    + z[1].clone() * c_u32(256)
                    + z[2].clone() * c_u32(256 * 256)
                    + z[3].clone() * c_u32(256 * 256 * 256)
                    + carry.clone() * Expr::constant(Val::from_u64(256 * 256 * 256 * 256));

                vec![
                    // the carry must be a boolean
                    carry.clone() * (carry - one()),
                    // x + y == z + carry * 2^32
                    expr1 - expr2,
                ]
            }
            Self::U32RightRotate12 | Self::U32RightRotate7 => {
                let input = var(1)
                    + var(2) * c_u32(256)
                    + var(3) * c_u32(256 * 256)
                    + var(4) * c_u32(256 * 256 * 256);
                let output = var(5)
                    + var(6) * c_u32(256)
                    + var(7) * c_u32(256 * 256)
                    + var(8) * c_u32(256 * 256 * 256);
                let two_pow_k = var(9)
                    + var(10) * c_u32(256)
                    + var(11) * c_u32(256 * 256)
                    + var(12) * c_u32(256 * 256 * 256);
                let two_pow_32_minus_k = var(13)
                    + var(14) * c_u32(256)
                    + var(15) * c_u32(256 * 256)
                    + var(16) * c_u32(256 * 256 * 256);
                let input_div = var(17)
                    + var(18) * c_u32(256)
                    + var(19) * c_u32(256 * 256)
                    + var(20) * c_u32(256 * 256 * 256);
                let input_rem = var(21)
                    + var(22) * c_u32(256)
                    + var(23) * c_u32(256 * 256)
                    + var(24) * c_u32(256 * 256 * 256);

                vec![
                    input - (input_div.clone() * two_pow_k + input_rem.clone()),
                    output - (input_div + input_rem * two_pow_32_minus_k),
                ]
            }
            Self::Compression => {
                let mut constraints = vec![];

                // A u32 word reconstructed from four consecutive byte columns.
                let w = |o: usize| {
                    var(o)
                        + var(o + 1) * c_u32(256)
                        + var(o + 2) * c_u32(65536)
                        + var(o + 3) * c_u32(16777216)
                };

                // state: 32 words from columns 1..=128.
                let mut state: Vec<Expr<Val>> = (0..32).map(|k| w(1 + 4 * k)).collect();
                let mut offset = 129usize;

                let mut a_in = vec![];
                let mut b_in = vec![];
                let mut c_in = vec![];
                let mut d_in = vec![];
                let mut mx_in = vec![];
                let mut my_in = vec![];
                let mut a_1 = vec![];
                let mut b_1 = vec![];
                let mut c_1 = vec![];
                let mut d_1 = vec![];

                for _ in 0..56 {
                    a_in.push(w(offset));
                    offset += 4;
                    b_in.push(w(offset));
                    offset += 4;
                    c_in.push(w(offset));
                    offset += 4;
                    d_in.push(w(offset));
                    offset += 4;
                    mx_in.push(w(offset));
                    offset += 4;
                    my_in.push(w(offset));
                    offset += 4;
                    a_1.push(w(offset));
                    offset += 4;
                    d_1.push(w(offset));
                    offset += 4;
                    c_1.push(w(offset));
                    offset += 4;
                    b_1.push(w(offset));
                    offset += 4;
                }

                let mut state_i = vec![];
                let mut state_i_8 = vec![];
                let mut i_i8_xor = vec![];
                let mut state_i_8_copy = vec![];
                let mut chaining_values = vec![];
                let mut i_cv_xor = vec![];
                let chaining_values_expected = state[0..8].to_vec();

                for _ in 0..8 {
                    state_i.push(w(offset));
                    offset += 4;
                    state_i_8.push(w(offset));
                    offset += 4;
                    i_i8_xor.push(w(offset));
                    offset += 4;
                    state_i_8_copy.push(w(offset));
                    offset += 4;
                    chaining_values.push(w(offset));
                    offset += 4;
                    i_cv_xor.push(w(offset));
                    offset += 4;
                }

                let mut state_out = vec![];
                for _ in 0..16 {
                    state_out.push(w(offset));
                    offset += 4;
                }

                // check state_in <-> temp variables relation
                let mut offset_2 = 0usize;
                for round_idx in 0..7 {
                    for j in 0..8 {
                        constraints.push(state[A[j]].clone() - a_in[offset_2].clone());
                        constraints.push(state[B[j]].clone() - b_in[offset_2].clone());
                        constraints.push(state[C[j]].clone() - c_in[offset_2].clone());
                        constraints.push(state[D[j]].clone() - d_in[offset_2].clone());
                        constraints.push(state[MX[j]].clone() - mx_in[offset_2].clone());
                        constraints.push(state[MY[j]].clone() - my_in[offset_2].clone());

                        state[A[j]] = a_1[offset_2].clone();
                        state[B[j]] = b_1[offset_2].clone();
                        state[C[j]] = c_1[offset_2].clone();
                        state[D[j]] = d_1[offset_2].clone();

                        offset_2 += 1;
                    }
                    if round_idx < 6 {
                        let mut permuted: [Expr<Val>; 16] = array::from_fn(|_| zero());
                        for i in 0..16 {
                            permuted[i] = state[16 + MSG_PERMUTATION[i]].clone();
                        }
                        state[16..(16 + 16)].clone_from_slice(&permuted);
                    }
                }

                // check state_out <-> XOR variables relation
                for i in 0..8 {
                    constraints.push(state[i].clone() - state_i[i].clone());
                    constraints.push(state[i + 8].clone() - state_i_8[i].clone());
                    constraints.push(i_i8_xor[i].clone() - state_out[i].clone());

                    constraints.push(state[i + 8].clone() - state_i_8_copy[i].clone());
                    constraints
                        .push(chaining_values_expected[i].clone() - chaining_values[i].clone());
                    constraints.push(i_cv_xor[i].clone() - state_out[i + 8].clone());
                }

                constraints
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn lookups(&self) -> Vec<Lookup<Expr<Val>>> {
        let u8_xor_idx = Self::U8Xor.position();
        let u32_xor_idx = Self::U32Xor.position();
        let u32_add_idx = Self::U32Add.position();
        let u32_right_rotate_8_idx = Self::U32RightRotate8.position();
        let u32_right_rotate_16_idx = Self::U32RightRotate16.position();
        let u32_right_rotate_12_idx = Self::U32RightRotate12.position();
        let u32_right_rotate_7_idx = Self::U32RightRotate7.position();
        let u8_pair_range_check_idx = Self::U8PairRangeCheck.position();
        let g_function_idx = Self::GFunction.position();
        let compression_idx = Self::Compression.position();

        match self {
            Self::U8Xor | Self::U8PairRangeCheck => {
                vec![
                    pull(
                        var(0),
                        vec![
                            c_usize(u8_xor_idx),
                            preprocessed_var(0),
                            preprocessed_var(1),
                            preprocessed_var(2),
                        ],
                    ),
                    pull(
                        var(1),
                        vec![
                            c_usize(u8_pair_range_check_idx),
                            preprocessed_var(0),
                            preprocessed_var(1),
                        ],
                    ),
                ]
            }

            // (4 push lookups to u8_xor circuit)
            Self::U32Xor => {
                let mut lookups = vec![pull_u32(var(0), u32_xor_idx, 1..12 + 1)];

                // push (A, B, A^B) tuples to U8Xor circuit for verification
                lookups.extend((0..4).map(|i| {
                    push(
                        one(),
                        vec![c_usize(u8_xor_idx), var(i + 1), var(i + 5), var(i + 9)],
                    )
                }));
                lookups
            }

            // (8 push lookups to pair_range_check)
            Self::U32Add => {
                // Pull
                let mut lookups = vec![pull_u32(var(13), u32_add_idx, 0..11 + 1)];

                // push (A, B) tuples to U8PairRangeCheck circuit for verification
                lookups.extend((0..4).map(|i| {
                    push(
                        one(),
                        vec![c_usize(u8_pair_range_check_idx), var(i), var(i + 4)],
                    )
                }));

                // push (A + B, 0) tuples to U8PairRangeCheck circuit. 0 is used just as a stub
                lookups.extend((0..4).map(|i| {
                    push(
                        one(),
                        vec![c_usize(u8_pair_range_check_idx), var(i + 8), zero()],
                    )
                }));
                lookups
            }

            // (2 push lookups to pair_range_check)
            Self::U32RightRotate8 => {
                let mut lookups = vec![pull(
                    var(0),
                    vec![
                        c_usize(u32_right_rotate_8_idx),
                        var(1)
                            + var(2) * c_u32(256)
                            + var(3) * c_u32(256 * 256)
                            + var(4) * c_u32(256 * 256 * 256),
                        // note var indices
                        var(2)
                            + var(3) * c_u32(256)
                            + var(4) * c_u32(256 * 256)
                            + var(1) * c_u32(256 * 256 * 256),
                    ],
                )];

                // range check only input u32 word (output is built from the same bytes)
                lookups.extend((0..2).map(|i| {
                    push(
                        one(),
                        vec![c_usize(u8_pair_range_check_idx), var(i + 1), var(i + 3)],
                    )
                }));

                lookups
            }

            // (2 push lookups to pair_range_check)
            Self::U32RightRotate16 => {
                let mut lookups = vec![pull(
                    var(0),
                    vec![
                        c_usize(u32_right_rotate_16_idx),
                        var(1)
                            + var(2) * c_u32(256)
                            + var(3) * c_u32(256 * 256)
                            + var(4) * c_u32(256 * 256 * 256),
                        // note var indices
                        var(3)
                            + var(4) * c_u32(256)
                            + var(1) * c_u32(256 * 256)
                            + var(2) * c_u32(256 * 256 * 256),
                    ],
                )];

                // range check only input u32 word (output is built from the same 4 bytes)
                lookups.extend((0..2).map(|i| {
                    push(
                        one(),
                        vec![c_usize(u8_pair_range_check_idx), var(i + 1), var(i + 3)],
                    )
                }));

                lookups
            }

            Self::U32RightRotate12 => {
                vec![pull(
                    var(0),
                    vec![
                        c_usize(u32_right_rotate_12_idx),
                        var(1)
                            + var(2) * c_u32(256)
                            + var(3) * c_u32(256 * 256)
                            + var(4) * c_u32(256 * 256 * 256),
                        var(5)
                            + var(6) * c_u32(256)
                            + var(7) * c_u32(256 * 256)
                            + var(8) * c_u32(256 * 256 * 256),
                    ],
                )]
            }

            Self::U32RightRotate7 => {
                vec![pull(
                    var(0),
                    vec![
                        c_usize(u32_right_rotate_7_idx),
                        var(1)
                            + var(2) * c_u32(256)
                            + var(3) * c_u32(256 * 256)
                            + var(4) * c_u32(256 * 256 * 256),
                        var(5)
                            + var(6) * c_u32(256)
                            + var(7) * c_u32(256 * 256)
                            + var(8) * c_u32(256 * 256 * 256),
                    ],
                )]
            }

            // Totally 81 byte columns:
            // multiplicity, a_in(4), b_in(4), c_in(4), d_in(4), mx_in(4), my_in(4),
            // a_0_tmp(4), a_0(4), d_0_tmp(4), d_0(4), c_0(4), b_0_tmp(4), b_0(4),
            // a_1_tmp(4), a_1(4), d_1_tmp(4), d_1(4), c_1(4), b_1_tmp(4), b_1(4)
            Self::GFunction => {
                vec![
                    // balancing the initial claim
                    pull(
                        var(0),
                        vec![
                            c_usize(g_function_idx),
                            var(1) // a_in
                                + var(2) * c_u32(256)
                                + var(3) * c_u32(256 * 256)
                                + var(4) * c_u32(256 * 256 * 256),
                            var(5) // b_in
                                + var(6) * c_u32(256)
                                + var(7) * c_u32(256 * 256)
                                + var(8) * c_u32(256 * 256 * 256),
                            var(9) // c_in
                                + var(10) * c_u32(256)
                                + var(11) * c_u32(256 * 256)
                                + var(12) * c_u32(256 * 256 * 256),
                            var(13) // d_in
                                + var(14) * c_u32(256)
                                + var(15) * c_u32(256 * 256)
                                + var(16) * c_u32(256 * 256 * 256),
                            var(17) // mx_in
                                + var(18) * c_u32(256)
                                + var(19) * c_u32(256 * 256)
                                + var(20) * c_u32(256 * 256 * 256),
                            var(21) // my_in
                                + var(22) * c_u32(256)
                                + var(23) * c_u32(256 * 256)
                                + var(24) * c_u32(256 * 256 * 256),
                            // note indices!
                            var(57) // a_1
                                + var(58) * c_u32(256)
                                + var(59) * c_u32(256 * 256)
                                + var(60) * c_u32(256 * 256 * 256),
                            var(65) // d_1
                                + var(66) * c_u32(256)
                                + var(67) * c_u32(256 * 256)
                                + var(68) * c_u32(256 * 256 * 256),
                            var(69) // c_1
                                + var(70) * c_u32(256)
                                + var(71) * c_u32(256 * 256)
                                + var(72) * c_u32(256 * 256 * 256),
                            var(77) // b_1
                                + var(78) * c_u32(256)
                                + var(79) * c_u32(256 * 256)
                                + var(80) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // interacting with lower-level circuits that constrain operations used in G function

                    // a_in + b_in = a_0_tmp
                    push(
                        one(),
                        vec![
                            c_usize(u32_add_idx),
                            var(1)
                                + var(2) * c_u32(256)
                                + var(3) * c_u32(256 * 256)
                                + var(4) * c_u32(256 * 256 * 256),
                            var(5)
                                + var(6) * c_u32(256)
                                + var(7) * c_u32(256 * 256)
                                + var(8) * c_u32(256 * 256 * 256),
                            var(25)
                                + var(26) * c_u32(256)
                                + var(27) * c_u32(256 * 256)
                                + var(28) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // a_0_tmp + mx_in = a_0
                    push(
                        one(),
                        vec![
                            c_usize(u32_add_idx),
                            var(25)
                                + var(26) * c_u32(256)
                                + var(27) * c_u32(256 * 256)
                                + var(28) * c_u32(256 * 256 * 256),
                            var(17)
                                + var(18) * c_u32(256)
                                + var(19) * c_u32(256 * 256)
                                + var(20) * c_u32(256 * 256 * 256),
                            var(29)
                                + var(30) * c_u32(256)
                                + var(31) * c_u32(256 * 256)
                                + var(32) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // d_in ^ a_0 = d_0_tmp
                    push(
                        one(),
                        vec![
                            c_usize(u32_xor_idx),
                            var(13)
                                + var(14) * c_u32(256)
                                + var(15) * c_u32(256 * 256)
                                + var(16) * c_u32(256 * 256 * 256),
                            var(29)
                                + var(30) * c_u32(256)
                                + var(31) * c_u32(256 * 256)
                                + var(32) * c_u32(256 * 256 * 256),
                            var(33)
                                + var(34) * c_u32(256)
                                + var(35) * c_u32(256 * 256)
                                + var(36) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // d_0_tmp >> 16 = d_0
                    push(
                        one(),
                        vec![
                            c_usize(u32_right_rotate_16_idx),
                            var(33)
                                + var(34) * c_u32(256)
                                + var(35) * c_u32(256 * 256)
                                + var(36) * c_u32(256 * 256 * 256),
                            var(37)
                                + var(38) * c_u32(256)
                                + var(39) * c_u32(256 * 256)
                                + var(40) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // c_in + d_0 = c_0
                    push(
                        one(),
                        vec![
                            c_usize(u32_add_idx),
                            var(9)
                                + var(10) * c_u32(256)
                                + var(11) * c_u32(256 * 256)
                                + var(12) * c_u32(256 * 256 * 256),
                            var(37)
                                + var(38) * c_u32(256)
                                + var(39) * c_u32(256 * 256)
                                + var(40) * c_u32(256 * 256 * 256),
                            var(41)
                                + var(42) * c_u32(256)
                                + var(43) * c_u32(256 * 256)
                                + var(44) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // b_in ^ c_0 = b_0_tmp
                    push(
                        one(),
                        vec![
                            c_usize(u32_xor_idx),
                            var(5)
                                + var(6) * c_u32(256)
                                + var(7) * c_u32(256 * 256)
                                + var(8) * c_u32(256 * 256 * 256),
                            var(41)
                                + var(42) * c_u32(256)
                                + var(43) * c_u32(256 * 256)
                                + var(44) * c_u32(256 * 256 * 256),
                            var(45)
                                + var(46) * c_u32(256)
                                + var(47) * c_u32(256 * 256)
                                + var(48) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // b_0_tmp >> 12 = b_0
                    push(
                        one(),
                        vec![
                            c_usize(u32_right_rotate_12_idx),
                            var(45)
                                + var(46) * c_u32(256)
                                + var(47) * c_u32(256 * 256)
                                + var(48) * c_u32(256 * 256 * 256),
                            var(49)
                                + var(50) * c_u32(256)
                                + var(51) * c_u32(256 * 256)
                                + var(52) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // a_0 + b_0 = a_1_tmp
                    push(
                        one(),
                        vec![
                            c_usize(u32_add_idx),
                            var(29)
                                + var(30) * c_u32(256)
                                + var(31) * c_u32(256 * 256)
                                + var(32) * c_u32(256 * 256 * 256),
                            var(49)
                                + var(50) * c_u32(256)
                                + var(51) * c_u32(256 * 256)
                                + var(52) * c_u32(256 * 256 * 256),
                            var(53)
                                + var(54) * c_u32(256)
                                + var(55) * c_u32(256 * 256)
                                + var(56) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // a_1_tmp, my_in, a_1
                    push(
                        one(),
                        vec![
                            c_usize(u32_add_idx),
                            var(53)
                                + var(54) * c_u32(256)
                                + var(55) * c_u32(256 * 256)
                                + var(56) * c_u32(256 * 256 * 256),
                            var(21)
                                + var(22) * c_u32(256)
                                + var(23) * c_u32(256 * 256)
                                + var(24) * c_u32(256 * 256 * 256),
                            var(57)
                                + var(58) * c_u32(256)
                                + var(59) * c_u32(256 * 256)
                                + var(60) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // d_0 ^ a_1 = d_1_tmp
                    push(
                        one(),
                        vec![
                            c_usize(u32_xor_idx),
                            var(37)
                                + var(38) * c_u32(256)
                                + var(39) * c_u32(256 * 256)
                                + var(40) * c_u32(256 * 256 * 256),
                            var(57)
                                + var(58) * c_u32(256)
                                + var(59) * c_u32(256 * 256)
                                + var(60) * c_u32(256 * 256 * 256),
                            var(61)
                                + var(62) * c_u32(256)
                                + var(63) * c_u32(256 * 256)
                                + var(64) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // d_1_tmp >> 8 = d_1
                    push(
                        one(),
                        vec![
                            c_usize(u32_right_rotate_8_idx),
                            var(61)
                                + var(62) * c_u32(256)
                                + var(63) * c_u32(256 * 256)
                                + var(64) * c_u32(256 * 256 * 256),
                            var(65)
                                + var(66) * c_u32(256)
                                + var(67) * c_u32(256 * 256)
                                + var(68) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // c_0 + d_1 = c_1
                    push(
                        one(),
                        vec![
                            c_usize(u32_add_idx),
                            var(41)
                                + var(42) * c_u32(256)
                                + var(43) * c_u32(256 * 256)
                                + var(44) * c_u32(256 * 256 * 256),
                            var(65)
                                + var(66) * c_u32(256)
                                + var(67) * c_u32(256 * 256)
                                + var(68) * c_u32(256 * 256 * 256),
                            var(69)
                                + var(70) * c_u32(256)
                                + var(71) * c_u32(256 * 256)
                                + var(72) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // b_0 ^ c_1 = b_1_tmp
                    push(
                        one(),
                        vec![
                            c_usize(u32_xor_idx),
                            var(49)
                                + var(50) * c_u32(256)
                                + var(51) * c_u32(256 * 256)
                                + var(52) * c_u32(256 * 256 * 256),
                            var(69)
                                + var(70) * c_u32(256)
                                + var(71) * c_u32(256 * 256)
                                + var(72) * c_u32(256 * 256 * 256),
                            var(73)
                                + var(74) * c_u32(256)
                                + var(75) * c_u32(256 * 256)
                                + var(76) * c_u32(256 * 256 * 256),
                        ],
                    ),
                    // b_1_tmp >> 7 = b_1
                    push(
                        one(),
                        vec![
                            c_usize(u32_right_rotate_7_idx),
                            var(73)
                                + var(74) * c_u32(256)
                                + var(75) * c_u32(256 * 256)
                                + var(76) * c_u32(256 * 256 * 256),
                            var(77)
                                + var(78) * c_u32(256)
                                + var(79) * c_u32(256 * 256)
                                + var(80) * c_u32(256 * 256 * 256),
                        ],
                    ),
                ]
            }

            // multiplicity,
            // state_in (32 * 4),
            // a_in (4), b_in (4), c_in (4), d_in (4), mx_in (4), my_in (4), a_1 (4), d_1 (4), c_1 (4), b_1 (4)
            // state_out (32 * 4),
            Self::Compression => {
                let mut lookups = vec![
                    // pulling state_in / state_out (to balance initial claim)
                    pull_state_in_state_out(var(0), compression_idx, 1..128 + 1, 2561..2624 + 1),
                ];

                // pushing data for 56 rounds of g_function
                for round in 0..56usize {
                    let start = 129 + round * 40;
                    lookups.push(push_round(one(), g_function_idx, start..start + 40));
                }

                // pushing data for state[i] ^= state[i + 8] operation (8)
                // and state[i + 8] ^= chaining_value[i] operation (8)
                for k in 0..16usize {
                    let start = 2369 + k * 12;
                    lookups.push(push_u32(one(), u32_xor_idx, start..start + 12));
                }

                lookups
            }
        }
    }

    fn circuit_inputs(&self) -> CircuitInputs<Val> {
        CircuitInputs {
            main_width: self.width(),
            preprocessed: self.preprocessed_trace(),
            constraints: self.constraints(),
            lookups: self.lookups(),
            ..Default::default()
        }
    }
}

/// Builds the 9-circuit system. Note that `U8Xor` and `U8PairRangeCheck` share
/// a single preprocessed circuit (the `U8Xor` variant exposes both channels),
/// so there are 9 circuits for 10 channels.
fn build_system(
    config: GoldilocksBlake3Config,
) -> (
    System<GoldilocksBlake3Config>,
    ProverKey<GoldilocksBlake3Config>,
) {
    let circuits = [
        Blake3CompressionCircuit::U8Xor,
        Blake3CompressionCircuit::U32Xor,
        Blake3CompressionCircuit::U32Add,
        Blake3CompressionCircuit::U32RightRotate8,
        Blake3CompressionCircuit::U32RightRotate16,
        Blake3CompressionCircuit::U32RightRotate12,
        Blake3CompressionCircuit::U32RightRotate7,
        Blake3CompressionCircuit::GFunction,
        Blake3CompressionCircuit::Compression,
    ];
    System::new(
        config,
        circuits
            .iter()
            .map(Blake3CompressionCircuit::circuit_inputs),
    )
}

fn config() -> GoldilocksBlake3Config {
    GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 64,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    )
}

struct Blake3CompressionClaims {
    claims: Vec<Vec<Val>>,
}

impl Blake3CompressionClaims {
    #[allow(clippy::too_many_lines)]
    fn witness(&self, system: &System<GoldilocksBlake3Config>) -> SystemWitness<Val> {
        // Grabbing values from the claims.

        let mut u32_xor_values_from_claims = vec![];
        let mut u32_add_values_from_claims = vec![];
        let mut byte_xor_values_from_claims = vec![];
        let mut byte_range_check_values_from_claims = vec![];
        let mut u32_rotate_right_8_values_from_claims = vec![];
        let mut u32_rotate_right_16_values_from_claims = vec![];
        let mut u32_rotate_right_12_values_from_claims = vec![];
        let mut u32_rotate_right_7_values_from_claims = vec![];
        let mut g_function_values_from_claims = vec![];
        let mut state_transition_values_from_claims = vec![];

        for claim in self.claims.clone() {
            // we should have at least the circuit index
            assert!(!claim.is_empty(), "wrong claim format");
            match claim[0].as_canonical_u64() {
                0u64 => {
                    // U8Xor claim: circuit_idx, A, B, A xor B (A, B are bytes)
                    assert!(claim.len() == 4, "[U8Xor] wrong claim format");
                    byte_xor_values_from_claims.push((claim[1], claim[2], claim[3]));
                }
                1u64 => {
                    // U32Xor claim: circuit_idx, A, B, A xor B (A, B are u32)
                    assert!(claim.len() == 4, "[U32Xor] wrong claim format");
                    let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                    let b_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                    let xor_u32 = u32::try_from(claim[3].as_canonical_u64()).unwrap();
                    u32_xor_values_from_claims.push((a_u32, b_u32, xor_u32));
                }
                2u64 => {
                    // U32Add claim: circuit_idx, A, B, A + B (A, B are u32)
                    assert!(claim.len() == 4, "[U32Add] wrong claim format");
                    let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                    let b_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                    let add_u32 = u32::try_from(claim[3].as_canonical_u64()).unwrap();
                    u32_add_values_from_claims.push((a_u32, b_u32, add_u32));
                }
                3u64 => {
                    // U32RotateRight8 claim: circuit_idx, A, A_rot
                    assert!(claim.len() == 3, "[U32RightRotate8] wrong claim format");
                    let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                    let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                    u32_rotate_right_8_values_from_claims.push((a_u32, rot_u32));
                }
                4u64 => {
                    // U32RotateRight16 claim: circuit_idx, A, A_rot
                    assert!(claim.len() == 3, "[U32RightRotate16] wrong claim format");
                    let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                    let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                    u32_rotate_right_16_values_from_claims.push((a_u32, rot_u32));
                }
                5u64 => {
                    // U32RotateRight12 claim: circuit_idx, A, A_rot
                    assert!(claim.len() == 3, "[U32RightRotate12] wrong claim format");
                    let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                    let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                    u32_rotate_right_12_values_from_claims.push((a_u32, rot_u32));
                }
                6u64 => {
                    // U32RotateRight7 claim: circuit_idx, A, A_rot
                    assert!(claim.len() == 3, "[U32RightRotate7] wrong claim format");
                    let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                    let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                    u32_rotate_right_7_values_from_claims.push((a_u32, rot_u32));
                }
                7u64 => {
                    // U8PairRangeCheck claim: circuit_idx, A, B
                    assert!(claim.len() == 3, "[U8PairRangeCheck] wrong claim format");
                    byte_range_check_values_from_claims.push((claim[1], claim[2]));
                }
                8u64 => {
                    // GFunction claim: circuit_idx, A, B, C, D, MX_IN, MY_IN, A1, D1, C1, B1
                    assert!(claim.len() == 11, "[GFunction] wrong claim format");
                    let a_in = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                    let b_in = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                    let c_in = u32::try_from(claim[3].as_canonical_u64()).unwrap();
                    let d_in = u32::try_from(claim[4].as_canonical_u64()).unwrap();
                    let mx_in = u32::try_from(claim[5].as_canonical_u64()).unwrap();
                    let my_in = u32::try_from(claim[6].as_canonical_u64()).unwrap();
                    let a_1 = u32::try_from(claim[7].as_canonical_u64()).unwrap();
                    let d_1 = u32::try_from(claim[8].as_canonical_u64()).unwrap();
                    let c_1 = u32::try_from(claim[9].as_canonical_u64()).unwrap();
                    let b_1 = u32::try_from(claim[10].as_canonical_u64()).unwrap();
                    g_function_values_from_claims
                        .push((a_in, b_in, c_in, d_in, mx_in, my_in, a_1, b_1, c_1, d_1));
                }
                9u64 => {
                    // StateTransition claim: circuit_idx, state_in[32], state_out[16]
                    assert!(claim.len() == 49, "[StateTransition] wrong claim format");
                    let state_in: [u32; 32] =
                        array::from_fn(|i| u32::try_from(claim[i + 1].as_canonical_u64()).unwrap());
                    let state_out: [u32; 16] = array::from_fn(|i| {
                        u32::try_from(claim[i + 1 + 32].as_canonical_u64()).unwrap()
                    });
                    state_transition_values_from_claims.push((state_in, state_out));
                }
                _ => panic!("unsupported circuit"),
            }
        }

        // Build traces. If a claim for a given circuit was not provided (and hence
        // no data is available), we use a zero trace and balance lookups with zeros.

        let mut state_transition_trace_values =
            Vec::<Val>::with_capacity(state_transition_values_from_claims.len());
        if state_transition_values_from_claims.is_empty() {
            state_transition_trace_values = Val::zero_vec(COMPRESSION_TRACE_WIDTH);
            for _ in 0..56 {
                g_function_values_from_claims
                    .push((0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32));
            }

            for _ in 0..8 {
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
            }
        } else {
            for (state_in_io, state_out_io) in state_transition_values_from_claims {
                let state_in_io_bytes = state_in_io
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<u8>>();
                state_transition_trace_values.push(Val::ONE); // multiplicity
                state_transition_trace_values.extend_from_slice(
                    state_in_io_bytes
                        .into_iter()
                        .map(Val::from_u8)
                        .collect::<Vec<Val>>()
                        .as_slice(),
                );

                let mut state = state_in_io;
                for round_idx in 0..7 {
                    for j in 0..8 {
                        let a_in = state[A[j]];
                        let b_in = state[B[j]];
                        let c_in = state[C[j]];
                        let d_in = state[D[j]];
                        let mx_in = state[MX[j]];
                        let my_in = state[MY[j]];

                        let a_0 = a_in.wrapping_add(b_in).wrapping_add(mx_in);
                        let d_0 = (d_in ^ a_0).rotate_right(16);
                        let c_0 = c_in.wrapping_add(d_0);
                        let b_0 = (b_in ^ c_0).rotate_right(12);

                        let a_1 = a_0.wrapping_add(b_0).wrapping_add(my_in);
                        let d_1 = (d_0 ^ a_1).rotate_right(8);
                        let c_1 = c_0.wrapping_add(d_1);
                        let b_1 = (b_0 ^ c_1).rotate_right(7);

                        g_function_values_from_claims
                            .push((a_in, b_in, c_in, d_in, mx_in, my_in, a_1, b_1, c_1, d_1)); // send data to G_Function circuit

                        state[A[j]] = a_1;
                        state[B[j]] = b_1;
                        state[C[j]] = c_1;
                        state[D[j]] = d_1;

                        for u32_val in
                            [a_in, b_in, c_in, d_in, mx_in, my_in, a_1, d_1, c_1, b_1].iter()
                        {
                            let bytes: [u8; 4] = u32_val.to_le_bytes();
                            state_transition_trace_values
                                .extend_from_slice(bytes.map(Val::from_u8).as_slice());
                        }
                    }

                    // execute permutation for the 6 first rounds
                    if round_idx < 6 {
                        let mut permuted = [0; 16];
                        for i in 0..16 {
                            permuted[i] = state[16 + MSG_PERMUTATION[i]];
                        }
                        state[16..(16 + 16)].copy_from_slice(&permuted);
                    }
                }

                for i in 0..8 {
                    let left = state[i];
                    let right = state[i + 8];
                    state[i] ^= state[i + 8]; // ^ state[i + 8]
                    let xor = state[i];

                    // save (state[i]), (state[i + 8]) and (state[i] ^ state[i + 8]) for looking up
                    let left_bytes: [u8; 4] = left.to_le_bytes();
                    let right_bytes: [u8; 4] = right.to_le_bytes();
                    let xor_bytes: [u8; 4] = xor.to_le_bytes();

                    state_transition_trace_values
                        .extend_from_slice(left_bytes.map(Val::from_u8).as_slice());
                    state_transition_trace_values
                        .extend_from_slice(right_bytes.map(Val::from_u8).as_slice());
                    state_transition_trace_values
                        .extend_from_slice(xor_bytes.map(Val::from_u8).as_slice());

                    u32_xor_values_from_claims.push((left, right, xor)); // send data to U32Xor circuit

                    let left = state[i + 8];
                    let right = state_in_io[i];
                    state[i + 8] ^= state_in_io[i]; // ^ chaining_value[i]
                    let xor = state[i + 8];

                    // save (state[i + 8]), (state_in_io[i]) and their xor for looking up
                    let left_bytes: [u8; 4] = left.to_le_bytes();
                    let right_bytes: [u8; 4] = right.to_le_bytes();
                    let xor_bytes: [u8; 4] = xor.to_le_bytes();

                    state_transition_trace_values
                        .extend_from_slice(left_bytes.map(Val::from_u8).as_slice());
                    state_transition_trace_values
                        .extend_from_slice(right_bytes.map(Val::from_u8).as_slice());
                    state_transition_trace_values
                        .extend_from_slice(xor_bytes.map(Val::from_u8).as_slice());

                    u32_xor_values_from_claims.push((left, right, xor)); // send data to U32Xor circuit
                }

                let mut state_out = state.to_vec();
                state_out.truncate(16); // compression output is first 16 u32 words of state_out

                debug_assert_eq!(state_out_io.to_vec(), state_out);
                let state_out_io_bytes = state_out_io
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<u8>>();
                state_transition_trace_values.extend_from_slice(
                    state_out_io_bytes
                        .into_iter()
                        .map(Val::from_u8)
                        .collect::<Vec<Val>>()
                        .as_slice(),
                );
            }
        }
        let mut state_transition_trace =
            RowMajorMatrix::new(state_transition_trace_values, COMPRESSION_TRACE_WIDTH);
        let height = state_transition_trace.height().next_power_of_two();
        let zero_rows_added = height - state_transition_trace.height();
        for _ in 0..zero_rows_added {
            // we have 56 communications with G_Function circuit
            for _ in 0..56 {
                g_function_values_from_claims
                    .push((0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32));
            }

            // we have 8 * 2 communications with U32_XOR circuit
            for _ in 0..8 {
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
            }
        }
        state_transition_trace.pad_to_height(height, Val::ZERO);

        // build GFunction trace columns:
        // multiplicity, a_in(4), b_in(4), c_in(4), d_in(4), mx_in(4), my_in(4),
        // a_0_tmp(4), a_0(4), d_0_tmp(4), d_0(4), c_0(4), b_0_tmp(4), b_0(4),
        // a_1_tmp(4), a_1(4), d_1_tmp(4), d_1(4), c_1(4), b_1_tmp(4), b_1(4))
        let mut g_function_trace_values =
            Vec::<Val>::with_capacity(g_function_values_from_claims.len());
        if g_function_values_from_claims.is_empty() {
            g_function_trace_values = Val::zero_vec(G_FUNCTION_TRACE_WIDTH);

            // 1 rot7
            u32_rotate_right_7_values_from_claims.push((0u32, 0u32));
            // 1 rot8
            u32_rotate_right_8_values_from_claims.push((0u32, 0u32));
            // 1 rot16
            u32_rotate_right_16_values_from_claims.push((0u32, 0u32));
            // 1 rot12
            u32_rotate_right_12_values_from_claims.push((0u32, 0u32));

            // 4 u32_xor
            for _ in 0..4 {
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
            }

            // 6 u32_add
            for _ in 0..6 {
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
            }
        } else {
            for (a_in, b_in, c_in, d_in, mx_in, my_in, a1, b1, c1, d1) in
                g_function_values_from_claims
            {
                let a_0_tmp = a_in.wrapping_add(b_in);
                u32_add_values_from_claims.push((a_in, b_in, a_0_tmp)); // send data to U32Add circuit

                let a_0 = a_0_tmp.wrapping_add(mx_in);
                u32_add_values_from_claims.push((a_0_tmp, mx_in, a_0)); // send data to U32Add circuit

                let d_0_tmp = d_in ^ a_0;
                u32_xor_values_from_claims.push((d_in, a_0, d_0_tmp)); // send data to U32Xor circuit

                let d_0 = d_0_tmp.rotate_right(16);
                u32_rotate_right_16_values_from_claims.push((d_0_tmp, d_0)); // send data to U32RightRotate16 circuit

                let c_0 = c_in.wrapping_add(d_0);
                u32_add_values_from_claims.push((c_in, d_0, c_0)); // send data to U32Add circuit

                let b_0_tmp = b_in ^ c_0;
                u32_xor_values_from_claims.push((b_in, c_0, b_0_tmp)); // send data to U32Xor circuit

                let b_0 = b_0_tmp.rotate_right(12);
                u32_rotate_right_12_values_from_claims.push((b_0_tmp, b_0)); // send data to U32RightRotate12 circuit

                let a_1_tmp = a_0.wrapping_add(b_0);
                u32_add_values_from_claims.push((a_0, b_0, a_1_tmp)); // send data to U32Add circuit

                let a_1 = a_1_tmp.wrapping_add(my_in);
                u32_add_values_from_claims.push((a_1_tmp, my_in, a_1)); // send data to U32Add circuit

                let d_1_tmp = d_0 ^ a_1;
                u32_xor_values_from_claims.push((d_0, a_1, d_1_tmp)); // send data to U32Xor circuit

                let d_1 = d_1_tmp.rotate_right(8);
                u32_rotate_right_8_values_from_claims.push((d_1_tmp, d_1));

                let c_1 = c_0.wrapping_add(d_1);
                u32_add_values_from_claims.push((c_0, d_1, c_1)); // send data to U32Add circuit

                let b_1_tmp = b_0 ^ c_1;
                u32_xor_values_from_claims.push((b_0, c_1, b_1_tmp)); // send data to U32Xor circuit

                let b_1 = b_1_tmp.rotate_right(7);
                u32_rotate_right_7_values_from_claims.push((b_1_tmp, b_1));

                debug_assert_eq!(a_1, a1);
                debug_assert_eq!(b_1, b1);
                debug_assert_eq!(c_1, c1);
                debug_assert_eq!(d_1, d1);

                g_function_trace_values.push(Val::ONE); // multiplicity
                for u32_val in [
                    a_in, b_in, c_in, d_in, mx_in, my_in, a_0_tmp, a_0, d_0_tmp, d_0, c_0, b_0_tmp,
                    b_0, a_1_tmp, a_1, d_1_tmp, d_1, c_1, b_1_tmp, b_1,
                ]
                .iter()
                {
                    let bytes: [u8; 4] = u32_val.to_le_bytes();
                    g_function_trace_values.extend_from_slice(bytes.map(Val::from_u8).as_slice());
                }
            }
        }
        let mut g_function_trace =
            RowMajorMatrix::new(g_function_trace_values, G_FUNCTION_TRACE_WIDTH);
        let height = g_function_trace.height().next_power_of_two();
        let zero_rows_added = height - g_function_trace.height();
        for _ in 0..zero_rows_added {
            u32_rotate_right_7_values_from_claims.push((0u32, 0u32));
            u32_rotate_right_8_values_from_claims.push((0u32, 0u32));
            u32_rotate_right_16_values_from_claims.push((0u32, 0u32));
            u32_rotate_right_12_values_from_claims.push((0u32, 0u32));

            for _ in 0..4 {
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
            }

            for _ in 0..6 {
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
            }
        }
        g_function_trace.pad_to_height(height, Val::ZERO);

        // build U32Xor trace (columns: multiplicity, A0..A3, B0..B3, A0^B0..A3^B3)
        let mut u32_xor_trace_values = Vec::<Val>::with_capacity(u32_xor_values_from_claims.len());
        if u32_xor_values_from_claims.is_empty() {
            u32_xor_trace_values = Val::zero_vec(U32_XOR_TRACE_WIDTH);

            // we also need to balance the U8Xor circuit lookups using zeroes
            for _ in 0..4 {
                byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
            }
        } else {
            for (left, right, xor) in u32_xor_values_from_claims {
                debug_assert_eq!(left ^ right, xor);

                let left_bytes: [u8; 4] = left.to_le_bytes();
                let right_bytes: [u8; 4] = right.to_le_bytes();
                let xor_bytes: [u8; 4] = xor.to_le_bytes();

                u32_xor_trace_values.push(Val::ONE); // multiplicity

                u32_xor_trace_values.extend_from_slice(left_bytes.map(Val::from_u8).as_slice());
                u32_xor_trace_values.extend_from_slice(right_bytes.map(Val::from_u8).as_slice());
                u32_xor_trace_values.extend_from_slice(xor_bytes.map(Val::from_u8).as_slice());

                // we send bytes to the U8Xor circuit, relying on lookup constraining
                for i in 0..4 {
                    byte_xor_values_from_claims.push((
                        Val::from_u8(left_bytes[i]),
                        Val::from_u8(right_bytes[i]),
                        Val::from_u8(xor_bytes[i]),
                    ));
                }
            }
        }
        let mut u32_xor_trace = RowMajorMatrix::new(u32_xor_trace_values, U32_XOR_TRACE_WIDTH);
        let height = u32_xor_trace.height().next_power_of_two();
        let zero_rows = height - u32_xor_trace.height();
        for _ in 0..zero_rows {
            // balance the U8Xor circuit lookups using zeroes for every padded row
            for _ in 0..4 {
                byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
            }
        }
        u32_xor_trace.pad_to_height(height, Val::ZERO);

        // build U32Add trace (columns: A0..A3, B0..B3, C0..C3, carry, multiplicity)
        let mut u32_add_trace_values = vec![];
        if u32_add_values_from_claims.is_empty() {
            u32_add_trace_values = Val::zero_vec(U32_ADD_TRACE_WIDTH);

            // balance the lookups using zeroes
            for _ in 0..8 {
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            }
        } else {
            for (left, right, sum) in u32_add_values_from_claims {
                let (z, carry) = left.overflowing_add(right);
                // actual addition result should match the value from the claim
                debug_assert_eq!(z, sum);

                let left_bytes: [u8; 4] = left.to_le_bytes();
                let right_bytes: [u8; 4] = right.to_le_bytes();
                let sum_bytes: [u8; 4] = sum.to_le_bytes();

                u32_add_trace_values.extend_from_slice(left_bytes.map(Val::from_u8).as_slice());
                u32_add_trace_values.extend_from_slice(right_bytes.map(Val::from_u8).as_slice());
                u32_add_trace_values.extend_from_slice(sum_bytes.map(Val::from_u8).as_slice());

                u32_add_trace_values.push(Val::from_bool(carry));
                u32_add_trace_values.push(Val::ONE); // multiplicity

                // send decomposed bytes to the U8PairRangeCheck circuit
                for i in 0..4 {
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(left_bytes[i]), Val::from_u8(right_bytes[i])));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(sum_bytes[i]), Val::ZERO));
                }
            }
        }
        let mut u32_add_trace = RowMajorMatrix::new(u32_add_trace_values, U32_ADD_TRACE_WIDTH);
        let height = u32_add_trace.height().next_power_of_two();
        let zero_rows = height - u32_add_trace.height();
        for _ in 0..zero_rows {
            // balance the lookups using zeroes for every padded row
            for _ in 0..8 {
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            }
        }
        u32_add_trace.pad_to_height(height, Val::ZERO);

        // build U32RotateRight8 trace (columns: multiplicity, a0..a3, rot0..rot3)
        let mut u32_rotate_right_8_trace_values = vec![];
        if u32_rotate_right_8_values_from_claims.is_empty() {
            u32_rotate_right_8_trace_values = Val::zero_vec(U32_RIGHT_ROTATE_8_TRACE_WIDTH);

            // balance U8PairRangeCheck circuit lookups using zeroes
            byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
        } else {
            for (val, rot) in u32_rotate_right_8_values_from_claims {
                u32_rotate_right_8_trace_values.push(Val::ONE); // multiplicity

                // actual rotate 8 result should match the value from the claim
                debug_assert_eq!(val.rotate_right(8), rot);

                let val_bytes: [u8; 4] = val.to_le_bytes();
                let rot_bytes: [u8; 4] = rot.to_le_bytes();

                u32_rotate_right_8_trace_values
                    .extend_from_slice(val_bytes.map(Val::from_u8).as_slice());
                u32_rotate_right_8_trace_values
                    .extend_from_slice(rot_bytes.map(Val::from_u8).as_slice());

                // send decomposed bytes to the U8PairRangeCheck circuit
                byte_range_check_values_from_claims
                    .push((Val::from_u8(val_bytes[0]), Val::from_u8(val_bytes[2])));
                byte_range_check_values_from_claims
                    .push((Val::from_u8(val_bytes[1]), Val::from_u8(val_bytes[3])));
            }
        }
        let mut u32_rotate_right_8_trace = RowMajorMatrix::new(
            u32_rotate_right_8_trace_values,
            U32_RIGHT_ROTATE_8_TRACE_WIDTH,
        );
        let height = u32_rotate_right_8_trace.height().next_power_of_two();
        let zero_rows = height - u32_rotate_right_8_trace.height();
        for _ in 0..zero_rows {
            byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
        }
        u32_rotate_right_8_trace.pad_to_height(height, Val::ZERO);

        // build U32RotateRight16 trace (columns: multiplicity, a0..a3, rot0..rot3)
        let mut u32_rotate_right_16_trace_values = vec![];
        if u32_rotate_right_16_values_from_claims.is_empty() {
            u32_rotate_right_16_trace_values = Val::zero_vec(U32_RIGHT_ROTATE_16_TRACE_WIDTH);

            byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
        } else {
            for (val, rot) in u32_rotate_right_16_values_from_claims {
                u32_rotate_right_16_trace_values.push(Val::ONE); // multiplicity

                // actual rotate 16 result should match the value from the claim
                debug_assert_eq!(val.rotate_right(16), rot);

                let a_bytes: [u8; 4] = val.to_le_bytes();
                let rot_bytes: [u8; 4] = rot.to_le_bytes();

                u32_rotate_right_16_trace_values
                    .extend_from_slice(a_bytes.map(Val::from_u8).as_slice());
                u32_rotate_right_16_trace_values
                    .extend_from_slice(rot_bytes.map(Val::from_u8).as_slice());

                // send decomposed bytes to the U8PairRangeCheck circuit
                byte_range_check_values_from_claims
                    .push((Val::from_u8(a_bytes[0]), Val::from_u8(a_bytes[2])));
                byte_range_check_values_from_claims
                    .push((Val::from_u8(a_bytes[1]), Val::from_u8(a_bytes[3])));
            }
        }
        let mut u32_rotate_right_16_trace = RowMajorMatrix::new(
            u32_rotate_right_16_trace_values,
            U32_RIGHT_ROTATE_16_TRACE_WIDTH,
        );
        let height = u32_rotate_right_16_trace.height().next_power_of_two();
        let zero_rows = height - u32_rotate_right_16_trace.height();
        for _ in 0..zero_rows {
            byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
        }
        u32_rotate_right_16_trace.pad_to_height(height, Val::ZERO);

        // The >>12 and >>7 gadgets reconstruct the word via a div/rem
        // decomposition; the div/rem bytes are not range-checked (see the
        // module docs / FIXME on the enum variants).
        fn rot_7_12_trace_values(k: u32, vals_from_claim: &[(u32, u32)]) -> RowMajorMatrix<Val> {
            let width = match k {
                7 => U32_RIGHT_ROTATE_7_TRACE_WIDTH,
                12 => U32_RIGHT_ROTATE_12_TRACE_WIDTH,
                _ => panic!("unexpected k"),
            };

            let mut values = vec![];
            if vals_from_claim.is_empty() {
                values = Val::zero_vec(width);
            } else {
                for (val, rot) in vals_from_claim {
                    values.push(Val::ONE); // multiplicity

                    // actual rotate result should match the value from the claim
                    debug_assert_eq!(val.rotate_right(k), *rot);

                    let two_pow_k = u32::try_from(2usize.pow(k)).unwrap();
                    let two_pow_32_minus_k = u32::try_from(2usize.pow(32 - k)).unwrap();

                    let input_div = val / two_pow_k;
                    let input_rem = val % two_pow_k;

                    let two_pow_k_bytes: [u8; 4] = two_pow_k.to_le_bytes();
                    let two_pow_32_minus_k_bytes: [u8; 4] = two_pow_32_minus_k.to_le_bytes();
                    let input_div_bytes: [u8; 4] = input_div.to_le_bytes();
                    let input_rem_bytes: [u8; 4] = input_rem.to_le_bytes();

                    let val_bytes: [u8; 4] = val.to_le_bytes();
                    let rot_bytes: [u8; 4] = rot.to_le_bytes();

                    values.extend_from_slice(val_bytes.map(Val::from_u8).as_slice());
                    values.extend_from_slice(rot_bytes.map(Val::from_u8).as_slice());
                    values.extend_from_slice(two_pow_k_bytes.map(Val::from_u8).as_slice());
                    values.extend_from_slice(two_pow_32_minus_k_bytes.map(Val::from_u8).as_slice());
                    values.extend_from_slice(input_div_bytes.map(Val::from_u8).as_slice());
                    values.extend_from_slice(input_rem_bytes.map(Val::from_u8).as_slice());
                }
            }

            let mut trace = RowMajorMatrix::new(values, width);
            let height = trace.height().next_power_of_two();
            trace.pad_to_height(height, Val::ZERO);

            trace
        }

        // build U32RotateRight12 trace
        let u32_rotate_right_12_trace =
            rot_7_12_trace_values(12, &u32_rotate_right_12_values_from_claims);

        // build U32RotateRight7 trace
        let u32_rotate_right_7_trace =
            rot_7_12_trace_values(7, &u32_rotate_right_7_values_from_claims);

        // finally build the U8Xor / U8PairRangeCheck trace (columns:
        // multiplicity_u8_xor, multiplicity_pair_range_check). Since this is the
        // lowest-level trace, its multiplicities are accumulated from the values
        // sent by every other circuit above.
        let mut u8_xor_range_check_trace_values = Vec::<Val>::with_capacity(
            BYTE_VALUES_NUM * BYTE_VALUES_NUM * U8_XOR_PAIR_RANGE_CHECK_TRACE_WIDTH,
        );
        for i in 0..BYTE_VALUES_NUM {
            for j in 0..BYTE_VALUES_NUM {
                let mut multiplicity_u8_xor = Val::ZERO;
                let mut multiplicity_u8_pair_range_check = Val::ZERO;

                for vals in byte_xor_values_from_claims.clone() {
                    if vals.0 == Val::from_usize(i)
                        && vals.1 == Val::from_usize(j)
                        && vals.2 == Val::from_usize(i ^ j)
                    {
                        multiplicity_u8_xor += Val::ONE;
                    }
                }

                for vals in byte_range_check_values_from_claims.clone() {
                    if vals.0 == Val::from_usize(i) && vals.1 == Val::from_usize(j) {
                        multiplicity_u8_pair_range_check += Val::ONE;
                    }
                }

                u8_xor_range_check_trace_values.push(multiplicity_u8_xor);
                u8_xor_range_check_trace_values.push(multiplicity_u8_pair_range_check);
            }
        }

        let traces = vec![
            RowMajorMatrix::new(
                u8_xor_range_check_trace_values,
                U8_XOR_PAIR_RANGE_CHECK_TRACE_WIDTH,
            ),
            u32_xor_trace,
            u32_add_trace,
            u32_rotate_right_8_trace,
            u32_rotate_right_16_trace,
            u32_rotate_right_12_trace,
            u32_rotate_right_7_trace,
            g_function_trace,
            state_transition_trace,
        ];

        SystemWitness::from_stage_1(traces, system)
    }
}

/// Proves and verifies the given claim set against a freshly built system.
fn run_test(claims: &Blake3CompressionClaims) {
    let (system, prover_key) = build_system(config());

    let witness = claims.witness(&system);

    let claims_slice: Vec<&[Val]> = claims.claims.iter().map(Vec::as_slice).collect();
    let claims_slice: &[&[Val]] = &claims_slice;

    let proof = system.prove_multiple_claims(&prover_key, claims_slice, witness);
    system
        .verify_multiple_claims(claims_slice, &proof)
        .expect("verification issue");
}

#[test]
fn test_compression_reference_compatibility() {
    let input: Vec<u8> = vec![0x54; 64];

    let (claim_data, expected) = blake3_new_update_finalize(&input);
    assert_eq!(claim_data.len(), 1);

    let claim_data = claim_data.first().unwrap();

    let state_in = [
        claim_data.cv.to_vec(),
        vec![
            IV[0],
            IV[1],
            IV[2],
            IV[3],
            claim_data.counter_low,
            claim_data.counter_high,
            claim_data.block_len,
            claim_data.flags,
        ],
        claim_data.block_words.to_vec(),
    ]
    .concat();

    let state_out = claim_data.output.to_vec();

    let mut actual = state_out
        .clone()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<u8>>();
    actual.truncate(32);

    assert_eq!(actual, expected.to_vec());

    // circuit testing
    let claims = Blake3CompressionClaims {
        claims: vec![
            [
                vec![Val::from_usize(
                    Blake3CompressionCircuit::Compression.position(),
                )],
                state_in.into_iter().map(Val::from_u32).collect(),
                state_out.into_iter().map(Val::from_u32).collect(),
            ]
            .concat(),
        ],
    };

    run_test(&claims);
}

#[test]
fn test_all_claims() {
    // computations IO

    let a_u8 = 0xa1u8;
    let b_u8 = 0xa8u8;
    let xor_u8 = a_u8 ^ b_u8;

    let a_u32 = 0x0000_00ffu32;
    let b_u32 = 0x0000_ff01u32;
    let xor_u32 = a_u32 ^ b_u32;
    let add_u32 = a_u32.wrapping_add(b_u32);
    let a_rot_8 = a_u32.rotate_right(8);
    let a_rot_16 = a_u32.rotate_right(16);
    let a_rot_12 = a_u32.rotate_right(12);
    let a_rot_7 = a_u32.rotate_right(7);

    // G function IO
    let a_in = 0x1111_1111u32;
    let b_in = 0x2222_2222u32;
    let c_in = 0x3333_3333u32;
    let d_in = 0x4444_4444u32;
    let mx_in = 0x5555_5555u32;
    let my_in = 0x6666_6666u32;

    let a_0_tmp = a_in.wrapping_add(b_in);
    let a_0 = a_0_tmp.wrapping_add(mx_in);
    let d_0_tmp = d_in ^ a_0;
    let d_0 = d_0_tmp.rotate_right(16);
    let c_0 = c_in.wrapping_add(d_0);
    let b_0_tmp = b_in ^ c_0;
    let b_0 = b_0_tmp.rotate_right(12);

    let a_1_tmp = a_0.wrapping_add(b_0);
    let a_1 = a_1_tmp.wrapping_add(my_in);
    let d_1_tmp = d_0 ^ a_1;
    let d_1 = d_1_tmp.rotate_right(8);
    let c_1 = c_0.wrapping_add(d_1);
    let b_1_tmp = b_0 ^ c_1;
    let b_1 = b_1_tmp.rotate_right(7);

    // compression IO
    let state_in = vec![
        0x0000_0000u32,
        0x0000_1111u32,
        0x0000_2222u32,
        0x0000_3333u32,
        0x0000_4444u32,
        0x0000_5555u32,
        0x0000_6666u32,
        0x0000_7777u32,
        0x0000_8888u32,
        0x0000_9999u32,
        0x0000_aaaau32,
        0x0000_bbbbu32,
        0x0000_ccccu32,
        0x0000_ddddu32,
        0x0000_eeeeu32,
        0x0000_ffffu32,
        0x0000_0000u32,
        0x1111_0000u32,
        0x2222_0000u32,
        0x3333_0000u32,
        0x4444_0000u32,
        0x5555_0000u32,
        0x6666_0000u32,
        0x7777_0000u32,
        0x8888_0000u32,
        0x9999_0000u32,
        0xaaaa_0000u32,
        0xbbbb_0000u32,
        0xcccc_0000u32,
        0xdddd_0000u32,
        0xeeee_0000u32,
        0xffff_0000u32,
    ];

    let state_out = vec![
        0xd304_e51cu32,
        0xc2df_34a0u32,
        0x5eba_7f1fu32,
        0x2ab9_650fu32,
        0xd9ce_f159u32,
        0x4e9d_3a6au32,
        0xcac2_e310u32,
        0xc6b9_be7eu32,
        0xad9f_d58au32,
        0x0899_e71bu32,
        0xca51_a599u32,
        0xc3fb_d7c0u32,
        0x751d_2f26u32,
        0x6cd0_ac6bu32,
        0xc58f_3c1du32,
        0xe6d6_5414u32,
    ];

    // claims construction
    let f = Val::from_u8;
    let f32 = Val::from_u32;

    // 1 u8 xor claim — leaf primitive
    let claims = Blake3CompressionClaims {
        claims: vec![vec![
            Val::from_usize(Blake3CompressionCircuit::U8Xor.position()),
            f(a_u8),
            f(b_u8),
            f(xor_u8),
        ]],
    };
    run_test(&claims);

    // 1 u32 xor claim — u32→u8 lookup chain
    let claims = Blake3CompressionClaims {
        claims: vec![vec![
            Val::from_usize(Blake3CompressionCircuit::U32Xor.position()),
            f32(a_u32),
            f32(b_u32),
            f32(xor_u32),
        ]],
    };
    run_test(&claims);

    // 1 u32 add claim — add→range check lookup chain
    let claims = Blake3CompressionClaims {
        claims: vec![vec![
            Val::from_usize(Blake3CompressionCircuit::U32Add.position()),
            f32(a_u32),
            f32(b_u32),
            f32(add_u32),
        ]],
    };
    run_test(&claims);

    // 1 claim per rotation variant
    let claims = Blake3CompressionClaims {
        claims: vec![
            vec![
                Val::from_usize(Blake3CompressionCircuit::U32RightRotate8.position()),
                f32(a_u32),
                f32(a_rot_8),
            ],
            vec![
                Val::from_usize(Blake3CompressionCircuit::U32RightRotate16.position()),
                f32(a_u32),
                f32(a_rot_16),
            ],
            vec![
                Val::from_usize(Blake3CompressionCircuit::U32RightRotate12.position()),
                f32(a_u32),
                f32(a_rot_12),
            ],
            vec![
                Val::from_usize(Blake3CompressionCircuit::U32RightRotate7.position()),
                f32(a_u32),
                f32(a_rot_7),
            ],
        ],
    };
    run_test(&claims);

    // 1 G-function claim — G→{add,xor,rotate} composition
    let claims = Blake3CompressionClaims {
        claims: vec![vec![
            Val::from_usize(Blake3CompressionCircuit::GFunction.position()),
            f32(a_in),
            f32(b_in),
            f32(c_in),
            f32(d_in),
            f32(mx_in),
            f32(my_in),
            f32(a_1),
            f32(d_1),
            f32(c_1),
            f32(b_1),
        ]],
    };
    run_test(&claims);

    // 1 compression claim — full end-to-end chain
    let claims = Blake3CompressionClaims {
        claims: vec![
            [
                vec![Val::from_usize(
                    Blake3CompressionCircuit::Compression.position(),
                )],
                state_in.into_iter().map(Val::from_u32).collect(),
                state_out.into_iter().map(Val::from_u32).collect(),
            ]
            .concat(),
        ],
    };
    run_test(&claims);
}

#[test]
fn g_function_test_vector() {
    let a_in = 0x1111_1111u32;
    let b_in = 0x2222_2222u32;
    let c_in = 0x3333_3333u32;
    let d_in = 0x4444_4444u32;
    let mx_in = 0x5555_5555u32;
    let my_in = 0x6666_6666u32;

    let a_0_tmp = a_in.wrapping_add(b_in);
    let a_0 = a_0_tmp.wrapping_add(mx_in);
    let d_0_tmp = d_in ^ a_0;
    let d_0 = d_0_tmp.rotate_right(16);
    let c_0 = c_in.wrapping_add(d_0);
    let b_0_tmp = b_in ^ c_0;
    let b_0 = b_0_tmp.rotate_right(12);

    let a_1_tmp = a_0.wrapping_add(b_0);
    let a_1 = a_1_tmp.wrapping_add(my_in);
    let d_1_tmp = d_0 ^ a_1;
    let d_1 = d_1_tmp.rotate_right(8);
    let c_1 = c_0.wrapping_add(d_1);
    let b_1_tmp = b_0 ^ c_1;
    let b_1 = b_1_tmp.rotate_right(7);

    assert_eq!(a_1, 0xcccc_cccb);
    assert_eq!(b_1, 0x45b6_4444);
    assert_eq!(c_1, 0x06ff_ffff);
    assert_eq!(d_1, 0x0700_0000);
}

#[test]
fn compression_test_vector() {
    let state_in = vec![
        0x0000_0000u32,
        0x0000_1111u32,
        0x0000_2222u32,
        0x0000_3333u32,
        0x0000_4444u32,
        0x0000_5555u32,
        0x0000_6666u32,
        0x0000_7777u32,
        0x0000_8888u32,
        0x0000_9999u32,
        0x0000_aaaau32,
        0x0000_bbbbu32,
        0x0000_ccccu32,
        0x0000_ddddu32,
        0x0000_eeeeu32,
        0x0000_ffffu32,
        0x0000_0000u32,
        0x1111_0000u32,
        0x2222_0000u32,
        0x3333_0000u32,
        0x4444_0000u32,
        0x5555_0000u32,
        0x6666_0000u32,
        0x7777_0000u32,
        0x8888_0000u32,
        0x9999_0000u32,
        0xaaaa_0000u32,
        0xbbbb_0000u32,
        0xcccc_0000u32,
        0xdddd_0000u32,
        0xeeee_0000u32,
        0xffff_0000u32,
    ];

    let mut state = state_in.clone();
    for round_idx in 0..7 {
        for j in 0..8 {
            let a_in = state[A[j]];
            let b_in = state[B[j]];
            let c_in = state[C[j]];
            let d_in = state[D[j]];
            let mx_in = state[MX[j]];
            let my_in = state[MY[j]];

            let a_0 = a_in.wrapping_add(b_in).wrapping_add(mx_in);
            let d_0 = (d_in ^ a_0).rotate_right(16);
            let c_0 = c_in.wrapping_add(d_0);
            let b_0 = (b_in ^ c_0).rotate_right(12);

            let a_1 = a_0.wrapping_add(b_0).wrapping_add(my_in);
            let d_1 = (d_0 ^ a_1).rotate_right(8);
            let c_1 = c_0.wrapping_add(d_1);
            let b_1 = (b_0 ^ c_1).rotate_right(7);

            state[A[j]] = a_1;
            state[B[j]] = b_1;
            state[C[j]] = c_1;
            state[D[j]] = d_1;
        }

        // execute permutation for the 6 first rounds
        if round_idx < 6 {
            let mut permuted = [0; 16];
            for i in 0..16 {
                permuted[i] = state[16 + MSG_PERMUTATION[i]];
            }
            state[16..(16 + 16)].copy_from_slice(&permuted);
        }
    }

    for i in 0..8 {
        state[i] ^= state[i + 8];
        state[i + 8] ^= state_in[i]; // ^chaining_value
    }

    let state_out = state[0..16].to_vec();

    let state_out_expected = vec![
        0xd304_e51cu32,
        0xc2df_34a0u32,
        0x5eba_7f1fu32,
        0x2ab9_650fu32,
        0xd9ce_f159u32,
        0x4e9d_3a6au32,
        0xcac2_e310u32,
        0xc6b9_be7eu32,
        0xad9f_d58au32,
        0x0899_e71bu32,
        0xca51_a599u32,
        0xc3fb_d7c0u32,
        0x751d_2f26u32,
        0x6cd0_ac6bu32,
        0xc58f_3c1du32,
        0xe6d6_5414u32,
    ];

    assert_eq!(state_out, state_out_expected);
}
