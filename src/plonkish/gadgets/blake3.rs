use crate::traits::PrimeField;

use super::bytes::Word;
use super::{ByteGadgets, ByteValue};
use crate::plonkish::CircuitBuilder;

const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
const ROOT: u32 = 8;

fn mix<F: PrimeField>(
    b: &mut CircuitBuilder<F>,
    bytes: &ByteGadgets,
    s: &mut [Word; 16],
    indices: [usize; 4],
    x: Word,
    y: Word,
) {
    let [a, c, d, e] = indices;
    s[a] = bytes.add(b, s[a], s[c]);
    s[a] = bytes.add(b, s[a], x);
    let xor = bytes.xor(b, s[e], s[a]);
    s[e] = bytes.rotate_right(b, xor, 16);
    s[d] = bytes.add(b, s[d], s[e]);
    let xor = bytes.xor(b, s[c], s[d]);
    s[c] = bytes.rotate_right(b, xor, 12);
    s[a] = bytes.add(b, s[a], s[c]);
    s[a] = bytes.add(b, s[a], y);
    let xor = bytes.xor(b, s[e], s[a]);
    s[e] = bytes.rotate_right(b, xor, 8);
    s[d] = bytes.add(b, s[d], s[e]);
    let xor = bytes.xor(b, s[c], s[d]);
    s[c] = bytes.rotate_right(b, xor, 7);
}

struct Output {
    cv: [Word; 8],
    block: [Word; 16],
    counter: u64,
    len: u32,
    flags: u32,
}

impl Output {
    fn compress<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        bytes: &ByteGadgets,
        root: bool,
    ) -> [Word; 8] {
        let mut state = std::array::from_fn(|i| match i {
            0..8 => self.cv[i],
            8..12 => bytes.word_constant(b, IV[i - 8]),
            12 => bytes.word_constant(b, u32::try_from(self.counter & 0xffff_ffff).unwrap()),
            13 => bytes.word_constant(b, u32::try_from(self.counter >> 32).unwrap()),
            14 => bytes.word_constant(b, self.len),
            _ => bytes.word_constant(b, self.flags | if root { ROOT } else { 0 }),
        });
        let mut message = self.block;
        for _ in 0..7 {
            for (j, indices) in [
                [0, 4, 8, 12],
                [1, 5, 9, 13],
                [2, 6, 10, 14],
                [3, 7, 11, 15],
                [0, 5, 10, 15],
                [1, 6, 11, 12],
                [2, 7, 8, 13],
                [3, 4, 9, 14],
            ]
            .into_iter()
            .enumerate()
            {
                mix(
                    b,
                    bytes,
                    &mut state,
                    indices,
                    message[2 * j],
                    message[2 * j + 1],
                );
            }
            message = PERMUTATION.map(|i| message[i]);
        }
        std::array::from_fn(|i| bytes.xor(b, state[i], state[i + 8]))
    }
}

fn subtree<F: PrimeField>(
    b: &mut CircuitBuilder<F>,
    bytes: &ByteGadgets,
    input: &[ByteValue],
    counter: u64,
) -> Output {
    let iv = IV.map(|x| bytes.word_constant(b, x));
    if input.len() > 1024 {
        let chunks = input.len().div_ceil(1024);
        let left_chunks = 1usize << (usize::BITS - 1 - (chunks - 1).leading_zeros());
        let split = left_chunks * 1024;
        let left = subtree(b, bytes, &input[..split], counter).compress(b, bytes, false);
        let right = subtree(
            b,
            bytes,
            &input[split..],
            counter + u64::try_from(left_chunks).unwrap(),
        )
        .compress(b, bytes, false);
        return Output {
            cv: iv,
            block: std::array::from_fn(|i| if i < 8 { left[i] } else { right[i - 8] }),
            counter: 0,
            len: 64,
            flags: PARENT,
        };
    }
    let zero = bytes.constant(b, 0);
    let blocks = input.len().div_ceil(64).max(1);
    let mut cv = iv;
    for i in 0..blocks {
        let start = i * 64;
        let len = input.len().saturating_sub(start).min(64);
        let block = std::array::from_fn(|word| {
            bytes.word(
                b,
                std::array::from_fn(|j| {
                    let offset = word * 4 + j;
                    if offset < len {
                        input[start + offset]
                    } else {
                        zero
                    }
                }),
            )
        });
        let last = i + 1 == blocks;
        let output = Output {
            cv,
            block,
            counter,
            len: u32::try_from(len).unwrap(),
            flags: if i == 0 { CHUNK_START } else { 0 } | if last { CHUNK_END } else { 0 },
        };
        if last {
            return output;
        }
        cv = output.compress(b, bytes, false);
    }
    unreachable!()
}

/// Standard unkeyed BLAKE3, 32-byte output, with a construction-time fixed
/// input length. Includes chunk counters and binary tree reduction; bytes
/// are constrained witnesses, not constants or trusted hash hints.
pub fn blake3<F: PrimeField>(
    b: &mut CircuitBuilder<F>,
    bytes: &ByteGadgets,
    input: &[ByteValue],
) -> [ByteValue; 32] {
    if b.compact_blake3_enabled() {
        use p3_symmetric::CryptographicHasher;
        let input: Vec<_> = input.iter().map(|v| v.value()).collect();
        let output = b.hint_many("BLAKE3 advice", &input, |values| {
            let bytes: Result<Vec<_>, String> = values
                .iter()
                .map(|&v| {
                    u8::try_from(super::bytes::integer(v)?)
                        .map_err(|_error| "non-byte hash input".into())
                })
                .collect();
            let digest: [u8; 32] = p3_blake3::Blake3.hash_iter(bytes?);
            Ok(digest.map(F::from_u8))
        });
        b.record_hash(input, &output);
        return output.map(|v| bytes.constrain_byte(b, v));
    }
    let output = subtree(b, bytes, input, 0).compress(b, bytes, true);
    let words = output.map(|w| bytes.word_bytes(b, w));
    std::array::from_fn(|i| words[i / 4][i % 4])
}
