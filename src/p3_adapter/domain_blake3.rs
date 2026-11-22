//! Domain-separated BLAKE3 hashing using BLAKE3's native `derive_key` mode.
//!
//! Prepending a byte tag to a 64-byte Merkle internal node (`left || right`)
//! pushes the input from 64 bytes (1 compression) to 65 bytes (2 compressions),
//! doubling Merkle hashing cost on CPU, CUDA, and recursive verifiers.
//!
//! BLAKE3's built-in `derive_key` mode avoids this completely: a context string
//! is hashed once at compile time with `DERIVE_KEY_CONTEXT` (`1 << 5`) into an
//! 8-word `context_key`, which replaces `IV` as the initial chaining value
//! while setting `DERIVE_KEY_MATERIAL` (`1 << 6`) in the block flags. Every
//! 64-byte Merkle compression remains a single BLAKE3 compression with zero
//! extra message bytes and zero performance overhead.

use p3_symmetric::CryptographicHasher;

/// Standard BLAKE3 initialization vector.
pub(crate) const BLAKE3_IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const ROOT: u32 = 1 << 3;
const DERIVE_KEY_CONTEXT: u32 = 1 << 5;
/// BLAKE3 `DERIVE_KEY_MATERIAL` flag (`1 << 6 = 64`).
pub(crate) const BLAKE3_DERIVE_KEY_MATERIAL: u32 = 1 << 6;

const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

/// Domain-separation context string for v6 Merkle leaf row hashes.
pub const V6_MERKLE_LEAF_CONTEXT: &str = "multi-stark/v6/merkle/leaf";
/// Domain-separation context string for v6 Merkle 2-to-1 internal compressions.
pub const V6_MERKLE_COMPRESS_CONTEXT: &str = "multi-stark/v6/merkle/compress";
/// Domain-separation context string for v6 single-proof Fiat-Shamir transcripts.
pub const V6_TRANSCRIPT_SINGLE_CONTEXT: &str = "multi-stark/v6/transcript/single";
/// Domain-separation context string for v6 batch-proof Fiat-Shamir transcripts.
pub const V6_TRANSCRIPT_BATCH_CONTEXT: &str = "multi-stark/v6/transcript/batch";
/// Domain-separation context string for v6 canonical VK / constraint-graph digests.
pub const V6_VK_CONTEXT: &str = "multi-stark/v6/vk";

const fn g_const(
    mut state: [u32; 16],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    x: u32,
    y: u32,
) -> [u32; 16] {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(x);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(y);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
    state
}

/// Computes the 8-word BLAKE3 `derive_key` context key for a context string of
/// at most 64 bytes at compile time.
const fn derive_context_key(ctx: &str) -> [u32; 8] {
    let bytes = ctx.as_bytes();
    assert!(
        bytes.len() <= 64,
        "BLAKE3 context string must be <= 64 bytes"
    );
    let mut block = [0u8; 64];
    let mut i = 0;
    while i < bytes.len() {
        block[i] = bytes[i];
        i += 1;
    }
    let mut msg = [0u32; 16];
    let mut w = 0;
    while w < 16 {
        msg[w] = u32::from_le_bytes([
            block[4 * w],
            block[4 * w + 1],
            block[4 * w + 2],
            block[4 * w + 3],
        ]);
        w += 1;
    }
    #[allow(clippy::cast_possible_truncation)]
    let block_len = bytes.len() as u32;
    let mut state = [
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        BLAKE3_IV[4],
        BLAKE3_IV[5],
        BLAKE3_IV[6],
        BLAKE3_IV[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        0,
        0,
        block_len,
        CHUNK_START | CHUNK_END | ROOT | DERIVE_KEY_CONTEXT,
    ];
    let mut r = 0;
    while r < 7 {
        let s = MSG_SCHEDULE[r];
        state = g_const(state, 0, 4, 8, 12, msg[s[0]], msg[s[1]]);
        state = g_const(state, 1, 5, 9, 13, msg[s[2]], msg[s[3]]);
        state = g_const(state, 2, 6, 10, 14, msg[s[4]], msg[s[5]]);
        state = g_const(state, 3, 7, 11, 15, msg[s[6]], msg[s[7]]);
        state = g_const(state, 0, 5, 10, 15, msg[s[8]], msg[s[9]]);
        state = g_const(state, 1, 6, 11, 12, msg[s[10]], msg[s[11]]);
        state = g_const(state, 2, 7, 8, 13, msg[s[12]], msg[s[13]]);
        state = g_const(state, 3, 4, 9, 14, msg[s[14]], msg[s[15]]);
        r += 1;
    }
    [
        state[0] ^ state[8],
        state[1] ^ state[9],
        state[2] ^ state[10],
        state[3] ^ state[11],
        state[4] ^ state[12],
        state[5] ^ state[13],
        state[6] ^ state[14],
        state[7] ^ state[15],
    ]
}

const KEY_MERKLE_LEAF: [u32; 8] = derive_context_key(V6_MERKLE_LEAF_CONTEXT);
const KEY_MERKLE_COMPRESS: [u32; 8] = derive_context_key(V6_MERKLE_COMPRESS_CONTEXT);
const KEY_TRANSCRIPT_SINGLE: [u32; 8] = derive_context_key(V6_TRANSCRIPT_SINGLE_CONTEXT);
const KEY_TRANSCRIPT_BATCH: [u32; 8] = derive_context_key(V6_TRANSCRIPT_BATCH_CONTEXT);
const KEY_VK: [u32; 8] = derive_context_key(V6_VK_CONTEXT);

/// Cryptographic domain for BLAKE3 hashing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Blake3Domain {
    /// Standard unkeyed BLAKE3 (`IV`, `flags = 0`), used by legacy `v5`.
    Plain,
    /// Protocol `v6` Merkle leaf row hashing (`derive_key("multi-stark/v6/merkle/leaf")`).
    MerkleLeaf,
    /// Protocol `v6` Merkle internal 2-to-1 compression (`derive_key("multi-stark/v6/merkle/compress")`).
    MerkleCompress,
    /// Protocol `v6` single-proof Fiat-Shamir transcript (`derive_key("multi-stark/v6/transcript/single")`).
    TranscriptSingle,
    /// Protocol `v6` batch-proof Fiat-Shamir transcript (`derive_key("multi-stark/v6/transcript/batch")`).
    TranscriptBatch,
    /// Protocol `v6` canonical verifying-key / constraint-graph digest (`derive_key("multi-stark/v6/vk")`).
    VerifyingKey,
}

impl Blake3Domain {
    /// Returns the `derive_key` context string, or `None` for [`Self::Plain`].
    #[inline]
    pub const fn context(self) -> Option<&'static str> {
        match self {
            Self::Plain => None,
            Self::MerkleLeaf => Some(V6_MERKLE_LEAF_CONTEXT),
            Self::MerkleCompress => Some(V6_MERKLE_COMPRESS_CONTEXT),
            Self::TranscriptSingle => Some(V6_TRANSCRIPT_SINGLE_CONTEXT),
            Self::TranscriptBatch => Some(V6_TRANSCRIPT_BATCH_CONTEXT),
            Self::VerifyingKey => Some(V6_VK_CONTEXT),
        }
    }

    /// Initial 8-word chaining value (`BLAKE3_IV` for [`Self::Plain`], or the
    /// precomputed `derive_key` context key for domain-separated modes).
    #[inline]
    pub const fn key_words(self) -> [u32; 8] {
        match self {
            Self::Plain => BLAKE3_IV,
            Self::MerkleLeaf => KEY_MERKLE_LEAF,
            Self::MerkleCompress => KEY_MERKLE_COMPRESS,
            Self::TranscriptSingle => KEY_TRANSCRIPT_SINGLE,
            Self::TranscriptBatch => KEY_TRANSCRIPT_BATCH,
            Self::VerifyingKey => KEY_VK,
        }
    }

    /// Extra domain flags OR'd into every BLAKE3 compression (`0` for
    /// [`Self::Plain`], [`BLAKE3_DERIVE_KEY_MATERIAL`] for `derive_key` modes).
    #[inline]
    pub const fn domain_flags(self) -> u32 {
        match self {
            Self::Plain => 0,
            Self::MerkleLeaf
            | Self::MerkleCompress
            | Self::TranscriptSingle
            | Self::TranscriptBatch
            | Self::VerifyingKey => BLAKE3_DERIVE_KEY_MATERIAL,
        }
    }

    /// Creates a streaming [`blake3::Hasher`] configured for this domain.
    #[inline]
    pub fn hasher(self) -> blake3::Hasher {
        match self.context() {
            None => blake3::Hasher::new(),
            Some(ctx) => blake3::Hasher::new_derive_key(ctx),
        }
    }

    /// Hashes a byte slice under this domain.
    #[inline]
    pub fn hash_bytes(self, input: &[u8]) -> [u8; 32] {
        let mut h = self.hasher();
        h.update(input);
        h.finalize().into()
    }
}

/// Plonky3 [`CryptographicHasher`] backed by domain-separated BLAKE3.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DomainBlake3 {
    domain: Blake3Domain,
}

impl DomainBlake3 {
    #[inline]
    pub const fn new(domain: Blake3Domain) -> Self {
        Self { domain }
    }

    #[inline]
    pub const fn domain(self) -> Blake3Domain {
        self.domain
    }
}

impl CryptographicHasher<u8, [u8; 32]> for DomainBlake3 {
    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        const BUFLEN: usize = 512;
        let mut hasher = self.domain.hasher();
        p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |buf| {
            hasher.update(buf);
        });
        hasher.finalize().into()
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut hasher = self.domain.hasher();
        for chunk in input {
            hasher.update(chunk);
        }
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_time_context_keys_match_blake3_derive_key() {
        let all_domains = [
            Blake3Domain::Plain,
            Blake3Domain::MerkleLeaf,
            Blake3Domain::MerkleCompress,
            Blake3Domain::TranscriptSingle,
            Blake3Domain::TranscriptBatch,
            Blake3Domain::VerifyingKey,
        ];
        // Test empty, 64-byte single-block, and multi-chunk (>1024 B) inputs.
        let inputs: Vec<Vec<u8>> = vec![
            vec![],
            (0..64u8).collect(),
            (0..2500usize)
                .map(|i| u8::try_from(i.wrapping_mul(37) & 0xFF).unwrap())
                .collect(),
        ];

        for input in &inputs {
            let mut seen = Vec::new();
            for domain in all_domains {
                let got = DomainBlake3::new(domain).hash_slice(input);
                let expected: [u8; 32] = match domain.context() {
                    None => *blake3::hash(input).as_bytes(),
                    Some(ctx) => blake3::derive_key(ctx, input),
                };
                assert_eq!(got, expected, "mismatch for domain {domain:?}");

                // For single-block inputs (<= 64 bytes), also cross-check our
                // compile-time key_words + domain_flags directly via single-block
                // compression against blake3::derive_key.
                if input.len() <= 64 {
                    let mut block = [0u8; 64];
                    block[..input.len()].copy_from_slice(input);
                    let mut msg = [0u32; 16];
                    for (w, &chunk) in msg.iter_mut().zip(block.as_chunks::<4>().0) {
                        *w = u32::from_le_bytes(chunk);
                    }
                    let kv = domain.key_words();
                    let mut state = [
                        kv[0],
                        kv[1],
                        kv[2],
                        kv[3],
                        kv[4],
                        kv[5],
                        kv[6],
                        kv[7],
                        BLAKE3_IV[0],
                        BLAKE3_IV[1],
                        BLAKE3_IV[2],
                        BLAKE3_IV[3],
                        0,
                        0,
                        u32::try_from(input.len()).unwrap(),
                        CHUNK_START | CHUNK_END | ROOT | domain.domain_flags(),
                    ];
                    for s in &MSG_SCHEDULE {
                        state = g_const(state, 0, 4, 8, 12, msg[s[0]], msg[s[1]]);
                        state = g_const(state, 1, 5, 9, 13, msg[s[2]], msg[s[3]]);
                        state = g_const(state, 2, 6, 10, 14, msg[s[4]], msg[s[5]]);
                        state = g_const(state, 3, 7, 11, 15, msg[s[6]], msg[s[7]]);
                        state = g_const(state, 0, 5, 10, 15, msg[s[8]], msg[s[9]]);
                        state = g_const(state, 1, 6, 11, 12, msg[s[10]], msg[s[11]]);
                        state = g_const(state, 2, 7, 8, 13, msg[s[12]], msg[s[13]]);
                        state = g_const(state, 3, 4, 9, 14, msg[s[14]], msg[s[15]]);
                    }
                    let mut out = [0u8; 32];
                    for i in 0..8 {
                        out[4 * i..4 * i + 4]
                            .copy_from_slice(&(state[i] ^ state[i + 8]).to_le_bytes());
                    }
                    assert_eq!(out, expected, "key_words mismatch for {domain:?}");
                }
                seen.push((domain, got));
            }
            // All 6 domains must produce pairwise distinct digests on the same input.
            for i in 0..seen.len() {
                for j in (i + 1)..seen.len() {
                    assert_ne!(
                        seen[i].1, seen[j].1,
                        "domain collision between {:?} and {:?}",
                        seen[i].0, seen[j].0
                    );
                }
            }
        }
    }
}
