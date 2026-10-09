//! Independently expected public values of the measured Init root.
pub(crate) const INIT_PUBLIC_WORDS: [u64; 18] = [
    0, 293, 3008464260, 3936129970, 1088962797, 1141056675, 2347219926, 1176003176, 683533119,
    3085659920, 1422858688, 3165160492, 1899567894, 784981652, 376864732, 1915641123, 2034217541,
    2163067803,
];

/// Read an independently supplied expected root statement. This file is a
/// verifier input, never a claim recovered from the packet being verified.
pub(crate) fn expected_words() -> Result<[u64; 18], Box<dyn std::error::Error>> {
    let Some(path) = std::env::var_os("MULTI_STARK_INIT_EXPECTED_CLAIMS") else {
        return Ok(INIT_PUBLIC_WORDS);
    };
    let bytes = std::fs::read(path)?;
    if bytes.len() != 160
        || u64::from_le_bytes(bytes[..8].try_into()?) != 1
        || u64::from_le_bytes(bytes[8..16].try_into()?) != 18
    {
        return Err("expected one independently supplied 18-word root claim".into());
    }
    let words = std::array::from_fn(|i| {
        u64::from_le_bytes(bytes[16 + 8 * i..24 + 8 * i].try_into().unwrap())
    });
    if words[0] != 0 || words.iter().any(|&word| word >= 0xffff_ffff_0000_0001) {
        return Err("invalid canonical root claim".into());
    }
    Ok(words)
}
