//! Independently expected public values of the measured Init root.
use std::{io::Read, path::Path};

pub(crate) const INIT_PUBLIC_WORDS: [u64; 18] = [
    0, 293, 3008464260, 3936129970, 1088962797, 1141056675, 2347219926, 1176003176, 683533119,
    3085659920, 1422858688, 3165160492, 1899567894, 784981652, 376864732, 1915641123, 2034217541,
    2163067803,
];

/// Read the expected statement selected by the environment, or the fixed
/// measured root statement when no file is selected.
pub(crate) fn expected_words() -> Result<[u64; 18], Box<dyn std::error::Error>> {
    let Some(path) = std::env::var_os("MULTI_STARK_INIT_EXPECTED_CLAIMS") else {
        return Ok(INIT_PUBLIC_WORDS);
    };
    expected_words_from_path(Path::new(&path))
}

/// Load one independently supplied verifier statement. The returned words
/// remain the expected statement even if the file or environment later changes.
pub(crate) fn expected_words_from_path(
    path: &Path,
) -> Result<[u64; 18], Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(161);
    std::fs::File::open(path)?
        .take(161)
        .read_to_end(&mut bytes)?;
    decode_expected_words(&bytes)
}

fn decode_expected_words(bytes: &[u8]) -> Result<[u64; 18], Box<dyn std::error::Error>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(words: [u64; 18]) -> Vec<u8> {
        [1, 18]
            .into_iter()
            .chain(words)
            .flat_map(u64::to_le_bytes)
            .collect()
    }

    #[test]
    fn accepts_default_and_other_canonical_statements() {
        assert_eq!(
            decode_expected_words(&encode(INIT_PUBLIC_WORDS)).unwrap(),
            INIT_PUBLIC_WORDS
        );
        let mut words = [0; 18];
        words[1] = 0xffff_ffff_0000_0000;
        words[17] = 42;
        assert_eq!(decode_expected_words(&encode(words)).unwrap(), words);
    }

    #[test]
    fn rejects_truncation_trailing_data_and_wrong_shape() {
        let encoded = encode(INIT_PUBLIC_WORDS);
        for length in 0..encoded.len() {
            assert!(
                decode_expected_words(&encoded[..length]).is_err(),
                "accepted length {length}"
            );
        }
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(decode_expected_words(&trailing).is_err());
        for (offset, values) in [(0, [0, 2, u64::MAX]), (8, [0, 17, 19])] {
            for value in values {
                let mut malformed = encoded.clone();
                malformed[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
                assert!(decode_expected_words(&malformed).is_err());
            }
        }
        let mut big_endian = encoded;
        big_endian[..8].copy_from_slice(&1u64.to_be_bytes());
        big_endian[8..16].copy_from_slice(&18u64.to_be_bytes());
        assert!(decode_expected_words(&big_endian).is_err());
    }

    #[test]
    fn rejects_nonzero_anchor_and_noncanonical_words() {
        let mut words = INIT_PUBLIC_WORDS;
        words[0] = 1;
        assert!(decode_expected_words(&encode(words)).is_err());
        for index in 1..18 {
            for value in [0xffff_ffff_0000_0001, u64::MAX] {
                let mut words = INIT_PUBLIC_WORDS;
                words[index] = value;
                assert!(
                    decode_expected_words(&encode(words)).is_err(),
                    "accepted noncanonical word {index}"
                );
            }
        }
    }

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "multi-stark-expected-claims-{}-{nonce}-{index}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn explicit_sources_produce_independent_owned_statements() {
        let directory = TempDir::new();
        let first_path = directory.0.join("first.bin");
        let second_path = directory.0.join("second.bin");
        let mut second = INIT_PUBLIC_WORDS;
        second[1] += 1;
        std::fs::write(&first_path, encode(INIT_PUBLIC_WORDS)).unwrap();
        std::fs::write(&second_path, encode(second)).unwrap();
        let first = expected_words_from_path(&first_path).unwrap();
        let loaded_second = expected_words_from_path(&second_path).unwrap();
        assert_eq!(first, INIT_PUBLIC_WORDS);
        assert_eq!(loaded_second, second);
        assert_ne!(first, loaded_second);

        std::fs::write(&first_path, encode(second)).unwrap();
        std::fs::remove_file(&second_path).unwrap();
        assert_eq!(first, INIT_PUBLIC_WORDS);
        assert_eq!(loaded_second, second);
        assert_eq!(expected_words_from_path(&first_path).unwrap(), second);
        assert!(expected_words_from_path(&second_path).is_err());

        let mut malformed = encode(INIT_PUBLIC_WORDS);
        malformed.extend([0; 4096]);
        std::fs::write(&first_path, malformed).unwrap();
        assert!(expected_words_from_path(&first_path).is_err());
        std::fs::write(&first_path, [0; 7]).unwrap();
        assert!(expected_words_from_path(&first_path).is_err());
    }
}
