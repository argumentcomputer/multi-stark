use multi_stark::p3_field::{PrimeCharacteristicRing, PrimeField64};
use multi_stark::plonkish::gadgets::{ByteGadgets, blake3};
use multi_stark::plonkish::{CircuitBuilder, WitnessError};
use multi_stark::types::Val;
use p3_blake3::Blake3;
use p3_symmetric::CryptographicHasher;

#[test]
fn blake3_matches_native_at_block_and_tree_boundaries() {
    for len in [0, 1, 63, 64, 65, 1023, 1024, 1025, 2048, 2049] {
        let mut builder = CircuitBuilder::<Val>::new();
        let bytes = ByteGadgets::new(&mut builder);
        let input: Vec<_> = (0..len)
            .map(|i| bytes.input(&mut builder, &format!("byte {i}")))
            .collect();
        let output = blake3(&mut builder, &bytes, &input);
        let circuit = builder.finish();
        for seed in [0u8, 197] {
            let message: Vec<_> = (0..len)
                .map(|i| u8::try_from(i % 251).unwrap().wrapping_add(seed))
                .collect();
            let expected: [u8; 32] = Blake3.hash_iter(message.iter().copied());
            let mut witness = circuit.witness();
            for (&wire, &value) in input.iter().zip(&message) {
                witness.set(wire.value(), Val::from_u8(value)).unwrap();
            }
            let assignment = witness.generate().unwrap();
            let actual = output.map(|byte| {
                u8::try_from(assignment.value(byte.value()).unwrap().as_canonical_u64()).unwrap()
            });
            assert_eq!(actual, expected, "length {len}, seed {seed}");
        }
    }
}

#[test]
fn byte_range_and_field_encoding_are_canonical() {
    let mut builder = CircuitBuilder::<Val>::new();
    let bytes = ByteGadgets::new(&mut builder);
    let input = bytes.input(&mut builder, "byte");
    let circuit = builder.finish();
    for x in [256, 511, 65536] {
        let mut witness = circuit.witness();
        witness.set(input.value(), Val::from_u32(x)).unwrap();
        assert!(witness.generate().is_err());
    }
    let mut builder = CircuitBuilder::<Val>::new();
    let bytes = ByteGadgets::new(&mut builder);
    let input = builder.input("field element");
    let encoded = bytes.encode_field64(&mut builder, input);
    let circuit = builder.finish();
    for x in [0, 1, 0xffff_ffff, 1 << 32, Val::ORDER_U64 - 1] {
        let mut witness = circuit.witness();
        witness.set(input, Val::from_u64(x)).unwrap();
        let assignment = witness.generate().unwrap();
        let actual = encoded.map(|v| {
            u8::try_from(assignment.value(v.value()).unwrap().as_canonical_u64()).unwrap()
        });
        assert_eq!(actual, x.to_le_bytes());
    }
    // Model sampled hash bytes, including candidates whose reduction would
    // alias 0 or 1. They must fail, not get accepted modulo Goldilocks p.
    let mut builder = CircuitBuilder::<Val>::new();
    let bytes = ByteGadgets::new(&mut builder);
    let input = std::array::from_fn(|i| bytes.input(&mut builder, &format!("sample[{i}]")));
    bytes.assert_at_most_u64(&mut builder, input, Val::ORDER_U64 - 1);
    let circuit = builder.finish();
    for x in [
        0,
        Val::ORDER_U64 - 1,
        Val::ORDER_U64,
        Val::ORDER_U64 + 1,
        u64::MAX,
    ] {
        let mut witness = circuit.witness();
        for (wire, value) in input.iter().zip(x.to_le_bytes()) {
            witness.set(wire.value(), Val::from_u8(value)).unwrap();
        }
        assert_eq!(
            witness.generate().is_ok(),
            x < Val::ORDER_U64,
            "candidate {x}"
        );
    }
}

#[test]
fn wrong_digest_is_rejected_by_relations_not_just_host_hashing() {
    let mut builder = CircuitBuilder::<Val>::new();
    let bytes = ByteGadgets::new(&mut builder);
    let input = bytes.input(&mut builder, "byte");
    let output = blake3(&mut builder, &bytes, &[input]);
    let expected: [_; 32] = std::array::from_fn(|i| builder.public_input(format!("digest[{i}]")));
    for (byte, expected) in output.iter().zip(expected) {
        builder.assert_equal(byte.value(), expected);
    }
    let circuit = builder.finish();
    let digest: [u8; 32] = Blake3.hash_iter([42]);
    for bad_index in 0..32 {
        let mut witness = circuit.witness();
        witness.set(input.value(), Val::from_u8(42)).unwrap();
        for (i, &wire) in expected.iter().enumerate() {
            witness
                .set(wire, Val::from_u8(digest[i] ^ u8::from(i == bad_index)))
                .unwrap();
        }
        assert!(matches!(
            witness.generate(),
            Err(WitnessError::UnsatisfiedGate { .. })
        ));
    }
}
