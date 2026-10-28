use crate::traits::PrimeField;

use crate::plonkish::{Bool, CircuitBuilder, Table, Value};

/// A constrained byte. Only gadget constructors can create it, so hashing
/// cannot accidentally accept unbounded byte inputs. The nibble decomposition
/// is materialized on demand: a byte selected between two constrained bytes
/// is itself a byte, and consumers that only read the value (the compact
/// BLAKE3 lowering) never pay for its nibbles.
#[derive(Clone, Copy, Debug)]
pub struct ByteValue {
    nibbles: Option<[Value; 2]>,
    value: Value,
}

impl ByteValue {
    pub fn value(self) -> Value {
        self.value
    }
}

#[derive(Clone, Copy)]
pub(super) struct Word {
    nibbles: [Value; 8],
    value: Value,
}

/// Shared small lookup tables for native-field byte/u32 arithmetic. Supports
/// prime fields larger than 2^33, including future large native fields.
pub struct ByteGadgets {
    nibble: Table,
    xor: Table,
    split: Table,
}

pub(super) fn integer<F: PrimeField>(value: F) -> Result<u64, String> {
    value
        .canonical_u64()
        .ok_or_else(|| "expected an integer fitting in 64 bits".into())
}

/// One constrained affine operation, used to pack bounded integer limbs.
fn affine<F: PrimeField>(b: &mut CircuitBuilder<F>, a: Value, c: F, d: Value) -> Value {
    b.affine([a, d], [c, F::ONE], F::ZERO)
}

impl ByteGadgets {
    pub fn bits<F: PrimeField>(&self, b: &mut CircuitBuilder<F>, byte: ByteValue) -> [Bool; 8] {
        let bits: [Value; 8] = b.hint_many("byte bits", &[byte.value], |v| {
            let value = integer(v[0])?;
            Ok(std::array::from_fn(|i| F::from_u64((value >> i) & 1)))
        });
        let bits = bits.map(|bit| b.assert_bool(bit));
        let mut packed = bits[7].value();
        for bit in bits[..7].iter().rev() {
            packed = affine(b, packed, F::TWO, bit.value());
        }
        b.assert_equal(packed, byte.value);
        bits
    }

    pub fn select<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        bit: Bool,
        yes: ByteValue,
        no: ByteValue,
    ) -> ByteValue {
        let nibbles = match (yes.nibbles, no.nibbles) {
            (Some(y), Some(n)) if y == n => Some(y),
            _ => None,
        };
        ByteValue {
            nibbles,
            value: b.select(bit, yes.value, no.value),
        }
    }

    /// The byte's nibbles, decomposing and range-checking them if the byte
    /// was produced without them.
    pub(super) fn nibbles<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        byte: ByteValue,
    ) -> [Value; 2] {
        byte.nibbles.unwrap_or_else(|| {
            let nibbles = self.nibble_hints(b, byte.value);
            let packed = self.pack(b, &nibbles);
            b.assert_equal(byte.value, packed);
            nibbles
        })
    }

    pub fn new<F: PrimeField>(b: &mut CircuitBuilder<F>) -> Self {
        assert!(
            F::prime_order_exceeds(1usize << 33),
            "u32 gadgets need a prime field larger than 2^33"
        );
        Self {
            nibble: b.fixed_table("nibble", (0..16).map(|n| vec![F::from_u8(n)]).collect()),
            xor: b.fixed_table(
                "nibble XOR",
                (0..16)
                    .flat_map(|a| {
                        (0..16).map(move |c| vec![F::from_u8(a), F::from_u8(c), F::from_u8(a ^ c)])
                    })
                    .collect(),
            ),
            split: b.fixed_table(
                "nibble split at bit 3",
                (0..16)
                    .map(|n| vec![F::from_u8(n), F::from_u8(n & 7), F::from_u8(n >> 3)])
                    .collect(),
            ),
        }
    }

    fn nibble_hints<F: PrimeField, const N: usize>(
        &self,
        b: &mut CircuitBuilder<F>,
        value: Value,
    ) -> [Value; N] {
        assert!(N <= 16);
        let nibbles = b.hint_many_pure("nibbles", &[value], |v| {
            let value = integer(v[0])?;
            Ok(std::array::from_fn(|i| {
                F::from_u64((value >> (i * 4)) & 15)
            }))
        });
        for n in nibbles {
            b.lookup(self.nibble, &[n]);
        }
        nibbles
    }

    fn pack<F: PrimeField>(&self, b: &mut CircuitBuilder<F>, nibbles: &[Value]) -> Value {
        let mut iter = nibbles.iter().rev();
        let mut result = *iter.next().expect("nonempty limb list");
        for &n in iter {
            result = affine(b, result, F::from_u8(16), n);
        }
        result
    }

    pub fn constrain_byte<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        value: Value,
    ) -> ByteValue {
        let nibbles = self.nibble_hints(b, value);
        let packed = self.pack(b, &nibbles);
        b.assert_equal(value, packed);
        ByteValue {
            nibbles: Some(nibbles),
            value,
        }
    }

    pub fn input<F: PrimeField>(&self, b: &mut CircuitBuilder<F>, name: &str) -> ByteValue {
        let value = b.input(name);
        self.constrain_byte(b, value)
    }

    pub fn constant<F: PrimeField>(&self, b: &mut CircuitBuilder<F>, value: u8) -> ByteValue {
        ByteValue {
            nibbles: Some([
                b.constant(F::from_u8(value & 15)),
                b.constant(F::from_u8(value >> 4)),
            ]),
            value: b.constant(F::from_u8(value)),
        }
    }

    fn byte_from_nibbles<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        nibbles: [Value; 2],
    ) -> ByteValue {
        ByteValue {
            value: self.pack(b, &nibbles),
            nibbles: Some(nibbles),
        }
    }

    pub(super) fn word<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        bytes: [ByteValue; 4],
    ) -> Word {
        let pairs = bytes.map(|byte| self.nibbles(b, byte));
        let nibbles = std::array::from_fn(|i| pairs[i / 2][i % 2]);
        Word {
            value: self.pack(b, &nibbles),
            nibbles,
        }
    }

    pub(super) fn word_constant<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        value: u32,
    ) -> Word {
        Word {
            value: b.constant(F::from_u32(value)),
            nibbles: std::array::from_fn(|i| b.constant(F::from_u32((value >> (4 * i)) & 15))),
        }
    }

    pub(super) fn word_bytes<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        word: Word,
    ) -> [ByteValue; 4] {
        std::array::from_fn(|i| {
            self.byte_from_nibbles(b, [word.nibbles[2 * i], word.nibbles[2 * i + 1]])
        })
    }

    pub(super) fn add<F: PrimeField>(&self, b: &mut CircuitBuilder<F>, a: Word, c: Word) -> Word {
        let sum = b.add(a.value, c.value);
        let [low, carry] = b.hint_many("u32 sum and carry", &[sum], |v| {
            let sum = integer(v[0])?;
            Ok([F::from_u64(sum & 0xffff_ffff), F::from_u64(sum >> 32)])
        });
        b.assert_bool(carry);
        b.constrain_gate(
            [sum, carry, low],
            [F::ZERO, F::ONE, -F::from_u64(1 << 32), F::NEG_ONE, F::ZERO],
        );
        let nibbles = self.nibble_hints(b, low);
        let packed = self.pack(b, &nibbles);
        b.assert_equal(low, packed);
        Word {
            value: low,
            nibbles,
        }
    }

    pub(super) fn xor<F: PrimeField>(&self, b: &mut CircuitBuilder<F>, a: Word, c: Word) -> Word {
        let nibbles = std::array::from_fn(|i| {
            let out = b.hint("nibble XOR", &[a.nibbles[i], c.nibbles[i]], |v| {
                Ok(F::from_u64(integer(v[0])? ^ integer(v[1])?))
            });
            b.lookup(self.xor, &[a.nibbles[i], c.nibbles[i], out]);
            out
        });
        Word {
            value: self.pack(b, &nibbles),
            nibbles,
        }
    }

    pub(super) fn rotate_right<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        a: Word,
        bits: usize,
    ) -> Word {
        let shifted: [Value; 8] = std::array::from_fn(|i| a.nibbles[(i + bits / 4) % 8]);
        let nibbles = if bits.is_multiple_of(4) {
            shifted
        } else {
            assert_eq!(bits % 4, 3, "only BLAKE3 rotation amounts are supported");
            let split: [[Value; 2]; 8] = shifted.map(|n| {
                let [lo, hi] = b.hint_many("split nibble", &[n], |v| {
                    let n = integer(v[0])?;
                    Ok([F::from_u64(n & 7), F::from_u64(n >> 3)])
                });
                b.lookup(self.split, &[n, lo, hi]);
                [lo, hi]
            });
            std::array::from_fn(|i| affine(b, split[(i + 1) % 8][0], F::TWO, split[i][1]))
        };
        Word {
            value: self.pack(b, &nibbles),
            nibbles,
        }
    }

    /// Enforce the integer inequality `bytes <= bound` using bounded
    /// subtraction limbs. No field reduction is allowed to hide overflow.
    pub fn assert_at_most_u64<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        bytes: [ByteValue; 8],
        bound: u64,
    ) {
        if bound == 0xffff_ffff_0000_0000 {
            let (high_max, low_sum) = self.goldilocks_boundary(b, &bytes);
            b.constrain_gate(
                [high_max.value(), low_sum, low_sum],
                [F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
            );
            return;
        }
        let bounded = self.is_at_most_u64(b, bytes, bound);
        let one = b.constant(F::ONE);
        b.assert_equal(bounded.value(), one);
    }

    /// Integer comparison, including noncanonical field encodings. The
    /// returned boolean is constrained by nibble subtraction, not a hint.
    pub fn is_at_most_u64<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        bytes: [ByteValue; 8],
        bound: u64,
    ) -> Bool {
        if bound == 0xffff_ffff_0000_0000 {
            let (high_max, low_sum) = self.goldilocks_boundary(b, &bytes);
            let invalid = b.mul(high_max.value(), low_sum);
            return b.is_zero(invalid);
        }
        let zero = b.constant(F::ZERO);
        let mut borrow = zero;
        let pairs: Vec<[Value; 2]> = bytes.iter().map(|&byte| self.nibbles(b, byte)).collect();
        for i in 0..16 {
            let x = pairs[i / 2][i % 2];
            let digit = (bound >> (4 * i)) & 15;
            let next = b.hint("comparison borrow", &[x, borrow], move |v| {
                Ok(F::from_bool(integer(v[0])? + integer(v[1])? > digit))
            });
            b.assert_bool(next);
            let diff = b.hint("comparison difference", &[x, borrow], move |v| {
                Ok(F::from_u64(
                    (digit + 16 - integer(v[0])? - integer(v[1])?) & 15,
                ))
            });
            b.lookup(self.nibble, &[diff]);
            let lhs = b.add(x, borrow);
            let lhs = b.add(lhs, diff);
            b.constrain_gate(
                [lhs, next, zero],
                [
                    F::ZERO,
                    F::ONE,
                    -F::from_u8(16),
                    F::ZERO,
                    -F::from_u64(digit),
                ],
            );
            borrow = next;
        }
        let within = b.affine([borrow, zero], [-F::ONE, F::ZERO], F::ONE);
        b.assert_bool(within)
    }

    fn goldilocks_boundary<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        bytes: &[ByteValue; 8],
    ) -> (Bool, Value) {
        // Each sum is in 0..=120. The upper word is all ones iff its
        // nibble sum is 120; the lower word is zero iff its sum is zero.
        let pairs: Vec<[Value; 2]> = bytes.iter().map(|&byte| self.nibbles(b, byte)).collect();
        let sums = [0, 4].map(|start| {
            let nibbles = pairs[start..start + 4].iter().flatten().copied();
            let zero = b.constant(F::ZERO);
            nibbles.fold(zero, |sum, nibble| b.add(sum, nibble))
        });
        let max = b.constant(F::from_u8(120));
        let difference = b.sub(sums[1], max);
        (b.is_zero(difference), sums[0])
    }

    /// Pack eight little-endian bytes as a field element. The caller must
    /// separately enforce canonicality when these bytes encode a field value.
    pub fn pack_u64<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        bytes: [ByteValue; 8],
    ) -> Value {
        let mut packed = bytes[7].value;
        for byte in bytes[..7].iter().rev() {
            packed = affine(b, packed, F::from_u16(256), byte.value);
        }
        packed
    }

    /// Canonical encoding for a prime field whose modulus fits in u64.
    /// Larger native fields need their own encoding, not this 8-byte API.
    pub fn encode_field64<F: PrimeField>(
        &self,
        b: &mut CircuitBuilder<F>,
        value: Value,
    ) -> [ByteValue; 8] {
        let order = F::modulus_u64().expect("field encoding is limited to 64 bits");
        let nibbles: [Value; 16] = self.nibble_hints(b, value);
        let bytes = std::array::from_fn(|i| {
            self.byte_from_nibbles(b, [nibbles[2 * i], nibbles[2 * i + 1]])
        });
        self.assert_at_most_u64(b, bytes, order - 1);
        let packed = self.pack(b, &nibbles);
        b.assert_equal(value, packed);
        bytes
    }
}
