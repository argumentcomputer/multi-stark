#[cfg(test)]
mod tests {
    use crate::builder::symbolic::{preprocessed_var, var};
    use crate::chips::SymbExpr;
    use crate::lookup::{Lookup, LookupAir};
    use crate::system::{System, SystemWitness};
    use crate::types::{CommitmentParameters, FriParameters, Val};
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use std::array;

    const BYTE_VALUES_NUM: usize = 256;

    // Preprocessed columns are: [A, B, A or B], where A and B are bytes
    const PREPROCESSED_TRACE_WIDTH: usize = 3;

    // multiplicity, byte0, byte1, bytes2, byte3
    const U32_FROM_LE_BYTES_TRACE_WIDTH: usize = 5;

    // Main trace consists of multiplicity for 'u8_or' and 'range_check' operations
    const U8_OR_PAIR_RANGE_CHECK_TRACE_WIDTH: usize = 2;

    // multiplicity, left0, left1, left2, left3, right0, right1, right2, right3, or0, or1, or2, or3
    const U32_OR_TRACE_WIDTH: usize = 13;

    // multiplicity, in_byte0, in_byte1, in_byte2, in_byte3, in_byte4, in_byte5, in_byte6, in_byte7
    const U64_SHIFT_32_TRACE_WIDTH: usize = 9;

    enum UtilityChip {
        U32FromLeBytes,
        U32Or,
        U8Or,
        U8PairRangeCheck,
        U64ShiftRight32AsU32,
        U64AsU32,
    }

    impl UtilityChip {
        fn position(&self) -> usize {
            match self {
                Self::U8Or => 0,
                Self::U8PairRangeCheck => 1,
                Self::U32FromLeBytes => 2,
                Self::U32Or => 3,
                Self::U64ShiftRight32AsU32 => 4,
                Self::U64AsU32 => 5,
            }
        }
    }

    impl<F: Field> BaseAir<F> for UtilityChip {
        fn width(&self) -> usize {
            match self {
                Self::U32FromLeBytes => U32_FROM_LE_BYTES_TRACE_WIDTH,
                Self::U32Or => U32_OR_TRACE_WIDTH,
                Self::U8Or | Self::U8PairRangeCheck => U8_OR_PAIR_RANGE_CHECK_TRACE_WIDTH,
                Self::U64ShiftRight32AsU32 | Self::U64AsU32 => U64_SHIFT_32_TRACE_WIDTH,
            }
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            match self {
                Self::U8Or | Self::U8PairRangeCheck => {
                    let bytes: [u8; BYTE_VALUES_NUM] =
                        array::from_fn(|idx| u8::try_from(idx).unwrap());
                    let mut trace_values = Vec::with_capacity(
                        BYTE_VALUES_NUM * BYTE_VALUES_NUM * PREPROCESSED_TRACE_WIDTH,
                    );
                    for i in 0..BYTE_VALUES_NUM {
                        for j in 0..BYTE_VALUES_NUM {
                            trace_values.push(F::from_u8(bytes[i]));
                            trace_values.push(F::from_u8(bytes[j]));
                            trace_values.push(F::from_u8(bytes[i] | bytes[j]));
                        }
                    }
                    Some(RowMajorMatrix::new(trace_values, PREPROCESSED_TRACE_WIDTH))
                }
                Self::U32FromLeBytes
                | Self::U32Or
                | Self::U64ShiftRight32AsU32
                | Self::U64AsU32 => None,
            }
        }
    }

    impl<AB> Air<AB> for UtilityChip
    where
        AB: AirBuilder,
        AB::Var: Copy,
        AB::F: Field,
    {
        fn eval(&self, _builder: &mut AB) {
            match self {
                Self::U32FromLeBytes
                | Self::U32Or
                | Self::U8Or
                | Self::U8PairRangeCheck
                | Self::U64ShiftRight32AsU32
                | Self::U64AsU32 => {}
            }
        }
    }

    impl UtilityChip {
        fn lookups(&self) -> Vec<Lookup<SymbExpr>> {
            let u8_or_idx = Self::U8Or.position();
            let u8_pair_range_check_idx = Self::U8PairRangeCheck.position();
            let u32_from_le_bytes_idx = Self::U32FromLeBytes.position();
            let u32_or_idx = Self::U32Or.position();
            let u64_shift_right_32_idx = Self::U64ShiftRight32AsU32.position();
            let u64_shift_left_32_idx = Self::U64AsU32.position();

            match self {
                Self::U32FromLeBytes => {
                    vec![
                        Lookup::pull(
                            var(0),
                            vec![
                                SymbExpr::from_usize(u32_from_le_bytes_idx),
                                var(1),
                                var(2),
                                var(3),
                                var(4),
                                var(1)
                                    + var(2) * SymbExpr::from_u32(256)
                                    + var(3) * SymbExpr::from_u32(256 * 256)
                                    + var(4) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u8_pair_range_check_idx),
                                var(1),
                                var(2),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u8_pair_range_check_idx),
                                var(3),
                                var(4),
                            ],
                        ),
                    ]
                }
                Self::U32Or => {
                    let mut lookups = vec![Lookup::pull(
                        var(0),
                        vec![
                            SymbExpr::from_usize(u32_or_idx),
                            var(1)
                                + var(2) * SymbExpr::from_u32(256)
                                + var(3) * SymbExpr::from_u32(256 * 256)
                                + var(4) * SymbExpr::from_u32(256 * 256 * 256),
                            var(5)
                                + var(6) * SymbExpr::from_u32(256)
                                + var(7) * SymbExpr::from_u32(256 * 256)
                                + var(8) * SymbExpr::from_u32(256 * 256 * 256),
                            var(9)
                                + var(10) * SymbExpr::from_u32(256)
                                + var(11) * SymbExpr::from_u32(256 * 256)
                                + var(12) * SymbExpr::from_u32(256 * 256 * 256),
                        ],
                    )];

                    lookups.extend((0..6).map(|i| {
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u8_pair_range_check_idx),
                                var(i + 1),
                                var(i + 7),
                            ],
                        )
                    }));
                    lookups
                }
                Self::U8Or | Self::U8PairRangeCheck => {
                    vec![
                        Lookup::pull(
                            var(0),
                            vec![
                                SymbExpr::from_usize(u8_or_idx),
                                preprocessed_var(0),
                                preprocessed_var(1),
                                preprocessed_var(2),
                            ],
                        ),
                        Lookup::pull(
                            var(1),
                            vec![
                                SymbExpr::from_usize(u8_pair_range_check_idx),
                                preprocessed_var(0),
                                preprocessed_var(1),
                            ],
                        ),
                    ]
                }
                Self::U64ShiftRight32AsU32 => {
                    let mut lookups = vec![Lookup::pull(
                        var(0),
                        vec![
                            SymbExpr::from_usize(u64_shift_right_32_idx),
                            var(1)
                                + var(2) * SymbExpr::from_u64(256)
                                + var(3) * SymbExpr::from_u64(256 * 256)
                                + var(4) * SymbExpr::from_u64(256 * 256 * 256)
                                + var(5) * SymbExpr::from_u64(256 * 256 * 256 * 256)
                                + var(6) * SymbExpr::from_u64(256 * 256 * 256 * 256 * 256)
                                + var(7) * SymbExpr::from_u64(256 * 256 * 256 * 256 * 256 * 256)
                                + var(8)
                                    * SymbExpr::from_u64(256 * 256 * 256 * 256 * 256 * 256 * 256), // we are ok here, since Goldilock has 64 bits
                            var(5)
                                + var(6) * SymbExpr::from_u64(256)
                                + var(7) * SymbExpr::from_u64(256 * 256)
                                + var(8) * SymbExpr::from_u64(256 * 256 * 256), // 32 bits shifting to the right means that 4 least significant bytes are simply ignored
                        ],
                    )];

                    lookups.extend((0..4).map(|i| {
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u8_pair_range_check_idx),
                                var(i + 1),
                                var(i + 5),
                            ],
                        )
                    }));

                    lookups
                }

                Self::U64AsU32 => {
                    let mut lookups = vec![Lookup::pull(
                        var(0),
                        vec![
                            SymbExpr::from_usize(u64_shift_left_32_idx),
                            var(1)
                                + var(2) * SymbExpr::from_u64(256)
                                + var(3) * SymbExpr::from_u64(256 * 256)
                                + var(4) * SymbExpr::from_u64(256 * 256 * 256)
                                + var(5) * SymbExpr::from_u64(256 * 256 * 256 * 256)
                                + var(6) * SymbExpr::from_u64(256 * 256 * 256 * 256 * 256)
                                + var(7) * SymbExpr::from_u64(256 * 256 * 256 * 256 * 256 * 256)
                                + var(8)
                                    * SymbExpr::from_u64(256 * 256 * 256 * 256 * 256 * 256 * 256), // we are ok here, since Goldilock has 64 bits
                            var(1)
                                + var(2) * SymbExpr::from_u64(256)
                                + var(3) * SymbExpr::from_u64(256 * 256)
                                + var(4) * SymbExpr::from_u64(256 * 256 * 256), // u64 as u32 means that we do 32 bits shifting to the left and 4 most significant bytes are simply ignored
                        ],
                    )];

                    lookups.extend((0..4).map(|i| {
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u8_pair_range_check_idx),
                                var(i + 1),
                                var(i + 5),
                            ],
                        )
                    }));

                    lookups
                }
            }
        }
    }

    struct UtilityChipClaims {
        claims: Vec<Vec<Val>>,
    }

    impl UtilityChipClaims {
        fn witness(
            &self,
            system: &System<UtilityChip>,
        ) -> (Vec<RowMajorMatrix<Val>>, SystemWitness) {
            let mut byte_or_values_from_claims = vec![];
            let mut byte_range_check_values_from_claims = vec![];
            let mut u32_from_le_bytes_values_from_claims = vec![];
            let mut u32_or_values_from_claims = vec![];
            let mut u64_shift_right_32_values_from_claims = vec![];
            let mut u64_to_u32_values_from_claims = vec![];

            for claim in self.claims.clone() {
                // we should have at least chip index
                assert!(!claim.is_empty(), "wrong claim format");
                match claim[0].as_canonical_u64() {
                    0u64 => {
                        // This is our U8Or claim. We should have chip_idx, A, B, A or B (where A, B are bytes)
                        assert_eq!(claim.len(), 4, "[U8Or] wrong claim format");
                        byte_or_values_from_claims.push((claim[1], claim[2], claim[3]));
                    }
                    1u64 => {
                        /* This is our U8PairRangeCheck claim. We should have chip_idx, A, B */

                        assert_eq!(claim.len(), 3, "[U8PairRangeCheck] wrong claim format");
                        byte_range_check_values_from_claims.push((claim[1], claim[2]));
                    }
                    2u64 => {
                        // this is our u32_from_le_bytes chip. We should have chip_idx, byte0, byte1, byte2, byte3, u32
                        assert_eq!(claim.len(), 6);

                        let byte0_val = u8::try_from(claim[1].as_canonical_u64()).unwrap();
                        let byte1_val = u8::try_from(claim[2].as_canonical_u64()).unwrap();
                        let byte2_val = u8::try_from(claim[3].as_canonical_u64()).unwrap();
                        let byte3_val = u8::try_from(claim[4].as_canonical_u64()).unwrap();

                        let u32_val = u32::try_from(claim[5].as_canonical_u64()).unwrap();

                        u32_from_le_bytes_values_from_claims
                            .push((byte0_val, byte1_val, byte2_val, byte3_val, u32_val));
                    }
                    3u64 => {
                        // this is our u32_or chip. We should have: chip_idx, left, right, or
                        assert_eq!(claim.len(), 4);

                        let left = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let right = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                        let or = u32::try_from(claim[3].as_canonical_u64()).unwrap();

                        u32_or_values_from_claims.push((left, right, or));
                    }

                    4u64 => {
                        // this is our u64_shift_right_32. We should have: chip_idx, u64 (input), u32 (output)
                        assert_eq!(claim.len(), 3);

                        let u64_val = claim[1].as_canonical_u64();
                        let shifted = u32::try_from(claim[2].as_canonical_u64()).unwrap();

                        u64_shift_right_32_values_from_claims.push((u64_val, shifted));
                    }
                    5u64 => {
                        // this is our u64_to_u32. We should have: chip_idx, u64 (input), u32 (output)
                        assert_eq!(claim.len(), 3);

                        let u64_val = claim[1].as_canonical_u64();
                        let shifted = u32::try_from(claim[2].as_canonical_u64()).unwrap();

                        u64_to_u32_values_from_claims.push((u64_val, shifted));
                    }

                    _ => {
                        panic!("unsupported chip")
                    }
                }
            }

            fn u64_to_u32(
                values_from_claims: &Vec<(u64, u32)>,
                range_check_values: &mut Vec<(Val, Val)>,
                shift_right: bool,
            ) -> RowMajorMatrix<Val> {
                let mut u64_to_u32_trace_values =
                    Vec::<Val>::with_capacity(values_from_claims.len());
                if values_from_claims.is_empty() {
                    u64_to_u32_trace_values = Val::zero_vec(U64_SHIFT_32_TRACE_WIDTH);

                    range_check_values.push((Val::ZERO, Val::ZERO));
                    range_check_values.push((Val::ZERO, Val::ZERO));
                    range_check_values.push((Val::ZERO, Val::ZERO));
                    range_check_values.push((Val::ZERO, Val::ZERO));
                } else {
                    for (u64_val, shifted) in values_from_claims {
                        let computed: u32 = if shift_right {
                            u32::try_from(*u64_val >> 32).unwrap()
                        } else {
                            *u64_val as u32
                        };
                        debug_assert_eq!(computed, *shifted);

                        let u64_bytes: [u8; 8] = u64_val.to_le_bytes();
                        u64_to_u32_trace_values.push(Val::ONE); // multiplicity
                        u64_to_u32_trace_values
                            .extend_from_slice(u64_bytes.map(Val::from_u8).as_slice());

                        range_check_values
                            .push((Val::from_u8(u64_bytes[0]), Val::from_u8(u64_bytes[4])));
                        range_check_values
                            .push((Val::from_u8(u64_bytes[1]), Val::from_u8(u64_bytes[5])));
                        range_check_values
                            .push((Val::from_u8(u64_bytes[2]), Val::from_u8(u64_bytes[6])));
                        range_check_values
                            .push((Val::from_u8(u64_bytes[3]), Val::from_u8(u64_bytes[7])));
                    }
                }
                let mut u64_to_u32_trace =
                    RowMajorMatrix::new(u64_to_u32_trace_values, U64_SHIFT_32_TRACE_WIDTH);
                let height = u64_to_u32_trace.height().next_power_of_two();
                let zero_rows = height - u64_to_u32_trace.height();
                for _ in 0..zero_rows {
                    range_check_values.push((Val::ZERO, Val::ZERO));
                    range_check_values.push((Val::ZERO, Val::ZERO));
                    range_check_values.push((Val::ZERO, Val::ZERO));
                    range_check_values.push((Val::ZERO, Val::ZERO));
                }
                u64_to_u32_trace.pad_to_height(height, Val::ZERO);
                u64_to_u32_trace
            }

            // u64_to_32
            let u64_to_u32_trace = u64_to_u32(
                &u64_to_u32_values_from_claims,
                &mut byte_range_check_values_from_claims,
                false,
            );

            // u64_shift_right_32
            let u64_shift_right_32_trace = u64_to_u32(
                &u64_shift_right_32_values_from_claims,
                &mut byte_range_check_values_from_claims,
                true,
            );

            // // u64_shift_right_32
            // let mut u64_shift_right_32_trace_values =
            //     Vec::<Val>::with_capacity(u64_shift_right_32_values_from_claims.len());
            // if u64_shift_right_32_values_from_claims.is_empty() {
            //     u64_shift_right_32_trace_values = Val::zero_vec(U64_SHIFT_32_TRACE_WIDTH);
            //
            //     byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            //     byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            //     byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            //     byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            // } else {
            //     for (u64_val, shifted) in u64_shift_right_32_values_from_claims {
            //         let computed = (u64_val >> 32) as u32;
            //         debug_assert_eq!(computed, shifted);
            //
            //         let u64_bytes: [u8; 8] = u64_val.to_le_bytes();
            //         u64_shift_right_32_trace_values.push(Val::ONE); // multiplicity
            //         u64_shift_right_32_trace_values
            //             .extend_from_slice(u64_bytes.map(Val::from_u8).as_slice());
            //
            //         byte_range_check_values_from_claims
            //             .push((Val::from_u8(u64_bytes[0]), Val::from_u8(u64_bytes[4])));
            //         byte_range_check_values_from_claims
            //             .push((Val::from_u8(u64_bytes[1]), Val::from_u8(u64_bytes[5])));
            //         byte_range_check_values_from_claims
            //             .push((Val::from_u8(u64_bytes[2]), Val::from_u8(u64_bytes[6])));
            //         byte_range_check_values_from_claims
            //             .push((Val::from_u8(u64_bytes[3]), Val::from_u8(u64_bytes[7])));
            //     }
            // }
            // let mut u64_shift_right_32_trace =
            //     RowMajorMatrix::new(u64_shift_right_32_trace_values, U64_SHIFT_32_TRACE_WIDTH);
            // let height = u64_shift_right_32_trace.height().next_power_of_two();
            // let zero_rows = height - u64_shift_right_32_trace.height();
            // for _ in 0..zero_rows {
            //     byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            //     byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            //     byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            //     byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            // }
            // u64_shift_right_32_trace.pad_to_height(height, Val::ZERO);

            // u32_or
            let mut u32_or_trace_values =
                Vec::<Val>::with_capacity(u32_or_values_from_claims.len());
            if u32_or_values_from_claims.is_empty() {
                u32_or_trace_values = Val::zero_vec(U32_OR_TRACE_WIDTH);

                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            } else {
                for (left, right, or) in u32_or_values_from_claims {
                    let computed = left | right;
                    debug_assert_eq!(or, computed);

                    let left_bytes: [u8; 4] = left.to_le_bytes();
                    let right_bytes: [u8; 4] = right.to_le_bytes();
                    let or_bytes: [u8; 4] = or.to_le_bytes();

                    u32_or_trace_values.push(Val::ONE); // multiplicity
                    u32_or_trace_values.extend_from_slice(left_bytes.map(Val::from_u8).as_slice());
                    u32_or_trace_values.extend_from_slice(right_bytes.map(Val::from_u8).as_slice());
                    u32_or_trace_values.extend_from_slice(or_bytes.map(Val::from_u8).as_slice());

                    /* we send decomposed bytes to U8PairRangeCheck chip, relying on lookup constraining */

                    byte_range_check_values_from_claims
                        .push((Val::from_u8(left_bytes[0]), Val::from_u8(right_bytes[2])));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(left_bytes[1]), Val::from_u8(right_bytes[3])));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(left_bytes[2]), Val::from_u8(or_bytes[0])));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(left_bytes[3]), Val::from_u8(or_bytes[1])));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(right_bytes[0]), Val::from_u8(or_bytes[2])));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(right_bytes[1]), Val::from_u8(or_bytes[3])));
                }
            }
            let mut u32_or_trace = RowMajorMatrix::new(u32_or_trace_values, U32_OR_TRACE_WIDTH);
            let height = u32_or_trace.height().next_power_of_two();
            let zero_rows = height - u32_or_trace.height();
            for _ in 0..zero_rows {
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            }
            u32_or_trace.pad_to_height(height, Val::ZERO);

            // u32_from_le
            let mut u32_from_le_bytes_trace_values =
                Vec::<Val>::with_capacity(u32_from_le_bytes_values_from_claims.len());
            if u32_from_le_bytes_values_from_claims.is_empty() {
                u32_from_le_bytes_trace_values = Val::zero_vec(U32_FROM_LE_BYTES_TRACE_WIDTH);

                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            } else {
                for (byte0, byte1, byte2, byte3, u32_val) in u32_from_le_bytes_values_from_claims {
                    let bytes: [u8; 4] = [byte0, byte1, byte2, byte3];
                    let computed = u32::from_le_bytes(bytes);
                    debug_assert_eq!(u32_val, computed);

                    u32_from_le_bytes_trace_values.push(Val::ONE); // multiplicity
                    u32_from_le_bytes_trace_values
                        .extend_from_slice(bytes.map(Val::from_u8).as_slice());

                    /* we send decomposed bytes to U8PairRangeCheck chip, relying on lookup constraining */

                    byte_range_check_values_from_claims
                        .push((Val::from_u8(byte0), Val::from_u8(byte1)));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(byte2), Val::from_u8(byte3)));
                }
            }

            let mut u32_from_le_bytes_trace = RowMajorMatrix::new(
                u32_from_le_bytes_trace_values,
                U32_FROM_LE_BYTES_TRACE_WIDTH,
            );
            let height = u32_from_le_bytes_trace.height().next_power_of_two();
            let zero_rows = height - u32_from_le_bytes_trace.height();
            for _ in 0..zero_rows {
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            }
            u32_from_le_bytes_trace.pad_to_height(height, Val::ZERO);

            // finally build U8Or / U8PairRangeCheck trace (columns: multiplicity_u8_or, multiplicity_pair_range_check)
            // since this it "lowest-level" trace, its multiplicities could be updated by other chips previously
            let mut u8_or_range_check_trace_values = Vec::<Val>::with_capacity(
                BYTE_VALUES_NUM * BYTE_VALUES_NUM * U8_OR_PAIR_RANGE_CHECK_TRACE_WIDTH,
            );
            for i in 0..BYTE_VALUES_NUM {
                for j in 0..BYTE_VALUES_NUM {
                    let mut multiplicity_u8_or = Val::ZERO;
                    let mut multiplicity_u8_pair_range_check = Val::ZERO;

                    for vals in byte_or_values_from_claims.clone() {
                        if vals.0 == Val::from_usize(i)
                            && vals.1 == Val::from_usize(j)
                            && vals.2 == Val::from_usize(i | j)
                        {
                            multiplicity_u8_or += Val::ONE;
                        }
                    }

                    for vals in byte_range_check_values_from_claims.clone() {
                        if vals.0 == Val::from_usize(i) && vals.1 == Val::from_usize(j) {
                            multiplicity_u8_pair_range_check += Val::ONE;
                        }
                    }

                    u8_or_range_check_trace_values.push(multiplicity_u8_or);
                    u8_or_range_check_trace_values.push(multiplicity_u8_pair_range_check);
                }
            }

            let traces = vec![
                RowMajorMatrix::new(
                    u8_or_range_check_trace_values,
                    U8_OR_PAIR_RANGE_CHECK_TRACE_WIDTH,
                ),
                // range check trace is entirely preprocessed, so it is "free"
                u32_from_le_bytes_trace,
                u32_or_trace,
                u64_shift_right_32_trace,
                u64_to_u32_trace,
            ];

            (traces.clone(), SystemWitness::from_stage_1(traces, system))
        }
    }

    #[test]
    fn test_u32_to_le_bytes() {
        let byte0 = 0xfe;
        let byte1 = 0xac;
        let byte2 = 0x68;
        let byte3 = 0x01;
        let u32_val = u32::from_le_bytes([byte0, byte1, byte2, byte3]);

        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let u8_circuit = LookupAir::new(UtilityChip::U8Or, UtilityChip::U8Or.lookups());
        let u32_or_circuit = LookupAir::new(UtilityChip::U32Or, UtilityChip::U32Or.lookups());
        let u32_from_le_bytes_circuit = LookupAir::new(
            UtilityChip::U32FromLeBytes,
            UtilityChip::U32FromLeBytes.lookups(),
        );
        let u64_shift_right_32_circuit = LookupAir::new(
            UtilityChip::U64ShiftRight32AsU32,
            UtilityChip::U64ShiftRight32AsU32.lookups(),
        );
        let u64_as_u32_circuit =
            LookupAir::new(UtilityChip::U64AsU32, UtilityChip::U64AsU32.lookups());

        let (system, prover_key) = System::new(
            commitment_parameters,
            vec![
                u8_circuit,
                u32_from_le_bytes_circuit,
                u32_or_circuit,
                u64_shift_right_32_circuit,
                u64_as_u32_circuit,
            ], // order matters
        );

        let claims = UtilityChipClaims {
            claims: vec![
                [
                    vec![Val::from_usize(UtilityChip::U32FromLeBytes.position())],
                    vec![
                        Val::from_u8(byte0),
                        Val::from_u8(byte1),
                        Val::from_u8(byte2),
                        Val::from_u8(byte3),
                    ],
                    vec![Val::from_u32(u32_val)],
                ]
                .concat(),
            ],
        };

        let (_traces, witness) = claims.witness(&system);

        let claims_slice: Vec<&[Val]> = claims.claims.iter().map(|v| v.as_slice()).collect();
        let claims_slice: &[&[Val]] = &claims_slice;

        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };

        let proof =
            system.prove_multiple_claims(fri_parameters, &prover_key, claims_slice, witness);
        system
            .verify_multiple_claims(fri_parameters, claims_slice, &proof)
            .expect("verification issue");
    }

    #[test]
    fn test_u8_or() {
        let byte0 = 0xfe;
        let byte1 = 0xac;
        let or = byte0 | byte1;

        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let u8_circuit = LookupAir::new(UtilityChip::U8Or, UtilityChip::U8Or.lookups());
        let u32_or_circuit = LookupAir::new(UtilityChip::U32Or, UtilityChip::U32Or.lookups());
        let u32_from_le_bytes_circuit = LookupAir::new(
            UtilityChip::U32FromLeBytes,
            UtilityChip::U32FromLeBytes.lookups(),
        );
        let u64_shift_right_32_circuit = LookupAir::new(
            UtilityChip::U64ShiftRight32AsU32,
            UtilityChip::U64ShiftRight32AsU32.lookups(),
        );
        let u64_as_u32_circuit =
            LookupAir::new(UtilityChip::U64AsU32, UtilityChip::U64AsU32.lookups());

        let (system, prover_key) = System::new(
            commitment_parameters,
            vec![
                u8_circuit,
                u32_from_le_bytes_circuit,
                u32_or_circuit,
                u64_shift_right_32_circuit,
                u64_as_u32_circuit,
            ], // order matters
        );

        let claims = UtilityChipClaims {
            claims: vec![
                [
                    vec![Val::from_usize(UtilityChip::U8Or.position())],
                    vec![Val::from_u8(byte0), Val::from_u8(byte1), Val::from_u8(or)],
                ]
                .concat(),
            ],
        };

        let (_traces, witness) = claims.witness(&system);

        let claims_slice: Vec<&[Val]> = claims.claims.iter().map(|v| v.as_slice()).collect();
        let claims_slice: &[&[Val]] = &claims_slice;

        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };

        let proof =
            system.prove_multiple_claims(fri_parameters, &prover_key, claims_slice, witness);
        system
            .verify_multiple_claims(fri_parameters, claims_slice, &proof)
            .expect("verification issue");
    }

    #[test]
    fn test_u32_or() {
        let left = 0xabcd1234u32;
        let right = 0x998123ffu32;
        let or = left | right;

        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let u8_circuit = LookupAir::new(UtilityChip::U8Or, UtilityChip::U8Or.lookups());
        let u32_or_circuit = LookupAir::new(UtilityChip::U32Or, UtilityChip::U32Or.lookups());
        let u32_from_le_bytes_circuit = LookupAir::new(
            UtilityChip::U32FromLeBytes,
            UtilityChip::U32FromLeBytes.lookups(),
        );
        let u64_shift_right_32_circuit = LookupAir::new(
            UtilityChip::U64ShiftRight32AsU32,
            UtilityChip::U64ShiftRight32AsU32.lookups(),
        );
        let u64_as_u32_circuit =
            LookupAir::new(UtilityChip::U64AsU32, UtilityChip::U64AsU32.lookups());

        let (system, prover_key) = System::new(
            commitment_parameters,
            vec![
                u8_circuit,
                u32_from_le_bytes_circuit,
                u32_or_circuit,
                u64_shift_right_32_circuit,
                u64_as_u32_circuit,
            ], // order matters
        );

        let claims = UtilityChipClaims {
            claims: vec![
                [
                    vec![Val::from_usize(UtilityChip::U32Or.position())],
                    vec![Val::from_u32(left), Val::from_u32(right), Val::from_u32(or)],
                ]
                .concat(),
            ],
        };

        let (_traces, witness) = claims.witness(&system);

        let claims_slice: Vec<&[Val]> = claims.claims.iter().map(|v| v.as_slice()).collect();
        let claims_slice: &[&[Val]] = &claims_slice;

        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };

        let proof =
            system.prove_multiple_claims(fri_parameters, &prover_key, claims_slice, witness);
        system
            .verify_multiple_claims(fri_parameters, claims_slice, &proof)
            .expect("verification issue");
    }

    #[test]
    fn test_u64_shift_right_32_as_u32() {
        let u64_val = 0xabcd12341f1f1f1fu64;
        let u64_val_right_shifted = u64_val >> 32;

        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let u8_circuit = LookupAir::new(UtilityChip::U8Or, UtilityChip::U8Or.lookups());
        let u32_or_circuit = LookupAir::new(UtilityChip::U32Or, UtilityChip::U32Or.lookups());
        let u32_from_le_bytes_circuit = LookupAir::new(
            UtilityChip::U32FromLeBytes,
            UtilityChip::U32FromLeBytes.lookups(),
        );
        let u64_shift_right_32_circuit = LookupAir::new(
            UtilityChip::U64ShiftRight32AsU32,
            UtilityChip::U64ShiftRight32AsU32.lookups(),
        );
        let u64_as_u32_circuit =
            LookupAir::new(UtilityChip::U64AsU32, UtilityChip::U64AsU32.lookups());

        let (system, prover_key) = System::new(
            commitment_parameters,
            vec![
                u8_circuit,
                u32_from_le_bytes_circuit,
                u32_or_circuit,
                u64_shift_right_32_circuit,
                u64_as_u32_circuit,
            ], // order matters
        );

        let claims = UtilityChipClaims {
            claims: vec![
                [
                    vec![Val::from_usize(
                        UtilityChip::U64ShiftRight32AsU32.position(),
                    )],
                    vec![
                        Val::from_u64(u64_val),
                        Val::from_u32(u64_val_right_shifted as u32),
                    ],
                ]
                .concat(),
            ],
        };

        let (_traces, witness) = claims.witness(&system);

        let claims_slice: Vec<&[Val]> = claims.claims.iter().map(|v| v.as_slice()).collect();
        let claims_slice: &[&[Val]] = &claims_slice;

        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };

        let proof =
            system.prove_multiple_claims(fri_parameters, &prover_key, claims_slice, witness);
        system
            .verify_multiple_claims(fri_parameters, claims_slice, &proof)
            .expect("verification issue");
    }

    #[test]
    fn test_u64_as_u32() {
        let u64_val = 0xabcd12341f1f1f1fu64;
        let u32_val = u64_val as u32;
        assert_eq!(u32_val, 0x1f1f1f1fu32);

        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let u8_circuit = LookupAir::new(UtilityChip::U8Or, UtilityChip::U8Or.lookups());
        let u32_or_circuit = LookupAir::new(UtilityChip::U32Or, UtilityChip::U32Or.lookups());
        let u32_from_le_bytes_circuit = LookupAir::new(
            UtilityChip::U32FromLeBytes,
            UtilityChip::U32FromLeBytes.lookups(),
        );
        let u64_shift_right_32_circuit = LookupAir::new(
            UtilityChip::U64ShiftRight32AsU32,
            UtilityChip::U64ShiftRight32AsU32.lookups(),
        );
        let u64_as_u32_circuit =
            LookupAir::new(UtilityChip::U64AsU32, UtilityChip::U64AsU32.lookups());

        let (system, prover_key) = System::new(
            commitment_parameters,
            vec![
                u8_circuit,
                u32_from_le_bytes_circuit,
                u32_or_circuit,
                u64_shift_right_32_circuit,
                u64_as_u32_circuit,
            ], // order matters
        );

        let claims = UtilityChipClaims {
            claims: vec![
                [
                    vec![Val::from_usize(UtilityChip::U64AsU32.position())],
                    vec![Val::from_u64(u64_val), Val::from_u32(u32_val)],
                ]
                .concat(),
            ],
        };

        let (_traces, witness) = claims.witness(&system);

        let claims_slice: Vec<&[Val]> = claims.claims.iter().map(|v| v.as_slice()).collect();
        let claims_slice: &[&[Val]] = &claims_slice;

        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };

        let proof =
            system.prove_multiple_claims(fri_parameters, &prover_key, claims_slice, witness);
        system
            .verify_multiple_claims(fri_parameters, claims_slice, &proof)
            .expect("verification issue");
    }
}
