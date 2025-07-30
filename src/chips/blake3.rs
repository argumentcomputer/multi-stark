#[cfg(test)]
mod tests {
    use crate::builder::symbolic::{Entry, SymbolicExpression, SymbolicVariable};
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
    const U32_RIGHT_ROTATE_10_TRACE_WIDTH: usize = 25;
    const U32_RIGHT_ROTATE_12_TRACE_WIDTH: usize = U32_RIGHT_ROTATE_10_TRACE_WIDTH;

    enum Blake3CompressionChips {
        U8Xor,
        U32Xor,
        U32Add,
        U32RightRotate8,
        U32RightRotate16,
        U32RightRotate12, // FIXME: currently underconstrained. Needs rewriting using Gabriel's advice
        U32RightRotate10, // FIXME: currently underconstrained. Needs rewriting using Gabriel's advice
        U8PairRangeCheck,
    }

    impl Blake3CompressionChips {
        fn position(&self) -> usize {
            match self {
                Blake3CompressionChips::U8Xor => 0,
                Blake3CompressionChips::U32Xor => 1,
                Blake3CompressionChips::U32Add => 2,
                Blake3CompressionChips::U32RightRotate8 => 3,
                Blake3CompressionChips::U32RightRotate16 => 4,
                Blake3CompressionChips::U32RightRotate12 => 5,
                Blake3CompressionChips::U32RightRotate10 => 6,
                Blake3CompressionChips::U8PairRangeCheck => 7,
            }
        }
    }

    impl<F: Field> BaseAir<F> for Blake3CompressionChips {
        fn width(&self) -> usize {
            match self {
                Self::U8Xor | Self::U8PairRangeCheck => U8_XOR_PAIR_RANGE_CHECK_TRACE_WIDTH,
                Self::U32Xor => U32_XOR_TRACE_WIDTH,
                Self::U32Add => U32_ADD_TRACE_WIDTH,
                Self::U32RightRotate8 => U32_RIGHT_ROTATE_8_TRACE_WIDTH,
                Self::U32RightRotate16 => U32_RIGHT_ROTATE_16_TRACE_WIDTH,
                Self::U32RightRotate12 => U32_RIGHT_ROTATE_12_TRACE_WIDTH,
                Self::U32RightRotate10 => U32_RIGHT_ROTATE_10_TRACE_WIDTH,
            }
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            match self {
                Self::U8Xor | Self::U8PairRangeCheck => {
                    let bytes: [u8; BYTE_VALUES_NUM] =
                        array::from_fn(|idx| u8::try_from(idx).unwrap());
                    let mut trace_values = Vec::with_capacity(
                        BYTE_VALUES_NUM * BYTE_VALUES_NUM * PREPROCESSED_TRACE_WIDTH,
                    );
                    for i in 0..256 {
                        for j in 0..256 {
                            trace_values.push(F::from_u8(bytes[i]));
                            trace_values.push(F::from_u8(bytes[j]));
                            trace_values.push(F::from_u8(bytes[i] ^ bytes[j]));
                        }
                    }
                    Some(RowMajorMatrix::new(trace_values, PREPROCESSED_TRACE_WIDTH))
                }
                Self::U32Xor => None,
                Self::U32Add => None,
                Self::U32RightRotate8 => None,
                Self::U32RightRotate16 => None,
                Self::U32RightRotate12 => None,
                Self::U32RightRotate10 => None,
            }
        }
    }

    impl<AB> Air<AB> for Blake3CompressionChips
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                Self::U8Xor | Self::U8PairRangeCheck => {}
                Self::U32Xor => {}
                Self::U32Add => {
                    let main = builder.main();
                    let local = main.row_slice(0).unwrap();
                    let x = &local[0..4];
                    let y = &local[4..8];
                    let z = &local[8..12];
                    let carry = local[12];
                    // the carry must be a boolean
                    builder.assert_bool(carry);

                    let expr1 = x[0]
                        + x[1] * AB::Expr::from_u32(256)
                        + x[2] * AB::Expr::from_u32(256 * 256)
                        + x[3] * AB::Expr::from_u32(256 * 256 * 256)
                        + y[0]
                        + y[1] * AB::Expr::from_u32(256)
                        + y[2] * AB::Expr::from_u32(256 * 256)
                        + y[3] * AB::Expr::from_u32(256 * 256 * 256);
                    let expr2 = z[0]
                        + z[1] * AB::Expr::from_u32(256)
                        + z[2] * AB::Expr::from_u32(256 * 256)
                        + z[3] * AB::Expr::from_u32(256 * 256 * 256)
                        + carry * AB::Expr::from_u64(256 * 256 * 256 * 256);
                    builder.assert_eq(expr1, expr2);
                }
                Self::U32RightRotate8 => {}
                Self::U32RightRotate16 => {}
                Self::U32RightRotate12 | Self::U32RightRotate10 => {
                    let main = builder.main();
                    let local = main.row_slice(0).unwrap();

                    let input = local[1]
                        + local[2] * AB::Expr::from_u32(256)
                        + local[3] * AB::Expr::from_u32(256 * 256)
                        + local[4] * AB::Expr::from_u32(256 * 256 * 256);

                    let output = local[5]
                        + local[6] * AB::Expr::from_u32(256)
                        + local[7] * AB::Expr::from_u32(256 * 256)
                        + local[8] * AB::Expr::from_u32(256 * 256 * 256);

                    let two_pow_k = local[9]
                        + local[10] * AB::Expr::from_u32(256)
                        + local[11] * AB::Expr::from_u32(256 * 256)
                        + local[12] * AB::Expr::from_u32(256 * 256 * 256);

                    let two_pow_32_minus_k = local[13]
                        + local[14] * AB::Expr::from_u32(256)
                        + local[15] * AB::Expr::from_u32(256 * 256)
                        + local[16] * AB::Expr::from_u32(256 * 256 * 256);

                    let input_div = local[17]
                        + local[18] * AB::Expr::from_u32(256)
                        + local[19] * AB::Expr::from_u32(256 * 256)
                        + local[20] * AB::Expr::from_u32(256 * 256 * 256);

                    let input_rem = local[21]
                        + local[22] * AB::Expr::from_u32(256)
                        + local[23] * AB::Expr::from_u32(256 * 256)
                        + local[24] * AB::Expr::from_u32(256 * 256 * 256);

                    builder.assert_eq(
                        input,
                        input_div.clone() * two_pow_k.clone() + input_rem.clone(),
                    );
                    builder.assert_eq(output, input_div + input_rem * two_pow_32_minus_k);
                }
            }
        }
    }

    impl Blake3CompressionChips {
        fn lookups(&self) -> Vec<Lookup<SymbExpr>> {
            let var =
                |i| SymbolicExpression::from(SymbolicVariable::new(Entry::Main { offset: 0 }, i));

            let preprocessed_var = |i| {
                SymbolicExpression::from(SymbolicVariable::new(
                    Entry::Preprocessed { offset: 0 },
                    i,
                ))
            };

            let u8_xor_idx = Blake3CompressionChips::U8Xor.position();
            let u32_xor_idx = Blake3CompressionChips::U32Xor.position();
            let u32_add_idx = Blake3CompressionChips::U32Add.position();
            let u32_right_rotate_8_idx = Blake3CompressionChips::U32RightRotate8.position();
            let u32_right_rotate_16_idx = Blake3CompressionChips::U32RightRotate16.position();
            let u32_right_rotate_12_idx = Blake3CompressionChips::U32RightRotate12.position();
            let u32_right_rotate_10_idx = Blake3CompressionChips::U32RightRotate10.position();
            let u8_pair_range_check_idx = Blake3CompressionChips::U8PairRangeCheck.position();
            match self {
                Self::U8Xor | Self::U8PairRangeCheck => {
                    vec![
                        Lookup::pull(
                            var(0),
                            vec![
                                SymbExpr::from_usize(u8_xor_idx),
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

                Self::U32Xor => {
                    let mut lookups = vec![Lookup::pull(
                        var(0),
                        vec![
                            SymbExpr::from_usize(u32_xor_idx),
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

                    // push (A, B, A^B) tuples to U8Xor chip for verification
                    lookups.extend((0..4).map(|i| {
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u8_xor_idx).clone(),
                                var(i + 1),
                                var(i + 5),
                                var(i + 9),
                            ],
                        )
                    }));
                    lookups
                }

                Self::U32Add => {
                    // Pull
                    let mut lookups = vec![Lookup::pull(
                        var(13),
                        vec![
                            SymbExpr::from_usize(u32_add_idx),
                            var(0)
                                + var(1) * SymbExpr::from_u32(256)
                                + var(2) * SymbExpr::from_u32(256 * 256)
                                + var(3) * SymbExpr::from_u32(256 * 256 * 256),
                            var(4)
                                + var(5) * SymbExpr::from_u32(256)
                                + var(6) * SymbExpr::from_u32(256 * 256)
                                + var(7) * SymbExpr::from_u32(256 * 256 * 256),
                            var(8)
                                + var(9) * SymbExpr::from_u32(256)
                                + var(10) * SymbExpr::from_u32(256 * 256)
                                + var(11) * SymbExpr::from_u32(256 * 256 * 256),
                        ],
                    )];
                    // push (A, B) tuples to U8PairRangeCheck chip for verification
                    lookups.extend((0..4).map(|i| {
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u8_pair_range_check_idx).clone(),
                                var(i),
                                var(i + 4),
                            ],
                        )
                    }));

                    // push (A + B, 0) tuples to U8PairRangeCheck chip for verification. 0 is used just as a stub
                    lookups.extend((0..4).map(|i| {
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u8_pair_range_check_idx).clone(),
                                var(i + 8),
                                SymbExpr::ZERO,
                            ],
                        )
                    }));
                    lookups
                }

                Self::U32RightRotate8 => {
                    let mut lookups = vec![Lookup::pull(
                        var(0),
                        vec![
                            SymbExpr::from_usize(u32_right_rotate_8_idx),
                            var(1)
                                + var(2) * SymbExpr::from_u32(256)
                                + var(3) * SymbExpr::from_u32(256 * 256)
                                + var(4) * SymbExpr::from_u32(256 * 256 * 256),
                            // note var indices
                            var(2)
                                + var(3) * SymbExpr::from_u32(256)
                                + var(4) * SymbExpr::from_u32(256 * 256)
                                + var(1) * SymbExpr::from_u32(256 * 256 * 256),
                        ],
                    )];

                    // range check only input u32 word (since output is constructed exactly from the same bytes)
                    lookups.extend((0..2).map(|i| {
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u8_pair_range_check_idx).clone(),
                                var(i + 1),
                                var(i + 3),
                            ],
                        )
                    }));

                    lookups
                }

                Self::U32RightRotate16 => {
                    let mut lookups = vec![Lookup::pull(
                        var(0),
                        vec![
                            SymbExpr::from_usize(u32_right_rotate_16_idx),
                            var(1)
                                + var(2) * SymbExpr::from_u32(256)
                                + var(3) * SymbExpr::from_u32(256 * 256)
                                + var(4) * SymbExpr::from_u32(256 * 256 * 256),
                            // note var indices
                            var(3)
                                + var(4) * SymbExpr::from_u32(256)
                                + var(1) * SymbExpr::from_u32(256 * 256)
                                + var(2) * SymbExpr::from_u32(256 * 256 * 256),
                        ],
                    )];

                    // range check only input u32 word (since output is constructed exactly from the same 4 bytes)
                    lookups.extend((0..2).map(|i| {
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u8_pair_range_check_idx).clone(),
                                var(i + 1),
                                var(i + 3),
                            ],
                        )
                    }));

                    lookups
                }

                Self::U32RightRotate12 => {
                    vec![Lookup::pull(
                        var(0),
                        vec![
                            SymbExpr::from_usize(u32_right_rotate_12_idx),
                            var(1)
                                + var(2) * SymbExpr::from_u32(256)
                                + var(3) * SymbExpr::from_u32(256 * 256)
                                + var(4) * SymbExpr::from_u32(256 * 256 * 256),
                            var(5)
                                + var(6) * SymbExpr::from_u32(256)
                                + var(7) * SymbExpr::from_u32(256 * 256)
                                + var(8) * SymbExpr::from_u32(256 * 256 * 256),
                        ],
                    )]
                }

                Self::U32RightRotate10 => {
                    vec![Lookup::pull(
                        var(0),
                        vec![
                            SymbExpr::from_usize(u32_right_rotate_10_idx),
                            var(1)
                                + var(2) * SymbExpr::from_u32(256)
                                + var(3) * SymbExpr::from_u32(256 * 256)
                                + var(4) * SymbExpr::from_u32(256 * 256 * 256),
                            var(5)
                                + var(6) * SymbExpr::from_u32(256)
                                + var(7) * SymbExpr::from_u32(256 * 256)
                                + var(8) * SymbExpr::from_u32(256 * 256 * 256),
                        ],
                    )]
                }
            }
        }
    }

    struct Blake3CompressionClaims {
        claims: Vec<Vec<Val>>,
    }

    impl Blake3CompressionClaims {
        fn witness(
            &self,
            system: &System<Blake3CompressionChips>,
        ) -> (Vec<RowMajorMatrix<Val>>, SystemWitness) {
            // extract values from a claims

            let mut u32_xor_values_from_claims = vec![];

            let mut u32_add_values_from_claims = vec![];

            let mut byte_xor_values_from_claims = vec![];

            let mut byte_range_check_values_from_claims = vec![];

            let mut u32_rotate_right_8_values_from_claims = vec![];
            let mut u32_rotate_right_16_values_from_claims = vec![];
            let mut u32_rotate_right_12_values_from_claims = vec![];
            let mut u32_rotate_right_10_values_from_claims = vec![];

            for claim in self.claims.clone() {
                // we should have at least chip index
                assert!(claim.len() > 0, "wrong claim format");
                match claim[0].as_canonical_u64() {
                    0u64 => {
                        // This is our U8Xor claim. We should have chip_idx, A, B, A xor B (where A, B are bytes)
                        assert!(claim.len() == 4, "[U8Xor] wrong claim format");
                        byte_xor_values_from_claims.push((claim[1], claim[2], claim[3]));
                    }
                    1u64 => {
                        /* This is our U32Xor claim. We should have chip_idx, A, B, A xor B (where A, B are u32) */

                        assert!(claim.len() == 4, "[U32Xor] wrong claim format");
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let b_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                        let xor_u32 = u32::try_from(claim[3].as_canonical_u64()).unwrap();

                        u32_xor_values_from_claims.push((a_u32, b_u32, xor_u32));

                        /* we decompose our input u32 words (A, B) and send their bytes to U8Xor chip, relying on lookup constraining */

                        let a_bytes: [u8; 4] = a_u32.to_le_bytes();
                        let b_bytes: [u8; 4] = b_u32.to_le_bytes();
                        let xor_bytes: [u8; 4] = xor_u32.to_le_bytes();

                        byte_xor_values_from_claims.push((
                            Val::from_u8(a_bytes[0]),
                            Val::from_u8(b_bytes[0]),
                            Val::from_u8(xor_bytes[0]),
                        ));
                        byte_xor_values_from_claims.push((
                            Val::from_u8(a_bytes[1]),
                            Val::from_u8(b_bytes[1]),
                            Val::from_u8(xor_bytes[1]),
                        ));
                        byte_xor_values_from_claims.push((
                            Val::from_u8(a_bytes[2]),
                            Val::from_u8(b_bytes[2]),
                            Val::from_u8(xor_bytes[2]),
                        ));
                        byte_xor_values_from_claims.push((
                            Val::from_u8(a_bytes[3]),
                            Val::from_u8(b_bytes[3]),
                            Val::from_u8(xor_bytes[3]),
                        ));
                    }

                    2u64 => {
                        /* This is our U32Add claim. We should have chip_idx, A, B, A + B (where A, B are u32) */

                        assert!(claim.len() == 4, "[U32Add] wrong claim format");
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let b_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                        let add_u32 = u32::try_from(claim[3].as_canonical_u64()).unwrap();

                        u32_add_values_from_claims.push((a_u32, b_u32, add_u32));

                        /* we decompose our input u32 words (A, B) and send their bytes to U8Xor chip, relying on lookup constraining */

                        let a_bytes: [u8; 4] = a_u32.to_le_bytes();
                        let b_bytes: [u8; 4] = b_u32.to_le_bytes();
                        let add_bytes: [u8; 4] = add_u32.to_le_bytes();

                        byte_range_check_values_from_claims
                            .push((Val::from_u8(a_bytes[0]), Val::from_u8(b_bytes[0])));
                        byte_range_check_values_from_claims
                            .push((Val::from_u8(a_bytes[1]), Val::from_u8(b_bytes[1])));
                        byte_range_check_values_from_claims
                            .push((Val::from_u8(a_bytes[2]), Val::from_u8(b_bytes[2])));
                        byte_range_check_values_from_claims
                            .push((Val::from_u8(a_bytes[3]), Val::from_u8(b_bytes[3])));

                        byte_range_check_values_from_claims
                            .push((Val::from_u8(add_bytes[0]), Val::ZERO));
                        byte_range_check_values_from_claims
                            .push((Val::from_u8(add_bytes[1]), Val::ZERO));
                        byte_range_check_values_from_claims
                            .push((Val::from_u8(add_bytes[2]), Val::ZERO));
                        byte_range_check_values_from_claims
                            .push((Val::from_u8(add_bytes[3]), Val::ZERO));
                    }
                    3u64 => {
                        /* This is our U32RotateRight8 claim. We should have chip_idx, A, A_rot */

                        assert!(claim.len() == 3, "[U32RightRotate8] wrong claim format");
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();

                        u32_rotate_right_8_values_from_claims.push((a_u32, rot_u32));

                        let a_bytes: [u8; 4] = a_u32.to_le_bytes();

                        byte_range_check_values_from_claims
                            .push((Val::from_u8(a_bytes[0]), Val::from_u8(a_bytes[2])));
                        byte_range_check_values_from_claims
                            .push((Val::from_u8(a_bytes[1]), Val::from_u8(a_bytes[3])));
                    }
                    4u64 => {
                        /* This is our U32RotateRight16 claim. We should have chip_idx, A, A_rot */

                        assert!(claim.len() == 3, "[U32RightRotate16] wrong claim format");
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();

                        u32_rotate_right_16_values_from_claims.push((a_u32, rot_u32));

                        let a_bytes: [u8; 4] = a_u32.to_le_bytes();

                        byte_range_check_values_from_claims
                            .push((Val::from_u8(a_bytes[0]), Val::from_u8(a_bytes[2])));
                        byte_range_check_values_from_claims
                            .push((Val::from_u8(a_bytes[1]), Val::from_u8(a_bytes[3])));
                    }
                    5u64 => {
                        /* This is our U32RotateRight12 claim. We should have chip_idx, A, A_rot */

                        assert!(claim.len() == 3, "[U32RightRotate12] wrong claim format");
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();

                        u32_rotate_right_12_values_from_claims.push((a_u32, rot_u32));
                    }
                    6u64 => {
                        /* This is our U32RotateRight10 claim. We should have chip_idx, A, A_rot */

                        assert!(claim.len() == 3, "[U32RightRotate10] wrong claim format");
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();

                        u32_rotate_right_10_values_from_claims.push((a_u32, rot_u32));
                    }

                    7u64 => {
                        /* This is our U8PairRangeCheck claim. We should have chip_idx, A, B */

                        assert!(claim.len() == 3, "[U8Xor] wrong claim format");
                        byte_range_check_values_from_claims.push((claim[1], claim[2]));
                    }
                    _ => panic!("unsupported chip"),
                }
            }

            // build U8Xor / U8PairRangeCheck trace (columns: multiplicity_u8_xor, multiplicity_pair_range_check)
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

            // build U32Xor trace (columns: multiplicity, A0, A1, A2, A3, B0, B1, B2, B3, A0^B0, A1^B1, A2^B2, A3^B3)
            let mut u32xor_trace_values =
                Vec::<Val>::with_capacity(u32_xor_values_from_claims.len());
            for (a, b, a_xor_b) in u32_xor_values_from_claims.into_iter() {
                debug_assert_eq!(a ^ b, a_xor_b);

                let a_bytes: [u8; 4] = a.to_le_bytes();
                let b_bytes: [u8; 4] = b.to_le_bytes();
                let a_xor_b_bytes: [u8; 4] = a_xor_b.to_le_bytes();

                u32xor_trace_values.push(Val::ONE); // multiplicity

                u32xor_trace_values.extend_from_slice(a_bytes.map(Val::from_u8).as_slice());
                u32xor_trace_values.extend_from_slice(b_bytes.map(Val::from_u8).as_slice());
                u32xor_trace_values.extend_from_slice(a_xor_b_bytes.map(Val::from_u8).as_slice());
            }

            // build U32Add trace (columns: A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3, carry, multiplicity)
            let mut u32_add_trace_values = vec![];
            for (a, b, c) in u32_add_values_from_claims {
                let (z, carry) = a.overflowing_add(b);
                // actual addition result should match to the value from claim
                debug_assert_eq!(z, c);

                let a_bytes: [u8; 4] = a.to_le_bytes();
                let b_bytes: [u8; 4] = b.to_le_bytes();
                let c_bytes: [u8; 4] = c.to_le_bytes();

                u32_add_trace_values.extend_from_slice(a_bytes.map(Val::from_u8).as_slice());
                u32_add_trace_values.extend_from_slice(b_bytes.map(Val::from_u8).as_slice());
                u32_add_trace_values.extend_from_slice(c_bytes.map(Val::from_u8).as_slice());

                u32_add_trace_values.push(Val::from_bool(carry));
                u32_add_trace_values.push(Val::ONE); // multiplicity
            }

            // build U32RotateRight8 trace (columns: multiplicity, a0, a1, a2, a3, rot0, rot1, rot2, rot3)
            let mut u32_rotate_right_8_trace_values = vec![];
            for (a, rot) in u32_rotate_right_8_values_from_claims {
                u32_rotate_right_8_trace_values.push(Val::ONE); // multiplicity

                // actual rotate 8 result should match to the value from claim
                debug_assert_eq!(a.rotate_right(8), rot);

                let a_bytes: [u8; 4] = a.to_le_bytes();
                let rot_bytes: [u8; 4] = rot.to_le_bytes();

                u32_rotate_right_8_trace_values
                    .extend_from_slice(a_bytes.map(Val::from_u8).as_slice());
                u32_rotate_right_8_trace_values
                    .extend_from_slice(rot_bytes.map(Val::from_u8).as_slice());
            }

            // build U32RotateRight16 trace (columns: multiplicity, a0, a1, a2, a3, rot0, rot1, rot2, rot3)
            let mut u32_rotate_right_16_trace_values = vec![];
            for (a, rot) in u32_rotate_right_16_values_from_claims {
                u32_rotate_right_16_trace_values.push(Val::ONE); // multiplicity

                // actual rotate 16 result should match to the value from claim
                debug_assert_eq!(a.rotate_right(16), rot);

                let a_bytes: [u8; 4] = a.to_le_bytes();
                let rot_bytes: [u8; 4] = rot.to_le_bytes();

                u32_rotate_right_16_trace_values
                    .extend_from_slice(a_bytes.map(Val::from_u8).as_slice());
                u32_rotate_right_16_trace_values
                    .extend_from_slice(rot_bytes.map(Val::from_u8).as_slice());
            }

            fn rot_10_12_trace_values(k: u32, vals_from_claim: Vec<(u32, u32)>) -> Vec<Val> {
                let mut values = vec![];
                for (a, rot) in vals_from_claim {
                    values.push(Val::ONE); // multiplicity

                    // actual rotate result should match to the value from claim
                    debug_assert_eq!(a.rotate_right(k), rot);

                    let two_pow_k = u32::try_from(2usize.pow(k)).unwrap();
                    let two_pow_32_minus_k = u32::try_from(2usize.pow(32 - k)).unwrap();

                    let input_div = a / two_pow_k;
                    let input_rem = a % two_pow_k;

                    let two_pow_k_bytes: [u8; 4] = two_pow_k.to_le_bytes();
                    let two_pow_32_minus_k_bytes: [u8; 4] = two_pow_32_minus_k.to_le_bytes();
                    let input_div_bytes: [u8; 4] = input_div.to_le_bytes();
                    let input_rem_bytes: [u8; 4] = input_rem.to_le_bytes();

                    let a_bytes: [u8; 4] = a.to_le_bytes();
                    let rot_bytes: [u8; 4] = rot.to_le_bytes();

                    values.extend_from_slice(a_bytes.map(Val::from_u8).as_slice());
                    values.extend_from_slice(rot_bytes.map(Val::from_u8).as_slice());
                    values.extend_from_slice(two_pow_k_bytes.map(Val::from_u8).as_slice());
                    values.extend_from_slice(two_pow_32_minus_k_bytes.map(Val::from_u8).as_slice());
                    values.extend_from_slice(input_div_bytes.map(Val::from_u8).as_slice());
                    values.extend_from_slice(input_rem_bytes.map(Val::from_u8).as_slice());
                }
                values
            }

            // build U32RotateRight12 trace (columns:
            //      multiplicity,
            //      a0, a1, a2, a3,
            //      rot0, rot1, rot2, rot3,
            //      two_pow_k_0, two_pow_k_1, two_pow_k_2, two_pow_k_3,
            //      two_pow_32_minus_k_0, two_pow_32_minus_k_1, two_pow_32_minus_k_2, two_pow_32_minus_k_3,
            //      value_div_0, value_div_1, value_div_2, value_div_3,
            //      value_rem_0, value_rem_1, value_rem_2, value_rem_3
            // )
            let u32_rotate_right_12_trace_values =
                rot_10_12_trace_values(12, u32_rotate_right_12_values_from_claims);

            // build U32RotateRight10 trace (columns same as for U32RotateRight12)
            let u32_rotate_right_10_trace_values =
                rot_10_12_trace_values(10, u32_rotate_right_10_values_from_claims);

            let traces = vec![
                RowMajorMatrix::new(
                    u8_xor_range_check_trace_values,
                    U8_XOR_PAIR_RANGE_CHECK_TRACE_WIDTH,
                ),
                RowMajorMatrix::new(u32xor_trace_values, U32_XOR_TRACE_WIDTH),
                RowMajorMatrix::new(u32_add_trace_values, U32_ADD_TRACE_WIDTH),
                RowMajorMatrix::new(
                    u32_rotate_right_8_trace_values,
                    U32_RIGHT_ROTATE_8_TRACE_WIDTH,
                ),
                RowMajorMatrix::new(
                    u32_rotate_right_16_trace_values,
                    U32_RIGHT_ROTATE_16_TRACE_WIDTH,
                ),
                RowMajorMatrix::new(
                    u32_rotate_right_12_trace_values,
                    U32_RIGHT_ROTATE_12_TRACE_WIDTH,
                ),
                RowMajorMatrix::new(
                    u32_rotate_right_10_trace_values,
                    U32_RIGHT_ROTATE_10_TRACE_WIDTH,
                ),
            ];

            (traces.clone(), SystemWitness::from_stage_1(traces, system))
        }
    }

    #[test]
    fn blake3_test_debug() {
        // computation

        let a_u8 = 0xa1u8;
        let b_u8 = 0xa8u8;
        let xor_u8 = a_u8 ^ b_u8;

        let a1_u8 = 0x01u8;
        let b1_u8 = 0x02u8;
        let xor1_u8 = a1_u8 ^ b1_u8;

        let a_u32 = 0x000000ffu32;
        let b_u32 = 0x0000ff01u32;
        let xor_u32 = a_u32 ^ b_u32;
        let add_u32 = a_u32.wrapping_add(b_u32);
        let a_rot_8 = a_u32.rotate_right(8);
        let a_rot_16 = a_u32.rotate_right(16);
        let a_rot_12 = a_u32.rotate_right(12);
        let a_rot_10 = a_u32.rotate_right(10);

        // circuit testing
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let u8_circuit = LookupAir::new(
            Blake3CompressionChips::U8Xor,
            Blake3CompressionChips::U8Xor.lookups(),
        );
        let u32_circuit = LookupAir::new(
            Blake3CompressionChips::U32Xor,
            Blake3CompressionChips::U32Xor.lookups(),
        );
        let u32_add_circuit = LookupAir::new(
            Blake3CompressionChips::U32Add,
            Blake3CompressionChips::U32Add.lookups(),
        );
        let u32_rotate_right_8_circuit = LookupAir::new(
            Blake3CompressionChips::U32RightRotate8,
            Blake3CompressionChips::U32RightRotate8.lookups(),
        );
        let u32_rotate_right_16_circuit = LookupAir::new(
            Blake3CompressionChips::U32RightRotate16,
            Blake3CompressionChips::U32RightRotate16.lookups(),
        );
        let u32_rotate_right_12_circuit = LookupAir::new(
            Blake3CompressionChips::U32RightRotate12,
            Blake3CompressionChips::U32RightRotate12.lookups(),
        );
        let u32_rotate_right_10_circuit = LookupAir::new(
            Blake3CompressionChips::U32RightRotate10,
            Blake3CompressionChips::U32RightRotate10.lookups(),
        );

        let (system, prover_key) = System::new(
            commitment_parameters,
            vec![
                u8_circuit,
                u32_circuit,
                u32_add_circuit,
                u32_rotate_right_8_circuit,
                u32_rotate_right_16_circuit,
                u32_rotate_right_12_circuit,
                u32_rotate_right_10_circuit,
            ],
        );

        let f = Val::from_u8;
        let f32 = Val::from_u32;
        let claims = Blake3CompressionClaims {
            claims: vec![
                vec![
                    Val::from_usize(Blake3CompressionChips::U8Xor.position()),
                    f(a_u8),
                    f(b_u8),
                    f(xor_u8),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U8Xor.position()),
                    f(a1_u8),
                    f(b1_u8),
                    f(xor1_u8),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Xor.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(xor_u32),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Add.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(add_u32),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate8.position()),
                    f32(a_u32),
                    f32(a_rot_8),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate16.position()),
                    f32(a_u32),
                    f32(a_rot_16),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate12.position()),
                    f32(a_u32),
                    f32(a_rot_12),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate10.position()),
                    f32(a_u32),
                    f32(a_rot_10),
                ],
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
