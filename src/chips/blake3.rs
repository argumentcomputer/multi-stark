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
    const U32_RIGHT_ROTATE_7_TRACE_WIDTH: usize = 25;
    const U32_RIGHT_ROTATE_12_TRACE_WIDTH: usize = U32_RIGHT_ROTATE_7_TRACE_WIDTH;

    // Totally 81 byte columns:
    // multiplicity, a_in(4), b_in(4), c_in(4), d_in(4), mx_in(4), my_in(4),
    // a_0_tmp(4), a_0(4), d_0_tmp(4), d_0(4), c_0(4), b_0_tmp(4), b_0(4),
    // a_1_tmp(4), a_1(4), d_1_tmp(4), d_1(4), c_1(4), b_1_tmp(4), b_1(4)
    const G_FUNCTION_TRACE_WIDHT: usize = 81;

    enum Blake3CompressionChips {
        U8Xor,
        U32Xor,
        U32Add,
        U32RightRotate8,
        U32RightRotate16,
        U32RightRotate12, // FIXME: currently underconstrained (range check is not performed). Needs rewriting using Gabriel's advice
        U32RightRotate7, // FIXME: currently underconstrained (range check is not performed). Needs rewriting using Gabriel's advice
        U8PairRangeCheck,
        GFunction, // FIXME: currently underconstrained (internal operations are not fully constrained).
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
                Blake3CompressionChips::U32RightRotate7 => 6,
                Blake3CompressionChips::U8PairRangeCheck => 7,
                Blake3CompressionChips::GFunction => 8,
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
                Self::U32RightRotate7 => U32_RIGHT_ROTATE_7_TRACE_WIDTH,
                Self::GFunction => G_FUNCTION_TRACE_WIDHT,
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
                Self::U32RightRotate7 => None,
                Self::GFunction => None,
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
                Self::U32RightRotate12 | Self::U32RightRotate7 => {
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
                Self::GFunction => { /* TODO: constrain operations */ }
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
            let u32_right_rotate_7_idx = Blake3CompressionChips::U32RightRotate7.position();
            let u8_pair_range_check_idx = Blake3CompressionChips::U8PairRangeCheck.position();
            let g_function_idx = Blake3CompressionChips::GFunction.position();

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

                Self::U32RightRotate7 => {
                    vec![Lookup::pull(
                        var(0),
                        vec![
                            SymbExpr::from_usize(u32_right_rotate_7_idx),
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

                // Totally 81 byte columns:
                // multiplicity, a_in(4), b_in(4), c_in(4), d_in(4), mx_in(4), my_in(4),
                // a_0_tmp(4), a_0(4), d_0_tmp(4), d_0(4), c_0(4), b_0_tmp(4), b_0(4),
                // a_1_tmp(4), a_1(4), d_1_tmp(4), d_1(4), c_1(4), b_1_tmp(4), b_1(4)
                Self::GFunction => {
                    vec![
                        // balancing the initial claim
                        Lookup::pull(
                            var(0),
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1) // a_in
                                    + var(2) * SymbExpr::from_u32(256)
                                    + var(3) * SymbExpr::from_u32(256 * 256)
                                    + var(4) * SymbExpr::from_u32(256 * 256 * 256),
                                var(5) // b_in
                                    + var(6) * SymbExpr::from_u32(256)
                                    + var(7) * SymbExpr::from_u32(256 * 256)
                                    + var(8) * SymbExpr::from_u32(256 * 256 * 256),
                                var(9) // c_in
                                    + var(10) * SymbExpr::from_u32(256)
                                    + var(11) * SymbExpr::from_u32(256 * 256)
                                    + var(12) * SymbExpr::from_u32(256 * 256 * 256),
                                var(13) // d_in
                                    + var(14) * SymbExpr::from_u32(256)
                                    + var(15) * SymbExpr::from_u32(256 * 256)
                                    + var(16) * SymbExpr::from_u32(256 * 256 * 256),
                                var(17) // mx_in
                                    + var(18) * SymbExpr::from_u32(256)
                                    + var(19) * SymbExpr::from_u32(256 * 256)
                                    + var(20) * SymbExpr::from_u32(256 * 256 * 256),
                                var(21) // my_in
                                    + var(22) * SymbExpr::from_u32(256)
                                    + var(23) * SymbExpr::from_u32(256 * 256)
                                    + var(24) * SymbExpr::from_u32(256 * 256 * 256),
                                // var(25) // a_0_tmp
                                //     + var(26) * SymbExpr::from_u32(256)
                                //     + var(27) * SymbExpr::from_u32(256 * 256)
                                //     + var(28) * SymbExpr::from_u32(256 * 256 * 256),
                                // var(29) // a_0
                                //     + var(30) * SymbExpr::from_u32(256)
                                //     + var(31) * SymbExpr::from_u32(256 * 256)
                                //     + var(32) * SymbExpr::from_u32(256 * 256 * 256),
                                // var(33) // d_0_tmp
                                //     + var(34) * SymbExpr::from_u32(256)
                                //     + var(35) * SymbExpr::from_u32(256 * 256)
                                //     + var(36) * SymbExpr::from_u32(256 * 256 * 256),
                                // var(37) // d_0
                                //     + var(38) * SymbExpr::from_u32(256)
                                //     + var(39) * SymbExpr::from_u32(256 * 256)
                                //     + var(40) * SymbExpr::from_u32(256 * 256 * 256),
                                // var(41) // c_0
                                //     + var(42) * SymbExpr::from_u32(256)
                                //     + var(43) * SymbExpr::from_u32(256 * 256)
                                //     + var(44) * SymbExpr::from_u32(256 * 256 * 256),
                                // var(45) // b_0_tmp
                                //     + var(46) * SymbExpr::from_u32(256)
                                //     + var(47) * SymbExpr::from_u32(256 * 256)
                                //     + var(48) * SymbExpr::from_u32(256 * 256 * 256),
                                // var(49) // b_0
                                //     + var(50) * SymbExpr::from_u32(256)
                                //     + var(51) * SymbExpr::from_u32(256 * 256)
                                //     + var(52) * SymbExpr::from_u32(256 * 256 * 256),
                                // var(53) // a_1_tmp
                                //     + var(54) * SymbExpr::from_u32(256)
                                //     + var(55) * SymbExpr::from_u32(256 * 256)
                                //     + var(56) * SymbExpr::from_u32(256 * 256 * 256),
                                var(57) // a_1
                                    + var(58) * SymbExpr::from_u32(256)
                                    + var(59) * SymbExpr::from_u32(256 * 256)
                                    + var(60) * SymbExpr::from_u32(256 * 256 * 256),
                                // var(61) // d_1_tmp
                                //     + var(62) * SymbExpr::from_u32(256)
                                //     + var(63) * SymbExpr::from_u32(256 * 256)
                                //     + var(64) * SymbExpr::from_u32(256 * 256 * 256),
                                var(65) // d_1
                                    + var(66) * SymbExpr::from_u32(256)
                                    + var(67) * SymbExpr::from_u32(256 * 256)
                                    + var(68) * SymbExpr::from_u32(256 * 256 * 256),
                                var(69) // c_1
                                    + var(70) * SymbExpr::from_u32(256)
                                    + var(71) * SymbExpr::from_u32(256 * 256)
                                    + var(72) * SymbExpr::from_u32(256 * 256 * 256),
                                // var(73) // b_1_tmp
                                //     + var(74) * SymbExpr::from_u32(256)
                                //     + var(75) * SymbExpr::from_u32(256 * 256)
                                //     + var(76) * SymbExpr::from_u32(256 * 256 * 256),
                                var(77) // b_1
                                    + var(78) * SymbExpr::from_u32(256)
                                    + var(79) * SymbExpr::from_u32(256 * 256)
                                    + var(80) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        // balancing lower-level chips used in G function

                        // a_in + b_in = a_0_tmp
                        // Lookup::push(
                        //     SymbExpr::ONE,
                        //     vec![
                        //         SymbExpr::from_usize(u32_add_idx),
                        //         var(1) // a_in
                        //             + var(2) * SymbExpr::from_u32(256)
                        //             + var(3) * SymbExpr::from_u32(256 * 256)
                        //             + var(4) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(5) // b_in
                        //             + var(6) * SymbExpr::from_u32(256)
                        //             + var(7) * SymbExpr::from_u32(256 * 256)
                        //             + var(8) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(25) // a_0_tmp
                        //             + var(26) * SymbExpr::from_u32(256)
                        //             + var(27) * SymbExpr::from_u32(256 * 256)
                        //             + var(28) * SymbExpr::from_u32(256 * 256 * 256),
                        //     ],
                        // ),

                        // a_0_tmp + mx_in = a_0
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_add_idx),
                                var(25) // a_0_tmp
                                    + var(26) * SymbExpr::from_u32(256)
                                    + var(27) * SymbExpr::from_u32(256 * 256)
                                    + var(28) * SymbExpr::from_u32(256 * 256 * 256),
                                var(17) // mx_in
                                    + var(18) * SymbExpr::from_u32(256)
                                    + var(19) * SymbExpr::from_u32(256 * 256)
                                    + var(20) * SymbExpr::from_u32(256 * 256 * 256),
                                var(29) // a_0
                                    + var(30) * SymbExpr::from_u32(256)
                                    + var(31) * SymbExpr::from_u32(256 * 256)
                                    + var(32) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        // d_in ^ a_0 = d_0_tmp
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(13) // d_in
                                    + var(14) * SymbExpr::from_u32(256)
                                    + var(15) * SymbExpr::from_u32(256 * 256)
                                    + var(16) * SymbExpr::from_u32(256 * 256 * 256),
                                var(29) // a_0
                                    + var(30) * SymbExpr::from_u32(256)
                                    + var(31) * SymbExpr::from_u32(256 * 256)
                                    + var(32) * SymbExpr::from_u32(256 * 256 * 256),
                                var(33) // d_0_tmp
                                    + var(34) * SymbExpr::from_u32(256)
                                    + var(35) * SymbExpr::from_u32(256 * 256)
                                    + var(36) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        // d_0_tmp >> 16 = d_0
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_right_rotate_16_idx),
                                var(33) // d_0_tmp
                                    + var(34) * SymbExpr::from_u32(256)
                                    + var(35) * SymbExpr::from_u32(256 * 256)
                                    + var(36) * SymbExpr::from_u32(256 * 256 * 256),
                                var(37) // d_0
                                    + var(38) * SymbExpr::from_u32(256)
                                    + var(39) * SymbExpr::from_u32(256 * 256)
                                    + var(40) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        // c_in + d_0 = c_0
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_add_idx),
                                var(9) // c_in
                                    + var(10) * SymbExpr::from_u32(256)
                                    + var(11) * SymbExpr::from_u32(256 * 256)
                                    + var(12) * SymbExpr::from_u32(256 * 256 * 256),
                                var(37) // d_0
                                    + var(38) * SymbExpr::from_u32(256)
                                    + var(39) * SymbExpr::from_u32(256 * 256)
                                    + var(40) * SymbExpr::from_u32(256 * 256 * 256),
                                var(41) // c_0
                                    + var(42) * SymbExpr::from_u32(256)
                                    + var(43) * SymbExpr::from_u32(256 * 256)
                                    + var(44) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        // b_in ^ c_0 = b_0_tmp
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(5) // b_in
                                    + var(6) * SymbExpr::from_u32(256)
                                    + var(7) * SymbExpr::from_u32(256 * 256)
                                    + var(8) * SymbExpr::from_u32(256 * 256 * 256),
                                var(41) // c_0
                                    + var(42) * SymbExpr::from_u32(256)
                                    + var(43) * SymbExpr::from_u32(256 * 256)
                                    + var(44) * SymbExpr::from_u32(256 * 256 * 256),
                                var(45) // b_0_tmp
                                    + var(46) * SymbExpr::from_u32(256)
                                    + var(47) * SymbExpr::from_u32(256 * 256)
                                    + var(48) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        // b_0_tmp >> 12 = b_0
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_right_rotate_12_idx),
                                var(45) // b_0_tmp
                                    + var(46) * SymbExpr::from_u32(256)
                                    + var(47) * SymbExpr::from_u32(256 * 256)
                                    + var(48) * SymbExpr::from_u32(256 * 256 * 256),
                                var(49) // b_0
                                    + var(50) * SymbExpr::from_u32(256)
                                    + var(51) * SymbExpr::from_u32(256 * 256)
                                    + var(52) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        // a_0 + b_0 = a_1_tmp
                        // Lookup::push(
                        //     SymbExpr::ONE,
                        //     vec![
                        //         SymbExpr::from_usize(u32_add_idx),
                        //         var(29) // a_0
                        //             + var(30) * SymbExpr::from_u32(256)
                        //             + var(31) * SymbExpr::from_u32(256 * 256)
                        //             + var(32) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(49) // b_0
                        //             + var(50) * SymbExpr::from_u32(256)
                        //             + var(51) * SymbExpr::from_u32(256 * 256)
                        //             + var(52) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(53) // a_1_tmp
                        //             + var(54) * SymbExpr::from_u32(256)
                        //             + var(55) * SymbExpr::from_u32(256 * 256)
                        //             + var(56) * SymbExpr::from_u32(256 * 256 * 256),
                        //     ],
                        // ),

                        // a_1_tmp, my_in, a_1
                        // Lookup::push(
                        //     SymbExpr::ONE,
                        //     vec![
                        //         SymbExpr::from_usize(u32_add_idx),
                        //         var(53) // a_1_tmp
                        //             + var(54) * SymbExpr::from_u32(256)
                        //             + var(55) * SymbExpr::from_u32(256 * 256)
                        //             + var(56) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(21) // my_in
                        //             + var(22) * SymbExpr::from_u32(256)
                        //             + var(23) * SymbExpr::from_u32(256 * 256)
                        //             + var(24) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(57) // a_1
                        //             + var(58) * SymbExpr::from_u32(256)
                        //             + var(59) * SymbExpr::from_u32(256 * 256)
                        //             + var(60) * SymbExpr::from_u32(256 * 256 * 256),
                        //     ],
                        // ),

                        // d_0 ^ a_1 = d_1_tmp
                        // Lookup::push(
                        //     SymbExpr::ONE,
                        //     vec![
                        //         SymbExpr::from_usize(u32_xor_idx),
                        //         var(37) // d_0
                        //             + var(38) * SymbExpr::from_u32(256)
                        //             + var(39) * SymbExpr::from_u32(256 * 256)
                        //             + var(40) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(57) // a_1
                        //             + var(58) * SymbExpr::from_u32(256)
                        //             + var(59) * SymbExpr::from_u32(256 * 256)
                        //             + var(60) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(61) // d_1_tmp
                        //             + var(62) * SymbExpr::from_u32(256)
                        //             + var(63) * SymbExpr::from_u32(256 * 256)
                        //             + var(64) * SymbExpr::from_u32(256 * 256 * 256),
                        //     ],
                        // ),

                        // d_1_tmp >> 8 = d_1
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_right_rotate_8_idx),
                                var(61) // d_1_tmp
                                    + var(62) * SymbExpr::from_u32(256)
                                    + var(63) * SymbExpr::from_u32(256 * 256)
                                    + var(64) * SymbExpr::from_u32(256 * 256 * 256),
                                var(65) // d_1
                                    + var(66) * SymbExpr::from_u32(256)
                                    + var(67) * SymbExpr::from_u32(256 * 256)
                                    + var(68) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        // c_0 + d_1 = c_1
                        // Lookup::push(
                        //     SymbExpr::ONE,
                        //     vec![
                        //         SymbExpr::from_usize(u32_add_idx),
                        //         var(41) // c_0
                        //             + var(42) * SymbExpr::from_u32(256)
                        //             + var(43) * SymbExpr::from_u32(256 * 256)
                        //             + var(44) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(65) // d_1
                        //             + var(66) * SymbExpr::from_u32(256)
                        //             + var(67) * SymbExpr::from_u32(256 * 256)
                        //             + var(68) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(69) // c_1
                        //             + var(70) * SymbExpr::from_u32(256)
                        //             + var(71) * SymbExpr::from_u32(256 * 256)
                        //             + var(72) * SymbExpr::from_u32(256 * 256 * 256),
                        //     ],
                        // ),

                        // b_0 ^ c_1 = b_1_tmp
                        // Lookup::push(
                        //     SymbExpr::ONE,
                        //     vec![
                        //         SymbExpr::from_usize(u32_xor_idx),
                        //         var(49) // b_0
                        //             + var(50) * SymbExpr::from_u32(256)
                        //             + var(51) * SymbExpr::from_u32(256 * 256)
                        //             + var(52) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(69) // c_1
                        //             + var(70) * SymbExpr::from_u32(256)
                        //             + var(71) * SymbExpr::from_u32(256 * 256)
                        //             + var(72) * SymbExpr::from_u32(256 * 256 * 256),
                        //         var(73) // b_1_tmp
                        //             + var(74) * SymbExpr::from_u32(256)
                        //             + var(75) * SymbExpr::from_u32(256 * 256)
                        //             + var(76) * SymbExpr::from_u32(256 * 256 * 256),
                        //     ],
                        // ),

                        // b_1_tmp >> 7 = b_1
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_right_rotate_7_idx),
                                var(73) // b_1_tmp
                                    + var(74) * SymbExpr::from_u32(256)
                                    + var(75) * SymbExpr::from_u32(256 * 256)
                                    + var(76) * SymbExpr::from_u32(256 * 256 * 256),
                                var(77) // b_1
                                    + var(78) * SymbExpr::from_u32(256)
                                    + var(79) * SymbExpr::from_u32(256 * 256)
                                    + var(80) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                    ]
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
            // Grabbing values from a claims

            let mut u32_xor_values_from_claims = vec![];

            let mut u32_add_values_from_claims = vec![];

            let mut byte_xor_values_from_claims = vec![];

            let mut byte_range_check_values_from_claims = vec![];

            let mut u32_rotate_right_8_values_from_claims = vec![];

            let mut u32_rotate_right_16_values_from_claims = vec![];

            let mut u32_rotate_right_12_values_from_claims = vec![];

            let mut u32_rotate_right_7_values_from_claims = vec![];

            let mut g_function_values_from_claims = vec![];

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
                    }

                    2u64 => {
                        /* This is our U32Add claim. We should have chip_idx, A, B, A + B (where A, B are u32) */

                        assert!(claim.len() == 4, "[U32Add] wrong claim format");
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let b_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                        let add_u32 = u32::try_from(claim[3].as_canonical_u64()).unwrap();

                        u32_add_values_from_claims.push((a_u32, b_u32, add_u32));
                    }
                    3u64 => {
                        /* This is our U32RotateRight8 claim. We should have chip_idx, A, A_rot */

                        assert!(claim.len() == 3, "[U32RightRotate8] wrong claim format");
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();

                        u32_rotate_right_8_values_from_claims.push((a_u32, rot_u32));
                    }
                    4u64 => {
                        /* This is our U32RotateRight16 claim. We should have chip_idx, A, A_rot */

                        assert!(claim.len() == 3, "[U32RightRotate16] wrong claim format");
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();

                        u32_rotate_right_16_values_from_claims.push((a_u32, rot_u32));
                    }
                    5u64 => {
                        /* This is our U32RotateRight12 claim. We should have chip_idx, A, A_rot */

                        assert!(claim.len() == 3, "[U32RightRotate12] wrong claim format");
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();

                        u32_rotate_right_12_values_from_claims.push((a_u32, rot_u32));
                    }
                    6u64 => {
                        /* This is our U32RotateRight7 claim. We should have chip_idx, A, A_rot */

                        assert!(claim.len() == 3, "[U32RightRotate7] wrong claim format");
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let rot_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();

                        u32_rotate_right_7_values_from_claims.push((a_u32, rot_u32));
                    }

                    7u64 => {
                        /* This is our U8PairRangeCheck claim. We should have chip_idx, A, B */

                        assert!(claim.len() == 3, "[U8Xor] wrong claim format");
                        byte_range_check_values_from_claims.push((claim[1], claim[2]));
                    }

                    8u64 => {
                        /* This is our GFunction claim. We should have chip_idx, A, B, C, D, MX_IN, MY_IN, A1, D1, C1, B1 */
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

                    _ => panic!("unsupported chip"),
                }
            }

            // Build traces. If claim for a given chip was not provided (and hence no data available), we just use zero trace
            // and balance lookups manually

            // build GFunction trace columns:
            // multiplicity, a_in(4), b_in(4), c_in(4), d_in(4), mx_in(4), my_in(4),
            // a_0_tmp(4), a_0(4), d_0_tmp(4), d_0(4), c_0(4), b_0_tmp(4), b_0(4),
            // a_1_tmp(4), a_1(4), d_1_tmp(4), d_1(4), c_1(4), b_1_tmp(4), b_1(4))
            let mut g_function_trace_values =
                Vec::<Val>::with_capacity(g_function_values_from_claims.len());
            if g_function_values_from_claims.is_empty() {
                g_function_trace_values = Val::zero_vec(G_FUNCTION_TRACE_WIDHT);
            } else {
                for (a_in, b_in, c_in, d_in, mx_in, my_in, a1, b1, c1, d1) in
                    g_function_values_from_claims.into_iter()
                {
                    let a_0_tmp = a_in.wrapping_add(b_in);
                    // u32_add_values_from_claims.push((a_in, b_in, a_0_tmp)); // send data to U32Add chip TODO

                    let a_0 = a_0_tmp.wrapping_add(mx_in);
                    u32_add_values_from_claims.push((a_0_tmp, mx_in, a_0)); // send data to U32Add chip

                    let d_0_tmp = d_in ^ a_0;
                    u32_xor_values_from_claims.push((d_in, a_0, d_0_tmp)); // send data to U32Xor chip

                    let d_0 = d_0_tmp.rotate_right(16);
                    u32_rotate_right_16_values_from_claims.push((d_0_tmp, d_0)); // send data to U32RightRotate16 chip

                    let c_0 = c_in.wrapping_add(d_0);
                    u32_add_values_from_claims.push((c_in, d_0, c_0)); // send data to U32Add chip

                    let b_0_tmp = b_in ^ c_0;
                    u32_xor_values_from_claims.push((b_in, c_0, b_0_tmp)); // send data to U32Xor chip

                    let b_0 = b_0_tmp.rotate_right(12);
                    u32_rotate_right_12_values_from_claims.push((b_0_tmp, b_0)); // send data to U32RightRotate12 chip

                    let a_1_tmp = a_0.wrapping_add(b_0);
                    // u32_add_values_from_claims.push((a_0, b_0, a_1_tmp)); // send data to U32Add chip TODO

                    let a_1 = a_1_tmp.wrapping_add(my_in);
                    // u32_add_values_from_claims.push((a_1_tmp, my_in, a_1)); // send data to U32Add chip TODO

                    let d_1_tmp = d_0 ^ a_1;
                    // u32_xor_values_from_claims.push((d_0, a_1, d_1_tmp)); // send data to U32Xor chip TODO

                    let d_1 = d_1_tmp.rotate_right(8);
                    u32_rotate_right_8_values_from_claims.push((d_1_tmp, d_1));

                    let c_1 = c_0.wrapping_add(d_1);
                    // u32_add_values_from_claims.push((c_0, d_1, c_1)); // send data to U32Add chip TODO

                    let b_1_tmp = b_0 ^ c_1;
                    // u32_xor_values_from_claims.push((b_0, c_1, b_1_tmp)); // send data to U32Xor chip TODO

                    let b_1 = b_1_tmp.rotate_right(7);
                    u32_rotate_right_7_values_from_claims.push((b_1_tmp, b_1));

                    debug_assert_eq!(a_1, a1);
                    debug_assert_eq!(b_1, b1);
                    debug_assert_eq!(c_1, c1);
                    debug_assert_eq!(d_1, d1);

                    let a_in_bytes: [u8; 4] = a_in.to_le_bytes();
                    let b_in_bytes: [u8; 4] = b_in.to_le_bytes();
                    let c_in_bytes: [u8; 4] = c_in.to_le_bytes();
                    let d_in_bytes: [u8; 4] = d_in.to_le_bytes();
                    let mx_in_bytes: [u8; 4] = mx_in.to_le_bytes();
                    let my_in_bytes: [u8; 4] = my_in.to_le_bytes();
                    let a_0_tmp_bytes: [u8; 4] = a_0_tmp.to_le_bytes();
                    let a_0_bytes: [u8; 4] = a_0.to_le_bytes();
                    let d_0_tmp_bytes: [u8; 4] = d_0_tmp.to_le_bytes();
                    let d_0_bytes: [u8; 4] = d_0.to_le_bytes();
                    let c_0_bytes: [u8; 4] = c_0.to_le_bytes();
                    let b_0_tmp_bytes: [u8; 4] = b_0_tmp.to_le_bytes();
                    let b_0_bytes: [u8; 4] = b_0.to_le_bytes();
                    let a_1_tmp_bytes: [u8; 4] = a_1_tmp.to_le_bytes();
                    let a_1_bytes: [u8; 4] = a_1.to_le_bytes();
                    let d_1_tmp_bytes: [u8; 4] = d_1_tmp.to_le_bytes();
                    let d_1_bytes: [u8; 4] = d_1.to_le_bytes();
                    let c_1_bytes: [u8; 4] = c_1.to_le_bytes();
                    let b_1_tmp_bytes: [u8; 4] = b_1_tmp.to_le_bytes();
                    let b_1_bytes: [u8; 4] = b_1.to_le_bytes();

                    g_function_trace_values.push(Val::ONE); // multiplicity
                    g_function_trace_values
                        .extend_from_slice(a_in_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(b_in_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(c_in_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(d_in_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(mx_in_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(my_in_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(a_0_tmp_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(a_0_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(d_0_tmp_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(d_0_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(c_0_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(b_0_tmp_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(b_0_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(a_1_tmp_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(a_1_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(d_1_tmp_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(d_1_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(c_1_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(b_1_tmp_bytes.map(Val::from_u8).as_slice());
                    g_function_trace_values
                        .extend_from_slice(b_1_bytes.map(Val::from_u8).as_slice());
                }
            }
            let mut g_function_trace =
                RowMajorMatrix::new(g_function_trace_values, G_FUNCTION_TRACE_WIDHT);
            let height = g_function_trace.height().next_power_of_two();
            g_function_trace.pad_to_height(height, Val::ZERO);

            // build U32Xor trace (columns: multiplicity, A0, A1, A2, A3, B0, B1, B2, B3, A0^B0, A1^B1, A2^B2, A3^B3)
            let mut u32_xor_trace_values =
                Vec::<Val>::with_capacity(u32_xor_values_from_claims.len());
            if u32_xor_values_from_claims.is_empty() {
                u32_xor_trace_values = Val::zero_vec(U32_XOR_TRACE_WIDTH);

                // we also need to balance the U8Xor chip lookups using zeroes

                byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
                byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
                byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
                byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
            } else {
                for (a, b, a_xor_b) in u32_xor_values_from_claims.into_iter() {
                    debug_assert_eq!(a ^ b, a_xor_b);

                    let a_bytes: [u8; 4] = a.to_le_bytes();
                    let b_bytes: [u8; 4] = b.to_le_bytes();
                    let a_xor_b_bytes: [u8; 4] = a_xor_b.to_le_bytes();

                    u32_xor_trace_values.push(Val::ONE); // multiplicity

                    u32_xor_trace_values.extend_from_slice(a_bytes.map(Val::from_u8).as_slice());
                    u32_xor_trace_values.extend_from_slice(b_bytes.map(Val::from_u8).as_slice());
                    u32_xor_trace_values
                        .extend_from_slice(a_xor_b_bytes.map(Val::from_u8).as_slice());

                    /* we send bytes to U8Xor chip, relying on lookup constraining */

                    byte_xor_values_from_claims.push((
                        Val::from_u8(a_bytes[0]),
                        Val::from_u8(b_bytes[0]),
                        Val::from_u8(a_xor_b_bytes[0]),
                    ));
                    byte_xor_values_from_claims.push((
                        Val::from_u8(a_bytes[1]),
                        Val::from_u8(b_bytes[1]),
                        Val::from_u8(a_xor_b_bytes[1]),
                    ));
                    byte_xor_values_from_claims.push((
                        Val::from_u8(a_bytes[2]),
                        Val::from_u8(b_bytes[2]),
                        Val::from_u8(a_xor_b_bytes[2]),
                    ));
                    byte_xor_values_from_claims.push((
                        Val::from_u8(a_bytes[3]),
                        Val::from_u8(b_bytes[3]),
                        Val::from_u8(a_xor_b_bytes[3]),
                    ));
                }
            }
            let mut u32_xor_trace = RowMajorMatrix::new(u32_xor_trace_values, U32_XOR_TRACE_WIDTH);
            let height = u32_xor_trace.height().next_power_of_two();
            u32_xor_trace.pad_to_height(height, Val::ZERO);

            // build U32Add trace (columns: A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3, carry, multiplicity)
            let mut u32_add_trace_values = vec![];
            if u32_add_values_from_claims.is_empty() {
                u32_add_trace_values = Val::zero_vec(U32_ADD_TRACE_WIDTH);

                // we also need to balance the lookups using zeroes
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));

                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            } else {
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

                    /* we send decomposed bytes to U8Xor chip, relying on lookup constraining */

                    byte_range_check_values_from_claims
                        .push((Val::from_u8(a_bytes[0]), Val::from_u8(b_bytes[0])));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(a_bytes[1]), Val::from_u8(b_bytes[1])));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(a_bytes[2]), Val::from_u8(b_bytes[2])));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(a_bytes[3]), Val::from_u8(b_bytes[3])));

                    byte_range_check_values_from_claims.push((Val::from_u8(c_bytes[0]), Val::ZERO));
                    byte_range_check_values_from_claims.push((Val::from_u8(c_bytes[1]), Val::ZERO));
                    byte_range_check_values_from_claims.push((Val::from_u8(c_bytes[2]), Val::ZERO));
                    byte_range_check_values_from_claims.push((Val::from_u8(c_bytes[3]), Val::ZERO));
                }
            }
            let mut u32_add_trace = RowMajorMatrix::new(u32_add_trace_values, U32_ADD_TRACE_WIDTH);
            let height = u32_add_trace.height().next_power_of_two();
            u32_add_trace.pad_to_height(height, Val::ZERO);

            // build U32RotateRight8 trace (columns: multiplicity, a0, a1, a2, a3, rot0, rot1, rot2, rot3)
            let mut u32_rotate_right_8_trace_values = vec![];
            if u32_rotate_right_8_values_from_claims.is_empty() {
                u32_rotate_right_8_trace_values = Val::zero_vec(U32_RIGHT_ROTATE_8_TRACE_WIDTH);

                // we also need to balance U8PairRangeCheck chip lookups using zeroes

                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            } else {
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

                    /* we send decomposed bytes to U8PairRangeCheck chip, relying on lookup constraining */

                    byte_range_check_values_from_claims
                        .push((Val::from_u8(a_bytes[0]), Val::from_u8(a_bytes[2])));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(a_bytes[1]), Val::from_u8(a_bytes[3])));
                }
            }
            let mut u32_rotate_right_8_trace = RowMajorMatrix::new(
                u32_rotate_right_8_trace_values,
                U32_RIGHT_ROTATE_8_TRACE_WIDTH,
            );
            let height = u32_rotate_right_8_trace.height().next_power_of_two();
            u32_rotate_right_8_trace.pad_to_height(height, Val::ZERO);

            // build U32RotateRight16 trace (columns: multiplicity, a0, a1, a2, a3, rot0, rot1, rot2, rot3)
            let mut u32_rotate_right_16_trace_values = vec![];
            if u32_rotate_right_16_values_from_claims.is_empty() {
                u32_rotate_right_16_trace_values = Val::zero_vec(U32_RIGHT_ROTATE_16_TRACE_WIDTH);

                // we also need to balance the lookups using zeroes

                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            } else {
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

                    /* we send decomposed bytes to U8PairRangeCheck chip, relying on lookup constraining */

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
            u32_rotate_right_16_trace.pad_to_height(height, Val::ZERO);

            fn rot_7_12_trace_values(
                k: u32,
                vals_from_claim: Vec<(u32, u32)>,
            ) -> RowMajorMatrix<Val> {
                let width = match k {
                    7 => U32_RIGHT_ROTATE_7_TRACE_WIDTH,
                    12 => U32_RIGHT_ROTATE_12_TRACE_WIDTH,
                    _ => panic!("unexpected k"),
                };

                let mut values = vec![];
                if vals_from_claim.is_empty() {
                    values = Val::zero_vec(width);
                } else {
                    for (a, rot) in vals_from_claim.clone() {
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
                        values.extend_from_slice(
                            two_pow_32_minus_k_bytes.map(Val::from_u8).as_slice(),
                        );
                        values.extend_from_slice(input_div_bytes.map(Val::from_u8).as_slice());
                        values.extend_from_slice(input_rem_bytes.map(Val::from_u8).as_slice());
                    }
                }

                let mut trace = RowMajorMatrix::new(values, width);
                let height = trace.height().next_power_of_two();
                trace.pad_to_height(height, Val::ZERO);

                trace
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
            let u32_rotate_right_12_trace =
                rot_7_12_trace_values(12, u32_rotate_right_12_values_from_claims);

            // build U32RotateRight7 trace (columns same as for U32RotateRight12)
            let u32_rotate_right_7_trace =
                rot_7_12_trace_values(7, u32_rotate_right_7_values_from_claims);

            // finally build U8Xor / U8PairRangeCheck trace (columns: multiplicity_u8_xor, multiplicity_pair_range_check)
            // since this it "lowest-level" trace, its multiplicities could be updated by other chips previously
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
        let a_rot_7 = a_u32.rotate_right(7);

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
        let u32_rotate_right_7_circuit = LookupAir::new(
            Blake3CompressionChips::U32RightRotate7,
            Blake3CompressionChips::U32RightRotate7.lookups(),
        );
        let g_function_circuit = LookupAir::new(
            Blake3CompressionChips::GFunction,
            Blake3CompressionChips::GFunction.lookups(),
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
                u32_rotate_right_7_circuit,
                g_function_circuit,
            ],
        );

        let f = Val::from_u8;
        let f32 = Val::from_u32;

        let a_in = 0x11111111u32;
        let b_in = 0x22222222u32;
        let c_in = 0x33333333u32;
        let d_in = 0x44444444u32;
        let mx_in = 0x55555555u32;
        let my_in = 0x66666666u32;

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
                // TODO: figure out why multiple invocation of U32Xor and U32Add chips causes UnbalancedChannel
                // vec![
                //     Val::from_usize(Blake3CompressionChips::U32Xor.position()),
                //     f32(a_u32),
                //     f32(b_u32),
                //     f32(xor_u32),
                // ],
                // vec![
                //     Val::from_usize(Blake3CompressionChips::U32Add.position()),
                //     f32(a_u32),
                //     f32(b_u32),
                //     f32(add_u32),
                // ],
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
                    Val::from_usize(Blake3CompressionChips::U32RightRotate7.position()),
                    f32(a_u32),
                    f32(a_rot_7),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::GFunction.position()),
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

    #[test]
    fn g_function_test_vector() {
        let a_in = 0x11111111u32;
        let b_in = 0x22222222u32;
        let c_in = 0x33333333u32;
        let d_in = 0x44444444u32;
        let mx_in = 0x55555555u32;
        let my_in = 0x66666666u32;

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

        assert_eq!(a_1, 0xcccccccb);
        assert_eq!(b_1, 0x45b64444);
        assert_eq!(c_1, 0x06ffffff);
        assert_eq!(d_1, 0x07000000);
    }
}
