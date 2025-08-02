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

    // multiplicity, [state_in_0 (4), ... state_in_31 (4), [a_in (4), b_in (4), c_in (4), d_in (4), mx_in (4), my_in (4), a_1 (4), b_1 (4), c_1 (4), d_1 (4)] (x56), [state_i (4), state_i_8 (4), i_i8_xor (4)] (x8), [state_i (4), chaining_value_i (4), i_cv_xor (4)] (x8), state_out_0 (4), ... state_out_31 (4)]
    const STATE_TRANSITION_TRACE_WIDTH: usize = 1 + 32 * 4 * 2 + 40 * 56 + 12 * 8 * 2;

    enum Blake3CompressionChips {
        U8Xor,
        U32Xor,
        U32Add,
        U32RightRotate8,
        U32RightRotate16,
        U32RightRotate12, // FIXME: currently underconstrained (range check is not performed). Needs rewriting using Gabriel's advice
        U32RightRotate7, // FIXME: currently underconstrained (range check is not performed). Needs rewriting using Gabriel's advice
        U8PairRangeCheck,
        GFunction,
        StateTransition,
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
                Blake3CompressionChips::StateTransition => 9,
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
                Self::StateTransition => STATE_TRANSITION_TRACE_WIDTH,
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
                    for i in 0..BYTE_VALUES_NUM {
                        for j in 0..BYTE_VALUES_NUM {
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
                Self::StateTransition => None,
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
                Self::GFunction => {}
                Self::StateTransition => {}
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
            let state_transition_idx = Blake3CompressionChips::StateTransition.position();

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

                // (4 push lookups to u8_xor_chip)
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

                // (8 push lookups to pair_range_check)
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

                // (2 push lookups to pair_range_check)
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

                // (2 push lookups to pair_range_check)
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
                        // interacting with lower-level chips that constrain operations used in G function

                        // a_in + b_in = a_0_tmp
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_add_idx),
                                var(1) // a_in
                                    + var(2) * SymbExpr::from_u32(256)
                                    + var(3) * SymbExpr::from_u32(256 * 256)
                                    + var(4) * SymbExpr::from_u32(256 * 256 * 256),
                                var(5) // b_in
                                    + var(6) * SymbExpr::from_u32(256)
                                    + var(7) * SymbExpr::from_u32(256 * 256)
                                    + var(8) * SymbExpr::from_u32(256 * 256 * 256),
                                var(25) // a_0_tmp
                                    + var(26) * SymbExpr::from_u32(256)
                                    + var(27) * SymbExpr::from_u32(256 * 256)
                                    + var(28) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
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
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_add_idx),
                                var(29) // a_0
                                    + var(30) * SymbExpr::from_u32(256)
                                    + var(31) * SymbExpr::from_u32(256 * 256)
                                    + var(32) * SymbExpr::from_u32(256 * 256 * 256),
                                var(49) // b_0
                                    + var(50) * SymbExpr::from_u32(256)
                                    + var(51) * SymbExpr::from_u32(256 * 256)
                                    + var(52) * SymbExpr::from_u32(256 * 256 * 256),
                                var(53) // a_1_tmp
                                    + var(54) * SymbExpr::from_u32(256)
                                    + var(55) * SymbExpr::from_u32(256 * 256)
                                    + var(56) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        // a_1_tmp, my_in, a_1
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_add_idx),
                                var(53) // a_1_tmp
                                    + var(54) * SymbExpr::from_u32(256)
                                    + var(55) * SymbExpr::from_u32(256 * 256)
                                    + var(56) * SymbExpr::from_u32(256 * 256 * 256),
                                var(21) // my_in
                                    + var(22) * SymbExpr::from_u32(256)
                                    + var(23) * SymbExpr::from_u32(256 * 256)
                                    + var(24) * SymbExpr::from_u32(256 * 256 * 256),
                                var(57) // a_1
                                    + var(58) * SymbExpr::from_u32(256)
                                    + var(59) * SymbExpr::from_u32(256 * 256)
                                    + var(60) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        // d_0 ^ a_1 = d_1_tmp
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(37) // d_0
                                    + var(38) * SymbExpr::from_u32(256)
                                    + var(39) * SymbExpr::from_u32(256 * 256)
                                    + var(40) * SymbExpr::from_u32(256 * 256 * 256),
                                var(57) // a_1
                                    + var(58) * SymbExpr::from_u32(256)
                                    + var(59) * SymbExpr::from_u32(256 * 256)
                                    + var(60) * SymbExpr::from_u32(256 * 256 * 256),
                                var(61) // d_1_tmp
                                    + var(62) * SymbExpr::from_u32(256)
                                    + var(63) * SymbExpr::from_u32(256 * 256)
                                    + var(64) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
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
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_add_idx),
                                var(41) // c_0
                                    + var(42) * SymbExpr::from_u32(256)
                                    + var(43) * SymbExpr::from_u32(256 * 256)
                                    + var(44) * SymbExpr::from_u32(256 * 256 * 256),
                                var(65) // d_1
                                    + var(66) * SymbExpr::from_u32(256)
                                    + var(67) * SymbExpr::from_u32(256 * 256)
                                    + var(68) * SymbExpr::from_u32(256 * 256 * 256),
                                var(69) // c_1
                                    + var(70) * SymbExpr::from_u32(256)
                                    + var(71) * SymbExpr::from_u32(256 * 256)
                                    + var(72) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
                        //b_0 ^ c_1 = b_1_tmp
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(49) // b_0
                                    + var(50) * SymbExpr::from_u32(256)
                                    + var(51) * SymbExpr::from_u32(256 * 256)
                                    + var(52) * SymbExpr::from_u32(256 * 256 * 256),
                                var(69) // c_1
                                    + var(70) * SymbExpr::from_u32(256)
                                    + var(71) * SymbExpr::from_u32(256 * 256)
                                    + var(72) * SymbExpr::from_u32(256 * 256 * 256),
                                var(73) // b_1_tmp
                                    + var(74) * SymbExpr::from_u32(256)
                                    + var(75) * SymbExpr::from_u32(256 * 256)
                                    + var(76) * SymbExpr::from_u32(256 * 256 * 256),
                            ],
                        ),
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

                // multiplicity,
                // state_in (32 * 4),
                // a_in (4), b_in (4), c_in (4), d_in (4), mx_in (4), my_in (4), a_1 (4), d_1 (4), c_1 (4), b_1 (4)
                // state_out (32 * 4),
                Self::StateTransition => {
                    // let mut offset = 1; // we start from 1 since at var(0) we have multiplicity
                    //
                    // let indices: [usize; 128] = array::from_fn(|i| i + offset);
                    // let state_in_symbolic = indices
                    //     .chunks(4)
                    //     .into_iter()
                    //     .map(|c|
                    //         var(c[0])
                    //             + var(c[1]) * SymbExpr::from_u32(256)
                    //             + var(c[2]) * SymbExpr::from_u32(256 * 256)
                    //             + var(c[3]) * SymbExpr::from_u32(256 * 256)
                    //     ).collect::<Vec<SymbExpr>>();
                    //
                    // offset += 128;
                    //
                    // let indices: [usize; 40] = array::from_fn(|i| i + offset);
                    // let g_function_io_symbolic = indices
                    //     .chunks(4)
                    //     .into_iter()
                    //     .map(|c|
                    //         var(c[0])
                    //             + var(c[1]) * SymbExpr::from_u32(256)
                    //             + var(c[2]) * SymbExpr::from_u32(256 * 256)
                    //             + var(c[3]) * SymbExpr::from_u32(256 * 256)
                    //     ).collect::<Vec<SymbExpr>>();
                    //
                    // offset += 40;
                    //
                    // let indices: [usize; 128] = array::from_fn(|i| i + offset);
                    // let state_out_symbolic = indices
                    //     .chunks(4)
                    //     .into_iter()
                    //     .map(|c|
                    //         var(c[0])
                    //             + var(c[1]) * SymbExpr::from_u32(256)
                    //             + var(c[2]) * SymbExpr::from_u32(256 * 256)
                    //             + var(c[3]) * SymbExpr::from_u32(256 * 256)
                    //     ).collect::<Vec<SymbExpr>>();
                    //
                    //
                    // vec![
                    //     Lookup::pull(
                    //         var(0),
                    //         [
                    //             vec![SymbExpr::from_usize(state_transition_idx)],
                    //             state_in_symbolic,
                    //             // g_function_io_symbolic,
                    //             state_out_symbolic
                    //         ].concat()
                    //     ),
                    // ]
                    vec![
                        Lookup::pull(
                            var(0),
                            vec![
                                SymbExpr::from_usize(state_transition_idx),
                                // state_in
                                var(1)
                                    + var(2) * SymbExpr::from_u32(256)
                                    + var(3) * SymbExpr::from_u32(65536)
                                    + var(4) * SymbExpr::from_u32(16777216),
                                var(5)
                                    + var(6) * SymbExpr::from_u32(256)
                                    + var(7) * SymbExpr::from_u32(65536)
                                    + var(8) * SymbExpr::from_u32(16777216),
                                var(9)
                                    + var(10) * SymbExpr::from_u32(256)
                                    + var(11) * SymbExpr::from_u32(65536)
                                    + var(12) * SymbExpr::from_u32(16777216),
                                var(13)
                                    + var(14) * SymbExpr::from_u32(256)
                                    + var(15) * SymbExpr::from_u32(65536)
                                    + var(16) * SymbExpr::from_u32(16777216),
                                var(17)
                                    + var(18) * SymbExpr::from_u32(256)
                                    + var(19) * SymbExpr::from_u32(65536)
                                    + var(20) * SymbExpr::from_u32(16777216),
                                var(21)
                                    + var(22) * SymbExpr::from_u32(256)
                                    + var(23) * SymbExpr::from_u32(65536)
                                    + var(24) * SymbExpr::from_u32(16777216),
                                var(25)
                                    + var(26) * SymbExpr::from_u32(256)
                                    + var(27) * SymbExpr::from_u32(65536)
                                    + var(28) * SymbExpr::from_u32(16777216),
                                var(29)
                                    + var(30) * SymbExpr::from_u32(256)
                                    + var(31) * SymbExpr::from_u32(65536)
                                    + var(32) * SymbExpr::from_u32(16777216),
                                var(33)
                                    + var(34) * SymbExpr::from_u32(256)
                                    + var(35) * SymbExpr::from_u32(65536)
                                    + var(36) * SymbExpr::from_u32(16777216),
                                var(37)
                                    + var(38) * SymbExpr::from_u32(256)
                                    + var(39) * SymbExpr::from_u32(65536)
                                    + var(40) * SymbExpr::from_u32(16777216),
                                var(41)
                                    + var(42) * SymbExpr::from_u32(256)
                                    + var(43) * SymbExpr::from_u32(65536)
                                    + var(44) * SymbExpr::from_u32(16777216),
                                var(45)
                                    + var(46) * SymbExpr::from_u32(256)
                                    + var(47) * SymbExpr::from_u32(65536)
                                    + var(48) * SymbExpr::from_u32(16777216),
                                var(49)
                                    + var(50) * SymbExpr::from_u32(256)
                                    + var(51) * SymbExpr::from_u32(65536)
                                    + var(52) * SymbExpr::from_u32(16777216),
                                var(53)
                                    + var(54) * SymbExpr::from_u32(256)
                                    + var(55) * SymbExpr::from_u32(65536)
                                    + var(56) * SymbExpr::from_u32(16777216),
                                var(57)
                                    + var(58) * SymbExpr::from_u32(256)
                                    + var(59) * SymbExpr::from_u32(65536)
                                    + var(60) * SymbExpr::from_u32(16777216),
                                var(61)
                                    + var(62) * SymbExpr::from_u32(256)
                                    + var(63) * SymbExpr::from_u32(65536)
                                    + var(64) * SymbExpr::from_u32(16777216),
                                var(65)
                                    + var(66) * SymbExpr::from_u32(256)
                                    + var(67) * SymbExpr::from_u32(65536)
                                    + var(68) * SymbExpr::from_u32(16777216),
                                var(69)
                                    + var(70) * SymbExpr::from_u32(256)
                                    + var(71) * SymbExpr::from_u32(65536)
                                    + var(72) * SymbExpr::from_u32(16777216),
                                var(73)
                                    + var(74) * SymbExpr::from_u32(256)
                                    + var(75) * SymbExpr::from_u32(65536)
                                    + var(76) * SymbExpr::from_u32(16777216),
                                var(77)
                                    + var(78) * SymbExpr::from_u32(256)
                                    + var(79) * SymbExpr::from_u32(65536)
                                    + var(80) * SymbExpr::from_u32(16777216),
                                var(81)
                                    + var(82) * SymbExpr::from_u32(256)
                                    + var(83) * SymbExpr::from_u32(65536)
                                    + var(84) * SymbExpr::from_u32(16777216),
                                var(85)
                                    + var(86) * SymbExpr::from_u32(256)
                                    + var(87) * SymbExpr::from_u32(65536)
                                    + var(88) * SymbExpr::from_u32(16777216),
                                var(89)
                                    + var(90) * SymbExpr::from_u32(256)
                                    + var(91) * SymbExpr::from_u32(65536)
                                    + var(92) * SymbExpr::from_u32(16777216),
                                var(93)
                                    + var(94) * SymbExpr::from_u32(256)
                                    + var(95) * SymbExpr::from_u32(65536)
                                    + var(96) * SymbExpr::from_u32(16777216),
                                var(97)
                                    + var(98) * SymbExpr::from_u32(256)
                                    + var(99) * SymbExpr::from_u32(65536)
                                    + var(100) * SymbExpr::from_u32(16777216),
                                var(101)
                                    + var(102) * SymbExpr::from_u32(256)
                                    + var(103) * SymbExpr::from_u32(65536)
                                    + var(104) * SymbExpr::from_u32(16777216),
                                var(105)
                                    + var(106) * SymbExpr::from_u32(256)
                                    + var(107) * SymbExpr::from_u32(65536)
                                    + var(108) * SymbExpr::from_u32(16777216),
                                var(109)
                                    + var(110) * SymbExpr::from_u32(256)
                                    + var(111) * SymbExpr::from_u32(65536)
                                    + var(112) * SymbExpr::from_u32(16777216),
                                var(113)
                                    + var(114) * SymbExpr::from_u32(256)
                                    + var(115) * SymbExpr::from_u32(65536)
                                    + var(116) * SymbExpr::from_u32(16777216),
                                var(117)
                                    + var(118) * SymbExpr::from_u32(256)
                                    + var(119) * SymbExpr::from_u32(65536)
                                    + var(120) * SymbExpr::from_u32(16777216),
                                var(121)
                                    + var(122) * SymbExpr::from_u32(256)
                                    + var(123) * SymbExpr::from_u32(65536)
                                    + var(124) * SymbExpr::from_u32(16777216),
                                var(125)
                                    + var(126) * SymbExpr::from_u32(256)
                                    + var(127) * SymbExpr::from_u32(65536)
                                    + var(128) * SymbExpr::from_u32(16777216),
                                // tmp_vars

                                // 0
                                // var(129) + var(130) * SymbExpr::from_u32(256) + var(131) * SymbExpr::from_u32(65536) + var(132) * SymbExpr::from_u32(16777216),
                                // var(133) + var(134) * SymbExpr::from_u32(256) + var(135) * SymbExpr::from_u32(65536) + var(136) * SymbExpr::from_u32(16777216),
                                // var(137) + var(138) * SymbExpr::from_u32(256) + var(139) * SymbExpr::from_u32(65536) + var(140) * SymbExpr::from_u32(16777216),
                                // var(141) + var(142) * SymbExpr::from_u32(256) + var(143) * SymbExpr::from_u32(65536) + var(144) * SymbExpr::from_u32(16777216),
                                // var(145) + var(146) * SymbExpr::from_u32(256) + var(147) * SymbExpr::from_u32(65536) + var(148) * SymbExpr::from_u32(16777216),
                                // var(149) + var(150) * SymbExpr::from_u32(256) + var(151) * SymbExpr::from_u32(65536) + var(152) * SymbExpr::from_u32(16777216),
                                // var(153) + var(154) * SymbExpr::from_u32(256) + var(155) * SymbExpr::from_u32(65536) + var(156) * SymbExpr::from_u32(16777216),
                                // var(157) + var(158) * SymbExpr::from_u32(256) + var(159) * SymbExpr::from_u32(65536) + var(160) * SymbExpr::from_u32(16777216),
                                // var(161) + var(162) * SymbExpr::from_u32(256) + var(163) * SymbExpr::from_u32(65536) + var(164) * SymbExpr::from_u32(16777216),
                                // var(165) + var(166) * SymbExpr::from_u32(256) + var(167) * SymbExpr::from_u32(65536) + var(168) * SymbExpr::from_u32(16777216),

                                // 1
                                // var(169) + var(170) * SymbExpr::from_u32(256) + var(171) * SymbExpr::from_u32(65536) + var(172) * SymbExpr::from_u32(16777216),
                                // var(173) + var(174) * SymbExpr::from_u32(256) + var(175) * SymbExpr::from_u32(65536) + var(176) * SymbExpr::from_u32(16777216),
                                // var(177) + var(178) * SymbExpr::from_u32(256) + var(179) * SymbExpr::from_u32(65536) + var(180) * SymbExpr::from_u32(16777216),
                                // var(181) + var(182) * SymbExpr::from_u32(256) + var(183) * SymbExpr::from_u32(65536) + var(184) * SymbExpr::from_u32(16777216),
                                // var(185) + var(186) * SymbExpr::from_u32(256) + var(187) * SymbExpr::from_u32(65536) + var(188) * SymbExpr::from_u32(16777216),
                                // var(189) + var(190) * SymbExpr::from_u32(256) + var(191) * SymbExpr::from_u32(65536) + var(192) * SymbExpr::from_u32(16777216),
                                // var(193) + var(194) * SymbExpr::from_u32(256) + var(195) * SymbExpr::from_u32(65536) + var(196) * SymbExpr::from_u32(16777216),
                                // var(197) + var(198) * SymbExpr::from_u32(256) + var(199) * SymbExpr::from_u32(65536) + var(200) * SymbExpr::from_u32(16777216),
                                // var(201) + var(202) * SymbExpr::from_u32(256) + var(203) * SymbExpr::from_u32(65536) + var(204) * SymbExpr::from_u32(16777216),
                                // var(205) + var(206) * SymbExpr::from_u32(256) + var(207) * SymbExpr::from_u32(65536) + var(208) * SymbExpr::from_u32(16777216),

                                // 2
                                // var(209) + var(210) * SymbExpr::from_u32(256) + var(211) * SymbExpr::from_u32(65536) + var(212) * SymbExpr::from_u32(16777216),
                                // var(213) + var(214) * SymbExpr::from_u32(256) + var(215) * SymbExpr::from_u32(65536) + var(216) * SymbExpr::from_u32(16777216),
                                // var(217) + var(218) * SymbExpr::from_u32(256) + var(219) * SymbExpr::from_u32(65536) + var(220) * SymbExpr::from_u32(16777216),
                                // var(221) + var(222) * SymbExpr::from_u32(256) + var(223) * SymbExpr::from_u32(65536) + var(224) * SymbExpr::from_u32(16777216),
                                // var(225) + var(226) * SymbExpr::from_u32(256) + var(227) * SymbExpr::from_u32(65536) + var(228) * SymbExpr::from_u32(16777216),
                                // var(229) + var(230) * SymbExpr::from_u32(256) + var(231) * SymbExpr::from_u32(65536) + var(232) * SymbExpr::from_u32(16777216),
                                // var(233) + var(234) * SymbExpr::from_u32(256) + var(235) * SymbExpr::from_u32(65536) + var(236) * SymbExpr::from_u32(16777216),
                                // var(237) + var(238) * SymbExpr::from_u32(256) + var(239) * SymbExpr::from_u32(65536) + var(240) * SymbExpr::from_u32(16777216),
                                // var(241) + var(242) * SymbExpr::from_u32(256) + var(243) * SymbExpr::from_u32(65536) + var(244) * SymbExpr::from_u32(16777216),
                                // var(245) + var(246) * SymbExpr::from_u32(256) + var(247) * SymbExpr::from_u32(65536) + var(248) * SymbExpr::from_u32(16777216),
                                //
                                // // 3
                                // var(249) + var(250) * SymbExpr::from_u32(256) + var(251) * SymbExpr::from_u32(65536) + var(252) * SymbExpr::from_u32(16777216),
                                // var(253) + var(254) * SymbExpr::from_u32(256) + var(255) * SymbExpr::from_u32(65536) + var(256) * SymbExpr::from_u32(16777216),
                                // var(257) + var(258) * SymbExpr::from_u32(256) + var(259) * SymbExpr::from_u32(65536) + var(260) * SymbExpr::from_u32(16777216),
                                // var(261) + var(262) * SymbExpr::from_u32(256) + var(263) * SymbExpr::from_u32(65536) + var(264) * SymbExpr::from_u32(16777216),
                                // var(265) + var(266) * SymbExpr::from_u32(256) + var(267) * SymbExpr::from_u32(65536) + var(268) * SymbExpr::from_u32(16777216),
                                // var(269) + var(270) * SymbExpr::from_u32(256) + var(271) * SymbExpr::from_u32(65536) + var(272) * SymbExpr::from_u32(16777216),
                                // var(273) + var(274) * SymbExpr::from_u32(256) + var(275) * SymbExpr::from_u32(65536) + var(276) * SymbExpr::from_u32(16777216),
                                // var(277) + var(278) * SymbExpr::from_u32(256) + var(279) * SymbExpr::from_u32(65536) + var(280) * SymbExpr::from_u32(16777216),
                                // var(281) + var(282) * SymbExpr::from_u32(256) + var(283) * SymbExpr::from_u32(65536) + var(284) * SymbExpr::from_u32(16777216),
                                // var(285) + var(286) * SymbExpr::from_u32(256) + var(287) * SymbExpr::from_u32(65536) + var(288) * SymbExpr::from_u32(16777216),

                                // 4
                                // var(289) + var(290) * SymbExpr::from_u32(256) + var(291) * SymbExpr::from_u32(65536) + var(292) * SymbExpr::from_u32(16777216),
                                // var(293) + var(294) * SymbExpr::from_u32(256) + var(295) * SymbExpr::from_u32(65536) + var(296) * SymbExpr::from_u32(16777216),
                                // var(297) + var(298) * SymbExpr::from_u32(256) + var(299) * SymbExpr::from_u32(65536) + var(300) * SymbExpr::from_u32(16777216),
                                // var(301) + var(302) * SymbExpr::from_u32(256) + var(303) * SymbExpr::from_u32(65536) + var(304) * SymbExpr::from_u32(16777216),
                                // var(305) + var(306) * SymbExpr::from_u32(256) + var(307) * SymbExpr::from_u32(65536) + var(308) * SymbExpr::from_u32(16777216),
                                // var(309) + var(310) * SymbExpr::from_u32(256) + var(311) * SymbExpr::from_u32(65536) + var(312) * SymbExpr::from_u32(16777216),
                                // var(313) + var(314) * SymbExpr::from_u32(256) + var(315) * SymbExpr::from_u32(65536) + var(316) * SymbExpr::from_u32(16777216),
                                // var(317) + var(318) * SymbExpr::from_u32(256) + var(319) * SymbExpr::from_u32(65536) + var(320) * SymbExpr::from_u32(16777216),
                                // var(321) + var(322) * SymbExpr::from_u32(256) + var(323) * SymbExpr::from_u32(65536) + var(324) * SymbExpr::from_u32(16777216),
                                // var(325) + var(326) * SymbExpr::from_u32(256) + var(327) * SymbExpr::from_u32(65536) + var(328) * SymbExpr::from_u32(16777216),

                                // 5
                                // var(329) + var(330) * SymbExpr::from_u32(256) + var(331) * SymbExpr::from_u32(65536) + var(332) * SymbExpr::from_u32(16777216),
                                // var(333) + var(334) * SymbExpr::from_u32(256) + var(335) * SymbExpr::from_u32(65536) + var(336) * SymbExpr::from_u32(16777216),
                                // var(337) + var(338) * SymbExpr::from_u32(256) + var(339) * SymbExpr::from_u32(65536) + var(340) * SymbExpr::from_u32(16777216),
                                // var(341) + var(342) * SymbExpr::from_u32(256) + var(343) * SymbExpr::from_u32(65536) + var(344) * SymbExpr::from_u32(16777216),
                                // var(345) + var(346) * SymbExpr::from_u32(256) + var(347) * SymbExpr::from_u32(65536) + var(348) * SymbExpr::from_u32(16777216),
                                // var(349) + var(350) * SymbExpr::from_u32(256) + var(351) * SymbExpr::from_u32(65536) + var(352) * SymbExpr::from_u32(16777216),
                                // var(353) + var(354) * SymbExpr::from_u32(256) + var(355) * SymbExpr::from_u32(65536) + var(356) * SymbExpr::from_u32(16777216),
                                // var(357) + var(358) * SymbExpr::from_u32(256) + var(359) * SymbExpr::from_u32(65536) + var(360) * SymbExpr::from_u32(16777216),
                                // var(361) + var(362) * SymbExpr::from_u32(256) + var(363) * SymbExpr::from_u32(65536) + var(364) * SymbExpr::from_u32(16777216),
                                // var(365) + var(366) * SymbExpr::from_u32(256) + var(367) * SymbExpr::from_u32(65536) + var(368) * SymbExpr::from_u32(16777216),

                                // 6
                                // var(369) + var(370) * SymbExpr::from_u32(256) + var(371) * SymbExpr::from_u32(65536) + var(372) * SymbExpr::from_u32(16777216),
                                // var(373) + var(374) * SymbExpr::from_u32(256) + var(375) * SymbExpr::from_u32(65536) + var(376) * SymbExpr::from_u32(16777216),
                                // var(377) + var(378) * SymbExpr::from_u32(256) + var(379) * SymbExpr::from_u32(65536) + var(380) * SymbExpr::from_u32(16777216),
                                // var(381) + var(382) * SymbExpr::from_u32(256) + var(383) * SymbExpr::from_u32(65536) + var(384) * SymbExpr::from_u32(16777216),
                                // var(385) + var(386) * SymbExpr::from_u32(256) + var(387) * SymbExpr::from_u32(65536) + var(388) * SymbExpr::from_u32(16777216),
                                // var(389) + var(390) * SymbExpr::from_u32(256) + var(391) * SymbExpr::from_u32(65536) + var(392) * SymbExpr::from_u32(16777216),
                                // var(393) + var(394) * SymbExpr::from_u32(256) + var(395) * SymbExpr::from_u32(65536) + var(396) * SymbExpr::from_u32(16777216),
                                // var(397) + var(398) * SymbExpr::from_u32(256) + var(399) * SymbExpr::from_u32(65536) + var(400) * SymbExpr::from_u32(16777216),
                                // var(401) + var(402) * SymbExpr::from_u32(256) + var(403) * SymbExpr::from_u32(65536) + var(404) * SymbExpr::from_u32(16777216),
                                // var(405) + var(406) * SymbExpr::from_u32(256) + var(407) * SymbExpr::from_u32(65536) + var(408) * SymbExpr::from_u32(16777216),

                                // 7
                                // var(409) + var(410) * SymbExpr::from_u32(256) + var(411) * SymbExpr::from_u32(65536) + var(412) * SymbExpr::from_u32(16777216),
                                // var(413) + var(414) * SymbExpr::from_u32(256) + var(415) * SymbExpr::from_u32(65536) + var(416) * SymbExpr::from_u32(16777216),
                                // var(417) + var(418) * SymbExpr::from_u32(256) + var(419) * SymbExpr::from_u32(65536) + var(420) * SymbExpr::from_u32(16777216),
                                // var(421) + var(422) * SymbExpr::from_u32(256) + var(423) * SymbExpr::from_u32(65536) + var(424) * SymbExpr::from_u32(16777216),
                                // var(425) + var(426) * SymbExpr::from_u32(256) + var(427) * SymbExpr::from_u32(65536) + var(428) * SymbExpr::from_u32(16777216),
                                // var(429) + var(430) * SymbExpr::from_u32(256) + var(431) * SymbExpr::from_u32(65536) + var(432) * SymbExpr::from_u32(16777216),
                                // var(433) + var(434) * SymbExpr::from_u32(256) + var(435) * SymbExpr::from_u32(65536) + var(436) * SymbExpr::from_u32(16777216),
                                // var(437) + var(438) * SymbExpr::from_u32(256) + var(439) * SymbExpr::from_u32(65536) + var(440) * SymbExpr::from_u32(16777216),
                                // var(441) + var(442) * SymbExpr::from_u32(256) + var(443) * SymbExpr::from_u32(65536) + var(444) * SymbExpr::from_u32(16777216),
                                // var(445) + var(446) * SymbExpr::from_u32(256) + var(447) * SymbExpr::from_u32(65536) + var(448) * SymbExpr::from_u32(16777216),

                                // 8
                                // var(449) + var(450) * SymbExpr::from_u32(256) + var(451) * SymbExpr::from_u32(65536) + var(452) * SymbExpr::from_u32(16777216),
                                // var(453) + var(454) * SymbExpr::from_u32(256) + var(455) * SymbExpr::from_u32(65536) + var(456) * SymbExpr::from_u32(16777216),
                                // var(457) + var(458) * SymbExpr::from_u32(256) + var(459) * SymbExpr::from_u32(65536) + var(460) * SymbExpr::from_u32(16777216),
                                // var(461) + var(462) * SymbExpr::from_u32(256) + var(463) * SymbExpr::from_u32(65536) + var(464) * SymbExpr::from_u32(16777216),
                                // var(465) + var(466) * SymbExpr::from_u32(256) + var(467) * SymbExpr::from_u32(65536) + var(468) * SymbExpr::from_u32(16777216),
                                // var(469) + var(470) * SymbExpr::from_u32(256) + var(471) * SymbExpr::from_u32(65536) + var(472) * SymbExpr::from_u32(16777216),
                                // var(473) + var(474) * SymbExpr::from_u32(256) + var(475) * SymbExpr::from_u32(65536) + var(476) * SymbExpr::from_u32(16777216),
                                // var(477) + var(478) * SymbExpr::from_u32(256) + var(479) * SymbExpr::from_u32(65536) + var(480) * SymbExpr::from_u32(16777216),
                                // var(481) + var(482) * SymbExpr::from_u32(256) + var(483) * SymbExpr::from_u32(65536) + var(484) * SymbExpr::from_u32(16777216),
                                // var(485) + var(486) * SymbExpr::from_u32(256) + var(487) * SymbExpr::from_u32(65536) + var(488) * SymbExpr::from_u32(16777216),

                                // 9
                                // var(489) + var(490) * SymbExpr::from_u32(256) + var(491) * SymbExpr::from_u32(65536) + var(492) * SymbExpr::from_u32(16777216),
                                // var(493) + var(494) * SymbExpr::from_u32(256) + var(495) * SymbExpr::from_u32(65536) + var(496) * SymbExpr::from_u32(16777216),
                                // var(497) + var(498) * SymbExpr::from_u32(256) + var(499) * SymbExpr::from_u32(65536) + var(500) * SymbExpr::from_u32(16777216),
                                // var(501) + var(502) * SymbExpr::from_u32(256) + var(503) * SymbExpr::from_u32(65536) + var(504) * SymbExpr::from_u32(16777216),
                                // var(505) + var(506) * SymbExpr::from_u32(256) + var(507) * SymbExpr::from_u32(65536) + var(508) * SymbExpr::from_u32(16777216),
                                // var(509) + var(510) * SymbExpr::from_u32(256) + var(511) * SymbExpr::from_u32(65536) + var(512) * SymbExpr::from_u32(16777216),
                                // var(513) + var(514) * SymbExpr::from_u32(256) + var(515) * SymbExpr::from_u32(65536) + var(516) * SymbExpr::from_u32(16777216),
                                // var(517) + var(518) * SymbExpr::from_u32(256) + var(519) * SymbExpr::from_u32(65536) + var(520) * SymbExpr::from_u32(16777216),
                                // var(521) + var(522) * SymbExpr::from_u32(256) + var(523) * SymbExpr::from_u32(65536) + var(524) * SymbExpr::from_u32(16777216),
                                // var(525) + var(526) * SymbExpr::from_u32(256) + var(527) * SymbExpr::from_u32(65536) + var(528) * SymbExpr::from_u32(16777216),

                                // 10
                                // var(529) + var(530) * SymbExpr::from_u32(256) + var(531) * SymbExpr::from_u32(65536) + var(532) * SymbExpr::from_u32(16777216),
                                // var(533) + var(534) * SymbExpr::from_u32(256) + var(535) * SymbExpr::from_u32(65536) + var(536) * SymbExpr::from_u32(16777216),
                                // var(537) + var(538) * SymbExpr::from_u32(256) + var(539) * SymbExpr::from_u32(65536) + var(540) * SymbExpr::from_u32(16777216),
                                // var(541) + var(542) * SymbExpr::from_u32(256) + var(543) * SymbExpr::from_u32(65536) + var(544) * SymbExpr::from_u32(16777216),
                                // var(545) + var(546) * SymbExpr::from_u32(256) + var(547) * SymbExpr::from_u32(65536) + var(548) * SymbExpr::from_u32(16777216),
                                // var(549) + var(550) * SymbExpr::from_u32(256) + var(551) * SymbExpr::from_u32(65536) + var(552) * SymbExpr::from_u32(16777216),
                                // var(553) + var(554) * SymbExpr::from_u32(256) + var(555) * SymbExpr::from_u32(65536) + var(556) * SymbExpr::from_u32(16777216),
                                // var(557) + var(558) * SymbExpr::from_u32(256) + var(559) * SymbExpr::from_u32(65536) + var(560) * SymbExpr::from_u32(16777216),
                                // var(561) + var(562) * SymbExpr::from_u32(256) + var(563) * SymbExpr::from_u32(65536) + var(564) * SymbExpr::from_u32(16777216),
                                // var(565) + var(566) * SymbExpr::from_u32(256) + var(567) * SymbExpr::from_u32(65536) + var(568) * SymbExpr::from_u32(16777216),

                                // 11
                                // var(569) + var(570) * SymbExpr::from_u32(256) + var(571) * SymbExpr::from_u32(65536) + var(572) * SymbExpr::from_u32(16777216),
                                // var(573) + var(574) * SymbExpr::from_u32(256) + var(575) * SymbExpr::from_u32(65536) + var(576) * SymbExpr::from_u32(16777216),
                                // var(577) + var(578) * SymbExpr::from_u32(256) + var(579) * SymbExpr::from_u32(65536) + var(580) * SymbExpr::from_u32(16777216),
                                // var(581) + var(582) * SymbExpr::from_u32(256) + var(583) * SymbExpr::from_u32(65536) + var(584) * SymbExpr::from_u32(16777216),
                                // var(585) + var(586) * SymbExpr::from_u32(256) + var(587) * SymbExpr::from_u32(65536) + var(588) * SymbExpr::from_u32(16777216),
                                // var(589) + var(590) * SymbExpr::from_u32(256) + var(591) * SymbExpr::from_u32(65536) + var(592) * SymbExpr::from_u32(16777216),
                                // var(593) + var(594) * SymbExpr::from_u32(256) + var(595) * SymbExpr::from_u32(65536) + var(596) * SymbExpr::from_u32(16777216),
                                // var(597) + var(598) * SymbExpr::from_u32(256) + var(599) * SymbExpr::from_u32(65536) + var(600) * SymbExpr::from_u32(16777216),
                                // var(601) + var(602) * SymbExpr::from_u32(256) + var(603) * SymbExpr::from_u32(65536) + var(604) * SymbExpr::from_u32(16777216),
                                // var(605) + var(606) * SymbExpr::from_u32(256) + var(607) * SymbExpr::from_u32(65536) + var(608) * SymbExpr::from_u32(16777216),

                                // 12
                                // var(609) + var(610) * SymbExpr::from_u32(256) + var(611) * SymbExpr::from_u32(65536) + var(612) * SymbExpr::from_u32(16777216),
                                // var(613) + var(614) * SymbExpr::from_u32(256) + var(615) * SymbExpr::from_u32(65536) + var(616) * SymbExpr::from_u32(16777216),
                                // var(617) + var(618) * SymbExpr::from_u32(256) + var(619) * SymbExpr::from_u32(65536) + var(620) * SymbExpr::from_u32(16777216),
                                // var(621) + var(622) * SymbExpr::from_u32(256) + var(623) * SymbExpr::from_u32(65536) + var(624) * SymbExpr::from_u32(16777216),
                                // var(625) + var(626) * SymbExpr::from_u32(256) + var(627) * SymbExpr::from_u32(65536) + var(628) * SymbExpr::from_u32(16777216),
                                // var(629) + var(630) * SymbExpr::from_u32(256) + var(631) * SymbExpr::from_u32(65536) + var(632) * SymbExpr::from_u32(16777216),
                                // var(633) + var(634) * SymbExpr::from_u32(256) + var(635) * SymbExpr::from_u32(65536) + var(636) * SymbExpr::from_u32(16777216),
                                // var(637) + var(638) * SymbExpr::from_u32(256) + var(639) * SymbExpr::from_u32(65536) + var(640) * SymbExpr::from_u32(16777216),
                                // var(641) + var(642) * SymbExpr::from_u32(256) + var(643) * SymbExpr::from_u32(65536) + var(644) * SymbExpr::from_u32(16777216),
                                // var(645) + var(646) * SymbExpr::from_u32(256) + var(647) * SymbExpr::from_u32(65536) + var(648) * SymbExpr::from_u32(16777216),

                                // 13
                                // var(649) + var(650) * SymbExpr::from_u32(256) + var(651) * SymbExpr::from_u32(65536) + var(652) * SymbExpr::from_u32(16777216),
                                // var(653) + var(654) * SymbExpr::from_u32(256) + var(655) * SymbExpr::from_u32(65536) + var(656) * SymbExpr::from_u32(16777216),
                                // var(657) + var(658) * SymbExpr::from_u32(256) + var(659) * SymbExpr::from_u32(65536) + var(660) * SymbExpr::from_u32(16777216),
                                // var(661) + var(662) * SymbExpr::from_u32(256) + var(663) * SymbExpr::from_u32(65536) + var(664) * SymbExpr::from_u32(16777216),
                                // var(665) + var(666) * SymbExpr::from_u32(256) + var(667) * SymbExpr::from_u32(65536) + var(668) * SymbExpr::from_u32(16777216),
                                // var(669) + var(670) * SymbExpr::from_u32(256) + var(671) * SymbExpr::from_u32(65536) + var(672) * SymbExpr::from_u32(16777216),
                                // var(673) + var(674) * SymbExpr::from_u32(256) + var(675) * SymbExpr::from_u32(65536) + var(676) * SymbExpr::from_u32(16777216),
                                // var(677) + var(678) * SymbExpr::from_u32(256) + var(679) * SymbExpr::from_u32(65536) + var(680) * SymbExpr::from_u32(16777216),
                                // var(681) + var(682) * SymbExpr::from_u32(256) + var(683) * SymbExpr::from_u32(65536) + var(684) * SymbExpr::from_u32(16777216),
                                // var(685) + var(686) * SymbExpr::from_u32(256) + var(687) * SymbExpr::from_u32(65536) + var(688) * SymbExpr::from_u32(16777216),

                                // 14
                                // var(689) + var(690) * SymbExpr::from_u32(256) + var(691) * SymbExpr::from_u32(65536) + var(692) * SymbExpr::from_u32(16777216),
                                // var(693) + var(694) * SymbExpr::from_u32(256) + var(695) * SymbExpr::from_u32(65536) + var(696) * SymbExpr::from_u32(16777216),
                                // var(697) + var(698) * SymbExpr::from_u32(256) + var(699) * SymbExpr::from_u32(65536) + var(700) * SymbExpr::from_u32(16777216),
                                // var(701) + var(702) * SymbExpr::from_u32(256) + var(703) * SymbExpr::from_u32(65536) + var(704) * SymbExpr::from_u32(16777216),
                                // var(705) + var(706) * SymbExpr::from_u32(256) + var(707) * SymbExpr::from_u32(65536) + var(708) * SymbExpr::from_u32(16777216),
                                // var(709) + var(710) * SymbExpr::from_u32(256) + var(711) * SymbExpr::from_u32(65536) + var(712) * SymbExpr::from_u32(16777216),
                                // var(713) + var(714) * SymbExpr::from_u32(256) + var(715) * SymbExpr::from_u32(65536) + var(716) * SymbExpr::from_u32(16777216),
                                // var(717) + var(718) * SymbExpr::from_u32(256) + var(719) * SymbExpr::from_u32(65536) + var(720) * SymbExpr::from_u32(16777216),
                                // var(721) + var(722) * SymbExpr::from_u32(256) + var(723) * SymbExpr::from_u32(65536) + var(724) * SymbExpr::from_u32(16777216),
                                // var(725) + var(726) * SymbExpr::from_u32(256) + var(727) * SymbExpr::from_u32(65536) + var(728) * SymbExpr::from_u32(16777216),

                                // 15
                                // var(729) + var(730) * SymbExpr::from_u32(256) + var(731) * SymbExpr::from_u32(65536) + var(732) * SymbExpr::from_u32(16777216),
                                // var(733) + var(734) * SymbExpr::from_u32(256) + var(735) * SymbExpr::from_u32(65536) + var(736) * SymbExpr::from_u32(16777216),
                                // var(737) + var(738) * SymbExpr::from_u32(256) + var(739) * SymbExpr::from_u32(65536) + var(740) * SymbExpr::from_u32(16777216),
                                // var(741) + var(742) * SymbExpr::from_u32(256) + var(743) * SymbExpr::from_u32(65536) + var(744) * SymbExpr::from_u32(16777216),
                                // var(745) + var(746) * SymbExpr::from_u32(256) + var(747) * SymbExpr::from_u32(65536) + var(748) * SymbExpr::from_u32(16777216),
                                // var(749) + var(750) * SymbExpr::from_u32(256) + var(751) * SymbExpr::from_u32(65536) + var(752) * SymbExpr::from_u32(16777216),
                                // var(753) + var(754) * SymbExpr::from_u32(256) + var(755) * SymbExpr::from_u32(65536) + var(756) * SymbExpr::from_u32(16777216),
                                // var(757) + var(758) * SymbExpr::from_u32(256) + var(759) * SymbExpr::from_u32(65536) + var(760) * SymbExpr::from_u32(16777216),
                                // var(761) + var(762) * SymbExpr::from_u32(256) + var(763) * SymbExpr::from_u32(65536) + var(764) * SymbExpr::from_u32(16777216),
                                // var(765) + var(766) * SymbExpr::from_u32(256) + var(767) * SymbExpr::from_u32(65536) + var(768) * SymbExpr::from_u32(16777216),

                                // 16
                                // var(769) + var(770) * SymbExpr::from_u32(256) + var(771) * SymbExpr::from_u32(65536) + var(772) * SymbExpr::from_u32(16777216),
                                // var(773) + var(774) * SymbExpr::from_u32(256) + var(775) * SymbExpr::from_u32(65536) + var(776) * SymbExpr::from_u32(16777216),
                                // var(777) + var(778) * SymbExpr::from_u32(256) + var(779) * SymbExpr::from_u32(65536) + var(780) * SymbExpr::from_u32(16777216),
                                // var(781) + var(782) * SymbExpr::from_u32(256) + var(783) * SymbExpr::from_u32(65536) + var(784) * SymbExpr::from_u32(16777216),
                                // var(785) + var(786) * SymbExpr::from_u32(256) + var(787) * SymbExpr::from_u32(65536) + var(788) * SymbExpr::from_u32(16777216),
                                // var(789) + var(790) * SymbExpr::from_u32(256) + var(791) * SymbExpr::from_u32(65536) + var(792) * SymbExpr::from_u32(16777216),
                                // var(793) + var(794) * SymbExpr::from_u32(256) + var(795) * SymbExpr::from_u32(65536) + var(796) * SymbExpr::from_u32(16777216),
                                // var(797) + var(798) * SymbExpr::from_u32(256) + var(799) * SymbExpr::from_u32(65536) + var(800) * SymbExpr::from_u32(16777216),
                                // var(801) + var(802) * SymbExpr::from_u32(256) + var(803) * SymbExpr::from_u32(65536) + var(804) * SymbExpr::from_u32(16777216),
                                // var(805) + var(806) * SymbExpr::from_u32(256) + var(807) * SymbExpr::from_u32(65536) + var(808) * SymbExpr::from_u32(16777216),

                                // 17
                                // var(809) + var(810) * SymbExpr::from_u32(256) + var(811) * SymbExpr::from_u32(65536) + var(812) * SymbExpr::from_u32(16777216),
                                // var(813) + var(814) * SymbExpr::from_u32(256) + var(815) * SymbExpr::from_u32(65536) + var(816) * SymbExpr::from_u32(16777216),
                                // var(817) + var(818) * SymbExpr::from_u32(256) + var(819) * SymbExpr::from_u32(65536) + var(820) * SymbExpr::from_u32(16777216),
                                // var(821) + var(822) * SymbExpr::from_u32(256) + var(823) * SymbExpr::from_u32(65536) + var(824) * SymbExpr::from_u32(16777216),
                                // var(825) + var(826) * SymbExpr::from_u32(256) + var(827) * SymbExpr::from_u32(65536) + var(828) * SymbExpr::from_u32(16777216),
                                // var(829) + var(830) * SymbExpr::from_u32(256) + var(831) * SymbExpr::from_u32(65536) + var(832) * SymbExpr::from_u32(16777216),
                                // var(833) + var(834) * SymbExpr::from_u32(256) + var(835) * SymbExpr::from_u32(65536) + var(836) * SymbExpr::from_u32(16777216),
                                // var(837) + var(838) * SymbExpr::from_u32(256) + var(839) * SymbExpr::from_u32(65536) + var(840) * SymbExpr::from_u32(16777216),
                                // var(841) + var(842) * SymbExpr::from_u32(256) + var(843) * SymbExpr::from_u32(65536) + var(844) * SymbExpr::from_u32(16777216),
                                // var(845) + var(846) * SymbExpr::from_u32(256) + var(847) * SymbExpr::from_u32(65536) + var(848) * SymbExpr::from_u32(16777216),

                                // 18
                                // var(849) + var(850) * SymbExpr::from_u32(256) + var(851) * SymbExpr::from_u32(65536) + var(852) * SymbExpr::from_u32(16777216),
                                // var(853) + var(854) * SymbExpr::from_u32(256) + var(855) * SymbExpr::from_u32(65536) + var(856) * SymbExpr::from_u32(16777216),
                                // var(857) + var(858) * SymbExpr::from_u32(256) + var(859) * SymbExpr::from_u32(65536) + var(860) * SymbExpr::from_u32(16777216),
                                // var(861) + var(862) * SymbExpr::from_u32(256) + var(863) * SymbExpr::from_u32(65536) + var(864) * SymbExpr::from_u32(16777216),
                                // var(865) + var(866) * SymbExpr::from_u32(256) + var(867) * SymbExpr::from_u32(65536) + var(868) * SymbExpr::from_u32(16777216),
                                // var(869) + var(870) * SymbExpr::from_u32(256) + var(871) * SymbExpr::from_u32(65536) + var(872) * SymbExpr::from_u32(16777216),
                                // var(873) + var(874) * SymbExpr::from_u32(256) + var(875) * SymbExpr::from_u32(65536) + var(876) * SymbExpr::from_u32(16777216),
                                // var(877) + var(878) * SymbExpr::from_u32(256) + var(879) * SymbExpr::from_u32(65536) + var(880) * SymbExpr::from_u32(16777216),
                                // var(881) + var(882) * SymbExpr::from_u32(256) + var(883) * SymbExpr::from_u32(65536) + var(884) * SymbExpr::from_u32(16777216),
                                // var(885) + var(886) * SymbExpr::from_u32(256) + var(887) * SymbExpr::from_u32(65536) + var(888) * SymbExpr::from_u32(16777216),

                                // 19
                                // var(889) + var(890) * SymbExpr::from_u32(256) + var(891) * SymbExpr::from_u32(65536) + var(892) * SymbExpr::from_u32(16777216),
                                // var(893) + var(894) * SymbExpr::from_u32(256) + var(895) * SymbExpr::from_u32(65536) + var(896) * SymbExpr::from_u32(16777216),
                                // var(897) + var(898) * SymbExpr::from_u32(256) + var(899) * SymbExpr::from_u32(65536) + var(900) * SymbExpr::from_u32(16777216),
                                // var(901) + var(902) * SymbExpr::from_u32(256) + var(903) * SymbExpr::from_u32(65536) + var(904) * SymbExpr::from_u32(16777216),
                                // var(905) + var(906) * SymbExpr::from_u32(256) + var(907) * SymbExpr::from_u32(65536) + var(908) * SymbExpr::from_u32(16777216),
                                // var(909) + var(910) * SymbExpr::from_u32(256) + var(911) * SymbExpr::from_u32(65536) + var(912) * SymbExpr::from_u32(16777216),
                                // var(913) + var(914) * SymbExpr::from_u32(256) + var(915) * SymbExpr::from_u32(65536) + var(916) * SymbExpr::from_u32(16777216),
                                // var(917) + var(918) * SymbExpr::from_u32(256) + var(919) * SymbExpr::from_u32(65536) + var(920) * SymbExpr::from_u32(16777216),
                                // var(921) + var(922) * SymbExpr::from_u32(256) + var(923) * SymbExpr::from_u32(65536) + var(924) * SymbExpr::from_u32(16777216),
                                // var(925) + var(926) * SymbExpr::from_u32(256) + var(927) * SymbExpr::from_u32(65536) + var(928) * SymbExpr::from_u32(16777216),

                                // 20
                                // var(929) + var(930) * SymbExpr::from_u32(256) + var(931) * SymbExpr::from_u32(65536) + var(932) * SymbExpr::from_u32(16777216),
                                // var(933) + var(934) * SymbExpr::from_u32(256) + var(935) * SymbExpr::from_u32(65536) + var(936) * SymbExpr::from_u32(16777216),
                                // var(937) + var(938) * SymbExpr::from_u32(256) + var(939) * SymbExpr::from_u32(65536) + var(940) * SymbExpr::from_u32(16777216),
                                // var(941) + var(942) * SymbExpr::from_u32(256) + var(943) * SymbExpr::from_u32(65536) + var(944) * SymbExpr::from_u32(16777216),
                                // var(945) + var(946) * SymbExpr::from_u32(256) + var(947) * SymbExpr::from_u32(65536) + var(948) * SymbExpr::from_u32(16777216),
                                // var(949) + var(950) * SymbExpr::from_u32(256) + var(951) * SymbExpr::from_u32(65536) + var(952) * SymbExpr::from_u32(16777216),
                                // var(953) + var(954) * SymbExpr::from_u32(256) + var(955) * SymbExpr::from_u32(65536) + var(956) * SymbExpr::from_u32(16777216),
                                // var(957) + var(958) * SymbExpr::from_u32(256) + var(959) * SymbExpr::from_u32(65536) + var(960) * SymbExpr::from_u32(16777216),
                                // var(961) + var(962) * SymbExpr::from_u32(256) + var(963) * SymbExpr::from_u32(65536) + var(964) * SymbExpr::from_u32(16777216),
                                // var(965) + var(966) * SymbExpr::from_u32(256) + var(967) * SymbExpr::from_u32(65536) + var(968) * SymbExpr::from_u32(16777216),

                                // 21
                                // var(969) + var(970) * SymbExpr::from_u32(256) + var(971) * SymbExpr::from_u32(65536) + var(972) * SymbExpr::from_u32(16777216),
                                // var(973) + var(974) * SymbExpr::from_u32(256) + var(975) * SymbExpr::from_u32(65536) + var(976) * SymbExpr::from_u32(16777216),
                                // var(977) + var(978) * SymbExpr::from_u32(256) + var(979) * SymbExpr::from_u32(65536) + var(980) * SymbExpr::from_u32(16777216),
                                // var(981) + var(982) * SymbExpr::from_u32(256) + var(983) * SymbExpr::from_u32(65536) + var(984) * SymbExpr::from_u32(16777216),
                                // var(985) + var(986) * SymbExpr::from_u32(256) + var(987) * SymbExpr::from_u32(65536) + var(988) * SymbExpr::from_u32(16777216),
                                // var(989) + var(990) * SymbExpr::from_u32(256) + var(991) * SymbExpr::from_u32(65536) + var(992) * SymbExpr::from_u32(16777216),
                                // var(993) + var(994) * SymbExpr::from_u32(256) + var(995) * SymbExpr::from_u32(65536) + var(996) * SymbExpr::from_u32(16777216),
                                // var(997) + var(998) * SymbExpr::from_u32(256) + var(999) * SymbExpr::from_u32(65536) + var(1000) * SymbExpr::from_u32(16777216),
                                // var(1001) + var(1002) * SymbExpr::from_u32(256) + var(1003) * SymbExpr::from_u32(65536) + var(1004) * SymbExpr::from_u32(16777216),
                                // var(1005) + var(1006) * SymbExpr::from_u32(256) + var(1007) * SymbExpr::from_u32(65536) + var(1008) * SymbExpr::from_u32(16777216),

                                // 22
                                // var(1009) + var(1010) * SymbExpr::from_u32(256) + var(1011) * SymbExpr::from_u32(65536) + var(1012) * SymbExpr::from_u32(16777216),
                                // var(1013) + var(1014) * SymbExpr::from_u32(256) + var(1015) * SymbExpr::from_u32(65536) + var(1016) * SymbExpr::from_u32(16777216),
                                // var(1017) + var(1018) * SymbExpr::from_u32(256) + var(1019) * SymbExpr::from_u32(65536) + var(1020) * SymbExpr::from_u32(16777216),
                                // var(1021) + var(1022) * SymbExpr::from_u32(256) + var(1023) * SymbExpr::from_u32(65536) + var(1024) * SymbExpr::from_u32(16777216),
                                // var(1025) + var(1026) * SymbExpr::from_u32(256) + var(1027) * SymbExpr::from_u32(65536) + var(1028) * SymbExpr::from_u32(16777216),
                                // var(1029) + var(1030) * SymbExpr::from_u32(256) + var(1031) * SymbExpr::from_u32(65536) + var(1032) * SymbExpr::from_u32(16777216),
                                // var(1033) + var(1034) * SymbExpr::from_u32(256) + var(1035) * SymbExpr::from_u32(65536) + var(1036) * SymbExpr::from_u32(16777216),
                                // var(1037) + var(1038) * SymbExpr::from_u32(256) + var(1039) * SymbExpr::from_u32(65536) + var(1040) * SymbExpr::from_u32(16777216),
                                // var(1041) + var(1042) * SymbExpr::from_u32(256) + var(1043) * SymbExpr::from_u32(65536) + var(1044) * SymbExpr::from_u32(16777216),
                                // var(1045) + var(1046) * SymbExpr::from_u32(256) + var(1047) * SymbExpr::from_u32(65536) + var(1048) * SymbExpr::from_u32(16777216),

                                // 23
                                // var(1049) + var(1050) * SymbExpr::from_u32(256) + var(1051) * SymbExpr::from_u32(65536) + var(1052) * SymbExpr::from_u32(16777216),
                                // var(1053) + var(1054) * SymbExpr::from_u32(256) + var(1055) * SymbExpr::from_u32(65536) + var(1056) * SymbExpr::from_u32(16777216),
                                // var(1057) + var(1058) * SymbExpr::from_u32(256) + var(1059) * SymbExpr::from_u32(65536) + var(1060) * SymbExpr::from_u32(16777216),
                                // var(1061) + var(1062) * SymbExpr::from_u32(256) + var(1063) * SymbExpr::from_u32(65536) + var(1064) * SymbExpr::from_u32(16777216),
                                // var(1065) + var(1066) * SymbExpr::from_u32(256) + var(1067) * SymbExpr::from_u32(65536) + var(1068) * SymbExpr::from_u32(16777216),
                                // var(1069) + var(1070) * SymbExpr::from_u32(256) + var(1071) * SymbExpr::from_u32(65536) + var(1072) * SymbExpr::from_u32(16777216),
                                // var(1073) + var(1074) * SymbExpr::from_u32(256) + var(1075) * SymbExpr::from_u32(65536) + var(1076) * SymbExpr::from_u32(16777216),
                                // var(1077) + var(1078) * SymbExpr::from_u32(256) + var(1079) * SymbExpr::from_u32(65536) + var(1080) * SymbExpr::from_u32(16777216),
                                // var(1081) + var(1082) * SymbExpr::from_u32(256) + var(1083) * SymbExpr::from_u32(65536) + var(1084) * SymbExpr::from_u32(16777216),
                                // var(1085) + var(1086) * SymbExpr::from_u32(256) + var(1087) * SymbExpr::from_u32(65536) + var(1088) * SymbExpr::from_u32(16777216),

                                // 24
                                // var(1089) + var(1090) * SymbExpr::from_u32(256) + var(1091) * SymbExpr::from_u32(65536) + var(1092) * SymbExpr::from_u32(16777216),
                                // var(1093) + var(1094) * SymbExpr::from_u32(256) + var(1095) * SymbExpr::from_u32(65536) + var(1096) * SymbExpr::from_u32(16777216),
                                // var(1097) + var(1098) * SymbExpr::from_u32(256) + var(1099) * SymbExpr::from_u32(65536) + var(1100) * SymbExpr::from_u32(16777216),
                                // var(1101) + var(1102) * SymbExpr::from_u32(256) + var(1103) * SymbExpr::from_u32(65536) + var(1104) * SymbExpr::from_u32(16777216),
                                // var(1105) + var(1106) * SymbExpr::from_u32(256) + var(1107) * SymbExpr::from_u32(65536) + var(1108) * SymbExpr::from_u32(16777216),
                                // var(1109) + var(1110) * SymbExpr::from_u32(256) + var(1111) * SymbExpr::from_u32(65536) + var(1112) * SymbExpr::from_u32(16777216),
                                // var(1113) + var(1114) * SymbExpr::from_u32(256) + var(1115) * SymbExpr::from_u32(65536) + var(1116) * SymbExpr::from_u32(16777216),
                                // var(1117) + var(1118) * SymbExpr::from_u32(256) + var(1119) * SymbExpr::from_u32(65536) + var(1120) * SymbExpr::from_u32(16777216),
                                // var(1121) + var(1122) * SymbExpr::from_u32(256) + var(1123) * SymbExpr::from_u32(65536) + var(1124) * SymbExpr::from_u32(16777216),
                                // var(1125) + var(1126) * SymbExpr::from_u32(256) + var(1127) * SymbExpr::from_u32(65536) + var(1128) * SymbExpr::from_u32(16777216),

                                // 25
                                // var(1129) + var(1130) * SymbExpr::from_u32(256) + var(1131) * SymbExpr::from_u32(65536) + var(1132) * SymbExpr::from_u32(16777216),
                                // var(1133) + var(1134) * SymbExpr::from_u32(256) + var(1135) * SymbExpr::from_u32(65536) + var(1136) * SymbExpr::from_u32(16777216),
                                // var(1137) + var(1138) * SymbExpr::from_u32(256) + var(1139) * SymbExpr::from_u32(65536) + var(1140) * SymbExpr::from_u32(16777216),
                                // var(1141) + var(1142) * SymbExpr::from_u32(256) + var(1143) * SymbExpr::from_u32(65536) + var(1144) * SymbExpr::from_u32(16777216),
                                // var(1145) + var(1146) * SymbExpr::from_u32(256) + var(1147) * SymbExpr::from_u32(65536) + var(1148) * SymbExpr::from_u32(16777216),
                                // var(1149) + var(1150) * SymbExpr::from_u32(256) + var(1151) * SymbExpr::from_u32(65536) + var(1152) * SymbExpr::from_u32(16777216),
                                // var(1153) + var(1154) * SymbExpr::from_u32(256) + var(1155) * SymbExpr::from_u32(65536) + var(1156) * SymbExpr::from_u32(16777216),
                                // var(1157) + var(1158) * SymbExpr::from_u32(256) + var(1159) * SymbExpr::from_u32(65536) + var(1160) * SymbExpr::from_u32(16777216),
                                // var(1161) + var(1162) * SymbExpr::from_u32(256) + var(1163) * SymbExpr::from_u32(65536) + var(1164) * SymbExpr::from_u32(16777216),
                                // var(1165) + var(1166) * SymbExpr::from_u32(256) + var(1167) * SymbExpr::from_u32(65536) + var(1168) * SymbExpr::from_u32(16777216),

                                // 26
                                // var(1169) + var(1170) * SymbExpr::from_u32(256) + var(1171) * SymbExpr::from_u32(65536) + var(1172) * SymbExpr::from_u32(16777216),
                                // var(1173) + var(1174) * SymbExpr::from_u32(256) + var(1175) * SymbExpr::from_u32(65536) + var(1176) * SymbExpr::from_u32(16777216),
                                // var(1177) + var(1178) * SymbExpr::from_u32(256) + var(1179) * SymbExpr::from_u32(65536) + var(1180) * SymbExpr::from_u32(16777216),
                                // var(1181) + var(1182) * SymbExpr::from_u32(256) + var(1183) * SymbExpr::from_u32(65536) + var(1184) * SymbExpr::from_u32(16777216),
                                // var(1185) + var(1186) * SymbExpr::from_u32(256) + var(1187) * SymbExpr::from_u32(65536) + var(1188) * SymbExpr::from_u32(16777216),
                                // var(1189) + var(1190) * SymbExpr::from_u32(256) + var(1191) * SymbExpr::from_u32(65536) + var(1192) * SymbExpr::from_u32(16777216),
                                // var(1193) + var(1194) * SymbExpr::from_u32(256) + var(1195) * SymbExpr::from_u32(65536) + var(1196) * SymbExpr::from_u32(16777216),
                                // var(1197) + var(1198) * SymbExpr::from_u32(256) + var(1199) * SymbExpr::from_u32(65536) + var(1200) * SymbExpr::from_u32(16777216),
                                // var(1201) + var(1202) * SymbExpr::from_u32(256) + var(1203) * SymbExpr::from_u32(65536) + var(1204) * SymbExpr::from_u32(16777216),
                                // var(1205) + var(1206) * SymbExpr::from_u32(256) + var(1207) * SymbExpr::from_u32(65536) + var(1208) * SymbExpr::from_u32(16777216),

                                // 27
                                // var(1209) + var(1210) * SymbExpr::from_u32(256) + var(1211) * SymbExpr::from_u32(65536) + var(1212) * SymbExpr::from_u32(16777216),
                                // var(1213) + var(1214) * SymbExpr::from_u32(256) + var(1215) * SymbExpr::from_u32(65536) + var(1216) * SymbExpr::from_u32(16777216),
                                // var(1217) + var(1218) * SymbExpr::from_u32(256) + var(1219) * SymbExpr::from_u32(65536) + var(1220) * SymbExpr::from_u32(16777216),
                                // var(1221) + var(1222) * SymbExpr::from_u32(256) + var(1223) * SymbExpr::from_u32(65536) + var(1224) * SymbExpr::from_u32(16777216),
                                // var(1225) + var(1226) * SymbExpr::from_u32(256) + var(1227) * SymbExpr::from_u32(65536) + var(1228) * SymbExpr::from_u32(16777216),
                                // var(1229) + var(1230) * SymbExpr::from_u32(256) + var(1231) * SymbExpr::from_u32(65536) + var(1232) * SymbExpr::from_u32(16777216),
                                // var(1233) + var(1234) * SymbExpr::from_u32(256) + var(1235) * SymbExpr::from_u32(65536) + var(1236) * SymbExpr::from_u32(16777216),
                                // var(1237) + var(1238) * SymbExpr::from_u32(256) + var(1239) * SymbExpr::from_u32(65536) + var(1240) * SymbExpr::from_u32(16777216),
                                // var(1241) + var(1242) * SymbExpr::from_u32(256) + var(1243) * SymbExpr::from_u32(65536) + var(1244) * SymbExpr::from_u32(16777216),
                                // var(1245) + var(1246) * SymbExpr::from_u32(256) + var(1247) * SymbExpr::from_u32(65536) + var(1248) * SymbExpr::from_u32(16777216),

                                // 28
                                // var(1249) + var(1250) * SymbExpr::from_u32(256) + var(1251) * SymbExpr::from_u32(65536) + var(1252) * SymbExpr::from_u32(16777216),
                                // var(1253) + var(1254) * SymbExpr::from_u32(256) + var(1255) * SymbExpr::from_u32(65536) + var(1256) * SymbExpr::from_u32(16777216),
                                // var(1257) + var(1258) * SymbExpr::from_u32(256) + var(1259) * SymbExpr::from_u32(65536) + var(1260) * SymbExpr::from_u32(16777216),
                                // var(1261) + var(1262) * SymbExpr::from_u32(256) + var(1263) * SymbExpr::from_u32(65536) + var(1264) * SymbExpr::from_u32(16777216),
                                // var(1265) + var(1266) * SymbExpr::from_u32(256) + var(1267) * SymbExpr::from_u32(65536) + var(1268) * SymbExpr::from_u32(16777216),
                                // var(1269) + var(1270) * SymbExpr::from_u32(256) + var(1271) * SymbExpr::from_u32(65536) + var(1272) * SymbExpr::from_u32(16777216),
                                // var(1273) + var(1274) * SymbExpr::from_u32(256) + var(1275) * SymbExpr::from_u32(65536) + var(1276) * SymbExpr::from_u32(16777216),
                                // var(1277) + var(1278) * SymbExpr::from_u32(256) + var(1279) * SymbExpr::from_u32(65536) + var(1280) * SymbExpr::from_u32(16777216),
                                // var(1281) + var(1282) * SymbExpr::from_u32(256) + var(1283) * SymbExpr::from_u32(65536) + var(1284) * SymbExpr::from_u32(16777216),
                                // var(1285) + var(1286) * SymbExpr::from_u32(256) + var(1287) * SymbExpr::from_u32(65536) + var(1288) * SymbExpr::from_u32(16777216),

                                // 29
                                // var(1289) + var(1290) * SymbExpr::from_u32(256) + var(1291) * SymbExpr::from_u32(65536) + var(1292) * SymbExpr::from_u32(16777216),
                                // var(1293) + var(1294) * SymbExpr::from_u32(256) + var(1295) * SymbExpr::from_u32(65536) + var(1296) * SymbExpr::from_u32(16777216),
                                // var(1297) + var(1298) * SymbExpr::from_u32(256) + var(1299) * SymbExpr::from_u32(65536) + var(1300) * SymbExpr::from_u32(16777216),
                                // var(1301) + var(1302) * SymbExpr::from_u32(256) + var(1303) * SymbExpr::from_u32(65536) + var(1304) * SymbExpr::from_u32(16777216),
                                // var(1305) + var(1306) * SymbExpr::from_u32(256) + var(1307) * SymbExpr::from_u32(65536) + var(1308) * SymbExpr::from_u32(16777216),
                                // var(1309) + var(1310) * SymbExpr::from_u32(256) + var(1311) * SymbExpr::from_u32(65536) + var(1312) * SymbExpr::from_u32(16777216),
                                // var(1313) + var(1314) * SymbExpr::from_u32(256) + var(1315) * SymbExpr::from_u32(65536) + var(1316) * SymbExpr::from_u32(16777216),
                                // var(1317) + var(1318) * SymbExpr::from_u32(256) + var(1319) * SymbExpr::from_u32(65536) + var(1320) * SymbExpr::from_u32(16777216),
                                // var(1321) + var(1322) * SymbExpr::from_u32(256) + var(1323) * SymbExpr::from_u32(65536) + var(1324) * SymbExpr::from_u32(16777216),
                                // var(1325) + var(1326) * SymbExpr::from_u32(256) + var(1327) * SymbExpr::from_u32(65536) + var(1328) * SymbExpr::from_u32(16777216),

                                // 30
                                // var(1329) + var(1330) * SymbExpr::from_u32(256) + var(1331) * SymbExpr::from_u32(65536) + var(1332) * SymbExpr::from_u32(16777216),
                                // var(1333) + var(1334) * SymbExpr::from_u32(256) + var(1335) * SymbExpr::from_u32(65536) + var(1336) * SymbExpr::from_u32(16777216),
                                // var(1337) + var(1338) * SymbExpr::from_u32(256) + var(1339) * SymbExpr::from_u32(65536) + var(1340) * SymbExpr::from_u32(16777216),
                                // var(1341) + var(1342) * SymbExpr::from_u32(256) + var(1343) * SymbExpr::from_u32(65536) + var(1344) * SymbExpr::from_u32(16777216),
                                // var(1345) + var(1346) * SymbExpr::from_u32(256) + var(1347) * SymbExpr::from_u32(65536) + var(1348) * SymbExpr::from_u32(16777216),
                                // var(1349) + var(1350) * SymbExpr::from_u32(256) + var(1351) * SymbExpr::from_u32(65536) + var(1352) * SymbExpr::from_u32(16777216),
                                // var(1353) + var(1354) * SymbExpr::from_u32(256) + var(1355) * SymbExpr::from_u32(65536) + var(1356) * SymbExpr::from_u32(16777216),
                                // var(1357) + var(1358) * SymbExpr::from_u32(256) + var(1359) * SymbExpr::from_u32(65536) + var(1360) * SymbExpr::from_u32(16777216),
                                // var(1361) + var(1362) * SymbExpr::from_u32(256) + var(1363) * SymbExpr::from_u32(65536) + var(1364) * SymbExpr::from_u32(16777216),
                                // var(1365) + var(1366) * SymbExpr::from_u32(256) + var(1367) * SymbExpr::from_u32(65536) + var(1368) * SymbExpr::from_u32(16777216),

                                // 31
                                // var(1369) + var(1370) * SymbExpr::from_u32(256) + var(1371) * SymbExpr::from_u32(65536) + var(1372) * SymbExpr::from_u32(16777216),
                                // var(1373) + var(1374) * SymbExpr::from_u32(256) + var(1375) * SymbExpr::from_u32(65536) + var(1376) * SymbExpr::from_u32(16777216),
                                // var(1377) + var(1378) * SymbExpr::from_u32(256) + var(1379) * SymbExpr::from_u32(65536) + var(1380) * SymbExpr::from_u32(16777216),
                                // var(1381) + var(1382) * SymbExpr::from_u32(256) + var(1383) * SymbExpr::from_u32(65536) + var(1384) * SymbExpr::from_u32(16777216),
                                // var(1385) + var(1386) * SymbExpr::from_u32(256) + var(1387) * SymbExpr::from_u32(65536) + var(1388) * SymbExpr::from_u32(16777216),
                                // var(1389) + var(1390) * SymbExpr::from_u32(256) + var(1391) * SymbExpr::from_u32(65536) + var(1392) * SymbExpr::from_u32(16777216),
                                // var(1393) + var(1394) * SymbExpr::from_u32(256) + var(1395) * SymbExpr::from_u32(65536) + var(1396) * SymbExpr::from_u32(16777216),
                                // var(1397) + var(1398) * SymbExpr::from_u32(256) + var(1399) * SymbExpr::from_u32(65536) + var(1400) * SymbExpr::from_u32(16777216),
                                // var(1401) + var(1402) * SymbExpr::from_u32(256) + var(1403) * SymbExpr::from_u32(65536) + var(1404) * SymbExpr::from_u32(16777216),
                                // var(1405) + var(1406) * SymbExpr::from_u32(256) + var(1407) * SymbExpr::from_u32(65536) + var(1408) * SymbExpr::from_u32(16777216),

                                // 32
                                // var(1409) + var(1410) * SymbExpr::from_u32(256) + var(1411) * SymbExpr::from_u32(65536) + var(1412) * SymbExpr::from_u32(16777216),
                                // var(1413) + var(1414) * SymbExpr::from_u32(256) + var(1415) * SymbExpr::from_u32(65536) + var(1416) * SymbExpr::from_u32(16777216),
                                // var(1417) + var(1418) * SymbExpr::from_u32(256) + var(1419) * SymbExpr::from_u32(65536) + var(1420) * SymbExpr::from_u32(16777216),
                                // var(1421) + var(1422) * SymbExpr::from_u32(256) + var(1423) * SymbExpr::from_u32(65536) + var(1424) * SymbExpr::from_u32(16777216),
                                // var(1425) + var(1426) * SymbExpr::from_u32(256) + var(1427) * SymbExpr::from_u32(65536) + var(1428) * SymbExpr::from_u32(16777216),
                                // var(1429) + var(1430) * SymbExpr::from_u32(256) + var(1431) * SymbExpr::from_u32(65536) + var(1432) * SymbExpr::from_u32(16777216),
                                // var(1433) + var(1434) * SymbExpr::from_u32(256) + var(1435) * SymbExpr::from_u32(65536) + var(1436) * SymbExpr::from_u32(16777216),
                                // var(1437) + var(1438) * SymbExpr::from_u32(256) + var(1439) * SymbExpr::from_u32(65536) + var(1440) * SymbExpr::from_u32(16777216),
                                // var(1441) + var(1442) * SymbExpr::from_u32(256) + var(1443) * SymbExpr::from_u32(65536) + var(1444) * SymbExpr::from_u32(16777216),
                                // var(1445) + var(1446) * SymbExpr::from_u32(256) + var(1447) * SymbExpr::from_u32(65536) + var(1448) * SymbExpr::from_u32(16777216),

                                // 33
                                // var(1449) + var(1450) * SymbExpr::from_u32(256) + var(1451) * SymbExpr::from_u32(65536) + var(1452) * SymbExpr::from_u32(16777216),
                                // var(1453) + var(1454) * SymbExpr::from_u32(256) + var(1455) * SymbExpr::from_u32(65536) + var(1456) * SymbExpr::from_u32(16777216),
                                // var(1457) + var(1458) * SymbExpr::from_u32(256) + var(1459) * SymbExpr::from_u32(65536) + var(1460) * SymbExpr::from_u32(16777216),
                                // var(1461) + var(1462) * SymbExpr::from_u32(256) + var(1463) * SymbExpr::from_u32(65536) + var(1464) * SymbExpr::from_u32(16777216),
                                // var(1465) + var(1466) * SymbExpr::from_u32(256) + var(1467) * SymbExpr::from_u32(65536) + var(1468) * SymbExpr::from_u32(16777216),
                                // var(1469) + var(1470) * SymbExpr::from_u32(256) + var(1471) * SymbExpr::from_u32(65536) + var(1472) * SymbExpr::from_u32(16777216),
                                // var(1473) + var(1474) * SymbExpr::from_u32(256) + var(1475) * SymbExpr::from_u32(65536) + var(1476) * SymbExpr::from_u32(16777216),
                                // var(1477) + var(1478) * SymbExpr::from_u32(256) + var(1479) * SymbExpr::from_u32(65536) + var(1480) * SymbExpr::from_u32(16777216),
                                // var(1481) + var(1482) * SymbExpr::from_u32(256) + var(1483) * SymbExpr::from_u32(65536) + var(1484) * SymbExpr::from_u32(16777216),
                                // var(1485) + var(1486) * SymbExpr::from_u32(256) + var(1487) * SymbExpr::from_u32(65536) + var(1488) * SymbExpr::from_u32(16777216),

                                // 34
                                // var(1489) + var(1490) * SymbExpr::from_u32(256) + var(1491) * SymbExpr::from_u32(65536) + var(1492) * SymbExpr::from_u32(16777216),
                                // var(1493) + var(1494) * SymbExpr::from_u32(256) + var(1495) * SymbExpr::from_u32(65536) + var(1496) * SymbExpr::from_u32(16777216),
                                // var(1497) + var(1498) * SymbExpr::from_u32(256) + var(1499) * SymbExpr::from_u32(65536) + var(1500) * SymbExpr::from_u32(16777216),
                                // var(1501) + var(1502) * SymbExpr::from_u32(256) + var(1503) * SymbExpr::from_u32(65536) + var(1504) * SymbExpr::from_u32(16777216),
                                // var(1505) + var(1506) * SymbExpr::from_u32(256) + var(1507) * SymbExpr::from_u32(65536) + var(1508) * SymbExpr::from_u32(16777216),
                                // var(1509) + var(1510) * SymbExpr::from_u32(256) + var(1511) * SymbExpr::from_u32(65536) + var(1512) * SymbExpr::from_u32(16777216),
                                // var(1513) + var(1514) * SymbExpr::from_u32(256) + var(1515) * SymbExpr::from_u32(65536) + var(1516) * SymbExpr::from_u32(16777216),
                                // var(1517) + var(1518) * SymbExpr::from_u32(256) + var(1519) * SymbExpr::from_u32(65536) + var(1520) * SymbExpr::from_u32(16777216),
                                // var(1521) + var(1522) * SymbExpr::from_u32(256) + var(1523) * SymbExpr::from_u32(65536) + var(1524) * SymbExpr::from_u32(16777216),
                                // var(1525) + var(1526) * SymbExpr::from_u32(256) + var(1527) * SymbExpr::from_u32(65536) + var(1528) * SymbExpr::from_u32(16777216),

                                // 35
                                // var(1529) + var(1530) * SymbExpr::from_u32(256) + var(1531) * SymbExpr::from_u32(65536) + var(1532) * SymbExpr::from_u32(16777216),
                                // var(1533) + var(1534) * SymbExpr::from_u32(256) + var(1535) * SymbExpr::from_u32(65536) + var(1536) * SymbExpr::from_u32(16777216),
                                // var(1537) + var(1538) * SymbExpr::from_u32(256) + var(1539) * SymbExpr::from_u32(65536) + var(1540) * SymbExpr::from_u32(16777216),
                                // var(1541) + var(1542) * SymbExpr::from_u32(256) + var(1543) * SymbExpr::from_u32(65536) + var(1544) * SymbExpr::from_u32(16777216),
                                // var(1545) + var(1546) * SymbExpr::from_u32(256) + var(1547) * SymbExpr::from_u32(65536) + var(1548) * SymbExpr::from_u32(16777216),
                                // var(1549) + var(1550) * SymbExpr::from_u32(256) + var(1551) * SymbExpr::from_u32(65536) + var(1552) * SymbExpr::from_u32(16777216),
                                // var(1553) + var(1554) * SymbExpr::from_u32(256) + var(1555) * SymbExpr::from_u32(65536) + var(1556) * SymbExpr::from_u32(16777216),
                                // var(1557) + var(1558) * SymbExpr::from_u32(256) + var(1559) * SymbExpr::from_u32(65536) + var(1560) * SymbExpr::from_u32(16777216),
                                // var(1561) + var(1562) * SymbExpr::from_u32(256) + var(1563) * SymbExpr::from_u32(65536) + var(1564) * SymbExpr::from_u32(16777216),
                                // var(1565) + var(1566) * SymbExpr::from_u32(256) + var(1567) * SymbExpr::from_u32(65536) + var(1568) * SymbExpr::from_u32(16777216),

                                // 36
                                // var(1569) + var(1570) * SymbExpr::from_u32(256) + var(1571) * SymbExpr::from_u32(65536) + var(1572) * SymbExpr::from_u32(16777216),
                                // var(1573) + var(1574) * SymbExpr::from_u32(256) + var(1575) * SymbExpr::from_u32(65536) + var(1576) * SymbExpr::from_u32(16777216),
                                // var(1577) + var(1578) * SymbExpr::from_u32(256) + var(1579) * SymbExpr::from_u32(65536) + var(1580) * SymbExpr::from_u32(16777216),
                                // var(1581) + var(1582) * SymbExpr::from_u32(256) + var(1583) * SymbExpr::from_u32(65536) + var(1584) * SymbExpr::from_u32(16777216),
                                // var(1585) + var(1586) * SymbExpr::from_u32(256) + var(1587) * SymbExpr::from_u32(65536) + var(1588) * SymbExpr::from_u32(16777216),
                                // var(1589) + var(1590) * SymbExpr::from_u32(256) + var(1591) * SymbExpr::from_u32(65536) + var(1592) * SymbExpr::from_u32(16777216),
                                // var(1593) + var(1594) * SymbExpr::from_u32(256) + var(1595) * SymbExpr::from_u32(65536) + var(1596) * SymbExpr::from_u32(16777216),
                                // var(1597) + var(1598) * SymbExpr::from_u32(256) + var(1599) * SymbExpr::from_u32(65536) + var(1600) * SymbExpr::from_u32(16777216),
                                // var(1601) + var(1602) * SymbExpr::from_u32(256) + var(1603) * SymbExpr::from_u32(65536) + var(1604) * SymbExpr::from_u32(16777216),
                                // var(1605) + var(1606) * SymbExpr::from_u32(256) + var(1607) * SymbExpr::from_u32(65536) + var(1608) * SymbExpr::from_u32(16777216),

                                // 37
                                // var(1609) + var(1610) * SymbExpr::from_u32(256) + var(1611) * SymbExpr::from_u32(65536) + var(1612) * SymbExpr::from_u32(16777216),
                                // var(1613) + var(1614) * SymbExpr::from_u32(256) + var(1615) * SymbExpr::from_u32(65536) + var(1616) * SymbExpr::from_u32(16777216),
                                // var(1617) + var(1618) * SymbExpr::from_u32(256) + var(1619) * SymbExpr::from_u32(65536) + var(1620) * SymbExpr::from_u32(16777216),
                                // var(1621) + var(1622) * SymbExpr::from_u32(256) + var(1623) * SymbExpr::from_u32(65536) + var(1624) * SymbExpr::from_u32(16777216),
                                // var(1625) + var(1626) * SymbExpr::from_u32(256) + var(1627) * SymbExpr::from_u32(65536) + var(1628) * SymbExpr::from_u32(16777216),
                                // var(1629) + var(1630) * SymbExpr::from_u32(256) + var(1631) * SymbExpr::from_u32(65536) + var(1632) * SymbExpr::from_u32(16777216),
                                // var(1633) + var(1634) * SymbExpr::from_u32(256) + var(1635) * SymbExpr::from_u32(65536) + var(1636) * SymbExpr::from_u32(16777216),
                                // var(1637) + var(1638) * SymbExpr::from_u32(256) + var(1639) * SymbExpr::from_u32(65536) + var(1640) * SymbExpr::from_u32(16777216),
                                // var(1641) + var(1642) * SymbExpr::from_u32(256) + var(1643) * SymbExpr::from_u32(65536) + var(1644) * SymbExpr::from_u32(16777216),
                                // var(1645) + var(1646) * SymbExpr::from_u32(256) + var(1647) * SymbExpr::from_u32(65536) + var(1648) * SymbExpr::from_u32(16777216),

                                // 38
                                // var(1649) + var(1650) * SymbExpr::from_u32(256) + var(1651) * SymbExpr::from_u32(65536) + var(1652) * SymbExpr::from_u32(16777216),
                                // var(1653) + var(1654) * SymbExpr::from_u32(256) + var(1655) * SymbExpr::from_u32(65536) + var(1656) * SymbExpr::from_u32(16777216),
                                // var(1657) + var(1658) * SymbExpr::from_u32(256) + var(1659) * SymbExpr::from_u32(65536) + var(1660) * SymbExpr::from_u32(16777216),
                                // var(1661) + var(1662) * SymbExpr::from_u32(256) + var(1663) * SymbExpr::from_u32(65536) + var(1664) * SymbExpr::from_u32(16777216),
                                // var(1665) + var(1666) * SymbExpr::from_u32(256) + var(1667) * SymbExpr::from_u32(65536) + var(1668) * SymbExpr::from_u32(16777216),
                                // var(1669) + var(1670) * SymbExpr::from_u32(256) + var(1671) * SymbExpr::from_u32(65536) + var(1672) * SymbExpr::from_u32(16777216),
                                // var(1673) + var(1674) * SymbExpr::from_u32(256) + var(1675) * SymbExpr::from_u32(65536) + var(1676) * SymbExpr::from_u32(16777216),
                                // var(1677) + var(1678) * SymbExpr::from_u32(256) + var(1679) * SymbExpr::from_u32(65536) + var(1680) * SymbExpr::from_u32(16777216),
                                // var(1681) + var(1682) * SymbExpr::from_u32(256) + var(1683) * SymbExpr::from_u32(65536) + var(1684) * SymbExpr::from_u32(16777216),
                                // var(1685) + var(1686) * SymbExpr::from_u32(256) + var(1687) * SymbExpr::from_u32(65536) + var(1688) * SymbExpr::from_u32(16777216),

                                // 39
                                // var(1689) + var(1690) * SymbExpr::from_u32(256) + var(1691) * SymbExpr::from_u32(65536) + var(1692) * SymbExpr::from_u32(16777216),
                                // var(1693) + var(1694) * SymbExpr::from_u32(256) + var(1695) * SymbExpr::from_u32(65536) + var(1696) * SymbExpr::from_u32(16777216),
                                // var(1697) + var(1698) * SymbExpr::from_u32(256) + var(1699) * SymbExpr::from_u32(65536) + var(1700) * SymbExpr::from_u32(16777216),
                                // var(1701) + var(1702) * SymbExpr::from_u32(256) + var(1703) * SymbExpr::from_u32(65536) + var(1704) * SymbExpr::from_u32(16777216),
                                // var(1705) + var(1706) * SymbExpr::from_u32(256) + var(1707) * SymbExpr::from_u32(65536) + var(1708) * SymbExpr::from_u32(16777216),
                                // var(1709) + var(1710) * SymbExpr::from_u32(256) + var(1711) * SymbExpr::from_u32(65536) + var(1712) * SymbExpr::from_u32(16777216),
                                // var(1713) + var(1714) * SymbExpr::from_u32(256) + var(1715) * SymbExpr::from_u32(65536) + var(1716) * SymbExpr::from_u32(16777216),
                                // var(1717) + var(1718) * SymbExpr::from_u32(256) + var(1719) * SymbExpr::from_u32(65536) + var(1720) * SymbExpr::from_u32(16777216),
                                // var(1721) + var(1722) * SymbExpr::from_u32(256) + var(1723) * SymbExpr::from_u32(65536) + var(1724) * SymbExpr::from_u32(16777216),
                                // var(1725) + var(1726) * SymbExpr::from_u32(256) + var(1727) * SymbExpr::from_u32(65536) + var(1728) * SymbExpr::from_u32(16777216),

                                // 40
                                // var(1729) + var(1730) * SymbExpr::from_u32(256) + var(1731) * SymbExpr::from_u32(65536) + var(1732) * SymbExpr::from_u32(16777216),
                                // var(1733) + var(1734) * SymbExpr::from_u32(256) + var(1735) * SymbExpr::from_u32(65536) + var(1736) * SymbExpr::from_u32(16777216),
                                // var(1737) + var(1738) * SymbExpr::from_u32(256) + var(1739) * SymbExpr::from_u32(65536) + var(1740) * SymbExpr::from_u32(16777216),
                                // var(1741) + var(1742) * SymbExpr::from_u32(256) + var(1743) * SymbExpr::from_u32(65536) + var(1744) * SymbExpr::from_u32(16777216),
                                // var(1745) + var(1746) * SymbExpr::from_u32(256) + var(1747) * SymbExpr::from_u32(65536) + var(1748) * SymbExpr::from_u32(16777216),
                                // var(1749) + var(1750) * SymbExpr::from_u32(256) + var(1751) * SymbExpr::from_u32(65536) + var(1752) * SymbExpr::from_u32(16777216),
                                // var(1753) + var(1754) * SymbExpr::from_u32(256) + var(1755) * SymbExpr::from_u32(65536) + var(1756) * SymbExpr::from_u32(16777216),
                                // var(1757) + var(1758) * SymbExpr::from_u32(256) + var(1759) * SymbExpr::from_u32(65536) + var(1760) * SymbExpr::from_u32(16777216),
                                // var(1761) + var(1762) * SymbExpr::from_u32(256) + var(1763) * SymbExpr::from_u32(65536) + var(1764) * SymbExpr::from_u32(16777216),
                                // var(1765) + var(1766) * SymbExpr::from_u32(256) + var(1767) * SymbExpr::from_u32(65536) + var(1768) * SymbExpr::from_u32(16777216),

                                // 41
                                // var(1769) + var(1770) * SymbExpr::from_u32(256) + var(1771) * SymbExpr::from_u32(65536) + var(1772) * SymbExpr::from_u32(16777216),
                                // var(1773) + var(1774) * SymbExpr::from_u32(256) + var(1775) * SymbExpr::from_u32(65536) + var(1776) * SymbExpr::from_u32(16777216),
                                // var(1777) + var(1778) * SymbExpr::from_u32(256) + var(1779) * SymbExpr::from_u32(65536) + var(1780) * SymbExpr::from_u32(16777216),
                                // var(1781) + var(1782) * SymbExpr::from_u32(256) + var(1783) * SymbExpr::from_u32(65536) + var(1784) * SymbExpr::from_u32(16777216),
                                // var(1785) + var(1786) * SymbExpr::from_u32(256) + var(1787) * SymbExpr::from_u32(65536) + var(1788) * SymbExpr::from_u32(16777216),
                                // var(1789) + var(1790) * SymbExpr::from_u32(256) + var(1791) * SymbExpr::from_u32(65536) + var(1792) * SymbExpr::from_u32(16777216),
                                // var(1793) + var(1794) * SymbExpr::from_u32(256) + var(1795) * SymbExpr::from_u32(65536) + var(1796) * SymbExpr::from_u32(16777216),
                                // var(1797) + var(1798) * SymbExpr::from_u32(256) + var(1799) * SymbExpr::from_u32(65536) + var(1800) * SymbExpr::from_u32(16777216),
                                // var(1801) + var(1802) * SymbExpr::from_u32(256) + var(1803) * SymbExpr::from_u32(65536) + var(1804) * SymbExpr::from_u32(16777216),
                                // var(1805) + var(1806) * SymbExpr::from_u32(256) + var(1807) * SymbExpr::from_u32(65536) + var(1808) * SymbExpr::from_u32(16777216),

                                // 42
                                // var(1809) + var(1810) * SymbExpr::from_u32(256) + var(1811) * SymbExpr::from_u32(65536) + var(1812) * SymbExpr::from_u32(16777216),
                                // var(1813) + var(1814) * SymbExpr::from_u32(256) + var(1815) * SymbExpr::from_u32(65536) + var(1816) * SymbExpr::from_u32(16777216),
                                // var(1817) + var(1818) * SymbExpr::from_u32(256) + var(1819) * SymbExpr::from_u32(65536) + var(1820) * SymbExpr::from_u32(16777216),
                                // var(1821) + var(1822) * SymbExpr::from_u32(256) + var(1823) * SymbExpr::from_u32(65536) + var(1824) * SymbExpr::from_u32(16777216),
                                // var(1825) + var(1826) * SymbExpr::from_u32(256) + var(1827) * SymbExpr::from_u32(65536) + var(1828) * SymbExpr::from_u32(16777216),
                                // var(1829) + var(1830) * SymbExpr::from_u32(256) + var(1831) * SymbExpr::from_u32(65536) + var(1832) * SymbExpr::from_u32(16777216),
                                // var(1833) + var(1834) * SymbExpr::from_u32(256) + var(1835) * SymbExpr::from_u32(65536) + var(1836) * SymbExpr::from_u32(16777216),
                                // var(1837) + var(1838) * SymbExpr::from_u32(256) + var(1839) * SymbExpr::from_u32(65536) + var(1840) * SymbExpr::from_u32(16777216),
                                // var(1841) + var(1842) * SymbExpr::from_u32(256) + var(1843) * SymbExpr::from_u32(65536) + var(1844) * SymbExpr::from_u32(16777216),
                                // var(1845) + var(1846) * SymbExpr::from_u32(256) + var(1847) * SymbExpr::from_u32(65536) + var(1848) * SymbExpr::from_u32(16777216),

                                // 43
                                // var(1849) + var(1850) * SymbExpr::from_u32(256) + var(1851) * SymbExpr::from_u32(65536) + var(1852) * SymbExpr::from_u32(16777216),
                                // var(1853) + var(1854) * SymbExpr::from_u32(256) + var(1855) * SymbExpr::from_u32(65536) + var(1856) * SymbExpr::from_u32(16777216),
                                // var(1857) + var(1858) * SymbExpr::from_u32(256) + var(1859) * SymbExpr::from_u32(65536) + var(1860) * SymbExpr::from_u32(16777216),
                                // var(1861) + var(1862) * SymbExpr::from_u32(256) + var(1863) * SymbExpr::from_u32(65536) + var(1864) * SymbExpr::from_u32(16777216),
                                // var(1865) + var(1866) * SymbExpr::from_u32(256) + var(1867) * SymbExpr::from_u32(65536) + var(1868) * SymbExpr::from_u32(16777216),
                                // var(1869) + var(1870) * SymbExpr::from_u32(256) + var(1871) * SymbExpr::from_u32(65536) + var(1872) * SymbExpr::from_u32(16777216),
                                // var(1873) + var(1874) * SymbExpr::from_u32(256) + var(1875) * SymbExpr::from_u32(65536) + var(1876) * SymbExpr::from_u32(16777216),
                                // var(1877) + var(1878) * SymbExpr::from_u32(256) + var(1879) * SymbExpr::from_u32(65536) + var(1880) * SymbExpr::from_u32(16777216),
                                // var(1881) + var(1882) * SymbExpr::from_u32(256) + var(1883) * SymbExpr::from_u32(65536) + var(1884) * SymbExpr::from_u32(16777216),
                                // var(1885) + var(1886) * SymbExpr::from_u32(256) + var(1887) * SymbExpr::from_u32(65536) + var(1888) * SymbExpr::from_u32(16777216),

                                // 44
                                // var(1889) + var(1890) * SymbExpr::from_u32(256) + var(1891) * SymbExpr::from_u32(65536) + var(1892) * SymbExpr::from_u32(16777216),
                                // var(1893) + var(1894) * SymbExpr::from_u32(256) + var(1895) * SymbExpr::from_u32(65536) + var(1896) * SymbExpr::from_u32(16777216),
                                // var(1897) + var(1898) * SymbExpr::from_u32(256) + var(1899) * SymbExpr::from_u32(65536) + var(1900) * SymbExpr::from_u32(16777216),
                                // var(1901) + var(1902) * SymbExpr::from_u32(256) + var(1903) * SymbExpr::from_u32(65536) + var(1904) * SymbExpr::from_u32(16777216),
                                // var(1905) + var(1906) * SymbExpr::from_u32(256) + var(1907) * SymbExpr::from_u32(65536) + var(1908) * SymbExpr::from_u32(16777216),
                                // var(1909) + var(1910) * SymbExpr::from_u32(256) + var(1911) * SymbExpr::from_u32(65536) + var(1912) * SymbExpr::from_u32(16777216),
                                // var(1913) + var(1914) * SymbExpr::from_u32(256) + var(1915) * SymbExpr::from_u32(65536) + var(1916) * SymbExpr::from_u32(16777216),
                                // var(1917) + var(1918) * SymbExpr::from_u32(256) + var(1919) * SymbExpr::from_u32(65536) + var(1920) * SymbExpr::from_u32(16777216),
                                // var(1921) + var(1922) * SymbExpr::from_u32(256) + var(1923) * SymbExpr::from_u32(65536) + var(1924) * SymbExpr::from_u32(16777216),
                                // var(1925) + var(1926) * SymbExpr::from_u32(256) + var(1927) * SymbExpr::from_u32(65536) + var(1928) * SymbExpr::from_u32(16777216),

                                // 45
                                // var(1929) + var(1930) * SymbExpr::from_u32(256) + var(1931) * SymbExpr::from_u32(65536) + var(1932) * SymbExpr::from_u32(16777216),
                                // var(1933) + var(1934) * SymbExpr::from_u32(256) + var(1935) * SymbExpr::from_u32(65536) + var(1936) * SymbExpr::from_u32(16777216),
                                // var(1937) + var(1938) * SymbExpr::from_u32(256) + var(1939) * SymbExpr::from_u32(65536) + var(1940) * SymbExpr::from_u32(16777216),
                                // var(1941) + var(1942) * SymbExpr::from_u32(256) + var(1943) * SymbExpr::from_u32(65536) + var(1944) * SymbExpr::from_u32(16777216),
                                // var(1945) + var(1946) * SymbExpr::from_u32(256) + var(1947) * SymbExpr::from_u32(65536) + var(1948) * SymbExpr::from_u32(16777216),
                                // var(1949) + var(1950) * SymbExpr::from_u32(256) + var(1951) * SymbExpr::from_u32(65536) + var(1952) * SymbExpr::from_u32(16777216),
                                // var(1953) + var(1954) * SymbExpr::from_u32(256) + var(1955) * SymbExpr::from_u32(65536) + var(1956) * SymbExpr::from_u32(16777216),
                                // var(1957) + var(1958) * SymbExpr::from_u32(256) + var(1959) * SymbExpr::from_u32(65536) + var(1960) * SymbExpr::from_u32(16777216),
                                // var(1961) + var(1962) * SymbExpr::from_u32(256) + var(1963) * SymbExpr::from_u32(65536) + var(1964) * SymbExpr::from_u32(16777216),
                                // var(1965) + var(1966) * SymbExpr::from_u32(256) + var(1967) * SymbExpr::from_u32(65536) + var(1968) * SymbExpr::from_u32(16777216),

                                // 46
                                // var(1969) + var(1970) * SymbExpr::from_u32(256) + var(1971) * SymbExpr::from_u32(65536) + var(1972) * SymbExpr::from_u32(16777216),
                                // var(1973) + var(1974) * SymbExpr::from_u32(256) + var(1975) * SymbExpr::from_u32(65536) + var(1976) * SymbExpr::from_u32(16777216),
                                // var(1977) + var(1978) * SymbExpr::from_u32(256) + var(1979) * SymbExpr::from_u32(65536) + var(1980) * SymbExpr::from_u32(16777216),
                                // var(1981) + var(1982) * SymbExpr::from_u32(256) + var(1983) * SymbExpr::from_u32(65536) + var(1984) * SymbExpr::from_u32(16777216),
                                // var(1985) + var(1986) * SymbExpr::from_u32(256) + var(1987) * SymbExpr::from_u32(65536) + var(1988) * SymbExpr::from_u32(16777216),
                                // var(1989) + var(1990) * SymbExpr::from_u32(256) + var(1991) * SymbExpr::from_u32(65536) + var(1992) * SymbExpr::from_u32(16777216),
                                // var(1993) + var(1994) * SymbExpr::from_u32(256) + var(1995) * SymbExpr::from_u32(65536) + var(1996) * SymbExpr::from_u32(16777216),
                                // var(1997) + var(1998) * SymbExpr::from_u32(256) + var(1999) * SymbExpr::from_u32(65536) + var(2000) * SymbExpr::from_u32(16777216),
                                // var(2001) + var(2002) * SymbExpr::from_u32(256) + var(2003) * SymbExpr::from_u32(65536) + var(2004) * SymbExpr::from_u32(16777216),
                                // var(2005) + var(2006) * SymbExpr::from_u32(256) + var(2007) * SymbExpr::from_u32(65536) + var(2008) * SymbExpr::from_u32(16777216),

                                // 47
                                // var(2009) + var(2010) * SymbExpr::from_u32(256) + var(2011) * SymbExpr::from_u32(65536) + var(2012) * SymbExpr::from_u32(16777216),
                                // var(2013) + var(2014) * SymbExpr::from_u32(256) + var(2015) * SymbExpr::from_u32(65536) + var(2016) * SymbExpr::from_u32(16777216),
                                // var(2017) + var(2018) * SymbExpr::from_u32(256) + var(2019) * SymbExpr::from_u32(65536) + var(2020) * SymbExpr::from_u32(16777216),
                                // var(2021) + var(2022) * SymbExpr::from_u32(256) + var(2023) * SymbExpr::from_u32(65536) + var(2024) * SymbExpr::from_u32(16777216),
                                // var(2025) + var(2026) * SymbExpr::from_u32(256) + var(2027) * SymbExpr::from_u32(65536) + var(2028) * SymbExpr::from_u32(16777216),
                                // var(2029) + var(2030) * SymbExpr::from_u32(256) + var(2031) * SymbExpr::from_u32(65536) + var(2032) * SymbExpr::from_u32(16777216),
                                // var(2033) + var(2034) * SymbExpr::from_u32(256) + var(2035) * SymbExpr::from_u32(65536) + var(2036) * SymbExpr::from_u32(16777216),
                                // var(2037) + var(2038) * SymbExpr::from_u32(256) + var(2039) * SymbExpr::from_u32(65536) + var(2040) * SymbExpr::from_u32(16777216),
                                // var(2041) + var(2042) * SymbExpr::from_u32(256) + var(2043) * SymbExpr::from_u32(65536) + var(2044) * SymbExpr::from_u32(16777216),
                                // var(2045) + var(2046) * SymbExpr::from_u32(256) + var(2047) * SymbExpr::from_u32(65536) + var(2048) * SymbExpr::from_u32(16777216),

                                // 48
                                // var(2049) + var(2050) * SymbExpr::from_u32(256) + var(2051) * SymbExpr::from_u32(65536) + var(2052) * SymbExpr::from_u32(16777216),
                                // var(2053) + var(2054) * SymbExpr::from_u32(256) + var(2055) * SymbExpr::from_u32(65536) + var(2056) * SymbExpr::from_u32(16777216),
                                // var(2057) + var(2058) * SymbExpr::from_u32(256) + var(2059) * SymbExpr::from_u32(65536) + var(2060) * SymbExpr::from_u32(16777216),
                                // var(2061) + var(2062) * SymbExpr::from_u32(256) + var(2063) * SymbExpr::from_u32(65536) + var(2064) * SymbExpr::from_u32(16777216),
                                // var(2065) + var(2066) * SymbExpr::from_u32(256) + var(2067) * SymbExpr::from_u32(65536) + var(2068) * SymbExpr::from_u32(16777216),
                                // var(2069) + var(2070) * SymbExpr::from_u32(256) + var(2071) * SymbExpr::from_u32(65536) + var(2072) * SymbExpr::from_u32(16777216),
                                // var(2073) + var(2074) * SymbExpr::from_u32(256) + var(2075) * SymbExpr::from_u32(65536) + var(2076) * SymbExpr::from_u32(16777216),
                                // var(2077) + var(2078) * SymbExpr::from_u32(256) + var(2079) * SymbExpr::from_u32(65536) + var(2080) * SymbExpr::from_u32(16777216),
                                // var(2081) + var(2082) * SymbExpr::from_u32(256) + var(2083) * SymbExpr::from_u32(65536) + var(2084) * SymbExpr::from_u32(16777216),
                                // var(2085) + var(2086) * SymbExpr::from_u32(256) + var(2087) * SymbExpr::from_u32(65536) + var(2088) * SymbExpr::from_u32(16777216),

                                // 49
                                // var(2089) + var(2090) * SymbExpr::from_u32(256) + var(2091) * SymbExpr::from_u32(65536) + var(2092) * SymbExpr::from_u32(16777216),
                                // var(2093) + var(2094) * SymbExpr::from_u32(256) + var(2095) * SymbExpr::from_u32(65536) + var(2096) * SymbExpr::from_u32(16777216),
                                // var(2097) + var(2098) * SymbExpr::from_u32(256) + var(2099) * SymbExpr::from_u32(65536) + var(2100) * SymbExpr::from_u32(16777216),
                                // var(2101) + var(2102) * SymbExpr::from_u32(256) + var(2103) * SymbExpr::from_u32(65536) + var(2104) * SymbExpr::from_u32(16777216),
                                // var(2105) + var(2106) * SymbExpr::from_u32(256) + var(2107) * SymbExpr::from_u32(65536) + var(2108) * SymbExpr::from_u32(16777216),
                                // var(2109) + var(2110) * SymbExpr::from_u32(256) + var(2111) * SymbExpr::from_u32(65536) + var(2112) * SymbExpr::from_u32(16777216),
                                // var(2113) + var(2114) * SymbExpr::from_u32(256) + var(2115) * SymbExpr::from_u32(65536) + var(2116) * SymbExpr::from_u32(16777216),
                                // var(2117) + var(2118) * SymbExpr::from_u32(256) + var(2119) * SymbExpr::from_u32(65536) + var(2120) * SymbExpr::from_u32(16777216),
                                // var(2121) + var(2122) * SymbExpr::from_u32(256) + var(2123) * SymbExpr::from_u32(65536) + var(2124) * SymbExpr::from_u32(16777216),
                                // var(2125) + var(2126) * SymbExpr::from_u32(256) + var(2127) * SymbExpr::from_u32(65536) + var(2128) * SymbExpr::from_u32(16777216),

                                // 50
                                // var(2129) + var(2130) * SymbExpr::from_u32(256) + var(2131) * SymbExpr::from_u32(65536) + var(2132) * SymbExpr::from_u32(16777216),
                                // var(2133) + var(2134) * SymbExpr::from_u32(256) + var(2135) * SymbExpr::from_u32(65536) + var(2136) * SymbExpr::from_u32(16777216),
                                // var(2137) + var(2138) * SymbExpr::from_u32(256) + var(2139) * SymbExpr::from_u32(65536) + var(2140) * SymbExpr::from_u32(16777216),
                                // var(2141) + var(2142) * SymbExpr::from_u32(256) + var(2143) * SymbExpr::from_u32(65536) + var(2144) * SymbExpr::from_u32(16777216),
                                // var(2145) + var(2146) * SymbExpr::from_u32(256) + var(2147) * SymbExpr::from_u32(65536) + var(2148) * SymbExpr::from_u32(16777216),
                                // var(2149) + var(2150) * SymbExpr::from_u32(256) + var(2151) * SymbExpr::from_u32(65536) + var(2152) * SymbExpr::from_u32(16777216),
                                // var(2153) + var(2154) * SymbExpr::from_u32(256) + var(2155) * SymbExpr::from_u32(65536) + var(2156) * SymbExpr::from_u32(16777216),
                                // var(2157) + var(2158) * SymbExpr::from_u32(256) + var(2159) * SymbExpr::from_u32(65536) + var(2160) * SymbExpr::from_u32(16777216),
                                // var(2161) + var(2162) * SymbExpr::from_u32(256) + var(2163) * SymbExpr::from_u32(65536) + var(2164) * SymbExpr::from_u32(16777216),
                                // var(2165) + var(2166) * SymbExpr::from_u32(256) + var(2167) * SymbExpr::from_u32(65536) + var(2168) * SymbExpr::from_u32(16777216),

                                // 51
                                // var(2169) + var(2170) * SymbExpr::from_u32(256) + var(2171) * SymbExpr::from_u32(65536) + var(2172) * SymbExpr::from_u32(16777216),
                                // var(2173) + var(2174) * SymbExpr::from_u32(256) + var(2175) * SymbExpr::from_u32(65536) + var(2176) * SymbExpr::from_u32(16777216),
                                // var(2177) + var(2178) * SymbExpr::from_u32(256) + var(2179) * SymbExpr::from_u32(65536) + var(2180) * SymbExpr::from_u32(16777216),
                                // var(2181) + var(2182) * SymbExpr::from_u32(256) + var(2183) * SymbExpr::from_u32(65536) + var(2184) * SymbExpr::from_u32(16777216),
                                // var(2185) + var(2186) * SymbExpr::from_u32(256) + var(2187) * SymbExpr::from_u32(65536) + var(2188) * SymbExpr::from_u32(16777216),
                                // var(2189) + var(2190) * SymbExpr::from_u32(256) + var(2191) * SymbExpr::from_u32(65536) + var(2192) * SymbExpr::from_u32(16777216),
                                // var(2193) + var(2194) * SymbExpr::from_u32(256) + var(2195) * SymbExpr::from_u32(65536) + var(2196) * SymbExpr::from_u32(16777216),
                                // var(2197) + var(2198) * SymbExpr::from_u32(256) + var(2199) * SymbExpr::from_u32(65536) + var(2200) * SymbExpr::from_u32(16777216),
                                // var(2201) + var(2202) * SymbExpr::from_u32(256) + var(2203) * SymbExpr::from_u32(65536) + var(2204) * SymbExpr::from_u32(16777216),
                                // var(2205) + var(2206) * SymbExpr::from_u32(256) + var(2207) * SymbExpr::from_u32(65536) + var(2208) * SymbExpr::from_u32(16777216),

                                // 52
                                // var(2209) + var(2210) * SymbExpr::from_u32(256) + var(2211) * SymbExpr::from_u32(65536) + var(2212) * SymbExpr::from_u32(16777216),
                                // var(2213) + var(2214) * SymbExpr::from_u32(256) + var(2215) * SymbExpr::from_u32(65536) + var(2216) * SymbExpr::from_u32(16777216),
                                // var(2217) + var(2218) * SymbExpr::from_u32(256) + var(2219) * SymbExpr::from_u32(65536) + var(2220) * SymbExpr::from_u32(16777216),
                                // var(2221) + var(2222) * SymbExpr::from_u32(256) + var(2223) * SymbExpr::from_u32(65536) + var(2224) * SymbExpr::from_u32(16777216),
                                // var(2225) + var(2226) * SymbExpr::from_u32(256) + var(2227) * SymbExpr::from_u32(65536) + var(2228) * SymbExpr::from_u32(16777216),
                                // var(2229) + var(2230) * SymbExpr::from_u32(256) + var(2231) * SymbExpr::from_u32(65536) + var(2232) * SymbExpr::from_u32(16777216),
                                // var(2233) + var(2234) * SymbExpr::from_u32(256) + var(2235) * SymbExpr::from_u32(65536) + var(2236) * SymbExpr::from_u32(16777216),
                                // var(2237) + var(2238) * SymbExpr::from_u32(256) + var(2239) * SymbExpr::from_u32(65536) + var(2240) * SymbExpr::from_u32(16777216),
                                // var(2241) + var(2242) * SymbExpr::from_u32(256) + var(2243) * SymbExpr::from_u32(65536) + var(2244) * SymbExpr::from_u32(16777216),
                                // var(2245) + var(2246) * SymbExpr::from_u32(256) + var(2247) * SymbExpr::from_u32(65536) + var(2248) * SymbExpr::from_u32(16777216),

                                // 53
                                // var(2249) + var(2250) * SymbExpr::from_u32(256) + var(2251) * SymbExpr::from_u32(65536) + var(2252) * SymbExpr::from_u32(16777216),
                                // var(2253) + var(2254) * SymbExpr::from_u32(256) + var(2255) * SymbExpr::from_u32(65536) + var(2256) * SymbExpr::from_u32(16777216),
                                // var(2257) + var(2258) * SymbExpr::from_u32(256) + var(2259) * SymbExpr::from_u32(65536) + var(2260) * SymbExpr::from_u32(16777216),
                                // var(2261) + var(2262) * SymbExpr::from_u32(256) + var(2263) * SymbExpr::from_u32(65536) + var(2264) * SymbExpr::from_u32(16777216),
                                // var(2265) + var(2266) * SymbExpr::from_u32(256) + var(2267) * SymbExpr::from_u32(65536) + var(2268) * SymbExpr::from_u32(16777216),
                                // var(2269) + var(2270) * SymbExpr::from_u32(256) + var(2271) * SymbExpr::from_u32(65536) + var(2272) * SymbExpr::from_u32(16777216),
                                // var(2273) + var(2274) * SymbExpr::from_u32(256) + var(2275) * SymbExpr::from_u32(65536) + var(2276) * SymbExpr::from_u32(16777216),
                                // var(2277) + var(2278) * SymbExpr::from_u32(256) + var(2279) * SymbExpr::from_u32(65536) + var(2280) * SymbExpr::from_u32(16777216),
                                // var(2281) + var(2282) * SymbExpr::from_u32(256) + var(2283) * SymbExpr::from_u32(65536) + var(2284) * SymbExpr::from_u32(16777216),
                                // var(2285) + var(2286) * SymbExpr::from_u32(256) + var(2287) * SymbExpr::from_u32(65536) + var(2288) * SymbExpr::from_u32(16777216),

                                // 54
                                // var(2289) + var(2290) * SymbExpr::from_u32(256) + var(2291) * SymbExpr::from_u32(65536) + var(2292) * SymbExpr::from_u32(16777216),
                                // var(2293) + var(2294) * SymbExpr::from_u32(256) + var(2295) * SymbExpr::from_u32(65536) + var(2296) * SymbExpr::from_u32(16777216),
                                // var(2297) + var(2298) * SymbExpr::from_u32(256) + var(2299) * SymbExpr::from_u32(65536) + var(2300) * SymbExpr::from_u32(16777216),
                                // var(2301) + var(2302) * SymbExpr::from_u32(256) + var(2303) * SymbExpr::from_u32(65536) + var(2304) * SymbExpr::from_u32(16777216),
                                // var(2305) + var(2306) * SymbExpr::from_u32(256) + var(2307) * SymbExpr::from_u32(65536) + var(2308) * SymbExpr::from_u32(16777216),
                                // var(2309) + var(2310) * SymbExpr::from_u32(256) + var(2311) * SymbExpr::from_u32(65536) + var(2312) * SymbExpr::from_u32(16777216),
                                // var(2313) + var(2314) * SymbExpr::from_u32(256) + var(2315) * SymbExpr::from_u32(65536) + var(2316) * SymbExpr::from_u32(16777216),
                                // var(2317) + var(2318) * SymbExpr::from_u32(256) + var(2319) * SymbExpr::from_u32(65536) + var(2320) * SymbExpr::from_u32(16777216),
                                // var(2321) + var(2322) * SymbExpr::from_u32(256) + var(2323) * SymbExpr::from_u32(65536) + var(2324) * SymbExpr::from_u32(16777216),
                                // var(2325) + var(2326) * SymbExpr::from_u32(256) + var(2327) * SymbExpr::from_u32(65536) + var(2328) * SymbExpr::from_u32(16777216),

                                // 55
                                // var(2329) + var(2330) * SymbExpr::from_u32(256) + var(2331) * SymbExpr::from_u32(65536) + var(2332) * SymbExpr::from_u32(16777216),
                                // var(2333) + var(2334) * SymbExpr::from_u32(256) + var(2335) * SymbExpr::from_u32(65536) + var(2336) * SymbExpr::from_u32(16777216),
                                // var(2337) + var(2338) * SymbExpr::from_u32(256) + var(2339) * SymbExpr::from_u32(65536) + var(2340) * SymbExpr::from_u32(16777216),
                                // var(2341) + var(2342) * SymbExpr::from_u32(256) + var(2343) * SymbExpr::from_u32(65536) + var(2344) * SymbExpr::from_u32(16777216),
                                // var(2345) + var(2346) * SymbExpr::from_u32(256) + var(2347) * SymbExpr::from_u32(65536) + var(2348) * SymbExpr::from_u32(16777216),
                                // var(2349) + var(2350) * SymbExpr::from_u32(256) + var(2351) * SymbExpr::from_u32(65536) + var(2352) * SymbExpr::from_u32(16777216),
                                // var(2353) + var(2354) * SymbExpr::from_u32(256) + var(2355) * SymbExpr::from_u32(65536) + var(2356) * SymbExpr::from_u32(16777216),
                                // var(2357) + var(2358) * SymbExpr::from_u32(256) + var(2359) * SymbExpr::from_u32(65536) + var(2360) * SymbExpr::from_u32(16777216),
                                // var(2361) + var(2362) * SymbExpr::from_u32(256) + var(2363) * SymbExpr::from_u32(65536) + var(2364) * SymbExpr::from_u32(16777216),
                                // var(2365) + var(2366) * SymbExpr::from_u32(256) + var(2367) * SymbExpr::from_u32(65536) + var(2368) * SymbExpr::from_u32(16777216),

                                // state[i] ^= state[i + 8] x8
                                // var(2369)
                                //     + var(2370) * SymbExpr::from_u32(256)
                                //     + var(2371) * SymbExpr::from_u32(65536)
                                //     + var(2372) * SymbExpr::from_u32(16777216),
                                // var(2373)
                                //     + var(2374) * SymbExpr::from_u32(256)
                                //     + var(2375) * SymbExpr::from_u32(65536)
                                //     + var(2376) * SymbExpr::from_u32(16777216),
                                // var(2377)
                                //     + var(2378) * SymbExpr::from_u32(256)
                                //     + var(2379) * SymbExpr::from_u32(65536)
                                //     + var(2380) * SymbExpr::from_u32(16777216),
                                // var(2381)
                                //     + var(2382) * SymbExpr::from_u32(256)
                                //     + var(2383) * SymbExpr::from_u32(65536)
                                //     + var(2384) * SymbExpr::from_u32(16777216),
                                // var(2385)
                                //     + var(2386) * SymbExpr::from_u32(256)
                                //     + var(2387) * SymbExpr::from_u32(65536)
                                //     + var(2388) * SymbExpr::from_u32(16777216),
                                // var(2389)
                                //     + var(2390) * SymbExpr::from_u32(256)
                                //     + var(2391) * SymbExpr::from_u32(65536)
                                //     + var(2392) * SymbExpr::from_u32(16777216),
                                // var(2393)
                                //     + var(2394) * SymbExpr::from_u32(256)
                                //     + var(2395) * SymbExpr::from_u32(65536)
                                //     + var(2396) * SymbExpr::from_u32(16777216),
                                // var(2397)
                                //     + var(2398) * SymbExpr::from_u32(256)
                                //     + var(2399) * SymbExpr::from_u32(65536)
                                //     + var(2400) * SymbExpr::from_u32(16777216),
                                // var(2401)
                                //     + var(2402) * SymbExpr::from_u32(256)
                                //     + var(2403) * SymbExpr::from_u32(65536)
                                //     + var(2404) * SymbExpr::from_u32(16777216),
                                // var(2405)
                                //     + var(2406) * SymbExpr::from_u32(256)
                                //     + var(2407) * SymbExpr::from_u32(65536)
                                //     + var(2408) * SymbExpr::from_u32(16777216),
                                // var(2409)
                                //     + var(2410) * SymbExpr::from_u32(256)
                                //     + var(2411) * SymbExpr::from_u32(65536)
                                //     + var(2412) * SymbExpr::from_u32(16777216),
                                // var(2413)
                                //     + var(2414) * SymbExpr::from_u32(256)
                                //     + var(2415) * SymbExpr::from_u32(65536)
                                //     + var(2416) * SymbExpr::from_u32(16777216),
                                // var(2417)
                                //     + var(2418) * SymbExpr::from_u32(256)
                                //     + var(2419) * SymbExpr::from_u32(65536)
                                //     + var(2420) * SymbExpr::from_u32(16777216),
                                // var(2421)
                                //     + var(2422) * SymbExpr::from_u32(256)
                                //     + var(2423) * SymbExpr::from_u32(65536)
                                //     + var(2424) * SymbExpr::from_u32(16777216),
                                // var(2425)
                                //     + var(2426) * SymbExpr::from_u32(256)
                                //     + var(2427) * SymbExpr::from_u32(65536)
                                //     + var(2428) * SymbExpr::from_u32(16777216),
                                // var(2429)
                                //     + var(2430) * SymbExpr::from_u32(256)
                                //     + var(2431) * SymbExpr::from_u32(65536)
                                //     + var(2432) * SymbExpr::from_u32(16777216),
                                // var(2433)
                                //     + var(2434) * SymbExpr::from_u32(256)
                                //     + var(2435) * SymbExpr::from_u32(65536)
                                //     + var(2436) * SymbExpr::from_u32(16777216),
                                // var(2437)
                                //     + var(2438) * SymbExpr::from_u32(256)
                                //     + var(2439) * SymbExpr::from_u32(65536)
                                //     + var(2440) * SymbExpr::from_u32(16777216),
                                // var(2441)
                                //     + var(2442) * SymbExpr::from_u32(256)
                                //     + var(2443) * SymbExpr::from_u32(65536)
                                //     + var(2444) * SymbExpr::from_u32(16777216),
                                // var(2445)
                                //     + var(2446) * SymbExpr::from_u32(256)
                                //     + var(2447) * SymbExpr::from_u32(65536)
                                //     + var(2448) * SymbExpr::from_u32(16777216),
                                // var(2449)
                                //     + var(2450) * SymbExpr::from_u32(256)
                                //     + var(2451) * SymbExpr::from_u32(65536)
                                //     + var(2452) * SymbExpr::from_u32(16777216),
                                // var(2453)
                                //     + var(2454) * SymbExpr::from_u32(256)
                                //     + var(2455) * SymbExpr::from_u32(65536)
                                //     + var(2456) * SymbExpr::from_u32(16777216),
                                // var(2457)
                                //     + var(2458) * SymbExpr::from_u32(256)
                                //     + var(2459) * SymbExpr::from_u32(65536)
                                //     + var(2460) * SymbExpr::from_u32(16777216),
                                // var(2461)
                                //     + var(2462) * SymbExpr::from_u32(256)
                                //     + var(2463) * SymbExpr::from_u32(65536)
                                //     + var(2464) * SymbExpr::from_u32(16777216),

                                // state[i + 8] ^= chaining_value[i]
                                // var(2465)
                                //     + var(2466) * SymbExpr::from_u32(256)
                                //     + var(2467) * SymbExpr::from_u32(65536)
                                //     + var(2468) * SymbExpr::from_u32(16777216),
                                // var(2469)
                                //     + var(2470) * SymbExpr::from_u32(256)
                                //     + var(2471) * SymbExpr::from_u32(65536)
                                //     + var(2472) * SymbExpr::from_u32(16777216),
                                // var(2473)
                                //     + var(2474) * SymbExpr::from_u32(256)
                                //     + var(2475) * SymbExpr::from_u32(65536)
                                //     + var(2476) * SymbExpr::from_u32(16777216),
                                // var(2477)
                                //     + var(2478) * SymbExpr::from_u32(256)
                                //     + var(2479) * SymbExpr::from_u32(65536)
                                //     + var(2480) * SymbExpr::from_u32(16777216),
                                // var(2481)
                                //     + var(2482) * SymbExpr::from_u32(256)
                                //     + var(2483) * SymbExpr::from_u32(65536)
                                //     + var(2484) * SymbExpr::from_u32(16777216),
                                // var(2485)
                                //     + var(2486) * SymbExpr::from_u32(256)
                                //     + var(2487) * SymbExpr::from_u32(65536)
                                //     + var(2488) * SymbExpr::from_u32(16777216),
                                // var(2489)
                                //     + var(2490) * SymbExpr::from_u32(256)
                                //     + var(2491) * SymbExpr::from_u32(65536)
                                //     + var(2492) * SymbExpr::from_u32(16777216),
                                // var(2493)
                                //     + var(2494) * SymbExpr::from_u32(256)
                                //     + var(2495) * SymbExpr::from_u32(65536)
                                //     + var(2496) * SymbExpr::from_u32(16777216),
                                // var(2497)
                                //     + var(2498) * SymbExpr::from_u32(256)
                                //     + var(2499) * SymbExpr::from_u32(65536)
                                //     + var(2500) * SymbExpr::from_u32(16777216),
                                // var(2501)
                                //     + var(2502) * SymbExpr::from_u32(256)
                                //     + var(2503) * SymbExpr::from_u32(65536)
                                //     + var(2504) * SymbExpr::from_u32(16777216),
                                // var(2505)
                                //     + var(2506) * SymbExpr::from_u32(256)
                                //     + var(2507) * SymbExpr::from_u32(65536)
                                //     + var(2508) * SymbExpr::from_u32(16777216),
                                // var(2509)
                                //     + var(2510) * SymbExpr::from_u32(256)
                                //     + var(2511) * SymbExpr::from_u32(65536)
                                //     + var(2512) * SymbExpr::from_u32(16777216),
                                // var(2513)
                                //     + var(2514) * SymbExpr::from_u32(256)
                                //     + var(2515) * SymbExpr::from_u32(65536)
                                //     + var(2516) * SymbExpr::from_u32(16777216),
                                // var(2517)
                                //     + var(2518) * SymbExpr::from_u32(256)
                                //     + var(2519) * SymbExpr::from_u32(65536)
                                //     + var(2520) * SymbExpr::from_u32(16777216),
                                // var(2521)
                                //     + var(2522) * SymbExpr::from_u32(256)
                                //     + var(2523) * SymbExpr::from_u32(65536)
                                //     + var(2524) * SymbExpr::from_u32(16777216),
                                // var(2525)
                                //     + var(2526) * SymbExpr::from_u32(256)
                                //     + var(2527) * SymbExpr::from_u32(65536)
                                //     + var(2528) * SymbExpr::from_u32(16777216),
                                // var(2529)
                                //     + var(2530) * SymbExpr::from_u32(256)
                                //     + var(2531) * SymbExpr::from_u32(65536)
                                //     + var(2532) * SymbExpr::from_u32(16777216),
                                // var(2533)
                                //     + var(2534) * SymbExpr::from_u32(256)
                                //     + var(2535) * SymbExpr::from_u32(65536)
                                //     + var(2536) * SymbExpr::from_u32(16777216),
                                // var(2537)
                                //     + var(2538) * SymbExpr::from_u32(256)
                                //     + var(2539) * SymbExpr::from_u32(65536)
                                //     + var(2540) * SymbExpr::from_u32(16777216),
                                // var(2541)
                                //     + var(2542) * SymbExpr::from_u32(256)
                                //     + var(2543) * SymbExpr::from_u32(65536)
                                //     + var(2544) * SymbExpr::from_u32(16777216),
                                // var(2545)
                                //     + var(2546) * SymbExpr::from_u32(256)
                                //     + var(2547) * SymbExpr::from_u32(65536)
                                //     + var(2548) * SymbExpr::from_u32(16777216),
                                // var(2549)
                                //     + var(2550) * SymbExpr::from_u32(256)
                                //     + var(2551) * SymbExpr::from_u32(65536)
                                //     + var(2552) * SymbExpr::from_u32(16777216),
                                // var(2553)
                                //     + var(2554) * SymbExpr::from_u32(256)
                                //     + var(2555) * SymbExpr::from_u32(65536)
                                //     + var(2556) * SymbExpr::from_u32(16777216),
                                // var(2557)
                                //     + var(2558) * SymbExpr::from_u32(256)
                                //     + var(2559) * SymbExpr::from_u32(65536)
                                //     + var(2560) * SymbExpr::from_u32(16777216),
                                var(2561)
                                    + var(2562) * SymbExpr::from_u32(256)
                                    + var(2563) * SymbExpr::from_u32(65536)
                                    + var(2564) * SymbExpr::from_u32(16777216),
                                var(2565)
                                    + var(2566) * SymbExpr::from_u32(256)
                                    + var(2567) * SymbExpr::from_u32(65536)
                                    + var(2568) * SymbExpr::from_u32(16777216),
                                var(2569)
                                    + var(2570) * SymbExpr::from_u32(256)
                                    + var(2571) * SymbExpr::from_u32(65536)
                                    + var(2572) * SymbExpr::from_u32(16777216),
                                var(2573)
                                    + var(2574) * SymbExpr::from_u32(256)
                                    + var(2575) * SymbExpr::from_u32(65536)
                                    + var(2576) * SymbExpr::from_u32(16777216),
                                var(2577)
                                    + var(2578) * SymbExpr::from_u32(256)
                                    + var(2579) * SymbExpr::from_u32(65536)
                                    + var(2580) * SymbExpr::from_u32(16777216),
                                var(2581)
                                    + var(2582) * SymbExpr::from_u32(256)
                                    + var(2583) * SymbExpr::from_u32(65536)
                                    + var(2584) * SymbExpr::from_u32(16777216),
                                var(2585)
                                    + var(2586) * SymbExpr::from_u32(256)
                                    + var(2587) * SymbExpr::from_u32(65536)
                                    + var(2588) * SymbExpr::from_u32(16777216),
                                var(2589)
                                    + var(2590) * SymbExpr::from_u32(256)
                                    + var(2591) * SymbExpr::from_u32(65536)
                                    + var(2592) * SymbExpr::from_u32(16777216),
                                var(2593)
                                    + var(2594) * SymbExpr::from_u32(256)
                                    + var(2595) * SymbExpr::from_u32(65536)
                                    + var(2596) * SymbExpr::from_u32(16777216),
                                var(2597)
                                    + var(2598) * SymbExpr::from_u32(256)
                                    + var(2599) * SymbExpr::from_u32(65536)
                                    + var(2600) * SymbExpr::from_u32(16777216),
                                var(2601)
                                    + var(2602) * SymbExpr::from_u32(256)
                                    + var(2603) * SymbExpr::from_u32(65536)
                                    + var(2604) * SymbExpr::from_u32(16777216),
                                var(2605)
                                    + var(2606) * SymbExpr::from_u32(256)
                                    + var(2607) * SymbExpr::from_u32(65536)
                                    + var(2608) * SymbExpr::from_u32(16777216),
                                var(2609)
                                    + var(2610) * SymbExpr::from_u32(256)
                                    + var(2611) * SymbExpr::from_u32(65536)
                                    + var(2612) * SymbExpr::from_u32(16777216),
                                var(2613)
                                    + var(2614) * SymbExpr::from_u32(256)
                                    + var(2615) * SymbExpr::from_u32(65536)
                                    + var(2616) * SymbExpr::from_u32(16777216),
                                var(2617)
                                    + var(2618) * SymbExpr::from_u32(256)
                                    + var(2619) * SymbExpr::from_u32(65536)
                                    + var(2620) * SymbExpr::from_u32(16777216),
                                var(2621)
                                    + var(2622) * SymbExpr::from_u32(256)
                                    + var(2623) * SymbExpr::from_u32(65536)
                                    + var(2624) * SymbExpr::from_u32(16777216),
                                var(2625)
                                    + var(2626) * SymbExpr::from_u32(256)
                                    + var(2627) * SymbExpr::from_u32(65536)
                                    + var(2628) * SymbExpr::from_u32(16777216),
                                var(2629)
                                    + var(2630) * SymbExpr::from_u32(256)
                                    + var(2631) * SymbExpr::from_u32(65536)
                                    + var(2632) * SymbExpr::from_u32(16777216),
                                var(2633)
                                    + var(2634) * SymbExpr::from_u32(256)
                                    + var(2635) * SymbExpr::from_u32(65536)
                                    + var(2636) * SymbExpr::from_u32(16777216),
                                var(2637)
                                    + var(2638) * SymbExpr::from_u32(256)
                                    + var(2639) * SymbExpr::from_u32(65536)
                                    + var(2640) * SymbExpr::from_u32(16777216),
                                var(2641)
                                    + var(2642) * SymbExpr::from_u32(256)
                                    + var(2643) * SymbExpr::from_u32(65536)
                                    + var(2644) * SymbExpr::from_u32(16777216),
                                var(2645)
                                    + var(2646) * SymbExpr::from_u32(256)
                                    + var(2647) * SymbExpr::from_u32(65536)
                                    + var(2648) * SymbExpr::from_u32(16777216),
                                var(2649)
                                    + var(2650) * SymbExpr::from_u32(256)
                                    + var(2651) * SymbExpr::from_u32(65536)
                                    + var(2652) * SymbExpr::from_u32(16777216),
                                var(2653)
                                    + var(2654) * SymbExpr::from_u32(256)
                                    + var(2655) * SymbExpr::from_u32(65536)
                                    + var(2656) * SymbExpr::from_u32(16777216),
                                var(2657)
                                    + var(2658) * SymbExpr::from_u32(256)
                                    + var(2659) * SymbExpr::from_u32(65536)
                                    + var(2660) * SymbExpr::from_u32(16777216),
                                var(2661)
                                    + var(2662) * SymbExpr::from_u32(256)
                                    + var(2663) * SymbExpr::from_u32(65536)
                                    + var(2664) * SymbExpr::from_u32(16777216),
                                var(2665)
                                    + var(2666) * SymbExpr::from_u32(256)
                                    + var(2667) * SymbExpr::from_u32(65536)
                                    + var(2668) * SymbExpr::from_u32(16777216),
                                var(2669)
                                    + var(2670) * SymbExpr::from_u32(256)
                                    + var(2671) * SymbExpr::from_u32(65536)
                                    + var(2672) * SymbExpr::from_u32(16777216),
                                var(2673)
                                    + var(2674) * SymbExpr::from_u32(256)
                                    + var(2675) * SymbExpr::from_u32(65536)
                                    + var(2676) * SymbExpr::from_u32(16777216),
                                var(2677)
                                    + var(2678) * SymbExpr::from_u32(256)
                                    + var(2679) * SymbExpr::from_u32(65536)
                                    + var(2680) * SymbExpr::from_u32(16777216),
                                var(2681)
                                    + var(2682) * SymbExpr::from_u32(256)
                                    + var(2683) * SymbExpr::from_u32(65536)
                                    + var(2684) * SymbExpr::from_u32(16777216),
                                var(2685)
                                    + var(2686) * SymbExpr::from_u32(256)
                                    + var(2687) * SymbExpr::from_u32(65536)
                                    + var(2688) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 0
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(129)
                                    + var(130) * SymbExpr::from_u32(256)
                                    + var(131) * SymbExpr::from_u32(65536)
                                    + var(132) * SymbExpr::from_u32(16777216),
                                var(133)
                                    + var(134) * SymbExpr::from_u32(256)
                                    + var(135) * SymbExpr::from_u32(65536)
                                    + var(136) * SymbExpr::from_u32(16777216),
                                var(137)
                                    + var(138) * SymbExpr::from_u32(256)
                                    + var(139) * SymbExpr::from_u32(65536)
                                    + var(140) * SymbExpr::from_u32(16777216),
                                var(141)
                                    + var(142) * SymbExpr::from_u32(256)
                                    + var(143) * SymbExpr::from_u32(65536)
                                    + var(144) * SymbExpr::from_u32(16777216),
                                var(145)
                                    + var(146) * SymbExpr::from_u32(256)
                                    + var(147) * SymbExpr::from_u32(65536)
                                    + var(148) * SymbExpr::from_u32(16777216),
                                var(149)
                                    + var(150) * SymbExpr::from_u32(256)
                                    + var(151) * SymbExpr::from_u32(65536)
                                    + var(152) * SymbExpr::from_u32(16777216),
                                var(153)
                                    + var(154) * SymbExpr::from_u32(256)
                                    + var(155) * SymbExpr::from_u32(65536)
                                    + var(156) * SymbExpr::from_u32(16777216),
                                var(157)
                                    + var(158) * SymbExpr::from_u32(256)
                                    + var(159) * SymbExpr::from_u32(65536)
                                    + var(160) * SymbExpr::from_u32(16777216),
                                var(161)
                                    + var(162) * SymbExpr::from_u32(256)
                                    + var(163) * SymbExpr::from_u32(65536)
                                    + var(164) * SymbExpr::from_u32(16777216),
                                var(165)
                                    + var(166) * SymbExpr::from_u32(256)
                                    + var(167) * SymbExpr::from_u32(65536)
                                    + var(168) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 1
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(169)
                                    + var(170) * SymbExpr::from_u32(256)
                                    + var(171) * SymbExpr::from_u32(65536)
                                    + var(172) * SymbExpr::from_u32(16777216),
                                var(173)
                                    + var(174) * SymbExpr::from_u32(256)
                                    + var(175) * SymbExpr::from_u32(65536)
                                    + var(176) * SymbExpr::from_u32(16777216),
                                var(177)
                                    + var(178) * SymbExpr::from_u32(256)
                                    + var(179) * SymbExpr::from_u32(65536)
                                    + var(180) * SymbExpr::from_u32(16777216),
                                var(181)
                                    + var(182) * SymbExpr::from_u32(256)
                                    + var(183) * SymbExpr::from_u32(65536)
                                    + var(184) * SymbExpr::from_u32(16777216),
                                var(185)
                                    + var(186) * SymbExpr::from_u32(256)
                                    + var(187) * SymbExpr::from_u32(65536)
                                    + var(188) * SymbExpr::from_u32(16777216),
                                var(189)
                                    + var(190) * SymbExpr::from_u32(256)
                                    + var(191) * SymbExpr::from_u32(65536)
                                    + var(192) * SymbExpr::from_u32(16777216),
                                var(193)
                                    + var(194) * SymbExpr::from_u32(256)
                                    + var(195) * SymbExpr::from_u32(65536)
                                    + var(196) * SymbExpr::from_u32(16777216),
                                var(197)
                                    + var(198) * SymbExpr::from_u32(256)
                                    + var(199) * SymbExpr::from_u32(65536)
                                    + var(200) * SymbExpr::from_u32(16777216),
                                var(201)
                                    + var(202) * SymbExpr::from_u32(256)
                                    + var(203) * SymbExpr::from_u32(65536)
                                    + var(204) * SymbExpr::from_u32(16777216),
                                var(205)
                                    + var(206) * SymbExpr::from_u32(256)
                                    + var(207) * SymbExpr::from_u32(65536)
                                    + var(208) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 2
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(209)
                                    + var(210) * SymbExpr::from_u32(256)
                                    + var(211) * SymbExpr::from_u32(65536)
                                    + var(212) * SymbExpr::from_u32(16777216),
                                var(213)
                                    + var(214) * SymbExpr::from_u32(256)
                                    + var(215) * SymbExpr::from_u32(65536)
                                    + var(216) * SymbExpr::from_u32(16777216),
                                var(217)
                                    + var(218) * SymbExpr::from_u32(256)
                                    + var(219) * SymbExpr::from_u32(65536)
                                    + var(220) * SymbExpr::from_u32(16777216),
                                var(221)
                                    + var(222) * SymbExpr::from_u32(256)
                                    + var(223) * SymbExpr::from_u32(65536)
                                    + var(224) * SymbExpr::from_u32(16777216),
                                var(225)
                                    + var(226) * SymbExpr::from_u32(256)
                                    + var(227) * SymbExpr::from_u32(65536)
                                    + var(228) * SymbExpr::from_u32(16777216),
                                var(229)
                                    + var(230) * SymbExpr::from_u32(256)
                                    + var(231) * SymbExpr::from_u32(65536)
                                    + var(232) * SymbExpr::from_u32(16777216),
                                var(233)
                                    + var(234) * SymbExpr::from_u32(256)
                                    + var(235) * SymbExpr::from_u32(65536)
                                    + var(236) * SymbExpr::from_u32(16777216),
                                var(237)
                                    + var(238) * SymbExpr::from_u32(256)
                                    + var(239) * SymbExpr::from_u32(65536)
                                    + var(240) * SymbExpr::from_u32(16777216),
                                var(241)
                                    + var(242) * SymbExpr::from_u32(256)
                                    + var(243) * SymbExpr::from_u32(65536)
                                    + var(244) * SymbExpr::from_u32(16777216),
                                var(245)
                                    + var(246) * SymbExpr::from_u32(256)
                                    + var(247) * SymbExpr::from_u32(65536)
                                    + var(248) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 3
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(249)
                                    + var(250) * SymbExpr::from_u32(256)
                                    + var(251) * SymbExpr::from_u32(65536)
                                    + var(252) * SymbExpr::from_u32(16777216),
                                var(253)
                                    + var(254) * SymbExpr::from_u32(256)
                                    + var(255) * SymbExpr::from_u32(65536)
                                    + var(256) * SymbExpr::from_u32(16777216),
                                var(257)
                                    + var(258) * SymbExpr::from_u32(256)
                                    + var(259) * SymbExpr::from_u32(65536)
                                    + var(260) * SymbExpr::from_u32(16777216),
                                var(261)
                                    + var(262) * SymbExpr::from_u32(256)
                                    + var(263) * SymbExpr::from_u32(65536)
                                    + var(264) * SymbExpr::from_u32(16777216),
                                var(265)
                                    + var(266) * SymbExpr::from_u32(256)
                                    + var(267) * SymbExpr::from_u32(65536)
                                    + var(268) * SymbExpr::from_u32(16777216),
                                var(269)
                                    + var(270) * SymbExpr::from_u32(256)
                                    + var(271) * SymbExpr::from_u32(65536)
                                    + var(272) * SymbExpr::from_u32(16777216),
                                var(273)
                                    + var(274) * SymbExpr::from_u32(256)
                                    + var(275) * SymbExpr::from_u32(65536)
                                    + var(276) * SymbExpr::from_u32(16777216),
                                var(277)
                                    + var(278) * SymbExpr::from_u32(256)
                                    + var(279) * SymbExpr::from_u32(65536)
                                    + var(280) * SymbExpr::from_u32(16777216),
                                var(281)
                                    + var(282) * SymbExpr::from_u32(256)
                                    + var(283) * SymbExpr::from_u32(65536)
                                    + var(284) * SymbExpr::from_u32(16777216),
                                var(285)
                                    + var(286) * SymbExpr::from_u32(256)
                                    + var(287) * SymbExpr::from_u32(65536)
                                    + var(288) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 4
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(289)
                                    + var(290) * SymbExpr::from_u32(256)
                                    + var(291) * SymbExpr::from_u32(65536)
                                    + var(292) * SymbExpr::from_u32(16777216),
                                var(293)
                                    + var(294) * SymbExpr::from_u32(256)
                                    + var(295) * SymbExpr::from_u32(65536)
                                    + var(296) * SymbExpr::from_u32(16777216),
                                var(297)
                                    + var(298) * SymbExpr::from_u32(256)
                                    + var(299) * SymbExpr::from_u32(65536)
                                    + var(300) * SymbExpr::from_u32(16777216),
                                var(301)
                                    + var(302) * SymbExpr::from_u32(256)
                                    + var(303) * SymbExpr::from_u32(65536)
                                    + var(304) * SymbExpr::from_u32(16777216),
                                var(305)
                                    + var(306) * SymbExpr::from_u32(256)
                                    + var(307) * SymbExpr::from_u32(65536)
                                    + var(308) * SymbExpr::from_u32(16777216),
                                var(309)
                                    + var(310) * SymbExpr::from_u32(256)
                                    + var(311) * SymbExpr::from_u32(65536)
                                    + var(312) * SymbExpr::from_u32(16777216),
                                var(313)
                                    + var(314) * SymbExpr::from_u32(256)
                                    + var(315) * SymbExpr::from_u32(65536)
                                    + var(316) * SymbExpr::from_u32(16777216),
                                var(317)
                                    + var(318) * SymbExpr::from_u32(256)
                                    + var(319) * SymbExpr::from_u32(65536)
                                    + var(320) * SymbExpr::from_u32(16777216),
                                var(321)
                                    + var(322) * SymbExpr::from_u32(256)
                                    + var(323) * SymbExpr::from_u32(65536)
                                    + var(324) * SymbExpr::from_u32(16777216),
                                var(325)
                                    + var(326) * SymbExpr::from_u32(256)
                                    + var(327) * SymbExpr::from_u32(65536)
                                    + var(328) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 5
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(329)
                                    + var(330) * SymbExpr::from_u32(256)
                                    + var(331) * SymbExpr::from_u32(65536)
                                    + var(332) * SymbExpr::from_u32(16777216),
                                var(333)
                                    + var(334) * SymbExpr::from_u32(256)
                                    + var(335) * SymbExpr::from_u32(65536)
                                    + var(336) * SymbExpr::from_u32(16777216),
                                var(337)
                                    + var(338) * SymbExpr::from_u32(256)
                                    + var(339) * SymbExpr::from_u32(65536)
                                    + var(340) * SymbExpr::from_u32(16777216),
                                var(341)
                                    + var(342) * SymbExpr::from_u32(256)
                                    + var(343) * SymbExpr::from_u32(65536)
                                    + var(344) * SymbExpr::from_u32(16777216),
                                var(345)
                                    + var(346) * SymbExpr::from_u32(256)
                                    + var(347) * SymbExpr::from_u32(65536)
                                    + var(348) * SymbExpr::from_u32(16777216),
                                var(349)
                                    + var(350) * SymbExpr::from_u32(256)
                                    + var(351) * SymbExpr::from_u32(65536)
                                    + var(352) * SymbExpr::from_u32(16777216),
                                var(353)
                                    + var(354) * SymbExpr::from_u32(256)
                                    + var(355) * SymbExpr::from_u32(65536)
                                    + var(356) * SymbExpr::from_u32(16777216),
                                var(357)
                                    + var(358) * SymbExpr::from_u32(256)
                                    + var(359) * SymbExpr::from_u32(65536)
                                    + var(360) * SymbExpr::from_u32(16777216),
                                var(361)
                                    + var(362) * SymbExpr::from_u32(256)
                                    + var(363) * SymbExpr::from_u32(65536)
                                    + var(364) * SymbExpr::from_u32(16777216),
                                var(365)
                                    + var(366) * SymbExpr::from_u32(256)
                                    + var(367) * SymbExpr::from_u32(65536)
                                    + var(368) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 6
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(369)
                                    + var(370) * SymbExpr::from_u32(256)
                                    + var(371) * SymbExpr::from_u32(65536)
                                    + var(372) * SymbExpr::from_u32(16777216),
                                var(373)
                                    + var(374) * SymbExpr::from_u32(256)
                                    + var(375) * SymbExpr::from_u32(65536)
                                    + var(376) * SymbExpr::from_u32(16777216),
                                var(377)
                                    + var(378) * SymbExpr::from_u32(256)
                                    + var(379) * SymbExpr::from_u32(65536)
                                    + var(380) * SymbExpr::from_u32(16777216),
                                var(381)
                                    + var(382) * SymbExpr::from_u32(256)
                                    + var(383) * SymbExpr::from_u32(65536)
                                    + var(384) * SymbExpr::from_u32(16777216),
                                var(385)
                                    + var(386) * SymbExpr::from_u32(256)
                                    + var(387) * SymbExpr::from_u32(65536)
                                    + var(388) * SymbExpr::from_u32(16777216),
                                var(389)
                                    + var(390) * SymbExpr::from_u32(256)
                                    + var(391) * SymbExpr::from_u32(65536)
                                    + var(392) * SymbExpr::from_u32(16777216),
                                var(393)
                                    + var(394) * SymbExpr::from_u32(256)
                                    + var(395) * SymbExpr::from_u32(65536)
                                    + var(396) * SymbExpr::from_u32(16777216),
                                var(397)
                                    + var(398) * SymbExpr::from_u32(256)
                                    + var(399) * SymbExpr::from_u32(65536)
                                    + var(400) * SymbExpr::from_u32(16777216),
                                var(401)
                                    + var(402) * SymbExpr::from_u32(256)
                                    + var(403) * SymbExpr::from_u32(65536)
                                    + var(404) * SymbExpr::from_u32(16777216),
                                var(405)
                                    + var(406) * SymbExpr::from_u32(256)
                                    + var(407) * SymbExpr::from_u32(65536)
                                    + var(408) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 7
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(409)
                                    + var(410) * SymbExpr::from_u32(256)
                                    + var(411) * SymbExpr::from_u32(65536)
                                    + var(412) * SymbExpr::from_u32(16777216),
                                var(413)
                                    + var(414) * SymbExpr::from_u32(256)
                                    + var(415) * SymbExpr::from_u32(65536)
                                    + var(416) * SymbExpr::from_u32(16777216),
                                var(417)
                                    + var(418) * SymbExpr::from_u32(256)
                                    + var(419) * SymbExpr::from_u32(65536)
                                    + var(420) * SymbExpr::from_u32(16777216),
                                var(421)
                                    + var(422) * SymbExpr::from_u32(256)
                                    + var(423) * SymbExpr::from_u32(65536)
                                    + var(424) * SymbExpr::from_u32(16777216),
                                var(425)
                                    + var(426) * SymbExpr::from_u32(256)
                                    + var(427) * SymbExpr::from_u32(65536)
                                    + var(428) * SymbExpr::from_u32(16777216),
                                var(429)
                                    + var(430) * SymbExpr::from_u32(256)
                                    + var(431) * SymbExpr::from_u32(65536)
                                    + var(432) * SymbExpr::from_u32(16777216),
                                var(433)
                                    + var(434) * SymbExpr::from_u32(256)
                                    + var(435) * SymbExpr::from_u32(65536)
                                    + var(436) * SymbExpr::from_u32(16777216),
                                var(437)
                                    + var(438) * SymbExpr::from_u32(256)
                                    + var(439) * SymbExpr::from_u32(65536)
                                    + var(440) * SymbExpr::from_u32(16777216),
                                var(441)
                                    + var(442) * SymbExpr::from_u32(256)
                                    + var(443) * SymbExpr::from_u32(65536)
                                    + var(444) * SymbExpr::from_u32(16777216),
                                var(445)
                                    + var(446) * SymbExpr::from_u32(256)
                                    + var(447) * SymbExpr::from_u32(65536)
                                    + var(448) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 8
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(449)
                                    + var(450) * SymbExpr::from_u32(256)
                                    + var(451) * SymbExpr::from_u32(65536)
                                    + var(452) * SymbExpr::from_u32(16777216),
                                var(453)
                                    + var(454) * SymbExpr::from_u32(256)
                                    + var(455) * SymbExpr::from_u32(65536)
                                    + var(456) * SymbExpr::from_u32(16777216),
                                var(457)
                                    + var(458) * SymbExpr::from_u32(256)
                                    + var(459) * SymbExpr::from_u32(65536)
                                    + var(460) * SymbExpr::from_u32(16777216),
                                var(461)
                                    + var(462) * SymbExpr::from_u32(256)
                                    + var(463) * SymbExpr::from_u32(65536)
                                    + var(464) * SymbExpr::from_u32(16777216),
                                var(465)
                                    + var(466) * SymbExpr::from_u32(256)
                                    + var(467) * SymbExpr::from_u32(65536)
                                    + var(468) * SymbExpr::from_u32(16777216),
                                var(469)
                                    + var(470) * SymbExpr::from_u32(256)
                                    + var(471) * SymbExpr::from_u32(65536)
                                    + var(472) * SymbExpr::from_u32(16777216),
                                var(473)
                                    + var(474) * SymbExpr::from_u32(256)
                                    + var(475) * SymbExpr::from_u32(65536)
                                    + var(476) * SymbExpr::from_u32(16777216),
                                var(477)
                                    + var(478) * SymbExpr::from_u32(256)
                                    + var(479) * SymbExpr::from_u32(65536)
                                    + var(480) * SymbExpr::from_u32(16777216),
                                var(481)
                                    + var(482) * SymbExpr::from_u32(256)
                                    + var(483) * SymbExpr::from_u32(65536)
                                    + var(484) * SymbExpr::from_u32(16777216),
                                var(485)
                                    + var(486) * SymbExpr::from_u32(256)
                                    + var(487) * SymbExpr::from_u32(65536)
                                    + var(488) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 9
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(489)
                                    + var(490) * SymbExpr::from_u32(256)
                                    + var(491) * SymbExpr::from_u32(65536)
                                    + var(492) * SymbExpr::from_u32(16777216),
                                var(493)
                                    + var(494) * SymbExpr::from_u32(256)
                                    + var(495) * SymbExpr::from_u32(65536)
                                    + var(496) * SymbExpr::from_u32(16777216),
                                var(497)
                                    + var(498) * SymbExpr::from_u32(256)
                                    + var(499) * SymbExpr::from_u32(65536)
                                    + var(500) * SymbExpr::from_u32(16777216),
                                var(501)
                                    + var(502) * SymbExpr::from_u32(256)
                                    + var(503) * SymbExpr::from_u32(65536)
                                    + var(504) * SymbExpr::from_u32(16777216),
                                var(505)
                                    + var(506) * SymbExpr::from_u32(256)
                                    + var(507) * SymbExpr::from_u32(65536)
                                    + var(508) * SymbExpr::from_u32(16777216),
                                var(509)
                                    + var(510) * SymbExpr::from_u32(256)
                                    + var(511) * SymbExpr::from_u32(65536)
                                    + var(512) * SymbExpr::from_u32(16777216),
                                var(513)
                                    + var(514) * SymbExpr::from_u32(256)
                                    + var(515) * SymbExpr::from_u32(65536)
                                    + var(516) * SymbExpr::from_u32(16777216),
                                var(517)
                                    + var(518) * SymbExpr::from_u32(256)
                                    + var(519) * SymbExpr::from_u32(65536)
                                    + var(520) * SymbExpr::from_u32(16777216),
                                var(521)
                                    + var(522) * SymbExpr::from_u32(256)
                                    + var(523) * SymbExpr::from_u32(65536)
                                    + var(524) * SymbExpr::from_u32(16777216),
                                var(525)
                                    + var(526) * SymbExpr::from_u32(256)
                                    + var(527) * SymbExpr::from_u32(65536)
                                    + var(528) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 10
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(529)
                                    + var(530) * SymbExpr::from_u32(256)
                                    + var(531) * SymbExpr::from_u32(65536)
                                    + var(532) * SymbExpr::from_u32(16777216),
                                var(533)
                                    + var(534) * SymbExpr::from_u32(256)
                                    + var(535) * SymbExpr::from_u32(65536)
                                    + var(536) * SymbExpr::from_u32(16777216),
                                var(537)
                                    + var(538) * SymbExpr::from_u32(256)
                                    + var(539) * SymbExpr::from_u32(65536)
                                    + var(540) * SymbExpr::from_u32(16777216),
                                var(541)
                                    + var(542) * SymbExpr::from_u32(256)
                                    + var(543) * SymbExpr::from_u32(65536)
                                    + var(544) * SymbExpr::from_u32(16777216),
                                var(545)
                                    + var(546) * SymbExpr::from_u32(256)
                                    + var(547) * SymbExpr::from_u32(65536)
                                    + var(548) * SymbExpr::from_u32(16777216),
                                var(549)
                                    + var(550) * SymbExpr::from_u32(256)
                                    + var(551) * SymbExpr::from_u32(65536)
                                    + var(552) * SymbExpr::from_u32(16777216),
                                var(553)
                                    + var(554) * SymbExpr::from_u32(256)
                                    + var(555) * SymbExpr::from_u32(65536)
                                    + var(556) * SymbExpr::from_u32(16777216),
                                var(557)
                                    + var(558) * SymbExpr::from_u32(256)
                                    + var(559) * SymbExpr::from_u32(65536)
                                    + var(560) * SymbExpr::from_u32(16777216),
                                var(561)
                                    + var(562) * SymbExpr::from_u32(256)
                                    + var(563) * SymbExpr::from_u32(65536)
                                    + var(564) * SymbExpr::from_u32(16777216),
                                var(565)
                                    + var(566) * SymbExpr::from_u32(256)
                                    + var(567) * SymbExpr::from_u32(65536)
                                    + var(568) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 11
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(569)
                                    + var(570) * SymbExpr::from_u32(256)
                                    + var(571) * SymbExpr::from_u32(65536)
                                    + var(572) * SymbExpr::from_u32(16777216),
                                var(573)
                                    + var(574) * SymbExpr::from_u32(256)
                                    + var(575) * SymbExpr::from_u32(65536)
                                    + var(576) * SymbExpr::from_u32(16777216),
                                var(577)
                                    + var(578) * SymbExpr::from_u32(256)
                                    + var(579) * SymbExpr::from_u32(65536)
                                    + var(580) * SymbExpr::from_u32(16777216),
                                var(581)
                                    + var(582) * SymbExpr::from_u32(256)
                                    + var(583) * SymbExpr::from_u32(65536)
                                    + var(584) * SymbExpr::from_u32(16777216),
                                var(585)
                                    + var(586) * SymbExpr::from_u32(256)
                                    + var(587) * SymbExpr::from_u32(65536)
                                    + var(588) * SymbExpr::from_u32(16777216),
                                var(589)
                                    + var(590) * SymbExpr::from_u32(256)
                                    + var(591) * SymbExpr::from_u32(65536)
                                    + var(592) * SymbExpr::from_u32(16777216),
                                var(593)
                                    + var(594) * SymbExpr::from_u32(256)
                                    + var(595) * SymbExpr::from_u32(65536)
                                    + var(596) * SymbExpr::from_u32(16777216),
                                var(597)
                                    + var(598) * SymbExpr::from_u32(256)
                                    + var(599) * SymbExpr::from_u32(65536)
                                    + var(600) * SymbExpr::from_u32(16777216),
                                var(601)
                                    + var(602) * SymbExpr::from_u32(256)
                                    + var(603) * SymbExpr::from_u32(65536)
                                    + var(604) * SymbExpr::from_u32(16777216),
                                var(605)
                                    + var(606) * SymbExpr::from_u32(256)
                                    + var(607) * SymbExpr::from_u32(65536)
                                    + var(608) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 12
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(609)
                                    + var(610) * SymbExpr::from_u32(256)
                                    + var(611) * SymbExpr::from_u32(65536)
                                    + var(612) * SymbExpr::from_u32(16777216),
                                var(613)
                                    + var(614) * SymbExpr::from_u32(256)
                                    + var(615) * SymbExpr::from_u32(65536)
                                    + var(616) * SymbExpr::from_u32(16777216),
                                var(617)
                                    + var(618) * SymbExpr::from_u32(256)
                                    + var(619) * SymbExpr::from_u32(65536)
                                    + var(620) * SymbExpr::from_u32(16777216),
                                var(621)
                                    + var(622) * SymbExpr::from_u32(256)
                                    + var(623) * SymbExpr::from_u32(65536)
                                    + var(624) * SymbExpr::from_u32(16777216),
                                var(625)
                                    + var(626) * SymbExpr::from_u32(256)
                                    + var(627) * SymbExpr::from_u32(65536)
                                    + var(628) * SymbExpr::from_u32(16777216),
                                var(629)
                                    + var(630) * SymbExpr::from_u32(256)
                                    + var(631) * SymbExpr::from_u32(65536)
                                    + var(632) * SymbExpr::from_u32(16777216),
                                var(633)
                                    + var(634) * SymbExpr::from_u32(256)
                                    + var(635) * SymbExpr::from_u32(65536)
                                    + var(636) * SymbExpr::from_u32(16777216),
                                var(637)
                                    + var(638) * SymbExpr::from_u32(256)
                                    + var(639) * SymbExpr::from_u32(65536)
                                    + var(640) * SymbExpr::from_u32(16777216),
                                var(641)
                                    + var(642) * SymbExpr::from_u32(256)
                                    + var(643) * SymbExpr::from_u32(65536)
                                    + var(644) * SymbExpr::from_u32(16777216),
                                var(645)
                                    + var(646) * SymbExpr::from_u32(256)
                                    + var(647) * SymbExpr::from_u32(65536)
                                    + var(648) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 13
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(649)
                                    + var(650) * SymbExpr::from_u32(256)
                                    + var(651) * SymbExpr::from_u32(65536)
                                    + var(652) * SymbExpr::from_u32(16777216),
                                var(653)
                                    + var(654) * SymbExpr::from_u32(256)
                                    + var(655) * SymbExpr::from_u32(65536)
                                    + var(656) * SymbExpr::from_u32(16777216),
                                var(657)
                                    + var(658) * SymbExpr::from_u32(256)
                                    + var(659) * SymbExpr::from_u32(65536)
                                    + var(660) * SymbExpr::from_u32(16777216),
                                var(661)
                                    + var(662) * SymbExpr::from_u32(256)
                                    + var(663) * SymbExpr::from_u32(65536)
                                    + var(664) * SymbExpr::from_u32(16777216),
                                var(665)
                                    + var(666) * SymbExpr::from_u32(256)
                                    + var(667) * SymbExpr::from_u32(65536)
                                    + var(668) * SymbExpr::from_u32(16777216),
                                var(669)
                                    + var(670) * SymbExpr::from_u32(256)
                                    + var(671) * SymbExpr::from_u32(65536)
                                    + var(672) * SymbExpr::from_u32(16777216),
                                var(673)
                                    + var(674) * SymbExpr::from_u32(256)
                                    + var(675) * SymbExpr::from_u32(65536)
                                    + var(676) * SymbExpr::from_u32(16777216),
                                var(677)
                                    + var(678) * SymbExpr::from_u32(256)
                                    + var(679) * SymbExpr::from_u32(65536)
                                    + var(680) * SymbExpr::from_u32(16777216),
                                var(681)
                                    + var(682) * SymbExpr::from_u32(256)
                                    + var(683) * SymbExpr::from_u32(65536)
                                    + var(684) * SymbExpr::from_u32(16777216),
                                var(685)
                                    + var(686) * SymbExpr::from_u32(256)
                                    + var(687) * SymbExpr::from_u32(65536)
                                    + var(688) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 14
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(689)
                                    + var(690) * SymbExpr::from_u32(256)
                                    + var(691) * SymbExpr::from_u32(65536)
                                    + var(692) * SymbExpr::from_u32(16777216),
                                var(693)
                                    + var(694) * SymbExpr::from_u32(256)
                                    + var(695) * SymbExpr::from_u32(65536)
                                    + var(696) * SymbExpr::from_u32(16777216),
                                var(697)
                                    + var(698) * SymbExpr::from_u32(256)
                                    + var(699) * SymbExpr::from_u32(65536)
                                    + var(700) * SymbExpr::from_u32(16777216),
                                var(701)
                                    + var(702) * SymbExpr::from_u32(256)
                                    + var(703) * SymbExpr::from_u32(65536)
                                    + var(704) * SymbExpr::from_u32(16777216),
                                var(705)
                                    + var(706) * SymbExpr::from_u32(256)
                                    + var(707) * SymbExpr::from_u32(65536)
                                    + var(708) * SymbExpr::from_u32(16777216),
                                var(709)
                                    + var(710) * SymbExpr::from_u32(256)
                                    + var(711) * SymbExpr::from_u32(65536)
                                    + var(712) * SymbExpr::from_u32(16777216),
                                var(713)
                                    + var(714) * SymbExpr::from_u32(256)
                                    + var(715) * SymbExpr::from_u32(65536)
                                    + var(716) * SymbExpr::from_u32(16777216),
                                var(717)
                                    + var(718) * SymbExpr::from_u32(256)
                                    + var(719) * SymbExpr::from_u32(65536)
                                    + var(720) * SymbExpr::from_u32(16777216),
                                var(721)
                                    + var(722) * SymbExpr::from_u32(256)
                                    + var(723) * SymbExpr::from_u32(65536)
                                    + var(724) * SymbExpr::from_u32(16777216),
                                var(725)
                                    + var(726) * SymbExpr::from_u32(256)
                                    + var(727) * SymbExpr::from_u32(65536)
                                    + var(728) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 15
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(729)
                                    + var(730) * SymbExpr::from_u32(256)
                                    + var(731) * SymbExpr::from_u32(65536)
                                    + var(732) * SymbExpr::from_u32(16777216),
                                var(733)
                                    + var(734) * SymbExpr::from_u32(256)
                                    + var(735) * SymbExpr::from_u32(65536)
                                    + var(736) * SymbExpr::from_u32(16777216),
                                var(737)
                                    + var(738) * SymbExpr::from_u32(256)
                                    + var(739) * SymbExpr::from_u32(65536)
                                    + var(740) * SymbExpr::from_u32(16777216),
                                var(741)
                                    + var(742) * SymbExpr::from_u32(256)
                                    + var(743) * SymbExpr::from_u32(65536)
                                    + var(744) * SymbExpr::from_u32(16777216),
                                var(745)
                                    + var(746) * SymbExpr::from_u32(256)
                                    + var(747) * SymbExpr::from_u32(65536)
                                    + var(748) * SymbExpr::from_u32(16777216),
                                var(749)
                                    + var(750) * SymbExpr::from_u32(256)
                                    + var(751) * SymbExpr::from_u32(65536)
                                    + var(752) * SymbExpr::from_u32(16777216),
                                var(753)
                                    + var(754) * SymbExpr::from_u32(256)
                                    + var(755) * SymbExpr::from_u32(65536)
                                    + var(756) * SymbExpr::from_u32(16777216),
                                var(757)
                                    + var(758) * SymbExpr::from_u32(256)
                                    + var(759) * SymbExpr::from_u32(65536)
                                    + var(760) * SymbExpr::from_u32(16777216),
                                var(761)
                                    + var(762) * SymbExpr::from_u32(256)
                                    + var(763) * SymbExpr::from_u32(65536)
                                    + var(764) * SymbExpr::from_u32(16777216),
                                var(765)
                                    + var(766) * SymbExpr::from_u32(256)
                                    + var(767) * SymbExpr::from_u32(65536)
                                    + var(768) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 16
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(769)
                                    + var(770) * SymbExpr::from_u32(256)
                                    + var(771) * SymbExpr::from_u32(65536)
                                    + var(772) * SymbExpr::from_u32(16777216),
                                var(773)
                                    + var(774) * SymbExpr::from_u32(256)
                                    + var(775) * SymbExpr::from_u32(65536)
                                    + var(776) * SymbExpr::from_u32(16777216),
                                var(777)
                                    + var(778) * SymbExpr::from_u32(256)
                                    + var(779) * SymbExpr::from_u32(65536)
                                    + var(780) * SymbExpr::from_u32(16777216),
                                var(781)
                                    + var(782) * SymbExpr::from_u32(256)
                                    + var(783) * SymbExpr::from_u32(65536)
                                    + var(784) * SymbExpr::from_u32(16777216),
                                var(785)
                                    + var(786) * SymbExpr::from_u32(256)
                                    + var(787) * SymbExpr::from_u32(65536)
                                    + var(788) * SymbExpr::from_u32(16777216),
                                var(789)
                                    + var(790) * SymbExpr::from_u32(256)
                                    + var(791) * SymbExpr::from_u32(65536)
                                    + var(792) * SymbExpr::from_u32(16777216),
                                var(793)
                                    + var(794) * SymbExpr::from_u32(256)
                                    + var(795) * SymbExpr::from_u32(65536)
                                    + var(796) * SymbExpr::from_u32(16777216),
                                var(797)
                                    + var(798) * SymbExpr::from_u32(256)
                                    + var(799) * SymbExpr::from_u32(65536)
                                    + var(800) * SymbExpr::from_u32(16777216),
                                var(801)
                                    + var(802) * SymbExpr::from_u32(256)
                                    + var(803) * SymbExpr::from_u32(65536)
                                    + var(804) * SymbExpr::from_u32(16777216),
                                var(805)
                                    + var(806) * SymbExpr::from_u32(256)
                                    + var(807) * SymbExpr::from_u32(65536)
                                    + var(808) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 17
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(809)
                                    + var(810) * SymbExpr::from_u32(256)
                                    + var(811) * SymbExpr::from_u32(65536)
                                    + var(812) * SymbExpr::from_u32(16777216),
                                var(813)
                                    + var(814) * SymbExpr::from_u32(256)
                                    + var(815) * SymbExpr::from_u32(65536)
                                    + var(816) * SymbExpr::from_u32(16777216),
                                var(817)
                                    + var(818) * SymbExpr::from_u32(256)
                                    + var(819) * SymbExpr::from_u32(65536)
                                    + var(820) * SymbExpr::from_u32(16777216),
                                var(821)
                                    + var(822) * SymbExpr::from_u32(256)
                                    + var(823) * SymbExpr::from_u32(65536)
                                    + var(824) * SymbExpr::from_u32(16777216),
                                var(825)
                                    + var(826) * SymbExpr::from_u32(256)
                                    + var(827) * SymbExpr::from_u32(65536)
                                    + var(828) * SymbExpr::from_u32(16777216),
                                var(829)
                                    + var(830) * SymbExpr::from_u32(256)
                                    + var(831) * SymbExpr::from_u32(65536)
                                    + var(832) * SymbExpr::from_u32(16777216),
                                var(833)
                                    + var(834) * SymbExpr::from_u32(256)
                                    + var(835) * SymbExpr::from_u32(65536)
                                    + var(836) * SymbExpr::from_u32(16777216),
                                var(837)
                                    + var(838) * SymbExpr::from_u32(256)
                                    + var(839) * SymbExpr::from_u32(65536)
                                    + var(840) * SymbExpr::from_u32(16777216),
                                var(841)
                                    + var(842) * SymbExpr::from_u32(256)
                                    + var(843) * SymbExpr::from_u32(65536)
                                    + var(844) * SymbExpr::from_u32(16777216),
                                var(845)
                                    + var(846) * SymbExpr::from_u32(256)
                                    + var(847) * SymbExpr::from_u32(65536)
                                    + var(848) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 18
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(849)
                                    + var(850) * SymbExpr::from_u32(256)
                                    + var(851) * SymbExpr::from_u32(65536)
                                    + var(852) * SymbExpr::from_u32(16777216),
                                var(853)
                                    + var(854) * SymbExpr::from_u32(256)
                                    + var(855) * SymbExpr::from_u32(65536)
                                    + var(856) * SymbExpr::from_u32(16777216),
                                var(857)
                                    + var(858) * SymbExpr::from_u32(256)
                                    + var(859) * SymbExpr::from_u32(65536)
                                    + var(860) * SymbExpr::from_u32(16777216),
                                var(861)
                                    + var(862) * SymbExpr::from_u32(256)
                                    + var(863) * SymbExpr::from_u32(65536)
                                    + var(864) * SymbExpr::from_u32(16777216),
                                var(865)
                                    + var(866) * SymbExpr::from_u32(256)
                                    + var(867) * SymbExpr::from_u32(65536)
                                    + var(868) * SymbExpr::from_u32(16777216),
                                var(869)
                                    + var(870) * SymbExpr::from_u32(256)
                                    + var(871) * SymbExpr::from_u32(65536)
                                    + var(872) * SymbExpr::from_u32(16777216),
                                var(873)
                                    + var(874) * SymbExpr::from_u32(256)
                                    + var(875) * SymbExpr::from_u32(65536)
                                    + var(876) * SymbExpr::from_u32(16777216),
                                var(877)
                                    + var(878) * SymbExpr::from_u32(256)
                                    + var(879) * SymbExpr::from_u32(65536)
                                    + var(880) * SymbExpr::from_u32(16777216),
                                var(881)
                                    + var(882) * SymbExpr::from_u32(256)
                                    + var(883) * SymbExpr::from_u32(65536)
                                    + var(884) * SymbExpr::from_u32(16777216),
                                var(885)
                                    + var(886) * SymbExpr::from_u32(256)
                                    + var(887) * SymbExpr::from_u32(65536)
                                    + var(888) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 19
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(889)
                                    + var(890) * SymbExpr::from_u32(256)
                                    + var(891) * SymbExpr::from_u32(65536)
                                    + var(892) * SymbExpr::from_u32(16777216),
                                var(893)
                                    + var(894) * SymbExpr::from_u32(256)
                                    + var(895) * SymbExpr::from_u32(65536)
                                    + var(896) * SymbExpr::from_u32(16777216),
                                var(897)
                                    + var(898) * SymbExpr::from_u32(256)
                                    + var(899) * SymbExpr::from_u32(65536)
                                    + var(900) * SymbExpr::from_u32(16777216),
                                var(901)
                                    + var(902) * SymbExpr::from_u32(256)
                                    + var(903) * SymbExpr::from_u32(65536)
                                    + var(904) * SymbExpr::from_u32(16777216),
                                var(905)
                                    + var(906) * SymbExpr::from_u32(256)
                                    + var(907) * SymbExpr::from_u32(65536)
                                    + var(908) * SymbExpr::from_u32(16777216),
                                var(909)
                                    + var(910) * SymbExpr::from_u32(256)
                                    + var(911) * SymbExpr::from_u32(65536)
                                    + var(912) * SymbExpr::from_u32(16777216),
                                var(913)
                                    + var(914) * SymbExpr::from_u32(256)
                                    + var(915) * SymbExpr::from_u32(65536)
                                    + var(916) * SymbExpr::from_u32(16777216),
                                var(917)
                                    + var(918) * SymbExpr::from_u32(256)
                                    + var(919) * SymbExpr::from_u32(65536)
                                    + var(920) * SymbExpr::from_u32(16777216),
                                var(921)
                                    + var(922) * SymbExpr::from_u32(256)
                                    + var(923) * SymbExpr::from_u32(65536)
                                    + var(924) * SymbExpr::from_u32(16777216),
                                var(925)
                                    + var(926) * SymbExpr::from_u32(256)
                                    + var(927) * SymbExpr::from_u32(65536)
                                    + var(928) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 20
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(929)
                                    + var(930) * SymbExpr::from_u32(256)
                                    + var(931) * SymbExpr::from_u32(65536)
                                    + var(932) * SymbExpr::from_u32(16777216),
                                var(933)
                                    + var(934) * SymbExpr::from_u32(256)
                                    + var(935) * SymbExpr::from_u32(65536)
                                    + var(936) * SymbExpr::from_u32(16777216),
                                var(937)
                                    + var(938) * SymbExpr::from_u32(256)
                                    + var(939) * SymbExpr::from_u32(65536)
                                    + var(940) * SymbExpr::from_u32(16777216),
                                var(941)
                                    + var(942) * SymbExpr::from_u32(256)
                                    + var(943) * SymbExpr::from_u32(65536)
                                    + var(944) * SymbExpr::from_u32(16777216),
                                var(945)
                                    + var(946) * SymbExpr::from_u32(256)
                                    + var(947) * SymbExpr::from_u32(65536)
                                    + var(948) * SymbExpr::from_u32(16777216),
                                var(949)
                                    + var(950) * SymbExpr::from_u32(256)
                                    + var(951) * SymbExpr::from_u32(65536)
                                    + var(952) * SymbExpr::from_u32(16777216),
                                var(953)
                                    + var(954) * SymbExpr::from_u32(256)
                                    + var(955) * SymbExpr::from_u32(65536)
                                    + var(956) * SymbExpr::from_u32(16777216),
                                var(957)
                                    + var(958) * SymbExpr::from_u32(256)
                                    + var(959) * SymbExpr::from_u32(65536)
                                    + var(960) * SymbExpr::from_u32(16777216),
                                var(961)
                                    + var(962) * SymbExpr::from_u32(256)
                                    + var(963) * SymbExpr::from_u32(65536)
                                    + var(964) * SymbExpr::from_u32(16777216),
                                var(965)
                                    + var(966) * SymbExpr::from_u32(256)
                                    + var(967) * SymbExpr::from_u32(65536)
                                    + var(968) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 21
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(969)
                                    + var(970) * SymbExpr::from_u32(256)
                                    + var(971) * SymbExpr::from_u32(65536)
                                    + var(972) * SymbExpr::from_u32(16777216),
                                var(973)
                                    + var(974) * SymbExpr::from_u32(256)
                                    + var(975) * SymbExpr::from_u32(65536)
                                    + var(976) * SymbExpr::from_u32(16777216),
                                var(977)
                                    + var(978) * SymbExpr::from_u32(256)
                                    + var(979) * SymbExpr::from_u32(65536)
                                    + var(980) * SymbExpr::from_u32(16777216),
                                var(981)
                                    + var(982) * SymbExpr::from_u32(256)
                                    + var(983) * SymbExpr::from_u32(65536)
                                    + var(984) * SymbExpr::from_u32(16777216),
                                var(985)
                                    + var(986) * SymbExpr::from_u32(256)
                                    + var(987) * SymbExpr::from_u32(65536)
                                    + var(988) * SymbExpr::from_u32(16777216),
                                var(989)
                                    + var(990) * SymbExpr::from_u32(256)
                                    + var(991) * SymbExpr::from_u32(65536)
                                    + var(992) * SymbExpr::from_u32(16777216),
                                var(993)
                                    + var(994) * SymbExpr::from_u32(256)
                                    + var(995) * SymbExpr::from_u32(65536)
                                    + var(996) * SymbExpr::from_u32(16777216),
                                var(997)
                                    + var(998) * SymbExpr::from_u32(256)
                                    + var(999) * SymbExpr::from_u32(65536)
                                    + var(1000) * SymbExpr::from_u32(16777216),
                                var(1001)
                                    + var(1002) * SymbExpr::from_u32(256)
                                    + var(1003) * SymbExpr::from_u32(65536)
                                    + var(1004) * SymbExpr::from_u32(16777216),
                                var(1005)
                                    + var(1006) * SymbExpr::from_u32(256)
                                    + var(1007) * SymbExpr::from_u32(65536)
                                    + var(1008) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 22
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1009)
                                    + var(1010) * SymbExpr::from_u32(256)
                                    + var(1011) * SymbExpr::from_u32(65536)
                                    + var(1012) * SymbExpr::from_u32(16777216),
                                var(1013)
                                    + var(1014) * SymbExpr::from_u32(256)
                                    + var(1015) * SymbExpr::from_u32(65536)
                                    + var(1016) * SymbExpr::from_u32(16777216),
                                var(1017)
                                    + var(1018) * SymbExpr::from_u32(256)
                                    + var(1019) * SymbExpr::from_u32(65536)
                                    + var(1020) * SymbExpr::from_u32(16777216),
                                var(1021)
                                    + var(1022) * SymbExpr::from_u32(256)
                                    + var(1023) * SymbExpr::from_u32(65536)
                                    + var(1024) * SymbExpr::from_u32(16777216),
                                var(1025)
                                    + var(1026) * SymbExpr::from_u32(256)
                                    + var(1027) * SymbExpr::from_u32(65536)
                                    + var(1028) * SymbExpr::from_u32(16777216),
                                var(1029)
                                    + var(1030) * SymbExpr::from_u32(256)
                                    + var(1031) * SymbExpr::from_u32(65536)
                                    + var(1032) * SymbExpr::from_u32(16777216),
                                var(1033)
                                    + var(1034) * SymbExpr::from_u32(256)
                                    + var(1035) * SymbExpr::from_u32(65536)
                                    + var(1036) * SymbExpr::from_u32(16777216),
                                var(1037)
                                    + var(1038) * SymbExpr::from_u32(256)
                                    + var(1039) * SymbExpr::from_u32(65536)
                                    + var(1040) * SymbExpr::from_u32(16777216),
                                var(1041)
                                    + var(1042) * SymbExpr::from_u32(256)
                                    + var(1043) * SymbExpr::from_u32(65536)
                                    + var(1044) * SymbExpr::from_u32(16777216),
                                var(1045)
                                    + var(1046) * SymbExpr::from_u32(256)
                                    + var(1047) * SymbExpr::from_u32(65536)
                                    + var(1048) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 23
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1049)
                                    + var(1050) * SymbExpr::from_u32(256)
                                    + var(1051) * SymbExpr::from_u32(65536)
                                    + var(1052) * SymbExpr::from_u32(16777216),
                                var(1053)
                                    + var(1054) * SymbExpr::from_u32(256)
                                    + var(1055) * SymbExpr::from_u32(65536)
                                    + var(1056) * SymbExpr::from_u32(16777216),
                                var(1057)
                                    + var(1058) * SymbExpr::from_u32(256)
                                    + var(1059) * SymbExpr::from_u32(65536)
                                    + var(1060) * SymbExpr::from_u32(16777216),
                                var(1061)
                                    + var(1062) * SymbExpr::from_u32(256)
                                    + var(1063) * SymbExpr::from_u32(65536)
                                    + var(1064) * SymbExpr::from_u32(16777216),
                                var(1065)
                                    + var(1066) * SymbExpr::from_u32(256)
                                    + var(1067) * SymbExpr::from_u32(65536)
                                    + var(1068) * SymbExpr::from_u32(16777216),
                                var(1069)
                                    + var(1070) * SymbExpr::from_u32(256)
                                    + var(1071) * SymbExpr::from_u32(65536)
                                    + var(1072) * SymbExpr::from_u32(16777216),
                                var(1073)
                                    + var(1074) * SymbExpr::from_u32(256)
                                    + var(1075) * SymbExpr::from_u32(65536)
                                    + var(1076) * SymbExpr::from_u32(16777216),
                                var(1077)
                                    + var(1078) * SymbExpr::from_u32(256)
                                    + var(1079) * SymbExpr::from_u32(65536)
                                    + var(1080) * SymbExpr::from_u32(16777216),
                                var(1081)
                                    + var(1082) * SymbExpr::from_u32(256)
                                    + var(1083) * SymbExpr::from_u32(65536)
                                    + var(1084) * SymbExpr::from_u32(16777216),
                                var(1085)
                                    + var(1086) * SymbExpr::from_u32(256)
                                    + var(1087) * SymbExpr::from_u32(65536)
                                    + var(1088) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 24
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1089)
                                    + var(1090) * SymbExpr::from_u32(256)
                                    + var(1091) * SymbExpr::from_u32(65536)
                                    + var(1092) * SymbExpr::from_u32(16777216),
                                var(1093)
                                    + var(1094) * SymbExpr::from_u32(256)
                                    + var(1095) * SymbExpr::from_u32(65536)
                                    + var(1096) * SymbExpr::from_u32(16777216),
                                var(1097)
                                    + var(1098) * SymbExpr::from_u32(256)
                                    + var(1099) * SymbExpr::from_u32(65536)
                                    + var(1100) * SymbExpr::from_u32(16777216),
                                var(1101)
                                    + var(1102) * SymbExpr::from_u32(256)
                                    + var(1103) * SymbExpr::from_u32(65536)
                                    + var(1104) * SymbExpr::from_u32(16777216),
                                var(1105)
                                    + var(1106) * SymbExpr::from_u32(256)
                                    + var(1107) * SymbExpr::from_u32(65536)
                                    + var(1108) * SymbExpr::from_u32(16777216),
                                var(1109)
                                    + var(1110) * SymbExpr::from_u32(256)
                                    + var(1111) * SymbExpr::from_u32(65536)
                                    + var(1112) * SymbExpr::from_u32(16777216),
                                var(1113)
                                    + var(1114) * SymbExpr::from_u32(256)
                                    + var(1115) * SymbExpr::from_u32(65536)
                                    + var(1116) * SymbExpr::from_u32(16777216),
                                var(1117)
                                    + var(1118) * SymbExpr::from_u32(256)
                                    + var(1119) * SymbExpr::from_u32(65536)
                                    + var(1120) * SymbExpr::from_u32(16777216),
                                var(1121)
                                    + var(1122) * SymbExpr::from_u32(256)
                                    + var(1123) * SymbExpr::from_u32(65536)
                                    + var(1124) * SymbExpr::from_u32(16777216),
                                var(1125)
                                    + var(1126) * SymbExpr::from_u32(256)
                                    + var(1127) * SymbExpr::from_u32(65536)
                                    + var(1128) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 25
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1129)
                                    + var(1130) * SymbExpr::from_u32(256)
                                    + var(1131) * SymbExpr::from_u32(65536)
                                    + var(1132) * SymbExpr::from_u32(16777216),
                                var(1133)
                                    + var(1134) * SymbExpr::from_u32(256)
                                    + var(1135) * SymbExpr::from_u32(65536)
                                    + var(1136) * SymbExpr::from_u32(16777216),
                                var(1137)
                                    + var(1138) * SymbExpr::from_u32(256)
                                    + var(1139) * SymbExpr::from_u32(65536)
                                    + var(1140) * SymbExpr::from_u32(16777216),
                                var(1141)
                                    + var(1142) * SymbExpr::from_u32(256)
                                    + var(1143) * SymbExpr::from_u32(65536)
                                    + var(1144) * SymbExpr::from_u32(16777216),
                                var(1145)
                                    + var(1146) * SymbExpr::from_u32(256)
                                    + var(1147) * SymbExpr::from_u32(65536)
                                    + var(1148) * SymbExpr::from_u32(16777216),
                                var(1149)
                                    + var(1150) * SymbExpr::from_u32(256)
                                    + var(1151) * SymbExpr::from_u32(65536)
                                    + var(1152) * SymbExpr::from_u32(16777216),
                                var(1153)
                                    + var(1154) * SymbExpr::from_u32(256)
                                    + var(1155) * SymbExpr::from_u32(65536)
                                    + var(1156) * SymbExpr::from_u32(16777216),
                                var(1157)
                                    + var(1158) * SymbExpr::from_u32(256)
                                    + var(1159) * SymbExpr::from_u32(65536)
                                    + var(1160) * SymbExpr::from_u32(16777216),
                                var(1161)
                                    + var(1162) * SymbExpr::from_u32(256)
                                    + var(1163) * SymbExpr::from_u32(65536)
                                    + var(1164) * SymbExpr::from_u32(16777216),
                                var(1165)
                                    + var(1166) * SymbExpr::from_u32(256)
                                    + var(1167) * SymbExpr::from_u32(65536)
                                    + var(1168) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 26
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1169)
                                    + var(1170) * SymbExpr::from_u32(256)
                                    + var(1171) * SymbExpr::from_u32(65536)
                                    + var(1172) * SymbExpr::from_u32(16777216),
                                var(1173)
                                    + var(1174) * SymbExpr::from_u32(256)
                                    + var(1175) * SymbExpr::from_u32(65536)
                                    + var(1176) * SymbExpr::from_u32(16777216),
                                var(1177)
                                    + var(1178) * SymbExpr::from_u32(256)
                                    + var(1179) * SymbExpr::from_u32(65536)
                                    + var(1180) * SymbExpr::from_u32(16777216),
                                var(1181)
                                    + var(1182) * SymbExpr::from_u32(256)
                                    + var(1183) * SymbExpr::from_u32(65536)
                                    + var(1184) * SymbExpr::from_u32(16777216),
                                var(1185)
                                    + var(1186) * SymbExpr::from_u32(256)
                                    + var(1187) * SymbExpr::from_u32(65536)
                                    + var(1188) * SymbExpr::from_u32(16777216),
                                var(1189)
                                    + var(1190) * SymbExpr::from_u32(256)
                                    + var(1191) * SymbExpr::from_u32(65536)
                                    + var(1192) * SymbExpr::from_u32(16777216),
                                var(1193)
                                    + var(1194) * SymbExpr::from_u32(256)
                                    + var(1195) * SymbExpr::from_u32(65536)
                                    + var(1196) * SymbExpr::from_u32(16777216),
                                var(1197)
                                    + var(1198) * SymbExpr::from_u32(256)
                                    + var(1199) * SymbExpr::from_u32(65536)
                                    + var(1200) * SymbExpr::from_u32(16777216),
                                var(1201)
                                    + var(1202) * SymbExpr::from_u32(256)
                                    + var(1203) * SymbExpr::from_u32(65536)
                                    + var(1204) * SymbExpr::from_u32(16777216),
                                var(1205)
                                    + var(1206) * SymbExpr::from_u32(256)
                                    + var(1207) * SymbExpr::from_u32(65536)
                                    + var(1208) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 27
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1209)
                                    + var(1210) * SymbExpr::from_u32(256)
                                    + var(1211) * SymbExpr::from_u32(65536)
                                    + var(1212) * SymbExpr::from_u32(16777216),
                                var(1213)
                                    + var(1214) * SymbExpr::from_u32(256)
                                    + var(1215) * SymbExpr::from_u32(65536)
                                    + var(1216) * SymbExpr::from_u32(16777216),
                                var(1217)
                                    + var(1218) * SymbExpr::from_u32(256)
                                    + var(1219) * SymbExpr::from_u32(65536)
                                    + var(1220) * SymbExpr::from_u32(16777216),
                                var(1221)
                                    + var(1222) * SymbExpr::from_u32(256)
                                    + var(1223) * SymbExpr::from_u32(65536)
                                    + var(1224) * SymbExpr::from_u32(16777216),
                                var(1225)
                                    + var(1226) * SymbExpr::from_u32(256)
                                    + var(1227) * SymbExpr::from_u32(65536)
                                    + var(1228) * SymbExpr::from_u32(16777216),
                                var(1229)
                                    + var(1230) * SymbExpr::from_u32(256)
                                    + var(1231) * SymbExpr::from_u32(65536)
                                    + var(1232) * SymbExpr::from_u32(16777216),
                                var(1233)
                                    + var(1234) * SymbExpr::from_u32(256)
                                    + var(1235) * SymbExpr::from_u32(65536)
                                    + var(1236) * SymbExpr::from_u32(16777216),
                                var(1237)
                                    + var(1238) * SymbExpr::from_u32(256)
                                    + var(1239) * SymbExpr::from_u32(65536)
                                    + var(1240) * SymbExpr::from_u32(16777216),
                                var(1241)
                                    + var(1242) * SymbExpr::from_u32(256)
                                    + var(1243) * SymbExpr::from_u32(65536)
                                    + var(1244) * SymbExpr::from_u32(16777216),
                                var(1245)
                                    + var(1246) * SymbExpr::from_u32(256)
                                    + var(1247) * SymbExpr::from_u32(65536)
                                    + var(1248) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 28
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1249)
                                    + var(1250) * SymbExpr::from_u32(256)
                                    + var(1251) * SymbExpr::from_u32(65536)
                                    + var(1252) * SymbExpr::from_u32(16777216),
                                var(1253)
                                    + var(1254) * SymbExpr::from_u32(256)
                                    + var(1255) * SymbExpr::from_u32(65536)
                                    + var(1256) * SymbExpr::from_u32(16777216),
                                var(1257)
                                    + var(1258) * SymbExpr::from_u32(256)
                                    + var(1259) * SymbExpr::from_u32(65536)
                                    + var(1260) * SymbExpr::from_u32(16777216),
                                var(1261)
                                    + var(1262) * SymbExpr::from_u32(256)
                                    + var(1263) * SymbExpr::from_u32(65536)
                                    + var(1264) * SymbExpr::from_u32(16777216),
                                var(1265)
                                    + var(1266) * SymbExpr::from_u32(256)
                                    + var(1267) * SymbExpr::from_u32(65536)
                                    + var(1268) * SymbExpr::from_u32(16777216),
                                var(1269)
                                    + var(1270) * SymbExpr::from_u32(256)
                                    + var(1271) * SymbExpr::from_u32(65536)
                                    + var(1272) * SymbExpr::from_u32(16777216),
                                var(1273)
                                    + var(1274) * SymbExpr::from_u32(256)
                                    + var(1275) * SymbExpr::from_u32(65536)
                                    + var(1276) * SymbExpr::from_u32(16777216),
                                var(1277)
                                    + var(1278) * SymbExpr::from_u32(256)
                                    + var(1279) * SymbExpr::from_u32(65536)
                                    + var(1280) * SymbExpr::from_u32(16777216),
                                var(1281)
                                    + var(1282) * SymbExpr::from_u32(256)
                                    + var(1283) * SymbExpr::from_u32(65536)
                                    + var(1284) * SymbExpr::from_u32(16777216),
                                var(1285)
                                    + var(1286) * SymbExpr::from_u32(256)
                                    + var(1287) * SymbExpr::from_u32(65536)
                                    + var(1288) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 29
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1289)
                                    + var(1290) * SymbExpr::from_u32(256)
                                    + var(1291) * SymbExpr::from_u32(65536)
                                    + var(1292) * SymbExpr::from_u32(16777216),
                                var(1293)
                                    + var(1294) * SymbExpr::from_u32(256)
                                    + var(1295) * SymbExpr::from_u32(65536)
                                    + var(1296) * SymbExpr::from_u32(16777216),
                                var(1297)
                                    + var(1298) * SymbExpr::from_u32(256)
                                    + var(1299) * SymbExpr::from_u32(65536)
                                    + var(1300) * SymbExpr::from_u32(16777216),
                                var(1301)
                                    + var(1302) * SymbExpr::from_u32(256)
                                    + var(1303) * SymbExpr::from_u32(65536)
                                    + var(1304) * SymbExpr::from_u32(16777216),
                                var(1305)
                                    + var(1306) * SymbExpr::from_u32(256)
                                    + var(1307) * SymbExpr::from_u32(65536)
                                    + var(1308) * SymbExpr::from_u32(16777216),
                                var(1309)
                                    + var(1310) * SymbExpr::from_u32(256)
                                    + var(1311) * SymbExpr::from_u32(65536)
                                    + var(1312) * SymbExpr::from_u32(16777216),
                                var(1313)
                                    + var(1314) * SymbExpr::from_u32(256)
                                    + var(1315) * SymbExpr::from_u32(65536)
                                    + var(1316) * SymbExpr::from_u32(16777216),
                                var(1317)
                                    + var(1318) * SymbExpr::from_u32(256)
                                    + var(1319) * SymbExpr::from_u32(65536)
                                    + var(1320) * SymbExpr::from_u32(16777216),
                                var(1321)
                                    + var(1322) * SymbExpr::from_u32(256)
                                    + var(1323) * SymbExpr::from_u32(65536)
                                    + var(1324) * SymbExpr::from_u32(16777216),
                                var(1325)
                                    + var(1326) * SymbExpr::from_u32(256)
                                    + var(1327) * SymbExpr::from_u32(65536)
                                    + var(1328) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 30
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1329)
                                    + var(1330) * SymbExpr::from_u32(256)
                                    + var(1331) * SymbExpr::from_u32(65536)
                                    + var(1332) * SymbExpr::from_u32(16777216),
                                var(1333)
                                    + var(1334) * SymbExpr::from_u32(256)
                                    + var(1335) * SymbExpr::from_u32(65536)
                                    + var(1336) * SymbExpr::from_u32(16777216),
                                var(1337)
                                    + var(1338) * SymbExpr::from_u32(256)
                                    + var(1339) * SymbExpr::from_u32(65536)
                                    + var(1340) * SymbExpr::from_u32(16777216),
                                var(1341)
                                    + var(1342) * SymbExpr::from_u32(256)
                                    + var(1343) * SymbExpr::from_u32(65536)
                                    + var(1344) * SymbExpr::from_u32(16777216),
                                var(1345)
                                    + var(1346) * SymbExpr::from_u32(256)
                                    + var(1347) * SymbExpr::from_u32(65536)
                                    + var(1348) * SymbExpr::from_u32(16777216),
                                var(1349)
                                    + var(1350) * SymbExpr::from_u32(256)
                                    + var(1351) * SymbExpr::from_u32(65536)
                                    + var(1352) * SymbExpr::from_u32(16777216),
                                var(1353)
                                    + var(1354) * SymbExpr::from_u32(256)
                                    + var(1355) * SymbExpr::from_u32(65536)
                                    + var(1356) * SymbExpr::from_u32(16777216),
                                var(1357)
                                    + var(1358) * SymbExpr::from_u32(256)
                                    + var(1359) * SymbExpr::from_u32(65536)
                                    + var(1360) * SymbExpr::from_u32(16777216),
                                var(1361)
                                    + var(1362) * SymbExpr::from_u32(256)
                                    + var(1363) * SymbExpr::from_u32(65536)
                                    + var(1364) * SymbExpr::from_u32(16777216),
                                var(1365)
                                    + var(1366) * SymbExpr::from_u32(256)
                                    + var(1367) * SymbExpr::from_u32(65536)
                                    + var(1368) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 31
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1369)
                                    + var(1370) * SymbExpr::from_u32(256)
                                    + var(1371) * SymbExpr::from_u32(65536)
                                    + var(1372) * SymbExpr::from_u32(16777216),
                                var(1373)
                                    + var(1374) * SymbExpr::from_u32(256)
                                    + var(1375) * SymbExpr::from_u32(65536)
                                    + var(1376) * SymbExpr::from_u32(16777216),
                                var(1377)
                                    + var(1378) * SymbExpr::from_u32(256)
                                    + var(1379) * SymbExpr::from_u32(65536)
                                    + var(1380) * SymbExpr::from_u32(16777216),
                                var(1381)
                                    + var(1382) * SymbExpr::from_u32(256)
                                    + var(1383) * SymbExpr::from_u32(65536)
                                    + var(1384) * SymbExpr::from_u32(16777216),
                                var(1385)
                                    + var(1386) * SymbExpr::from_u32(256)
                                    + var(1387) * SymbExpr::from_u32(65536)
                                    + var(1388) * SymbExpr::from_u32(16777216),
                                var(1389)
                                    + var(1390) * SymbExpr::from_u32(256)
                                    + var(1391) * SymbExpr::from_u32(65536)
                                    + var(1392) * SymbExpr::from_u32(16777216),
                                var(1393)
                                    + var(1394) * SymbExpr::from_u32(256)
                                    + var(1395) * SymbExpr::from_u32(65536)
                                    + var(1396) * SymbExpr::from_u32(16777216),
                                var(1397)
                                    + var(1398) * SymbExpr::from_u32(256)
                                    + var(1399) * SymbExpr::from_u32(65536)
                                    + var(1400) * SymbExpr::from_u32(16777216),
                                var(1401)
                                    + var(1402) * SymbExpr::from_u32(256)
                                    + var(1403) * SymbExpr::from_u32(65536)
                                    + var(1404) * SymbExpr::from_u32(16777216),
                                var(1405)
                                    + var(1406) * SymbExpr::from_u32(256)
                                    + var(1407) * SymbExpr::from_u32(65536)
                                    + var(1408) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 32
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1409)
                                    + var(1410) * SymbExpr::from_u32(256)
                                    + var(1411) * SymbExpr::from_u32(65536)
                                    + var(1412) * SymbExpr::from_u32(16777216),
                                var(1413)
                                    + var(1414) * SymbExpr::from_u32(256)
                                    + var(1415) * SymbExpr::from_u32(65536)
                                    + var(1416) * SymbExpr::from_u32(16777216),
                                var(1417)
                                    + var(1418) * SymbExpr::from_u32(256)
                                    + var(1419) * SymbExpr::from_u32(65536)
                                    + var(1420) * SymbExpr::from_u32(16777216),
                                var(1421)
                                    + var(1422) * SymbExpr::from_u32(256)
                                    + var(1423) * SymbExpr::from_u32(65536)
                                    + var(1424) * SymbExpr::from_u32(16777216),
                                var(1425)
                                    + var(1426) * SymbExpr::from_u32(256)
                                    + var(1427) * SymbExpr::from_u32(65536)
                                    + var(1428) * SymbExpr::from_u32(16777216),
                                var(1429)
                                    + var(1430) * SymbExpr::from_u32(256)
                                    + var(1431) * SymbExpr::from_u32(65536)
                                    + var(1432) * SymbExpr::from_u32(16777216),
                                var(1433)
                                    + var(1434) * SymbExpr::from_u32(256)
                                    + var(1435) * SymbExpr::from_u32(65536)
                                    + var(1436) * SymbExpr::from_u32(16777216),
                                var(1437)
                                    + var(1438) * SymbExpr::from_u32(256)
                                    + var(1439) * SymbExpr::from_u32(65536)
                                    + var(1440) * SymbExpr::from_u32(16777216),
                                var(1441)
                                    + var(1442) * SymbExpr::from_u32(256)
                                    + var(1443) * SymbExpr::from_u32(65536)
                                    + var(1444) * SymbExpr::from_u32(16777216),
                                var(1445)
                                    + var(1446) * SymbExpr::from_u32(256)
                                    + var(1447) * SymbExpr::from_u32(65536)
                                    + var(1448) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 33
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1449)
                                    + var(1450) * SymbExpr::from_u32(256)
                                    + var(1451) * SymbExpr::from_u32(65536)
                                    + var(1452) * SymbExpr::from_u32(16777216),
                                var(1453)
                                    + var(1454) * SymbExpr::from_u32(256)
                                    + var(1455) * SymbExpr::from_u32(65536)
                                    + var(1456) * SymbExpr::from_u32(16777216),
                                var(1457)
                                    + var(1458) * SymbExpr::from_u32(256)
                                    + var(1459) * SymbExpr::from_u32(65536)
                                    + var(1460) * SymbExpr::from_u32(16777216),
                                var(1461)
                                    + var(1462) * SymbExpr::from_u32(256)
                                    + var(1463) * SymbExpr::from_u32(65536)
                                    + var(1464) * SymbExpr::from_u32(16777216),
                                var(1465)
                                    + var(1466) * SymbExpr::from_u32(256)
                                    + var(1467) * SymbExpr::from_u32(65536)
                                    + var(1468) * SymbExpr::from_u32(16777216),
                                var(1469)
                                    + var(1470) * SymbExpr::from_u32(256)
                                    + var(1471) * SymbExpr::from_u32(65536)
                                    + var(1472) * SymbExpr::from_u32(16777216),
                                var(1473)
                                    + var(1474) * SymbExpr::from_u32(256)
                                    + var(1475) * SymbExpr::from_u32(65536)
                                    + var(1476) * SymbExpr::from_u32(16777216),
                                var(1477)
                                    + var(1478) * SymbExpr::from_u32(256)
                                    + var(1479) * SymbExpr::from_u32(65536)
                                    + var(1480) * SymbExpr::from_u32(16777216),
                                var(1481)
                                    + var(1482) * SymbExpr::from_u32(256)
                                    + var(1483) * SymbExpr::from_u32(65536)
                                    + var(1484) * SymbExpr::from_u32(16777216),
                                var(1485)
                                    + var(1486) * SymbExpr::from_u32(256)
                                    + var(1487) * SymbExpr::from_u32(65536)
                                    + var(1488) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 34
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1489)
                                    + var(1490) * SymbExpr::from_u32(256)
                                    + var(1491) * SymbExpr::from_u32(65536)
                                    + var(1492) * SymbExpr::from_u32(16777216),
                                var(1493)
                                    + var(1494) * SymbExpr::from_u32(256)
                                    + var(1495) * SymbExpr::from_u32(65536)
                                    + var(1496) * SymbExpr::from_u32(16777216),
                                var(1497)
                                    + var(1498) * SymbExpr::from_u32(256)
                                    + var(1499) * SymbExpr::from_u32(65536)
                                    + var(1500) * SymbExpr::from_u32(16777216),
                                var(1501)
                                    + var(1502) * SymbExpr::from_u32(256)
                                    + var(1503) * SymbExpr::from_u32(65536)
                                    + var(1504) * SymbExpr::from_u32(16777216),
                                var(1505)
                                    + var(1506) * SymbExpr::from_u32(256)
                                    + var(1507) * SymbExpr::from_u32(65536)
                                    + var(1508) * SymbExpr::from_u32(16777216),
                                var(1509)
                                    + var(1510) * SymbExpr::from_u32(256)
                                    + var(1511) * SymbExpr::from_u32(65536)
                                    + var(1512) * SymbExpr::from_u32(16777216),
                                var(1513)
                                    + var(1514) * SymbExpr::from_u32(256)
                                    + var(1515) * SymbExpr::from_u32(65536)
                                    + var(1516) * SymbExpr::from_u32(16777216),
                                var(1517)
                                    + var(1518) * SymbExpr::from_u32(256)
                                    + var(1519) * SymbExpr::from_u32(65536)
                                    + var(1520) * SymbExpr::from_u32(16777216),
                                var(1521)
                                    + var(1522) * SymbExpr::from_u32(256)
                                    + var(1523) * SymbExpr::from_u32(65536)
                                    + var(1524) * SymbExpr::from_u32(16777216),
                                var(1525)
                                    + var(1526) * SymbExpr::from_u32(256)
                                    + var(1527) * SymbExpr::from_u32(65536)
                                    + var(1528) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 35
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1529)
                                    + var(1530) * SymbExpr::from_u32(256)
                                    + var(1531) * SymbExpr::from_u32(65536)
                                    + var(1532) * SymbExpr::from_u32(16777216),
                                var(1533)
                                    + var(1534) * SymbExpr::from_u32(256)
                                    + var(1535) * SymbExpr::from_u32(65536)
                                    + var(1536) * SymbExpr::from_u32(16777216),
                                var(1537)
                                    + var(1538) * SymbExpr::from_u32(256)
                                    + var(1539) * SymbExpr::from_u32(65536)
                                    + var(1540) * SymbExpr::from_u32(16777216),
                                var(1541)
                                    + var(1542) * SymbExpr::from_u32(256)
                                    + var(1543) * SymbExpr::from_u32(65536)
                                    + var(1544) * SymbExpr::from_u32(16777216),
                                var(1545)
                                    + var(1546) * SymbExpr::from_u32(256)
                                    + var(1547) * SymbExpr::from_u32(65536)
                                    + var(1548) * SymbExpr::from_u32(16777216),
                                var(1549)
                                    + var(1550) * SymbExpr::from_u32(256)
                                    + var(1551) * SymbExpr::from_u32(65536)
                                    + var(1552) * SymbExpr::from_u32(16777216),
                                var(1553)
                                    + var(1554) * SymbExpr::from_u32(256)
                                    + var(1555) * SymbExpr::from_u32(65536)
                                    + var(1556) * SymbExpr::from_u32(16777216),
                                var(1557)
                                    + var(1558) * SymbExpr::from_u32(256)
                                    + var(1559) * SymbExpr::from_u32(65536)
                                    + var(1560) * SymbExpr::from_u32(16777216),
                                var(1561)
                                    + var(1562) * SymbExpr::from_u32(256)
                                    + var(1563) * SymbExpr::from_u32(65536)
                                    + var(1564) * SymbExpr::from_u32(16777216),
                                var(1565)
                                    + var(1566) * SymbExpr::from_u32(256)
                                    + var(1567) * SymbExpr::from_u32(65536)
                                    + var(1568) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 36
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1569)
                                    + var(1570) * SymbExpr::from_u32(256)
                                    + var(1571) * SymbExpr::from_u32(65536)
                                    + var(1572) * SymbExpr::from_u32(16777216),
                                var(1573)
                                    + var(1574) * SymbExpr::from_u32(256)
                                    + var(1575) * SymbExpr::from_u32(65536)
                                    + var(1576) * SymbExpr::from_u32(16777216),
                                var(1577)
                                    + var(1578) * SymbExpr::from_u32(256)
                                    + var(1579) * SymbExpr::from_u32(65536)
                                    + var(1580) * SymbExpr::from_u32(16777216),
                                var(1581)
                                    + var(1582) * SymbExpr::from_u32(256)
                                    + var(1583) * SymbExpr::from_u32(65536)
                                    + var(1584) * SymbExpr::from_u32(16777216),
                                var(1585)
                                    + var(1586) * SymbExpr::from_u32(256)
                                    + var(1587) * SymbExpr::from_u32(65536)
                                    + var(1588) * SymbExpr::from_u32(16777216),
                                var(1589)
                                    + var(1590) * SymbExpr::from_u32(256)
                                    + var(1591) * SymbExpr::from_u32(65536)
                                    + var(1592) * SymbExpr::from_u32(16777216),
                                var(1593)
                                    + var(1594) * SymbExpr::from_u32(256)
                                    + var(1595) * SymbExpr::from_u32(65536)
                                    + var(1596) * SymbExpr::from_u32(16777216),
                                var(1597)
                                    + var(1598) * SymbExpr::from_u32(256)
                                    + var(1599) * SymbExpr::from_u32(65536)
                                    + var(1600) * SymbExpr::from_u32(16777216),
                                var(1601)
                                    + var(1602) * SymbExpr::from_u32(256)
                                    + var(1603) * SymbExpr::from_u32(65536)
                                    + var(1604) * SymbExpr::from_u32(16777216),
                                var(1605)
                                    + var(1606) * SymbExpr::from_u32(256)
                                    + var(1607) * SymbExpr::from_u32(65536)
                                    + var(1608) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 37
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1609)
                                    + var(1610) * SymbExpr::from_u32(256)
                                    + var(1611) * SymbExpr::from_u32(65536)
                                    + var(1612) * SymbExpr::from_u32(16777216),
                                var(1613)
                                    + var(1614) * SymbExpr::from_u32(256)
                                    + var(1615) * SymbExpr::from_u32(65536)
                                    + var(1616) * SymbExpr::from_u32(16777216),
                                var(1617)
                                    + var(1618) * SymbExpr::from_u32(256)
                                    + var(1619) * SymbExpr::from_u32(65536)
                                    + var(1620) * SymbExpr::from_u32(16777216),
                                var(1621)
                                    + var(1622) * SymbExpr::from_u32(256)
                                    + var(1623) * SymbExpr::from_u32(65536)
                                    + var(1624) * SymbExpr::from_u32(16777216),
                                var(1625)
                                    + var(1626) * SymbExpr::from_u32(256)
                                    + var(1627) * SymbExpr::from_u32(65536)
                                    + var(1628) * SymbExpr::from_u32(16777216),
                                var(1629)
                                    + var(1630) * SymbExpr::from_u32(256)
                                    + var(1631) * SymbExpr::from_u32(65536)
                                    + var(1632) * SymbExpr::from_u32(16777216),
                                var(1633)
                                    + var(1634) * SymbExpr::from_u32(256)
                                    + var(1635) * SymbExpr::from_u32(65536)
                                    + var(1636) * SymbExpr::from_u32(16777216),
                                var(1637)
                                    + var(1638) * SymbExpr::from_u32(256)
                                    + var(1639) * SymbExpr::from_u32(65536)
                                    + var(1640) * SymbExpr::from_u32(16777216),
                                var(1641)
                                    + var(1642) * SymbExpr::from_u32(256)
                                    + var(1643) * SymbExpr::from_u32(65536)
                                    + var(1644) * SymbExpr::from_u32(16777216),
                                var(1645)
                                    + var(1646) * SymbExpr::from_u32(256)
                                    + var(1647) * SymbExpr::from_u32(65536)
                                    + var(1648) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 38
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1649)
                                    + var(1650) * SymbExpr::from_u32(256)
                                    + var(1651) * SymbExpr::from_u32(65536)
                                    + var(1652) * SymbExpr::from_u32(16777216),
                                var(1653)
                                    + var(1654) * SymbExpr::from_u32(256)
                                    + var(1655) * SymbExpr::from_u32(65536)
                                    + var(1656) * SymbExpr::from_u32(16777216),
                                var(1657)
                                    + var(1658) * SymbExpr::from_u32(256)
                                    + var(1659) * SymbExpr::from_u32(65536)
                                    + var(1660) * SymbExpr::from_u32(16777216),
                                var(1661)
                                    + var(1662) * SymbExpr::from_u32(256)
                                    + var(1663) * SymbExpr::from_u32(65536)
                                    + var(1664) * SymbExpr::from_u32(16777216),
                                var(1665)
                                    + var(1666) * SymbExpr::from_u32(256)
                                    + var(1667) * SymbExpr::from_u32(65536)
                                    + var(1668) * SymbExpr::from_u32(16777216),
                                var(1669)
                                    + var(1670) * SymbExpr::from_u32(256)
                                    + var(1671) * SymbExpr::from_u32(65536)
                                    + var(1672) * SymbExpr::from_u32(16777216),
                                var(1673)
                                    + var(1674) * SymbExpr::from_u32(256)
                                    + var(1675) * SymbExpr::from_u32(65536)
                                    + var(1676) * SymbExpr::from_u32(16777216),
                                var(1677)
                                    + var(1678) * SymbExpr::from_u32(256)
                                    + var(1679) * SymbExpr::from_u32(65536)
                                    + var(1680) * SymbExpr::from_u32(16777216),
                                var(1681)
                                    + var(1682) * SymbExpr::from_u32(256)
                                    + var(1683) * SymbExpr::from_u32(65536)
                                    + var(1684) * SymbExpr::from_u32(16777216),
                                var(1685)
                                    + var(1686) * SymbExpr::from_u32(256)
                                    + var(1687) * SymbExpr::from_u32(65536)
                                    + var(1688) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 39
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1689)
                                    + var(1690) * SymbExpr::from_u32(256)
                                    + var(1691) * SymbExpr::from_u32(65536)
                                    + var(1692) * SymbExpr::from_u32(16777216),
                                var(1693)
                                    + var(1694) * SymbExpr::from_u32(256)
                                    + var(1695) * SymbExpr::from_u32(65536)
                                    + var(1696) * SymbExpr::from_u32(16777216),
                                var(1697)
                                    + var(1698) * SymbExpr::from_u32(256)
                                    + var(1699) * SymbExpr::from_u32(65536)
                                    + var(1700) * SymbExpr::from_u32(16777216),
                                var(1701)
                                    + var(1702) * SymbExpr::from_u32(256)
                                    + var(1703) * SymbExpr::from_u32(65536)
                                    + var(1704) * SymbExpr::from_u32(16777216),
                                var(1705)
                                    + var(1706) * SymbExpr::from_u32(256)
                                    + var(1707) * SymbExpr::from_u32(65536)
                                    + var(1708) * SymbExpr::from_u32(16777216),
                                var(1709)
                                    + var(1710) * SymbExpr::from_u32(256)
                                    + var(1711) * SymbExpr::from_u32(65536)
                                    + var(1712) * SymbExpr::from_u32(16777216),
                                var(1713)
                                    + var(1714) * SymbExpr::from_u32(256)
                                    + var(1715) * SymbExpr::from_u32(65536)
                                    + var(1716) * SymbExpr::from_u32(16777216),
                                var(1717)
                                    + var(1718) * SymbExpr::from_u32(256)
                                    + var(1719) * SymbExpr::from_u32(65536)
                                    + var(1720) * SymbExpr::from_u32(16777216),
                                var(1721)
                                    + var(1722) * SymbExpr::from_u32(256)
                                    + var(1723) * SymbExpr::from_u32(65536)
                                    + var(1724) * SymbExpr::from_u32(16777216),
                                var(1725)
                                    + var(1726) * SymbExpr::from_u32(256)
                                    + var(1727) * SymbExpr::from_u32(65536)
                                    + var(1728) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 40
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1729)
                                    + var(1730) * SymbExpr::from_u32(256)
                                    + var(1731) * SymbExpr::from_u32(65536)
                                    + var(1732) * SymbExpr::from_u32(16777216),
                                var(1733)
                                    + var(1734) * SymbExpr::from_u32(256)
                                    + var(1735) * SymbExpr::from_u32(65536)
                                    + var(1736) * SymbExpr::from_u32(16777216),
                                var(1737)
                                    + var(1738) * SymbExpr::from_u32(256)
                                    + var(1739) * SymbExpr::from_u32(65536)
                                    + var(1740) * SymbExpr::from_u32(16777216),
                                var(1741)
                                    + var(1742) * SymbExpr::from_u32(256)
                                    + var(1743) * SymbExpr::from_u32(65536)
                                    + var(1744) * SymbExpr::from_u32(16777216),
                                var(1745)
                                    + var(1746) * SymbExpr::from_u32(256)
                                    + var(1747) * SymbExpr::from_u32(65536)
                                    + var(1748) * SymbExpr::from_u32(16777216),
                                var(1749)
                                    + var(1750) * SymbExpr::from_u32(256)
                                    + var(1751) * SymbExpr::from_u32(65536)
                                    + var(1752) * SymbExpr::from_u32(16777216),
                                var(1753)
                                    + var(1754) * SymbExpr::from_u32(256)
                                    + var(1755) * SymbExpr::from_u32(65536)
                                    + var(1756) * SymbExpr::from_u32(16777216),
                                var(1757)
                                    + var(1758) * SymbExpr::from_u32(256)
                                    + var(1759) * SymbExpr::from_u32(65536)
                                    + var(1760) * SymbExpr::from_u32(16777216),
                                var(1761)
                                    + var(1762) * SymbExpr::from_u32(256)
                                    + var(1763) * SymbExpr::from_u32(65536)
                                    + var(1764) * SymbExpr::from_u32(16777216),
                                var(1765)
                                    + var(1766) * SymbExpr::from_u32(256)
                                    + var(1767) * SymbExpr::from_u32(65536)
                                    + var(1768) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 41
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1769)
                                    + var(1770) * SymbExpr::from_u32(256)
                                    + var(1771) * SymbExpr::from_u32(65536)
                                    + var(1772) * SymbExpr::from_u32(16777216),
                                var(1773)
                                    + var(1774) * SymbExpr::from_u32(256)
                                    + var(1775) * SymbExpr::from_u32(65536)
                                    + var(1776) * SymbExpr::from_u32(16777216),
                                var(1777)
                                    + var(1778) * SymbExpr::from_u32(256)
                                    + var(1779) * SymbExpr::from_u32(65536)
                                    + var(1780) * SymbExpr::from_u32(16777216),
                                var(1781)
                                    + var(1782) * SymbExpr::from_u32(256)
                                    + var(1783) * SymbExpr::from_u32(65536)
                                    + var(1784) * SymbExpr::from_u32(16777216),
                                var(1785)
                                    + var(1786) * SymbExpr::from_u32(256)
                                    + var(1787) * SymbExpr::from_u32(65536)
                                    + var(1788) * SymbExpr::from_u32(16777216),
                                var(1789)
                                    + var(1790) * SymbExpr::from_u32(256)
                                    + var(1791) * SymbExpr::from_u32(65536)
                                    + var(1792) * SymbExpr::from_u32(16777216),
                                var(1793)
                                    + var(1794) * SymbExpr::from_u32(256)
                                    + var(1795) * SymbExpr::from_u32(65536)
                                    + var(1796) * SymbExpr::from_u32(16777216),
                                var(1797)
                                    + var(1798) * SymbExpr::from_u32(256)
                                    + var(1799) * SymbExpr::from_u32(65536)
                                    + var(1800) * SymbExpr::from_u32(16777216),
                                var(1801)
                                    + var(1802) * SymbExpr::from_u32(256)
                                    + var(1803) * SymbExpr::from_u32(65536)
                                    + var(1804) * SymbExpr::from_u32(16777216),
                                var(1805)
                                    + var(1806) * SymbExpr::from_u32(256)
                                    + var(1807) * SymbExpr::from_u32(65536)
                                    + var(1808) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 42
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1809)
                                    + var(1810) * SymbExpr::from_u32(256)
                                    + var(1811) * SymbExpr::from_u32(65536)
                                    + var(1812) * SymbExpr::from_u32(16777216),
                                var(1813)
                                    + var(1814) * SymbExpr::from_u32(256)
                                    + var(1815) * SymbExpr::from_u32(65536)
                                    + var(1816) * SymbExpr::from_u32(16777216),
                                var(1817)
                                    + var(1818) * SymbExpr::from_u32(256)
                                    + var(1819) * SymbExpr::from_u32(65536)
                                    + var(1820) * SymbExpr::from_u32(16777216),
                                var(1821)
                                    + var(1822) * SymbExpr::from_u32(256)
                                    + var(1823) * SymbExpr::from_u32(65536)
                                    + var(1824) * SymbExpr::from_u32(16777216),
                                var(1825)
                                    + var(1826) * SymbExpr::from_u32(256)
                                    + var(1827) * SymbExpr::from_u32(65536)
                                    + var(1828) * SymbExpr::from_u32(16777216),
                                var(1829)
                                    + var(1830) * SymbExpr::from_u32(256)
                                    + var(1831) * SymbExpr::from_u32(65536)
                                    + var(1832) * SymbExpr::from_u32(16777216),
                                var(1833)
                                    + var(1834) * SymbExpr::from_u32(256)
                                    + var(1835) * SymbExpr::from_u32(65536)
                                    + var(1836) * SymbExpr::from_u32(16777216),
                                var(1837)
                                    + var(1838) * SymbExpr::from_u32(256)
                                    + var(1839) * SymbExpr::from_u32(65536)
                                    + var(1840) * SymbExpr::from_u32(16777216),
                                var(1841)
                                    + var(1842) * SymbExpr::from_u32(256)
                                    + var(1843) * SymbExpr::from_u32(65536)
                                    + var(1844) * SymbExpr::from_u32(16777216),
                                var(1845)
                                    + var(1846) * SymbExpr::from_u32(256)
                                    + var(1847) * SymbExpr::from_u32(65536)
                                    + var(1848) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 43
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1849)
                                    + var(1850) * SymbExpr::from_u32(256)
                                    + var(1851) * SymbExpr::from_u32(65536)
                                    + var(1852) * SymbExpr::from_u32(16777216),
                                var(1853)
                                    + var(1854) * SymbExpr::from_u32(256)
                                    + var(1855) * SymbExpr::from_u32(65536)
                                    + var(1856) * SymbExpr::from_u32(16777216),
                                var(1857)
                                    + var(1858) * SymbExpr::from_u32(256)
                                    + var(1859) * SymbExpr::from_u32(65536)
                                    + var(1860) * SymbExpr::from_u32(16777216),
                                var(1861)
                                    + var(1862) * SymbExpr::from_u32(256)
                                    + var(1863) * SymbExpr::from_u32(65536)
                                    + var(1864) * SymbExpr::from_u32(16777216),
                                var(1865)
                                    + var(1866) * SymbExpr::from_u32(256)
                                    + var(1867) * SymbExpr::from_u32(65536)
                                    + var(1868) * SymbExpr::from_u32(16777216),
                                var(1869)
                                    + var(1870) * SymbExpr::from_u32(256)
                                    + var(1871) * SymbExpr::from_u32(65536)
                                    + var(1872) * SymbExpr::from_u32(16777216),
                                var(1873)
                                    + var(1874) * SymbExpr::from_u32(256)
                                    + var(1875) * SymbExpr::from_u32(65536)
                                    + var(1876) * SymbExpr::from_u32(16777216),
                                var(1877)
                                    + var(1878) * SymbExpr::from_u32(256)
                                    + var(1879) * SymbExpr::from_u32(65536)
                                    + var(1880) * SymbExpr::from_u32(16777216),
                                var(1881)
                                    + var(1882) * SymbExpr::from_u32(256)
                                    + var(1883) * SymbExpr::from_u32(65536)
                                    + var(1884) * SymbExpr::from_u32(16777216),
                                var(1885)
                                    + var(1886) * SymbExpr::from_u32(256)
                                    + var(1887) * SymbExpr::from_u32(65536)
                                    + var(1888) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 44
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1889)
                                    + var(1890) * SymbExpr::from_u32(256)
                                    + var(1891) * SymbExpr::from_u32(65536)
                                    + var(1892) * SymbExpr::from_u32(16777216),
                                var(1893)
                                    + var(1894) * SymbExpr::from_u32(256)
                                    + var(1895) * SymbExpr::from_u32(65536)
                                    + var(1896) * SymbExpr::from_u32(16777216),
                                var(1897)
                                    + var(1898) * SymbExpr::from_u32(256)
                                    + var(1899) * SymbExpr::from_u32(65536)
                                    + var(1900) * SymbExpr::from_u32(16777216),
                                var(1901)
                                    + var(1902) * SymbExpr::from_u32(256)
                                    + var(1903) * SymbExpr::from_u32(65536)
                                    + var(1904) * SymbExpr::from_u32(16777216),
                                var(1905)
                                    + var(1906) * SymbExpr::from_u32(256)
                                    + var(1907) * SymbExpr::from_u32(65536)
                                    + var(1908) * SymbExpr::from_u32(16777216),
                                var(1909)
                                    + var(1910) * SymbExpr::from_u32(256)
                                    + var(1911) * SymbExpr::from_u32(65536)
                                    + var(1912) * SymbExpr::from_u32(16777216),
                                var(1913)
                                    + var(1914) * SymbExpr::from_u32(256)
                                    + var(1915) * SymbExpr::from_u32(65536)
                                    + var(1916) * SymbExpr::from_u32(16777216),
                                var(1917)
                                    + var(1918) * SymbExpr::from_u32(256)
                                    + var(1919) * SymbExpr::from_u32(65536)
                                    + var(1920) * SymbExpr::from_u32(16777216),
                                var(1921)
                                    + var(1922) * SymbExpr::from_u32(256)
                                    + var(1923) * SymbExpr::from_u32(65536)
                                    + var(1924) * SymbExpr::from_u32(16777216),
                                var(1925)
                                    + var(1926) * SymbExpr::from_u32(256)
                                    + var(1927) * SymbExpr::from_u32(65536)
                                    + var(1928) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 45
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1929)
                                    + var(1930) * SymbExpr::from_u32(256)
                                    + var(1931) * SymbExpr::from_u32(65536)
                                    + var(1932) * SymbExpr::from_u32(16777216),
                                var(1933)
                                    + var(1934) * SymbExpr::from_u32(256)
                                    + var(1935) * SymbExpr::from_u32(65536)
                                    + var(1936) * SymbExpr::from_u32(16777216),
                                var(1937)
                                    + var(1938) * SymbExpr::from_u32(256)
                                    + var(1939) * SymbExpr::from_u32(65536)
                                    + var(1940) * SymbExpr::from_u32(16777216),
                                var(1941)
                                    + var(1942) * SymbExpr::from_u32(256)
                                    + var(1943) * SymbExpr::from_u32(65536)
                                    + var(1944) * SymbExpr::from_u32(16777216),
                                var(1945)
                                    + var(1946) * SymbExpr::from_u32(256)
                                    + var(1947) * SymbExpr::from_u32(65536)
                                    + var(1948) * SymbExpr::from_u32(16777216),
                                var(1949)
                                    + var(1950) * SymbExpr::from_u32(256)
                                    + var(1951) * SymbExpr::from_u32(65536)
                                    + var(1952) * SymbExpr::from_u32(16777216),
                                var(1953)
                                    + var(1954) * SymbExpr::from_u32(256)
                                    + var(1955) * SymbExpr::from_u32(65536)
                                    + var(1956) * SymbExpr::from_u32(16777216),
                                var(1957)
                                    + var(1958) * SymbExpr::from_u32(256)
                                    + var(1959) * SymbExpr::from_u32(65536)
                                    + var(1960) * SymbExpr::from_u32(16777216),
                                var(1961)
                                    + var(1962) * SymbExpr::from_u32(256)
                                    + var(1963) * SymbExpr::from_u32(65536)
                                    + var(1964) * SymbExpr::from_u32(16777216),
                                var(1965)
                                    + var(1966) * SymbExpr::from_u32(256)
                                    + var(1967) * SymbExpr::from_u32(65536)
                                    + var(1968) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 46
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(1969)
                                    + var(1970) * SymbExpr::from_u32(256)
                                    + var(1971) * SymbExpr::from_u32(65536)
                                    + var(1972) * SymbExpr::from_u32(16777216),
                                var(1973)
                                    + var(1974) * SymbExpr::from_u32(256)
                                    + var(1975) * SymbExpr::from_u32(65536)
                                    + var(1976) * SymbExpr::from_u32(16777216),
                                var(1977)
                                    + var(1978) * SymbExpr::from_u32(256)
                                    + var(1979) * SymbExpr::from_u32(65536)
                                    + var(1980) * SymbExpr::from_u32(16777216),
                                var(1981)
                                    + var(1982) * SymbExpr::from_u32(256)
                                    + var(1983) * SymbExpr::from_u32(65536)
                                    + var(1984) * SymbExpr::from_u32(16777216),
                                var(1985)
                                    + var(1986) * SymbExpr::from_u32(256)
                                    + var(1987) * SymbExpr::from_u32(65536)
                                    + var(1988) * SymbExpr::from_u32(16777216),
                                var(1989)
                                    + var(1990) * SymbExpr::from_u32(256)
                                    + var(1991) * SymbExpr::from_u32(65536)
                                    + var(1992) * SymbExpr::from_u32(16777216),
                                var(1993)
                                    + var(1994) * SymbExpr::from_u32(256)
                                    + var(1995) * SymbExpr::from_u32(65536)
                                    + var(1996) * SymbExpr::from_u32(16777216),
                                var(1997)
                                    + var(1998) * SymbExpr::from_u32(256)
                                    + var(1999) * SymbExpr::from_u32(65536)
                                    + var(2000) * SymbExpr::from_u32(16777216),
                                var(2001)
                                    + var(2002) * SymbExpr::from_u32(256)
                                    + var(2003) * SymbExpr::from_u32(65536)
                                    + var(2004) * SymbExpr::from_u32(16777216),
                                var(2005)
                                    + var(2006) * SymbExpr::from_u32(256)
                                    + var(2007) * SymbExpr::from_u32(65536)
                                    + var(2008) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 47
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(2009)
                                    + var(2010) * SymbExpr::from_u32(256)
                                    + var(2011) * SymbExpr::from_u32(65536)
                                    + var(2012) * SymbExpr::from_u32(16777216),
                                var(2013)
                                    + var(2014) * SymbExpr::from_u32(256)
                                    + var(2015) * SymbExpr::from_u32(65536)
                                    + var(2016) * SymbExpr::from_u32(16777216),
                                var(2017)
                                    + var(2018) * SymbExpr::from_u32(256)
                                    + var(2019) * SymbExpr::from_u32(65536)
                                    + var(2020) * SymbExpr::from_u32(16777216),
                                var(2021)
                                    + var(2022) * SymbExpr::from_u32(256)
                                    + var(2023) * SymbExpr::from_u32(65536)
                                    + var(2024) * SymbExpr::from_u32(16777216),
                                var(2025)
                                    + var(2026) * SymbExpr::from_u32(256)
                                    + var(2027) * SymbExpr::from_u32(65536)
                                    + var(2028) * SymbExpr::from_u32(16777216),
                                var(2029)
                                    + var(2030) * SymbExpr::from_u32(256)
                                    + var(2031) * SymbExpr::from_u32(65536)
                                    + var(2032) * SymbExpr::from_u32(16777216),
                                var(2033)
                                    + var(2034) * SymbExpr::from_u32(256)
                                    + var(2035) * SymbExpr::from_u32(65536)
                                    + var(2036) * SymbExpr::from_u32(16777216),
                                var(2037)
                                    + var(2038) * SymbExpr::from_u32(256)
                                    + var(2039) * SymbExpr::from_u32(65536)
                                    + var(2040) * SymbExpr::from_u32(16777216),
                                var(2041)
                                    + var(2042) * SymbExpr::from_u32(256)
                                    + var(2043) * SymbExpr::from_u32(65536)
                                    + var(2044) * SymbExpr::from_u32(16777216),
                                var(2045)
                                    + var(2046) * SymbExpr::from_u32(256)
                                    + var(2047) * SymbExpr::from_u32(65536)
                                    + var(2048) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 48
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(2049)
                                    + var(2050) * SymbExpr::from_u32(256)
                                    + var(2051) * SymbExpr::from_u32(65536)
                                    + var(2052) * SymbExpr::from_u32(16777216),
                                var(2053)
                                    + var(2054) * SymbExpr::from_u32(256)
                                    + var(2055) * SymbExpr::from_u32(65536)
                                    + var(2056) * SymbExpr::from_u32(16777216),
                                var(2057)
                                    + var(2058) * SymbExpr::from_u32(256)
                                    + var(2059) * SymbExpr::from_u32(65536)
                                    + var(2060) * SymbExpr::from_u32(16777216),
                                var(2061)
                                    + var(2062) * SymbExpr::from_u32(256)
                                    + var(2063) * SymbExpr::from_u32(65536)
                                    + var(2064) * SymbExpr::from_u32(16777216),
                                var(2065)
                                    + var(2066) * SymbExpr::from_u32(256)
                                    + var(2067) * SymbExpr::from_u32(65536)
                                    + var(2068) * SymbExpr::from_u32(16777216),
                                var(2069)
                                    + var(2070) * SymbExpr::from_u32(256)
                                    + var(2071) * SymbExpr::from_u32(65536)
                                    + var(2072) * SymbExpr::from_u32(16777216),
                                var(2073)
                                    + var(2074) * SymbExpr::from_u32(256)
                                    + var(2075) * SymbExpr::from_u32(65536)
                                    + var(2076) * SymbExpr::from_u32(16777216),
                                var(2077)
                                    + var(2078) * SymbExpr::from_u32(256)
                                    + var(2079) * SymbExpr::from_u32(65536)
                                    + var(2080) * SymbExpr::from_u32(16777216),
                                var(2081)
                                    + var(2082) * SymbExpr::from_u32(256)
                                    + var(2083) * SymbExpr::from_u32(65536)
                                    + var(2084) * SymbExpr::from_u32(16777216),
                                var(2085)
                                    + var(2086) * SymbExpr::from_u32(256)
                                    + var(2087) * SymbExpr::from_u32(65536)
                                    + var(2088) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 49
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(2089)
                                    + var(2090) * SymbExpr::from_u32(256)
                                    + var(2091) * SymbExpr::from_u32(65536)
                                    + var(2092) * SymbExpr::from_u32(16777216),
                                var(2093)
                                    + var(2094) * SymbExpr::from_u32(256)
                                    + var(2095) * SymbExpr::from_u32(65536)
                                    + var(2096) * SymbExpr::from_u32(16777216),
                                var(2097)
                                    + var(2098) * SymbExpr::from_u32(256)
                                    + var(2099) * SymbExpr::from_u32(65536)
                                    + var(2100) * SymbExpr::from_u32(16777216),
                                var(2101)
                                    + var(2102) * SymbExpr::from_u32(256)
                                    + var(2103) * SymbExpr::from_u32(65536)
                                    + var(2104) * SymbExpr::from_u32(16777216),
                                var(2105)
                                    + var(2106) * SymbExpr::from_u32(256)
                                    + var(2107) * SymbExpr::from_u32(65536)
                                    + var(2108) * SymbExpr::from_u32(16777216),
                                var(2109)
                                    + var(2110) * SymbExpr::from_u32(256)
                                    + var(2111) * SymbExpr::from_u32(65536)
                                    + var(2112) * SymbExpr::from_u32(16777216),
                                var(2113)
                                    + var(2114) * SymbExpr::from_u32(256)
                                    + var(2115) * SymbExpr::from_u32(65536)
                                    + var(2116) * SymbExpr::from_u32(16777216),
                                var(2117)
                                    + var(2118) * SymbExpr::from_u32(256)
                                    + var(2119) * SymbExpr::from_u32(65536)
                                    + var(2120) * SymbExpr::from_u32(16777216),
                                var(2121)
                                    + var(2122) * SymbExpr::from_u32(256)
                                    + var(2123) * SymbExpr::from_u32(65536)
                                    + var(2124) * SymbExpr::from_u32(16777216),
                                var(2125)
                                    + var(2126) * SymbExpr::from_u32(256)
                                    + var(2127) * SymbExpr::from_u32(65536)
                                    + var(2128) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 50
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(2129)
                                    + var(2130) * SymbExpr::from_u32(256)
                                    + var(2131) * SymbExpr::from_u32(65536)
                                    + var(2132) * SymbExpr::from_u32(16777216),
                                var(2133)
                                    + var(2134) * SymbExpr::from_u32(256)
                                    + var(2135) * SymbExpr::from_u32(65536)
                                    + var(2136) * SymbExpr::from_u32(16777216),
                                var(2137)
                                    + var(2138) * SymbExpr::from_u32(256)
                                    + var(2139) * SymbExpr::from_u32(65536)
                                    + var(2140) * SymbExpr::from_u32(16777216),
                                var(2141)
                                    + var(2142) * SymbExpr::from_u32(256)
                                    + var(2143) * SymbExpr::from_u32(65536)
                                    + var(2144) * SymbExpr::from_u32(16777216),
                                var(2145)
                                    + var(2146) * SymbExpr::from_u32(256)
                                    + var(2147) * SymbExpr::from_u32(65536)
                                    + var(2148) * SymbExpr::from_u32(16777216),
                                var(2149)
                                    + var(2150) * SymbExpr::from_u32(256)
                                    + var(2151) * SymbExpr::from_u32(65536)
                                    + var(2152) * SymbExpr::from_u32(16777216),
                                var(2153)
                                    + var(2154) * SymbExpr::from_u32(256)
                                    + var(2155) * SymbExpr::from_u32(65536)
                                    + var(2156) * SymbExpr::from_u32(16777216),
                                var(2157)
                                    + var(2158) * SymbExpr::from_u32(256)
                                    + var(2159) * SymbExpr::from_u32(65536)
                                    + var(2160) * SymbExpr::from_u32(16777216),
                                var(2161)
                                    + var(2162) * SymbExpr::from_u32(256)
                                    + var(2163) * SymbExpr::from_u32(65536)
                                    + var(2164) * SymbExpr::from_u32(16777216),
                                var(2165)
                                    + var(2166) * SymbExpr::from_u32(256)
                                    + var(2167) * SymbExpr::from_u32(65536)
                                    + var(2168) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 51
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(2169)
                                    + var(2170) * SymbExpr::from_u32(256)
                                    + var(2171) * SymbExpr::from_u32(65536)
                                    + var(2172) * SymbExpr::from_u32(16777216),
                                var(2173)
                                    + var(2174) * SymbExpr::from_u32(256)
                                    + var(2175) * SymbExpr::from_u32(65536)
                                    + var(2176) * SymbExpr::from_u32(16777216),
                                var(2177)
                                    + var(2178) * SymbExpr::from_u32(256)
                                    + var(2179) * SymbExpr::from_u32(65536)
                                    + var(2180) * SymbExpr::from_u32(16777216),
                                var(2181)
                                    + var(2182) * SymbExpr::from_u32(256)
                                    + var(2183) * SymbExpr::from_u32(65536)
                                    + var(2184) * SymbExpr::from_u32(16777216),
                                var(2185)
                                    + var(2186) * SymbExpr::from_u32(256)
                                    + var(2187) * SymbExpr::from_u32(65536)
                                    + var(2188) * SymbExpr::from_u32(16777216),
                                var(2189)
                                    + var(2190) * SymbExpr::from_u32(256)
                                    + var(2191) * SymbExpr::from_u32(65536)
                                    + var(2192) * SymbExpr::from_u32(16777216),
                                var(2193)
                                    + var(2194) * SymbExpr::from_u32(256)
                                    + var(2195) * SymbExpr::from_u32(65536)
                                    + var(2196) * SymbExpr::from_u32(16777216),
                                var(2197)
                                    + var(2198) * SymbExpr::from_u32(256)
                                    + var(2199) * SymbExpr::from_u32(65536)
                                    + var(2200) * SymbExpr::from_u32(16777216),
                                var(2201)
                                    + var(2202) * SymbExpr::from_u32(256)
                                    + var(2203) * SymbExpr::from_u32(65536)
                                    + var(2204) * SymbExpr::from_u32(16777216),
                                var(2205)
                                    + var(2206) * SymbExpr::from_u32(256)
                                    + var(2207) * SymbExpr::from_u32(65536)
                                    + var(2208) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 52
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(2209)
                                    + var(2210) * SymbExpr::from_u32(256)
                                    + var(2211) * SymbExpr::from_u32(65536)
                                    + var(2212) * SymbExpr::from_u32(16777216),
                                var(2213)
                                    + var(2214) * SymbExpr::from_u32(256)
                                    + var(2215) * SymbExpr::from_u32(65536)
                                    + var(2216) * SymbExpr::from_u32(16777216),
                                var(2217)
                                    + var(2218) * SymbExpr::from_u32(256)
                                    + var(2219) * SymbExpr::from_u32(65536)
                                    + var(2220) * SymbExpr::from_u32(16777216),
                                var(2221)
                                    + var(2222) * SymbExpr::from_u32(256)
                                    + var(2223) * SymbExpr::from_u32(65536)
                                    + var(2224) * SymbExpr::from_u32(16777216),
                                var(2225)
                                    + var(2226) * SymbExpr::from_u32(256)
                                    + var(2227) * SymbExpr::from_u32(65536)
                                    + var(2228) * SymbExpr::from_u32(16777216),
                                var(2229)
                                    + var(2230) * SymbExpr::from_u32(256)
                                    + var(2231) * SymbExpr::from_u32(65536)
                                    + var(2232) * SymbExpr::from_u32(16777216),
                                var(2233)
                                    + var(2234) * SymbExpr::from_u32(256)
                                    + var(2235) * SymbExpr::from_u32(65536)
                                    + var(2236) * SymbExpr::from_u32(16777216),
                                var(2237)
                                    + var(2238) * SymbExpr::from_u32(256)
                                    + var(2239) * SymbExpr::from_u32(65536)
                                    + var(2240) * SymbExpr::from_u32(16777216),
                                var(2241)
                                    + var(2242) * SymbExpr::from_u32(256)
                                    + var(2243) * SymbExpr::from_u32(65536)
                                    + var(2244) * SymbExpr::from_u32(16777216),
                                var(2245)
                                    + var(2246) * SymbExpr::from_u32(256)
                                    + var(2247) * SymbExpr::from_u32(65536)
                                    + var(2248) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 53
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(2249)
                                    + var(2250) * SymbExpr::from_u32(256)
                                    + var(2251) * SymbExpr::from_u32(65536)
                                    + var(2252) * SymbExpr::from_u32(16777216),
                                var(2253)
                                    + var(2254) * SymbExpr::from_u32(256)
                                    + var(2255) * SymbExpr::from_u32(65536)
                                    + var(2256) * SymbExpr::from_u32(16777216),
                                var(2257)
                                    + var(2258) * SymbExpr::from_u32(256)
                                    + var(2259) * SymbExpr::from_u32(65536)
                                    + var(2260) * SymbExpr::from_u32(16777216),
                                var(2261)
                                    + var(2262) * SymbExpr::from_u32(256)
                                    + var(2263) * SymbExpr::from_u32(65536)
                                    + var(2264) * SymbExpr::from_u32(16777216),
                                var(2265)
                                    + var(2266) * SymbExpr::from_u32(256)
                                    + var(2267) * SymbExpr::from_u32(65536)
                                    + var(2268) * SymbExpr::from_u32(16777216),
                                var(2269)
                                    + var(2270) * SymbExpr::from_u32(256)
                                    + var(2271) * SymbExpr::from_u32(65536)
                                    + var(2272) * SymbExpr::from_u32(16777216),
                                var(2273)
                                    + var(2274) * SymbExpr::from_u32(256)
                                    + var(2275) * SymbExpr::from_u32(65536)
                                    + var(2276) * SymbExpr::from_u32(16777216),
                                var(2277)
                                    + var(2278) * SymbExpr::from_u32(256)
                                    + var(2279) * SymbExpr::from_u32(65536)
                                    + var(2280) * SymbExpr::from_u32(16777216),
                                var(2281)
                                    + var(2282) * SymbExpr::from_u32(256)
                                    + var(2283) * SymbExpr::from_u32(65536)
                                    + var(2284) * SymbExpr::from_u32(16777216),
                                var(2285)
                                    + var(2286) * SymbExpr::from_u32(256)
                                    + var(2287) * SymbExpr::from_u32(65536)
                                    + var(2288) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 54
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(2289)
                                    + var(2290) * SymbExpr::from_u32(256)
                                    + var(2291) * SymbExpr::from_u32(65536)
                                    + var(2292) * SymbExpr::from_u32(16777216),
                                var(2293)
                                    + var(2294) * SymbExpr::from_u32(256)
                                    + var(2295) * SymbExpr::from_u32(65536)
                                    + var(2296) * SymbExpr::from_u32(16777216),
                                var(2297)
                                    + var(2298) * SymbExpr::from_u32(256)
                                    + var(2299) * SymbExpr::from_u32(65536)
                                    + var(2300) * SymbExpr::from_u32(16777216),
                                var(2301)
                                    + var(2302) * SymbExpr::from_u32(256)
                                    + var(2303) * SymbExpr::from_u32(65536)
                                    + var(2304) * SymbExpr::from_u32(16777216),
                                var(2305)
                                    + var(2306) * SymbExpr::from_u32(256)
                                    + var(2307) * SymbExpr::from_u32(65536)
                                    + var(2308) * SymbExpr::from_u32(16777216),
                                var(2309)
                                    + var(2310) * SymbExpr::from_u32(256)
                                    + var(2311) * SymbExpr::from_u32(65536)
                                    + var(2312) * SymbExpr::from_u32(16777216),
                                var(2313)
                                    + var(2314) * SymbExpr::from_u32(256)
                                    + var(2315) * SymbExpr::from_u32(65536)
                                    + var(2316) * SymbExpr::from_u32(16777216),
                                var(2317)
                                    + var(2318) * SymbExpr::from_u32(256)
                                    + var(2319) * SymbExpr::from_u32(65536)
                                    + var(2320) * SymbExpr::from_u32(16777216),
                                var(2321)
                                    + var(2322) * SymbExpr::from_u32(256)
                                    + var(2323) * SymbExpr::from_u32(65536)
                                    + var(2324) * SymbExpr::from_u32(16777216),
                                var(2325)
                                    + var(2326) * SymbExpr::from_u32(256)
                                    + var(2327) * SymbExpr::from_u32(65536)
                                    + var(2328) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // ROUND 55
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(g_function_idx),
                                var(2329)
                                    + var(2330) * SymbExpr::from_u32(256)
                                    + var(2331) * SymbExpr::from_u32(65536)
                                    + var(2332) * SymbExpr::from_u32(16777216),
                                var(2333)
                                    + var(2334) * SymbExpr::from_u32(256)
                                    + var(2335) * SymbExpr::from_u32(65536)
                                    + var(2336) * SymbExpr::from_u32(16777216),
                                var(2337)
                                    + var(2338) * SymbExpr::from_u32(256)
                                    + var(2339) * SymbExpr::from_u32(65536)
                                    + var(2340) * SymbExpr::from_u32(16777216),
                                var(2341)
                                    + var(2342) * SymbExpr::from_u32(256)
                                    + var(2343) * SymbExpr::from_u32(65536)
                                    + var(2344) * SymbExpr::from_u32(16777216),
                                var(2345)
                                    + var(2346) * SymbExpr::from_u32(256)
                                    + var(2347) * SymbExpr::from_u32(65536)
                                    + var(2348) * SymbExpr::from_u32(16777216),
                                var(2349)
                                    + var(2350) * SymbExpr::from_u32(256)
                                    + var(2351) * SymbExpr::from_u32(65536)
                                    + var(2352) * SymbExpr::from_u32(16777216),
                                var(2353)
                                    + var(2354) * SymbExpr::from_u32(256)
                                    + var(2355) * SymbExpr::from_u32(65536)
                                    + var(2356) * SymbExpr::from_u32(16777216),
                                var(2357)
                                    + var(2358) * SymbExpr::from_u32(256)
                                    + var(2359) * SymbExpr::from_u32(65536)
                                    + var(2360) * SymbExpr::from_u32(16777216),
                                var(2361)
                                    + var(2362) * SymbExpr::from_u32(256)
                                    + var(2363) * SymbExpr::from_u32(65536)
                                    + var(2364) * SymbExpr::from_u32(16777216),
                                var(2365)
                                    + var(2366) * SymbExpr::from_u32(256)
                                    + var(2367) * SymbExpr::from_u32(65536)
                                    + var(2368) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // state[i] ^= state[i + 8]
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2369)
                                    + var(2370) * SymbExpr::from_u32(256)
                                    + var(2371) * SymbExpr::from_u32(65536)
                                    + var(2372) * SymbExpr::from_u32(16777216),
                                var(2373)
                                    + var(2374) * SymbExpr::from_u32(256)
                                    + var(2375) * SymbExpr::from_u32(65536)
                                    + var(2376) * SymbExpr::from_u32(16777216),
                                var(2377)
                                    + var(2378) * SymbExpr::from_u32(256)
                                    + var(2379) * SymbExpr::from_u32(65536)
                                    + var(2380) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2381)
                                    + var(2382) * SymbExpr::from_u32(256)
                                    + var(2383) * SymbExpr::from_u32(65536)
                                    + var(2384) * SymbExpr::from_u32(16777216),
                                var(2385)
                                    + var(2386) * SymbExpr::from_u32(256)
                                    + var(2387) * SymbExpr::from_u32(65536)
                                    + var(2388) * SymbExpr::from_u32(16777216),
                                var(2389)
                                    + var(2390) * SymbExpr::from_u32(256)
                                    + var(2391) * SymbExpr::from_u32(65536)
                                    + var(2392) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2393)
                                    + var(2394) * SymbExpr::from_u32(256)
                                    + var(2395) * SymbExpr::from_u32(65536)
                                    + var(2396) * SymbExpr::from_u32(16777216),
                                var(2397)
                                    + var(2398) * SymbExpr::from_u32(256)
                                    + var(2399) * SymbExpr::from_u32(65536)
                                    + var(2400) * SymbExpr::from_u32(16777216),
                                var(2401)
                                    + var(2402) * SymbExpr::from_u32(256)
                                    + var(2403) * SymbExpr::from_u32(65536)
                                    + var(2404) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2405)
                                    + var(2406) * SymbExpr::from_u32(256)
                                    + var(2407) * SymbExpr::from_u32(65536)
                                    + var(2408) * SymbExpr::from_u32(16777216),
                                var(2409)
                                    + var(2410) * SymbExpr::from_u32(256)
                                    + var(2411) * SymbExpr::from_u32(65536)
                                    + var(2412) * SymbExpr::from_u32(16777216),
                                var(2413)
                                    + var(2414) * SymbExpr::from_u32(256)
                                    + var(2415) * SymbExpr::from_u32(65536)
                                    + var(2416) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2417)
                                    + var(2418) * SymbExpr::from_u32(256)
                                    + var(2419) * SymbExpr::from_u32(65536)
                                    + var(2420) * SymbExpr::from_u32(16777216),
                                var(2421)
                                    + var(2422) * SymbExpr::from_u32(256)
                                    + var(2423) * SymbExpr::from_u32(65536)
                                    + var(2424) * SymbExpr::from_u32(16777216),
                                var(2425)
                                    + var(2426) * SymbExpr::from_u32(256)
                                    + var(2427) * SymbExpr::from_u32(65536)
                                    + var(2428) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2429)
                                    + var(2430) * SymbExpr::from_u32(256)
                                    + var(2431) * SymbExpr::from_u32(65536)
                                    + var(2432) * SymbExpr::from_u32(16777216),
                                var(2433)
                                    + var(2434) * SymbExpr::from_u32(256)
                                    + var(2435) * SymbExpr::from_u32(65536)
                                    + var(2436) * SymbExpr::from_u32(16777216),
                                var(2437)
                                    + var(2438) * SymbExpr::from_u32(256)
                                    + var(2439) * SymbExpr::from_u32(65536)
                                    + var(2440) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2441)
                                    + var(2442) * SymbExpr::from_u32(256)
                                    + var(2443) * SymbExpr::from_u32(65536)
                                    + var(2444) * SymbExpr::from_u32(16777216),
                                var(2445)
                                    + var(2446) * SymbExpr::from_u32(256)
                                    + var(2447) * SymbExpr::from_u32(65536)
                                    + var(2448) * SymbExpr::from_u32(16777216),
                                var(2449)
                                    + var(2450) * SymbExpr::from_u32(256)
                                    + var(2451) * SymbExpr::from_u32(65536)
                                    + var(2452) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2453)
                                    + var(2454) * SymbExpr::from_u32(256)
                                    + var(2455) * SymbExpr::from_u32(65536)
                                    + var(2456) * SymbExpr::from_u32(16777216),
                                var(2457)
                                    + var(2458) * SymbExpr::from_u32(256)
                                    + var(2459) * SymbExpr::from_u32(65536)
                                    + var(2460) * SymbExpr::from_u32(16777216),
                                var(2461)
                                    + var(2462) * SymbExpr::from_u32(256)
                                    + var(2463) * SymbExpr::from_u32(65536)
                                    + var(2464) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        // state[i + 8] ^= chaining_value[i]
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2465)
                                    + var(2466) * SymbExpr::from_u32(256)
                                    + var(2467) * SymbExpr::from_u32(65536)
                                    + var(2468) * SymbExpr::from_u32(16777216),
                                var(2469)
                                    + var(2470) * SymbExpr::from_u32(256)
                                    + var(2471) * SymbExpr::from_u32(65536)
                                    + var(2472) * SymbExpr::from_u32(16777216),
                                var(2473)
                                    + var(2474) * SymbExpr::from_u32(256)
                                    + var(2475) * SymbExpr::from_u32(65536)
                                    + var(2476) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2477)
                                    + var(2478) * SymbExpr::from_u32(256)
                                    + var(2479) * SymbExpr::from_u32(65536)
                                    + var(2480) * SymbExpr::from_u32(16777216),
                                var(2481)
                                    + var(2482) * SymbExpr::from_u32(256)
                                    + var(2483) * SymbExpr::from_u32(65536)
                                    + var(2484) * SymbExpr::from_u32(16777216),
                                var(2485)
                                    + var(2486) * SymbExpr::from_u32(256)
                                    + var(2487) * SymbExpr::from_u32(65536)
                                    + var(2488) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2489)
                                    + var(2490) * SymbExpr::from_u32(256)
                                    + var(2491) * SymbExpr::from_u32(65536)
                                    + var(2492) * SymbExpr::from_u32(16777216),
                                var(2493)
                                    + var(2494) * SymbExpr::from_u32(256)
                                    + var(2495) * SymbExpr::from_u32(65536)
                                    + var(2496) * SymbExpr::from_u32(16777216),
                                var(2497)
                                    + var(2498) * SymbExpr::from_u32(256)
                                    + var(2499) * SymbExpr::from_u32(65536)
                                    + var(2500) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2501)
                                    + var(2502) * SymbExpr::from_u32(256)
                                    + var(2503) * SymbExpr::from_u32(65536)
                                    + var(2504) * SymbExpr::from_u32(16777216),
                                var(2505)
                                    + var(2506) * SymbExpr::from_u32(256)
                                    + var(2507) * SymbExpr::from_u32(65536)
                                    + var(2508) * SymbExpr::from_u32(16777216),
                                var(2509)
                                    + var(2510) * SymbExpr::from_u32(256)
                                    + var(2511) * SymbExpr::from_u32(65536)
                                    + var(2512) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2513)
                                    + var(2514) * SymbExpr::from_u32(256)
                                    + var(2515) * SymbExpr::from_u32(65536)
                                    + var(2516) * SymbExpr::from_u32(16777216),
                                var(2517)
                                    + var(2518) * SymbExpr::from_u32(256)
                                    + var(2519) * SymbExpr::from_u32(65536)
                                    + var(2520) * SymbExpr::from_u32(16777216),
                                var(2521)
                                    + var(2522) * SymbExpr::from_u32(256)
                                    + var(2523) * SymbExpr::from_u32(65536)
                                    + var(2524) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2525)
                                    + var(2526) * SymbExpr::from_u32(256)
                                    + var(2527) * SymbExpr::from_u32(65536)
                                    + var(2528) * SymbExpr::from_u32(16777216),
                                var(2529)
                                    + var(2530) * SymbExpr::from_u32(256)
                                    + var(2531) * SymbExpr::from_u32(65536)
                                    + var(2532) * SymbExpr::from_u32(16777216),
                                var(2533)
                                    + var(2534) * SymbExpr::from_u32(256)
                                    + var(2535) * SymbExpr::from_u32(65536)
                                    + var(2536) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2537)
                                    + var(2538) * SymbExpr::from_u32(256)
                                    + var(2539) * SymbExpr::from_u32(65536)
                                    + var(2540) * SymbExpr::from_u32(16777216),
                                var(2541)
                                    + var(2542) * SymbExpr::from_u32(256)
                                    + var(2543) * SymbExpr::from_u32(65536)
                                    + var(2544) * SymbExpr::from_u32(16777216),
                                var(2545)
                                    + var(2546) * SymbExpr::from_u32(256)
                                    + var(2547) * SymbExpr::from_u32(65536)
                                    + var(2548) * SymbExpr::from_u32(16777216),
                            ],
                        ),
                        Lookup::push(
                            SymbExpr::ONE,
                            vec![
                                SymbExpr::from_usize(u32_xor_idx),
                                var(2549)
                                    + var(2550) * SymbExpr::from_u32(256)
                                    + var(2551) * SymbExpr::from_u32(65536)
                                    + var(2552) * SymbExpr::from_u32(16777216),
                                var(2553)
                                    + var(2554) * SymbExpr::from_u32(256)
                                    + var(2555) * SymbExpr::from_u32(65536)
                                    + var(2556) * SymbExpr::from_u32(16777216),
                                var(2557)
                                    + var(2558) * SymbExpr::from_u32(256)
                                    + var(2559) * SymbExpr::from_u32(65536)
                                    + var(2560) * SymbExpr::from_u32(16777216),
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

            let mut state_transition_values_from_claims = vec![];

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

                    9u64 => {
                        /* This is our StateTransition claim. We should have chip_idx, state_in[32], state_out[32] */
                        assert!(claim.len() == 65, "[StateTransition] wrong claim format");

                        let state_in: [u32; 32] = array::from_fn(|i| {
                            u32::try_from(claim[i + 1].as_canonical_u64()).unwrap()
                        });
                        let state_out: [u32; 32] = array::from_fn(|i| {
                            u32::try_from(claim[i + 1 + 32].as_canonical_u64()).unwrap()
                        });

                        state_transition_values_from_claims.push((state_in, state_out));
                    }

                    _ => panic!("unsupported chip"),
                }
            }

            // Build traces. If claim for a given chip was not provided (and hence no data available), we just use zero trace
            // and balance lookups providing zero values

            let mut state_transition_trace_values =
                Vec::<Val>::with_capacity(state_transition_values_from_claims.len());
            if state_transition_values_from_claims.is_empty() {
                state_transition_trace_values = Val::zero_vec(STATE_TRANSITION_TRACE_WIDTH);

                // TODO: lookups for padded rows
            } else {
                for (state_in_io, state_out_io) in state_transition_values_from_claims.into_iter() {
                    let state_in_io_bytes = state_in_io
                        .into_iter()
                        .flat_map(|u32_in_io| u32_in_io.to_le_bytes())
                        .collect::<Vec<u8>>();
                    state_transition_trace_values.push(Val::ONE); // multiplicity
                    state_transition_trace_values.extend_from_slice(
                        state_in_io_bytes
                            .into_iter()
                            .map(|byte| Val::from_u8(byte))
                            .collect::<Vec<Val>>()
                            .as_slice(),
                    );

                    const MSG_PERMUTATION: [usize; 16] =
                        [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

                    let a = [0, 1, 2, 3, 0, 1, 2, 3];
                    let b = [4, 5, 6, 7, 5, 6, 7, 4];
                    let c = [8, 9, 10, 11, 10, 11, 8, 9];
                    let d = [12, 13, 14, 15, 15, 12, 13, 14];
                    let mx = [16, 18, 20, 22, 24, 26, 28, 30];
                    let my = [17, 19, 21, 23, 25, 27, 29, 31];

                    let mut state = state_in_io;
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

                            g_function_values_from_claims
                                .push((a_in, b_in, c_in, d_in, mx_in, my_in, a_1, b_1, c_1, d_1)); // send data to G_Function chip

                            state[a[j]] = a_1;
                            state[b[j]] = b_1;
                            state[c[j]] = c_1;
                            state[d[j]] = d_1;

                            let a_in_bytes: [u8; 4] = a_in.to_le_bytes();
                            let b_in_bytes: [u8; 4] = b_in.to_le_bytes();
                            let c_in_bytes: [u8; 4] = c_in.to_le_bytes();
                            let d_in_bytes: [u8; 4] = d_in.to_le_bytes();
                            let mx_in_bytes: [u8; 4] = mx_in.to_le_bytes();
                            let my_in_bytes: [u8; 4] = my_in.to_le_bytes();
                            let a_1_bytes: [u8; 4] = a_1.to_le_bytes();
                            let d_1_bytes: [u8; 4] = d_1.to_le_bytes();
                            let c_1_bytes: [u8; 4] = c_1.to_le_bytes();
                            let b_1_bytes: [u8; 4] = b_1.to_le_bytes();

                            state_transition_trace_values
                                .extend_from_slice(a_in_bytes.map(Val::from_u8).as_slice());
                            state_transition_trace_values
                                .extend_from_slice(b_in_bytes.map(Val::from_u8).as_slice());
                            state_transition_trace_values
                                .extend_from_slice(c_in_bytes.map(Val::from_u8).as_slice());
                            state_transition_trace_values
                                .extend_from_slice(d_in_bytes.map(Val::from_u8).as_slice());
                            state_transition_trace_values
                                .extend_from_slice(mx_in_bytes.map(Val::from_u8).as_slice());
                            state_transition_trace_values
                                .extend_from_slice(my_in_bytes.map(Val::from_u8).as_slice());
                            state_transition_trace_values
                                .extend_from_slice(a_1_bytes.map(Val::from_u8).as_slice());
                            state_transition_trace_values
                                .extend_from_slice(d_1_bytes.map(Val::from_u8).as_slice());
                            state_transition_trace_values
                                .extend_from_slice(c_1_bytes.map(Val::from_u8).as_slice());
                            state_transition_trace_values
                                .extend_from_slice(b_1_bytes.map(Val::from_u8).as_slice());
                        }

                        // execute permutation for the 6 first rounds
                        if round_idx < 6 {
                            let mut permuted = [0; 16];
                            for i in 0..16 {
                                permuted[i] = state[16 + MSG_PERMUTATION[i]];
                            }
                            for i in 0..16 {
                                state[i + 16] = permuted[i];
                            }
                        }
                    }

                    for i in 0..8 {
                        let a = state[i];
                        let b = state[i + 8];
                        state[i] ^= state[i + 8]; // ^ state[i + 8]
                        let xor = state[i];

                        // save (state[i]), (state[i + 8]) and (state[i] ^ state[i + 8]) to StateTransition trace for looking up
                        let a_bytes: [u8; 4] = a.to_le_bytes();
                        let b_bytes: [u8; 4] = b.to_le_bytes();
                        let xor_bytes: [u8; 4] = xor.to_le_bytes();

                        state_transition_trace_values
                            .extend_from_slice(a_bytes.map(Val::from_u8).as_slice());
                        state_transition_trace_values
                            .extend_from_slice(b_bytes.map(Val::from_u8).as_slice());
                        state_transition_trace_values
                            .extend_from_slice(xor_bytes.map(Val::from_u8).as_slice());

                        u32_xor_values_from_claims.push((a, b, xor)); // send data to U32Xor chip

                        let a = state[i + 8];
                        let b = state_in_io[i];
                        state[i + 8] ^= state_in_io[i]; // ^ chaining_value[i]
                        let xor = state[i + 8];

                        // save (state[i + 8]), (state_in_io[i]) and (state[i + 8] ^ state_in_io[i]) to StateTransition trace for looking up
                        let a_bytes: [u8; 4] = a.to_le_bytes();
                        let b_bytes: [u8; 4] = b.to_le_bytes();
                        let xor_bytes: [u8; 4] = xor.to_le_bytes();

                        state_transition_trace_values
                            .extend_from_slice(a_bytes.map(Val::from_u8).as_slice());
                        state_transition_trace_values
                            .extend_from_slice(b_bytes.map(Val::from_u8).as_slice());
                        state_transition_trace_values
                            .extend_from_slice(xor_bytes.map(Val::from_u8).as_slice());

                        u32_xor_values_from_claims.push((a, b, xor)); // send data to U32Xor chip
                    }

                    debug_assert_eq!(state_out_io, state);
                    let state_out_io_bytes = state_out_io
                        .into_iter()
                        .flat_map(|u32_out_io| u32_out_io.to_le_bytes())
                        .collect::<Vec<u8>>();
                    state_transition_trace_values.extend_from_slice(
                        state_out_io_bytes
                            .into_iter()
                            .map(|byte| Val::from_u8(byte))
                            .collect::<Vec<Val>>()
                            .as_slice(),
                    );
                }
            }
            let mut state_transition_trace =
                RowMajorMatrix::new(state_transition_trace_values, STATE_TRANSITION_TRACE_WIDTH);
            let height = state_transition_trace.height().next_power_of_two();
            let zero_rows_added = height - state_transition_trace.height();
            for _ in 0..zero_rows_added {
                // TODO: lookups for padded rows
            }
            state_transition_trace.pad_to_height(height, Val::ZERO);

            // build GFunction trace columns:
            // multiplicity, a_in(4), b_in(4), c_in(4), d_in(4), mx_in(4), my_in(4),
            // a_0_tmp(4), a_0(4), d_0_tmp(4), d_0(4), c_0(4), b_0_tmp(4), b_0(4),
            // a_1_tmp(4), a_1(4), d_1_tmp(4), d_1(4), c_1(4), b_1_tmp(4), b_1(4))
            let mut g_function_trace_values =
                Vec::<Val>::with_capacity(g_function_values_from_claims.len());
            if g_function_values_from_claims.is_empty() {
                g_function_trace_values = Val::zero_vec(G_FUNCTION_TRACE_WIDHT);

                // 1 rot7
                u32_rotate_right_7_values_from_claims.push((0u32, 0u32));
                // 1 rot8
                u32_rotate_right_8_values_from_claims.push((0u32, 0u32));
                // 1 rot16
                u32_rotate_right_16_values_from_claims.push((0u32, 0u32));
                // 1 rot12
                u32_rotate_right_12_values_from_claims.push((0u32, 0u32));

                // 4 u32_xor
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));

                // 6 u32_add
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
            } else {
                for (a_in, b_in, c_in, d_in, mx_in, my_in, a1, b1, c1, d1) in
                    g_function_values_from_claims.into_iter()
                {
                    let a_0_tmp = a_in.wrapping_add(b_in);
                    u32_add_values_from_claims.push((a_in, b_in, a_0_tmp)); // send data to U32Add chip

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
                    u32_add_values_from_claims.push((a_0, b_0, a_1_tmp)); // send data to U32Add chip

                    let a_1 = a_1_tmp.wrapping_add(my_in);
                    u32_add_values_from_claims.push((a_1_tmp, my_in, a_1)); // send data to U32Add chip

                    let d_1_tmp = d_0 ^ a_1;
                    u32_xor_values_from_claims.push((d_0, a_1, d_1_tmp)); // send data to U32Xor chip

                    let d_1 = d_1_tmp.rotate_right(8);
                    u32_rotate_right_8_values_from_claims.push((d_1_tmp, d_1));

                    let c_1 = c_0.wrapping_add(d_1);
                    u32_add_values_from_claims.push((c_0, d_1, c_1)); // send data to U32Add chip

                    let b_1_tmp = b_0 ^ c_1;
                    u32_xor_values_from_claims.push((b_0, c_1, b_1_tmp)); // send data to U32Xor chip

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
            let zero_rows_added = height - g_function_trace.height();
            for _ in 0..zero_rows_added {
                u32_rotate_right_7_values_from_claims.push((0u32, 0u32));
                u32_rotate_right_8_values_from_claims.push((0u32, 0u32));
                u32_rotate_right_16_values_from_claims.push((0u32, 0u32));
                u32_rotate_right_12_values_from_claims.push((0u32, 0u32));

                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                u32_xor_values_from_claims.push((0u32, 0u32, 0u32));

                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                u32_add_values_from_claims.push((0u32, 0u32, 0u32));
            }
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
            let zero_rows = height - u32_xor_trace.height();
            for _ in 0..zero_rows {
                // we also need to balance the U8Xor chip lookups using zeroes for every padded row
                byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
                byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
                byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
                byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
            }
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
            let zero_rows = height - u32_add_trace.height();
            for _ in 0..zero_rows {
                // we also need to balance the lookups using zeroes for every padded row
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));

                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            }
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
            let zero_rows = height - u32_rotate_right_8_trace.height();
            for _ in 0..zero_rows {
                // we also need to balance the lookups using zeroes for every padded row
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            }
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
            let zero_rows = height - u32_rotate_right_16_trace.height();
            for _ in 0..zero_rows {
                // we also need to balance the lookups using zeroes for every padded row
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
            }
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
                state_transition_trace,
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

        let state_transition_circuit = LookupAir::new(
            Blake3CompressionChips::StateTransition,
            Blake3CompressionChips::StateTransition.lookups(),
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
                state_transition_circuit,
            ],
        );

        let f = Val::from_u8;
        let f32 = Val::from_u32;

        // G_function IO
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

        // State transition IO
        let state_in = vec![
            0x00000000u32,
            0x00001111u32,
            0x00002222u32,
            0x00003333u32,
            0x00004444u32,
            0x00005555u32,
            0x00006666u32,
            0x00007777u32,
            0x00008888u32,
            0x00009999u32,
            0x0000aaaau32,
            0x0000bbbbu32,
            0x0000ccccu32,
            0x0000ddddu32,
            0x0000eeeeu32,
            0x0000ffffu32,
            0x00000000u32,
            0x11110000u32,
            0x22220000u32,
            0x33330000u32,
            0x44440000u32,
            0x55550000u32,
            0x66660000u32,
            0x77770000u32,
            0x88880000u32,
            0x99990000u32,
            0xaaaa0000u32,
            0xbbbb0000u32,
            0xcccc0000u32,
            0xdddd0000u32,
            0xeeee0000u32,
            0xffff0000u32,
        ];

        let state_out = vec![
            0xd304e51cu32,
            0xc2df34a0u32,
            0x5eba7f1fu32,
            0x2ab9650fu32,
            0xd9cef159u32,
            0x4e9d3a6au32,
            0xcac2e310u32,
            0xc6b9be7eu32,
            0xad9fd58au32,
            0x0899e71bu32,
            0xca51a599u32,
            0xc3fbd7c0u32,
            0x751d2f26u32,
            0x6cd0ac6bu32,
            0xc58f3c1du32,
            0xe6d65414u32,
            0xbbbb0000u32,
            0xffff0000u32,
            0x55550000u32,
            0x00000000u32,
            0x11110000u32,
            0x99990000u32,
            0x88880000u32,
            0x66660000u32,
            0xeeee0000u32,
            0xaaaa0000u32,
            0x22220000u32,
            0xcccc0000u32,
            0x33330000u32,
            0x44440000u32,
            0x77770000u32,
            0xdddd0000u32,
        ];

        let claims = Blake3CompressionClaims {
            claims: vec![
                // 3 u8 xor claims
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
                    Val::from_usize(Blake3CompressionChips::U8Xor.position()),
                    f(a1_u8),
                    f(b1_u8),
                    f(xor1_u8),
                ],
                // 5 u32 xor claims
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Xor.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(xor_u32),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Xor.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(xor_u32),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Xor.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(xor_u32),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Xor.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(xor_u32),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Xor.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(xor_u32),
                ],
                // 3 u32 addition claims
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Add.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(add_u32),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Add.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(add_u32),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Add.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(add_u32),
                ],
                // 3 u32 right rotate to 8 claims
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate8.position()),
                    f32(a_u32),
                    f32(a_rot_8),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate8.position()),
                    f32(a_u32),
                    f32(a_rot_8),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate8.position()),
                    f32(a_u32),
                    f32(a_rot_8),
                ],
                // 3 u32 right rotate to 16 claims
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate16.position()),
                    f32(a_u32),
                    f32(a_rot_16),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate16.position()),
                    f32(a_u32),
                    f32(a_rot_16),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate16.position()),
                    f32(a_u32),
                    f32(a_rot_16),
                ],
                // 3 u32 right rotate to 12 claims
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate12.position()),
                    f32(a_u32),
                    f32(a_rot_12),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate12.position()),
                    f32(a_u32),
                    f32(a_rot_12),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate12.position()),
                    f32(a_u32),
                    f32(a_rot_12),
                ],
                // 3 u32 right rotate to 7 claims
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate7.position()),
                    f32(a_u32),
                    f32(a_rot_7),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate7.position()),
                    f32(a_u32),
                    f32(a_rot_7),
                ],
                vec![
                    Val::from_usize(Blake3CompressionChips::U32RightRotate7.position()),
                    f32(a_u32),
                    f32(a_rot_7),
                ],
                // 6 G_function claims
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
                // 1 state transition claim
                vec![
                    vec![Val::from_usize(
                        Blake3CompressionChips::StateTransition.position(),
                    )],
                    state_in.into_iter().map(Val::from_u32).collect(),
                    state_out.into_iter().map(Val::from_u32).collect(),
                ]
                .concat(),
            ],
        };

        let (traces, witness) = claims.witness(&system);

        // // byte chip trace
        // println!("BYTE CHIP");
        // print_trace(&traces[0], U8_XOR_PAIR_RANGE_CHECK_TRACE_WIDTH, true);
        //
        // // u32 xor trace
        // println!("U32 XOR");
        // print_trace(&traces[1], U32_XOR_TRACE_WIDTH, false);
        //
        // // u32 add trace
        // println!("U32 ADD");
        // print_trace(&traces[2], U32_ADD_TRACE_WIDTH, false);
        //
        // // u32 rotate right 8 trace
        // println!("U32 RIGHT_ROTATE 8");
        // print_trace(&traces[3], U32_RIGHT_ROTATE_8_TRACE_WIDTH, false);
        //
        // // u32 rotate right 16 trace
        // println!("U32 RIGHT_ROTATE 16");
        // print_trace(&traces[4], U32_RIGHT_ROTATE_16_TRACE_WIDTH, false);
        //
        // // u32 rotate right 12 trace
        // println!("U32 RIGHT_ROTATE 12");
        // print_trace(&traces[5], U32_RIGHT_ROTATE_12_TRACE_WIDTH, false);
        //
        // // u32 rotate right 7 trace
        // println!("U32 RIGHT_ROTATE 7");
        // print_trace(&traces[6], U32_RIGHT_ROTATE_7_TRACE_WIDTH, false);
        //
        // // g function trace
        // println!("G_FUNCTION");
        // print_trace(&traces[7], G_FUNCTION_TRACE_WIDHT, false);

        // println!("STATE_TRANSITION");
        // print_trace(&traces[8], STATE_TRANSITION_TRACE_WIDTH, true);

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

    #[test]
    fn state_transition_test_vector() {
        let state_in = vec![
            0x00000000u32,
            0x00001111u32,
            0x00002222u32,
            0x00003333u32,
            0x00004444u32,
            0x00005555u32,
            0x00006666u32,
            0x00007777u32,
            0x00008888u32,
            0x00009999u32,
            0x0000aaaau32,
            0x0000bbbbu32,
            0x0000ccccu32,
            0x0000ddddu32,
            0x0000eeeeu32,
            0x0000ffffu32,
            0x00000000u32,
            0x11110000u32,
            0x22220000u32,
            0x33330000u32,
            0x44440000u32,
            0x55550000u32,
            0x66660000u32,
            0x77770000u32,
            0x88880000u32,
            0x99990000u32,
            0xaaaa0000u32,
            0xbbbb0000u32,
            0xcccc0000u32,
            0xdddd0000u32,
            0xeeee0000u32,
            0xffff0000u32,
        ];

        const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

        let a = [0, 1, 2, 3, 0, 1, 2, 3];
        let b = [4, 5, 6, 7, 5, 6, 7, 4];
        let c = [8, 9, 10, 11, 10, 11, 8, 9];
        let d = [12, 13, 14, 15, 15, 12, 13, 14];
        let mx = [16, 18, 20, 22, 24, 26, 28, 30];
        let my = [17, 19, 21, 23, 25, 27, 29, 31];

        let mut state = state_in.clone();
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

            // execute permutation for the 6 first rounds
            if round_idx < 6 {
                let mut permuted = [0; 16];
                for i in 0..16 {
                    permuted[i] = state[16 + MSG_PERMUTATION[i]];
                }
                for i in 0..16 {
                    state[i + 16] = permuted[i];
                }
            }
        }

        for i in 0..8 {
            state[i] ^= state[i + 8];
            state[i + 8] ^= state_in[i]; // ^chaining_value
        }

        let state_out = state;

        let state_out_expected = vec![
            0xd304e51cu32,
            0xc2df34a0u32,
            0x5eba7f1fu32,
            0x2ab9650fu32,
            0xd9cef159u32,
            0x4e9d3a6au32,
            0xcac2e310u32,
            0xc6b9be7eu32,
            0xad9fd58au32,
            0x0899e71bu32,
            0xca51a599u32,
            0xc3fbd7c0u32,
            0x751d2f26u32,
            0x6cd0ac6bu32,
            0xc58f3c1du32,
            0xe6d65414u32,
            0xbbbb0000u32,
            0xffff0000u32,
            0x55550000u32,
            0x00000000u32,
            0x11110000u32,
            0x99990000u32,
            0x88880000u32,
            0x66660000u32,
            0xeeee0000u32,
            0xaaaa0000u32,
            0x22220000u32,
            0xcccc0000u32,
            0x33330000u32,
            0x44440000u32,
            0x77770000u32,
            0xdddd0000u32,
        ];

        assert_eq!(state_out, state_out_expected);
    }

    // useful for debugging
    fn print_trace(trace: &RowMajorMatrix<Val>, width: usize, ignore_zeroes: bool) {
        println!();
        for row in trace.values.chunks(width) {
            if ignore_zeroes {
                if row.iter().all(|&x| x == Val::ZERO) {
                } else {
                    println!("{:?}", row);
                }
            } else {
                println!("{:?}", row);
            }
        }
        println!();
    }
}
