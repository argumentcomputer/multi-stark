#[cfg(test)]
mod tests {
    use crate::builder::symbolic::{preprocessed_var, var};
    use crate::chips::{SymbExpr, blake3_new_update_finalize};
    use crate::lookup::{Lookup, LookupAir};
    use crate::system::{System, SystemWitness};
    use crate::types::{CommitmentParameters, FriParameters, Val};
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use std::array;
    use std::ops::Range;

    // Blake3-specific constants

    const IV: [u32; 8] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
        0x5BE0CD19,
    ];
    const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
    const A: [usize; 8] = [0, 1, 2, 3, 0, 1, 2, 3];
    const B: [usize; 8] = [4, 5, 6, 7, 5, 6, 7, 4];
    const C: [usize; 8] = [8, 9, 10, 11, 10, 11, 8, 9];
    const D: [usize; 8] = [12, 13, 14, 15, 15, 12, 13, 14];
    const MX: [usize; 8] = [16, 18, 20, 22, 24, 26, 28, 30];
    const MY: [usize; 8] = [17, 19, 21, 23, 25, 27, 29, 31];

    // Circuit constants

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

    // multiplicity,
    // [state_in_0 (4), ... state_in_31 (4),
    // [a_in (4), b_in (4), c_in (4), d_in (4), mx_in (4), my_in (4), a_1 (4), b_1 (4), c_1 (4), d_1 (4)] (x56),
    // [state_i (4), state_i_8 (4), i_i8_xor (4)] (x8),
    // [state_i_8 (4), chaining_value_i (4), i_cv_xor (4)] (x8),
    // state_out_0 (4), ... state_out_16 (4)
    //
    // 1 + 32 * 4 + 40 * 56 + 12 * 8 * 2 + 16 * 4
    const COMPRESSION_TRACE_WIDTH: usize = 2625;

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
        Compression,
    }

    impl Blake3CompressionChips {
        fn position(&self) -> usize {
            match self {
                Self::U8Xor => 0,
                Self::U32Xor => 1,
                Self::U32Add => 2,
                Self::U32RightRotate8 => 3,
                Self::U32RightRotate16 => 4,
                Self::U32RightRotate12 => 5,
                Self::U32RightRotate7 => 6,
                Self::U8PairRangeCheck => 7,
                Self::GFunction => 8,
                Self::Compression => 9,
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
                Self::Compression => COMPRESSION_TRACE_WIDTH,
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
                Self::U32Xor
                | Self::U32Add
                | Self::U32RightRotate8
                | Self::U32RightRotate16
                | Self::U32RightRotate12
                | Self::U32RightRotate7
                | Self::GFunction
                | Self::Compression => None,
            }
        }
    }

    impl<AB> Air<AB> for Blake3CompressionChips
    where
        AB: AirBuilder,
        AB::Var: Copy,
        AB::F: Field,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                Self::U8Xor
                | Self::U8PairRangeCheck
                | Self::U32Xor
                | Self::U32RightRotate8
                | Self::U32RightRotate16
                | Self::GFunction => {}
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
                Self::Compression => {
                    let main = builder.main();
                    let columns = main.row_slice(0).unwrap();

                    let mut offset = 1usize;
                    let indices: [usize; 128] = array::from_fn(|i| i + 1);

                    let mut state = indices
                        .chunks(4)
                        .map(|word_indices| {
                            columns[word_indices[0]]
                                + columns[word_indices[1]] * AB::Expr::from_u32(256)
                                + columns[word_indices[2]] * AB::Expr::from_u32(256 * 256)
                                + columns[word_indices[3]] * AB::Expr::from_u32(256 * 256 * 256)
                        })
                        .collect::<Vec<_>>();
                    debug_assert_eq!(state.len(), 32);
                    offset += 128;

                    let mut a_in = vec![];
                    let mut b_in = vec![];
                    let mut c_in = vec![];
                    let mut d_in = vec![];
                    let mut mx_in = vec![];
                    let mut my_in = vec![];

                    let mut a_1 = vec![];
                    let mut b_1 = vec![];
                    let mut c_1 = vec![];
                    let mut d_1 = vec![];

                    for _ in 0..56 {
                        let a_in_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let b_in_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let c_in_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let d_in_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let mx_in_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let my_in_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;

                        let a_1_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let d_1_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let c_1_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let b_1_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;

                        a_in.push(a_in_i);
                        b_in.push(b_in_i);
                        c_in.push(c_in_i);
                        d_in.push(d_in_i);
                        mx_in.push(mx_in_i);
                        my_in.push(my_in_i);

                        a_1.push(a_1_i);
                        b_1.push(b_1_i);
                        c_1.push(c_1_i);
                        d_1.push(d_1_i);
                    }

                    let mut state_i = vec![];
                    let mut state_i_8 = vec![];
                    let mut i_i8_xor = vec![];
                    let mut state_i_8_copy = vec![];
                    let mut chaining_values = vec![];
                    let mut i_cv_xor = vec![];
                    let chaining_values_expected = state[0..8].to_vec();

                    for _ in 0..8 {
                        let state_i_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let state_i_8_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let i_i8_xor_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let state_i_8_updated_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let chaining_values_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;
                        let i_cv_xor_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;

                        state_i.push(state_i_i);
                        state_i_8.push(state_i_8_i);
                        i_i8_xor.push(i_i8_xor_i);
                        state_i_8_copy.push(state_i_8_updated_i);
                        chaining_values.push(chaining_values_i);
                        i_cv_xor.push(i_cv_xor_i);
                    }

                    let mut state_out = vec![];
                    for _ in 0..16 {
                        let state_out_i = columns[offset]
                            + columns[offset + 1] * AB::Expr::from_u32(256)
                            + columns[offset + 2] * AB::Expr::from_u32(256 * 256)
                            + columns[offset + 3] * AB::Expr::from_u32(256 * 256 * 256);
                        offset += 4;

                        state_out.push(state_out_i);
                    }

                    // check state_in <-> temp variables relation
                    let mut offset_2 = 0usize;
                    for round_idx in 0..7 {
                        for j in 0..8 {
                            let a_in_expected = state[A[j]].clone();
                            let b_in_expected = state[B[j]].clone();
                            let c_in_expected = state[C[j]].clone();
                            let d_in_expected = state[D[j]].clone();
                            let mx_in_expected = state[MX[j]].clone();
                            let my_in_expected = state[MY[j]].clone();

                            builder.assert_eq(a_in_expected.clone(), a_in[offset_2].clone());
                            builder.assert_eq(b_in_expected.clone(), b_in[offset_2].clone());
                            builder.assert_eq(c_in_expected.clone(), c_in[offset_2].clone());
                            builder.assert_eq(d_in_expected.clone(), d_in[offset_2].clone());
                            builder.assert_eq(mx_in_expected.clone(), mx_in[offset_2].clone());
                            builder.assert_eq(my_in_expected.clone(), my_in[offset_2].clone());

                            state[A[j]] = a_1[offset_2].clone();
                            state[B[j]] = b_1[offset_2].clone();
                            state[C[j]] = c_1[offset_2].clone();
                            state[D[j]] = d_1[offset_2].clone();

                            offset_2 += 1;
                        }
                        if round_idx < 6 {
                            let mut permuted = [AB::Expr::ZERO; 16];
                            for i in 0..16 {
                                permuted[i] = state[16 + MSG_PERMUTATION[i]].clone();
                            }
                            for i in 0..16 {
                                state[i + 16] = permuted[i].clone();
                            }
                        }
                    }

                    // check state_out <-> XOR variables relation
                    for i in 0..8 {
                        builder.assert_eq(state[i].clone(), state_i[i].clone());
                        builder.assert_eq(state[i + 8].clone(), state_i_8[i].clone());
                        builder.assert_eq(i_i8_xor[i].clone(), state_out[i].clone());

                        builder.assert_eq(state[i + 8].clone(), state_i_8_copy[i].clone()); // TODO: probably we can save one column in the traces
                        builder.assert_eq(
                            chaining_values_expected[i].clone(),
                            chaining_values[i].clone(),
                        );
                        builder.assert_eq(i_cv_xor[i].clone(), state_out[i + 8].clone());
                    }
                }
            }
        }
    }

    impl Blake3CompressionChips {
        fn lookups(&self) -> Vec<Lookup<SymbExpr>> {
            let u8_xor_idx = Self::U8Xor.position();
            let u32_xor_idx = Self::U32Xor.position();
            let u32_add_idx = Self::U32Add.position();
            let u32_right_rotate_8_idx = Self::U32RightRotate8.position();
            let u32_right_rotate_16_idx = Self::U32RightRotate16.position();
            let u32_right_rotate_12_idx = Self::U32RightRotate12.position();
            let u32_right_rotate_7_idx = Self::U32RightRotate7.position();
            let u8_pair_range_check_idx = Self::U8PairRangeCheck.position();
            let g_function_idx = Self::GFunction.position();
            let compression_idx = Self::Compression.position();

            fn pull_state_in_state_out(
                multiplicity: SymbExpr,
                chip_idx: usize,
                state_in_range: Range<usize>,
                state_out_range: Range<usize>,
                var: fn(usize) -> SymbExpr,
            ) -> Lookup<SymbExpr> {
                assert_eq!(state_in_range.len(), 128);
                assert_eq!(state_out_range.len(), 64);

                let in_i = state_in_range.collect::<Vec<usize>>();

                let out_i = state_out_range.collect::<Vec<usize>>();

                let state_in = in_i
                    .chunks(4)
                    .map(|i| {
                        var(i[0])
                            + var(i[1]) * SymbExpr::from_u32(256)
                            + var(i[2]) * SymbExpr::from_u32(65536)
                            + var(i[3]) * SymbExpr::from_u32(16777216)
                    })
                    .collect::<Vec<SymbExpr>>();

                let state_out = out_i
                    .chunks(4)
                    .map(|i| {
                        var(i[0])
                            + var(i[1]) * SymbExpr::from_u32(256)
                            + var(i[2]) * SymbExpr::from_u32(65536)
                            + var(i[3]) * SymbExpr::from_u32(16777216)
                    })
                    .collect::<Vec<SymbExpr>>();

                Lookup::pull(
                    multiplicity,
                    [vec![SymbExpr::from_usize(chip_idx)], state_in, state_out].concat(),
                )
            }

            fn push_round(
                multiplicity: SymbExpr,
                chip_idx: usize,
                v_ind: Range<usize>,
                var: fn(usize) -> SymbExpr,
            ) -> Lookup<SymbExpr> {
                assert_eq!(v_ind.len(), 40);

                let i = v_ind.collect::<Vec<usize>>();

                Lookup::push(
                    multiplicity,
                    vec![
                        SymbExpr::from_usize(chip_idx),
                        var(i[0])
                            + var(i[1]) * SymbExpr::from_u32(256)
                            + var(i[2]) * SymbExpr::from_u32(65536)
                            + var(i[3]) * SymbExpr::from_u32(16777216),
                        var(i[4])
                            + var(i[5]) * SymbExpr::from_u32(256)
                            + var(i[6]) * SymbExpr::from_u32(65536)
                            + var(i[7]) * SymbExpr::from_u32(16777216),
                        var(i[8])
                            + var(i[9]) * SymbExpr::from_u32(256)
                            + var(i[10]) * SymbExpr::from_u32(65536)
                            + var(i[11]) * SymbExpr::from_u32(16777216),
                        var(i[12])
                            + var(i[13]) * SymbExpr::from_u32(256)
                            + var(i[14]) * SymbExpr::from_u32(65536)
                            + var(i[15]) * SymbExpr::from_u32(16777216),
                        var(i[16])
                            + var(i[17]) * SymbExpr::from_u32(256)
                            + var(i[18]) * SymbExpr::from_u32(65536)
                            + var(i[19]) * SymbExpr::from_u32(16777216),
                        var(i[20])
                            + var(i[21]) * SymbExpr::from_u32(256)
                            + var(i[22]) * SymbExpr::from_u32(65536)
                            + var(i[23]) * SymbExpr::from_u32(16777216),
                        var(i[24])
                            + var(i[25]) * SymbExpr::from_u32(256)
                            + var(i[26]) * SymbExpr::from_u32(65536)
                            + var(i[27]) * SymbExpr::from_u32(16777216),
                        var(i[28])
                            + var(i[29]) * SymbExpr::from_u32(256)
                            + var(i[30]) * SymbExpr::from_u32(65536)
                            + var(i[31]) * SymbExpr::from_u32(16777216),
                        var(i[32])
                            + var(i[33]) * SymbExpr::from_u32(256)
                            + var(i[34]) * SymbExpr::from_u32(65536)
                            + var(i[35]) * SymbExpr::from_u32(16777216),
                        var(i[36])
                            + var(i[37]) * SymbExpr::from_u32(256)
                            + var(i[38]) * SymbExpr::from_u32(65536)
                            + var(i[39]) * SymbExpr::from_u32(16777216),
                    ],
                )
            }

            fn push_u32(
                multiplicity: SymbExpr,
                chip_idx: usize,
                v_ind: Range<usize>,
                var: fn(usize) -> SymbExpr,
            ) -> Lookup<SymbExpr> {
                lookup_u32_inner(Lookup::push, multiplicity, chip_idx, v_ind, var)
            }

            fn pull_u32(
                multiplicity: SymbExpr,
                chip_idx: usize,
                v_ind: Range<usize>,
                var: fn(usize) -> SymbExpr,
            ) -> Lookup<SymbExpr> {
                lookup_u32_inner(Lookup::pull, multiplicity, chip_idx, v_ind, var)
            }

            fn lookup_u32_inner(
                lookup_fn: fn(SymbExpr, Vec<SymbExpr>) -> Lookup<SymbExpr>,
                multiplicity: SymbExpr,
                chip_idx: usize,
                v_ind: Range<usize>,
                var: fn(usize) -> SymbExpr,
            ) -> Lookup<SymbExpr> {
                assert_eq!(v_ind.len(), 12);

                let i = v_ind.collect::<Vec<usize>>();

                lookup_fn(
                    multiplicity,
                    vec![
                        SymbExpr::from_usize(chip_idx),
                        var(i[0])
                            + var(i[1]) * SymbExpr::from_u32(256)
                            + var(i[2]) * SymbExpr::from_u32(256 * 256)
                            + var(i[3]) * SymbExpr::from_u32(256 * 256 * 256),
                        var(i[4])
                            + var(i[5]) * SymbExpr::from_u32(256)
                            + var(i[6]) * SymbExpr::from_u32(256 * 256)
                            + var(i[7]) * SymbExpr::from_u32(256 * 256 * 256),
                        var(i[8])
                            + var(i[9]) * SymbExpr::from_u32(256)
                            + var(i[10]) * SymbExpr::from_u32(256 * 256)
                            + var(i[11]) * SymbExpr::from_u32(256 * 256 * 256),
                    ],
                )
            }

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
                    let mut lookups = vec![pull_u32(var(0), u32_xor_idx, 1..12 + 1, var)];

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
                    let mut lookups = vec![pull_u32(var(13), u32_add_idx, 0..11 + 1, var)];

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
                                // note indices!
                                var(57) // a_1
                                    + var(58) * SymbExpr::from_u32(256)
                                    + var(59) * SymbExpr::from_u32(256 * 256)
                                    + var(60) * SymbExpr::from_u32(256 * 256 * 256),
                                var(65) // d_1
                                    + var(66) * SymbExpr::from_u32(256)
                                    + var(67) * SymbExpr::from_u32(256 * 256)
                                    + var(68) * SymbExpr::from_u32(256 * 256 * 256),
                                var(69) // c_1
                                    + var(70) * SymbExpr::from_u32(256)
                                    + var(71) * SymbExpr::from_u32(256 * 256)
                                    + var(72) * SymbExpr::from_u32(256 * 256 * 256),
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
                Self::Compression => {
                    vec![
                        // pulling state_in / state_out (to balance initial claim)
                        pull_state_in_state_out(
                            var(0),
                            compression_idx,
                            1..128 + 1,
                            2561..2624 + 1,
                            var,
                        ),
                        // pushing data for 56 rounds of g_function
                        push_round(SymbExpr::ONE, g_function_idx, 129..168 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 169..208 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 209..248 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 249..288 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 289..328 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 329..368 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 369..408 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 409..448 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 449..488 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 489..528 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 529..568 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 569..608 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 609..648 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 649..688 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 689..728 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 729..768 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 769..808 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 809..848 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 849..888 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 889..928 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 929..968 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 969..1008 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1009..1048 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1049..1088 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1089..1128 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1129..1168 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1169..1208 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1209..1248 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1249..1288 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1289..1328 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1329..1368 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1369..1408 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1409..1448 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1449..1488 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1489..1528 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1529..1568 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1569..1608 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1609..1648 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1649..1688 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1689..1728 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1729..1768 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1769..1808 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1809..1848 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1849..1888 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1889..1928 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1929..1968 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 1969..2008 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 2009..2048 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 2049..2088 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 2089..2128 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 2129..2168 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 2169..2208 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 2209..2248 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 2249..2288 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 2289..2328 + 1, var),
                        push_round(SymbExpr::ONE, g_function_idx, 2329..2368 + 1, var),
                        // pushing data for state[i] ^= state[i + 8] operation
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2369..2380 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2381..2392 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2393..2404 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2405..2416 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2417..2428 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2429..2440 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2441..2452 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2453..2464 + 1, var),
                        //  pushing data for  state[i + 8] ^= chaining_value[i] operation
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2465..2476 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2477..2488 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2489..2500 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2501..2512 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2513..2524 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2525..2536 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2537..2548 + 1, var),
                        push_u32(SymbExpr::ONE, u32_xor_idx, 2549..2560 + 1, var),
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
                assert!(!claim.is_empty(), "wrong claim format");
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
                        /* This is our StateTransition claim. We should have chip_idx, state_in[32], state_out[16] */
                        assert!(claim.len() == 49, "[StateTransition] wrong claim format");

                        let state_in: [u32; 32] = array::from_fn(|i| {
                            u32::try_from(claim[i + 1].as_canonical_u64()).unwrap()
                        });
                        let state_out: [u32; 16] = array::from_fn(|i| {
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
                state_transition_trace_values = Val::zero_vec(COMPRESSION_TRACE_WIDTH);
                for _ in 0..56 {
                    g_function_values_from_claims
                        .push((0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32));
                }

                for _ in 0..8 {
                    u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                    u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                }
            } else {
                for (state_in_io, state_out_io) in state_transition_values_from_claims {
                    let state_in_io_bytes = state_in_io
                        .into_iter()
                        .flat_map(|u32_in_io| u32_in_io.to_le_bytes())
                        .collect::<Vec<u8>>();
                    state_transition_trace_values.push(Val::ONE); // multiplicity
                    state_transition_trace_values.extend_from_slice(
                        state_in_io_bytes
                            .into_iter()
                            .map(Val::from_u8)
                            .collect::<Vec<Val>>()
                            .as_slice(),
                    );

                    let mut state = state_in_io;
                    for round_idx in 0..7 {
                        for j in 0..8 {
                            let a_in = state[A[j]];
                            let b_in = state[B[j]];
                            let c_in = state[C[j]];
                            let d_in = state[D[j]];
                            let mx_in = state[MX[j]];
                            let my_in = state[MY[j]];

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

                            state[A[j]] = a_1;
                            state[B[j]] = b_1;
                            state[C[j]] = c_1;
                            state[D[j]] = d_1;

                            for u32_val in
                                [a_in, b_in, c_in, d_in, mx_in, my_in, a_1, d_1, c_1, b_1].iter()
                            {
                                let bytes: [u8; 4] = u32_val.to_le_bytes();
                                state_transition_trace_values
                                    .extend_from_slice(bytes.map(Val::from_u8).as_slice());
                            }
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
                        let left = state[i];
                        let right = state[i + 8];
                        state[i] ^= state[i + 8]; // ^ state[i + 8]
                        let xor = state[i];

                        // save (state[i]), (state[i + 8]) and (state[i] ^ state[i + 8]) to StateTransition trace for looking up
                        let left_bytes: [u8; 4] = left.to_le_bytes();
                        let right_bytes: [u8; 4] = right.to_le_bytes();
                        let xor_bytes: [u8; 4] = xor.to_le_bytes();

                        state_transition_trace_values
                            .extend_from_slice(left_bytes.map(Val::from_u8).as_slice());
                        state_transition_trace_values
                            .extend_from_slice(right_bytes.map(Val::from_u8).as_slice());
                        state_transition_trace_values
                            .extend_from_slice(xor_bytes.map(Val::from_u8).as_slice());

                        u32_xor_values_from_claims.push((left, right, xor)); // send data to U32Xor chip

                        let left = state[i + 8];
                        let right = state_in_io[i];
                        state[i + 8] ^= state_in_io[i]; // ^ chaining_value[i]
                        let xor = state[i + 8];

                        // save (state[i + 8]), (state_in_io[i]) and (state[i + 8] ^ state_in_io[i]) to StateTransition trace for looking up
                        let left_bytes: [u8; 4] = left.to_le_bytes();
                        let right_bytes: [u8; 4] = right.to_le_bytes();
                        let xor_bytes: [u8; 4] = xor.to_le_bytes();

                        state_transition_trace_values
                            .extend_from_slice(left_bytes.map(Val::from_u8).as_slice());
                        state_transition_trace_values
                            .extend_from_slice(right_bytes.map(Val::from_u8).as_slice());
                        state_transition_trace_values
                            .extend_from_slice(xor_bytes.map(Val::from_u8).as_slice());

                        u32_xor_values_from_claims.push((left, right, xor)); // send data to U32Xor chip
                    }

                    let mut state_out = state.to_vec();
                    state_out.truncate(16); // compression output is first 16 u32 words of state_out

                    debug_assert_eq!(state_out_io.to_vec(), state_out);
                    let state_out_io_bytes = state_out_io
                        .into_iter()
                        .flat_map(|u32_out_io| u32_out_io.to_le_bytes())
                        .collect::<Vec<u8>>();
                    state_transition_trace_values.extend_from_slice(
                        state_out_io_bytes
                            .into_iter()
                            .map(Val::from_u8)
                            .collect::<Vec<Val>>()
                            .as_slice(),
                    );
                }
            }
            let mut state_transition_trace =
                RowMajorMatrix::new(state_transition_trace_values, COMPRESSION_TRACE_WIDTH);
            let height = state_transition_trace.height().next_power_of_two();
            let zero_rows_added = height - state_transition_trace.height();
            for _ in 0..zero_rows_added {
                // we have 56 communications with G_Function chip
                for _ in 0..56 {
                    g_function_values_from_claims
                        .push((0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32));
                }

                // we have 8 * 2 communications with U32_XOR chip
                for _ in 0..8 {
                    u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                    u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                }
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
                for _ in 0..4 {
                    u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                }

                // 6 u32_add
                for _ in 0..6 {
                    u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                }
            } else {
                for (a_in, b_in, c_in, d_in, mx_in, my_in, a1, b1, c1, d1) in
                    g_function_values_from_claims
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

                    g_function_trace_values.push(Val::ONE); // multiplicity
                    for u32_val in [
                        a_in, b_in, c_in, d_in, mx_in, my_in, a_0_tmp, a_0, d_0_tmp, d_0, c_0,
                        b_0_tmp, b_0, a_1_tmp, a_1, d_1_tmp, d_1, c_1, b_1_tmp, b_1,
                    ]
                    .iter()
                    {
                        let bytes: [u8; 4] = u32_val.to_le_bytes();
                        g_function_trace_values
                            .extend_from_slice(bytes.map(Val::from_u8).as_slice());
                    }
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

                for _ in 0..4 {
                    u32_xor_values_from_claims.push((0u32, 0u32, 0u32));
                }

                for _ in 0..6 {
                    u32_add_values_from_claims.push((0u32, 0u32, 0u32));
                }
            }
            g_function_trace.pad_to_height(height, Val::ZERO);

            // build U32Xor trace (columns: multiplicity, A0, A1, A2, A3, B0, B1, B2, B3, A0^B0, A1^B1, A2^B2, A3^B3)
            let mut u32_xor_trace_values =
                Vec::<Val>::with_capacity(u32_xor_values_from_claims.len());
            if u32_xor_values_from_claims.is_empty() {
                u32_xor_trace_values = Val::zero_vec(U32_XOR_TRACE_WIDTH);

                // we also need to balance the U8Xor chip lookups using zeroes

                for _ in 0..4 {
                    byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
                }
            } else {
                for (left, right, xor) in u32_xor_values_from_claims {
                    debug_assert_eq!(left ^ right, xor);

                    let left_bytes: [u8; 4] = left.to_le_bytes();
                    let right_bytes: [u8; 4] = right.to_le_bytes();
                    let xor_bytes: [u8; 4] = xor.to_le_bytes();

                    u32_xor_trace_values.push(Val::ONE); // multiplicity

                    u32_xor_trace_values.extend_from_slice(left_bytes.map(Val::from_u8).as_slice());
                    u32_xor_trace_values
                        .extend_from_slice(right_bytes.map(Val::from_u8).as_slice());
                    u32_xor_trace_values.extend_from_slice(xor_bytes.map(Val::from_u8).as_slice());

                    /* we send bytes to U8Xor chip, relying on lookup constraining */

                    for i in 0..4 {
                        byte_xor_values_from_claims.push((
                            Val::from_u8(left_bytes[i]),
                            Val::from_u8(right_bytes[i]),
                            Val::from_u8(xor_bytes[i]),
                        ));
                    }
                }
            }
            let mut u32_xor_trace = RowMajorMatrix::new(u32_xor_trace_values, U32_XOR_TRACE_WIDTH);
            let height = u32_xor_trace.height().next_power_of_two();
            let zero_rows = height - u32_xor_trace.height();
            for _ in 0..zero_rows {
                // we also need to balance the U8Xor chip lookups using zeroes for every padded row
                for _ in 0..4 {
                    byte_xor_values_from_claims.push((Val::ZERO, Val::ZERO, Val::ZERO));
                }
            }
            u32_xor_trace.pad_to_height(height, Val::ZERO);

            // build U32Add trace (columns: A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3, carry, multiplicity)
            let mut u32_add_trace_values = vec![];
            if u32_add_values_from_claims.is_empty() {
                u32_add_trace_values = Val::zero_vec(U32_ADD_TRACE_WIDTH);

                // we also need to balance the lookups using zeroes

                for _ in 0..8 {
                    byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                }
            } else {
                for (left, right, sum) in u32_add_values_from_claims {
                    let (z, carry) = left.overflowing_add(right);
                    // actual addition result should match to the value from claim
                    debug_assert_eq!(z, sum);

                    let left_bytes: [u8; 4] = left.to_le_bytes();
                    let right_bytes: [u8; 4] = right.to_le_bytes();
                    let sum_bytes: [u8; 4] = sum.to_le_bytes();

                    u32_add_trace_values.extend_from_slice(left_bytes.map(Val::from_u8).as_slice());
                    u32_add_trace_values
                        .extend_from_slice(right_bytes.map(Val::from_u8).as_slice());
                    u32_add_trace_values.extend_from_slice(sum_bytes.map(Val::from_u8).as_slice());

                    u32_add_trace_values.push(Val::from_bool(carry));
                    u32_add_trace_values.push(Val::ONE); // multiplicity

                    /* we send decomposed bytes to U8Xor chip, relying on lookup constraining */

                    for i in 0..4 {
                        byte_range_check_values_from_claims
                            .push((Val::from_u8(left_bytes[i]), Val::from_u8(right_bytes[i])));
                        byte_range_check_values_from_claims
                            .push((Val::from_u8(sum_bytes[i]), Val::ZERO));
                    }
                }
            }
            let mut u32_add_trace = RowMajorMatrix::new(u32_add_trace_values, U32_ADD_TRACE_WIDTH);
            let height = u32_add_trace.height().next_power_of_two();
            let zero_rows = height - u32_add_trace.height();
            for _ in 0..zero_rows {
                // we also need to balance the lookups using zeroes for every padded row

                for _ in 0..8 {
                    byte_range_check_values_from_claims.push((Val::ZERO, Val::ZERO));
                }
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
                for (val, rot) in u32_rotate_right_8_values_from_claims {
                    u32_rotate_right_8_trace_values.push(Val::ONE); // multiplicity

                    // actual rotate 8 result should match to the value from claim
                    debug_assert_eq!(val.rotate_right(8), rot);

                    let val_bytes: [u8; 4] = val.to_le_bytes();
                    let rot_bytes: [u8; 4] = rot.to_le_bytes();

                    u32_rotate_right_8_trace_values
                        .extend_from_slice(val_bytes.map(Val::from_u8).as_slice());
                    u32_rotate_right_8_trace_values
                        .extend_from_slice(rot_bytes.map(Val::from_u8).as_slice());

                    /* we send decomposed bytes to U8PairRangeCheck chip, relying on lookup constraining */

                    byte_range_check_values_from_claims
                        .push((Val::from_u8(val_bytes[0]), Val::from_u8(val_bytes[2])));
                    byte_range_check_values_from_claims
                        .push((Val::from_u8(val_bytes[1]), Val::from_u8(val_bytes[3])));
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
                for (val, rot) in u32_rotate_right_16_values_from_claims {
                    u32_rotate_right_16_trace_values.push(Val::ONE); // multiplicity

                    // actual rotate 16 result should match to the value from claim
                    debug_assert_eq!(val.rotate_right(16), rot);

                    let a_bytes: [u8; 4] = val.to_le_bytes();
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
                vals_from_claim: &[(u32, u32)],
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
                    for (val, rot) in vals_from_claim {
                        values.push(Val::ONE); // multiplicity

                        // actual rotate result should match to the value from claim
                        debug_assert_eq!(val.rotate_right(k), *rot);

                        let two_pow_k = u32::try_from(2usize.pow(k)).unwrap();
                        let two_pow_32_minus_k = u32::try_from(2usize.pow(32 - k)).unwrap();

                        let input_div = val / two_pow_k;
                        let input_rem = val % two_pow_k;

                        let two_pow_k_bytes: [u8; 4] = two_pow_k.to_le_bytes();
                        let two_pow_32_minus_k_bytes: [u8; 4] = two_pow_32_minus_k.to_le_bytes();
                        let input_div_bytes: [u8; 4] = input_div.to_le_bytes();
                        let input_rem_bytes: [u8; 4] = input_rem.to_le_bytes();

                        let val_bytes: [u8; 4] = val.to_le_bytes();
                        let rot_bytes: [u8; 4] = rot.to_le_bytes();

                        values.extend_from_slice(val_bytes.map(Val::from_u8).as_slice());
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

            // build U32RotateRight12 trace
            let u32_rotate_right_12_trace =
                rot_7_12_trace_values(12, &u32_rotate_right_12_values_from_claims);

            // build U32RotateRight7 trace
            let u32_rotate_right_7_trace =
                rot_7_12_trace_values(7, &u32_rotate_right_7_values_from_claims);

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
    fn test_compression_reference_compatibility() {
        let input: Vec<u8> = vec![0x54; 64];

        let (claim_data, expected) = blake3_new_update_finalize(&input);
        assert_eq!(claim_data.len(), 1);

        let claim_data = claim_data.first().unwrap();

        let state_in = [
            claim_data.cv.to_vec(),
            vec![
                IV[0],
                IV[1],
                IV[2],
                IV[3],
                claim_data.counter_low,
                claim_data.counter_high,
                claim_data.block_len,
                claim_data.flags,
            ],
            claim_data.block_words.to_vec(),
        ]
        .concat();

        let state_out = claim_data.output.to_vec();

        let mut actual = state_out
            .clone()
            .into_iter()
            .flat_map(|u32_val| u32_val.to_le_bytes())
            .collect::<Vec<u8>>();
        actual.truncate(32);

        assert_eq!(actual, expected.to_vec());

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
            Blake3CompressionChips::Compression,
            Blake3CompressionChips::Compression.lookups(),
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

        let claims = Blake3CompressionClaims {
            claims: vec![
                [
                    vec![Val::from_usize(
                        Blake3CompressionChips::Compression.position(),
                    )],
                    state_in.into_iter().map(Val::from_u32).collect(),
                    state_out.into_iter().map(Val::from_u32).collect(),
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
    fn test_all_claims() {
        // computations IO

        let a_u8 = 0xa1u8;
        let b_u8 = 0xa8u8;
        let xor_u8 = a_u8 ^ b_u8;

        let a_u32 = 0x000000ffu32;
        let b_u32 = 0x0000ff01u32;
        let xor_u32 = a_u32 ^ b_u32;
        let add_u32 = a_u32.wrapping_add(b_u32);
        let a_rot_8 = a_u32.rotate_right(8);
        let a_rot_16 = a_u32.rotate_right(16);
        let a_rot_12 = a_u32.rotate_right(12);
        let a_rot_7 = a_u32.rotate_right(7);

        // G function IO
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

        // compression IO
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
        ];

        fn run_test(claims: &Blake3CompressionClaims) {
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
                Blake3CompressionChips::Compression,
                Blake3CompressionChips::Compression.lookups(),
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

        // claims construction
        let f = Val::from_u8;
        let f32 = Val::from_u32;

        let claims = Blake3CompressionClaims {
            claims: vec![
                // 3 u8 xor claims
                vec![
                    Val::from_usize(Blake3CompressionChips::U8Xor.position()),
                    f(a_u8),
                    f(b_u8),
                    f(xor_u8),
                ],
            ],
        };

        run_test(&claims);

        let claims = Blake3CompressionClaims {
            claims: vec![
                // 5 u8 xor claims
                vec![
                    Val::from_usize(Blake3CompressionChips::U8Xor.position()),
                    f(a_u8),
                    f(b_u8),
                    f(xor_u8),
                ]; 5
            ],
        };
        run_test(&claims);

        let claims = Blake3CompressionClaims {
            claims: vec![
                // 2 u32 xor claims
                vec![
                    Val::from_usize(Blake3CompressionChips::U8Xor.position()),
                    f(a_u8),
                    f(b_u8),
                    f(xor_u8),
                ]; 2
            ],
        };
        run_test(&claims);

        let claims = Blake3CompressionClaims {
            claims: vec![
                // 3 u32 xor claims
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Xor.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(xor_u32),
                ]; 3
            ],
        };
        run_test(&claims);

        let claims = Blake3CompressionClaims {
            claims: vec![
                // 6 u32 add claims
                vec![
                    Val::from_usize(Blake3CompressionChips::U32Add.position()),
                    f32(a_u32),
                    f32(b_u32),
                    f32(add_u32),
                ]; 6
            ],
        };
        run_test(&claims);

        let claims = Blake3CompressionClaims {
            claims: vec![
                // Right rotate claims (1 per each operation)
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
            ],
        };
        run_test(&claims);

        let claims = Blake3CompressionClaims {
            // 3 G function claims
            claims: vec![
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
                ];
                3
            ],
        };
        run_test(&claims);

        let claims = Blake3CompressionClaims {
            // 11 Compression claims
            claims: vec![
                [
                    vec![Val::from_usize(
                        Blake3CompressionChips::Compression.position(),
                    )],
                    state_in.clone().into_iter().map(Val::from_u32).collect(),
                    state_out.clone().into_iter().map(Val::from_u32).collect(),
                ]
                .concat();
                5
            ],
        };
        run_test(&claims);

        let claims = Blake3CompressionClaims {
            // Compression + G_Function claims
            claims: vec![
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
                [
                    vec![Val::from_usize(
                        Blake3CompressionChips::Compression.position(),
                    )],
                    state_in.clone().into_iter().map(Val::from_u32).collect(),
                    state_out.clone().into_iter().map(Val::from_u32).collect(),
                ]
                .concat(),
            ],
        };
        run_test(&claims);
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
    fn compression_test_vector() {
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

        let mut state = state_in.clone();
        for round_idx in 0..7 {
            for j in 0..8 {
                let a_in = state[A[j]];
                let b_in = state[B[j]];
                let c_in = state[C[j]];
                let d_in = state[D[j]];
                let mx_in = state[MX[j]];
                let my_in = state[MY[j]];

                let a_0 = a_in.wrapping_add(b_in).wrapping_add(mx_in);
                let d_0 = (d_in ^ a_0).rotate_right(16);
                let c_0 = c_in.wrapping_add(d_0);
                let b_0 = (b_in ^ c_0).rotate_right(12);

                let a_1 = a_0.wrapping_add(b_0).wrapping_add(my_in);
                let d_1 = (d_0 ^ a_1).rotate_right(8);
                let c_1 = c_0.wrapping_add(d_1);
                let b_1 = (b_0 ^ c_1).rotate_right(7);

                state[A[j]] = a_1;
                state[B[j]] = b_1;
                state[C[j]] = c_1;
                state[D[j]] = d_1;
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

        let state_out = state[0..16].to_vec();

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
        ];

        assert_eq!(state_out, state_out_expected);
    }
}
