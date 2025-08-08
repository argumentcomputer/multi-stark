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

    enum UtilityChip {
        U32FromLeBytes,
        U32Or,
        U8Or,
        U8PairRangeCheck,
    }

    impl UtilityChip {
        fn position(&self) -> usize {
            match self {
                Self::U32FromLeBytes => 0,
                Self::U32Or => 1,
                Self::U8Or => 2,
                Self::U8PairRangeCheck => 3,
            }
        }
    }

    impl<F: Field> BaseAir<F> for UtilityChip {
        fn width(&self) -> usize {
            match self {
                Self::U32FromLeBytes => U32_FROM_LE_BYTES_TRACE_WIDTH,
                Self::U32Or => todo!(),
                Self::U8Or | Self::U8PairRangeCheck => U8_OR_PAIR_RANGE_CHECK_TRACE_WIDTH,
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
                Self::U32FromLeBytes => None,
                Self::U32Or => None,
            }
        }
    }

    impl<AB> Air<AB> for UtilityChip
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, _builder: &mut AB) {
            match self {
                Self::U32FromLeBytes => {}
                Self::U32Or => {}
                Self::U8Or => {}
                Self::U8PairRangeCheck => {}
            }
        }
    }

    impl UtilityChip {
        fn lookups(&self) -> Vec<Lookup<SymbExpr>> {
            let u32_from_le_bytes_idx = Self::U32FromLeBytes.position();
            // let u32_or_idx = Self::U32Or.position();
            let u8_or_idx = Self::U8Or.position();
            let u8_pair_range_check_idx = Self::U8PairRangeCheck.position();

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
                    todo!();
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

            for claim in self.claims.clone() {
                // we should have at least chip index
                assert!(!claim.is_empty(), "wrong claim format");
                match claim[0].as_canonical_u64() {
                    0u64 => {
                        // this is out u32_from_le_bytes chip. We should have chip_idx, byte0, byte1, byte2, byte3, u32
                        assert_eq!(claim.len(), 6);

                        let byte0_val = u8::try_from(claim[1].as_canonical_u64()).unwrap();
                        let byte1_val = u8::try_from(claim[2].as_canonical_u64()).unwrap();
                        let byte2_val = u8::try_from(claim[3].as_canonical_u64()).unwrap();
                        let byte3_val = u8::try_from(claim[4].as_canonical_u64()).unwrap();

                        let u32_val = u32::try_from(claim[5].as_canonical_u64()).unwrap();

                        u32_from_le_bytes_values_from_claims
                            .push((byte0_val, byte1_val, byte2_val, byte3_val, u32_val));
                    }
                    1u64 => {
                        // u32 or
                        todo!();
                    }

                    2u64 => {
                        // This is our U8Or claim. We should have chip_idx, A, B, A or B (where A, B are bytes)
                        assert_eq!(claim.len(), 4, "[U8Or] wrong claim format");
                        byte_or_values_from_claims.push((claim[1], claim[2], claim[3]));
                    }

                    3u64 => {
                        /* This is our U8PairRangeCheck claim. We should have chip_idx, A, B */

                        assert_eq!(claim.len(), 3, "[U8PairRangeCheck] wrong claim format");
                        byte_range_check_values_from_claims.push((claim[1], claim[2]));
                    }

                    _ => {
                        panic!("unsupported chip")
                    }
                }
            }

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
                u32_from_le_bytes_trace,
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
        let u32_from_le_bytes_circuit = LookupAir::new(
            UtilityChip::U32FromLeBytes,
            UtilityChip::U32FromLeBytes.lookups(),
        );

        let (system, prover_key) = System::new(
            commitment_parameters,
            vec![u8_circuit, u32_from_le_bytes_circuit],
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
        let u32_from_le_bytes_circuit = LookupAir::new(
            UtilityChip::U32FromLeBytes,
            UtilityChip::U32FromLeBytes.lookups(),
        );

        let (system, prover_key) = System::new(
            commitment_parameters,
            vec![u8_circuit, u32_from_le_bytes_circuit],
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
}
