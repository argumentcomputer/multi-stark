#[cfg(test)]
mod tests {
    use std::array;
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
    use p3_matrix::dense::RowMajorMatrix;
    use crate::builder::symbolic::{Entry, SymbolicExpression, SymbolicVariable};
    use crate::chips::Expr;
    use crate::lookup::{Lookup, LookupAir};
    use crate::system::{System, SystemWitness};
    use crate::types::{CommitmentParameters, FriParameters, Val};

    // A, B, A ^ B (where A, B are bytes)
    const PREPROCESSED_TRACE_WIDTH: usize = 3;

    // multiplicity
    const U8XOR_TRACE_WIDTH: usize = 1;

    // multiplicity, a0, a1, a2, a3, b0, b1, b2, b3, a0^b0, a1^b1, a2^b2, a3^b3
    const U32XOR_TRACE_WIDTH: usize = 13;

    const BYTE_VALUES_NUM: usize = 256;

    enum Blake3CompressionChips {
        U8Xor,
        U32Xor,
    }

    impl Blake3CompressionChips {
        fn position(&self) -> usize {
            match self {
                Blake3CompressionChips::U8Xor => 0,
                Blake3CompressionChips::U32Xor => 1,
            }
        }
    }

    impl<F: Field> BaseAir<F> for Blake3CompressionChips {
        fn width(&self) -> usize {
            match self {
                Self::U8Xor => U8XOR_TRACE_WIDTH,
                Self::U32Xor => U32XOR_TRACE_WIDTH,
            }
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            let bytes: [u8; BYTE_VALUES_NUM] = array::from_fn(|idx| u8::try_from(idx).unwrap());
            let mut trace_values =
                Vec::with_capacity(BYTE_VALUES_NUM * BYTE_VALUES_NUM * PREPROCESSED_TRACE_WIDTH);
            for i in 0..BYTE_VALUES_NUM {
                for j in 0..BYTE_VALUES_NUM {
                    trace_values.push(F::from_u8(bytes[i]));
                    trace_values.push(F::from_u8(bytes[j]));
                    trace_values.push(F::from_u8(bytes[i] ^ bytes[j]));
                }
            }
            Some(RowMajorMatrix::new(trace_values, PREPROCESSED_TRACE_WIDTH))
        }
    }


    impl<AB> Air<AB> for Blake3CompressionChips
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, _builder: &mut AB) { /* no regular P3 constraints (we rely entirely on lookup) */
        }
    }

    impl Blake3CompressionChips {
        fn lookups(&self) -> Vec<Lookup<Expr>>{
            let var =
                |i| SymbolicExpression::from(SymbolicVariable::new(Entry::Main { offset: 0 }, i));

            let preprocessed_var = |i| {
                SymbolicExpression::from(SymbolicVariable::new(
                    Entry::Preprocessed { offset: 0 },
                    i,
                ))
            };

            let u8_xor_idx = Expr::from(Val::from_usize(Blake3CompressionChips::U8Xor.position()));
            let u32_xor_idx = Expr::from(Val::from_usize(Blake3CompressionChips::U32Xor.position()));
            match self {
                Self::U8Xor => {
                    vec![Lookup::pull(
                        var(0),
                        vec![
                            u8_xor_idx,
                            preprocessed_var(0),
                            preprocessed_var(1),
                            preprocessed_var(2),
                        ],
                    )]
                }

                Self::U32Xor => {
                    vec![
                        Lookup::pull(
                            var(0),
                            vec![
                                u32_xor_idx,
                                var(1)
                                    + var(2) * Expr::from_u32(256)
                                    + var(3) * Expr::from_u32(256 * 256)
                                    + var(4) * Expr::from_u32(256 * 256 * 256),
                                var(5)
                                    + var(6) * Expr::from_u32(256)
                                    + var(7) * Expr::from_u32(256 * 256)
                                    + var(8) * Expr::from_u32(256 * 256 * 256),
                                var(9)
                                    + var(10) * Expr::from_u32(256)
                                    + var(11) * Expr::from_u32(256 * 256)
                                    + var(12) * Expr::from_u32(256 * 256 * 256),
                            ]
                        ),

                        Lookup::push(Expr::ONE, vec![u8_xor_idx.clone(), var(1), var(5), var(9)]),
                        Lookup::push(Expr::ONE, vec![u8_xor_idx.clone(), var(2), var(6), var(10)]),
                        Lookup::push(Expr::ONE, vec![u8_xor_idx.clone(), var(3), var(7), var(11)]),
                        Lookup::push(Expr::ONE, vec![u8_xor_idx.clone(), var(4), var(8), var(12)]),
                    ]
                }
            }


        }
    }

    struct Blake3CompressionClaims {
        claims: Vec<Vec<Val>>
    }

    impl Blake3CompressionClaims {
        fn witness(&self, system: &System<Blake3CompressionChips>) -> (Vec<RowMajorMatrix<Val>>, SystemWitness) {

            // extract values from a claims

            let mut u32_values_from_claims = vec![];

            let mut byte_values_from_claims = vec![];

            for claim in self.claims.clone() {
                // we should have at least chip index
                assert!(claim.len() > 0, "wrong claim format");
                match claim[0].as_canonical_u64() {
                    0u64 => {
                        // This is our U8Xor claim. We should have chip_idx, A, B, A xor B (where A, B are bytes)
                        assert!(claim.len() == 4);
                        byte_values_from_claims.push((claim[1], claim[2]));
                    },
                    1u64 => {
                        // This is our U32Xor claim. We should have chip_idx, A, B, A xor B (where A, B are u32)
                        assert!(claim.len() == 4);
                        let a_u32 = u32::try_from(claim[1].as_canonical_u64()).unwrap();
                        let b_u32 = u32::try_from(claim[2].as_canonical_u64()).unwrap();
                        let xor_u32 = u32::try_from(claim[3].as_canonical_u64()).unwrap();

                        u32_values_from_claims.push((a_u32, b_u32, xor_u32));

                        // we decompose our input u32 words (A, B) and send their bytes to U8Xor chip, relying on lookup constraining
                        let a_bytes: [u8; 4] = a_u32.to_le_bytes();
                        let b_bytes: [u8; 4] = b_u32.to_le_bytes();

                        byte_values_from_claims.push((Val::from_u8(a_bytes[0]), Val::from_u8(b_bytes[0])));
                        byte_values_from_claims.push((Val::from_u8(a_bytes[1]), Val::from_u8(b_bytes[1])));
                        byte_values_from_claims.push((Val::from_u8(a_bytes[2]), Val::from_u8(b_bytes[2])));
                        byte_values_from_claims.push((Val::from_u8(a_bytes[3]), Val::from_u8(b_bytes[3])));
                    }
                    _ => panic!("unsupported chip")
                }
            }

            // build U8Xor trace (columns: multiplicity)
            let mut u8xor_trace_values = Vec::<Val>::with_capacity(BYTE_VALUES_NUM * BYTE_VALUES_NUM);
            for i in 0..BYTE_VALUES_NUM {
                for j in 0..BYTE_VALUES_NUM {
                    let mut multiplicity = Val::ZERO;
                    for vals in byte_values_from_claims.clone() {
                        if vals.0 == Val::from_usize(i) && vals.1 == Val::from_usize(j) {
                            multiplicity += Val::ONE;
                        }
                    }
                    u8xor_trace_values.push(multiplicity);
                }
            }

            // build U32Xor trace (columns: multiplicity, A0, A1, A2, A3, B0, B1, B2, B3, A0^B0, A1^B1, A2^B2, A3^B3)
            let mut u32xor_trace_values = Vec::<Val>::with_capacity(u32_values_from_claims.len());
            for (a, b, a_xor_b) in u32_values_from_claims.into_iter() {
                let a_bytes: [u8; 4] = a.to_le_bytes();
                let b_bytes: [u8; 4] = b.to_le_bytes();
                let a_xor_b_bytes: [u8; 4] = a_xor_b.to_le_bytes();

                u32xor_trace_values.push(Val::ONE); // multiplicity

                u32xor_trace_values.push(Val::from_u8(a_bytes[0]));
                u32xor_trace_values.push(Val::from_u8(a_bytes[1]));
                u32xor_trace_values.push(Val::from_u8(a_bytes[2]));
                u32xor_trace_values.push(Val::from_u8(a_bytes[3]));

                u32xor_trace_values.push(Val::from_u8(b_bytes[0]));
                u32xor_trace_values.push(Val::from_u8(b_bytes[1]));
                u32xor_trace_values.push(Val::from_u8(b_bytes[2]));
                u32xor_trace_values.push(Val::from_u8(b_bytes[3]));

                u32xor_trace_values.push(Val::from_u8(a_xor_b_bytes[0]));
                u32xor_trace_values.push(Val::from_u8(a_xor_b_bytes[1]));
                u32xor_trace_values.push(Val::from_u8(a_xor_b_bytes[2]));
                u32xor_trace_values.push(Val::from_u8(a_xor_b_bytes[3]));
            }

            let traces = vec![
                RowMajorMatrix::new(u8xor_trace_values, U8XOR_TRACE_WIDTH),
                RowMajorMatrix::new(u32xor_trace_values, U32XOR_TRACE_WIDTH)
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

        let a_u32 = 0x000000ff;
        let b_u32 = 0x0000ff01;
        let xor_u32 = a_u32 ^ b_u32;


        // circuit testing
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let u8_circuit = LookupAir::new(Blake3CompressionChips::U8Xor, Blake3CompressionChips::U8Xor.lookups());
        let u32_circuit = LookupAir::new(Blake3CompressionChips::U32Xor, Blake3CompressionChips::U32Xor.lookups());

        let (system, prover_key) = System::new(commitment_parameters, vec![
            u8_circuit,
            u32_circuit,
        ]);

        let f = Val::from_u8;
        let f32 = Val::from_u32;
        let claims = Blake3CompressionClaims {
            claims: vec![
                vec![Val::from_usize(Blake3CompressionChips::U8Xor.position()), f(a_u8), f(b_u8), f(xor_u8)],
                vec![Val::from_usize(Blake3CompressionChips::U8Xor.position()), f(a1_u8), f(b1_u8), f(xor1_u8)],
                vec![Val::from_usize(Blake3CompressionChips::U32Xor.position()), f32(a_u32), f32(b_u32), f32(xor_u32)],
            ]
        };

        let (traces, witness) = claims.witness(&system);

        let claims_slice: Vec<&[Val]> = claims.claims.iter().map(|v| v.as_slice()).collect();
        let claims_slice: &[&[Val]] = &claims_slice;

        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };

        let proof = system.prove_multiple_claims(fri_parameters, &prover_key, claims_slice, witness);
        system.verify_multiple_claims(fri_parameters, claims_slice, &proof).expect("verification issue");
    }
}