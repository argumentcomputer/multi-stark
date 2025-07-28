#[cfg(test)]
mod tests {
    use crate::builder::symbolic::{Entry, SymbolicExpression, SymbolicVariable};
    use crate::lookup::{Lookup, LookupAir};
    use crate::system::{Circuit, System, SystemWitness};
    use crate::types::{FriParameters, Val, new_stark_config};
    use itertools::Itertools;
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_challenger::{HashChallenger, SerializingChallenger64};
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_fri::{TwoAdicFriPcs, create_benchmark_fri_params};
    use p3_goldilocks::Goldilocks;
    use p3_keccak::{Keccak256Hash, KeccakF};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
    use p3_uni_stark::{prove, verify};
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use std::array;

    /// How many `a * b = c` operations to do per row.
    const REPETITIONS: usize = 1; // This should be < 255 so it can fit into a u8
    const TRACE_WIDTH: usize = REPETITIONS * 3;

    struct MultiplicationAir {}

    impl MultiplicationAir {
        fn random_valid_trace<F: Field>(
            &self,
            trace_height: usize,
            valid: bool,
        ) -> RowMajorMatrix<F>
        where
            StandardUniform: Distribution<F>,
        {
            let mut rng = SmallRng::seed_from_u64(1);

            let mut trace_values = F::zero_vec(trace_height * TRACE_WIDTH);

            for (i, (a, b, c)) in trace_values.iter_mut().tuples().enumerate() {
                let row = i / REPETITIONS;
                *a = F::from_usize(i);

                *b = if row == 0 {
                    a.square() + F::ONE
                } else {
                    rng.random()
                };

                *c = *a * *b;

                if !valid {
                    *c *= F::TWO;
                }
            }
            RowMajorMatrix::new(trace_values, TRACE_WIDTH)
        }
    }

    impl<F> BaseAir<F> for MultiplicationAir {
        fn width(&self) -> usize {
            TRACE_WIDTH
        }
    }

    impl<AB: AirBuilder> Air<AB> for MultiplicationAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();

            let main_local = main.row_slice(0).expect("Matrix is empty?");
            let main_next = main.row_slice(1).expect("Matrix only has 1 row?");
            for i in 0..REPETITIONS {
                let start = i * 3;
                let a = main_local[start].clone();
                let b = main_local[start + 1].clone();
                let c = main_local[start + 2].clone();

                builder.assert_zero(a.clone() * b.clone() - c);

                builder
                    .when_first_row()
                    .assert_eq(a.clone() * a.clone() + AB::Expr::ONE, b);

                let next_a = main_next[start].clone();
                builder
                    .when_transition()
                    .assert_eq(a + AB::Expr::from_u8(REPETITIONS as u8), next_a);
            }
        }
    }

    #[test]
    fn test_mul_air_trace() {
        let log_rows = 10;
        let air = MultiplicationAir {};
        let trace = air.random_valid_trace(1 << log_rows, true);

        type F = Goldilocks;
        type EF = BinomialExtensionField<F, 2>;
        type Sponge = PaddingFreeSponge<KeccakF, 25, 17, 4>; // Poseidon2 is also possible
        type KeccakCompressionFunction = CompressionFunctionFromHasher<Sponge, 2, 4>;

        let dft = Radix2DitParallel::<Goldilocks>::default();
        let u64_hash = Sponge::new(KeccakF {});
        let field_hash = SerializingHasher::new(u64_hash);
        let compress = KeccakCompressionFunction::new(u64_hash);
        let val_mmcs = MerkleTreeMmcs::<
            [F; p3_keccak::VECTOR_LEN],
            [u64; p3_keccak::VECTOR_LEN],
            SerializingHasher<Sponge>,
            KeccakCompressionFunction,
            4,
        >::new(field_hash, compress);

        let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
        let fri_params = create_benchmark_fri_params(challenge_mmcs);
        let pcs = TwoAdicFriPcs::<F, _, _, _>::new(dft, val_mmcs, fri_params);
        let challenger = SerializingChallenger64::from_hasher(vec![], Keccak256Hash {});
        let stark_config = p3_uni_stark::StarkConfig::<
            _,
            EF,
            SerializingChallenger64<Goldilocks, HashChallenger<u8, Keccak256Hash, 32>>,
        >::new(pcs, challenger);

        let proof = prove(&stark_config, &air, trace, &vec![]);

        let serialized_proof = postcard::to_allocvec(&proof).expect("unable to serialize");
        println!("Postcard proof size: {} bytes", serialized_proof.len());

        let config = bincode::config::standard()
            .with_little_endian()
            .with_fixed_int_encoding();
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, config).expect("Failed to serialize proof");
        println!("Bincode proof size: {} bytes", proof_bytes.len());

        verify(&stark_config, &air, &proof, &vec![]).expect("verification issue");
    }

    // Z, M, A
    const WIDTH: usize = 3;

    // Z(i+1) = Z(i) * M(i) + A(i):
    //
    // Starting from Z=1, M=1, A=1, then for each step A increases by one and M increases by two
    //      Z     M   A
    // 0: | 1   | 1 | 1 |
    // 1: | 2   | 3 | 2 |
    // 2: | 8   | 5 | 3 |
    // 3: | 43  | 7 | 4 |
    // 4: | 305 | 9 | 5 |
    //
    // . . .
    //
    struct ChainedMultiplyAndAdd {}

    impl ChainedMultiplyAndAdd {
        fn random_trace<F: Field>(&self, height: usize) -> RowMajorMatrix<F>
        where
            StandardUniform: Distribution<F>,
        {
            let f = F::from_u32;

            let mut trace_values = F::zero_vec(height * WIDTH);

            let mut z_prev = f(1);
            let mut m_prev = f(1);
            let mut a_prev = f(1);
            for (i, (z_i, m_i, a_i)) in trace_values.iter_mut().tuples().enumerate() {
                // initialize
                if i == 0 {
                    *z_i = z_prev;
                    *m_i = m_prev;
                    *a_i = a_prev;
                } else {
                    // set current values
                    *z_i = z_prev * m_prev + a_prev;
                    *m_i = m_prev + F::TWO;
                    *a_i = a_prev + F::ONE;

                    // update previous values
                    z_prev = *z_i;
                    m_prev = *m_i;
                    a_prev = *a_i;
                }
            }

            RowMajorMatrix::new(trace_values, WIDTH)
        }
    }

    impl<F> BaseAir<F> for ChainedMultiplyAndAdd {
        fn width(&self) -> usize {
            WIDTH
        }
    }

    impl<AB: AirBuilder> Air<AB> for ChainedMultiplyAndAdd {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let curr = main.row_slice(0).expect("matrix is empty?");
            let next = main.row_slice(1).expect("matrix has only 1 row?");

            let z = curr[0].clone();
            let m = curr[1].clone();
            let a = curr[2].clone();

            let z_next = next[0].clone();
            let m_next = next[1].clone();
            let a_next = next[2].clone();

            // z(0) = 1, m(0) = 1, a(0) = 1
            builder.when_first_row().assert_zeros([
                a.clone() - AB::Expr::ONE,
                m.clone() - AB::Expr::ONE,
                z.clone() - AB::Expr::ONE,
            ]);

            // z(i+1) = z(i) * m(i) + a(i)
            builder
                .when_transition()
                .assert_eq(z.clone() * m.clone() + a.clone(), z_next.clone());

            // m(i+1) = m(i) + 2
            builder
                .when_transition()
                .assert_eq(m.clone() + AB::Expr::TWO, m_next.clone());

            // a(i+1) = a(i) + 1
            builder
                .when_transition()
                .assert_eq(a.clone() + AB::Expr::ONE, a_next.clone());
        }
    }

    #[test]
    fn test_chained_multiply_and_add() {
        let log_rows = 10;
        let air = ChainedMultiplyAndAdd {};
        let trace: RowMajorMatrix<Goldilocks> = air.random_trace(1 << log_rows);

        type F = Goldilocks;
        type EF = BinomialExtensionField<F, 2>;
        type Sponge = PaddingFreeSponge<KeccakF, 25, 17, 4>; // Poseidon2 is also possible
        type KeccakCompressionFunction = CompressionFunctionFromHasher<Sponge, 2, 4>;

        let dft = Radix2DitParallel::<Goldilocks>::default();
        let u64_hash = Sponge::new(KeccakF {});
        let field_hash = SerializingHasher::new(u64_hash);
        let compress = KeccakCompressionFunction::new(u64_hash);
        let val_mmcs = MerkleTreeMmcs::<
            [F; p3_keccak::VECTOR_LEN],
            [u64; p3_keccak::VECTOR_LEN],
            SerializingHasher<Sponge>,
            KeccakCompressionFunction,
            4,
        >::new(field_hash, compress);

        let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
        let fri_params = create_benchmark_fri_params(challenge_mmcs);
        let pcs = TwoAdicFriPcs::<F, _, _, _>::new(dft, val_mmcs, fri_params);
        let challenger = SerializingChallenger64::from_hasher(vec![], Keccak256Hash {});
        let stark_config = p3_uni_stark::StarkConfig::<
            _,
            EF,
            SerializingChallenger64<Goldilocks, HashChallenger<u8, Keccak256Hash, 32>>,
        >::new(pcs, challenger);

        let proof = prove(&stark_config, &air, trace, &vec![]);

        let serialized_proof = postcard::to_allocvec(&proof).expect("unable to serialize");
        println!("Postcard proof size: {} bytes", serialized_proof.len());

        let config = bincode::config::standard()
            .with_little_endian()
            .with_fixed_int_encoding();
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, config).expect("Failed to serialize proof");
        println!("Bincode proof size: {} bytes", proof_bytes.len());

        verify(&stark_config, &air, &proof, &vec![]).expect("verification issue");
    }

    enum RangeChecking {
        RangeCheckU8,
        XorU8,
    }

    impl<F> BaseAir<F> for RangeChecking {
        fn width(&self) -> usize {
            match self {
                // multiplicity, value
                RangeChecking::RangeCheckU8 => 2,
                // multiplicity, value1, value2, xor
                RangeChecking::XorU8 => 4,
            }
        }
    }

    impl<AB> Air<AB> for RangeChecking
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                RangeChecking::RangeCheckU8 => {
                    let main = builder.main();
                    let current_row = main.row_slice(0).unwrap();
                    let next_row = main.row_slice(1).unwrap();

                    let current_byte = current_row[1];
                    let next_byte = next_row[1];

                    // we start from 0
                    builder.when_first_row().assert_zero(current_byte.into());

                    // every next byte equals to current byte + 1
                    builder.when_transition().assert_eq(
                        current_byte.into() + <AB as AirBuilder>::Expr::ONE,
                        next_byte.into(),
                    );

                    // we end with 255
                    builder
                        .when_last_row()
                        .assert_eq(current_byte.into(), <AB as AirBuilder>::Expr::from_u8(255));
                }
                RangeChecking::XorU8 => {
                    // println!("XorU8 eval");
                }
            }
        }
    }

    impl RangeChecking {
        fn lookups(&self) -> Vec<Lookup<SymbolicExpression<Val>>> {
            let var = |index| {
                SymbolicExpression::from(SymbolicVariable::new(Entry::Main { offset: 0 }, index))
            };
            let range_check_idx = SymbolicExpression::<Val>::from_u8(0);
            let xor_u8_idx = SymbolicExpression::<Val>::from_u8(1);

            match self {
                RangeChecking::RangeCheckU8 => vec![Lookup {
                    multiplicity: -var(0),
                    args: vec![range_check_idx, var(1)],
                }],

                RangeChecking::XorU8 => vec![
                    Lookup {
                        multiplicity: -var(0),
                        args: vec![xor_u8_idx.clone(), var(1), var(2), var(3)],
                    },
                    // send values of 2 bytes and their XOR for range checking
                    Lookup {
                        multiplicity: SymbolicExpression::<Val>::ONE,
                        args: vec![range_check_idx.clone(), var(1)],
                    },
                    Lookup {
                        multiplicity: SymbolicExpression::<Val>::ONE,
                        args: vec![range_check_idx.clone(), var(2)],
                    },
                    Lookup {
                        multiplicity: SymbolicExpression::<Val>::ONE,
                        args: vec![range_check_idx.clone(), var(3)],
                    },
                ],
            }
        }

        fn trace_range_check_u8(byte_values_to_check: Vec<u8>) -> RowMajorMatrix<Val> {
            let mut multiplicities = Val::zero_vec(256);
            for b in byte_values_to_check {
                multiplicities[b as usize] += Val::ONE;
            }

            let bytes: [Val; 256] = array::from_fn(|idx| Val::from_usize(idx));
            let mut trace_values = Vec::with_capacity(multiplicities.len() + bytes.len());
            for i in 0..256 {
                trace_values.push(multiplicities[i]);
                trace_values.push(bytes[i]);
            }
            RowMajorMatrix::new(trace_values, 2)
        }

        fn trace_byte_xor_u8(byte_values_to_check: Vec<u8>) -> Vec<RowMajorMatrix<Val>> {
            // we expect 3 bytes in input: a, b, a ^ b.
            assert_eq!(byte_values_to_check.len(), 3);

            let range_check_u8_trace = Self::trace_range_check_u8(byte_values_to_check.clone());

            // println!("byte trace: {:?}", range_check_u8_trace);

            let bytes: [u8; 256] = array::from_fn(|idx| idx as u8);
            let mut trace_values = Vec::with_capacity(256 * 256 * 4);
            for i in 0..256 {
                for j in 0..256 {
                    // multiplicity
                    if (bytes[i] == byte_values_to_check[0])
                        && (bytes[j] == byte_values_to_check[1])
                        && ((bytes[i] ^ bytes[j]) == byte_values_to_check[2])
                    {
                        trace_values.push(Val::ONE);
                    } else {
                        trace_values.push(Val::ZERO);
                    }

                    trace_values.push(Val::from_u8(bytes[i])); // a
                    trace_values.push(Val::from_u8(bytes[j])); // b
                    trace_values.push(Val::from_u8(bytes[i] ^ bytes[j])); // a ^ b
                }
            }

            let byte_xor_u8_trace = RowMajorMatrix::new(trace_values, 4);

            vec![range_check_u8_trace, byte_xor_u8_trace]
        }
    }

    enum ByteOperations {
        XOR,
    }

    impl<F> BaseAir<F> for ByteOperations {
        fn width(&self) -> usize {
            match self {
                // multiplicity, a, b, a ^ b
                ByteOperations::XOR => 4,
            }
        }
    }

    impl<AB> Air<AB> for ByteOperations
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, _builder: &mut AB) {}
    }

    impl ByteOperations {
        fn lookups(&self) -> Vec<Lookup<SymbolicExpression<Val>>> {
            let var = |index| {
                SymbolicExpression::from(SymbolicVariable::new(Entry::Main { offset: 0 }, index))
            };
            let xor_u8_idx = SymbolicExpression::<Val>::from_u8(0);
            vec![Lookup {
                multiplicity: -var(0),
                args: vec![xor_u8_idx.clone(), var(1), var(2), var(3)],
            }]
        }
    }

    #[test]
    fn test_xor_u8() {
        // create our circuit's system
        let xor_u8 = Circuit::from_air(LookupAir {
            inner_air: ByteOperations::XOR,
            lookups: ByteOperations::XOR.lookups(),
        })
        .unwrap();
        let system = System::new(vec![xor_u8]);

        let chip_idx = 0u8;
        let xor_data = vec![0xf0, 0x0e, 0xfe];

        // we expect 0-th circuit to provide xor data
        let claim = [chip_idx, xor_data[0], xor_data[1], xor_data[2]]
            .map(Val::from_u8)
            .to_vec();

        let bytes: [u8; 256] = array::from_fn(|idx| idx as u8);
        let mut trace_values = Vec::with_capacity(256 * 256 * 4);
        for i in 0..256 {
            for j in 0..256 {
                if bytes[i] == xor_data[0]
                    && bytes[j] == xor_data[1]
                    && ((bytes[i] ^ bytes[j]) == xor_data[2])
                {
                    trace_values.push(Val::ONE);
                } else {
                    trace_values.push(Val::ZERO);
                }
                trace_values.push(Val::from_u8(bytes[i])); // a
                trace_values.push(Val::from_u8(bytes[j])); // b
                trace_values.push(Val::from_u8(bytes[i] ^ bytes[j])); // a ^ b
            }
        }

        let xor_u8_traces = vec![RowMajorMatrix::new(trace_values, 4)];

        let witness = SystemWitness::from_stage_1(xor_u8_traces, &system);
        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };

        let config = new_stark_config(&fri_parameters);
        let proof = system.prove(&config, &claim, witness);
        system.verify(&config, &claim, &proof).unwrap()
    }

    #[test]
    fn test_range_checking() {
        // create our circuit's system
        let range_check_u8 = Circuit::from_air(LookupAir {
            inner_air: RangeChecking::RangeCheckU8,
            lookups: RangeChecking::RangeCheckU8.lookups(),
        })
        .unwrap();
        let system = System::new(vec![range_check_u8]);

        // we call 0-th chip (range_check_u8) from our circuit with the value 255.
        // This claim actually increments the multiplicity of 255, so we should balance it with our lookup
        let value_to_check = 255u8;
        let claim = [0, value_to_check].map(Val::from_u8).to_vec();

        let range_checking_trace = RangeChecking::trace_range_check_u8(vec![value_to_check]);

        let witness = SystemWitness::from_stage_1(vec![range_checking_trace], &system);
        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let config = new_stark_config(&fri_parameters);
        let proof = system.prove(&config, &claim, witness);
        system.verify(&config, &claim, &proof).unwrap()
    }

    enum U32 {
        RangeCheckU8,
        RotateRight16,
        RotateRight8,
    }

    impl<F> BaseAir<F> for U32 {
        fn width(&self) -> usize {
            match self {
                // multiplicity, in (where in is byte)
                Self::RangeCheckU8 => 2,

                // multiplicity, in_0, in_1, in_2, in_3, out_0, out_1, out_2, out_3 (where in_i / out_i are bytes)
                Self::RotateRight8 => 9,
                Self::RotateRight16 => 9,
            }
        }
    }

    impl<AB> Air<AB> for U32
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                Self::RangeCheckU8 => {
                    let main = builder.main();
                    let local = main.row_slice(0).unwrap();
                    let next = main.row_slice(1).unwrap();
                    let byte = &local[1];
                    let next_byte = &next[1];
                    builder.when_first_row().assert_zero(byte.clone());
                    builder
                        .when_transition()
                        .assert_eq(byte.clone() + AB::Expr::ONE, next_byte.clone());
                    builder
                        .when_last_row()
                        .assert_eq(byte.clone(), AB::Expr::from_u8(255));
                }
                Self::RotateRight8 => {
                    // no regular P3 constraints (we rely on lookup)
                }
                Self::RotateRight16 => {
                    // no regular P3 constraints (we rely on lookup)
                }
            }
        }
    }

    type Symbolic = SymbolicExpression<Val>;

    impl U32 {
        fn lookups(&self) -> Vec<Lookup<Symbolic>> {
            let var = |index| {
                SymbolicExpression::from(SymbolicVariable::new(Entry::Main { offset: 0 }, index))
            };
            let range_check_u8_chip = Symbolic::from_u8(0);
            let rotate_right_8_chip = Symbolic::from_u8(1);
            let rotate_right_16_chip = Symbolic::from_u8(2);

            match self {
                // provide byte value
                Self::RangeCheckU8 => {
                    vec![Lookup {
                        multiplicity: -var(0),
                        args: vec![range_check_u8_chip.clone(), var(1)],
                    }]
                }
                Self::RotateRight8 => {
                    // provide input u32 and rotated u32
                    vec![
                        Lookup {
                            multiplicity: -var(0),
                            args: vec![
                                rotate_right_8_chip.clone(),
                                var(1)
                                    + var(2) * Symbolic::from_u32(256)
                                    + var(3) * Symbolic::from_u32(256 * 256)
                                    + var(4) * Symbolic::from_u32(256 * 256 * 256),
                                // note var indices
                                var(2)
                                    + var(3) * Symbolic::from_u32(256)
                                    + var(4) * Symbolic::from_u32(256 * 256)
                                    + var(1) * Symbolic::from_u32(256 * 256 * 256),
                            ],
                        },
                        // require decomposed bytes for range checking
                        Lookup {
                            multiplicity: Symbolic::TWO,
                            args: vec![range_check_u8_chip.clone(), var(1)],
                        },
                        Lookup {
                            multiplicity: Symbolic::TWO,
                            args: vec![range_check_u8_chip.clone(), var(2)],
                        },
                        Lookup {
                            multiplicity: Symbolic::TWO,
                            args: vec![range_check_u8_chip.clone(), var(3)],
                        },
                        Lookup {
                            multiplicity: Symbolic::TWO,
                            args: vec![range_check_u8_chip.clone(), var(4)],
                        },
                    ]
                }
                Self::RotateRight16 => {
                    vec![
                        Lookup {
                            multiplicity: -var(0),
                            args: vec![
                                rotate_right_16_chip.clone(),
                                var(1)
                                    + var(2) * Symbolic::from_u32(256)
                                    + var(3) * Symbolic::from_u32(256 * 256)
                                    + var(4) * Symbolic::from_u32(256 * 256 * 256),
                                // note var indices
                                var(3)
                                    + var(4) * Symbolic::from_u32(256)
                                    + var(1) * Symbolic::from_u32(256 * 256)
                                    + var(2) * Symbolic::from_u32(256 * 256 * 256),
                            ],
                        },
                        // require decomposed bytes for range checking
                        Lookup {
                            multiplicity: Symbolic::TWO,
                            args: vec![range_check_u8_chip.clone(), var(1)],
                        },
                        Lookup {
                            multiplicity: Symbolic::TWO,
                            args: vec![range_check_u8_chip.clone(), var(2)],
                        },
                        Lookup {
                            multiplicity: Symbolic::TWO,
                            args: vec![range_check_u8_chip.clone(), var(3)],
                        },
                        Lookup {
                            multiplicity: Symbolic::TWO,
                            args: vec![range_check_u8_chip.clone(), var(4)],
                        },
                    ]
                }
            }
        }

        fn traces(&self, input: u32, output: u32) -> Vec<RowMajorMatrix<Val>> {
            let input_bytes: [u8; 4] = input.to_le_bytes();
            let output_bytes: [u8; 4] = output.to_le_bytes();
            let values = [
                vec![Val::ONE],
                input_bytes.map(Val::from_u8).to_vec(),
                output_bytes.map(Val::from_u8).to_vec(),
            ]
            .concat();
            let rotate_trace = RowMajorMatrix::new(values, 9);

            // for the byte chip trace we have to increment multiplicities for the bytes encountered in the IO while decomposition
            let mut multiplicity = Val::zero_vec(256);
            for byte in [input_bytes, output_bytes].concat() {
                multiplicity[byte as usize] += Val::ONE;
            }

            let bytes: [u8; 256] = array::from_fn(|idx| u8::try_from(idx).unwrap());
            let mut values = Vec::<Val>::with_capacity(256 * 2);
            for i in 0..256 {
                values.push(multiplicity[i]);
                values.push(Val::from_u8(bytes[i]));
            }

            let byte_trace = RowMajorMatrix::new(values, 2);
            vec![byte_trace, rotate_trace]
        }
    }

    #[test]
    fn test_rotate_8() {
        let system = System::new(vec![
            Circuit::from_air(LookupAir {
                inner_air: U32::RangeCheckU8,
                lookups: U32::RangeCheckU8.lookups(),
            })
            .unwrap(),
            Circuit::from_air(LookupAir {
                inner_air: U32::RotateRight8,
                lookups: U32::RotateRight8.lookups(),
            })
            .unwrap(),
        ]);

        let rotate_right_8_chip_idx = 1;
        let a = 0xffff0000u32;
        let rotate_right_8 = a.rotate_right(8);
        let claim = [rotate_right_8_chip_idx, a, rotate_right_8]
            .map(Val::from_u32)
            .to_vec();

        let witness =
            SystemWitness::from_stage_1(U32::RotateRight8.traces(a, rotate_right_8), &system);

        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let config = new_stark_config(&fri_parameters);
        let proof = system.prove(&config, &claim, witness);
        system.verify(&config, &claim, &proof).unwrap()
    }

    #[test]
    fn test_rotate_16() {
        let system = System::new(vec![
            Circuit::from_air(LookupAir {
                inner_air: U32::RangeCheckU8,
                lookups: U32::RangeCheckU8.lookups(),
            })
            .unwrap(),
            Circuit::from_air(LookupAir {
                inner_air: U32::RotateRight16,
                lookups: U32::RotateRight16.lookups(),
            })
            .unwrap(),
        ]);

        let rotate_right_16_chip_idx = 2;
        let a = 0xffff0000u32;
        let rotate_right_16 = a.rotate_right(16);
        let claim = [rotate_right_16_chip_idx, a, rotate_right_16]
            .map(Val::from_u32)
            .to_vec();

        let witness =
            SystemWitness::from_stage_1(U32::RotateRight16.traces(a, rotate_right_16), &system);

        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let config = new_stark_config(&fri_parameters);
        let proof = system.prove(&config, &claim, witness);
        system.verify(&config, &claim, &proof).unwrap()
    }

    enum RotateRight {
        RangeCheckU8,
        RotateRight10,
        RotateRight12,
    }

    impl<F> BaseAir<F> for RotateRight {
        fn width(&self) -> usize {
            match self {
                // multiplicity, byte_val
                Self::RangeCheckU8 => 2,

                // multiplicity,
                // in_0, in_1, in_2, in_3,
                // out_0, out_1, out_2, out_3,
                // two_pow_k_0, two_pow_k_1, two_pow_k_2, two_pow_k_3,
                // two_pow_32_minus_k_0, two_pow_32_minus_k_1, two_pow_32_minus_k_2, two_pow_32_minus_k_3,
                // value_div_0, value_div_1, value_div_2, value_div_3
                // value_rem_0, value_rem_1, value_rem_2, value_rem_3
                Self::RotateRight10 => 25,
                Self::RotateRight12 => 25,
            }
        }
    }

    impl<AB> Air<AB> for RotateRight
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                Self::RangeCheckU8 => {
                    let main = builder.main();
                    let local = main.row_slice(0).unwrap();
                    let next = main.row_slice(1).unwrap();
                    let byte = &local[1];
                    let next_byte = &next[1];
                    builder.when_first_row().assert_zero(byte.clone());
                    builder
                        .when_transition()
                        .assert_eq(byte.clone() + AB::Expr::ONE, next_byte.clone());
                    builder
                        .when_last_row()
                        .assert_eq(byte.clone(), AB::Expr::from_u8(255));
                }
                Self::RotateRight10 | Self::RotateRight12 => {
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

    impl RotateRight {
        fn lookups(&self) -> Vec<Lookup<Symbolic>> {
            let var = |index| {
                SymbolicExpression::from(SymbolicVariable::new(Entry::Main { offset: 0 }, index))
            };
            let range_check_u8_chip = Symbolic::from_u8(0);
            let rotate_right_10_chip = Symbolic::from_u8(1);
            let rotate_right_12_chip = Symbolic::from_u8(2);

            match self {
                Self::RangeCheckU8 => {
                    // provide byte value
                    vec![Lookup {
                        multiplicity: -var(0),
                        args: vec![range_check_u8_chip.clone(), var(1)],
                    }]
                }
                Self::RotateRight10 => {
                    vec![
                        Lookup {
                            multiplicity: -var(0),
                            args: vec![
                                rotate_right_10_chip.clone(),
                                var(1)
                                    + var(2) * Symbolic::from_u32(256)
                                    + var(3) * Symbolic::from_u32(256 * 256)
                                    + var(4) * Symbolic::from_u32(256 * 256 * 256),
                                var(5)
                                    + var(6) * Symbolic::from_u32(256)
                                    + var(7) * Symbolic::from_u32(256 * 256)
                                    + var(8) * Symbolic::from_u32(256 * 256 * 256),
                            ],
                        },
                        // require decomposed bytes for range checking
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(1)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(2)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(3)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(4)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(5)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(6)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(7)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(8)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(9)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(10)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(11)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(12)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(13)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(14)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(15)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(16)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(17)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(18)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(19)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(20)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(21)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(22)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(23)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(24)],
                        },
                    ]
                }
                Self::RotateRight12 => {
                    vec![
                        Lookup {
                            multiplicity: -var(0),
                            args: vec![
                                rotate_right_12_chip.clone(),
                                var(1)
                                    + var(2) * Symbolic::from_u32(256)
                                    + var(3) * Symbolic::from_u32(256 * 256)
                                    + var(4) * Symbolic::from_u32(256 * 256 * 256),
                                var(5)
                                    + var(6) * Symbolic::from_u32(256)
                                    + var(7) * Symbolic::from_u32(256 * 256)
                                    + var(8) * Symbolic::from_u32(256 * 256 * 256),
                            ],
                        },
                        // require decomposed bytes for range checking
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(1)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(2)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(3)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(4)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(5)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(6)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(7)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(8)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(9)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(10)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(11)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(12)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(13)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(14)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(15)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(16)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(17)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(18)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(19)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(20)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(21)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(22)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(23)],
                        },
                        Lookup {
                            multiplicity: Symbolic::ONE,
                            args: vec![range_check_u8_chip.clone(), var(24)],
                        },
                    ]
                }
            }
        }

        fn traces(&self, input: u32, output: u32) -> Vec<RowMajorMatrix<Val>> {
            let input_bytes: [u8; 4] = input.to_le_bytes();
            let output_bytes: [u8; 4] = output.to_le_bytes();

            let k: u32 = match self {
                Self::RangeCheckU8 => {
                    // this function is not expected to be invoked on range_check chip
                    panic!("should never happen");
                }
                Self::RotateRight10 => 10,
                Self::RotateRight12 => 12,
            };

            let two_pow_k = u32::try_from(2usize.pow(k)).unwrap();
            let two_pow_32_minus_k = u32::try_from(2usize.pow(32 - k)).unwrap();

            let input_div = input / two_pow_k;
            let input_rem = input % two_pow_k;

            let two_pow_k_bytes: [u8; 4] = two_pow_k.to_le_bytes();
            let two_pow_32_minus_k_bytes: [u8; 4] = two_pow_32_minus_k.to_le_bytes();
            let input_div_bytes: [u8; 4] = input_div.to_le_bytes();
            let input_rem_bytes: [u8; 4] = input_rem.to_le_bytes();

            let values = [
                vec![Val::ONE],
                input_bytes.map(Val::from_u8).to_vec(),
                output_bytes.map(Val::from_u8).to_vec(),
                two_pow_k_bytes.map(Val::from_u8).to_vec(),
                two_pow_32_minus_k_bytes.map(Val::from_u8).to_vec(),
                input_div_bytes.map(Val::from_u8).to_vec(),
                input_rem_bytes.map(Val::from_u8).to_vec(),
            ]
            .concat();

            let rotate_trace = RowMajorMatrix::new(values, 25);

            // for the byte chip trace we have to increment multiplicities for the bytes encountered in the IO while decomposition
            let mut multiplicity = Val::zero_vec(256);
            for byte in [
                input_bytes,
                output_bytes,
                two_pow_k_bytes,
                two_pow_32_minus_k_bytes,
                input_div_bytes,
                input_rem_bytes,
            ]
            .concat()
            {
                multiplicity[byte as usize] += Val::ONE;
            }

            let bytes: [u8; 256] = array::from_fn(|idx| u8::try_from(idx).unwrap());
            let mut values = Vec::<Val>::with_capacity(256 * 2);
            for i in 0..256 {
                values.push(multiplicity[i]);
                values.push(Val::from_u8(bytes[i]));
            }

            let byte_trace = RowMajorMatrix::new(values, 2);
            vec![byte_trace, rotate_trace]
        }
    }

    #[test]
    fn test_right_rotate_10() {
        let system = System::new(vec![
            Circuit::from_air(LookupAir {
                inner_air: RotateRight::RangeCheckU8,
                lookups: RotateRight::RangeCheckU8.lookups(),
            })
            .unwrap(),
            Circuit::from_air(LookupAir {
                inner_air: RotateRight::RotateRight10,
                lookups: RotateRight::RotateRight10.lookups(),
            })
            .unwrap(),
        ]);

        let rotate_right_10_chip_idx = 1;
        let a = 0xffff0000u32;
        let k = 10;
        let rotate_right_10 = a.rotate_right(k);
        let claim = [rotate_right_10_chip_idx, a, rotate_right_10]
            .map(Val::from_u32)
            .to_vec();

        let witness = SystemWitness::from_stage_1(
            RotateRight::RotateRight10.traces(a, rotate_right_10),
            &system,
        );

        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let config = new_stark_config(&fri_parameters);
        let proof = system.prove(&config, &claim, witness);
        system.verify(&config, &claim, &proof).unwrap()
    }

    #[test]
    fn test_right_rotate_12() {
        let system = System::new(vec![
            Circuit::from_air(LookupAir {
                inner_air: RotateRight::RangeCheckU8,
                lookups: RotateRight::RangeCheckU8.lookups(),
            })
            .unwrap(),
            Circuit::from_air(LookupAir {
                inner_air: RotateRight::RotateRight12,
                lookups: RotateRight::RotateRight12.lookups(),
            })
            .unwrap(),
        ]);

        let rotate_right_12_chip_idx = 2;
        let a = 0xffff0000u32;
        let k = 12;
        let rotate_right_12 = a.rotate_right(k);
        let claim = [rotate_right_12_chip_idx, a, rotate_right_12]
            .map(Val::from_u32)
            .to_vec();

        let witness = SystemWitness::from_stage_1(
            RotateRight::RotateRight12.traces(a, rotate_right_12),
            &system,
        );

        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let config = new_stark_config(&fri_parameters);
        let proof = system.prove(&config, &claim, witness);
        system.verify(&config, &claim, &proof).unwrap()
    }

    #[test]
    fn test_custom_right_rotate() {
        let a = 0xff00ff00u32;
        let k = 10;

        // right rotation to k bits expressed that can be expressed as multiplication and addition
        fn rot(value: u32, k: u32) -> u32 {
            assert!(k < 32);
            let two_pow_k = u32::try_from(2usize.pow(k)).unwrap();
            let two_pow_32_minus_k = u32::try_from(2usize.pow(32 - k)).unwrap();
            (value / two_pow_k) + (value % two_pow_k) * two_pow_32_minus_k
        }

        assert_eq!(a.rotate_right(k), rot(a, k));

        let k = 12;

        assert_eq!(a.rotate_right(k), rot(a, k));
    }
}
