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
        fn eval(&self, builder: &mut AB) {}
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
}
