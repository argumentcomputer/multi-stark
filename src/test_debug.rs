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

    pub enum ByteOperations {
        RangeU8Chip,
        ByteXorChip,
    }

    impl<F> BaseAir<F> for ByteOperations {
        fn width(&self) -> usize {
            match self {
                // multiplicity, byte_value
                Self::RangeU8Chip => 2,
                // multiplicity, byte_value_1, byte_value_2, xor_result
                Self::ByteXorChip => 4,
            }
        }
    }

    impl<AB> Air<AB> for ByteOperations
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            // TODO: p3 constraints are not speicified yet
        }
    }

    type Expr = SymbolicExpression<Val>;
    impl ByteOperations {
        fn lookups(&self) -> Vec<Lookup<Expr>> {
            let var = |index| Expr::from(SymbolicVariable::new(Entry::Main { offset: 0 }, index));

            let range_u8_chip_idx = Expr::from_u32(0);

            let byte_xor_chip_idx = Expr::from_u32(1);

            match self {
                Self::RangeU8Chip => vec![
                    // We are receiving one byte (and subtract the multiplicity) while performing range_check
                    Lookup {
                        multiplicity: -var(0),
                        args: vec![range_u8_chip_idx.clone(), var(1)],
                    },
                ],

                Self::ByteXorChip => vec![
                    // we have to send (require) every byte that participates in XOR (left, right and result) to a RangeCheck chip, e.g. setting multiplicity to 1 for every lookup
                    Lookup {
                        multiplicity: Expr::ONE,
                        args: vec![range_u8_chip_idx.clone(), var(1)],
                    },
                    Lookup {
                        multiplicity: Expr::ONE,
                        args: vec![range_u8_chip_idx.clone(), var(2)],
                    },
                    Lookup {
                        multiplicity: Expr::ONE,
                        args: vec![range_u8_chip_idx.clone(), var(3)],
                    },
                    // we have to receive values of 3 bytes that participate in XOR operation (left, right and result)
                    Lookup {
                        multiplicity: -var(0),
                        args: vec![byte_xor_chip_idx.clone(), var(1), var(2), var(3)],
                    },
                ],
            }
        }

        fn system() -> System<ByteOperations> {
            let range_u8 = Circuit::from_air(LookupAir {
                inner_air: ByteOperations::RangeU8Chip,
                lookups: ByteOperations::RangeU8Chip.lookups(),
            })
            .unwrap();
            let byte_xor = Circuit::from_air(LookupAir {
                inner_air: ByteOperations::ByteXorChip,
                lookups: ByteOperations::ByteXorChip.lookups(),
            })
            .unwrap();
            System::new([range_u8, byte_xor])
        }
    }

    #[test]
    fn test_byte_operations() {
        let system = ByteOperations::system();
        let claim = [1, 0x01, 0x02, 0x03].map(Val::from_u32).to_vec();

        let f = Val::from_u8;
        let mut multiplicities = Val::zero_vec(256);
        // set multiplicities for the bytes from the claim
        multiplicities[0x01] = Val::ONE;
        multiplicities[0x02] = Val::ONE;
        multiplicities[0x03] = Val::ONE;

        let bytes: [Val; 256] = array::from_fn(|idx| Val::from_usize(idx));
        let byte_trace_values = multiplicities
            .iter()
            .zip(bytes.to_vec())
            .flat_map(|(x, y)| vec![*x, y])
            .collect();

        let byte_trace = RowMajorMatrix::<Val>::new(byte_trace_values, 2);

        let bytes: [u8; 256] = array::from_fn(|idx| idx as u8);

        let byte_xor_trace_values: Vec<Val> = bytes
            .into_iter()
            .flat_map(|x| {
                bytes.into_iter().map(move |y| {
                    vec![
                        Val::ONE,
                        Val::from_u8(x),
                        Val::from_u8(y),
                        Val::from_u8(x ^ y),
                    ]
                })
            })
            .flatten()
            .collect();
        assert_eq!(byte_xor_trace_values.len(), 256 * 256 * 4);

        let byte_xor_trace = RowMajorMatrix::<Val>::new(byte_xor_trace_values, 4);

        let witness = SystemWitness::from_stage_1(vec![byte_trace, byte_xor_trace], &system);

        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let config = new_stark_config(&fri_parameters);
        let proof = system.prove(&config, &claim, witness);
        system.verify(&config, &claim, &proof).unwrap();
    }

    //
    //
    // pub enum ByteCS {
    //     ByteChip,
    //     U32AddChip,
    // }
    //
    // // ByteChip should have a preprocessed column, but we can't do it yet
    // // it will have a column for multiplicity and a column for each byte
    // // Example
    // // | multiplicity | byte |
    // // |            9 |    0 |
    // // |            4 |    1 |
    // // |            8 |    2 |
    // // |            0 |    3 |
    // // |            3 |    4 |
    // // |            0 |    5 |
    // // ...
    //
    // impl<F> BaseAir<F> for ByteCS {
    //     fn width(&self) -> usize {
    //         match self {
    //             Self::ByteChip => 2,
    //             // 4 bytes for x, 4 bytes for y, 4 bytes for z, 1 byte for the carry, 1 column for the multiplicity
    //             Self::U32AddChip => 14,
    //         }
    //     }
    //
    //     fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
    //         match self {
    //             // eventually the byte column will be here
    //             Self::ByteChip => None,
    //             Self::U32AddChip => None,
    //         }
    //     }
    // }
    //
    // impl<AB> Air<AB> for ByteCS
    // where
    //     AB: AirBuilder,
    //     AB::Var: Copy,
    // {
    //     fn eval(&self, builder: &mut AB) {
    //         match self {
    //             Self::ByteChip => {
    //                 let main = builder.main();
    //                 let local = main.row_slice(0).unwrap();
    //                 let next = main.row_slice(1).unwrap();
    //                 let byte = &local[1];
    //                 let next_byte = &next[1];
    //                 builder.when_first_row().assert_zero(byte.clone());
    //                 builder
    //                     .when_transition()
    //                     .assert_eq(byte.clone() + AB::Expr::ONE, next_byte.clone());
    //                 builder
    //                     .when_last_row()
    //                     .assert_eq(byte.clone(), AB::Expr::from_u8(255));
    //             }
    //             Self::U32AddChip => {
    //                 let main = builder.main();
    //                 let local = main.row_slice(0).unwrap();
    //                 let x0 = &local[0];
    //                 let x1 = &local[1];
    //                 let x2 = &local[2];
    //                 let x3 = &local[3];
    //                 let y0 = &local[4];
    //                 let y1 = &local[5];
    //                 let y2 = &local[6];
    //                 let y3 = &local[7];
    //                 let z0 = &local[8];
    //                 let z1 = &local[9];
    //                 let z2 = &local[10];
    //                 let z3 = &local[11];
    //                 let carry = &local[12];
    //                 let _multiplicity = &local[13];
    //                 // the carry must be a boolean
    //                 builder.assert_bool(carry.clone());
    //
    //                 let expr1 = x0.clone()
    //                     + x1.clone() * AB::Expr::from_u32(256)
    //                     + x2.clone() * AB::Expr::from_u32(256 * 256)
    //                     + x3.clone() * AB::Expr::from_u32(256 * 256 * 256)
    //                     + y0.clone()
    //                     + y1.clone() * AB::Expr::from_u32(256)
    //                     + y2.clone() * AB::Expr::from_u32(256 * 256)
    //                     + y3.clone() * AB::Expr::from_u32(256 * 256 * 256);
    //                 let expr2 = z0.clone()
    //                     + z1.clone() * AB::Expr::from_u32(256)
    //                     + z2.clone() * AB::Expr::from_u32(256 * 256)
    //                     + z3.clone() * AB::Expr::from_u32(256 * 256 * 256)
    //                     + carry.clone() * AB::Expr::from_u64(256 * 256 * 256 * 256);
    //                 builder.assert_eq(expr1, expr2);
    //             }
    //         }
    //     }
    // }
    //
    // impl ByteCS {
    //     fn lookups(&self) -> Vec<Lookup<Expr>> {
    //         let var = |index| Expr::from(SymbolicVariable::new(Entry::Main { offset: 0 }, index));
    //         let byte_index = Expr::from_u8(0);
    //         let u32_index = Expr::from_u8(1);
    //         match self {
    //             Self::ByteChip => vec![
    //                 // Provide/Receive
    //                 Lookup {
    //                     multiplicity: -var(0),
    //                     args: vec![byte_index, var(1)],
    //                 },
    //             ],
    //             Self::U32AddChip => vec![
    //                 // Provide/Receive
    //                 Lookup {
    //                     multiplicity: -var(13),
    //                     args: vec![
    //                         u32_index,
    //                         var(0)
    //                             + var(1) * Expr::from_u32(256)
    //                             + var(2) * Expr::from_u32(256 * 256)
    //                             + var(3) * Expr::from_u32(256 * 256 * 256),
    //                         var(4)
    //                             + var(5) * Expr::from_u32(256)
    //                             + var(6) * Expr::from_u32(256 * 256)
    //                             + var(7) * Expr::from_u32(256 * 256 * 256),
    //                         var(8)
    //                             + var(9) * Expr::from_u32(256)
    //                             + var(10) * Expr::from_u32(256 * 256)
    //                             + var(11) * Expr::from_u32(256 * 256 * 256),
    //                     ],
    //                 },
    //                 // Require/Send
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index.clone(), var(0)],
    //                 },
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index.clone(), var(1)],
    //                 },
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index.clone(), var(2)],
    //                 },
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index.clone(), var(3)],
    //                 },
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index.clone(), var(4)],
    //                 },
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index.clone(), var(5)],
    //                 },
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index.clone(), var(6)],
    //                 },
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index.clone(), var(7)],
    //                 },
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index.clone(), var(8)],
    //                 },
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index.clone(), var(9)],
    //                 },
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index.clone(), var(10)],
    //                 },
    //                 Lookup {
    //                     multiplicity: Expr::ONE,
    //                     args: vec![byte_index, var(11)],
    //                 },
    //             ],
    //         }
    //     }
    // }
    //
    // fn byte_system() -> System<ByteCS> {
    //     let byte_chip = Circuit::from_air(LookupAir {
    //         inner_air: ByteCS::ByteChip,
    //         lookups: ByteCS::ByteChip.lookups(),
    //     })
    //         .unwrap();
    //     let u32_add_chip = Circuit::from_air(LookupAir {
    //         inner_air: ByteCS::U32AddChip,
    //         lookups: ByteCS::U32AddChip.lookups(),
    //     })
    //         .unwrap();
    //     System::new([byte_chip, u32_add_chip])
    // }
    //
    // pub struct AddCalls {
    //     pub calls: Vec<(u32, u32)>,
    // }
    //
    // impl AddCalls {
    //     pub fn witness(&self, system: &System<ByteCS>) -> SystemWitness {
    //         let byte_width = 2;
    //         let add_width = 14;
    //         let mut byte_trace = RowMajorMatrix::new(vec![Val::ZERO; byte_width * 256], byte_width);
    //         let add_height = add_width * self.calls.len().next_power_of_two();
    //         let mut add_trace = RowMajorMatrix::new(vec![Val::ZERO; add_height], add_width);
    //         self.traces(&mut byte_trace, &mut add_trace);
    //         let traces = vec![byte_trace, add_trace];
    //         SystemWitness::from_stage_1(traces, system)
    //     }
    //
    //     pub fn traces(
    //         &self,
    //         byte_trace: &mut RowMajorMatrix<Val>,
    //         add_trace: &mut RowMajorMatrix<Val>,
    //     ) {
    //         for i in 0..256 {
    //             byte_trace.row_mut(i)[1] = Val::from_usize(i);
    //         }
    //         for (row_index, (x, y)) in self.calls.iter().enumerate() {
    //             let x_bytes = x.to_le_bytes();
    //             let y_bytes = y.to_le_bytes();
    //             let (z, carry) = x.overflowing_add(*y);
    //             let z_bytes = z.to_le_bytes();
    //             let add_row = add_trace.row_mut(row_index);
    //             add_row[0..4]
    //                 .iter_mut()
    //                 .zip(x_bytes.iter())
    //                 .for_each(|(col, val)| *col = Val::from_u8(*val));
    //             add_row[4..8]
    //                 .iter_mut()
    //                 .zip(y_bytes.iter())
    //                 .for_each(|(col, val)| *col = Val::from_u8(*val));
    //             add_row[8..12]
    //                 .iter_mut()
    //                 .zip(z_bytes.iter())
    //                 .for_each(|(col, val)| *col = Val::from_u8(*val));
    //             add_row[12] = Val::from_u8(carry as u8);
    //             add_row[13] = Val::ONE;
    //             x_bytes.iter().for_each(|byte| {
    //                 byte_trace.row_mut(*byte as usize)[0] += Val::ONE;
    //             });
    //             y_bytes.iter().for_each(|byte| {
    //                 byte_trace.row_mut(*byte as usize)[0] += Val::ONE;
    //             });
    //             z_bytes.iter().for_each(|byte| {
    //                 byte_trace.row_mut(*byte as usize)[0] += Val::ONE;
    //             });
    //         }
    //     }
    // }
    //
    // #[test]
    // fn byte_trace() {
    //     let system = byte_system();
    //     let calls = AddCalls {
    //         calls: vec![(3, 4), (7, 9)],
    //     };
    //     let witness = calls.witness(&system);
    //     println!("BYTE TRACE");
    //     println!("{:?}", witness.traces[0]);
    //     println!("ADD TRACE");
    //     println!("{:?}", witness.traces[1]);
    // }
    //
    // #[test]
    // fn u32_add_proof() {
    //     let system = byte_system();
    //     let calls = AddCalls {
    //         calls: vec![(8000, 10000)],
    //     };
    //     let witness = calls.witness(&system);
    //
    //     println!("BYTE TRACE");
    //     println!("{:?}", witness.traces[0]);
    //
    //     println!("ADD TRACE");
    //     println!("{:?}", witness.traces[1]);
    //
    //     let claim = [1, 8000, 10000, 18000].map(Val::from_u32).to_vec();
    //     let fri_parameters = FriParameters {
    //         log_blowup: 1,
    //         log_final_poly_len: 0,
    //         num_queries: 64,
    //         proof_of_work_bits: 0,
    //     };
    //     let config = new_stark_config(&fri_parameters);
    //     let proof = system.prove(&config, &claim, witness);
    //     system.verify(&config, &claim, &proof).unwrap();
    // }
}
