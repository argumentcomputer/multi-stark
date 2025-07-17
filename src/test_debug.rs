#[cfg(test)]
mod tests {
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
}
