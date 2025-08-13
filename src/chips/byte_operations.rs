#[cfg(test)]
mod tests {
    use crate::builder::symbolic::{preprocessed_var, var};
    use crate::chips::SymbExpr;
    use crate::lookup::{Lookup, LookupAir};
    use crate::system::{System, SystemWitness};
    use crate::types::{CommitmentParameters, FriParameters, Val};
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use std::array;

    // Preprocessed columns are: [A, B, A xor B, A and B, A or B], where A and B are bytes
    const PREPROCESSED_TRACE_WIDTH: usize = 5;
    // Main trace consists of multiplicities for each operation: `xor`, `and`, `or` and range check
    const TRACE_WIDTH: usize = 4;
    const BYTE_VALUES_NUM: usize = 256;

    struct ByteCS {}

    enum ByteOperation {
        Xor,
        And,
        Or,
        PairU8Range,
    }

    impl ByteOperation {
        fn position(&self) -> usize {
            match self {
                Self::Xor => 0,
                Self::And => 1,
                Self::Or => 2,
                Self::PairU8Range => 3,
            }
        }
    }

    impl<F: Field> BaseAir<F> for ByteCS {
        fn width(&self) -> usize {
            TRACE_WIDTH
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            let bytes: [u8; BYTE_VALUES_NUM] = array::from_fn(|idx| u8::try_from(idx).unwrap());
            let mut trace_values =
                Vec::with_capacity(BYTE_VALUES_NUM * BYTE_VALUES_NUM * PREPROCESSED_TRACE_WIDTH);
            for i in 0..256 {
                for j in 0..256 {
                    trace_values.push(F::from_u8(bytes[i]));
                    trace_values.push(F::from_u8(bytes[j]));
                    trace_values.push(F::from_u8(bytes[i] ^ bytes[j]));
                    trace_values.push(F::from_u8(bytes[i] & bytes[j]));
                    trace_values.push(F::from_u8(bytes[i] | bytes[j]));
                }
            }
            Some(RowMajorMatrix::new(trace_values, PREPROCESSED_TRACE_WIDTH))
        }
    }

    impl<AB> Air<AB> for ByteCS
    where
        AB: AirBuilder,
        AB::Var: Copy,
        AB::F: Field,
    {
        fn eval(&self, _builder: &mut AB) { /* no regular P3 constraints (we rely entirely on lookup) */
        }
    }

    impl ByteCS {
        fn lookups(&self) -> Vec<Lookup<SymbExpr>> {
            let xor_idx = ByteOperation::Xor.position();
            let and_idx = ByteOperation::And.position();
            let or_idx = ByteOperation::Or.position();
            let pair_range_check_idx = ByteOperation::PairU8Range.position();

            let mut lookups = [xor_idx, and_idx, or_idx]
                .into_iter()
                .map(|i| {
                    Lookup::pull(
                        var(i),
                        vec![
                            SymbExpr::from_usize(i),
                            preprocessed_var(0),
                            preprocessed_var(1),
                            preprocessed_var(2 + i),
                        ],
                    )
                })
                .collect::<Vec<_>>();
            // Range checks do not have a return value
            lookups.push(Lookup::pull(
                var(pair_range_check_idx),
                vec![
                    SymbExpr::from_usize(pair_range_check_idx),
                    preprocessed_var(0),
                    preprocessed_var(1),
                ],
            ));
            lookups
        }
    }

    struct ByteCalls {
        calls: Vec<(ByteOperation, u8, u8)>,
    }

    impl ByteCalls {
        fn witness(&self, system: &System<ByteCS>) -> SystemWitness {
            let mut byte_trace =
                RowMajorMatrix::new(vec![Val::ZERO; TRACE_WIDTH * 256 * 256], TRACE_WIDTH);
            for (op, x, y) in self.calls.iter() {
                let row_index = 256 * (*x as usize) + *y as usize;
                let row = byte_trace.row_mut(row_index);
                let position = op.position();
                row[position] += Val::ONE;
            }
            let traces = vec![byte_trace];
            SystemWitness::from_stage_1(traces, system)
        }
    }

    #[test]
    #[ignore]
    fn byte_test() {
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let circuit = LookupAir::new(ByteCS {}, ByteCS {}.lookups());
        let (system, key) = System::new(commitment_parameters, vec![circuit]);
        let calls = ByteCalls {
            calls: vec![
                (ByteOperation::Xor, 10, 5),
                (ByteOperation::And, 30, 20),
                (ByteOperation::Or, 100, 40),
                (ByteOperation::PairU8Range, 200, 100),
            ],
        };
        let witness = calls.witness(&system);
        let f = Val::from_u32;
        let claim1 = &[f(0), f(10), f(5), f(10 ^ 5)];
        let claim2 = &[f(1), f(30), f(20), f(30 & 20)];
        let claim3 = &[f(2), f(100), f(40), f(100 | 40)];
        let claim4 = &[f(3), f(200), f(100)];
        let claims: &[&[Val]] = &[claim1, claim2, claim3, claim4];
        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let proof = system.prove_multiple_claims(fri_parameters, &key, claims, witness);
        system
            .verify_multiple_claims(fri_parameters, claims, &proof)
            .unwrap();
    }
}
