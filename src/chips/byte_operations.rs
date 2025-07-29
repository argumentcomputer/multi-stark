#[cfg(test)]
mod tests {
    use crate::builder::symbolic::{Entry, SymbolicExpression, SymbolicVariable};
    use crate::lookup::{Lookup, LookupAir};
    use crate::system::{System, SystemWitness};
    use crate::types::{CommitmentParameters, FriParameters, Val};
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use std::array;

    // Preprocessed columns are: [A, B, A xor B, A and B, A or B], where A and B are bytes
    const PREPROCESSED_TRACE_WIDTH: usize = 5;
    const BYTE_VALUES_NUM: usize = 256;

    enum ByteOperations {
        Xor,
        And,
        Or,
    }

    impl<F: Field> BaseAir<F> for ByteOperations {
        fn width(&self) -> usize {
            1
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

    impl<AB> Air<AB> for ByteOperations
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, _builder: &mut AB) { /* no regular P3 constraints (we rely entirely on lookup) */
        }
    }

    impl ByteOperations {
        fn lookups(&self) -> Vec<Lookup<SymbolicExpression<Val>>> {
            let var =
                |i| SymbolicExpression::from(SymbolicVariable::new(Entry::Main { offset: 0 }, i));

            let preprocessed_var = |i| {
                SymbolicExpression::from(SymbolicVariable::new(
                    Entry::Preprocessed { offset: 0 },
                    i,
                ))
            };

            let xor_idx = SymbolicExpression::<Val>::from_u8(0u8);
            let and_idx = SymbolicExpression::<Val>::from_u8(1u8);
            let or_idx = SymbolicExpression::<Val>::from_u8(2u8);

            // we have to provide exactly one lookup: [A, B, A op B], depending on the required operation in order to balance the claim
            match self {
                Self::Xor => vec![Lookup {
                    multiplicity: -var(0),
                    args: vec![
                        xor_idx,
                        preprocessed_var(0),
                        preprocessed_var(1),
                        preprocessed_var(2),
                    ], // XOR result is stored in var(2)
                }],
                Self::And => vec![Lookup {
                    multiplicity: -var(0),
                    args: vec![
                        and_idx,
                        preprocessed_var(0),
                        preprocessed_var(1),
                        preprocessed_var(3),
                    ], // AND result is stored in var(4)
                }],
                Self::Or => vec![Lookup {
                    multiplicity: -var(0),
                    args: vec![
                        or_idx,
                        preprocessed_var(0),
                        preprocessed_var(1),
                        preprocessed_var(4),
                    ], // OR result is stored in var(5)
                }],
            }
        }
        fn multiplicity_trace(&self, a: u8, b: u8, a_op_b: u8) -> RowMajorMatrix<Val> {
            let mut trace_values = Vec::with_capacity(BYTE_VALUES_NUM * BYTE_VALUES_NUM);
            for i in 0..BYTE_VALUES_NUM {
                for j in 0..BYTE_VALUES_NUM {
                    let multiplicity = match self {
                        Self::Xor => {
                            if (a as usize) == i && (b as usize) == j && (a_op_b as usize) == i ^ j
                            {
                                Val::ONE
                            } else {
                                Val::ZERO
                            }
                        }
                        Self::And => {
                            if (a as usize) == i && (b as usize) == j && (a_op_b as usize) == i & j
                            {
                                Val::ONE
                            } else {
                                Val::ZERO
                            }
                        }
                        Self::Or => {
                            if (a as usize) == i && (b as usize) == j && (a_op_b as usize) == i | j
                            {
                                Val::ONE
                            } else {
                                Val::ZERO
                            }
                        }
                    };
                    trace_values.push(multiplicity)
                }
            }

            RowMajorMatrix::new(trace_values, 1)
        }
    }

    #[test]
    fn xor() {
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let circuit = LookupAir::new(ByteOperations::Xor, ByteOperations::Xor.lookups());
        let (system, prover_key) = System::new(commitment_parameters, vec![circuit]);

        let xor_chip_idx = 0;
        let a = 0x0f;
        let b = 0xf0;
        let a_xor_b = a ^ b;
        let claim = [xor_chip_idx, a, b, a_xor_b].map(Val::from_u8).to_vec();

        let witness = SystemWitness::from_stage_1(
            vec![ByteOperations::Xor.multiplicity_trace(a, b, a_xor_b)],
            &system,
        );

        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 1,
        };

        let proof = system.prove(fri_parameters, &prover_key, &claim, witness);
        system
            .verify(fri_parameters, &claim, &proof)
            .expect("verification issue");
    }

    #[test]
    fn and() {
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let circuit = LookupAir::new(ByteOperations::And, ByteOperations::And.lookups());
        let (system, prover_key) = System::new(commitment_parameters, vec![circuit]);
        let and_chip_idx = 1;
        let a = 0x15;
        let b = 0x87;
        let a_and_b = a & b;
        let claim = [and_chip_idx, a, b, a_and_b].map(Val::from_u8).to_vec();

        let witness = SystemWitness::from_stage_1(
            vec![ByteOperations::And.multiplicity_trace(a, b, a_and_b)],
            &system,
        );

        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 1,
        };
        let proof = system.prove(fri_parameters, &prover_key, &claim, witness);
        system
            .verify(fri_parameters, &claim, &proof)
            .expect("verification issue");
    }

    #[test]
    fn or() {
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let circuit = LookupAir::new(ByteOperations::Or, ByteOperations::Or.lookups());

        let (system, prover_key) = System::new(commitment_parameters, vec![circuit]);
        let or_chip_idx = 2;
        let a = 0xf1;
        let b = 0x38;
        let a_or_b = a | b;
        let claim = [or_chip_idx, a, b, a_or_b].map(Val::from_u8).to_vec();

        let witness = SystemWitness::from_stage_1(
            vec![ByteOperations::Or.multiplicity_trace(a, b, a_or_b)],
            &system,
        );

        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 1,
        };
        let proof = system.prove(fri_parameters, &prover_key, &claim, witness);
        system
            .verify(fri_parameters, &claim, &proof)
            .expect("verification issue");
    }
}
