#[cfg(test)]
mod tests {
    use crate::builder::symbolic::{Entry, SymbolicExpression, SymbolicVariable};
    use crate::lookup::{Lookup, LookupAir};
    use crate::system::{Circuit, System, SystemWitness};
    use crate::types::{FriParameters, Val, new_stark_config};
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use std::array;

    // Columns are: [multiplicity, A, B, A xor B, A and B, A or B, row_index, row_index / 256], where A and B are bytes
    const TRACE_WIDTH: usize = 8;
    const BYTE_VALUES_NUM: usize = 256;

    enum ByteOperations {
        Xor,
        And,
        Or,
    }

    impl<F> BaseAir<F> for ByteOperations {
        fn width(&self) -> usize {
            TRACE_WIDTH
        }
    }

    impl<AB> Air<AB> for ByteOperations
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let current = main.row_slice(0).unwrap();
            let next = main.row_slice(1).unwrap();

            let a_curr = current[1].into();
            let b_curr = current[2].into();

            let row_index = current[6].into();
            let row_index_next = next[6].into();
            let row_index_div_256 = current[7].into();

            // we start from zero bytes
            builder.when_first_row().assert_zero(a_curr.clone());
            builder.when_first_row().assert_zero(b_curr.clone());

            // we end with 0xff bytes
            builder
                .when_last_row()
                .assert_eq(a_curr.clone(), AB::Expr::from_u8(u8::MAX));
            builder
                .when_last_row()
                .assert_eq(b_curr.clone(), AB::Expr::from_u8(u8::MAX));

            // we check that row index increments by one with each new row
            builder
                .when_transition()
                .assert_eq(row_index.clone() + AB::Expr::ONE, row_index_next);

            // we check that A is [0,0,0,0 ... , 1,1,1,1 ... , 2,2,2,2 ... , ... 255, 255, 255]
            builder.assert_eq(a_curr.clone(), row_index_div_256.clone());

            // we check that B is [0, 1, 2, ... 255, 0, 1, 2, ... 255, 0, 1, 2... 255]
            builder.assert_eq(
                b_curr.clone(),
                row_index.clone()
                    - (AB::Expr::from_u32(u32::try_from(BYTE_VALUES_NUM).unwrap())
                        * row_index_div_256),
            );
        }
    }

    impl ByteOperations {
        fn lookups(&self) -> Vec<Lookup<SymbolicExpression<Val>>> {
            let var =
                |i| SymbolicExpression::from(SymbolicVariable::new(Entry::Main { offset: 0 }, i));
            let xor_idx = SymbolicExpression::<Val>::from_u8(0u8);
            let and_idx = SymbolicExpression::<Val>::from_u8(1u8);
            let or_idx = SymbolicExpression::<Val>::from_u8(2u8);

            // we have to provide exactly one lookup: [A, B, A op B], depending on the required operation in order to balance the claim
            match self {
                Self::Xor => vec![Lookup {
                    multiplicity: -var(0),
                    args: vec![xor_idx, var(1), var(2), var(3)], // XOR result is stored in var(3)
                }],
                Self::And => vec![Lookup {
                    multiplicity: -var(0),
                    args: vec![and_idx, var(1), var(2), var(4)], // AND result is stored in var(4)
                }],
                Self::Or => vec![Lookup {
                    multiplicity: -var(0),
                    args: vec![or_idx, var(1), var(2), var(5)], // OR result is stored in var(5)
                }],
            }
        }
        fn traces(&self, a: u8, b: u8, a_op_b: u8) -> RowMajorMatrix<Val> {
            let bytes: [u8; BYTE_VALUES_NUM] = array::from_fn(|idx| u8::try_from(idx).unwrap());
            let mut trace_values =
                Vec::with_capacity(BYTE_VALUES_NUM * BYTE_VALUES_NUM * TRACE_WIDTH);
            let mut row_index = 0;
            for i in 0..256 {
                for j in 0..256 {
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

                    trace_values.push(multiplicity);
                    trace_values.push(Val::from_u8(bytes[i]));
                    trace_values.push(Val::from_u8(bytes[j]));
                    trace_values.push(Val::from_u8(bytes[i] ^ bytes[j]));
                    trace_values.push(Val::from_u8(bytes[i] & bytes[j]));
                    trace_values.push(Val::from_u8(bytes[i] | bytes[j]));
                    trace_values.push(Val::from_u32(row_index));
                    trace_values.push(Val::from_u32(
                        row_index / u32::try_from(BYTE_VALUES_NUM).unwrap(),
                    ));

                    row_index += 1;
                }
            }

            RowMajorMatrix::new(trace_values, TRACE_WIDTH)
        }
    }

    #[test]
    fn xor() {
        let circuit = Circuit::from_air(LookupAir {
            inner_air: ByteOperations::Xor,
            lookups: ByteOperations::Xor.lookups(),
        })
        .unwrap();

        let system = System::new(vec![circuit]);
        let xor_chip_idx = 0;
        let a = 0x0f;
        let b = 0xf0;
        let a_xor_b = a ^ b;
        let claim = [xor_chip_idx, a, b, a_xor_b].map(Val::from_u8).to_vec();

        let witness =
            SystemWitness::from_stage_1(vec![ByteOperations::Xor.traces(a, b, a_xor_b)], &system);

        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 1,
        };

        let config = new_stark_config(&fri_parameters);
        let proof = system.prove(&config, &claim, witness);
        system
            .verify(&config, &claim, &proof)
            .expect("verification issue");
    }

    #[test]
    fn and() {
        let circuit = Circuit::from_air(LookupAir {
            inner_air: ByteOperations::And,
            lookups: ByteOperations::And.lookups(),
        })
        .unwrap();

        let system = System::new(vec![circuit]);
        let and_chip_idx = 1;
        let a = 0x15;
        let b = 0x87;
        let a_and_b = a & b;
        let claim = [and_chip_idx, a, b, a_and_b].map(Val::from_u8).to_vec();

        let witness =
            SystemWitness::from_stage_1(vec![ByteOperations::And.traces(a, b, a_and_b)], &system);

        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 1,
        };
        let config = new_stark_config(&fri_parameters);
        let proof = system.prove(&config, &claim, witness);
        system
            .verify(&config, &claim, &proof)
            .expect("verification issue");
    }

    #[test]
    fn or() {
        let circuit = Circuit::from_air(LookupAir {
            inner_air: ByteOperations::Or,
            lookups: ByteOperations::Or.lookups(),
        })
        .unwrap();

        let system = System::new(vec![circuit]);
        let or_chip_idx = 2;
        let a = 0xf1;
        let b = 0x38;
        let a_or_b = a | b;
        let claim = [or_chip_idx, a, b, a_or_b].map(Val::from_u8).to_vec();

        let witness =
            SystemWitness::from_stage_1(vec![ByteOperations::Or.traces(a, b, a_or_b)], &system);

        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 1,
        };
        let config = new_stark_config(&fri_parameters);
        let proof = system.prove(&config, &claim, witness);
        system
            .verify(&config, &claim, &proof)
            .expect("verification issue");
    }
}
