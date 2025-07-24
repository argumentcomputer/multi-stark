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

    // Columns are: [multiplicity, A, B, A xor B, A and B, A or B], where A and B are bytes
    const TRACE_WIDTH: usize = 6;

    enum ByteOperations {
        XOR,
        AND,
        OR,
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

            // zero column is multiplicity. We don't need it here
            let a_curr = current[1].into();
            let b_curr = current[2].into();
            let a_next = next[1].into();
            let b_next = next[2].into();

            // we start from zero bytes
            builder.when_first_row().assert_zero(a_curr.clone());
            builder.when_first_row().assert_zero(b_curr.clone());

            // TODO: constrain a to be [0, 0, 0, 0,.... 1, 1, 1, 1... 2, 2, 2, 2...]
            // TODO: constrain b to be [0, 1, 2, ... 255, 0, 1, 2, ... 255, 0, 1, 2... 255]

            // we end with 0xff bytes
            builder
                .when_last_row()
                .assert_eq(a_curr.clone(), AB::Expr::from_u8(0xff));
            builder
                .when_last_row()
                .assert_eq(b_curr.clone(), AB::Expr::from_u8(0xff));
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
                Self::XOR => vec![Lookup {
                    multiplicity: -var(0),
                    args: vec![xor_idx, var(1), var(2), var(3)], // XOR result is stored in var(3)
                }],
                Self::AND => vec![Lookup {
                    multiplicity: -var(0),
                    args: vec![and_idx, var(1), var(2), var(4)], // AND result is stored in var(4)
                }],
                Self::OR => vec![Lookup {
                    multiplicity: -var(0),
                    args: vec![or_idx, var(1), var(2), var(5)], // OR result is stored in var(5)
                }],
            }
        }
        fn traces(&self, a: u8, b: u8, a_op_b: u8) -> RowMajorMatrix<Val> {
            let bytes: [u8; 256] = array::from_fn(|idx| idx as u8);
            let mut trace_values = Vec::with_capacity(256 * 256 * TRACE_WIDTH);
            for i in 0..256 {
                for j in 0..256 {
                    let multiplicity = match self {
                        Self::XOR => {
                            if (a as usize) == i && (b as usize) == j && (a_op_b as usize) == i ^ j
                            {
                                Val::ONE
                            } else {
                                Val::ZERO
                            }
                        }
                        Self::AND => {
                            if (a as usize) == i && (b as usize) == j && (a_op_b as usize) == i & j
                            {
                                Val::ONE
                            } else {
                                Val::ZERO
                            }
                        }
                        Self::OR => {
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
                }
            }
            RowMajorMatrix::new(trace_values, TRACE_WIDTH)
        }
    }

    #[test]
    fn xor() {
        let circuit = Circuit::from_air(LookupAir {
            inner_air: ByteOperations::XOR,
            lookups: ByteOperations::XOR.lookups(),
        })
        .unwrap();

        let system = System::new(vec![circuit]);
        let xor_chip_idx = 0;
        let a = 0x0f;
        let b = 0xf0;
        let a_xor_b = a ^ b;
        let claim = [xor_chip_idx, a, b, a_xor_b].map(Val::from_u8).to_vec();

        let witness =
            SystemWitness::from_stage_1(vec![ByteOperations::XOR.traces(a, b, a_xor_b)], &system);

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
            inner_air: ByteOperations::AND,
            lookups: ByteOperations::AND.lookups(),
        })
        .unwrap();

        let system = System::new(vec![circuit]);
        let and_chip_idx = 1;
        let a = 0x15;
        let b = 0x87;
        let a_and_b = a & b;
        let claim = [and_chip_idx, a, b, a_and_b].map(Val::from_u8).to_vec();

        let witness =
            SystemWitness::from_stage_1(vec![ByteOperations::AND.traces(a, b, a_and_b)], &system);

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
            inner_air: ByteOperations::OR,
            lookups: ByteOperations::OR.lookups(),
        })
        .unwrap();

        let system = System::new(vec![circuit]);
        let or_chip_idx = 2;
        let a = 0xf1;
        let b = 0x38;
        let a_or_b = a | b;
        let claim = [or_chip_idx, a, b, a_or_b].map(Val::from_u8).to_vec();

        let witness =
            SystemWitness::from_stage_1(vec![ByteOperations::OR.traces(a, b, a_or_b)], &system);

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
