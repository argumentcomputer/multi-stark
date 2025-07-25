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
}
