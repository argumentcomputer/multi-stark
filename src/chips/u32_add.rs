#[cfg(test)]
mod tests {
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::{Matrix, dense::RowMajorMatrix};

    use crate::chips::SymbExpr;
    use crate::{
        builder::symbolic::{preprocessed_var, var},
        lookup::{Lookup, LookupAir},
        system::{ProverKey, System, SystemWitness},
        types::{CommitmentParameters, FriParameters, Val},
    };

    enum U32CS {
        ByteChip,
        U32AddChip,
    }

    // ByteChip will have a preprocessed column for the bytes and
    // a column for the multiplicities
    // Example
    // | multiplicity | byte |
    // |            9 |    0 |
    // |            4 |    1 |
    // |            8 |    2 |
    // |            0 |    3 |
    // |            3 |    4 |
    // |            0 |    5 |
    // ...

    impl<F: Field> BaseAir<F> for U32CS {
        fn width(&self) -> usize {
            match self {
                Self::ByteChip => 1,
                // 4 bytes for x, 4 bytes for y, 4 bytes for z, 1 byte for the carry, 1 column for the multiplicity
                Self::U32AddChip => 14,
            }
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            match self {
                Self::ByteChip => Some(RowMajorMatrix::new((0..256).map(F::from_u32).collect(), 1)),
                Self::U32AddChip => None,
            }
        }
    }

    impl<AB> Air<AB> for U32CS
    where
        AB: AirBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                Self::ByteChip => {}
                Self::U32AddChip => {
                    let main = builder.main();
                    let local = main.row_slice(0).unwrap();
                    let x = &local[0..4];
                    let y = &local[4..8];
                    let z = &local[8..12];
                    let carry = local[12];
                    // the carry must be a boolean
                    builder.assert_bool(carry);

                    let expr1 = x[0]
                        + x[1] * AB::Expr::from_u32(256)
                        + x[2] * AB::Expr::from_u32(256 * 256)
                        + x[3] * AB::Expr::from_u32(256 * 256 * 256)
                        + y[0]
                        + y[1] * AB::Expr::from_u32(256)
                        + y[2] * AB::Expr::from_u32(256 * 256)
                        + y[3] * AB::Expr::from_u32(256 * 256 * 256);
                    let expr2 = z[0]
                        + z[1] * AB::Expr::from_u32(256)
                        + z[2] * AB::Expr::from_u32(256 * 256)
                        + z[3] * AB::Expr::from_u32(256 * 256 * 256)
                        + carry * AB::Expr::from_u64(256 * 256 * 256 * 256);
                    builder.assert_eq(expr1, expr2);
                }
            }
        }
    }

    impl U32CS {
        fn lookups(&self) -> Vec<Lookup<SymbExpr>> {
            let byte_index = SymbExpr::from_u8(0);
            let u32_index = SymbExpr::from_u8(1);
            match self {
                Self::ByteChip => vec![Lookup::pull(var(0), vec![byte_index, preprocessed_var(0)])],
                Self::U32AddChip => {
                    // Pull
                    let mut lookups = vec![Lookup::pull(
                        var(13),
                        vec![
                            u32_index,
                            var(0)
                                + var(1) * SymbExpr::from_u32(256)
                                + var(2) * SymbExpr::from_u32(256 * 256)
                                + var(3) * SymbExpr::from_u32(256 * 256 * 256),
                            var(4)
                                + var(5) * SymbExpr::from_u32(256)
                                + var(6) * SymbExpr::from_u32(256 * 256)
                                + var(7) * SymbExpr::from_u32(256 * 256 * 256),
                            var(8)
                                + var(9) * SymbExpr::from_u32(256)
                                + var(10) * SymbExpr::from_u32(256 * 256)
                                + var(11) * SymbExpr::from_u32(256 * 256 * 256),
                        ],
                    )];
                    // Push
                    lookups
                        .extend((0..12).map(|i| {
                            Lookup::push(SymbExpr::ONE, vec![byte_index.clone(), var(i)])
                        }));
                    lookups
                }
            }
        }
    }

    fn byte_system(commitment_parameters: CommitmentParameters) -> (System<U32CS>, ProverKey) {
        let byte_chip = LookupAir::new(U32CS::ByteChip, U32CS::ByteChip.lookups());
        let u32_add_chip = LookupAir::new(U32CS::U32AddChip, U32CS::U32AddChip.lookups());
        System::new(commitment_parameters, [byte_chip, u32_add_chip])
    }

    struct AddCalls {
        calls: Vec<(u32, u32)>,
    }

    impl AddCalls {
        fn witness(&self, system: &System<U32CS>) -> SystemWitness {
            let byte_width = 1;
            let add_width = 14;
            let mut byte_trace = RowMajorMatrix::new(vec![Val::ZERO; byte_width * 256], byte_width);
            let add_height = add_width * self.calls.len().next_power_of_two();
            let mut add_trace = RowMajorMatrix::new(vec![Val::ZERO; add_height], add_width);
            self.traces(&mut byte_trace, &mut add_trace);
            let traces = vec![byte_trace, add_trace];
            SystemWitness::from_stage_1(traces, system)
        }

        fn traces(
            &self,
            byte_trace: &mut RowMajorMatrix<Val>,
            add_trace: &mut RowMajorMatrix<Val>,
        ) {
            for (row_index, (x, y)) in self.calls.iter().enumerate() {
                let x_bytes = x.to_le_bytes();
                let y_bytes = y.to_le_bytes();
                let (z, carry) = x.overflowing_add(*y);
                let z_bytes = z.to_le_bytes();
                let add_row = add_trace.row_mut(row_index);
                add_row[0..4]
                    .iter_mut()
                    .zip(x_bytes.iter())
                    .for_each(|(col, val)| *col = Val::from_u8(*val));
                add_row[4..8]
                    .iter_mut()
                    .zip(y_bytes.iter())
                    .for_each(|(col, val)| *col = Val::from_u8(*val));
                add_row[8..12]
                    .iter_mut()
                    .zip(z_bytes.iter())
                    .for_each(|(col, val)| *col = Val::from_u8(*val));
                add_row[12] = Val::from_u8(u8::from(carry));
                add_row[13] = Val::ONE;
                for byte in x_bytes.iter() {
                    byte_trace.row_mut(*byte as usize)[0] += Val::ONE;
                }
                for byte in y_bytes.iter() {
                    byte_trace.row_mut(*byte as usize)[0] += Val::ONE;
                }
                for byte in z_bytes.iter() {
                    byte_trace.row_mut(*byte as usize)[0] += Val::ONE;
                }
            }
        }
    }

    #[test]
    fn u32_add_proof() {
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let (system, key) = byte_system(commitment_parameters);
        let calls = AddCalls {
            calls: vec![(10, 5), (30, 20), (100, 100), (8000, 10000)],
        };
        let witness = calls.witness(&system);
        let f = Val::from_u32;
        let claim1 = &[f(1), f(10), f(5), f(15)];
        let claim2 = &[f(1), f(30), f(20), f(50)];
        let claim3 = &[f(1), f(100), f(100), f(200)];
        let claim4 = &[f(1), f(8000), f(10000), f(18000)];
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
