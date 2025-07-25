use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{Matrix, dense::RowMajorMatrix};

use crate::{
    builder::symbolic::{Entry, SymbolicExpression, SymbolicVariable},
    lookup::{Lookup, LookupAir},
    system::{ProverKey, System, SystemWitness},
    types::{CommitmentParameters, Val},
};

pub enum ByteCS {
    ByteChip,
    U32AddChip,
}

// ByteChip should have a preprocessed column, but we can't do it yet
// it will have a column for multiplicity and a column for each byte
// Example
// | multiplicity | byte |
// |            9 |    0 |
// |            4 |    1 |
// |            8 |    2 |
// |            0 |    3 |
// |            3 |    4 |
// |            0 |    5 |
// ...

impl<F> BaseAir<F> for ByteCS {
    fn width(&self) -> usize {
        match self {
            Self::ByteChip => 2,
            // 4 bytes for x, 4 bytes for y, 4 bytes for z, 1 byte for the carry, 1 column for the multiplicity
            Self::U32AddChip => 14,
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            // eventually the byte column will be here
            Self::ByteChip => None,
            Self::U32AddChip => None,
        }
    }
}

impl<AB> Air<AB> for ByteCS
where
    AB: AirBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::ByteChip => {
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
            Self::U32AddChip => {
                let main = builder.main();
                let local = main.row_slice(0).unwrap();
                let x0 = &local[0];
                let x1 = &local[1];
                let x2 = &local[2];
                let x3 = &local[3];
                let y0 = &local[4];
                let y1 = &local[5];
                let y2 = &local[6];
                let y3 = &local[7];
                let z0 = &local[8];
                let z1 = &local[9];
                let z2 = &local[10];
                let z3 = &local[11];
                let carry = &local[12];
                let _multiplicity = &local[13];
                // the carry must be a boolean
                builder.assert_bool(carry.clone());

                let expr1 = x0.clone()
                    + x1.clone() * AB::Expr::from_u32(256)
                    + x2.clone() * AB::Expr::from_u32(256 * 256)
                    + x3.clone() * AB::Expr::from_u32(256 * 256 * 256)
                    + y0.clone()
                    + y1.clone() * AB::Expr::from_u32(256)
                    + y2.clone() * AB::Expr::from_u32(256 * 256)
                    + y3.clone() * AB::Expr::from_u32(256 * 256 * 256);
                let expr2 = z0.clone()
                    + z1.clone() * AB::Expr::from_u32(256)
                    + z2.clone() * AB::Expr::from_u32(256 * 256)
                    + z3.clone() * AB::Expr::from_u32(256 * 256 * 256)
                    + carry.clone() * AB::Expr::from_u64(256 * 256 * 256 * 256);
                builder.assert_eq(expr1, expr2);
            }
        }
    }
}

type Expr = SymbolicExpression<Val>;

impl ByteCS {
    pub fn lookups(&self) -> Vec<Lookup<Expr>> {
        let var = |index| Expr::from(SymbolicVariable::new(Entry::Main { offset: 0 }, index));
        let byte_index = Expr::from_u8(0);
        let u32_index = Expr::from_u8(1);
        match self {
            Self::ByteChip => vec![
                // Provide/Receive
                Lookup {
                    multiplicity: -var(0),
                    args: vec![byte_index, var(1)],
                },
            ],
            Self::U32AddChip => vec![
                // Provide/Receive
                Lookup {
                    multiplicity: -var(13),
                    args: vec![
                        u32_index,
                        var(0)
                            + var(1) * Expr::from_u32(256)
                            + var(2) * Expr::from_u32(256 * 256)
                            + var(3) * Expr::from_u32(256 * 256 * 256),
                        var(4)
                            + var(5) * Expr::from_u32(256)
                            + var(6) * Expr::from_u32(256 * 256)
                            + var(7) * Expr::from_u32(256 * 256 * 256),
                        var(8)
                            + var(9) * Expr::from_u32(256)
                            + var(10) * Expr::from_u32(256 * 256)
                            + var(11) * Expr::from_u32(256 * 256 * 256),
                    ],
                },
                // Require/Send
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index.clone(), var(0)],
                },
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index.clone(), var(1)],
                },
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index.clone(), var(2)],
                },
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index.clone(), var(3)],
                },
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index.clone(), var(4)],
                },
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index.clone(), var(5)],
                },
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index.clone(), var(6)],
                },
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index.clone(), var(7)],
                },
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index.clone(), var(8)],
                },
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index.clone(), var(9)],
                },
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index.clone(), var(10)],
                },
                Lookup {
                    multiplicity: Expr::ONE,
                    args: vec![byte_index, var(11)],
                },
            ],
        }
    }
}

pub fn byte_system(commitment_parameters: &CommitmentParameters) -> (System<ByteCS>, ProverKey) {
    let byte_chip = LookupAir {
        inner_air: ByteCS::ByteChip,
        lookups: ByteCS::ByteChip.lookups(),
    };
    let u32_add_chip = LookupAir {
        inner_air: ByteCS::U32AddChip,
        lookups: ByteCS::U32AddChip.lookups(),
    };
    System::new(commitment_parameters, [byte_chip, u32_add_chip])
}

pub struct AddCalls {
    pub calls: Vec<(u32, u32)>,
}

impl AddCalls {
    pub fn witness(&self, system: &System<ByteCS>) -> SystemWitness {
        let byte_width = 2;
        let add_width = 14;
        let mut byte_trace = RowMajorMatrix::new(vec![Val::ZERO; byte_width * 256], byte_width);
        let add_height = add_width * self.calls.len().next_power_of_two();
        let mut add_trace = RowMajorMatrix::new(vec![Val::ZERO; add_height], add_width);
        self.traces(&mut byte_trace, &mut add_trace);
        let traces = vec![byte_trace, add_trace];
        SystemWitness::from_stage_1(traces, system)
    }

    pub fn traces(
        &self,
        byte_trace: &mut RowMajorMatrix<Val>,
        add_trace: &mut RowMajorMatrix<Val>,
    ) {
        for i in 0..256 {
            byte_trace.row_mut(i)[1] = Val::from_usize(i);
        }
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
            add_row[12] = Val::from_u8(carry as u8);
            add_row[13] = Val::ONE;
            x_bytes.iter().for_each(|byte| {
                byte_trace.row_mut(*byte as usize)[0] += Val::ONE;
            });
            y_bytes.iter().for_each(|byte| {
                byte_trace.row_mut(*byte as usize)[0] += Val::ONE;
            });
            z_bytes.iter().for_each(|byte| {
                byte_trace.row_mut(*byte as usize)[0] += Val::ONE;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{FriParameters, StarkConfig};

    use super::*;

    #[test]
    fn byte_trace() {
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let (system, _key) = byte_system(&commitment_parameters);
        let calls = AddCalls {
            calls: vec![(3, 4), (7, 9), (2, 9)],
        };
        let witness = calls.witness(&system);
        println!("BYTE TRACE");
        println!("{:?}", witness.traces[0]);
        println!("ADD TRACE");
        println!("{:?}", witness.traces[1]);
    }

    #[test]
    fn u32_add_proof() {
        let commitment_parameters = CommitmentParameters { log_blowup: 1 };
        let (system, key) = byte_system(&commitment_parameters);
        let calls = AddCalls {
            calls: vec![(8000, 10000)],
        };
        let witness = calls.witness(&system);
        let claim = [1, 8000, 10000, 18000].map(Val::from_u32).to_vec();
        let fri_parameters = FriParameters {
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let config = StarkConfig::new(&commitment_parameters, &fri_parameters);
        let proof = system.prove(&config, &key, &claim, witness);
        system.verify(&config, &claim, &proof).unwrap();
    }
}
