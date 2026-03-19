//! Example with a preprocessed trace and lookups.
//!
//! Defines two circuits:
//! - **RangeTable**: a read-only byte table (0..256) committed as a preprocessed
//!   trace. Each row pulls a lookup weighted by its multiplicity column.
//! - **Squares**: computes `x * x` and range-checks both bytes of the result
//!   via lookup pushes into the RangeTable.
//!
//! Run with:
//! ```sh
//! cargo run --example preprocessed_proof --release
//! ```

use multi_stark::builder::symbolic::{SymbolicExpression, preprocessed_var, var};
use multi_stark::lookup::{Lookup, LookupAir};
use multi_stark::system::{System, SystemWitness};
use multi_stark::types::{CommitmentParameters, FriParameters, Val};
use multi_stark::{
    p3_air::{Air, AirBuilder, BaseAir, WindowAccess},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::dense::RowMajorMatrix,
};

type SymbExpr = SymbolicExpression<Val>;

/// Two-circuit system: a preprocessed range table and a squaring circuit.
enum SquaresCS {
    /// Preprocessed column: bytes 0..256.  Main column: multiplicity.
    RangeTable,
    /// Columns: [x, x², low_byte, high_byte, multiplicity].
    Squares,
}

impl<F: Field> BaseAir<F> for SquaresCS {
    fn width(&self) -> usize {
        match self {
            Self::RangeTable => 1,
            Self::Squares => 5,
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            Self::RangeTable => Some(RowMajorMatrix::new((0..256).map(F::from_u32).collect(), 1)),
            Self::Squares => None,
        }
    }
}

impl<AB> Air<AB> for SquaresCS
where
    AB: AirBuilder,
    AB::Var: Copy,
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::RangeTable => {} // constrained entirely via lookups
            Self::Squares => {
                let main = builder.main();
                let local = main.current_slice();
                let x = local[0];
                let x_squared = local[1];
                let low = local[2];
                let high = local[3];
                // x² == x * x
                builder.assert_eq(x_squared, x * x);
                // x² == low + 256 * high  (byte decomposition)
                builder.assert_eq(x_squared, low + high * AB::Expr::from_u32(256));
            }
        }
    }
}

impl SquaresCS {
    fn lookups(&self) -> Vec<Lookup<SymbExpr>> {
        match self {
            // RangeTable pulls: multiplicity × (preprocessed byte value)
            Self::RangeTable => vec![Lookup::pull(var(0), vec![preprocessed_var(0)])],
            // Squares pushes each byte into the range table for validation
            Self::Squares => vec![
                Lookup::push(var(4), vec![var(2)]), // low byte
                Lookup::push(var(4), vec![var(3)]), // high byte
            ],
        }
    }
}

fn main() {
    let commitment_parameters = CommitmentParameters {
        log_blowup: 1,
        cap_height: 0,
    };

    let range_table = LookupAir::new(SquaresCS::RangeTable, SquaresCS::RangeTable.lookups());
    let squares = LookupAir::new(SquaresCS::Squares, SquaresCS::Squares.lookups());
    let (system, key) = System::new(commitment_parameters, [range_table, squares]);

    // Build traces: square every value 0..16
    let n = 16u32;
    let f = Val::from_u32;

    // Range-table main trace: multiplicity per byte value (256 rows × 1 col)
    let mut range_mults = vec![Val::ZERO; 256];
    // Squares trace: 16 rows × 5 cols
    let mut sq_values = Vec::with_capacity(5 * n as usize);
    for x in 0..n {
        let sq = x * x;
        let low = sq & 0xFF;
        let high = (sq >> 8) & 0xFF;
        sq_values.extend([f(x), f(sq), f(low), f(high), Val::ONE]);
        range_mults[low as usize] += Val::ONE;
        range_mults[high as usize] += Val::ONE;
    }

    let range_trace = RowMajorMatrix::new(range_mults, 1);
    let squares_trace = RowMajorMatrix::new(sq_values, 5);
    let witness = SystemWitness::from_stage_1(vec![range_trace, squares_trace], &system);

    let fri_parameters = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 64,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };

    let no_claims: &[&[Val]] = &[];
    let proof = system.prove_multiple_claims(fri_parameters, &key, no_claims, witness);
    system
        .verify_multiple_claims(fri_parameters, no_claims, &proof)
        .unwrap();
    println!("Preprocessed proof verified successfully!");

    let bytes = proof.to_bytes().expect("serialization failed");
    println!("Proof size: {} bytes", bytes.len());
}
