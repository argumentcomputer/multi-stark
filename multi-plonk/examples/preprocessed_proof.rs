//! Example with preprocessed trace and lookups, ported to multi-plonk.
//!
//! Two circuits:
//! - **RangeTable**: read-only bytes 0..256 committed as a preprocessed trace.
//!   Each row pulls a lookup weighted by its multiplicity column.
//! - **Squares**: computes x * x and range-checks both bytes of the result via
//!   lookup pushes into RangeTable.
//!
//! Run with:
//! ```sh
//! cargo run --example preprocessed_proof --release
//! ```

use ark_ff::{One, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::test_rng;

use multi_plonk::air::{Air, AirBuilder, BaseAir};
use multi_plonk::builder::symbolic::{SymbolicExpression, preprocessed_var, var};
use multi_plonk::lookup::{Lookup, LookupAir};
use multi_plonk::matrix::Matrix;
use multi_plonk::system::{System, SystemWitness};
use multi_plonk::types::{PlonkConfig, Val};

enum SquaresCS {
    /// Preprocessed column: bytes 0..256. Main column: multiplicity.
    RangeTable,
    /// Columns: [x, x², low_byte, high_byte, multiplicity].
    Squares,
}

impl BaseAir for SquaresCS {
    fn width(&self) -> usize {
        match self {
            Self::RangeTable => 1,
            Self::Squares => 5,
        }
    }

    fn preprocessed_trace(&self) -> Option<Matrix<Val>> {
        match self {
            Self::RangeTable => Some(Matrix::new((0u64..256).map(Val::from).collect(), 1)),
            Self::Squares => None,
        }
    }
}

impl<AB> Air<AB> for SquaresCS
where
    AB: AirBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::RangeTable => {}
            Self::Squares => {
                let local = builder.main_local();
                let x = local[0];
                let x_sq = local[1];
                let low = local[2];
                let high = local[3];
                // x² == x * x
                builder.assert_eq(x_sq, x.into() * x.into());
                // x² == low + 256 * high  (byte decomposition)
                let c256: AB::Expr = Val::from(256u64).into();
                builder.assert_eq(x_sq, low.into() + high.into() * c256);
            }
        }
    }
}

impl SquaresCS {
    fn lookups(&self) -> Vec<Lookup<SymbolicExpression>> {
        match self {
            // RangeTable pulls: multiplicity × (preprocessed byte value)
            Self::RangeTable => vec![Lookup::pull(var(0), vec![preprocessed_var(0)])],
            // Squares pushes each byte of x² into the range table
            Self::Squares => vec![
                Lookup::push(var(4), vec![var(2)]), // low byte
                Lookup::push(var(4), vec![var(3)]), // high byte
            ],
        }
    }
}

fn main() {
    let mut rng = test_rng();
    // max_constraint_degree = 2 (x² = x*x), trace size up to 256 rows,
    // q_factor = 2, quotient_degree ≤ 256, preprocessed polys degree ≤ 255
    let config = PlonkConfig::setup(512, &mut rng);

    let range_table = LookupAir::new(SquaresCS::RangeTable, SquaresCS::RangeTable.lookups());
    let squares = LookupAir::new(SquaresCS::Squares, SquaresCS::Squares.lookups());
    let (system, key) = System::new(&config, [range_table, squares]);

    let n = 16usize;
    let f = |x: usize| Val::from(x as u64);

    // Range-table main trace: multiplicity per byte value (256 rows × 1 col)
    let mut range_mults = vec![Val::zero(); 256];
    // Squares trace: 16 rows × 5 cols
    let mut sq_values = Vec::with_capacity(5 * n);
    for x in 0..n {
        let sq = x * x;
        let low = sq & 0xFF;
        let high = (sq >> 8) & 0xFF;
        sq_values.extend([f(x), f(sq), f(low), f(high), Val::one()]);
        range_mults[low] += Val::one();
        range_mults[high] += Val::one();
    }

    let range_trace = Matrix::new(range_mults, 1);
    let squares_trace = Matrix::new(sq_values, 5);
    let witness = SystemWitness::from_stage_1(vec![range_trace, squares_trace], &system);

    let no_claims: &[&[Val]] = &[];
    let proof = system.prove_multiple_claims(&config, &key, no_claims, &witness);
    system
        .verify_multiple_claims(&config, no_claims, &proof)
        .expect("verify failed");
    println!("Preprocessed proof verified successfully!");

    let mut bytes = vec![];
    proof.serialize_uncompressed(&mut bytes).expect("serialize");
    println!(
        "Proof size: {} bytes (uncompressed), {} bytes (compressed)",
        bytes.len(),
        proof.compressed_size()
    );
}
