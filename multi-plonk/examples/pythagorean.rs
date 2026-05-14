//! End-to-end smoke test: prove `a² + b² == c²` over a 4-row trace using
//! the multi-plonk KZG-based prover/verifier on BLS12-381.

use ark_ff::Zero;
use ark_serialize::CanonicalSerialize;
use ark_std::test_rng;

use multi_plonk::air::{Air, AirBuilder, BaseAir};
use multi_plonk::lookup::LookupAir;
use multi_plonk::matrix::Matrix;
use multi_plonk::system::{System, SystemWitness};
use multi_plonk::types::{PlonkConfig, Val};

struct PythagoreanAir;

impl BaseAir for PythagoreanAir {
    fn width(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for PythagoreanAir {
    fn eval(&self, builder: &mut AB) {
        let row = builder.main_local();
        let a: AB::Expr = row[0].into();
        let b: AB::Expr = row[1].into();
        let c: AB::Expr = row[2].into();
        // a² + b² == c²
        builder.assert_eq(a.clone() * a + b.clone() * b, c.clone() * c);
    }
}

fn main() {
    let mut rng = test_rng();
    // largest expected polynomial degree fits comfortably below this:
    //   trace size n = 4, max constraint degree 2 → coset size 8,
    //   quotient degree ≤ 2n - 1 = 7.
    let config = PlonkConfig::setup(64, &mut rng);

    let air = PythagoreanAir;
    let width = air.width();
    let lookup_air = LookupAir::new(air, vec![]);
    let (system, key) = System::new(&config, [lookup_air]);

    let f = |x: u64| Val::from(x);
    let trace = Matrix::new(
        vec![
            f(3),
            f(4),
            f(5),
            f(5),
            f(12),
            f(13),
            f(8),
            f(15),
            f(17),
            f(7),
            f(24),
            f(25),
        ],
        width,
    );
    let witness = SystemWitness::from_stage_1(vec![trace], &system);

    let no_claims = &[];
    let proof = system.prove_multiple_claims(&config, &key, no_claims, &witness);
    println!(
        "intermediate accumulators: {:?}",
        proof.intermediate_accumulators
    );
    assert_eq!(
        proof.intermediate_accumulators.last().copied(),
        Some(Val::zero())
    );

    system
        .verify_multiple_claims(&config, no_claims, &proof)
        .expect("verify");
    println!("Verified.");

    let mut bytes = vec![];
    proof.serialize_uncompressed(&mut bytes).expect("serialize");
    let compressed_size = proof.compressed_size();
    println!(
        "Proof size: {} bytes (uncompressed), {} bytes (compressed)",
        bytes.len(),
        compressed_size
    );
}
