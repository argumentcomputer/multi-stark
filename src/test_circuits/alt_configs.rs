//! Smoke tests for the alternative [`StarkGenericConfig`] instantiations
//! shipped by the crate ([`crate::koala_bear_poseidon2`]), sharing a single
//! config-generic harness: a minimal multiplication circuit is proven and
//! verified, and a tampered proof must be rejected.

use crate::builder::symbolic::{SymbolicExpression, var};
use crate::config::{StarkGenericConfig, Val};
use crate::koala_bear_poseidon2::KoalaBearPoseidon2Config;
use crate::lookup::{Lookup, LookupAir};
use crate::system::{System, SystemWitness};
use crate::types::{CommitmentParameters, FriParameters};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

/// A minimal AIR: enforces `a * b == c` per row, with a self-canceling
/// push/pull lookup pair to exercise the stage 2 machinery.
struct MulAir;

impl<F> BaseAir<F> for MulAir {
    fn width(&self) -> usize {
        3
    }
}

impl<AB> Air<AB> for MulAir
where
    AB: AirBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        builder.assert_eq(local[0] * local[1], local[2]);
    }
}

fn lookups<F: Field>() -> Vec<Lookup<SymbolicExpression<F>>> {
    let one: SymbolicExpression<F> = F::ONE.into();
    vec![
        Lookup::push(one.clone(), vec![var(0), var(2)]),
        Lookup::pull(one, vec![var(0), var(2)]),
    ]
}

fn smoke_test<SC: StarkGenericConfig>(config: SC) {
    let (system, key) = System::new(config, [LookupAir::new(MulAir, lookups())]);
    let f = Val::<SC>::from_u32;
    let trace = RowMajorMatrix::new(
        vec![
            f(2),
            f(3),
            f(6),
            f(4),
            f(5),
            f(20),
            f(7),
            f(8),
            f(56),
            f(0),
            f(0),
            f(0),
        ],
        3,
    );
    let witness = SystemWitness::from_stage_1(vec![trace], &system);
    let no_claims: &[&[Val<SC>]] = &[];
    let proof = system.prove_multiple_claims(&key, no_claims, witness);
    system
        .verify_multiple_claims(no_claims, &proof)
        .expect("proof failed to verify");

    // Tampered proofs must still be rejected under this config.
    let mut tampered = proof;
    tampered.intermediate_accumulators[0] += SC::Challenge::ONE;
    assert!(system.verify_multiple_claims(no_claims, &tampered).is_err());
}

fn test_parameters() -> (CommitmentParameters, FriParameters) {
    let commitment_parameters = CommitmentParameters {
        log_blowup: 1,
        cap_height: 0,
    };
    let fri_parameters = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 64,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };
    (commitment_parameters, fri_parameters)
}

#[test]
fn koala_bear_poseidon2_smoke_test() {
    let (commitment_parameters, fri_parameters) = test_parameters();
    smoke_test(KoalaBearPoseidon2Config::new(
        commitment_parameters,
        fri_parameters,
    ));
}
