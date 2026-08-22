//! A second [`StarkGenericConfig`] instantiation — BabyBear field with a
//! degree-4 binomial extension and Poseidon2 hashing — differing from the
//! reference Goldilocks/Keccak config in both the field and the hash axes.
//!
//! This module exists to prove that the protocol is actually generic: if a
//! change compiles only for the reference config, the smoke test here
//! catches it.

use crate::config::StarkGenericConfig;
use crate::lookup::Lookup;
use crate::p3_adapter::{LookupAir, SymbolicExpression, var};
use crate::system::{System, SystemWitness};
use crate::types::{CommitmentParameters, FriParameters};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, DuplexChallenger};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_fri::{FriParameters as InnerFriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Val = BabyBear;
type Perm = Poseidon2BabyBear<16>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, Hash, Compress, 2, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

struct BabyBearPoseidon2Config {
    pcs: Pcs,
    perm: Perm,
    /// Field elements observed into every fresh challenger: a domain tag
    /// plus a digest of the protocol parameters (see the transcript contract
    /// on [`StarkGenericConfig::initialise_challenger`]).
    challenger_seed: Vec<Val>,
    max_log_degree: usize,
    max_quotient_degree: usize,
    log_blowup: usize,
}

impl BabyBearPoseidon2Config {
    fn new(commitment_parameters: CommitmentParameters, fri_parameters: FriParameters) -> Self {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = Hash::new(perm.clone());
        let compress = Compress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress, commitment_parameters.cap_height);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let inner_parameters = InnerFriParameters {
            log_blowup: commitment_parameters.log_blowup,
            log_final_poly_len: fri_parameters.log_final_poly_len,
            max_log_arity: fri_parameters.max_log_arity,
            num_queries: fri_parameters.num_queries,
            commit_proof_of_work_bits: fri_parameters.commit_proof_of_work_bits,
            query_proof_of_work_bits: fri_parameters.query_proof_of_work_bits,
            mmcs: challenge_mmcs,
        };
        let pcs = Pcs::new(Dft::default(), val_mmcs, inner_parameters);
        let mut challenger_seed: Vec<Val> = b"multi-stark/v0"
            .iter()
            .map(|&byte| Val::from_u8(byte))
            .collect();
        challenger_seed.extend(
            [
                commitment_parameters.log_blowup,
                commitment_parameters.cap_height,
                fri_parameters.log_final_poly_len,
                fri_parameters.max_log_arity,
                fri_parameters.num_queries,
                fri_parameters.commit_proof_of_work_bits,
                fri_parameters.query_proof_of_work_bits,
            ]
            .map(Val::from_usize),
        );
        let max_log_degree = Val::TWO_ADICITY - commitment_parameters.log_blowup;
        let max_quotient_degree = 1 << commitment_parameters.log_blowup;
        Self {
            pcs,
            perm,
            challenger_seed,
            max_log_degree,
            max_quotient_degree,
            log_blowup: commitment_parameters.log_blowup,
        }
    }
}

impl StarkGenericConfig for BabyBearPoseidon2Config {
    type Pcs = Pcs;
    type Challenge = Challenge;
    type Challenger = Challenger;

    fn pcs(&self) -> &Pcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Challenger {
        let mut challenger = Challenger::new(self.perm.clone());
        for value in &self.challenger_seed {
            challenger.observe(*value);
        }
        challenger
    }

    fn max_log_degree(&self) -> usize {
        self.max_log_degree
    }

    fn max_quotient_degree(&self) -> usize {
        self.max_quotient_degree
    }

    fn log_blowup(&self) -> usize {
        self.log_blowup
    }
}

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

fn lookups() -> Vec<Lookup<SymbolicExpression<Val>>> {
    let one: SymbolicExpression<Val> = Val::ONE.into();
    vec![
        Lookup::push(one.clone(), vec![var(0), var(2)]),
        Lookup::pull(one, vec![var(0), var(2)]),
    ]
}

#[test]
fn baby_bear_poseidon2_smoke_test() {
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
    let config = BabyBearPoseidon2Config::new(commitment_parameters, fri_parameters);
    let (system, key) = System::new(config, [LookupAir::new(MulAir, lookups())]);
    let f = Val::from_u32;
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
    let no_claims: &[&[Val]] = &[];
    let proof = system.prove_multiple_claims(&key, no_claims, witness);
    system
        .verify_multiple_claims(no_claims, &proof)
        .expect("BabyBear/Poseidon2 proof failed to verify");

    // Tampered proofs must still be rejected under this config.
    let mut tampered = proof;
    tampered.intermediate_accumulators[0] += Challenge::ONE;
    assert!(system.verify_multiple_claims(no_claims, &tampered).is_err());
}
