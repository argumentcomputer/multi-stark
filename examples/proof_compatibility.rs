//! Deterministic CPU/CUDA proof-byte compatibility workload.
//!
//! The default shape deliberately crosses every production CUDA size gate:
//! height > 1024 for resident FRI, more than 2^18 cells for GPU MMCS, and more
//! than 10 million source cells for parallel LDE waves. `cuda/smoke.sh` runs
//! this example once without and once with `cuda`, then compares the complete
//! serialized proof files.

use std::path::PathBuf;

use multi_stark::p3_adapter::LookupAir;
use multi_stark::system::{System, SystemWitness};
use multi_stark::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
use multi_stark::{
    p3_air::{Air, AirBuilder, BaseAir, WindowAccess},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::dense::RowMajorMatrix,
};

const WIDTH: usize = 40;
const LOG_HEIGHT: usize = 18;

struct WidePythagoreanAir;

impl<F> BaseAir<F> for WidePythagoreanAir {
    fn width(&self) -> usize {
        WIDTH
    }
}

impl<AB: AirBuilder> Air<AB> for WidePythagoreanAir
where
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        builder.assert_eq(
            local[0] * local[0] + local[1] * local[1],
            local[2] * local[2],
        );
        for value in &local[3..] {
            builder.assert_zero(*value);
        }
    }
}

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: proof_compatibility OUTPUT");
    let config = GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    );
    let (system, key) = System::new(config, [LookupAir::new(WidePythagoreanAir, vec![])]);
    let height = 1 << LOG_HEIGHT;
    let mut values = Val::zero_vec(height * WIDTH);
    for row in values.as_chunks_mut::<WIDTH>().0 {
        row[0] = Val::from_u8(3);
        row[1] = Val::from_u8(4);
        row[2] = Val::from_u8(5);
    }
    let proof = system.prove_multiple_claims(
        &key,
        &[],
        SystemWitness::from_stage_1(vec![RowMajorMatrix::new(values, WIDTH)], &system),
    );
    system.verify_multiple_claims(&[], &proof).unwrap();
    let bytes = proof.to_bytes().expect("proof serialization failed");
    std::fs::write(&output, &bytes).expect("failed to write proof bytes");
    println!("wrote {} bytes to {}", bytes.len(), output.display());
}
