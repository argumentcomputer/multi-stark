//! Public-bound Goldilocks quadratic folding, the arithmetic part of a FRI verifier.
use crate::goldilocks::{GoldGate, Operation, P, Relation, packed_witness};
use flock_prover::{
    challenger::FsChallenger,
    circuit::builder::ShapeBuilder,
    field::F128,
    lincheck::LincheckCircuit,
    pcs::{
        PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k_or_default},
    },
    prover::{UnionSlotProverInput, prove_fast_ligerito_union_circuit},
    union::UnionInstance,
    verifier::verify_ligerito_union_circuit,
};
use serde_json::json;
use std::{fs, path::Path, sync::Arc, time::Instant};
const DOMAIN: &[u8] = b"multi-stark-flock-quadratic-fold-v1";

pub fn run(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let _ = flock_prover::init_perf_thread_pool();
    let start = Instant::now();
    let nu = 7;
    let add = Arc::new(Relation::new(Operation::Add));
    let mul = Arc::new(Relation::new(Operation::Mul));
    let mut b = ShapeBuilder::new(nu);
    let add_slot = b.slot(GoldGate {
        relation: add.clone(),
        nu,
    });
    let mul_slot = b.slot(GoldGate {
        relation: mul.clone(),
        nu,
    });
    let beta = [b.public_input(), b.public_input()];
    let mut acc = [b.public_input(), b.public_input()];
    let seven = b.fixed_public_input(F128::new(7, 0));
    let mut inputs = vec![
        F128::new(13, 0),
        F128::new(17, 0),
        F128::new(P - 1, 0),
        F128::new(99, 0),
        F128::new(7, 0),
    ];
    let mut native = [P - 1, 99];
    let add_native = |a: u64, b: u64| ((u128::from(a) + u128::from(b)) % u128::from(P)) as u64;
    let mul_native = |a: u64, b: u64| ((u128::from(a) * u128::from(b)) % u128::from(P)) as u64;
    // Horner recurrence acc <- acc*beta + coefficient in F_p[u]/(u^2-7).
    // This is a component test: no Merkle membership or Fiat-Shamir is claimed.
    for i in 0..16 {
        let coefficient = [b.input(), b.input()];
        inputs.extend([F128::new(i + 1, 0), F128::new(i + 2, 0)]);
        let t0 = b.gate(mul_slot, &[acc[0], beta[0]])[0];
        let t1 = b.gate(mul_slot, &[acc[1], beta[1]])[0];
        let t2 = b.gate(mul_slot, &[acc[0], beta[1]])[0];
        let t3 = b.gate(mul_slot, &[acc[1], beta[0]])[0];
        let t1 = b.gate(mul_slot, &[t1, seven])[0];
        let real = b.gate(add_slot, &[t0, t1])[0];
        let imag = b.gate(add_slot, &[t2, t3])[0];
        acc = [
            b.gate(add_slot, &[real, coefficient[0]])[0],
            b.gate(add_slot, &[imag, coefficient[1]])[0],
        ];
        native = [
            add_native(
                add_native(
                    mul_native(native[0], 13),
                    mul_native(mul_native(native[1], 17), 7),
                ),
                i + 1,
            ),
            add_native(
                add_native(mul_native(native[0], 17), mul_native(native[1], 13)),
                i + 2,
            ),
        ];
    }
    b.publish(acc[0]);
    b.publish(acc[1]);
    let shape = b.finish().map_err(|e| format!("shape: {e:?}"))?;
    let witness = shape.run(&inputs, &[]);
    assert_eq!(
        &witness.public[witness.public.len() - 2..],
        &native.map(|x| F128::new(x, 0))
    );
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let profile = LigeritoProfile::Slim;
    let m = union.dense_m();
    let batch = embedded_initial_k_or_default(m, profile);
    let params = PcsParams {
        m,
        log_inv_rate: profile.log_inv_rate(),
        log_batch_size: batch,
        profile,
        num_lanes: union.commit_lanes(batch),
        merkle_hash: flock_prover::merkle::HashKind::Blake3,
    };
    let add_block = add.block(nu);
    let mul_block = mul.block(nu);
    let mut ordered = [
        (
            shape.registry_slot(add_slot),
            add.as_ref(),
            &add_block,
            add_slot,
        ),
        (
            shape.registry_slot(mul_slot),
            mul.as_ref(),
            &mul_block,
            mul_slot,
        ),
    ];
    ordered.sort_by_key(|x| x.0);
    let circuits: Vec<&dyn LincheckCircuit> = ordered
        .iter()
        .map(|x| x.2.csc_lincheck_circuit() as &dyn LincheckCircuit)
        .collect();
    let prepare_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let slots = ordered
        .iter()
        .zip(&circuits)
        .map(|(x, &lc)| {
            UnionSlotProverInput::new(packed_witness(x.1, witness.rows::<GoldGate>(x.3), nu), lc)
        })
        .collect();
    let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &witness.public,
        &params,
        slots,
        vec![],
        &mut FsChallenger::new(DOMAIN),
    );
    let prove_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    verify_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &witness.public,
        &circuits,
        &commitment,
        &proof,
        &params,
        &mut FsChallenger::new(DOMAIN),
    )
    .map_err(|e| format!("verify: {e:?}"))?;
    let verify_seconds = start.elapsed().as_secs_f64();
    for i in 0..witness.public.len() {
        let mut bad = witness.public.clone();
        bad[i].lo ^= 1;
        assert!(
            verify_ligerito_union_circuit(
                &union,
                &shape.circuit,
                &bad,
                &circuits,
                &commitment,
                &proof,
                &params,
                &mut FsChallenger::new(DOMAIN)
            )
            .is_err()
        );
    }
    fs::create_dir_all(out)?;
    let bytes = bincode::serialize(&(&commitment, &proof))?;
    fs::write(out.join("proof.bin"), &bytes)?;
    let report = json!({"scope":"16 quadratic Goldilocks Horner steps, not a complete FRI verifier",
        "flock_rev":"b684b1258e4b1f202bec24afd660ace851b09e5e","profile":"slim",
        "addition_bits_per_row":add.useful_bits(),"multiplication_bits_per_row":mul.useful_bits(),
        "dense_m":m,"prepare_seconds":prepare_seconds,"prove_seconds":prove_seconds,
        "verify_seconds":verify_seconds,"proof_and_commitment_bytes":bytes.len(),"verified":true,
        "altered_public_values_rejected":witness.public.len()});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
