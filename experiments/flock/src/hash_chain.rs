use flock_prover::{
    circuit::builder::{GateType, SlotWitness},
    field::F128,
    r1cs_hashes::blake3::{Compression, build_block_r1cs, io_schema},
    schedule::TableType,
};
#[cfg(feature = "counters")]
use {
    flock_prover::{
        challenger::{FsChallenger, fs_count},
        circuit::builder::ShapeBuilder,
        field::gf2_128::op_count,
        merkle::{HashKind, hash_count},
        pcs::{
            PcsParams,
            ligerito::{LigeritoProfile, embedded_initial_k_or_default},
        },
        prover::{UnionSlotProverInput, prove_fast_ligerito_union_circuit},
        r1cs_hashes::blake3::generate_witness_batch_major_partial,
        transcript_record::RecordingChallenger,
        union::UnionInstance,
        verifier::{verify_ligerito_union_circuit, verify_ligerito_union_circuit_deferred},
    },
    serde_json::json,
    std::{array::from_fn, fs, path::Path, time::Instant},
};
#[cfg(feature = "counters")]
const DOMAIN: &[u8] = b"multi-stark-flock-hash-chain-v1";
#[cfg(feature = "counters")]
const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn pack(w: &[u32]) -> F128 {
    F128::new(
        u64::from(w[0]) | (u64::from(w[1]) << 32),
        u64::from(w[2]) | (u64::from(w[3]) << 32),
    )
}
fn unpack(w: F128) -> [u32; 4] {
    [
        w.lo as u32,
        (w.lo >> 32) as u32,
        w.hi as u32,
        (w.hi >> 32) as u32,
    ]
}

// Flock's Boolean compression relation exposes every input and both output halves.
pub(crate) struct Blake3Gate {
    pub(crate) nu: usize,
}
impl GateType for Blake3Gate {
    type Row = Compression;
    type Hint = ();
    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&build_block_r1cs(self.nu)).with_io_schema(io_schema())
    }
    fn eval(&self, inputs: &[F128], _: &(), outputs: &mut Vec<F128>) -> Compression {
        let mut cv = [0; 8];
        cv[..4].copy_from_slice(&unpack(inputs[0]));
        cv[4..].copy_from_slice(&unpack(inputs[1]));
        let mut m = [0; 16];
        for i in 0..4 {
            m[4 * i..4 * i + 4].copy_from_slice(&unpack(inputs[2 + i]));
        }
        let (counter, len, flags) = (
            inputs[6].lo,
            inputs[6].hi as u32,
            (inputs[6].hi >> 32) as u32,
        );
        let out = flock_hash::blake3_compress(&cv, &m, counter, len, flags);
        outputs.extend(out.as_chunks::<4>().0.iter().map(|x| pack(x)));
        (cv, m, counter, len, flags)
    }
    fn witness(&self, _: &[Compression], _: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

#[cfg(feature = "counters")]
pub fn run(n: usize, profile_name: &str, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !(1..=65536).contains(&n) {
        return Err("length must be in 1..=65536".into());
    }
    let profile = match profile_name {
        "fast" => LigeritoProfile::Fast,
        "slim" => LigeritoProfile::Slim,
        _ => return Err("only strict Fast and Slim profiles are accepted".into()),
    };
    let _ = flock_prover::init_perf_thread_pool();
    let started = Instant::now();
    let nu = n.next_power_of_two().ilog2() as usize;
    let mut b = ShapeBuilder::new(nu);
    let slot = b.slot(Blake3Gate { nu });
    let cv = [
        b.fixed_public_input(pack(&IV[..4])),
        b.fixed_public_input(pack(&IV[4..])),
    ];
    let params = b.fixed_public_input(F128::new(0, 64 | (11u64 << 32)));
    let mut digest = [b.public_input(), b.public_input()];
    let mut input = vec![
        pack(&IV[..4]),
        pack(&IV[4..]),
        F128::new(0, 64 | (11u64 << 32)),
        F128::new(1, 2),
        F128::new(3, 4),
    ];
    for i in 0..n {
        let siblings = [b.input(), b.input()];
        input.extend([F128::new(i as u64 + 5, 6), F128::new(7, i as u64 + 8)]);
        let row = b.gate(
            slot,
            &[
                cv[0],
                cv[1],
                digest[0],
                digest[1],
                siblings[0],
                siblings[1],
                params,
            ],
        );
        digest = [row[0], row[1]];
    }
    b.publish(digest[0]);
    b.publish(digest[1]);
    let shape = b.finish().map_err(|e| format!("circuit: {e:?}"))?;
    let witness = shape.run(&input, &[]);
    // Native cross-check uses the standard hash API, independent of GateType::eval.
    let mut expected = [0u8; 32];
    expected[..16].copy_from_slice(&[1u64.to_le_bytes(), 2u64.to_le_bytes()].concat());
    expected[16..].copy_from_slice(&[3u64.to_le_bytes(), 4u64.to_le_bytes()].concat());
    for i in 0..n {
        let mut msg = [0u8; 64];
        msg[..32].copy_from_slice(&expected);
        for (j, v) in input[5 + 2 * i..7 + 2 * i].iter().enumerate() {
            msg[32 + 16 * j..40 + 16 * j].copy_from_slice(&v.lo.to_le_bytes());
            msg[40 + 16 * j..48 + 16 * j].copy_from_slice(&v.hi.to_le_bytes());
        }
        expected = blake3_hash(&msg);
    }
    let expected_words: [F128; 2] = from_fn(|i| {
        F128::new(
            u64::from_le_bytes(expected[16 * i..16 * i + 8].try_into().unwrap()),
            u64::from_le_bytes(expected[16 * i + 8..16 * i + 16].try_into().unwrap()),
        )
    });
    assert_eq!(&witness.public[witness.public.len() - 2..], &expected_words);
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let m = union.dense_m();
    let batch = embedded_initial_k_or_default(m, profile);
    let params = PcsParams {
        m,
        log_inv_rate: profile.log_inv_rate(),
        log_batch_size: batch,
        profile,
        num_lanes: union.commit_lanes(batch),
        merkle_hash: HashKind::Blake3,
    };
    let r1cs = build_block_r1cs(nu);
    let lc = r1cs.csc_lincheck_circuit();
    let prepared_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "shape ready: {n} hashes, m={m}, public_words={}",
        witness.public.len()
    );
    let started = Instant::now();
    let slot_input = UnionSlotProverInput::new(
        generate_witness_batch_major_partial(witness.rows::<Blake3Gate>(slot), nu),
        lc,
    );
    let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &witness.public,
        &params,
        vec![slot_input],
        vec![],
        &mut FsChallenger::new(DOMAIN),
    );
    let prove_seconds = started.elapsed().as_secs_f64();
    op_count::reset();
    hash_count::reset();
    fs_count::reset();
    let mut ch = RecordingChallenger::new(FsChallenger::new(DOMAIN));
    let started = Instant::now();
    verify_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &witness.public,
        &[lc],
        &commitment,
        &proof,
        &params,
        &mut ch,
    )
    .map_err(|e| format!("verify: {e:?}"))?;
    let verify_seconds = started.elapsed().as_secs_f64();
    let ops = op_count::snapshot();
    let hashes = hash_count::snapshot();
    let fs_ops = fs_count::snapshot();
    let (_, deferred_ops) = op_count::measure(|| {
        verify_ligerito_union_circuit_deferred(
            &union,
            &shape.circuit,
            &witness.public,
            &[lc],
            &commitment,
            &proof,
            &params,
            &mut FsChallenger::new(DOMAIN),
        )
        .expect("same proof through deferred verifier")
    });
    // Every public value, including the dynamic leaf and root, must bind.
    for i in 0..witness.public.len() {
        let mut wrong = witness.public.clone();
        wrong[i].lo ^= 1;
        assert!(
            verify_ligerito_union_circuit(
                &union,
                &shape.circuit,
                &wrong,
                &[lc],
                &commitment,
                &proof,
                &params,
                &mut FsChallenger::new(DOMAIN)
            )
            .is_err()
        );
    }
    // A different transcript domain must also reject the exact same proof.
    assert!(
        verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &witness.public,
            &[lc],
            &commitment,
            &proof,
            &params,
            &mut FsChallenger::new(b"wrong-domain")
        )
        .is_err()
    );
    let mut bad = proof.clone();
    bad.boolean.as_mut().unwrap().lincheck.z_partial[0].lo ^= 1;
    assert!(
        verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &witness.public,
            &[lc],
            &commitment,
            &bad,
            &params,
            &mut FsChallenger::new(DOMAIN)
        )
        .is_err()
    );
    fs::create_dir_all(out)?;
    let encoded = bincode::serialize(&(&commitment, &proof))?;
    fs::write(out.join("proof.bin"), &encoded)?;
    let report = json!({
        "flock_rev": "b684b1258e4b1f202bec24afd660ace851b09e5e",
        "scope": "Public-bound BLAKE3 chain prototype; not an Init FRI verifier or a Groth16 wrapper",
        "profile": profile_name, "hashes": n, "dense_m": m,
        "public_words": witness.public.len(), "proof_and_commitment_bytes": encoded.len(),
        "prepare_seconds": prepared_seconds, "prove_seconds": prove_seconds,
        "verify_seconds": verify_seconds, "verified": true,
        "altered_public_values_rejected": witness.public.len(), "wrong_domain_rejected": true,
        "corrupted_lincheck_rejected": true,
        "verifier_f128_multiplications": ops.muls_excluding_inv(), "verifier_f128_inversions": ops.invs,
        "deferred_f128_multiplications": deferred_ops.muls_excluding_inv(),
        "deferred_f128_inversions": deferred_ops.invs,
        "deferred_scope": "Comparison only: leaves matrix, wiring and layout assertions outstanding; not complete verification",
        "merkle_leaf_calls": hashes.0, "merkle_leaf_compressions": hashes.1,
        "merkle_pair_calls": hashes.2, "transcript_squeezes": fs_ops.0,
        "transcript_squeezed_bytes": fs_ops.1, "pow_checks": fs_ops.2,
        "transcript_shape_digest": ch.shape().digest_hex(),
        "warning": "Native operation counters are not a complete Groth16 constraint count. Full verifier used; no deferred checks omitted."
    });
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(feature = "counters")]
fn blake3_hash(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}
