//! End-to-end integration, zero-knowledge randomization, fail-closed security,
//! and proof-size tests for Protocol `v6-zk` over Goldilocks
//! $\mathbb{F}_{p^3} = \mathbb{F}_p[X]/(X^3 - X - 1)$.

#[cfg(not(feature = "cuda"))]
use std::fs;
#[cfg(not(feature = "cuda"))]
use std::path::PathBuf;
#[cfg(not(feature = "cuda"))]
use std::sync::atomic::{AtomicU64, Ordering};

use multi_stark::batch::{BatchMessage, BatchProof, Retention, ShardInput};
use multi_stark::config::ProofConfig;
use multi_stark::expr::{Expr, ExtExpr, RowOffset};
use multi_stark::lookup::Lookup;
use multi_stark::p3_adapter::fri_batching::FriBatching;
use multi_stark::p3_adapter::zk::{ZK_LEAF_SALT_BYTES, ZK_RANDOMIZER_WIDTH, min_zk_log_degree};
use multi_stark::prover::Proof;
use multi_stark::system::{CircuitInputs, System, SystemWitness};
use multi_stark::types::{
    CommitmentParameters, ExtVal3, FriParameters, GoldilocksBlake3V6Config,
    GoldilocksBlake3V6ZkConfig, GoldilocksBlake3V6ZkFp2Config, Val,
};
use multi_stark::verifier::VerificationError;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::MerkleCap;

#[cfg(not(feature = "cuda"))]
static DIR_SEQ: AtomicU64 = AtomicU64::new(0);

#[cfg(not(feature = "cuda"))]
fn fresh_spill_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "multi-stark-fp3-v6-zk-{tag}-{}-{nanos}-{seq}",
        std::process::id()
    ))
}

fn f(v: u32) -> Val {
    Val::from_u32(v)
}

fn zk_fri_params(num_queries: usize, pow_bits: usize) -> FriParameters {
    FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries,
        commit_proof_of_work_bits: pow_bits,
        query_proof_of_work_bits: pow_bits,
    }
}

/// Single-circuit shrinker-style AIR with a private witness `(a, b)` proving
/// `a + b == target` and `a^3 == a3` (degree-3 constraint + transition + claim
/// lookup on `target`, so `a, b` remain completely private to the witness).
fn single_circuit_private_sum_inputs(rows: usize) -> Vec<CircuitInputs<Val>> {
    assert!(rows.is_power_of_two() && rows >= 8);
    let a = Expr::main(0);
    let b = Expr::main(1);
    let target = Expr::main(2);
    let a3 = Expr::main(3);
    let mult = Expr::main(4);
    let target_next = Expr::main_next(2);

    let constraints = vec![
        a.clone() + b - target.clone(),
        a.clone() * a.clone() * a - a3,
        // Keep `target` constant across all rows via transition constraint.
        Expr::IsTransition * (target_next - target.clone()),
    ];
    let lookups = vec![Lookup::pull(mult, vec![Expr::constant(f(77)), target])];

    vec![CircuitInputs {
        main_width: 5,
        preprocessed: None,
        constraints,
        ext_constraints: vec![],
        lookups,
        lookup_group_size: 1,
    }]
}

fn build_single_circuit_witness(
    rows: usize,
    target: u32,
    witness_seed: u32,
) -> (Vec<RowMajorMatrix<Val>>, Vec<Vec<Val>>) {
    let mut values = Vec::with_capacity(rows * 5);
    for r in 0..rows {
        let a_u = (witness_seed.wrapping_add(u32::try_from(r).unwrap().wrapping_mul(17))) % target;
        let b_u = target - a_u;
        let a = f(a_u);
        let b = f(b_u);
        let t = f(target);
        let a3 = a * a * a;
        let mult = if r == 0 { Val::ONE } else { Val::ZERO };
        values.extend([a, b, t, a3, mult]);
    }
    let trace = RowMajorMatrix::new(values, 5);
    let claims = vec![vec![f(77), f(target)]];
    (vec![trace], claims)
}

/// Multi-circuit system (`C = 4`, mixed heights, preprocessed ROM, grouped
/// lookups, degree-3 `ExtExpr`, and sparse activation) sized to satisfy the
/// `v6-zk` minimum height floor (`>= 256` rows).
fn multi_circuit_zk_inputs(
    lookup_group_size: usize,
    work_rows: usize,
    aux_rows: usize,
    with_ext_constraint: bool,
) -> Vec<CircuitInputs<Val>> {
    assert!(work_rows.is_power_of_two() && work_rows >= 64);
    assert!(aux_rows.is_power_of_two() && aux_rows >= 64);

    // Circuit 0: 256-row preprocessed byte ROM (`prep[0] = 0..255`).
    let byte_prep = RowMajorMatrix::new((0..256u32).map(f).collect(), 1);
    let byte_table = CircuitInputs {
        main_width: 1,
        preprocessed: Some(byte_prep),
        constraints: vec![],
        ext_constraints: vec![],
        lookups: vec![Lookup::pull(
            Expr::main(0),
            vec![Expr::constant(f(10)), Expr::preprocessed(0)],
        )],
        lookup_group_size: 1,
    };

    // Circuit 1: `aux_rows`-row preprocessed aux ROM (can be deactivated).
    let aux_prep = RowMajorMatrix::new(
        (0..u32::try_from(aux_rows).unwrap())
            .map(|r| f(1000 + r))
            .collect(),
        1,
    );
    let aux_table = CircuitInputs {
        main_width: 1,
        preprocessed: Some(aux_prep),
        constraints: vec![Expr::main(0) * (Expr::preprocessed(0) - Expr::preprocessed(0))],
        ext_constraints: vec![],
        lookups: vec![Lookup::pull(
            Expr::main(0),
            vec![Expr::constant(f(20)), Expr::preprocessed(0)],
        )],
        lookup_group_size: 1,
    };

    // Circuit 2: `work_rows`-row arithmetic + lookup + transition + optional cubic ExtExpr.
    let work_prep = RowMajorMatrix::new(
        (0..u32::try_from(work_rows).unwrap())
            .map(|r| f(r + 1))
            .collect(),
        1,
    );
    let a = Expr::main(0);
    let b = Expr::main(1);
    let c = Expr::main(2);
    let d = Expr::main(3);
    let sum = Expr::main(4);
    let mult = Expr::main(5);
    let tag = Expr::preprocessed(0);
    let tag_next = Expr::preprocessed_next(0);
    let a_next = Expr::main_next(0);

    let ext_constraints = if with_ext_constraint {
        let lhs = ExtExpr::from(a.clone()) * ExtExpr::Coords(vec![b.clone(), c.clone(), d.clone()]);
        let rhs = ExtExpr::Coords(vec![
            a.clone() * b.clone(),
            a.clone() * c.clone(),
            a.clone() * d.clone(),
        ]);
        let s2 = ExtExpr::stage2(0, 3, RowOffset::Current);
        let s2_id = s2.clone() * ExtExpr::from(a.clone()) - ExtExpr::from(a.clone()) * s2;
        vec![lhs - rhs, s2_id]
    } else {
        vec![]
    };

    let work_constraints = vec![
        sum.clone() - (a.clone() + b.clone() + c.clone() + d.clone() + tag.clone()),
        Expr::IsTransition * (a_next.clone() - (a.clone() + Expr::constant(Val::ONE))),
    ];
    let work_lookups = vec![
        Lookup::pull(
            mult,
            vec![Expr::constant(f(30)), tag, tag_next, sum, a_next],
        ),
        Lookup::push(Expr::constant(Val::ONE), vec![Expr::constant(f(10)), a]),
        Lookup::push(Expr::constant(Val::ONE), vec![Expr::constant(f(10)), b]),
        Lookup::push(Expr::constant(Val::ONE), vec![Expr::constant(f(10)), c]),
        Lookup::push(Expr::constant(Val::ONE), vec![Expr::constant(f(10)), d]),
    ];
    let work_circuit = CircuitInputs {
        main_width: 6,
        preprocessed: Some(work_prep),
        constraints: work_constraints,
        ext_constraints,
        lookups: work_lookups,
        lookup_group_size,
    };

    // Circuit 3: zero-lookup quadratic circuit (`main[0]^2 == main[1]`).
    let sq = CircuitInputs {
        main_width: 2,
        preprocessed: None,
        constraints: vec![Expr::main(0) * Expr::main(0) - Expr::main(1)],
        ext_constraints: vec![],
        lookups: vec![],
        lookup_group_size: 1,
    };

    vec![byte_table, aux_table, work_circuit, sq]
}

fn build_multi_circuit_zk_witness(
    work_rows: usize,
    aux_rows: usize,
    sq_rows: usize,
    aux_active: bool,
    seed: u32,
) -> (Vec<RowMajorMatrix<Val>>, Vec<Vec<Val>>) {
    let mut byte_mults = vec![0u32; 256];
    let mut work_values = Vec::with_capacity(work_rows * 6);
    let mut claims = Vec::new();
    let max_base = 256usize.saturating_sub(work_rows);
    let base_a = if max_base == 0 {
        0
    } else {
        seed % u32::try_from(max_base).unwrap()
    };

    for r in 0..work_rows {
        let r32 = u32::try_from(r).unwrap();
        let a = (base_a + r32) & 0xFF;
        let b = (r32 * 7 + seed * 3 + 5) & 0xFF;
        let c = (r32 * 13 + seed * 11 + 17) & 0xFF;
        let d = (r32 * 29 + seed * 19 + 41) & 0xFF;
        let tag = r32 + 1;
        let tag_next = u32::try_from((r + 1) % work_rows).unwrap() + 1;
        let a_next = if r + 1 < work_rows {
            (base_a + r32 + 1) & 0xFF
        } else {
            base_a & 0xFF
        };
        let sum = a + b + c + d + tag;
        let mult = u32::from(r % 16 == 0);
        if mult == 1 {
            claims.push(vec![f(30), f(tag), f(tag_next), f(sum), f(a_next)]);
        }
        byte_mults[usize::try_from(a).unwrap()] += 1;
        byte_mults[usize::try_from(b).unwrap()] += 1;
        byte_mults[usize::try_from(c).unwrap()] += 1;
        byte_mults[usize::try_from(d).unwrap()] += 1;
        work_values.extend([f(a), f(b), f(c), f(d), f(sum), f(mult)]);
    }

    let byte_trace = RowMajorMatrix::new(byte_mults.into_iter().map(f).collect(), 1);
    let aux_trace = if aux_active {
        let mut aux_mults = vec![Val::ZERO; aux_rows];
        aux_mults[3] = Val::ONE;
        claims.push(vec![f(20), f(1003)]);
        RowMajorMatrix::new(aux_mults, 1)
    } else {
        RowMajorMatrix::new(Vec::new(), 1)
    };
    let work_trace = RowMajorMatrix::new(work_values, 6);
    let sq_trace = RowMajorMatrix::new(
        (0..u32::try_from(sq_rows).unwrap())
            .flat_map(|i| [f(i + 2 + seed), f((i + 2 + seed) * (i + 2 + seed))])
            .collect(),
        2,
    );

    (vec![byte_trace, aux_trace, work_trace, sq_trace], claims)
}

/// 1-circuit shrinker-style `v6-zk` roundtrips across `log_blowup = 2, 3, 4`,
/// `Tensor` & `Horner` batching, `cap_height = 0, 1`, PoW, serialization, and
/// both `ExtVal3` and `ExtVal2`.
#[test]
fn v6_zk_single_circuit_roundtrip_matrix() {
    let rows = 128usize; // >= next_pow2(2*24 + 6) = 64
    let fri = zk_fri_params(24, 0);

    for log_blowup in [2usize, 3, 4] {
        for batching in [FriBatching::Tensor, FriBatching::Horner] {
            for cap_height in [0usize, 1] {
                #[cfg(feature = "cuda")]
                if cap_height != 0 {
                    continue;
                }
                let config = GoldilocksBlake3V6ZkConfig::new_with_batching(
                    CommitmentParameters {
                        log_blowup,
                        cap_height,
                    },
                    fri,
                    batching,
                );
                let (system, key) = System::new(config, single_circuit_private_sum_inputs(rows));
                let (traces, claims) = build_single_circuit_witness(rows, 500, 11);
                let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();
                let witness = SystemWitness::from_stage_1(traces, &system);
                let proof = system.prove_multiple_claims(&key, &claim_refs, witness);

                // 1-circuit ZK has no boundary mask matrix `U` in stage_2,
                // and exactly 1 randomizer matrix `R` of width `W_R = 4` in quotient.
                assert_eq!(proof.stage_2_opened_values.len(), 1);
                assert_eq!(proof.quotient_opened_values.len(), 2);
                assert_eq!(
                    proof.quotient_opened_values[1][0].len(),
                    ZK_RANDOMIZER_WIDTH
                );
                assert_eq!(proof.intermediate_accumulators, vec![ExtVal3::ZERO]);

                system
                    .verify_multiple_claims(&claim_refs, &proof)
                    .unwrap_or_else(|e| {
                        panic!("1-circuit ZK verify failed (lb={log_blowup}, {batching:?}, cap={cap_height}): {e:?}")
                    });

                let bytes = proof.to_bytes().expect("serialize ZK proof");
                let decoded = Proof::<GoldilocksBlake3V6ZkConfig>::from_bytes(&bytes)
                    .expect("deserialize ZK proof");
                system
                    .verify_multiple_claims(&claim_refs, &decoded)
                    .expect("deserialized 1-circuit ZK proof must verify");
            }
        }
    }

    // Also test positive PoW bits + quadratic `ExtVal2` config.
    let fp2_cfg = GoldilocksBlake3V6ZkFp2Config::new(
        CommitmentParameters {
            log_blowup: 2,
            cap_height: 0,
        },
        zk_fri_params(20, 4),
    );
    let (sys_fp2, key_fp2) = System::new(fp2_cfg, single_circuit_private_sum_inputs(64));
    let (traces, claims) = build_single_circuit_witness(64, 300, 5);
    let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();
    let witness = SystemWitness::from_stage_1(traces, &sys_fp2);
    let proof = sys_fp2.prove_multiple_claims(&key_fp2, &claim_refs, witness);
    sys_fp2
        .verify_multiple_claims(&claim_refs, &proof)
        .expect("Fp2 v6-zk proof with PoW must verify");
}

/// Multi-circuit (`C_active > 1`) `v6-zk` roundtrips with preprocessed traces,
/// mixed heights (`N_min < N_max`), grouped lookups, sparse activation,
/// `intermediate_accumulators` blinding, streaming prover, disk spill, and
/// multi-shard batch proofs across `log_blowup = 2, 3, 4`.
#[test]
fn v6_zk_multi_circuit_roundtrip_and_batch_and_spill() {
    let fri = zk_fri_params(20, 0); // min height = next_pow2(46) = 64

    for log_blowup in [2usize, 3, 4] {
        for (group_size, aux_active) in [(1usize, true), (2usize, false)] {
            let work_rows = 128usize;
            let aux_rows = 64usize;
            let sq_rows = 64usize;
            let config = GoldilocksBlake3V6ZkConfig::new(
                CommitmentParameters {
                    log_blowup,
                    cap_height: 0,
                },
                fri,
            );
            let inputs = multi_circuit_zk_inputs(group_size, work_rows, aux_rows, true);
            let (system, key) = System::new(config, inputs);
            let (traces, claims) =
                build_multi_circuit_zk_witness(work_rows, aux_rows, sq_rows, aux_active, 7);
            let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();

            let witness = SystemWitness::from_stage_1(traces.clone(), &system);
            let proof = system.prove_multiple_claims(&key, &claim_refs, witness);

            let num_active = if aux_active { 4 } else { 3 };
            // Multi-circuit ZK commits boundary mask `U` as matrix `#num_active`
            // in `stage_2_trace` and randomizer `R` as matrix `#num_active` in `quotient`.
            assert_eq!(proof.stage_2_opened_values.len(), num_active + 1);
            assert_eq!(
                proof.stage_2_opened_values[num_active][0].len(),
                (num_active - 1) * 3
            );
            assert_eq!(proof.quotient_opened_values.len(), num_active + 1);
            assert_eq!(
                proof.quotient_opened_values[num_active][0].len(),
                ZK_RANDOMIZER_WIDTH
            );
            assert_eq!(proof.intermediate_accumulators.len(), num_active);
            assert_eq!(proof.intermediate_accumulators.last(), Some(&ExtVal3::ZERO));

            system
                .verify_multiple_claims(&claim_refs, &proof)
                .unwrap_or_else(|e| {
                    panic!(
                        "multi-circuit ZK failed (lb={log_blowup}, gs={group_size}, aux={aux_active}): {e:?}"
                    )
                });
        }
    }

    // Verify streaming + disk spill + `new_without_preprocessed` on CPU.
    #[cfg(not(feature = "cuda"))]
    {
        let dir = fresh_spill_dir("multi-circuit");
        let work_rows = 128usize;
        let aux_rows = 64usize;
        let sq_rows = 64usize;
        let inputs = multi_circuit_zk_inputs(2, work_rows, aux_rows, true);

        let cfg_mem = GoldilocksBlake3V6ZkConfig::new(
            CommitmentParameters {
                log_blowup: 2,
                cap_height: 1,
            },
            fri,
        )
        .without_disk_spill();
        let cfg_spill = GoldilocksBlake3V6ZkConfig::new(
            CommitmentParameters {
                log_blowup: 2,
                cap_height: 1,
            },
            fri,
        )
        .with_disk_spill(&dir);

        let (sys_mem, _) = System::new(cfg_mem, inputs.clone());
        let (sys_spill, key_spill) = System::new_without_preprocessed(cfg_spill, inputs);

        // Deterministic public preprocessed salt guarantees identical setup commitments!
        assert_eq!(
            sys_mem.preprocessed_commit.as_ref().unwrap().roots(),
            sys_spill.preprocessed_commit.as_ref().unwrap().roots(),
            "independent v6-zk setups must produce identical preprocessed_commit"
        );
        assert_eq!(sys_mem.vk_digest, sys_spill.vk_digest);

        let (traces, claims) =
            build_multi_circuit_zk_witness(work_rows, aux_rows, sq_rows, true, 13);
        let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();
        let streaming_proof =
            sys_spill.prove_multiple_claims_streaming(&key_spill, &claim_refs, traces);
        // Cross-verify spilled streaming proof against independent in-memory verifier!
        sys_mem
            .verify_multiple_claims(&claim_refs, &streaming_proof)
            .expect("in-memory verifier must accept disk-spilled streaming v6-zk proof");
        let _ = fs::remove_dir_all(&dir);
    }

    // Multi-shard batch proof in `v6-zk` (`Retention::Retain`).
    {
        let work_rows = 64usize;
        let aux_rows = 64usize;
        let sq_rows = 64usize;
        let config = GoldilocksBlake3V6ZkConfig::new(
            CommitmentParameters {
                log_blowup: 2,
                cap_height: 0,
            },
            fri,
        );
        let inputs = multi_circuit_zk_inputs(1, work_rows, aux_rows, false);
        let (system, key) = System::new(config, inputs);
        let (t0, mut c0) = build_multi_circuit_zk_witness(work_rows, aux_rows, sq_rows, false, 3);
        let (t1, c1) = build_multi_circuit_zk_witness(work_rows, aux_rows, sq_rows, true, 9);
        // Move one claim from shard 0 to a batch-level message so shards have non-zero residuals.
        let moved = c0.pop().unwrap();
        let messages = vec![BatchMessage::push(moved)];
        let shards = vec![
            ShardInput {
                claims: c0,
                witness: SystemWitness::from_stage_1(t0, &system),
            },
            ShardInput {
                claims: c1,
                witness: SystemWitness::from_stage_1(t1, &system),
            },
        ];
        let batch_proof = system.prove_batch(&key, shards, messages);
        assert_ne!(
            batch_proof.proofs[0].intermediate_accumulators.last(),
            Some(&ExtVal3::ZERO),
            "shard 0 should have a non-zero residual before batch balancing"
        );
        system
            .verify_batch(&batch_proof)
            .expect("multi-shard v6-zk batch proof must verify");
        let bytes = batch_proof.to_bytes().expect("serialize batch proof");
        let decoded = BatchProof::<GoldilocksBlake3V6ZkConfig>::from_bytes(&bytes)
            .expect("deserialize batch proof");
        system
            .verify_batch(&decoded)
            .expect("deserialized v6-zk batch proof must verify");
        // Ensure `Retention::Retain` path works via `prove_batch_with` too.
        let (t_a, c_a) = build_multi_circuit_zk_witness(work_rows, aux_rows, sq_rows, false, 1);
        let batch2 = system.prove_batch_with(&key, &[c_a], vec![], Retention::Retain, |_| {
            SystemWitness::from_stage_1(t_a.clone(), &system)
        });
        system.verify_batch(&batch2).unwrap();
    }
}

/// ZK randomization & indistinguishability tests:
/// 1. Same statement + same witness proved twice -> distinct Merkle roots,
///    distinct opened trace values, distinct leaf salts, distinct blinded
///    `intermediate_accumulators[0..C-1]`, BOTH verify.
/// 2. Same public statement + two DIFFERENT private witnesses -> identically
///    structured transcripts, BOTH verify.
#[test]
fn v6_zk_randomization_and_witness_indistinguishability() {
    let fri = zk_fri_params(24, 0);
    let config = GoldilocksBlake3V6ZkConfig::new(
        CommitmentParameters {
            log_blowup: 2,
            cap_height: 0,
        },
        fri,
    );

    // Part A: Multi-circuit same statement + same witness proved twice.
    {
        let work_rows = 128usize;
        let aux_rows = 64usize;
        let sq_rows = 64usize;
        let inputs = multi_circuit_zk_inputs(1, work_rows, aux_rows, true);
        let (system, key) = System::new(config, inputs);
        let (traces, claims) =
            build_multi_circuit_zk_witness(work_rows, aux_rows, sq_rows, true, 19);
        let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();

        let p1 = system.prove_multiple_claims(
            &key,
            &claim_refs,
            SystemWitness::from_stage_1(traces.clone(), &system),
        );
        let p2 = system.prove_multiple_claims(
            &key,
            &claim_refs,
            SystemWitness::from_stage_1(traces, &system),
        );

        system.verify_multiple_claims(&claim_refs, &p1).unwrap();
        system.verify_multiple_claims(&claim_refs, &p2).unwrap();

        // Distinct commitments for every prover round.
        assert_ne!(
            p1.commitments.stage_1_trace.roots(),
            p2.commitments.stage_1_trace.roots(),
            "stage_1 Merkle root must be freshly randomized"
        );
        assert_ne!(
            p1.commitments.stage_2_trace.roots(),
            p2.commitments.stage_2_trace.roots(),
            "stage_2 Merkle root must be freshly randomized"
        );
        assert_ne!(
            p1.commitments.quotient_chunks.roots(),
            p2.commitments.quotient_chunks.roots(),
            "quotient Merkle root must be freshly randomized"
        );

        // Distinct OOD openings (witness, stage_2, boundary mask U, quotient, randomizer R).
        assert_ne!(p1.stage_1_opened_values, p2.stage_1_opened_values);
        assert_ne!(p1.stage_2_opened_values, p2.stage_2_opened_values);
        assert_ne!(p1.quotient_opened_values, p2.quotient_opened_values);

        // Distinct leaf salts and opened leaf rows across all prover rounds.
        for round in 0..3 {
            assert_ne!(
                p1.opening_proof.input_openings[round].opening_proof.salts,
                p2.opening_proof.input_openings[round].opening_proof.salts,
                "leaf salts in round {round} must differ across proofs"
            );
            assert_ne!(
                p1.opening_proof.input_openings[round].opened_values,
                p2.opening_proof.input_openings[round].opened_values,
                "opened blinded leaves in round {round} must differ across proofs"
            );
            // Every leaf salt is exactly 16 bytes (128 bits).
            for group_salts in &p1.opening_proof.input_openings[round].opening_proof.salts {
                assert!(!group_salts.is_empty());
                for salt in group_salts {
                    assert_eq!(salt.len(), ZK_LEAF_SALT_BYTES);
                }
            }
        }

        // Distinct blinded interior `intermediate_accumulators[0..C-1]`, while final == 0.
        let num_active = p1.intermediate_accumulators.len();
        assert_eq!(num_active, 4);
        for i in 0..(num_active - 1) {
            assert_ne!(
                p1.intermediate_accumulators[i], p2.intermediate_accumulators[i],
                "blinded intermediate_accumulator[{i}] must be freshly masked"
            );
        }
        assert_eq!(p1.intermediate_accumulators[num_active - 1], ExtVal3::ZERO);
        assert_eq!(p2.intermediate_accumulators[num_active - 1], ExtVal3::ZERO);
    }

    // Part B: Same public statement + TWO DIFFERENT witnesses -> identically
    // structured transcripts, both verify.
    {
        let rows = 128usize;
        let cfg = GoldilocksBlake3V6ZkConfig::new(
            CommitmentParameters {
                log_blowup: 2,
                cap_height: 0,
            },
            fri,
        );
        let (system, key) = System::new(cfg, single_circuit_private_sum_inputs(rows));

        let (t1, claims1) = build_single_circuit_witness(rows, 10_000, 1);
        let (t2, claims2) = build_single_circuit_witness(rows, 10_000, 4_321);
        assert_eq!(
            claims1, claims2,
            "both witnesses satisfy the SAME public statement"
        );
        assert_ne!(
            t1[0].values, t2[0].values,
            "the two private witnesses must differ"
        );

        let claim_refs: Vec<&[Val]> = claims1.iter().map(Vec::as_slice).collect();
        let p1 = system.prove_multiple_claims(
            &key,
            &claim_refs,
            SystemWitness::from_stage_1(t1, &system),
        );
        let p2 = system.prove_multiple_claims(
            &key,
            &claim_refs,
            SystemWitness::from_stage_1(t2, &system),
        );

        system.verify_multiple_claims(&claim_refs, &p1).unwrap();
        system.verify_multiple_claims(&claim_refs, &p2).unwrap();

        // Identical structural shapes.
        assert_eq!(p1.active, p2.active);
        assert_eq!(p1.log_degrees, p2.log_degrees);
        assert_eq!(p1.intermediate_accumulators, p2.intermediate_accumulators);
        assert_eq!(
            p1.stage_1_opened_values.len(),
            p2.stage_1_opened_values.len()
        );
        assert_eq!(
            p1.stage_2_opened_values.len(),
            p2.stage_2_opened_values.len()
        );
        assert_eq!(
            p1.quotient_opened_values.len(),
            p2.quotient_opened_values.len()
        );
        assert_eq!(
            p1.opening_proof.commit_phase_commits.len(),
            p2.opening_proof.commit_phase_commits.len()
        );
        assert_eq!(
            p1.opening_proof.input_openings.len(),
            p2.opening_proof.input_openings.len()
        );
    }
}

/// Comprehensive fail-closed soundness and security negative controls:
/// - Tampered opened leaf salt (input commitment + FRI commit phase)
/// - Tampered blinded leaf value
/// - Tampered `R(X)` randomizer OOD evaluation
/// - Tampered `U(X)` boundary mask OOD evaluation
/// - Tampered `intermediate_accumulators` (interior & final)
/// - Tampered `MerkleCap` root count (`C2`) on every commitment + FRI commit phase
/// - Cross-mode replay (`v6` <-> `v6-zk`, `Tensor` <-> `Horner`, single <-> batch)
/// - Under-sized trace rejected fail-closed at verifier
/// - Truncated `R(X)` width `< 4` rejected fail-closed at verifier
#[test]
fn v6_zk_fail_closed_security_negative_controls() {
    let fri = zk_fri_params(20, 0); // min_log_degree = 6 (64 rows)
    let work_rows = 128usize;
    let aux_rows = 64usize;
    let sq_rows = 64usize;

    let make_system = |batching: FriBatching| {
        let cfg = GoldilocksBlake3V6ZkConfig::new_with_batching(
            CommitmentParameters {
                log_blowup: 2,
                cap_height: 0,
            },
            fri,
            batching,
        );
        System::new(cfg, multi_circuit_zk_inputs(1, work_rows, aux_rows, true))
    };

    let (system, key) = make_system(FriBatching::Tensor);
    let (traces, claims) = build_multi_circuit_zk_witness(work_rows, aux_rows, sq_rows, true, 5);
    let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();
    let valid_proof = system.prove_multiple_claims(
        &key,
        &claim_refs,
        SystemWitness::from_stage_1(traces.clone(), &system),
    );
    system
        .verify_multiple_claims(&claim_refs, &valid_proof)
        .unwrap();

    // 1. Tampered opened leaf salt in input commitments (stage_1, stage_2, quotient, preprocessed).
    for round in 0..valid_proof.opening_proof.input_openings.len() {
        let mut tampered = valid_proof.clone();
        tampered.opening_proof.input_openings[round]
            .opening_proof
            .salts[0][0][0] ^= 0x01;
        assert!(
            matches!(
                system.verify_multiple_claims(&claim_refs, &tampered),
                Err(VerificationError::InvalidOpeningArgument(_))
            ),
            "tampered input leaf salt in round {round} must be rejected"
        );
    }

    // 2. Tampered leaf salt inside FRI commit-phase Merkle opening.
    {
        let mut tampered = valid_proof.clone();
        tampered.opening_proof.commit_phase_openings[0]
            .opening_proof
            .salts[0][0][0] ^= 0x80;
        assert!(
            matches!(
                system.verify_multiple_claims(&claim_refs, &tampered),
                Err(VerificationError::InvalidOpeningArgument(_))
            ),
            "tampered FRI commit-phase leaf salt must be rejected"
        );
    }

    // 3. Tampered blinded leaf value.
    {
        let mut tampered = valid_proof.clone();
        tampered.opening_proof.input_openings[0].opened_values[0][0][0] += Val::ONE;
        assert!(
            matches!(
                system.verify_multiple_claims(&claim_refs, &tampered),
                Err(VerificationError::InvalidOpeningArgument(_))
            ),
            "tampered opened blinded leaf value must be rejected"
        );
    }

    // 4. Tampered `R(X)` OOD evaluation.
    {
        let mut tampered = valid_proof.clone();
        let r_idx = tampered.quotient_opened_values.len() - 1;
        tampered.quotient_opened_values[r_idx][0][0] += ExtVal3::ONE;
        assert!(
            system
                .verify_multiple_claims(&claim_refs, &tampered)
                .is_err(),
            "tampered R(X) OOD evaluation must be rejected"
        );
    }

    // 5. Truncated `R(X)` opening width (`W_R < 4`).
    {
        let mut tampered = valid_proof.clone();
        let r_idx = tampered.quotient_opened_values.len() - 1;
        tampered.quotient_opened_values[r_idx][0].pop();
        assert!(
            matches!(
                system.verify_multiple_claims(&claim_refs, &tampered),
                Err(VerificationError::InvalidProofShape)
            ),
            "truncated R(X) width < 4 must be rejected as InvalidProofShape"
        );
    }

    // 6. Tampered `U(X)` boundary mask OOD evaluation.
    {
        let mut tampered = valid_proof.clone();
        let u_idx = tampered.stage_2_opened_values.len() - 1;
        tampered.stage_2_opened_values[u_idx][0][0] += ExtVal3::ONE;
        assert!(
            system
                .verify_multiple_claims(&claim_refs, &tampered)
                .is_err(),
            "tampered boundary mask U(X) OOD evaluation must be rejected"
        );
    }

    // 7. Tampered `intermediate_accumulators` (interior blinded slot AND final slot).
    {
        let mut tampered_interior = valid_proof.clone();
        tampered_interior.intermediate_accumulators[0] += ExtVal3::ONE;
        assert!(
            system
                .verify_multiple_claims(&claim_refs, &tampered_interior)
                .is_err(),
            "tampered interior blinded accumulator must be rejected"
        );

        let mut tampered_final = valid_proof.clone();
        let last = tampered_final.intermediate_accumulators.len() - 1;
        tampered_final.intermediate_accumulators[last] += ExtVal3::ONE;
        assert!(
            matches!(
                system.verify_multiple_claims(&claim_refs, &tampered_final),
                Err(VerificationError::UnbalancedChannel)
            ),
            "tampered final accumulator must be rejected as UnbalancedChannel"
        );
    }

    // 8. C2: Inflated `MerkleCap` root count on stage_1, stage_2, quotient,
    //    FRI commit-phase commitment, and preprocessed commitment.
    {
        let inflate = |cap: &MerkleCap<Val, [u8; 32]>| {
            let mut roots = cap.roots().to_vec();
            roots.push([0xAAu8; 32]);
            MerkleCap::new(roots)
        };

        let mut t1 = valid_proof.clone();
        t1.commitments.stage_1_trace = inflate(&t1.commitments.stage_1_trace);
        assert!(matches!(
            system.verify_multiple_claims(&claim_refs, &t1),
            Err(VerificationError::InvalidProofShape)
        ));

        let mut t2 = valid_proof.clone();
        t2.commitments.stage_2_trace = inflate(&t2.commitments.stage_2_trace);
        assert!(matches!(
            system.verify_multiple_claims(&claim_refs, &t2),
            Err(VerificationError::InvalidProofShape)
        ));

        let mut t3 = valid_proof.clone();
        t3.commitments.quotient_chunks = inflate(&t3.commitments.quotient_chunks);
        assert!(matches!(
            system.verify_multiple_claims(&claim_refs, &t3),
            Err(VerificationError::InvalidProofShape)
        ));

        let mut t_fri = valid_proof.clone();
        t_fri.opening_proof.commit_phase_commits[0] =
            inflate(&t_fri.opening_proof.commit_phase_commits[0]);
        assert!(matches!(
            system.verify_multiple_claims(&claim_refs, &t_fri),
            Err(VerificationError::InvalidOpeningArgument(_))
        ));

        // C2 on preprocessed commitment cap root count.
        let (mut bad_prep_sys, _) = make_system(FriBatching::Tensor);
        bad_prep_sys.preprocessed_commit =
            Some(inflate(bad_prep_sys.preprocessed_commit.as_ref().unwrap()));
        assert!(matches!(
            bad_prep_sys.verify_multiple_claims(&claim_refs, &valid_proof),
            Err(VerificationError::InvalidSystem)
        ));
    }

    // 9. Trap 1: Adversarial proof claiming `log_degrees` below `min_log_degree`.
    {
        let mut undersized = valid_proof.clone();
        // Circuit 3 has no preprocessed height so `log_degrees[3]` is untrusted.
        let min_ld = system.config.min_log_degree();
        undersized.log_degrees[3] = u8::try_from(min_ld - 1).unwrap();
        assert!(
            matches!(
                system.verify_multiple_claims(&claim_refs, &undersized),
                Err(VerificationError::InvalidProofShape)
            ),
            "proof with trace height below ZK floor must be rejected fail-closed"
        );
    }

    // 10. Cross-mode replay:
    //     (a) `Tensor` proof rejected by `Horner` `v6-zk` verifier.
    //     (b) `v6-zk` proof bytes rejected by non-ZK `v6` deserialization/verifier
    //         AND `v6` proof bytes rejected by `v6-zk` deserialization/verifier.
    {
        let (horner_sys, _) = make_system(FriBatching::Horner);
        assert!(
            horner_sys
                .verify_multiple_claims(&claim_refs, &valid_proof)
                .is_err(),
            "Tensor v6-zk proof must not verify under Horner v6-zk verifier"
        );

        let v6_cfg = GoldilocksBlake3V6Config::new(
            CommitmentParameters {
                log_blowup: 2,
                cap_height: 0,
            },
            fri,
        );
        let (v6_sys, v6_key) = System::new(
            v6_cfg,
            multi_circuit_zk_inputs(1, work_rows, aux_rows, true),
        );
        // Distinct VK digests and challenger seeds between `v6` and `v6-zk`.
        assert_ne!(v6_sys.vk_digest, system.vk_digest);
        assert_ne!(
            v6_sys.config.challenger_seed(),
            system.config.challenger_seed()
        );
        let v6_proof = v6_sys.prove_multiple_claims(
            &v6_key,
            &claim_refs,
            SystemWitness::from_stage_1(traces, &v6_sys),
        );
        v6_sys
            .verify_multiple_claims(&claim_refs, &v6_proof)
            .unwrap();

        let v6_bytes = v6_proof.to_bytes().unwrap();
        let zk_bytes = valid_proof.to_bytes().unwrap();
        // Cross-deserialization or verification MUST fail in BOTH directions.
        if let Ok(decoded_as_zk) = Proof::<GoldilocksBlake3V6ZkConfig>::from_bytes(&v6_bytes) {
            assert!(
                system
                    .verify_multiple_claims(&claim_refs, &decoded_as_zk)
                    .is_err(),
                "non-ZK v6 proof must never verify under v6-zk verifier"
            );
        }
        if let Ok(decoded_as_v6) = Proof::<GoldilocksBlake3V6Config>::from_bytes(&zk_bytes) {
            assert!(
                v6_sys
                    .verify_multiple_claims(&claim_refs, &decoded_as_v6)
                    .is_err(),
                "v6-zk proof must never verify under non-ZK v6 verifier"
            );
        }
    }
}

/// Trap 1 prover enforcement: proving an under-sized trace (`N < min_zk_height`)
/// MUST panic fail-closed.
#[test]
#[should_panic(expected = "below v6-zk minimum height")]
fn v6_zk_rejects_undersized_trace_at_prove() {
    let fri = zk_fri_params(20, 0); // min_log_degree = 6 -> 64 rows
    assert_eq!(min_zk_log_degree(20, 3), 6);
    let config = GoldilocksBlake3V6ZkConfig::new(
        CommitmentParameters {
            log_blowup: 2,
            cap_height: 0,
        },
        fri,
    );
    let (system, key) = System::new(config, single_circuit_private_sum_inputs(32));
    let (traces, claims) = build_single_circuit_witness(32, 100, 1);
    let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();
    let witness = SystemWitness::from_stage_1(traces, &system);
    let _ = system.prove_multiple_claims(&key, &claim_refs, witness);
}

/// Trap 1 setup enforcement: preprocessed trace height below `min_zk_height`
/// MUST be rejected fail-closed at `System::new`.
#[test]
#[should_panic(expected = "below v6-zk minimum height")]
fn v6_zk_rejects_undersized_preprocessed_trace_at_setup() {
    let config = GoldilocksBlake3V6ZkConfig::new(
        CommitmentParameters {
            log_blowup: 2,
            cap_height: 0,
        },
        zk_fri_params(20, 0), // min height = 64
    );
    let tiny_prep = CircuitInputs {
        main_width: 1,
        preprocessed: Some(RowMajorMatrix::new(vec![Val::ZERO; 16], 1)),
        constraints: vec![Expr::main(0) - Expr::preprocessed(0)],
        ext_constraints: vec![],
        lookups: vec![],
        lookup_group_size: 1,
    };
    let _ = System::new(config, [tiny_prep]);
}

/// Trap 2 config enforcement: `W_R < 4` MUST be rejected fail-closed.
#[test]
#[should_panic(expected = "randomizer width W_R >= 4")]
fn v6_zk_rejects_randomizer_width_below_4() {
    let _ = GoldilocksBlake3V6ZkConfig::new(
        CommitmentParameters {
            log_blowup: 2,
            cap_height: 0,
        },
        zk_fri_params(20, 0),
    )
    .with_randomizer_width(3);
}

/// Trap 4 enforcement (a): `log_blowup = 0` in ZK config MUST be rejected.
#[test]
#[should_panic(expected = "log_blowup >= 1")]
fn v6_zk_rejects_zero_log_blowup() {
    let _ = GoldilocksBlake3V6ZkConfig::new(
        CommitmentParameters {
            log_blowup: 0,
            cap_height: 0,
        },
        zk_fri_params(20, 0),
    );
}

/// Trap 4 enforcement (b): degree-3 constraint at `log_blowup = 1` (`q_zk = 4 > 2`)
/// MUST be rejected fail-closed at `System::new`.
#[test]
#[should_panic(expected = "needs quotient degree 4, but the PCS only supports 2")]
fn v6_zk_rejects_degree_3_at_log_blowup_1() {
    let config = GoldilocksBlake3V6ZkConfig::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        zk_fri_params(20, 0),
    );
    let _ = System::new(config, single_circuit_private_sum_inputs(64));
}

/// Trap 4 enforcement (c): degree-5 constraint at `log_blowup = 2` (`q_zk = 8 > 4`)
/// MUST be rejected fail-closed at `System::new`, and succeeds at `log_blowup = 3`.
#[test]
#[should_panic(expected = "needs quotient degree 8, but the PCS only supports 4")]
fn v6_zk_rejects_degree_5_at_log_blowup_2() {
    let x = Expr::main(0);
    let y = Expr::main(1);
    let x5 = x.clone() * x.clone() * x.clone() * x.clone() * x;
    let deg5_circuit = CircuitInputs {
        main_width: 2,
        preprocessed: None,
        constraints: vec![x5 - y],
        ext_constraints: vec![],
        lookups: vec![],
        lookup_group_size: 1,
    };
    let config = GoldilocksBlake3V6ZkConfig::new(
        CommitmentParameters {
            log_blowup: 2,
            cap_height: 0,
        },
        zk_fri_params(20, 0),
    );
    let _ = System::new(config, [deg5_circuit]);
}

/// Companion positive check for Trap 4: degree-5 constraint succeeds cleanly at
/// `log_blowup = 3` (`q_zk = 8 <= 8`).
#[test]
fn v6_zk_accepts_degree_5_at_log_blowup_3() {
    let x = Expr::main(0);
    let y = Expr::main(1);
    let x5 = x.clone() * x.clone() * x.clone() * x.clone() * x;
    let deg5_circuit = CircuitInputs {
        main_width: 2,
        preprocessed: None,
        constraints: vec![x5 - y],
        ext_constraints: vec![],
        lookups: vec![],
        lookup_group_size: 1,
    };
    let config = GoldilocksBlake3V6ZkConfig::new(
        CommitmentParameters {
            log_blowup: 3,
            cap_height: 0,
        },
        zk_fri_params(20, 0),
    );
    let (system, key) = System::new(config, [deg5_circuit]);
    let rows = 64usize;
    let mut vals = Vec::with_capacity(rows * 2);
    for r in 0..rows {
        let v = f(u32::try_from(r % 11).unwrap());
        let v5 = v * v * v * v * v;
        vals.extend([v, v5]);
    }
    let trace = RowMajorMatrix::new(vals, 2);
    let witness = SystemWitness::from_stage_1(vec![trace], &system);
    let no_claims: &[&[Val]] = &[];
    let proof = system.prove_multiple_claims(&key, no_claims, witness);
    system.verify_multiple_claims(no_claims, &proof).unwrap();
}

/// Measures and reports serialized proof sizes for non-ZK `v6` vs `v6-zk` on
/// both 1-circuit shrinker-style and 4-circuit multi-circuit systems at
/// 128-bit security parameters (`log_blowup = 2`, `num_queries = 64`, `Q` queries).
#[test]
fn v6_vs_v6_zk_proof_size_comparison() {
    let fri = zk_fri_params(64, 0); // min_zk_height = 256 (2*64 + 6 = 134 -> 256)
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: 0,
    };

    // 1-circuit (N = 256, degree 3).
    {
        let rows = 256usize;
        let (traces, claims) = build_single_circuit_witness(rows, 999, 7);
        let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();

        let (sys_v6, key_v6) = System::new(
            GoldilocksBlake3V6Config::new(comm, fri),
            single_circuit_private_sum_inputs(rows),
        );
        let p_v6 = sys_v6.prove_multiple_claims(
            &key_v6,
            &claim_refs,
            SystemWitness::from_stage_1(traces.clone(), &sys_v6),
        );
        sys_v6.verify_multiple_claims(&claim_refs, &p_v6).unwrap();
        let v6_bytes = p_v6.to_bytes().unwrap().len();

        let (sys_zk, key_zk) = System::new(
            GoldilocksBlake3V6ZkConfig::new(comm, fri),
            single_circuit_private_sum_inputs(rows),
        );
        let p_zk = sys_zk.prove_multiple_claims(
            &key_zk,
            &claim_refs,
            SystemWitness::from_stage_1(traces, &sys_zk),
        );
        sys_zk.verify_multiple_claims(&claim_refs, &p_zk).unwrap();
        let zk_bytes = p_zk.to_bytes().unwrap().len();

        eprintln!(
            "[proof-size] 1-circuit (N=256, lb=2, Q=64): v6={v6_bytes} B, v6-zk={zk_bytes} B (+{} B)",
            zk_bytes.saturating_sub(v6_bytes)
        );
        assert!(zk_bytes > v6_bytes);
    }

    // 4-circuit multi-circuit (N = 256, lookups + preprocessed + ExtExpr).
    {
        let rows = 256usize;
        let (traces, claims) = build_multi_circuit_zk_witness(rows, rows, rows, true, 11);
        let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();

        let (sys_v6, key_v6) = System::new(
            GoldilocksBlake3V6Config::new(comm, fri),
            multi_circuit_zk_inputs(1, rows, rows, true),
        );
        let p_v6 = sys_v6.prove_multiple_claims(
            &key_v6,
            &claim_refs,
            SystemWitness::from_stage_1(traces.clone(), &sys_v6),
        );
        sys_v6.verify_multiple_claims(&claim_refs, &p_v6).unwrap();
        let v6_bytes = p_v6.to_bytes().unwrap().len();

        let (sys_zk, key_zk) = System::new(
            GoldilocksBlake3V6ZkConfig::new(comm, fri),
            multi_circuit_zk_inputs(1, rows, rows, true),
        );
        let p_zk = sys_zk.prove_multiple_claims(
            &key_zk,
            &claim_refs,
            SystemWitness::from_stage_1(traces, &sys_zk),
        );
        sys_zk.verify_multiple_claims(&claim_refs, &p_zk).unwrap();
        let zk_bytes = p_zk.to_bytes().unwrap().len();

        eprintln!(
            "[proof-size] 4-circuit (N=256, lb=2, Q=64): v6={v6_bytes} B, v6-zk={zk_bytes} B (+{} B)",
            zk_bytes.saturating_sub(v6_bytes)
        );
        assert!(zk_bytes > v6_bytes);
    }
}

/// Multi-arity FRI (`max_log_arity = 2, 3, 4`) + `log_final_poly_len = 8` for
/// Protocol `v6-zk` (`Ext3` hiding) across both single-circuit uniform height
/// and multi-circuit mismatched heights strictly above 1024 LDE rows (runs on
/// both CPU and CUDA builds).
#[test]
fn v6_zk_multi_arity_and_final_poly_roundtrip() {
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: 0,
    };

    // Single-circuit 2048-row trace -> ZK LDE height = 2 * 2048 * 4 = 16384 = 2^14 > 1024.
    let single_rows = 2048usize;
    let (single_traces, single_claims) = build_single_circuit_witness(single_rows, 999, 13);
    let single_claim_refs: Vec<&[Val]> = single_claims.iter().map(Vec::as_slice).collect();

    // Multi-circuit mismatched heights: byte ROM / aux / work = 256 (ZK LDE 2048 = 2^11),
    // sq = 4096 (ZK LDE 32768 = 2^15) — all > 2^(8+2) = 1024, exercising mixed log_arity
    // including full max_log_arity = 2, 3, 4!
    let (multi_traces, multi_claims) = build_multi_circuit_zk_witness(256, 256, 4096, true, 17);
    let multi_claim_refs: Vec<&[Val]> = multi_claims.iter().map(Vec::as_slice).collect();

    for max_log_arity in [2usize, 3, 4] {
        let fri = FriParameters {
            log_final_poly_len: 8,
            max_log_arity,
            num_queries: 24,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        };

        let (sys, key) = System::new(
            GoldilocksBlake3V6ZkConfig::new(comm, fri),
            single_circuit_private_sum_inputs(single_rows),
        );
        let proof = sys.prove_multiple_claims(
            &key,
            &single_claim_refs,
            SystemWitness::from_stage_1(single_traces.clone(), &sys),
        );
        sys.verify_multiple_claims(&single_claim_refs, &proof)
            .unwrap_or_else(|e| {
                panic!(
                    "v6-zk single-circuit multi-arity verify failed (arity={max_log_arity}): {e:?}"
                )
            });
        let single_arities: Vec<usize> = proof
            .opening_proof
            .commit_phase_openings
            .iter()
            .map(|s| usize::from(s.log_arity))
            .collect();
        assert!(
            single_arities.contains(&max_log_arity),
            "v6-zk single-circuit expected log_arity={max_log_arity} in {single_arities:?}"
        );
        let bytes = proof.to_bytes().unwrap();
        let decoded = Proof::<GoldilocksBlake3V6ZkConfig>::from_bytes(&bytes).unwrap();
        sys.verify_multiple_claims(&single_claim_refs, &decoded)
            .unwrap();

        let (msys, mkey) = System::new(
            GoldilocksBlake3V6ZkConfig::new(comm, fri),
            multi_circuit_zk_inputs(2, 256, 256, true),
        );
        let mproof = msys.prove_multiple_claims(
            &mkey,
            &multi_claim_refs,
            SystemWitness::from_stage_1(multi_traces.clone(), &msys),
        );
        msys.verify_multiple_claims(&multi_claim_refs, &mproof)
            .unwrap_or_else(|e| {
                panic!(
                    "v6-zk multi-circuit multi-arity verify failed (arity={max_log_arity}): {e:?}"
                )
            });
        let marities: Vec<usize> = mproof
            .opening_proof
            .commit_phase_openings
            .iter()
            .map(|s| usize::from(s.log_arity))
            .collect();
        assert!(
            marities.contains(&max_log_arity) && marities.contains(&1),
            "v6-zk multi-circuit expected mixed arities containing {max_log_arity} and 1, got {marities:?}"
        );
    }
}
