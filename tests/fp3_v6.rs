//! End-to-end integration tests and CPU benchmarks for Protocol `v6` and
//! Goldilocks cubic trinomial extension $\mathbb{F}_{p^3} = \mathbb{F}_p[X]/(X^3 - X - 1)$.

#[cfg(not(feature = "cuda"))]
use std::fs;
#[cfg(not(feature = "cuda"))]
use std::path::PathBuf;
#[cfg(not(feature = "cuda"))]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use multi_stark::batch::{BatchMessage, BatchProof, Retention, ShardInput};
use multi_stark::config::ProofConfig;
use multi_stark::expr::{Expr, ExtExpr, RowOffset};
use multi_stark::lookup::Lookup;
use multi_stark::p3_adapter::fri_batching::FriBatching;
use multi_stark::plonkish::CircuitBuilder;
use multi_stark::plonkish::gadgets::{ByteGadgets, blake3};
use multi_stark::prover::Proof;
use multi_stark::system::{CircuitInputs, System, SystemWitness};
use multi_stark::types::{
    CommitmentParameters, ExtVal3, FriParameters, GoldilocksBlake3Config, GoldilocksBlake3V6Config,
    GoldilocksBlake3V6Fp2Config, Val,
};
use multi_stark::witness::{PreparedWitness, TraceSource};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

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
        "multi-stark-fp3-v6-{tag}-{}-{nanos}-{seq}",
        std::process::id()
    ))
}

fn f(v: u32) -> Val {
    Val::from_u32(v)
}

/// Builds a 4-circuit system exercising:
/// - Circuit 0: Preprocessed 256-row byte ROM (`prep[0] = 0..255`, `main[0] = mult`)
/// - Circuit 1: Preprocessed 16-row aux ROM (can be left inactive for sparse tests)
/// - Circuit 2: Multi-lookup arithmetic + transition + degree-3 `ExtExpr` circuit
/// - Circuit 3: Zero-lookup quadratic circuit (`main[0]^2 == main[1]`)
fn multi_circuit_inputs(
    lookup_group_size: usize,
    work_rows: usize,
    with_ext_constraint: bool,
) -> Vec<CircuitInputs<Val>> {
    assert!(work_rows.is_power_of_two() && work_rows >= 4);

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

    let aux_prep = RowMajorMatrix::new((100..116u32).map(f).collect(), 1);
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

    // Optional degree-3 ExtExpr exercising CubicTrinomial mul/add/sub in ConstraintGraph:
    // On every row, `(s2_0 - s2_0) * (s2_0 + const_ext) == 0` AND a real algebraic identity:
    // `a * (b + X*c + X^2*d) - (a*b + X*(a*c) + X^2*(a*d)) == 0`.
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

fn build_multi_circuit_witness(
    work_rows: usize,
    aux_active: bool,
    seed: u32,
) -> (Vec<RowMajorMatrix<Val>>, Vec<Vec<Val>>) {
    let mut byte_mults = vec![0u32; 256];
    let mut work_values = Vec::with_capacity(work_rows * 6);
    let mut claims = Vec::new();
    let base_a = seed % 120;

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
        let mult = u32::from(r % 2 == 0);
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
        let mut aux_mults = vec![Val::ZERO; 16];
        aux_mults[3] = Val::ONE;
        claims.push(vec![f(20), f(103)]);
        RowMajorMatrix::new(aux_mults, 1)
    } else {
        RowMajorMatrix::new(Vec::new(), 1)
    };
    let work_trace = RowMajorMatrix::new(work_values, 6);
    let sq_trace = RowMajorMatrix::new(
        (0..8u32)
            .flat_map(|i| [f(i + 2 + seed), f((i + 2 + seed) * (i + 2 + seed))])
            .collect(),
        2,
    );

    (vec![byte_trace, aux_trace, work_trace, sq_trace], claims)
}

/// Builds a U32Add + ByteTable circuit pair (`num_add_rows` power of two).
fn u32_add_system_inputs(lookup_group_size: usize) -> Vec<CircuitInputs<Val>> {
    let byte_prep = RowMajorMatrix::new((0..256u32).map(f).collect(), 1);
    let byte_table = CircuitInputs {
        main_width: 1,
        preprocessed: Some(byte_prep),
        constraints: vec![],
        ext_constraints: vec![],
        lookups: vec![Lookup::pull(
            Expr::main(0),
            vec![Expr::constant(f(0)), Expr::preprocessed(0)],
        )],
        lookup_group_size: 1,
    };

    let carry = Expr::main(12);
    let x = Expr::main(0)
        + Expr::main(1) * Expr::constant(f(256))
        + Expr::main(2) * Expr::constant(f(256 * 256))
        + Expr::main(3) * Expr::constant(f(256 * 256 * 256));
    let y = Expr::main(4)
        + Expr::main(5) * Expr::constant(f(256))
        + Expr::main(6) * Expr::constant(f(256 * 256))
        + Expr::main(7) * Expr::constant(f(256 * 256 * 256));
    let z = Expr::main(8)
        + Expr::main(9) * Expr::constant(f(256))
        + Expr::main(10) * Expr::constant(f(256 * 256))
        + Expr::main(11) * Expr::constant(f(256 * 256 * 256));
    let two_32 = Expr::constant(Val::from_u64(1u64 << 32));
    let constraints = vec![
        carry.clone() * (carry.clone() - Expr::constant(Val::ONE)),
        x.clone() + y.clone() - (z.clone() + carry * two_32),
    ];
    let mut lookups = vec![Lookup::pull(
        Expr::main(13),
        vec![Expr::constant(f(1)), x, y, z],
    )];
    lookups.extend((0..12).map(|i| {
        Lookup::push(
            Expr::constant(Val::ONE),
            vec![Expr::constant(f(0)), Expr::main(i)],
        )
    }));
    let u32_add = CircuitInputs {
        main_width: 14,
        preprocessed: None,
        constraints,
        ext_constraints: vec![],
        lookups,
        lookup_group_size,
    };

    vec![byte_table, u32_add]
}

fn u32_add_witness(num_rows: usize) -> (Vec<RowMajorMatrix<Val>>, Vec<Vec<Val>>) {
    assert!(num_rows.is_power_of_two());
    let mut byte_mults = vec![0u32; 256];
    let mut add_values = Vec::with_capacity(num_rows * 14);
    let mut claims = Vec::with_capacity(16);

    for r in 0..num_rows {
        let r32 = u32::try_from(r).unwrap();
        let x = r32.wrapping_mul(2_654_435_761).wrapping_add(17);
        let y = r32.wrapping_mul(1_597_334_677).wrapping_add(1_000_003);
        let (z, overflow) = x.overflowing_add(y);
        let xb = x.to_le_bytes();
        let yb = y.to_le_bytes();
        let zb = z.to_le_bytes();
        for &b in xb.iter().chain(&yb).chain(&zb) {
            byte_mults[usize::from(b)] += 1;
        }
        let mult = u32::from(r < 16);
        if mult == 1 {
            claims.push(vec![f(1), f(x), f(y), f(z)]);
        }
        for &b in xb.iter().chain(&yb).chain(&zb) {
            add_values.push(Val::from_u8(b));
        }
        add_values.push(Val::from_u8(u8::from(overflow)));
        add_values.push(f(mult));
    }
    let byte_trace = RowMajorMatrix::new(byte_mults.into_iter().map(f).collect(), 1);
    let add_trace = RowMajorMatrix::new(add_values, 14);
    (vec![byte_trace, add_trace], claims)
}

#[test]
fn fp3_v6_single_proof_matrix() {
    let cases = [
        // (batching, group_size, log_blowup, cap_height, max_log_arity, pow_bits, aux_active)
        (FriBatching::Tensor, 1, 2, 0, 1, 0, true),
        (FriBatching::Tensor, 2, 2, 1, 2, 4, false),
        (FriBatching::Tensor, 3, 2, 2, 2, 0, true),
        (FriBatching::Tensor, 4, 3, 1, 2, 4, false),
        (FriBatching::Horner, 2, 2, 0, 1, 4, false),
        (FriBatching::Horner, 4, 2, 2, 2, 0, true),
    ];

    for (batching, group_size, log_blowup, cap_height, max_log_arity, pow_bits, aux_active) in cases
    {
        let comm = CommitmentParameters {
            log_blowup,
            cap_height: if cfg!(feature = "cuda") {
                0
            } else {
                cap_height
            },
        };
        let fri = FriParameters {
            log_final_poly_len: 1,
            max_log_arity: if cfg!(feature = "cuda") {
                1
            } else {
                max_log_arity
            },
            num_queries: 12,
            commit_proof_of_work_bits: pow_bits,
            query_proof_of_work_bits: pow_bits,
        };
        let config = GoldilocksBlake3V6Config::new_with_batching(comm, fri, batching);
        let inputs = multi_circuit_inputs(group_size, 32, true);
        let (system, key) = System::new(config, inputs);
        let (traces, claims) = build_multi_circuit_witness(32, aux_active, 7);
        let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();

        let witness = SystemWitness::from_stage_1(traces.clone(), &system);
        let proof = system.prove_multiple_claims(&key, &claim_refs, witness);
        system
            .verify_multiple_claims(&claim_refs, &proof)
            .unwrap_or_else(|e| {
                panic!(
                    "failed for batching={batching:?}, k={group_size}, cap={cap_height}, arity={max_log_arity}, pow={pow_bits}, aux={aux_active}: {e:?}"
                )
            });

        // Serialization roundtrip + deterministic repeatability (including positive PoW).
        let bytes = proof.to_bytes().unwrap();
        let decoded = Proof::<GoldilocksBlake3V6Config>::from_bytes(&bytes).unwrap();
        system
            .verify_multiple_claims(&claim_refs, &decoded)
            .unwrap();

        let witness2 = SystemWitness::from_stage_1(traces, &system);
        let proof2 = system.prove_multiple_claims(&key, &claim_refs, witness2);
        assert_eq!(
            proof2.to_bytes().unwrap(),
            bytes,
            "v6 F_p^3 proof must be deterministic (batching={batching:?}, pow={pow_bits})"
        );
    }
}

#[test]
fn fp3_v6_plonkish_proof_with_blake3_gadget() {
    let mut builder = CircuitBuilder::<Val>::new();
    let bytes = ByteGadgets::new(&mut builder);
    let x = builder.public_input("x");
    let inv_x = builder.inverse(x);
    let prod = builder.mul(x, inv_x);
    builder.expose_public(prod);

    let msg_wires: Vec<_> = (0..16)
        .map(|i| bytes.input(&mut builder, &format!("m[{i}]")))
        .collect();
    let digest = blake3(&mut builder, &bytes, &msg_wires);
    builder.expose_public(digest[0].value());

    let lowered = builder
        .finish()
        .lower_to_multi_stark(Val::from_u32(7))
        .unwrap();

    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: if cfg!(feature = "cuda") { 0 } else { 1 },
    };
    let fri = FriParameters {
        log_final_poly_len: 1,
        max_log_arity: if cfg!(feature = "cuda") { 1 } else { 2 },
        num_queries: 12,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };
    let config = GoldilocksBlake3V6Config::new(comm, fri);
    let (system, key) = System::new(config, lowered.circuit_inputs());

    let mut w = lowered.witness();
    w.set(x, Val::from_u32(42)).unwrap();
    let msg: Vec<u8> = (0..16u8)
        .map(|i| i.wrapping_mul(19).wrapping_add(3))
        .collect();
    for (&wire, &byte) in msg_wires.iter().zip(&msg) {
        w.set(wire.value(), Val::from_u8(byte)).unwrap();
    }
    let assignment = w.generate().unwrap();
    let expected_first_byte = blake3::hash(&msg).as_bytes()[0];
    let public_vals = [
        Val::from_u32(42),
        Val::ONE,
        Val::from_u8(expected_first_byte),
    ];
    let claims = lowered.claims(&public_vals).unwrap();
    let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();
    let traces = lowered.traces(&assignment).unwrap();

    let witness = SystemWitness::from_stage_1(traces.clone(), &system);
    let proof = system.prove_multiple_claims(&key, &claim_refs, witness);
    system.verify_multiple_claims(&claim_refs, &proof).unwrap();

    // Lazy lookup + new_without_preprocessed must match byte-for-byte.
    let lazy_witness = SystemWitness::from_stage_1_lazy(traces, &system);
    let lazy_proof = system.prove_multiple_claims(&key, &claim_refs, lazy_witness);
    assert_eq!(lazy_proof.to_bytes().unwrap(), proof.to_bytes().unwrap());
}

#[test]
fn fp3_v6_batch_proof_retain_regenerate_and_messages() {
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: if cfg!(feature = "cuda") { 0 } else { 1 },
    };
    let fri = FriParameters {
        log_final_poly_len: 1,
        max_log_arity: if cfg!(feature = "cuda") { 1 } else { 2 },
        num_queries: 10,
        commit_proof_of_work_bits: 2,
        query_proof_of_work_bits: 2,
    };

    for batching in [FriBatching::Tensor, FriBatching::Horner] {
        let config = GoldilocksBlake3V6Config::new_with_batching(comm, fri, batching);
        // 3 circuits:
        // 0: Preprocessed Byte ROM (active only in shard 0)
        // 1: Arithmetic producer that pushes to byte ROM and sends a cross-shard message
        // 2: Consumer that receives the cross-shard message and exposes a public claim
        let byte_prep = RowMajorMatrix::new((0..256u32).map(f).collect(), 1);
        let inputs = vec![
            CircuitInputs {
                main_width: 1,
                preprocessed: Some(byte_prep),
                constraints: vec![],
                ext_constraints: vec![],
                lookups: vec![Lookup::pull(
                    Expr::main(0),
                    vec![Expr::constant(f(1)), Expr::preprocessed(0)],
                )],
                lookup_group_size: 1,
            },
            CircuitInputs {
                main_width: 2,
                preprocessed: None,
                constraints: vec![Expr::main(0) + Expr::constant(f(5)) - Expr::main(1)],
                ext_constraints: vec![],
                lookups: vec![
                    Lookup::push(
                        Expr::constant(Val::ONE),
                        vec![Expr::constant(f(1)), Expr::main(0)],
                    ),
                    Lookup::push(
                        Expr::constant(Val::ONE),
                        vec![Expr::constant(f(2)), Expr::main(1)],
                    ),
                ],
                lookup_group_size: 2,
            },
            CircuitInputs {
                main_width: 2,
                preprocessed: None,
                constraints: vec![Expr::main(0) - Expr::main(1)],
                ext_constraints: vec![],
                lookups: vec![
                    Lookup::pull(
                        Expr::constant(Val::ONE),
                        vec![Expr::constant(f(2)), Expr::main(0)],
                    ),
                    Lookup::pull(
                        Expr::constant(Val::ONE),
                        vec![Expr::constant(f(3)), Expr::main(1)],
                    ),
                ],
                lookup_group_size: 2,
            },
        ];

        let (system, key) = System::new(config, inputs);

        // Shard 0: byte ROM + producer rows (a = 10..14 -> b = 15..19).
        // Shard 1: consumer rows pulling channel 2 and exposing channel 3 claims.
        // Plus batch messages pushing an extra byte lookup (pulled by shard 0's byte table)
        // and a balanced push/pull pair on channel 99.
        let mut byte_mults = vec![Val::ZERO; 256];
        let mut prod_vals = Vec::new();
        let mut consumer_vals = Vec::new();
        let mut shard1_claims = Vec::new();
        for a in 10..14u32 {
            byte_mults[usize::try_from(a).unwrap()] = Val::ONE;
            let b = a + 5;
            prod_vals.extend([f(a), f(b)]);
            consumer_vals.extend([f(b), f(b)]);
            shard1_claims.push(vec![f(3), f(b)]);
        }
        byte_mults[77] = Val::ONE;
        let batch_messages = vec![
            BatchMessage::push(vec![f(1), f(77)]),
            BatchMessage::push(vec![f(99), f(1234)]),
            BatchMessage::pull(vec![f(99), f(1234)]),
        ];

        let shard0_traces = vec![
            RowMajorMatrix::new(byte_mults, 1),
            RowMajorMatrix::new(prod_vals, 2),
            RowMajorMatrix::new(Vec::new(), 2),
        ];
        let shard1_traces = vec![
            RowMajorMatrix::new(Vec::new(), 1),
            RowMajorMatrix::new(Vec::new(), 2),
            RowMajorMatrix::new(consumer_vals, 2),
        ];
        let shard_traces = [shard0_traces, shard1_traces];
        let shard_claims = [Vec::new(), shard1_claims];

        let retained_batch = system.prove_batch(
            &key,
            (0..2)
                .map(|s| ShardInput {
                    witness: SystemWitness::from_stage_1(shard_traces[s].clone(), &system),
                    claims: shard_claims[s].clone(),
                })
                .collect(),
            batch_messages.clone(),
        );
        let regenerated_batch = system.prove_batch_with(
            &key,
            &shard_claims,
            batch_messages,
            Retention::Regenerate,
            |s| {
                let sources = shard_traces[s]
                    .iter()
                    .cloned()
                    .map(TraceSource::Host)
                    .collect();
                PreparedWitness::from_stage_1_lazy(sources, &system)
            },
        );

        // Retain and Regenerate must produce byte-identical batch proofs.
        let batch_bytes = retained_batch.to_bytes().unwrap();
        assert_eq!(
            regenerated_batch.to_bytes().unwrap(),
            batch_bytes,
            "Retain and Regenerate batch proofs must be byte-identical ({batching:?})"
        );

        let decoded = BatchProof::<GoldilocksBlake3V6Config>::from_bytes(&batch_bytes).unwrap();
        system.verify_batch(&decoded).unwrap();
        let zero_ef = <ExtVal3 as PrimeCharacteristicRing>::ZERO;
        for shard_idx in 0..2 {
            let residual = system
                .verify_batch_shard(&decoded.preamble, shard_idx, &decoded.proofs[shard_idx])
                .unwrap();
            assert_ne!(
                residual, zero_ef,
                "cross-shard lookup traffic must yield non-zero per-shard residual"
            );
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn fp3_v6_streaming_and_disk_spill_byte_identical() {
    use multi_stark::p3_adapter::spill::active_spill_files;

    let spill_dir = fresh_spill_dir("single-and-batch");
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: if cfg!(feature = "cuda") { 0 } else { 1 },
    };
    let fri = FriParameters {
        log_final_poly_len: 1,
        max_log_arity: if cfg!(feature = "cuda") { 1 } else { 2 },
        num_queries: 10,
        commit_proof_of_work_bits: 2,
        query_proof_of_work_bits: 2,
    };
    let inputs = multi_circuit_inputs(3, 64, true);

    let (mem_sys, mem_key) = System::new(
        GoldilocksBlake3V6Config::new(comm, fri).without_disk_spill(),
        inputs.clone(),
    );
    let (spill_sys, spill_key) = System::new(
        GoldilocksBlake3V6Config::new(comm, fri).with_disk_spill(&spill_dir),
        inputs.clone(),
    );
    let (noprep_sys, noprep_key) = System::new_without_preprocessed(
        GoldilocksBlake3V6Config::new(comm, fri).with_disk_spill(&spill_dir),
        inputs,
    );
    assert_eq!(mem_sys.vk_digest, noprep_sys.vk_digest);
    assert_eq!(mem_sys.vk_digest, spill_sys.vk_digest);

    let (traces, claims) = build_multi_circuit_witness(64, false, 11);
    let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();

    let mem_proof = mem_sys.prove_multiple_claims(
        &mem_key,
        &claim_refs,
        SystemWitness::from_stage_1(traces.clone(), &mem_sys),
    );
    let mem_bytes = mem_proof.to_bytes().unwrap();

    let spill_proof = spill_sys.prove_multiple_claims(
        &spill_key,
        &claim_refs,
        SystemWitness::from_stage_1_lazy(traces.clone(), &spill_sys),
    );
    assert_eq!(spill_proof.to_bytes().unwrap(), mem_bytes);

    let stage_1 = noprep_sys.prove_stage_1_streaming(traces.clone());
    let stream_proof = noprep_sys.prove_committed(&noprep_key, &claim_refs, stage_1);
    assert_eq!(stream_proof.to_bytes().unwrap(), mem_bytes);
    mem_sys
        .verify_multiple_claims(&claim_refs, &stream_proof)
        .unwrap();

    // Multi-shard batch streaming + disk spill byte identity.
    let (traces2, claims2) = build_multi_circuit_witness(64, true, 23);
    let mem_batch = mem_sys.prove_batch(
        &mem_key,
        vec![
            ShardInput {
                witness: SystemWitness::from_stage_1(traces.clone(), &mem_sys),
                claims: claims.clone(),
            },
            ShardInput {
                witness: SystemWitness::from_stage_1(traces2.clone(), &mem_sys),
                claims: claims2.clone(),
            },
        ],
        vec![],
    );
    let shard_traces = [traces, traces2];
    let shard_claims = [claims, claims2];
    let spill_batch = noprep_sys.prove_batch_with(
        &noprep_key,
        &shard_claims,
        vec![],
        Retention::Regenerate,
        |s| {
            let sources = shard_traces[s]
                .iter()
                .cloned()
                .map(TraceSource::Host)
                .collect();
            PreparedWitness::from_stage_1_lazy(sources, &noprep_sys)
        },
    );
    assert_eq!(
        spill_batch.to_bytes().unwrap(),
        mem_batch.to_bytes().unwrap(),
        "spilled lazy F_p^3 batch proof must be byte-identical to in-memory"
    );
    mem_sys.verify_batch(&spill_batch).unwrap();

    drop(spill_key);
    drop(noprep_key);
    drop(spill_sys);
    drop(noprep_sys);
    assert_eq!(active_spill_files(), 0);
    if spill_dir.exists() {
        let leftover: Vec<_> = fs::read_dir(&spill_dir)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert!(leftover.is_empty(), "leftover spill files: {leftover:?}");
        let _ = fs::remove_dir_all(&spill_dir);
    }
}

#[test]
fn v6_domain_separation_and_batching_cross_rejection() {
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: 0,
    };
    let fri = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 10,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };
    // Edge case: single width-1 circuit, no lookups -> minimal opening structure.
    let single_col = vec![CircuitInputs {
        main_width: 1,
        preprocessed: None,
        constraints: vec![Expr::main(0) * (Expr::main(0) - Expr::constant(Val::ONE))],
        ext_constraints: vec![],
        lookups: vec![],
        lookup_group_size: 1,
    }];
    let trace = vec![RowMajorMatrix::new_col(vec![
        Val::ZERO,
        Val::ONE,
        Val::ONE,
        Val::ZERO,
    ])];

    // 1. v6 F_p^3 Tensor vs Horner cross-rejection (and width=1 edge case).
    let (sys_t, key_t) = System::new(
        GoldilocksBlake3V6Config::new_with_batching(comm, fri, FriBatching::Tensor),
        single_col.clone(),
    );
    let (sys_h, key_h) = System::new(
        GoldilocksBlake3V6Config::new_with_batching(comm, fri, FriBatching::Horner),
        single_col.clone(),
    );
    let proof_t = sys_t.prove_multiple_claims(
        &key_t,
        &[],
        SystemWitness::from_stage_1(trace.clone(), &sys_t),
    );
    let proof_h = sys_h.prove_multiple_claims(
        &key_h,
        &[],
        SystemWitness::from_stage_1(trace.clone(), &sys_h),
    );
    sys_t.verify_multiple_claims(&[], &proof_t).unwrap();
    sys_h.verify_multiple_claims(&[], &proof_h).unwrap();
    assert!(
        sys_h.verify_multiple_claims(&[], &proof_t).is_err(),
        "v6 Tensor proof must not verify under v6 Horner"
    );
    assert!(
        sys_t.verify_multiple_claims(&[], &proof_h).is_err(),
        "v6 Horner proof must not verify under v6 Tensor"
    );

    // 2. v5 F_p^2 vs v6 F_p^2 (Horner and Tensor) cross-rejection via serialized wire bytes.
    let (sys_v5, key_v5) = System::new(GoldilocksBlake3Config::new(comm, fri), single_col.clone());
    let (sys_v6_h, key_v6_h) = System::new(
        GoldilocksBlake3V6Fp2Config::new_with_batching(comm, fri, FriBatching::Horner),
        single_col.clone(),
    );
    let (sys_v6_t, key_v6_t) = System::new(
        GoldilocksBlake3V6Fp2Config::new_with_batching(comm, fri, FriBatching::Tensor),
        single_col,
    );
    let proof_v5 = sys_v5.prove_multiple_claims(
        &key_v5,
        &[],
        SystemWitness::from_stage_1(trace.clone(), &sys_v5),
    );
    let proof_v6_h = sys_v6_h.prove_multiple_claims(
        &key_v6_h,
        &[],
        SystemWitness::from_stage_1(trace.clone(), &sys_v6_h),
    );
    let proof_v6_t = sys_v6_t.prove_multiple_claims(
        &key_v6_t,
        &[],
        SystemWitness::from_stage_1(trace.clone(), &sys_v6_t),
    );
    sys_v5.verify_multiple_claims(&[], &proof_v5).unwrap();
    sys_v6_h.verify_multiple_claims(&[], &proof_v6_h).unwrap();
    sys_v6_t.verify_multiple_claims(&[], &proof_v6_t).unwrap();

    // Wire bytes of v5 Proof deserialize into v6 F_p^2 Proof (same struct layout),
    // but verification MUST fail due to v6 domain-separated BLAKE3 + transcript + VK digest.
    let v5_as_v6 =
        Proof::<GoldilocksBlake3V6Fp2Config>::from_bytes(&proof_v5.to_bytes().unwrap()).unwrap();
    assert!(sys_v6_h.verify_multiple_claims(&[], &v5_as_v6).is_err());
    assert!(sys_v6_t.verify_multiple_claims(&[], &v5_as_v6).is_err());

    let v6_as_v5 =
        Proof::<GoldilocksBlake3Config>::from_bytes(&proof_v6_h.to_bytes().unwrap()).unwrap();
    assert!(sys_v5.verify_multiple_claims(&[], &v6_as_v5).is_err());

    // 3. Single-proof vs 1-shard batch-proof domain separation (`TranscriptSingle` != `TranscriptBatch`).
    let batch1 = sys_t.prove_batch(
        &key_t,
        vec![ShardInput {
            witness: SystemWitness::from_stage_1(trace, &sys_t),
            claims: vec![],
        }],
        vec![],
    );
    sys_t.verify_batch(&batch1).unwrap();
    assert!(
        sys_t
            .verify_multiple_claims(&[], &batch1.proofs[0])
            .is_err(),
        "1-shard v6 batch proof must not verify as a single proof"
    );
    let mut forged_batch = batch1;
    forged_batch.proofs[0] = proof_t;
    assert!(
        sys_t.verify_batch(&forged_batch).is_err(),
        "single v6 proof must not verify inside a 1-shard batch"
    );
}

#[test]
fn v6_vk_digest_binds_full_constraint_graph_med5() {
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: 0,
    };
    let fri = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 10,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };
    let prep = RowMajorMatrix::new((0..8u32).map(f).collect(), 1);
    // Two circuits with IDENTICAL shape metadata (widths, preprocessed trace,
    // constraint_count, max_constraint_degree, num_lookups, lookup_group_size),
    // differing only in the constraint formula:
    //   Circuit A: main[0] * main[1] - main[2] == 0
    //   Circuit B: main[0] * main[1] + main[2] == 0
    let inputs_a = vec![CircuitInputs {
        main_width: 3,
        preprocessed: Some(prep.clone()),
        constraints: vec![Expr::main(0) * Expr::main(1) - Expr::main(2)],
        ext_constraints: vec![],
        lookups: vec![],
        lookup_group_size: 1,
    }];
    let inputs_b = vec![CircuitInputs {
        main_width: 3,
        preprocessed: Some(prep),
        constraints: vec![Expr::main(0) * Expr::main(1) + Expr::main(2)],
        ext_constraints: vec![],
        lookups: vec![],
        lookup_group_size: 1,
    }];

    let (sys_a, key_a) = System::new(GoldilocksBlake3V6Config::new(comm, fri), inputs_a.clone());
    let (sys_b, _key_b) = System::new(GoldilocksBlake3V6Config::new(comm, fri), inputs_b);
    let (sys_a_noprep, _) = System::new_without_preprocessed(
        GoldilocksBlake3V6Config::new(comm, fri),
        inputs_a.clone(),
    );
    let (sys_a_fp2, _) = System::new(GoldilocksBlake3V6Fp2Config::new(comm, fri), inputs_a);

    assert_ne!(
        sys_a.vk_digest, sys_b.vk_digest,
        "MED-5: circuits with identical shapes but distinct constraint graphs must have distinct vk_digest"
    );
    assert_eq!(
        sys_a.vk_digest, sys_a_noprep.vk_digest,
        "new and new_without_preprocessed must compute identical vk_digest"
    );
    assert_ne!(
        sys_a.vk_digest, sys_a_fp2.vk_digest,
        "F_p^3 and F_p^2 systems must have distinct vk_digest"
    );

    // Witness satisfying BOTH A and B (all zeros!) — yet cross-verification MUST fail
    // because the transcript binds `vk_digest`!
    let zero_trace = vec![RowMajorMatrix::new(vec![Val::ZERO; 8 * 3], 3)];
    let proof_a =
        sys_a.prove_multiple_claims(&key_a, &[], SystemWitness::from_stage_1(zero_trace, &sys_a));
    sys_a.verify_multiple_claims(&[], &proof_a).unwrap();
    assert!(
        sys_b.verify_multiple_claims(&[], &proof_a).is_err(),
        "MED-5: even on a witness satisfying both circuits, proof for A must fail verification under B"
    );
}

#[test]
fn v6_fp3_canonical_deserialization_and_tamper_rejection() {
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: 0,
    };
    let fri = FriParameters {
        log_final_poly_len: 1,
        max_log_arity: if cfg!(feature = "cuda") { 1 } else { 2 },
        num_queries: 10,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };
    let config = GoldilocksBlake3V6Config::new(comm, fri);
    let (system, key) = System::new(config, multi_circuit_inputs(2, 16, true));
    let (traces, claims) = build_multi_circuit_witness(16, true, 5);
    let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();

    let proof = system.prove_multiple_claims(
        &key,
        &claim_refs,
        SystemWitness::from_stage_1(traces.clone(), &system),
    );
    system.verify_multiple_claims(&claim_refs, &proof).unwrap();
    let proof_bytes = proof.to_bytes().unwrap();

    // 1. Trailing garbage bytes rejected for Proof.
    let mut with_trailing = proof_bytes.clone();
    with_trailing.push(0x42);
    assert!(
        Proof::<GoldilocksBlake3V6Config>::from_bytes(&with_trailing).is_err(),
        "Proof::from_bytes must reject trailing bytes"
    );

    // 2. Non-canonical Goldilocks field elements (>= P) rejected in Proof::from_bytes.
    let p_bytes = Val::ORDER_U64.to_le_bytes();
    let max_bytes = u64::MAX.to_le_bytes();
    // Locate a known canonical u64 coordinate of `intermediate_accumulators[0]` in the wire bytes.
    let target_u64 = <ExtVal3 as BasedVectorSpace<Val>>::as_basis_coefficients_slice(
        &proof.intermediate_accumulators[0],
    )[0]
    .as_canonical_u64()
    .to_le_bytes();
    let pos = proof_bytes
        .windows(8)
        .position(|w| w == target_u64)
        .expect("serialized accumulator coordinate not found in proof bytes");
    for bad_word in [p_bytes, max_bytes] {
        let mut non_canon = proof_bytes.clone();
        non_canon[pos..pos + 8].copy_from_slice(&bad_word);
        assert!(
            Proof::<GoldilocksBlake3V6Config>::from_bytes(&non_canon).is_err(),
            "Proof::from_bytes must reject non-canonical Goldilocks word {bad_word:?}"
        );
    }

    // 3. Semantic tamper checks: accumulator, opened values, FRI final_poly, Merkle sibling, claim.
    let one_ef = <ExtVal3 as PrimeCharacteristicRing>::ONE;
    let mut bad = proof.clone();
    bad.intermediate_accumulators[0] += one_ef;
    assert!(system.verify_multiple_claims(&claim_refs, &bad).is_err());

    let mut bad = proof.clone();
    bad.stage_1_opened_values[0][0][0] += one_ef;
    assert!(system.verify_multiple_claims(&claim_refs, &bad).is_err());

    let mut bad = proof.clone();
    bad.stage_2_opened_values[0][0][0] += one_ef;
    assert!(system.verify_multiple_claims(&claim_refs, &bad).is_err());

    let mut bad = proof.clone();
    bad.quotient_opened_values[0][0][0] += one_ef;
    assert!(system.verify_multiple_claims(&claim_refs, &bad).is_err());

    let mut bad = proof.clone();
    bad.preprocessed_opened_values.as_mut().unwrap()[0][0][0] += one_ef;
    assert!(system.verify_multiple_claims(&claim_refs, &bad).is_err());

    let mut bad = proof.clone();
    bad.opening_proof.final_poly[0] += one_ef;
    assert!(system.verify_multiple_claims(&claim_refs, &bad).is_err());

    let mut bad = proof.clone();
    if let Some(first_sib) = bad.opening_proof.input_openings[0]
        .opening_proof
        .sibling_hashes
        .first_mut()
    {
        first_sib[0] ^= 1;
        assert!(system.verify_multiple_claims(&claim_refs, &bad).is_err());
    }

    let mut wrong_claims = claims.clone();
    wrong_claims[0][1] += Val::ONE;
    let wrong_claim_refs: Vec<&[Val]> = wrong_claims.iter().map(Vec::as_slice).collect();
    assert!(
        system
            .verify_multiple_claims(&wrong_claim_refs, &proof)
            .is_err()
    );

    // 4. BatchProof trailing bytes, non-canonical field element, and non-target shard header validation.
    let batch = system.prove_batch(
        &key,
        vec![
            ShardInput {
                witness: SystemWitness::from_stage_1(traces.clone(), &system),
                claims: claims.clone(),
            },
            ShardInput {
                witness: SystemWitness::from_stage_1(traces, &system),
                claims,
            },
        ],
        vec![],
    );
    let batch_bytes = batch.to_bytes().unwrap();
    let mut batch_trailing = batch_bytes.clone();
    batch_trailing.push(0x99);
    assert!(
        BatchProof::<GoldilocksBlake3V6Config>::from_bytes(&batch_trailing).is_err(),
        "BatchProof::from_bytes must reject trailing bytes"
    );

    let batch_target = <ExtVal3 as BasedVectorSpace<Val>>::as_basis_coefficients_slice(
        &batch.proofs[0].intermediate_accumulators[0],
    )[0]
    .as_canonical_u64()
    .to_le_bytes();
    let bpos = batch_bytes
        .windows(8)
        .position(|w| w == batch_target)
        .expect("serialized accumulator coordinate not found in batch bytes");
    let mut batch_non_canon = batch_bytes;
    batch_non_canon[bpos..bpos + 8].copy_from_slice(&p_bytes);
    assert!(
        BatchProof::<GoldilocksBlake3V6Config>::from_bytes(&batch_non_canon).is_err(),
        "BatchProof::from_bytes must reject non-canonical Goldilocks word"
    );

    // Corrupting shard 1's preamble `log_degrees` length must be rejected even when
    // verifying only shard 0 via `verify_batch_shard`.
    let mut malformed_batch = batch;
    malformed_batch.preamble.headers[1].log_degrees.pop();
    assert!(
        system
            .verify_batch_shard(&malformed_batch.preamble, 0, &malformed_batch.proofs[0])
            .is_err(),
        "verify_batch_shard must reject malformed non-target shard headers"
    );
}

fn min_and_median_ms(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(f64::total_cmp);
    (samples[0], samples[samples.len() / 2])
}

#[test]
#[ignore = "CPU benchmark: run with `cargo test --release --features parallel --test fp3_v6 bench_fp2_vs_fp3_cpu -- --ignored --nocapture`"]
fn bench_fp2_vs_fp3_cpu() {
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: 0,
    };
    let fri = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 2,
        num_queries: 64,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };

    for &rows in &[1usize << 12, 1usize << 14, 1usize << 16] {
        let inputs = u32_add_system_inputs(4);
        let (traces, claims) = u32_add_witness(rows);
        let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();

        let (sys_v5, key_v5) = System::new(GoldilocksBlake3Config::new(comm, fri), inputs.clone());
        let (sys_v6_fp2_t, key_v6_fp2_t) = System::new(
            GoldilocksBlake3V6Fp2Config::new_with_batching(comm, fri, FriBatching::Tensor),
            inputs.clone(),
        );
        let (sys_v6_fp3_h, key_v6_fp3_h) = System::new(
            GoldilocksBlake3V6Config::new_with_batching(comm, fri, FriBatching::Horner),
            inputs.clone(),
        );
        let (sys_v6_fp3_t, key_v6_fp3_t) = System::new(
            GoldilocksBlake3V6Config::new_with_batching(comm, fri, FriBatching::Tensor),
            inputs,
        );

        // Warmup all 4 configs.
        let p_v5 = sys_v5.prove_multiple_claims(
            &key_v5,
            &claim_refs,
            SystemWitness::from_stage_1(traces.clone(), &sys_v5),
        );
        let p_v6_fp2_t = sys_v6_fp2_t.prove_multiple_claims(
            &key_v6_fp2_t,
            &claim_refs,
            SystemWitness::from_stage_1(traces.clone(), &sys_v6_fp2_t),
        );
        let p_v6_fp3_h = sys_v6_fp3_h.prove_multiple_claims(
            &key_v6_fp3_h,
            &claim_refs,
            SystemWitness::from_stage_1(traces.clone(), &sys_v6_fp3_h),
        );
        let p_v6_fp3_t = sys_v6_fp3_t.prove_multiple_claims(
            &key_v6_fp3_t,
            &claim_refs,
            SystemWitness::from_stage_1(traces.clone(), &sys_v6_fp3_t),
        );

        // Interleaved timed rounds to cancel host load drift.
        let mut prove = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..5 {
            let w = SystemWitness::from_stage_1(traces.clone(), &sys_v5);
            let t0 = Instant::now();
            let _ = sys_v5.prove_multiple_claims(&key_v5, &claim_refs, w);
            prove[0].push(t0.elapsed().as_secs_f64() * 1e3);

            let w = SystemWitness::from_stage_1(traces.clone(), &sys_v6_fp2_t);
            let t0 = Instant::now();
            let _ = sys_v6_fp2_t.prove_multiple_claims(&key_v6_fp2_t, &claim_refs, w);
            prove[1].push(t0.elapsed().as_secs_f64() * 1e3);

            let w = SystemWitness::from_stage_1(traces.clone(), &sys_v6_fp3_h);
            let t0 = Instant::now();
            let _ = sys_v6_fp3_h.prove_multiple_claims(&key_v6_fp3_h, &claim_refs, w);
            prove[2].push(t0.elapsed().as_secs_f64() * 1e3);

            let w = SystemWitness::from_stage_1(traces.clone(), &sys_v6_fp3_t);
            let t0 = Instant::now();
            let _ = sys_v6_fp3_t.prove_multiple_claims(&key_v6_fp3_t, &claim_refs, w);
            prove[3].push(t0.elapsed().as_secs_f64() * 1e3);
        }

        let mut verify = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..20 {
            let t0 = Instant::now();
            sys_v5.verify_multiple_claims(&claim_refs, &p_v5).unwrap();
            verify[0].push(t0.elapsed().as_secs_f64() * 1e3);

            let t0 = Instant::now();
            sys_v6_fp2_t
                .verify_multiple_claims(&claim_refs, &p_v6_fp2_t)
                .unwrap();
            verify[1].push(t0.elapsed().as_secs_f64() * 1e3);

            let t0 = Instant::now();
            sys_v6_fp3_h
                .verify_multiple_claims(&claim_refs, &p_v6_fp3_h)
                .unwrap();
            verify[2].push(t0.elapsed().as_secs_f64() * 1e3);

            let t0 = Instant::now();
            sys_v6_fp3_t
                .verify_multiple_claims(&claim_refs, &p_v6_fp3_t)
                .unwrap();
            verify[3].push(t0.elapsed().as_secs_f64() * 1e3);
        }

        let sizes = [
            p_v5.to_bytes().unwrap().len(),
            p_v6_fp2_t.to_bytes().unwrap().len(),
            p_v6_fp3_h.to_bytes().unwrap().len(),
            p_v6_fp3_t.to_bytes().unwrap().len(),
        ];
        let labels = [
            "`v5` `F_p^2` Horner",
            "`v6` `F_p^2` Tensor",
            "`v6` `F_p^3` Horner",
            "`v6` `F_p^3` Tensor",
        ];

        println!(
            "\n=== U32Add + ByteTable: {rows} rows (2^{}) | blowup=4, queries=64, arity=4 ===",
            rows.ilog2()
        );
        println!("| Config | Prove min / med (ms) | Verify min / med (ms) | Proof size (bytes) |");
        println!("|---|---:|---:|---:|");
        for i in 0..4 {
            let (p_min, p_med) = min_and_median_ms(prove[i].clone());
            let (v_min, v_med) = min_and_median_ms(verify[i].clone());
            println!(
                "| {} | {:.2} / {:.2} | {:.2} / {:.2} | {} |",
                labels[i], p_min, p_med, v_min, v_med, sizes[i]
            );
        }
    }
}

fn pack_le_u64s(words: [u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, word) in words.into_iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
    out
}

fn quad_v6_system(
    cap_height: usize,
) -> (
    System<GoldilocksBlake3V6Config>,
    multi_stark::system::ProverKey<GoldilocksBlake3V6Config>,
) {
    let cfg = GoldilocksBlake3V6Config::new_with_batching(
        CommitmentParameters {
            log_blowup: 2,
            cap_height,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 16,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
        FriBatching::Tensor,
    );
    let a = Expr::main(0);
    let b = Expr::main(1);
    System::new(
        cfg,
        [CircuitInputs {
            main_width: 2,
            preprocessed: None,
            constraints: vec![a.clone() * a - b],
            ext_constraints: vec![],
            lookups: vec![],
            lookup_group_size: 1,
        }],
    )
}

fn quad_trace_v6(log_degree: usize) -> RowMajorMatrix<Val> {
    let rows = 1usize << log_degree;
    let mut vals = Vec::with_capacity(rows * 2);
    for i in 0..rows {
        let x = Val::from_usize(i + 1);
        vals.push(x);
        vals.push(x * x);
    }
    RowMajorMatrix::new(vals, 2)
}

#[test]
fn v6_fp3_rejects_merkle_cap_forgery_and_grinding() {
    use multi_stark::verifier::VerificationError;
    use p3_fri::verifier::FriError;
    use p3_merkle_tree::MerkleTreeError;
    use p3_symmetric::MerkleCap;

    // #240 #1A: verify_batch_shard forgery via 2-root non-target MerkleCap.
    let (system, key) = quad_v6_system(0);
    let shards = vec![
        ShardInput {
            witness: SystemWitness::from_stage_1(vec![quad_trace_v6(5)], &system),
            claims: vec![],
        },
        ShardInput {
            witness: SystemWitness::from_stage_1(vec![quad_trace_v6(5)], &system),
            claims: vec![],
        },
    ];
    let messages_a = vec![
        BatchMessage {
            args: vec![Val::from_u64(7)],
            multiplicity: Val::from_u64(1),
        },
        BatchMessage {
            args: vec![Val::from_u64(999)],
            multiplicity: Val::from_u64(0),
        },
    ];
    let batch_a = system.prove_batch(&key, shards, messages_a);
    let _ = system
        .verify_batch_shard(&batch_a.preamble, 0, &batch_a.proofs[0])
        .unwrap();

    let r1 = batch_a.preamble.headers[1].stage_1_trace.roots()[0];
    let mut preamble_b = batch_a.preamble.clone();
    preamble_b.headers[1].stage_1_trace = MerkleCap::new(vec![r1, pack_le_u64s([1, 5, 0, 2])]);
    preamble_b.headers[1].log_degrees = vec![7];
    preamble_b.headers[1].claims = vec![vec![Val::from_u64(999)]];
    preamble_b.messages = vec![];

    let (_, beta_a, gamma_a) = system.batch_challenger(&batch_a.preamble);
    let (_, beta_b, gamma_b) = system.batch_challenger(&preamble_b);
    assert_eq!((beta_a, gamma_a), (beta_b, gamma_b));
    assert!(matches!(
        system.verify_batch_shard(&preamble_b, 0, &batch_a.proofs[0]),
        Err(VerificationError::BatchShapeMismatch)
    ));

    // #240 #1B: full verify_batch preamble/message forgery.
    let shards_b = vec![
        ShardInput {
            witness: SystemWitness::from_stage_1(vec![quad_trace_v6(5)], &system),
            claims: vec![],
        },
        ShardInput {
            witness: SystemWitness::from_stage_1(vec![quad_trace_v6(5)], &system),
            claims: vec![],
        },
    ];
    let messages_b = vec![
        BatchMessage {
            args: vec![Val::from_u64(5)],
            multiplicity: Val::from_u64(0),
        },
        BatchMessage {
            args: vec![Val::from_u64(0)],
            multiplicity: Val::from_u64(0),
        },
    ];
    let batch_full_a = system.prove_batch(&key, shards_b, messages_b);
    system.verify_batch(&batch_full_a).unwrap();

    let r1_full = batch_full_a.preamble.headers[1].stage_1_trace.roots()[0];
    let forged_cap = MerkleCap::new(vec![r1_full, pack_le_u64s([1, 5, 0, 2])]);
    let mut batch_full_b = batch_full_a.clone();
    batch_full_b.preamble.headers[1].stage_1_trace = forged_cap.clone();
    batch_full_b.proofs[1].commitments.stage_1_trace = forged_cap;
    batch_full_b.preamble.messages = vec![BatchMessage {
        args: vec![],
        multiplicity: Val::from_u64(0),
    }];
    let (_, ba, ga) = system.batch_challenger(&batch_full_a.preamble);
    let (_, bb, gb) = system.batch_challenger(&batch_full_b.preamble);
    assert_eq!((ba, ga), (bb, gb));
    assert!(matches!(
        system.verify_batch(&batch_full_b),
        Err(VerificationError::BatchShapeMismatch)
    ));

    // #240 #1C: single-proof extra/truncated cap roots on stage-1, stage-2, quotient, and FRI.
    let proof = system.prove_multiple_claims(
        &key,
        &[],
        SystemWitness::from_stage_1(vec![quad_trace_v6(4)], &system),
    );
    system.verify_multiple_claims(&[], &proof).unwrap();

    let inflate = |cap: &MerkleCap<Val, [u8; 32]>| {
        let mut roots = cap.roots().to_vec();
        roots.push([0x5A; 32]);
        MerkleCap::new(roots)
    };
    let mut p_s1 = proof.clone();
    p_s1.commitments.stage_1_trace = inflate(&proof.commitments.stage_1_trace);
    assert!(matches!(
        system.verify_multiple_claims(&[], &p_s1),
        Err(VerificationError::InvalidProofShape)
    ));

    let mut p_s2 = proof.clone();
    p_s2.commitments.stage_2_trace = inflate(&proof.commitments.stage_2_trace);
    assert!(matches!(
        system.verify_multiple_claims(&[], &p_s2),
        Err(VerificationError::InvalidProofShape)
    ));

    let mut p_q = proof.clone();
    p_q.commitments.quotient_chunks = inflate(&proof.commitments.quotient_chunks);
    assert!(matches!(
        system.verify_multiple_claims(&[], &p_q),
        Err(VerificationError::InvalidProofShape)
    ));

    let mut p_fri = proof;
    p_fri.opening_proof.commit_phase_commits[0] =
        inflate(&p_fri.opening_proof.commit_phase_commits[0]);
    assert!(matches!(
        system.verify_multiple_claims(&[], &p_fri),
        Err(VerificationError::InvalidOpeningArgument(
            FriError::CommitPhaseMmcsError(MerkleTreeError::CapMismatch)
        ))
    ));

    #[cfg(not(feature = "cuda"))]
    {
        let (sys1, key1) = quad_v6_system(1);
        let proof1 = sys1.prove_multiple_claims(
            &key1,
            &[],
            SystemWitness::from_stage_1(vec![quad_trace_v6(4)], &sys1),
        );
        sys1.verify_multiple_claims(&[], &proof1).unwrap();
        let fri_roots = proof1.opening_proof.commit_phase_commits[0]
            .roots()
            .to_vec();
        let mut trunc = proof1;
        trunc.opening_proof.commit_phase_commits[0] = MerkleCap::new(vec![fri_roots[0]]);
        assert!(matches!(
            sys1.verify_multiple_claims(&[], &trunc),
            Err(VerificationError::InvalidOpeningArgument(
                FriError::CommitPhaseMmcsError(MerkleTreeError::CapMismatch)
            ))
        ));
    }
}

#[test]
fn v6_fp3_rejects_preprocessed_malleability_and_bad_preamble_dimensions() {
    use multi_stark::verifier::VerificationError;

    // M24(b): num_preprocessed == 0 must reject `Some(vec![])` to prevent proof malleability.
    let (system, key) = quad_v6_system(0);
    let proof = system.prove_multiple_claims(
        &key,
        &[],
        SystemWitness::from_stage_1(vec![quad_trace_v6(4)], &system),
    );
    assert!(proof.preprocessed_opened_values.is_none());
    system.verify_multiple_claims(&[], &proof).unwrap();

    let mut malleable = proof;
    malleable.preprocessed_opened_values = Some(vec![]);
    assert!(matches!(
        system.verify_multiple_claims(&[], &malleable),
        Err(VerificationError::InvalidProofShape)
    ));
    let encoded = malleable.to_bytes().unwrap();
    let decoded = Proof::<GoldilocksBlake3V6Config>::from_bytes(&encoded).unwrap();
    assert!(matches!(
        system.verify_multiple_claims(&[], &decoded),
        Err(VerificationError::InvalidProofShape)
    ));

    // M24(c): verify_preamble_headers must enforce preprocessed_height and
    // max_log_quotient_domain for non-target shards in verify_batch_shard.
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: 0,
    };
    let fri = FriParameters {
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 16,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };
    let inputs = u32_add_system_inputs(4);
    let (sys_prep, key_prep) = System::new(
        GoldilocksBlake3V6Config::new_with_batching(comm, fri, FriBatching::Tensor),
        inputs,
    );
    let (traces, claims) = u32_add_witness(32);
    let byte_table = traces[0].clone();
    let add_half = |start: usize, end: usize| {
        RowMajorMatrix::new(traces[1].values[start * 14..end * 14].to_vec(), 14)
    };
    let shards = vec![
        ShardInput {
            witness: SystemWitness::from_stage_1(vec![byte_table, add_half(0, 16)], &sys_prep),
            claims: claims.clone(),
        },
        ShardInput {
            witness: SystemWitness::from_stage_1(
                vec![RowMajorMatrix::new(vec![], 1), add_half(16, 32)],
                &sys_prep,
            ),
            claims: vec![],
        },
    ];
    let batch = sys_prep.prove_batch(&key_prep, shards, vec![]);
    let _ = sys_prep
        .verify_batch_shard(&batch.preamble, 0, &batch.proofs[0])
        .unwrap();

    // Non-target shard 1 claims wrong preprocessed height for circuit 0 (256-row byte table).
    let mut bad_prep = batch.preamble.clone();
    bad_prep.headers[1].active[0] = true;
    bad_prep.headers[1].log_degrees.insert(0, 4);
    assert!(matches!(
        sys_prep.verify_batch_shard(&bad_prep, 0, &batch.proofs[0]),
        Err(VerificationError::BatchShapeMismatch)
    ));

    // Non-target shard 1 satisfies `log_degree <= max_log_degree` yet violates
    // `log_degree + log2(quotient_degree) <= max_log_quotient_domain` (u32_add has
    // lookup_group_size = 4 => quotient_degree >= 2).
    assert!(sys_prep.circuits[1].quotient_degree() >= 2);
    let mut bad_quot = batch.preamble;
    bad_quot.headers[1].log_degrees[0] =
        u8::try_from(sys_prep.config.max_log_quotient_domain()).unwrap();
    assert!(usize::from(bad_quot.headers[1].log_degrees[0]) <= sys_prep.config.max_log_degree());
    assert!(matches!(
        sys_prep.verify_batch_shard(&bad_quot, 0, &batch.proofs[0]),
        Err(VerificationError::BatchShapeMismatch)
    ));
}

/// L9(d): Exercise LDE heights strictly above 1024 (`512 << 2 = 2048 > 1024`)
/// so CUDA resident/streamed FRI `_ext3` kernels are covered on GPU as well as
/// CPU `v6` `F_p^3` (both `Tensor` and `Horner` batching).
#[test]
fn v6_fp3_large_lde_above_1024_rows_roundtrip() {
    let comm = CommitmentParameters {
        log_blowup: 2,
        cap_height: 0,
    };
    let fri = FriParameters {
        log_final_poly_len: 1,
        max_log_arity: 2,
        num_queries: 24,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
    };
    let inputs = u32_add_system_inputs(4);
    let (traces, claims) = u32_add_witness(512);
    let claim_refs: Vec<&[Val]> = claims.iter().map(Vec::as_slice).collect();
    assert_eq!(traces[1].height(), 512);
    assert!(traces[1].height() << comm.log_blowup > 1024);

    for batching in [FriBatching::Tensor, FriBatching::Horner] {
        let (system, key) = System::new(
            GoldilocksBlake3V6Config::new_with_batching(comm, fri, batching),
            inputs.clone(),
        );
        let proof = system.prove_multiple_claims(
            &key,
            &claim_refs,
            SystemWitness::from_stage_1(traces.clone(), &system),
        );
        system.verify_multiple_claims(&claim_refs, &proof).unwrap();

        let bytes = proof.to_bytes().unwrap();
        let decoded = Proof::<GoldilocksBlake3V6Config>::from_bytes(&bytes).unwrap();
        system
            .verify_multiple_claims(&claim_refs, &decoded)
            .unwrap();

        let mut tampered = decoded;
        tampered.stage_1_opened_values[1][0][0] += ExtVal3::ONE;
        assert!(
            system
                .verify_multiple_claims(&claim_refs, &tampered)
                .is_err()
        );
    }
}
