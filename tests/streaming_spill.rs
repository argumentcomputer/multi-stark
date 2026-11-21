//! End-to-end verification of A5 streaming/lazy lookup evaluation and
//! disk-backed LDE spilling:
//! - 100% byte-identical `Proof::to_bytes()` and `BatchProof::to_bytes()` across
//!   in-memory, lazy-lookup, `new_without_preprocessed`, and disk-spilled paths.
//! - Coverage of preprocessed circuits (used in constraints and lookups), sparse
//!   activation (including inactive preprocessed circuits), lookup groups `k > 1`,
//!   `log_blowup = 1, 2`, `cap_height = 0, 1, 2`, `max_log_arity = 1, 2`, positive
//!   PoW, `TraceSource::Generated`, manual `Stage1` construction, and multi-shard
//!   batches (`Retain` + `Regenerate` + `BatchMessage`s).
//! - Verification of zero upfront `LookupValues` payloads, zero unintended
//!   re-materialization of spilled LDEs, and automatic cleanup on drop and panic.

use std::fs;
use std::path::PathBuf;
#[cfg(not(feature = "cuda"))]
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use multi_stark::batch::{BatchMessage, BatchProof, Retention, ShardInput};
use multi_stark::config::ProofConfig;
use multi_stark::expr::Expr;
use multi_stark::lookup::{Lookup, LookupValues};
use multi_stark::prover::{Proof, Stage1};
use multi_stark::system::{CircuitInputs, System, SystemWitness};
use multi_stark::traits::Pcs as _;
use multi_stark::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val};
#[cfg(not(feature = "cuda"))]
use multi_stark::witness::TraceGenerator;
use multi_stark::witness::{PreparedWitness, TraceSource};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

static DIR_SEQ: AtomicU64 = AtomicU64::new(0);

fn fresh_spill_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "multi-stark-a5-{tag}-{}-{nanos}-{seq}",
        std::process::id()
    ))
}

fn f(v: u32) -> Val {
    Val::from_u32(v)
}

/// Deterministic `TraceGenerator` wrapping a frozen row-major matrix, used to
/// verify `TraceSource::Generated` on both lazy and disk-spilled paths.
#[cfg(not(feature = "cuda"))]
struct MatrixGenerator {
    matrix: RowMajorMatrix<Val>,
}

#[cfg(not(feature = "cuda"))]
impl TraceGenerator<Val> for MatrixGenerator {
    fn height(&self) -> usize {
        self.matrix.height()
    }

    fn width(&self) -> usize {
        self.matrix.width()
    }

    fn write_rows(&self, first: usize, output: &mut [Val]) {
        let width = self.width();
        let height = self.height();
        let rows = output.len() / width;
        for local in 0..rows {
            let src = (first + local) % height;
            output[local * width..(local + 1) * width]
                .copy_from_slice(&self.matrix.values[src * width..(src + 1) * width]);
        }
    }
}

/// Builds a 4-circuit system exercising:
/// - Circuit 0: Preprocessed 256-row byte table (`prep[0] = 0..255`, `main[0] = mult`)
///   whose preprocessed column is directly read by its lookup expression.
/// - Circuit 1: Preprocessed 16-row auxiliary table (`prep[0] = 100..116`, `main[0] = mult`)
///   that can be left INACTIVE to exercise sparse activation of preprocessed circuits.
/// - Circuit 2: Active arithmetic + multi-lookup circuit with configurable
///   `lookup_group_size`, a next-row transition constraint (`main[0]' == main[0] + 1`
///   on transitions), a preprocessed selector column (`prep[0] = row_idx`), and
///   5 lookups per row (1 claim pull + 4 byte-table pushes, exercising partial
///   and full lookup groups across `k = 1..=4`).
/// - Circuit 3: Zero-lookup active circuit (pass-through accumulator) with a
///   degree-2 constraint (`main[0]^2 == main[1]`).
fn multi_circuit_inputs(lookup_group_size: usize, work_rows: usize) -> Vec<CircuitInputs<Val>> {
    assert!(work_rows.is_power_of_two() && work_rows >= 4);

    // Circuit 0: preprocessed byte table (height 256).
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

    // Circuit 1: optional preprocessed table (height 16), left inactive in sparse tests.
    let aux_prep = RowMajorMatrix::new((100..116u32).map(f).collect(), 1);
    let aux_table = CircuitInputs {
        main_width: 1,
        preprocessed: Some(aux_prep),
        constraints: vec![
            // Simple constraint referencing preprocessed column.
            Expr::main(0) * (Expr::preprocessed(0) - Expr::preprocessed(0)),
        ],
        ext_constraints: vec![],
        lookups: vec![Lookup::pull(
            Expr::main(0),
            vec![Expr::constant(f(20)), Expr::preprocessed(0)],
        )],
        lookup_group_size: 1,
    };

    // Circuit 2: arithmetic + 5 lookups per row (cols: a, b, c, d, sum, mult).
    // Preprocessed col 0 holds a row tag `tag_r = r + 1`, read in BOTH a
    // constraint and the claim lookup (including next-row wrap-around!).
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

    let work_constraints = vec![
        // sum == a + b + c + d + tag
        sum.clone() - (a.clone() + b.clone() + c.clone() + d.clone() + tag.clone()),
        // Transition constraint using Next row: on transitions, a' == a + 1.
        Expr::IsTransition * (a_next.clone() - (a.clone() + Expr::constant(Val::ONE))),
    ];
    let work_lookups = vec![
        // Pull public claim: [30, tag, tag_next, sum, a_next] — exercises both
        // current and next-row wraparound for main AND preprocessed columns!
        Lookup::pull(
            mult,
            vec![Expr::constant(f(30)), tag, tag_next, sum, a_next],
        ),
        // Push 4 bytes into Circuit 0's byte table.
        Lookup::push(Expr::constant(Val::ONE), vec![Expr::constant(f(10)), a]),
        Lookup::push(Expr::constant(Val::ONE), vec![Expr::constant(f(10)), b]),
        Lookup::push(Expr::constant(Val::ONE), vec![Expr::constant(f(10)), c]),
        Lookup::push(Expr::constant(Val::ONE), vec![Expr::constant(f(10)), d]),
    ];
    let work_circuit = CircuitInputs {
        main_width: 6,
        preprocessed: Some(work_prep),
        constraints: work_constraints,
        ext_constraints: vec![],
        lookups: work_lookups,
        lookup_group_size,
    };

    // Circuit 3: zero-lookup active circuit (x, x^2).
    let x = Expr::main(0);
    let x2 = Expr::main(1);
    let no_lookup_circuit = CircuitInputs {
        main_width: 2,
        preprocessed: None,
        constraints: vec![x.clone() * x - x2],
        ext_constraints: vec![],
        lookups: vec![],
        lookup_group_size: 1,
    };

    vec![byte_table, aux_table, work_circuit, no_lookup_circuit]
}

/// Builds traces and matching public claims for `multi_circuit_inputs`.
/// Circuit 1 (preprocessed) is deliberately deactivated (`height == 0`) when
/// `activate_aux == false`, and Circuit 3 is deactivated when `activate_no_lookup == false`.
fn build_traces_and_claims(
    work_rows: usize,
    seed: u32,
    activate_aux: bool,
    activate_no_lookup: bool,
) -> (Vec<RowMajorMatrix<Val>>, Vec<Vec<Val>>) {
    let mut byte_counts = vec![0u32; 256];
    let mut work_vals = Vec::with_capacity(work_rows * 6);
    let mut claims = Vec::with_capacity(work_rows);

    let work_rows_u32 = u32::try_from(work_rows).unwrap();
    for r in 0..work_rows_u32 {
        let a = seed + r;
        assert!(a < 256);
        let a_next = seed + ((r + 1) % work_rows_u32);
        let b = (seed * 3 + r * 7 + 11) % 256;
        let c = (seed * 5 + r * 13 + 29) % 256;
        let d = (seed * 7 + r * 17 + 43) % 256;
        let tag = r + 1;
        let tag_next = ((r + 1) % work_rows_u32) + 1;
        let sum = a + b + c + d + tag;
        byte_counts[a as usize] += 1;
        byte_counts[b as usize] += 1;
        byte_counts[c as usize] += 1;
        byte_counts[d as usize] += 1;
        work_vals.extend_from_slice(&[f(a), f(b), f(c), f(d), f(sum), Val::ONE]);
        claims.push(vec![f(30), f(tag), f(tag_next), f(sum), f(a_next)]);
    }

    let byte_trace = RowMajorMatrix::new(byte_counts.into_iter().map(f).collect(), 1);
    let aux_trace = if activate_aux {
        let mut aux_vals = vec![Val::ZERO; 16];
        aux_vals[3] = Val::ONE;
        claims.push(vec![f(20), f(103)]);
        RowMajorMatrix::new(aux_vals, 1)
    } else {
        RowMajorMatrix::new(vec![], 1)
    };
    let work_trace = RowMajorMatrix::new(work_vals, 6);
    let no_lookup_trace = if activate_no_lookup {
        RowMajorMatrix::new(
            (0..8u32)
                .flat_map(|v| [f(v + 2), f((v + 2) * (v + 2))])
                .collect(),
            2,
        )
    } else {
        RowMajorMatrix::new(vec![], 2)
    };

    (
        vec![byte_trace, aux_trace, work_trace, no_lookup_trace],
        claims,
    )
}

fn make_config(
    log_blowup: usize,
    cap_height: usize,
    max_log_arity: usize,
    pow_bits: usize,
) -> GoldilocksBlake3Config {
    GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup,
            cap_height,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity,
            num_queries: 16,
            commit_proof_of_work_bits: pow_bits,
            query_proof_of_work_bits: pow_bits,
        },
    )
}

#[test]
fn streaming_and_spilled_proofs_are_byte_identical_across_configs() {
    // Test matrix:
    // (lookup_group_size, log_blowup, cap_height, max_log_arity, pow_bits, work_rows, activate_aux, fc4336b_blake3_pin)
    let cases = [
        // Default ungrouped lookups, blowup 2, cap 0, binary FRI, sparse inactive preprocessed circuit.
        (
            1usize,
            1usize,
            0usize,
            1usize,
            0usize,
            16usize,
            false,
            "f227d2e461a5e8219d53a79add0895d331d772813e7a14782d9d603871b7613d",
        ),
        // Grouped lookups k = 2, blowup 4 (q=4 supported), cap 1, higher arity, positive PoW.
        (
            2,
            2,
            1,
            2,
            4,
            32,
            false,
            "bbb754d7390697ae17394266c2cb262c5c937a3584d5b56646cbf3a8a3b9b7a5",
        ),
        // Grouped lookups k = 3, blowup 2 (q=4 -> log_blowup=2), cap 2, all circuits active.
        (
            3,
            2,
            2,
            1,
            0,
            16,
            true,
            "ae7b731bf392f7a6988677261ab4ee30f21cac2ac143d018da2b4141e286ea8e",
        ),
        // Grouped lookups k = 4 (5 lookups => groups of 4 and 1!), blowup 2, cap 0.
        (
            4,
            2,
            0,
            2,
            3,
            16,
            false,
            "cd603a7772b7df77ab3785ed6db70b16a3011849c4e71a276f8ffed7be51622f",
        ),
    ];

    for (
        case_idx,
        &(k, log_blowup, cap_height, max_log_arity, pow_bits, work_rows, activate_aux, ref_pin),
    ) in cases.iter().enumerate()
    {
        #[cfg(feature = "cuda")]
        if cap_height != 0 || max_log_arity > 1 {
            continue;
        }
        let inputs = multi_circuit_inputs(k, work_rows);
        let (traces, claims_owned) = build_traces_and_claims(work_rows, 5, activate_aux, true);
        let claim_slices: Vec<&[Val]> = claims_owned.iter().map(Vec::as_slice).collect();

        // 1. Baseline: System::new + SystemWitness::from_stage_1 + in-memory config.
        let cfg_base = make_config(log_blowup, cap_height, max_log_arity, pow_bits);
        let (sys_base, key_base) = System::new(cfg_base, inputs.clone());
        let wit_base = SystemWitness::from_stage_1(traces.clone(), &sys_base);
        let proof_base = sys_base.prove_multiple_claims(&key_base, &claim_slices, wit_base);
        sys_base
            .verify_multiple_claims(&claim_slices, &proof_base)
            .expect("baseline proof must verify");
        let expected_bytes = proof_base.to_bytes().unwrap();
        assert_eq!(
            blake3::hash(&expected_bytes).to_hex().as_str(),
            ref_pin,
            "case {case_idx}: proof bytes drifted from fc4336b baseline ({} bytes)",
            expected_bytes.len()
        );

        // 2. Lazy lookups in-memory (System::new + SystemWitness::from_stage_1_lazy).
        let wit_lazy = SystemWitness::from_stage_1_lazy(traces.clone(), &sys_base);
        assert!(
            wit_lazy.lookups.iter().all(|l| !l.is_materialized()),
            "lazy witness must not materialize lookup payloads"
        );
        let proof_lazy = sys_base.prove_multiple_claims(&key_base, &claim_slices, wit_lazy);
        assert_eq!(
            proof_lazy.to_bytes().unwrap(),
            expected_bytes,
            "case {case_idx}: lazy in-memory proof bytes must match baseline"
        );

        // 3. System::new_without_preprocessed + lazy in-memory.
        let cfg_no_prep = make_config(log_blowup, cap_height, max_log_arity, pow_bits);
        let (sys_no_prep, key_no_prep) =
            System::new_without_preprocessed(cfg_no_prep, inputs.clone());
        assert!(
            sys_no_prep
                .circuits
                .iter()
                .all(|c| c.preprocessed.is_none())
        );
        let wit_no_prep = SystemWitness::from_stage_1_lazy(traces.clone(), &sys_no_prep);
        let proof_no_prep =
            sys_no_prep.prove_multiple_claims(&key_no_prep, &claim_slices, wit_no_prep);
        assert_eq!(
            proof_no_prep.to_bytes().unwrap(),
            expected_bytes,
            "case {case_idx}: new_without_preprocessed lazy in-memory proof bytes must match"
        );

        // 4. Full disk-spilled + materialized SystemWitness::from_stage_1.
        let spill_dir = fresh_spill_dir(&format!("case-{case_idx}"));
        let cfg_spilled = make_config(log_blowup, cap_height, max_log_arity, pow_bits)
            .with_disk_spill(&spill_dir);
        #[cfg(not(feature = "cuda"))]
        assert_eq!(cfg_spilled.spill_dir(), Some(spill_dir.as_path()));
        #[cfg(feature = "cuda")]
        assert!(cfg_spilled.spill_dir().is_none());
        let (sys_spilled, key_spilled) =
            System::new_without_preprocessed(cfg_spilled, inputs.clone());
        #[cfg(not(feature = "cuda"))]
        {
            let prep_data = key_spilled.preprocessed_data.as_ref().unwrap();
            assert!(prep_data.is_spilled(), "preprocessed LDEs must be spilled");
            assert_eq!(prep_data.materialized_spilled_count(), 0);
        }
        let wit_mat_spilled = SystemWitness::from_stage_1(traces.clone(), &sys_base);
        let proof_mat_spilled =
            sys_spilled.prove_multiple_claims(&key_spilled, &claim_slices, wit_mat_spilled);
        assert_eq!(
            proof_mat_spilled.to_bytes().unwrap(),
            expected_bytes,
            "case {case_idx}: spilled materialized proof bytes must match"
        );

        // 5. Full disk-spilled + PreparedWitness::from_stage_1_lazy (mixing Host + Generated).
        let mixed_sources: Vec<TraceSource<Val>> = traces
            .iter()
            .enumerate()
            .map(|(idx, mat)| {
                #[cfg(not(feature = "cuda"))]
                if idx == 2 {
                    return TraceSource::Generated(Arc::new(MatrixGenerator {
                        matrix: mat.clone(),
                    }));
                }
                let _ = idx;
                TraceSource::Host(mat.clone())
            })
            .collect();
        let prep_wit = PreparedWitness::from_stage_1_lazy(mixed_sources, &sys_spilled);
        assert!(prep_wit.lookups.iter().all(|l| !l.is_materialized()));
        let stage_1 = sys_spilled.prove_stage_1(prep_wit);
        #[cfg(not(feature = "cuda"))]
        {
            assert!(stage_1.stage_1_trace_data.is_spilled());
            assert_eq!(stage_1.stage_1_trace_data.materialized_spilled_count(), 0);
        }
        let proof_prep_spilled = sys_spilled.prove_committed(&key_spilled, &claim_slices, stage_1);
        assert_eq!(
            proof_prep_spilled.to_bytes().unwrap(),
            expected_bytes,
            "case {case_idx}: spilled PreparedWitness::from_stage_1_lazy proof bytes must match"
        );

        // 6. Full disk-spilled + streaming iterator (`prove_multiple_claims_streaming`).
        let proof_streamed = sys_spilled.prove_multiple_claims_streaming(
            &key_spilled,
            &claim_slices,
            traces.clone(),
        );
        sys_base
            .verify_multiple_claims(&claim_slices, &proof_streamed)
            .expect("baseline verifier must accept disk-spilled streamed proof");
        assert_eq!(
            proof_streamed.to_bytes().unwrap(),
            expected_bytes,
            "case {case_idx}: spilled streaming proof bytes must match"
        );

        // 7. Manual `Stage1 { ... }` construction with shape-only lookups (matching Ix's
        //    `proof-compression/src/pipeline/prover.rs` pattern) on spilled config.
        let pcs = sys_spilled.config.pcs();
        let active: Vec<bool> = traces.iter().map(|t| t.height() > 0).collect();
        let active_indices: Vec<usize> = active
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| a.then_some(i))
            .collect();
        let mut log_degrees = Vec::new();
        let evals: Vec<_> = active_indices
            .iter()
            .map(|&ci| {
                let mat = traces[ci].clone();
                let h = mat.height();
                log_degrees.push(h.ilog2() as usize);
                (pcs.natural_domain_for_degree(h), mat)
            })
            .collect();
        let (s1_commit, s1_data) = pcs.commit(evals);
        let shape_lookups: Vec<LookupValues<Val>> = active_indices
            .iter()
            .map(|&ci| {
                let slot_widths: Vec<usize> = sys_spilled.circuits[ci]
                    .graph
                    .lookups
                    .iter()
                    .map(|l| l.args.len())
                    .collect();
                LookupValues::shape_only(traces[ci].height(), &slot_widths)
            })
            .collect();
        let manual_stage_1 = Stage1 {
            active,
            active_indices,
            log_degrees,
            stage_1_trace_commit: s1_commit,
            stage_1_trace_data: s1_data,
            lookups: shape_lookups,
        };
        let proof_manual = sys_spilled.prove_committed(&key_spilled, &claim_slices, manual_stage_1);
        assert_eq!(
            proof_manual.to_bytes().unwrap(),
            expected_bytes,
            "case {case_idx}: manual Stage1 with shape_only lookups must match baseline"
        );

        #[cfg(not(feature = "cuda"))]
        {
            let prep_data = key_spilled.preprocessed_data.as_ref().unwrap();
            assert_eq!(
                prep_data.materialized_spilled_count(),
                0,
                "case {case_idx}: spilled preprocessed LDEs must never materialize into RAM"
            );
        }

        // Drop prover key and verify zero lingering files in spill_dir.
        drop(key_spilled);
        drop(sys_spilled);
        #[cfg(not(feature = "cuda"))]
        assert_eq!(
            fs::read_dir(&spill_dir).unwrap().count(),
            0,
            "case {case_idx}: spill_dir must contain zero leftover files after drop"
        );
        let _ = fs::remove_dir_all(&spill_dir);

        // Round-trip serialization check.
        let decoded = Proof::<GoldilocksBlake3Config>::from_bytes(&expected_bytes).unwrap();
        sys_base
            .verify_multiple_claims(&claim_slices, &decoded)
            .unwrap();
    }
}

/// Verifies multi-shard batch proving (`prove_batch`, `Retention::Retain`,
/// `Retention::Regenerate`, `batch_round_one`/`batch_round_two`, cross-shard
/// byte table sharing, sparse activation, `BatchMessage`s, and positive PoW)
/// produces 100% byte-identical `BatchProof::to_bytes()` under lazy lookups and
/// disk spilling.
#[test]
fn streaming_and_spilled_batches_are_byte_identical() {
    let work_rows = 8usize;
    let inputs = multi_circuit_inputs(2, work_rows);

    // Build 3 shards whose work circuits ALL push into shard 0's byte table!
    // Shards 1 and 2 deactivate the byte table (sparse preprocessed activation).
    // Shard 2 also exercises a BatchMessage balance pair with the preamble.
    let mut byte_counts = vec![0u32; 256];
    let mut shard_work_traces = Vec::new();
    let mut shard_claims: Vec<Vec<Vec<Val>>> = Vec::new();

    let work_rows_u32 = u32::try_from(work_rows).unwrap();
    for shard in 0..3u32 {
        let mut work_vals = Vec::with_capacity(work_rows * 6);
        let mut claims = Vec::with_capacity(work_rows);
        for r in 0..work_rows_u32 {
            let a = shard * 20 + r;
            let a_next = shard * 20 + ((r + 1) % work_rows_u32);
            let b = (shard * 31 + r * 7 + 5) % 256;
            let c = (shard * 47 + r * 11 + 19) % 256;
            let d = (shard * 59 + r * 13 + 37) % 256;
            let tag = r + 1;
            let tag_next = ((r + 1) % work_rows_u32) + 1;
            let sum = a + b + c + d + tag;
            byte_counts[a as usize] += 1;
            byte_counts[b as usize] += 1;
            byte_counts[c as usize] += 1;
            byte_counts[d as usize] += 1;
            work_vals.extend_from_slice(&[f(a), f(b), f(c), f(d), f(sum), Val::ONE]);
            claims.push(vec![f(30), f(tag), f(tag_next), f(sum), f(a_next)]);
        }
        shard_work_traces.push(RowMajorMatrix::new(work_vals, 6));
        shard_claims.push(claims);
    }

    // Add one extra byte lookup pulled by shard 0's byte table and pushed by a
    // batch message, plus a matching push/pull pair on channel 99.
    byte_counts[77] += 1;
    let batch_messages = vec![
        BatchMessage::push(vec![f(10), f(77)]),
        BatchMessage::push(vec![f(99), f(1234)]),
        BatchMessage::pull(vec![f(99), f(1234)]),
    ];

    let byte_trace_full = RowMajorMatrix::new(byte_counts.into_iter().map(f).collect(), 1);
    let empty_byte = RowMajorMatrix::new(vec![], 1);
    let empty_aux = RowMajorMatrix::new(vec![], 1);
    let no_lookup_trace = RowMajorMatrix::new(
        (0..4u32)
            .flat_map(|v| [f(v + 1), f((v + 1) * (v + 1))])
            .collect(),
        2,
    );

    let shard_traces = |shard: usize| -> Vec<RowMajorMatrix<Val>> {
        let bt = if shard == 0 {
            byte_trace_full.clone()
        } else {
            empty_byte.clone()
        };
        let nl = if shard == 1 {
            RowMajorMatrix::new(vec![], 2)
        } else {
            no_lookup_trace.clone()
        };
        vec![bt, empty_aux.clone(), shard_work_traces[shard].clone(), nl]
    };

    // Baseline in-memory materialized batch.
    let batch_cap_height = if cfg!(feature = "cuda") { 0 } else { 1 };
    let cfg_base = make_config(2, batch_cap_height, 1, 3);
    let (sys_base, key_base) = System::new(cfg_base, inputs.clone());
    let base_shards: Vec<ShardInput<GoldilocksBlake3Config>> = (0..3)
        .map(|shard| ShardInput {
            claims: shard_claims[shard].clone(),
            witness: SystemWitness::from_stage_1(shard_traces(shard), &sys_base),
        })
        .collect();
    let batch_base = sys_base.prove_batch(&key_base, base_shards, batch_messages.clone());
    sys_base
        .verify_batch(&batch_base)
        .expect("baseline batch must verify");
    let expected_bytes = batch_base.to_bytes().unwrap();
    #[cfg(not(feature = "cuda"))]
    assert_eq!(
        blake3::hash(&expected_bytes).to_hex().as_str(),
        "d293d541b35ec6bbfa5e64383cbd0559fa9b341578a710a063d9b7fb42eace53",
        "batch proof bytes drifted from fc4336b baseline ({} bytes)",
        expected_bytes.len()
    );

    // Disk-spilled + lazy lookups under BOTH Retention::Retain and Retention::Regenerate.
    let spill_dir = fresh_spill_dir("batch");
    let cfg_spilled = make_config(2, batch_cap_height, 1, 3).with_disk_spill(&spill_dir);
    let (sys_spilled, key_spilled) = System::new_without_preprocessed(cfg_spilled, inputs.clone());

    // Retention::Retain with SystemWitness::from_stage_1_lazy.
    let lazy_shards: Vec<ShardInput<GoldilocksBlake3Config>> = (0..3)
        .map(|shard| {
            let wit = SystemWitness::from_stage_1_lazy(shard_traces(shard), &sys_spilled);
            assert!(wit.lookups.iter().all(|l| !l.is_materialized()));
            ShardInput {
                claims: shard_claims[shard].clone(),
                witness: wit,
            }
        })
        .collect();
    let batch_retain = sys_spilled.prove_batch(&key_spilled, lazy_shards, batch_messages.clone());
    sys_base
        .verify_batch(&batch_retain)
        .expect("baseline verifier must accept spilled Retain batch");
    assert_eq!(batch_retain.to_bytes().unwrap(), expected_bytes);

    // Retention::Regenerate with PreparedWitness::from_stage_1_lazy.
    let mut builds = 0usize;
    let batch_regen = sys_spilled.prove_batch_with(
        &key_spilled,
        &shard_claims,
        batch_messages.clone(),
        Retention::Regenerate,
        |shard| {
            builds += 1;
            let sources = shard_traces(shard)
                .into_iter()
                .map(TraceSource::Host)
                .collect();
            PreparedWitness::from_stage_1_lazy(sources, &sys_spilled)
        },
    );
    assert_eq!(builds, 6, "3 shards × 2 rounds under Retention::Regenerate");
    assert_eq!(batch_regen.to_bytes().unwrap(), expected_bytes);

    // Explicit batch_round_one + batch_round_two with streaming iterator.
    let barrier = sys_spilled.batch_round_one(
        (0..3).map(|shard| {
            let wit = SystemWitness::from_stage_1_lazy(shard_traces(shard), &sys_spilled);
            (shard_claims[shard].clone(), wit)
        }),
        Retention::Regenerate,
    );
    assert_eq!(barrier.shards(), 3);
    let batch_two_step =
        sys_spilled.batch_round_two(&key_spilled, barrier, batch_messages, |shard| {
            SystemWitness::from_stage_1_lazy(shard_traces(shard), &sys_spilled)
        });
    assert_eq!(batch_two_step.to_bytes().unwrap(), expected_bytes);

    // Round-trip decode & verify.
    let decoded = BatchProof::<GoldilocksBlake3Config>::from_bytes(&expected_bytes).unwrap();
    sys_base.verify_batch(&decoded).unwrap();

    #[cfg(not(feature = "cuda"))]
    {
        let prep_data = key_spilled.preprocessed_data.as_ref().unwrap();
        assert_eq!(prep_data.materialized_spilled_count(), 0);
    }

    drop(key_spilled);
    drop(sys_spilled);
    #[cfg(not(feature = "cuda"))]
    assert_eq!(
        fs::read_dir(&spill_dir).unwrap().count(),
        0,
        "batch spill_dir must contain zero leftover files after drop"
    );
    let _ = fs::remove_dir_all(&spill_dir);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn spill_files_are_used_and_cleaned_up_on_drop_and_panic() {
    use multi_stark::p3_adapter::spill::{active_spill_files, total_spilled_bytes};

    let spill_dir = fresh_spill_dir("cleanup-panic");
    let bytes_before = total_spilled_bytes();

    let inputs = multi_circuit_inputs(2, 16);
    let (traces, claims_owned) = build_traces_and_claims(16, 11, false, true);
    let claim_slices: Vec<&[Val]> = claims_owned.iter().map(Vec::as_slice).collect();

    // Verify spill creation, non-zero spilled bytes, and cleanup on normal drop.
    {
        let cfg = make_config(2, 0, 1, 0).with_disk_spill(&spill_dir);
        let (sys, key) = System::new_without_preprocessed(cfg, inputs.clone());
        let stage_1 = sys.prove_stage_1_streaming(traces.clone());
        assert!(active_spill_files() > 0, "spill files must be active");
        assert!(
            total_spilled_bytes() > bytes_before,
            "spilling must write LDE bytes to temporary files"
        );
        let proof = sys.prove_committed(&key, &claim_slices, stage_1);
        sys.verify_multiple_claims(&claim_slices, &proof).unwrap();
    }
    assert_eq!(
        fs::read_dir(&spill_dir).unwrap().count(),
        0,
        "normal drop must leave zero files in spill_dir"
    );

    // Verify cleanup even when unwinding from a panic while ProverKey + Stage1
    // hold open spill files.
    let panic_dir = fresh_spill_dir("panic-unwind");
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let cfg = make_config(2, 0, 1, 0).with_disk_spill(&panic_dir);
        let (sys, key) = System::new_without_preprocessed(cfg, inputs);
        let stage_1 = sys.prove_stage_1_streaming(traces);
        assert!(key.preprocessed_data.as_ref().unwrap().is_spilled());
        assert!(stage_1.stage_1_trace_data.is_spilled());
        panic!("simulated failure during proving");
    }));
    assert!(panicked.is_err());
    assert_eq!(
        fs::read_dir(&panic_dir).unwrap().count(),
        0,
        "panic unwind must leave zero files in spill_dir"
    );

    // Verify `.without_disk_spill()` disables spilling cleanly.
    let cfg_disabled = make_config(1, 0, 1, 0)
        .with_disk_spill(&spill_dir)
        .without_disk_spill();
    assert!(cfg_disabled.spill_dir().is_none());

    let _ = fs::remove_dir_all(&spill_dir);
    let _ = fs::remove_dir_all(&panic_dir);
}

#[test]
#[should_panic(expected = "preprocessed trace height")]
fn lazy_witness_rejects_mismatched_preprocessed_height() {
    let inputs = multi_circuit_inputs(1, 8);
    let cfg = make_config(1, 0, 1, 0);
    let (sys, _key) = System::new_without_preprocessed(cfg, inputs);
    // Byte table expects height 256, supply height 16 instead.
    let bad_traces = vec![
        RowMajorMatrix::new(vec![Val::ZERO; 16], 1),
        RowMajorMatrix::new(vec![], 1),
        RowMajorMatrix::new(vec![], 6),
        RowMajorMatrix::new(vec![], 2),
    ];
    let _ = SystemWitness::from_stage_1_lazy(bad_traces, &sys);
}

#[test]
#[should_panic(expected = "cannot prove with every circuit deactivated")]
fn streaming_stage_1_rejects_all_empty_traces() {
    let inputs = multi_circuit_inputs(1, 8);
    let cfg = make_config(1, 0, 1, 0);
    let (sys, _key) = System::new_without_preprocessed(cfg, inputs);
    let empty_traces = vec![
        RowMajorMatrix::new(vec![], 1),
        RowMajorMatrix::new(vec![], 1),
        RowMajorMatrix::new(vec![], 6),
        RowMajorMatrix::new(vec![], 2),
    ];
    let _ = sys.prove_stage_1_streaming(empty_traces);
}

/// Exercises multi-chunk stage-2 evaluation (`height = 8192 > chunk_rows = 4096`)
/// with next-row lookup wraparound crossing chunk boundaries (row 4095 -> 4096
/// and row 8191 -> 0) and grouped lookups (`k = 2`), verifying 100% byte-identical
/// proofs across baseline, lazy in-memory, and disk-spilled paths.
#[test]
fn multi_chunk_stage_2_crosses_chunk_boundaries_byte_identically() {
    let work_rows = 8192usize;
    let work_rows_u32 = u32::try_from(work_rows).unwrap();
    let prep = RowMajorMatrix::new((0..work_rows_u32).map(|r| f(r + 1)).collect(), 1);
    let a = Expr::main(0);
    let a_next = Expr::main_next(0);
    let b = Expr::main(1);
    let tag = Expr::preprocessed(0);
    let tag_next = Expr::preprocessed_next(0);

    // Self-canceling push/pull lookups per row that read BOTH current and next
    // main & preprocessed rows, crossing the 4096-row stage-2 chunk boundary!
    let circuit = CircuitInputs {
        main_width: 2,
        preprocessed: Some(prep),
        constraints: vec![
            Expr::IsTransition * (a_next.clone() - (a.clone() + Expr::constant(Val::ONE))),
        ],
        ext_constraints: vec![],
        lookups: vec![
            Lookup::push(
                Expr::constant(Val::ONE),
                vec![tag.clone(), a.clone(), b.clone()],
            ),
            Lookup::pull(
                Expr::constant(Val::ONE),
                vec![tag_next.clone(), a_next.clone(), b.clone()],
            ),
            Lookup::push(Expr::constant(Val::ONE), vec![tag_next, a_next, b.clone()]),
            Lookup::pull(Expr::constant(Val::ONE), vec![tag, a, b]),
        ],
        lookup_group_size: 2,
    };

    let trace = RowMajorMatrix::new(
        (0..work_rows_u32)
            .flat_map(|r| [f(r + 10), f((r * 17 + 3) % 1000)])
            .collect(),
        2,
    );
    let no_claims: &[&[Val]] = &[];

    let (sys_base, key_base) = System::new(make_config(2, 0, 1, 0), [circuit.clone()]);
    let wit_base = SystemWitness::from_stage_1(vec![trace.clone()], &sys_base);
    let proof_base = sys_base.prove_multiple_claims(&key_base, no_claims, wit_base);
    sys_base
        .verify_multiple_claims(no_claims, &proof_base)
        .unwrap();
    let expected_bytes = proof_base.to_bytes().unwrap();

    let wit_lazy = SystemWitness::from_stage_1_lazy(vec![trace.clone()], &sys_base);
    let proof_lazy = sys_base.prove_multiple_claims(&key_base, no_claims, wit_lazy);
    assert_eq!(proof_lazy.to_bytes().unwrap(), expected_bytes);

    let spill_dir = fresh_spill_dir("multi-chunk");
    let (sys_spilled, key_spilled) = System::new_without_preprocessed(
        make_config(2, 0, 1, 0).with_disk_spill(&spill_dir),
        [circuit],
    );
    let proof_spilled =
        sys_spilled.prove_multiple_claims_streaming(&key_spilled, no_claims, [trace]);
    assert_eq!(proof_spilled.to_bytes().unwrap(), expected_bytes);
    #[cfg(not(feature = "cuda"))]
    assert_eq!(
        key_spilled
            .preprocessed_data
            .as_ref()
            .unwrap()
            .materialized_spilled_count(),
        0
    );
    drop(key_spilled);
    drop(sys_spilled);
    #[cfg(not(feature = "cuda"))]
    assert_eq!(fs::read_dir(&spill_dir).unwrap().count(), 0);
    let _ = fs::remove_dir_all(&spill_dir);
}
