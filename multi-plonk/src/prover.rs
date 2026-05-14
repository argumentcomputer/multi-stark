//! Multi-circuit PLONK-style prover with KZG commitments.
//!
//! Pipeline (mirrors `multi-stark` but in a single field):
//!
//! 1. **Stage 1**: iFFT each trace column → monomial polynomials `T_i(X)`.
//!    Commit them all as one labeled batch via SonicKZG10.
//! 2. **Lookup challenges (β, γ)**: sampled from the Fiat-Shamir sponge after
//!    observing claims and stage-1 commitments.
//! 3. **Stage 2**: build the LogUp accumulator + per-row message inverses,
//!    iFFT each column, commit as a second labeled batch.
//! 4. **Constraint challenge α** + **quotient**: evaluate constraints on a
//!    coset domain of size `next_pow2(max_constraint_degree) * trace_size`,
//!    fold via powers of α, divide by Z_H. iFFT result back to a single
//!    quotient polynomial (no chunking) and commit it.
//! 5. **OOD opening**: sample ζ. Open every committed polynomial at ζ; for
//!    polynomials with row-rotation constraints (stage 1, stage 2,
//!    preprocessed) also open at ζ·ω_i where ω_i is the trace-domain
//!    generator.
//!
//! Each (commit, open) call uses the SAME Poseidon sponge that drives
//! Fiat-Shamir, so every pcs operation is bound to the transcript.

use ark_ff::{FftField, Field, One, PrimeField, Zero};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain, univariate::DensePolynomial};
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial, PolynomialCommitment};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;

use ark_crypto_primitives::sponge::{CryptographicSponge, FieldElementSize};

use crate::air::Air;
use crate::builder::folder::ProverConstraintFolder;
use crate::lookup::{Lookup, fingerprint};
use crate::system::{ProverKey, System, SystemWitness, ifft_column};
use crate::types::{
    Commitment, CommitmentState, OpeningProof, PC, PlonkConfig, Sponge, UniPoly, Val,
};

/// Layout of opened values:
///
/// `values_at_zeta` (length = #preprocessed + #stage1 + #stage2 + #quotient)
/// is concatenated in fixed order: all preprocessed columns (ordered as the
/// preprocessed bundle), then all stage-1 columns (concatenated per circuit),
/// then all stage-2 columns, then one quotient value per circuit.
///
/// `values_at_zeta_omega` (length = sum over circuits of preprocessed_width +
/// stage_1_width + stage_2_width) is concatenated per circuit: preprocessed |
/// stage1 | stage2.
#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct Proof {
    pub stage_1_commitments: Vec<Commitment>,
    pub stage_2_commitments: Vec<Commitment>,
    pub quotient_commitments: Vec<Commitment>,
    pub log_degrees: Vec<u8>,
    pub intermediate_accumulators: Vec<Val>,
    pub values_at_zeta: Vec<Val>,
    pub values_at_zeta_omega: Vec<Val>,
    pub proof_at_zeta: OpeningProof,
    pub proofs_at_zeta_omega: Vec<OpeningProof>,
}

impl<A> System<A>
where
    A: crate::air::BaseAir
        + Air<crate::builder::symbolic::SymbolicAirBuilder>
        + for<'a> Air<ProverConstraintFolder<'a>>,
{
    pub fn prove(
        &self,
        config: &PlonkConfig,
        key: &ProverKey,
        claim: &[Val],
        witness: &SystemWitness,
    ) -> Proof {
        self.prove_multiple_claims(config, key, &[claim], witness)
    }

    pub fn prove_multiple_claims(
        &self,
        config: &PlonkConfig,
        key: &ProverKey,
        claims: &[&[Val]],
        witness: &SystemWitness,
    ) -> Proof {
        let mut sponge = config.make_sponge();

        // ── Stage 1: commit trace columns ────────────────────────────────────
        let mut stage_1_polys: Vec<LabeledPolynomial<Val, UniPoly>> = vec![];
        let mut stage_1_widths: Vec<usize> = vec![];
        let mut log_degrees: Vec<u8> = vec![];
        for (i, trace) in witness.traces.iter().enumerate() {
            let n = trace.height();
            log_degrees.push(u8::try_from(log2(n)).expect("log2(trace_height) fits u8"));
            stage_1_widths.push(trace.width());
            for (col_idx, col) in trace.columns().into_iter().enumerate() {
                let poly = ifft_column(&col);
                let label = format!("stage1_c{i}_col{col_idx}");
                stage_1_polys.push(LabeledPolynomial::new(label, poly, None, None));
            }
        }
        let (stage_1_commitments, stage_1_states) =
            PC::commit(&config.committer_key, &stage_1_polys, None).expect("stage 1 commit failed");

        // observe preprocessed + stage 1 commitments + degrees + claims
        for c in &self.preprocessed_commitments {
            absorb_commitment(&mut sponge, c.commitment());
        }
        for c in &stage_1_commitments {
            absorb_commitment(&mut sponge, c.commitment());
        }
        for d in &log_degrees {
            sponge.absorb(&Val::from(u64::from(*d)));
        }
        for claim in claims {
            sponge.absorb(claim);
        }

        // ── Lookup challenges ────────────────────────────────────────────────
        let lookup_challenge = squeeze_field(&mut sponge);
        sponge.absorb(&lookup_challenge);
        let fingerprint_challenge = squeeze_field(&mut sponge);
        sponge.absorb(&fingerprint_challenge);

        // initial accumulator from claims
        let mut acc = Val::zero();
        for claim in claims {
            let message =
                lookup_challenge + fingerprint(&fingerprint_challenge, claim.iter().copied());
            acc += message.inverse().expect("zero claim message");
        }

        // ── Stage 2: lookup traces ───────────────────────────────────────────
        let (stage_2_traces, intermediate_accumulators) = Lookup::stage_2_traces(
            &witness.lookups,
            lookup_challenge,
            &fingerprint_challenge,
            acc,
        );
        let mut stage_2_polys: Vec<LabeledPolynomial<Val, UniPoly>> = vec![];
        let mut stage_2_widths: Vec<usize> = vec![];
        for (i, trace) in stage_2_traces.iter().enumerate() {
            stage_2_widths.push(trace.width());
            for (col_idx, col) in trace.columns().into_iter().enumerate() {
                let poly = ifft_column(&col);
                let label = format!("stage2_c{i}_col{col_idx}");
                stage_2_polys.push(LabeledPolynomial::new(label, poly, None, None));
            }
        }
        let (stage_2_commitments, stage_2_states) =
            PC::commit(&config.committer_key, &stage_2_polys, None).expect("stage 2 commit failed");
        for c in &stage_2_commitments {
            absorb_commitment(&mut sponge, c.commitment());
        }

        // ── Constraint challenge α + quotient ────────────────────────────────
        let alpha = squeeze_field(&mut sponge);

        // For each circuit, evaluate all constraints on a coset domain large
        // enough to recover the composition polynomial, divide by Z_H, iFFT
        // the result back to a single quotient polynomial.
        let mut quotient_polys: Vec<LabeledPolynomial<Val, UniPoly>> = vec![];
        let mut acc_input = acc;
        for (i, circuit) in self.circuits.iter().enumerate() {
            let n = 1usize << log_degrees[i];
            let q_factor = next_pow2(circuit.max_constraint_degree.max(2));
            let coset_size = q_factor * n;
            let coset = GeneralEvaluationDomain::<Val>::new(coset_size)
                .expect("coset size unsupported by FFT")
                .get_coset(Val::GENERATOR)
                .expect("could not build coset");

            let preprocessed_trace_polys: Vec<UniPoly> =
                if let Some(prep_idx) = self.preprocessed_indices[i] {
                    polys_for_circuit(
                        &key.preprocessed_polys,
                        prep_idx_to_poly_offset(self, prep_idx),
                        circuit.preprocessed_width,
                    )
                } else {
                    vec![]
                };
            let stage_1_trace_polys: Vec<UniPoly> =
                polys_for_circuit_stage1(&stage_1_polys, &stage_1_widths, i);
            let stage_2_trace_polys: Vec<UniPoly> =
                polys_for_circuit_stage2(&stage_2_polys, &stage_2_widths, i);

            // Evaluate every column on the coset.
            let preprocessed_evals: Vec<Vec<Val>> = preprocessed_trace_polys
                .iter()
                .map(|p| coset.fft(&p.coeffs))
                .collect();
            let stage_1_evals: Vec<Vec<Val>> = stage_1_trace_polys
                .iter()
                .map(|p| coset.fft(&p.coeffs))
                .collect();
            let stage_2_evals: Vec<Vec<Val>> = stage_2_trace_polys
                .iter()
                .map(|p| coset.fft(&p.coeffs))
                .collect();

            // Selectors and Z_H on the coset.
            let zh_inv = vanishing_inv_on_coset(coset_size, n);
            let (is_first_row, is_last_row, is_transition) = row_selectors_on_coset(&coset, n);

            // Stage-2 public values.
            let next_acc = intermediate_accumulators[i];
            let stage_2_publics = [lookup_challenge, fingerprint_challenge, acc_input, next_acc];
            acc_input = next_acc; // for next circuit iteration

            // Powers of alpha for this circuit's constraints — reversed so
            // that the i-th constraint is multiplied by α^{k-1-i}, matching
            // the verifier's Horner scheme `acc = acc * α + x`.
            let mut alpha_powers = Vec::with_capacity(circuit.constraint_count);
            let mut p = Val::one();
            for _ in 0..circuit.constraint_count {
                alpha_powers.push(p);
                p *= alpha;
            }
            alpha_powers.reverse();

            // Step between local and "next" row in the coset (= q_factor).
            let next_step = q_factor;

            // Evaluate composition row-by-row on the coset.
            let mut composition_evals = Vec::with_capacity(coset_size);
            for k in 0..coset_size {
                let k_next = (k + next_step) % coset_size;
                let preprocessed_local: Vec<Val> =
                    preprocessed_evals.iter().map(|e| e[k]).collect();
                let preprocessed_next: Vec<Val> =
                    preprocessed_evals.iter().map(|e| e[k_next]).collect();
                let stage_1_local: Vec<Val> = stage_1_evals.iter().map(|e| e[k]).collect();
                let stage_1_next: Vec<Val> = stage_1_evals.iter().map(|e| e[k_next]).collect();
                let stage_2_local: Vec<Val> = stage_2_evals.iter().map(|e| e[k]).collect();
                let stage_2_next: Vec<Val> = stage_2_evals.iter().map(|e| e[k_next]).collect();

                let mut folder = ProverConstraintFolder {
                    preprocessed_local: &preprocessed_local,
                    preprocessed_next: &preprocessed_next,
                    stage_1_local: &stage_1_local,
                    stage_1_next: &stage_1_next,
                    stage_2_local: &stage_2_local,
                    stage_2_next: &stage_2_next,
                    stage_2_public_values: &stage_2_publics,
                    is_first_row: is_first_row[k],
                    is_last_row: is_last_row[k],
                    is_transition: is_transition[k],
                    alpha_powers: &alpha_powers,
                    accumulator: Val::zero(),
                    constraint_index: 0,
                };
                circuit.air.eval(&mut folder);
                composition_evals.push(folder.accumulator * zh_inv[k]);
            }

            // iFFT composition_evals (already on coset) → coefficient form
            // (this divides out the coset offset implicitly via coset.ifft).
            let q_coeffs = coset.ifft(&composition_evals);
            let mut q_poly = DensePolynomial { coeffs: q_coeffs };
            // trim trailing zeros — real quotient has degree ≤ (q_factor-1)·n
            while q_poly.coeffs.last().is_some_and(|c| c.is_zero()) {
                q_poly.coeffs.pop();
            }
            let label = format!("quotient_c{i}");
            quotient_polys.push(LabeledPolynomial::new(label, q_poly, None, None));
        }
        let (quotient_commitments, quotient_states) =
            PC::commit(&config.committer_key, &quotient_polys, None)
                .expect("quotient commit failed");
        for c in &quotient_commitments {
            absorb_commitment(&mut sponge, c.commitment());
        }

        // ── OOD opening at ζ and ζ·ω_i ───────────────────────────────────────
        let zeta = squeeze_field(&mut sponge);

        // Build a unified list of (poly, state, commitment) entries so we can
        // batch-open per point in a single SonicKZG10 call.
        // Order per circuit: preprocessed | stage1 | stage2 | quotient
        // (preprocessed = the slice of `key.preprocessed_polys` for this circuit)
        let mut all_polys: Vec<&LabeledPolynomial<Val, UniPoly>> = vec![];
        let mut all_states: Vec<&CommitmentState> = vec![];
        let mut all_commitments: Vec<&LabeledCommitment<Commitment>> = vec![];
        // Track per-circuit ranges into all_*.
        let mut circuit_ranges: Vec<CircuitRanges> = vec![];

        for (i, circuit) in self.circuits.iter().enumerate() {
            let r = CircuitRanges {
                prep: self.preprocessed_indices[i].map(|p| {
                    let start = poly_offset_for_prep(self, p);
                    start..start + circuit.preprocessed_width
                }),
                stage_1: {
                    let start = stage_1_widths[..i].iter().sum::<usize>();
                    start..start + stage_1_widths[i]
                },
                stage_2: {
                    let start = stage_2_widths[..i].iter().sum::<usize>();
                    start..start + stage_2_widths[i]
                },
                quotient: i..i + 1,
            };
            circuit_ranges.push(r);
        }

        // Push into all_* for each batch (preprocessed first, stage1, stage2,
        // quotient). Order matters for verifier reconstruction.
        for (poly, state, comm) in zip3(
            &key.preprocessed_polys,
            &key.preprocessed_states,
            &self.preprocessed_commitments,
        ) {
            all_polys.push(poly);
            all_states.push(state);
            all_commitments.push(comm);
        }
        for (poly, state, comm) in zip3(&stage_1_polys, &stage_1_states, &stage_1_commitments) {
            all_polys.push(poly);
            all_states.push(state);
            all_commitments.push(comm);
        }
        for (poly, state, comm) in zip3(&stage_2_polys, &stage_2_states, &stage_2_commitments) {
            all_polys.push(poly);
            all_states.push(state);
            all_commitments.push(comm);
        }
        for (poly, state, comm) in zip3(&quotient_polys, &quotient_states, &quotient_commitments) {
            all_polys.push(poly);
            all_states.push(state);
            all_commitments.push(comm);
        }

        // Section bases inside `all_*` (stage1 = after all preprocessed).
        let n_prep = key.preprocessed_polys.len();
        let n_s1 = stage_1_polys.len();
        let stage_1_base = n_prep;
        let stage_2_base = stage_1_base + n_s1;

        // Open at zeta — batches every poly across every circuit.
        let proof_at_zeta = PC::open(
            &config.committer_key,
            all_polys.iter().copied(),
            all_commitments.iter().copied(),
            &zeta,
            &mut sponge,
            all_states.iter().copied(),
            None,
        )
        .expect("open at zeta failed");

        // Compute the openings (evaluations) at zeta + zeta*omega.
        let mut values_at_zeta = Vec::with_capacity(all_polys.len());
        for poly in &all_polys {
            values_at_zeta.push(eval_poly(poly.polynomial(), zeta));
        }

        // Per-circuit open at zeta*omega — only the trace polys (preprocessed,
        // stage1, stage2). Quotient does NOT need rotation.
        let mut values_at_zeta_omega: Vec<Val> = Vec::new();
        let mut proofs_at_zeta_omega: Vec<OpeningProof> = Vec::new();
        for (i, _circuit) in self.circuits.iter().enumerate() {
            let n = 1usize << log_degrees[i];
            let trace_domain = GeneralEvaluationDomain::<Val>::new(n).unwrap();
            let zeta_omega = zeta * trace_domain.group_gen();

            // collect this circuit's trace polys
            let mut circ_polys: Vec<&LabeledPolynomial<Val, UniPoly>> = vec![];
            let mut circ_states: Vec<&CommitmentState> = vec![];
            let mut circ_commits: Vec<&LabeledCommitment<Commitment>> = vec![];
            if let Some(prep) = &circuit_ranges[i].prep {
                for j in prep.clone() {
                    circ_polys.push(all_polys[j]);
                    circ_states.push(all_states[j]);
                    circ_commits.push(all_commitments[j]);
                    values_at_zeta_omega.push(eval_poly(all_polys[j].polynomial(), zeta_omega));
                }
            }
            for j in circuit_ranges[i].stage_1.clone().map(|x| stage_1_base + x) {
                circ_polys.push(all_polys[j]);
                circ_states.push(all_states[j]);
                circ_commits.push(all_commitments[j]);
                values_at_zeta_omega.push(eval_poly(all_polys[j].polynomial(), zeta_omega));
            }
            for j in circuit_ranges[i].stage_2.clone().map(|x| stage_2_base + x) {
                circ_polys.push(all_polys[j]);
                circ_states.push(all_states[j]);
                circ_commits.push(all_commitments[j]);
                values_at_zeta_omega.push(eval_poly(all_polys[j].polynomial(), zeta_omega));
            }

            let proof_zw = PC::open(
                &config.committer_key,
                circ_polys.iter().copied(),
                circ_commits.iter().copied(),
                &zeta_omega,
                &mut sponge,
                circ_states.iter().copied(),
                None,
            )
            .expect("open at zeta*omega failed");
            proofs_at_zeta_omega.push(proof_zw);
        }

        let strip = |labeled: &[LabeledCommitment<Commitment>]| -> Vec<Commitment> {
            labeled.iter().map(|lc| *lc.commitment()).collect()
        };

        Proof {
            stage_1_commitments: strip(&stage_1_commitments),
            stage_2_commitments: strip(&stage_2_commitments),
            quotient_commitments: strip(&quotient_commitments),
            log_degrees,
            intermediate_accumulators,
            values_at_zeta,
            values_at_zeta_omega,
            proof_at_zeta,
            proofs_at_zeta_omega,
        }
    }
}

/// Per-circuit ranges into the global `all_polys` lists. `prep` is `None` if
/// the circuit has no preprocessed trace.
struct CircuitRanges {
    prep: Option<std::ops::Range<usize>>,
    stage_1: std::ops::Range<usize>,
    stage_2: std::ops::Range<usize>,
    #[allow(dead_code)]
    quotient: std::ops::Range<usize>,
}

fn polys_for_circuit(
    polys: &[LabeledPolynomial<Val, UniPoly>],
    start: usize,
    width: usize,
) -> Vec<UniPoly> {
    polys[start..start + width]
        .iter()
        .map(|lp| lp.polynomial().clone())
        .collect()
}

fn polys_for_circuit_stage1(
    polys: &[LabeledPolynomial<Val, UniPoly>],
    widths: &[usize],
    circuit_idx: usize,
) -> Vec<UniPoly> {
    let start = widths[..circuit_idx].iter().sum();
    let width = widths[circuit_idx];
    polys[start..start + width]
        .iter()
        .map(|lp| lp.polynomial().clone())
        .collect()
}

fn polys_for_circuit_stage2(
    polys: &[LabeledPolynomial<Val, UniPoly>],
    widths: &[usize],
    circuit_idx: usize,
) -> Vec<UniPoly> {
    let start = widths[..circuit_idx].iter().sum();
    let width = widths[circuit_idx];
    polys[start..start + width]
        .iter()
        .map(|lp| lp.polynomial().clone())
        .collect()
}

fn prep_idx_to_poly_offset<A>(system: &System<A>, prep_idx: usize) -> usize {
    // Sum the preprocessed widths of the circuits that have a preprocessed
    // bundle index < prep_idx.
    let mut off = 0;
    for (c_idx, slot) in system.preprocessed_indices.iter().enumerate() {
        if let Some(p) = slot
            && *p < prep_idx
        {
            off += system.circuits[c_idx].preprocessed_width;
        }
    }
    off
}

fn poly_offset_for_prep<A>(system: &System<A>, prep_idx: usize) -> usize {
    prep_idx_to_poly_offset(system, prep_idx)
}

fn next_pow2(n: usize) -> usize {
    n.next_power_of_two()
}

/// Z_H(x) = x^n - 1; returns 1/Z_H at every coset point.
fn vanishing_inv_on_coset(coset_size: usize, n: usize) -> Vec<Val> {
    let coset = GeneralEvaluationDomain::<Val>::new(coset_size)
        .expect("coset size unsupported")
        .get_coset(Val::GENERATOR)
        .expect("get_coset failed");
    coset
        .elements()
        .map(|x| {
            let v = x.pow([n as u64]) - Val::one();
            v.inverse().expect("Z_H zero on coset (coset overlaps H?)")
        })
        .collect()
}

/// Per-coset-point row selectors for a trace of size `n`.
/// `is_first_row(x) = L_0(x) = (x^n - 1) / (n * (x - 1))` evaluated on coset.
/// We compute them via a vanishing-poly based formula consistent with the
/// shape used by multi-stark.
fn row_selectors_on_coset(
    coset: &GeneralEvaluationDomain<Val>,
    n: usize,
) -> (Vec<Val>, Vec<Val>, Vec<Val>) {
    let n_inv = Val::from(n as u64).inverse().unwrap();
    let omega = GeneralEvaluationDomain::<Val>::new(n).unwrap().group_gen();
    let omega_inv = omega.inverse().unwrap();
    let mut is_first = Vec::with_capacity(coset.size());
    let mut is_last = Vec::with_capacity(coset.size());
    let mut is_trans = Vec::with_capacity(coset.size());
    for x in coset.elements() {
        let xn_minus_one = x.pow([n as u64]) - Val::one();
        // L_i(x) = ω^i * (x^n - 1) / (n * (x - ω^i))
        //
        // L_0     : ω^0 = 1, so the leading factor is 1
        // L_{n-1} : ω^{n-1} = ω^{-1}
        let denom_first = (x - Val::one()) * Val::from(n as u64);
        let l0 = xn_minus_one * denom_first.inverse().unwrap();

        let omega_nm1 = omega_inv;
        let denom_last = (x - omega_nm1) * Val::from(n as u64);
        let lnm1 = xn_minus_one * omega_nm1 * denom_last.inverse().unwrap();

        let trans = Val::one() - lnm1;
        let _ = n_inv;
        is_first.push(l0);
        is_last.push(lnm1);
        is_trans.push(trans);
    }
    (is_first, is_last, is_trans)
}

/// Squeeze a single field element from the sponge.
fn squeeze_field(sponge: &mut Sponge) -> Val {
    let bits_needed = (Val::MODULUS_BIT_SIZE as usize).saturating_sub(1);
    let v: Vec<Val> =
        sponge.squeeze_field_elements_with_sizes(&[FieldElementSize::Truncated(bits_needed)]);
    v[0]
}

/// Absorb a Sonic commitment (kzg10::Commitment) into the sponge.
fn absorb_commitment(sponge: &mut Sponge, c: &Commitment) {
    // Serialize the commitment to bytes and absorb.
    let mut bytes = vec![];
    c.serialize_uncompressed(&mut bytes).expect("serialize");
    sponge.absorb(&bytes);
}

fn eval_poly(p: &UniPoly, x: Val) -> Val {
    // Horner.
    let mut acc = Val::zero();
    for c in p.coeffs.iter().rev() {
        acc = acc * x + *c;
    }
    acc
}

fn zip3<'a, A, B, C>(
    a: &'a [A],
    b: &'a [B],
    c: &'a [C],
) -> impl Iterator<Item = (&'a A, &'a B, &'a C)> {
    a.iter()
        .zip(b.iter())
        .zip(c.iter())
        .map(|((x, y), z)| (x, y, z))
}
