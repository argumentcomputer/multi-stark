//! Multi-circuit PLONK-style verifier.
//!
//! Mirrors `prover.rs`: replays the Fiat-Shamir sponge, deserialises every
//! commitment / opening proof, runs SonicKZG10 `check` for each opening
//! point, then for each circuit recomputes the composition polynomial at ζ
//! from the opened values and verifies
//!
//! ```text
//!   composition(ζ)  ==  Z_H(ζ) · quotient(ζ)
//! ```

use ark_crypto_primitives::sponge::{CryptographicSponge, FieldElementSize};
use ark_ff::{Field, One, PrimeField, Zero};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};

use crate::air::Air;
use crate::builder::folder::VerifierConstraintFolder;
use crate::lookup::fingerprint;
use crate::prover::Proof;
use crate::system::System;
use crate::types::{Commitment, PC, PlonkConfig, Sponge, Val};
use crate::{ensure, ensure_eq};

#[derive(Debug)]
pub enum VerificationError {
    InvalidProofShape,
    InvalidSystem,
    UnbalancedChannel,
    InvalidOpeningArgument,
    OodEvaluationMismatch,
}

impl<A> System<A>
where
    A: crate::air::BaseAir + for<'a> Air<VerifierConstraintFolder<'a>>,
{
    pub fn verify(
        &self,
        config: &PlonkConfig,
        claim: &[Val],
        proof: &Proof,
    ) -> Result<(), VerificationError> {
        self.verify_multiple_claims(config, &[claim], proof)
    }

    pub fn verify_multiple_claims(
        &self,
        config: &PlonkConfig,
        claims: &[&[Val]],
        proof: &Proof,
    ) -> Result<(), VerificationError> {
        ensure!(!self.circuits.is_empty(), VerificationError::InvalidSystem);
        ensure_eq!(
            proof.intermediate_accumulators.len(),
            self.circuits.len(),
            VerificationError::InvalidProofShape
        );
        ensure_eq!(
            proof.log_degrees.len(),
            self.circuits.len(),
            VerificationError::InvalidProofShape
        );
        // Lookup balance.
        ensure_eq!(
            proof.intermediate_accumulators.last().copied(),
            Some(Val::zero()),
            VerificationError::UnbalancedChannel
        );
        ensure_eq!(
            proof.proofs_at_zeta_omega.len(),
            self.circuits.len(),
            VerificationError::InvalidProofShape
        );

        // Re-label raw commitments using the same scheme the prover uses.
        let stage_1_commitments = relabel_stage_1(self, &proof.stage_1_commitments);
        let stage_2_commitments = relabel_stage_2(self, &proof.stage_2_commitments);
        let quotient_commitments = relabel_quotient(self, &proof.quotient_commitments);

        // ── replay sponge ────────────────────────────────────────────────────
        let mut sponge = config.make_sponge();
        for c in &self.preprocessed_commitments {
            absorb_commitment(&mut sponge, c.commitment());
        }
        for c in &stage_1_commitments {
            absorb_commitment(&mut sponge, c.commitment());
        }
        for d in &proof.log_degrees {
            sponge.absorb(&Val::from(u64::from(*d)));
        }
        for claim in claims {
            sponge.absorb(claim);
        }
        let lookup_challenge = squeeze_field(&mut sponge);
        sponge.absorb(&lookup_challenge);
        let fingerprint_challenge = squeeze_field(&mut sponge);
        sponge.absorb(&fingerprint_challenge);
        for c in &stage_2_commitments {
            absorb_commitment(&mut sponge, c.commitment());
        }
        let alpha = squeeze_field(&mut sponge);
        for c in &quotient_commitments {
            absorb_commitment(&mut sponge, c.commitment());
        }
        let zeta = squeeze_field(&mut sponge);

        // ── compute initial accumulator from claims ──────────────────────────
        let mut acc = Val::zero();
        for claim in claims {
            let message =
                lookup_challenge + fingerprint(&fingerprint_challenge, claim.iter().copied());
            acc += message.inverse().expect("zero claim message");
        }

        // ── shape checks for opened-value vectors ────────────────────────────
        let n_prep_polys = self.preprocessed_commitments.len();
        let n_s1_polys: usize = self.circuits.iter().map(|c| c.stage_1_width).sum();
        let n_s2_polys: usize = self.circuits.iter().map(|c| c.stage_2_width).sum();
        let n_q_polys = self.circuits.len();
        ensure_eq!(
            stage_1_commitments.len(),
            n_s1_polys,
            VerificationError::InvalidProofShape
        );
        ensure_eq!(
            stage_2_commitments.len(),
            n_s2_polys,
            VerificationError::InvalidProofShape
        );
        ensure_eq!(
            quotient_commitments.len(),
            n_q_polys,
            VerificationError::InvalidProofShape
        );
        let total_polys = n_prep_polys + n_s1_polys + n_s2_polys + n_q_polys;
        ensure_eq!(
            proof.values_at_zeta.len(),
            total_polys,
            VerificationError::InvalidProofShape
        );

        // ── verify single-point opens at zeta ────────────────────────────────
        let all_commitments: Vec<&LabeledCommitment<Commitment>> = self
            .preprocessed_commitments
            .iter()
            .chain(stage_1_commitments.iter())
            .chain(stage_2_commitments.iter())
            .chain(quotient_commitments.iter())
            .collect();
        let opens: Vec<Val> = proof.values_at_zeta.clone();

        let ok = PC::check(
            &config.verifier_key,
            all_commitments.iter().copied(),
            &zeta,
            opens,
            &proof.proof_at_zeta,
            &mut sponge,
            None,
        )
        .map_err(|_e| VerificationError::InvalidOpeningArgument)?;
        ensure!(ok, VerificationError::InvalidOpeningArgument);

        // ── verify per-circuit opens at zeta·omega ───────────────────────────
        // values_at_zeta_omega is laid out per circuit in the same order as
        // pushed by the prover: prep | stage1 | stage2.
        let mut zw_idx = 0;
        for (i, circuit) in self.circuits.iter().enumerate() {
            let n = 1usize << proof.log_degrees[i];
            let trace_domain = GeneralEvaluationDomain::<Val>::new(n)
                .ok_or(VerificationError::InvalidProofShape)?;
            let zeta_omega = zeta * trace_domain.group_gen();

            let mut commits_for_circuit: Vec<&LabeledCommitment<Commitment>> = vec![];
            let mut values_for_circuit: Vec<Val> = vec![];
            // preprocessed slice
            if let Some(prep_idx) = self.preprocessed_indices[i] {
                let start = preprocessed_poly_offset(self, prep_idx);
                let end = start + circuit.preprocessed_width;
                for c in &self.preprocessed_commitments[start..end] {
                    commits_for_circuit.push(c);
                    values_for_circuit.push(proof.values_at_zeta_omega[zw_idx]);
                    zw_idx += 1;
                }
            }
            // stage1 slice
            let s1_start: usize = self.circuits[..i].iter().map(|c| c.stage_1_width).sum();
            for c in &stage_1_commitments[s1_start..s1_start + circuit.stage_1_width] {
                commits_for_circuit.push(c);
                values_for_circuit.push(proof.values_at_zeta_omega[zw_idx]);
                zw_idx += 1;
            }
            // stage2 slice
            let s2_start: usize = self.circuits[..i].iter().map(|c| c.stage_2_width).sum();
            for c in &stage_2_commitments[s2_start..s2_start + circuit.stage_2_width] {
                commits_for_circuit.push(c);
                values_for_circuit.push(proof.values_at_zeta_omega[zw_idx]);
                zw_idx += 1;
            }

            let ok = PC::check(
                &config.verifier_key,
                commits_for_circuit.iter().copied(),
                &zeta_omega,
                values_for_circuit,
                &proof.proofs_at_zeta_omega[i],
                &mut sponge,
                None,
            )
            .map_err(|_e| VerificationError::InvalidOpeningArgument)?;
            ensure!(ok, VerificationError::InvalidOpeningArgument);
        }
        ensure_eq!(
            zw_idx,
            proof.values_at_zeta_omega.len(),
            VerificationError::InvalidProofShape
        );

        // ── per-circuit OOD check ────────────────────────────────────────────
        let mut acc_input = acc;
        // Ranges are identical to the prover's "all_polys" layout:
        // [preprocessed total | stage1 total | stage2 total | quotient total]
        let stage1_base = n_prep_polys;
        let stage2_base = stage1_base + n_s1_polys;
        let quotient_base = stage2_base + n_s2_polys;

        // First, walk preprocessed circuit-by-circuit to consume the prep slice
        // of values_at_zeta.
        // We re-index using the preprocessed_indices map.
        let mut zeta_omega_idx = 0usize;
        for (i, circuit) in self.circuits.iter().enumerate() {
            let n = 1usize << proof.log_degrees[i];
            let trace_domain = GeneralEvaluationDomain::<Val>::new(n)
                .ok_or(VerificationError::InvalidProofShape)?;
            let next_acc = proof.intermediate_accumulators[i];

            // Slice the opened values for this circuit.
            // Preprocessed locals/nexts.
            let prep_local: Vec<Val>;
            let prep_next: Vec<Val>;
            if let Some(prep_idx) = self.preprocessed_indices[i] {
                let start = preprocessed_poly_offset(self, prep_idx);
                let w = circuit.preprocessed_width;
                prep_local = proof.values_at_zeta[start..start + w].to_vec();
                prep_next = proof.values_at_zeta_omega[zeta_omega_idx..zeta_omega_idx + w].to_vec();
                zeta_omega_idx += w;
            } else {
                prep_local = vec![];
                prep_next = vec![];
            }
            // Stage1 locals/nexts.
            let s1_start: usize = self.circuits[..i].iter().map(|c| c.stage_1_width).sum();
            let s1_w = circuit.stage_1_width;
            let s1_local = proof.values_at_zeta
                [stage1_base + s1_start..stage1_base + s1_start + s1_w]
                .to_vec();
            let s1_next =
                proof.values_at_zeta_omega[zeta_omega_idx..zeta_omega_idx + s1_w].to_vec();
            zeta_omega_idx += s1_w;
            // Stage2 locals/nexts.
            let s2_start: usize = self.circuits[..i].iter().map(|c| c.stage_2_width).sum();
            let s2_w = circuit.stage_2_width;
            let s2_local = proof.values_at_zeta
                [stage2_base + s2_start..stage2_base + s2_start + s2_w]
                .to_vec();
            let s2_next =
                proof.values_at_zeta_omega[zeta_omega_idx..zeta_omega_idx + s2_w].to_vec();
            zeta_omega_idx += s2_w;

            // Stage 2 publics.
            let stage_2_publics = [lookup_challenge, fingerprint_challenge, acc_input, next_acc];
            acc_input = next_acc;

            // Selectors at zeta.
            let omega = trace_domain.group_gen();
            let omega_inv = omega.inverse().unwrap();
            let zeta_n = zeta.pow([n as u64]);
            let zh = zeta_n - Val::one();
            let n_field = Val::from(n as u64);
            let l0 = zh * ((zeta - Val::one()) * n_field).inverse().unwrap();
            let l_nm1 = zh * omega_inv * ((zeta - omega_inv) * n_field).inverse().unwrap();
            let trans = Val::one() - l_nm1;

            let mut folder = VerifierConstraintFolder {
                preprocessed_local: &prep_local,
                preprocessed_next: &prep_next,
                stage_1_local: &s1_local,
                stage_1_next: &s1_next,
                stage_2_local: &s2_local,
                stage_2_next: &s2_next,
                stage_2_public_values: &stage_2_publics,
                is_first_row: l0,
                is_last_row: l_nm1,
                is_transition: trans,
                alpha,
                accumulator: Val::zero(),
            };
            circuit.air.eval(&mut folder);
            let composition = folder.accumulator;

            // Quotient value at zeta (one per circuit).
            let q_value = proof.values_at_zeta[quotient_base + i];

            // composition(ζ) == Z_H(ζ) · quotient(ζ)
            ensure_eq!(
                composition,
                zh * q_value,
                VerificationError::OodEvaluationMismatch
            );
        }

        Ok(())
    }
}

fn preprocessed_poly_offset<A>(system: &System<A>, prep_idx: usize) -> usize {
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

fn squeeze_field(sponge: &mut Sponge) -> Val {
    let bits_needed = (Val::MODULUS_BIT_SIZE as usize).saturating_sub(1);
    let v: Vec<Val> =
        sponge.squeeze_field_elements_with_sizes(&[FieldElementSize::Truncated(bits_needed)]);
    v[0]
}

fn absorb_commitment(sponge: &mut Sponge, c: &Commitment) {
    let mut bytes = vec![];
    use ark_serialize::CanonicalSerialize;
    c.serialize_uncompressed(&mut bytes).expect("serialize");
    sponge.absorb(&bytes);
}

fn relabel_stage_1<A>(
    system: &System<A>,
    raw: &[Commitment],
) -> Vec<LabeledCommitment<Commitment>> {
    let mut out = vec![];
    let mut idx = 0;
    for (i, c) in system.circuits.iter().enumerate() {
        for col_idx in 0..c.stage_1_width {
            out.push(LabeledCommitment::new(
                format!("stage1_c{i}_col{col_idx}"),
                raw[idx],
                None,
            ));
            idx += 1;
        }
    }
    out
}

fn relabel_stage_2<A>(
    system: &System<A>,
    raw: &[Commitment],
) -> Vec<LabeledCommitment<Commitment>> {
    let mut out = vec![];
    let mut idx = 0;
    for (i, c) in system.circuits.iter().enumerate() {
        for col_idx in 0..c.stage_2_width {
            out.push(LabeledCommitment::new(
                format!("stage2_c{i}_col{col_idx}"),
                raw[idx],
                None,
            ));
            idx += 1;
        }
    }
    out
}

fn relabel_quotient<A>(
    _system: &System<A>,
    raw: &[Commitment],
) -> Vec<LabeledCommitment<Commitment>> {
    raw.iter()
        .enumerate()
        .map(|(i, c)| LabeledCommitment::new(format!("quotient_c{i}"), *c, None))
        .collect()
}
