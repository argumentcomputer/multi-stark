//! Authenticate the fixed copy permutation at the wiring verifier's point.
use anyhow::{Result, ensure};
use flock_prover::{
    challenger::{Challenger, FsChallenger},
    circuit::{Circuit, SigmaAssertion},
    field::F128,
    merkle::HashKind,
    pcs::{
        self, Commitment, DirectEqInd, PackedDirectClaim, PackedDirectClaimRef, PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k},
    },
    transcript_record::RecordingChallenger,
    zerocheck::PaddingSpec,
};
use flock_terminal_exporter::{
    Stage4FlockInnerLigeritoWitnessV1, Stage4FlockTranscriptWitnessV1,
    export_packed_direct_ligerito,
};
use ix_stage4_trace::{F128AlgebraTraceV1, F128CircuitStructureMatrixIdV1};
use ix_terminal_circuit::*;
use std::{fs, path::Path, time::Instant};
const DOMAIN: &[u8] = b"multi-stark-fixed-sigma-terminal-v1";
const PHASE: ConstraintPhase = ConstraintPhase::Pcs;

pub struct FixedSigma {
    id: F128CircuitStructureMatrixIdV1,
    mu: usize,
    metadata: Vec<u8>,
    commitment: Commitment,
    topology: [u8; 32],
    transcript: Stage4FlockTranscriptWitnessV1,
    opening: Stage4FlockInnerLigeritoWitnessV1,
}

fn bind<Ch: Challenger>(ch: &mut Ch, digest: &[u8; 32], metadata: &[u8], point: &[F128]) {
    ch.observe_bytes(digest);
    ch.observe_bytes(metadata);
    ch.observe_f128_slice(point);
}
impl FixedSigma {
    pub fn prepare(
        circuit: &Circuit,
        sigma: &SigmaAssertion,
        id: F128CircuitStructureMatrixIdV1,
        out: &Path,
    ) -> Result<Self> {
        let started = Instant::now();
        let digest = circuit.digest();
        let mu = circuit.cells().mu();
        ensure!(
            id.circuit_digest == digest && id.row_variables as usize == circuit.cells().nu(),
            "sigma fixed identity"
        );
        ensure!(sigma.rho.len() == mu, "sigma point dimension");
        let m = (mu + 7).max(22);
        let profile = LigeritoProfile::Slim;
        let params = PcsParams {
            m,
            profile,
            log_inv_rate: profile.log_inv_rate(),
            log_batch_size: embedded_initial_k(m, profile)
                .ok_or_else(|| anyhow::anyhow!("missing fixed-sigma profile"))?,
            num_lanes: None,
            merkle_hash: HashKind::Blake3,
        };
        let mask = circuit.live_mask();
        let mut table = vec![F128::ZERO; 1usize << (m - 7)];
        for (i, (&destination, value)) in circuit.sigma().iter().zip(&mut table).enumerate() {
            if mask.is_live(i) {
                *value = F128::new(destination as u64, 0);
            }
        }
        let (commitment, data) = pcs::commit(&table, &params);
        let metadata = bincode::serialize(&commitment)?;
        let topology = *blake3::hash(&metadata).as_bytes();
        let mut point = sigma.rho.clone();
        point.resize(m - 7, F128::ZERO);
        let setup_seconds = started.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        bind(&mut ch, &digest, &metadata, &point);
        let proof = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
            table,
            &data,
            &commitment,
            &[],
            &[],
            &[PackedDirectClaim {
                point: point.clone(),
                value: sigma.value,
                eq_ind: DirectEqInd::EqPoint(point.clone()),
            }],
            &PaddingSpec::dense(m),
            &params
                .ligerito_prover_config()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?,
            params.opening_grinding(),
            &mut ch,
        );
        drop(data);
        let prove_seconds = start.elapsed().as_secs_f64();
        let encoded = bincode::serialize(&proof)?;
        let proof = bincode::deserialize(&encoded)?;
        let mut recording = RecordingChallenger::new(FsChallenger::with_chained_blake3(DOMAIN));
        bind(&mut recording, &digest, &metadata, &point);
        pcs::verify_opening_batch_ligerito_mixed_with_grinding(
            &commitment,
            &[],
            &[],
            &[],
            &[PackedDirectClaimRef {
                point: &point,
                value: sigma.value,
            }],
            &proof,
            &params
                .ligerito_verifier_config()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?,
            params.opening_grinding(),
            &mut recording,
        )
        .map_err(|e| anyhow::anyhow!("fixed sigma proof: {e:?}"))?;
        let mut wrong = FsChallenger::with_chained_blake3(DOMAIN);
        bind(&mut wrong, &digest, &metadata, &point);
        ensure!(
            pcs::verify_opening_batch_ligerito_mixed_with_grinding(
                &commitment,
                &[],
                &[],
                &[],
                &[PackedDirectClaimRef {
                    point: &point,
                    value: sigma.value + F128::ONE
                }],
                &proof,
                &params
                    .ligerito_verifier_config()
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?,
                params.opening_grinding(),
                &mut wrong
            )
            .is_err(),
            "altered sigma value accepted"
        );
        let opening = export_packed_direct_ligerito(
            &recording,
            &commitment,
            &proof,
            sigma.value,
            topology,
            Some(point.len() as u64),
        )?;
        ensure!(
            recording.values().get(..point.len()) == Some(point.as_slice()),
            "sigma point transcript"
        );
        let transcript = Stage4FlockTranscriptWitnessV1::from_recording_with_algebra(
            &recording,
            DOMAIN,
            F128AlgebraTraceV1::default(),
            &[],
        )?;
        fs::write(out.join("fixed-sigma-proof.bin"), &encoded)?;
        fs::write(out.join("fixed-sigma-key.bin"), &metadata)?;
        fs::write(
            out.join("fixed-sigma.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"scope":"native fixed-table opening; terminal constraints measured separately","proof_bytes":encoded.len(),"setup_seconds":setup_seconds,"prove_seconds":prove_seconds,"m":m,"verified":true,"altered_value_rejected":true,"key_blake3":blake3::hash(&metadata).to_hex().to_string()}),
            )?,
        )?;
        flock_prover::scratch::clear();
        Ok(Self {
            id,
            mu,
            metadata,
            commitment,
            topology,
            transcript,
            opening,
        })
    }

    pub fn constrain(
        &self,
        builder: &mut R1csBuilder,
        claim: &F128CircuitStructureClaimVariablesV1,
    ) -> Result<()> {
        ensure!(claim.matrix == self.id, "sigma matrix identity");
        let base = self.id.column_variables as usize - 3;
        ensure!(
            claim.row_point.len() == self.id.row_variables as usize
                && claim.column_point.len() == base + 3,
            "sigma matrix point"
        );
        let slot_bits = self.mu - claim.row_point.len();
        ensure!(slot_bits <= base, "sigma slot width");
        for (i, value) in claim.column_point[slot_bits..].iter().enumerate() {
            let expected = if i < base - slot_bits {
                0u128
            } else {
                (2u128 >> (i - (base - slot_bits))) & 1
            };
            let constant = alloc_f128_constant(builder, expected.to_le_bytes(), PHASE)?;
            enforce_f128_equal(builder, value, &constant, PHASE);
        }
        let mut point: Vec<_> = claim
            .row_point
            .iter()
            .chain(&claim.column_point[..slot_bits])
            .cloned()
            .collect();
        point.resize(
            self.commitment.params.m - 7,
            alloc_f128_constant(builder, [0; 16], PHASE)?,
        );
        let tape = constrain_chained_blake3_transcript(
            builder,
            self.transcript.chained_blake3(),
            self.transcript.observed_values(),
            self.transcript.byte_payloads(),
            self.transcript.challenges(),
        )?;
        bind_bytes(builder, &tape.byte_payloads[0], &self.id.circuit_digest)?;
        bind_bytes(builder, &tape.byte_payloads[1], &self.metadata)?;
        ensure!(
            tape.observed_values.len() > point.len(),
            "sigma transcript point length"
        );
        for (actual, expected) in tape.observed_values.iter().take(point.len()).zip(&point) {
            enforce_f128_equal(builder, actual, expected, PHASE);
        }
        let mut cap = Vec::new();
        for digest in &self.commitment.cap {
            for word in digest.as_chunks::<16>().0 {
                cap.push(F128TranscriptWordV1::from_f128_variables(
                    &alloc_f128_constant(builder, *word, PHASE)?,
                ));
            }
        }
        let frontend = F128MergedPcsFrontendCircuitOutputV1 {
            topology_digest: self.topology,
            commitment_cap: cap,
            ring_switches: vec![],
            packed_direct_claims: vec![],
            batching_challenges: vec![],
            rho: point,
            running: alloc_f128_constant(builder, [0; 16], PHASE)?,
            q_eval: claim.value.clone(),
        };
        constrain_f128_inner_ligerito(
            builder,
            self.opening.trace(),
            F128InnerLigeritoCircuitInputsV1 {
                observed_values: &tape.observed_values,
                challenges: &tape.challenges,
                byte_payloads: &tape.byte_payloads,
                private_values: self.opening.private_values(),
                private_digests: self.opening.private_digests(),
                frontend: &frontend,
            },
        )?;
        Ok(())
    }
}

fn bind_bytes(
    builder: &mut R1csBuilder,
    words: &[F128TranscriptWordV1],
    bytes: &[u8],
) -> Result<()> {
    ensure!(
        words.len() == bytes.len().div_ceil(16),
        "fixed sigma metadata length"
    );
    for (i, word) in words.iter().enumerate() {
        for (bit, expression) in word.bit_expressions().iter().enumerate() {
            let expected = bytes
                .get(i * 16 + bit / 8)
                .map_or(0, |byte| (byte >> (bit % 8)) & 1);
            builder.enforce_zero(
                PHASE,
                expression.clone().minus(&LinearCombination::from_constant(
                    ark_bls12_381::Fr::from(expected),
                )),
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    #[test]
    fn metadata_binds_every_bit_and_zero_padding() {
        let bytes = [0x93u8; 19];
        let mut builder = R1csBuilder::new();
        let mut words = Vec::new();
        let mut variables = Vec::new();
        for chunk in bytes.chunks(16) {
            let mut raw = [0u8; 16];
            raw[..chunk.len()].copy_from_slice(chunk);
            let value = alloc_f128_private(&mut builder, raw, PHASE).unwrap();
            words.push(F128TranscriptWordV1::from_f128_variables(&value));
            variables.push(value);
        }
        bind_bytes(&mut builder, &words, &bytes).unwrap();
        let (r, w) = builder.finish().unwrap();
        for value in &variables {
            for bit in value.bit_variables() {
                let mut bad = w.clone();
                bad.set(*bit, Fr::from(1u64) - w.assignment()[bit.index() as usize])
                    .unwrap();
                assert!(r.check(&bad).is_err());
            }
        }
        assert!(bind_bytes(&mut R1csBuilder::new(), &words[..1], &bytes).is_err());
    }
}
