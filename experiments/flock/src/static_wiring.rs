//! Open the fixed wiring polynomial against a verifier-owned commitment.
use crate::counts as op_count;
use flock_prover::{
    challenger::{Challenger, FsChallenger},
    circuit::{Circuit, SigmaAssertion},
    field::F128,
    merkle::HashKind,
    pcs::{
        self, BatchOpeningProofLigerito, Commitment, DirectEqInd, PackedDirectClaim,
        PackedDirectClaimRef, PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k},
    },
    product_gkr::{LiveMask, s_id_basis},
    zerocheck::PaddingSpec,
};
use std::time::Instant;

struct Key {
    commitment: Commitment,
    circuit_digest: Vec<u8>,
    mask: LiveMask,
    mu: usize,
    base_bits: usize,
    pins: Vec<(usize, usize)>,
}
impl Key {
    fn point(&self, sigma: &SigmaAssertion) -> Vec<F128> {
        let mut point = sigma.rho.clone();
        point.resize(self.commitment.params.m - 7, F128::ZERO);
        point
    }
    fn transcript(&self, sigma: &SigmaAssertion) -> FsChallenger {
        let mut ch = FsChallenger::new(b"multi-stark-flock-fixed-wiring-v1");
        ch.observe_bytes(&self.circuit_digest);
        ch.observe_bytes(&bincode::serialize(&self.commitment).unwrap());
        ch.observe_bytes(&bincode::serialize(sigma).unwrap());
        ch
    }
    fn verify(&self, sigma: &SigmaAssertion, proof: &BatchOpeningProofLigerito) -> bool {
        if sigma.rho.len() != self.mu
            || sigma.nu != self.mask.nu
            || sigma.base_bits != self.base_bits
            || sigma.element_constants.is_some()
            || sigma.boolean_pins.len() != self.pins.len()
        {
            return false;
        }
        if self.mask.live_eval(&sigma.rho) != sigma.live_value
            || self.mask.masked_id_eval(&s_id_basis(self.mu), &sigma.rho) != sigma.masked_id_value
        {
            return false;
        }
        for ((index, point, value), &(expected, count)) in sigma.boolean_pins.iter().zip(&self.pins)
        {
            if *index != expected
                || point.len() != self.mask.nu
                || LiveMask::eq_prefix_sum(point, count) != *value
            {
                return false;
            }
        }
        let point = self.point(sigma);
        let params = &self.commitment.params;
        pcs::verify_opening_batch_ligerito_mixed_with_grinding(
            &self.commitment,
            &[],
            &[],
            &[],
            &[PackedDirectClaimRef {
                point: &point,
                value: sigma.value,
            }],
            proof,
            &params.ligerito_verifier_config().unwrap(),
            params.opening_grinding(),
            &mut self.transcript(sigma),
        )
        .is_ok()
    }
}

pub fn measure(
    circuit: &Circuit,
    sigma: &SigmaAssertion,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if circuit.registry().num_element() != 0 {
        return Err("fixed wiring experiment supports Boolean tables only".into());
    }
    let start = Instant::now();
    let mu = circuit.cells().mu();
    let m = (mu + 7).max(22);
    let profile = LigeritoProfile::Fast;
    let params = PcsParams {
        m,
        profile,
        log_inv_rate: profile.log_inv_rate(),
        log_batch_size: embedded_initial_k(m, profile)
            .ok_or("missing strict wiring PCS profile")?,
        num_lanes: None,
        merkle_hash: HashKind::Blake3,
    };
    let mask = circuit.live_mask();
    let mut table = vec![F128::ZERO; 1 << (m - 7)];
    for (i, (&destination, value)) in circuit.sigma().iter().zip(&mut table).enumerate() {
        if mask.is_live(i) {
            *value = F128::new(destination as u64, 0);
        }
    }
    let (commitment, data) = pcs::commit(&table, &params);
    let pins = circuit
        .registry()
        .boolean_types()
        .iter()
        .enumerate()
        .filter(|(_, ty)| ty.const_pin.is_some())
        .map(|(i, _)| (i, circuit.counts()[i]))
        .collect();
    let key = Key {
        commitment,
        circuit_digest: circuit.digest().to_vec(),
        mask,
        mu,
        base_bits: (mu - circuit.cells().nu())
            .max(circuit.registry().num_boolean().next_power_of_two().ilog2() as usize),
        pins,
    };
    let setup_seconds = start.elapsed().as_secs_f64();
    let point = key.point(sigma);
    let start = Instant::now();
    let proof = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
        table,
        &data,
        &key.commitment,
        &[],
        &[],
        &[PackedDirectClaim {
            point: point.clone(),
            value: sigma.value,
            eq_ind: DirectEqInd::EqPoint(point),
        }],
        &PaddingSpec::dense(m),
        &params.ligerito_prover_config()?,
        params.opening_grinding(),
        &mut key.transcript(sigma),
    );
    let prove_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let (valid, ops) = op_count::measure(|| key.verify(sigma, &proof));
    assert!(valid);
    let verify_seconds = start.elapsed().as_secs_f64();
    let encoded = bincode::serialize(&proof)?;
    let decoded = bincode::deserialize(&encoded)?;
    assert!(key.verify(sigma, &decoded));
    let mut mutations = Vec::new();
    let mut bad = sigma.clone();
    bad.value += F128::ONE;
    mutations.push(bad);
    let mut bad = sigma.clone();
    bad.live_value += F128::ONE;
    mutations.push(bad);
    let mut bad = sigma.clone();
    bad.masked_id_value += F128::ONE;
    mutations.push(bad);
    let mut bad = sigma.clone();
    bad.rho[0] += F128::ONE;
    mutations.push(bad);
    let mut bad = sigma.clone();
    bad.base_bits += 1;
    mutations.push(bad);
    let mut bad = sigma.clone();
    bad.boolean_pins.clear();
    mutations.push(bad);
    if !sigma.boolean_pins.is_empty() {
        let mut bad = sigma.clone();
        bad.boolean_pins[0].2 += F128::ONE;
        mutations.push(bad);
    }
    for bad in &mutations {
        assert!(!key.verify(bad, &proof));
    }
    // The key is derived from the circuit, never supplied by the proof.
    let mut other_key = key;
    other_key.circuit_digest[0] ^= 1;
    assert!(!other_key.verify(sigma, &proof));
    Ok(serde_json::json!({
        "scope": "Fixed wiring discharge only; not a Groth16 verifier or complete deferred discharge",
        "table_field_elements": 1usize << (m-7), "profile": profile.as_str(),
        "setup_seconds": setup_seconds, "prove_seconds": prove_seconds,
        "verify_seconds": verify_seconds, "proof_bytes": encoded.len(),
        "verifier_f128_multiplications": ops.muls_excluding_inv(), "verifier_f128_inversions": ops.invs,
        "verified": true, "serialized_roundtrip_verified": true,
        "mutations_rejected": mutations.len()+1,
    }))
}
