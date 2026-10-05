//! Batch proving with regenerated preprocessing and witnesses. Setup retains
//! verifier metadata and commitments; polynomial data lives for one shard.

use p3_matrix::{Matrix, dense::RowMajorMatrix};

use super::{KzgCommitment, KzgConfig, Scalar, pcs::KzgProverData};
use crate::{
    batch::{BatchPreamble, BatchProof, ShardHeader},
    config::ProofConfig,
    lookup::LookupValues,
    prover::Proof,
    system::{Circuit, CircuitInputs, ProverKey, System, SystemWitness},
    traits::Pcs,
};

pub struct ShardedKzg {
    pub system: System<KzgConfig>,
}

impl ShardedKzg {
    /// Inputs are trusted circuit definitions, supplied one at a time.
    pub fn new(config: KzgConfig, inputs: impl IntoIterator<Item = CircuitInputs<Scalar>>) -> Self {
        let mut circuits = Vec::new();
        let mut indices = Vec::new();
        let mut commitment = KzgCommitment(vec![], vec![]);
        for input in inputs {
            let (mut local, _key) = System::new(config.clone(), [input]);
            let mut circuit = local.circuits.remove(0);
            circuit.preprocessed = None;
            circuits.push(circuit);
            if let Some(mut c) = local.preprocessed_commit {
                indices.push(Some(commitment.0.len()));
                commitment.0.append(&mut c.0);
                commitment.1.append(&mut c.1);
            } else {
                indices.push(None);
            }
        }
        Self {
            system: System {
                config,
                circuits,
                preprocessed_indices: indices,
                preprocessed_commit: (!commitment.0.is_empty()).then_some(commitment),
            },
        }
    }

    /// The schedule assigns every circuit to exactly one shard. All claims
    /// are fixed by the caller; additional batch balance messages are disallowed.
    /// Sources must reproduce the setup and round-one commitments exactly.
    pub fn prove(
        &mut self,
        claims: &[Vec<Vec<Scalar>>],
        schedule: &[Vec<usize>],
        mut fixed: impl FnMut(usize) -> CircuitInputs<Scalar>,
        mut witness: impl FnMut(usize) -> Vec<RowMajorMatrix<Scalar>>,
    ) -> Result<BatchProof<KzgConfig>, &'static str> {
        self.check_schedule(claims, schedule)?;
        let result = self.prove_inner(claims, schedule, &mut fixed, &mut witness);
        self.clear_fixed();
        result
    }

    fn prove_inner(
        &mut self,
        claims: &[Vec<Vec<Scalar>>],
        schedule: &[Vec<usize>],
        fixed: &mut impl FnMut(usize) -> CircuitInputs<Scalar>,
        witness: &mut impl FnMut(usize) -> Vec<RowMajorMatrix<Scalar>>,
    ) -> Result<BatchProof<KzgConfig>, &'static str> {
        let mut headers = Vec::new();
        for (shard, active) in schedule.iter().enumerate() {
            headers.push(self.commit_shard(&claims[shard], active, witness(shard))?);
        }
        let preamble = BatchPreamble {
            headers,
            messages: vec![],
        };
        let mut proofs = Vec::new();
        for shard in 0..schedule.len() {
            proofs.push(self.prove_shard(&preamble, shard, &mut *fixed, witness(shard))?);
        }
        Ok(BatchProof { preamble, proofs })
    }

    /// Round one needs only main commitments. The caller may persist this header;
    /// no lookup payload or regenerated preprocessing is retained here.
    pub fn commit_shard(
        &self,
        claims: &[Vec<Scalar>],
        active: &[usize],
        traces: Vec<RowMajorMatrix<Scalar>>,
    ) -> Result<ShardHeader<KzgConfig>, &'static str> {
        if active.is_empty() || active.iter().any(|&i| i >= self.system.circuits.len()) {
            return Err("invalid active circuits");
        }
        self.check_traces(&traces, active)?;
        let lookups = traces
            .iter()
            .zip(&self.system.circuits)
            .map(|(trace, circuit)| {
                let widths: Vec<_> = circuit.graph.lookups.iter().map(|l| l.args.len()).collect();
                LookupValues::shape_only(trace.height(), &widths)
            })
            .collect();
        let stage = self.system.prove_stage_1(SystemWitness { traces, lookups });
        let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
        Ok(stage.header(&refs))
    }

    /// Resume one shard against a complete, fixed preamble. Call `verify` on
    /// the complete batch with independently expected claims and schedule.
    pub fn prove_shard(
        &mut self,
        preamble: &BatchPreamble<KzgConfig>,
        shard: usize,
        mut fixed: impl FnMut(usize) -> CircuitInputs<Scalar>,
        traces: Vec<RowMajorMatrix<Scalar>>,
    ) -> Result<Proof<KzgConfig>, &'static str> {
        let header = preamble.headers.get(shard).ok_or("invalid shard index")?;
        if header.active.len() != self.system.circuits.len() || !preamble.messages.is_empty() {
            return Err("invalid preamble");
        }
        let active: Vec<_> = header
            .active
            .iter()
            .enumerate()
            .filter_map(|(i, &on)| on.then_some(i))
            .collect();
        if active.is_empty() {
            return Err("empty shard");
        }
        let result = (|| {
            self.check_traces(&traces, &active)?;
            let key = self.load_fixed(&active, &mut fixed)?;
            let stage = self
                .system
                .prove_stage_1(SystemWitness::from_stage_1(traces, &self.system));
            let refs: Vec<_> = header.claims.iter().map(Vec::as_slice).collect();
            if stage.header(&refs) != *header {
                return Err("regenerated witness differs from round one");
            }
            Ok(self
                .system
                .prove_batch_shard(&key, stage, &refs, preamble, shard))
        })();
        self.clear_fixed();
        result
    }

    pub fn verify(
        &self,
        proof: &BatchProof<KzgConfig>,
        expected_claims: &[Vec<Vec<Scalar>>],
        schedule: &[Vec<usize>],
    ) -> Result<(), &'static str> {
        self.check_schedule(expected_claims, schedule)?;
        if proof.preamble.headers.len() != schedule.len() || !proof.preamble.messages.is_empty() {
            return Err("unexpected batch shape or balance messages");
        }
        for (shard, header) in proof.preamble.headers.iter().enumerate() {
            let expected: Vec<_> = (0..self.system.circuits.len())
                .map(|i| schedule[shard].contains(&i))
                .collect();
            if header.active != expected || header.claims != expected_claims[shard] {
                return Err("batch differs from expected statement or schedule");
            }
        }
        self.system
            .verify_batch(proof)
            .map_err(|_e| "invalid KZG batch proof")
    }

    fn check_schedule(
        &self,
        claims: &[Vec<Vec<Scalar>>],
        schedule: &[Vec<usize>],
    ) -> Result<(), &'static str> {
        if schedule.is_empty() || schedule.len() != claims.len() {
            return Err("invalid shard count");
        }
        let mut seen = vec![false; self.system.circuits.len()];
        for shard in schedule {
            if shard.is_empty() {
                return Err("empty shard");
            }
            for &i in shard {
                let slot = seen.get_mut(i).ok_or("invalid circuit index")?;
                if *slot {
                    return Err("repeated circuit");
                }
                *slot = true;
            }
        }
        if seen.iter().any(|&b| !b) {
            return Err("missing circuit");
        }
        Ok(())
    }

    fn check_traces(
        &self,
        traces: &[RowMajorMatrix<Scalar>],
        active: &[usize],
    ) -> Result<(), &'static str> {
        if traces.len() != self.system.circuits.len() {
            return Err("invalid trace count");
        }
        for (i, (trace, circuit)) in traces.iter().zip(&self.system.circuits).enumerate() {
            if trace.width() != circuit.main_width
                || (trace.height() > 0) != active.contains(&i)
                || (trace.height() > 0
                    && circuit.preprocessed_width > 0
                    && trace.height() != circuit.preprocessed_height)
            {
                return Err("trace differs from shard schedule or fixed height");
            }
        }
        Ok(())
    }

    fn clear_fixed(&mut self) {
        for circuit in &mut self.system.circuits {
            circuit.preprocessed = None;
        }
    }

    fn load_fixed(
        &mut self,
        active: &[usize],
        fixed: &mut impl FnMut(usize) -> CircuitInputs<Scalar>,
    ) -> Result<ProverKey<KzgConfig>, &'static str> {
        self.clear_fixed();
        let domains = self
            .system
            .circuits
            .iter()
            .zip(&self.system.preprocessed_indices)
            .filter(|(_, slot)| slot.is_some())
            .map(|(c, _)| {
                self.system
                    .config
                    .pcs()
                    .natural_domain_for_degree(c.preprocessed_height)
            })
            .collect();
        let mut data = self
            .system
            .preprocessed_commit
            .clone()
            .map(|commitment| KzgProverData::sparse(commitment, domains));
        for &i in active {
            let (mut local, key) = System::new(self.system.config.clone(), [fixed(i)]);
            let mut circuit = local.circuits.remove(0);
            if !same_circuit(&self.system.circuits[i], &circuit) {
                return Err("regenerated circuit differs from setup");
            }
            if let Some(slot) = self.system.preprocessed_indices[i] {
                data.as_mut()
                    .unwrap()
                    .insert(slot, key.preprocessed_data.ok_or("missing preprocessing")?)?;
            }
            self.system.circuits[i].preprocessed = circuit.preprocessed.take();
        }
        Ok(ProverKey {
            preprocessed_data: data,
        })
    }
}

fn same_circuit(a: &Circuit<Scalar>, b: &Circuit<Scalar>) -> bool {
    a.graph == b.graph
        && a.main_width == b.main_width
        && a.preprocessed_width == b.preprocessed_width
        && a.preprocessed_height == b.preprocessed_height
        && a.num_lookups == b.num_lookups
        && a.stage_2_width == b.stage_2_width
        && a.num_publics == b.num_publics
        && a.lookup_group_size == b.lookup_group_size
        && a.constraint_count == b.constraint_count
        && a.max_constraint_degree == b.max_constraint_degree
}
