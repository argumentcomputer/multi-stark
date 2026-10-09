//! [`ProofConfig`] instantiation for the KZG backend: BLS12-381
//! scalar field (its own challenge field, `D = 1`), Blake3 transcript,
//! monomial KZG commitments.

use std::sync::Arc;

use ark_serialize::CanonicalSerialize;

use crate::config::ProofConfig;
use crate::traits::Pcs;

use super::field::Scalar;
use super::pcs::KzgPcs;
use super::srs::Srs;
use super::transcript::Blake3Transcript;

#[derive(Clone)]
pub struct KzgConfig {
    pcs: KzgPcs,
    /// Bytes observed into every fresh challenger: a domain tag plus a
    /// digest of the protocol parameters INCLUDING the SRS (see the
    /// transcript contract on [`ProofConfig::initialise_challenger`]).
    transcript_seed: Vec<u8>,
    max_log_degree: usize,
    stream_lookups: bool,
    stream_quotient: bool,
}

impl KzgConfig {
    /// Public parameters are caller-supplied and taken on trust here:
    /// call [`Srs::validate`] first on parameters you did not generate.
    ///
    /// # Panics
    /// Panics if the SRS length is not a power of two (trace domains
    /// are, and `max_log_degree` is read off the SRS).
    pub fn new(srs: Arc<Srs>, max_quotient_degree: usize) -> Self {
        assert!(
            srs.max_len() >= 2 && srs.max_len().is_power_of_two(),
            "SRS length must be a power of two"
        );
        let max_log_degree = p3_util::log2_strict_usize(srs.max_len());
        let mut transcript_seed = b"multi-stark/kzg/v3".to_vec();
        for parameter in [max_log_degree, max_quotient_degree] {
            transcript_seed.extend(u64::try_from(parameter).unwrap().to_le_bytes());
        }
        // Bind the anchors and their τ multiples; validated powers follow.
        srs.g1[0]
            .serialize_compressed(&mut transcript_seed)
            .expect("serialization into a Vec cannot fail");
        srs.g2
            .serialize_compressed(&mut transcript_seed)
            .expect("serialization into a Vec cannot fail");
        srs.g1[1]
            .serialize_compressed(&mut transcript_seed)
            .expect("serialization into a Vec cannot fail");
        srs.tau_g2
            .serialize_compressed(&mut transcript_seed)
            .expect("serialization into a Vec cannot fail");
        Self {
            pcs: KzgPcs::new(srs, max_quotient_degree),
            transcript_seed,
            max_log_degree,
            stream_lookups: false,
            stream_quotient: false,
        }
    }

    /// Reconstruct and commit one lookup trace at a time from committed columns.
    /// This changes memory use, not the proof or its transcript.
    pub fn with_streaming_lookups(mut self) -> Self {
        self.stream_lookups = true;
        self
    }

    /// Evaluate each quotient on trace-sized cosets, bounding temporary matrices.
    pub fn with_streaming_quotient(mut self) -> Self {
        self.stream_quotient = true;
        self
    }
}

fn lookup_trace(
    circuit: &crate::system::Circuit<Scalar>,
    main: &p3_matrix::dense::RowMajorMatrix<Scalar>,
    fixed: Option<&p3_matrix::dense::RowMajorMatrix<Scalar>>,
    beta: Scalar,
    gamma: Scalar,
    tile_rows: usize,
) -> (p3_matrix::dense::RowMajorMatrix<Scalar>, Scalar) {
    use crate::traits::Algebra;
    use p3_matrix::{Matrix, dense::RowMajorMatrix};
    use p3_maybe_rayon::prelude::*;

    let width = circuit.stage_2_width;
    let tile_values = tile_rows * width;
    let wave_values = tile_values * current_num_threads();
    let mut trace = vec![Scalar::ZERO; main.height() * width];
    let mut local = Scalar::ZERO;
    // A wave has at most one tile per worker, bounding temporary lookup
    // payloads even when their message construction uses nested parallelism.
    for (wave_index, wave) in trace.chunks_mut(wave_values).enumerate() {
        let wave_start = wave_index * wave_values / width;
        let mut offsets: Vec<_> = wave
            .par_chunks_mut(tile_values)
            .enumerate()
            .map(|(tile, output)| {
                let start = wave_start + tile * tile_rows;
                let end = start + output.len() / width;
                let values =
                    crate::system::compute_lookup_values_range(circuit, main, fixed, start..end);
                let (mut chunks, next) = crate::lookup::LookupValues::stage_2_traces(
                    &[values],
                    &[circuit.lookup_group_size],
                    beta,
                    &gamma,
                    Scalar::ZERO,
                );
                output.copy_from_slice(&chunks.remove(0).values);
                next[0]
            })
            .collect();
        // Each tile starts at zero. Its exclusive prefix of previous totals
        // restores the same row/group accumulator order as a serial sweep.
        for offset in &mut offsets {
            let total = *offset;
            *offset = local;
            local += total;
        }
        wave.par_chunks_mut(tile_values)
            .zip(offsets)
            .for_each(|(output, offset)| {
                for value in output {
                    *value += offset;
                }
            });
    }
    (RowMajorMatrix::new(trace, width), local)
}

#[allow(clippy::too_many_arguments)]
fn scalar_quotient_values(
    circuit: &crate::system::Circuit<Scalar>,
    publics: &[Scalar],
    trace_domain: super::domain::Radix2Coset,
    quotient_domain: super::domain::Radix2Coset,
    fixed: Option<&p3_matrix::dense::RowMajorMatrix<Scalar>>,
    main: &p3_matrix::dense::RowMajorMatrix<Scalar>,
    stage2: &p3_matrix::dense::RowMajorMatrix<Scalar>,
    alpha: Scalar,
    constraint_count: usize,
) -> Vec<Scalar> {
    use crate::traits::{Algebra, EvaluationDomain, Field, Packed};
    use p3_matrix::{Matrix, dense::RowMajorMatrix};
    use p3_maybe_rayon::prelude::*;

    let size = quotient_domain.size();
    let step = size / trace_domain.size();
    assert_eq!(main.height(), size);
    assert_eq!(stage2.height(), size);
    assert!(fixed.is_none_or(|m| m.height() == size));
    assert_eq!(constraint_count, circuit.constraint_count());
    let started = std::time::Instant::now();
    let selectors = trace_domain.selectors_on_coset(quotient_domain);
    let selector_seconds = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let normalizer = (Scalar::from_usize(trace_domain.size()) * trace_domain.generator()).inverse();
    let delta = [(publics[3] - publics[2]) * normalizer];
    let mut weights: Vec<_> = alpha.powers().take(constraint_count).collect();
    weights.reverse();
    let row = |matrix: &RowMajorMatrix<Scalar>, index: usize| {
        let start = matrix.width * index;
        start..start + matrix.width
    };
    let mut values = vec![Scalar::ZERO; size];
    values
        .par_chunks_mut(1 << 10)
        .enumerate()
        .for_each(|(tile, output)| {
            // One scratch set per tile avoids allocating row pairs and graph
            // values for every point of the quotient domain.
            let mut nodes = Vec::with_capacity(circuit.graph.nodes.len());
            let mut constraints = Vec::with_capacity(constraint_count);
            for (offset, output) in output.iter_mut().enumerate() {
                let i = tile * (1 << 10) + offset;
                let next = (i + step) % size;
                let fixed_rows = fixed.map_or([&[][..], &[][..]], |m| {
                    [&m.values[row(m, i)], &m.values[row(m, next)]]
                });
                let view = crate::eval::VarValues {
                    preprocessed: fixed_rows,
                    main: [&main.values[row(main, i)], &main.values[row(main, next)]],
                    stage2: [
                        &stage2.values[row(stage2, i)],
                        &stage2.values[row(stage2, next)],
                    ],
                    publics,
                    is_first_row: selectors.is_first_row[i],
                    is_last_row: selectors.is_last_row[i],
                    is_transition: selectors.is_transition[i],
                };
                circuit.graph.sweep(&view, &mut nodes);
                constraints.clear();
                constraints.extend(circuit.graph.zeros.iter().map(|id| nodes[id.index()]));
                crate::lookup::logup_constraint_values(
                    &circuit.graph.lookups,
                    &nodes,
                    view.stage2[0],
                    view.stage2[1],
                    publics,
                    &delta,
                    view.is_last_row,
                    Scalar::ZERO,
                    1,
                    circuit.lookup_group_size,
                    &mut constraints,
                );
                *output = Scalar::batched_linear_combination(&constraints, &weights)
                    * selectors.inv_vanishing[i];
            }
        });
    tracing::info!(
        rows = size,
        selector_seconds,
        constraint_seconds = started.elapsed().as_secs_f64(),
        "KZG quotient constraints evaluated"
    );
    values
}

impl ProofConfig for KzgConfig {
    fn accelerated_quotient_values(
        &self,
        circuit: &crate::system::Circuit<Scalar>,
        publics: &[Scalar],
        trace_domain: super::domain::Radix2Coset,
        quotient_domain: super::domain::Radix2Coset,
        preprocessed: Option<(&super::pcs::KzgProverData, usize)>,
        stage_1: (&super::pcs::KzgProverData, usize),
        stage_2: (&super::pcs::KzgProverData, usize),
        alpha: Scalar,
        constraint_count: usize,
    ) -> Option<Vec<Scalar>> {
        let started = std::time::Instant::now();
        let fixed = preprocessed.map(|(data, index)| {
            self.pcs
                .get_evaluations_on_domain(data, index, quotient_domain)
        });
        let main = self
            .pcs
            .get_evaluations_on_domain(stage_1.0, stage_1.1, quotient_domain);
        let lookup = self
            .pcs
            .get_evaluations_on_domain(stage_2.0, stage_2.1, quotient_domain);
        tracing::info!(
            seconds = started.elapsed().as_secs_f64(),
            "KZG quotient evaluations materialized"
        );
        Some(scalar_quotient_values(
            circuit,
            publics,
            trace_domain,
            quotient_domain,
            fixed.as_ref(),
            &main,
            &lookup,
            alpha,
            constraint_count,
        ))
    }

    fn accelerated_lookup_commit(
        &self,
        inputs: &[crate::config::LookupCommitInput<'_, Self>],
        beta: Scalar,
        gamma: Scalar,
        mut acc: Scalar,
    ) -> Option<crate::config::AcceleratedLookupCommitment<Self>> {
        if !self.stream_lookups {
            return None;
        }
        let mut parts = Vec::new();
        let mut accumulators = Vec::new();
        for (i, input) in inputs.iter().enumerate() {
            let start = std::time::Instant::now();
            let domain = input.stage_1.0.matrices[input.stage_1.1].domain;
            let fixed = input
                .preprocessed
                .map(|(data, slot)| self.pcs.get_evaluations_on_domain(data, slot, domain));
            let main = self
                .pcs
                .get_evaluations_on_domain(input.stage_1.0, input.stage_1.1, domain);
            let evaluation_seconds = start.elapsed().as_secs_f64();
            let trace_started = std::time::Instant::now();
            let (trace, local) =
                lookup_trace(input.circuit, &main, fixed.as_ref(), beta, gamma, 1 << 16);
            let trace_seconds = trace_started.elapsed().as_secs_f64();
            drop(main);
            drop(fixed);
            acc += local;
            accumulators.push(acc);
            let commit_started = std::time::Instant::now();
            let (_, data) = self.pcs.commit(vec![(domain, trace)]);
            let commit_seconds = commit_started.elapsed().as_secs_f64();
            parts.push(data);
            tracing::info!(
                circuit = i,
                evaluation_seconds,
                trace_seconds,
                commit_seconds,
                seconds = start.elapsed().as_secs_f64(),
                "KZG lookup committed"
            );
        }
        let (commitment, data) = super::pcs::KzgProverData::concatenate(parts);
        Some((commitment, data, accumulators))
    }

    fn accelerated_quotient_commit(
        &self,
        inputs: &[crate::config::QuotientCommitInput<'_, Self>],
        alpha: Scalar,
    ) -> Option<(super::pcs::KzgCommitment, super::pcs::KzgProverData)> {
        if !self.stream_quotient {
            return None;
        }
        use crate::traits::{Algebra, EvaluationDomain, TwoAdicField};
        let mut parts = Vec::with_capacity(inputs.len());
        for (i, input) in inputs.iter().enumerate() {
            let start = std::time::Instant::now();
            let n = input.trace_domain.size();
            let ratio = input.quotient_domain.size() / n;
            let generator = Scalar::two_adic_generator(input.quotient_domain.log_size);
            let mut shift = input.quotient_domain.shift;
            let mut values = vec![Scalar::ZERO; input.quotient_domain.size()];
            for coset in 0..ratio {
                let lde_started = std::time::Instant::now();
                let domain = super::domain::Radix2Coset {
                    log_size: input.trace_domain.log_size,
                    shift,
                };
                let fixed = input
                    .preprocessed
                    .map(|(data, slot)| self.pcs.get_evaluations_on_domain(data, slot, domain));
                let main =
                    self.pcs
                        .get_evaluations_on_domain(input.stage_1.0, input.stage_1.1, domain);
                let stage2 =
                    self.pcs
                        .get_evaluations_on_domain(input.stage_2.0, input.stage_2.1, domain);
                let lde_seconds = lde_started.elapsed().as_secs_f64();
                let chunk = scalar_quotient_values(
                    input.circuit,
                    &input.lookup_publics,
                    input.trace_domain,
                    domain,
                    fixed.as_ref(),
                    &main,
                    &stage2,
                    alpha,
                    input.constraint_count,
                );
                for (row, value) in chunk.into_iter().enumerate() {
                    values[row * ratio + coset] = value;
                }
                shift *= generator;
                tracing::info!(
                    circuit = i,
                    coset,
                    lde_seconds,
                    seconds = start.elapsed().as_secs_f64(),
                    "KZG quotient coset evaluated"
                );
            }
            let (_, data) = self.pcs.commit_quotient(vec![(
                input.quotient_domain,
                p3_matrix::dense::RowMajorMatrix::new_col(values),
                ratio,
            )]);
            parts.push(data);
            tracing::info!(
                circuit = i,
                seconds = start.elapsed().as_secs_f64(),
                "KZG quotient committed"
            );
        }
        Some(super::pcs::KzgProverData::concatenate(parts))
    }

    fn omit_inactive_preprocessed_openings(&self) -> bool {
        true
    }
    fn omit_unused_next_row_openings(&self) -> bool {
        true
    }

    type Pcs = KzgPcs;
    type Challenge = Scalar;
    type Challenger = Blake3Transcript;

    fn pcs(&self) -> &KzgPcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Blake3Transcript {
        let mut challenger = Blake3Transcript::new();
        challenger.observe_bytes(&self.transcript_seed);
        challenger
    }

    fn max_log_degree(&self) -> usize {
        self.max_log_degree
    }

    fn max_log_quotient_domain(&self) -> usize {
        <Scalar as crate::traits::TwoAdicField>::TWO_ADICITY
    }

    fn max_quotient_degree(&self) -> usize {
        self.pcs.max_quotient_degree()
    }

    fn log_blowup(&self) -> usize {
        // KZG commits polynomials, not evaluation blowups.
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;
    use crate::lookup::Lookup;
    use crate::prover::Proof;
    use crate::system::{CircuitInputs, System, SystemWitness};
    use crate::traits::{Algebra, Field};
    use p3_matrix::dense::RowMajorMatrix;

    #[test]
    fn scalar_quotient_matches_portable_evaluator_with_next_rows() {
        use crate::traits::{EvaluationDomain, TwoAdicField};
        let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(8, b"quotient-scratch")), 8);
        let fixed = RowMajorMatrix::new_col((0..8).map(Scalar::from_usize).collect());
        let (mut system, _) = System::new(
            config,
            [CircuitInputs {
                main_width: 1,
                preprocessed: Some(fixed),
                constraints: vec![Expr::main_next(0) - Expr::main(0)],
                lookups: vec![
                    Lookup::push(
                        Expr::preprocessed(0),
                        vec![Expr::main(0), Expr::main_next(0)],
                    ),
                    Lookup::pull(
                        Expr::constant(Scalar::ONE),
                        vec![Expr::preprocessed_next(0)],
                    ),
                ],
                ..Default::default()
            }],
        );
        let base = super::super::domain::Radix2Coset {
            log_size: 3,
            shift: Scalar::ONE,
        };
        for group in [1, 2, 3] {
            let circuit = &mut system.circuits[0];
            circuit.lookup_group_size = group;
            circuit.stage_2_width =
                crate::lookup::stage2_width(circuit.graph.lookups.len(), group, 1);
            circuit.constraint_count = circuit.graph.zeros.len()
                + crate::lookup::logup_constraint_count(circuit.graph.lookups.len(), group, 1);
            for log_size in [3, 4, 11] {
                let domain = super::super::domain::Radix2Coset {
                    log_size,
                    shift: Scalar::two_adic_generator(16),
                };
                let matrix = |width| {
                    RowMajorMatrix::new(
                        (0..domain.size() * width)
                            .map(|i| Scalar::from_usize(i * 7 + 13))
                            .collect(),
                        width,
                    )
                };
                let fixed = Some(matrix(1));
                let main = matrix(1);
                let lookup = matrix(circuit.stage_2_width);
                let publics = [3, 5, 7, 19].map(Scalar::from_u8);
                let alpha = Scalar::from_u8(29);
                let reference = crate::prover::quotient_values::<KzgConfig>(
                    &circuit,
                    &publics,
                    base,
                    domain,
                    &fixed,
                    &main,
                    &lookup,
                    alpha,
                    circuit.constraint_count(),
                );
                let actual = scalar_quotient_values(
                    &circuit,
                    &publics,
                    base,
                    domain,
                    fixed.as_ref(),
                    &main,
                    &lookup,
                    alpha,
                    circuit.constraint_count(),
                );
                assert_eq!(actual, reference);
            }
        }
    }

    #[test]
    fn lookup_tiles_match_serial_trace_across_waves_and_next_rows() {
        let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(2, b"lookup-tiles")), 8);
        let rows = 17 * (p3_maybe_rayon::prelude::current_num_threads() + 1) + 3;
        let main =
            RowMajorMatrix::new_col((0..rows).map(|i| Scalar::from_usize(i % 29 + 1)).collect());
        let fixed =
            RowMajorMatrix::new_col((0..rows).map(|i| Scalar::from_usize(i % 11)).collect());
        let beta = Scalar::from_u8(31);
        let gamma = Scalar::from_u8(7);
        let initial = Scalar::from_u8(23);
        let lookups = vec![
            Lookup::push(
                Expr::constant(Scalar::ONE),
                vec![Expr::main(0), Expr::preprocessed(0)],
            ),
            Lookup::pull(Expr::constant(Scalar::from_u8(2)), vec![Expr::main_next(0)]),
            Lookup::push(
                Expr::preprocessed(0),
                vec![Expr::main(0) + Expr::main_next(0)],
            ),
        ];
        for count in [0, 1, 3] {
            let (mut system, _) = System::new(
                config.clone(),
                [CircuitInputs {
                    main_width: 1,
                    preprocessed: Some(RowMajorMatrix::new_col(vec![Scalar::ONE; 2])),
                    lookups: lookups[..count].to_vec(),
                    ..Default::default()
                }],
            );
            let mut circuit = system.circuits.remove(0);
            for group in [1, 2, 3] {
                circuit.lookup_group_size = group;
                circuit.stage_2_width = crate::lookup::stage2_width(count, group, 1);
                let values = crate::system::compute_lookup_values_range(
                    &circuit,
                    &main,
                    Some(&fixed),
                    0..rows,
                );
                let (expected, totals) = crate::lookup::LookupValues::stage_2_traces(
                    &[values],
                    &[group],
                    beta,
                    &gamma,
                    initial,
                );
                for tile_rows in [1, 17, 1 << 16] {
                    let (actual, total) =
                        lookup_trace(&circuit, &main, Some(&fixed), beta, gamma, tile_rows);
                    assert_eq!(
                        actual.values, expected[0].values,
                        "lookups={count}, group={group}, tile={tile_rows}"
                    );
                    assert_eq!(total + initial, totals[0]);
                }
            }
        }
    }

    /// `a·b = c` per row, with a self-canceling push/pull lookup pair to
    /// exercise the stage-2 machinery — the KZG twin of the BabyBear
    /// config's smoke test.
    fn mul_circuit() -> CircuitInputs<Scalar> {
        let m = Expr::main;
        CircuitInputs {
            main_width: 3,
            constraints: vec![m(0) * m(1) - m(2)],
            lookups: vec![
                Lookup::push(Expr::constant(Scalar::from_u32(1)), vec![m(0), m(2)]),
                Lookup::pull(Expr::constant(Scalar::from_u32(1)), vec![m(0), m(2)]),
            ],
            ..Default::default()
        }
    }

    fn kzg_system() -> (System<KzgConfig>, crate::system::ProverKey<KzgConfig>) {
        let srs = Arc::new(Srs::unsafe_dev_setup(1 << 8, b"test"));
        let config = KzgConfig::new(srs, 8);
        System::new(config, [mul_circuit()])
    }

    #[test]
    fn streamed_lookups_preserve_proof_bytes() {
        let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(16, b"streamed-lookups")), 8);
        let definitions = vec![
            mul_circuit(),
            CircuitInputs {
                main_width: 1,
                constraints: vec![Expr::main(0) - Expr::constant(Scalar::from_u8(5))],
                ..Default::default()
            },
        ];
        let (serial, serial_key) = System::new(config.clone(), definitions.clone());
        let (streamed, streamed_key) = System::new(config.with_streaming_lookups(), definitions);
        let traces = vec![
            RowMajorMatrix::new(
                [2, 3, 6, 4, 5, 20, 7, 8, 56, 1, 1, 1]
                    .map(Scalar::from_u8)
                    .to_vec(),
                3,
            ),
            RowMajorMatrix::new_col(vec![Scalar::from_u8(5); 2]),
        ];
        let expected = serial.prove_multiple_claims(
            &serial_key,
            &[],
            SystemWitness::from_stage_1(traces.clone(), &serial),
        );
        let actual = streamed.prove_multiple_claims(
            &streamed_key,
            &[],
            SystemWitness::from_stage_1(traces, &streamed),
        );
        assert_eq!(actual.to_bytes().unwrap(), expected.to_bytes().unwrap());
        serial.verify_multiple_claims(&[], &actual).unwrap();
    }

    fn witness(system: &System<KzgConfig>) -> SystemWitness<Scalar> {
        let f = Scalar::from_u32;
        let trace = RowMajorMatrix::new([2, 3, 6, 4, 5, 20, 7, 8, 56, 1, 1, 1].map(f).to_vec(), 3);
        SystemWitness::from_stage_1(vec![trace], system)
    }

    #[test]
    fn kzg_prove_verify() {
        let (system, key) = kzg_system();
        let no_claims: &[&[Scalar]] = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness(&system));
        system
            .verify_multiple_claims(no_claims, &proof)
            .expect("KZG proof failed to verify");
    }

    #[test]
    fn opens_next_rows_only_when_the_graph_reads_them() {
        let f = Scalar::from_u8;
        let (system, key) = System::new(
            KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(8, b"next-rows")), 4),
            [
                CircuitInputs {
                    main_width: 1,
                    preprocessed: Some(RowMajorMatrix::new_col(vec![f(7); 4])),
                    constraints: vec![Expr::main_next(0) - Expr::preprocessed_next(0)],
                    ..Default::default()
                },
                CircuitInputs {
                    main_width: 1,
                    preprocessed: Some(RowMajorMatrix::new_col(vec![f(9); 2])),
                    constraints: vec![Expr::main(0) - Expr::preprocessed(0)],
                    ..Default::default()
                },
            ],
        );
        let proof = system.prove_multiple_claims(
            &key,
            &[],
            SystemWitness::from_stage_1(
                vec![
                    RowMajorMatrix::new_col(vec![f(7); 4]),
                    RowMajorMatrix::new_col(vec![f(9); 2]),
                ],
                &system,
            ),
        );
        system.verify_multiple_claims(&[], &proof).unwrap();
        assert_eq!(
            proof
                .stage_1_opened_values
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            [2, 1]
        );
        assert_eq!(
            proof
                .preprocessed_opened_values
                .as_ref()
                .unwrap()
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            [2, 1]
        );
        let codec =
            super::super::compact::FixedProofCodec::new(&system, &proof.log_degrees).unwrap();
        let decoded = codec.decode(&codec.encode(&proof).unwrap()).unwrap();
        system.verify_multiple_claims(&[], &decoded).unwrap();
        let mut bad = proof.clone();
        bad.stage_1_opened_values[0].pop();
        assert!(system.verify_multiple_claims(&[], &bad).is_err());
        let mut bad = proof.clone();
        bad.preprocessed_opened_values.as_mut().unwrap()[0][1][0] += Scalar::ONE;
        assert!(system.verify_multiple_claims(&[], &bad).is_err());
        let mut bad = proof.clone();
        bad.stage_1_opened_values[1].push(vec![Scalar::ZERO]);
        assert!(system.verify_multiple_claims(&[], &bad).is_err());
        // KZG commits columns separately, so inactive fixed matrices need no openings.
        let partial = system.prove_multiple_claims(
            &key,
            &[],
            SystemWitness::from_stage_1(
                vec![
                    RowMajorMatrix::new(vec![], 1),
                    RowMajorMatrix::new_col(vec![f(9); 2]),
                ],
                &system,
            ),
        );
        system.verify_multiple_claims(&[], &partial).unwrap();
        assert_eq!(partial.active, [false, true]);
        assert_eq!(
            partial.preprocessed_opened_values.as_ref().unwrap()[0].len(),
            0
        );
        let mut bad = partial.clone();
        bad.preprocessed_opened_values.as_mut().unwrap()[1].clear();
        assert!(system.verify_multiple_claims(&[], &bad).is_err());
        let mut bad = partial;
        bad.preprocessed_opened_values.as_mut().unwrap()[0].push(vec![f(7)]);
        assert!(system.verify_multiple_claims(&[], &bad).is_err());
    }

    #[test]
    fn quotient_slices_fit_an_srs_exactly_as_large_as_the_trace() {
        let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(4, b"exact-srs")), 4);
        let (system, key) = System::new(config, [mul_circuit()]);
        let proof = system.prove_multiple_claims(&key, &[], witness(&system));
        system.verify_multiple_claims(&[], &proof).unwrap();
    }

    #[test]
    fn batch_messages_work_with_scalar_challenges() {
        use crate::batch::{BatchMessage, ShardInput};
        let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(2, b"batch")), 4);
        let (system, key) = System::new(
            config,
            [CircuitInputs {
                main_width: 1,
                preprocessed: Some(RowMajorMatrix::new_col(vec![Scalar::ONE, Scalar::ZERO])),
                lookups: vec![Lookup::pull(Expr::preprocessed(0), vec![Expr::main(0)])],
                ..Default::default()
            }],
        );
        let value = Scalar::from_u8(9);
        let witness =
            SystemWitness::from_stage_1(vec![RowMajorMatrix::new_col(vec![value; 2])], &system);
        let mut proof = system.prove_batch(
            &key,
            vec![ShardInput {
                claims: vec![],
                witness,
            }],
            vec![BatchMessage {
                args: vec![value],
                multiplicity: Scalar::ONE,
            }],
        );
        system.verify_batch(&proof).unwrap();
        proof.preamble.messages[0].args[0] += Scalar::ONE;
        assert!(system.verify_batch(&proof).is_err());
    }

    #[test]
    fn kzg_tampering_rejected() {
        let (system, key) = kzg_system();
        let no_claims: &[&[Scalar]] = &[];
        let prove = || system.prove_multiple_claims(&key, no_claims, witness(&system));

        let mut tampered = prove();
        tampered.intermediate_accumulators[0] += <Scalar as Algebra<Scalar>>::ONE;
        assert!(system.verify_multiple_claims(no_claims, &tampered).is_err());

        let mut tampered = prove();
        tampered.stage_1_opened_values[0][0][0] += <Scalar as Algebra<Scalar>>::ONE;
        assert!(system.verify_multiple_claims(no_claims, &tampered).is_err());

        let mut tampered = prove();
        tampered.quotient_opened_values[0][0][0] += <Scalar as Algebra<Scalar>>::ONE;
        assert!(system.verify_multiple_claims(no_claims, &tampered).is_err());
    }

    #[test]
    fn kzg_wrong_claim_rejected() {
        let (system, key) = kzg_system();
        let no_claims: &[&[Scalar]] = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness(&system));
        let claim = [Scalar::from_u32(42)];
        assert!(system.verify(&claim, &proof).is_err());
    }

    #[test]
    fn kzg_serialization_round_trip() {
        let (system, key) = kzg_system();
        let no_claims: &[&[Scalar]] = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness(&system));
        let bytes = proof.to_bytes().expect("serialize");
        let proof2 = Proof::<KzgConfig>::from_bytes(&bytes).expect("deserialize");
        system.verify_multiple_claims(no_claims, &proof2).unwrap();
    }

    /// Two circuits at different trace heights: ζ·g differs per height,
    /// so the opening carries three distinct points and the per-point
    /// witness batching (and cross-point pairing batch) is exercised.
    #[test]
    fn kzg_two_circuits_two_heights() {
        let srs = Arc::new(Srs::unsafe_dev_setup(1 << 8, b"test"));
        let config = KzgConfig::new(srs, 8);
        let (system, key) = System::new(config, [mul_circuit(), mul_circuit()]);
        let f = Scalar::from_u32;
        let small = RowMajorMatrix::new([2, 3, 6, 4, 5, 20, 7, 8, 56, 1, 1, 1].map(f).to_vec(), 3);
        let mut long = small.values.clone();
        for _ in 0..2 {
            long.extend(long.clone());
        }
        let witness =
            SystemWitness::from_stage_1(vec![small, RowMajorMatrix::new(long, 3)], &system);
        let no_claims: &[&[Scalar]] = &[];
        let proof = system.prove_multiple_claims(&key, no_claims, witness);
        let bytes = proof.to_bytes().expect("serialize");
        println!(
            "KZG proof: {} bytes (two circuits, heights 4 and 16)",
            bytes.len()
        );
        system.verify_multiple_claims(no_claims, &proof).unwrap();
    }
}
