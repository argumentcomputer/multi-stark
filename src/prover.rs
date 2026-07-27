//! Multi-circuit STARK prover, constraint-IR version.
//!
//! Parallel to [`crate::prover`]. The protocol (transcript, commitments,
//! lookup argument, quotient, FRI) is identical; the only difference is how
//! the composition polynomial is built: instead of evaluating an AIR against
//! a folder, the prover runs the compiled circuit's dense sweep in
//! `PackedVal` on the quotient domain and folds the constraint values with
//! the constraint challenge. Stage-2 columns are treated as ordinary base
//! columns (they were committed flattened), so no extension packing is
//! needed here.
//!
//! Reuses the proof types, lookup witness, and PCS plumbing of the existing
//! prover unchanged.

use crate::config::{
    Com, Domain, EvaluationsOnDomain, PackedChallenge, PackedVal, PcsProof, StarkGenericConfig, Val,
};
use crate::eval::VarValues;
use crate::lookup::{LookupValues, fingerprint};
use crate::system::{ProverKey, System, SystemWitness};

use bincode::config::{Configuration, Fixint, LittleEndian, standard};
use bincode::error::{DecodeError, EncodeError};
use bincode::serde::{decode_from_slice, encode_to_vec};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{LagrangeSelectors, OpenedValuesForRound, Pcs, PolynomialSpace};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{
    Algebra, BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

/// Polynomial commitments included in the proof.
#[derive(Serialize, Deserialize)]
pub struct Commitments<Com> {
    /// Commitment to the stage 1 (main) execution traces.
    pub stage_1_trace: Com,
    /// Commitment to the stage 2 (lookup) execution traces.
    pub stage_2_trace: Com,
    /// Commitment to the quotient polynomial chunks.
    pub quotient_chunks: Com,
}

/// A STARK proof for a multi-circuit system.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Proof<SC: StarkGenericConfig> {
    /// Activation bitmap over the system's canonical circuit set: circuit i
    /// is covered by this proof iff `active[i]`. Every other per-circuit
    /// sequence in this proof is indexed by ACTIVE position.
    pub active: Vec<bool>,
    pub commitments: Commitments<Com<SC>>,
    /// Per-active-circuit intermediate accumulator values for the lookup
    /// argument.
    pub intermediate_accumulators: Vec<SC::Challenge>,
    /// Log2 of the trace degree for each active circuit.
    pub log_degrees: Vec<u8>,
    /// PCS opening proof covering all rounds.
    pub opening_proof: PcsProof<SC>,
    pub quotient_opened_values: OpenedValuesForRound<SC::Challenge>,
    pub preprocessed_opened_values: Option<OpenedValuesForRound<SC::Challenge>>,
    pub stage_1_opened_values: OpenedValuesForRound<SC::Challenge>,
    pub stage_2_opened_values: OpenedValuesForRound<SC::Challenge>,
}

impl<SC: StarkGenericConfig> Proof<SC> {
    fn serde_config() -> Configuration<LittleEndian, Fixint> {
        standard().with_little_endian().with_fixed_int_encoding()
    }

    #[inline]
    pub fn to_bytes(&self) -> Result<Vec<u8>, EncodeError> {
        encode_to_vec(self, Self::serde_config())
    }

    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let (proof, _num_bytes) = decode_from_slice(bytes, Self::serde_config())?;
        Ok(proof)
    }
}

impl<SC> System<SC>
where
    SC: StarkGenericConfig,
    Val<SC>: TwoAdicField + Ord,
{
    /// Generates a STARK proof for the system with a single claim.
    pub fn prove(
        &self,
        key: &ProverKey<SC>,
        claim: &[Val<SC>],
        witness: SystemWitness<Val<SC>>,
    ) -> Proof<SC> {
        self.prove_multiple_claims(key, &[claim], witness)
    }

    /// Generates a STARK proof for the system with multiple claims.
    ///
    /// # Panics
    /// Panics if every circuit's trace is empty (nothing to prove).
    #[tracing::instrument(level = "info", skip_all, name = "stark/prove")]
    pub fn prove_multiple_claims(
        &self,
        key: &ProverKey<SC>,
        claims: &[&[Val<SC>]],
        witness: SystemWitness<Val<SC>>,
    ) -> Proof<SC> {
        let pcs = self.config.pcs();
        let mut challenger = self.config.initialise_challenger();

        self.observe_shape(&mut challenger);

        // Sparse activation: a circuit with an empty stage-1 trace is
        // inactive for this proof. The bitmap is bound before any commitment.
        let active: Vec<bool> = witness.traces.iter().map(|t| t.height() > 0).collect();
        for &is_active in &active {
            challenger.observe(Val::<SC>::from_bool(is_active));
        }
        let active_indices: Vec<usize> = active
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| a.then_some(i))
            .collect();
        assert!(
            !active_indices.is_empty(),
            "cannot prove with every circuit deactivated (all traces empty)"
        );
        let mut active_pos: Vec<Option<usize>> = vec![None; active.len()];
        for (pos, &ci) in active_indices.iter().enumerate() {
            active_pos[ci] = Some(pos);
        }

        // Stage 1 commit.
        let _g = tracing::info_span!("stark/stage1_commit").entered();
        let mut log_degrees = vec![];
        let evaluations = witness
            .traces
            .into_iter()
            .zip(active.clone())
            .filter_map(|(trace, is_active)| is_active.then_some(trace))
            .map(|trace| {
                let degree = trace.height();
                let log_degree = log2_strict_usize(degree);
                let trace_domain = pcs.natural_domain_for_degree(degree);
                log_degrees.push(log_degree);
                (trace_domain, trace)
            });
        let (stage_1_trace_commit, stage_1_trace_data) = pcs.commit(evaluations);
        drop(_g);

        if let Some(commit) = &self.preprocessed_commit {
            challenger.observe(commit.clone());
        }
        challenger.observe(stage_1_trace_commit.clone());
        for log_degree in &log_degrees {
            challenger.observe(Val::<SC>::from_usize(*log_degree));
        }

        // Observe the claims, length-prefixed, before sampling challenges.
        challenger.observe(Val::<SC>::from_usize(claims.len()));
        for claim in claims {
            challenger.observe(Val::<SC>::from_usize(claim.len()));
            challenger.observe_slice(claim);
        }

        // Lookup challenges.
        let lookup_argument_challenge: SC::Challenge = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: SC::Challenge = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);

        // Initial accumulator from the claims.
        let mut acc = SC::Challenge::ZERO;
        for claim in claims {
            let message = lookup_argument_challenge
                + fingerprint(&fingerprint_challenge, claim.iter().cloned());
            acc += message.inverse();
        }

        // Stage 2 (lookup) trace construction.
        let _g = tracing::info_span!("stark/lookup_construction").entered();
        let active_lookups: Vec<_> = witness
            .lookups
            .into_iter()
            .zip(&active)
            .filter_map(|(l, &is_active)| is_active.then_some(l))
            .collect();
        let (stage_2_traces, intermediate_accumulators) = LookupValues::stage_2_traces(
            &active_lookups,
            lookup_argument_challenge,
            &fingerprint_challenge,
            acc,
        );
        drop(active_lookups);
        drop(_g);

        // Stage 2 commit (flattened extension traces).
        let _g = tracing::info_span!("stark/stage2_commit").entered();
        let evaluations = stage_2_traces.into_iter().map(|trace| {
            let degree = trace.height();
            let trace_domain = pcs.natural_domain_for_degree(degree);
            (trace_domain, trace.flatten_to_base())
        });
        let (stage_2_trace_commit, stage_2_trace_data) = pcs.commit(evaluations);
        drop(_g);
        challenger.observe(stage_2_trace_commit.clone());

        for acc in &intermediate_accumulators {
            challenger.observe_algebra_element(*acc);
        }

        // Constraint challenge.
        let constraint_challenge: SC::Challenge = challenger.sample_algebra_element();

        // Quotient computation and commit.
        let _g = tracing::info_span!("stark/quotient").entered();
        debug_assert_eq!(intermediate_accumulators.len(), active_indices.len());
        debug_assert_eq!(log_degrees.len(), active_indices.len());
        let dft = Radix2DitParallel::<Val<SC>>::default();
        let quotient_evaluations = active_indices
            .iter()
            .zip(log_degrees.iter())
            .zip(intermediate_accumulators.iter())
            .enumerate()
            .map(|(pos, ((&ci, log_degree), next_acc))| {
                let circuit = &self.circuits[ci];
                let quotient_degree = circuit.quotient_degree();
                let log_quotient_degree = log2_strict_usize(quotient_degree);
                let trace_domain = pcs.natural_domain_for_degree(1 << log_degree);
                let quotient_domain =
                    trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));
                let preprocessed_trace_on_quotient_domain = key
                    .preprocessed_data
                    .as_ref()
                    .zip(self.preprocessed_indices[ci])
                    .map(|(preprocessed_trace_data, preprocessed_idx)| {
                        pcs.get_evaluations_on_domain(
                            preprocessed_trace_data,
                            preprocessed_idx,
                            quotient_domain,
                        )
                    });
                let stage_1_trace_on_quotient_domain =
                    pcs.get_evaluations_on_domain(&stage_1_trace_data, pos, quotient_domain);
                let stage_2_trace_on_quotient_domain =
                    pcs.get_evaluations_on_domain(&stage_2_trace_data, pos, quotient_domain);

                // The four lookup publics (β, γ, acc, next_acc) as flat base
                // coordinates, in the layout the synthesized constraints read.
                let mut lookup_publics: Vec<Val<SC>> = Vec::new();
                for ef in [
                    lookup_argument_challenge,
                    fingerprint_challenge,
                    acc,
                    *next_acc,
                ] {
                    lookup_publics.extend_from_slice(ef.as_basis_coefficients_slice());
                }

                let quotient_values = quotient_values::<SC>(
                    circuit,
                    &lookup_publics,
                    trace_domain,
                    quotient_domain,
                    &preprocessed_trace_on_quotient_domain,
                    &stage_1_trace_on_quotient_domain,
                    &stage_2_trace_on_quotient_domain,
                    constraint_challenge,
                    circuit.constraint_count(),
                );
                let quotient_flat =
                    RowMajorMatrix::new_col(quotient_values).flatten_to_base::<Val<SC>>();
                let coefficients =
                    dft.coset_idft_batch(quotient_flat, quotient_domain.first_point());
                let ext_degree = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
                let n = 1 << log_degree;
                let width = quotient_degree * ext_degree;
                let mut sliced = Vec::with_capacity(n * width);
                for row in 0..n {
                    for chunk in 0..quotient_degree {
                        sliced.extend_from_slice(
                            &coefficients.values[(chunk * n + row) * ext_degree
                                ..(chunk * n + row + 1) * ext_degree],
                        );
                    }
                }
                let quotient_chunks_evals = dft
                    .coset_dft_batch(
                        RowMajorMatrix::new(sliced, width),
                        trace_domain.first_point(),
                    )
                    .to_row_major_matrix();
                acc = *next_acc;
                (trace_domain, quotient_chunks_evals)
            });
        let (quotient_commit, quotient_data) = pcs.commit(quotient_evaluations);
        challenger.observe(quotient_commit.clone());
        drop(_g);

        let commitments = Commitments {
            stage_1_trace: stage_1_trace_commit,
            stage_2_trace: stage_2_trace_commit,
            quotient_chunks: quotient_commit,
        };

        // FRI opening.
        let _g = tracing::info_span!("stark/fri_open").entered();
        let zeta: SC::Challenge = challenger.sample_algebra_element();
        let mut round0_openings = vec![];
        let mut round1_openings = vec![];
        let mut round2_openings = vec![];
        let mut round3_openings = vec![];
        for &log_degree in log_degrees.iter() {
            let trace_domain = pcs.natural_domain_for_degree(1 << log_degree);
            let zeta_next = trace_domain
                .next_point(zeta)
                .expect("domain has no next point");
            round1_openings.push(vec![zeta, zeta_next]);
            round2_openings.push(vec![zeta, zeta_next]);
            round3_openings.push(vec![zeta]);
        }
        for (prep_index, &pos) in self.preprocessed_indices.iter().zip(&active_pos) {
            if prep_index.is_some() {
                match pos {
                    Some(pos) => {
                        let trace_domain = pcs.natural_domain_for_degree(1 << log_degrees[pos]);
                        let zeta_next = trace_domain
                            .next_point(zeta)
                            .expect("domain has no next point");
                        round0_openings.push(vec![zeta, zeta_next]);
                    }
                    None => round0_openings.push(vec![]),
                }
            }
        }
        let mut rounds = vec![
            (&stage_1_trace_data, round1_openings),
            (&stage_2_trace_data, round2_openings),
            (&quotient_data, round3_openings),
        ];
        if self.preprocessed_commit.is_some() {
            rounds.push((key.preprocessed_data.as_ref().unwrap(), round0_openings));
        }
        let (opened_values, opening_proof) = pcs.open(rounds, &mut challenger);
        drop(_g);
        let mut opened_values_iter = opened_values.into_iter();
        let stage_1_opened_values = opened_values_iter.next().unwrap();
        let stage_2_opened_values = opened_values_iter.next().unwrap();
        let quotient_opened_values = opened_values_iter.next().unwrap();
        let preprocessed_opened_values = opened_values_iter.next();
        debug_assert!(opened_values_iter.next().is_none());
        let log_degrees = log_degrees
            .into_iter()
            .map(|n| n.try_into().unwrap())
            .collect();
        Proof {
            active,
            commitments,
            intermediate_accumulators,
            log_degrees,
            opening_proof,
            quotient_opened_values,
            preprocessed_opened_values,
            stage_1_opened_values,
            stage_2_opened_values,
        }
    }
}

/// Evaluates the folded constraints on the quotient domain and divides by
/// the vanishing polynomial, producing the quotient values.
#[allow(clippy::too_many_arguments)]
fn quotient_values<SC>(
    circuit: &crate::system::Circuit<Val<SC>>,
    lookup_publics: &[Val<SC>],
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    preprocessed_on_quotient_domain: &Option<EvaluationsOnDomain<'_, SC>>,
    stage_1_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    stage_2_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    alpha: SC::Challenge,
    constraint_count: usize,
) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
    Val<SC>: TwoAdicField + Ord,
{
    let quotient_size = quotient_domain.size();
    let main_width = circuit.main_width;
    let stage_2_width = circuit.stage_2_width;
    let preprocessed_width = circuit.preprocessed_width;
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_vanishing.push(Val::<SC>::default());
    }

    // α powers in reverse (constraint i of k weighted by α^{k-1-i}),
    // decomposed per basis coordinate for the batched base-field fold.
    let mut alpha_powers = alpha.powers().collect_n(constraint_count);
    alpha_powers.reverse();
    let decomposed_alpha_powers: Vec<Vec<Val<SC>>> =
        (0..<SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION)
            .map(|i| {
                alpha_powers
                    .iter()
                    .map(|x| x.as_basis_coefficients_slice()[i])
                    .collect()
            })
            .collect();

    // Public coordinates broadcast to packed base values.
    let publics_packed: Vec<PackedVal<SC>> = lookup_publics
        .iter()
        .map(|&c| PackedVal::<SC>::from(c))
        .collect();

    let inner = |i_start: usize| {
        quotient_values_inner::<SC>(
            circuit,
            &sels,
            quotient_size,
            preprocessed_on_quotient_domain,
            stage_1_on_quotient_domain,
            stage_2_on_quotient_domain,
            main_width,
            stage_2_width,
            preprocessed_width,
            &publics_packed,
            &decomposed_alpha_powers,
            next_step,
            i_start,
        )
    };
    #[cfg(feature = "parallel")]
    {
        (0..quotient_size)
            .into_par_iter()
            .step_by(PackedVal::<SC>::WIDTH)
            .flat_map_iter(inner)
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..quotient_size)
            .step_by(PackedVal::<SC>::WIDTH)
            .flat_map_iter(inner)
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn quotient_values_inner<SC>(
    circuit: &crate::system::Circuit<Val<SC>>,
    sels: &LagrangeSelectors<Vec<Val<SC>>>,
    quotient_size: usize,
    preprocessed_on_quotient_domain: &Option<EvaluationsOnDomain<'_, SC>>,
    stage_1_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    stage_2_on_quotient_domain: &EvaluationsOnDomain<'_, SC>,
    main_width: usize,
    stage_2_width: usize,
    preprocessed_width: usize,
    publics_packed: &[PackedVal<SC>],
    decomposed_alpha_powers: &[Vec<Val<SC>>],
    next_step: usize,
    i_start: usize,
) -> impl Iterator<Item = SC::Challenge>
where
    SC: StarkGenericConfig,
    Val<SC>: TwoAdicField + Ord,
{
    let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;
    let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
    let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
    let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
    let inv_vanishing = *PackedVal::<SC>::from_slice(&sels.inv_vanishing[i_range]);

    // Packed two-row windows of each trace, as base columns.
    let preprocessed_pair: Option<Vec<PackedVal<SC>>> = preprocessed_on_quotient_domain
        .as_ref()
        .map(|m| m.vertically_packed_row_pair::<PackedVal<SC>>(i_start, next_step));
    let stage_1_pair =
        stage_1_on_quotient_domain.vertically_packed_row_pair::<PackedVal<SC>>(i_start, next_step);
    let stage_2_pair =
        stage_2_on_quotient_domain.vertically_packed_row_pair::<PackedVal<SC>>(i_start, next_step);

    let (stage_1_cur, stage_1_next) = stage_1_pair.split_at(main_width);
    let (stage_2_cur, stage_2_next) = stage_2_pair.split_at(stage_2_width);
    let (preprocessed_cur, preprocessed_next): (&[PackedVal<SC>], &[PackedVal<SC>]) =
        match &preprocessed_pair {
            Some(pair) => pair.split_at(preprocessed_width),
            None => (&[], &[]),
        };

    let view = VarValues {
        preprocessed: [preprocessed_cur, preprocessed_next],
        main: [stage_1_cur, stage_1_next],
        stage2: [stage_2_cur, stage_2_next],
        publics: publics_packed,
        is_first_row,
        is_last_row,
        is_transition,
    };
    let mut buf = Vec::new();
    circuit.compiled.sweep(&view, &mut buf);
    let constraint_values = circuit.compiled.constraint_values(&buf);

    // Fold the base constraint values with α through the decomposed path,
    // reassembling one packed extension accumulator.
    let accumulator = PackedChallenge::<SC>::from_basis_coefficients_fn(|coeff_idx| {
        PackedVal::<SC>::batched_linear_combination(
            &constraint_values,
            &decomposed_alpha_powers[coeff_idx],
        )
    });
    let quotient = accumulator * inv_vanishing;

    (0..quotient_size.min(PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
        SC::Challenge::from_basis_coefficients_fn(|coeff_idx| {
            <PackedChallenge<SC> as BasedVectorSpace<PackedVal<SC>>>::as_basis_coefficients_slice(
                &quotient,
            )[coeff_idx]
                .as_slice()[idx_in_packing]
        })
    })
}
