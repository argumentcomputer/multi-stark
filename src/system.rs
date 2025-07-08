use crate::{
    ensure,
    types::{Challenger, ExtVal, PackedExtVal, PackedVal, Pcs, StarkConfig, Val},
};
use p3_air::{Air, BaseAirWithPublicValues};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{OpenedValues, OpenedValuesForRound, Pcs as PcsTrait, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{
    Domain, PcsError, ProverConstraintFolder, StarkGenericConfig, SymbolicAirBuilder,
    get_max_constraint_degree, get_symbolic_constraints,
};
use p3_util::log2_strict_usize;
use std::{collections::BTreeMap as Map, iter::once};

pub type Name = &'static str;

/// Each circuit is required to have at least 3 arguments. Namely,
/// the accumulator, the lookup challenge and the fingerprint challenge.
pub const MIN_IO_SIZE: usize = 3;

pub struct System<A> {
    pub circuits: Vec<Circuit<A>>,
    pub circuit_names: Map<Name, usize>,
}

impl<A> System<A> {
    pub fn new<Iter: Iterator<Item = (Name, Circuit<A>)>>(iter: Iter) -> Self {
        let mut circuits = vec![];
        let mut circuit_names = Map::new();
        iter.for_each(|(name, circuit)| {
            let idx = circuits.len();
            if let Some(prev_idx) = circuit_names.insert(name, idx) {
                eprintln!("Warning: circuit of name `{name}` was redefined");
                circuits[prev_idx] = circuit;
            } else {
                circuits.push(circuit);
            }
        });
        Self {
            circuits,
            circuit_names,
        }
    }
}

pub struct Circuit<A> {
    pub air: A,
    pub constraint_count: usize,
    pub max_constraint_degree: usize,
    pub preprocessed_width: usize,
    pub stage1_width: usize,
    pub stage2_width: usize,
}

impl<A> Circuit<A> {
    pub fn width(&self) -> usize {
        self.stage1_width + self.stage2_width + self.preprocessed_width
    }
}

pub struct CircuitWitness {
    pub trace: RowMajorMatrix<Val>,
}

pub struct SystemWitness {
    pub circuits: Vec<CircuitWitness>,
}

pub struct Claim {
    pub circuit_name: Name,
    pub args: Vec<Val>,
}

type Commitment = <Pcs as PcsTrait<ExtVal, Challenger>>::Commitment;
type PcsProof = <Pcs as PcsTrait<ExtVal, Challenger>>::Proof;

pub struct Commitments {
    // TODO add stage 2
    pub stage1_trace: Commitment,
    pub quotient_chunks: Commitment,
}

pub struct Proof {
    pub claim: Claim,
    pub commitments: Commitments,
    pub log_degrees: Vec<u8>,
    pub opening_proof: PcsProof,
    pub stage1_opened_values: OpenedValuesForRound<ExtVal>,
    pub quotient_opened_values: OpenedValuesForRound<ExtVal>,
}

#[derive(Debug)]
pub enum VerificationError<PcsErr> {
    InvalidClaim,
    InvalidProofShape(u32),
    InvalidOpeningArgument(PcsErr),
    OodEvaluationMismatch,
}

impl<A: BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>> Circuit<A> {
    pub fn from_air_single_stage(air: A) -> Result<Self, String> {
        let io_size = air.num_public_values();
        ensure!(io_size == MIN_IO_SIZE, "Incompatible IO size");
        let preprocessed_width = air.preprocessed_trace().map_or(0, |mat| mat.width());
        let constraint_count = get_symbolic_constraints(&air, preprocessed_width, io_size).len();
        let max_constraint_degree = get_max_constraint_degree(&air, preprocessed_width, io_size);
        let stage1_width = air.width() - preprocessed_width;
        let stage2_width = 0;
        Ok(Self {
            air,
            max_constraint_degree,
            preprocessed_width,
            constraint_count,
            stage1_width,
            stage2_width,
        })
    }

    pub fn is_well_formed(&self) -> Result<(), String> {
        let io_size = self.air.num_public_values();
        let preprocessed_width = self.air.preprocessed_trace().map_or(0, |mat| mat.width());
        let constraint_count =
            get_symbolic_constraints(&self.air, preprocessed_width, io_size).len();
        let max_constraint_count =
            get_max_constraint_degree(&self.air, preprocessed_width, io_size);
        let width = self.air.width();
        // As of now, only the minimum IO size is supported.
        ensure!(io_size == MIN_IO_SIZE, "Incompatible IO size");
        ensure!(
            self.constraint_count == constraint_count,
            "Incompatible constraint count"
        );
        ensure!(
            self.max_constraint_degree == max_constraint_count,
            "Incompatible constraint degree"
        );
        ensure!(
            self.preprocessed_width == preprocessed_width,
            "Incompatible widths"
        );
        ensure!(self.width() == width, "Incompatible widths");
        Ok(())
    }
}

impl<A: BaseAirWithPublicValues<Val> + Air<SymbolicAirBuilder<Val>>> System<A> {
    pub fn is_well_formed(&self) -> Result<(), String> {
        ensure!(
            self.circuits.len() == self.circuit_names.len(),
            "Map of names is not well-formed"
        );
        let mut idxs = self.circuit_names.values().copied().collect::<Vec<_>>();
        idxs.sort();
        ensure!(
            idxs == (0..self.circuits.len()).collect::<Vec<_>>(),
            "Map of names is not well-formed"
        );
        self.circuits.iter().try_for_each(|c| c.is_well_formed())?;
        Ok(())
    }
}

impl<A: BaseAirWithPublicValues<Val> + for<'a> Air<ProverConstraintFolder<'a, StarkConfig>>>
    System<A>
{
    pub fn prove(&self, config: &StarkConfig, claim: Claim, witness: SystemWitness) -> Proof {
        // initialize pcs and challenger
        let pcs = config.pcs();
        let mut challenger = config.initialise_challenger();

        // commit to stage 1 traces
        let mut log_degrees = vec![];
        let evaluations = witness.circuits.into_iter().map(|witness| {
            let trace = witness.trace;
            let degree = trace.height();
            let log_degree = log2_strict_usize(degree);
            let trace_domain =
                <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(pcs, degree);
            log_degrees.push(log_degree);
            (trace_domain, trace)
        });
        let (stage1_trace_commit, stage1_trace_data) =
            <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, evaluations);
        // TODO: do we have to observe the log_degrees?
        challenger.observe(stage1_trace_commit);

        // generate lookup challenges
        // TODO use `ExtVal` instead of `Val`
        let lookup_argument_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);
        // TODO commit to stage 2 traces

        // observe the claim
        let circuit_index = Val::from_usize(*self.circuit_names.get(claim.circuit_name).unwrap());
        challenger.observe(circuit_index);
        challenger.observe_slice(&claim.args);
        // construct the accumulator from the claim
        let claim_iter = claim.args.iter().rev().copied().chain(once(circuit_index));
        let init_acc = fingerprint_reverse(fingerprint_challenge, claim_iter);
        let public_values = vec![init_acc, lookup_argument_challenge, fingerprint_challenge];

        // generate constraint challenge
        let constraint_challenge: ExtVal = challenger.sample_algebra_element();

        // commit to evaluations of the quotient polynomials
        let mut quotient_degrees = vec![];
        let quotient_evaluations = self
            .circuits
            .iter()
            .zip(log_degrees.iter())
            .enumerate()
            .flat_map(|(idx, (circuit, log_degree))| {
                let air = &circuit.air;
                // quotient degree is at most 1 less than the max degree, padded to a power of two
                let quotient_degree =
                    (circuit.max_constraint_degree.max(2) - 1).next_power_of_two();
                let log_quotient_degree = log2_strict_usize(quotient_degree);
                let trace_domain = <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                    pcs,
                    1 << log_degree,
                );
                let quotient_domain =
                    trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));
                // TODO add stage 2 traces
                let stage1_trace_on_quotient_domain =
                    <Pcs as PcsTrait<ExtVal, Challenger>>::get_evaluations_on_domain(
                        pcs,
                        &stage1_trace_data,
                        idx,
                        quotient_domain,
                    );
                // compute the quotient values which are elements of the extension field and flatten it to the base field
                let quotient_values = quotient_values(
                    air,
                    &public_values,
                    trace_domain,
                    quotient_domain,
                    stage1_trace_on_quotient_domain,
                    constraint_challenge,
                    circuit.constraint_count,
                );
                let quotient_flat =
                    RowMajorMatrix::new_col(quotient_values).flatten_to_base::<Val>();
                // note that, in general, the quotients have a degree that is greater than the trace polynomials,
                // so for FRI to work so we must split into smaller polynomials
                let quotient_sub_evaluations =
                    quotient_domain.split_evals(quotient_degree, quotient_flat);
                let quotient_sub_domains = quotient_domain.split_domains(quotient_degree);
                // need to save the quotient degree for later
                quotient_degrees.push(quotient_degree);
                quotient_sub_domains
                    .into_iter()
                    .zip(quotient_sub_evaluations)
            });
        let (quotient_commit, quotient_data) =
            <Pcs as PcsTrait<ExtVal, Challenger>>::commit(pcs, quotient_evaluations);
        challenger.observe(quotient_commit);

        // save the commitments
        let commitments = Commitments {
            stage1_trace: stage1_trace_commit,
            quotient_chunks: quotient_commit,
        };

        // generate the out of domain point and prove polynomial evaluations
        let zeta: ExtVal = challenger.sample_algebra_element();
        let mut round1_openings = vec![];
        let mut round2_openings = vec![];
        log_degrees.iter().zip(quotient_degrees.iter()).for_each(
            |(log_degree, quotient_degree)| {
                let trace_domain = <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                    pcs,
                    1 << log_degree,
                );
                let zeta_next = trace_domain.next_point(zeta).unwrap();
                round1_openings.push(vec![zeta, zeta_next]);
                round2_openings.extend(vec![vec![zeta]; *quotient_degree]);
            },
        );
        let rounds = vec![
            (&stage1_trace_data, round1_openings),
            (&quotient_data, round2_openings),
        ];
        let (opened_values, opening_proof) = pcs.open(rounds, &mut challenger);
        let mut opened_values_iter = opened_values.into_iter();
        let stage1_opened_values = opened_values_iter.next().unwrap();
        let quotient_opened_values = opened_values_iter.next().unwrap();
        debug_assert!(opened_values_iter.next().is_none());
        let log_degrees = log_degrees.into_iter().map(|n| n as u8).collect();
        Proof {
            claim,
            commitments,
            stage1_opened_values,
            quotient_opened_values,
            opening_proof,
            log_degrees,
        }
    }

    pub fn verify(
        &self,
        config: &StarkConfig,
        proof: &Proof,
    ) -> Result<(), VerificationError<PcsError<StarkConfig>>> {
        let Proof {
            commitments,
            stage1_opened_values,
            quotient_opened_values,
            opening_proof,
            claim,
            log_degrees,
        } = proof;
        let num_chips = self.circuits.len();

        // check the claim and proof shape
        let circuit_index = Val::from_usize(
            *self
                .circuit_names
                .get(claim.circuit_name)
                .ok_or(VerificationError::InvalidClaim)?,
        );
        // stage 1 round
        ensure!(
            stage1_opened_values.len() == num_chips,
            VerificationError::InvalidProofShape(1)
        );
        for (i, circuit) in self.circuits.iter().enumerate() {
            // zeta and zeta_next
            let num_openings = 2;
            ensure!(
                stage1_opened_values[i].len() == num_openings,
                VerificationError::InvalidProofShape(2)
            );
            for j in 0..num_openings {
                ensure!(
                    stage1_opened_values[i][j].len() == circuit.width(),
                    VerificationError::InvalidProofShape(3)
                );
            }
        }
        // TODO missing stage 2 round
        // quotient round
        let mut quotient_degrees = vec![];
        for circuit in self.circuits.iter() {
            let quotient_degree = (circuit.max_constraint_degree.max(2) - 1).next_power_of_two();
            quotient_degrees.push(quotient_degree);
        }
        let quotient_size: usize = quotient_degrees.iter().sum();
        ensure!(
            quotient_opened_values.len() == quotient_size,
            VerificationError::InvalidProofShape(4)
        );
        for i in 0..quotient_size {
            // zeta
            let num_openings = 1;
            ensure!(
                quotient_opened_values[i].len() == num_openings,
                VerificationError::InvalidProofShape(2)
            );
            ensure!(
                quotient_opened_values[i][0].len() == <ExtVal as BasedVectorSpace<Val>>::DIMENSION,
                VerificationError::InvalidProofShape(3)
            );
        }

        // initialize pcs and challenger
        let pcs = config.pcs();
        let mut challenger = config.initialise_challenger();

        // observe stage1 commitment
        challenger.observe(commitments.stage1_trace);

        // generate lookup challenges
        // TODO use `ExtVal` instead of `Val`
        let lookup_argument_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(lookup_argument_challenge);
        let fingerprint_challenge: Val = challenger.sample_algebra_element();
        challenger.observe_algebra_element(fingerprint_challenge);
        // TODO commit to stage 2 traces

        // observe the claim
        challenger.observe(circuit_index);
        challenger.observe_slice(&claim.args);
        // construct the accumulator from the claim
        let claim_iter = claim.args.iter().rev().copied().chain(once(circuit_index));
        let init_acc = fingerprint_reverse(fingerprint_challenge, claim_iter);
        let public_values = vec![init_acc, lookup_argument_challenge, fingerprint_challenge];
        // TODO stage 2

        // generate constraint challenge
        let constraint_challenge: ExtVal = challenger.sample_algebra_element();

        // observe quotient commitment
        challenger.observe(commitments.quotient_chunks);

        // generate out of domain points and verify the PCS opening
        let zeta: ExtVal = challenger.sample_algebra_element();
        // let stage1_trace_evaluations = vec![];
        // let quotient_chunks_evaluations = vec![];
        log_degrees.iter().zip(quotient_degrees.iter()).for_each(
            |(log_degree, quotient_degree)| {
                let log_quotient_degree = log2_strict_usize(*quotient_degree);
                let trace_domain = <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                    pcs,
                    1 << log_degree,
                );
                let quotient_domain =
                    trace_domain.create_disjoint_domain((1 << log_degree) << log_quotient_degree);
                let quotient_chunks_domains = quotient_domain.split_domains(*quotient_degree);
                let unshifted_quotient_chunks_domains = quotient_chunks_domains
                    .iter()
                    .map(|domain| {
                        <Pcs as PcsTrait<ExtVal, Challenger>>::natural_domain_for_degree(
                            pcs,
                            domain.size(),
                        )
                    })
                    .collect::<Vec<_>>();
                // stage1_trace_evaluations.push((trace_domain, vec![(zeta, opened_values[0][i][0])]))
            },
        );
        // let coms_to_verify = vec![
        //     (commitments.stage1_trace, stage1_trace_evaluations),
        //     (commitments.quotient_chunks, quotient_chunks_evaluations),
        // ];
        // pcs.verify(coms_to_verify, opening_proof, &mut challenger)
        //     .map_err(VerificationError::InvalidOpeningArgument)?;

        Ok(())
    }
}

// Compute a fingerprint of the coefficients in reverse using Horner's method:
fn fingerprint_reverse<F: Field, Iter: Iterator<Item = F>>(r: F, coeffs: Iter) -> F {
    coeffs.fold(F::ZERO, |acc, coeff| acc * r + coeff)
}

// TODO take stage 2 traces and update the accumulator
fn quotient_values<A, Mat>(
    air: &A,
    public_values: &Vec<Val>,
    trace_domain: Domain<StarkConfig>,
    quotient_domain: Domain<StarkConfig>,
    trace_on_quotient_domain: Mat,
    alpha: ExtVal,
    constraint_count: usize,
) -> Vec<ExtVal>
where
    A: for<'a> Air<ProverConstraintFolder<'a, StarkConfig>>,
    Mat: Matrix<Val> + Sync,
{
    let quotient_size = quotient_domain.size();
    let width = trace_on_quotient_domain.width();
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    for _ in quotient_size..PackedVal::WIDTH {
        sels.is_first_row.push(Val::default());
        sels.is_last_row.push(Val::default());
        sels.is_transition.push(Val::default());
        sels.inv_vanishing.push(Val::default());
    }

    let mut alpha_powers = alpha.powers().collect_n(constraint_count);
    alpha_powers.reverse();

    let decomposed_alpha_powers: Vec<_> = (0..<ExtVal as BasedVectorSpace<Val>>::DIMENSION)
        .map(|i| {
            alpha_powers
                .iter()
                .map(|x| x.as_basis_coefficients_slice()[i])
                .collect()
        })
        .collect();
    (0..quotient_size)
        .into_par_iter()
        .step_by(PackedVal::WIDTH)
        .flat_map_iter(|i_start| {
            let i_range = i_start..i_start + PackedVal::WIDTH;

            let is_first_row = *PackedVal::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_vanishing = *PackedVal::from_slice(&sels.inv_vanishing[i_range]);

            let main = RowMajorMatrix::new(
                trace_on_quotient_domain.vertically_packed_row_pair(i_start, next_step),
                width,
            );

            let accumulator = PackedExtVal::ZERO;
            let mut folder = ProverConstraintFolder {
                main: main.as_view(),
                public_values,
                is_first_row,
                is_last_row,
                is_transition,
                alpha_powers: &alpha_powers,
                decomposed_alpha_powers: &decomposed_alpha_powers,
                accumulator,
                constraint_index: 0,
            };
            air.eval(&mut folder);

            let quotient = folder.accumulator * inv_vanishing;

            (0..core::cmp::min(quotient_size, PackedVal::WIDTH)).map(move |idx_in_packing| {
                ExtVal::from_basis_coefficients_fn(|coeff_idx| {
                    <PackedExtVal as BasedVectorSpace<PackedVal>>::as_basis_coefficients_slice(
                        &quotient,
                    )[coeff_idx]
                        .as_slice()[idx_in_packing]
                })
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use p3_air::{AirBuilderWithPublicValues, BaseAir};

    use crate::types::{FriParameters, new_stark_config};

    use super::*;
    #[test]
    fn multi_stark_test() {
        enum CS {
            Pythagorean,
            Complex,
        }
        impl<F> BaseAir<F> for CS {
            fn width(&self) -> usize {
                match self {
                    CS::Pythagorean => 3,
                    CS::Complex => 6,
                }
            }
        }
        impl<F> BaseAirWithPublicValues<F> for CS {
            fn num_public_values(&self) -> usize {
                MIN_IO_SIZE
            }
        }
        impl<AB: AirBuilderWithPublicValues> Air<AB> for CS {
            fn eval(&self, builder: &mut AB) {
                match self {
                    CS::Pythagorean => {
                        let main = builder.main();
                        let local = main.row_slice(0).unwrap();
                        let expr1 = local[0] * local[0] + local[1] * local[1];
                        let expr2 = local[2] * local[2];
                        // this extra `local[0]` multiplication is there to increase the maximum constraint degree
                        builder.assert_eq(local[0] * expr1, local[0] * expr2);
                    }
                    CS::Complex => {
                        let main = builder.main();
                        let local = main.row_slice(0).unwrap();
                        // (a + ib)(c + id) = (ac - bd) + i(ad + bc)
                        let expr1 = local[0] * local[2] - local[1] * local[3];
                        let expr2 = local[4];
                        let expr3 = local[0] * local[3] + local[1] * local[2];
                        let expr4 = local[5];
                        builder.assert_eq(expr1, expr2);
                        builder.assert_eq(expr3, expr4);
                    }
                }
            }
        }
        let pythagorean_circuit = Circuit::from_air_single_stage(CS::Pythagorean).unwrap();
        let complex_circuit = Circuit::from_air_single_stage(CS::Complex).unwrap();
        let system = System::new(
            [
                ("pythagorean", pythagorean_circuit),
                ("complex", complex_circuit),
            ]
            .into_iter(),
        );
        let f = Val::from_u32;
        let witness = SystemWitness {
            circuits: vec![
                CircuitWitness {
                    trace: RowMajorMatrix::new(
                        [3, 4, 5, 5, 12, 13, 8, 15, 17, 7, 24, 25].map(f).to_vec(),
                        3,
                    ),
                },
                CircuitWitness {
                    trace: RowMajorMatrix::new(
                        [4, 2, 3, 1, 10, 10, 3, 2, 5, 1, 13, 13].map(f).to_vec(),
                        6,
                    ),
                },
            ],
        };
        // lookup arguments not yet implemented so the claim doesn't matter
        let dummy_claim = Claim {
            circuit_name: "complex",
            args: vec![],
        };
        let fri_parameters = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 64,
            proof_of_work_bits: 0,
        };
        let config = new_stark_config(fri_parameters);
        let proof = system.prove(&config, dummy_claim, witness);
        system.verify(&config, &proof).unwrap();
    }
}
