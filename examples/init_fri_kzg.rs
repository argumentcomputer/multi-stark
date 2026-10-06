//! KZG census for the saved Init intermediate proof; no full-size setup or proof.
#[cfg(feature = "kzg")]
#[path = "support/init_fri.rs"]
mod init_fri;

#[cfg(not(feature = "kzg"))]
fn main() {
    eprintln!("enable --features kzg,parallel");
}

#[cfg(feature = "kzg")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    census::run()
}

#[cfg(feature = "kzg")]
mod census {
    use multi_stark::{
        ark_adapter::{KzgConfig, Scalar, Srs},
        expr::{Expr, RowOffset, Source},
        graph::Node,
        lookup::{Lookup, MAX_LOOKUP_GROUP, logup_max_degree, stage2_width},
        plonkish::{
            Circuit, CircuitBuilder,
            foreign::GoldilocksCircuit,
            gadgets::{ByteGadgets, blake3},
            verifier::*,
        },
        system::{CircuitInputs, System},
        traits::{Algebra, Field, TwoAdicField},
        types::Val,
    };
    use p3_field::PrimeField64;
    use p3_matrix::{Matrix, dense::RowMajorMatrix};
    use std::{collections::BTreeSet, fs, path::PathBuf, sync::Arc, time::Instant};

    struct Trace {
        name: String,
        height: usize,
        input: CircuitInputs<Scalar>,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Cost {
        advice: usize,
        fixed: usize,
        lookup: usize,
        group: usize,
        quotient: usize,
        points: usize,
        fields: usize,
    }

    // Compile the real lowering's constraints. Only fixed matrix values are
    // irrelevant here: widths/degrees/rotations are properties of its graph.
    fn cost(input: &CircuitInputs<Scalar>, height: usize, srs: usize, budget: usize) -> Cost {
        let fixed = input.preprocessed.as_ref().unwrap().width();
        let mut tiny = input.clone();
        tiny.preprocessed = Some(RowMajorMatrix::new(vec![Scalar::ZERO; 2 * fixed], fixed));
        tiny.lookup_group_size = 1;
        let (system, _) = System::new(
            KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(2, b"shape-only")), 8),
            [tiny],
        );
        let graph = &system.circuits[0].graph;
        let multiplier = if height < srs { 2 } else { 1 };
        let (_, quotient, group, lookup) = (1..=MAX_LOOKUP_GROUP)
            .filter_map(|group| {
                let degree = graph
                    .max_constraint_degree
                    .max(logup_max_degree(graph, group)) as usize;
                let q = (degree.max(2) - 1).next_power_of_two();
                let width = stage2_width(input.lookups.len(), group, 1);
                (q <= budget).then_some((
                    width * (48 * multiplier + 64) + q * (48 * multiplier + 32),
                    q,
                    group,
                    width,
                ))
            })
            .min()
            .expect("quotient budget");
        let next = |source| {
            usize::from(graph.nodes.iter().any(
                |n| matches!(n, Node::Var(c) if c.source == source && c.offset == RowOffset::Next),
            ))
        };
        Cost {
            advice: input.main_width,
            fixed,
            lookup,
            group,
            quotient,
            points: multiplier * (input.main_width + lookup + quotient),
            fields: input.main_width * (1 + next(Source::Main))
                + fixed * (1 + next(Source::Preprocessed))
                + 2 * lookup
                + quotient,
        }
    }

    // Small representative traces provide the exact column constraints without
    // building the full scalar IR or any full-size preprocessing matrix.
    fn prototypes(source: &Circuit<Val>, expanded: bool) -> Circuit<Scalar> {
        let mut b = CircuitBuilder::new();
        b.fixed_table(
            "u16",
            (0..=u16::MAX).map(|v| vec![Scalar::from_u16(v)]).collect(),
        );
        for table in source.tables() {
            b.fixed_table(
                table.name(),
                table
                    .rows()
                    .iter()
                    .map(|r| {
                        r.iter()
                            .map(|v| Scalar::from_u64(v.as_canonical_u64()))
                            .collect()
                    })
                    .collect(),
            );
        }
        if expanded {
            let _ = ByteGadgets::new(&mut b);
        }
        b.finish()
    }

    fn definitions(
        source: &Circuit<Val>,
        expanded: bool,
        main_height: usize,
        main_rows: usize,
        table_count: usize,
    ) -> Vec<Trace> {
        let proto = prototypes(source, expanded);
        assert_eq!(proto.stats().tables, table_count);
        let main_width = proto.multi_stark_layout().unwrap().main_width;
        let compiled = proto
            .lower_to_multi_stark(Scalar::ONE)
            .unwrap()
            .merge_table_traces(1 << 22)
            .unwrap();
        let main = compiled.circuit_input(0).unwrap();
        let table = compiled.circuit_input(1).unwrap();
        let mut traces = vec![
            Trace {
                name: "arithmetic".into(),
                height: main_height,
                input: main,
            },
            Trace {
                name: "merged_tables".into(),
                height: table.preprocessed.as_ref().unwrap().height(),
                input: table,
            },
        ];
        if !expanded {
            let mut b = CircuitBuilder::<Scalar>::new();
            b.enable_compact_blake3();
            let bytes = ByteGadgets::new(&mut b);
            let input: Vec<_> = (0..64)
                .map(|i| bytes.input(&mut b, &format!("b{i}")))
                .collect();
            let _ = blake3(&mut b, &bytes, &input);
            let compiled = b.finish().lower_to_multi_stark(Scalar::ONE).unwrap();
            assert_eq!(compiled.main_width(), main_width);
            // Hash bridge is an extra fixed column and lookup on the main trace.
            traces[0].input = compiled.circuit_input(0).unwrap();
            let layouts = source.multi_stark_layout().unwrap().custom_traces;
            for (i, &(height, advice, fixed)) in layouts.iter().enumerate() {
                let input = compiled.circuit_input(4 + i).unwrap();
                assert_eq!(
                    (
                        input.main_width,
                        input.preprocessed.as_ref().unwrap().width()
                    ),
                    (advice, fixed)
                );
                traces.push(Trace {
                    name: format!("hash_{i}"),
                    height,
                    input,
                });
            }
        }
        assert!(main_rows <= main_height);
        traces
    }

    // The partitioned lowering adds one fixed activation column and one
    // lookup per arithmetic trace. Shared tables/hash traces remain unchanged.
    fn partition(traces: &[Trace], used_rows: usize, cap: usize) -> Vec<Trace> {
        assert!(cap.is_power_of_two() && cap >= 2 && used_rows > cap);
        let original = &traces[0].input;
        let fixed_width = original.preprocessed.as_ref().unwrap().width();
        let mut result = Vec::new();
        for (index, offset) in (0..used_rows).step_by(cap).enumerate() {
            let mut input = original.clone();
            input.preprocessed = Some(RowMajorMatrix::new(
                vec![Scalar::ZERO; 2 * (fixed_width + 1)],
                fixed_width + 1,
            ));
            input.lookups.push(Lookup::pull(
                Expr::preprocessed(u32::try_from(fixed_width).unwrap()),
                vec![
                    Expr::constant(Scalar::ONE),
                    Expr::constant(Scalar::from_u8(3)),
                    Expr::constant(Scalar::from_usize(index)),
                ],
            ));
            result.push(Trace {
                name: format!("arithmetic_{index}"),
                height: (used_rows - offset).min(cap).max(2).next_power_of_two(),
                input,
            });
        }
        result.extend(traces[1..].iter().map(|t| Trace {
            name: t.name.clone(),
            height: t.height,
            input: t.input.clone(),
        }));
        result
    }

    fn report(traces: &[Trace], mode: &str, budget: usize) {
        let srs = traces.iter().map(|t| t.height).max().unwrap();
        let heights: BTreeSet<_> = traces.iter().map(|t| t.height).collect();
        let mut points = heights.len() + 1;
        let mut fields = traces.len() - 1;
        let mut base_bytes = 0usize;
        let mut max_quotient_inputs = 0usize;
        let mut supported = true;
        for trace in traces {
            let c = cost(&trace.input, trace.height, srs, budget);
            points += c.points;
            fields += c.fields;
            let qdomain = trace.height * c.quotient;
            supported &= qdomain.ilog2() <= u32::try_from(Scalar::TWO_ADICITY).unwrap();
            let arrays = 32 * trace.height * (c.advice + c.fixed + c.lookup + c.quotient);
            let quotient_inputs = 32 * qdomain * (c.advice + c.fixed + c.lookup);
            base_bytes += arrays;
            max_quotient_inputs = max_quotient_inputs.max(quotient_inputs);
            println!(
                "TRACE mode={mode} budget={budget} name={} height={} advice={} fixed={} lookup={} group={} quotient={} quotient_domain={qdomain} base_arrays_bytes={arrays} quotient_inputs_bytes={quotient_inputs}",
                trace.name, trace.height, c.advice, c.fixed, c.lookup, c.group, c.quotient
            );
        }
        let proof_bytes = 5 + 48 * points + 32 * fields;
        let srs_resident = srs * size_of::<ark_bls12_381::G1Affine>()
            + (srs.ilog2() as usize + 3) * size_of::<ark_bls12_381::G2Affine>();
        let srs_compressed = srs * 48 + (srs.ilog2() as usize + 3) * 96;
        println!(
            "TOTAL mode={mode} budget={budget} traces={} proof_bytes={proof_bytes} packet_bytes={} points={points} fields={fields} srs_len={srs} srs_resident_bytes={srs_resident} srs_compressed_bytes={srs_compressed} base_arrays_bytes={base_bytes} max_quotient_inputs_bytes={max_quotient_inputs} supported={supported}",
            traces.len(),
            proof_bytes + 32 + 18 * 8
        );
    }

    pub(super) fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args.len() != 2 {
            return Err("usage: init_fri_kzg <recovered-artifacts> <output-dir>".into());
        }
        let out = PathBuf::from(&args[1]);
        fs::create_dir_all(&out)?;
        fs::write(
            out.join("binary-blake3.txt"),
            ::blake3::hash(&fs::read(std::env::current_exe()?)?)
                .to_hex()
                .as_str(),
        )?;
        let fixture = super::init_fri::load(&PathBuf::from(&args[0]))?;
        let plan =
            VerifierPlan::validate(&fixture.key, fixture.profile, VerifierLimits::default())?;
        let start = Instant::now();
        let (source, inputs) = plan.build(
            fixture.schema,
            ImplementationOptions {
                compact_blake3: true,
            },
        )?;
        let prepared = plan.expand_witness(ProofEnvelope::Ordinary {
            proof: &fixture.proof,
            claims: &fixture.claims,
        })?;
        let mut witness = source.witness();
        inputs.assign_statement(
            &mut witness,
            &Statement {
                claims: fixture.claims,
                messages: vec![],
            },
        )?;
        prepared.assign_proof(&mut witness, &inputs)?;
        let assignment = witness.generate()?;
        assert_eq!(assignment.public_values(), fixture.public);
        drop(assignment);
        drop(prepared);
        drop(inputs);
        println!(
            "Full source witness satisfied; original 18 public words preserved; {:?}",
            start.elapsed()
        );
        let source_layout = source.multi_stark_layout()?;
        let source_stats = source.stats();
        let hash_rows = source_layout.used_rows
            - source_stats.gates
            - source_stats.lookups
            - source_stats.publics
            - 1;
        for expanded in [false, true] {
            let mode = if expanded { "generic" } else { "compact" };
            let stats = if expanded {
                GoldilocksCircuit::estimate_with_expanded_hashes(&source)
            } else {
                GoldilocksCircuit::estimate(&source)
            };
            let rows = stats.gates
                + stats.lookups
                + stats.publics
                + 1
                + if expanded { 0 } else { hash_rows };
            println!(
                "CENSUS mode={mode} values={} gates={} lookups={} tables={} publics={} used_rows={rows} hash_rows={} elapsed_seconds={}",
                stats.values,
                stats.gates,
                stats.lookups,
                stats.tables,
                stats.publics,
                if expanded { 0 } else { hash_rows },
                start.elapsed().as_secs_f64()
            );
            let traces = definitions(
                &source,
                expanded,
                rows.next_power_of_two(),
                rows,
                stats.tables,
            );
            for budget in [2, 4, 8] {
                report(&traces, mode, budget);
            }
            for log in [24, 26, 28] {
                let cap = 1usize << log;
                if rows > cap {
                    let partitioned = partition(&traces, rows, cap);
                    for budget in [2, 4] {
                        report(&partitioned, &format!("{mode}_partition{log}"), budget);
                    }
                }
            }
        }
        println!(
            "Census only. Packet assumes fixed trusted profile plus 18 canonical u64 claims. Memory figures are named dense arrays, not peak RSS. No full KZG proof generated."
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use multi_stark::{
            ark_adapter::{KzgConfig, Srs, compact::FixedProofCodec},
            system::{System, SystemWitness},
        };
        use std::sync::Arc;

        #[test]
        fn census_layout_matches_both_hash_lowerings() {
            let mut b = CircuitBuilder::<Val>::new();
            b.enable_compact_blake3();
            let bytes = ByteGadgets::new(&mut b);
            let input: Vec<_> = (0..65)
                .map(|i| bytes.input(&mut b, &format!("b{i}")))
                .collect();
            for byte in blake3(&mut b, &bytes, &input) {
                b.expose_public(byte.value());
            }
            let source = b.finish();
            for expanded in [false, true] {
                let mapped = if expanded {
                    GoldilocksCircuit::new_with_expanded_hashes(&source)
                } else {
                    GoldilocksCircuit::new(&source)
                };
                let stats = mapped.circuit.stats();
                let layout = mapped.circuit.multi_stark_layout().unwrap();
                let predicted = definitions(
                    &source,
                    expanded,
                    layout.main_height,
                    layout.used_rows,
                    stats.tables,
                );
                let actual = mapped
                    .circuit
                    .lower_to_multi_stark(Scalar::ONE)
                    .unwrap()
                    .merge_table_traces(1 << 22)
                    .unwrap();
                assert_eq!(predicted.len(), actual.num_circuits());
                for (trace, input) in predicted.iter().zip(actual.circuit_inputs()) {
                    assert_eq!(trace.height, input.preprocessed.as_ref().unwrap().height());
                    assert_eq!(trace.input.main_width, input.main_width);
                    assert_eq!(
                        trace.input.preprocessed.as_ref().unwrap().width(),
                        input.preprocessed.as_ref().unwrap().width()
                    );
                    assert_eq!(trace.input.lookups.len(), input.lookups.len());
                    let srs = predicted.iter().map(|t| t.height).max().unwrap();
                    assert_eq!(
                        cost(&trace.input, trace.height, srs, 8),
                        cost(&input, trace.height, srs, 8)
                    );
                }
            }
        }

        #[test]
        fn partition_shape_matches_actual_lowering() {
            let mut b = CircuitBuilder::<Scalar>::new();
            let x = b.input("x");
            for _ in 0..21 {
                b.assert_zero(x);
            }
            let circuit = b.finish();
            let rows = circuit.multi_stark_layout().unwrap().used_rows;
            let unpartitioned = circuit.lower_to_multi_stark(Scalar::ONE).unwrap();
            let traces = vec![Trace {
                name: "arithmetic".into(),
                height: unpartitioned.main_height(),
                input: unpartitioned.circuit_input(0).unwrap(),
            }];
            let predicted = partition(&traces, rows, 8);
            let mut b = CircuitBuilder::<Scalar>::new();
            let x = b.input("x");
            for _ in 0..21 {
                b.assert_zero(x);
            }
            let actual = b
                .finish()
                .lower_to_multi_stark_with_max_height(Scalar::ONE, 8)
                .unwrap();
            assert_eq!(predicted.len(), actual.num_circuits());
            for (trace, input) in predicted.iter().zip(actual.circuit_inputs()) {
                assert_eq!(trace.height, input.preprocessed.as_ref().unwrap().height());
                assert_eq!(
                    cost(&trace.input, trace.height, 8, 8),
                    cost(&input, trace.height, 8, 8)
                );
            }
        }

        #[test]
        fn census_size_matches_verified_compact_proof() {
            let mut b = CircuitBuilder::<Scalar>::new();
            let t = b.fixed_table("small", vec![vec![Scalar::ONE], vec![Scalar::TWO]]);
            let x = b.input("x");
            b.lookup(t, &[x]);
            let y = b.mul(x, x);
            b.expose_public(y);
            let circuit = b.finish();
            let mut witness = circuit.witness();
            witness.set(x, Scalar::TWO).unwrap();
            let assignment = witness.generate().unwrap();
            let lowered = circuit
                .lower_to_multi_stark(Scalar::ONE)
                .unwrap()
                .merge_table_traces(4)
                .unwrap();
            let height = lowered.main_height();
            let definitions = lowered.kzg_circuit_inputs(height, 8).unwrap();
            let mut points = 0;
            let mut fields = definitions.len() - 1;
            let mut heights = BTreeSet::new();
            for input in &definitions {
                let h = input.preprocessed.as_ref().unwrap().height();
                heights.insert(h);
                let c = cost(input, h, height, 8);
                assert_eq!(c.group, input.lookup_group_size);
                points += c.points;
                fields += c.fields;
            }
            let config = KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(height, b"census-test")), 8);
            let (system, key) = System::new(config, definitions);
            let claims = lowered.claims(&[Scalar::from_u8(4)]).unwrap();
            let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
            let proof = system.prove_multiple_claims(
                &key,
                &refs,
                SystemWitness::from_stage_1(lowered.traces(&assignment).unwrap(), &system),
            );
            system.verify_multiple_claims(&refs, &proof).unwrap();
            assert_eq!(proof.opening_proof.0.len(), heights.len() + 1);
            let codec = FixedProofCodec::new(&system, &proof.log_degrees).unwrap();
            let bytes = codec.encode(&proof).unwrap();
            assert_eq!(
                bytes.len(),
                5 + 48 * (points + heights.len() + 1) + 32 * fields
            );
            system
                .verify_multiple_claims(&refs, &codec.decode(&bytes).unwrap())
                .unwrap();
        }
    }
}
