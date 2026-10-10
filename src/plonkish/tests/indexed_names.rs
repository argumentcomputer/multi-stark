use super::*;
use crate::plonkish::builder::InputName;
use std::{mem::size_of, time::Instant};

struct Fixture {
    circuit: Circuit<Val>,
    inputs: Vec<(Value, Val)>,
}

impl Fixture {
    fn new(rounds: usize) -> Self {
        let mut builder = CircuitBuilder::<Val>::new();
        let bytes =
            builder.fixed_table("bytes", (0..=255).map(|i| vec![Val::from_u16(i)]).collect());
        let mut inputs = Vec::new();
        for i in 0..rounds {
            let constant = builder.constant(Val::from_usize(17 + i % 31));
            let bit = builder.input("choice");
            inputs.push((bit, Val::from_usize(i % 2)));
            let bit = builder.assert_bool(bit);
            let byte = builder.input("byte");
            inputs.push((byte, Val::from_usize(7 + i % 249)));
            builder.lookup(bytes, &[byte]);
            let wide = builder.input("wide");
            inputs.push((wide, Val::from_u64(P - 1 - (i % 257) as u64)));
            let product = builder.mul(wide, byte);
            let shifted = builder.add(product, constant);
            let selected = builder.select(bit, shifted, byte);
            let copied = builder.hint("copy", &[selected], |values| Ok(values[0]));
            builder.assert_equal(copied, selected);
            builder.constant(Val::from_usize(333 + i % 37));
            if i + 1 == rounds {
                builder.expose_public(copied);
            }
        }
        Self {
            circuit: builder.finish(),
            inputs,
        }
    }

    fn assignment(&self) -> Assignment<Val> {
        let mut witness = self.circuit.witness();
        for &(wire, value) in &self.inputs {
            witness.set(wire, value).unwrap();
        }
        witness.generate().unwrap()
    }
}

fn translate<const INDEXED: bool>(
    source: &Circuit<Val>,
    builder: &mut CircuitBuilder<Scalar>,
) -> GoldilocksInputs {
    if INDEXED {
        GoldilocksCircuit::translate(source, builder, false)
    } else {
        GoldilocksCircuit::translate_with_inputs(source, builder, false, |builder, i| {
            builder.input(format!("goldilocks[{i}]"))
        })
    }
}

#[derive(Clone, Copy)]
struct Timing {
    census_seconds: f64,
    emission_seconds: f64,
    total_seconds: f64,
}

fn preallocated<const INDEXED: bool>(source: &Circuit<Val>) -> (GoldilocksCircuit, Timing) {
    let started = Instant::now();
    let stats = {
        let mut builder = CircuitBuilder::counting();
        translate::<INDEXED>(source, &mut builder);
        builder.stats()
    };
    let census_seconds = started.elapsed().as_secs_f64();
    let emission_started = Instant::now();
    let mut builder = CircuitBuilder::new();
    builder.reserve(stats);
    let inputs = translate::<INDEXED>(source, &mut builder);
    assert_eq!(builder.stats(), stats);
    let translated = GoldilocksCircuit {
        circuit: builder.finish(),
        inputs,
    };
    let emission_seconds = emission_started.elapsed().as_secs_f64();
    let total_seconds = started.elapsed().as_secs_f64();
    (
        translated,
        Timing {
            census_seconds,
            emission_seconds,
            total_seconds,
        },
    )
}

fn assert_layout(expected: &GoldilocksCircuit, actual: &GoldilocksCircuit) {
    assert_eq!(expected.circuit.stats(), actual.circuit.stats());
    assert_eq!(expected.inputs.is_input, actual.inputs.is_input);
    assert_eq!(
        expected
            .inputs
            .wires
            .iter()
            .map(|v| v.index)
            .collect::<Vec<_>>(),
        actual
            .inputs
            .wires
            .iter()
            .map(|v| v.index)
            .collect::<Vec<_>>()
    );
    for (left, right) in expected.circuit.recipes.iter().zip(&actual.circuit.recipes) {
        match (left, right) {
            (Recipe::Input(a), Recipe::Input(b))
            | (Recipe::Arithmetic(a), Recipe::Arithmetic(b))
            | (Recipe::Hint(a), Recipe::Hint(b)) => assert_eq!(a, b),
            (Recipe::Constant(a), Recipe::Constant(b)) => assert_eq!(a, b),
            _ => panic!("different recipe kind"),
        }
    }
    for (left, right) in expected.circuit.gates.iter().zip(&actual.circuit.gates) {
        assert_eq!(left.wires.map(|v| v.index), right.wires.map(|v| v.index));
        assert_eq!(left.coefficients, right.coefficients);
    }
    for (left, right) in expected.circuit.tables.iter().zip(&actual.circuit.tables) {
        assert_eq!(left.name, right.name);
        assert_eq!(left.rows, right.rows);
    }
    for (left, right) in expected.circuit.lookups.iter().zip(&actual.circuit.lookups) {
        assert_eq!(left.table.index, right.table.index);
        assert_eq!(
            left.values.iter().map(|v| v.index).collect::<Vec<_>>(),
            right.values.iter().map(|v| v.index).collect::<Vec<_>>()
        );
    }
    for (left, right) in expected.circuit.hints.iter().zip(&actual.circuit.hints) {
        assert_eq!(
            (&left.name, &left.dependencies, left.start, left.len),
            (&right.name, &right.dependencies, right.start, right.len)
        );
    }
    assert_eq!(
        expected
            .circuit
            .publics
            .iter()
            .map(|v| v.index)
            .collect::<Vec<_>>(),
        actual
            .circuit
            .publics
            .iter()
            .map(|v| v.index)
            .collect::<Vec<_>>()
    );
    for (left, right) in expected
        .circuit
        .input_names
        .iter()
        .zip(&actual.circuit.input_names)
    {
        assert_eq!(left.to_string(), right.to_string());
    }
}

fn assign(translated: &GoldilocksCircuit, source: &Assignment<Val>) -> Assignment<Scalar> {
    let mut witness = translated.circuit.witness();
    translated.assign(source, &mut witness).unwrap();
    witness.generate().unwrap()
}

#[test]
fn indexed_names_preserve_translation_layout_values_and_missing_inputs() {
    let fixture = Fixture::new(5);
    let source = fixture.assignment();
    let (expected, _) = preallocated::<false>(&fixture.circuit);
    let actual = GoldilocksCircuit::new_preallocated(&fixture.circuit);
    assert_layout(&expected, &actual);
    let old_assignment = assign(&expected, &source);
    let new_assignment = assign(&actual, &source);
    assert_eq!(old_assignment.values(), new_assignment.values());
    assert_eq!(
        old_assignment.public_values(),
        new_assignment.public_values()
    );
    let indices: Vec<_> = actual
        .inputs
        .is_input
        .iter()
        .enumerate()
        .filter_map(|(index, &input)| input.then_some(index))
        .collect();
    assert!(indices.windows(2).any(|pair| pair[1] != pair[0] + 1));
    for missing in [
        indices[0],
        indices[indices.len() / 2],
        *indices.last().unwrap(),
    ] {
        for translated in [&expected, &actual] {
            let mut witness = translated.circuit.witness();
            for &index in &indices {
                if index != missing {
                    witness
                        .set(
                            translated.inputs.wires[index],
                            Scalar::from_u64(source.values()[index].as_canonical_u64()),
                        )
                        .unwrap();
                }
            }
            assert_eq!(
                witness.generate().err(),
                Some(WitnessError::MissingInput {
                    name: format!("goldilocks[{missing}]")
                })
            );
        }
    }
}

#[test]
fn indexed_names_preserve_owned_names_and_lazy_census() {
    struct UnusedName;
    impl From<UnusedName> for String {
        fn from(_: UnusedName) -> Self {
            panic!("counting must not materialize an input name")
        }
    }
    let mut census = CircuitBuilder::<Scalar>::counting();
    census.input(UnusedName);
    census.input_indexed("goldilocks", usize::MAX);
    assert_eq!(census.stats().inputs, 2);
    for name in ["", "input α[17]"] {
        let mut builder = CircuitBuilder::<Scalar>::new();
        builder.input(name);
        assert_eq!(
            builder.finish().witness().generate().err(),
            Some(WitnessError::MissingInput { name: name.into() })
        );
    }
    let mut builder = CircuitBuilder::<Scalar>::new();
    builder.input_indexed("goldilocks", usize::MAX);
    assert_eq!(
        builder.finish().witness().generate().err(),
        Some(WitnessError::MissingInput {
            name: format!("goldilocks[{}]", usize::MAX)
        })
    );
}

#[test]
fn indexed_names_preserve_verified_kzg_bytes() {
    use crate::{
        ark_adapter::{KzgConfig, Srs, compact::FixedProofCodec},
        system::{System, SystemWitness},
    };
    use std::sync::Arc;
    let fixture = Fixture::new(2);
    let source = fixture.assignment();
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(1 << 16, b"indexed-input-names")),
        4,
    );
    let mut proofs = Vec::new();
    for indexed in [false, true] {
        let translated = if indexed {
            GoldilocksCircuit::new_preallocated(&fixture.circuit)
        } else {
            preallocated::<false>(&fixture.circuit).0
        };
        let assignment = assign(&translated, &source);
        let compiled = translated
            .circuit
            .lower_to_multi_stark(Scalar::from_u8(88))
            .unwrap();
        let claims = compiled.claims(assignment.public_values()).unwrap();
        let refs = claims.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let traces = compiled.traces(&assignment).unwrap();
        let (system, key) = System::new(config.clone(), compiled.circuit_inputs());
        let proof =
            system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(traces, &system));
        system.verify_multiple_claims(&refs, &proof).unwrap();
        proofs.push(
            FixedProofCodec::new(&system, &proof.log_degrees)
                .unwrap()
                .encode(&proof)
                .unwrap(),
        );
    }
    assert_eq!(proofs[0], proofs[1]);
    println!(
        "INDEXED_NAMES_PROOF bytes={} blake3={}",
        proofs[0].len(),
        blake3::hash(&proofs[0])
    );
}

fn name_storage(circuit: &Circuit<Scalar>) -> (usize, usize) {
    let entries = circuit.input_names.capacity() * size_of::<InputName>();
    let owned = circuit
        .input_names
        .iter()
        .map(|name| match name {
            InputName::Owned(name) => name.capacity(),
            InputName::Indexed { .. } => 0,
        })
        .sum();
    (entries, owned)
}

#[test]
#[ignore = "isolated CPU foreign translation naming comparison"]
fn indexed_names_translation_benchmark() {
    use std::hint::black_box;
    let fixture = Fixture::new(1 << 15);
    let expected = preallocated::<false>(&fixture.circuit).0;
    let actual = preallocated::<true>(&fixture.circuit).0;
    assert_layout(&expected, &actual);
    let (entries, owned) = name_storage(&actual.circuit);
    assert_eq!(owned, 0);
    let (eager_entries, eager_owned) = name_storage(&expected.circuit);
    let old_entries = expected.circuit.input_names.capacity() * size_of::<String>();
    assert!(entries < old_entries + eager_owned);
    println!(
        "INDEXED_NAMES_MEMORY string_entry_bytes={} input_name_entry_bytes={} inputs={} legacy_string_entries_bytes={} eager_input_name_entries_bytes={} eager_name_owned_capacity_bytes={} indexed_name_entries_bytes={} indexed_name_owned_capacity_bytes=0 scope=retained_name_storage_excluding_allocator_overhead",
        size_of::<String>(),
        size_of::<InputName>(),
        actual.circuit.stats().inputs,
        old_entries,
        eager_entries,
        eager_owned,
        entries
    );
    drop(expected);
    drop(actual);
    for iteration in 0..5 {
        for indexed in if iteration % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let (translated, timing) = if indexed {
                preallocated::<true>(black_box(&fixture.circuit))
            } else {
                preallocated::<false>(black_box(&fixture.circuit))
            };
            let stats = black_box(translated.circuit.stats());
            println!(
                "INDEXED_NAMES_BENCH iteration={iteration} indexed={indexed} source_values={} inputs={} values={} gates={} census_seconds={:.9} emission_seconds={:.9} total_seconds={:.9}",
                fixture.circuit.num_values(),
                stats.inputs,
                stats.values,
                stats.gates,
                timing.census_seconds,
                timing.emission_seconds,
                timing.total_seconds
            );
        }
    }
}
