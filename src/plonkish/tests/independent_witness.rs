use super::*;
use crate::{
    ark_adapter::field::Scalar,
    plonkish::{CircuitBuilder, foreign::GoldilocksCircuit},
    traits::Algebra,
    types::Val,
};

const P: u64 = 0xffff_ffff_0000_0001;

struct Fixture {
    circuit: Circuit<Val>,
    assignment: Assignment<Val>,
}

impl Fixture {
    fn new(rounds: usize) -> Self {
        let mut b = CircuitBuilder::<Val>::new();
        let bytes = b.fixed_table("byte", (0..=255).map(|i| vec![Val::from_u16(i)]).collect());
        let mut inputs = Vec::new();
        let boundary = [0, 1, P - 1, u64::from(u32::MAX), 1 << 32, P - (1 << 32)];
        for i in 0..rounds {
            let constant = b.constant(Val::from_usize(19 + i % 31));
            let wide = b.input("wide");
            let factor = b.input("factor");
            let byte = b.input("byte");
            let bit = b.input("bit");
            inputs.extend([
                (wide, Val::from_u64(boundary[i % boundary.len()])),
                (factor, Val::from_u64(P - 3 - (i % 127) as u64)),
                (byte, Val::from_usize(i % 256)),
                (bit, Val::from_usize(i % 2)),
            ]);
            b.lookup(bytes, &[byte]);
            let bit = b.assert_bool(bit);
            let product = b.mul(wide, factor);
            let shifted = b.add(product, constant);
            let chosen = b.select(bit, shifted, byte);
            let copies = b.hint_many("two copies", &[chosen], |values| Ok([values[0]; 2]));
            for copy in copies {
                b.assert_equal(copy, chosen);
            }
            b.constant(Val::from_usize(333 + i % 37));
            if i + 1 == rounds {
                b.expose_public(copies[0]);
            }
        }
        let circuit = b.finish();
        let mut witness = circuit.witness();
        for (input, value) in inputs {
            witness.set(input, value).unwrap();
        }
        let assignment = witness.generate().unwrap();
        Self {
            circuit,
            assignment,
        }
    }

    fn translate(&self, chunk_size: usize) -> GoldilocksCircuit {
        let mut b = CircuitBuilder::new();
        let inputs =
            GoldilocksCircuit::translate_with_chunk_size(&self.circuit, &mut b, false, chunk_size);
        GoldilocksCircuit {
            circuit: b.finish(),
            inputs,
        }
    }

    fn witness<'a>(&self, translated: &'a GoldilocksCircuit) -> Witness<'a, Scalar> {
        let mut witness = translated.circuit.witness();
        translated.assign(&self.assignment, &mut witness).unwrap();
        witness
    }
}

fn assert_relation(left: &Circuit<Scalar>, right: &Circuit<Scalar>) {
    assert_eq!(left.stats(), right.stats());
    for (left, right) in left.recipes.iter().zip(&right.recipes) {
        match (left, right) {
            (Recipe::Input(a), Recipe::Input(b))
            | (Recipe::Arithmetic(a), Recipe::Arithmetic(b))
            | (Recipe::Hint(a), Recipe::Hint(b)) => assert_eq!(a, b),
            (Recipe::Constant(a), Recipe::Constant(b)) => assert_eq!(a, b),
            _ => panic!("different recipe kind"),
        }
    }
    for (left, right) in left.gates.iter().zip(&right.gates) {
        assert_eq!(left.wires.map(Value::index), right.wires.map(Value::index));
        assert_eq!(left.coefficients, right.coefficients);
    }
    for (left, right) in left.tables.iter().zip(&right.tables) {
        assert_eq!(left.name, right.name);
        assert_eq!(left.rows, right.rows);
    }
    for (left, right) in left.lookups.iter().zip(&right.lookups) {
        assert_eq!(left.table.index, right.table.index);
        assert_eq!(
            left.values.iter().map(|v| v.index).collect::<Vec<_>>(),
            right.values.iter().map(|v| v.index).collect::<Vec<_>>()
        );
    }
    for (left, right) in left.hints.iter().zip(&right.hints) {
        assert_eq!(
            (&left.name, &left.dependencies, left.start, left.len),
            (&right.name, &right.dependencies, right.start, right.len)
        );
    }
    assert_eq!(
        left.publics.iter().map(|v| v.index).collect::<Vec<_>>(),
        right.publics.iter().map(|v| v.index).collect::<Vec<_>>()
    );
    assert_eq!(
        left.input_names
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        right
            .input_names
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
}

#[test]
fn independent_advice_preserves_translation_assignments_and_errors() {
    let fixture = Fixture::new(24);
    let serial = fixture.translate(0);
    let scheduled = fixture.translate(64);
    assert_relation(&serial.circuit, &scheduled.circuit);
    assert_eq!(
        GoldilocksCircuit::estimate(&fixture.circuit),
        scheduled.circuit.stats()
    );
    assert!(serial.circuit.witness_chunks.is_empty());
    assert_eq!(
        scheduled.circuit.witness_chunks.len() > 1,
        cfg!(feature = "parallel")
    );
    let expected = fixture
        .witness(&scheduled)
        .generate_with(arithmetic_value)
        .unwrap();
    let actual = fixture.witness(&scheduled).generate().unwrap();
    assert_eq!(expected.values(), actual.values());
    assert_eq!(expected.public_values(), actual.public_values());
    assert_eq!(
        fixture.witness(&serial).generate().unwrap().values(),
        actual.values()
    );
    assert!(scheduled.circuit.hints.iter().any(|hint| hint.len > 1));
    let count = scheduled.circuit.input_names.len();
    for missing in [0, count / 2, count - 1] {
        let mut expected = fixture.witness(&scheduled);
        expected.inputs[missing] = None;
        let mut actual = fixture.witness(&scheduled);
        actual.inputs[missing] = None;
        let expected = expected.generate_with(arithmetic_value).err();
        assert_eq!(actual.generate().err(), expected);
        assert_eq!(
            expected,
            Some(WitnessError::MissingInput {
                name: scheduled.circuit.input_names[missing].to_string(),
            })
        );
    }
    let mut expected = fixture.witness(&scheduled);
    let mut actual = fixture.witness(&scheduled);
    expected.inputs[0] = Some(Scalar::from_u64(P));
    actual.inputs[0] = Some(Scalar::from_u64(P));
    let error = expected.generate_with(arithmetic_value).err();
    assert!(
        error.is_some(),
        "noncanonical Goldilocks alias must be rejected"
    );
    assert_eq!(actual.generate().err(), error);
}

fn pure_blocks() -> (CircuitBuilder<Scalar>, Vec<usize>) {
    let mut b = CircuitBuilder::<Scalar>::new();
    let mut ends = Vec::new();
    let root = b.input("first");
    let constant = b.constant(Scalar::from_u8(7));
    for (i, name) in ["first hint", "middle hint", "last hint"]
        .into_iter()
        .enumerate()
    {
        let input = if i == 0 { root } else { b.input(name) };
        let outputs = b.hint_many(name, &[input, root, constant], |args| {
            if args[0] == Scalar::ZERO {
                Err("zero advice input".into())
            } else {
                Ok([args[0], args[1], args[2]])
            }
        });
        b.assert_equal(outputs[0], input);
        b.assert_equal(outputs[1], root);
        b.assert_equal(outputs[2], constant);
        let derived = b.add(outputs[0], constant);
        b.expose_public(derived);
        ends.push(b.stats().values);
    }
    (b, ends)
}

#[test]
fn independent_advice_preserves_original_recipe_error_precedence() {
    let (mut builder, ends) = pure_blocks();
    assert!(builder.certify_witness_chunks(ends));
    let circuit = builder.finish();
    for first in [None, Some(Scalar::ZERO), Some(Scalar::ONE)] {
        for middle in [None, Some(Scalar::ZERO), Some(Scalar::ONE)] {
            for last in [None, Some(Scalar::ZERO), Some(Scalar::ONE)] {
                let witness = || {
                    let mut witness = circuit.witness();
                    witness.inputs = vec![first, middle, last];
                    witness
                };
                let expected = witness().generate_with(arithmetic_value);
                for actual in [witness().generate(), witness().generate_scheduled()] {
                    match (&expected, actual) {
                        (Ok(expected), Ok(actual)) => {
                            assert_eq!(actual.values(), expected.values());
                            assert_eq!(actual.public_values(), expected.public_values());
                        }
                        (Err(expected), Err(actual)) => assert_eq!(&actual, expected),
                        _ => panic!("scheduled and serial error behavior differs"),
                    }
                }
            }
        }
    }
}

#[test]
fn independent_advice_certificate_rejects_incomplete_or_dependent_chunks() {
    let fixture = || {
        let (builder, ends) = pure_blocks();
        (builder.finish(), ends)
    };
    let (circuit, ends) = fixture();
    assert!(circuit.independent_witness_chunks(&ends));
    for invalid in [
        vec![],
        vec![circuit.num_values()],
        vec![0, circuit.num_values()],
        vec![ends[0], ends[0], circuit.num_values()],
        vec![ends[0], circuit.num_values() - 1],
        vec![circuit.hints[0].start + 1, circuit.num_values()],
    ] {
        assert!(!circuit.independent_witness_chunks(&invalid));
    }
    for invalid in 0..10 {
        let (mut circuit, ends) = fixture();
        let hint = circuit.hints[0].start;
        let arithmetic = circuit
            .recipes
            .iter()
            .position(|recipe| matches!(recipe, Recipe::Arithmetic(_)))
            .unwrap();
        let Recipe::Arithmetic(gate) = circuit.recipes[arithmetic] else {
            unreachable!()
        };
        match invalid {
            0 => circuit.hints[0].len = 0,
            1 => circuit.hints[0].len = usize::MAX,
            2 => circuit.hints[0].start += 1,
            3 => circuit.hints[0].dependencies[0] = hint,
            4 => circuit.recipes[hint + 1] = Recipe::Constant(Scalar::ZERO),
            5 => circuit.hints[1].dependencies[0] = hint,
            6 => circuit.gates[gate].wires[0].owner = u64::MAX,
            7 => circuit.gates[gate].wires[2].index += 1,
            8 => circuit.gates[gate].coefficients[3] = Scalar::ZERO,
            9 => circuit.gates[gate].wires[0].index = arithmetic,
            _ => unreachable!(),
        }
        assert!(
            !circuit.independent_witness_chunks(&ends),
            "invalid case {invalid}"
        );
    }
}

#[test]
fn independent_advice_keeps_generic_nonfresh_expanded_and_appended_plans_serial() {
    use std::sync::{Arc, Mutex};
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut b = CircuitBuilder::<Scalar>::new();
    for i in 0..3 {
        let calls = calls.clone();
        b.hint("ordered callback", &[], move |_| {
            calls.lock().unwrap().push(i);
            Ok(Scalar::ZERO)
        });
    }
    let circuit = b.finish();
    assert!(circuit.witness_chunks.is_empty());
    circuit.witness().generate().unwrap();
    assert_eq!(*calls.lock().unwrap(), [0, 1, 2]);

    let fixture = Fixture::new(8);
    for expanded in [false, true] {
        let mut b = CircuitBuilder::new();
        if !expanded {
            b.hint("preexisting callback", &[], |_| Ok(Scalar::ZERO));
        }
        GoldilocksCircuit::translate_with_chunk_size(&fixture.circuit, &mut b, expanded, 1);
        assert!(b.finish().witness_chunks.is_empty());
    }
    assert!(
        Fixture::new(0)
            .translate(usize::MAX)
            .circuit
            .witness_chunks
            .is_empty()
    );
    let mut census = CircuitBuilder::counting();
    assert!(!census.fresh_witness_plan());
    GoldilocksCircuit::translate_with_chunk_size(&fixture.circuit, &mut census, false, 1);
    assert_eq!(census.stats(), fixture.translate(0).circuit.stats());

    let (mut b, ends) = pure_blocks();
    assert!(b.certify_witness_chunks(ends));
    let appended = b.hint("appended advice", &[], |_| Ok(Scalar::from_u8(23)));
    let constant = b.constant(Scalar::from_u8(23));
    b.assert_equal(appended, constant);
    b.expose_public(appended);
    let circuit = b.finish();
    assert_ne!(circuit.witness_chunks.last(), Some(&circuit.num_values()));
    let witness = || {
        let mut witness = circuit.witness();
        witness.inputs.fill(Some(Scalar::ONE));
        witness
    };
    let expected = witness().generate_with(arithmetic_value).unwrap();
    let actual = witness().generate().unwrap();
    assert_eq!(actual.values(), expected.values());
    assert_eq!(actual.public_values().last(), Some(&Scalar::from_u8(23)));
}

#[test]
fn independent_advice_preserves_verified_kzg_bytes() {
    use crate::{
        ark_adapter::{KzgConfig, Srs, compact::FixedProofCodec},
        system::{System, SystemWitness},
    };
    use std::sync::Arc;
    let fixture = Fixture::new(2);
    let translated = fixture.translate(16);
    assert_eq!(
        translated.circuit.witness_chunks.len() > 1,
        cfg!(feature = "parallel")
    );
    let assignments = [
        fixture
            .witness(&translated)
            .generate_with(arithmetic_value)
            .unwrap(),
        fixture.witness(&translated).generate().unwrap(),
    ];
    assert_eq!(assignments[0].values(), assignments[1].values());
    let compiled = translated
        .circuit
        .lower_to_multi_stark(Scalar::from_u8(89))
        .unwrap();
    let claims = compiled.claims(assignments[0].public_values()).unwrap();
    let refs = claims.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(
            1 << 16,
            b"independent-witness-advice",
        )),
        4,
    );
    let (system, key) = System::new(config, compiled.circuit_inputs());
    let mut proofs = Vec::new();
    for assignment in assignments {
        let traces = compiled.traces(&assignment).unwrap();
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
        "INDEPENDENT_ADVICE_PROOF bytes={} blake3={}",
        proofs[0].len(),
        blake3::hash(&proofs[0])
    );
}

#[test]
#[ignore = "isolated serial versus certified parallel foreign witness comparison"]
fn independent_advice_assignment_benchmark() {
    use std::{hint::black_box, time::Instant};
    let fixture = Fixture::new(1 << 13);
    let started = Instant::now();
    let translated = GoldilocksCircuit::new_preallocated(&fixture.circuit);
    let compile_seconds = started.elapsed().as_secs_f64();
    assert_eq!(
        translated.circuit.witness_chunks.len() > 1,
        cfg!(feature = "parallel")
    );
    let expected = fixture
        .witness(&translated)
        .generate_with(arithmetic_value)
        .unwrap();
    let evaluate = |parallel| {
        let witness = fixture.witness(black_box(&translated));
        let started = Instant::now();
        let assignment = if parallel {
            witness.generate().unwrap()
        } else {
            witness.generate_with(arithmetic_value).unwrap()
        };
        let seconds = started.elapsed().as_secs_f64();
        assert_eq!(black_box(assignment.values()), expected.values());
        assert_eq!(assignment.public_values(), expected.public_values());
        seconds
    };
    evaluate(false);
    evaluate(true);
    println!(
        "INDEPENDENT_ADVICE_SCOPE values={} hints={} chunks={} chunk_capacity_bytes={} output_bytes={} compile_seconds={compile_seconds:.9} parallel_feature={}",
        translated.circuit.num_values(),
        translated.circuit.hints.len(),
        translated.circuit.witness_chunks.len(),
        translated.circuit.witness_chunks.capacity() * std::mem::size_of::<usize>(),
        translated.circuit.num_values() * std::mem::size_of::<Scalar>(),
        cfg!(feature = "parallel")
    );
    for iteration in 0..5 {
        let mut seconds = [0.0; 2];
        for parallel in if iteration % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            seconds[usize::from(parallel)] = evaluate(parallel);
        }
        println!(
            "INDEPENDENT_ADVICE_BENCH iteration={iteration} serial_seconds={:.9} scheduled_seconds={:.9}",
            seconds[0], seconds[1]
        );
    }
}
