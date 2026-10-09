use multi_stark::{
    plonkish::{CircuitBuilder, ValueSource, WitnessError},
    system::{System, extension_params},
    traits::{Algebra, Field},
    types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val},
};

#[test]
fn translation_metadata_preserves_identity_and_relations() {
    let mut b = CircuitBuilder::<Val>::new();
    let x = b.input("x");
    let y = b.mul(x, x);
    let hint = b.hint("copy", &[y], |v| Ok(v[0]));
    b.assert_equal(y, hint);
    b.expose_public(hint);
    let c = b.finish();
    let sources: Vec<_> = c.value_sources().collect();
    assert_eq!(sources.len(), c.num_values());
    assert!(matches!(sources[0], ValueSource::Constant(v) if v == Val::ZERO));
    assert!(matches!(sources[x.index()], ValueSource::Input));
    assert!(matches!(sources[hint.index()], ValueSource::Hint));
    let ValueSource::Arithmetic(index) = c.value_source(y).unwrap() else {
        panic!("missing arithmetic definition");
    };
    assert_eq!(c.gates()[index].wires, [x, x, y]);

    let mut witness = c.witness();
    witness.set(x, Val::from_u8(3)).unwrap();
    let assignment = witness.generate().unwrap();
    let values = assignment.values(c.id()).unwrap();
    assert_eq!(values[y.index()], Val::from_u8(9));
    c.check_values(values).unwrap();
    let mut forged = values.to_vec();
    forged[hint.index()] += Val::ONE;
    assert!(c.check_values(&forged).is_err());
    assert!(matches!(
        c.check_values(&[]),
        Err(WitnessError::ValueCount { .. })
    ));

    let mut other = CircuitBuilder::<Val>::new();
    let foreign = other.input("x");
    assert!(matches!(
        c.value_source(foreign),
        Err(WitnessError::ForeignValue)
    ));
    assert_eq!(
        assignment.values(other.finish().id()),
        Err(WitnessError::ForeignAssignment)
    );
}

fn describe_hash(b: &mut CircuitBuilder<Val>) {
    let x = b.input("byte");
    let digest = std::array::from_fn(|i| b.input(format!("digest[{i}]")));
    b.constrain_blake3(vec![x], &digest);
    b.expose_public(digest[0]);
}

#[test]
fn sizing_and_reservation_preserve_compact_hash_relations() {
    let mut counter = CircuitBuilder::counting();
    describe_hash(&mut counter);
    let estimate = counter.stats();
    let mut b = CircuitBuilder::new();
    b.reserve(estimate);
    describe_hash(&mut b);
    assert!(b.compact_blake3_enabled());
    assert_eq!(b.stats(), estimate);
    let c = b.finish();
    let (input, output) = c.blake3_calls().next().unwrap();
    let expected = blake3::hash(&[42]);
    let mut witness = c.witness();
    witness.set(input[0], Val::from_u8(42)).unwrap();
    for (&wire, &byte) in output.iter().zip(expected.as_bytes()) {
        witness.set(wire, Val::from_u8(byte)).unwrap();
    }
    let assignment = witness.generate().unwrap();
    let mut forged = assignment.values(c.id()).unwrap().to_vec();
    forged[output[0].index()] += Val::ONE;
    assert_eq!(c.check_values(&forged), Err(WitnessError::HashMismatch));
    forged[input[0].index()] = Val::from_u16(256);
    assert!(c.check_values(&forged).is_err());
}

#[test]
#[should_panic(expected = "counting builder cannot produce a circuit")]
fn sizing_cannot_be_used_as_a_circuit() {
    CircuitBuilder::<Val>::counting().finish();
}

#[test]
fn backend_graph_inspection_matches_setup() {
    let mut b = CircuitBuilder::<Val>::new();
    let x = b.input("x");
    let y = b.mul(x, x);
    b.expose_public(y);
    let compiled = b.finish().lower_to_multi_stark(Val::from_u8(7)).unwrap();
    let inputs = compiled.circuit_inputs();
    let graphs: Vec<_> = inputs
        .iter()
        .map(|input| {
            input
                .compile_graph(&extension_params::<GoldilocksBlake3Config>())
                .unwrap()
        })
        .collect();
    let config = GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
    );
    let (system, _) = System::new(config, inputs);
    for (expected, circuit) in graphs.iter().zip(&system.circuits) {
        assert_eq!(expected, &circuit.graph);
    }
}
