use super::*;
use crate::{
    ark_adapter::{KzgConfig, Srs, compact::FixedProofCodec, srs::PublicSetup},
    plonkish::CircuitBuilder,
    system::{System, SystemWitness},
};
use std::sync::Arc;

#[test]
fn metadata_profile_matches_lowering_and_actual_codec() {
    for (arity, extra_gates, shifted) in [(1, 0, false), (3, 12, false), (5, 12, true)] {
        let namespace = Scalar::from_u8(94);
        let mut builder = CircuitBuilder::<Scalar>::new();
        let first = builder.fixed_table(
            "first",
            (1..=3).map(|n| vec![Scalar::from_u8(n); arity]).collect(),
        );
        builder.fixed_table(
            "second",
            (4..=5).map(|n| vec![Scalar::from_u8(n); arity]).collect(),
        );
        let input = builder.public_input("value");
        let one = builder.constant(Scalar::ONE);
        for _ in 0..=extra_gates {
            builder.assert_equal(input, one);
        }
        builder.lookup(first, &vec![input; arity]);
        let circuit = builder.finish();
        let height = circuit.multi_stark_layout().unwrap().main_height;
        let compiled = circuit
            .lower_to_multi_stark(namespace)
            .unwrap()
            .merge_table_traces(height)
            .unwrap();
        let table_height = compiled.trace_dimensions(1).unwrap().0;
        assert_eq!(compiled.num_circuits(), 2);
        assert_eq!(height == table_height, extra_gates == 0);
        let metadata = MultiStarkCircuit::<Scalar>::ordinary_merged_kzg_profile(
            namespace,
            compiled.main_width(),
            height,
            table_height,
            2,
            shifted,
        )
        .unwrap();
        assert!(
            metadata
                .iter()
                .all(|circuit| circuit.preprocessed.is_none())
        );
        let mut srs = Srs::unsafe_dev_setup(height, b"ordinary-metadata-fixture");
        if !shifted {
            // These parameters exercise the public-degree protocol with a known trapdoor.
            srs.public_setup = Some(PublicSetup {
                max_degree: 2 * height - 2,
                id: [7; 32],
            });
            srs.degree_keys.clear();
        }
        let definitions: Vec<_> = (0..2)
            .map(|index| {
                compiled
                    .kzg_circuit_input_with_degree_policy(index, height, 2, shifted)
                    .unwrap()
                    .unwrap()
            })
            .collect();
        let (system, key) = System::new(KzgConfig::new(Arc::new(srs), 2), definitions);
        for (actual, projected) in system.circuits.iter().zip(&metadata) {
            let dimensions = |c: &Circuit<Scalar>| {
                (
                    c.main_width,
                    c.preprocessed_width,
                    c.preprocessed_height,
                    c.num_lookups,
                    c.stage_2_width,
                    c.num_publics,
                    c.lookup_group_size,
                    c.constraint_count,
                    c.max_constraint_degree,
                )
            };
            assert_eq!(dimensions(actual), dimensions(projected));
            let encode = |graph: &crate::graph::ConstraintGraph<Scalar>| {
                bincode::serde::encode_to_vec(graph, bincode::config::standard()).unwrap()
            };
            assert_eq!(encode(&actual.graph), encode(&projected.graph));
        }
        let mut witness = compiled.witness();
        witness.set(input, Scalar::ONE).unwrap();
        let assignment = witness.generate().unwrap();
        let claims = compiled.claims(&[Scalar::ONE]).unwrap();
        let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
        let proof = system.prove_multiple_claims(
            &key,
            &refs,
            SystemWitness::from_stage_1(compiled.traces(&assignment).unwrap(), &system),
        );
        system.verify_multiple_claims(&refs, &proof).unwrap();
        let logs = [height.ilog2() as u8, table_height.ilog2() as u8];
        let actual = FixedProofCodec::new(&system, &logs)
            .unwrap()
            .encode(&proof)
            .unwrap();
        let projected =
            FixedProofCodec::for_profile(&metadata, &logs, logs[0] as usize, shifted).unwrap();
        assert_eq!(
            projected.encoded_len(proof.opening_proof.0.len()).unwrap(),
            actual.len()
        );
        assert_eq!(projected.encode(&proof).unwrap(), actual);
        assert!(projected.encoded_len(0).is_err());
        assert!(projected.encoded_len(4).is_err());
    }
}

#[test]
fn metadata_profile_rejects_invalid_geometry_and_degree_budget() {
    let namespace = Scalar::from_u8(94);
    for (width, main, table, quotient) in [
        (2, 16, 8, 2),
        (3, 1, 2, 2),
        (3, 16, 32, 2),
        (3, 15, 8, 2),
        (3, 16, 7, 2),
        (3, 16, 8, 0),
        (3, 16, 8, 1),
        (usize::MAX, 16, 8, 2),
    ] {
        assert!(
            MultiStarkCircuit::<Scalar>::ordinary_merged_kzg_profile(
                namespace, width, main, table, quotient, false
            )
            .is_err()
        );
    }
}
