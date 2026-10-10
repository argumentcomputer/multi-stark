use multi_stark::{
    ark_adapter::Scalar,
    expr::Source,
    graph::Node,
    lookup::{MAX_LOOKUP_GROUP, logup_max_degree},
    plonkish::foreign::kzg_lookup_group,
    system::Circuit,
    traits::TwoAdicField,
};

pub(crate) const INNER_TRACE_CAP: usize = 1 << 24;
pub(crate) const PUBLIC_MAX_DEGREE: usize = (1 << 28) - 2;
pub(crate) const QUOTIENT_BUDGET: usize = 2;

#[derive(Debug, serde::Serialize)]
pub(crate) struct MetadataAudit {
    pub circuit_count: usize,
    pub inner_trace_cap: usize,
    pub public_max_degree: usize,
    pub quotient_budget: usize,
    pub shifted_degree_bounds: bool,
    pub graphs_unchanged: bool,
    pub lookup_group_selection_unchanged: bool,
    pub main_coefficient_bytes: usize,
    pub fixed_coefficient_bytes: usize,
    pub circuits: Vec<CircuitAudit>,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct CircuitAudit {
    pub index: usize,
    pub height: usize,
    pub main_width: usize,
    pub fixed_width: usize,
    pub stage2_width: usize,
    pub lookup_count: usize,
    pub lookup_group: usize,
    pub current_v4_lookup_group: usize,
    pub quotient_degree: usize,
    pub max_constraint_degree: usize,
    pub constraint_count: usize,
    pub graph_nodes: usize,
    pub lookup_prefix_nodes: usize,
}

/// Check the supplied compiled AIR against current v4 tuning without
/// reconstructing traces or altering its relation. Provenance of the supplied
/// verifier profile and its committed coefficients is a separate obligation.
pub(crate) fn audit(
    circuits: &[Circuit<Scalar>],
    widths: &[usize],
    heights: &[usize],
) -> Result<MetadataAudit, String> {
    if circuits.is_empty() || circuits.len() != widths.len() || widths.len() != heights.len() {
        return Err("saved circuit, width and height counts differ".into());
    }
    let mut audited = Vec::with_capacity(circuits.len());
    let mut main_coefficient_bytes = 0usize;
    let mut fixed_coefficient_bytes = 0usize;
    for (index, ((circuit, &width), &height)) in
        circuits.iter().zip(widths).zip(heights).enumerate()
    {
        let result = audit_circuit(index, circuit, width, height)
            .map_err(|error| format!("saved circuit {index}: {error}"))?;
        let bytes = |width: usize| {
            height
                .checked_mul(width)
                .and_then(|cells| cells.checked_mul(size_of::<Scalar>()))
                .ok_or("coefficient byte count overflow")
        };
        main_coefficient_bytes = main_coefficient_bytes
            .checked_add(bytes(width)?)
            .ok_or("main coefficient byte total overflow")?;
        fixed_coefficient_bytes = fixed_coefficient_bytes
            .checked_add(bytes(circuit.preprocessed_width)?)
            .ok_or("fixed coefficient byte total overflow")?;
        audited.push(result);
    }
    Ok(MetadataAudit {
        circuit_count: circuits.len(),
        inner_trace_cap: INNER_TRACE_CAP,
        public_max_degree: PUBLIC_MAX_DEGREE,
        quotient_budget: QUOTIENT_BUDGET,
        shifted_degree_bounds: false,
        graphs_unchanged: true,
        lookup_group_selection_unchanged: true,
        main_coefficient_bytes,
        fixed_coefficient_bytes,
        circuits: audited,
    })
}

fn audit_circuit(
    index: usize,
    circuit: &Circuit<Scalar>,
    width: usize,
    height: usize,
) -> Result<CircuitAudit, String> {
    if height < 2 || !height.is_power_of_two() || height > INNER_TRACE_CAP {
        return Err("trace height exceeds the diagnostic inner cap".into());
    }
    if width == 0
        || circuit.main_width != width
        || circuit.preprocessed_height != height
        || circuit.preprocessed_width == 0
        || circuit.preprocessed.is_some()
        || [width, circuit.preprocessed_width, circuit.stage_2_width]
            .into_iter()
            .any(|width| u32::try_from(width).is_err())
    {
        return Err("saved main/fixed dimensions are inconsistent".into());
    }
    super::shape::validate_graph(circuit)?;
    let graph = &circuit.graph;
    if graph.nodes[..graph.lookup_prefix_len]
        .iter()
        .any(|node| match node {
            Node::Public(_) => true,
            Node::Var(column) => column.source == Source::Stage2,
            _ => false,
        })
    {
        return Err("lookup prefix depends on challenge-stage values".into());
    }
    let mut degrees = Vec::<u32>::with_capacity(graph.nodes.len());
    for (index, node) in graph.nodes.iter().enumerate() {
        let degree = match *node {
            Node::Const(_) | Node::Public(_) | Node::IsTransition => 0,
            Node::Var(_) | Node::IsFirstRow | Node::IsLastRow => 1,
            Node::Add(a, b) | Node::Sub(a, b) => degrees[a.index()].max(degrees[b.index()]),
            Node::Mul(a, b) => degrees[a.index()]
                .checked_add(degrees[b.index()])
                .ok_or("graph degree overflow")?,
            Node::Neg(a) => degrees[a.index()],
        };
        if graph.degrees[index] != degree {
            return Err("stored graph degrees differ from their expressions".into());
        }
        degrees.push(degree);
    }
    let graph_degree = graph
        .zeros
        .iter()
        .map(|id| degrees[id.index()])
        .max()
        .unwrap_or(0);
    if graph.max_constraint_degree != graph_degree {
        return Err("stored constraint-root degree differs from its expressions".into());
    }
    let degree_budget = u32::try_from(QUOTIENT_BUDGET + 1).unwrap();
    if graph_degree > degree_budget
        || graph.lookups.iter().any(|lookup| {
            std::iter::once(&lookup.multiplicity)
                .chain(&lookup.args)
                .any(|id| degrees[id.index()] > degree_budget)
        })
    {
        return Err("graph expressions exceed the quotient budget".into());
    }
    // Lookup node degrees are bounded before the chooser evaluates every
    // group, including groups that ultimately exceed the quotient budget.
    debug_assert!((MAX_LOOKUP_GROUP as u32 + 1) * degree_budget < u32::MAX);
    let selected = kzg_lookup_group(
        graph,
        circuit.num_lookups,
        height,
        INNER_TRACE_CAP,
        QUOTIENT_BUDGET,
        false,
    )
    .map_err(|error| format!("v4 lookup selection failed: {error:?}"))?;
    if selected != circuit.lookup_group_size {
        return Err(format!(
            "v4 selects lookup group {selected}, but the saved profile uses {}; retuning is required",
            circuit.lookup_group_size
        ));
    }
    let max_constraint_degree = graph_degree.max(logup_max_degree(graph, selected)) as usize;
    if circuit.max_constraint_degree != max_constraint_degree {
        return Err("saved maximum constraint degree is inconsistent".into());
    }
    let quotient_degree = (max_constraint_degree.max(2) - 1)
        .checked_next_power_of_two()
        .ok_or("quotient degree overflow")?;
    if quotient_degree > QUOTIENT_BUDGET
        || height.ilog2() as usize + quotient_degree.ilog2() as usize > Scalar::TWO_ADICITY
    {
        return Err("quotient domain exceeds the configured budget".into());
    }
    Ok(CircuitAudit {
        index,
        height,
        main_width: width,
        fixed_width: circuit.preprocessed_width,
        stage2_width: circuit.stage_2_width,
        lookup_count: circuit.num_lookups,
        lookup_group: circuit.lookup_group_size,
        current_v4_lookup_group: selected,
        quotient_degree,
        max_constraint_degree,
        constraint_count: circuit.constraint_count,
        graph_nodes: graph.nodes.len(),
        lookup_prefix_nodes: graph.lookup_prefix_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use multi_stark::{
        expr::{ColRef, RowOffset},
        graph::{ConstraintGraph, NodeId},
        lookup::Lookup,
        traits::Algebra,
    };

    fn fixture(lookups: usize, group: usize) -> Circuit<Scalar> {
        let graph = ConstraintGraph {
            nodes: vec![
                Node::Const(Scalar::ONE),
                Node::Var(ColRef {
                    source: Source::Main,
                    offset: RowOffset::Current,
                    index: 0,
                }),
            ],
            degrees: vec![0, 1],
            zeros: vec![],
            lookups: vec![Lookup::push(NodeId(0), vec![NodeId(1)]); lookups],
            lookup_prefix_len: 2,
            max_constraint_degree: 0,
        };
        let stage_2_width = lookups.div_ceil(group).max(1);
        Circuit {
            max_constraint_degree: logup_max_degree(&graph, group) as usize,
            graph,
            main_width: 1,
            preprocessed: None,
            preprocessed_width: 1,
            preprocessed_height: 16,
            num_lookups: lookups,
            stage_2_width,
            num_publics: 4,
            lookup_group_size: group,
            constraint_count: stage_2_width,
        }
    }

    #[test]
    fn accepts_unchanged_v4_metadata_without_matrices() {
        let circuits = [fixture(4, 2), fixture(1, 1)];
        let before = bincode::serde::encode_to_vec(&circuits, bincode::config::standard()).unwrap();
        let result = audit(&circuits, &[1, 1], &[16, 16]).unwrap();
        assert_eq!(result.circuit_count, 2);
        assert!(result.graphs_unchanged && result.lookup_group_selection_unchanged);
        assert_eq!(result.circuits[0].quotient_degree, 2);
        assert_eq!(result.circuits[1].quotient_degree, 1);
        assert_eq!(result.main_coefficient_bytes, 2 * 16 * 32);
        assert_eq!(result.fixed_coefficient_bytes, 2 * 16 * 32);
        assert_eq!(
            bincode::serde::encode_to_vec(&circuits, bincode::config::standard()).unwrap(),
            before
        );
    }

    #[test]
    fn rejects_profiles_requiring_retuning() {
        let circuit = fixture(4, 1);
        assert!(
            audit(&[circuit], &[1], &[16])
                .unwrap_err()
                .contains("retuning is required")
        );
    }

    #[test]
    fn rejects_malformed_dimensions_and_degrees() {
        assert!(audit(&[], &[], &[]).is_err());
        assert!(audit(&[fixture(4, 2)], &[], &[16]).is_err());
        assert!(audit(&[fixture(4, 2)], &[1], &[3]).is_err());
        assert!(audit(&[fixture(4, 2)], &[1], &[INNER_TRACE_CAP * 2]).is_err());
        assert!(audit(&[fixture(4, 2)], &[usize::MAX], &[16]).is_err());
        for case in 0..8 {
            let mut circuit = fixture(4, 2);
            match case {
                0 => circuit.graph.degrees[1] = 0,
                1 => circuit.graph.max_constraint_degree = 1,
                2 => circuit.max_constraint_degree += 1,
                3 => circuit.stage_2_width += 1,
                4 => circuit.constraint_count += 1,
                5 => circuit.graph.lookups[0].args[0] = NodeId(9),
                6 => circuit.graph.nodes[1] = Node::Add(NodeId(1), NodeId(0)),
                7 => circuit.graph.nodes[0] = Node::Public(0),
                _ => unreachable!(),
            }
            assert!(audit(&[circuit], &[1], &[16]).is_err(), "case {case}");
        }
    }
}
