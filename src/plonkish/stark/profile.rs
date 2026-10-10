use super::{LoweringError, MultiStarkCircuit, computation_relations, merged_table_lookup};
use crate::{
    ark_adapter::Scalar,
    expr::CircuitSpec,
    graph::{ExtensionParams, compile},
    lookup::{logup_constraint_count, logup_max_degree, stage2_width},
    system::Circuit,
    traits::{Algebra, Field, TwoAdicField},
};

impl MultiStarkCircuit<Scalar> {
    /// Derive the two ordinary KZG AIR profiles without allocating fixed or
    /// witness matrices. The computation and merged table use the same
    /// relations and lookup tuning as physical lowering.
    pub fn ordinary_merged_kzg_profile(
        namespace: Scalar,
        main_width: usize,
        main_height: usize,
        table_height: usize,
        quotient_budget: usize,
        shifted_degree_bounds: bool,
    ) -> Result<[Circuit<Scalar>; 2], LoweringError> {
        if main_width < 3
            || [main_height, table_height]
                .iter()
                .any(|&height| height < 2 || !height.is_power_of_two())
            || table_height > main_height
            || !quotient_budget.is_power_of_two()
        {
            return Err(LoweringError::InvalidTraceHeight);
        }
        let fixed_width = main_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(9))
            .ok_or(LoweringError::SizeOverflow)?;
        let table_width = main_width
            .checked_add(1)
            .ok_or(LoweringError::SizeOverflow)?;
        if u32::try_from(fixed_width).is_err() {
            return Err(LoweringError::SizeOverflow);
        }
        let cells = main_height
            .checked_mul(main_width)
            .ok_or(LoweringError::SizeOverflow)?;
        if !Scalar::prime_order_exceeds(cells.max(4)) {
            return Err(LoweringError::FieldTooSmall);
        }
        let (relation, lookups) = computation_relations(namespace, main_width, false);
        let main = metadata(
            CircuitSpec {
                main_width,
                preprocessed_width: fixed_width,
                stage2_width: stage2_width(lookups.len(), 1, 1),
                num_publics: 4,
                constraints: vec![relation],
                ext_constraints: vec![],
                lookups,
            },
            main_height,
            main_height,
            quotient_budget,
            shifted_degree_bounds,
        )?;
        let table = metadata(
            CircuitSpec {
                main_width: 1,
                preprocessed_width: table_width,
                stage2_width: 1,
                num_publics: 4,
                constraints: vec![],
                ext_constraints: vec![],
                lookups: vec![merged_table_lookup(namespace, table_width)],
            },
            table_height,
            main_height,
            quotient_budget,
            shifted_degree_bounds,
        )?;
        Ok([main, table])
    }
}

fn metadata(
    spec: CircuitSpec<Scalar>,
    height: usize,
    max_trace_len: usize,
    quotient_budget: usize,
    shifted_degree_bounds: bool,
) -> Result<Circuit<Scalar>, LoweringError> {
    let graph = compile(
        &spec,
        &ExtensionParams {
            degree: 1,
            w: Scalar::ZERO,
            karatsuba: false,
        },
    )
    .expect("ordinary lowering has valid AIR expressions");
    let group = super::super::foreign::kzg_lookup_group(
        &graph,
        spec.lookups.len(),
        height,
        max_trace_len,
        quotient_budget,
        shifted_degree_bounds,
    )?;
    let max_constraint_degree = graph
        .max_constraint_degree
        .max(logup_max_degree(&graph, group)) as usize;
    let quotient = (max_constraint_degree.max(2) - 1)
        .checked_next_power_of_two()
        .ok_or(LoweringError::SizeOverflow)?;
    if height.ilog2() as usize + quotient.ilog2() as usize > Scalar::TWO_ADICITY {
        return Err(LoweringError::InvalidTraceHeight);
    }
    Ok(Circuit {
        constraint_count: graph.zeros.len() + logup_constraint_count(spec.lookups.len(), group, 1),
        graph,
        main_width: spec.main_width,
        preprocessed: None,
        preprocessed_width: spec.preprocessed_width,
        preprocessed_height: height,
        num_lookups: spec.lookups.len(),
        stage_2_width: stage2_width(spec.lookups.len(), group, 1),
        num_publics: 4,
        lookup_group_size: group,
        max_constraint_degree,
    })
}

#[cfg(test)]
mod tests;
