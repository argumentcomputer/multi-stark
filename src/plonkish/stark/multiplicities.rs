use crate::traits::Field;

use super::super::Circuit;

#[cfg(feature = "parallel")]
use p3_maybe_rayon::prelude::*;

#[cfg(any(feature = "parallel", test))]
const MIN_LOOKUPS_PER_JOB: usize = 1 << 16;
#[cfg(any(feature = "parallel", test))]
const MAX_WORKING_BYTES: usize = 64 << 20;

#[cfg(any(feature = "parallel", test))]
struct HistogramPlan {
    offsets: Vec<usize>,
    rows: usize,
    row_width: usize,
    chunk_len: usize,
    jobs: usize,
    working_bytes: usize,
}

#[cfg(any(feature = "parallel", test))]
fn histogram_plan<F: Field>(
    lookup_count: usize,
    dimensions: impl ExactSizeIterator<Item = (usize, usize)>,
    workers: usize,
    budget_bytes: usize,
) -> Option<HistogramPlan> {
    let requested_jobs = workers.min(lookup_count / MIN_LOOKUPS_PER_JOB);
    if requested_jobs < 2 || dimensions.len() == 0 {
        return None;
    }
    let offset_count = dimensions.len().checked_add(1)?;
    let offset_bytes = offset_count.checked_mul(size_of::<usize>())?;
    if offset_bytes > budget_bytes {
        return None;
    }
    let mut offsets = Vec::new();
    offsets.try_reserve_exact(offset_count).ok()?;
    offsets.push(0usize);
    let mut row_width = 0usize;
    for (rows, width) in dimensions {
        if rows == 0 || width == 0 {
            return None;
        }
        offsets.push(offsets.last()?.checked_add(rows)?);
        row_width = row_width.max(width);
    }
    let rows = *offsets.last()?;
    let per_job_bytes = rows
        .checked_mul(size_of::<usize>())?
        .checked_add(row_width.checked_mul(size_of::<F>())?)?
        .checked_add(size_of::<Vec<usize>>() + size_of::<Vec<F>>())?;
    // Dense initialization and reduction should not exceed the lookup work.
    let requested_jobs = requested_jobs
        .min(lookup_count / rows)
        .min(budget_bytes.checked_sub(offset_bytes)? / per_job_bytes);
    if requested_jobs < 2 {
        return None;
    }
    let chunk_len = lookup_count.div_ceil(requested_jobs);
    let jobs = lookup_count.div_ceil(chunk_len);
    let working_bytes = jobs.checked_mul(per_job_bytes)?.checked_add(offset_bytes)?;
    if working_bytes > budget_bytes {
        return None;
    }
    Some(HistogramPlan {
        offsets,
        rows,
        row_width,
        chunk_len,
        jobs,
        working_bytes,
    })
}

fn serial_counts<F: Field>(circuit: &Circuit<F>, values: &[F]) -> Vec<Vec<F>> {
    let mut counts: Vec<Vec<F>> = circuit
        .tables
        .iter()
        .map(|table| F::zero_vec(table.rows.len().max(2).next_power_of_two()))
        .collect();
    let mut args = Vec::new();
    for lookup in &circuit.lookups {
        let table = &circuit.tables[lookup.table.index];
        args.clear();
        args.extend(lookup.values.iter().map(|v| values[v.index]));
        let row = table.indices[&args];
        counts[lookup.table.index][row] += F::ONE;
    }
    counts
}

#[cfg(feature = "parallel")]
fn parallel_counts<F: Field>(
    circuit: &Circuit<F>,
    values: &[F],
    plan: &HistogramPlan,
) -> Option<Vec<Vec<F>>> {
    // One allocation per explicit chunk bounds live histograms, including
    // both operands of every in-place reduction.
    let counts = circuit
        .lookups
        .par_chunks(plan.chunk_len)
        .map(|lookups| {
            let mut counts = Vec::new();
            counts.try_reserve_exact(plan.rows).ok()?;
            counts.resize(plan.rows, 0usize);
            let mut args = Vec::new();
            args.try_reserve_exact(plan.row_width).ok()?;
            for lookup in lookups {
                let table = &circuit.tables[lookup.table.index];
                args.clear();
                args.extend(lookup.values.iter().map(|v| values[v.index]));
                let row = *table.indices.get(&args)?;
                counts[plan.offsets[lookup.table.index] + row] += 1;
            }
            Some(counts)
        })
        .reduce_with(|left, right| {
            let mut left = left?;
            let right = right?;
            // Disjoint chunks contribute once per lookup, so every count is
            // bounded by the length of the original lookup vector.
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            Some(left)
        })??;
    Some(
        circuit
            .tables
            .iter()
            .enumerate()
            .map(|(index, table)| {
                let mut result = F::zero_vec(table.rows.len().max(2).next_power_of_two());
                for (output, &count) in result
                    .iter_mut()
                    .zip(&counts[plan.offsets[index]..plan.offsets[index + 1]])
                {
                    *output = F::from_usize(count);
                }
                result
            })
            .collect(),
    )
}

pub(super) fn table_counts<F: Field>(circuit: &Circuit<F>, values: &[F]) -> Vec<Vec<F>> {
    #[cfg(feature = "parallel")]
    if let Some(plan) = histogram_plan::<F>(
        circuit.lookups.len(),
        circuit
            .tables
            .iter()
            .map(|table| (table.rows.len(), table.width())),
        current_num_threads(),
        MAX_WORKING_BYTES,
    ) {
        tracing::debug!(
            jobs = plan.jobs,
            rows = plan.rows,
            working_bytes = plan.working_bytes,
            "Plonkish table histogram admitted"
        );
        if let Some(counts) = parallel_counts(circuit, values, &plan) {
            return counts;
        }
    }
    serial_counts(circuit, values)
}

#[cfg(test)]
mod tests;
