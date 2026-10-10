//! Quotient construction on trace-sized device cosets, with coefficient-only output.

use super::*;
use crate::ark_adapter::{config::KzgConfig, field::Scalar, pcs::CommittedMatrix};
use crate::config::QuotientCommitInput;
use crate::expr::{RowOffset, Source};
use crate::graph::{ConstraintGraph, Node};
use crate::traits::Algebra;

pub(super) const TILE: usize = 128;
pub(super) const MAX_BLOCKS: usize = 256;
pub(super) const HEADROOM: usize = 1 << 30;

mod distributed;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
pub(super) struct Instruction {
    value: [u64; 4],
    a: u32,
    b: u32,
    op: u32,
    out: u32,
}

#[repr(C)]
pub(super) struct Lookup {
    multiplicity: u32,
    arg_start: u32,
    arg_count: u32,
}

#[repr(C, align(32))]
pub(super) struct Column {
    input: *const [u64; 4],
    resident: *const c_void,
    count: usize,
    padding: u64,
    constant: [u64; 4],
    slot: u32,
    reserved: u32,
}

#[repr(C, align(32))]
struct Parameters {
    alpha: [u64; 4],
    generator: [u64; 4],
    generator_inverse: [u64; 4],
    delta: [u64; 4],
    quotient_shift_inverse: [u64; 4],
    trace_log: u32,
    quotient_log: u32,
    slots: u32,
    stage2_start: u32,
    group_size: u32,
    blocks: u32,
}

unsafe extern "C" {
    fn multi_stark_kzg_quotient(
        device: i32,
        parameters: *const Parameters,
        columns: *const Column,
        column_count: usize,
        evaluation_columns: usize,
        nodes: *const Instruction,
        node_count: usize,
        roots: *const u32,
        root_count: usize,
        lookups: *const Lookup,
        lookup_count: usize,
        args: *const u32,
        arg_count: usize,
        publics: *const [u64; 4],
        public_count: usize,
        coset_shifts: *const [u64; 4],
        inverse_vanishing: *const [u64; 4],
        outputs: *const *mut [u64; 4],
    ) -> i32;
}

pub(in crate::ark_adapter) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    super::enabled()
        && *ENABLED.get_or_init(|| {
            match std::env::var("MULTI_STARK_KZG_CUDA_QUOTIENT").as_deref() {
                Ok("cpu") => false,
                Ok("cuda") | Err(std::env::VarError::NotPresent) => true,
                _ => panic!("MULTI_STARK_KZG_CUDA_QUOTIENT must be cpu or cuda"),
            }
        })
}

pub(super) struct Program {
    pub(super) nodes: Vec<Instruction>,
    pub(super) roots: Vec<u32>,
    pub(super) lookups: Vec<Lookup>,
    pub(super) args: Vec<u32>,
    pub(super) slots: usize,
}

fn encode(graph: &ConstraintGraph<Scalar>, widths: [usize; 3], publics: usize) -> Option<Program> {
    encode_range(graph, widths, publics, graph.nodes.len(), &graph.zeros)
}

pub(super) fn encode_lookup(
    graph: &ConstraintGraph<Scalar>,
    widths: [usize; 3],
) -> Option<Program> {
    encode_range(graph, widths, 0, graph.lookup_prefix_len, &[])
}

fn encode_range(
    graph: &ConstraintGraph<Scalar>,
    widths: [usize; 3],
    publics: usize,
    n: usize,
    roots: &[crate::graph::NodeId],
) -> Option<Program> {
    let node_range = graph.nodes.get(..n)?;
    let mut last: Vec<_> = (0..n).collect();
    for (i, node) in node_range.iter().enumerate() {
        let mut consume = |id: crate::graph::NodeId| {
            if id.index() >= i {
                return None;
            }
            last[id.index()] = i;
            Some(())
        };
        match *node {
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                consume(a)?;
                consume(b)?;
            }
            Node::Neg(a) => consume(a)?,
            _ => {}
        }
    }
    for id in roots.iter().chain(
        graph
            .lookups
            .iter()
            .flat_map(|lookup| core::iter::once(&lookup.multiplicity).chain(lookup.args.iter())),
    ) {
        *last.get_mut(id.index())? = n;
    }
    let mut mapping = vec![0u32; n];
    let mut free = Vec::new();
    let mut retire = vec![Vec::new(); n + 1];
    let mut slots = 0u32;
    let mut nodes = Vec::with_capacity(n);
    for (i, node) in node_range.iter().enumerate() {
        free.append(&mut retire[i]);
        let out = free.pop().unwrap_or_else(|| {
            let slot = slots;
            slots += 1;
            slot
        });
        mapping[i] = out;
        if last[i] < n {
            retire[last[i] + 1].push(out);
        }
        let mut encoded = Instruction {
            out,
            ..Default::default()
        };
        match *node {
            Node::Const(value) => encoded.value = value.0.0.0,
            Node::Var(column) => {
                let source = match column.source {
                    Source::Preprocessed => 0,
                    Source::Main => 1,
                    Source::Stage2 => 2,
                };
                if column.index as usize >= widths[source] {
                    return None;
                }
                encoded.op = 1;
                encoded.a = u32::try_from(widths[..source].iter().sum::<usize>())
                    .ok()?
                    .checked_add(column.index)?;
                encoded.b = u32::from(column.offset == RowOffset::Next);
            }
            Node::Public(index) => {
                if index as usize >= publics {
                    return None;
                }
                encoded.op = 2;
                encoded.a = index;
            }
            Node::IsFirstRow => encoded.op = 3,
            Node::IsLastRow => encoded.op = 4,
            Node::IsTransition => encoded.op = 5,
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                encoded.op = match node {
                    Node::Add(..) => 6,
                    Node::Sub(..) => 7,
                    _ => 8,
                };
                encoded.a = mapping[a.index()];
                encoded.b = mapping[b.index()];
            }
            Node::Neg(a) => {
                encoded.op = 9;
                encoded.a = mapping[a.index()];
            }
        }
        nodes.push(encoded);
    }
    let roots = roots.iter().map(|id| mapping[id.index()]).collect();
    let mut lookups = Vec::with_capacity(graph.lookups.len());
    let mut args = Vec::new();
    for lookup in &graph.lookups {
        lookups.push(Lookup {
            multiplicity: mapping[lookup.multiplicity.index()],
            arg_start: u32::try_from(args.len()).ok()?,
            arg_count: u32::try_from(lookup.args.len()).ok()?,
        });
        args.extend(lookup.args.iter().map(|id| mapping[id.index()]));
    }
    Some(Program {
        nodes,
        roots,
        lookups,
        args,
        slots: slots as usize,
    })
}

pub(super) fn matrix<'a>(
    data: (&'a crate::ark_adapter::pcs::KzgProverData, usize),
) -> Option<&'a CommittedMatrix> {
    data.0.matrices.get(data.1)
}

pub(super) fn allocation<T>(count: usize) -> Option<usize> {
    count
        .checked_add(31)?
        .checked_div(32)?
        .checked_mul(32)?
        .checked_mul(size_of::<T>())
}

fn memory_bound(
    program: &Program,
    n: usize,
    ratio: usize,
    columns: usize,
    evaluated: usize,
    publics: usize,
) -> Option<usize> {
    let blocks = n.div_ceil(TILE).min(MAX_BLOCKS);
    let elements = evaluated
        .checked_mul(n)?
        .checked_add(n)?
        .checked_add(n.checked_mul(ratio)?)?
        .checked_add(
            blocks
                .checked_mul(TILE)?
                .checked_mul(program.slots.max(1))?,
        )?;
    // Base_dev_ptr is in-place. Both static twiddle tables fit in 21 MiB;
    // the remaining headroom covers allocator granularity and runtime state.
    // The memory query excludes live allocations and credits reusable pool pages.
    [
        allocation::<Fr>(elements)?,
        allocation::<Instruction>(program.nodes.len())?,
        allocation::<u32>(program.roots.len())?,
        allocation::<Lookup>(program.lookups.len())?,
        allocation::<u32>(program.args.len())?,
        allocation::<Column>(columns)?,
        allocation::<Fr>(publics)?,
        HEADROOM,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

pub(in crate::ark_adapter) fn coefficients(
    input: &QuotientCommitInput<'_, KzgConfig>,
    alpha: Scalar,
) -> Option<Vec<Vec<Fr>>> {
    coefficients_impl(input, alpha, None)
}

fn coefficients_impl(
    input: &QuotientCommitInput<'_, KzgConfig>,
    alpha: Scalar,
    distributed_options: Option<distributed::Options>,
) -> Option<Vec<Vec<Fr>>> {
    let trace = input.trace_domain;
    let quotient = input.quotient_domain;
    if trace.log_size < 10
        || quotient.log_size > 28
        || trace.log_size > quotient.log_size
        || trace.shift != Scalar::ONE
        || quotient.shift == Scalar::ZERO
    {
        return None;
    }
    let n = 1usize.checked_shl(trace.log_size as u32)?;
    let ratio = 1usize.checked_shl((quotient.log_size - trace.log_size) as u32)?;
    let circuit = input.circuit;
    let group_size = circuit.lookup_group_size.max(1);
    let groups = circuit.graph.lookups.len().div_ceil(group_size).max(1);
    if input.lookup_publics.len() < 4
        || group_size > crate::lookup::MAX_LOOKUP_GROUP
        || groups != circuit.stage_2_width
        || input.constraint_count != circuit.graph.zeros.len().checked_add(groups)?
    {
        return None;
    }
    let fixed = match input.preprocessed {
        Some(data) => Some(matrix(data)?),
        None => None,
    };
    let main = matrix(input.stage_1)?;
    let stage2 = matrix(input.stage_2)?;
    let widths = [
        fixed.map_or(0, CommittedMatrix::width),
        main.width(),
        stage2.width(),
    ];
    if widths
        != [
            circuit.preprocessed_width,
            circuit.main_width,
            circuit.stage_2_width,
        ]
        || [fixed, Some(main), Some(stage2)]
            .into_iter()
            .flatten()
            .any(|m| m.domain != trace)
    {
        return None;
    }
    let program = encode(&circuit.graph, widths, input.lookup_publics.len())?;
    let generator = trace.generator().0;
    let generator_inverse = generator.inverse()?;
    let mut shift = quotient.shift.0;
    let mut shifts = Vec::with_capacity(ratio);
    let mut inverse_vanishing = Vec::with_capacity(ratio);
    for _ in 0..ratio {
        shifts.push(shift);
        inverse_vanishing.push((shift.pow([n as u64]) - Fr::ONE).inverse()?);
        shift *= quotient.generator().0;
    }
    let all_columns: Vec<_> = [fixed, Some(main), Some(stage2)]
        .into_iter()
        .flatten()
        .flat_map(CommittedMatrix::cuda_columns)
        .collect();
    if all_columns
        .iter()
        .any(|(column, _, _)| column.is_empty() || column.len() > n)
    {
        return None;
    }
    let evaluated = all_columns
        .iter()
        .filter(|(_, _, constant)| !constant)
        .count();
    let bytes = memory_bound(
        &program,
        n,
        ratio,
        all_columns.len(),
        evaluated,
        input.lookup_publics.len(),
    )?;
    let parameters = Parameters {
        alpha: alpha.0.0.0,
        generator: generator.0.0,
        generator_inverse: generator_inverse.0.0,
        delta: ((input.lookup_publics[3].0 - input.lookup_publics[2].0)
            * (Fr::from(n as u64) * generator).inverse()?)
        .0
        .0,
        quotient_shift_inverse: quotient.shift.0.inverse()?.0.0,
        trace_log: trace.log_size as u32,
        quotient_log: quotient.log_size as u32,
        slots: u32::try_from(program.slots.max(1)).ok()?,
        stage2_start: u32::try_from(widths[0].checked_add(widths[1])?).ok()?,
        group_size: group_size as u32,
        blocks: n.div_ceil(TILE).min(MAX_BLOCKS) as u32,
    };
    let eligible = eligible_devices(&all_columns, bytes);
    if let Some(options) = distributed_options {
        return distributed::coefficients(
            &parameters,
            &program,
            &all_columns,
            &input.lookup_publics,
            &shifts,
            &inverse_vanishing,
            options,
        );
    }
    if distributed::enabled(eligible.is_empty()) {
        if let Some(output) = distributed::coefficients(
            &parameters,
            &program,
            &all_columns,
            &input.lookup_publics,
            &shifts,
            &inverse_vanishing,
            distributed::options(),
        ) {
            return Some(output);
        }
    }
    if eligible.is_empty() {
        tracing::debug!(bytes, "KZG quotient exceeds device memory budget");
        return None;
    }
    let mut outputs: Vec<_> = (0..ratio).map(|_| vec![Fr::ZERO; n]).collect();
    let output_pointers: Vec<*mut [u64; 4]> = outputs
        .iter_mut()
        .map(|values| values.as_mut_ptr().cast())
        .collect();
    let mut columns = column_descriptors(&all_columns);
    let device = acquire_columns(&eligible, &mut columns, &all_columns, bytes)?;
    let started = std::time::Instant::now();
    // Every borrowed coefficient and output buffer outlives the synchronous call.
    check(
        unsafe {
            multi_stark_kzg_quotient(
                device.id(),
                &parameters,
                columns.as_ptr(),
                columns.len(),
                evaluated,
                program.nodes.as_ptr(),
                program.nodes.len(),
                program.roots.as_ptr(),
                program.roots.len(),
                program.lookups.as_ptr(),
                program.lookups.len(),
                program.args.as_ptr(),
                program.args.len(),
                input.lookup_publics.as_ptr().cast(),
                input.lookup_publics.len(),
                shifts.as_ptr().cast(),
                inverse_vanishing.as_ptr().cast(),
                output_pointers.as_ptr(),
            )
        },
        "resident quotient",
    );
    tracing::info!(
        device = device.id(),
        rows = n,
        ratio,
        evaluated,
        bytes,
        seconds = started.elapsed().as_secs_f64(),
        "KZG quotient evaluated on device"
    );
    Some(outputs)
}

pub(super) type PolynomialView<'a> = (&'a [Fr], Option<&'a ResidentPolynomial>, bool);

pub(super) fn eligible_devices(
    all_columns: &[PolynomialView<'_>],
    bytes: usize,
) -> Vec<(usize, usize)> {
    let devices = Devices::get();
    let mut eligible = Vec::new();
    for index in 0..devices.ids.len() {
        if bytes <= devices.potential_budget(index) {
            let local = all_columns
                .iter()
                .filter(|(_, resident, _)| resident.is_some_and(|r| r.reservation.index == index))
                .count();
            eligible.push((index, local));
        }
    }
    eligible.sort_by_key(|(_, local)| std::cmp::Reverse(*local));
    eligible
}

pub(super) fn column_descriptors(all_columns: &[PolynomialView<'_>]) -> Vec<Column> {
    let mut next_slot = 0;
    let columns: Vec<_> = all_columns
        .iter()
        .map(|(column, _, constant)| {
            let slot = if *constant {
                u32::MAX
            } else {
                let slot = next_slot;
                next_slot += 1;
                slot
            };
            Column {
                input: column.as_ptr().cast(),
                resident: std::ptr::null(),
                count: column.len(),
                padding: 0,
                constant: column[0].0.0,
                slot,
                reserved: 0,
            }
        })
        .collect();
    columns
}

pub(super) fn resident_column_descriptors(all_columns: &[PolynomialView<'_>]) -> Vec<Column> {
    let mut columns = column_descriptors(all_columns);
    for (descriptor, (_, resident, _)) in columns.iter_mut().zip(all_columns) {
        descriptor.resident = resident.map_or(std::ptr::null(), |r| r.context as *const c_void);
    }
    columns
}

pub(super) fn acquire_columns(
    eligible: &[(usize, usize)],
    columns: &mut [Column],
    all_columns: &[PolynomialView<'_>],
    bytes: usize,
) -> Option<Device<'static>> {
    if eligible.is_empty() {
        return None;
    }
    let devices = Devices::get();
    let device = {
        let mut busy = devices.busy.lock().unwrap();
        loop {
            if let Some(&(index, _)) = eligible.iter().find(|&&(index, _)| !busy[index]) {
                busy[index] = true;
                break Device { devices, index };
            }
            busy = devices.available.wait(busy).unwrap();
        }
    };
    if !device.prepare_budget(bytes) {
        return None;
    }
    for (column, (_, resident, _)) in columns.iter_mut().zip(all_columns) {
        if let Some(resident) = resident.filter(|r| r.reservation.index == device.index) {
            column.resident = resident.context as *const c_void;
        }
    }
    Some(device)
}
