pub(super) use super::super::distributed::Options;
use super::super::distributed::{COPY_BYTES, Plan};
use super::quotient::PolynomialView;
use super::*;

pub(super) fn options() -> Options {
    Options::from_env(
        "MULTI_STARK_KZG_CUDA_LOOKUP_TILE_ROWS",
        "MULTI_STARK_KZG_CUDA_LOOKUP_HOST_STAGING",
    )
}

pub(super) fn enabled(single_rejected: bool) -> bool {
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| {
        match std::env::var("MULTI_STARK_KZG_CUDA_DISTRIBUTED_LOOKUP").as_deref() {
            Ok("0") | Err(std::env::VarError::NotPresent) => 0,
            Ok("1") => 1,
            Ok("force") => 2,
            _ => panic!("MULTI_STARK_KZG_CUDA_DISTRIBUTED_LOOKUP must be 0, 1 or force"),
        }
    }) {
        1 => single_rejected,
        2 => true,
        _ => false,
    }
}

unsafe extern "C" {
    fn multi_stark_kzg_lookup_distributed(
        devices: *const i32,
        tile_rows: usize,
        force_host: bool,
        parameters: *const Parameters,
        columns: *const Column,
        column_count: usize,
        evaluation_columns: usize,
        nodes: *const Instruction,
        node_count: usize,
        lookups: *const Lookup,
        lookup_count: usize,
        args: *const u32,
        arg_count: usize,
        outputs: *const *mut [u64; 4],
        total: *mut [u64; 4],
    ) -> i32;
}

fn memory_bounds(
    program: &Program,
    n: usize,
    tile: usize,
    groups: usize,
    columns: usize,
    evaluated: usize,
) -> Option<[usize; 4]> {
    let metadata = [
        allocation::<Instruction>(program.nodes.len())?,
        allocation::<Lookup>(program.lookups.len())?,
        allocation::<u32>(program.args.len())?,
        allocation::<Column>(columns)?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    let scratch = tile
        .div_ceil(quotient::TILE)
        .min(quotient::MAX_BLOCKS)
        .checked_mul(quotient::TILE)?
        .checked_mul(program.slots.max(1))?;
    let tiles = allocation::<Fr>(tile.checked_add(1)?.checked_mul(evaluated)?)?.checked_mul(2)?;
    let quarter = allocation::<Fr>((n / 4).checked_mul(groups)?)?;
    let headroom = quotient::HEADROOM.checked_add(COPY_BYTES.checked_mul(2)?)?;
    let scan = quarter
        .checked_mul(4)?
        .checked_add(allocation::<Fr>(1)?)?
        .checked_add(allocation::<Fr>(5)?)?
        .checked_add(headroom)?;
    // Every group's local quarter stays alive while output columns are gathered in batches.
    let gather = quarter
        .checked_add(allocation::<Fr>(n)?)?
        .checked_add(headroom)?;
    let mut peaks = [0; 4];
    for (index, peak) in peaks.iter_mut().enumerate() {
        let owned = evaluated / 2 + usize::from(index % 2 == 0 && evaluated % 2 != 0);
        let sweep = allocation::<Fr>(owned.checked_mul(n)?)?
            .checked_add(quarter.checked_mul(2)?)?
            .checked_add(tiles)?
            .checked_add(allocation::<Fr>(scratch)?)?
            .checked_add(metadata)?
            .checked_add(headroom)?;
        *peak = sweep.max(scan).max(gather);
    }
    Some(peaks)
}

pub(super) fn coefficients(
    parameters: &Parameters,
    program: &Program,
    all_columns: &[PolynomialView<'_>],
    groups: usize,
    options: Options,
) -> Option<(Vec<Vec<Fr>>, Scalar)> {
    if !(10..=27).contains(&parameters.trace_log)
        || groups == 0
        || !(1..=crate::lookup::MAX_LOOKUP_GROUP).contains(&(parameters.group_size as usize))
        || groups
            != program
                .lookups
                .len()
                .div_ceil(parameters.group_size as usize)
        || !options.tile_rows.is_power_of_two()
        || !(128..=1 << 20).contains(&options.tile_rows)
    {
        return None;
    }
    let n = 1usize.checked_shl(parameters.trace_log)?;
    let tile = options.tile_rows.min(n / 4);
    let evaluated = all_columns
        .iter()
        .filter(|(_, _, constant)| !constant)
        .count();
    let peaks = memory_bounds(program, n, tile, groups, all_columns.len(), evaluated)?;
    let plan = Plan::new(&peaks, options)?;
    let mut outputs: Vec<_> = (0..groups).map(|_| vec![Fr::ZERO; n]).collect();
    let pointers: Vec<*mut [u64; 4]> = outputs.iter_mut().map(|v| v.as_mut_ptr().cast()).collect();
    let columns = quotient::resident_column_descriptors(all_columns);
    let mut total = [0; 4];
    let _group = plan.acquire(&peaks, options)?;
    let started = std::time::Instant::now();
    check(
        unsafe {
            multi_stark_kzg_lookup_distributed(
                plan.ordinals.as_ptr(),
                tile,
                options.force_host,
                parameters,
                columns.as_ptr(),
                columns.len(),
                evaluated,
                program.nodes.as_ptr(),
                program.nodes.len(),
                program.lookups.as_ptr(),
                program.lookups.len(),
                program.args.as_ptr(),
                program.args.len(),
                pointers.as_ptr(),
                &mut total,
            )
        },
        "distributed lookup",
    );
    tracing::info!(
        ordinals = ?plan.ordinals, ?peaks, rows = n, tile, evaluated, groups,
        seconds = started.elapsed().as_secs_f64(),
        "KZG lookup distributed across peer pairs"
    );
    Some((outputs, Scalar(Fr::new_unchecked(BigInt(total)))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distributed_lookup_phase_bounds() {
        let program = Program {
            nodes: vec![],
            roots: vec![],
            lookups: vec![],
            args: vec![],
            slots: 16,
        };
        let peaks = memory_bounds(&program, 1 << 27, 1 << 18, 4, 18, 18).unwrap();
        assert!(peaks.iter().all(|&peak| peak > 45 << 30 && peak < 46 << 30));
        let larger = memory_bounds(&program, 1 << 27, 1 << 18, 21, 18, 18).unwrap();
        assert!(larger.iter().all(|&peak| peak > 85 << 30));
        assert!(memory_bounds(&program, 1 << 27, 1 << 18, usize::MAX, 18, 18).is_none());
    }
}
