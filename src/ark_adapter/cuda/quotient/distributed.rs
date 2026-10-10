use super::*;

use super::super::distributed::{self as shared, COPY_BYTES, Plan};
pub(super) use shared::Options;

pub(super) fn options() -> Options {
    Options::from_env(
        "MULTI_STARK_KZG_CUDA_QUOTIENT_TILE_ROWS",
        "MULTI_STARK_KZG_CUDA_QUOTIENT_HOST_STAGING",
    )
}

pub(super) fn enabled(single_rejected: bool) -> bool {
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| {
        match std::env::var("MULTI_STARK_KZG_CUDA_DISTRIBUTED_QUOTIENT").as_deref() {
            Ok("0") | Err(std::env::VarError::NotPresent) => 0,
            Ok("1") => 1,
            Ok("force") => 2,
            _ => panic!("MULTI_STARK_KZG_CUDA_DISTRIBUTED_QUOTIENT must be 0, 1 or force"),
        }
    }) {
        1 => single_rejected,
        2 => true,
        _ => false,
    }
}

unsafe extern "C" {
    fn multi_stark_kzg_quotient_distributed(
        devices: *const i32,
        tile_rows: usize,
        force_host: bool,
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
        shifts: *const [u64; 4],
        inverse_vanishing: *const [u64; 4],
        outputs: *const *mut [u64; 4],
    ) -> i32;
}

fn memory_bounds(
    program: &Program,
    n: usize,
    tile: usize,
    columns: usize,
    evaluated: usize,
    publics: usize,
) -> Option<[usize; 4]> {
    let metadata = [
        allocation::<Instruction>(program.nodes.len())?,
        allocation::<u32>(program.roots.len())?,
        allocation::<Lookup>(program.lookups.len())?,
        allocation::<u32>(program.args.len())?,
        allocation::<Column>(columns)?,
        allocation::<Fr>(publics)?,
        HEADROOM,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    let scratch = tile
        .div_ceil(TILE)
        .min(MAX_BLOCKS)
        .checked_mul(TILE)?
        .checked_mul(program.slots.max(1))?;
    let tile_elements = tile
        .checked_add(1)?
        .checked_mul(evaluated)?
        .checked_mul(2)?;
    let mut peaks = [0; 4];
    for (index, peak) in peaks.iter_mut().enumerate() {
        let owned = evaluated / 2 + usize::from(index % 2 == 0 && evaluated % 2 != 0);
        let sweep = owned
            .checked_mul(n)?
            .checked_add(n)?
            .checked_add(n / 2)?
            .checked_add(tile_elements)?
            .checked_add(scratch)?;
        let sweep = allocation::<Fr>(sweep)?.checked_add(metadata)?;
        let merge = allocation::<Fr>(n.checked_mul(2)?.checked_add(n / 2)?)?
            .checked_add(COPY_BYTES.checked_mul(2)?)?
            .checked_add(HEADROOM)?;
        *peak = if index == 0 { sweep.max(merge) } else { sweep };
    }
    Some(peaks)
}

pub(super) fn coefficients(
    parameters: &Parameters,
    program: &Program,
    all_columns: &[PolynomialView<'_>],
    publics: &[Scalar],
    shifts: &[Fr],
    inverse_vanishing: &[Fr],
    options: Options,
) -> Option<Vec<Vec<Fr>>> {
    if !(10..=27).contains(&parameters.trace_log)
        || parameters.quotient_log != parameters.trace_log + 1
        || shifts.len() != 2
        || inverse_vanishing.len() != 2
        || !options.tile_rows.is_power_of_two()
        || !(128..=1 << 20).contains(&options.tile_rows)
    {
        return None;
    }
    let n = 1usize.checked_shl(parameters.trace_log)?;
    let tile = options.tile_rows.min(n / 2);
    let evaluated = all_columns
        .iter()
        .filter(|(_, _, constant)| !constant)
        .count();
    let peaks = memory_bounds(
        program,
        n,
        tile,
        all_columns.len(),
        evaluated,
        publics.len(),
    )?;
    let plan = Plan::new(&peaks, options)?;
    let mut outputs = vec![vec![Fr::ZERO; n]; 2];
    let pointers: Vec<*mut [u64; 4]> = outputs.iter_mut().map(|v| v.as_mut_ptr().cast()).collect();
    let columns = resident_column_descriptors(all_columns);
    let ordinals = plan.ordinals;
    let _group = plan.acquire(&peaks, options)?;
    let started = std::time::Instant::now();
    check(
        unsafe {
            multi_stark_kzg_quotient_distributed(
                ordinals.as_ptr(),
                tile,
                options.force_host,
                parameters,
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
                publics.as_ptr().cast(),
                publics.len(),
                shifts.as_ptr().cast(),
                inverse_vanishing.as_ptr().cast(),
                pointers.as_ptr(),
            )
        },
        "distributed quotient",
    );
    tracing::info!(
        ?ordinals,
        ?peaks,
        rows = n,
        tile,
        evaluated,
        seconds = started.elapsed().as_secs_f64(),
        "KZG quotient distributed across peer pairs"
    );
    Some(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distributed_quotient_plan_bounds_and_pairs() {
        let access = vec![
            vec![false, true, false, false],
            vec![true, false, false, false],
            vec![false, false, false, true],
            vec![false, false, true, false],
        ];
        assert_eq!(shared::pair_indices(&access), Some([0, 1, 2, 3]));
        let mut asymmetric = access.clone();
        asymmetric[3][2] = false;
        assert!(shared::pair_indices(&asymmetric).is_none());
        let program = Program {
            nodes: vec![],
            roots: vec![],
            lookups: vec![],
            args: vec![],
            slots: 16,
        };
        let peaks = memory_bounds(&program, 1 << 27, 1 << 18, 22, 22, 4).unwrap();
        assert!(
            peaks
                .iter()
                .all(|&bytes| bytes > 51 << 30 && bytes < 52 << 30)
        );
        assert!(memory_bounds(&program, usize::MAX, 1 << 18, 22, 22, 4).is_none());
    }
}
