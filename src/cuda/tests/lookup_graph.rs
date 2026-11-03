use super::*;
use crate::expr::{CircuitSpec, Expr};
use crate::graph::compile;
use crate::lookup::{Lookup, stage2_width};
use crate::types::GoldilocksBlake3Config;

#[test]
#[ignore = "requires a CUDA GPU; compares graph lookup LDEs over device and host-only traces"]
fn graph_lookup_reads_host_preprocessed_trace() {
    let gpu = CudaDft::without_auxiliary_devices(configured_device());
    let ext = crate::system::extension_params::<GoldilocksBlake3Config>();
    let (lookup_count, group_size, log_blowup) = (5, 2, 1);
    let spec = CircuitSpec {
        main_width: 3,
        preprocessed_width: 2,
        stage2_width: stage2_width(lookup_count, group_size, 2),
        num_publics: 0,
        constraints: vec![],
        ext_constraints: vec![],
        lookups: (0..lookup_count)
            .map(|index| {
                Lookup::push(
                    Expr::preprocessed(0) - Expr::constant(Goldilocks::from_usize(index)),
                    vec![
                        Expr::main_next(0),
                        Expr::preprocessed_next(1),
                        Expr::main(2) * Expr::preprocessed(1),
                    ],
                )
            })
            .collect(),
    };
    let graph = compile(&spec, &ext).unwrap();
    // Several 2^16-row kernel chunks, so tiled next-row reads cross chunk
    // boundaries and wrap around the last row.
    let n = 1 << 18;
    let trace = |width: usize, seed: usize| {
        RowMajorMatrix::new(
            (0..n * width)
                .map(|index| Goldilocks::from_usize((index * index + 17 * index + seed) % 1009))
                .collect(),
            width,
        )
    };
    let prep_trace = trace(2, 41);
    let main_trace = trace(3, 7);
    let prep = gpu.coset_lde_batch_resident(&prep_trace, log_blowup, Goldilocks::GENERATOR);
    let main = gpu.coset_lde_batch_resident(&main_trace, log_blowup, Goldilocks::GENERATOR);
    let beta = [Goldilocks::from_u8(3), Goldilocks::from_u8(5)];
    let gamma = [Goldilocks::from_u8(7), Goldilocks::from_u8(11)];
    let run = |prep_host: Option<&RowMajorMatrix<Goldilocks>>| {
        let (lde, total) = lookup_graph_lde_resident(
            &gpu,
            &graph,
            Some((&prep, prep_host)),
            &main,
            n,
            group_size,
            beta,
            gamma,
            ext.w,
            log_blowup,
        )
        .expect("graph lookup encoding");
        (lde.to_row_major_matrix().values, total)
    };

    let expected = run(None);
    // SAFETY: no CUDA work is in flight on these LDEs.
    unsafe { prep.release_trace() };
    assert!(!prep.has_device_trace());
    assert!(
        run(Some(&prep_trace)) == expected,
        "host preprocessed trace"
    );
    // SAFETY: as above; `main_trace` outlives every later use of `main`.
    unsafe {
        main.release_trace();
        main.attach_trace(&main_trace);
    }
    assert!(
        run(Some(&prep_trace)) == expected,
        "host preprocessed and main traces"
    );
}
