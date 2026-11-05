use super::*;
use crate::config::ProofConfig;
use crate::cuda::witness::commit_with;
use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config};
use crate::witness::{TraceGenerator, TraceSource};
use p3_field::coset::TwoAdicMultiplicativeCoset;

/// Device memory owned by a test, freed on drop.
fn device_allocation(bytes: usize) -> CudaLde {
    let width = 2;
    let height = (bytes / (8 * width)).next_power_of_two();
    let device = configured_device();
    let mut handle = core::ptr::null_mut();
    // SAFETY: the out-pointer is valid for writing; success transfers the
    // allocation to the returned handle.
    let status = unsafe { multi_stark_cuda_zero_lde_create(device, &mut handle, height, width) };
    check_cuda(status, "test device allocation");
    CudaLde {
        device_id: device,
        handle: NonNull::new(handle).expect("null test allocation"),
        height,
        width,
    }
}

/// Device state several generated sources share, released on request and
/// allocated again by the next tile, as a population record is.
struct SharedState {
    bytes: usize,
    resident: std::sync::Mutex<Option<CudaLde>>,
    releases: std::sync::atomic::AtomicUsize,
}

impl SharedState {
    fn new(bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            bytes,
            resident: std::sync::Mutex::new(None),
            releases: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}

/// A generated source serving the rows of a host matrix. Each tile write
/// allocates the workspace it declares, and the shared state if released.
struct HostRows {
    matrix: RowMajorMatrix<Goldilocks>,
    state: Option<Arc<SharedState>>,
}

const WRITER_WORKSPACE: usize = 1 << 20;

impl HostRows {
    fn new(matrix: RowMajorMatrix<Goldilocks>, state: Option<Arc<SharedState>>) -> Self {
        Self { matrix, state }
    }

    fn runs(
        &self,
        first: usize,
        rows: usize,
        mut copy: impl FnMut(usize, &[Goldilocks]) -> Result<(), String>,
    ) -> Result<(), String> {
        let (width, height) = (self.matrix.width(), self.matrix.height());
        let mut done = 0;
        while done < rows {
            let row = (first + done) % height;
            let run = (height - row).min(rows - done);
            copy(
                done * width,
                &self.matrix.values[row * width..(row + run) * width],
            )?;
            done += run;
        }
        Ok(())
    }
}

impl TraceGenerator<Goldilocks> for HostRows {
    fn height(&self) -> usize {
        self.matrix.height()
    }

    fn width(&self) -> usize {
        self.matrix.width()
    }

    fn write_rows(&self, first: usize, output: &mut [Goldilocks]) {
        self.runs(
            first,
            output.len() / self.matrix.width(),
            |offset, values| {
                output[offset..offset + values.len()].copy_from_slice(values);
                Ok(())
            },
        )
        .unwrap();
    }

    fn device_workspace_bytes(&self) -> usize {
        WRITER_WORKSPACE
    }

    fn device_state_bytes(&self) -> usize {
        self.state.as_ref().map_or(0, |state| state.bytes)
    }

    fn device_state_resident(&self) -> bool {
        self.state
            .as_ref()
            .is_none_or(|state| state.resident.lock().unwrap().is_some())
    }

    fn release_device(&self, _device_id: i32) {
        if let Some(state) = &self.state
            && state.resident.lock().unwrap().take().is_some()
        {
            state
                .releases
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn write_device_rows(&self, view: DeviceTraceView<'_>) -> Result<(), String> {
        if let Some(state) = &self.state {
            let mut resident = state.resident.lock().unwrap();
            if resident.is_none() {
                *resident = Some(device_allocation(state.bytes));
            }
        }
        let _workspace = device_allocation(WRITER_WORKSPACE);
        self.runs(view.first_row(), view.rows(), |offset, values| {
            // SAFETY: the tile holds `rows * width` words from `as_mut_ptr`,
            // and `offset + values.len()` stays within it.
            let status = unsafe {
                multi_stark_cuda_copy_to_device(
                    view.device_id(),
                    view.as_mut_ptr().add(offset),
                    values.as_ptr().cast(),
                    values.len(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(format!("tile upload failed with CUDA status {status}"))
            }
        })
    }
}

#[test]
#[ignore = "requires a CUDA GPU; commits generated and host sources through both prepared constructions"]
fn generated_sources_commit_like_host_matrices() {
    let config = GoldilocksBlake3Config::with_device(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 8,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
        Some(configured_device()),
    );
    let pcs: &crate::types::InnerPcs = config.pcs();
    // Without an override the batch fits resident on any CUDA device.
    // Heights repeat so that height groups mix both kinds of source. The
    // 2^20 x 10 matrix puts the batch over the concurrent-construction
    // threshold, and the 4100-column group is hashed on the host.
    let shapes = [
        (1 << 4, 3),
        (1 << 10, 7),
        (1 << 10, 2),
        (1 << 6, 4100),
        (1 << 16, 5),
        (1 << 20, 10),
        (1 << 12, 1),
    ];
    let matrices: Vec<_> = shapes
        .iter()
        .enumerate()
        .map(|(seed, &(height, width))| {
            RowMajorMatrix::new(
                (0..height * width)
                    .map(|index| Goldilocks::from_u64((index as u64 * 0x9e37_79b9) ^ seed as u64))
                    .collect(),
                width,
            )
        })
        .collect();
    const STATE: usize = 8 << 20;
    let state = SharedState::new(STATE);
    let evaluations = |generated: &dyn Fn(usize) -> bool| {
        matrices
            .iter()
            .enumerate()
            .map(|(index, matrix)| {
                let domain = TwoAdicMultiplicativeCoset::new(
                    Goldilocks::ONE,
                    log2_strict_usize(matrix.height()),
                )
                .unwrap();
                let source = if generated(index) {
                    TraceSource::Generated(Arc::new(HostRows::new(
                        matrix.clone(),
                        Some(Arc::clone(&state)),
                    )))
                } else {
                    TraceSource::Host(matrix.clone())
                };
                (domain, source)
            })
            .collect::<Vec<_>>()
    };
    let (expected, _) = commit_with(pcs, evaluations(&|_| false), None);
    let patterns: [&dyn Fn(usize) -> bool; 3] =
        [&|index| index % 2 == 0, &|index| index % 2 == 1, &|_| true];
    for keep_traces in [None, Some(true), Some(false)] {
        for generated in patterns {
            let (commitment, data) = commit_with(pcs, evaluations(generated), keep_traces);
            assert_eq!(commitment, expected, "keep_traces={keep_traces:?}");
            for (index, &(height, width)) in shapes.iter().enumerate() {
                let host_hashed = width * size_of::<Goldilocks>() > mmcs::MAX_DEVICE_LEAF_ROW_BYTES;
                let Some(lde) = data.resident(index) else {
                    assert!(
                        host_hashed,
                        "matrix {index} of height {height} left the device"
                    );
                    continue;
                };
                assert_eq!(
                    lde.has_device_trace(),
                    keep_traces != Some(false),
                    "matrix {index}: device trace residency"
                );
                // A released trace regenerates through its generator, which
                // may first have to restore the shared state.
                let regeneration = if keep_traces == Some(false) && generated(index) {
                    STATE + WRITER_WORKSPACE
                } else {
                    0
                };
                assert_eq!(data.regeneration_bytes(index), regeneration);
            }
            // Releasing the shared state makes the next generated commitment
            // restore it.
            data.resident(0)
                .into_iter()
                .chain(data.resident(1))
                .for_each(CudaLde::release_generator_device);
        }
    }
    assert!(state.releases.load(std::sync::atomic::Ordering::Relaxed) > 0);
}

/// A batch whose lookups run short of device memory releases the generators'
/// shared state during lookup admission. Lookups that then regenerate a
/// trace must admit restoring that state, or fall back to the host, rather
/// than fail to allocate it; the proof must equal host population's.
#[test]
#[ignore = "requires exclusive access to a CUDA device: fills most of its memory"]
fn generated_traces_prove_identically_under_lookup_pressure() {
    use crate::expr::Expr;
    use crate::lookup::Lookup;
    use crate::system::{CircuitInputs, System, SystemWitness};
    use crate::witness::PreparedWitness;
    let config = GoldilocksBlake3Config::with_device(
        CommitmentParameters {
            log_blowup: 1,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 8,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
        },
        Some(configured_device()),
    );
    let inputs = [false, true].map(|pull| {
        let args = vec![Expr::main(0), Expr::main_next(1)];
        CircuitInputs {
            main_width: 2,
            lookups: vec![if pull {
                Lookup::pull(Expr::constant(Goldilocks::ONE), args)
            } else {
                Lookup::push(Expr::constant(Goldilocks::ONE), args)
            }],
            ..Default::default()
        }
    });
    let (system, key) = System::new(config, inputs);
    let height: usize = 1 << 20;
    let trace = RowMajorMatrix::new(
        (0..2 * height)
            .map(|index| Goldilocks::from_u64(index as u64 * 0x9e37_79b9))
            .collect(),
        2,
    );
    let traces = vec![trace.clone(), trace];
    let no_claims: &[&[Goldilocks]] = &[];
    let host = SystemWitness::from_stage_1(traces.clone(), &system);
    let lookups = host.lookups.clone();
    let expected = system.prove_multiple_claims(&key, no_claims, host);

    let (_, total) = device_memory_info(configured_device());
    let state = SharedState::new(total / 32);
    let witness = PreparedWitness {
        traces: traces
            .into_iter()
            .map(|trace| {
                TraceSource::Generated(Arc::new(HostRows::new(trace, Some(Arc::clone(&state)))))
            })
            .collect(),
        lookups,
    };
    let stage_1 = system.prove_stage_1(witness);
    // Without their device traces the graph lookups regenerate rows, which
    // restores the shared state once lookup admission has released it.
    for index in 0..2 {
        let lde = stage_1.stage_1_trace_data.resident(index).unwrap();
        // SAFETY: commitment is complete and nothing reads the trace until
        // the lookups regenerate it.
        unsafe { lde.release_trace() };
    }
    // Leave less free than one lookup job's output and temporaries, so that
    // admission releases the state and restoring it could not fit.
    let (free, _) = device_memory_info(configured_device());
    let mut ballast = Vec::new();
    let mut remaining = free.saturating_sub(total / 4096);
    while remaining >= 1 << 20 {
        let chunk = (remaining / 2 + 1).next_power_of_two() / 2;
        let chunk = chunk.min(1 << 30);
        ballast.push(device_allocation(chunk));
        remaining -= chunk;
    }
    let proof = system.prove_committed(&key, no_claims, stage_1);
    drop(ballast);
    assert!(state.releases.load(std::sync::atomic::Ordering::Relaxed) > 0);
    system.verify_multiple_claims(no_claims, &proof).unwrap();
    assert_eq!(
        bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap(),
        bincode::serde::encode_to_vec(&expected, bincode::config::standard()).unwrap()
    );
}
