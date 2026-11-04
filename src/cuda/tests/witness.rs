use super::*;
use crate::config::ProofConfig;
use crate::cuda::witness::commit_with;
use crate::types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config};
use crate::witness::{TraceGenerator, TraceSource};
use p3_field::coset::TwoAdicMultiplicativeCoset;

/// A generated source serving the rows of a host matrix.
struct HostRows(RowMajorMatrix<Goldilocks>);

impl HostRows {
    fn runs(
        &self,
        first: usize,
        rows: usize,
        mut copy: impl FnMut(usize, &[Goldilocks]) -> Result<(), String>,
    ) -> Result<(), String> {
        let (width, height) = (self.0.width(), self.0.height());
        let mut done = 0;
        while done < rows {
            let row = (first + done) % height;
            let run = (height - row).min(rows - done);
            copy(
                done * width,
                &self.0.values[row * width..(row + run) * width],
            )?;
            done += run;
        }
        Ok(())
    }
}

impl TraceGenerator<Goldilocks> for HostRows {
    fn height(&self) -> usize {
        self.0.height()
    }

    fn width(&self) -> usize {
        self.0.width()
    }

    fn write_rows(&self, first: usize, output: &mut [Goldilocks]) {
        self.runs(first, output.len() / self.0.width(), |offset, values| {
            output[offset..offset + values.len()].copy_from_slice(values);
            Ok(())
        })
        .unwrap();
    }

    fn write_device_rows(&self, view: DeviceTraceView<'_>) -> Result<(), String> {
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
                    TraceSource::Generated(Arc::new(HostRows(matrix.clone())))
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
            }
        }
    }
}
