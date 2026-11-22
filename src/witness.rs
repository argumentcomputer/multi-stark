//! Owned, deterministic main-trace sources. A producer allocates device memory
//! only for the workspace it declares to commitment admission.

use std::sync::Arc;

use crate::traits::Field;
use p3_matrix::{Matrix, dense::RowMajorMatrix};

/// A frozen source which reproduces the same canonical cells on every call.
pub trait TraceGenerator<F: Field>: Send + Sync {
    fn height(&self) -> usize;
    fn width(&self) -> usize;
    /// Fill contiguous rows, wrapping at the padded height for lookup halos.
    fn write_rows(&self, first: usize, output: &mut [F]);

    /// Fill a device tile synchronously on its owning device and calling stream.
    #[cfg(feature = "cuda")]
    fn write_device_rows(&self, output: crate::cuda::DeviceTraceView<'_>) -> Result<(), String>;

    /// Device scratch one `write_device_rows` call may allocate besides its
    /// output tile. Up to [`crate::cuda::MAX_CONCURRENT_LDES`] calls on
    /// different sources can run at once, and commitment admission reserves
    /// this for each.
    #[cfg(feature = "cuda")]
    fn device_workspace_bytes(&self) -> usize {
        0
    }

    /// Device bytes of the state [`Self::release_device`] frees, such as
    /// uploaded seeds; the next tile allocates it again. Generated sources of
    /// one batch are taken to share this state, so admission reserves the
    /// largest of them once rather than their sum.
    #[cfg(feature = "cuda")]
    fn device_state_bytes(&self) -> usize {
        0
    }

    /// Whether the state [`Self::device_state_bytes`] counts is on the
    /// device now, so that writing a tile allocates none of it.
    #[cfg(feature = "cuda")]
    fn device_state_resident(&self) -> bool {
        true
    }

    /// Free whatever the source keeps on `device_id` to serve tiles faster,
    /// such as seeds left resident between a commitment and the lookup pass
    /// that regenerates rows from them. Tiles must still be served afterwards.
    #[cfg(feature = "cuda")]
    fn release_device(&self, _device_id: i32) {}
}

#[derive(Clone)]
pub enum TraceSource<F: Field> {
    Host(RowMajorMatrix<F>),
    Generated(Arc<dyn TraceGenerator<F>>),
}

impl<F: Field> TraceSource<F> {
    pub fn height(&self) -> usize {
        match self {
            Self::Host(m) => m.height(),
            Self::Generated(g) => g.height(),
        }
    }

    pub fn width(&self) -> usize {
        match self {
            Self::Host(m) => m.width(),
            Self::Generated(g) => g.width(),
        }
    }

    pub fn materialize(self) -> RowMajorMatrix<F> {
        match self {
            Self::Host(m) => m,
            Self::Generated(g) => {
                let mut values = vec![
                    F::ZERO;
                    g.height()
                        .checked_mul(g.width())
                        .expect("trace size overflow")
                ];
                g.write_rows(0, &mut values);
                RowMajorMatrix::new(values, g.width())
            }
        }
    }
}

impl<F: Field> From<RowMajorMatrix<F>> for TraceSource<F> {
    fn from(matrix: RowMajorMatrix<F>) -> Self {
        Self::Host(matrix)
    }
}

#[derive(Clone)]
pub struct PreparedWitness<F: Field> {
    pub traces: Vec<TraceSource<F>>,
    pub lookups: Vec<crate::lookup::LookupValues<F>>,
}

impl<F: Field> PreparedWitness<F> {
    /// Builds a lazy prepared witness from stage-1 trace sources without
    /// materializing [`crate::lookup::LookupValues`] upfront.
    ///
    /// Lookup expressions are evaluated on demand in bounded row chunks during
    /// stage-2 trace construction, producing proofs byte-identical to
    /// [`crate::system::SystemWitness::from_stage_1`].
    ///
    /// # Panics
    /// Panics if the number of traces differs from the number of circuits, or
    /// if an active circuit with a preprocessed trace receives a trace of a
    /// different height.
    pub fn from_stage_1_lazy<SC>(
        traces: Vec<TraceSource<F>>,
        system: &crate::system::System<SC>,
    ) -> Self
    where
        SC: crate::config::ProofConfig,
        SC::Pcs: crate::traits::Pcs<F = F>,
    {
        assert_eq!(
            traces.len(),
            system.circuits.len(),
            "expected one trace per circuit"
        );
        let lookups = traces
            .iter()
            .zip(&system.circuits)
            .enumerate()
            .map(|(i, (trace, circuit))| {
                let height = trace.height();
                if circuit.preprocessed_height != 0 && height != 0 {
                    assert_eq!(
                        height, circuit.preprocessed_height,
                        "circuit {i}: main trace height must equal preprocessed trace height"
                    );
                }
                let slot_widths: Vec<usize> = circuit
                    .graph
                    .lookups
                    .iter()
                    .map(|lookup| lookup.args.len())
                    .collect();
                crate::lookup::LookupValues::shape_only(height, &slot_widths)
            })
            .collect();
        Self { traces, lookups }
    }
}

impl<F: Field> From<crate::system::SystemWitness<F>> for PreparedWitness<F> {
    fn from(witness: crate::system::SystemWitness<F>) -> Self {
        Self {
            traces: witness.traces.into_iter().map(TraceSource::Host).collect(),
            lookups: witness.lookups,
        }
    }
}
