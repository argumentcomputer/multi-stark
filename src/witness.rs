//! Owned, deterministic main-trace sources. Producers do not allocate device memory.

use std::sync::Arc;

use p3_field::Field;
use p3_matrix::{Matrix, dense::RowMajorMatrix};

/// A frozen source which reproduces the same canonical cells on every call.
pub trait TraceGenerator<F: Field>: Send + Sync {
    fn height(&self) -> usize;
    fn width(&self) -> usize;
    fn host_bytes(&self) -> usize;
    /// Fill contiguous rows, wrapping at the padded height for lookup halos.
    fn write_rows(&self, first: usize, output: &mut [F]);

    /// Fill a device tile synchronously on its owning device and calling stream.
    #[cfg(feature = "cuda")]
    fn write_device_rows(&self, output: crate::cuda::DeviceTraceView<'_>) -> Result<(), String>;

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

#[derive(Clone)]
pub struct PreparedWitness<F: Field> {
    pub traces: Vec<TraceSource<F>>,
    pub lookups: Vec<crate::lookup::LookupValues<F>>,
}

impl<F: Field> From<crate::system::SystemWitness<F>> for PreparedWitness<F> {
    fn from(witness: crate::system::SystemWitness<F>) -> Self {
        Self {
            traces: witness.traces.into_iter().map(TraceSource::Host).collect(),
            lookups: witness.lookups,
        }
    }
}
