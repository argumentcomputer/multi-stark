//! Minimal row-major matrix used for traces.
//!
//! Mirrors the subset of `p3_matrix::dense::RowMajorMatrix` actually needed
//! by the prover/verifier.

use crate::types::Val;

#[derive(Clone, Debug)]
pub struct Matrix<F = Val> {
    pub values: Vec<F>,
    pub width: usize,
}

impl<F: Clone> Matrix<F> {
    pub fn new(values: Vec<F>, width: usize) -> Self {
        if width == 0 {
            assert!(values.is_empty(), "non-empty values with zero width");
        } else {
            assert_eq!(values.len() % width, 0, "non-rectangular matrix");
        }
        Self { values, width }
    }

    #[inline]
    pub fn height(&self) -> usize {
        if self.width == 0 {
            0
        } else {
            self.values.len() / self.width
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn row(&self, i: usize) -> &[F] {
        &self.values[i * self.width..(i + 1) * self.width]
    }

    pub fn rows(&self) -> impl Iterator<Item = &[F]> {
        self.values.chunks_exact(self.width)
    }

    /// Returns each column as an owned `Vec<F>`. Used to feed iFFT.
    pub fn columns(&self) -> Vec<Vec<F>> {
        let h = self.height();
        let w = self.width;
        let mut cols: Vec<Vec<F>> = (0..w).map(|_| Vec::with_capacity(h)).collect();
        for row in self.rows() {
            for (c, v) in row.iter().enumerate() {
                cols[c].push(v.clone());
            }
        }
        cols
    }

    /// Build a matrix from a list of equally-sized columns.
    pub fn from_columns(cols: &[Vec<F>]) -> Self {
        let width = cols.len();
        if width == 0 {
            return Self {
                values: vec![],
                width: 0,
            };
        }
        let height = cols[0].len();
        for c in cols {
            assert_eq!(c.len(), height, "columns must have the same length");
        }
        let mut values = Vec::with_capacity(height * width);
        for i in 0..height {
            for c in cols {
                values.push(c[i].clone());
            }
        }
        Self { values, width }
    }
}
