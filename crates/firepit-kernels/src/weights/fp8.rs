use super::DenseMatrix;
use cuda_async::device_operation::DeviceOp;
use cuda_core::f8e4m3fn;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::Tensor;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8Scaling {
    Blocks { block_n: usize, block_k: usize },
    Rows,
}

#[derive(Clone)]
pub struct Fp8Matrix {
    pub weights: DenseMatrix<f8e4m3fn>,
    pub scales: Arc<Tensor<f32>>,
    pub scaling: Fp8Scaling,
}

pub struct HostFp8Matrix {
    pub bits: Vec<f8e4m3fn>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub scaling: Fp8Scaling,
}

impl HostFp8Matrix {
    pub fn from_parts(
        bits: Vec<f8e4m3fn>,
        scales: Vec<f32>,
        rows: usize,
        cols: usize,
        block_n: usize,
        block_k: usize,
    ) -> Result<Self, &'static str> {
        if block_n == 0
            || block_k == 0
            || rows % block_n != 0
            || cols % block_k != 0
            || (rows / block_n).checked_mul(cols / block_k) != Some(scales.len())
        {
            return Err("invalid FP8 scale grid");
        }
        Self::new(
            bits,
            scales,
            rows,
            cols,
            Fp8Scaling::Blocks { block_n, block_k },
        )
    }

    pub fn with_row_scales(
        bits: Vec<f8e4m3fn>,
        scales: Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, &'static str> {
        if scales.len() != rows {
            return Err("FP8 row scales must have one value per row");
        }
        Self::new(bits, scales, rows, cols, Fp8Scaling::Rows)
    }

    pub fn zeros(rows: usize, cols: usize, scaling: Fp8Scaling) -> Result<Self, &'static str> {
        let bits = vec![f8e4m3fn(0); rows * cols];
        match scaling {
            Fp8Scaling::Blocks { block_n, block_k } => {
                let scales = vec![0.0; (rows / block_n) * (cols / block_k)];
                Self::from_parts(bits, scales, rows, cols, block_n, block_k)
            }
            Fp8Scaling::Rows => Self::with_row_scales(bits, vec![0.0; rows], rows, cols),
        }
    }

    fn new(
        bits: Vec<f8e4m3fn>,
        scales: Vec<f32>,
        rows: usize,
        cols: usize,
        scaling: Fp8Scaling,
    ) -> Result<Self, &'static str> {
        if rows == 0
            || cols == 0
            || cols % super::Q8_BLOCK != 0
            || rows.checked_mul(cols) != Some(bits.len())
            || scales.iter().any(|scale| !scale.is_finite())
        {
            return Err("invalid FP8 matrix or scales");
        }
        Ok(Self {
            bits,
            scales,
            rows,
            cols,
            scaling,
        })
    }

    pub fn stack(parts: Vec<Self>) -> Result<Self, &'static str> {
        let mut parts = parts.into_iter();
        let mut stacked = parts.next().ok_or("no FP8 matrices to stack")?;
        for part in parts {
            if (part.cols, part.scaling) != (stacked.cols, stacked.scaling) {
                return Err("stacked FP8 matrices differ in columns or scaling");
            }
            stacked.bits.extend(part.bits);
            stacked.scales.extend(part.scales);
            stacked.rows += part.rows;
        }
        Ok(stacked)
    }

    pub fn upload(
        self,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Fp8Matrix, cutile::error::Error> {
        let grid = match self.scaling {
            Fp8Scaling::Blocks { block_n, block_k } => [self.rows / block_n, self.cols / block_k],
            Fp8Scaling::Rows => [1, self.rows],
        };
        let scales = api::copy_host_vec_to_device(&Arc::new(self.scales))
            .reshape(&grid)
            .sync_on(stream)?;
        Ok(Fp8Matrix {
            weights: DenseMatrix::upload(self.bits, self.rows, self.cols, stream)?,
            scales: Arc::new(scales),
            scaling: self.scaling,
        })
    }
}
