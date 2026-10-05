use super::DenseMatrix;
use cuda_async::device_operation::DeviceOp;
use cuda_core::f8e4m3fn;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::Tensor;
use std::sync::Arc;

#[derive(Clone)]
pub struct Fp8Matrix {
    pub weights: DenseMatrix<f8e4m3fn>,
    pub scales: Arc<Tensor<f32>>,
    pub block_n: usize,
    pub block_k: usize,
}

pub struct HostFp8Matrix {
    pub bits: Vec<f8e4m3fn>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub block_n: usize,
    pub block_k: usize,
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
        if rows == 0
            || cols == 0
            || block_n == 0
            || block_k == 0
            || rows % block_n != 0
            || cols % block_k != 0
            || cols % super::Q8_BLOCK != 0
            || rows.checked_mul(cols) != Some(bits.len())
            || rows
                .checked_div(block_n)
                .and_then(|r| r.checked_mul(cols / block_k))
                != Some(scales.len())
        {
            return Err("invalid FP8 matrix or scale grid");
        }
        Ok(Self {
            bits,
            scales,
            rows,
            cols,
            block_n,
            block_k,
        })
    }

    pub fn upload(
        self,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Fp8Matrix, cutile::error::Error> {
        let scales = api::copy_host_vec_to_device(&Arc::new(self.scales))
            .reshape(&[self.rows / self.block_n, self.cols / self.block_k])
            .sync_on(stream)?;
        Ok(Fp8Matrix {
            weights: DenseMatrix::upload(self.bits, self.rows, self.cols, stream)?,
            scales: Arc::new(scales),
            block_n: self.block_n,
            block_k: self.block_k,
        })
    }
}
