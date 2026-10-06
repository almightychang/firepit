use super::Q8_BLOCK;
use cuda_async::device_operation::DeviceOp;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{Tensor, TensorView};
use std::sync::Arc;

pub const NVFP4_GROUP: usize = 16;
const GROUPS_PER_BLOCK: usize = Q8_BLOCK / NVFP4_GROUP;

#[derive(Clone)]
pub struct Nvfp4Matrix {
    pub(crate) packed: Arc<Tensor<f4e2m1fnx2>>,
    pub(crate) scales: Arc<Tensor<f8e4m3fn>>,
    pub(crate) row_scales: Arc<Tensor<f32>>,
    pub rows: usize,
    pub cols: usize,
}

impl Nvfp4Matrix {
    pub(crate) fn packed_matrix(&self) -> TensorView<'_, f4e2m1fnx2> {
        self.packed
            .view(&[self.rows, self.cols / 2])
            .expect("NVFP4 storage has the declared shape")
    }

    pub(crate) fn scale_matrix(&self) -> TensorView<'_, f8e4m3fn> {
        self.scales
            .view(&[self.rows, self.cols / NVFP4_GROUP])
            .expect("NVFP4 scales have the declared shape")
    }
}

pub struct HostNvfp4Matrix {
    pub packed: Vec<f4e2m1fnx2>,
    pub scales: Vec<f8e4m3fn>,
    pub row_scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

fn is_nan_e4m3(scale: f8e4m3fn) -> bool {
    scale.0 & 0x7f == 0x7f
}

impl HostNvfp4Matrix {
    pub fn from_parts(
        packed: Vec<f4e2m1fnx2>,
        scales: Vec<f8e4m3fn>,
        global_scale: f32,
        rows: usize,
        cols: usize,
    ) -> Result<Self, &'static str> {
        if rows == 0
            || cols == 0
            || cols % Q8_BLOCK != 0
            || rows.checked_mul(cols / 2) != Some(packed.len())
            || rows.checked_mul(cols / NVFP4_GROUP) != Some(scales.len())
            || scales.iter().any(|&scale| is_nan_e4m3(scale))
            || !global_scale.is_finite()
            || global_scale < 0.0
        {
            return Err("invalid NVFP4 matrix, block scales, or global scale");
        }
        Ok(Self {
            packed,
            scales,
            row_scales: vec![global_scale; rows],
            rows,
            cols,
        })
    }

    pub fn zeros(rows: usize, cols: usize) -> Result<Self, &'static str> {
        Self::from_parts(
            vec![f4e2m1fnx2::from_bits(0); rows * cols / 2],
            vec![f8e4m3fn(0); rows * cols / NVFP4_GROUP],
            0.0,
            rows,
            cols,
        )
    }

    pub fn stack(parts: Vec<Self>) -> Result<Self, &'static str> {
        let mut parts = parts.into_iter();
        let mut stacked = parts.next().ok_or("no NVFP4 matrices to stack")?;
        for part in parts {
            if part.cols != stacked.cols {
                return Err("stacked NVFP4 matrices differ in columns");
            }
            stacked.packed.extend(part.packed);
            stacked.scales.extend(part.scales);
            stacked.row_scales.extend(part.row_scales);
            stacked.rows += part.rows;
        }
        Ok(stacked)
    }

    pub fn upload(
        self,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Nvfp4Matrix, cutile::error::Error> {
        let blocks = self.cols / Q8_BLOCK;
        let packed = api::copy_host_vec_to_device(&Arc::new(self.packed))
            .reshape(&[self.rows, blocks, Q8_BLOCK / 2])
            .sync_on(stream)?;
        let scales = api::copy_host_vec_to_device(&Arc::new(self.scales))
            .reshape(&[self.rows, blocks, GROUPS_PER_BLOCK])
            .sync_on(stream)?;
        let row_scales = api::copy_host_vec_to_device(&Arc::new(self.row_scales))
            .reshape(&[1, self.rows])
            .sync_on(stream)?;
        Ok(Nvfp4Matrix {
            packed: Arc::new(packed),
            scales: Arc::new(scales),
            row_scales: Arc::new(row_scales),
            rows: self.rows,
            cols: self.cols,
        })
    }
}
