mod format;
mod kernel;

use cuda_async::device_operation::{DeviceOp, GraphNode};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{Partition, Tensor, TensorView};
use cutile::tile_kernel::TileKernel;
pub use format::{FormatError, HostQ8Matrix, Q8_BLOCK};
use half::f16;
use std::sync::Arc;

pub struct Q8Matrix {
    qs: Arc<Tensor<i8>>,
    scales: Arc<Tensor<f16>>,
    pub rows: usize,
    cols: usize,
}

impl HostQ8Matrix {
    pub fn upload(
        &self,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Q8Matrix, cutile::error::Error> {
        let qs = api::copy_host_vec_to_device(&Arc::new(self.qs.clone()))
            .reshape(&[self.rows, self.cols / Q8_BLOCK, Q8_BLOCK])
            .sync_on(stream)?;
        let scales = api::copy_host_vec_to_device(&Arc::new(self.scales.clone()))
            .reshape(&[self.rows, self.cols / Q8_BLOCK])
            .sync_on(stream)?;
        Ok(Q8Matrix {
            qs: Arc::new(qs),
            scales: Arc::new(scales),
            rows: self.rows,
            cols: self.cols,
        })
    }
}

#[derive(Clone, Copy)]
pub struct GemvTile {
    pub rows: usize,
    pub blocks: usize,
}

pub const DEFAULT_GEMV_TILE: GemvTile = GemvTile {
    rows: 16,
    blocks: 4,
};

impl GemvTile {
    pub fn fits(self, rows: usize, cols: usize) -> bool {
        rows % self.rows == 0 && cols % Q8_BLOCK == 0 && (cols / Q8_BLOCK) % self.blocks == 0
    }
}

pub fn q8_gemv<'a>(
    w: &Q8Matrix,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
) -> impl DeviceOp<
    Output = (
        Partition<&'a mut Tensor<f32>>,
        Arc<Tensor<i8>>,
        Arc<Tensor<f16>>,
        &'a TensorView<'a, f32>,
        i32,
    ),
> + GraphNode
+ 'a {
    let batch = x.shape()[0] as usize;
    assert!(batch > 0 && tile.rows > 0 && tile.blocks > 0);
    assert_eq!(
        x.shape(),
        &[batch as i32, (w.cols / Q8_BLOCK) as i32, Q8_BLOCK as i32]
    );
    assert!(tile.fits(w.rows, w.cols));
    kernel::q8_gemv_kernel(
        y,
        w.qs.clone(),
        w.scales.clone(),
        x,
        (w.cols / Q8_BLOCK) as i32,
    )
    .generics(vec![
        batch.to_string(),
        tile.rows.to_string(),
        tile.blocks.to_string(),
    ])
}
