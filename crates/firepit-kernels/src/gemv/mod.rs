mod dense;
mod expert;
mod q8;

pub use expert::expert_gemv;

use cuda_async::cuda_graph::Scope;
use cuda_async::device_operation::DeviceOp;
use cutile::DType;
use cutile::core::ElementType;
use cutile::tensor::{Partition, Tensor, TensorView};
use std::sync::Arc;

use crate::weights::{Q8_BLOCK, Weight};

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

pub struct Gemv<'a, T: DType + ElementType> {
    weight: &'a Weight<T>,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
}

pub fn gemv<'a, T: DType + ElementType>(
    weight: &'a Weight<T>,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
) -> Gemv<'a, T> {
    Gemv { weight, x, y, tile }
}

impl<T: DType + ElementType> Gemv<'_, T> {
    pub fn sync_on(
        self,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<(), cuda_async::error::DeviceError> {
        match self.weight {
            Weight::Dense(w) => {
                dense::dense_gemv(w, self.x, self.y, self.tile).sync_on(stream)?;
            }
            Weight::Q8(w) => {
                q8::q8_gemv(w, self.x, self.y, self.tile).sync_on(stream)?;
            }
        }
        Ok(())
    }

    pub fn record(self, scope: &Scope) -> Result<(), cuda_async::error::DeviceError> {
        match self.weight {
            Weight::Dense(w) => {
                scope.record(dense::dense_gemv(w, self.x, self.y, self.tile))?;
            }
            Weight::Q8(w) => {
                scope.record(q8::q8_gemv(w, self.x, self.y, self.tile))?;
            }
        }
        Ok(())
    }
}
