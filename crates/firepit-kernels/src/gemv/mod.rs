mod dense;
mod expert;
mod q8;

pub use expert::{expert_gemv, expert_gemv_view};

use crate::sink::Sink;
use cutile::tensor::{Partition, Tensor, TensorView};

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

pub struct Gemv<'a> {
    weight: &'a Weight,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
}

pub fn gemv<'a>(
    weight: &'a Weight,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
) -> Gemv<'a> {
    Gemv { weight, x, y, tile }
}

impl Gemv<'_> {
    pub fn submit<S: Sink>(self, sink: &mut S) -> Result<(), cuda_async::error::DeviceError> {
        match self.weight {
            Weight::Dense(w) => {
                sink.run(dense::dense_gemv(w, self.x, self.y, self.tile))?;
            }
            Weight::Q8(w) => {
                sink.run(q8::q8_gemv(w, self.x, self.y, self.tile))?;
            }
        }
        Ok(())
    }
}
