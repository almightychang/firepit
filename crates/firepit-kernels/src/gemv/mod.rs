mod dense;
mod expert;
mod fp8;
mod q8;

pub use expert::{expert_gemv, expert_gemv_view};

use crate::moe::ExpertSelection;
use crate::sink::Sink;
use cutile::tensor::{Partition, Tensor, TensorView, TensorViewMut};

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
            Weight::Fp8(w) => {
                sink.run(fp8::dense_gemv(w, self.x, self.y, self.tile))?;
            }
        }
        Ok(())
    }
}

pub struct ExpertGemvView<'a> {
    weight: &'a Weight,
    x: &'a TensorView<'a, f32>,
    selection: &'a ExpertSelection,
    y: Partition<TensorViewMut<'a, f32>>,
    rows_per_expert: usize,
    input_stride: usize,
    tile: GemvTile,
}

pub fn expert_gemv_weight_view<'a>(
    weight: &'a Weight,
    x: &'a TensorView<'a, f32>,
    selection: &'a ExpertSelection,
    y: Partition<TensorViewMut<'a, f32>>,
    rows_per_expert: usize,
    input_stride: usize,
    tile: GemvTile,
) -> ExpertGemvView<'a> {
    ExpertGemvView {
        weight,
        x,
        selection,
        y,
        rows_per_expert,
        input_stride,
        tile,
    }
}

impl ExpertGemvView<'_> {
    pub fn submit<S: Sink>(self, sink: &mut S) -> Result<(), cuda_async::error::DeviceError> {
        match self.weight {
            Weight::Dense(w) => sink.run(expert::expert_gemv_view(
                w,
                self.x,
                self.selection,
                self.y,
                self.rows_per_expert,
                self.input_stride,
                self.tile,
            ))?,
            Weight::Fp8(w) => sink.run(fp8::expert_gemv_view(
                w,
                self.x,
                self.selection,
                self.y,
                self.rows_per_expert,
                self.input_stride,
                self.tile,
            ))?,
            Weight::Q8(_) => {
                return Err(cuda_async::error::DeviceError::Internal(
                    "Q8 expert GEMV is unsupported".into(),
                ));
            }
        }
        Ok(())
    }
}
