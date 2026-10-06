mod dense;
mod expert;
mod fp8;
mod nvfp4;
mod q8;

pub use expert::{expert_gemv, expert_gemv_view};

use crate::moe::ExpertSelection;
use crate::sink::Sink;
use cutile::tensor::{Partition, Tensor, TensorView, TensorViewMut};
use cutile::tile_kernel::CompileOptions;

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

const FP8_TARGET_ROW_TILES: usize = 64;
const FP8_MAX_ROW_TILE: usize = 128;

pub fn fp8_tile(
    rows: usize,
    cols: usize,
    block_n: usize,
    block_k: usize,
    groups: usize,
) -> GemvTile {
    assert!(rows > 0 && cols > 0 && block_n > 0 && block_k > 0 && groups > 0);
    let target = rows
        .saturating_mul(groups)
        .div_ceil(FP8_TARGET_ROW_TILES)
        .clamp(1, FP8_MAX_ROW_TILE)
        .next_power_of_two();
    let mut tile = GemvTile {
        rows: target,
        blocks: DEFAULT_GEMV_TILE.blocks,
    };
    while rows % tile.rows != 0 || block_n % tile.rows != 0 {
        tile.rows /= 2;
    }
    while (cols / Q8_BLOCK) % tile.blocks != 0 || block_k % (tile.blocks * Q8_BLOCK) != 0 {
        tile.blocks /= 2;
    }
    tile
}

pub fn row_scaled_tile(rows: usize, cols: usize, groups: usize) -> GemvTile {
    fp8_tile(rows, cols, rows, cols, groups)
}

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
    compile_options: Option<CompileOptions>,
}

pub fn gemv<'a>(
    weight: &'a Weight,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
) -> Gemv<'a> {
    Gemv {
        weight,
        x,
        y,
        tile,
        compile_options: None,
    }
}

impl Gemv<'_> {
    pub fn compile_options(mut self, options: CompileOptions) -> Self {
        self.compile_options = Some(options);
        self
    }

    pub fn submit<S: Sink>(self, sink: &mut S) -> Result<(), cuda_async::error::DeviceError> {
        match self.weight {
            Weight::Dense(w) => {
                sink.run(dense::dense_gemv(
                    w,
                    self.x,
                    self.y,
                    self.tile,
                    self.compile_options,
                ))?;
            }
            Weight::Q8(w) => {
                sink.run(q8::q8_gemv(
                    w,
                    self.x,
                    self.y,
                    self.tile,
                    self.compile_options,
                ))?;
            }
            Weight::Fp8(w) => {
                fp8::dense_gemv(w, self.x, self.y, self.tile, self.compile_options, sink)?;
            }
            Weight::Nvfp4(w) => {
                nvfp4::dense_gemv(w, self.x, self.y, self.tile, self.compile_options, sink)?;
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
    compile_options: Option<CompileOptions>,
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
        compile_options: None,
    }
}

impl ExpertGemvView<'_> {
    pub fn compile_options(mut self, options: CompileOptions) -> Self {
        self.compile_options = Some(options);
        self
    }

    pub fn submit<S: Sink>(self, sink: &mut S) -> Result<(), cuda_async::error::DeviceError> {
        match self.weight {
            Weight::Dense(w) => sink.run(expert::expert_gemv_view_with_options(
                w,
                self.x,
                self.selection,
                self.y,
                self.rows_per_expert,
                self.input_stride,
                self.tile,
                self.compile_options,
            ))?,
            Weight::Fp8(w) => fp8::expert_gemv_view(
                w,
                self.x,
                self.selection,
                self.y,
                self.rows_per_expert,
                self.input_stride,
                self.tile,
                self.compile_options,
                sink,
            )?,
            Weight::Nvfp4(w) => nvfp4::expert_gemv_view(
                w,
                self.x,
                self.selection,
                self.y,
                self.rows_per_expert,
                self.input_stride,
                self.tile,
                self.compile_options,
                sink,
            )?,
            Weight::Q8(_) => {
                return Err(cuda_async::error::DeviceError::Internal(
                    "Q8 expert GEMV is unsupported".into(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_tiles_cover_scale_blocks_and_model_shapes() {
        for (rows, cols, groups, expected_rows) in [
            (8192, 2048, 1, 128),
            (4096, 2048, 1, 64),
            (2048, 4096, 1, 32),
            (512, 2048, 1, 8),
            (1024, 2048, 8, 128),
            (2048, 512, 8, 128),
        ] {
            let tile = fp8_tile(rows, cols, 128, 128, groups);
            assert_eq!(tile.rows, expected_rows);
            assert_eq!(tile.blocks, 4);
            assert!(tile.fits(rows, cols));
            assert_eq!(128 % tile.rows, 0);
            assert_eq!(128 % (tile.blocks * Q8_BLOCK), 0);
        }
    }
}
