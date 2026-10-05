use super::GemvTile;
use crate::moe::ExpertSelection;
use crate::sink::Sink;
use crate::weights::{Fp8Matrix, Q8_BLOCK};
use cuda_async::error::DeviceError;
use cuda_core::f8e4m3fn;
use cutile::tensor::{Partition, Tensor, TensorView, TensorViewMut};
use cutile::tile_kernel::{CompileOptions, TileKernel};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn dense_fp8_kernel<
        const B: i32,
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const BLOCK_N: i32,
        const BLOCK_K: i32,
    >(
        y: &mut Tensor<f32, { [B, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        scales: &Tensor<f32, { [-1, -1] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT]);
        let x_part = x.partition(shape![B, NB, 32]);
        let scale_part = scales.partition(shape![1, 1]);
        let mut acc: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
        let scale_row = pid.1 * BN / BLOCK_N;
        let steps_per_scale = BLOCK_K / KT;
        for scale_col in 0i32..(blocks * 32 / BLOCK_K) {
            let scale: Tile<f32, { [1, 1] }> = scale_part.load([scale_row, scale_col]);
            let mut sum: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
            for step in 0i32..steps_per_scale {
                let kc = scale_col * steps_per_scale + step;
                let weights: Tile<f8e4m3fn, { [BN, KT] }> = w_part.load([pid.1, kc]);
                let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
                let weights: Tile<f32, { [B, BN, KT] }> = weights
                    .reshape(shape![1, BN, KT])
                    .broadcast(shape![B, BN, KT]);
                let xb: Tile<f32, { [B, NB, 32] }> = x_part.load([0, kc, 0]);
                let xb: Tile<f32, { [B, BN, KT] }> =
                    xb.reshape(shape![B, 1, KT]).broadcast(shape![B, BN, KT]);
                let part: Tile<f32, { [B, BN] }> = reduce_sum(weights * xb, 2i32);
                let part: Tile<f32, { [B, BN] }> = part.reshape(shape![B, BN]);
                sum = sum + part;
            }
            let scale: Tile<f32, { [B, BN] }> = scale.broadcast(shape![B, BN]);
            acc = acc + sum * scale;
        }
        y.store(acc);
    }

    #[cutile::entry()]
    pub fn expert_fp8_kernel<
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const BLOCK_N: i32,
        const BLOCK_K: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        scales: &Tensor<f32, { [-1, -1] }>,
        x: &Tensor<f32, { [-1, -1, 32] }>,
        expert_ids: &Tensor<i32, { [-1, -1] }>,
        blocks: i32,
        row_tiles: i32,
        rows_per_expert: i32,
        input_stride: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let expert_tile: Tile<i32, { [1, 1] }> =
            expert_ids.partition(shape![1, 1]).load([0, pid.0]);
        let expert: i32 = tile_to_scalar(expert_tile.reshape(shape![]));
        let w_part = w.partition(shape![BN, KT]);
        let x_part = x.partition(shape![1, NB, 32]);
        let scale_part = scales.partition(shape![1, 1]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        let scale_row = (expert * rows_per_expert + pid.1 * BN) / BLOCK_N;
        let steps_per_scale = BLOCK_K / KT;
        for scale_col in 0i32..(blocks * 32 / BLOCK_K) {
            let scale: Tile<f32, { [1, 1] }> = scale_part.load([scale_row, scale_col]);
            let mut sum: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
            for step in 0i32..steps_per_scale {
                let kc = scale_col * steps_per_scale + step;
                let weights: Tile<f8e4m3fn, { [BN, KT] }> =
                    w_part.load([expert * row_tiles + pid.1, kc]);
                let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
                let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([pid.0 * input_stride, kc, 0]);
                let xb: Tile<f32, { [BN, KT] }> =
                    xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
                let products: Tile<f32, { [BN] }> = reduce_sum(weights * xb, 1i32);
                sum = sum + products.reshape(shape![1, BN]);
            }
            let scale: Tile<f32, { [1, BN] }> = scale.broadcast(shape![1, BN]);
            acc = acc + sum * scale;
        }
        y.store(acc);
    }
}

fn weight_matrix(w: &Fp8Matrix) -> TensorView<'_, f8e4m3fn> {
    w.weights
        .weights
        .view(&[w.weights.rows, w.weights.cols])
        .expect("FP8 matrix storage has the declared shape")
}

pub(super) fn dense_gemv<'a, S: Sink>(
    w: &Fp8Matrix,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
    options: Option<CompileOptions>,
    sink: &mut S,
) -> Result<(), DeviceError> {
    let batch = x.shape()[0] as usize;
    assert!(tile.fits(w.weights.rows, w.weights.cols));
    assert_eq!(
        x.shape(),
        &[
            batch as i32,
            (w.weights.cols / Q8_BLOCK) as i32,
            Q8_BLOCK as i32
        ]
    );
    assert_eq!(w.block_n % tile.rows, 0);
    assert_eq!(w.block_k % (tile.blocks * Q8_BLOCK), 0);
    let weights = weight_matrix(w);
    sink.run(
        kernels::dense_fp8_kernel(
            y,
            &weights,
            w.scales.clone(),
            x,
            (w.weights.cols / Q8_BLOCK) as i32,
        )
        .generics(vec![
            batch.to_string(),
            tile.rows.to_string(),
            tile.blocks.to_string(),
            (tile.blocks * Q8_BLOCK).to_string(),
            w.block_n.to_string(),
            w.block_k.to_string(),
        ])
        .compile_options(options.unwrap_or_default()),
    )
}

pub(super) fn expert_gemv_view<'a, S: Sink>(
    w: &Fp8Matrix,
    x: &'a TensorView<'a, f32>,
    selection: &'a ExpertSelection,
    y: Partition<TensorViewMut<'a, f32>>,
    rows_per_expert: usize,
    input_stride: usize,
    tile: GemvTile,
    options: Option<CompileOptions>,
    sink: &mut S,
) -> Result<(), DeviceError> {
    assert_eq!(w.weights.rows, selection.dims().experts() * rows_per_expert);
    assert!(tile.fits(rows_per_expert, w.weights.cols));
    assert_eq!(rows_per_expert % w.block_n, 0);
    assert_eq!(w.block_n % tile.rows, 0);
    assert_eq!(w.block_k % (tile.blocks * Q8_BLOCK), 0);
    let weights = weight_matrix(w);
    sink.run(
        kernels::expert_fp8_kernel(
            y,
            &weights,
            w.scales.clone(),
            x,
            selection.indices(),
            (w.weights.cols / Q8_BLOCK) as i32,
            (rows_per_expert / tile.rows) as i32,
            rows_per_expert as i32,
            input_stride as i32,
        )
        .generics(vec![
            tile.rows.to_string(),
            tile.blocks.to_string(),
            (tile.blocks * Q8_BLOCK).to_string(),
            w.block_n.to_string(),
            w.block_k.to_string(),
        ])
        .compile_options(options.unwrap_or_default()),
    )
}
