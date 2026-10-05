use super::GemvTile;
use crate::moe::ExpertSelection;
use crate::weights::{Nvfp4Matrix, Q8_BLOCK};
use cuda_async::device_operation::{DeviceOp, GraphNode};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::tensor::{Partition, Tensor, TensorView, TensorViewMut};
use cutile::tile_kernel::{CompileOptions, TileKernel};
use std::sync::Arc;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn dense_nvfp4_kernel<const B: i32, const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [B, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1, 16] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1, 2] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, NB, 16]);
        let scale_part = scales.partition(shape![BN, NB, 2]);
        let x_part = x.partition(shape![B, NB, 32]);
        let mut acc: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
        for kc in 0i32..(blocks / NB) {
            let packed: Tile<f4e2m1fnx2, { [BN, NB, 16] }> = w_part.load([pid.1, kc, 0]);
            let weights: Tile<f4e2m1fn, { [BN, NB, 32] }> = packed.unpack(shape![BN, NB, 32]);
            let weights: Tile<f32, { [BN, NB, 32] }> = convert_tile(weights);
            let weights: Tile<f32, { [B, BN, NB, 32] }> = weights
                .reshape(shape![1, BN, NB, 32])
                .broadcast(shape![B, BN, NB, 32]);
            let xb: Tile<f32, { [B, NB, 32] }> = x_part.load([0, kc, 0]);
            let xb: Tile<f32, { [B, BN, NB, 32] }> = xb
                .reshape(shape![B, 1, NB, 32])
                .broadcast(shape![B, BN, NB, 32]);
            let products: Tile<f32, { [B, BN, NB, 2, 16] }> =
                (weights * xb).reshape(shape![B, BN, NB, 2, 16]);
            let groups: Tile<f32, { [B, BN, NB, 2] }> = reduce_sum(products, 4i32);
            let groups: Tile<f32, { [B, BN, NB, 2] }> = groups.reshape(shape![B, BN, NB, 2]);
            let scale: Tile<f8e4m3fn, { [BN, NB, 2] }> = scale_part.load([pid.1, kc, 0]);
            let scale: Tile<f32, { [BN, NB, 2] }> = convert_tile(scale);
            let scale: Tile<f32, { [B, BN, NB, 2] }> = scale
                .reshape(shape![1, BN, NB, 2])
                .broadcast(shape![B, BN, NB, 2]);
            let blocks_sum: Tile<f32, { [B, BN, NB] }> = reduce_sum(groups * scale, 3i32);
            let sum: Tile<f32, { [B, BN] }> = reduce_sum(blocks_sum, 2i32);
            acc = acc + sum.reshape(shape![B, BN]);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, pid.1]);
        y.store(acc * row_scale.broadcast(shape![B, BN]));
    }

    #[cutile::entry()]
    pub fn expert_nvfp4_kernel<const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1, 16] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1, 2] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        x: &Tensor<f32, { [-1, -1, 32] }>,
        expert_ids: &Tensor<i32, { [-1, -1] }>,
        blocks: i32,
        row_tiles: i32,
        input_stride: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let expert_tile: Tile<i32, { [1, 1] }> =
            expert_ids.partition(shape![1, 1]).load([0, pid.0]);
        let expert: i32 = tile_to_scalar(expert_tile.reshape(shape![]));
        let row_tile = expert * row_tiles + pid.1;
        let w_part = w.partition(shape![BN, NB, 16]);
        let scale_part = scales.partition(shape![BN, NB, 2]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..(blocks / NB) {
            let packed: Tile<f4e2m1fnx2, { [BN, NB, 16] }> = w_part.load([row_tile, kc, 0]);
            let weights: Tile<f4e2m1fn, { [BN, NB, 32] }> = packed.unpack(shape![BN, NB, 32]);
            let weights: Tile<f32, { [BN, NB, 32] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([pid.0 * input_stride, kc, 0]);
            let xb: Tile<f32, { [BN, NB, 32] }> = xb.broadcast(shape![BN, NB, 32]);
            let products: Tile<f32, { [BN, NB, 2, 16] }> =
                (weights * xb).reshape(shape![BN, NB, 2, 16]);
            let groups: Tile<f32, { [BN, NB, 2] }> = reduce_sum(products, 3i32);
            let groups: Tile<f32, { [BN, NB, 2] }> = groups.reshape(shape![BN, NB, 2]);
            let scale: Tile<f8e4m3fn, { [BN, NB, 2] }> = scale_part.load([row_tile, kc, 0]);
            let scale: Tile<f32, { [BN, NB, 2] }> = convert_tile(scale);
            let blocks_sum: Tile<f32, { [BN, NB] }> = reduce_sum(groups * scale, 2i32);
            let sum: Tile<f32, { [BN] }> = reduce_sum(blocks_sum, 1i32);
            acc = acc + sum.reshape(shape![1, BN]);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, row_tile]);
        y.store(acc * row_scale);
    }
}

pub(super) fn dense_gemv<'a>(
    w: &'a Nvfp4Matrix,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
    options: Option<CompileOptions>,
) -> impl DeviceOp<
    Output = (
        Partition<&'a mut Tensor<f32>>,
        Arc<Tensor<f4e2m1fnx2>>,
        Arc<Tensor<f8e4m3fn>>,
        Arc<Tensor<f32>>,
        &'a TensorView<'a, f32>,
        i32,
    ),
> + GraphNode
+ 'a {
    let batch = x.shape()[0] as usize;
    assert!(batch > 0 && tile.fits(w.rows, w.cols));
    assert_eq!(
        x.shape(),
        &[batch as i32, (w.cols / Q8_BLOCK) as i32, Q8_BLOCK as i32]
    );
    kernels::dense_nvfp4_kernel(
        y,
        w.packed.clone(),
        w.scales.clone(),
        w.row_scales.clone(),
        x,
        (w.cols / Q8_BLOCK) as i32,
    )
    .generics(vec![
        batch.to_string(),
        tile.rows.to_string(),
        tile.blocks.to_string(),
    ])
    .compile_options(options.unwrap_or_default())
}

pub(super) fn expert_gemv_view<'a>(
    w: &'a Nvfp4Matrix,
    x: &'a TensorView<'a, f32>,
    selection: &'a ExpertSelection,
    y: Partition<TensorViewMut<'a, f32>>,
    rows_per_expert: usize,
    input_stride: usize,
    tile: GemvTile,
    options: Option<CompileOptions>,
) -> impl DeviceOp<
    Output = (
        Partition<TensorViewMut<'a, f32>>,
        Arc<Tensor<f4e2m1fnx2>>,
        Arc<Tensor<f8e4m3fn>>,
        Arc<Tensor<f32>>,
        &'a TensorView<'a, f32>,
        &'a Tensor<i32>,
        i32,
        i32,
        i32,
    ),
> + GraphNode
+ 'a {
    assert_eq!(w.rows, selection.dims().experts() * rows_per_expert);
    assert!(tile.fits(rows_per_expert, w.cols));
    kernels::expert_nvfp4_kernel(
        y,
        w.packed.clone(),
        w.scales.clone(),
        w.row_scales.clone(),
        x,
        selection.indices(),
        (w.cols / Q8_BLOCK) as i32,
        (rows_per_expert / tile.rows) as i32,
        input_stride as i32,
    )
    .generics(vec![tile.rows.to_string(), tile.blocks.to_string()])
    .compile_options(options.unwrap_or_default())
}
