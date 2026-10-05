use super::GemvTile;
use crate::moe::ExpertSelection;
use crate::sink::Sink;
use crate::weights::{NVFP4_GROUP, Nvfp4Matrix, Q8_BLOCK};
use cuda_async::error::DeviceError;
use cutile::tensor::{Partition, Tensor, TensorView, TensorViewMut};
use cutile::tile_kernel::{CompileOptions, TileKernel};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn dense_nvfp4_kernel<
        const B: i32,
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const KT_PACKED: i32,
        const KT_GROUPS: i32,
    >(
        y: &mut Tensor<f32, { [B, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let x_part = x.partition(shape![B, NB, 32]);
        let mut acc: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
        for kc in 0i32..(blocks / NB) {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = w_part.load([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let weights: Tile<f32, { [B, BN, KT] }> = weights
                .reshape(shape![1, BN, KT])
                .broadcast(shape![B, BN, KT]);
            let xb: Tile<f32, { [B, NB, 32] }> = x_part.load([0, kc, 0]);
            let xb: Tile<f32, { [B, BN, KT] }> =
                xb.reshape(shape![B, 1, KT]).broadcast(shape![B, BN, KT]);
            let products: Tile<f32, { [B, BN, KT_GROUPS, 16] }> =
                (weights * xb).reshape(shape![B, BN, KT_GROUPS, 16]);
            let groups: Tile<f32, { [B, BN, KT_GROUPS] }> = reduce_sum(products, 3i32);
            let groups: Tile<f32, { [B, BN, KT_GROUPS] }> =
                groups.reshape(shape![B, BN, KT_GROUPS]);
            let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> = scale_part.load([pid.1, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let scale: Tile<f32, { [B, BN, KT_GROUPS] }> = scale
                .reshape(shape![1, BN, KT_GROUPS])
                .broadcast(shape![B, BN, KT_GROUPS]);
            let sum: Tile<f32, { [B, BN] }> = reduce_sum(groups * scale, 2i32);
            acc = acc + sum.reshape(shape![B, BN]);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, pid.1]);
        y.store(acc * row_scale.broadcast(shape![B, BN]));
    }

    #[cutile::entry()]
    pub fn expert_nvfp4_kernel<
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const KT_PACKED: i32,
        const KT_GROUPS: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
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
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..(blocks / NB) {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = w_part.load([row_tile, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([pid.0 * input_stride, kc, 0]);
            let xb: Tile<f32, { [BN, KT] }> = xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
            let products: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                (weights * xb).reshape(shape![BN, KT_GROUPS, 16]);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = reduce_sum(products, 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> = scale_part.load([row_tile, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let sum: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
            acc = acc + sum.reshape(shape![1, BN]);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, row_tile]);
        y.store(acc * row_scale);
    }
}

fn tile_generics(tile: GemvTile) -> [String; 4] {
    let kt = tile.blocks * Q8_BLOCK;
    [
        tile.blocks.to_string(),
        kt.to_string(),
        (kt / 2).to_string(),
        (kt / NVFP4_GROUP).to_string(),
    ]
}

pub(super) fn dense_gemv<'a, S: Sink>(
    w: &Nvfp4Matrix,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
    options: Option<CompileOptions>,
    sink: &mut S,
) -> Result<(), DeviceError> {
    let batch = x.shape()[0] as usize;
    assert!(batch > 0 && tile.fits(w.rows, w.cols));
    assert_eq!(
        x.shape(),
        &[batch as i32, (w.cols / Q8_BLOCK) as i32, Q8_BLOCK as i32]
    );
    let (packed, scales) = (w.packed_matrix(), w.scale_matrix());
    let mut generics = vec![batch.to_string(), tile.rows.to_string()];
    generics.extend(tile_generics(tile));
    sink.run(
        kernels::dense_nvfp4_kernel(
            y,
            &packed,
            &scales,
            w.row_scales.clone(),
            x,
            (w.cols / Q8_BLOCK) as i32,
        )
        .generics(generics)
        .compile_options(options.unwrap_or_default()),
    )
}

pub(super) fn expert_gemv_view<'a, S: Sink>(
    w: &Nvfp4Matrix,
    x: &'a TensorView<'a, f32>,
    selection: &'a ExpertSelection,
    y: Partition<TensorViewMut<'a, f32>>,
    rows_per_expert: usize,
    input_stride: usize,
    tile: GemvTile,
    options: Option<CompileOptions>,
    sink: &mut S,
) -> Result<(), DeviceError> {
    assert_eq!(w.rows, selection.dims().experts() * rows_per_expert);
    assert!(tile.fits(rows_per_expert, w.cols));
    let (packed, scales) = (w.packed_matrix(), w.scale_matrix());
    let mut generics = vec![tile.rows.to_string()];
    generics.extend(tile_generics(tile));
    sink.run(
        kernels::expert_nvfp4_kernel(
            y,
            &packed,
            &scales,
            w.row_scales.clone(),
            x,
            selection.indices(),
            (w.cols / Q8_BLOCK) as i32,
            (rows_per_expert / tile.rows) as i32,
            input_stride as i32,
        )
        .generics(generics)
        .compile_options(options.unwrap_or_default()),
    )
}
