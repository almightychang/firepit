// cuTile shape syntax requires literals here; keep them tied to the host format.
const _: [(); crate::weights::Q8_BLOCK] = [(); 32];

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn q8_gemv_kernel<const B: i32, const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [B, BN] }>,
        qs: &Tensor<i8, { [-1, -1, 32] }>,
        scales: &Tensor<f16, { [-1, -1] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_part = qs.partition(shape![BN, NB, 32]);
        let s_part = scales.partition(shape![BN, NB]);
        let x_part = x.partition(shape![B, NB, 32]);
        let mut acc: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
        for kc in 0i32..(blocks / NB) {
            let q: Tile<i8, { [BN, NB, 32] }> = q_part.load([pid.1, kc, 0i32]);
            let q: Tile<f32, { [BN, NB, 32] }> = convert_tile(q);
            let q: Tile<f32, { [B, BN, NB, 32] }> = q
                .reshape(shape![1, BN, NB, 32])
                .broadcast(shape![B, BN, NB, 32]);
            let s: Tile<f16, { [BN, NB] }> = s_part.load([pid.1, kc]);
            let s: Tile<f32, { [BN, NB] }> = convert_tile(s);
            let xb: Tile<f32, { [B, NB, 32] }> = x_part.load([0i32, kc, 0i32]);
            let xb: Tile<f32, { [B, BN, NB, 32] }> = xb
                .reshape(shape![B, 1, NB, 32])
                .broadcast(shape![B, BN, NB, 32]);
            let part: Tile<f32, { [B, BN, NB] }> = reduce_sum(q * xb, 3i32);
            let part: Tile<f32, { [B, BN, NB] }> = part.reshape(shape![B, BN, NB]);
            let s: Tile<f32, { [B, BN, NB] }> =
                s.reshape(shape![1, BN, NB]).broadcast(shape![B, BN, NB]);
            let scaled: Tile<f32, { [B, BN] }> = reduce_sum(part * s, 2i32);
            let scaled: Tile<f32, { [B, BN] }> = scaled.reshape(shape![B, BN]);
            acc = acc + scaled;
        }
        y.store(acc);
    }
}

use super::GemvTile;
use crate::weights::{Q8_BLOCK, Q8Matrix};
use cuda_async::device_operation::{DeviceOp, GraphNode};
use cutile::tensor::{Partition, Tensor, TensorView};
use cutile::tile_kernel::{CompileOptions, TileKernel};
use half::f16;
use std::sync::Arc;

pub(super) fn q8_gemv<'a>(
    w: &Q8Matrix,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
    options: Option<CompileOptions>,
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
    kernels::q8_gemv_kernel(
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
    .compile_options(options.unwrap_or_default())
}
