#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn dense_gemv_kernel<T: ElementType, const B: i32, const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [B, BN] }>,
        w: &Tensor<T, { [-1, -1, 32] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, NB, 32]);
        let x_part = x.partition(shape![B, NB, 32]);
        let mut acc: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
        for kc in 0i32..(blocks / NB) {
            let weights: Tile<T, { [BN, NB, 32] }> = w_part.load([pid.1, kc, 0i32]);
            let weights: Tile<f32, { [BN, NB, 32] }> = convert_tile(weights);
            let weights: Tile<f32, { [B, BN, NB, 32] }> = weights
                .reshape(shape![1, BN, NB, 32])
                .broadcast(shape![B, BN, NB, 32]);
            let xb: Tile<f32, { [B, NB, 32] }> = x_part.load([0i32, kc, 0i32]);
            let xb: Tile<f32, { [B, BN, NB, 32] }> = xb
                .reshape(shape![B, 1, NB, 32])
                .broadcast(shape![B, BN, NB, 32]);
            let part: Tile<f32, { [B, BN, NB] }> = reduce_sum(weights * xb, 3i32);
            let part: Tile<f32, { [B, BN, NB] }> = part.reshape(shape![B, BN, NB]);
            let sum: Tile<f32, { [B, BN] }> = reduce_sum(part, 2i32);
            let sum: Tile<f32, { [B, BN] }> = sum.reshape(shape![B, BN]);
            acc = acc + sum;
        }
        y.store(acc);
    }
}

use super::GemvTile;
use crate::weights::{DenseMatrix, Q8_BLOCK};
use cuda_async::device_operation::{DeviceOp, GraphNode};
use cutile::DType;
use cutile::core::ElementType;
use cutile::tensor::{Partition, Tensor, TensorView};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

pub(super) fn dense_gemv<'a, T: DType + ElementType>(
    w: &DenseMatrix<T>,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
) -> impl DeviceOp<
    Output = (
        Partition<&'a mut Tensor<f32>>,
        Arc<Tensor<T>>,
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
    kernels::dense_gemv_kernel(y, w.weights.clone(), x, (w.cols / Q8_BLOCK) as i32).generics(vec![
        T::DTYPE.as_str().to_string(),
        batch.to_string(),
        tile.rows.to_string(),
        tile.blocks.to_string(),
    ])
}
