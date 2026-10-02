use crate::gemv::GemvTile;
use crate::moe::ExpertSelection;
use crate::weights::{DenseMatrix, Q8_BLOCK};
use cuda_async::device_operation::{DeviceOp, GraphNode};
use cutile::DType;
use cutile::core::ElementType;
use cutile::tensor::{Partition, Tensor, TensorView, TensorViewMut};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn expert_gemv_kernel<T: ElementType, const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<T, { [-1, -1, 32] }>,
        x: &Tensor<f32, { [-1, -1, 32] }>,
        expert_ids: &Tensor<i32, { [-1] }>,
        blocks: i32,
        row_tiles: i32,
        input_stride: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let expert_tile: Tile<i32, { [1] }> = expert_ids.partition(shape![1]).load([pid.0]);
        let expert: i32 = tile_to_scalar(expert_tile.reshape(shape![]));
        let w_part = w.partition(shape![BN, NB, 32]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..(blocks / NB) {
            let weights: Tile<T, { [BN, NB, 32] }> =
                w_part.load([expert * row_tiles + pid.1, kc, 0]);
            let weights: Tile<f32, { [BN, NB, 32] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([pid.0 * input_stride, kc, 0]);
            let xb: Tile<f32, { [BN, NB, 32] }> = xb.broadcast(shape![BN, NB, 32]);
            let products: Tile<f32, { [BN, NB] }> = reduce_sum(weights * xb, 2);
            let products: Tile<f32, { [BN] }> = reduce_sum(products, 1);
            acc = acc + products.reshape(shape![1, BN]);
        }
        y.store(acc);
    }
}

pub fn expert_gemv<'a, T: DType + ElementType>(
    w: &DenseMatrix<T>,
    x: &'a TensorView<'a, f32>,
    selection: &'a ExpertSelection,
    y: Partition<&'a mut Tensor<f32>>,
    rows_per_expert: usize,
    input_stride: usize,
    tile: GemvTile,
) -> impl DeviceOp<
    Output = (
        Partition<&'a mut Tensor<f32>>,
        Arc<Tensor<T>>,
        &'a TensorView<'a, f32>,
        &'a Tensor<i32>,
        i32,
        i32,
        i32,
    ),
> + GraphNode
+ 'a {
    let dims = selection.dims();
    assert!(rows_per_expert > 0 && input_stride <= 1);
    assert_eq!(
        w.rows,
        dims.experts()
            .checked_mul(rows_per_expert)
            .expect("expert matrix rows overflow")
    );
    assert!(tile.rows > 0 && tile.blocks > 0 && tile.fits(rows_per_expert, w.cols));
    assert_eq!(
        x.shape(),
        &[
            (if input_stride == 0 { 1 } else { dims.top_k() }) as i32,
            (w.cols / Q8_BLOCK) as i32,
            Q8_BLOCK as i32
        ]
    );
    kernels::expert_gemv_kernel(
        y,
        w.weights.clone(),
        x,
        selection.indices(),
        (w.cols / Q8_BLOCK) as i32,
        (rows_per_expert / tile.rows) as i32,
        input_stride as i32,
    )
    .generics(vec![
        T::DTYPE.as_str().to_string(),
        tile.rows.to_string(),
        tile.blocks.to_string(),
    ])
}

pub fn expert_gemv_view<'a, T: DType + ElementType>(
    w: &DenseMatrix<T>,
    x: &'a TensorView<'a, f32>,
    selection: &'a ExpertSelection,
    y: Partition<TensorViewMut<'a, f32>>,
    rows_per_expert: usize,
    input_stride: usize,
    tile: GemvTile,
) -> impl DeviceOp<
    Output = (
        Partition<TensorViewMut<'a, f32>>,
        Arc<Tensor<T>>,
        &'a TensorView<'a, f32>,
        &'a Tensor<i32>,
        i32,
        i32,
        i32,
    ),
> + GraphNode
+ 'a {
    let dims = selection.dims();
    assert!(rows_per_expert > 0 && input_stride <= 1);
    assert_eq!(w.rows, dims.experts() * rows_per_expert);
    assert!(tile.fits(rows_per_expert, w.cols));
    kernels::expert_gemv_kernel(
        y,
        w.weights.clone(),
        x,
        selection.indices(),
        (w.cols / Q8_BLOCK) as i32,
        (rows_per_expert / tile.rows) as i32,
        input_stride as i32,
    )
    .generics(vec![
        T::DTYPE.as_str().to_string(),
        tile.rows.to_string(),
        tile.blocks.to_string(),
    ])
}
