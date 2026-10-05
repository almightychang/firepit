use super::GemmTile;
use crate::moe::PrefillPlan;
use crate::sink::Sink;
use crate::weights::{NVFP4_GROUP, Nvfp4Matrix, Q8_BLOCK};
use cutile::tensor::{Partition, Tensor, TensorView, TensorViewMut};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn nvfp4_gemm_kernel<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const K: i32,
        const BK_PACKED: i32,
        const BK_GROUPS: i32,
    >(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
    ) {
        let block: (i32, i32, i32) = get_tile_block_id();
        let x_tiles = x.partition(shape![BM, BK]);
        let w_tiles = w.partition(shape![BN, BK_PACKED]);
        let scale_tiles = scales.partition(shape![BN, BK_GROUPS]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for part in 0i32..(K / BK) {
            let values: Tile<f32, { [BM, BK] }> = x_tiles.load([block.0, part]);
            let hi: Tile<bf16, { [BM, BK] }> = convert_tile(values);
            let hi_f32: Tile<f32, { [BM, BK] }> = convert_tile(hi);
            let lo: Tile<bf16, { [BM, BK] }> = convert_tile(values - hi_f32);
            let packed: Tile<f4e2m1fnx2, { [BN, BK_PACKED] }> = w_tiles.load([block.1, part]);
            let weights: Tile<f4e2m1fn, { [BN, BK] }> = packed.unpack(shape![BN, BK]);
            let weights: Tile<f32, { [BN, BK] }> = convert_tile(weights);
            let scale: Tile<f8e4m3fn, { [BN, BK_GROUPS] }> = scale_tiles.load([block.1, part]);
            let scale: Tile<f32, { [BN, BK_GROUPS] }> = convert_tile(scale);
            let scale: Tile<f32, { [BN, BK_GROUPS, 16] }> = scale
                .reshape(shape![BN, BK_GROUPS, 1])
                .broadcast(shape![BN, BK_GROUPS, 16]);
            let weights: Tile<f32, { [BN, BK] }> =
                (weights.reshape(shape![BN, BK_GROUPS, 16]) * scale).reshape(shape![BN, BK]);
            let weights: Tile<bf16, { [BN, BK] }> = convert_tile(weights);
            let weights: Tile<bf16, { [BK, BN] }> = weights.transpose();
            acc = mma(hi, weights, acc);
            acc = mma(lo, weights, acc);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, block.1]);
        y.store(acc * row_scale.broadcast(shape![BM, BN]));
    }

    #[cutile::entry()]
    pub fn grouped_nvfp4_gemm_kernel<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const K: i32,
        const BK_PACKED: i32,
        const BK_GROUPS: i32,
    >(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        tile_experts: &Tensor<i32, { [-1] }>,
        row_tiles: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let expert_tile: Tile<i32, { [1] }> = tile_experts.partition(shape![1]).load([pid.0]);
        let expert: i32 = tile_to_scalar(expert_tile.reshape(shape![]));
        let row_tile = expert * row_tiles + pid.1;
        let x_tiles = x.partition(shape![BM, BK]);
        let w_tiles = w.partition(shape![BN, BK_PACKED]);
        let scale_tiles = scales.partition(shape![BN, BK_GROUPS]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for part in 0i32..(K / BK) {
            let values: Tile<f32, { [BM, BK] }> = x_tiles.load([pid.0, part]);
            let hi: Tile<bf16, { [BM, BK] }> = convert_tile(values);
            let hi_f32: Tile<f32, { [BM, BK] }> = convert_tile(hi);
            let lo: Tile<bf16, { [BM, BK] }> = convert_tile(values - hi_f32);
            let packed: Tile<f4e2m1fnx2, { [BN, BK_PACKED] }> = w_tiles.load([row_tile, part]);
            let weights: Tile<f4e2m1fn, { [BN, BK] }> = packed.unpack(shape![BN, BK]);
            let weights: Tile<f32, { [BN, BK] }> = convert_tile(weights);
            let scale: Tile<f8e4m3fn, { [BN, BK_GROUPS] }> = scale_tiles.load([row_tile, part]);
            let scale: Tile<f32, { [BN, BK_GROUPS] }> = convert_tile(scale);
            let scale: Tile<f32, { [BN, BK_GROUPS, 16] }> = scale
                .reshape(shape![BN, BK_GROUPS, 1])
                .broadcast(shape![BN, BK_GROUPS, 16]);
            let weights: Tile<f32, { [BN, BK] }> =
                (weights.reshape(shape![BN, BK_GROUPS, 16]) * scale).reshape(shape![BN, BK]);
            let weights: Tile<bf16, { [BN, BK] }> = convert_tile(weights);
            let weights: Tile<bf16, { [BK, BN] }> = weights.transpose();
            acc = mma(hi, weights, acc);
            acc = mma(lo, weights, acc);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, row_tile]);
        y.store(acc * row_scale.broadcast(shape![BM, BN]));
    }
}

fn tile_generics(tile: GemmTile, cols: usize) -> Vec<String> {
    vec![
        tile.m.to_string(),
        tile.n.to_string(),
        tile.k.to_string(),
        cols.to_string(),
        (tile.k / 2).to_string(),
        (tile.k / NVFP4_GROUP).to_string(),
    ]
}

fn assert_tile_fits(weight: &Nvfp4Matrix, tile: GemmTile) {
    assert!(tile.m > 0 && tile.n > 0 && tile.k % Q8_BLOCK == 0);
    assert!(weight.rows % tile.n == 0 && weight.cols % tile.k == 0);
}

pub struct Nvfp4Gemm<'a> {
    weight: &'a Nvfp4Matrix,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemmTile,
}

pub fn nvfp4_gemm<'a>(
    weight: &'a Nvfp4Matrix,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemmTile,
) -> Nvfp4Gemm<'a> {
    let m = x.shape()[0] as usize;
    assert!(m > 0 && m % tile.m == 0);
    assert_eq!(x.shape(), &[m as i32, weight.cols as i32]);
    assert_tile_fits(weight, tile);
    assert_eq!(y.partition_shape, vec![tile.m, tile.n]);
    Nvfp4Gemm { weight, x, y, tile }
}

impl Nvfp4Gemm<'_> {
    pub fn submit<S: Sink>(self, sink: &mut S) -> Result<(), cuda_async::error::DeviceError> {
        use cutile::tile_kernel::TileKernel;
        let w = self.weight.packed_matrix();
        let scales = self.weight.scale_matrix();
        sink.run(
            kernels::nvfp4_gemm_kernel(self.y, self.x, &w, &scales, &self.weight.row_scales)
                .generics(tile_generics(self.tile, self.weight.cols)),
        )?;
        Ok(())
    }
}

pub struct Nvfp4GroupedGemmView<'a> {
    weight: &'a Nvfp4Matrix,
    x: &'a TensorView<'a, f32>,
    y: Partition<TensorViewMut<'a, f32>>,
    plan: &'a PrefillPlan,
    rows_per_expert: usize,
    tile: GemmTile,
}

pub fn nvfp4_grouped_gemm_view<'a>(
    weight: &'a Nvfp4Matrix,
    x: &'a TensorView<'a, f32>,
    y: Partition<TensorViewMut<'a, f32>>,
    plan: &'a PrefillPlan,
    rows_per_expert: usize,
    tile: GemmTile,
) -> Nvfp4GroupedGemmView<'a> {
    assert_eq!(tile.m, plan.layout().tile_rows());
    assert_eq!(weight.rows, plan.dims().experts() * rows_per_expert);
    assert_eq!(
        x.shape(),
        &[plan.layout().padded_rows() as i32, weight.cols as i32]
    );
    assert_tile_fits(weight, tile);
    assert_eq!(rows_per_expert % tile.n, 0);
    assert_eq!(y.partition_shape, vec![tile.m, tile.n]);
    Nvfp4GroupedGemmView {
        weight,
        x,
        y,
        plan,
        rows_per_expert,
        tile,
    }
}

impl Nvfp4GroupedGemmView<'_> {
    pub fn submit<S: Sink>(self, sink: &mut S) -> Result<(), cuda_async::error::DeviceError> {
        use cutile::tile_kernel::TileKernel;
        let w = self.weight.packed_matrix();
        let scales = self.weight.scale_matrix();
        sink.run(
            kernels::grouped_nvfp4_gemm_kernel(
                self.y,
                self.x,
                &w,
                &scales,
                &self.weight.row_scales,
                self.plan.tile_experts(),
                (self.rows_per_expert / self.tile.n) as i32,
            )
            .generics(tile_generics(self.tile, self.weight.cols)),
        )?;
        Ok(())
    }
}
