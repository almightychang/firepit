use crate::moe::PrefillPlan;
use crate::sink::Sink;
use crate::weights::DenseMatrix;
use cutile::tensor::{Partition, Tensor, TensorView};
use half::bf16;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn gemm_kernel<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<bf16, { [-1, K] }>,
    ) {
        let x_tiles = x.partition(shape![BM, BK]);
        let w_tiles = w.partition(shape![BN, BK]);
        let block: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for part in 0i32..(K / BK) {
            let values: Tile<f32, { [BM, BK] }> = x_tiles.load([block.0, part]);
            let hi: Tile<bf16, { [BM, BK] }> = convert_tile(values);
            let hi_f32: Tile<f32, { [BM, BK] }> = convert_tile(hi);
            let lo: Tile<bf16, { [BM, BK] }> = convert_tile(values - hi_f32);
            let weights: Tile<bf16, { [BK, BN] }> = w_tiles.load([block.1, part]).transpose();
            acc = mma(hi, weights, acc);
            acc = mma(lo, weights, acc);
        }
        y.store(acc);
    }

    #[cutile::entry()]
    pub fn grouped_gemm_kernel<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<bf16, { [-1, K] }>,
        token_rows: &Tensor<i32, { [-1] }>,
        tile_experts: &Tensor<i32, { [-1] }>,
        rows_per_expert: i32,
        gather_tokens: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let expert_tile: Tile<i32, { [1] }> = tile_experts.partition(shape![1]).load([pid.0]);
        let expert: i32 = tile_to_scalar(expert_tile.reshape(shape![]));
        let rows: Tile<i32, { [BM] }> = iota(shape![BM]);
        let x_tiles = x.partition(shape![1, BK]);
        let w_tiles = w.partition(shape![BN, BK]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for part in 0i32..(K / BK) {
            let mut values: Tile<f32, { [BM, BK] }> = constant(0.0f32, shape![BM, BK]);
            for row in 0i32..BM {
                let sorted = pid.0 * BM + row;
                let token = if gather_tokens != 0 {
                    let token_tile: Tile<i32, { [1] }> =
                        token_rows.partition(shape![1]).load([sorted]);
                    let token: i32 = tile_to_scalar(token_tile.reshape(shape![]));
                    max(token, 0i32)
                } else {
                    sorted
                };
                let value: Tile<f32, { [1, BK] }> = x_tiles.load([token, part]);
                let mask = eq_tile(rows, row.broadcast(shape![BM]))
                    .reshape(shape![BM, 1])
                    .broadcast(shape![BM, BK]);
                values = select(mask, value.broadcast(shape![BM, BK]), values);
            }
            let hi: Tile<bf16, { [BM, BK] }> = convert_tile(values);
            let hi_f32: Tile<f32, { [BM, BK] }> = convert_tile(hi);
            let lo: Tile<bf16, { [BM, BK] }> = convert_tile(values - hi_f32);
            let weights: Tile<bf16, { [BK, BN] }> = w_tiles
                .load([expert * (rows_per_expert / BN) + pid.1, part])
                .transpose();
            acc = mma(hi, weights, acc);
            acc = mma(lo, weights, acc);
        }
        y.store(acc);
    }
}

#[derive(Clone, Copy)]
pub struct GemmTile {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

pub const DEFAULT_GEMM_TILE: GemmTile = GemmTile {
    m: 64,
    n: 128,
    k: 32,
};

pub const DEFAULT_GROUPED_GEMM_TILE: GemmTile = GemmTile {
    m: 16,
    n: 128,
    k: 32,
};

#[derive(Clone, Copy)]
pub enum GroupInput {
    Tokens,
    Padded,
}

pub struct GroupedGemm<'a> {
    weight: &'a DenseMatrix<bf16>,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    plan: &'a PrefillPlan,
    rows_per_expert: usize,
    input: GroupInput,
    tile: GemmTile,
}

pub fn grouped_gemm<'a>(
    weight: &'a DenseMatrix<bf16>,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    plan: &'a PrefillPlan,
    rows_per_expert: usize,
    input: GroupInput,
    tile: GemmTile,
) -> GroupedGemm<'a> {
    let dims = plan.dims();
    assert_eq!(tile.m, plan.layout().tile_rows());
    assert!(tile.m > 0 && tile.n > 0 && tile.k > 0);
    assert_eq!(weight.rows, dims.experts() * rows_per_expert);
    assert_eq!(x.shape()[1] as usize, weight.cols);
    assert_eq!(
        x.shape()[0] as usize,
        match input {
            GroupInput::Tokens => plan.layout().tokens(),
            GroupInput::Padded => plan.layout().padded_rows(),
        }
    );
    assert_eq!(y.partition_shape, vec![tile.m, tile.n]);
    assert_eq!(
        y.grid().expect("valid output grid"),
        (
            plan.layout().tile_experts().len() as u32,
            (rows_per_expert / tile.n) as u32,
            1
        )
    );
    assert_eq!(rows_per_expert % tile.n, 0);
    assert_eq!(weight.cols % tile.k, 0);
    GroupedGemm {
        weight,
        x,
        y,
        plan,
        rows_per_expert,
        input,
        tile,
    }
}

impl GroupedGemm<'_> {
    pub fn submit<S: Sink>(self, sink: &mut S) -> Result<(), cuda_async::error::DeviceError> {
        use cutile::tile_kernel::TileKernel;
        let w = self
            .weight
            .weights
            .view(&[self.weight.rows, self.weight.cols])
            .expect("DenseMatrix storage has the declared shape");
        sink.run(
            kernels::grouped_gemm_kernel(
                self.y,
                self.x,
                &w,
                self.plan.token_rows(),
                self.plan.tile_experts(),
                self.rows_per_expert as i32,
                match self.input {
                    GroupInput::Tokens => 1,
                    GroupInput::Padded => 0,
                },
            )
            .generics(vec![
                self.tile.m.to_string(),
                self.tile.n.to_string(),
                self.tile.k.to_string(),
                self.weight.cols.to_string(),
            ]),
        )?;
        Ok(())
    }
}

pub struct Gemm<'a> {
    weight: &'a DenseMatrix<bf16>,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemmTile,
}

pub fn gemm<'a>(
    weight: &'a DenseMatrix<bf16>,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemmTile,
) -> Gemm<'a> {
    assert!(tile.m > 0 && tile.n > 0 && tile.k > 0);
    assert_eq!(x.shape().len(), 2);
    let m = x.shape()[0] as usize;
    assert!(m > 0 && m % tile.m == 0);
    assert_eq!(x.shape()[1] as usize, weight.cols);
    assert!(weight.rows % tile.n == 0 && weight.cols % tile.k == 0);
    assert_eq!(y.partition_shape, vec![tile.m, tile.n]);
    assert_eq!(
        y.grid().expect("valid output grid"),
        ((m / tile.m) as u32, (weight.rows / tile.n) as u32, 1)
    );
    Gemm { weight, x, y, tile }
}

impl Gemm<'_> {
    pub fn submit<S: Sink>(self, sink: &mut S) -> Result<(), cuda_async::error::DeviceError> {
        use cutile::tile_kernel::TileKernel;

        let w = self
            .weight
            .weights
            .view(&[self.weight.rows, self.weight.cols])
            .expect("DenseMatrix storage has the declared shape");
        sink.run(kernels::gemm_kernel(self.y, self.x, &w).generics(vec![
            self.tile.m.to_string(),
            self.tile.n.to_string(),
            self.tile.k.to_string(),
            self.weight.cols.to_string(),
        ]))?;
        Ok(())
    }
}
