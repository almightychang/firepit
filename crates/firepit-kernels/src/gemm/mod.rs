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
