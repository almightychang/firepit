#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn embedding<const H: i32>(
        out: &mut Tensor<f32, { [1, H] }>,
        table: &Tensor<f32, { [-1, H] }>,
        tokens: &Tensor<i32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let token: Tile<i32, { [1] }> = tokens.partition(shape![1]).load([pid.0]);
        let index: i32 = tile_to_scalar(token.reshape(shape![]));
        let row: Tile<f32, { [1, H] }> = table.partition(shape![1, H]).load([index, 0]);
        out.store(row);
    }
}

pub use kernels::embedding;
