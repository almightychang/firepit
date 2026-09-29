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

pub(super) use kernels::dense_gemv_kernel;
