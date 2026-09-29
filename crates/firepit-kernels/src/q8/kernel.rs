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

pub(super) use kernels::q8_gemv_kernel;
