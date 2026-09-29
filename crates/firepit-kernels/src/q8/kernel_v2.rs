#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn quantize_x_kernel(
        xq: &mut Tensor<i8, { [1, 1, 32] }>,
        dx: &mut Tensor<f32, { [1, 1] }>,
        x: &Tensor<f32, { [-1, -1, 32] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let xb: Tile<f32, { [1, 1, 32] }> =
            x.partition(shape![1, 1, 32]).load([pid.0, pid.1, 0i32]);
        let max_abs: Tile<f32, { [1, 1] }> = reduce_max(absf(xb), 2i32);
        let max_abs: Tile<f32, { [1, 1] }> = max_abs.reshape(shape![1, 1]);
        let denom: Tile<f32, { [1, 1] }> = constant(127.0f32, shape![1, 1]);
        let scale: Tile<f32, { [1, 1] }> = max_abs / denom;
        let zero: Tile<f32, { [1, 1] }> = constant(0.0f32, shape![1, 1]);
        let one: Tile<f32, { [1, 1] }> = constant(1.0f32, shape![1, 1]);
        let divisor: Tile<f32, { [1, 1] }> = select(eq_tile(scale, zero), one, scale);
        let divisor: Tile<f32, { [1, 1, 32] }> =
            divisor.reshape(shape![1, 1, 1]).broadcast(shape![1, 1, 32]);
        let values: Tile<f32, { [1, 1, 32] }> = xb / divisor;
        let zero: Tile<f32, { [1, 1, 32] }> = constant(0.0f32, shape![1, 1, 32]);
        let positive: Tile<f32, { [1, 1, 32] }> = constant(0.5f32, shape![1, 1, 32]);
        let negative: Tile<f32, { [1, 1, 32] }> = constant(-0.5f32, shape![1, 1, 32]);
        let offset: Tile<f32, { [1, 1, 32] }> = select(
            cmpf(values, zero, predicate::LessThan, cmp_ordering::Ordered),
            negative,
            positive,
        );
        let q: Tile<i32, { [1, 1, 32] }> = ftoi(values + offset, rounding::NearestIntToZero);
        let q: Tile<i8, { [1, 1, 32] }> = trunci(q, overflow::NoSignedWrap);
        xq.store(q);
        dx.store(scale);
    }

    #[cutile::entry()]
    pub fn q8_gemv_v1b_kernel<const B: i32, const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [B, BN] }>,
        qs: &Tensor<i8, { [-1, -1, 32] }>,
        scales: &Tensor<f16, { [-1, -1] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_part = qs.partition(shape![BN, NB, 32]);
        let s_part = scales.partition(shape![BN, NB]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut y_rows: PartitionMut<f32, { [1, BN] }> = y.partition_mut(shape![1, BN]);
        for b in 0i32..B {
            let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
            for kc in 0i32..(blocks / NB) {
                let q: Tile<i8, { [BN, NB, 32] }> = q_part.load([pid.1, kc, 0i32]);
                let q: Tile<f32, { [BN, NB, 32] }> = convert_tile(q);
                let s: Tile<f16, { [BN, NB] }> = s_part.load([pid.1, kc]);
                let s: Tile<f32, { [BN, NB] }> = convert_tile(s);
                let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([b, kc, 0i32]);
                let xb: Tile<f32, { [BN, NB, 32] }> = xb.broadcast(shape![BN, NB, 32]);
                let dot: Tile<f32, { [BN, NB] }> = reduce_sum(q * xb, 2i32);
                let dot: Tile<f32, { [BN, NB] }> = dot.reshape(shape![BN, NB]);
                let scaled: Tile<f32, { [BN] }> = reduce_sum(dot * s, 1i32);
                let scaled: Tile<f32, { [BN] }> = scaled.reshape(shape![BN]);
                acc = acc + scaled;
            }
            y_rows.store(acc.reshape(shape![1, BN]), [b, 0i32]);
        }
    }

    #[cutile::entry()]
    pub fn q8_gemv_v2_kernel<const B: i32, const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [B, BN] }>,
        qs: &Tensor<i8, { [-1, -1, 32] }>,
        scales: &Tensor<f16, { [-1, -1] }>,
        xq: &Tensor<i8, { [B, -1, 32] }>,
        dx: &Tensor<f32, { [B, -1] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_part = qs.partition(shape![BN, NB, 32]);
        let s_part = scales.partition(shape![BN, NB]);
        let x_part = xq.partition(shape![1, NB, 32]);
        let dx_part = dx.partition(shape![1, NB]);
        let mut y_rows: PartitionMut<f32, { [1, BN] }> = y.partition_mut(shape![1, BN]);
        for b in 0i32..B {
            let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
            for kc in 0i32..(blocks / NB) {
                let q: Tile<i8, { [BN, NB, 32] }> = q_part.load([pid.1, kc, 0i32]);
                let q: Tile<i32, { [BN, NB, 32] }> = exti(q);
                let s: Tile<f16, { [BN, NB] }> = s_part.load([pid.1, kc]);
                let s: Tile<f32, { [BN, NB] }> = convert_tile(s);
                let xb: Tile<i8, { [1, NB, 32] }> = x_part.load([b, kc, 0i32]);
                let xb: Tile<i32, { [1, NB, 32] }> = exti(xb);
                let xb: Tile<i32, { [BN, NB, 32] }> = xb.broadcast(shape![BN, NB, 32]);
                let dot: Tile<i32, { [BN, NB] }> = reduce_sum(q * xb, 2i32);
                let dot: Tile<i32, { [BN, NB] }> = dot.reshape(shape![BN, NB]);
                let dot: Tile<f32, { [BN, NB] }> = convert_tile(dot);
                let dxb: Tile<f32, { [1, NB] }> = dx_part.load([b, kc]);
                let dxb: Tile<f32, { [BN, NB] }> = dxb.broadcast(shape![BN, NB]);
                let scaled: Tile<f32, { [BN] }> = reduce_sum(dot * s * dxb, 1i32);
                let scaled: Tile<f32, { [BN] }> = scaled.reshape(shape![BN]);
                acc = acc + scaled;
            }
            y_rows.store(acc.reshape(shape![1, BN]), [b, 0i32]);
        }
    }
}

pub use kernels::{q8_gemv_v1b_kernel, q8_gemv_v2_kernel, quantize_x_kernel};
