#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn gdn_step<const HEADS: i32, const D: i32, const R: i32>(
        out: &mut Tensor<f32, { [1, R] }>,
        state: &mut Tensor<f32, { [1, R, D] }>,
        input: &Tensor<f32, { [1, -1] }>,
        inv_d: f32,
        decay: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let row = pid.1;
        let x = input.partition(shape![1, D]).load([0, head]);
        let k: Tile<f32, { [1, R, D] }> = x.reshape(shape![1, 1, D]).broadcast(shape![1, R, D]);
        let v: Tile<f32, { [1, R] }> = input
            .partition(shape![1, R])
            .load([0, head * (D / R) + row]);
        let v: Tile<f32, { [1, R, D] }> = v.reshape(shape![1, R, 1]).broadcast(shape![1, R, D]);
        let old: Tile<f32, { [1, R, D] }> = state.load();
        let next = old * decay.broadcast(shape![1, R, D]) + v * k;
        state.store(next);
        let y: Tile<f32, { [1, R] }> = reduce_sum(next * k, 2);
        let y: Tile<f32, { [1, R] }> = y.reshape(shape![1, R]);
        out.store(y * inv_d.broadcast(shape![1, R]));
    }

    #[cutile::entry()]
    pub fn kv_append<const MAX_SEQ: i32, const HEADS: i32, const D: i32, const R: i32>(
        k_cache: &mut Tensor<f16, { [MAX_SEQ, 1, R] }>,
        v_cache: &mut Tensor<f16, { [MAX_SEQ, 1, R] }>,
        input: &Tensor<f32, { [1, -1] }>,
        position: &Tensor<i32, { [1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let pos: Tile<i32, { [1] }> = position.partition(shape![1]).load([0]);
        let index: i32 = tile_to_scalar(pos.reshape(shape![]));
        let value: Tile<f32, { [1, R] }> = input
            .partition(shape![1, R])
            .load([0, pid.1 * (D / R) + pid.2]);
        let value: Tile<f16, { [1, 1, R] }> = convert_tile(value.reshape(shape![1, 1, R]));
        let mut kp = k_cache.partition_mut(shape![1, 1, R]);
        let mut vp = v_cache.partition_mut(shape![1, 1, R]);
        kp.store(value, [index, 0, 0]);
        vp.store(value, [index, 0, 0]);
    }

    #[cutile::entry()]
    pub fn attn_step<const HEADS: i32, const D: i32, const R: i32>(
        out: &mut Tensor<f32, { [1, R] }>,
        k_cache: &Tensor<f16, { [-1, HEADS, D] }>,
        v_cache: &Tensor<f16, { [-1, HEADS, D] }>,
        input: &Tensor<f32, { [1, -1] }>,
        position: &Tensor<i32, { [1] }>,
        inv_d: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let pos: Tile<i32, { [1] }> = position.partition(shape![1]).load([0]);
        let limit: i32 = tile_to_scalar(pos.reshape(shape![]));
        let q: Tile<f32, { [1, D] }> = input.partition(shape![1, D]).load([0, pid.0]);
        let kp = k_cache.partition(shape![1, 1, D]);
        let vp = v_cache.partition(shape![1, 1, R]);
        let mut maximum: Tile<f32, { [1, 1] }> = constant(-1e30f32, shape![1, 1]);
        for t in 0i32..(limit + 1) {
            let key: Tile<f16, { [1, 1, D] }> = kp.load([t, pid.0, 0]);
            let key: Tile<f32, { [1, 1, D] }> = convert_tile(key);
            let key: Tile<f32, { [1, D] }> = key.reshape(shape![1, D]);
            let score: Tile<f32, { [1] }> = reduce_sum(q * key, 1);
            let score: Tile<f32, { [1, 1] }> =
                score.reshape(shape![1, 1]) * inv_d.broadcast(shape![1, 1]);
            maximum = max_tile(maximum, score);
        }
        let mut denom: Tile<f32, { [1, 1] }> = constant(0.0f32, shape![1, 1]);
        let mut acc: Tile<f32, { [1, R] }> = constant(0.0f32, shape![1, R]);
        for t in 0i32..(limit + 1) {
            let key: Tile<f16, { [1, 1, D] }> = kp.load([t, pid.0, 0]);
            let key: Tile<f32, { [1, 1, D] }> = convert_tile(key);
            let key: Tile<f32, { [1, D] }> = key.reshape(shape![1, D]);
            let score: Tile<f32, { [1] }> = reduce_sum(q * key, 1);
            let score: Tile<f32, { [1, 1] }> =
                score.reshape(shape![1, 1]) * inv_d.broadcast(shape![1, 1]);
            let weight = exp(score - maximum);
            denom = denom + weight;
            let value: Tile<f16, { [1, 1, R] }> = vp.load([t, pid.0, pid.1]);
            let value: Tile<f32, { [1, 1, R] }> = convert_tile(value);
            let value: Tile<f32, { [1, R] }> = value.reshape(shape![1, R]);
            acc = acc + value * weight.broadcast(shape![1, R]);
        }
        out.store(acc / denom.broadcast(shape![1, R]));
    }
}

pub use kernels::{attn_step, gdn_step, kv_append};
