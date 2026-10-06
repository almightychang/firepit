#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn cache_append<const MAX_SEQ: i32, const KV_HEADS: i32, const D: i32>(
        k_cache: &mut Tensor<f32, { [MAX_SEQ, 1, D] }>,
        v_cache: &mut Tensor<f32, { [MAX_SEQ, 1, D] }>,
        key: &Tensor<f32, { [KV_HEADS, D] }>,
        value: &Tensor<f32, { [KV_HEADS, D] }>,
        position: &Tensor<i32, { [1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let pos: Tile<i32, { [1] }> = position.partition(shape![1]).load([0]);
        let index: i32 = tile_to_scalar(pos.reshape(shape![]));
        let k: Tile<f32, { [1, 1, D] }> = key
            .partition(shape![1, D])
            .load([pid.1, 0])
            .reshape(shape![1, 1, D]);
        let v: Tile<f32, { [1, 1, D] }> = value
            .partition(shape![1, D])
            .load([pid.1, 0])
            .reshape(shape![1, 1, D]);
        k_cache
            .partition_mut(shape![1, 1, D])
            .store(k, [index, 0, 0]);
        v_cache
            .partition_mut(shape![1, 1, D])
            .store(v, [index, 0, 0]);
    }

    #[cutile::entry()]
    pub fn cache_append_prefill<const KV_HEADS: i32, const D: i32>(
        k_cache: &mut Tensor<f32, { [1, 1, D] }>,
        v_cache: &mut Tensor<f32, { [1, 1, D] }>,
        key: &Tensor<f32, { [-1, KV_HEADS, D] }>,
        value: &Tensor<f32, { [-1, KV_HEADS, D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let k = key.partition(shape![1, 1, D]).load([pid.0, pid.1, 0]);
        let v = value.partition(shape![1, 1, D]).load([pid.0, pid.1, 0]);
        k_cache.partition_mut(shape![1, 1, D]).store(k, [0, 0, 0]);
        v_cache.partition_mut(shape![1, 1, D]).store(v, [0, 0, 0]);
    }

    #[cutile::entry()]
    pub fn causal_attention_prefill<
        const BM: i32,
        const BN: i32,
        const D: i32,
        const Q_HEADS: i32,
        const KV_HEADS: i32,
    >(
        out: &mut Tensor<f32, { [BM, 1, D] }>,
        query: &Tensor<f32, { [-1, Q_HEADS, D] }>,
        k_cache: &Tensor<f32, { [-1, KV_HEADS, D] }>,
        v_cache: &Tensor<f32, { [-1, KV_HEADS, D] }>,
        input_pos: i32,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let kv_head = pid.1 / (Q_HEADS / KV_HEADS);
        let q: Tile<f32, { [BM, D] }> = query
            .partition(shape![BM, 1, D])
            .load([pid.0, pid.1, 0])
            .reshape(shape![BM, D]);
        let q_hi: Tile<bf16, { [BM, D] }> = convert_tile(q);
        let q_hi_f32: Tile<f32, { [BM, D] }> = convert_tile(q_hi);
        let q_lo: Tile<bf16, { [BM, D] }> = convert_tile(q - q_hi_f32);
        let k_part = k_cache.partition(shape![BN, 1, D]);
        let v_part = v_cache.partition(shape![BN, 1, D]);
        let row_iota: Tile<i32, { [BM] }> = iota(shape![BM]);
        let col_iota: Tile<i32, { [BN] }> = iota(shape![BN]);
        let row: Tile<i32, { [BM, 1] }> = broadcast_scalar(input_pos + pid.0 * BM, shape![BM, 1])
            + row_iota.reshape(shape![BM, 1]);
        let col: Tile<i32, { [1, BN] }> = col_iota.reshape(shape![1, BN]);
        let mut maximum: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, shape![BM, 1]);
        let mut denominator: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        let mut numerator: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);
        let end = min(input_pos + (pid.0 + 1) * BM, input_pos + query.shape()[0]);
        for block in 0i32..ceil_div(end, BN) {
            let k: Tile<f32, { [BN, D] }> = k_part.load([block, kv_head, 0]).reshape(shape![BN, D]);
            let k_hi: Tile<bf16, { [BN, D] }> = convert_tile(k);
            let k_hi_f32: Tile<f32, { [BN, D] }> = convert_tile(k_hi);
            let k_hi = k_hi.transpose();
            let k_lo: Tile<bf16, { [BN, D] }> = convert_tile(k - k_hi_f32);
            let k_lo = k_lo.transpose();
            let mut score: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            score = mma(q_hi, k_hi, score);
            score = mma(q_hi, k_lo, score);
            score = mma(q_lo, k_hi, score);
            let column =
                (broadcast_scalar(block * BN, shape![1, BN]) + col).broadcast(shape![BM, BN]);
            let valid = ge_tile(row.broadcast(shape![BM, BN]), column);
            let scaled: Tile<f32, { [BM, BN] }> = score * scale.broadcast(shape![BM, BN]);
            let masked: Tile<f32, { [BM, BN] }> = constant(f32::NEG_INFINITY, shape![BM, BN]);
            score = select(valid, scaled, masked);
            let block_max: Tile<f32, { [BM] }> = reduce_max(score, 1);
            let block_max = block_max.reshape(shape![BM, 1]);
            let next_max = max_tile(maximum, block_max);
            let alpha = exp(maximum - next_max);
            let p = exp(score - next_max.broadcast(shape![BM, BN]));
            let block_sum: Tile<f32, { [BM] }> = reduce_sum(p, 1);
            denominator = denominator * alpha + block_sum.reshape(shape![BM, 1]);
            numerator = numerator * alpha.broadcast(shape![BM, D]);
            let v: Tile<f32, { [BN, D] }> = v_part.load([block, kv_head, 0]).reshape(shape![BN, D]);
            let v_hi: Tile<bf16, { [BN, D] }> = convert_tile(v);
            let v_hi_f32: Tile<f32, { [BN, D] }> = convert_tile(v_hi);
            let v_lo: Tile<bf16, { [BN, D] }> = convert_tile(v - v_hi_f32);
            let p_hi: Tile<bf16, { [BM, BN] }> = convert_tile(p);
            let p_hi_f32: Tile<f32, { [BM, BN] }> = convert_tile(p_hi);
            let p_lo: Tile<bf16, { [BM, BN] }> = convert_tile(p - p_hi_f32);
            numerator = mma(p_hi, v_hi, numerator);
            numerator = mma(p_hi, v_lo, numerator);
            numerator = mma(p_lo, v_hi, numerator);
            maximum = next_max;
        }
        out.store((numerator / denominator.broadcast(shape![BM, D])).reshape(shape![BM, 1, D]));
    }

    #[cutile::entry()]
    pub fn online_attention<const Q_HEADS: i32, const KV_HEADS: i32, const D: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        query: &Tensor<f32, { [Q_HEADS, D] }>,
        k_cache: &Tensor<f32, { [-1, KV_HEADS, D] }>,
        v_cache: &Tensor<f32, { [-1, KV_HEADS, D] }>,
        position: &Tensor<i32, { [1] }>,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let pos: Tile<i32, { [1] }> = position.partition(shape![1]).load([0]);
        let limit: i32 = tile_to_scalar(pos.reshape(shape![]));
        let q: Tile<f32, { [1, D] }> = query.partition(shape![1, D]).load([pid.0, 0]);
        let kv_head = pid.0 / (Q_HEADS / KV_HEADS);
        let kp = k_cache.partition(shape![1, 1, D]);
        let vp = v_cache.partition(shape![1, 1, D]);
        let mut maximum: Tile<f32, { [1, 1] }> = constant(f32::NEG_INFINITY, shape![1, 1]);
        let mut denominator: Tile<f32, { [1, 1] }> = constant(0.0f32, shape![1, 1]);
        let mut numerator: Tile<f32, { [1, D] }> = constant(0.0f32, shape![1, D]);
        for t in 0i32..(limit + 1) {
            let key: Tile<f32, { [1, 1, D] }> = kp.load([t, kv_head, 0]);
            let key: Tile<f32, { [1, D] }> = key.reshape(shape![1, D]);
            let score: Tile<f32, { [1] }> = reduce_sum(q * key, 1);
            let score: Tile<f32, { [1, 1] }> =
                score.reshape(shape![1, 1]) * scale.broadcast(shape![1, 1]);
            let next_max = max_tile(maximum, score);
            let old_weight = exp(maximum - next_max);
            let new_weight = exp(score - next_max);
            let value: Tile<f32, { [1, 1, D] }> = vp.load([t, kv_head, 0]);
            let value: Tile<f32, { [1, D] }> = value.reshape(shape![1, D]);
            numerator = numerator * old_weight.broadcast(shape![1, D])
                + value * new_weight.broadcast(shape![1, D]);
            denominator = denominator * old_weight + new_weight;
            maximum = next_max;
        }
        out.store(numerator / denominator.broadcast(shape![1, D]));
    }

    #[cutile::entry()]
    pub fn output_gate<const D: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        input: &Tensor<f32, { [-1, -1, D] }>,
        gate: &Tensor<f32, { [-1, -1, D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let x: Tile<f32, { [1, D] }> = input
            .partition(shape![1, 1, D])
            .load([pid.0, pid.1, 0])
            .reshape(shape![1, D]);
        let g: Tile<f32, { [1, D] }> = gate
            .partition(shape![1, 1, D])
            .load([pid.0, pid.1, 0])
            .reshape(shape![1, D]);
        let one: Tile<f32, { [1, D] }> = constant(1.0f32, shape![1, D]);
        out.store(x / (one + exp(g * constant(-1.0f32, shape![1, D]))));
    }
}

pub use kernels::{
    cache_append, cache_append_prefill, causal_attention_prefill, online_attention, output_gate,
};

#[derive(Debug, Clone, Copy)]
pub struct PrefillTile {
    pub m: usize,
    pub n: usize,
}

pub const DEFAULT_PREFILL_TILE: PrefillTile = PrefillTile { m: 16, n: 32 };

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionDimensionError {
    InvalidDimensions,
}

impl std::fmt::Display for AttentionDimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AttentionDimensionError {}

#[derive(Debug, Clone, Copy)]
pub struct AttentionDims {
    max_seq: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl AttentionDims {
    pub fn new(
        max_seq: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, AttentionDimensionError> {
        if max_seq == 0
            || max_seq > i32::MAX as usize
            || q_heads == 0
            || q_heads > i32::MAX as usize
            || kv_heads == 0
            || kv_heads > q_heads
            || q_heads % kv_heads != 0
            || head_dim == 0
            || head_dim > i32::MAX as usize
            || !head_dim.is_power_of_two()
            || q_heads
                .checked_mul(head_dim)
                .is_none_or(|x| x > i32::MAX as usize)
            || max_seq
                .checked_mul(kv_heads)
                .and_then(|x| x.checked_mul(head_dim))
                .is_none_or(|x| x > i32::MAX as usize)
        {
            return Err(AttentionDimensionError::InvalidDimensions);
        }
        Ok(Self {
            max_seq,
            q_heads,
            kv_heads,
            head_dim,
        })
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
    pub fn q_heads(&self) -> usize {
        self.q_heads
    }
    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
    pub fn cache_generics(&self) -> Vec<String> {
        vec![
            self.max_seq.to_string(),
            self.kv_heads.to_string(),
            self.head_dim.to_string(),
        ]
    }
    pub fn attention_generics(&self) -> Vec<String> {
        vec![
            self.q_heads.to_string(),
            self.kv_heads.to_string(),
            self.head_dim.to_string(),
        ]
    }
    pub fn gate_generics(&self) -> Vec<String> {
        vec![self.head_dim.to_string()]
    }

    pub fn prefill_generics(&self, tile: PrefillTile) -> Vec<String> {
        vec![
            tile.m.to_string(),
            tile.n.to_string(),
            self.head_dim.to_string(),
            self.q_heads.to_string(),
            self.kv_heads.to_string(),
        ]
    }
    pub fn prefill_cache_generics(&self) -> Vec<String> {
        vec![self.kv_heads.to_string(), self.head_dim.to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::AttentionDims;

    #[test]
    fn rejects_invalid_dimensions() {
        for (max_seq, q_heads, kv_heads, head_dim) in [
            (0, 4, 2, 32),
            (8, 0, 2, 32),
            (8, 4, 0, 32),
            (8, 4, 3, 32),
            (8, 2, 4, 32),
            (8, 4, 2, 0),
            (8, 4, 2, 48),
            (i32::MAX as usize + 1, 4, 2, 32),
            (8, i32::MAX as usize + 1, 2, 32),
        ] {
            assert!(AttentionDims::new(max_seq, q_heads, kv_heads, head_dim).is_err());
        }
        assert!(AttentionDims::new(262144, 16, 2, 256).is_ok());
    }
}
