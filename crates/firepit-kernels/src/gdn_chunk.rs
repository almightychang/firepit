#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn first_phase_matrices<
        const C: i32,
        const DK: i32,
        const KEY_HEADS: i32,
        const VALUE_HEADS: i32,
    >(
        ut: &mut Tensor<f32, { [1, C, C] }>,
        intra: &mut Tensor<f32, { [1, C, C] }>,
        cum_out: &mut Tensor<f32, { [1, C] }>,
        query: &Tensor<f32, { [-1, C, KEY_HEADS, DK] }>,
        key: &Tensor<f32, { [-1, C, KEY_HEADS, DK] }>,
        gate: &Tensor<f32, { [-1, C, VALUE_HEADS] }>,
        beta: &Tensor<f32, { [-1, C, VALUE_HEADS] }>,
        chunk_size: i32,
        eps: f32,
        inv_sqrt_dk: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0 % VALUE_HEADS;
        let chunk = pid.0 / VALUE_HEADS;
        let key_head = head / (VALUE_HEADS / KEY_HEADS);
        let q: Tile<f32, { [C, DK] }> = query
            .partition(shape![1, C, 1, DK])
            .load([chunk, 0, key_head, 0])
            .reshape(shape![C, DK]);
        let k: Tile<f32, { [C, DK] }> = key
            .partition(shape![1, C, 1, DK])
            .load([chunk, 0, key_head, 0])
            .reshape(shape![C, DK]);
        let g: Tile<f32, { [C] }> = gate
            .partition(shape![1, C, 1])
            .load([chunk, 0, head])
            .reshape(shape![C]);
        let beta: Tile<f32, { [C] }> = beta
            .partition(shape![1, C, 1])
            .load([chunk, 0, head])
            .reshape(shape![C]);
        let rows: Tile<i32, { [C] }> = iota(shape![C]);
        let valid = lt_tile(rows, chunk_size.broadcast(shape![C]));
        let zero_c: Tile<f32, { [C] }> = constant(0.0f32, shape![C]);
        let g = select(valid, g, zero_c);
        let beta = select(valid, beta, zero_c);
        let cum = scan_sum(g, 0i32, reverse::Forward, 0.0f32);
        cum_out.store(cum.reshape(shape![1, C]));
        let q_norm: Tile<f32, { [C] }> = reduce_sum(q * q, 1);
        let k_norm: Tile<f32, { [C] }> = reduce_sum(k * k, 1);
        let q_scale = rsqrt(q_norm + eps.broadcast(shape![C]), ftz::Disabled)
            * inv_sqrt_dk.broadcast(shape![C]);
        let k_scale = rsqrt(k_norm + eps.broadcast(shape![C]), ftz::Disabled);
        let q = q * q_scale.reshape(shape![C, 1]).broadcast(shape![C, DK]);
        let k = k * k_scale.reshape(shape![C, 1]).broadcast(shape![C, DK]);
        let k_beta = k * beta.reshape(shape![C, 1]).broadcast(shape![C, DK]);
        let row_cum = cum.reshape(shape![C, 1]).broadcast(shape![C, C]);
        let col_cum = cum.reshape(shape![1, C]).broadcast(shape![C, C]);
        let row_index = rows.reshape(shape![C, 1]).broadcast(shape![C, C]);
        let col_index = rows.reshape(shape![1, C]).broadcast(shape![C, C]);
        let causal = ge_tile(row_index, col_index);
        let zero_cc: Tile<f32, { [C, C] }> = constant(0.0f32, shape![C, C]);
        let decay: Tile<f32, { [C, C] }> = select(causal, exp(row_cum - col_cum), zero_cc);
        let k_hi: Tile<bf16, { [C, DK] }> = convert_tile(k_beta);
        let k_lo: Tile<bf16, { [C, DK] }> = convert_tile(k_beta - convert_tile(k_hi));
        let kt: Tile<f32, { [DK, C] }> = k.transpose();
        let kt_hi: Tile<bf16, { [DK, C] }> = convert_tile(kt);
        let kt_lo: Tile<bf16, { [DK, C] }> = convert_tile(kt - convert_tile(kt_hi));
        let mut ut_tile = mma(k_hi, kt_hi, zero_cc);
        ut_tile = mma(k_hi, kt_lo, ut_tile);
        ut_tile = mma(k_lo, kt_hi, ut_tile);
        let ut_result: Tile<f32, { [C, C] }> = ut_tile * decay;
        ut.store(ut_result.reshape(shape![1, C, C]));
        let q_hi: Tile<bf16, { [C, DK] }> = convert_tile(q);
        let q_lo: Tile<bf16, { [C, DK] }> = convert_tile(q - convert_tile(q_hi));
        let mut intra_tile = mma(q_hi, kt_hi, zero_cc);
        intra_tile = mma(q_hi, kt_lo, intra_tile);
        intra_tile = mma(q_lo, kt_hi, intra_tile);
        let intra_result: Tile<f32, { [C, C] }> = intra_tile * decay;
        intra.store(intra_result.reshape(shape![1, C, C]));
    }

    #[cutile::entry()]
    pub fn invert_diagonal<const C: i32, const B: i32>(
        out: &mut Tensor<f32, { [1, B, B] }>,
        ut: &Tensor<f32, { [-1, C, C] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let blocks = C / B;
        let flat = pid.0 / blocks;
        let block = pid.0 % blocks;
        let a: Tile<f32, { [B, B] }> = ut
            .partition(shape![1, B, B])
            .load([flat, block, block])
            .reshape(shape![B, B]);
        let indices: Tile<i32, { [B] }> = iota(shape![B]);
        let row_index = indices.reshape(shape![B, 1]).broadcast(shape![B, B]);
        let col_index = indices.reshape(shape![1, B]).broadcast(shape![B, B]);
        let zero: Tile<f32, { [B, B] }> = constant(0.0f32, shape![B, B]);
        let zero_row: Tile<f32, { [1, B] }> = constant(0.0f32, shape![1, B]);
        let one: Tile<f32, { [B, B] }> = constant(1.0f32, shape![B, B]);
        let mut inverse = zero - select(gt_tile(row_index, col_index), a, zero);
        for row in 2i32..B {
            let coefficients: Tile<f32, { [1, B] }> =
                extract(a, [scalar_to_tile(row), scalar_to_tile(0i32)]);
            let earlier = lt_tile(indices.reshape(shape![1, B]), row.broadcast(shape![1, B]));
            let coefficients = zero_row - select(earlier, coefficients, zero_row);
            let products = coefficients.transpose().broadcast(shape![B, B]) * inverse;
            let sum: Tile<f32, { [B] }> = reduce_sum(products, 0);
            let original: Tile<f32, { [B] }> = reduce_sum(coefficients, 0);
            let updated = original + sum;
            let target = eq_tile(row_index, row.broadcast(shape![B, B]));
            inverse = select(
                target,
                updated.reshape(shape![1, B]).broadcast(shape![B, B]),
                inverse,
            );
        }
        let inverse = inverse + select(eq_tile(row_index, col_index), one, zero);
        out.store(inverse.reshape(shape![1, B, B]));
    }

    #[cutile::entry()]
    pub fn merge_inverse<const C: i32, const B: i32>(
        inverse: &mut Tensor<f32, { [1, C, C] }>,
        previous: &Tensor<f32, { [-1, C, C] }>,
        ut: &Tensor<f32, { [-1, C, C] }>,
        diagonals: &Tensor<f32, { [-1, B, B] }>,
        i: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let blocks = C / B;
        let ut_blocks = ut.partition(shape![1, B, B]);
        let diagonal_blocks = diagonals.partition(shape![1, B, B]);
        let zero: Tile<f32, { [B, B] }> = constant(0.0f32, shape![B, B]);
        let old: Tile<f32, { [1, C, C] }> = previous.partition(shape![1, C, C]).load([pid.0, 0, 0]);
        inverse.store(old);
        {
            let diagonal: Tile<f32, { [B, B] }> = diagonal_blocks
                .load([pid.0 * blocks + i, 0, 0])
                .reshape(shape![B, B]);
            inverse
                .partition_mut(shape![1, B, B])
                .store(diagonal.reshape(shape![1, B, B]), [0, i, i]);
            for j in 0i32..i {
                let mut sum = zero;
                for p in j..i {
                    let a: Tile<f32, { [B, B] }> =
                        ut_blocks.load([pid.0, i, p]).reshape(shape![B, B]);
                    let x: Tile<f32, { [B, B] }> = previous
                        .partition(shape![1, B, B])
                        .load([pid.0, p, j])
                        .reshape(shape![B, B]);
                    let a_hi: Tile<bf16, { [B, B] }> = convert_tile(a);
                    let x_hi: Tile<bf16, { [B, B] }> = convert_tile(x);
                    let a_lo: Tile<bf16, { [B, B] }> = convert_tile(a - convert_tile(a_hi));
                    let x_lo: Tile<bf16, { [B, B] }> = convert_tile(x - convert_tile(x_hi));
                    sum = mma(a_hi, x_hi, sum);
                    sum = mma(a_hi, x_lo, sum);
                    sum = mma(a_lo, x_hi, sum);
                }
                let d_hi: Tile<bf16, { [B, B] }> = convert_tile(diagonal);
                let d_lo: Tile<bf16, { [B, B] }> = convert_tile(diagonal - convert_tile(d_hi));
                let s_hi: Tile<bf16, { [B, B] }> = convert_tile(sum);
                let s_lo: Tile<bf16, { [B, B] }> = convert_tile(sum - convert_tile(s_hi));
                let mut result = mma(d_hi, s_hi, zero);
                result = mma(d_hi, s_lo, result);
                result = mma(d_lo, s_hi, result);
                inverse
                    .partition_mut(shape![1, B, B])
                    .store((zero - result).reshape(shape![1, B, B]), [0, i, j]);
            }
        }
    }

    #[cutile::entry()]
    pub fn apply_inverse<
        const C: i32,
        const DK: i32,
        const DV: i32,
        const KEY_HEADS: i32,
        const VALUE_HEADS: i32,
    >(
        u: &mut Tensor<f32, { [1, C, DV] }>,
        w: &mut Tensor<f32, { [1, C, DK] }>,
        inverse: &Tensor<f32, { [-1, C, C] }>,
        cum: &Tensor<f32, { [-1, C] }>,
        key: &Tensor<f32, { [-1, C, KEY_HEADS, DK] }>,
        value: &Tensor<f32, { [-1, C, VALUE_HEADS, DV] }>,
        beta: &Tensor<f32, { [-1, C, VALUE_HEADS] }>,
        chunk_size: i32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0 % VALUE_HEADS;
        let chunk = pid.0 / VALUE_HEADS;
        let key_head = head / (VALUE_HEADS / KEY_HEADS);
        let k: Tile<f32, { [C, DK] }> = key
            .partition(shape![1, C, 1, DK])
            .load([chunk, 0, key_head, 0])
            .reshape(shape![C, DK]);
        let v: Tile<f32, { [C, DV] }> = value
            .partition(shape![1, C, 1, DV])
            .load([chunk, 0, head, 0])
            .reshape(shape![C, DV]);
        let beta: Tile<f32, { [C] }> = beta
            .partition(shape![1, C, 1])
            .load([chunk, 0, head])
            .reshape(shape![C]);
        let gate_cum: Tile<f32, { [C] }> = cum
            .partition(shape![1, C])
            .load([pid.0, 0])
            .reshape(shape![C]);
        let rows: Tile<i32, { [C] }> = iota(shape![C]);
        let zero_c: Tile<f32, { [C] }> = constant(0.0f32, shape![C]);
        let beta = select(lt_tile(rows, chunk_size.broadcast(shape![C])), beta, zero_c);
        let norm: Tile<f32, { [C] }> = reduce_sum(k * k, 1);
        let k = k * rsqrt(norm + eps.broadcast(shape![C]), ftz::Disabled)
            .reshape(shape![C, 1])
            .broadcast(shape![C, DK]);
        let k_beta = k * beta.reshape(shape![C, 1]).broadcast(shape![C, DK]);
        let v_beta = v * beta.reshape(shape![C, 1]).broadcast(shape![C, DV]);
        let decayed_k_beta = k_beta * exp(gate_cum).reshape(shape![C, 1]).broadcast(shape![C, DK]);
        let x: Tile<f32, { [C, C] }> = inverse
            .partition(shape![1, C, C])
            .load([pid.0, 0, 0])
            .reshape(shape![C, C]);
        let x_hi: Tile<bf16, { [C, C] }> = convert_tile(x);
        let x_lo: Tile<bf16, { [C, C] }> = convert_tile(x - convert_tile(x_hi));
        let v_hi: Tile<bf16, { [C, DV] }> = convert_tile(v_beta);
        let v_lo: Tile<bf16, { [C, DV] }> = convert_tile(v_beta - convert_tile(v_hi));
        let zero_v: Tile<f32, { [C, DV] }> = constant(0.0f32, shape![C, DV]);
        let mut u_result = mma(x_hi, v_hi, zero_v);
        u_result = mma(x_hi, v_lo, u_result);
        u_result = mma(x_lo, v_hi, u_result);
        u.store(u_result.reshape(shape![1, C, DV]));
        let k_hi: Tile<bf16, { [C, DK] }> = convert_tile(decayed_k_beta);
        let k_lo: Tile<bf16, { [C, DK] }> = convert_tile(decayed_k_beta - convert_tile(k_hi));
        let zero_k: Tile<f32, { [C, DK] }> = constant(0.0f32, shape![C, DK]);
        let mut w_result = mma(x_hi, k_hi, zero_k);
        w_result = mma(x_hi, k_lo, w_result);
        w_result = mma(x_lo, k_hi, w_result);
        w.store(w_result.reshape(shape![1, C, DK]));
    }
}

pub use kernels::{apply_inverse, first_phase_matrices, invert_diagonal, merge_inverse};
