#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn conv1d_silu<const CHANNEL_TILE: i32, const KERNEL_TILE: i32, const KERNEL: i32>(
        out: &mut Tensor<f32, { [CHANNEL_TILE] }>,
        input: &Tensor<f32, { [-1] }>,
        state: &mut Tensor<f32, { [CHANNEL_TILE, KERNEL_TILE] }>,
        weight: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let previous: Tile<f32, { [CHANNEL_TILE, KERNEL_TILE] }> = state.load();
        let coefficients: Tile<f32, { [CHANNEL_TILE, KERNEL_TILE] }> = weight
            .partition(shape![CHANNEL_TILE, KERNEL_TILE])
            .load([pid.0, 0]);
        let current: Tile<f32, { [CHANNEL_TILE] }> =
            input.partition(shape![CHANNEL_TILE]).load([pid.0]);
        let positions: Tile<i32, { [KERNEL_TILE] }> = iota(shape![KERNEL_TILE]);
        let columns: Tile<i32, { [1, KERNEL_TILE] }> = positions.reshape(shape![1, KERNEL_TILE]);
        let current_column: Tile<bool, { [CHANNEL_TILE, KERNEL_TILE] }> = eq_tile(
            columns.broadcast(shape![CHANNEL_TILE, KERNEL_TILE]),
            (KERNEL - 1i32).broadcast(shape![CHANNEL_TILE, KERNEL_TILE]),
        );
        let window = select(
            current_column,
            current
                .reshape(shape![CHANNEL_TILE, 1])
                .broadcast(shape![CHANNEL_TILE, KERNEL_TILE]),
            previous,
        );
        let valid_columns: Tile<bool, { [CHANNEL_TILE, KERNEL_TILE] }> = lt_tile(
            columns.broadcast(shape![CHANNEL_TILE, KERNEL_TILE]),
            KERNEL.broadcast(shape![CHANNEL_TILE, KERNEL_TILE]),
        );
        let zero: Tile<f32, { [CHANNEL_TILE, KERNEL_TILE] }> =
            constant(0.0f32, shape![CHANNEL_TILE, KERNEL_TILE]);
        let window = select(valid_columns, window, zero);
        let sum: Tile<f32, { [CHANNEL_TILE] }> = reduce_sum(window * coefficients, 1);
        let sum: Tile<f32, { [CHANNEL_TILE] }> = sum.reshape(shape![CHANNEL_TILE]);
        let one: Tile<f32, { [CHANNEL_TILE] }> = constant(1.0f32, shape![CHANNEL_TILE]);
        let negative: Tile<f32, { [CHANNEL_TILE] }> = constant(-1.0f32, shape![CHANNEL_TILE]);
        out.store(sum / (one + exp(sum * negative)));

        let slots: Tile<i32, { [CHANNEL_TILE, KERNEL_TILE] }> =
            columns.broadcast(shape![CHANNEL_TILE, KERNEL_TILE]);
        let mut shifted: Tile<f32, { [CHANNEL_TILE, KERNEL_TILE] }> = current
            .reshape(shape![CHANNEL_TILE, 1])
            .broadcast(shape![CHANNEL_TILE, KERNEL_TILE]);
        for slot in 0i32..(KERNEL - 2i32) {
            let next: Tile<f32, { [CHANNEL_TILE, 1] }> = extract(
                previous,
                [scalar_to_tile(0i32), scalar_to_tile(slot + 1i32)],
            );
            let at_slot: Tile<bool, { [CHANNEL_TILE, KERNEL_TILE] }> =
                eq_tile(slots, slot.broadcast(shape![CHANNEL_TILE, KERNEL_TILE]));
            shifted = select(
                at_slot,
                next.broadcast(shape![CHANNEL_TILE, KERNEL_TILE]),
                shifted,
            );
        }
        state.store(shifted);
    }

    #[cutile::entry()]
    pub fn conv1d_silu_prefill<
        const SEQUENCE_TILE: i32,
        const CHANNEL_TILE: i32,
        const KERNEL_TILE: i32,
        const KERNEL: i32,
    >(
        out: &mut Tensor<f32, { [SEQUENCE_TILE, CHANNEL_TILE] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        state: &mut Tensor<f32, { [1, CHANNEL_TILE, KERNEL_TILE] }>,
        weight: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let coefficients: Tile<f32, { [CHANNEL_TILE, KERNEL_TILE] }> = weight
            .partition(shape![CHANNEL_TILE, KERNEL_TILE])
            .load([pid.1, 0]);
        let positions: Tile<i32, { [KERNEL_TILE] }> = iota(shape![KERNEL_TILE]);
        let slots: Tile<i32, { [CHANNEL_TILE, KERNEL_TILE] }> = positions
            .reshape(shape![1, KERNEL_TILE])
            .broadcast(shape![CHANNEL_TILE, KERNEL_TILE]);
        let current_column = eq_tile(
            slots,
            (KERNEL - 1i32).broadcast(shape![CHANNEL_TILE, KERNEL_TILE]),
        );
        let valid_columns = lt_tile(slots, KERNEL.broadcast(shape![CHANNEL_TILE, KERNEL_TILE]));
        let zero: Tile<f32, { [CHANNEL_TILE, KERNEL_TILE] }> =
            constant(0.0f32, shape![CHANNEL_TILE, KERNEL_TILE]);
        let one: Tile<f32, { [CHANNEL_TILE] }> = constant(1.0f32, shape![CHANNEL_TILE]);
        let negative: Tile<f32, { [CHANNEL_TILE] }> = constant(-1.0f32, shape![CHANNEL_TILE]);
        let mut previous: Tile<f32, { [CHANNEL_TILE, KERNEL_TILE] }> =
            state.load().reshape(shape![CHANNEL_TILE, KERNEL_TILE]);
        for time in 0i32..input.shape()[0] {
            let current: Tile<f32, { [CHANNEL_TILE] }> = input
                .partition(shape![1, CHANNEL_TILE])
                .load([time, pid.1])
                .reshape(shape![CHANNEL_TILE]);
            let current = current
                .reshape(shape![CHANNEL_TILE, 1])
                .broadcast(shape![CHANNEL_TILE, KERNEL_TILE]);
            let window = select(
                valid_columns,
                select(current_column, current, previous),
                zero,
            );
            let sum: Tile<f32, { [CHANNEL_TILE] }> = reduce_sum(window * coefficients, 1);
            let sum = sum.reshape(shape![CHANNEL_TILE]);
            let activated = sum / (one + exp(sum * negative));
            out.partition_mut(shape![1, CHANNEL_TILE])
                .store(activated.reshape(shape![1, CHANNEL_TILE]), [time, 0]);

            let mut shifted = current;
            for slot in 0i32..(KERNEL - 2i32) {
                let next: Tile<f32, { [CHANNEL_TILE, 1] }> = extract(
                    previous,
                    [scalar_to_tile(0i32), scalar_to_tile(slot + 1i32)],
                );
                let at_slot = eq_tile(slots, slot.broadcast(shape![CHANNEL_TILE, KERNEL_TILE]));
                shifted = select(
                    at_slot,
                    next.broadcast(shape![CHANNEL_TILE, KERNEL_TILE]),
                    shifted,
                );
            }
            previous = shifted;
        }
        state.store(previous.reshape(shape![1, CHANNEL_TILE, KERNEL_TILE]));
    }

    #[cutile::entry()]
    pub fn prepare_chunk_qk<const KEY_HEADS: i32, const DK: i32>(
        query: &mut Tensor<f32, { [1, 1, DK] }>,
        key: &mut Tensor<f32, { [1, 1, DK] }>,
        q_input: &Tensor<f32, { [-1, -1] }>,
        k_input: &Tensor<f32, { [-1, -1] }>,
        valid: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let time = pid.0;
        if time < valid {
            let q: Tile<f32, { [1, 1, DK] }> = q_input
                .partition(shape![1, DK])
                .load([time, pid.1])
                .reshape(shape![1, 1, DK]);
            let k: Tile<f32, { [1, 1, DK] }> = k_input
                .partition(shape![1, DK])
                .load([time, pid.1])
                .reshape(shape![1, 1, DK]);
            query.store(q);
            key.store(k);
        }
    }

    #[cutile::entry()]
    pub fn prepare_chunk_value_gate<const VALUE_HEADS: i32, const DV: i32>(
        value: &mut Tensor<f32, { [1, 1, DV] }>,
        gate: &mut Tensor<f32, { [1, 1] }>,
        beta: &mut Tensor<f32, { [1, 1] }>,
        v_input: &Tensor<f32, { [-1, -1] }>,
        a: &Tensor<f32, { [-1, VALUE_HEADS] }>,
        b: &Tensor<f32, { [-1, VALUE_HEADS] }>,
        a_log: &Tensor<f32, { [-1] }>,
        dt_bias: &Tensor<f32, { [-1] }>,
        valid: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let time = pid.0;
        let head = pid.1;
        if time < valid {
            let v: Tile<f32, { [1, 1, DV] }> = v_input
                .partition(shape![1, DV])
                .load([time, head])
                .reshape(shape![1, 1, DV]);
            value.store(v);
            let a_value: Tile<f32, { [1] }> = a
                .partition(shape![1, 1])
                .load([time, head])
                .reshape(shape![1]);
            let b_value: Tile<f32, { [1] }> = b
                .partition(shape![1, 1])
                .load([time, head])
                .reshape(shape![1]);
            let al: Tile<f32, { [1] }> = a_log.partition(shape![1]).load([head]);
            let dt: Tile<f32, { [1] }> = dt_bias.partition(shape![1]).load([head]);
            let one: Tile<f32, { [1] }> = constant(1.0f32, shape![1]);
            let zero: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
            let negative: Tile<f32, { [1] }> = constant(-1.0f32, shape![1]);
            let selected = one / (one + exp(b_value * negative));
            let x = a_value + dt;
            let tiny = exp(min_tile(x, zero));
            let series = tiny - tiny * tiny * constant(0.5f32, shape![1])
                + tiny * tiny * tiny * constant(0.33333334f32, shape![1]);
            let ordinary = max_tile(x, zero) + log(one + exp(absf(x) * negative));
            let cutoff: Tile<f32, { [1] }> = constant(-8.0f32, shape![1]);
            let softplus = select(lt_tile(x, cutoff), series, ordinary);
            let log_decay = exp(al) * softplus * negative;
            beta.store(selected.reshape(shape![1, 1]));
            gate.store(log_decay.reshape(shape![1, 1]));
        }
    }

    #[cutile::entry()]
    pub fn delta_step<const KEY_HEADS: i32, const VALUE_HEADS: i32, const DK: i32, const T: i32>(
        out: &mut Tensor<f32, { [1, 1, T] }>,
        state: &mut Tensor<f32, { [1, DK, T] }>,
        query: &Tensor<f32, { [-1, DK] }>,
        key: &Tensor<f32, { [-1, DK] }>,
        value: &Tensor<f32, { [-1, -1] }>,
        b: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        dt_bias: &Tensor<f32, { [-1] }>,
        eps: f32,
        inv_sqrt_dk: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let key_head = head / (VALUE_HEADS / KEY_HEADS);
        let q: Tile<f32, { [1, DK] }> = query.partition(shape![1, DK]).load([key_head, 0]);
        let k: Tile<f32, { [1, DK] }> = key.partition(shape![1, DK]).load([key_head, 0]);
        let q_norm: Tile<f32, { [1] }> = reduce_sum(q * q, 1);
        let k_norm: Tile<f32, { [1] }> = reduce_sum(k * k, 1);
        let q_scale = rsqrt(q_norm + eps.broadcast(shape![1]), ftz::Disabled)
            * inv_sqrt_dk.broadcast(shape![1]);
        let k_scale = rsqrt(k_norm + eps.broadcast(shape![1]), ftz::Disabled);
        let q: Tile<f32, { [1, DK, T] }> =
            (q * q_scale.reshape(shape![1, 1]).broadcast(shape![1, DK]))
                .reshape(shape![1, DK, 1])
                .broadcast(shape![1, DK, T]);
        let k: Tile<f32, { [1, DK, T] }> =
            (k * k_scale.reshape(shape![1, 1]).broadcast(shape![1, DK]))
                .reshape(shape![1, DK, 1])
                .broadcast(shape![1, DK, T]);

        let b: Tile<f32, { [1] }> = b.partition(shape![1]).load([head]);
        let a: Tile<f32, { [1] }> = a.partition(shape![1]).load([head]);
        let a_log: Tile<f32, { [1] }> = a_log.partition(shape![1]).load([head]);
        let dt: Tile<f32, { [1] }> = dt_bias.partition(shape![1]).load([head]);
        let one: Tile<f32, { [1] }> = constant(1.0f32, shape![1]);
        let zero: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
        let negative: Tile<f32, { [1] }> = constant(-1.0f32, shape![1]);
        let beta = one / (one + exp(b * negative));
        let x = a + dt;
        let tiny = exp(min_tile(x, zero));
        let series = tiny - tiny * tiny * constant(0.5f32, shape![1])
            + tiny * tiny * tiny * constant(0.33333334f32, shape![1]);
        let ordinary = max_tile(x, zero) + log(one + exp(absf(x) * negative));
        let cutoff: Tile<f32, { [1] }> = constant(-8.0f32, shape![1]);
        let softplus = select(lt_tile(x, cutoff), series, ordinary);
        let decay = exp(exp(a_log) * softplus * negative);

        let old: Tile<f32, { [1, DK, T] }> = state.load();
        let decayed = old * decay.reshape(shape![1, 1, 1]).broadcast(shape![1, DK, T]);
        let memory: Tile<f32, { [1, 1, T] }> = reduce_sum(decayed * k, 1);
        let memory: Tile<f32, { [1, T] }> = memory.reshape(shape![1, T]);
        let v: Tile<f32, { [1, T] }> = value.partition(shape![1, T]).load([head, pid.2]);
        let delta = (v - memory) * beta.reshape(shape![1, 1]).broadcast(shape![1, T]);
        let next = decayed + k * delta.reshape(shape![1, 1, T]).broadcast(shape![1, DK, T]);
        state.store(next);
        let result: Tile<f32, { [1, 1, T] }> = reduce_sum(next * q, 1);
        let result: Tile<f32, { [1, T] }> = result.reshape(shape![1, T]);
        out.store(result.reshape(shape![1, 1, T]));
    }

    #[cutile::entry()]
    pub fn gated_silu<const DV: i32>(
        out: &mut Tensor<f32, { [1, DV] }>,
        input: &Tensor<f32, { [-1, DV] }>,
        gate: &Tensor<f32, { [-1, DV] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let x: Tile<f32, { [1, DV] }> = input.partition(shape![1, DV]).load([pid.0, 0]);
        let z: Tile<f32, { [1, DV] }> = gate.partition(shape![1, DV]).load([pid.0, 0]);
        let one: Tile<f32, { [1, DV] }> = constant(1.0f32, shape![1, DV]);
        let negative: Tile<f32, { [1, DV] }> = constant(-1.0f32, shape![1, DV]);
        out.store(x * z / (one + exp(z * negative)));
    }
}

pub use kernels::{
    conv1d_silu, conv1d_silu_prefill, delta_step, gated_silu, prepare_chunk_qk,
    prepare_chunk_value_gate,
};

use crate::weights::Q8_BLOCK;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnDimensionError;

impl std::fmt::Display for GdnDimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid GDN dimensions")
    }
}

impl std::error::Error for GdnDimensionError {}

#[derive(Debug, Clone, Copy)]
pub struct GdnDims {
    hidden: usize,
    key_heads: usize,
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel: usize,
    conv_dim: usize,
    eps: f32,
}

impl GdnDims {
    pub fn new(
        hidden: usize,
        key_heads: usize,
        value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        conv_kernel: usize,
        eps: f32,
    ) -> Result<Self, GdnDimensionError> {
        let valid = (|| {
            let key_dim = key_heads.checked_mul(key_head_dim)?;
            let value_dim = value_heads.checked_mul(value_head_dim)?;
            let conv_dim = key_dim.checked_mul(2)?.checked_add(value_dim)?;
            let state_tile = (conv_kernel.checked_sub(1))?.checked_next_power_of_two()?;
            (hidden > 0
                && hidden % Q8_BLOCK == 0
                && key_heads > 0
                && value_heads > 0
                && value_heads % key_heads == 0
                && key_head_dim > 0
                && key_head_dim.is_power_of_two()
                && value_head_dim > 0
                && value_head_dim.is_power_of_two()
                && conv_kernel >= 2
                && eps.is_finite()
                && eps > 0.0
                && hidden <= i32::MAX as usize
                && value_heads <= i32::MAX as usize
                && conv_dim <= i32::MAX as usize
                && value_dim % Q8_BLOCK == 0
                && state_tile <= i32::MAX as usize
                && value_heads
                    .checked_mul(key_head_dim)?
                    .checked_mul(value_head_dim)?
                    <= i32::MAX as usize
                && conv_dim.checked_mul(conv_kernel)? <= i32::MAX as usize)
                .then_some(conv_dim)
        })();
        let conv_dim = valid.ok_or(GdnDimensionError)?;
        Ok(Self {
            hidden,
            key_heads,
            value_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel,
            conv_dim,
            eps,
        })
    }

    pub fn hidden(self) -> usize {
        self.hidden
    }
    pub fn key_heads(self) -> usize {
        self.key_heads
    }
    pub fn value_heads(self) -> usize {
        self.value_heads
    }
    pub fn key_head_dim(self) -> usize {
        self.key_head_dim
    }
    pub fn value_head_dim(self) -> usize {
        self.value_head_dim
    }
    pub fn key_dim(self) -> usize {
        self.key_heads * self.key_head_dim
    }
    pub fn value_dim(self) -> usize {
        self.value_heads * self.value_head_dim
    }
    pub fn conv_kernel(self) -> usize {
        self.conv_kernel
    }
    pub fn conv_dim(self) -> usize {
        self.conv_dim
    }
    pub fn eps(self) -> f32 {
        self.eps
    }
    pub fn inv_sqrt_key_dim(self) -> f32 {
        1.0 / (self.key_head_dim as f32).sqrt()
    }
    pub fn conv_generics(self, channel_tile: usize) -> Vec<String> {
        vec![
            channel_tile.to_string(),
            self.conv_kernel.next_power_of_two().to_string(),
            self.conv_kernel.to_string(),
        ]
    }
    pub fn delta_generics(self, value_tile: usize) -> Vec<String> {
        vec![
            self.key_heads.to_string(),
            self.value_heads.to_string(),
            self.key_head_dim.to_string(),
            value_tile.to_string(),
        ]
    }
}
