#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn conv1d_silu<const STATE_TILE: i32>(
        out: &mut Tensor<f32, { [1] }>,
        input: &Tensor<f32, { [-1] }>,
        state: &mut Tensor<f32, { [1, STATE_TILE] }>,
        weight: &Tensor<f32, { [-1, -1] }>,
        kernel_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let previous: Tile<f32, { [1, STATE_TILE] }> = state.load();
        let taps = weight.partition(shape![1, 1]);
        let mut sum: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
        for tap in 0i32..(kernel_size - 1i32) {
            let past: Tile<f32, { [1, 1] }> =
                extract(previous, [scalar_to_tile(0i32), scalar_to_tile(tap)]);
            let coefficient: Tile<f32, { [1, 1] }> = taps.load([pid.0, tap]);
            sum = sum + (past * coefficient).reshape(shape![1]);
        }
        let current: Tile<f32, { [1] }> = input.partition(shape![1]).load([pid.0]);
        let coefficient: Tile<f32, { [1, 1] }> = taps.load([pid.0, kernel_size - 1i32]);
        sum = sum + current * coefficient.reshape(shape![1]);
        let one: Tile<f32, { [1] }> = constant(1.0f32, shape![1]);
        let negative: Tile<f32, { [1] }> = constant(-1.0f32, shape![1]);
        out.store(sum / (one + exp(sum * negative)));

        let positions: Tile<i32, { [STATE_TILE] }> = iota(shape![STATE_TILE]);
        let slots: Tile<i32, { [1, STATE_TILE] }> = positions.reshape(shape![1, STATE_TILE]);
        let mut shifted: Tile<f32, { [1, STATE_TILE] }> = current
            .reshape(shape![1, 1])
            .broadcast(shape![1, STATE_TILE]);
        for slot in 0i32..(kernel_size - 2i32) {
            let next: Tile<f32, { [1, 1] }> = extract(
                previous,
                [scalar_to_tile(0i32), scalar_to_tile(slot + 1i32)],
            );
            let at_slot: Tile<bool, { [1, STATE_TILE] }> =
                eq_tile(slots, slot.broadcast(shape![1, STATE_TILE]));
            shifted = select(at_slot, next.broadcast(shape![1, STATE_TILE]), shifted);
        }
        state.store(shifted);
    }
}

pub use kernels::conv1d_silu;

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
}

impl GdnDims {
    pub fn new(
        hidden: usize,
        key_heads: usize,
        value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        conv_kernel: usize,
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
                && key_head_dim > 0
                && value_head_dim > 0
                && conv_kernel >= 2
                && hidden <= i32::MAX as usize
                && value_heads <= i32::MAX as usize
                && conv_dim <= i32::MAX as usize
                && state_tile <= i32::MAX as usize
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
    pub fn conv_generics(self) -> Vec<String> {
        vec![(self.conv_kernel - 1).next_power_of_two().to_string()]
    }
}
