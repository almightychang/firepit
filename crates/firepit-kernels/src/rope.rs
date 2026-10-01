#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn rope<const HEADS: i32, const D: i32, const R: i32, const HALF: i32>(
        out: &mut Tensor<f32, { [1, R] }>,
        input: &Tensor<f32, { [-1, D] }>,
        cos: &Tensor<f32, { [-1, D] }>,
        sin: &Tensor<f32, { [-1, D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let x: Tile<f32, { [1, R] }> = input.partition(shape![1, R]).load([pid.0, pid.1]);
        let c: Tile<f32, { [1, R] }> = cos.partition(shape![1, R]).load([pid.0 / HEADS, pid.1]);
        let s: Tile<f32, { [1, R] }> = sin.partition(shape![1, R]).load([pid.0 / HEADS, pid.1]);
        let x0: Tile<f32, { [1, HALF] }> = extract(x, [scalar_to_tile(0i32), scalar_to_tile(0i32)]);
        let x1: Tile<f32, { [1, HALF] }> = extract(x, [scalar_to_tile(0i32), scalar_to_tile(1i32)]);
        let negative: Tile<f32, { [1, HALF] }> = x1 * constant(-1.0f32, shape![1, HALF]);
        let rotated: Tile<f32, { [1, R] }> = cat(negative, x0, 1);
        out.store(x * c + rotated * s);
    }
}

pub use kernels::rope;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeDimensionError {
    InvalidDimensions,
}

impl std::fmt::Display for RopeDimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RopeDimensionError {}

#[derive(Debug, Clone, Copy)]
pub struct RopeDims {
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
}

impl RopeDims {
    pub fn new(
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f32,
    ) -> Result<Self, RopeDimensionError> {
        if heads == 0
            || heads > i32::MAX as usize
            || head_dim == 0
            || head_dim > i32::MAX as usize
            || rotary_dim < 2
            || rotary_dim > i32::MAX as usize
            || rotary_dim > head_dim
            || !rotary_dim.is_power_of_two()
            || head_dim % rotary_dim != 0
            || !theta.is_finite()
            || theta <= 0.0
        {
            return Err(RopeDimensionError::InvalidDimensions);
        }
        Ok(Self {
            heads,
            head_dim,
            rotary_dim,
            theta,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }
}

pub fn generics(dims: &RopeDims) -> Vec<String> {
    vec![
        dims.heads.to_string(),
        dims.head_dim.to_string(),
        dims.rotary_dim.to_string(),
        (dims.rotary_dim / 2).to_string(),
    ]
}

pub fn frequencies(positions: &[usize], dims: &RopeDims) -> (Vec<f32>, Vec<f32>) {
    let half = dims.rotary_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / dims.theta.powf(i as f32 / half as f32))
        .collect();
    let mut cos = Vec::with_capacity(positions.len() * dims.head_dim);
    let mut sin = Vec::with_capacity(positions.len() * dims.head_dim);
    for &position in positions {
        let angles: Vec<f32> = inv.iter().map(|&freq| position as f32 * freq).collect();
        for _ in 0..2 {
            cos.extend(angles.iter().map(|x| x.cos()));
            sin.extend(angles.iter().map(|x| x.sin()));
        }
        cos.resize(cos.len() + dims.head_dim - dims.rotary_dim, 1.0);
        sin.resize(sin.len() + dims.head_dim - dims.rotary_dim, 0.0);
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::{RopeDimensionError, RopeDims};

    #[test]
    fn rejects_invalid_dimensions() {
        let bad = [
            (0, 256, 64, 10_000_000.0),
            (1, 0, 64, 10_000_000.0),
            (1, 256, 1, 10_000_000.0),
            (1, 256, 48, 10_000_000.0),
            (1, 128, 256, 10_000_000.0),
            (1, 192, 128, 10_000_000.0),
            (1, 256, 64, 0.0),
            (1, 256, 64, -1.0),
            (1, 256, 64, f32::NAN),
            (1, 256, 64, f32::INFINITY),
            (i32::MAX as usize + 1, 256, 64, 10_000_000.0),
            (1, i32::MAX as usize + 1, 64, 10_000_000.0),
            (
                1,
                i32::MAX as usize + 1,
                i32::MAX as usize + 1,
                10_000_000.0,
            ),
        ];
        for (heads, head_dim, rotary_dim, theta) in bad {
            assert_eq!(
                RopeDims::new(heads, head_dim, rotary_dim, theta).unwrap_err(),
                RopeDimensionError::InvalidDimensions
            );
        }
        assert!(RopeDims::new(16, 256, 64, 10_000_000.0).is_ok());
        assert!(RopeDims::new(2, 128, 128, 10_000_000.0).is_ok());
    }
}
