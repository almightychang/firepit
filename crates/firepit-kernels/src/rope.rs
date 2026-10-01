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

fn validate(head_dim: usize, rotary_dim: usize) {
    assert!(head_dim > 0 && head_dim <= i32::MAX as usize);
    assert!(rotary_dim >= 2 && rotary_dim <= head_dim && rotary_dim.is_power_of_two());
    assert_eq!(head_dim % rotary_dim, 0);
}

pub fn generics(heads: usize, head_dim: usize, rotary_dim: usize) -> Vec<String> {
    validate(head_dim, rotary_dim);
    assert!(heads > 0 && heads <= i32::MAX as usize);
    vec![
        heads.to_string(),
        head_dim.to_string(),
        rotary_dim.to_string(),
        (rotary_dim / 2).to_string(),
    ]
}

pub fn frequencies(
    positions: &[usize],
    theta: f32,
    head_dim: usize,
    rotary_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    validate(head_dim, rotary_dim);
    assert!(theta.is_finite() && theta > 0.0);
    let half = rotary_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(i as f32 / half as f32))
        .collect();
    let mut cos = Vec::with_capacity(positions.len() * head_dim);
    let mut sin = Vec::with_capacity(positions.len() * head_dim);
    for &position in positions {
        let angles: Vec<f32> = inv.iter().map(|&freq| position as f32 * freq).collect();
        for _ in 0..2 {
            cos.extend(angles.iter().map(|x| x.cos()));
            sin.extend(angles.iter().map(|x| x.sin()));
        }
        cos.resize(cos.len() + head_dim - rotary_dim, 1.0);
        sin.resize(sin.len() + head_dim - rotary_dim, 0.0);
    }
    (cos, sin)
}
