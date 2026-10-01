#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn rope<const HEADS: i32>(
        out: &mut Tensor<f32, { [1, 256] }>,
        input: &Tensor<f32, { [-1, 256] }>,
        cos: &Tensor<f32, { [-1, 64] }>,
        sin: &Tensor<f32, { [-1, 64] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let x: Tile<f32, { [1, 256] }> = input.partition(shape![1, 256]).load([pid.0, 0]);
        let c: Tile<f32, { [1, 64] }> = cos.partition(shape![1, 64]).load([pid.0 / HEADS, 0]);
        let s: Tile<f32, { [1, 64] }> = sin.partition(shape![1, 64]).load([pid.0 / HEADS, 0]);
        let x0: Tile<f32, { [1, 32] }> = extract(x, [scalar_to_tile(0i32), scalar_to_tile(0i32)]);
        let x1: Tile<f32, { [1, 32] }> = extract(x, [scalar_to_tile(0i32), scalar_to_tile(1i32)]);
        let c0: Tile<f32, { [1, 32] }> = extract(c, [scalar_to_tile(0i32), scalar_to_tile(0i32)]);
        let c1: Tile<f32, { [1, 32] }> = extract(c, [scalar_to_tile(0i32), scalar_to_tile(1i32)]);
        let s0: Tile<f32, { [1, 32] }> = extract(s, [scalar_to_tile(0i32), scalar_to_tile(0i32)]);
        let s1: Tile<f32, { [1, 32] }> = extract(s, [scalar_to_tile(0i32), scalar_to_tile(1i32)]);
        let rotated: Tile<f32, { [1, 64] }> = cat(x0 * c0 - x1 * s0, x1 * c1 + x0 * s1, 1);
        let pass0: Tile<f32, { [1, 64] }> =
            extract(x, [scalar_to_tile(0i32), scalar_to_tile(1i32)]);
        let pass1: Tile<f32, { [1, 64] }> =
            extract(x, [scalar_to_tile(0i32), scalar_to_tile(2i32)]);
        let pass2: Tile<f32, { [1, 64] }> =
            extract(x, [scalar_to_tile(0i32), scalar_to_tile(3i32)]);
        let first: Tile<f32, { [1, 128] }> = cat(rotated, pass0, 1);
        let second: Tile<f32, { [1, 128] }> = cat(pass1, pass2, 1);
        let result: Tile<f32, { [1, 256] }> = cat(first, second, 1);
        out.store(result);
    }
}

pub use kernels::rope;

pub fn frequencies(positions: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let inv: Vec<f32> = (0..32)
        .map(|i| 1.0 / 10_000_000.0f32.powf(i as f32 / 32.0))
        .collect();
    let mut cos = Vec::with_capacity(positions.len() * 64);
    let mut sin = Vec::with_capacity(positions.len() * 64);
    for &position in positions {
        let angles: Vec<f32> = inv.iter().map(|&freq| position as f32 * freq).collect();
        for _ in 0..2 {
            cos.extend(angles.iter().map(|x| x.cos()));
            sin.extend(angles.iter().map(|x| x.sin()));
        }
    }
    (cos, sin)
}
