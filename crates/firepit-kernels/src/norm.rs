#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn rms_norm<const H: i32>(
        out: &mut Tensor<f32, { [1, H] }>,
        input: &Tensor<f32, { [-1, -1, H] }>,
        weight: &Tensor<f32, { [H] }>,
        inv_h: f32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let x: Tile<f32, { [1, H] }> = input
            .partition(shape![1, 1, H])
            .load([pid.0, pid.1, 0])
            .reshape(shape![1, H]);
        let squares: Tile<f32, { [1] }> = reduce_sum(x * x, 1);
        let squares: Tile<f32, { [1, 1] }> = squares.reshape(shape![1, 1]);
        let scale = rsqrt(
            squares * inv_h.broadcast(shape![1, 1]) + eps.broadcast(shape![1, 1]),
            ftz::Disabled,
        );
        let gamma: Tile<f32, { [H] }> = weight.partition(shape![H]).load([0]);
        let gamma: Tile<f32, { [1, H] }> = gamma.reshape(shape![1, H]);
        out.store(x * scale.broadcast(shape![1, H]) * gamma);
    }

    #[cutile::entry()]
    pub fn add_norm<const H: i32>(
        out: &mut Tensor<f32, { [1, H] }>,
        normed: &mut Tensor<f32, { [1, H] }>,
        residual: &Tensor<f32, { [-1, H] }>,
        delta: &Tensor<f32, { [-1, H] }>,
        weight: &Tensor<f32, { [H] }>,
        inv_h: f32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let r: Tile<f32, { [1, H] }> = residual.partition(shape![1, H]).load([pid.0, 0]);
        let d: Tile<f32, { [1, H] }> = delta.partition(shape![1, H]).load([pid.0, 0]);
        let x = r + d;
        out.store(x);
        let squares: Tile<f32, { [1] }> = reduce_sum(x * x, 1);
        let squares: Tile<f32, { [1, 1] }> = squares.reshape(shape![1, 1]);
        let scale = rsqrt(
            squares * inv_h.broadcast(shape![1, 1]) + eps.broadcast(shape![1, 1]),
            ftz::Disabled,
        );
        let gamma: Tile<f32, { [H] }> = weight.partition(shape![H]).load([0]);
        let gamma: Tile<f32, { [1, H] }> = gamma.reshape(shape![1, H]);
        normed.store(x * scale.broadcast(shape![1, H]) * gamma);
    }
}

pub use kernels::{add_norm, rms_norm};
