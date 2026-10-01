#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn swiglu<const I: i32>(
        out: &mut Tensor<f32, { [1, I] }>,
        gate_up: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part = gate_up.partition(shape![1, I]);
        let gate: Tile<f32, { [1, I] }> = part.load([pid.0, 0i32]);
        let up: Tile<f32, { [1, I] }> = part.load([pid.0, 1i32]);
        let one: Tile<f32, { [1, I] }> = constant(1.0f32, shape![1, I]);
        let minus_one: Tile<f32, { [1, I] }> = constant(-1.0f32, shape![1, I]);
        out.store(gate / (one + exp(gate * minus_one)) * up);
    }
}

pub use kernels::swiglu;
