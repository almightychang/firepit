#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn add<const H: i32>(
        out: &mut Tensor<f32, { [1, H] }>,
        a: &Tensor<f32, { [-1, H] }>,
        b: &Tensor<f32, { [-1, H] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let ap = a.partition(shape![1, H]).load([pid.0, 0]);
        let bp = b.partition(shape![1, H]).load([pid.0, 0]);
        out.store(ap + bp);
    }
}

pub use kernels::add;
