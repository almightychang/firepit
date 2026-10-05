use cutile::api;
use cutile::prelude::DeviceOp;
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use std::error::Error;
use std::sync::Arc;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn reduce_then_mul<const N: i32>(
        out: &mut Tensor<f32, { [N] }>,
        pairs: &Tensor<f32, { [-1, 2] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let values: Tile<f32, { [N, 2] }> = pairs.partition(shape![N, 2]).load([0, 0]);
        let sums: Tile<f32, { [N] }> = reduce_sum(values, 1i32);
        let factor: Tile<f32, { [N] }> = scale.partition(shape![N]).load([0]);
        out.store(sums * factor);
    }

    #[cutile::entry()]
    pub fn reduce_reshape_then_mul<const N: i32>(
        out: &mut Tensor<f32, { [N] }>,
        pairs: &Tensor<f32, { [-1, 2] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let values: Tile<f32, { [N, 2] }> = pairs.partition(shape![N, 2]).load([0, 0]);
        let sums: Tile<f32, { [N] }> = reduce_sum(values, 1i32);
        let sums: Tile<f32, { [N] }> = sums.reshape(shape![N]);
        let factor: Tile<f32, { [N] }> = scale.partition(shape![N]).load([0]);
        out.store(sums * factor);
    }

    #[cutile::entry()]
    pub fn reduce_then_mul_fixed(
        out: &mut Tensor<f32, { [4] }>,
        pairs: &Tensor<f32, { [-1, 2] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let values: Tile<f32, { [4, 2] }> = pairs.partition(shape![4, 2]).load([0, 0]);
        let sums: Tile<f32, { [4] }> = reduce_sum(values, 1i32);
        let factor: Tile<f32, { [4] }> = scale.partition(shape![4]).load([0]);
        out.store(sums * factor);
    }
}

fn report(case: &str, result: Result<Vec<f32>, Box<dyn Error>>) {
    match result {
        Ok(values) => println!("{case}: ok {values:?}"),
        Err(error) => println!("{case}: ERROR {error}"),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let pairs = api::DeviceOpReshape::reshape(
        api::copy_host_vec_to_device(&Arc::new((0..8).map(|v| v as f32).collect::<Vec<f32>>())),
        &[4, 2],
    )
    .sync()?;
    let scale = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32, 2.0, 3.0, 4.0])).sync()?;
    report("reduce_sum then mul, const generic N=4", (|| {
        let mut out = api::zeros::<f32>(&[4]).sync()?;
        kernels::reduce_then_mul((&mut out).partition([4]), &pairs, &scale)
            .generics(vec!["4".to_string()])
            .sync()?;
        Ok(out.dup().to_host_vec().sync()?)
    })());
    report("reduce_sum, reshape, then mul, const generic N=4", (|| {
        let mut out = api::zeros::<f32>(&[4]).sync()?;
        kernels::reduce_reshape_then_mul((&mut out).partition([4]), &pairs, &scale)
            .generics(vec!["4".to_string()])
            .sync()?;
        Ok(out.dup().to_host_vec().sync()?)
    })());
    report("reduce_sum then mul, literal shape", (|| {
        let mut out = api::zeros::<f32>(&[4]).sync()?;
        kernels::reduce_then_mul_fixed((&mut out).partition([4]), &pairs, &scale).sync()?;
        Ok(out.dup().to_host_vec().sync()?)
    })());
    Ok(())
}
