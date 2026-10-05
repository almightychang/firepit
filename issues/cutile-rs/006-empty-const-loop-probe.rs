use cutile::api;
use cutile::prelude::DeviceOp;
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::ToHostVecOp;
use cutile::tile_kernel::TileKernel;
use std::error::Error;
use std::sync::Arc;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn extract_in_const_loop<const N: i32>(
        out: &mut Tensor<f32, { [4, 1] }>,
        input: &Tensor<f32, { [4, 4] }>,
    ) {
        let tile: Tile<f32, { [4, 4] }> = input.partition(shape![4, 4]).load([0, 0]);
        let mut acc: Tile<f32, { [4, 1] }> = constant(0.0f32, shape![4, 1]);
        for slot in 0i32..(N - 2i32) {
            let column: Tile<f32, { [4, 1] }> =
                extract(tile, [scalar_to_tile(0i32), scalar_to_tile(slot + 1i32)]);
            acc = acc + column;
        }
        out.store(acc);
    }

    #[cutile::entry()]
    pub fn extract_in_runtime_loop(
        out: &mut Tensor<f32, { [4, 1] }>,
        input: &Tensor<f32, { [4, 4] }>,
        n: i32,
    ) {
        let tile: Tile<f32, { [4, 4] }> = input.partition(shape![4, 4]).load([0, 0]);
        let mut acc: Tile<f32, { [4, 1] }> = constant(0.0f32, shape![4, 1]);
        for slot in 0i32..(n - 2i32) {
            let column: Tile<f32, { [4, 1] }> =
                extract(tile, [scalar_to_tile(0i32), scalar_to_tile(slot + 1i32)]);
            acc = acc + column;
        }
        out.store(acc);
    }

    #[cutile::entry()]
    pub fn load_in_const_loop<const N: i32>(
        out: &mut Tensor<f32, { [1, 4] }>,
        input: &Tensor<f32, { [-1, 4] }>,
    ) {
        let rows = input.partition(shape![1, 4]);
        let mut acc: Tile<f32, { [1, 4] }> = constant(0.0f32, shape![1, 4]);
        for row in 0i32..(N - 2i32) {
            let next: Tile<f32, { [1, 4] }> = rows.load([row + 1i32, 0]);
            acc = acc + next;
        }
        out.store(acc);
    }

    #[cutile::entry()]
    pub fn plain_const_loop<const N: i32>(out: &mut Tensor<f32, { [1, 4] }>) {
        let mut acc: Tile<f32, { [1, 4] }> = constant(0.0f32, shape![1, 4]);
        for _ in 0i32..(N - 2i32) {
            acc = acc + constant(1.0f32, shape![1, 4]);
        }
        out.store(acc);
    }
}

fn report(case: &str, result: Result<Vec<f32>, Box<dyn Error>>) {
    match result {
        Ok(values) => println!("{case}: ok {values:?}"),
        Err(error) => println!("{case}: ERROR {error}"),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    for n in [2, 3, 4] {
        let input = api::DeviceOpReshape::reshape(
            api::copy_host_vec_to_device(&Arc::new((0..16).map(|v| v as f32).collect::<Vec<f32>>())),
            &[4, 4],
        )
        .sync()?;
        report(&format!("extract, const N={n}"), (|| {
            let mut out = api::zeros::<f32>(&[4, 1]).sync()?;
            kernels::extract_in_const_loop((&mut out).partition([4, 1]), &input)
                .generics(vec![n.to_string()])
                .sync()?;
            Ok(out.dup().to_host_vec().sync()?)
        })());
        report(&format!("extract, runtime n={n}"), (|| {
            let mut out = api::zeros::<f32>(&[4, 1]).sync()?;
            kernels::extract_in_runtime_loop((&mut out).partition([4, 1]), &input, n).sync()?;
            Ok(out.dup().to_host_vec().sync()?)
        })());
        report(&format!("load, const N={n}"), (|| {
            let mut out = api::zeros::<f32>(&[1, 4]).sync()?;
            kernels::load_in_const_loop((&mut out).partition([1, 4]), &input)
                .generics(vec![n.to_string()])
                .sync()?;
            Ok(out.dup().to_host_vec().sync()?)
        })());
        report(&format!("plain, const N={n}"), (|| {
            let mut out = api::zeros::<f32>(&[1, 4]).sync()?;
            kernels::plain_const_loop((&mut out).partition([1, 4]))
                .generics(vec![n.to_string()])
                .sync()?;
            Ok(out.dup().to_host_vec().sync()?)
        })());
    }
    Ok(())
}
