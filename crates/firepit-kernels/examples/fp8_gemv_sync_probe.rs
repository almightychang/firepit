use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    // Reads the same FP8 tiles and sums the raw bytes: the memory floor.
    #[cutile::entry()]
    pub fn fp8_load<const BN: i32, const KT: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT]);
        let mut acc: Tile<u32, { [BN, KT] }> = constant(0u32, shape![BN, KT]);
        for kc in 0i32..chunks {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = wp.load([pid.1, kc]);
            let bits: Tile<u8, { [BN, KT] }> = bitcast(weights);
            let wide: Tile<u32, { [BN, KT] }> = exti(bits);
            acc = acc + wide;
        }
        let sum: Tile<u32, { [BN] }> = reduce_sum(acc, 1i32);
        let out: Tile<f32, { [BN] }> = convert_tile(sum);
        y.store(out.reshape(shape![1, BN]));
    }

    // y = W x for FP8 W: convert, multiply by the broadcast input, reduce over K per tile.
    #[cutile::entry()]
    pub fn fp8_dot<const BN: i32, const KT: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        x: &Tensor<f32, { [1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT]);
        let xp = x.partition(shape![1, KT]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = wp.load([pid.1, kc]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, KT] }> = xp.load([0, kc]);
            let part: Tile<f32, { [BN] }> =
                reduce_sum(weights * xb.broadcast(shape![BN, KT]), 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse())
        .collect::<std::result::Result<_, _>>()?;
    let [rows, cols, bn] = args[..] else {
        return Err("usage: fp8_gemv_sync_probe N K BN".into());
    };
    let kt = 128;
    if rows % bn != 0 || cols % kt != 0 {
        return Err("BN must divide N and 128 must divide K".into());
    }
    cutile::jit_cache::enable_default()?;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    // 0x38 is 1.0 in E4M3, so every output row is K for both kernels.
    let w: Arc<Tensor<f8e4m3fn>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(vec![f8e4m3fn(0x38); rows * cols]))
            .reshape(&[rows, cols])
            .sync_on(&stream)?,
    );
    let x: Arc<Tensor<f32>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
            .reshape(&[1, cols])
            .sync_on(&stream)?,
    );
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let chunks = (cols / kt) as i32;
    let generics = vec![bn.to_string(), kt.to_string()];
    let options = BenchOptions::default();
    println!("kernel,N,K,BN,ms");

    let ms = do_bench(&stream, &options, |s| {
        kernels::fp8_load((&mut y).partition([1, bn]), w.clone(), chunks)
            .generics(generics.clone())
            .sync_on(s)
            .map(|_| ())
            .map_err(Into::into)
    })?
    .median_ms();
    println!("fp8_load,{rows},{cols},{bn},{ms:.6}");
    let ms = do_bench(&stream, &options, |s| {
        kernels::fp8_dot((&mut y).partition([1, bn]), w.clone(), x.clone(), chunks)
            .generics(generics.clone())
            .sync_on(s)
            .map(|_| ())
            .map_err(Into::into)
    })?
    .median_ms();
    println!("fp8_dot,{rows},{cols},{bn},{ms:.6}");
    if y.dup().to_host_vec().sync_on(&stream)? != vec![cols as f32; rows] {
        return Err("fp8_dot: every row should be K".into());
    }
    Ok(())
}
