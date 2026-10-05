use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f4e2m1fnx2};
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

    #[cutile::entry()]
    pub fn unpack_row_sum<const BN: i32, const KT: i32, const KT_PACKED: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let values: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let values: Tile<f32, { [BN, KT] }> = convert_tile(values);
            let part: Tile<f32, { [BN] }> = reduce_sum(values, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn unpack_store<const BN: i32, const KT: i32, const KT_PACKED: i32>(
        y: &mut Tensor<f32, { [BN, KT] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> =
            w.partition(shape![BN, KT_PACKED]).load([pid.0, pid.1]);
        let values: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
        let values: Tile<f32, { [BN, KT] }> = convert_tile(values);
        y.store(values);
    }
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    cutile::jit_cache::enable_default()?;
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse())
        .collect::<std::result::Result<_, _>>()?;
    let [rows, cols, bn, kt] = args[..] else {
        return Err("usage: nvfp4_unpack_reduce_probe N K BN KT".into());
    };
    if rows % bn != 0 || cols % kt != 0 || kt % 2 != 0 {
        return Err("BN must divide N and KT (even) must divide K".into());
    }
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    // 0x22 packs two FP4 E2M1 values of 1.0.
    let w: Arc<Tensor<f4e2m1fnx2>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(vec![
            f4e2m1fnx2::from_bits(0x22);
            rows * cols / 2
        ]))
        .reshape(&[rows, cols / 2])
        .sync_on(&stream)?,
    );
    let generics = vec![bn.to_string(), kt.to_string(), (kt / 2).to_string()];
    let options = BenchOptions::default();
    println!("kernel,N,K,BN,KT,ms");

    let mut sums = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let ms = do_bench(&stream, &options, |stream| {
        kernels::unpack_row_sum(
            (&mut sums).partition([1, bn]),
            w.clone(),
            (cols / kt) as i32,
        )
        .generics(generics.clone())
        .sync_on(stream)
        .map(|_| ())
        .map_err(Into::into)
    })?
    .median_ms();
    println!("unpack_row_sum,{rows},{cols},{bn},{kt},{ms:.6}");
    if sums.dup().to_host_vec().sync_on(&stream)? != vec![cols as f32; rows] {
        return Err("unpack_row_sum: every row should sum to K".into());
    }

    let mut values = api::zeros::<f32>(&[rows, cols]).sync_on(&stream)?;
    let ms = do_bench(&stream, &options, |stream| {
        kernels::unpack_store((&mut values).partition([bn, kt]), w.clone())
            .generics(generics.clone())
            .sync_on(stream)
            .map(|_| ())
            .map_err(Into::into)
    })?
    .median_ms();
    println!("unpack_store,{rows},{cols},{bn},{kt},{ms:.6}");
    if values.dup().to_host_vec().sync_on(&stream)? != vec![1.0f32; rows * cols] {
        return Err("unpack_store: every value should be 1.0".into());
    }
    Ok(())
}
