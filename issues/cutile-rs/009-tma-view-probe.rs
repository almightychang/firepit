use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream};
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// The same bytes and the same [BN, KT] tile per step, read through three views of one
// contiguous buffer. Only the rank and the extent of the innermost dimension differ.
#[cutile::module]
mod views {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn inner_32<const BN: i32, const NB: i32>(
        y: &mut Tensor<u32, { [1, BN] }>,
        w: &Tensor<u8, { [-1, -1, 32] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, NB, 32]);
        let mut acc: Tile<u32, { [BN] }> = constant(0u32, shape![BN]);
        for kc in 0i32..chunks {
            let bytes: Tile<u8, { [BN, NB, 32] }> = wp.load([pid.1, kc, 0]);
            let wide: Tile<u32, { [BN, NB, 32] }> = exti(bytes);
            let part: Tile<u32, { [BN, NB] }> = reduce_sum(wide, 2i32);
            let part: Tile<u32, { [BN] }> = reduce_sum(part, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn inner_128<const BN: i32, const NB: i32>(
        y: &mut Tensor<u32, { [1, BN] }>,
        w: &Tensor<u8, { [-1, -1, 128] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, NB, 128]);
        let mut acc: Tile<u32, { [BN] }> = constant(0u32, shape![BN]);
        for kc in 0i32..chunks {
            let bytes: Tile<u8, { [BN, NB, 128] }> = wp.load([pid.1, kc, 0]);
            let wide: Tile<u32, { [BN, NB, 128] }> = exti(bytes);
            let part: Tile<u32, { [BN, NB] }> = reduce_sum(wide, 2i32);
            let part: Tile<u32, { [BN] }> = reduce_sum(part, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn rows<const BN: i32, const KT: i32>(
        y: &mut Tensor<u32, { [1, BN] }>,
        w: &Tensor<u8, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT]);
        let mut acc: Tile<u32, { [BN] }> = constant(0u32, shape![BN]);
        for kc in 0i32..chunks {
            let bytes: Tile<u8, { [BN, KT] }> = wp.load([pid.1, kc]);
            let wide: Tile<u32, { [BN, KT] }> = exti(bytes);
            let part: Tile<u32, { [BN] }> = reduce_sum(wide, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
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
        return Err("usage: tma_view_probe N K BN KT".into());
    };
    if rows % bn != 0 || kt % 128 != 0 || cols % kt != 0 {
        return Err("BN must divide N, KT must be a multiple of 128 dividing K".into());
    }
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    let bytes = vec![1u8; rows * cols];
    let upload = |shape: &[usize]| -> Result<Arc<Tensor<u8>>> {
        Ok(Arc::new(
            api::copy_host_vec_to_device(&Arc::new(bytes.clone()))
                .reshape(shape)
                .sync_on(&stream)?,
        ))
    };
    let w32 = upload(&[rows, cols / 32, 32])?;
    let w128 = upload(&[rows, cols / 128, 128])?;
    let w2 = upload(&[rows, cols])?;
    let mut y = api::zeros::<u32>(&[1, rows]).sync_on(&stream)?;
    let chunks = (cols / kt) as i32;
    let options = BenchOptions::default();
    let size = rows * cols;
    let print = |variant: &str, ms: f32| {
        let gbps = size as f64 / (ms as f64 * 1e6);
        println!("{variant},{rows},{cols},{bn},{kt},{ms:.6},{gbps:.1}");
    };
    println!("view,N,K,BN,KT,ms,GBps");

    let ms = do_bench(&stream, &options, |stream| {
        views::inner_32((&mut y).partition([1, bn]), w32.clone(), chunks)
            .generics(vec![bn.to_string(), (kt / 32).to_string()])
            .sync_on(stream)
            .map(|_| ())
            .map_err(Into::into)
    })?
    .median_ms();
    print("[N,K/32,32]", ms);

    let ms = do_bench(&stream, &options, |stream| {
        views::inner_128((&mut y).partition([1, bn]), w128.clone(), chunks)
            .generics(vec![bn.to_string(), (kt / 128).to_string()])
            .sync_on(stream)
            .map(|_| ())
            .map_err(Into::into)
    })?
    .median_ms();
    print("[N,K/128,128]", ms);

    let ms = do_bench(&stream, &options, |stream| {
        views::rows((&mut y).partition([1, bn]), w2.clone(), chunks)
            .generics(vec![bn.to_string(), kt.to_string()])
            .sync_on(stream)
            .map(|_| ())
            .map_err(Into::into)
    })?
    .median_ms();
    print("[N,K]", ms);

    let expected = vec![cols as u32; rows];
    let got = y.dup().to_host_vec().sync_on(&stream)?;
    if got != expected {
        return Err("rows sum mismatch".into());
    }
    Ok(())
}
