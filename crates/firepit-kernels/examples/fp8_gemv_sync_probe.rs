use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// FP8 GEMV bodies at one row tile per run, from a byte-sum floor up to the product shape:
// the input read as [1, K] or as [1, K/32, 32] reshaped per tile, with or without the
// 128-block scale loop. Each launch follows a read of a 64 MiB buffer to evict L2 with
// clean lines; time the kernels with nsys.
const LAUNCHES: usize = 10;
const SCRUB_TILE: usize = 4096;
const SCRUB_LEN: usize = 16 << 20;
const BLOCK: usize = 128;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn scrub_read<const B: i32>(
        sums: &mut Tensor<f32, { [1] }>,
        buffer: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let values: Tile<f32, { [B] }> = buffer.partition(shape![B]).load([pid.0]);
        let sum: Tile<f32, { [] }> = reduce_sum(values, 0i32);
        sums.store(sum.reshape(shape![1]));
    }

    // Sums the raw FP8 bytes: the memory floor.
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

    // Input as [1, K], no scales.
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

    // Input as [1, K/32, 32] loaded as [1, NB, 32] and reshaped to [1, KT], no scales.
    #[cutile::entry()]
    pub fn fp8_dot_blocked_input<const BN: i32, const NB: i32, const KT: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        x: &Tensor<f32, { [1, -1, 32] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT]);
        let xp = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = wp.load([pid.1, kc]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = xp.load([0, kc, 0]);
            let xb: Tile<f32, { [BN, KT] }> = xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
            let part: Tile<f32, { [BN] }> = reduce_sum(weights * xb, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    // Input as [1, K], with a [1, 1] scale per 128x128 block applied to each tile's sum.
    #[cutile::entry()]
    pub fn fp8_dot_scaled<const BN: i32, const KT: i32, const BLOCK: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        scales: &Tensor<f32, { [-1, -1] }>,
        x: &Tensor<f32, { [1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT]);
        let xp = x.partition(shape![1, KT]);
        let sp = scales.partition(shape![1, 1]);
        let scale_row = pid.1 * BN / BLOCK;
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let scale: Tile<f32, { [1, 1] }> = sp.load([scale_row, kc]);
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = wp.load([pid.1, kc]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, KT] }> = xp.load([0, kc]);
            let part: Tile<f32, { [BN] }> =
                reduce_sum(weights * xb.broadcast(shape![BN, KT]), 1i32);
            let part: Tile<f32, { [BN] }> = part.reshape(shape![BN]);
            acc = acc + part * scale.reshape(shape![1]).broadcast(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }
}

struct Probe {
    stream: Arc<Stream>,
    buffer: Arc<Tensor<f32>>,
    sums: Tensor<f32>,
}

impl Probe {
    fn repeat(
        &mut self,
        label: &str,
        mut launch: impl FnMut(&Arc<Stream>) -> Result<()>,
    ) -> Result<()> {
        println!("{label}");
        for _ in 0..LAUNCHES {
            kernels::scrub_read((&mut self.sums).partition([1]), self.buffer.clone())
                .generics(vec![SCRUB_TILE.to_string()])
                .sync_on(&self.stream)?;
            launch(&self.stream)?;
        }
        Ok(())
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
    if rows % BLOCK != 0 || rows % bn != 0 || cols % kt != 0 {
        return Err("N must be a multiple of 128 and of BN, and K of 128".into());
    }
    cutile::jit_cache::enable_default()?;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    let mut probe = Probe {
        buffer: Arc::new(api::zeros::<f32>(&[SCRUB_LEN]).sync_on(&stream)?),
        sums: api::zeros::<f32>(&[SCRUB_LEN / SCRUB_TILE]).sync_on(&stream)?,
        stream: stream.clone(),
    };
    // 0x38 is 1.0 in E4M3 and every scale is 1.0, so every output row is K.
    let w: Arc<Tensor<f8e4m3fn>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(vec![f8e4m3fn(0x38); rows * cols]))
            .reshape(&[rows, cols])
            .sync_on(&stream)?,
    );
    let scales: Arc<Tensor<f32>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; rows / BLOCK * (cols / BLOCK)]))
            .reshape(&[rows / BLOCK, cols / BLOCK])
            .sync_on(&stream)?,
    );
    let x: Arc<Tensor<f32>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
            .reshape(&[1, cols])
            .sync_on(&stream)?,
    );
    let x_blocked: Arc<Tensor<f32>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
            .reshape(&[1, cols / 32, 32])
            .sync_on(&stream)?,
    );
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let chunks = (cols / kt) as i32;
    let expected = vec![cols as f32; rows];
    let shape = format!("N={rows} K={cols} BN={bn}");

    probe.repeat(&format!("RUN fp8_load {shape}"), |s| {
        kernels::fp8_load((&mut y).partition([1, bn]), w.clone(), chunks)
            .generics(vec![bn.to_string(), kt.to_string()])
            .sync_on(s)?;
        Ok(())
    })?;
    probe.repeat(&format!("RUN fp8_dot {shape}"), |s| {
        kernels::fp8_dot((&mut y).partition([1, bn]), w.clone(), x.clone(), chunks)
            .generics(vec![bn.to_string(), kt.to_string()])
            .sync_on(s)?;
        Ok(())
    })?;
    if y.dup().to_host_vec().sync_on(&stream)? != expected {
        return Err("fp8_dot: every row should be K".into());
    }
    probe.repeat(&format!("RUN fp8_dot_blocked_input {shape}"), |s| {
        kernels::fp8_dot_blocked_input(
            (&mut y).partition([1, bn]),
            w.clone(),
            x_blocked.clone(),
            chunks,
        )
        .generics(vec![bn.to_string(), (kt / 32).to_string(), kt.to_string()])
        .sync_on(s)?;
        Ok(())
    })?;
    if y.dup().to_host_vec().sync_on(&stream)? != expected {
        return Err("fp8_dot_blocked_input: every row should be K".into());
    }
    probe.repeat(&format!("RUN fp8_dot_scaled {shape}"), |s| {
        kernels::fp8_dot_scaled(
            (&mut y).partition([1, bn]),
            w.clone(),
            scales.clone(),
            x.clone(),
            chunks,
        )
        .generics(vec![bn.to_string(), kt.to_string(), BLOCK.to_string()])
        .sync_on(s)?;
        Ok(())
    })?;
    if y.dup().to_host_vec().sync_on(&stream)? != expected {
        return Err("fp8_dot_scaled: every row should be K".into());
    }
    Ok(())
}
