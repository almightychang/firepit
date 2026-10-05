use cuda_async::device_operation::DeviceOp;
use cuda_async::scheduling_policies::{SchedulingPolicy, SingleStream};
use cuda_core::{Device, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::gemv::{DEFAULT_GEMV_TILE, GemvTile, fp8_tile, gemv, nvfp4_tile};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{DenseMatrix, HostFp8Matrix, HostNvfp4Matrix, Weight};
use half::bf16;
use std::error::Error;
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// Every launch is preceded by a read of a buffer twice the L2 size, so each GEMV reads its
// weights from DRAM into an L2 of clean lines, as in graph decode. Kernel times are read from nsys, not from events;
// the manifest lines tell the analysis which configuration each instance belongs to.
const WARMUP: usize = 2;
const REPS: usize = 8;
const SCRUB_TILE: usize = 4096;
const SCALE_BLOCK: usize = 128;

#[cutile::module]
mod probes {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn scrub_write<const B: i32>(buffer: &mut Tensor<f32, { [B] }>) {
        let zeros: Tile<f32, { [B] }> = constant(0.0f32, shape![B]);
        buffer.store(zeros);
    }

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

    #[cutile::entry()]
    pub fn rows_load<const BN: i32, const KT: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT]);
        let mut acc: Tile<u32, { [BN] }> = constant(0u32, shape![BN]);
        for kc in 0i32..chunks {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = wp.load([pid.1, kc]);
            let bits: Tile<u8, { [BN, KT] }> = bitcast(weights);
            let wide: Tile<u32, { [BN, KT] }> = exti(bits);
            let part: Tile<u32, { [BN] }> = reduce_sum(wide, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        let out: Tile<f32, { [BN] }> = bitcast(acc);
        y.store(out.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn rows_load_pipelined<const BN: i32, const KT: i32, const LATENCY: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT]);
        let mut acc: Tile<u32, { [BN] }> = constant(0u32, shape![BN]);
        for kc in 0i32..chunks {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = wp.load_pipelined::<LATENCY>([pid.1, kc]);
            let bits: Tile<u8, { [BN, KT] }> = bitcast(weights);
            let wide: Tile<u32, { [BN, KT] }> = exti(bits);
            let part: Tile<u32, { [BN] }> = reduce_sum(wide, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        let out: Tile<f32, { [BN] }> = bitcast(acc);
        y.store(out.reshape(shape![1, BN]));
    }
}

struct Weights {
    fp8: Weight,
    bf16: Weight,
    nvfp4: Weight,
    fp8_rows: Arc<Tensor<f8e4m3fn>>,
}

impl Weights {
    fn upload(stream: &Arc<Stream>, rows: usize, cols: usize) -> Result<Self> {
        let fp8_bytes = vec![f8e4m3fn(0x38); rows * cols];
        let scales = vec![1.0f32; rows.div_ceil(SCALE_BLOCK) * cols.div_ceil(SCALE_BLOCK)];
        Ok(Self {
            fp8: Weight::Fp8(
                HostFp8Matrix::from_parts(
                    fp8_bytes.clone(),
                    scales,
                    rows,
                    cols,
                    SCALE_BLOCK,
                    SCALE_BLOCK,
                )?
                .upload(stream)?,
            ),
            bf16: Weight::Dense(DenseMatrix::upload(
                vec![bf16::from_f32(1.0); rows * cols],
                rows,
                cols,
                stream,
            )?),
            nvfp4: Weight::Nvfp4(
                HostNvfp4Matrix::from_parts(
                    vec![f4e2m1fnx2::from_bits(0x22); rows * cols / 2],
                    vec![f8e4m3fn(0x38); rows * cols / 16],
                    1.0,
                    rows,
                    cols,
                )?
                .upload(stream)?,
            ),
            fp8_rows: Arc::new(
                api::copy_host_vec_to_device(&Arc::new(fp8_bytes))
                    .reshape(&[rows, cols])
                    .sync_on(stream)?,
            ),
        })
    }
}

fn dense_tile(rows: usize, cols: usize) -> GemvTile {
    let mut tile = DEFAULT_GEMV_TILE;
    while rows % tile.rows != 0 {
        tile.rows /= 2;
    }
    while (cols / 32) % tile.blocks != 0 {
        tile.blocks /= 2;
    }
    tile
}

#[derive(Clone, Copy, PartialEq)]
enum Scrub {
    Read,
    Write,
}

struct Probe {
    stream: Arc<Stream>,
    buffer: Arc<Tensor<f32>>,
    sums: Tensor<f32>,
    mode: Scrub,
}

impl Probe {
    fn scrub(&mut self) -> Result<()> {
        match self.mode {
            Scrub::Read => {
                probes::scrub_read((&mut self.sums).partition([1]), self.buffer.clone())
                    .generics(vec![SCRUB_TILE.to_string()])
                    .sync_on(&self.stream)?;
            }
            Scrub::Write => {
                let mut buffer =
                    api::zeros::<f32>(&[self.buffer.shape()[0] as usize]).sync_on(&self.stream)?;
                probes::scrub_write((&mut buffer).partition([SCRUB_TILE]))
                    .generics(vec![SCRUB_TILE.to_string()])
                    .sync_on(&self.stream)?;
            }
        }
        Ok(())
    }

    fn repeat(
        &mut self,
        label: &str,
        mut launch: impl FnMut(&Arc<Stream>) -> Result<()>,
    ) -> Result<()> {
        println!("CFG {label} launches={}", WARMUP + REPS);
        for _ in 0..WARMUP + REPS {
            self.scrub()?;
            launch(&self.stream)?;
        }
        Ok(())
    }
}

fn product_gemv(
    weight: &Weight,
    tile: GemvTile,
    rows: usize,
    cols: usize,
    stream: &Arc<Stream>,
) -> Result<impl FnMut(&Arc<Stream>) -> Result<()> + use<>> {
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
        .reshape(&[1, cols / 32, 32])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(stream)?;
    let weight = weight.clone();
    Ok(move |stream: &Arc<Stream>| -> Result<()> {
        gemv(
            &weight,
            &x.view(&[1, cols / 32, 32])?,
            (&mut y).partition([1, tile.rows]),
            tile,
        )
        .submit(&mut Eager(stream))?;
        Ok(())
    })
}

fn sweep(probe: &mut Probe, rows: usize, cols: usize) -> Result<()> {
    let w = Weights::upload(&probe.stream, rows, cols)?;
    let stream = probe.stream.clone();
    let fp8 = fp8_tile(rows, cols, SCALE_BLOCK, SCALE_BLOCK, 1);
    let bf = dense_tile(rows, cols);
    let nv = nvfp4_tile(rows, cols, 1);
    let shape = format!("N={rows} K={cols}");
    probe.repeat(
        &format!("sweep fp8 {shape} BN={} KT={}", fp8.rows, fp8.blocks * 32),
        product_gemv(&w.fp8, fp8, rows, cols, &stream)?,
    )?;
    probe.repeat(
        &format!("sweep bf16 {shape} BN={} KT={}", bf.rows, bf.blocks * 32),
        product_gemv(&w.bf16, bf, rows, cols, &stream)?,
    )?;
    probe.repeat(
        &format!("sweep nvfp4 {shape} BN={} KT={}", nv.rows, nv.blocks * 32),
        product_gemv(&w.nvfp4, nv, rows, cols, &stream)?,
    )?;
    let kt = fp8.blocks * 32;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    probe.repeat(
        &format!("sweep rows {shape} BN={} KT={kt}", fp8.rows),
        |stream| {
            probes::rows_load(
                (&mut y).partition([1, fp8.rows]),
                w.fp8_rows.clone(),
                (cols / kt) as i32,
            )
            .generics(vec![fp8.rows.to_string(), kt.to_string()])
            .sync_on(stream)?;
            Ok(())
        },
    )?;
    Ok(())
}

fn pipeline(probe: &mut Probe, rows: usize, cols: usize) -> Result<()> {
    let w = Weights::upload(&probe.stream, rows, cols)?;
    let tile = fp8_tile(rows, cols, SCALE_BLOCK, SCALE_BLOCK, 1);
    let kt = tile.blocks * 32;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&probe.stream)?;
    for latency in [2, 4, 8] {
        probe.repeat(
            &format!(
                "pipeline rows N={rows} K={cols} BN={} KT={kt} LATENCY={latency}",
                tile.rows
            ),
            |stream| {
                probes::rows_load_pipelined(
                    (&mut y).partition([1, tile.rows]),
                    w.fp8_rows.clone(),
                    (cols / kt) as i32,
                )
                .generics(vec![
                    tile.rows.to_string(),
                    kt.to_string(),
                    latency.to_string(),
                ])
                .sync_on(stream)?;
                Ok(())
            },
        )?;
    }
    Ok(())
}

// Two independent load-only GEMV stand-ins (the same bytes as the product kernels) on
// (a) one stream back to back, and (b) two streams: the first is scheduled on its own stream
// and polled once, which launches it without waiting, then the second runs on the probe
// stream. The analysis reads start skew and overlap from nsys.
fn rows_op<'a>(
    y: &'a mut Tensor<f32>,
    w: &Arc<Tensor<f8e4m3fn>>,
    rows: usize,
    cols: usize,
) -> impl DeviceOp<Output = ()> + use<'a> {
    let tile = fp8_tile(rows, cols, SCALE_BLOCK, SCALE_BLOCK, 1);
    let kt = tile.blocks * 32;
    probes::rows_load(y.partition([1, tile.rows]), w.clone(), (cols / kt) as i32)
        .generics(vec![tile.rows.to_string(), kt.to_string()])
        .then(|_| cuda_async::device_operation::value(()))
}

fn concurrent(
    probe: &mut Probe,
    label: &str,
    first: usize,
    second: usize,
    cols: usize,
) -> Result<()> {
    let policy: Arc<dyn SchedulingPolicy> = Arc::new(SingleStream::new(probe.stream.device())?);
    let upload = |rows: usize| -> Result<Arc<Tensor<f8e4m3fn>>> {
        Ok(Arc::new(
            api::copy_host_vec_to_device(&Arc::new(vec![f8e4m3fn(0x38); rows * cols]))
                .reshape(&[rows, cols])
                .sync_on(&probe.stream)?,
        ))
    };
    let (wa, wb) = (upload(first)?, upload(second)?);
    let mut ya = api::zeros::<f32>(&[1, first]).sync_on(&probe.stream)?;
    let mut yb = api::zeros::<f32>(&[1, second]).sync_on(&probe.stream)?;
    println!("CFG concurrent-serial {label} launches={}", WARMUP + REPS);
    for _ in 0..WARMUP + REPS {
        probe.scrub()?;
        rows_op(&mut ya, &wa, first, cols)
            .then(|_| rows_op(&mut yb, &wb, second, cols))
            .sync_on(&probe.stream)?;
    }
    println!("CFG concurrent-parallel {label} launches={}", WARMUP + REPS);
    let mut context = Context::from_waker(Waker::noop());
    for _ in 0..WARMUP + REPS {
        probe.scrub()?;
        let mut pending = Box::pin(rows_op(&mut ya, &wa, first, cols).schedule(&policy)?);
        let mut done = pending.as_mut().poll(&mut context);
        rows_op(&mut yb, &wb, second, cols).sync_on(&probe.stream)?;
        while let Poll::Pending = done {
            done = pending.as_mut().poll(&mut context);
        }
        if let Poll::Ready(result) = done {
            result?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    cutile::jit_cache::enable_default()?;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    let l2 = stream.device().l2_cache_size_bytes().unwrap_or(0);
    let scrub_len = ((2 * l2).max(64 << 20) / 4).next_multiple_of(SCRUB_TILE);
    println!("L2 {l2} bytes, scrub {} MiB", scrub_len * 4 >> 20);
    let mut probe = Probe {
        buffer: Arc::new(api::zeros::<f32>(&[scrub_len]).sync_on(&stream)?),
        sums: api::zeros::<f32>(&[scrub_len / SCRUB_TILE]).sync_on(&stream)?,
        stream,
        mode: Scrub::Write,
    };
    println!("SCRUB write");
    sweep(&mut probe, 2048, 4096)?;
    probe.mode = Scrub::Read;
    println!("SCRUB read");
    for rows in [512, 2048] {
        for cols in [128, 256, 512, 1024, 2048, 4096, 8192] {
            sweep(&mut probe, rows, cols)?;
        }
    }
    for cols in [128, 2048, 8192] {
        pipeline(&mut probe, 512, cols)?;
    }
    concurrent(
        &mut probe,
        "rows 4096x2048 + rows 512x2048",
        4096,
        512,
        2048,
    )?;
    concurrent(&mut probe, "rows 32x2048 + rows 32x2048", 32, 32, 2048)?;
    Ok(())
}
