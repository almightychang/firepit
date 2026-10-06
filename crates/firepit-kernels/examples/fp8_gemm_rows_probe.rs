use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::gemm::DEFAULT_GEMM_TILE;
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// The prefill FP8 GEMM with a per-tensor scale, in the forms that differ between fc8667a and
// 49ec6a2 and in diagnostic variants: the scale on the input every iteration (the block kernel
// with one block), on the accumulator after the loop (the rows kernel), left out, applied by a
// separate kernel after the store, and on the weight tile every iteration. Same tile for all.
// The process first allocates BALLAST_GIB one-GiB buffers; each launch follows a read of a
// 64 MiB buffer; time the kernels with nsys.
const LAUNCHES: usize = 10;
const SCRUB_TILE: usize = 4096;
const SCRUB_LEN: usize = 16 << 20;

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

    // fp8_gemm_kernel as in fc8667a; per-tensor means BLOCK_N = N and BLOCK_K = K.
    #[cutile::entry()]
    pub fn input_scaled<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const K: i32,
        const BLOCK_N: i32,
        const BLOCK_K: i32,
    >(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f8e4m3fn, { [-1, K] }>,
        scales: &Tensor<f32, { [-1, -1] }>,
    ) {
        let x_tiles = x.partition(shape![BM, BK]);
        let w_tiles = w.partition(shape![BN, BK]);
        let scale_tiles = scales.partition(shape![1, 1]);
        let block: (i32, i32, i32) = get_tile_block_id();
        let scale_row = block.1 * BN / BLOCK_N;
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for part in 0i32..(K / BK) {
            let values: Tile<f32, { [BM, BK] }> = x_tiles.load([block.0, part]);
            let scale: Tile<f32, { [1, 1] }> = scale_tiles.load([scale_row, part * BK / BLOCK_K]);
            let values: Tile<f32, { [BM, BK] }> = values * scale.broadcast(shape![BM, BK]);
            let hi: Tile<bf16, { [BM, BK] }> = convert_tile(values);
            let hi_f32: Tile<f32, { [BM, BK] }> = convert_tile(hi);
            let lo: Tile<bf16, { [BM, BK] }> = convert_tile(values - hi_f32);
            let weights: Tile<f8e4m3fn, { [BN, BK] }> = w_tiles.load([block.1, part]);
            let weights: Tile<bf16, { [BN, BK] }> = convert_tile(weights);
            let weights: Tile<bf16, { [BK, BN] }> = weights.transpose();
            acc = mma(hi, weights, acc);
            acc = mma(lo, weights, acc);
        }
        y.store(acc);
    }

    // fp8_rows_gemm_kernel as in 49ec6a2: no scale in the loop, row scales on the accumulator.
    #[cutile::entry()]
    pub fn accumulator_scaled<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f8e4m3fn, { [-1, K] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
    ) {
        let block: (i32, i32, i32) = get_tile_block_id();
        let x_tiles = x.partition(shape![BM, BK]);
        let w_tiles = w.partition(shape![BN, BK]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for part in 0i32..(K / BK) {
            let values: Tile<f32, { [BM, BK] }> = x_tiles.load([block.0, part]);
            let hi: Tile<bf16, { [BM, BK] }> = convert_tile(values);
            let hi_f32: Tile<f32, { [BM, BK] }> = convert_tile(hi);
            let lo: Tile<bf16, { [BM, BK] }> = convert_tile(values - hi_f32);
            let weights: Tile<f8e4m3fn, { [BN, BK] }> = w_tiles.load([block.1, part]);
            let weights: Tile<bf16, { [BN, BK] }> = convert_tile(weights);
            let weights: Tile<bf16, { [BK, BN] }> = weights.transpose();
            acc = mma(hi, weights, acc);
            acc = mma(lo, weights, acc);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, block.1]);
        y.store(acc * row_scale.broadcast(shape![BM, BN]));
    }

    // Diagnostic: the rows kernel without the scale (wrong unless every scale is 1).
    #[cutile::entry()]
    pub fn unscaled<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f8e4m3fn, { [-1, K] }>,
    ) {
        let block: (i32, i32, i32) = get_tile_block_id();
        let x_tiles = x.partition(shape![BM, BK]);
        let w_tiles = w.partition(shape![BN, BK]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for part in 0i32..(K / BK) {
            let values: Tile<f32, { [BM, BK] }> = x_tiles.load([block.0, part]);
            let hi: Tile<bf16, { [BM, BK] }> = convert_tile(values);
            let hi_f32: Tile<f32, { [BM, BK] }> = convert_tile(hi);
            let lo: Tile<bf16, { [BM, BK] }> = convert_tile(values - hi_f32);
            let weights: Tile<f8e4m3fn, { [BN, BK] }> = w_tiles.load([block.1, part]);
            let weights: Tile<bf16, { [BN, BK] }> = convert_tile(weights);
            let weights: Tile<bf16, { [BK, BN] }> = weights.transpose();
            acc = mma(hi, weights, acc);
            acc = mma(lo, weights, acc);
        }
        y.store(acc);
    }

    // The column scale applied by a second, element-wise kernel after `unscaled`.
    #[cutile::entry()]
    pub fn scale_columns<const BM: i32, const BN: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
    ) {
        let block: (i32, i32, i32) = get_tile_block_id();
        let values: Tile<f32, { [BM, BN] }> = y.load();
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, block.1]);
        y.store(values * row_scale.broadcast(shape![BM, BN]));
    }

    // The row scales applied to the weight tile every iteration, as the grouped kernel does.
    #[cutile::entry()]
    pub fn weight_scaled<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f8e4m3fn, { [-1, K] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
    ) {
        let block: (i32, i32, i32) = get_tile_block_id();
        let x_tiles = x.partition(shape![BM, BK]);
        let w_tiles = w.partition(shape![BN, BK]);
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, block.1]);
        let scales: Tile<f32, { [BK, BN] }> = row_scale.broadcast(shape![BK, BN]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for part in 0i32..(K / BK) {
            let values: Tile<f32, { [BM, BK] }> = x_tiles.load([block.0, part]);
            let hi: Tile<bf16, { [BM, BK] }> = convert_tile(values);
            let hi_f32: Tile<f32, { [BM, BK] }> = convert_tile(hi);
            let lo: Tile<bf16, { [BM, BK] }> = convert_tile(values - hi_f32);
            let weights: Tile<f8e4m3fn, { [BN, BK] }> = w_tiles.load([block.1, part]);
            let weights: Tile<f32, { [BN, BK] }> = convert_tile(weights);
            let weights: Tile<f32, { [BK, BN] }> = weights.transpose();
            let weights: Tile<bf16, { [BK, BN] }> = convert_tile(weights * scales);
            acc = mma(hi, weights, acc);
            acc = mma(lo, weights, acc);
        }
        y.store(acc);
    }
}

struct Probe {
    stream: Arc<Stream>,
    buffer: Arc<Tensor<f32>>,
    sums: Tensor<f32>,
}

impl Probe {
    fn scrub(&mut self) -> Result<()> {
        kernels::scrub_read((&mut self.sums).partition([1]), self.buffer.clone())
            .generics(vec![SCRUB_TILE.to_string()])
            .sync_on(&self.stream)?;
        Ok(())
    }

    fn repeat(
        &mut self,
        label: &str,
        mut launch: impl FnMut(&Arc<Stream>) -> Result<()>,
    ) -> Result<()> {
        println!("RUN {label} launches={LAUNCHES}");
        for _ in 0..LAUNCHES {
            self.scrub()?;
            launch(&self.stream)?;
        }
        Ok(())
    }
}

struct Lcg(u64);

impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 56) as u8
    }
}

fn max_relative(a: &[f32], b: &[f32]) -> f32 {
    let scale = b
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()))
        .max(f32::MIN_POSITIVE);
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (p, q)| m.max((p - q).abs()))
        / scale
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse())
        .collect::<std::result::Result<_, _>>()?;
    let [m, n, k, ballast_gib] = args[..] else {
        return Err("usage: fp8_gemm_rows_probe M N K BALLAST_GIB".into());
    };
    let tile = DEFAULT_GEMM_TILE;
    if m % tile.m != 0 || n % tile.n != 0 || k % tile.k != 0 {
        return Err("M, N, K must be multiples of the GEMM tile".into());
    }
    cutile::jit_cache::enable_default()?;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    let ballast: Vec<Tensor<f32>> = (0..ballast_gib)
        .map(|_| api::zeros::<f32>(&[1 << 28]).sync_on(&stream))
        .collect::<std::result::Result<_, _>>()?;
    let mut probe = Probe {
        buffer: Arc::new(api::zeros::<f32>(&[SCRUB_LEN]).sync_on(&stream)?),
        sums: api::zeros::<f32>(&[SCRUB_LEN / SCRUB_TILE]).sync_on(&stream)?,
        stream: stream.clone(),
    };
    let mut rng = Lcg(5);
    let scale = 0.375f32;
    let x: Arc<Tensor<f32>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(
            (0..m * k)
                .map(|_| (rng.byte() as f32 - 128.0) / 64.0)
                .collect::<Vec<f32>>(),
        ))
        .reshape(&[m, k])
        .sync_on(&stream)?,
    );
    let w: Arc<Tensor<f8e4m3fn>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(
            (0..n * k)
                .map(|_| f8e4m3fn(rng.byte() & 0xbf))
                .collect::<Vec<_>>(),
        ))
        .reshape(&[n, k])
        .sync_on(&stream)?,
    );
    let block_scale: Arc<Tensor<f32>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(vec![scale]))
            .reshape(&[1, 1])
            .sync_on(&stream)?,
    );
    let row_scales: Arc<Tensor<f32>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(vec![scale; n]))
            .reshape(&[1, n])
            .sync_on(&stream)?,
    );
    let mut y = api::zeros::<f32>(&[m, n]).sync_on(&stream)?;
    let base = vec![
        tile.m.to_string(),
        tile.n.to_string(),
        tile.k.to_string(),
        k.to_string(),
    ];
    let shape = format!("M={m} N={n} K={k}");
    let mut outputs = Vec::new();

    probe.repeat(&format!("input_scaled {shape}"), |s| {
        let mut generics = base.clone();
        generics.extend([n.to_string(), k.to_string()]);
        kernels::input_scaled(
            (&mut y).partition([tile.m, tile.n]),
            x.clone(),
            w.clone(),
            block_scale.clone(),
        )
        .generics(generics)
        .sync_on(s)?;
        Ok(())
    })?;
    outputs.push(y.dup().to_host_vec().sync_on(&stream)?);
    probe.repeat(&format!("accumulator_scaled {shape}"), |s| {
        kernels::accumulator_scaled(
            (&mut y).partition([tile.m, tile.n]),
            x.clone(),
            w.clone(),
            row_scales.clone(),
        )
        .generics(base.clone())
        .sync_on(s)?;
        Ok(())
    })?;
    outputs.push(y.dup().to_host_vec().sync_on(&stream)?);
    probe.repeat(&format!("unscaled {shape}"), |s| {
        kernels::unscaled((&mut y).partition([tile.m, tile.n]), x.clone(), w.clone())
            .generics(base.clone())
            .sync_on(s)?;
        Ok(())
    })?;
    probe.repeat(&format!("unscaled+scale_columns {shape}"), |s| {
        kernels::unscaled((&mut y).partition([tile.m, tile.n]), x.clone(), w.clone())
            .generics(base.clone())
            .sync_on(s)?;
        kernels::scale_columns((&mut y).partition([tile.m, tile.n]), row_scales.clone())
            .generics(vec![tile.m.to_string(), tile.n.to_string()])
            .sync_on(s)?;
        Ok(())
    })?;
    outputs.push(y.dup().to_host_vec().sync_on(&stream)?);
    probe.repeat(&format!("weight_scaled {shape}"), |s| {
        kernels::weight_scaled(
            (&mut y).partition([tile.m, tile.n]),
            x.clone(),
            w.clone(),
            row_scales.clone(),
        )
        .generics(base.clone())
        .sync_on(s)?;
        Ok(())
    })?;
    outputs.push(y.dup().to_host_vec().sync_on(&stream)?);
    println!(
        "CHECK {shape} accumulator-vs-input {:.3e} separate-vs-input {:.3e} weight-vs-input {:.3e}",
        max_relative(&outputs[1], &outputs[0]),
        max_relative(&outputs[2], &outputs[0]),
        max_relative(&outputs[3], &outputs[0])
    );
    drop(ballast);
    Ok(())
}
