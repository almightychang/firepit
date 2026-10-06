use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::gemm::DEFAULT_GEMM_TILE;
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// Whether the rows FP8 GEMM main loop is slower than the per-tensor one because of its shape:
// the rows kernel as in 49ec6a2, the same body handling two K parts per iteration, and the same
// body multiplying the input by a [1, 1] value loaded every iteration (1.0, as the per-tensor
// kernel loads its scale). The per-tensor kernel of fc8667a is the reference.
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
    pub fn per_tensor<
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

    // fp8_rows_gemm_kernel as in 49ec6a2.
    #[cutile::entry()]
    pub fn rows<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
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

    // The rows kernel with two K parts per iteration; K / BK must be even.
    #[cutile::entry()]
    pub fn rows_two_parts<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f8e4m3fn, { [-1, K] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
    ) {
        let block: (i32, i32, i32) = get_tile_block_id();
        let x_tiles = x.partition(shape![BM, BK]);
        let w_tiles = w.partition(shape![BN, BK]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for pair in 0i32..(K / BK / 2) {
            let first = 2i32 * pair;
            let values: Tile<f32, { [BM, BK] }> = x_tiles.load([block.0, first]);
            let hi: Tile<bf16, { [BM, BK] }> = convert_tile(values);
            let hi_f32: Tile<f32, { [BM, BK] }> = convert_tile(hi);
            let lo: Tile<bf16, { [BM, BK] }> = convert_tile(values - hi_f32);
            let weights: Tile<f8e4m3fn, { [BN, BK] }> = w_tiles.load([block.1, first]);
            let weights: Tile<bf16, { [BN, BK] }> = convert_tile(weights);
            let weights: Tile<bf16, { [BK, BN] }> = weights.transpose();
            acc = mma(hi, weights, acc);
            acc = mma(lo, weights, acc);
            let second = first + 1i32;
            let values: Tile<f32, { [BM, BK] }> = x_tiles.load([block.0, second]);
            let hi: Tile<bf16, { [BM, BK] }> = convert_tile(values);
            let hi_f32: Tile<f32, { [BM, BK] }> = convert_tile(hi);
            let lo: Tile<bf16, { [BM, BK] }> = convert_tile(values - hi_f32);
            let weights: Tile<f8e4m3fn, { [BN, BK] }> = w_tiles.load([block.1, second]);
            let weights: Tile<bf16, { [BN, BK] }> = convert_tile(weights);
            let weights: Tile<bf16, { [BK, BN] }> = weights.transpose();
            acc = mma(hi, weights, acc);
            acc = mma(lo, weights, acc);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, block.1]);
        y.store(acc * row_scale.broadcast(shape![BM, BN]));
    }

    // The rows kernel multiplying the input by a [1, 1] value loaded every iteration.
    #[cutile::entry()]
    pub fn rows_unit_load<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f8e4m3fn, { [-1, K] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        unit: &Tensor<f32, { [-1, -1] }>,
    ) {
        let block: (i32, i32, i32) = get_tile_block_id();
        let x_tiles = x.partition(shape![BM, BK]);
        let w_tiles = w.partition(shape![BN, BK]);
        let unit_tiles = unit.partition(shape![1, 1]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for part in 0i32..(K / BK) {
            let values: Tile<f32, { [BM, BK] }> = x_tiles.load([block.0, part]);
            let one: Tile<f32, { [1, 1] }> = unit_tiles.load([0i32, 0i32]);
            let values: Tile<f32, { [BM, BK] }> = values * one.broadcast(shape![BM, BK]);
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

fn device_matrix<T: cutile::DType + Send + Sync + 'static>(
    stream: &Arc<Stream>,
    values: Vec<T>,
    shape: &[usize],
) -> Result<Arc<Tensor<T>>> {
    Ok(Arc::new(
        api::copy_host_vec_to_device(&Arc::new(values))
            .reshape(shape)
            .sync_on(stream)?,
    ))
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
        return Err("usage: fp8_gemm_unroll_probe M N K BALLAST_GIB".into());
    };
    let tile = DEFAULT_GEMM_TILE;
    if m % tile.m != 0 || n % tile.n != 0 || k % (2 * tile.k) != 0 {
        return Err("M and N must be multiples of the GEMM tile, K of two K tiles".into());
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
    let x = device_matrix(
        &stream,
        (0..m * k)
            .map(|_| (rng.byte() as f32 - 128.0) / 64.0)
            .collect(),
        &[m, k],
    )?;
    let w = device_matrix(
        &stream,
        (0..n * k).map(|_| f8e4m3fn(rng.byte() & 0xbf)).collect(),
        &[n, k],
    )?;
    let block_scale = device_matrix(&stream, vec![scale], &[1, 1])?;
    let unit = device_matrix(&stream, vec![1.0f32], &[1, 1])?;
    let row_scales = device_matrix(&stream, vec![scale; n], &[1, n])?;
    let mut y = api::zeros::<f32>(&[m, n]).sync_on(&stream)?;
    let base = vec![
        tile.m.to_string(),
        tile.n.to_string(),
        tile.k.to_string(),
        k.to_string(),
    ];
    let shape = format!("M={m} N={n} K={k}");
    let mut outputs = Vec::new();

    probe.repeat(&format!("per_tensor {shape}"), |s| {
        let mut generics = base.clone();
        generics.extend([n.to_string(), k.to_string()]);
        kernels::per_tensor(
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
    probe.repeat(&format!("rows {shape}"), |s| {
        kernels::rows(
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
    probe.repeat(&format!("rows_two_parts {shape}"), |s| {
        kernels::rows_two_parts(
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
    probe.repeat(&format!("rows_unit_load {shape}"), |s| {
        kernels::rows_unit_load(
            (&mut y).partition([tile.m, tile.n]),
            x.clone(),
            w.clone(),
            row_scales.clone(),
            unit.clone(),
        )
        .generics(base.clone())
        .sync_on(s)?;
        Ok(())
    })?;
    outputs.push(y.dup().to_host_vec().sync_on(&stream)?);
    println!(
        "CHECK {shape} rows-vs-per_tensor {:.3e} two_parts-vs-rows {:.3e} unit_load-vs-rows {:.3e}",
        max_relative(&outputs[1], &outputs[0]),
        max_relative(&outputs[2], &outputs[1]),
        max_relative(&outputs[3], &outputs[1])
    );
    drop(ballast);
    Ok(())
}
