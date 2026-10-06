use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f8e4m3fn};
use cutile::DType;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{IntoPartition, PartitionMut, Tensor, TensorViewMut};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::gemm::{DEFAULT_GEMM_TILE, GemmTile};
use half::bf16;
use std::error::Error;
use std::ops::Range;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// Prefill GEMM over a stacked weight (members stacked by rows, zero-padded at the end), run as
// one GEMM over all rows, as one GEMM per member reading a row slice of the stacked weight and
// writing a column slice of the stacked output, and per member into its own output. Each GEMM
// takes its tile from the prefill rule for its row count.
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

    // gemm_kernel as in 3c3619e.
    #[cutile::entry()]
    pub fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<bf16, { [-1, K] }>,
    ) {
        let x_tiles = x.partition(shape![BM, BK]);
        let w_tiles = w.partition(shape![BN, BK]);
        let block: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for part in 0i32..(K / BK) {
            let values: Tile<f32, { [BM, BK] }> = x_tiles.load([block.0, part]);
            let hi: Tile<bf16, { [BM, BK] }> = convert_tile(values);
            let hi_f32: Tile<f32, { [BM, BK] }> = convert_tile(hi);
            let lo: Tile<bf16, { [BM, BK] }> = convert_tile(values - hi_f32);
            let weights: Tile<bf16, { [BK, BN] }> = w_tiles.load([block.1, part]).transpose();
            acc = mma(hi, weights, acc);
            acc = mma(lo, weights, acc);
        }
        y.store(acc);
    }

    // fp8_rows_gemm_kernel as in 3c3619e.
    #[cutile::entry()]
    pub fn fp8_rows_gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
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
}

// gemm_tile in crates/firepit/src/prefill.rs.
fn prefill_tile(cols: usize, rows: usize) -> GemmTile {
    let mut tile = DEFAULT_GEMM_TILE;
    while rows % tile.n != 0 {
        tile.n /= 2;
    }
    while cols % tile.k != 0 {
        tile.k /= 2;
    }
    tile
}

fn generics(tile: GemmTile, k: usize) -> Vec<String> {
    vec![
        tile.m.to_string(),
        tile.n.to_string(),
        tile.k.to_string(),
        k.to_string(),
    ]
}

enum Weight {
    Bf16(Arc<Tensor<bf16>>),
    Fp8 {
        weights: Arc<Tensor<f8e4m3fn>>,
        row_scales: Arc<Tensor<f32>>,
    },
}

struct Stacked {
    x: Arc<Tensor<f32>>,
    weight: Weight,
    k: usize,
}

impl Stacked {
    // One GEMM of the rows `rows` of the stacked weight into `y`, which is [M, rows.len()].
    fn launch(
        &self,
        y: TensorViewMut<'_, f32>,
        rows: Range<usize>,
        stream: &Arc<Stream>,
    ) -> Result<()> {
        let tile = prefill_tile(self.k, rows.len());
        let y = y.partition([tile.m, tile.n]);
        let x = self.x.view(&[self.x.shape()[0] as usize, self.k])?;
        match &self.weight {
            Weight::Bf16(weights) => {
                let w = weights.slice(&[rows, 0..self.k])?;
                kernels::gemm(y, &x, &w)
                    .generics(generics(tile, self.k))
                    .sync_on(stream)?;
            }
            Weight::Fp8 {
                weights,
                row_scales,
            } => {
                let w = weights.slice(&[rows.clone(), 0..self.k])?;
                let scales = row_scales.slice(&[0..1, rows])?;
                kernels::fp8_rows_gemm(y, &x, &w, &scales)
                    .generics(generics(tile, self.k))
                    .sync_on(stream)?;
            }
        }
        Ok(())
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

fn device_matrix<T: DType + Send + Sync + 'static>(
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

fn columns(values: &[f32], width: usize, range: &Range<usize>) -> Vec<f32> {
    values
        .chunks(width)
        .flat_map(|row| row[range.clone()].iter().copied())
        .collect()
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let usage =
        "usage: gemm_member_slices_probe M K BALLAST_GIB bf16|fp8 STACKED_ROWS MEMBER_ROWS...";
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 6 {
        return Err(usage.into());
    }
    let numbers = |values: &[String]| -> Result<Vec<usize>> {
        Ok(values
            .iter()
            .map(|a| a.parse())
            .collect::<std::result::Result<_, _>>()?)
    };
    let [m, k, ballast_gib] = numbers(&args[0..3])?[..] else {
        return Err(usage.into());
    };
    let format = args[3].as_str();
    let stacked_rows: usize = args[4].parse()?;
    let members: Vec<Range<usize>> = numbers(&args[5..])?
        .iter()
        .scan(0, |start, &rows| {
            let range = *start..*start + rows;
            *start += rows;
            Some(range)
        })
        .collect();
    if members.last().map_or(0, |r| r.end) > stacked_rows || m % DEFAULT_GEMM_TILE.m != 0 {
        return Err("members must fit the stacked rows and M the GEMM tile".into());
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
    let x = device_matrix(
        &stream,
        (0..m * k)
            .map(|_| (rng.byte() as f32 - 128.0) / 64.0)
            .collect(),
        &[m, k],
    )?;
    let padding = members.last().map_or(0, |r| r.end)..stacked_rows;
    let mut weight_value = |row: usize| {
        let value = rng.byte();
        if padding.contains(&row) { 0 } else { value }
    };
    let weight = match format {
        "bf16" => Weight::Bf16(device_matrix(
            &stream,
            (0..stacked_rows * k)
                .map(|i| bf16::from_f32((weight_value(i / k) as f32 - 128.0) / 512.0))
                .collect(),
            &[stacked_rows, k],
        )?),
        "fp8" => Weight::Fp8 {
            weights: device_matrix(
                &stream,
                (0..stacked_rows * k)
                    .map(|i| f8e4m3fn(weight_value(i / k) & 0xbf))
                    .collect(),
                &[stacked_rows, k],
            )?,
            row_scales: device_matrix(
                &stream,
                (0..stacked_rows)
                    .map(|row| 0.25 + (row % 7) as f32 / 16.0)
                    .collect(),
                &[1, stacked_rows],
            )?,
        },
        _ => return Err(usage.into()),
    };
    let stacked = Stacked { x, weight, k };
    let shape = format!(
        "M={m} K={k} {format} stacked={stacked_rows} members={}",
        args[5..].join(",")
    );

    let mut y = api::zeros::<f32>(&[m, stacked_rows]).sync_on(&stream)?;
    probe.repeat(&format!("stacked {shape}"), |s| {
        stacked.launch(y.slice_mut(&[0..m, 0..stacked_rows])?, 0..stacked_rows, s)
    })?;
    let whole = y.dup().to_host_vec().sync_on(&stream)?;

    let mut y = api::zeros::<f32>(&[m, stacked_rows]).sync_on(&stream)?;
    probe.repeat(&format!("member_slices {shape}"), |s| {
        for rows in &members {
            stacked.launch(y.slice_mut(&[0..m, rows.clone()])?, rows.clone(), s)?;
        }
        Ok(())
    })?;
    let sliced = y.dup().to_host_vec().sync_on(&stream)?;

    let mut outputs: Vec<Tensor<f32>> = members
        .iter()
        .map(|rows| api::zeros::<f32>(&[m, rows.len()]).sync_on(&stream))
        .collect::<std::result::Result<_, _>>()?;
    probe.repeat(&format!("member_tensors {shape}"), |s| {
        for (rows, y) in members.iter().zip(&mut outputs) {
            stacked.launch(y.slice_mut(&[0..m, 0..rows.len()])?, rows.clone(), s)?;
        }
        Ok(())
    })?;

    for (rows, y) in members.iter().zip(&outputs) {
        let reference = columns(&whole, stacked_rows, rows);
        let own = y.dup().to_host_vec().sync_on(&stream)?;
        println!(
            "CHECK {shape} member {}..{} slices-vs-stacked {:.3e} tensors-vs-stacked {:.3e}",
            rows.start,
            rows.end,
            max_relative(&columns(&sliced, stacked_rows, rows), &reference),
            max_relative(&own, &reference)
        );
    }
    drop(ballast);
    Ok(())
}
