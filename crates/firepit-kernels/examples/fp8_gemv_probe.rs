use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::gemv::{GemvTile, fp8_tile, gemv};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{DenseMatrix, HostFp8Matrix, HostNvfp4Matrix, Weight};
use half::bf16;
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const SCALE_BLOCK: usize = 128;

// Load-only probes widen the raw bits to u32 and sum them, so every loaded byte reaches the
// stored result without any float conversion; the sum is bitcast to f32 only to be stored.
#[cutile::module]
mod probes {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn fp8_bits<const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, NB, 32]);
        let mut acc: Tile<u32, { [BN] }> = constant(0u32, shape![BN]);
        for kc in 0i32..(blocks / NB) {
            let weights: Tile<f8e4m3fn, { [BN, NB, 32] }> = wp.load([pid.1, kc, 0]);
            let bits: Tile<u8, { [BN, NB, 32] }> = bitcast(weights);
            let wide: Tile<u32, { [BN, NB, 32] }> = exti(bits);
            let part: Tile<u32, { [BN, NB] }> = reduce_sum(wide, 2i32);
            let part: Tile<u32, { [BN] }> = reduce_sum(part, 1i32);
            let part: Tile<u32, { [BN] }> = part.reshape(shape![BN]);
            acc = acc + part;
        }
        let out: Tile<f32, { [BN] }> = bitcast(acc);
        y.store(out.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn fp8_sum<const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, NB, 32]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..(blocks / NB) {
            let weights: Tile<f8e4m3fn, { [BN, NB, 32] }> = wp.load([pid.1, kc, 0]);
            let weights: Tile<f32, { [BN, NB, 32] }> = convert_tile(weights);
            let part: Tile<f32, { [BN, NB] }> = reduce_sum(weights, 2i32);
            let part: Tile<f32, { [BN] }> = reduce_sum(part, 1i32);
            let part: Tile<f32, { [BN] }> = part.reshape(shape![BN]);
            acc = acc + part;
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn fp8_via_bf16_sum<const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, NB, 32]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..(blocks / NB) {
            let weights: Tile<f8e4m3fn, { [BN, NB, 32] }> = wp.load([pid.1, kc, 0]);
            let weights: Tile<bf16, { [BN, NB, 32] }> = convert_tile(weights);
            let weights: Tile<f32, { [BN, NB, 32] }> = convert_tile(weights);
            let part: Tile<f32, { [BN, NB] }> = reduce_sum(weights, 2i32);
            let part: Tile<f32, { [BN] }> = reduce_sum(part, 1i32);
            let part: Tile<f32, { [BN] }> = part.reshape(shape![BN]);
            acc = acc + part;
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn bf16_bits<const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<bf16, { [-1, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, NB, 32]);
        let mut acc: Tile<u32, { [BN] }> = constant(0u32, shape![BN]);
        for kc in 0i32..(blocks / NB) {
            let weights: Tile<bf16, { [BN, NB, 32] }> = wp.load([pid.1, kc, 0]);
            let bits: Tile<u16, { [BN, NB, 32] }> = bitcast(weights);
            let wide: Tile<u32, { [BN, NB, 32] }> = exti(bits);
            let part: Tile<u32, { [BN, NB] }> = reduce_sum(wide, 2i32);
            let part: Tile<u32, { [BN] }> = reduce_sum(part, 1i32);
            let part: Tile<u32, { [BN] }> = part.reshape(shape![BN]);
            acc = acc + part;
        }
        let out: Tile<f32, { [BN] }> = bitcast(acc);
        y.store(out.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn bf16_sum<const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<bf16, { [-1, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, NB, 32]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..(blocks / NB) {
            let weights: Tile<bf16, { [BN, NB, 32] }> = wp.load([pid.1, kc, 0]);
            let weights: Tile<f32, { [BN, NB, 32] }> = convert_tile(weights);
            let part: Tile<f32, { [BN, NB] }> = reduce_sum(weights, 2i32);
            let part: Tile<f32, { [BN] }> = reduce_sum(part, 1i32);
            let part: Tile<f32, { [BN] }> = part.reshape(shape![BN]);
            acc = acc + part;
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    // FP8 GEMV whose K tile is two 128-element scale blocks, so each load moves as many bytes
    // as the bf16 GEMV tile with the same rows and NB.
    #[cutile::entry()]
    pub fn fp8_wide_gemv<const BN: i32, const NB: i32, const NB2: i32, const BLOCK_N: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1, 32] }>,
        scales: &Tensor<f32, { [-1, -1] }>,
        x: &Tensor<f32, { [1, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, NB2, 32]);
        let xp = x.partition(shape![1, NB2, 32]);
        let sp = scales.partition(shape![1, 2]);
        let scale_row = pid.1 * BN / BLOCK_N;
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..(blocks / NB2) {
            let weights: Tile<f8e4m3fn, { [BN, NB2, 32] }> = wp.load([pid.1, kc, 0]);
            let weights: Tile<f32, { [BN, NB2, 32] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB2, 32] }> = xp.load([0, kc, 0]);
            let xb: Tile<f32, { [BN, NB2, 32] }> = xb.broadcast(shape![BN, NB2, 32]);
            let products: Tile<f32, { [BN, NB2] }> = reduce_sum(weights * xb, 2i32);
            let halves: Tile<f32, { [BN, 2, NB] }> = products.reshape(shape![BN, 2, NB]);
            let halves: Tile<f32, { [BN, 2] }> = reduce_sum(halves, 2i32);
            let halves: Tile<f32, { [BN, 2] }> = halves.reshape(shape![BN, 2]);
            let scale: Tile<f32, { [1, 2] }> = sp.load([scale_row, kc]);
            let scaled: Tile<f32, { [BN, 2] }> = halves * scale.broadcast(shape![BN, 2]);
            let sum: Tile<f32, { [BN] }> = reduce_sum(scaled, 1i32);
            acc = acc + sum.reshape(shape![1, BN]);
        }
        y.store(acc);
    }

    #[cutile::entry()]
    pub fn nvfp4_bits<const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<u8, { [-1, -1, 16] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, NB, 16]);
        let mut acc: Tile<u32, { [BN] }> = constant(0u32, shape![BN]);
        for kc in 0i32..(blocks / NB) {
            let bits: Tile<u8, { [BN, NB, 16] }> = wp.load([pid.1, kc, 0]);
            let wide: Tile<u32, { [BN, NB, 16] }> = exti(bits);
            let part: Tile<u32, { [BN, NB] }> = reduce_sum(wide, 2i32);
            let part: Tile<u32, { [BN] }> = reduce_sum(part, 1i32);
            let part: Tile<u32, { [BN] }> = part.reshape(shape![BN]);
            acc = acc + part;
        }
        let out: Tile<f32, { [BN] }> = bitcast(acc);
        y.store(out.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn nvfp4_sum<const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1, 16] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, NB, 16]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..(blocks / NB) {
            let packed: Tile<f4e2m1fnx2, { [BN, NB, 16] }> = wp.load([pid.1, kc, 0]);
            let weights: Tile<f4e2m1fn, { [BN, NB, 32] }> = packed.unpack(shape![BN, NB, 32]);
            let weights: Tile<f32, { [BN, NB, 32] }> = convert_tile(weights);
            let part: Tile<f32, { [BN, NB] }> = reduce_sum(weights, 2i32);
            let part: Tile<f32, { [BN] }> = reduce_sum(part, 1i32);
            let part: Tile<f32, { [BN] }> = part.reshape(shape![BN]);
            acc = acc + part;
        }
        y.store(acc.reshape(shape![1, BN]));
    }
}

fn parse_args() -> Result<(usize, usize, GemvTile)> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse())
        .collect::<std::result::Result<_, _>>()?;
    let (rows, cols, tile) = match args[..] {
        [rows, cols] => (
            rows,
            cols,
            fp8_tile(rows, cols, SCALE_BLOCK, SCALE_BLOCK, 1),
        ),
        [rows, cols, tile_rows, blocks] => (
            rows,
            cols,
            GemvTile {
                rows: tile_rows,
                blocks,
            },
        ),
        _ => return Err("usage: fp8_gemv_probe N K [TILE_ROWS NB]".into()),
    };
    if rows % SCALE_BLOCK != 0
        || cols % SCALE_BLOCK != 0
        || !tile.rows.is_power_of_two()
        || !tile.blocks.is_power_of_two()
        || !tile.fits(rows, cols)
        || SCALE_BLOCK % tile.rows != 0
        || SCALE_BLOCK % (tile.blocks * 32) != 0
    {
        return Err(format!(
            "N and K must be multiples of {SCALE_BLOCK}, and the tile must fit the shape and the \
             {SCALE_BLOCK}x{SCALE_BLOCK} scale blocks"
        )
        .into());
    }
    Ok((rows, cols, tile))
}

fn print_row(variant: &str, rows: usize, cols: usize, tile: GemvTile, bytes: usize, ms: f32) {
    let ms = ms as f64;
    let gbps = bytes as f64 / (ms * 1e6);
    println!(
        "{variant},{rows},{cols},{},{},{ms:.6},{gbps:.1}",
        tile.rows, tile.blocks
    );
}

fn upload<T: cutile::DType>(
    stream: &Arc<Stream>,
    data: Vec<T>,
    shape: &[usize],
) -> Result<Arc<Tensor<T>>> {
    Ok(Arc::new(
        api::copy_host_vec_to_device(&Arc::new(data))
            .reshape(shape)
            .sync_on(stream)?,
    ))
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    cutile::jit_cache::enable_default()?;
    let (rows, cols, tile) = parse_args()?;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    let blocks = cols / 32;
    let fp8_bytes = vec![f8e4m3fn(0x38); rows * cols];
    let bf16_values = vec![bf16::from_f32(1.0); rows * cols];
    let nvfp4_bytes = vec![0x22u8; rows * cols / 2];
    let scales = vec![1.0f32; rows / SCALE_BLOCK * (cols / SCALE_BLOCK)];

    let fp8 = Weight::Fp8(
        HostFp8Matrix::from_parts(
            fp8_bytes.clone(),
            scales.clone(),
            rows,
            cols,
            SCALE_BLOCK,
            SCALE_BLOCK,
        )?
        .upload(&stream)?,
    );
    let dense = Weight::Dense(DenseMatrix::upload(
        bf16_values.clone(),
        rows,
        cols,
        &stream,
    )?);
    let nvfp4 = Weight::Nvfp4(
        HostNvfp4Matrix::from_parts(
            nvfp4_bytes
                .iter()
                .map(|&byte| f4e2m1fnx2::from_bits(byte))
                .collect(),
            vec![f8e4m3fn(0x38); rows * cols / 16],
            1.0,
            rows,
            cols,
        )?
        .upload(&stream)?,
    );
    let fw = upload(&stream, fp8_bytes, &[rows, blocks, 32])?;
    let bw = upload(&stream, bf16_values, &[rows, blocks, 32])?;
    let nw_bits = upload(&stream, nvfp4_bytes.clone(), &[rows, blocks, 16])?;
    let nw = upload(
        &stream,
        nvfp4_bytes.into_iter().map(f4e2m1fnx2::from_bits).collect(),
        &[rows, blocks, 16],
    )?;
    let sw = upload(&stream, scales, &[rows / SCALE_BLOCK, cols / SCALE_BLOCK])?;
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
        .reshape(&[1, blocks, 32])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let (fp8_size, bf16_size, nvfp4_size) = (rows * cols, 2 * rows * cols, rows * cols / 2);
    let generics = vec![tile.rows.to_string(), tile.blocks.to_string()];
    let options = BenchOptions::default();
    println!("variant,N,K,rows,NB,ms,GBps");

    for (variant, weight, size) in [
        ("P0", &fp8, fp8_size),
        ("P0b", &dense, bf16_size),
        ("P0c", &nvfp4, nvfp4_size),
    ] {
        let ms = do_bench(&stream, &options, |stream| {
            gemv(
                weight,
                &x.view(&[1, blocks, 32])?,
                (&mut y).partition([1, tile.rows]),
                tile,
            )
            .submit(&mut Eager(stream))
            .map_err(Into::into)
        })?
        .median_ms();
        print_row(variant, rows, cols, tile, size, ms);
    }

    macro_rules! probe {
        ($variant:literal, $kernel:ident, $weights:expr, $size:expr) => {{
            let ms = do_bench(&stream, &options, |stream| {
                probes::$kernel(
                    (&mut y).partition([1, tile.rows]),
                    $weights.clone(),
                    blocks as i32,
                )
                .generics(generics.clone())
                .sync_on(stream)
                .map(|_| ())
                .map_err(Into::into)
            })?
            .median_ms();
            print_row($variant, rows, cols, tile, $size, ms);
        }};
    }
    probe!("P1", fp8_bits, fw, fp8_size);
    probe!("P2", fp8_sum, fw, fp8_size);
    probe!("P3", fp8_via_bf16_sum, fw, fp8_size);
    probe!("P4", bf16_bits, bw, bf16_size);
    probe!("P5", bf16_sum, bw, bf16_size);
    probe!("N1", nvfp4_bits, nw_bits, nvfp4_size);
    probe!("N2", nvfp4_sum, nw, nvfp4_size);

    let wide = GemvTile {
        rows: tile.rows,
        blocks: 2 * tile.blocks,
    };
    if wide.blocks * 32 == 2 * SCALE_BLOCK && wide.fits(rows, cols) {
        let ms = do_bench(&stream, &options, |stream| {
            probes::fp8_wide_gemv(
                (&mut y).partition([1, tile.rows]),
                fw.clone(),
                sw.clone(),
                &x.view(&[1, blocks, 32])?,
                blocks as i32,
            )
            .generics(vec![
                tile.rows.to_string(),
                tile.blocks.to_string(),
                wide.blocks.to_string(),
                SCALE_BLOCK.to_string(),
            ])
            .sync_on(stream)
            .map(|_| ())
            .map_err(Into::into)
        })?
        .median_ms();
        print_row("P6", rows, cols, wide, fp8_size, ms);
    } else {
        eprintln!(
            "P6 skipped: it needs NB*32 = {SCALE_BLOCK} so the wide tile spans two scale blocks"
        );
    }
    Ok(())
}
