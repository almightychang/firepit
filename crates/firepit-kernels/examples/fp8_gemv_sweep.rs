use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::tensor::{IntoPartition, PartitionMut};
use firepit_kernels::gemv::{
    DEFAULT_GEMV_TILE, GemvTile, expert_gemv_view, expert_gemv_weight_view, gemv,
};
use firepit_kernels::moe::{ExpertSelection, MoeDims};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{DenseMatrix, HostFp8Matrix, Weight};
use half::bf16;
use std::error::Error;
use std::sync::Arc;

fn dense(
    stream: &Arc<Stream>,
    rows: usize,
    cols: usize,
    only: Option<GemvTile>,
) -> Result<(), Box<dyn Error>> {
    let fp8 = Weight::Fp8(
        HostFp8Matrix::from_parts(
            vec![f8e4m3fn(0x38); rows * cols],
            vec![1.0; rows / 128 * (cols / 128)],
            rows,
            cols,
            128,
            128,
        )?
        .upload(stream)?,
    );
    let bf16 = Weight::Dense(DenseMatrix::upload(
        vec![bf16::from_f32(1.0); rows * cols],
        rows,
        cols,
        stream,
    )?);
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
        .reshape(&[1, cols / 32, 32])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(stream)?;
    let reference = do_bench(stream, &BenchOptions::default(), |stream| {
        gemv(
            &bf16,
            &x.view(&[1, cols / 32, 32])?,
            (&mut y).partition([1, DEFAULT_GEMV_TILE.rows]),
            DEFAULT_GEMV_TILE,
        )
        .submit(&mut Eager(stream))
        .map_err(Into::into)
    })?
    .median_ms();
    for tile_rows in [4, 8, 16, 32, 64, 128] {
        for blocks in [1, 2, 4] {
            if only.is_some_and(|tile| tile.rows != tile_rows || tile.blocks != blocks) {
                continue;
            }
            let tile = GemvTile {
                rows: tile_rows,
                blocks,
            };
            let ms = do_bench(stream, &BenchOptions::default(), |stream| {
                gemv(
                    &fp8,
                    &x.view(&[1, cols / 32, 32])?,
                    (&mut y).partition([1, tile.rows]),
                    tile,
                )
                .submit(&mut Eager(stream))
                .map_err(Into::into)
            })?
            .median_ms();
            println!("dense,{rows},{cols},{tile_rows},{blocks},{ms:.6},{reference:.6}");
        }
    }
    Ok(())
}

fn expert(
    stream: &Arc<Stream>,
    rows: usize,
    cols: usize,
    experts: usize,
    input_stride: usize,
    only: Option<GemvTile>,
) -> Result<(), Box<dyn Error>> {
    if input_stride > 1 {
        return Err("INPUT_STRIDE must be 0 or 1".into());
    }
    let dims = MoeDims::new(cols, cols, cols, experts, experts)?;
    let selection = ExpertSelection::parse(
        &(0..experts as i32).collect::<Vec<_>>(),
        &vec![1.0; experts],
        &dims,
        stream,
    )?;
    let total = experts * rows;
    let fp8 = Weight::Fp8(
        HostFp8Matrix::from_parts(
            vec![f8e4m3fn(0x38); total * cols],
            vec![1.0; total / 128 * (cols / 128)],
            total,
            cols,
            128,
            128,
        )?
        .upload(stream)?,
    );
    let bf16 = DenseMatrix::upload(vec![bf16::from_f32(1.0); total * cols], total, cols, stream)?;
    let input_rows = if input_stride == 0 { 1 } else { experts };
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; input_rows * cols]))
        .reshape(&[input_rows, cols / 32, 32])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[experts, rows]).sync_on(stream)?;
    let reference = do_bench(stream, &BenchOptions::default(), |stream| {
        expert_gemv_view(
            &bf16,
            &x.view(&[input_rows, cols / 32, 32])?,
            &selection,
            y.slice_mut(&[0..experts, 0..rows])?
                .partition([1, DEFAULT_GEMV_TILE.rows]),
            rows,
            input_stride,
            DEFAULT_GEMV_TILE,
        )
        .sync_on(stream)
        .map(|_| ())
        .map_err(Into::into)
    })?
    .median_ms();
    for tile_rows in [4, 8, 16, 32, 64, 128] {
        for blocks in [1, 2, 4] {
            if only.is_some_and(|tile| tile.rows != tile_rows || tile.blocks != blocks) {
                continue;
            }
            let tile = GemvTile {
                rows: tile_rows,
                blocks,
            };
            let ms = do_bench(stream, &BenchOptions::default(), |stream| {
                expert_gemv_weight_view(
                    &fp8,
                    &x.view(&[input_rows, cols / 32, 32])?,
                    &selection,
                    y.slice_mut(&[0..experts, 0..rows])?
                        .partition([1, tile.rows]),
                    rows,
                    input_stride,
                    tile,
                )
                .submit(&mut Eager(stream))
                .map_err(Into::into)
            })?
            .median_ms();
            println!(
                "expert_stride{input_stride},{rows},{cols},{tile_rows},{blocks},{ms:.6},{reference:.6}"
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    assert!(!cfg!(debug_assertions), "measurement requires --release");
    cutile::jit_cache::enable_default()?;
    let mut args = std::env::args().skip(1);
    let kind = args
        .next()
        .ok_or("usage: fp8_gemv_sweep dense ROWS COLS [TILE_ROWS BLOCKS] | expert ROWS COLS SELECTED_EXPERTS INPUT_STRIDE [TILE_ROWS BLOCKS]")?;
    let rows: usize = args.next().ok_or("missing ROWS")?.parse()?;
    let cols: usize = args.next().ok_or("missing COLS")?.parse()?;
    let experts = if kind == "expert" {
        args.next().ok_or("missing SELECTED_EXPERTS")?.parse()?
    } else {
        1
    };
    let input_stride = if kind == "expert" {
        args.next().ok_or("missing INPUT_STRIDE")?.parse()?
    } else {
        0
    };
    let only = match args.next() {
        Some(rows) => Some(GemvTile {
            rows: rows.parse()?,
            blocks: args.next().ok_or("missing BLOCKS")?.parse()?,
        }),
        None => None,
    };
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let stream = Device::new(0)?.new_stream()?;
    match kind.as_str() {
        "dense" => dense(&stream, rows, cols, only),
        "expert" => expert(&stream, rows, cols, experts, input_stride, only),
        _ => Err("kind must be dense or expert".into()),
    }
}
