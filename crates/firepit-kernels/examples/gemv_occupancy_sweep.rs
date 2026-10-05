use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::tensor::{IntoPartition, PartitionMut};
use cutile::tile_kernel::{CompileOptions, ToHostVecOp};
use firepit_kernels::gemv::{GemvTile, expert_gemv_weight_view, gemv};
use firepit_kernels::moe::{ExpertSelection, MoeDims};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{DenseMatrix, HostFp8Matrix, Weight};
use half::bf16;
use std::error::Error;
use std::sync::Arc;

#[derive(Clone, Copy)]
struct Hints {
    occupancy: Option<i32>,
    workers: Option<i32>,
}

impl Hints {
    fn options(self) -> Option<CompileOptions> {
        if self.occupancy.is_none() && self.workers.is_none() {
            return None;
        }
        let mut options = CompileOptions::default();
        if let Some(occupancy) = self.occupancy {
            options = options.occupancy(occupancy);
        }
        if let Some(workers) = self.workers {
            options = options.num_worker_warps_per_cta(workers);
        }
        Some(options)
    }

    fn occupancy_label(self) -> String {
        self.occupancy
            .map_or("default".into(), |value| value.to_string())
    }

    fn workers_label(self) -> String {
        self.workers
            .map_or("default".into(), |value| value.to_string())
    }
}

fn hint(value: &str, choices: &[i32]) -> Result<Option<i32>, Box<dyn Error>> {
    if value == "default" {
        return Ok(None);
    }
    let value = value.parse()?;
    if choices.contains(&value) {
        Ok(Some(value))
    } else {
        Err("unsupported hint value".into())
    }
}

fn print_row(
    kind: &str,
    rows: usize,
    cols: usize,
    tile: GemvTile,
    hints: Hints,
    weight_bytes: usize,
    ms: f32,
) {
    println!(
        "{kind},{rows},{cols},{},{},{},{},{ms:.6},{:.3}",
        tile.rows,
        tile.blocks,
        hints.occupancy_label(),
        hints.workers_label(),
        weight_bytes as f64 / (ms as f64 * 1e6),
    );
}

fn dense(
    stream: &Arc<Stream>,
    rows: usize,
    cols: usize,
    tile: GemvTile,
    hints: Hints,
) -> Result<(), Box<dyn Error>> {
    let fp8 = Weight::Fp8(
        HostFp8Matrix::from_parts(
            vec![f8e4m3fn(0x38); rows * cols],
            vec![bf16::from_f32(1.0); rows / 128 * (cols / 128)],
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
    for (kind, weight, bytes_per_weight) in [("dense_fp8", &fp8, 1), ("dense_bf16", &bf16, 2)] {
        let ms = do_bench(stream, &BenchOptions::default(), |stream| {
            let x_view = x.view(&[1, cols / 32, 32])?;
            let op = gemv(weight, &x_view, (&mut y).partition([1, tile.rows]), tile);
            let op = match hints.options() {
                Some(options) => op.compile_options(options),
                None => op,
            };
            op.submit(&mut Eager(stream)).map_err(Into::into)
        })?
        .median_ms();
        let output: Vec<f32> = y.dup().to_host_vec().sync_on(stream)?;
        if !output.iter().all(|&value| value == cols as f32) {
            return Err(format!("{kind} produced an unexpected output").into());
        }
        print_row(
            kind,
            rows,
            cols,
            tile,
            hints,
            rows * cols * bytes_per_weight,
            ms,
        );
    }
    Ok(())
}

fn expert(
    stream: &Arc<Stream>,
    rows: usize,
    cols: usize,
    experts: usize,
    input_stride: usize,
    tile: GemvTile,
    hints: Hints,
) -> Result<(), Box<dyn Error>> {
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
            vec![bf16::from_f32(1.0); total / 128 * (cols / 128)],
            total,
            cols,
            128,
            128,
        )?
        .upload(stream)?,
    );
    let bf16 = Weight::Dense(DenseMatrix::upload(
        vec![bf16::from_f32(1.0); total * cols],
        total,
        cols,
        stream,
    )?);
    let input_rows = if input_stride == 0 { 1 } else { experts };
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; input_rows * cols]))
        .reshape(&[input_rows, cols / 32, 32])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[experts, rows]).sync_on(stream)?;
    for (kind, weight, bytes_per_weight) in [("expert_fp8", &fp8, 1), ("expert_bf16", &bf16, 2)] {
        let ms = do_bench(stream, &BenchOptions::default(), |stream| {
            let x_view = x.view(&[input_rows, cols / 32, 32])?;
            let op = expert_gemv_weight_view(
                weight,
                &x_view,
                &selection,
                y.slice_mut(&[0..experts, 0..rows])?
                    .partition([1, tile.rows]),
                rows,
                input_stride,
                tile,
            );
            let op = match hints.options() {
                Some(options) => op.compile_options(options),
                None => op,
            };
            op.submit(&mut Eager(stream)).map_err(Into::into)
        })?
        .median_ms();
        let output: Vec<f32> = y.dup().to_host_vec().sync_on(stream)?;
        if !output.iter().all(|&value| value == cols as f32) {
            return Err(format!("{kind} produced an unexpected output").into());
        }
        print_row(
            kind,
            rows,
            cols,
            tile,
            hints,
            total * cols * bytes_per_weight,
            ms,
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    assert!(!cfg!(debug_assertions), "measurement requires --release");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 7 && args.len() != 9 {
        return Err("usage: gemv_occupancy_sweep dense N K ROWS BLOCKS OCCUPANCY WORKERS | expert N K EXPERTS INPUT_STRIDE ROWS BLOCKS OCCUPANCY WORKERS (hints: default or integer)".into());
    }
    let kind = args[0].as_str();
    if (kind == "dense") != (args.len() == 7) || (kind == "expert") != (args.len() == 9) {
        return Err("kind and argument count disagree".into());
    }
    let rows: usize = args[1].parse()?;
    let cols: usize = args[2].parse()?;
    let (experts, input_stride, offset) = if kind == "expert" {
        (args[3].parse()?, args[4].parse()?, 5)
    } else {
        (1, 0, 3)
    };
    let tile = GemvTile {
        rows: args[offset].parse()?,
        blocks: args[offset + 1].parse()?,
    };
    let hints = Hints {
        occupancy: hint(&args[offset + 2], &[1, 2, 3, 4])?,
        workers: hint(&args[offset + 3], &[4, 8])?,
    };
    if rows == 0
        || cols == 0
        || experts == 0
        || input_stride > 1
        || tile.rows == 0
        || tile.blocks == 0
        || !tile.rows.is_power_of_two()
        || !tile.blocks.is_power_of_two()
        || rows % 128 != 0
        || cols % 128 != 0
        || 128 % tile.rows != 0
        || 128 % (tile.blocks * 32) != 0
        || !tile.fits(rows, cols)
    {
        return Err(
            "shape/tile must fit 128x128 FP8 scale blocks; INPUT_STRIDE must be 0 or 1".into(),
        );
    }
    cutile::jit_cache::enable_default()?;
    let stream = Device::new(0)?.new_stream()?;
    println!("kind,N,K,rows,blocks,occupancy,workers,ms,GBps");
    match kind {
        "dense" => dense(&stream, rows, cols, tile, hints),
        "expert" => expert(&stream, rows, cols, experts, input_stride, tile, hints),
        _ => unreachable!(),
    }
}
