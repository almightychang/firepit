use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream};
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::error::Error;
use cutile::jit_cache::{self, FileSystemJitStore};
use cutile::tensor::{IntoPartition, Tensor};
use firepit_kernels::q8::{HostQ8Matrix, Q8Matrix, q8_gemv, q8_gemv_v1b, q8_gemv_v2, quantize_x};
use half::f16;
use std::sync::Arc;
use std::time::Duration;

const PEAK_GBPS: f64 = 960.0;
const SHAPES: [(usize, usize); 3] = [(2560, 10240), (6144, 2560), (2560, 248320)];

fn matrix(rows: usize, cols: usize) -> HostQ8Matrix {
    let qs = (0..rows * cols)
        .map(|i| ((i * 17 % 31) as i8) - 15)
        .collect();
    let scales = (0..rows * cols / 32)
        .map(|i| f16::from_f32(((i % 13) + 1) as f32 / 32.0))
        .collect();
    HostQ8Matrix {
        qs,
        scales,
        rows,
        cols,
    }
}

fn options() -> BenchOptions {
    BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(150),
        min_reps: 15,
        max_reps: 60,
        clear_l2: true,
    }
}

fn measure_quantize<const B: i32>(
    x: &Tensor<f32>,
    cols: usize,
    stream: &Arc<Stream>,
) -> Result<f64, Error> {
    let mut xq = Some(
        api::zeros::<i8>(&[B as usize, cols / 32, 32])
            .sync_on(stream)?
            .partition([1, 1, 32]),
    );
    let mut dx = Some(
        api::zeros::<f32>(&[B as usize, cols / 32])
            .sync_on(stream)?
            .partition([1, 1]),
    );
    let result = do_bench(stream, &options(), |stream| {
        let (next_xq, next_dx, _) = quantize_x(
            x,
            xq.take().expect("xq partition"),
            dx.take().expect("dx partition"),
        )
        .sync_on(stream)?;
        xq = Some(next_xq);
        dx = Some(next_dx);
        Ok(())
    })?;
    Ok(result.median_ms() as f64)
}

fn measure<const B: i32, const BN: i32, const NB: i32>(
    w: &Q8Matrix,
    x: &Tensor<f32>,
    xq: &Tensor<i8>,
    dx: &Tensor<f32>,
    stream: &Arc<Stream>,
) -> Result<(f64, f64, f64), Error> {
    let run = |variant: u8| -> Result<f64, Error> {
        let mut y = Some(
            api::zeros::<f32>(&[B as usize, w.rows])
                .sync_on(stream)?
                .partition([B as usize, BN as usize]),
        );
        let result = do_bench(stream, &options(), |stream| {
            let current = y.take().expect("output partition");
            let next = match variant {
                1 => q8_gemv::<B, BN, NB>(w, x, current).sync_on(stream)?.0,
                2 => q8_gemv_v1b::<B, BN, NB>(w, x, current).sync_on(stream)?.0,
                3 => {
                    q8_gemv_v2::<B, BN, NB>(w, xq, dx, current)
                        .sync_on(stream)?
                        .0
                }
                _ => unreachable!(),
            };
            y = Some(next);
            Ok(())
        })?;
        Ok(result.median_ms() as f64)
    };
    Ok((run(1)?, run(2)?, run(3)?))
}

fn run_batch<const B: i32>(rows: usize, cols: usize, stream: &Arc<Stream>) -> Result<(), Error> {
    let w = matrix(rows, cols).upload(stream)?;
    let x_host = Arc::new(
        (0..B as usize * cols)
            .map(|i| ((i * 11 % 37) as f32 - 18.0) / 37.0)
            .collect::<Vec<f32>>(),
    );
    let x = api::copy_host_vec_to_device(&x_host)
        .reshape(&[B as usize, cols / 32, 32])
        .sync_on(stream)?;
    let xq = api::zeros::<i8>(&[B as usize, cols / 32, 32])
        .sync_on(stream)?
        .partition([1, 1, 32]);
    let dx = api::zeros::<f32>(&[B as usize, cols / 32])
        .sync_on(stream)?
        .partition([1, 1]);
    let (xq, dx, _) = quantize_x(&x, xq, dx).sync_on(stream)?;
    let xq = xq.unpartition();
    let dx = dx.unpartition();
    let quantize_ms = measure_quantize::<B>(&x, cols, stream)?;
    let (bn, nb, (v1, v1b, v2)) = if rows == 2560 {
        (32, 8, measure::<B, 32, 8>(&w, &x, &xq, &dx, stream)?)
    } else {
        (64, 8, measure::<B, 64, 8>(&w, &x, &xq, &dx, stream)?)
    };
    let gbps = rows as f64 * cols as f64 * 1.0625 / (v2 * 1e6);
    println!(
        "{cols},{rows},{B},{bn},{nb},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2}",
        v1 * 1000.0,
        v1b * 1000.0,
        v2 * 1000.0,
        quantize_ms * 1000.0,
        (v2 + quantize_ms) * 1000.0,
        gbps,
        gbps / PEAK_GBPS * 100.0
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    jit_cache::enable(Arc::new(FileSystemJitStore::default_location()?));
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    println!("K,N,B,BN,NB,v1_us,v1b_us,v2_us,quantize_us,v2_total_us,v2_GB/s,v2_peak_%");
    for (cols, rows) in SHAPES {
        run_batch::<1>(rows, cols, &stream)?;
        run_batch::<4>(rows, cols, &stream)?;
        run_batch::<8>(rows, cols, &stream)?;
    }
    Ok(())
}
