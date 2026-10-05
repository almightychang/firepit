use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::tensor::PartitionMut;
use firepit_kernels::gemm::{DEFAULT_GEMM_TILE, fp8_gemm, gemm};
use firepit_kernels::gemv::{DEFAULT_GEMV_TILE, gemv};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{DenseMatrix, HostFp8Matrix, Weight};
use half::bf16;
use std::error::Error;
use std::sync::Arc;

fn measure(
    stream: &Arc<Stream>,
    rows: usize,
    cols: usize,
    tokens: usize,
    block_n: usize,
    block_k: usize,
) -> Result<(), Box<dyn Error>> {
    let fp8 = HostFp8Matrix::from_parts(
        vec![f8e4m3fn(0x38); rows * cols],
        vec![1.0; (rows / block_n) * (cols / block_k)],
        rows,
        cols,
        block_n,
        block_k,
    )?
    .upload(stream)?;
    let dense = DenseMatrix::upload(vec![bf16::from_f32(1.0); rows * cols], rows, cols, stream)?;
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
        .reshape(&[1, cols / 32, 32])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(stream)?;
    let fp8_gemv_ms = do_bench(stream, &BenchOptions::default(), |stream| {
        gemv(
            &Weight::Fp8(fp8.clone()),
            &x.view(&[1, cols / 32, 32])?,
            (&mut y).partition([1, DEFAULT_GEMV_TILE.rows]),
            DEFAULT_GEMV_TILE,
        )
        .submit(&mut Eager(stream))
        .map_err(Into::into)
    })?
    .median_ms();
    let dense_gemv_ms = do_bench(stream, &BenchOptions::default(), |stream| {
        gemv(
            &Weight::Dense(dense.clone()),
            &x.view(&[1, cols / 32, 32])?,
            (&mut y).partition([1, DEFAULT_GEMV_TILE.rows]),
            DEFAULT_GEMV_TILE,
        )
        .submit(&mut Eager(stream))
        .map_err(Into::into)
    })?
    .median_ms();

    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; tokens * cols]))
        .reshape(&[tokens, cols])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[tokens, rows]).sync_on(stream)?;
    let fp8_gemm_ms = do_bench(stream, &BenchOptions::default(), |stream| {
        fp8_gemm(
            &fp8,
            &x.view(&[tokens, cols])?,
            (&mut y).partition([DEFAULT_GEMM_TILE.m, DEFAULT_GEMM_TILE.n]),
            DEFAULT_GEMM_TILE,
        )
        .submit(&mut Eager(stream))
        .map_err(Into::into)
    })?
    .median_ms();
    let dense_gemm_ms = do_bench(stream, &BenchOptions::default(), |stream| {
        gemm(
            &dense,
            &x.view(&[tokens, cols])?,
            (&mut y).partition([DEFAULT_GEMM_TILE.m, DEFAULT_GEMM_TILE.n]),
            DEFAULT_GEMM_TILE,
        )
        .submit(&mut Eager(stream))
        .map_err(Into::into)
    })?
    .median_ms();
    println!(
        "N={rows} K={cols} M={tokens} block={block_n}x{block_k} fp8_gemv_ms={fp8_gemv_ms:.6} bf16_gemv_ms={dense_gemv_ms:.6} fp8_gemm_ms={fp8_gemm_ms:.6} bf16_gemm_ms={dense_gemm_ms:.6}"
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    assert!(!cfg!(debug_assertions), "measurement requires --release");
    cutile::jit_cache::enable_default()?;
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse())
        .collect::<Result<_, _>>()?;
    if args.len() != 5 {
        return Err("usage: fp8_measure ROWS COLS TOKENS BLOCK_ROWS BLOCK_COLS".into());
    }
    let stream = Device::new(0)?.new_stream()?;
    measure(&stream, args[0], args[1], args[2], args[3], args[4])
}
