use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{IntoPartition, ToHostVec};
use firepit_kernels::q8::{
    HostQ8Matrix, q8_gemv, q8_gemv_v1b, q8_gemv_v2, quantize_x, quantize_x_cpu,
};
use half::f16;
use std::sync::Arc;

fn fixture(rows: usize, cols: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * cols / 32 * 34);
    for block in 0..rows * cols / 32 {
        let scale = f16::from_f32(((block % 13) + 1) as f32 / 32.0);
        bytes.extend(scale.to_bits().to_le_bytes());
        for i in 0..32 {
            bytes.push((((block * 17 + i * 7) % 31) as i8 - 15) as u8);
        }
    }
    bytes
}

fn run<const B: i32>(
    rows: usize,
    cols: usize,
    bn: usize,
    nb: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let host = HostQ8Matrix::from_gguf_q8_0(&fixture(rows, cols), rows, cols)?;
    let x_host: Vec<f32> = (0..B as usize * cols)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) / 37.0)
        .collect();
    let expected = host.gemv_reference(&x_host)?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let w = host.upload(&stream)?;
    let x = api::copy_host_vec_to_device(&Arc::new(x_host))
        .reshape(&[B as usize, cols / 32, 32])
        .sync_on(&stream)?;
    let y = api::zeros::<f32>(&[B as usize, rows])
        .sync_on(&stream)?
        .partition([B as usize, bn]);
    let (y, _, _, _, _) = match (bn, nb) {
        (16, 4) => q8_gemv::<B, 16, 4>(&w, &x, y).sync_on(&stream)?,
        (32, 8) => q8_gemv::<B, 32, 8>(&w, &x, y).sync_on(&stream)?,
        _ => unreachable!(),
    };
    let actual: Vec<f32> = y.unpartition().to_host_vec().sync_on(&stream)?;
    let mut max_relative = 0.0f64;
    for (got, want) in actual.into_iter().zip(expected) {
        let relative = (got as f64 - want).abs() / want.abs().max(1.0);
        max_relative = max_relative.max(relative);
    }
    Ok(max_relative)
}

fn run_variants<const B: i32>(
    rows: usize,
    cols: usize,
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    let host = HostQ8Matrix::from_gguf_q8_0(&fixture(rows, cols), rows, cols)?;
    let x_host: Vec<f32> = (0..B as usize * cols)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) / 37.0)
        .collect();
    let f32_reference = host.gemv_reference(&x_host)?;
    let quantized = quantize_x_cpu(&x_host, B as usize, cols)?;
    let quantized_reference = host.gemv_quantized_reference(&quantized)?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let w = host.upload(&stream)?;
    let x = api::copy_host_vec_to_device(&Arc::new(x_host))
        .reshape(&[B as usize, cols / 32, 32])
        .sync_on(&stream)?;
    let xq = api::zeros::<i8>(&[B as usize, cols / 32, 32])
        .sync_on(&stream)?
        .partition([1, 1, 32]);
    let dx = api::zeros::<f32>(&[B as usize, cols / 32])
        .sync_on(&stream)?
        .partition([1, 1]);
    let (xq, dx, _) = quantize_x(&x, xq, dx).sync_on(&stream)?;
    let xq = Arc::new(xq.unpartition());
    let dx = Arc::new(dx.unpartition());
    let gpu_xq: Vec<i8> = xq.clone().to_host_vec().sync_on(&stream)?;
    let gpu_dx: Vec<f32> = dx.clone().to_host_vec().sync_on(&stream)?;
    assert_eq!(gpu_xq, quantized.xq);
    for (gpu, cpu) in gpu_dx.iter().zip(&quantized.dx) {
        assert!(
            (gpu - cpu).abs() < 1e-7,
            "quantization scale {gpu} != {cpu}"
        );
    }
    let y = api::zeros::<f32>(&[B as usize, rows])
        .sync_on(&stream)?
        .partition([B as usize, 16]);
    let (y, _, _, _, _) = q8_gemv_v1b::<B, 16, 4>(&w, &x, y).sync_on(&stream)?;
    let v1b: Vec<f32> = y.unpartition().to_host_vec().sync_on(&stream)?;
    let y = api::zeros::<f32>(&[B as usize, rows])
        .sync_on(&stream)?
        .partition([B as usize, 16]);
    let (y, _, _, _, _, _) = q8_gemv_v2::<B, 16, 4>(&w, &xq, &dx, y).sync_on(&stream)?;
    let v2: Vec<f32> = y.unpartition().to_host_vec().sync_on(&stream)?;
    let relative = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
    let v1b_error = v1b
        .iter()
        .zip(&f32_reference)
        .map(|(&got, &want)| relative(got as f64, want))
        .fold(0.0f64, f64::max);
    let v2_error = v2
        .iter()
        .zip(&quantized_reference)
        .map(|(&got, &want)| relative(got as f64, want))
        .fold(0.0f64, f64::max);
    let quantization_error = quantized_reference
        .iter()
        .zip(&f32_reference)
        .map(|(&got, &want)| relative(got, want))
        .fold(0.0f64, f64::max);
    Ok((v1b_error, v2_error, quantization_error))
}

#[test]
fn v2_small_shapes_all_batches() -> Result<(), Box<dyn std::error::Error>> {
    for (batch, errors) in [
        (1, run_variants::<1>(32, 128)?),
        (2, run_variants::<2>(32, 128)?),
        (4, run_variants::<4>(32, 128)?),
        (8, run_variants::<8>(32, 128)?),
    ] {
        println!(
            "B={batch} v1b={:.3e} v2_vs_quantized={:.3e} quantization_vs_f32={:.3e}",
            errors.0, errors.1, errors.2
        );
        assert!(errors.0 < 2e-4);
        assert!(errors.1 < 2e-4);
    }
    Ok(())
}

#[test]
fn v2_multiple_k_chunks() -> Result<(), Box<dyn std::error::Error>> {
    let (v1b, v2, quantization) = run_variants::<4>(64, 256)?;
    println!(
        "multiple chunks v1b={v1b:.3e} v2_vs_quantized={v2:.3e} quantization_vs_f32={quantization:.3e}"
    );
    assert!(v1b < 2e-4);
    assert!(v2 < 2e-4);
    Ok(())
}

#[test]
fn quantize_zero_block_and_halfway_values() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = vec![0.0f32; 64];
    input[32] = 127.0;
    input[33] = 0.5;
    input[34] = -0.5;
    input[35] = 1.5;
    input[36] = -1.5;
    let expected = quantize_x_cpu(&input, 1, 64)?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let x = api::copy_host_vec_to_device(&Arc::new(input))
        .reshape(&[1, 2, 32])
        .sync_on(&stream)?;
    let xq = api::zeros::<i8>(&[1, 2, 32])
        .sync_on(&stream)?
        .partition([1, 1, 32]);
    let dx = api::zeros::<f32>(&[1, 2])
        .sync_on(&stream)?
        .partition([1, 1]);
    let (xq, dx, _) = quantize_x(&x, xq, dx).sync_on(&stream)?;
    let actual_xq: Vec<i8> = xq.unpartition().to_host_vec().sync_on(&stream)?;
    let actual_dx: Vec<f32> = dx.unpartition().to_host_vec().sync_on(&stream)?;
    assert_eq!(actual_xq, expected.xq);
    assert_eq!(actual_dx, expected.dx);
    Ok(())
}

#[test]
fn small_shapes_all_batches() -> Result<(), Box<dyn std::error::Error>> {
    for error in [
        run::<1>(32, 128, 16, 4)?,
        run::<2>(32, 128, 16, 4)?,
        run::<4>(32, 128, 16, 4)?,
        run::<8>(32, 128, 16, 4)?,
    ] {
        assert!(error < 2e-4, "relative error {error}");
        println!("relative error {error:.3e}");
    }
    Ok(())
}

#[test]
#[ignore = "large GPU and CPU reference run"]
fn model_shapes() -> Result<(), Box<dyn std::error::Error>> {
    for (rows, cols) in [(10240, 2560), (2560, 6144), (248320, 2560)] {
        let error = run::<1>(rows, cols, 32, 8)?;
        assert!(
            error < 2e-4,
            "shape ({cols}, {rows}) relative error {error}"
        );
        println!("({cols}, {rows}) relative error {error:.3e}");
    }
    Ok(())
}
