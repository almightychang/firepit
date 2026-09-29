use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use firepit_kernels::dense::{DenseMatrix, dense_gemv};
use firepit_kernels::q8::DEFAULT_GEMV_TILE;
use half::bf16;
use std::sync::Arc;

#[test]
fn bf16_and_f32_small_shapes_all_batches() -> Result<(), Box<dyn std::error::Error>> {
    let (rows, cols) = (32, 128);
    let weights: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) / 256.0)
        .collect();
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let bf16_weights: Vec<bf16> = weights.iter().map(|&x| bf16::from_f32(x)).collect();
    let bf16_matrix = DenseMatrix::upload(bf16_weights.clone(), rows, cols, &stream)?;
    let f32_matrix = DenseMatrix::upload(weights.clone(), rows, cols, &stream)?;

    for batch in 1usize..=4 {
        let padded = batch.next_power_of_two();
        let mut input: Vec<f32> = (0..batch * cols)
            .map(|i| ((i * 11 % 37) as f32 - 18.0) / 37.0)
            .collect();
        input.resize(padded * cols, 0.0);
        let x = api::copy_host_vec_to_device(&Arc::new(input.clone()))
            .reshape(&[padded, cols / 32, 32])
            .sync_on(&stream)?;
        let x_view = x.view(&[padded, cols / 32, 32])?;
        let mut y = api::zeros::<f32>(&[padded, rows]).sync_on(&stream)?;
        dense_gemv(
            &bf16_matrix,
            &x_view,
            (&mut y).partition([padded, DEFAULT_GEMV_TILE.rows]),
            DEFAULT_GEMV_TILE,
        )
        .sync_on(&stream)?;
        let actual_bf16: Vec<f32> = y.to_host_vec().sync_on(&stream)?;
        let mut y = api::zeros::<f32>(&[padded, rows]).sync_on(&stream)?;
        dense_gemv(
            &f32_matrix,
            &x_view,
            (&mut y).partition([padded, DEFAULT_GEMV_TILE.rows]),
            DEFAULT_GEMV_TILE,
        )
        .sync_on(&stream)?;
        let actual_f32: Vec<f32> = y.to_host_vec().sync_on(&stream)?;
        for (index, (bf16_got, f32_got)) in actual_bf16[..batch * rows]
            .iter()
            .zip(&actual_f32[..batch * rows])
            .enumerate()
        {
            let row = index % rows;
            let b = index / rows;
            let expected_bf16: f64 = (0..cols)
                .map(|k| bf16_weights[row * cols + k].to_f32() as f64 * input[b * cols + k] as f64)
                .sum();
            let expected_f32: f64 = (0..cols)
                .map(|k| weights[row * cols + k] as f64 * input[b * cols + k] as f64)
                .sum();
            assert!((*bf16_got as f64 - expected_bf16).abs() < 1e-4);
            assert!((*f32_got as f64 - expected_f32).abs() < 1e-4);
        }
    }
    Ok(())
}
