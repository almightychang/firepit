mod common;

use common::{check, oracle, strategy};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::elementwise::add;
use firepit_kernels::norm::{add_norm, rms_norm};
use proptest::prelude::*;
use std::sync::Arc;

fn gpu(
    stream: &Arc<Stream>,
    input: &[f32],
    gamma: &[f32],
    eps: f32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let width = gamma.len();
    let batch = input.len() / width;
    let x = api::copy_host_vec_to_device(&Arc::new(input.to_vec()))
        .reshape(&[batch, 1, width])
        .sync_on(stream)?;
    let weight = api::copy_host_vec_to_device(&Arc::new(gamma.to_vec())).sync_on(stream)?;
    let mut output = api::zeros::<f32>(&[batch, width]).sync_on(stream)?;
    rms_norm(
        (&mut output).partition([1, width]),
        &x,
        &weight,
        1.0 / width as f32,
        eps,
    )
    .generics(vec![width.to_string()])
    .sync_on(stream)?;
    Ok(output.to_host_vec().sync_on(stream)?)
}

#[test]
fn rms_norm_matches_f64_oracle() {
    let stream = common::stream();
    common::runner(64, file!())
        .run(&strategy::norm_case(), |case| {
            let actual = gpu(&stream, &case.input, &case.gamma, case.eps)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert_eq!(actual.len(), case.batch * case.width);
            let expected = oracle::rms_norm(&case.input, &case.gamma, case.eps);
            prop_assert!(check::max_relative(&actual, &expected) < 2e-5);
            Ok(())
        })
        .unwrap();
}

#[test]
fn rms_norm_preserves_batch_independence_and_weight_scale() -> Result<(), Box<dyn std::error::Error>>
{
    let stream = common::stream();
    let width = 128;
    let input: Vec<f32> = (0..4 * width)
        .map(|i| ((i * 13 % 71) as f32 - 35.0) / 8.0)
        .collect();
    let gamma: Vec<f32> = (0..width).map(|i| (i as f32 - 64.0) / 64.0).collect();
    let actual = gpu(&stream, &input, &gamma, 1e-5)?;
    for b in 0..4 {
        let single = gpu(&stream, &input[b * width..(b + 1) * width], &gamma, 1e-5)?;
        assert_eq!(&actual[b * width..(b + 1) * width], single);
    }
    let scaled: Vec<f32> = gamma.iter().map(|x| x * 2.0).collect();
    let doubled = gpu(&stream, &input, &scaled, 1e-5)?;
    let expected: Vec<f32> = actual.iter().map(|x| x * 2.0).collect();
    assert!(check::max_abs(&doubled, &expected) < 1e-5);
    let zero = gpu(&stream, &input, &vec![0.0; width], 1e-5)?;
    assert!(zero.iter().all(|x| *x == 0.0));
    Ok(())
}

#[test]
fn rms_norm_reads_interleaved_heads_of_wider_rows() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let (batch, heads, width) = (3, 4, 64);
    let pitch = 2 * width;
    let first = 1;
    let row = (first + heads + 1) * pitch;
    let buffer: Vec<f32> = (0..batch * row)
        .map(|i| ((i * 29 % 97) as f32 - 48.0) / 16.0)
        .collect();
    let gamma: Vec<f32> = (0..width).map(|i| 0.5 + i as f32 / width as f32).collect();
    let packed: Vec<f32> = (0..batch)
        .flat_map(|b| (0..heads).map(move |h| b * row + (first + h) * pitch))
        .flat_map(|start| buffer[start..start + width].to_vec())
        .collect();
    let expected = gpu(&stream, &packed, &gamma, 1e-6)?;
    let rows = api::copy_host_vec_to_device(&Arc::new(buffer))
        .reshape(&[batch, row / pitch, pitch])
        .sync_on(&stream)?;
    let x = rows.slice(&[0..batch, first..first + heads, 0..width])?;
    let weight = api::copy_host_vec_to_device(&Arc::new(gamma)).sync_on(&stream)?;
    let mut output = api::zeros::<f32>(&[batch, heads * width]).sync_on(&stream)?;
    rms_norm(
        (&mut output).partition([1, width]),
        &x,
        &weight,
        1.0 / width as f32,
        1e-6,
    )
    .generics(vec![width.to_string()])
    .sync_on(&stream)?;
    assert_eq!(output.to_host_vec().sync_on(&stream)?, expected);
    Ok(())
}

type Rows = Vec<f32>;

fn upload(
    stream: &Arc<Stream>,
    values: &[f32],
    width: usize,
) -> Result<cutile::tensor::Tensor<f32>, Box<dyn std::error::Error>> {
    Ok(api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .reshape(&[values.len() / width, width])
        .sync_on(stream)?)
}

fn gpu_add_norm(
    stream: &Arc<Stream>,
    residual: &[f32],
    delta: &[f32],
    gamma: &[f32],
    eps: f32,
) -> Result<(Rows, Rows), Box<dyn std::error::Error>> {
    let width = gamma.len();
    let batch = residual.len() / width;
    let (r, d) = (
        upload(stream, residual, width)?,
        upload(stream, delta, width)?,
    );
    let weight = api::copy_host_vec_to_device(&Arc::new(gamma.to_vec())).sync_on(stream)?;
    let mut out = api::zeros::<f32>(&[batch, width]).sync_on(stream)?;
    let mut normed = api::zeros::<f32>(&[batch, width]).sync_on(stream)?;
    add_norm(
        (&mut out).partition([1, width]),
        (&mut normed).partition([1, width]),
        &r,
        &d,
        &weight,
        1.0 / width as f32,
        eps,
    )
    .generics(vec![width.to_string()])
    .sync_on(stream)?;
    Ok((
        out.to_host_vec().sync_on(stream)?,
        normed.to_host_vec().sync_on(stream)?,
    ))
}

fn gpu_add_then_norm(
    stream: &Arc<Stream>,
    residual: &[f32],
    delta: &[f32],
    gamma: &[f32],
    eps: f32,
) -> Result<(Rows, Rows), Box<dyn std::error::Error>> {
    let width = gamma.len();
    let batch = residual.len() / width;
    let (r, d) = (
        upload(stream, residual, width)?,
        upload(stream, delta, width)?,
    );
    let mut sum = api::zeros::<f32>(&[batch, width]).sync_on(stream)?;
    add((&mut sum).partition([1, width]), &r, &d)
        .generics(vec![width.to_string()])
        .sync_on(stream)?;
    let sum: Rows = sum.to_host_vec().sync_on(stream)?;
    let normed = gpu(stream, &sum, gamma, eps)?;
    Ok((sum, normed))
}

fn add_norm_case() -> impl Strategy<Value = (strategy::NormCase, Rows)> {
    strategy::norm_case().prop_flat_map(|case| {
        let len = case.input.len();
        (Just(case), strategy::values(len))
    })
}

#[test]
fn add_norm_matches_f64_oracle() {
    let stream = common::stream();
    common::runner(64, file!())
        .run(&add_norm_case(), |(case, delta)| {
            let (out, normed) = gpu_add_norm(&stream, &case.input, &delta, &case.gamma, case.eps)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let sum: Rows = case.input.iter().zip(&delta).map(|(r, d)| r + d).collect();
            prop_assert_eq!(&out, &sum);
            let expected = oracle::rms_norm(&sum, &case.gamma, case.eps);
            prop_assert!(check::max_relative(&normed, &expected) < 2e-5);
            Ok(())
        })
        .unwrap();
}

#[test]
fn add_norm_matches_add_then_norm_bits() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let width = 2048;
    let residual: Rows = (0..3 * width)
        .map(|i| ((i * 13 % 71) as f32 - 35.0) / 8.0)
        .collect();
    let delta: Rows = (0..3 * width)
        .map(|i| ((i * 29 % 53) as f32 - 26.0) / 16.0)
        .collect();
    let gamma: Rows = (0..width).map(|i| 0.5 + (i % 17) as f32 / 16.0).collect();
    let bits = |values: &[f32]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let (out, normed) = gpu_add_norm(&stream, &residual, &delta, &gamma, 1e-6)?;
    let (sum, separate) = gpu_add_then_norm(&stream, &residual, &delta, &gamma, 1e-6)?;
    assert_eq!(bits(&out), bits(&sum));
    assert_eq!(bits(&normed), bits(&separate));
    Ok(())
}
