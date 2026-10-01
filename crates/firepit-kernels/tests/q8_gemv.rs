mod common;

use common::{check, oracle, strategy};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use firepit_kernels::q8::{DEFAULT_GEMV_TILE, GemvTile, HostQ8Matrix, q8_gemv};
use half::f16;
use proptest::prelude::*;
use std::sync::Arc;

fn gpu(
    stream: &Arc<Stream>,
    qs: &[i8],
    scales: &[f16],
    input: &[f32],
    rows: usize,
    cols: usize,
    tile: GemvTile,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let batch = input.len() / cols;
    let padded = batch.next_power_of_two();
    let weights =
        HostQ8Matrix::from_parts(qs.to_vec(), scales.to_vec(), rows, cols)?.upload(stream)?;
    let mut padded_input = input.to_vec();
    padded_input.resize(padded * cols, 0.0);
    let x = api::copy_host_vec_to_device(&Arc::new(padded_input))
        .reshape(&[padded, cols / 32, 32])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[padded, rows]).sync_on(stream)?;
    let x_view = x.view(&[padded, cols / 32, 32])?;
    q8_gemv(
        &weights,
        &x_view,
        (&mut y).partition([padded, tile.rows]),
        tile,
    )
    .sync_on(stream)?;
    let mut actual: Vec<f32> = y.to_host_vec().sync_on(stream)?;
    actual.truncate(batch * rows);
    Ok(actual)
}

#[test]
fn q8_matches_f64_oracle() {
    let stream = common::stream();
    common::runner(8, file!())
        .run(&strategy::q8_case(), |case| {
            let scales: Vec<f16> = case.scales.iter().copied().map(f16::from_f32).collect();
            let actual = gpu(
                &stream,
                &case.qs,
                &scales,
                &case.input,
                case.rows,
                case.cols,
                DEFAULT_GEMV_TILE,
            )
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert_eq!(actual.len(), case.batch * case.rows);
            let expected = oracle::q8(&case.qs, &scales, &case.input, case.rows, case.cols);
            prop_assert!(check::max_relative(&actual, &expected) < 5e-4);
            Ok(())
        })
        .unwrap();
}

#[test]
fn q8_preserves_linearity_batch_and_row_order() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let (rows, cols) = (32, 128);
    let qs: Vec<i8> = (0..rows * cols).map(|i| (i * 13 % 17) as i8 - 8).collect();
    let scales: Vec<f16> = (0..rows * cols / 32)
        .map(|i| f16::from_f32(((i % 7) + 1) as f32 / 32.0))
        .collect();
    let x: Vec<f32> = (0..4 * cols)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) / 37.0)
        .collect();
    let z: Vec<f32> = (0..4 * cols)
        .map(|i| ((i * 7 % 43) as f32 - 21.0) / 43.0)
        .collect();
    let combined: Vec<f32> = x.iter().zip(&z).map(|(a, b)| 0.5 * a - 1.25 * b).collect();
    let fx = gpu(&stream, &qs, &scales, &x, rows, cols, DEFAULT_GEMV_TILE)?;
    let fz = gpu(&stream, &qs, &scales, &z, rows, cols, DEFAULT_GEMV_TILE)?;
    let fc = gpu(
        &stream,
        &qs,
        &scales,
        &combined,
        rows,
        cols,
        DEFAULT_GEMV_TILE,
    )?;
    let expected: Vec<f32> = fx
        .iter()
        .zip(&fz)
        .map(|(a, b)| 0.5 * a - 1.25 * b)
        .collect();
    assert!(check::max_abs(&fc, &expected) < 2e-5);

    for b in 0..4 {
        let single = gpu(
            &stream,
            &qs,
            &scales,
            &x[b * cols..(b + 1) * cols],
            rows,
            cols,
            DEFAULT_GEMV_TILE,
        )?;
        assert_eq!(&fx[b * rows..(b + 1) * rows], single);
    }

    let mut reversed_qs = qs.clone();
    let mut reversed_scales = scales.clone();
    for row in 0..rows {
        reversed_qs[row * cols..(row + 1) * cols]
            .copy_from_slice(&qs[(rows - 1 - row) * cols..(rows - row) * cols]);
        reversed_scales[row * cols / 32..(row + 1) * cols / 32]
            .copy_from_slice(&scales[(rows - 1 - row) * cols / 32..(rows - row) * cols / 32]);
    }
    let permuted = gpu(
        &stream,
        &reversed_qs,
        &reversed_scales,
        &x,
        rows,
        cols,
        DEFAULT_GEMV_TILE,
    )?;
    for b in 0..4 {
        for row in 0..rows {
            assert_eq!(permuted[b * rows + row], fx[b * rows + rows - 1 - row]);
        }
    }
    Ok(())
}

#[test]
#[ignore = "large GPU and CPU reference run"]
fn model_shapes() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    for (rows, cols) in [(10240, 2560), (2560, 6144), (248320, 2560)] {
        let qs: Vec<i8> = (0..rows * cols).map(|i| (i * 13 % 17) as i8 - 8).collect();
        let scales = vec![f16::from_f32(1.0 / 256.0); rows * cols / 32];
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i * 11 % 37) as f32 - 18.0) / 37.0)
            .collect();
        let tile = GemvTile {
            rows: 32,
            blocks: 8,
        };
        let actual = gpu(&stream, &qs, &scales, &input, rows, cols, tile)?;
        let expected = oracle::q8(&qs, &scales, &input, rows, cols);
        assert!(check::max_relative(&actual, &expected) < 2e-4);
    }
    Ok(())
}
