mod common;

use common::{check, oracle, strategy};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use firepit_kernels::gemv::DEFAULT_GEMV_TILE;
use firepit_kernels::gemv::gemv;
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{DenseMatrix, Weight};
use half::bf16;
use proptest::prelude::*;
use std::sync::Arc;

fn gpu(
    stream: &Arc<Stream>,
    weights: Vec<f32>,
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let batch = input.len() / cols;
    let padded = batch.next_power_of_two();
    let weights: Vec<bf16> = weights
        .into_iter()
        .map(|weight| {
            let rounded = bf16::from_f32(weight);
            assert_eq!(rounded.to_f32(), weight);
            rounded
        })
        .collect();
    let matrix = DenseMatrix::upload(weights, rows, cols, stream)?;
    let mut padded_input = input.to_vec();
    padded_input.resize(padded * cols, 0.0);
    let x = api::copy_host_vec_to_device(&Arc::new(padded_input))
        .reshape(&[padded, cols / 32, 32])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[padded, rows]).sync_on(stream)?;
    let x_view = x.view(&[padded, cols / 32, 32])?;
    gemv(
        &Weight::Dense(matrix.clone()),
        &x_view,
        (&mut y).partition([padded, DEFAULT_GEMV_TILE.rows]),
        DEFAULT_GEMV_TILE,
    )
    .submit(&mut Eager(stream))?;
    let mut actual: Vec<f32> = y.to_host_vec().sync_on(stream)?;
    actual.truncate(batch * rows);
    Ok(actual)
}

#[test]
fn dense_matches_f64_oracle() {
    let stream = common::stream();
    common::runner(64, file!())
        .run(&strategy::dense_case(), |case| {
            let weights: Vec<f32> = case
                .weights
                .iter()
                .map(|&weight| bf16::from_f32(weight).to_f32())
                .collect();
            let actual = gpu(&stream, weights.clone(), &case.input, case.rows, case.cols)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert_eq!(actual.len(), case.batch * case.rows);
            let expected = oracle::dense(&weights, &case.input, case.rows, case.cols);
            let magnitude = oracle::dense_magnitude(&weights, &case.input, case.rows, case.cols);
            prop_assert!(
                check::inner_product_excess(&actual, &expected, &magnitude, case.cols) <= 1.0
            );
            Ok(())
        })
        .unwrap();
}

#[test]
fn dense_preserves_linearity_batch_and_row_order() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let (rows, cols) = (32, 128);
    let weights: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) / 128.0)
        .collect();
    let x: Vec<f32> = (0..4 * cols)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) / 37.0)
        .collect();
    let z: Vec<f32> = (0..4 * cols)
        .map(|i| ((i * 7 % 43) as f32 - 21.0) / 43.0)
        .collect();
    let combined: Vec<f32> = x.iter().zip(&z).map(|(a, b)| 0.5 * a - 1.25 * b).collect();
    let fx = gpu(&stream, weights.clone(), &x, rows, cols)?;
    let fz = gpu(&stream, weights.clone(), &z, rows, cols)?;
    let fc = gpu(&stream, weights.clone(), &combined, rows, cols)?;
    let expected: Vec<f32> = fx
        .iter()
        .zip(&fz)
        .map(|(a, b)| 0.5 * a - 1.25 * b)
        .collect();
    assert!(check::max_abs(&fc, &expected) < 2e-5);

    for b in 0..4 {
        let single = gpu(
            &stream,
            weights.clone(),
            &x[b * cols..(b + 1) * cols],
            rows,
            cols,
        )?;
        assert_eq!(&fx[b * rows..(b + 1) * rows], single);
    }

    let mut reversed = weights.clone();
    for row in 0..rows {
        reversed[row * cols..(row + 1) * cols]
            .copy_from_slice(&weights[(rows - 1 - row) * cols..(rows - row) * cols]);
    }
    let permuted = gpu(&stream, reversed, &x, rows, cols)?;
    for b in 0..4 {
        for row in 0..rows {
            assert_eq!(permuted[b * rows + row], fx[b * rows + rows - 1 - row]);
        }
    }
    Ok(())
}
