mod common;

use common::{check, oracle, strategy};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use firepit_kernels::gemm::{DEFAULT_GEMM_TILE, GemmTile, gemm};
use firepit_kernels::gemv::{DEFAULT_GEMV_TILE, gemv};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{DenseMatrix, Weight};
use half::bf16;
use proptest::prelude::*;
use std::sync::Arc;

const BF16_SPLIT_ERROR: f64 = (1.0 / 256.0) * (1.0 / 256.0);

#[cutile::module]
mod hi_only {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn kernel(
        y: &mut Tensor<f32, { [64, 128] }>,
        x: &Tensor<f32, { [-1, 128] }>,
        w: &Tensor<bf16, { [-1, 128] }>,
    ) {
        let block: (i32, i32, i32) = get_tile_block_id();
        let values: Tile<f32, { [64, 128] }> = x.partition(shape![64, 128]).load([block.0, 0i32]);
        let hi: Tile<bf16, { [64, 128] }> = convert_tile(values);
        let weights: Tile<bf16, { [128, 128] }> = w
            .partition(shape![128, 128])
            .load([block.1, 0i32])
            .transpose();
        let acc: Tile<f32, { [64, 128] }> = constant(0.0f32, shape![64, 128]);
        y.store(mma(hi, weights, acc));
    }
}

fn run(
    stream: &Arc<Stream>,
    matrix: &DenseMatrix<bf16>,
    input: &[f32],
    rows: usize,
    cols: usize,
    tile: GemmTile,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let batch = input.len() / cols;
    let padded = batch.div_ceil(tile.m) * tile.m;
    let mut data = input.to_vec();
    data.resize(padded * cols, 0.0);
    let x = api::copy_host_vec_to_device(&Arc::new(data))
        .reshape(&[padded, cols])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[padded, rows]).sync_on(stream)?;
    let x_view = x.view(&[padded, cols])?;
    gemm(matrix, &x_view, (&mut y).partition([tile.m, tile.n]), tile).submit(&mut Eager(stream))?;
    let mut actual: Vec<f32> = y.to_host_vec().sync_on(stream)?;
    actual.truncate(batch * rows);
    Ok(actual)
}

#[test]
fn matches_f64_and_gemv_rows() {
    let stream = common::stream();
    common::runner(64, file!())
        .run(&strategy::gemm_case(), |case| {
            let weights: Vec<bf16> = case.weights.iter().map(|&v| bf16::from_f32(v)).collect();
            let weight_f32: Vec<f32> = weights.iter().map(|&v| v.to_f32()).collect();
            let matrix = DenseMatrix::upload(weights, case.rows, case.cols, &stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let actual = run(
                &stream,
                &matrix,
                &case.input,
                case.rows,
                case.cols,
                DEFAULT_GEMM_TILE,
            )
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let expected = oracle::dense(&weight_f32, &case.input, case.rows, case.cols);
            let magnitude = oracle::dense_magnitude(&weight_f32, &case.input, case.rows, case.cols);
            for ((&a, &e), &s) in actual.iter().zip(&expected).zip(&magnitude) {
                let bound = (check::gamma_tilde(case.cols) + BF16_SPLIT_ERROR) * s;
                prop_assert!((a as f64 - e).abs() <= bound.max(f64::MIN_POSITIVE));
            }

            let weight = Weight::Dense(matrix);
            for row in 0..case.batch {
                let input = case.input[row * case.cols..(row + 1) * case.cols].to_vec();
                let x = api::copy_host_vec_to_device(&Arc::new(input))
                    .reshape(&[1, case.cols / 32, 32])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let mut y = api::zeros::<f32>(&[1, case.rows])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let view = x
                    .view(&[1, case.cols / 32, 32])
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                gemv(
                    &weight,
                    &view,
                    (&mut y).partition([1, DEFAULT_GEMV_TILE.rows]),
                    DEFAULT_GEMV_TILE,
                )
                .submit(&mut Eager(&stream))
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let gemv_row: Vec<f32> = y
                    .to_host_vec()
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                for col in 0..case.rows {
                    let magnitude = magnitude[row * case.rows + col];
                    let bound =
                        (2.0 * check::gamma_tilde(case.cols) + BF16_SPLIT_ERROR) * magnitude;
                    prop_assert!(
                        (actual[row * case.rows + col] as f64 - gemv_row[col] as f64).abs()
                            <= bound.max(f64::MIN_POSITIVE)
                    );
                }
            }
            Ok(())
        })
        .unwrap();
}

#[test]
fn omitting_lo_exceeds_bound() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let (m, n, k) = (64, 128, 128);
    let input = vec![1.0 + 1.0 / 512.0; m * k];
    let weights = vec![bf16::from_f32(1.0); n * k];
    let x = api::copy_host_vec_to_device(&Arc::new(input.clone()))
        .reshape(&[m, k])
        .sync_on(&stream)?;
    let w = api::copy_host_vec_to_device(&Arc::new(weights))
        .reshape(&[n, k])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[m, n]).sync_on(&stream)?;
    hi_only::kernel(
        (&mut y).partition([m, n]),
        &x.view(&[m, k])?,
        &w.view(&[n, k])?,
    )
    .sync_on(&stream)?;
    let actual: Vec<f32> = y.to_host_vec().sync_on(&stream)?;
    let expected = k as f64 * input[0] as f64;
    let bound = (check::gamma_tilde(k) + BF16_SPLIT_ERROR) * expected;
    assert!((actual[0] as f64 - expected).abs() > bound);
    let matrix = DenseMatrix::upload(vec![bf16::from_f32(1.0); n * k], n, k, &stream)?;
    let corrected = run(&stream, &matrix, &input, n, k, DEFAULT_GEMM_TILE)?;
    assert!((corrected[0] as f64 - expected).abs() <= bound);
    Ok(())
}

#[test]
fn configurable_tile_matches_default() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let (m, n, k) = (65, 128, 128);
    let input: Vec<f32> = (0..m * k)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) / 17.0)
        .collect();
    let weights: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(((i * 7 % 29) as f32 - 14.0) / 19.0))
        .collect();
    let matrix = DenseMatrix::upload(weights, n, k, &stream)?;
    let default = run(&stream, &matrix, &input, n, k, DEFAULT_GEMM_TILE)?;
    let other = run(
        &stream,
        &matrix,
        &input,
        n,
        k,
        GemmTile {
            m: 32,
            n: 64,
            k: 64,
        },
    )?;
    for (&a, &b) in default.iter().zip(&other) {
        assert!((a - b).abs() <= 2e-4);
    }
    Ok(())
}
