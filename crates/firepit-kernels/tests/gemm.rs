mod common;

use common::{check, oracle, strategy};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{IntoPartition, PartitionMut, ToHostVec};
use firepit_kernels::gemm::{
    DEFAULT_GEMM_TILE, GemmTile, fp8_gemm_view, gemm, gemm_view, nvfp4_gemm_view,
};
use firepit_kernels::gemv::{DEFAULT_GEMV_TILE, gemv};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{DenseMatrix, HostFp8Matrix, HostNvfp4Matrix, NVFP4_GROUP, Weight};
use half::bf16;
use proptest::prelude::*;
use std::ops::Range;
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

fn member_tile(rows: usize, cols: usize) -> GemmTile {
    let mut tile = DEFAULT_GEMM_TILE;
    while rows % tile.n != 0 {
        tile.n /= 2;
    }
    while cols % tile.k != 0 {
        tile.k /= 2;
    }
    tile
}

#[test]
fn member_row_views_match_the_stacked_gemm() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let (m, cols, stacked) = (64, 128, 272);
    let members = [0..256, 256..257];
    let byte = |i: usize| ((i * 37 + 11) % 251) as u8;
    let data: Vec<f32> = (0..m * cols)
        .map(|i| (byte(i) as f32 - 125.0) / 64.0)
        .collect();
    let x = api::copy_host_vec_to_device(&Arc::new(data))
        .reshape(&[m, cols])
        .sync_on(&stream)?;
    let x = x.view(&[m, cols])?;
    let weights = [
        Weight::Dense(DenseMatrix::upload(
            (0..stacked * cols)
                .map(|i| bf16::from_f32((byte(i) as f32 - 125.0) / 512.0))
                .collect(),
            stacked,
            cols,
            &stream,
        )?),
        Weight::Fp8(
            HostFp8Matrix::with_row_scales(
                (0..stacked * cols)
                    .map(|i| cuda_core::f8e4m3fn(byte(i) & 0xbf))
                    .collect(),
                (0..stacked)
                    .map(|row| 0.25 + (row % 7) as f32 / 16.0)
                    .collect(),
                stacked,
                cols,
            )?
            .upload(&stream)?,
        ),
        Weight::Nvfp4(
            HostNvfp4Matrix::from_parts(
                (0..stacked * cols / 2)
                    .map(|i| cuda_core::f4e2m1fnx2::from_bits(byte(i)))
                    .collect(),
                (0..stacked * cols / NVFP4_GROUP)
                    .map(|i| cuda_core::f8e4m3fn(0x30 + byte(i) % 16))
                    .collect(),
                0.5,
                stacked,
                cols,
            )?
            .upload(&stream)?,
        ),
    ];
    let launch = |weight: &Weight,
                  y: &mut cutile::tensor::Tensor<f32>,
                  rows: Range<usize>|
     -> Result<(), Box<dyn std::error::Error>> {
        let tile = member_tile(rows.len(), cols);
        let out = y
            .slice_mut(&[0..m, rows.clone()])?
            .partition([tile.m, tile.n]);
        match weight {
            Weight::Dense(w) => gemm_view(w, rows, &x, out, tile).submit(&mut Eager(&stream))?,
            Weight::Fp8(w) => fp8_gemm_view(w, rows, &x, out, tile).submit(&mut Eager(&stream))?,
            Weight::Nvfp4(w) => {
                nvfp4_gemm_view(w, rows, &x, out, tile).submit(&mut Eager(&stream))?
            }
            Weight::Q8(_) => unreachable!(),
        }
        Ok(())
    };
    for weight in &weights {
        let mut whole = api::zeros::<f32>(&[m, stacked]).sync_on(&stream)?;
        launch(weight, &mut whole, 0..stacked)?;
        let mut sliced = api::zeros::<f32>(&[m, stacked]).sync_on(&stream)?;
        for rows in &members {
            launch(weight, &mut sliced, rows.clone())?;
        }
        let whole: Vec<f32> = whole.to_host_vec().sync_on(&stream)?;
        let sliced: Vec<f32> = sliced.to_host_vec().sync_on(&stream)?;
        for row in 0..m {
            let used = row * stacked..row * stacked + members[1].end;
            let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&sliced[used.clone()]), bits(&whole[used]));
        }
    }
    Ok(())
}
