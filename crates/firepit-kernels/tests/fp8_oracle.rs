mod common;

use common::{check, oracle};
use cuda_async::device_operation::DeviceOp;
use cuda_core::{Stream, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{IntoPartition, PartitionMut, ToHostVec};
use firepit_kernels::gemm::{
    DEFAULT_GEMM_TILE, DEFAULT_GROUPED_GEMM_TILE, fp8_gemm, fp8_grouped_gemm_view,
};
use firepit_kernels::gemv::{DEFAULT_GEMV_TILE, expert_gemv_weight_view, gemv};
use firepit_kernels::moe::{ExpertSelection, MoeDims, PrefillPlan};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{Fp8Matrix, HostFp8Matrix, Weight};
use half::bf16;
use proptest::prelude::*;
use std::sync::Arc;

const BF16_UNIT_ROUNDOFF: f64 = 1.0 / 256.0;
const BF16_SPLIT_ERROR: f64 = BF16_UNIT_ROUNDOFF * BF16_UNIT_ROUNDOFF;

fn within(actual: f32, expected: f64, bound: f64) -> bool {
    (actual as f64 - expected).abs() <= bound.max(f64::MIN_POSITIVE)
}

fn gemm_within_bound(actual: f32, expected: f64, magnitude: f64, cols: usize) -> bool {
    let bound = (check::gamma_tilde(cols + 1) + BF16_SPLIT_ERROR) * magnitude;
    within(actual, expected, bound)
}

fn grouped_gemm_within_bound(actual: f32, expected: f64, magnitude: f64, cols: usize) -> bool {
    let bound = (check::gamma_tilde(cols) + BF16_UNIT_ROUNDOFF + BF16_SPLIT_ERROR) * magnitude;
    within(actual, expected, bound)
}

fn gpu(
    stream: &Arc<Stream>,
    bits: Vec<f8e4m3fn>,
    scales: Vec<bf16>,
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let matrix = HostFp8Matrix::from_parts(bits, scales, rows, cols, 128, 128)?.upload(stream)?;
    let x = api::copy_host_vec_to_device(&Arc::new(input.to_vec()))
        .reshape(&[1, cols / 32, 32])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(stream)?;
    gemv(
        &Weight::Fp8(matrix.clone()),
        &x.view(&[1, cols / 32, 32])?,
        (&mut y).partition([1, DEFAULT_GEMV_TILE.rows]),
        DEFAULT_GEMV_TILE,
    )
    .submit(&mut Eager(stream))?;
    let gemv: Vec<f32> = y.to_host_vec().sync_on(stream)?;

    let data: Vec<f32> = input.iter().copied().cycle().take(64 * cols).collect();
    let x = api::copy_host_vec_to_device(&Arc::new(data))
        .reshape(&[64, cols])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[64, rows]).sync_on(stream)?;
    fp8_gemm(
        &matrix,
        &x.view(&[64, cols])?,
        (&mut y).partition([DEFAULT_GEMM_TILE.m, DEFAULT_GEMM_TILE.n]),
        DEFAULT_GEMM_TILE,
    )
    .submit(&mut Eager(stream))?;
    let gemm: Vec<f32> = y.to_host_vec().sync_on(stream)?;
    Ok((gemv, gemm[..rows].to_vec()))
}

fn grouped_gpu(
    stream: &Arc<Stream>,
    matrix: &Fp8Matrix,
    input: &[f32],
    dims: &MoeDims,
    rows_per_expert: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let cols = input.len();
    let ids: Vec<i32> = (0..64).map(|i| (i % 2) as i32).collect();
    let plan = PrefillPlan::parse(
        &ids,
        &vec![1.0; 64],
        dims,
        DEFAULT_GROUPED_GEMM_TILE.m,
        stream,
    )?;
    let padded = plan.layout().padded_rows();
    let data: Vec<f32> = input.iter().copied().cycle().take(padded * cols).collect();
    let x = api::copy_host_vec_to_device(&Arc::new(data))
        .reshape(&[padded, cols])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[padded, rows_per_expert]).sync_on(stream)?;
    fp8_grouped_gemm_view(
        matrix,
        &x.view(&[padded, cols])?,
        y.slice_mut(&[0..padded, 0..rows_per_expert])?
            .partition([DEFAULT_GROUPED_GEMM_TILE.m, DEFAULT_GROUPED_GEMM_TILE.n]),
        &plan,
        rows_per_expert,
        DEFAULT_GROUPED_GEMM_TILE,
    )
    .submit(&mut Eager(stream))?;
    Ok(y.to_host_vec().sync_on(stream)?)
}

#[test]
fn fp8_gemv_and_gemm_match_dequantized_f64_oracle() {
    let stream = common::stream();
    const VALUES: [(u8, f32); 8] = [
        (0xb8, -1.0),
        (0x38, 1.0),
        (0x30, 0.5),
        (0xc0, -2.0),
        (0x40, 2.0),
        (0x39, 1.125),
        (0x3b, 1.375),
        (0x35, 0.8125),
    ];
    const SCALES: [f32; 3] = [0.8125, 1.203125, 2.25];
    common::runner(24, file!())
        .run(
            &(0usize..2, 0usize..2, any::<u64>()),
            |(n_blocks, k_blocks, seed)| {
                let rows = (n_blocks + 1) * 128;
                let cols = (k_blocks + 1) * 128;
                let bits: Vec<f8e4m3fn> = (0..rows * cols)
                    .map(|i| {
                        f8e4m3fn(VALUES[(i.wrapping_mul(17) ^ seed as usize) % VALUES.len()].0)
                    })
                    .collect();
                let scales: Vec<bf16> = (0..rows / 128 * (cols / 128))
                    .map(|i| bf16::from_f32(SCALES[(i + seed as usize) % SCALES.len()]))
                    .collect();
                let weights: Vec<f32> = (0..rows * cols)
                    .map(|i| {
                        let value = VALUES.iter().find(|entry| entry.0 == bits[i].0).unwrap().1;
                        value * scales[(i / cols / 128) * (cols / 128) + (i % cols / 128)].to_f32()
                    })
                    .collect();
                let input: Vec<f32> = (0..cols)
                    .map(|i| ((i.wrapping_mul(13) ^ seed as usize) % 41) as f32 / 37.0 - 0.5)
                    .collect();
                let (gemv, gemm) = gpu(&stream, bits, scales, &input, rows, cols)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let expected = oracle::dense(&weights, &input, rows, cols);
                let magnitude = oracle::dense_magnitude(&weights, &input, rows, cols);
                prop_assert!(
                    check::inner_product_excess(&gemv, &expected, &magnitude, cols + 1) <= 1.0
                );
                for ((&actual, &value), &sum) in gemm.iter().zip(&expected).zip(&magnitude) {
                    prop_assert!(gemm_within_bound(actual, value, sum, cols));
                }
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn fp8_expert_kernels_match_f64_oracle() {
    let stream = common::stream();
    common::runner(12, file!())
        .run(&(0usize..2, any::<u64>()), |(k_blocks, seed)| {
            let cols = (k_blocks + 1) * 128;
            let rows = 256;
            let dims = MoeDims::new(cols, 128, 128, 2, 1)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let values = [(0x38u8, 1.0f32), (0x39, 1.125), (0xb8, -1.0), (0x30, 0.5)];
            let bits: Vec<f8e4m3fn> = (0..rows * cols)
                .map(|i| f8e4m3fn(values[(i.wrapping_mul(7) ^ seed as usize) % values.len()].0))
                .collect();
            let scales: Vec<bf16> = (0..rows / 128 * (cols / 128))
                .map(|i| bf16::from_f32([0.8125, 1.203125, 2.25, 0.75][(i + seed as usize) % 4]))
                .collect();
            let weights: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let value = values.iter().find(|entry| entry.0 == bits[i].0).unwrap().1;
                    value * scales[(i / cols / 128) * (cols / 128) + (i % cols / 128)].to_f32()
                })
                .collect();
            let matrix = HostFp8Matrix::from_parts(bits, scales, rows, cols, 128, 128)
                .map_err(|e| TestCaseError::fail(e.to_string()))?
                .upload(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let input: Vec<f32> = (0..cols)
                .map(|i| ((i * 13 + seed as usize) % 43) as f32 / 37.0 - 0.5)
                .collect();
            let expected = oracle::dense(&weights, &input, rows, cols);
            let magnitude = oracle::dense_magnitude(&weights, &input, rows, cols);
            let selected = (seed % 2) as usize;
            let selection = ExpertSelection::parse(&[selected as i32], &[1.0], &dims, &stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let x = api::copy_host_vec_to_device(&Arc::new(input.clone()))
                .reshape(&[1, cols / 32, 32])
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let mut y = api::zeros::<f32>(&[1, 128])
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            expert_gemv_weight_view(
                &Weight::Fp8(matrix.clone()),
                &x.view(&[1, cols / 32, 32])
                    .map_err(|e| TestCaseError::fail(e.to_string()))?,
                &selection,
                y.slice_mut(&[0..1, 0..128])
                    .map_err(|e| TestCaseError::fail(e.to_string()))?
                    .partition([1, DEFAULT_GEMV_TILE.rows]),
                128,
                0,
                DEFAULT_GEMV_TILE,
            )
            .submit(&mut Eager(&stream))
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let actual: Vec<f32> = y
                .to_host_vec()
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert!(
                check::inner_product_excess(
                    &actual,
                    &expected[selected * 128..(selected + 1) * 128],
                    &magnitude[selected * 128..(selected + 1) * 128],
                    cols + 1,
                ) <= 1.0
            );

            let actual = grouped_gpu(&stream, &matrix, &input, &dims, 128)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            for expert in 0..2 {
                for row in expert * 32..(expert + 1) * 32 {
                    for output in 0..128 {
                        let index = expert * 128 + output;
                        prop_assert!(grouped_gemm_within_bound(
                            actual[row * 128 + output],
                            expected[index],
                            magnitude[index],
                            cols,
                        ));
                    }
                }
            }
            Ok(())
        })
        .unwrap();
}

#[test]
fn fp8_extreme_finite_values_match_f64_oracle() {
    let stream = common::stream();
    for case in 0..3 {
        common::runner(8, file!())
            .run(&any::<u64>(), |seed| {
                let (scale, encodings, amplitude) = match case {
                    0 => (
                        bf16::from_bits(0x0001),
                        [0x7e, 0xfe, 0x7e, 0xfe],
                        2.0f32.powi(20),
                    ),
                    1 => (bf16::from_bits(0x7f7f), [0x01, 0x81, 0x01, 0x81], 0.5),
                    _ => (
                        bf16::from_f32(1.0 / 128.0),
                        [0x7e, 0xfe, 0x01, 0x81],
                        2.0f32.powi(20),
                    ),
                };
                let (rows, cols) = (128usize, 256usize);
                let bits: Vec<f8e4m3fn> = (0..rows * cols)
                    .map(|i| f8e4m3fn(encodings[(i.wrapping_mul(7) ^ seed as usize) % 4]))
                    .collect();
                let weights: Vec<f32> = bits
                    .iter()
                    .map(|bit| {
                        let value = match bit.0 {
                            0x7e => 448.0,
                            0xfe => -448.0,
                            0x01 => 2.0f32.powi(-9),
                            0x81 => -2.0f32.powi(-9),
                            _ => unreachable!(),
                        };
                        value * scale.to_f32()
                    })
                    .collect();
                let input: Vec<f32> = (0..cols)
                    .map(|i| {
                        if (i + seed as usize) % 3 == 0 {
                            -amplitude
                        } else {
                            amplitude
                        }
                    })
                    .collect();
                let (gemv_output, gemm_output) =
                    gpu(&stream, bits.clone(), vec![scale; 2], &input, rows, cols)
                        .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let expected = oracle::dense(&weights, &input, rows, cols);
                let magnitude = oracle::dense_magnitude(&weights, &input, rows, cols);
                prop_assert!(expected.iter().all(|value| value.is_finite()));
                prop_assert!(gemv_output.iter().all(|value| value.is_finite()));
                prop_assert!(
                    check::inner_product_excess(&gemv_output, &expected, &magnitude, cols + 1)
                        <= 1.0
                );
                for ((&actual, &value), &sum) in gemm_output.iter().zip(&expected).zip(&magnitude) {
                    prop_assert!(gemm_within_bound(actual, value, sum, cols));
                }

                let dims = MoeDims::new(cols, 128, 128, 2, 1)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let selection = ExpertSelection::parse(&[1], &[1.0], &dims, &stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let expert_bits = bits.iter().copied().cycle().take(2 * rows * cols).collect();
                let expert = HostFp8Matrix::from_parts(
                    expert_bits,
                    vec![scale; 4],
                    2 * rows,
                    cols,
                    128,
                    128,
                )
                .map_err(|e| TestCaseError::fail(e.to_string()))?
                .upload(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let x = api::copy_host_vec_to_device(&Arc::new(input.clone()))
                    .reshape(&[1, cols / 32, 32])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let mut y = api::zeros::<f32>(&[1, rows])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                expert_gemv_weight_view(
                    &Weight::Fp8(expert),
                    &x.view(&[1, cols / 32, 32])
                        .map_err(|e| TestCaseError::fail(e.to_string()))?,
                    &selection,
                    y.slice_mut(&[0..1, 0..rows])
                        .map_err(|e| TestCaseError::fail(e.to_string()))?
                        .partition([1, DEFAULT_GEMV_TILE.rows]),
                    rows,
                    0,
                    DEFAULT_GEMV_TILE,
                )
                .submit(&mut Eager(&stream))
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let actual: Vec<f32> = y
                    .to_host_vec()
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                prop_assert!(actual.iter().all(|value| value.is_finite()));
                prop_assert!(
                    check::inner_product_excess(&actual, &expected, &magnitude, cols + 1) <= 1.0
                );
                Ok(())
            })
            .unwrap();
    }
}
