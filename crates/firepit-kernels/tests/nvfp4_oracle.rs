mod common;

use common::{check, oracle};
use cuda_async::device_operation::DeviceOp;
use cuda_core::{Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{IntoPartition, PartitionMut, ToHostVec};
use firepit_kernels::gemm::{
    DEFAULT_GEMM_TILE, DEFAULT_GROUPED_GEMM_TILE, nvfp4_gemm, nvfp4_grouped_gemm_view,
};
use firepit_kernels::gemv::{DEFAULT_GEMV_TILE, expert_gemv_weight_view, gemv};
use firepit_kernels::moe::{ExpertSelection, MoeDims, PrefillPlan};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{HostNvfp4Matrix, NVFP4_GROUP, Weight};
use proptest::prelude::*;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const BF16_SPLIT_ERROR: f64 = (1.0 / 256.0) * (1.0 / 256.0);
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const PART_ROWS: usize = 128;

fn e2m1(code: u8) -> f32 {
    let magnitude = E2M1[(code & 0x7) as usize];
    if code & 0x8 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn e4m3(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let fraction = (bits & 0x07) as f32;
    if exponent == 0 {
        sign * fraction * 2.0f32.powi(-9)
    } else {
        sign * (1.0 + fraction / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    }
}

struct Case {
    codes: Vec<u8>,
    scales: Vec<u8>,
    globals: [f32; 2],
    cols: usize,
}

impl Case {
    fn rows(&self) -> usize {
        self.codes.len() / self.cols
    }

    fn host(&self) -> Result<HostNvfp4Matrix> {
        let part = |index: usize| -> Result<HostNvfp4Matrix> {
            let codes = &self.codes[index * PART_ROWS * self.cols..][..PART_ROWS * self.cols];
            let scales = &self.scales[index * PART_ROWS * self.cols / NVFP4_GROUP..]
                [..PART_ROWS * self.cols / NVFP4_GROUP];
            Ok(HostNvfp4Matrix::from_parts(
                codes
                    .chunks_exact(2)
                    .map(|pair| f4e2m1fnx2::from_nibbles(pair[0], pair[1]))
                    .collect(),
                scales.iter().map(|&bits| f8e4m3fn(bits)).collect(),
                self.globals[index],
                PART_ROWS,
                self.cols,
            )?)
        };
        Ok(HostNvfp4Matrix::stack(vec![part(0)?, part(1)?])?)
    }

    fn block_scaled(&self) -> Vec<f32> {
        self.codes
            .iter()
            .enumerate()
            .map(|(i, &code)| e2m1(code) * e4m3(self.scales[i / NVFP4_GROUP]))
            .collect()
    }

    fn expected(&self, input: &[f32]) -> (Vec<f64>, Vec<f64>) {
        let weights = self.block_scaled();
        let scale = |values: Vec<f64>| -> Vec<f64> {
            values
                .iter()
                .enumerate()
                .map(|(row, &value)| value * self.globals[row / PART_ROWS] as f64)
                .collect()
        };
        (
            scale(oracle::dense(&weights, input, self.rows(), self.cols)),
            scale(oracle::dense_magnitude(
                &weights,
                input,
                self.rows(),
                self.cols,
            )),
        )
    }
}

fn gemm_within_bound(actual: &[f32], expected: &[f64], magnitude: &[f64], cols: usize) -> bool {
    actual
        .iter()
        .zip(expected)
        .zip(magnitude)
        .all(|((&a, &e), &m)| {
            let bound = (check::gamma_tilde(cols + 1) + BF16_SPLIT_ERROR) * m;
            (a as f64 - e).abs() <= bound.max(f64::MIN_POSITIVE)
        })
}

fn dense_gpu(stream: &Arc<Stream>, case: &Case, input: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let (rows, cols) = (case.rows(), case.cols);
    let matrix = Weight::Nvfp4(case.host()?.upload(stream)?);
    let x = api::copy_host_vec_to_device(&Arc::new(input.to_vec()))
        .reshape(&[1, cols / 32, 32])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(stream)?;
    gemv(
        &matrix,
        &x.view(&[1, cols / 32, 32])?,
        (&mut y).partition([1, DEFAULT_GEMV_TILE.rows]),
        DEFAULT_GEMV_TILE,
    )
    .submit(&mut Eager(stream))?;
    let gemv: Vec<f32> = y.to_host_vec().sync_on(stream)?;

    let Weight::Nvfp4(nvfp4) = &matrix else {
        unreachable!()
    };
    let data: Vec<f32> = input.iter().copied().cycle().take(64 * cols).collect();
    let x = api::copy_host_vec_to_device(&Arc::new(data))
        .reshape(&[64, cols])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[64, rows]).sync_on(stream)?;
    nvfp4_gemm(
        nvfp4,
        &x.view(&[64, cols])?,
        (&mut y).partition([DEFAULT_GEMM_TILE.m, DEFAULT_GEMM_TILE.n]),
        DEFAULT_GEMM_TILE,
    )
    .submit(&mut Eager(stream))?;
    let gemm: Vec<f32> = y.to_host_vec().sync_on(stream)?;
    Ok((gemv, gemm[..rows].to_vec()))
}

fn expert_gpu(
    stream: &Arc<Stream>,
    case: &Case,
    input: &[f32],
    selected: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let cols = case.cols;
    let dims = MoeDims::new(cols, PART_ROWS, PART_ROWS, 2, 1)?;
    let matrix = Weight::Nvfp4(case.host()?.upload(stream)?);
    let selection = ExpertSelection::parse(&[selected as i32], &[1.0], &dims, stream)?;
    let x = api::copy_host_vec_to_device(&Arc::new(input.to_vec()))
        .reshape(&[1, cols / 32, 32])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[1, PART_ROWS]).sync_on(stream)?;
    expert_gemv_weight_view(
        &matrix,
        &x.view(&[1, cols / 32, 32])?,
        &selection,
        y.slice_mut(&[0..1, 0..PART_ROWS])?
            .partition([1, DEFAULT_GEMV_TILE.rows]),
        PART_ROWS,
        0,
        DEFAULT_GEMV_TILE,
    )
    .submit(&mut Eager(stream))?;
    let gemv: Vec<f32> = y.to_host_vec().sync_on(stream)?;

    let Weight::Nvfp4(nvfp4) = &matrix else {
        unreachable!()
    };
    let ids: Vec<i32> = (0..64).map(|i| (i % 2) as i32).collect();
    let plan = PrefillPlan::parse(
        &ids,
        &vec![1.0; 64],
        &dims,
        DEFAULT_GROUPED_GEMM_TILE.m,
        stream,
    )?;
    let padded = plan.layout().padded_rows();
    let data: Vec<f32> = input.iter().copied().cycle().take(padded * cols).collect();
    let x = api::copy_host_vec_to_device(&Arc::new(data))
        .reshape(&[padded, cols])
        .sync_on(stream)?;
    let mut y = api::zeros::<f32>(&[padded, PART_ROWS]).sync_on(stream)?;
    nvfp4_grouped_gemm_view(
        nvfp4,
        &x.view(&[padded, cols])?,
        y.slice_mut(&[0..padded, 0..PART_ROWS])?
            .partition([DEFAULT_GROUPED_GEMM_TILE.m, DEFAULT_GROUPED_GEMM_TILE.n]),
        &plan,
        PART_ROWS,
        DEFAULT_GROUPED_GEMM_TILE,
    )
    .submit(&mut Eager(stream))?;
    Ok((gemv, y.to_host_vec().sync_on(stream)?))
}

fn check_case(
    stream: &Arc<Stream>,
    case: &Case,
    input: &[f32],
    selected: usize,
) -> std::result::Result<(), TestCaseError> {
    let fail = |e: Box<dyn std::error::Error>| TestCaseError::fail(e.to_string());
    let cols = case.cols;
    let (expected, magnitude) = case.expected(input);
    prop_assert!(expected.iter().all(|value| value.is_finite()));

    let (gemv, gemm) = dense_gpu(stream, case, input).map_err(fail)?;
    prop_assert!(check::inner_product_excess(&gemv, &expected, &magnitude, cols + 2) <= 1.0);
    prop_assert!(gemm_within_bound(&gemm, &expected, &magnitude, cols));

    let (expert, grouped) = expert_gpu(stream, case, input, selected).map_err(fail)?;
    let rows = selected * PART_ROWS..(selected + 1) * PART_ROWS;
    prop_assert!(
        check::inner_product_excess(&expert, &expected[rows.clone()], &magnitude[rows], cols + 2)
            <= 1.0
    );
    let tile_rows = DEFAULT_GROUPED_GEMM_TILE.m;
    for (index, row) in grouped.chunks_exact(PART_ROWS).enumerate() {
        let rows = (index / tile_rows) * PART_ROWS..(index / tile_rows + 1) * PART_ROWS;
        prop_assert!(gemm_within_bound(
            row,
            &expected[rows.clone()],
            &magnitude[rows],
            cols
        ));
    }
    Ok(())
}

#[test]
fn nvfp4_kernels_match_dequantized_f64_oracle() {
    let stream = common::stream();
    const SCALES: [u8; 6] = [0x00, 0x01, 0x30, 0x38, 0x3b, 0x7e];
    const GLOBALS: [f32; 3] = [1.0, 3.2152449e-5, 0.75];
    common::runner(16, file!())
        .run(&(0usize..2, any::<u64>()), |(k_blocks, seed)| {
            let cols = (k_blocks + 1) * 128;
            let rows = 2 * PART_ROWS;
            let seed = seed as usize;
            let case = Case {
                codes: (0..rows * cols)
                    .map(|i| ((i.wrapping_mul(29) ^ seed) % 16) as u8)
                    .collect(),
                scales: (0..rows * cols / NVFP4_GROUP)
                    .map(|i| SCALES[(i.wrapping_mul(7) ^ seed) % SCALES.len()])
                    .collect(),
                globals: [GLOBALS[seed % 3], GLOBALS[(seed / 3) % 3]],
                cols,
            };
            let input: Vec<f32> = (0..cols)
                .map(|i| ((i.wrapping_mul(13) ^ seed) % 41) as f32 / 37.0 - 0.5)
                .collect();
            check_case(&stream, &case, &input, seed % 2)
        })
        .unwrap();
}

#[test]
fn nvfp4_extreme_finite_values_match_f64_oracle() {
    let stream = common::stream();
    for case in 0..3 {
        common::runner(4, file!())
            .run(&any::<u64>(), |seed| {
                let seed = seed as usize;
                let (codes, scales, globals, amplitude) = match case {
                    0 => (
                        [0x7, 0xf, 0x0, 0x8],
                        [0x7e, 0x7e],
                        [2.0f32.powi(-100), 1.0],
                        1.0,
                    ),
                    1 => (
                        [0x1, 0x9, 0x1, 0x9],
                        [0x01, 0x01],
                        [2.0f32.powi(100), 1.0],
                        2.0f32.powi(20),
                    ),
                    _ => (
                        [0x7, 0x9, 0xf, 0x1],
                        [0x7e, 0x01],
                        [1.0, 0.75],
                        2.0f32.powi(20),
                    ),
                };
                let cols = 256;
                let rows = 2 * PART_ROWS;
                let case = Case {
                    codes: (0..rows * cols)
                        .map(|i| codes[(i.wrapping_mul(7) ^ seed) % 4])
                        .collect(),
                    scales: (0..rows * cols / NVFP4_GROUP)
                        .map(|i| scales[(i ^ seed) % 2])
                        .collect(),
                    globals,
                    cols,
                };
                let input: Vec<f32> = (0..cols)
                    .map(|i| {
                        if (i + seed) % 3 == 0 {
                            -amplitude
                        } else {
                            amplitude
                        }
                    })
                    .collect();
                check_case(&stream, &case, &input, seed % 2)
            })
            .unwrap();
    }
}
