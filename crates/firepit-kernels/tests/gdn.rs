mod common;

use common::{check, strategy};
use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::gdn::{GdnDims, conv1d_silu};
use proptest::prelude::*;
use std::sync::Arc;

fn oracle(
    input: &[f32],
    state: &[f32],
    weight: &[f32],
    channels: usize,
    kernel: usize,
) -> Vec<f64> {
    (0..channels)
        .map(|channel| {
            let row = channel * (kernel - 1);
            let taps = channel * kernel;
            let sum: f64 = (0..kernel - 1)
                .map(|t| state[row + t] as f64 * weight[taps + t] as f64)
                .sum::<f64>()
                + input[channel] as f64 * weight[taps + kernel - 1] as f64;
            sum / (1.0 + (-sum).exp())
        })
        .collect()
}

fn gpu_step(
    stream: &Arc<cuda_core::Stream>,
    input: &[f32],
    state: &[f32],
    weight: &[f32],
    kernel: usize,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let channels = input.len();
    let x = api::copy_host_vec_to_device(&Arc::new(input.to_vec())).sync_on(stream)?;
    let mut s = api::copy_host_vec_to_device(&Arc::new(state.to_vec()))
        .reshape(&[channels, kernel - 1])
        .sync_on(stream)?;
    let previous = s.dup().sync_on(stream)?;
    let w = api::copy_host_vec_to_device(&Arc::new(weight.to_vec()))
        .reshape(&[channels, kernel])
        .sync_on(stream)?;
    let mut out = api::zeros::<f32>(&[channels]).sync_on(stream)?;
    conv1d_silu(
        (&mut out).partition([1]),
        &x,
        &previous,
        s.partition([1, (kernel - 1).next_power_of_two()]),
        &w,
        kernel as i32,
    )
    .generics(vec![(kernel - 1).next_power_of_two().to_string()])
    .sync_on(stream)?;
    Ok((
        out.to_host_vec().sync_on(stream)?,
        s.to_host_vec().sync_on(stream)?,
    ))
}

#[test]
fn conv_matches_f64_oracle() {
    let stream = common::stream();
    common::runner(64, file!())
        .run(
            &(
                prop::sample::select(vec![32usize, 64, 96, 128]),
                prop::sample::select(vec![2usize, 3, 4, 7]),
            )
                .prop_flat_map(|(channels, kernel)| {
                    (
                        Just((channels, kernel)),
                        strategy::values(channels),
                        strategy::values(channels * (kernel - 1)),
                        strategy::values(channels * kernel),
                    )
                }),
            |((channels, kernel), input, state, weight)| {
                let (got, updated) = gpu_step(&stream, &input, &state, &weight, kernel)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let expected = oracle(&input, &state, &weight, channels, kernel);
                prop_assert!(check::max_relative(&got, &expected) < 2e-5);
                for channel in 0..channels {
                    let start = channel * (kernel - 1);
                    prop_assert_eq!(
                        &updated[start..start + kernel - 2],
                        &state[start + 1..start + kernel - 1]
                    );
                    prop_assert_eq!(updated[start + kernel - 2], input[channel]);
                }
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn token_steps_match_full_causal_conv() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let (channels, kernel, steps) = (64, 4, 9);
    let weight: Vec<f32> = (0..channels * kernel)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 20.0)
        .collect();
    let inputs: Vec<Vec<f32>> = (0..steps)
        .map(|t| {
            (0..channels)
                .map(|c| ((t * 17 + c * 7) % 31) as f32 / 16.0 - 1.0)
                .collect()
        })
        .collect();
    let mut state = vec![0.0; channels * (kernel - 1)];
    for t in 0..steps {
        let (got, next) = gpu_step(&stream, &inputs[t], &state, &weight, kernel)?;
        let full: Vec<f64> = (0..channels)
            .map(|c| {
                let sum: f64 = (0..kernel)
                    .map(|tap| {
                        let time = t as isize + tap as isize - kernel as isize + 1;
                        if time < 0 {
                            0.0
                        } else {
                            inputs[time as usize][c] as f64 * weight[c * kernel + tap] as f64
                        }
                    })
                    .sum();
                sum / (1.0 + (-sum).exp())
            })
            .collect();
        let error = check::max_relative(&got, &full);
        assert!(
            error < 2e-6,
            "step {t} error={error} got={:?} want={:?} state={:?}",
            &got[..4],
            &full[..4],
            &next[..12]
        );
        for c in 0..channels {
            for slot in 0..kernel - 1 {
                let time = t as isize + slot as isize - kernel as isize + 2;
                let want = if time < 0 {
                    0.0
                } else {
                    inputs[time as usize][c]
                };
                assert_eq!(next[c * (kernel - 1) + slot], want);
            }
        }
        state = next;
    }
    Ok(())
}

#[test]
fn detects_reversed_taps_and_stale_state() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let channels = 32;
    let kernel = 4;
    let input0: Vec<f32> = (0..channels).map(|i| i as f32 / 16.0 + 0.25).collect();
    let input1: Vec<f32> = (0..channels).map(|i| i as f32 / 32.0 - 0.3).collect();
    let weight: Vec<f32> = (0..channels).flat_map(|_| [0.1, 0.4, -0.2, 0.8]).collect();
    let zero = vec![0.0; channels * (kernel - 1)];
    let (first, state) = gpu_step(&stream, &input0, &zero, &weight, kernel)?;
    let reversed: Vec<f32> = weight
        .chunks(kernel)
        .flat_map(|row| row.iter().rev().copied())
        .collect();
    let (wrong_taps, _) = gpu_step(&stream, &input0, &zero, &reversed, kernel)?;
    let (second, _) = gpu_step(&stream, &input1, &state, &weight, kernel)?;
    let (stale, _) = gpu_step(&stream, &input1, &zero, &weight, kernel)?;
    assert!(check::max_abs(&first, &wrong_taps) > 0.01);
    assert!(check::max_abs(&second, &stale) > 0.01);
    println!(
        "reversed_taps_max_abs={:.8e} stale_state_step1_max_abs={:.8e}",
        check::max_abs(&first, &wrong_taps),
        check::max_abs(&second, &stale)
    );
    Ok(())
}

#[test]
fn rejects_invalid_dimensions() {
    assert!(GdnDims::new(2048, 16, 32, 128, 128, 4).is_ok());
    for dims in [
        (0, 16, 32, 128, 128, 4),
        (2048, 0, 32, 128, 128, 4),
        (2048, 16, 32, 128, 128, 1),
    ] {
        assert!(GdnDims::new(dims.0, dims.1, dims.2, dims.3, dims.4, dims.5).is_err());
    }
}
