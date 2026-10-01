mod common;

use common::{check, oracle, strategy};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::norm::rms_norm;
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
        .reshape(&[batch, width])
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
