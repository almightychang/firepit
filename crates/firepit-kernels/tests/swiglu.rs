mod common;

use common::{check, oracle, strategy};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::swiglu::swiglu;
use proptest::prelude::*;
use std::sync::Arc;

fn gpu(
    stream: &Arc<Stream>,
    gate_up: &[f32],
    width: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let batch = gate_up.len() / (2 * width);
    let input = api::copy_host_vec_to_device(&Arc::new(gate_up.to_vec()))
        .reshape(&[batch, 2 * width])
        .sync_on(stream)?;
    let mut output = api::zeros::<f32>(&[batch, width]).sync_on(stream)?;
    swiglu((&mut output).partition([1, width]), &input)
        .generics(vec![width.to_string()])
        .sync_on(stream)?;
    Ok(output.to_host_vec().sync_on(stream)?)
}

#[test]
fn swiglu_matches_f64_oracle() {
    let stream = common::stream();
    common::runner(64, file!())
        .run(&strategy::swiglu_case(), |case| {
            let actual = gpu(&stream, &case.gate_up, case.width)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert_eq!(actual.len(), case.batch * case.width);
            let expected = oracle::swiglu(&case.gate_up, case.width);
            prop_assert!(check::max_relative(&actual, &expected) < 1e-5);
            Ok(())
        })
        .unwrap();
}

#[test]
fn silu_sweep_and_extremes() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let width = 512;
    let mut gate: Vec<f32> = (0..width).map(|i| (i as f32 - 256.0) / 8.0).collect();
    gate[..6].copy_from_slice(&[-1000.0, -100.0, -0.0, 0.0, 100.0, 1000.0]);
    let mut gate_up = gate;
    gate_up.extend(vec![1.0; width]);
    let actual = gpu(&stream, &gate_up, width)?;
    let expected = oracle::swiglu(&gate_up, width);
    let error = check::max_relative(&actual, &expected);
    println!("SiLU sweep max_relative={error:.8e}");
    assert!(error < 1e-5);
    assert!(actual[0].is_sign_negative());
    assert_eq!(actual[4], 100.0);

    gate_up[0] = f32::NAN;
    let nan = gpu(&stream, &gate_up, width)?;
    assert!(nan[0].is_nan());
    Ok(())
}
