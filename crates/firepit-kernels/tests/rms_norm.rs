use cutile::api::{self, DeviceOpReshape};
use cutile::prelude::DeviceOp;
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::hybrid::rms_norm;
use std::sync::Arc;

#[test]
fn learned_scale_matches_cpu_for_each_row() {
    const H: usize = 128;
    let input: Vec<f32> = (0..2 * H).map(|i| (i as f32 - 96.0) / 32.0).collect();
    let weight: Vec<f32> = (0..H).map(|i| 0.5 + i as f32 / H as f32).collect();
    let x = api::copy_host_vec_to_device(&Arc::new(input.clone()))
        .reshape(&[2, H])
        .sync()
        .unwrap();
    let gamma = api::copy_host_vec_to_device(&Arc::new(weight.clone()))
        .sync()
        .unwrap();
    let mut output = api::zeros::<f32>(&[2, H]).sync().unwrap();
    rms_norm(
        (&mut output).partition([1, H]),
        &x,
        &gamma,
        1.0 / H as f32,
        1e-5,
    )
    .generics(vec![H.to_string()])
    .sync()
    .unwrap();

    let actual: Vec<f32> = output.dup().to_host_vec().sync().unwrap();
    for row in 0..2 {
        let values = &input[row * H..(row + 1) * H];
        let mean_square = values.iter().map(|v| v * v).sum::<f32>() / H as f32;
        let scale = (mean_square + 1e-5).sqrt().recip();
        for col in 0..H {
            let expected = values[col] * scale * weight[col];
            assert!((actual[row * H + col] - expected).abs() < 1e-5);
        }
    }
}
