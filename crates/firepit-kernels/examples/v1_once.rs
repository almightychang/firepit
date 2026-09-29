use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::IntoPartition;
use firepit_kernels::q8::{HostQ8Matrix, q8_gemv};
use half::f16;
use std::sync::Arc;

fn run<const B: i32>(rows: usize, cols: usize, stream: &Arc<cuda_core::Stream>) -> Result<(), Box<dyn std::error::Error>> {
    let host = HostQ8Matrix {
        qs: (0..rows * cols).map(|i| ((i * 17 % 31) as i8) - 15).collect(),
        scales: (0..rows * cols / 32).map(|i| f16::from_f32(((i % 13) + 1) as f32 / 32.0)).collect(),
        rows,
        cols,
    };
    let w = host.upload(stream)?;
    let x_host: Vec<f32> = (0..B as usize * cols).map(|i| (i % 37) as f32 / 37.0).collect();
    let x = api::copy_host_vec_to_device(&Arc::new(x_host)).reshape(&[B as usize, cols / 32, 32]).sync_on(stream)?;
    for _ in 0..2 {
        let y = api::zeros::<f32>(&[B as usize, rows]).sync_on(stream)?.partition([B as usize, 64]);
        q8_gemv::<B, 64, 8>(&w, &x, y).sync_on(stream)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    run::<1>(248320, 2560, &stream)?;
    run::<8>(248320, 2560, &stream)?;
    Ok(())
}
