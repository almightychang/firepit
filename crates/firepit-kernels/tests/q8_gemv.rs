use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use firepit_kernels::q8::{DEFAULT_GEMV_TILE, GemvTile, HostQ8Matrix, q8_gemv};
use half::f16;
use std::sync::Arc;

fn fixture(rows: usize, cols: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * cols / 32 * 34);
    for block in 0..rows * cols / 32 {
        let scale = f16::from_f32(((block % 13) + 1) as f32 / 32.0);
        bytes.extend(scale.to_bits().to_le_bytes());
        for i in 0..32 {
            bytes.push((((block * 17 + i * 7) % 31) as i8 - 15) as u8);
        }
    }
    bytes
}

fn run<const B: i32>(
    rows: usize,
    cols: usize,
    tile: GemvTile,
) -> Result<f64, Box<dyn std::error::Error>> {
    let host = HostQ8Matrix::from_gguf_q8_0(&fixture(rows, cols), rows, cols)?;
    let x_host: Vec<f32> = (0..B as usize * cols)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) / 37.0)
        .collect();
    let expected = host.gemv_reference(&x_host)?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let w = host.upload(&stream)?;
    let x = api::copy_host_vec_to_device(&Arc::new(x_host))
        .reshape(&[B as usize, cols / 32, 32])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[B as usize, rows]).sync_on(&stream)?;
    let x_view = x.view(&[B as usize, cols / 32, 32])?;
    q8_gemv(
        &w,
        &x_view,
        (&mut y).partition([B as usize, tile.rows]),
        tile,
    )
    .sync_on(&stream)?;
    let actual: Vec<f32> = y.to_host_vec().sync_on(&stream)?;
    let mut max_relative = 0.0f64;
    for (got, want) in actual.into_iter().zip(expected) {
        let relative = (got as f64 - want).abs() / want.abs().max(1.0);
        max_relative = max_relative.max(relative);
    }
    Ok(max_relative)
}

#[test]
fn small_shapes_all_batches() -> Result<(), Box<dyn std::error::Error>> {
    for error in [
        run::<1>(32, 128, DEFAULT_GEMV_TILE)?,
        run::<2>(32, 128, DEFAULT_GEMV_TILE)?,
        run::<4>(32, 128, DEFAULT_GEMV_TILE)?,
        run::<8>(32, 128, DEFAULT_GEMV_TILE)?,
    ] {
        assert!(error < 2e-4, "relative error {error}");
        println!("relative error {error:.3e}");
    }
    Ok(())
}

#[test]
#[ignore = "large GPU and CPU reference run"]
fn model_shapes() -> Result<(), Box<dyn std::error::Error>> {
    for (rows, cols) in [(10240, 2560), (2560, 6144), (248320, 2560)] {
        let error = run::<1>(
            rows,
            cols,
            GemvTile {
                rows: 32,
                blocks: 8,
            },
        )?;
        assert!(
            error < 2e-4,
            "shape ({cols}, {rows}) relative error {error}"
        );
        println!("({cols}, {rows}) relative error {error:.3e}");
    }
    Ok(())
}
