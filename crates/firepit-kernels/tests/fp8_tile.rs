use cuda_core::{Device, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::prelude::DeviceOp;
use cutile::tensor::{IntoPartition, PartitionMut, ToHostVec};
use cutile::tile_kernel::ToHostVecOp;
use firepit_kernels::gemm::{
    DEFAULT_GEMM_TILE, DEFAULT_GROUPED_GEMM_TILE, fp8_gemm, fp8_grouped_gemm_view,
};
use firepit_kernels::gemv::{GemvTile, expert_gemv_weight_view, gemv};
use firepit_kernels::moe::{ExpertSelection, MoeDims, PrefillPlan};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{HostFp8Matrix, Weight};
use half::bf16;
use std::sync::Arc;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn convert_fp8(out: &mut Tensor<f32, { [16] }>, input: &Tensor<f8e4m3fn, { [16] }>) {
        let values: Tile<f8e4m3fn, { [16] }> = input.partition(shape![16]).load([0]);
        let values: Tile<f32, { [16] }> = convert_tile(values);
        out.store(values);
    }
}

#[test]
fn fp8_tile_converts_to_f32() -> Result<(), Box<dyn std::error::Error>> {
    let values = vec![f8e4m3fn(0x38); 16];
    let input = api::copy_host_vec_to_device(&Arc::new(values)).sync()?;
    let mut output = api::zeros::<f32>(&[16]).sync()?;
    kernels::convert_fp8((&mut output).partition([16]), &input).sync()?;
    let actual: Vec<f32> = output.dup().to_host_vec().sync()?;
    assert_eq!(actual, vec![1.0; 16]);
    Ok(())
}

#[test]
fn fp8_gemv_and_gemm_apply_block_scales() -> Result<(), Box<dyn std::error::Error>> {
    let stream = Device::new(0)?.new_stream()?;
    let gemv_tile = GemvTile {
        rows: 16,
        blocks: 2,
    };
    let matrix = HostFp8Matrix::from_parts(
        vec![f8e4m3fn(0x38); 128 * 256],
        vec![bf16::from_f32(0.5), bf16::from_f32(2.0)],
        128,
        256,
        128,
        128,
    )?
    .upload(&stream)?;
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; 256]))
        .reshape(&[1, 256 / 32, 32])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[1, 128]).sync_on(&stream)?;
    let x_view = x.view(&[1, 256 / 32, 32])?;
    gemv(
        &Weight::Fp8(matrix.clone()),
        &x_view,
        (&mut y).partition([1, gemv_tile.rows]),
        gemv_tile,
    )
    .submit(&mut Eager(&stream))?;
    let gemv_output: Vec<f32> = y.dup().to_host_vec().sync_on(&stream)?;
    assert!(
        gemv_output
            .iter()
            .all(|&value| (value - 320.0).abs() < 1e-3)
    );

    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; 64 * 256]))
        .reshape(&[64, 256])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[64, 128]).sync_on(&stream)?;
    let x_view = x.view(&[64, 256])?;
    fp8_gemm(
        &matrix,
        &x_view,
        (&mut y).partition([DEFAULT_GEMM_TILE.m, DEFAULT_GEMM_TILE.n]),
        DEFAULT_GEMM_TILE,
    )
    .submit(&mut Eager(&stream))?;
    let gemm_output: Vec<f32> = y.dup().to_host_vec().sync_on(&stream)?;
    assert!(
        gemm_output
            .iter()
            .all(|&value| (value - 320.0).abs() < 1e-3)
    );
    Ok(())
}

#[test]
fn fp8_expert_gemv_and_grouped_gemm_select_expert() -> Result<(), Box<dyn std::error::Error>> {
    let stream = Device::new(0)?.new_stream()?;
    let gemv_tile = GemvTile {
        rows: 16,
        blocks: 2,
    };
    let dims = MoeDims::new(128, 128, 128, 2, 1)?;
    let matrix = HostFp8Matrix::from_parts(
        vec![f8e4m3fn(0x38); 2 * 128 * 128],
        vec![bf16::from_f32(0.5), bf16::from_f32(2.0)],
        2 * 128,
        128,
        128,
        128,
    )?
    .upload(&stream)?;
    let selection = ExpertSelection::parse(&[1], &[1.0], &dims, &stream)?;
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; 128]))
        .reshape(&[1, 4, 32])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[1, 128]).sync_on(&stream)?;
    expert_gemv_weight_view(
        &Weight::Fp8(matrix.clone()),
        &x.view(&[1, 4, 32])?,
        &selection,
        y.slice_mut(&[0..1, 0..128])?.partition([1, gemv_tile.rows]),
        128,
        0,
        gemv_tile,
    )
    .submit(&mut Eager(&stream))?;
    let actual: Vec<f32> = y.to_host_vec().sync_on(&stream)?;
    assert!(actual.iter().all(|&value| (value - 256.0).abs() < 1e-3));

    let ids: Vec<i32> = (0..64).map(|i| (i % 2) as i32).collect();
    let plan = PrefillPlan::parse(
        &ids,
        &vec![1.0; 64],
        &dims,
        DEFAULT_GROUPED_GEMM_TILE.m,
        &stream,
    )?;
    let rows = plan.layout().padded_rows();
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; rows * 128]))
        .reshape(&[rows, 128])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[rows, 128]).sync_on(&stream)?;
    fp8_grouped_gemm_view(
        &matrix,
        &x.view(&[rows, 128])?,
        y.slice_mut(&[0..rows, 0..128])?
            .partition([DEFAULT_GROUPED_GEMM_TILE.m, DEFAULT_GROUPED_GEMM_TILE.n]),
        &plan,
        128,
        DEFAULT_GROUPED_GEMM_TILE,
    )
    .submit(&mut Eager(&stream))?;
    let actual: Vec<f32> = y.to_host_vec().sync_on(&stream)?;
    assert!(
        actual[..32 * 128]
            .iter()
            .all(|&value| (value - 64.0).abs() < 1e-3)
    );
    assert!(
        actual[32 * 128..]
            .iter()
            .all(|&value| (value - 256.0).abs() < 1e-3)
    );
    Ok(())
}
