use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit::{Checkpoint, SafetensorsFile};
use firepit_kernels::dense::{DenseMatrix, dense_gemv};
use firepit_kernels::hybrid::rms_norm;
use firepit_kernels::q8::DEFAULT_GEMV_TILE;
use std::path::PathBuf;
use std::sync::Arc;

fn home(path: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(path)
}

#[test]
#[ignore = "needs Qwen3.6 weights in ~/Models and refs/qwen36-prefix4.safetensors"]
fn embedding_rows_match_hf_residual0() -> Result<(), Box<dyn std::error::Error>> {
    let weights = Checkpoint::open(home("Models/qwen3.6-35b-a3b"))?;
    let refs = SafetensorsFile::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4.safetensors"),
    )?;
    let tokens: Vec<usize> = serde_json::from_str(refs.metadata("tokens").ok_or("no tokens")?)?;
    for (step, &token) in tokens.iter().enumerate() {
        let row = weights.row("model.language_model.embed_tokens.weight", token)?;
        let expected = refs.tensor(&format!("step{step}/residual0"))?;
        assert_eq!(row.shape, expected.shape);
        assert_eq!(row.data, expected.data, "step {step} token {token}");
    }
    Ok(())
}

#[test]
#[ignore = "needs Qwen3.6 weights in ~/Models and refs/qwen36-prefix4.safetensors"]
fn zero_layer_exit_matches_hf() -> Result<(), Box<dyn std::error::Error>> {
    const H: usize = 2048;
    const EPS: f32 = 1e-6;
    let weights = Checkpoint::open(home("Models/qwen3.6-35b-a3b"))?;
    let refs = SafetensorsFile::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4.safetensors"),
    )?;
    let tokens: Vec<usize> = serde_json::from_str(refs.metadata("tokens").ok_or("no tokens")?)?;
    assert_eq!(tokens.len(), 5);
    let norm = weights.tensor("model.language_model.norm.weight")?;
    assert_eq!(norm.shape, [H]);
    let head = weights.tensor_bf16("lm_head.weight")?;
    assert_eq!(head.shape, [248320, H]);
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let gamma =
        api::copy_host_vec_to_device(&Arc::new(norm.data.iter().map(|x| 1.0 + x).collect()))
            .sync_on(&stream)?;
    let wrong_gamma = api::copy_host_vec_to_device(&Arc::new(norm.data)).sync_on(&stream)?;
    let head = DenseMatrix::upload(head.data, 248320, H, &stream)?;
    let mut max_norm = 0.0f32;
    let mut max_logits = 0.0f32;
    let mut wrong_scale_error = 0.0f32;
    for step in 0..tokens.len() {
        let residual = refs.tensor(&format!("step{step}/residual0"))?;
        let expected_norm = refs.tensor(&format!("step{step}/exit_norm"))?;
        let expected_logits = refs.tensor(&format!("step{step}/logits"))?;
        let x = api::copy_host_vec_to_device(&Arc::new(residual.data))
            .reshape(&[1, H])
            .sync_on(&stream)?;
        let mut normalized = api::zeros::<f32>(&[1, H]).sync_on(&stream)?;
        rms_norm(
            (&mut normalized).partition([1, H]),
            &x,
            &gamma,
            1.0 / H as f32,
            EPS,
        )
        .generics(vec![H.to_string()])
        .sync_on(&stream)?;
        let actual_norm: Vec<f32> = normalized.dup().to_host_vec().sync_on(&stream)?;
        max_norm = max_norm.max(max_error(&actual_norm, &expected_norm.data));
        let mut wrong = api::zeros::<f32>(&[1, H]).sync_on(&stream)?;
        rms_norm(
            (&mut wrong).partition([1, H]),
            &x,
            &wrong_gamma,
            1.0 / H as f32,
            EPS,
        )
        .generics(vec![H.to_string()])
        .sync_on(&stream)?;
        let wrong: Vec<f32> = wrong.to_host_vec().sync_on(&stream)?;
        wrong_scale_error = wrong_scale_error.max(max_error(&wrong, &expected_norm.data));

        let mut logits = api::zeros::<f32>(&[1, 248320]).sync_on(&stream)?;
        let x_view = normalized.view(&[1, H / 32, 32])?;
        dense_gemv(
            &head,
            &x_view,
            (&mut logits).partition([1, DEFAULT_GEMV_TILE.rows]),
            DEFAULT_GEMV_TILE,
        )
        .sync_on(&stream)?;
        let actual_logits: Vec<f32> = logits.to_host_vec().sync_on(&stream)?;
        max_logits = max_logits.max(max_error(&actual_logits, &expected_logits.data));
        let actual_top = argmax(&actual_logits);
        let expected_top = argmax(&expected_logits.data);
        println!("step{step} top-1 HF={expected_top} Rust={actual_top}");
        assert_eq!(actual_top, expected_top);
    }
    println!(
        "exit_norm max_abs={max_norm:.8e} logits max_abs={max_logits:.8e} wrong_scale max_abs={wrong_scale_error:.8e}"
    );
    assert!(max_norm <= 1e-5);
    assert!(max_logits <= 1e-3);
    assert!(wrong_scale_error > 1e-5);
    Ok(())
}

fn max_error(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}
