use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit::{Checkpoint, SafetensorsFile};
use firepit_kernels::dense::{DenseMatrix, dense_gemv};
use firepit_kernels::hybrid::rms_norm;
use firepit_kernels::q8::DEFAULT_GEMV_TILE;
use firepit_kernels::swiglu::swiglu;
use half::bf16;
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

const EXPERT_H: usize = 2048;
const EXPERT_I: usize = 512;

struct Expert {
    gate_up: DenseMatrix<bf16>,
    down: DenseMatrix<bf16>,
    gate_up_host: Vec<bf16>,
    down_host: Vec<bf16>,
}

fn load_expert(
    checkpoint: &Checkpoint,
    layer: usize,
    stream: &Arc<cuda_core::Stream>,
) -> Result<Expert, Box<dyn std::error::Error>> {
    let name = |projection| {
        format!("model.language_model.layers.{layer}.mlp.shared_expert.{projection}_proj.weight")
    };
    let gate = checkpoint.tensor_bf16(&name("gate"))?;
    let up = checkpoint.tensor_bf16(&name("up"))?;
    let down = checkpoint.tensor_bf16(&name("down"))?;
    assert_eq!(gate.shape, [EXPERT_I, EXPERT_H]);
    assert_eq!(up.shape, [EXPERT_I, EXPERT_H]);
    assert_eq!(down.shape, [EXPERT_H, EXPERT_I]);
    let mut gate_up_host = gate.data;
    gate_up_host.extend(up.data);
    let gate_up = DenseMatrix::upload(gate_up_host.clone(), 2 * EXPERT_I, EXPERT_H, stream)?;
    let down_matrix = DenseMatrix::upload(down.data.clone(), EXPERT_H, EXPERT_I, stream)?;
    Ok(Expert {
        gate_up,
        down: down_matrix,
        gate_up_host,
        down_host: down.data,
    })
}

fn gpu_expert(
    stream: &Arc<cuda_core::Stream>,
    gate_up: &DenseMatrix<bf16>,
    down: &DenseMatrix<bf16>,
    input: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let x = api::copy_host_vec_to_device(&Arc::new(input.to_vec()))
        .reshape(&[1, EXPERT_H])
        .sync_on(stream)?;
    let mut projected = api::zeros::<f32>(&[1, 2 * EXPERT_I]).sync_on(stream)?;
    dense_gemv(
        gate_up,
        &x.view(&[1, EXPERT_H / 32, 32])?,
        (&mut projected).partition([1, DEFAULT_GEMV_TILE.rows]),
        DEFAULT_GEMV_TILE,
    )
    .sync_on(stream)?;
    let mut activated = api::zeros::<f32>(&[1, EXPERT_I]).sync_on(stream)?;
    swiglu((&mut activated).partition([1, EXPERT_I]), &projected)
        .generics(vec![EXPERT_I.to_string()])
        .sync_on(stream)?;
    let mut output = api::zeros::<f32>(&[1, EXPERT_H]).sync_on(stream)?;
    dense_gemv(
        down,
        &activated.view(&[1, EXPERT_I / 32, 32])?,
        (&mut output).partition([1, DEFAULT_GEMV_TILE.rows]),
        DEFAULT_GEMV_TILE,
    )
    .sync_on(stream)?;
    Ok(output.to_host_vec().sync_on(stream)?)
}

fn cpu_expert(input: &[f32], expert: &Expert) -> Vec<f64> {
    let projected: Vec<f64> = expert
        .gate_up_host
        .chunks_exact(EXPERT_H)
        .map(|row| {
            row.iter()
                .zip(input)
                .map(|(&w, &x)| w.to_f32() as f64 * x as f64)
                .sum()
        })
        .collect();
    let activated: Vec<f64> = projected[..EXPERT_I]
        .iter()
        .zip(&projected[EXPERT_I..])
        .map(|(&gate, &up)| gate / (1.0 + (-gate).exp()) * up)
        .collect();
    expert
        .down_host
        .chunks_exact(EXPERT_I)
        .map(|row| {
            row.iter()
                .zip(&activated)
                .map(|(&w, &x)| w.to_f32() as f64 * x)
                .sum()
        })
        .collect()
}

fn relative_max(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let max_ref = expected.iter().map(|x| x.abs()).fold(0.0, f64::max);
    actual
        .iter()
        .zip(expected)
        .map(|(&got, &want)| (got as f64 - want).abs())
        .fold(0.0, f64::max)
        / max_ref
}

#[test]
#[ignore = "needs Qwen3.6 weights and regenerated reference trace"]
fn shared_expert_matches_hf() -> Result<(), Box<dyn std::error::Error>> {
    let weights = Checkpoint::open(home("Models/qwen3.6-35b-a3b"))?;
    let refs = SafetensorsFile::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4.safetensors"),
    )?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let mut worst = 0.0f64;
    let mut wrong_order = 0.0f64;
    for layer in 0..4 {
        let expert = load_expert(&weights, layer, &stream)?;
        for step in 0..5 {
            let key = format!("step{step}/layer{layer}/shared_expert");
            let input = refs.tensor(&format!("{key}/in"))?;
            let expected = refs.tensor(&format!("{key}/out"))?;
            assert_eq!(input.shape, [EXPERT_H]);
            assert_eq!(expected.shape, [EXPERT_H]);
            let actual = gpu_expert(&stream, &expert.gate_up, &expert.down, &input.data)?;
            let expected_f64: Vec<f64> = expected.data.iter().map(|&x| x as f64).collect();
            worst = worst.max(relative_max(&actual, &expected_f64));
            if layer == 0 && step == 0 {
                let mut swapped = expert.gate_up_host[EXPERT_I * EXPERT_H..].to_vec();
                swapped.extend_from_slice(&expert.gate_up_host[..EXPERT_I * EXPERT_H]);
                let swapped = DenseMatrix::upload(swapped, 2 * EXPERT_I, EXPERT_H, &stream)?;
                let wrong = gpu_expert(&stream, &swapped, &expert.down, &input.data)?;
                wrong_order = relative_max(&wrong, &expected_f64);
            }
        }
    }
    println!("shared_expert HF 20 max_relative={worst:.8e} swapped_gate_up={wrong_order:.8e}");
    assert!(worst <= 1e-5);
    assert!(wrong_order > 1e-5);
    Ok(())
}

#[test]
#[ignore = "needs Qwen3.6 weights"]
fn shared_expert_gaussian_matches_f64() -> Result<(), Box<dyn std::error::Error>> {
    let weights = Checkpoint::open(home("Models/qwen3.6-35b-a3b"))?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let gaussian: Vec<f32> = (0..EXPERT_H)
        .map(|i| {
            let u = ((i * 37 % 2047) + 1) as f64 / 2048.0;
            let v = ((i * 199 % 2047) + 1) as f64 / 2048.0;
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        })
        .collect();
    let mut worst = 0.0f64;
    for layer in 0..4 {
        let expert = load_expert(&weights, layer, &stream)?;
        let actual = gpu_expert(&stream, &expert.gate_up, &expert.down, &gaussian)?;
        let expected = cpu_expert(&gaussian, &expert);
        worst = worst.max(relative_max(&actual, &expected));
    }
    println!("shared_expert Gaussian f64 max_relative={worst:.8e}");
    assert!(worst <= 1e-5);
    Ok(())
}
