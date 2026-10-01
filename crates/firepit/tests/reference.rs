use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit::{Checkpoint, SafetensorsFile};
use firepit_kernels::dense::{DenseMatrix, dense_gemv};
use firepit_kernels::hybrid::rms_norm;
use firepit_kernels::q8::DEFAULT_GEMV_TILE;
use firepit_kernels::rope::{frequencies, generics as rope_generics, rope};
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

#[test]
#[ignore = "needs Qwen3.6 weights and regenerated reference trace"]
fn attention_projection_and_rope_match_hf() -> Result<(), Box<dyn std::error::Error>> {
    const H: usize = 2048;
    const D: usize = 256;
    let model_dir = home("Models/qwen3.6-35b-a3b");
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let text = &config["text_config"];
    assert_eq!(text["hidden_size"].as_u64(), Some(H as u64));
    assert_eq!(text["head_dim"].as_u64(), Some(D as u64));
    let q_heads = text["num_attention_heads"].as_u64().ok_or("no q heads")? as usize;
    let kv_heads = text["num_key_value_heads"].as_u64().ok_or("no kv heads")? as usize;
    let theta = text["rope_parameters"]["rope_theta"]
        .as_f64()
        .ok_or("no rope theta")? as f32;
    let rotary_dim = (D as f64
        * text["rope_parameters"]["partial_rotary_factor"]
            .as_f64()
            .ok_or("no rotary factor")?) as usize;
    let norm_eps = text["rms_norm_eps"].as_f64().ok_or("no norm eps")? as f32;
    let weights = Checkpoint::open(model_dir)?;
    let refs = SafetensorsFile::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4.safetensors"),
    )?;
    let positions = [0, 1, 2, 3, 4, 100, 4095, 65535, 262143];
    let (cos, sin) = frequencies(&positions, theta, D, rotary_dim);
    let cos_ref = refs.tensor("rope/cos")?;
    let sin_ref = refs.tensor("rope/sin")?;
    assert_eq!(
        refs.tensor("rope/positions")?.data,
        positions.map(|p| p as f32)
    );
    assert_eq!(cos_ref.shape, [positions.len(), rotary_dim]);
    assert_eq!(sin_ref.shape, [positions.len(), rotary_dim]);
    for (row, position) in positions.iter().enumerate() {
        let start = row * D;
        let reference = row * rotary_dim;
        println!(
            "rope position={position} cos_max_abs={:.8e} sin_max_abs={:.8e}",
            max_error(
                &cos[start..start + rotary_dim],
                &cos_ref.data[reference..reference + rotary_dim]
            ),
            max_error(
                &sin[start..start + rotary_dim],
                &sin_ref.data[reference..reference + rotary_dim]
            )
        );
        assert!(
            max_error(
                &cos[start..start + rotary_dim],
                &cos_ref.data[reference..reference + rotary_dim]
            ) < 1e-6
        );
        assert!(
            max_error(
                &sin[start..start + rotary_dim],
                &sin_ref.data[reference..reference + rotary_dim]
            ) < 1e-6
        );
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let prefix = "model.language_model.layers.3.self_attn";
    let load = |name: &str, rows| -> Result<_, Box<dyn std::error::Error>> {
        let t = weights.tensor_bf16(&format!("{prefix}.{name}_proj.weight"))?;
        assert_eq!(t.shape, [rows, H]);
        Ok(DenseMatrix::upload(t.data, rows, H, &stream)?)
    };
    let q_weights = load("q", q_heads * D * 2)?;
    let k_weights = load("k", kv_heads * D)?;
    let v_weights = load("v", kv_heads * D)?;
    let norm_weight = |name: &str| -> Result<_, Box<dyn std::error::Error>> {
        let t = weights.tensor(&format!("{prefix}.{name}_norm.weight"))?;
        assert_eq!(t.shape, [D]);
        Ok(
            api::copy_host_vec_to_device(&Arc::new(t.data.into_iter().map(|w| 1.0 + w).collect()))
                .sync_on(&stream)?,
        )
    };
    let q_gamma = norm_weight("q")?;
    let k_gamma = norm_weight("k")?;
    let mut worst = [0.0f64; 4];
    let mut wrong_pair = 0.0f64;
    let mut wrong_full = 0.0f64;
    for step in 0..5 {
        let key = format!("step{step}/layer3/attn");
        let input = refs.tensor(&format!("{key}/in"))?;
        assert_eq!(input.shape, [H]);
        let x = api::copy_host_vec_to_device(&Arc::new(input.data))
            .reshape(&[1, H])
            .sync_on(&stream)?;
        let x_view = x.view(&[1, H / 32, 32])?;
        let project = |matrix: &DenseMatrix<bf16>, rows| -> Result<_, Box<dyn std::error::Error>> {
            let mut output = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
            dense_gemv(
                matrix,
                &x_view,
                (&mut output).partition([1, DEFAULT_GEMV_TILE.rows]),
                DEFAULT_GEMV_TILE,
            )
            .sync_on(&stream)?;
            Ok(output)
        };
        let q_proj = project(&q_weights, q_heads * D * 2)?;
        let k_proj = project(&k_weights, kv_heads * D)?;
        let v_proj = project(&v_weights, kv_heads * D)?;
        let q_proj_host: Vec<f32> = q_proj.dup().to_host_vec().sync_on(&stream)?;
        let gate: Vec<f32> = q_proj_host
            .chunks_exact(2 * D)
            .flat_map(|row| row[D..].iter().copied())
            .collect();
        let q_in = q_proj.view(&[q_heads, 2 * D])?;
        let gate_view = q_in.slice(&[0..q_heads, D..2 * D])?;
        assert_eq!(gate_view.shape(), &[q_heads as i32, D as i32]);
        let q_in = q_in.slice(&[0..q_heads, 0..D])?;
        let mut q_norm = api::zeros::<f32>(&[q_heads, D]).sync_on(&stream)?;
        rms_norm(
            (&mut q_norm).partition([1, D]),
            &q_in,
            &q_gamma,
            1.0 / D as f32,
            norm_eps,
        )
        .generics(vec![D.to_string()])
        .sync_on(&stream)?;
        let k_in = k_proj.view(&[kv_heads, D])?;
        let mut k_norm = api::zeros::<f32>(&[kv_heads, D]).sync_on(&stream)?;
        rms_norm(
            (&mut k_norm).partition([1, D]),
            &k_in,
            &k_gamma,
            1.0 / D as f32,
            norm_eps,
        )
        .generics(vec![D.to_string()])
        .sync_on(&stream)?;
        let (cos, sin) = frequencies(&[step], theta, D, rotary_dim);
        let c = api::copy_host_vec_to_device(&Arc::new(cos))
            .reshape(&[1, D])
            .sync_on(&stream)?;
        let s = api::copy_host_vec_to_device(&Arc::new(sin))
            .reshape(&[1, D])
            .sync_on(&stream)?;
        let mut q = api::zeros::<f32>(&[q_heads, D]).sync_on(&stream)?;
        let mut k = api::zeros::<f32>(&[kv_heads, D]).sync_on(&stream)?;
        rope((&mut q).partition([1, rotary_dim]), &q_norm, &c, &s)
            .generics(rope_generics(q_heads, D, rotary_dim))
            .sync_on(&stream)?;
        rope((&mut k).partition([1, rotary_dim]), &k_norm, &c, &s)
            .generics(rope_generics(kv_heads, D, rotary_dim))
            .sync_on(&stream)?;
        let got = [
            q.to_host_vec().sync_on(&stream)?,
            k.to_host_vec().sync_on(&stream)?,
            v_proj.to_host_vec().sync_on(&stream)?,
            gate,
        ];
        for (i, name) in ["q", "k", "v", "gate"].iter().enumerate() {
            let expected = refs.tensor(&format!("{key}/{name}"))?;
            let expected_f64: Vec<f64> = expected.data.iter().map(|&v| v as f64).collect();
            let error = relative_max(&got[i], &expected_f64);
            worst[i] = worst[i].max(error);
            println!("step{step} {name} max_relative={error:.8e}");
        }
        if step == 1 {
            let before: Vec<f32> = q_norm.to_host_vec().sync_on(&stream)?;
            let expected = refs.tensor(&format!("{key}/q"))?;
            let (c, s) = frequencies(&[step], theta, D, rotary_dim);
            let mut adjacent = before.clone();
            let mut full = before.clone();
            for head in 0..q_heads {
                for i in 0..rotary_dim {
                    let pair = if i % 2 == 0 { i + 1 } else { i - 1 };
                    adjacent[head * D + i] = before[head * D + i] * c[i]
                        + if i % 2 == 0 {
                            -before[head * D + pair]
                        } else {
                            before[head * D + pair]
                        } * s[i];
                }
                for i in rotary_dim..D {
                    let pair = if i < D / 2 { i + D / 2 } else { i - D / 2 };
                    let angle = step as f32 * theta.powf(-((i % (D / 2)) as f32) / (D / 2) as f32);
                    full[head * D + i] = before[head * D + i] * angle.cos()
                        + if i < D / 2 {
                            -before[head * D + pair]
                        } else {
                            before[head * D + pair]
                        } * angle.sin();
                }
            }
            let target: Vec<f64> = expected.data.iter().map(|&v| v as f64).collect();
            wrong_pair = relative_max(&adjacent, &target);
            wrong_full = relative_max(&full, &target);
        }
    }
    println!(
        "attention HF max_relative q={:.8e} k={:.8e} v={:.8e} gate={:.8e} adjacent={wrong_pair:.8e} full={wrong_full:.8e}",
        worst[0], worst[1], worst[2], worst[3]
    );
    assert!(worst.iter().all(|&x| x < 1e-5));
    assert!(wrong_pair > 1e-5 && wrong_full > 1e-5);
    Ok(())
}
