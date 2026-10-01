use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor, ToHostVec};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit::{Checkpoint, SafetensorsFile};
use firepit_kernels::attention::{AttentionDims, cache_append, online_attention, output_gate};
use firepit_kernels::dense::{DenseMatrix, dense_gemv};
use firepit_kernels::hybrid::rms_norm;
use firepit_kernels::moe::{MoeDims, combine, expert_gemv, top_k};
use firepit_kernels::q8::{DEFAULT_GEMV_TILE, GemvTile};
use firepit_kernels::rope::{RopeDims, frequencies, generics as rope_generics, rope};
use half::bf16;
use std::path::PathBuf;
use std::sync::Arc;

#[allow(dead_code)]
#[path = "../../firepit-kernels/tests/common/check.rs"]
mod check;

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

#[path = "common/expert.rs"]
mod expert;
use expert::{cpu_expert, gpu_expert, gpu_expert_tensor, load_expert};

fn moe_dims(
    model_dir: &std::path::Path,
) -> Result<firepit_kernels::moe::MoeDims, Box<dyn std::error::Error>> {
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let text = &config["text_config"];
    let get = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
        Ok(text[name].as_u64().ok_or(format!("missing {name}"))? as usize)
    };
    Ok(firepit_kernels::moe::MoeDims::new(
        get("hidden_size")?,
        get("moe_intermediate_size")?,
        get("shared_expert_intermediate_size")?,
        get("num_experts")?,
        get("num_experts_per_tok")?,
    )?)
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
    let model_dir = home("Models/qwen3.6-35b-a3b");
    let dims = moe_dims(&model_dir)?;
    let weights = Checkpoint::open(model_dir)?;
    let refs = SafetensorsFile::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4.safetensors"),
    )?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let mut worst = 0.0f64;
    let mut wrong_order = 0.0f64;
    for layer in 0..4 {
        let expert = load_expert(&weights, layer, &stream, dims)?;
        for step in 0..5 {
            let key = format!("step{step}/layer{layer}/shared_expert");
            let input = refs.tensor(&format!("{key}/in"))?;
            let expected = refs.tensor(&format!("{key}/out"))?;
            assert_eq!(input.shape, [dims.hidden()]);
            assert_eq!(expected.shape, [dims.hidden()]);
            let actual = gpu_expert(&stream, &expert.gate_up, &expert.down, &input.data, dims)?;
            let expected_f64: Vec<f64> = expected.data.iter().map(|&x| x as f64).collect();
            worst = worst.max(relative_max(&actual, &expected_f64));
            if layer == 0 && step == 0 {
                let mut swapped =
                    expert.gate_up_host[dims.shared_intermediate() * dims.hidden()..].to_vec();
                swapped.extend_from_slice(
                    &expert.gate_up_host[..dims.shared_intermediate() * dims.hidden()],
                );
                let swapped = DenseMatrix::upload(
                    swapped,
                    2 * dims.shared_intermediate(),
                    dims.hidden(),
                    &stream,
                )?;
                let wrong = gpu_expert(&stream, &swapped, &expert.down, &input.data, dims)?;
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
    let model_dir = home("Models/qwen3.6-35b-a3b");
    let dims = moe_dims(&model_dir)?;
    let weights = Checkpoint::open(model_dir)?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let gaussian: Vec<f32> = (0..dims.hidden())
        .map(|i| {
            let u = ((i * 37 % 2047) + 1) as f64 / 2048.0;
            let v = ((i * 199 % 2047) + 1) as f64 / 2048.0;
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        })
        .collect();
    let mut worst = 0.0f64;
    for layer in 0..4 {
        let expert = load_expert(&weights, layer, &stream, dims)?;
        let actual = gpu_expert(&stream, &expert.gate_up, &expert.down, &gaussian, dims)?;
        let expected = cpu_expert(&gaussian, &expert, dims);
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
    let q_rope = RopeDims::new(q_heads, D, rotary_dim, theta)?;
    let k_rope = RopeDims::new(kv_heads, D, rotary_dim, theta)?;
    let norm_eps = text["rms_norm_eps"].as_f64().ok_or("no norm eps")? as f32;
    let weights = Checkpoint::open(model_dir)?;
    let refs = SafetensorsFile::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4.safetensors"),
    )?;
    let positions = [0, 1, 2, 3, 4, 100, 4095, 65535, 262143];
    let (cos, sin) = frequencies(&positions, &q_rope);
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
        let (cos, sin) = frequencies(&[step], &q_rope);
        let c = api::copy_host_vec_to_device(&Arc::new(cos))
            .reshape(&[1, D])
            .sync_on(&stream)?;
        let s = api::copy_host_vec_to_device(&Arc::new(sin))
            .reshape(&[1, D])
            .sync_on(&stream)?;
        let mut q = api::zeros::<f32>(&[q_heads, D]).sync_on(&stream)?;
        let mut k = api::zeros::<f32>(&[kv_heads, D]).sync_on(&stream)?;
        rope((&mut q).partition([1, rotary_dim]), &q_norm, &c, &s)
            .generics(rope_generics(&q_rope))
            .sync_on(&stream)?;
        rope((&mut k).partition([1, rotary_dim]), &k_norm, &c, &s)
            .generics(rope_generics(&k_rope))
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
            let (c, s) = frequencies(&[step], &q_rope);
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

fn alternate_attention(
    dims: &AttentionDims,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    gate: &[f32],
    length: usize,
) -> Vec<f32> {
    let (d, qh, kvh) = (dims.head_dim(), dims.q_heads(), dims.kv_heads());
    let mut out = vec![0.0; qh * d];
    for h in 0..qh {
        let kh = h % kvh;
        let scores: Vec<f64> = (0..length)
            .map(|t| {
                let base = (t * kvh + kh) * d;
                (0..d)
                    .map(|i| query[h * d + i] as f64 * keys[base + i] as f64)
                    .sum::<f64>()
                    * dims.scale() as f64
            })
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|&s| (s - maximum).exp()).collect();
        let denominator: f64 = weights.iter().sum();
        for i in 0..d {
            let value: f64 = (0..length)
                .map(|t| weights[t] * values[(t * kvh + kh) * d + i] as f64)
                .sum::<f64>()
                / denominator;
            out[h * d + i] = (value / (1.0 + (-(gate[h * d + i] as f64)).exp())) as f32;
        }
    }
    out
}

#[test]
#[ignore = "needs Qwen3.6 weights and regenerated reference trace"]
fn attention_decode_matches_hf() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = home("Models/qwen3.6-35b-a3b");
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let text = &config["text_config"];
    let parse = |key: &str| -> Result<usize, Box<dyn std::error::Error>> {
        Ok(text[key].as_u64().ok_or(format!("missing {key}"))? as usize)
    };
    let dims = AttentionDims::new(
        parse("max_position_embeddings")?,
        parse("num_attention_heads")?,
        parse("num_key_value_heads")?,
        parse("head_dim")?,
    )?;
    let weights = Checkpoint::open(model_dir)?;
    let refs = SafetensorsFile::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4.safetensors"),
    )?;
    let o = weights.tensor_bf16("model.language_model.layers.3.self_attn.o_proj.weight")?;
    let hidden = parse("hidden_size")?;
    assert_eq!(o.shape, [hidden, dims.q_heads() * dims.head_dim()]);
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let matrix = DenseMatrix::upload(o.data, hidden, dims.q_heads() * dims.head_dim(), &stream)?;
    let mut k_cache =
        api::zeros::<f32>(&[dims.max_seq(), dims.kv_heads(), dims.head_dim()]).sync_on(&stream)?;
    let mut v_cache =
        api::zeros::<f32>(&[dims.max_seq(), dims.kv_heads(), dims.head_dim()]).sync_on(&stream)?;
    let mut history_k = Vec::new();
    let mut history_v = Vec::new();
    let mut worst_input = 0.0f64;
    let mut worst_delta = 0.0f64;
    let mut wrong_scale = 0.0f64;
    let mut wrong_gqa = 0.0f64;
    for step in 0..5 {
        let prefix = format!("step{step}/layer3/attn");
        let q = refs.tensor(&format!("{prefix}/q"))?;
        let k = refs.tensor(&format!("{prefix}/k"))?;
        let v = refs.tensor(&format!("{prefix}/v"))?;
        let gate = refs.tensor(&format!("{prefix}/gate"))?;
        let expected_input = refs.tensor(&format!("{prefix}/o_proj_in"))?;
        let expected_delta = refs.tensor(&format!("{prefix}/delta"))?;
        assert_eq!(q.shape, [dims.q_heads() * dims.head_dim()]);
        assert_eq!(k.shape, [dims.kv_heads() * dims.head_dim()]);
        assert_eq!(v.shape, k.shape);
        let q_device = api::copy_host_vec_to_device(&Arc::new(q.data.clone()))
            .reshape(&[dims.q_heads(), dims.head_dim()])
            .sync_on(&stream)?;
        let k_device = api::copy_host_vec_to_device(&Arc::new(k.data.clone()))
            .reshape(&[dims.kv_heads(), dims.head_dim()])
            .sync_on(&stream)?;
        let v_device = api::copy_host_vec_to_device(&Arc::new(v.data.clone()))
            .reshape(&[dims.kv_heads(), dims.head_dim()])
            .sync_on(&stream)?;
        let g_device = api::copy_host_vec_to_device(&Arc::new(gate.data.clone()))
            .reshape(&[dims.q_heads(), dims.head_dim()])
            .sync_on(&stream)?;
        let position =
            api::copy_host_vec_to_device(&Arc::new(vec![step as i32])).sync_on(&stream)?;
        cache_append(
            k_cache.partition([dims.max_seq(), 1, dims.head_dim()]),
            v_cache.partition([dims.max_seq(), 1, dims.head_dim()]),
            &k_device,
            &v_device,
            &position,
        )
        .generics(dims.cache_generics())
        .sync_on(&stream)?;
        let mut attention =
            api::zeros::<f32>(&[dims.q_heads(), dims.head_dim()]).sync_on(&stream)?;
        online_attention(
            (&mut attention).partition([1, dims.head_dim()]),
            &q_device,
            &k_cache,
            &v_cache,
            &position,
            dims.scale(),
        )
        .generics(dims.attention_generics())
        .sync_on(&stream)?;
        let mut gated = api::zeros::<f32>(&[dims.q_heads(), dims.head_dim()]).sync_on(&stream)?;
        output_gate(
            (&mut gated).partition([1, dims.head_dim()]),
            &attention,
            &g_device,
        )
        .generics(dims.gate_generics())
        .sync_on(&stream)?;
        let actual_input: Vec<f32> = gated.dup().to_host_vec().sync_on(&stream)?;
        let expected_f64: Vec<f64> = expected_input.data.iter().map(|&x| x as f64).collect();
        let input_error = relative_max(&actual_input, &expected_f64);
        worst_input = worst_input.max(input_error);
        let mut delta = api::zeros::<f32>(&[1, hidden]).sync_on(&stream)?;
        dense_gemv(
            &matrix,
            &gated.view(&[1, dims.q_heads() * dims.head_dim() / 32, 32])?,
            (&mut delta).partition([1, DEFAULT_GEMV_TILE.rows]),
            DEFAULT_GEMV_TILE,
        )
        .sync_on(&stream)?;
        let actual_delta: Vec<f32> = delta.to_host_vec().sync_on(&stream)?;
        let expected_f64: Vec<f64> = expected_delta.data.iter().map(|&x| x as f64).collect();
        let delta_error = relative_max(&actual_delta, &expected_f64);
        worst_delta = worst_delta.max(delta_error);
        println!(
            "step{step} o_proj_in max_relative={input_error:.8e} delta max_relative={delta_error:.8e}"
        );
        history_k.extend(k.data.iter().copied());
        history_v.extend(v.data.iter().copied());
        if step == 4 {
            let mut unscaled =
                api::zeros::<f32>(&[dims.q_heads(), dims.head_dim()]).sync_on(&stream)?;
            online_attention(
                (&mut unscaled).partition([1, dims.head_dim()]),
                &q_device,
                &k_cache,
                &v_cache,
                &position,
                1.0,
            )
            .generics(dims.attention_generics())
            .sync_on(&stream)?;
            let mut wrong =
                api::zeros::<f32>(&[dims.q_heads(), dims.head_dim()]).sync_on(&stream)?;
            output_gate(
                (&mut wrong).partition([1, dims.head_dim()]),
                &unscaled,
                &g_device,
            )
            .generics(dims.gate_generics())
            .sync_on(&stream)?;
            wrong_scale = relative_max(
                &wrong.to_host_vec().sync_on(&stream)?,
                &expected_input
                    .data
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
            );
            wrong_gqa = relative_max(
                &alternate_attention(&dims, &q.data, &history_k, &history_v, &gate.data, step + 1),
                &expected_input
                    .data
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
            );
        }
    }
    println!(
        "attention decode HF o_proj_in={worst_input:.8e} delta={worst_delta:.8e} no_scale={wrong_scale:.8e} modulo_gqa={wrong_gqa:.8e}"
    );
    assert!(worst_input < 1e-5 && worst_delta < 1e-5);
    assert!(wrong_scale > 1e-5 && wrong_gqa > 1e-5);
    Ok(())
}

fn expert_outputs(
    stream: &Arc<cuda_core::Stream>,
    dims: MoeDims,
    gate_up: &DenseMatrix<bf16>,
    down: &DenseMatrix<bf16>,
    x: &Tensor<f32>,
    ids: &Tensor<i32>,
) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let mut projected =
        api::zeros::<f32>(&[dims.top_k(), 2 * dims.intermediate()]).sync_on(stream)?;
    expert_gemv(
        gate_up,
        &x.view(&[1, dims.hidden() / 32, 32])?,
        ids,
        projected.partition([1, DEFAULT_GEMV_TILE.rows]),
        dims.experts(),
        2 * dims.intermediate(),
        dims.top_k(),
        0,
        DEFAULT_GEMV_TILE,
    )
    .sync_on(stream)?;
    let mut activated = api::zeros::<f32>(&[dims.top_k(), dims.intermediate()]).sync_on(stream)?;
    firepit_kernels::swiglu::swiglu(activated.partition([1, dims.intermediate()]), &projected)
        .generics(vec![dims.intermediate().to_string()])
        .sync_on(stream)?;
    let mut outputs = api::zeros::<f32>(&[dims.top_k(), dims.hidden()]).sync_on(stream)?;
    expert_gemv(
        down,
        &activated.view(&[dims.top_k(), dims.intermediate() / 32, 32])?,
        ids,
        outputs.partition([1, DEFAULT_GEMV_TILE.rows]),
        dims.experts(),
        dims.hidden(),
        dims.top_k(),
        1,
        DEFAULT_GEMV_TILE,
    )
    .sync_on(stream)?;
    Ok(outputs)
}

#[test]
#[ignore = "needs Qwen3.6 weights and regenerated MoE reference trace"]
fn moe_matches_hf() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = home("Models/qwen3.6-35b-a3b");
    let dims = moe_dims(&model_dir)?;
    let checkpoint = Checkpoint::open(model_dir)?;
    let refs = SafetensorsFile::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4.safetensors"),
    )?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let mut worst = [0.0f64; 4];
    let mut near_ties = 0usize;
    let mut wrong_renorm = 0.0f64;
    let mut wrong_k = 0.0f64;
    for layer in 0..4 {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let router = checkpoint.tensor_bf16(&format!("{prefix}.gate.weight"))?;
        assert_eq!(router.shape, [dims.experts(), dims.hidden()]);
        let router_host = router.data.clone();
        let router = DenseMatrix::upload(router.data, dims.experts(), dims.hidden(), &stream)?;
        let gate_up = checkpoint.tensor_bf16(&format!("{prefix}.experts.gate_up_proj"))?;
        assert_eq!(
            gate_up.shape,
            [dims.experts(), 2 * dims.intermediate(), dims.hidden()]
        );
        let gate_up = DenseMatrix::upload(
            gate_up.data,
            dims.experts() * 2 * dims.intermediate(),
            dims.hidden(),
            &stream,
        )?;
        let down = checkpoint.tensor_bf16(&format!("{prefix}.experts.down_proj"))?;
        assert_eq!(
            down.shape,
            [dims.experts(), dims.hidden(), dims.intermediate()]
        );
        let down = DenseMatrix::upload(
            down.data,
            dims.experts() * dims.hidden(),
            dims.intermediate(),
            &stream,
        )?;
        let shared = load_expert(&checkpoint, layer, &stream, dims)?;
        let shared_gate = checkpoint.tensor_bf16(&format!("{prefix}.shared_expert_gate.weight"))?;
        assert_eq!(shared_gate.shape, [1, dims.hidden()]);
        let shared_gate = DenseMatrix::upload(shared_gate.data, 1, dims.hidden(), &stream)?;
        for step in 0..5 {
            let key = format!("step{step}/layer{layer}/moe");
            let input = refs.tensor(&format!("{key}/in"))?;
            assert_eq!(input.shape, [dims.hidden()]);
            let x = api::copy_host_vec_to_device(&Arc::new(input.data.clone()))
                .reshape(&[1, dims.hidden()])
                .sync_on(&stream)?;
            let mut logits = api::zeros::<f32>(&[1, dims.experts()]).sync_on(&stream)?;
            dense_gemv(
                &router,
                &x.view(&[1, dims.hidden() / 32, 32])?,
                logits.partition([1, DEFAULT_GEMV_TILE.rows]),
                DEFAULT_GEMV_TILE,
            )
            .sync_on(&stream)?;
            let got_logits: Vec<f32> = logits.dup().to_host_vec().sync_on(&stream)?;
            let hf_logits = refs.tensor(&format!("{key}/logits"))?;
            assert_eq!(hf_logits.shape, [dims.experts()]);
            let mut exact = Vec::with_capacity(dims.experts());
            let mut magnitude = Vec::with_capacity(dims.experts());
            for row in router_host.chunks_exact(dims.hidden()) {
                let terms = row
                    .iter()
                    .zip(&input.data)
                    .map(|(&w, &x)| w.to_f32() as f64 * x as f64);
                exact.push(terms.clone().sum());
                magnitude.push(terms.map(f64::abs).sum());
            }
            assert!(
                check::inner_product_excess(&got_logits, &exact, &magnitude, dims.hidden()) <= 1.0
            );
            let logits_error = relative_max(
                &got_logits,
                &hf_logits.data.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            );
            worst[0] = worst[0].max(logits_error);
            assert!(logits_error < 1e-5, "{key} router logits {logits_error}");

            let padded = dims.top_k().next_power_of_two();
            let mut ids = api::zeros::<i32>(&[padded]).sync_on(&stream)?;
            let mut weights = api::zeros::<f32>(&[padded]).sync_on(&stream)?;
            top_k(
                &logits.view(&[dims.experts()])?,
                ids.partition([padded]),
                weights.partition([padded]),
                dims.experts() as i32,
                dims.top_k() as i32,
            )
            .generics(dims.top_k_generics())
            .sync_on(&stream)?;
            let got_ids: Vec<i32> = ids.dup().to_host_vec().sync_on(&stream)?;
            let got_weights: Vec<f32> = weights.dup().to_host_vec().sync_on(&stream)?;
            let hf_ids = refs.tensor(&format!("{key}/indices"))?;
            let hf_weights = refs.tensor(&format!("{key}/weights"))?;
            assert_eq!(hf_ids.shape, [dims.top_k()]);
            assert_eq!(hf_weights.shape, [dims.top_k()]);
            let mut has_near_tie = false;
            for rank in 0..dims.top_k() {
                let expected_id = hf_ids.data[rank] as usize;
                let actual_id = got_ids[rank] as usize;
                if actual_id != expected_id {
                    let gap =
                        (hf_logits.data[actual_id] - hf_logits.data[expected_id]).abs() as f64;
                    let bound = 2.0
                        * check::gamma_tilde(dims.hidden())
                        * (magnitude[actual_id] + magnitude[expected_id]);
                    assert!(
                        gap <= bound,
                        "{key} rank {rank}: expert gap {gap} > {bound}"
                    );
                    near_ties += 1;
                    has_near_tie = true;
                }
            }
            let weight_error = max_error(&got_weights[..dims.top_k()], &hf_weights.data) as f64;
            worst[1] = worst[1].max(weight_error);
            if !has_near_tie {
                assert!(weight_error < 1e-5, "{key} top-k weights {weight_error}");
            }

            let mut hf_ids_padded: Vec<i32> = hf_ids.data.iter().map(|&v| v as i32).collect();
            hf_ids_padded.resize(padded, 0);
            let mut hf_weights_padded = hf_weights.data;
            hf_weights_padded.resize(padded, 0.0);
            let hf_ids_gpu =
                api::copy_host_vec_to_device(&Arc::new(hf_ids_padded)).sync_on(&stream)?;
            let hf_weights_gpu =
                api::copy_host_vec_to_device(&Arc::new(hf_weights_padded)).sync_on(&stream)?;
            let hf_outputs = expert_outputs(&stream, dims, &gate_up, &down, &x, &hf_ids_gpu)?;
            let zero_shared = api::zeros::<f32>(&[1, dims.hidden()]).sync_on(&stream)?;
            let zero_gate = api::zeros::<f32>(&[1, 1]).sync_on(&stream)?;
            let mut routed = api::zeros::<f32>(&[1, dims.hidden()]).sync_on(&stream)?;
            combine(
                routed.partition([1, dims.hidden()]),
                &hf_outputs,
                &hf_weights_gpu,
                &zero_shared,
                &zero_gate,
                dims.top_k() as i32,
            )
            .generics(dims.combine_generics())
            .sync_on(&stream)?;
            let got_routed: Vec<f32> = routed.to_host_vec().sync_on(&stream)?;
            let hf_routed = refs.tensor(&format!("{key}/routed"))?;
            let routed_error = relative_max(
                &got_routed,
                &hf_routed.data.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            );
            worst[2] = worst[2].max(routed_error);
            assert!(routed_error < 1e-5, "{key} routed {routed_error}");

            if layer == 0 && step == 0 {
                let maximum = hf_logits
                    .data
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let denominator: f64 = hf_logits
                    .data
                    .iter()
                    .map(|&v| ((v - maximum) as f64).exp())
                    .sum();
                let mut no_renorm: Vec<f32> = hf_ids
                    .data
                    .iter()
                    .map(|&id| {
                        (((hf_logits.data[id as usize] - maximum) as f64).exp() / denominator)
                            as f32
                    })
                    .collect();
                no_renorm.resize(padded, 0.0);
                let mut top_k_minus_one = hf_weights_gpu.dup().to_host_vec().sync_on(&stream)?;
                top_k_minus_one[dims.top_k() - 1] = 0.0;
                let sum: f32 = top_k_minus_one[..dims.top_k() - 1].iter().sum();
                for weight in &mut top_k_minus_one[..dims.top_k() - 1] {
                    *weight /= sum;
                }
                let wrong_output =
                    |scores: Vec<f32>| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                        let scores =
                            api::copy_host_vec_to_device(&Arc::new(scores)).sync_on(&stream)?;
                        let mut out = api::zeros::<f32>(&[1, dims.hidden()]).sync_on(&stream)?;
                        combine(
                            out.partition([1, dims.hidden()]),
                            &hf_outputs,
                            &scores,
                            &zero_shared,
                            &zero_gate,
                            dims.top_k() as i32,
                        )
                        .generics(dims.combine_generics())
                        .sync_on(&stream)?;
                        Ok(out.to_host_vec().sync_on(&stream)?)
                    };
                let target: Vec<f64> = hf_routed.data.iter().map(|&v| v as f64).collect();
                wrong_renorm = relative_max(&wrong_output(no_renorm)?, &target);
                wrong_k = relative_max(&wrong_output(top_k_minus_one)?, &target);
                assert!(wrong_renorm > 1e-5 && wrong_k > 1e-5);
            }

            let own_outputs = expert_outputs(&stream, dims, &gate_up, &down, &x, &ids)?;
            let shared_output =
                gpu_expert_tensor(&stream, &shared.gate_up, &shared.down, &x, dims)?;
            let mut gate = api::zeros::<f32>(&[1, 1]).sync_on(&stream)?;
            dense_gemv(
                &shared_gate,
                &x.view(&[1, dims.hidden() / 32, 32])?,
                gate.partition([1, 1]),
                GemvTile {
                    rows: 1,
                    blocks: DEFAULT_GEMV_TILE.blocks,
                },
            )
            .sync_on(&stream)?;
            let mut delta = api::zeros::<f32>(&[1, dims.hidden()]).sync_on(&stream)?;
            combine(
                delta.partition([1, dims.hidden()]),
                &own_outputs,
                &weights,
                &shared_output,
                &gate,
                dims.top_k() as i32,
            )
            .generics(dims.combine_generics())
            .sync_on(&stream)?;
            let got_delta: Vec<f32> = delta.to_host_vec().sync_on(&stream)?;
            let hf_delta = refs.tensor(&format!("{key}/delta"))?;
            let delta_error = relative_max(
                &got_delta,
                &hf_delta.data.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            );
            worst[3] = worst[3].max(delta_error);
            assert!(delta_error < 1e-5, "{key} delta {delta_error}");
            println!(
                "layer{layer} step{step} logits={logits_error:.8e} top_weights={weight_error:.8e} routed={routed_error:.8e} delta={delta_error:.8e}"
            );
        }
    }
    println!(
        "MoE HF 4x5 worst logits={:.8e} top_weights={:.8e} routed={:.8e} delta={:.8e} near_ties={near_ties} no_renorm={wrong_renorm:.8e} top_k_minus_one={wrong_k:.8e}",
        worst[0], worst[1], worst[2], worst[3]
    );
    Ok(())
}
