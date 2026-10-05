use crate::checkpoint::{Checkpoint, Stream};
use crate::config::Fp8Config;
use crate::{LayerKind, ModelConfig, ModelDims, Result};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::tensor::Tensor;
use firepit_kernels::weights::{HostFp8Matrix, Weight};
use serde_json::Value;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct LoadOptions {
    pub max_seq: usize,
    pub layers: Option<usize>,
}

pub struct GdnWeights {
    pub qkv: Weight,
    pub z: Weight,
    pub b: Weight,
    pub a: Weight,
    pub conv: Tensor<f32>,
    pub a_log: Tensor<f32>,
    pub dt_bias: Tensor<f32>,
    pub gated_norm: Tensor<f32>,
    pub out: Weight,
}

pub struct AttentionWeights {
    pub q_gate: Weight,
    pub k: Weight,
    pub v: Weight,
    pub q_norm: Tensor<f32>,
    pub k_norm: Tensor<f32>,
    pub out: Weight,
}

pub struct MoeWeights {
    pub router: Weight,
    pub gate_up: Weight,
    pub down: Weight,
    pub shared_gate_up: Weight,
    pub shared_down: Weight,
    pub shared_gate: Weight,
}

pub enum MixerWeights {
    Gdn(GdnWeights),
    Attention(AttentionWeights),
}

pub struct LayerWeights {
    pub input_norm: Tensor<f32>,
    pub mixer: MixerWeights,
    pub post_norm: Tensor<f32>,
    pub moe: MoeWeights,
}

pub struct Model {
    pub(crate) device_ordinal: usize,
    pub(crate) dims: ModelDims,
    pub(crate) embedding: Tensor<f32>,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) final_norm: Tensor<f32>,
    pub(crate) lm_head: Weight,
}

impl Model {
    pub fn dims(&self) -> &ModelDims {
        &self.dims
    }
    pub fn from_checkpoint(
        dir: impl AsRef<Path>,
        options: LoadOptions,
        device: &Arc<Device>,
    ) -> Result<Self> {
        let stream = device.new_stream()?;
        let dir = dir.as_ref();
        let config = ModelConfig::try_from(serde_json::from_reader::<_, Value>(File::open(
            dir.join("config.json"),
        )?)?)?;
        let dims = config.dims(options.max_seq, options.layers)?;
        let fp8 = config.fp8();
        let checkpoint = Checkpoint::open(dir)?;
        let h = dims.hidden();
        let embedding = checkpoint.embedding(
            "model.language_model.embed_tokens.weight",
            dims.vocab(),
            h,
            &stream,
        )?;
        let mut layers = Vec::with_capacity(dims.layers().len());
        for (index, &kind) in dims.layers().iter().enumerate() {
            let prefix = format!("model.language_model.layers.{index}");
            let input_norm =
                checkpoint.vector(&format!("{prefix}.input_layernorm.weight"), h, 1.0, &stream)?;
            let mixer = match kind {
                LayerKind::Gdn => {
                    MixerWeights::Gdn(load_gdn(&checkpoint, &prefix, &dims, fp8, &stream)?)
                }
                LayerKind::Attention => MixerWeights::Attention(load_attention(
                    &checkpoint,
                    &prefix,
                    &dims,
                    fp8,
                    &stream,
                )?),
            };
            let post_norm = checkpoint.vector(
                &format!("{prefix}.post_attention_layernorm.weight"),
                h,
                1.0,
                &stream,
            )?;
            let moe = load_moe(&checkpoint, &prefix, &dims, fp8, &stream)?;
            layers.push(LayerWeights {
                input_norm,
                mixer,
                post_norm,
                moe,
            });
        }
        let final_norm = checkpoint.vector("model.language_model.norm.weight", h, 1.0, &stream)?;
        let lm_head = matrix(
            &checkpoint,
            "lm_head.weight",
            &[dims.vocab(), h],
            fp8,
            &stream,
        )?;
        Ok(Self {
            device_ordinal: stream.device().ordinal(),
            dims,
            embedding,
            layers,
            final_norm,
            lm_head,
        })
    }
}

fn fp8_host(
    checkpoint: &Checkpoint,
    name: &str,
    shape: &[usize],
    config: &Fp8Config,
) -> Result<HostFp8Matrix> {
    if !config.expects_fp8(name) || shape.len() != 2 {
        return Err(format!("{name}: FP8 tensor contradicts configuration").into());
    }
    let data = checkpoint.tensor_fp8(name)?;
    let scales = checkpoint.tensor_bf16(&name.replace(".weight", ".weight_scale_inv"))?;
    if data.shape != shape || scales.shape != [shape[0] / config.block_n, shape[1] / config.block_k]
    {
        return Err(format!("{name}: invalid FP8 data or scale shape").into());
    }
    Ok(HostFp8Matrix::from_parts(
        data.data,
        scales.data,
        shape[0],
        shape[1],
        config.block_n,
        config.block_k,
    )?)
}

fn matrix(
    checkpoint: &Checkpoint,
    name: &str,
    shape: &[usize],
    fp8: Option<&Fp8Config>,
    stream: &Stream,
) -> Result<Weight> {
    let scale_name = name.replace(".weight", ".weight_scale_inv");
    let has_scale = checkpoint.contains(&scale_name);
    if has_scale != fp8.is_some_and(|config| config.expects_fp8(name)) {
        return Err(format!("{name}: FP8 scale presence contradicts configuration").into());
    }
    if has_scale {
        let config = fp8.ok_or("FP8 scale without quantization configuration")?;
        let host = fp8_host(checkpoint, name, shape, config)?;
        Ok(Weight::Fp8(host.upload(stream)?))
    } else {
        Ok(Weight::Dense(checkpoint.matrix_bf16(name, shape, stream)?))
    }
}

fn load_gdn(
    checkpoint: &Checkpoint,
    prefix: &str,
    dims: &ModelDims,
    fp8: Option<&Fp8Config>,
    stream: &Stream,
) -> Result<GdnWeights> {
    let d = dims.gdn();
    let path = format!("{prefix}.linear_attn");
    let projection = |name: &str, rows: usize| {
        matrix(
            checkpoint,
            &format!("{path}.in_proj_{name}.weight"),
            &[rows, dims.hidden()],
            fp8,
            &stream,
        )
    };
    let conv_host = checkpoint.tensor(&format!("{path}.conv1d.weight"))?;
    if conv_host.shape != [d.conv_dim(), 1, d.conv_kernel()] {
        return Err("invalid conv weight shape".into());
    }
    use cutile::api::{self, DeviceOpReshape};
    let conv = api::copy_host_vec_to_device(&Arc::new(conv_host.data))
        .reshape(&[d.conv_dim(), d.conv_kernel()])
        .sync_on(stream)?;
    Ok(GdnWeights {
        qkv: projection("qkv", d.conv_dim())?,
        z: projection("z", d.value_dim())?,
        b: projection("b", d.value_heads())?,
        a: projection("a", d.value_heads())?,
        conv,
        a_log: checkpoint.vector(&format!("{path}.A_log"), d.value_heads(), 0.0, &stream)?,
        dt_bias: checkpoint.vector(&format!("{path}.dt_bias"), d.value_heads(), 0.0, &stream)?,
        gated_norm: checkpoint.vector(
            &format!("{path}.norm.weight"),
            d.value_head_dim(),
            0.0,
            &stream,
        )?,
        out: matrix(
            checkpoint,
            &format!("{path}.out_proj.weight"),
            &[dims.hidden(), d.value_dim()],
            fp8,
            &stream,
        )?,
    })
}

fn load_attention(
    checkpoint: &Checkpoint,
    prefix: &str,
    dims: &ModelDims,
    fp8: Option<&Fp8Config>,
    stream: &Stream,
) -> Result<AttentionWeights> {
    let d = dims.attention();
    let path = format!("{prefix}.self_attn");
    let q = d.q_heads() * d.head_dim();
    let kv = d.kv_heads() * d.head_dim();
    Ok(AttentionWeights {
        q_gate: matrix(
            checkpoint,
            &format!("{path}.q_proj.weight"),
            &[2 * q, dims.hidden()],
            fp8,
            &stream,
        )?,
        k: matrix(
            checkpoint,
            &format!("{path}.k_proj.weight"),
            &[kv, dims.hidden()],
            fp8,
            &stream,
        )?,
        v: matrix(
            checkpoint,
            &format!("{path}.v_proj.weight"),
            &[kv, dims.hidden()],
            fp8,
            &stream,
        )?,
        q_norm: checkpoint.vector(&format!("{path}.q_norm.weight"), d.head_dim(), 1.0, &stream)?,
        k_norm: checkpoint.vector(&format!("{path}.k_norm.weight"), d.head_dim(), 1.0, &stream)?,
        out: matrix(
            checkpoint,
            &format!("{path}.o_proj.weight"),
            &[dims.hidden(), q],
            fp8,
            &stream,
        )?,
    })
}

fn load_moe(
    checkpoint: &Checkpoint,
    prefix: &str,
    dims: &ModelDims,
    fp8: Option<&Fp8Config>,
    stream: &Stream,
) -> Result<MoeWeights> {
    let d = dims.moe();
    let path = format!("{prefix}.mlp");
    let shared_path = format!("{path}.shared_expert");
    let shared_gate_up = if let Some(config) = fp8 {
        let mut gate = fp8_host(
            checkpoint,
            &format!("{shared_path}.gate_proj.weight"),
            &[d.shared_intermediate(), dims.hidden()],
            config,
        )?;
        let up = fp8_host(
            checkpoint,
            &format!("{shared_path}.up_proj.weight"),
            &[d.shared_intermediate(), dims.hidden()],
            config,
        )?;
        gate.bits.extend(up.bits);
        gate.scales.extend(up.scales);
        Weight::Fp8(
            HostFp8Matrix::from_parts(
                gate.bits,
                gate.scales,
                2 * d.shared_intermediate(),
                dims.hidden(),
                config.block_n,
                config.block_k,
            )?
            .upload(stream)?,
        )
    } else {
        let gate_up = checkpoint.stacked_bf16(
            &format!("{shared_path}.gate_proj.weight"),
            &format!("{shared_path}.up_proj.weight"),
            d.shared_intermediate(),
            dims.hidden(),
        )?;
        Weight::Dense(gate_up.upload_matrix(stream)?)
    };
    let (gate_up, down) = if let Some(config) = fp8 {
        let mut gate_up_bits = Vec::new();
        let mut gate_up_scales = Vec::new();
        let mut down_bits = Vec::new();
        let mut down_scales = Vec::new();
        for expert in 0..d.experts() {
            let path = format!("{path}.experts.{expert}");
            for projection in ["gate_proj", "up_proj"] {
                let host = fp8_host(
                    checkpoint,
                    &format!("{path}.{projection}.weight"),
                    &[d.intermediate(), dims.hidden()],
                    config,
                )?;
                gate_up_bits.extend(host.bits);
                gate_up_scales.extend(host.scales);
            }
            let host = fp8_host(
                checkpoint,
                &format!("{path}.down_proj.weight"),
                &[dims.hidden(), d.intermediate()],
                config,
            )?;
            down_bits.extend(host.bits);
            down_scales.extend(host.scales);
        }
        let gate_up = HostFp8Matrix::from_parts(
            gate_up_bits,
            gate_up_scales,
            d.experts() * 2 * d.intermediate(),
            dims.hidden(),
            config.block_n,
            config.block_k,
        )?
        .upload(stream)?;
        let down = HostFp8Matrix::from_parts(
            down_bits,
            down_scales,
            d.experts() * dims.hidden(),
            d.intermediate(),
            config.block_n,
            config.block_k,
        )?
        .upload(stream)?;
        (Weight::Fp8(gate_up), Weight::Fp8(down))
    } else {
        (
            Weight::Dense(checkpoint.matrix_bf16(
                &format!("{path}.experts.gate_up_proj"),
                &[d.experts(), 2 * d.intermediate(), dims.hidden()],
                stream,
            )?),
            Weight::Dense(checkpoint.matrix_bf16(
                &format!("{path}.experts.down_proj"),
                &[d.experts(), dims.hidden(), d.intermediate()],
                stream,
            )?),
        )
    };
    Ok(MoeWeights {
        router: matrix(
            checkpoint,
            &format!("{path}.gate.weight"),
            &[d.experts(), dims.hidden()],
            fp8,
            &stream,
        )?,
        gate_up,
        down,
        shared_gate_up,
        shared_down: matrix(
            checkpoint,
            &format!("{shared_path}.down_proj.weight"),
            &[dims.hidden(), d.shared_intermediate()],
            fp8,
            &stream,
        )?,
        shared_gate: matrix(
            checkpoint,
            &format!("{path}.shared_expert_gate.weight"),
            &[1, dims.hidden()],
            fp8,
            &stream,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_used_mib() -> Result<usize> {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()?;
        if !output.status.success() {
            return Err("nvidia-smi failed".into());
        }
        Ok(std::str::from_utf8(&output.stdout)?.trim().parse()?)
    }

    #[test]
    #[ignore = "needs local Qwen3.6 checkpoint and GPU"]
    fn loads_prefix_four() -> Result<()> {
        let dir = Path::new(env!("HOME")).join("Models/qwen3.6-35b-a3b");
        let device = Device::new(0)?;
        let before = gpu_used_mib()?;
        let model = Model::from_checkpoint(
            dir,
            LoadOptions {
                max_seq: 5,
                layers: Some(4),
            },
            &device,
        )?;
        let after = gpu_used_mib()?;
        println!(
            "GPU memory before={before} MiB after={after} MiB delta={} MiB",
            after.saturating_sub(before)
        );
        assert_eq!(model.layers.len(), 4);
        assert_eq!(
            model.dims.layers(),
            &[
                LayerKind::Gdn,
                LayerKind::Gdn,
                LayerKind::Gdn,
                LayerKind::Attention
            ]
        );
        Ok(())
    }
}
