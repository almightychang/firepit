use crate::checkpoint::{Checkpoint, HostBf16Tensor, Stream};
use crate::{LayerKind, ModelConfig, ModelDims, Result, WeightFormat};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::tensor::Tensor;
use firepit_kernels::weights::{HostFp8Matrix, HostNvfp4Matrix, NVFP4_GROUP, Weight};
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
        let json = |name: &str| -> Result<Value> {
            Ok(serde_json::from_reader(File::open(dir.join(name))?)?)
        };
        let modelopt = dir
            .join("hf_quant_config.json")
            .exists()
            .then(|| json("hf_quant_config.json"))
            .transpose()?;
        let config = ModelConfig::new(json("config.json")?, modelopt)?;
        let dims = config.dims(options.max_seq, options.layers)?;
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
                    MixerWeights::Gdn(load_gdn(&checkpoint, &prefix, &dims, &config, &stream)?)
                }
                LayerKind::Attention => MixerWeights::Attention(load_attention(
                    &checkpoint,
                    &prefix,
                    &dims,
                    &config,
                    &stream,
                )?),
            };
            let post_norm = checkpoint.vector(
                &format!("{prefix}.post_attention_layernorm.weight"),
                h,
                1.0,
                &stream,
            )?;
            let moe = load_moe(&checkpoint, &prefix, &dims, &config, &stream)?;
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
            [dims.vocab(), h],
            &config,
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

enum HostWeight {
    Dense(HostBf16Tensor),
    Fp8(HostFp8Matrix),
    Nvfp4(HostNvfp4Matrix),
}

impl HostWeight {
    fn load(
        checkpoint: &Checkpoint,
        name: &str,
        [rows, cols]: [usize; 2],
        format: WeightFormat,
    ) -> Result<Self> {
        let companion = |suffix: &str| name.replace(".weight", suffix);
        let present = [".weight_scale_inv", ".weight_scale", ".weight_scale_2"]
            .map(|suffix| checkpoint.contains(&companion(suffix)));
        let expected = match format {
            WeightFormat::Bf16 => [false, false, false],
            WeightFormat::Fp8Blocks { .. } => [true, false, false],
            WeightFormat::Fp8Tensor => [false, true, false],
            WeightFormat::Nvfp4 => [false, true, true],
        };
        if present != expected {
            return Err(format!("{name}: scale tensors contradict {format:?}").into());
        }
        let shape_error = || format!("{name}: invalid {format:?} data or scale shape");
        match format {
            WeightFormat::Bf16 => {
                let data = checkpoint.tensor_bf16(name)?;
                if data.shape != [rows, cols] {
                    return Err(shape_error().into());
                }
                Ok(Self::Dense(data))
            }
            WeightFormat::Fp8Blocks { block_n, block_k } => {
                let data = checkpoint.tensor_fp8(name)?;
                let scales = checkpoint.tensor_bf16(&companion(".weight_scale_inv"))?;
                if data.shape != [rows, cols] || scales.shape != [rows / block_n, cols / block_k] {
                    return Err(shape_error().into());
                }
                Ok(Self::Fp8(HostFp8Matrix::from_parts(
                    data.data,
                    scales
                        .data
                        .into_iter()
                        .map(|scale| scale.to_f32())
                        .collect(),
                    rows,
                    cols,
                    block_n,
                    block_k,
                )?))
            }
            WeightFormat::Fp8Tensor => {
                let data = checkpoint.tensor_fp8(name)?;
                if data.shape != [rows, cols] {
                    return Err(shape_error().into());
                }
                let scale = checkpoint.scalar_f32(&companion(".weight_scale"))?;
                Ok(Self::Fp8(HostFp8Matrix::from_parts(
                    data.data,
                    vec![scale],
                    rows,
                    cols,
                    rows,
                    cols,
                )?))
            }
            WeightFormat::Nvfp4 => {
                let packed = checkpoint.tensor_nvfp4(name)?;
                let scales = checkpoint.tensor_fp8(&companion(".weight_scale"))?;
                if packed.shape != [rows, cols / 2] || scales.shape != [rows, cols / NVFP4_GROUP] {
                    return Err(shape_error().into());
                }
                Ok(Self::Nvfp4(HostNvfp4Matrix::from_parts(
                    packed.data,
                    scales.data,
                    checkpoint.scalar_f32(&companion(".weight_scale_2"))?,
                    rows,
                    cols,
                )?))
            }
        }
    }

    fn stack(parts: Vec<Self>) -> Result<Self> {
        fn all<T>(parts: Vec<HostWeight>, pick: fn(HostWeight) -> Option<T>) -> Result<Vec<T>> {
            parts
                .into_iter()
                .map(|part| pick(part).ok_or_else(|| "stacked weights mix formats".into()))
                .collect()
        }
        Ok(match parts.first().ok_or("no weights to stack")? {
            Self::Dense(_) => {
                let parts = all(parts, |part| match part {
                    Self::Dense(tensor) => Some(tensor),
                    _ => None,
                })?;
                let cols = parts[0].shape[1];
                if parts.iter().any(|part| part.shape[1] != cols) {
                    return Err("stacked dense weights differ in columns".into());
                }
                Self::Dense(HostBf16Tensor {
                    shape: vec![parts.iter().map(|part| part.shape[0]).sum(), cols],
                    data: parts.into_iter().flat_map(|part| part.data).collect(),
                })
            }
            Self::Fp8(_) => Self::Fp8(HostFp8Matrix::stack(all(parts, |part| match part {
                Self::Fp8(matrix) => Some(matrix),
                _ => None,
            })?)?),
            Self::Nvfp4(_) => {
                Self::Nvfp4(HostNvfp4Matrix::stack(all(parts, |part| match part {
                    Self::Nvfp4(matrix) => Some(matrix),
                    _ => None,
                })?)?)
            }
        })
    }

    fn upload(self, stream: &Stream) -> Result<Weight> {
        Ok(match self {
            Self::Dense(tensor) => Weight::Dense(tensor.upload_matrix(stream)?),
            Self::Fp8(matrix) => Weight::Fp8(matrix.upload(stream)?),
            Self::Nvfp4(matrix) => Weight::Nvfp4(matrix.upload(stream)?),
        })
    }
}

fn host(
    checkpoint: &Checkpoint,
    name: &str,
    shape: [usize; 2],
    config: &ModelConfig,
) -> Result<HostWeight> {
    HostWeight::load(checkpoint, name, shape, config.format(name))
}

fn matrix(
    checkpoint: &Checkpoint,
    name: &str,
    shape: [usize; 2],
    config: &ModelConfig,
    stream: &Stream,
) -> Result<Weight> {
    host(checkpoint, name, shape, config)?.upload(stream)
}

fn load_gdn(
    checkpoint: &Checkpoint,
    prefix: &str,
    dims: &ModelDims,
    config: &ModelConfig,
    stream: &Stream,
) -> Result<GdnWeights> {
    let d = dims.gdn();
    let path = format!("{prefix}.linear_attn");
    let projection = |name: &str, rows: usize| {
        matrix(
            checkpoint,
            &format!("{path}.in_proj_{name}.weight"),
            [rows, dims.hidden()],
            config,
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
            [dims.hidden(), d.value_dim()],
            config,
            &stream,
        )?,
    })
}

fn load_attention(
    checkpoint: &Checkpoint,
    prefix: &str,
    dims: &ModelDims,
    config: &ModelConfig,
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
            [2 * q, dims.hidden()],
            config,
            &stream,
        )?,
        k: matrix(
            checkpoint,
            &format!("{path}.k_proj.weight"),
            [kv, dims.hidden()],
            config,
            &stream,
        )?,
        v: matrix(
            checkpoint,
            &format!("{path}.v_proj.weight"),
            [kv, dims.hidden()],
            config,
            &stream,
        )?,
        q_norm: checkpoint.vector(&format!("{path}.q_norm.weight"), d.head_dim(), 1.0, &stream)?,
        k_norm: checkpoint.vector(&format!("{path}.k_norm.weight"), d.head_dim(), 1.0, &stream)?,
        out: matrix(
            checkpoint,
            &format!("{path}.o_proj.weight"),
            [dims.hidden(), q],
            config,
            &stream,
        )?,
    })
}

fn load_moe(
    checkpoint: &Checkpoint,
    prefix: &str,
    dims: &ModelDims,
    config: &ModelConfig,
    stream: &Stream,
) -> Result<MoeWeights> {
    let d = dims.moe();
    let h = dims.hidden();
    let path = format!("{prefix}.mlp");
    let shared_path = format!("{path}.shared_expert");
    let shared_gate_up = HostWeight::stack(
        ["gate_proj", "up_proj"]
            .map(|projection| {
                host(
                    checkpoint,
                    &format!("{shared_path}.{projection}.weight"),
                    [d.shared_intermediate(), h],
                    config,
                )
            })
            .into_iter()
            .collect::<Result<_>>()?,
    )?
    .upload(stream)?;
    let expert = |expert: usize, projection: &str, shape: [usize; 2]| {
        host(
            checkpoint,
            &format!("{path}.experts.{expert}.{projection}.weight"),
            shape,
            config,
        )
    };
    let (gate_up, down) =
        if config.format(&format!("{path}.experts.0.gate_proj.weight")) == WeightFormat::Bf16 {
            (
                Weight::Dense(checkpoint.matrix_bf16(
                    &format!("{path}.experts.gate_up_proj"),
                    &[d.experts(), 2 * d.intermediate(), h],
                    stream,
                )?),
                Weight::Dense(checkpoint.matrix_bf16(
                    &format!("{path}.experts.down_proj"),
                    &[d.experts(), h, d.intermediate()],
                    stream,
                )?),
            )
        } else {
            let gate_up = (0..d.experts())
                .flat_map(|e| {
                    ["gate_proj", "up_proj"]
                        .map(|projection| expert(e, projection, [d.intermediate(), h]))
                })
                .collect::<Result<_>>()?;
            let down = (0..d.experts())
                .map(|e| expert(e, "down_proj", [h, d.intermediate()]))
                .collect::<Result<_>>()?;
            (
                HostWeight::stack(gate_up)?.upload(stream)?,
                HostWeight::stack(down)?.upload(stream)?,
            )
        };
    Ok(MoeWeights {
        router: matrix(
            checkpoint,
            &format!("{path}.gate.weight"),
            [d.experts(), h],
            config,
            stream,
        )?,
        gate_up,
        down,
        shared_gate_up,
        shared_down: matrix(
            checkpoint,
            &format!("{shared_path}.down_proj.weight"),
            [h, d.shared_intermediate()],
            config,
            stream,
        )?,
        shared_gate: matrix(
            checkpoint,
            &format!("{path}.shared_expert_gate.weight"),
            [1, h],
            config,
            stream,
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
