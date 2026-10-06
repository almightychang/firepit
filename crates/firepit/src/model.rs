use crate::checkpoint::{Checkpoint, HostBf16Tensor, Stream};
use crate::{LayerKind, ModelConfig, ModelDims, Result, WeightFormat};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::tensor::Tensor;
use firepit_kernels::weights::{HostFp8Matrix, HostNvfp4Matrix, NVFP4_GROUP, Weight};
use safetensors::Dtype;
use serde_json::Value;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::str::FromStr;
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
        Self::from_checkpoint_with_overrides(dir, options, &[], device)
    }

    pub fn from_checkpoint_with_overrides(
        dir: impl AsRef<Path>,
        options: LoadOptions,
        overrides: &[WeightOverride],
        device: &Arc<Device>,
    ) -> Result<Self> {
        let stream = device.new_stream()?;
        let sources = Sources::open(dir.as_ref(), overrides)?;
        let dims = sources.base.config.dims(options.max_seq, options.layers)?;
        let checkpoint = sources.base();
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
                LayerKind::Gdn => MixerWeights::Gdn(load_gdn(&sources, &prefix, &dims, &stream)?),
                LayerKind::Attention => {
                    MixerWeights::Attention(load_attention(&sources, &prefix, &dims, &stream)?)
                }
            };
            let post_norm = checkpoint.vector(
                &format!("{prefix}.post_attention_layernorm.weight"),
                h,
                1.0,
                &stream,
            )?;
            let moe = load_moe(&sources, &prefix, &dims, &stream)?;
            layers.push(LayerWeights {
                input_norm,
                mixer,
                post_norm,
                moe,
            });
        }
        let final_norm = checkpoint.vector("model.language_model.norm.weight", h, 1.0, &stream)?;
        let lm_head = sources.matrix("lm_head.weight", [dims.vocab(), h], &stream)?;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightGroup {
    LmHead,
    Projections,
}

impl WeightGroup {
    fn of(weight: &str) -> Option<Self> {
        let projection = [
            ".linear_attn.in_proj_",
            ".linear_attn.out_proj.",
            ".self_attn.q_proj.",
            ".self_attn.k_proj.",
            ".self_attn.v_proj.",
            ".self_attn.o_proj.",
        ];
        if weight == "lm_head.weight" {
            Some(Self::LmHead)
        } else if projection.iter().any(|part| weight.contains(part)) {
            Some(Self::Projections)
        } else {
            None
        }
    }
}

impl FromStr for WeightGroup {
    type Err = String;

    fn from_str(name: &str) -> std::result::Result<Self, String> {
        match name {
            "lm_head" => Ok(Self::LmHead),
            "projections" => Ok(Self::Projections),
            _ => Err(format!(
                "unknown weight group {name}; expected lm_head or projections"
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct WeightOverride {
    pub group: WeightGroup,
    pub dir: PathBuf,
}

impl FromStr for WeightOverride {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, String> {
        let (group, dir) = value
            .split_once('=')
            .ok_or_else(|| format!("override {value} must be GROUP=DIR"))?;
        Ok(Self {
            group: group.parse()?,
            dir: PathBuf::from(dir),
        })
    }
}

struct Source {
    checkpoint: Checkpoint,
    config: ModelConfig,
}

impl Source {
    fn open(dir: &Path) -> Result<Self> {
        let json = |name: &str| -> Result<Value> {
            Ok(serde_json::from_reader(File::open(dir.join(name))?)?)
        };
        let modelopt = dir
            .join("hf_quant_config.json")
            .exists()
            .then(|| json("hf_quant_config.json"))
            .transpose()?;
        Ok(Self {
            checkpoint: Checkpoint::open(dir)?,
            config: ModelConfig::new(json("config.json")?, modelopt)?,
        })
    }
}

struct Sources {
    base: Source,
    overrides: Vec<(WeightGroup, Source)>,
}

impl Sources {
    fn open(dir: &Path, overrides: &[WeightOverride]) -> Result<Self> {
        let mut sources = Self {
            base: Source::open(dir)?,
            overrides: Vec::with_capacity(overrides.len()),
        };
        for WeightOverride { group, dir } in overrides {
            if sources.overrides.iter().any(|(seen, _)| seen == group) {
                return Err(format!("{group:?} is overridden twice").into());
            }
            sources.overrides.push((*group, Source::open(dir)?));
        }
        Ok(sources)
    }

    fn base(&self) -> &Checkpoint {
        &self.base.checkpoint
    }

    fn source(&self, weight: &str) -> &Source {
        WeightGroup::of(weight)
            .and_then(|group| self.overrides.iter().find(|(g, _)| *g == group))
            .map_or(&self.base, |(_, source)| source)
    }

    fn format(&self, weight: &str) -> WeightFormat {
        self.source(weight).config.format(weight)
    }

    fn host(&self, name: &str, shape: [usize; 2]) -> Result<HostWeight> {
        let source = self.source(name);
        HostWeight::load(&source.checkpoint, name, shape, source.config.format(name))
    }

    fn matrix(&self, name: &str, shape: [usize; 2], stream: &Stream) -> Result<Weight> {
        self.host(name, shape)?.upload(stream)
    }
}

fn check_companions(checkpoint: &Checkpoint, name: &str, format: WeightFormat) -> Result<()> {
    let present = [".weight_scale_inv", ".weight_scale", ".weight_scale_2"]
        .map(|suffix| checkpoint.contains(&name.replace(".weight", suffix)));
    let expected = match format {
        WeightFormat::Bf16 => [false, false, false],
        WeightFormat::Fp8Blocks { .. } => [true, false, false],
        WeightFormat::Fp8Tensor => [false, true, false],
        WeightFormat::Nvfp4 => [false, true, true],
    };
    if present != expected {
        return Err(format!("{name}: scale tensors contradict {format:?}").into());
    }
    Ok(())
}

pub struct WeightReport {
    pub name: String,
    pub format: WeightFormat,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

pub fn verify_checkpoint(dir: &Path) -> Result<Vec<WeightReport>> {
    let source = Source::open(dir)?;
    let dims = source.config.dims(1, None)?;
    let mut names: Vec<&str> = source
        .checkpoint
        .names()
        .filter(|name| name.ends_with(".weight"))
        .collect();
    names.sort_unstable();
    let report = names
        .into_iter()
        .map(|name| {
            let format = source.config.format(name);
            check_companions(&source.checkpoint, name, format)?;
            let (dtype, shape) = source.checkpoint.stored(name)?;
            let stored_as_format = match format {
                WeightFormat::Bf16 => !matches!(dtype, Dtype::U8 | Dtype::F8_E4M3),
                WeightFormat::Fp8Blocks { .. } | WeightFormat::Fp8Tensor => dtype == Dtype::F8_E4M3,
                WeightFormat::Nvfp4 => dtype == Dtype::U8,
            };
            if !stored_as_format {
                return Err(format!("{name}: {dtype:?} data contradicts {format:?}").into());
            }
            Ok(WeightReport {
                name: name.to_owned(),
                format,
                dtype,
                shape,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let lm_head = report
        .iter()
        .find(|weight| weight.name == "lm_head.weight")
        .ok_or("checkpoint has no lm_head.weight")?;
    let stored_cols = match lm_head.format {
        WeightFormat::Nvfp4 => dims.hidden() / 2,
        _ => dims.hidden(),
    };
    if lm_head.shape != [dims.vocab(), stored_cols] {
        return Err(format!("lm_head.weight has shape {:?}", lm_head.shape).into());
    }
    Ok(report)
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
        check_companions(checkpoint, name, format)?;
        let companion = |suffix: &str| name.replace(".weight", suffix);
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
                Ok(Self::Fp8(HostFp8Matrix::with_row_scales(
                    data.data,
                    vec![scale; rows],
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

fn load_gdn(
    sources: &Sources,
    prefix: &str,
    dims: &ModelDims,
    stream: &Stream,
) -> Result<GdnWeights> {
    let checkpoint = sources.base();
    let d = dims.gdn();
    let path = format!("{prefix}.linear_attn");
    let projection = |name: &str, rows: usize| {
        sources.matrix(
            &format!("{path}.in_proj_{name}.weight"),
            [rows, dims.hidden()],
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
        out: sources.matrix(
            &format!("{path}.out_proj.weight"),
            [dims.hidden(), d.value_dim()],
            &stream,
        )?,
    })
}

fn load_attention(
    sources: &Sources,
    prefix: &str,
    dims: &ModelDims,
    stream: &Stream,
) -> Result<AttentionWeights> {
    let checkpoint = sources.base();
    let d = dims.attention();
    let path = format!("{prefix}.self_attn");
    let q = d.q_heads() * d.head_dim();
    let kv = d.kv_heads() * d.head_dim();
    Ok(AttentionWeights {
        q_gate: sources.matrix(
            &format!("{path}.q_proj.weight"),
            [2 * q, dims.hidden()],
            &stream,
        )?,
        k: sources.matrix(
            &format!("{path}.k_proj.weight"),
            [kv, dims.hidden()],
            &stream,
        )?,
        v: sources.matrix(
            &format!("{path}.v_proj.weight"),
            [kv, dims.hidden()],
            &stream,
        )?,
        q_norm: checkpoint.vector(&format!("{path}.q_norm.weight"), d.head_dim(), 1.0, &stream)?,
        k_norm: checkpoint.vector(&format!("{path}.k_norm.weight"), d.head_dim(), 1.0, &stream)?,
        out: sources.matrix(
            &format!("{path}.o_proj.weight"),
            [dims.hidden(), q],
            &stream,
        )?,
    })
}

fn load_moe(
    sources: &Sources,
    prefix: &str,
    dims: &ModelDims,
    stream: &Stream,
) -> Result<MoeWeights> {
    let checkpoint = sources.base();
    let d = dims.moe();
    let (h, i) = (dims.hidden(), d.intermediate());
    let path = format!("{prefix}.mlp");
    let routed_format = sources.format(&format!("{path}.experts.0.gate_proj.weight"));
    let shared_format = sources.format(&format!("{path}.shared_expert.gate_proj.weight"));
    if routed_format != shared_format {
        return Err(format!(
            "{path}: shared expert is {shared_format:?} but routed experts are {routed_format:?}"
        )
        .into());
    }
    let shared = |projection: &str, shape: [usize; 2]| {
        sources.host(&format!("{path}.shared_expert.{projection}.weight"), shape)
    };
    let expert = |expert: usize, projection: &str, shape: [usize; 2]| {
        sources.host(
            &format!("{path}.experts.{expert}.{projection}.weight"),
            shape,
        )
    };
    let fused = |name: &str, [experts, rows, cols]: [usize; 3]| -> Result<HostWeight> {
        let name = format!("{path}.experts.{name}");
        let tensor = checkpoint.tensor_bf16(&name)?;
        if tensor.shape != [experts, rows, cols] {
            return Err(format!("{name}: expected [{experts}, {rows}, {cols}]").into());
        }
        Ok(HostWeight::Dense(HostBf16Tensor {
            shape: vec![experts * rows, cols],
            data: tensor.data,
        }))
    };
    let (mut gate_up, mut down) = if routed_format == WeightFormat::Bf16 {
        (
            vec![fused("gate_up_proj", [d.experts(), 2 * i, h])?],
            vec![fused("down_proj", [d.experts(), h, i])?],
        )
    } else {
        (
            (0..d.experts())
                .flat_map(|e| {
                    ["gate_proj", "up_proj"].map(|projection| expert(e, projection, [i, h]))
                })
                .collect::<Result<Vec<_>>>()?,
            (0..d.experts())
                .map(|e| expert(e, "down_proj", [h, i]))
                .collect::<Result<Vec<_>>>()?,
        )
    };
    gate_up.extend([shared("gate_proj", [i, h])?, shared("up_proj", [i, h])?]);
    down.push(shared("down_proj", [h, i])?);
    Ok(MoeWeights {
        router: sources.matrix(&format!("{path}.gate.weight"), [d.experts(), h], stream)?,
        gate_up: HostWeight::stack(gate_up)?.upload(stream)?,
        down: HostWeight::stack(down)?.upload(stream)?,
        shared_gate: sources.matrix(
            &format!("{path}.shared_expert_gate.weight"),
            [1, h],
            stream,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use safetensors::Dtype;
    use safetensors::tensor::{TensorView, serialize_to_file};

    const PROJECTION: &str = "model.language_model.layers.0.linear_attn.in_proj_z.weight";

    fn small_config() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/small-qwen.json")).unwrap()
    }

    fn lm_head_shape(config: &Value) -> [usize; 2] {
        let text = &config["text_config"];
        let size = |name: &str| text[name].as_u64().unwrap() as usize;
        [size("vocab_size"), size("hidden_size")]
    }

    fn write_checkpoint(dir: &Path, fill: f32) -> Result<()> {
        let config = small_config();
        let [vocab, hidden] = lm_head_shape(&config);
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("config.json"), config.to_string())?;
        let bytes = |len: usize| -> Vec<u8> {
            (0..len)
                .flat_map(|_| bf16::from_f32(fill).to_le_bytes())
                .collect()
        };
        let (lm_head, projection) = (bytes(vocab * hidden), bytes(4 * hidden));
        let shard = "model.safetensors";
        serialize_to_file(
            [
                (
                    "lm_head.weight",
                    TensorView::new(Dtype::BF16, vec![vocab, hidden], &lm_head)?,
                ),
                (
                    PROJECTION,
                    TensorView::new(Dtype::BF16, vec![4, hidden], &projection)?,
                ),
            ],
            None,
            &dir.join(shard),
        )?;
        let index = serde_json::json!({
            "weight_map": {"lm_head.weight": shard, PROJECTION: shard},
        });
        std::fs::write(dir.join("model.safetensors.index.json"), index.to_string())?;
        Ok(())
    }

    fn dense_values(weight: HostWeight) -> Vec<f32> {
        match weight {
            HostWeight::Dense(tensor) => tensor.data.iter().map(|x| x.to_f32()).collect(),
            _ => panic!("expected a dense weight"),
        }
    }

    #[test]
    fn overrides_replace_only_their_weight_group() -> Result<()> {
        let root = std::env::temp_dir().join(format!("firepit-overrides-{}", std::process::id()));
        let (base, other) = (root.join("base"), root.join("other"));
        write_checkpoint(&base, 1.0)?;
        write_checkpoint(&other, 2.0)?;
        let shape = lm_head_shape(&small_config());
        let lm_head_override: WeightOverride = format!("lm_head={}", other.display()).parse()?;
        let sources = Sources::open(&base, &[lm_head_override.clone()])?;
        assert!(
            dense_values(sources.host("lm_head.weight", shape)?)
                .iter()
                .all(|&x| x == 2.0)
        );
        assert!(
            dense_values(sources.host(PROJECTION, [4, shape[1]])?)
                .iter()
                .all(|&x| x == 1.0)
        );
        assert!(Sources::open(&base, &[lm_head_override.clone(), lm_head_override]).is_err());
        let projections: WeightOverride = format!("projections={}", other.display()).parse()?;
        let sources = Sources::open(&base, &[projections])?;
        assert!(
            dense_values(sources.host("lm_head.weight", shape)?)
                .iter()
                .all(|&x| x == 1.0)
        );
        assert!(
            dense_values(sources.host(PROJECTION, [4, shape[1]])?)
                .iter()
                .all(|&x| x == 2.0)
        );
        assert!(sources.host(PROJECTION, [8, shape[1]]).is_err());
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn verification_reports_formats_and_rejects_stray_scales() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("firepit-verify-{}", std::process::id()));
        write_checkpoint(&dir, 1.0)?;
        let report = verify_checkpoint(&dir)?;
        let lm_head = report
            .iter()
            .find(|weight| weight.name == "lm_head.weight")
            .ok_or("missing lm_head")?;
        assert_eq!(
            (lm_head.format, lm_head.dtype, lm_head.shape.clone()),
            (
                WeightFormat::Bf16,
                Dtype::BF16,
                lm_head_shape(&small_config()).to_vec()
            )
        );
        let index_path = dir.join("model.safetensors.index.json");
        let mut index: Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        index["weight_map"]["lm_head.weight_scale"] = "model.safetensors".into();
        std::fs::write(&index_path, index.to_string())?;
        assert!(verify_checkpoint(&dir).is_err());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn weight_groups_parse_and_classify_names() {
        assert_eq!("lm_head".parse(), Ok(WeightGroup::LmHead));
        assert_eq!("projections".parse(), Ok(WeightGroup::Projections));
        assert!("experts".parse::<WeightGroup>().is_err());
        assert!("lm_head".parse::<WeightOverride>().is_err());
        let layer = "model.language_model.layers.3";
        for (weight, group) in [
            ("lm_head.weight".to_owned(), Some(WeightGroup::LmHead)),
            (
                format!("{layer}.self_attn.q_proj.weight"),
                Some(WeightGroup::Projections),
            ),
            (
                format!("{layer}.self_attn.o_proj.weight"),
                Some(WeightGroup::Projections),
            ),
            (
                format!("{layer}.linear_attn.in_proj_qkv.weight"),
                Some(WeightGroup::Projections),
            ),
            (
                format!("{layer}.linear_attn.out_proj.weight"),
                Some(WeightGroup::Projections),
            ),
            (format!("{layer}.mlp.experts.0.gate_proj.weight"), None),
            (format!("{layer}.mlp.shared_expert.up_proj.weight"), None),
            (format!("{layer}.mlp.gate.weight"), None),
        ] {
            assert_eq!(WeightGroup::of(&weight), group, "{weight}");
        }
    }

    /// None where the device does not report it, such as GB10 unified memory ("[N/A]").
    fn gpu_used_mib() -> Result<Option<usize>> {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()?;
        if !output.status.success() {
            return Err("nvidia-smi failed".into());
        }
        Ok(std::str::from_utf8(&output.stdout)?.trim().parse().ok())
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
        match (before, after) {
            (Some(before), Some(after)) => println!(
                "GPU memory before={before} MiB after={after} MiB delta={} MiB",
                after.saturating_sub(before)
            ),
            _ => println!("GPU memory: not reported by nvidia-smi"),
        }
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
