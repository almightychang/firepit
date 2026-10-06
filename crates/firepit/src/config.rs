use crate::Result;
use firepit_kernels::attention::AttentionDims;
use firepit_kernels::gdn::GdnDims;
use firepit_kernels::moe::MoeDims;
use firepit_kernels::rope::RopeDims;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    Bf16,
    Fp8Blocks { block_n: usize, block_k: usize },
    Fp8Tensor,
    Nvfp4,
}

pub struct Fp8Config {
    block_n: usize,
    block_k: usize,
    excluded: HashSet<String>,
}

impl Fp8Config {
    fn parse(value: &Value) -> Result<Self> {
        if value["fmt"] != "e4m3" || value["activation_scheme"] != "dynamic" {
            return Err("unsupported FP8 quantization configuration".into());
        }
        let blocks = value["weight_block_size"]
            .as_array()
            .ok_or("missing FP8 block size")?;
        if blocks.len() != 2 {
            return Err("FP8 block size must have two dimensions".into());
        }
        let block_n: usize = blocks[0]
            .as_u64()
            .ok_or("invalid FP8 block rows")?
            .try_into()?;
        let block_k: usize = blocks[1]
            .as_u64()
            .ok_or("invalid FP8 block columns")?
            .try_into()?;
        let gemm = firepit_kernels::gemm::DEFAULT_GEMM_TILE;
        let gemv = firepit_kernels::gemv::DEFAULT_GEMV_TILE;
        if block_n == 0
            || block_k == 0
            || block_n % gemm.n != 0
            || block_k % (gemv.blocks * firepit_kernels::weights::Q8_BLOCK) != 0
        {
            return Err("unsupported FP8 block shape".into());
        }
        let excluded = value["modules_to_not_convert"]
            .as_array()
            .ok_or("missing excluded FP8 modules")?
            .iter()
            .map(|name| {
                name.as_str()
                    .map(str::to_owned)
                    .ok_or("invalid excluded FP8 module")
            })
            .collect::<std::result::Result<HashSet<_>, _>>()?;
        Ok(Self {
            block_n,
            block_k,
            excluded,
        })
    }

    fn format(&self, module: &str) -> WeightFormat {
        if self.excluded.contains(module) {
            WeightFormat::Bf16
        } else {
            WeightFormat::Fp8Blocks {
                block_n: self.block_n,
                block_k: self.block_k,
            }
        }
    }
}

pub struct ModelOptConfig {
    layers: HashMap<String, WeightFormat>,
}

impl ModelOptConfig {
    fn parse(value: &Value) -> Result<Self> {
        let quantization = &value["quantization"];
        if value["producer"]["name"] != "modelopt"
            || quantization["quant_algo"] != "MIXED_PRECISION"
        {
            return Err("unsupported ModelOpt quantization configuration".into());
        }
        let layers = quantization["quantized_layers"]
            .as_object()
            .ok_or("missing ModelOpt quantized layers")?
            .iter()
            .map(|(module, layer)| {
                let format = match (layer["quant_algo"].as_str(), layer["group_size"].as_u64()) {
                    (Some("FP8"), None) => WeightFormat::Fp8Tensor,
                    (Some("W4A16_NVFP4"), Some(group))
                        if group == firepit_kernels::weights::NVFP4_GROUP as u64 =>
                    {
                        WeightFormat::Nvfp4
                    }
                    _ => return Err(format!("{module}: unsupported ModelOpt layer quantization")),
                };
                Ok((module.clone(), format))
            })
            .collect::<std::result::Result<_, _>>()?;
        Ok(Self { layers })
    }

    fn format(&self, module: &str) -> WeightFormat {
        let module = match module.split_once(".experts.") {
            Some((base, _)) => format!("{base}.experts"),
            None => module.to_owned(),
        };
        self.layers
            .get(&module)
            .copied()
            .unwrap_or(WeightFormat::Bf16)
    }
}

pub enum Quantization {
    Fp8(Fp8Config),
    ModelOpt(ModelOptConfig),
}

impl Quantization {
    fn parse(config: &Value, modelopt: Option<&Value>) -> Result<Option<Self>> {
        let method = config
            .get("quantization_config")
            .map(|q| {
                q["quant_method"]
                    .as_str()
                    .ok_or("invalid quantization method")
            })
            .transpose()?;
        match (method, modelopt) {
            (None, None) => Ok(None),
            (Some("fp8"), None) => Ok(Some(Self::Fp8(Fp8Config::parse(
                &config["quantization_config"],
            )?))),
            (Some("modelopt"), Some(modelopt)) => {
                Ok(Some(Self::ModelOpt(ModelOptConfig::parse(modelopt)?)))
            }
            _ => Err("quantization method and ModelOpt configuration disagree".into()),
        }
    }

    pub fn format(&self, weight: &str) -> WeightFormat {
        let module = weight.strip_suffix(".weight").unwrap_or(weight);
        match self {
            Self::Fp8(config) => config.format(module),
            Self::ModelOpt(config) => config.format(module),
        }
    }
}

pub struct ModelConfig {
    text: Value,
    quantization: Option<Quantization>,
}

impl TryFrom<Value> for ModelConfig {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: Value) -> Result<Self> {
        Self::new(value, None)
    }
}

impl ModelConfig {
    pub fn new(value: Value, modelopt: Option<Value>) -> Result<Self> {
        let quantization = Quantization::parse(&value, modelopt.as_ref())?;
        let text = if value.get("text_config").is_some() {
            value["text_config"].clone()
        } else {
            value
        };
        ModelDims::parse(&text, 1, None)?;
        Ok(Self { text, quantization })
    }

    pub fn open(dir: &Path) -> Result<Self> {
        let json = |name: &str| -> Result<Value> {
            Ok(serde_json::from_reader(File::open(dir.join(name))?)?)
        };
        let modelopt = dir
            .join("hf_quant_config.json")
            .exists()
            .then(|| json("hf_quant_config.json"))
            .transpose()?;
        Self::new(json("config.json")?, modelopt)
    }

    pub fn max_positions(&self) -> Result<usize> {
        Ok(self.text["max_position_embeddings"]
            .as_u64()
            .ok_or("missing or invalid max_position_embeddings")?
            .try_into()?)
    }

    pub fn dims(&self, max_seq: usize, prefix_layers: Option<usize>) -> Result<ModelDims> {
        ModelDims::parse(&self.text, max_seq, prefix_layers)
    }

    pub fn text(&self) -> &Value {
        &self.text
    }

    pub fn format(&self, weight: &str) -> WeightFormat {
        self.quantization
            .as_ref()
            .map_or(WeightFormat::Bf16, |quantization| {
                quantization.format(weight)
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Gdn,
    Attention,
}

pub struct ModelDims {
    hidden: usize,
    vocab: usize,
    layers: Vec<LayerKind>,
    attention: AttentionDims,
    q_rope: RopeDims,
    k_rope: RopeDims,
    moe: MoeDims,
    gdn: GdnDims,
    norm_eps: f32,
    max_seq: usize,
}

impl ModelDims {
    pub fn parse(text: &Value, max_seq: usize, prefix_layers: Option<usize>) -> Result<Self> {
        let integer = |name: &str| -> Result<usize> {
            Ok(text[name]
                .as_u64()
                .ok_or_else(|| format!("missing or invalid {name}"))?
                .try_into()?)
        };
        let float = |name: &str| -> Result<f32> {
            let value = text[name]
                .as_f64()
                .ok_or_else(|| format!("missing or invalid {name}"))?;
            if !value.is_finite() || value.abs() > f32::MAX as f64 {
                return Err(format!("invalid {name}").into());
            }
            Ok(value as f32)
        };
        let hidden = integer("hidden_size")?;
        let vocab = integer("vocab_size")?;
        let norm_eps = float("rms_norm_eps")?;
        let full_layers = integer("num_hidden_layers")?;
        let layer_types = text["layer_types"]
            .as_array()
            .ok_or("missing layer_types")?;
        if layer_types.len() != full_layers
            || max_seq == 0
            || max_seq > integer("max_position_embeddings")?
        {
            return Err("invalid layer count or max_seq".into());
        }
        let requested = prefix_layers.unwrap_or(full_layers);
        if requested > full_layers {
            return Err("prefix exceeds layer count".into());
        }
        let layers = layer_types[..requested]
            .iter()
            .map(|kind| match kind.as_str() {
                Some("linear_attention") => Ok(LayerKind::Gdn),
                Some("full_attention") => Ok(LayerKind::Attention),
                _ => Err("unknown layer type"),
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let head_dim = integer("head_dim")?;
        let q_heads = integer("num_attention_heads")?;
        let kv_heads = integer("num_key_value_heads")?;
        let attention = AttentionDims::new(max_seq, q_heads, kv_heads, head_dim)?;
        let rotary_factor = text["rope_parameters"]["partial_rotary_factor"]
            .as_f64()
            .or_else(|| text["partial_rotary_factor"].as_f64())
            .ok_or("missing partial_rotary_factor")?;
        let rotary_dim = head_dim as f64 * rotary_factor;
        if !rotary_dim.is_finite() || rotary_dim.fract() != 0.0 || rotary_dim < 0.0 {
            return Err("invalid rotary dimension".into());
        }
        let theta = text["rope_parameters"]["rope_theta"]
            .as_f64()
            .ok_or("missing rope_theta")? as f32;
        let q_rope = RopeDims::new(q_heads, head_dim, rotary_dim as usize, theta)?;
        let k_rope = RopeDims::new(kv_heads, head_dim, rotary_dim as usize, theta)?;
        let shared_intermediate = integer("shared_expert_intermediate_size")?;
        if shared_intermediate != integer("moe_intermediate_size")? {
            return Err(
                "shared_expert_intermediate_size must equal moe_intermediate_size: the shared \
                 expert runs as an always-selected routed expert"
                    .into(),
            );
        }
        let moe = MoeDims::new(
            hidden,
            integer("moe_intermediate_size")?,
            shared_intermediate,
            integer("num_experts")?,
            integer("num_experts_per_tok")?,
        )?;
        let gdn = GdnDims::new(
            hidden,
            integer("linear_num_key_heads")?,
            integer("linear_num_value_heads")?,
            integer("linear_key_head_dim")?,
            integer("linear_value_head_dim")?,
            integer("linear_conv_kernel_dim")?,
            norm_eps,
        )?;
        if hidden == 0
            || vocab == 0
            || vocab
                .checked_mul(hidden)
                .is_none_or(|size| size > i32::MAX as usize)
            || !norm_eps.is_finite()
            || norm_eps <= 0.0
        {
            return Err("invalid hidden, vocab, or norm epsilon".into());
        }
        Ok(Self {
            hidden,
            vocab,
            layers,
            attention,
            q_rope,
            k_rope,
            moe,
            gdn,
            norm_eps,
            max_seq,
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn layers(&self) -> &[LayerKind] {
        &self.layers
    }
    pub fn attention(&self) -> AttentionDims {
        self.attention
    }
    pub fn q_rope(&self) -> RopeDims {
        self.q_rope
    }
    pub fn k_rope(&self) -> RopeDims {
        self.k_rope
    }
    pub fn moe(&self) -> MoeDims {
        self.moe
    }
    pub fn gdn(&self) -> GdnDims {
        self.gdn
    }
    pub fn norm_eps(&self) -> f32 {
        self.norm_eps
    }
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn modelopt(layers: Value) -> Value {
        json!({
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": layers},
        })
    }

    fn modelopt_method() -> Value {
        json!({"quantization_config": {"quant_method": "modelopt"}})
    }

    #[test]
    fn modelopt_layers_select_weight_formats() -> Result<()> {
        let layers = json!({
            "model.language_model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            "model.language_model.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
            "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
        });
        let quantization = Quantization::parse(&modelopt_method(), Some(&modelopt(layers)))?
            .ok_or("missing quantization")?;
        let layer = "model.language_model.layers.0";
        for (weight, format) in [
            (
                format!("{layer}.mlp.experts.7.down_proj.weight"),
                WeightFormat::Nvfp4,
            ),
            (
                format!("{layer}.linear_attn.in_proj_qkv.weight"),
                WeightFormat::Fp8Tensor,
            ),
            ("lm_head.weight".to_owned(), WeightFormat::Nvfp4),
            (format!("{layer}.mlp.gate.weight"), WeightFormat::Bf16),
            (
                format!("{layer}.linear_attn.in_proj_a.weight"),
                WeightFormat::Bf16,
            ),
        ] {
            assert_eq!(quantization.format(&weight), format, "{weight}");
        }
        Ok(())
    }

    #[test]
    fn rejects_inconsistent_quantization() {
        let layer = |algo: Value| modelopt(json!({"lm_head": algo}));
        let nvfp4 = layer(json!({"quant_algo": "W4A16_NVFP4", "group_size": 16}));
        assert!(Quantization::parse(&modelopt_method(), None).is_err());
        assert!(Quantization::parse(&json!({}), Some(&nvfp4)).is_err());
        for algo in [
            json!({"quant_algo": "W4A16_NVFP4", "group_size": 32}),
            json!({"quant_algo": "W4A16_NVFP4"}),
            json!({"quant_algo": "W4A8_AWQ"}),
        ] {
            assert!(Quantization::parse(&modelopt_method(), Some(&layer(algo))).is_err());
        }
    }
}
