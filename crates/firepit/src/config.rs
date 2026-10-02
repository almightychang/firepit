use crate::Result;
use firepit_kernels::attention::AttentionDims;
use firepit_kernels::gdn::GdnDims;
use firepit_kernels::moe::MoeDims;
use firepit_kernels::rope::RopeDims;
use serde_json::Value;

pub struct ModelConfig {
    text: Value,
}

impl TryFrom<Value> for ModelConfig {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: Value) -> Result<Self> {
        let text = if value.get("text_config").is_some() {
            value["text_config"].clone()
        } else {
            value
        };
        ModelDims::parse(&text, 1, None)?;
        Ok(Self { text })
    }
}

impl ModelConfig {
    pub fn dims(&self, max_seq: usize, prefix_layers: Option<usize>) -> Result<ModelDims> {
        ModelDims::parse(&self.text, max_seq, prefix_layers)
    }

    pub fn text(&self) -> &Value {
        &self.text
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
        let moe = MoeDims::new(
            hidden,
            integer("moe_intermediate_size")?,
            integer("shared_expert_intermediate_size")?,
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
