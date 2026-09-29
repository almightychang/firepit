use crate::engine::EngineId;
use crate::session::{LayerState, next_session_instance};
use crate::{Result, Session, SessionId};
use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::Tensor;
use firepit_kernels::q8::{DEFAULT_GEMV_TILE, HostQ8Matrix, Q8_BLOCK, Q8Matrix};
use firepit_kernels::tile::splits_into_row_tiles;
use half::f16;
use std::sync::Arc;

pub type Stream = Arc<cuda_core::Stream>;

pub struct ModelConfig {
    pub gdn_layers: usize,
    pub attention_layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub max_seq: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            gdn_layers: 1,
            attention_layers: 1,
            hidden: 256,
            heads: 8,
            head_dim: 32,
            vocab: 512,
            max_seq: 32,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum LayerDef {
    Gdn,
    Attention,
}

#[derive(Clone)]
pub struct ModelDims {
    layers: Vec<LayerDef>,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    vocab: usize,
    max_seq: usize,
}

impl ModelDims {
    pub fn gdn_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l, LayerDef::Gdn))
            .count()
    }
    pub fn attention_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l, LayerDef::Attention))
            .count()
    }
    pub fn hidden(&self) -> usize {
        self.hidden
    }
    pub fn heads(&self) -> usize {
        self.heads
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
    pub(crate) fn layers(&self) -> &[LayerDef] {
        &self.layers
    }
}

#[derive(Debug)]
pub struct DimensionError;
impl std::fmt::Display for DimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "model dimensions do not fit the kernels")
    }
}
impl std::error::Error for DimensionError {}

impl TryFrom<ModelConfig> for ModelDims {
    type Error = DimensionError;

    fn try_from(config: ModelConfig) -> std::result::Result<Self, Self::Error> {
        if config.heads == 0
            || config.vocab == 0
            || config.max_seq == 0
            || config.heads.checked_mul(config.head_dim) != Some(config.hidden)
            || !splits_into_row_tiles(config.head_dim)
            || !DEFAULT_GEMV_TILE.fits(config.hidden, config.hidden)
            || !DEFAULT_GEMV_TILE.fits(config.vocab, config.hidden)
            || config.hidden > i32::MAX as usize
            || config.vocab > i32::MAX as usize
            || config
                .gdn_layers
                .checked_add(config.attention_layers)
                .is_none()
            || config.max_seq > i32::MAX as usize
        {
            return Err(DimensionError);
        }
        Ok(Self {
            layers: std::iter::repeat_n(LayerDef::Gdn, config.gdn_layers)
                .chain(std::iter::repeat_n(
                    LayerDef::Attention,
                    config.attention_layers,
                ))
                .collect(),
            hidden: config.hidden,
            heads: config.heads,
            head_dim: config.head_dim,
            vocab: config.vocab,
            max_seq: config.max_seq,
        })
    }
}

pub(crate) struct LayerWeights {
    pub input: Q8Matrix,
    pub output: Q8Matrix,
}

pub(crate) struct ModelWeights {
    pub layers: Vec<LayerWeights>,
    pub output_head: Q8Matrix,
}

#[derive(Clone, Copy)]
pub(crate) enum WeightRef {
    LayerInput(usize),
    LayerOutput(usize),
    OutputHead,
}

#[derive(Clone, Copy)]
pub(crate) enum NormRef {
    Layer(usize),
    Final,
}

impl ModelWeights {
    pub(crate) fn get(&self, weight: WeightRef) -> &Q8Matrix {
        match weight {
            WeightRef::LayerInput(layer) => &self.layers[layer].input,
            WeightRef::LayerOutput(layer) => &self.layers[layer].output,
            WeightRef::OutputHead => &self.output_head,
        }
    }
}

pub struct Model {
    dims: ModelDims,
    pub(crate) embedding: Tensor<f32>,
    pub(crate) weights: ModelWeights,
    layer_norms: Vec<Tensor<f32>>,
    final_norm: Tensor<f32>,
}

impl Model {
    pub fn synthetic(dims: ModelDims, stream: &Stream) -> Result<Self> {
        let config = &dims;
        let h = config.hidden;
        let table: Vec<f32> = (0..config.vocab * h)
            .map(|i| ((i * 19 % 67) as f32 - 33.0) / 256.0)
            .collect();
        let embedding = api::copy_host_vec_to_device(&Arc::new(table))
            .reshape(&[config.vocab, h])
            .sync_on(stream)?;
        let mut seed = 0;
        let mut layers = Vec::with_capacity(config.layers.len());
        for _ in config.layers() {
            let input = synthetic_q8(h, h, seed).upload(stream)?;
            seed += 1;
            let output = synthetic_q8(h, h, seed).upload(stream)?;
            seed += 1;
            layers.push(LayerWeights { input, output });
        }
        let output_head = synthetic_q8(config.vocab, h, seed).upload(stream)?;
        let unit_norm = Arc::new(vec![1.0f32; h]);
        let layer_norms = (0..config.layers.len())
            .map(|_| api::copy_host_vec_to_device(&unit_norm).sync_on(stream))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let final_norm = api::copy_host_vec_to_device(&unit_norm).sync_on(stream)?;
        Ok(Self {
            dims,
            embedding,
            weights: ModelWeights {
                layers,
                output_head,
            },
            layer_norms,
            final_norm,
        })
    }

    pub fn dims(&self) -> &ModelDims {
        &self.dims
    }

    pub(crate) fn norm(&self, norm: NormRef) -> &Tensor<f32> {
        match norm {
            NormRef::Layer(layer) => &self.layer_norms[layer],
            NormRef::Final => &self.final_norm,
        }
    }

    pub(crate) fn new_session(
        &self,
        id: SessionId,
        engine: EngineId,
        stream: &Stream,
    ) -> Result<Session> {
        let config = &self.dims;
        let mut state = Vec::new();
        for layer in config.layers() {
            state.push(match layer {
                LayerDef::Gdn => LayerState::Gdn {
                    recurrent: api::zeros::<f32>(&[config.heads, config.head_dim, config.head_dim])
                        .sync_on(stream)?,
                },
                LayerDef::Attention => LayerState::Attention {
                    k: api::zeros::<f16>(&[config.max_seq, config.heads, config.head_dim])
                        .sync_on(stream)?,
                    v: api::zeros::<f16>(&[config.max_seq, config.heads, config.head_dim])
                        .sync_on(stream)?,
                },
            });
        }
        Ok(Session {
            id,
            engine,
            instance: next_session_instance(),
            position: 0,
            state,
            poisoned: false,
        })
    }
}

fn synthetic_q8(rows: usize, cols: usize, seed: usize) -> HostQ8Matrix {
    HostQ8Matrix::from_parts(
        (0..rows * cols)
            .map(|i| ((i * 13 + seed * 7) % 17) as i8 - 8)
            .collect(),
        vec![f16::from_f32(1.0 / 256.0); rows * cols / Q8_BLOCK],
        rows,
        cols,
    )
    .expect("validated model dimensions")
}
