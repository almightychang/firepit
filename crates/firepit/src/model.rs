use crate::session::{LayerState, next_session_instance};
use crate::{Result, Session, SessionId};
use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::Tensor;
use firepit_kernels::q8::{DEFAULT_GEMV_TILE, HostQ8Matrix, Q8Matrix};
use firepit_kernels::tile::splits_into_row_tiles;
use half::f16;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
pub struct ModelDims {
    gdn_layers: usize,
    attention_layers: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    vocab: usize,
    max_seq: usize,
}

impl ModelDims {
    pub fn gdn_layers(&self) -> usize {
        self.gdn_layers
    }
    pub fn attention_layers(&self) -> usize {
        self.attention_layers
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
            gdn_layers: config.gdn_layers,
            attention_layers: config.attention_layers,
            hidden: config.hidden,
            heads: config.heads,
            head_dim: config.head_dim,
            vocab: config.vocab,
            max_seq: config.max_seq,
        })
    }
}

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelId(u64);

pub struct Model {
    pub(crate) id: ModelId,
    dims: ModelDims,
    pub(crate) embedding: Tensor<f32>,
    pub(crate) weights: Vec<Q8Matrix>,
}

impl Model {
    pub fn synthetic(dims: ModelDims, stream: &Stream) -> Result<Self> {
        let config = dims;
        let h = config.hidden;
        let table: Vec<f32> = (0..config.vocab * h)
            .map(|i| ((i * 19 % 67) as f32 - 33.0) / 256.0)
            .collect();
        let embedding = api::copy_host_vec_to_device(&Arc::new(table))
            .reshape(&[config.vocab, h])
            .sync_on(stream)?;
        let layer_rows = std::iter::repeat_n(h, (config.gdn_layers + config.attention_layers) * 2);
        let weights = layer_rows
            .chain([config.vocab])
            .enumerate()
            .map(|(n, rows)| synthetic_q8(rows, h, n).upload(stream))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            id: ModelId(NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed)),
            dims,
            embedding,
            weights,
        })
    }

    pub fn dims(&self) -> &ModelDims {
        &self.dims
    }

    pub(crate) fn new_session(&self, id: SessionId, stream: &Stream) -> Result<Session> {
        let config = &self.dims;
        let mut state = Vec::new();
        for _ in 0..self.dims.gdn_layers {
            state.push(LayerState::Gdn {
                recurrent: api::zeros::<f32>(&[config.heads, config.head_dim, config.head_dim])
                    .sync_on(stream)?,
            });
        }
        for _ in 0..self.dims.attention_layers {
            state.push(LayerState::Attention {
                k: api::zeros::<f16>(&[config.max_seq, config.heads, config.head_dim])
                    .sync_on(stream)?,
                v: api::zeros::<f16>(&[config.max_seq, config.heads, config.head_dim])
                    .sync_on(stream)?,
            });
        }
        Ok(Session {
            id,
            model: self.id,
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
        vec![f16::from_f32(1.0 / 256.0); rows * cols / 32],
        rows,
        cols,
    )
    .expect("validated model dimensions")
}
