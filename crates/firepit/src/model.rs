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

impl ModelConfig {
    fn fits_kernels(&self) -> bool {
        self.heads > 0
            && self.vocab > 0
            && self.max_seq > 0
            && self.heads * self.head_dim == self.hidden
            && splits_into_row_tiles(self.head_dim)
            && DEFAULT_GEMV_TILE.fits(self.hidden, self.hidden)
            && DEFAULT_GEMV_TILE.fits(self.vocab, self.hidden)
    }
}

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

pub struct Model {
    pub(crate) id: u64,
    pub config: ModelConfig,
    pub(crate) embedding: Tensor<f32>,
    pub(crate) weights: Vec<Q8Matrix>,
}

impl Model {
    pub fn synthetic(config: ModelConfig, stream: &Stream) -> Result<Self> {
        if !config.fits_kernels() {
            return Err("model dimensions do not fit the kernels".into());
        }
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
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            config,
            embedding,
            weights,
        })
    }

    pub fn new_session(&self, id: SessionId, stream: &Stream) -> Result<Session> {
        let config = &self.config;
        let mut state = Vec::new();
        for _ in 0..self.config.gdn_layers {
            state.push(LayerState::Gdn {
                recurrent: api::zeros::<f32>(&[config.heads, config.head_dim, config.head_dim])
                    .sync_on(stream)?,
            });
        }
        for _ in 0..self.config.attention_layers {
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
        })
    }
}

fn synthetic_q8(rows: usize, cols: usize, seed: usize) -> HostQ8Matrix {
    HostQ8Matrix {
        qs: (0..rows * cols)
            .map(|i| ((i * 13 + seed * 7) % 17) as i8 - 8)
            .collect(),
        scales: vec![f16::from_f32(1.0 / 256.0); rows * cols / 32],
        rows,
        cols,
    }
}
