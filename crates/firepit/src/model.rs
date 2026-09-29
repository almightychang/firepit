use crate::session::{LayerState, next_session_instance};
use crate::{Result, Session, SessionId};
use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::Tensor;
use firepit_kernels::q8::{HostQ8Matrix, Q8Matrix};
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

pub struct Model {
    pub config: ModelConfig,
    pub(crate) embedding: Tensor<f32>,
    pub(crate) weights: Vec<Q8Matrix>,
}

impl Model {
    pub fn synthetic(config: ModelConfig, stream: &Stream) -> Result<Self> {
        if config.hidden == 0
            || config.heads.checked_mul(config.head_dim) != Some(config.hidden)
            || config.hidden % 32 != 0
            || config.vocab == 0
            || config.max_seq == 0
        {
            return Err("invalid model dimensions".into());
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
                conv: api::zeros::<f32>(&[config.heads, config.head_dim]).sync_on(stream)?,
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
