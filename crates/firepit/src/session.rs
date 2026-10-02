use crate::checkpoint::Stream;
use crate::{LayerKind, ModelDims, Result};
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::Tensor;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
const MAX_BATCH_SESSIONS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u64);

pub enum LayerState {
    Gdn {
        conv: Tensor<f32>,
        recurrent: Tensor<f32>,
    },
    Attention {
        k: Tensor<f32>,
        v: Tensor<f32>,
    },
}

pub struct Session {
    pub(crate) id: SessionId,
    pub(crate) engine: u64,
    pub(crate) instance: u64,
    pub(crate) position: usize,
    pub(crate) state: Vec<LayerState>,
    pub(crate) poisoned: bool,
}

impl Session {
    pub fn id(&self) -> SessionId {
        self.id
    }
    pub fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn new(
        id: SessionId,
        engine: u64,
        dims: &ModelDims,
        stream: &Stream,
    ) -> Result<Self> {
        let mut state = Vec::with_capacity(dims.layers().len());
        for kind in dims.layers() {
            state.push(match kind {
                LayerKind::Gdn => {
                    let g = dims.gdn();
                    LayerState::Gdn {
                        conv: api::zeros::<f32>(&[g.conv_dim(), g.conv_kernel() - 1])
                            .sync_on(stream)?,
                        recurrent: api::zeros::<f32>(&[
                            g.value_heads(),
                            g.key_head_dim(),
                            g.value_head_dim(),
                        ])
                        .sync_on(stream)?,
                    }
                }
                LayerKind::Attention => {
                    let a = dims.attention();
                    LayerState::Attention {
                        k: api::zeros::<f32>(&[a.max_seq(), a.kv_heads(), a.head_dim()])
                            .sync_on(stream)?,
                        v: api::zeros::<f32>(&[a.max_seq(), a.kv_heads(), a.head_dim()])
                            .sync_on(stream)?,
                    }
                }
            });
        }
        Ok(Self {
            id,
            engine,
            instance: NEXT_SESSION.fetch_add(1, Ordering::Relaxed),
            position: 0,
            state,
            poisoned: false,
        })
    }
}

pub struct Batch<'a> {
    pub(crate) engine: u64,
    pub(crate) sessions: Vec<&'a mut Session>,
    pub(crate) tokens: Vec<u32>,
    pub(crate) original_rows: Vec<usize>,
}

impl<'a> Batch<'a> {
    pub(crate) fn from_entries(
        engine: u64,
        entries: impl IntoIterator<Item = (&'a mut Session, u32)>,
        dims: &ModelDims,
    ) -> Result<Self> {
        let mut entries: Vec<_> = entries
            .into_iter()
            .enumerate()
            .map(|(original, (session, token))| (session, token, original))
            .collect();
        if entries.is_empty() || entries.len() > MAX_BATCH_SESSIONS {
            return Err("invalid batch size".into());
        }
        let mut ids = HashSet::new();
        for (session, token, _) in &entries {
            if !ids.insert(session.id)
                || session.engine != engine
                || session.poisoned
                || session.position >= dims.max_seq()
                || *token as usize >= dims.vocab()
            {
                return Err("invalid session or token".into());
            }
        }
        entries.sort_by_key(|(session, _, _)| session.id);
        let mut sessions = Vec::with_capacity(entries.len());
        let mut tokens = Vec::with_capacity(entries.len());
        let mut original_rows = Vec::with_capacity(entries.len());
        for (session, token, original) in entries {
            sessions.push(session);
            tokens.push(token);
            original_rows.push(original);
        }
        Ok(Self {
            engine,
            sessions,
            tokens,
            original_rows,
        })
    }
}
