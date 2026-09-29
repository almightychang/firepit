use crate::engine::EngineId;
use crate::model::ModelId;
use cutile::tensor::Tensor;
use half::f16;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u64);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct SessionInstance(u64);
static NEXT_SESSION_INSTANCE: AtomicU64 = AtomicU64::new(1);
pub(crate) fn next_session_instance() -> SessionInstance {
    SessionInstance(NEXT_SESSION_INSTANCE.fetch_add(1, Ordering::Relaxed))
}

pub struct Session {
    pub(crate) id: SessionId,
    pub(crate) model: ModelId,
    pub(crate) instance: SessionInstance,
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
}

pub(crate) enum LayerState {
    Gdn { recurrent: Tensor<f32> },
    Attention { k: Tensor<f16>, v: Tensor<f16> },
}

pub struct Batch<'s> {
    pub(crate) engine: EngineId,
    pub(crate) sessions: Vec<&'s mut Session>,
    pub(crate) tokens: Vec<u32>,
    pub(crate) original_rows: Vec<usize>,
}

impl<'s> Batch<'s> {
    pub(crate) fn from_checked(
        engine: EngineId,
        entries: impl IntoIterator<Item = (&'s mut Session, u32)>,
    ) -> Self {
        let mut entries: Vec<_> = entries
            .into_iter()
            .enumerate()
            .map(|(original, (session, token))| (session, token, original))
            .collect();
        entries.sort_by_key(|(session, _, _)| session.id);
        let (sessions, rest): (Vec<_>, Vec<_>) = entries
            .into_iter()
            .map(|(session, token, original)| (session, (token, original)))
            .unzip();
        let (tokens, original_rows) = rest.into_iter().unzip();
        Self {
            engine,
            sessions,
            tokens,
            original_rows,
        }
    }
}
