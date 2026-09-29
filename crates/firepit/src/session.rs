use crate::Result;
use cutile::tensor::Tensor;
use half::f16;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

pub type SessionId = u64;
static NEXT_SESSION_INSTANCE: AtomicU64 = AtomicU64::new(1);
pub(crate) fn next_session_instance() -> u64 {
    NEXT_SESSION_INSTANCE.fetch_add(1, Ordering::Relaxed)
}

pub struct Session {
    pub id: SessionId,
    pub(crate) instance: u64,
    pub position: usize,
    pub(crate) state: Vec<LayerState>,
}

pub enum LayerState {
    Gdn {
        recurrent: Tensor<f32>,
        conv: Tensor<f32>,
    },
    Attention {
        k: Tensor<f16>,
        v: Tensor<f16>,
    },
}

pub struct Batch<'s> {
    pub(crate) sessions: Vec<&'s mut Session>,
    pub(crate) tokens: Vec<u32>,
    pub(crate) original_rows: Vec<usize>,
}

impl<'s> Batch<'s> {
    pub fn new(sessions: Vec<&'s mut Session>, tokens: Vec<u32>) -> Result<Self> {
        if sessions.is_empty() || sessions.len() != tokens.len() || sessions.len() > 4 {
            return Err("batch must have 1..=4 sessions and one token per session".into());
        }
        let ids: HashSet<_> = sessions.iter().map(|s| s.id).collect();
        if ids.len() != sessions.len() {
            return Err("duplicate session id".into());
        }
        let mut entries: Vec<_> = sessions
            .into_iter()
            .zip(tokens)
            .enumerate()
            .map(|(original, (session, token))| (session, token, original))
            .collect();
        entries.sort_by_key(|(session, _, _)| session.id);
        let (sessions, rest): (Vec<_>, Vec<_>) = entries
            .into_iter()
            .map(|(session, token, original)| (session, (token, original)))
            .unzip();
        let (tokens, original_rows): (Vec<_>, Vec<_>) = rest.into_iter().unzip();
        Ok(Self {
            sessions,
            tokens,
            original_rows,
        })
    }
}
