use crate::engine::EngineId;
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
    pub(crate) engine: EngineId,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Engine, Model, ModelConfig, ModelDims, Result};
    use cuda_core::Device;
    use std::sync::Arc;

    fn permutations(items: &mut [usize], at: usize, visit: &mut impl FnMut(&[usize])) {
        if at == items.len() {
            visit(items);
            return;
        }
        for next in at..items.len() {
            items.swap(at, next);
            permutations(items, at + 1, visit);
            items.swap(at, next);
        }
    }

    #[test]
    fn batch_order_restores_every_permutation_up_to_four_sessions() -> Result<()> {
        let device = Device::new(0)?;
        let stream = device.new_stream()?;
        let dims = ModelDims::try_from(ModelConfig::default())?;
        let model = Arc::new(Model::synthetic(dims, &stream)?);
        let engine = Engine::new(model, &device)?;
        let ids = [SessionId(9), SessionId(2), SessionId(7), SessionId(4)];
        let mut sessions = ids
            .iter()
            .map(|&id| engine.new_session(id))
            .collect::<Result<Vec<_>>>()?;

        for size in 1..=4 {
            let mut order: Vec<usize> = (0..size).collect();
            permutations(&mut order, 0, &mut |order| {
                let mut entries: Vec<_> = sessions[..size].iter_mut().enumerate().collect();
                entries.sort_by_key(|(index, _)| order.iter().position(|item| item == index));
                let batch = engine
                    .batch(
                        entries
                            .into_iter()
                            .map(|(index, session)| (session, 10 + index as u32)),
                    )
                    .unwrap();

                let mut expected: Vec<_> = order.iter().copied().enumerate().collect();
                expected.sort_by_key(|&(_, index)| ids[index]);
                for (canonical, (original, index)) in expected.into_iter().enumerate() {
                    assert_eq!(batch.sessions[canonical].id(), ids[index]);
                    assert_eq!(batch.tokens[canonical], 10 + index as u32);
                    assert_eq!(batch.original_rows[canonical], original);
                }
            });
        }
        Ok(())
    }
}
