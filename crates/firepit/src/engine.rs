use crate::Result;
use crate::checkpoint::Stream;
use crate::execute::run;
use crate::model::Model;
use crate::plan::StepPlan;
#[cfg(feature = "testing")]
pub use crate::prefill::IsolatedLayerTrace;
use crate::prefill::{self, PrefillPlan, ScoredRows};
use crate::session::{Batch, Session, SessionId};
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
#[cfg(feature = "testing")]
use cutile::tile_kernel::ToHostVecOp;
use firepit_kernels::rope;
use firepit_kernels::sink::{Eager, Graph, Trace, Traced};
use firepit_kernels::tile::{MIN_ROW_TILE, channel_tile, row_tile};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ENGINE: AtomicU64 = AtomicU64::new(1);
const TARGET_BLOCKS_PER_SM: usize = 1;

fn last_scored_rows(session: &mut Session, tokens: &[u32]) -> Result<ScoredRows> {
    ScoredRows::last(tokens.len()).map_err(|error| {
        session.poisoned = true;
        error
    })
}

#[cfg(feature = "testing")]
#[derive(Clone, Copy, Debug, Default)]
pub enum VerificationFault {
    #[default]
    None,
    ResetRopePosition,
    FirstRowLmHead,
}

#[cfg(feature = "testing")]
pub struct SessionStateSnapshot {
    pub layer: usize,
    pub name: &'static str,
    pub values: Vec<f32>,
}

#[cfg(feature = "testing")]
#[derive(Default)]
pub struct RoutingLayerTrace {
    pub router_logits: Vec<Vec<f32>>,
    pub selected: Vec<Vec<i32>>,
    pub hidden: Vec<Vec<f32>>,
}

#[cfg(feature = "testing")]
impl RoutingLayerTrace {
    pub fn append(&mut self, other: Self) {
        self.router_logits.extend(other.router_logits);
        self.selected.extend(other.selected);
        self.hidden.extend(other.hidden);
    }
}

struct CachedPlan {
    instances: Vec<u64>,
    plan: StepPlan,
}

pub struct Engine {
    model: Arc<Model>,
    stream: Stream,
    id: u64,
    gdn_value_tile: usize,
    gdn_conv_tile: usize,
    plans: HashMap<usize, CachedPlan>,
    #[cfg(test)]
    fail_after_gpu_work: bool,
}

impl Engine {
    pub fn new(model: Arc<Model>, device: &Arc<Device>) -> Result<Self> {
        if model.device_ordinal != device.ordinal() {
            return Err("model and engine must use the same device".into());
        }
        let sm =
            cuda_core::simt::CudaContext::new(device.ordinal())?.multiprocessor_count()? as usize;
        let g = model.dims.gdn();
        let gdn_value_tile = row_tile(
            g.value_head_dim(),
            g.value_heads(),
            sm,
            TARGET_BLOCKS_PER_SM,
            MIN_ROW_TILE,
        );
        let gdn_conv_tile = channel_tile(g.conv_dim(), sm, TARGET_BLOCKS_PER_SM);
        Ok(Self {
            model,
            stream: device.new_stream()?,
            id: NEXT_ENGINE.fetch_add(1, Ordering::Relaxed),
            gdn_value_tile,
            gdn_conv_tile,
            plans: HashMap::new(),
            #[cfg(test)]
            fail_after_gpu_work: false,
        })
    }

    pub fn new_session(&self, id: SessionId) -> Result<Session> {
        Session::new(id, self.id, &self.model.dims, &self.stream)
    }

    pub fn close_session(&mut self, session: Session) -> Result<()> {
        if session.engine != self.id {
            return Err("session belongs to another engine".into());
        }
        self.plans
            .retain(|_, entry| !entry.instances.contains(&session.instance));
        Ok(())
    }

    pub fn batch<'a>(
        &self,
        entries: impl IntoIterator<Item = (&'a mut Session, u32)>,
    ) -> Result<Batch<'a>> {
        Batch::from_entries(self.id, entries, &self.model.dims)
    }

    pub fn step_eager(&mut self, batch: Batch<'_>) -> Result<Vec<f32>> {
        self.step_with(batch, |model, plan, sessions, stream| {
            run(model, plan, sessions, &mut Eager(stream))
        })
        .map(|(logits, ())| logits)
    }

    pub fn step_eager_traced(&mut self, batch: Batch<'_>) -> Result<(Vec<f32>, Trace)> {
        self.step_with(batch, |model, plan, sessions, stream| {
            let mut sink = Traced::new(stream);
            run(model, plan, sessions, &mut sink)?;
            Ok(sink.into_trace())
        })
    }

    pub fn prefill(&mut self, session: &mut Session, tokens: &[u32]) -> Result<Vec<f32>> {
        let rows = last_scored_rows(session, tokens)?;
        self.prefill_scored(session, tokens, rows)
    }

    pub fn prefill_scored(
        &mut self,
        session: &mut Session,
        tokens: &[u32],
        rows: ScoredRows,
    ) -> Result<Vec<f32>> {
        self.prefill_with(session, tokens, rows, |model, plan, session, stream| {
            prefill::run(model, plan, session, stream, &mut Eager(stream))
        })
        .map(|(logits, ())| logits)
    }

    pub fn prefill_traced(
        &mut self,
        session: &mut Session,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, Trace)> {
        let rows = last_scored_rows(session, tokens)?;
        self.prefill_with(session, tokens, rows, |model, plan, session, stream| {
            let mut sink = Traced::new(stream);
            prefill::run(model, plan, session, stream, &mut sink)?;
            Ok(sink.into_trace())
        })
    }

    #[cfg(feature = "testing")]
    pub fn prefill_observed(
        &mut self,
        session: &mut Session,
        tokens: &[u32],
        fault: VerificationFault,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let rows = last_scored_rows(session, tokens)?;
        self.prefill_with(session, tokens, rows, |model, plan, session, stream| {
            if matches!(fault, VerificationFault::ResetRopePosition) {
                plan.reset_rope_positions(model, stream)?;
            }
            let mut hidden = Vec::with_capacity(model.dims.layers().len());
            prefill::run_observed(
                model,
                plan,
                session,
                stream,
                matches!(fault, VerificationFault::FirstRowLmHead),
                &mut hidden,
            )?;
            Ok(hidden)
        })
    }

    #[cfg(feature = "testing")]
    pub fn prefill_routing_trace(
        &mut self,
        session: &mut Session,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, Vec<RoutingLayerTrace>)> {
        let rows = last_scored_rows(session, tokens)?;
        self.prefill_with(session, tokens, rows, |model, plan, session, stream| {
            prefill::run_routing_observed(model, plan, session, stream)
        })
    }

    #[cfg(feature = "testing")]
    pub fn prefill_isolated_layer(
        &mut self,
        tokens: &[u32],
        layer: usize,
        input: &[f32],
    ) -> Result<IsolatedLayerTrace> {
        if tokens.is_empty()
            || tokens.len() > self.model.dims.max_seq()
            || tokens
                .iter()
                .any(|&token| token as usize >= self.model.dims.vocab())
        {
            return Err("invalid isolated layer tokens".into());
        }
        let mut session = self.new_session(SessionId(layer as u64 + 1))?;
        let mut plan = PrefillPlan::new(
            &self.model,
            tokens,
            0,
            self.gdn_conv_tile,
            self.gdn_value_tile,
            ScoredRows::last(tokens.len())?,
            &self.stream,
        )?;
        prefill::run_isolated_layer(
            &self.model,
            &mut plan,
            &mut session,
            layer,
            input,
            &self.stream,
        )
    }

    #[cfg(feature = "testing")]
    pub fn step_eager_routing_trace(
        &mut self,
        session: &mut Session,
        token: u32,
    ) -> Result<(Vec<f32>, Vec<RoutingLayerTrace>)> {
        let batch = self.batch([(session, token)])?;
        self.step_with(batch, |model, plan, sessions, stream| {
            crate::execute::run_routing_observed(model, plan, sessions, stream)
        })
    }

    #[cfg(feature = "testing")]
    pub fn state_snapshot(&self, session: &Session) -> Result<Vec<SessionStateSnapshot>> {
        if session.engine != self.id || session.poisoned {
            return Err("invalid session for snapshot".into());
        }
        let mut result = Vec::new();
        for (layer, state) in session.state.iter().enumerate() {
            let parts: [(&str, &cutile::tensor::Tensor<f32>); 2] = match state {
                crate::session::LayerState::Gdn { conv, recurrent } => {
                    [("conv", conv), ("recurrent", recurrent)]
                }
                crate::session::LayerState::Attention { k, v } => [("k", k), ("v", v)],
            };
            for (name, tensor) in parts {
                let values = match state {
                    crate::session::LayerState::Attention { .. } => {
                        let shape = tensor.shape();
                        let mut values: Vec<f32> =
                            tensor.dup().to_host_vec().sync_on(&self.stream)?;
                        values.truncate(session.position * shape[1] as usize * shape[2] as usize);
                        values
                    }
                    _ => tensor.dup().to_host_vec().sync_on(&self.stream)?,
                };
                result.push(SessionStateSnapshot {
                    layer,
                    name,
                    values,
                });
            }
        }
        Ok(result)
    }

    fn prefill_with<T>(
        &mut self,
        session: &mut Session,
        tokens: &[u32],
        scored: ScoredRows,
        execute: impl FnOnce(&Model, &mut PrefillPlan, &mut Session, &Stream) -> Result<T>,
    ) -> Result<(Vec<f32>, T)> {
        let result = (|| -> Result<(Vec<f32>, T)> {
            if session.engine != self.id
                || session.poisoned
                || tokens.is_empty()
                || session
                    .position
                    .checked_add(tokens.len())
                    .is_none_or(|end| end > self.model.dims.max_seq())
                || tokens
                    .iter()
                    .any(|&token| token as usize >= self.model.dims.vocab())
            {
                return Err("invalid prefill session or tokens".into());
            }
            for op in StepPlan::ops(&self.model) {
                if let crate::plan::Op::Gemv { layer, name, .. } = op {
                    if !matches!(
                        crate::execute::weight(&self.model, layer, name),
                        firepit_kernels::weights::Weight::Dense(_)
                    ) {
                        return Err("prefill requires dense bf16 weights".into());
                    }
                }
            }
            let mut plan = PrefillPlan::new(
                &self.model,
                tokens,
                session.position,
                self.gdn_conv_tile,
                self.gdn_value_tile,
                scored,
                &self.stream,
            )?;
            let output = execute(&self.model, &mut plan, session, &self.stream)?;
            Ok((plan.logits(&self.stream)?, output))
        })();
        match result {
            Ok((logits, output)) => {
                session.position += tokens.len();
                Ok((logits, output))
            }
            Err(error) => {
                session.poisoned = true;
                Err(error)
            }
        }
    }

    pub fn step(&mut self, batch: Batch<'_>) -> Result<Vec<f32>> {
        self.step_with(batch, |model, plan, sessions, stream| {
            if plan.graph.is_none() {
                let graph = CudaGraph::scope(stream, |scope| {
                    run(model, plan, sessions, &mut Graph(scope))
                        .map_err(|e| cuda_async::error::DeviceError::Internal(e.to_string()))
                })?;
                plan.graph = Some(graph);
            }
            plan.graph.as_ref().unwrap().launch().sync_on(stream)?;
            Ok(())
        })
        .map(|(logits, ())| logits)
    }

    fn step_with<T>(
        &mut self,
        mut batch: Batch<'_>,
        execute: impl FnOnce(&Model, &mut StepPlan, &mut [&mut Session], &Stream) -> Result<T>,
    ) -> Result<(Vec<f32>, T)> {
        if batch.engine != self.id {
            return Err("batch belongs to another engine".into());
        }
        let actual = batch.sessions.len();
        let size = actual.next_power_of_two();
        let instances: Vec<u64> = batch.sessions.iter().map(|s| s.instance).collect();
        if self
            .plans
            .get(&size)
            .is_none_or(|entry| entry.instances != instances)
        {
            self.plans.insert(
                size,
                CachedPlan {
                    instances,
                    plan: StepPlan::new(
                        &self.model,
                        size,
                        self.gdn_value_tile,
                        self.gdn_conv_tile,
                        &self.stream,
                    )?,
                },
            );
        }
        let plan = &mut self.plans.get_mut(&size).unwrap().plan;
        let result = (|| -> Result<(Vec<f32>, T)> {
            let mut tokens: Vec<i32> = batch.tokens.iter().map(|&x| x as i32).collect();
            tokens.resize(size, 0);
            let source = api::copy_host_vec_to_device(&Arc::new(tokens)).sync_on(&self.stream)?;
            api::memcpy(&mut plan.tokens, &source).sync_on(&self.stream)?;
            let positions: Vec<usize> = batch
                .sessions
                .iter()
                .map(|s| s.position)
                .chain(std::iter::repeat(0))
                .take(size)
                .collect();
            for (position, &value) in plan.positions.iter_mut().zip(&positions) {
                let source = api::copy_host_vec_to_device(&Arc::new(vec![value as i32]))
                    .sync_on(&self.stream)?;
                api::memcpy(position, &source).sync_on(&self.stream)?;
            }
            let (cos, sin) = rope::frequencies(&positions, &self.model.dims.q_rope());
            let cosine = api::copy_host_vec_to_device(&Arc::new(cos))
                .reshape(&[size, self.model.dims.attention().head_dim()])
                .sync_on(&self.stream)?;
            let sine = api::copy_host_vec_to_device(&Arc::new(sin))
                .reshape(&[size, self.model.dims.attention().head_dim()])
                .sync_on(&self.stream)?;
            api::memcpy(&mut plan.cos, &cosine).sync_on(&self.stream)?;
            api::memcpy(&mut plan.sin, &sine).sync_on(&self.stream)?;
            let output = execute(&self.model, plan, &mut batch.sessions, &self.stream)?;
            #[cfg(test)]
            if self.fail_after_gpu_work {
                return Err("injected failure after GPU work".into());
            }
            Ok((plan.logits(&self.stream)?, output))
        })();
        let (canonical, output) = match result {
            Ok(result) => result,
            Err(error) => {
                for session in &mut batch.sessions {
                    session.poisoned = true;
                }
                return Err(error);
            }
        };
        for session in &mut batch.sessions {
            session.position += 1;
        }
        let vocab = self.model.dims.vocab();
        let mut logits = vec![0.0; actual * vocab];
        for (canonical_row, original_row) in batch.original_rows.into_iter().enumerate() {
            logits[original_row * vocab..(original_row + 1) * vocab]
                .copy_from_slice(&canonical[canonical_row * vocab..(canonical_row + 1) * vocab]);
        }
        Ok((logits, output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SafetensorsFile;
    use crate::plan::Op;
    use crate::session::LayerState;
    use cutile::tensor::Tensor;
    use cutile::tile_kernel::ToHostVecOp;
    use std::path::Path;

    fn small_model(device: &Arc<Device>) -> Result<Arc<Model>> {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/small-qwen.json"))?;
        Ok(Arc::new(Model::random_from_config(
            &config["text_config"],
            8,
            7,
            &device.new_stream()?,
        )?))
    }

    fn assert_tensor_bits(
        a: &Tensor<f32>,
        a_stream: &Stream,
        b: &Tensor<f32>,
        b_stream: &Stream,
    ) -> Result<()> {
        let a: Vec<f32> = a.dup().to_host_vec().sync_on(a_stream)?;
        let b: Vec<f32> = b.dup().to_host_vec().sync_on(b_stream)?;
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }

    fn assert_session_bits(
        a: &Session,
        a_stream: &Stream,
        b: &Session,
        b_stream: &Stream,
    ) -> Result<()> {
        assert_eq!(a.position, b.position);
        for (left, right) in a.state.iter().zip(&b.state) {
            match (left, right) {
                (
                    LayerState::Gdn {
                        conv: ac,
                        recurrent: ar,
                    },
                    LayerState::Gdn {
                        conv: bc,
                        recurrent: br,
                    },
                ) => {
                    assert_tensor_bits(ac, a_stream, bc, b_stream)?;
                    assert_tensor_bits(ar, a_stream, br, b_stream)?;
                }
                (
                    LayerState::Attention { k: ak, v: av },
                    LayerState::Attention { k: bk, v: bv },
                ) => {
                    assert_tensor_bits(ak, a_stream, bk, b_stream)?;
                    assert_tensor_bits(av, a_stream, bv, b_stream)?;
                }
                _ => panic!("different layer kinds"),
            }
        }
        Ok(())
    }

    fn top5(values: &[f32]) -> Vec<usize> {
        let mut indices: Vec<_> = (0..values.len()).collect();
        indices.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        indices.truncate(5);
        indices.sort_unstable();
        indices
    }

    fn assert_trace(trace: &Trace, expected: impl IntoIterator<Item = &'static str>) {
        let kinds: Vec<_> = expected.into_iter().collect();
        assert_eq!(
            trace.spans.iter().map(|span| span.kind).collect::<Vec<_>>(),
            kinds
        );
        let summary = trace.summary();
        assert_eq!(
            summary.iter().map(|row| row.count).sum::<usize>(),
            trace.spans.len()
        );
        for row in summary {
            let spans: Vec<_> = trace
                .spans
                .iter()
                .filter(|span| span.kind == row.kind)
                .collect();
            assert_eq!(
                row.gpu_ms,
                spans.iter().map(|span| span.gpu_ms).sum::<f64>()
            );
            assert_eq!(
                row.wall_ms,
                spans.iter().map(|span| span.wall_ms).sum::<f64>()
            );
        }
    }

    #[test]
    fn traced_execution_matches_eager_bits_and_plan_order() -> Result<()> {
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        let mut eager = Engine::new(model.clone(), &device)?;
        let mut traced = Engine::new(model, &device)?;
        let mut eager_session = eager.new_session(SessionId(1))?;
        let mut traced_session = traced.new_session(SessionId(1))?;

        let plain_logits = eager.prefill(&mut eager_session, &[1, 2])?;
        let (traced_logits, trace) = traced.prefill_traced(&mut traced_session, &[1, 2])?;
        assert_eq!(
            plain_logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            traced_logits
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_session_bits(
            &eager_session,
            &eager.stream,
            &traced_session,
            &traced.stream,
        )?;
        let plan = PrefillPlan::new(
            &traced.model,
            &[1, 2],
            0,
            traced.gdn_conv_tile,
            traced.gdn_value_tile,
            ScoredRows::last(2)?,
            &traced.stream,
        )?;
        assert_trace(&trace, plan.ops.iter().map(|op| op.kind()));
        assert!(
            trace
                .spans
                .iter()
                .filter(|span| span.kind == "moe_host_sort")
                .all(|span| span.gpu_ms == 0.0)
        );

        let plain_logits = eager.step_eager(eager.batch([(&mut eager_session, 3)])?)?;
        let (traced_logits, trace) =
            traced.step_eager_traced(traced.batch([(&mut traced_session, 3)])?)?;
        assert_eq!(
            plain_logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            traced_logits
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_session_bits(
            &eager_session,
            &eager.stream,
            &traced_session,
            &traced.stream,
        )?;
        assert_trace(
            &trace,
            StepPlan::ops(&traced.model).into_iter().map(Op::kind),
        );
        Ok(())
    }

    #[test]
    fn gpu_failure_poisons_session() -> Result<()> {
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        let mut engine = Engine::new(model, &device)?;
        let mut session = engine.new_session(SessionId(1))?;
        engine.fail_after_gpu_work = true;
        let batch = engine.batch([(&mut session, 1)])?;
        assert!(engine.step_eager(batch).is_err());
        assert_eq!(session.position(), 0);
        assert!(engine.batch([(&mut session, 1)]).is_err());
        Ok(())
    }

    #[test]
    fn prefill_small_model_runs() -> Result<()> {
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        let mut engine = Engine::new(model, &device)?;
        let mut session = engine.new_session(SessionId(1))?;
        let logits = engine.prefill(&mut session, &[1, 2])?;
        assert_eq!(logits.len(), engine.model.dims.vocab());
        assert_eq!(session.position(), 2);
        Ok(())
    }

    #[test]
    fn invalid_prefill_poisons_session() -> Result<()> {
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        let mut engine = Engine::new(model, &device)?;
        for tokens in [vec![], vec![engine.model.dims.vocab() as u32]] {
            let mut session = engine.new_session(SessionId(1))?;
            assert!(engine.prefill(&mut session, &tokens).is_err());
            assert_eq!(session.position(), 0);
            assert!(engine.batch([(&mut session, 1)]).is_err());
            engine.close_session(session)?;
        }
        let mut session = engine.new_session(SessionId(1))?;
        let overlong = vec![1; engine.model.dims.max_seq() + 1];
        assert!(engine.prefill(&mut session, &overlong).is_err());
        assert_eq!(session.position(), 0);
        assert!(engine.batch([(&mut session, 1)]).is_err());
        Ok(())
    }

    #[test]
    fn prefill_small_model_diagnostics() -> Result<()> {
        use cutile::tile_kernel::ToHostVecOp;
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        let mut prefill_engine = Engine::new(model.clone(), &device)?;
        let mut decode_engine = Engine::new(model, &device)?;
        let mut p = prefill_engine.new_session(SessionId(1))?;
        let mut s = decode_engine.new_session(SessionId(1))?;
        for tokens in [[1, 2].as_slice(), [3, 4].as_slice()] {
            let actual = prefill_engine.prefill(&mut p, tokens)?;
            let mut expected = Vec::new();
            for &token in tokens {
                let batch = decode_engine.batch([(&mut s, token)])?;
                expected = decode_engine.step_eager(batch)?;
            }
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            println!("position={} logits max_abs={max}", p.position());
            assert!(actual.iter().all(|x| x.is_finite()));
            for (layer, (ps, ds)) in p.state.iter().zip(&s.state).enumerate() {
                match (ps, ds) {
                    (
                        crate::session::LayerState::Gdn {
                            conv: pc,
                            recurrent: pr,
                        },
                        crate::session::LayerState::Gdn {
                            conv: dc,
                            recurrent: dr,
                        },
                    ) => {
                        for (name, pv, dv) in [("conv", pc, dc), ("recurrent", pr, dr)] {
                            let a = pv.dup().to_host_vec().sync_on(&prefill_engine.stream)?;
                            let b = dv.dup().to_host_vec().sync_on(&decode_engine.stream)?;
                            let max = a
                                .iter()
                                .zip(&b)
                                .map(|(x, y)| (x - y).abs())
                                .fold(0.0f32, f32::max);
                            println!("layer={layer} {name} max_abs={max}");
                        }
                    }
                    (
                        crate::session::LayerState::Attention { k: pk, v: pv },
                        crate::session::LayerState::Attention { k: dk, v: dv },
                    ) => {
                        for (name, x, y) in [("k", pk, dk), ("v", pv, dv)] {
                            let a = x.dup().to_host_vec().sync_on(&prefill_engine.stream)?;
                            let b = y.dup().to_host_vec().sync_on(&decode_engine.stream)?;
                            let end = p.position() * a.len() / prefill_engine.model.dims.max_seq();
                            let max = a[..end]
                                .iter()
                                .zip(&b[..end])
                                .map(|(x, y)| (x - y).abs())
                                .fold(0.0f32, f32::max);
                            println!("layer={layer} {name} max_abs={max}");
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
        Ok(())
    }

    #[test]
    fn scored_rows_match_last_row_and_steps() -> Result<()> {
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        let mut plain = Engine::new(model.clone(), &device)?;
        let mut scored = Engine::new(model.clone(), &device)?;
        let mut decode = Engine::new(model, &device)?;
        let mut plain_session = plain.new_session(SessionId(1))?;
        let mut scored_session = scored.new_session(SessionId(1))?;
        let mut decode_session = decode.new_session(SessionId(1))?;
        let tokens = [1, 2, 3, 4];
        let last = plain.prefill(&mut plain_session, &tokens)?;
        let scored_last = scored.prefill_scored(
            &mut scored_session,
            &tokens,
            ScoredRows::last(tokens.len())?,
        )?;
        assert_eq!(
            last.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            scored_last.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );

        let mut all = Engine::new(Arc::clone(&decode.model), &device)?;
        let mut all_session = all.new_session(SessionId(1))?;
        let rows = all.prefill_scored(
            &mut all_session,
            &tokens,
            ScoredRows::parse(1..4, tokens.len())?,
        )?;
        let vocab = rows.len() / 3;
        for (index, &token) in tokens.iter().enumerate() {
            let batch = decode.batch([(&mut decode_session, token)])?;
            let step = decode.step_eager(batch)?;
            if index >= 1 {
                let row = &rows[(index - 1) * vocab..index * vocab];
                let max_abs = row
                    .iter()
                    .zip(&step)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!("scored row {index} vs step max_abs={max_abs}");
                assert!(
                    row.iter()
                        .zip(&step)
                        .all(|(a, b)| (a - b).abs() <= 3e-4 * (1.0 + b.abs()))
                );
            }
        }
        assert!(ScoredRows::parse(0..0, 4).is_err());
        assert!(ScoredRows::parse(0..5, 4).is_err());
        Ok(())
    }

    #[test]
    fn reset_rope_position_breaks_append() -> Result<()> {
        let device = Device::new(0)?;
        let mut config: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/small-qwen.json"))?;
        config["text_config"]["layer_types"] =
            serde_json::json!(["full_attention", "full_attention"]);
        let model = Arc::new(Model::random_from_config(
            &config["text_config"],
            8,
            7,
            &device.new_stream()?,
        )?);
        let mut whole = Engine::new(model.clone(), &device)?;
        let mut append = Engine::new(model, &device)?;
        let mut ws = whole.new_session(SessionId(1))?;
        let mut as_ = append.new_session(SessionId(1))?;
        let expected = whole.prefill(&mut ws, &[1, 2, 3, 4])?;
        append.prefill(&mut as_, &[1, 2])?;
        let (actual, _) =
            append.prefill_observed(&mut as_, &[3, 4], VerificationFault::ResetRopePosition)?;
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("reset rope append logits max_abs={max}");
        println!(
            "reset rope top5={:?} expected={:?}",
            top5(&actual),
            top5(&expected)
        );
        assert_ne!(
            top5(&actual),
            top5(&expected),
            "resetting RoPE must fail append top5"
        );
        Ok(())
    }

    #[test]
    fn first_row_lm_head_breaks_prefill_vs_step() -> Result<()> {
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        let mut prefill = Engine::new(model.clone(), &device)?;
        let mut decode = Engine::new(model, &device)?;
        let mut ps = prefill.new_session(SessionId(1))?;
        let mut ds = decode.new_session(SessionId(1))?;
        let (actual, _) =
            prefill.prefill_observed(&mut ps, &[1, 2], VerificationFault::FirstRowLmHead)?;
        decode.step_eager(decode.batch([(&mut ds, 1)])?)?;
        let expected = decode.step_eager(decode.batch([(&mut ds, 2)])?)?;
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("first row lm_head prefill/step logits max_abs={max}");
        println!(
            "first row lm_head top5={:?} expected={:?}",
            top5(&actual),
            top5(&expected)
        );
        assert_ne!(
            top5(&actual),
            top5(&expected),
            "first-row lm_head must fail prefill/step top5"
        );
        Ok(())
    }

    #[test]
    fn small_model_batch_matches_individual_and_graph() -> Result<()> {
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        for size in 1..=4 {
            let mut together = Engine::new(model.clone(), &device)?;
            let mut graph = Engine::new(model.clone(), &device)?;
            let mut singles = (0..size)
                .map(|_| Engine::new(model.clone(), &device))
                .collect::<Result<Vec<_>>>()?;
            let mut together_sessions = (0..size)
                .map(|i| together.new_session(SessionId(i as u64 + 1)))
                .collect::<Result<Vec<_>>>()?;
            let mut graph_sessions = (0..size)
                .map(|i| graph.new_session(SessionId(i as u64 + 1)))
                .collect::<Result<Vec<_>>>()?;
            let mut single_sessions = singles
                .iter()
                .enumerate()
                .map(|(i, e)| e.new_session(SessionId(i as u64 + 10)))
                .collect::<Result<Vec<_>>>()?;
            for step in 0..3 {
                let tokens: Vec<u32> = (0..size).map(|i| (i * 13 + step * 5 + 3) as u32).collect();
                let expected = together.step_eager(
                    together.batch(together_sessions.iter_mut().zip(tokens.iter().copied()))?,
                )?;
                let actual = graph
                    .step(graph.batch(graph_sessions.iter_mut().zip(tokens.iter().copied()))?)?;
                assert_eq!(expected, actual);
                for i in 0..size {
                    let single_batch = singles[i].batch([(&mut single_sessions[i], tokens[i])])?;
                    let single = singles[i].step_eager(single_batch)?;
                    let max_error = expected[i * 64..(i + 1) * 64]
                        .iter()
                        .zip(&single)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_error <= 1e-5,
                        "B={size} step={step} row={i} max_abs={max_error}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "needs local Qwen3.6 checkpoint and GPU"]
    fn prefix_logits_match_hf_eager() -> Result<()> {
        let dir = Path::new(env!("HOME")).join("Models/qwen3.6-35b-a3b");
        let device = Device::new(0)?;
        let refs = SafetensorsFile::open(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4.safetensors"),
        )?;
        let tokens: Vec<u32> =
            serde_json::from_str(refs.metadata("tokens").ok_or("missing tokens")?)?;
        let mut model = Arc::new(Model::from_checkpoint(
            dir,
            crate::LoadOptions {
                max_seq: tokens.len(),
                layers: Some(4),
            },
            &device,
        )?);
        let mut graph_engine = Engine::new(model.clone(), &device)?;
        let mut graph_session = graph_engine.new_session(SessionId(2))?;
        let mut engine = Engine::new(model.clone(), &device)?;
        let mut session = engine.new_session(SessionId(1))?;
        for (step, &token) in tokens.iter().enumerate() {
            let batch = engine.batch([(&mut session, token)])?;
            let actual = engine.step_eager(batch)?;
            let graph_batch = graph_engine.batch([(&mut graph_session, token)])?;
            let graphed = graph_engine.step(graph_batch)?;
            let expected = refs.tensor(&format!("step{step}/prefix_logits"))?;
            let relative = actual
                .iter()
                .zip(&expected.data)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                / expected
                    .data
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
            let top = |values: &[f32]| {
                values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            println!(
                "step{step} relative={relative:.8e} top={} hf_top={}",
                top(&actual),
                top(&expected.data)
            );
            assert_eq!(actual, graphed);
            assert_eq!(top(&actual), top(&expected.data));
            assert!(relative < 1e-3);
        }
        drop(session);
        drop(graph_session);
        drop(engine);
        drop(graph_engine);
        Arc::get_mut(&mut model)
            .ok_or("model still shared")?
            .layers
            .swap(0, 1);
        let mut shuffled = Engine::new(model, &device)?;
        let mut shuffled_session = shuffled.new_session(SessionId(3))?;
        let shuffled_batch = shuffled.batch([(&mut shuffled_session, tokens[0])])?;
        let shuffled_logits = shuffled.step_eager(shuffled_batch)?;
        let expected = refs.tensor("step0/prefix_logits")?;
        let error = shuffled_logits
            .iter()
            .zip(&expected.data)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            .sqrt()
            / expected
                .data
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
        println!("shuffled_layer_relative={error:.8e}");
        assert!(error > 1e-3);
        Ok(())
    }
}
