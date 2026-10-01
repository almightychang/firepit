use crate::model::{RowTile, Stream};
use crate::plan::{Op, Slot, StepPlan};
use crate::session::LayerState;
use crate::{Batch, Model, Result, Session, SessionId};
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api;
use cutile::tensor::{IntoPartition, PartitionMut, Tensor};
use cutile::tile_kernel::{CompileOptions, TileKernel};
use firepit_kernels::gemv::{DEFAULT_GEMV_TILE, gemv};
use firepit_kernels::sink::{Eager, Graph, Sink};
use firepit_kernels::tile::{MIN_ROW_TILE, row_tile};
use firepit_kernels::weights::Q8_BLOCK;
use firepit_kernels::{elementwise, embedding, norm, synthetic};
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

const TARGET_BLOCKS_PER_SM: i32 = 1;
const MAX_BATCH_SESSIONS: usize = 4;
static NEXT_ENGINE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct EngineId(u64);

fn input_output(
    buffers: &mut [Tensor<f32>],
    input: Slot,
    output: Slot,
) -> (&Tensor<f32>, &mut Tensor<f32>) {
    let input = input.index();
    let output = output.index();
    assert_ne!(input, output);
    let (before, from_output) = buffers.split_at_mut(output);
    let (out, after) = from_output.split_first_mut().unwrap();
    let input = if input < output {
        &before[input]
    } else {
        &after[input - output - 1]
    };
    (input, out)
}

fn two_inputs_output(
    buffers: &mut [Tensor<f32>],
    first: Slot,
    second: Slot,
    output: Slot,
) -> (&Tensor<f32>, &Tensor<f32>, &mut Tensor<f32>) {
    let out_index = output.index();
    let (before, at_output) = buffers.split_at_mut(out_index);
    let (out, after) = at_output.split_first_mut().unwrap();
    let get = |slot: Slot| -> &Tensor<f32> {
        let index = slot.index();
        assert_ne!(index, out_index);
        if index < out_index {
            &before[index]
        } else {
            &after[index - out_index - 1]
        }
    };
    (get(first), get(second), out)
}

fn run(
    model: &Model,
    plan: &mut StepPlan,
    sessions: &mut [&mut Session],
    sink: &mut impl Sink,
) -> Result<()> {
    let batch = plan.batch;
    let c = &model.dims();
    let (h, heads, d, max_seq) = (c.hidden(), c.heads(), c.head_dim(), c.max_seq());
    for op in &plan.ops {
        match *op {
            Op::Embed { out: output } => {
                sink.run(
                    embedding::embedding(
                        (&mut plan.buffers[output.index()]).partition([1, h]),
                        &model.embedding,
                        &plan.tokens,
                    )
                    .generics(vec![h.to_string()]),
                )?;
            }
            Op::RmsNorm {
                norm,
                input,
                out: output,
            } => {
                let (activation, out) = input_output(&mut plan.buffers, input, output);
                sink.run(
                    norm::rms_norm(
                        out.partition([1, h]),
                        activation,
                        model.norm(norm),
                        1.0 / h as f32,
                        c.rms_norm_eps(),
                    )
                    .generics(vec![h.to_string()]),
                )?;
            }
            Op::Add {
                residual: input,
                delta: second,
                out: output,
            } => {
                let (current, delta, out) =
                    two_inputs_output(&mut plan.buffers, input, second, output);
                sink.run(
                    elementwise::add(out.partition([1, h]), current, delta)
                        .generics(vec![h.to_string()]),
                )?;
            }
            Op::Gemv {
                weight,
                input,
                out: output,
            } => {
                let (input_tensor, out) = input_output(&mut plan.buffers, input, output);
                let activation = input_tensor.view(&[batch, h])?;
                let x = activation.view(&[batch, h / Q8_BLOCK, Q8_BLOCK])?;
                gemv(
                    model.weights.get(weight),
                    &x,
                    out.partition([batch, DEFAULT_GEMV_TILE.rows]),
                    DEFAULT_GEMV_TILE,
                )
                .submit(sink)?;
            }
            Op::Session {
                layer,
                input,
                out: output,
            } => {
                let (input_tensor, out) = input_output(&mut plan.buffers, input, output);
                for (b, session) in sessions.iter_mut().enumerate() {
                    let row_input = input_tensor.slice(&[b..b + 1, 0..h])?;
                    let row_output = out.slice_mut(&[b * heads..(b + 1) * heads, 0..d])?;
                    match &mut session.state[layer] {
                        LayerState::Gdn { recurrent } => {
                            let r = plan.row.0;
                            sink.run(
                                synthetic::gdn_step(
                                    row_output.partition(c.session_output_tile(plan.row)),
                                    recurrent.partition(c.recurrent_tile(plan.row)),
                                    &row_input,
                                    1.0 / d as f32,
                                    c.gdn_decay(),
                                )
                                .generics(vec![heads.to_string(), d.to_string(), r.to_string()])
                                .compile_options(
                                    CompileOptions::default().occupancy(TARGET_BLOCKS_PER_SM),
                                ),
                            )?;
                        }
                        LayerState::Attention { k, v } => {
                            sink.run(
                                synthetic::kv_append(
                                    k.partition(c.cache_tile(plan.row)),
                                    v.partition(c.cache_tile(plan.row)),
                                    &row_input,
                                    &plan.positions[b],
                                )
                                .generics(vec![
                                    max_seq.to_string(),
                                    heads.to_string(),
                                    d.to_string(),
                                    plan.row.0.to_string(),
                                ])
                                .compile_options(
                                    CompileOptions::default().occupancy(TARGET_BLOCKS_PER_SM),
                                ),
                            )?;
                            sink.run(
                                synthetic::attn_step(
                                    row_output.partition(c.session_output_tile(plan.row)),
                                    &*k,
                                    &*v,
                                    &row_input,
                                    &plan.positions[b],
                                    1.0 / d as f32,
                                )
                                .generics(vec![
                                    heads.to_string(),
                                    d.to_string(),
                                    plan.row.0.to_string(),
                                ])
                                .compile_options(
                                    CompileOptions::default().occupancy(TARGET_BLOCKS_PER_SM),
                                ),
                            )?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

pub struct Engine {
    model: Arc<Model>,
    id: EngineId,
    stream: Stream,
    row: RowTile,
    plans: HashMap<usize, CachedPlan>,
    #[cfg(test)]
    fail_after_gpu_work: bool,
}

struct CachedPlan {
    instances: Vec<crate::session::SessionInstance>,
    plan: StepPlan,
}

impl Engine {
    pub fn new(model: Arc<Model>, device: &Arc<Device>) -> Result<Self> {
        let stream = device.new_stream()?;
        let sm =
            cuda_core::simt::CudaContext::new(device.ordinal())?.multiprocessor_count()? as usize;
        let row = RowTile(row_tile(
            model.dims().head_dim(),
            model.dims().heads(),
            sm,
            TARGET_BLOCKS_PER_SM as usize,
            MIN_ROW_TILE,
        ));
        Ok(Self {
            model,
            id: EngineId(NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed)),
            stream,
            row,
            plans: HashMap::new(),
            #[cfg(test)]
            fail_after_gpu_work: false,
        })
    }
    pub fn new_session(&self, id: SessionId) -> Result<Session> {
        self.model.new_session(id, self.id, &self.stream)
    }

    pub fn close_session(&mut self, session: Session) -> Result<()> {
        if session.engine != self.id {
            return Err("session belongs to another engine".into());
        }
        self.plans
            .retain(|_, entry| !entry.instances.contains(&session.instance));
        Ok(())
    }

    pub fn batch<'s>(
        &self,
        entries: impl IntoIterator<Item = (&'s mut Session, u32)>,
    ) -> Result<Batch<'s>> {
        let entries: Vec<_> = entries.into_iter().collect();
        if entries.is_empty() || entries.len() > MAX_BATCH_SESSIONS {
            return Err(format!("batch must have 1..={MAX_BATCH_SESSIONS} sessions").into());
        }
        let mut ids = HashSet::new();
        for (session, token) in &entries {
            if !ids.insert(session.id) {
                return Err("duplicate session id".into());
            }
            if session.engine != self.id {
                return Err("session belongs to another engine".into());
            }
            if *token as usize >= self.model.dims().vocab() {
                return Err("token outside vocabulary".into());
            }
            if session.position >= self.model.dims().max_seq() {
                return Err("session reached max_seq".into());
            }
            if session.poisoned {
                return Err("session is poisoned".into());
            }
        }
        Ok(Batch::from_checked(self.id, entries))
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
    }
    pub fn step_eager(&mut self, batch: Batch<'_>) -> Result<Vec<f32>> {
        self.step_with(batch, |model, plan, sessions, stream| {
            run(model, plan, sessions, &mut Eager(stream))
        })
    }
    fn step_with(
        &mut self,
        mut batch: Batch<'_>,
        execute: impl FnOnce(&Model, &mut StepPlan, &mut [&mut Session], &Stream) -> Result<()>,
    ) -> Result<Vec<f32>> {
        if batch.engine != self.id {
            return Err("batch belongs to another engine".into());
        }
        let actual = batch.sessions.len();
        let size = actual.next_power_of_two();
        let instances: Vec<crate::session::SessionInstance> = batch
            .sessions
            .iter()
            .map(|session| session.instance)
            .collect();
        if self
            .plans
            .get(&size)
            .is_none_or(|entry| entry.instances != instances)
        {
            let plan = StepPlan::new(&self.model, size, self.row, &self.stream)?;
            self.plans.insert(size, CachedPlan { instances, plan });
        }
        let plan = &mut self.plans.get_mut(&size).unwrap().plan;
        let mut tokens: Vec<i32> = batch.tokens.iter().map(|&x| x as i32).collect();
        tokens.resize(size, 0);
        let result = (|| -> Result<Vec<f32>> {
            let source = api::copy_host_vec_to_device(&Arc::new(tokens)).sync_on(&self.stream)?;
            api::memcpy(&mut plan.tokens, &source).sync_on(&self.stream)?;
            for (slot, session) in plan.positions.iter_mut().zip(&batch.sessions) {
                let source = api::copy_host_vec_to_device(&Arc::new(vec![session.position as i32]))
                    .sync_on(&self.stream)?;
                api::memcpy(slot, &source).sync_on(&self.stream)?;
            }
            execute(&self.model, plan, &mut batch.sessions, &self.stream)?;
            #[cfg(test)]
            if self.fail_after_gpu_work {
                return Err("injected failure after GPU work".into());
            }
            plan.logits(&self.stream)
        })();
        let canonical_logits = match result {
            Ok(logits) => logits,
            Err(error) => {
                for session in &mut batch.sessions {
                    session.poisoned = true;
                }
                return Err(error);
            }
        };
        for session in batch.sessions {
            session.position += 1;
        }
        let vocab = self.model.dims().vocab();
        let mut logits = vec![0.0f32; actual * vocab];
        for (canonical, original) in batch.original_rows.into_iter().enumerate() {
            logits[original * vocab..(original + 1) * vocab]
                .copy_from_slice(&canonical_logits[canonical * vocab..(canonical + 1) * vocab]);
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelConfig, ModelDims};

    #[test]
    fn gpu_failure_poisons_batch_sessions() -> Result<()> {
        let device = Device::new(0)?;
        let stream = device.new_stream()?;
        let dims = ModelDims::try_from(ModelConfig::default())?;
        let model = Arc::new(Model::synthetic(dims, &stream)?);
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
    fn cache_is_bounded_and_close_removes_related_plans() -> Result<()> {
        let device = Device::new(0)?;
        let stream = device.new_stream()?;
        let dims = ModelDims::try_from(ModelConfig::default())?;
        let model = Arc::new(Model::synthetic(dims, &stream)?);
        let mut engine = Engine::new(model.clone(), &device)?;
        let mut sessions = (1..=5)
            .map(|id| engine.new_session(SessionId(id)))
            .collect::<Result<Vec<_>>>()?;

        for size in 1..=MAX_BATCH_SESSIONS {
            let batch = engine.batch(sessions[..size].iter_mut().map(|session| (session, 1)))?;
            engine.step(batch)?;
            if size == 3 {
                assert_eq!(engine.plans.get(&4).unwrap().instances.len(), 3);
            }
        }
        assert_eq!(engine.plans.len(), 3);
        assert_eq!(engine.plans.get(&4).unwrap().instances.len(), 4);

        let original = engine.plans.get(&1).unwrap().instances[0];
        let batch = engine.batch([(&mut sessions[4], 1)])?;
        engine.step(batch)?;
        assert!(engine.plans.get(&1).unwrap().instances[0] != original);
        assert_eq!(engine.plans.len(), 3);

        engine.close_session(sessions.remove(0))?;
        assert_eq!(engine.plans.len(), 1);
        assert!(engine.plans.contains_key(&1));
        Ok(())
    }
}
