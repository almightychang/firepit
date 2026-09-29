use crate::model::Stream;
use crate::plan::{Op, SessionKind, Slot, StepPlan};
use crate::session::LayerState;
use crate::{Batch, Model, Result, Session, SessionId};
use cuda_async::cuda_graph::{CudaGraph, Scope};
use cuda_async::device_operation::{DeviceOp, GraphNode};
use cuda_core::Device;
use cutile::api;
use cutile::tensor::{IntoPartition, PartitionMut, Tensor};
use cutile::tile_kernel::{CompileOptions, TileKernel};
use firepit_kernels::hybrid;
use firepit_kernels::q8::{DEFAULT_GEMV_TILE, q8_gemv};
use firepit_kernels::tile::{MIN_ROW_TILE, row_tile};
use std::collections::HashMap;
use std::sync::Arc;

const TARGET_BLOCKS_PER_SM: i32 = 1;

enum Sink<'a> {
    Eager(&'a Stream),
    Graph(&'a Scope),
}
fn submit<T: Send, N: GraphNode + DeviceOp<Output = T>>(sink: &Sink<'_>, op: N) -> Result<()> {
    match sink {
        Sink::Eager(stream) => {
            op.sync_on(stream)?;
        }
        Sink::Graph(scope) => {
            scope.record(op)?;
        }
    }
    Ok(())
}

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
    sink: &Sink<'_>,
) -> Result<()> {
    let batch = sessions.len();
    let c = &model.config;
    let (h, heads, d, max_seq) = (c.hidden, c.heads, c.head_dim, c.max_seq);
    for op in &plan.ops {
        match *op {
            Op::Embed { out: output } => {
                submit(
                    sink,
                    hybrid::embedding(
                        (&mut plan.buffers[output.index()]).partition([1, h]),
                        &model.embedding,
                        &plan.tokens,
                    )
                    .generics(vec![h.to_string()]),
                )?;
            }
            Op::RmsNorm { input, out: output } => {
                let (activation, out) = input_output(&mut plan.buffers, input, output);
                submit(
                    sink,
                    hybrid::rms_norm(out.partition([1, h]), activation, 1.0 / h as f32)
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
                submit(
                    sink,
                    hybrid::add(out.partition([1, h]), current, delta)
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
                let x = activation.view(&[batch, h / 32, 32])?;
                let w = &model.weights[weight];
                submit(
                    sink,
                    q8_gemv(
                        w,
                        &x,
                        out.partition([batch, DEFAULT_GEMV_TILE.rows]),
                        DEFAULT_GEMV_TILE,
                    ),
                )?;
            }
            Op::Session {
                layer,
                kind,
                input,
                out: output,
            } => {
                let (input_tensor, out) = input_output(&mut plan.buffers, input, output);
                for (b, session) in sessions.iter_mut().enumerate() {
                    let row_input = input_tensor.slice(&[b..b + 1, 0..h])?;
                    let row_output = out.slice_mut(&[b * heads..(b + 1) * heads, 0..d])?;
                    match (&mut session.state[layer], kind) {
                        (LayerState::Gdn { recurrent }, SessionKind::Gdn) => {
                            let r = plan.grid.row;
                            submit(
                                sink,
                                hybrid::gdn_step(
                                    row_output.partition(plan.grid.output_tile()),
                                    recurrent.partition(plan.grid.recurrent_tile()),
                                    &row_input,
                                    1.0 / d as f32,
                                )
                                .generics(vec![heads.to_string(), d.to_string(), r.to_string()])
                                .compile_options(
                                    CompileOptions::default().occupancy(TARGET_BLOCKS_PER_SM),
                                ),
                            )?;
                        }
                        (LayerState::Attention { k, v }, SessionKind::Attention) => {
                            submit(
                                sink,
                                hybrid::kv_append(
                                    k.partition(plan.grid.cache_tile()),
                                    v.partition(plan.grid.cache_tile()),
                                    &row_input,
                                    &plan.positions[b],
                                )
                                .generics(vec![
                                    max_seq.to_string(),
                                    heads.to_string(),
                                    d.to_string(),
                                    plan.grid.row.to_string(),
                                ])
                                .compile_options(
                                    CompileOptions::default().occupancy(TARGET_BLOCKS_PER_SM),
                                ),
                            )?;
                            submit(
                                sink,
                                hybrid::attn_step(
                                    row_output.partition(plan.grid.output_tile()),
                                    &*k,
                                    &*v,
                                    &row_input,
                                    &plan.positions[b],
                                    1.0 / d as f32,
                                )
                                .generics(vec![
                                    heads.to_string(),
                                    d.to_string(),
                                    plan.grid.row.to_string(),
                                ])
                                .compile_options(
                                    CompileOptions::default().occupancy(TARGET_BLOCKS_PER_SM),
                                ),
                            )?;
                        }
                        _ => return Err("session state does not match model layer".into()),
                    }
                }
            }
        }
    }
    Ok(())
}

pub enum Execution {
    Eager,
    Graph,
}
pub struct Engine {
    pub model: Arc<Model>,
    stream: Stream,
    row: usize,
    plans: HashMap<(u8, Vec<SessionId>), StepPlan>,
}

impl Engine {
    pub fn new(model: Arc<Model>, device: &Arc<Device>) -> Result<Self> {
        let stream = device.new_stream()?;
        let sm =
            cuda_core::simt::CudaContext::new(device.ordinal())?.multiprocessor_count()? as usize;
        let row = row_tile(
            model.config.head_dim,
            model.config.heads,
            sm,
            TARGET_BLOCKS_PER_SM as usize,
            MIN_ROW_TILE,
        );
        Ok(Self {
            model,
            stream,
            row,
            plans: HashMap::new(),
        })
    }
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
    pub fn row_tile(&self) -> usize {
        self.row
    }
    pub fn step(&mut self, batch: Batch<'_>) -> Result<Vec<f32>> {
        self.step_with(batch, Execution::Graph)
    }
    pub fn step_with(&mut self, mut batch: Batch<'_>, execution: Execution) -> Result<Vec<f32>> {
        if batch.sessions.iter().any(|s| s.model != self.model.id) {
            return Err("session belongs to another model".into());
        }
        if batch
            .sessions
            .iter()
            .any(|s| s.position >= self.model.config.max_seq)
        {
            return Err("session reached max_seq".into());
        }
        let key = (0u8, batch.sessions.iter().map(|s| s.id).collect::<Vec<_>>());
        let instances: Vec<u64> = batch
            .sessions
            .iter()
            .map(|session| session.instance)
            .collect();
        if self
            .plans
            .get(&key)
            .is_none_or(|plan| plan.instances != instances)
        {
            let mut plan =
                StepPlan::new(&self.model, batch.sessions.len(), self.row, &self.stream)?;
            plan.instances = instances;
            self.plans.insert(key.clone(), plan);
        }
        let plan = self.plans.get_mut(&key).unwrap();
        let tokens: Vec<i32> = batch.tokens.iter().map(|&x| x as i32).collect();
        if batch
            .tokens
            .iter()
            .any(|&x| x as usize >= self.model.config.vocab)
        {
            return Err("token outside vocabulary".into());
        }
        let source = api::copy_host_vec_to_device(&Arc::new(tokens)).sync_on(&self.stream)?;
        api::memcpy(&mut plan.tokens, &source).sync_on(&self.stream)?;
        for (slot, session) in plan.positions.iter_mut().zip(&batch.sessions) {
            let source = api::copy_host_vec_to_device(&Arc::new(vec![session.position as i32]))
                .sync_on(&self.stream)?;
            api::memcpy(slot, &source).sync_on(&self.stream)?;
        }
        match execution {
            Execution::Eager => run(
                &self.model,
                plan,
                &mut batch.sessions,
                &Sink::Eager(&self.stream),
            )?,
            Execution::Graph => {
                if plan.graph.is_none() {
                    let graph = CudaGraph::scope(&self.stream, |scope| {
                        run(&self.model, plan, &mut batch.sessions, &Sink::Graph(scope))
                            .map_err(|e| cuda_async::error::DeviceError::Internal(e.to_string()))
                    })?;
                    plan.graph = Some(graph);
                }
                plan.graph
                    .as_ref()
                    .unwrap()
                    .launch()
                    .sync_on(&self.stream)?;
            }
        }
        let canonical_logits = plan.logits(&self.stream)?;
        for session in batch.sessions {
            session.position += 1;
        }
        let mut logits = vec![0.0f32; canonical_logits.len()];
        for (canonical, original) in batch.original_rows.into_iter().enumerate() {
            logits[original * self.model.config.vocab..(original + 1) * self.model.config.vocab]
                .copy_from_slice(
                    &canonical_logits[canonical * self.model.config.vocab
                        ..(canonical + 1) * self.model.config.vocab],
                );
        }
        Ok(logits)
    }
}
