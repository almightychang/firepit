use crate::checkpoint::Stream;
use crate::qwen_model::{MixerWeights, QwenModel};
use crate::qwen_plan::{NormName, Op, QwenPlan, Slot, WeightName};
use crate::{LayerKind, Result};
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{IntoPartition, PartitionMut, Tensor};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::gemv::{DEFAULT_GEMV_TILE, GemvTile, expert_gemv_view, gemv};
use firepit_kernels::sink::{Eager, Graph, Sink};
use firepit_kernels::tile::{MIN_ROW_TILE, row_tile};
use firepit_kernels::weights::{Q8_BLOCK, Weight};
use firepit_kernels::{attention, elementwise, embedding, gdn, moe, norm, rope, swiglu};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_QWEN_ENGINE: AtomicU64 = AtomicU64::new(1);
static NEXT_QWEN_SESSION: AtomicU64 = AtomicU64::new(1);
const MAX_BATCH_SESSIONS: usize = 4;
const TARGET_BLOCKS_PER_SM: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u64);

pub enum QwenLayerState {
    Gdn {
        conv: Tensor<f32>,
        recurrent: Tensor<f32>,
    },
    Attention {
        k: Tensor<f32>,
        v: Tensor<f32>,
    },
}

pub struct QwenSession {
    id: SessionId,
    engine: u64,
    instance: u64,
    position: usize,
    state: Vec<QwenLayerState>,
    poisoned: bool,
}

impl QwenSession {
    pub fn id(&self) -> SessionId {
        self.id
    }
    pub fn position(&self) -> usize {
        self.position
    }
}

pub struct QwenBatch<'a> {
    engine: u64,
    sessions: Vec<&'a mut QwenSession>,
    tokens: Vec<u32>,
    original_rows: Vec<usize>,
}

struct CachedPlan {
    instances: Vec<u64>,
    plan: QwenPlan,
}

pub struct QwenEngine {
    model: Arc<QwenModel>,
    stream: Stream,
    id: u64,
    gdn_value_tile: usize,
    plans: HashMap<usize, CachedPlan>,
    #[cfg(test)]
    fail_after_gpu_work: bool,
}

fn input_output(
    buffers: &mut [Tensor<f32>],
    input: Slot,
    output: Slot,
) -> (&Tensor<f32>, &mut Tensor<f32>) {
    let i = input.index();
    let o = output.index();
    assert_ne!(i, o);
    let (before, at) = buffers.split_at_mut(o);
    let (out, after) = at.split_first_mut().unwrap();
    (if i < o { &before[i] } else { &after[i - o - 1] }, out)
}

fn two_inputs_output(
    buffers: &mut [Tensor<f32>],
    a: Slot,
    b: Slot,
    output: Slot,
) -> (&Tensor<f32>, &Tensor<f32>, &mut Tensor<f32>) {
    let o = output.index();
    let (before, at) = buffers.split_at_mut(o);
    let (out, after) = at.split_first_mut().unwrap();
    let get = |slot: Slot| {
        let i = slot.index();
        assert_ne!(i, o);
        if i < o { &before[i] } else { &after[i - o - 1] }
    };
    (get(a), get(b), out)
}

fn three_inputs_output(
    buffers: &mut [Tensor<f32>],
    a: Slot,
    b: Slot,
    c: Slot,
    output: Slot,
) -> (&Tensor<f32>, &Tensor<f32>, &Tensor<f32>, &mut Tensor<f32>) {
    let o = output.index();
    let (before, at) = buffers.split_at_mut(o);
    let (out, after) = at.split_first_mut().unwrap();
    let get = |slot: Slot| {
        let i = slot.index();
        assert_ne!(i, o);
        if i < o { &before[i] } else { &after[i - o - 1] }
    };
    (get(a), get(b), get(c), out)
}

fn gemv_tile(rows: usize, cols: usize) -> GemvTile {
    let mut tile = DEFAULT_GEMV_TILE;
    while rows % tile.rows != 0 {
        tile.rows /= 2;
    }
    while (cols / Q8_BLOCK) % tile.blocks != 0 {
        tile.blocks /= 2;
    }
    tile
}

fn weight(model: &QwenModel, layer: usize, name: WeightName) -> &Weight {
    use WeightName::*;
    if matches!(name, LmHead) {
        return &model.lm_head;
    }
    let layer = &model.layers[layer];
    match name {
        GdnQkv => match &layer.mixer {
            MixerWeights::Gdn(w) => &w.qkv,
            _ => unreachable!(),
        },
        GdnZ => match &layer.mixer {
            MixerWeights::Gdn(w) => &w.z,
            _ => unreachable!(),
        },
        GdnB => match &layer.mixer {
            MixerWeights::Gdn(w) => &w.b,
            _ => unreachable!(),
        },
        GdnA => match &layer.mixer {
            MixerWeights::Gdn(w) => &w.a,
            _ => unreachable!(),
        },
        GdnOut => match &layer.mixer {
            MixerWeights::Gdn(w) => &w.out,
            _ => unreachable!(),
        },
        AttnQGate => match &layer.mixer {
            MixerWeights::Attention(w) => &w.q_gate,
            _ => unreachable!(),
        },
        AttnK => match &layer.mixer {
            MixerWeights::Attention(w) => &w.k,
            _ => unreachable!(),
        },
        AttnV => match &layer.mixer {
            MixerWeights::Attention(w) => &w.v,
            _ => unreachable!(),
        },
        AttnOut => match &layer.mixer {
            MixerWeights::Attention(w) => &w.out,
            _ => unreachable!(),
        },
        Router => &layer.moe.router,
        SharedGateUp => &layer.moe.shared_gate_up,
        SharedDown => &layer.moe.shared_down,
        SharedGate => &layer.moe.shared_gate,
        LmHead => unreachable!(),
    }
}

fn run(
    model: &QwenModel,
    plan: &mut QwenPlan,
    sessions: &mut [&mut QwenSession],
    sink: &mut impl Sink,
) -> Result<()> {
    let d = &model.dims;
    let g = d.gdn();
    let a = d.attention();
    let m = d.moe();
    let batch = plan.batch;
    for (op_index, &op) in plan.ops.iter().enumerate() {
        let result = (|| -> Result<()> {
            match op {
                Op::Embed { out } => sink.run(
                    embedding::embedding(
                        (&mut plan.buffers[out.index()]).partition([1, d.hidden()]),
                        &model.embedding,
                        &plan.tokens,
                    )
                    .generics(vec![d.hidden().to_string()]),
                )?,
                Op::Norm {
                    layer,
                    name,
                    input,
                    out,
                } => {
                    let (source, target) = input_output(&mut plan.buffers, input, out);
                    let (width, rows, gamma, eps) = match name {
                        NormName::Input => (
                            d.hidden(),
                            batch,
                            &model.layers[layer].input_norm,
                            d.norm_eps(),
                        ),
                        NormName::Post => (
                            d.hidden(),
                            batch,
                            &model.layers[layer].post_norm,
                            d.norm_eps(),
                        ),
                        NormName::Final => (d.hidden(), batch, &model.final_norm, d.norm_eps()),
                        NormName::Q => {
                            let w = match &model.layers[layer].mixer {
                                MixerWeights::Attention(w) => w,
                                _ => unreachable!(),
                            };
                            (a.head_dim(), batch * a.q_heads(), &w.q_norm, d.norm_eps())
                        }
                        NormName::K => {
                            let w = match &model.layers[layer].mixer {
                                MixerWeights::Attention(w) => w,
                                _ => unreachable!(),
                            };
                            (a.head_dim(), batch * a.kv_heads(), &w.k_norm, d.norm_eps())
                        }
                        NormName::GdnGated => {
                            let w = match &model.layers[layer].mixer {
                                MixerWeights::Gdn(w) => w,
                                _ => unreachable!(),
                            };
                            (
                                g.value_head_dim(),
                                batch * g.value_heads(),
                                &w.gated_norm,
                                g.eps(),
                            )
                        }
                    };
                    if matches!(name, NormName::Q) {
                        let q_gate = source.view(&[rows, 2 * width])?;
                        let q = q_gate.slice(&[0..rows, 0..width])?;
                        sink.run(
                            norm::rms_norm(
                                target.partition([1, width]),
                                &q,
                                gamma,
                                1.0 / width as f32,
                                eps,
                            )
                            .generics(vec![width.to_string()]),
                        )?;
                    } else {
                        let input = source.view(&[rows, width])?;
                        sink.run(
                            norm::rms_norm(
                                target.partition([1, width]),
                                &input,
                                gamma,
                                1.0 / width as f32,
                                eps,
                            )
                            .generics(vec![width.to_string()]),
                        )?;
                    }
                }
                Op::Gemv {
                    layer,
                    name,
                    input,
                    out,
                } => {
                    let (source, target) = input_output(&mut plan.buffers, input, out);
                    let cols = source
                        .shape()
                        .iter()
                        .map(|&x| x as usize)
                        .product::<usize>()
                        / batch;
                    let rows = target.shape()[1] as usize;
                    let x = source.view(&[batch, cols / Q8_BLOCK, Q8_BLOCK])?;
                    let tile = gemv_tile(rows, cols);
                    gemv(
                        weight(model, layer, name),
                        &x,
                        target.partition([batch, tile.rows]),
                        tile,
                    )
                    .submit(sink)?;
                }
                Op::GdnConv { layer } => {
                    let w = match &model.layers[layer].mixer {
                        MixerWeights::Gdn(w) => w,
                        _ => unreachable!(),
                    };
                    let (source, target) = input_output(&mut plan.buffers, Slot::Qkv, Slot::Conv);
                    for (b, session) in sessions.iter_mut().enumerate() {
                        let source_row = source.slice(&[b..b + 1, 0..g.conv_dim()])?;
                        let x = source_row.view(&[g.conv_dim()])?;
                        let y = target.slice_mut(&[b * g.conv_dim()..(b + 1) * g.conv_dim()])?;
                        let QwenLayerState::Gdn { conv, .. } = &mut session.state[layer] else {
                            unreachable!()
                        };
                        sink.run(
                            gdn::conv1d_silu(
                                y.partition([1]),
                                &x,
                                conv.partition([1, (g.conv_kernel() - 1).next_power_of_two()]),
                                &w.conv,
                                g.conv_kernel() as i32,
                            )
                            .generics(g.conv_generics()),
                        )?;
                    }
                }
                Op::GdnDelta { layer } => {
                    let w = match &model.layers[layer].mixer {
                        MixerWeights::Gdn(w) => w,
                        _ => unreachable!(),
                    };
                    let (conv, b_values, a_values, output) = three_inputs_output(
                        &mut plan.buffers,
                        Slot::Conv,
                        Slot::B,
                        Slot::A,
                        Slot::Mixer,
                    );
                    for (b, session) in sessions.iter_mut().enumerate() {
                        let x = conv.slice(&[b * g.conv_dim()..(b + 1) * g.conv_dim()])?;
                        let q_slice = x.slice(&[0..g.key_dim()])?;
                        let k_slice = x.slice(&[g.key_dim()..2 * g.key_dim()])?;
                        let v_slice = x.slice(&[2 * g.key_dim()..g.conv_dim()])?;
                        let query = q_slice.view(&[g.key_heads(), g.key_head_dim()])?;
                        let key = k_slice.view(&[g.key_heads(), g.key_head_dim()])?;
                        let value = v_slice.view(&[g.value_heads(), g.value_head_dim()])?;
                        let b_slice = b_values.slice(&[b..b + 1, 0..g.value_heads()])?;
                        let a_slice = a_values.slice(&[b..b + 1, 0..g.value_heads()])?;
                        let b_gate = b_slice.view(&[g.value_heads()])?;
                        let a_gate = a_slice.view(&[g.value_heads()])?;
                        let y = output.slice_mut(&[
                            b * g.value_heads()..(b + 1) * g.value_heads(),
                            0..1,
                            0..g.value_head_dim(),
                        ])?;
                        let QwenLayerState::Gdn { recurrent, .. } = &mut session.state[layer]
                        else {
                            unreachable!()
                        };
                        let tile = plan.gdn_value_tile;
                        sink.run(
                            gdn::delta_step(
                                y.partition([1, 1, tile]),
                                recurrent.partition([1, g.key_head_dim(), tile]),
                                &query,
                                &key,
                                &value,
                                &b_gate,
                                &a_gate,
                                &w.a_log,
                                &w.dt_bias,
                                g.eps(),
                                g.inv_sqrt_key_dim(),
                            )
                            .generics(g.delta_generics(tile)),
                        )?;
                    }
                }
                Op::GdnGate => {
                    let (source, gate, target) =
                        two_inputs_output(&mut plan.buffers, Slot::MixerNorm, Slot::Z, Slot::Gated);
                    let x = source.view(&[batch * g.value_heads(), g.value_head_dim()])?;
                    let z = gate.view(&[batch * g.value_heads(), g.value_head_dim()])?;
                    sink.run(
                        gdn::gated_silu(target.partition([1, g.value_head_dim()]), &x, &z)
                            .generics(vec![g.value_head_dim().to_string()]),
                    )?;
                }
                Op::AttnRope { query } => {
                    let (input, out, rdims) = if query {
                        (Slot::QNorm, Slot::QRot, d.q_rope())
                    } else {
                        (Slot::KNorm, Slot::KRot, d.k_rope())
                    };
                    let (source, target) = input_output(&mut plan.buffers, input, out);
                    let x = source.view(&[batch * rdims.heads(), rdims.head_dim()])?;
                    sink.run(
                        rope::rope(
                            target.partition([1, rdims.rotary_dim()]),
                            &x,
                            &plan.cos,
                            &plan.sin,
                        )
                        .generics(rope::generics(&rdims)),
                    )?;
                }
                Op::CacheAppend { layer } => {
                    let keys = &plan.buffers[Slot::KRot.index()];
                    let values = &plan.buffers[Slot::V.index()];
                    for (b, session) in sessions.iter_mut().enumerate() {
                        let key_row = keys
                            .slice(&[b * a.kv_heads()..(b + 1) * a.kv_heads(), 0..a.head_dim()])?;
                        let value_row =
                            values.slice(&[b..b + 1, 0..a.kv_heads() * a.head_dim()])?;
                        let key = key_row.view(&[a.kv_heads(), a.head_dim()])?;
                        let value = value_row.view(&[a.kv_heads(), a.head_dim()])?;
                        let QwenLayerState::Attention { k, v } = &mut session.state[layer] else {
                            unreachable!()
                        };
                        sink.run(
                            attention::cache_append(
                                k.partition([a.max_seq(), 1, a.head_dim()]),
                                v.partition([a.max_seq(), 1, a.head_dim()]),
                                &key,
                                &value,
                                &plan.positions[b],
                            )
                            .generics(a.cache_generics()),
                        )?;
                    }
                }
                Op::OnlineAttention { layer } => {
                    let (queries, out) = input_output(&mut plan.buffers, Slot::QRot, Slot::AttnOut);
                    for (b, session) in sessions.iter_mut().enumerate() {
                        let q_row = queries
                            .slice(&[b * a.q_heads()..(b + 1) * a.q_heads(), 0..a.head_dim()])?;
                        let q = q_row.view(&[a.q_heads(), a.head_dim()])?;
                        let y = out.slice_mut(&[
                            b * a.q_heads()..(b + 1) * a.q_heads(),
                            0..a.head_dim(),
                        ])?;
                        let QwenLayerState::Attention { k, v } = &session.state[layer] else {
                            unreachable!()
                        };
                        sink.run(
                            attention::online_attention(
                                y.partition([1, a.head_dim()]),
                                &q,
                                k,
                                v,
                                &plan.positions[b],
                                a.scale(),
                            )
                            .generics(a.attention_generics()),
                        )?;
                    }
                }
                Op::OutputGate => {
                    let (source, q_gate, target) = two_inputs_output(
                        &mut plan.buffers,
                        Slot::AttnOut,
                        Slot::QGate,
                        Slot::AttnGated,
                    );
                    let width = a.head_dim();
                    let q_gate_rows = q_gate.view(&[batch * a.q_heads(), 2 * width])?;
                    let gate = q_gate_rows.slice(&[0..batch * a.q_heads(), width..2 * width])?;
                    let x = source.view(&[batch * a.q_heads(), width])?;
                    sink.run(
                        attention::output_gate(target.partition([1, width]), &x, &gate)
                            .generics(a.gate_generics()),
                    )?;
                }
                Op::TopK { layer } => {
                    let logits = &plan.buffers[Slot::Router.index()];
                    for b in 0..sessions.len() {
                        let router_row = logits.slice(&[b..b + 1, 0..m.experts()])?;
                        let row = router_row.view(&[m.experts()])?;
                        plan.selections[layer][b].select(&row, sink)?;
                    }
                }
                Op::ExpertGateUp { layer } => {
                    let (input, output) =
                        input_output(&mut plan.buffers, Slot::Normed, Slot::ExpertUp);
                    for b in 0..sessions.len() {
                        let input_row = input.slice(&[b..b + 1, 0..d.hidden()])?;
                        let x = input_row.view(&[1, d.hidden() / Q8_BLOCK, Q8_BLOCK])?;
                        let y = output.slice_mut(&[
                            b * m.top_k()..(b + 1) * m.top_k(),
                            0..2 * m.intermediate(),
                        ])?;
                        let tile = gemv_tile(2 * m.intermediate(), d.hidden());
                        sink.run(expert_gemv_view(
                            &model.layers[layer].moe.gate_up,
                            &x,
                            &plan.selections[layer][b],
                            y.partition([1, tile.rows]),
                            2 * m.intermediate(),
                            0,
                            tile,
                        ))?;
                    }
                }
                Op::ExpertSwiglu => {
                    let (source, target) =
                        input_output(&mut plan.buffers, Slot::ExpertUp, Slot::ExpertAct);
                    for b in 0..sessions.len() {
                        let x = source.slice(&[
                            b * m.top_k()..(b + 1) * m.top_k(),
                            0..2 * m.intermediate(),
                        ])?;
                        let y = target.slice_mut(&[
                            b * m.top_k()..(b + 1) * m.top_k(),
                            0..m.intermediate(),
                        ])?;
                        sink.run(
                            swiglu::swiglu(y.partition([1, m.intermediate()]), &x)
                                .generics(vec![m.intermediate().to_string()]),
                        )?;
                    }
                }
                Op::ExpertDown { layer } => {
                    let (source, target) =
                        input_output(&mut plan.buffers, Slot::ExpertAct, Slot::ExpertOut);
                    for b in 0..sessions.len() {
                        let act_rows = source
                            .slice(&[b * m.top_k()..(b + 1) * m.top_k(), 0..m.intermediate()])?;
                        let x =
                            act_rows.view(&[m.top_k(), m.intermediate() / Q8_BLOCK, Q8_BLOCK])?;
                        let y = target
                            .slice_mut(&[b * m.top_k()..(b + 1) * m.top_k(), 0..d.hidden()])?;
                        let tile = gemv_tile(d.hidden(), m.intermediate());
                        sink.run(expert_gemv_view(
                            &model.layers[layer].moe.down,
                            &x,
                            &plan.selections[layer][b],
                            y.partition([1, tile.rows]),
                            d.hidden(),
                            1,
                            tile,
                        ))?;
                    }
                }
                Op::SharedSwiglu => {
                    let (source, target) =
                        input_output(&mut plan.buffers, Slot::SharedUp, Slot::SharedAct);
                    sink.run(
                        swiglu::swiglu(target.partition([1, m.shared_intermediate()]), source)
                            .generics(vec![m.shared_intermediate().to_string()]),
                    )?;
                }
                Op::Combine { layer } => {
                    let (routed, shared, gate, out) = three_inputs_output(
                        &mut plan.buffers,
                        Slot::ExpertOut,
                        Slot::Shared,
                        Slot::SharedGate,
                        Slot::MoeDelta,
                    );
                    for b in 0..sessions.len() {
                        let r =
                            routed.slice(&[b * m.top_k()..(b + 1) * m.top_k(), 0..d.hidden()])?;
                        let s = shared.slice(&[b..b + 1, 0..d.hidden()])?;
                        let g = gate.slice(&[b..b + 1, 0..1])?;
                        let y = out.slice_mut(&[b..b + 1, 0..d.hidden()])?;
                        sink.run(moe::combine_view(
                            y.partition([1, d.hidden()]),
                            &r,
                            &plan.selections[layer][b],
                            &s,
                            &g,
                        ))?;
                    }
                }
                Op::Add { input, delta, out } => {
                    let (a, b, target) = two_inputs_output(&mut plan.buffers, input, delta, out);
                    sink.run(
                        elementwise::add(target.partition([1, d.hidden()]), a, b)
                            .generics(vec![d.hidden().to_string()]),
                    )?;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            return Err(format!("op {op_index} {op:?}: {error}").into());
        }
    }
    Ok(())
}

impl QwenEngine {
    pub fn new(model: Arc<QwenModel>, device: &Arc<Device>) -> Result<Self> {
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
        Ok(Self {
            model,
            stream: device.new_stream()?,
            id: NEXT_QWEN_ENGINE.fetch_add(1, Ordering::Relaxed),
            gdn_value_tile,
            plans: HashMap::new(),
            #[cfg(test)]
            fail_after_gpu_work: false,
        })
    }

    pub fn new_session(&self, id: SessionId) -> Result<QwenSession> {
        let d = &self.model.dims;
        let mut state = Vec::with_capacity(d.layers().len());
        for kind in d.layers() {
            state.push(match kind {
                LayerKind::Gdn => {
                    let g = d.gdn();
                    QwenLayerState::Gdn {
                        conv: api::zeros::<f32>(&[g.conv_dim(), g.conv_kernel() - 1])
                            .sync_on(&self.stream)?,
                        recurrent: api::zeros::<f32>(&[
                            g.value_heads(),
                            g.key_head_dim(),
                            g.value_head_dim(),
                        ])
                        .sync_on(&self.stream)?,
                    }
                }
                LayerKind::Attention => {
                    let a = d.attention();
                    QwenLayerState::Attention {
                        k: api::zeros::<f32>(&[a.max_seq(), a.kv_heads(), a.head_dim()])
                            .sync_on(&self.stream)?,
                        v: api::zeros::<f32>(&[a.max_seq(), a.kv_heads(), a.head_dim()])
                            .sync_on(&self.stream)?,
                    }
                }
            });
        }
        Ok(QwenSession {
            id,
            engine: self.id,
            instance: NEXT_QWEN_SESSION.fetch_add(1, Ordering::Relaxed),
            position: 0,
            state,
            poisoned: false,
        })
    }

    pub fn close_session(&mut self, session: QwenSession) -> Result<()> {
        if session.engine != self.id {
            return Err("session belongs to another engine".into());
        }
        self.plans
            .retain(|_, entry| !entry.instances.contains(&session.instance));
        Ok(())
    }

    pub fn batch<'a>(
        &self,
        entries: impl IntoIterator<Item = (&'a mut QwenSession, u32)>,
    ) -> Result<QwenBatch<'a>> {
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
                || session.engine != self.id
                || session.poisoned
                || session.position >= self.model.dims.max_seq()
                || *token as usize >= self.model.dims.vocab()
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
        Ok(QwenBatch {
            engine: self.id,
            sessions,
            tokens,
            original_rows,
        })
    }

    pub fn step_eager(&mut self, batch: QwenBatch<'_>) -> Result<Vec<f32>> {
        self.step_with(batch, |model, plan, sessions, stream| {
            run(model, plan, sessions, &mut Eager(stream))
        })
    }

    pub fn step(&mut self, batch: QwenBatch<'_>) -> Result<Vec<f32>> {
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

    fn step_with(
        &mut self,
        mut batch: QwenBatch<'_>,
        execute: impl FnOnce(&QwenModel, &mut QwenPlan, &mut [&mut QwenSession], &Stream) -> Result<()>,
    ) -> Result<Vec<f32>> {
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
                    plan: QwenPlan::new(&self.model, size, self.gdn_value_tile, &self.stream)?,
                },
            );
        }
        let plan = &mut self.plans.get_mut(&size).unwrap().plan;
        let result = (|| -> Result<Vec<f32>> {
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
            execute(&self.model, plan, &mut batch.sessions, &self.stream)?;
            #[cfg(test)]
            if self.fail_after_gpu_work {
                return Err("injected failure after GPU work".into());
            }
            plan.logits(&self.stream)
        })();
        let canonical = match result {
            Ok(logits) => logits,
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
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SafetensorsFile;
    use std::path::Path;

    fn small_model(device: &Arc<Device>) -> Result<Arc<QwenModel>> {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/small-qwen.json"))?;
        Ok(Arc::new(QwenModel::random_from_config(
            &config["text_config"],
            8,
            7,
            &device.new_stream()?,
        )?))
    }

    #[test]
    fn gpu_failure_poisons_session() -> Result<()> {
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        let mut engine = QwenEngine::new(model, &device)?;
        let mut session = engine.new_session(SessionId(1))?;
        engine.fail_after_gpu_work = true;
        let batch = engine.batch([(&mut session, 1)])?;
        assert!(engine.step_eager(batch).is_err());
        assert_eq!(session.position(), 0);
        assert!(engine.batch([(&mut session, 1)]).is_err());
        Ok(())
    }

    #[test]
    fn small_model_batch_matches_individual_and_graph() -> Result<()> {
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        for size in 1..=4 {
            let mut together = QwenEngine::new(model.clone(), &device)?;
            let mut graph = QwenEngine::new(model.clone(), &device)?;
            let mut singles = (0..size)
                .map(|_| QwenEngine::new(model.clone(), &device))
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
        let refs = SafetensorsFile::open(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4.safetensors"),
        )?;
        let tokens: Vec<u32> =
            serde_json::from_str(refs.metadata("tokens").ok_or("missing tokens")?)?;
        let mut model = Arc::new(QwenModel::from_checkpoint(
            dir,
            crate::LoadOptions {
                max_seq: tokens.len(),
                layers: Some(4),
            },
        )?);
        let device = Device::new(0)?;
        let mut graph_engine = QwenEngine::new(model.clone(), &device)?;
        let mut graph_session = graph_engine.new_session(SessionId(2))?;
        let mut engine = QwenEngine::new(model.clone(), &device)?;
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
        let mut shuffled = QwenEngine::new(model, &device)?;
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
