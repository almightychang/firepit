use crate::Result;
#[cfg(feature = "testing")]
use crate::checkpoint::Stream;
#[cfg(feature = "testing")]
use crate::engine::RoutingLayerTrace;
use crate::model::{MixerWeights, Model, Projection};
use crate::plan::{NormName, Op, Slot, StepPlan, WeightName};
use crate::session::{LayerState, Session};
#[cfg(feature = "testing")]
use cuda_async::device_operation::DeviceOp;
use cutile::tensor::{IntoPartition, PartitionMut, Tensor, TensorView};
use cutile::tile_kernel::TileKernel;
#[cfg(feature = "testing")]
use cutile::tile_kernel::ToHostVecOp;
use firepit_kernels::gemv::{
    DEFAULT_GEMV_TILE, GemvTile, expert_gemv_weight_view, fp8_tile, gemv, row_scaled_tile,
};
use firepit_kernels::sink::Sink;
use firepit_kernels::weights::{Fp8Scaling, Q8_BLOCK, Weight};
use firepit_kernels::{attention, elementwise, embedding, gdn, moe, norm, rope, swiglu};
use std::ops::Range;

pub(crate) fn input_output(
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

pub(crate) fn two_inputs_output(
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

pub(crate) fn three_inputs_output(
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

pub(crate) fn gemv_tile(rows: usize, cols: usize) -> GemvTile {
    let mut tile = DEFAULT_GEMV_TILE;
    while rows % tile.rows != 0 {
        tile.rows /= 2;
    }
    while (cols / Q8_BLOCK) % tile.blocks != 0 {
        tile.blocks /= 2;
    }
    tile
}

pub(crate) fn fp8_gemv_tile(
    scaling: Fp8Scaling,
    rows: usize,
    cols: usize,
    groups: usize,
) -> GemvTile {
    match scaling {
        Fp8Scaling::Blocks { block_n, block_k } => fp8_tile(rows, cols, block_n, block_k, groups),
        Fp8Scaling::Rows => row_scaled_tile(rows, cols, groups),
    }
}

pub(crate) fn weight_gemv_tile(
    weight: &Weight,
    rows: usize,
    cols: usize,
    groups: usize,
) -> GemvTile {
    match weight {
        Weight::Fp8(w) => fp8_gemv_tile(w.scaling, rows, cols, groups),
        Weight::Nvfp4(_) => row_scaled_tile(rows, cols, groups),
        _ => gemv_tile(rows, cols),
    }
}

pub(crate) fn weight(model: &Model, layer: usize, name: WeightName) -> &Weight {
    if matches!(name, WeightName::LmHead) {
        return &model.lm_head;
    }
    let layer = &model.layers[layer];
    match (name, &layer.mixer) {
        (WeightName::MixerProjection(group), mixer) => &mixer.projections()[group].weight,
        (WeightName::GdnOut, MixerWeights::Gdn(w)) => &w.out,
        (WeightName::AttnOut, MixerWeights::Attention(w)) => &w.out,
        (WeightName::MoeProjection(group), _) => &layer.moe.projections[group].weight,
        _ => unreachable!(),
    }
}

pub(crate) fn row_width(tensor: &Tensor<f32>, rows: usize) -> usize {
    tensor
        .shape()
        .iter()
        .map(|&x| x as usize)
        .product::<usize>()
        / rows
}

pub(crate) fn heads(columns: &Range<usize>, pitch: usize) -> Result<Range<usize>> {
    if columns.start % pitch != 0 || columns.end % pitch != 0 {
        return Err(format!("columns {columns:?} do not split into heads of {pitch}").into());
    }
    Ok(columns.start / pitch..columns.end / pitch)
}

struct ProjectionRows<'a> {
    flat: TensorView<'a, f32>,
    width: usize,
    columns: Range<usize>,
}

impl<'a> ProjectionRows<'a> {
    fn new(buffer: &'a Tensor<f32>, columns: Range<usize>) -> Result<Self> {
        let width = buffer.shape()[1] as usize;
        Ok(Self {
            flat: buffer.view(&[buffer.shape()[0] as usize * width])?,
            width,
            columns,
        })
    }

    fn row(&self, row: usize) -> Result<TensorView<'_, f32>> {
        let start = row * self.width;
        Ok(self
            .flat
            .slice(&[start + self.columns.start..start + self.columns.end])?)
    }
}

struct NormInput<'a> {
    gamma: &'a Tensor<f32>,
    eps: f32,
    width: usize,
    pitch: usize,
    heads: Range<usize>,
}

impl<'a> NormInput<'a> {
    fn rows(gamma: &'a Tensor<f32>, eps: f32, width: usize) -> Self {
        Self {
            gamma,
            eps,
            width,
            pitch: width,
            heads: 0..1,
        }
    }

    fn new(model: &'a Model, layer: usize, name: NormName) -> Result<Self> {
        let d = &model.dims;
        let head_dim = d.attention().head_dim();
        Ok(match (name, &model.layers[layer].mixer) {
            (NormName::Input, _) => {
                Self::rows(&model.layers[layer].input_norm, d.norm_eps(), d.hidden())
            }
            (NormName::Post, _) => {
                Self::rows(&model.layers[layer].post_norm, d.norm_eps(), d.hidden())
            }
            (NormName::Final, _) => Self::rows(&model.final_norm, d.norm_eps(), d.hidden()),
            (NormName::Q, MixerWeights::Attention(w)) => {
                let (_, columns) = model.projection(layer, Projection::AttnQGate)?;
                Self {
                    gamma: &w.q_norm,
                    eps: d.norm_eps(),
                    width: head_dim,
                    pitch: 2 * head_dim,
                    heads: heads(&columns, 2 * head_dim)?,
                }
            }
            (NormName::K, MixerWeights::Attention(w)) => {
                let (_, columns) = model.projection(layer, Projection::AttnK)?;
                Self {
                    gamma: &w.k_norm,
                    eps: d.norm_eps(),
                    width: head_dim,
                    pitch: head_dim,
                    heads: heads(&columns, head_dim)?,
                }
            }
            (NormName::GdnGated, MixerWeights::Gdn(w)) => {
                let g = d.gdn();
                Self {
                    gamma: &w.gated_norm,
                    eps: g.eps(),
                    width: g.value_head_dim(),
                    pitch: g.value_head_dim(),
                    heads: 0..g.value_heads(),
                }
            }
            _ => unreachable!(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_norm(
    model: &Model,
    layer: usize,
    name: NormName,
    source: &Tensor<f32>,
    target: &mut Tensor<f32>,
    rows: usize,
    selected: Range<usize>,
    sink: &mut impl Sink,
) -> Result<()> {
    let norm = NormInput::new(model, layer, name)?;
    let shaped = source.view(&[rows, row_width(source, rows) / norm.pitch, norm.pitch])?;
    let y = target.slice_mut(&[selected.clone(), 0..norm.heads.len() * norm.width])?;
    let x = shaped.slice(&[selected, norm.heads, 0..norm.width])?;
    sink.run(
        norm::rms_norm(
            y.partition([1, norm.width]),
            &x,
            norm.gamma,
            1.0 / norm.width as f32,
            norm.eps,
        )
        .generics(vec![norm.width.to_string()]),
    )?;
    Ok(())
}

pub(crate) fn gdn_gate(
    model: &Model,
    layer: usize,
    buffers: &mut [Tensor<f32>],
    rows: usize,
    sink: &mut impl Sink,
) -> Result<()> {
    let g = model.dims.gdn();
    let (z, columns) = Slot::projection(model, layer, Projection::GdnZ)?;
    let (source, gate, target) = two_inputs_output(buffers, Slot::MixerNorm, z, Slot::Gated);
    let x = source.view(&[rows, g.value_dim()])?;
    let z = gate.slice(&[0..rows, columns])?;
    sink.run(
        gdn::gated_silu(target.partition([1, g.value_head_dim()]), &x, &z)
            .generics(vec![g.value_head_dim().to_string()]),
    )?;
    Ok(())
}

pub(crate) fn output_gate(
    model: &Model,
    layer: usize,
    buffers: &mut [Tensor<f32>],
    rows: usize,
    sink: &mut impl Sink,
) -> Result<()> {
    let a = model.dims.attention();
    let (q_gate, columns) = Slot::projection(model, layer, Projection::AttnQGate)?;
    let (source, q_gate, target) =
        two_inputs_output(buffers, Slot::AttnOut, q_gate, Slot::AttnGated);
    let (width, pitch) = (a.head_dim(), 2 * a.head_dim());
    let x = source.view(&[rows, a.q_heads(), width])?;
    let shaped = q_gate.view(&[rows, row_width(q_gate, rows) / pitch, pitch])?;
    let gate = shaped.slice(&[0..rows, heads(&columns, pitch)?, width..pitch])?;
    sink.run(
        attention::output_gate(target.partition([1, width]), &x, &gate).generics(a.gate_generics()),
    )?;
    Ok(())
}

pub(crate) fn run(
    model: &Model,
    plan: &mut StepPlan,
    sessions: &mut [&mut Session],
    sink: &mut impl Sink,
) -> Result<()> {
    run_inner(model, plan, sessions, sink, |_, _| Ok(()))
}

#[cfg(feature = "testing")]
pub(crate) fn run_routing_observed(
    model: &Model,
    plan: &mut StepPlan,
    sessions: &mut [&mut Session],
    stream: &Stream,
) -> Result<Vec<RoutingLayerTrace>> {
    if sessions.len() != 1 {
        return Err("routing observation requires one session".into());
    }
    let mut layers = Vec::<RoutingLayerTrace>::new();
    let top_k = model.dims.moe().top_k();
    let hidden = model.dims.hidden();
    run_inner(
        model,
        plan,
        sessions,
        &mut firepit_kernels::sink::Eager(stream),
        |op, plan| {
            match op {
                Op::TopK { layer } => {
                    let (router, columns) = Slot::projection(model, layer, Projection::Router)?;
                    let logits: Vec<f32> = plan.buffers[router.index()]
                        .dup()
                        .to_host_vec()
                        .sync_on(stream)?;
                    let ids: Vec<i32> = plan.selections[layer][0]
                        .indices()
                        .dup()
                        .to_host_vec()
                        .sync_on(stream)?;
                    let mut trace = RoutingLayerTrace::default();
                    trace.router_logits.push(logits[columns].to_vec());
                    trace.selected.push(ids[..top_k].to_vec());
                    layers.push(trace);
                }
                Op::Add {
                    delta: Slot::MoeDelta,
                    out,
                    ..
                } => {
                    let values: Vec<f32> = plan.buffers[out.index()]
                        .dup()
                        .to_host_vec()
                        .sync_on(stream)?;
                    layers
                        .last_mut()
                        .ok_or("hidden without router")?
                        .hidden
                        .push(values[..hidden].to_vec());
                }
                _ => {}
            }
            Ok(())
        },
    )?;
    Ok(layers)
}

fn run_inner(
    model: &Model,
    plan: &mut StepPlan,
    sessions: &mut [&mut Session],
    sink: &mut impl Sink,
    mut after_op: impl FnMut(Op, &StepPlan) -> Result<()>,
) -> Result<()> {
    let d = &model.dims;
    let g = d.gdn();
    let a = d.attention();
    let m = d.moe();
    let slots = m.with_shared_expert().top_k();
    let batch = plan.batch;
    for (op_index, &op) in plan.ops.iter().enumerate() {
        let result = sink.span(op.kind(), |sink| -> Result<()> {
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
                    rms_norm(model, layer, name, source, target, batch, 0..batch, sink)?;
                }
                Op::Gemv {
                    layer,
                    name,
                    input,
                    out,
                } => {
                    let (source, target) = input_output(&mut plan.buffers, input, out);
                    let cols = row_width(source, batch);
                    let rows = target.shape()[1] as usize;
                    let x = source.view(&[batch, cols / Q8_BLOCK, Q8_BLOCK])?;
                    let w = weight(model, layer, name);
                    let tile = if batch == 1 {
                        weight_gemv_tile(w, rows, cols, 1)
                    } else {
                        gemv_tile(rows, cols)
                    };
                    gemv(w, &x, target.partition([batch, tile.rows]), tile).submit(sink)?;
                }
                Op::GdnConv { layer } => {
                    let w = match &model.layers[layer].mixer {
                        MixerWeights::Gdn(w) => w,
                        _ => unreachable!(),
                    };
                    let (qkv, columns) = Slot::projection(model, layer, Projection::GdnQkv)?;
                    let (source, target) = input_output(&mut plan.buffers, qkv, Slot::Conv);
                    let source = ProjectionRows::new(source, columns)?;
                    let tile = plan.gdn_conv_tile;
                    let kernel_tile = g.conv_kernel().next_power_of_two();
                    for (b, session) in sessions.iter_mut().enumerate() {
                        let x = source.row(b)?;
                        let y = target.slice_mut(&[b * g.conv_dim()..(b + 1) * g.conv_dim()])?;
                        let LayerState::Gdn { conv, .. } = &mut session.state[layer] else {
                            unreachable!()
                        };
                        sink.run(
                            gdn::conv1d_silu(
                                y.partition([tile]),
                                &x,
                                conv.partition([tile, kernel_tile]),
                                &w.conv,
                            )
                            .generics(g.conv_generics(tile)),
                        )?;
                    }
                }
                Op::GdnDelta { layer } => {
                    let w = match &model.layers[layer].mixer {
                        MixerWeights::Gdn(w) => w,
                        _ => unreachable!(),
                    };
                    let (b_slot, b_columns) = Slot::projection(model, layer, Projection::GdnB)?;
                    let (a_slot, a_columns) = Slot::projection(model, layer, Projection::GdnA)?;
                    let (conv, b_values, a_values, output) = three_inputs_output(
                        &mut plan.buffers,
                        Slot::Conv,
                        b_slot,
                        a_slot,
                        Slot::Mixer,
                    );
                    let b_values = ProjectionRows::new(b_values, b_columns)?;
                    let a_values = ProjectionRows::new(a_values, a_columns)?;
                    for (b, session) in sessions.iter_mut().enumerate() {
                        let x = conv.slice(&[b * g.conv_dim()..(b + 1) * g.conv_dim()])?;
                        let q_slice = x.slice(&[0..g.key_dim()])?;
                        let k_slice = x.slice(&[g.key_dim()..2 * g.key_dim()])?;
                        let v_slice = x.slice(&[2 * g.key_dim()..g.conv_dim()])?;
                        let query = q_slice.view(&[g.key_heads(), g.key_head_dim()])?;
                        let key = k_slice.view(&[g.key_heads(), g.key_head_dim()])?;
                        let value = v_slice.view(&[g.value_heads(), g.value_head_dim()])?;
                        let b_gate = b_values.row(b)?;
                        let a_gate = a_values.row(b)?;
                        let y = output.slice_mut(&[
                            b * g.value_heads()..(b + 1) * g.value_heads(),
                            0..1,
                            0..g.value_head_dim(),
                        ])?;
                        let LayerState::Gdn { recurrent, .. } = &mut session.state[layer] else {
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
                Op::GdnGate { layer } => gdn_gate(model, layer, &mut plan.buffers, batch, sink)?,
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
                    let (v, columns) = Slot::projection(model, layer, Projection::AttnV)?;
                    let values = ProjectionRows::new(&plan.buffers[v.index()], columns)?;
                    for (b, session) in sessions.iter_mut().enumerate() {
                        let key_row = keys
                            .slice(&[b * a.kv_heads()..(b + 1) * a.kv_heads(), 0..a.head_dim()])?;
                        let value_row = values.row(b)?;
                        let key = key_row.view(&[a.kv_heads(), a.head_dim()])?;
                        let value = value_row.view(&[a.kv_heads(), a.head_dim()])?;
                        let LayerState::Attention { k, v } = &mut session.state[layer] else {
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
                        let LayerState::Attention { k, v } = &session.state[layer] else {
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
                Op::OutputGate { layer } => {
                    output_gate(model, layer, &mut plan.buffers, batch, sink)?
                }
                Op::TopK { layer } => {
                    let (router, router_columns) =
                        Slot::projection(model, layer, Projection::Router)?;
                    let (gate, gate_columns) =
                        Slot::projection(model, layer, Projection::SharedGate)?;
                    let logits =
                        ProjectionRows::new(&plan.buffers[router.index()], router_columns)?;
                    let gates = ProjectionRows::new(&plan.buffers[gate.index()], gate_columns)?;
                    for b in 0..sessions.len() {
                        let gate = gates.row(b)?;
                        let gate = gate.view(&[1, 1])?;
                        plan.selections[layer][b].select(&m, &logits.row(b)?, &gate, sink)?;
                    }
                }
                Op::ExpertGateUp { layer } => {
                    let (input, output) =
                        input_output(&mut plan.buffers, Slot::Normed, Slot::ExpertUp);
                    for b in 0..sessions.len() {
                        let input_row = input.slice(&[b..b + 1, 0..d.hidden()])?;
                        let x = input_row.view(&[1, d.hidden() / Q8_BLOCK, Q8_BLOCK])?;
                        let y = output
                            .slice_mut(&[b * slots..(b + 1) * slots, 0..2 * m.intermediate()])?;
                        let w = &model.layers[layer].moe.gate_up;
                        let tile = weight_gemv_tile(w, 2 * m.intermediate(), d.hidden(), slots);
                        expert_gemv_weight_view(
                            w,
                            &x,
                            &plan.selections[layer][b],
                            y.partition([1, tile.rows]),
                            2 * m.intermediate(),
                            0,
                            tile,
                        )
                        .submit(sink)?;
                    }
                }
                Op::ExpertSwiglu => {
                    let (source, target) =
                        input_output(&mut plan.buffers, Slot::ExpertUp, Slot::ExpertAct);
                    for b in 0..sessions.len() {
                        let x =
                            source.slice(&[b * slots..(b + 1) * slots, 0..2 * m.intermediate()])?;
                        let y =
                            target.slice_mut(&[b * slots..(b + 1) * slots, 0..m.intermediate()])?;
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
                        let act_rows =
                            source.slice(&[b * slots..(b + 1) * slots, 0..m.intermediate()])?;
                        let x = act_rows.view(&[slots, m.intermediate() / Q8_BLOCK, Q8_BLOCK])?;
                        let y = target.slice_mut(&[b * slots..(b + 1) * slots, 0..d.hidden()])?;
                        let w = &model.layers[layer].moe.down;
                        let tile = weight_gemv_tile(w, d.hidden(), m.intermediate(), slots);
                        expert_gemv_weight_view(
                            w,
                            &x,
                            &plan.selections[layer][b],
                            y.partition([1, tile.rows]),
                            d.hidden(),
                            1,
                            tile,
                        )
                        .submit(sink)?;
                    }
                }
                Op::Combine { layer } => {
                    let (routed, out) =
                        input_output(&mut plan.buffers, Slot::ExpertOut, Slot::MoeDelta);
                    for b in 0..sessions.len() {
                        let r = routed.slice(&[b * slots..(b + 1) * slots, 0..d.hidden()])?;
                        let y = out.slice_mut(&[b..b + 1, 0..d.hidden()])?;
                        sink.run(moe::combine_view(
                            y.partition([1, d.hidden()]),
                            &r,
                            &plan.selections[layer][b],
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
        });
        if let Err(error) = result {
            return Err(format!("op {op_index} {op:?}: {error}").into());
        }
        after_op(op, plan)?;
    }
    Ok(())
}
