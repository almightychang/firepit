use crate::Result;
#[cfg(feature = "testing")]
use crate::checkpoint::Stream;
#[cfg(feature = "testing")]
use crate::engine::RoutingLayerTrace;
use crate::model::{MixerWeights, Model, Projection};
use crate::plan::{GdnTiles, NormName, Op, Slot, WeightName};
use crate::store::{Ctx, Parts, Read, View};
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

pub(crate) fn row_width(tensor: &Tensor<f32>, rows: usize) -> usize {
    tensor
        .shape()
        .iter()
        .map(|&x| x as usize)
        .product::<usize>()
        / rows
}

struct ProjectionRows<'a> {
    flat: TensorView<'a, f32>,
    width: usize,
    columns: Range<usize>,
}

impl<'a> ProjectionRows<'a> {
    fn new(view: &View<'a>) -> Result<Self> {
        let width = view.tensor.shape()[1] as usize;
        Ok(Self {
            flat: view
                .tensor
                .view(&[view.tensor.shape()[0] as usize * width])?,
            width,
            columns: view.columns.clone(),
        })
    }

    fn row(&self, row: usize) -> Result<TensorView<'_, f32>> {
        let start = row * self.width;
        Ok(self
            .flat
            .slice(&[start + self.columns.start..start + self.columns.end])?)
    }
}

pub(crate) fn heads(columns: &Range<usize>, pitch: usize) -> Range<usize> {
    columns.start / pitch..columns.end / pitch
}

pub(crate) fn norm_read(layer: usize, name: NormName, input: Slot) -> Read {
    match name {
        NormName::Q => Read::Member {
            layer,
            member: Projection::AttnQGate,
        },
        NormName::K => Read::Member {
            layer,
            member: Projection::AttnK,
        },
        _ => Read::Whole(input),
    }
}

struct NormInput<'a> {
    gamma: &'a Tensor<f32>,
    eps: f32,
    width: usize,
    pitch: usize,
}

impl<'a> NormInput<'a> {
    fn rows(gamma: &'a Tensor<f32>, eps: f32, width: usize) -> Self {
        Self {
            gamma,
            eps,
            width,
            pitch: width,
        }
    }

    fn new(model: &'a Model, layer: usize, name: NormName) -> Self {
        let d = &model.dims;
        let head_dim = d.attention().head_dim();
        match (name, &model.layers[layer].mixer) {
            (NormName::Input, _) => {
                Self::rows(&model.layers[layer].input_norm, d.norm_eps(), d.hidden())
            }
            (NormName::Post, _) => {
                Self::rows(&model.layers[layer].post_norm, d.norm_eps(), d.hidden())
            }
            (NormName::Final, _) => Self::rows(&model.final_norm, d.norm_eps(), d.hidden()),
            (NormName::Q, MixerWeights::Attention(w)) => Self {
                gamma: &w.q_norm,
                eps: d.norm_eps(),
                width: head_dim,
                pitch: Projection::AttnQGate.head_pitch(head_dim),
            },
            (NormName::K, MixerWeights::Attention(w)) => Self {
                gamma: &w.k_norm,
                eps: d.norm_eps(),
                width: head_dim,
                pitch: Projection::AttnK.head_pitch(head_dim),
            },
            (NormName::GdnGated, MixerWeights::Gdn(w)) => {
                let g = d.gdn();
                Self {
                    gamma: &w.gated_norm,
                    eps: g.eps(),
                    width: g.value_head_dim(),
                    pitch: g.value_head_dim(),
                }
            }
            _ => unreachable!(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_norm(
    model: &Model,
    layer: usize,
    name: NormName,
    source: &View<'_>,
    target: &mut Tensor<f32>,
    rows: usize,
    selected: Range<usize>,
    sink: &mut impl Sink,
) -> Result<()> {
    let norm = NormInput::new(model, layer, name);
    let heads = heads(&source.columns, norm.pitch);
    let tensor = source.tensor;
    let shaped = tensor.view(&[rows, row_width(tensor, rows) / norm.pitch, norm.pitch])?;
    let y = target.slice_mut(&[selected.clone(), 0..heads.len() * norm.width])?;
    let x = shaped.slice(&[selected, heads, 0..norm.width])?;
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
    source: &Tensor<f32>,
    gate: &Tensor<f32>,
    columns: Range<usize>,
    target: &mut Tensor<f32>,
    rows: usize,
    sink: &mut impl Sink,
) -> Result<()> {
    let g = model.dims.gdn();
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
    source: &Tensor<f32>,
    q_gate: &Tensor<f32>,
    columns: Range<usize>,
    target: &mut Tensor<f32>,
    rows: usize,
    sink: &mut impl Sink,
) -> Result<()> {
    let a = model.dims.attention();
    let width = a.head_dim();
    let pitch = Projection::AttnQGate.head_pitch(width);
    let x = source.view(&[rows, a.q_heads(), width])?;
    let shaped = q_gate.view(&[rows, row_width(q_gate, rows) / pitch, pitch])?;
    let gate = shaped.slice(&[0..rows, heads(&columns, pitch), width..pitch])?;
    sink.run(
        attention::output_gate(target.partition([1, width]), &x, &gate).generics(a.gate_generics()),
    )?;
    Ok(())
}

pub(crate) fn run(
    ops: &[Op],
    tiles: GdnTiles,
    ctx: &mut Ctx<'_, '_, '_>,
    sink: &mut impl Sink,
) -> Result<()> {
    run_inner(ops, tiles, ctx, sink, |_, _| Ok(()))
}

#[cfg(feature = "testing")]
pub(crate) fn run_routing_observed(
    ops: &[Op],
    tiles: GdnTiles,
    ctx: &mut Ctx<'_, '_, '_>,
    stream: &Stream,
) -> Result<Vec<RoutingLayerTrace>> {
    if ctx.sessions() != 1 {
        return Err("routing observation requires one session".into());
    }
    let mut layers = Vec::<RoutingLayerTrace>::new();
    let top_k = ctx.model().dims.moe().top_k();
    let hidden = ctx.model().dims.hidden();
    run_inner(
        ops,
        tiles,
        ctx,
        &mut firepit_kernels::sink::Eager(stream),
        |op, ctx| {
            match op {
                Op::TopK { layer } => {
                    let router = ctx.peek(Read::Member {
                        layer,
                        member: Projection::Router,
                    })?;
                    let logits: Vec<f32> = router.tensor.dup().to_host_vec().sync_on(stream)?;
                    let ids: Vec<i32> = ctx
                        .peek_selection(layer, 0)
                        .indices()
                        .dup()
                        .to_host_vec()
                        .sync_on(stream)?;
                    let mut trace = RoutingLayerTrace::default();
                    trace.router_logits.push(logits[router.columns].to_vec());
                    trace.selected.push(ids[..top_k].to_vec());
                    layers.push(trace);
                }
                Op::Add {
                    delta: Slot::MoeDelta,
                    out,
                    ..
                } => {
                    let values: Vec<f32> = ctx
                        .peek(Read::Whole(out))?
                        .tensor
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
    ops: &[Op],
    tiles: GdnTiles,
    ctx: &mut Ctx<'_, '_, '_>,
    sink: &mut impl Sink,
    mut after_op: impl FnMut(Op, &Ctx<'_, '_, '_>) -> Result<()>,
) -> Result<()> {
    let model = ctx.model();
    let d = &model.dims;
    let g = d.gdn();
    let a = d.attention();
    let m = d.moe();
    let slots = m.with_shared_expert().top_k();
    let batch = ctx.batch();
    let sessions = ctx.sessions();
    for (op_index, &op) in ops.iter().enumerate() {
        let result = sink.span(op.kind(), |sink| -> Result<()> {
            ctx.begin(op_index, op.kind(), op.layer());
            let Parts {
                mut buffers,
                inputs,
                mut selections,
                mut states,
                weights,
            } = ctx.parts();
            match op {
                Op::Embed { out } => {
                    let ([], target) = buffers.io([], out)?;
                    sink.run(
                        embedding::embedding(
                            target.partition([1, d.hidden()]),
                            &model.embedding,
                            inputs.tokens(),
                        )
                        .generics(vec![d.hidden().to_string()]),
                    )?
                }
                Op::Norm {
                    layer,
                    name,
                    input,
                    out,
                } => {
                    let ([source], target) = buffers.io([norm_read(layer, name, input)], out)?;
                    rms_norm(model, layer, name, &source, target, batch, 0..batch, sink)?;
                }
                Op::Gemv {
                    layer,
                    name,
                    input,
                    out,
                } => {
                    let ([source], target) = buffers.io([Read::Whole(input)], out)?;
                    let cols = row_width(source.tensor, batch);
                    let rows = target.shape()[1] as usize;
                    let x = source.tensor.view(&[batch, cols / Q8_BLOCK, Q8_BLOCK])?;
                    let w = weights.matrix(layer, name);
                    let tile = if batch == 1 {
                        weight_gemv_tile(w, rows, cols, 1)
                    } else {
                        gemv_tile(rows, cols)
                    };
                    gemv(w, &x, target.partition([batch, tile.rows]), tile).submit(sink)?;
                }
                Op::GdnConv { layer } => {
                    let MixerWeights::Gdn(w) = &model.layers[layer].mixer else {
                        unreachable!()
                    };
                    let qkv = Read::Member {
                        layer,
                        member: Projection::GdnQkv,
                    };
                    let ([qkv], target) = buffers.io([qkv], Slot::Conv)?;
                    let source = ProjectionRows::new(&qkv)?;
                    let kernel_tile = g.conv_kernel().next_power_of_two();
                    for b in 0..sessions {
                        let x = source.row(b)?;
                        let y = target.slice_mut(&[b * g.conv_dim()..(b + 1) * g.conv_dim()])?;
                        let (conv, _) = states.gdn(b, layer)?;
                        sink.run(
                            gdn::conv1d_silu(
                                y.partition([tiles.conv]),
                                &x,
                                conv.partition([tiles.conv, kernel_tile]),
                                &w.conv,
                            )
                            .generics(g.conv_generics(tiles.conv)),
                        )?;
                    }
                }
                Op::GdnDelta { layer } => {
                    let MixerWeights::Gdn(w) = &model.layers[layer].mixer else {
                        unreachable!()
                    };
                    let member = |member| Read::Member { layer, member };
                    let ([conv, b_values, a_values], output) = buffers.io(
                        [
                            Read::Whole(Slot::Conv),
                            member(Projection::GdnB),
                            member(Projection::GdnA),
                        ],
                        Slot::Mixer,
                    )?;
                    let b_values = ProjectionRows::new(&b_values)?;
                    let a_values = ProjectionRows::new(&a_values)?;
                    for b in 0..sessions {
                        let x = conv
                            .tensor
                            .slice(&[b * g.conv_dim()..(b + 1) * g.conv_dim()])?;
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
                        let (_, recurrent) = states.gdn(b, layer)?;
                        sink.run(
                            gdn::delta_step(
                                y.partition([1, 1, tiles.value]),
                                recurrent.partition([1, g.key_head_dim(), tiles.value]),
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
                            .generics(g.delta_generics(tiles.value)),
                        )?;
                    }
                }
                Op::GdnGate { layer } => {
                    let z = Read::Member {
                        layer,
                        member: Projection::GdnZ,
                    };
                    let ([source, z], target) =
                        buffers.io([Read::Whole(Slot::MixerNorm), z], Slot::Gated)?;
                    gdn_gate(
                        model,
                        source.tensor,
                        z.tensor,
                        z.columns,
                        target,
                        batch,
                        sink,
                    )?
                }
                Op::AttnRope { query } => {
                    let (input, out, rdims) = if query {
                        (Slot::QNorm, Slot::QRot, d.q_rope())
                    } else {
                        (Slot::KNorm, Slot::KRot, d.k_rope())
                    };
                    let ([source], target) = buffers.io([Read::Whole(input)], out)?;
                    let x = source
                        .tensor
                        .view(&[batch * rdims.heads(), rdims.head_dim()])?;
                    sink.run(
                        rope::rope(
                            target.partition([1, rdims.rotary_dim()]),
                            &x,
                            inputs.cos(),
                            inputs.sin(),
                        )
                        .generics(rope::generics(&rdims)),
                    )?;
                }
                Op::CacheAppend { layer } => {
                    let v = Read::Member {
                        layer,
                        member: Projection::AttnV,
                    };
                    let [keys, values] = buffers.read([Read::Whole(Slot::KRot), v])?;
                    let values = ProjectionRows::new(&values)?;
                    for b in 0..sessions {
                        let key_row = keys
                            .tensor
                            .slice(&[b * a.kv_heads()..(b + 1) * a.kv_heads(), 0..a.head_dim()])?;
                        let value_row = values.row(b)?;
                        let key = key_row.view(&[a.kv_heads(), a.head_dim()])?;
                        let value = value_row.view(&[a.kv_heads(), a.head_dim()])?;
                        let (k, v) = states.attention(b, layer)?;
                        sink.run(
                            attention::cache_append(
                                k.partition([a.max_seq(), 1, a.head_dim()]),
                                v.partition([a.max_seq(), 1, a.head_dim()]),
                                &key,
                                &value,
                                inputs.position(b),
                            )
                            .generics(a.cache_generics()),
                        )?;
                    }
                }
                Op::OnlineAttention { layer } => {
                    let ([queries], out) = buffers.io([Read::Whole(Slot::QRot)], Slot::AttnOut)?;
                    for b in 0..sessions {
                        let q_row = queries
                            .tensor
                            .slice(&[b * a.q_heads()..(b + 1) * a.q_heads(), 0..a.head_dim()])?;
                        let q = q_row.view(&[a.q_heads(), a.head_dim()])?;
                        let y = out.slice_mut(&[
                            b * a.q_heads()..(b + 1) * a.q_heads(),
                            0..a.head_dim(),
                        ])?;
                        let (k, v) = states.attention_ref(b, layer)?;
                        sink.run(
                            attention::online_attention(
                                y.partition([1, a.head_dim()]),
                                &q,
                                k,
                                v,
                                inputs.position(b),
                                a.scale(),
                            )
                            .generics(a.attention_generics()),
                        )?;
                    }
                }
                Op::OutputGate { layer } => {
                    let q_gate = Read::Member {
                        layer,
                        member: Projection::AttnQGate,
                    };
                    let ([source, q_gate], target) =
                        buffers.io([Read::Whole(Slot::AttnOut), q_gate], Slot::AttnGated)?;
                    output_gate(
                        model,
                        source.tensor,
                        q_gate.tensor,
                        q_gate.columns,
                        target,
                        batch,
                        sink,
                    )?
                }
                Op::TopK { layer } => {
                    let member = |member| Read::Member { layer, member };
                    let [logits, gates] = buffers
                        .read([member(Projection::Router), member(Projection::SharedGate)])?;
                    let logits = ProjectionRows::new(&logits)?;
                    let gates = ProjectionRows::new(&gates)?;
                    for b in 0..sessions {
                        let gate = gates.row(b)?;
                        let gate = gate.view(&[1, 1])?;
                        selections
                            .get_mut(layer, b)
                            .select(&m, &logits.row(b)?, &gate, sink)?;
                    }
                }
                Op::ExpertGateUp { layer } => {
                    let ([input], output) =
                        buffers.io([Read::Whole(Slot::Normed)], Slot::ExpertUp)?;
                    let w = weights.matrix(layer, WeightName::ExpertGateUp);
                    let tile = weight_gemv_tile(w, 2 * m.intermediate(), d.hidden(), slots);
                    for b in 0..sessions {
                        let input_row = input.tensor.slice(&[b..b + 1, 0..d.hidden()])?;
                        let x = input_row.view(&[1, d.hidden() / Q8_BLOCK, Q8_BLOCK])?;
                        let y = output
                            .slice_mut(&[b * slots..(b + 1) * slots, 0..2 * m.intermediate()])?;
                        expert_gemv_weight_view(
                            w,
                            &x,
                            selections.get(layer, b),
                            y.partition([1, tile.rows]),
                            2 * m.intermediate(),
                            0,
                            tile,
                        )
                        .submit(sink)?;
                    }
                }
                Op::ExpertSwiglu => {
                    let ([source], target) =
                        buffers.io([Read::Whole(Slot::ExpertUp)], Slot::ExpertAct)?;
                    for b in 0..sessions {
                        let x = source
                            .tensor
                            .slice(&[b * slots..(b + 1) * slots, 0..2 * m.intermediate()])?;
                        let y =
                            target.slice_mut(&[b * slots..(b + 1) * slots, 0..m.intermediate()])?;
                        sink.run(
                            swiglu::swiglu(y.partition([1, m.intermediate()]), &x)
                                .generics(vec![m.intermediate().to_string()]),
                        )?;
                    }
                }
                Op::ExpertDown { layer } => {
                    let ([source], target) =
                        buffers.io([Read::Whole(Slot::ExpertAct)], Slot::ExpertOut)?;
                    let w = weights.matrix(layer, WeightName::ExpertDown);
                    let tile = weight_gemv_tile(w, d.hidden(), m.intermediate(), slots);
                    for b in 0..sessions {
                        let act_rows = source
                            .tensor
                            .slice(&[b * slots..(b + 1) * slots, 0..m.intermediate()])?;
                        let x = act_rows.view(&[slots, m.intermediate() / Q8_BLOCK, Q8_BLOCK])?;
                        let y = target.slice_mut(&[b * slots..(b + 1) * slots, 0..d.hidden()])?;
                        expert_gemv_weight_view(
                            w,
                            &x,
                            selections.get(layer, b),
                            y.partition([1, tile.rows]),
                            d.hidden(),
                            1,
                            tile,
                        )
                        .submit(sink)?;
                    }
                }
                Op::Combine { layer } => {
                    let ([routed], out) =
                        buffers.io([Read::Whole(Slot::ExpertOut)], Slot::MoeDelta)?;
                    for b in 0..sessions {
                        let r = routed
                            .tensor
                            .slice(&[b * slots..(b + 1) * slots, 0..d.hidden()])?;
                        let y = out.slice_mut(&[b..b + 1, 0..d.hidden()])?;
                        sink.run(moe::combine_view(
                            y.partition([1, d.hidden()]),
                            &r,
                            selections.get(layer, b),
                        ))?;
                    }
                }
                Op::Add { input, delta, out } => {
                    let ([a, b], target) =
                        buffers.io([Read::Whole(input), Read::Whole(delta)], out)?;
                    sink.run(
                        elementwise::add(target.partition([1, d.hidden()]), a.tensor, b.tensor)
                            .generics(vec![d.hidden().to_string()]),
                    )?;
                }
            }
            Ok(())
        });
        if let Err(error) = result {
            return Err(format!("op {op_index} {op:?}: {error}").into());
        }
        after_op(op, ctx)?;
    }
    Ok(())
}
