use crate::Result;
use crate::model::{MixerWeights, Model};
use crate::plan::{NormName, Op, Slot, StepPlan, WeightName};
use crate::session::{LayerState, Session};
use cutile::tensor::{IntoPartition, PartitionMut, Tensor};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::gemv::{DEFAULT_GEMV_TILE, GemvTile, expert_gemv_view, gemv};
use firepit_kernels::sink::Sink;
use firepit_kernels::weights::{Q8_BLOCK, Weight};
use firepit_kernels::{attention, elementwise, embedding, gdn, moe, norm, rope, swiglu};

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

fn weight(model: &Model, layer: usize, name: WeightName) -> &Weight {
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

pub(crate) fn run(
    model: &Model,
    plan: &mut StepPlan,
    sessions: &mut [&mut Session],
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
                        let LayerState::Gdn { conv, .. } = &mut session.state[layer] else {
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
