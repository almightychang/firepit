use crate::Result;
use crate::checkpoint::Stream;
use crate::execute::gemv_tile;
use crate::execute::{input_output, three_inputs_output, two_inputs_output, weight};
use crate::model::{MixerWeights, Model};
use crate::plan::{NormName, Op, Slot, StepPlan, WeightName};
use crate::session::{GdnStateLayout, LayerState, Session};
use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{IntoPartition, PartitionMut, Reshape, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::attention::DEFAULT_PREFILL_TILE;
use firepit_kernels::gemm::{
    DEFAULT_GEMM_TILE, DEFAULT_GROUPED_GEMM_TILE, GemmTile, GroupInput, gemm, grouped_gemm,
};
use firepit_kernels::gemv::gemv;
use firepit_kernels::moe::PrefillPlan as MoePrefillPlan;
use firepit_kernels::sink::{Eager, Sink};
use firepit_kernels::weights::Weight;
use firepit_kernels::{attention, elementwise, embedding, gdn, gdn_chunk, moe, norm, rope, swiglu};
use std::sync::Arc;

const CHUNK: usize = gdn_chunk::DEFAULT_CHUNK_SIZE;
const BLOCK: usize = gdn_chunk::DEFAULT_BLOCK_SIZE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillSlot {
    Sequence(Slot),
    Query,
    Key,
    Value,
    Gate,
    Beta,
    Ut,
    Intra,
    Cum,
    Diagonal,
    Inverse,
    NextInverse,
    U,
    W,
}

impl PrefillSlot {
    fn shape(self, model: &Model, rows: usize, len: usize) -> Vec<usize> {
        let g = model.dims.gdn();
        let flat = rows / CHUNK * g.value_heads();
        match self {
            Self::Sequence(slot) => match slot {
                Slot::Mixer => vec![rows, g.value_heads(), g.value_head_dim()],
                Slot::Conv => vec![rows, g.conv_dim()],
                Slot::AttnOut => vec![
                    rows,
                    model.dims.attention().q_heads(),
                    model.dims.attention().head_dim(),
                ],
                Slot::ExpertUp | Slot::ExpertAct | Slot::ExpertOut => {
                    vec![1, slot.shape(model, 1)[1]]
                }
                Slot::Logits => vec![1, model.dims.vocab()],
                Slot::MoeDelta => vec![len, model.dims.hidden()],
                _ => slot.shape(model, rows),
            },
            Self::Query | Self::Key => vec![rows, g.key_heads(), g.key_head_dim()],
            Self::Value => vec![rows, g.value_heads(), g.value_head_dim()],
            Self::Gate | Self::Beta => vec![rows, g.value_heads()],
            Self::Ut | Self::Intra | Self::Inverse | Self::NextInverse => vec![flat, CHUNK, CHUNK],
            Self::Cum => vec![flat, CHUNK],
            Self::Diagonal => vec![flat * CHUNK / BLOCK, BLOCK, BLOCK],
            Self::U => vec![flat, CHUNK, g.value_head_dim()],
            Self::W => vec![flat, CHUNK, g.key_head_dim()],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PrefillOp {
    Kernel(Op),
    Conv { layer: usize },
    PrepareQk,
    PrepareValueGate { layer: usize },
    ChunkMatrices,
    ChunkDiagonal,
    ChunkMerge { row: usize },
    ChunkApply,
    ChunkSecond { layer: usize },
    CacheAppend { layer: usize },
    CausalAttention { layer: usize },
    TopK,
    HostSort,
    ExpertGateUp { layer: usize },
    ExpertDown { layer: usize },
    Combine,
}

struct ChunkSlots {
    q: Tensor<f32>,
    k: Tensor<f32>,
    v: Tensor<f32>,
    gate: Tensor<f32>,
    beta: Tensor<f32>,
    ut: Tensor<f32>,
    intra: Tensor<f32>,
    cum: Tensor<f32>,
    diagonal: Tensor<f32>,
    inverse: Tensor<f32>,
    next_inverse: Tensor<f32>,
    u: Tensor<f32>,
    w: Tensor<f32>,
}

impl ChunkSlots {
    fn new(model: &Model, rows: usize, stream: &Stream) -> Result<Self> {
        let zeros = |shape: &[usize]| -> Result<Tensor<f32>> {
            Ok(api::zeros::<f32>(shape).sync_on(stream)?)
        };
        Ok(Self {
            q: zeros(&PrefillSlot::Query.shape(model, rows, rows))?,
            k: zeros(&PrefillSlot::Key.shape(model, rows, rows))?,
            v: zeros(&PrefillSlot::Value.shape(model, rows, rows))?,
            gate: zeros(&PrefillSlot::Gate.shape(model, rows, rows))?,
            beta: zeros(&PrefillSlot::Beta.shape(model, rows, rows))?,
            ut: zeros(&PrefillSlot::Ut.shape(model, rows, rows))?,
            intra: zeros(&PrefillSlot::Intra.shape(model, rows, rows))?,
            cum: zeros(&PrefillSlot::Cum.shape(model, rows, rows))?,
            diagonal: zeros(&PrefillSlot::Diagonal.shape(model, rows, rows))?,
            inverse: zeros(&PrefillSlot::Inverse.shape(model, rows, rows))?,
            next_inverse: zeros(&PrefillSlot::NextInverse.shape(model, rows, rows))?,
            u: zeros(&PrefillSlot::U.shape(model, rows, rows))?,
            w: zeros(&PrefillSlot::W.shape(model, rows, rows))?,
        })
    }
}

pub struct PrefillPlan {
    pub ops: Vec<PrefillOp>,
    buffers: Vec<Tensor<f32>>,
    tokens: Tensor<i32>,
    cos: Tensor<f32>,
    sin: Tensor<f32>,
    chunk: Option<ChunkSlots>,
    conv_spare: Option<Tensor<f32>>,
    recurrent_spare: Option<Tensor<f32>>,
    moe: Option<MoePrefillPlan>,
    top_ids: Tensor<i32>,
    top_weights: Tensor<f32>,
    len: usize,
    rows: usize,
    position: usize,
    conv_tile: usize,
    value_tile: usize,
}

impl PrefillPlan {
    pub fn new(
        model: &Model,
        tokens: &[u32],
        position: usize,
        conv_tile: usize,
        value_tile: usize,
        stream: &Stream,
    ) -> Result<Self> {
        let len = tokens.len();
        let rows = len.div_ceil(CHUNK) * CHUNK;
        let mut ops = Vec::new();
        for op in StepPlan::ops(model) {
            match op {
                Op::GdnConv { layer } => ops.push(PrefillOp::Conv { layer }),
                Op::GdnDelta { layer } => {
                    ops.push(PrefillOp::PrepareQk);
                    ops.push(PrefillOp::PrepareValueGate { layer });
                    ops.push(PrefillOp::ChunkMatrices);
                    ops.push(PrefillOp::ChunkDiagonal);
                    for row in 0..CHUNK / BLOCK {
                        ops.push(PrefillOp::ChunkMerge { row });
                    }
                    ops.push(PrefillOp::ChunkApply);
                    ops.push(PrefillOp::ChunkSecond { layer });
                }
                Op::CacheAppend { layer } => ops.push(PrefillOp::CacheAppend { layer }),
                Op::OnlineAttention { layer } => ops.push(PrefillOp::CausalAttention { layer }),
                Op::TopK { .. } => {
                    ops.push(PrefillOp::TopK);
                    ops.push(PrefillOp::HostSort);
                }
                Op::ExpertGateUp { layer } => ops.push(PrefillOp::ExpertGateUp { layer }),
                Op::ExpertDown { layer } => ops.push(PrefillOp::ExpertDown { layer }),
                Op::Combine { .. } => ops.push(PrefillOp::Combine),
                _ => ops.push(PrefillOp::Kernel(op)),
            }
        }
        let buffers = Slot::ALL
            .iter()
            .map(|&slot| {
                let shape = PrefillSlot::Sequence(slot).shape(model, rows, len);
                api::zeros::<f32>(&shape).sync_on(stream)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut padded_tokens: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
        padded_tokens.resize(rows, 0);
        let tokens = api::copy_host_vec_to_device(&Arc::new(padded_tokens)).sync_on(stream)?;
        let positions: Vec<usize> = (position..position + rows).collect();
        let (cos, sin) = rope::frequencies(&positions, &model.dims.q_rope());
        let d = model.dims.attention().head_dim();
        let cos = api::copy_host_vec_to_device(&Arc::new(cos))
            .reshape(&[rows, d])
            .sync_on(stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin))
            .reshape(&[rows, d])
            .sync_on(stream)?;
        let chunk = if model
            .dims
            .layers()
            .iter()
            .any(|kind| matches!(kind, crate::LayerKind::Gdn))
        {
            Some(ChunkSlots::new(model, rows, stream)?)
        } else {
            None
        };
        let (conv_spare, recurrent_spare) = if chunk.is_some() {
            let layout = GdnStateLayout::new(model.dims.gdn());
            (
                Some(api::zeros::<f32>(&layout.conv_storage()).sync_on(stream)?),
                Some(api::zeros::<f32>(&layout.recurrent_storage()).sync_on(stream)?),
            )
        } else {
            (None, None)
        };
        let top_width = model.dims.moe().top_k().next_power_of_two();
        let top_ids = api::zeros::<i32>(&[len, top_width]).sync_on(stream)?;
        let top_weights = api::zeros::<f32>(&[len, top_width]).sync_on(stream)?;
        Ok(Self {
            ops,
            buffers,
            tokens,
            cos,
            sin,
            chunk,
            conv_spare,
            recurrent_spare,
            moe: None,
            top_ids,
            top_weights,
            len,
            rows,
            position,
            conv_tile,
            value_tile,
        })
    }

    pub fn logits(&self, stream: &Stream) -> Result<Vec<f32>> {
        Ok(self.buffers[Slot::Logits.index()]
            .dup()
            .to_host_vec()
            .sync_on(stream)?)
    }
}

fn gemm_tile(cols: usize, rows: usize, grouped: bool) -> GemmTile {
    let mut tile = if grouped {
        DEFAULT_GROUPED_GEMM_TILE
    } else {
        DEFAULT_GEMM_TILE
    };
    while rows % tile.n != 0 {
        tile.n /= 2;
    }
    while cols % tile.k != 0 {
        tile.k /= 2;
    }
    tile
}

pub(crate) fn run(
    model: &Model,
    plan: &mut PrefillPlan,
    session: &mut Session,
    stream: &Stream,
) -> Result<()> {
    let d = &model.dims;
    let g = d.gdn();
    let a = d.attention();
    let m = d.moe();
    let rows = plan.rows;
    let len = plan.len;
    let mut sink = Eager(stream);
    for (index, op) in plan.ops.clone().into_iter().enumerate() {
        let result = (|| -> Result<()> {
            match op {
                PrefillOp::Kernel(Op::Embed { out }) => sink.run(
                    embedding::embedding(
                        (&mut plan.buffers[out.index()]).partition([1, d.hidden()]),
                        &model.embedding,
                        &plan.tokens,
                    )
                    .generics(vec![d.hidden().to_string()]),
                )?,
                PrefillOp::Kernel(Op::Norm {
                    layer,
                    name,
                    input,
                    out,
                }) => {
                    let (source, target) = input_output(&mut plan.buffers, input, out);
                    let (width, count, gamma, eps) = match name {
                        NormName::Input => (
                            d.hidden(),
                            rows,
                            &model.layers[layer].input_norm,
                            d.norm_eps(),
                        ),
                        NormName::Post => (
                            d.hidden(),
                            rows,
                            &model.layers[layer].post_norm,
                            d.norm_eps(),
                        ),
                        NormName::Final => (d.hidden(), 1, &model.final_norm, d.norm_eps()),
                        NormName::Q => {
                            let MixerWeights::Attention(w) = &model.layers[layer].mixer else {
                                unreachable!()
                            };
                            (a.head_dim(), rows * a.q_heads(), &w.q_norm, d.norm_eps())
                        }
                        NormName::K => {
                            let MixerWeights::Attention(w) = &model.layers[layer].mixer else {
                                unreachable!()
                            };
                            (a.head_dim(), rows * a.kv_heads(), &w.k_norm, d.norm_eps())
                        }
                        NormName::GdnGated => {
                            let MixerWeights::Gdn(w) = &model.layers[layer].mixer else {
                                unreachable!()
                            };
                            (
                                g.value_head_dim(),
                                rows * g.value_heads(),
                                &w.gated_norm,
                                g.eps(),
                            )
                        }
                    };
                    if matches!(name, NormName::Final) {
                        let shaped = source.view(&[rows, width])?;
                        let x = shaped.slice(&[len - 1..len, 0..width])?;
                        let y = target.slice_mut(&[len - 1..len, 0..width])?;
                        sink.run(
                            norm::rms_norm(
                                y.partition([1, width]),
                                &x,
                                gamma,
                                1.0 / width as f32,
                                eps,
                            )
                            .generics(vec![width.to_string()]),
                        )?;
                    } else if matches!(name, NormName::Q) {
                        let shaped = source.view(&[count, 2 * width])?;
                        let x = shaped.slice(&[0..count, 0..width])?;
                        sink.run(
                            norm::rms_norm(
                                target.partition([1, width]),
                                &x,
                                gamma,
                                1.0 / width as f32,
                                eps,
                            )
                            .generics(vec![width.to_string()]),
                        )?;
                    } else {
                        let x = source.view(&[count, width])?;
                        sink.run(
                            norm::rms_norm(
                                target.partition([1, width]),
                                &x,
                                gamma,
                                1.0 / width as f32,
                                eps,
                            )
                            .generics(vec![width.to_string()]),
                        )?;
                    }
                }
                PrefillOp::Kernel(Op::Gemv {
                    layer,
                    name,
                    input,
                    out,
                }) => {
                    let (source, target) = input_output(&mut plan.buffers, input, out);
                    let w = weight(model, layer, name);
                    let Weight::Dense(dense) = w else {
                        return Err("prefill requires dense bf16 weights".into());
                    };
                    let cols = source
                        .shape()
                        .iter()
                        .map(|&x| x as usize)
                        .product::<usize>()
                        / rows;
                    let output_rows = dense.rows;
                    if matches!(name, WeightName::LmHead) {
                        let shaped = source.view(&[rows, cols])?;
                        let last = shaped.slice(&[len - 1..len, 0..cols])?;
                        let x = last.view(&[
                            1,
                            cols / firepit_kernels::weights::Q8_BLOCK,
                            firepit_kernels::weights::Q8_BLOCK,
                        ])?;
                        gemv(
                            w,
                            &x,
                            target.partition([1, gemv_tile(output_rows, cols).rows]),
                            gemv_tile(output_rows, cols),
                        )
                        .submit(&mut sink)?;
                    } else {
                        let x = source.view(&[rows, cols])?;
                        let tile = gemm_tile(cols, output_rows, false);
                        gemm(dense, &x, target.partition([tile.m, tile.n]), tile)
                            .submit(&mut sink)?;
                    }
                }
                PrefillOp::Kernel(Op::GdnGate) => {
                    let (source, gate, target) =
                        two_inputs_output(&mut plan.buffers, Slot::MixerNorm, Slot::Z, Slot::Gated);
                    let x = source.view(&[rows * g.value_heads(), g.value_head_dim()])?;
                    let z = gate.view(&[rows * g.value_heads(), g.value_head_dim()])?;
                    sink.run(
                        gdn::gated_silu(target.partition([1, g.value_head_dim()]), &x, &z)
                            .generics(vec![g.value_head_dim().to_string()]),
                    )?;
                }
                PrefillOp::Kernel(Op::AttnRope { query }) => {
                    let (input, out, rdims) = if query {
                        (Slot::QNorm, Slot::QRot, d.q_rope())
                    } else {
                        (Slot::KNorm, Slot::KRot, d.k_rope())
                    };
                    let (source, target) = input_output(&mut plan.buffers, input, out);
                    let x = source.view(&[rows * rdims.heads(), rdims.head_dim()])?;
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
                PrefillOp::Kernel(Op::OutputGate) => {
                    let (source, qgate, target) = two_inputs_output(
                        &mut plan.buffers,
                        Slot::AttnOut,
                        Slot::QGate,
                        Slot::AttnGated,
                    );
                    let width = a.head_dim();
                    let count = rows * a.q_heads();
                    let x = source.view(&[count, width])?;
                    let shaped = qgate.view(&[count, 2 * width])?;
                    let gate = shaped.slice(&[0..count, width..2 * width])?;
                    sink.run(
                        attention::output_gate(target.partition([1, width]), &x, &gate)
                            .generics(a.gate_generics()),
                    )?;
                }
                PrefillOp::Kernel(Op::ExpertSwiglu) => {
                    let count = plan
                        .moe
                        .as_ref()
                        .ok_or("missing MoE layout")?
                        .layout()
                        .padded_rows();
                    let (source, target) =
                        input_output(&mut plan.buffers, Slot::ExpertUp, Slot::ExpertAct);
                    sink.run(
                        swiglu::swiglu(target.partition([1, m.intermediate()]), source)
                            .generics(vec![m.intermediate().to_string()]),
                    )?;
                    debug_assert_eq!(source.shape()[0] as usize, count);
                }
                PrefillOp::Kernel(Op::SharedSwiglu) => {
                    let (source, target) =
                        input_output(&mut plan.buffers, Slot::SharedUp, Slot::SharedAct);
                    sink.run(
                        swiglu::swiglu(target.partition([1, m.shared_intermediate()]), source)
                            .generics(vec![m.shared_intermediate().to_string()]),
                    )?;
                }
                PrefillOp::Kernel(Op::Add { input, delta, out }) => {
                    let (x, y, target) = two_inputs_output(&mut plan.buffers, input, delta, out);
                    let x = x.slice(&[0..len, 0..d.hidden()])?;
                    let y = y.slice(&[0..len, 0..d.hidden()])?;
                    let target = target.slice_mut(&[0..len, 0..d.hidden()])?;
                    sink.run(
                        elementwise::add(target.partition([1, d.hidden()]), &x, &y)
                            .generics(vec![d.hidden().to_string()]),
                    )?;
                }
                PrefillOp::Kernel(_) => unreachable!(),
                _ => run_special(model, plan, session, &mut sink, op, stream)?,
            }
            Ok(())
        })();
        if let Err(error) = result {
            return Err(format!("prefill op {index} {op:?}: {error}").into());
        }
    }
    Ok(())
}

fn run_special(
    model: &Model,
    plan: &mut PrefillPlan,
    session: &mut Session,
    sink: &mut Eager<'_>,
    op: PrefillOp,
    stream: &Stream,
) -> Result<()> {
    let d = &model.dims;
    let g = d.gdn();
    let a = d.attention();
    let m = d.moe();
    let (len, rows) = (plan.len, plan.rows);
    let chunks = rows / CHUNK;
    match op {
        PrefillOp::Conv { layer } => {
            let MixerWeights::Gdn(w) = &model.layers[layer].mixer else {
                unreachable!()
            };
            let layout = GdnStateLayout::new(g);
            let (source, target) = input_output(&mut plan.buffers, Slot::Qkv, Slot::Conv);
            let shaped = source.view(&[rows, g.conv_dim()])?;
            let x = shaped.slice(&[0..len, 0..g.conv_dim()])?;
            let y = target.slice_mut(&[0..len, 0..g.conv_dim()])?;
            let LayerState::Gdn { conv, .. } = &mut session.state[layer] else {
                unreachable!()
            };
            let old = std::mem::replace(conv, plan.conv_spare.take().unwrap());
            let mut state = old.reshape(&layout.conv_prefill())?;
            let launched = sink.run(
                gdn::conv1d_silu_prefill(
                    y.partition([len.next_power_of_two(), plan.conv_tile]),
                    &x,
                    (&mut state).partition(layout.conv_prefill_grid(plan.conv_tile)),
                    &w.conv,
                    g.conv_kernel() as i32,
                )
                .generics(vec![
                    len.next_power_of_two().to_string(),
                    plan.conv_tile.to_string(),
                    g.conv_kernel().next_power_of_two().to_string(),
                ]),
            );
            let spare = std::mem::replace(conv, state.reshape(&layout.conv_storage())?);
            plan.conv_spare = Some(spare);
            launched?;
        }
        PrefillOp::PrepareQk => {
            let conv = &plan.buffers[Slot::Conv.index()];
            let x = conv.view(&[rows, g.conv_dim()])?;
            let q_slice = x.slice(&[0..rows, 0..g.key_dim()])?;
            let k_slice = x.slice(&[0..rows, g.key_dim()..2 * g.key_dim()])?;
            let c = plan.chunk.as_mut().unwrap();
            sink.run(
                gdn::prepare_chunk_qk(
                    (&mut c.q).partition([1, 1, g.key_head_dim()]),
                    (&mut c.k).partition([1, 1, g.key_head_dim()]),
                    &q_slice,
                    &k_slice,
                    len as i32,
                )
                .generics(vec![
                    g.key_heads().to_string(),
                    g.key_head_dim().to_string(),
                ]),
            )?;
        }
        PrefillOp::PrepareValueGate { layer } => {
            let MixerWeights::Gdn(w) = &model.layers[layer].mixer else {
                unreachable!()
            };
            let x = plan.buffers[Slot::Conv.index()].view(&[rows, g.conv_dim()])?;
            let v_slice = x.slice(&[0..rows, 2 * g.key_dim()..g.conv_dim()])?;
            let a_shaped = plan.buffers[Slot::A.index()].view(&[rows, g.value_heads()])?;
            let b_shaped = plan.buffers[Slot::B.index()].view(&[rows, g.value_heads()])?;
            let av = a_shaped.slice(&[0..rows, 0..g.value_heads()])?;
            let bv = b_shaped.slice(&[0..rows, 0..g.value_heads()])?;
            let c = plan.chunk.as_mut().unwrap();
            sink.run(
                gdn::prepare_chunk_value_gate(
                    (&mut c.v).partition([1, 1, g.value_head_dim()]),
                    (&mut c.gate).partition([1, 1]),
                    (&mut c.beta).partition([1, 1]),
                    &v_slice,
                    &av,
                    &bv,
                    &w.a_log,
                    &w.dt_bias,
                    len as i32,
                )
                .generics(vec![
                    g.value_heads().to_string(),
                    g.value_head_dim().to_string(),
                ]),
            )?;
        }
        PrefillOp::ChunkMatrices => {
            let c = plan.chunk.as_mut().unwrap();
            let q =
                c.q.view(&[chunks, CHUNK, g.key_heads(), g.key_head_dim()])?;
            let k =
                c.k.view(&[chunks, CHUNK, g.key_heads(), g.key_head_dim()])?;
            let gate = c.gate.view(&[chunks, CHUNK, g.value_heads()])?;
            let beta = c.beta.view(&[chunks, CHUNK, g.value_heads()])?;
            sink.run(
                gdn_chunk::first_phase_matrices(
                    (&mut c.ut).partition([1, CHUNK, CHUNK]),
                    (&mut c.intra).partition([1, CHUNK, CHUNK]),
                    (&mut c.cum).partition([1, CHUNK]),
                    &q,
                    &k,
                    &gate,
                    &beta,
                    CHUNK as i32,
                    g.eps(),
                    g.inv_sqrt_key_dim(),
                )
                .generics(vec![
                    CHUNK.to_string(),
                    g.key_head_dim().to_string(),
                    g.key_heads().to_string(),
                    g.value_heads().to_string(),
                ]),
            )?;
        }
        PrefillOp::ChunkDiagonal => {
            let c = plan.chunk.as_mut().unwrap();
            sink.run(
                gdn_chunk::invert_diagonal((&mut c.diagonal).partition([1, BLOCK, BLOCK]), &c.ut)
                    .generics(vec![CHUNK.to_string(), BLOCK.to_string()]),
            )?;
        }
        PrefillOp::ChunkMerge { row } => {
            let c = plan.chunk.as_mut().unwrap();
            sink.run(
                gdn_chunk::merge_inverse(
                    (&mut c.next_inverse).partition([1, CHUNK, CHUNK]),
                    &c.inverse,
                    &c.ut,
                    &c.diagonal,
                    row as i32,
                )
                .generics(vec![CHUNK.to_string(), BLOCK.to_string()]),
            )?;
            std::mem::swap(&mut c.inverse, &mut c.next_inverse);
        }
        PrefillOp::ChunkApply => {
            let c = plan.chunk.as_mut().unwrap();
            let k =
                c.k.view(&[chunks, CHUNK, g.key_heads(), g.key_head_dim()])?;
            let v =
                c.v.view(&[chunks, CHUNK, g.value_heads(), g.value_head_dim()])?;
            let beta = c.beta.view(&[chunks, CHUNK, g.value_heads()])?;
            sink.run(
                gdn_chunk::apply_inverse(
                    (&mut c.u).partition([1, CHUNK, g.value_head_dim()]),
                    (&mut c.w).partition([1, CHUNK, g.key_head_dim()]),
                    &c.inverse,
                    &c.cum,
                    &k,
                    &v,
                    &beta,
                    CHUNK as i32,
                    g.eps(),
                )
                .generics(vec![
                    CHUNK.to_string(),
                    g.key_head_dim().to_string(),
                    g.value_head_dim().to_string(),
                    g.key_heads().to_string(),
                    g.value_heads().to_string(),
                ]),
            )?;
        }
        PrefillOp::ChunkSecond { layer } => {
            let c = plan.chunk.as_ref().unwrap();
            let layout = GdnStateLayout::new(g);
            let q =
                c.q.view(&[chunks, CHUNK, g.key_heads(), g.key_head_dim()])?;
            let k =
                c.k.view(&[chunks, CHUNK, g.key_heads(), g.key_head_dim()])?;
            let LayerState::Gdn { recurrent, .. } = &mut session.state[layer] else {
                unreachable!()
            };
            let target = &mut plan.buffers[Slot::Mixer.index()];
            let old = std::mem::replace(recurrent, plan.recurrent_spare.take().unwrap());
            let mut state = old.reshape(&layout.recurrent_prefill())?;
            let launched = sink.run(
                gdn_chunk::second_phase(
                    target.partition([rows.next_power_of_two(), 1, plan.value_tile]),
                    (&mut state).partition(layout.recurrent_prefill_grid(plan.value_tile)),
                    &c.u,
                    &c.w,
                    &c.intra,
                    &c.cum,
                    &q,
                    &k,
                    chunks as i32,
                    g.eps(),
                    g.inv_sqrt_key_dim(),
                )
                .generics(vec![
                    rows.next_power_of_two().to_string(),
                    CHUNK.to_string(),
                    g.key_head_dim().to_string(),
                    plan.value_tile.to_string(),
                    g.key_heads().to_string(),
                    g.value_heads().to_string(),
                ]),
            );
            let spare = std::mem::replace(recurrent, state.reshape(&layout.recurrent_storage())?);
            plan.recurrent_spare = Some(spare);
            launched?;
        }
        PrefillOp::CacheAppend { layer } => {
            let k_shaped =
                plan.buffers[Slot::KRot.index()].view(&[rows, a.kv_heads(), a.head_dim()])?;
            let v_shaped =
                plan.buffers[Slot::V.index()].view(&[rows, a.kv_heads(), a.head_dim()])?;
            let keys = k_shaped.slice(&[0..len, 0..a.kv_heads(), 0..a.head_dim()])?;
            let values = v_shaped.slice(&[0..len, 0..a.kv_heads(), 0..a.head_dim()])?;
            let LayerState::Attention { k, v } = &mut session.state[layer] else {
                unreachable!()
            };
            let kc = k.slice_mut(&[
                plan.position..plan.position + len,
                0..a.kv_heads(),
                0..a.head_dim(),
            ])?;
            let vc = v.slice_mut(&[
                plan.position..plan.position + len,
                0..a.kv_heads(),
                0..a.head_dim(),
            ])?;
            sink.run(
                attention::cache_append_prefill(
                    kc.partition([1, 1, a.head_dim()]),
                    vc.partition([1, 1, a.head_dim()]),
                    &keys,
                    &values,
                )
                .generics(a.prefill_cache_generics()),
            )?;
        }
        PrefillOp::CausalAttention { layer } => {
            let (query, target) = input_output(&mut plan.buffers, Slot::QRot, Slot::AttnOut);
            let q_shaped = query.view(&[rows, a.q_heads(), a.head_dim()])?;
            let q = q_shaped.slice(&[0..len, 0..a.q_heads(), 0..a.head_dim()])?;
            let y = target.slice_mut(&[0..len, 0..a.q_heads(), 0..a.head_dim()])?;
            let LayerState::Attention { k, v } = &session.state[layer] else {
                unreachable!()
            };
            sink.run(
                attention::causal_attention_prefill(
                    y.partition([DEFAULT_PREFILL_TILE.m, 1, a.head_dim()]),
                    &q,
                    k,
                    v,
                    plan.position as i32,
                    a.scale(),
                )
                .generics(a.prefill_generics(DEFAULT_PREFILL_TILE)),
            )?;
        }
        PrefillOp::TopK => {
            let logits = plan.buffers[Slot::Router.index()].slice(&[0..len, 0..m.experts()])?;
            sink.run(
                moe::top_k_prefill(
                    &logits,
                    (&mut plan.top_ids).partition([1, m.top_k().next_power_of_two()]),
                    (&mut plan.top_weights).partition([1, m.top_k().next_power_of_two()]),
                    m.experts() as i32,
                    m.top_k() as i32,
                )
                .generics(m.top_k_generics()),
            )?;
        }
        PrefillOp::HostSort => {
            let ids = plan.top_ids.dup().to_host_vec().sync_on(stream)?;
            let weights = plan.top_weights.dup().to_host_vec().sync_on(stream)?;
            let top_width = m.top_k().next_power_of_two();
            let ids: Vec<i32> = ids
                .chunks_exact(top_width)
                .flat_map(|row| row[..m.top_k()].iter().copied())
                .collect();
            let weights: Vec<f32> = weights
                .chunks_exact(top_width)
                .flat_map(|row| row[..m.top_k()].iter().copied())
                .collect();
            let layout =
                MoePrefillPlan::parse(&ids, &weights, &m, DEFAULT_GROUPED_GEMM_TILE.m, stream)?;
            let count = layout.layout().padded_rows();
            for slot in [Slot::ExpertUp, Slot::ExpertAct, Slot::ExpertOut] {
                let width = match slot {
                    Slot::ExpertUp => 2 * m.intermediate(),
                    Slot::ExpertAct => m.intermediate(),
                    _ => d.hidden(),
                };
                plan.buffers[slot.index()] = api::zeros::<f32>(&[count, width]).sync_on(stream)?;
            }
            plan.moe = Some(layout);
        }
        PrefillOp::ExpertGateUp { layer } => {
            let layout = plan.moe.as_ref().ok_or("missing MoE layout")?;
            let (source, target) = input_output(&mut plan.buffers, Slot::Normed, Slot::ExpertUp);
            let x = source.slice(&[0..len, 0..d.hidden()])?;
            let tile = gemm_tile(d.hidden(), 2 * m.intermediate(), true);
            grouped_gemm(
                &model.layers[layer].moe.gate_up,
                &x,
                target.partition([tile.m, tile.n]),
                layout,
                2 * m.intermediate(),
                GroupInput::Tokens,
                tile,
            )
            .submit(sink)?;
        }
        PrefillOp::ExpertDown { layer } => {
            let layout = plan.moe.as_ref().ok_or("missing MoE layout")?;
            let (source, target) =
                input_output(&mut plan.buffers, Slot::ExpertAct, Slot::ExpertOut);
            let x = source.view(&[layout.layout().padded_rows(), m.intermediate()])?;
            let tile = gemm_tile(m.intermediate(), d.hidden(), true);
            grouped_gemm(
                &model.layers[layer].moe.down,
                &x,
                target.partition([tile.m, tile.n]),
                layout,
                d.hidden(),
                GroupInput::Padded,
                tile,
            )
            .submit(sink)?;
        }
        PrefillOp::Combine => {
            let layout = plan.moe.as_ref().ok_or("missing MoE layout")?;
            let (routed, shared, gate, target) = three_inputs_output(
                &mut plan.buffers,
                Slot::ExpertOut,
                Slot::Shared,
                Slot::SharedGate,
                Slot::MoeDelta,
            );
            let shared = shared.slice(&[0..len, 0..d.hidden()])?;
            let gate = gate.slice(&[0..len, 0..1])?;
            sink.run(moe::combine_prefill_view(
                target.partition([1, d.hidden()]),
                routed,
                layout,
                &shared,
                &gate,
            ))?;
        }
        PrefillOp::Kernel(_) => unreachable!(),
    }
    Ok(())
}
