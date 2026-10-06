use crate::Result;
use crate::checkpoint::Stream;
#[cfg(feature = "testing")]
use crate::engine::RoutingLayerTrace;
use crate::execute::{
    gdn_gate, heads, input_output, output_gate, rms_norm, row_width, two_inputs_output, weight,
    weight_gemv_tile,
};
use crate::model::{MixerWeights, Model, Projection};
use crate::plan::{NormName, Op, Slot, StepPlan, WeightName};
use crate::session::{GdnStateLayout, LayerState, Session};
use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{IntoPartition, PartitionMut, Reshape, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::attention::DEFAULT_PREFILL_TILE;
use firepit_kernels::gemm::{
    DEFAULT_GEMM_TILE, DEFAULT_GROUPED_GEMM_TILE, GemmTile, fp8_gemm, fp8_grouped_gemm_view, gemm,
    grouped_gemm_view, nvfp4_gemm, nvfp4_grouped_gemm_view,
};
use firepit_kernels::gemv::gemv;
use firepit_kernels::moe::{PrefillLayout as MoePrefillLayout, PrefillPlan as MoePrefillPlan};
use firepit_kernels::sink::Sink;
use firepit_kernels::weights::Weight;
use firepit_kernels::{attention, elementwise, embedding, gdn, gdn_chunk, moe, rope, swiglu};
use std::sync::Arc;

const CHUNK: usize = gdn_chunk::DEFAULT_CHUNK_SIZE;
const BLOCK: usize = gdn_chunk::DEFAULT_BLOCK_SIZE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoredRows {
    start: usize,
    end: usize,
    tokens: usize,
}

impl ScoredRows {
    pub fn parse(range: std::ops::Range<usize>, tokens: usize) -> Result<Self> {
        if range.start >= range.end || range.end > tokens {
            return Err("invalid scored rows".into());
        }
        Ok(Self {
            start: range.start,
            end: range.end,
            tokens,
        })
    }

    pub fn last(tokens: usize) -> Result<Self> {
        let start = tokens.checked_sub(1).ok_or("empty scored sequence")?;
        Self::parse(start..tokens, tokens)
    }

    pub fn range(self) -> std::ops::Range<usize> {
        self.start..self.end
    }

    fn len(self) -> usize {
        self.end - self.start
    }

    fn outer(self) -> std::ops::Range<usize> {
        if self.len() == 1 {
            return self.range();
        }
        let tile = DEFAULT_GEMM_TILE.m;
        self.start / tile * tile..self.end.div_ceil(tile) * tile
    }
}

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
    TopK { layer: usize },
    HostSort,
    Gather,
    ExpertGateUp { layer: usize },
    ExpertDown { layer: usize },
    Combine,
}

impl PrefillOp {
    pub(crate) fn kind(self) -> &'static str {
        match self {
            Self::Kernel(Op::Gemv {
                name: WeightName::LmHead,
                ..
            }) => "lm_head",
            Self::Kernel(Op::Gemv { .. }) => "gemm",
            Self::Kernel(op) => op.kind(),
            Self::Conv { .. } => "gdn_conv",
            Self::PrepareQk | Self::PrepareValueGate { .. } => "gdn_prepare",
            Self::ChunkMatrices => "gdn_chunk_matrices",
            Self::ChunkDiagonal => "gdn_chunk_diagonal",
            Self::ChunkMerge { .. } => "gdn_chunk_merge",
            Self::ChunkApply => "gdn_chunk_apply",
            Self::ChunkSecond { .. } => "gdn_chunk_second",
            Self::CacheAppend { .. } => "cache_append",
            Self::CausalAttention { .. } => "causal_attention",
            Self::TopK { .. } => "moe_top_k",
            Self::HostSort => "moe_host_sort",
            Self::Gather => "moe_gather",
            Self::ExpertGateUp { .. } | Self::ExpertDown { .. } => "moe_grouped_gemm",
            Self::Combine => "moe_combine",
        }
    }
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
    moe_sorted: Tensor<f32>,
    top_ids: Tensor<i32>,
    top_weights: Tensor<f32>,
    len: usize,
    rows: usize,
    position: usize,
    conv_tile: usize,
    value_tile: usize,
    scored: ScoredRows,
}

impl PrefillPlan {
    pub fn new(
        model: &Model,
        tokens: &[u32],
        position: usize,
        conv_tile: usize,
        value_tile: usize,
        scored: ScoredRows,
        stream: &Stream,
    ) -> Result<Self> {
        let len = tokens.len();
        if scored.tokens != len {
            return Err("scored rows belong to another sequence length".into());
        }
        let rows = len.div_ceil(CHUNK) * CHUNK;
        if scored.outer().end > rows {
            return Err("scored rows exceed padded sequence".into());
        }
        let m = model.dims.moe();
        let max_moe_rows = MoePrefillLayout::max_padded_rows(
            len,
            &m.with_shared_expert(),
            DEFAULT_GROUPED_GEMM_TILE.m,
        )?;
        let mut ops = Vec::new();
        for op in StepPlan::ops(model)? {
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
                Op::TopK { layer } => {
                    ops.push(PrefillOp::TopK { layer });
                    ops.push(PrefillOp::HostSort);
                    ops.push(PrefillOp::Gather);
                }
                Op::ExpertGateUp { layer } => ops.push(PrefillOp::ExpertGateUp { layer }),
                Op::ExpertDown { layer } => ops.push(PrefillOp::ExpertDown { layer }),
                Op::Combine { .. } => ops.push(PrefillOp::Combine),
                _ => ops.push(PrefillOp::Kernel(op)),
            }
        }
        let buffers = Slot::all(model)
            .map(|slot| {
                let mut shape = PrefillSlot::Sequence(slot).shape(model, rows, len);
                if matches!(slot, Slot::ExpertUp | Slot::ExpertAct | Slot::ExpertOut) {
                    shape[0] = max_moe_rows;
                }
                if slot == Slot::Logits {
                    shape = vec![scored.outer().len(), model.dims.vocab()];
                }
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
        let top_width = m.with_shared_expert().width();
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
            moe_sorted: api::zeros::<f32>(&[max_moe_rows, model.dims.hidden()]).sync_on(stream)?,
            top_ids,
            top_weights,
            len,
            rows,
            position,
            conv_tile,
            value_tile,
            scored,
        })
    }

    pub fn logits(&self, stream: &Stream) -> Result<Vec<f32>> {
        let all: Vec<f32> = self.buffers[Slot::Logits.index()]
            .dup()
            .to_host_vec()
            .sync_on(stream)?;
        let first = (self.scored.start - self.scored.outer().start)
            * (all.len() / self.scored.outer().len());
        let end = first + self.scored.len() * (all.len() / self.scored.outer().len());
        if first == 0 && end == all.len() {
            Ok(all)
        } else {
            Ok(all[first..end].to_vec())
        }
    }

    #[cfg(feature = "testing")]
    pub(crate) fn reset_rope_positions(&mut self, model: &Model, stream: &Stream) -> Result<()> {
        let positions: Vec<usize> = (0..self.rows).collect();
        let (cos, sin) = rope::frequencies(&positions, &model.dims.q_rope());
        let shape = &[self.rows, model.dims.attention().head_dim()];
        self.cos = api::copy_host_vec_to_device(&Arc::new(cos))
            .reshape(shape)
            .sync_on(stream)?;
        self.sin = api::copy_host_vec_to_device(&Arc::new(sin))
            .reshape(shape)
            .sync_on(stream)?;
        Ok(())
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
    sink: &mut impl Sink,
) -> Result<()> {
    run_inner(
        model,
        plan,
        session,
        stream,
        sink,
        plan.scored.start,
        |_, _| Ok(()),
    )
}

#[cfg(feature = "testing")]
pub(crate) fn run_observed(
    model: &Model,
    plan: &mut PrefillPlan,
    session: &mut Session,
    stream: &Stream,
    first_row_lm_head: bool,
    hidden: &mut Vec<Vec<f32>>,
) -> Result<()> {
    let row = if first_row_lm_head {
        0
    } else {
        plan.scored.start
    };
    let last = plan.len - 1;
    let width = model.dims.hidden();
    run_inner(
        model,
        plan,
        session,
        stream,
        &mut firepit_kernels::sink::Eager(stream),
        row,
        |op, plan| {
            if let PrefillOp::Kernel(Op::Add {
                delta: Slot::MoeDelta,
                out,
                ..
            }) = op
            {
                let values: Vec<f32> = plan.buffers[out.index()]
                    .dup()
                    .to_host_vec()
                    .sync_on(stream)?;
                hidden.push(values[last * width..(last + 1) * width].to_vec());
            }
            Ok(())
        },
    )
}

#[cfg(feature = "testing")]
pub(crate) fn run_routing_observed(
    model: &Model,
    plan: &mut PrefillPlan,
    session: &mut Session,
    stream: &Stream,
) -> Result<Vec<RoutingLayerTrace>> {
    let mut layers = Vec::<RoutingLayerTrace>::new();
    let last_row = plan.scored.start;
    let top_k = model.dims.moe().top_k();
    let hidden = model.dims.hidden();
    run_inner(
        model,
        plan,
        session,
        stream,
        &mut firepit_kernels::sink::Eager(stream),
        last_row,
        |op, plan| {
            match op {
                PrefillOp::TopK { layer } => {
                    let (router, columns) = Slot::projection(model, layer, Projection::Router)?;
                    let router = &plan.buffers[router.index()];
                    let logits: Vec<f32> = router.dup().to_host_vec().sync_on(stream)?;
                    let ids: Vec<i32> = plan.top_ids.dup().to_host_vec().sync_on(stream)?;
                    let width = model.dims.moe().with_shared_expert().width();
                    let mut trace = RoutingLayerTrace::default();
                    trace.router_logits = logits
                        .chunks_exact(row_width(router, plan.rows))
                        .take(plan.len)
                        .map(|row| row[columns.clone()].to_vec())
                        .collect();
                    trace.selected = ids
                        .chunks_exact(width)
                        .take(plan.len)
                        .map(|row| row[..top_k].to_vec())
                        .collect();
                    layers.push(trace);
                }
                PrefillOp::Kernel(Op::Add {
                    delta: Slot::MoeDelta,
                    out,
                    ..
                }) => {
                    let values: Vec<f32> = plan.buffers[out.index()]
                        .dup()
                        .to_host_vec()
                        .sync_on(stream)?;
                    layers.last_mut().ok_or("hidden without router")?.hidden = values
                        .chunks_exact(hidden)
                        .take(plan.len)
                        .map(|row| row.to_vec())
                        .collect();
                }
                _ => {}
            }
            Ok(())
        },
    )?;
    Ok(layers)
}

#[cfg(feature = "testing")]
pub struct IsolatedLayerTrace {
    pub mixer: Vec<f32>,
    pub moe: Vec<f32>,
    pub hidden: Vec<f32>,
    pub selected: Vec<i32>,
}

#[cfg(feature = "testing")]
pub(crate) fn run_isolated_layer(
    model: &Model,
    plan: &mut PrefillPlan,
    session: &mut Session,
    layer: usize,
    input: &[f32],
    stream: &Stream,
) -> Result<IsolatedLayerTrace> {
    let width = model.dims.hidden();
    if layer >= model.dims.layers().len() || input.len() != plan.len * width {
        return Err("invalid isolated layer input".into());
    }
    let start = plan.ops.iter().position(|op| matches!(
        op,
        PrefillOp::Kernel(Op::Norm { layer: found, name: NormName::Input, .. }) if *found == layer
    )).ok_or("isolated layer start missing")?;
    let end = plan
        .ops
        .iter()
        .enumerate()
        .skip(start + 1)
        .find(|(_, op)| {
            matches!(
                op,
                PrefillOp::Kernel(Op::Norm {
                    name: NormName::Input | NormName::Final,
                    ..
                })
            )
        })
        .map(|(index, _)| index)
        .ok_or("isolated layer end missing")?;
    if !matches!(
        plan.ops[end - 1],
        PrefillOp::Kernel(Op::Add {
            delta: Slot::MoeDelta,
            ..
        })
    ) {
        return Err("isolated layer has no final residual".into());
    }
    plan.ops = plan.ops[start..end].to_vec();
    let mut padded = input.to_vec();
    padded.resize(plan.rows * width, 0.0);
    plan.buffers[Slot::Residual0.index()] = api::copy_host_vec_to_device(&Arc::new(padded))
        .reshape(&[plan.rows, width])
        .sync_on(stream)?;
    let mut trace = IsolatedLayerTrace {
        mixer: Vec::new(),
        moe: Vec::new(),
        hidden: Vec::new(),
        selected: Vec::new(),
    };
    let last = plan.len - 1;
    let last_row = |slot: Slot, plan: &PrefillPlan| -> Result<Vec<f32>> {
        let values: Vec<f32> = plan.buffers[slot.index()]
            .dup()
            .to_host_vec()
            .sync_on(stream)?;
        Ok(values[last * width..(last + 1) * width].to_vec())
    };
    run_inner(
        model,
        plan,
        session,
        stream,
        &mut firepit_kernels::sink::Eager(stream),
        last,
        |op, plan| {
            match op {
                PrefillOp::TopK { .. } => {
                    let width = model.dims.moe().with_shared_expert().width();
                    let ids: Vec<i32> = plan.top_ids.dup().to_host_vec().sync_on(stream)?;
                    trace.selected =
                        ids[last * width..last * width + model.dims.moe().top_k()].to_vec();
                }
                PrefillOp::Kernel(Op::Add {
                    delta: Slot::Delta, ..
                }) => {
                    trace.mixer = last_row(Slot::Delta, plan)?;
                }
                PrefillOp::Kernel(Op::Add {
                    delta: Slot::MoeDelta,
                    ..
                }) => {
                    trace.moe = last_row(Slot::MoeDelta, plan)?;
                    trace.hidden = last_row(Slot::Residual0, plan)?;
                }
                _ => {}
            }
            Ok(())
        },
    )?;
    Ok(trace)
}

fn run_inner(
    model: &Model,
    plan: &mut PrefillPlan,
    session: &mut Session,
    stream: &Stream,
    sink: &mut impl Sink,
    lm_head_row: usize,
    mut after_op: impl FnMut(PrefillOp, &PrefillPlan) -> Result<()>,
) -> Result<()> {
    let d = &model.dims;
    let m = d.moe();
    let rows = plan.rows;
    let len = plan.len;
    for (index, op) in plan.ops.clone().into_iter().enumerate() {
        let result = sink.span(op.kind(), |sink| -> Result<()> {
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
                    let selected = match name {
                        NormName::Final if plan.scored.len() == 1 => lm_head_row..lm_head_row + 1,
                        NormName::Final => plan.scored.outer(),
                        _ => 0..rows,
                    };
                    let (source, target) = input_output(&mut plan.buffers, input, out);
                    rms_norm(model, layer, name, source, target, rows, selected, sink)?;
                }
                PrefillOp::Kernel(Op::Gemv {
                    layer,
                    name,
                    input,
                    out,
                }) => {
                    let (source, target) = input_output(&mut plan.buffers, input, out);
                    let w = weight(model, layer, name);
                    let cols = row_width(source, rows);
                    let output_rows = w.rows();
                    if matches!(name, WeightName::LmHead) {
                        let shaped = source.view(&[rows, cols])?;
                        if plan.scored.len() == 1 {
                            let selected =
                                shaped.slice(&[lm_head_row..lm_head_row + 1, 0..cols])?;
                            let x = selected.view(&[
                                1,
                                cols / firepit_kernels::weights::Q8_BLOCK,
                                firepit_kernels::weights::Q8_BLOCK,
                            ])?;
                            let tile = weight_gemv_tile(w, output_rows, cols, 1);
                            gemv(w, &x, target.partition([1, tile.rows]), tile).submit(sink)?;
                        } else {
                            let selected = shaped.slice(&[plan.scored.outer(), 0..cols])?;
                            let tile = gemm_tile(cols, output_rows, false);
                            match w {
                                Weight::Dense(dense) => {
                                    gemm(dense, &selected, target.partition([tile.m, tile.n]), tile)
                                        .submit(sink)?
                                }
                                Weight::Fp8(fp8) => fp8_gemm(
                                    fp8,
                                    &selected,
                                    target.partition([tile.m, tile.n]),
                                    tile,
                                )
                                .submit(sink)?,
                                Weight::Nvfp4(nvfp4) => nvfp4_gemm(
                                    nvfp4,
                                    &selected,
                                    target.partition([tile.m, tile.n]),
                                    tile,
                                )
                                .submit(sink)?,
                                Weight::Q8(_) => return Err("Q8 prefill is unsupported".into()),
                            }
                        }
                    } else {
                        let x = source.view(&[rows, cols])?;
                        let tile = gemm_tile(cols, output_rows, false);
                        match w {
                            Weight::Dense(dense) => {
                                gemm(dense, &x, target.partition([tile.m, tile.n]), tile)
                                    .submit(sink)?
                            }
                            Weight::Fp8(fp8) => {
                                fp8_gemm(fp8, &x, target.partition([tile.m, tile.n]), tile)
                                    .submit(sink)?
                            }
                            Weight::Nvfp4(nvfp4) => {
                                nvfp4_gemm(nvfp4, &x, target.partition([tile.m, tile.n]), tile)
                                    .submit(sink)?
                            }
                            Weight::Q8(_) => return Err("Q8 prefill is unsupported".into()),
                        }
                    }
                }
                PrefillOp::Kernel(Op::GdnGate { layer }) => {
                    gdn_gate(model, layer, &mut plan.buffers, rows, sink)?
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
                PrefillOp::Kernel(Op::OutputGate { layer }) => {
                    output_gate(model, layer, &mut plan.buffers, rows, sink)?
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
                    let x = source.slice(&[0..count, 0..2 * m.intermediate()])?;
                    let y = target.slice_mut(&[0..count, 0..m.intermediate()])?;
                    sink.run(
                        swiglu::swiglu(y.partition([1, m.intermediate()]), &x)
                            .generics(vec![m.intermediate().to_string()]),
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
                _ => run_special(model, plan, session, sink, op, stream)?,
            }
            Ok(())
        });
        if let Err(error) = result {
            return Err(format!("prefill op {index} {op:?}: {error}").into());
        }
        after_op(op, plan)?;
    }
    Ok(())
}

fn run_special(
    model: &Model,
    plan: &mut PrefillPlan,
    session: &mut Session,
    sink: &mut impl Sink,
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
            let (qkv, columns) = Slot::projection(model, layer, Projection::GdnQkv)?;
            let (source, target) = input_output(&mut plan.buffers, qkv, Slot::Conv);
            let x = source.slice(&[0..len, columns])?;
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
                )
                .generics({
                    let mut generics = vec![len.next_power_of_two().to_string()];
                    generics.extend(g.conv_generics(plan.conv_tile));
                    generics
                }),
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
            let (a_slot, a_columns) = Slot::projection(model, layer, Projection::GdnA)?;
            let (b_slot, b_columns) = Slot::projection(model, layer, Projection::GdnB)?;
            let av = plan.buffers[a_slot.index()].slice(&[0..rows, a_columns])?;
            let bv = plan.buffers[b_slot.index()].slice(&[0..rows, b_columns])?;
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
            let (v, columns) = Slot::projection(model, layer, Projection::AttnV)?;
            let v = &plan.buffers[v.index()];
            let v_shaped = v.view(&[rows, row_width(v, rows) / a.head_dim(), a.head_dim()])?;
            let keys = k_shaped.slice(&[0..len, 0..a.kv_heads(), 0..a.head_dim()])?;
            let values =
                v_shaped.slice(&[0..len, heads(&columns, a.head_dim())?, 0..a.head_dim()])?;
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
        PrefillOp::TopK { layer } => {
            let (router, router_columns) = Slot::projection(model, layer, Projection::Router)?;
            let (gate, gate_columns) = Slot::projection(model, layer, Projection::SharedGate)?;
            let logits = plan.buffers[router.index()].slice(&[0..len, router_columns])?;
            let gate = plan.buffers[gate.index()].slice(&[0..len, gate_columns])?;
            let width = m.with_shared_expert().width();
            sink.run(
                moe::top_k_prefill(
                    &logits,
                    &gate,
                    (&mut plan.top_ids).partition([1, width]),
                    (&mut plan.top_weights).partition([1, width]),
                    m.experts() as i32,
                    m.top_k() as i32,
                )
                .generics(m.top_k_generics()),
            )?;
        }
        PrefillOp::HostSort => {
            let selected = m.with_shared_expert();
            let ids = plan.top_ids.dup().to_host_vec().sync_on(stream)?;
            let weights = plan.top_weights.dup().to_host_vec().sync_on(stream)?;
            let ids: Vec<i32> = ids
                .chunks_exact(selected.width())
                .flat_map(|row| row[..selected.top_k()].iter().copied())
                .collect();
            let weights: Vec<f32> = weights
                .chunks_exact(selected.width())
                .flat_map(|row| row[..selected.top_k()].iter().copied())
                .collect();
            let layout = MoePrefillPlan::parse(
                &ids,
                &weights,
                &selected,
                DEFAULT_GROUPED_GEMM_TILE.m,
                stream,
            )?;
            let count = layout.layout().padded_rows();
            if count > plan.moe_sorted.shape()[0] as usize {
                return Err("MoE padded rows exceed prefill buffer capacity".into());
            }
            plan.moe = Some(layout);
        }
        PrefillOp::Gather => {
            let layout = plan.moe.as_ref().ok_or("missing MoE layout")?;
            let count = layout.layout().padded_rows();
            let x = plan.buffers[Slot::Normed.index()].slice(&[0..len, 0..d.hidden()])?;
            let sorted = plan.moe_sorted.slice_mut(&[0..count, 0..d.hidden()])?;
            sink.run(moe::gather_prefill_view(
                sorted.partition([1, DEFAULT_GROUPED_GEMM_TILE.k]),
                &x,
                layout,
                DEFAULT_GROUPED_GEMM_TILE.k,
            ))?;
        }
        PrefillOp::ExpertGateUp { layer } => {
            let layout = plan.moe.as_ref().ok_or("missing MoE layout")?;
            let count = layout.layout().padded_rows();
            let x = plan.moe_sorted.slice(&[0..count, 0..d.hidden()])?;
            let target = &mut plan.buffers[Slot::ExpertUp.index()];
            let y = target.slice_mut(&[0..count, 0..2 * m.intermediate()])?;
            let tile = gemm_tile(d.hidden(), 2 * m.intermediate(), true);
            match &model.layers[layer].moe.gate_up {
                Weight::Dense(w) => grouped_gemm_view(
                    w,
                    &x,
                    y.partition([tile.m, tile.n]),
                    layout,
                    2 * m.intermediate(),
                    tile,
                )
                .submit(sink)?,
                Weight::Fp8(w) => fp8_grouped_gemm_view(
                    w,
                    &x,
                    y.partition([tile.m, tile.n]),
                    layout,
                    2 * m.intermediate(),
                    tile,
                )
                .submit(sink)?,
                Weight::Nvfp4(w) => nvfp4_grouped_gemm_view(
                    w,
                    &x,
                    y.partition([tile.m, tile.n]),
                    layout,
                    2 * m.intermediate(),
                    tile,
                )
                .submit(sink)?,
                Weight::Q8(_) => return Err("Q8 expert prefill is unsupported".into()),
            }
        }
        PrefillOp::ExpertDown { layer } => {
            let layout = plan.moe.as_ref().ok_or("missing MoE layout")?;
            let (source, target) =
                input_output(&mut plan.buffers, Slot::ExpertAct, Slot::ExpertOut);
            let count = layout.layout().padded_rows();
            let x = source.slice(&[0..count, 0..m.intermediate()])?;
            let y = target.slice_mut(&[0..count, 0..d.hidden()])?;
            let tile = gemm_tile(m.intermediate(), d.hidden(), true);
            match &model.layers[layer].moe.down {
                Weight::Dense(w) => grouped_gemm_view(
                    w,
                    &x,
                    y.partition([tile.m, tile.n]),
                    layout,
                    d.hidden(),
                    tile,
                )
                .submit(sink)?,
                Weight::Fp8(w) => fp8_grouped_gemm_view(
                    w,
                    &x,
                    y.partition([tile.m, tile.n]),
                    layout,
                    d.hidden(),
                    tile,
                )
                .submit(sink)?,
                Weight::Nvfp4(w) => nvfp4_grouped_gemm_view(
                    w,
                    &x,
                    y.partition([tile.m, tile.n]),
                    layout,
                    d.hidden(),
                    tile,
                )
                .submit(sink)?,
                Weight::Q8(_) => return Err("Q8 expert prefill is unsupported".into()),
            }
        }
        PrefillOp::Combine => {
            let layout = plan.moe.as_ref().ok_or("missing MoE layout")?;
            let (routed, target) = input_output(&mut plan.buffers, Slot::ExpertOut, Slot::MoeDelta);
            let routed = routed.slice(&[0..layout.layout().padded_rows(), 0..d.hidden()])?;
            sink.run(moe::combine_prefill_sliced(
                target.partition([1, d.hidden()]),
                &routed,
                layout,
            ))?;
        }
        PrefillOp::Kernel(_) => unreachable!(),
    }
    Ok(())
}
