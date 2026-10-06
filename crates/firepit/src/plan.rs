use crate::checkpoint::Stream;
use crate::model::{Model, Projection};
use crate::{LayerKind, Result};
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::Tensor;
use cutile::tile_kernel::ToHostVecOp;
use firepit_kernels::moe::ExpertSelection;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Residual0,
    Residual1,
    Normed,
    Delta,
    Conv,
    Mixer,
    MixerNorm,
    Gated,
    QNorm,
    KNorm,
    QRot,
    KRot,
    AttnOut,
    AttnGated,
    ExpertUp,
    ExpertAct,
    ExpertOut,
    MoeDelta,
    Logits,
    Projection(usize),
}

impl Slot {
    pub const ALL: [Self; 19] = [
        Self::Residual0,
        Self::Residual1,
        Self::Normed,
        Self::Delta,
        Self::Conv,
        Self::Mixer,
        Self::MixerNorm,
        Self::Gated,
        Self::QNorm,
        Self::KNorm,
        Self::QRot,
        Self::KRot,
        Self::AttnOut,
        Self::AttnGated,
        Self::ExpertUp,
        Self::ExpertAct,
        Self::ExpertOut,
        Self::MoeDelta,
        Self::Logits,
    ];

    pub(crate) fn all(model: &Model) -> impl Iterator<Item = Self> {
        let projections = (0..model.projection_layouts.len()).map(Self::Projection);
        Self::ALL.into_iter().chain(projections)
    }

    pub(crate) fn projection(
        model: &Model,
        layer: usize,
        member: Projection,
    ) -> Result<(Self, Range<usize>)> {
        let (layout, columns) = model.projection(layer, member)?;
        Ok((Self::Projection(layout), columns))
    }

    pub(crate) fn projection_heads(
        model: &Model,
        layer: usize,
        member: Projection,
    ) -> Result<(Self, Range<usize>)> {
        let (layout, heads) = model.projection_heads(layer, member)?;
        Ok((Self::Projection(layout), heads))
    }

    pub fn index(self) -> usize {
        match self {
            Self::Projection(layout) => Self::ALL.len() + layout,
            fixed => Self::ALL
                .iter()
                .position(|&slot| slot == fixed)
                .expect("every fixed slot is in Slot::ALL"),
        }
    }

    pub(crate) fn shape(self, model: &Model, batch: usize) -> Vec<usize> {
        let d = &model.dims;
        let g = d.gdn();
        let a = d.attention();
        let m = d.moe();
        if let Self::Projection(layout) = self {
            return vec![batch, model.projection_layouts[layout].rows()];
        }
        if self == Self::Conv {
            return vec![batch * g.conv_dim()];
        }
        if self == Self::Mixer {
            return vec![batch * g.value_heads(), 1, g.value_head_dim()];
        }
        if self == Self::AttnOut {
            return vec![batch * a.q_heads(), a.head_dim()];
        }
        if self == Self::QRot {
            return vec![batch * a.q_heads(), a.head_dim()];
        }
        if self == Self::KRot {
            return vec![batch * a.kv_heads(), a.head_dim()];
        }
        let width = match self {
            Self::Residual0 | Self::Residual1 | Self::Normed | Self::Delta | Self::MoeDelta => {
                d.hidden()
            }
            Self::Conv => g.conv_dim(),
            Self::Mixer | Self::MixerNorm | Self::Gated => g.value_dim(),
            Self::KNorm | Self::KRot => a.kv_heads() * a.head_dim(),
            Self::QNorm | Self::QRot | Self::AttnOut | Self::AttnGated => {
                a.q_heads() * a.head_dim()
            }
            Self::ExpertUp => 2 * m.intermediate(),
            Self::ExpertAct => m.intermediate(),
            Self::ExpertOut => d.hidden(),
            Self::Logits => d.vocab(),
            Self::Projection(_) => unreachable!(),
        };
        let rows = if matches!(self, Self::ExpertUp | Self::ExpertAct | Self::ExpertOut) {
            batch * m.with_shared_expert().top_k()
        } else {
            batch
        };
        vec![rows, width]
    }
}

#[derive(Clone, Copy, Debug)]
pub enum NormName {
    Input,
    Post,
    Final,
    Q,
    K,
    GdnGated,
}

#[derive(Clone, Copy, Debug)]
pub enum WeightName {
    MixerProjection(usize),
    GdnOut,
    AttnOut,
    MoeProjection(usize),
    LmHead,
}

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Embed {
        out: Slot,
    },
    Norm {
        layer: usize,
        name: NormName,
        input: Slot,
        out: Slot,
    },
    Gemv {
        layer: usize,
        name: WeightName,
        input: Slot,
        out: Slot,
    },
    GdnConv {
        layer: usize,
    },
    GdnDelta {
        layer: usize,
    },
    GdnGate {
        layer: usize,
    },
    AttnRope {
        query: bool,
    },
    CacheAppend {
        layer: usize,
    },
    OnlineAttention {
        layer: usize,
    },
    OutputGate {
        layer: usize,
    },
    TopK {
        layer: usize,
    },
    ExpertGateUp {
        layer: usize,
    },
    ExpertSwiglu,
    ExpertDown {
        layer: usize,
    },
    Combine {
        layer: usize,
    },
    Add {
        input: Slot,
        delta: Slot,
        out: Slot,
    },
}

impl Op {
    pub(crate) fn kind(self) -> &'static str {
        match self {
            Self::Embed { .. } => "embedding",
            Self::Norm { .. } => "norm",
            Self::Gemv {
                name: WeightName::LmHead,
                ..
            } => "lm_head",
            Self::Gemv { .. } => "gemv",
            Self::GdnConv { .. } => "gdn_conv",
            Self::GdnDelta { .. } => "gdn_delta",
            Self::GdnGate { .. } => "gdn_gate",
            Self::AttnRope { .. } => "rope",
            Self::CacheAppend { .. } => "cache_append",
            Self::OnlineAttention { .. } => "online_attention",
            Self::OutputGate { .. } => "output_gate",
            Self::TopK { .. } => "moe_top_k",
            Self::ExpertGateUp { .. } | Self::ExpertDown { .. } => "moe_expert_gemv",
            Self::ExpertSwiglu => "swiglu",
            Self::Combine { .. } => "moe_combine",
            Self::Add { .. } => "add",
        }
    }
}

pub struct StepPlan {
    pub ops: Vec<Op>,
    pub batch: usize,
    pub buffers: Vec<Tensor<f32>>,
    pub tokens: Tensor<i32>,
    pub positions: Vec<Tensor<i32>>,
    pub cos: Tensor<f32>,
    pub sin: Tensor<f32>,
    pub selections: Vec<Vec<ExpertSelection>>,
    pub graph: Option<CudaGraph<()>>,
    pub gdn_value_tile: usize,
    pub gdn_conv_tile: usize,
}

impl StepPlan {
    pub(crate) fn ops(model: &Model) -> Result<Vec<Op>> {
        let mut ops = vec![Op::Embed {
            out: Slot::Residual0,
        }];
        let mut residual = Slot::Residual0;
        for (layer, kind) in model.dims.layers().iter().enumerate() {
            let weights = &model.layers[layer];
            ops.push(Op::Norm {
                layer,
                name: NormName::Input,
                input: residual,
                out: Slot::Normed,
            });
            for (group, projection) in weights.mixer.projections().iter().enumerate() {
                ops.push(Op::Gemv {
                    layer,
                    name: WeightName::MixerProjection(group),
                    input: Slot::Normed,
                    out: Slot::Projection(projection.layout),
                });
            }
            match kind {
                LayerKind::Gdn => {
                    ops.push(Op::GdnConv { layer });
                    ops.push(Op::GdnDelta { layer });
                    ops.push(Op::Norm {
                        layer,
                        name: NormName::GdnGated,
                        input: Slot::Mixer,
                        out: Slot::MixerNorm,
                    });
                    ops.push(Op::GdnGate { layer });
                    ops.push(Op::Gemv {
                        layer,
                        name: WeightName::GdnOut,
                        input: Slot::Gated,
                        out: Slot::Delta,
                    });
                }
                LayerKind::Attention => {
                    ops.push(Op::Norm {
                        layer,
                        name: NormName::Q,
                        input: Slot::projection(model, layer, Projection::AttnQGate)?.0,
                        out: Slot::QNorm,
                    });
                    ops.push(Op::Norm {
                        layer,
                        name: NormName::K,
                        input: Slot::projection(model, layer, Projection::AttnK)?.0,
                        out: Slot::KNorm,
                    });
                    ops.push(Op::AttnRope { query: true });
                    ops.push(Op::AttnRope { query: false });
                    ops.push(Op::CacheAppend { layer });
                    ops.push(Op::OnlineAttention { layer });
                    ops.push(Op::OutputGate { layer });
                    ops.push(Op::Gemv {
                        layer,
                        name: WeightName::AttnOut,
                        input: Slot::AttnGated,
                        out: Slot::Delta,
                    });
                }
            }
            let next = if residual == Slot::Residual0 {
                Slot::Residual1
            } else {
                Slot::Residual0
            };
            ops.push(Op::Add {
                input: residual,
                delta: Slot::Delta,
                out: next,
            });
            ops.push(Op::Norm {
                layer,
                name: NormName::Post,
                input: next,
                out: Slot::Normed,
            });
            for (group, projection) in weights.moe.projections.iter().enumerate() {
                ops.push(Op::Gemv {
                    layer,
                    name: WeightName::MoeProjection(group),
                    input: Slot::Normed,
                    out: Slot::Projection(projection.layout),
                });
            }
            ops.push(Op::TopK { layer });
            ops.push(Op::ExpertGateUp { layer });
            ops.push(Op::ExpertSwiglu);
            ops.push(Op::ExpertDown { layer });
            ops.push(Op::Combine { layer });
            let after_moe = if next == Slot::Residual0 {
                Slot::Residual1
            } else {
                Slot::Residual0
            };
            ops.push(Op::Add {
                input: next,
                delta: Slot::MoeDelta,
                out: after_moe,
            });
            residual = after_moe;
        }
        ops.push(Op::Norm {
            layer: 0,
            name: NormName::Final,
            input: residual,
            out: Slot::Normed,
        });
        ops.push(Op::Gemv {
            layer: 0,
            name: WeightName::LmHead,
            input: Slot::Normed,
            out: Slot::Logits,
        });
        Ok(ops)
    }

    pub fn new(
        model: &Model,
        batch: usize,
        gdn_value_tile: usize,
        gdn_conv_tile: usize,
        stream: &Stream,
    ) -> Result<Self> {
        let ops = Self::ops(model)?;
        let buffers = Slot::all(model)
            .map(|slot| api::zeros::<f32>(&slot.shape(model, batch)).sync_on(stream))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let tokens = api::zeros::<i32>(&[batch]).sync_on(stream)?;
        let positions = (0..batch)
            .map(|_| api::zeros::<i32>(&[1]).sync_on(stream))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let d = model.dims.attention().head_dim();
        let cos = api::zeros::<f32>(&[batch, d]).sync_on(stream)?;
        let sin = api::zeros::<f32>(&[batch, d]).sync_on(stream)?;
        let selections = (0..model.layers.len())
            .map(|_| {
                (0..batch)
                    .map(|_| ExpertSelection::empty(&model.dims.moe().with_shared_expert(), stream))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            ops,
            batch,
            buffers,
            tokens,
            positions,
            cos,
            sin,
            selections,
            graph: None,
            gdn_value_tile,
            gdn_conv_tile,
        })
    }

    pub fn logits(&self, stream: &Stream) -> Result<Vec<f32>> {
        Ok(self.buffers[Slot::Logits.index()]
            .dup()
            .to_host_vec()
            .sync_on(stream)?)
    }
}
