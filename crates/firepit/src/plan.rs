use crate::checkpoint::Stream;
use crate::model::{Model, Projection};
use crate::session::Session;
use crate::store::{Ctx, DecodeStore};
use crate::{LayerKind, Result};
use cuda_async::cuda_graph::CudaGraph;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightName {
    MixerProjection(usize),
    GdnOut,
    AttnOut,
    MoeProjection(usize),
    ExpertGateUp,
    ExpertDown,
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
    pub(crate) fn layer(self) -> Option<usize> {
        match self {
            Self::Norm {
                name: NormName::Final,
                ..
            }
            | Self::Gemv {
                name: WeightName::LmHead,
                ..
            }
            | Self::Embed { .. }
            | Self::AttnRope { .. }
            | Self::ExpertSwiglu
            | Self::Add { .. } => None,
            Self::Norm { layer, .. }
            | Self::Gemv { layer, .. }
            | Self::GdnConv { layer }
            | Self::GdnDelta { layer }
            | Self::GdnGate { layer }
            | Self::CacheAppend { layer }
            | Self::OnlineAttention { layer }
            | Self::OutputGate { layer }
            | Self::TopK { layer }
            | Self::ExpertGateUp { layer }
            | Self::ExpertDown { layer }
            | Self::Combine { layer } => Some(layer),
        }
    }

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

#[derive(Clone, Copy, Debug)]
pub struct GdnTiles {
    pub value: usize,
    pub conv: usize,
}

pub struct StepPlan {
    pub ops: Vec<Op>,
    pub(crate) store: DecodeStore,
    pub graph: Option<CudaGraph<()>>,
    pub tiles: GdnTiles,
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

    pub fn new(model: &Model, batch: usize, tiles: GdnTiles, stream: &Stream) -> Result<Self> {
        Ok(Self {
            ops: Self::ops(model)?,
            store: DecodeStore::new(model, batch, stream)?,
            graph: None,
            tiles,
        })
    }

    pub(crate) fn context<'a, 's, 'm>(
        &'a mut self,
        model: &'m Model,
        sessions: &'a mut [&'s mut Session],
    ) -> (&'a [Op], GdnTiles, Ctx<'a, 's, 'm>) {
        (
            &self.ops,
            self.tiles,
            Ctx::new(model, &mut self.store, sessions),
        )
    }
}
