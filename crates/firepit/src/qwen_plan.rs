use crate::checkpoint::Stream;
use crate::qwen_model::QwenModel;
use crate::{LayerKind, Result};
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::Tensor;
use cutile::tile_kernel::ToHostVecOp;
use firepit_kernels::moe::ExpertSelection;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Residual0,
    Residual1,
    Normed,
    Delta,
    Qkv,
    Z,
    B,
    A,
    Conv,
    Mixer,
    MixerNorm,
    Gated,
    QGate,
    K,
    V,
    QNorm,
    KNorm,
    QRot,
    KRot,
    AttnOut,
    AttnGated,
    Router,
    ExpertUp,
    ExpertAct,
    ExpertOut,
    SharedUp,
    SharedAct,
    Shared,
    SharedGate,
    MoeDelta,
    Logits,
}

impl Slot {
    pub const ALL: [Self; 31] = [
        Self::Residual0,
        Self::Residual1,
        Self::Normed,
        Self::Delta,
        Self::Qkv,
        Self::Z,
        Self::B,
        Self::A,
        Self::Conv,
        Self::Mixer,
        Self::MixerNorm,
        Self::Gated,
        Self::QGate,
        Self::K,
        Self::V,
        Self::QNorm,
        Self::KNorm,
        Self::QRot,
        Self::KRot,
        Self::AttnOut,
        Self::AttnGated,
        Self::Router,
        Self::ExpertUp,
        Self::ExpertAct,
        Self::ExpertOut,
        Self::SharedUp,
        Self::SharedAct,
        Self::Shared,
        Self::SharedGate,
        Self::MoeDelta,
        Self::Logits,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    fn shape(self, model: &QwenModel, batch: usize) -> Vec<usize> {
        let d = &model.dims;
        let g = d.gdn();
        let a = d.attention();
        let m = d.moe();
        if self == Self::Conv {
            return vec![batch * g.conv_dim()];
        }
        if self == Self::Mixer {
            return vec![batch * g.value_heads(), 1, g.value_head_dim()];
        }
        if matches!(self, Self::MixerNorm | Self::Gated) {
            return vec![batch * g.value_heads(), g.value_head_dim()];
        }
        if self == Self::AttnOut {
            return vec![batch * a.q_heads(), a.head_dim()];
        }
        if matches!(self, Self::QNorm | Self::AttnGated) {
            return vec![batch * a.q_heads(), a.head_dim()];
        }
        if self == Self::KNorm {
            return vec![batch * a.kv_heads(), a.head_dim()];
        }
        if self == Self::QRot {
            return vec![batch * a.q_heads(), a.head_dim()];
        }
        if self == Self::KRot {
            return vec![batch * a.kv_heads(), a.head_dim()];
        }
        let width = match self {
            Self::Residual0
            | Self::Residual1
            | Self::Normed
            | Self::Delta
            | Self::Shared
            | Self::MoeDelta => d.hidden(),
            Self::Qkv | Self::Conv => g.conv_dim(),
            Self::Z | Self::Mixer | Self::MixerNorm | Self::Gated => g.value_dim(),
            Self::B | Self::A => g.value_heads(),
            Self::QGate => 2 * a.q_heads() * a.head_dim(),
            Self::K | Self::V | Self::KNorm | Self::KRot => a.kv_heads() * a.head_dim(),
            Self::QNorm | Self::QRot | Self::AttnOut | Self::AttnGated => {
                a.q_heads() * a.head_dim()
            }
            Self::Router => m.experts(),
            Self::ExpertUp => 2 * m.intermediate(),
            Self::ExpertAct => m.intermediate(),
            Self::ExpertOut => d.hidden(),
            Self::SharedUp => 2 * m.shared_intermediate(),
            Self::SharedAct => m.shared_intermediate(),
            Self::SharedGate => 1,
            Self::Logits => d.vocab(),
        };
        let rows = if matches!(self, Self::ExpertUp | Self::ExpertAct | Self::ExpertOut) {
            batch * m.top_k()
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
    GdnQkv,
    GdnZ,
    GdnB,
    GdnA,
    GdnOut,
    AttnQGate,
    AttnK,
    AttnV,
    AttnOut,
    Router,
    SharedGateUp,
    SharedDown,
    SharedGate,
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
    GdnGate,
    AttnRope {
        query: bool,
    },
    CacheAppend {
        layer: usize,
    },
    OnlineAttention {
        layer: usize,
    },
    OutputGate,
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
    SharedSwiglu,
    Combine {
        layer: usize,
    },
    Add {
        input: Slot,
        delta: Slot,
        out: Slot,
    },
}

pub struct QwenPlan {
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
}

impl QwenPlan {
    pub fn new(
        model: &QwenModel,
        batch: usize,
        gdn_value_tile: usize,
        stream: &Stream,
    ) -> Result<Self> {
        let mut ops = vec![Op::Embed {
            out: Slot::Residual0,
        }];
        let mut residual = Slot::Residual0;
        for (layer, kind) in model.dims.layers().iter().enumerate() {
            ops.push(Op::Norm {
                layer,
                name: NormName::Input,
                input: residual,
                out: Slot::Normed,
            });
            match kind {
                LayerKind::Gdn => {
                    for (name, out) in [
                        (WeightName::GdnQkv, Slot::Qkv),
                        (WeightName::GdnZ, Slot::Z),
                        (WeightName::GdnB, Slot::B),
                        (WeightName::GdnA, Slot::A),
                    ] {
                        ops.push(Op::Gemv {
                            layer,
                            name,
                            input: Slot::Normed,
                            out,
                        });
                    }
                    ops.push(Op::GdnConv { layer });
                    ops.push(Op::GdnDelta { layer });
                    ops.push(Op::Norm {
                        layer,
                        name: NormName::GdnGated,
                        input: Slot::Mixer,
                        out: Slot::MixerNorm,
                    });
                    ops.push(Op::GdnGate);
                    ops.push(Op::Gemv {
                        layer,
                        name: WeightName::GdnOut,
                        input: Slot::Gated,
                        out: Slot::Delta,
                    });
                }
                LayerKind::Attention => {
                    for (name, out) in [
                        (WeightName::AttnQGate, Slot::QGate),
                        (WeightName::AttnK, Slot::K),
                        (WeightName::AttnV, Slot::V),
                    ] {
                        ops.push(Op::Gemv {
                            layer,
                            name,
                            input: Slot::Normed,
                            out,
                        });
                    }
                    ops.push(Op::Norm {
                        layer,
                        name: NormName::Q,
                        input: Slot::QGate,
                        out: Slot::QNorm,
                    });
                    ops.push(Op::Norm {
                        layer,
                        name: NormName::K,
                        input: Slot::K,
                        out: Slot::KNorm,
                    });
                    ops.push(Op::AttnRope { query: true });
                    ops.push(Op::AttnRope { query: false });
                    ops.push(Op::CacheAppend { layer });
                    ops.push(Op::OnlineAttention { layer });
                    ops.push(Op::OutputGate);
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
            ops.push(Op::Gemv {
                layer,
                name: WeightName::Router,
                input: Slot::Normed,
                out: Slot::Router,
            });
            ops.push(Op::TopK { layer });
            ops.push(Op::ExpertGateUp { layer });
            ops.push(Op::ExpertSwiglu);
            ops.push(Op::ExpertDown { layer });
            ops.push(Op::Gemv {
                layer,
                name: WeightName::SharedGateUp,
                input: Slot::Normed,
                out: Slot::SharedUp,
            });
            ops.push(Op::SharedSwiglu);
            ops.push(Op::Gemv {
                layer,
                name: WeightName::SharedDown,
                input: Slot::SharedAct,
                out: Slot::Shared,
            });
            ops.push(Op::Gemv {
                layer,
                name: WeightName::SharedGate,
                input: Slot::Normed,
                out: Slot::SharedGate,
            });
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
        let buffers = Slot::ALL
            .iter()
            .map(|&slot| api::zeros::<f32>(&slot.shape(model, batch)).sync_on(stream))
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
                    .map(|_| ExpertSelection::empty(&model.dims.moe(), stream))
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
        })
    }

    pub fn logits(&self, stream: &Stream) -> Result<Vec<f32>> {
        Ok(self.buffers[Slot::Logits.index()]
            .dup()
            .to_host_vec()
            .sync_on(stream)?)
    }
}
