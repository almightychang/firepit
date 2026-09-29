use crate::model::{NormRef, RowTile, Stream, WeightRef};
use crate::{Model, Result};
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::Tensor;
use cutile::tile_kernel::ToHostVecOp;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Parity {
    Even,
    Odd,
}

impl Parity {
    fn flip(self) -> Self {
        match self {
            Self::Even => Self::Odd,
            Self::Odd => Self::Even,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Slot {
    Residual(Parity),
    Normed,
    SessionOut,
    Projected,
    Logits,
}

impl Slot {
    const ALL: [Slot; 6] = [
        Self::Residual(Parity::Even),
        Self::Residual(Parity::Odd),
        Self::Normed,
        Self::SessionOut,
        Self::Projected,
        Self::Logits,
    ];

    pub(crate) fn index(self) -> usize {
        Self::ALL.iter().position(|&slot| slot == self).unwrap()
    }

    fn shape(self, model: &Model, batch: usize) -> Vec<usize> {
        match self {
            Self::SessionOut => model.dims().session_output_shape(batch),
            Self::Logits => vec![batch, model.dims().vocab()],
            _ => vec![batch, model.dims().hidden()],
        }
    }
}

pub(crate) enum Op {
    Embed {
        out: Slot,
    },
    RmsNorm {
        norm: NormRef,
        input: Slot,
        out: Slot,
    },
    Add {
        residual: Slot,
        delta: Slot,
        out: Slot,
    },
    Gemv {
        weight: WeightRef,
        input: Slot,
        out: Slot,
    },
    Session {
        layer: usize,
        input: Slot,
        out: Slot,
    },
}

pub(crate) struct StepPlan {
    pub ops: Vec<Op>,
    pub(crate) batch: usize,
    pub(crate) buffers: Vec<Tensor<f32>>,
    pub(crate) tokens: Tensor<i32>,
    pub(crate) positions: Vec<Tensor<i32>>,
    pub(crate) graph: Option<CudaGraph<()>>,
    pub(crate) row: RowTile,
}

impl StepPlan {
    pub(crate) fn new(model: &Model, batch: usize, row: RowTile, stream: &Stream) -> Result<Self> {
        let c = &model.dims();
        let mut ops = vec![Op::Embed {
            out: Slot::Residual(Parity::Even),
        }];
        let mut main = Parity::Even;
        for (layer, _) in c.layers().iter().enumerate() {
            ops.push(Op::RmsNorm {
                norm: NormRef::Layer(layer),
                input: Slot::Residual(main),
                out: Slot::Normed,
            });
            ops.push(Op::Gemv {
                weight: WeightRef::LayerInput(layer),
                input: Slot::Normed,
                out: Slot::Projected,
            });
            ops.push(Op::Session {
                layer,
                input: Slot::Projected,
                out: Slot::SessionOut,
            });
            ops.push(Op::Gemv {
                weight: WeightRef::LayerOutput(layer),
                input: Slot::SessionOut,
                out: Slot::Projected,
            });
            ops.push(Op::Add {
                residual: Slot::Residual(main),
                delta: Slot::Projected,
                out: Slot::Residual(main.flip()),
            });
            main = main.flip();
        }
        ops.push(Op::RmsNorm {
            norm: NormRef::Final,
            input: Slot::Residual(main),
            out: Slot::Normed,
        });
        ops.push(Op::Gemv {
            weight: WeightRef::OutputHead,
            input: Slot::Normed,
            out: Slot::Logits,
        });
        let buffers = Slot::ALL
            .iter()
            .map(|slot| api::zeros::<f32>(&slot.shape(model, batch)).sync_on(stream))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let tokens = api::zeros::<i32>(&[batch]).sync_on(stream)?;
        let positions = (0..batch)
            .map(|_| api::zeros::<i32>(&[1]).sync_on(stream))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            ops,
            batch,
            buffers,
            tokens,
            positions,
            graph: None,
            row,
        })
    }

    pub(crate) fn logits(&self, stream: &Stream) -> Result<Vec<f32>> {
        Ok(self.buffers[Slot::Logits.index()]
            .dup()
            .to_host_vec()
            .sync_on(stream)?)
    }
}
