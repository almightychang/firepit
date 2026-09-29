use crate::model::Stream;
use crate::{Model, Result};
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::DeviceOp;
use cutile::api;
use cutile::tensor::Tensor;
use cutile::tile_kernel::ToHostVecOp;

#[derive(Clone, Copy)]
pub enum SessionKind {
    Gdn,
    Attention,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Parity {
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
pub enum Slot {
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

    fn shape(self, model: &Model, grid: &SessionGrid, batch: usize) -> Vec<usize> {
        match self {
            Self::SessionOut => grid.output_shape(batch),
            Self::Logits => vec![batch, model.config.vocab],
            _ => vec![batch, model.config.hidden],
        }
    }
}

pub enum Op {
    Embed {
        out: Slot,
    },
    RmsNorm {
        input: Slot,
        out: Slot,
    },
    Add {
        residual: Slot,
        delta: Slot,
        out: Slot,
    },
    Gemv {
        weight: usize,
        input: Slot,
        out: Slot,
    },
    Session {
        layer: usize,
        kind: SessionKind,
        input: Slot,
        out: Slot,
    },
}

pub(crate) struct SessionGrid {
    pub row: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
}

impl SessionGrid {
    pub fn output_shape(&self, batch: usize) -> Vec<usize> {
        vec![batch * self.heads, self.head_dim]
    }
    pub fn output_tile(&self) -> [usize; 2] {
        [1, self.row]
    }
    pub fn recurrent_tile(&self) -> [usize; 3] {
        [1, self.row, self.head_dim]
    }
    pub fn cache_tile(&self) -> [usize; 3] {
        [self.max_seq, 1, self.row]
    }
}

pub struct StepPlan {
    pub ops: Vec<Op>,
    pub(crate) buffers: Vec<Tensor<f32>>,
    pub(crate) tokens: Tensor<i32>,
    pub(crate) positions: Vec<Tensor<i32>>,
    pub(crate) graph: Option<CudaGraph<()>>,
    pub(crate) grid: SessionGrid,
    pub(crate) instances: Vec<u64>,
}

impl StepPlan {
    pub(crate) fn new(model: &Model, batch: usize, row: usize, stream: &Stream) -> Result<Self> {
        let c = &model.config;
        let grid = SessionGrid {
            row,
            heads: c.heads,
            head_dim: c.head_dim,
            max_seq: c.max_seq,
        };
        let mut ops = vec![Op::Embed {
            out: Slot::Residual(Parity::Even),
        }];
        let mut main = Parity::Even;
        for layer in 0..(c.gdn_layers + c.attention_layers) {
            ops.push(Op::RmsNorm {
                input: Slot::Residual(main),
                out: Slot::Normed,
            });
            ops.push(Op::Gemv {
                weight: layer * 2,
                input: Slot::Normed,
                out: Slot::Projected,
            });
            let kind = if layer < c.gdn_layers {
                SessionKind::Gdn
            } else {
                SessionKind::Attention
            };
            ops.push(Op::Session {
                layer,
                kind,
                input: Slot::Projected,
                out: Slot::SessionOut,
            });
            ops.push(Op::Gemv {
                weight: layer * 2 + 1,
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
            input: Slot::Residual(main),
            out: Slot::Normed,
        });
        ops.push(Op::Gemv {
            weight: model.weights.len() - 1,
            input: Slot::Normed,
            out: Slot::Logits,
        });
        let buffers = Slot::ALL
            .iter()
            .map(|slot| api::zeros::<f32>(&slot.shape(model, &grid, batch)).sync_on(stream))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let tokens = api::zeros::<i32>(&[batch]).sync_on(stream)?;
        let positions = (0..batch)
            .map(|_| api::zeros::<i32>(&[1]).sync_on(stream))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            ops,
            buffers,
            tokens,
            positions,
            graph: None,
            grid,
            instances: Vec::new(),
        })
    }

    pub(crate) fn logits(&self, stream: &Stream) -> Result<Vec<f32>> {
        Ok(self.buffers[Slot::Logits.index()]
            .dup()
            .to_host_vec()
            .sync_on(stream)?)
    }
}
