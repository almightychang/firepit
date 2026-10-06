use crate::Result;
use crate::checkpoint::Stream;
use crate::model::{MixerWeights, Model, Projection};
use crate::plan::{Slot, WeightName};
use crate::session::{LayerState, Session};
use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::Tensor;
use cutile::tile_kernel::ToHostVecOp;
use firepit_kernels::moe::ExpertSelection;
use firepit_kernels::weights::Weight;
use std::cell::RefCell;
use std::ops::Range;
use std::sync::Arc;

pub(crate) struct DecodeStore {
    buffers: Vec<Tensor<f32>>,
    tokens: Tensor<i32>,
    positions: Vec<Tensor<i32>>,
    cos: Tensor<f32>,
    sin: Tensor<f32>,
    selections: Vec<Vec<ExpertSelection>>,
}

impl DecodeStore {
    pub(crate) fn new(model: &Model, batch: usize, stream: &Stream) -> Result<Self> {
        let buffers = Slot::all(model)
            .map(|slot| api::zeros::<f32>(&slot.shape(model, batch)).sync_on(stream))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let positions = (0..batch)
            .map(|_| api::zeros::<i32>(&[1]).sync_on(stream))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let head_dim = model.dims.attention().head_dim();
        let selections = (0..model.layers.len())
            .map(|_| {
                (0..batch)
                    .map(|_| ExpertSelection::empty(&model.dims.moe().with_shared_expert(), stream))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            buffers,
            tokens: api::zeros::<i32>(&[batch]).sync_on(stream)?,
            positions,
            cos: api::zeros::<f32>(&[batch, head_dim]).sync_on(stream)?,
            sin: api::zeros::<f32>(&[batch, head_dim]).sync_on(stream)?,
            selections,
        })
    }

    pub(crate) fn batch(&self) -> usize {
        self.positions.len()
    }

    pub(crate) fn upload_step(
        &mut self,
        tokens: Vec<i32>,
        positions: &[usize],
        cos: Vec<f32>,
        sin: Vec<f32>,
        stream: &Stream,
    ) -> Result<()> {
        let source = api::copy_host_vec_to_device(&Arc::new(tokens)).sync_on(stream)?;
        api::memcpy(&mut self.tokens, &source).sync_on(stream)?;
        for (position, &value) in self.positions.iter_mut().zip(positions) {
            let source =
                api::copy_host_vec_to_device(&Arc::new(vec![value as i32])).sync_on(stream)?;
            api::memcpy(position, &source).sync_on(stream)?;
        }
        let shape = self
            .cos
            .shape()
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        for (target, values) in [(&mut self.cos, cos), (&mut self.sin, sin)] {
            let source = api::copy_host_vec_to_device(&Arc::new(values))
                .reshape(&shape)
                .sync_on(stream)?;
            api::memcpy(target, &source).sync_on(stream)?;
        }
        Ok(())
    }

    pub(crate) fn logits(&self, stream: &Stream) -> Result<Vec<f32>> {
        Ok(self.buffers[Slot::Logits.index()]
            .dup()
            .to_host_vec()
            .sync_on(stream)?)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Read {
    Whole(Slot),
    Member { layer: usize, member: Projection },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpNode {
    pub op: u32,
    pub kind: &'static str,
    pub layer: Option<u16>,
    pub uses: Vec<Use>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Use {
    pub buffer: Buffer,
    pub write: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Buffer {
    Slot(Slot),
    Member { layer: u16, member: Projection },
    Token,
    Position(u16),
    Cos,
    Sin,
    Selection { layer: u16, session: u16 },
    State { layer: u16, session: u16 },
    Matrix { layer: u16, name: WeightName },
}

fn index(value: usize) -> u16 {
    u16::try_from(value).expect("layer and session indices fit u16")
}

impl From<Read> for Buffer {
    fn from(read: Read) -> Self {
        match read {
            Read::Whole(slot) => Self::Slot(slot),
            Read::Member { layer, member } => Self::Member {
                layer: index(layer),
                member,
            },
        }
    }
}

type Log = RefCell<Vec<OpNode>>;

#[derive(Clone, Copy)]
struct Recorder<'c>(Option<&'c Log>);

impl Recorder<'_> {
    fn read(self, buffer: Buffer) {
        self.note(buffer, false);
    }

    fn write(self, buffer: Buffer) {
        self.note(buffer, true);
    }

    fn note(self, buffer: Buffer, write: bool) {
        if let Some(log) = self.0 {
            log.borrow_mut()
                .last_mut()
                .expect("an op begins before it uses anything")
                .uses
                .push(Use { buffer, write });
        }
    }
}

pub(crate) struct View<'a> {
    pub tensor: &'a Tensor<f32>,
    pub columns: Range<usize>,
}

pub(crate) struct Ctx<'a, 's, 'm> {
    model: &'m Model,
    store: &'a mut DecodeStore,
    sessions: &'a mut [&'s mut Session],
    log: Option<Log>,
}

impl<'a, 's, 'm> Ctx<'a, 's, 'm> {
    pub(crate) fn new(
        model: &'m Model,
        store: &'a mut DecodeStore,
        sessions: &'a mut [&'s mut Session],
    ) -> Self {
        Self {
            model,
            store,
            sessions,
            log: None,
        }
    }

    pub(crate) fn record(&mut self) {
        self.log = Some(RefCell::new(Vec::new()));
    }

    pub(crate) fn nodes(self) -> Vec<OpNode> {
        self.log.map(RefCell::into_inner).unwrap_or_default()
    }

    pub(crate) fn begin(&mut self, op: usize, kind: &'static str, layer: Option<usize>) {
        if let Some(log) = &self.log {
            log.borrow_mut().push(OpNode {
                op: u32::try_from(op).expect("op index fits u32"),
                kind,
                layer: layer.map(index),
                uses: Vec::new(),
            });
        }
    }

    pub(crate) fn model(&self) -> &'m Model {
        self.model
    }

    pub(crate) fn batch(&self) -> usize {
        self.store.batch()
    }

    pub(crate) fn sessions(&self) -> usize {
        self.sessions.len()
    }

    pub(crate) fn parts(&mut self) -> Parts<'_, 's, 'm> {
        let store = &mut *self.store;
        let recorder = Recorder(self.log.as_ref());
        Parts {
            buffers: Buffers {
                model: self.model,
                buffers: &mut store.buffers,
                recorder,
            },
            inputs: Inputs {
                tokens: &store.tokens,
                positions: &store.positions,
                cos: &store.cos,
                sin: &store.sin,
                recorder,
            },
            selections: Selections {
                selections: &mut store.selections,
                recorder,
            },
            states: States {
                sessions: &mut *self.sessions,
                recorder,
            },
            weights: Weights {
                model: self.model,
                recorder,
            },
        }
    }

    #[cfg(feature = "testing")]
    pub(crate) fn peek(&self, read: Read) -> Result<View<'_>> {
        let (slot, columns) = resolve(self.model, &self.store.buffers, read)?;
        Ok(View {
            tensor: &self.store.buffers[slot.index()],
            columns,
        })
    }

    #[cfg(feature = "testing")]
    pub(crate) fn peek_selection(&self, layer: usize, session: usize) -> &ExpertSelection {
        &self.store.selections[layer][session]
    }
}

pub(crate) struct Parts<'c, 's, 'm> {
    pub buffers: Buffers<'c, 'm>,
    pub inputs: Inputs<'c>,
    pub selections: Selections<'c>,
    pub states: States<'c, 's>,
    pub weights: Weights<'c, 'm>,
}

fn resolve(model: &Model, buffers: &[Tensor<f32>], read: Read) -> Result<(Slot, Range<usize>)> {
    match read {
        Read::Whole(slot) => {
            let tensor = &buffers[slot.index()];
            let elements: usize = tensor.shape().iter().map(|&x| x as usize).product();
            Ok((slot, 0..elements / tensor.shape()[0] as usize))
        }
        Read::Member { layer, member } => Slot::projection(model, layer, member),
    }
}

pub(crate) struct Buffers<'c, 'm> {
    model: &'m Model,
    buffers: &'c mut Vec<Tensor<f32>>,
    recorder: Recorder<'c>,
}

impl Buffers<'_, '_> {
    pub(crate) fn io<const N: usize>(
        &mut self,
        reads: [Read; N],
        write: Slot,
    ) -> Result<([View<'_>; N], &mut Tensor<f32>)> {
        for read in reads {
            self.recorder.read(read.into());
        }
        self.recorder.write(Buffer::Slot(write));
        let resolved = reads
            .into_iter()
            .map(|read| resolve(self.model, self.buffers, read))
            .collect::<Result<Vec<_>>>()?;
        let output = write.index();
        if let Some((slot, _)) = resolved.iter().find(|(slot, _)| slot.index() == output) {
            return Err(format!("{slot:?} is both read and written").into());
        }
        let (before, at) = self.buffers.split_at_mut(output);
        let (target, after) = at.split_first_mut().expect("written slot exists");
        let mut resolved = resolved.into_iter();
        let views = std::array::from_fn(|_| {
            let (slot, columns) = resolved.next().expect("one resolved read per read");
            let index = slot.index();
            let tensor = if index < output {
                &before[index]
            } else {
                &after[index - output - 1]
            };
            View { tensor, columns }
        });
        Ok((views, target))
    }

    pub(crate) fn read<const N: usize>(&self, reads: [Read; N]) -> Result<[View<'_>; N]> {
        for read in reads {
            self.recorder.read(read.into());
        }
        let resolved = reads
            .into_iter()
            .map(|read| resolve(self.model, self.buffers, read))
            .collect::<Result<Vec<_>>>()?;
        let mut resolved = resolved.into_iter();
        Ok(std::array::from_fn(|_| {
            let (slot, columns) = resolved.next().expect("one resolved read per read");
            View {
                tensor: &self.buffers[slot.index()],
                columns,
            }
        }))
    }
}

pub(crate) struct Inputs<'c> {
    tokens: &'c Tensor<i32>,
    positions: &'c [Tensor<i32>],
    cos: &'c Tensor<f32>,
    sin: &'c Tensor<f32>,
    recorder: Recorder<'c>,
}

impl<'c> Inputs<'c> {
    pub(crate) fn tokens(&self) -> &'c Tensor<i32> {
        self.recorder.read(Buffer::Token);
        self.tokens
    }

    pub(crate) fn position(&self, session: usize) -> &'c Tensor<i32> {
        self.recorder.read(Buffer::Position(index(session)));
        &self.positions[session]
    }

    pub(crate) fn cos(&self) -> &'c Tensor<f32> {
        self.recorder.read(Buffer::Cos);
        self.cos
    }

    pub(crate) fn sin(&self) -> &'c Tensor<f32> {
        self.recorder.read(Buffer::Sin);
        self.sin
    }
}

fn selection(layer: usize, session: usize) -> Buffer {
    Buffer::Selection {
        layer: index(layer),
        session: index(session),
    }
}

pub(crate) struct Selections<'c> {
    selections: &'c mut Vec<Vec<ExpertSelection>>,
    recorder: Recorder<'c>,
}

impl Selections<'_> {
    pub(crate) fn get(&self, layer: usize, session: usize) -> &ExpertSelection {
        self.recorder.read(selection(layer, session));
        &self.selections[layer][session]
    }

    pub(crate) fn get_mut(&mut self, layer: usize, session: usize) -> &mut ExpertSelection {
        self.recorder.write(selection(layer, session));
        &mut self.selections[layer][session]
    }
}

fn state(layer: usize, session: usize) -> Buffer {
    Buffer::State {
        layer: index(layer),
        session: index(session),
    }
}

pub(crate) struct States<'c, 's> {
    sessions: &'c mut [&'s mut Session],
    recorder: Recorder<'c>,
}

impl States<'_, '_> {
    pub(crate) fn gdn(
        &mut self,
        session: usize,
        layer: usize,
    ) -> Result<(&mut Tensor<f32>, &mut Tensor<f32>)> {
        self.recorder.write(state(layer, session));
        match &mut self.sessions[session].state[layer] {
            LayerState::Gdn { conv, recurrent } => Ok((conv, recurrent)),
            LayerState::Attention { .. } => Err(format!("layer {layer} has no GDN state").into()),
        }
    }

    pub(crate) fn attention(
        &mut self,
        session: usize,
        layer: usize,
    ) -> Result<(&mut Tensor<f32>, &mut Tensor<f32>)> {
        self.recorder.write(state(layer, session));
        match &mut self.sessions[session].state[layer] {
            LayerState::Attention { k, v } => Ok((k, v)),
            LayerState::Gdn { .. } => Err(format!("layer {layer} has no attention cache").into()),
        }
    }

    pub(crate) fn attention_ref(
        &self,
        session: usize,
        layer: usize,
    ) -> Result<(&Tensor<f32>, &Tensor<f32>)> {
        self.recorder.read(state(layer, session));
        match &self.sessions[session].state[layer] {
            LayerState::Attention { k, v } => Ok((k, v)),
            LayerState::Gdn { .. } => Err(format!("layer {layer} has no attention cache").into()),
        }
    }
}

pub(crate) struct Weights<'c, 'm> {
    model: &'m Model,
    recorder: Recorder<'c>,
}

impl<'m> Weights<'_, 'm> {
    pub(crate) fn matrix(&self, layer: usize, name: WeightName) -> &'m Weight {
        self.recorder.read(Buffer::Matrix {
            layer: index(layer),
            name,
        });
        weight(self.model, layer, name)
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
        (WeightName::ExpertGateUp, _) => &layer.moe.gate_up,
        (WeightName::ExpertDown, _) => &layer.moe.down,
        _ => unreachable!(),
    }
}
