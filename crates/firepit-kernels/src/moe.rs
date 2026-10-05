use cuda_async::device_operation::{DeviceOp, GraphNode};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{Partition, PartitionMut, Tensor, TensorView, TensorViewMut};
use cutile::tile_kernel::TileKernel;
use cutile::tile_kernel::ToHostVecOp;
use std::sync::Arc;

use crate::sink::Sink;
use crate::weights::Q8_BLOCK;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn top_k_prefill<const SIZE: i32, const TOP: i32>(
        logits: &Tensor<f32, { [-1, -1] }>,
        indices: &mut Tensor<i32, { [1, TOP] }>,
        weights: &mut Tensor<f32, { [1, TOP] }>,
        experts: i32,
        k: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let values: Tile<f32, { [SIZE] }> = logits
            .partition(shape![1, SIZE])
            .load([pid.0, 0])
            .reshape(shape![SIZE]);
        let positions: Tile<i32, { [SIZE] }> = iota(shape![SIZE]);
        let ranks: Tile<i32, { [TOP] }> = iota(shape![TOP]);
        let valid: Tile<bool, { [SIZE] }> = lt_tile(positions, experts.broadcast(shape![SIZE]));
        let negative: Tile<f32, { [SIZE] }> = constant(f32::NEG_INFINITY, shape![SIZE]);
        let mut remaining: Tile<f32, { [SIZE] }> = select(valid, values, negative);
        let maximum: Tile<f32, { [1] }> = reduce_max(remaining, 0i32);
        let origin: f32 = tile_to_scalar(maximum.reshape(shape![]));
        let mut total: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
        let mut selected_ids: Tile<i32, { [TOP] }> = constant(0i32, shape![TOP]);
        let mut selected_scores: Tile<f32, { [TOP] }> = constant(0.0f32, shape![TOP]);
        for rank in 0i32..k {
            let peak_tile: Tile<f32, { [1] }> = reduce_max(remaining, 0i32);
            let peak: f32 = tile_to_scalar(peak_tile.reshape(shape![]));
            let matching: Tile<bool, { [SIZE] }> = eq_tile(remaining, peak.broadcast(shape![SIZE]));
            let candidates: Tile<i32, { [SIZE] }> = select(
                matching,
                positions,
                (experts - 1i32).broadcast(shape![SIZE]),
            );
            let winner_tile: Tile<i32, { [1] }> = reduce_min(candidates, 0i32);
            let winner: i32 = tile_to_scalar(winner_tile.reshape(shape![]));
            let score: Tile<f32, { [1] }> = exp((peak - origin).broadcast(shape![1]));
            let slot: Tile<bool, { [TOP] }> = eq_tile(ranks, rank.broadcast(shape![TOP]));
            selected_ids = select(slot, winner.broadcast(shape![TOP]), selected_ids);
            selected_scores = select(slot, score.broadcast(shape![TOP]), selected_scores);
            total = total + score;
            let chosen: Tile<bool, { [SIZE] }> = eq_tile(positions, winner.broadcast(shape![SIZE]));
            remaining = select(chosen, negative, remaining);
        }
        indices.store(selected_ids.reshape(shape![1, TOP]));
        weights.store((selected_scores / total.broadcast(shape![TOP])).reshape(shape![1, TOP]));
    }

    #[cutile::entry()]
    pub fn gather_prefill_kernel<const BK: i32, const K: i32>(
        sorted: &mut Tensor<f32, { [1, BK] }>,
        x: &Tensor<f32, { [-1, K] }>,
        token_rows: &Tensor<i32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let token: i32 = tile_to_scalar(
            token_rows
                .partition(shape![1])
                .load([pid.0])
                .reshape(shape![]),
        );
        let value: Tile<f32, { [1, BK] }> = if token >= 0i32 {
            x.partition(shape![1, BK]).load([token, pid.1])
        } else {
            constant(0.0f32, shape![1, BK])
        };
        sorted.store(value);
    }

    #[cutile::entry()]
    pub fn combine_prefill_kernel<const H: i32, const TOP: i32>(
        out: &mut Tensor<f32, { [1, H] }>,
        routed: &Tensor<f32, { [-1, H] }>,
        inverse: &Tensor<i32, { [-1, -1] }>,
        weights: &Tensor<f32, { [-1, -1] }>,
        shared: &Tensor<f32, { [-1, H] }>,
        gate: &Tensor<f32, { [-1, 1] }>,
        k: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let locations: Tile<i32, { [1, TOP] }> = inverse.partition(shape![1, TOP]).load([pid.0, 0]);
        let rank_weights: Tile<f32, { [1, TOP] }> =
            weights.partition(shape![1, TOP]).load([pid.0, 0]);
        let mut sum: Tile<f32, { [1, H] }> = constant(0.0f32, shape![1, H]);
        for rank in 0i32..k {
            let location: Tile<i32, { [1, 1] }> =
                extract(locations, [scalar_to_tile(0i32), scalar_to_tile(rank)]);
            let row: i32 = tile_to_scalar(location.reshape(shape![]));
            let value: Tile<f32, { [1, H] }> = routed.partition(shape![1, H]).load([row, 0]);
            let weight: Tile<f32, { [1, 1] }> =
                extract(rank_weights, [scalar_to_tile(0i32), scalar_to_tile(rank)]);
            sum = sum + value * weight.broadcast(shape![1, H]);
        }
        let shared: Tile<f32, { [1, H] }> = shared.partition(shape![1, H]).load([pid.0, 0]);
        let gate: Tile<f32, { [1, 1] }> = gate.partition(shape![1, 1]).load([pid.0, 0]);
        let one: Tile<f32, { [1, 1] }> = constant(1.0f32, shape![1, 1]);
        let sigmoid = one / (one + exp(gate * constant(-1.0f32, shape![1, 1])));
        out.store(sum + shared * sigmoid.broadcast(shape![1, H]));
    }
}

pub use kernels::top_k_prefill;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeDimensionError;

impl std::fmt::Display for MoeDimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid MoE dimensions")
    }
}

impl std::error::Error for MoeDimensionError {}

#[derive(Debug, Clone, Copy)]
pub struct MoeDims {
    hidden: usize,
    intermediate: usize,
    shared_intermediate: usize,
    experts: usize,
    top_k: usize,
}

impl MoeDims {
    pub fn new(
        hidden: usize,
        intermediate: usize,
        shared_intermediate: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<Self, MoeDimensionError> {
        if hidden == 0
            || intermediate == 0
            || shared_intermediate == 0
            || experts == 0
            || top_k == 0
            || top_k > experts
            || hidden % Q8_BLOCK != 0
            || intermediate % Q8_BLOCK != 0
            || shared_intermediate % Q8_BLOCK != 0
            || !hidden.is_power_of_two()
            || !intermediate.is_power_of_two()
            || !shared_intermediate.is_power_of_two()
            || shared_intermediate
                .checked_mul(2)
                .is_none_or(|n| n > i32::MAX as usize)
            || experts
                .checked_next_power_of_two()
                .is_none_or(|n| n > i32::MAX as usize)
            || top_k
                .checked_next_power_of_two()
                .is_none_or(|n| n > i32::MAX as usize)
            || hidden
                .checked_mul(experts)
                .is_none_or(|n| n > i32::MAX as usize)
            || intermediate
                .checked_mul(2)
                .and_then(|n| n.checked_mul(experts))
                .is_none_or(|n| n > i32::MAX as usize)
        {
            return Err(MoeDimensionError);
        }
        Ok(Self {
            hidden,
            intermediate,
            shared_intermediate,
            experts,
            top_k,
        })
    }

    pub fn top_k_generics(self) -> Vec<String> {
        vec![
            self.experts.next_power_of_two().to_string(),
            self.top_k.next_power_of_two().to_string(),
        ]
    }
    pub fn combine_generics(self) -> Vec<String> {
        vec![
            self.hidden.to_string(),
            self.top_k.next_power_of_two().to_string(),
        ]
    }

    pub fn hidden(self) -> usize {
        self.hidden
    }
    pub fn intermediate(self) -> usize {
        self.intermediate
    }
    pub fn shared_intermediate(self) -> usize {
        self.shared_intermediate
    }
    pub fn experts(self) -> usize {
        self.experts
    }
    pub fn top_k(self) -> usize {
        self.top_k
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertSelectionError {
    Count,
    Index,
    Duplicate,
    Weight,
    Tile,
}

impl std::fmt::Display for ExpertSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid expert selection: {self:?}")
    }
}

impl std::error::Error for ExpertSelectionError {}

pub struct ExpertSelection {
    indices: Tensor<i32>,
    weights: Tensor<f32>,
    inverse: Tensor<i32>,
    dims: MoeDims,
}

impl ExpertSelection {
    pub fn empty(
        dims: &MoeDims,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let padded = dims.top_k.next_power_of_two();
        Ok(Self {
            indices: api::zeros::<i32>(&[1, padded]).sync_on(stream)?,
            weights: api::zeros::<f32>(&[1, padded]).sync_on(stream)?,
            inverse: api::copy_host_vec_to_device(&Arc::new((0..padded as i32).collect()))
                .reshape(&[1, padded])
                .sync_on(stream)?,
            dims: *dims,
        })
    }

    pub fn select(
        &mut self,
        logits: &TensorView<'_, f32>,
        sink: &mut impl Sink,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(logits.shape(), &[self.dims.experts as i32]);
        let padded = self.dims.top_k.next_power_of_two();
        let rows = logits.view(&[1, self.dims.experts])?;
        sink.run(
            kernels::top_k_prefill(
                &rows,
                self.indices.partition([1, padded]),
                self.weights.partition([1, padded]),
                self.dims.experts as i32,
                self.dims.top_k as i32,
            )
            .generics(self.dims.top_k_generics()),
        )?;
        Ok(())
    }

    pub fn top_k(
        logits: &TensorView<'_, f32>,
        dims: &MoeDims,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert_eq!(logits.shape(), &[dims.experts as i32]);
        let mut selection = Self::empty(dims, stream)?;
        selection.select(logits, &mut crate::sink::Eager(stream))?;
        Ok(selection)
    }

    pub fn parse(
        indices: &[i32],
        weights: &[f32],
        dims: &MoeDims,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if indices.len() != dims.top_k || weights.len() != dims.top_k {
            return Err(ExpertSelectionError::Count.into());
        }
        let mut seen = vec![false; dims.experts];
        for &index in indices {
            if index < 0 || index as usize >= dims.experts {
                return Err(ExpertSelectionError::Index.into());
            }
            if std::mem::replace(&mut seen[index as usize], true) {
                return Err(ExpertSelectionError::Duplicate.into());
            }
        }
        let total: f32 = weights.iter().sum();
        if weights
            .iter()
            .any(|&weight| !weight.is_finite() || weight < 0.0)
            || !total.is_finite()
            || total <= 0.0
        {
            return Err(ExpertSelectionError::Weight.into());
        }
        let padded = dims.top_k.next_power_of_two();
        let mut padded_indices = indices.to_vec();
        padded_indices.resize(padded, 0);
        let mut padded_weights = weights.to_vec();
        padded_weights.resize(padded, 0.0);
        Ok(Self {
            indices: api::copy_host_vec_to_device(&Arc::new(padded_indices))
                .reshape(&[1, padded])
                .sync_on(stream)?,
            weights: api::copy_host_vec_to_device(&Arc::new(padded_weights))
                .reshape(&[1, padded])
                .sync_on(stream)?,
            inverse: api::copy_host_vec_to_device(&Arc::new((0..padded as i32).collect()))
                .reshape(&[1, padded])
                .sync_on(stream)?,
            dims: *dims,
        })
    }

    pub fn indices(&self) -> &Tensor<i32> {
        &self.indices
    }

    pub(crate) fn dims(&self) -> MoeDims {
        self.dims
    }

    pub fn weights(&self) -> &Tensor<f32> {
        &self.weights
    }
}

#[derive(Debug, Clone)]
pub struct PrefillLayout {
    tokens: usize,
    tile_rows: usize,
    token_rows: Vec<i32>,
    tile_experts: Vec<i32>,
    inverse: Vec<i32>,
    ranges: Vec<(usize, usize)>,
}

impl PrefillLayout {
    pub fn max_padded_rows(
        tokens: usize,
        dims: &MoeDims,
        tile_rows: usize,
    ) -> Result<usize, ExpertSelectionError> {
        if tokens == 0 || tile_rows == 0 || !tile_rows.is_power_of_two() {
            return Err(ExpertSelectionError::Tile);
        }
        let routes = tokens
            .checked_mul(dims.top_k)
            .ok_or(ExpertSelectionError::Count)?;
        let occupied = dims.experts.min(routes);
        routes
            .checked_add(
                occupied
                    .checked_mul(tile_rows - 1)
                    .ok_or(ExpertSelectionError::Count)?,
            )
            .filter(|&rows| rows <= i32::MAX as usize)
            .ok_or(ExpertSelectionError::Count)
    }

    pub fn parse(
        indices: &[i32],
        weights: &[f32],
        dims: &MoeDims,
        tile_rows: usize,
    ) -> Result<Self, ExpertSelectionError> {
        if indices.is_empty() || indices.len() % dims.top_k != 0 || weights.len() != indices.len() {
            return Err(ExpertSelectionError::Count);
        }
        for (ids, scores) in indices
            .chunks_exact(dims.top_k)
            .zip(weights.chunks_exact(dims.top_k))
        {
            let total: f32 = scores.iter().sum();
            if scores.iter().any(|&x| !x.is_finite() || x < 0.0)
                || !total.is_finite()
                || total <= 0.0
            {
                return Err(ExpertSelectionError::Weight);
            }
            let mut seen = vec![false; dims.experts];
            for &id in ids {
                if id < 0 || id as usize >= dims.experts {
                    return Err(ExpertSelectionError::Index);
                }
                if std::mem::replace(&mut seen[id as usize], true) {
                    return Err(ExpertSelectionError::Duplicate);
                }
            }
        }
        Self::from_indices(indices, dims, tile_rows)
    }

    fn from_indices(
        indices: &[i32],
        dims: &MoeDims,
        tile_rows: usize,
    ) -> Result<Self, ExpertSelectionError> {
        if indices.is_empty()
            || indices.len() % dims.top_k != 0
            || indices.len() > i32::MAX as usize
        {
            return Err(ExpertSelectionError::Count);
        }
        if tile_rows == 0 || tile_rows > i32::MAX as usize || !tile_rows.is_power_of_two() {
            return Err(ExpertSelectionError::Tile);
        }
        let tokens = indices.len() / dims.top_k;
        if tokens
            .checked_mul(dims.top_k.next_power_of_two())
            .is_none_or(|len| len > i32::MAX as usize)
        {
            return Err(ExpertSelectionError::Count);
        }
        let mut entries = Vec::with_capacity(indices.len());
        for (token, ids) in indices.chunks_exact(dims.top_k).enumerate() {
            let mut seen = vec![false; dims.experts];
            for (slot, &id) in ids.iter().enumerate() {
                if id < 0 || id as usize >= dims.experts {
                    return Err(ExpertSelectionError::Index);
                }
                if std::mem::replace(&mut seen[id as usize], true) {
                    return Err(ExpertSelectionError::Duplicate);
                }
                entries.push((id as usize, token, slot));
            }
        }
        entries.sort_by_key(|entry| entry.0);
        let mut token_rows = Vec::new();
        let mut tile_experts = Vec::new();
        let mut inverse = vec![0i32; indices.len()];
        let mut ranges = vec![(0usize, 0usize); dims.experts];
        let mut cursor = 0;
        for (expert, range) in ranges.iter_mut().enumerate() {
            range.0 = token_rows.len();
            while cursor < entries.len() && entries[cursor].0 == expert {
                let (_, token, slot) = entries[cursor];
                if token_rows.len() > i32::MAX as usize {
                    return Err(ExpertSelectionError::Count);
                }
                inverse[token * dims.top_k + slot] = token_rows.len() as i32;
                token_rows.push(token as i32);
                cursor += 1;
                range.1 += 1;
            }
            let padded = range
                .1
                .checked_next_multiple_of(tile_rows)
                .ok_or(ExpertSelectionError::Count)?;
            let end = range
                .0
                .checked_add(padded)
                .filter(|&end| end <= i32::MAX as usize)
                .ok_or(ExpertSelectionError::Count)?;
            token_rows.resize(end, -1);
            tile_experts.extend(std::iter::repeat_n(expert as i32, padded / tile_rows));
        }
        if token_rows.len() > i32::MAX as usize {
            return Err(ExpertSelectionError::Count);
        }
        Ok(Self {
            tokens,
            tile_rows,
            token_rows,
            tile_experts,
            inverse,
            ranges,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn padded_rows(&self) -> usize {
        self.token_rows.len()
    }
    pub fn padding_rows(&self) -> usize {
        self.token_rows.len() - self.inverse.len()
    }
    pub fn token_rows(&self) -> &[i32] {
        &self.token_rows
    }
    pub fn tile_experts(&self) -> &[i32] {
        &self.tile_experts
    }
    pub fn inverse(&self) -> &[i32] {
        &self.inverse
    }
    pub fn ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }
    pub fn tile_rows(&self) -> usize {
        self.tile_rows
    }
}

pub struct PrefillPlan {
    dims: MoeDims,
    layout: PrefillLayout,
    indices: Tensor<i32>,
    weights: Tensor<f32>,
    token_rows: Tensor<i32>,
    tile_experts: Tensor<i32>,
    inverse: Tensor<i32>,
}

impl PrefillPlan {
    pub fn top_k(
        logits: &TensorView<'_, f32>,
        dims: &MoeDims,
        tile_rows: usize,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let shape = logits.shape();
        if shape.len() != 2 || shape[0] <= 0 || shape[1] != dims.experts as i32 {
            return Err(ExpertSelectionError::Count.into());
        }
        let tokens = shape[0] as usize;
        let padded = dims.top_k.next_power_of_two();
        if tokens
            .checked_mul(padded)
            .is_none_or(|len| len > i32::MAX as usize)
        {
            return Err(ExpertSelectionError::Count.into());
        }
        let mut indices = api::zeros::<i32>(&[tokens, padded]).sync_on(stream)?;
        let mut weights = api::zeros::<f32>(&[tokens, padded]).sync_on(stream)?;
        let host: Vec<i32> = kernels::top_k_prefill(
            logits,
            indices.partition([1, padded]),
            weights.partition([1, padded]),
            dims.experts as i32,
            dims.top_k as i32,
        )
        .generics(dims.top_k_generics())
        .then(|(_, ids, _, _, _)| ids.unpartition().dup().to_host_vec())
        .sync_on(stream)?;
        let ids: Vec<i32> = host
            .chunks_exact(padded)
            .flat_map(|row| row[..dims.top_k].iter().copied())
            .collect();
        let layout = PrefillLayout::from_indices(&ids, dims, tile_rows)?;
        Self::upload(layout, indices, weights, dims, stream)
    }

    pub fn parse(
        indices: &[i32],
        weights: &[f32],
        dims: &MoeDims,
        tile_rows: usize,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let layout = PrefillLayout::parse(indices, weights, dims, tile_rows)?;
        let padded = dims.top_k.next_power_of_two();
        let mut ids = vec![0; layout.tokens * padded];
        let mut scores = vec![0.0; layout.tokens * padded];
        for token in 0..layout.tokens {
            ids[token * padded..token * padded + dims.top_k]
                .copy_from_slice(&indices[token * dims.top_k..(token + 1) * dims.top_k]);
            scores[token * padded..token * padded + dims.top_k]
                .copy_from_slice(&weights[token * dims.top_k..(token + 1) * dims.top_k]);
        }
        let gpu_ids = api::copy_host_vec_to_device(&Arc::new(ids))
            .reshape(&[layout.tokens, padded])
            .sync_on(stream)?;
        let gpu_scores = api::copy_host_vec_to_device(&Arc::new(scores))
            .reshape(&[layout.tokens, padded])
            .sync_on(stream)?;
        Self::upload(layout, gpu_ids, gpu_scores, dims, stream)
    }

    fn upload(
        layout: PrefillLayout,
        indices: Tensor<i32>,
        weights: Tensor<f32>,
        dims: &MoeDims,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let upload = |data: Vec<i32>| api::copy_host_vec_to_device(&Arc::new(data)).sync_on(stream);
        let inverse = layout
            .inverse
            .chunks_exact(dims.top_k)
            .flat_map(|row| {
                row.iter().copied().chain(std::iter::repeat_n(
                    0,
                    dims.top_k.next_power_of_two() - dims.top_k,
                ))
            })
            .collect();
        Ok(Self {
            dims: *dims,
            token_rows: upload(layout.token_rows.clone())?,
            tile_experts: upload(layout.tile_experts.clone())?,
            inverse: api::copy_host_vec_to_device(&Arc::new(inverse))
                .reshape(&[layout.tokens, dims.top_k.next_power_of_two()])
                .sync_on(stream)?,
            layout,
            indices,
            weights,
        })
    }

    pub fn layout(&self) -> &PrefillLayout {
        &self.layout
    }
    pub fn indices(&self) -> &Tensor<i32> {
        &self.indices
    }
    pub fn weights(&self) -> &Tensor<f32> {
        &self.weights
    }
    pub(crate) fn tile_experts(&self) -> &Tensor<i32> {
        &self.tile_experts
    }
    pub(crate) fn dims(&self) -> MoeDims {
        self.dims
    }
}

pub fn gather_prefill<'a>(
    sorted: Partition<&'a mut Tensor<f32>>,
    x: &'a TensorView<'a, f32>,
    plan: &'a PrefillPlan,
    tile_cols: usize,
) -> impl DeviceOp<
    Output = (
        Partition<&'a mut Tensor<f32>>,
        &'a TensorView<'a, f32>,
        &'a Tensor<i32>,
    ),
> + GraphNode
+ 'a {
    assert_eq!(sorted.partition_shape, vec![1, tile_cols]);
    assert_eq!(
        x.shape(),
        &[plan.layout.tokens as i32, plan.dims.hidden as i32]
    );
    assert_eq!(
        sorted.grid().expect("valid output grid"),
        (
            plan.layout.padded_rows() as u32,
            (plan.dims.hidden / tile_cols) as u32,
            1,
        )
    );
    assert_eq!(plan.dims.hidden % tile_cols, 0);
    kernels::gather_prefill_kernel(sorted, x, &plan.token_rows)
        .generics(vec![tile_cols.to_string(), plan.dims.hidden.to_string()])
}

pub fn gather_prefill_view<'a>(
    sorted: Partition<TensorViewMut<'a, f32>>,
    x: &'a TensorView<'a, f32>,
    plan: &'a PrefillPlan,
    tile_cols: usize,
) -> impl DeviceOp<
    Output = (
        Partition<TensorViewMut<'a, f32>>,
        &'a TensorView<'a, f32>,
        &'a Tensor<i32>,
    ),
> + GraphNode
+ 'a {
    assert_eq!(sorted.partition_shape, vec![1, tile_cols]);
    assert_eq!(
        x.shape(),
        &[plan.layout.tokens as i32, plan.dims.hidden as i32]
    );
    assert_eq!(
        sorted.grid().expect("valid output grid"),
        (
            plan.layout.padded_rows() as u32,
            (plan.dims.hidden / tile_cols) as u32,
            1,
        )
    );
    assert_eq!(plan.dims.hidden % tile_cols, 0);
    kernels::gather_prefill_kernel(sorted, x, &plan.token_rows)
        .generics(vec![tile_cols.to_string(), plan.dims.hidden.to_string()])
}

pub fn combine_prefill<'a>(
    out: Partition<&'a mut Tensor<f32>>,
    routed: &'a Tensor<f32>,
    plan: &'a PrefillPlan,
    shared: &'a Tensor<f32>,
    gate: &'a Tensor<f32>,
) -> impl DeviceOp<
    Output = (
        Partition<&'a mut Tensor<f32>>,
        &'a Tensor<f32>,
        &'a Tensor<i32>,
        &'a Tensor<f32>,
        &'a Tensor<f32>,
        &'a Tensor<f32>,
        i32,
    ),
> + GraphNode
+ 'a {
    let dims = plan.dims;
    assert_eq!(out.partition_shape, vec![1, dims.hidden]);
    assert_eq!(
        routed.shape(),
        &[plan.layout.padded_rows() as i32, dims.hidden as i32]
    );
    assert_eq!(
        shared.shape(),
        &[plan.layout.tokens as i32, dims.hidden as i32]
    );
    assert_eq!(gate.shape(), &[plan.layout.tokens as i32, 1]);
    kernels::combine_prefill_kernel(
        out,
        routed,
        &plan.inverse,
        &plan.weights,
        shared,
        gate,
        dims.top_k as i32,
    )
    .generics(dims.combine_generics())
}

pub fn combine_prefill_view<'a>(
    out: Partition<&'a mut Tensor<f32>>,
    routed: &'a Tensor<f32>,
    plan: &'a PrefillPlan,
    shared: &'a TensorView<'a, f32>,
    gate: &'a TensorView<'a, f32>,
) -> impl DeviceOp<
    Output = (
        Partition<&'a mut Tensor<f32>>,
        &'a Tensor<f32>,
        &'a Tensor<i32>,
        &'a Tensor<f32>,
        &'a TensorView<'a, f32>,
        &'a TensorView<'a, f32>,
        i32,
    ),
> + GraphNode
+ 'a {
    let dims = plan.dims;
    assert_eq!(out.partition_shape, vec![1, dims.hidden]);
    assert_eq!(
        routed.shape(),
        &[plan.layout.padded_rows() as i32, dims.hidden as i32]
    );
    assert_eq!(
        shared.shape(),
        &[plan.layout.tokens as i32, dims.hidden as i32]
    );
    assert_eq!(gate.shape(), &[plan.layout.tokens as i32, 1]);
    kernels::combine_prefill_kernel(
        out,
        routed,
        &plan.inverse,
        &plan.weights,
        shared,
        gate,
        dims.top_k as i32,
    )
    .generics(dims.combine_generics())
}

pub fn combine_prefill_sliced<'a>(
    out: Partition<&'a mut Tensor<f32>>,
    routed: &'a TensorView<'a, f32>,
    plan: &'a PrefillPlan,
    shared: &'a TensorView<'a, f32>,
    gate: &'a TensorView<'a, f32>,
) -> impl DeviceOp<
    Output = (
        Partition<&'a mut Tensor<f32>>,
        &'a TensorView<'a, f32>,
        &'a Tensor<i32>,
        &'a Tensor<f32>,
        &'a TensorView<'a, f32>,
        &'a TensorView<'a, f32>,
        i32,
    ),
> + GraphNode
+ 'a {
    let dims = plan.dims;
    assert_eq!(out.partition_shape, vec![1, dims.hidden]);
    assert_eq!(
        routed.shape(),
        &[plan.layout.padded_rows() as i32, dims.hidden as i32]
    );
    assert_eq!(
        shared.shape(),
        &[plan.layout.tokens as i32, dims.hidden as i32]
    );
    assert_eq!(gate.shape(), &[plan.layout.tokens as i32, 1]);
    kernels::combine_prefill_kernel(
        out,
        routed,
        &plan.inverse,
        &plan.weights,
        shared,
        gate,
        dims.top_k as i32,
    )
    .generics(dims.combine_generics())
}

pub fn combine<'a>(
    out: Partition<&'a mut Tensor<f32>>,
    routed: &'a Tensor<f32>,
    selection: &'a ExpertSelection,
    shared: &'a Tensor<f32>,
    gate: &'a Tensor<f32>,
) -> impl DeviceOp<
    Output = (
        Partition<&'a mut Tensor<f32>>,
        &'a Tensor<f32>,
        &'a Tensor<i32>,
        &'a Tensor<f32>,
        &'a Tensor<f32>,
        &'a Tensor<f32>,
        i32,
    ),
> + GraphNode
+ 'a {
    let dims = selection.dims;
    assert_eq!(routed.shape(), &[dims.top_k as i32, dims.hidden as i32]);
    assert_eq!(shared.shape(), &[1, dims.hidden as i32]);
    assert_eq!(gate.shape(), &[1, 1]);
    kernels::combine_prefill_kernel(
        out,
        routed,
        &selection.inverse,
        &selection.weights,
        shared,
        gate,
        dims.top_k as i32,
    )
    .generics(dims.combine_generics())
}

pub fn combine_view<'a>(
    out: Partition<TensorViewMut<'a, f32>>,
    routed: &'a TensorView<'a, f32>,
    selection: &'a ExpertSelection,
    shared: &'a TensorView<'a, f32>,
    gate: &'a TensorView<'a, f32>,
) -> impl DeviceOp<
    Output = (
        Partition<TensorViewMut<'a, f32>>,
        &'a TensorView<'a, f32>,
        &'a Tensor<i32>,
        &'a Tensor<f32>,
        &'a TensorView<'a, f32>,
        &'a TensorView<'a, f32>,
        i32,
    ),
> + GraphNode
+ 'a {
    let dims = selection.dims;
    assert_eq!(routed.shape(), &[dims.top_k as i32, dims.hidden as i32]);
    assert_eq!(shared.shape(), &[1, dims.hidden as i32]);
    assert_eq!(gate.shape(), &[1, 1]);
    kernels::combine_prefill_kernel(
        out,
        routed,
        &selection.inverse,
        &selection.weights,
        shared,
        gate,
        dims.top_k as i32,
    )
    .generics(dims.combine_generics())
}

#[cfg(test)]
mod tests {
    use super::MoeDims;

    #[test]
    fn dimension_validation() {
        assert!(MoeDims::new(2048, 512, 512, 256, 8).is_ok());
        for dims in [
            (0, 512, 512, 256, 8),
            (2048, 0, 512, 256, 8),
            (2048, 512, 0, 256, 8),
            (2048, 512, 512, 0, 8),
            (2048, 512, 512, 256, 0),
            (2048, 512, 512, 4, 8),
            (48, 512, 512, 256, 8),
            (2048, 96, 512, 256, 8),
        ] {
            assert!(MoeDims::new(dims.0, dims.1, dims.2, dims.3, dims.4).is_err());
        }
    }
}
