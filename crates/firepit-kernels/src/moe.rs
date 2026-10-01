use cuda_async::device_operation::{DeviceOp, GraphNode};
use cutile::api;
use cutile::tensor::{Partition, PartitionMut, Tensor, TensorView};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

use crate::weights::Q8_BLOCK;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn top_k<const SIZE: i32, const TOP: i32>(
        logits: &Tensor<f32, { [-1] }>,
        indices: &mut Tensor<i32, { [TOP] }>,
        weights: &mut Tensor<f32, { [TOP] }>,
        experts: i32,
        k: i32,
    ) {
        let values: Tile<f32, { [SIZE] }> = logits.partition(shape![SIZE]).load([0]);
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
        indices.store(selected_ids);
        weights.store(selected_scores / total.broadcast(shape![TOP]));
    }

    #[cutile::entry()]
    pub fn combine_kernel<const H: i32>(
        out: &mut Tensor<f32, { [1, H] }>,
        routed: &Tensor<f32, { [-1, H] }>,
        weights: &Tensor<f32, { [-1] }>,
        shared: &Tensor<f32, { [1, H] }>,
        gate: &Tensor<f32, { [1, 1] }>,
        k: i32,
    ) {
        let mut sum: Tile<f32, { [1, H] }> = constant(0.0f32, shape![1, H]);
        for rank in 0i32..k {
            let row: Tile<f32, { [1, H] }> = routed.partition(shape![1, H]).load([rank, 0]);
            let weight: Tile<f32, { [1] }> = weights.partition(shape![1]).load([rank]);
            sum = sum + row * weight.reshape(shape![1, 1]).broadcast(shape![1, H]);
        }
        let shared: Tile<f32, { [1, H] }> = shared.partition(shape![1, H]).load([0, 0]);
        let gate: Tile<f32, { [1, 1] }> = gate.partition(shape![1, 1]).load([0, 0]);
        let one: Tile<f32, { [1, 1] }> = constant(1.0f32, shape![1, 1]);
        let sigmoid = one / (one + exp(gate * constant(-1.0f32, shape![1, 1])));
        out.store(sum + shared * sigmoid.broadcast(shape![1, H]));
    }
}

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
        vec![self.hidden.to_string()]
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
    dims: MoeDims,
}

impl ExpertSelection {
    pub fn top_k(
        logits: &TensorView<'_, f32>,
        dims: &MoeDims,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert_eq!(logits.shape(), &[dims.experts as i32]);
        let padded = dims.top_k.next_power_of_two();
        let mut indices = api::zeros::<i32>(&[padded]).sync_on(stream)?;
        let mut weights = api::zeros::<f32>(&[padded]).sync_on(stream)?;
        kernels::top_k(
            logits,
            indices.partition([padded]),
            weights.partition([padded]),
            dims.experts as i32,
            dims.top_k as i32,
        )
        .generics(dims.top_k_generics())
        .sync_on(stream)?;
        Ok(Self {
            indices,
            weights,
            dims: *dims,
        })
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
            indices: api::copy_host_vec_to_device(&Arc::new(padded_indices)).sync_on(stream)?,
            weights: api::copy_host_vec_to_device(&Arc::new(padded_weights)).sync_on(stream)?,
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
    kernels::combine_kernel(
        out,
        routed,
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
