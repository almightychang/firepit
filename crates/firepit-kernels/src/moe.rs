use cuda_async::device_operation::{DeviceOp, GraphNode};
use cutile::DType;
use cutile::core::ElementType;
use cutile::tensor::{Partition, Tensor, TensorView};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

use crate::dense::DenseMatrix;
use crate::q8::{GemvTile, Q8_BLOCK};

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
            let candidates: Tile<i32, { [SIZE] }> =
                select(matching, positions, experts.broadcast(shape![SIZE]));
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
    pub fn expert_gemv_kernel<T: ElementType, const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<T, { [-1, -1, 32] }>,
        x: &Tensor<f32, { [-1, -1, 32] }>,
        expert_ids: &Tensor<i32, { [-1] }>,
        blocks: i32,
        row_tiles: i32,
        experts: i32,
        input_stride: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let expert_tile: Tile<i32, { [1] }> = expert_ids.partition(shape![1]).load([pid.0]);
        let expert: i32 = tile_to_scalar(expert_tile.reshape(shape![]));
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        if expert >= 0 && expert < experts {
            let w_part = w.partition(shape![BN, NB, 32]);
            let x_part = x.partition(shape![1, NB, 32]);
            for kc in 0i32..(blocks / NB) {
                let weights: Tile<T, { [BN, NB, 32] }> =
                    w_part.load([expert * row_tiles + pid.1, kc, 0]);
                let weights: Tile<f32, { [BN, NB, 32] }> = convert_tile(weights);
                let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([pid.0 * input_stride, kc, 0]);
                let xb: Tile<f32, { [BN, NB, 32] }> = xb.broadcast(shape![BN, NB, 32]);
                let products: Tile<f32, { [BN, NB] }> = reduce_sum(weights * xb, 2);
                let products: Tile<f32, { [BN] }> = reduce_sum(products, 1);
                acc = acc + products.reshape(shape![1, BN]);
            }
        }
        y.store(acc);
    }

    #[cutile::entry()]
    pub fn combine<const H: i32>(
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

pub use kernels::{combine, top_k};

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

pub fn expert_gemv<'a, T: DType + ElementType>(
    w: &DenseMatrix<T>,
    x: &'a TensorView<'a, f32>,
    expert_ids: &'a Tensor<i32>,
    y: Partition<&'a mut Tensor<f32>>,
    experts: usize,
    rows_per_expert: usize,
    active: usize,
    input_stride: usize,
    tile: GemvTile,
) -> impl DeviceOp<
    Output = (
        Partition<&'a mut Tensor<f32>>,
        Arc<Tensor<T>>,
        &'a TensorView<'a, f32>,
        &'a Tensor<i32>,
        i32,
        i32,
        i32,
        i32,
    ),
> + GraphNode
+ 'a {
    assert!(
        experts > 0 && experts <= i32::MAX as usize && rows_per_expert > 0 && input_stride <= 1
    );
    assert_eq!(
        w.rows,
        experts
            .checked_mul(rows_per_expert)
            .expect("expert matrix rows overflow")
    );
    assert!(tile.rows > 0 && tile.blocks > 0 && tile.fits(rows_per_expert, w.cols));
    assert!(active > 0 && active <= expert_ids.shape()[0] as usize);
    assert_eq!(
        x.shape(),
        &[
            (if input_stride == 0 { 1 } else { active }) as i32,
            (w.cols / Q8_BLOCK) as i32,
            Q8_BLOCK as i32
        ]
    );
    kernels::expert_gemv_kernel(
        y,
        w.weights.clone(),
        x,
        expert_ids,
        (w.cols / Q8_BLOCK) as i32,
        (rows_per_expert / tile.rows) as i32,
        experts as i32,
        input_stride as i32,
    )
    .generics(vec![
        T::DTYPE.as_str().to_string(),
        tile.rows.to_string(),
        tile.blocks.to_string(),
    ])
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
