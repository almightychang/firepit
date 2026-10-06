use cuda_async::cuda_graph::{CudaGraph, Scope};
use cuda_async::device_operation::DeviceOp;
use cuda_async::error::DeviceError;
use cuda_core::{Device, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::DType;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::gemv::row_scaled_tile;
use firepit_kernels::weights::{NVFP4_GROUP, Q8_BLOCK};
use firepit_kernels::{elementwise, gdn, norm, swiglu};
use std::error::Error;
use std::path::Path;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// Element-wise fusion candidates of one decode step, each recorded into a CUDA graph as the
// kernels decode runs today ("separate") and as one fused kernel ("fused"), with every layer
// on its own buffers and weights. The process first allocates BALLAST_GIB one-GiB buffers;
// each replay follows a read of a 64 MiB buffer. Time with nsys --cuda-graph-trace=node.
// Dimensions come from the model's config.json.
const WARMUP: usize = 1;
const REPLAYS: usize = 10;
const SCRUB_TILE: usize = 4096;
const SCRUB_LEN: usize = 16 << 20;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn scrub_read<const B: i32>(
        sums: &mut Tensor<f32, { [1] }>,
        buffer: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let values: Tile<f32, { [B] }> = buffer.partition(shape![B]).load([pid.0]);
        let sum: Tile<f32, { [] }> = reduce_sum(values, 0i32);
        sums.store(sum.reshape(shape![1]));
    }

    // dense_fp8_rows_kernel as in e54f4e7.
    #[cutile::entry()]
    pub fn fp8_rows_gemv<const BN: i32, const NB: i32, const KT: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        x: &Tensor<f32, { [1, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..(blocks / NB) {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = w_part.load([pid.1, kc]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let weights: Tile<f32, { [1, BN, KT] }> = weights.reshape(shape![1, BN, KT]);
            let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([0, kc, 0]);
            let xb: Tile<f32, { [1, BN, KT] }> =
                xb.reshape(shape![1, 1, KT]).broadcast(shape![1, BN, KT]);
            let part: Tile<f32, { [1, BN] }> = reduce_sum(weights * xb, 2i32);
            acc = acc + part.reshape(shape![1, BN]);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, pid.1]);
        y.store(acc * row_scale.broadcast(shape![1, BN]));
    }

    // fp8_rows_gemv adding the residual row to its output tile.
    #[cutile::entry()]
    pub fn fp8_rows_gemv_add<const BN: i32, const NB: i32, const KT: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        x: &Tensor<f32, { [1, -1, 32] }>,
        residual: &Tensor<f32, { [1, -1] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..(blocks / NB) {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = w_part.load([pid.1, kc]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let weights: Tile<f32, { [1, BN, KT] }> = weights.reshape(shape![1, BN, KT]);
            let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([0, kc, 0]);
            let xb: Tile<f32, { [1, BN, KT] }> =
                xb.reshape(shape![1, 1, KT]).broadcast(shape![1, BN, KT]);
            let part: Tile<f32, { [1, BN] }> = reduce_sum(weights * xb, 2i32);
            acc = acc + part.reshape(shape![1, BN]);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, pid.1]);
        let r: Tile<f32, { [1, BN] }> = residual.partition(shape![1, BN]).load([0, pid.1]);
        y.store(acc * row_scale.broadcast(shape![1, BN]) + r);
    }

    // combine_prefill_kernel as in e54f4e7.
    #[cutile::entry()]
    pub fn combine<const H: i32>(
        out: &mut Tensor<f32, { [1, H] }>,
        routed: &Tensor<f32, { [-1, H] }>,
        inverse: &Tensor<i32, { [-1, -1] }>,
        weights: &Tensor<f32, { [-1, -1] }>,
        slots: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut sum: Tile<f32, { [1, H] }> = constant(0.0f32, shape![1, H]);
        for rank in 0i32..slots {
            let location: Tile<i32, { [1, 1] }> =
                inverse.partition(shape![1, 1]).load([pid.0, rank]);
            let row: i32 = tile_to_scalar(location.reshape(shape![]));
            let value: Tile<f32, { [1, H] }> = routed.partition(shape![1, H]).load([row, 0]);
            let weight: Tile<f32, { [1, 1] }> = weights.partition(shape![1, 1]).load([pid.0, rank]);
            sum = sum + value * weight.broadcast(shape![1, H]);
        }
        out.store(sum);
    }

    // combine writing the residual sum.
    #[cutile::entry()]
    pub fn combine_add<const H: i32>(
        out: &mut Tensor<f32, { [1, H] }>,
        routed: &Tensor<f32, { [-1, H] }>,
        inverse: &Tensor<i32, { [-1, -1] }>,
        weights: &Tensor<f32, { [-1, -1] }>,
        residual: &Tensor<f32, { [-1, H] }>,
        slots: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut sum: Tile<f32, { [1, H] }> = residual.partition(shape![1, H]).load([pid.0, 0]);
        for rank in 0i32..slots {
            let location: Tile<i32, { [1, 1] }> =
                inverse.partition(shape![1, 1]).load([pid.0, rank]);
            let row: i32 = tile_to_scalar(location.reshape(shape![]));
            let value: Tile<f32, { [1, H] }> = routed.partition(shape![1, H]).load([row, 0]);
            let weight: Tile<f32, { [1, 1] }> = weights.partition(shape![1, 1]).load([pid.0, rank]);
            sum = sum + value * weight.broadcast(shape![1, H]);
        }
        out.store(sum);
    }

    // combine writing the residual sum and its rms_norm.
    #[cutile::entry()]
    pub fn combine_add_norm<const H: i32>(
        out: &mut Tensor<f32, { [1, H] }>,
        normed: &mut Tensor<f32, { [1, H] }>,
        routed: &Tensor<f32, { [-1, H] }>,
        inverse: &Tensor<i32, { [-1, -1] }>,
        weights: &Tensor<f32, { [-1, -1] }>,
        residual: &Tensor<f32, { [-1, H] }>,
        gamma: &Tensor<f32, { [H] }>,
        inv_h: f32,
        eps: f32,
        slots: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut sum: Tile<f32, { [1, H] }> = residual.partition(shape![1, H]).load([pid.0, 0]);
        for rank in 0i32..slots {
            let location: Tile<i32, { [1, 1] }> =
                inverse.partition(shape![1, 1]).load([pid.0, rank]);
            let row: i32 = tile_to_scalar(location.reshape(shape![]));
            let value: Tile<f32, { [1, H] }> = routed.partition(shape![1, H]).load([row, 0]);
            let weight: Tile<f32, { [1, 1] }> = weights.partition(shape![1, 1]).load([pid.0, rank]);
            sum = sum + value * weight.broadcast(shape![1, H]);
        }
        out.store(sum);
        let squares: Tile<f32, { [1] }> = reduce_sum(sum * sum, 1);
        let squares: Tile<f32, { [1, 1] }> = squares.reshape(shape![1, 1]);
        let scale = rsqrt(
            squares * inv_h.broadcast(shape![1, 1]) + eps.broadcast(shape![1, 1]),
            ftz::Disabled,
        );
        let gamma: Tile<f32, { [H] }> = gamma.partition(shape![H]).load([0]);
        let gamma: Tile<f32, { [1, H] }> = gamma.reshape(shape![1, H]);
        normed.store(sum * scale.broadcast(shape![1, H]) * gamma);
    }

    // add followed by rms_norm, writing both.
    #[cutile::entry()]
    pub fn add_norm<const H: i32>(
        out: &mut Tensor<f32, { [1, H] }>,
        normed: &mut Tensor<f32, { [1, H] }>,
        a: &Tensor<f32, { [-1, H] }>,
        b: &Tensor<f32, { [-1, H] }>,
        gamma: &Tensor<f32, { [H] }>,
        inv_h: f32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let ap: Tile<f32, { [1, H] }> = a.partition(shape![1, H]).load([pid.0, 0]);
        let bp: Tile<f32, { [1, H] }> = b.partition(shape![1, H]).load([pid.0, 0]);
        let sum = ap + bp;
        out.store(sum);
        let squares: Tile<f32, { [1] }> = reduce_sum(sum * sum, 1);
        let squares: Tile<f32, { [1, 1] }> = squares.reshape(shape![1, 1]);
        let scale = rsqrt(
            squares * inv_h.broadcast(shape![1, 1]) + eps.broadcast(shape![1, 1]),
            ftz::Disabled,
        );
        let gamma: Tile<f32, { [H] }> = gamma.partition(shape![H]).load([0]);
        let gamma: Tile<f32, { [1, H] }> = gamma.reshape(shape![1, H]);
        normed.store(sum * scale.broadcast(shape![1, H]) * gamma);
    }

    // rms_norm over one value head followed by gated_silu with the matching z columns.
    #[cutile::entry()]
    pub fn norm_gate<const DV: i32>(
        out: &mut Tensor<f32, { [1, DV] }>,
        input: &Tensor<f32, { [-1, -1, DV] }>,
        gate: &Tensor<f32, { [-1, -1] }>,
        gamma: &Tensor<f32, { [DV] }>,
        inv_dv: f32,
        eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let x: Tile<f32, { [1, DV] }> = input
            .partition(shape![1, 1, DV])
            .load([pid.0, pid.1, 0])
            .reshape(shape![1, DV]);
        let squares: Tile<f32, { [1] }> = reduce_sum(x * x, 1);
        let squares: Tile<f32, { [1, 1] }> = squares.reshape(shape![1, 1]);
        let scale = rsqrt(
            squares * inv_dv.broadcast(shape![1, 1]) + eps.broadcast(shape![1, 1]),
            ftz::Disabled,
        );
        let gamma: Tile<f32, { [DV] }> = gamma.partition(shape![DV]).load([0]);
        let gamma: Tile<f32, { [1, DV] }> = gamma.reshape(shape![1, DV]);
        let x = x * scale.broadcast(shape![1, DV]) * gamma;
        let z: Tile<f32, { [1, DV] }> = gate.partition(shape![1, DV]).load([pid.0, pid.1]);
        let one: Tile<f32, { [1, DV] }> = constant(1.0f32, shape![1, DV]);
        let negative: Tile<f32, { [1, DV] }> = constant(-1.0f32, shape![1, DV]);
        out.store(x * z / (one + exp(z * negative)));
    }

    // delta_step as in e54f4e7 with one whole value head per block, followed by the head's
    // rms_norm and gated_silu.
    #[cutile::entry()]
    pub fn delta_norm_gate<
        const KEY_HEADS: i32,
        const VALUE_HEADS: i32,
        const DK: i32,
        const DV: i32,
    >(
        out: &mut Tensor<f32, { [1, DV] }>,
        state: &mut Tensor<f32, { [1, DK, DV] }>,
        query: &Tensor<f32, { [-1, DK] }>,
        key: &Tensor<f32, { [-1, DK] }>,
        value: &Tensor<f32, { [-1, -1] }>,
        b: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        dt_bias: &Tensor<f32, { [-1] }>,
        gate: &Tensor<f32, { [-1, -1] }>,
        gamma: &Tensor<f32, { [DV] }>,
        eps: f32,
        inv_sqrt_dk: f32,
        inv_dv: f32,
        norm_eps: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let key_head = head / (VALUE_HEADS / KEY_HEADS);
        let q: Tile<f32, { [1, DK] }> = query.partition(shape![1, DK]).load([key_head, 0]);
        let k: Tile<f32, { [1, DK] }> = key.partition(shape![1, DK]).load([key_head, 0]);
        let q_norm: Tile<f32, { [1] }> = reduce_sum(q * q, 1);
        let k_norm: Tile<f32, { [1] }> = reduce_sum(k * k, 1);
        let q_scale = rsqrt(q_norm + eps.broadcast(shape![1]), ftz::Disabled)
            * inv_sqrt_dk.broadcast(shape![1]);
        let k_scale = rsqrt(k_norm + eps.broadcast(shape![1]), ftz::Disabled);
        let q: Tile<f32, { [1, DK, DV] }> =
            (q * q_scale.reshape(shape![1, 1]).broadcast(shape![1, DK]))
                .reshape(shape![1, DK, 1])
                .broadcast(shape![1, DK, DV]);
        let k: Tile<f32, { [1, DK, DV] }> =
            (k * k_scale.reshape(shape![1, 1]).broadcast(shape![1, DK]))
                .reshape(shape![1, DK, 1])
                .broadcast(shape![1, DK, DV]);

        let b: Tile<f32, { [1] }> = b.partition(shape![1]).load([head]);
        let a: Tile<f32, { [1] }> = a.partition(shape![1]).load([head]);
        let a_log: Tile<f32, { [1] }> = a_log.partition(shape![1]).load([head]);
        let dt: Tile<f32, { [1] }> = dt_bias.partition(shape![1]).load([head]);
        let one: Tile<f32, { [1] }> = constant(1.0f32, shape![1]);
        let zero: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
        let negative: Tile<f32, { [1] }> = constant(-1.0f32, shape![1]);
        let beta = one / (one + exp(b * negative));
        let x = a + dt;
        let tiny = exp(min_tile(x, zero));
        let series = tiny - tiny * tiny * constant(0.5f32, shape![1])
            + tiny * tiny * tiny * constant(0.33333334f32, shape![1]);
        let ordinary = max_tile(x, zero) + log(one + exp(absf(x) * negative));
        let cutoff: Tile<f32, { [1] }> = constant(-8.0f32, shape![1]);
        let softplus = select(lt_tile(x, cutoff), series, ordinary);
        let decay = exp(exp(a_log) * softplus * negative);

        let old: Tile<f32, { [1, DK, DV] }> = state.load();
        let memory: Tile<f32, { [1, 1, DV] }> = reduce_sum(old * k, 1);
        let memory: Tile<f32, { [1, DV] }> =
            memory.reshape(shape![1, DV]) * decay.reshape(shape![1, 1]).broadcast(shape![1, DV]);
        let v: Tile<f32, { [1, DV] }> = value.partition(shape![1, DV]).load([head, 0i32]);
        let delta = (v - memory) * beta.reshape(shape![1, 1]).broadcast(shape![1, DV]);
        let next = old * decay.reshape(shape![1, 1, 1]).broadcast(shape![1, DK, DV])
            + k * delta.reshape(shape![1, 1, DV]).broadcast(shape![1, DK, DV]);
        state.store(next);
        let result: Tile<f32, { [1, 1, DV] }> = reduce_sum(next * q, 1);
        let result: Tile<f32, { [1, DV] }> = result.reshape(shape![1, DV]);

        let squares: Tile<f32, { [1] }> = reduce_sum(result * result, 1);
        let squares: Tile<f32, { [1, 1] }> = squares.reshape(shape![1, 1]);
        let scale = rsqrt(
            squares * inv_dv.broadcast(shape![1, 1]) + norm_eps.broadcast(shape![1, 1]),
            ftz::Disabled,
        );
        let gamma: Tile<f32, { [DV] }> = gamma.partition(shape![DV]).load([0]);
        let gamma: Tile<f32, { [1, DV] }> = gamma.reshape(shape![1, DV]);
        let normed = result * scale.broadcast(shape![1, DV]) * gamma;
        let z: Tile<f32, { [1, DV] }> = gate.partition(shape![1, DV]).load([head, 0i32]);
        let ones: Tile<f32, { [1, DV] }> = constant(1.0f32, shape![1, DV]);
        let negatives: Tile<f32, { [1, DV] }> = constant(-1.0f32, shape![1, DV]);
        let gated = normed * z / (ones + exp(z * negatives));
        out.store(gated);
    }

    // expert_nvfp4_kernel as in e54f4e7.
    #[cutile::entry()]
    pub fn expert_nvfp4<
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const KT_PACKED: i32,
        const KT_GROUPS: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        x: &Tensor<f32, { [-1, -1, 32] }>,
        expert_ids: &Tensor<i32, { [-1, -1] }>,
        blocks: i32,
        row_tiles: i32,
        input_stride: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let expert_tile: Tile<i32, { [1, 1] }> =
            expert_ids.partition(shape![1, 1]).load([0, pid.0]);
        let expert: i32 = tile_to_scalar(expert_tile.reshape(shape![]));
        let row_tile = expert * row_tiles + pid.1;
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..(blocks / NB) {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = w_part.load([row_tile, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([pid.0 * input_stride, kc, 0]);
            let xb: Tile<f32, { [BN, KT] }> = xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
            let products: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                (weights * xb).reshape(shape![BN, KT_GROUPS, 16]);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = reduce_sum(products, 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> = scale_part.load([row_tile, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let sum: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
            acc = acc + sum.reshape(shape![1, BN]);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, row_tile]);
        y.store(acc * row_scale);
    }

    // expert_nvfp4 over BN gate rows and the BN up rows `up_tiles` row tiles further, ending
    // with swiglu of the two.
    #[cutile::entry()]
    pub fn expert_nvfp4_swiglu<
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const KT_PACKED: i32,
        const KT_GROUPS: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        x: &Tensor<f32, { [-1, -1, 32] }>,
        expert_ids: &Tensor<i32, { [-1, -1] }>,
        blocks: i32,
        row_tiles: i32,
        up_tiles: i32,
        input_stride: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let expert_tile: Tile<i32, { [1, 1] }> =
            expert_ids.partition(shape![1, 1]).load([0, pid.0]);
        let expert: i32 = tile_to_scalar(expert_tile.reshape(shape![]));
        let gate_tile = expert * row_tiles + pid.1;
        let up_tile = gate_tile + up_tiles;
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut gate_acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        let mut up_acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..(blocks / NB) {
            let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([pid.0 * input_stride, kc, 0]);
            let xb: Tile<f32, { [BN, KT] }> = xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);

            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = w_part.load([gate_tile, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let products: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                (weights * xb).reshape(shape![BN, KT_GROUPS, 16]);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = reduce_sum(products, 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> = scale_part.load([gate_tile, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let sum: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
            gate_acc = gate_acc + sum.reshape(shape![1, BN]);

            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = w_part.load([up_tile, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let products: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                (weights * xb).reshape(shape![BN, KT_GROUPS, 16]);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = reduce_sum(products, 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> = scale_part.load([up_tile, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let sum: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
            up_acc = up_acc + sum.reshape(shape![1, BN]);
        }
        let gate_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, gate_tile]);
        let up_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, up_tile]);
        let gate = gate_acc * gate_scale;
        let up = up_acc * up_scale;
        let one: Tile<f32, { [1, BN] }> = constant(1.0f32, shape![1, BN]);
        let minus_one: Tile<f32, { [1, BN] }> = constant(-1.0f32, shape![1, BN]);
        y.store(gate / (one + exp(gate * minus_one)) * up);
    }
}

struct Dims {
    hidden: usize,
    key_heads: usize,
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    attention_out: usize,
    intermediate: usize,
    slots: usize,
    eps: f32,
    gdn_layers: usize,
    attention_layers: usize,
}

impl Dims {
    fn from_config(path: &Path) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)?;
        let text = config.get("text_config").unwrap_or(&config);
        let int = |key: &str| -> Result<usize> {
            Ok(text[key].as_u64().ok_or(format!("config has no {key}"))? as usize)
        };
        let kinds = text["layer_types"]
            .as_array()
            .ok_or("config has no layer_types")?;
        let gdn_layers = kinds
            .iter()
            .filter(|k| k.as_str() == Some("linear_attention"))
            .count();
        let shared = int("shared_expert_intermediate_size")?;
        let intermediate = int("moe_intermediate_size")?;
        if shared != intermediate {
            return Err("the probe runs the shared expert as one more routed slot".into());
        }
        Ok(Self {
            hidden: int("hidden_size")?,
            key_heads: int("linear_num_key_heads")?,
            value_heads: int("linear_num_value_heads")?,
            key_head_dim: int("linear_key_head_dim")?,
            value_head_dim: int("linear_value_head_dim")?,
            attention_out: int("num_attention_heads")? * int("head_dim")?,
            intermediate,
            slots: int("num_experts_per_tok")? + 1,
            eps: text["rms_norm_eps"]
                .as_f64()
                .ok_or("config has no rms_norm_eps")? as f32,
            gdn_layers,
            attention_layers: kinds.len() - gdn_layers,
        })
    }

    fn value_dim(&self) -> usize {
        self.value_heads * self.value_head_dim
    }

    fn layers(&self) -> usize {
        self.gdn_layers + self.attention_layers
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 32
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }

    fn unit(&mut self) -> f32 {
        (self.next() % 2001) as f32 / 1000.0 - 1.0
    }

    fn units(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.unit()).collect()
    }
}

fn device<T: DType + Send + Sync + 'static>(
    stream: &Arc<Stream>,
    values: Vec<T>,
    shape: &[usize],
) -> Result<Tensor<T>> {
    Ok(api::copy_host_vec_to_device(&Arc::new(values))
        .reshape(shape)
        .sync_on(stream)?)
}

fn zeros(stream: &Arc<Stream>, shape: &[usize]) -> Result<Tensor<f32>> {
    Ok(api::zeros::<f32>(shape).sync_on(stream)?)
}

fn host(stream: &Arc<Stream>, tensor: &Tensor<f32>) -> Result<Vec<f32>> {
    Ok(tensor.dup().to_host_vec().sync_on(stream)?)
}

fn max_relative(a: &[f32], b: &[f32]) -> f32 {
    let scale = b
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()))
        .max(f32::MIN_POSITIVE);
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (p, q)| m.max((p - q).abs()))
        / scale
}

fn internal(error: impl std::fmt::Display) -> DeviceError {
    DeviceError::Internal(error.to_string())
}

struct Probe {
    stream: Arc<Stream>,
    buffer: Arc<Tensor<f32>>,
    sums: Tensor<f32>,
}

impl Probe {
    fn replay(
        &mut self,
        name: &str,
        kernels: usize,
        record: impl FnOnce(&Scope) -> std::result::Result<(), DeviceError>,
    ) -> Result<()> {
        let graph = CudaGraph::scope(&self.stream, record)?;
        println!(
            "GRAPH {name} kernels={kernels} replays={}",
            WARMUP + REPLAYS
        );
        for _ in 0..WARMUP + REPLAYS {
            kernels::scrub_read((&mut self.sums).partition([1]), self.buffer.clone())
                .generics(vec![SCRUB_TILE.to_string()])
                .sync_on(&self.stream)?;
            graph.launch().sync_on(&self.stream)?;
        }
        Ok(())
    }
}

struct GemvAddLayer {
    weights: Tensor<f8e4m3fn>,
    row_scales: Tensor<f32>,
    x: Tensor<f32>,
    residual: Tensor<f32>,
    delta: Tensor<f32>,
    out: Tensor<f32>,
}

// (a) GdnOut / AttnOut GEMV followed by the residual add.
fn gemv_add(probe: &mut Probe, dims: &Dims, rng: &mut Lcg) -> Result<()> {
    let stream = probe.stream.clone();
    let hidden = dims.hidden;
    let mut layers = (0..dims.layers())
        .map(|layer| {
            let cols = if layer < dims.gdn_layers {
                dims.value_dim()
            } else {
                dims.attention_out
            };
            Ok(GemvAddLayer {
                weights: device(
                    &stream,
                    (0..hidden * cols)
                        .map(|_| f8e4m3fn(rng.byte() & 0xbf))
                        .collect(),
                    &[hidden, cols],
                )?,
                row_scales: device(&stream, rng.units(hidden), &[1, hidden])?,
                x: device(&stream, rng.units(cols), &[1, cols / Q8_BLOCK, Q8_BLOCK])?,
                residual: device(&stream, rng.units(hidden), &[1, hidden])?,
                delta: zeros(&stream, &[1, hidden])?,
                out: zeros(&stream, &[1, hidden])?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let generics = |cols: usize| {
        let tile = row_scaled_tile(hidden, cols, 1);
        (
            tile.rows,
            vec![
                tile.rows.to_string(),
                tile.blocks.to_string(),
                (tile.blocks * Q8_BLOCK).to_string(),
            ],
            (cols / Q8_BLOCK) as i32,
        )
    };
    let count = layers.len();
    probe.replay("gemv_add separate", 2 * count, |s| {
        for layer in layers.iter_mut() {
            let (rows, g, blocks) = generics(layer.x.shape()[1] as usize * Q8_BLOCK);
            s.record(
                kernels::fp8_rows_gemv(
                    (&mut layer.delta).partition([1, rows]),
                    &layer.weights,
                    &layer.row_scales,
                    &layer.x,
                    blocks,
                )
                .generics(g),
            )?;
            s.record(
                elementwise::add(
                    (&mut layer.out).partition([1, hidden]),
                    &layer.residual,
                    &layer.delta,
                )
                .generics(vec![hidden.to_string()]),
            )?;
        }
        Ok(())
    })?;
    let separate = host(&stream, &layers[0].out)?;
    probe.replay("gemv_add fused", count, |s| {
        for layer in layers.iter_mut() {
            let (rows, g, blocks) = generics(layer.x.shape()[1] as usize * Q8_BLOCK);
            s.record(
                kernels::fp8_rows_gemv_add(
                    (&mut layer.out).partition([1, rows]),
                    &layer.weights,
                    &layer.row_scales,
                    &layer.x,
                    &layer.residual,
                    blocks,
                )
                .generics(g),
            )?;
        }
        Ok(())
    })?;
    println!(
        "CHECK gemv_add max_rel {:.3e}",
        max_relative(&host(&stream, &layers[0].out)?, &separate)
    );
    Ok(())
}

struct CombineLayer {
    routed: Tensor<f32>,
    inverse: Tensor<i32>,
    weights: Tensor<f32>,
    residual: Tensor<f32>,
    gamma: Tensor<f32>,
    delta: Tensor<f32>,
    out: Tensor<f32>,
    normed: Tensor<f32>,
}

// (a, b) MoE combine followed by the residual add and the next rms_norm.
fn combine_add_norm(probe: &mut Probe, dims: &Dims, rng: &mut Lcg) -> Result<()> {
    let stream = probe.stream.clone();
    let (hidden, slots) = (dims.hidden, dims.slots);
    let inv_h = 1.0 / hidden as f32;
    let mut layers = (0..dims.layers())
        .map(|_| {
            Ok(CombineLayer {
                routed: device(&stream, rng.units(slots * hidden), &[slots, hidden])?,
                inverse: device(&stream, (0..slots as i32).rev().collect(), &[1, slots])?,
                weights: device(&stream, rng.units(slots), &[1, slots])?,
                residual: device(&stream, rng.units(hidden), &[1, hidden])?,
                gamma: device(&stream, rng.units(hidden), &[hidden])?,
                delta: zeros(&stream, &[1, hidden])?,
                out: zeros(&stream, &[1, hidden])?,
                normed: zeros(&stream, &[1, hidden])?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let count = layers.len();
    let h = vec![hidden.to_string()];
    let norm_launch =
        |s: &Scope, layer: &mut CombineLayer| -> std::result::Result<(), DeviceError> {
            let x = layer.out.view(&[1, 1, hidden]).map_err(internal)?;
            s.record(
                norm::rms_norm(
                    (&mut layer.normed).partition([1, hidden]),
                    &x,
                    &layer.gamma,
                    inv_h,
                    dims.eps,
                )
                .generics(vec![hidden.to_string()]),
            )?;
            Ok(())
        };
    probe.replay("combine_add_norm separate", 3 * count, |s| {
        for layer in layers.iter_mut() {
            s.record(
                kernels::combine(
                    (&mut layer.delta).partition([1, hidden]),
                    &layer.routed,
                    &layer.inverse,
                    &layer.weights,
                    slots as i32,
                )
                .generics(h.clone()),
            )?;
            s.record(
                elementwise::add(
                    (&mut layer.out).partition([1, hidden]),
                    &layer.residual,
                    &layer.delta,
                )
                .generics(h.clone()),
            )?;
            norm_launch(s, layer)?;
        }
        Ok(())
    })?;
    let separate = (
        host(&stream, &layers[0].out)?,
        host(&stream, &layers[0].normed)?,
    );
    probe.replay("combine_add_norm combine_add+norm", 2 * count, |s| {
        for layer in layers.iter_mut() {
            s.record(
                kernels::combine_add(
                    (&mut layer.out).partition([1, hidden]),
                    &layer.routed,
                    &layer.inverse,
                    &layer.weights,
                    &layer.residual,
                    slots as i32,
                )
                .generics(h.clone()),
            )?;
            norm_launch(s, layer)?;
        }
        Ok(())
    })?;
    let two = (
        host(&stream, &layers[0].out)?,
        host(&stream, &layers[0].normed)?,
    );
    probe.replay("combine_add_norm fused", count, |s| {
        for layer in layers.iter_mut() {
            s.record(
                kernels::combine_add_norm(
                    (&mut layer.out).partition([1, hidden]),
                    (&mut layer.normed).partition([1, hidden]),
                    &layer.routed,
                    &layer.inverse,
                    &layer.weights,
                    &layer.residual,
                    &layer.gamma,
                    inv_h,
                    dims.eps,
                    slots as i32,
                )
                .generics(h.clone()),
            )?;
        }
        Ok(())
    })?;
    let fused = (
        host(&stream, &layers[0].out)?,
        host(&stream, &layers[0].normed)?,
    );
    println!(
        "CHECK combine_add_norm combine_add+norm residual {:.3e} normed {:.3e} fused residual {:.3e} normed {:.3e}",
        max_relative(&two.0, &separate.0),
        max_relative(&two.1, &separate.1),
        max_relative(&fused.0, &separate.0),
        max_relative(&fused.1, &separate.1)
    );

    // (b) on the GEMV side: add followed by rms_norm, residual and delta given.
    probe.replay("add_norm separate", 2 * count, |s| {
        for layer in layers.iter_mut() {
            s.record(
                elementwise::add(
                    (&mut layer.out).partition([1, hidden]),
                    &layer.residual,
                    &layer.routed,
                )
                .generics(h.clone()),
            )?;
            norm_launch(s, layer)?;
        }
        Ok(())
    })?;
    let separate = (
        host(&stream, &layers[0].out)?,
        host(&stream, &layers[0].normed)?,
    );
    probe.replay("add_norm fused", count, |s| {
        for layer in layers.iter_mut() {
            s.record(
                kernels::add_norm(
                    (&mut layer.out).partition([1, hidden]),
                    (&mut layer.normed).partition([1, hidden]),
                    &layer.residual,
                    &layer.routed,
                    &layer.gamma,
                    inv_h,
                    dims.eps,
                )
                .generics(h.clone()),
            )?;
        }
        Ok(())
    })?;
    println!(
        "CHECK add_norm residual {:.3e} normed {:.3e}",
        max_relative(&host(&stream, &layers[0].out)?, &separate.0),
        max_relative(&host(&stream, &layers[0].normed)?, &separate.1)
    );
    Ok(())
}

struct DeltaLayer {
    state: Tensor<f32>,
    query: Tensor<f32>,
    key: Tensor<f32>,
    value: Tensor<f32>,
    b: Tensor<f32>,
    a: Tensor<f32>,
    a_log: Tensor<f32>,
    dt_bias: Tensor<f32>,
    z: Tensor<f32>,
    gamma: Tensor<f32>,
    mixer: Tensor<f32>,
    mixer_norm: Tensor<f32>,
    gated: Tensor<f32>,
}

// (c) GDN delta step followed by the head rms_norm and gated_silu.
fn delta_norm_gate(probe: &mut Probe, dims: &Dims, rng: &mut Lcg, value_tile: usize) -> Result<()> {
    let stream = probe.stream.clone();
    let (kh, vh, dk, dv) = (
        dims.key_heads,
        dims.value_heads,
        dims.key_head_dim,
        dims.value_head_dim,
    );
    let initial: Vec<f32> = (0..vh * dk * dv).map(|_| rng.unit() * 0.1).collect();
    let make = |layers: usize| {
        (0..layers)
            .map(|layer| {
                let mut rng = Lcg(1000 + layer as u64);
                Ok(DeltaLayer {
                    state: device(&stream, initial.clone(), &[vh, dk, dv])?,
                    query: device(&stream, rng.units(kh * dk), &[kh, dk])?,
                    key: device(&stream, rng.units(kh * dk), &[kh, dk])?,
                    value: device(&stream, rng.units(vh * dv), &[vh, dv])?,
                    b: device(&stream, rng.units(vh), &[vh])?,
                    a: device(&stream, rng.units(vh), &[vh])?,
                    a_log: device(&stream, rng.units(vh), &[vh])?,
                    dt_bias: device(&stream, rng.units(vh), &[vh])?,
                    z: device(&stream, rng.units(vh * dv), &[vh, dv])?,
                    gamma: device(&stream, rng.units(dv), &[dv])?,
                    mixer: zeros(&stream, &[vh, 1, dv])?,
                    mixer_norm: zeros(&stream, &[vh, dv])?,
                    gated: zeros(&stream, &[vh, dv])?,
                })
            })
            .collect::<Result<Vec<_>>>()
    };
    let gdn_eps = dims.eps;
    let inv_sqrt_dk = 1.0 / (dk as f32).sqrt();
    let inv_dv = 1.0 / dv as f32;
    let delta_generics = |tile: usize| {
        vec![
            kh.to_string(),
            vh.to_string(),
            dk.to_string(),
            tile.to_string(),
        ]
    };
    let count = dims.gdn_layers;
    let mut separate_layers = make(count)?;
    let delta = |s: &Scope, layer: &mut DeltaLayer| -> std::result::Result<(), DeviceError> {
        s.record(
            gdn::delta_step(
                (&mut layer.mixer).partition([1, 1, value_tile]),
                (&mut layer.state).partition([1, dk, value_tile]),
                &layer.query,
                &layer.key,
                &layer.value,
                &layer.b,
                &layer.a,
                &layer.a_log,
                &layer.dt_bias,
                gdn_eps,
                inv_sqrt_dk,
            )
            .generics(delta_generics(value_tile)),
        )?;
        Ok(())
    };
    probe.replay("delta_norm_gate separate", 3 * count, |s| {
        for layer in separate_layers.iter_mut() {
            delta(s, layer)?;
            s.record(
                norm::rms_norm(
                    (&mut layer.mixer_norm).partition([1, dv]),
                    &layer.mixer,
                    &layer.gamma,
                    inv_dv,
                    dims.eps,
                )
                .generics(vec![dv.to_string()]),
            )?;
            s.record(
                gdn::gated_silu(
                    (&mut layer.gated).partition([1, dv]),
                    &layer.mixer_norm,
                    &layer.z,
                )
                .generics(vec![dv.to_string()]),
            )?;
        }
        Ok(())
    })?;
    let separate = host(&stream, &separate_layers[0].gated)?;
    drop(separate_layers);

    let mut two_layers = make(count)?;
    probe.replay("delta_norm_gate delta+norm_gate", 2 * count, |s| {
        for layer in two_layers.iter_mut() {
            delta(s, layer)?;
            s.record(
                kernels::norm_gate(
                    (&mut layer.gated).partition([1, dv]),
                    &layer.mixer,
                    &layer.z,
                    &layer.gamma,
                    inv_dv,
                    dims.eps,
                )
                .generics(vec![dv.to_string()]),
            )?;
        }
        Ok(())
    })?;
    let two = host(&stream, &two_layers[0].gated)?;
    drop(two_layers);

    let mut fused_layers = make(count)?;
    probe.replay("delta_norm_gate fused", count, |s| {
        for layer in fused_layers.iter_mut() {
            s.record(
                kernels::delta_norm_gate(
                    (&mut layer.gated).partition([1, dv]),
                    (&mut layer.state).partition([1, dk, dv]),
                    &layer.query,
                    &layer.key,
                    &layer.value,
                    &layer.b,
                    &layer.a,
                    &layer.a_log,
                    &layer.dt_bias,
                    &layer.z,
                    &layer.gamma,
                    gdn_eps,
                    inv_sqrt_dk,
                    inv_dv,
                    dims.eps,
                )
                .generics(vec![
                    kh.to_string(),
                    vh.to_string(),
                    dk.to_string(),
                    dv.to_string(),
                ]),
            )?;
        }
        Ok(())
    })?;
    println!(
        "CHECK delta_norm_gate delta+norm_gate {:.3e} fused {:.3e}",
        max_relative(&two, &separate),
        max_relative(&host(&stream, &fused_layers[0].gated)?, &separate)
    );
    Ok(())
}

struct ExpertLayer {
    packed: Tensor<f4e2m1fnx2>,
    scales: Tensor<f8e4m3fn>,
    row_scales: Tensor<f32>,
    x: Tensor<f32>,
    experts: Tensor<i32>,
    gate_up: Tensor<f32>,
    act: Tensor<f32>,
}

// (d) Expert gate_up GEMV followed by swiglu; the probe keeps only the selected experts.
fn gate_up_swiglu(probe: &mut Probe, dims: &Dims, rng: &mut Lcg) -> Result<()> {
    let stream = probe.stream.clone();
    let (hidden, slots, inter) = (dims.hidden, dims.slots, dims.intermediate);
    let rows = slots * 2 * inter;
    let mut layers = (0..dims.layers())
        .map(|_| {
            Ok(ExpertLayer {
                packed: device(
                    &stream,
                    (0..rows * hidden / 2)
                        .map(|_| f4e2m1fnx2::from_bits(rng.byte()))
                        .collect(),
                    &[rows, hidden / 2],
                )?,
                scales: device(
                    &stream,
                    (0..rows * hidden / NVFP4_GROUP)
                        .map(|_| f8e4m3fn(rng.byte() & 0x3f))
                        .collect(),
                    &[rows, hidden / NVFP4_GROUP],
                )?,
                row_scales: device(&stream, rng.units(rows), &[1, rows])?,
                x: device(
                    &stream,
                    rng.units(hidden),
                    &[1, hidden / Q8_BLOCK, Q8_BLOCK],
                )?,
                experts: device(&stream, (0..slots as i32).collect(), &[1, slots])?,
                gate_up: zeros(&stream, &[slots, 2 * inter])?,
                act: zeros(&stream, &[slots, inter])?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let generics = |bn: usize, blocks: usize| {
        let kt = blocks * Q8_BLOCK;
        vec![
            bn.to_string(),
            blocks.to_string(),
            kt.to_string(),
            (kt / 2).to_string(),
            (kt / NVFP4_GROUP).to_string(),
        ]
    };
    let tile = row_scaled_tile(2 * inter, hidden, slots);
    let blocks = (hidden / Q8_BLOCK) as i32;
    let count = layers.len();
    println!(
        "TILE gate_up rows {} blocks {} (grid {}x{})",
        tile.rows,
        tile.blocks,
        slots,
        2 * inter / tile.rows
    );
    probe.replay("gate_up_swiglu separate", 2 * count, |s| {
        for layer in layers.iter_mut() {
            s.record(
                kernels::expert_nvfp4(
                    (&mut layer.gate_up).partition([1, tile.rows]),
                    &layer.packed,
                    &layer.scales,
                    &layer.row_scales,
                    &layer.x,
                    &layer.experts,
                    blocks,
                    (2 * inter / tile.rows) as i32,
                    0,
                )
                .generics(generics(tile.rows, tile.blocks)),
            )?;
            s.record(
                swiglu::swiglu((&mut layer.act).partition([1, inter]), &layer.gate_up)
                    .generics(vec![inter.to_string()]),
            )?;
        }
        Ok(())
    })?;
    let separate = host(&stream, &layers[0].act)?;
    for bn in [tile.rows, tile.rows / 2] {
        probe.replay(&format!("gate_up_swiglu fused BN{bn}"), count, |s| {
            for layer in layers.iter_mut() {
                s.record(
                    kernels::expert_nvfp4_swiglu(
                        (&mut layer.act).partition([1, bn]),
                        &layer.packed,
                        &layer.scales,
                        &layer.row_scales,
                        &layer.x,
                        &layer.experts,
                        blocks,
                        (2 * inter / bn) as i32,
                        (inter / bn) as i32,
                        0,
                    )
                    .generics(generics(bn, tile.blocks)),
                )?;
            }
            Ok(())
        })?;
        println!(
            "CHECK gate_up_swiglu BN{bn} max_rel {:.3e}",
            max_relative(&host(&stream, &layers[0].act)?, &separate)
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [config, ballast_gib, value_tile] = &args[..] else {
        return Err("usage: fusion_probe CONFIG_JSON BALLAST_GIB GDN_VALUE_TILE".into());
    };
    let dims = Dims::from_config(Path::new(config))?;
    let value_tile: usize = value_tile.parse()?;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    let ballast: Vec<Tensor<f32>> = (0..ballast_gib.parse::<usize>()?)
        .map(|_| api::zeros::<f32>(&[1 << 28]).sync_on(&stream))
        .collect::<std::result::Result<_, _>>()?;
    let mut probe = Probe {
        buffer: Arc::new(api::zeros::<f32>(&[SCRUB_LEN]).sync_on(&stream)?),
        sums: api::zeros::<f32>(&[SCRUB_LEN / SCRUB_TILE]).sync_on(&stream)?,
        stream: stream.clone(),
    };
    let mut rng = Lcg(7);
    gemv_add(&mut probe, &dims, &mut rng)?;
    combine_add_norm(&mut probe, &dims, &mut rng)?;
    delta_norm_gate(&mut probe, &dims, &mut rng, value_tile)?;
    gate_up_swiglu(&mut probe, &dims, &mut rng)?;
    drop(ballast);
    Ok(())
}
