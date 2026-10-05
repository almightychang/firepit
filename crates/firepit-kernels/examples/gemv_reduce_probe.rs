use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::gemv::fp8_tile;
use half::bf16;
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// Each GEMV body comes in two forms over the same data and the same product tile: the product
// form, which finishes a K reduction every iteration, and an accumulating form, which adds
// element-wise products into a [BN, KT] accumulator and reduces once after the loop.
// Kernel times come from nsys; each launch follows a read of twice the L2 size.
const WARMUP: usize = 2;
const REPS: usize = 8;
const SCRUB_TILE: usize = 4096;
const SCALE_BLOCK: usize = 128;
const KT: usize = 128;
const BLOCKS_PER_TILE: usize = KT / 32;
const NVFP4_GROUP: usize = 16;

#[cutile::module]
mod probes {
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

    #[cutile::entry()]
    pub fn load_accumulate<const BN: i32, const KT: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT]);
        let mut acc: Tile<u32, { [BN, KT] }> = constant(0u32, shape![BN, KT]);
        for kc in 0i32..chunks {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = wp.load([pid.1, kc]);
            let bits: Tile<u8, { [BN, KT] }> = bitcast(weights);
            let wide: Tile<u32, { [BN, KT] }> = exti(bits);
            acc = acc + wide;
        }
        let sum: Tile<u32, { [BN] }> = reduce_sum(acc, 1i32);
        let out: Tile<f32, { [BN] }> = bitcast(sum);
        y.store(out.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn load_reduce<const BN: i32, const KT: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT]);
        let mut acc: Tile<u32, { [BN] }> = constant(0u32, shape![BN]);
        for kc in 0i32..chunks {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = wp.load([pid.1, kc]);
            let bits: Tile<u8, { [BN, KT] }> = bitcast(weights);
            let wide: Tile<u32, { [BN, KT] }> = exti(bits);
            let part: Tile<u32, { [BN] }> = reduce_sum(wide, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        let out: Tile<f32, { [BN] }> = bitcast(acc);
        y.store(out.reshape(shape![1, BN]));
    }

    // dense_fp8_kernel with B = 1, as in crates/firepit-kernels/src/gemv/fp8.rs.
    #[cutile::entry()]
    pub fn fp8_reduce<
        const B: i32,
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const BLOCK_N: i32,
        const BLOCK_K: i32,
    >(
        y: &mut Tensor<f32, { [B, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        scales: &Tensor<f32, { [-1, -1] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT]);
        let x_part = x.partition(shape![B, NB, 32]);
        let scale_part = scales.partition(shape![1, 1]);
        let mut acc: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
        let scale_row = pid.1 * BN / BLOCK_N;
        let steps_per_scale = BLOCK_K / KT;
        for scale_col in 0i32..(blocks * 32 / BLOCK_K) {
            let scale: Tile<f32, { [1, 1] }> = scale_part.load([scale_row, scale_col]);
            let mut sum: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
            for step in 0i32..steps_per_scale {
                let kc = scale_col * steps_per_scale + step;
                let weights: Tile<f8e4m3fn, { [BN, KT] }> = w_part.load([pid.1, kc]);
                let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
                let weights: Tile<f32, { [B, BN, KT] }> = weights
                    .reshape(shape![1, BN, KT])
                    .broadcast(shape![B, BN, KT]);
                let xb: Tile<f32, { [B, NB, 32] }> = x_part.load([0, kc, 0]);
                let xb: Tile<f32, { [B, BN, KT] }> =
                    xb.reshape(shape![B, 1, KT]).broadcast(shape![B, BN, KT]);
                let part: Tile<f32, { [B, BN] }> = reduce_sum(weights * xb, 2i32);
                let part: Tile<f32, { [B, BN] }> = part.reshape(shape![B, BN]);
                sum = sum + part;
            }
            let scale: Tile<f32, { [B, BN] }> = scale.broadcast(shape![B, BN]);
            acc = acc + sum * scale;
        }
        y.store(acc);
    }

    #[cutile::entry()]
    pub fn fp8_accumulate<
        const B: i32,
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const BLOCK_N: i32,
        const BLOCK_K: i32,
    >(
        y: &mut Tensor<f32, { [B, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        scales: &Tensor<f32, { [-1, -1] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT]);
        let x_part = x.partition(shape![B, NB, 32]);
        let scale_part = scales.partition(shape![1, 1]);
        let mut acc: Tile<f32, { [B, BN, KT] }> = constant(0.0f32, shape![B, BN, KT]);
        let scale_row = pid.1 * BN / BLOCK_N;
        let steps_per_scale = BLOCK_K / KT;
        for scale_col in 0i32..(blocks * 32 / BLOCK_K) {
            let scale: Tile<f32, { [1, 1] }> = scale_part.load([scale_row, scale_col]);
            let scale: Tile<f32, { [B, BN, KT] }> =
                scale.reshape(shape![1, 1, 1]).broadcast(shape![B, BN, KT]);
            for step in 0i32..steps_per_scale {
                let kc = scale_col * steps_per_scale + step;
                let weights: Tile<f8e4m3fn, { [BN, KT] }> = w_part.load([pid.1, kc]);
                let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
                let weights: Tile<f32, { [B, BN, KT] }> = weights
                    .reshape(shape![1, BN, KT])
                    .broadcast(shape![B, BN, KT]);
                let xb: Tile<f32, { [B, NB, 32] }> = x_part.load([0, kc, 0]);
                let xb: Tile<f32, { [B, BN, KT] }> =
                    xb.reshape(shape![B, 1, KT]).broadcast(shape![B, BN, KT]);
                acc = acc + weights * xb * scale;
            }
        }
        let sum: Tile<f32, { [B, BN] }> = reduce_sum(acc, 2i32);
        y.store(sum.reshape(shape![B, BN]));
    }

    // dense_gemv_kernel with T = bf16 and B = 1, as in crates/firepit-kernels/src/gemv/dense.rs.
    #[cutile::entry()]
    pub fn bf16_reduce<const B: i32, const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [B, BN] }>,
        w: &Tensor<bf16, { [-1, -1, 32] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, NB, 32]);
        let x_part = x.partition(shape![B, NB, 32]);
        let mut acc: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
        for kc in 0i32..(blocks / NB) {
            let weights: Tile<bf16, { [BN, NB, 32] }> = w_part.load([pid.1, kc, 0i32]);
            let weights: Tile<f32, { [BN, NB, 32] }> = convert_tile(weights);
            let weights: Tile<f32, { [B, BN, NB, 32] }> = weights
                .reshape(shape![1, BN, NB, 32])
                .broadcast(shape![B, BN, NB, 32]);
            let xb: Tile<f32, { [B, NB, 32] }> = x_part.load([0i32, kc, 0i32]);
            let xb: Tile<f32, { [B, BN, NB, 32] }> = xb
                .reshape(shape![B, 1, NB, 32])
                .broadcast(shape![B, BN, NB, 32]);
            let part: Tile<f32, { [B, BN, NB] }> = reduce_sum(weights * xb, 3i32);
            let part: Tile<f32, { [B, BN, NB] }> = part.reshape(shape![B, BN, NB]);
            let sum: Tile<f32, { [B, BN] }> = reduce_sum(part, 2i32);
            let sum: Tile<f32, { [B, BN] }> = sum.reshape(shape![B, BN]);
            acc = acc + sum;
        }
        y.store(acc);
    }

    #[cutile::entry()]
    pub fn bf16_accumulate<const B: i32, const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [B, BN] }>,
        w: &Tensor<bf16, { [-1, -1, 32] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, NB, 32]);
        let x_part = x.partition(shape![B, NB, 32]);
        let mut acc: Tile<f32, { [B, BN, NB, 32] }> = constant(0.0f32, shape![B, BN, NB, 32]);
        for kc in 0i32..(blocks / NB) {
            let weights: Tile<bf16, { [BN, NB, 32] }> = w_part.load([pid.1, kc, 0i32]);
            let weights: Tile<f32, { [BN, NB, 32] }> = convert_tile(weights);
            let weights: Tile<f32, { [B, BN, NB, 32] }> = weights
                .reshape(shape![1, BN, NB, 32])
                .broadcast(shape![B, BN, NB, 32]);
            let xb: Tile<f32, { [B, NB, 32] }> = x_part.load([0i32, kc, 0i32]);
            let xb: Tile<f32, { [B, BN, NB, 32] }> = xb
                .reshape(shape![B, 1, NB, 32])
                .broadcast(shape![B, BN, NB, 32]);
            acc = acc + weights * xb;
        }
        let part: Tile<f32, { [B, BN, NB] }> = reduce_sum(acc, 3i32);
        let part: Tile<f32, { [B, BN, NB] }> = part.reshape(shape![B, BN, NB]);
        let sum: Tile<f32, { [B, BN] }> = reduce_sum(part, 2i32);
        y.store(sum.reshape(shape![B, BN]));
    }

    // expert_nvfp4_kernel without the expert indirection, as in
    // crates/firepit-kernels/src/gemv/nvfp4.rs.
    #[cutile::entry()]
    pub fn nvfp4_reduce<
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
        x: &Tensor<f32, { [1, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..(blocks / NB) {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = w_part.load([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([0, kc, 0]);
            let xb: Tile<f32, { [BN, KT] }> = xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
            let products: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                (weights * xb).reshape(shape![BN, KT_GROUPS, 16]);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = reduce_sum(products, 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> = scale_part.load([pid.1, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let sum: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
            acc = acc + sum.reshape(shape![1, BN]);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, pid.1]);
        y.store(acc * row_scale);
    }

    #[cutile::entry()]
    pub fn nvfp4_accumulate<
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
        x: &Tensor<f32, { [1, -1, 32] }>,
        blocks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [BN, KT_GROUPS, 16] }> =
            constant(0.0f32, shape![BN, KT_GROUPS, 16]);
        for kc in 0i32..(blocks / NB) {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = w_part.load([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([0, kc, 0]);
            let xb: Tile<f32, { [BN, KT] }> = xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
            let products: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                (weights * xb).reshape(shape![BN, KT_GROUPS, 16]);
            let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> = scale_part.load([pid.1, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let scale: Tile<f32, { [BN, KT_GROUPS, 16] }> = scale
                .reshape(shape![BN, KT_GROUPS, 1])
                .broadcast(shape![BN, KT_GROUPS, 16]);
            acc = acc + products * scale;
        }
        let groups: Tile<f32, { [BN, KT_GROUPS] }> = reduce_sum(acc, 2i32);
        let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
        let sum: Tile<f32, { [BN] }> = reduce_sum(groups, 1i32);
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, pid.1]);
        y.store(sum.reshape(shape![1, BN]) * row_scale);
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
    }
}

fn upload<T: cutile::DType>(
    stream: &Arc<Stream>,
    data: Vec<T>,
    shape: &[usize],
) -> Result<Arc<Tensor<T>>> {
    Ok(Arc::new(
        api::copy_host_vec_to_device(&Arc::new(data))
            .reshape(shape)
            .sync_on(stream)?,
    ))
}

struct Probe {
    stream: Arc<Stream>,
    buffer: Arc<Tensor<f32>>,
    sums: Tensor<f32>,
}

impl Probe {
    fn repeat(
        &mut self,
        label: &str,
        mut launch: impl FnMut(&Arc<Stream>) -> Result<()>,
    ) -> Result<()> {
        println!("CFG {label} launches={}", WARMUP + REPS);
        for _ in 0..WARMUP + REPS {
            probes::scrub_read((&mut self.sums).partition([1]), self.buffer.clone())
                .generics(vec![SCRUB_TILE.to_string()])
                .sync_on(&self.stream)?;
            launch(&self.stream)?;
        }
        Ok(())
    }
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

fn fp8_group(probe: &mut Probe, rows: usize, cols: usize, rng: &mut Lcg) -> Result<()> {
    let stream = probe.stream.clone();
    let bn = fp8_tile(rows, cols, SCALE_BLOCK, SCALE_BLOCK, 1).rows;
    let w = upload(
        &stream,
        (0..rows * cols)
            .map(|_| f8e4m3fn((0x28 + rng.next() % 0x20) as u8 | ((rng.next() & 1) << 7) as u8))
            .collect(),
        &[rows, cols],
    )?;
    let scales = upload(
        &stream,
        (0..rows / SCALE_BLOCK * (cols / SCALE_BLOCK))
            .map(|_| 0.5 + (rng.unit() + 1.0) * 0.75)
            .collect(),
        &[rows / SCALE_BLOCK, cols / SCALE_BLOCK],
    )?;
    let x = upload(
        &stream,
        (0..cols).map(|_| rng.unit()).collect(),
        &[1, cols / 32, 32],
    )?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let chunks = (cols / KT) as i32;
    let shape = format!("N={rows} K={cols} BN={bn}");
    let rows_generics = vec![bn.to_string(), KT.to_string()];
    let fp8_generics = vec![
        "1".to_string(),
        bn.to_string(),
        BLOCKS_PER_TILE.to_string(),
        KT.to_string(),
        SCALE_BLOCK.to_string(),
        SCALE_BLOCK.to_string(),
    ];
    probe.repeat(&format!("L0 load_accumulate {shape}"), |s| {
        probes::load_accumulate((&mut y).partition([1, bn]), w.clone(), chunks)
            .generics(rows_generics.clone())
            .sync_on(s)?;
        Ok(())
    })?;
    probe.repeat(&format!("L1 load_reduce {shape}"), |s| {
        probes::load_reduce((&mut y).partition([1, bn]), w.clone(), chunks)
            .generics(rows_generics.clone())
            .sync_on(s)?;
        Ok(())
    })?;
    let mut run = |label: &str, accumulate: bool, probe: &mut Probe| -> Result<Vec<f32>> {
        probe.repeat(&format!("{label} {shape}"), |s| {
            if accumulate {
                probes::fp8_accumulate(
                    (&mut y).partition([1, bn]),
                    w.clone(),
                    scales.clone(),
                    x.clone(),
                    (cols / 32) as i32,
                )
                .generics(fp8_generics.clone())
                .sync_on(s)?;
            } else {
                probes::fp8_reduce(
                    (&mut y).partition([1, bn]),
                    w.clone(),
                    scales.clone(),
                    x.clone(),
                    (cols / 32) as i32,
                )
                .generics(fp8_generics.clone())
                .sync_on(s)?;
            }
            Ok(())
        })?;
        Ok(y.dup().to_host_vec().sync_on(&stream)?)
    };
    let reduced = run("F0 fp8_reduce", false, probe)?;
    let accumulated = run("F1 fp8_accumulate", true, probe)?;
    println!(
        "CHECK F1-F0 {shape} max_rel {:.3e}",
        max_relative(&accumulated, &reduced)
    );
    Ok(())
}

fn bf16_group(probe: &mut Probe, rows: usize, cols: usize, rng: &mut Lcg) -> Result<()> {
    let stream = probe.stream.clone();
    let bn = 16;
    let nb = BLOCKS_PER_TILE;
    let w = upload(
        &stream,
        (0..rows * cols)
            .map(|_| bf16::from_f32(rng.unit()))
            .collect(),
        &[rows, cols / 32, 32],
    )?;
    let x = upload(
        &stream,
        (0..cols).map(|_| rng.unit()).collect(),
        &[1, cols / 32, 32],
    )?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let shape = format!("N={rows} K={cols} BN={bn}");
    let generics = vec!["1".to_string(), bn.to_string(), nb.to_string()];
    let mut run = |label: &str, accumulate: bool, probe: &mut Probe| -> Result<Vec<f32>> {
        probe.repeat(&format!("{label} {shape}"), |s| {
            if accumulate {
                probes::bf16_accumulate(
                    (&mut y).partition([1, bn]),
                    w.clone(),
                    x.clone(),
                    (cols / 32) as i32,
                )
                .generics(generics.clone())
                .sync_on(s)?;
            } else {
                probes::bf16_reduce(
                    (&mut y).partition([1, bn]),
                    w.clone(),
                    x.clone(),
                    (cols / 32) as i32,
                )
                .generics(generics.clone())
                .sync_on(s)?;
            }
            Ok(())
        })?;
        Ok(y.dup().to_host_vec().sync_on(&stream)?)
    };
    let reduced = run("B0 bf16_reduce", false, probe)?;
    let accumulated = run("B1 bf16_accumulate", true, probe)?;
    println!(
        "CHECK B1-B0 {shape} max_rel {:.3e}",
        max_relative(&accumulated, &reduced)
    );
    Ok(())
}

fn nvfp4_group(probe: &mut Probe, rows: usize, cols: usize, rng: &mut Lcg) -> Result<()> {
    let stream = probe.stream.clone();
    let bn = 128;
    let w = upload(
        &stream,
        (0..rows * cols / 2)
            .map(|_| f4e2m1fnx2::from_bits(rng.next() as u8))
            .collect(),
        &[rows, cols / 2],
    )?;
    let scales = upload(
        &stream,
        (0..rows * cols / NVFP4_GROUP)
            .map(|_| f8e4m3fn((0x30 + rng.next() % 0x10) as u8))
            .collect(),
        &[rows, cols / NVFP4_GROUP],
    )?;
    let row_scales = upload(&stream, vec![1.0f32; rows], &[1, rows])?;
    let x = upload(
        &stream,
        (0..cols).map(|_| rng.unit()).collect(),
        &[1, cols / 32, 32],
    )?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let shape = format!("N={rows} K={cols} BN={bn}");
    let generics = vec![
        bn.to_string(),
        BLOCKS_PER_TILE.to_string(),
        KT.to_string(),
        (KT / 2).to_string(),
        (KT / NVFP4_GROUP).to_string(),
    ];
    let mut run = |label: &str, accumulate: bool, probe: &mut Probe| -> Result<Vec<f32>> {
        probe.repeat(&format!("{label} {shape}"), |s| {
            if accumulate {
                probes::nvfp4_accumulate(
                    (&mut y).partition([1, bn]),
                    w.clone(),
                    scales.clone(),
                    row_scales.clone(),
                    x.clone(),
                    (cols / 32) as i32,
                )
                .generics(generics.clone())
                .sync_on(s)?;
            } else {
                probes::nvfp4_reduce(
                    (&mut y).partition([1, bn]),
                    w.clone(),
                    scales.clone(),
                    row_scales.clone(),
                    x.clone(),
                    (cols / 32) as i32,
                )
                .generics(generics.clone())
                .sync_on(s)?;
            }
            Ok(())
        })?;
        Ok(y.dup().to_host_vec().sync_on(&stream)?)
    };
    let reduced = run("N0 nvfp4_reduce", false, probe)?;
    let accumulated = run("N1 nvfp4_accumulate", true, probe)?;
    println!(
        "CHECK N1-N0 {shape} max_rel {:.3e}",
        max_relative(&accumulated, &reduced)
    );
    Ok(())
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (group, rows) = match &args[..] {
        [group, rows] => (group.as_str(), rows.parse::<usize>()?),
        _ => return Err("usage: gemv_reduce_probe {fp8|bf16|nvfp4} N".into()),
    };
    cutile::jit_cache::enable_default()?;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    let l2 = stream.device().l2_cache_size_bytes().unwrap_or(0);
    let scrub_len = ((2 * l2).max(64 << 20) / 4).next_multiple_of(SCRUB_TILE);
    let mut probe = Probe {
        buffer: Arc::new(api::zeros::<f32>(&[scrub_len]).sync_on(&stream)?),
        sums: api::zeros::<f32>(&[scrub_len / SCRUB_TILE]).sync_on(&stream)?,
        stream,
    };
    let mut rng = Lcg(rows as u64);
    for cols in [128, 256, 512, 1024, 2048, 4096, 8192] {
        match group {
            "fp8" => fp8_group(&mut probe, rows, cols, &mut rng)?,
            "bf16" => bf16_group(&mut probe, rows, cols, &mut rng)?,
            "nvfp4" => nvfp4_group(&mut probe, rows, cols, &mut rng)?,
            _ => return Err(format!("unknown group {group}").into()),
        }
    }
    Ok(())
}
