use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::gemv::{GemvTile, fp8_tile, gemv};
use firepit_kernels::sink::{Eager, Graph};
use firepit_kernels::weights::{DenseMatrix, HostFp8Matrix, Weight};
use half::bf16;
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// Three open questions on the decode GEMVs: the FP8 kernel with a per-tensor scale (one scale
// block spanning the whole matrix) against 128-blocks and against a flat K loop that applies a
// row-scale vector once at the end; the NVFP4 expert body with pipelining hints, without the
// input, and with f16 conversion; and lm_head replayed from a CUDA graph against eager launches.
// Kernel times come from nsys; each launch follows a read of twice the L2 size.
const WARMUP: usize = 2;
const REPS: usize = 8;
const SCRUB_TILE: usize = 4096;
const KT: usize = 128;
const SCALE_BLOCK: usize = 128;
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

    // dense_fp8_kernel's body with the scale loop removed: one K loop, then one row scale.
    #[cutile::entry()]
    pub fn fp8_flat<const B: i32, const BN: i32, const NB: i32, const KT: i32>(
        y: &mut Tensor<f32, { [B, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT]);
        let x_part = x.partition(shape![B, NB, 32]);
        let mut acc: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
        for kc in 0i32..chunks {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = w_part.load([pid.1, kc]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let weights: Tile<f32, { [B, BN, KT] }> = weights
                .reshape(shape![1, BN, KT])
                .broadcast(shape![B, BN, KT]);
            let xb: Tile<f32, { [B, NB, 32] }> = x_part.load([0, kc, 0]);
            let xb: Tile<f32, { [B, BN, KT] }> =
                xb.reshape(shape![B, 1, KT]).broadcast(shape![B, BN, KT]);
            let part: Tile<f32, { [B, BN] }> = reduce_sum(weights * xb, 2i32);
            acc = acc + part.reshape(shape![B, BN]);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, pid.1]);
        y.store(acc * row_scale.broadcast(shape![B, BN]));
    }

    // expert_nvfp4_kernel without the expert indirection.
    #[cutile::entry()]
    pub fn nv_product<
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
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..chunks {
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
    pub fn nv_pipelined<
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const KT_PACKED: i32,
        const KT_GROUPS: i32,
        const LATENCY: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        x: &Tensor<f32, { [1, -1, 32] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> =
                w_part.load_pipelined::<LATENCY>([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([0, kc, 0]);
            let xb: Tile<f32, { [BN, KT] }> = xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
            let products: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                (weights * xb).reshape(shape![BN, KT_GROUPS, 16]);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = reduce_sum(products, 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> =
                scale_part.load_pipelined::<LATENCY>([pid.1, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let sum: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
            acc = acc + sum.reshape(shape![1, BN]);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, pid.1]);
        y.store(acc * row_scale);
    }

    #[cutile::entry()]
    pub fn nv_no_input<const BN: i32, const KT: i32, const KT_PACKED: i32, const KT_GROUPS: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = w_part.load([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let products: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                weights.reshape(shape![BN, KT_GROUPS, 16]);
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
    pub fn nv_via_f16<
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
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = w_part.load([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f16, { [BN, KT] }> = convert_tile(weights);
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
    pub fn nv_f16_multiply<
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
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let x_part = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = w_part.load([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f16, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([0, kc, 0]);
            let xb: Tile<f16, { [1, NB, 32] }> = convert_tile(xb);
            let xb: Tile<f16, { [BN, KT] }> = xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
            let products: Tile<f32, { [BN, KT] }> = convert_tile(weights * xb);
            let products: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                products.reshape(shape![BN, KT_GROUPS, 16]);
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
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 56) as u8
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

struct Probe {
    stream: Arc<Stream>,
    buffer: Arc<Tensor<f32>>,
    sums: Tensor<f32>,
}

impl Probe {
    fn scrub(&mut self) -> Result<()> {
        probes::scrub_read((&mut self.sums).partition([1]), self.buffer.clone())
            .generics(vec![SCRUB_TILE.to_string()])
            .sync_on(&self.stream)?;
        Ok(())
    }

    fn repeat(
        &mut self,
        label: &str,
        mut launch: impl FnMut(&Arc<Stream>) -> Result<()>,
    ) -> Result<()> {
        println!("CFG {label} launches={}", WARMUP + REPS);
        for _ in 0..WARMUP + REPS {
            self.scrub()?;
            launch(&self.stream)?;
        }
        Ok(())
    }
}

fn fp8(probe: &mut Probe) -> Result<()> {
    let stream = probe.stream.clone();
    let mut rng = Lcg(3);
    for (rows, cols) in [(8192, 2048), (4096, 2048), (2048, 4096), (512, 2048)] {
        let bytes: Vec<f8e4m3fn> = (0..rows * cols)
            .map(|_| f8e4m3fn(rng.next() & 0x3f | 0x20))
            .collect();
        let x_values: Vec<f32> = (0..cols)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0)
            .collect();
        let x = api::copy_host_vec_to_device(&Arc::new(x_values))
            .reshape(&[1, cols / 32, 32])
            .sync_on(&stream)?;
        let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
        let per_tensor = Weight::Fp8(
            HostFp8Matrix::from_parts(bytes.clone(), vec![0.5], rows, cols, rows, cols)?
                .upload(&stream)?,
        );
        let blocks = Weight::Fp8(
            HostFp8Matrix::from_parts(
                bytes.clone(),
                vec![0.5; rows / SCALE_BLOCK * (cols / SCALE_BLOCK)],
                rows,
                cols,
                SCALE_BLOCK,
                SCALE_BLOCK,
            )?
            .upload(&stream)?,
        );
        let tile = fp8_tile(rows, cols, rows, cols, 1);
        let block_tile = fp8_tile(rows, cols, SCALE_BLOCK, SCALE_BLOCK, 1);
        let shape = format!("N={rows} K={cols} BN={}", tile.rows);
        let mut outputs = Vec::new();
        for (label, weight, tile) in [
            ("fp8-per-tensor", &per_tensor, tile),
            ("fp8-blocks", &blocks, block_tile),
        ] {
            probe.repeat(&format!("{label} {shape}"), |s| {
                gemv(
                    weight,
                    &x.view(&[1, cols / 32, 32])?,
                    (&mut y).partition([1, tile.rows]),
                    tile,
                )
                .submit(&mut Eager(s))?;
                Ok(())
            })?;
            outputs.push(y.dup().to_host_vec().sync_on(&stream)?);
        }
        let w = upload(&stream, bytes, &[rows, cols])?;
        let row_scales = upload(&stream, vec![0.5f32; rows], &[1, rows])?;
        let xs = Arc::new(x.dup().sync_on(&stream)?);
        probe.repeat(&format!("fp8-flat {shape}"), |s| {
            probes::fp8_flat(
                (&mut y).partition([1, tile.rows]),
                w.clone(),
                row_scales.clone(),
                xs.clone(),
                (cols / KT) as i32,
            )
            .generics(vec![
                "1".into(),
                tile.rows.to_string(),
                (KT / 32).to_string(),
                KT.to_string(),
            ])
            .sync_on(s)?;
            Ok(())
        })?;
        let flat = y.dup().to_host_vec().sync_on(&stream)?;
        println!(
            "CHECK fp8 {shape} blocks-vs-per-tensor {:.3e} flat-vs-per-tensor {:.3e}",
            max_relative(&outputs[1], &outputs[0]),
            max_relative(&flat, &outputs[0])
        );
    }
    Ok(())
}

fn nvfp4(probe: &mut Probe, rows: usize, cols: usize) -> Result<()> {
    let bn = 128;
    let stream = probe.stream.clone();
    let mut rng = Lcg(rows as u64 + cols as u64);
    let w = upload(
        &stream,
        (0..rows * cols / 2)
            .map(|_| f4e2m1fnx2::from_bits(rng.next()))
            .collect(),
        &[rows, cols / 2],
    )?;
    let s = upload(
        &stream,
        (0..rows * cols / NVFP4_GROUP)
            .map(|_| f8e4m3fn(0x30 + rng.next() % 0x10))
            .collect(),
        &[rows, cols / NVFP4_GROUP],
    )?;
    let row_scales = upload(&stream, vec![1.0f32; rows], &[1, rows])?;
    let x = upload(
        &stream,
        (0..cols)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0)
            .collect(),
        &[1, cols / 32, 32],
    )?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let chunks = (cols / KT) as i32;
    let shape = format!("N={rows} K={cols} BN={bn}");
    let base = vec![
        bn.to_string(),
        (KT / 32).to_string(),
        KT.to_string(),
        (KT / 2).to_string(),
        (KT / NVFP4_GROUP).to_string(),
    ];
    probe.repeat(&format!("nv-product {shape}"), |st| {
        probes::nv_product(
            (&mut y).partition([1, bn]),
            w.clone(),
            s.clone(),
            row_scales.clone(),
            x.clone(),
            chunks,
        )
        .generics(base.clone())
        .sync_on(st)?;
        Ok(())
    })?;
    let reference = y.dup().to_host_vec().sync_on(&stream)?;
    for latency in [2, 4, 8] {
        let mut generics = base.clone();
        generics.push(latency.to_string());
        probe.repeat(&format!("nv-pipelined{latency} {shape}"), |st| {
            probes::nv_pipelined(
                (&mut y).partition([1, bn]),
                w.clone(),
                s.clone(),
                row_scales.clone(),
                x.clone(),
                chunks,
            )
            .generics(generics.clone())
            .sync_on(st)?;
            Ok(())
        })?;
    }
    probe.repeat(&format!("nv-no-input {shape}"), |st| {
        probes::nv_no_input(
            (&mut y).partition([1, bn]),
            w.clone(),
            s.clone(),
            row_scales.clone(),
            chunks,
        )
        .generics(vec![
            bn.to_string(),
            KT.to_string(),
            (KT / 2).to_string(),
            (KT / NVFP4_GROUP).to_string(),
        ])
        .sync_on(st)?;
        Ok(())
    })?;
    probe.repeat(&format!("nv-via-f16 {shape}"), |st| {
        probes::nv_via_f16(
            (&mut y).partition([1, bn]),
            w.clone(),
            s.clone(),
            row_scales.clone(),
            x.clone(),
            chunks,
        )
        .generics(base.clone())
        .sync_on(st)?;
        Ok(())
    })?;
    let via_f16 = y.dup().to_host_vec().sync_on(&stream)?;
    probe.repeat(&format!("nv-f16-multiply {shape}"), |st| {
        probes::nv_f16_multiply(
            (&mut y).partition([1, bn]),
            w.clone(),
            s.clone(),
            row_scales.clone(),
            x.clone(),
            chunks,
        )
        .generics(base.clone())
        .sync_on(st)?;
        Ok(())
    })?;
    let f16_multiply = y.dup().to_host_vec().sync_on(&stream)?;
    println!(
        "CHECK nvfp4 {shape} via-f16 {:.3e} f16-multiply {:.3e}",
        max_relative(&via_f16, &reference),
        max_relative(&f16_multiply, &reference)
    );
    Ok(())
}

fn lm_head(probe: &mut Probe) -> Result<()> {
    let (rows, cols) = (248320, 2048);
    let stream = probe.stream.clone();
    let weight = Weight::Dense(DenseMatrix::upload(
        vec![bf16::from_f32(1.0); rows * cols],
        rows,
        cols,
        &stream,
    )?);
    let tile = GemvTile {
        rows: 16,
        blocks: KT / 32,
    };
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
        .reshape(&[1, cols / 32, 32])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let view = x.view(&[1, cols / 32, 32])?;
    probe.repeat(&format!("lmhead-eager N={rows} K={cols} BN=16"), |s| {
        gemv(&weight, &view, (&mut y).partition([1, tile.rows]), tile).submit(&mut Eager(s))?;
        Ok(())
    })?;
    let graph = CudaGraph::scope(&stream, |scope| {
        gemv(&weight, &view, (&mut y).partition([1, tile.rows]), tile)
            .submit(&mut Graph(scope))
            .map_err(|e| cuda_async::error::DeviceError::Internal(e.to_string()))
    })?;
    probe.repeat(&format!("lmhead-graph N={rows} K={cols} BN=16"), |s| {
        graph.launch().sync_on(s)?;
        Ok(())
    })
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let group = std::env::args()
        .nth(1)
        .ok_or("usage: gemv_residual_probe {fp8|nvfp4|lmhead}")?;
    cutile::jit_cache::enable_default()?;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    let l2 = stream.device().l2_cache_size_bytes().unwrap_or(0);
    let scrub_len = ((2 * l2).max(64 << 20) / 4).next_multiple_of(SCRUB_TILE);
    let mut probe = Probe {
        buffer: Arc::new(api::zeros::<f32>(&[scrub_len]).sync_on(&stream)?),
        sums: api::zeros::<f32>(&[scrub_len / SCRUB_TILE]).sync_on(&stream)?,
        stream,
    };
    match group.as_str() {
        "fp8" => fp8(&mut probe),
        "nvfp4" => {
            for cols in [512, 2048, 4096] {
                nvfp4(&mut probe, 9216, cols)?;
            }
            nvfp4(&mut probe, 18432, 512)
        }
        "lmhead" => lm_head(&mut probe),
        _ => Err(format!("unknown group {group}").into()),
    }
}
