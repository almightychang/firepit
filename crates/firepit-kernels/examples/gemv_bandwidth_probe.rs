use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::TileKernel;
use half::bf16;
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// Same bytes, different numbers of CTAs: smaller row tiles, or K split across CTAs that write
// partial sums to separate rows of the output. NVFP4 expert-shaped kernels add one step of the
// body at a time. Kernel times come from nsys; each launch follows a read of twice the L2 size.
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

    #[cutile::entry()]
    pub fn load_split<const BN: i32, const KT: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT]);
        let mut acc: Tile<u32, { [BN, KT] }> = constant(0u32, shape![BN, KT]);
        let first = pid.0 * chunks;
        for kc in first..(first + chunks) {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = wp.load([pid.1, kc]);
            let bits: Tile<u8, { [BN, KT] }> = bitcast(weights);
            let wide: Tile<u32, { [BN, KT] }> = exti(bits);
            acc = acc + wide;
        }
        let sum: Tile<u32, { [BN] }> = reduce_sum(acc, 1i32);
        let out: Tile<f32, { [BN] }> = bitcast(sum);
        y.store(out.reshape(shape![1, BN]));
    }

    // dense_fp8_kernel with B = 1 and the scale-block loop split across pid.0.
    #[cutile::entry()]
    pub fn fp8_split<
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
        scale_cols: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT]);
        let x_part = x.partition(shape![B, NB, 32]);
        let scale_part = scales.partition(shape![1, 1]);
        let mut acc: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
        let scale_row = pid.1 * BN / BLOCK_N;
        let steps_per_scale = BLOCK_K / KT;
        let first = pid.0 * scale_cols;
        for scale_col in first..(first + scale_cols) {
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

    // expert_fp8_kernel's 2D body (no batch dimension, no expert indirection), split as above.
    #[cutile::entry()]
    pub fn fp8_2d_split<
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const BLOCK_N: i32,
        const BLOCK_K: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        scales: &Tensor<f32, { [-1, -1] }>,
        x: &Tensor<f32, { [1, -1, 32] }>,
        scale_cols: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT]);
        let x_part = x.partition(shape![1, NB, 32]);
        let scale_part = scales.partition(shape![1, 1]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        let scale_row = pid.1 * BN / BLOCK_N;
        let steps_per_scale = BLOCK_K / KT;
        let first = pid.0 * scale_cols;
        for scale_col in first..(first + scale_cols) {
            let scale: Tile<f32, { [1, 1] }> = scale_part.load([scale_row, scale_col]);
            let mut sum: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
            for step in 0i32..steps_per_scale {
                let kc = scale_col * steps_per_scale + step;
                let weights: Tile<f8e4m3fn, { [BN, KT] }> = w_part.load([pid.1, kc]);
                let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
                let xb: Tile<f32, { [1, NB, 32] }> = x_part.load([0, kc, 0]);
                let xb: Tile<f32, { [BN, KT] }> =
                    xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
                let products: Tile<f32, { [BN] }> = reduce_sum(weights * xb, 1i32);
                sum = sum + products.reshape(shape![1, BN]);
            }
            let scale: Tile<f32, { [1, BN] }> = scale.broadcast(shape![1, BN]);
            acc = acc + sum * scale;
        }
        y.store(acc);
    }

    // dense_gemv_kernel with T = bf16 and B = 1, the K loop split across pid.0.
    #[cutile::entry()]
    pub fn bf16_split<const B: i32, const BN: i32, const NB: i32>(
        y: &mut Tensor<f32, { [B, BN] }>,
        w: &Tensor<bf16, { [-1, -1, 32] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, NB, 32]);
        let x_part = x.partition(shape![B, NB, 32]);
        let mut acc: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
        let first = pid.0 * chunks;
        for kc in first..(first + chunks) {
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
    pub fn nvfp4_packed<const BN: i32, const KT_PACKED: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<u8, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let mut acc: Tile<u32, { [BN, KT_PACKED] }> = constant(0u32, shape![BN, KT_PACKED]);
        for kc in 0i32..chunks {
            let bits: Tile<u8, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let wide: Tile<u32, { [BN, KT_PACKED] }> = exti(bits);
            acc = acc + wide;
        }
        let sum: Tile<u32, { [BN] }> = reduce_sum(acc, 1i32);
        let out: Tile<f32, { [BN] }> = bitcast(sum);
        y.store(out.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn nvfp4_packed_scales<const BN: i32, const KT_PACKED: i32, const KT_GROUPS: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<u8, { [-1, -1] }>,
        scales: &Tensor<u8, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let sp = scales.partition(shape![BN, KT_GROUPS]);
        let mut acc: Tile<u32, { [BN, KT_PACKED] }> = constant(0u32, shape![BN, KT_PACKED]);
        let mut sacc: Tile<u32, { [BN, KT_GROUPS] }> = constant(0u32, shape![BN, KT_GROUPS]);
        for kc in 0i32..chunks {
            let bits: Tile<u8, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let wide: Tile<u32, { [BN, KT_PACKED] }> = exti(bits);
            acc = acc + wide;
            let sbits: Tile<u8, { [BN, KT_GROUPS] }> = sp.load([pid.1, kc]);
            let swide: Tile<u32, { [BN, KT_GROUPS] }> = exti(sbits);
            sacc = sacc + swide;
        }
        let sum: Tile<u32, { [BN] }> = reduce_sum(acc, 1i32);
        let ssum: Tile<u32, { [BN] }> = reduce_sum(sacc, 1i32);
        let out: Tile<f32, { [BN] }> = bitcast(sum.reshape(shape![BN]) + ssum.reshape(shape![BN]));
        y.store(out.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn nvfp4_unpacked<const BN: i32, const KT: i32, const KT_PACKED: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let mut acc: Tile<f32, { [BN, KT] }> = constant(0.0f32, shape![BN, KT]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let values: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let values: Tile<f32, { [BN, KT] }> = convert_tile(values);
            acc = acc + values;
        }
        let sum: Tile<f32, { [BN] }> = reduce_sum(acc, 1i32);
        y.store(sum.reshape(shape![1, BN]));
    }

    // expert_nvfp4_kernel without the expert indirection.
    #[cutile::entry()]
    pub fn nvfp4_product<
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

fn ctas(probe: &mut Probe, rows: usize, cols: usize) -> Result<()> {
    let stream = probe.stream.clone();
    let fp8 = upload(&stream, vec![f8e4m3fn(0x38); rows * cols], &[rows, cols])?;
    let scales = upload(
        &stream,
        vec![1.0f32; rows / SCALE_BLOCK * (cols / SCALE_BLOCK)],
        &[rows / SCALE_BLOCK, cols / SCALE_BLOCK],
    )?;
    let dense = upload(
        &stream,
        vec![bf16::from_f32(1.0); rows * cols],
        &[rows, cols / 32, 32],
    )?;
    let x = upload(&stream, vec![1.0f32; cols], &[1, cols / 32, 32])?;
    let chunks = cols / KT;
    let nb = (KT / 32).to_string();
    let mut runs: Vec<(usize, usize)> = [128, 64, 32, 16, 8]
        .into_iter()
        .filter(|&bn| rows % bn == 0)
        .map(|bn| (bn, 1))
        .collect();
    let product_bn = (rows / 64).clamp(8, 128).min(rows);
    for splits in [2, 4, 8] {
        if chunks % splits == 0 {
            runs.push((product_bn, splits));
        }
    }
    for (bn, splits) in runs {
        let mut y = api::zeros::<f32>(&[splits, rows]).sync_on(&stream)?;
        let per = (chunks / splits) as i32;
        let shape = format!(
            "N={rows} K={cols} BN={bn} split={splits} ctas={}",
            rows / bn * splits
        );
        probe.repeat(&format!("load {shape}"), |s| {
            probes::load_split((&mut y).partition([1, bn]), fp8.clone(), per)
                .generics(vec![bn.to_string(), KT.to_string()])
                .sync_on(s)?;
            Ok(())
        })?;
        probe.repeat(&format!("fp8 {shape}"), |s| {
            probes::fp8_split(
                (&mut y).partition([1, bn]),
                fp8.clone(),
                scales.clone(),
                x.clone(),
                per,
            )
            .generics(vec![
                "1".into(),
                bn.to_string(),
                nb.clone(),
                KT.to_string(),
                SCALE_BLOCK.to_string(),
                SCALE_BLOCK.to_string(),
            ])
            .sync_on(s)?;
            Ok(())
        })?;
        probe.repeat(&format!("fp8_2d {shape}"), |s| {
            probes::fp8_2d_split(
                (&mut y).partition([1, bn]),
                fp8.clone(),
                scales.clone(),
                x.clone(),
                per,
            )
            .generics(vec![
                bn.to_string(),
                nb.clone(),
                KT.to_string(),
                SCALE_BLOCK.to_string(),
                SCALE_BLOCK.to_string(),
            ])
            .sync_on(s)?;
            Ok(())
        })?;
        probe.repeat(&format!("bf16 {shape}"), |s| {
            probes::bf16_split((&mut y).partition([1, bn]), dense.clone(), x.clone(), per)
                .generics(vec!["1".into(), bn.to_string(), nb.clone()])
                .sync_on(s)?;
            Ok(())
        })?;
    }
    Ok(())
}

fn lm_head(probe: &mut Probe) -> Result<()> {
    let (rows, cols, bn) = (248320, 2048, 16);
    let stream = probe.stream.clone();
    let dense = upload(
        &stream,
        vec![bf16::from_f32(1.0); rows * cols],
        &[rows, cols / 32, 32],
    )?;
    let x = upload(&stream, vec![1.0f32; cols], &[1, cols / 32, 32])?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    probe.repeat(
        &format!("bf16 N={rows} K={cols} BN={bn} split=1 ctas={}", rows / bn),
        |s| {
            probes::bf16_split(
                (&mut y).partition([1, bn]),
                dense.clone(),
                x.clone(),
                (cols / KT) as i32,
            )
            .generics(vec!["1".into(), bn.to_string(), (KT / 32).to_string()])
            .sync_on(s)?;
            Ok(())
        },
    )
}

fn nvfp4(probe: &mut Probe, rows: usize, cols: usize) -> Result<()> {
    let bn = 128;
    let stream = probe.stream.clone();
    let bytes: Vec<u8> = (0..rows * cols / 2).map(|i| (i * 37 % 251) as u8).collect();
    let scale_bytes: Vec<u8> = (0..rows * cols / NVFP4_GROUP)
        .map(|i| 0x30 + (i % 16) as u8)
        .collect();
    let packed_u8 = upload(&stream, bytes.clone(), &[rows, cols / 2])?;
    let packed = upload(
        &stream,
        bytes.into_iter().map(f4e2m1fnx2::from_bits).collect(),
        &[rows, cols / 2],
    )?;
    let scales_u8 = upload(&stream, scale_bytes.clone(), &[rows, cols / NVFP4_GROUP])?;
    let scales = upload(
        &stream,
        scale_bytes.into_iter().map(f8e4m3fn).collect(),
        &[rows, cols / NVFP4_GROUP],
    )?;
    let row_scales = upload(&stream, vec![1.0f32; rows], &[1, rows])?;
    let x = upload(&stream, vec![1.0f32; cols], &[1, cols / 32, 32])?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let chunks = (cols / KT) as i32;
    let (kt, packed_kt, groups) = (
        KT.to_string(),
        (KT / 2).to_string(),
        (KT / NVFP4_GROUP).to_string(),
    );
    let shape = format!("N={rows} K={cols} BN={bn} split=1 ctas={}", rows / bn);
    probe.repeat(&format!("nv_packed {shape}"), |s| {
        probes::nvfp4_packed((&mut y).partition([1, bn]), packed_u8.clone(), chunks)
            .generics(vec![bn.to_string(), packed_kt.clone()])
            .sync_on(s)?;
        Ok(())
    })?;
    probe.repeat(&format!("nv_packed_scales {shape}"), |s| {
        probes::nvfp4_packed_scales(
            (&mut y).partition([1, bn]),
            packed_u8.clone(),
            scales_u8.clone(),
            chunks,
        )
        .generics(vec![bn.to_string(), packed_kt.clone(), groups.clone()])
        .sync_on(s)?;
        Ok(())
    })?;
    probe.repeat(&format!("nv_unpacked {shape}"), |s| {
        probes::nvfp4_unpacked((&mut y).partition([1, bn]), packed.clone(), chunks)
            .generics(vec![bn.to_string(), kt.clone(), packed_kt.clone()])
            .sync_on(s)?;
        Ok(())
    })?;
    probe.repeat(&format!("nv_product {shape}"), |s| {
        probes::nvfp4_product(
            (&mut y).partition([1, bn]),
            packed.clone(),
            scales.clone(),
            row_scales.clone(),
            x.clone(),
            (cols / 32) as i32,
        )
        .generics(vec![
            bn.to_string(),
            (KT / 32).to_string(),
            kt.clone(),
            packed_kt.clone(),
            groups.clone(),
        ])
        .sync_on(s)?;
        Ok(())
    })
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let group = std::env::args()
        .nth(1)
        .ok_or("usage: gemv_bandwidth_probe {ctas|nvfp4}")?;
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
        "ctas" => {
            for (rows, cols) in [(8192, 2048), (2048, 4096), (1024, 2048)] {
                ctas(&mut probe, rows, cols)?;
            }
            lm_head(&mut probe)
        }
        "nvfp4" => {
            for cols in [512, 1024, 2048, 4096] {
                nvfp4(&mut probe, 9216, cols)?;
            }
            nvfp4(&mut probe, 18432, 512)
        }
        _ => Err(format!("unknown group {group}").into()),
    }
}
