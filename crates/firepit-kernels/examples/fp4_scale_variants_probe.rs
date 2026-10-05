use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use half::bf16;
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// The NVFP4 matrix body without the input vector (row sums of E2M1 values times their E4M3
// group-of-16 scales), with only the way the scales are applied changed. Every output is
// checked against an f64 host reference from the ModelOpt definition. The process first
// allocates BALLAST_GIB one-GiB buffers, as a loaded model would; each launch follows a read
// of a 64 MiB buffer to evict L2 with clean lines; time the kernels with nsys.
const LAUNCHES: usize = 10;
const SCRUB_TILE: usize = 4096;
const SCRUB_LEN: usize = 16 << 20;
const KT: usize = 128;
const GROUP: usize = 16;
const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

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

    // Floor: unpack, convert, group-of-16 sums, no scales.
    #[cutile::entry()]
    pub fn groups_only<const BN: i32, const KT: i32, const KT_PACKED: i32, const KT_GROUPS: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let mut acc: Tile<f32, { [BN, KT_GROUPS] }> = constant(0.0f32, shape![BN, KT_GROUPS]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let values: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let values: Tile<f32, { [BN, KT] }> = convert_tile(values);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> =
                reduce_sum(values.reshape(shape![BN, KT_GROUPS, 16]), 2i32);
            acc = acc + groups.reshape(shape![BN, KT_GROUPS]);
        }
        let sum: Tile<f32, { [BN] }> = reduce_sum(acc, 1i32);
        y.store(sum.reshape(shape![1, BN]));
    }

    // (a) group sums times the converted scale tile, reduced per row.
    #[cutile::entry()]
    pub fn scale_groups<
        const BN: i32,
        const KT: i32,
        const KT_PACKED: i32,
        const KT_GROUPS: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let sp = scales.partition(shape![BN, KT_GROUPS]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let values: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let values: Tile<f32, { [BN, KT] }> = convert_tile(values);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> =
                reduce_sum(values.reshape(shape![BN, KT_GROUPS, 16]), 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> = sp.load([pid.1, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let part: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    // (b) the scale tile expanded to [BN, KT] and multiplied element-wise, one reduction.
    #[cutile::entry()]
    pub fn scale_expanded<
        const BN: i32,
        const KT: i32,
        const KT_PACKED: i32,
        const KT_GROUPS: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let sp = scales.partition(shape![BN, KT_GROUPS]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let values: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let values: Tile<f32, { [BN, KT] }> = convert_tile(values);
            let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> = sp.load([pid.1, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let scale: Tile<f32, { [BN, KT] }> = scale
                .reshape(shape![BN, KT_GROUPS, 1])
                .broadcast(shape![BN, KT_GROUPS, 16])
                .reshape(shape![BN, KT]);
            let part: Tile<f32, { [BN] }> = reduce_sum(values * scale, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    // (c) the scales read as bytes and decoded from E4M3 with integer operations.
    #[cutile::entry()]
    pub fn scale_bits<const BN: i32, const KT: i32, const KT_PACKED: i32, const KT_GROUPS: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<u8, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let sp = scales.partition(shape![BN, KT_GROUPS]);
        let three: Tile<u32, { [BN, KT_GROUPS] }> = constant(3u32, shape![BN, KT_GROUPS]);
        let seven: Tile<u32, { [BN, KT_GROUPS] }> = constant(7u32, shape![BN, KT_GROUPS]);
        let fifteen: Tile<u32, { [BN, KT_GROUPS] }> = constant(15u32, shape![BN, KT_GROUPS]);
        let bias: Tile<u32, { [BN, KT_GROUPS] }> = constant(120u32, shape![BN, KT_GROUPS]);
        let exponent_shift: Tile<u32, { [BN, KT_GROUPS] }> = constant(23u32, shape![BN, KT_GROUPS]);
        let mantissa_shift: Tile<u32, { [BN, KT_GROUPS] }> = constant(20u32, shape![BN, KT_GROUPS]);
        let zero: Tile<u32, { [BN, KT_GROUPS] }> = constant(0u32, shape![BN, KT_GROUPS]);
        let sign_shift: Tile<u32, { [BN, KT_GROUPS] }> = constant(31u32, shape![BN, KT_GROUPS]);
        let subnormal_step: Tile<f32, { [BN, KT_GROUPS] }> =
            constant(0.001953125f32, shape![BN, KT_GROUPS]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let values: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let values: Tile<f32, { [BN, KT] }> = convert_tile(values);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> =
                reduce_sum(values.reshape(shape![BN, KT_GROUPS, 16]), 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let bytes: Tile<u8, { [BN, KT_GROUPS] }> = sp.load([pid.1, kc]);
            let bytes: Tile<u32, { [BN, KT_GROUPS] }> = exti(bytes);
            let exponent: Tile<u32, { [BN, KT_GROUPS] }> = (bytes >> three) & fifteen;
            let mantissa: Tile<u32, { [BN, KT_GROUPS] }> = bytes & seven;
            let normal: Tile<f32, { [BN, KT_GROUPS] }> =
                bitcast(((exponent + bias) << exponent_shift) | (mantissa << mantissa_shift));
            let mantissa_value: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(mantissa);
            let subnormal: Tile<f32, { [BN, KT_GROUPS] }> = mantissa_value * subnormal_step;
            let magnitude: Tile<f32, { [BN, KT_GROUPS] }> =
                select(eq_tile(exponent, zero), subnormal, normal);
            let magnitude: Tile<u32, { [BN, KT_GROUPS] }> = bitcast(magnitude);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> =
                bitcast(magnitude | ((bytes >> seven) << sign_shift));
            let part: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    // (e) scales converted ahead of time and stored as f32.
    #[cutile::entry()]
    pub fn scale_f32<const BN: i32, const KT: i32, const KT_PACKED: i32, const KT_GROUPS: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f32, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let sp = scales.partition(shape![BN, KT_GROUPS]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let values: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let values: Tile<f32, { [BN, KT] }> = convert_tile(values);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> =
                reduce_sum(values.reshape(shape![BN, KT_GROUPS, 16]), 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = sp.load([pid.1, kc]);
            let part: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    // (e2) scales converted ahead of time and stored as bf16 (exact: E4M3 fits in bf16).
    #[cutile::entry()]
    pub fn scale_bf16<const BN: i32, const KT: i32, const KT_PACKED: i32, const KT_GROUPS: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<bf16, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let sp = scales.partition(shape![BN, KT_GROUPS]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let values: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let values: Tile<f32, { [BN, KT] }> = convert_tile(values);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> =
                reduce_sum(values.reshape(shape![BN, KT_GROUPS, 16]), 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let scale: Tile<bf16, { [BN, KT_GROUPS] }> = sp.load([pid.1, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let part: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
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

fn e2m1(code: u8) -> f64 {
    let magnitude = E2M1[(code & 0x7) as usize];
    if code & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn e4m3(code: u8) -> f64 {
    let (exponent, mantissa) = ((code >> 3) & 0xf, (code & 0x7) as f64);
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f64.powi(exponent as i32 - 7)
    };
    if code & 0x80 != 0 {
        -magnitude
    } else {
        magnitude
    }
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
        println!("{label}");
        for _ in 0..LAUNCHES {
            kernels::scrub_read((&mut self.sums).partition([1]), self.buffer.clone())
                .generics(vec![SCRUB_TILE.to_string()])
                .sync_on(&self.stream)?;
            launch(&self.stream)?;
        }
        Ok(())
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

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse())
        .collect::<std::result::Result<_, _>>()?;
    let [rows, cols, bn, ballast_gib] = args[..] else {
        return Err("usage: fp4_scale_variants_probe N K BN BALLAST_GIB".into());
    };
    if rows % bn != 0 || cols % KT != 0 {
        return Err("BN must divide N and 128 must divide K".into());
    }
    cutile::jit_cache::enable_default()?;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    let ballast: Vec<Tensor<f32>> = (0..ballast_gib)
        .map(|_| api::zeros::<f32>(&[1 << 28]).sync_on(&stream))
        .collect::<std::result::Result<_, _>>()?;
    let mut probe = Probe {
        buffer: Arc::new(api::zeros::<f32>(&[SCRUB_LEN]).sync_on(&stream)?),
        sums: api::zeros::<f32>(&[SCRUB_LEN / SCRUB_TILE]).sync_on(&stream)?,
        stream: stream.clone(),
    };

    // Random E2M1 pairs and finite E4M3 scales (positive and negative, normal and subnormal).
    let mut rng = Lcg((rows * cols) as u64);
    let packed: Vec<u8> = (0..rows * cols / 2).map(|_| rng.next()).collect();
    let scale_codes: Vec<u8> = (0..rows * cols / GROUP)
        .map(|_| rng.next() & 0xbf)
        .collect();
    let reference: Vec<f64> = (0..rows)
        .map(|r| {
            (0..cols)
                .map(|k| {
                    let byte = packed[(r * cols + k) / 2];
                    let code = if k % 2 == 0 { byte & 0xf } else { byte >> 4 };
                    e2m1(code) * e4m3(scale_codes[(r * cols + k) / GROUP])
                })
                .sum()
        })
        .collect();
    let w = upload(
        &stream,
        packed.into_iter().map(f4e2m1fnx2::from_bits).collect(),
        &[rows, cols / 2],
    )?;
    let s_e4m3 = upload(
        &stream,
        scale_codes.iter().map(|&c| f8e4m3fn(c)).collect(),
        &[rows, cols / GROUP],
    )?;
    let s_u8 = upload(&stream, scale_codes.clone(), &[rows, cols / GROUP])?;
    let s_f32 = upload(
        &stream,
        scale_codes.iter().map(|&c| e4m3(c) as f32).collect(),
        &[rows, cols / GROUP],
    )?;
    let s_bf16 = upload(
        &stream,
        scale_codes
            .iter()
            .map(|&c| bf16::from_f64(e4m3(c)))
            .collect(),
        &[rows, cols / GROUP],
    )?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let chunks = (cols / KT) as i32;
    let generics = vec![
        bn.to_string(),
        KT.to_string(),
        (KT / 2).to_string(),
        (KT / GROUP).to_string(),
    ];
    let shape = format!("N={rows} K={cols} BN={bn} ballast={ballast_gib}");
    let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let check = |name: &str, y: &Tensor<f32>| -> Result<()> {
        let got = y.dup().to_host_vec().sync_on(&stream)?;
        let error = got
            .iter()
            .zip(&reference)
            .fold(0.0f64, |m, (&g, &r)| m.max((g as f64 - r).abs()))
            / scale;
        println!("CHECK {name} {shape} max_rel_vs_reference {error:.3e}");
        Ok(())
    };

    probe.repeat(&format!("RUN groups_only {shape}"), |s| {
        kernels::groups_only((&mut y).partition([1, bn]), w.clone(), chunks)
            .generics(generics.clone())
            .sync_on(s)?;
        Ok(())
    })?;
    probe.repeat(&format!("RUN scale_groups {shape}"), |s| {
        kernels::scale_groups(
            (&mut y).partition([1, bn]),
            w.clone(),
            s_e4m3.clone(),
            chunks,
        )
        .generics(generics.clone())
        .sync_on(s)?;
        Ok(())
    })?;
    check("scale_groups", &y)?;
    probe.repeat(&format!("RUN scale_expanded {shape}"), |s| {
        kernels::scale_expanded(
            (&mut y).partition([1, bn]),
            w.clone(),
            s_e4m3.clone(),
            chunks,
        )
        .generics(generics.clone())
        .sync_on(s)?;
        Ok(())
    })?;
    check("scale_expanded", &y)?;
    probe.repeat(&format!("RUN scale_bits {shape}"), |s| {
        kernels::scale_bits((&mut y).partition([1, bn]), w.clone(), s_u8.clone(), chunks)
            .generics(generics.clone())
            .sync_on(s)?;
        Ok(())
    })?;
    check("scale_bits", &y)?;
    probe.repeat(&format!("RUN scale_f32 {shape}"), |s| {
        kernels::scale_f32(
            (&mut y).partition([1, bn]),
            w.clone(),
            s_f32.clone(),
            chunks,
        )
        .generics(generics.clone())
        .sync_on(s)?;
        Ok(())
    })?;
    check("scale_f32", &y)?;
    probe.repeat(&format!("RUN scale_bf16 {shape}"), |s| {
        kernels::scale_bf16(
            (&mut y).partition([1, bn]),
            w.clone(),
            s_bf16.clone(),
            chunks,
        )
        .generics(generics.clone())
        .sync_on(s)?;
        Ok(())
    })?;
    check("scale_bf16", &y)?;
    drop(ballast);
    Ok(())
}
