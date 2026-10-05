use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit::{Checkpoint, Dtype};
use firepit_kernels::gemv::{GemvTile, gemv};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{DenseMatrix, HostFp8Matrix, Weight};
use half::bf16;
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// The same GEMV kernels on the same shapes with only the weight values changed: zero, a
// constant, a repeating pattern, random bytes, and real checkpoint weights. Also an NVFP4
// expert body that decodes E2M1 with integer bit operations instead of unpack, and the FP8
// product body at row tiles 128 down to 8 for SASS. Kernel times come from nsys; each launch
// follows a read of twice the L2 size.
const WARMUP: usize = 2;
const REPS: usize = 8;
const SCRUB_TILE: usize = 4096;
const KT: usize = 128;
const SCALE_BLOCK: usize = 128;
const NVFP4_GROUP: usize = 16;
const FP8_DIR: &str = "/work/qwen3.6-35b-a3b-fp8";
const BF16_DIR: &str = "/work/qwen3.6-35b-a3b";
const NVFP4_DIR: &str = "/work/firepit-home/models/qwen3.6/nvfp4";

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
    pub fn load_only<const BN: i32, const KT: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<u8, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT]);
        let mut acc: Tile<u32, { [BN, KT] }> = constant(0u32, shape![BN, KT]);
        for kc in 0i32..chunks {
            let bits: Tile<u8, { [BN, KT] }> = wp.load([pid.1, kc]);
            let wide: Tile<u32, { [BN, KT] }> = exti(bits);
            acc = acc + wide;
        }
        let sum: Tile<u32, { [BN] }> = reduce_sum(acc, 1i32);
        let out: Tile<f32, { [BN] }> = bitcast(sum);
        y.store(out.reshape(shape![1, BN]));
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

    // The same product with E2M1 decoded from the packed bytes by integer bit operations:
    // the low nibble is the even element and the high nibble the odd one, so the input is
    // read as pairs and split into its even and odd halves instead of interleaving weights.
    #[cutile::entry()]
    pub fn nvfp4_bitdecode<
        const BN: i32,
        const KT_PACKED: i32,
        const KT_GROUPS: i32,
        const PAIRS_PER_GROUP: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<u8, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        x: &Tensor<f32, { [1, -1, 2] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT_PACKED]);
        let scale_part = scales.partition(shape![BN, KT_GROUPS]);
        let x_part = x.partition(shape![1, KT_PACKED, 2]);
        let lanes: Tile<i32, { [2] }> = iota(shape![2]);
        let lanes: Tile<i32, { [1, 1, 2] }> = lanes.reshape(shape![1, 1, 2]);
        let even: Tile<bool, { [1, KT_PACKED, 2] }> = eq_tile(
            lanes.broadcast(shape![1, KT_PACKED, 2]),
            0i32.broadcast(shape![1, KT_PACKED, 2]),
        );
        let zero_x: Tile<f32, { [1, KT_PACKED, 2] }> = constant(0.0f32, shape![1, KT_PACKED, 2]);
        let one: Tile<u32, { [BN, KT_PACKED] }> = constant(1u32, shape![BN, KT_PACKED]);
        let three: Tile<u32, { [BN, KT_PACKED] }> = constant(3u32, shape![BN, KT_PACKED]);
        let four: Tile<u32, { [BN, KT_PACKED] }> = constant(4u32, shape![BN, KT_PACKED]);
        let fifteen: Tile<u32, { [BN, KT_PACKED] }> = constant(15u32, shape![BN, KT_PACKED]);
        let bias: Tile<u32, { [BN, KT_PACKED] }> = constant(126u32, shape![BN, KT_PACKED]);
        let exponent_shift: Tile<u32, { [BN, KT_PACKED] }> = constant(23u32, shape![BN, KT_PACKED]);
        let mantissa_shift: Tile<u32, { [BN, KT_PACKED] }> = constant(22u32, shape![BN, KT_PACKED]);
        let sign_shift: Tile<u32, { [BN, KT_PACKED] }> = constant(31u32, shape![BN, KT_PACKED]);
        let half_bits: Tile<u32, { [BN, KT_PACKED] }> =
            constant(0x3f00_0000u32, shape![BN, KT_PACKED]);
        let zero_bits: Tile<u32, { [BN, KT_PACKED] }> = constant(0u32, shape![BN, KT_PACKED]);
        let mut acc: Tile<f32, { [1, BN] }> = constant(0.0f32, shape![1, BN]);
        for kc in 0i32..chunks {
            let bytes: Tile<u8, { [BN, KT_PACKED] }> = w_part.load([pid.1, kc]);
            let bytes: Tile<u32, { [BN, KT_PACKED] }> = exti(bytes);
            let low: Tile<u32, { [BN, KT_PACKED] }> = bytes & fifteen;
            let high: Tile<u32, { [BN, KT_PACKED] }> = (bytes >> four) & fifteen;

            let exponent: Tile<u32, { [BN, KT_PACKED] }> = (low >> one) & three;
            let mantissa: Tile<u32, { [BN, KT_PACKED] }> = low & one;
            let normal: Tile<u32, { [BN, KT_PACKED] }> =
                ((exponent + bias) << exponent_shift) | (mantissa << mantissa_shift);
            let subnormal: Tile<u32, { [BN, KT_PACKED] }> =
                select(eq_tile(mantissa, one), half_bits, zero_bits);
            let magnitude: Tile<u32, { [BN, KT_PACKED] }> =
                select(eq_tile(exponent, zero_bits), subnormal, normal);
            let low_bits: Tile<u32, { [BN, KT_PACKED] }> =
                magnitude | ((low >> three) << sign_shift);
            let low_values: Tile<f32, { [BN, KT_PACKED] }> = bitcast(low_bits);

            let exponent: Tile<u32, { [BN, KT_PACKED] }> = (high >> one) & three;
            let mantissa: Tile<u32, { [BN, KT_PACKED] }> = high & one;
            let normal: Tile<u32, { [BN, KT_PACKED] }> =
                ((exponent + bias) << exponent_shift) | (mantissa << mantissa_shift);
            let subnormal: Tile<u32, { [BN, KT_PACKED] }> =
                select(eq_tile(mantissa, one), half_bits, zero_bits);
            let magnitude: Tile<u32, { [BN, KT_PACKED] }> =
                select(eq_tile(exponent, zero_bits), subnormal, normal);
            let high_bits: Tile<u32, { [BN, KT_PACKED] }> =
                magnitude | ((high >> three) << sign_shift);
            let high_values: Tile<f32, { [BN, KT_PACKED] }> = bitcast(high_bits);

            let pairs: Tile<f32, { [1, KT_PACKED, 2] }> = x_part.load([0, kc, 0]);
            let x_even: Tile<f32, { [1, KT_PACKED] }> =
                reduce_sum(select(even, pairs, zero_x), 2i32);
            let x_even: Tile<f32, { [1, KT_PACKED] }> = x_even.reshape(shape![1, KT_PACKED]);
            let x_odd: Tile<f32, { [1, KT_PACKED] }> =
                reduce_sum(select(even, zero_x, pairs), 2i32);
            let x_odd: Tile<f32, { [1, KT_PACKED] }> = x_odd.reshape(shape![1, KT_PACKED]);
            let products: Tile<f32, { [BN, KT_PACKED] }> = low_values
                * x_even.broadcast(shape![BN, KT_PACKED])
                + high_values * x_odd.broadcast(shape![BN, KT_PACKED]);
            let products: Tile<f32, { [BN, KT_GROUPS, PAIRS_PER_GROUP] }> =
                products.reshape(shape![BN, KT_GROUPS, PAIRS_PER_GROUP]);
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

// Byte sources with the FP8 NaN codes (0x7f, 0xff) moved to a finite value.
fn bytes(kind: &str, len: usize, constant: u8) -> Vec<u8> {
    let finite = |b: u8| if b & 0x7f == 0x7f { b & 0xbe } else { b };
    let mut rng = Lcg(len as u64);
    match kind {
        "zero" => vec![0; len],
        "const" => vec![constant; len],
        "pattern" => (0..len).map(|i| finite((i * 37 % 251) as u8)).collect(),
        "random" => (0..len).map(|_| finite(rng.next())).collect(),
        _ => unreachable!(),
    }
}

fn find(checkpoint: &Checkpoint, dtype: Dtype, shape: &[usize], contains: &str) -> Result<String> {
    let mut names: Vec<&str> = checkpoint
        .names()
        .filter(|name| name.ends_with(".weight") && name.contains(contains))
        .collect();
    names.sort();
    for name in names {
        let (stored, dims) = checkpoint.stored(name)?;
        if stored == dtype && dims == shape {
            return Ok(name.to_string());
        }
    }
    Err(format!("no {dtype:?} tensor {shape:?} containing {contains:?}").into())
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

fn product(
    probe: &mut Probe,
    label: &str,
    weight: &Weight,
    rows: usize,
    cols: usize,
    tile: GemvTile,
) -> Result<()> {
    let stream = probe.stream.clone();
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
        .reshape(&[1, cols / 32, 32])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    probe.repeat(label, |s| {
        gemv(
            weight,
            &x.view(&[1, cols / 32, 32])?,
            (&mut y).partition([1, tile.rows]),
            tile,
        )
        .submit(&mut Eager(s))?;
        Ok(())
    })
}

fn fp8(probe: &mut Probe) -> Result<()> {
    let (rows, cols, bn) = (8192, 2048, 128);
    let stream = probe.stream.clone();
    let checkpoint = Checkpoint::open(FP8_DIR)?;
    let name = find(&checkpoint, Dtype::F8_E4M3, &[rows, cols], "layers.3.")?;
    println!("REAL fp8 {name}");
    let real = checkpoint
        .tensor_fp8(&name)?
        .data
        .into_iter()
        .map(|b| b.0)
        .collect::<Vec<u8>>();
    let real_scales: Vec<f32> = checkpoint
        .tensor_bf16(&name.replace(".weight", ".weight_scale_inv"))?
        .data
        .into_iter()
        .map(|s| s.to_f32())
        .collect();
    for kind in ["zero", "const", "pattern", "random", "real"] {
        let data = if kind == "real" {
            real.clone()
        } else {
            bytes(kind, rows * cols, 0x38)
        };
        let scales = if kind == "real" {
            real_scales.clone()
        } else {
            vec![1.0; rows / SCALE_BLOCK * (cols / SCALE_BLOCK)]
        };
        let w = upload(&stream, data.clone(), &[rows, cols])?;
        let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
        probe.repeat(&format!("fp8-load {kind} N={rows} K={cols} BN={bn}"), |s| {
            probes::load_only((&mut y).partition([1, bn]), w.clone(), (cols / KT) as i32)
                .generics(vec![bn.to_string(), KT.to_string()])
                .sync_on(s)?;
            Ok(())
        })?;
        let weight = Weight::Fp8(
            HostFp8Matrix::from_parts(
                data.into_iter().map(f8e4m3fn).collect(),
                scales,
                rows,
                cols,
                SCALE_BLOCK,
                SCALE_BLOCK,
            )?
            .upload(&stream)?,
        );
        product(
            probe,
            &format!("fp8-product {kind} N={rows} K={cols} BN={bn}"),
            &weight,
            rows,
            cols,
            GemvTile {
                rows: bn,
                blocks: KT / 32,
            },
        )?;
    }
    Ok(())
}

fn bf16_case(
    probe: &mut Probe,
    rows: usize,
    cols: usize,
    real: Vec<bf16>,
    kinds: &[&str],
    label: &str,
) -> Result<()> {
    let stream = probe.stream.clone();
    for &kind in kinds {
        let data: Vec<bf16> = match kind {
            "real" => real.clone(),
            "zero" => vec![bf16::from_f32(0.0); rows * cols],
            "const" => vec![bf16::from_f32(1.0); rows * cols],
            "pattern" => (0..rows * cols)
                .map(|i| bf16::from_f32(((i * 37 % 251) as f32 - 125.0) / 64.0))
                .collect(),
            "random" => {
                let mut rng = Lcg(7);
                (0..rows * cols)
                    .map(|_| {
                        bf16::from_bits(u16::from_le_bytes([rng.next(), rng.next() & 0x3f | 0x3c]))
                    })
                    .collect()
            }
            _ => unreachable!(),
        };
        let weight = Weight::Dense(DenseMatrix::upload(data, rows, cols, &stream)?);
        product(
            probe,
            &format!("{label} {kind} N={rows} K={cols} BN=16"),
            &weight,
            rows,
            cols,
            GemvTile {
                rows: 16,
                blocks: KT / 32,
            },
        )?;
    }
    Ok(())
}

fn nvfp4(probe: &mut Probe) -> Result<()> {
    let (experts, rows_per, cols, bn) = (9, 1024, 2048, 128);
    let rows = experts * rows_per;
    let stream = probe.stream.clone();
    let checkpoint = Checkpoint::open(NVFP4_DIR)?;
    let mut real_packed = Vec::new();
    let mut real_scales = Vec::new();
    for expert in 0..experts {
        for part in ["gate_proj", "up_proj"] {
            let suffix = format!(".experts.{expert}.{part}.weight");
            let name = checkpoint
                .names()
                .filter(|n| n.contains("layers.0.") && n.ends_with(&suffix))
                .min()
                .ok_or(format!("no {suffix}"))?
                .to_string();
            if expert == 0 && part == "gate_proj" {
                println!("REAL nvfp4 {name}");
            }
            real_packed.extend(
                checkpoint
                    .tensor_nvfp4(&name)?
                    .data
                    .into_iter()
                    .map(|b| b.to_bits()),
            );
            real_scales.extend(
                checkpoint
                    .tensor_fp8(&name.replace(".weight", ".weight_scale"))?
                    .data
                    .into_iter()
                    .map(|b| b.0),
            );
        }
    }
    let x_values: Vec<f32> = (0..cols)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0)
        .collect();
    let x = upload(&stream, x_values.clone(), &[1, cols / 32, 32])?;
    let x_pairs = upload(&stream, x_values, &[1, cols / 2, 2])?;
    let row_scales = upload(&stream, vec![1.0f32; rows], &[1, rows])?;
    let chunks = (cols / KT) as i32;
    for kind in ["zero", "const", "pattern", "random", "real"] {
        let packed = if kind == "real" {
            real_packed.clone()
        } else {
            bytes(kind, rows * cols / 2, 0x22)
        };
        let scales = if kind == "real" {
            real_scales.clone()
        } else {
            bytes(kind, rows * cols / NVFP4_GROUP, 0x38)
                .into_iter()
                .map(|b| b & 0x7f)
                .collect()
        };
        let w_u8 = upload(&stream, packed.clone(), &[rows, cols / 2])?;
        let w = upload(
            &stream,
            packed.into_iter().map(f4e2m1fnx2::from_bits).collect(),
            &[rows, cols / 2],
        )?;
        let s = upload(
            &stream,
            scales.into_iter().map(f8e4m3fn).collect(),
            &[rows, cols / NVFP4_GROUP],
        )?;
        let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
        let shape = format!("N={rows} K={cols} BN={bn}");
        probe.repeat(&format!("nvfp4-load {kind} {shape}"), |st| {
            probes::load_only(
                (&mut y).partition([1, bn]),
                w_u8.clone(),
                (cols / 2 / (KT / 2)) as i32,
            )
            .generics(vec![bn.to_string(), (KT / 2).to_string()])
            .sync_on(st)?;
            Ok(())
        })?;
        probe.repeat(&format!("nvfp4-product {kind} {shape}"), |st| {
            probes::nvfp4_product(
                (&mut y).partition([1, bn]),
                w.clone(),
                s.clone(),
                row_scales.clone(),
                x.clone(),
                (cols / 32) as i32,
            )
            .generics(vec![
                bn.to_string(),
                (KT / 32).to_string(),
                KT.to_string(),
                (KT / 2).to_string(),
                (KT / NVFP4_GROUP).to_string(),
            ])
            .sync_on(st)?;
            Ok(())
        })?;
        let reference = y.dup().to_host_vec().sync_on(&stream)?;
        probe.repeat(&format!("nvfp4-bitdecode {kind} {shape}"), |st| {
            probes::nvfp4_bitdecode(
                (&mut y).partition([1, bn]),
                w_u8.clone(),
                s.clone(),
                row_scales.clone(),
                x_pairs.clone(),
                chunks,
            )
            .generics(vec![
                bn.to_string(),
                (KT / 2).to_string(),
                (KT / NVFP4_GROUP).to_string(),
                (NVFP4_GROUP / 2).to_string(),
            ])
            .sync_on(st)?;
            Ok(())
        })?;
        let decoded = y.dup().to_host_vec().sync_on(&stream)?;
        let scale = reference
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(f32::MIN_POSITIVE);
        let diff = decoded
            .iter()
            .zip(&reference)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        println!(
            "CHECK bitdecode-product {kind} max_abs {diff:.3e} max_rel {:.3e}",
            diff / scale
        );
    }
    Ok(())
}

// lm_head launched ten times in a row without the L2 read in between, next to the usual
// read-then-launch case, to separate the measurement setup from the kernel.
fn lm_head_back_to_back(probe: &mut Probe) -> Result<()> {
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
    product(
        probe,
        &format!("lmhead-scrubbed const N={rows} K={cols} BN=16"),
        &weight,
        rows,
        cols,
        tile,
    )?;
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
        .reshape(&[1, cols / 32, 32])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    println!(
        "CFG lmhead-back-to-back const N={rows} K={cols} BN=16 launches={}",
        WARMUP + REPS
    );
    for _ in 0..WARMUP + REPS {
        gemv(
            &weight,
            &x.view(&[1, cols / 32, 32])?,
            (&mut y).partition([1, tile.rows]),
            tile,
        )
        .submit(&mut Eager(&stream))?;
    }
    Ok(())
}

fn fp8_tiles(probe: &mut Probe) -> Result<()> {
    let (rows, cols) = (8192, 2048);
    let stream = probe.stream.clone();
    let weight = Weight::Fp8(
        HostFp8Matrix::from_parts(
            bytes("random", rows * cols, 0)
                .into_iter()
                .map(f8e4m3fn)
                .collect(),
            vec![1.0; rows / SCALE_BLOCK * (cols / SCALE_BLOCK)],
            rows,
            cols,
            SCALE_BLOCK,
            SCALE_BLOCK,
        )?
        .upload(&stream)?,
    );
    for bn in [128, 64, 32, 16, 8] {
        product(
            probe,
            &format!("fp8-tile random N={rows} K={cols} BN={bn}"),
            &weight,
            rows,
            cols,
            GemvTile {
                rows: bn,
                blocks: KT / 32,
            },
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let group = std::env::args()
        .nth(1)
        .ok_or("usage: gemv_data_probe {fp8|bf16|lmhead|nvfp4|fp8tiles|lmburst}")?;
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
        "bf16" => {
            let checkpoint = Checkpoint::open(BF16_DIR)?;
            let name = find(&checkpoint, Dtype::BF16, &[8192, 2048], "layers.3.")?;
            println!("REAL bf16 {name}");
            let real = checkpoint.tensor_bf16(&name)?.data;
            bf16_case(
                &mut probe,
                8192,
                2048,
                real,
                &["zero", "const", "pattern", "random", "real"],
                "bf16-product",
            )
        }
        "lmhead" => {
            let checkpoint = Checkpoint::open(BF16_DIR)?;
            let real = checkpoint.tensor_bf16("lm_head.weight")?.data;
            bf16_case(
                &mut probe,
                248320,
                2048,
                real,
                &["zero", "const", "random", "real"],
                "lmhead",
            )
        }
        "nvfp4" => nvfp4(&mut probe),
        "fp8tiles" => fp8_tiles(&mut probe),
        "lmburst" => lm_head_back_to_back(&mut probe),
        _ => Err(format!("unknown group {group}").into()),
    }
}
