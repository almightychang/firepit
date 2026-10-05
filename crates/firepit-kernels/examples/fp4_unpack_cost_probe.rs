use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// FP4 bodies at one shape per run, from a packed-byte sum up to the NVFP4 group-scale sum
// without the input vector. Each launch follows a read of a 64 MiB buffer to evict L2 with
// clean lines; time the kernels with nsys.
const LAUNCHES: usize = 10;
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

    // Sums the packed bytes: the memory floor.
    #[cutile::entry()]
    pub fn fp4_load<const BN: i32, const KT_PACKED: i32>(
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
        let out: Tile<f32, { [BN] }> = convert_tile(sum);
        y.store(out.reshape(shape![1, BN]));
    }

    // Unpack and convert, accumulate element-wise, reduce once after the loop.
    #[cutile::entry()]
    pub fn fp4_unpack<const BN: i32, const KT: i32, const KT_PACKED: i32>(
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

    // Unpack, convert, and sum groups of 16 every tile; no scales.
    #[cutile::entry()]
    pub fn fp4_groups<const BN: i32, const KT: i32, const KT_PACKED: i32, const KT_GROUPS: i32>(
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

    // NVFP4 without the input vector: group-of-16 sums times the E4M3 group scales.
    #[cutile::entry()]
    pub fn fp4_scaled<const BN: i32, const KT: i32, const KT_PACKED: i32, const KT_GROUPS: i32>(
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

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse())
        .collect::<std::result::Result<_, _>>()?;
    let [rows, cols, bn] = args[..] else {
        return Err("usage: fp4_unpack_cost_probe N K BN".into());
    };
    let kt = 128;
    if rows % bn != 0 || cols % kt != 0 {
        return Err("BN must divide N and 128 must divide K".into());
    }
    cutile::jit_cache::enable_default()?;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    let mut probe = Probe {
        buffer: Arc::new(api::zeros::<f32>(&[SCRUB_LEN]).sync_on(&stream)?),
        sums: api::zeros::<f32>(&[SCRUB_LEN / SCRUB_TILE]).sync_on(&stream)?,
        stream: stream.clone(),
    };
    // 0x22 packs two E2M1 1.0 values and 0x38 is 1.0 in E4M3, so every output row is K.
    let bytes = vec![0x22u8; rows * cols / 2];
    let w_bytes: Arc<Tensor<u8>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(bytes.clone()))
            .reshape(&[rows, cols / 2])
            .sync_on(&stream)?,
    );
    let w: Arc<Tensor<f4e2m1fnx2>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(
            bytes
                .into_iter()
                .map(f4e2m1fnx2::from_bits)
                .collect::<Vec<_>>(),
        ))
        .reshape(&[rows, cols / 2])
        .sync_on(&stream)?,
    );
    let scales: Arc<Tensor<f8e4m3fn>> = Arc::new(
        api::copy_host_vec_to_device(&Arc::new(vec![f8e4m3fn(0x38); rows * cols / 16]))
            .reshape(&[rows, cols / 16])
            .sync_on(&stream)?,
    );
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let chunks = (cols / kt) as i32;
    let expected = vec![cols as f32; rows];
    let shape = format!("N={rows} K={cols} BN={bn}");
    let (kt_s, packed_s, groups_s) = (kt.to_string(), (kt / 2).to_string(), (kt / 16).to_string());

    probe.repeat(&format!("RUN fp4_load {shape}"), |s| {
        kernels::fp4_load((&mut y).partition([1, bn]), w_bytes.clone(), chunks)
            .generics(vec![bn.to_string(), packed_s.clone()])
            .sync_on(s)?;
        Ok(())
    })?;
    probe.repeat(&format!("RUN fp4_unpack {shape}"), |s| {
        kernels::fp4_unpack((&mut y).partition([1, bn]), w.clone(), chunks)
            .generics(vec![bn.to_string(), kt_s.clone(), packed_s.clone()])
            .sync_on(s)?;
        Ok(())
    })?;
    if y.dup().to_host_vec().sync_on(&stream)? != expected {
        return Err("fp4_unpack: every row should be K".into());
    }
    probe.repeat(&format!("RUN fp4_groups {shape}"), |s| {
        kernels::fp4_groups((&mut y).partition([1, bn]), w.clone(), chunks)
            .generics(vec![
                bn.to_string(),
                kt_s.clone(),
                packed_s.clone(),
                groups_s.clone(),
            ])
            .sync_on(s)?;
        Ok(())
    })?;
    if y.dup().to_host_vec().sync_on(&stream)? != expected {
        return Err("fp4_groups: every row should be K".into());
    }
    probe.repeat(&format!("RUN fp4_scaled {shape}"), |s| {
        kernels::fp4_scaled(
            (&mut y).partition([1, bn]),
            w.clone(),
            scales.clone(),
            chunks,
        )
        .generics(vec![
            bn.to_string(),
            kt_s.clone(),
            packed_s.clone(),
            groups_s.clone(),
        ])
        .sync_on(s)?;
        Ok(())
    })?;
    if y.dup().to_host_vec().sync_on(&stream)? != expected {
        return Err("fp4_scaled: every row should be K".into());
    }
    Ok(())
}
