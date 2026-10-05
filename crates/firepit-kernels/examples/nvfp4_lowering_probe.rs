use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::gemv::{GemvTile, gemv};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{HostNvfp4Matrix, Weight};
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// Each kernel adds one step of the NVFP4 GEMV body on top of the previous one, so the first
// variant whose SASS goes through shared memory names the step that triggers it.
#[cutile::module]
mod ladder {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn unpack_sum<const BN: i32, const KT: i32, const KT_PACKED: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let part: Tile<f32, { [BN] }> = reduce_sum(weights, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn unpack_groups<
        const BN: i32,
        const KT: i32,
        const KT_PACKED: i32,
        const KT_GROUPS: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let groups: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                weights.reshape(shape![BN, KT_GROUPS, 16]);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = reduce_sum(groups, 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let part: Tile<f32, { [BN] }> = reduce_sum(groups, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn unpack_dot<const BN: i32, const NB: i32, const KT: i32, const KT_PACKED: i32>(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        x: &Tensor<f32, { [1, -1, 32] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let xp = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = xp.load([0, kc, 0]);
            let xb: Tile<f32, { [BN, KT] }> = xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
            let part: Tile<f32, { [BN] }> = reduce_sum(weights * xb, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn unpack_dot_groups<
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const KT_PACKED: i32,
        const KT_GROUPS: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        x: &Tensor<f32, { [1, -1, 32] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let xp = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = xp.load([0, kc, 0]);
            let xb: Tile<f32, { [BN, KT] }> = xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
            let products: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                (weights * xb).reshape(shape![BN, KT_GROUPS, 16]);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = reduce_sum(products, 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let part: Tile<f32, { [BN] }> = reduce_sum(groups, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
    }

    #[cutile::entry()]
    pub fn unpack_dot_scaled<
        const BN: i32,
        const NB: i32,
        const KT: i32,
        const KT_PACKED: i32,
        const KT_GROUPS: i32,
    >(
        y: &mut Tensor<f32, { [1, BN] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        x: &Tensor<f32, { [1, -1, 32] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let wp = w.partition(shape![BN, KT_PACKED]);
        let sp = scales.partition(shape![BN, KT_GROUPS]);
        let xp = x.partition(shape![1, NB, 32]);
        let mut acc: Tile<f32, { [BN] }> = constant(0.0f32, shape![BN]);
        for kc in 0i32..chunks {
            let packed: Tile<f4e2m1fnx2, { [BN, KT_PACKED] }> = wp.load([pid.1, kc]);
            let weights: Tile<f4e2m1fn, { [BN, KT] }> = packed.unpack(shape![BN, KT]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let xb: Tile<f32, { [1, NB, 32] }> = xp.load([0, kc, 0]);
            let xb: Tile<f32, { [BN, KT] }> = xb.reshape(shape![1, KT]).broadcast(shape![BN, KT]);
            let products: Tile<f32, { [BN, KT_GROUPS, 16] }> =
                (weights * xb).reshape(shape![BN, KT_GROUPS, 16]);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = reduce_sum(products, 2i32);
            let groups: Tile<f32, { [BN, KT_GROUPS] }> = groups.reshape(shape![BN, KT_GROUPS]);
            let scale: Tile<f8e4m3fn, { [BN, KT_GROUPS] }> = sp.load([pid.1, kc]);
            let scale: Tile<f32, { [BN, KT_GROUPS] }> = convert_tile(scale);
            let part: Tile<f32, { [BN] }> = reduce_sum(groups * scale, 1i32);
            acc = acc + part.reshape(shape![BN]);
        }
        y.store(acc.reshape(shape![1, BN]));
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
    cutile::jit_cache::enable_default()?;
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse())
        .collect::<std::result::Result<_, _>>()?;
    let [rows, cols, tile_rows, blocks_per_tile] = args[..] else {
        return Err("usage: nvfp4_lowering_probe N K BN NB".into());
    };
    let tile = GemvTile {
        rows: tile_rows,
        blocks: blocks_per_tile,
    };
    if !tile.fits(rows, cols) {
        return Err("BN must divide N and NB*32 must divide K".into());
    }
    let (kt, blocks) = (tile.blocks * 32, cols / 32);
    let chunks = (cols / kt) as i32;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;

    let packed: Vec<f4e2m1fnx2> = vec![f4e2m1fnx2::from_bits(0x22); rows * cols / 2];
    let scales = vec![f8e4m3fn(0x38); rows * cols / 16];
    let product = Weight::Nvfp4(
        HostNvfp4Matrix::from_parts(packed.clone(), scales.clone(), 1.0, rows, cols)?
            .upload(&stream)?,
    );
    let w = upload(&stream, packed, &[rows, cols / 2])?;
    let s = upload(&stream, scales, &[rows, cols / 16])?;
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
        .reshape(&[1, blocks, 32])
        .sync_on(&stream)?;
    let x = Arc::new(x);
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;

    let options = BenchOptions::default();
    let bytes = rows * cols / 2;
    let print = |variant: &str, ms: f32| {
        let gbps = bytes as f64 / (ms as f64 * 1e6);
        println!("{variant},{rows},{cols},{tile_rows},{blocks_per_tile},{ms:.6},{gbps:.1}");
    };
    let (kt_s, packed_s, groups_s) = (kt.to_string(), (kt / 2).to_string(), (kt / 16).to_string());
    let (bn_s, nb_s) = (tile.rows.to_string(), tile.blocks.to_string());
    println!("variant,N,K,BN,NB,ms,GBps");

    let ms = do_bench(&stream, &options, |stream| {
        ladder::unpack_sum((&mut y).partition([1, tile.rows]), w.clone(), chunks)
            .generics(vec![bn_s.clone(), kt_s.clone(), packed_s.clone()])
            .sync_on(stream)
            .map(|_| ())
            .map_err(Into::into)
    })?
    .median_ms();
    print("unpack_sum", ms);

    let ms = do_bench(&stream, &options, |stream| {
        ladder::unpack_groups((&mut y).partition([1, tile.rows]), w.clone(), chunks)
            .generics(vec![
                bn_s.clone(),
                kt_s.clone(),
                packed_s.clone(),
                groups_s.clone(),
            ])
            .sync_on(stream)
            .map(|_| ())
            .map_err(Into::into)
    })?
    .median_ms();
    print("unpack_groups", ms);

    let ms = do_bench(&stream, &options, |stream| {
        ladder::unpack_dot(
            (&mut y).partition([1, tile.rows]),
            w.clone(),
            x.clone(),
            chunks,
        )
        .generics(vec![
            bn_s.clone(),
            nb_s.clone(),
            kt_s.clone(),
            packed_s.clone(),
        ])
        .sync_on(stream)
        .map(|_| ())
        .map_err(Into::into)
    })?
    .median_ms();
    print("unpack_dot", ms);

    let ms = do_bench(&stream, &options, |stream| {
        ladder::unpack_dot_groups(
            (&mut y).partition([1, tile.rows]),
            w.clone(),
            x.clone(),
            chunks,
        )
        .generics(vec![
            bn_s.clone(),
            nb_s.clone(),
            kt_s.clone(),
            packed_s.clone(),
            groups_s.clone(),
        ])
        .sync_on(stream)
        .map(|_| ())
        .map_err(Into::into)
    })?
    .median_ms();
    print("unpack_dot_groups", ms);

    let ms = do_bench(&stream, &options, |stream| {
        ladder::unpack_dot_scaled(
            (&mut y).partition([1, tile.rows]),
            w.clone(),
            s.clone(),
            x.clone(),
            chunks,
        )
        .generics(vec![
            bn_s.clone(),
            nb_s.clone(),
            kt_s.clone(),
            packed_s.clone(),
            groups_s.clone(),
        ])
        .sync_on(stream)
        .map(|_| ())
        .map_err(Into::into)
    })?
    .median_ms();
    print("unpack_dot_scaled", ms);

    let ms = do_bench(&stream, &options, |stream| {
        gemv(
            &product,
            &x.view(&[1, blocks, 32])?,
            (&mut y).partition([1, tile.rows]),
            tile,
        )
        .submit(&mut Eager(stream))
        .map_err(Into::into)
    })?
    .median_ms();
    print("product", ms);
    Ok(())
}
