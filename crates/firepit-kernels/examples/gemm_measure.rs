use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor, ToHostVec};
use cutile::tile_kernel::{PartitionOp, TileKernel};
use firepit_kernels::gemm::{DEFAULT_GEMM_TILE, GemmTile, gemm};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::DenseMatrix;
use half::bf16;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

#[cutile::module]
mod kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<bf16, { [-1, K] }>,
        y: &Tensor<bf16, { [K, -1] }>,
    ) {
        let x_tiles = x.partition(shape![BM, BK]);
        let y_tiles = y.partition(shape![BK, BN]);
        let block: (i32, i32, i32) = get_tile_block_id();
        let mut acc = z.load();
        for part in 0i32..(K / BK) {
            acc = mma(
                x_tiles.load([block.0, part]),
                y_tiles.load([part, block.1]),
                acc,
            );
        }
        z.store(acc);
    }
}

fn run(
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    bk: usize,
    repeats: usize,
    check: bool,
) -> Result<(), Box<dyn Error>> {
    if m % bm != 0 || n % bn != 0 || k % bk != 0 || repeats == 0 {
        return Err("shape must be divisible by tile and repeats must be positive".into());
    }
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let a: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i * 13 % 31) as f32 - 15.0) / 16.0))
        .collect();
    let b: Vec<bf16> = (0..k * n)
        .map(|i| bf16::from_f32(((i * 7 % 29) as f32 - 14.0) / 16.0))
        .collect();
    let x: Arc<Tensor<bf16>> = api::copy_host_vec_to_device(&Arc::new(a.clone()))
        .reshape(&[m, k])
        .sync_on(&stream)?
        .into();
    let y: Arc<Tensor<bf16>> = api::copy_host_vec_to_device(&Arc::new(b.clone()))
        .reshape(&[k, n])
        .sync_on(&stream)?
        .into();
    let mut z = api::zeros::<f32>(&[m, n])
        .partition([bm, bn])
        .sync_on(&stream)?;
    let generics = vec![
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
        k.to_string(),
    ];
    let (result, _, _) = kernel::gemm(z, x.clone(), y.clone())
        .generics(generics.clone())
        .sync_on(&stream)?;
    z = result;
    if check {
        let actual: Vec<f32> = z.unpartition().to_host_vec().sync_on(&stream)?;
        let u = f32::EPSILON as f64 / 2.0;
        let gamma = (8.0 * (k as f64).sqrt() * u + k as f64 * u * u / (1.0 - u)).exp_m1();
        let mut excess = 0.0f64;
        for row in 0..m {
            for col in 0..n {
                let (mut expected, mut magnitude) = (0.0f64, 0.0f64);
                for p in 0..k {
                    let product = a[row * k + p].to_f32() as f64 * b[p * n + col].to_f32() as f64;
                    expected += product;
                    magnitude += product.abs();
                }
                excess = excess.max(
                    ((actual[row * n + col] as f64 - expected).abs())
                        / (gamma * magnitude).max(f64::MIN_POSITIVE),
                );
            }
        }
        println!("oracle excess={excess:.6}");
        if excess > 1.0 {
            return Err("Higham–Mary bound exceeded".into());
        }
    }
    let mut z = api::zeros::<f32>(&[m, n])
        .partition([bm, bn])
        .sync_on(&stream)?;
    let start = Instant::now();
    for _ in 0..repeats {
        let (result, _, _) = kernel::gemm(z, x.clone(), y.clone())
            .generics(generics.clone())
            .sync_on(&stream)?;
        z = result;
    }
    let ms = start.elapsed().as_secs_f64() * 1e3 / repeats as f64;
    let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e9);
    println!("M={m} N={n} K={k} BM={bm} BN={bn} BK={bk} ms={ms:.4} TFLOPS={tflops:.3}");
    Ok(())
}

fn run_split(
    m: usize,
    n: usize,
    k: usize,
    repeats: usize,
    tile: GemmTile,
) -> Result<(), Box<dyn Error>> {
    if repeats == 0 || m % tile.m != 0 || n % tile.n != 0 || k % tile.k != 0 {
        return Err("shape must be divisible by tile and repeats must be positive".into());
    }
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) / 16.0)
        .collect();
    let b: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(((i * 7 % 29) as f32 - 14.0) / 16.0))
        .collect();
    let x = api::copy_host_vec_to_device(&Arc::new(a))
        .reshape(&[m, k])
        .sync_on(&stream)?;
    let w = DenseMatrix::upload(b, n, k, &stream)?;
    let mut y = api::zeros::<f32>(&[m, n]).sync_on(&stream)?;
    let view = x.view(&[m, k])?;
    gemm(&w, &view, (&mut y).partition([tile.m, tile.n]), tile).submit(&mut Eager(&stream))?;
    let start = Instant::now();
    for _ in 0..repeats {
        gemm(&w, &view, (&mut y).partition([tile.m, tile.n]), tile).submit(&mut Eager(&stream))?;
    }
    let ms = start.elapsed().as_secs_f64() * 1e3 / repeats as f64;
    let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e9);
    println!(
        "split M={m} N={n} K={k} BM={} BN={} BK={} ms={ms:.4} TFLOPS={tflops:.3}",
        tile.m, tile.n, tile.k
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|s| s == "--split") {
        if args.len() != 5 {
            return Err("usage: gemm_measure --split M N K repeats".into());
        }
        let values: Vec<usize> = args[1..]
            .iter()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?;
        return run_split(
            values[0],
            values[1],
            values[2],
            values[3],
            DEFAULT_GEMM_TILE,
        );
    }
    if args.first().is_some_and(|s| s == "--check") {
        let bm = args.get(1).map_or(Ok(32), |s| s.parse())?;
        let bn = args.get(2).map_or(Ok(64), |s| s.parse())?;
        let bk = args.get(3).map_or(Ok(32), |s| s.parse())?;
        return run(128, 128, 128, bm, bn, bk, 10, true);
    }
    if args.len() != 7 {
        return Err("usage: gemm_measure M N K BM BN BK repeats | --check [BM BN BK]".into());
    }
    let values: Vec<usize> = args.iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    run(
        values[0], values[1], values[2], values[3], values[4], values[5], values[6], false,
    )
}
