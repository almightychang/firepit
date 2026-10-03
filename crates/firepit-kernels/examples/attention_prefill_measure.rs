use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::TileKernel;
use firepit_kernels::attention::{AttentionDims, DEFAULT_PREFILL_TILE, causal_attention_prefill};
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

fn measure(t: usize, q_heads: usize, kv_heads: usize, d: usize) -> Result<(), Box<dyn Error>> {
    let dims = AttentionDims::new(t, q_heads, kv_heads, d)?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let q: Vec<f32> = (0..t * q_heads * d)
        .map(|i| ((i * 13 % 127) as f32 - 63.0) / 64.0)
        .collect();
    let k: Vec<f32> = (0..t * kv_heads * d)
        .map(|i| ((i * 17 % 131) as f32 - 65.0) / 64.0)
        .collect();
    let v: Vec<f32> = (0..t * kv_heads * d)
        .map(|i| ((i * 19 % 137) as f32 - 68.0) / 64.0)
        .collect();
    let q = api::copy_host_vec_to_device(&Arc::new(q))
        .reshape(&[t, q_heads, d])
        .sync_on(&stream)?;
    let k = api::copy_host_vec_to_device(&Arc::new(k))
        .reshape(&[t, kv_heads, d])
        .sync_on(&stream)?;
    let v = api::copy_host_vec_to_device(&Arc::new(v))
        .reshape(&[t, kv_heads, d])
        .sync_on(&stream)?;
    let mut out = api::zeros::<f32>(&[t, q_heads, d]).sync_on(&stream)?;
    let tile = DEFAULT_PREFILL_TILE;
    causal_attention_prefill(
        (&mut out).partition([tile.m, 1, d]),
        &q,
        &k,
        &v,
        0,
        dims.scale(),
    )
    .generics(dims.prefill_generics(tile))
    .sync_on(&stream)?;
    let mut times = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        causal_attention_prefill(
            (&mut out).partition([tile.m, 1, d]),
            &q,
            &k,
            &v,
            0,
            dims.scale(),
        )
        .generics(dims.prefill_generics(tile))
        .sync_on(&stream)?;
        times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    let ms = times[times.len() / 2];
    let tflops = 2.0 * t as f64 * t as f64 * q_heads as f64 * d as f64 / (ms * 1e9);
    println!(
        "T={t} H={q_heads} Hkv={kv_heads} D={d} BM={} BN={} ms={ms:.4} TFLOPS={tflops:.3}",
        tile.m, tile.n
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    cutile::jit_cache::enable_default()?;
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    if args.len() != 3 {
        return Err("usage: attention_prefill_measure Q_HEADS KV_HEADS HEAD_DIM".into());
    }
    for t in [512, 1024, 2048, 4096] {
        measure(t, args[0], args[1], args[2])?;
    }
    Ok(())
}
