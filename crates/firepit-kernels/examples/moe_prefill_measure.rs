use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream};
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::gemm::{DEFAULT_GEMM_TILE, DEFAULT_GROUPED_GEMM_TILE, gemm, grouped_gemm};
use firepit_kernels::moe::{
    MoeDims, PrefillLayout, PrefillPlan, combine_prefill, gather_prefill, top_k_prefill,
};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::DenseMatrix;
use half::bf16;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn dims(path: &Path) -> Result<MoeDims, Box<dyn Error>> {
    let config: serde_json::Value = serde_json::from_reader(File::open(path.join("config.json"))?)?;
    let text = &config["text_config"];
    let get = |name: &str| -> Result<usize, Box<dyn Error>> {
        Ok(text[name].as_u64().ok_or(format!("missing {name}"))? as usize)
    };
    Ok(MoeDims::new(
        get("hidden_size")?,
        get("moe_intermediate_size")?,
        get("shared_expert_intermediate_size")?,
        get("num_experts")?,
        get("num_experts_per_tok")?,
    )?)
}

fn measure(stream: &Arc<Stream>, dims: MoeDims, tokens: usize) -> Result<(), Box<dyn Error>> {
    let (h, i, e, k) = (
        dims.hidden(),
        dims.intermediate(),
        dims.experts(),
        dims.top_k(),
    );
    let selected = dims.with_shared_expert();
    let (stacked, slots) = (selected.experts(), selected.top_k());
    let router_tile = DEFAULT_GEMM_TILE;
    let tile = DEFAULT_GROUPED_GEMM_TILE;
    let padded_tokens = tokens.div_ceil(router_tile.m) * router_tile.m;
    let x_data: Vec<f32> = (0..padded_tokens * h)
        .map(|index| ((index * 17 % 59) as f32 - 29.0) / 128.0)
        .collect();
    let x = api::copy_host_vec_to_device(&Arc::new(x_data))
        .reshape(&[padded_tokens, h])
        .sync_on(stream)?;
    let router_data: Vec<bf16> = (0..e * h)
        .map(|index| bf16::from_f32(((index * 13 % 47) as f32 - 23.0) / 1024.0))
        .collect();
    let router = DenseMatrix::upload(router_data, e, h, stream)?;
    let gate_up = DenseMatrix::upload(
        vec![bf16::from_f32(0.01); stacked * 2 * i * h],
        stacked * 2 * i,
        h,
        stream,
    )?;
    let down = DenseMatrix::upload(
        vec![bf16::from_f32(0.01); stacked * h * i],
        stacked * h,
        i,
        stream,
    )?;
    let mut logits = api::zeros::<f32>(&[padded_tokens, e]).sync_on(stream)?;
    let router_ms = do_bench(stream, &BenchOptions::default(), |stream| {
        gemm(
            &router,
            &x.view(&[padded_tokens, h])?,
            logits.partition([router_tile.m, router_tile.n]),
            router_tile,
        )
        .submit(&mut Eager(stream))
        .map_err(Into::into)
    })?
    .median_ms();
    let logits_view = logits.slice(&[0..tokens, 0..e])?;
    let gate = api::zeros::<f32>(&[tokens, 1]).sync_on(stream)?;
    let top_width = selected.width();
    let mut ids = api::zeros::<i32>(&[tokens, top_width]).sync_on(stream)?;
    let mut scores = api::zeros::<f32>(&[tokens, top_width]).sync_on(stream)?;
    let top_ms = do_bench(stream, &BenchOptions::default(), |stream| {
        top_k_prefill(
            &logits_view,
            &gate,
            ids.partition([1, top_width]),
            scores.partition([1, top_width]),
            e as i32,
            k as i32,
        )
        .generics(dims.top_k_generics())
        .sync_on(stream)
        .map(|_| ())
        .map_err(Into::into)
    })?
    .median_ms();
    let host_ids: Vec<i32> = ids.dup().to_host_vec().sync_on(stream)?;
    let host_scores: Vec<f32> = scores.dup().to_host_vec().sync_on(stream)?;
    let flat_ids: Vec<i32> = host_ids
        .chunks_exact(top_width)
        .flat_map(|row| row[..slots].iter().copied())
        .collect();
    let flat_scores: Vec<f32> = host_scores
        .chunks_exact(top_width)
        .flat_map(|row| row[..slots].iter().copied())
        .collect();
    let mut sort_times = Vec::new();
    for _ in 0..9 {
        let start = Instant::now();
        let layout = PrefillLayout::parse(&flat_ids, &flat_scores, &selected, tile.m)?;
        std::hint::black_box(layout);
        sort_times.push(start.elapsed().as_secs_f64() * 1e3);
    }
    sort_times.sort_by(f64::total_cmp);
    let sort_ms = sort_times[sort_times.len() / 2];
    let plan = PrefillPlan::parse(&flat_ids, &flat_scores, &selected, tile.m, stream)?;
    let rows = plan.layout().padded_rows();
    let mut sorted = api::zeros::<f32>(&[rows, h]).sync_on(stream)?;
    let gather_ms = do_bench(stream, &BenchOptions::default(), |stream| {
        gather_prefill(
            sorted.partition([1, tile.k]),
            &x.slice(&[0..tokens, 0..h])?,
            &plan,
            tile.k,
        )
        .sync_on(stream)
        .map(|_| ())
        .map_err(Into::into)
    })?
    .median_ms();
    let mut projected = api::zeros::<f32>(&[rows, 2 * i]).sync_on(stream)?;
    let gate_up_ms = do_bench(stream, &BenchOptions::default(), |stream| {
        grouped_gemm(
            &gate_up,
            &sorted.view(&[rows, h])?,
            projected.partition([tile.m, tile.n]),
            &plan,
            2 * i,
            tile,
        )
        .submit(&mut Eager(stream))
        .map_err(Into::into)
    })?
    .median_ms();
    let mut activated = api::zeros::<f32>(&[rows, i]).sync_on(stream)?;
    let activation_ms = do_bench(stream, &BenchOptions::default(), |stream| {
        firepit_kernels::swiglu::swiglu(activated.partition([1, i]), &projected)
            .generics(vec![i.to_string()])
            .sync_on(stream)
            .map(|_| ())
            .map_err(Into::into)
    })?
    .median_ms();
    let mut routed = api::zeros::<f32>(&[rows, h]).sync_on(stream)?;
    let down_ms = do_bench(stream, &BenchOptions::default(), |stream| {
        grouped_gemm(
            &down,
            &activated.view(&[rows, i])?,
            routed.partition([tile.m, tile.n]),
            &plan,
            h,
            tile,
        )
        .submit(&mut Eager(stream))
        .map_err(Into::into)
    })?
    .median_ms();
    let mut output = api::zeros::<f32>(&[tokens, h]).sync_on(stream)?;
    let combine_ms = do_bench(stream, &BenchOptions::default(), |stream| {
        combine_prefill(output.partition([1, h]), &routed, &plan)
            .sync_on(stream)
            .map(|_| ())
            .map_err(Into::into)
    })?
    .median_ms();
    println!(
        "T={tokens} padded_rows={rows} wasted_rows={} router_ms={router_ms:.6} top_k_ms={top_ms:.6} host_sort_ms={sort_ms:.6} gather_ms={gather_ms:.6} gate_up_ms={gate_up_ms:.6} swiglu_ms={activation_ms:.6} down_ms={down_ms:.6} combine_ms={combine_ms:.6}",
        plan.layout().padding_rows()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    assert!(!cfg!(debug_assertions), "measurement requires --release");
    cutile::jit_cache::enable_default()?;
    let model = std::env::args()
        .nth(1)
        .ok_or("usage: moe_prefill_measure MODEL_DIR")?;
    let dims = dims(Path::new(&model))?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    for tokens in [512, 2048, 5000] {
        measure(&stream, dims, tokens)?;
    }
    Ok(())
}
