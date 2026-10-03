use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::TileKernel;
use firepit_kernels::gdn::conv1d_silu_prefill;
use firepit_kernels::tile::channel_tile;
use std::error::Error;
use std::sync::Arc;

fn measure(
    stream: &Arc<cuda_core::Stream>,
    conv_dim: usize,
    kernel: usize,
    time: usize,
    tile: usize,
) -> Result<(), Box<dyn Error>> {
    let sequence_tile = time.next_power_of_two();
    let kernel_tile = kernel.next_power_of_two();
    let input_len = time.checked_mul(conv_dim).ok_or("input too large")?;
    let state_len = conv_dim.checked_mul(kernel - 1).ok_or("state too large")?;
    let weight_len = conv_dim.checked_mul(kernel).ok_or("weight too large")?;
    let input = api::copy_host_vec_to_device(&Arc::new(vec![0.25f32; input_len]))
        .reshape(&[time, conv_dim])
        .sync_on(stream)?;
    let mut state = api::copy_host_vec_to_device(&Arc::new(vec![0.5f32; state_len]))
        .reshape(&[1, conv_dim, kernel - 1])
        .sync_on(stream)?;
    let weight = api::copy_host_vec_to_device(&Arc::new(vec![0.125f32; weight_len]))
        .reshape(&[conv_dim, kernel])
        .sync_on(stream)?;
    let mut output = api::zeros::<f32>(&[time, conv_dim]).sync_on(stream)?;
    let generics = vec![
        sequence_tile.to_string(),
        tile.to_string(),
        kernel_tile.to_string(),
    ];
    let launch = |state: &mut cutile::tensor::Tensor<f32>,
                  output: &mut cutile::tensor::Tensor<f32>,
                  stream: &Arc<cuda_core::Stream>| {
        conv1d_silu_prefill(
            output.partition([sequence_tile, tile]),
            &input,
            state.partition([1, tile, kernel_tile]),
            &weight,
            kernel as i32,
        )
        .generics(generics.clone())
        .sync_on(stream)
        .map(|_| ())
        .map_err(Into::into)
    };
    launch(&mut state, &mut output, stream)?;
    let samples = do_bench(stream, &BenchOptions::default(), |stream| {
        launch(&mut state, &mut output, stream)
    })?;
    let ms = samples.median_ms() as f64;
    let bytes = (2 * input_len + 2 * state_len + weight_len) as f64 * 4.0;
    let gbps = bytes / (ms * 1e6);
    println!(
        "T={time} C={conv_dim} K={kernel} time_tile={sequence_tile} channel_tile={tile} samples={} median_ms={ms:.6} effective_GBps={gbps:.3}",
        samples.reps(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    assert!(!cfg!(debug_assertions), "measurement requires --release");
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse())
        .collect::<Result<_, _>>()?;
    if args.len() < 3 {
        return Err("usage: conv_prefill_measure CONV_DIM K T [T ...]".into());
    }
    let (conv_dim, kernel) = (args[0], args[1]);
    if conv_dim == 0
        || kernel < 2
        || conv_dim
            .checked_mul(kernel)
            .is_none_or(|len| len > i32::MAX as usize)
        || kernel
            .checked_next_power_of_two()
            .is_none_or(|tile| tile > i32::MAX as usize)
    {
        return Err("invalid CONV_DIM or K".into());
    }
    cutile::jit_cache::enable_default()?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let sm_count = cuda_core::simt::CudaContext::new(device.ordinal())?.multiprocessor_count()?;
    let tile = channel_tile(conv_dim, sm_count as usize, 1);
    println!("device={} sm_count={sm_count}", device.name()?);
    for &time in &args[2..] {
        if time == 0
            || time > i32::MAX as usize / conv_dim
            || time
                .checked_next_power_of_two()
                .is_none_or(|tile| tile > i32::MAX as usize)
        {
            return Err("invalid T".into());
        }
        measure(&stream, conv_dim, kernel, time, tile)?;
    }
    Ok(())
}
