use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::tensor::{OWNED, PartitionMut, ToHostVec};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::gdn::conv1d_silu_prefill;
use std::error::Error;
use std::sync::Arc;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry]
    pub fn owned_prefix_probe<const CT: i32, const MAP: [i32; 2]>(
        mut out: MappedPartitionMut<f32, { [1, CT] }, MAP>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let times = num_tiles(&out, 0).into_dim();
        let input_tiles = input.partition(shape![1, CT]);
        for index in out.iter_indices() {
            let (_, channel) = index.components();
            let mut sum: Tile<f32, { [1, CT] }> = constant(0.0f32, shape![1, CT]);
            for time in times {
                sum = sum + input_tiles.load([time, channel]);
                out.store(sum, coord((time, channel)));
            }
        }
    }

    #[cutile::entry]
    pub fn owned_prefix_probe_with_state<const CT: i32, const MAP: [i32; 2]>(
        mut out: MappedPartitionMut<f32, { [1, CT] }, MAP>,
        mut state: MappedPartitionMut<f32, { [1, CT] }, MAP>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let times = num_tiles(&out, 0).into_dim();
        let input_tiles = input.partition(shape![1, CT]);
        for index in out.iter_indices_with(&state) {
            let (_, channel) = index.components();
            let mut sum: Tile<f32, { [1, CT] }> = constant(0.0f32, shape![1, CT]);
            for time in times {
                sum = sum + input_tiles.load([time, channel]);
                out.store(sum, coord((time, channel)));
                state.store(sum, coord((time, channel)));
            }
        }
    }

    #[cutile::entry]
    pub fn owned_prefix_probe_distinct_state<
        const CT: i32,
        const KT: i32,
        const OUT_MAP: [i32; 2],
        const STATE_MAP: [i32; 2],
    >(
        mut out: MappedPartitionMut<f32, { [1, CT] }, OUT_MAP>,
        mut state: MappedPartitionMut<f32, { [CT, KT] }, STATE_MAP>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let times = num_tiles(&out, 0).into_dim();
        let input_tiles = input.partition(shape![1, CT]);
        let mut sum: Tile<f32, { [1, CT] }> = constant(0.0f32, shape![1, CT]);
        for index in out.iter_indices() {
            let (_, channel) = index.components();
            for time in times {
                sum = sum + input_tiles.load([time, channel]);
                out.store(sum, coord((time, channel)));
            }
        }
        for index in state.iter_indices() {
            state.store(sum.reshape(shape![CT, 1]).broadcast(shape![CT, KT]), index);
        }
    }

    #[cutile::entry]
    pub fn owned_conv_prefill_inplace<const CT: i32, const KT: i32, const OUT_MAP: [i32; 2]>(
        mut out: MappedPartitionMut<f32, { [1, CT] }, OUT_MAP>,
        state: &mut Tensor<f32, { [CT, KT] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        weight: &Tensor<f32, { [-1, -1] }>,
        kernel_size: i32,
    ) {
        let times = num_tiles(&out, 0).into_dim();
        let input_tiles = input.partition(shape![1, CT]);
        let weight_tiles = weight.partition(shape![CT, KT]);
        let positions: Tile<i32, { [KT] }> = iota(shape![KT]);
        let slots = positions.reshape(shape![1, KT]).broadcast(shape![CT, KT]);
        let current_column = eq_tile(slots, (kernel_size - 1i32).broadcast(shape![CT, KT]));
        let valid_columns = lt_tile(slots, kernel_size.broadcast(shape![CT, KT]));
        let zero: Tile<f32, { [CT, KT] }> = constant(0.0f32, shape![CT, KT]);
        let one: Tile<f32, { [CT] }> = constant(1.0f32, shape![CT]);
        let negative: Tile<f32, { [CT] }> = constant(-1.0f32, shape![CT]);
        let mut previous: Tile<f32, { [CT, KT] }> = state.load();
        for index in out.iter_indices() {
            let (_, channel) = index.components();
            let coefficients = weight_tiles.load([channel, 0]);
            for time in times {
                let current: Tile<f32, { [CT] }> =
                    input_tiles.load([time, channel]).reshape(shape![CT]);
                let current = current.reshape(shape![CT, 1]).broadcast(shape![CT, KT]);
                let window = select(
                    valid_columns,
                    select(current_column, current, previous),
                    zero,
                );
                let sum: Tile<f32, { [CT] }> = reduce_sum(window * coefficients, 1);
                let sum = sum.reshape(shape![CT]);
                let activated = sum / (one + exp(sum * negative));
                out.store(activated.reshape(shape![1, CT]), coord((time, channel)));
                let mut shifted = current;
                for slot in 0i32..(kernel_size - 2i32) {
                    let next: Tile<f32, { [CT, 1] }> = extract(
                        previous,
                        [scalar_to_tile(0i32), scalar_to_tile(slot + 1i32)],
                    );
                    let at_slot = eq_tile(slots, slot.broadcast(shape![CT, KT]));
                    shifted = select(at_slot, next.broadcast(shape![CT, KT]), shifted);
                }
                previous = shifted;
            }
        }
        state.store(previous);
    }

    #[cutile::entry]
    pub fn owned_chunk_shape<const DK: i32, const VT: i32, const MAP: [i32; 3]>(
        mut out: MappedPartitionMut<f32, { [1, 1, VT] }, MAP>,
        mut state_out: MappedPartitionMut<f32, { [1, DK, VT] }, MAP>,
        input: &Tensor<f32, { [-1, -1, -1] }>,
        state_in: &Tensor<f32, { [-1, -1, -1] }>,
    ) {
        let chunks = num_tiles(&out, 0).into_dim();
        let input_tiles = input.partition(shape![1, 1, VT]);
        let initial_tiles = state_in.partition(shape![1, DK, VT]);
        let mut acc: Tile<f32, { [1, DK, VT] }> = constant(0.0f32, shape![1, DK, VT]);
        for index in out.iter_indices() {
            let (_, head, column) = index.components();
            acc = initial_tiles.load([0, head, column]);
            for chunk in chunks {
                let value: Tile<f32, { [1, 1, VT] }> = input_tiles.load([chunk, head, column]);
                out.store(value, coord((chunk, head, column)));
                acc = acc + value.broadcast(shape![1, DK, VT]);
            }
        }
        for index in state_out.iter_indices() {
            state_out.store(acc, index);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    cutile::jit_cache::enable_default()?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    println!("device={}", device.name()?);
    for time in [3usize, 5, 9] {
        let channels = 64usize;
        let tile = 16usize;
        let input_host: Vec<f32> = (0..time * channels).map(|i| (i % 17) as f32).collect();
        let input = api::copy_host_vec_to_device(&Arc::new(input_host.clone()))
            .reshape(&[time, channels])
            .sync_on(&stream)?;
        let mut output = api::zeros::<f32>(&[time, channels]).sync_on(&stream)?;
        let mut state = api::zeros::<f32>(&[time, channels]).sync_on(&stream)?;
        let mut distinct_output = api::zeros::<f32>(&[time, channels]).sync_on(&stream)?;
        let mut distinct_state = api::zeros::<f32>(&[channels, 4]).sync_on(&stream)?;
        let map = [OWNED, 1];
        let generics = vec![tile.to_string(), "0".into(), "1".into()];
        let blocks = (channels / tile) as u32;
        let before = cutile::jit_cache::jit_backend_compile_count();
        kernels::owned_prefix_probe((&mut output).partition([1, tile]).map(map, blocks), &input)
            .generics(generics.clone())
            .sync_on(&stream)?;
        let scan_compiles = cutile::jit_cache::jit_backend_compile_count() - before;
        kernels::owned_prefix_probe_with_state(
            (&mut output).partition([1, tile]).map(map, blocks),
            (&mut state).partition([1, tile]).map(map, blocks),
            &input,
        )
        .generics(generics)
        .sync_on(&stream)?;
        let both_compiles = cutile::jit_cache::jit_backend_compile_count() - before;
        let actual: Vec<f32> = output.to_host_vec().sync_on(&stream)?;
        let states: Vec<f32> = state.to_host_vec().sync_on(&stream)?;
        let mut expected = vec![0f32; time * channels];
        for t in 0..time {
            for c in 0..channels {
                expected[t * channels + c] = input_host[t * channels + c]
                    + if t > 0 {
                        expected[(t - 1) * channels + c]
                    } else {
                        0.0
                    };
            }
        }
        assert_eq!(actual, expected);
        assert_eq!(states, expected);
        println!(
            "T={time} scan_compiles={scan_compiles} cumulative_compiles={both_compiles} values=ok"
        );
        let distinct_before = cutile::jit_cache::jit_backend_compile_count();
        kernels::owned_prefix_probe_distinct_state(
            (&mut distinct_output).partition([1, tile]).map(map, blocks),
            (&mut distinct_state)
                .partition([tile, 4])
                .map([1, OWNED], blocks),
            &input,
        )
        .generics(vec![
            tile.to_string(),
            "4".into(),
            "0".into(),
            "1".into(),
            "1".into(),
            "0".into(),
        ])
        .sync_on(&stream)?;
        let distinct_compiles = cutile::jit_cache::jit_backend_compile_count() - distinct_before;
        let actual: Vec<f32> = distinct_state.to_host_vec().sync_on(&stream)?;
        for c in 0..channels {
            for k in 0..4 {
                assert_eq!(actual[c * 4 + k], expected[(time - 1) * channels + c]);
            }
        }
        println!("T={time} distinct_state_compiles={distinct_compiles} values=ok");
    }
    for time in [3usize, 5, 9, 16, 32, 64, 128, 256, 512, 1024] {
        check_conv(&stream, time)?;
    }
    check_chunk_shape(&stream)?;
    Ok(())
}

fn check_chunk_shape(stream: &Arc<cuda_core::Stream>) -> Result<(), Box<dyn Error>> {
    let chunks = 3usize;
    let heads = 2usize;
    let dk = 4usize;
    let width = 8usize;
    let vt = 4usize;
    let input_host: Vec<f32> = (0..chunks * heads * width)
        .map(|i| (i % 7) as f32)
        .collect();
    let state_host = vec![0.5f32; heads * dk * width];
    let input = api::copy_host_vec_to_device(&Arc::new(input_host.clone()))
        .reshape(&[chunks, heads, width])
        .sync_on(stream)?;
    let initial = api::copy_host_vec_to_device(&Arc::new(state_host))
        .reshape(&[1, heads * dk, width])
        .sync_on(stream)?;
    let mut output = api::zeros::<f32>(&[chunks, heads, width]).sync_on(stream)?;
    let mut state = api::zeros::<f32>(&[1, heads * dk, width]).sync_on(stream)?;
    let blocks = (heads * width / vt) as u32;
    let map = [OWNED, 1, 1];
    let before = cutile::tile_kernel::jit_compile_count();
    kernels::owned_chunk_shape(
        (&mut output).partition([1, 1, vt]).map(map, blocks),
        (&mut state).partition([1, dk, vt]).map(map, blocks),
        &input,
        &initial,
    )
    .generics(vec![
        dk.to_string(),
        vt.to_string(),
        "0".into(),
        "1".into(),
        "1".into(),
    ])
    .sync_on(stream)?;
    let jit = cutile::tile_kernel::jit_compile_count() - before;
    let output: Vec<f32> = output.to_host_vec().sync_on(stream)?;
    let state: Vec<f32> = state.to_host_vec().sync_on(stream)?;
    assert_eq!(output, input_host);
    for h in 0..heads {
        for d in 0..dk {
            for v in 0..width {
                let expected = 0.5
                    + (0..chunks)
                        .map(|c| input_host[(c * heads + h) * width + v])
                        .sum::<f32>();
                assert_eq!(state[(h * dk + d) * width + v], expected);
            }
        }
    }
    println!(
        "chunk_shape chunks={chunks} heads={heads} dk={dk} width={width} vt={vt} jit={jit} values=ok"
    );
    Ok(())
}

fn check_conv(stream: &Arc<cuda_core::Stream>, time: usize) -> Result<(), Box<dyn Error>> {
    let channels = 64usize;
    let kernel = 4usize;
    let tile = 16usize;
    let input_host: Vec<f32> = (0..time * channels)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.03125)
        .collect();
    let state_host: Vec<f32> = (0..channels * (kernel - 1))
        .map(|i| ((i % 13) as f32 - 6.0) * 0.0625)
        .collect();
    let weight_host: Vec<f32> = (0..channels * kernel)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.03125)
        .collect();
    let input = api::copy_host_vec_to_device(&Arc::new(input_host.clone()))
        .reshape(&[time, channels])
        .sync_on(stream)?;
    let weight = api::copy_host_vec_to_device(&Arc::new(weight_host.clone()))
        .reshape(&[channels, kernel])
        .sync_on(stream)?;
    let mut owned_output = api::zeros::<f32>(&[time, channels]).sync_on(stream)?;
    let mut owned_state = api::copy_host_vec_to_device(&Arc::new(state_host.clone()))
        .reshape(&[channels, kernel - 1])
        .sync_on(stream)?;
    let mut old_output = api::zeros::<f32>(&[time, channels]).sync_on(stream)?;
    let mut old_state = api::copy_host_vec_to_device(&Arc::new(state_host.clone()))
        .reshape(&[1, channels, kernel - 1])
        .sync_on(stream)?;
    let owned_generics = vec![tile.to_string(), kernel.to_string(), "0".into(), "1".into()];
    let old_generics = vec![
        time.next_power_of_two().to_string(),
        tile.to_string(),
        kernel.to_string(),
    ];
    let blocks = (channels / tile) as u32;
    let mut launch_owned = |stream: &Arc<cuda_core::Stream>| -> Result<(), cutile::error::Error> {
        kernels::owned_conv_prefill_inplace(
            (&mut owned_output)
                .partition([1, tile])
                .map([OWNED, 1], blocks),
            (&mut owned_state).partition([tile, kernel]),
            &input,
            &weight,
            kernel as i32,
        )
        .generics(owned_generics.clone())
        .sync_on(stream)?;
        Ok(())
    };
    let mut launch_old = |stream: &Arc<cuda_core::Stream>| -> Result<(), cutile::error::Error> {
        conv1d_silu_prefill(
            (&mut old_output).partition([time.next_power_of_two(), tile]),
            &input,
            (&mut old_state).partition([1, tile, kernel]),
            &weight,
            kernel as i32,
        )
        .generics(old_generics.clone())
        .sync_on(stream)?;
        Ok(())
    };
    let before = cutile::jit_cache::jit_backend_compile_count();
    let jit_before = cutile::tile_kernel::jit_compile_count();
    launch_owned(stream)?;
    let owned_compiles = cutile::jit_cache::jit_backend_compile_count() - before;
    let owned_jit = cutile::tile_kernel::jit_compile_count() - jit_before;
    launch_old(stream)?;
    let old_compiles = cutile::jit_cache::jit_backend_compile_count() - before - owned_compiles;
    let old_jit = cutile::tile_kernel::jit_compile_count() - jit_before - owned_jit;
    let owned_ms = if time >= 256 {
        Some(
            do_bench(stream, &BenchOptions::default(), |stream| {
                launch_owned(stream)
            })?
            .median_ms(),
        )
    } else {
        None
    };
    let old_ms = if time >= 256 {
        Some(
            do_bench(stream, &BenchOptions::default(), |stream| {
                launch_old(stream)
            })?
            .median_ms(),
        )
    } else {
        None
    };
    let owned_output: Vec<f32> = owned_output.to_host_vec().sync_on(stream)?;
    let old_output: Vec<f32> = old_output.to_host_vec().sync_on(stream)?;
    let owned_state: Vec<f32> = owned_state.to_host_vec().sync_on(stream)?;
    let mut expected_state = state_host.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let mut max_oracle_error = 0.0f64;
    let mut max_old_error = 0.0f32;
    for t in 0..time {
        for c in 0..channels {
            let sum: f64 = (0..kernel - 1)
                .map(|k| expected_state[c * (kernel - 1) + k] * weight_host[c * kernel + k] as f64)
                .sum::<f64>()
                + input_host[t * channels + c] as f64 * weight_host[c * kernel + kernel - 1] as f64;
            let expected = sum / (1.0 + (-sum).exp());
            if time < 256 || t >= kernel - 1 {
                max_oracle_error =
                    max_oracle_error.max((owned_output[t * channels + c] as f64 - expected).abs());
            }
            if time < 256 || t >= kernel - 1 {
                max_old_error = max_old_error
                    .max((owned_output[t * channels + c] - old_output[t * channels + c]).abs());
            }
            for k in 0..kernel - 2 {
                expected_state[c * (kernel - 1) + k] = expected_state[c * (kernel - 1) + k + 1];
            }
            expected_state[c * (kernel - 1) + kernel - 2] = input_host[t * channels + c] as f64;
        }
    }
    let max_state_error = owned_state
        .iter()
        .zip(&expected_state)
        .map(|(&a, &b)| (a as f64 - b).abs())
        .fold(0.0f64, f64::max);
    assert!(max_oracle_error < 1e-5, "oracle error {max_oracle_error}");
    assert!(max_old_error < 1e-5, "old error {max_old_error}");
    assert_eq!(max_state_error, 0.0);
    println!(
        "conv T={time} owned_jit={owned_jit} old_jit={old_jit} owned_compiles={owned_compiles} old_compiles={old_compiles} max_oracle_error={max_oracle_error:.3e} max_old_error={max_old_error:.3e} max_state_error={max_state_error:.3e} owned_ms={owned_ms:?} old_ms={old_ms:?}"
    );
    Ok(())
}
