use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::{Device, Event, IntoResult, Stream, sys};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;
use std::time::Instant;

const H: usize = 32;
const D: usize = 128;
const L: usize = 30;
const PEAK_BYTES_PER_US: f64 = 960_000.0;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn state_step<const R: i32>(
        state: &mut Tensor<f32, { [1, R, 128] }>,
        k: &Tensor<f32, { [-1, 128] }>,
        v: &Tensor<f32, { [-1, 128] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let old: Tile<f32, { [1, R, 128] }> = state.load();
        let kh: Tile<f32, { [1, R] }> = k.partition(shape![1, R]).load([pid.0, pid.1]);
        let vh: Tile<f32, { [1, 128] }> = v.partition(shape![1, 128]).load([pid.0, 0i32]);
        let kh: Tile<f32, { [1, R, 128] }> =
            kh.reshape(shape![1, R, 1]).broadcast(shape![1, R, 128]);
        let vh: Tile<f32, { [1, R, 128] }> =
            vh.reshape(shape![1, 1, 128]).broadcast(shape![1, R, 128]);
        let a: Tile<f32, { [1, R, 128] }> = constant(0.5f32, shape![1, R, 128]);
        state.store(old * a + kh * vh);
    }
}

fn median(mut samples: Vec<f32>) -> f64 {
    samples.sort_by(f32::total_cmp);
    let mid = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[mid - 1] as f64 + samples[mid] as f64) / 2.0
    } else {
        samples[mid] as f64
    }
}

struct Inputs {
    k: Tensor<f32>,
    v: Tensor<f32>,
}

fn inputs(b: usize, stream: &Arc<Stream>) -> Result<Inputs, Box<dyn std::error::Error>> {
    let rows = b * H * D;
    let k_host: Vec<f32> = (0..rows)
        .map(|i| 1.0 + ((i / D) % 7) as f32 / 16.0 + (i % D % 11) as f32 / 128.0)
        .collect();
    let v_host: Vec<f32> = (0..rows)
        .map(|i| 1.0 - ((i / D) % 5) as f32 / 32.0 + (i % D % 13) as f32 / 256.0)
        .collect();
    let k = api::copy_host_vec_to_device(&Arc::new(k_host))
        .reshape(&[b * H, D])
        .sync_on(stream)?;
    let v = api::copy_host_vec_to_device(&Arc::new(v_host))
        .reshape(&[b * H, D])
        .sync_on(stream)?;
    Ok(Inputs { k, v })
}

fn separate_state(
    b: usize,
    stream: &Arc<Stream>,
) -> Result<Vec<Vec<Tensor<f32>>>, Box<dyn std::error::Error>> {
    let mut states = Vec::with_capacity(b);
    for _ in 0..b {
        let mut layers = Vec::with_capacity(L);
        for _ in 0..L {
            layers.push(api::zeros::<f32>(&[H, D, D]).sync_on(stream)?);
        }
        states.push(layers);
    }
    Ok(states)
}

fn stacked_state(
    b: usize,
    stream: &Arc<Stream>,
) -> Result<Vec<Tensor<f32>>, Box<dyn std::error::Error>> {
    let mut states = Vec::with_capacity(L);
    for _ in 0..L {
        states.push(api::zeros::<f32>(&[b * H, D, D]).sync_on(stream)?);
    }
    Ok(states)
}

fn capture_linear<const R: i32>(
    stream: &Arc<Stream>,
    states: &mut [Vec<Tensor<f32>>],
    inputs: &Inputs,
) -> Result<CudaGraph<()>, Box<dyn std::error::Error>> {
    Ok(CudaGraph::scope(stream, |scope| {
        for layer in 0..L {
            for (b, session) in states.iter_mut().enumerate() {
                let k = inputs.k.slice(&[b * H..(b + 1) * H, 0..D])?;
                let v = inputs.v.slice(&[b * H..(b + 1) * H, 0..D])?;
                scope.record(
                    kernels::state_step(
                        (&mut session[layer]).partition([1, R as usize, D]),
                        &k,
                        &v,
                    )
                    .generics(vec![R.to_string()]),
                )?;
            }
        }
        Ok(())
    })?)
}

fn capture_stacked<const R: i32>(
    stream: &Arc<Stream>,
    states: &mut [Tensor<f32>],
    inputs: &Inputs,
) -> Result<CudaGraph<()>, Box<dyn std::error::Error>> {
    Ok(CudaGraph::scope(stream, |scope| {
        for state in states.iter_mut() {
            scope.record(
                kernels::state_step(
                    (&mut *state).partition([1, R as usize, D]),
                    &inputs.k,
                    &inputs.v,
                )
                .generics(vec![R.to_string()]),
            )?;
        }
        Ok(())
    })?)
}

struct RawGraph {
    device: Arc<Device>,
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    _events: Vec<Event>,
    _contexts: Vec<ExecutionContext>,
}

impl RawGraph {
    fn launch(&self, stream: &Arc<Stream>) -> Result<(), Box<dyn std::error::Error>> {
        self.device.bind_to_thread()?;
        // SAFETY: exec is live until RawGraph drops, and stream belongs to this device.
        unsafe { sys::cuGraphLaunch(self.exec, stream.cu_stream()).result()? };
        Ok(())
    }
}

impl Drop for RawGraph {
    fn drop(&mut self) {
        let _ = self.device.bind_to_thread();
        // SAFETY: RawGraph uniquely owns both graph handles after all replay work is synchronized.
        unsafe {
            let _ = sys::cuGraphExecDestroy(self.exec).result();
            let _ = sys::cuGraphDestroy(self.graph).result();
        }
    }
}

fn capture_forked(
    device: &Arc<Device>,
    main: &Arc<Stream>,
    branches: &[Arc<Stream>],
    states: &mut [Vec<Tensor<f32>>],
    inputs: &Inputs,
) -> Result<RawGraph, Box<dyn std::error::Error>> {
    let start = device.new_event()?;
    let mut done = Vec::with_capacity(branches.len());
    let contexts: Vec<_> = branches
        .iter()
        .cloned()
        .map(ExecutionContext::new)
        .collect();
    device.bind_to_thread()?;
    // SAFETY: all streams share one CUDA context; start/wait/join events make a valid
    // forked capture, and every state/input buffer remains alive through graph replay.
    unsafe { main.begin_capture(sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED)? };
    start.record(main)?;
    for (b, branch) in branches.iter().enumerate() {
        branch.wait_event(&start)?;
        for layer in 0..L {
            let k = inputs.k.slice(&[b * H..(b + 1) * H, 0..D])?;
            let v = inputs.v.slice(&[b * H..(b + 1) * H, 0..D])?;
            // SAFETY: capture records work without executing it; the branch owns its
            // session state exclusively, and all graph buffers outlive the graph.
            unsafe {
                kernels::state_step((&mut states[b][layer]).partition([1, D, D]), &k, &v)
                    .generics(vec!["128".to_string()])
                    .execute(&contexts[b])?;
            }
        }
        let event = device.new_event()?;
        event.record(branch)?;
        main.wait_event(&event)?;
        done.push(event);
    }
    // SAFETY: begin_capture succeeded and every branch was joined on main.
    let graph = unsafe { main.end_capture()? };
    let mut exec = std::ptr::null_mut();
    // SAFETY: graph is valid and exec points to writable storage for the driver handle.
    unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0).result()? };
    // SAFETY: exec and stream are live on the same CUDA device.
    unsafe { sys::cuGraphUpload(exec, main.cu_stream()).result()? };
    // SAFETY: this thread is bound to device above, and main remains live.
    unsafe { main.synchronize()? };
    done.push(start);
    Ok(RawGraph {
        device: device.clone(),
        graph,
        exec,
        _events: done,
        _contexts: contexts,
    })
}

fn time_graph(
    device: &Arc<Device>,
    stream: &Arc<Stream>,
    mut launch: impl FnMut() -> Result<(), Box<dyn std::error::Error>>,
) -> Result<f64, Box<dyn std::error::Error>> {
    for _ in 0..8 {
        launch()?;
    }
    // SAFETY: new_event binds this thread to device, and stream remains live.
    unsafe { stream.synchronize()? };
    let start = device.new_event()?;
    let end = device.new_event()?;
    let mut samples = Vec::with_capacity(30);
    for _ in 0..30 {
        start.record(stream)?;
        launch()?;
        end.record(stream)?;
        end.synchronize()?;
        samples.push(start.elapsed_time(&end)?);
    }
    Ok(median(samples) * 1000.0)
}

fn run(
    b: usize,
    device: &Arc<Device>,
    stream: &Arc<Stream>,
) -> Result<(), Box<dyn std::error::Error>> {
    let inputs = inputs(b, stream)?;
    let mut prewarm = api::zeros::<f32>(&[H, D, D]).sync_on(stream)?;
    let k0 = inputs.k.slice(&[0..H, 0..D])?;
    let v0 = inputs.v.slice(&[0..H, 0..D])?;
    for rows in [128, 32, 16] {
        match rows {
            128 => kernels::state_step((&mut prewarm).partition([1, 128, D]), &k0, &v0)
                .generics(vec!["128".to_string()])
                .sync_on(stream)?,
            32 => kernels::state_step((&mut prewarm).partition([1, 32, D]), &k0, &v0)
                .generics(vec!["32".to_string()])
                .sync_on(stream)?,
            16 => kernels::state_step((&mut prewarm).partition([1, 16, D]), &k0, &v0)
                .generics(vec!["16".to_string()])
                .sync_on(stream)?,
            _ => unreachable!(),
        };
    }
    drop(prewarm);
    let mut linear = separate_state(b, stream)?;
    let mut linear_p = separate_state(b, stream)?;
    let mut linear_pp = separate_state(b, stream)?;
    let mut forked = separate_state(b, stream)?;
    let mut stacked = stacked_state(b, stream)?;
    let mut stacked_p = stacked_state(b, stream)?;
    let branches: Vec<_> = (0..b)
        .map(|_| device.new_stream())
        .collect::<Result<_, _>>()?;

    let start = Instant::now();
    let v1 = capture_linear::<128>(stream, &mut linear, &inputs)?;
    let v1_capture = start.elapsed().as_secs_f64() * 1e6;
    let start = Instant::now();
    let v1p = capture_linear::<32>(stream, &mut linear_p, &inputs)?;
    let v1p_capture = start.elapsed().as_secs_f64() * 1e6;
    let start = Instant::now();
    let v1pp = capture_linear::<16>(stream, &mut linear_pp, &inputs)?;
    let v1pp_capture = start.elapsed().as_secs_f64() * 1e6;
    let start = Instant::now();
    let v2 = capture_forked(device, stream, &branches, &mut forked, &inputs)?;
    let v2_capture = start.elapsed().as_secs_f64() * 1e6;
    let start = Instant::now();
    let v3 = capture_stacked::<128>(stream, &mut stacked, &inputs)?;
    let v3_capture = start.elapsed().as_secs_f64() * 1e6;
    let start = Instant::now();
    let v3p = capture_stacked::<32>(stream, &mut stacked_p, &inputs)?;
    let v3p_capture = start.elapsed().as_secs_f64() * 1e6;

    let v1_us = time_graph(device, stream, || {
        // SAFETY: graph and buffers stay alive until the end event synchronizes.
        unsafe { v1.launch().async_on(stream)? };
        Ok(())
    })?;
    let v1p_us = time_graph(device, stream, || {
        // SAFETY: graph and buffers stay alive until the end event synchronizes.
        unsafe { v1p.launch().async_on(stream)? };
        Ok(())
    })?;
    let v1pp_us = time_graph(device, stream, || {
        // SAFETY: graph and buffers stay alive until the end event synchronizes.
        unsafe { v1pp.launch().async_on(stream)? };
        Ok(())
    })?;
    let v2_us = time_graph(device, stream, || v2.launch(stream))?;
    let v3_us = time_graph(device, stream, || {
        // SAFETY: graph and buffers stay alive until the end event synchronizes.
        unsafe { v3.launch().async_on(stream)? };
        Ok(())
    })?;
    let v3p_us = time_graph(device, stream, || {
        // SAFETY: graph and buffers stay alive until the end event synchronizes.
        unsafe { v3p.launch().async_on(stream)? };
        Ok(())
    })?;
    drop(v1);
    drop(v1p);
    drop(v1pp);
    drop(v2);
    drop(v3);
    drop(v3p);
    let state_len = H * D * D;
    for layer in 0..L {
        let stacked_layer = stacked.remove(0).to_host_vec().sync_on(stream)?;
        let stacked_p_layer = stacked_p.remove(0).to_host_vec().sync_on(stream)?;
        for session in 0..b {
            let linear_state = linear[session].remove(0).to_host_vec().sync_on(stream)?;
            let linear_p_state = linear_p[session].remove(0).to_host_vec().sync_on(stream)?;
            let linear_pp_state = linear_pp[session].remove(0).to_host_vec().sync_on(stream)?;
            let forked_state = forked[session].remove(0).to_host_vec().sync_on(stream)?;
            let offset = session * state_len;
            assert_eq!(
                linear_state, forked_state,
                "V1/V2 state mismatch at {session}/{layer}"
            );
            assert_eq!(
                linear_state, linear_p_state,
                "V1/V1′ state mismatch at {session}/{layer}"
            );
            assert_eq!(
                linear_state, linear_pp_state,
                "V1/V1″ state mismatch at {session}/{layer}"
            );
            assert_eq!(
                linear_state.as_slice(),
                &stacked_layer[offset..offset + state_len],
                "V1/V3 state mismatch at {session}/{layer}"
            );
            assert_eq!(
                linear_state.as_slice(),
                &stacked_p_layer[offset..offset + state_len],
                "V1/V3′ state mismatch at {session}/{layer}"
            );
            for head in 0..H {
                let global_head = session * H + head;
                for row in [0, 31, 63, 127] {
                    for col in [0, 63, 127] {
                        let kh = 1.0 + (global_head % 7) as f32 / 16.0 + (row % 11) as f32 / 128.0;
                        let vh = 1.0 - (global_head % 5) as f32 / 32.0 + (col % 13) as f32 / 256.0;
                        let expected = 2.0 * kh * vh;
                        let actual = linear_state[head * D * D + row * D + col];
                        assert!(
                            (actual - expected).abs() < 1e-5,
                            "unexpected state value at {session}/{layer}/{head}/{row}/{col}"
                        );
                    }
                }
            }
        }
    }
    let theory = 4.0 * 1024.0 * 1024.0 * b as f64 * L as f64 / PEAK_BYTES_PER_US;
    let launch_gap = if b > 1 {
        (v1p_us - v3p_us) / (L * (b - 1)) as f64
    } else {
        f64::NAN
    };
    println!(
        "{b},{v1_us:.3},{v1p_us:.3},{v1pp_us:.3},{v2_us:.3},{v3_us:.3},{v3p_us:.3},{theory:.3},{:.3},{launch_gap:.3},{v1_capture:.1},{v1p_capture:.1},{v1pp_capture:.1},{v2_capture:.1},{v3_capture:.1},{v3p_capture:.1}",
        v1p_us / v3p_us
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    println!(
        "B,V1_us,V1p_us,V1pp_us,V2_us,V3_us,V3p_us,theory_us,V1p/V3p,launch_gap_us,V1_capture_us,V1p_capture_us,V1pp_capture_us,V2_capture_us,V3_capture_us,V3p_capture_us"
    );
    for b in [1, 2, 4, 8] {
        run(b, &device, &stream)?;
    }
    Ok(())
}
