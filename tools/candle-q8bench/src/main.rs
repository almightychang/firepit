use candle_core::cuda_backend::cudarc::driver::{CudaStream, sys};
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module, Tensor};
use half::f16;
use std::borrow::Cow;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SHAPES: [(usize, usize); 3] = [(2560, 10240), (6144, 2560), (2560, 248320)];

fn weight_bytes(rows: usize, cols: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * cols / 32 * 34);
    for block in 0..rows * cols / 32 {
        let scale = f16::from_f32(((block % 13) + 1) as f32 / 32.0);
        bytes.extend(scale.to_bits().to_le_bytes());
        for i in 0..32 {
            bytes.push((((block * 32 + i) * 17 % 31) as i8 - 15) as u8);
        }
    }
    bytes
}

fn median(mut times: Vec<f32>) -> f64 {
    times.sort_by(f32::total_cmp);
    let mid = times.len() / 2;
    if times.len() % 2 == 0 {
        (times[mid - 1] as f64 + times[mid] as f64) / 2.0
    } else {
        times[mid] as f64
    }
}

fn time_one(
    stream: &Arc<CudaStream>,
    mm: &QMatMul,
    x: &Tensor,
) -> Result<f32, Box<dyn std::error::Error>> {
    let start = stream
        .context()
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = stream
        .context()
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(stream)?;
    let output = mm.forward(x)?;
    stream.synchronize()?;
    end.record(stream)?;
    end.synchronize()?;
    let ms = start.elapsed_ms(&end)?;
    drop(output);
    Ok(ms)
}

fn measure(
    stream: &Arc<CudaStream>,
    mm: &QMatMul,
    x: &Tensor,
) -> Result<(f64, usize, usize), Box<dyn std::error::Error>> {
    let warmup = Instant::now();
    loop {
        mm.forward(x)?;
        stream.synchronize()?;
        if warmup.elapsed() >= Duration::from_millis(50) {
            break;
        }
    }
    let estimate = time_one(stream, mm, x)?.max(1e-4) as f64;
    let reps = (150.0 / estimate).round().clamp(15.0, 60.0) as usize;
    let l2 = stream
        .context()
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?
        as usize;
    let flush_bytes = (l2 * 2).clamp(64 << 20, 512 << 20);
    let mut flush = stream.alloc_zeros::<u8>(flush_bytes)?;
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        stream.memset_zeros(&mut flush)?;
        times.push(time_one(stream, mm, x)?);
    }
    Ok((median(times), reps, flush_bytes))
}

fn run(rows: usize, cols: usize, batch: usize) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = weight_bytes(rows, cols);
    let x_host: Vec<f32> = (0..batch * cols)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) / 37.0)
        .collect();
    let cpu = Device::Cpu;
    let gpu = Device::new_cuda(0)?;
    let stream = gpu.as_cuda_device()?.cuda_stream();
    let gpu_storage = QStorage::from_data(Cow::Borrowed(&bytes), &gpu, GgmlDType::Q8_0)?;
    let gpu_mm = QMatMul::from_qtensor(QTensor::new(gpu_storage, (rows, cols))?)?;
    let x_gpu = Tensor::from_vec(x_host.clone(), (batch, cols), &gpu)?;
    let (ms, reps, flush_bytes) = measure(&stream, &gpu_mm, &x_gpu)?;
    let gpu_y = gpu_mm
        .forward(&x_gpu)?
        .to_device(&cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let cpu_storage = QStorage::from_data(Cow::Borrowed(&bytes), &cpu, GgmlDType::Q8_0)?;
    let cpu_mm = QMatMul::from_qtensor(QTensor::new(cpu_storage, (rows, cols))?)?;
    let x_cpu = Tensor::from_vec(x_host, (batch, cols), &cpu)?;
    let cpu_y = cpu_mm.forward(&x_cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let max_diff = gpu_y
        .iter()
        .zip(&cpu_y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_ref = cpu_y.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let rel_err = max_diff / max_ref;
    let gbps = rows as f64 * cols as f64 * 1.0625 / (ms * 1e6);
    println!(
        "{cols},{rows},{batch},{:.3},{:.2},{:.2},{rel_err:.3e},{reps},{flush_bytes}",
        ms * 1000.0,
        gbps,
        gbps / 960.0 * 100.0
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("K,N,B,us,GB/s,peak_pct,rel_err,reps,l2_flush_bytes");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() == 3 {
        let cols: usize = args[0].parse()?;
        let rows: usize = args[1].parse()?;
        let batch: usize = args[2].parse()?;
        if !SHAPES.contains(&(cols, rows)) || ![1, 4, 8].contains(&batch) {
            return Err("unsupported K, N, B combination".into());
        }
        return run(rows, cols, batch);
    }
    if !args.is_empty() {
        return Err("usage: candle-q8bench [K N B]".into());
    }
    for (cols, rows) in SHAPES {
        for batch in [1, 4, 8] {
            run(rows, cols, batch)?;
        }
    }
    Ok(())
}
