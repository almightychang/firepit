use cuda_core::Device;
use firepit::{Engine, Model, ModelConfig, ModelDims, SessionId};
use std::sync::Arc;
use std::time::Instant;

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

#[test]
#[ignore = "GPU timing run"]
fn step_times() -> Result<(), Box<dyn std::error::Error>> {
    assert!(!cfg!(debug_assertions), "timing needs --release");
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let model = Arc::new(Model::synthetic(
        ModelDims::try_from(ModelConfig::default())?,
        &stream,
    )?);
    let sm = cuda_core::simt::CudaContext::new(device.ordinal())?.multiprocessor_count()?;
    println!("device={} sm_count={sm}", device.name()?);
    for batch in [1usize, 2, 4] {
        for eager in [true, false] {
            let mut engine = Engine::new(model.clone(), &device)?;
            let mut sessions: Vec<_> = (0..batch)
                .map(|i| engine.new_session(SessionId(i as u64 + 1)))
                .collect::<Result<_, _>>()?;
            let mut samples = Vec::new();
            for step in 0..25 {
                let tokens: Vec<u32> = (0..batch)
                    .map(|i| ((step * 7 + i * 19 + 3) % 256) as u32)
                    .collect();
                let started = Instant::now();
                let batch = engine.batch(sessions.iter_mut().zip(tokens))?;
                let logits = if eager {
                    engine.step_eager(batch)?
                } else {
                    engine.step(batch)?
                };
                assert!(logits.iter().all(|x| x.is_finite()));
                if step >= 5 {
                    samples.push(started.elapsed().as_secs_f64() * 1000.0);
                }
            }
            let label = if eager { "eager" } else { "graph" };
            println!(
                "B={batch} {label} median_step_ms={:.4} samples={}",
                median(samples),
                20
            );
        }
    }
    Ok(())
}
