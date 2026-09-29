use cuda_core::Device;
use firepit::{Batch, Engine, Execution, Model, ModelConfig};
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
    let model = Arc::new(Model::synthetic(ModelConfig::default(), &stream)?);
    let sm = cuda_core::simt::CudaContext::new(device.ordinal())?.multiprocessor_count()?;
    println!("device={} sm_count={sm}", device.name()?);
    for batch in [1usize, 2, 4] {
        for mode in [Execution::Eager, Execution::Graph] {
            let mut engine = Engine::new(model.clone(), &device)?;
            if batch == 1 && matches!(mode, Execution::Eager) {
                println!("row_tile={}", engine.row_tile());
            }
            let mut sessions: Vec<_> = (0..batch)
                .map(|i| model.new_session(i as u64 + 1, engine.stream()))
                .collect::<Result<_, _>>()?;
            let mut samples = Vec::new();
            for step in 0..25 {
                let tokens: Vec<u32> = (0..batch)
                    .map(|i| ((step * 7 + i * 19 + 3) % 256) as u32)
                    .collect();
                let refs = sessions.iter_mut().collect();
                let started = Instant::now();
                let logits = engine.step_with(
                    Batch::new(refs, tokens)?,
                    match mode {
                        Execution::Eager => Execution::Eager,
                        Execution::Graph => Execution::Graph,
                    },
                )?;
                assert!(logits.iter().all(|x| x.is_finite()));
                if step >= 5 {
                    samples.push(started.elapsed().as_secs_f64() * 1000.0);
                }
            }
            let label = if matches!(mode, Execution::Eager) {
                "eager"
            } else {
                "graph"
            };
            println!(
                "B={batch} {label} median_step_ms={:.4} samples={}",
                median(samples),
                20
            );
        }
    }
    Ok(())
}
