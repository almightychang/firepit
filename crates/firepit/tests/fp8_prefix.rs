use cuda_core::Device;
use firepit::{Engine, LoadOptions, Model, Result, SessionId};
use std::path::Path;
use std::sync::Arc;

fn logits(dir: &Path, device: &Arc<Device>) -> Result<Vec<f32>> {
    let model = Arc::new(Model::from_checkpoint(
        dir,
        LoadOptions {
            max_seq: 32,
            layers: Some(1),
        },
        device,
    )?);
    let mut engine = Engine::new(model, device)?;
    let mut session = engine.new_session(SessionId(1))?;
    engine.prefill(&mut session, &[760, 6511, 314, 9338, 369])
}

#[test]
#[ignore = "requires local official FP8 and BF16 checkpoints"]
fn fp8_and_bf16_prefix_logits_difference() -> Result<()> {
    let models = Path::new(env!("HOME")).join("Models");
    let device = Device::new(0)?;
    println!("loading FP8 prefix");
    let fp8 = logits(&models.join("qwen3.6-35b-a3b-fp8"), &device)?;
    println!("loading BF16 prefix");
    let bf16 = logits(&models.join("qwen3.6-35b-a3b"), &device)?;
    assert_eq!(fp8.len(), bf16.len());
    let max_abs = fp8
        .iter()
        .zip(&bf16)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rms = (fp8
        .iter()
        .zip(&bf16)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum::<f64>()
        / fp8.len() as f64)
        .sqrt();
    let top = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    println!(
        "FP8 vs BF16 prefix layers=1 logits={} max_abs={max_abs} rms={rms} fp8_top={} bf16_top={}",
        fp8.len(),
        top(&fp8),
        top(&bf16)
    );
    assert!(max_abs.is_finite() && rms.is_finite());
    Ok(())
}
