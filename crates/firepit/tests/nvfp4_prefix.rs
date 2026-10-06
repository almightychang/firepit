use cuda_core::Device;
use firepit::{Engine, LoadOptions, Model, Result, ScoredRows, SessionId};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[test]
#[ignore = "requires the ModelOpt NVFP4 checkpoint and a GPU"]
fn nvfp4_prefix_runs_prefill_and_decode() -> Result<()> {
    let dir = std::env::var_os("FIREPIT_NVFP4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("HOME")).join("Models/qwen3.6-35b-a3b-nvfp4"));
    let device = Device::new(0)?;
    let model = Arc::new(Model::from_checkpoint(
        &dir,
        LoadOptions {
            max_seq: 32,
            layers: Some(2),
        },
        &device,
    )?);
    let vocab = model.dims().vocab();
    let mut engine = Engine::new(model, &device)?;
    let tokens = [760, 6511, 314, 9338, 369];

    let mut scored = engine.new_session(SessionId(1))?;
    let rows = ScoredRows::parse(0..tokens.len(), tokens.len())?;
    let all = engine.prefill_scored(&mut scored, &tokens, rows)?;
    let mut session = engine.new_session(SessionId(2))?;
    let last = engine.prefill(&mut session, &tokens)?;
    let batch = engine.batch([(&mut session, 11)])?;
    let step = engine.step_eager(batch)?;

    assert_eq!(all.len(), tokens.len() * vocab);
    assert_eq!((last.len(), step.len()), (vocab, vocab));
    assert!(all.iter().chain(&last).chain(&step).all(|x| x.is_finite()));
    let gemm_last = &all[(tokens.len() - 1) * vocab..];
    let max_abs = gemm_last
        .iter()
        .zip(&last)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("NVFP4 layers=2 lm_head GEMM vs GEMV last row max_abs={max_abs}");
    Ok(())
}

#[test]
#[ignore = "requires the ModelOpt NVFP4 checkpoint and a GPU"]
fn nvfp4_flow_shows_the_four_projection_groups() -> Result<()> {
    use firepit::Projection::*;
    use firepit_kernels::weights::{Fp8Scaling, MatrixFormat};
    let dir = std::env::var_os("FIREPIT_NVFP4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("HOME")).join("Models/qwen3.6-35b-a3b-nvfp4"));
    let device = Device::new(0)?;
    let model = Arc::new(Model::from_checkpoint(
        &dir,
        LoadOptions {
            max_seq: 1,
            layers: Some(4),
        },
        &device,
    )?);
    let mut engine = Engine::new(model.clone(), &device)?;
    let mut session = engine.new_session(SessionId(1))?;
    let nodes = engine.step_flow(engine.batch([(&mut session, 0)])?, true)?;
    let analysis = firepit::flow::analyze(&model, &nodes);
    let mut groups: Vec<_> = analysis
        .fused()
        .map(|gemv| (gemv.members.clone(), gemv.format, gemv.padding))
        .collect();
    groups.sort_by_key(|(members, ..)| format!("{members:?}"));
    groups.dedup();
    assert_eq!(
        groups,
        [
            (
                vec![AttnQGate, AttnK, AttnV],
                MatrixFormat::Fp8(Fp8Scaling::Rows),
                0
            ),
            (vec![GdnB, GdnA], MatrixFormat::Bf16, 0),
            (vec![GdnQkv, GdnZ], MatrixFormat::Fp8(Fp8Scaling::Rows), 0),
            (vec![Router, SharedGate], MatrixFormat::Bf16, 15),
        ]
    );
    assert!(analysis.candidates().is_empty());
    assert!(analysis.unproduced.is_empty());
    Ok(())
}
