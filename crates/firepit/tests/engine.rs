use cuda_core::Device;
use firepit::{Engine, Model, ModelConfig, ModelDims, Session, SessionId};
use std::sync::Arc;

const VOCAB: usize = 512;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn setup() -> Result<(Arc<Model>, Arc<cuda_core::Device>), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let model = Arc::new(Model::synthetic(
        ModelDims::try_from(ModelConfig::default())?,
        &stream,
    )?);
    Ok((model, device))
}

fn compare(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn make_sessions(engine: &Engine, n: usize) -> Result<Vec<Session>, Box<dyn std::error::Error>> {
    (0..n)
        .map(|i| engine.new_session(SessionId(i as u64 + 1)))
        .collect()
}

fn batched(
    engine: &mut Engine,
    sessions: &mut [Session],
    tokens: &[u32],
    eager: bool,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let batch = engine.batch(sessions.iter_mut().zip(tokens.iter().copied()))?;
    Ok(if eager {
        engine.step_eager(batch)?
    } else {
        engine.step(batch)?
    })
}

#[test]
fn batch_matches_individual_steps() -> TestResult {
    let (model, device) = setup()?;
    for b in [1, 2, 3, 4] {
        let mut together = Engine::new(model.clone(), &device)?;
        let mut sessions = make_sessions(&together, b)?;
        let mut singles: Vec<_> = (0..b)
            .map(|_| Engine::new(model.clone(), &device))
            .collect::<Result<_, _>>()?;
        let mut single_sessions: Vec<_> = singles
            .iter()
            .enumerate()
            .map(|(i, e)| e.new_session(SessionId(i as u64 + 101)))
            .collect::<Result<_, _>>()?;
        let mut worst = 0.0f32;
        for step in 0..4 {
            let tokens: Vec<u32> = (0..b)
                .map(|i| ((i * 17 + step * 5 + 3) % VOCAB) as u32)
                .collect();
            let actual = batched(&mut together, &mut sessions, &tokens, true)?;
            for i in 0..b {
                let batch = singles[i].batch([(&mut single_sessions[i], tokens[i])])?;
                let expected = singles[i].step_eager(batch)?;
                worst = worst.max(compare(&actual[i * VOCAB..(i + 1) * VOCAB], &expected));
            }
        }
        println!("batch B={b} N=4 max_abs_error={worst:.6e}");
        assert!(worst <= 1e-4);
    }
    Ok(())
}

#[test]
fn one_session_input_does_not_change_other_output() -> TestResult {
    let (model, device) = setup()?;
    let mut a = Engine::new(model.clone(), &device)?;
    let mut b = Engine::new(model.clone(), &device)?;
    let mut sa = make_sessions(&a, 2)?;
    let mut sb = make_sessions(&b, 2)?;
    let mut worst = 0.0f32;
    let mut changed = 0.0f32;
    for step in 0..4 {
        let out_a = batched(&mut a, &mut sa, &[3 + step, 7 + step], true)?;
        let out_b = batched(&mut b, &mut sb, &[31 + step, 7 + step], true)?;
        worst = worst.max(compare(&out_a[VOCAB..], &out_b[VOCAB..]));
        changed = changed.max(compare(&out_a[..VOCAB], &out_b[..VOCAB]));
    }
    println!("isolation other_max_abs={worst:.6e} changed_max_abs={changed:.6e}");
    assert_eq!(worst, 0.0);
    assert!(changed > 1e-4);
    Ok(())
}

#[test]
fn graph_replay_matches_eager() -> TestResult {
    let (model, device) = setup()?;
    for b in [1, 2, 3, 4] {
        let mut eager = Engine::new(model.clone(), &device)?;
        let mut graph = Engine::new(model.clone(), &device)?;
        let mut se = make_sessions(&eager, b)?;
        let mut sg = make_sessions(&graph, b)?;
        let mut worst = 0.0f32;
        for step in 0..4 {
            let tokens: Vec<u32> = (0..b)
                .map(|i| ((i * 19 + step * 11 + 2) % VOCAB) as u32)
                .collect();
            let a = batched(&mut eager, &mut se, &tokens, true)?;
            let g = batched(&mut graph, &mut sg, &tokens, false)?;
            worst = worst.max(compare(&a, &g));
        }
        println!("graph B={b} N=4 max_abs_error={worst:.6e}");
        assert!(worst <= 1e-4);
    }
    Ok(())
}

#[test]
fn graph_recaptures_when_three_and_four_sessions_share_a_plan_size() -> TestResult {
    let (model, device) = setup()?;
    let mut eager = Engine::new(model.clone(), &device)?;
    let mut graph = Engine::new(model, &device)?;
    let mut se = make_sessions(&eager, 4)?;
    let mut sg = make_sessions(&graph, 4)?;

    for width in [3, 4, 3] {
        let tokens = [3, 7, 11, 13];
        let a = batched(&mut eager, &mut se[..width], &tokens[..width], true)?;
        let g = batched(&mut graph, &mut sg[..width], &tokens[..width], false)?;
        assert_eq!(g.len(), width * VOCAB);
        assert!(compare(&a, &g) <= 1e-4);
    }
    Ok(())
}

#[test]
fn graph_handles_reordered_batch_and_replaced_session() -> TestResult {
    let (model, device) = setup()?;
    let mut eager = Engine::new(model.clone(), &device)?;
    let mut graph = Engine::new(model.clone(), &device)?;
    let mut se = make_sessions(&eager, 2)?;
    let mut sg = make_sessions(&graph, 2)?;
    for step in 0..3 {
        let eager_rows = if step % 2 == 0 {
            let (left, right) = se.split_at_mut(1);
            eager
                .step_eager(eager.batch([(&mut left[0], 3 + step), (&mut right[0], 11 + step)])?)?
        } else {
            let (left, right) = se.split_at_mut(1);
            eager
                .step_eager(eager.batch([(&mut right[0], 11 + step), (&mut left[0], 3 + step)])?)?
        };
        let graph_rows = if step % 2 == 0 {
            let (left, right) = sg.split_at_mut(1);
            graph.step(graph.batch([(&mut left[0], 3 + step), (&mut right[0], 11 + step)])?)?
        } else {
            let (left, right) = sg.split_at_mut(1);
            graph.step(graph.batch([(&mut right[0], 11 + step), (&mut left[0], 3 + step)])?)?
        };
        assert!(compare(&eager_rows, &graph_rows) <= 1e-4);
    }
    let mut fresh_e = eager.new_session(SessionId(1))?;
    let mut fresh_g = graph.new_session(SessionId(1))?;
    let e = eager.step_eager(eager.batch([(&mut fresh_e, 19), (&mut se[1], 21)])?)?;
    let g = graph.step(graph.batch([(&mut fresh_g, 19), (&mut sg[1], 21)])?)?;
    let error = compare(&e, &g);
    println!("reordered/replaced graph max_abs_error={error:.6e}");
    assert!(error <= 1e-4);
    Ok(())
}

#[test]
fn rejects_dimensions_the_kernels_cannot_run() -> TestResult {
    for config in [
        ModelConfig {
            vocab: 33,
            ..ModelConfig::default()
        },
        ModelConfig {
            hidden: 64,
            heads: 2,
            ..ModelConfig::default()
        },
        ModelConfig {
            head_dim: 24,
            hidden: 192,
            ..ModelConfig::default()
        },
    ] {
        assert!(ModelDims::try_from(config).is_err());
    }
    Ok(())
}

#[test]
fn rejects_session_from_another_model() -> TestResult {
    let (model, device) = setup()?;
    let other = Model::synthetic(
        ModelDims::try_from(ModelConfig::default())?,
        &device.new_stream()?,
    )?;
    let engine = Engine::new(model, &device)?;
    let mut stranger = Engine::new(Arc::new(other), &device)?.new_session(SessionId(1))?;
    assert!(engine.batch([(&mut stranger, 0)]).is_err());
    Ok(())
}

#[test]
fn batch_rejects_invalid_inputs_and_cross_engine_use() -> TestResult {
    let (model, device) = setup()?;
    let engine = Engine::new(model.clone(), &device)?;
    let mut other = Engine::new(model, &device)?;
    let mut a = engine.new_session(SessionId(1))?;
    let mut b = engine.new_session(SessionId(1))?;
    assert!(engine.batch([(&mut a, 0), (&mut b, 0)]).is_err());
    assert!(
        engine
            .batch(std::iter::empty::<(&mut Session, u32)>())
            .is_err()
    );
    assert!(engine.batch([(&mut a, VOCAB as u32)]).is_err());
    let mut c = engine.new_session(SessionId(2))?;
    let mut d = engine.new_session(SessionId(3))?;
    assert!(
        engine
            .batch([(&mut a, 0), (&mut c, 0), (&mut d, 0)])
            .is_ok()
    );
    let batch = engine.batch([(&mut a, 0)])?;
    assert!(other.step(batch).is_err());
    let short_dims = ModelDims::try_from(ModelConfig {
        max_seq: 1,
        ..ModelConfig::default()
    })?;
    let short_model = Arc::new(Model::synthetic(short_dims, &device.new_stream()?)?);
    let mut short = Engine::new(short_model, &device)?;
    let mut at_end = short.new_session(SessionId(3))?;
    let batch = short.batch([(&mut at_end, 0)])?;
    short.step(batch)?;
    assert!(short.batch([(&mut at_end, 0)]).is_err());
    Ok(())
}
