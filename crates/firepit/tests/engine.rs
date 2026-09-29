use cuda_core::Device;
use firepit::{Batch, Engine, Execution, Model, ModelConfig, Session};
use std::sync::Arc;

const VOCAB: usize = 512;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn setup() -> Result<(Arc<Model>, Arc<cuda_core::Device>), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let model = Arc::new(Model::synthetic(ModelConfig::default(), &stream)?);
    Ok((model, device))
}

fn compare(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn make_sessions(
    model: &Model,
    engine: &Engine,
    n: usize,
) -> Result<Vec<Session>, Box<dyn std::error::Error>> {
    (0..n)
        .map(|i| model.new_session(i as u64 + 1, engine.stream()))
        .collect()
}

fn batched(
    engine: &mut Engine,
    sessions: &mut [Session],
    tokens: &[u32],
    execution: Execution,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let refs: Vec<&mut Session> = sessions.iter_mut().collect();
    engine.step_with(Batch::new(refs, tokens.to_vec())?, execution)
}

#[test]
fn batch_matches_individual_steps() -> TestResult {
    let (model, device) = setup()?;
    for b in [1, 2, 4] {
        let mut together = Engine::new(model.clone(), &device)?;
        let mut sessions = make_sessions(&model, &together, b)?;
        let mut singles: Vec<_> = (0..b)
            .map(|_| Engine::new(model.clone(), &device))
            .collect::<Result<_, _>>()?;
        let mut single_sessions: Vec<_> = singles
            .iter()
            .enumerate()
            .map(|(i, e)| model.new_session(i as u64 + 101, e.stream()))
            .collect::<Result<_, _>>()?;
        let mut worst = 0.0f32;
        for step in 0..4 {
            let tokens: Vec<u32> = (0..b)
                .map(|i| ((i * 17 + step * 5 + 3) % VOCAB) as u32)
                .collect();
            let actual = batched(&mut together, &mut sessions, &tokens, Execution::Eager)?;
            for i in 0..b {
                let expected = singles[i].step_with(
                    Batch::new(vec![&mut single_sessions[i]], vec![tokens[i]])?,
                    Execution::Eager,
                )?;
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
    let mut sa = make_sessions(&model, &a, 2)?;
    let mut sb = make_sessions(&model, &b, 2)?;
    let mut worst = 0.0f32;
    let mut changed = 0.0f32;
    for step in 0..4 {
        let out_a = batched(&mut a, &mut sa, &[3 + step, 7 + step], Execution::Eager)?;
        let out_b = batched(&mut b, &mut sb, &[31 + step, 7 + step], Execution::Eager)?;
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
    for b in [1, 2, 4] {
        let mut eager = Engine::new(model.clone(), &device)?;
        let mut graph = Engine::new(model.clone(), &device)?;
        let mut se = make_sessions(&model, &eager, b)?;
        let mut sg = make_sessions(&model, &graph, b)?;
        let mut worst = 0.0f32;
        for step in 0..4 {
            let tokens: Vec<u32> = (0..b)
                .map(|i| ((i * 19 + step * 11 + 2) % VOCAB) as u32)
                .collect();
            let a = batched(&mut eager, &mut se, &tokens, Execution::Eager)?;
            let g = batched(&mut graph, &mut sg, &tokens, Execution::Graph)?;
            worst = worst.max(compare(&a, &g));
        }
        println!("graph B={b} N=4 max_abs_error={worst:.6e}");
        assert!(worst <= 1e-4);
    }
    Ok(())
}

#[test]
fn graph_handles_reordered_batch_and_replaced_session() -> TestResult {
    let (model, device) = setup()?;
    let mut eager = Engine::new(model.clone(), &device)?;
    let mut graph = Engine::new(model.clone(), &device)?;
    let mut se = make_sessions(&model, &eager, 2)?;
    let mut sg = make_sessions(&model, &graph, 2)?;
    for step in 0..3 {
        let eager_rows = if step % 2 == 0 {
            let (left, right) = se.split_at_mut(1);
            eager.step_with(
                Batch::new(vec![&mut left[0], &mut right[0]], vec![3 + step, 11 + step])?,
                Execution::Eager,
            )?
        } else {
            let (left, right) = se.split_at_mut(1);
            eager.step_with(
                Batch::new(vec![&mut right[0], &mut left[0]], vec![11 + step, 3 + step])?,
                Execution::Eager,
            )?
        };
        let graph_rows = if step % 2 == 0 {
            let (left, right) = sg.split_at_mut(1);
            graph.step_with(
                Batch::new(vec![&mut left[0], &mut right[0]], vec![3 + step, 11 + step])?,
                Execution::Graph,
            )?
        } else {
            let (left, right) = sg.split_at_mut(1);
            graph.step_with(
                Batch::new(vec![&mut right[0], &mut left[0]], vec![11 + step, 3 + step])?,
                Execution::Graph,
            )?
        };
        assert!(compare(&eager_rows, &graph_rows) <= 1e-4);
    }
    let mut fresh_e = model.new_session(1, eager.stream())?;
    let mut fresh_g = model.new_session(1, graph.stream())?;
    let e = eager.step_with(
        Batch::new(vec![&mut fresh_e, &mut se[1]], vec![19, 21])?,
        Execution::Eager,
    )?;
    let g = graph.step_with(
        Batch::new(vec![&mut fresh_g, &mut sg[1]], vec![19, 21])?,
        Execution::Graph,
    )?;
    let error = compare(&e, &g);
    println!("reordered/replaced graph max_abs_error={error:.6e}");
    assert!(error <= 1e-4);
    Ok(())
}

#[test]
fn rejects_dimensions_the_kernels_cannot_run() -> TestResult {
    let stream = Device::new(0)?.new_stream()?;
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
        assert!(Model::synthetic(config, &stream).is_err());
    }
    Ok(())
}

#[test]
fn rejects_session_from_another_model() -> TestResult {
    let (model, device) = setup()?;
    let other = Model::synthetic(ModelConfig::default(), &device.new_stream()?)?;
    let mut engine = Engine::new(model, &device)?;
    let mut stranger = other.new_session(1, engine.stream())?;
    assert!(
        engine
            .step(Batch::new(vec![&mut stranger], vec![0])?)
            .is_err()
    );
    Ok(())
}
