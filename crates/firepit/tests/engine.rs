use cuda_core::Device;
use firepit::{Engine, Model, ModelConfig, Session, SessionId};
use std::sync::Arc;

const VOCAB: usize = 64;
type TestResult = Result<(), Box<dyn std::error::Error>>;

fn config() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/small-qwen.json")).unwrap()
}

fn setup() -> Result<(Arc<Model>, Arc<Device>), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let text = &config()["text_config"];
    let model = Model::random_from_config(text, 8, 7, &device.new_stream()?)?;
    Ok((Arc::new(model), device))
}

fn compare(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn sessions(engine: &Engine, n: usize) -> Result<Vec<Session>, Box<dyn std::error::Error>> {
    (0..n)
        .map(|i| engine.new_session(SessionId(i as u64 + 1)))
        .collect()
}

fn step(
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
    for size in 1..=4 {
        let mut together = Engine::new(model.clone(), &device)?;
        let mut many = sessions(&together, size)?;
        let mut single = (0..size)
            .map(|_| Engine::new(model.clone(), &device))
            .collect::<Result<Vec<_>, _>>()?;
        let mut own = single
            .iter()
            .enumerate()
            .map(|(i, engine)| engine.new_session(SessionId(i as u64 + 11)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut worst = 0.0f32;
        for t in 0..4 {
            let tokens: Vec<u32> = (0..size)
                .map(|i| ((i * 17 + t * 5 + 3) % VOCAB) as u32)
                .collect();
            let actual = step(&mut together, &mut many, &tokens, true)?;
            for i in 0..size {
                let batch = single[i].batch([(&mut own[i], tokens[i])])?;
                let expected = single[i].step_eager(batch)?;
                worst = worst.max(compare(&actual[i * VOCAB..(i + 1) * VOCAB], &expected));
            }
        }
        println!("batch B={size} max_abs={worst:.8e}");
        assert!(worst <= 1e-5);
    }
    Ok(())
}

#[test]
fn one_session_input_does_not_change_other_output() -> TestResult {
    let (model, device) = setup()?;
    let mut a = Engine::new(model.clone(), &device)?;
    let mut b = Engine::new(model, &device)?;
    let mut sa = sessions(&a, 2)?;
    let mut sb = sessions(&b, 2)?;
    let mut other = 0.0f32;
    let mut changed = 0.0f32;
    for t in 0..4 {
        let out_a = step(&mut a, &mut sa, &[3 + t, 7 + t], true)?;
        let out_b = step(&mut b, &mut sb, &[31 + t, 7 + t], true)?;
        other = other.max(compare(&out_a[VOCAB..], &out_b[VOCAB..]));
        changed = changed.max(compare(&out_a[..VOCAB], &out_b[..VOCAB]));
    }
    println!("isolation other_max_abs={other:.8e} changed_max_abs={changed:.8e}");
    assert_eq!(other, 0.0);
    assert!(changed > 1e-4);
    Ok(())
}

#[test]
fn graph_replay_matches_eager() -> TestResult {
    let (model, device) = setup()?;
    for size in 1..=4 {
        let mut eager = Engine::new(model.clone(), &device)?;
        let mut graph = Engine::new(model.clone(), &device)?;
        let mut se = sessions(&eager, size)?;
        let mut sg = sessions(&graph, size)?;
        for t in 0..4 {
            let tokens: Vec<u32> = (0..size)
                .map(|i| ((i * 19 + t * 11 + 2) % VOCAB) as u32)
                .collect();
            let a = step(&mut eager, &mut se, &tokens, true)?;
            let g = step(&mut graph, &mut sg, &tokens, false)?;
            assert_eq!(a, g);
        }
    }
    Ok(())
}

#[test]
fn graph_recaptures_for_three_and_four_sessions() -> TestResult {
    let (model, device) = setup()?;
    let mut eager = Engine::new(model.clone(), &device)?;
    let mut graph = Engine::new(model, &device)?;
    let mut se = sessions(&eager, 4)?;
    let mut sg = sessions(&graph, 4)?;
    for width in [3, 4, 3] {
        let tokens = [3, 7, 11, 13];
        let a = step(&mut eager, &mut se[..width], &tokens[..width], true)?;
        let g = step(&mut graph, &mut sg[..width], &tokens[..width], false)?;
        assert_eq!(a, g);
    }
    Ok(())
}

#[test]
fn graph_handles_reordered_batch_and_replaced_session() -> TestResult {
    let (model, device) = setup()?;
    let mut eager = Engine::new(model.clone(), &device)?;
    let mut graph = Engine::new(model, &device)?;
    let mut se = sessions(&eager, 2)?;
    let mut sg = sessions(&graph, 2)?;
    for t in 0..3 {
        let a = if t % 2 == 0 {
            let (left, right) = se.split_at_mut(1);
            let batch = eager.batch([(&mut left[0], 3 + t), (&mut right[0], 11 + t)])?;
            eager.step_eager(batch)?
        } else {
            let (left, right) = se.split_at_mut(1);
            let batch = eager.batch([(&mut right[0], 11 + t), (&mut left[0], 3 + t)])?;
            eager.step_eager(batch)?
        };
        let g = if t % 2 == 0 {
            let (left, right) = sg.split_at_mut(1);
            let batch = graph.batch([(&mut left[0], 3 + t), (&mut right[0], 11 + t)])?;
            graph.step(batch)?
        } else {
            let (left, right) = sg.split_at_mut(1);
            let batch = graph.batch([(&mut right[0], 11 + t), (&mut left[0], 3 + t)])?;
            graph.step(batch)?
        };
        assert_eq!(a, g);
    }
    let mut fresh_e = eager.new_session(SessionId(1))?;
    let mut fresh_g = graph.new_session(SessionId(1))?;
    let eb = eager.batch([(&mut fresh_e, 19), (&mut se[1], 21)])?;
    let gb = graph.batch([(&mut fresh_g, 19), (&mut sg[1], 21)])?;
    assert_eq!(eager.step_eager(eb)?, graph.step(gb)?);
    Ok(())
}

#[test]
fn validates_config_and_batch_inputs() -> TestResult {
    let config = config();
    let parsed = ModelConfig::try_from(config.clone())?;
    assert_eq!(parsed.dims(8, None)?.layers().len(), 2);
    assert!(parsed.dims(0, None).is_err());
    assert!(parsed.dims(8, Some(3)).is_err());
    let mut bad = config;
    bad["text_config"]["layer_types"][0] = "unknown".into();
    assert!(ModelConfig::try_from(bad).is_err());
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
    let batch = engine.batch([(&mut a, 0)])?;
    assert!(other.step(batch).is_err());
    Ok(())
}

#[test]
fn close_session_and_max_seq() -> TestResult {
    let (model, device) = setup()?;
    let mut engine = Engine::new(model, &device)?;
    let mut session = engine.new_session(SessionId(1))?;
    for _ in 0..8 {
        let batch = engine.batch([(&mut session, 1)])?;
        engine.step(batch)?;
    }
    assert!(engine.batch([(&mut session, 1)]).is_err());
    engine.close_session(session)?;
    let mut replacement = engine.new_session(SessionId(1))?;
    let batch = engine.batch([(&mut replacement, 1)])?;
    engine.step(batch)?;
    Ok(())
}
