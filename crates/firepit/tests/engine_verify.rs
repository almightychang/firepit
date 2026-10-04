use cuda_core::Device;
use firepit::{
    Engine, LoadOptions, Model, RoutingLayerTrace, SafetensorsFile, SessionId,
    SessionStateSnapshot, VerificationFault,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

fn model_path() -> PathBuf {
    std::env::var_os("FIREPIT_VERIFY_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join("Models/qwen3.6-35b-a3b")
        })
}

fn reference_path() -> PathBuf {
    std::env::var_os("FIREPIT_VERIFY_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../refs/qwen36-engine-prefill.safetensors")
        })
}

fn layers() -> TestResult<Option<usize>> {
    Ok(std::env::var("FIREPIT_VERIFY_LAYERS")
        .ok()
        .map(|x| x.parse())
        .transpose()?)
}

fn tokens() -> TestResult<Vec<u32>> {
    let path = reference_path();
    let path = if path.exists() {
        path
    } else if layers()?.is_some() {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4-prefill.safetensors")
    } else {
        path
    };
    let reference = SafetensorsFile::open(path)?;
    let values: Vec<u32> = serde_json::from_str(
        reference
            .metadata("tokens")
            .ok_or("reference has no tokens")?,
    )?;
    if values.len() != 150 {
        return Err(format!("expected 150 tokens, got {}", values.len()).into());
    }
    Ok(values)
}

fn setup() -> TestResult<(Arc<Model>, Arc<Device>, Vec<u32>)> {
    let tokens = tokens()?;
    let device = Device::new(0)?;
    let model = Arc::new(Model::from_checkpoint(
        model_path(),
        LoadOptions {
            max_seq: tokens.len(),
            layers: layers()?,
        },
        &device,
    )?);
    Ok((model, device, tokens))
}

fn top(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    indices.truncate(count);
    indices
}

fn metrics(actual: &[f32], expected: &[f32]) -> (f64, f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut squared = 0.0f64;
    let mut norm = 0.0f64;
    for (&a, &e) in actual.iter().zip(expected) {
        let diff = (a as f64 - e as f64).abs();
        max_abs = max_abs.max(diff);
        max_rel = max_rel.max(diff / (e as f64).abs().max(f32::MIN_POSITIVE as f64));
        squared += diff * diff;
        norm += (e as f64) * (e as f64);
    }
    (
        max_abs,
        max_rel,
        squared.sqrt() / norm.sqrt().max(f64::MIN_POSITIVE),
    )
}

fn show(label: &str, actual: &[f32], expected: &[f32]) {
    assert!(
        actual.iter().chain(expected).all(|x| x.is_finite()),
        "{label} contains non-finite values"
    );
    let (abs, rel, l2) = metrics(actual, expected);
    println!("{label:<24} max_abs={abs:.6e} max_rel={rel:.6e} rel_l2={l2:.6e}");
}

fn assert_top(label: &str, actual: &[f32], expected: &[f32]) {
    let actual_top = top(actual, 5);
    let expected_top = top(expected, 5);
    println!(
        "{label} top1={} expected={} top5={actual_top:?} expected_top5={expected_top:?}",
        actual_top[0], expected_top[0]
    );
    assert_eq!(actual_top[0], expected_top[0], "{label} top1");
    let mut a = actual_top;
    let mut e = expected_top;
    a.sort_unstable();
    e.sort_unstable();
    assert_eq!(a, e, "{label} top5 set");
}

fn compare_states(
    label: &str,
    left: &Engine,
    ls: &firepit::Session,
    right: &Engine,
    rs: &firepit::Session,
) -> TestResult<()> {
    assert_eq!(ls.position(), rs.position());
    let left_state = left.state_snapshot(ls)?;
    let right_state = right.state_snapshot(rs)?;
    assert_eq!(left_state.len(), right_state.len());
    for (a, b) in left_state.iter().zip(right_state) {
        assert_eq!((a.layer, a.name), (b.layer, b.name));
        show(
            &format!("{label} L{} {}", a.layer, a.name),
            &a.values,
            &b.values,
        );
    }
    Ok(())
}

fn relative_l2<'a>(pairs: impl IntoIterator<Item = (&'a f32, &'a f32)>) -> f64 {
    let (error, norm) = pairs.into_iter().fold((0.0, 0.0), |(error, norm), (a, b)| {
        let diff = *a as f64 - *b as f64;
        (error + diff * diff, norm + (*b as f64) * (*b as f64))
    });
    error.sqrt() / norm.sqrt().max(f64::MIN_POSITIVE)
}

fn margin(logits: &[f32], k: usize) -> f32 {
    assert!(k < logits.len());
    let ranks = top(logits, k + 1);
    logits[ranks[k - 1]] - logits[ranks[k]]
}

fn sorted_ids(ids: &[i32]) -> Vec<i32> {
    let mut ids = ids.to_vec();
    ids.sort_unstable();
    ids
}

fn print_routing_comparison(
    label: &str,
    left: &[RoutingLayerTrace],
    right: &[RoutingLayerTrace],
    left_state: &[SessionStateSnapshot],
    right_state: &[SessionStateSnapshot],
) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left_state.len(), right_state.len());
    assert_eq!(left_state.len(), 2 * left.len());
    println!(
        "{label}: layer | flips | flip_margin_min | flip_margin_max | all_margin_min | state_max_rel_l2 | next_input_rel_l2 | state_parts"
    );
    let mut first = None;
    for (layer, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(a.router_logits.len(), b.router_logits.len());
        assert_eq!(a.selected.len(), b.selected.len());
        assert_eq!(a.hidden.len(), b.hidden.len());
        assert_eq!(a.router_logits.len(), a.selected.len());
        assert_eq!(a.router_logits.len(), a.hidden.len());
        let mut flips = 0;
        let mut flip_min = f32::INFINITY;
        let mut flip_max = f32::NEG_INFINITY;
        let mut all_min = f32::INFINITY;
        for token in 0..a.selected.len() {
            assert_eq!(a.router_logits[token].len(), b.router_logits[token].len());
            assert_eq!(a.selected[token].len(), b.selected[token].len());
            assert_eq!(a.hidden[token].len(), b.hidden[token].len());
            let a_margin = margin(&a.router_logits[token], a.selected[token].len());
            let b_margin = margin(&b.router_logits[token], b.selected[token].len());
            all_min = all_min.min(a_margin).min(b_margin);
            if sorted_ids(&a.selected[token]) != sorted_ids(&b.selected[token]) {
                flips += 1;
                flip_min = flip_min.min(a_margin).min(b_margin);
                flip_max = flip_max.max(a_margin).max(b_margin);
                if first.is_none() {
                    let router_max_abs = a.router_logits[token]
                        .iter()
                        .zip(&b.router_logits[token])
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    first = Some((
                        layer,
                        token,
                        router_max_abs,
                        a_margin,
                        b_margin,
                        sorted_ids(&a.selected[token]),
                        sorted_ids(&b.selected[token]),
                    ));
                }
            }
        }
        let states_a = &left_state[2 * layer..2 * layer + 2];
        let states_b = &right_state[2 * layer..2 * layer + 2];
        let state_parts: Vec<_> = states_a
            .iter()
            .zip(states_b)
            .map(|(a, b)| {
                assert_eq!((a.layer, a.name), (b.layer, b.name));
                (a.name, relative_l2(a.values.iter().zip(&b.values)))
            })
            .collect();
        let state_rel_l2 = state_parts
            .iter()
            .map(|(_, value)| *value)
            .fold(0.0f64, f64::max);
        let hidden_rel_l2 = relative_l2(
            a.hidden
                .iter()
                .zip(&b.hidden)
                .flat_map(|(a, b)| a.iter().zip(b)),
        );
        let flip_range = if flips == 0 {
            "- | -".to_string()
        } else {
            format!("{flip_min:.6e} | {flip_max:.6e}")
        };
        println!(
            "{layer:>5} | {flips:>5} | {flip_range} | {all_min:.6e} | {state_rel_l2:.6e} | {hidden_rel_l2:.6e} | {}:{:.6e},{}:{:.6e}",
            state_parts[0].0, state_parts[0].1, state_parts[1].0, state_parts[1].1,
        );
    }
    if let Some((layer, token, max_abs, left_margin, right_margin, left_ids, right_ids)) = first {
        println!(
            "{label} first_flip layer={layer} token={token} router_max_abs={max_abs:.6e} left_margin={left_margin:.6e} right_margin={right_margin:.6e} left_ids={left_ids:?} right_ids={right_ids:?}"
        );
    } else {
        println!("{label} first_flip=none");
    }
}

#[test]
#[ignore = "requires Qwen3.6 checkpoint"]
fn routing_flip_prefill_vs_step() -> TestResult<()> {
    let (model, device, tokens) = setup()?;
    let mut prefill = Engine::new(model.clone(), &device)?;
    let mut decode = Engine::new(model, &device)?;
    let mut ps = prefill.new_session(SessionId(1))?;
    let mut ds = decode.new_session(SessionId(1))?;
    let (_, prefill_trace) = prefill.prefill_routing_trace(&mut ps, &tokens)?;
    let mut decode_trace: Vec<RoutingLayerTrace> = Vec::new();
    for &token in &tokens {
        let (_, step_trace) = decode.step_eager_routing_trace(&mut ds, token)?;
        if decode_trace.is_empty() {
            decode_trace = step_trace;
        } else {
            assert_eq!(decode_trace.len(), step_trace.len());
            for (layer, step) in decode_trace.iter_mut().zip(step_trace) {
                layer.append(step);
            }
        }
    }
    print_routing_comparison(
        "prefill/step",
        &prefill_trace,
        &decode_trace,
        &prefill.state_snapshot(&ps)?,
        &decode.state_snapshot(&ds)?,
    );
    Ok(())
}

#[test]
#[ignore = "requires Qwen3.6 checkpoint"]
fn routing_flip_append_vs_whole() -> TestResult<()> {
    let (model, device, tokens) = setup()?;
    let mut append = Engine::new(model.clone(), &device)?;
    let mut whole = Engine::new(model, &device)?;
    let mut asession = append.new_session(SessionId(1))?;
    let mut wsession = whole.new_session(SessionId(1))?;
    let (_, mut append_trace) = append.prefill_routing_trace(&mut asession, &tokens[..100])?;
    let (_, second_trace) = append.prefill_routing_trace(&mut asession, &tokens[100..])?;
    assert_eq!(append_trace.len(), second_trace.len());
    for (layer, second) in append_trace.iter_mut().zip(second_trace) {
        layer.append(second);
    }
    let (_, whole_trace) = whole.prefill_routing_trace(&mut wsession, &tokens)?;
    print_routing_comparison(
        "append/whole",
        &append_trace,
        &whole_trace,
        &append.state_snapshot(&asession)?,
        &whole.state_snapshot(&wsession)?,
    );
    Ok(())
}

#[test]
#[ignore = "requires Qwen3.6 checkpoint and full-layer engine reference"]
fn hf_prefill() -> TestResult<()> {
    let reference = SafetensorsFile::open(reference_path())?;
    assert_eq!(reference.metadata("mode"), Some("engine"));
    let (model, device, tokens) = setup()?;
    assert_eq!(
        reference.metadata("layers").unwrap().parse::<usize>()?,
        model.dims().layers().len()
    );
    let mut engine = Engine::new(model, &device)?;
    let mut session = engine.new_session(SessionId(1))?;
    let (actual, hidden) =
        engine.prefill_observed(&mut session, &tokens, VerificationFault::None)?;
    assert_eq!(hidden.len(), engine.state_snapshot(&session)?.len() / 2);
    let expected = reference.tensor("logits")?.data;
    show("HF logits", &actual, &expected);
    for (layer, row) in hidden.iter().enumerate() {
        show(
            &format!("HF L{layer} hidden"),
            row,
            &reference.tensor(&format!("layer{layer}/hidden"))?.data,
        );
    }
    assert_top("HF logits", &actual, &expected);
    Ok(())
}

#[test]
#[ignore = "requires Qwen3.6 checkpoint"]
fn prefill_vs_step() -> TestResult<()> {
    let (model, device, tokens) = setup()?;
    let mut prefill = Engine::new(model.clone(), &device)?;
    let mut decode = Engine::new(model, &device)?;
    let mut ps = prefill.new_session(SessionId(1))?;
    let mut ds = decode.new_session(SessionId(1))?;
    let actual = prefill.prefill(&mut ps, &tokens)?;
    let mut expected = Vec::new();
    for token in tokens {
        let batch = decode.batch([(&mut ds, token)])?;
        expected = decode.step_eager(batch)?;
    }
    show("prefill vs step logits", &actual, &expected);
    compare_states("prefill/step", &prefill, &ps, &decode, &ds)?;
    assert_top("prefill vs step", &actual, &expected);
    Ok(())
}

#[test]
#[ignore = "requires Qwen3.6 checkpoint"]
fn append_vs_whole() -> TestResult<()> {
    let (model, device, tokens) = setup()?;
    let mut append = Engine::new(model.clone(), &device)?;
    let mut whole = Engine::new(model, &device)?;
    let mut asession = append.new_session(SessionId(1))?;
    let mut wsession = whole.new_session(SessionId(1))?;
    append.prefill(&mut asession, &tokens[..100])?;
    let actual = append.prefill(&mut asession, &tokens[100..])?;
    let expected = whole.prefill(&mut wsession, &tokens)?;
    show("append vs whole logits", &actual, &expected);
    compare_states("append/whole", &append, &asession, &whole, &wsession)?;
    assert_top("append vs whole", &actual, &expected);
    Ok(())
}

#[test]
#[ignore = "requires Qwen3.6 checkpoint and a --tokens-json engine reference"]
fn hf_locate_divergence() -> TestResult<()> {
    let reference = SafetensorsFile::open(reference_path())?;
    let tokens: Vec<u32> = serde_json::from_str(
        reference
            .metadata("tokens")
            .ok_or("reference has no tokens")?,
    )?;
    let device = Device::new(0)?;
    let model = Arc::new(Model::from_checkpoint(
        model_path(),
        LoadOptions {
            max_seq: tokens.len(),
            layers: layers()?,
        },
        &device,
    )?);
    let mut engine = Engine::new(model, &device)?;
    let mut session = engine.new_session(SessionId(1))?;
    let (logits, trace) = engine.prefill_routing_trace(&mut session, &tokens)?;
    let last = tokens.len() - 1;
    println!("layer | firepit experts | HF experts | same | hidden rel_l2");
    for (layer, observed) in trace.iter().enumerate() {
        let ours = sorted_ids(&observed.selected[last]);
        let theirs = sorted_ids(
            &reference
                .tensor(&format!("layer{layer}/experts"))?
                .data
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>(),
        );
        let expected = reference.tensor(&format!("layer{layer}/hidden"))?.data;
        let rel = relative_l2(observed.hidden[last].iter().zip(&expected));
        println!(
            "{layer:5} | {ours:?} | {theirs:?} | {} | {rel:.6e}",
            ours == theirs
        );
    }
    let expected = reference.tensor("logits")?.data;
    println!("firepit top5 {:?}", top(&logits, 5));
    println!("HF top5 {:?}", top(&expected, 5));
    Ok(())
}

#[test]
#[ignore = "requires Qwen3.6 checkpoint and --isolate-layers engine reference"]
fn hf_isolated_layers() -> TestResult<()> {
    let reference = SafetensorsFile::open(reference_path())?;
    let tokens: Vec<u32> = serde_json::from_str(
        reference
            .metadata("tokens")
            .ok_or("reference has no tokens")?,
    )?;
    let indices: Vec<usize> = serde_json::from_str(
        reference
            .metadata("isolate_layers")
            .ok_or("reference has no isolate_layers")?,
    )?;
    let device = Device::new(0)?;
    let model = Arc::new(Model::from_checkpoint(
        model_path(),
        LoadOptions {
            max_seq: tokens.len(),
            layers: layers()?,
        },
        &device,
    )?);
    let mut engine = Engine::new(model, &device)?;
    println!("layer | mixer rel_l2 | moe rel_l2 | hidden rel_l2 | firepit experts | HF experts");
    for layer in indices {
        let input = reference.tensor(&format!("layer{layer}/input"))?;
        let actual = engine.prefill_isolated_layer(&tokens, layer, &input.data)?;
        let expected_mixer = reference.tensor(&format!("layer{layer}/mixer"))?;
        let expected_moe = reference.tensor(&format!("layer{layer}/moe"))?;
        let expected_hidden = reference.tensor(&format!("layer{layer}/hidden"))?;
        assert_eq!(
            actual.mixer.len(),
            expected_mixer.data.len(),
            "layer {layer} mixer shape"
        );
        assert_eq!(
            actual.moe.len(),
            expected_moe.data.len(),
            "layer {layer} MoE shape"
        );
        assert_eq!(
            actual.hidden.len(),
            expected_hidden.data.len(),
            "layer {layer} hidden shape"
        );
        let experts = reference.tensor(&format!("layer{layer}/experts"))?;
        let theirs = sorted_ids(
            &experts
                .data
                .iter()
                .map(|&value| value as i32)
                .collect::<Vec<_>>(),
        );
        let ours = sorted_ids(&actual.selected);
        println!(
            "{layer:5} | {:.6e} | {:.6e} | {:.6e} | {ours:?} | {theirs:?}",
            relative_l2(actual.mixer.iter().zip(&expected_mixer.data)),
            relative_l2(actual.moe.iter().zip(&expected_moe.data)),
            relative_l2(actual.hidden.iter().zip(&expected_hidden.data)),
        );
    }
    Ok(())
}
