use cuda_core::Device;
use firepit::{Engine, LoadOptions, Model, SafetensorsFile, SessionId, VerificationFault};
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
