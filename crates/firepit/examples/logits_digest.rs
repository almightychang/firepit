use cuda_core::Device;
use firepit::generation::greedy;
use firepit::{Engine, LoadOptions, Model, Result, SessionId};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

// Greedy decode that prints, for the prefill and every decode step, the chosen token and a
// digest of the logits bits, so two builds can be compared bit for bit across processes.
struct Args {
    model: PathBuf,
    prompt: String,
    steps: usize,
    max_seq: usize,
}

fn parse_args() -> Result<Args> {
    let mut values = env::args().skip(1);
    let mut model = None;
    let mut prompt = None;
    let mut steps = None;
    let mut max_seq = None;
    while let Some(flag) = values.next() {
        let value = values.next().ok_or(format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--prompt" => prompt = Some(value),
            "--steps" => steps = Some(value.parse()?),
            "--max-seq" => max_seq = Some(value.parse()?),
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    Ok(Args {
        model: model.ok_or("missing --model")?,
        prompt: prompt.ok_or("missing --prompt")?,
        steps: steps.ok_or("missing --steps")?,
        max_seq: max_seq.ok_or("missing --max-seq")?,
    })
}

fn digest(logits: &[f32]) -> u64 {
    logits.iter().fold(0xcbf29ce484222325, |hash, value| {
        value
            .to_bits()
            .to_le_bytes()
            .iter()
            .fold(hash, |hash, &byte| {
                (hash ^ byte as u64).wrapping_mul(0x100000001b3)
            })
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let tokenizer = Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|error| error.to_string())?;
    let prompt = tokenizer
        .encode(args.prompt.as_str(), false)
        .map_err(|error| error.to_string())?;
    let device = Device::new(0)?;
    let model = Arc::new(Model::from_checkpoint(
        &args.model,
        LoadOptions {
            max_seq: args.max_seq,
            layers: None,
        },
        &device,
    )?);
    let mut engine = Engine::new(model, &device)?;
    let mut session = engine.new_session(SessionId(1))?;
    let mut logits = engine.prefill(&mut session, prompt.get_ids())?;
    for step in 0..=args.steps {
        let token = greedy(&logits)?;
        println!("STEP {step} token {token} digest {:016x}", digest(&logits));
        if step == args.steps {
            break;
        }
        let batch = engine.batch([(&mut session, token)])?;
        logits = engine.step(batch)?;
    }
    Ok(())
}
