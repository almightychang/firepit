use cuda_core::Device;
use firepit::generation::{eos_ids, greedy};
use firepit::{Engine, LoadOptions, Model, Result, SessionId};
use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

struct Args {
    model: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    max_seq: usize,
    layers: Option<usize>,
    eager: bool,
}

fn parse_args() -> Result<Args> {
    let mut values = env::args().skip(1);
    let mut model = None;
    let mut prompt = None;
    let mut max_new_tokens = None;
    let mut max_seq = None;
    let mut layers = None;
    let mut eager = false;
    while let Some(flag) = values.next() {
        if flag == "--eager" {
            eager = true;
            continue;
        }
        let value = values.next().ok_or(format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--prompt" => prompt = Some(value),
            "--max-new-tokens" => max_new_tokens = Some(value.parse()?),
            "--max-seq" => max_seq = Some(value.parse()?),
            "--layers" => layers = Some(value.parse()?),
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    Ok(Args {
        model: model.ok_or("missing --model")?,
        prompt: prompt.ok_or("missing --prompt")?,
        max_new_tokens: max_new_tokens.ok_or("missing --max-new-tokens")?,
        max_seq: max_seq.ok_or("missing --max-seq")?,
        layers,
        eager,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let started = Instant::now();
    let tokenizer = Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|error| error.to_string())?;
    let prompt = tokenizer
        .encode(args.prompt.as_str(), false)
        .map_err(|error| error.to_string())?;
    let input = prompt.get_ids();
    if input.is_empty() || input.len() > args.max_seq {
        return Err("prompt is empty or exceeds max_seq".into());
    }
    let eos = eos_ids(&args.model, &tokenizer)?;
    let device = Device::new(0)?;
    let model = Arc::new(Model::from_checkpoint(
        &args.model,
        LoadOptions {
            max_seq: args.max_seq,
            layers: args.layers,
        },
        &device,
    )?);
    let mut engine = Engine::new(model, &device)?;
    let mut session = engine.new_session(SessionId(1))?;
    let prefill_started = Instant::now();
    let mut logits = engine.prefill(&mut session, input)?;
    let prefill_time = prefill_started.elapsed();
    let mut generated = Vec::new();
    let mut decode_times = Vec::new();
    let mut stream = tokenizer.decode_stream(true);
    print!("{}", args.prompt);
    io::stdout().flush()?;
    for index in 0..args.max_new_tokens {
        let token = greedy(&logits)?;
        generated.push(token);
        if eos.contains(&token) {
            break;
        }
        if let Some(chunk) = stream.step(token).map_err(|error| error.to_string())? {
            print!("{chunk}");
            io::stdout().flush()?;
        }
        if index + 1 == args.max_new_tokens || session.position() >= args.max_seq {
            break;
        }
        let step = Instant::now();
        let batch = engine.batch([(&mut session, token)])?;
        logits = if args.eager {
            engine.step_eager(batch)?
        } else {
            engine.step(batch)?
        };
        decode_times.push(step.elapsed());
    }
    println!();
    println!("token_ids: {generated:?}");
    println!("prefill_time: {:?}", prefill_time);
    println!("decode_step_times: {:?}", decode_times);
    println!("total_time: {:?}", started.elapsed());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires local Qwen3.6 checkpoint"]
    fn reference_prompt_tokens() -> Result<()> {
        let path = std::path::Path::new(env!("HOME")).join("Models/qwen3.6-35b-a3b/tokenizer.json");
        let tokenizer = Tokenizer::from_file(path).map_err(|error| error.to_string())?;
        let encoding = tokenizer
            .encode("The capital of France is", false)
            .map_err(|error| error.to_string())?;
        assert_eq!(encoding.get_ids(), &[760, 6511, 314, 9338, 369]);
        Ok(())
    }
}
