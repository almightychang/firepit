mod catalog;
mod remote;
mod store;

use catalog::{MEASUREMENTS, ModelRef, RECIPES};
use clap::{Parser, Subcommand};
use cuda_core::Device;
use firepit::generation::{eos_ids, greedy};
use firepit::{Engine, LoadOptions, Model, Result, SessionId};
use remote::HuggingFace;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use store::Store;
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(
    name = "firepit",
    about = "Run measured Qwen checkpoints on cuTile kernels"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List supported models, modes, sizes, and measured speed and quality
    Models,
    /// Download, assemble, and verify a model
    Pull { model: ModelRef },
    /// Generate from a prompt (argument or stdin), pulling the model first if needed
    Run {
        model: ModelRef,
        prompt: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        /// Run decode steps eagerly instead of through the CUDA graph
        #[arg(long)]
        eager: bool,
    },
}

fn models(store: &Store) -> Result<()> {
    println!(
        "{:<15} {:>7} {:>9} {:>15} {:>8} {:>7} {:>8}  {:<6}  description",
        "model", "size", "decode", "PPL", "KL", "top-1", "measured", "status"
    );
    for recipe in RECIPES {
        let name = format!(
            "{}:{}{}",
            recipe.model,
            recipe.mode,
            if recipe.default { "*" } else { "" }
        );
        let status = match store.installed(recipe) {
            Ok(Some(_)) => "pulled",
            Ok(None) => "-",
            Err(_) => "stale",
        };
        let measured = &recipe.measured;
        let ppl = format!(
            "{:+.2}±{:.2}%",
            100.0 * measured.delta_nll.exp_m1(),
            100.0 * measured.delta_nll_ci
        );
        println!(
            "{name:<15} {:>5.1}GB {:>5.1}t/s {ppl:>15} {:>8.2e} {:>6.2}% {:>8}  {status:<6}  {}",
            recipe.size_gb,
            1000.0 / measured.decode_ms,
            measured.kl,
            100.0 * measured.top1,
            measured.at,
            recipe.summary
        );
    }
    println!(
        "* default mode. decode: GB10 batch 1, max-seq 256. PPL change (95% CI) and KL against HF \
         bf16 on 72 WikiText-2 chunks, from {MEASUREMENTS} at the measured commit."
    );
    Ok(())
}

fn run(dir: &Path, prompt: &str, max_new_tokens: usize, eager: bool) -> Result<()> {
    let tokenizer =
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|error| error.to_string())?;
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|error| error.to_string())?;
    let input = encoding.get_ids();
    if input.is_empty() {
        return Err("prompt is empty".into());
    }
    let max_seq = input.len() + max_new_tokens;
    let eos = eos_ids(dir, &tokenizer)?;
    let device = Device::new(0)?;
    let loading = Instant::now();
    let model = Arc::new(Model::from_checkpoint(
        dir,
        LoadOptions {
            max_seq,
            layers: None,
        },
        &device,
    )?);
    eprintln!("loaded in {:.1}s", loading.elapsed().as_secs_f64());
    let mut engine = Engine::new(model, &device)?;
    let mut session = engine.new_session(SessionId(1))?;
    let started = Instant::now();
    let mut logits = engine.prefill(&mut session, input)?;
    let prefill = started.elapsed();
    let mut stream = tokenizer.decode_stream(true);
    let mut generated = 0;
    print!("{prompt}");
    io::stdout().flush()?;
    let decoding = Instant::now();
    while generated < max_new_tokens {
        let token = greedy(&logits)?;
        if eos.contains(&token) {
            break;
        }
        generated += 1;
        if let Some(chunk) = stream.step(token).map_err(|error| error.to_string())? {
            print!("{chunk}");
            io::stdout().flush()?;
        }
        if generated == max_new_tokens {
            break;
        }
        let batch = engine.batch([(&mut session, token)])?;
        logits = if eager {
            engine.step_eager(batch)?
        } else {
            engine.step(batch)?
        };
    }
    println!();
    let decode = decoding.elapsed().as_secs_f64();
    eprintln!(
        "prefill {} tokens in {:.3}s, decoded {generated} tokens at {:.1} tok/s",
        input.len(),
        prefill.as_secs_f64(),
        generated as f64 / decode
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let store = Store::from_env()?;
    match cli.command {
        Command::Models => models(&store),
        Command::Pull { model } => {
            let dir = store.pull(model.recipe(), &HuggingFace::new())?;
            println!("{model} is ready at {}", dir.display());
            Ok(())
        }
        Command::Run {
            model,
            prompt,
            max_new_tokens,
            eager,
        } => {
            let prompt = match prompt {
                Some(prompt) => prompt,
                None => {
                    let mut prompt = String::new();
                    io::stdin().read_to_string(&mut prompt)?;
                    prompt
                }
            };
            let dir = store.pull(model.recipe(), &HuggingFace::new())?;
            run(&dir, &prompt, max_new_tokens, eager)
        }
    }
}
