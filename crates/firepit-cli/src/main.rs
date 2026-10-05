mod catalog;
mod remote;
mod store;

use catalog::{MEASUREMENTS, ModelRef, RECIPES};
use clap::{Parser, Subcommand};
use firepit::Result;
use remote::HuggingFace;
use store::Store;

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
    }
}
