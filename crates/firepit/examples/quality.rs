use cuda_core::Device;
use firepit::quality::{QualityChunk, mean, quantile, t95, truncated_kl};
use firepit::{Engine, LoadOptions, Model, Result, ScoredRows, SessionId};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Mode {
    Prefill,
    Decode,
}

struct Args {
    model: PathBuf,
    reference: PathBuf,
    chunks: usize,
    mode: Mode,
    layers: Option<usize>,
}

fn args() -> Result<Args> {
    let mut values = env::args().skip(1);
    let (mut model, mut reference, mut chunks, mut mode, mut layers) =
        (None, None, None, None, None);
    while let Some(flag) = values.next() {
        let value = values.next().ok_or(format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--ref" => reference = Some(PathBuf::from(value)),
            "--chunks" => chunks = Some(value.parse::<usize>()?),
            "--mode" => {
                mode = Some(match value.as_str() {
                    "prefill" => Mode::Prefill,
                    "decode" => Mode::Decode,
                    _ => return Err("--mode must be prefill or decode".into()),
                })
            }
            "--layers" => layers = Some(value.parse::<usize>()?),
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    let chunks = chunks.ok_or("missing --chunks")?;
    if chunks == 0 {
        return Err("--chunks must be positive".into());
    }
    Ok(Args {
        model: model.ok_or("missing --model")?,
        reference: reference.ok_or("missing --ref")?,
        chunks,
        mode: mode.ok_or("missing --mode")?,
        layers,
    })
}

fn reference_path(root: &PathBuf, index: usize) -> PathBuf {
    root.join(format!("chunk-{index:06}.safetensors"))
}

fn format_ci(values: &[f64]) -> Result<String> {
    Ok(match t95(values)? {
        Some((center, half)) => format!("{center:.6e} ± {half:.6e}"),
        None => format!("{:.6e} ± n/a (one chunk)", mean(values)),
    })
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("quality measurement requires --release".into());
    }
    let args = args()?;
    let started = Instant::now();
    let first = QualityChunk::open(&reference_path(&args.reference, 0))?;
    let (chunk_size, score_start, top_k, layers) = (
        first.chunk_size,
        first.score_start,
        first.top_k,
        first.layers,
    );
    let device = Device::new(0)?;
    let model = Arc::new(Model::from_checkpoint(
        &args.model,
        LoadOptions {
            max_seq: chunk_size,
            layers: args.layers,
        },
        &device,
    )?);
    if layers != model.dims().layers().len() {
        return Err("reference layer count differs from model".into());
    }
    let vocab = model.dims().vocab();
    let mut engine = Engine::new(model, &device)?;
    let scored = chunk_size - score_start;
    let mut first = Some(first);
    let mut chunk_kl = Vec::with_capacity(args.chunks);
    let mut chunk_top1 = Vec::with_capacity(args.chunks);
    let mut all_kl = Vec::with_capacity(args.chunks * scored);
    let mut quarters_kl: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    let mut quarters_top1: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    let mut tail_max = 0.0f64;
    println!("chunk mean_KL top1_match tail_mass_max seconds");
    for index in 0..args.chunks {
        let chunk_started = Instant::now();
        let chunk = if index == 0 {
            first.take().unwrap()
        } else {
            QualityChunk::open(&reference_path(&args.reference, index))?
        };
        if chunk.chunk_size != chunk_size
            || chunk.score_start != score_start
            || chunk.layers != layers
            || chunk.top_k != top_k
        {
            return Err("quality chunks have inconsistent dimensions".into());
        }
        let mut session = engine.new_session(SessionId(index as u64 + 1))?;
        let mut row_scores = Vec::with_capacity(scored);
        match args.mode {
            Mode::Prefill => {
                let range =
                    ScoredRows::parse(chunk.score_start..chunk.chunk_size, chunk.chunk_size)?;
                let logits = engine.prefill_scored(&mut session, &chunk.tokens, range)?;
                if logits.len() != scored * vocab {
                    return Err("unexpected scored logits length".into());
                }
                for (offset, row) in logits.chunks_exact(vocab).enumerate() {
                    let (ids, logp, tail) = chunk.row(offset)?;
                    row_scores.push(truncated_kl(ids, logp, tail, row)?);
                }
            }
            Mode::Decode => {
                engine.prefill(&mut session, &chunk.tokens[..chunk.score_start])?;
                for offset in 0..scored {
                    let (_, token) = chunk.scored_token(offset)?;
                    let batch = engine.batch([(&mut session, token)])?;
                    let logits = engine.step(batch)?;
                    let (ids, logp, tail) = chunk.row(offset)?;
                    row_scores.push(truncated_kl(ids, logp, tail, &logits)?);
                }
            }
        }
        engine.close_session(session)?;
        let values: Vec<f64> = row_scores.iter().map(|row| row.kl).collect();
        let top: Vec<f64> = row_scores
            .iter()
            .map(|row| f64::from(row.same_top1 as u8))
            .collect();
        for (offset, row) in row_scores.iter().enumerate() {
            let quarter = offset * 4 / scored;
            quarters_kl[quarter].push(row.kl);
            quarters_top1[quarter].push(f64::from(row.same_top1 as u8));
            tail_max = tail_max.max(row.tail_mass);
        }
        let kl = mean(&values);
        let top1 = mean(&top);
        chunk_kl.push(kl);
        chunk_top1.push(top1);
        all_kl.extend(values);
        println!(
            "{index:>5} {kl:.6e} {:.4}% {:.6e} {:.1}",
            top1 * 100.0,
            row_scores
                .iter()
                .map(|r| r.tail_mass)
                .fold(0.0f64, f64::max),
            chunk_started.elapsed().as_secs_f64()
        );
    }
    all_kl.sort_by(f64::total_cmp);
    println!(
        "overall mean_KL_95CI={} median={:.6e} p99={:.6e} p99.9={:.6e} max={:.6e}",
        format_ci(&chunk_kl)?,
        quantile(&all_kl, 0.5),
        quantile(&all_kl, 0.99),
        quantile(&all_kl, 0.999),
        all_kl[all_kl.len() - 1]
    );
    println!(
        "overall top1_95CI={} tail_mass_max={tail_max:.6e} elapsed_seconds={:.1}",
        format_ci(&chunk_top1)?,
        started.elapsed().as_secs_f64()
    );
    println!("quarter mean_KL top1_match");
    for quarter in 0..4 {
        if !quarters_kl[quarter].is_empty() {
            println!(
                "{quarter:>7} {:.6e} {:.4}%",
                mean(&quarters_kl[quarter]),
                mean(&quarters_top1[quarter]) * 100.0
            );
        }
    }
    Ok(())
}
