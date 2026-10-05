use cuda_core::Device;
use firepit::quality::{
    KlDiagnosis, KlResult, QualityChunk, diagnose_kl, mean, quantile, t95, truncated_kl,
};
use firepit::{Engine, LoadOptions, Model, Result, ScoredRows, SessionId};
use std::collections::BTreeSet;
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
    top_kl: usize,
    only_chunks: Option<BTreeSet<usize>>,
}

fn args() -> Result<Args> {
    let mut values = env::args().skip(1);
    let (mut model, mut reference, mut chunks, mut mode, mut layers) =
        (None, None, None, None, None);
    let (mut top_kl, mut only_chunks) = (0, None);
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
            "--top-kl" => top_kl = value.parse()?,
            "--only-chunks" => {
                if value.is_empty() {
                    return Err("--only-chunks requires comma-separated indices".into());
                }
                only_chunks = Some(
                    value
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<BTreeSet<usize>, _>>()?,
                );
            }
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    let chunks = chunks.ok_or("missing --chunks")?;
    if chunks == 0 {
        return Err("--chunks must be positive".into());
    }
    if only_chunks
        .as_ref()
        .is_some_and(|set| set.iter().any(|&index| index >= chunks))
    {
        return Err("--only-chunks index must be less than --chunks".into());
    }
    Ok(Args {
        model: model.ok_or("missing --model")?,
        reference: reference.ok_or("missing --ref")?,
        chunks,
        mode: mode.ok_or("missing --mode")?,
        layers,
        top_kl,
        only_chunks,
    })
}

struct TopKlRow {
    chunk: usize,
    row: usize,
    next_token: Option<u32>,
    diagnosis: KlDiagnosis,
}

struct RowMeasurement {
    kl: KlResult,
    nll: Option<f64>,
    hf_nll: Option<f64>,
}

fn score_row(
    chunk: &QualityChunk,
    chunk_index: usize,
    offset: usize,
    logits: &[f32],
    top: &mut Vec<TopKlRow>,
    limit: usize,
) -> Result<RowMeasurement> {
    let (ids, logp, tail) = chunk.row(offset)?;
    let score = truncated_kl(ids, logp, tail, logits)?;
    if limit > 0 && (top.len() < limit || score.kl > top[top.len() - 1].diagnosis.score.kl) {
        let row = chunk.score_start + offset;
        top.push(TopKlRow {
            chunk: chunk_index,
            row,
            next_token: chunk.tokens.get(row + 1).copied(),
            diagnosis: diagnose_kl(ids, logp, logits, score),
        });
        top.sort_by(|a, b| b.diagnosis.score.kl.total_cmp(&a.diagnosis.score.kl));
        top.truncate(limit);
    }
    let target = chunk.target_token(offset)?;
    let nll = target
        .map(|token| score.target_nll(logits, token))
        .transpose()?;
    let hf_nll = chunk.target_logp(offset).map(|value| -value as f64);
    if target.is_some() && chunk.target_logp.is_some() && hf_nll.is_none() {
        return Err("missing HF target probability".into());
    }
    Ok(RowMeasurement {
        kl: score,
        nll,
        hf_nll,
    })
}

fn probabilities<T: std::fmt::Display>(pairs: &[(T, f64)]) -> String {
    pairs
        .iter()
        .map(|(id, p)| format!("({id},{p:.6e})"))
        .collect::<Vec<_>>()
        .join(" ")
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
    let selected: Vec<usize> = args.only_chunks.as_ref().map_or_else(
        || (0..args.chunks).collect(),
        |indices| indices.iter().copied().collect(),
    );
    let first = QualityChunk::open(&reference_path(&args.reference, selected[0]))?;
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
    let has_hf_targets = first.target_logp.is_some();
    let mut first = Some(first);
    let mut chunk_kl = Vec::with_capacity(selected.len());
    let mut chunk_top1 = Vec::with_capacity(selected.len());
    let mut all_kl = Vec::with_capacity(selected.len() * scored);
    let mut top_rows = Vec::with_capacity(args.top_kl.min(selected.len() * scored));
    let mut quarters_kl: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    let mut quarters_top1: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    let mut tail_max = 0.0f64;
    let mut all_nll = Vec::new();
    let mut all_hf_nll = Vec::new();
    let mut chunk_delta = Vec::new();
    let mut all_dp = Vec::new();
    let mut quarters_delta: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    println!("chunk mean_KL top1_match tail_mass_max seconds");
    for (selected_offset, &index) in selected.iter().enumerate() {
        let chunk_started = Instant::now();
        let chunk = if selected_offset == 0 {
            first.take().unwrap()
        } else {
            QualityChunk::open(&reference_path(&args.reference, index))?
        };
        if chunk.chunk_size != chunk_size
            || chunk.score_start != score_start
            || chunk.layers != layers
            || chunk.top_k != top_k
            || chunk.target_logp.is_some() != has_hf_targets
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
                    row_scores.push(score_row(
                        &chunk,
                        index,
                        offset,
                        row,
                        &mut top_rows,
                        args.top_kl,
                    )?);
                }
            }
            Mode::Decode => {
                engine.prefill(&mut session, &chunk.tokens[..chunk.score_start])?;
                for offset in 0..scored {
                    let (_, token) = chunk.scored_token(offset)?;
                    let batch = engine.batch([(&mut session, token)])?;
                    let logits = engine.step(batch)?;
                    row_scores.push(score_row(
                        &chunk,
                        index,
                        offset,
                        &logits,
                        &mut top_rows,
                        args.top_kl,
                    )?);
                }
            }
        }
        engine.close_session(session)?;
        let values: Vec<f64> = row_scores.iter().map(|row| row.kl.kl).collect();
        let top: Vec<f64> = row_scores
            .iter()
            .map(|row| f64::from(row.kl.same_top1 as u8))
            .collect();
        let mut deltas = Vec::new();
        for (offset, row) in row_scores.iter().enumerate() {
            let quarter = offset * 4 / scored;
            quarters_kl[quarter].push(row.kl.kl);
            quarters_top1[quarter].push(f64::from(row.kl.same_top1 as u8));
            tail_max = tail_max.max(row.kl.tail_mass);
            if let Some(nll) = row.nll {
                all_nll.push(nll);
                if let Some(hf_nll) = row.hf_nll {
                    let delta = nll - hf_nll;
                    all_hf_nll.push(hf_nll);
                    deltas.push(delta);
                    quarters_delta[quarter].push(delta);
                    all_dp.push((-nll).exp() - (-hf_nll).exp());
                }
            }
        }
        if has_hf_targets {
            chunk_delta.push(mean(&deltas));
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
                .map(|r| r.kl.tail_mass)
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
    println!(
        "target rows={} firepit_mean_NLL={:.6e} firepit_PPL={:.6e}",
        all_nll.len(),
        mean(&all_nll),
        mean(&all_nll).exp()
    );
    if has_hf_targets {
        all_dp.sort_by(f64::total_cmp);
        let delta = mean(&chunk_delta);
        println!(
            "HF_mean_NLL={:.6e} HF_PPL={:.6e} delta_NLL_95CI={} ln_PPL_ratio={delta:.6e}",
            mean(&all_hf_nll),
            mean(&all_hf_nll).exp(),
            format_ci(&chunk_delta)?
        );
        println!(
            "delta_p (firepit-HF) p01={:.6e} p05={:.6e} p50={:.6e} p95={:.6e} p99={:.6e}",
            quantile(&all_dp, 0.01),
            quantile(&all_dp, 0.05),
            quantile(&all_dp, 0.5),
            quantile(&all_dp, 0.95),
            quantile(&all_dp, 0.99)
        );
        println!("quarter mean_delta_NLL (firepit-HF)");
        for (quarter, values) in quarters_delta.iter().enumerate() {
            if !values.is_empty() {
                println!("{quarter:>7} {:.6e}", mean(values));
            }
        }
    }
    if args.top_kl > 0 {
        println!("top KL positions (next_token=n/a at the final row of a chunk)");
        for item in &top_rows {
            let d = &item.diagnosis;
            let next = item
                .next_token
                .map_or_else(|| "n/a".to_owned(), |id| id.to_string());
            let hf_firepit_top1 = d
                .hf_firepit_top1
                .map_or_else(|| "tail".to_owned(), |p| format!("{p:.6e}"));
            println!(
                "chunk={} row={} next_token={} KL={:.6e} top_K_term={:.6e} tail_term={:.6e} p_tail={:.6e} q_tail={:.6e}",
                item.chunk,
                item.row,
                next,
                d.score.kl,
                d.score.top_kl,
                d.score.tail_kl,
                d.score.tail_mass,
                d.score.firepit_tail_mass
            );
            println!("  HF top5: {}", probabilities(&d.hf_top));
            println!("  firepit top5: {}", probabilities(&d.firepit_top));
            println!(
                "  firepit_p(HF_top1)={:.6e} HF_p(firepit_top1)={hf_firepit_top1}",
                d.firepit_hf_top1
            );
        }
    }
    Ok(())
}
