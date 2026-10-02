use cuda_core::Device;
use firepit::{Engine, LoadOptions, Model, Result, SessionId};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Args {
    model: PathBuf,
    batches: Vec<usize>,
    steps: usize,
    warmup: usize,
    max_seq: usize,
    layers: Option<usize>,
}

fn parse_args() -> Result<Args> {
    let mut values = env::args().skip(1);
    let (mut model, mut batches, mut steps, mut warmup, mut max_seq, mut layers) =
        (None, None, None, None, None, None);
    while let Some(flag) = values.next() {
        let value = values.next().ok_or(format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--batches" => {
                batches = Some(
                    value
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?,
                )
            }
            "--steps" => steps = Some(value.parse()?),
            "--warmup" => warmup = Some(value.parse()?),
            "--max-seq" => max_seq = Some(value.parse()?),
            "--layers" => layers = Some(value.parse()?),
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    Ok(Args {
        model: model.ok_or("missing --model")?,
        batches: batches.ok_or("missing --batches")?,
        steps: steps.ok_or("missing --steps")?,
        warmup: warmup.ok_or("missing --warmup")?,
        max_seq: max_seq.ok_or("missing --max-seq")?,
        layers,
    })
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.warmup + args.steps > args.max_seq {
        return Err("warmup + steps exceeds max_seq".into());
    }
    let device = Device::new(0)?;
    let loading = Instant::now();
    let model = Arc::new(Model::from_checkpoint(
        &args.model,
        LoadOptions {
            max_seq: args.max_seq,
            layers: args.layers,
        },
        &device,
    )?);
    println!("load {:.1}s", loading.elapsed().as_secs_f64());
    let mut engine = Engine::new(model, &device)?;
    let token = 760;
    for &batch in &args.batches {
        let mut sessions = (0..batch)
            .map(|id| engine.new_session(SessionId(id as u64)))
            .collect::<Result<Vec<_>>>()?;
        let mut times = Vec::with_capacity(args.steps);
        for step in 0..args.warmup + args.steps {
            let started = Instant::now();
            let entries = sessions.iter_mut().map(|session| (session, token));
            let logits = engine.step(engine.batch(entries)?)?;
            if logits.is_empty() {
                return Err("empty logits".into());
            }
            if step >= args.warmup {
                times.push(started.elapsed());
            }
        }
        times.sort();
        let median = percentile(&times, 0.5);
        println!(
            "batch {batch}: step median {:.2} ms, p90 {:.2} ms, {:.1} tok/s total, {:.1} tok/s per session",
            median.as_secs_f64() * 1e3,
            percentile(&times, 0.9).as_secs_f64() * 1e3,
            batch as f64 / median.as_secs_f64(),
            1.0 / median.as_secs_f64(),
        );
        for session in sessions {
            engine.close_session(session)?;
        }
    }
    Ok(())
}
