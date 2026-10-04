use cuda_core::Device;
use firepit::{Engine, LoadOptions, Model, Result, SessionId};
use firepit_kernels::sink::Trace;
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
    prefill_lengths: Option<Vec<usize>>,
    profile: bool,
}

fn parse_args() -> Result<Args> {
    let mut values = env::args().skip(1);
    let (
        mut model,
        mut batches,
        mut steps,
        mut warmup,
        mut max_seq,
        mut layers,
        mut prefill_lengths,
        mut profile,
    ) = (None, None, None, None, None, None, None, false);
    while let Some(flag) = values.next() {
        if flag == "--profile" {
            profile = true;
            continue;
        }
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
            "--prefill-lengths" => {
                prefill_lengths = Some(
                    value
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?,
                )
            }
            _ => return Err(format!("unknown argument {flag}").into()),
        }
    }
    Ok(Args {
        model: model.ok_or("missing --model")?,
        batches: if prefill_lengths.is_some() {
            batches.unwrap_or_default()
        } else {
            batches.ok_or("missing --batches")?
        },
        steps: if prefill_lengths.is_some() {
            steps.unwrap_or_default()
        } else {
            steps.ok_or("missing --steps")?
        },
        warmup: if prefill_lengths.is_some() {
            warmup.unwrap_or_default()
        } else {
            warmup.ok_or("missing --warmup")?
        },
        max_seq: max_seq.ok_or("missing --max-seq")?,
        layers,
        prefill_lengths,
        profile,
    })
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn print_trace(trace: &Trace) {
    let summary = trace.summary();
    let total_gpu: f64 = summary.iter().map(|row| row.gpu_ms).sum();
    println!(
        "trace spans={} gpu_total_ms={total_gpu:.3}",
        trace.spans.len()
    );
    println!("kind count gpu_ms wall_ms gpu_share_pct");
    for row in summary {
        let share = if total_gpu == 0.0 {
            0.0
        } else {
            100.0 * row.gpu_ms / total_gpu
        };
        println!(
            "{} {} {:.3} {:.3} {:.1}",
            row.kind, row.count, row.gpu_ms, row.wall_ms, share
        );
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if (args.prefill_lengths.is_some() || args.profile) && cfg!(debug_assertions) {
        return Err("prefill measurement and profiling require --release".into());
    }
    if args.prefill_lengths.is_none() && args.warmup + args.steps > args.max_seq {
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
    if let Some(lengths) = args.prefill_lengths {
        for length in lengths {
            if length == 0 || length > args.max_seq {
                return Err("invalid prefill length".into());
            }
            let tokens = vec![token; length];
            let mut warm = engine.new_session(SessionId(1))?;
            let _ = engine.prefill(&mut warm, &tokens)?;
            engine.close_session(warm)?;
            let mut session = engine.new_session(SessionId(2))?;
            let started = Instant::now();
            let (logits, trace) = if args.profile {
                let (logits, trace) = engine.prefill_traced(&mut session, &tokens)?;
                (logits, Some(trace))
            } else {
                (engine.prefill(&mut session, &tokens)?, None)
            };
            println!(
                "prefill T={length} wall_ms={:.3} logits={}",
                started.elapsed().as_secs_f64() * 1e3,
                logits.len()
            );
            if let Some(trace) = trace {
                print_trace(&trace);
            }
            engine.close_session(session)?;
        }
        return Ok(());
    }
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
        if args.profile {
            let mut profiled = (0..batch)
                .map(|id| engine.new_session(SessionId(id as u64 + 1000)))
                .collect::<Result<Vec<_>>>()?;
            let entries = profiled.iter_mut().map(|session| (session, token));
            let (_, trace) = engine.step_eager_traced(engine.batch(entries)?)?;
            print_trace(&trace);
            for session in profiled {
                engine.close_session(session)?;
            }
        }
        for session in sessions {
            engine.close_session(session)?;
        }
    }
    Ok(())
}
