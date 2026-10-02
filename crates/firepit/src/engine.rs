use crate::Result;
use crate::checkpoint::Stream;
use crate::execute::run;
use crate::model::Model;
use crate::plan::StepPlan;
use crate::session::{Batch, Session, SessionId};
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use firepit_kernels::rope;
use firepit_kernels::sink::{Eager, Graph};
use firepit_kernels::tile::{MIN_ROW_TILE, row_tile};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ENGINE: AtomicU64 = AtomicU64::new(1);
const TARGET_BLOCKS_PER_SM: usize = 1;

struct CachedPlan {
    instances: Vec<u64>,
    plan: StepPlan,
}

pub struct Engine {
    model: Arc<Model>,
    stream: Stream,
    id: u64,
    gdn_value_tile: usize,
    plans: HashMap<usize, CachedPlan>,
    #[cfg(test)]
    fail_after_gpu_work: bool,
}

impl Engine {
    pub fn new(model: Arc<Model>, device: &Arc<Device>) -> Result<Self> {
        if model.device_ordinal != device.ordinal() {
            return Err("model and engine must use the same device".into());
        }
        let sm =
            cuda_core::simt::CudaContext::new(device.ordinal())?.multiprocessor_count()? as usize;
        let g = model.dims.gdn();
        let gdn_value_tile = row_tile(
            g.value_head_dim(),
            g.value_heads(),
            sm,
            TARGET_BLOCKS_PER_SM,
            MIN_ROW_TILE,
        );
        Ok(Self {
            model,
            stream: device.new_stream()?,
            id: NEXT_ENGINE.fetch_add(1, Ordering::Relaxed),
            gdn_value_tile,
            plans: HashMap::new(),
            #[cfg(test)]
            fail_after_gpu_work: false,
        })
    }

    pub fn new_session(&self, id: SessionId) -> Result<Session> {
        Session::new(id, self.id, &self.model.dims, &self.stream)
    }

    pub fn close_session(&mut self, session: Session) -> Result<()> {
        if session.engine != self.id {
            return Err("session belongs to another engine".into());
        }
        self.plans
            .retain(|_, entry| !entry.instances.contains(&session.instance));
        Ok(())
    }

    pub fn batch<'a>(
        &self,
        entries: impl IntoIterator<Item = (&'a mut Session, u32)>,
    ) -> Result<Batch<'a>> {
        Batch::from_entries(self.id, entries, &self.model.dims)
    }

    pub fn step_eager(&mut self, batch: Batch<'_>) -> Result<Vec<f32>> {
        self.step_with(batch, |model, plan, sessions, stream| {
            run(model, plan, sessions, &mut Eager(stream))
        })
    }

    pub fn step(&mut self, batch: Batch<'_>) -> Result<Vec<f32>> {
        self.step_with(batch, |model, plan, sessions, stream| {
            if plan.graph.is_none() {
                let graph = CudaGraph::scope(stream, |scope| {
                    run(model, plan, sessions, &mut Graph(scope))
                        .map_err(|e| cuda_async::error::DeviceError::Internal(e.to_string()))
                })?;
                plan.graph = Some(graph);
            }
            plan.graph.as_ref().unwrap().launch().sync_on(stream)?;
            Ok(())
        })
    }

    fn step_with(
        &mut self,
        mut batch: Batch<'_>,
        execute: impl FnOnce(&Model, &mut StepPlan, &mut [&mut Session], &Stream) -> Result<()>,
    ) -> Result<Vec<f32>> {
        if batch.engine != self.id {
            return Err("batch belongs to another engine".into());
        }
        let actual = batch.sessions.len();
        let size = actual.next_power_of_two();
        let instances: Vec<u64> = batch.sessions.iter().map(|s| s.instance).collect();
        if self
            .plans
            .get(&size)
            .is_none_or(|entry| entry.instances != instances)
        {
            self.plans.insert(
                size,
                CachedPlan {
                    instances,
                    plan: StepPlan::new(&self.model, size, self.gdn_value_tile, &self.stream)?,
                },
            );
        }
        let plan = &mut self.plans.get_mut(&size).unwrap().plan;
        let result = (|| -> Result<Vec<f32>> {
            let mut tokens: Vec<i32> = batch.tokens.iter().map(|&x| x as i32).collect();
            tokens.resize(size, 0);
            let source = api::copy_host_vec_to_device(&Arc::new(tokens)).sync_on(&self.stream)?;
            api::memcpy(&mut plan.tokens, &source).sync_on(&self.stream)?;
            let positions: Vec<usize> = batch
                .sessions
                .iter()
                .map(|s| s.position)
                .chain(std::iter::repeat(0))
                .take(size)
                .collect();
            for (position, &value) in plan.positions.iter_mut().zip(&positions) {
                let source = api::copy_host_vec_to_device(&Arc::new(vec![value as i32]))
                    .sync_on(&self.stream)?;
                api::memcpy(position, &source).sync_on(&self.stream)?;
            }
            let (cos, sin) = rope::frequencies(&positions, &self.model.dims.q_rope());
            let cosine = api::copy_host_vec_to_device(&Arc::new(cos))
                .reshape(&[size, self.model.dims.attention().head_dim()])
                .sync_on(&self.stream)?;
            let sine = api::copy_host_vec_to_device(&Arc::new(sin))
                .reshape(&[size, self.model.dims.attention().head_dim()])
                .sync_on(&self.stream)?;
            api::memcpy(&mut plan.cos, &cosine).sync_on(&self.stream)?;
            api::memcpy(&mut plan.sin, &sine).sync_on(&self.stream)?;
            execute(&self.model, plan, &mut batch.sessions, &self.stream)?;
            #[cfg(test)]
            if self.fail_after_gpu_work {
                return Err("injected failure after GPU work".into());
            }
            plan.logits(&self.stream)
        })();
        let canonical = match result {
            Ok(logits) => logits,
            Err(error) => {
                for session in &mut batch.sessions {
                    session.poisoned = true;
                }
                return Err(error);
            }
        };
        for session in &mut batch.sessions {
            session.position += 1;
        }
        let vocab = self.model.dims.vocab();
        let mut logits = vec![0.0; actual * vocab];
        for (canonical_row, original_row) in batch.original_rows.into_iter().enumerate() {
            logits[original_row * vocab..(original_row + 1) * vocab]
                .copy_from_slice(&canonical[canonical_row * vocab..(canonical_row + 1) * vocab]);
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SafetensorsFile;
    use std::path::Path;

    fn small_model(device: &Arc<Device>) -> Result<Arc<Model>> {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/small-qwen.json"))?;
        Ok(Arc::new(Model::random_from_config(
            &config["text_config"],
            8,
            7,
            &device.new_stream()?,
        )?))
    }

    #[test]
    fn gpu_failure_poisons_session() -> Result<()> {
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        let mut engine = Engine::new(model, &device)?;
        let mut session = engine.new_session(SessionId(1))?;
        engine.fail_after_gpu_work = true;
        let batch = engine.batch([(&mut session, 1)])?;
        assert!(engine.step_eager(batch).is_err());
        assert_eq!(session.position(), 0);
        assert!(engine.batch([(&mut session, 1)]).is_err());
        Ok(())
    }

    #[test]
    fn small_model_batch_matches_individual_and_graph() -> Result<()> {
        let device = Device::new(0)?;
        let model = small_model(&device)?;
        for size in 1..=4 {
            let mut together = Engine::new(model.clone(), &device)?;
            let mut graph = Engine::new(model.clone(), &device)?;
            let mut singles = (0..size)
                .map(|_| Engine::new(model.clone(), &device))
                .collect::<Result<Vec<_>>>()?;
            let mut together_sessions = (0..size)
                .map(|i| together.new_session(SessionId(i as u64 + 1)))
                .collect::<Result<Vec<_>>>()?;
            let mut graph_sessions = (0..size)
                .map(|i| graph.new_session(SessionId(i as u64 + 1)))
                .collect::<Result<Vec<_>>>()?;
            let mut single_sessions = singles
                .iter()
                .enumerate()
                .map(|(i, e)| e.new_session(SessionId(i as u64 + 10)))
                .collect::<Result<Vec<_>>>()?;
            for step in 0..3 {
                let tokens: Vec<u32> = (0..size).map(|i| (i * 13 + step * 5 + 3) as u32).collect();
                let expected = together.step_eager(
                    together.batch(together_sessions.iter_mut().zip(tokens.iter().copied()))?,
                )?;
                let actual = graph
                    .step(graph.batch(graph_sessions.iter_mut().zip(tokens.iter().copied()))?)?;
                assert_eq!(expected, actual);
                for i in 0..size {
                    let single_batch = singles[i].batch([(&mut single_sessions[i], tokens[i])])?;
                    let single = singles[i].step_eager(single_batch)?;
                    let max_error = expected[i * 64..(i + 1) * 64]
                        .iter()
                        .zip(&single)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_error <= 1e-5,
                        "B={size} step={step} row={i} max_abs={max_error}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "needs local Qwen3.6 checkpoint and GPU"]
    fn prefix_logits_match_hf_eager() -> Result<()> {
        let dir = Path::new(env!("HOME")).join("Models/qwen3.6-35b-a3b");
        let device = Device::new(0)?;
        let refs = SafetensorsFile::open(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../refs/qwen36-prefix4.safetensors"),
        )?;
        let tokens: Vec<u32> =
            serde_json::from_str(refs.metadata("tokens").ok_or("missing tokens")?)?;
        let mut model = Arc::new(Model::from_checkpoint(
            dir,
            crate::LoadOptions {
                max_seq: tokens.len(),
                layers: Some(4),
            },
            &device,
        )?);
        let mut graph_engine = Engine::new(model.clone(), &device)?;
        let mut graph_session = graph_engine.new_session(SessionId(2))?;
        let mut engine = Engine::new(model.clone(), &device)?;
        let mut session = engine.new_session(SessionId(1))?;
        for (step, &token) in tokens.iter().enumerate() {
            let batch = engine.batch([(&mut session, token)])?;
            let actual = engine.step_eager(batch)?;
            let graph_batch = graph_engine.batch([(&mut graph_session, token)])?;
            let graphed = graph_engine.step(graph_batch)?;
            let expected = refs.tensor(&format!("step{step}/prefix_logits"))?;
            let relative = actual
                .iter()
                .zip(&expected.data)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                / expected
                    .data
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
            let top = |values: &[f32]| {
                values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            println!(
                "step{step} relative={relative:.8e} top={} hf_top={}",
                top(&actual),
                top(&expected.data)
            );
            assert_eq!(actual, graphed);
            assert_eq!(top(&actual), top(&expected.data));
            assert!(relative < 1e-3);
        }
        drop(session);
        drop(graph_session);
        drop(engine);
        drop(graph_engine);
        Arc::get_mut(&mut model)
            .ok_or("model still shared")?
            .layers
            .swap(0, 1);
        let mut shuffled = Engine::new(model, &device)?;
        let mut shuffled_session = shuffled.new_session(SessionId(3))?;
        let shuffled_batch = shuffled.batch([(&mut shuffled_session, tokens[0])])?;
        let shuffled_logits = shuffled.step_eager(shuffled_batch)?;
        let expected = refs.tensor("step0/prefix_logits")?;
        let error = shuffled_logits
            .iter()
            .zip(&expected.data)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            .sqrt()
            / expected
                .data
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
        println!("shuffled_layer_relative={error:.8e}");
        assert!(error > 1e-3);
        Ok(())
    }
}
